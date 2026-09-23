/**
 * 固定 packed PLE（i2 / i4）と独立 Torch CPU fixture の突合（ADR 0097）。
 *
 * 資産はリポジトリ内のフィクスチャで、実体は**系列出力とまったく同じ形**の `krm` 容器
 * （`model-NNNNN-of-NNNNN.krm` の part 列 + 資産 `ple_index` / `ple.values.*` /
 * `ple.scales.*`）。生成器は `tools/export-recipes/gemma4/tests/ple_fixture.py` で、
 * recipe の本番経路と同じ関数（`karume.ple.ple_assets` + `karume.pipeline.publish_model`）が
 * 焼く。読み口も系列出力と同じ 1 本（`helpers/ple-series.ts` の `seriesPleHandle`）なので、
 * この門は「fixture 専用の読み手」ではなく**配布される形**を読んでいる。
 *
 * 門は 5 本:
 * ① **行読み**（予算 0）の値が torch の CPU 参照と**ビット一致**する
 * ② **全量読み**（予算 = 全量）の値が①とビット一致し、常駐 block と residentBytes が索引どおり
 * ③ 索引の格納宣言が実体とずれていれば値が割れる（片方だけ焼き直した組み合わせ）
 * ④ 索引の受理集合（schema 3）— 旧版・格納の綴り違い・詰め数で割り切れない次元は読まない
 * ⑤ 索引と容器の資産の**両方向**の突合（`readGemma4PleIndex`）が、片側を欠いた容器を落とす
 *
 * NOTE: `values` と `scales` は**別々の block 境界**を持つ（生成器が `BLOCK_BYTES = 144` で
 * 切るので i4 は 3 : 1・i2 は 2 : 1）。表ごとに別の block を引く読み手の誤りを、境界の揃った
 * fixture では拾えないためである。②の本数の断定は索引から導く（焼き直しで動いてよい）が、
 * 「境界が揃っていないこと」自体は下で名指しの断定を置く。
 */

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { type OpenedContainer, parseSafetensors } from "@karume/runtime";
import { openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { createGemma4Ple } from "../src/gemma/ple.ts";
import {
  gemma4PleAssetSource,
  gemma4PleTotalBytes,
  parseGemma4PleIndex,
  PLE_INDEX_ASSET,
  readGemma4PleIndex,
} from "../src/gemma/ple-index.ts";
import { SERIES_MODEL_FILE, seriesPleHandle } from "./helpers/ple-series.ts";

/** 索引の資産を生のまま読む（受理集合の門は doctored な文書を `parseGemma4PleIndex` へ渡す）。 */
const readIndexDocument = async (
  opened: Pick<OpenedContainer, "asset">,
): Promise<Record<string, unknown>> => {
  const reader = opened.asset(PLE_INDEX_ASSET);
  return JSON.parse(new TextDecoder().decode(await reader.read(0, reader.length)));
};

for (const dtype of ["i2", "i4"] as const) {
  Deno.test(`PLE ${dtype}: 行読みと全量読みの結果が独立 CPU 参照とビット一致する`, async () => {
    const dir = new URL(`./fixtures/gemma4-ple-packed/${dtype}/`, import.meta.url);
    const opened = await openSeriesContainer(new URL(SERIES_MODEL_FILE, dir));
    const { index, openBlock } = await seriesPleHandle(opened, `test: ${dtype} fixture`);
    const oracle = parseSafetensors(
      (await Deno.readFile(new URL("oracle.safetensors", dir))).buffer,
    );
    const iv = oracle.tensors.get("ids")!, ev = oracle.tensors.get("expected")!;
    const ids = Array.from(new Int32Array(oracle.buffer, iv.byteOffset, iv.byteLength / 4));
    const expected = new Uint32Array(oracle.buffer, ev.byteOffset, ev.byteLength / 4);
    const distinct = new Set(ids).size;
    const factor = dtype === "i2" ? 4 : 2;
    assertEquals(index.storage, dtype);
    assertEquals(index.values.rowBytes, index.layers * index.dim / factor);
    assertEquals(
      gemma4PleTotalBytes(index),
      index.tokens * (index.layers * (index.dim / factor + 4)),
    );
    // 表ごとに境界が違うこと（モジュール doc の NOTE）。揃った fixture では、scales を
    // values の block 番号で引く読み手の誤りが素通りする。
    assert(
      index.values.blocks.length > 1 && index.values.blocks.length !== index.scales.blocks.length,
      `values ${index.values.blocks.length} block / scales ${index.scales.blocks.length} block` +
        " — 表ごとに違う切り目を踏ませる fixture でなくなっている",
    );

    // ① 行読み（予算 0 = 常駐も行キャッシュも持たない）。
    const rows = createGemma4Ple({
      index,
      openBlock,
      vocabSize: index.tokens,
      maxResidentBytes: 0,
    });
    const byRows = await rows.gather(ids);
    assert(byRows.dtype === "f32");
    assertEquals(new Uint32Array(byRows.data.buffer), expected);
    assertEquals(rows.stats().loads, 0, "予算 0 なのに block 全量を読んでいる");
    assertEquals(rows.stats().rowReads, distinct * 2, "行読みは values / scales の 2 表ぶん");
    rows.dispose();

    // ② 全量読み（予算 = 全量 — 参照 id は全 token を覆うので、全 block が下限に届く）。
    const blocks = createGemma4Ple({
      index,
      openBlock,
      vocabSize: index.tokens,
      maxResidentBytes: gemma4PleTotalBytes(index),
    });
    const byBlocks = await blocks.gather(ids);
    assert(byBlocks.dtype === "f32");
    assertEquals(
      [...new Uint32Array(byBlocks.data.buffer)],
      [...new Uint32Array(byRows.data.buffer)],
      "全量読みの値が行読みとビット一致しない",
    );
    const loaded = index.values.blocks.length + index.scales.blocks.length;
    assertEquals(blocks.stats().loads, loaded);
    assertEquals(blocks.stats().rowReads, 0);
    assertEquals(blocks.stats().resident, loaded);
    assertEquals(blocks.stats().residentBytes, gemma4PleTotalBytes(index));
    // 常駐しているので引き直しても読みは増えない。
    await blocks.gather(ids);
    assertEquals(blocks.stats().loads, loaded, "常駐している block を読み直している");
    blocks.dispose();
    assertEquals(blocks.stats().residentBytes, 0);
    await assertRejects(() => blocks.gather(ids));

    // ③ 索引の格納宣言だけを差し替えると、形も dtype も合ったまま別の値になる — 参照と割れる。
    const mismatched = createGemma4Ple({
      index: { ...index, storage: dtype === "i2" ? "i4" : "i2" },
      openBlock,
      vocabSize: index.tokens,
      maxResidentBytes: 0,
    });
    try {
      const wrong = await mismatched.gather(ids);
      assert(wrong.dtype === "f32");
      assert(
        [...new Uint32Array(wrong.data.buffer)].some((word, at) => word !== expected[at]),
        "格納宣言を取り違えても値が変わらない（索引の storage が効いていない）",
      );
    } finally {
      mismatched.dispose();
    }

    // ④ 受理集合（schema 3 の門）— 旧版・格納の綴り違い・詰め数で割り切れない次元は読まない。
    const raw = await readIndexDocument(opened);
    assertEquals(raw.schema, 3, "容器の索引が schema 3 でない");
    for (const invalid of [{ schema: 1 }, { schema: 2 }, { storage: "i8" }, { dim: 31 }]) {
      assertThrows(() => parseGemma4PleIndex({ ...raw, ...invalid }));
    }
  });

  /**
   * ⑤ 索引と容器の資産の突合（`readGemma4PleIndex` → `assertGemma4PleAssets`）。
   *
   * 旧 sidecar の読み口ではこの門が掛からなかった（索引をテスト側で組み立てていたので、
   * 「索引が指す資産が容器に無い」は定義上起きなかった）。容器を読む形にした今は、片方だけ
   * 焼き直した配布形を**重みの block を 1 バイトも取る前に**落とせる。ここは資産の宣言を
   * 間引いた面を渡して、恒真でないことを確かめる。
   */
  Deno.test(`PLE ${dtype}: 索引と容器の資産が食い違えば読む前に落ちる`, async () => {
    const dir = new URL(`./fixtures/gemma4-ple-packed/${dtype}/`, import.meta.url);
    const opened = await openSeriesContainer(new URL(SERIES_MODEL_FILE, dir));
    const source = gemma4PleAssetSource(opened);
    const dropped = Object.keys(source.assets).find((name) => name.startsWith("ple.values."));
    assert(dropped !== undefined, "容器に values の資産が無い");

    // (a) 索引が指す block を容器が宣言していない。
    const missing = { ...source.assets };
    delete missing[dropped];
    await assertRejects(
      () => readGemma4PleIndex("test: 資産欠け", { ...source, assets: missing }),
      Error,
      `${dropped}: 容器が宣言していない`,
    );

    // (b) 索引が指していない PLE 資産が容器に在る（配布形が宣言した資産を 1 本も読まない形）。
    await assertRejects(
      () =>
        readGemma4PleIndex("test: 余分な資産", {
          ...source,
          assets: { ...source.assets, "ple.values.999": "ple-values" },
        }),
      Error,
      "ple.values.999: 役割 'ple-values' の資産だが索引が指していない",
    );

    // (c) 役割が違う（値と scale を取り違えた焼き直し）。
    await assertRejects(
      () =>
        readGemma4PleIndex("test: 役割違い", {
          ...source,
          assets: { ...source.assets, [dropped]: "ple-scales" },
        }),
      Error,
      `${dropped}: 役割 'ple-scales'`,
    );
  });
}
