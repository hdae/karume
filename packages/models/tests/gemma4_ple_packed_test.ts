/**
 * 固定 packed PLE（i2 / i4）と独立 Torch CPU fixture の突合（ADR 0097）。
 *
 * 資産は recipe の系列出力と同じ旧形 sidecar（`ple.json` + `ple-NNNNN.safetensors`）で、容器の
 * 資産と同じ面へは `helpers/ple-series.ts` が畳む（recipe が `krm` を書くのは段 3 — ADR 0109
 * 決定 8）。値は移行前後でビット同一なので、この突合の力は変わらない。
 *
 * 門は 3 本:
 * ① **行読み**（予算 0）の値が torch の CPU 参照と**ビット一致**する
 * ② **全量読み**（予算 = 全量）の値が①とビット一致し、常駐 block と residentBytes が索引どおり
 * ③ 索引の格納宣言が実体とずれていれば fail loudly（片方だけ焼き直した組み合わせ）
 */

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { parseSafetensors } from "@karume/runtime";
import { createGemma4Ple } from "../src/gemma/ple.ts";
import { gemma4PleTotalBytes, parseGemma4PleIndex } from "../src/gemma/ple-index.ts";
import { openSeriesPle } from "./helpers/ple-series.ts";

for (const dtype of ["i2", "i4"] as const) {
  Deno.test(`PLE ${dtype}: 行読みと全量読みの結果が独立 CPU 参照とビット一致する`, async () => {
    const dir = new URL(`./fixtures/gemma4-ple-packed/${dtype}/`, import.meta.url);
    const { index, openBlock } = await openSeriesPle(dir);
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

    // 受理集合（schema 3 の門）— 旧版・格納の綴り違い・詰め数で割り切れない次元は読まない。
    const raw = JSON.parse(await Deno.readTextFile(new URL("ple.json", dir)));
    for (const invalid of [{ schema: 1 }, { schema: 2 }, { storage: "i8" }, { dim: 31 }]) {
      assertThrows(() => parseGemma4PleIndex({ ...raw, ...invalid }));
    }
  });
}
