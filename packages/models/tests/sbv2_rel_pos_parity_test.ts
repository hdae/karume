// DeBERTa の相対位置の添字表について、**ホスト TS 生成器と Python 側生成器のバイト一致**を
// 実データで固定する（ADR 0044 波 3 — flow / voice の `sbv2_relattn_parity_test.ts` と同じ責務）。
//
// text_encoder は表を焼き込まず**グラフ入力**で受ける。つまり実行時に表を作るのはホストで、
// ゴールデンの表を作るのは Python（`patch_deberta.build_rel_pos_tables`）— 式が割れると
// **shape は合ったまま値だけが誤る**（別の位置埋め込みを gather する）。しかも誤り方によっては
// 「ホストとゴールデンが同じ誤りを共有して E2E がすり抜ける」ので、E2E とは別にここで両者を
// 直接突き合わせる。
//
// 元の式は torch が **float32** で計算するが、ホストは float64。境界の ceil が一致することは
// 実測命題なので（ADR 0044 波 3）、ここが実データでそれを縛る唯一の場所になる。
//
// GPU は使わない。系列資産（`outputs/series/deberta-i8/sbv2-22layer/`）と配布形の
// `symbols.json` が無い環境では SKIP する。

import { assert, assertEquals } from "@std/assert";
import { openModel, parseSafetensors } from "@karume/runtime";
import { buildRelPosTables } from "../src/sbv2/text/rel-pos-tables.ts";
import { parseJpExtraRules } from "../src/sbv2/text/symbols.ts";

/** 表を入力で受ける text_encoder の系列（配布形と同じ variant）。 */
const SERIES_ROOT = new URL("../../../outputs/series/deberta-i8/sbv2-22layer/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/** バケット規則の出所（配布形の資産 — TS 側に写経しないための 1 箇所）。 */
const SYMBOLS_FILE = new URL(
  "../../../models/karume-sbv2-fn/shared/text/symbols.json",
  import.meta.url,
);

/** io に格納された表のテンソルキー（= グラフ入力名）。 */
const C2P_KEY = "input.c2p_pos";
const P2C_KEY = "input.p2c_pos";

const listDir = (url: URL): readonly Deno.DirEntry[] => {
  try {
    return [...Deno.readDirSync(url)];
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) {
      return [];
    }
    throw cause;
  }
};

const readSymbols = (): { positionBuckets: number; maxPosition: number } | undefined => {
  try {
    const raw = JSON.parse(Deno.readTextFileSync(SYMBOLS_FILE));
    return parseJpExtraRules(raw, SYMBOLS_FILE.pathname).bertRelPos;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) {
      return undefined;
    }
    throw cause;
  }
};

const CASES = listDir(SERIES_ROOT)
  .filter((entry) =>
    entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX)
  )
  .map((entry) => entry.name)
  .sort();
const RULE = readSymbols();

if (CASES.length === 0 || RULE === undefined) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} の golden io か配布形の symbols.json が無いため ` +
      "DeBERTa 相対位置表のパリティ検証を SKIP する。生成: cd tools/exporter && " +
      "uv run --with 'transformers==5.14.1' python export_deberta.py --dtype i8 --layers 22",
  );
}

/** 要素ごとの厳密一致（添字表は i32 なので `Object.is` で十分に強い）。 */
const assertSameElements = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  where: string,
): void => {
  assertEquals(actual.length, expected.length, `${where}: 要素数`);
  for (let index = 0; index < expected.length; index += 1) {
    if (!Object.is(actual[index], expected[index])) {
      throw new Error(
        `${where}: 要素 ${index} が食い違う（TS=${actual[index]} / Python=${expected[index]}）`,
      );
    }
  }
};

/** 資産が揃ったケースだけ（規則を一緒に持たせて、テスト本体では undefined を扱わない）。 */
const PARITY_CASES = RULE === undefined ? [] : CASES.map((ioFile) => ({ ioFile, rule: RULE }));

for (const { ioFile, rule } of PARITY_CASES) {
  Deno.test(`DeBERTa 相対位置表パリティ: ${ioFile}（TS 生成 ↔ golden の Python 生成）`, async () => {
    const bytes = await Deno.readFile(new URL(ioFile, SERIES_ROOT));
    const io = parseSafetensors(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    );
    const c2pView = io.tensors.get(C2P_KEY);
    const p2cView = io.tensors.get(P2C_KEY);
    assert(c2pView !== undefined && p2cView !== undefined, `${ioFile} に添字表の入力が無い`);
    assertEquals(c2pView.dtype, "I32", `${C2P_KEY} の格納 dtype`);
    assertEquals(p2cView.dtype, "I32", `${P2C_KEY} の格納 dtype`);

    // 実長はゴールデン側の shape から採る（TS 側で決め打つと、ケースが増えたときに
    // 「同じ長さどうしを比べて常に緑」になる）。
    assertEquals(c2pView.shape.length, 2, `${C2P_KEY} の rank`);
    assertEquals(c2pView.shape, p2cView.shape, "2 本の表の shape");
    const [rows, columns] = c2pView.shape;
    assertEquals(rows, columns, "相対位置表は正方（self-attention）");

    const built = buildRelPosTables(rows, rule.positionBuckets, rule.maxPosition);
    assertEquals(built.c2pPos.shape, [...c2pView.shape], `${ioFile}: c2p_pos の shape`);
    assertSameElements(
      built.c2pPos.data,
      new Int32Array(io.buffer, c2pView.byteOffset, c2pView.byteLength / 4),
      `${ioFile}: c2p_pos`,
    );
    assertSameElements(
      built.p2cPos.data,
      new Int32Array(io.buffer, p2cView.byteOffset, p2cView.byteLength / 4),
      `${ioFile}: p2c_pos`,
    );
  });
}

Deno.test({
  name: "DeBERTa の添字表はコンテナに 1 本も焼き込まれていない",
  ignore: CASES.length === 0,
  fn: async () => {
    // 昇格が効いていれば、T で切り出す `sym_prefix_slice` は 1 本も残らない（残っていたら
    // Tmax=512 の `[1,512,512]` を抱えたまま = 2MiB の死荷重）。値は正しいまま容量だけが
    // 戻る類なので、E2E では捕まらない。
    const bytes = await Deno.readFile(new URL(MODEL_FILE, SERIES_ROOT));
    const model = openModel(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    );
    const baked = model.graph.nodes.filter((node) => node.op === "sym_prefix_slice");
    assertEquals(baked.length, 0, "焼き込み表（sym_prefix_slice）の本数");
    assertEquals(
      model.graph.inputs.map((input) => input.name),
      ["input_ids", "attention_mask", "c2p_pos", "p2c_pos"],
      "グラフ入力の並び",
    );
  },
});
