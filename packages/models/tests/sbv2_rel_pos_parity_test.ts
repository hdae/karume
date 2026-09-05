// DeBERTa の相対位置の添字表について、**ホスト TS 生成器と Python 側生成器のバイト一致**を
// 実データで固定する（ADR 0045 波 3 — flow / voice の `sbv2_relattn_parity_test.ts` と同じ責務）。
//
// text_encoder は表を焼き込まず**グラフ入力**で受ける。つまり実行時に表を作るのはホストで、
// ゴールデンの表を作るのは Python（`patch_deberta.build_rel_pos_tables`）— 式が割れると
// **shape は合ったまま値だけが誤る**（別の位置埋め込みを gather する）。しかも誤り方によっては
// 「ホストとゴールデンが同じ誤りを共有して E2E がすり抜ける」ので、E2E とは別にここで両者を
// 直接突き合わせる。
//
// 元の式は torch が **float32** で計算するが、ホストは float64。境界の ceil が一致することは
// 実測命題なので（ADR 0045 波 3）、ここが実データでそれを縛る唯一の場所になる。
//
// GPU は使わない。系列資産（`outputs/series/deberta-i8/sbv2-22layer/`）と配布形の
// `symbols.json` が無い環境では SKIP する。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { parseSafetensors, prepareModel } from "@karume/runtime";
import { readShard, resolveShards } from "../../runtime/tests/helpers/shard-files.ts";
import { buildRelPosTables } from "../src/sbv2/text/rel-pos-tables.ts";
import { parseJpExtraRules } from "../src/sbv2/text/symbols.ts";

/** 表を入力で受ける text_encoder の系列（配布形と同じ variant）。 */
const SERIES_ROOT = new URL("../../../outputs/series/deberta-i8/sbv2-22layer/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/** バケット規則の出所（配布形の資産 — TS 側に写経しないための 1 箇所）。 */
const SYMBOLS_FILE = new URL(
  "../../../models/karume-sbv2-jvnv/shared/text/symbols.json",
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
    // グラフを持つのは配布形の**先頭 shard** だけ（ADR 0081）。
    const model = prepareModel(await readShard(resolveShards(new URL(MODEL_FILE, SERIES_ROOT))[0]));
    const baked = model.graph.nodes.filter((node) => node.op === "sym_prefix_slice");
    assertEquals(baked.length, 0, "焼き込み表（sym_prefix_slice）の本数");
    assertEquals(
      model.graph.inputs.map((input) => input.name),
      ["input_ids", "attention_mask", "c2p_pos", "p2c_pos"],
      "グラフ入力の並び",
    );
  },
});

// ---- 資産の要らない門（上のパリティは golden が無いと 0 本走る）----------------
//
// 対の `sbv2_relattn_parity_test.ts` には資産不要のテストが 2 本あるのに、こちら側は
// golden が無いと `PARITY_CASES` が空になり **1 本も走らない**（引数門も数学的性質も
// 未検証のまま緑）。Python 側とのバイト一致は上のパリティにしか置けないが、式が満たす
// べき構造（引数の受理集合・転置・Toeplitz）は資産なしで縛れる。

Deno.test("buildRelPosTables: 引数の受理集合（式が成立する下限を割る値は落とす）", () => {
  // 長さ 0 は表が空になるだけでなく、対角配列 `2·length-1` が負長になる。
  assertThrows(() => buildRelPosTables(0, 256, 512), RangeError, "1 以上の整数でない");
  // bucketSize 1 は mid = 0 で対数の分母（log((maxPosition-1)/mid)）が壊れる。
  assertThrows(() => buildRelPosTables(4, 1, 512), RangeError, "bucketSize 1 が 2 以上");
  // maxPosition 1 は log(0/…) で -Infinity を作る。
  assertThrows(() => buildRelPosTables(4, 256, 1), RangeError, "maxPosition 1 が 2 以上");
});

Deno.test("buildRelPosTables: p2c は c2p の転置（バケット化が奇関数であることの帰結）", () => {
  const length = 7;
  const { c2pPos, p2cPos } = buildRelPosTables(length, 256, 512);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      assertEquals(
        p2cPos.data[i * length + j],
        c2pPos.data[j * length + i],
        `p2c[${i}][${j}] が c2p[${j}][${i}] と違う`,
      );
    }
  }
});

Deno.test("buildRelPosTables: 表は i−j にしか依存しない（Toeplitz）", () => {
  const length = 7;
  const { c2pPos } = buildRelPosTables(length, 256, 512);
  for (let i = 0; i + 1 < length; i += 1) {
    for (let j = 0; j + 1 < length; j += 1) {
      assertEquals(
        c2pPos.data[i * length + j],
        c2pPos.data[(i + 1) * length + (j + 1)],
        `対角 ${i - j} の値が位置で変わる`,
      );
    }
  }
  // 上の 2 本が恒真でないことの対: `maxPosition` は遠方（対数域）の値を実際に動かす。
  // 近傍だけの表（bucketSize 256 に対し length 7）では対数域へ入らないので、
  // 対数域へ届く形（mid = 4 に対し |i−j| が 4 を超える）で見る。
  const wide = buildRelPosTables(16, 8, 512);
  const narrow = buildRelPosTables(16, 8, 64);
  assert(
    [...wide.c2pPos.data].some((value, index) => value !== narrow.c2pPos.data[index]),
    "maxPosition を変えても表が同じ（バケット規則を読んでいない）",
  );
  // 近傍（線形域）は maxPosition に依らない — 差が出るのは遠方だけであることの固定。
  assertEquals(wide.c2pPos.data[0 * 16 + 1], narrow.c2pPos.data[0 * 16 + 1]);
});
