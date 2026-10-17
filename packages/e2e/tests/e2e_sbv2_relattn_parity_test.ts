// 相対位置注意の `(T,T)` 表について、**ホスト TS 生成器と Python 側生成器のバイト一致**を
// 実データで固定する（ADR 0013 の「新しい検証責務」そのもの）。
//
// flow / voice は表を焼き込まず**グラフ入力**で受ける。つまり実行時に表を作るのはホストで、
// ゴールデンの表を作るのは Python（`tools/export-recipes/sbv2/patch.py` の
// `build_relattn_tables`）— 式が割れると
// **shape は合ったまま値だけが誤る**。しかも誤り方によっては「ホストとゴールデンが同じ
// 誤りを共有して E2E がすり抜ける」ので、E2E とは別にここで両者を直接突き合わせる。
//
// このテストが緑である限り、E2E がゴールデン由来の表で走ることと、ホスト生成の表で走る
// ことは**同じ計算**になる（バイト一致なので入れ替えても値が動かない）。
//
// GPU は使わない。資産（`outputs/series/sbv2-F1/{flow,voice}/`）が無い環境では該当ケースを
// SKIP する。
//
// The asset-free generator tests (fault injection, window semantics, input validation) stay in
// packages/models/tests/sbv2_relattn_parity_test.ts; only the cases that read the series assets
// live here (ADR 0129).

import { type DirEntry, isNotFound, readDirSync, readFile } from "@karume/test-support/fs";
import { test } from "@karume/test-support/test";
import { assert, assertEquals } from "@std/assert";
import { parseSafetensors, prepareContainer } from "@karume/runtime";
import { openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../models/tests/helpers/series-graphs.ts";
import { buildRelattnTables, RELATTN_WINDOW_SIZE } from "../../models/src/sbv2/relattn-tables.ts";

/**
 * 系列 root。綴りの `sbv2-F1` は `sbv2/export.py` の `default_out_root()` が `--model-dir` の
 * ディレクトリ名から導いたもので、**当面この 1 話者を決め打ち**する。
 */
const SERIES_NAME = "sbv2-F1";
const MODELS_ROOT = new URL(`../../../outputs/series/${SERIES_NAME}/`, import.meta.url);
const MODEL_FILE = "model.krm";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/** 表を**グラフ入力**で受けるターゲット（front は焼き込みなので対象外 — ADR 0013）。 */
const TABLE_INPUT_TARGETS = ["flow", "voice"] as const;

/** io に格納された表のテンソルキー（= グラフ入力名）。 */
const IDX_K_KEY = "input.idx_k";
const VALID_KEY = "input.valid";

const listDir = (url: URL): readonly DirEntry[] => {
  try {
    return [...readDirSync(url)];
  } catch (cause) {
    if (isNotFound(cause)) {
      return [];
    }
    throw cause;
  }
};

const discoverCases = (target: string): readonly string[] =>
  listDir(new URL(`${target}/`, MODELS_ROOT))
    .filter(
      (entry) => entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX),
    )
    .map((entry) => entry.name)
    .sort();

const AVAILABLE = TABLE_INPUT_TARGETS.flatMap((target) =>
  discoverCases(target).map((ioFile) => ({ target, ioFile })),
);

if (AVAILABLE.length === 0) {
  console.warn(
    `[karume] ${MODELS_ROOT.pathname} に flow / voice が無いため相対位置表のパリティ検証を ` +
      "SKIP する。生成: cd tools/exporter && uv run --group sbv2 python export_sbv2.py",
  );
}

/**
 * 要素ごとの厳密一致（`Object.is`）。`0` と `-0` を区別するので f32 でもビット同一を主張
 * できる（`===` は `-0 === 0` が真になり、符号付きゼロの取り違えを見逃す）。
 */
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

for (const { target, ioFile } of AVAILABLE) {
  test(`相対位置表パリティ: ${target} / ${ioFile}（TS 生成 ↔ golden の Python 生成）`, async () => {
    const bytes = await readFile(new URL(`${target}/${ioFile}`, MODELS_ROOT));
    const io = parseSafetensors(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    );
    const idxKView = io.tensors.get(IDX_K_KEY);
    const validView = io.tensors.get(VALID_KEY);
    assert(idxKView !== undefined && validView !== undefined, `${ioFile} に表の入力が無い`);
    assertEquals(idxKView.dtype, "I32", `${IDX_K_KEY} の格納 dtype`);
    assertEquals(validView.dtype, "F32", `${VALID_KEY} の格納 dtype`);

    // 実長はゴールデン側の shape から採る（TS 側で長さを決め打ちすると、ケースが
    // 増えたときに「同じ長さどうしを比べて常に緑」になる）。
    assertEquals(idxKView.shape.length, 2, `${IDX_K_KEY} の rank`);
    assertEquals(idxKView.shape, validView.shape, "2 本の表の shape");
    const [rows, columns] = idxKView.shape;
    assertEquals(rows, columns, "相対位置表は正方（self-attention）");

    const built = buildRelattnTables(rows);
    assertEquals(built.idxK.shape, [...idxKView.shape], `${target}/${ioFile}: idx_k の shape`);
    assertSameElements(
      built.idxK.data,
      new Int32Array(io.buffer, idxKView.byteOffset, idxKView.byteLength / 4),
      `${target}/${ioFile}: idx_k`,
    );
    assertSameElements(
      built.valid.data,
      new Float32Array(io.buffer, validView.byteOffset, validView.byteLength / 4),
      `${target}/${ioFile}: valid`,
    );
  });
}

for (const target of TABLE_INPUT_TARGETS) {
  test({
    name: `相対位置表の窓幅: ${target} の焼き込み表と TS 定数が一致する`,
    ignore: discoverCases(target).length === 0,
    fn: async () => {
      // 沈黙誤値クラスの門（ADR 0013）。key 側の表はホストが作るが、**value 側の表
      // `idx_v` はコンテナに焼き込まれていて幅が `2w+1`**。ここが TS の窓幅定数と食い違えば
      // ホストは違う幅の埋め込みを前提に添字を作っている — shape エラーにならず黙って
      // 誤るクラスなので、コンテナから読んだ幅で TS 側の定数を検算する。
      // グラフ記述は容器の part 0 にある（container-v1 §8）— 重みの block は 1 つも取らない。
      // グラフ名は helpers/series-graphs.ts の 1 本から引く（門番と同じ正本）。この系列では
      // 部品ディレクトリ名と同じ綴りになるが、それは事実であって規則ではない。
      const opened = await openSeriesContainer(new URL(`${target}/${MODEL_FILE}`, MODELS_ROOT));
      const model = prepareContainer(opened, seriesGraph(SERIES_NAME, target));
      const baked = model.graph.nodes
        .filter((node) => node.op === "sym_prefix_slice")
        .map((node) => model.graph.values[node.ins[0]].shape);

      // 表が入力へ昇格した後に焼き込みが残るのは value 側の 1 本だけ（key 側の 2 本が
      // 残っていたら昇格が効いていない＝ 134MB の定数を抱えている）。
      assertEquals(baked.length, 1, `${target} の焼き込み表の本数`);
      assertEquals(baked[0].length, 4, `${target} の焼き込み表の rank`);
      assertEquals(baked[0][3], 2 * RELATTN_WINDOW_SIZE + 1, `${target} の相対位置埋め込みの幅`);
    },
  });
}
