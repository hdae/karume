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
// No GPU. Where the assets come from (ADR 0139): the golden tables are read from the test-asset
// root, `goldenDir("sbv2-F1")` = `outputs/test-assets/goldens/sbv2-F1/{flow,voice}/io.*` (fetched
// with `node tools/test-assets/fetch.ts`); the voice window check reads the graph of the
// distribution's F1 voice container (`models/karume-sbv2-jvnv/`, through its manifest — the graph
// descriptor is the same for every quant). No distribution ships a flow graph, so the flow window
// check is an implementation-time series check (`outputs/series/sbv2-F1/flow/`, opt-in
// `KARUME_SERIES=1`). Each case SKIPs explicitly when the assets it reads are absent — except a
// target's goldens absent from a complete fetch of the pinned dataset (`testAssetsFetched()`),
// which FAIL (the dataset at the pin lacks them), and a series flow directory that holds goldens
// but no container, which FAILs as a half-present series.
//
// The asset-free generator tests (fault injection, window semantics, input validation) stay in
// packages/models/tests/sbv2_relattn_parity_test.ts; only the cases that read real-weight assets
// (the goldens, the distribution mirror, the series) live here (ADR 0129).

import { type DirEntry, isNotFound, readDirSync, readFile } from "@karume/test-support/fs";
import { goldenDir, TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { parseSafetensors, prepareContainer, type PreparedModel } from "@karume/runtime";
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../models/tests/helpers/series-graphs.ts";
import {
  mirrorPresent,
  mirrorRoot,
  openSelectedContainer,
  selectMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";
import { buildRelattnTables, RELATTN_WINDOW_SIZE } from "../../models/src/sbv2/relattn-tables.ts";

/**
 * 系列 root。綴りの `sbv2-F1` は `sbv2/export.py` の `default_out_root()` が `--model-dir` の
 * ディレクトリ名から導いたもので、**当面この 1 話者を決め打ち**する。
 */
const SERIES_NAME = "sbv2-F1";
/** The goldens of the series (`outputs/test-assets/goldens/sbv2-F1/`). */
const GOLDEN_ROOT = goldenDir(SERIES_NAME);
/** The series itself — read only by the implementation-time flow window check. */
const SERIES_ROOT = new URL(`../../../outputs/series/${SERIES_NAME}/`, import.meta.url);
const MODEL_FILE = "model.krm";
/** The distribution that ships the voice graph (`models/karume-sbv2-jvnv/`). */
const MIRROR_ROOT = mirrorRoot("karume-sbv2-jvnv");
/**
 * The mirror selection of the voice window check: the speaker of the series, and one quant — the
 * graph descriptor (where `idx_v` is baked) is the same for every quant of the voice graph.
 */
const MIRROR_CHOICE = { model: "F1", quant: "i8" } as const;
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";
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

const discoverCases = (root: URL, target: string): readonly string[] =>
  listDir(new URL(`${target}/`, root))
    .filter(
      (entry) => entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX),
    )
    .map((entry) => entry.name)
    .sort();

/** Whether the test-asset root holds a complete fetch of the pin (missing goldens then FAIL). */
const FETCHED = testAssetsFetched();

const AVAILABLE = TABLE_INPUT_TARGETS.flatMap((target) =>
  discoverCases(GOLDEN_ROOT, target).map((ioFile) => ({ target, ioFile })),
);
/** The targets with no golden at all: a FAIL after a complete fetch, a visible SKIP otherwise. */
const MISSING_TARGETS = TABLE_INPUT_TARGETS.filter(
  (target) => discoverCases(GOLDEN_ROOT, target).length === 0,
);

for (const target of MISSING_TARGETS) {
  const where = new URL(`${target}/`, GOLDEN_ROOT).pathname;
  if (FETCHED) {
    test(`relative-position table parity: ${target} has goldens`, () => {
      throw new Error(
        `${where} holds no io.* golden, but outputs/test-assets/ holds a complete fetch of ` +
          `${TEST_ASSETS.repo}@${TEST_ASSETS.revision}: the dataset at the pin lacks them`,
      );
    });
  } else {
    test.skip(
      `relative-position table parity: ${target} (skipped: no goldens under ` +
        `outputs/test-assets/goldens/${SERIES_NAME}/${target}/; fetch: ${FETCH_COMMAND})`,
      () => {},
    );
  }
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
    const bytes = await readFile(new URL(`${target}/${ioFile}`, GOLDEN_ROOT));
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

/** Where each target's graph is read from, and whether this run can read it. */
type WindowSource = {
  readonly target: (typeof TABLE_INPUT_TARGETS)[number];
  readonly runnable: boolean;
  /** The title suffix (the opt-in of an implementation-time case, empty otherwise). */
  readonly suffix: string;
  readonly open: () => Promise<PreparedModel>;
};

const MIRROR_PRESENT = mirrorPresent(MIRROR_ROOT);
const FLOW_MODEL = new URL(`flow/${MODEL_FILE}`, SERIES_ROOT);
/**
 * Whether the series' flow target exists (its container or its goldens). MUST: read only under
 * SERIES_CHECKS. Goldens without the container run the case, whose open then FAILs — a
 * half-present series is not a SKIP.
 */
const FLOW_SERIES_PRESENT =
  SERIES_CHECKS && (modelPresent(FLOW_MODEL) || discoverCases(SERIES_ROOT, "flow").length > 0);

const WINDOW_SOURCES: readonly WindowSource[] = [
  {
    // The voice graph is shipped: read it from the distribution. The graph name is the
    // manifest's weights key (container-v1 §2.1).
    target: "voice",
    runnable: MIRROR_PRESENT,
    suffix: "",
    open: async () =>
      prepareContainer(
        await openSelectedContainer(
          await selectMirrorComponent(MIRROR_ROOT, "voice", MIRROR_CHOICE),
        ),
        "voice",
      ),
  },
  {
    // No distribution ships a flow graph: an implementation-time series check (ADR 0139
    // decision 6). MUST: the series is touched only under SERIES_CHECKS.
    target: "flow",
    runnable: FLOW_SERIES_PRESENT,
    suffix: SERIES_CHECKS_OPT_IN,
    open: async () =>
      prepareContainer(await openSeriesContainer(FLOW_MODEL), seriesGraph(SERIES_NAME, "flow")),
  },
];

if (!MIRROR_PRESENT) {
  console.warn(
    `[karume] skipping the voice window check: the distribution mirror ` +
      `${MIRROR_ROOT.pathname} is absent (or not karume/5).`,
  );
}
if (SERIES_CHECKS && !FLOW_SERIES_PRESENT) {
  console.warn(
    `[karume] skipping the flow window check: ${new URL("flow/", SERIES_ROOT).pathname} holds ` +
      "neither the container nor goldens. Export: " +
      "cd tools/export-recipes && uv run --group sbv2 python -m sbv2.export --model-dir ../../inputs/sbv2/F1",
  );
}

for (const { target, runnable, suffix, open } of WINDOW_SOURCES) {
  test.skipIf(!runnable)(
    `relative-position window: the baked ${target} table matches the TS constant${suffix}`,
    async () => {
      // 沈黙誤値クラスの門（ADR 0013）。key 側の表はホストが作るが、**value 側の表
      // `idx_v` はコンテナに焼き込まれていて幅が `2w+1`**。ここが TS の窓幅定数と食い違えば
      // ホストは違う幅の埋め込みを前提に添字を作っている — shape エラーにならず黙って
      // 誤るクラスなので、コンテナから読んだ幅で TS 側の定数を検算する。
      // The graph descriptor is in part 0 of the container (container-v1 §8) — no weight block
      // is read.
      const model = await open();
      const baked = model.graph.nodes
        .filter((node) => node.op === "sym_prefix_slice")
        .map((node) => model.graph.values[node.ins[0]].shape);

      // 表が入力へ昇格した後に焼き込みが残るのは value 側の 1 本だけ（key 側の 2 本が
      // 残っていたら昇格が効いていない＝ 134MB の定数を抱えている）。
      assertEquals(baked.length, 1, `${target} の焼き込み表の本数`);
      assertEquals(baked[0].length, 4, `${target} の焼き込み表の rank`);
      assertEquals(baked[0][3], 2 * RELATTN_WINDOW_SIZE + 1, `${target} の相対位置埋め込みの幅`);
    },
  );
}
