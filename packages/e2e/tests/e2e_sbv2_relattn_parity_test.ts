// Pins, on real data, the **byte parity of the host TS generator and the Python generator** for
// the `(T,T)` tables of the relative-position attention (exactly the "new verification
// responsibility" of ADR 0013).
//
// flow / voice do not bake the tables in but take them as **graph inputs**. So at run time the
// host builds the tables, while Python (`build_relattn_tables` in
// `tools/export-recipes/sbv2/patch.py`) builds the golden ones — if the formulas diverge,
// **the shape still matches and only the values are wrong**. Worse, some kinds of error "are
// shared by the host and the golden and slip through the E2E", so the two are compared directly
// here, apart from the E2E.
//
// As long as this test is green, the E2E running on the golden's tables and on host-built tables
// is **the same computation** (byte parity, so swapping them does not move a value).
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
 * The series root. The spelling `sbv2-F1` is what `default_out_root()` in `sbv2/export.py`
 * derives from the `--model-dir` directory name; **for now this one speaker is hard-coded**.
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

/**
 * The targets that take the tables as **graph inputs** (front bakes them, so it is out of scope —
 * ADR 0013).
 */
const TABLE_INPUT_TARGETS = ["flow", "voice"] as const;

/** The tensor keys of the tables stored in io (= the graph input names). */
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
 * Exact element-wise equality (`Object.is`). It tells `0` from `-0`, so it can claim bit identity
 * even for f32 (`===` holds for `-0 === 0` and would miss a mixed-up signed zero).
 */
const assertSameElements = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  where: string,
): void => {
  assertEquals(actual.length, expected.length, `${where}: element count`);
  for (let index = 0; index < expected.length; index += 1) {
    if (!Object.is(actual[index], expected[index])) {
      throw new Error(
        `${where}: element ${index} differs (TS=${actual[index]} / Python=${expected[index]})`,
      );
    }
  }
};

for (const { target, ioFile } of AVAILABLE) {
  test(`relative-position table parity: ${target} / ${ioFile} (TS generator ↔ the golden's Python generator)`, async () => {
    const bytes = await readFile(new URL(`${target}/${ioFile}`, GOLDEN_ROOT));
    const io = parseSafetensors(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    );
    const idxKView = io.tensors.get(IDX_K_KEY);
    const validView = io.tensors.get(VALID_KEY);
    assert(idxKView !== undefined && validView !== undefined, `${ioFile} has no table input`);
    assertEquals(idxKView.dtype, "I32", `${IDX_K_KEY} storage dtype`);
    assertEquals(validView.dtype, "F32", `${VALID_KEY} storage dtype`);

    // The actual length is taken from the golden's shape (fixing the length on the TS side would,
    // once cases are added, "compare equal lengths with each other and always be green").
    assertEquals(idxKView.shape.length, 2, `${IDX_K_KEY} rank`);
    assertEquals(idxKView.shape, validView.shape, "shape of the two tables");
    const [rows, columns] = idxKView.shape;
    assertEquals(rows, columns, "the relative-position table is square (self-attention)");

    const built = buildRelattnTables(rows);
    assertEquals(built.idxK.shape, [...idxKView.shape], `${target}/${ioFile}: idx_k shape`);
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
      // The check for the silent-wrong-value class (ADR 0013). The host builds the key-side
      // tables, but **the value-side table `idx_v` is baked into the container, with width
      // `2w+1`**. If that disagrees with the TS window-width constant, the host builds indices
      // assuming embeddings of another width — a class that goes silently wrong without a shape
      // error, so the TS constant is cross-checked against the width read from the container.
      // The graph descriptor is in part 0 of the container (container-v1 §8) — no weight block
      // is read.
      const model = await open();
      const baked = model.graph.nodes
        .filter((node) => node.op === "sym_prefix_slice")
        .map((node) => model.graph.values[node.ins[0]].shape);

      // After the tables were promoted to inputs, only the one value-side table stays baked (if
      // the two key-side ones remained, the promotion did not take effect = 134 MB of constants
      // are carried).
      assertEquals(baked.length, 1, `${target}: count of baked tables`);
      assertEquals(baked[0].length, 4, `${target}: rank of the baked table`);
      assertEquals(
        baked[0][3],
        2 * RELATTN_WINDOW_SIZE + 1,
        `${target}: width of the relative-position embedding`,
      );
    },
  );
}
