// The **real-asset** gate of ADR 0005's "all cases SKIP = explicit FAIL" (the GPU one is
// tests/gpu_gate_test.ts).
//
// The real-weight e2e tests (birefnet / dacvae / deberta / depth_anything / embeddinggemma / gemma4
// / irodori / minicpm5 / sbv2 / siglip2 / vowel_detector) are `ignore: !available ||
// !GPU_AVAILABLE`, and `available` is false when the listing of the series root is empty. On a
// machine that has a GPU but not a single real-weight asset (a new work machine, or a worktree
// taken to another host), they all SKIP and the full verify (`vp test`) still turns green — a silent
// apparent success that reads "not verified" as "verified", so this gate fails one test for it.
//
// It looks at **more than presence** (ADR 0108 stage 3, acceptance ②). A series is now a `krm`
// container (the part sequence of `model.krm`), so a directory that exists but still holds the old
// shards can really occur. That is a different breakage from "no assets": the e2e side's
// `modelPresent` turns false and it becomes a **silent SKIP**. The second gate checks, per series,
// (a) that the container opens as a part sequence (`openContainer` checks the part count and
// lengths too) and (b) that the graph name the container declares is the expected weights key. It
// does not read the blocks' bytes (the three-way hash split of §7 — each e2e test checks the actual
// bytes).
//
// The source of truth for the graph-name table is the one in
// `packages/models/tests/helpers/series-graphs.ts`, and **every e2e test reads from the same one**.
// If the gate and the e2e tests kept separate tables, rewriting only one would fail nothing (the
// gate says "as expected" while the e2e test reads another graph).
//
// Where the assets exist, this gate itself **passes (it shows up as one green test)**. Making it
// ignore would make "the gate works" indistinguishable from "the gate is gone" (the same reason as
// the GPU gate).
//
// NOTE: it looks only at the one root `outputs/series/` shared by the 11 real-weight e2e tests. The
// distribution mirrors (`models/`) belong to distribution_gate_test.ts; their SKIPs are outside
// this gate's reach.
//
// ADR 0139 decision 5 moves this gate to the test-asset root `outputs/test-assets/` (the fetched
// cache of the pinned private dataset): it is red while no fetch completed there (no `.revision`)
// or the fetched revision differs from the pin (`TEST_ASSETS` in
// `@karume/test-support/test-assets`), with the same opt-out `KARUME_ALLOW_NO_ASSETS=1`. The series
// checks below stay until the families have moved off `outputs/series/` (ADR 0139 staging 4);
// they are retired with `series-graphs.ts` then.

import {
  isNotFound,
  makeTempDirSync,
  mkdirSync,
  removeSync,
  statSync,
  writeFileSync,
} from "@karume/test-support/fs";
import { assert, assertEquals, assertFalse, assertStringIncludes } from "@std/assert";
import {
  readFetchedRevision,
  REVISION_FILE,
  TEST_ASSETS,
  TEST_ASSETS_ROOT,
  testAssetsFetched,
  type TestAssetsPin,
} from "@karume/test-support/test-assets";
import { describe, it, test } from "vite-plus/test";
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { SERIES_GRAPHS } from "../../models/tests/helpers/series-graphs.ts";

/**
 * The opt-out that explicitly allows "all cases SKIP without real assets" (the same form as
 * `KARUME_ALLOW_NO_GPU`).
 *
 * MUST: the default is to fail loudly. Passing on a work machine that touches only the models
 * package, or on a CI without assets, requires this environment variable as a statement of intent.
 */
const ALLOW_NO_ASSETS = process.env.KARUME_ALLOW_NO_ASSETS === "1";

/**
 * The single root where the real-weight e2e tests look for assets (the parent of each e2e test's
 * `SERIES_ROOT` / `SERIES_PARENT`).
 */
const SERIES_PARENT = new URL("../../../outputs/series/", import.meta.url);

/**
 * The representative path of a series container (in reality the part sequence
 * `model-NNNNN-of-NNNNN.krm`).
 */
const MODEL_FILE = "model.krm";

/**
 * The representative path of the container of a series + component directory (the component is an
 * empty string when the weights sit at the series root).
 */
const representativeOf = (parent: URL, series: string, component: string): URL =>
  new URL(
    component === "" ? MODEL_FILE : `${component}/${MODEL_FILE}`,
    new URL(`${series}/`, parent),
  );

/**
 * Whether at least one container of the series in the table (`SERIES_GRAPHS`) exists.
 *
 * MUST: it looks only at the containers of the series in the table, with the same predicate
 * `modelPresent` that the e2e tests use to decide a SKIP. Counting the directories under the root
 * would turn the gate green on a machine with only series outside the table (`anima-*` and the
 * like) or only empty directories, while every real-weight e2e test SKIPs. `modelPresent`
 * propagates I/O errors other than NotFound — reading a permission error and the like as "no
 * assets" would have the gate report a broken environment as missing assets.
 */
const seriesAssetsPresent = (parent: URL): boolean =>
  Object.entries(SERIES_GRAPHS).some(([series, components]) =>
    Object.keys(components).some((component) =>
      modelPresent(representativeOf(parent, series, component)),
    ),
  );

/** Whether a directory exists (absence is false; any other I/O error propagates). */
const directoryPresent = (url: URL): boolean => {
  try {
    const stat = statSync(url);
    return stat.isDirectory;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/** The command that fetches the pinned test assets (named in the gate's messages). */
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";

/**
 * The verdict of the test-asset check (ADR 0139 decision 5) — a pure function of the revision the
 * last completed fetch recorded and the pin: `undefined` when they agree, otherwise why the gate is
 * red. An absent root reads as "no completed fetch" (`readFetchedRevision` returns `undefined`).
 *
 * MUST: a fetched revision that differs from the pin is red, not a warning — the goldens of one
 * revision are the expectations of the tests of that commit, and an older set would compare the
 * new tests against stale references (or skip on files the older set does not have).
 */
const testAssetsVerdict = (
  fetched: string | undefined,
  pin: TestAssetsPin,
  root: URL,
): string | undefined => {
  const remedy =
    `Run \`${FETCH_COMMAND}\` from the repository root (ADR 0005: a run whose real-weight tests ` +
    "lack their goldens is a FAIL). To pass deliberately on a machine without test assets, set " +
    "KARUME_ALLOW_NO_ASSETS=1.";
  if (fetched === undefined) {
    return (
      `${root.pathname} holds no completed fetch of ${pin.repo} (no ${REVISION_FILE}); the pinned ` +
      `revision is ${pin.revision}. ${remedy}`
    );
  }
  if (fetched !== pin.revision) {
    return (
      `${root.pathname} was fetched at revision ${fetched} of ${pin.repo}, but the pinned revision ` +
      `is ${pin.revision}. ${remedy}`
    );
  }
  return undefined;
};

test.skipIf(ALLOW_NO_ASSETS)(
  "assets gate: the test-asset root holds a completed fetch of the pinned revision (ADR 0139 decision 5)",
  () => {
    const verdict = testAssetsVerdict(readFetchedRevision(), TEST_ASSETS, TEST_ASSETS_ROOT);
    assert(verdict === undefined, verdict);
  },
);

test.skipIf(ALLOW_NO_ASSETS)(
  "assets gate: an all-SKIP run with not a single real-weight series is an explicit FAIL (ADR 0005)",
  () => {
    assert(
      seriesAssetsPresent(SERIES_PARENT),
      `${SERIES_PARENT.pathname} holds not a single container of the series in the table (packages/models/tests/helpers/series-graphs.ts), ` +
        "so every real-weight e2e test was SKIPped. " +
        "ADR 0005 treats this as a FAIL (the release decision requires the golden comparison on real assets). " +
        "To pass deliberately in an environment without assets, set KARUME_ALLOW_NO_ASSETS=1.",
    );
  },
);

test.skipIf(ALLOW_NO_ASSETS)(
  "assets gate: the containers of the present series open and declare the expected weights key (ADR 0108 stage 3)",
  async () => {
    const broken: string[] = [];
    for (const [series, components] of Object.entries(SERIES_GRAPHS)) {
      const root = new URL(`${series}/`, SERIES_PARENT);
      if (!directoryPresent(root)) continue;
      for (const [component, expected] of Object.entries(components)) {
        const where = component === "" ? series : `${series}/${component}`;
        const representative = representativeOf(SERIES_PARENT, series, component);
        try {
          // Resolving the part sequence, checking the part count and lengths, parsing the two
          // documents and merging with the binding table all run here.
          const opened = await openSeriesContainer(representative);
          const declared = Object.keys(opened.graphs);
          if (declared.length !== 1 || declared[0] !== expected) {
            broken.push(
              `${where}: the graph name is ${declared.join(" / ")} (expected ${expected})`,
            );
          }
        } catch (cause) {
          broken.push(
            `${where}: cannot open the container — ${cause instanceof Error ? cause.message : cause}`,
          );
        }
      }
    }
    assertEquals(
      broken,
      [],
      `The series under ${SERIES_PARENT.pathname} cannot be read as containers, or do not declare the expected ` +
        "weights key. A directory that exists but still holds the old shards makes the e2e tests SKIP silently " +
        "(ADR 0005 makes this a FAIL). Re-bake it with the export of tools/export-recipes, " +
        "or, for a retired series, remove its row from the table above.",
    );
  },
);

describe("the assets gate's verdict (whether at least one container of the series in the table exists)", () => {
  /** Takes the verdict with a temporary directory as the series root, then cleans up. */
  const judge = (arrange: (parent: URL) => void): boolean => {
    const parent = new URL(`file://${makeTempDirSync()}/`);
    try {
      arrange(parent);
      return seriesAssetsPresent(parent);
    } finally {
      removeSync(parent, { recursive: true });
    }
  };

  it("with only directories of series outside the table, reports no assets even if a container exists", () => {
    assertFalse(
      judge((parent) => {
        const root = new URL("anima-f16/", parent);
        mkdirSync(root);
        writeFileSync(new URL("model-00001-of-00001.krm", root), new Uint8Array());
      }),
    );
  });

  it("reports no assets when a series directory of the table exists but holds no container", () => {
    assertFalse(judge((parent) => mkdirSync(new URL("birefnet-hr-1024-f16/", parent))));
  });

  it("reports assets when a series of the table holds a container part", () => {
    assert(
      judge((parent) => {
        const root = new URL("birefnet-hr-1024-f16/", parent);
        mkdirSync(root);
        writeFileSync(new URL("model-00001-of-00001.krm", root), new Uint8Array());
      }),
    );
  });

  it("reports assets when a component directory of a series of the table holds a container", () => {
    assert(
      judge((parent) => {
        const component = new URL("dacvae-32dim/decoder/", parent);
        mkdirSync(component, { recursive: true });
        writeFileSync(new URL("model-00001-of-00001.krm", component), new Uint8Array());
      }),
    );
  });
});

describe("assets gate verdict (the fetched revision of the test-asset root against the pin)", () => {
  const pin: TestAssetsPin = {
    repo: "hdae/karume-test-assets",
    revision: "0123456789abcdef0123456789abcdef01234567",
  };

  /**
   * Reads the verdict for the root `<temporary dir>/<child>` arranged by `arrange`, then removes
   * the temporary directory.
   */
  const verdictOf = (arrange: (root: URL) => void, child = ""): string | undefined => {
    const dir = new URL(`file://${makeTempDirSync({ prefix: "karume-test-assets-" })}/`);
    const root = new URL(child, dir);
    try {
      arrange(root);
      return testAssetsVerdict(readFetchedRevision(root), pin, root);
    } finally {
      removeSync(dir, { recursive: true });
    }
  };

  it("is red when no fetch completed (no .revision), naming the pin and the fetch command", () => {
    const verdict = verdictOf(() => {});
    assert(verdict !== undefined);
    assertStringIncludes(verdict, `no ${REVISION_FILE}`);
    assertStringIncludes(verdict, pin.revision);
    assertStringIncludes(verdict, FETCH_COMMAND);
  });

  it("is red when the root itself is absent", () => {
    const verdict = verdictOf(() => {}, "missing/");
    assert(verdict !== undefined);
    assertStringIncludes(verdict, `no ${REVISION_FILE}`);
  });

  it("is red when the fetched revision differs from the pin, naming both", () => {
    const stale = "f".repeat(40);
    const verdict = verdictOf((root) =>
      writeFileSync(new URL(REVISION_FILE, root), new TextEncoder().encode(`${stale}\n`)),
    );
    assert(verdict !== undefined);
    assertStringIncludes(verdict, `fetched at revision ${stale}`);
    assertStringIncludes(verdict, `pinned revision is ${pin.revision}`);
    assertStringIncludes(verdict, FETCH_COMMAND);
  });

  it("is green when the fetched revision is the pin (the trailing newline is not part of it)", () => {
    assertEquals(
      verdictOf((root) =>
        writeFileSync(new URL(REVISION_FILE, root), new TextEncoder().encode(`${pin.revision}\n`)),
      ),
      undefined,
    );
  });
});

describe("testAssetsFetched (a complete fetch of the pin, which makes a missing asset a FAIL)", () => {
  const pin: TestAssetsPin = {
    repo: "hdae/karume-test-assets",
    revision: "0123456789abcdef0123456789abcdef01234567",
  };

  const fetchedWith = (record: string | undefined): boolean => {
    const root = new URL(`file://${makeTempDirSync({ prefix: "karume-test-assets-" })}/`);
    try {
      if (record !== undefined) {
        writeFileSync(new URL(REVISION_FILE, root), new TextEncoder().encode(record));
      }
      return testAssetsFetched(root, pin);
    } finally {
      removeSync(root, { recursive: true });
    }
  };

  it("is true only when the record is the pin", () => {
    assert(fetchedWith(`${pin.revision}\n`));
    assertFalse(fetchedWith(`${"f".repeat(40)}\n`));
  });

  it("is false when no complete fetch left a record (none, or a partial fetch)", () => {
    assertFalse(fetchedWith(undefined));
  });
});
