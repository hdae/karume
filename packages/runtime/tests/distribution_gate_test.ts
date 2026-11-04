// The **distribution-mirror** version of the ADR 0005 guard "all cases SKIP is an explicit FAIL"
// (the real-weight series version is tests/e2e_assets_gate_test.ts, the GPU version is
// tests/gpu_gate_test.ts).
//
// The checks rooted at the distribution mirrors (`models/karume-<family>/`) — the real-asset e2e of
// anima / sbv2 / irodori / gemma4 / QAT / wan (packages/models/tests/e2e_*_test.ts and
// helpers/irodori-assets.ts) and the distribution sections of the fusion hit counts (anima /
// anima-extra / irodori / gemma4 / gemma4-qat-mobile / wan — tests/e2e_*_fusion_counts_test.ts) —
// all `ignore` themselves on the absence of `karume.json` and go green leaving only a
// `console.warn`. On a machine with only `outputs/series/` (a new work machine, a worktree taken
// to another host), e2e_assets_gate_test.ts stays green while the real-asset checks vanish
// entirely, so one test fails here.
//
// The reach of the absence check (a whole mirror missing) is **every published mirror** (ruling of
// 2026-09-25). Mirrors that no e2e reads yet (siglip2 and others) are listed too — matching the
// guard's reach to "what the e2e reads today" would bring the silent SKIP back the day an e2e is
// added and the guard update is forgotten. The unpublished vowel-detector is not listed (it is
// listed once its distribution exists). A mirror that the e2e (and the fusion hit-count check)
// roots on is listed even when unpublished (wan2.1).
//
// It checks **more than presence** (ADR 0108 stage 3, acceptance item ②). A `karume.json` that is
// present but still of an old major, or that declares a part that is missing or whose length
// differs from the declaration, either fails the e2e above with "assets that cannot be opened" or,
// worse, makes it read **leftovers of a previous write** against this run's expected values. Both
// are failures other than "the assets are missing", so the guard parses the manifest and checks
// that the parts exist and have the right lengths. It does not read contents (the block sha256) —
// every e2e matches the actual bytes on the `openContainer` path (§7 keeps the three hashes
// apart).
//
// Since ADR 0139 decision 5 the check covers every model × quant × weights key a manifest
// declares, not only the default selection, and it also opens each container's two documents
// through the manifest and checks that its graph is named by the weights key — the graph-name
// check that the assets gate ran on the series (`series-graphs.ts`) moves here.
//
// The reach of the manifest parse and the default-selection check is derived from **the layout of
// `models/`**, not from the roster (ruling of 2026-10-11). `models/` is a root that holds only
// distributions (docs/assets-layout.md), so every child directory goes through hub's parser.
// Relying on the roster would let a new distribution that `karume dist` built but nobody added to
// the roster (such as the unpublished vowel-detector) pass as "built but unreadable by hub". The
// roster {@link DISTRIBUTIONS} is responsible only for detecting "something that should be there is
// missing", which cannot be derived from the layout.
//
// On a machine that has the assets, this guard itself **passes (it shows up as one green case)**.
// Making it ignore would make "is the guard working, or has the guard vanished too" impossible to
// tell apart (the same reason as the other two guards).

import {
  type DirEntry,
  isFilesystemLoop,
  isNotFound,
  makeTempDirSync,
  mkdirSync,
  readDirSync,
  readTextFile,
  removeSync,
  statSync,
  symlinkSync,
  writeFileSync,
  writeTextFileSync,
} from "@karume/test-support/fs";
import {
  assert,
  assertEquals,
  AssertionError,
  assertRejects,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import {
  type ContainerRef,
  loadManifest,
  type Manifest,
  type ModelEntry,
  openContainerSource,
  parseManifest,
} from "@karume/hub";
import { openContainer, parseIrDeclarationValue } from "../mod.ts";
import { type WrittenContainer, writeModelContainer } from "./helpers/container-write.ts";
import { mirrorDirectory } from "../../models/tests/helpers/mirror.ts";

/**
 * The opt-out that explicitly allows "all SKIP without distribution mirrors" (the same form as
 * `KARUME_ALLOW_NO_ASSETS`).
 *
 * MUST: the default is fail loudly. To pass on work machines or CI without distributions, this
 * environment variable is required as a statement of intent (kept separately from the real-weight
 * series opt-out — machines that have only one of the two exist).
 */
const ALLOW_NO_DISTRIBUTION = process.env.KARUME_ALLOW_NO_DISTRIBUTION === "1";

/**
 * The root that holds the distribution mirrors (built the same way as the URL the checks above read
 * `karume.json` from).
 */
const MODELS_ROOT = new URL("../../../models/", import.meta.url);

/**
 * The distribution mirrors the guard **requires to exist** (the roster of the absence check — the
 * reach of the manifest checks is derived from the layout by {@link discoverDistributions}) = the
 * 15 published repositories (including `karume-umt5-xxl` and `karume-wan2.2`, first published in
 * 0.14.0, `karume-moss-image`, first published in 0.15.1, and `karume-gemma4-qat-mobile`,
 * published under this name in 0.16.0) + the unpublished `karume-wan2.1`.
 *
 * The two published ones were on the guard's roster before publication: `karume-umt5-xxl` is the
 * cross-repo reference root of the e2e of Wan's GPU path (ADR 0119 stage 10d — without it the
 * absence would be a silent SKIP), and `karume-wan2.2` is the root of Wan2.2's end-to-end e2e
 * (`e2e_wan_ti2v_pipeline_test.ts`) and of the DiT e2e's record of the practical presets
 * (`e2e_wan_ti2v_dit_test.ts` — it reads the bundle of presets from the manifest) (ADR 0121 stage
 * 8).
 *
 * `karume-gemma4-qat-mobile` is the mobile QAT distribution first published in 0.15.1 as
 * `karume-gemma4-qat`, renamed on 2026-11-04 and published under the new name in 0.16.0: the
 * mobile QAT public-entry e2e and the fusion hit-count check root on it. `karume-gemma4-qat` now
 * holds the q4_0 QAT line (E2B / E4B / 12B with their drafters, published in 0.16.0), which the
 * q4_0 QAT mirror e2e, its sha256 rows and the 12B e2e root on.
 *
 * `karume-wan2.1` is unpublished (not published to HF — ADR 0118 addendum (2026-10-15)), but Wan's
 * end-to-end e2e (`e2e_wan_pipeline_test.ts`) and the fusion hit-count check root on it (ADR 0118
 * stage 7).
 *
 * `karume-moss-image` was on the roster before its publication as well (ADR 0138 addendum of
 * 2026-11-01): the pipeline e2e (`packages/e2e/tests/e2e_moss_pipeline_test.ts`) roots on it, so
 * its absence must not be a silent skip.
 */
const DISTRIBUTIONS = [
  "karume-anima",
  "karume-anima-extra",
  "karume-birefnet-hr",
  "karume-depth-anything-v2",
  "karume-gemma4",
  "karume-gemma4-qat",
  "karume-gemma4-qat-mobile",
  "karume-irodori-v4-small",
  "karume-irodori-v4.1-small",
  "karume-lucida",
  "karume-moss-image",
  "karume-sbv2-jvnv",
  "karume-siglip2",
  "karume-wan2.1",
  "karume-wan2.2",
  "karume-umt5-xxl",
] as const;

/**
 * The distribution manifest major this version's reader accepts (older versions are not read — ADR
 * 0109 decision 1).
 */
const MANIFEST_FORMAT = "karume/5";

/**
 * The roster of the absence check (in the form the per-mirror checks look up "is it on the
 * roster").
 */
const REQUIRED: ReadonlySet<string> = new Set(DISTRIBUTIONS);

/**
 * Suffixes of `karume dist`'s working directories (`.staging` of
 * `karume.artifacts.staged_publication` = being written, `.old` = set aside before the swap).
 * Neither is a mirror, so neither goes through the manifest check — reading one being written would
 * turn the check red mid-build, and the set-aside one is reclaimed by the next build.
 */
const WORK_SEAT_SUFFIXES = [".staging", ".old"] as const;

const distributionRoot = (models: URL, series: string): URL => new URL(`${series}/`, models);

/**
 * Enumerates the distribution mirrors directly under `models` **from the layout** (in name order,
 * excluding working directories and files).
 *
 * NOTE: a child may be a symbolic link (stat follows links) — some machines keep the assets
 * elsewhere and expose them through a link. A machine without `models` itself returns empty (the
 * roster check names the absence and fails). I/O errors other than NotFound while listing `models`
 * propagate (the same reason as {@link manifestPresent}). A child whose stat fails is kept in the
 * list by {@link isMirrorCandidate} and fails by name inside the per-mirror test.
 */
const discoverDistributions = (models: URL): readonly string[] => {
  let entries: DirEntry[];
  try {
    entries = [...readDirSync(models)];
  } catch (cause) {
    if (isNotFound(cause)) return [];
    throw cause;
  }
  return entries
    .map((entry) => entry.name)
    .filter((name) => !WORK_SEAT_SUFFIXES.some((suffix) => name.endsWith(suffix)))
    .filter((name) => isMirrorCandidate(new URL(name, models)))
    .sort();
};

/**
 * Whether a child directly under `models` goes through the manifest check (directories do, files
 * do not).
 *
 * MUST: a child whose stat throws (a broken symbolic link, a link that cannot be followed, an
 * unreadable child) **stays on the checked side**. This runs at module top level (at test
 * registration), so throwing would fail the whole file without naming any mirror and take the
 * absence check down with it. Keeping it makes the same error show up inside the per-mirror test
 * as a FAIL carrying that mirror's name (the exception is not swallowed; only where it surfaces
 * moves).
 */
const isMirrorCandidate = (child: URL): boolean => {
  try {
    return statSync(child).isDirectory;
  } catch {
    return true;
  }
};

/**
 * Whether that series' manifest is in place.
 *
 * MUST: anything other than NotFound propagates — reading a permission error or the like as "the
 * assets are missing" would make the guard itself report a broken environment as absent assets.
 * This only checks that `karume.json` is a file; {@link assertDistributionIntact} checks the
 * contents.
 */
const manifestPresent = (root: URL): boolean => {
  try {
    return statSync(new URL("karume.json", root)).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/**
 * For the `ignore` decision at registration: whether the manifest is **known to be absent**.
 *
 * MUST: an I/O error other than NotFound falls to "not known to be absent" (`false`) and lets the
 * test run — throwing at registration would fail the whole file without a name (the same reason as
 * {@link isMirrorCandidate}). {@link manifestPresent} inside the test that runs throws the same
 * error as a FAIL carrying that mirror's name.
 */
const manifestKnownAbsent = (root: URL): boolean => {
  try {
    return !manifestPresent(root);
  } catch {
    return false;
  }
};

/** The file length (`undefined` when absent; other I/O errors propagate). */
const fileBytes = (url: URL): number | undefined => {
  try {
    const stat = statSync(url);
    return stat.isFile ? stat.size : undefined;
  } catch (cause) {
    if (isNotFound(cause)) return undefined;
    throw cause;
  }
};

/**
 * Asserts that the `karume.json` of every mirror in `series` is present directly under `models`
 * (naming the missing ones).
 */
const assertNoDistributionMissing = (models: URL, series: readonly string[]): void => {
  const missing = series.filter((name) => !manifestPresent(distributionRoot(models, name)));
  assert(
    missing.length === 0,
    `models/ is missing the karume.json of distribution mirrors: ${missing.join(", ")}. ` +
      "On a machine without a distribution mirror, the real-asset e2e and the fusion hit-count checks rooted on it SKIP silently, " +
      "so by ADR 0005 this is treated as a FAIL. " +
      "Build it with dist.py in tools/export-recipes (--pipeline and the arguments are in the usage examples at the top of dist.py — " +
      "karume-anima-extra, built with cross-repo references, follows the procedure in docs/release-runbook.md). " +
      "To pass on purpose in an environment without distributions, set KARUME_ALLOW_NO_DISTRIBUTION=1.",
  );
};

/**
 * Asserts that the mirror at `root` has a `karume/5` manifest and that **every container it
 * declares** — every model × weights key × dtype label, which covers every quant's selection —
 * is intact (ADR 0139 decision 5):
 *
 * - each part is present with its declared size (cross-repo parts are skipped: they live in their
 *   own mirror, which this gate checks under its own name — ADR 0038 §7);
 * - the container opens through the manifest (header, the two documents checked against the
 *   declared length and sha256, the declared part lengths) and its one graph is named by the
 *   weights key (container-v1 §2.1). A cross-repo container is opened from its sibling mirror.
 *
 * Block bytes are never read (the block sha256 is checked by the tests that build Sessions — §7
 * keeps the three hashes apart), so the cost is one stat per part plus part 0's documents per
 * container: about two hundred containers over the fifteen mirrors open in about a second on the
 * development machine (2026-11-02), GPU-free.
 */
const assertDistributionIntact = async (root: URL, series: string): Promise<void> => {
  // MUST: the absence of a mirror that is not on the roster is visible only here — `models/` is a
  // root for distributions only, so a child directory without `karume.json` is a half-built one or
  // the leftovers of a previous build, and an e2e reading it would fail in a different way.
  assert(
    manifestPresent(root),
    `models/${series}/ has no karume.json. models/ is a root that holds only distributions, so ` +
      "this is a half-built or leftover directory. Rebuild it with dist.py in tools/export-recipes, or remove it.",
  );
  const manifest = parseManifest(await readTextFile(new URL("karume.json", root)));
  // The parse checks the major itself too, but `karume/5` is asserted **by name** — the day hub
  // accepts more majors, this keeps the guard from letting "a mirror still on the old major"
  // through.
  assertEquals(manifest.format, MANIFEST_FORMAT, `${series}: manifest format`);
  const broken: string[] = [];
  const intact: {
    readonly where: string;
    readonly key: string;
    readonly container: ContainerRef;
  }[] = [];
  for (const [model, entry] of Object.entries(manifest.models)) {
    for (const [key, dtypes] of Object.entries(entry.weights)) {
      for (const [dtype, { container }] of Object.entries(dtypes)) {
        const where = `${model}/${key}/${dtype}`;
        const missing: string[] = [];
        for (const [index, part] of container.parts.entries()) {
          if (part.repo !== undefined) continue;
          const actual = fileBytes(new URL(part.path, root));
          if (actual === undefined) {
            missing.push(`${where}[${index}] ${part.path}: the declared part is absent`);
          } else if (actual !== part.size) {
            missing.push(`${where}[${index}] ${part.path}: ${actual} bytes, declared ${part.size}`);
          }
        }
        broken.push(...missing);
        // A container with a missing or short part cannot be opened; it is reported once, above.
        if (missing.length === 0) intact.push({ where, key, container });
      }
    }
  }
  if (intact.length > 0) {
    const loaded = await loadManifest(mirrorDirectory(root));
    for (const { where, key, container } of intact) {
      try {
        const opened = await openContainer(
          { kind: "source", source: openContainerSource(loaded, container) },
          container.descriptor,
        );
        const graphs = Object.keys(opened.graphs);
        if (graphs.length !== 1 || graphs[0] !== key) {
          broken.push(
            `${where}: the container's graphs are ${graphs.join(" / ")}, expected ${key}`,
          );
        }
      } catch (cause) {
        broken.push(
          `${where}: the container does not open — ${cause instanceof Error ? cause.message : cause}`,
        );
      }
    }
  }
  assertEquals(
    broken,
    [],
    `models/${series}/ does not match its manifest. A tool or test reading it would hit a ` +
      "half-built or stale container, so ADR 0005 counts this as a FAIL. Rebuild the mirror with " +
      "tools/export-recipes dist.py.",
  );
};

/**
 * Whether label / description is present (whitespace only counts as absent — the same as the model
 * card display).
 */
const hasText = (value: string | undefined): boolean => (value ?? "").trim() !== "";

/**
 * Whether two weights maps are the same (the same set of components, with the same dtype label per
 * component).
 */
const sameWeights = (
  left: Readonly<Record<string, string>>,
  right: Readonly<Record<string, string>>,
): boolean => {
  const roles = Object.keys(left);
  return (
    roles.length === Object.keys(right).length &&
    roles.every((role) => Object.hasOwn(right, role) && right[role] === left[role])
  );
};

/** The session knobs that select int8 activations (C2 of {@link defaultPresetViolations}). */
const A8_KNOBS = ["linearCompute", "attentionCompute", "convCompute"] as const;

/**
 * Returns where one model's default preset and preset list break the three constraints a
 * distribution must satisfy (one sentence per violation, naming the preset; empty when they are
 * satisfied). DECIDED: ADR 0128 decision 2.
 *
 * - C1: the default preset runs on plain WebGPU — `quants[defaultQuant].gpuFeatures` is absent or
 *   has no true member (in today's vocabulary: it does not request `shaderF16`).
 * - C2: a preset with any of {@link A8_KNOBS} (`linearCompute`, `attentionCompute`,
 *   `convCompute`) = `"a8"` has a sibling preset under another name with none of the three
 *   and an identical weights map (same roles, same labels = same containers) — the three knobs
 *   since ADR 0128's addendum of 2026-10-29 (amendment 5; `convCompute` is ADR 0133 decision 6).
 *   This sibling is the fallback the runtime's warning on devices without dp4a points to
 *   (packages/runtime/src/runtime/a8-emulation-warning.ts), and it needs no extra download.
 * - C3: in a model with two or more presets, every preset has a label and a description that are
 *   non-empty after trim (what the selection UI and the model card show to people). A model with a
 *   single preset has nothing to choose, so it is excluded.
 */
const defaultPresetViolations = (entry: ModelEntry): string[] => {
  const violations: string[] = [];
  const requested = Object.entries(entry.quants[entry.defaultQuant]?.gpuFeatures ?? {})
    .filter(([, value]) => value === true)
    .map(([feature]) => feature);
  if (requested.length > 0) {
    violations.push(
      `C1: default preset '${entry.defaultQuant}' requests GPU features` +
        ` (${requested.join(", ")}) — the default must run on plain WebGPU`,
    );
  }
  const presets = Object.entries(entry.quants);
  for (const [name, quant] of presets) {
    const knobs = A8_KNOBS.filter((knob) => quant.session[knob] === "a8").map(
      (knob) => `${knob}: "a8"`,
    );
    if (knobs.length === 0) continue;
    const sibling = presets.some(
      ([other, candidate]) =>
        other !== name &&
        A8_KNOBS.every((knob) => candidate.session[knob] !== "a8") &&
        sameWeights(quant.weights, candidate.weights),
    );
    if (!sibling) {
      violations.push(
        `C2: preset '${name}' has int8 activations (${knobs.join(", ")}), but ` +
          "there is no sibling preset with the same weights and no activation quantization",
      );
    }
  }
  if (presets.length >= 2) {
    for (const [name, quant] of presets) {
      for (const field of ["label", "description"] as const) {
        if (!hasText(quant[field])) {
          violations.push(`C3: preset '${name}' has no ${field} (whitespace only counts as none)`);
        }
      }
    }
  }
  return violations;
};

/**
 * Asserts that every model in the manifest satisfies the constraints of ADR 0128 decision 2
 * (C1–C3), naming the series and the model.
 */
const assertDefaultPresetPolicy = (manifest: Manifest, series: string): void => {
  for (const [model, entry] of Object.entries(manifest.models)) {
    assertEquals(
      defaultPresetViolations(entry),
      [],
      `models/${series}/: the default preset and preset list of model '${model}' ` +
        "break the constraints of ADR 0128 decision 2. Fix the preset declarations in the karume dist recipe (tools/export-recipes) " +
        "and rebuild.",
    );
  }
};

test.skipIf(ALLOW_NO_DISTRIBUTION)(
  "distribution gate: all SKIP with distribution mirrors missing is an explicit FAIL (ADR 0005)",
  () => assertNoDistributionMissing(MODELS_ROOT, DISTRIBUTIONS),
);

for (const series of discoverDistributions(MODELS_ROOT)) {
  const root = distributionRoot(MODELS_ROOT, series);
  test.skipIf(REQUIRED.has(series) && manifestKnownAbsent(root))(
    `distribution gate: ${series} parses as karume/5 and every declared container is intact (parts, documents, graph name)`,
    () => assertDistributionIntact(root, series),
  );
  test.skipIf(REQUIRED.has(series) && manifestKnownAbsent(root))(
    `distribution gate: the default preset and preset list of ${series} satisfy the constraints of ADR 0128 decision 2 (C1–C3)`,
    async () => {
      const manifest = parseManifest(await readTextFile(new URL("karume.json", root)));
      assertDefaultPresetPolicy(manifest, series);
    },
  );
}

// What follows is fault injection into the guard's verdict itself. It is self-contained in a
// temporary directory (no real assets, no GPU), so it runs whether or not the opt-out is set — a
// regression in the verdict is visible even on CI where the guard is ignored.

/** Calls `run` with a temporary directory as the root and cleans up. */
const withTemporaryRoot = async (run: (root: URL) => Promise<void> | void): Promise<void> => {
  const root = new URL(`file://${makeTempDirSync({ prefix: "karume-distribution-" })}/`);
  try {
    await run(root);
  } finally {
    removeSync(root, { recursive: true });
  }
};

describe("distribution gate verdict (are the karume.json files of the listed mirrors all present)", () => {
  /** Builds a root with a `karume.json` (any content) in every listed mirror. */
  const arrangeAll = (models: URL): void => {
    for (const series of DISTRIBUTIONS) {
      const root = distributionRoot(models, series);
      mkdirSync(root, { recursive: true });
      writeTextFileSync(new URL("karume.json", root), "{}");
    }
  };

  it("passes when every listed mirror is present", async () => {
    await withTemporaryRoot((models) => {
      arrangeAll(models);
      assertNoDistributionMissing(models, DISTRIBUTIONS);
    });
  });

  it("whichever listed mirror is missing, it fails naming that mirror", async () => {
    await withTemporaryRoot((models) => {
      arrangeAll(models);
      for (const series of DISTRIBUTIONS) {
        const manifest = new URL("karume.json", distributionRoot(models, series));
        removeSync(manifest);
        const error = assertThrows(
          () => assertNoDistributionMissing(models, DISTRIBUTIONS),
          AssertionError,
        );
        assertStringIncludes(error.message, `karume.json of distribution mirrors: ${series}. `);
        writeTextFileSync(manifest, "{}");
      }
    });
  });

  it("treats the mirror as absent when karume.json is a directory", async () => {
    await withTemporaryRoot((models) => {
      arrangeAll(models);
      const manifest = new URL("karume.json", distributionRoot(models, "karume-lucida"));
      removeSync(manifest);
      mkdirSync(manifest);
      const error = assertThrows(
        () => assertNoDistributionMissing(models, DISTRIBUTIONS),
        AssertionError,
      );
      assertStringIncludes(error.message, "karume.json of distribution mirrors: karume-lucida. ");
    });
  });
});

describe("reach of the distribution gate (the mirror list derived from the layout of models/)", () => {
  it("picks up every child directory in name order and skips dist working directories and files", async () => {
    await withTemporaryRoot((models) => {
      for (const name of ["karume-b", "karume-a", "karume-a.staging", "karume-b.old"]) {
        mkdirSync(new URL(`${name}/`, models));
      }
      writeTextFileSync(new URL("README.md", models), "");
      assertEquals(discoverDistributions(models), ["karume-a", "karume-b"]);
    });
  });

  it("picks up a new mirror that is not on the roster (a forgotten roster update does not drop it from the manifest checks)", async () => {
    await withTemporaryRoot((models) => {
      const unlisted = "karume-vowel-detector";
      assert(!REQUIRED.has(unlisted), "test precondition: a name that is not on the roster");
      mkdirSync(new URL(`${unlisted}/`, models));
      assertEquals(discoverDistributions(models), [unlisted]);
    });
  });

  it("picks up a symbolic link to a directory as a mirror too", async () => {
    await withTemporaryRoot((root) => {
      const models = new URL("models/", root);
      const target = new URL("elsewhere/", root);
      mkdirSync(models);
      mkdirSync(target);
      symlinkSync(target.pathname, new URL("karume-linked", models).pathname);
      assertEquals(discoverDistributions(models), ["karume-linked"]);
    });
  });

  it("picks up broken links and links that cannot be followed without throwing (the file does not fail at registration and vanish)", async () => {
    await withTemporaryRoot((models) => {
      symlinkSync(new URL("nowhere", models).pathname, new URL("karume-broken", models).pathname);
      symlinkSync("karume-loop", new URL("karume-loop", models).pathname);
      assertEquals(discoverDistributions(models), ["karume-broken", "karume-loop"]);
    });
  });

  it("a mirror that is a broken link fails naming that directory", async () => {
    await withTemporaryRoot(async (models) => {
      symlinkSync(new URL("nowhere", models).pathname, new URL("karume-broken", models).pathname);
      await assertRejects(
        () => assertDistributionIntact(distributionRoot(models, "karume-broken"), "karume-broken"),
        AssertionError,
        "models/karume-broken/ has no karume.json",
      );
    });
  });

  it("a link that cannot be followed does not throw in the ignore decision, lets the test run, and throws naming the path inside the test", async () => {
    await withTemporaryRoot(async (models) => {
      symlinkSync("karume-loop", new URL("karume-loop", models).pathname);
      const root = distributionRoot(models, "karume-loop");
      assertEquals(manifestKnownAbsent(root), false);
      const error = await assertRejects(
        () => assertDistributionIntact(root, "karume-loop"),
        Error,
        "karume-loop/karume.json",
      );
      assert(isFilesystemLoop(error), `expected ELOOP, got ${String(error)}`);
    });
  });

  it("returns empty when models/ itself is missing (the roster check names the absence and fails)", async () => {
    await withTemporaryRoot((root) => {
      assertEquals(discoverDistributions(new URL("missing/", root)), []);
    });
  });

  it("a mirror without karume.json fails naming that directory", async () => {
    await withTemporaryRoot(async (root) => {
      await assertRejects(
        () => assertDistributionIntact(root, "karume-half-built"),
        AssertionError,
        "models/karume-half-built/ has no karume.json",
      );
    });
  });
});

describe("distribution gate verdict (every declared container against the mirror's files)", () => {
  const revision = "0123456789abcdef0123456789abcdef01234567";

  /**
   * One real container: the graph `graph` (`linear` with one f32 weight), split into small parts.
   */
  const writeContainer = (graph: string): Promise<WrittenContainer> =>
    writeModelContainer(
      {
        graphs: {
          [graph]: parseIrDeclarationValue({
            format: "karume-ir",
            version: 2,
            requires: { ops: ["linear"] },
            symbols: [],
            inputs: [{ name: "x", dtype: "f32", shape: [2, 2] }],
            outputs: ["y"],
            initializers: { w: {} },
            values: { w: { dtype: "f32", shape: [2, 2] }, y: { dtype: "f32", shape: [2, 2] } },
            states: {},
            nodes: [{ op: "linear", ins: ["x", "w"], outs: ["y"], attrs: {} }],
          }),
        },
        consts: [],
        weights: [
          {
            graph,
            initializer: "w",
            bytes: new Uint8Array(Float32Array.from([1, 2, 3, 4]).buffer),
            encoding: { codec: "f32" },
          },
        ],
        assets: [],
        provenance: { license: "test" },
      },
      { partBytes: 4096, blockBytes: 512 },
    );

  const hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
    [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");

  /** A container written to disk under `stem` and the manifest entry that declares it. */
  type Placed = {
    readonly paths: readonly string[];
    readonly entry: (options?: { readonly repo?: string }) => Record<string, unknown>;
  };

  /** Writes `written` under `root` as `<stem>-NNNNN-of-NNNNN.krm` (zero-length parts too). */
  const place = async (root: URL, stem: string, written: WrittenContainer): Promise<Placed> => {
    const total = String(written.parts.length).padStart(5, "0");
    const paths = written.parts.map(
      (_, index) => `${stem}-${String(index + 1).padStart(5, "0")}-of-${total}.krm`,
    );
    mkdirSync(new URL(stem.slice(0, stem.lastIndexOf("/") + 1), root), { recursive: true });
    const parts = await Promise.all(
      written.parts.map(async (bytes, index) => {
        writeFileSync(new URL(paths[index]!, root), bytes);
        return { path: paths[index]!, size: bytes.byteLength, sha256: await hex(bytes) };
      }),
    );
    const descriptor = {
      graph: {
        length: written.graphDescriptorBytes.byteLength,
        sha256: await hex(written.graphDescriptorBytes),
      },
      model: {
        length: written.modelDescriptorBytes.byteLength,
        sha256: await hex(written.modelDescriptorBytes),
      },
    };
    return {
      paths,
      entry: ({ repo } = {}) => ({
        container: {
          descriptor,
          parts: repo === undefined ? parts : parts.map((part) => ({ ...part, repo, revision })),
        },
      }),
    };
  };

  /** The manifest of one model `m` whose component `depth` has the quants f32 (default) and i8. */
  const manifestText = (
    f32: Record<string, unknown>,
    i8: Record<string, unknown>,
    format = MANIFEST_FORMAT,
  ): string =>
    JSON.stringify({
      format,
      generator: "karume/0.15.1",
      defaultModel: "m",
      models: {
        m: {
          pipeline: "depth-anything/1",
          weights: { depth: { f32, i8 } },
          assets: {},
          quants: {
            f32: { weights: { depth: "f32" }, session: {}, label: "F32", description: "Full." },
            i8: { weights: { depth: "i8" }, session: {}, label: "I8", description: "Small." },
          },
          defaultQuant: "f32",
          pipelineConfig: {},
        },
      },
    });

  /** An intact mirror; the i8 container's graph is `i8Graph` (the weights key when omitted). */
  const arrange = async (
    root: URL,
    options: { readonly format?: string; readonly i8Graph?: string } = {},
  ): Promise<{ readonly f32: Placed; readonly i8: Placed }> => {
    const f32 = await place(root, "m/depth/model.f32", await writeContainer("depth"));
    const i8 = await place(
      root,
      "m/depth/model.i8",
      await writeContainer(options.i8Graph ?? "depth"),
    );
    writeTextFileSync(
      new URL("karume.json", root),
      manifestText(f32.entry(), i8.entry(), options.format),
    );
    return { f32, i8 };
  };

  it("an intact mirror passes", async () => {
    await withTemporaryRoot(async (root) => {
      await arrange(root);
      await assertDistributionIntact(root, "synthetic");
    });
  });

  it("a part of the default quant is absent: FAIL naming the container and the part", async () => {
    await withTemporaryRoot(async (root) => {
      const { f32 } = await arrange(root);
      const last = f32.paths.length - 1;
      removeSync(new URL(f32.paths[last]!, root));
      await assertRejects(
        () => assertDistributionIntact(root, "synthetic"),
        AssertionError,
        `m/depth/f32[${last}] ${f32.paths[last]}: the declared part is absent`,
      );
    });
  });

  it("a part of a non-default quant is absent: FAIL too (every quant is checked, not only the default)", async () => {
    await withTemporaryRoot(async (root) => {
      const { i8 } = await arrange(root);
      const last = i8.paths.length - 1;
      removeSync(new URL(i8.paths[last]!, root));
      await assertRejects(
        () => assertDistributionIntact(root, "synthetic"),
        AssertionError,
        `m/depth/i8[${last}] ${i8.paths[last]}: the declared part is absent`,
      );
    });
  });

  it("a part's length differs from its declaration: FAIL with both lengths", async () => {
    await withTemporaryRoot(async (root) => {
      const { f32 } = await arrange(root);
      const last = f32.paths.length - 1;
      const at = new URL(f32.paths[last]!, root);
      const size = statSync(at).size;
      writeFileSync(at, new Uint8Array(size - 1));
      await assertRejects(
        () => assertDistributionIntact(root, "synthetic"),
        AssertionError,
        `m/depth/f32[${last}] ${f32.paths[last]}: ${size - 1} bytes, declared ${size}`,
      );
    });
  });

  it("the container's graph is not named by the weights key: FAIL naming both", async () => {
    await withTemporaryRoot(async (root) => {
      await arrange(root, { i8Graph: "other" });
      await assertRejects(
        () => assertDistributionIntact(root, "synthetic"),
        AssertionError,
        "m/depth/i8: the container's graphs are other, expected depth",
      );
    });
  });

  it("part 0's documents differ from the declared sha256: FAIL, the container does not open", async () => {
    await withTemporaryRoot(async (root) => {
      const { f32 } = await arrange(root);
      // Another container of the same shape (a graph name of the same length) over the declared
      // part 0: the part lengths agree, the documents' sha256 does not.
      const other = await writeContainer("deptx");
      writeFileSync(new URL(f32.paths[0]!, root), other.parts[0]!);
      await assertRejects(
        () => assertDistributionIntact(root, "synthetic"),
        AssertionError,
        "m/depth/f32: the container does not open",
      );
    });
  });

  it("the manifest is not karume/5: FAIL", async () => {
    await withTemporaryRoot(async (root) => {
      await arrange(root, { format: "karume/4" });
      await assertRejects(() => assertDistributionIntact(root, "synthetic"), Error, "karume/4");
    });
  });

  it("a cross-repo container is opened from its sibling mirror (its parts are not stat'ed here)", async () => {
    await withTemporaryRoot(async (models) => {
      const base = new URL("karume-base/", models);
      const derived = new URL("karume-derived/", models);
      mkdirSync(base);
      mkdirSync(derived);
      const { f32, i8 } = await arrange(base);
      const repo = "hdae/karume-base";
      writeTextFileSync(
        new URL("karume.json", derived),
        manifestText(f32.entry({ repo }), i8.entry({ repo })),
      );
      await assertDistributionIntact(derived, "karume-derived");
      removeSync(base, { recursive: true });
      await assertRejects(
        () => assertDistributionIntact(derived, "karume-derived"),
        Error,
        "'hdae/karume-base'",
      );
    });
  });
});

describe("distribution gate verdict (default preset and preset list — ADR 0128 decision 2)", () => {
  const sha256 = "0".repeat(64);
  /**
   * A preset declaration written into the manifest (kept as JSON — type and value checks are left
   * to hub's parser).
   */
  type FabricatedQuant = {
    readonly weights: Readonly<Record<string, string>>;
    readonly [key: string]: unknown;
  };
  const labelled = { label: "Preset", description: "A preset for the test." };
  /**
   * An i8 preset that runs on plain WebGPU with no activation quantization (the template for a C2
   * sibling).
   */
  const plainI8 = { weights: { text: "i8", dit: "i8" }, session: {}, ...labelled };
  /** A preset with int8 activations on i8 weights. */
  const a8I8 = {
    weights: { text: "i8", dit: "i8" },
    session: { linearCompute: "a8", attentionCompute: "a8" },
    ...labelled,
  };

  /**
   * Runs a one-model manifest declaring `quants` through hub's parser and returns that model's
   * declaration (the weights containers are built from the components × labels the quants refer
   * to — so that it passes the parser's complete-mapping check).
   */
  const entryOf = (
    quants: Readonly<Record<string, FabricatedQuant>>,
    defaultQuant: string,
  ): ModelEntry => {
    const weights: Record<string, Record<string, unknown>> = {};
    for (const quant of Object.values(quants)) {
      for (const [role, label] of Object.entries(quant.weights)) {
        weights[role] = {
          ...weights[role],
          [label]: {
            container: {
              descriptor: { graph: { length: 10, sha256 }, model: { length: 5, sha256 } },
              parts: [
                { path: `m/${role}/model.${label}-00001-of-00002.krm`, size: 39, sha256 },
                { path: `m/${role}/model.${label}-00002-of-00002.krm`, size: 8, sha256 },
              ],
            },
          },
        };
      }
    }
    const manifest = parseManifest(
      JSON.stringify({
        format: MANIFEST_FORMAT,
        generator: "karume/0.15.0",
        defaultModel: "m",
        models: {
          m: {
            pipeline: "anima/1",
            weights,
            assets: {},
            quants,
            defaultQuant,
            pipelineConfig: {},
          },
        },
      }),
    );
    return manifest.models["m"]!;
  };

  /**
   * Asserts exactly one violation whose sentence names the preset and contains the constraint's
   * label.
   */
  const assertSingleViolation = (entry: ModelEntry, constraint: string, preset: string): void => {
    const violations = defaultPresetViolations(entry);
    assertEquals(violations.length, 1, violations.join("\n"));
    assertStringIncludes(violations[0]!, `${constraint}:`);
    assertStringIncludes(violations[0]!, `'${preset}'`);
  };

  it("two presets with label and description, with a sibling for a8, have no violation", () => {
    assertEquals(defaultPresetViolations(entryOf({ a8: a8I8, i8: plainI8 }, "a8")), []);
  });

  it("a default that requests shaderF16 is a C1 violation naming the default preset", () => {
    const f16 = {
      weights: { text: "f16", dit: "f16" },
      session: { linearCompute: "f16" },
      gpuFeatures: { shaderF16: true },
      ...labelled,
    };
    const entry = entryOf({ f16, i8: plainI8 }, "f16");
    assertSingleViolation(entry, "C1", "f16");
    assertStringIncludes(defaultPresetViolations(entry)[0]!, "shaderF16");
  });

  it("shaderF16 on a non-default preset is not a C1 violation", () => {
    const f16 = {
      weights: { text: "f16", dit: "f16" },
      session: {},
      gpuFeatures: { shaderF16: true },
      ...labelled,
    };
    assertEquals(defaultPresetViolations(entryOf({ f16, i8: plainI8 }, "i8")), []);
  });

  it("an a8 preset without a sibling is a C2 violation naming that preset and its a8 knobs", () => {
    const f32 = { weights: { text: "f32", dit: "f32" }, session: {}, ...labelled };
    const entry = entryOf({ a8: a8I8, f32 }, "a8");
    assertSingleViolation(entry, "C2", "a8");
    const [violation] = defaultPresetViolations(entry);
    assertStringIncludes(violation!, `linearCompute: "a8"`);
    assertStringIncludes(violation!, `attentionCompute: "a8"`);
  });

  it("if the would-be sibling's weights differ in even one label, C2 stays violated (an extra download is needed)", () => {
    const nearly = { weights: { text: "f32", dit: "i8" }, session: {}, ...labelled };
    assertSingleViolation(entryOf({ a8: a8I8, nearly }, "a8"), "C2", "a8");
  });

  it("even with the same weights, C2 stays violated if the would-be sibling keeps one of the a8 knobs", () => {
    const attentionOnly = { ...a8I8, session: { attentionCompute: "a8" } };
    const entry = entryOf({ a8: a8I8, attentionOnly }, "a8");
    const violations = defaultPresetViolations(entry);
    assertEquals(violations.length, 2, violations.join("\n"));
    assertStringIncludes(violations[0]!, "C2: preset 'a8'");
    assertStringIncludes(violations[1]!, "C2: preset 'attentionOnly'");
  });

  it("a preset whose only a8 knob is convCompute is a C2 violation without a sibling, naming the knob", () => {
    const conv8 = { ...plainI8, session: { convCompute: "a8" } };
    const f32 = { weights: { text: "f32", dit: "f32" }, session: {}, ...labelled };
    const entry = entryOf({ conv8, f32 }, "f32");
    assertSingleViolation(entry, "C2", "conv8");
    assertStringIncludes(defaultPresetViolations(entry)[0]!, `convCompute: "a8"`);
  });

  it("a preset whose only a8 knob is convCompute passes C2 with a same-weights sibling without a8", () => {
    const conv8 = { ...plainI8, session: { convCompute: "a8" } };
    assertEquals(defaultPresetViolations(entryOf({ conv8, i8: plainI8 }, "i8")), []);
  });

  it('a would-be sibling that keeps convCompute: "a8" is not a sibling (C2 stays violated)', () => {
    const keepsConv8 = { ...plainI8, session: { convCompute: "a8" } };
    const entry = entryOf({ a8: a8I8, keepsConv8 }, "a8");
    const violations = defaultPresetViolations(entry);
    assertEquals(violations.length, 2, violations.join("\n"));
    assertStringIncludes(violations[0]!, "C2: preset 'a8'");
    assertStringIncludes(violations[1]!, "C2: preset 'keepsConv8'");
  });

  it("a preset without description is a C3 violation naming that preset", () => {
    const withoutDescription = { weights: plainI8.weights, session: {}, label: "Preset" };
    const entry = entryOf({ a8: a8I8, i8: withoutDescription }, "a8");
    assertSingleViolation(entry, "C3", "i8");
    assertStringIncludes(defaultPresetViolations(entry)[0]!, "description");
  });

  it("a whitespace-only label counts as absent and is a C3 violation naming that preset", () => {
    const entry = entryOf({ a8: { ...a8I8, label: " \t " }, i8: plainI8 }, "a8");
    assertSingleViolation(entry, "C3", "a8");
    assertStringIncludes(defaultPresetViolations(entry)[0]!, "label");
  });

  it("a model with a single preset is outside C3 even without label and description", () => {
    const f32 = { weights: { text: "f32", dit: "f32" }, session: {} };
    assertEquals(defaultPresetViolations(entryOf({ f32 }, "f32")), []);
  });
});
