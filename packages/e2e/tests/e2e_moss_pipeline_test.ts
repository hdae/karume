/**
 * MOSS-Image through its public pipeline over the local distribution mirror — `moss.Pipeline`
 * (`@karume/models/moss`) built by `fromPretrained` from `models/karume-moss-image/` (the recipe
 * `tools/export-recipes/moss/distribution.py`; ADR 0138 addendum of 2026-11-01).
 *
 * ## Checks
 *
 * 1. **The mirror's manifest resolves** (no GPU): without `quant` to the practical preset
 *    `f32+dit8-a8-attn8-online` (the manifest's `defaultQuant`), which declares no GPU feature (ADR
 *    0128 C1), with `quant: "f32+dit8"` to the reference preset; `pipelineConfig` is upstream's
 *    request defaults (1024², 50 steps, guidance 3, shift 5, the empty negative prompt).
 * 2. **The pipeline gives the series path's image bit for bit** (GPU): 256² × 2 steps, the prompt
 *    `fridge-note` (its text read from upstream's text-embedding asset — not spelled here), seed 42,
 *    `quant: "f32+dit8"`, and guidance, shift and the negative prompt **left to the manifest**. The
 *    image's sha256 MUST equal the existing row
 *    `f32+dit8-2step-fridge-note-qwen3_5-i8-seed42-256x256-g3-shift5` of `moss.json`, which
 *    `e2e_moss_prompt_test.ts` wrote from the series with every knob explicit (guidance 3, shift 5,
 *    `negativePrompt` ""): the same graphs and weights (the mirror's parts are the series' bytes),
 *    so the pipeline's defaults, its preset and its assembly add nothing. This file **never writes
 *    or rewrites a row** — on an environment without that row the case skips explicitly, and the
 *    prompt test's reference gate is red there.
 *
 * Faults this catches (injected 2026-11-01, each turned case 2 red): the default quant used in
 * place of `quant` (the practical preset's image), a default guidance other than the manifest's.
 *
 * Without a GPU, the mirror or the text-embedding asset the file skips explicitly, naming the
 * command that gets it (ADR 0005). The text-embedding asset is a golden of the test-asset root
 * (ADR 0139 — `goldenDir("moss-image-sft-text-embeds")`, fetched with
 * `node tools/test-assets/fetch.ts`).
 */

import { afterAll, beforeAll, describe, expect, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";
import type { GpuContext } from "@karume/runtime";
import { Pipeline } from "../../models/moss.ts";
import { mossSelectionOf } from "../../models/src/moss/pipeline.ts";
import { MOSS_PRACTICAL_PRESET, MOSS_REFERENCE_PRESET } from "../../models/src/moss/preset.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  MOSS_FETCH_GOLDENS,
  MOSS_TEXT_EMBEDS,
  testAssetsRunnable,
} from "../../models/tests/helpers/moss-dit.ts";
import { readMossUpstreamPrompts } from "../../models/tests/helpers/moss-text.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import { verifyRunningGpu } from "../../runtime/tests/helpers/environment.ts";
import {
  f32ArtifactBytes,
  openReferences,
  sha256Hex,
} from "../../runtime/tests/helpers/reference.ts";

/** The local mirror (`dist.py --pipeline moss-image` builds it — docs/assets-layout.md). */
const DIST_ROOT = new URL("../../../models/karume-moss-image/", import.meta.url);
const ASSEMBLE = "cd tools/export-recipes && uv run python dist.py --pipeline moss-image";

const MANIFEST_TEXT = await readTextIfPresent(new URL("karume.json", DIST_ROOT));
const MANIFEST = MANIFEST_TEXT === undefined ? undefined : parseManifest(MANIFEST_TEXT);
/** Present, or absent from a complete fetch of the pin (then its read FAILs naming it). */
const EMBEDS_RUNNABLE = testAssetsRunnable([MOSS_TEXT_EMBEDS]);

/** The rows of `moss.json` — read only (this file never writes one). */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/moss.json", import.meta.url),
  { mode: undefined },
);

/** The generation of check 2 (the prompt test's seed path — its row ID names these values). */
const PROMPT = "fridge-note";
const SIZE = { width: 256, height: 256 } as const;
const STEPS = 2;
const SEED = 42;
const CASE_ID = `${MOSS_REFERENCE_PRESET}-${STEPS}step-${PROMPT}-qwen3_5-i8-seed${SEED}-256x256-g3-shift5`;
const EXPECTED = references.lookup(CASE_ID);

const skipped = (ok: boolean, what: string, command: string): string =>
  ok ? "" : ` (SKIP: no ${what} — ${command})`;
const MIRROR_SKIP = skipped(MANIFEST !== undefined, "distribution mirror", ASSEMBLE);

describe.skipIf(MANIFEST === undefined)(`MOSS-Image mirror manifest${MIRROR_SKIP}`, () => {
  test("resolves to the practical preset by default, the reference preset by name, with upstream's defaults", () => {
    assert(MANIFEST !== undefined);
    const practical = mossSelectionOf(MANIFEST, {});
    expect(practical.preset.name).toBe(MOSS_PRACTICAL_PRESET);
    expect(practical.gpuFeatures).toBeUndefined();
    expect(practical.config).toEqual({
      scheduler: { shift: 5 },
      defaults: { width: 1024, height: 1024, steps: 50, guidance: 3, negativePrompt: "" },
    });
    expect(mossSelectionOf(MANIFEST, { quant: MOSS_REFERENCE_PRESET }).preset.name).toBe(
      MOSS_REFERENCE_PRESET,
    );
  });
});

const RUNNABLE = GPU_AVAILABLE && MANIFEST !== undefined && EMBEDS_RUNNABLE;

describe.skipIf(!RUNNABLE || EXPECTED === undefined)(
  `MOSS-Image pipeline from the mirror on the GPU${skipped(GPU_AVAILABLE, "GPU adapter", "a WebGPU device")}${MIRROR_SKIP}${skipped(EMBEDS_RUNNABLE, `upstream text embeddings ${MOSS_TEXT_EMBEDS.pathname}`, MOSS_FETCH_GOLDENS)}${skipped(EXPECTED !== undefined, `row ${CASE_ID} of this environment`, "run e2e_moss_prompt_test.ts with KARUME_REFERENCE=write")}`,
  () => {
    let gpu: GpuContext | undefined;
    beforeAll(async () => {
      // The adapter the environment key names, acquired as the prompt test acquires it.
      gpu = verifyRunningGpu(await acquireTestGpu());
    });
    afterAll(async () => {
      if (gpu === undefined) return;
      await settleReleases(gpu);
      gpu.destroy();
    });

    it(`${CASE_ID}: the image of the series path, bit for bit, with the manifest's defaults`, async () => {
      assert(gpu !== undefined);
      const prompt = (await readMossUpstreamPrompts()).find(({ name }) => name === PROMPT);
      assert(prompt !== undefined, `upstream's asset has no prompt '${PROMPT}'`);
      const started = performance.now();
      await using pipeline = await Pipeline.fromPretrained(nodeDirectory(DIST_ROOT.pathname), {
        quant: MOSS_REFERENCE_PRESET,
        gpu,
      });
      const built = performance.now();
      const image = await pipeline.generate({
        prompt: prompt.text,
        ...SIZE,
        steps: STEPS,
        seed: SEED,
      });
      const done = performance.now();
      console.info(
        `[moss-pipeline] ${CASE_ID}: fromPretrained ${((built - started) / 1000).toFixed(1)} s,` +
          ` generate ${((done - built) / 1000).toFixed(1)} s`,
      );
      assertEquals([image.width, image.height], [SIZE.width, SIZE.height]);
      expect(image.data.length).toBe(3 * SIZE.width * SIZE.height);
      const actual = await sha256Hex(
        f32ArtifactBytes("image", [3, SIZE.height, SIZE.width], image.data),
      );
      expect(actual).toBe(EXPECTED);
    });
  },
);
