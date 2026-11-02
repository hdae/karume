/**
 * The Qwen3.5 text encoder on the real GPU (ADR 0137 decision 6): `Qwen3_5TextEncoder`
 * (`packages/models/src/qwen3_5/encoder.ts`) over the distribution's i8 text encoder, against
 * Karume's CPU reference execution of the same container.
 *
 * ## Assets (outside git — ADR 0139)
 *
 * - The container `text_encoder` of the distribution mirror `models/karume-moss-image/` (model
 *   `sft`, 34 parts, 7.9 GB), read through its manifest — a byte copy of the series
 *   `qwen3.5-9b-text-moss-image-sft-i8-dyn` written by {@link GENERATE_CONTAINER}, whose name the
 *   sha256 case IDs keep verbatim (ADR 0139 decision 7).
 * - The CPU reference, the golden directory `qwen3_5-export-2026-10-31` of the test-asset root
 *   (`goldenDir`, fetched with `node tools/test-assets/fetch.ts`): `ir.full-i8.<case>.f32` (the f32
 *   output `[1, T, 4096]`) and `inputs.full.<case>.safetensors` (the four graph inputs it was computed
 *   from — the recipe's `inputs_for` over the ids of MOSS-Image's prompt asset), written by
 *   {@link GENERATE_REFERENCE}: Karume's TypeScript reference executor walks the container's graph op
 *   by op on the CPU (`applyReferenceOp` of `packages/runtime/src/reference/ops.ts`), decoding the
 *   same i8 weights. It is the kept oracle of ADR 0129 (Karume's own reference, not a torch golden),
 *   so these checks run by default; the export's distance from upstream is the recipe's record
 *   (`full.json` there, ≈ 3.6–4.4 % rel. L2 for i8 against f32 upstream).
 *
 * ## Checks
 *
 * 1. **Assets complete**: every reference file — a FAIL when one is absent from a complete fetch of
 *    the test-asset dataset at the pin (`testAssetsFetched`) or from a partial set; skipped when none
 *    is there and the dataset was not fetched. The mirror's absence SKIPs checks 2–4 instead (the
 *    distribution gate owns that red).
 * 2. **Host, no GPU**: the inputs the encoder builds from the recorded ids are bit-identical to the
 *    inputs the CPU reference was fed (positions and both RoPE rows), the references were computed
 *    from this container (part 0's sha256), and the graph contract holds on the real declaration.
 * 3. **GPU vs CPU reference**: each case's rows sit within {@link GPU_REL_RMS_BAND}, then the rows'
 *    sha256 is checked against this environment's row (ADR 0106 — the band says "close to the
 *    oracle", the row says "not one bit moved on this machine").
 * 4. **The band is live**: with `rope_cos` / `rope_sin` swapped on the way into the same Session (a
 *    host-input fault no shape check sees), the rows leave the band (measured: by three orders of
 *    magnitude — {@link GPU_REL_RMS_BAND}).
 *
 * No GPU, or no assets: explicit SKIP naming what is missing and how to get it (ADR 0005) — except
 * a CPU reference absent from a complete fetch of the pinned dataset, which FAILs (check 1).
 */

import { isNotFound, readFile, statSync } from "@karume/test-support/fs";
import { goldenDir, testAssetsFetched } from "@karume/test-support/test-assets";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, fail } from "@std/assert";
import {
  type GpuContext,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
  type RunInputs,
} from "@karume/runtime";
import { type Qwen3_5TextComponent, Qwen3_5TextEncoder } from "../../models/src/qwen3_5/encoder.ts";
import { qwen3_5GraphContract } from "../../models/src/qwen3_5/graph-contract.ts";
import {
  QWEN3_5_INPUT_IDS,
  QWEN3_5_POSITIONS,
  QWEN3_5_ROPE_COS,
  QWEN3_5_ROPE_SIN,
  qwen3_5SessionInputs,
} from "../../models/src/qwen3_5/session-io.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import {
  assertTestAssetsPresent,
  MOSS_FETCH_GOLDENS,
  MOSS_GET_MIRROR,
  MOSS_MIRROR,
  mossMirrorPresent,
  openMossMirrorContainer,
  part0Digest,
  testAssetsRunnable,
} from "../../models/tests/helpers/moss-dit.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import {
  f32ArtifactBytes,
  openReferences,
  type ReferenceSettlement,
  referenceMismatchMessage,
  registerReferenceGate,
  settleOrObserve,
} from "../../runtime/tests/helpers/reference.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";

/**
 * The series the container was exported as. MUST stay verbatim: the case IDs spell it (ADR 0139
 * decision 7).
 */
const SERIES_NAME = "qwen3.5-9b-text-moss-image-sft-i8-dyn";
/** The component = the graph name = the manifest's weights key. */
const COMPONENT = "text_encoder";
const REFERENCE_DIR = goldenDir("qwen3_5-export-2026-10-31");

const GENERATE_CONTAINER =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' --inexact python -m qwen3_5.export write --storage i8";
const GENERATE_REFERENCE =
  `${GENERATE_CONTAINER}, then recompute the CPU reference of that series on the exporting machine` +
  " (Karume's TypeScript reference executor over the container — about 12 minutes;" +
  " docs/research/2026-10-31-qwen3_5-encoder-gpu.md §1), then upload the goldens" +
  ' (tools/test-assets/README.md "Upload the assets of a new model")';

/**
 * The sha256 of part 0 (header + both descriptors, which declare every block's sha256) of the
 * container the CPU reference was computed from (the i8 export of 2026-10-31 09:02 UTC). A re-export
 * that moves any byte fails check 2 instead of failing the band against stale rows.
 */
const REFERENCE_PART0_SHA256 = "e9b581dc711758547dc1f89bed26f07388b927a7b410370ce338337f1fc260ff";

/** The recorded prompts (MOSS-Image's prompt asset `moss-image-sft-text-embeds`). */
const CASES = [
  /** The empty prompt — upstream's negative: the chat template alone. */
  { name: "negative", tokens: 10 },
  /** The README's quick-start prompt. */
  { name: "fridge-note", tokens: 63 },
] as const;

type Case = (typeof CASES)[number];

/**
 * The accepted relative RMS error (relRms) `‖GPU − CPU‖₂ / ‖CPU‖₂` over all rows of one encoding.
 *
 * Derivation (RTX 3080 Ti, Dawn / Vulkan, 2026-10-31 — research 2026-10-31-qwen3_5-encoder-gpu): the
 * measured errors were 5.09e-6 (`negative`, T = 10) and 3.62e-6 (`fridge-note`, T = 63); the worst
 * × 5, rounded up to two significant figures, is 2.6e-5 (the rule of umT5's band, ADR 0119 decision
 * 8). Faults measured against the same references (negative / fridge-note): one linear's per-channel
 * scale × 1.01 (layer 15's MLP `down_proj`) 2.87e-3 / 1.70e-3 — the smallest, 65× the band; that
 * scale × 0 (the layer's MLP output zeroed) 0.27 / 0.16; a GDN layer's `out_proj` scale × 0 (layer
 * 14) 0.15 / 0.16; swapped RoPE rows 0.080 / 0.067.
 */
const GPU_REL_RMS_BAND = 2.6e-5;

const inputsUrl = (entry: Case): URL =>
  new URL(`inputs.full.${entry.name}.safetensors`, REFERENCE_DIR);
const referenceUrl = (entry: Case): URL => new URL(`ir.full-i8.${entry.name}.f32`, REFERENCE_DIR);

/** File presence (NotFound is absence; any other I/O failure propagates — not a silent SKIP). */
const filePresent = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const MODEL_PRESENT = mossMirrorPresent();
const REFERENCE_FILES = CASES.flatMap((entry) => [inputsUrl(entry), referenceUrl(entry)]);
/** Present, or absent from a complete fetch of the pin — then checks 1–4 FAIL naming the files. */
const REFERENCES_RUNNABLE = testAssetsRunnable(REFERENCE_FILES);
const ASSETS_AVAILABLE = MODEL_PRESENT && REFERENCES_RUNNABLE;
/** Check 1 runs on a fetched root and on a partial set (which it fails). */
const REFERENCES_CHECKED = REFERENCES_RUNNABLE || REFERENCE_FILES.some(filePresent);
const RUNNABLE = ASSETS_AVAILABLE && GPU_AVAILABLE;

const skipped = (ok: boolean, what: string, command: string): string =>
  ok ? "" : ` (SKIP: no ${what} — ${command})`;
const MIRROR_SKIP = skipped(MODEL_PRESENT, `mirror ${MOSS_MIRROR.pathname}`, MOSS_GET_MIRROR);
const REFERENCES_SKIP = skipped(
  REFERENCES_RUNNABLE,
  `CPU reference in ${REFERENCE_DIR.pathname}`,
  MOSS_FETCH_GOLDENS,
);

if (!MODEL_PRESENT) {
  console.warn(
    `[karume] the mirror ${MOSS_MIRROR.pathname} is absent, so the Qwen3.5 text encoder e2e is ` +
      `SKIPPED (7.9 GB, outside git). Get it: ${MOSS_GET_MIRROR}`,
  );
} else if (!REFERENCES_RUNNABLE) {
  console.warn(
    `[karume] ${REFERENCE_DIR.pathname} lacks the CPU reference, so the Qwen3.5 text encoder e2e is ` +
      `SKIPPED. Fetch: ${MOSS_FETCH_GOLDENS}`,
  );
}

/** One recorded case: the four inputs the CPU reference was fed and its output rows. */
type Recorded = {
  readonly inputs: Readonly<
    Record<string, { readonly shape: readonly number[]; readonly bytes: Uint8Array<ArrayBuffer> }>
  >;
  readonly ids: Int32Array<ArrayBuffer>;
  readonly reference: Float32Array<ArrayBuffer>;
};

const loadRecorded = async (entry: Case): Promise<Recorded> => {
  const file = await readFile(inputsUrl(entry));
  const parsed = parseSafetensors(file.buffer);
  const inputs: Record<
    string,
    { readonly shape: readonly number[]; readonly bytes: Uint8Array<ArrayBuffer> }
  > = {};
  for (const name of [QWEN3_5_INPUT_IDS, QWEN3_5_POSITIONS, QWEN3_5_ROPE_COS, QWEN3_5_ROPE_SIN]) {
    const view = parsed.tensors.get(`input.${name}`);
    assert(view !== undefined, `${entry.name}: the recorded inputs lack input.${name}`);
    inputs[name] = {
      shape: view.shape,
      bytes: file.slice(view.byteOffset, view.byteOffset + view.byteLength),
    };
  }
  assertEquals(parsed.tensors.size, 4, `${entry.name}: the recorded inputs hold other tensors`);
  const ids = new Int32Array(inputs[QWEN3_5_INPUT_IDS].bytes.buffer);
  assertEquals(ids.length, entry.tokens, `${entry.name}: token count`);
  const reference = new Float32Array((await readFile(referenceUrl(entry))).buffer);
  return { inputs, ids, reference };
};

const bytesOf = (data: Float32Array<ArrayBuffer>): Uint8Array<ArrayBuffer> =>
  new Uint8Array(data.buffer, data.byteOffset, data.byteLength);

/** The first byte index where two byte strings differ (−1 = identical, including the length). */
const firstMismatch = (a: Uint8Array, b: Uint8Array): number => {
  const length = Math.min(a.length, b.length);
  for (let index = 0; index < length; index += 1) if (a[index] !== b[index]) return index;
  return a.length === b.length ? -1 : length;
};

/** Distance of the GPU rows from the CPU rows (f64 accumulation). */
type Distance = {
  readonly relRms: number;
  readonly maxAbs: number;
  /** The smallest cosine similarity of one GPU row with its CPU row. */
  readonly rowCosineMin: number;
};

const distance = (
  actual: Float32Array<ArrayBuffer>,
  expected: Float32Array<ArrayBuffer>,
  width: number,
): Distance => {
  assertEquals(actual.length, expected.length, "element count");
  let difference = 0;
  let norm = 0;
  let maxAbs = 0;
  let rowCosineMin = 1;
  for (let row = 0; row < actual.length / width; row += 1) {
    let dot = 0;
    let actualNorm = 0;
    let expectedNorm = 0;
    for (let column = 0; column < width; column += 1) {
      const a = actual[row * width + column];
      const e = expected[row * width + column];
      difference += (a - e) ** 2;
      norm += e * e;
      maxAbs = Math.max(maxAbs, Math.abs(a - e));
      dot += a * e;
      actualNorm += a * a;
      expectedNorm += e * e;
    }
    rowCosineMin = Math.min(rowCosineMin, dot / Math.sqrt(actualNorm * expectedNorm));
  }
  return { relRms: Math.sqrt(difference / norm), maxAbs, rowCosineMin };
};

const formatDistance = ({ relRms, maxAbs, rowCosineMin }: Distance): string =>
  `relRms ${relRms.toExponential(3)}, max |Δ| ${maxAbs.toExponential(3)}, ` +
  `min row cosine ${rowCosineMin.toFixed(9)}`;

/**
 * The prepared container as the encoder's component, with one switch for check 4: when
 * `fault.swapRope` is set, the Session receives `rope_cos` and `rope_sin` swapped. The weights and
 * the Session are the real ones (one upload serves every check).
 */
const faultableComponent = (
  prepared: PreparedModel,
  fault: { swapRope: boolean },
): Qwen3_5TextComponent => ({
  graph: prepared.graph,
  createSession: async (gpu, options) => {
    const session = await prepared.createContainerSession(gpu, options);
    const swapped = (inputs: RunInputs): RunInputs =>
      fault.swapRope
        ? {
            ...inputs,
            [QWEN3_5_ROPE_COS]: inputs[QWEN3_5_ROPE_SIN],
            [QWEN3_5_ROPE_SIN]: inputs[QWEN3_5_ROPE_COS],
          }
        : inputs;
    return {
      run: (inputs, bindings, generation) => session.run(swapped(inputs), bindings, generation),
      diagnostics: () => session.diagnostics(),
      dispose: () => session.dispose(),
    };
  },
});

const results = openResults("qwen3_5");
const references = openReferences(
  new URL("../../models/tests/fixtures/references/qwen3_5.json", import.meta.url),
);
const caseIdOf = (entry: Case): string => `${SERIES_NAME}/${entry.name}`;
const CASE_IDS = CASES.map(caseIdOf);
if (RUNNABLE) references.warnMissing(CASE_IDS);

test.skipIf(!REFERENCES_CHECKED)(
  `Qwen3.5 encoder assets: every CPU reference file is present${skipped(REFERENCES_CHECKED, `CPU reference in ${REFERENCE_DIR.pathname}`, MOSS_FETCH_GOLDENS)}`,
  () => {
    if (testAssetsFetched()) assertTestAssetsPresent(REFERENCE_FILES);
    assertEquals(
      REFERENCE_FILES.filter((url) => !filePresent(url)).map((url) => url.pathname),
      [],
      `missing CPU reference files (fetch: ${MOSS_FETCH_GOLDENS})`,
    );
  },
);

test.skipIf(!ASSETS_AVAILABLE)(
  "Qwen3.5 encoder host (no GPU): the encoder's inputs are bit-identical to the CPU reference's, " +
    `the reference comes from this container, and the graph contract holds${MIRROR_SKIP}${REFERENCES_SKIP}`,
  async () => {
    assertTestAssetsPresent(REFERENCE_FILES);
    const opened = await openMossMirrorContainer(COMPONENT);
    assertEquals(
      await part0Digest(opened),
      REFERENCE_PART0_SHA256,
      `the container is not the one the CPU reference was computed from — regenerate: ${GENERATE_REFERENCE}`,
    );
    const prepared = prepareContainer(opened, COMPONENT);
    const contract = qwen3_5GraphContract(prepared, "e2e");
    assertEquals(
      { hiddenSize: contract.hiddenSize, vocabularySize: contract.vocabularySize },
      { hiddenSize: 4096, vocabularySize: 248_320 },
    );
    for (const entry of CASES) {
      const recorded = await loadRecorded(entry);
      assertEquals(recorded.reference.length, entry.tokens * contract.hiddenSize, entry.name);
      const built = qwen3_5SessionInputs(recorded.ids, contract.vocabularySize);
      for (const name of [QWEN3_5_POSITIONS, QWEN3_5_ROPE_COS, QWEN3_5_ROPE_SIN] as const) {
        const tensor = built[name];
        assertEquals(tensor.shape, recorded.inputs[name].shape, `${entry.name} ${name} shape`);
        const at = firstMismatch(bytesOf(tensor.data), recorded.inputs[name].bytes);
        assertEquals(
          at,
          -1,
          `${entry.name}: the host's ${name} differs from the recorded one at byte ${at}`,
        );
      }
      assertEquals(built[QWEN3_5_INPUT_IDS].shape, recorded.inputs[QWEN3_5_INPUT_IDS].shape);
    }
  },
);

describe.skipIf(!RUNNABLE)(
  `Qwen3.5 text encoder (real GPU): the rows of each recorded prompt match Karume's CPU reference${skipped(GPU_AVAILABLE, "GPU adapter", "a WebGPU device")}${MIRROR_SKIP}${REFERENCES_SKIP}`,
  () => {
    const fault = { swapRope: false };
    let gpu: GpuContext | undefined;
    let encoder: Qwen3_5TextEncoder | undefined;
    let width = 0;

    beforeAll(async () => {
      assertTestAssetsPresent(REFERENCE_FILES);
      // The graph name is the manifest's weights key.
      const prepared = prepareContainer(await openMossMirrorContainer(COMPONENT), COMPONENT);
      gpu = await acquireTestGpu();
      // The rows written below are this machine's: the key's adapter must be the one that runs.
      assertAdapterMatchesEnvironment(gpu);
      encoder = await Qwen3_5TextEncoder.open(faultableComponent(prepared, fault), gpu);
      width = encoder.hiddenSize;
    });

    afterAll(async () => {
      await encoder?.dispose();
      if (gpu !== undefined) {
        await settleReleases(gpu);
        gpu.destroy();
      }
    });

    const opened = (): Qwen3_5TextEncoder => {
      if (encoder === undefined) throw new Error("the encoder was not opened by beforeAll");
      return encoder;
    };

    for (const entry of CASES) {
      it(`${entry.name} (T = ${entry.tokens}): within the band of the CPU reference, then the sha256 row`, async () => {
        let settlement: ReferenceSettlement | undefined;
        await runRecordedCase(results, { id: caseIdOf(entry) }, async ({ comparisons }) => {
          const recorded = await loadRecorded(entry);
          const rows = await opened().encode(recorded.ids);
          const measured = distance(rows, recorded.reference, width);
          comparisons.push({
            output: "last_hidden_state",
            reference: "cpu-reference",
            practical: "gpu",
            relRms: measured.relRms,
            maxAbs: measured.maxAbs,
            band: { metric: "relRms", floor: 0, ceiling: GPU_REL_RMS_BAND },
          });
          console.log(
            `[qwen3_5] ${entry.name}: GPU vs CPU reference — ${formatDistance(measured)}`,
          );
          assert(
            measured.relRms <= GPU_REL_RMS_BAND,
            `${entry.name}: GPU vs CPU reference ${formatDistance(measured)} — outside the band ${GPU_REL_RMS_BAND}`,
          );
          const outcome = await settleOrObserve(references, results, {
            id: caseIdOf(entry),
            artifact: `${entry.name}.safetensors`,
            bytes: f32ArtifactBytes("last_hidden_state", [1, entry.tokens, width], rows),
          });
          settlement = outcome.settlement;
          return outcome.fields;
        });
        if (settlement?.check.status === "fail") {
          fail(referenceMismatchMessage(caseIdOf(entry), settlement, references));
        }
      });
    }

    it("the band is live: rope_cos / rope_sin swapped into the same Session leave it", async () => {
      const [entry] = CASES;
      const recorded = await loadRecorded(entry);
      fault.swapRope = true;
      let measured: Distance;
      try {
        measured = distance(await opened().encode(recorded.ids), recorded.reference, width);
      } finally {
        fault.swapRope = false;
      }
      console.log(`[qwen3_5] ${entry.name} with swapped RoPE rows: ${formatDistance(measured)}`);
      assert(
        measured.relRms > GPU_REL_RMS_BAND,
        `${entry.name}: swapped RoPE rows stay within the band ${GPU_REL_RMS_BAND} (${formatDistance(measured)})`,
      );
    });
  },
);

registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
