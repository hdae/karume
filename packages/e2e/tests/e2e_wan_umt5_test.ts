/**
 * The **port check** of Wan2.1's umT5 (the text encoder of the i8 series) (real GPU, ADR 0119 step 10c — decisions 3 / 4 / 5 / 8).
 *
 * The text_encoder container of the distribution mirror `models/karume-umt5-xxl/` (opened through its manifest —
 * ADR 0139 decision 1; int8 per-channel weights, f32 activations, inputs token ids `[1,L]` and the bucket table
 * `[L,L]`, L a symbolic dimension of 2 to 512) runs on the karume runtime and is compared with the CPU layer-by-layer
 * reference (the same int8 fake-quantized weights — recipe `wan/umt5_reference.py`). The wiring into the pipeline
 * is stage 10d; this file checks the text_encoder's forward on its own.
 *
 * The goldens are one `reference.<case>.safetensors` per case, in the test-asset root (`node tools/test-assets/fetch.ts`
 * — ADR 0139 decision 2; written by `cd tools/export-recipes && uv run --group wan --inexact python -m
 * wan.umt5_export reference`). Their content, reader, band and judgement are in `helpers/umt5-golden.ts` (shared with
 * the public `umt5.Encoder` e2e).
 *
 * ## Checks
 *
 * 1. **Both input paths agree** (no GPU needed): the id sequence from passing the golden's source text through the TS
 *    tokenizer (`WanPromptEncoder` — the asset of 10a) and the TS bucket table (`buildUmt5RelativePositionBuckets`)
 *    are **bit-identical** to the golden's inputs. The GPU checks run on these TS inputs (the product path). It also
 *    checks that the golden was taken from the current container (the part 0 sha256).
 * 2. **The band of the normalized ratio r** (decision 8 ①): r = (max|GPU − f64| ÷ max|f64|) ÷ (max|CPU f32 − f64| ÷
 *    max|f64|) (the same metric as the DiT's full-size check — `DIT_FULL_NORMALIZED_BAND` of `e2e_wan_dit_test.ts`).
 *    The band is the worst r of the 6 band-setting cases × 5, rounded up to 2 significant digits; the 4 acceptance
 *    cases are judged separately ({@link UMT5_NORMALIZED_BAND}).
 * 3. **Fault injection** (on the 4 acceptance cases; each must leave the band): shifting the relative-position table by
 *    1, replacing one token of the id sequence with another id, swapping the relative-position tables of layers 0 / 1
 *    (blocks swapped in the container's supply — a separate Session), and one fault specific to i8 (doubling the
 *    per-channel scale of one linear in a middle layer — {@link scaledScaleContainer}, a separate Session).
 *
 * Without the mirror, the goldens or a GPU the cases SKIP explicitly, naming what is missing (ADR 0005). A golden
 * set present only in part is a FAIL, not a SKIP (the completeness test).
 *
 * ## Default run vs opt-in upstream parity
 *
 * Gate 2 (the r band against the CPU f64 reference) and gate 3 (the fault injections that prove the band
 * is live) compare against the recipe's goldens under a tolerance, so they are opt-in behind
 * `KARUME_UPSTREAM_PARITY=1` (helpers/upstream-parity.ts). The default run keeps asset completeness,
 * gate 1 (host tokenizer / bucket-table bit identity and the golden's container sha256), the graph's
 * input / output structure, and each case's TS-input forward having the golden's output shape with no
 * non-finite value.
 *
 * ## Public `umt5.Encoder` through the distribution mirror
 *
 * The last two tests of this file drive the public `umt5.Encoder` (`@karume/models/umt5`) built by
 * `fromPretrained(nodeDirectory(…))` over the distribution mirror `models/karume-umt5-xxl`, against the same 10
 * goldens and the same band. They cover what the runtime-level gate above does not: the public pipeline path
 * (manifest selection, tokenizer asset), the encoder's own tokenizer ids, and abort / dispose. Their skip
 * conditions are their own (mirror + goldens), independent of Wan's tokenizer asset. The band check on the mirror
 * is opt-in like the runtime-level band; ids, shape, finiteness and abort / dispose stay in the default run.
 */

import { readFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { fetchAssets, loadManifest, resolveSelection } from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";
import {
  type BoundContainer,
  type GpuContext,
  type OpenedContainer,
  prepareContainer,
  type PreparedModel,
  type Session,
  type SessionDiagnostics,
  type Tensor,
} from "@karume/runtime";
import * as umt5 from "../../models/umt5.ts";
import { parseUmt5TokenizerAsset, Umt5Tokenizer } from "../../models/src/umt5/tokenizer.ts";
import { parseWanTokenizerAsset, WanPromptEncoder } from "../../models/src/wan/text/tokenizer.ts";
import {
  buildUmt5RelativePositionBuckets,
  type I32Tensor,
} from "../../models/src/umt5/relative-position.ts";
import { type Umt5SessionInputs, umt5SessionInputs } from "../../models/src/umt5/session-io.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import {
  BUCKETS_KEY,
  type CaseRole,
  CASES,
  COMPONENT,
  difference,
  filePresent,
  firstMismatch,
  formatRatio,
  GENERATE,
  type Golden,
  FETCH,
  GOLDEN_DIR,
  goldenUrl,
  INPUT_IDS_KEY,
  judgeBand,
  loadGolden,
  ratioOf,
  referenceRatioOf,
  UMT5_NORMALIZED_BAND,
} from "../../models/tests/helpers/umt5-golden.ts";
import {
  mirrorPresent,
  mirrorRoot,
  openSelectedContainer,
  readMirrorAsset,
  selectMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import { TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/** The distribution mirror (the default output of `dist.py --pipeline umt5`). */
const DIST_ROOT = mirrorRoot("karume-umt5-xxl");
const ASSEMBLE = "cd tools/export-recipes && uv run python dist.py --pipeline umt5";
/** The preset (explicit — omitted, it silently moves to the manifest's default. The goldens were taken from the i8 weights). */
const MIRROR_MODEL = "xxl";
const MIRROR_QUANT = "i8";
/**
 * The manifest's component name and the tokenizer asset's key (spelled as in `src/umt5/encoder.ts`). The component key is also the
 * container's graph name (the manifest's weights key — container-v1 §2.1).
 */
const TEXT_ENCODER = COMPONENT;
const TOKENIZER = "tokenizer";

/** Opens the mirror's text_encoder container (range reads, through the manifest). */
const openTextEncoder = async (): Promise<OpenedContainer> =>
  await openSelectedContainer(
    await selectMirrorComponent(DIST_ROOT, TEXT_ENCODER, {
      model: MIRROR_MODEL,
      quant: MIRROR_QUANT,
    }),
  );

/**
 * Wan's tokenizer asset (prompt_clean + umT5 — the `umt5_tokenizer` of the published Wan2.2 distribution, the same
 * bytes as Wan2.1's).
 */
const WAN_MIRROR = mirrorRoot("karume-wan2.2");

/**
 * The relative-position tables of the 2 layers the fault injection swaps (a layer index off by one — the first 2
 * layers). The container's initializer names are spelled as in the recipe's `umt5_reference.layer_keys`.
 */
const relativeBiasTable = (layer: number): string =>
  `encoder.block.${layer}.layer.0.SelfAttention.relative_attention_bias.weight`;
const SWAPPED_TABLES: readonly [string, string] = [relativeBiasTable(0), relativeBiasTable(1)];

/**
 * The linear whose per-channel scale the i8-specific fault injection doubles ({@link scaledScaleContainer}). It is the
 * FFN output `wo` in the middle of the 24 layers (layer 12): the output is added linearly to the residual, so a factor
 * (1 + δ) shows up as shifting that layer's FFN contribution by δ, and the relation between δ and r is readable. q / k
 * were avoided — umT5 has no norm or scale right after q / k (upstream `UMT5Attention.forward` — `scores = q·kᵀ` plus
 * the relative-position bias, then softmax), so the factor is not cancelled, but it acts nonlinearly as a softmax
 * temperature, which makes the relation between δ and r hard to read at subtle levels (a different situation from the
 * reason the DiT chose V — the RMS norm right after q / k).
 */
const I8_SCALE_FAULT_WEIGHT = "encoder.block.12.layer.1.DenseReluDense.wo.weight";
/** The factor that must land outside the band (the size of an i8 scale mix-up — axis, value, binding). */
const GROSS_SCALE_FACTOR = 2;

const DIST_PRESENT = mirrorPresent(DIST_ROOT);
const expectedFiles = CASES.map(({ name }) => goldenUrl(name));
const GOLDENS_PRESENT = expectedFiles.every(filePresent);
const ASSETS_AVAILABLE = DIST_PRESENT && GOLDENS_PRESENT;
const ANY_GOLDEN_PRESENT = expectedFiles.some(filePresent);
/**
 * A complete fetch of the pinned test-asset dataset makes a missing golden a FAIL (the completeness test below): the
 * dataset at the pin lacks a file the cases read. Without one, the cases SKIP (the assets gate is red then).
 */
const FETCHED = testAssetsFetched();
const MISSING_GOLDENS = FETCHED
  ? `the test-asset dataset at revision ${TEST_ASSETS.revision} lacks them`
  : `fetch: ${FETCH}`;
const tokenizerText = mirrorPresent(WAN_MIRROR)
  ? new TextDecoder().decode(await readMirrorAsset(WAN_MIRROR, "umt5_tokenizer"))
  : undefined;

if (!DIST_PRESENT) {
  console.warn(
    `[karume] the mirror ${DIST_ROOT.pathname} is absent (or not karume/5), so the umT5 e2e is SKIPPED. ` +
      `Assemble it: ${ASSEMBLE} (an all-SKIP is a FAIL of the gate distribution_gate_test.ts)`,
  );
} else if (!GOLDENS_PRESENT) {
  console.warn(
    `[karume] the umT5 goldens under ${GOLDEN_DIR.pathname} are not all present, so the umT5 e2e is ` +
      `SKIPPED (${MISSING_GOLDENS})`,
  );
} else if (tokenizerText === undefined) {
  console.warn(
    `[karume] the mirror ${WAN_MIRROR.pathname} is absent (or not karume/5, or a sibling it references is ` +
      `absent), so the Wan umT5 port check (its inputs come from Wan's TS tokenizer) is SKIPPED ` +
      "(fetch the karume-wan2.2 distribution into models/)",
  );
}

const hex = (bytes: ArrayBuffer): string =>
  [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, "0")).join("");

/**
 * The sha256 of the container's part 0 (header + 2 documents — it declares the sha256 of every block), compared with the golden's metadata. Hashed
 * from the mirror's file, not taken from the manifest's declaration.
 */
const containerPart0Sha256 = async (): Promise<string> => {
  const { container } = await selectMirrorComponent(DIST_ROOT, TEXT_ENCODER, {
    model: MIRROR_MODEL,
    quant: MIRROR_QUANT,
  });
  const [part0] = container.parts;
  assert(
    part0.repo === undefined,
    `${TEXT_ENCODER}: part 0 is a cross-repo reference (${part0.repo})`,
  );
  return hex(await crypto.subtle.digest("SHA-256", await readFile(new URL(part0.path, DIST_ROOT))));
};

/** The TS tokenizer (a run without the asset has already SKIPped — reaching here without it fails as contradicting conditions). */
const tokenizerEncoder = (): WanPromptEncoder => {
  if (tokenizerText === undefined) {
    throw new Error(`${WAN_MIRROR.pathname} is absent (contradicts the skip condition)`);
  }
  return new WanPromptEncoder(parseWanTokenizerAsset(JSON.parse(tokenizerText)));
};

/**
 * Builds the graph inputs through the product path (TS tokenizer + TS bucket table) and returns them after checking
 * that they are **bit-identical** to the golden's inputs (the Python path) (check 1 — the GPU checks run only on
 * matching inputs).
 */
const hostInputs = (encoder: WanPromptEncoder, golden: Golden, name: string) => {
  const ids = encoder.encode(golden.meta.prompt, name);
  const at = firstMismatch(ids, golden.ids);
  assertEquals(
    at,
    -1,
    `${name}: the TS id sequence diverges from the golden at index ${at} (TS has ${ids.length})`,
  );
  const buckets = buildUmt5RelativePositionBuckets(ids.length);
  const cell = firstMismatch(buckets.data, golden.buckets);
  assertEquals(
    cell,
    -1,
    `${name}: the TS bucket table diverges from the golden at element ${cell}`,
  );
  return { ids, buckets, inputs: umt5SessionInputs(ids, buckets) };
};

/** The relative-position table shifted by 1 (`table'[i][j] = bucket(j + 1 − i)` — fault injection of a host distance off by one). */
const shiftedBuckets = (length: number): I32Tensor => {
  const wide = buildUmt5RelativePositionBuckets(length + 1);
  const data = new Int32Array(length * length);
  for (let row = 0; row < length; row += 1) {
    const start = row * (length + 1) + 1;
    data.set(wide.data.subarray(start, start + length), row * length);
  }
  return { dtype: "i32", shape: [length, length], data };
};

/**
 * The id sequence with one token in the middle replaced by another id from the same sequence (an in-vocabulary id —
 * injecting a fault that silently passes as a different word, not one that fails on an out-of-range id).
 */
const replacedToken = (ids: Int32Array<ArrayBuffer>): Int32Array<ArrayBuffer> => {
  const at = Math.floor((ids.length - 1) / 2);
  const other = ids.find((id, index) => index < ids.length - 1 && id !== ids[at]);
  if (other === undefined) throw new Error("the sequence has no other id to substitute");
  const replaced = ids.slice();
  replaced[at] = other;
  return replaced;
};

/**
 * A supply with the blocks of two layers' relative-position tables swapped (fault injection — a mix-up of the layer
 * tables). The container's files are not rewritten; the blocks `readBlock` returns are swapped (both are genuine blocks
 * that passed the sha256 verification — as the MUST of `BoundContainer.readBlock` requires, a returned buffer is not
 * modified).
 */
const swappedTablesContainer = (
  opened: OpenedContainer,
  graph: string,
  [first, second]: readonly [string, string],
): BoundContainer => {
  const bound = Object.hasOwn(opened.graphs, graph) ? opened.graphs[graph] : undefined;
  const blockOf = (weight: string): string => {
    const supply = bound?.supplies.get(weight);
    if (supply === undefined || supply.blocks.length !== 1 || supply.scale !== undefined) {
      throw new Error(`graph '${graph}': '${weight}' is not a whole single-block F32`);
    }
    return supply.blocks[0].id;
  };
  const swap = new Map([
    [blockOf(first), blockOf(second)],
    [blockOf(second), blockOf(first)],
  ]);
  return {
    graphs: opened.graphs,
    readBlock: (id) => opened.readBlock(swap.get(id) ?? id),
  };
};

/**
 * A supply with the per-channel scale of one linear (`weight`) multiplied by `factor` (the i8-specific fault injection —
 * {@link I8_SCALE_FAULT_WEIGHT}; the DiT's `scaledScaleContainer` with a factor argument added). The container's files are
 * not rewritten: a **copy** of the fetched block (after the sha256 verification) is scaled and returned (the MUST of
 * `BoundContainer.readBlock` — a returned buffer is not modified). The scale payload is an f32 sequence from the start of
 * the block (as Session construction reads it with `subarray(0, payloadBytes)`).
 */
const scaledScaleContainer = (
  opened: OpenedContainer,
  graph: string,
  weight: string,
  factor: number,
): BoundContainer => {
  const bound = Object.hasOwn(opened.graphs, graph) ? opened.graphs[graph] : undefined;
  const scale = bound?.supplies.get(weight)?.scale;
  if (scale === undefined) {
    throw new Error(`graph '${graph}': '${weight}' has no companion scale (not an i8 linear)`);
  }
  return {
    graphs: opened.graphs,
    readBlock: async (id) => {
      const bytes = await opened.readBlock(id);
      if (id !== scale.id) return bytes;
      const scaled = bytes.slice();
      const values = new Float32Array(scaled.buffer, 0, scale.payloadBytes / 4);
      for (let index = 0; index < values.length; index += 1) values[index] *= factor;
      return scaled;
    },
  };
};

const gib = (bytes: number): string => `${(bytes / 2 ** 30).toFixed(2)} GiB`;

/** The observation of one forward (the wall time and the allocations alive right after the run). */
type Run = {
  readonly label: string;
  readonly wallMs: number;
  readonly weightBytes: number;
  readonly arenaBytes: number;
  readonly backingBytes: number;
};

const observeRun = (label: string, diagnostics: SessionDiagnostics, wallMs: number): Run => ({
  label,
  wallMs,
  weightBytes: diagnostics.weights.allocatedBytes,
  arenaBytes: diagnostics.lastRun?.allocatedBytes ?? 0,
  backingBytes: diagnostics.planBacking.residentBytes + diagnostics.planBacking.inputBytes,
});

const formatRun = (run: Run): string =>
  `${run.label}: wall ${(run.wallMs / 1000).toFixed(2)} s, allocated ${gib(
    run.weightBytes + run.arenaBytes + run.backingBytes,
  )} (weights ${gib(run.weightBytes)}, arena ${gib(run.arenaBytes)}, resident ${gib(
    run.backingBytes,
  )})`;

/** Runs one forward and returns the output (f32 `[1, L, 4096]`) and the observation (waits for the releases before the next run). */
const forward = async (
  session: Session,
  prepared: PreparedModel,
  gpu: GpuContext,
  inputs: Umt5SessionInputs,
  label: string,
): Promise<{ readonly output: Extract<Tensor, { readonly dtype: "f32" }>; readonly run: Run }> => {
  const started = performance.now();
  const outputs = await session.run(inputs);
  const wallMs = performance.now() - started;
  const output = outputs[prepared.graph.outputs[0]];
  if (output.dtype !== "f32") throw new Error(`${label}: the output is ${output.dtype}`);
  const run = observeRun(label, session.diagnostics(), wallMs);
  await settleReleases(gpu);
  return { output, run };
};

/**
 * A resource a describe's `beforeAll` opens (device, Session, encoder), or a loud failure when that
 * `beforeAll` did not get that far. The `afterAll` hooks read the raw `undefined` to close only what
 * was opened.
 */
const requireOpened = <T>(value: T | undefined, what: string): T => {
  if (value === undefined) throw new Error(`the ${what} was not opened by beforeAll`);
  return value;
};

const results = openResults("wan-umt5");

// Only a root with no golden at all that is not a complete fetch SKIPs; with any one present or a complete fetch, a
// missing one is a FAIL. The mirror's completeness is the distribution gate's (distribution_gate_test.ts).
test.skipIf(!ANY_GOLDEN_PRESENT && !FETCHED)(
  "Wan umT5 goldens: the goldens of every case (6 band, 4 accept) are present",
  () => {
    assertEquals(
      expectedFiles.filter((url) => !filePresent(url)).map((url) => url.pathname),
      [],
      `missing goldens (${MISSING_GOLDENS})`,
    );
  },
);

test.skipIf(!ASSETS_AVAILABLE || tokenizerText === undefined)(
  "Wan umT5 host (no GPU needed): the TS tokenizer's id sequence and the TS bucket table are bit-identical to " +
    "the golden's inputs (the Python path), and the golden was taken from the current container",
  async () => {
    const encoder = tokenizerEncoder();
    const part0 = await containerPart0Sha256();
    for (const { name, role } of CASES) {
      const golden = await loadGolden(name);
      assertEquals(golden.meta.role, role, `${name}: role`);
      assertEquals(
        golden.meta.part0Sha256,
        part0,
        `${name}: the container the golden was taken from (part 0 sha256) differs from the current one — regenerate: ${GENERATE}`,
      );
      hostInputs(encoder, golden, name);
    }
  },
);

/** One case's r (material for the band candidate — recorded before the verdict, so runs outside the band are kept too). */
type CaseRatio = { readonly name: string; readonly role: CaseRole; readonly ratio: number };

/**
 * Prints a candidate for the not-yet-derived band and stops red (the worst r of the band-setting cases × 5, rounded
 * up to 2 significant digits — the acceptance r values are listed but not used for the candidate). When any
 * band-setting case failed before producing r, no candidate is given (a worst value with gaps is not the worst).
 */
const bandCandidate = (ratios: readonly CaseRatio[]): string => {
  const expected = CASES.filter(({ role }) => role === "band").length;
  const decided = ratios.filter(({ role }) => role === "band");
  const accepted =
    ratios
      .filter(({ role }) => role === "accept")
      .map(({ name, ratio }) => `${name} ${ratio.toPrecision(3)}`)
      .join(" / ") || "none";
  if (decided.length !== expected) {
    return (
      `of the ${expected} band-setting cases only ${decided.length} produced r — no candidate (investigate the failed cases first). ` +
      `Acceptance r: ${accepted}`
    );
  }
  const worst = Math.max(...decided.map(({ ratio }) => ratio));
  const unit = 10 ** (Math.floor(Math.log10(worst * 5)) - 1);
  const candidate = Number((Math.ceil((worst * 5) / unit) * unit).toPrecision(2));
  return (
    `worst r of the ${expected} band-setting cases (${decided
      .map(({ name, ratio }) => `${name} ${ratio.toPrecision(3)}`)
      .join(" / ")}) ${worst.toPrecision(3)} × 5 = ${(worst * 5).toPrecision(
      3,
    )} (rounded up to 2 significant digits: ${candidate}). ` +
    `Acceptance r (not used to set the band): ${accepted}`
  );
};

describe.skipIf(!ASSETS_AVAILABLE || tokenizerText === undefined || !GPU_AVAILABLE)(
  "Wan umT5 (real GPU, default run): the graph takes input_ids and the bucket table and returns one " +
    "output, and each case's TS-input forward has the golden's output shape with no non-finite value",
  () => {
    // Assigned in beforeAll; the its do not run when beforeAll throws.
    let encoder!: WanPromptEncoder;
    let prepared!: PreparedModel;
    let gpu: GpuContext | undefined;
    let session: Session | undefined;
    let deviceLost: string | undefined;

    beforeAll(async () => {
      encoder = tokenizerEncoder();
      prepared = prepareContainer(await openTextEncoder(), TEXT_ENCODER);
      assertEquals(
        prepared.graph.inputs.map(({ name }) => name),
        [INPUT_IDS_KEY, BUCKETS_KEY],
        "graph inputs",
      );
      assertEquals(prepared.graph.outputs.length, 1, "the graph has one output");
      gpu = await acquireTestGpu({
        onDeviceLost: (info) => {
          deviceLost = `${info.reason}: ${info.message}`;
        },
      });
      assertAdapterMatchesEnvironment(gpu);
      session = await prepared.createContainerSession(gpu);
    });

    afterAll(async () => {
      // Same order as the former finally blocks: the Session first, then the device.
      if (gpu === undefined) return;
      try {
        if (session !== undefined) {
          await session.dispose();
          await settleReleases(gpu);
        }
      } finally {
        await settleReleases(gpu);
        gpu.destroy();
      }
    });

    for (const { name, role } of CASES) {
      it(`${name}（${role}）`, async () => {
        const device = requireOpened(gpu, "device");
        const golden = await loadGolden(name);
        const { inputs } = hostInputs(encoder, golden, name);
        const { output } = await forward(
          requireOpened(session, "Session"),
          prepared,
          device,
          inputs,
          name,
        );
        assertEquals(output.shape, golden.outputShape, `${name}: output shape`);
        const nonFinite = output.data.filter((value) => !Number.isFinite(value)).length;
        assertEquals(nonFinite, 0, `${name}: non-finite`);
        assertEquals(deviceLost, undefined, "device lost");
      });
    }
  },
);

/**
 * Registers the hooks of one Session scope in the caller's describe: `beforeAll` opens the Session
 * over `prepared()` on `gpu()`, `afterAll` disposes it and waits for the releases. Each scope gets
 * its own describe so two Sessions never coexist (the scopes run one after another, in registration
 * order). Returns the accessor the its use.
 */
const registerSessionScope = (
  prepared: () => PreparedModel,
  gpu: () => GpuContext,
): (() => Session) => {
  let session: Session | undefined;
  beforeAll(async () => {
    session = await prepared().createContainerSession(gpu());
  });
  afterAll(async () => {
    if (session === undefined) return;
    await session.dispose();
    await settleReleases(gpu());
  });
  return () => requireOpened(session, "Session");
};

describe.skipIf(
  !UPSTREAM_PARITY || !ASSETS_AVAILABLE || tokenizerText === undefined || !GPU_AVAILABLE,
)(
  "Wan umT5 port check (real GPU / CPU layer-by-layer f64): the normalized ratio r of one forward on the TS inputs is " +
    "inside the band and the 4 fault injections are outside it" +
    UPSTREAM_PARITY_OPT_IN,
  () => {
    const band = UMT5_NORMALIZED_BAND;
    const ratios: CaseRatio[] = [];
    // Assigned in beforeAll; the its do not run when beforeAll throws.
    let encoder!: WanPromptEncoder;
    let graph!: string;
    let opened!: OpenedContainer;
    let prepared!: PreparedModel;
    let gpu: GpuContext | undefined;
    let deviceLost: string | undefined;
    const device = (): GpuContext => requireOpened(gpu, "device");

    beforeAll(async () => {
      encoder = tokenizerEncoder();
      graph = TEXT_ENCODER;
      opened = await openTextEncoder();
      prepared = prepareContainer(opened, graph);
      assertEquals(
        prepared.graph.inputs.map(({ name }) => name),
        [INPUT_IDS_KEY, BUCKETS_KEY],
        "graph inputs",
      );
      assertEquals(prepared.graph.outputs.length, 1, "the graph has one output");
      gpu = await acquireTestGpu({
        onDeviceLost: (info) => {
          deviceLost = `${info.reason}: ${info.message}`;
        },
      });
      assertAdapterMatchesEnvironment(gpu);
    });

    afterAll(async () => {
      // Runs after the Session scopes below have closed theirs (the former outer finally block).
      if (gpu === undefined) return;
      await settleReleases(gpu);
      gpu.destroy();
    });

    describe("main Session", () => {
      const session = registerSessionScope(() => prepared, device);
      for (const { name, role } of CASES) {
        it(`${name}（${role}）`, async () => {
          let caseNote = "";
          await runRecordedCase(
            results,
            {
              id: name,
              failureNote: () => caseNote,
            },
            async ({ measurements }) => {
              const golden = await loadGolden(name);
              const { ids, inputs } = hostInputs(encoder, golden, name);
              const referenceRatio = referenceRatioOf(golden, name);
              const { output, run } = await forward(session(), prepared, device(), inputs, name);
              assertEquals(output.shape, golden.outputShape, `${name}: output shape`);
              const diff = difference(output.data, golden.expected);
              const diffF32 = difference(output.data, golden.reference);
              const referenceDiff = difference(golden.reference, golden.expected);
              const ratio = ratioOf(diff) / referenceRatio;
              ratios.push({ name, role, ratio });
              const unbounded = { atol: Number.POSITIVE_INFINITY, rtol: 0 };
              measurements.push(
                {
                  output: "output@f64-reference",
                  maxAbs: diff.maxAbs,
                  maxRel: diff.maxRel,
                  // Map the verdict's bound to an absolute value (an unbounded band while not derived — the verdict
                  // below turns it red).
                  tolerance:
                    band === undefined
                      ? unbounded
                      : { atol: band * referenceRatio * diff.referenceMaxAbs, rtol: 0 },
                  stage: "karume",
                },
                {
                  output: "output@f32-reference",
                  maxAbs: diffF32.maxAbs,
                  maxRel: diffF32.maxRel,
                  tolerance: unbounded,
                  stage: "karume",
                },
                {
                  output: "f32-reference@f64-reference",
                  maxAbs: referenceDiff.maxAbs,
                  maxRel: referenceDiff.maxRel,
                  tolerance: unbounded,
                  stage: "karume",
                },
              );
              const notes = [
                `L ${ids.length}`,
                formatRatio("GPU / f64 reference", diff, referenceRatio),
                `CPU f32 reference / f64 reference (the normalization denominator): ratio ${referenceRatio.toExponential(3)}`,
                formatRun(run),
              ];
              assertEquals(diff.nonFinite, 0, `${name}: non-finite`);

              const faults: { label: string; ratio: number }[] = [];
              if (role === "accept") {
                const injected: readonly (readonly [string, Umt5SessionInputs])[] = [
                  [
                    "relative-position table shifted by 1",
                    umt5SessionInputs(ids, shiftedBuckets(ids.length)),
                  ],
                  [
                    "one token in the middle of the id sequence replaced by another id",
                    umt5SessionInputs(
                      replacedToken(ids),
                      buildUmt5RelativePositionBuckets(ids.length),
                    ),
                  ],
                ];
                for (const [label, faulty] of injected) {
                  const { output: faultOutput } = await forward(
                    session(),
                    prepared,
                    device(),
                    faulty,
                    `${name} ${label}`,
                  );
                  const faultDiff = difference(faultOutput.data, golden.expected);
                  faults.push({ label, ratio: ratioOf(faultDiff) / referenceRatio });
                  notes.push(`fault injection ${formatRatio(label, faultDiff, referenceRatio)}`);
                }
              }
              // Print before the verdict (a run outside the band keeps its ratio at hand too).
              caseNote = notes.join(" / ");
              console.log(`[wan-umt5] ${name}: ${caseNote}`);
              assertEquals(deviceLost, undefined, "device lost");
              judgeBand(band, name, ratio, faults);
              return { status: "pass", note: caseNote };
            },
          );
        });
      }
    });

    const swapLabel = "relative-position tables of layers 0 / 1 swapped";
    // Built after the main Session is torn down (never two at once — 5.3 GiB of weights × 2 exceeds the B570's
    // ceiling of 9,600 MiB).
    describe(`fault Session: ${swapLabel}`, () => {
      let swapped!: PreparedModel;
      beforeAll(() => {
        swapped = prepareContainer(swappedTablesContainer(opened, graph, SWAPPED_TABLES), graph);
      });
      const faultSession = registerSessionScope(() => swapped, device);
      for (const { name } of CASES.filter(({ role }) => role === "accept")) {
        const id = `${name}/table-swap`;
        it(`${id} (fault injection ${swapLabel} → outside the band)`, async () => {
          let note = "";
          await runRecordedCase(results, { id, failureNote: () => note }, async () => {
            const golden = await loadGolden(name);
            const { inputs } = hostInputs(encoder, golden, name);
            const referenceRatio = referenceRatioOf(golden, name);
            const { output } = await forward(faultSession(), swapped, device(), inputs, id);
            const faultDiff = difference(output.data, golden.expected);
            const ratio = ratioOf(faultDiff) / referenceRatio;
            note = `fault injection ${formatRatio(swapLabel, faultDiff, referenceRatio)} (${
              band === undefined
                ? "band not derived yet"
                : `${(ratio / band).toFixed(1)} × the band`
            })`;
            console.log(`[wan-umt5] ${id}: ${note}`);
            if (band === undefined)
              throw new Error(`${id}: band not derived yet (fault-injection r ${ratio})`);
            assert(ratio > band, `${id}: r ${ratio.toPrecision(3)} stayed inside the band ${band}`);
            return { status: "pass", note };
          });
        });
      }
    });

    // The i8-specific fault injection (a scale factor). The previous Session is torn down before the next one is
    // built (the same reason as the table swap — never two at once).
    const label = `scale of ${I8_SCALE_FAULT_WEIGHT} × ${GROSS_SCALE_FACTOR}`;
    describe(`fault Session: ${label}`, () => {
      let scaled!: PreparedModel;
      beforeAll(() => {
        scaled = prepareContainer(
          scaledScaleContainer(opened, graph, I8_SCALE_FAULT_WEIGHT, GROSS_SCALE_FACTOR),
          graph,
        );
      });
      const scaledSession = registerSessionScope(() => scaled, device);
      for (const { name } of CASES.filter(({ role }) => role === "accept")) {
        const id = `${name}/scale-fault`;
        it(`${id} (fault injection ${label} → outside the band)`, async () => {
          let note = "";
          await runRecordedCase(results, { id, failureNote: () => note }, async () => {
            const golden = await loadGolden(name);
            const { inputs } = hostInputs(encoder, golden, name);
            const referenceRatio = referenceRatioOf(golden, name);
            const { output } = await forward(scaledSession(), scaled, device(), inputs, id);
            const faultDiff = difference(output.data, golden.expected);
            const ratio = ratioOf(faultDiff) / referenceRatio;
            note = `fault injection ${formatRatio(label, faultDiff, referenceRatio)} (${
              band === undefined
                ? "band not derived yet"
                : `${(ratio / band).toPrecision(3)} × the band`
            })`;
            console.log(`[wan-umt5] ${id}: ${note}`);
            assertEquals(deviceLost, undefined, "device lost");
            if (band === undefined) {
              throw new Error(`${id}: band not derived yet (fault-injection r ${ratio})`);
            }
            assert(ratio > band, `${id}: r ${ratio.toPrecision(3)} stayed inside the band ${band}`);
            return { status: "pass", note };
          });
        });
      }
    });

    if (band === undefined) {
      it("band candidate (not derived yet)", () => {
        throw new Error(`band not derived yet — ${bandCandidate(ratios)}`);
      });
    }
  },
);

/**
 * The end-to-end e2e of `umt5.Encoder` (`@karume/models/umt5`) (real GPU) — built over the distribution mirror
 * `models/karume-umt5-xxl` with `fromPretrained(nodeDirectory(…))`, it checks **the same 10 goldens** as Wan's umT5 port
 * check (`e2e_wan_umt5_test.ts`) with **the same verdict** (the normalized ratio r inside the band —
 * `helpers/umt5-golden.ts`).
 *
 * The input is the golden metadata's `cleaned` (the string after the upstream preprocessing `prompt_clean`): the
 * golden's `input_ids` are that string passed through transformers 5.14.1, and the standalone encoder does no
 * preprocessing (strings after Wan's preprocessing fall inside the set the standalone tokenizer accepts —
 * `umt5_tokenizer_test.ts`).
 *
 * ## Checks
 *
 * 1. **Host (no GPU needed)**: the tokenizer built from the mirror's tokenizer asset (the manifest's `tokenizer` — the
 *    hub verifies its sha256 before handing it over) produces from `cleaned` a sequence **bit-identical** to the
 *    golden's `input_ids`. The part 0 sha256 of the container the golden was taken from equals the part 0 of the
 *    `text_encoder` the mirror's manifest declares (= the golden was taken from the distributed weights).
 * 2. **Real GPU**: `tokenize(cleaned)` and `encode(cleaned).inputIds` match the golden's `input_ids`, and the
 *    normalized ratio r of `hiddenStates` (`[L, 4096]`) is inside the band with 0 non-finite values (`judgeBand` — the
 *    fault injections belong to Wan's check, so none are passed here).
 * 3. **Real GPU**: an already-aborted signal throws `signal.reason` unwrapped before the run, and the next `encode` on
 *    the same encoder runs. `encode` after `dispose` fails.
 *
 * An environment without the mirror, the goldens or a GPU SKIPs explicitly (ADR 0005) — a missing mirror as a whole is
 * made a FAIL by the guard `distribution_gate_test.ts`, and missing goldens by the completeness test of
 * `e2e_wan_umt5_test.ts`.
 */

/** The mirror's selection (the manifest's declaration — the preset is explicit). */
const loadMirror = async () => {
  const loaded = await loadManifest(nodeDirectory(DIST_ROOT));
  return {
    loaded,
    selection: resolveSelection(loaded.manifest, { model: MIRROR_MODEL, quant: MIRROR_QUANT }),
  };
};

/** The first mismatch between one case's ids (from the string after the golden metadata's preprocessing) and the golden's id sequence. */
const idsMismatch = (ids: Int32Array, golden: Golden): number => firstMismatch(ids, golden.ids);

test.skipIf(!DIST_PRESENT || !GOLDENS_PRESENT)(
  "umT5 encoder host (no GPU needed): the mirror's tokenizer produces the golden's id sequence from the golden's " +
    "preprocessed string, and the golden was taken from part 0 of the mirror's text_encoder",
  async () => {
    const { loaded, selection } = await loadMirror();
    assert(Object.hasOwn(selection.assets, TOKENIZER), `the mirror has no asset '${TOKENIZER}'`);
    const assets = await fetchAssets(loaded, { [TOKENIZER]: selection.assets[TOKENIZER] });
    const tokenizer = new Umt5Tokenizer(
      parseUmt5TokenizerAsset(JSON.parse(new TextDecoder().decode(assets[TOKENIZER]))),
    );
    const part0 = selection.containers[TEXT_ENCODER].parts[0].sha256;
    for (const { name } of CASES) {
      const golden = await loadGolden(name);
      assertEquals(
        golden.meta.part0Sha256,
        part0,
        `${name}: the container the golden was taken from (part 0 sha256) differs from the mirror's text_encoder`,
      );
      const ids = tokenizer.tokenize(golden.meta.cleaned, name);
      const at = idsMismatch(ids, golden);
      assertEquals(
        at,
        -1,
        `${name}: the id sequence diverges from the golden at index ${at} (${ids.length} ids)`,
      );
    }
  },
);

/**
 * Registers the hooks that build the mirror's `umt5.Encoder` on a test device in the caller's
 * describe: `beforeAll` acquires the device and runs `fromPretrained`, `afterAll` disposes the
 * encoder (idempotent — an it may already have disposed it), waits for the releases, fails on a
 * device loss and destroys the device. Returns the accessors the its use and the device-lost message (undefined while none).
 */
const registerMirrorEncoder = (): {
  readonly gpu: () => GpuContext;
  readonly encoder: () => umt5.Encoder;
  readonly deviceLost: () => string | undefined;
} => {
  let gpu: GpuContext | undefined;
  let encoder: umt5.Encoder | undefined;
  let deviceLost: string | undefined;
  beforeAll(async () => {
    gpu = await acquireTestGpu({
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    encoder = await umt5.Encoder.fromPretrained(nodeDirectory(DIST_ROOT), {
      gpu,
      model: MIRROR_MODEL,
      quant: MIRROR_QUANT,
    });
  });
  afterAll(async () => {
    if (gpu === undefined) return;
    try {
      if (encoder !== undefined) {
        await encoder.dispose();
        await settleReleases(gpu);
      }
      // The former test bodies checked this after the dispose (a device loss during dispose).
      assertEquals(deviceLost, undefined, "device lost");
    } finally {
      gpu.destroy();
    }
  });
  return {
    gpu: () => requireOpened(gpu, "device"),
    encoder: () => requireOpened(encoder, "encoder"),
    deviceLost: () => deviceLost,
  };
};

describe.skipIf(!DIST_PRESENT || !GOLDENS_PRESENT || !GPU_AVAILABLE)(
  "umT5 encoder (real GPU, default run): built by fromPretrained(nodeDirectory), tokenize and inputIds " +
    "match the golden ids and hiddenStates has the golden's shape with no non-finite value for all 10 " +
    "goldens; an aborted signal throws its reason before the run, and encode fails after dispose",
  () => {
    const mirror = registerMirrorEncoder();

    for (const { name } of CASES) {
      it(name, async () => {
        const encoder = mirror.encoder();
        const golden = await loadGolden(name);
        const text = golden.meta.cleaned;
        const tokenized = encoder.tokenize(text);
        assertEquals(idsMismatch(tokenized, golden), -1, `${name}: tokenize ids`);

        const encoding = await encoder.encode(text);
        assertEquals(idsMismatch(encoding.inputIds, golden), -1, `${name}: inputIds`);
        assertEquals(encoding.length, golden.ids.length, `${name}: length`);
        const [, rows, width] = golden.outputShape;
        assertEquals([encoding.length, encoding.hiddenSize], [rows, width], `${name}: shape`);
        const nonFinite = encoding.hiddenStates.filter((value) => !Number.isFinite(value)).length;
        assertEquals(nonFinite, 0, `${name}: non-finite`);
        assertEquals(mirror.deviceLost(), undefined, "device lost");
        await settleReleases(mirror.gpu());
      });
    }

    it("an already-aborted signal throws its reason as is before the run, and the next encode runs", async () => {
      const encoder = mirror.encoder();
      const controller = new AbortController();
      const reason = new Error("umt5-encoder: test abort");
      controller.abort(reason);
      const error = await assertRejects(() =>
        encoder.encode("a cat", { signal: controller.signal }),
      );
      assertStrictEquals(error, reason);
      const after = await encoder.encode("a cat");
      assertEquals(after.length, encoder.tokenize("a cat").length);
    });

    // The former test body's tail after its finally block (dispose, then the rejected encode).
    it("encode fails after dispose", async () => {
      const encoder = mirror.encoder();
      await encoder.dispose();
      await settleReleases(mirror.gpu());
      await assertRejects(() => encoder.encode("a cat"), Error, "after dispose");
      assertEquals(mirror.deviceLost(), undefined, "device lost");
    });
  },
);

describe.skipIf(!UPSTREAM_PARITY || !DIST_PRESENT || !GOLDENS_PRESENT || !GPU_AVAILABLE)(
  "umT5 encoder (real GPU): built by fromPretrained(nodeDirectory), the normalized ratio r of hiddenStates for the 10 goldens " +
    "is inside the band (the same goldens and band as Wan's umT5 port check), and tokenize and inputIds match the golden ids" +
    UPSTREAM_PARITY_OPT_IN,
  () => {
    const mirror = registerMirrorEncoder();

    for (const { name } of CASES) {
      it(name, async () => {
        const encoder = mirror.encoder();
        const golden = await loadGolden(name);
        const text = golden.meta.cleaned;
        const tokenized = encoder.tokenize(text);
        assertEquals(idsMismatch(tokenized, golden), -1, `${name}: tokenize ids`);

        const encoding = await encoder.encode(text);
        assertEquals(idsMismatch(encoding.inputIds, golden), -1, `${name}: inputIds`);
        assertEquals(encoding.length, golden.ids.length, `${name}: length`);
        const [, rows, width] = golden.outputShape;
        assertEquals([encoding.length, encoding.hiddenSize], [rows, width], `${name}: shape`);
        const diff = difference(encoding.hiddenStates, golden.expected);
        const referenceRatio = referenceRatioOf(golden, name);
        const ratio = ratioOf(diff) / referenceRatio;
        console.log(
          `[umt5-encoder] ${name}: L ${encoding.length} / ${formatRatio(
            "GPU / f64 reference",
            diff,
            referenceRatio,
          )}`,
        );
        assertEquals(diff.nonFinite, 0, `${name}: non-finite`);
        assertEquals(mirror.deviceLost(), undefined, "device lost");
        judgeBand(UMT5_NORMALIZED_BAND, name, ratio, []);
        await settleReleases(mirror.gpu());
      });
    }
  },
);
