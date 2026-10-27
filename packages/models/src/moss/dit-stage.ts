/**
 * The MOSS-Image DiT stage — the Sessions of `dit_text`, `dit` and `dit_skip22`, their lifetimes and
 * headroom checks, the text context and the host CFG (ADR NNNN decisions 3–5, plan items 15-2, 15-3,
 * 15-8, plan step 2). The sampler loop, the VAE and the pipeline are later steps; this module runs
 * the DiT side of one generation.
 *
 * ## Stage order and lifetimes (plan item 15-8)
 *
 * 1. {@link runMossTextStage}: one `dit_text` Session runs each branch once (the refiner has no
 *    modulation, so its output does not depend on the step), reads the output back once and is
 *    closed — before the DiT pair opens (an open condition Session would double the output slot —
 *    the ADR 0114 pitfall). It always runs with f32 compute (plan item 15-11), whatever the preset.
 * 2. {@link withMossDitPair}: the `dit` Session (positive branch — the lender) and the `dit_skip22`
 *    Session (negative branch — borrows every weight of `dit`, `SessionOptions.sharedWeights`) are
 *    open together for the whole generation. Each Session sees one shape (S₊ = N + T₊ and
 *    S₋ = N + T₋), so neither rebuilds its plan backing between steps (`planBacking.buildCount`
 *    stays 1). The text context of each branch is a resident tensor written once. Without a
 *    negative branch (guidance 1) only `dit` opens.
 *
 * Both stages open their Sessions with the reference path's options (no preset knob): the headroom
 * estimates below take the device limits and the bindings only, so a knob passed to the Sessions
 * alone would make the check size other Sessions than the ones it admits. The preset knobs arrive
 * together with their estimate options (plan step 5).
 *
 * ## Headroom (plan item 15-8 — the Anima form)
 *
 * Before a stage opens, a trial allocation (`fitsHeadroom` — ADR 0112) checks the bytes the stage
 * needs, and a refusal fails loudly with the stage, the bytes and the size. The bytes are the
 * runtime estimate's `peakAccountedBytes` (weights + the bound of the kept plan backings, with the
 * device limits and the stage's symbol bindings) + the largest weight part (the upload staging) +
 * the margin of `../anima/residency.ts` ({@link HEADROOM_MARGIN_BYTES} — the allocator's block size
 * and wgpu's budget line):
 *
 * - `dit_text`: one estimate at T = max(T₊, T₋) (the two branches run in turn on one Session).
 * - the DiT pair: the **sum** of the `dit` (S₊) and `dit_skip22` (S₋) estimates — both Sessions are
 *   open at once — plus the two resident text contexts. The borrower's estimate counts no weight
 *   (its initializers are all shared — `moss_dit_host_test.ts` checks the estimate,
 *   `e2e_moss_dit_test.ts` the open Sessions), so the weights are counted once.
 *
 *   The text contexts are added on top of the estimates on purpose, as an upper bound: an estimate
 *   counts every graph input as an input buffer its backing owns, but a resident input is not owned
 *   by the backing (`PlanBackingStats.inputBytes`). Where the largest scenario is over the plan
 *   backing budget (1024²), the estimate therefore already holds a buffer the size of the context
 *   and the addition counts it twice (about S · 3,072 · 4 B per branch — 51 MB at 1024²);
 *   where the budget term dominates (small sizes), the estimate holds no context and the addition is
 *   what counts it. One formula that is never short, rather than a copy of the estimate's own
 *   formula.
 *
 * Before each probe the stage waits for the release of what the previous stage destroyed (the
 * Anima form — `../session/settle-released-memory.ts`).
 *
 * The estimate leaves out the i8a8 linears' quantised intermediates (`runtime/estimate.ts`); this
 * step runs the reference path only (f32 compute — no a8), and step 5 records the measured peaks
 * beside the estimates. The check is a fact of the moment, not a reservation (another process may
 * take the memory after it).
 *
 * ## CFG (host, f32)
 *
 * {@link mossClassifierFreeGuidance}: upstream's `uncond + scale · (cond − uncond)` on the two f32
 * samples, each operation rounded to f32 (`Math.fround` of a double result is the f32 result for +,
 * −, × — double has more than 2·24 + 2 bits). The scale is a Python float that torch rounds to f32
 * before it multiplies an f32 tensor.
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import {
  fitsHeadroom,
  type GpuContext,
  type ResidentTensor,
  type RunInputs,
  type Session,
  type SessionDiagnostics,
  type SharedWeight,
  type Tensor,
} from "@karume/runtime";

import { HEADROOM_MARGIN_BYTES } from "../anima/residency.ts";
import type { ModelComponent } from "../hub/components.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { settleReleasedMemory } from "../session/settle-released-memory.ts";
import {
  MOSS_DIT,
  MOSS_DIT_INPUTS,
  MOSS_DIT_SKIP22,
  MOSS_DIT_TEXT,
  MOSS_DIT_TEXT_INPUTS,
  type MossDitContract,
} from "./dit-contract.ts";
import { type MossCells, mossRopeTables, mossTextRopeTables } from "./dit-rope.ts";
import { mossTimestepsProj } from "./dit-timestep.ts";
import {
  mossHeadRows,
  mossImageTokens,
  mossIsImage,
  mossLatentTokens,
  mossTextContext,
} from "./dit-tokens.ts";
import {
  assertMossImageSize,
  assertMossTextEmbeddings,
  MOSS_PIXELS_PER_CELL,
  MOSS_TEXT_WIDTH,
  type MossTextEmbeddings,
} from "./request.ts";

const OWNER = "MOSS-Image";

/** The three components of the DiT series. */
export type MossDitComponents = {
  readonly dit: ModelComponent;
  readonly ditSkip22: ModelComponent;
  readonly ditText: ModelComponent;
};

/**
 * What the stage reads (built once per pipeline — the contract settled in admission). No Session
 * options: module doc "Stage order and lifetimes".
 */
export type MossDitStageState = {
  readonly gpu: GpuContext;
  readonly components: MossDitComponents;
  readonly contract: MossDitContract;
};

/** One value per CFG branch (`negative` undefined = guidance 1, no negative branch). */
export type MossBranches<T> = { readonly positive: T; readonly negative: T | undefined };

/** One branch's text: the encoder output and its token count. */
export type MossBranchText = {
  readonly embeddings: MossTextEmbeddings;
  readonly tokens: number;
};

/** An accepted request of the DiT stage ({@link planMossDit}). */
export type MossDitPlan = {
  readonly cells: MossCells;
  readonly texts: MossBranches<MossBranchText>;
};

/**
 * The acceptance check of the DiT stage (`./request.ts` — size and text embeddings; every failure
 * a `ModelInputError`). `negative` undefined means guidance 1 (no negative branch).
 */
export const planMossDit = (request: {
  readonly width: number;
  readonly height: number;
  readonly positive: MossTextEmbeddings;
  readonly negative: MossTextEmbeddings | undefined;
}): MossDitPlan => {
  const cells = assertMossImageSize(request.width, request.height);
  const branch = (embeddings: MossTextEmbeddings, label: string): MossBranchText => ({
    embeddings,
    tokens: assertMossTextEmbeddings(embeddings, label),
  });
  return {
    cells,
    texts: {
      positive: branch(request.positive, "positive"),
      negative: request.negative === undefined ? undefined : branch(request.negative, "negative"),
    },
  };
};

/** The token count of each branch of the joint sequence (S = N + T). */
const sequenceLengths = (plan: MossDitPlan): MossBranches<number> => {
  const image = mossImageTokens(plan.cells);
  return {
    positive: image + plan.texts.positive.tokens,
    negative: plan.texts.negative === undefined ? undefined : image + plan.texts.negative.tokens,
  };
};

/** The device limits the estimates take (the stage passes `gpu.limits`). */
export type MossDeviceLimits = {
  readonly maxStorageBufferBindingSize: number;
  readonly maxBufferSize: number;
};

const estimatedPeak = (
  limits: MossDeviceLimits,
  component: ModelComponent,
  bindings: Readonly<Record<string, number>>,
): number =>
  component.estimate({
    maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
    maxBufferSize: limits.maxBufferSize,
    bindings,
  }).peakAccountedBytes;

/** The graph's one symbol (the contract checked there is exactly one). */
const symbolOf = (component: ModelComponent): string => component.graph.symbols[0];

/** Bytes of one resident text context `[1, S, D]` (f32). */
const contextBytes = (sequence: number, width: number): number => sequence * width * 4;

/** The bytes each stage needs (module doc "Headroom"). */
export type MossDitStageNeeds = {
  readonly text: number;
  readonly pair: number;
};

/**
 * The bytes of the two stages of a plan on a device of `limits` (module doc "Headroom"). NOTE:
 * `export` is for the tests that pin the sum (not on a public surface).
 */
export const mossDitStageNeeds = (
  input: {
    readonly limits: MossDeviceLimits;
    readonly components: MossDitComponents;
    readonly contract: Pick<MossDitContract, "width">;
  },
  plan: MossDitPlan,
): MossDitStageNeeds => {
  const { limits, components, contract } = input;
  const { texts } = plan;
  const longest = Math.max(texts.positive.tokens, texts.negative?.tokens ?? 0);
  const text =
    estimatedPeak(limits, components.ditText, { [symbolOf(components.ditText)]: longest }) +
    components.ditText.maxPartBytes +
    HEADROOM_MARGIN_BYTES;
  const lengths = sequenceLengths(plan);
  const negative =
    lengths.negative === undefined
      ? 0
      : estimatedPeak(limits, components.ditSkip22, {
          [symbolOf(components.ditSkip22)]: lengths.negative,
        }) + contextBytes(lengths.negative, contract.width);
  const pair =
    estimatedPeak(limits, components.dit, { [symbolOf(components.dit)]: lengths.positive }) +
    contextBytes(lengths.positive, contract.width) +
    negative +
    Math.max(components.dit.maxPartBytes, components.ditSkip22.maxPartBytes) +
    HEADROOM_MARGIN_BYTES;
  return { text, pair };
};

/**
 * Fails loudly when `bytes` do not fit the device now (module doc "Headroom").
 *
 * MUST: wait for the release of what an earlier stage destroyed before the trial allocation
 * (`../session/settle-released-memory.ts`). Observed 2026-10-27 (RTX 3080 Ti, Dawn): right after
 * a 1024² pair (10.1 GB) was closed, the probe for a 7.9 GB pair was refused on an otherwise empty
 * 12 GiB device; with this wait first, the same sequence passed.
 */
const assertHeadroom = async (
  gpu: GpuContext,
  bytes: number,
  stage: string,
  plan: MossDitPlan,
): Promise<void> => {
  await settleReleasedMemory(gpu);
  if (await fitsHeadroom(gpu, bytes)) return;
  const { rows, cols } = plan.cells;
  throw new Error(
    `${OWNER}: the ${stage} stage needs ${bytes} bytes of VRAM (estimate + largest part + margin)` +
      ` that this device cannot allocate now — ${cols * MOSS_PIXELS_PER_CELL}×` +
      `${rows * MOSS_PIXELS_PER_CELL} px with text of` +
      ` ${plan.texts.positive.tokens} / ${plan.texts.negative?.tokens ?? "-"} tokens does not fit`,
  );
};

const f32Tensor = (shape: readonly number[], data: Float32Array<ArrayBuffer>): Tensor => ({
  dtype: "f32",
  shape,
  data,
});

const outputOf = (outputs: Readonly<Record<string, Tensor>>, name: string, where: string) => {
  if (!Object.hasOwn(outputs, name)) throw new Error(`${OWNER}: ${where} returned no '${name}'`);
  const tensor = outputs[name];
  if (tensor.dtype !== "f32") throw new Error(`${OWNER}: ${where} output is ${tensor.dtype}`);
  return tensor.data;
};

/** Fails loudly on the first non-finite value of a stage output. */
const assertFinite = (values: Float32Array<ArrayBuffer>, where: string): void => {
  const broken = values.findIndex((value) => !Number.isFinite(value));
  if (broken !== -1) {
    throw new Error(`${OWNER}: ${where} element ${broken} is non-finite (${values[broken]})`);
  }
};

/**
 * The inputs of one `dit_text` run. NOTE: `export` is for the host tests that pin them against the
 * recipe's goldens (not on a public surface).
 */
export const mossTextStageInputs = (contract: MossDitContract, text: MossBranchText): RunInputs => {
  const rope = mossTextRopeTables(contract.ropeBase, text.tokens);
  const ropeShape = [1, 1, text.tokens, contract.ropeWidth];
  return {
    [MOSS_DIT_TEXT_INPUTS.textEmbeds]: f32Tensor(
      [1, text.tokens, MOSS_TEXT_WIDTH],
      text.embeddings.data,
    ),
    [MOSS_DIT_TEXT_INPUTS.ropeCos]: f32Tensor(ropeShape, rope.cos),
    [MOSS_DIT_TEXT_INPUTS.ropeSin]: f32Tensor(ropeShape, rope.sin),
  };
};

/** One branch's refined text (`[T, D]` — the `dit_text` output read back). */
export type MossRefinedText = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly tokens: number;
};

/**
 * Stage 1 (module doc): the `dit_text` Session runs each branch once and is closed. `observe`
 * receives the Session's diagnostics after each run.
 */
export const runMossTextStage = async (
  state: MossDitStageState,
  plan: MossDitPlan,
  observe?: (diagnostics: SessionDiagnostics) => void,
): Promise<MossBranches<MossRefinedText>> => {
  const { gpu, components, contract } = state;
  await assertHeadroom(
    gpu,
    mossDitStageNeeds({ ...state, limits: gpu.limits }, plan).text,
    MOSS_DIT_TEXT,
    plan,
  );
  // f32 compute in every preset (plan item 15-11): none of the preset's knobs, now or in step 5.
  const session = await components.ditText.createSession(gpu, {});
  let failure: { readonly error: unknown } | undefined;
  try {
    const run = async (text: MossBranchText, label: string): Promise<MossRefinedText> => {
      const outputs = await session.run(mossTextStageInputs(contract, text));
      observe?.(session.diagnostics());
      const data = outputOf(outputs, contract.outputs.ditText, MOSS_DIT_TEXT);
      if (data.length !== text.tokens * contract.width) {
        throw new Error(
          `${OWNER}: ${MOSS_DIT_TEXT} returned ${data.length} values for T = ${text.tokens}`,
        );
      }
      assertFinite(data, `the ${label} ${MOSS_DIT_TEXT} output`);
      return { data, tokens: text.tokens };
    };
    const positive = await run(plan.texts.positive, "positive");
    const negative =
      plan.texts.negative === undefined ? undefined : await run(plan.texts.negative, "negative");
    return { positive, negative };
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    // MUST: a failure to close does not overwrite the main failure (`disposeSteps`).
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      () => session.dispose(),
    ]);
  }
};

/** One branch of the open pair: its Session and the inputs fixed for the generation. */
type BranchRunner = {
  readonly session: Session;
  readonly output: string;
  readonly tokens: number;
  readonly sequence: number;
  readonly context: ResidentTensor;
  readonly isImage: Tensor;
  readonly ropeCos: Tensor;
  readonly ropeSin: Tensor;
};

/** The open DiT pair (valid inside {@link withMossDitPair}'s body only). */
export type MossDitPair = {
  /**
   * One step's two forwards on latents `[C, H, W]` at a training-grid index: the head rows
   * (upstream's samples `[C, H, W]`) of each branch.
   */
  readonly predict: (
    latents: Float32Array<ArrayBuffer>,
    gridIndex: number,
  ) => Promise<MossBranches<Float32Array<ArrayBuffer>>>;
  /** Each Session's diagnostics (the backing build count is the per-Session observation point). */
  readonly diagnostics: () => MossBranches<SessionDiagnostics>;
};

const lentWeights = (
  lender: Session,
  names: readonly string[],
): Readonly<Record<string, SharedWeight>> =>
  Object.fromEntries(names.map((name) => [name, lender.exportWeight(name)]));

/**
 * Stage 2 (module doc): opens the DiT pair for `plan` with the refined texts of
 * {@link runMossTextStage}, runs `body` and closes everything (borrower before lender, then the
 * resident text contexts) — also when `body` throws.
 */
export const withMossDitPair = async <T>(
  state: MossDitStageState,
  plan: MossDitPlan,
  refined: MossBranches<MossRefinedText>,
  body: (pair: MossDitPair) => Promise<T>,
): Promise<T> => {
  const { gpu, components, contract } = state;
  const { cells } = plan;
  if (refined.positive.tokens !== plan.texts.positive.tokens) {
    throw new Error(`${OWNER}: the positive refined text has ${refined.positive.tokens} tokens`);
  }
  if (refined.negative?.tokens !== plan.texts.negative?.tokens) {
    throw new Error(
      `${OWNER}: the negative refined text (${refined.negative?.tokens ?? "none"} tokens) and the` +
        ` plan (${plan.texts.negative?.tokens ?? "none"}) disagree`,
    );
  }
  await assertHeadroom(
    gpu,
    mossDitStageNeeds({ ...state, limits: gpu.limits }, plan).pair,
    `${MOSS_DIT} + ${MOSS_DIT_SKIP22}`,
    plan,
  );
  let lender: Session | undefined;
  let borrower: Session | undefined;
  const residents: ResidentTensor[] = [];
  let failure: { readonly error: unknown } | undefined;
  try {
    lender = await components.dit.createSession(gpu, {});
    const open = async (
      session: Session,
      output: string,
      text: MossRefinedText,
      label: string,
    ): Promise<BranchRunner> => {
      const sequence = mossImageTokens(cells) + text.tokens;
      const context = await gpu.createResident(
        contextBytes(sequence, contract.width),
        `moss ${label} text_context`,
      );
      residents.push(context);
      context.write(mossTextContext(text.data, contract.width, cells, text.tokens));
      const rope = mossRopeTables(contract.ropeBase, cells, text.tokens);
      const ropeShape = [1, 1, sequence, contract.ropeWidth];
      return {
        session,
        output,
        tokens: text.tokens,
        sequence,
        context,
        isImage: { dtype: "bool", shape: [1, sequence, 1], data: mossIsImage(cells, text.tokens) },
        ropeCos: f32Tensor(ropeShape, rope.cos),
        ropeSin: f32Tensor(ropeShape, rope.sin),
      };
    };
    const positive = await open(lender, contract.outputs.dit, refined.positive, "positive");
    let negative: BranchRunner | undefined;
    if (refined.negative !== undefined) {
      borrower = await components.ditSkip22.createSession(gpu, {
        sharedWeights: lentWeights(lender, contract.borrowed),
      });
      negative = await open(borrower, contract.outputs.ditSkip22, refined.negative, "negative");
    }
    const channels = contract.latentChannels;
    const run = async (
      branch: BranchRunner,
      latents: Float32Array<ArrayBuffer>,
      proj: Float32Array<ArrayBuffer>,
      label: string,
    ): Promise<Float32Array<ArrayBuffer>> => {
      const inputs: RunInputs = {
        [MOSS_DIT_INPUTS.latentTokens]: f32Tensor(
          [1, branch.sequence, channels],
          mossLatentTokens(latents, channels, cells, branch.tokens),
        ),
        [MOSS_DIT_INPUTS.textContext]: branch.context,
        [MOSS_DIT_INPUTS.isImage]: branch.isImage,
        [MOSS_DIT_INPUTS.ropeCos]: branch.ropeCos,
        [MOSS_DIT_INPUTS.ropeSin]: branch.ropeSin,
        [MOSS_DIT_INPUTS.timestepsProj]: f32Tensor([1, contract.projWidth], proj),
      };
      const outputs = await branch.session.run(inputs);
      const rows = outputOf(outputs, branch.output, label);
      const sample = mossHeadRows(rows, channels, cells, branch.tokens);
      assertFinite(sample, `the ${label} DiT output`);
      return sample;
    };
    const pair: MossDitPair = {
      predict: async (latents, gridIndex) => {
        const proj = mossTimestepsProj(contract.trainingGrid, gridIndex);
        const cond = await run(positive, latents, proj, MOSS_DIT);
        const uncond =
          negative === undefined ? undefined : await run(negative, latents, proj, MOSS_DIT_SKIP22);
        return { positive: cond, negative: uncond };
      },
      diagnostics: () => ({
        positive: positive.session.diagnostics(),
        negative: negative?.session.diagnostics(),
      }),
    };
    return await body(pair);
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    // MUST: the borrower closes before its lender (the runtime refuses to dispose a lender while a
    // borrower is alive), Sessions before the residents they bind, and a failure to close does not
    // overwrite the main failure.
    const opened = { borrower, lender };
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      () => opened.borrower?.dispose(),
      () => opened.lender?.dispose(),
      ...residents.map((resident) => () => resident.dispose()),
    ]);
  }
};

/**
 * Upstream's CFG of one step on the two f32 samples (module doc "CFG"): element-wise
 * `uncond + f32(scale) · (cond − uncond)`, every operation rounded to f32.
 */
export const mossClassifierFreeGuidance = (
  cond: Float32Array<ArrayBuffer>,
  uncond: Float32Array<ArrayBuffer>,
  scale: number,
): Float32Array<ArrayBuffer> => {
  if (cond.length !== uncond.length) {
    throw new Error(`${OWNER}: CFG of ${cond.length} and ${uncond.length} values`);
  }
  if (!Number.isFinite(scale)) throw new Error(`${OWNER}: CFG scale ${scale} is not finite`);
  const f32 = Math.fround;
  const factor = f32(scale);
  const out = new Float32Array(cond.length);
  for (let index = 0; index < cond.length; index += 1) {
    out[index] = f32(uncond[index] + f32(factor * f32(cond[index] - uncond[index])));
  }
  return out;
};
