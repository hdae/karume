/**
 * Wan's DiT stage — the S-shaped graph contract ({@link ditContract}) and the denoise loop
 * (patchify → DiT → unpatchify → CFG → UniPC — {@link runWanDenoise}).
 *
 * The stage order and Session lifetimes (text → DiT → VAE, each created and closed one stage at a
 * time) are defined by the doc at the top of `./pipeline.ts`; this module only runs the one DiT stage.
 * Shared by the Wan2.1 / 2.2 classes (ADR 0121 decision 10).
 * `owner` is the message prefix (`"wan.Pipeline"` for Wan2.1, `"wan.Ti2vPipeline"` for Wan2.2).
 *
 * NOTE: the public types (`GenerateEvent` / `RunComponent`) are taken from `./pipeline.ts` with
 * `import type` (type-only references are erased, so there is no import cycle). Even with the
 * `vae_encoder` stage added at stage 9 (I2V) they stay in `./pipeline.ts` — type-only references cause
 * no cycle, so there is no reason to move them to a separate module.
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state —
 * CLAUDE.md).
 */

import type { GpuContext, RunInputs, SessionDiagnostics, SessionOptions } from "@karume/runtime";

import { settleAbort } from "../concurrency/abort.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import type { GraphOwner, ModelComponent } from "../hub/components.ts";
import type { GenerateEvent, RunComponent } from "./pipeline.ts";
import type { WanDitInputForm } from "./descriptor.ts";
import type { WanGenerationKnobs } from "./plan.ts";
import {
  patchifyLatents,
  unpatchifyTokens,
  type WanPatchGeometry,
  type WanTokenGrid,
  wanTokenGrid,
  wanTokenWidth,
} from "./dit-tokens.ts";
import { type WanRopeBase, wanRopeTables, wanRopeWidth } from "./dit-rope.ts";
import { timestepsProj } from "./dit-timestep.ts";
import { asF32, assertDims, firstNonFinite, valueShape } from "./graph-io.ts";
import { WanRandn } from "./random.ts";
import { WAN_UNIPC_CONFIG, wanClassifierFreeGuidance, WanUniPcSampler } from "./scheduler.ts";

/** The component key (the series' graph name = the weights name of the step 7 manifest). */
export const TRANSFORMER = "transformer";

/** The asset name of the RoPE base table declared by the `transformer` container (role `rope-base`). */
export const ROPE_BASE = "rope_base";

/** Input names of the DiT's S-shaped graph (forward argument names of recipe `wan/export_dit.py`). */
const DIT_TOKENS = "tokens";
const DIT_TIMESTEPS_PROJ = "timesteps_proj";
export const DIT_CONTEXT = "encoder_hidden_states";
const DIT_ROPE_COS = "rope_cos";
const DIT_ROPE_SIN = "rope_sin";
/**
 * The two extra inputs of the TI2V DiT (named as the tail of `INPUT_NAMES` in recipe
 * `wan/ti2v_export_dit.py`).
 */
const DIT_TIMESTEPS_PROJ_CONDITION = "timesteps_proj_condition";
const DIT_CONDITION_MASK = "condition_mask";

/**
 * A form outside the exhaustive set (a descriptor or internal state holding an out-of-type value —
 * not silently run as t2v).
 */
const unknownDitInputForm = (form: never, owner: string): Error =>
  new Error(`${owner}: unknown DiT input form '${String(form)}' ("t2v" / "ti2v")`);

/** The set of graph input names per DiT input form (an unknown form fails loudly). */
const ditInputNames = (form: WanDitInputForm, owner: string): readonly string[] => {
  const t2v = [DIT_TOKENS, DIT_TIMESTEPS_PROJ, DIT_CONTEXT, DIT_ROPE_COS, DIT_ROPE_SIN];
  switch (form) {
    case "t2v":
      return t2v;
    case "ti2v":
      return [...t2v, DIT_TIMESTEPS_PROJ_CONDITION, DIT_CONDITION_MASK];
    default:
      throw unknownDitInputForm(form, owner);
  }
};

/**
 * The DiT patch (the stride is the upstream transformer config's `patch_size [1, 2, 2]` — an
 * architecture constant). The channel count comes from the VAE chunk graph declaration (axis 0 of
 * `latent` — `WanVaeChunkLayout.latentChannels`) (16 for Wan2.1). The last dimension of the graph's
 * `tokens` (`C·1·2·2` — 64 for Wan2.1) is matched by {@link ditContract} at construction.
 *
 * NOTE: `export` is for the family admission and for tests that pin the checks without a GPU (not on
 * the `mod.ts` / subpath surface — ADR 0008).
 */
export const wanDitPatch = (latentChannels: number): WanPatchGeometry => ({
  channels: latentChannels,
  patchFrames: 1,
  patchHeight: 2,
  patchWidth: 2,
});

/** The declaration of a graph input (fails loudly if absent). */
const graphInput = (
  owner: string,
  model: GraphOwner,
  name: string,
): GraphOwner["graph"]["inputs"][number] => {
  const spec = model.graph.inputs.find((input) => input.name === name);
  if (spec === undefined) throw new Error(`${owner}: transformer graph input '${name}' is missing`);
  return spec;
};

/** The shape of a graph input (fails loudly if absent). */
const inputShape = (owner: string, model: GraphOwner, name: string): readonly (number | string)[] =>
  graphInput(owner, model, name).shape;

/** The dtype and shape of a graph input (fails loudly if they differ from what the host builds). */
const assertInput = (
  owner: string,
  model: GraphOwner,
  name: string,
  dtype: GraphOwner["graph"]["inputs"][number]["dtype"],
  expected: readonly (number | string)[],
): void => {
  const spec = graphInput(owner, model, name);
  if (spec.dtype !== dtype) {
    throw new Error(
      `${owner}: transformer graph input '${name}' has dtype ${spec.dtype}, not ${dtype}`,
    );
  }
  assertDims(owner, spec.shape, expected, `'${name}'`);
};

/** A static dimension (fails loudly on a symbolic dimension). */
const staticDim = (
  owner: string,
  dims: readonly (number | string)[],
  axis: number,
  where: string,
): number => {
  const dim = dims.at(axis);
  if (typeof dim !== "number") throw new Error(`${owner}: ${where} axis ${axis} is not static`);
  return dim;
};

/** The DiT graph contract confirmed at construction. */
export type DitContract = {
  /** The name of the graph output (`[1, S, C·1·2·2]`). */
  readonly output: string;
  /** W of `timesteps_proj [1, W]`. */
  readonly projWidth: number;
  /** rows of `encoder_hidden_states [1, rows, width]` (the row count to pad with zeros up to). */
  readonly contextRows: number;
  readonly contextWidth: number;
  /**
   * The patch used for matching the declaration ({@link runWanDenoise} patchifies / unpatchifies with
   * it — the same value is not derived twice, once for matching and once for the loop).
   */
  readonly patch: WanPatchGeometry;
  /**
   * The input form matched against the declaration ({@link runWanDenoise} decides from it whether to
   * build the condition inputs).
   */
  readonly form: WanDitInputForm;
};

/**
 * Matches the DiT graph declaration against the inputs the host builds (`patch`, the RoPE base table).
 * The channel count of `patch` comes from the VAE declaration ({@link wanDitPatch}), so a channel
 * mismatch between the VAE latent and the DiT's `tokens` also fails here.
 *
 * MUST: fail at construction. The host preprocessing builds with its own constants, so even if the
 * graph was exported with other dimensions the host side runs to the end, and the failure would come
 * only from the Session's shape check after the DiT weights are uploaded. On the fetch surface it is
 * called in the family admission (before the weight parts are fetched) — matching against the
 * embedding asset (`assertEmbedsFitContext` in `text-stage.ts`) waits until the asset bytes arrive.
 *
 * MUST: match not only the last dimension but also the rank, batch (B = 1 — decision 5) and the
 * variable S. The host builds `tokens [1, S, C·1·2·2]` and `rope_cos / rope_sin [1, S, 1, w]` and reads
 * the output as `[1, S, C·1·2·2]` ({@link runWanDenoise}), so a declaration with batch 2, a fixed S or
 * a different rank would also get as far as the Session's shape check. S changes with the size and
 * frame count, so it is a symbolic dimension, and all four carry **the same symbol** (IR symbols have
 * no bounds, so no bound matching is needed).
 *
 * MUST: the set of graph input names matches exactly what `form` (the generation descriptor's
 * `ditInputForm`) expects. The host builds only the expected inputs, so an extra input (a sixth one in
 * a 2.1 declaration, or a TI2V DiT opened as 2.1) would get as far as the Session's input check. This
 * check comes **after** the checks of the existing five — it does not change the message for an asset
 * missing an input (the "'X' is missing" wording). For TI2V the dtype and shape of the two extra inputs are checked too
 * (the condition-side time is f32 with the same `[1, W]` as `timesteps_proj`; the condition mask is a
 * bool `[1, S, 1]` with the same symbol S as `tokens`).
 *
 * NOTE: `export` is for the family admission and for tests that pin the checks without a GPU (not on
 * the `mod.ts` / subpath surface — ADR 0008).
 */
export const ditContract = (
  transformer: GraphOwner,
  ropeBase: WanRopeBase,
  patch: WanPatchGeometry,
  form: WanDitInputForm,
  owner: string,
): DitContract => {
  // An unknown form fails with the same message whatever the asset contains (the set matching itself
  // comes after the five checks).
  const expected = ditInputNames(form, owner);
  const tokenWidth = wanTokenWidth(patch);
  const tokens = inputShape(owner, transformer, DIT_TOKENS);
  const sequence = tokens.at(1);
  if (typeof sequence !== "string") {
    throw new Error(
      `${owner}: '${DIT_TOKENS}' axis 1 is not a symbolic dimension (${String(sequence)}) — ` +
        "the host passes a different S for each size and frame count",
    );
  }
  assertDims(owner, tokens, [1, sequence, tokenWidth], `'${DIT_TOKENS}'`);
  const [output] = transformer.graph.outputs;
  if (transformer.graph.outputs.length !== 1) {
    throw new Error(
      `${owner}: transformer has ${transformer.graph.outputs.length} outputs (expected one S-shaped output)`,
    );
  }
  assertDims(
    owner,
    valueShape(owner, transformer, output),
    [1, sequence, tokenWidth],
    `transformer output '${output}'`,
  );
  const ropeWidth = wanRopeWidth(ropeBase);
  for (const name of [DIT_ROPE_COS, DIT_ROPE_SIN]) {
    assertDims(
      owner,
      inputShape(owner, transformer, name),
      [1, sequence, 1, ropeWidth],
      `'${name}'`,
    );
  }
  const proj = inputShape(owner, transformer, DIT_TIMESTEPS_PROJ);
  const projWidth = staticDim(owner, proj, 1, DIT_TIMESTEPS_PROJ);
  assertDims(owner, proj, [1, projWidth], `'${DIT_TIMESTEPS_PROJ}'`);
  const context = inputShape(owner, transformer, DIT_CONTEXT);
  const contextRows = staticDim(owner, context, 1, DIT_CONTEXT);
  const contextWidth = staticDim(owner, context, 2, DIT_CONTEXT);
  assertDims(owner, context, [1, contextRows, contextWidth], `'${DIT_CONTEXT}'`);
  const declared = transformer.graph.inputs.map((input) => input.name);
  const sortedDeclared = [...declared].sort();
  const sortedExpected = [...expected].sort();
  if (
    sortedDeclared.length !== sortedExpected.length ||
    sortedDeclared.some((name, index) => name !== sortedExpected[index])
  ) {
    throw new Error(
      `${owner}: transformer graph inputs are [${declared.join(", ")}]` +
        ` (expected: [${expected.join(", ")}] — input form '${form}')`,
    );
  }
  switch (form) {
    case "t2v":
      break;
    case "ti2v":
      assertInput(owner, transformer, DIT_TIMESTEPS_PROJ_CONDITION, "f32", [1, projWidth]);
      assertInput(owner, transformer, DIT_CONDITION_MASK, "bool", [1, sequence, 1]);
      break;
    default:
      throw unknownDitInputForm(form, owner);
  }
  return { output, projWidth, contextRows, contextWidth, patch, form };
};

/** The DiT's context input (positive padded to `[rows, width]`, and CFG's uncond side). */
export type WanContexts = {
  readonly positive: Float32Array<ArrayBuffer>;
  readonly negative: Float32Array<ArrayBuffer> | undefined;
};

/**
 * The built material the DiT stage reads (this stage's part of the pipeline's internal state — taken
 * structurally).
 */
type WanDenoiseState = {
  readonly gpu: GpuContext;
  readonly sessionOptions: SessionOptions;
  readonly transformer: ModelComponent;
  readonly ropeBase: WanRopeBase;
  readonly dit: DitContract;
  readonly onRunDiagnostics?: (component: RunComponent, diagnostics: SessionDiagnostics) => void;
};

/**
 * The two condition inputs of the TI2V DiT (ADR 0121 decision 3). The condition-side time has the
 * same `projShape` as the generation side. For T2V the condition mask is all false and the
 * condition-side time is the same array as the generation side. For I2V the tokens of the first
 * latent frame are true and the condition-side time is the proj at t = 0 ({@link wanConditionMask},
 * {@link runWanDenoise}).
 */
type DitConditionInputs = {
  readonly proj: Float32Array<ArrayBuffer>;
  readonly mask: Uint32Array<ArrayBuffer>;
  readonly maskShape: readonly number[];
};

/**
 * The inputs of one DiT forward (the S-shaped graph's inputs — 5 for t2v, 7 for ti2v with the two of
 * `condition` added). The arrays the host built are passed as is (not copied — a borrow that is not
 * modified until the run settles). When `condition` is undefined only the five are passed (keys,
 * order and arrays the same as before the condition inputs were added — the bytes that reach the
 * Wan2.1 Session do not change).
 *
 * MUST: this is the only place that builds the DiT inputs (ADR 0121 decision 10 — shared by 2.1 and
 * 2.2).
 *
 * NOTE: `export` is for tests that pin how the inputs are built without a GPU (not on the `mod.ts` /
 * subpath surface — ADR 0008).
 */
export const ditInputs = (input: {
  readonly tokens: Float32Array<ArrayBuffer>;
  readonly tokenShape: readonly number[];
  readonly proj: Float32Array<ArrayBuffer>;
  readonly projShape: readonly number[];
  readonly context: Float32Array<ArrayBuffer>;
  readonly contextShape: readonly number[];
  readonly rope: {
    readonly cos: Float32Array<ArrayBuffer>;
    readonly sin: Float32Array<ArrayBuffer>;
  };
  readonly ropeShape: readonly number[];
  readonly condition: DitConditionInputs | undefined;
}): RunInputs => ({
  [DIT_TOKENS]: { dtype: "f32", shape: input.tokenShape, data: input.tokens },
  [DIT_TIMESTEPS_PROJ]: { dtype: "f32", shape: input.projShape, data: input.proj },
  [DIT_CONTEXT]: { dtype: "f32", shape: input.contextShape, data: input.context },
  [DIT_ROPE_COS]: { dtype: "f32", shape: input.ropeShape, data: input.rope.cos },
  [DIT_ROPE_SIN]: { dtype: "f32", shape: input.ropeShape, data: input.rope.sin },
  ...(input.condition === undefined
    ? {}
    : {
        [DIT_TIMESTEPS_PROJ_CONDITION]: {
          dtype: "f32",
          shape: input.projShape,
          data: input.condition.proj,
        },
        [DIT_CONDITION_MASK]: {
          dtype: "bool",
          shape: input.condition.maskShape,
          data: input.condition.mask,
        },
      }),
});

/**
 * The condition mask (ADR 0121 decision 3 — built once before the loop). A DiT of input form "t2v" has
 * no condition inputs, so it is undefined (fails loudly if a condition latent is passed). For "ti2v",
 * a text-only request (T2V — an axis separate from the input form) gives all false, and I2V
 * (`conditioned`) sets true the tokens of the first latent frame — the `rows·cols` tokens that come
 * first and contiguous under the token index `(f·H' + h)·W' + w` (880 at 1280×704).
 *
 * MUST: I2V is built only when the temporal patch is 1 — it assumes the frame of the first tokens
 * matches the first latent frame (upstream diffusers' `first_frame_mask[0][0][:, ::2, ::2]` assumes the
 * same).
 *
 * NOTE: `export` is for tests that pin the mask's shape without a GPU (not on the `mod.ts` / subpath
 * surface — ADR 0008).
 */
export const wanConditionMask = (
  form: WanDitInputForm,
  grid: WanTokenGrid,
  patch: WanPatchGeometry,
  conditioned: boolean,
  owner: string,
): Uint32Array<ArrayBuffer> | undefined => {
  switch (form) {
    case "t2v":
      if (conditioned) {
        throw new Error(
          `${owner}: a DiT of input form 't2v' does not accept a condition latent (it has no condition mask input)`,
        );
      }
      return undefined;
    case "ti2v": {
      const mask = new Uint32Array(grid.count);
      if (conditioned) {
        if (patch.patchFrames !== 1) {
          throw new Error(
            `${owner}: with temporal patch ${patch.patchFrames} the first latent frame does not align with the token frames` +
              " (the I2V condition mask assumes temporal patch 1)",
          );
        }
        mask.fill(1, 0, grid.rows * grid.cols);
      }
      return mask;
    }
    default:
      throw unknownDitInputForm(form, owner);
  }
};

/**
 * A copy of the model input with its first latent frame replaced by the condition latent (upstream
 * diffusers' `expand_timesteps` branch `(1 − m)·condition + m·latents` — m is 0 for the first frame
 * and 1 elsewhere. ADR 0121 decision 11). `latents` is `[C, F, H, W]`, `condition` the single-frame
 * `[C, 1, H, W]`; as upstream, the value of the formula broadcast over all frames is written element
 * by element.
 *
 * The formula is computed literally (not just copying the first frame): m is 0 or 1, so each term is
 * the condition, the latent or a signed zero, and the sum involves no rounding — it gives the same
 * values as the f32 upstream down to signed zeros (a plain copy would disagree with upstream's +0 for
 * elements where the latent is -0).
 *
 * NOTE: `export` is for tests that pin the replacement without a GPU (not on the `mod.ts` / subpath
 * surface — ADR 0008).
 */
export const withWanFirstFrameCondition = (
  latents: Float32Array<ArrayBuffer>,
  condition: Float32Array<ArrayBuffer>,
  latentShape: readonly [number, number, number, number],
): Float32Array<ArrayBuffer> => {
  const [channels, frames, height, width] = latentShape;
  const plane = height * width;
  if (latents.length !== channels * frames * plane || condition.length !== channels * plane) {
    throw new Error(
      `Condition replacement: latent ${latents.length} elements, condition ${condition.length} elements do not match the latent shape ` +
        `[${latentShape.join(", ")}] (the condition is 1 frame)`,
    );
  }
  const out = new Float32Array(latents.length);
  for (let channel = 0; channel < channels; channel += 1) {
    const from = channel * plane;
    for (let frame = 0; frame < frames; frame += 1) {
      // Upstream's `first_frame_mask` (0 for the first frame).
      const keep = frame === 0 ? 0 : 1;
      const base = (channel * frames + frame) * plane;
      for (let index = 0; index < plane; index += 1) {
        out[base + index] = (1 - keep) * condition[from + index] + keep * latents[base + index];
      }
    }
  }
  return out;
};

/**
 * The DiT stage (creates the Session, runs `steps` iterations and closes it — `end` comes after the
 * close).
 *
 * Passing `condition` (the I2V condition latent `[C, 1, H, W]` — normalized, finite) conditions in the
 * form of upstream diffusers' `expand_timesteps` branch (ADR 0121 decision 11): at each step only the
 * first latent frame of the model input is replaced ({@link withWanFirstFrameCondition}), UniPC
 * advances the unreplaced latent, and after the loop the latent is replaced once more before
 * returning. The condition mask is true for the first frame's tokens ({@link wanConditionMask}); the
 * condition-side time is the proj at t = 0, built once. What `copyLatents` of `denoise-step` returns
 * is the scheduler state (before the replacement — the same value as diffusers' callback). Compared
 * with the official Wan2.2 form (replacing initially and after each step), only the first frame of the
 * intermediate latents differs and the final output has the same values (UniPC updates element-wise
 * and the model input is the same in both forms — the replaced elements do not flow into other
 * elements).
 *
 * When `condition` is undefined (a text-only request), only the same arrays as before run (the
 * condition mask is all false, the condition-side time is the same array as the generation side, the
 * model input is the latent itself — the T2V sha rows do not change).
 *
 * MUST: check the finiteness of the latent after each step's update (an O(N) host scan — about 2.1
 * million elements per step at 81 frames). Passing a non-finite latent silently to the next step and
 * the VAE stage would let the clamp after the VAE turn ±Inf into ±1, beyond detection.
 *
 * MUST (caller's precondition): when `plan.initial` is latents, `plan.initial.data` is an array the
 * caller copied before the entry check and that only the caller owns — this function uses it
 * without copying, and UniPC keeps the step-0 latent by reference. Copying is the responsibility of the
 * caller (`ownWanRequestLatents` in `family.ts`, called at the entry of each class's `generate`) and is
 * enforced by neither the type nor the plan. Passing an outside caller's array directly lets
 * modifications during the run (including NaN that never passed the check) reach the corrector.
 */
export const runWanDenoise = async (
  state: WanDenoiseState,
  plan: WanGenerationKnobs,
  contexts: WanContexts,
  condition: Float32Array<ArrayBuffer> | undefined,
  emit: (event: GenerateEvent) => Promise<void>,
  signal: AbortSignal | undefined,
  owner: string,
): Promise<Float32Array<ArrayBuffer>> => {
  const { dit } = state;
  const { patch } = dit;
  const { latentShape, schedule } = plan;
  const grid = wanTokenGrid(latentShape, patch);
  const rope = wanRopeTables(state.ropeBase, grid);
  const tokenShape = [1, grid.count, wanTokenWidth(patch)];
  const ropeShape = [1, grid.count, 1, wanRopeWidth(state.ropeBase)];
  const contextShape = [1, dit.contextRows, dit.contextWidth];
  const projShape = [1, dit.projWidth];
  // An unknown form, or a form that takes no condition, fails here (before the Session is created).
  const conditionMask = wanConditionMask(dit.form, grid, patch, condition !== undefined, owner);
  const maskShape = [1, grid.count, 1];
  // The condition-side time (upstream's `first_frame_mask · t` — the first frame's tokens get t = 0).
  // Independent of the step, so built once.
  const conditionProj = condition === undefined ? undefined : timestepsProj(0, dit.projWidth);
  const { positive, negative } = contexts;
  const elements = latentShape.reduce((product, dim) => product * dim, 1);
  // The latents of `plan.initial` are an owned array copied at the entry of the class's `generate`
  // (`ownWanRequestLatents`), so they are not copied here (UniPC keeps the step-0 latent by reference — passing
  // the caller's array directly would let modifications during the run reach the corrector).
  let current: Float32Array<ArrayBuffer> =
    plan.initial.kind === "latents"
      ? plan.initial.data
      : new WanRandn(plan.initial.seed).normals(elements);
  const observe = state.onRunDiagnostics;

  await emit({ kind: "stage", component: "transformer", at: "start" });
  const session = await state.transformer.createSession(state.gpu, state.sessionOptions);
  let failure: { readonly error: unknown } | undefined;
  try {
    const predict = async (
      tokens: Float32Array<ArrayBuffer>,
      proj: Float32Array<ArrayBuffer>,
      context: Float32Array<ArrayBuffer>,
    ): Promise<Float32Array<ArrayBuffer>> => {
      // For T2V the condition-side time is the same array as the generation side (decision 3 — both
      // only read it, a borrow).
      const conditionInputs =
        conditionMask === undefined
          ? undefined
          : { proj: conditionProj ?? proj, mask: conditionMask, maskShape };
      const outputs = await session.run(
        ditInputs({
          tokens,
          tokenShape,
          proj,
          projShape,
          context,
          contextShape,
          rope,
          ropeShape,
          condition: conditionInputs,
        }),
      );
      observe?.("transformer", session.diagnostics());
      return unpatchifyTokens(asF32(outputs[dit.output], "DiT output"), latentShape, patch);
    };
    const sampler = new WanUniPcSampler(schedule, WAN_UNIPC_CONFIG);
    for (let index = 0; index < plan.steps; index += 1) {
      // Before each step (a step's two forwards are indivisible — cancellation takes effect before
      // the next step).
      await settleAbort(signal);
      const timestep = schedule.timesteps[index];
      const proj = timestepsProj(timestep, dit.projWidth);
      // CFG runs uncond → cond as two sequential passes (B = 1 — decision 5). Same latent, so one
      // patchify. The combination happens on the host. For I2V only the model input has its first
      // frame replaced (UniPC gets the unreplaced `current`).
      const modelInput =
        condition === undefined
          ? current
          : withWanFirstFrameCondition(current, condition, latentShape);
      const tokens = patchifyLatents(modelInput, latentShape, patch);
      const uncond = negative === undefined ? undefined : await predict(tokens, proj, negative);
      const cond = await predict(tokens, proj, positive);
      const velocity =
        uncond === undefined ? cond : wanClassifierFreeGuidance(cond, uncond, plan.guidance);
      current = sampler.step(velocity, current);
      const broken = firstNonFinite(current);
      if (broken !== -1) {
        throw new Error(
          `${owner}: after the update of step ${index + 1}/${plan.steps}, latent element ${broken} is non-finite` +
            ` (${current[broken]}) — the DiT output, the CFG combination or the UniPC update overflowed` +
            " (not passed to the VAE stage)",
        );
      }
      const snapshot = current;
      await emit({
        kind: "denoise-step",
        step: index + 1,
        steps: plan.steps,
        timestep,
        copyLatents: () => ({ data: Float32Array.from(snapshot), shape: [...latentShape] }),
      });
    }
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    // MUST: a failure to close does not overwrite the main failure (a run failure, a throw from
    // `onEvent`, the non-finite check) — the same form as the VAE stage (the doc of `disposeSteps`).
    // If both fail, an AggregateError with the main failure first.
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      () => session.dispose(),
    ]);
  }
  // For I2V the latent is replaced once after the loop before it goes to the VAE stage (as upstream
  // diffusers — the replaced elements come only from the finite condition and latents that passed the
  // check, so finiteness holds).
  const latents =
    condition === undefined ? current : withWanFirstFrameCondition(current, condition, latentShape);
  await emit({ kind: "stage", component: "transformer", at: "end" });
  return latents;
};
