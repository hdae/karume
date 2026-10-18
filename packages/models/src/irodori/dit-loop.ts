/**
 * The 2 paths of the denoise loop (step 7) and the step before it (step 6' — run the context K/V
 * projection `dit_context` once per generation): {@link runDitLoopOnHost}, which goes down to the
 * host on every forward, and the GPU-resident path {@link runDitLoopResident}, which bundles the
 * whole loop into 1 batch. The caller builds the material of one loop ({@link DitLoop}); this
 * module only runs the material it receives.
 *
 * The description of the 10 steps (what runs in which order) is canonical in the doc at the top of
 * {@link "./pipeline.ts"}. **Choosing which path runs does not happen here either** — the choice
 * is in `generateLatent` of `./pipeline.ts`, where a device with timing enabled (ADR 0021 — it
 * cannot open a batch) and an `onEvent` subscription pick the host path.
 *
 * MUST: do not fold the 2 paths into a generic step executor. The bit identity of the outputs holds
 * only because "the queued operations have the same form, one operation at a time" (the MUST of
 * {@link runDitLoopResident}); if sharing the code moves the order on one side, the WAV sha256
 * check breaks. Step 6' is part of this: both paths "pass the 3 conditions through `dit_context`
 * once with the same Session options, and hand the output bytes to `dit` as resident tensors as
 * they are" (DECIDED: ADR 0114). Only the copy differs (a GPU copy, or a `write` through the
 * host); the K / V bytes `dit` reads are the same.
 *
 * MUST: every module is free of side effects (no execution at import time, no global mutable state
 * — CLAUDE.md).
 */

import { disposeSteps } from "../session/dispose-steps.ts";
import {
  type BatchScope,
  createSessionFromContainer,
  type GpuContext,
  type IrDeclaration,
  openMemoryContainer,
  type ResidentTensor,
  type Session,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";

import type { IrodoriSegment } from "./host/mask.ts";
import type { SequencePlan } from "./host/round.ts";
import { type CfgVariant, combineCfg, eulerStep } from "./host/sampler.ts";
import {
  COMBINE_INPUTS,
  COMBINE_OUTPUT,
  combineGraph,
  EULER_INPUTS,
  EULER_OUTPUT,
  eulerGraph,
} from "./host/sampler-graph.ts";
import { timestepEmbedding } from "./host/t-embed.ts";
import type { GeneratedLatent, IrodoriLatentSnapshot, IrodoriState } from "./pipeline.ts";
import {
  asF32,
  type EmitEvent,
  f32,
  observer,
  outputAt,
  outputNameAt,
  withStageSession,
} from "./stage.ts";
import type { ModelComponent } from "../hub/components.ts";
import { withSession } from "../session/with-session.ts";

/**
 * Builds the accessor that returns an intermediate latent (**lazy copy** — copies only on call).
 *
 * A consumer that subscribes only to progress pays no copy cost at all, and since the internal
 * array is never handed out, the accident "the subscriber holds on to the input of the next step"
 * cannot happen structurally.
 *
 * MUST: copy the array as of **creation**, not as of the call (bind it through the argument). The
 * `x` of the DiT loop is **replaced by a new array** every step, so this binding is exactly "the
 * latent of that step". Capturing the loop variable would return the latent of another step to a
 * subscriber that calls later.
 *
 * MUST: copy `shape` as well as `data`. Returned by reference, two copies taken in the same step
 * would share one array, and if the subscriber rewrote the first `shape`, the second copy would
 * silently claim a different shape. An alignment that makes the contract of the public event
 * surface (`IrodoriGenerateEvent.copyLatents` / `AnimaGenerateEvent.copyLatents`) one across
 * families (the same form as the same-named function on the anima side, `src/anima/pipeline.ts`).
 *
 * NOTE: `export` is for the test that pins the independence without a GPU (not exposed on `mod.ts`
 * / the subpath surface — ADR 0008).
 */
export const latentSnapshot =
  (latent: Float32Array<ArrayBuffer>, shape: readonly number[]): (() => IrodoriLatentSnapshot) =>
  (): IrodoriLatentSnapshot => ({ data: new Float32Array(latent), shape: [...shape] });

/**
 * Result of the latent step. `plan.targetSamples` is needed to cut the waveform, so it is **carried
 * along from where it was decided** (`GeneratedLatent` is the public surface, so it is not mixed
 * in).
 */
export type LatentStage = {
  readonly latent: GeneratedLatent;
  readonly plan: SequencePlan;
};

/** One CFG variant (the segment to drop, the strength, the mask with only that segment False). */
export type UncondVariant = {
  readonly segment: IrodoriSegment;
  readonly scale: number;
  readonly mask: Tensor;
};

/**
 * The material of one DiT loop (the 2 paths read **the same thing** — the host computation is
 * identical on both).
 */
export type DitLoop = {
  readonly frames: number;
  /** Initial noise `[frames × latentDim]`. */
  readonly initial: Float32Array<ArrayBuffer>;
  readonly schedule: Float32Array<ArrayBuffer>;
  readonly frequencies: Float32Array<ArrayBuffer>;
  /**
   * The 3 right-padded conditions (graph input name of `dit_context` → Tensor). Read only once, in
   * step 6'; they reach `dit` as resident tensors projected to K / V.
   */
  readonly conditions: Readonly<Record<string, Tensor>>;
  readonly condMask: Tensor;
  readonly uncondVariants: readonly UncondVariant[];
};

/** Result of the loop (the final latent and how many times `dit` ran). */
type DitLoopResult = {
  readonly x: Float32Array<ArrayBuffer>;
  readonly forwards: number;
};

/**
 * Allocator of resident tensors (puts what it allocates on the caller's cleanup — the caller
 * decides the order of release).
 */
type CreateResident = (byteLength: number, label: string) => Promise<ResidentTensor>;

/**
 * Opens one batch span, queues `body`, and always closes it whatever the outcome (the 2 spans of
 * the resident path — the step-6' span before the loop and the loop body — close the same way).
 *
 * MUST: always close the span (left open, the per-device lock never returns and every later run
 * waits forever).
 * MUST: do not overwrite a failure of `body` with a failure of `finish` ({@link disposeSteps}
 * carries both).
 */
const withBatch = async (
  gpu: GpuContext,
  body: (batch: BatchScope) => Promise<void>,
): Promise<void> => {
  const batch = await gpu.beginBatch();
  let failure: { readonly error: unknown } | undefined;
  try {
    await body(batch);
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      () => batch.finish(),
    ]);
  }
};

/**
 * Step 6' (host path): runs `dit_context` once with `run`, receives the outputs on the host and
 * writes them into resident tensors. A device with timing enabled cannot open a batch (ADR 0021),
 * so the resident path's batch before the loop cannot be used.
 *
 * The host copy (context K / V — 178MiB class) lives only inside this function (it is not held
 * during the loop).
 * MUST: the Session of `dit_context` is closed **before** the resident tensors are allocated (it is
 * already released when `withStageSession` returns) — the output slots and the resident tensors
 * are never in VRAM at the same time.
 */
const projectContextOnHost = async (
  state: IrodoriState,
  emit: EmitEvent,
  conditions: Readonly<Record<string, Tensor>>,
  createResident: CreateResident,
): Promise<Record<string, ResidentTensor>> => {
  const outputs = await withStageSession(
    state,
    emit,
    "dit-context",
    state.ditContext,
    state.ditSessionOptions,
    async (run) => await run(conditions),
  );
  const context: Record<string, ResidentTensor> = {};
  for (const { name, byteLength } of state.ditContextOutputs) {
    if (!Object.hasOwn(outputs, name)) {
      throw new Error(`dit_context の出力 '${name}' が実行結果に無い`);
    }
    const tensor = await createResident(byteLength, `irodori.${name}`);
    // `write` checks the size for an exact match (matching the byteLength admission derived against
    // the run result).
    tensor.write(outputs[name].data);
    context[name] = tensor;
  }
  return context;
};

/**
 * Step 6' (resident path): queues `dit_context` once in a batch before the loop, and copies the
 * outputs into resident tensors with `copyOutputs` (nothing goes down to the host).
 *
 * MUST: the destination resident tensors are allocated **before** the Session is opened, and the
 * Session is closed **before** `dit` is opened (it is already released on return). This keeps the
 * VRAM peak at the resident K / V — queued in the same batch as the loop, the output slots of
 * `dit_context` would stay until `finish` and the peak would double (a Session cannot be disposed
 * inside a batch — ADR 0054 addendum).
 */
const projectContextOnGpu = async (
  state: IrodoriState,
  conditions: Readonly<Record<string, Tensor>>,
  createResident: CreateResident,
): Promise<Record<string, ResidentTensor>> => {
  const context: Record<string, ResidentTensor> = {};
  for (const { name, byteLength } of state.ditContextOutputs) {
    context[name] = await createResident(byteLength, `irodori.${name}`);
  }
  // Observations are passed here per enqueue (the observation hook of `withSession` covers only
  // `run`).
  const observe = observer(state, "dit-context");
  await withSession(
    state.gpu,
    state.ditContext,
    state.ditSessionOptions,
    undefined,
    async (_run, session) => {
      await withBatch(state.gpu, async (batch) => {
        await session.enqueue(conditions, { batch, copyOutputs: context });
        if (observe !== undefined) observe(session.diagnostics());
      });
    },
  );
  return context;
};

/**
 * The loop that goes down to the host on every forward (`run` → readback → `combineCfg` +
 * `eulerStep` → re-upload).
 *
 * It is **the numeric source of truth**, and also the only path on a device with timing enabled
 * (`gpuTiming` — it cannot open the batch the resident path uses) and with a generation-event
 * subscription (`onEvent` — the middle of 1 batch cannot be observed).
 * Only the K / V of step 6' reach `dit` as resident tensors (178MiB is not uploaded per forward).
 *
 * MUST: the resident tensors are released **after** the Session of `dit` is closed (while a baked
 * reference remains, `dispose` fails loudly — the same as {@link runDitLoopResident}).
 */
export const runDitLoopOnHost = async (
  state: IrodoriState,
  emit: EmitEvent,
  loop: DitLoop,
): Promise<DitLoopResult> => {
  const { config, gpu } = state;
  let x = loop.initial;
  let forwards = 0;
  const residents: ResidentTensor[] = [];
  let failure: { readonly error: unknown } | undefined;
  try {
    const context = await projectContextOnHost(
      state,
      emit,
      loop.conditions,
      async (bytes, label) => {
        const tensor = await gpu.createResident(bytes, label);
        residents.push(tensor);
        return tensor;
      },
    );
    await withStageSession(state, emit, "dit", state.dit, state.ditSessionOptions, async (run) => {
      for (let step = 0; step < config.steps; step += 1) {
        const t = loop.schedule[step];
        const tNext = loop.schedule[step + 1];
        const tEmbed = f32(timestepEmbedding(t, loop.frequencies), [1, config.timestepEmbedDim]);
        const xTensor = f32(x, [1, loop.frames, config.latentDim]);
        const cond = asF32(
          outputAt(
            state.dit,
            await run({ x_t: xTensor, t_embed: tEmbed, mask: loop.condMask, ...context }),
            0,
          ),
          "dit の速度場",
        );
        forwards += 1;
        const variants: CfgVariant[] = [];
        if (t >= config.cfgMinT && t <= config.cfgMaxT) {
          // MUST: the combination order is SEGMENT_ORDER (text → speaker → caption) — the doc of
          // `combineCfg`.
          for (const variant of loop.uncondVariants) {
            const outputs = await run({
              x_t: xTensor,
              t_embed: tEmbed,
              mask: variant.mask,
              ...context,
            });
            forwards += 1;
            variants.push({
              scale: variant.scale,
              velocity: asF32(
                outputAt(state.dit, outputs, 0),
                `dit の速度場（uncond ${variant.segment}）`,
              ),
            });
          }
        }
        x = eulerStep(x, combineCfg(cond, variants), Math.fround(tNext - t));
        await emit({
          kind: "denoise-step",
          step: step + 1,
          steps: config.steps,
          t,
          copyLatents: latentSnapshot(x, [loop.frames, config.latentDim]),
        });
      }
    });
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    // The Session of `dit` is already released when `withStageSession` returns. Even on failure,
    // release of every resident tensor is attempted, and the original failure is kept.
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      ...residents.map((tensor) => () => tensor.dispose()),
    ]);
  }
  return { x, forwards };
};

/**
 * The GPU-resident path that bundles the whole loop into **1 batch** (H-5).
 *
 * The latent, the velocity fields, the intermediate CFG results and the context K / V (step 6' —
 * built by {@link projectContextOnGpu} in a batch before the loop) all live in
 * {@link ResidentTensor}s, and the loop only queues `dit` and the 2 small graphs built on the host
 * ({@link combineGraph} / {@link eulerGraph}) with `enqueue`. The only trip to the host is the
 * final one (`x_t.read()`), and the loop's fences collapse into the single one of `batch.finish()`
 * (the batch before the loop pays one more fence of its own).
 *
 * MUST: do not wait on `Session.run` inside the span (self-deadlock — the doc of `beginBatch`).
 * MUST: the way operations are queued has the same form as {@link runDitLoopOnHost}, one operation
 * at a time (step 6' once, the variant order, the base of the differences, the argument order). A
 * mismatch moves the last digits and breaks the WAV sha256 check. This path rounds the strength
 * `scale` to f32 before passing it to the GPU, while the host path multiplies in JS f64, but
 * `parseCfgScales` accepts only f32-exact values, so the outputs of the 2 paths match
 * **unconditionally, whatever the distribution**.
 * MUST: the resident tensors are released **after** every Session is closed (while a baked
 * reference remains, `dispose` fails loudly).
 */
export const runDitLoopResident = async (
  state: IrodoriState,
  loop: DitLoop,
): Promise<DitLoopResult> => {
  const { config, gpu } = state;
  const { frames } = loop;
  const observe = observer(state, "dit");
  const latentBytes = frames * config.latentDim * 4;
  // The symbolic dimension cannot be bound from a resident input (a resident tensor has no shape),
  // so it is given explicitly on every enqueue. The name is the one admission settled (the check
  // "there is one symbol" lives there too — `admitIrodori`).
  const bindings = { [state.ditSymbol]: frames };
  const velocity = outputNameAt(state.dit, 0);
  const residents: ResidentTensor[] = [];
  const sessions: Session[] = [];
  let failure: { readonly error: unknown } | undefined;
  try {
    const createResident = async (bytes: number, label: string): Promise<ResidentTensor> => {
      const tensor = await gpu.createResident(bytes, label);
      residents.push(tensor);
      return tensor;
    };
    const xT = await createResident(latentBytes, "irodori.x_t");
    const vCond = await createResident(latentBytes, "irodori.v_cond");
    const vVariant = await createResident(latentBytes, "irodori.v_variant");
    const accumulator = await createResident(latentBytes, "irodori.cfg_acc");
    xT.write(loop.initial);
    // Step 6'. The Session of `dit_context` is closed before `dit` opens (the MUST of
    // projectContextOnGpu).
    const context = await projectContextOnGpu(state, loop.conditions, createResident);
    const open = async (model: ModelComponent, options: SessionOptions): Promise<Session> => {
      const session = await model.createSession(gpu, options);
      sessions.push(session);
      return session;
    };
    // The small graphs the host builds hold no weights at all (no reason to write a `krm`), so only
    // the declaration goes into an in-memory container and becomes a Session.
    const openHostGraph = async (name: string, declaration: IrDeclaration): Promise<Session> => {
      // There is no supply at all, so `tensors` is an empty table (the same spelling as
      // `greedy-output.ts`).
      const bound = openMemoryContainer({ graphs: { [name]: declaration }, tensors: {} });
      const session = await createSessionFromContainer(gpu, bound, name, {});
      sessions.push(session);
      return session;
    };
    const dit = await open(state.dit, state.ditSessionOptions);
    const combine = await openHostGraph("combine", combineGraph(frames, config.latentDim));
    const euler = await openHostGraph("euler", eulerGraph(frames, config.latentDim));
    // The strengths do not depend on the step, so they are built once.
    const scales = loop.uncondVariants.map((variant) => f32(Float32Array.of(variant.scale), [1]));

    let forwards = 0;
    await withBatch(gpu, async (batch) => {
      for (let step = 0; step < config.steps; step += 1) {
        const t = loop.schedule[step];
        const tEmbed = f32(timestepEmbedding(t, loop.frequencies), [1, config.timestepEmbedDim]);
        const forward = async (mask: Tensor, target: ResidentTensor): Promise<void> => {
          await dit.enqueue(
            { x_t: xT, t_embed: tEmbed, mask, ...context },
            { batch, bindings, copyOutputs: { [velocity]: target } },
          );
          forwards += 1;
          if (observe !== undefined) observe(dit.diagnostics());
        };
        await forward(loop.condMask, vCond);
        // Only the first forward of the span places a fence to ground the estimate (P-2). Without
        // it there is no measurement for the whole span = the chunk stays at its initial 16, and
        // about 96,000 dispatches split into 6,000 submits. The cost is one fence (≈11 ms).
        if (step === 0) await batch.settle();
        const guided = t >= config.cfgMinT && t <= config.cfgMaxT && loop.uncondVariants.length > 0;
        if (guided) {
          // MUST: the combination order is SEGMENT_ORDER (text → speaker → caption) — the doc of
          // `combineCfg`.
          for (let index = 0; index < loop.uncondVariants.length; index += 1) {
            await forward(loop.uncondVariants[index].mask, vVariant);
            // The augend for k = 0 is cond itself (`let value = base` in the canonical
            // `combineCfg`). The same buffer is only read through 2 inputs, acc_in and cond, so it
            // is legal on WebGPU too.
            await combine.enqueue(
              {
                [COMBINE_INPUTS.accumulator]: index === 0 ? vCond : accumulator,
                [COMBINE_INPUTS.cond]: vCond,
                [COMBINE_INPUTS.variant]: vVariant,
                [COMBINE_INPUTS.scale]: scales[index],
              },
              { batch, copyOutputs: { [COMBINE_OUTPUT]: accumulator } },
            );
          }
        }
        await euler.enqueue(
          {
            [EULER_INPUTS.x]: xT,
            [EULER_INPUTS.velocity]: guided ? accumulator : vCond,
            [EULER_INPUTS.deltaT]: f32(
              Float32Array.of(Math.fround(loop.schedule[step + 1] - t)),
              [1],
            ),
          },
          { batch, copyOutputs: { [EULER_OUTPUT]: xT } },
        );
      }
    });
    return { x: new Float32Array(await xT.read()), forwards };
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    // Drop the Sessions' baked references first. Even on failure, release of every resource is
    // attempted, and the original failure is kept.
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      ...sessions.map((session) => () => session.dispose()),
      ...residents.map((tensor) => () => tensor.dispose()),
    ]);
  }
};
