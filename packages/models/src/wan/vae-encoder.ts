/**
 * The VAE encoder stage of Wan2.2 TI2V's I2V — turns one condition image into a latent with encode
 * chunk 0 (ADR 0121 decision 11, stage 9).
 *
 * The encoder is three symbolic-length graphs (recipe `wan/export_vae_encoder.py`). It is split before
 * and after the mid block's attention because the IR's dimension expressions are linear in a single
 * symbol, and the sequence length h·w that attention creates cannot be written as one axis (ADR 0121
 * addendum "stage 9 plan rulings and stage 9a results"):
 *
 * | Component          | Input → output                                          | Symbols  |
 * | ------------------ | ------------------------------------------------------- | -------- |
 * | `vae_encoder_pre`  | patchified image `[3·p², 1, k·h, k·w]` → `[C, 1, h, w]` | h, w     |
 * | `vae_encoder_attn` | `[C, S]` → `[C, S]`                                     | S (h·w)  |
 * | `vae_encoder_post` | `[C, 1, h, w]` → mu `[Cz, 1, h, w]`                     | h, w     |
 *
 * h, w are the latent height and width, k the entry factor (spatial compression ÷ p — p is the
 * descriptor's `vaePatchSize`), C the mid block's channel count, Cz the latent channel count. The
 * three Sessions are created on the same GpuContext; pre → attn → post are enqueued in one batch
 * and hand over through resident tensors (`[C, 1, h, w]` and `[C, S]` are reinterpretations of the
 * same bytes, so nothing is copied again). There is a single read-back, of mu, and mu is normalized
 * by `normalizeWanLatents` into the DiT's condition latent.
 *
 * The stage order and Session lifetimes (encoder → text → DiT → VAE, each created and closed one stage
 * at a time) are defined by the doc at the top of `./pipeline.ts`; this module only runs the one
 * encoder stage. `owner` is the message prefix (`"wan.Ti2vPipeline"`).
 *
 * NOTE: `export` is for the pipeline internals and for tests that pin the checks without a GPU (not on
 * the `mod.ts` / subpath surface — ADR 0008).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state —
 * CLAUDE.md).
 */

import type { GpuContext, ResidentTensor, Session, SessionDiagnostics } from "@karume/runtime";

import { settleAbort } from "../concurrency/abort.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { settleReleasedMemory } from "../session/settle-released-memory.ts";
import type { GraphOwner, ModelComponent } from "../hub/components.ts";
import type { GenerateEvent, RunComponent } from "./pipeline.ts";
import type { WanGenerationDescriptor } from "./descriptor.ts";
import type { PlanLayout } from "./plan.ts";
import { assertDims, firstNonFinite, valueOf } from "./graph-io.ts";
import type { WanI2vImageInput } from "./i2v-preprocess.ts";
import { normalizeWanLatents } from "./latents.ts";
import { wanSpatialCompression } from "./tile-decode.ts";

/**
 * Component keys (the series' graph names = the manifest's weights names — recipe
 * `wan/export_vae_encoder.py`).
 */
export const VAE_ENCODER_PRE = "vae_encoder_pre";
export const VAE_ENCODER_ATTN = "vae_encoder_attn";
export const VAE_ENCODER_POST = "vae_encoder_post";

/** The encoder's three components (in the order they run within the stage). */
export const WAN_VAE_ENCODER_KEYS: readonly string[] = [
  VAE_ENCODER_PRE,
  VAE_ENCODER_ATTN,
  VAE_ENCODER_POST,
];

/** The channel count of the condition image (RGB — before patchify). */
const RGB_CHANNELS = 3;

/** Bytes per element (every semantic dtype is 4 bytes — ADR 0009). */
const BYTES_PER_ELEMENT = 4;

/** The encoder's three-graph contract, confirmed at construction ({@link wanVaeEncoderContract}). */
export type WanVaeEncoderContract = {
  readonly pre: { readonly input: string; readonly output: string };
  /** `sequence` is the sequence-length symbol (the name of the binding `S = h·w`). */
  readonly attn: { readonly input: string; readonly output: string; readonly sequence: string };
  /**
   * `height` / `width` are post's symbols (a resident input is not a binding source, so they are
   * passed as bindings).
   */
  readonly post: {
    readonly input: string;
    readonly output: string;
    readonly height: string;
    readonly width: string;
  };
  /** The mid block's channel count (pre's exit = the width of attn's sequence = post's entry). */
  readonly hiddenChannels: number;
  /** The latent channel count (post's exit = mu's channel count — same as the VAE decoder's latent). */
  readonly latentChannels: number;
  /** The channel count at pre's entry (`3·p²` — the patchified image). */
  readonly patchChannels: number;
  /** The spatial factor k at pre's entry (the patchified image's edge = k × the latent edge). */
  readonly inputScale: number;
  /**
   * The spatial compression (image pixels / latent — the same single function as the decoder,
   * `wanSpatialCompression`).
   */
  readonly compression: number;
};

/**
 * The encoder's three components (the suppliers opened at construction and the contract confirmed in
 * admission).
 */
export type WanVaeEncoder = {
  readonly pre: ModelComponent;
  readonly attn: ModelComponent;
  readonly post: ModelComponent;
  readonly contract: WanVaeEncoderContract;
};

/**
 * The canonical notation of a single-symbol dimension times a coefficient k (the IR dimension
 * expression `coeff·sym` — coefficient 1 is omitted).
 */
const scaledSymbol = (scale: number, symbol: string): string =>
  scale === 1 ? symbol : `${scale}${symbol}`;

/** The graph's sole input and output (one f32 → one — the host builds and reads only that). */
const soleIo = (owner: string, part: string, model: GraphOwner) => {
  const { inputs, outputs } = model.graph;
  if (inputs.length !== 1 || outputs.length !== 1) {
    throw new Error(
      `${owner}: ${part} has ${inputs.length} inputs → ${outputs.length} outputs (only 1 → 1 is built)`,
    );
  }
  const [input] = inputs;
  const [output] = outputs;
  const value = valueOf(owner, model, output);
  const declared: readonly (readonly [string, string])[] = [
    [`input '${input.name}'`, input.dtype],
    [`output '${output}'`, value.dtype],
  ];
  for (const [what, dtype] of declared) {
    if (dtype !== "f32") {
      throw new Error(`${owner}: ${part} ${what} has dtype ${dtype}, not f32`);
    }
  }
  return { input: input.name, inputShape: input.shape, output, outputShape: value.shape };
};

/** The axis declaration is a bare symbol (coefficient 1, offset 0 — one of the declared symbols). */
const symbolAt = (
  owner: string,
  part: string,
  model: GraphOwner,
  dims: readonly (number | string)[],
  axis: number,
  where: string,
): string => {
  const dim = dims.at(axis);
  if (typeof dim !== "string" || !model.graph.symbols.includes(dim)) {
    throw new Error(
      `${owner}: ${part} ${where} axis ${axis} is not a bare symbol (${String(dim)})`,
    );
  }
  return dim;
};

/** The axis declaration is a static dimension. */
const staticAt = (
  owner: string,
  part: string,
  dims: readonly (number | string)[],
  axis: number,
  where: string,
): number => {
  const dim = dims.at(axis);
  if (typeof dim !== "number") {
    throw new Error(`${owner}: ${part} ${where} axis ${axis} is not static (${String(dim)})`);
  }
  return dim;
};

/** The declared symbols are exactly `expected` (binding sources are only what the host passes). */
const assertSymbols = (
  owner: string,
  part: string,
  model: GraphOwner,
  expected: readonly string[],
): void => {
  const declared = [...model.graph.symbols].sort();
  const wanted = [...expected].sort();
  if (declared.length !== wanted.length || declared.some((name, at) => name !== wanted[at])) {
    throw new Error(
      `${owner}: ${part} symbols are [${model.graph.symbols.join(", ")}] (expected: [${expected.join(
        ", ",
      )}])`,
    );
  }
};

/** The encoder's bindings for one size ({@link wanVaeEncoderBinding}). */
export type WanVaeEncoderBinding = {
  /** The latent height and width (post's symbol values; attn's sequence length is `height·width`). */
  readonly height: number;
  readonly width: number;
  /** The shape of pre's entry `[3·p², 1, k·h, k·w]` (matches the shape of the preprocessing output). */
  readonly inputShape: readonly [number, number, number, number];
};

/**
 * The encoder's bindings for the output size `size` (the latent height and width and the shape of
 * pre's entry). A size that does not divide evenly is a plain `Error` (the entry check narrows the
 * accepted sizes first, so a failure here means the asset and the descriptor disagree).
 *
 * MUST: the admission coverage check ({@link wanVaeEncoderContract}) and the stage
 * ({@link encodeWanImageStage}) derive the bindings with this same function — derived on two paths,
 * the stage could run a size admission accepted with a different shape.
 */
export const wanVaeEncoderBinding = (
  contract: WanVaeEncoderContract,
  size: { readonly width: number; readonly height: number },
  owner: string,
): WanVaeEncoderBinding => {
  const height = size.height / contract.compression;
  const width = size.width / contract.compression;
  if (!Number.isInteger(height) || !Number.isInteger(width) || height < 1 || width < 1) {
    throw new Error(
      `${owner}: ${size.width}×${size.height} with the VAE encoder's spatial compression ${contract.compression}` +
        " does not give a latent of positive integers",
    );
  }
  return {
    height,
    width,
    inputShape: [
      contract.patchChannels,
      1,
      contract.inputScale * height,
      contract.inputScale * width,
    ],
  };
};

/**
 * Matches the declarations of the encoder's three graphs against the inputs the host builds, the
 * outputs it reads and the VAE decoder's latent, and checks that they cover every accepted size (a
 * family admission check — before the weight parts are fetched).
 *
 * - Each graph is one f32 input → one output. pre is `[3·p², 1, k·h, k·w]` → `[C, 1, h, w]` (k =
 *   spatial compression ÷ p; only the two symbols h, w), attn is `[C, S]` → `[C, S]` (only the symbol
 *   S), post is `[C, 1, h, w]` → `[Cz, 1, h, w]` (only the two symbols h, w). C is the same in all
 *   three; Cz is the decoder's latent channel count.
 * - Coverage: a symbolic-length graph has no per-size shapes, so for each accepted size the bindings
 *   ({@link wanVaeEncoderBinding}) must come out as integers. The entry factor k is matched with the
 *   value derived from the decoder's spatial compression, so an encoder with a different compression
 *   fails the shape check.
 *
 * MUST: call in admission. The host builds the inputs with its own constants and passes the bindings,
 * so a differing declaration would fail only inside the batch after the three Sessions are created
 * (failure attribution is per batch, and it cannot even be narrowed to which graph).
 *
 * NOTE: symbol names are read from the graph declarations (not spelled out on the host — bindings are
 * passed by name).
 */
export const wanVaeEncoderContract = (
  parts: { readonly pre: GraphOwner; readonly attn: GraphOwner; readonly post: GraphOwner },
  layout: PlanLayout,
  generation: WanGenerationDescriptor,
  owner: string,
): WanVaeEncoderContract => {
  const patchSize = generation.vaePatchSize;
  const compression = wanSpatialCompression(layout, generation);
  const inputScale = compression / patchSize;
  if (!Number.isInteger(inputScale) || inputScale < 1) {
    throw new Error(
      `${owner}: spatial compression ${compression} is not divisible by the patchify patch size ${patchSize} (the scale of the VAE encoder input)`,
    );
  }
  const patchChannels = RGB_CHANNELS * patchSize * patchSize;

  const pre = soleIo(owner, VAE_ENCODER_PRE, parts.pre);
  const preWhere = `output '${pre.output}'`;
  const hiddenChannels = staticAt(owner, VAE_ENCODER_PRE, pre.outputShape, 0, preWhere);
  const preHeight = symbolAt(owner, VAE_ENCODER_PRE, parts.pre, pre.outputShape, 2, preWhere);
  const preWidth = symbolAt(owner, VAE_ENCODER_PRE, parts.pre, pre.outputShape, 3, preWhere);
  assertSymbols(owner, VAE_ENCODER_PRE, parts.pre, [preHeight, preWidth]);
  assertDims(
    owner,
    pre.outputShape,
    [hiddenChannels, 1, preHeight, preWidth],
    `${VAE_ENCODER_PRE} ${preWhere}`,
  );
  assertDims(
    owner,
    pre.inputShape,
    [patchChannels, 1, scaledSymbol(inputScale, preHeight), scaledSymbol(inputScale, preWidth)],
    `${VAE_ENCODER_PRE} input '${pre.input}'`,
  );

  const attn = soleIo(owner, VAE_ENCODER_ATTN, parts.attn);
  const sequence = symbolAt(
    owner,
    VAE_ENCODER_ATTN,
    parts.attn,
    attn.inputShape,
    1,
    `input '${attn.input}'`,
  );
  assertSymbols(owner, VAE_ENCODER_ATTN, parts.attn, [sequence]);
  for (const [what, dims] of [
    [`input '${attn.input}'`, attn.inputShape],
    [`output '${attn.output}'`, attn.outputShape],
  ] as const) {
    assertDims(owner, dims, [hiddenChannels, sequence], `${VAE_ENCODER_ATTN} ${what}`);
  }

  const post = soleIo(owner, VAE_ENCODER_POST, parts.post);
  const postWhere = `input '${post.input}'`;
  const postHeight = symbolAt(owner, VAE_ENCODER_POST, parts.post, post.inputShape, 2, postWhere);
  const postWidth = symbolAt(owner, VAE_ENCODER_POST, parts.post, post.inputShape, 3, postWhere);
  assertSymbols(owner, VAE_ENCODER_POST, parts.post, [postHeight, postWidth]);
  assertDims(
    owner,
    post.inputShape,
    [hiddenChannels, 1, postHeight, postWidth],
    `${VAE_ENCODER_POST} ${postWhere}`,
  );
  assertDims(
    owner,
    post.outputShape,
    [layout.latentChannels, 1, postHeight, postWidth],
    `${VAE_ENCODER_POST} output '${post.output}'`,
  );

  const contract: WanVaeEncoderContract = {
    pre: { input: pre.input, output: pre.output },
    attn: { input: attn.input, output: attn.output, sequence },
    post: { input: post.input, output: post.output, height: postHeight, width: postWidth },
    hiddenChannels,
    latentChannels: layout.latentChannels,
    patchChannels,
    inputScale,
    compression,
  };
  for (const size of generation.acceptedSizes) wanVaeEncoderBinding(contract, size, owner);
  return contract;
};

/**
 * The built material the encoder stage reads (this stage's part of the pipeline's internal state —
 * taken structurally).
 */
type WanVaeEncodeState = {
  readonly gpu: GpuContext;
  readonly vaeEncoder: WanVaeEncoder;
  readonly onRunDiagnostics?: (component: RunComponent, diagnostics: SessionDiagnostics) => void;
};

/** The stage's three Sessions (in the order they run within the stage). */
type EncoderSessions = { readonly pre: Session; readonly attn: Session; readonly post: Session };

/** The hand-over residents (pre's exit, attn's exit, mu). */
type EncoderResidents = {
  readonly pre: ResidentTensor;
  readonly attn: ResidentTensor;
  readonly mu: ResidentTensor;
};

/**
 * Runs the three graphs in one batch and reads mu (one read-back at the batch's settlement — and one
 * fence).
 *
 * MUST: do not issue Session construction, `run` or `dispose` inside the span (the MUST of
 * `GpuContext.beginBatch` — a self-deadlock on the span lock). The caller creates the Sessions and
 * residents before the span.
 */
const runEncoderBatch = async (
  gpu: GpuContext,
  sessions: EncoderSessions,
  residents: EncoderResidents,
  contract: WanVaeEncoderContract,
  binding: WanVaeEncoderBinding,
  pixels: Float32Array<ArrayBuffer>,
): Promise<ArrayBuffer> => {
  const batch = await gpu.beginBatch();
  try {
    await sessions.pre.enqueue(
      { [contract.pre.input]: { dtype: "f32", shape: binding.inputShape, data: pixels } },
      { batch, copyOutputs: { [contract.pre.output]: residents.pre } },
    );
    // A resident input is not a source for symbol bindings (it carries only bytes and a size), so the
    // symbols are passed as bindings.
    await sessions.attn.enqueue(
      { [contract.attn.input]: residents.pre },
      {
        batch,
        bindings: { [contract.attn.sequence]: binding.height * binding.width },
        copyOutputs: { [contract.attn.output]: residents.attn },
      },
    );
    await sessions.post.enqueue(
      { [contract.post.input]: residents.attn },
      {
        batch,
        bindings: {
          [contract.post.height]: binding.height,
          [contract.post.width]: binding.width,
        },
        copyOutputs: { [contract.post.output]: residents.mu },
      },
    );
  } catch (cause) {
    // MUST: close the span and return the lock (otherwise the next batch / run on the same device waits
    // forever). The main failure is rethrown to the caller (the settlement on the finish side is a copy
    // of the same failure the span recorded, so it is discarded — the same form as the VAE decoder's
    // `decodeWanVaeTile`).
    await batch.finish().catch(() => undefined);
    throw cause;
  }
  const read = await batch.finishAndRead({ mu: residents.mu });
  return read["mu"];
};

/**
 * The encoder stage (creates the three Sessions and the hand-over residents, runs them in one batch,
 * closes them and waits for the release — `end` comes after the wait). Returns the normalized
 * condition latent `[Cz, 1, h, w]` (`normalizeWanLatents` — the DiT stage replaces the first latent
 * frame with it).
 *
 * MUST: close and wait for the release before the DiT stage is created (ADR 0121 decision 11 — the DiT
 * stage has only 0.4–0.8 GiB of capacity margin, so the encoder's allocation (at 1280×704, 103 MiB of
 * weights and 900 MiB of planned backing) must not remain. On the B570 the release of `destroy()` is
 * delayed until the next device poll — the doc of `settleReleasedMemory`). A failed stage is also
 * closed and then waited on (nothing remains into the DiT stage of the same pipeline's next
 * generation). A failure to close or to wait does not overwrite the main failure (the same form as the
 * DiT stage). The Session's execution options are `{}` (the quant's `session` is taken only by the DiT
 * Session — the same contract as the VAE decoder).
 * MUST: close Sessions → residents in that order (destroying a resident is refused while a baked-in
 * reference remains).
 *
 * MUST: check finiteness after normalization (O(Cz·h·w)). Passing a non-finite condition latent to the
 * DiT, under the replacement formula `(1 − m)·cond + m·latents`, would fail at the step-1 latent check
 * with a message pointing at the DiT (blaming the wrong stage for the true cause).
 */
export const encodeWanImageStage = async (
  state: WanVaeEncodeState,
  image: WanI2vImageInput,
  generation: Pick<WanGenerationDescriptor, "latents">,
  emit: (event: GenerateEvent) => Promise<void>,
  signal: AbortSignal | undefined,
  owner: string,
): Promise<Float32Array<ArrayBuffer>> => {
  const { contract } = state.vaeEncoder;
  const binding = wanVaeEncoderBinding(contract, image, owner);
  if (
    image.shape.length !== binding.inputShape.length ||
    image.shape.some((dim, axis) => dim !== binding.inputShape[axis])
  ) {
    throw new Error(
      `${owner}: the preprocessed condition image shape [${image.shape.join(", ")}] differs from the VAE encoder input ` +
        `[${binding.inputShape.join(", ")}]`,
    );
  }
  const area = binding.height * binding.width;
  const observe = state.onRunDiagnostics;

  await emit({ kind: "stage", component: "vae_encoder", at: "start" });
  const sessions: Session[] = [];
  const residents: ResidentTensor[] = [];
  const createSession = async (component: ModelComponent): Promise<Session> => {
    const session = await component.createSession(state.gpu, {});
    sessions.push(session);
    return session;
  };
  const createResident = async (channels: number, label: string): Promise<ResidentTensor> => {
    const resident = await state.gpu.createResident(channels * area * BYTES_PER_ELEMENT, label);
    residents.push(resident);
    return resident;
  };
  let latents: Float32Array<ArrayBuffer>;
  let failure: { readonly error: unknown } | undefined;
  try {
    const opened: EncoderSessions = {
      pre: await createSession(state.vaeEncoder.pre),
      attn: await createSession(state.vaeEncoder.attn),
      post: await createSession(state.vaeEncoder.post),
    };
    const carried: EncoderResidents = {
      pre: await createResident(contract.hiddenChannels, `wan ${VAE_ENCODER_PRE}`),
      attn: await createResident(contract.hiddenChannels, `wan ${VAE_ENCODER_ATTN}`),
      mu: await createResident(contract.latentChannels, `wan ${VAE_ENCODER_POST}`),
    };
    // Before the run (the three-graph batch is indivisible — cancellation takes effect here).
    await settleAbort(signal);
    const read = await runEncoderBatch(state.gpu, opened, carried, contract, binding, image.pixels);
    observe?.(VAE_ENCODER_PRE, opened.pre.diagnostics());
    observe?.(VAE_ENCODER_ATTN, opened.attn.diagnostics());
    observe?.(VAE_ENCODER_POST, opened.post.diagnostics());
    const mu = new Float32Array(read);
    latents = normalizeWanLatents(mu, generation.latents);
    const broken = firstNonFinite(latents);
    if (broken !== -1) {
      const pixel = broken % area;
      throw new Error(
        `${owner}: condition image latent (after normalizing the VAE encoder's mu) channel ${Math.floor(
          broken / area,
        )}, position (x=${pixel % binding.width}, y=${Math.floor(pixel / binding.width)}) is non-finite` +
          ` (${latents[broken]}) — not passed to the DiT stage`,
      );
    }
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    // The release wait is placed for a failed stage too — if the caller runs the next generation on
    // the same pipeline, its DiT stage would otherwise be created without waiting for the release of
    // the encoder's allocation. A failure to wait is handled like a failure to close (it does not
    // overwrite the main failure).
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      ...sessions.map((session) => () => session.dispose()),
      ...residents.map((resident) => () => resident.dispose()),
      () => settleReleasedMemory(state.gpu),
    ]);
  }
  await emit({ kind: "stage", component: "vae_encoder", at: "end" });
  return latents;
};
