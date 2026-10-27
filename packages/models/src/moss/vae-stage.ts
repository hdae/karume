/**
 * The MOSS-Image VAE stage — the untiled FLUX.2 decoder as three symbolic-length graphs, opened
 * after the DiT pair is closed (ADR NNNN decisions 5 and 8, plan items 15-8 and 15-9, plan step 3).
 *
 * ## The run (`./vae-contract.ts` for the graphs)
 *
 * The final DiT latents `[128, H/16, W/16]` go through the host's BatchNorm restoration and
 * unpatchify (`./vae-host.ts`) into the decode input `z [32, H/8, W/8]`. The three Sessions are
 * created on the same device and run in **one batch**: `vae_decoder_pre` writes `[1, C, h, w]` into
 * a resident tensor, `vae_decoder_attn` reads the same bytes as `[1, C, S]` (S = h·w) and writes a
 * second resident, which `vae_decoder_post` reads as `[1, C, h, w]` — nothing is copied between the
 * graphs and there is one read-back, of the image (the Wan2.2 VAE encoder form —
 * `../wan/vae-encoder.ts`). The image is clamped to [-1, 1] as upstream does after `decode`.
 *
 * ## Lifetime (plan item 15-8)
 *
 * The stage runs after the DiT stage has closed its Sessions (the order `dit_text` → DiT pair →
 * VAE, each closed before the next opens — `./dit-stage.ts`). It creates its three Sessions and
 * three residents, runs, and closes everything before it returns — also when it fails — then
 * waits for the release (`../session/settle-released-memory.ts`), so the next generation's first
 * stage does not see the decoder's allocation.
 *
 * ## Headroom (plan item 15-8 — the Anima form)
 *
 * Before it opens, the stage waits for the release of what the previous stage destroyed and then
 * probes the device with a trial allocation (`fitsHeadroom` — ADR 0112) of: the **sum** of the
 * three graphs' estimates (`peakAccountedBytes` with the device limits and the stage's bindings —
 * the three Sessions are open together) + the three hand-over residents (the estimates count a
 * graph's input and output buffers as its backing's, but a resident is a separate allocation, so
 * they are added — an upper bound, as for the DiT's text contexts) + the largest weight part (the
 * upload staging) + the margin of `../anima/residency.ts`. A refusal fails loudly naming the stage,
 * the bytes and the size. The plan's hand count — 1,920 B per pixel at the full-resolution resnet
 * 192 → 96 (§1.5) — is a cross-check only (the tests compare it with the estimate and the
 * measurement). The check is a fact of the moment, not a reservation.
 *
 * ## Acceptance (plan item 15-9)
 *
 * {@link planMossVae} checks the size (`./request.ts`) and the area against this device's area
 * limit (`mossVaeAreaLimit` — the binding over the decoder's bytes per pixel) with an error that
 * names the limit, before any GPU work.
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import {
  fitsHeadroom,
  type GpuContext,
  type ResidentTensor,
  type Session,
  type SessionDiagnostics,
} from "@karume/runtime";

import { HEADROOM_MARGIN_BYTES } from "../anima/residency.ts";
import type { ModelComponent } from "../hub/components.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { settleReleasedMemory } from "../session/settle-released-memory.ts";
import type { MossCells } from "./dit-rope.ts";
import { assertMossImageArea, assertMossImageSize } from "./request.ts";
import {
  MOSS_VAE_DECODER_ATTN,
  MOSS_VAE_DECODER_POST,
  MOSS_VAE_DECODER_PRE,
  type MossVaeContract,
  mossVaeAreaLimit,
  type MossVaeDeviceLimits,
} from "./vae-contract.ts";
import { mossClampImage, mossVaeInput } from "./vae-host.ts";

const OWNER = "MOSS-Image";

/** Bytes per element (every semantic dtype is 4 bytes — ADR 0009). */
const BYTES_PER_ELEMENT = 4;

/** The RGB channels of the decoded image. */
const IMAGE_CHANNELS = 3;

/** The three components of the VAE series. */
export type MossVaeComponents = {
  readonly pre: ModelComponent;
  readonly attn: ModelComponent;
  readonly post: ModelComponent;
};

/** What the stage reads (built once per pipeline — the contract settled in admission). */
export type MossVaeStageState = {
  readonly gpu: GpuContext;
  readonly components: MossVaeComponents;
  readonly contract: MossVaeContract;
};

/** An accepted VAE request ({@link planMossVae}). */
export type MossVaePlan = {
  readonly width: number;
  readonly height: number;
  /** The DiT latent grid (16-px cells). */
  readonly cells: MossCells;
  /** The VAE latent grid (the decode input's h × w — 2 × the DiT cells). */
  readonly latent: { readonly height: number; readonly width: number };
};

/** The acceptance check of the VAE stage (module doc "Acceptance"). */
export const planMossVae = (
  request: { readonly width: number; readonly height: number },
  contract: Pick<MossVaeContract, "bytesPerCell" | "spatialScale">,
  limits: MossVaeDeviceLimits,
): MossVaePlan => {
  const cells = assertMossImageSize(request.width, request.height);
  assertMossImageArea(request.width, request.height, mossVaeAreaLimit(contract, limits));
  return {
    width: request.width,
    height: request.height,
    cells,
    latent: { height: 2 * cells.rows, width: 2 * cells.cols },
  };
};

/** The bindings of the three graphs for a plan. */
const bindingsOf = (contract: MossVaeContract, plan: MossVaePlan) => ({
  pre: { [contract.pre.height]: plan.latent.height, [contract.pre.width]: plan.latent.width },
  attn: { [contract.attn.sequence]: plan.latent.height * plan.latent.width },
  post: { [contract.post.height]: plan.latent.height, [contract.post.width]: plan.latent.width },
});

/** The bytes of the three hand-over residents (pre's and attn's outputs, the image). */
const residentBytes = (contract: MossVaeContract, plan: MossVaePlan) => {
  const cells = plan.latent.height * plan.latent.width;
  return {
    hidden: contract.hiddenChannels * cells * BYTES_PER_ELEMENT,
    image: IMAGE_CHANNELS * plan.width * plan.height * BYTES_PER_ELEMENT,
  };
};

/**
 * The bytes the stage needs on a device of `limits` (module doc "Headroom"). NOTE: `export` is for
 * the tests that pin the sum against the measurement (not on a public surface).
 */
export const mossVaeStageNeeds = (
  input: {
    readonly limits: MossVaeDeviceLimits;
    readonly components: MossVaeComponents;
    readonly contract: MossVaeContract;
  },
  plan: MossVaePlan,
): number => {
  const { limits, components, contract } = input;
  const bindings = bindingsOf(contract, plan);
  const estimate = (component: ModelComponent, values: Readonly<Record<string, number>>) =>
    component.estimate({
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxBufferSize: limits.maxBufferSize,
      bindings: values,
    }).peakAccountedBytes;
  const residents = residentBytes(contract, plan);
  return (
    estimate(components.pre, bindings.pre) +
    estimate(components.attn, bindings.attn) +
    estimate(components.post, bindings.post) +
    2 * residents.hidden +
    residents.image +
    Math.max(
      components.pre.maxPartBytes,
      components.attn.maxPartBytes,
      components.post.maxPartBytes,
    ) +
    HEADROOM_MARGIN_BYTES
  );
};

/** One component of the stage, as `observe` names it. */
export type MossVaeComponent =
  | typeof MOSS_VAE_DECODER_PRE
  | typeof MOSS_VAE_DECODER_ATTN
  | typeof MOSS_VAE_DECODER_POST;

/**
 * The VAE stage (module doc): the image `[3, H, W]` in [-1, 1] of the final DiT latents
 * `[128, H/16, W/16]`. `observe` receives each Session's diagnostics after the batch.
 */
export const decodeMossVae = async (
  state: MossVaeStageState,
  plan: MossVaePlan,
  latents: Float32Array<ArrayBuffer>,
  observe?: (component: MossVaeComponent, diagnostics: SessionDiagnostics) => void,
): Promise<Float32Array<ArrayBuffer>> => {
  const { gpu, components, contract } = state;
  const z = mossVaeInput(latents, contract.batchNorm, plan.cells);
  const bytes = mossVaeStageNeeds({ ...state, limits: gpu.limits }, plan);
  // MUST: wait for the release of what the DiT stage destroyed before the trial allocation (the
  // DiT stage's observation — `./dit-stage.ts`).
  await settleReleasedMemory(gpu);
  if (!(await fitsHeadroom(gpu, bytes))) {
    throw new Error(
      `${OWNER}: the VAE stage needs ${bytes} bytes of VRAM (estimates + residents + largest part +` +
        ` margin) that this device cannot allocate now — ${plan.width}×${plan.height} px does not fit`,
    );
  }
  const bindings = bindingsOf(contract, plan);
  const residents = residentBytes(contract, plan);
  const sessions: Session[] = [];
  const held: ResidentTensor[] = [];
  const open = async (component: ModelComponent): Promise<Session> => {
    const session = await component.createSession(gpu, {});
    sessions.push(session);
    return session;
  };
  const resident = async (byteLength: number, label: string): Promise<ResidentTensor> => {
    const tensor = await gpu.createResident(byteLength, label);
    held.push(tensor);
    return tensor;
  };
  let failure: { readonly error: unknown } | undefined;
  try {
    const pre = await open(components.pre);
    const attn = await open(components.attn);
    const post = await open(components.post);
    const hidden = await resident(residents.hidden, `moss ${MOSS_VAE_DECODER_PRE}`);
    const attended = await resident(residents.hidden, `moss ${MOSS_VAE_DECODER_ATTN}`);
    const image = await resident(residents.image, `moss ${MOSS_VAE_DECODER_POST}`);
    // MUST: no Session construction, run or dispose inside the batch span (`GpuContext.beginBatch`
    // — a self-deadlock on the span lock); everything is created above.
    const batch = await gpu.beginBatch();
    let read: Readonly<Record<string, ArrayBuffer>>;
    try {
      await pre.enqueue(
        {
          [contract.pre.input]: {
            dtype: "f32",
            shape: [1, contract.latentChannels, plan.latent.height, plan.latent.width],
            data: z,
          },
        },
        { batch, copyOutputs: { [contract.pre.output]: hidden } },
      );
      // A resident input is not a source of symbol bindings (it carries bytes only), so every
      // symbol is passed.
      await attn.enqueue(
        { [contract.attn.input]: hidden },
        { batch, bindings: bindings.attn, copyOutputs: { [contract.attn.output]: attended } },
      );
      await post.enqueue(
        { [contract.post.input]: attended },
        { batch, bindings: bindings.post, copyOutputs: { [contract.post.output]: image } },
      );
    } catch (cause) {
      // MUST: close the span and return the lock; the main failure is the caller's (the
      // settlement on the finish side repeats it — the Wan encoder's form).
      await batch.finish().catch(() => undefined);
      throw cause;
    }
    read = await batch.finishAndRead({ image });
    observe?.(MOSS_VAE_DECODER_PRE, pre.diagnostics());
    observe?.(MOSS_VAE_DECODER_ATTN, attn.diagnostics());
    observe?.(MOSS_VAE_DECODER_POST, post.diagnostics());
    const decoded = new Float32Array(read["image"]);
    const broken = decoded.findIndex((value) => !Number.isFinite(value));
    if (broken !== -1) {
      const plane = plan.width * plan.height;
      const pixel = broken % plane;
      throw new Error(
        `${OWNER}: the decoded image channel ${Math.floor(broken / plane)}, pixel (x=${pixel % plan.width},` +
          ` y=${Math.floor(pixel / plan.width)}) is non-finite (${decoded[broken]})`,
      );
    }
    return mossClampImage(decoded);
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    // MUST: Sessions close before the residents they bind; a failure to close or to wait does not
    // overwrite the main failure; the release wait runs on failure too (module doc "Lifetime").
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      ...sessions.map((session) => () => session.dispose()),
      ...held.map((tensor) => () => tensor.dispose()),
      () => settleReleasedMemory(gpu),
    ]);
  }
};
