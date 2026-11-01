/**
 * The MOSS-Image text-encoder stage — prompt strings → the Qwen3.5 text tower's rows `[T, 4096]`
 * that `dit_text` reads (ADR 0138 decision 9's prompt form; the encoder is ADR 0137's, its consumer
 * pattern ADR 0137 decision 4 and Wan's umT5 text stage — ADR 0119 decision 11).
 *
 * ## Host side (in the acceptance check — `./generate.ts`)
 *
 * {@link mossPromptIds}: upstream's `encode_prompt` order — the chat template
 * (`renderQwen3_5ChatPrompt`: one user message, generation prompt, thinking enabled), then the
 * Qwen3.5 tokenizer with the length policy of the maintainer's ruling 50 (an over-length prompt is
 * rejected with the tokenizer's `ModelInputError`; `overflow: "truncate"` cuts it as upstream
 * does). Both run before any GPU work, so a prompt the template or the bound refuses never opens the
 * encoder.
 *
 * ## The stage ({@link encodeMossPrompts} — the first stage of a prompt-form generation)
 *
 * 1. The stage's own headroom check (`fitsHeadroom` after waiting for released memory): the
 *    encoder's estimate at T = max(T₊, T₋) (the two prompts run in turn on one Session) + the
 *    largest weight part (the upload staging) + the margin of `../session/headroom-margin.ts` —
 *    the form of the other MOSS stages (`./dit-stage.ts` "Headroom").
 * 2. `Qwen3_5TextEncoder.open` (about 7.4 GiB of i8 weights — it cannot stay beside the DiT pair or
 *    the VAE on a 12 GB device, so it is opened per generation), the positive then the negative
 *    encoding, and `dispose`.
 * 3. MUST: the encoder is disposed on every path — success, a failed run, a cancellation — and the
 *    released memory is awaited before the stage returns or rethrows (`disposeSteps`; a failure to
 *    close never overwrites the main failure). Without the wait the next stage's probe sees the
 *    encoder's buffers (Dawn returns destroyed buffers only when a later submit completes —
 *    `../session/settle-released-memory.ts`).
 *
 * The encoder takes no execution knob (ADR 0137 decision 5): no preset changes it, as no preset
 * changes `dit_text` or the VAE.
 *
 * ## Admission ({@link mossTextEncoderContract} — before any weight byte is uploaded)
 *
 * The caller opens the Qwen3.5 text series and its tokenizer asset itself (no manifest or
 * distribution yet — ADR 0138 decision 12) and settles the contract once, as for the DiT and the
 * VAE: the encoder's graph contract (`qwen3_5GraphContract`), its row width against the DiT's text
 * width ({@link MOSS_TEXT_WIDTH} — a narrower or wider tower would fail only in `dit_text`'s shape
 * check after the upload), its one symbolic length (the estimate's binding), and the tokenizer's
 * bound against the encoder's and the DiT's (the MUST of ADR 0137's consequences: a prompt between
 * two different bounds would pass one and fail in the other).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import {
  type AdmissionReport,
  type EstimateOptions,
  fitsHeadroom,
  type GpuContext,
} from "@karume/runtime";

import { HEADROOM_MARGIN_BYTES } from "../session/headroom-margin.ts";
import { settleAbort } from "../concurrency/abort.ts";
import type { GraphOwner, ModelComponent } from "../hub/components.ts";
import { renderQwen3_5ChatPrompt } from "../qwen3_5/chat.ts";
import { type Qwen3_5TextComponent, Qwen3_5TextEncoder } from "../qwen3_5/encoder.ts";
import { type Qwen3_5GraphContract, qwen3_5GraphContract } from "../qwen3_5/graph-contract.ts";
import { QWEN3_5_MAX_TOKENS } from "../qwen3_5/session-io.ts";
import type { Qwen3_5Overflow, Qwen3_5Tokenizer } from "../qwen3_5/tokenizer.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { settleReleasedMemory } from "../session/settle-released-memory.ts";
import type { MossBranches, MossBranchText, MossDeviceLimits } from "./dit-stage.ts";
import { MOSS_MAX_TEXT_TOKENS, MOSS_TEXT_WIDTH } from "./request.ts";

const OWNER = "MOSS-Image";

/** The stage's name in messages and logs (the Qwen3.5 series' component and graph name). */
export const MOSS_TEXT_ENCODER = "text_encoder";

/** The encoder's contract as MOSS-Image settles it ({@link mossTextEncoderContract}). */
export type MossTextEncoderContract = Qwen3_5GraphContract & {
  /** The graph's one symbolic length T (the estimate's binding). */
  readonly length: string;
};

/**
 * What the stage reads of the text tower's component: the encoder's entry (the graph and the
 * Session) and the two figures of its headroom need. A `ModelComponent` over the series container
 * is one.
 */
export type MossTextEncoderComponent = Qwen3_5TextComponent &
  Pick<ModelComponent, "maxPartBytes"> & {
    readonly estimate: (options: EstimateOptions) => Pick<AdmissionReport, "peakAccountedBytes">;
  };

/**
 * The opened Qwen3.5 text series a prompt-form generation needs: the text tower, its tokenizer and
 * the contract settled for them ({@link mossTextEncoderContract}).
 */
export type MossTextEncoderState = {
  readonly component: MossTextEncoderComponent;
  readonly tokenizer: Qwen3_5Tokenizer;
  readonly contract: MossTextEncoderContract;
};

/**
 * The admission of the text encoder and its tokenizer (module doc "Admission"). Every failure is a
 * plain `Error` (the series and the asset are the caller's assembly, not a request).
 */
export const mossTextEncoderContract = (
  component: GraphOwner,
  tokenizer: Pick<Qwen3_5Tokenizer, "maxLength">,
): MossTextEncoderContract => {
  const contract = qwen3_5GraphContract(component, OWNER);
  if (contract.hiddenSize !== MOSS_TEXT_WIDTH) {
    throw new Error(
      `${OWNER}: the text encoder's rows are ${contract.hiddenSize} wide, but the DiT reads text` +
        ` rows of ${MOSS_TEXT_WIDTH} (dit_text's text_embeds)`,
    );
  }
  const { symbols } = component.graph;
  if (symbols.length !== 1) {
    throw new Error(
      `${OWNER}: the text encoder declares symbols [${symbols.join(", ")}] (exactly one, T)`,
    );
  }
  const { maxLength } = tokenizer;
  if (maxLength !== QWEN3_5_MAX_TOKENS || maxLength !== MOSS_MAX_TEXT_TOKENS) {
    throw new Error(
      `${OWNER}: the tokenizer asset's maxLength ${maxLength} differs from the text encoder's` +
        ` bound ${QWEN3_5_MAX_TOKENS} or the DiT's ${MOSS_MAX_TEXT_TOKENS}`,
    );
  }
  return { ...contract, length: symbols[0] };
};

/**
 * The text encoder of a prompt-form request — absent from a state is the caller's assembly, not
 * the request (a plain `Error`).
 */
export const requireMossTextEncoder = <T>(text: T | undefined): T => {
  if (text === undefined) {
    throw new Error(
      `${OWNER}: the request gives prompt strings, but no text encoder was passed (open the Qwen3.5` +
        " text series and its tokenizer and pass them as `text`)",
    );
  }
  return text;
};

/**
 * One prompt's token ids as upstream's `encode_prompt` makes them (module doc "Host side"). `label`
 * names the request field in the messages (`prompt` / `negativePrompt`); every rejection is the
 * template's or the tokenizer's `ModelInputError`.
 */
export const mossPromptIds = (
  tokenizer: Pick<Qwen3_5Tokenizer, "tokenize">,
  prompt: string,
  label: string,
  overflow: Qwen3_5Overflow | undefined,
): Int32Array<ArrayBuffer> => {
  const where = `${OWNER}: ${label}`;
  return tokenizer.tokenize(renderQwen3_5ChatPrompt(prompt, where), {
    label: where,
    ...(overflow === undefined ? {} : { overflow }),
  });
};

/**
 * The bytes the stage needs on a device of `limits` for `prompts` (module doc, step 1). NOTE:
 * `export` is for the first headroom check of `./generate.ts` and the host tests (not on a public
 * surface).
 */
export const mossTextEncoderStageNeed = (
  input: {
    readonly limits: MossDeviceLimits;
    readonly component: Pick<MossTextEncoderComponent, "estimate" | "maxPartBytes">;
    readonly contract: Pick<MossTextEncoderContract, "length">;
  },
  prompts: MossBranches<{ readonly length: number }>,
): number => {
  const { limits, component, contract } = input;
  const longest = Math.max(prompts.positive.length, prompts.negative?.length ?? 0);
  return (
    component.estimate({
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxBufferSize: limits.maxBufferSize,
      bindings: { [contract.length]: longest },
    }).peakAccountedBytes +
    component.maxPartBytes +
    HEADROOM_MARGIN_BYTES
  );
};

/** What the stage reads (the series settled in admission). */
export type MossTextEncoderStageState = {
  readonly gpu: GpuContext;
  readonly component: MossTextEncoderComponent;
  readonly contract: MossTextEncoderContract;
};

/** Options of {@link encodeMossPrompts}. */
export type MossTextEncoderStageOptions = {
  /**
   * Cancels the stage: checked before the headroom check and before each encoding (the encoder's
   * own check — one run is indivisible). Throws `signal.reason` as is, after the encoder is
   * disposed and its memory released.
   */
  readonly signal?: AbortSignal;
};

/**
 * The stage (module doc): encodes the positive prompt, then the negative one (absent at guidance 1),
 * on one encoder opened for this call, and returns each branch's rows as the DiT stage's text.
 */
export const encodeMossPrompts = async (
  state: MossTextEncoderStageState,
  prompts: MossBranches<Int32Array<ArrayBuffer>>,
  options: MossTextEncoderStageOptions = {},
): Promise<MossBranches<MossBranchText>> => {
  const { gpu, component } = state;
  const { signal } = options;
  await settleAbort(signal);
  const bytes = mossTextEncoderStageNeed({ ...state, limits: gpu.limits }, prompts);
  await settleReleasedMemory(gpu);
  if (!(await fitsHeadroom(gpu, bytes))) {
    throw new Error(
      `${OWNER}: the ${MOSS_TEXT_ENCODER} stage needs ${bytes} bytes of VRAM (estimate + largest` +
        ` part + margin) that this device cannot allocate now — text of ${prompts.positive.length}` +
        ` / ${prompts.negative?.length ?? "-"} tokens does not fit`,
    );
  }
  const opened: { encoder?: Qwen3_5TextEncoder } = {};
  let failure: { readonly error: unknown } | undefined;
  try {
    const encoder = await Qwen3_5TextEncoder.open(component, gpu);
    opened.encoder = encoder;
    const encode = async (ids: Int32Array<ArrayBuffer>): Promise<MossBranchText> => ({
      embeddings: {
        data: await encoder.encode(ids, { ...(signal === undefined ? {} : { signal }) }),
        shape: [ids.length, encoder.hiddenSize],
      },
      tokens: ids.length,
    });
    const positive = await encode(prompts.positive);
    const negative = prompts.negative === undefined ? undefined : await encode(prompts.negative);
    return { positive, negative };
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    // MUST (module doc, step 3): dispose, then wait for the release, on every path; a failure to
    // close does not overwrite the main failure.
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      () => opened.encoder?.dispose(),
      () => settleReleasedMemory(gpu),
    ]);
  }
};
