/**
 * Assembly of the reference conditions — reference audio → DACVAE latent (step 0), latent →
 * `speaker` → prepending the mean token (step 4), and the helper that right-pads a condition state
 * to its declared length (the part of step 6 left on the host).
 *
 * The description of the 10 steps (what runs in which order) is canonical in the doc at the top of
 * {@link "./pipeline.ts"}. This module holds only the assembly of the one speaker condition; the
 * text / caption conditions (steps 2 / 3) stay in `generateLatent` of `./pipeline.ts`, because they
 * are read in the same place as the 2 runs of backbone in one session.
 *
 * MUST: every module is free of side effects (no execution at import time, no global mutable state
 * — CLAUDE.md).
 */

import { ModelInputError } from "../errors.ts";
import { patchReferenceLatent, SPEAKER_MIN_TOKENS } from "./host/patch.ts";
import { prependMeanToken } from "./host/pooling.ts";
import { normalizeReference, reflectPadToHop } from "./host/reference.ts";
import type { SpeakerInput, IrodoriState } from "./pipeline.ts";
import { asF32, type EmitEvent, f32, outputAt, withStageSession } from "./stage.ts";

/**
 * Intermediate state of one condition (carries the row count along with the values — the width's
 * source of truth is the config).
 */
export type ConditionState = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly rows: number;
};

/**
 * Zero supply when a condition is absent (no reference / empty caption). Right padding turns it
 * into all-zero rows.
 */
export const emptyCondition = (): ConditionState => ({ data: new Float32Array(0), rows: 0 });

/** Zero-pads a condition state on the right to its declared length (left on the host — ADR 0047). */
export const rightPad = (
  state: ConditionState,
  rows: number,
  width: number,
  where: string,
): Float32Array<ArrayBuffer> => {
  if (state.rows > rows) {
    throw new Error(
      `irodori.Pipeline: ${where} length ${state.rows} exceeds the declared length ${rows}`,
    );
  }
  if (state.data.length !== state.rows * width) {
    throw new Error(
      `irodori.Pipeline: ${where} element count ${state.data.length} differs from ${state.rows}×${width}`,
    );
  }
  const padded = new Float32Array(rows * width);
  padded.set(state.data);
  return padded;
};

/** The distribution declarations the reference-audio check reads. */
type ReferenceAudioShape = {
  readonly sampleRate: number;
  readonly hopLength: number;
  readonly speakerRows: number;
  readonly speakerPatchSize: number;
};

/**
 * Maximum length (in samples) the reference audio is truncated to.
 *
 * The maximum is derived from `speakerRows` — the speaker condition is "1 mean token + the patched
 * reference", so the reference that fits is `(speakerRows − 1) × speakerPatchSize` frames (3,000
 * frames = 120 s with the real v4-small weights). **No seconds constant on the TS side** (the MUST
 * of config.ts — no form where only the host keeps an old maximum after the weights are swapped).
 */
const referenceMaxSamples = (config: ReferenceAudioShape): number =>
  (config.speakerRows - 1) * config.speakerPatchSize * config.hopLength;

/**
 * The check of the reference audio's acceptance set (rate and length). Both the entry of generation
 * ({@link assertSpeakerReference}) and the point right before the codec encoder
 * ({@link encodeReferenceAudio}) call this one function.
 *
 * The length is checked as "does the patch after the codec reach the speaker minimum of
 * {@link SPEAKER_MIN_TOKENS} tokens". The codec encoder turns one hop into one frame (input
 * `[1, frames, hopLength]`) and rounds the tail up to a multiple of the hop with reflect padding, so
 * the frame count is `ceil(samples after truncation / hopLength)`. A reference that does not reach
 * it either fails at the patch after the codec encoder ran on the GPU (0 tokens), or delivers S = 1,
 * outside the export minimum, to the speaker graph (1 token).
 *
 * Both violations are the very fields of `SpeakerInput.audio` the caller passed, and the
 * remedy is "convert it / pass a longer reference again", so they are input-caused (ADR 0107
 * decision 2).
 */
const assertReferenceAudio = (
  audio: { readonly data: Float32Array<ArrayBuffer>; readonly sampleRate: number },
  config: ReferenceAudioShape,
): void => {
  if (audio.sampleRate !== config.sampleRate) {
    // There is no resampling (the convention of ADR 0048 — no silent approximation; conversion is
    // the caller's responsibility).
    throw new ModelInputError(
      `irodori.Pipeline: the reference audio is ${audio.sampleRate}Hz (the distribution is ${config.sampleRate}Hz)` +
        " — there is no resampler, so convert it before passing it",
    );
  }
  const samples = Math.min(audio.data.length, referenceMaxSamples(config));
  const tokens = Math.floor(Math.ceil(samples / config.hopLength) / config.speakerPatchSize);
  if (tokens < SPEAKER_MIN_TOKENS) {
    const minSamples = (SPEAKER_MIN_TOKENS * config.speakerPatchSize - 1) * config.hopLength + 1;
    throw new ModelInputError(
      `irodori.Pipeline: ${audio.data.length} samples of reference audio make only ${tokens} speaker tokens` +
        ` (the minimum is ${SPEAKER_MIN_TOKENS} tokens = at least ${minSamples} samples)`,
    );
  }
};

/**
 * Reference audio → DACVAE latent (host preprocessing + `codec_encoder`).
 *
 * The acceptance set is checked with {@link assertReferenceAudio} before the codec encoder runs (the
 * entry of generation has already passed the same check, so this is the contract of the low-level
 * path).
 *
 * MUST: truncation ({@link referenceMaxSamples}) comes **before** normalization. Moved after it,
 * the loudness of the discarded part would leak into the LUFS (upstream also takes
 * `wav[:, :int(max_ref_seconds·sr)]` first).
 */
const encodeReferenceAudio = async (
  state: IrodoriState,
  emit: EmitEvent,
  audio: { readonly data: Float32Array<ArrayBuffer>; readonly sampleRate: number },
): Promise<Float32Array<ArrayBuffer>> => {
  const { config } = state;
  assertReferenceAudio(audio, config);
  const maxSamples = referenceMaxSamples(config);
  const limited =
    audio.data.length > maxSamples
      ? (audio.data.slice(0, maxSamples) as Float32Array<ArrayBuffer>)
      : audio.data;
  const padded = reflectPadToHop(
    normalizeReference(limited, config.sampleRate).data,
    config.hopLength,
  );
  const frames = padded.length / config.hopLength;
  return await withStageSession(
    state,
    emit,
    "codec-encoder",
    state.codecEncoder,
    {},
    async (run) => {
      const outputs = await run({ wav: f32(padded, [1, frames, config.hopLength]) });
      return asF32(outputAt(state.codecEncoder, outputs, 0), "codec encoder output");
    },
  );
};

/**
 * The distribution declarations that decide the shape of the speaker condition (only the fields the
 * checks of direct inputs read).
 */
type SpeakerShape = {
  readonly speakerDim: number;
  readonly speakerRows: number;
  readonly latentDim: number;
  readonly speakerPatchSize: number;
};

/**
 * Checks, on a value the caller passed, that the row count of the speaker condition fits the
 * declared length.
 *
 * {@link rightPad} checks the same relation, but that is the final check after the condition is
 * assembled (the internal invariant "whichever path built it, it does not exceed the declared
 * length"), so it stays a plain `Error`.
 */
const assertSpeakerRows = (rows: number, speakerRows: number, where: string): void => {
  if (rows > speakerRows) {
    throw new ModelInputError(
      `irodori.Pipeline: ${where} makes ${rows} rows of speaker condition, exceeding the declared length ${speakerRows}`,
    );
  }
};

/**
 * Uses a directly given speaker state as the condition as is (upstream's `speaker_state_override`
 * path).
 *
 * MUST: it does **not** go through the `speaker` graph, `speaker_norm` or the mean-token prepend.
 * Processing it would put the distributed embedding into the condition as a different, doubly
 * normalized vector.
 *
 * Both the shape and the row count are those of the embedding the caller passed, so a rejection is
 * input-caused (ADR 0107 decision 2 — the one remedy is "fix the array you pass").
 *
 * NOTE: `export` is for the tests that call the check directly (reaching {@link encodeSpeaker} needs
 * a GPU and real assets). Not exposed on `mod.ts` / the subpath surface (ADR 0008).
 */
export const conditionFromStateOverride = (
  stateOverride: Float32Array<ArrayBuffer>,
  config: SpeakerShape,
): ConditionState => {
  if (stateOverride.length === 0 || stateOverride.length % config.speakerDim !== 0) {
    throw new ModelInputError(
      `irodori.Pipeline: speaker.stateOverride length ${stateOverride.length} is not a positive` +
        ` multiple of speakerDim ${config.speakerDim}`,
    );
  }
  const rows = stateOverride.length / config.speakerDim;
  assertSpeakerRows(rows, config.speakerRows, "speaker.stateOverride");
  return { data: stateOverride, rows };
};

/**
 * Patches a reference latent the caller passed directly, and checks it up to the row maximum (the
 * one check of a direct input).
 *
 * The acceptance condition stays canonical in {@link patchReferenceLatent} alone. That function also
 * receives the output of the codec encoder (= a model output), where a violation is an internal
 * invariant and a plain `Error` is right. Only this path, where the origin is known to be the
 * caller, raises **the thrown type** to input-caused (ADR 0107 decision 2).
 *
 * NOTE: `export` for the same reason as {@link conditionFromStateOverride}.
 */
export const patchRequestedLatent = (
  latent: Float32Array<ArrayBuffer>,
  config: SpeakerShape,
): ReturnType<typeof patchReferenceLatent> => {
  let patched;
  try {
    patched = patchReferenceLatent(latent, config.latentDim, config.speakerPatchSize);
  } catch (cause) {
    throw new ModelInputError(
      `irodori.Pipeline: speaker.latent — ${cause instanceof Error ? cause.message : String(cause)}`,
      { cause },
    );
  }
  // The row count plus 1 for the mean token is the condition length. Checked before the speaker
  // graph runs.
  assertSpeakerRows(patched.tokens + 1, config.speakerRows, "speaker.latent");
  return patched;
};

/**
 * Checks the acceptance set of the reference (audio / latent) at the entry of generation, before the
 * GPU runs (`assertIrodoriRequest` in `pipeline.ts`). The speaker condition is assembled in step 4,
 * so without this check an input error would fail only after the text / caption steps (and, for
 * audio, the codec encoder) ran on the GPU.
 *
 * MUST: the acceptance conditions are not copied — this only calls the same checks as step 4
 * ({@link assertReferenceAudio} / {@link patchRequestedLatent}) (the return values are discarded).
 */
export const assertSpeakerReference = (
  input: SpeakerInput | undefined,
  config: SpeakerShape & ReferenceAudioShape,
): void => {
  if (input === undefined || "stateOverride" in input) return;
  if ("audio" in input) {
    assertReferenceAudio(input.audio, config);
    return;
  }
  patchRequestedLatent(input.latent, config);
};

/**
 * Assembles the speaker condition (reference audio / reference latent / a directly given embedding /
 * the zero short-circuit without a reference).
 */
export const encodeSpeaker = async (
  state: IrodoriState,
  emit: EmitEvent,
  input: SpeakerInput | undefined,
): Promise<ConditionState> => {
  const { config } = state;
  if (input === undefined) {
    // Without a reference, zeros are placed without running the graph (exact agreement with
    // upstream's `no_ref` was measured by the exporter's `_no_reference_evidence`). The segment mask
    // is all 0 too, so the contribution is exactly 0.
    return emptyCondition();
  }
  if ("stateOverride" in input) return conditionFromStateOverride(input.stateOverride, config);
  // The **origin** of a patch violation differs between the 2 paths (the caller's latent =
  // input-caused / the output of the codec encoder = internal invariant), so the condition stays one
  // and only the thrown type differs. On the audio path the row count fits the declared length
  // thanks to the truncation, so that is left to the final check ({@link rightPad}).
  const patched =
    "audio" in input
      ? patchReferenceLatent(
          await encodeReferenceAudio(state, emit, input.audio),
          config.latentDim,
          config.speakerPatchSize,
        )
      : patchRequestedLatent(input.latent, config);
  const encoded = await withStageSession(state, emit, "speaker", state.speaker, {}, async (run) => {
    const outputs = await run({
      latent: f32(patched.data, [1, patched.tokens, patched.width]),
    });
    return asF32(outputAt(state.speaker, outputs, 0), "speaker output");
  });
  // The mean token is prepended outside the graph (ADR 0047 decision 4).
  return {
    data: prependMeanToken(encoded, patched.tokens, config.speakerDim),
    rows: patched.tokens + 1,
  };
};
