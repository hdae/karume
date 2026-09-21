/**
 * 参照条件の組立て — 参照音声 → DACVAE latent（段 0）と、latent → `speaker` → 平均トークン
 * 前置（段 4）、および条件 state を宣言長へ右 pad する口（段 6 のホスト残置）。
 *
 * 10 段の説明（何をどの順で回すか）は {@link "./pipeline.ts"} 冒頭の doc が正本。ここが持つ
 * のは speaker 条件 1 本ぶんの組立てだけで、text / caption の条件（段 2 / 3）は backbone の
 * 1 セッション 2 run と同じ席で読むため `./pipeline.ts` の `generateLatent` に残る。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import { patchReferenceLatent } from "./host/patch.ts";
import { prependMeanToken } from "./host/pooling.ts";
import { normalizeReference, reflectPadToHop } from "./host/reference.ts";
import type { IrodoriSpeakerInput, IrodoriState } from "./pipeline.ts";
import { asF32, type EmitEvent, f32, outputAt, withStageSession } from "./stage.ts";

/** 条件 1 本ぶんの中間状態（行数を値と一緒に持ち歩く — 幅は config が正本）。 */
export type ConditionState = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly rows: number;
};

/** 条件を載せない（参照なし / caption 空）ときのゼロ供給。右 pad で全 0 行になる。 */
export const emptyCondition = (): ConditionState => ({ data: new Float32Array(0), rows: 0 });

/** 条件 state を宣言長へ右詰め 0 pad する（ADR 0047 のホスト残置）。 */
export const rightPad = (
  state: ConditionState,
  rows: number,
  width: number,
  where: string,
): Float32Array<ArrayBuffer> => {
  if (state.rows > rows) {
    throw new Error(
      `IrodoriPipeline: ${where} の長さ ${state.rows} が宣言長 ${rows} を超えている`,
    );
  }
  if (state.data.length !== state.rows * width) {
    throw new Error(
      `IrodoriPipeline: ${where} の要素数 ${state.data.length} が ${state.rows}×${width} と違う`,
    );
  }
  const padded = new Float32Array(rows * width);
  padded.set(state.data);
  return padded;
};

/**
 * 参照音声 → DACVAE latent（ホスト前処理 + `codec_encoder`）。
 *
 * 切り詰めの上限は `speakerRows` から導く — speaker 条件は「平均トークン 1 本 + patch した
 * 参照」なので、載る参照は `(speakerRows − 1) × speakerPatchSize` フレーム（実重み v4-small で
 * 3,000 フレーム = 120 秒）。**TS 側に秒数を定数で置かない**（config.ts の MUST — 重みを
 * 差し替えたときにホストだけ古い上限を持つ形を作らない）。
 *
 * MUST: 切り詰めは正規化より**前**。後ろに回すと、捨てる区間の音量が LUFS に混ざる
 * （上流も `wav[:, :int(max_ref_seconds·sr)]` を先に取る）。
 */
const encodeReferenceAudio = async (
  state: IrodoriState,
  emit: EmitEvent,
  audio: { readonly data: Float32Array<ArrayBuffer>; readonly sampleRate: number },
): Promise<Float32Array<ArrayBuffer>> => {
  const { config } = state;
  if (audio.sampleRate !== config.sampleRate) {
    // リサンプルは持たない（ADR 0048 の流儀 — 黙って近似せず、変換は呼び出し側の責務にする）。
    throw new Error(
      `IrodoriPipeline: 参照音声が ${audio.sampleRate}Hz（配布形は ${config.sampleRate}Hz）` +
        " — リサンプルは持たないので、あらかじめ変換して渡す",
    );
  }
  const maxSamples = (config.speakerRows - 1) * config.speakerPatchSize * config.hopLength;
  const limited = audio.data.length > maxSamples
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
      return asF32(outputAt(state.codecEncoder, outputs, 0), "codec encoder の出力");
    },
  );
};

/** speaker 条件を組む（参照音声 / 参照 latent / 埋め込み直接指定 / 参照なしのゼロ短絡）。 */
export const encodeSpeaker = async (
  state: IrodoriState,
  emit: EmitEvent,
  input: IrodoriSpeakerInput | undefined,
): Promise<ConditionState> => {
  const { config } = state;
  if (input === undefined) {
    // 参照なしはグラフを回さずゼロを置く（上流の `no_ref` と厳密に一致することは exporter の
    // `_no_reference_evidence` が実測済み）。区間マスクも全 0 になるので寄与は厳密に 0。
    return emptyCondition();
  }
  if ("stateOverride" in input) {
    const { stateOverride } = input;
    if (stateOverride.length === 0 || stateOverride.length % config.speakerDim !== 0) {
      throw new Error(
        `IrodoriPipeline: speaker.stateOverride の長さ ${stateOverride.length} が` +
          ` speakerDim ${config.speakerDim} の正の倍数でない`,
      );
    }
    // MUST: `speaker` グラフも `speaker_norm` も平均トークン前置も通さない（上流
    // `encode_conditions` の `speaker_state_override` 経路）。加工すると、配られた埋め込みが
    // 二重に正規化された別のベクトルとして条件に入る。
    return { data: stateOverride, rows: stateOverride.length / config.speakerDim };
  }
  const latent = "audio" in input
    ? await encodeReferenceAudio(state, emit, input.audio)
    : input.latent;
  const patched = patchReferenceLatent(latent, config.latentDim, config.speakerPatchSize);
  const encoded = await withStageSession(
    state,
    emit,
    "speaker",
    state.speaker,
    {},
    async (run) => {
      const outputs = await run({
        latent: f32(patched.data, [1, patched.tokens, patched.width]),
      });
      return asF32(outputAt(state.speaker, outputs, 0), "speaker の出力");
    },
  );
  // 平均トークンの前置はグラフの外（ADR 0047 決定 4）。
  return {
    data: prependMeanToken(encoded, patched.tokens, config.speakerDim),
    rows: patched.tokens + 1,
  };
};
