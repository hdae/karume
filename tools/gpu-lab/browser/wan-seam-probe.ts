/**
 * Wan2.2 TI2V 5B の DiT のパスの切れ目の下見（perf-ledger H-37 段 1）— 1 step のホストの処理と、DiT 1 パスの
 * 入力の上げ・出力の読み戻しと同じバイト数の転送を、生成とは別に実寸で測るマイクロベンチ。生成の中の区間
 * （`wan-plan.ts` の `summarizeWanSeams`）と並べて、切れ目のどこに時間が要るかを分ける材料にする。
 *
 * GPU lab の Wan のタブ（「切れ目の下見」）と Deno の CLI（`tools/wan-seams/main.ts`）が同じこのモジュールを使う。
 *
 * - **寸法**（{@link wanSeamSizes}）: DiT 1 パスの入力 7 本と出力のバイト数（純関数 — `wan-seam-probe_test.ts`）。
 * - **ホストの処理**（{@link benchWanHostSeams}）: 製品のホストの関数（`packages/models/src/wan/`）を実寸の配列で回す。
 * - **転送**（{@link benchWanTransfer}）: 渡された device で writeBuffer → 空の submit → `onSubmittedWorkDone`（上げ）と
 *   copyBufferToBuffer → MAP_READ の mapAsync → `getMappedRange().slice(0)`（読み戻し）。runtime の run の経路そのもの
 *   ではなく、同じ本数・バイト数の素の WebGPU の操作（runtime の上げと読み戻しの下限の目安）。
 * - **見積もり**（{@link estimateWanSeamStep}）と **照合**（{@link checkWanSeamInputBytes}）。
 *
 * NOTE: ホストと転送のベンチは時間と GPU に依存するのでテストしない（寸法と照合と見積もりの算式だけを縛る）。
 *
 * MUST: 副作用ゼロ。WebGPU のグローバル（`GPUBufferUsage` / `GPUMapMode`）はモジュールスコープで読まない
 * （関数の中で読む — 読まない環境でも import が落ちないように）。
 */
import { TEMPORAL_COMPRESSION } from "../../../packages/models/src/wan/plan.ts";
import { patchifyLatents, unpatchifyTokens } from "../../../packages/models/src/wan/dit-tokens.ts";
import { timestepsProj } from "../../../packages/models/src/wan/dit-timestep.ts";
import { firstNonFinite } from "../../../packages/models/src/wan/graph-io.ts";
import {
  WAN_UNIPC_CONFIG,
  wanClassifierFreeGuidance,
  WanUniPcSampler,
  wanUniPcSchedule,
} from "../../../packages/models/src/wan/scheduler.ts";
import {
  wanDitPatch,
  withWanFirstFrameCondition,
} from "../../../packages/models/src/wan/dit-loop.ts";
import { medianOf, type WanSize } from "./wan-plan.ts";

const F32_BYTES = 4;

/**
 * Wan2.2 TI2V 5B の DiT の入出力の寸法を決める定数。`WanPipelineConfig` は公開していないので、上流の事実を
 * ここに写す（照合は生成の後の {@link checkWanSeamInputBytes} — 写し違いはそこで目に見える）。
 *
 * 出典: docs/research/2026-10-04-wan22-ti2v-5b-upstream.md（VAE の z_dim 48・圧縮 4×16×16 / DiT の
 * `patch_size [1,2,2]`・head_dim 128・text_len 512 / text_dim 4096・freq_dim 256）と
 * `packages/models/src/wan/dit-loop.ts` の入力の取り決め（`tokens [1,S,C·1·2·2]`・`rope_cos / rope_sin
 * [1,S,1,head_dim]`・`encoder_hidden_states [1,512,4096]`・`timesteps_proj` と `timesteps_proj_condition`
 * `[1,256]`・`condition_mask [1,S,1]` の bool〈GPU では 1 要素 4 バイト〉）。時間圧縮は製品の
 * `TEMPORAL_COMPRESSION`（潜在フレーム = (frames − 1) / 4 + 1）。
 */
const WAN22_DIT = {
  latentChannels: 48,
  spatialCompression: 16,
  /** DiT の patch（時間 1・空間 2×2）— 1 token の画素の辺は 16 × 2 = 32。 */
  patchSpatial: 2,
  ropeWidth: 128,
  contextRows: 512,
  contextWidth: 4096,
  projWidth: 256,
} as const;

/** DiT 1 パスの入力の名前（`dit-loop.ts` の TI2V の 7 本）。 */
export const WAN_SEAM_INPUTS = [
  "tokens",
  "rope_cos",
  "rope_sin",
  "encoder_hidden_states",
  "condition_mask",
  "timesteps_proj",
  "timesteps_proj_condition",
] as const;

export type WanSeamInputName = (typeof WAN_SEAM_INPUTS)[number];

export type WanSeamRequest = WanSize & { readonly frames: number };

export type WanSeamSizes = WanSeamRequest & {
  /** 潜在の形 `[C, F, H, W]`。 */
  readonly latentShape: readonly [number, number, number, number];
  /** DiT のトークン数 S。 */
  readonly tokens: number;
  /** 入力 1 本ごとのバイト数。 */
  readonly inputs: Readonly<Record<WanSeamInputName, number>>;
  readonly inputBytes: number;
  /** 出力 `[1, S, 192]` の f32 のバイト数。 */
  readonly outputBytes: number;
};

/** DiT 1 パスの入出力のバイト数（Wan2.2 TI2V 5B）。4n+1 でないフレーム数・32 で割れない寸法は落とす。 */
export const wanSeamSizes = ({ frames, width, height }: WanSeamRequest): WanSeamSizes => {
  const pixelsPerToken = WAN22_DIT.spatialCompression * WAN22_DIT.patchSpatial;
  if (!Number.isInteger(frames) || frames < 1 || (frames - 1) % TEMPORAL_COMPRESSION !== 0) {
    throw new RangeError(`フレーム数 ${frames} が 4n+1 でない`);
  }
  if (
    !Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0 ||
    width % pixelsPerToken !== 0 || height % pixelsPerToken !== 0
  ) {
    throw new RangeError(`寸法 ${width}x${height} が ${pixelsPerToken} で割り切れない`);
  }
  const latentShape = [
    WAN22_DIT.latentChannels,
    (frames - 1) / TEMPORAL_COMPRESSION + 1,
    height / WAN22_DIT.spatialCompression,
    width / WAN22_DIT.spatialCompression,
  ] as const;
  const tokens = latentShape[1] * (height / pixelsPerToken) * (width / pixelsPerToken);
  const tokenWidth = WAN22_DIT.latentChannels * WAN22_DIT.patchSpatial ** 2;
  const inputs: Record<WanSeamInputName, number> = {
    tokens: tokens * tokenWidth * F32_BYTES,
    rope_cos: tokens * WAN22_DIT.ropeWidth * F32_BYTES,
    rope_sin: tokens * WAN22_DIT.ropeWidth * F32_BYTES,
    encoder_hidden_states: WAN22_DIT.contextRows * WAN22_DIT.contextWidth * F32_BYTES,
    condition_mask: tokens * F32_BYTES,
    timesteps_proj: WAN22_DIT.projWidth * F32_BYTES,
    timesteps_proj_condition: WAN22_DIT.projWidth * F32_BYTES,
  };
  return {
    frames,
    width,
    height,
    latentShape,
    tokens,
    inputs,
    inputBytes: Object.values(inputs).reduce((total, bytes) => total + bytes, 0),
    outputBytes: tokens * tokenWidth * F32_BYTES,
  };
};

/** 計測の回数（ウォームアップの後の回数の中央値を取る）。 */
const WARMUP = 2;
const MEASURED = 10;
/** UniPC の 1 step は状態が進むので、この step 数の schedule を通した 1 step の中央値。 */
const UNIPC_STEPS = 50;
const UNIPC_SHIFT = 5.0;

/** ホストの処理 1 回の時間（ms・中央値）。 */
export type WanHostSeamMs = {
  readonly patchify: number;
  readonly unpatchify: number;
  readonly cfg: number;
  readonly uniPcStep: number;
  readonly nonFinite: number;
  /** I2V の先頭フレームの置き換え（T2V では回らない）。 */
  readonly firstFrame: number;
  readonly timestepsProj: number;
};

const timeMedian = (now: () => number, action: () => unknown): number => {
  for (let round = 0; round < WARMUP; round += 1) action();
  const spans: number[] = [];
  for (let round = 0; round < MEASURED; round += 1) {
    const started = now();
    action();
    spans.push(now() - started);
  }
  return medianOf(spans);
};

/**
 * 1 step のホストの処理を実寸の配列で測る（同期 — ブラウザでは主スレッドを数秒止める）。値は決定的な
 * 有限値の列（中身は時間に効かない — 有限性の検査は全要素を走査する）。
 */
export const benchWanHostSeams = (
  sizes: WanSeamSizes,
  now: () => number = () => performance.now(),
): WanHostSeamMs => {
  const { latentShape } = sizes;
  const patch = wanDitPatch(WAN22_DIT.latentChannels);
  const elements = latentShape.reduce((product, dim) => product * dim, 1);
  const latents = Float32Array.from({ length: elements }, (_, index) => Math.sin(index));
  const other = Float32Array.from({ length: elements }, (_, index) => Math.cos(index));
  const condition = latents.slice(0, latentShape[0] * latentShape[2] * latentShape[3]);
  const tokens = patchifyLatents(latents, latentShape, patch);
  const schedule = wanUniPcSchedule(UNIPC_STEPS, UNIPC_SHIFT, WAN_UNIPC_CONFIG.numTrainTimesteps);
  const sampler = new WanUniPcSampler(schedule, WAN_UNIPC_CONFIG);
  const uniPcSpans: number[] = [];
  let current = latents;
  for (let index = 0; index < UNIPC_STEPS; index += 1) {
    const started = now();
    current = sampler.step(other, current);
    uniPcSpans.push(now() - started);
  }
  return {
    patchify: timeMedian(now, () => patchifyLatents(latents, latentShape, patch)),
    unpatchify: timeMedian(now, () => unpatchifyTokens(tokens, latentShape, patch)),
    cfg: timeMedian(now, () => wanClassifierFreeGuidance(latents, other, 5)),
    uniPcStep: medianOf(uniPcSpans),
    nonFinite: timeMedian(now, () => firstNonFinite(latents)),
    firstFrame: timeMedian(now, () => withWanFirstFrameCondition(latents, condition, latentShape)),
    timestepsProj: timeMedian(now, () => timestepsProj(999, WAN22_DIT.projWidth)),
  };
};

/** 転送 1 回の時間（ms・中央値）。 */
export type WanTransferMs = {
  /** 入力 7 本の writeBuffer の発行から `onSubmittedWorkDone` の解決まで。 */
  readonly uploadMs: number;
  /** そのうち writeBuffer の発行だけ（ホストの写しの分）。 */
  readonly uploadIssueMs: number;
  /** 出力と同じバイト数の copy の submit から `getMappedRange().slice(0)` まで（staging の作成を含む）。 */
  readonly readbackMs: number;
};

/** 読み戻しの値の検査に使う値（転送のバッファを埋める — f32 で正確に表せる値）。 */
const READBACK_FILL = 1.5;

/**
 * DiT 1 パスの入力と同じ本数・バイト数の上げと、出力と同じバイト数の読み戻しを測る（ウォームアップ 2 回 +
 * 計測 10 回の中央値）。読み戻した先頭と末尾の要素を検査する（違えば落とす）。作ったバッファは全て破棄する。
 */
export const benchWanTransfer = async (
  device: GPUDevice,
  sizes: WanSeamSizes,
  now: () => number = () => performance.now(),
): Promise<WanTransferMs> => {
  const created: GPUBuffer[] = [];
  const buffer = (size: number, usage: number): GPUBuffer => {
    const made = device.createBuffer({ size, usage });
    created.push(made);
    return made;
  };
  try {
    const inputs = WAN_SEAM_INPUTS.map((name) => ({
      gpu: buffer(sizes.inputs[name], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
      host: new Float32Array(sizes.inputs[name] / F32_BYTES).fill(0.5),
    }));
    const source = buffer(
      sizes.outputBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    device.queue.writeBuffer(
      source,
      0,
      new Float32Array(sizes.outputBytes / F32_BYTES).fill(READBACK_FILL),
    );
    await device.queue.onSubmittedWorkDone();
    const upload: number[] = [];
    const issue: number[] = [];
    const readback: number[] = [];
    for (let round = 0; round < WARMUP + MEASURED; round += 1) {
      const started = now();
      for (const input of inputs) device.queue.writeBuffer(input.gpu, 0, input.host);
      const issued = now();
      device.queue.submit([]);
      await device.queue.onSubmittedWorkDone();
      const uploaded = now();
      const staging = buffer(sizes.outputBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(source, 0, staging, 0, sizes.outputBytes);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      const readDone = now();
      staging.destroy();
      if (data[0] !== READBACK_FILL || data.at(-1) !== READBACK_FILL) {
        throw Error(
          `読み戻しの値が違う（先頭 ${data[0]}・末尾 ${data.at(-1)} — ${READBACK_FILL} のはず）`,
        );
      }
      if (round < WARMUP) continue;
      upload.push(uploaded - started);
      issue.push(issued - started);
      readback.push(readDone - uploaded);
    }
    return {
      uploadMs: medianOf(upload),
      uploadIssueMs: medianOf(issue),
      readbackMs: medianOf(readback),
    };
  } finally {
    for (const made of created) made.destroy();
  }
};

/**
 * 1 step（CFG あり — DiT 2 パス）の切れ目の見積もり（ms）。ホスト = patchify + unpatchify × 2 + CFG + UniPC +
 * 有限性の検査 + timesteps_proj（I2V は + 先頭フレームの置き換え）、転送 = 2 ×（上げ + 読み戻し）。
 */
export type WanSeamEstimate = {
  readonly hostT2vMs: number;
  readonly hostI2vMs: number;
  readonly transferMs: number;
  readonly perStepT2vMs: number;
  readonly perStepI2vMs: number;
};

const PASSES_PER_STEP = 2;

export const estimateWanSeamStep = (
  host: WanHostSeamMs,
  transfer: Pick<WanTransferMs, "uploadMs" | "readbackMs">,
): WanSeamEstimate => {
  const hostT2vMs = host.patchify + PASSES_PER_STEP * host.unpatchify + host.cfg +
    host.uniPcStep + host.nonFinite + host.timestepsProj;
  const hostI2vMs = hostT2vMs + host.firstFrame;
  const transferMs = PASSES_PER_STEP * (transfer.uploadMs + transfer.readbackMs);
  return {
    hostT2vMs,
    hostI2vMs,
    transferMs,
    perStepT2vMs: hostT2vMs + transferMs,
    perStepI2vMs: hostI2vMs + transferMs,
  };
};

/** 下見 1 回の結果（JSON の `seamProbe[]` / CLI の行の `probe`）。 */
export type WanSeamProbe = {
  readonly at: string;
  readonly sizes: WanSeamSizes;
  readonly hostMs: WanHostSeamMs;
  readonly transfer: WanTransferMs;
  readonly estimate: WanSeamEstimate;
};

/** ホスト → 転送の順に測って見積もる（`now` は計測の時計）。 */
export const probeWanSeams = async (
  device: GPUDevice,
  request: WanSeamRequest,
  now: () => number = () => performance.now(),
): Promise<WanSeamProbe> => {
  const sizes = wanSeamSizes(request);
  const hostMs = benchWanHostSeams(sizes, now);
  const transfer = await benchWanTransfer(device, sizes, now);
  return {
    at: new Date().toISOString(),
    sizes,
    hostMs,
    transfer,
    estimate: estimateWanSeamStep(hostMs, transfer),
  };
};

/** 入力 1 本あたりの照合の許容（runtime のバッファの整列ぶん）。 */
export const WAN_SEAM_INPUT_SLACK_BYTES = 256;

/** 生成の DiT の Session が所有した入力の総和と、{@link wanSeamSizes} の入力の合計の照合。 */
export type WanSeamInputCheck = {
  readonly expectedBytes: number;
  /** transformer の `SessionDiagnostics.planBacking.inputBytes`（backing が所有する入力バッファの総和）。 */
  readonly planInputBytes: number;
  readonly toleranceBytes: number;
  /** 差が入力 1 本あたり {@link WAN_SEAM_INPUT_SLACK_BYTES} 以内か（偽なら寸法の写しか前提が違う）。 */
  readonly matches: boolean;
};

export const checkWanSeamInputBytes = (
  sizes: Pick<WanSeamSizes, "inputBytes">,
  planInputBytes: number,
): WanSeamInputCheck => {
  const toleranceBytes = WAN_SEAM_INPUTS.length * WAN_SEAM_INPUT_SLACK_BYTES;
  return {
    expectedBytes: sizes.inputBytes,
    planInputBytes,
    toleranceBytes,
    matches: Math.abs(planInputBytes - sizes.inputBytes) <= toleranceBytes,
  };
};

const MIB = 1024 * 1024;

/** 下見の 1 行の要約（状態行と CLI の出力）。 */
export const formatWanSeamProbe = (probe: WanSeamProbe): string => {
  const { sizes, hostMs, transfer, estimate } = probe;
  return `${sizes.frames}f ${sizes.width}x${sizes.height}（S ${sizes.tokens}）· ホスト T2V ${
    estimate.hostT2vMs.toFixed(1)
  } ms（UniPC ${hostMs.uniPcStep.toFixed(1)}・有限性 ${hostMs.nonFinite.toFixed(1)}・CFG ${
    hostMs.cfg.toFixed(1)
  }・patchify ${hostMs.patchify.toFixed(1)}・unpatchify ${hostMs.unpatchify.toFixed(1)}）· 上げ ${
    (sizes.inputBytes / MIB).toFixed(1)
  } MiB ${transfer.uploadMs.toFixed(1)} ms · 読み戻し ${(sizes.outputBytes / MIB).toFixed(1)} MiB ${
    transfer.readbackMs.toFixed(1)
  } ms · 1 step の見積もり T2V ${estimate.perStepT2vMs.toFixed(1)} ms / I2V ${
    estimate.perStepI2vMs.toFixed(1)
  } ms`;
};
