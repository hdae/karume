/**
 * Wan の DiT の段 — S 形グラフの取り決め（{@link ditContract}）と denoise のループ
 * （patchify → DiT → unpatchify → CFG → UniPC — {@link runWanDenoise}）。
 *
 * 段の順序と Session の寿命（text → DiT → VAE を 1 段ずつ張って畳む）は `./pipeline.ts` 冒頭の doc が
 * 正本で、ここは DiT の段 1 本を回すだけ。Wan2.1 / 2.2 の class が共有する（ADR 0121 決定 10）。
 * `owner` は文言の接頭辞（Wan2.1 は `"WanPipeline"`・Wan2.2 は `"WanTi2vPipeline"`）。
 *
 * NOTE: 公開型（`WanGenerateEvent` / `WanRunComponent`）は `./pipeline.ts` から `import type` で取る
 * （型だけの参照は消去されるので循環 import にならない）。段 9（I2V）で `vae_encoder` の段が増えても置き場は
 * `./pipeline.ts` のまま — 型だけの参照で循環が起きないので、独立のモジュールへ移す理由が無い。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import type { GpuContext, RunInputs, SessionDiagnostics, SessionOptions } from "@karume/runtime";

import { settleAbort } from "../concurrency/abort.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import type { GraphOwner, ModelComponent } from "../hub/components.ts";
import type { WanGenerateEvent, WanRunComponent } from "./pipeline.ts";
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

/** 部品のキー（系列のグラフ名 = 段 7 の manifest の weights 名）。 */
export const TRANSFORMER = "transformer";

/** `transformer` の容器が宣言する RoPE の素表の資産名（役割 `rope-base`）。 */
export const ROPE_BASE = "rope_base";

/** DiT の S 形グラフの入力名（recipe `wan/export_dit.py` の forward の引数名）。 */
const DIT_TOKENS = "tokens";
const DIT_TIMESTEPS_PROJ = "timesteps_proj";
export const DIT_CONTEXT = "encoder_hidden_states";
const DIT_ROPE_COS = "rope_cos";
const DIT_ROPE_SIN = "rope_sin";
/** TI2V の DiT の追加入力 2 本（recipe `wan/ti2v_export_dit.py` の `INPUT_NAMES` の末尾と同名）。 */
const DIT_TIMESTEPS_PROJ_CONDITION = "timesteps_proj_condition";
const DIT_CONDITION_MASK = "condition_mask";

/** 網羅の外の形（記述子・内部状態が型の外の値を持つ — 黙って t2v として回さない）。 */
const unknownDitInputForm = (form: never, owner: string): Error =>
  new Error(`${owner}: DiT の入力の形 '${String(form)}' は未知（"t2v" / "ti2v"）`);

/** DiT の入力の形ごとのグラフ入力名の集合（未知の形は fail loudly）。 */
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
 * DiT の patch（刻みは上流 transformer の config `patch_size [1, 2, 2]` — アーキ定数）。チャネル数は
 * VAE の chunk グラフの宣言（`latent` の軸 0 — `WanVaeChunkLayout.latentChannels`）から受ける
 * （Wan2.1 は 16）。グラフの `tokens` の最終次元（`C·1·2·2` — Wan2.1 は 64）は {@link ditContract} が
 * 構築時に突き合わせる。
 *
 * NOTE: `export` は家族 admission と、GPU 無しで門を縛るテストのため（`mod.ts` / サブパス面には
 * 出さない — ADR 0008）。
 */
export const wanDitPatch = (latentChannels: number): WanPatchGeometry => ({
  channels: latentChannels,
  patchFrames: 1,
  patchHeight: 2,
  patchWidth: 2,
});

/** グラフ入力の宣言（無ければ fail loudly）。 */
const graphInput = (
  owner: string,
  model: GraphOwner,
  name: string,
): GraphOwner["graph"]["inputs"][number] => {
  const spec = model.graph.inputs.find((input) => input.name === name);
  if (spec === undefined) throw new Error(`${owner}: transformer のグラフ入力 '${name}' が無い`);
  return spec;
};

/** グラフ入力の形（無ければ fail loudly）。 */
const inputShape = (
  owner: string,
  model: GraphOwner,
  name: string,
): readonly (number | string)[] => graphInput(owner, model, name).shape;

/** グラフ入力の dtype と形（ホストが組む dtype・形と違えば fail loudly）。 */
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
      `${owner}: transformer のグラフ入力 '${name}' の dtype ${spec.dtype} が ${dtype} でない`,
    );
  }
  assertDims(owner, spec.shape, expected, `'${name}'`);
};

/** 静的次元（記号次元なら fail loudly）。 */
const staticDim = (
  owner: string,
  dims: readonly (number | string)[],
  axis: number,
  where: string,
): number => {
  const dim = dims.at(axis);
  if (typeof dim !== "number") throw new Error(`${owner}: ${where} の軸 ${axis} が静的でない`);
  return dim;
};

/** 構築時に確かめた DiT のグラフの取り決め。 */
export type DitContract = {
  /** グラフ出力の名前（`[1, S, C·1·2·2]`）。 */
  readonly output: string;
  /** `timesteps_proj [1, W]` の W。 */
  readonly projWidth: number;
  /** `encoder_hidden_states [1, rows, width]` の rows（ゼロで埋める先の行数）。 */
  readonly contextRows: number;
  readonly contextWidth: number;
  /**
   * 宣言との照合に使った patch（{@link runWanDenoise} はこれで patchify / unpatchify する — 同じ値を
   * 照合とループの 2 経路で導かない）。
   */
  readonly patch: WanPatchGeometry;
  /** 宣言と照合した入力の形（{@link runWanDenoise} は条件入力をこれで組むか決める）。 */
  readonly form: WanDitInputForm;
};

/**
 * DiT のグラフ宣言を、ホストが組む入力（`patch`・RoPE の素表）と突き合わせる。`patch` のチャネル数は
 * VAE の宣言から来る（{@link wanDitPatch}）ので、VAE の潜在と DiT の `tokens` のチャネルの食い違いも
 * ここで落ちる。
 *
 * MUST: 構築時に落とす。ホストの前処理は自分の定数で組むので、グラフが別の寸法で焼かれていても
 * ホスト側は最後まで通り、落ちるのは DiT の重みを上げた後の Session の shape 検査になる。取得面では
 * 家族 admission（重みの part を取る前）で呼ぶ — 埋め込み資産との突合（`text-stage.ts` の
 * `assertEmbedsFitContext`）は資産のバイト列が届いてから。
 *
 * MUST: 最終次元だけでなく rank・batch（B = 1 — 決定 5）・可変の S まで照合する。ホストは
 * `tokens [1, S, C·1·2·2]`・`rope_cos / rope_sin [1, S, 1, w]` を組み、出力を
 * `[1, S, C·1·2·2]` として読む（{@link runWanDenoise}）ので、batch 2・固定の S・rank 違いの宣言も
 * Session の shape 検査まで通ってしまう。S は寸法とフレーム数ごとに変わるので記号次元で、4 本とも
 * **同じ記号**であること（IR の記号は上下限を持たないので、上限の突合は要らない）。
 *
 * MUST: グラフ入力の名前の集合は `form`（世代の記述子の `ditInputForm`）の期待とちょうど一致させる。
 * ホストは期待の入力だけを組むので、余分な入力（2.1 の宣言の 6 本目・TI2V の DiT を 2.1 として開く）は
 * Session の入力検査まで通ってしまう。この検査は既存の 5 本の検査の**後**に置く — 入力が欠けた資産の
 * 文言（`'X' が無い`）を変えない。TI2V は追加の 2 本の dtype と形も見る（条件側の時刻は f32 で
 * `timesteps_proj` と同じ `[1, W]`・条件マスクは bool の `[1, S, 1]` で S は `tokens` と同じ記号）。
 *
 * NOTE: `export` は家族 admission と、GPU 無しで門を縛るテストのため（`mod.ts` / サブパス面には
 * 出さない — ADR 0008）。
 */
export const ditContract = (
  transformer: GraphOwner,
  ropeBase: WanRopeBase,
  patch: WanPatchGeometry,
  form: WanDitInputForm,
  owner: string,
): DitContract => {
  // 未知の形は資産の中身に関係なく同じ文言で落とす（集合の照合そのものは 5 本の検査の後）。
  const expected = ditInputNames(form, owner);
  const tokenWidth = wanTokenWidth(patch);
  const tokens = inputShape(owner, transformer, DIT_TOKENS);
  const sequence = tokens.at(1);
  if (typeof sequence !== "string") {
    throw new Error(
      `${owner}: '${DIT_TOKENS}' の軸 1 が記号次元でない（${String(sequence)}）— ` +
        "ホストは S を寸法とフレーム数ごとに変えて渡す",
    );
  }
  assertDims(owner, tokens, [1, sequence, tokenWidth], `'${DIT_TOKENS}'`);
  const [output] = transformer.graph.outputs;
  if (transformer.graph.outputs.length !== 1) {
    throw new Error(
      `${owner}: transformer の出力が ${transformer.graph.outputs.length} 本（1 本の S 形）`,
    );
  }
  assertDims(
    owner,
    valueShape(owner, transformer, output),
    [1, sequence, tokenWidth],
    `transformer の出力 '${output}'`,
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
      `${owner}: transformer のグラフ入力が [${declared.join(", ")}]` +
        `（期待: [${expected.join(", ")}] — 入力の形 '${form}'）`,
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

/** DiT の文脈入力の中身（`[rows, width]` へ詰めた positive と、CFG の uncond 側）。 */
export type WanContexts = {
  readonly positive: Float32Array<ArrayBuffer>;
  readonly negative: Float32Array<ArrayBuffer> | undefined;
};

/** DiT の段が読む構築済みの材料（パイプラインの内部状態のうちこの段の分 — 構造で受ける）。 */
type WanDenoiseState = {
  readonly gpu: GpuContext;
  readonly sessionOptions: SessionOptions;
  readonly transformer: ModelComponent;
  readonly ropeBase: WanRopeBase;
  readonly dit: DitContract;
  readonly onRunDiagnostics?: (
    component: WanRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
};

/**
 * TI2V の DiT の条件入力 2 本（ADR 0121 決定 3）。条件側の時刻の形は生成側と同じ `projShape`。
 * T2V は条件マスクが全て偽で、条件側の時刻は生成側と同じ配列。I2V は先頭の潜在フレームのトークンが真で、
 * 条件側の時刻は t = 0 の proj（{@link wanConditionMask}・{@link runWanDenoise}）。
 */
type DitConditionInputs = {
  readonly proj: Float32Array<ArrayBuffer>;
  readonly mask: Uint32Array<ArrayBuffer>;
  readonly maskShape: readonly number[];
};

/**
 * DiT の 1 回の forward の入力（S 形グラフの入力 — t2v は 5 本・ti2v は `condition` の 2 本を足した
 * 7 本）。ホストが組んだ配列をそのまま渡す（写さない — run が settle するまで書き換えない借用）。
 * `condition` が undefined なら 5 本だけ（キーも順も配列も条件入力の追加前と同じ — Wan2.1 の
 * Session に渡るバイト列を変えない）。
 *
 * MUST: DiT の入力を組むのはこの 1 か所（ADR 0121 決定 10 — 2.1 と 2.2 が共有する）。
 *
 * NOTE: `export` は GPU 無しで入力の組み方を縛るテストのため（`mod.ts` / サブパス面には出さない —
 * ADR 0008）。
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
  ...(input.condition === undefined ? {} : {
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
 * 条件マスク（ADR 0121 決定 3 — ループの前に 1 回だけ作る）。入力の形 "t2v" の DiT は条件入力を持たないので
 * undefined（条件の潜在を渡されたら fail loudly）。"ti2v" は、テキストだけの要求（T2V — 入力の形とは別の軸）なら
 * 全て偽、I2V（`conditioned`）なら先頭の潜在フレームのトークン — トークン添字 `(f·H' + h)·W' + w` で先頭に連続して
 * 並ぶ `rows·cols` 個（1280×704 で 880）— が真。
 *
 * MUST: I2V は時間の patch が 1 のときだけ組む — 先頭のトークンのフレームが先頭の潜在フレームと一致する前提（上流
 * diffusers の `first_frame_mask[0][0][:, ::2, ::2]` も同じ前提）。
 *
 * NOTE: `export` は GPU 無しでマスクの形を縛るテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
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
          `${owner}: 入力の形 't2v' の DiT は条件の潜在を受けない（条件マスクの入力が無い）`,
        );
      }
      return undefined;
    case "ti2v": {
      const mask = new Uint32Array(grid.count);
      if (conditioned) {
        if (patch.patchFrames !== 1) {
          throw new Error(
            `${owner}: 時間の patch ${patch.patchFrames} では先頭の潜在フレームがトークンのフレームに揃わない` +
              "（I2V の条件マスクは時間の patch 1 の前提）",
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
 * モデル入力の先頭の潜在フレームを条件の潜在で置き換えた写し（上流 diffusers の `expand_timesteps` 分岐の
 * `(1 − m)·condition + m·latents` — m は先頭フレームが 0・他が 1。ADR 0121 決定 11）。`latents` は `[C, F, H, W]`、
 * `condition` は 1 フレームの `[C, 1, H, W]` で、上流と同じく全フレームへ broadcast した式の値を要素ごとに書く。
 *
 * 式を逐語に計算する（先頭フレームを写すだけにしない）: m が 0 か 1 なので各項は条件・潜在・符号付きゼロのどれかで、
 * 和は丸めを含まない — f32 の上流と符号付きゼロまで同じ値になる（写すだけだと、潜在が -0 の要素で上流の +0 と割れる）。
 *
 * NOTE: `export` は GPU 無しで置き換えを縛るテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */
export const withWanFirstFrameCondition = (
  latents: Float32Array,
  condition: Float32Array,
  latentShape: readonly [number, number, number, number],
): Float32Array<ArrayBuffer> => {
  const [channels, frames, height, width] = latentShape;
  const plane = height * width;
  if (latents.length !== channels * frames * plane || condition.length !== channels * plane) {
    throw new Error(
      `条件の置き換え: 潜在 ${latents.length} 要素・条件 ${condition.length} 要素が潜在の形 ` +
        `[${latentShape.join(", ")}]（条件は 1 フレーム）と合わない`,
    );
  }
  const out = new Float32Array(latents.length);
  for (let channel = 0; channel < channels; channel += 1) {
    const from = channel * plane;
    for (let frame = 0; frame < frames; frame += 1) {
      // 上流の `first_frame_mask`（先頭フレームが 0）。
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
 * DiT の段（Session を張り、steps 回まわして畳む — `end` は畳んだ後）。
 *
 * `condition`（I2V の条件の潜在 `[C, 1, H, W]` — 正規化済み・有限）を渡すと、上流 diffusers の `expand_timesteps`
 * 分岐の形で条件づける（ADR 0121 決定 11）: 各 step でモデル入力の先頭の潜在フレームだけを置き換え
 * （{@link withWanFirstFrameCondition}）、UniPC は置き換えない潜在で進め、ループの後に 1 回置き換えてから返す。
 * 条件マスクは先頭フレームのトークンが真（{@link wanConditionMask}）、条件側の時刻は t = 0 の proj を 1 回だけ作る。
 * `denoise-step` の `copyLatents` が返すのはスケジューラの状態（置き換える前 — diffusers の callback と同じ値）。
 * 公式 Wan2.2 の形（初期と各 step の後に置き換える）とは途中の潜在の先頭フレームだけが違い、最終出力は同じ値になる
 * （UniPC は要素ごとの更新で、モデル入力は両方の形で同じ — 置き換える要素は他の要素へ流れない）。
 *
 * `condition` が undefined（テキストだけの要求）なら今までと同じ配列だけを回す（条件マスクは全て偽・条件側の時刻は
 * 生成側と同じ配列・モデル入力は潜在そのもの — T2V の sha 行を変えない）。
 *
 * MUST: 各 step の更新後の潜在の有限性を見る（O(N) のホスト走査 — 81 フレームで 1 step 約 210 万
 * 要素）。非有限の潜在を黙って次の step と VAE の段へ渡すと、VAE の後のクランプが ±Inf を ±1 に
 * 変えて検出できなくなる。
 */
export const runWanDenoise = async (
  state: WanDenoiseState,
  plan: WanGenerationKnobs,
  contexts: WanContexts,
  condition: Float32Array<ArrayBuffer> | undefined,
  emit: (event: WanGenerateEvent) => Promise<void>,
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
  // 未知の形・条件を受けない形はここ（Session を張る前）で落ちる。
  const conditionMask = wanConditionMask(dit.form, grid, patch, condition !== undefined, owner);
  const maskShape = [1, grid.count, 1];
  // 条件側の時刻（上流は `first_frame_mask · t` — 先頭フレームのトークンは t = 0）。step に依らないので 1 回だけ作る。
  const conditionProj = condition === undefined ? undefined : timestepsProj(0, dit.projWidth);
  const { positive, negative } = contexts;
  const elements = latentShape.reduce((product, dim) => product * dim, 1);
  let current: Float32Array<ArrayBuffer> = plan.initial.kind === "latents"
    ? Float32Array.from(plan.initial.data)
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
      // T2V の条件側の時刻は生成側と同じ配列（決定 3 — どちらも読むだけの借用）。
      const conditionInputs = conditionMask === undefined
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
      return unpatchifyTokens(asF32(outputs[dit.output], "DiT の出力"), latentShape, patch);
    };
    const sampler = new WanUniPcSampler(schedule, WAN_UNIPC_CONFIG);
    for (let index = 0; index < plan.steps; index += 1) {
      // 各 step の前（step の 2 回の forward は不可分 — 中断は次の step の前で効く）。
      await settleAbort(signal);
      const timestep = schedule.timesteps[index];
      const proj = timestepsProj(timestep, dit.projWidth);
      // CFG は uncond → cond の逐次 2 回（B = 1 — 決定 5）。同じ潜在なので patchify は 1 回。
      // 合成はホストで。I2V はモデル入力だけ先頭フレームを置き換える（UniPC へは置き換えない `current`）。
      const modelInput = condition === undefined
        ? current
        : withWanFirstFrameCondition(current, condition, latentShape);
      const tokens = patchifyLatents(modelInput, latentShape, patch);
      const uncond = negative === undefined ? undefined : await predict(tokens, proj, negative);
      const cond = await predict(tokens, proj, positive);
      const velocity = uncond === undefined
        ? cond
        : wanClassifierFreeGuidance(cond, uncond, plan.guidance);
      current = sampler.step(velocity, current);
      const broken = firstNonFinite(current);
      if (broken !== -1) {
        throw new Error(
          `${owner}: step ${index + 1}/${plan.steps} の更新後の潜在の要素 ${broken} が非有限` +
            `（${current[broken]}）— DiT の出力・CFG の合成・UniPC の更新のどこかが溢れた` +
            "（VAE の段へは渡さない）",
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
    // MUST: 畳む失敗で本体の失敗（run の失敗・`onEvent` の throw・非有限の門）を上書きしない —
    // VAE の段と同じ形（`disposeSteps` の doc）。両方が落ちたら本体を先頭にした AggregateError。
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      () => session.dispose(),
    ]);
  }
  // I2V はループの後に 1 回置き換えてから VAE の段へ渡す（上流 diffusers と同じ — 置き換えた要素は有限の条件と
  // 門を通った潜在だけから作るので、有限性は保たれる）。
  const latents = condition === undefined
    ? current
    : withWanFirstFrameCondition(current, condition, latentShape);
  await emit({ kind: "stage", component: "transformer", at: "end" });
  return latents;
};
