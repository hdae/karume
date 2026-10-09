/**
 * Wan2.2 TI2V の I2V の VAE encoder の段 — 条件画像 1 枚を encode の chunk 0 で潜在にする（ADR 0121 決定 11・段 9）。
 *
 * encoder は記号長の 3 グラフ（recipe `wan/export_vae_encoder.py`）。mid block の attention の前後で分けてあるのは、
 * IR の次元式が記号 1 つの一次式で、attention が作る系列長 h·w を 1 軸に書けないから（ADR 0121 の追記「段 9 の計画の
 * 裁定と段 9a の結果」）:
 *
 * | 部品               | 入力 → 出力                                             | 記号     |
 * | ------------------ | ------------------------------------------------------- | -------- |
 * | `vae_encoder_pre`  | patchify 済みの画像 `[3·p², 1, k·h, k·w]` → `[C, 1, h, w]` | h, w     |
 * | `vae_encoder_attn` | `[C, S]` → `[C, S]`                                     | S（h·w） |
 * | `vae_encoder_post` | `[C, 1, h, w]` → mu `[Cz, 1, h, w]`                     | h, w     |
 *
 * h, w は潜在の高さ・幅、k は入口の倍率（空間の圧縮 ÷ p — p は記述子の `vaePatchSize`）、C は mid block のチャネル数、
 * Cz は潜在のチャネル数。3 Session は同じ GpuContext に張り、1 本の batch で pre → attn → post を enqueue して常駐
 * テンソルで受け渡す（`[C, 1, h, w]` と `[C, S]` は同じバイト列の読み替えなので写し直さない）。読み戻しは mu の 1 回
 * だけで、mu は `normalizeWanLatents` で正規化して DiT の条件の潜在にする。
 *
 * 段の順序と Session の寿命（encoder → text → DiT → VAE を 1 段ずつ張って畳む）は `./pipeline.ts` 冒頭の doc が正本で、
 * ここは encoder の段 1 本を回すだけ。`owner` は文言の接頭辞（`"WanTi2vPipeline"`）。
 *
 * NOTE: `export` はパイプラインの内部と、GPU 無しで門を縛るテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import type { GpuContext, ResidentTensor, Session, SessionDiagnostics } from "@karume/runtime";

import { settleAbort } from "../concurrency/abort.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { settleReleasedMemory } from "../session/settle-released-memory.ts";
import type { GraphOwner, ModelComponent } from "../hub/components.ts";
import type { WanGenerateEvent, WanRunComponent } from "./pipeline.ts";
import type { WanGenerationDescriptor } from "./descriptor.ts";
import type { PlanLayout } from "./plan.ts";
import { assertDims, firstNonFinite, valueOf } from "./graph-io.ts";
import type { WanI2vImageInput } from "./i2v-preprocess.ts";
import { normalizeWanLatents } from "./latents.ts";
import { wanSpatialCompression } from "./tile-decode.ts";

/** 部品のキー（系列のグラフ名 = manifest の weights 名 — recipe `wan/export_vae_encoder.py`）。 */
export const VAE_ENCODER_PRE = "vae_encoder_pre";
export const VAE_ENCODER_ATTN = "vae_encoder_attn";
export const VAE_ENCODER_POST = "vae_encoder_post";

/** encoder の 3 部品（段の中で回す順）。 */
export const WAN_VAE_ENCODER_KEYS: readonly string[] = [
  VAE_ENCODER_PRE,
  VAE_ENCODER_ATTN,
  VAE_ENCODER_POST,
];

/** 条件画像のチャネル数（RGB — patchify の前）。 */
const RGB_CHANNELS = 3;

/** 要素あたりのバイト数（意味論 dtype は全て 4 バイト — ADR 0009）。 */
const BYTES_PER_ELEMENT = 4;

/** 構築時に確かめた encoder の 3 グラフの取り決め（{@link wanVaeEncoderContract}）。 */
export type WanVaeEncoderContract = {
  readonly pre: { readonly input: string; readonly output: string };
  /** `sequence` は系列長の記号（束縛 `S = h·w` の名前）。 */
  readonly attn: { readonly input: string; readonly output: string; readonly sequence: string };
  /** `height` / `width` は post の記号（常駐の入力は束縛の源にならないので bindings で渡す）。 */
  readonly post: {
    readonly input: string;
    readonly output: string;
    readonly height: string;
    readonly width: string;
  };
  /** mid block のチャネル数（pre の出口 = attn の系列の幅 = post の入口）。 */
  readonly hiddenChannels: number;
  /** 潜在のチャネル数（post の出口 = mu の本数 — VAE decoder の潜在と同じ）。 */
  readonly latentChannels: number;
  /** pre の入口のチャネル数（`3·p²` — patchify 済みの画像）。 */
  readonly patchChannels: number;
  /** pre の入口の空間の倍率 k（patchify 済みの画像の辺 = k × 潜在の辺）。 */
  readonly inputScale: number;
  /** 空間の圧縮（画像の画素 / 潜在 — decoder と同じ 1 本 `wanSpatialCompression`）。 */
  readonly compression: number;
};

/** encoder の 3 部品（構築時に開いた供給口と、admission で確かめた取り決め）。 */
export type WanVaeEncoder = {
  readonly pre: ModelComponent;
  readonly attn: ModelComponent;
  readonly post: ModelComponent;
  readonly contract: WanVaeEncoderContract;
};

/** 記号 1 つの次元を係数 k 倍した正準表記（IR の次元式 `coeff·sym` — 係数 1 は省く）。 */
const scaledSymbol = (scale: number, symbol: string): string =>
  scale === 1 ? symbol : `${scale}${symbol}`;

/** グラフの唯一の入力と出力（f32 の 1 本 → 1 本 — ホストはそれだけを組んで読む）。 */
const soleIo = (owner: string, part: string, model: GraphOwner) => {
  const { inputs, outputs } = model.graph;
  if (inputs.length !== 1 || outputs.length !== 1) {
    throw new Error(
      `${owner}: ${part} の入出力が ${inputs.length} 本 → ${outputs.length} 本（1 本 → 1 本だけを組む）`,
    );
  }
  const [input] = inputs;
  const [output] = outputs;
  const value = valueOf(owner, model, output);
  const declared: readonly (readonly [string, string])[] = [
    [`入力 '${input.name}'`, input.dtype],
    [`出力 '${output}'`, value.dtype],
  ];
  for (const [what, dtype] of declared) {
    if (dtype !== "f32") {
      throw new Error(`${owner}: ${part} の${what} の dtype ${dtype} が f32 でない`);
    }
  }
  return { input: input.name, inputShape: input.shape, output, outputShape: value.shape };
};

/** 軸の宣言が記号そのもの（係数 1・オフセット 0 — 宣言の記号の 1 つ）であること。 */
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
      `${owner}: ${part} の${where} の軸 ${axis} が記号そのものでない（${String(dim)}）`,
    );
  }
  return dim;
};

/** 軸の宣言が静的次元であること。 */
const staticAt = (
  owner: string,
  part: string,
  dims: readonly (number | string)[],
  axis: number,
  where: string,
): number => {
  const dim = dims.at(axis);
  if (typeof dim !== "number") {
    throw new Error(`${owner}: ${part} の${where} の軸 ${axis} が静的でない（${String(dim)}）`);
  }
  return dim;
};

/** 宣言の記号の集合がちょうど `expected` であること（束縛の源はホストが渡す分だけ）。 */
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
      `${owner}: ${part} の記号が [${model.graph.symbols.join(", ")}]（期待: [${
        expected.join(", ")
      }]）`,
    );
  }
};

/** 寸法 1 つに対する encoder の束縛（{@link wanVaeEncoderBinding}）。 */
export type WanVaeEncoderBinding = {
  /** 潜在の高さ・幅（post の記号の値・attn の系列長は `height·width`）。 */
  readonly height: number;
  readonly width: number;
  /** pre の入口の形 `[3·p², 1, k·h, k·w]`（前処理の出力の形と一致すること）。 */
  readonly inputShape: readonly [number, number, number, number];
};

/**
 * 出力寸法 `size` の encoder の束縛（潜在の高さ・幅と pre の入口の形）。割り切れない寸法は素の `Error`（受理寸法は
 * 入口の門が先に絞るので、ここで落ちるのは資産と記述子の齟齬）。
 *
 * MUST: admission の被覆の門（{@link wanVaeEncoderContract}）と段（{@link encodeWanImageStage}）は同じこの 1 本で束縛を
 * 導く — 2 経路で導くと、admission が通した寸法を段が別の形で回しうる。
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
      `${owner}: ${size.width}×${size.height} は VAE encoder の空間の圧縮 ${contract.compression} で` +
        "潜在が正の整数にならない",
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
 * encoder の 3 グラフの宣言を、ホストが組む入力と読む出力・VAE decoder の潜在と突き合わせ、受理寸法を全部覆えるかを
 * 見る（家族 admission の門 — 重みの part を取る前）。
 *
 * - 各グラフは f32 の入力 1 本 → 出力 1 本。pre は `[3·p², 1, k·h, k·w]` → `[C, 1, h, w]`（k = 空間の圧縮 ÷ p・
 *   記号は h, w の 2 つだけ）、attn は `[C, S]` → `[C, S]`（記号は S だけ）、post は `[C, 1, h, w]` → `[Cz, 1, h, w]`
 *   （記号は h, w の 2 つだけ）。C は 3 本で同じ・Cz は decoder の潜在のチャネル数。
 * - 被覆: 記号長のグラフに寸法ごとの形は無いので、受理寸法ごとに束縛（{@link wanVaeEncoderBinding}）が整数で
 *   導けること。入口の倍率 k は decoder の空間の圧縮から導いた値で照合するので、圧縮の違う encoder は形の検査で落ちる。
 *
 * MUST: admission で呼ぶ。ホストは自分の定数で入力を組み、束縛を渡すので、宣言が違っていても落ちるのは 3 つの
 * Session を張った後の batch の中（失敗の帰属は batch 単位で、どのグラフかも絞れない）になる。
 *
 * NOTE: 記号の名前はグラフの宣言から読む（ホストに綴らない — 束縛は名前で渡す）。
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
      `${owner}: 空間の圧縮 ${compression} が patchify の倍率 ${patchSize} で割り切れない（VAE encoder の入口の倍率）`,
    );
  }
  const patchChannels = RGB_CHANNELS * patchSize * patchSize;

  const pre = soleIo(owner, VAE_ENCODER_PRE, parts.pre);
  const preWhere = `出力 '${pre.output}'`;
  const hiddenChannels = staticAt(owner, VAE_ENCODER_PRE, pre.outputShape, 0, preWhere);
  const preHeight = symbolAt(owner, VAE_ENCODER_PRE, parts.pre, pre.outputShape, 2, preWhere);
  const preWidth = symbolAt(owner, VAE_ENCODER_PRE, parts.pre, pre.outputShape, 3, preWhere);
  assertSymbols(owner, VAE_ENCODER_PRE, parts.pre, [preHeight, preWidth]);
  assertDims(
    owner,
    pre.outputShape,
    [hiddenChannels, 1, preHeight, preWidth],
    `${VAE_ENCODER_PRE} の${preWhere}`,
  );
  assertDims(
    owner,
    pre.inputShape,
    [
      patchChannels,
      1,
      scaledSymbol(inputScale, preHeight),
      scaledSymbol(inputScale, preWidth),
    ],
    `${VAE_ENCODER_PRE} の入力 '${pre.input}'`,
  );

  const attn = soleIo(owner, VAE_ENCODER_ATTN, parts.attn);
  const sequence = symbolAt(
    owner,
    VAE_ENCODER_ATTN,
    parts.attn,
    attn.inputShape,
    1,
    `入力 '${attn.input}'`,
  );
  assertSymbols(owner, VAE_ENCODER_ATTN, parts.attn, [sequence]);
  for (
    const [what, dims] of [
      [`入力 '${attn.input}'`, attn.inputShape],
      [`出力 '${attn.output}'`, attn.outputShape],
    ] as const
  ) {
    assertDims(owner, dims, [hiddenChannels, sequence], `${VAE_ENCODER_ATTN} の${what}`);
  }

  const post = soleIo(owner, VAE_ENCODER_POST, parts.post);
  const postWhere = `入力 '${post.input}'`;
  const postHeight = symbolAt(owner, VAE_ENCODER_POST, parts.post, post.inputShape, 2, postWhere);
  const postWidth = symbolAt(owner, VAE_ENCODER_POST, parts.post, post.inputShape, 3, postWhere);
  assertSymbols(owner, VAE_ENCODER_POST, parts.post, [postHeight, postWidth]);
  assertDims(
    owner,
    post.inputShape,
    [hiddenChannels, 1, postHeight, postWidth],
    `${VAE_ENCODER_POST} の${postWhere}`,
  );
  assertDims(
    owner,
    post.outputShape,
    [layout.latentChannels, 1, postHeight, postWidth],
    `${VAE_ENCODER_POST} の出力 '${post.output}'`,
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

/** encoder の段が読む構築済みの材料（パイプラインの内部状態のうちこの段の分 — 構造で受ける）。 */
type WanVaeEncodeState = {
  readonly gpu: GpuContext;
  readonly vaeEncoder: WanVaeEncoder;
  readonly onRunDiagnostics?: (
    component: WanRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
};

/** 段の 3 Session（段の中で回す順）。 */
type EncoderSessions = { readonly pre: Session; readonly attn: Session; readonly post: Session };

/** 受け渡しの常駐（pre の出口・attn の出口・mu）。 */
type EncoderResidents = {
  readonly pre: ResidentTensor;
  readonly attn: ResidentTensor;
  readonly mu: ResidentTensor;
};

/**
 * 3 グラフを 1 本の batch で回して mu を読む（読み戻しは batch の決着の 1 回 — フェンスも 1 本）。
 *
 * MUST: 区間の中で Session の構築・`run`・`dispose` を発行しない（`GpuContext.beginBatch` の MUST — 区間ロックの
 * 自己デッドロック）。Session と常駐は呼び手が区間の前に張る。
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
    // 常駐の入力は記号の束縛の源にならない（バイト列と大きさしか持たない）ので、記号は bindings で渡す。
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
    // MUST: 区間を閉じてロックを返す（閉じないと同じ device の次の batch / run が永久に待つ）。
    // 失敗の本体は呼び手へ投げ直す（finish 側の決着は区間が記録した同じ失敗の写しなので捨てる — VAE decoder の
    // `decodeWanVaeTile` と同じ形）。
    await batch.finish().catch(() => undefined);
    throw cause;
  }
  const read = await batch.finishAndRead({ mu: residents.mu });
  return read["mu"];
};

/**
 * encoder の段（3 Session と受け渡しの常駐を張り、1 本の batch で回し、畳んで解放を待つ — `end` は待った後）。返すのは
 * 正規化した条件の潜在 `[Cz, 1, h, w]`（`normalizeWanLatents` — DiT の段は先頭の潜在フレームをこれで置き換える）。
 *
 * MUST: DiT の段を張る前に畳み、解放を待つ（ADR 0121 決定 11 — DiT の段の容量の余裕は 0.4〜0.8 GiB しか無く、
 * encoder の確保〈1280×704 で重み 103 MiB・計画の backing 900 MiB〉を残さない。B570 は `destroy()` の解放が次の
 * device poll まで遅れる — `settleReleasedMemory` の doc）。失敗した段も畳んでから解放を待つ（同じ pipeline の次の
 * 生成の DiT の段に残さない）。畳む失敗と待ちの失敗で本体の失敗を上書きしない（DiT の段と同じ形）。
 * Session の実行オプションは `{}`（quant の `session` は DiT の Session だけが受ける — VAE decoder と同じ取り決め）。
 * MUST: Session → 常駐の順で畳む（焼き込みの参照が残る間、常駐の破棄は拒まれる）。
 *
 * MUST: 正規化の後の有限性を見る（O(Cz·h·w)）。非有限の条件の潜在を DiT へ渡すと、置き換えの式
 * `(1 − m)·cond + m·latents` の上では落ちる所が step 1 の潜在の門になり、文言が DiT を指す（真因の段を取り違える）。
 */
export const encodeWanImageStage = async (
  state: WanVaeEncodeState,
  image: WanI2vImageInput,
  generation: Pick<WanGenerationDescriptor, "latents">,
  emit: (event: WanGenerateEvent) => Promise<void>,
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
      `${owner}: 条件画像の前処理の形 [${image.shape.join(", ")}] が VAE encoder の入口 ` +
        `[${binding.inputShape.join(", ")}] と違う`,
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
    // run の前（3 グラフの batch は不可分 — 中断はここで効く）。
    await settleAbort(signal);
    const read = await runEncoderBatch(
      state.gpu,
      opened,
      carried,
      contract,
      binding,
      image.pixels,
    );
    observe?.(VAE_ENCODER_PRE, opened.pre.diagnostics());
    observe?.(VAE_ENCODER_ATTN, opened.attn.diagnostics());
    observe?.(VAE_ENCODER_POST, opened.post.diagnostics());
    const mu = new Float32Array(read);
    latents = normalizeWanLatents(mu, generation.latents);
    const broken = firstNonFinite(latents);
    if (broken !== -1) {
      const pixel = broken % area;
      throw new Error(
        `${owner}: 条件画像の潜在（VAE encoder の mu の正規化の後）の channel ${
          Math.floor(broken / area)
        }・位置 (x=${pixel % binding.width}, y=${Math.floor(pixel / binding.width)}) が非有限` +
          `（${latents[broken]}）— DiT の段へは渡さない`,
      );
    }
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    // 解放待ちは失敗した段でも置く — 呼び手が同じ pipeline で次の生成を回すと、その DiT の段が encoder の確保の
    // 解放を待たずに張られる。待ちの失敗は畳む失敗と同じ扱い（本体の失敗を上書きしない）。
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
