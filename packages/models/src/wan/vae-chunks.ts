/**
 * Wan2.1 の動画 VAE の **chunk グラフ 2 本**（`vae_decoder_first` / `vae_decoder_next`）を、常駐の
 * 因果キャッシュで回すホスト側（ADR 0118 決定 2 と追記 2026-10-02）。タイル 1 枚ぶんの chunk 列まで
 * を持つ — タイル計画・ブレンド・貼り付けとクランプは段 5 の層。
 *
 * ## グラフの取り決め（recipe `tools/export-recipes/wan/vae_patch.py` が書く形）
 *
 * - 入力: 潜在 1 フレーム `latent [C,1,t,t]`（逆正規化 `z / (1/std) + mean` はホスト —
 *   `latents.ts`）と cache `cache_NN [Cin,2,h,w]`（NN は上流の `feat_idx` の順）。
 * - 出力: クランプ前のフレーム（first は `[3,1,8t,8t]`・next は `[3,4,8t,8t]`）と、更新後の cache
 *   （**入力と同じ順** — 出力 1+k が cache 入力 k の更新後）。
 * - first は upsample3d の `time_conv` の cache を持たない（最初の chunk では走らない — 上流の
 *   `'Rep'`）。その cache はタイルの頭のゼロのまま next の 1 回目が読む。
 *
 * タイル辺 `t` と cache の形はここに literal で置かず、開いた資産のグラフ宣言から導く
 * （{@link wanVaeChunkLayout} — タイル辺の差し替えを配布物だけで済ませる。決定 2）。
 *
 * ## 1 タイル = 1 batch・フレームだけを読み戻す（追記 2026-10-02 の手順）
 *
 * `Session.run` / `enqueueRead` は全グラフ出力（cache 30〜32 本・約 0.56〜0.58 GiB）を読み戻すので
 * **使わない** MUST NOT。手順は `beginBatch` → first を `enqueue({ copyOutputs })` → next を
 * chunk 数 −1 回 → `finishAndRead({ フレームの常駐 })`。cache の出力は `copyOutputs` で同じ常駐へ
 * 写す（入力に束ねた常駐を写し先にも使う — 写しは run の全 dispatch の後に積まれる FIFO なので、
 * 読んでから上書きする順が保たれる。ADR 0054 決定 3）。フレームの常駐は chunk ごとに別に持つ
 * （写しは区間の FIFO で積まれ、読み戻しは決着時に 1 度だけ走るので、同じ常駐へ写すと最後の
 * chunk しか残らない）。
 *
 * ## cache のゼロ化（タイルの頭）
 *
 * first の cache 入力（ゼロであることが因果パディングの意味）と `time_conv` の 2 本（first が
 * 書かず next の 1 回目が読む）の**全部**がゼロである必要がある — 絞り込める対象は無い。
 * runtime の `ResidentTensor` には GPU 側でゼロを書く口（`clearBuffer`）が無いので、
 * `ResidentTensor.write`（`queue.writeBuffer`）でホストからゼロを書く。タイル 32 で 1 タイル
 * 0.58 GiB のホスト書き込みで、B570（ReBAR）では staging が VRAM の heap に載る（ADR 0118
 * 決定 2 が避けたいとした形）。GPU 側でゼロを書く口を足すかは runtime の公開面の判断（未決）。
 */

import type { BatchScope, GpuContext, ResidentTensor, Session, Tensor } from "@karume/runtime";
import type { GraphOwner } from "../hub/components.ts";

/** 潜在の入力名（recipe の forward の引数名がそのまま IR の入力名になる）。 */
export const WAN_VAE_LATENT_INPUT = "latent";

/** cache の入力名（`cache_` + 上流の `feat_idx` の 2 桁）。 */
const CACHE_INPUT = /^cache_\d{2}$/;

/** cache が持ち越すフレーム数（上流の `CACHE_T`）。 */
const CACHE_FRAMES = 2;

/** first / next の chunk が出すフレーム数（時間 upsample 2 段 = 4 倍・最初の chunk は 1 枚）。 */
const FIRST_FRAMES = 1;
const NEXT_FRAMES = 4;

/** RGB の 3 チャネル。 */
const SAMPLE_CHANNELS = 3;

/** 要素あたりのバイト数（意味論 dtype は全て 4 バイト — ADR 0009）。 */
const BYTES_PER_ELEMENT = 4;

/** chunk グラフの宣言が取り決め（このモジュールの doc）から外れた。 */
export class WanVaeChunkError extends Error {
  override readonly name = "WanVaeChunkError";
}

/** chunk グラフ 1 本の入出力（検査済み）。 */
export type WanVaeChunkGraph = {
  /** cache の入力名（グラフ入力の順）。 */
  readonly caches: readonly string[];
  /** フレームの出力名。 */
  readonly frameOutput: string;
  /** 更新後の cache の出力名（`caches[k]` の更新後が `cacheOutputs[k]`）。 */
  readonly cacheOutputs: readonly string[];
  /** フレームの形 `[3, T, 8t, 8t]`（T は first 1・next 4）。 */
  readonly frameShape: readonly number[];
};

/** chunk グラフ 2 本の取り決め（開いた資産のグラフ宣言から導いた値）。 */
export type WanVaeChunkLayout = {
  /** 潜在のチャネル数（`latent` の軸 0）。 */
  readonly latentChannels: number;
  /** 潜在タイルの辺（`latent` の軸 2 / 3）。 */
  readonly tile: number;
  /** 出力フレームの辺（`8t`）。 */
  readonly sampleTile: number;
  /** cache の全体（next の入力の順・名前 → 形）。 */
  readonly cacheShapes: ReadonlyMap<string, readonly number[]>;
  readonly first: WanVaeChunkGraph;
  readonly next: WanVaeChunkGraph;
};

const staticShape = (dims: readonly (number | string)[], where: string): readonly number[] =>
  dims.map((dim, axis) => {
    if (typeof dim !== "number") {
      throw new WanVaeChunkError(`${where} の軸 ${axis} が静的次元でない（${String(dim)}）`);
    }
    return dim;
  });

const sameShape = (a: readonly number[], b: readonly number[]): boolean =>
  a.length === b.length && a.every((dim, axis) => dim === b[axis]);

const elements = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

/** グラフの値の静的な形（宣言の無い名前は fail loudly — `Object.hasOwn` で引く）。 */
const valueShape = (owner: GraphOwner, name: string, where: string): readonly number[] => {
  if (!Object.hasOwn(owner.graph.values, name)) {
    throw new WanVaeChunkError(`${where}: 値 '${name}' の宣言が無い`);
  }
  return staticShape(owner.graph.values[name].shape, `${where} の '${name}'`);
};

/** 1 本のグラフ宣言を検査し、入出力の取り決めを引く（潜在の形も返す）。 */
const chunkGraph = (
  owner: GraphOwner,
  where: string,
  frames: number,
): { readonly graph: WanVaeChunkGraph; readonly latent: readonly number[] } => {
  const { inputs, outputs } = owner.graph;
  const [latentSpec, ...cacheSpecs] = inputs;
  if (latentSpec?.name !== WAN_VAE_LATENT_INPUT) {
    throw new WanVaeChunkError(`${where}: 先頭の入力が '${WAN_VAE_LATENT_INPUT}' でない`);
  }
  const latent = staticShape(latentSpec.shape, `${where} の '${WAN_VAE_LATENT_INPUT}'`);
  const [channels, latentFrames, rows, cols] = latent;
  if (latent.length !== 4 || latentFrames !== 1 || rows !== cols || channels < 1) {
    throw new WanVaeChunkError(`${where}: 潜在は [C,1,t,t]（${latent.join(",")}）`);
  }
  if (outputs.length !== inputs.length) {
    throw new WanVaeChunkError(
      `${where}: 出力 ${outputs.length} 本が入力 ${inputs.length} 本と違う（フレーム + cache）`,
    );
  }
  if (cacheSpecs.length === 0) {
    // chunk 間の状態は因果キャッシュだけが運ぶ（cache 0 本では chunk を繋げない）。
    throw new WanVaeChunkError(`${where}: cache の入力が 0 本（cache_NN で chunk を繋ぐ取り決め）`);
  }
  const [frameOutput, ...cacheOutputs] = outputs;
  const frameShape = valueShape(owner, frameOutput, `${where} のフレーム出力`);
  const side = frameShape[2];
  if (!sameShape(frameShape, [SAMPLE_CHANNELS, frames, side, side]) || side % rows !== 0) {
    throw new WanVaeChunkError(
      `${where}: フレームの形 [${frameShape.join(",")}] が [3,${frames},8t,8t] でない`,
    );
  }
  cacheSpecs.forEach((spec, index) => {
    if (!CACHE_INPUT.test(spec.name)) {
      throw new WanVaeChunkError(`${where}: 入力 '${spec.name}' が cache_NN の綴りでない`);
    }
    const shape = staticShape(spec.shape, `${where} の '${spec.name}'`);
    if (shape.length !== 4 || shape[1] !== CACHE_FRAMES) {
      throw new WanVaeChunkError(`${where}: '${spec.name}' は [Cin,2,h,w]（${shape.join(",")}）`);
    }
    const output = cacheOutputs[index];
    const updated = valueShape(owner, output, `${where} の出力`);
    if (!sameShape(shape, updated)) {
      throw new WanVaeChunkError(
        `${where}: 出力 ${
          index + 1
        }（'${output}'）の形が '${spec.name}' と違う（入力と同じ順の取り決め）`,
      );
    }
  });
  return {
    graph: { caches: cacheSpecs.map((spec) => spec.name), frameOutput, cacheOutputs, frameShape },
    latent,
  };
};

/**
 * 開いた資産の first / next のグラフ宣言から、chunk 列の取り決めを導いて検査する。
 *
 * MUST: 形はここで資産から引く（タイル辺・cache の形をホストに literal で置かない — 決定 2）。
 * 2 本の潜在の形・フレームの辺が揃い、first の cache が next の cache の部分列（同じ名前・同じ形・
 * 同じ順）であることを見る（first の外の cache は next の 1 回目でゼロのまま読まれる）。
 */
export const wanVaeChunkLayout = (first: GraphOwner, next: GraphOwner): WanVaeChunkLayout => {
  const firstGraph = chunkGraph(first, "vae_decoder_first", FIRST_FRAMES);
  const nextGraph = chunkGraph(next, "vae_decoder_next", NEXT_FRAMES);
  if (!sameShape(firstGraph.latent, nextGraph.latent)) {
    throw new WanVaeChunkError(
      `潜在の形が 2 本で違う（first [${firstGraph.latent.join(",")}]・next [${
        nextGraph.latent.join(",")
      }]）`,
    );
  }
  const sampleTile = nextGraph.graph.frameShape[2];
  if (firstGraph.graph.frameShape[2] !== sampleTile) {
    throw new WanVaeChunkError("フレームの辺が first と next で違う");
  }
  const cacheShapes = new Map<string, readonly number[]>(
    next.graph.inputs.slice(1).map((spec) => [spec.name, staticShape(spec.shape, spec.name)]),
  );
  let cursor = 0;
  for (const name of firstGraph.graph.caches) {
    const shape = cacheShapes.get(name);
    const firstShape = staticShape(
      first.graph.inputs.find((spec) => spec.name === name)?.shape ?? [],
      name,
    );
    if (shape === undefined || !sameShape(shape, firstShape)) {
      throw new WanVaeChunkError(`first の '${name}' が next に同じ形で無い`);
    }
    const position = nextGraph.graph.caches.indexOf(name);
    if (position < cursor) {
      throw new WanVaeChunkError(`first の cache の順が next と違う（'${name}'）`);
    }
    cursor = position + 1;
  }
  const [latentChannels, , tile] = nextGraph.latent;
  return {
    latentChannels,
    tile,
    sampleTile,
    cacheShapes,
    first: firstGraph.graph,
    next: nextGraph.graph,
  };
};

/** chunk 数 F の出力フレーム数（`1 + 4(F−1)`）。 */
export const wanVaeFrameCount = (chunks: number): number =>
  FIRST_FRAMES + NEXT_FRAMES * (chunks - 1);

/**
 * chunk 列が使う常駐テンソル一式（cache 全部 + chunk ごとのフレーム）。タイルを跨いで使い回す。
 *
 * MUST: {@link WanVaeChunkCaches.zero} と `dispose` は、この常駐を使う batch の決着の後にだけ呼ぶ
 * （`ResidentTensor.write` は発行済みの enqueue を追い越しうる — runtime の契約）。
 */
export class WanVaeChunkCaches {
  readonly layout: WanVaeChunkLayout;
  readonly #gpu: GpuContext;
  readonly #caches: ReadonlyMap<string, ResidentTensor>;
  readonly #zeros: Float32Array<ArrayBuffer>;
  readonly #frames: ResidentTensor[] = [];

  private constructor(
    gpu: GpuContext,
    layout: WanVaeChunkLayout,
    caches: ReadonlyMap<string, ResidentTensor>,
  ) {
    this.#gpu = gpu;
    this.layout = layout;
    this.#caches = caches;
    const largest = Math.max(...[...layout.cacheShapes.values()].map(elements));
    this.#zeros = new Float32Array(largest);
  }

  /** cache の常駐を確保する（WebGPU の新しいバッファはゼロで始まる）。 */
  static async create(gpu: GpuContext, layout: WanVaeChunkLayout): Promise<WanVaeChunkCaches> {
    const caches = new Map<string, ResidentTensor>();
    try {
      for (const [name, shape] of layout.cacheShapes) {
        caches.set(name, await gpu.createResident(elements(shape) * BYTES_PER_ELEMENT, name));
      }
    } catch (cause) {
      for (const resident of caches.values()) resident.dispose();
      throw cause;
    }
    return new WanVaeChunkCaches(gpu, layout, caches);
  }

  /** cache の常駐の合計バイト数。 */
  get cacheBytes(): number {
    return [...this.#caches.values()].reduce((sum, resident) => sum + resident.byteLength, 0);
  }

  /** フレームの常駐の合計バイト数（確保済みの分）。 */
  get frameBytes(): number {
    return this.#frames.reduce((sum, resident) => sum + resident.byteLength, 0);
  }

  /** cache 全部をゼロに書く（タイルの頭 — 絞り込める対象が無い理由はモジュールの doc）。 */
  zero(): void {
    for (const resident of this.#caches.values()) {
      resident.write(this.#zeros.subarray(0, resident.byteLength / BYTES_PER_ELEMENT));
    }
  }

  /** そのグラフの cache 入力（名前 → 常駐）。 */
  inputs(graph: WanVaeChunkGraph): Record<string, ResidentTensor> {
    return Object.fromEntries(graph.caches.map((name) => [name, this.#cache(name)]));
  }

  /** そのグラフの `copyOutputs`（更新後の cache → 同じ常駐・フレーム → `frame`）。 */
  copyOutputs(graph: WanVaeChunkGraph, frame: ResidentTensor): Record<string, ResidentTensor> {
    const copies: Record<string, ResidentTensor> = { [graph.frameOutput]: frame };
    graph.caches.forEach((name, index) => {
      copies[graph.cacheOutputs[index]] = this.#cache(name);
    });
    return copies;
  }

  /** chunk 0..count−1 のフレームの常駐（足りなければ確保する — chunk 0 は first の形）。 */
  async frames(count: number): Promise<readonly ResidentTensor[]> {
    while (this.#frames.length < count) {
      const index = this.#frames.length;
      const shape = index === 0 ? this.layout.first.frameShape : this.layout.next.frameShape;
      this.#frames.push(
        await this.#gpu.createResident(elements(shape) * BYTES_PER_ELEMENT, `frame ${index}`),
      );
    }
    return this.#frames.slice(0, count);
  }

  /** 常駐を全て返す（Session より先に呼んでよい — 焼き込みの参照が残っていれば runtime が拒む）。 */
  dispose(): void {
    for (const resident of [...this.#caches.values(), ...this.#frames]) resident.dispose();
  }

  #cache(name: string): ResidentTensor {
    const resident = this.#caches.get(name);
    if (resident === undefined) throw new WanVaeChunkError(`cache '${name}' の常駐が無い`);
    return resident;
  }
}

/** 潜在 `[C,F,t,t]` の chunk 数 F（長さが合わなければ fail loudly）。 */
export const wanVaeChunkCount = (layout: WanVaeChunkLayout, latents: Float32Array): number => {
  const plane = layout.latentChannels * layout.tile * layout.tile;
  const chunks = latents.length / plane;
  if (!Number.isInteger(chunks) || chunks < 1) {
    throw new WanVaeChunkError(
      `潜在の要素数 ${latents.length} が [${layout.latentChannels},F,${layout.tile},${layout.tile}] でない`,
    );
  }
  return chunks;
};

/** 潜在 `[C,F,t,t]` から chunk `index` の 1 フレーム `[C,1,t,t]` を切り出す（写しを返す）。 */
export const wanVaeLatentChunk = (
  layout: WanVaeChunkLayout,
  latents: Float32Array,
  index: number,
): Tensor => {
  const chunks = wanVaeChunkCount(layout, latents);
  // 範囲外を通すと隣のチャネルの平面か末尾の外（空の subarray）を読み、ゼロ混じりの値を黙って返す。
  if (!Number.isInteger(index) || index < 0 || index >= chunks) {
    throw new RangeError(`chunk ${index} が [0, ${chunks}) の外`);
  }
  const area = layout.tile * layout.tile;
  const data = new Float32Array(layout.latentChannels * area);
  for (let channel = 0; channel < layout.latentChannels; channel += 1) {
    const from = (channel * chunks + index) * area;
    data.set(latents.subarray(from, from + area), channel * area);
  }
  return { dtype: "f32", shape: [layout.latentChannels, 1, layout.tile, layout.tile], data };
};

/**
 * chunk 1 本を batch に積む（cache は常駐の入力に束ね、更新後の cache とフレームを `copyOutputs`）。
 *
 * `graph` と `session` は対で渡す（first の chunk は first の宣言と Session）。
 */
export const enqueueWanVaeChunk = (
  batch: BatchScope,
  session: Session,
  graph: WanVaeChunkGraph,
  caches: WanVaeChunkCaches,
  latent: Tensor,
  frame: ResidentTensor,
): Promise<void> =>
  session.enqueue(
    { [WAN_VAE_LATENT_INPUT]: latent, ...caches.inputs(graph) },
    { batch, copyOutputs: caches.copyOutputs(graph, frame) },
  );

/**
 * chunk のフレーム（`[3,T_k,s,s]` を chunk 順に）を時間軸で連結し `[3, 1 + 4(F−1), s, s]` にする。
 */
export const concatWanVaeFrames = (
  layout: WanVaeChunkLayout,
  chunks: readonly ArrayBuffer[],
): Float32Array<ArrayBuffer> => {
  const plane = layout.sampleTile * layout.sampleTile;
  const total = wanVaeFrameCount(chunks.length);
  const out = new Float32Array(SAMPLE_CHANNELS * total * plane);
  let offset = 0;
  chunks.forEach((buffer, index) => {
    const frames = index === 0 ? FIRST_FRAMES : NEXT_FRAMES;
    const values = new Float32Array(buffer);
    if (values.length !== SAMPLE_CHANNELS * frames * plane) {
      throw new WanVaeChunkError(`chunk ${index} のフレームの要素数 ${values.length} が合わない`);
    }
    for (let channel = 0; channel < SAMPLE_CHANNELS; channel += 1) {
      const from = channel * frames * plane;
      out.set(values.subarray(from, from + frames * plane), (channel * total + offset) * plane);
    }
    offset += frames;
  });
  return out;
};

/**
 * タイル 1 枚の chunk 列を回す（1 タイル = 1 batch・フレームだけを読み戻す — モジュールの doc）。
 *
 * `latents` は逆正規化済みの `[C,F,t,t]`。戻りは**クランプ前**の `[3, 1 + 4(F−1), 8t, 8t]`。
 * タイルの頭で cache をゼロに書く（{@link WanVaeChunkCaches.zero}）。
 *
 * MUST: 同じ device の別の batch・run を並行に発行しない（batch は device の区間ロックを持つ）。
 */
export const decodeWanVaeTile = async (
  gpu: GpuContext,
  sessions: { readonly first: Session; readonly next: Session },
  caches: WanVaeChunkCaches,
  latents: Float32Array,
): Promise<Float32Array<ArrayBuffer>> => {
  const { layout } = caches;
  const chunks = wanVaeChunkCount(layout, latents);
  const frames = await caches.frames(chunks);
  caches.zero();
  const batch = await gpu.beginBatch();
  try {
    for (let index = 0; index < chunks; index += 1) {
      const head = index === 0;
      await enqueueWanVaeChunk(
        batch,
        head ? sessions.first : sessions.next,
        head ? layout.first : layout.next,
        caches,
        wanVaeLatentChunk(layout, latents, index),
        frames[index],
      );
    }
  } catch (cause) {
    // MUST: 区間を閉じてロックを返す（閉じないと同じ device の次の batch / run が永久に待つ）。
    // 失敗の本体は呼び手へ投げ直す（finish 側の決着は同じ失敗の写しなので捨てる）。
    await batch.finish().catch(() => undefined);
    throw cause;
  }
  const names = frames.map((_, index) => `frame${index}`);
  const read = await batch.finishAndRead(
    Object.fromEntries(frames.map((resident, index) => [names[index], resident])),
  );
  return concatWanVaeFrames(layout, names.map((name) => read[name]));
};
