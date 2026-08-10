/**
 * 実行レシピ — エンコード層の中間表現と、その汎用実行ループ。
 *
 * 構造: 「**導出**（executor がステップ列 → レシピ列へ落とす）→ **実行**（ここが順に
 * bind group を組んで dispatch を積む）」。導出相は GPU コマンドを 1 つも出さず、run 寿命の
 * 実体（{@link RunArena} のバッファ）にも触れない — Session 常駐の実体（重み・per-channel
 * scale・params）だけを直参照として畳み込む。
 *
 * MUST: レシピは `GPUBindGroup` を持たない。束ねる相手（ノード出力・一時）は run ごとに
 * プール再利用で入れ替わるため、持てるのは「どの位置に何を束ねるか」という**手順**だけ。
 * MUST: 一時バッファの寿命は dispatch 境界で表す（{@link TempRecipe}）。まとめて確保 /
 * まとめて解放に均すと、入れ子の生存区間を持つ形（i8a8 attention）でプール再利用と
 * `peakTransientBytes` が現行と変わる。
 */

import type { RunArena } from "../gpu/arena.ts";
import type { SubmitScheduler } from "../gpu/submit.ts";
import { ExecutionError } from "./plan.ts";

/**
 * run 寿命に依らない出どころ（Session 常駐の直参照 または 値名）。別名化の元にもなる。
 * `resident` は重み / per-channel scale / params — 導出相で実体まで解決してよい。
 */
export type ValueSource =
  | { readonly kind: "resident"; readonly buffer: GPUBuffer }
  | { readonly kind: "value"; readonly name: string };

/** 同一ステップ内で閉じた一時領域への参照（`id` は {@link StepRecipe.temps} の添字）。 */
export type TempSource = { readonly kind: "temp"; readonly id: number };

/** bind group の 1 スロットが指す実体の出どころ。 */
export type BindingSource = ValueSource | TempSource;

/** bind group の 1 エントリ。MUST: 束縛番号はカーネル側の宣言と一致させる。 */
export type BindingRecipe = {
  readonly binding: number;
  readonly source: BindingSource;
};

/**
 * 1 dispatch ぶんのレシピ。
 *
 * MUST: `params` は**全カーネル共通で binding 0**（31 箇所の bind group が例外なくこの形）
 * なので、{@link BindingRecipe} の列とは別枠で持つ。
 */
export type DispatchRecipe = {
  /** パイプラインキー（GPU 時間内訳の帰属先 — ADR 0021）。 */
  readonly key: string;
  readonly pipeline: GPUComputePipeline;
  readonly layout: GPUBindGroupLayout;
  /** 内容アドレスキャッシュ済みの params（Session 常駐）。 */
  readonly params: GPUBuffer;
  readonly bindings: readonly BindingRecipe[];
  readonly workgroups: readonly [number, number, number];
};

/**
 * ノード内一時の確保仕様。**寿命は dispatch 境界の添字で表す**（モジュール doc の MUST）。
 *
 * 実行相の展開は `allocStorage` → `retain(…, 0)` で、解放は同じ境界に複数あれば**確保の逆順**
 * （現行の各 `#build*` が LIFO で返しているのと同じ順）。
 */
export type TempRecipe = {
  readonly byteLength: number;
  /** {@link StepRecipe.dispatches} のこの添字の**直前**に確保する。 */
  readonly allocBefore: number;
  /** {@link StepRecipe.dispatches} のこの添字の**直後**に解放する。 */
  readonly releaseAfter: number;
};

/**
 * ステップ出力の確保仕様。`alias` は入力実体をそのまま出力の実体にする形（reshape と
 * 恒等 expand — ADR 0011）で、確保も dispatch も出ない。
 */
export type OutputRecipe =
  | { readonly kind: "alloc"; readonly byteLength: number }
  | { readonly kind: "alias"; readonly source: ValueSource };

/** 実行ステップ 1 つ（素のノード または 融合ステップ）のレシピ。 */
export type StepRecipe = {
  readonly outputName: string;
  readonly output: OutputRecipe;
  /** 出力値の将来の消費回数（`retain` の `uses`）。 */
  readonly uses: number;
  /** グラフ出力（プールへ返さず readback 可能に保つ）。 */
  readonly pinned: boolean;
  readonly temps: readonly TempRecipe[];
  readonly dispatches: readonly DispatchRecipe[];
  /** ステップ末尾に解放する入力の**延べ列**（現行簿記と同一順）。 */
  readonly releases: readonly string[];
};

/** 構築中の可変な一時仕様（{@link TempRecipe} の可変版）。 */
type MutableTemp = {
  readonly byteLength: number;
  readonly allocBefore: number;
  releaseAfter: number;
};

/**
 * 1 ステップぶんのレシピを組み立てる器。**dispatch を積んだ位置**から一時の寿命を導くので、
 * 各 `#build*` は現行の `arena.allocStorage` / `arena.release` と同じ位置で
 * {@link StepRecipeBuilder.allocTemp} / {@link StepRecipeBuilder.releaseTemp} を呼べばよい。
 */
export class StepRecipeBuilder {
  readonly #temps: MutableTemp[] = [];
  readonly #dispatches: DispatchRecipe[] = [];

  /** ノード内一時を 1 本宣言する（実行相で `allocStorage` + `retain(…, 0)` に展開される）。 */
  allocTemp(byteLength: number): TempSource {
    const id = this.#temps.length;
    // 解放位置は releaseTemp が確定させる。宣言だけで解放が来なければ実行相の参照計数が
    // 閉じず、run 末尾の assertDrained が落とす（沈黙で漏れない）。
    this.#temps.push({ byteLength, allocBefore: this.#dispatches.length, releaseAfter: -1 });
    return { kind: "temp", id };
  }

  /** 一時の解放位置を「ここまでに積んだ最後の dispatch の直後」に確定する。 */
  releaseTemp(temp: TempSource): void {
    this.#temps[temp.id].releaseAfter = this.#dispatches.length - 1;
  }

  dispatch(recipe: DispatchRecipe): void {
    this.#dispatches.push(recipe);
  }

  get temps(): readonly TempRecipe[] {
    return this.#temps;
  }

  get dispatches(): readonly DispatchRecipe[] {
    return this.#dispatches;
  }
}

/** レシピ実行に要る run 寿命の文脈。 */
export type StepExecution = {
  readonly device: GPUDevice;
  readonly scheduler: SubmitScheduler;
  readonly arena: RunArena;
  /**
   * 値名 → run 寿命の実体（グラフ入力とノード出力のみ）。Session 常駐の重み / scale は
   * `resident` として畳み込み済みなので、ここには載らない。
   */
  readonly env: Map<string, GPUBuffer>;
};

const resolveValue = (source: ValueSource, env: ReadonlyMap<string, GPUBuffer>): GPUBuffer => {
  if (source.kind === "resident") return source.buffer;
  const buffer = env.get(source.name);
  if (buffer === undefined) throw new ExecutionError(`値 '${source.name}' のバッファが無い`);
  return buffer;
};

const resolveBinding = (
  source: BindingSource,
  env: ReadonlyMap<string, GPUBuffer>,
  temps: readonly GPUBuffer[],
): GPUBuffer => (source.kind === "temp" ? temps[source.id] : resolveValue(source, env));

const encodeDispatch = (
  recipe: DispatchRecipe,
  run: StepExecution,
  temps: readonly GPUBuffer[],
): void => {
  const bindGroup = run.device.createBindGroup({
    layout: recipe.layout,
    entries: [
      { binding: 0, resource: { buffer: recipe.params } },
      ...recipe.bindings.map((entry) => ({
        binding: entry.binding,
        resource: { buffer: resolveBinding(entry.source, run.env, temps) },
      })),
    ],
  });
  run.scheduler.dispatch(recipe.pipeline, bindGroup, recipe.workgroups, recipe.key);
};

/**
 * レシピ 1 ステップぶんの実行（確保 → retain → 全 dispatch のエンコード → 入力の解放 →
 * 定義ぶんの解放）。
 *
 * MUST: 簿記は素のノードと融合ステップで**この 1 本**に閉じる（現行 `#encodeStep` の不変条件を
 * そのまま引き継ぐ）。ステップごとに手書きの解放簿記を置くと、アリーナの参照計数が別実装に
 * 分かれ、1 本でもずれると例外なしの沈黙誤値になる。
 * MUST: 出力の確保は当該ステップのどの dispatch のエンコードよりも前（ADR 0004 の不変条件）。
 * この順序が「まだ読まれる入力が出力として配り直される」事故を構造的に防いでいる。
 */
export const executeStepRecipe = (recipe: StepRecipe, run: StepExecution): void => {
  const { arena, env } = run;
  const out = recipe.output.kind === "alias"
    ? resolveValue(recipe.output.source, env)
    : arena.allocStorage(recipe.output.byteLength);
  // MUST: 別名でも retain は「定義ぶんの 1 + 出力値の消費回数」を**実バッファに積む**
  // （別名越しの消費まで数えるため — アリーナのエイリアス節）。
  arena.retain(out, recipe.uses, { pinned: recipe.pinned });
  env.set(recipe.outputName, out);

  const temps: GPUBuffer[] = [];
  recipe.dispatches.forEach((dispatch, index) => {
    recipe.temps.forEach((temp, id) => {
      if (temp.allocBefore !== index) return;
      const buffer = arena.allocStorage(temp.byteLength);
      arena.retain(buffer, 0);
      temps[id] = buffer;
    });
    encodeDispatch(dispatch, run, temps);
    // MUST: 同一境界の解放は確保の逆順（プールへ戻る順が現行と一致する）。
    for (let id = recipe.temps.length - 1; id >= 0; id -= 1) {
      if (recipe.temps[id].releaseAfter === index) arena.release(temps[id]);
    }
  });

  // MUST: 解放はステップ境界（当該ステップの全 dispatch をエンコードし終えた後）のみ。
  for (const name of recipe.releases) {
    const buffer = env.get(name);
    if (buffer !== undefined) arena.release(buffer);
  }
  // MUST: retain が積んだ定義ぶんの 1 をここで返す。消費者ゼロの中間出力（グラフ出力にも
  // ならない到達不能な値）が解放されるのはこの 1 本だけで、抜けるとプール再利用から外れて
  // peakTransientBytes が実際より大きく出る。
  arena.release(out);
};
