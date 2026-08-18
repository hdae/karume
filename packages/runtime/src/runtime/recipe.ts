/**
 * 実行レシピ — エンコード層の中間表現と、その汎用実行ループ。
 *
 * 構造: 「**導出**（executor がステップ列 → レシピ列へ落とす）→ **実行**（ここが順に
 * bind group を組んで dispatch を積む）」。導出相は GPU コマンドを 1 つも出さず、run 寿命の
 * 実体（{@link RunArena} のバッファ）にも触れない — Session 常駐の実体（重み・per-channel
 * scale・params）だけを直参照として畳み込む。
 *
 * MUST: レシピは `GPUBindGroup` を持たない。束ねる相手（ノード出力・一時）はアリーナ経路では
 * run ごとにプール再利用で入れ替わるため、持てるのは「どの位置に何を束ねるか」という**手順**
 * だけ。焼き込んだ bind group（{@link bakeBindGroups}）を持つのは **backing 側**で、レシピ自体は
 * 実体に依らないまま複数の backing から共有される。
 * MUST: 一時バッファの寿命は dispatch 境界で表す（{@link TempRecipe}）。まとめて確保 /
 * まとめて解放に均すと、入れ子の生存区間を持つ形（i8a8 attention）でプール再利用と
 * `peakTransientBytes` が現行と変わる。
 *
 * 実行相は 2 つある。**アリーナ経路**（{@link executeStepRecipe}）は run ごとに
 * {@link RunArena} で確保・参照計数する従来の形。**slot 経路**は {@link derivePlanSlots} が
 * 導いた slot 表を Session 常駐バッファへ割り付け、確保・retain・release・assertDrained を
 * 丸ごと省く。さらに束縛先が run を跨いで固定されるので、{@link bakeBindGroups} が全 dispatch の
 * bind group を**構築時に 1 度だけ**組み、run に残るのは {@link executeBakedPlan} の dispatch
 * だけになる（createBindGroup も env の構築も出ない）。
 *
 * MUST: **generation run（{@link GenerationEncoding} を伴う実行）はアリーナ経路だけ**。context
 * 所有の実体（state スロット・論理長 uniform）は Session 所有の bind group へ焼き込めない
 * （ADR 0066 決定 5 の焼き込み単位の分離 — 実装は波 D-4）ので、焼き込み経路の入口 2 本は
 * `assertStateFree` で fail loudly にしてある。
 */

import { type RunArena, toSizeClass } from "../gpu/arena.ts";
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

/**
 * {@link GenerationContext} が所有する実体（ADR 0066 決定 5 — 所有権の分離）。
 *
 * `state` は名前付きスロット・`lengths` は論理長 uniform（追記 4 の搬送路）で、どちらも
 * **Session 所有ではない**。レシピは名前と種別だけを持ち、実体は run ごとに
 * {@link GenerationEncoding} が配る — これが「レシピは context の識別子を知らない」
 * （計画鍵に context が載らない）の実体側の根拠になる。
 */
export type GenerationSource =
  | { readonly kind: "state"; readonly name: string }
  | { readonly kind: "lengths" };

/** bind group の 1 スロットが指す実体の出どころ。 */
export type BindingSource = ValueSource | TempSource | GenerationSource;

/** bind group の 1 エントリ。MUST: 束縛番号はカーネル側の宣言と一致させる。 */
export type BindingRecipe = {
  readonly binding: number;
  readonly source: BindingSource;
};

/**
 * dispatch の workgroup 数。**静的な 3 つ組**（既存の全ビルダ）か、**論理長から算出する
 * 純関数**（states 形 — ADR 0067 決定 4「dispatch 数はホストが論理長から算出」）のどちらか。
 *
 * MUST: 判別共用体で持つ（同じ dispatch に静的値と算出関数を並存させない）。両方持てる形に
 * すると「どちらが正本か」が dispatch ごとに割れ、片方だけ直された実装が例外なしに
 * 別の仕事量で回る。
 */
export type DispatchWorkgroups =
  | readonly [number, number, number]
  | ((past: number, query: number) => readonly [number, number, number]);

/**
 * 1 dispatch ぶんのレシピ。
 *
 * MUST: `params` は**全カーネル共通で binding 0**（31 箇所の bind group が例外なくこの形）
 * なので、{@link BindingRecipe} の列とは別枠で持つ。
 */
type DispatchRecipe = {
  /** パイプラインキー（GPU 時間内訳の帰属先 — ADR 0021）。 */
  readonly key: string;
  readonly pipeline: GPUComputePipeline;
  readonly layout: GPUBindGroupLayout;
  /** 内容アドレスキャッシュ済みの params（Session 常駐）。 */
  readonly params: GPUBuffer;
  readonly bindings: readonly BindingRecipe[];
  readonly workgroups: DispatchWorkgroups;
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
 * 恒等 expand — ADR 0011）で、確保も dispatch も出ない。簿記と組んだ
 * {@link StepOutput} が外向きの形（この判別ユニオン単体では消費者が居ない）。
 */
type OutputRecipe =
  | { readonly kind: "alloc"; readonly byteLength: number }
  | { readonly kind: "alias"; readonly source: ValueSource };

/** ステップ出力 1 本ぶんの確保仕様と簿記（{@link StepRecipe.outputs} の要素）。 */
export type StepOutput = OutputRecipe & {
  readonly name: string;
  /** 出力値の将来の消費回数（`retain` の `uses`）。 */
  readonly uses: number;
  /** グラフ出力（プールへ返さず readback 可能に保つ）。 */
  readonly pinned: boolean;
};

/** 実行ステップ 1 つ（素のノード または 融合ステップ）のレシピ。 */
export type StepRecipe = {
  /**
   * 出力 slot 昇順の確保仕様と簿記（ADR 0068 決定 1 — 素のノードでは `node.outs` と同順・同長、
   * 融合ステップは常に 1 本）。
   *
   * MUST: **確保 → retain → env 登録 → 定義ぶんの解放を、この列の昇順で**行う。順序を共有する
   * のは 4 箇所 — 実行（{@link executeStepRecipe}）・slot 導出（{@link derivePlanSlots}）・
   * 焼き込み（{@link bakeBindGroups}）・列の組み立て（recipe-builder の `#buildStep`）。1 箇所でも
   * 順序が割れると例外は出ず、プール再利用の相手（= slot の本数と総バイト数、別名判定）だけが
   * 実行と導出で食い違う。
   */
  readonly outputs: readonly StepOutput[];
  readonly temps: readonly TempRecipe[];
  readonly dispatches: readonly DispatchRecipe[];
  /** ステップ末尾に解放する入力の**延べ列**（現行簿記と同一順）。 */
  readonly releases: readonly string[];
  /**
   * このステップが **state スロットへ書く**か（`state_append` を含むか — ADR 0067 決定 5）。
   *
   * MUST: 失敗時の poison 判定（ADR 0066 追記 3）の唯一の判別点。物理 ring が上書きされうる
   * のは書き dispatch を submit した run だけなので、executor はこの真偽で「submit カウンタの
   * スナップショットを取る位置」を決める。読むだけの states 形 attention は false。
   */
  readonly writesState: boolean;
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

/**
 * generation run 1 回ぶんの context 側の実体と論理長（ADR 0066 決定 5 / 追記 4）。
 *
 * MUST: 論理長は**エンコード時の値**（`lengths` uniform に書いた値と同じ組）。dispatch 数を
 * ここから算出する（{@link DispatchWorkgroups}）ので、uniform と食い違うと「GPU が走査する
 * 範囲」と「ホストが撃った workgroup 数」がずれ、S の一部が未書き込みのまま読まれる。
 */
export type GenerationEncoding = {
  /** スロット名 → context 所有のバッファ。 */
  readonly slots: ReadonlyMap<string, GPUBuffer>;
  /** 論理長 uniform（`{past, query}` の 8 バイト）。 */
  readonly lengths: GPUBuffer;
  readonly past: number;
  readonly query: number;
};

/**
 * generation run の**論理長に対する**受け口（導出相が state ノードから集める — 実体は
 * src/runtime/recipe-builder.ts）。state ノードを持たないグラフでは両方とも空になる。
 *
 * MUST: 導出済み計画（executor の `PreparedPlan`）と同じ寿命で持つ。ヒット run は計画も導出も
 * 走らせないので、レシピ列と一緒に畳んでおかないと「2 run 目だけ検査が消える」形になる。
 */
export type GenerationLimits = {
  /**
   * state ノードの物理 chunk 行数 `M`（同値は畳んだ集合）。`queryLength` はこの**最小**まで
   * — 超えた行は宣言 shape に載らない（ADR 0066 決定 4 の固定長 chunk + pad）。
   */
  readonly chunkRows: ReadonlySet<number>;
  /**
   * full スロット（attrs `window` 宣言なし）の名前 → 行容量 `C`。sliding は ring なので
   * 対象外（ADR 0067 決定 4 ④ — 「full スロットは実行時に `pastLength + queryLength ≤ C`」）。
   */
  readonly fullCapacities: ReadonlyMap<string, number>;
};

/**
 * generation run の**エンコード前**の検査（ADR 0066 決定 4 / ADR 0067 決定 4 ④）。
 *
 * MUST: dispatch を 1 本も積む前に落とす。`queryLength > M` は q / ins の宣言 shape の外を
 * 読み書きする形で、`pastLength + queryLength > C` の full スロットは `state_append` の書きが
 * 範囲外へ落ちる（robustness で捨てられる = **静かに書かれない**）— どちらも例外も警告も
 * 出ないまま、次 step の読者が残骸を過去 KV として食う。
 * MUST: `queryLength ≤ chunkLength` と u32 値域はここで重ねて見ない（`GenerationContext` の
 * `writeLengths` / `advance` が持つ — 二重簿記の禁止）。
 */
export const assertGenerationRun = (
  limits: GenerationLimits,
  pastLength: number,
  queryLength: number,
): void => {
  for (const rows of limits.chunkRows) {
    if (queryLength > rows) {
      throw new ExecutionError(
        `queryLength ${queryLength} が state ノードの物理 chunk 行数 ${rows} を超える` +
          "（prefill は固定長 chunk + pad・decode は 1 — ADR 0066 決定 4）",
      );
    }
  }
  for (const [name, capacity] of limits.fullCapacities) {
    if (pastLength + queryLength > capacity) {
      throw new ExecutionError(
        `state '${name}': pastLength ${pastLength} + queryLength ${queryLength} が full スロットの` +
          `容量 ${capacity} を超える（ADR 0067 決定 4 ④ — sliding でないスロットは巻かない）`,
      );
    }
  }
};

/** bind group を組んで dispatch を積むのに要る文脈（アリーナ経路と slot 経路の共通部）。 */
type EncodeContext = {
  readonly device: GPUDevice;
  readonly scheduler: SubmitScheduler;
  /**
   * 値名 → 実体（グラフ入力とノード出力のみ）。Session 常駐の重み / scale は `resident` として
   * 畳み込み済みなので、ここには載らない。
   */
  readonly env: Map<string, GPUBuffer>;
  /** generation run のときだけ渡る context 側の面（1-shot 実行では undefined）。 */
  readonly generation?: GenerationEncoding;
};

/** レシピ実行に要る run 寿命の文脈（アリーナ簿記あり — {@link executeStepRecipe}）。 */
type StepExecution = EncodeContext & {
  readonly arena: RunArena;
};

const resolveValue = (source: ValueSource, env: ReadonlyMap<string, GPUBuffer>): GPUBuffer => {
  if (source.kind === "resident") return source.buffer;
  const buffer = env.get(source.name);
  if (buffer === undefined) throw new ExecutionError(`値 '${source.name}' のバッファが無い`);
  return buffer;
};

/**
 * context 所有の実体を引く。
 *
 * MUST: generation 情報が無い実行で state / 論理長を束ねようとしたら fail loudly（計画層が
 * 先に落とすので到達しない防波堤 — 通すと `undefined` を bind group へ渡して validation で
 * 落ち、真因から遠い診断になる）。
 */
const resolveGeneration = (
  source: GenerationSource,
  generation: GenerationEncoding | undefined,
): GPUBuffer => {
  if (generation === undefined) {
    throw new ExecutionError(
      `${
        source.kind === "lengths" ? "論理長 uniform" : `state スロット '${source.name}'`
      }を束ねる dispatch を GenerationContext 無しで実行しようとした`,
    );
  }
  if (source.kind === "lengths") return generation.lengths;
  const buffer = generation.slots.get(source.name);
  if (buffer === undefined) {
    throw new ExecutionError(`state スロット '${source.name}' の実体が GenerationContext に無い`);
  }
  return buffer;
};

const resolveBinding = (
  source: BindingSource,
  env: ReadonlyMap<string, GPUBuffer>,
  temps: readonly GPUBuffer[],
  generation: GenerationEncoding | undefined,
): GPUBuffer => {
  switch (source.kind) {
    case "temp":
      return temps[source.id];
    case "state":
    case "lengths":
      return resolveGeneration(source, generation);
    default:
      return resolveValue(source, env);
  }
};

/**
 * 論理長から算出する形（{@link DispatchWorkgroups}）を実数へ落とす。
 *
 * MUST: 算出は**エンコードのたび**（毎 run 論理長が変わる）。ここで畳んで持ち回ると
 * 「レシピは bindings の純関数」が崩れ、前 step の workgroup 数で回る。
 */
const resolveWorkgroups = (
  recipe: DispatchRecipe,
  generation: GenerationEncoding | undefined,
): readonly [number, number, number] => {
  if (typeof recipe.workgroups !== "function") return recipe.workgroups;
  if (generation === undefined) {
    throw new ExecutionError(
      `dispatch '${recipe.key}': workgroup 数を論理長から算出する dispatch を ` +
        "GenerationContext 無しで実行しようとした",
    );
  }
  return recipe.workgroups(generation.past, generation.query);
};

/**
 * レシピ列が context 所有の実体を 1 つも束ねないことの門（**焼き込み経路の入口**）。
 *
 * MUST: slot backing（{@link derivePlanSlots} / {@link bakeBindGroups}）は state を扱えない。
 * 焼き込み単位の分離（ADR 0066 決定 5 — 実装は波 D-4）が無いまま焼くと、bind group が
 * **前の context の KV を束ねたまま**回り続ける沈黙 stale 読みの入り口になる。
 */
const assertStateFree = (recipes: readonly StepRecipe[], where: string): void => {
  for (const recipe of recipes) {
    for (const dispatch of recipe.dispatches) {
      for (const entry of dispatch.bindings) {
        if (entry.source.kind !== "state" && entry.source.kind !== "lengths") continue;
        throw new ExecutionError(
          `${where}: dispatch '${dispatch.key}' が GenerationContext 所有の実体を束ねている` +
            "（state を含む bind group の焼き込みは波 D-4 — ADR 0066 決定 5 の分離焼き込み）",
        );
      }
      if (typeof dispatch.workgroups === "function") {
        throw new ExecutionError(
          `${where}: dispatch '${dispatch.key}' の workgroup 数が論理長から算出される` +
            "（焼き込み経路は run を跨いで固定の dispatch 数しか扱えない）",
        );
      }
    }
  }
};

/**
 * 1 dispatch ぶんの bind group を組む。
 *
 * MUST: アリーナ経路（{@link encodeDispatch}）と焼き込み（{@link bakeBindGroups}）は**この 1 本**
 * を共有する。エントリの並べ方が 2 実装に分かれると、焼き込み側だけ束縛番号や params の位置が
 * ずれても validation は通り（layout さえ満たせばよい）、値だけが静かに変わる。
 */
const createBindGroup = (
  device: GPUDevice,
  recipe: DispatchRecipe,
  resolve: (source: BindingSource) => GPUBuffer,
): GPUBindGroup =>
  device.createBindGroup({
    layout: recipe.layout,
    entries: [
      { binding: 0, resource: { buffer: recipe.params } },
      ...recipe.bindings.map((entry) => ({
        binding: entry.binding,
        resource: { buffer: resolve(entry.source) },
      })),
    ],
  });

const encodeDispatch = (
  recipe: DispatchRecipe,
  run: EncodeContext,
  temps: readonly GPUBuffer[],
): void => {
  const bindGroup = createBindGroup(
    run.device,
    recipe,
    (source) => resolveBinding(source, run.env, temps, run.generation),
  );
  run.scheduler.dispatch(
    recipe.pipeline,
    bindGroup,
    resolveWorkgroups(recipe, run.generation),
    recipe.key,
  );
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
 * MUST: 出力は**出力 slot 昇順**で確保・retain・env 登録し、定義ぶんの解放も同じ昇順で返す
 * （{@link StepRecipe.outputs} の順序規約 — 導出・焼き込みと共有する 1 本）。
 */
export const executeStepRecipe = (recipe: StepRecipe, run: StepExecution): void => {
  const { arena, env } = run;
  const outs = recipe.outputs.map((output) => {
    const buffer = output.kind === "alias"
      ? resolveValue(output.source, env)
      : arena.allocStorage(output.byteLength);
    // MUST: 別名でも retain は「定義ぶんの 1 + 出力値の消費回数」を**実バッファに積む**
    // （別名越しの消費まで数えるため — アリーナのエイリアス節）。
    arena.retain(buffer, output.uses, { pinned: output.pinned });
    env.set(output.name, buffer);
    return buffer;
  });

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
  for (const buffer of outs) arena.release(buffer);
};

/** 1 ステップぶんの slot 割当（{@link PlanSlots.steps} の要素）。 */
type StepSlots = {
  /**
   * {@link StepRecipe.outputs} と同順・同長の slot 添字。別名の出力（確保が出ない）は
   * undefined。
   */
  readonly outputs: readonly (number | undefined)[];
  /** {@link StepRecipe.temps} と同順・同長の slot 添字。 */
  readonly temps: readonly number[];
};

/**
 * レシピ列 1 本ぶんの transient slot 表（{@link derivePlanSlots}）。
 *
 * slot は「run の間ずっと GPU に存在する中間バッファ 1 本」で、RunArena が run ごとに
 * createBuffer していた実体をそのまま Session 常駐へ移した形。
 */
type PlanSlots = {
  /** slot ごとのバイト数（{@link toSizeClass} 済み — RunArena が実際に確保する大きさ）。 */
  readonly bytes: readonly number[];
  /** {@link StepRecipe} 列と同順・同長の割当。 */
  readonly steps: readonly StepSlots[];
  /**
   * グラフ出力として pin された slot（プールへ返らない = 他の値と共有されない）。
   * MUST: readback を許してよいのはこの集合だけ — 他の slot は run 中に別の値へ配り直され、
   * 次 run では前 run の残骸が居るため（中間値 readback 拒否の規律）。
   */
  readonly pinned: ReadonlySet<number>;
};

/**
 * レシピ列から transient slot 表を導く。
 *
 * MUST: {@link RunArena} のサイズクラス LIFO を**仮想的に再生**して導く（独自パッキング禁止）。
 * ここが鏡写しである限り slot の本数と総バイト数は現行 run の実確保と一致し、常駐化しても
 * VRAM のピークは新しく生まれない。詰め直して減らそうとした瞬間、①現行の footprint 前提が
 * 崩れ ②「別の値が同じ実体を掴んでよいか」の判断がアリーナと 2 実装に分かれる。
 *
 * MUST: 純関数（GPU 資源を作らず、レシピも変更しない）。
 */
export const derivePlanSlots = (recipes: readonly StepRecipe[]): PlanSlots => {
  assertStateFree(recipes, "slot 導出");
  const bytes: number[] = [];
  // サイズクラス → 空き slot（LIFO — RunArena.#pool と同じ形）。
  const pool = new Map<number, number[]>();
  const refs = new Map<number, number>();
  const pinned = new Set<number>();
  // 値名 → slot。slot を持たない値（グラフ入力・重み = プール対象外）は載せない。
  const env = new Map<string, number>();
  const steps: StepSlots[] = [];

  const alloc = (byteLength: number): number => {
    const sizeClass = toSizeClass(byteLength);
    const reused = pool.get(sizeClass)?.pop();
    if (reused !== undefined) return reused;
    bytes.push(sizeClass);
    return bytes.length - 1;
  };
  const retain = (slot: number | undefined, uses: number, isPinned: boolean): void => {
    if (slot === undefined) return;
    if (isPinned) pinned.add(slot);
    refs.set(slot, (refs.get(slot) ?? 0) + uses + 1);
  };
  const release = (slot: number | undefined): void => {
    if (slot === undefined) return;
    const left = (refs.get(slot) ?? 0) - 1;
    if (left < 0) throw new ExecutionError("slot 導出: 参照カウントが負（消費計数の誤り）");
    refs.set(slot, left);
    if (left > 0 || pinned.has(slot)) return;
    const bucket = pool.get(bytes[slot]);
    if (bucket === undefined) pool.set(bytes[slot], [slot]);
    else bucket.push(slot);
  };

  for (const recipe of recipes) {
    // MUST: 出力 slot 昇順（{@link StepRecipe.outputs} の順序規約 — 実行相と同順）。
    const outputs = recipe.outputs.map((output) => {
      // 別名は「入力の実体をそのまま出力にする」— 元が slot でなければ（グラフ入力・重み）
      // この値も slot を持たない。
      const slot = output.kind === "alias"
        ? (output.source.kind === "value" ? env.get(output.source.name) : undefined)
        : alloc(output.byteLength);
      retain(slot, output.uses, output.pinned);
      if (slot === undefined) env.delete(output.name);
      else env.set(output.name, slot);
      return slot;
    });

    const temps: number[] = [];
    recipe.dispatches.forEach((_, index) => {
      recipe.temps.forEach((temp, id) => {
        if (temp.allocBefore !== index) return;
        const slot = alloc(temp.byteLength);
        retain(slot, 0, false);
        temps[id] = slot;
      });
      // MUST: 同一境界の解放は確保の逆順（executeStepRecipe と同じ順で LIFO に戻す）。
      for (let id = recipe.temps.length - 1; id >= 0; id -= 1) {
        if (recipe.temps[id].releaseAfter === index) release(temps[id]);
      }
    });

    for (const name of recipe.releases) release(env.get(name));
    for (const slot of outputs) release(slot);
    steps.push({
      outputs: recipe.outputs.map((output, slot) =>
        output.kind === "alias" ? undefined : outputs[slot]
      ),
      temps,
    });
  }
  return { bytes, steps, pinned };
};

/** slot 添字 → 常駐バッファ。引けないのは slot 表とレシピ列の不整合（内部の不変条件破れ）。 */
const resolveSlot = (slot: number | undefined, buffers: readonly GPUBuffer[]): GPUBuffer => {
  const buffer = slot === undefined ? undefined : buffers[slot];
  if (buffer === undefined) throw new ExecutionError(`slot ${slot} のバッファが無い`);
  return buffer;
};

/** 焼き込み済みの slot backing 実行資材（{@link bakeBindGroups}）。 */
type BakedPlan = {
  /**
   * 全ステップ × 全 dispatch の bind group。外側は {@link StepRecipe} 列と、内側は
   * {@link StepRecipe.dispatches} と同順・同長。
   */
  readonly groups: readonly (readonly GPUBindGroup[])[];
  /**
   * 全ステップを展開し終えた時点の値名 → 実体（アリーナ経路の run 末尾の `env` と同じもの）。
   * グラフ出力の読み戻し先を構築時に確定するのに使う。
   */
  readonly values: ReadonlyMap<string, GPUBuffer>;
};

/**
 * slot backing の bind group を焼き込む（構築時に 1 度だけ）。
 *
 * 焼き込めるのは束縛先が run を跨いで固定だから: params / 重み / per-channel scale は
 * `resident` の直参照、ノード出力・一時は {@link PlanSlots} の常駐 slot、グラフ入力は backing が
 * 所有する常駐バッファ（`inputs`）。run ごとに変わるのは**入力バッファの中身だけ**で、
 * bind group が指す実体は 1 つも動かない。
 *
 * MUST: 値名 → 実体の写像はアリーナ経路（{@link executeStepRecipe}）と**同じ順で**展開する
 * — 出力を先に env へ載せてから当該ステップの dispatch を解決する順序が崩れると、出力を
 * 自分の入力にも束ねるステップだけが別の実体を掴む。出力が複数ある形では**出力 slot 昇順**まで
 * 揃える（{@link StepRecipe.outputs} の順序規約）。
 * MUST: 呼ぶのは run の errorScope 区間の内側だけ（createBindGroup の validation 失敗は
 * 例外にならない）。
 */
export const bakeBindGroups = (
  recipes: readonly StepRecipe[],
  slots: PlanSlots,
  context: {
    readonly device: GPUDevice;
    /** slot 添字 → 常駐バッファ（{@link PlanSlots.bytes} と同順・同長）。 */
    readonly buffers: readonly GPUBuffer[];
    /** グラフ入力名 → backing 所有の常駐バッファ。 */
    readonly inputs: ReadonlyMap<string, GPUBuffer>;
  },
): BakedPlan => {
  assertStateFree(recipes, "bind group の焼き込み");
  const { device, buffers, inputs } = context;
  const values = new Map<string, GPUBuffer>(inputs);
  const groups: (readonly GPUBindGroup[])[] = [];
  recipes.forEach((recipe, index) => {
    const step = slots.steps[index];
    recipe.outputs.forEach((output, slot) => {
      values.set(
        output.name,
        output.kind === "alias"
          ? resolveValue(output.source, values)
          : resolveSlot(step.outputs[slot], buffers),
      );
    });
    const temps = step.temps.map((slot) => resolveSlot(slot, buffers));
    groups.push(
      recipe.dispatches.map((dispatch) =>
        createBindGroup(
          device,
          dispatch,
          (source) => resolveBinding(source, values, temps, undefined),
        )
      ),
    );
  });
  return { groups, values };
};

/**
 * 焼き込み済み backing の実行（slot 経路）。run が出す GPU 操作はこの dispatch だけで、
 * 確保・retain・release も createBindGroup も出ない。
 *
 * MUST: 積むコマンド列は {@link executeStepRecipe} と**同一**（bind 先の実体が run を跨いで
 * 同じになるだけ）。前 run の残骸が slot に残っていても正しいのは full-write（ADR 0014 —
 * 全ノードが出力の全バイトを書く）が根拠で、プール再利用の安全性と同じ 1 本の不変条件。
 */
export const executeBakedPlan = (
  recipes: readonly StepRecipe[],
  groups: readonly (readonly GPUBindGroup[])[],
  scheduler: SubmitScheduler,
): void => {
  recipes.forEach((recipe, index) => {
    const stepGroups = groups[index];
    recipe.dispatches.forEach((dispatch, id) => {
      // 焼き込み経路に来るレシピは `assertStateFree` を通っている（= 静的な 3 つ組だけ）。
      scheduler.dispatch(
        dispatch.pipeline,
        stepGroups[id],
        resolveWorkgroups(dispatch, undefined),
        dispatch.key,
      );
    });
  });
};
