/**
 * 導出相 — 実行ステップ列をレシピ列へ落とす（{@link RecipeBuilder}）。
 *
 * 構造: 「計画（純関数・plan.ts）→ **導出**（ここ）→ **実行**（レシピ型と汎用ループ —
 * src/runtime/recipe.ts）」。ステップ 1 つごとに pipeline / layout を引き、params を書き、
 * bind 面と workgroup 数を決めて {@link StepRecipe} を組む。op 別の踏み分け（カーネル変種の
 * 選択・融合ルールの replay・i8a8 / f16 の opt-in 経路）はこの層に閉じる。
 *
 * ## 置き場（導出相の入口）
 *
 * **このファイルが導出相の入口**で、演算族ごとの導出は `recipe-builders/` 配下の 6 ファイル
 * （elementwise / layout / linear / norm / attention / conv）に分かれている。ここが持つのは
 * 名前解決と全件 settle（{@link RecipeBuilder.buildRecipes}）・融合ステップの replay・契約
 * kind の dispatch・params キャッシュと実績・重みスロットの解決だけで、族別関数へは
 * {@link RecipeBuildFace} という狭い面を 1 つ渡す。
 * MUST: 消費側は**この綴り（`recipe-builder.ts`）で import する**。族別の 6 ファイルへの分割は
 * 実装の都合にすぎず、{@link RecipeBuilder} が導出相の唯一の入口であることは変わらない。
 * MUST: 実体（値）の依存は **recipe-builder → recipe-builders の一方向**に保つ。族別側から
 * ここへの参照は `import type` だけ（消去されるので実行時の import グラフに逆辺が出ない）で、
 * 両者が要る**値**（params の usage）は `recipe-builders/params-usage.ts` に置く。
 *
 * MUST: 導出相は **run 寿命の状態に触れない**（{@link RunArena} の確保も dispatch の発行も
 * しない）。触れるのは Session 以上の寿命を持つ実体（重み・per-channel scale・params キャッシュ
 * = Session 常駐、パイプライン = device 常駐の {@link SessionPipelines}）だけで、これが
 * 「導出相の成果物を解決済み bindings をキーに Session へ常駐させてよい」根拠そのもの
 * （executor.ts の PreparedPlan）。
 * MUST: 依存は executor.ts → ここの**一方向**（型 import を含めて逆辺を作らない）。Session の
 * 状態は {@link RecipeBuilderContext} という構造的な面だけで受け取る。
 */

import { gridStrideWorkgroups } from "../codegen/dispatch.ts";
import type { WeightStorage } from "../kernels/weight-storage.ts";
import type { ScoreStorage } from "../kernels/score-storage.ts";
import type { IrGraph } from "../format/ir.ts";
import type { RunArena } from "../gpu/arena.ts";
import type { GpuContext } from "../gpu/device.ts";
import type { SessionPipelines } from "../gpu/pipeline-cache.ts";
import { CAST_OP, numel, stateReadonly, WEIGHT_SLOTS, WHERE_OP } from "../ops.ts";
import type { ExecStep, FusedOperand, FusedStep, PackedActivations } from "./fusion.ts";
import { ExecutionError, type NodePlan } from "./plan.ts";
import {
  type BindingRecipe,
  type BindingSource,
  type GenerationLimits,
  type StepOutput,
  type StepRecipe,
  StepRecipeBuilder,
  type TempSource,
  validateStepRecipe,
  type ValueSource,
} from "./recipe.ts";
import type {
  ComputePrecision,
  I8a8Dot,
  LinearGemvReduce,
  ParamsCacheStats,
  RmsNormReduce,
  StateAttentionReduce,
} from "./session-types.ts";
import type { ResidentWeight } from "./weight-residency.ts";
import {
  buildArgmax,
  buildCumsum,
  buildElementwise,
  buildMaskedFill,
  buildRowReduce,
  buildSoftmax,
  buildTopk,
} from "./recipe-builders/elementwise.ts";
import {
  buildCat,
  buildFlip,
  buildGather,
  buildPad,
  buildStridedCopy,
  buildUpsampleBilinear2d,
} from "./recipe-builders/layout.ts";
import { buildBmm, buildEmbedding, buildLinear, buildMatmul } from "./recipe-builders/linear.ts";
import { buildLayerNorm, buildRmsNorm, buildStaticQuantize } from "./recipe-builders/norm.ts";
import {
  buildAttention,
  buildReadonlyStateAttention,
  buildStateAppend,
  buildStateAttention,
} from "./recipe-builders/attention.ts";
import {
  buildConv1d,
  buildConv2d,
  buildConvTranspose1d,
  buildDeformConv2d,
  buildGruScan,
} from "./recipe-builders/conv.ts";
import { PARAMS_STORAGE_USAGE, PARAMS_UNIFORM_USAGE } from "./recipe-builders/params-usage.ts";

/**
 * 導出相が読む Session の状態（**必要な欄だけ**の構造的な面）。
 *
 * MUST: executor.ts の `SessionState` を import しない — Session → RecipeBuilder の一方向
 * import を型でも崩さないため。Session は自分の状態をそのまま渡す（構造的に適合する）。
 * MUST: run 寿命の器（{@link RunArena} の run アリーナ・env・スケジューラ）は載せない。
 * 載せた時点で「導出相は run 寿命の状態に触れない」がモジュール doc の宣言だけになり、
 * 導出済み計画を Session へ常駐させる根拠が型の上から消える。
 */
export type RecipeBuilderContext = {
  readonly gpu: GpuContext;
  readonly graph: IrGraph;
  /**
   * パイプラインの引き先（device 寿命のキャッシュ + この Session の使用記録 —
   * {@link SessionPipelines}）。実体は device 側にあるので、run 寿命の器を載せない規律
   * （下の MUST）とは無関係に Session を跨いで生き残る。
   */
  readonly cache: SessionPipelines;
  /** params の確保先（**Session 常駐**の weights アリーナ — `#writeParams` の MUST）。 */
  readonly weights: RunArena;
  readonly weightBuffers: ReadonlyMap<string, GPUBuffer>;
  readonly paramsCache: Map<string, GPUBuffer>;
  /**
   * 圧縮のまま常駐した重み（席・scale・group 長 — カーネル変種の選択と追加束縛の導出元。
   * 表に無い値は f32 として読む）。
   */
  readonly residentWeights: ReadonlyMap<string, ResidentWeight>;
  readonly linearCompute: "f32" | "a8" | "f16";
  readonly attentionCompute: ComputePrecision;
  readonly attentionScoreStorage: ScoreStorage;
  /** states 形 attention ③PV の縮約形（executor の {@link SessionState} が既定を決める）。 */
  readonly stateAttentionReduce: StateAttentionReduce;
  readonly linearGemvReduce: LinearGemvReduce;
  readonly rmsNormReduce: RmsNormReduce;
  /**
   * 行ブロック gemv の並列度目標（`SessionOptions.linearGemvRowsThreadTarget` — Session 生成時に
   * 固定される静的なノブ。`undefined` はカーネル側の既定）。
   */
  readonly linearGemvRowsThreadTarget: number | undefined;
  /**
   * i8a8 の整数内積変種（**族ごとに別席** — 「両変種はビット同一」が attention だけ実機で
   * 反証されているため。executor の {@link SessionState} が既定を決める）。
   */
  readonly linearI8a8Dot: I8a8Dot;
  readonly attentionI8a8Dot: I8a8Dot;
  /**
   * 行ブロック枚数の強制（**テスト専用** — executor の `ROW_BLOCK_SPLIT`）。分解経路は
   * 融合ルールが受け取るが、states 形の行ブロック（ADR 0067 決定 7）は導出相が直接割るので
   * ここにも同じノブが要る（強制分割 parity の足を経路ごとに別のノブにしない）。
   */
  readonly rowBlockSplit: number | undefined;
  readonly useCounts: ReadonlyMap<string, number>;
  readonly outputNames: ReadonlySet<string>;
};

/**
 * state ノードのビルドに要る文脈（束縛解決済みのスロット容量と、run 前検査の集積先）。
 *
 * MUST: 集積は導出相の 1 度きり（{@link GenerationLimits} は導出済み計画と同じ寿命で持ち、
 * ヒット run はここを走らせない）。run ごとに数え直す形にすると、ヒット run だけ検査が
 * 消えるか、2 実装に分かれる。
 */
export type StateBuildContext = {
  /** スロット名 → 束縛解決済みの容量込み具体形（`GenerationContext` が渡す）。 */
  readonly shapes: ReadonlyMap<string, readonly number[]>;
  readonly chunkRows: Set<number>;
  readonly fullCapacities: Map<string, number>;
};

/**
 * 族別導出（`recipe-builders/` 配下）が触れてよい共有サービスの面。
 *
 * MUST: 族別モジュールへ渡すのは**この面だけ**で、{@link RecipeBuilder} そのものは渡さない。
 * params キャッシュの所有者と params 実績のカウンタを 1 つに保つのがこの面の役目で、
 * `writeParams` がその両方を閉じ込める（面を広げて `paramsCache` を直接見せた時点で、
 * 実績の数え上げが族ごとに散る）。
 */
export type RecipeBuildFace = {
  /**
   * 導出相の状態のうち、族別導出が**実際に読む欄だけ**。
   *
   * MUST: 可変の `paramsCache` / `weights`（{@link RecipeBuilderContext}）をここに載せない —
   * `readonly` はプロパティ参照に掛かるだけで Map もアリーナも中身は書ける。全体をそのまま
   * 渡すと、上の「params キャッシュの所有者を 1 つに保つ」が doc の宣言だけになる。
   * 欄を増やすのは読む側が実際に現れたときだけ。
   */
  readonly state: Pick<
    RecipeBuilderContext,
    | "gpu"
    | "cache"
    | "linearCompute"
    | "attentionCompute"
    | "attentionScoreStorage"
    | "stateAttentionReduce"
    | "linearGemvReduce"
    | "rmsNormReduce"
    | "linearGemvRowsThreadTarget"
    | "linearI8a8Dot"
    | "attentionI8a8Dot"
    | "rowBlockSplit"
  >;
  readonly writeParams: (params: Uint32Array<ArrayBuffer>, usage: number) => GPUBuffer;
  readonly weightStorage: (step: NodePlan) => WeightStorage;
  readonly weightScaleBindings: (step: NodePlan, binding: number) => readonly BindingRecipe[];
  readonly weightGroupSize: (step: NodePlan) => number;
};

/**
 * 導出相の本体。Session は 1 個だけ持ち、run / enqueue のミス経路から
 * {@link RecipeBuilder.buildRecipes} を呼ぶ。
 */
export class RecipeBuilder {
  readonly #state: RecipeBuilderContext;
  /** 族別導出へ渡す面（private helper を閉じ込めた 1 個。構築はコンストラクタの 1 度きり）。 */
  readonly #face: RecipeBuildFace;
  /** 進行中 run の params 実績（run の頭でリセットし、決着時に Session の診断へ移す）。 */
  #paramsAllocCount = 0;
  #paramsReuseCount = 0;

  constructor(state: RecipeBuilderContext) {
    this.#state = state;
    this.#face = {
      state,
      writeParams: (params, usage) => this.#writeParams(params, usage),
      weightStorage: (step) => this.#weightStorage(step),
      weightScaleBindings: (step, binding) => this.#weightScaleBindings(step, binding),
      weightGroupSize: (step) => this.#weightGroupSize(step),
    };
  }

  /** params 実績を run の頭でリセットする（Session が run / enqueue の入口で 1 度だけ呼ぶ）。 */
  resetParamsStats(): void {
    this.#paramsAllocCount = 0;
    this.#paramsReuseCount = 0;
  }

  /** リセット以降に積んだ params 実績（Session の `lastRunParams` はこの値そのもの）。 */
  get paramsStats(): ParamsCacheStats {
    return { allocCount: this.#paramsAllocCount, reuseCount: this.#paramsReuseCount };
  }

  /**
   * 導出相 — ステップ列をレシピ列へ落とす。GPU コマンドを 1 つも出さず、run 寿命の実体
   * （{@link RunArena} のバッファ）にも触れない（モジュール doc の MUST）。
   *
   * @param stateShapes 束縛解決済みの state スロット shape（スロット名 → 容量込みの具体形）。
   *   MUST: レシピは「bindings ∪ 容量」の純関数のまま — context の**識別子は渡さない**
   *   （ADR 0066 決定 5。実体は run ごとに `GenerationEncoding` が配る）。省略した導出で
   *   state ノードに当たれば fail loudly（黙って従来形として組まない）。
   */
  async buildRecipes(
    steps: readonly ExecStep[],
    stateShapes?: ReadonlyMap<string, readonly number[]>,
  ): Promise<{
    readonly recipes: readonly StepRecipe[];
    readonly generation: GenerationLimits;
  }> {
    // 実体は実行相まで決まらないので、導出相は「その値名が既に定義済みか」だけを追う
    // （束縛漏れを実行相へ持ち越さず、ここで fail loudly にする）。
    const defined = new Set(this.#state.graph.inputs.map((spec) => spec.name));
    const states: StateBuildContext = {
      shapes: stateShapes ?? new Map(),
      chunkRows: new Set(),
      fullCapacities: new Map(),
    };
    const recipes: StepRecipe[] = [];
    // 名前は元の順に解決し、各ステップのコンパイル待ちだけを重ねる。
    // DECIDED: docs/decisions/0042-prepared-execution-plan.md#非同期コンパイル2026-09-11
    const pending = steps.map((step) => this.#buildStep(step, defined, states));
    // 途中の失敗で先に抜けると、後続の params 書き込みが run の後始末を追い越す。
    // 全件を待ち、失敗も元のステップ順で返す。GPU コマンドの順序はレシピ列が保つ。
    const settled = await Promise.allSettled(pending);
    for (const result of settled) {
      if (result.status === "rejected") throw result.reason;
      recipes.push(result.value);
    }
    return {
      recipes,
      generation: { chunkRows: states.chunkRows, fullCapacities: states.fullCapacities },
    };
  }

  /**
   * 値名 → bind 面の出どころ。重みは Session 常駐なので実体まで解決し、それ以外
   * （グラフ入力・ノード出力・別名）は値名のまま残す。ノード内一時にはなりえないので
   * 戻りは {@link ValueSource}（別名元としてもそのまま使える）。
   */
  #bindingSource(name: string, defined: ReadonlySet<string>): ValueSource {
    const resident = this.#state.weightBuffers.get(name);
    if (resident !== undefined) return { kind: "resident", buffer: resident };
    if (!defined.has(name)) throw new ExecutionError(`値 '${name}' のバッファが無い`);
    return { kind: "value", name };
  }

  /**
   * 実行ステップ 1 つ（素のノード または 融合ステップ）のレシピ。
   *
   * MUST: 確保 → retain → 本体 → 入力の release（延べ）→ 定義ぶんの release、という簿記は
   * 両者で**1 本**に閉じる（再生側は src/runtime/transient-plan.ts）。融合ごとに手書きの解放簿記を
   * 置くと、計画の参照計数が融合の本数だけ別実装になり、1 本でもずれると例外なしの
   * 沈黙誤値になる（早すぎる解放なら配り直しで値が化け、多すぎれば peak が落ちない）。
   * MUST: 出力列は**出力 slot 昇順**で組む（{@link StepRecipe.outputs} の順序規約 — 実行・
   * 計画・焼き込みと共有する 1 本）。融合ステップは単一出力（fusion.ts の窓ガード）。
   */
  async #buildStep(
    step: ExecStep,
    defined: Set<string>,
    states: StateBuildContext,
  ): Promise<StepRecipe> {
    // 素のノードの出力列は `node.outs` と同順・同長（plan.ts の {@link NodePlan.outputs}）。
    const outSlots: readonly { readonly name: string; readonly shape: readonly number[] }[] =
      step.kind === "node"
        ? step.plan.outputs
        : [{ name: step.outputName, shape: step.outputShape }];
    // bind 面のオペランド順（重複無し）と解放簿記の延べ列は別物。素のノードでは
    // どちらも node.ins に一致し、融合ステップだけが 2 つを別々に宣言する。
    const bindNames = step.kind === "node" ? step.plan.node.ins : step.binds;
    const consumedNames = step.kind === "node" ? step.plan.node.ins : step.ins;
    const binds = bindNames.map((name) => this.#bindingSource(name, defined));
    // 要素順を変えない別名では**入力バッファをそのまま出力の実体にする**
    // （別名 — ADR 0011）。dispatch も確保も出さない。要素数一致は planGraph が済ませているので、
    // 別名先の実バッファは宣言 shape ぶんの大きさを必ず満たす。
    // MUST: 別名元は bind 面の先頭（temp にはなりえない）。
    // MUST: 別名化しうるop（reshape / expand / permute）は**単一出力**なので、別名の出力列は
    // 常に 1 本きり。多出力 op を別名化する形はここでは表せない（増やすときは alias の
    // 「どの入力を」まで slot ごとに宣言することになる）。
    const aliasesInput = step.kind === "node" && step.aliasesInput;
    const outputs: readonly StepOutput[] = outSlots.map(({ name, shape }): StepOutput => {
      const bookkeeping = {
        name,
        uses: this.#state.useCounts.get(name) ?? 0,
        pinned: this.#state.outputNames.has(name),
      };
      return aliasesInput
        ? { ...bookkeeping, kind: "alias", source: binds[0] }
        : { ...bookkeeping, kind: "alloc", byteLength: numel(shape) * 4 };
    });
    // 出力は「その値名」として束ねる。実行相が dispatch より前に env へ載せるので、
    // 同一ステップ内の bind もこの列で解決できる。
    const outs: readonly BindingSource[] = outputs.map((output) => ({
      kind: "value",
      name: output.name,
    }));

    // 入力参照を検証した後に定義を進め、実行順と同じ名前解決を保つ。
    for (const output of outputs) defined.add(output.name);
    const builder = new StepRecipeBuilder();
    if (step.kind === "node") {
      await this.#buildNode(
        step.plan,
        step.aliasesInput,
        binds,
        outs,
        builder,
        states,
        step.packedActivations,
      );
    } else {
      await this.#buildFused(step, binds, outs, builder);
    }

    const recipe: StepRecipe = {
      outputs,
      temps: builder.temps,
      dispatches: builder.dispatches,
      releases: consumedNames,
      // MUST: 判別は契約（`state_append` だけが書き手 — ADR 0067 決定 5）。融合ステップは
      // state を触るノードを含めない（fusion.ts の窓ガード）ので常に false。
      writesState: step.kind === "node" && step.plan.contract.kind === "stateAppend",
    };
    // MUST: 宣言の静的検査はこの 1 箇所（レシピが外へ出る唯一の口）。素のノードの各 `#build*` と
    // 融合の replay が同じ器（{@link StepRecipeBuilder}）へ宣言するので、経路ごとに検査を置くと
    // 追加した経路だけ無検査で通る形になる。
    validateStepRecipe(recipe);
    return recipe;
  }

  /**
   * 融合ステップの dispatch 列。
   *
   * bind 面の既定は「params, 入力…, 出力」（融合 4 ルール共通）で、`operands` を宣言した
   * ルールだけがステップ内一時を混ぜた並びを取る。**一時の確保・解放は
   * {@link StepRecipeBuilder} に replay させる**（寿命の導出点を 2 つに増やさない）ので、
   * 計画の簿記は素のノードと同じ 1 本（src/runtime/transient-plan.ts の再生）に閉じたままになる。
   *
   * MUST: {@link FusedStep} は**単一出力**（fusion.ts の窓ガードが契約の出力数 1 を要求する）
   * なので、`{ kind: "output" }` が指すのは常に `outs[0]`。多出力の融合を入れるなら、
   * オペランドが出力 slot を持つ形（`{ kind: "output", slot }`）から設計し直す。
   */
  async #buildFused(
    step: FusedStep,
    binds: readonly BindingSource[],
    outs: readonly BindingSource[],
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const temps: TempSource[] = [];
    const resolve = (operand: FusedOperand): BindingSource => {
      if (operand.kind === "output") return outs[0];
      if (operand.kind === "bind") {
        const source = binds[operand.index];
        if (source === undefined) {
          throw new ExecutionError(
            `融合ルール '${step.rule}': bind 添字 ${operand.index} が宣言 ${binds.length} 本の外`,
          );
        }
        return source;
      }
      if (operand.kind === "weightScale") {
        const name = step.binds[operand.index];
        const resident = this.#state.residentWeights.get(name);
        if (resident === undefined || resident.storage === "f16") {
          throw new ExecutionError(
            `融合ルール '${step.rule}': bind ${operand.index} の常駐scaleが無い`,
          );
        }
        // 既存の重みscaleを借りる。所有者・sharedWeightsのリースはSessionのまま。
        return { kind: "resident", buffer: resident.scale };
      }
      const temp = temps[operand.id];
      // 未確保の一時を束ねるのは寿命宣言の破れ（確保より前の dispatch から読んでいる）。
      if (temp === undefined) {
        throw new ExecutionError(`融合ルール '${step.rule}': 一時 ${operand.id} が未確保`);
      }
      return temp;
    };
    for (const [index, dispatch] of step.dispatches.entries()) {
      for (const [id, temp] of step.temps.entries()) {
        if (temp.allocBefore === index) temps[id] = builder.allocTemp(temp.byteLength);
      }
      const { pipeline, layout, roles } = await this.#state.cache.get(
        dispatch.key,
        dispatch.wgsl(),
      );
      const params = this.#writeParams(
        dispatch.params,
        dispatch.paramsStorage === true ? PARAMS_STORAGE_USAGE : PARAMS_UNIFORM_USAGE,
      );
      const operands = dispatch.operands ??
        [
          ...binds.map((_, at): FusedOperand => ({ kind: "bind", index: at })),
          { kind: "output" } as const,
        ];
      const { workgroups } = dispatch;
      builder.dispatch({
        key: dispatch.key,
        pipeline,
        layout,
        roles,
        params,
        bindings: operands.map((operand, slot) => ({
          binding: slot + 1,
          source: resolve(operand),
        })),
        workgroups: workgroups.kind === "tiled"
          ? workgroups.counts
          : [gridStrideWorkgroups(workgroups.items, workgroups.size, limit), 1, 1],
      });
      // MUST: 同一境界の解放は確保の逆順（計画の再生と同じ順）。
      for (let id = step.temps.length - 1; id >= 0; id -= 1) {
        if (step.temps[id].releaseAfter === index) builder.releaseTemp(temps[id]);
      }
    }
  }

  /**
   * 素のノード 1 つの本体（確保・retain・解放は計画の再生が済ませる）。
   *
   * MUST: op 別ビルダは**出力列**（`outs` — 出力 slot 昇順）を受け、単一出力のカーネルは
   * `outs[0]` だけを書く（ADR 0068 決定 1）。多出力 op はスロットを明示して自分の列を書く。
   * 出力の**確保**の順序は列の昇順で、実行・slot 導出・焼き込みと共有の 1 本
   * （{@link StepRecipe.outputs} の順序規約）。
   */
  async #buildNode(
    step: NodePlan,
    aliasesInput: boolean,
    binds: readonly BindingSource[],
    outs: readonly BindingSource[],
    builder: StepRecipeBuilder,
    states: StateBuildContext,
    /** packed int8 活性の役割（ADR 0105 — 受理は fusion.ts の 1 箇所で済んでいる）。 */
    packed: PackedActivations | undefined,
  ): Promise<void> {
    switch (step.contract.kind) {
      case "unary":
      case "binary":
        await buildElementwise(
          this.#face,
          step,
          { op: step.contract.name, dtype: step.inputDtypes[0] },
          binds,
          outs,
          builder,
        );
        break;
      case "cast":
        await buildElementwise(
          this.#face,
          step,
          { op: CAST_OP, dtype: step.inputDtypes[0], to: step.outputs[0].dtype },
          binds,
          outs,
          builder,
        );
        break;
      case "where":
        // 生成入力は**値スロット**の dtype（スロット 0 は条件で常に bool）。
        await buildElementwise(
          this.#face,
          step,
          { op: WHERE_OP, dtype: step.inputDtypes[1] },
          binds,
          outs,
          builder,
        );
        break;
      case "cumsum":
        await buildCumsum(this.#face, step, binds, outs, builder);
        break;
      case "matmul":
        await buildMatmul(this.#face, step, binds, outs, builder);
        break;
      case "bmm":
        await buildBmm(this.#face, step, binds, outs, builder);
        break;
      case "gather":
        await buildGather(this.#face, step, binds, outs, builder);
        break;
      case "rowReduce":
        await buildRowReduce(this.#face, step, step.contract.name, binds, outs, builder);
        break;
      case "argmax":
        await buildArgmax(this.#face, step, binds, outs, builder);
        break;
      case "topk":
        await buildTopk(this.#face, step, binds, outs, builder);
        break;
      case "permute":
        // 要素順が変わらない場合は既存aliasの簿記だけで完結する（ADR 0011）。
        if (!aliasesInput) {
          await buildStridedCopy(this.#face, step, "permute", binds, outs, builder);
        }
        break;
      case "slice":
      case "symPrefixSlice":
        await buildStridedCopy(this.#face, step, step.contract.kind, binds, outs, builder);
        break;
      case "expand":
        // 恒等 expand は別名化済み（0 dispatch）。複製軸が 1 つでもあれば実体化コピー。
        if (!aliasesInput) {
          await buildStridedCopy(this.#face, step, step.contract.kind, binds, outs, builder);
        }
        break;
      case "cat":
        await buildCat(this.#face, step, binds, outs, builder);
        break;
      case "pad":
        await buildPad(this.#face, step, binds, outs, builder);
        break;
      case "flip":
        await buildFlip(this.#face, step, binds, outs, builder);
        break;
      case "linear":
        await buildLinear(this.#face, step, binds, outs, builder, packed);
        break;
      case "layerNorm":
        await buildLayerNorm(this.#face, step, binds, outs, builder);
        break;
      case "staticQuantize":
        await buildStaticQuantize(this.#face, step, binds, outs, builder, packed);
        break;
      case "rmsNorm":
        await buildRmsNorm(this.#face, step, binds, outs, builder);
        break;
      case "softmax":
        await buildSoftmax(this.#face, step, false, binds, outs, builder);
        break;
      case "safeSoftmax":
        await buildSoftmax(this.#face, step, true, binds, outs, builder);
        break;
      case "attention":
        // 欄の有無が形を判別する（ADR 0067 決定 4）。states 形は別族カーネル
        // （src/kernels/state-attention.ts）で、融合 attention とは 1 バイトも共有しない。
        // states 形はさらに 2 つ: 今 step の k/v も読む従来形と、past だけを読む **readonly**
        // 形（ADR 0096 段 2 §1.2 — ins が q 1 本で、束縛が 1 本ずつ詰まった別カーネル）。
        if (Object.keys(step.node.states).length === 0) {
          await buildAttention(this.#face, step, binds, outs, builder);
        } else if (stateReadonly(step.node.attrs, `nodes (${step.node.op})`)) {
          await buildReadonlyStateAttention(this.#face, step, binds, outs, builder, states);
        } else {
          await buildStateAttention(this.#face, step, binds, outs, builder, states);
        }
        break;
      case "embedding":
        await buildEmbedding(this.#face, step, binds, outs, builder);
        break;
      case "maskedFill":
        await buildMaskedFill(this.#face, step, binds, outs, builder);
        break;
      case "conv1d":
        await buildConv1d(this.#face, step, binds, outs, builder);
        break;
      case "conv2d":
        await buildConv2d(this.#face, step, binds, outs, builder);
        break;
      case "convTranspose1d":
        await buildConvTranspose1d(this.#face, step, binds, outs, builder);
        break;
      case "deformConv2d":
        await buildDeformConv2d(this.#face, step, binds, outs, builder);
        break;
      case "upsampleBilinear2d":
        await buildUpsampleBilinear2d(this.#face, step, binds, outs, builder);
        break;
      case "gruScan":
        await buildGruScan(this.#face, step, binds, outs, builder);
        break;
      case "reshape":
        // 別名化は #buildStep で済んでいる（この op は 1 dispatch も出さない）。
        break;
      case "stateAppend":
        await buildStateAppend(this.#face, step, binds, builder, states);
        break;
      default: {
        // MUST: 未処理の kind をここで止める。switch が素通りすると出力バッファに 1 バイトも
        // 書かれないまま次のノードへ進み、配り直しの残骸がそのまま値になる
        // （full-write 不変条件の破れ — ADR 0014）。型としては到達不能で、op を足したときの
        // 結線漏れをコンパイル時に赤くするのがこの分岐の役目。
        const unhandled: never = step.contract;
        throw new ExecutionError(`未処理の op kind: ${JSON.stringify(unhandled)}`);
      }
    }
  }

  /**
   * このノードの重みスロットに常駐している重み（席・scale・group 長）。圧縮常駐していない
   * 値では `undefined`（= f32 として読む）。
   *
   * MUST: スロット位置は適格判定と**同じ表**（{@link WEIGHT_SLOTS}）から引く。片方だけ
   * ずれると、圧縮のまま上げた重みを f32 カーネルが読む（あるいはその逆）ビット列の
   * 読み替えになり、例外は 1 つも出ない。
   */
  #residentWeight(step: NodePlan): ResidentWeight | undefined {
    return this.#state.residentWeights.get(this.#weightSlotName(step));
  }

  /** このノードの重みスロットに束縛された値名（{@link WEIGHT_SLOTS} が正本）。 */
  #weightSlotName(step: NodePlan): string {
    const slot = WEIGHT_SLOTS.get(step.node.op);
    if (slot === undefined) {
      throw new ExecutionError(`op '${step.node.op}' に重みスロットの定義が無い`);
    }
    return step.node.ins[slot];
  }

  /** このノードの重みスロットの格納形（カーネル変種の選択 — ADR 0018）。 */
  #weightStorage(step: NodePlan): WeightStorage {
    return this.#residentWeight(step)?.storage ?? "f32";
  }

  /**
   * i8 / i4 変種の追加束縛（per-channel / group scale）。f32 / f16 では空配列になり、
   * bind group は従来のままになる（ADR 0019 / 0069）。
   *
   * MUST: 束縛番号はカーネル側の定数（`*_SCALE_BINDING`）から引く。WGSL の宣言と executor が
   * 別々に番号を持つと、変種を足したときに片方だけずれる。
   * NOTE: 「i8 / i4 常駐なのに scale バッファが無い」の実行時検査は要らない —
   * {@link ResidentWeight} が席と対で scale を要求しており、型が保証している。
   */
  #weightScaleBindings(step: NodePlan, binding: number): readonly BindingRecipe[] {
    const resident = this.#residentWeight(step);
    if (resident === undefined || resident.storage === "f16") return [];
    return [{ binding, source: { kind: "resident", buffer: resident.scale } }];
  }

  /**
   * i4 常駐の重みの group 長（executor が宣言から写した値 — ADR 0069）。
   *
   * 呼ぶのは席が i4 と分かっている経路だけなので、それ以外は黙って既定へ落とさず
   * fail loudly にする — 既定で埋めると「group 64 の資産が group 32 のパイプラインで走る」
   * 沈黙誤値になる。
   */
  #weightGroupSize(step: NodePlan): number {
    const resident = this.#residentWeight(step);
    if (resident?.storage !== "i4") {
      throw new ExecutionError(
        `initializer '${this.#weightSlotName(step)}': i4 常駐なのに group_size が無い`,
      );
    }
    return resident.groupSize;
  }

  /**
   * dispatch の params uniform / storage を確保して書く。**内容そのものをキーにした Session
   * 常駐キャッシュ**を通すので、同じ内容の params は Session の生涯で 1 度しか確保・転送
   * されない（params の全バイトは グラフ・node.attrs・解決済み shape の純関数で、実行時の
   * テンソル値に依存しない）。キーは usage と全要素を連結した文字列そのもの — ハッシュでは
   * ないので衝突が原理的に無い。
   *
   * MUST: 確保先は **weights アリーナ（Session 常駐）**で、RunArena からは取らない。run ごとの
   * アリーナは run 末尾で破棄されるため、そこから配るとキャッシュが破棄済みバッファを指す。
   * MUST: キャッシュしたバッファは**一度書いたら二度と書き換えない**。ADR 0004 が
   * `allocHostWritten` をプール対象外にしているのは「`queue.writeBuffer` が未 submit の先行
   * エンコードを追い越す」ためだが、ここは同じバッファへの 2 度目の writeBuffer が存在しない
   * ので追い越しハザードが原理的に生じない（配り直しとは別レイヤの、内容同一性による
   * 共有）。
   * MUST: 破棄は weights アリーナの dispose に相乗りする（`Session.dispose` →
   * `RunArena.destroy` の flush-before-destroy）。キャッシュ専用の破棄経路を新設すると
   * flush の担い手が 2 つになり、どちらが先に走るかで破棄済みバッファ参照の submit が生まれる。
   * NOTE: 失敗 run（`scheduler.discard` 経路）の後もキャッシュは有効。バッファの内容は不変で、
   * discard が捨てるのは未 submit のエンコードだけなので、params の値は影響を受けない。
   * NOTE: **このキャッシュは Session 寿命で無界**（by-design — `docs/limitations.md`）。
   * 同範囲の他キャッシュ（`SubmitScheduler.MEASURED_HISTORY` /
   * `PREPARED_PLAN_CAPACITY`）が上限を持つのに対し、ここは追い出しを持たない —— params を
   * 追い出す = 破棄することになるが、生きている導出済み計画がその実体を**直参照で畳み込んで
   * いる**ため、安全にやるには参照計数という別の簿記が要る。可変 shape を同一 Session で
   * 多数回回す用途（可変長 TTS / 系列長可変の埋め込み）では 1 run につきノード種ぶんの小
   * バッファが積み上がるので、`diagnostics().weights.allocCount` の伸びを見て Session を
   * 切り直すこと。無界であること自体は tests/gpu_params_cache_test.ts が門にしている。
   */
  #writeParams(params: Uint32Array<ArrayBuffer>, usage: number): GPUBuffer {
    const key = `u${usage}:${params.join(",")}`;
    const cached = this.#state.paramsCache.get(key);
    if (cached !== undefined) {
      this.#paramsReuseCount += 1;
      return cached;
    }
    const buffer = this.#state.weights.allocHostWritten(params.byteLength, usage);
    this.#state.gpu.device.queue.writeBuffer(buffer, 0, params);
    this.#state.paramsCache.set(key, buffer);
    this.#paramsAllocCount += 1;
    return buffer;
  }
}
