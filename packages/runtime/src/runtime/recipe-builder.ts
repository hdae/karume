/**
 * 導出相 — 実行ステップ列をレシピ列へ落とす（{@link RecipeBuilder}）。
 *
 * 構造: 「計画（純関数・plan.ts）→ **導出**（ここ）→ **実行**（レシピ型と汎用ループ —
 * src/runtime/recipe.ts）」。ステップ 1 つごとに pipeline / layout を引き、params を書き、
 * bind 面と workgroup 数を決めて {@link StepRecipe} を組む。op 別の踏み分け（カーネル変種の
 * 選択・融合ルールの replay・i8a8 / f16 の opt-in 経路）はこのモジュールに閉じる。
 *
 * MUST: 導出相は **run 寿命の状態に触れない**（{@link RunArena} の確保も dispatch の発行も
 * しない）。触れるのは Session 常駐の実体（重み・per-channel scale・params キャッシュ・
 * {@link PipelineCache}）だけで、これが「導出相の成果物を解決済み bindings をキーに Session へ
 * 常駐させてよい」根拠そのもの（executor.ts の PreparedPlan）。
 * MUST: 依存は executor.ts → ここの**一方向**（型 import を含めて逆辺を作らない）。Session の
 * 状態は {@link RecipeBuilderContext} という構造的な面だけで受け取る。
 */

import { gridStrideWorkgroups, tiledWorkgroups } from "../codegen/dispatch.ts";
import {
  ELEMENTWISE_WORKGROUP_SIZE,
  elementwiseKey,
  elementwiseParams,
  type ElementwiseSpec,
  elementwiseWgsl,
} from "../codegen/elementwise.ts";
import {
  AXIS_REDUCE_WORKGROUP_SIZE,
  axisReduceKey,
  axisReduceParams,
  axisReduceWgsl,
  reduceKey,
  reduceParams,
  reduceWgsl,
} from "../codegen/reduce.ts";
import {
  catOutOffset,
  catOutStrides,
  expandSrcStrides,
  permuteSrcStrides,
  prefixSliceSrcStrides,
  sliceSrcOffset,
  sliceSrcStrides,
  STRIDED_WORKGROUP_SIZE,
  STRIDED_WRITE_WORKGROUP_SIZE,
  stridedKey,
  stridedParams,
  stridedWgsl,
  stridedWriteKey,
  stridedWriteParams,
  stridedWriteWgsl,
} from "../codegen/strided.ts";
import {
  CONV1D_SCALE_BINDING,
  CONV1D_WORKGROUP_SIZE,
  type Conv1dDims,
  conv1dIgemmKey,
  conv1dIgemmParams,
  conv1dIgemmWgsl,
  conv1dKey,
  conv1dParams,
  conv1dUsesVec4,
  conv1dWgsl,
} from "../kernels/conv1d.ts";
import {
  CONV2D_SCALE_BINDING,
  CONV2D_WORKGROUP_SIZE,
  type Conv2dDims,
  conv2dIgemmKey,
  conv2dIgemmMTile,
  conv2dIgemmParams,
  conv2dIgemmWgsl,
  conv2dKey,
  conv2dParams,
  conv2dUsesVec4,
  conv2dWgsl,
} from "../kernels/conv2d.ts";
import {
  CONV_TRANSPOSE1D_SCALE_BINDING,
  CONV_TRANSPOSE1D_WORKGROUP_SIZE,
  convTranspose1dKey,
  convTranspose1dParams,
  convTranspose1dWgsl,
} from "../kernels/conv-transpose1d.ts";
import { CUMSUM_KEY, CUMSUM_WGSL, CUMSUM_WORKGROUP_SIZE, cumsumParams } from "../kernels/cumsum.ts";
import {
  DEFORM_CONV2D_KEY,
  DEFORM_CONV2D_WGSL,
  DEFORM_CONV2D_WORKGROUP_SIZE,
  deformConv2dParams,
} from "../kernels/deform-conv2d.ts";
import { FLIP_KEY, FLIP_WGSL, FLIP_WORKGROUP_SIZE, flipParams } from "../kernels/flip.ts";
import { gruScanKey, gruScanParams, gruScanWgsl } from "../kernels/gru-scan.ts";
import { PAD_KEY, PAD_WGSL, PAD_WORKGROUP_SIZE, padParams } from "../kernels/pad.ts";
import {
  UPSAMPLE_BILINEAR2D_KEY,
  UPSAMPLE_BILINEAR2D_WGSL,
  UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE,
  upsampleBilinear2dParams,
} from "../kernels/upsample-bilinear2d.ts";
import {
  EMBEDDING_SCALE_BINDING,
  EMBEDDING_WORKGROUP_SIZE,
  embeddingKey,
  embeddingParams,
  embeddingWgsl,
} from "../kernels/embedding.ts";
import { LAYER_NORM_KEY, LAYER_NORM_WGSL, layerNormParams } from "../kernels/layer-norm.ts";
import { RMS_NORM_KEY, RMS_NORM_WGSL, rmsNormParams } from "../kernels/rms-norm.ts";
import { LINEAR_SCALE_BINDING, linearKey, linearParams, linearWgsl } from "../kernels/linear.ts";
import {
  LINEAR_ACT_SCALE_BINDING,
  LINEAR_I8A8_MAX_K,
  linearI8a8Key,
  linearI8a8Params,
  linearI8a8UsesVec4,
  linearI8a8Wgsl,
} from "../kernels/linear-i8a8.ts";
import {
  QUANTIZE_ROWS_KEY,
  QUANTIZE_ROWS_WGSL,
  quantizeRowsParams,
} from "../kernels/quantize-rows.ts";
import type { WeightStorage } from "../kernels/weight-storage.ts";
import {
  MASKED_FILL_KEY,
  MASKED_FILL_WGSL,
  MASKED_FILL_WORKGROUP_SIZE,
  maskedFillParams,
} from "../kernels/masked-fill.ts";
import {
  SAFE_SOFTMAX_KEY,
  SAFE_SOFTMAX_WGSL,
  SOFTMAX_KEY,
  SOFTMAX_WGSL,
  softmaxParams,
} from "../kernels/softmax.ts";
import {
  ATTENTION_STATS_STRIDE,
  attentionPvKey,
  attentionPvParams,
  attentionPvWgsl,
  attentionQkKey,
  attentionQkParams,
  attentionQkWgsl,
  attentionStatsKey,
  attentionStatsParams,
  attentionStatsRegCache,
  attentionStatsWgsl,
} from "../kernels/attention.ts";
import { defaultI8a8Geometry, i8a8TileM, i8a8TileN } from "../kernels/i8a8-geometry.ts";
import {
  ATTENTION_PV_V_SCALE_BINDING,
  ATTENTION_QK_K_SCALE_BINDING,
  ATTENTION_QK_Q_SCALE_BINDING,
  attentionPvI8a8Key,
  attentionPvI8a8Params,
  attentionPvI8a8UsesVec4,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
  attentionQkI8a8Params,
  attentionQkI8a8UsesVec4,
  attentionQkI8a8Wgsl,
} from "../kernels/attention-i8a8.ts";
import {
  attentionScoreUsesF16,
  type ScoreStorage,
  scoreStorageBytes,
} from "../kernels/score-storage.ts";
import type { KarumeModel } from "../format/container.ts";
import type { IrDtype } from "../format/ir.ts";
import type { RunArena } from "../gpu/arena.ts";
import type { GpuContext } from "../gpu/device.ts";
import type { PipelineCache } from "../gpu/pipeline-cache.ts";
import { bmmKey, bmmParams, bmmWgsl } from "../kernels/bmm.ts";
import {
  ATTENTION_QK_MASK_BINDING,
  type GemmCompute,
  gemmMTileGeometry,
  gemmUsesVec4,
} from "../kernels/gemm.ts";
import {
  defaultGemmGeometry,
  gemmGeometryForRows,
  gemmTileM,
  gemmTileN,
} from "../kernels/gemm-geometry.ts";
import { GATHER_KEY, GATHER_WGSL, GATHER_WORKGROUP_SIZE, gatherParams } from "../kernels/gather.ts";
import { matmulKey, matmulParams, matmulWgsl } from "../kernels/matmul.ts";
import {
  attentionScale,
  type BinaryOpName,
  CAST_OP,
  catDim,
  conv1dAttrs,
  conv2dAttrs,
  convTranspose1dAttrs,
  deformConv2dAttrs,
  flipDim,
  layerNormAttrs,
  maskedFillValue,
  numel,
  padAttrs,
  permuteDims,
  reduceDim,
  type ReduceOpName,
  rmsNormEps,
  scalarParamValues,
  sliceAttrs,
  type UnaryOpName,
  WEIGHT_SLOTS,
  WHERE_OP,
} from "../ops.ts";
import type { ExecStep, FusedOperand, FusedStep } from "./fusion.ts";
import { ExecutionError, type NodePlan } from "./plan.ts";
import {
  type BindingRecipe,
  type BindingSource,
  type OutputRecipe,
  type StepRecipe,
  StepRecipeBuilder,
  type TempSource,
  type ValueSource,
} from "./recipe.ts";
import type { ComputePrecision, I8a8Dot, ParamsCacheStats } from "./session-types.ts";

/** elementwise 族の生成入力のうち rank に依らない部分（rank はエンコード時に決まる）。 */
type ElementwiseOp =
  | { readonly op: UnaryOpName | BinaryOpName | typeof WHERE_OP; readonly dtype: IrDtype }
  | { readonly op: typeof CAST_OP; readonly dtype: IrDtype; readonly to: IrDtype };

const PARAMS_STORAGE_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const PARAMS_UNIFORM_USAGE = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/**
 * 導出相が読む Session の状態（**必要な欄だけ**の構造的な面）。
 *
 * MUST: executor.ts の `SessionState` を import しない — Session → RecipeBuilder の一方向
 * import を型でも崩さないため。Session は自分の状態をそのまま渡す（構造的に適合する）。
 * MUST: run 寿命の器（{@link RunArena} の run アリーナ・env・スケジューラ）は載せない。
 * 載せた時点で「導出相は run 寿命の状態に触れない」がモジュール doc の宣言だけになり、
 * 導出済み計画を Session へ常駐させる根拠が型の上から消える。
 */
type RecipeBuilderContext = {
  readonly gpu: GpuContext;
  readonly model: KarumeModel;
  readonly cache: PipelineCache;
  /** params の確保先（**Session 常駐**の weights アリーナ — `#writeParams` の MUST）。 */
  readonly weights: RunArena;
  readonly weightBuffers: ReadonlyMap<string, GPUBuffer>;
  readonly paramsCache: Map<string, GPUBuffer>;
  readonly weightStorages: ReadonlyMap<string, WeightStorage>;
  readonly weightScaleBuffers: ReadonlyMap<string, GPUBuffer>;
  readonly linearCompute: "f32" | "i8a8" | "f16";
  readonly attentionCompute: ComputePrecision;
  readonly attentionScoreStorage: ScoreStorage;
  readonly i8a8Dot: I8a8Dot;
  readonly useCounts: ReadonlyMap<string, number>;
  readonly outputNames: ReadonlySet<string>;
};

/**
 * 導出相の本体。Session は 1 個だけ持ち、run / enqueue のミス経路から
 * {@link RecipeBuilder.buildRecipes} を呼ぶ。
 */
export class RecipeBuilder {
  readonly #state: RecipeBuilderContext;
  /** 進行中 run の params 実績（run の頭でリセットし、決着時に Session の診断へ移す）。 */
  #paramsAllocCount = 0;
  #paramsReuseCount = 0;

  constructor(state: RecipeBuilderContext) {
    this.#state = state;
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
   */
  async buildRecipes(steps: readonly ExecStep[]): Promise<readonly StepRecipe[]> {
    // 実体は実行相まで決まらないので、導出相は「その値名が既に定義済みか」だけを追う
    // （束縛漏れを実行相へ持ち越さず、ここで fail loudly にする）。
    const defined = new Set(this.#state.model.graph.inputs.map((spec) => spec.name));
    const recipes: StepRecipe[] = [];
    for (const step of steps) recipes.push(await this.#buildStep(step, defined));
    return recipes;
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
   * 両者で**1 本**に閉じる（実行側は {@link executeStepRecipe}）。融合ごとに手書きの解放簿記を
   * 置くと、アリーナの参照計数が融合の本数だけ別実装になり、1 本でもずれると例外なしの
   * 沈黙誤値になる（早すぎる解放ならプール再利用で値が化け、多すぎれば peak が落ちない）。
   */
  async #buildStep(step: ExecStep, defined: Set<string>): Promise<StepRecipe> {
    const outputName = step.kind === "node" ? step.plan.outputName : step.outputName;
    const outputShape = step.kind === "node" ? step.plan.outputShape : step.outputShape;
    // bind 面のオペランド順（重複無し）と解放簿記の延べ列は別物。素のノードでは
    // どちらも node.ins に一致し、融合ステップだけが 2 つを別々に宣言する。
    const bindNames = step.kind === "node" ? step.plan.node.ins : step.binds;
    const consumedNames = step.kind === "node" ? step.plan.node.ins : step.ins;
    const binds = bindNames.map((name) => this.#bindingSource(name, defined));
    // reshape と恒等 expand は要素順を変えないので**入力バッファをそのまま出力の実体にする**
    // （別名 — ADR 0011）。dispatch も確保も出さない。要素数一致は planGraph が済ませているので、
    // 別名先の実バッファは宣言 shape ぶんの大きさを必ず満たす。
    // MUST: 別名元は bind 面の先頭（temp にはなりえない）。
    const output: OutputRecipe = step.kind === "node" && step.aliasesInput
      ? { kind: "alias", source: binds[0] }
      : { kind: "alloc", byteLength: numel(outputShape) * 4 };
    // 出力は「この値名」として束ねる。実行相が dispatch より前に env へ載せるので、
    // 同一ステップ内の bind もこの 1 本で解決できる。
    const out: BindingSource = { kind: "value", name: outputName };

    const builder = new StepRecipeBuilder();
    if (step.kind === "node") {
      await this.#buildNode(step.plan, step.aliasesInput, binds, out, builder);
    } else {
      await this.#buildFused(step, binds, out, builder);
    }
    defined.add(outputName);

    return {
      outputName,
      output,
      uses: this.#state.useCounts.get(outputName) ?? 0,
      pinned: this.#state.outputNames.has(outputName),
      temps: builder.temps,
      dispatches: builder.dispatches,
      releases: consumedNames,
    };
  }

  /**
   * 融合ステップの dispatch 列。
   *
   * bind 面の既定は「params, 入力…, 出力」（融合 4 ルール共通）で、`operands` を宣言した
   * ルールだけがステップ内一時を混ぜた並びを取る。**一時の確保・解放は
   * {@link StepRecipeBuilder} に replay させる**（寿命の導出点を 2 つに増やさない）ので、
   * 実行相の簿記は素のノードと同じ 1 本（{@link executeStepRecipe}）に閉じたままになる。
   */
  async #buildFused(
    step: FusedStep,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const temps: TempSource[] = [];
    const resolve = (operand: FusedOperand): BindingSource => {
      if (operand.kind === "output") return out;
      if (operand.kind === "bind") {
        const source = binds[operand.index];
        if (source === undefined) {
          throw new ExecutionError(
            `融合ルール '${step.rule}': bind 添字 ${operand.index} が宣言 ${binds.length} 本の外`,
          );
        }
        return source;
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
      const { pipeline, layout } = await this.#state.cache.get(dispatch.key, dispatch.wgsl());
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
        params,
        bindings: operands.map((operand, slot) => ({
          binding: slot + 1,
          source: resolve(operand),
        })),
        workgroups: workgroups.kind === "tiled"
          ? workgroups.counts
          : [gridStrideWorkgroups(workgroups.items, workgroups.size, limit), 1, 1],
      });
      // MUST: 同一境界の解放は確保の逆順（{@link executeStepRecipe} と同じ LIFO）。
      for (let id = step.temps.length - 1; id >= 0; id -= 1) {
        if (step.temps[id].releaseAfter === index) builder.releaseTemp(temps[id]);
      }
    }
  }

  /** 素のノード 1 つの本体（確保・retain・解放は {@link executeStepRecipe} が済ませる）。 */
  async #buildNode(
    step: NodePlan,
    aliasesInput: boolean,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    switch (step.contract.kind) {
      case "unary":
      case "binary":
        await this.#buildElementwise(
          step,
          { op: step.contract.name, dtype: step.inputDtypes[0] },
          binds,
          out,
          builder,
        );
        break;
      case "cast":
        await this.#buildElementwise(
          step,
          { op: CAST_OP, dtype: step.inputDtypes[0], to: step.outputDtype },
          binds,
          out,
          builder,
        );
        break;
      case "where":
        // 生成入力は**値スロット**の dtype（スロット 0 は条件で常に bool）。
        await this.#buildElementwise(
          step,
          { op: WHERE_OP, dtype: step.inputDtypes[1] },
          binds,
          out,
          builder,
        );
        break;
      case "cumsum":
        await this.#buildCumsum(step, binds, out, builder);
        break;
      case "matmul":
        await this.#buildMatmul(step, binds, out, builder);
        break;
      case "bmm":
        await this.#buildBmm(step, binds, out, builder);
        break;
      case "gather":
        await this.#buildGather(step, binds, out, builder);
        break;
      case "rowReduce":
        await this.#buildRowReduce(step, step.contract.name, binds, out, builder);
        break;
      case "permute":
      case "slice":
      case "symPrefixSlice":
        await this.#buildStridedCopy(step, step.contract.kind, binds, out, builder);
        break;
      case "expand":
        // 恒等 expand は別名化済み（0 dispatch）。複製軸が 1 つでもあれば実体化コピー。
        if (!aliasesInput) {
          await this.#buildStridedCopy(step, step.contract.kind, binds, out, builder);
        }
        break;
      case "cat":
        await this.#buildCat(step, binds, out, builder);
        break;
      case "pad":
        await this.#buildPad(step, binds, out, builder);
        break;
      case "flip":
        await this.#buildFlip(step, binds, out, builder);
        break;
      case "linear":
        await this.#buildLinear(step, binds, out, builder);
        break;
      case "layerNorm":
        await this.#buildLayerNorm(step, binds, out, builder);
        break;
      case "rmsNorm":
        await this.#buildRmsNorm(step, binds, out, builder);
        break;
      case "softmax":
        await this.#buildSoftmax(step, false, binds, out, builder);
        break;
      case "safeSoftmax":
        await this.#buildSoftmax(step, true, binds, out, builder);
        break;
      case "attention":
        await this.#buildAttention(step, binds, out, builder);
        break;
      case "embedding":
        await this.#buildEmbedding(step, binds, out, builder);
        break;
      case "maskedFill":
        await this.#buildMaskedFill(step, binds, out, builder);
        break;
      case "conv1d":
        await this.#buildConv1d(step, binds, out, builder);
        break;
      case "conv2d":
        await this.#buildConv2d(step, binds, out, builder);
        break;
      case "convTranspose1d":
        await this.#buildConvTranspose1d(step, binds, out, builder);
        break;
      case "deformConv2d":
        await this.#buildDeformConv2d(step, binds, out, builder);
        break;
      case "upsampleBilinear2d":
        await this.#buildUpsampleBilinear2d(step, binds, out, builder);
        break;
      case "gruScan":
        await this.#buildGruScan(step, binds, out, builder);
        break;
      case "reshape":
        // 別名化は #buildStep で済んでいる（この op は 1 dispatch も出さない）。
        break;
      default: {
        // MUST: 未処理の kind をここで止める。switch が素通りすると出力バッファに 1 バイトも
        // 書かれないまま次のノードへ進み、プール再利用の残骸がそのまま値になる
        // （full-write 不変条件の破れ — ADR 0014）。型としては到達不能で、op を足したときの
        // 結線漏れをコンパイル時に赤くするのがこの分岐の役目。
        const unhandled: never = step.contract;
        throw new ExecutionError(`未処理の op kind: ${JSON.stringify(unhandled)}`);
      }
    }
  }

  async #buildElementwise(
    step: NodePlan,
    element: ElementwiseOp,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    // rank 0（スカラ）は codegen の rank ≥ 1 契約に合わせて長さ 1 の 1 次元に正規化する。
    const outShape = step.outputShape.length === 0 ? [1] : [...step.outputShape];
    const rank = outShape.length;
    const spec: ElementwiseSpec = element.op === CAST_OP
      ? { op: CAST_OP, rank, dtype: element.dtype, to: element.to }
      : { op: element.op, rank, dtype: element.dtype };
    const key = elementwiseKey(spec);
    const { pipeline, layout } = await this.#state.cache.get(key, elementwiseWgsl(spec));
    // attrs のスカラ（clamp の min/max など）は params の末尾に f32 で載る（並びは契約表）。
    const params = this.#writeParams(
      elementwiseParams(
        spec,
        outShape,
        step.inputShapes,
        scalarParamValues(step.contract, step.node.attrs, `nodes (${step.node.op})`),
      ),
      PARAMS_STORAGE_USAGE,
    );
    const groups = gridStrideWorkgroups(
      numel(outShape),
      ELEMENTWISE_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: binds.length + 1, source: out },
      ],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * reduce 族（sum / amax / amin）。**縮約軸で 2 変種に踏み分ける**:
   *
   * - 最終次元 → 行 reduce（既存カーネル・1 行 = 1 workgroup + 256 幅ツリー）
   * - それ以外 → 軸 reduce（1 スレッド = 1 出力・コアレス読み。縮約順序は行 reduce と
   *   厳密に一致 = 出力ビット同一 — src/codegen/reduce.ts の {@link axisReduceWgsl}）
   *
   * MUST: 分岐は**軸だけ**で決める。速度で選ぶ余地を作ると、最終次元でも軸変種が走る形が
   * でき、既定経路のビット不変（PNG sha256 門）が実行時条件に依存してしまう。
   */
  async #buildRowReduce(
    step: NodePlan,
    op: ReduceOpName,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const inputShape = step.inputShapes[0];
    const axis = reduceDim(step.node.attrs, `nodes (${step.node.op})`);
    // 要素型はキーに載る（bool の sum は u32 を読んで i32 の個数を書く — ADR 0009）。
    const spec = { op, dtype: step.inputDtypes[0] };
    const lastDim = axis === inputShape.length - 1;
    const outCount = numel(step.outputShape);
    const key = lastDim ? reduceKey(spec) : axisReduceKey(spec);
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      lastDim ? reduceWgsl(spec) : axisReduceWgsl(spec),
    );
    const inner = numel(inputShape.slice(axis + 1));
    const params = this.#writeParams(
      lastDim
        ? reduceParams(outCount, inputShape[axis])
        : axisReduceParams(outCount, inputShape[axis], inner),
      PARAMS_UNIFORM_USAGE,
    );
    // 行 reduce は 1 行 = 1 workgroup、軸 reduce は 1 スレッド = 1 出力。どちらも上限を
    // 超えたら縮退させ、カーネル側の grid-stride で回す。
    const groups = gridStrideWorkgroups(
      outCount,
      lastDim ? 1 : AXIS_REDUCE_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: out }],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * cumsum（最終次元の前縁和）。**1 invocation = 1 行**の逐次走査で、行方向を grid-stride で
   * 回す（形の根拠は src/kernels/cumsum.ts）。
   */
  async #buildCumsum(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const shape = step.outputShape;
    const dim = shape[shape.length - 1];
    const rows = numel(shape.slice(0, -1));
    const { pipeline, layout } = await this.#state.cache.get(CUMSUM_KEY, CUMSUM_WGSL);
    const params = this.#writeParams(cumsumParams(rows, dim), PARAMS_UNIFORM_USAGE);
    const groups = gridStrideWorkgroups(
      rows,
      CUMSUM_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: CUMSUM_KEY,
      pipeline,
      layout,
      params,
      bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: out }],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * permute / 非恒等 expand / slice / sym_prefix_slice の実体化コピー（strided 読み 1 カーネル族 —
   * ADR 0011 / 0010 / 0014）。出力は常に連続で、入力側だけを stride で読む。expand の複製軸は
   * stride 0、sym_prefix_slice は **Tmax 形の入力**の連続 stride、slice は入力の連続 stride と
   * **開始位置 offset**。恒等 expand は別名化されるのでここへは来ない。
   */
  async #buildStridedCopy(
    step: NodePlan,
    kind: "permute" | "expand" | "slice" | "symPrefixSlice",
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const srcShape = step.inputShapes[0];
    const outShape = step.outputShape;
    const where = `nodes (${step.node.op})`;
    const srcStrides = kind === "permute"
      ? permuteSrcStrides(srcShape, permuteDims(step.node.attrs, where))
      : kind === "expand"
      ? expandSrcStrides(srcShape, outShape)
      : kind === "slice"
      ? sliceSrcStrides(srcShape)
      // MUST: 束縛後の outShape ではなく **srcShape（Tmax 形）** から組む（prefixSliceSrcStrides
      // の MUST — 送り幅を縮めると 2 行目以降が別の行を読む）。
      : prefixSliceSrcStrides(srcShape);
    // offset は ADR 0011 の (offset, strides[4]) モデルそのもの。permute / expand /
    // sym_prefix_slice は入力の先頭から読むので 0 で、**slice だけが 0 以外**を取る
    // （ADR 0011 が予告した「可変点 1 語」がここ）。
    let offset = 0;
    if (kind === "slice") {
      const { dim, start } = sliceAttrs(step.node.attrs, where);
      offset = sliceSrcOffset(srcShape, dim, start);
    }
    // 要素型はキーに載る（bool マスク / i32 添字の expand が実測にある — ADR 0009）。
    const spec = { dtype: step.outputDtype };
    const key = stridedKey(spec);
    const { pipeline, layout } = await this.#state.cache.get(key, stridedWgsl(spec));
    const params = this.#writeParams(
      stridedParams(outShape, srcStrides, offset),
      PARAMS_STORAGE_USAGE,
    );
    const groups = gridStrideWorkgroups(
      numel(outShape),
      STRIDED_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: out }],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * cat（strided 書きコピー族 — ADR 0014）。入力ごとに **1 dispatch** で出力の部分領域へ書く。
   * 入力は連続で読み、出力へ `(offset, 出力の連続 strides)` で書く（読み族の双対）。
   *
   * MUST: 書き出し位置の総和が連結軸の長さとちょうど一致することを確かめる（full-write）。
   * 一致しなければ出力のどこかが**未書き込みのまま**残り、プール再利用なら前の値がそのまま
   * 結果になる。契約の shape 規則（軸長 = 入力の軸長の総和）と同じ事実をエンコード側の
   * 積み上げからも確かめる形で、片方だけの誤りを内部矛盾として落とす。
   */
  async #buildCat(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const outShape = step.outputShape;
    const where = `nodes (${step.node.op})`;
    const dim = catDim(step.node.attrs, where);
    const outStrides = catOutStrides(outShape);
    const spec = { dtype: step.outputDtype };
    const key = stridedWriteKey(spec);
    const { pipeline, layout } = await this.#state.cache.get(key, stridedWriteWgsl(spec));
    let written = 0;
    for (const [index, source] of binds.entries()) {
      const srcShape = step.inputShapes[index];
      const params = this.#writeParams(
        stridedWriteParams(srcShape, outStrides, catOutOffset(outShape, dim, written)),
        PARAMS_STORAGE_USAGE,
      );
      const groups = gridStrideWorkgroups(
        numel(srcShape),
        STRIDED_WRITE_WORKGROUP_SIZE,
        this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
      );
      builder.dispatch({
        key,
        pipeline,
        layout,
        params,
        bindings: [{ binding: 1, source }, { binding: 2, source: out }],
        workgroups: [groups, 1, 1],
      });
      written += srcShape[dim];
    }
    if (written !== outShape[dim]) {
      throw new ExecutionError(
        `${where}: cat の書き込みが出力全域を覆わない（軸 ${dim} に ${written} / 出力は ${
          outShape[dim]
        }）`,
      );
    }
  }

  /**
   * pad（最終次元の定数 0 埋め）。**1 dispatch で出力の全バイトを書く**（範囲内は転写・
   * 範囲外は 0 — ADR 0014 の full-write）。ゼロ初期化されたバッファを前提にしない。
   */
  async #buildPad(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const srcShape = step.inputShapes[0];
    const outShape = step.outputShape;
    const { left, right } = padAttrs(step.node.attrs, `nodes (${step.node.op})`);
    const { pipeline, layout } = await this.#state.cache.get(PAD_KEY, PAD_WGSL);
    const params = this.#writeParams(
      padParams(numel(srcShape.slice(0, -1)), srcShape[srcShape.length - 1], left, right),
      PARAMS_UNIFORM_USAGE,
    );
    const groups = gridStrideWorkgroups(
      numel(outShape),
      PAD_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: PAD_KEY,
      pipeline,
      layout,
      params,
      bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: out }],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * flip（静的軸の添字反転）。軸の位置は `[outer, len, inner]` の 3 分割に畳んで渡す
   * （rank を params に載せない — src/kernels/flip.ts）。
   */
  async #buildFlip(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const shape = step.outputShape;
    const dim = flipDim(step.node.attrs, `nodes (${step.node.op})`);
    const { pipeline, layout } = await this.#state.cache.get(FLIP_KEY, FLIP_WGSL);
    const params = this.#writeParams(
      // MUST: 軸の前後で分ける（`slice(0, dim)` と `slice(dim + 1)`）。境界を 1 つずらすと
      // 反転する軸が隣にずれ、shape も要素数も変わらないまま値だけが誤る。
      flipParams(numel(shape.slice(0, dim)), shape[dim], numel(shape.slice(dim + 1))),
      PARAMS_UNIFORM_USAGE,
    );
    const groups = gridStrideWorkgroups(
      numel(shape),
      FLIP_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: FLIP_KEY,
      pipeline,
      layout,
      params,
      bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: out }],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * deform_conv2d（DCNv2 — ADR 0055）。出力 1 要素 = 1 invocation の grid-stride 1 本だけで、
   * 踏み分けは無い（groups / offset_groups / stride / dilation は契約に欄が無い = 1 固定）。
   *
   * MUST: Kh / Kw は**重みの第 3 / 第 4 軸**をこの順で読む（conv2d と同じ教訓）。
   * offset / mask の形は契約検査が出力空間と突き合わせ済み。
   */
  async #buildDeformConv2d(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const [x, weight] = step.inputShapes;
    const outShape = step.outputShape;
    const { padding } = deformConv2dAttrs(step.node.attrs, `nodes (${step.node.op})`);
    const { pipeline, layout } = await this.#state.cache.get(
      DEFORM_CONV2D_KEY,
      DEFORM_CONV2D_WGSL,
    );
    const params = this.#writeParams(
      deformConv2dParams({
        batch: outShape[0],
        channelsIn: x[1],
        channelsOut: outShape[1],
        heightIn: x[2],
        widthIn: x[3],
        heightOut: outShape[2],
        widthOut: outShape[3],
        kernelH: weight[2],
        kernelW: weight[3],
        paddingH: padding[0],
        paddingW: padding[1],
      }),
      PARAMS_UNIFORM_USAGE,
    );
    const workgroups = gridStrideWorkgroups(
      numel(outShape),
      DEFORM_CONV2D_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: DEFORM_CONV2D_KEY,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 6, source: out },
      ],
      workgroups: [workgroups, 1, 1],
    });
  }

  /**
   * gru_scan / gru_scan_reverse（GRU の隠れ側スキャン — ADR 0056）。
   *
   * 1 workgroup = 1 バッチ要素（バッチ方向だけ grid-stride で dispatch 上限を跨ぐ）で、
   * 時間ループはカーネル内。走査方向は **op 名**から引く（attrs に欄は無い）。
   *
   * MUST: 幾何は h0 と W_hh から取る（T は gi の先頭・N と H は h0）。gi の最終次元 3H から
   * H を割り戻すと、契約検査が突き合わせているはずの取り違えを 1 か所で作り直すことになる。
   */
  async #buildGruScan(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const direction = step.node.op === "gru_scan_reverse" ? "reverse" : "forward";
    const key = gruScanKey(direction);
    const { pipeline, layout } = await this.#state.cache.get(key, gruScanWgsl(direction));
    const [time, batch, hidden] = step.outputShape;
    const params = this.#writeParams(
      gruScanParams({ time, batch, hidden }),
      PARAMS_UNIFORM_USAGE,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 5, source: out },
      ],
      workgroups: [
        gridStrideWorkgroups(batch, 1, this.#state.gpu.limits.maxComputeWorkgroupsPerDimension),
        1,
        1,
      ],
    });
  }

  /**
   * upsample_bilinear2d（NCHW の空間 2 軸を双線形 resample・`align_corners = True` 専業）。
   * 出力 1 要素 = 1 invocation の grid-stride で、params は空間 4 長だけ運ぶ
   * （N·C はカーネル側が平面添字へ畳む — src/kernels/upsample-bilinear2d.ts）。
   */
  async #buildUpsampleBilinear2d(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const srcShape = step.inputShapes[0];
    const outShape = step.outputShape;
    const { pipeline, layout } = await this.#state.cache.get(
      UPSAMPLE_BILINEAR2D_KEY,
      UPSAMPLE_BILINEAR2D_WGSL,
    );
    const params = this.#writeParams(
      upsampleBilinear2dParams(
        numel(outShape),
        srcShape[2],
        srcShape[3],
        outShape[2],
        outShape[3],
      ),
      PARAMS_UNIFORM_USAGE,
    );
    const groups = gridStrideWorkgroups(
      numel(outShape),
      UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: UPSAMPLE_BILINEAR2D_KEY,
      pipeline,
      layout,
      params,
      bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: out }],
      workgroups: [groups, 1, 1],
    });
  }

  async #buildMatmul(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const [a, b] = step.inputShapes;
    const [m, k] = a;
    const n = b[1];
    // v4（vec4 の読み書き）は形状から導く 1 ビット。導出時に評価してキーと WGSL の
    // 両方へ渡す（同一キー ⇔ 同一バイト列は保たれる）。
    const v4 = gemmUsesVec4(k, n);
    // MUST: タイル幾何は行数 M のバケット（src/kernels/gemm-geometry.ts）。キー・WGSL・
    // dispatch の 3 つに**同じ m** を通す — 1 つでも渡し忘れると出力タイルが静かに欠ける。
    const key = matmulKey(v4, m);
    const { pipeline, layout } = await this.#state.cache.get(key, matmulWgsl(v4, m));
    const params = this.#writeParams(matmulParams(m, n, k), PARAMS_UNIFORM_USAGE);
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const where = `matmul [${a.join(",")}] × [${b.join(",")}]`;
    const geometry = gemmGeometryForRows(m);
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: binds[1] },
        { binding: 3, source: out },
      ],
      workgroups: [
        tiledWorkgroups(n, gemmTileN(geometry), limit, where),
        tiledWorkgroups(m, gemmTileM(geometry), limit, where),
        1,
      ],
    });
  }

  /**
   * バッチ matmul（rank-3）。タイル 2 軸に加えて**バッチを z 軸**へ載せる。
   * MUST: matmul と同じ「1 workgroup = 1 タイル」なので、3 軸とも上限超過は fail loudly。
   */
  async #buildBmm(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const [a, b] = step.inputShapes;
    const [batch, m, k] = a;
    const n = b[2];
    const v4 = gemmUsesVec4(k, n);
    // 幾何のバケットは**行列 1 枚の m**（バッチは z 軸で、タイル幾何とは独立）。
    const key = bmmKey(v4, m);
    const { pipeline, layout } = await this.#state.cache.get(key, bmmWgsl(v4, m));
    const params = this.#writeParams(bmmParams(m, n, k), PARAMS_UNIFORM_USAGE);
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const where = `bmm [${a.join(",")}] × [${b.join(",")}]`;
    const geometry = gemmGeometryForRows(m);
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: binds[1] },
        { binding: 3, source: out },
      ],
      workgroups: [
        tiledWorkgroups(n, gemmTileN(geometry), limit, where),
        tiledWorkgroups(m, gemmTileM(geometry), limit, where),
        // バッチは 1 workgroup = 1 バッチ。ここも縮退させるとバッチが丸ごと未書き込みになる。
        tiledWorkgroups(batch, 1, limit, where),
      ],
    });
  }

  /**
   * 最終次元の gather。出力は連続で、`row = i / J` から `src[row * D + index[i]]` を引く。
   * 範囲外添字の扱いは src/kernels/gather.ts の裁定（GPU は NaN 汚染 / CPU 参照は throw）。
   */
  async #buildGather(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const srcShape = step.inputShapes[0];
    const outShape = step.outputShape;
    const count = numel(outShape);
    const { pipeline, layout } = await this.#state.cache.get(GATHER_KEY, GATHER_WGSL);
    const params = this.#writeParams(
      gatherParams(count, outShape[outShape.length - 1], srcShape[srcShape.length - 1]),
      PARAMS_UNIFORM_USAGE,
    );
    const groups = gridStrideWorkgroups(
      count,
      GATHER_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: GATHER_KEY,
      pipeline,
      layout,
      params,
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: binds[1] },
        { binding: 3, source: out },
      ],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * linear（融合 op — ADR 0012）。先行次元を平坦化して `[m,k] × [k,n] + bias` の 2 次元
   * GEMM に落とす。重みは `[n,k]` の転置レイアウトのままカーネルが読む（転置コピー無し）。
   * MUST: matmul と同じ「1 workgroup = 1 タイル」なので、上限超過は fail loudly。
   */
  async #buildLinear(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const [x, weight] = step.inputShapes;
    const n = weight[0];
    const k = weight[1];
    const m = numel(x.slice(0, -1));
    const weightStorage = this.#weightStorage(step);
    // w8a8 は **opt-in × i8 常駐 × k > 0 × k % 4 == 0** の 4 条件が揃ったときだけ（ADR 0025 予定）。
    // 既定の "f32" では 1 バイトも挙動が変わらない。
    // MUST: `k > 0` を含める。`k == 0` は契約上有効な退化 shape（src/ops.ts の linear は
    // in=0 を許す）だが、i8a8 経路の ① `quantize_rows` は `dim >= 1` を要求するので、拾うと
    // i8a8 固有の CodegenError になる — 縮約が空 = 量子化する活性がそもそも無く、i8a8 の門と
    // しての意味を持たない例外なので、経路の選択で失敗の理由が変わらないよう通常経路へ落とす。
    // NOTE: K=0 自体は通常経路でも 0 バイト束縛が最小束縛サイズを割って落ちる（この述語とは
    // 無関係の別要因）。
    if (
      this.#state.linearCompute === "i8a8" && weightStorage === "i8" && k > 0 && k % 4 === 0
    ) {
      await this.#buildLinearI8a8(step, binds, out, builder, m, n, k);
      return;
    }
    // f16 計算変種（ADR 0028）。**i8 常駐の重みとは組めない**（w8a16 は未実装）ので、
    // 黙って f32 へ落とさずここで落とす — 落とすと「i8 の層だけ f16 が効かない」形になり、
    // 診断のキーを見ない限り気づけない。
    const compute: GemmCompute = this.#state.linearCompute === "f16" ? "f16" : "f32";
    if (compute === "f16" && weightStorage === "i8") {
      throw new ExecutionError(
        `linear [${x.join(",")}] × [${weight.join(",")}]: ` +
          "linearCompute 'f16' は i8 常駐の重みとは組めない（w8a16 は未実装 — ADR 0028）。" +
          "linearCompute を 'i8a8' にするか、この重みを f32 / f16 格納で持つこと",
      );
    }
    const v4 = gemmUsesVec4(k, n);
    // MUST: タイル幾何は平坦化後の行数 m のバケット（src/kernels/gemm-geometry.ts）。
    // キー・WGSL・dispatch に**同じ m** を通す。
    const key = linearKey(weightStorage, v4, compute, m);
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      linearWgsl(weightStorage, v4, compute, m),
    );
    const params = this.#writeParams(linearParams(m, n, k), PARAMS_UNIFORM_USAGE);
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const where = `linear [${x.join(",")}] × [${weight.join(",")}]`;
    const geometry = gemmGeometryForRows(m);
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 4, source: out },
        ...this.#weightScaleBindings(step, weightStorage, LINEAR_SCALE_BINDING),
      ],
      workgroups: [
        tiledWorkgroups(n, gemmTileN(geometry), limit, where),
        tiledWorkgroups(m, gemmTileM(geometry), limit, where),
        1,
      ],
    });
  }

  /**
   * linear の **w8a8 変種**（opt-in — {@link SessionOptions.linearCompute}）。**1 ノード =
   * 2 dispatch**（融合 attention と同じ「複数 dispatch で 1 ノード」の扱い）:
   *
   * ① `quantize_rows`（活性を per-token i8 へ・行方向 grid-stride）→ ② i8a8 GEMM（整数内積・
   * 1 workgroup = 1 出力タイルなので上限超過は fail loudly）。
   *
   * MUST: 一時バッファ（`xq` / `xs`）は宣言 → ノード末尾で解放する。これで実行相の参照計数が
   * 閉じ（`assertDrained`）、失敗経路でも `arena.destroy()` が拾う。
   * MUST: `k` のオーバフロー門は fail loudly。黙って通すと i32 の巻き戻りで符号ごと化ける。
   */
  async #buildLinearI8a8(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
    m: number,
    n: number,
    k: number,
  ): Promise<void> {
    const [x, weight] = step.inputShapes;
    const where = `linear i8a8 [${x.join(",")}] × [${weight.join(",")}]`;
    if (k > LINEAR_I8A8_MAX_K) {
      throw new ExecutionError(
        `${where}: k=${k} が i8a8 経路の i32 縮約の門 ${LINEAR_I8A8_MAX_K} を超える` +
          "（linearCompute を 'f32' にするか、この linear を i8 常駐から外す）",
      );
    }
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;

    // 量子化した活性 `xq`（i8 を 4 詰め）と per-token scale `xs`。ノード内で閉じた一時領域。
    const xq = builder.allocTemp(Math.max(4, m * (k / 4) * 4));
    const xs = builder.allocTemp(Math.max(4, m * 4));

    // ① 活性の per-token 量子化（1 行 = 1 workgroup・行方向 grid-stride）
    const { pipeline: quantizePipeline, layout: quantizeLayout } = await this.#state.cache.get(
      QUANTIZE_ROWS_KEY,
      QUANTIZE_ROWS_WGSL,
    );
    builder.dispatch({
      key: QUANTIZE_ROWS_KEY,
      pipeline: quantizePipeline,
      layout: quantizeLayout,
      params: this.#writeParams(quantizeRowsParams(m, k), PARAMS_UNIFORM_USAGE),
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: xq },
        { binding: 3, source: xs },
      ],
      workgroups: [gridStrideWorkgroups(m, 1, limit), 1, 1],
    });

    // ② 整数内積の GEMM。タイル幾何は op → 幾何の純関数が決める（src/kernels/i8a8-geometry.ts）
    // — キーに載るので「同一キー → バイト同一 WGSL」は保たれる。
    const v4 = linearI8a8UsesVec4(n);
    const geometry = defaultI8a8Geometry("linear");
    const key = linearI8a8Key(v4, this.#state.i8a8Dot === "dp4a", geometry);
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      linearI8a8Wgsl(v4, this.#state.i8a8Dot === "dp4a", geometry),
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params: this.#writeParams(linearI8a8Params(m, n, k), PARAMS_UNIFORM_USAGE),
      bindings: [
        { binding: 1, source: xq },
        { binding: 2, source: binds[1] },
        { binding: 3, source: binds[2] },
        { binding: 4, source: out },
        ...this.#weightScaleBindings(step, "i8", LINEAR_SCALE_BINDING),
        { binding: LINEAR_ACT_SCALE_BINDING, source: xs },
      ],
      workgroups: [
        tiledWorkgroups(n, i8a8TileN(geometry), limit, where),
        tiledWorkgroups(m, i8a8TileM(geometry), limit, where),
        1,
      ],
    });

    // MUST: ノード境界で一時バッファを返す（アリーナの不変条件）。
    builder.releaseTemp(xs);
    builder.releaseTemp(xq);
  }

  /** layer_norm（最終次元・affine あり）。1 行 = 1 workgroup で、行方向は grid-stride。 */
  async #buildLayerNorm(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const shape = step.outputShape;
    const dim = shape[shape.length - 1];
    const rows = numel(shape.slice(0, -1));
    const { eps } = layerNormAttrs(step.node.attrs, `nodes (${step.node.op})`);
    const { pipeline, layout } = await this.#state.cache.get(LAYER_NORM_KEY, LAYER_NORM_WGSL);
    const params = this.#writeParams(
      layerNormParams(rows, dim, eps),
      PARAMS_UNIFORM_USAGE,
    );
    const groups = gridStrideWorkgroups(
      rows,
      1,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: LAYER_NORM_KEY,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 4, source: out },
      ],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * rms_norm（最終次元・weight のみ）。layer_norm と同じ 1 行 = 1 workgroup の形で、
   * 縮約は二乗和 1 パス（ADR 0017）。
   */
  async #buildRmsNorm(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const shape = step.outputShape;
    const dim = shape[shape.length - 1];
    const rows = numel(shape.slice(0, -1));
    const eps = rmsNormEps(step.node.attrs, `nodes (${step.node.op})`);
    const { pipeline, layout } = await this.#state.cache.get(RMS_NORM_KEY, RMS_NORM_WGSL);
    const params = this.#writeParams(rmsNormParams(rows, dim, eps), PARAMS_UNIFORM_USAGE);
    const groups = gridStrideWorkgroups(
      rows,
      1,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: RMS_NORM_KEY,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 3, source: out },
      ],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * softmax（最終次元、safe-softmax）。layer_norm と同じ 1 行 = 1 workgroup の形。
   *
   * `safe` は safe_softmax 変種（行 max が −inf の行に 0 を書く — ADR 0044）。カーネルは
   * 同じ生成関数から出た 2 本で、dispatch の形（バインド・workgroup 数）は同じ。
   */
  async #buildSoftmax(
    step: NodePlan,
    safe: boolean,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const shape = step.outputShape;
    const dim = shape[shape.length - 1];
    const rows = numel(shape.slice(0, -1));
    const key = safe ? SAFE_SOFTMAX_KEY : SOFTMAX_KEY;
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      safe ? SAFE_SOFTMAX_WGSL : SOFTMAX_WGSL,
    );
    const params = this.#writeParams(softmaxParams(rows, dim, safe), PARAMS_UNIFORM_USAGE);
    const groups = gridStrideWorkgroups(
      rows,
      1,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: out }],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * 融合 attention（ADR 0023）。**1 ノード = 3 dispatch**（`cat` と同じ「複数 dispatch で
   * 1 ノード」の扱い。full-write は**ノードの出力 O について**成立する）:
   *
   * ① QK gemm（S を実体化・scale はタイル充填時に q/k 両方へ）→ ② 行統計（m と 1/Σexp）→
   * ③ PV gemm（A タイル充填時に `exp(S−m)·inv` を評価 = P 非実体化）。
   *
   * 省略可能な第 4 入力 `mask[1,1,M,N]`（加算型）は **① の束縛が 1 本増えるだけ**で、
   * dispatch 数も ②③ の経路も変わらない（S が mask 済みで出てくる）。
   *
   * i8a8 変種（opt-in）では ① が 3 dispatch・③ が 3 dispatch に増える（最大 7）。**適格判定は
   * 段ごとに独立**（① は `D % 4 == 0`・③ は `N % 4 == 0`）なので、片方だけ i8a8 の**混成**が
   * 起こりうる — 満たさない段だけが f32 経路へ**沈黙で**縮退する（linear の `k % 4` と同じ
   * 流儀で、落ちたことは診断のパイプラインキーにだけ出る）。
   *
   * MUST: ① と ③ は 1 workgroup = 1 タイルなので、3 軸とも上限超過は fail loudly
   * （{@link tiledWorkgroups}）。縮退させるとタイルが欠落し、例外なしに O の一部が
   * 未書き込み（プール再利用なら前の値）で残る。② だけが行方向 grid-stride。
   * MUST: 一時バッファ（S / 行統計）は宣言 → ノード末尾で解放する。これで実行相の参照計数が
   * 閉じ（`assertDrained`）、失敗経路でも `arena.destroy()` が拾う（確保と破棄を 1 箇所へ —
   * ADR 0004）。
   */
  async #buildAttention(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const [q, k, v] = step.inputShapes;
    // B と H は 1 本のバッチ軸へ畳む（契約は rank-4 head-first で、3 者の B / H は一致）。
    const batch = q[0] * q[1];
    const rows = q[2];
    const cols = k[2];
    const depth = q[3];
    const scale = attentionScale(step.node.attrs, `nodes (${step.node.op})`);
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const where = `attention [${q.join(",")}] × [${k.join(",")}] × [${v.join(",")}]`;

    // f16 計算変種（ADR 0028）。3 カーネルが**同時に**切り替わる — S の格納形が ① の書き手と
    // ②③ の読み手で一致していなければならないので、段ごとの混成はあり得ない。
    const compute: GemmCompute = this.#state.attentionCompute === "f16" ? "f16" : "f32";
    // i8a8 変種（設計 §7 の波 1 + 波 2）。**② 行統計は f32 のまま**（S の格納形も f32 の
    // ままなので、②③ が読む S は ① がどちらの経路で書いても同型）。
    // MUST: 適格判定は**段ごとに独立**。① は q / k のパック方向が D・③ は P̃ / Vᵀ の
    // パック方向が N なので、条件も別々になる（両方満たさない形・片方だけの形が実在する）。
    const i8a8 = this.#state.attentionCompute === "i8a8";
    const qkI8a8 = i8a8 && depth % 4 === 0;
    const pvI8a8 = i8a8 && cols % 4 === 0;
    // S の**格納形**（案 γ 波 1 — 計算形と直交する第 2 の軸）。`pack2x16float` の 2 要素／語
    // なので、書き手 ①QK が v4 経路を取る形（`D % 4 == 0 && N % 4 == 0`）だけが適格で、
    // 非適格は f32 格納へ**沈黙で**縮退する（検出器はパイプラインキー）。
    // MUST: 3 カーネルが**同時に**切り替わる — S の格納形が書き手と読み手で一致していなければ
    // ならないので、段ごとの混成はあり得ない（i8a8 の適格判定が段ごとに独立なのとは別の話で、
    // ①が i8a8 でも f32 でも S の格納形は 1 つに決まる）。
    const scoreStorage: ScoreStorage =
      this.#state.attentionScoreStorage === "f16" && attentionScoreUsesF16(depth, cols)
        ? "f16"
        : "f32";

    // 加算 mask（省略可能な第 4 入力 — 契約は src/ops.ts）。① の epilogue だけの軸で、
    // ②③ からは「S が既に mask 済み」に見える。
    const mask = binds[3];
    // MUST: i8a8 の ①QK は別カーネル（src/kernels/attention-i8a8.ts）で epilogue を持たない。
    // 黙って f32 経路へ落とすと「i8a8 を頼んだのに効かない」沈黙になり、mask を落とすと
    // 値が壊れるので、組み合わせそのものを**一時バッファを取る前に** fail loudly にする。
    // MUST: 判定は **要求されたモード**（`i8a8`）で見る — `qkI8a8`（D % 4 の適格判定込み）で
    // 見ると D % 4 != 0 のときだけ拒否をすり抜け、f32 の ①QK と i8a8 の ③PV の混成で走って
    // しまう。「mask × i8a8 は無条件に fail loudly」が契約（ADR 0023 / docs/limitations.md）。
    if (mask !== undefined && i8a8) {
      throw new ExecutionError(
        `${where}: 加算 mask 付きの attention は attentionCompute 'i8a8' と組めない` +
          "（①QK の i8a8 変種は mask の epilogue を持たない — attentionCompute を 'f32' か " +
          "'f16' にすること）",
      );
    }

    // S[batch, M, N] と行統計 [batch·M, 2]。O は実行相が確保済みなので、峰は
    // O + S + 統計（分解経路の S + P + 恒等 expand の 3 枚から 1 枚ぶん減る）。
    // f16 変種（`:c16` の array<f16> / s16 の pack2x16float）では S が半分のバイト数になる
    // （1024px の DiT で 1,073.7MB → 536.9MB）。
    const scores = builder.allocTemp(
      batch * rows * cols * (compute === "f16" ? 2 : scoreStorageBytes(scoreStorage)),
    );
    const stats = builder.allocTemp(batch * rows * ATTENTION_STATS_STRIDE * 4);

    // ① QK gemm — 縮約次元は D、出力の列は N。
    if (qkI8a8) {
      await this.#buildAttentionQkI8a8(
        builder,
        binds,
        scores,
        scoreStorage,
        { batch, rows, cols, depth },
        scale,
        where,
      );
    } else {
      const qkV4 = gemmUsesVec4(depth, cols);
      const hasMask = mask !== undefined;
      const qkKey = attentionQkKey(qkV4, compute, scoreStorage, hasMask);
      const { pipeline: qkPipeline, layout: qkLayout } = await this.#state.cache.get(
        qkKey,
        attentionQkWgsl(qkV4, compute, scoreStorage, hasMask),
      );
      const geometry = defaultGemmGeometry();
      builder.dispatch({
        key: qkKey,
        pipeline: qkPipeline,
        layout: qkLayout,
        params: this.#writeParams(
          attentionQkParams(rows, cols, depth, scale),
          PARAMS_UNIFORM_USAGE,
        ),
        bindings: [
          { binding: 1, source: binds[0] },
          { binding: 2, source: binds[1] },
          { binding: 3, source: scores },
          ...(mask === undefined ? [] : [{ binding: ATTENTION_QK_MASK_BINDING, source: mask }]),
        ],
        workgroups: [
          tiledWorkgroups(cols, gemmTileN(geometry), limit, `${where} ①QK`),
          tiledWorkgroups(rows, gemmTileM(geometry), limit, `${where} ①QK`),
          tiledWorkgroups(batch, 1, limit, `${where} ①QK`),
        ],
      });
    }

    // ② 行統計 — 1 行 = 1 workgroup で、行方向は grid-stride（softmax と同じ形）。
    // S を 1 回だけ読む regcache 変種は dim 依存の生成なので、`epc` がキー軸に増える
    // （値はどちらもビット同一 — src/kernels/attention.ts）。
    const statsRegCache = attentionStatsRegCache(cols);
    const statsKey = attentionStatsKey(compute, scoreStorage, statsRegCache);
    const { pipeline: statsPipeline, layout: statsLayout } = await this.#state.cache.get(
      statsKey,
      attentionStatsWgsl(compute, scoreStorage, statsRegCache),
    );
    builder.dispatch({
      key: statsKey,
      pipeline: statsPipeline,
      layout: statsLayout,
      params: this.#writeParams(
        attentionStatsParams(batch * rows, cols, statsRegCache),
        PARAMS_UNIFORM_USAGE,
      ),
      bindings: [{ binding: 1, source: scores }, { binding: 2, source: stats }],
      workgroups: [gridStrideWorkgroups(batch * rows, 1, limit), 1, 1],
    });

    // ③ PV gemm — 縮約次元は N、出力の列は D。
    if (pvI8a8) {
      await this.#buildAttentionPvI8a8(
        builder,
        binds,
        scores,
        scoreStorage,
        stats,
        out,
        { batch, rows, cols, depth },
        where,
      );
    } else {
      const pvV4 = gemmUsesVec4(cols, depth);
      const pvKey = attentionPvKey(pvV4, compute, scoreStorage);
      const { pipeline: pvPipeline, layout: pvLayout } = await this.#state.cache.get(
        pvKey,
        attentionPvWgsl(pvV4, compute, scoreStorage),
      );
      const geometry = defaultGemmGeometry();
      builder.dispatch({
        key: pvKey,
        pipeline: pvPipeline,
        layout: pvLayout,
        params: this.#writeParams(attentionPvParams(rows, depth, cols), PARAMS_UNIFORM_USAGE),
        bindings: [
          { binding: 1, source: scores },
          { binding: 2, source: binds[2] },
          { binding: 3, source: stats },
          { binding: 4, source: out },
        ],
        workgroups: [
          tiledWorkgroups(depth, gemmTileN(geometry), limit, `${where} ③PV`),
          tiledWorkgroups(rows, gemmTileM(geometry), limit, `${where} ③PV`),
          tiledWorkgroups(batch, 1, limit, `${where} ③PV`),
        ],
      });
    }

    // MUST: ノード境界で一時バッファを返す（アリーナの不変条件 — 抜けると assertDrained で
    // 落ちるか、プール再利用から外れて peak が過大に出る）。
    builder.releaseTemp(stats);
    builder.releaseTemp(scores);
  }

  /**
   * 融合 attention ①QK の **i8a8 変種**（opt-in — {@link SessionOptions.attentionCompute}）。
   * ① が 1 dispatch から **3 dispatch** に増える（ノード全体では 3 → 5）:
   *
   * (a) `quantize_rows`（q を per-token i8 へ）→ (b) `quantize_rows`（k を per-token i8 へ）→
   * (c) i8a8 GEMM（整数内積 + dequant）。
   *
   * 量子化カーネルは linear の w8a8 と**同じ 1 本**（`QUANTIZE_ROWS_KEY` を共有する — 縮約軸
   * D が q / k とも最内連続なので、行 = token の per-token 量子化がそのまま要求どおりの形に
   * なる）。診断では linear の活性量子化と合算されるので、内訳は E2E のキー本数検査で担保する。
   *
   * MUST: 一時バッファ（`qq` / `qs` / `kq` / `ks`）は宣言 → 末尾で解放する。
   * MUST: `D` のオーバフロー門は fail loudly（黙って通すと i32 の巻き戻りで符号ごと化ける。
   * 実測形の D ≤ 384 に対して門は桁で余裕があるが、置かないと退行の受け皿が消える）。
   */
  async #buildAttentionQkI8a8(
    builder: StepRecipeBuilder,
    binds: readonly BindingSource[],
    scores: TempSource,
    scoreStorage: ScoreStorage,
    shape: {
      readonly batch: number;
      readonly rows: number;
      readonly cols: number;
      readonly depth: number;
    },
    scale: number,
    where: string,
  ): Promise<void> {
    const { batch, rows, cols, depth } = shape;
    if (depth > LINEAR_I8A8_MAX_K) {
      throw new ExecutionError(
        `${where}: D=${depth} が i8a8 経路の i32 縮約の門 ${LINEAR_I8A8_MAX_K} を超える` +
          "（attentionCompute を 'f32' にすること）",
      );
    }
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;

    // 量子化した q / k（i8 を 4 詰め）と per-token scale。ノード内で閉じた一時領域で、
    // q 側は**行** scale・k 側は**出力列**の scale になる（同じ per-token 量子化の別の読み方）。
    const qq = builder.allocTemp(Math.max(4, batch * rows * depth));
    const qs = builder.allocTemp(Math.max(4, batch * rows * 4));
    const kq = builder.allocTemp(Math.max(4, batch * cols * depth));
    const ks = builder.allocTemp(Math.max(4, batch * cols * 4));

    // (a)(b) 活性の per-token 量子化（1 行 = 1 workgroup・行方向 grid-stride）
    const { pipeline: quantizePipeline, layout: quantizeLayout } = await this.#state.cache.get(
      QUANTIZE_ROWS_KEY,
      QUANTIZE_ROWS_WGSL,
    );
    const quantize = (
      source: BindingSource,
      payload: TempSource,
      scales: TempSource,
      count: number,
    ): void => {
      builder.dispatch({
        key: QUANTIZE_ROWS_KEY,
        pipeline: quantizePipeline,
        layout: quantizeLayout,
        params: this.#writeParams(quantizeRowsParams(count, depth), PARAMS_UNIFORM_USAGE),
        bindings: [
          { binding: 1, source },
          { binding: 2, source: payload },
          { binding: 3, source: scales },
        ],
        workgroups: [gridStrideWorkgroups(count, 1, limit), 1, 1],
      });
    };
    quantize(binds[0], qq, qs, batch * rows);
    quantize(binds[1], kq, ks, batch * cols);

    // (c) 整数内積の GEMM（半スケールは dequant 側で q / k の両方へ — 設計 §2.1）。
    // 幾何は ③PV と**別に**選ぶ（③ だけ N = D の 1 タイル化が勝つ — 実測）。
    const v4 = attentionQkI8a8UsesVec4(cols);
    const dp4a = this.#state.i8a8Dot === "dp4a";
    const geometry = defaultI8a8Geometry("attention_qk");
    const key = attentionQkI8a8Key(v4, dp4a, scoreStorage, geometry);
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      attentionQkI8a8Wgsl(v4, dp4a, scoreStorage, geometry),
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params: this.#writeParams(
        attentionQkI8a8Params(rows, cols, depth, scale),
        PARAMS_UNIFORM_USAGE,
      ),
      bindings: [
        { binding: 1, source: qq },
        { binding: 2, source: kq },
        { binding: 3, source: scores },
        { binding: ATTENTION_QK_Q_SCALE_BINDING, source: qs },
        { binding: ATTENTION_QK_K_SCALE_BINDING, source: ks },
      ],
      workgroups: [
        tiledWorkgroups(cols, i8a8TileN(geometry), limit, `${where} ①QK i8a8`),
        tiledWorkgroups(rows, i8a8TileM(geometry), limit, `${where} ①QK i8a8`),
        tiledWorkgroups(batch, 1, limit, `${where} ①QK i8a8`),
      ],
    });

    // MUST: ノード境界で一時バッファを返す（確保と破棄を 1 箇所へ — ADR 0004）。
    builder.releaseTemp(ks);
    builder.releaseTemp(kq);
    builder.releaseTemp(qs);
    builder.releaseTemp(qq);
  }

  /**
   * 融合 attention ③PV の **i8a8 変種**（opt-in — {@link SessionOptions.attentionCompute}）。
   * ③ が 1 dispatch から **3 dispatch** に増える（①も i8a8 ならノード全体で 3 → 7）:
   *
   * (a) `strided`（v`[B·H,N,D]` → Vᵀ`[B·H,D,N]` の permute）→ (b) `quantize_rows`
   * （Vᵀ を行 = `(b,h,d)` で量子化）→ (c) i8a8 GEMM（整数内積 + dequant）。
   *
   * **新カーネルを 1 本も作らずに per-column 量子化が得られる**のがこの並びの要点（設計 §2.3）:
   * Vᵀ の「行」は `(b,h,d)` なので `quantize_rows` の per-token 量子化がそのまま
   * **V の per-column（N 全体の amax）scale** になり、同時に dp4a が要求する **N 連続パック**も
   * 手に入る。MUST: V を転置せずに量子化してはならない — scale が縮約軸 n の上で変わり、
   * `f32(acc)·s` 形の前提（s が n に依存しない）が壊れる（例外の出ない誤値）。
   *
   * P̃ 側は量子化カーネルを通らない（A タイル充填が `round(127·exp(S−m))` を作る）ので、
   * dispatch も一時バッファも増えない — ②行統計は f32 のまま 1 バイトも変えない。
   *
   * MUST: 一時バッファ（`vt` / `vq` / `vs`）は宣言 → 末尾で解放する。
   * MUST: `N` のオーバフロー門は fail loudly（|acc| ≤ N·127²。実測形の最大 N = 16,384 に対し
   * 門は桁で余裕があるが、置かないと退行の受け皿が消える）。
   */
  async #buildAttentionPvI8a8(
    builder: StepRecipeBuilder,
    binds: readonly BindingSource[],
    scores: TempSource,
    scoreStorage: ScoreStorage,
    stats: TempSource,
    out: BindingSource,
    shape: {
      readonly batch: number;
      readonly rows: number;
      readonly cols: number;
      readonly depth: number;
    },
    where: string,
  ): Promise<void> {
    const { batch, rows, cols, depth } = shape;
    if (cols > LINEAR_I8A8_MAX_K) {
      throw new ExecutionError(
        `${where}: N=${cols} が i8a8 経路の i32 縮約の門 ${LINEAR_I8A8_MAX_K} を超える` +
          "（attentionCompute を 'f32' にすること）",
      );
    }
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;

    // Vᵀ（f32・permute の実体化）と、その量子化結果。どれもノード内で閉じた一時領域。
    const vt = builder.allocTemp(Math.max(4, batch * depth * cols * 4));
    const vq = builder.allocTemp(Math.max(4, batch * depth * cols));
    const vs = builder.allocTemp(Math.max(4, batch * depth * 4));

    // (a) v[B·H,N,D] → Vᵀ[B·H,D,N]（既存の strided 読みコピー族 — permute そのもの）。
    // MUST: stride は**入力** `[B·H,N,D]` の連続 stride から組む（出力 shape から組むと
    // D == N のときだけ一致する。実測形は D != N なので露見するが、単体テストが本来の検出器）。
    const stridedSpec = { dtype: "f32" } as const;
    const permuteKey = stridedKey(stridedSpec);
    const { pipeline: permutePipeline, layout: permuteLayout } = await this.#state.cache.get(
      permuteKey,
      stridedWgsl(stridedSpec),
    );
    builder.dispatch({
      key: permuteKey,
      pipeline: permutePipeline,
      layout: permuteLayout,
      params: this.#writeParams(
        stridedParams(
          [batch, depth, cols],
          permuteSrcStrides([batch, cols, depth], [0, 2, 1]),
          0,
        ),
        PARAMS_STORAGE_USAGE,
      ),
      bindings: [{ binding: 1, source: binds[2] }, { binding: 2, source: vt }],
      workgroups: [
        gridStrideWorkgroups(batch * depth * cols, STRIDED_WORKGROUP_SIZE, limit),
        1,
        1,
      ],
    });

    // (b) Vᵀ の量子化（行 = (b,h,d)・行長 N — per-column scale と N 連続パックが同時に出る）
    const { pipeline: quantizePipeline, layout: quantizeLayout } = await this.#state.cache.get(
      QUANTIZE_ROWS_KEY,
      QUANTIZE_ROWS_WGSL,
    );
    builder.dispatch({
      key: QUANTIZE_ROWS_KEY,
      pipeline: quantizePipeline,
      layout: quantizeLayout,
      params: this.#writeParams(quantizeRowsParams(batch * depth, cols), PARAMS_UNIFORM_USAGE),
      bindings: [
        { binding: 1, source: vt },
        { binding: 2, source: vq },
        { binding: 3, source: vs },
      ],
      workgroups: [gridStrideWorkgroups(batch * depth, 1, limit), 1, 1],
    });

    // (c) 整数内積の GEMM（P̃ は A タイル充填で作る = 非実体化のまま）
    const v4 = attentionPvI8a8UsesVec4(depth);
    const dp4a = this.#state.i8a8Dot === "dp4a";
    const geometry = defaultI8a8Geometry("attention_pv");
    const key = attentionPvI8a8Key(v4, dp4a, scoreStorage, geometry);
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      attentionPvI8a8Wgsl(v4, dp4a, scoreStorage, geometry),
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params: this.#writeParams(attentionPvI8a8Params(rows, depth, cols), PARAMS_UNIFORM_USAGE),
      bindings: [
        { binding: 1, source: scores },
        { binding: 2, source: vq },
        { binding: 3, source: stats },
        { binding: 4, source: out },
        { binding: ATTENTION_PV_V_SCALE_BINDING, source: vs },
      ],
      workgroups: [
        tiledWorkgroups(depth, i8a8TileN(geometry), limit, `${where} ③PV i8a8`),
        tiledWorkgroups(rows, i8a8TileM(geometry), limit, `${where} ③PV i8a8`),
        tiledWorkgroups(batch, 1, limit, `${where} ③PV i8a8`),
      ],
    });

    // MUST: ノード境界で一時バッファを返す（確保と破棄を 1 箇所へ — ADR 0004）。
    builder.releaseTemp(vs);
    builder.releaseTemp(vq);
    builder.releaseTemp(vt);
  }

  /**
   * embedding（行 gather）。範囲外添字の扱いは src/kernels/embedding.ts の裁定
   * （GPU は NaN 汚染 / CPU 参照は throw）。attrs の padding_idx は forward に効かないので
   * カーネルへ渡さない。
   */
  async #buildEmbedding(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const weight = step.inputShapes[0];
    const count = numel(step.outputShape);
    const weightStorage = this.#weightStorage(step);
    const key = embeddingKey(weightStorage);
    const { pipeline, layout } = await this.#state.cache.get(key, embeddingWgsl(weightStorage));
    const params = this.#writeParams(
      embeddingParams(count, weight[1], weight[0]),
      PARAMS_UNIFORM_USAGE,
    );
    const groups = gridStrideWorkgroups(
      count,
      EMBEDDING_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: binds[1] },
        { binding: 3, source: out },
        ...this.#weightScaleBindings(step, weightStorage, EMBEDDING_SCALE_BINDING),
      ],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * masked_fill。出力と x は同形・連続で、mask だけを右詰め broadcast の stride で読む。
   * stride の組み立ては strided 族の expand と同じ規則（{@link expandSrcStrides}）。
   */
  async #buildMaskedFill(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const outShape = step.outputShape;
    // 規則は expand と同一だが、診断の主語は masked_fill の側に付け替える（グラフに expand が
    // 無いのに「expand の入力」と出ると原因の当たりを外す）。
    const maskStrides = expandSrcStrides(step.inputShapes[1], outShape, {
      src: "masked_fill の mask",
      out: "masked_fill の出力",
    });
    const value = maskedFillValue(step.node.attrs, `nodes (${step.node.op})`);
    const { pipeline, layout } = await this.#state.cache.get(
      MASKED_FILL_KEY,
      MASKED_FILL_WGSL,
    );
    const params = this.#writeParams(
      maskedFillParams(outShape, maskStrides, value),
      PARAMS_STORAGE_USAGE,
    );
    const groups = gridStrideWorkgroups(
      numel(outShape),
      MASKED_FILL_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: MASKED_FILL_KEY,
      pipeline,
      layout,
      params,
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: binds[1] },
        { binding: 3, source: out },
      ],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * conv1d（groups / dilation は attrs）。**groups で 2 カーネルを踏み分ける**（conv2d と同型）:
   * `groups == 1` は implicit GEMM、`groups > 1` は直接畳み込み（1 スレッド = 1 出力要素の
   * grid-stride）。
   */
  async #buildConv1d(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const [x, weight] = step.inputShapes;
    const outShape = step.outputShape;
    const { stride, padding, dilation, groups } = conv1dAttrs(
      step.node.attrs,
      `nodes (${step.node.op})`,
    );
    const dims: Conv1dDims = {
      batch: outShape[0],
      channelsIn: x[1],
      channelsOut: outShape[1],
      lengthIn: x[2],
      lengthOut: outShape[2],
      kernel: weight[2],
      stride,
      padding,
      dilation,
      groups,
    };
    const weightStorage = this.#weightStorage(step);
    if (groups === 1) {
      await this.#buildConv1dIgemm(step, binds, out, builder, dims, weightStorage);
      return;
    }
    const key = conv1dKey(weightStorage);
    const { pipeline, layout } = await this.#state.cache.get(key, conv1dWgsl(weightStorage));
    const params = this.#writeParams(conv1dParams(dims), PARAMS_UNIFORM_USAGE);
    const workgroups = gridStrideWorkgroups(
      numel(outShape),
      CONV1D_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 4, source: out },
        ...this.#weightScaleBindings(step, weightStorage, CONV1D_SCALE_BINDING),
      ],
      workgroups: [workgroups, 1, 1],
    });
  }

  /**
   * conv1d の implicit GEMM（`C[Cout, N] = W[Cout, K] × Xcol[K, N]` — ADR 0024 の 1D 版）。
   *
   * MUST: GEMM 骨格と同じ「1 workgroup = 1 出力タイル」なので、dispatch 上限超過は
   * fail loudly（grid-stride で縮退させるとタイルが欠落し、full-write が黙って壊れる）。
   * MUST: バッチは **z 軸**（bmm / conv2d と同じ）。N 側へ畳むと出力が `[Cout][B·Lout]` に
   * なって NCL と軸が入れ替わる — B = 1 でだけ一致するので実測形では露見しない。
   *
   * m タイルの述語は conv2d と**同じ 1 本**（{@link conv2dIgemmMTile} — M = Cout の関数で
   * しかないので次元に依らない）。どちらのタイル形でも出力はビット同一なので、これは純粋な
   * dispatch の割り直しで数値契約に触れない。
   */
  async #buildConv1dIgemm(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
    dims: Conv1dDims,
    weightStorage: WeightStorage,
  ): Promise<void> {
    const m = dims.channelsOut;
    const kFlat = dims.channelsIn * dims.kernel;
    const v4 = conv1dUsesVec4(kFlat, dims.lengthOut, dims.stride);
    const mTile = conv2dIgemmMTile(m);
    // 生成・キーと同じ解決点から幾何を引く（dispatch のタイル辺が WGSL の辺と構造的に一致）。
    const geometry = gemmMTileGeometry(mTile);
    const key = conv1dIgemmKey(weightStorage, v4, mTile);
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      conv1dIgemmWgsl(weightStorage, v4, mTile),
    );
    const params = this.#writeParams(conv1dIgemmParams(dims), PARAMS_UNIFORM_USAGE);
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const where = `conv1d [${step.inputShapes[0].join(",")}] * [${step.inputShapes[1].join(",")}]`;
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 4, source: out },
        ...this.#weightScaleBindings(step, weightStorage, CONV1D_SCALE_BINDING),
      ],
      workgroups: [
        tiledWorkgroups(dims.lengthOut, gemmTileN(geometry), limit, where),
        tiledWorkgroups(m, gemmTileM(geometry), limit, where),
        tiledWorkgroups(dims.batch, 1, limit, where),
      ],
    });
  }

  /**
   * conv2d（stride / padding / dilation は H/W の 2 成分）。**groups で 2 カーネルを踏み分ける**
   * （ADR 0024）: `groups == 1` は implicit GEMM、`groups > 1` は直接畳み込み。
   *
   * MUST: Kh / Kw は**重みの第 3 / 第 4 軸**をこの順で読む。入れ替えても正方カーネルでは
   * 数値が一致するので、テストは Kh ≠ Kw の形で固定する（src/kernels/conv2d.ts）。
   */
  async #buildConv2d(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const [x, weight] = step.inputShapes;
    const outShape = step.outputShape;
    const { stride, padding, dilation, groups } = conv2dAttrs(
      step.node.attrs,
      `nodes (${step.node.op})`,
    );
    const dims: Conv2dDims = {
      batch: outShape[0],
      channelsIn: x[1],
      channelsOut: outShape[1],
      heightIn: x[2],
      widthIn: x[3],
      heightOut: outShape[2],
      widthOut: outShape[3],
      kernelH: weight[2],
      kernelW: weight[3],
      strideH: stride[0],
      strideW: stride[1],
      paddingH: padding[0],
      paddingW: padding[1],
      dilationH: dilation[0],
      dilationW: dilation[1],
      groups,
    };
    const weightStorage = this.#weightStorage(step);
    if (groups === 1) {
      await this.#buildConv2dIgemm(step, binds, out, builder, dims, weightStorage);
      return;
    }
    const key = conv2dKey(weightStorage);
    const { pipeline, layout } = await this.#state.cache.get(key, conv2dWgsl(weightStorage));
    const params = this.#writeParams(conv2dParams(dims), PARAMS_UNIFORM_USAGE);
    const workgroups = gridStrideWorkgroups(
      numel(outShape),
      CONV2D_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 4, source: out },
        ...this.#weightScaleBindings(step, weightStorage, CONV2D_SCALE_BINDING),
      ],
      workgroups: [workgroups, 1, 1],
    });
  }

  /**
   * conv2d の implicit GEMM（`C[Cout, N] = W[Cout, K] × Xcol[K, N]` — ADR 0024）。
   *
   * MUST: GEMM 骨格と同じ「1 workgroup = 1 出力タイル」なので、dispatch 上限超過は
   * fail loudly（grid-stride で縮退させるとタイルが欠落し、full-write が黙って壊れる）。
   * MUST: バッチは **z 軸**（bmm と同じ）。N 側へ畳むと出力が `[Cout][B·Hout·Wout]` になり
   * NCHW と軸が入れ替わる — B = 1 でだけ一致するので実測形では露見しない。
   *
   * m タイルは形状の関数（{@link conv2dIgemmMTile}）で 64 行 / 32 行を選ぶ（ADR 0024 隣接）。
   * **n タイルは常に 64**（2048px の n 上限超過の扱いを動かさない）。どちらのタイル形でも
   * 出力はビット同一なので、これは純粋な dispatch の割り直しで数値契約に触れない。
   */
  async #buildConv2dIgemm(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
    dims: Conv2dDims,
    weightStorage: WeightStorage,
  ): Promise<void> {
    const m = dims.channelsOut;
    const n = dims.heightOut * dims.widthOut;
    const kFlat = dims.channelsIn * dims.kernelH * dims.kernelW;
    const v4 = conv2dUsesVec4(kFlat, dims.widthOut, dims.strideW);
    const mTile = conv2dIgemmMTile(m);
    // 生成・キーと同じ解決点から幾何を引く（dispatch のタイル辺が WGSL の辺と構造的に一致）。
    const geometry = gemmMTileGeometry(mTile);
    const key = conv2dIgemmKey(weightStorage, v4, mTile);
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      conv2dIgemmWgsl(weightStorage, v4, mTile),
    );
    const params = this.#writeParams(conv2dIgemmParams(dims), PARAMS_UNIFORM_USAGE);
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const where = `conv2d [${step.inputShapes[0].join(",")}] * [${step.inputShapes[1].join(",")}]`;
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 4, source: out },
        ...this.#weightScaleBindings(step, weightStorage, CONV2D_SCALE_BINDING),
      ],
      workgroups: [
        tiledWorkgroups(n, gemmTileN(geometry), limit, where),
        tiledWorkgroups(m, gemmTileM(geometry), limit, where),
        tiledWorkgroups(dims.batch, 1, limit, where),
      ],
    });
  }

  /**
   * conv_transpose1d（gather 形）。1 スレッド = 1 出力要素で出力全域を書く（ADR 0014）。
   *
   * MUST: Cin は**重みの第 1 軸**（`[Cin, Cout, K]`）。x[1] と一致することは契約検査済みだが、
   * ここで weight[0] を使うのは「転置レイアウトの正本は重み側」という読みを崩さないため。
   */
  async #buildConvTranspose1d(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const [x, weight] = step.inputShapes;
    const outShape = step.outputShape;
    const { stride, padding } = convTranspose1dAttrs(step.node.attrs, `nodes (${step.node.op})`);
    const weightStorage = this.#weightStorage(step);
    const key = convTranspose1dKey(weightStorage);
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      convTranspose1dWgsl(weightStorage),
    );
    const params = this.#writeParams(
      convTranspose1dParams({
        batch: outShape[0],
        channelsIn: weight[0],
        channelsOut: outShape[1],
        lengthIn: x[2],
        lengthOut: outShape[2],
        kernel: weight[2],
        stride,
        padding,
      }),
      PARAMS_UNIFORM_USAGE,
    );
    const workgroups = gridStrideWorkgroups(
      numel(outShape),
      CONV_TRANSPOSE1D_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 4, source: out },
        ...this.#weightScaleBindings(step, weightStorage, CONV_TRANSPOSE1D_SCALE_BINDING),
      ],
      workgroups: [workgroups, 1, 1],
    });
  }

  /**
   * このノードの重みスロットの格納形（カーネル変種の選択 — ADR 0018）。
   *
   * MUST: スロット位置は適格判定と**同じ表**（{@link WEIGHT_SLOTS}）から引く。片方だけ
   * ずれると、圧縮のまま上げた重みを f32 カーネルが読む（あるいはその逆）ビット列の
   * 読み替えになり、例外は 1 つも出ない。
   */
  #weightStorage(step: NodePlan): WeightStorage {
    const slot = WEIGHT_SLOTS.get(step.node.op);
    if (slot === undefined) {
      throw new ExecutionError(`op '${step.node.op}' に重みスロットの定義が無い`);
    }
    return this.#state.weightStorages.get(step.node.ins[slot]) ?? "f32";
  }

  /**
   * i8 変種の追加束縛（per-channel scale）。f32 / f16 では空配列になり、bind group は従来の
   * ままになる（ADR 0019）。
   *
   * MUST: 束縛番号はカーネル側の定数（`*_SCALE_BINDING`）から引く。WGSL の宣言と executor が
   * 別々に番号を持つと、変種を足したときに片方だけずれる。
   */
  #weightScaleBindings(
    step: NodePlan,
    storage: WeightStorage,
    binding: number,
  ): readonly BindingRecipe[] {
    if (storage !== "i8") return [];
    const slot = WEIGHT_SLOTS.get(step.node.op);
    if (slot === undefined) {
      throw new ExecutionError(`op '${step.node.op}' に重みスロットの定義が無い`);
    }
    const name = step.node.ins[slot];
    const buffer = this.#state.weightScaleBuffers.get(name);
    if (buffer === undefined) {
      throw new ExecutionError(`initializer '${name}': i8 常駐なのに scale バッファが無い`);
    }
    return [{ binding, source: { kind: "resident", buffer } }];
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
   * ので追い越しハザードが原理的に生じない（プール再利用とは別レイヤの、内容同一性による
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
