/**
 * 融合パス（純関数）: 計画済みノード列（plan.ts の {@link NodePlan}）→ 実行ステップ列。
 *
 * IR の公開語彙は 1 つも増やさない。エクスポータが出す**隣接ノードの決まった並び**を、実行時
 * にだけ private カーネル 1 本へ畳む peephole で、掴めなかった形は素のノード列にそのまま
 * 落ちる（fallback は常に正しい既存経路）。GPU に触れないのでアダプタ無し環境で検証できる。
 *
 * ## 設計の約束
 *
 * - MUST: matcher を executor の走査ループへ直書きしない。ここが唯一の判定点で、反例
 *   （use-count 2 / graph output / near-shape / dtype 違い / 順序違い）を GPU 無しで網羅できる。
 * - MUST: 融合ステップは外部入力の**延べ列**（{@link FusedStep.ins}）を宣言し、解放簿記は
 *   executor の既存経路（素のノードと同一）に合流させる。融合ごとに手書きの retain/release を
 *   置くと、アリーナの参照計数が融合の本数だけ別実装になり、1 本ずれても例外は出ずに
 *   沈黙誤値になる。
 * - MUST: 融合は演算列を潰すが**値は変えない**。丸め位置の保存はカーネル側の責務で、その
 *   手段（workgroup memory 往復による丸め障壁）が仕様保証でないことは各カーネルの docstring に
 *   書いてある。
 *
 * ## 畳む先は 1 dispatch とは限らない
 *
 * 4 ルール（silu / upsample2x / rope / adaln）は「N ノード → private カーネル 1 dispatch」だが、
 * {@link ROW_BLOCK_ATTENTION_RULE} は**演算ではなく中間の実体化幅**を畳むので、ステップ内で
 * 閉じた一時（{@link FusedStep.temps}）を挟んだ dispatch 列になる。どちらも
 * {@link FusedStep} 1 つ = 実行ステップ 1 つで、解放簿記の合流点は変わらない。
 *
 * ## 適用順
 *
 * {@link FUSION_RULES} の**宣言順**（silu → upsample2x → rope → adaln → rowBlockAttention）。
 * 5 ルールの先頭 op は `sigmoid` / `reshape` / `mul|slice` / `layer_norm` / `bmm` で互いに素
 * なので、この順序は結果に効かない（順序が意味を持つのは先頭 op が重なったときだけ —
 * 重なりが生じていないことは tests/runtime_fusion_test.ts が {@link FusionRule.heads} から
 * 機械的に検査する）。窓の**内側**に他ルールの先頭 op が現れる形（rowBlockAttention の窓は
 * `reshape` / `expand` を 5 本含む）は、掴めた時点で走査が窓幅ぶん進むので発火しえない。
 *
 * ## 窓内 passthrough
 *
 * 鎖は**隣接しているとは限らない**。adaLN は `layer_norm` と `mul` の間にエクスポータが
 * 変調ベクトルの `reshape` を 2〜3 本挟む（実測 85 鎖すべて）。RoPE は `cat` と続く `mul` の
 * 間に cos / sin 表の `sym_prefix_slice` が挟まる形が**表の初出 1 箇所だけ**にある。
 * ルールは連続窓
 * （{@link FusionMatch.window}）を宣言し、そのうち畳むノード（`chain`）以外を
 * **passthrough** として融合ステップの**前**に素のノードのまま並べる。並べ替えが合法なのは
 * passthrough が鎖の定義する値を 1 つも消費しない場合だけで、そこは
 * {@link passthroughIsIndependent} が機械的に見る。
 */

import {
  catDim,
  LAYER_NORM_OP,
  layerNormAttrs,
  numel,
  SAFE_SOFTMAX_OP,
  sliceAttrs,
  softmaxDim,
  SYM_PREFIX_SLICE_OP,
} from "../ops.ts";
import { tiledWorkgroups } from "../codegen/dispatch.ts";
import {
  ELEMENTWISE_WORKGROUP_SIZE,
  elementwiseKey,
  elementwiseParams,
  type ElementwiseSpec,
  elementwiseWgsl,
} from "../codegen/elementwise.ts";
import { ADALN_NORM_KEY, ADALN_NORM_WGSL, adalnNormParams } from "../kernels/adaln-norm.ts";
import { bmmKey, bmmParams, bmmRowWindowParams, bmmWgsl } from "../kernels/bmm.ts";
import { gemmUsesVec4 } from "../kernels/gemm.ts";
import { gemmGeometryForRows, gemmTileM, gemmTileN } from "../kernels/gemm-geometry.ts";
import { SAFE_SOFTMAX_KEY, SAFE_SOFTMAX_WGSL, softmaxParams } from "../kernels/softmax.ts";
import { ROPE_KEY, ROPE_WGSL, ROPE_WORKGROUP_SIZE, ropeParams } from "../kernels/rope.ts";
import {
  SILU_WORKGROUP_SIZE,
  siluKey,
  type SiluMulOrder,
  siluParams,
  siluWgsl,
} from "../kernels/silu.ts";
import {
  upsample2xParams,
  UPSAMPLE_2X_KEY,
  UPSAMPLE_2X_WGSL,
  UPSAMPLE_2X_WORKGROUP_SIZE,
} from "../kernels/upsample2x.ts";
import { ExecutionError, type NodePlan } from "./plan.ts";

/** 融合ルールの識別子（{@link FUSION_RULES} の宣言順と 1 対 1）。 */
type FusionRuleName = "silu" | "upsample2x" | "rope" | "adaln" | "rowBlockAttention";

/**
 * 診断カウンタの見出し。融合 4 ルールに加えて、0 dispatch の別名化のうち**条件付きで外れうる**
 * 恒等 expand（{@link ExecStep} の `aliasesInput`）を数える。reshape の別名化は無条件なので
 * 数えない（外れようがない = 観測する意味がない）。
 */
type FusionCounterName = FusionRuleName | "identityExpand";

/** ルール別の適用回数。「融合が黙って外れて性能だけ落ちる」事故の唯一の観測点。 */
export type FusionCounts = Readonly<Record<FusionCounterName, number>>;

/**
 * 融合ステップ内で束縛しうる実体。
 *
 * MUST: ステップ内一時（{@link FusedStep.temps}）を指せるのはここだけで、外部入力は
 * {@link FusedStep.binds} の**添字**で指す（値名の再解決を executor に持たせない）。
 */
export type FusedOperand =
  | { readonly kind: "bind"; readonly index: number }
  | { readonly kind: "temp"; readonly id: number }
  | { readonly kind: "output" };

/**
 * dispatch の workgroup 数の決め方。
 *
 * - `gridStride` = 要素 / 行の被覆数を割る形（elementwise・reduce・融合 4 ルール）。上限を
 *   超えたら縮退し、カーネル側の grid-stride が残りを回す。
 * - `tiled` = **1 workgroup = 1 出力タイル**の GEMM 族。grid-stride で縮退できないので、
 *   上限超過は宣言側（`tiledWorkgroups`）が fail loudly にする。
 */
type FusedWorkgroups =
  | { readonly kind: "gridStride"; readonly items: number; readonly size: number }
  | { readonly kind: "tiled"; readonly counts: readonly [number, number, number] };

/** 融合カーネル 1 dispatch ぶんの生成入力（全ルール共通の形）。 */
type FusedDispatch = {
  readonly key: string;
  /**
   * MUST: 同一キーには常にバイト単位で同一の WGSL（PipelineCache の決定性契約）。
   * thunk なのは、キャッシュ済みでも本文の突き合わせが必要な一方、計画時に全ステップぶんを
   * 文字列で持つ必要が無いため。
   */
  readonly wgsl: () => string;
  /** binding 0 の Params。既定は uniform（16 バイト整列の MUST は各カーネル側）。 */
  readonly params: Uint32Array<ArrayBuffer>;
  /**
   * params を **storage** で束ねる（可変長 params を持つ elementwise 族だけ — 素のノードの
   * `#buildElementwise` と同じ）。省略時は uniform。
   */
  readonly paramsStorage?: boolean;
  /**
   * binding 1 以降のオペランド列。**省略できるのは 1 dispatch のルールだけ**で、そのときは
   * 「{@link FusedStep.binds} を宣言順 → 末尾に出力」（融合 4 ルール共通の形）になる。
   */
  readonly operands?: readonly FusedOperand[];
  readonly workgroups: FusedWorkgroups;
};

/**
 * ステップ内一時の確保仕様。形も意味も recipe.ts の `TempRecipe` と同じで、**寿命は
 * dispatch 境界の添字**で表す。
 *
 * MUST: 宣言した一時には必ず解放境界がある（`releaseAfter` が確保より前だと実行相の参照
 * 計数が閉じず、run 末尾の `assertDrained` が落とす）。
 */
type FusedTemp = {
  readonly byteLength: number;
  /** {@link FusedStep.dispatches} のこの添字の**直前**に確保する。 */
  readonly allocBefore: number;
  /** {@link FusedStep.dispatches} のこの添字の**直後**に解放する。 */
  readonly releaseAfter: number;
};

/**
 * 融合ステップ（元の連続ノード列 1 本を private カーネルの dispatch 列へ置換したもの）。
 *
 * dispatch は**複数取りうる**。畳んだ結果が 1 dispatch にならないルール
 * （{@link ROW_BLOCK_ATTENTION_RULE}）は、ステップ内で閉じた一時（{@link FusedStep.temps}）を
 * 挟んで数本を並べる。MUST: それでも解放簿記の根拠は {@link FusedStep.ins} の延べ列 1 本の
 * ままで、ステップ境界の外へは一時が 1 本も漏れない。
 */
export type FusedStep = {
  readonly kind: "fused";
  readonly rule: FusionRuleName;
  /**
   * 元ノード列が消費した**外部**入力の延べ列（重複込み）。内部値は 1 本も実体化しないので
   * 含めない。MUST: 元の延べ回数と 1 つも違えてはならない（アリーナの参照計数の唯一の根拠）。
   */
  readonly ins: readonly string[];
  /** bind group のオペランド順（binding 1 から。重複無し）。 */
  readonly binds: readonly string[];
  /**
   * 畳んだノードの本数。**走査幅ではない** — 窓内 passthrough を持つルールでは
   * 窓幅（{@link FusionHit.advance}）の方が大きい。
   */
  readonly nodeCount: number;
  readonly outputName: string;
  readonly outputShape: readonly number[];
  /** ステップ内で閉じた一時（{@link FusedStep.dispatches} の添字で寿命を表す）。 */
  readonly temps: readonly FusedTemp[];
  readonly dispatches: readonly FusedDispatch[];
};

type NodeStep = {
  readonly kind: "node";
  readonly plan: NodePlan;
  /**
   * 出力を入力バッファの別名にする（0 dispatch — ADR 0011）。reshape は常に真、expand は
   * 束縛後の入出力 shape が rank を含め完全一致するとき（= 複製軸を持たない恒等写像）だけ真。
   */
  readonly aliasesInput: boolean;
};

export type ExecStep = NodeStep | FusedStep;

export type FusionPlan = {
  readonly steps: readonly ExecStep[];
  readonly counts: FusionCounts;
};

/**
 * 判定に要る device の能力（**granted limit の値そのもの**）。
 *
 * MUST: ここに載るのは「計画を決める入力」であって計測値ではない。実行時に測って選び直す形
 * （オートチューン）は ADR 0022 で禁じている — 同じ device・同じ束縛なら常に同じ計画が出る、
 * が prepared plan キャッシュとキーの意味の前提。
 */
export type FusionLimits = {
  /** ストレージ束縛 1 本の上限。行ブロック枚数を決める唯一の device 側入力。 */
  readonly maxStorageBufferBindingSize: number;
  /** 1 軸あたりの workgroup 数の上限（タイル型 dispatch の fail loudly 用）。 */
  readonly maxComputeWorkgroupsPerDimension: number;
};

/** 判定に要るグラフ全体の事実（executor の Session 状態から渡す）。 */
type FusionContext = {
  /** 値名 → グラフ内の消費回数（plan.ts の countUses）。 */
  readonly useCounts: ReadonlyMap<string, number>;
  readonly outputNames: ReadonlySet<string>;
  readonly limits: FusionLimits;
  /**
   * 行ブロック枚数の強制（**テスト専用** — executor の `ROW_BLOCK_SPLIT`）。上限に収まる
   * 最小枚数の代わりにこの枚数で割る。上限に収まらない枚数は fail loudly。
   */
  readonly rowBlockSplit?: number;
};

const sameShape = (a: readonly number[], b: readonly number[]): boolean =>
  a.length === b.length && a.every((dim, index) => dim === b[index]);

/** 鎖の全ノードが f32 専業か（融合カーネルは全て f32 固定）。 */
const allF32 = (chain: readonly NodePlan[]): boolean =>
  chain.every((step) =>
    step.outputDtype === "f32" && step.inputDtypes.every((dtype) => dtype === "f32")
  );

/**
 * 鎖の内部値（最終ノード以外の出力）が全て「消費者ちょうど 1 本・graph output でない」か。
 *
 * MUST: 融合後は内部値のバッファを 1 本も作らないので、外部 consumer や readback が 1 つでも
 * あれば値が消える。ここが全ルール共通の適格条件。
 */
const internalsArePrivate = (chain: readonly NodePlan[], context: FusionContext): boolean =>
  chain.slice(0, -1).every((step) =>
    (context.useCounts.get(step.outputName) ?? 0) === 1 &&
    !context.outputNames.has(step.outputName)
  );

/**
 * 鎖が消費した外部入力の延べ列（重複込み・元の node.ins の並び順）。
 * 内部値は畳まれて消えるので除く。
 */
const externalIns = (chain: readonly NodePlan[]): readonly string[] => {
  const internal = new Set(chain.slice(0, -1).map((step) => step.outputName));
  return chain.flatMap((step) => step.node.ins.filter((name) => !internal.has(name)));
};

/**
 * 窓内 passthrough を融合ステップより**前**へ動かしてよいか。
 *
 * MUST: passthrough は鎖が定義する値（内部値も**最終出力も**）を 1 つも消費してはならない。
 * 消費していれば「まだ計算されていない値を読むノード」を先に置くことになり、順序の
 * 入れ替えが非合法になる（executor は steps を順に encode するだけなので、ここで弾かないと
 * `値 'x' のバッファが無い` か、名前が使い回されていれば沈黙誤値になる）。
 *
 * NOTE: 窓内 passthrough を持つ現行 2 ルール（adaln / rope）に限れば、passthrough が読める
 * 鎖の値は**内部値だけ**（最終出力は passthrough より後に定義されるので読めない）で、それは
 * {@link internalsArePrivate}（consumer ちょうど 1 本）が先に落とす — **反例を
 * 単独では構成できない**（tests/runtime_fusion_test.ts のフォールト注入で確認済み）。
 * ここが独立に効くのは「鎖の**最終**ノードが passthrough より前に来る」窓を持つ将来の
 * ルールで、internalsArePrivate は最終出力を見ないので代替にならない。窓の仕組み側の
 * 不変条件として `defineRule` に置く。
 */
const passthroughIsIndependent = (
  chain: readonly NodePlan[],
  passthrough: readonly NodePlan[],
): boolean => {
  if (passthrough.length === 0) return true;
  const defined = new Set(chain.map((step) => step.outputName));
  return passthrough.every((step) => step.node.ins.every((name) => !defined.has(name)));
};

/** 融合ルール 1 件の適用結果（融合ステップ + 窓内 passthrough + 走査幅）。 */
type FusionHit = {
  /** 融合ステップより**前**に素のまま並べる窓内ノード（元のノード順）。 */
  readonly passthrough: readonly NodePlan[];
  readonly step: FusedStep;
  /** 走査を進める幅（= 窓のノード数 = passthrough + 畳んだ鎖）。 */
  readonly advance: number;
};

/** ルールの本体（match で掴み、build で宣言的にステップへ落とす）。 */
type FusionRule = {
  readonly name: FusionRuleName;
  /** このルールが掴みうる先頭 op（適用順の互いに素性を機械検査するための宣言）。 */
  readonly heads: readonly string[];
  readonly apply: (
    nodes: readonly NodePlan[],
    index: number,
    context: FusionContext,
  ) => FusionHit | undefined;
};

/**
 * ルールが掴んだ窓。`window` は連続ノード列、`chain` はそのうち**畳む**部分列
 * （実際のノード順）で、差分が窓内 passthrough になる。
 */
type FusionMatch = {
  /** 連続窓（走査幅の唯一の根拠）。 */
  readonly window: readonly NodePlan[];
  /** 畳むノード列（`window` の部分列）。 */
  readonly chain: readonly NodePlan[];
};

/**
 * ステップ内一時の寿命宣言が dispatch 列の内側で閉じているか。
 *
 * MUST: `allocBefore ≤ releaseAfter < dispatch 数`。外れた宣言は executor の replay で
 * 「未確保の一時を束ねる」か「解放されない一時が残る」になり、前者は bind 面が組めず、
 * 後者は run 末尾の `assertDrained` まで気づけない。ルールの本数だけ手書きさせず、
 * 宣言の受け口 1 箇所で落とす。
 */
const assertTempLifetimes = (
  name: FusionRuleName,
  temps: readonly FusedTemp[],
  dispatchCount: number,
): void => {
  temps.forEach((temp, id) => {
    if (
      !Number.isSafeInteger(temp.byteLength) || temp.byteLength < 1 ||
      temp.allocBefore < 0 || temp.releaseAfter < temp.allocBefore ||
      temp.releaseAfter >= dispatchCount
    ) {
      throw new ExecutionError(
        `融合ルール '${name}': 一時 ${id} の寿命宣言 [${temp.allocBefore}, ${temp.releaseAfter}] が` +
          ` dispatch ${dispatchCount} 本の内側で閉じていない（${temp.byteLength}B）`,
      );
    }
  });
};

/**
 * match（掴む）と build（binds / kernel key / params を宣言する）の分離を型で強制する。
 *
 * MUST: 解放簿記の根拠（`ins` の延べ列）・畳んだ本数（`nodeCount`）・走査幅（`advance`）・
 * passthrough は**掴んだ窓と鎖から導く**。ルール側に宣言させると、同じ事実がルールの
 * 本数だけ複製され、1 本ずれても例外は出ない。
 */
const defineRule = <Matched extends FusionMatch>(rule: {
  readonly name: FusionRuleName;
  readonly heads: readonly string[];
  readonly match: (
    nodes: readonly NodePlan[],
    index: number,
    context: FusionContext,
  ) => Matched | undefined;
  readonly build: (
    matched: Matched,
  ) => Omit<FusedStep, "kind" | "rule" | "ins" | "nodeCount">;
}): FusionRule => ({
  name: rule.name,
  heads: rule.heads,
  apply: (nodes, index, context) => {
    const matched = rule.match(nodes, index, context);
    if (matched === undefined) return undefined;
    const folded = new Set(matched.chain);
    const passthrough = matched.window.filter((step) => !folded.has(step));
    // MUST: 鎖は窓の部分列。外れていれば窓外のノードを畳んだ（= 走査幅が足りず二重実行）か
    // 窓内を取りこぼした（= 未実行）ことになり、どちらも例外なしの沈黙誤値になる。
    if (passthrough.length + matched.chain.length !== matched.window.length) {
      throw new ExecutionError(
        `融合ルール '${rule.name}': 畳んだ鎖 ${matched.chain.length} 本が窓 ${matched.window.length} 本の部分列でない`,
      );
    }
    if (!passthroughIsIndependent(matched.chain, passthrough)) return undefined;
    const built = rule.build(matched);
    assertTempLifetimes(rule.name, built.temps, built.dispatches.length);
    return {
      passthrough,
      advance: matched.window.length,
      step: {
        kind: "fused",
        rule: rule.name,
        ins: externalIns(matched.chain),
        nodeCount: matched.chain.length,
        ...built,
      },
    };
  },
});

type SiluMatch = FusionMatch & {
  readonly xName: string;
  readonly outputName: string;
  readonly outputShape: readonly number[];
  readonly multiplyOrder: SiluMulOrder;
};

/**
 * SiLU: `sigmoid(x) → mul(x, sigmoid)` の連続 2 ノード。
 *
 * MUST: 中間 sigmoid 値は唯一の consumer が直後の mul で、graph output でないこと。
 * MUST: 全スロットが同 shape の f32 だけ。broadcast SiLU や別入力との gate へ一般化しない
 * （「式が似ている」で受理集合を広げると、fallback が正しいという保証の外へ出る）。
 *
 * 外部入力の延べ回数: x が sigmoid と mul で各 1 回 = 2 回。
 */
const SILU_RULE = defineRule<SiluMatch>({
  name: "silu",
  heads: ["sigmoid"],
  match: (nodes, index, context) => {
    const sigmoid = nodes[index];
    if (sigmoid?.node.op !== "sigmoid") return undefined;
    const mul = nodes[index + 1];
    if (mul?.node.op !== "mul") return undefined;
    const chain = [sigmoid, mul];
    if (!allF32(chain)) return undefined;

    const xName = sigmoid.node.ins[0];
    const intermediateName = sigmoid.outputName;
    let multiplyOrder: SiluMulOrder;
    if (mul.node.ins[0] === xName && mul.node.ins[1] === intermediateName) {
      multiplyOrder = "x-sigmoid";
    } else if (mul.node.ins[0] === intermediateName && mul.node.ins[1] === xName) {
      multiplyOrder = "sigmoid-x";
    } else {
      return undefined;
    }

    const shape = sigmoid.inputShapes[0];
    if (shape.length < 1 || shape.some((dim) => dim < 1)) return undefined;
    if (
      !sameShape(sigmoid.outputShape, shape) ||
      mul.inputShapes.some((inputShape) => !sameShape(inputShape, shape)) ||
      !sameShape(mul.outputShape, shape)
    ) return undefined;
    if (!internalsArePrivate(chain, context)) return undefined;

    return {
      window: chain,
      chain,
      xName,
      outputName: mul.outputName,
      outputShape: mul.outputShape,
      multiplyOrder,
    };
  },
  build: (matched) => {
    const count = numel(matched.outputShape);
    return {
      binds: [matched.xName],
      outputName: matched.outputName,
      outputShape: matched.outputShape,
      temps: [],
      dispatches: [{
        key: siluKey(matched.multiplyOrder),
        wgsl: () => siluWgsl(matched.multiplyOrder),
        params: siluParams(count),
        workgroups: { kind: "gridStride", items: count, size: SILU_WORKGROUP_SIZE },
      }],
    };
  },
});

type Upsample2xMatch = FusionMatch & {
  readonly inputName: string;
  readonly inputShape: readonly number[];
  readonly outputName: string;
  readonly outputShape: readonly number[];
  readonly width: number;
};

/**
 * VAE nearest-exact x2: エクスポータが出す連続 6 ノード
 * `reshape → expand(width x2) → reshape → reshape → expand(height x2) → reshape`。
 *
 * 結線・解決済み shape・dtype・内部 use-count が**全て**一致した場合だけ 1 pass へ置換する。
 * MUST: f32 rank4 NCHW の各空間軸 2 倍だけ。一般の broadcast / resize へ広げない。
 */
const UPSAMPLE_2X_RULE = defineRule<Upsample2xMatch>({
  name: "upsample2x",
  heads: ["reshape"],
  match: (nodes, index, context) => {
    const first = nodes[index];
    if (first?.node.op !== "reshape") return undefined;
    const horizontal = nodes[index + 1];
    const flatten = nodes[index + 2];
    const addHeightAxis = nodes[index + 3];
    const vertical = nodes[index + 4];
    const output = nodes[index + 5];
    if (
      horizontal === undefined || flatten === undefined || addHeightAxis === undefined ||
      vertical === undefined || output === undefined
    ) return undefined;
    const chain = [first, horizontal, flatten, addHeightAxis, vertical, output];
    if (
      horizontal.node.op !== "expand" || flatten.node.op !== "reshape" ||
      addHeightAxis.node.op !== "reshape" || vertical.node.op !== "expand" ||
      output.node.op !== "reshape"
    ) return undefined;
    if (!allF32(chain)) return undefined;

    const inputName = first.node.ins[0];
    if (
      horizontal.node.ins[0] !== first.outputName ||
      flatten.node.ins[0] !== horizontal.outputName ||
      addHeightAxis.node.ins[0] !== flatten.outputName ||
      vertical.node.ins[0] !== addHeightAxis.outputName ||
      output.node.ins[0] !== vertical.outputName
    ) return undefined;

    const inputShape = first.inputShapes[0];
    if (inputShape.length !== 4 || inputShape.some((dim) => dim < 1)) return undefined;
    const [batch, channels, height, width] = inputShape;
    const batchChannels = batch * channels;
    const flatRows = batchChannels * height;
    const outHeight = height * 2;
    const outWidth = width * 2;
    if (
      ![batchChannels, flatRows, outHeight, outWidth].every((value) => Number.isSafeInteger(value))
    ) return undefined;

    if (
      !sameShape(first.outputShape, [flatRows, width, 1]) ||
      !sameShape(horizontal.outputShape, [flatRows, width, 2]) ||
      !sameShape(flatten.outputShape, [batchChannels, height, outWidth]) ||
      !sameShape(addHeightAxis.outputShape, [batchChannels, height, 1, outWidth]) ||
      !sameShape(vertical.outputShape, [batchChannels, height, 2, outWidth]) ||
      !sameShape(output.outputShape, [batch, channels, outHeight, outWidth])
    ) return undefined;
    if (!internalsArePrivate(chain, context)) return undefined;

    return {
      window: chain,
      chain,
      inputName,
      inputShape,
      outputName: output.outputName,
      outputShape: output.outputShape,
      width,
    };
  },
  build: (matched) => {
    const sourceCount = numel(matched.inputShape);
    return {
      binds: [matched.inputName],
      outputName: matched.outputName,
      outputShape: matched.outputShape,
      temps: [],
      dispatches: [{
        key: UPSAMPLE_2X_KEY,
        wgsl: () => UPSAMPLE_2X_WGSL,
        // 巨大 shape は GPU 確保より先に u32 添字の契約で fail loudly させる。
        params: upsample2xParams(sourceCount, matched.width),
        workgroups: {
          kind: "gridStride",
          items: sourceCount,
          size: UPSAMPLE_2X_WORKGROUP_SIZE,
        },
      }],
    };
  },
});

type RopeMatch = FusionMatch & {
  readonly xName: string;
  readonly cosName: string;
  readonly sinName: string;
  readonly outputName: string;
  readonly outputShape: readonly number[];
  readonly sequence: number;
  readonly headDim: number;
};

/**
 * half-split RoPE: エクスポータが作る**連続 7 ノード**。attention の slice-first と
 * text encoder / Gemma 系の direct-mul-first の 2 順序だけを受理する。
 *
 * - slice-first: `slice×2, neg, cat, mul(x,cos), mul(cat,sin), add`
 * - direct-first: `mul(x,cos), slice×2, neg, cat, mul(cat,sin), add`
 *
 * MUST: 受理するのは `[1,H,S,D]`（D は正の偶数）/ table `[1,1,S,D]` / dim=3 の
 * `0-D/2` / `D/2-D` だけ。head 幅 D は**実測 2 種（128 と 256）**あるので slice の境界から
 * 導くが、偶奇 RoPE（`x[0::2]` / `x[1::2]` 形）・別 broadcast・別 cat 軸は「式が似ている」で
 * 広げない — 受理集合を広げた瞬間、「掴めなければ既存経路で必ず正しい」という fallback の
 * 保証が効かなくなる。カーネル（kernels/rope.ts）は `head_dim` / `half_dim` を uniform で
 * 受けるので、D の一般化に WGSL の変更は要らない。
 *
 * ## 窓内 passthrough
 *
 * `cat` と続く `mul` の間に、cos / sin 表を実行時 T へ縮める {@link SYM_PREFIX_SLICE_OP} が
 * **1 本だけ**挟まる形が実測にある（Gemma 系。表は θ 系統ごとに 1 度作って全層で使い回すので、
 * 挟まるのは各系統の初出 1 箇所だけ）。この 1 本は鎖の値を消費しないので融合ステップの前へ
 * 動かせる（合法性の判定は {@link passthroughIsIndependent}）。
 *
 * 外部入力の延べ回数: x が slice×2 と direct mul で 3 回、cos / sin が各 1 回。
 */
const ROPE_RULE = defineRule<RopeMatch>({
  name: "rope",
  heads: ["mul", "slice"],
  match: (nodes, index, context) => {
    // 全ノードで呼ばれるので、先頭 op を見てから配列を作る（ノードごとの短命 slice は
    // dispatch 削減ぶんを CPU 側で食い返す）。
    const leading = nodes[index];
    const directFirst = leading?.node.op === "mul";
    if (!directFirst && leading?.node.op !== "slice") return undefined;
    let cursor = index + (directFirst ? 1 : 0);
    const first = nodes[cursor];
    const second = nodes[cursor + 1];
    const neg = nodes[cursor + 2];
    const cat = nodes[cursor + 3];
    if (
      first?.node.op !== "slice" || second?.node.op !== "slice" || neg?.node.op !== "neg" ||
      cat?.node.op !== "cat"
    ) return undefined;
    cursor += 4;
    // 窓内 passthrough は cos / sin 表の sym_prefix_slice 1 本だけ（実測形）。ここを
    // 「任意 op の任意本数」に広げると、鎖と無関係なノードを跨いだ並べ替えまで受理してしまう。
    if (nodes[cursor]?.node.op === SYM_PREFIX_SLICE_OP) cursor += 1;
    const direct = directFirst ? leading : nodes[cursor];
    if (!directFirst) cursor += 1;
    const cross = nodes[cursor];
    const add = nodes[cursor + 1];
    if (
      direct?.node.op !== "mul" || cross?.node.op !== "mul" || add?.node.op !== "add"
    ) return undefined;
    const windowEnd = cursor + 2;

    // MUST: use-count と解放簿記は**実際のノード順**で持つ。役割順に並べ替えた列を使うと
    // direct-first だけ内部値の集合がずれる。
    const chain = directFirst
      ? [direct, first, second, neg, cat, cross, add]
      : [first, second, neg, cat, direct, cross, add];
    if (!allF32(chain)) return undefined;

    const xName = first.node.ins[0];
    if (
      second.node.ins[0] !== xName || neg.node.ins[0] !== second.outputName ||
      cat.node.ins[0] !== neg.outputName || cat.node.ins[1] !== first.outputName ||
      direct.node.ins[0] !== xName || cross.node.ins[0] !== cat.outputName ||
      add.node.ins[0] !== direct.outputName || add.node.ins[1] !== cross.outputName
    ) return undefined;

    const xShape = first.inputShapes[0];
    if (xShape.length !== 4 || xShape[0] !== 1) return undefined;
    const [, heads, sequence, headDim] = xShape;
    if (heads < 1 || sequence < 1 || headDim < 2 || headDim % 2 !== 0) return undefined;
    const halfDim = headDim / 2;

    const firstSlice = sliceAttrs(first.node.attrs, "RoPE first slice");
    const secondSlice = sliceAttrs(second.node.attrs, "RoPE second slice");
    if (
      firstSlice.dim !== 3 || firstSlice.start !== 0 || firstSlice.end !== halfDim ||
      secondSlice.dim !== 3 || secondSlice.start !== halfDim || secondSlice.end !== headDim ||
      catDim(cat.node.attrs, "RoPE cat") !== 3
    ) return undefined;

    const halfShape = [1, heads, sequence, halfDim];
    const fullShape = [1, heads, sequence, headDim];
    if (
      !sameShape(first.outputShape, halfShape) || !sameShape(second.outputShape, halfShape) ||
      !sameShape(neg.outputShape, halfShape) || !sameShape(cat.outputShape, fullShape) ||
      !sameShape(direct.outputShape, fullShape) || !sameShape(cross.outputShape, fullShape) ||
      !sameShape(add.outputShape, fullShape)
    ) return undefined;

    const tableShape = [1, 1, sequence, headDim];
    if (
      !sameShape(direct.inputShapes[1], tableShape) ||
      !sameShape(cross.inputShapes[1], tableShape)
    ) return undefined;
    if (!internalsArePrivate(chain, context)) return undefined;

    return {
      window: nodes.slice(index, windowEnd),
      chain,
      xName,
      cosName: direct.node.ins[1],
      sinName: cross.node.ins[1],
      outputName: add.outputName,
      outputShape: add.outputShape,
      sequence,
      headDim,
    };
  },
  build: (matched) => {
    const count = numel(matched.outputShape);
    return {
      binds: [matched.xName, matched.cosName, matched.sinName],
      outputName: matched.outputName,
      outputShape: matched.outputShape,
      temps: [],
      dispatches: [{
        key: ROPE_KEY,
        wgsl: () => ROPE_WGSL,
        params: ropeParams(count, matched.sequence, matched.headDim),
        workgroups: { kind: "gridStride", items: count, size: ROPE_WORKGROUP_SIZE },
      }],
    };
  },
});

type AdalnMatch = FusionMatch & {
  readonly binds: readonly string[];
  readonly outputName: string;
  readonly outputShape: readonly number[];
  readonly rows: number;
  readonly dim: number;
  readonly eps: number;
};

/** 変調ベクトルの broadcast 形（先行軸を全て 1 にした `[1,…,1,dim]`）。 */
const modulationShape = (rowShape: readonly number[], dim: number): readonly number[] => [
  ...rowShape.slice(0, -1).map(() => 1),
  dim,
];

/**
 * adaLN（DiT の変調）: エクスポータが出す**窓 6 / 7 ノード**。
 *
 * ```
 * layer_norm(x, w, b)              -> t          [.., dim]
 * reshape × 2〜3                    （窓内 passthrough — 変調ベクトルの unsqueeze）
 * add(scale, one[1])               -> s          [1,..,1,dim]
 * mul(t, s)                        -> p          [.., dim]
 * add(p, shift)                    -> y          [.., dim]
 * ```
 *
 * 畳むのは `layer_norm / add / mul / add` の 4 本で、間の reshape は 0 dispatch の別名のまま
 * 融合ステップの前に並ぶ（{@link passthroughIsIndependent} が並べ替えの合法性を見る）。
 *
 * MUST: `one` は**値を仮定せず**バッファとして束ね、カーネルが `one[0]` を読む。IR の
 * initializer の中身は融合パスからは見えない（見えたとしても `1.0` を焼き込んだ瞬間、
 * 「掴めなければ必ず正しい」の外側に出る）。
 * MUST: mul / add の入力順は実測形どおりに固定する（有限値では可換でも NaN payload の
 * 選ばれ方がバックエンドで違いうる）。SiLU のように順序変種を key へ載せる形にはしない —
 * 実測が 1 順序しかないので受理集合を広げない。
 *
 * 外部入力の延べ回数: x / ln weight / ln bias / scale / one / shift が各 1 回 = 6 回。
 */
const ADALN_RULE = defineRule<AdalnMatch>({
  name: "adaln",
  heads: [LAYER_NORM_OP],
  match: (nodes, index, context) => {
    const norm = nodes[index];
    if (norm?.node.op !== LAYER_NORM_OP) return undefined;
    // 窓内 passthrough は「直後に並ぶ連続 reshape」だけ。実測は shift / scale の 2 本と、
    // gate を足した 3 本の 2 形（gate は鎖の外で消費される）。
    let cursor = index + 1;
    while (nodes[cursor]?.node.op === "reshape") cursor += 1;
    const passthroughCount = cursor - index - 1;
    if (passthroughCount < 2 || passthroughCount > 3) return undefined;

    const modulate = nodes[cursor];
    const multiply = nodes[cursor + 1];
    const offset = nodes[cursor + 2];
    if (modulate === undefined || multiply === undefined || offset === undefined) return undefined;
    if (
      modulate.node.op !== "add" || multiply.node.op !== "mul" || offset.node.op !== "add"
    ) return undefined;

    // MUST: use-count と解放簿記は**畳むノードだけ**から導く（passthrough は素のノードとして
    // 既存経路が数える）。
    const chain = [norm, modulate, multiply, offset];
    if (!allF32(chain)) return undefined;

    if (
      multiply.node.ins[0] !== norm.outputName ||
      multiply.node.ins[1] !== modulate.outputName ||
      offset.node.ins[0] !== multiply.outputName
    ) return undefined;

    const rowShape = norm.inputShapes[0];
    if (rowShape.length < 2 || rowShape.some((extent) => extent < 1)) return undefined;
    const dim = rowShape[rowShape.length - 1];
    const rows = numel(rowShape.slice(0, -1));
    const affineShape = [dim];
    const modShape = modulationShape(rowShape, dim);
    if (
      !sameShape(norm.outputShape, rowShape) ||
      !sameShape(norm.inputShapes[1], affineShape) ||
      !sameShape(norm.inputShapes[2], affineShape) ||
      !sameShape(modulate.inputShapes[0], modShape) ||
      !sameShape(modulate.inputShapes[1], [1]) ||
      !sameShape(modulate.outputShape, modShape) ||
      !sameShape(multiply.inputShapes[0], rowShape) ||
      !sameShape(multiply.inputShapes[1], modShape) ||
      !sameShape(multiply.outputShape, rowShape) ||
      !sameShape(offset.inputShapes[0], rowShape) ||
      !sameShape(offset.inputShapes[1], modShape) ||
      !sameShape(offset.outputShape, rowShape)
    ) return undefined;
    if (!internalsArePrivate(chain, context)) return undefined;

    // bind 面はカーネルの binding 1〜6 と 1 対 1 なので**重複を許さない**（同じ値名が 2 スロットに
    // 来る形は実測に無く、{@link FusedStep.binds} の「重複無し」も崩す）。
    const binds = [
      norm.node.ins[0],
      norm.node.ins[1],
      norm.node.ins[2],
      modulate.node.ins[0],
      modulate.node.ins[1],
      offset.node.ins[1],
    ];
    if (new Set(binds).size !== binds.length) return undefined;

    const { eps } = layerNormAttrs(norm.node.attrs, `融合 adaln の ${norm.node.op}`);
    return {
      window: nodes.slice(index, cursor + 3),
      chain,
      binds,
      outputName: offset.outputName,
      outputShape: offset.outputShape,
      rows,
      dim,
      eps,
    };
  },
  build: (matched) => ({
    binds: matched.binds,
    outputName: matched.outputName,
    outputShape: matched.outputShape,
    temps: [],
    dispatches: [{
      key: ADALN_NORM_KEY,
      wgsl: () => ADALN_NORM_WGSL,
      params: adalnNormParams(matched.rows, matched.dim, matched.eps),
      // 1 行 = 1 workgroup（行方向 grid-stride）なので、割り数は 1 で行数がそのまま
      // workgroup 数になる。`@workgroup_size` は行内の 256 スレッドで別物。
      workgroups: { kind: "gridStride", items: matched.rows, size: 1 },
    }],
  }),
});

/** 行ブロック 1 枚（クエリ行の半開区間 `[offset, offset + rows)`）。 */
type RowBlock = {
  readonly offset: number;
  readonly rows: number;
};

/**
 * クエリ行を「1 枚がストレージ束縛の上限に収まる**最小枚数**」へ等分する純関数。
 *
 * 入力は解決済みの静的な数だけ（device の granted limit・1 行あたりのバイト数・行数）で、
 * 実測は 1 つも混ざらない — 実行時オートチューン禁止（ADR 0022）を満たす唯一の形。
 * したがって同じ device・同じ束縛からは常に同じ枚数が出て、prepared plan のキー
 * （解決済み bindings）が枚数まで含意する（S が違えば別キーなので枚数も別に導かれる）。
 *
 * 等分にするのは、末尾だけ極端に短いブロックを作らないため（幾何のバケットが 1 枚だけ
 * 別になり、パイプラインが 1 本余計に生える）。端数は先頭から 1 行ずつ配る。
 *
 * MUST: **1 行でも上限に入らない形は fail loudly**。ここで黙って分割を諦めると、確保も
 * 束縛も失敗するグラフが「融合が外れただけ」に見える。
 *
 * @param forced 枚数の強制（テスト専用）。上限に収まらない枚数は同じく fail loudly。
 */
export const planRowBlocks = (
  rows: number,
  bytesPerRow: number,
  limit: number,
  forced?: number,
): readonly RowBlock[] => {
  const where = `行ブロック分割（行 ${rows} × ${bytesPerRow}B / 上限 ${limit}B）`;
  for (const [name, value] of [["行数", rows], ["1 行のバイト数", bytesPerRow]] as const) {
    if (!Number.isSafeInteger(value) || value < 1) {
      throw new ExecutionError(`${where}: ${name} は正整数（${value}）`);
    }
  }
  if (bytesPerRow > limit) {
    throw new ExecutionError(
      `${where}: クエリ 1 行ぶんのスコア ${bytesPerRow}B が既にストレージ束縛の上限を超える` +
        "（行ブロックでは分割しきれない形）",
    );
  }
  const count = forced ?? Math.ceil(rows / Math.floor(limit / bytesPerRow));
  if (!Number.isSafeInteger(count) || count < 1 || count > rows) {
    throw new ExecutionError(
      `${where}: 枚数 ${count} は 1 以上 ${rows} 以下の整数でなければならない`,
    );
  }
  const base = Math.floor(rows / count);
  const remainder = rows % count;
  const widest = base + (remainder > 0 ? 1 : 0);
  if (widest * bytesPerRow > limit) {
    throw new ExecutionError(
      `${where}: ${count} 枚では 1 枚 ${widest * bytesPerRow}B が上限に収まらない`,
    );
  }
  const blocks: RowBlock[] = [];
  for (let index = 0, offset = 0; index < count; index += 1) {
    const blockRows = base + (index < remainder ? 1 : 0);
    blocks.push({ offset, rows: blockRows });
    offset += blockRows;
  }
  return blocks;
};

type RowBlockAttentionMatch = FusionMatch & {
  /** `[q, kᵀ, mask, v]`（bind 面の並び — 重複無し）。 */
  readonly binds: readonly string[];
  readonly outputName: string;
  readonly outputShape: readonly number[];
  /** B·H を畳んだバッチ軸。 */
  readonly heads: number;
  /** クエリ行数 M（= S）。 */
  readonly queries: number;
  /** キー列数 N（= C）。 */
  readonly keys: number;
  /** head 幅 D（q / k / v で共通）。 */
  readonly headDim: number;
  readonly blocks: readonly RowBlock[];
  readonly limits: FusionLimits;
};

/**
 * 分解 attention の**行ブロック実行**: エクスポータが出す連続 9 ノード
 *
 * ```
 * bmm(q[H,M,D], kᵀ[H,D,N])   -> S     [H,M,N]
 * reshape                     -> S4    [1,H,M,N]
 * add(S4, mask[1,1,1,N])      -> Sm    [1,H,M,N]
 * safe_softmax(Sm, dim=3)     -> P4    [1,H,M,N]
 * expand（恒等）→ reshape      -> P     [H,M,N]
 * expand（恒等）→ reshape      -> V     [H,N,D]
 * bmm(P, V)                   -> O     [H,M,D]
 * ```
 *
 * を、**クエリ行のブロックごとの同型 4 dispatch** へ置き換える。畳んでいるのは演算ではなく
 * **中間の実体化幅**で、`S` / `Sm` / `P` はブロック 1 枚ぶんのステップ内一時になる
 * （全 M を実体化すると `H·M·N·4` バイトがストレージ束縛の上限を越える機がある — WebGPU core
 * 既定の `maxStorageBufferBindingSize` は 128MiB）。
 *
 * ## ビット同一の根拠
 *
 * - 2 本の bmm は**行の担当割りだけ**を変える（{@link "../kernels/gemm.ts"} `BmmRowWindow`）。
 *   1 出力要素の K 縮約順も丸めの並びも 1 文字も動かない（src/kernels/gemm-geometry.ts の
 *   数値契約）。ブロックごとに幾何のバケットが変わりうるが、幾何が決めるのは担当割りだけ。
 * - `add`（mask は `[1,1,1,N]` broadcast）と `safe_softmax`（最終次元の行内縮約）はどちらも
 *   **行内で閉じている**ので、行を切っても 1 行あたりの演算列が変わらない。したがって
 *   ブロックバッファ相手に既存カーネルをそのまま撃てる（行オフセットは要らない）。
 *
 * ## 常時融合・枚数は静的
 *
 * 掴めた窓は**必ず**融合し、枚数 n は {@link planRowBlocks} が device の granted limit と
 * 解決済み shape だけから決める。**n = 1 の機では素の 4 dispatch 列と完全に同一**
 * （行窓変種を使わず `bmmKey(v4, m)` の既存キー・既存 params のまま）で、追加コストはゼロ。
 *
 * ## 適用順と head 衝突
 *
 * 先頭 op は `bmm` で、既存 4 ルールの先頭 op（`sigmoid` / `reshape` / `mul` / `slice` /
 * `layer_norm`）と互いに素なので宣言順は結果に効かない。窓の内側には `reshape` /
 * `expand` が 5 本あるが、掴んだ時点で走査は窓幅ぶん進むので内側で別ルールが発火する余地は
 * 無い（掴めなかったときだけ内側の `reshape` が upsample2x の先頭として試され、6 ノードの
 * 綴りが違うので落ちる）。
 *
 * 外部入力の延べ回数: q / kᵀ / mask / v が各 1 回 = 4 回。
 */
const ROW_BLOCK_ATTENTION_RULE = defineRule<RowBlockAttentionMatch>({
  name: "rowBlockAttention",
  heads: ["bmm"],
  match: (nodes, index, context) => {
    const qk = nodes[index];
    if (qk?.node.op !== "bmm") return undefined;
    const reshapeScores = nodes[index + 1];
    const addMask = nodes[index + 2];
    const softmax = nodes[index + 3];
    const expandP = nodes[index + 4];
    const reshapeP = nodes[index + 5];
    const expandV = nodes[index + 6];
    const reshapeV = nodes[index + 7];
    const pv = nodes[index + 8];
    if (
      reshapeScores?.node.op !== "reshape" || addMask?.node.op !== "add" ||
      softmax?.node.op !== SAFE_SOFTMAX_OP || expandP?.node.op !== "expand" ||
      reshapeP?.node.op !== "reshape" || expandV?.node.op !== "expand" ||
      reshapeV?.node.op !== "reshape" || pv?.node.op !== "bmm"
    ) return undefined;
    const chain = [
      qk,
      reshapeScores,
      addMask,
      softmax,
      expandP,
      reshapeP,
      expandV,
      reshapeV,
      pv,
    ];
    if (!allF32(chain)) return undefined;

    // 結線（窓の並びだけでは「同じ形の別の鎖が隣り合っている」を排除できない）。
    if (
      reshapeScores.node.ins[0] !== qk.outputName ||
      addMask.node.ins[0] !== reshapeScores.outputName ||
      softmax.node.ins[0] !== addMask.outputName ||
      expandP.node.ins[0] !== softmax.outputName ||
      reshapeP.node.ins[0] !== expandP.outputName ||
      reshapeV.node.ins[0] !== expandV.outputName ||
      pv.node.ins[0] !== reshapeP.outputName ||
      pv.node.ins[1] !== reshapeV.outputName
    ) return undefined;

    const [qShape, ktShape] = qk.inputShapes;
    if (qShape.length !== 3 || ktShape.length !== 3) return undefined;
    const [heads, queries, headDim] = qShape;
    const keys = ktShape[2];
    if (heads < 1 || queries < 1 || headDim < 1 || keys < 1) return undefined;
    const scores3 = [heads, queries, keys];
    const scores4 = [1, heads, queries, keys];
    if (
      !sameShape(ktShape, [heads, headDim, keys]) ||
      !sameShape(qk.outputShape, scores3) ||
      !sameShape(reshapeScores.outputShape, scores4) ||
      !sameShape(addMask.inputShapes[1], [1, 1, 1, keys]) ||
      !sameShape(addMask.outputShape, scores4) ||
      !sameShape(softmax.outputShape, scores4) ||
      // 恒等 expand（複製軸を持たない）でなければ、素の列は実体化コピーを 1 本出す。
      !sameShape(expandP.inputShapes[0], scores4) ||
      !sameShape(expandP.outputShape, scores4) ||
      !sameShape(reshapeP.outputShape, scores3) ||
      !sameShape(expandV.inputShapes[0], [1, heads, keys, headDim]) ||
      !sameShape(expandV.outputShape, [1, heads, keys, headDim]) ||
      !sameShape(reshapeV.outputShape, [heads, keys, headDim]) ||
      !sameShape(pv.outputShape, [heads, queries, headDim])
    ) return undefined;
    // 縮約軸は最終次元固定（契約が既に見ているが、窓の受理集合としても明示する）。
    if (softmaxDim(softmax.node.attrs ?? {}, "行ブロック attention の safe_softmax") !== 3) {
      return undefined;
    }
    if (!internalsArePrivate(chain, context)) return undefined;

    const binds = [qk.node.ins[0], qk.node.ins[1], addMask.node.ins[1], expandV.node.ins[0]];
    if (new Set(binds).size !== binds.length) return undefined;

    // ここから先は fail loudly の領域（掴めた窓は必ず融合する）。素の列へ落としても
    // 実体化幅は増えるだけなので、「分割しきれない」を沈黙の fallback にしてはならない。
    const blocks = planRowBlocks(
      queries,
      heads * keys * 4,
      context.limits.maxStorageBufferBindingSize,
      context.rowBlockSplit,
    );
    return {
      window: chain,
      chain,
      binds,
      outputName: pv.outputName,
      outputShape: pv.outputShape,
      heads,
      queries,
      keys,
      headDim,
      blocks,
      limits: context.limits,
    };
  },
  build: (matched) => {
    const { heads, queries, keys, headDim, blocks, limits } = matched;
    // n = 1 は行窓変種を使わない（既存キー・既存 params のまま = 素の 4 dispatch 列と同一）。
    const windowed = blocks.length > 1;
    const dispatchLimit = limits.maxComputeWorkgroupsPerDimension;
    const temps: FusedTemp[] = [];
    const dispatches: FusedDispatch[] = [];
    for (const block of blocks) {
      const first = dispatches.length;
      const rows = block.rows;
      const bytes = heads * rows * keys * 4;
      // 一時は 3 本とも「次の dispatch が読み終えたら返す」— 同時生存は常に 2 本で、
      // 3 本目はプール再利用で 1 本目の実体を掴む（ブロックを跨いでも同じ）。
      const scores = temps.length;
      temps.push({ byteLength: bytes, allocBefore: first, releaseAfter: first + 1 });
      const masked = temps.length;
      temps.push({ byteLength: bytes, allocBefore: first + 1, releaseAfter: first + 2 });
      const probabilities = temps.length;
      temps.push({ byteLength: bytes, allocBefore: first + 2, releaseAfter: first + 3 });

      // ① QK: A（q）だけ全 M ストライド + 行オフセットで読み、S はブロックとして書く。
      const qkV4 = gemmUsesVec4(headDim, keys);
      const qkGeometry = gemmGeometryForRows(rows);
      const qkWhere =
        `行ブロック bmm(QK) [${heads},${rows},${headDim}] × [${heads},${headDim},${keys}]`;
      dispatches.push({
        key: bmmKey(qkV4, rows, windowed ? "a" : undefined),
        wgsl: () => bmmWgsl(qkV4, rows, windowed ? "a" : undefined),
        params: windowed
          ? bmmRowWindowParams(rows, keys, headDim, block.offset, queries)
          : bmmParams(rows, keys, headDim),
        operands: [
          { kind: "bind", index: 0 },
          { kind: "bind", index: 1 },
          { kind: "temp", id: scores },
        ],
        workgroups: {
          kind: "tiled",
          counts: [
            tiledWorkgroups(keys, gemmTileN(qkGeometry), dispatchLimit, qkWhere),
            tiledWorkgroups(rows, gemmTileM(qkGeometry), dispatchLimit, qkWhere),
            tiledWorkgroups(heads, 1, dispatchLimit, qkWhere),
          ],
        },
      });

      // ② 加算 mask。ブロックバッファ相手なので素の elementwise がそのまま撃てる。
      const maskedShape = [1, heads, rows, keys];
      const elementwise: ElementwiseSpec = { op: "add", rank: 4, dtype: "f32" };
      dispatches.push({
        key: elementwiseKey(elementwise),
        wgsl: () => elementwiseWgsl(elementwise),
        params: elementwiseParams(elementwise, maskedShape, [maskedShape, [1, 1, 1, keys]]),
        paramsStorage: true,
        operands: [
          { kind: "temp", id: scores },
          { kind: "bind", index: 2 },
          { kind: "temp", id: masked },
        ],
        workgroups: {
          kind: "gridStride",
          items: numel(maskedShape),
          size: ELEMENTWISE_WORKGROUP_SIZE,
        },
      });

      // ③ safe_softmax（1 行 = 1 workgroup）。行内で閉じるので行を切っても値は動かない。
      dispatches.push({
        key: SAFE_SOFTMAX_KEY,
        wgsl: () => SAFE_SOFTMAX_WGSL,
        params: softmaxParams(heads * rows, keys, true),
        operands: [
          { kind: "temp", id: masked },
          { kind: "temp", id: probabilities },
        ],
        workgroups: { kind: "gridStride", items: heads * rows, size: 1 },
      });

      // ④ PV: A（P）はブロックとして読み、出力だけ全 M ストライド + 行オフセットで書く。
      const pvV4 = gemmUsesVec4(keys, headDim);
      const pvGeometry = gemmGeometryForRows(rows);
      const pvWhere =
        `行ブロック bmm(PV) [${heads},${rows},${keys}] × [${heads},${keys},${headDim}]`;
      dispatches.push({
        key: bmmKey(pvV4, rows, windowed ? "c" : undefined),
        wgsl: () => bmmWgsl(pvV4, rows, windowed ? "c" : undefined),
        params: windowed
          ? bmmRowWindowParams(rows, headDim, keys, block.offset, queries)
          : bmmParams(rows, headDim, keys),
        operands: [
          { kind: "temp", id: probabilities },
          { kind: "bind", index: 3 },
          { kind: "output" },
        ],
        workgroups: {
          kind: "tiled",
          counts: [
            tiledWorkgroups(headDim, gemmTileN(pvGeometry), dispatchLimit, pvWhere),
            tiledWorkgroups(rows, gemmTileM(pvGeometry), dispatchLimit, pvWhere),
            tiledWorkgroups(heads, 1, dispatchLimit, pvWhere),
          ],
        },
      });
    }
    return {
      binds: matched.binds,
      outputName: matched.outputName,
      outputShape: matched.outputShape,
      temps,
      dispatches,
    };
  },
});

/** MUST: この配列の順が適用順（冒頭「適用順」節）。 */
export const FUSION_RULES: readonly FusionRule[] = [
  SILU_RULE,
  UPSAMPLE_2X_RULE,
  ROPE_RULE,
  ADALN_RULE,
  ROW_BLOCK_ATTENTION_RULE,
];

/** 恒等 expand の別名化条件（ADR 0011 の追記）。1 軸でも複製があれば実体化コピーへ戻す。 */
const aliasesInput = (plan: NodePlan): boolean =>
  plan.contract.kind === "reshape" ||
  (plan.contract.kind === "expand" && sameShape(plan.inputShapes[0], plan.outputShape));

/** ノード列を走査して融合ステップへ畳む。掴めなかったノードは素のまま並ぶ。 */
export const planFusions = (
  nodes: readonly NodePlan[],
  context: FusionContext,
): FusionPlan => {
  const steps: ExecStep[] = [];
  const counts: Record<FusionCounterName, number> = {
    silu: 0,
    upsample2x: 0,
    rope: 0,
    adaln: 0,
    rowBlockAttention: 0,
    identityExpand: 0,
  };
  const pushNode = (plan: NodePlan): void => {
    const alias = aliasesInput(plan);
    if (alias && plan.contract.kind === "expand") counts.identityExpand += 1;
    steps.push({ kind: "node", plan, aliasesInput: alias });
  };
  for (let index = 0; index < nodes.length;) {
    const hit = FUSION_RULES.reduce<FusionHit | undefined>(
      (found, rule) => found ?? rule.apply(nodes, index, context),
      undefined,
    );
    if (hit !== undefined) {
      // MUST: passthrough が先（融合ステップは passthrough の出力を入力に取りうる）。
      for (const plan of hit.passthrough) pushNode(plan);
      steps.push(hit.step);
      counts[hit.step.rule] += 1;
      index += hit.advance;
      continue;
    }
    pushNode(nodes[index]);
    index += 1;
  }
  return { steps, counts };
};
