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
 * ## 適用順
 *
 * {@link FUSION_RULES} の**宣言順**（silu → upsample2x → rope）。現行 3 ルールの先頭 op は
 * `sigmoid` / `reshape` / `mul|slice` で互いに素なので、この順序は結果に効かない
 * （順序が意味を持つのは先頭 op が重なったときだけ — 重なりが生じていないことは
 * tests/runtime_fusion_test.ts が {@link FusionRule.heads} から機械的に検査する）。
 */

import { catDim, numel, sliceAttrs } from "../ops.ts";
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
import type { NodePlan } from "./plan.ts";

/** 融合ルールの識別子（{@link FUSION_RULES} の宣言順と 1 対 1）。 */
export type FusionRuleName = "silu" | "upsample2x" | "rope";

/**
 * 診断カウンタの見出し。融合 3 ルールに加えて、0 dispatch の別名化のうち**条件付きで外れうる**
 * 恒等 expand（{@link ExecStep} の `aliasesInput`）を数える。reshape の別名化は無条件なので
 * 数えない（外れようがない = 観測する意味がない）。
 */
export type FusionCounterName = FusionRuleName | "identityExpand";

/** ルール別の適用回数。「融合が黙って外れて性能だけ落ちる」事故の唯一の観測点。 */
export type FusionCounts = Readonly<Record<FusionCounterName, number>>;

/** 融合カーネル 1 dispatch ぶんの生成入力（全ルール共通の形）。 */
export type FusedDispatch = {
  readonly key: string;
  /**
   * MUST: 同一キーには常にバイト単位で同一の WGSL（PipelineCache の決定性契約）。
   * thunk なのは、キャッシュ済みでも本文の突き合わせが必要な一方、計画時に全ステップぶんを
   * 文字列で持つ必要が無いため。
   */
  readonly wgsl: () => string;
  /** 16 バイトの uniform Params（融合カーネルは全て uniform で受ける）。 */
  readonly params: Uint32Array<ArrayBuffer>;
  /** grid-stride の被覆対象数（workgroup 数はここから決まる）。 */
  readonly gridItems: number;
  readonly workgroupSize: number;
};

/** 融合ステップ（元の連続ノード列 1 本を private カーネル 1 dispatch へ置換したもの）。 */
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
  /** 置換した連続ノードの本数（走査を進める幅）。 */
  readonly nodeCount: number;
  readonly outputName: string;
  readonly outputShape: readonly number[];
  readonly dispatch: FusedDispatch;
};

export type NodeStep = {
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

/** 判定に要るグラフ全体の事実（executor の Session 状態から渡す）。 */
export type FusionContext = {
  /** 値名 → グラフ内の消費回数（plan.ts の countUses）。 */
  readonly useCounts: ReadonlyMap<string, number>;
  readonly outputNames: ReadonlySet<string>;
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

/** ルールの本体（match で掴み、build で宣言的にステップへ落とす）。 */
type FusionRule = {
  readonly name: FusionRuleName;
  /** このルールが掴みうる先頭 op（適用順の互いに素性を機械検査するための宣言）。 */
  readonly heads: readonly string[];
  readonly apply: (
    nodes: readonly NodePlan[],
    index: number,
    context: FusionContext,
  ) => FusedStep | undefined;
};

/**
 * match（掴む）と build（binds / kernel key / params を宣言する）の分離を型で強制する。
 *
 * MUST: 解放簿記の根拠（`ins` の延べ列）と走査幅（`nodeCount`）は**掴んだ鎖から導く**。
 * ルール側に宣言させると、同じ事実がルールの本数だけ複製され、1 本ずれても例外は出ない。
 */
const defineRule = <Matched extends { readonly chain: readonly NodePlan[] }>(rule: {
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
    return {
      kind: "fused",
      rule: rule.name,
      ins: externalIns(matched.chain),
      nodeCount: matched.chain.length,
      ...rule.build(matched),
    };
  },
});

type SiluMatch = {
  readonly chain: readonly NodePlan[];
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
      dispatch: {
        key: siluKey(matched.multiplyOrder),
        wgsl: () => siluWgsl(matched.multiplyOrder),
        params: siluParams(count),
        gridItems: count,
        workgroupSize: SILU_WORKGROUP_SIZE,
      },
    };
  },
});

type Upsample2xMatch = {
  readonly chain: readonly NodePlan[];
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
      dispatch: {
        key: UPSAMPLE_2X_KEY,
        wgsl: () => UPSAMPLE_2X_WGSL,
        // 巨大 shape は GPU 確保より先に u32 添字の契約で fail loudly させる。
        params: upsample2xParams(sourceCount, matched.width),
        gridItems: sourceCount,
        workgroupSize: UPSAMPLE_2X_WORKGROUP_SIZE,
      },
    };
  },
});

type RopeMatch = {
  readonly chain: readonly NodePlan[];
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
 * text encoder の direct-mul-first の 2 順序だけを受理する。
 *
 * - slice-first: `slice×2, neg, cat, mul(x,cos), mul(cat,sin), add`
 * - direct-first: `mul(x,cos), slice×2, neg, cat, mul(cat,sin), add`
 *
 * MUST: 実測した `[1,H,S,128]` / table `[1,1,S,128]` / dim=3 の 0-64 / 64-128 だけを受理する。
 * 偶奇 RoPE・別 head 幅・別 broadcast を「式が似ている」で広げない — 受理集合を広げた瞬間、
 * 「掴めなければ既存経路で必ず正しい」という fallback の保証が効かなくなる。
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
    const first = nodes[index + (directFirst ? 1 : 0)];
    const second = nodes[index + (directFirst ? 2 : 1)];
    const neg = nodes[index + (directFirst ? 3 : 2)];
    const cat = nodes[index + (directFirst ? 4 : 3)];
    const direct = nodes[index + (directFirst ? 0 : 4)];
    const cross = nodes[index + 5];
    const add = nodes[index + 6];
    if (
      first === undefined || second === undefined || neg === undefined || cat === undefined ||
      direct === undefined || cross === undefined || add === undefined
    ) return undefined;
    if (
      first.node.op !== "slice" || second.node.op !== "slice" || neg.node.op !== "neg" ||
      cat.node.op !== "cat" ||
      direct.node.op !== "mul" || cross.node.op !== "mul" || add.node.op !== "add"
    ) return undefined;

    // MUST: use-count と解放簿記は**実際のノード順**で持つ。役割順に並べ替えた列を使うと
    // direct-first だけ内部値の集合がずれる。
    const chain = nodes.slice(index, index + 7);
    if (!allF32(chain)) return undefined;

    const xName = first.node.ins[0];
    if (
      second.node.ins[0] !== xName || neg.node.ins[0] !== second.outputName ||
      cat.node.ins[0] !== neg.outputName || cat.node.ins[1] !== first.outputName ||
      direct.node.ins[0] !== xName || cross.node.ins[0] !== cat.outputName ||
      add.node.ins[0] !== direct.outputName || add.node.ins[1] !== cross.outputName
    ) return undefined;

    const firstSlice = sliceAttrs(first.node.attrs, "RoPE first slice");
    const secondSlice = sliceAttrs(second.node.attrs, "RoPE second slice");
    if (
      firstSlice.dim !== 3 || firstSlice.start !== 0 || firstSlice.end !== 64 ||
      secondSlice.dim !== 3 || secondSlice.start !== 64 || secondSlice.end !== 128 ||
      catDim(cat.node.attrs, "RoPE cat") !== 3
    ) return undefined;

    const xShape = first.inputShapes[0];
    if (xShape.length !== 4 || xShape[0] !== 1 || xShape[3] !== 128) return undefined;
    const [, heads, sequence, headDim] = xShape;
    if (heads < 1 || sequence < 1) return undefined;
    const halfShape = [1, heads, sequence, 64];
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
      dispatch: {
        key: ROPE_KEY,
        wgsl: () => ROPE_WGSL,
        params: ropeParams(count, matched.sequence, matched.headDim),
        gridItems: count,
        workgroupSize: ROPE_WORKGROUP_SIZE,
      },
    };
  },
});

/** MUST: この配列の順が適用順（冒頭「適用順」節）。 */
export const FUSION_RULES: readonly FusionRule[] = [SILU_RULE, UPSAMPLE_2X_RULE, ROPE_RULE];

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
    identityExpand: 0,
  };
  for (let index = 0; index < nodes.length;) {
    const fused = FUSION_RULES.reduce<FusedStep | undefined>(
      (hit, rule) => hit ?? rule.apply(nodes, index, context),
      undefined,
    );
    if (fused !== undefined) {
      steps.push(fused);
      counts[fused.rule] += 1;
      index += fused.nodeCount;
      continue;
    }
    const plan = nodes[index];
    const alias = aliasesInput(plan);
    if (alias && plan.contract.kind === "expand") counts.identityExpand += 1;
    steps.push({ kind: "node", plan, aliasesInput: alias });
    index += 1;
  }
  return { steps, counts };
};
