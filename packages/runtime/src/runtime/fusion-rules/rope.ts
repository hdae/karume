/**
 * 融合ルール `rope`（half-split RoPE の 7 ノード）。全ルール共通の適格条件と `defineRule` は
 * {@link "../fusion-rule.ts"}、適用順を決める宣言表は {@link "../fusion.ts"} の
 * `FUSION_RULES`。
 */

import { catDim, numel, sliceAttrs, SYM_PREFIX_SLICE_OP } from "../../ops.ts";
import {
  ROPE_BSHD_KEY,
  ROPE_KEY,
  ROPE_WORKGROUP_SIZE,
  type RopeLayout,
  ropeParams,
  ropeWgsl,
} from "../../kernels/rope.ts";
import {
  allF32,
  defineRule,
  type FusionMatch,
  internalsArePrivate,
  sameShape,
} from "../fusion-rule.ts";

type RopeMatch = FusionMatch & {
  readonly xName: string;
  readonly cosName: string;
  readonly sinName: string;
  readonly outputName: string;
  readonly outputShape: readonly number[];
  readonly tableAxisSize: number;
  readonly headDim: number;
  readonly layout: RopeLayout;
};

/**
 * half-split RoPE: エクスポータが作る**連続 7 ノード**。attention の slice-first と
 * text encoder / Gemma 系の direct-mul-first の 2 順序だけを受理する。
 *
 * - slice-first: `slice×2, neg, cat, mul(x,cos), mul(cat,sin), add`
 * - direct-first: `mul(x,cos), slice×2, neg, cat, mul(cat,sin), add`
 *
 * MUST: `[1,H,S,D]` / table `[1,1,S,D]` または `[1,S,H,D]` / table `[1,S,1,D]`
 * （D は正の偶数）/ dim=3 の
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
 * 動かせる（合法性の判定は {@link "../fusion-rule.ts"} の `passthroughIsIndependent`）。
 *
 * 外部入力の延べ回数: x が slice×2 と direct mul で 3 回、cos / sin が各 1 回。
 */
export const ROPE_RULE = defineRule<RopeMatch>({
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
      second.node.ins[0] !== xName || neg.node.ins[0] !== second.outputs[0].name ||
      cat.node.ins[0] !== neg.outputs[0].name || cat.node.ins[1] !== first.outputs[0].name ||
      direct.node.ins[0] !== xName || cross.node.ins[0] !== cat.outputs[0].name ||
      add.node.ins[0] !== direct.outputs[0].name || add.node.ins[1] !== cross.outputs[0].name
    ) return undefined;

    const xShape = first.inputShapes[0];
    if (xShape.length !== 4 || xShape[0] !== 1) return undefined;
    const [, axis1, axis2, headDim] = xShape;
    if (axis1 < 1 || axis2 < 1 || headDim < 2 || headDim % 2 !== 0) return undefined;
    const tablesMatch = (shape: readonly number[]): boolean =>
      sameShape(direct.inputShapes[1], shape) && sameShape(cross.inputShapes[1], shape);
    // 両者が一致する退化形は既存BHSDを優先し、既存のキー・本文を維持する。
    const layout = tablesMatch([1, 1, axis2, headDim])
      ? "bhsd"
      : tablesMatch([1, axis1, 1, headDim])
      ? "bshd"
      : undefined;
    if (layout === undefined) return undefined;
    // 両レイアウトとも第2軸が表添字の除数（BHSD=S、BSHD=H）。
    const tableAxisSize = axis2;
    const halfDim = headDim / 2;

    const firstSlice = sliceAttrs(first.node.attrs, "RoPE first slice");
    const secondSlice = sliceAttrs(second.node.attrs, "RoPE second slice");
    if (
      firstSlice.dim !== 3 || firstSlice.start !== 0 || firstSlice.end !== halfDim ||
      secondSlice.dim !== 3 || secondSlice.start !== halfDim || secondSlice.end !== headDim ||
      catDim(cat.node.attrs, "RoPE cat") !== 3
    ) return undefined;

    const halfShape = [1, axis1, axis2, halfDim];
    const fullShape = [1, axis1, axis2, headDim];
    if (
      !sameShape(first.outputs[0].shape, halfShape) ||
      !sameShape(second.outputs[0].shape, halfShape) ||
      !sameShape(neg.outputs[0].shape, halfShape) || !sameShape(cat.outputs[0].shape, fullShape) ||
      !sameShape(direct.outputs[0].shape, fullShape) ||
      !sameShape(cross.outputs[0].shape, fullShape) ||
      !sameShape(add.outputs[0].shape, fullShape)
    ) return undefined;

    if (!internalsArePrivate(chain, context)) return undefined;

    return {
      window: nodes.slice(index, windowEnd),
      chain,
      xName,
      cosName: direct.node.ins[1],
      sinName: cross.node.ins[1],
      outputName: add.outputs[0].name,
      outputShape: add.outputs[0].shape,
      tableAxisSize,
      headDim,
      layout,
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
        key: matched.layout === "bhsd" ? ROPE_KEY : ROPE_BSHD_KEY,
        wgsl: () => ropeWgsl(matched.layout),
        params: ropeParams(count, matched.tableAxisSize, matched.headDim, matched.layout),
        workgroups: { kind: "gridStride", items: count, size: ROPE_WORKGROUP_SIZE },
      }],
    };
  },
});
