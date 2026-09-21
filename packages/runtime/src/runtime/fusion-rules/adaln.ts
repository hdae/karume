/**
 * 融合ルール `adaln`（DiT の変調 — 窓 6 / 7 ノード）。全ルール共通の適格条件と `defineRule`
 * は {@link "../fusion-rule.ts"}、適用順を決める宣言表は {@link "../fusion.ts"} の
 * `FUSION_RULES`。
 */

import { LAYER_NORM_OP, layerNormAttrs, numel } from "../../ops.ts";
import { ADALN_NORM_KEY, ADALN_NORM_WGSL, adalnNormParams } from "../../kernels/adaln-norm.ts";
import {
  allF32,
  defineRule,
  type FusionMatch,
  internalsArePrivate,
  sameShape,
} from "../fusion-rule.ts";

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
 * 融合ステップの前に並ぶ（{@link "../fusion-rule.ts"} の `passthroughIsIndependent` が
 * 並べ替えの合法性を見る）。
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
export const ADALN_RULE = defineRule<AdalnMatch>({
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
      multiply.node.ins[0] !== norm.outputs[0].name ||
      multiply.node.ins[1] !== modulate.outputs[0].name ||
      offset.node.ins[0] !== multiply.outputs[0].name
    ) return undefined;

    const rowShape = norm.inputShapes[0];
    if (rowShape.length < 2 || rowShape.some((extent) => extent < 1)) return undefined;
    const dim = rowShape[rowShape.length - 1];
    const rows = numel(rowShape.slice(0, -1));
    const affineShape = [dim];
    const modShape = modulationShape(rowShape, dim);
    if (
      !sameShape(norm.outputs[0].shape, rowShape) ||
      !sameShape(norm.inputShapes[1], affineShape) ||
      !sameShape(norm.inputShapes[2], affineShape) ||
      !sameShape(modulate.inputShapes[0], modShape) ||
      !sameShape(modulate.inputShapes[1], [1]) ||
      !sameShape(modulate.outputs[0].shape, modShape) ||
      !sameShape(multiply.inputShapes[0], rowShape) ||
      !sameShape(multiply.inputShapes[1], modShape) ||
      !sameShape(multiply.outputs[0].shape, rowShape) ||
      !sameShape(offset.inputShapes[0], rowShape) ||
      !sameShape(offset.inputShapes[1], modShape) ||
      !sameShape(offset.outputs[0].shape, rowShape)
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
      outputName: offset.outputs[0].name,
      outputShape: offset.outputs[0].shape,
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
