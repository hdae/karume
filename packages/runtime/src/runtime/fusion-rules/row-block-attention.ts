/**
 * 融合ルール `rowBlockAttention`（分解 attention の 9 ノードを**クエリ行のブロック実行**へ）と
 * その分割の純関数 {@link planRowBlocks}。全ルール共通の適格条件と `defineRule` は
 * {@link "../fusion-rule.ts"}、適用順を決める宣言表は {@link "../fusion.ts"} の
 * `FUSION_RULES`。
 */

import { numel, SAFE_SOFTMAX_OP, softmaxDim } from "../../ops.ts";
import { tiledWorkgroups } from "../../codegen/dispatch.ts";
import {
  ELEMENTWISE_WORKGROUP_SIZE,
  elementwiseKey,
  elementwiseParams,
  type ElementwiseSpec,
  elementwiseWgsl,
} from "../../codegen/elementwise.ts";
import { bmmKey, bmmParams, bmmRowWindowParams, bmmWgsl } from "../../kernels/bmm.ts";
import { gemmUsesVec4 } from "../../kernels/gemm.ts";
import { gemmGeometryForRows, gemmTileM, gemmTileN } from "../../kernels/gemm-geometry.ts";
import { SAFE_SOFTMAX_KEY, SAFE_SOFTMAX_WGSL, softmaxParams } from "../../kernels/softmax.ts";
import { ExecutionError } from "../plan.ts";
import {
  allF32,
  defineRule,
  type FusedDispatch,
  type FusedTemp,
  type FusionLimits,
  type FusionMatch,
  internalsArePrivate,
  sameShape,
} from "../fusion-rule.ts";

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
 * - 2 本の bmm は**行の担当割りだけ**を変える（{@link "../../kernels/gemm.ts"} `BmmRowWindow`）。
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
 * 先頭 op は `bmm` で、他ルールの先頭 op（`rms_norm` / `sigmoid` / `reshape` / `mul` / `slice` /
 * `layer_norm`）と互いに素なので宣言順は結果に効かない。窓の内側には `reshape` /
 * `expand` が 5 本あるが、掴んだ時点で走査は窓幅ぶん進むので内側で別ルールが発火する余地は
 * 無い（掴めなかったときだけ内側の `reshape` が upsample2x の先頭として試され、6 ノードの
 * 綴りが違うので落ちる）。
 *
 * 外部入力の延べ回数: q / kᵀ / mask / v が各 1 回 = 4 回。
 */
export const ROW_BLOCK_ATTENTION_RULE = defineRule<RowBlockAttentionMatch>({
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
      reshapeScores.node.ins[0] !== qk.outputs[0].name ||
      addMask.node.ins[0] !== reshapeScores.outputs[0].name ||
      softmax.node.ins[0] !== addMask.outputs[0].name ||
      expandP.node.ins[0] !== softmax.outputs[0].name ||
      reshapeP.node.ins[0] !== expandP.outputs[0].name ||
      reshapeV.node.ins[0] !== expandV.outputs[0].name ||
      pv.node.ins[0] !== reshapeP.outputs[0].name ||
      pv.node.ins[1] !== reshapeV.outputs[0].name
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
      !sameShape(qk.outputs[0].shape, scores3) ||
      !sameShape(reshapeScores.outputs[0].shape, scores4) ||
      !sameShape(addMask.inputShapes[1], [1, 1, 1, keys]) ||
      !sameShape(addMask.outputs[0].shape, scores4) ||
      !sameShape(softmax.outputs[0].shape, scores4) ||
      // 恒等 expand（複製軸を持たない）でなければ、素の列は実体化コピーを 1 本出す。
      !sameShape(expandP.inputShapes[0], scores4) ||
      !sameShape(expandP.outputs[0].shape, scores4) ||
      !sameShape(reshapeP.outputs[0].shape, scores3) ||
      !sameShape(expandV.inputShapes[0], [1, heads, keys, headDim]) ||
      !sameShape(expandV.outputs[0].shape, [1, heads, keys, headDim]) ||
      !sameShape(reshapeV.outputs[0].shape, [heads, keys, headDim]) ||
      !sameShape(pv.outputs[0].shape, [heads, queries, headDim])
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
      outputName: pv.outputs[0].name,
      outputShape: pv.outputs[0].shape,
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
      // 3 本目は配り直しで 1 本目の実体を掴む（ブロックを跨いでも同じ）。
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
