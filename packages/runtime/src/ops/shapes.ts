import { STRIDED_RANK } from "../codegen/strided.ts";
import {
  attentionScale,
  catDim,
  conv1dAttrs,
  conv2dAttrs,
  convTranspose1dAttrs,
  cumsumDim,
  deformConv2dAttrs,
  flipDim,
  layerNormAttrs,
  padAttrs,
  permuteDims,
  reduceDim,
  rmsNormEps,
  sliceAttrs,
  softmaxDim,
  symPrefixSliceAttrs,
  upsampleBilinear2dAttrs,
} from "./attrs.ts";
import { assertArity, type OpContract, scalarParamValues } from "./contracts.ts";
import { OpContractError } from "./names.ts";

export const numel = (shape: readonly number[]): number =>
  shape.reduce((count, dim) => count * dim, 1);

/**
 * torch 準拠の右詰め broadcast。次元は「一致」または「片方が 1」のみ許す。
 * MUST: 結果を `max(a, b)` で決めない — 0 と 1 の組（`max` なら 1）が torch では 0 になる。
 */
export const broadcastShapes = (
  a: readonly number[],
  b: readonly number[],
  where: string,
): number[] => {
  const rank = Math.max(a.length, b.length);
  const out: number[] = [];
  for (let i = 0; i < rank; i += 1) {
    const da = a[a.length - rank + i] ?? 1;
    const db = b[b.length - rank + i] ?? 1;
    if (da === db || db === 1) {
      out.push(da);
    } else if (da === 1) {
      out.push(db);
    } else {
      throw new OpContractError(
        `${where}: shape [${a.join(",")}] と [${b.join(",")}] は右詰め broadcast できない`,
      );
    }
  }
  return out;
};

/**
 * 出力 shape が**入力から導けない** op のための追加入力。
 *
 * reshape / expand は「出力の宣言 shape が目標形」という契約（ADR 0011）なので、宣言を
 * 渡さずには計算できない。permute は attrs の並べ替え表が要る。
 */
export type ShapeContext = {
  /** 束縛解決済みの宣言 shape（reshape / expand で必須）。 */
  readonly declared?: readonly number[];
  /** ノードの attrs（permute / layer_norm / softmax / conv1d / sym_prefix_slice で必須）。 */
  readonly attrs?: Readonly<Record<string, unknown>>;
  /**
   * シンボル束縛（sym_prefix_slice で必須）。prefix 長は `coeff·sym+offset` で、入力 shape
   * からは導けない。
   *
   * MUST: 参照は `Object.hasOwn` のみ（`bindings[sym] !== undefined` は Object.prototype 由来の
   * `toString` 等が素通りして以後の算術が NaN 化する — 横断の不変条件）。
   */
  readonly bindings?: Readonly<Record<string, number>>;
};

/**
 * strided コピー族（permute / expand / sym_prefix_slice / masked_fill）の rank 上限。
 *
 * MUST: **契約層で見る**。カーネルの params は rank {@link STRIDED_RANK} 固定で、超過は
 * codegen 層（`stridedParams`）まで落ちて初めて CodegenError になる — 利用者から見ると
 * 「契約検査は通ったのに実行段で内部エラー」で、どの op のどの入力が悪いのか出ない。
 * ここで落とせば診断が op と入力の名前つきになり、CPU 参照（同じ契約表を通る）と GPU の
 * 受理範囲も揃う。
 */
const assertStridedRank = (rank: number, what: string, where: string): void => {
  if (rank < 1 || rank > STRIDED_RANK) {
    throw new OpContractError(
      `${where}: ${what} の rank ${rank} は 1..${STRIDED_RANK} の外（strided カーネルの上限）`,
    );
  }
};

const requireDeclared = (
  context: ShapeContext,
  found: OpContract,
  where: string,
): readonly number[] => {
  if (context.declared === undefined) {
    throw new OpContractError(
      `${where}: op '${found.name}' の出力 shape は宣言が目標形（ShapeContext.declared が要る）`,
    );
  }
  return context.declared;
};

/**
 * 単一出力 op の出力列（出力が 1 本であることをアームごとに明示する — ADR 0068 決定 1）。
 * 多出力 op のアームは列を直接組んで返す。
 */
const sole = (shape: number[]): number[][] => [shape];

/**
 * 束縛解決済みの入力 shape から**出力 slot 順の shape 列**を計算する（ADR 0068 決定 1）。
 * 列の長さは契約が宣言する出力数（出力 dtype 写像の列長）と一致する — 現状の op は全て 1 本。
 */
export const computeOutputShape = (
  found: OpContract,
  inputShapes: readonly (readonly number[])[],
  where: string,
  context: ShapeContext = {},
): number[][] => {
  assertArity(found, inputShapes.length, "入力 shape 数", where);
  switch (found.kind) {
    case "unary":
      // MUST: スカラ attr の値域と**キーを跨ぐ不変条件**（clamp の min <= max）をここで見る。
      // 全ノードが必ず通る共通経路はこの計算だけで、attrs スキーマはキー単位の検査しか
      // 表せない（src/runtime/plan.ts の planGraph が全ノードでここを呼ぶ）。
      scalarParamValues(found, context.attrs ?? {}, where);
      return sole([...inputShapes[0]]);
    case "cast":
      return sole([...inputShapes[0]]);
    case "binary":
      return sole(broadcastShapes(inputShapes[0], inputShapes[1], `${where} (${found.name})`));
    case "where": {
      // torch と同じく 3 者を右詰め broadcast する（条件も値と同じ規則で広がる）。
      const [cond, a, b] = inputShapes;
      const label = `${where} (${found.name})`;
      return sole(broadcastShapes(broadcastShapes(cond, a, label), b, label));
    }
    case "cumsum": {
      const shape = inputShapes[0];
      const dim = cumsumDim(context.attrs ?? {}, where);
      // MUST: 最終次元以外は受理しない（softmax と同じ理由 — 行カーネルは縮約軸が連続で
      // あることを前提にしていて、通せば黙って別の軸を畳む）。
      if (shape.length < 1 || dim !== shape.length - 1) {
        throw new OpContractError(
          `${where}: cumsum は最終次元のみ（attrs.dim=${dim} / 入力 [${shape.join(",")}]）`,
        );
      }
      // 長さ 0 の軸は素通し（前縁和の identity は 0 で、空行は空行のまま）。
      return sole([...shape]);
    }
    case "matmul": {
      const [a, b] = inputShapes;
      if (a.length !== 2 || b.length !== 2) {
        throw new OpContractError(
          `${where}: matmul は rank-2 × rank-2 のみ（[${a.join(",")}] × [${b.join(",")}]）`,
        );
      }
      if (a[1] !== b[0]) {
        throw new OpContractError(
          `${where}: matmul の縮約次元が不一致 [${a.join(",")}] × [${b.join(",")}]`,
        );
      }
      return sole([a[0], b[1]]);
    }
    case "bmm": {
      const [a, b] = inputShapes;
      // MUST: rank-2 を通さない（matmul の担当）。兼用にすると「バッチ軸を落とした形」が
      // 同じ op 名で通り、B の取り違えが shape 検査を素通りする。
      if (a.length !== 3 || b.length !== 3) {
        throw new OpContractError(
          `${where}: bmm は rank-3 × rank-3 のみ（rank-2 は matmul）: [${a.join(",")}] × [${
            b.join(",")
          }]`,
        );
      }
      if (a[0] !== b[0]) {
        throw new OpContractError(
          `${where}: bmm のバッチ次元が不一致 [${a.join(",")}] × [${b.join(",")}]`,
        );
      }
      if (a[2] !== b[1]) {
        throw new OpContractError(
          `${where}: bmm の縮約次元が不一致 [${a.join(",")}] × [${b.join(",")}]`,
        );
      }
      return sole([a[0], a[1], b[2]]);
    }
    case "gather": {
      const [src, index] = inputShapes;
      // 契約は「最終次元固定」— 先行次元は src と index で完全一致し、最終次元だけが自由。
      // torch の一般 gather（他軸の長さが src 以下でよい）より狭いが、実測形はこれで足りる。
      if (src.length === 0 || index.length !== src.length) {
        throw new OpContractError(
          `${where}: gather は src と index が同じ rank（1 以上）: [${src.join(",")}] / [${
            index.join(",")
          }]`,
        );
      }
      const mismatch = src.findIndex((dim, d) => d < src.length - 1 && dim !== index[d]);
      if (mismatch >= 0) {
        throw new OpContractError(
          `${where}: gather の先行次元 ${mismatch} が不一致 [${src.join(",")}] / [${
            index.join(",")
          }]`,
        );
      }
      // 出力は index と同形（値は src から引く）。添字の値域は実行時データ依存なので
      // shape 契約では見ない（方針は src/kernels/gather.ts と reference/ops.ts）。
      return sole([...index]);
    }
    case "rowReduce": {
      const shape = inputShapes[0];
      if (shape.length === 0) {
        throw new OpContractError(
          `${where}: reduce の入力は rank 1 以上（スカラは縮約できない）`,
        );
      }
      const dim = reduceDim(context.attrs ?? {}, where);
      // MUST: 負値・rank 外は fail loudly（負の軸表記の正規化はエクスポータ境界の責務で、
      // ここで `% rank` を補うと「宣言と実 rank の食い違い」を黙って別の軸へ吸収してしまう）。
      if (dim >= shape.length) {
        throw new OpContractError(
          `${where}: op '${found.name}' の attrs.dim=${dim} が rank ${shape.length} の範囲外`,
        );
      }
      // 空軸の amax/amin は identity が定義できない（torch も同様に拒否する）。sum は 0。
      if (shape[dim] === 0 && found.name !== "sum") {
        throw new OpContractError(`${where}: op '${found.name}' は長さ 0 の軸を縮約できない`);
      }
      return sole([...shape.slice(0, dim), ...shape.slice(dim + 1)]);
    }
    case "argmax": {
      const shape = inputShapes[0];
      // reduce 族と違い軸は attrs ではなく**最終次元固定**（欄の不存在が「他の軸は語彙に
      // 無い」の宣言 — ADR 0068 決定 2）。
      if (shape.length === 0) {
        throw new OpContractError(
          `${where}: argmax の入力は rank 1 以上（スカラは縮約できない）`,
        );
      }
      const last = shape.length - 1;
      // 長さ 0 の軸に「最大値の添字」は無い（torch も拒否する）。amax / amin と同じ絞りで、
      // カーネルの番兵 index が出力へ漏れる形を契約側で止める。
      if (shape[last] === 0) {
        throw new OpContractError(`${where}: argmax は長さ 0 の最終次元を縮約できない`);
      }
      // **rank 保存**（`keepdim` 相当の欄が無い固定形 — 最終次元を 1 に潰す）。
      return sole([...shape.slice(0, last), 1]);
    }
    case "reshape": {
      const target = requireDeclared(context, found, where);
      const source = inputShapes[0];
      // 契約は要素数一致だけ（要素順は変えない）。ここを緩めると別名化した実バッファの
      // 大きさと宣言 shape が食い違い、readback が範囲外まで読む。
      if (numel(source) !== numel(target)) {
        throw new OpContractError(
          `${where}: reshape の要素数が合わない [${source.join(",")}] → [${target.join(",")}]`,
        );
      }
      return sole([...target]);
    }
    case "expand": {
      const target = requireDeclared(context, found, where);
      const source = inputShapes[0];
      assertStridedRank(source.length, "expand の入力", where);
      assertStridedRank(target.length, "expand の出力", where);
      if (target.length < source.length) {
        throw new OpContractError(
          `${where}: expand は rank を下げられない [${source.join(",")}] → [${target.join(",")}]`,
        );
      }
      // 右詰めで、入力の各次元は「目標と一致」か「長さ 1（stride 0 で複製）」のみ。
      const offset = target.length - source.length;
      source.forEach((extent, index) => {
        if (extent !== 1 && extent !== target[offset + index]) {
          throw new OpContractError(
            `${where}: expand は長さ 1 でない次元 ${index}（${extent}）を ${
              target[offset + index]
            } に拡張できない`,
          );
        }
      });
      return sole([...target]);
    }
    case "slice": {
      const source = inputShapes[0];
      // 実行は strided 読みコピー族の流用（ADR 0014）なので rank 上限も同じ。
      assertStridedRank(source.length, "slice の入力", where);
      const { dim, start, end } = sliceAttrs(context.attrs ?? {}, where);
      if (dim >= source.length) {
        throw new OpContractError(
          `${where}: slice の dim ${dim} が入力 rank ${source.length} の外`,
        );
      }
      // MUST: 範囲外の切り出しを通さない。GPU では例外なしに隣の値（別の行・別のバッファ）を
      // 読む形になり、shape 検査だけが「宣言どおり」で素通りする。
      if (end > source[dim]) {
        throw new OpContractError(
          `${where}: slice の end ${end} が軸 ${dim} の長さ ${source[dim]} を超える`,
        );
      }
      // MUST: キーを跨ぐ不変条件（clamp の min <= max と同じ分担）— attrs スキーマは
      // キー単位の検査しか表せない。逆転を許すと長さが負になり、要素数だけが 0 で通る。
      if (start > end) {
        throw new OpContractError(`${where}: slice の start ${start} が end ${end} を超える`);
      }
      const out = [...source];
      out[dim] = end - start;
      return sole(out);
    }
    case "cat": {
      const dim = catDim(context.attrs ?? {}, where);
      const first = inputShapes[0];
      // 実行は strided 書きコピー族（ADR 0014）— 出力側の stride を params に載せるので、
      // rank 上限は出力（= 入力と同 rank）に効く。
      assertStridedRank(first.length, "cat の入力", where);
      if (dim >= first.length) {
        throw new OpContractError(`${where}: cat の dim ${dim} が入力 rank ${first.length} の外`);
      }
      let total = 0;
      inputShapes.forEach((shape, index) => {
        if (shape.length !== first.length) {
          throw new OpContractError(
            `${where}: cat の入力 ${index} の rank ${shape.length} が入力 0 の ${first.length} と違う`,
          );
        }
        // MUST: 連結軸**以外**は全一致を要求する（torch と同じ）。緩めると出力の一部が
        // どの入力にも書かれないまま残り、full-write 不変条件が破れる。
        shape.forEach((extent, axis) => {
          if (axis !== dim && extent !== first[axis]) {
            throw new OpContractError(
              `${where}: cat の入力 ${index} [${shape.join(",")}] が入力 0 [${
                first.join(",")
              }] と軸 ${axis} で違う（連結軸は ${dim}）`,
            );
          }
        });
        total += shape[dim];
      });
      const out = [...first];
      // 出力の軸長 = 入力の軸長の総和。この規則そのものが「全入力で出力全域を覆う」
      // （full-write — ADR 0014）の担保で、executor 側は書き出し位置の総和をこれと突き合わせる。
      out[dim] = total;
      return sole(out);
    }
    case "pad": {
      const source = inputShapes[0];
      if (source.length < 1) {
        throw new OpContractError(`${where}: pad の入力は rank 1 以上（最終次元が要る）`);
      }
      const { left, right } = padAttrs(context.attrs ?? {}, where);
      const out = [...source];
      out[out.length - 1] = source[source.length - 1] + left + right;
      return sole(out);
    }
    case "flip": {
      const source = inputShapes[0];
      const dim = flipDim(context.attrs ?? {}, where);
      if (source.length < 1 || dim >= source.length) {
        throw new OpContractError(
          `${where}: flip の dim ${dim} が入力 rank ${source.length} の外`,
        );
      }
      // 反転は shape を変えない（恒等 shape 規則）。
      return sole([...source]);
    }
    case "symPrefixSlice": {
      const source = inputShapes[0];
      // 出力 rank は入力と同じ（各軸の先頭を切り出すだけ）なので入力側だけ見れば足りる。
      assertStridedRank(source.length, "sym_prefix_slice の入力", where);
      const { sym, slices } = symPrefixSliceAttrs(context.attrs ?? {}, where);
      const bindings = context.bindings;
      if (bindings === undefined || !Object.hasOwn(bindings, sym)) {
        throw new OpContractError(
          `${where}: sym_prefix_slice の sym '${sym}' が束縛されていない（ShapeContext.bindings）`,
        );
      }
      const bound = bindings[sym];
      const out = [...source];
      for (const slice of slices) {
        if (slice.dim >= source.length) {
          throw new OpContractError(
            `${where}: sym_prefix_slice の dim ${slice.dim} が入力 rank ${source.length} の外`,
          );
        }
        const length = slice.coeff * bound + slice.offset;
        // MUST: 定数側（Tmax 形）を超える prefix を許さない。超えた分は定数バッファの
        // 範囲外読み出しになり、GPU では例外なしに隣の値が出る。
        if (length > source[slice.dim]) {
          throw new OpContractError(
            `${where}: sym_prefix_slice の prefix 長 ${slice.coeff}·${sym}+${slice.offset}=${length} が定数次元 ${
              source[slice.dim]
            } を超える（Tmax 超過）`,
          );
        }
        out[slice.dim] = length;
      }
      return sole(out);
    }
    case "linear": {
      const [x, weight, bias] = inputShapes;
      if (x.length < 1 || weight.length !== 2 || bias.length !== 1) {
        throw new OpContractError(
          `${where}: linear は x[…,in] × W[out,in] + b[out]（rank ≥ 1 / 2 / 1）: [${
            x.join(",")
          }] / [${weight.join(",")}] / [${bias.join(",")}]`,
        );
      }
      const [out, inFeatures] = weight;
      if (x[x.length - 1] !== inFeatures) {
        throw new OpContractError(
          `${where}: linear の入力特徴数が不一致 [${x.join(",")}] × [${weight.join(",")}]`,
        );
      }
      if (bias[0] !== out) {
        throw new OpContractError(
          `${where}: linear の bias 長 ${bias[0]} が出力特徴数 ${out} と違う`,
        );
      }
      return sole([...x.slice(0, -1), out]);
    }
    case "layerNorm": {
      const [x, weight, bias] = inputShapes;
      const { normalizedShape } = layerNormAttrs(context.attrs ?? {}, where);
      // 契約は「最終次元のみ」（attrs 検査済み）。ここでは実 shape の末尾との一致を見る。
      if (x.length < 1 || x[x.length - 1] !== normalizedShape[0]) {
        throw new OpContractError(
          `${where}: layer_norm の normalized_shape [${normalizedShape.join(",")}] が入力 [${
            x.join(",")
          }] の最終次元と違う`,
        );
      }
      for (const [name, shape] of [["weight", weight], ["bias", bias]] as const) {
        if (shape.length !== 1 || shape[0] !== normalizedShape[0]) {
          throw new OpContractError(
            `${where}: layer_norm の ${name} [${shape.join(",")}] が normalized_shape [${
              normalizedShape.join(",")
            }] と違う`,
          );
        }
      }
      return sole([...x]);
    }
    case "rmsNorm": {
      const [x, weight] = inputShapes;
      // MUST: eps はここでも引く（attrs スキーマを通らない経路 — CPU 参照の直呼び — でも
      // 値域が効くようにする。unary の scalarParamValues と同じ役割）。
      rmsNormEps(context.attrs ?? {}, where);
      if (x.length < 1) {
        throw new OpContractError(`${where}: rms_norm の入力は rank 1 以上（最終次元が要る）`);
      }
      const dim = x[x.length - 1];
      // MUST: 長さ 0 の軸は縮約できない（二乗和 0 / 要素数 0 で mean が 0/0 = NaN になる —
      // softmax と同じ絞り方）。
      if (dim === 0) {
        throw new OpContractError(`${where}: rms_norm は長さ 0 の軸を正規化できない`);
      }
      // 正規化長の正本は **weight の長さ**（attrs に normalized_shape の欄を作らない — ADR 0017）。
      if (weight.length !== 1 || weight[0] !== dim) {
        throw new OpContractError(
          `${where}: rms_norm の weight [${weight.join(",")}] が入力 [${
            x.join(",")
          }] の最終次元長 ${dim} の rank1 でない`,
        );
      }
      return sole([...x]);
    }
    case "conv2d": {
      const [x, weight, bias] = inputShapes;
      const { stride, padding, dilation, groups } = conv2dAttrs(context.attrs ?? {}, where);
      if (x.length !== 4 || weight.length !== 4 || bias.length !== 1) {
        throw new OpContractError(
          `${where}: conv2d は x[B,Cin,H,W] / W[Cout,Cin/groups,Kh,Kw] / b[Cout]（rank 4 / 4 / 1）: [${
            x.join(",")
          }] / [${weight.join(",")}] / [${bias.join(",")}]`,
        );
      }
      const [, channelsIn] = x;
      const [channelsOut, weightIn] = weight;
      // MUST: conv1d と同じ規律 — 割り切れない形はカーネルの `Cin/groups` が切り捨てになり、
      // 読む入力チャネル帯が黙ってずれる。
      if (channelsIn % groups !== 0 || channelsOut % groups !== 0) {
        throw new OpContractError(
          `${where}: conv2d の groups ${groups} が Cin ${channelsIn} / Cout ${channelsOut} を割り切らない`,
        );
      }
      // MUST: 重みは **[Cout, Cin/groups, Kh, Kw]**。要素数が合う取り違え（[Cin/groups, Cout,
      // Kh, Kw] や Kh/Kw の入れ替え）は shape 検査を素通りしうるので、テストは
      // Cin ≠ Cout・Kh ≠ Kw の非対称形で固定する（conv_transpose1d の教訓 — ADR 0017）。
      if (weightIn !== channelsIn / groups) {
        throw new OpContractError(
          `${where}: conv2d の重みは [Cout, Cin/groups, Kh, Kw]（Cin/groups = ${
            channelsIn / groups
          }）のはずが [${weight.join(",")}]（x は [${x.join(",")}] / groups ${groups}）`,
        );
      }
      if (bias[0] !== channelsOut) {
        throw new OpContractError(
          `${where}: conv2d の bias 長 ${bias[0]} が出力チャネル ${channelsOut} と違う`,
        );
      }
      // 空間軸は H / W で独立に同じ一般形を適用する（axis は診断の主語）。
      const spatial = (axis: 0 | 1, name: string): number => {
        const length = x[2 + axis];
        const kernel = weight[2 + axis];
        const span = length + 2 * padding[axis] - dilation[axis] * (kernel - 1) - 1;
        if (span < 0) {
          throw new OpContractError(
            `${where}: conv2d の入力 ${name} ${length}（padding ${padding[axis]}）が dilation ${
              dilation[axis]
            } 込みのカーネル張り ${dilation[axis] * (kernel - 1) + 1} に足りない`,
          );
        }
        return Math.floor(span / stride[axis]) + 1;
      };
      return sole([x[0], channelsOut, spatial(0, "H"), spatial(1, "W")]);
    }
    // safe_softmax は shape 規則も attrs も softmax と同一（違いは空行の値だけ — ADR 0044）。
    case "softmax":
    case "safeSoftmax": {
      const shape = inputShapes[0];
      const dim = softmaxDim(context.attrs ?? {}, where);
      // MUST: 一般 dim を「そのうち実装する」として受理しない。最終次元以外は行カーネルの
      // 前提（縮約軸が連続）が崩れ、通せば黙って別の軸を畳む。
      if (shape.length < 1 || dim !== shape.length - 1) {
        throw new OpContractError(
          `${where}: ${found.name} は最終次元のみ（attrs.dim=${dim} / 入力 [${shape.join(",")}]）`,
        );
      }
      if (shape[shape.length - 1] === 0) {
        // 空軸の softmax は amax の identity が定義できない（行 reduce と同じ理由）。
        throw new OpContractError(`${where}: ${found.name} は長さ 0 の軸を縮約できない`);
      }
      return sole([...shape]);
    }
    case "attention": {
      const [q, k, v, mask] = inputShapes;
      // MUST: scale はここでも引く（attrs スキーマを通らない CPU 参照の直呼びでも値域を効かせる
      // ため。rms_norm の eps / unary の scalarParamValues と同じ役割）。
      attentionScale(context.attrs ?? {}, where);
      const show = `[${q.join(",")}] / [${k.join(",")}] / [${v.join(",")}]`;
      if (q.length !== 4 || k.length !== 4 || v.length !== 4) {
        throw new OpContractError(
          `${where}: attention は q[B,H,M,D] / k[B,Hkv,N,D] / v[B,Hkv,N,D] の rank-4 のみ: ${show}`,
        );
      }
      // MUST: B は**完全一致**（積だけを見ると B/H の取り違えが素通りする — カーネルは B·H を
      // 1 本の軸に畳むので、値にも出ない形が作れる）。
      if (q[0] !== k[0] || q[0] !== v[0]) {
        throw new OpContractError(`${where}: attention の軸 0（B）が不一致 ${show}`);
      }
      // MUST: k / v の Hkv も**完全一致**（GQA で緩めるのは q との関係だけ — ADR 0067 決定 1 は
      // 「k/v 間の Hkv 一致・D 3 者同一・N=0 拒否は取り違え検出線としてそのまま維持」）。
      if (k[1] !== v[1]) {
        throw new OpContractError(`${where}: attention の Hkv（k / v の軸 1）が不一致 ${show}`);
      }
      // MUST: `Hkv ≥ 1` は下の等値短絡より**前**に見る — `(H,Hkv) = (0,0)` は `q[1] !== k[1]` を
      // 満たさないので整除枝に落ちず、「head 軸を丸ごと落とした IR」が素通りする。`H = 0` 単独は
      // `Hkv ≥ 1` とのペアになるので下の `H ≥ Hkv` 枝が落とす（Python 側 `_attention` と鏡像）。
      if (k[1] < 1) {
        throw new OpContractError(
          `${where}: attention の Hkv ${k[1]} が正でない（H は Hkv の正の整数倍 — GQA は` +
            ` H % Hkv == 0 かつ H ≥ Hkv ≥ 1・ADR 0067 決定 1）${show}`,
        );
      }
      // GQA = **整除 broadcast**（ADR 0067 決定 1）。`H % Hkv == 0` かつ `H ≥ Hkv` だけを受理し、
      // `r = H / Hkv` は導出値（attrs 欄を作らない）。Hkv = 1 の MQA も同式で表す。
      // MUST: `H < Hkv` を別条件で弾く — `H = 0` は `0 % Hkv == 0` を満たすので、整除だけを見ると
      // 「H を丸ごと落とした IR」が素通りする（`Hkv = 0` は上の `Hkv ≥ 1` 枝が先に落とす）。
      // broadcast の向きは常に kv → q で、q 側を増やす形は語彙に無い。
      if (q[1] !== k[1] && (q[1] < k[1] || q[1] % k[1] !== 0)) {
        throw new OpContractError(
          `${where}: attention の H ${q[1]} が Hkv ${k[1]} の正の整数倍でない（GQA は` +
            ` H % Hkv == 0 かつ H ≥ Hkv — ADR 0067 決定 1）${show}`,
        );
      }
      // MUST: D は 3 者とも同じ（v 側だけ別の長さを許すと、取り違えが要素数で捕まらない）。
      if (q[3] !== k[3] || q[3] !== v[3]) {
        throw new OpContractError(`${where}: attention の D（軸 3）が不一致 ${show}`);
      }
      if (k[2] !== v[2]) {
        throw new OpContractError(`${where}: attention の N（k / v の軸 2）が不一致 ${show}`);
      }
      // 空軸の softmax は amax の identity が定義できない（softmax / 行 reduce と同じ理由）。
      if (k[2] === 0) {
        throw new OpContractError(`${where}: attention は長さ 0 の N を縮約できない ${show}`);
      }
      if (mask !== undefined) {
        // MUST: mask は **[1,1,M,N] ちょうど**。B·H へ broadcast する加算項なので、
        // `[B,1,M,N]` / `[1,H,M,N]` のような「一部の軸だけ実体を持つ」形を通すと、
        // カーネル（B·H を 1 軸に畳んで先頭バッチの mask を全バッチへ配る）が黙って
        // 別のバッチの mask を適用する。広げるなら添字算術とセットで契約を改版する。
        const shown = `${show} + mask [${mask.join(",")}]`;
        if (mask.length !== 4 || mask[0] !== 1 || mask[1] !== 1) {
          throw new OpContractError(
            `${where}: attention の mask は [1,1,M,N] の rank-4 のみ（B / H は broadcast 固定）: ${shown}`,
          );
        }
        if (mask[2] !== q[2] || mask[3] !== k[2]) {
          throw new OpContractError(
            `${where}: attention の mask の M / N が q / k と不一致 ${shown}`,
          );
        }
      }
      return sole([...q]);
    }
    case "embedding": {
      const [weight, index] = inputShapes;
      if (weight.length !== 2) {
        throw new OpContractError(
          `${where}: embedding の weight は rank-2 [V,H]: [${weight.join(",")}]`,
        );
      }
      if (index.length < 1) {
        throw new OpContractError(
          `${where}: embedding の index は rank 1 以上（スカラ添字は無い）`,
        );
      }
      // 添字の値域 0 <= index < V は実行時データ依存なので shape 契約では見ない
      // （範囲外の扱いは src/kernels/embedding.ts の裁定 — GPU は NaN 汚染 / CPU 参照は throw）。
      return sole([...index, weight[1]]);
    }
    case "maskedFill": {
      const [x, mask] = inputShapes;
      // 出力は x と同形。mask も右詰め broadcast の stride を組むので rank 1 以上が要る
      // （rank 0 の mask は契約を素通りして codegen 層で落ちる形になっていた）。
      assertStridedRank(x.length, "masked_fill の x", where);
      assertStridedRank(mask.length, "masked_fill の mask", where);
      // MUST: 出力は**常に x と同形**（mask 側は右詰め broadcast で読むだけ）。broadcastShapes を
      // そのまま使うと mask が x を広げる形（mask [4] × x [1]）まで通り、埋め値が本来無い要素へ
      // 漏れる。ここは「mask が x に収まる」ことだけを見る非対称な検査。
      if (mask.length > x.length) {
        throw new OpContractError(
          `${where}: masked_fill の mask rank ${mask.length} が x rank ${x.length} を超える（mask は右詰め broadcast のみ）`,
        );
      }
      const offset = x.length - mask.length;
      mask.forEach((extent, index) => {
        if (extent !== 1 && extent !== x[offset + index]) {
          throw new OpContractError(
            `${where}: masked_fill の mask [${mask.join(",")}] が x [${
              x.join(",")
            }] へ右詰め broadcast できない（次元 ${index}）`,
          );
        }
      });
      return sole([...x]);
    }
    case "conv1d": {
      const [x, weight, bias] = inputShapes;
      const { stride, padding, dilation, groups } = conv1dAttrs(context.attrs ?? {}, where);
      if (x.length !== 3 || weight.length !== 3 || bias.length !== 1) {
        throw new OpContractError(
          `${where}: conv1d は x[B,Cin,L] / W[Cout,Cin/groups,K] / b[Cout]（rank 3 / 3 / 1）: [${
            x.join(",")
          }] / [${weight.join(",")}] / [${bias.join(",")}]`,
        );
      }
      const [, channelsIn, length] = x;
      const [channelsOut, weightIn, kernel] = weight;
      // MUST: グループ分割は両側で割り切れることが契約（depthwise は groups = Cin = Cout）。
      // 割り切れない形を通すとカーネルの `Cin/groups` が切り捨てになり、読む入力チャネルが
      // 黙ってずれる。
      if (channelsIn % groups !== 0 || channelsOut % groups !== 0) {
        throw new OpContractError(
          `${where}: conv1d の groups ${groups} が Cin ${channelsIn} / Cout ${channelsOut} を割り切らない`,
        );
      }
      if (weightIn !== channelsIn / groups) {
        throw new OpContractError(
          `${where}: conv1d の重みは [Cout, Cin/groups, K]（Cin/groups = ${
            channelsIn / groups
          }）のはずが [${weight.join(",")}]（x は [${x.join(",")}] / groups ${groups}）`,
        );
      }
      if (bias[0] !== channelsOut) {
        throw new OpContractError(
          `${where}: conv1d の bias 長 ${bias[0]} が出力チャネル ${channelsOut} と違う`,
        );
      }
      // dilation の一般形。K=1 でも d·(K−1) = 0 なので従来式と一致する。
      const span = length + 2 * padding - dilation * (kernel - 1) - 1;
      if (span < 0) {
        throw new OpContractError(
          `${where}: conv1d の入力長 ${length}（padding ${padding}）が dilation ${dilation} 込みのカーネル張り ${
            dilation * (kernel - 1) + 1
          } に足りない`,
        );
      }
      return sole([x[0], channelsOut, Math.floor(span / stride) + 1]);
    }
    case "convTranspose1d": {
      const [x, weight, bias] = inputShapes;
      const { stride, padding } = convTranspose1dAttrs(context.attrs ?? {}, where);
      if (x.length !== 3 || weight.length !== 3 || bias.length !== 1) {
        throw new OpContractError(
          `${where}: conv_transpose1d は x[B,Cin,L] / W[Cin,Cout,K] / b[Cout]（rank 3 / 3 / 1）: [${
            x.join(",")
          }] / [${weight.join(",")}] / [${bias.join(",")}]`,
        );
      }
      const [, channelsIn, length] = x;
      const [weightIn, channelsOut, kernel] = weight;
      // MUST: 重みは conv1d と**転置**の [Cin, Cout, K]。取り違えても要素数が合う形
      // （Cin == Cout）が作れるので、テストは非対称チャネル数で固定する（ADR 0015）。
      if (weightIn !== channelsIn) {
        throw new OpContractError(
          `${where}: conv_transpose1d の重みは [Cin, Cout, K]（Cin = ${channelsIn}）のはずが [${
            weight.join(",")
          }]（x は [${x.join(",")}]）`,
        );
      }
      if (bias[0] !== channelsOut) {
        throw new OpContractError(
          `${where}: conv_transpose1d の bias 長 ${bias[0]} が出力チャネル ${channelsOut} と違う`,
        );
      }
      // MUST: 受理するのは出力長がちょうど L·stride になる形だけ（`2p == K − s`）。一般形
      // `(L−1)·s − 2p + K` は記号長 L の一次式にはなるが、実測（dec の ups 5 本）が全て
      // この形なので、広げるのは需要が出てから — 黙って一般形を通さない（ADR 0015）。
      if (2 * padding !== kernel - stride) {
        throw new OpContractError(
          `${where}: conv_transpose1d は 2·padding == K − stride の形のみ受理（K ${kernel} / stride ${stride} / padding ${padding} — 出力長が L·stride にならない）`,
        );
      }
      return sole([x[0], channelsOut, length * stride]);
    }
    case "deformConv2d": {
      const [x, weight, offset, mask, bias] = inputShapes;
      const { padding } = deformConv2dAttrs(context.attrs ?? {}, where);
      if (
        x.length !== 4 || weight.length !== 4 || offset.length !== 4 || mask.length !== 4 ||
        bias.length !== 1
      ) {
        throw new OpContractError(
          `${where}: deform_conv2d は x[B,Cin,H,W] / W[Cout,Cin,Kh,Kw] / offset[B,2·Kh·Kw,Hout,Wout] / mask[B,Kh·Kw,Hout,Wout] / b[Cout]（rank 4/4/4/4/1）: [${
            x.join(",")
          }] / [${weight.join(",")}] / [${offset.join(",")}] / [${mask.join(",")}] / [${
            bias.join(",")
          }]`,
        );
      }
      const [batch, channelsIn] = x;
      const [channelsOut, weightIn, kernelH, kernelW] = weight;
      // MUST: groups の欄が無い = 1 固定なので、重みの第 2 軸は Cin そのもの。取り違え
      // （[Cin, Cout, Kh, Kw]）は要素数が合う形が作れるので、テストは Cin ≠ Cout で固定する。
      if (weightIn !== channelsIn) {
        throw new OpContractError(
          `${where}: deform_conv2d の重みは [Cout, Cin, Kh, Kw]（Cin = ${channelsIn}）のはずが [${
            weight.join(",")
          }]（x は [${x.join(",")}]）`,
        );
      }
      if (bias[0] !== channelsOut) {
        throw new OpContractError(
          `${where}: deform_conv2d の bias 長 ${bias[0]} が出力チャネル ${channelsOut} と違う`,
        );
      }
      // 空間長は stride 1 / dilation 1 の一般形（欄が無い = 1 固定）。
      const spatial = (axis: 0 | 1, name: string): number => {
        const kernel = weight[2 + axis];
        const length = x[2 + axis] + 2 * padding[axis] - (kernel - 1);
        if (length < 1) {
          throw new OpContractError(
            `${where}: deform_conv2d の入力 ${name} ${x[2 + axis]}（padding ${
              padding[axis]
            }）がカーネル張り ${kernel} に足りない`,
          );
        }
        return length;
      };
      const heightOut = spatial(0, "H");
      const widthOut = spatial(1, "W");
      const taps = kernelH * kernelW;
      // MUST: offset / mask は**突き合わせるだけ**（出力形の正本は x + weight + attrs）。
      // offset 側から Hout を取ると「offset だけ形が違う IR」が素通りする。
      // offset_groups の欄が無い = 1 固定なので、チャネル長は 2·Kh·Kw / Kh·Kw ちょうど。
      for (
        const [name, shape, channels] of [
          ["offset", offset, 2 * taps],
          ["mask", mask, taps],
        ] as const
      ) {
        if (
          shape[0] !== batch || shape[1] !== channels || shape[2] !== heightOut ||
          shape[3] !== widthOut
        ) {
          throw new OpContractError(
            `${where}: deform_conv2d の ${name} は [${batch},${channels},${heightOut},${widthOut}]（offset_groups = 1）のはずが [${
              shape.join(",")
            }]`,
          );
        }
      }
      return sole([batch, channelsOut, heightOut, widthOut]);
    }
    case "upsampleBilinear2d": {
      const x = inputShapes[0];
      const { outputSize } = upsampleBilinear2dAttrs(context.attrs ?? {}, where);
      if (x.length !== 4) {
        throw new OpContractError(
          `${where}: upsample_bilinear2d は x[B,C,H,W]（rank 4）のみ: [${x.join(",")}]`,
        );
      }
      // MUST: 長さ 0 の空間軸を通さない。scale は `(in − 1) / (out − 1)` なので in = 0 は
      // 負の scale になり、読み出しが入力の外へ出る（GPU では例外なしに隣の値が出る）。
      for (const [name, axis] of [["H", 2], ["W", 3]] as const) {
        if (x[axis] === 0) {
          throw new OpContractError(
            `${where}: upsample_bilinear2d は長さ 0 の空間軸 ${name} を補間できない（[${
              x.join(",")
            }]）`,
          );
        }
      }
      return sole([x[0], x[1], outputSize[0], outputSize[1]]);
    }
    case "gruScan": {
      const [gi, initial, weight, bias] = inputShapes;
      if (gi.length !== 3 || initial.length !== 2 || weight.length !== 2 || bias.length !== 1) {
        throw new OpContractError(
          `${where}: ${found.name} は gi[T,N,3H] / h0[N,H] / W_hh[3H,H] / b_hh[3H]（rank 3/2/2/1）: [${
            gi.join(",")
          }] / [${initial.join(",")}] / [${weight.join(",")}] / [${bias.join(",")}]`,
        );
      }
      const [time, batch, gates] = gi;
      const [initialBatch, hidden] = initial;
      // MUST: 隠れ幅の正本は h0 の最終次元 1 か所（gi / W_hh / b_hh とは**突き合わせるだけ**）。
      // 同じ事実を 2 か所から取ると、3H と H の取り違えが素通りする形が作れる。
      if (batch !== initialBatch) {
        throw new OpContractError(
          `${where}: ${found.name} のバッチが gi [${gi.join(",")}] と h0 [${
            initial.join(",")
          }] で不一致`,
        );
      }
      if (gates !== 3 * hidden) {
        throw new OpContractError(
          `${where}: ${found.name} の gi の最終次元 ${gates} が 3·H（H = ${hidden}）でない（ゲートは r / z / n の 3 本）`,
        );
      }
      if (weight[0] !== 3 * hidden || weight[1] !== hidden) {
        throw new OpContractError(
          `${where}: ${found.name} の W_hh は [3H, H] = [${3 * hidden},${hidden}] のはずが [${
            weight.join(",")
          }]`,
        );
      }
      if (bias[0] !== 3 * hidden) {
        throw new OpContractError(
          `${where}: ${found.name} の b_hh 長 ${bias[0]} が 3·H = ${3 * hidden} と違う`,
        );
      }
      return sole([time, batch, hidden]);
    }
    case "permute": {
      const source = inputShapes[0];
      // 出力 rank は入力と同じ（並べ替えるだけ）なので入力側だけ見れば足りる。
      assertStridedRank(source.length, "permute の入力", where);
      const dims = permuteDims(context.attrs ?? {}, where);
      if (dims.length !== source.length) {
        throw new OpContractError(
          `${where}: permute の dims [${dims.join(",")}] が入力 rank ${source.length} と違う`,
        );
      }
      return sole(dims.map((dim) => {
        if (dim >= source.length) {
          throw new OpContractError(
            `${where}: permute の dims に入力 rank ${source.length} 外の軸 ${dim} がある`,
          );
        }
        return source[dim];
      }));
    }
  }
};
