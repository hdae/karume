/**
 * strided 実体化コピーカーネルの WGSL 生成（ADR 0011）。
 *
 * 入力を `(offset, strides[STRIDED_RANK])` で読み、**出力は常に連続**。permute は軸を
 * 並べ替えた stride、expand は複製する軸の stride 0 で表現する — レイアウト op の実体化を
 * 1 カーネル族に閉じ込め、elementwise / reduce / matmul の codegen を連続入力前提のまま
 * 温存するための形（ADR 0011 の案 B）。
 *
 * MUST: 生成は決定的 — 同一キーからバイト単位で同一の WGSL が出ること（キーと WGSL を
 * 同じ正準化から組み立てる）。**要素型はキーに含める**（ADR 0009）— bool マスクの expand と
 * i32 添字の expand が実測に存在し、載せないと別要素型の dispatch に同じパイプラインが
 * 割り当たってビット列の読み替えが例外なしに通る。
 * MUST: grid-stride ループ前提。1 次元の dispatch 上限（仕様既定 65535）は実モデルで超える。
 *
 * rank は **STRIDED_RANK に固定**して 1 dtype = 1 パイプラインにする。実 rank が足りない
 * ぶんは呼び出し側が左詰めで 1（dims）と 0（strides）に埋める — 右詰め broadcast と同じ
 * 埋め方で、余った先行軸の座標は常に 0 になり読み出し位置に寄与しない。
 *
 * 本ファイルは**読み族と書き族の 2 つ**を持つ（ADR 0014）。書き族（cat）は読み族の双対 —
 * 入力を連続で読み、出力へ `(offset, strides)` で書く。同じ params レイアウトと同じ決定性
 * キー規約を共有し、走査の向きだけが逆になる。
 */

import { type IrDtype, SEMANTIC_DTYPES } from "../format/ir.ts";
import { CodegenError } from "./errors.ts";
import { assertU32Params } from "./params.ts";

/** DeBERTa front の値は全て rank ≤ 4（ADR 0011）。rank ≥ 5 は rank 下げ正規化で先に潰す。 */
export const STRIDED_RANK = 4;
export const STRIDED_WORKGROUP_SIZE = 256;

export type StridedSpec = {
  /** 要素の意味論 dtype（入力・出力とも同型 — レイアウト op は値に触らない）。 */
  readonly dtype: IrDtype;
};

/** 意味論 dtype → WGSL のスカラ型。bool の格納は u32 の 0 / 1（ADR 0009）。 */
const WGSL_SCALAR: Readonly<Record<IrDtype, string>> = {
  f32: "f32",
  i32: "i32",
  bool: "u32",
};

const canonicalize = (spec: StridedSpec): IrDtype => {
  const found = SEMANTIC_DTYPES.find((dtype) => dtype === spec.dtype);
  if (found === undefined) {
    throw new CodegenError(`strided codegen: 要素型 '${spec.dtype}' は意味論 dtype でない`);
  }
  return found;
};

/**
 * MUST: WGSL に埋まる生成パラメータは全てキーに載せる（rank・workgroup サイズ・要素型）。
 * 載せずに定数を変えると、旧定数でコンパイル済みのパイプラインが同じキーで再利用される。
 */
export const stridedKey = (spec: StridedSpec): string =>
  `strided:v1:${canonicalize(spec)}:r${STRIDED_RANK}:wg${STRIDED_WORKGROUP_SIZE}`;

/**
 * params のレイアウト（storage, u32 配列）:
 * `[0]=要素数 n, [1..STRIDED_RANK]=出力 dims, 続けて入力 strides を同数, 末尾に offset`。
 * uniform ではなく storage で渡すのは、この族に workgroupBarrier が無く一様性解析の制約を
 * 受けないため（elementwise と同じ理由）。
 */
export const stridedWgsl = (spec: StridedSpec): string => {
  const scalar = WGSL_SCALAR[canonicalize(spec)];
  const dimAt = (d: number): number => 1 + d;
  const strideAt = (d: number): number => 1 + STRIDED_RANK + d;
  const offsetAt = 1 + 2 * STRIDED_RANK;

  // 出力の平坦添字を最終軸から divmod で分解する（出力は常に連続）。
  const decode: string[] = ["    var rem = i;"];
  for (let d = STRIDED_RANK - 1; d >= 1; d -= 1) {
    decode.push(`    let c${d} = rem % params[${dimAt(d)}u]; rem = rem / params[${dimAt(d)}u];`);
  }
  decode.push("    let c0 = rem;");

  const terms = Array.from(
    { length: STRIDED_RANK },
    (_, d) => `c${d} * params[${strideAt(d)}u]`,
  );

  return [
    `// karume strided copy (rank ${STRIDED_RANK}, ${scalar}, generated)`,
    "@group(0) @binding(0) var<storage, read> params: array<u32>;",
    `@group(0) @binding(1) var<storage, read> src: array<${scalar}>;`,
    `@group(0) @binding(2) var<storage, read_write> out: array<${scalar}>;`,
    "",
    `@compute @workgroup_size(${STRIDED_WORKGROUP_SIZE})`,
    "fn main(",
    "  @builtin(global_invocation_id) gid: vec3<u32>,",
    "  @builtin(num_workgroups) nwg: vec3<u32>,",
    ") {",
    "  let n = params[0u];",
    `  let stride = nwg.x * ${STRIDED_WORKGROUP_SIZE}u;`,
    "  var i = gid.x;",
    "  while (i < n) {",
    ...decode,
    `    let src_index = params[${offsetAt}u] + ${terms.join(" + ")};`,
    "    out[i] = src[src_index];",
    "    i = i + stride;",
    "  }",
    "}",
    "",
  ].join("\n");
};

const assertRank = (shape: readonly number[], what: string): void => {
  if (shape.length < 1 || shape.length > STRIDED_RANK) {
    throw new CodegenError(
      `strided params: ${what} の rank ${shape.length} は 1..${STRIDED_RANK} の外`,
    );
  }
};

/** 連続レイアウトの stride（右から running product）。 */
const contiguousStrides = (shape: readonly number[]): number[] => {
  const strides = new Array<number>(shape.length).fill(0);
  let running = 1;
  for (let d = shape.length - 1; d >= 0; d -= 1) {
    strides[d] = running;
    running *= shape[d];
  }
  return strides;
};

/**
 * permute の読み出し stride。`dims[d]` = 出力の次元 d が取る入力の次元番号なので、
 * 出力の次元 d を 1 進めると入力側は `dims[d]` 軸を 1 進む。
 * 並べ替えの妥当性（全単射・rank 一致）は契約表が済ませている前提。
 */
export const permuteSrcStrides = (
  srcShape: readonly number[],
  dims: readonly number[],
): number[] => {
  assertRank(srcShape, "permute の入力");
  if (dims.length !== srcShape.length) {
    throw new CodegenError(
      `strided params: permute の dims [${dims.join(",")}] が入力 rank ${srcShape.length} と違う`,
    );
  }
  const contiguous = contiguousStrides(srcShape);
  return dims.map((dim) => {
    const stride = contiguous[dim];
    if (stride === undefined) {
      throw new CodegenError(`strided params: permute の dims に軸 ${dim} が入っている`);
    }
    return stride;
  });
};

/** {@link expandSrcStrides} の診断の主語（既定は expand そのもの）。 */
export type BroadcastLabels = {
  readonly src: string;
  readonly out: string;
};

const EXPAND_LABELS: BroadcastLabels = { src: "expand の入力", out: "expand の出力" };

/**
 * 右詰め broadcast の読み出し stride。複製する次元（入力の長さが 1）は **stride 0**、
 * rank が増えたぶんの先行軸も stride 0。契約側（長さ 1 の次元しか拡張しない）は済ませている前提。
 *
 * MUST: 診断の主語は呼び出し側が差し替える。この規則は masked_fill の mask 読みでも共有する
 * （規則を 2 箇所に書けば必ず割れる）が、そのとき「expand の入力の rank が…」と出ると、
 * グラフに expand が 1 本も無いのに expand を疑わせる誤誘導になる。
 */
export const expandSrcStrides = (
  srcShape: readonly number[],
  outShape: readonly number[],
  labels: BroadcastLabels = EXPAND_LABELS,
): number[] => {
  assertRank(srcShape, labels.src);
  assertRank(outShape, labels.out);
  if (outShape.length < srcShape.length) {
    throw new CodegenError(
      `strided params: ${labels.out} rank ${outShape.length} が ${labels.src} rank ${srcShape.length} 未満`,
    );
  }
  const contiguous = contiguousStrides(srcShape);
  const offset = outShape.length - srcShape.length;
  return outShape.map((_, d) =>
    d < offset || srcShape[d - offset] === 1 ? 0 : contiguous[d - offset]
  );
};

/**
 * slice の読み出し stride（ADR 0014）。読み方は**入力の連続 stride そのもの**で、切り出しは
 * offset だけで表す（ADR 0011 が予告した「可変点は params の offset 1 語」）。
 */
export const sliceSrcStrides = (srcShape: readonly number[]): number[] => {
  assertRank(srcShape, "slice の入力");
  return contiguousStrides(srcShape);
};

/**
 * slice の読み出し先頭位置 = `start × 当該軸の入力 stride`。
 *
 * MUST: stride は**入力**の shape から組む（{@link sliceSrcStrides} と同じ表）。出力側の
 * 縮んだ shape から組むと切り出し軸より前の送り幅が縮み、2 行目以降が別の行を読む
 * （sym_prefix_slice の MUST と同型の罠）。
 */
export const sliceSrcOffset = (
  srcShape: readonly number[],
  dim: number,
  start: number,
): number => {
  assertRank(srcShape, "slice の入力");
  if (!Number.isSafeInteger(dim) || dim < 0 || dim >= srcShape.length) {
    throw new CodegenError(
      `strided params: slice の dim ${dim} が入力 rank ${srcShape.length} の外`,
    );
  }
  if (!Number.isSafeInteger(start) || start < 0 || start > srcShape[dim]) {
    throw new CodegenError(
      `strided params: slice の start ${start} が軸 ${dim}（長さ ${srcShape[dim]}）の外`,
    );
  }
  return start * contiguousStrides(srcShape)[dim];
};

/**
 * sym_prefix_slice の読み出し stride（ADR 0010）。入力は **Tmax で焼いた定数の全長**で、
 * 出力はその先頭を切り出した連続バッファ。offset は 0 で、読み方は入力の連続 stride そのもの。
 *
 * MUST: stride は**入力（Tmax 形）の shape** から組む。束縛後の出力 shape から組むと行の
 * 送り幅が縮み、2 行目以降が定数表の別の行を読む（T = Tmax のときだけ正しく、それ以外で
 * 静かに壊れる形）。
 */
export const prefixSliceSrcStrides = (srcShape: readonly number[]): number[] => {
  assertRank(srcShape, "sym_prefix_slice の入力");
  return contiguousStrides(srcShape);
};

/**
 * params バッファの中身を組み立てる。実 rank に足りないぶんは**左詰めで**
 * dims=1 / strides=0 に埋める（右詰め broadcast と同じ埋め方 — 余った先行軸の座標は
 * 常に 0 になり読み出し位置に寄与しない）。
 *
 * 読み族・書き族で**同じレイアウト**（走査する側の dims と、相手側の strides + offset）を
 * 使う — 走査の向きが逆になるだけで params の意味構造は双対そのものだから。
 */
const packStridedParams = (
  dims: readonly number[],
  strides: readonly number[],
  offset: number,
  labels: { readonly shape: string; readonly rank: string },
): Uint32Array<ArrayBuffer> => {
  assertRank(dims, labels.shape);
  if (strides.length !== dims.length) {
    throw new CodegenError(
      `strided params: stride 本数 ${strides.length} が${labels.rank} rank ${dims.length} と違う`,
    );
  }
  const n = dims.reduce((count, dim) => count * dim, 1);
  assertU32Params("strided params", {
    ...Object.fromEntries(dims.map((dim, d) => [`dims[${d}]`, dim])),
    ...Object.fromEntries(strides.map((stride, d) => [`strides[${d}]`, stride])),
    n,
    offset,
  });
  const pad = STRIDED_RANK - dims.length;
  const params = new Uint32Array(2 + 2 * STRIDED_RANK);
  params[0] = n;
  for (let d = 0; d < STRIDED_RANK; d += 1) {
    params[1 + d] = d < pad ? 1 : dims[d - pad];
    params[1 + STRIDED_RANK + d] = d < pad ? 0 : strides[d - pad];
  }
  params[1 + 2 * STRIDED_RANK] = offset;
  return params;
};

export const stridedParams = (
  outShape: readonly number[],
  srcStrides: readonly number[],
  offset: number,
): Uint32Array<ArrayBuffer> =>
  packStridedParams(outShape, srcStrides, offset, { shape: "strided の出力", rank: "出力" });

// ---- strided 書きコピー族（読み族の双対 — ADR 0014）----------------------

export const STRIDED_WRITE_WORKGROUP_SIZE = 256;

/**
 * MUST: 読み族と**同じ決定性キー規約**（要素型・rank・workgroup サイズを載せる）。族名だけを
 * 変えて他を揃えるのは、双方が同じ可変点しか持たないことの表明でもある。
 */
export const stridedWriteKey = (spec: StridedSpec): string =>
  `strided_write:v1:${canonicalize(spec)}:r${STRIDED_RANK}:wg${STRIDED_WRITE_WORKGROUP_SIZE}`;

/**
 * strided **書き**コピーの WGSL（cat の実行 — ADR 0014）。入力を連続で読み、出力へ
 * `(offset, strides[STRIDED_RANK])` で書く（{@link stridedWgsl} の双対）。
 *
 * params のレイアウトは読み族と同じ:
 * `[0]=要素数 n, [1..STRIDED_RANK]=**入力** dims, 続けて**出力** strides を同数, 末尾に offset`。
 *
 * MUST: 1 dispatch が書くのは出力の部分領域だけ。**ノード単位で出力全域を覆う**ことは
 * 呼び出し側（executor の cat）が offset の総和と出力の軸長を突き合わせて担保する
 * （full-write 不変条件 — ADR 0004 / 0014）。
 */
export const stridedWriteWgsl = (spec: StridedSpec): string => {
  const scalar = WGSL_SCALAR[canonicalize(spec)];
  const dimAt = (d: number): number => 1 + d;
  const strideAt = (d: number): number => 1 + STRIDED_RANK + d;
  const offsetAt = 1 + 2 * STRIDED_RANK;

  // 入力の平坦添字を最終軸から divmod で分解する（入力は常に連続）。
  const decode: string[] = ["    var rem = i;"];
  for (let d = STRIDED_RANK - 1; d >= 1; d -= 1) {
    decode.push(`    let c${d} = rem % params[${dimAt(d)}u]; rem = rem / params[${dimAt(d)}u];`);
  }
  decode.push("    let c0 = rem;");

  const terms = Array.from(
    { length: STRIDED_RANK },
    (_, d) => `c${d} * params[${strideAt(d)}u]`,
  );

  return [
    `// karume strided write (rank ${STRIDED_RANK}, ${scalar}, generated)`,
    "@group(0) @binding(0) var<storage, read> params: array<u32>;",
    `@group(0) @binding(1) var<storage, read> src: array<${scalar}>;`,
    `@group(0) @binding(2) var<storage, read_write> out: array<${scalar}>;`,
    "",
    `@compute @workgroup_size(${STRIDED_WRITE_WORKGROUP_SIZE})`,
    "fn main(",
    "  @builtin(global_invocation_id) gid: vec3<u32>,",
    "  @builtin(num_workgroups) nwg: vec3<u32>,",
    ") {",
    "  let n = params[0u];",
    `  let stride = nwg.x * ${STRIDED_WRITE_WORKGROUP_SIZE}u;`,
    "  var i = gid.x;",
    "  while (i < n) {",
    ...decode,
    `    let out_index = params[${offsetAt}u] + ${terms.join(" + ")};`,
    "    out[out_index] = src[i];",
    "    i = i + stride;",
    "  }",
    "}",
    "",
  ].join("\n");
};

/**
 * 書き族の params（走査するのは**入力**なので dims は入力 shape、strides は出力側）。
 */
export const stridedWriteParams = (
  srcShape: readonly number[],
  outStrides: readonly number[],
  offset: number,
): Uint32Array<ArrayBuffer> =>
  packStridedParams(srcShape, outStrides, offset, {
    shape: "strided 書きの入力",
    rank: "入力",
  });

/**
 * cat の書き込み stride（出力の連続 stride）と、入力 `k` の書き出し先頭位置。
 *
 * offset は「連結軸の先行入力の長さの総和 × その軸の出力 stride」。ここを取り違えると
 * 出力の一部が二重に書かれ、残りが**未書き込みのまま**（プール再利用なら前の値）残る。
 */
export const catOutStrides = (outShape: readonly number[]): number[] => {
  assertRank(outShape, "cat の出力");
  return contiguousStrides(outShape);
};

export const catOutOffset = (
  outShape: readonly number[],
  dim: number,
  written: number,
): number => {
  assertRank(outShape, "cat の出力");
  if (!Number.isSafeInteger(dim) || dim < 0 || dim >= outShape.length) {
    throw new CodegenError(`strided params: cat の dim ${dim} が出力 rank ${outShape.length} の外`);
  }
  if (!Number.isSafeInteger(written) || written < 0 || written > outShape[dim]) {
    throw new CodegenError(
      `strided params: cat の書き出し位置 ${written} が軸 ${dim}（長さ ${outShape[dim]}）の外`,
    );
  }
  return written * contiguousStrides(outShape)[dim];
};
