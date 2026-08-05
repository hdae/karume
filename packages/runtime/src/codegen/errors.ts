// WGSL 生成層のエラー語彙。生成器（elementwise / reduce / matmul）と dispatch 数の決定が
// 共有するため、循環 import を作らないよう独立した葉モジュールに置く。

/** codegen の入力が契約に反する（未知 op・rank 0・負のサイズなど）。 */
export class CodegenError extends Error {
  override readonly name = "CodegenError";
}

/** dispatch 数が device の上限を超え、カーネル側に縮退の余地も無い。 */
export class DispatchLimitError extends Error {
  override readonly name = "DispatchLimitError";
}
