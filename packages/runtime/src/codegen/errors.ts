// Error vocabulary of the WGSL generation layer. The generators (elementwise / reduce / matmul) and
// the dispatch-count decision share it, so it lives in an independent leaf module to avoid an
// import cycle.

/** The codegen input violates its contract (unknown op, rank 0, negative size, ...). */
export class CodegenError extends Error {
  override readonly name = "CodegenError";
}

/** The dispatch count exceeds the device limit and the kernel has no room to fall back either. */
export class DispatchLimitError extends Error {
  override readonly name = "DispatchLimitError";
}
