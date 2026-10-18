/**
 * The discipline for pushing WebGPU errorScopes, and the typed errors that come out of it.
 *
 * Most WebGPU failures **do not become synchronous exceptions** — an invalid pipeline / invalid
 * buffer is returned, and the dispatch or writeBuffer that uses it becomes a no-op without even a
 * warning. The only job of this layer is "catch that silence with an errorScope and turn it into a
 * loud exception"; it is independent of acquiring the device and normalizing its capabilities
 * (`device.ts`).
 *
 * MUST: this layer does not import `device.ts`. The errorScope discipline is closed over one WebGPU
 * property alone — a per-device LIFO stack — and does not need to know GpuContext. The split also
 * **removes a runtime cycle structurally** — `PipelineCache` depends only on this file, so
 * GpuContext can own a PipelineCache without a context.ts ⇄ pipeline-cache.ts loop.
 *
 * NOTE: device-specific errors such as `GpuDeviceLostError` stay in `context.ts` (an errorScope
 * does not catch device loss — `popErrorScope` after the loss resolves with null).
 */

/** A validation error caught by an errorScope (an invalid pipeline and the like). */
export class GpuValidationError extends Error {
  override readonly name = "GpuValidationError";
}

/**
 * An out-of-memory error caught by an errorScope (an allocation request exceeded what the device
 * has left). Unlike validation it is decided by the user's model size and environment, so it is a
 * separate type that callers can branch on.
 */
export class GpuOutOfMemoryError extends Error {
  override readonly name = "GpuOutOfMemoryError";
}

/**
 * An internal error caught by an errorScope (the operation failed for implementation reasons — a
 * shader too complex to compile, for example). The WGSL itself is valid, so it branches differently
 * from validation: for the user it is not "the code is invalid" but "this kernel shape does not go
 * through in this environment", and without a separate type a codegen bug and an environment limit
 * would collapse into the same report.
 */
export class GpuInternalError extends Error {
  override readonly name = "GpuInternalError";
}

/**
 * Pushes the two scopes internal + validation, runs `body` (pipeline creation) and turns any caught
 * error into an exception. **Pipeline creation paths use this one** (with validation alone,
 * internal errors pass straight through).
 *
 * Both are needed because the failures show up differently — exceeding a limit or a type mismatch
 * is validation, an implementation reason (a shader too complex to compile, for example) is
 * internal, and neither becomes a synchronous exception; **an invalid pipeline is simply
 * returned**. Without the scopes the dispatch becomes a no-op and the output is all 0.
 *
 * MUST NOT: wait for asynchronous work inside `body`. errorScopes form a stack and the pop happens
 * right after `body` runs synchronously, so an operation encoded after an await is absorbed by a
 * different scope.
 */
export const withPipelineScope = async <T>(
  device: GPUDevice,
  label: string,
  body: () => T,
): Promise<T> => {
  device.pushErrorScope("internal");
  device.pushErrorScope("validation");
  let value: T;
  try {
    value = body();
  } catch (cause) {
    // MUST: pop both scopes even when body throws. Leaving even one on the stack makes later
    // validation results land in the wrong scope, and from then on errors are permanently
    // invisible. A failure of the pop itself is swallowed (cleanup does not overwrite the body's
    // exception — the same discipline as discardFailureScopes).
    const validation = device.popErrorScope().catch(() => null);
    const internal = device.popErrorScope().catch(() => null);
    await Promise.all([validation, internal]);
    throw cause;
  }
  // MUST: **issue** the two pops **in the same synchronous section** (only the already-issued
  // promises are awaited). A pop takes the top of the stack unconditionally, so an await between
  // the two issues lets the second pop take a scope pushed elsewhere in the gap, and the failure is
  // misattributed (the same discipline as popFailureScopes).
  const validation = device.popErrorScope();
  const internal = device.popErrorScope();
  const [validationError, internalError] = await Promise.all([validation, internal]);
  // MUST: when both caught something, return internal. Later operations that touch a pipeline
  // invalidated by internal (`getBindGroupLayout` and the like) raise a **derived** validation
  // error, so returning validation first would throw away the root-cause internal error and report
  // an "invalid pipeline" with no visible cause. The same judgement as returning OOM before
  // validation (docs/research/2026-08-08-vram-oom-misreport.md).
  if (internalError !== null) {
    throw new GpuInternalError(`${label}: ${internalError.message}`);
  }
  if (validationError !== null) {
    throw new GpuValidationError(`${label}: ${validationError.message}`);
  }
  return value;
};

/**
 * Pushes the two scopes out-of-memory + validation. Sections that allocate are wrapped in this
 * pair.
 *
 * Both are needed because the failures show up differently — a `createBuffer` over a limit is
 * validation, the device running out is out-of-memory, and neither becomes a synchronous exception;
 * **an invalid buffer is simply returned**. Without the scopes, a `writeBuffer` into it becomes a
 * no-op without even a warning, and processing continues with empty weights or empty intermediate
 * buffers. The matching pop always uses {@link popFailureScopes} /
 * {@link discardFailureScopes} (the order and synchronicity of the pops are closed in there).
 */
export const pushFailureScopes = (device: GPUDevice): void => {
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
};

/**
 * Pops the two scopes of {@link pushFailureScopes} and returns any caught failure as a typed
 * exception (undefined when nothing was caught).
 *
 * MUST: **issue** the two pops **in the same synchronous section** (only the already-issued
 * promises are awaited). A pop takes the top of the stack unconditionally, so an await between the
 * two issues lets the second pop take a scope pushed elsewhere in the gap, and the failure is
 * misattributed.
 *
 * MUST: when both caught something, return out-of-memory. When an allocation fails because the
 * device ran out, `createBuffer` returns an invalid buffer, and the later `createBindGroup` /
 * `writeBuffer` that use it raise a **derived** validation error,
 * `Buffer with '' label is invalid`. Returning validation first would throw away the root-cause OOM
 * and report an indistinguishable "invalid buffer"
 * (docs/research/2026-08-08-vram-oom-misreport.md).
 */
export const popFailureScopes = async (
  device: GPUDevice,
  label: string,
): Promise<GpuValidationError | GpuOutOfMemoryError | undefined> => {
  const validation = device.popErrorScope();
  const outOfMemory = device.popErrorScope();
  const [validationError, outOfMemoryError] = await Promise.all([validation, outOfMemory]);
  if (outOfMemoryError !== null) {
    return new GpuOutOfMemoryError(`${label}: ${outOfMemoryError.message}`);
  }
  if (validationError !== null) {
    return new GpuValidationError(`${label}: ${validationError.message}`);
  }
  return undefined;
};

/**
 * Cleanup on a failure path. Pops the two scopes of {@link pushFailureScopes} and discards the
 * results.
 *
 * The return value is "a GPU-side failure cannot be ruled out" — true when either scope caught a
 * failure or the pop itself failed. The caller rethrows the body's exception as is, so a caught
 * failure is not turned into an exception. It returns a boolean so that a failure path that failed
 * on the host side can still decide whether to keep or discard the resources made in the section
 * (an allocation that hit a non-fatal OOM is invalid) — the catch of the `Session` run reads it.
 *
 * MUST: pop both scopes even when the body exits with an exception. Leaving one on the stack makes
 * later validation results land in the wrong scope, and errors become permanently invisible. A
 * failure of the pop itself is swallowed too (cleanup does not overwrite the body's exception).
 * MUST: issue the two pops in the same synchronous section (the same discipline as
 * {@link popFailureScopes}).
 */
export const discardFailureScopes = async (device: GPUDevice): Promise<boolean> => {
  const settle = (pop: Promise<GPUError | null>): Promise<boolean> =>
    pop.then(
      (error) => error !== null,
      () => true,
    );
  const validation = settle(device.popErrorScope());
  const outOfMemory = settle(device.popErrorScope());
  const [validationFailed, outOfMemoryFailed] = await Promise.all([validation, outOfMemory]);
  return validationFailed || outOfMemoryFailed;
};
