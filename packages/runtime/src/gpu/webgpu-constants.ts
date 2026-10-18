/**
 * Our own definitions of the WebGPU bit-flag constants (`GPUBufferUsage` / `GPUMapMode`).
 *
 * MUST: do **not read** the platform namespaces (bare `GPUBufferUsage` / `GPUMapMode`). They are
 * **global bindings** that do not exist in an environment without WebGPU, so reading them at module
 * scope throws `ReferenceError` on a mere `import` (unlike a property access such as
 * `navigator.gpu`, an undefined identifier does not evaluate to undefined). A user imports the
 * package first in order to show a "WebGPU is not supported" notice, so a failing import means a
 * blank screen before that notice. This is an invariant of the same kind as, and next to, "every
 * module is side-effect free" (no execution at import time).
 *
 * MUST: every usage / mapAsync flag inside the runtime goes **through this module**. A bare
 * reference at function scope would not throw, but mixing the two makes the rule "only a bare
 * reference at module scope is a violation", which cannot be detected mechanically. With one rule,
 * the only survivors of `rg 'GPUBufferUsage|GPUMapMode' src/` are this file (and its test), and a
 * violation is found with a single grep.
 *
 * The values are transcribed from the bit values frozen in the WebGPU specification
 * (https://www.w3.org/TR/webgpu/). A transcription error would be a silent malfunction (a missing
 * usage makes buffer creation fail with a validation error, an extra one passes silently), so where
 * the namespaces exist tests/webgpu_constants_test.ts checks every bit against the real one.
 */

/** Bit values of `GPUBufferUsage` (specification §buffer-usage). */
export const BUFFER_USAGE = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
} as const;

/** Bit values of `GPUMapMode` (specification §buffer-mapping). */
export const MAP_MODE = {
  READ: 0x0001,
  WRITE: 0x0002,
} as const;
