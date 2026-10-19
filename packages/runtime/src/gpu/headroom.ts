/**
 * A **trial allocation** of free VRAM — checks "would an allocation of `bytes` succeed right now"
 * by going once through the only non-fatal allocation path (`createBuffer`).
 *
 * ## Why a trial allocation
 *
 * The only allocations for which WebGPU guarantees "running out does not lose the device" are the
 * out-of-memory cases of `createBuffer` / `createTexture` (specification: "If the allocation fails
 * without side-effects, generate an out-of-memory error"). Implementations follow that line —
 * wgpu-core (Deno) treats only the OOM of `create_buffer` as non-fatal, while an OOM in the staging
 * that `queue.writeBuffer` creates internally, in a submit or in bind group creation invalidates
 * the device (`handle_hal_error` → `lose()`). Dawn (Chrome) has the same rule
 * (`Device::HandleError` — the only calls that route OOM to the errorScope are createBuffer /
 * createTexture / createQuerySet). Moreover Deno delivers that OOM to the errorScope as an
 * **ordinary** `GPUOutOfMemoryError`, and `device.lost` does not resolve until the next call. So a
 * design that "evicts after hitting an OOM" is already too late when the OOM it hit was on the
 * staging side (docs/research/2026-09-27-h35-oom-device-lost.md). Before a step that involves a
 * large allocation (uploading weights), check here first whether that amount fits.
 *
 * ## What it measures
 *
 * A wgpu OOM is not a real allocation failure but a pre-check against the `VK_EXT_memory_budget`
 * budget (OOM when usage + request ≥ 97% of the budget), so this trial allocation reads "the budget
 * left on the device-local heap" directly through the same check. The budget also moves with other
 * processes' usage (Mesa uses `usage + free × 0.9`), so the result is **a fact of that moment**,
 * not a reservation. The caller cannot rule out that a later allocation fails for another reason
 * (the final check is still the out-of-memory errorScope).
 *
 * MUST: allocate STORAGE (a check of the device-local heap). A host-visible kind (MAP_*) would
 * bring the system-RAM heap into the check and would not answer for the heap in question.
 * MUST: destroy the allocated buffers at once and return only after `onSubmittedWorkDone` (= a
 * device poll) has settled the release. wgpu's poll runs release (maintain) → the 99% line check,
 * in that order, so this wait does not endanger the device.
 * Do not issue `queue.submit([])` — the submit-side 99% line check runs before the release, so near
 * the ceiling it would hit the check while still holding the trial allocation's usage.
 * MUST NOT: expose the result as "free memory" (binary search for the ceiling stays in diagnostic
 * tools — making an always-moving value the face of the API breeds guesswork: ADR 0070 decision 5).
 */

import type { GpuContext } from "./context.ts";
import { RUNTIME_INTERNAL } from "./context.ts";
import {
  discardFailureScopes,
  GpuOutOfMemoryError,
  popFailureScopes,
  pushFailureScopes,
} from "./error-scope.ts";
import { BUFFER_USAGE } from "./webgpu-constants.ts";

/**
 * Splits `bytes` into **equal** pieces of at most `maxBufferSize` (each piece 4-byte aligned; the
 * last piece the same size as the others ± 4).
 *
 * Equal pieces because the allocator (gpu-allocator) carves requests smaller than a block (256 MiB
 * by default) out of a block — a single small remainder piece could cause a real allocation (a new
 * block) larger than the amount the pre-check saw. With equal pieces every piece is
 * `bytes / pieces`, and when `bytes` is in the GiB range every piece is a dedicated allocation.
 *
 * NOTE: exported so that a test can pin the split without a GPU (not exposed from `mod.ts`).
 */
export const probePieces = (bytes: number, maxBufferSize: number): number[] => {
  if (!Number.isSafeInteger(bytes) || bytes < 0) {
    throw new RangeError(
      `Headroom probe byte count must be a safe integer >= 0 (received: ${bytes})`,
    );
  }
  if (!Number.isSafeInteger(maxBufferSize) || maxBufferSize < 4) {
    throw new RangeError(
      `maxBufferSize is less than 4 or not an integer (received: ${maxBufferSize})`,
    );
  }
  if (bytes === 0) return [];
  let count = Math.ceil(bytes / maxBufferSize);
  let each = Math.ceil(bytes / count / 4) * 4;
  // When rounding up to 4-byte alignment pushes a piece over maxBufferSize (bytes just above a
  // multiple of maxBufferSize), add one more piece.
  if (each > maxBufferSize) {
    count += 1;
    each = Math.ceil(bytes / count / 4) * 4;
  }
  return Array.from({ length: count }, () => each);
};

/**
 * Tries whether an allocation of `bytes` succeeds now (details in the module header).
 *
 * - `true`: the allocation succeeded; everything was destroyed and the release awaited (the next
 *   allocation may expect the same amount — it is not a reservation)
 * - `false`: out-of-memory (the device is alive — the caller decides here whether to evict or fall
 *   back)
 * - throw: validation (a bug) / `GpuDeviceLostError` (an already invalid device — it surfaces
 *   during the release wait)
 *
 * MUST: keep everything from the push to the **issue** of the pop in one synchronous section (an
 * await in between interleaves the LIFO with errorScopes elsewhere — "Invariants of errorScope
 * sections" at the top of `gpu/context.ts`).
 */
export const fitsHeadroom = async (gpu: GpuContext, bytes: number): Promise<boolean> => {
  const pieces = probePieces(bytes, gpu.limits.maxBufferSize);
  if (pieces.length === 0) return true;
  const device = gpu.device;
  const probes: GPUBuffer[] = [];
  pushFailureScopes(device);
  try {
    for (const size of pieces) {
      probes.push(
        device.createBuffer({ label: "karume-headroom-probe", size, usage: BUFFER_USAGE.STORAGE }),
      );
    }
  } catch (cause) {
    // Even on a synchronous exception (an implementation throwing outside the spec), do not leave
    // the two scopes on the stack.
    await discardFailureScopes(device);
    for (const probe of probes) probe.destroy();
    throw cause;
  }
  const failure = await popFailureScopes(device, "headroom probe");
  for (const probe of probes) probe.destroy();
  // Settle the release (poll). On an already invalidated device this becomes GpuDeviceLostError.
  await gpu[RUNTIME_INTERNAL].raceDeviceLost(
    device.queue.onSubmittedWorkDone(),
    "headroom probe cleanup",
  );
  if (failure === undefined) return true;
  if (failure instanceof GpuOutOfMemoryError) return false;
  throw failure;
};
