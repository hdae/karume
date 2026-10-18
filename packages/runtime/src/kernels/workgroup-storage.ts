/**
 * Leaf module that counts workgroup storage (WGSL `var<workgroup>`) the same way WebGPU counts it
 * when comparing against the device limit.
 *
 * WebGPU compute pipeline validation requires "the sum of `roundUp(16, SizeOf(T))` over each type T
 * of the `var<workgroup>` variables statically used by the entry point ≤
 * `maxComputeWorkgroupStorageSize`" (validation of GPUProgrammableStage in the WebGPU spec). Each
 * variable is rounded up to 16 bytes, so the total element count times the byte width cannot be
 * used for the limit comparison (it comes out too small).
 */

/** Bytes one `var<workgroup>` occupies in the limit comparison (`roundUp(16, SizeOf(T))`). */
export const workgroupVariableBytes = (sizeOf: number): number => Math.ceil(sizeOf / 16) * 16;
