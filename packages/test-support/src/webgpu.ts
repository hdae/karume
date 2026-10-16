/**
 * Helpers for building WebGPU platform objects by hand in tests.
 *
 * `@webgpu/types` brands its interfaces (`readonly __brand: "GPUDeviceLostInfo"` etc.) so that
 * structurally identical objects are not interchangeable. Fakes in tests legitimately construct
 * such objects; this is the one place that supplies the brand.
 */

/** A hand-built platform object: every observable field, plus the nominal brand. */
export const platformObject = <T extends { readonly __brand: string }>(
  fields: Omit<T, "__brand">,
): T => fields as T;
