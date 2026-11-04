// The u32 range check for uniforms (a single one).
//
// Each uniform field is read as `u32` on the WGSL side while the host value is an f64, so an
// assignment into a `Uint32Array` silently takes the value modulo 2^32. Writing the check per
// params builder splits into styles ("check only safe-integer / also check the upper bound /
// check nothing"), and on the weak side an oversized value passes narrowed all the way to the
// dispatch (a silent wrong value where only the element count differs), so the check lives here
// in one place.

import { CodegenError } from "./errors.ts";

/**
 * Checks that the values put on a uniform (and the intermediate values used to derive them) fit
 * in the u32 range.
 *
 * MUST: a non-integer, a negative value or one above `0xffff_ffff` fails loudly. It must be neither
 * rounded nor truncated — a value passed here becomes WGSL index arithmetic as is.
 * MUST: the message carries the parameter name and the value (diagnosis cannot start without
 * knowing which field overflowed).
 *
 * @param where the head of the failure message (e.g. `"gather params"`).
 * @param values parameter name → value. The name is a uniform field name or a derivation
 *   (e.g. `"4 * n"`).
 */
export const assertU32Params = (where: string, values: Readonly<Record<string, number>>): void => {
  for (const [name, value] of Object.entries(values)) {
    if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) {
      throw new CodegenError(`${where}: ${name} must be a non-negative u32 integer (${value})`);
    }
  }
};

/**
 * Checks that an in-row loop `i = start; while (i < bound) { …; i = i + step; }` cannot wrap.
 *
 * MUST: `bound + step ≤ 2^32`. The last index below `bound` is at most `bound − 1`, so the final
 * addition reaches at most `bound − 1 + step`; past 2^32 − 1 it wraps to a small value that is
 * still below `bound`, and the loop never ends (the GPU hangs or the device is lost). The loop is
 * inside one row, so the dispatch-count check (`gridStrideWorkgroups` in src/codegen/dispatch.ts)
 * never sees it — each row kernel's params call this with its widest in-row step.
 *
 * @param where the head of the failure message (e.g. `"softmax params"`).
 * @param bound the loop bound's name and value (the uniform field or the derivation that bounds the
 *   loop, e.g. `{ dim }`). Each value is u32-checked first.
 * @param step the loop's constant step (the workgroup size or the lane count).
 */
export const assertRowLoopBound = (
  where: string,
  bound: Readonly<Record<string, number>>,
  step: number,
): void => {
  assertU32Params(where, bound);
  for (const [name, value] of Object.entries(bound)) {
    if (value + step > 2 ** 32) {
      throw new CodegenError(
        `${where}: ${name} ${value} + in-row step ${step} exceeds 2^32 (the u32 loop index wraps around on the final addition and the loop never ends)`,
      );
    }
  }
};
