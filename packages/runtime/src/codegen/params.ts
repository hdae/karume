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
