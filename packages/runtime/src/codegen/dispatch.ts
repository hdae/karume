/**
 * How dispatch counts are chosen. The number of workgroups per dimension has a limit (65535 by
 * default in the spec), and real models do exceed it (16 heads × 4096 tokens = 65536 rows). Kernel
 * families behave differently when it is exceeded, so the decision is confined to these two
 * functions.
 */

import { DispatchLimitError } from "./errors.ts";

/**
 * Dispatch count of a grid-stride kernel. Exceeding the limit falls back to fewer workgroups — the
 * kernel derives the stride from `num_workgroups` and loops over the rest, so fewer workgroups
 * still cover the whole range.
 *
 * MUST: `items + stride ≤ 2^32` (stride = returned workgroup count × `workgroupSize`). The kernel
 * adds `i = i + stride` in u32 and tests `i < n`, so beyond that the last addition wraps around and
 * the loop never ends (a GPU hang, not an exception). Switching the WGSL side to a
 * subtraction-based termination test would change the WGSL of every grid-stride kernel, so the
 * check lives here in one place.
 */
export const gridStrideWorkgroups = (
  items: number,
  workgroupSize: number,
  limit: number,
): number => {
  const groups = Math.min(limit, Math.ceil(items / workgroupSize));
  const stride = groups * workgroupSize;
  if (items + stride > 2 ** 32) {
    throw new DispatchLimitError(
      `grid-stride: element count ${items} + stride ${stride} exceeds 2^32 (the u32 loop index wraps around on the final addition and the loop never ends)`,
    );
  }
  return groups;
};

/**
 * Dispatch count of a kernel that covers the whole range with one workgroup = one tile (matmul).
 * MUST: exceeding the limit fails loudly — a fallback would drop tiles, and part of the output
 * would stay unwritten (or keep the previous value when the buffer is reused) without an exception.
 */
export const tiledWorkgroups = (
  items: number,
  tile: number,
  limit: number,
  where: string,
): number => {
  const groups = Math.ceil(items / tile);
  if (groups > limit) {
    throw new DispatchLimitError(
      `${where}: required workgroup count ${groups} exceeds the device limit ${limit} (${items} / tile ${tile})`,
    );
  }
  return groups;
};
