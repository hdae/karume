/**
 * The usage of a dispatch's params buffer (two kinds: uniform / storage).
 *
 * The per-op-family derivation and the fusion replay of {@link "../recipe-builder.ts"} use the
 * same values, so **the values live here and nowhere else**. Importing `recipe-builder.ts` by
 * value from a per-op-family module would create a back edge against "entry → per-op-family"
 * (a type-only reference is erased, so it does not create one).
 */

import { BUFFER_USAGE } from "../../gpu/webgpu-constants.ts";

export const PARAMS_STORAGE_USAGE = BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST;
export const PARAMS_UNIFORM_USAGE = BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST;
