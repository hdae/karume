/**
 * Fixed kernel of `state_append` (the single-purpose op that writes this step's k / v into a state
 * slot — ADR 0067 decision 5). 1 node = 1 dispatch, an **effect op** that defines no value (0
 * outputs).
 *
 * Maps **only the first `queryLength` rows** of the input `[B,Hkv,M,D]` to the logical rows
 * `[P, P+Q)` of the slot `[B,Hkv,C,D]`. The physical-row mapping is {@link stateSlotRowWgsl}
 * (sliding is a `col % C` ring — why the modulus is the capacity `C` and not the window `W` is in
 * that function's doc), and it **MUST share the identical string with the readers (①QK / ③PV)**
 * (ADR 0067 decision 4 "same formula for reads and writes" — a different formula on one side
 * silently points every read at another row once the ring has wrapped).
 *
 * MUST: **pad rows (`row ≥ Q`) are not written** (slots are outside full-write — ADR 0066 addendum
 * 6; leftovers are overwritten by the next step's append with the same formula, and readers do not
 * read outside the resident range). Writing them gives "a slot with meaningless values inside the
 * window", which the next step's readers consume as past.
 * MUST: the work is proportional to `B·Hkv·Q·D` (not to the capacity `C` — ADR 0066 decision 3).
 * MUST: with sliding and `Q > C`, among the logical rows that map to the same physical row **only
 * the last one** writes (`row + C ≥ Q`). Ring semantics say the newest row remains, but letting
 * every row write in parallel makes the winner implementation-dependent — silent nondeterminism.
 * Full is injective (`dst = P + row`), so it does not collide (this check is generated only in the
 * sliding variant).
 *
 * NOTE: `P + Q ≤ C` for full slots (ADR 0067 decision 4 ④) is a **run-time check on the context
 * side** and is not looked at here (`P` is a run-time value, not in params). If it is violated, the
 * write target `dst = (kv_plane·C + P + row)·D + d` of rows with `P + row ≥ C` **silently overwrites
 * rows `[0, P+row−C)` of the next kv plane** for every plane except the last (only the last plane
 * falls out of range and is discarded). The ①QK / ③PV read `(kv_plane·C + slot_row(col))·D` misreads
 * the same neighbouring plane. Another head's KV breaks without an exception, so the context-side
 * check is MUST.
 */

import { CodegenError } from "../codegen/errors.ts";
import { gridStrideWorkgroups } from "../codegen/dispatch.ts";
import { assertU32Params } from "../codegen/params.ts";
import { STATE_LENGTHS_STRUCT, stateSlotRowWgsl, stateVariantKeyPart } from "./state-attention.ts";
import {
  assertStateStorageDepth,
  assertStateStorageFull,
  type StateStorage,
  stateSlotElementType,
  stateSlotPackWgsl,
  stateStorageKeyPart,
} from "./state-storage.ts";

export const STATE_APPEND_WORKGROUP_SIZE = 256;

export const stateAppendKey = (sliding: boolean, storage: StateStorage = "f32"): string =>
  `state_append:v1:f32:wg${STATE_APPEND_WORKGROUP_SIZE}${stateVariantKeyPart(sliding, false)}${stateStorageKeyPart(storage)}`;

/**
 * Bindings (**binding 0 = static params, last = logical lengths** — the same order as the three
 * states-form kernels):
 *
 * | binding | Resource                               |
 * | ------- | -------------------------------------- |
 * | 0       | `Params` (uniform)                     |
 * | 1       | `x` `[B,Hkv,M,D]` (read)               |
 * | 2       | `slot` `[B,Hkv,C,D]` (write)           |
 * | 3       | `Lengths` (uniform — owned by context) |
 *
 * Word order of params (**this table is the source of truth**): `kv_planes` (= `B·Hkv`) /
 * `chunk_rows` (= `M`) / `depth` (= `D`) / `capacity` (= `C`) / `window` (= `W`, `0` = full).
 */
export const stateAppendWgsl = (sliding: boolean, storage: StateStorage = "f32"): string => {
  assertStateStorageFull("state_append", sliding, storage);
  return `// karume state_append (this step's k / v into the state slot, f32${
    sliding ? ", ring write of the sliding window" : ""
  }${storage === "f16" ? ", KV in f16 storage, written a word at a time" : ""})
struct Params {
  kv_planes: u32,
  chunk_rows: u32,
  depth: u32,
  capacity: u32,
  window: u32,
}
${STATE_LENGTHS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<${stateSlotElementType(storage)}>;
@group(0) @binding(2) var<storage, read_write> slot: array<${stateSlotElementType(storage)}>;
@group(0) @binding(3) var<uniform> lengths: Lengths;

${stateSlotRowWgsl(sliding)}
${storage === "f16" ? `\n${stateSlotPackWgsl()}\n` : ""}
@compute @workgroup_size(${STATE_APPEND_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let past = lengths.past;
  let query = lengths.query;
${
  storage === "f16"
    ? `  // Total work is **B·Hkv·Q·D/2** words (pad rows are not in the index space = not written). One
  // invocation owns one word, because writing half a word is a read-modify-write that races
  let words = params.depth / 2u;
  let total = params.kv_planes * query * words;
  let stride = nwg.x * ${STATE_APPEND_WORKGROUP_SIZE}u;
  var i = gid.x;
  while (i < total) {
    let w = i % words;
    let rest = i / words;
    let row = rest % query;
    let kv_plane = rest / query;
    let src = (kv_plane * params.chunk_rows + row) * params.depth + w * 2u;
    let dst = (kv_plane * params.capacity + slot_row(past + row)) * words + w;
    slot[dst] = f16_bits(x[src]) | (f16_bits(x[src + 1u]) << 16u);
    i = i + stride;
  }
}
`
    : `  // Total work is **B·Hkv·Q·D** (pad rows are not in the index space = not written)
  let total = params.kv_planes * query * params.depth;
  let stride = nwg.x * ${STATE_APPEND_WORKGROUP_SIZE}u;
  var i = gid.x;
  while (i < total) {
    let d = i % params.depth;
    let rest = i / params.depth;
    let row = rest % query;
    let kv_plane = rest / query;
    let src = (kv_plane * params.chunk_rows + row) * params.depth + d;
    let dst = (kv_plane * params.capacity + slot_row(past + row)) * params.depth + d;
${
  sliding
    ? `    // With Q > C the ring wraps and several logical rows map to the same physical row. **Only
    // the last logical row** writes (letting every row write in parallel makes the winner
    // implementation-dependent = silent nondeterminism). The modulus is **capacity**, as in
    // slot_row (cutting by window would, when C > W, silently skip rows that alias nothing)
    if (row + params.capacity >= query) {
      slot[dst] = x[src];
    }`
    : "    slot[dst] = x[src];"
}
    i = i + stride;
  }
}
`
}`;
};

/** Static geometry of `state_append` (the type side of the uniform word order). */
export type StateAppendGeometry = {
  /** `B·Hkv` (number of slot planes — shared by the input and the slot). */
  readonly kvPlanes: number;
  /** `M` (row stride of the input = physical chunk row count). */
  readonly chunkRows: number;
  /** `D`. */
  readonly depth: number;
  /** `C` (row capacity of the slot). */
  readonly capacity: number;
  /** `W` (`0` = full). */
  readonly window: number;
};

/**
 * Uniform Params (5 words rounded up to 16-byte units, 32 bytes allocated — allocating in 16-byte
 * units is a project convention (the fusion pass fixes the same form in ADR 0040), not a
 * requirement of the spec).
 *
 * MUST: `W ≤ C` (a window larger than the slot capacity violates ADR 0067 decision 4 ③). Whether it
 * is sliding is `window > 0` (`stateSliding` in src/kernels/state-attention.ts), and `W = 0` = full.
 * The caller derives the key selection, the WGSL generation and these params, all three, **from the
 * same `window` value** (making only one of them sliding turns `col % 0u` into an
 * implementation-dependent value).
 */
export const stateAppendParams = (
  geometry: StateAppendGeometry,
  storage: StateStorage = "f32",
): Uint32Array<ArrayBuffer> => {
  const { kvPlanes, chunkRows, depth, capacity, window } = geometry;
  assertStateStorageDepth("state_append params", depth, storage);
  assertU32Params("state_append params", {
    kv_planes: kvPlanes,
    chunk_rows: chunkRows,
    depth,
    capacity,
    window,
  });
  if (kvPlanes < 1 || chunkRows < 1 || depth < 1 || capacity < 1) {
    throw new CodegenError(
      `state_append params: kv_planes / chunk_rows / depth / capacity must be positive integers` +
        ` (${kvPlanes} / ${chunkRows} / ${depth} / ${capacity})`,
    );
  }
  if (window > capacity) {
    throw new CodegenError(
      `state_append params: window ${window} exceeds capacity ${capacity} (ADR 0067 decision 4 ③)`,
    );
  }
  const params = new Uint32Array(8);
  params[0] = kvPlanes;
  params[1] = chunkRows;
  params[2] = depth;
  params[3] = capacity;
  params[4] = window;
  return params;
};

/**
 * Workgroup count `[⌈B·Hkv·Q·D / 256⌉, 1, 1]` (grid-stride, so exceeding the limit is clamped).
 *
 * MUST: the capacity `C` is not a parameter (a dispatch proportional to `C` is the form ADR 0066
 * decision 3 rejects). It is **proportional** to `Q` and does not depend on `P` (only this step's Q
 * rows are written).
 */
export const stateAppendWorkgroups = (
  geometry: Pick<StateAppendGeometry, "kvPlanes" | "depth">,
  query: number,
  limit: number,
  where: string,
): [number, number, number] => {
  const { kvPlanes, depth } = geometry;
  assertU32Params(where, { kv_planes: kvPlanes, depth, query });
  if (kvPlanes < 1 || depth < 1 || query < 1) {
    throw new CodegenError(
      `${where}: kv_planes / depth / query must be positive integers (${kvPlanes} / ${depth} / ${query})`,
    );
  }
  const total = kvPlanes * query * depth;
  assertU32Params(where, { "kv_planes * query * depth": total });
  return [gridStrideWorkgroups(total, STATE_APPEND_WORKGROUP_SIZE, limit), 1, 1];
};
