/**
 * The kernel of `expert_linear` (`y[t, j] = weight[index[t, j]] · x[t (, j)]` — ADR 0145 decision 4).
 *
 * The GEMV form of `linear` (src/kernels/linear-gemv.ts, ADR 0082) with the weight chosen per row:
 * one thread owns one output column of one pick row (`row = t · picks + j`), reads the pick's expert
 * id from `index` on the GPU, and walks that expert's weight row in ascending k. The routing never
 * goes to the host, so decode keeps one fence per run.
 *
 * ## Numeric contract
 *
 * There is no default path to be bit-identical to, so the contract is the op's own: per output, a
 * sequential f32 sum over `k` in ascending order, each element decoded as `f32(i32(u) − 8) · step`
 * (i4) or read as is (f32), with the multiply-add spelled as `linear`'s sequential GEMV spells it
 * ({@link unitMacs} — the same text). The step is f32 for `int4-sym-g` and an f16 widened to f32
 * for `int4-sym-g-f16` (exact — ADR 0069 decision 3). MUST: one thread holds the whole reduction of
 * an output, so the result does not depend on how rows are split across workgroups.
 *
 * ## Layout
 *
 * `weight` is `[E, n, k]`: expert `e`'s row `c` starts at word `(e · n + c) · (k / unit)`, and its
 * group `g` step at `(e · n + c) · (k / G) + g` (the group codec's scale is `[E, n · k / G]`, row
 * major — `groupScaleShape`). With f16 steps two share a u32 word (element `s` = half `s % 2` of
 * word `s / 2`), read with `unpack2x16float`, so no `shader-f16` feature is needed.
 * `x` is `[T, k]` (shared by the picks of a token: row `t = row / picks`) or `[T, picks, k]` (row
 * `row`), `out` is `[T, picks, n]`.
 *
 * ## Dispatch
 *
 * `[ceil(n / COLS), min(rows, limit), 1]`; the rows grid-stride along y (T · picks can exceed
 * `maxComputeWorkgroupsPerDimension`). A column past `n` returns at once (no shared memory, no
 * barrier).
 *
 * ## Out-of-range ids (ADR 0061)
 *
 * The planner admits only an index produced by a `topk` over `E` (ADR 0145 decision 2), so an id is
 * in `[0, E)` by construction. Should the invariant break, the row is written as quiet NaN instead
 * of reading another expert, as `gather` does; the CPU reference throws.
 *
 * ## The pooled form (ADR 0146 decision 2)
 *
 * Under an expert cache the weight holds K slots (`[K, n, k]`) and a table `i32[E]` (binding
 * {@link EXPERT_LINEAR_TABLE_BINDING}) maps an expert id to its slot, −1 when the expert is not
 * resident. A resident pick addresses its slot exactly where the resident kernel addresses the
 * expert (`e` replaced by the slot, the rest of the text shared), so a row whose picks are all
 * resident is bit-identical to the resident kernel's. A non-resident pick writes 0 (every output
 * is still written — ADR 0014): the host reads the routing and runs the row again after loading.
 * The runtime alone writes the table, so a slot of K or more breaks an invariant and writes NaN.
 *
 * ## The grouped form (runs of two or more tokens over long weight rows)
 *
 * Rows of different tokens often pick the same expert, and the one-row form then reads and decodes
 * that expert's weight row once per row. In the grouped form the thread of the first row that
 * picked an expert walks the expert's weight row for the rows that picked it, up to
 * {@link EXPERT_LINEAR_GROUP_ROWS} rows in one walk ({@link EXPERT_LINEAR_WALK_ROWS} lists the
 * walks), each row keeping its own sum; a row an earlier row leads is written by that row's
 * thread. Which rows share an expert is read from `index` on the GPU (a scan of the run's rows per
 * thread), so nothing goes to the host. An expert many rows picked is walked by one thread per
 * column, a walk after another.
 *
 * {@link expertLinearGrouped} says which dispatch takes it, by a static rule of the shape (never a
 * run-time choice): finding the rows costs a short weight row about what the shared walk saves, so
 * only a row of {@link EXPERT_LINEAR_GROUP_MIN_WORDS} words or more is grouped, and only a run of
 * the token counts the form was measured at (up to {@link EXPERT_LINEAR_GROUP_MAX_TOKENS}).
 *
 * MUST: a row's sum is the one-row form's — the same elements in ascending k through the same
 * multiply-add text (`expert_word` holds {@link unitMacs}, as `linear`'s row block calls its word
 * function per row), so the two forms write the same words; the GPU test and the kernel check
 * compare them as u32. Every output is still written exactly once: the rows the kernel does not sum
 * (an id out of range, an expert that is not resident, a broken slot) are written by their own
 * threads, as in the one-row form. The one-row form's text does not change with this form.
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import { unitMacs } from "./linear-gemv.ts";
import { i4GroupKeyPart, i4GroupShift } from "./weight-storage.ts";

/** The weight storages `expert_linear` takes (ADR 0145 decisions 2 and 3). */
export type ExpertLinearStorage = "f32" | "i4" | "i4f16";

/** Output columns per workgroup. */
export const EXPERT_LINEAR_COLS = 32;

/**
 * The binding of the i4 step table (i4 / i4f16 only; bindings 0–4 are dims / x / index / w / out).
 * The WGSL declares it from this constant, so the recipe builder and the kernel cannot drift.
 */
export const EXPERT_LINEAR_SCALE_BINDING = 5;

/** The binding of the slot table of the pooled form (module doc). */
export const EXPERT_LINEAR_TABLE_BINDING = 6;

/** Weight words issued ahead of their multiply-adds (memory parallelism, as `linear`'s GEMV). */
export const EXPERT_LINEAR_UNROLL = 4;

/**
 * The rows of one expert a thread finds in one scan of the grouped form (module doc), and so the
 * most rows one walk of the expert's weight row carries; more rows of one expert take another scan.
 */
export const EXPERT_LINEAR_GROUP_ROWS = 8;

/**
 * The row counts a walk of the grouped form is generated for, largest first. The rows of a scan are
 * walked in the largest count they still fill (5 rows: a walk of 4, then a walk of 1), which keeps
 * the WGSL to one loop per count listed here instead of one per count up to
 * {@link EXPERT_LINEAR_GROUP_ROWS}. MUST: descending, ending in 1 (every row is walked).
 */
export const EXPERT_LINEAR_WALK_ROWS: readonly number[] = [8, 4, 2, 1];

/** {@link EXPERT_LINEAR_WALK_ROWS} that a scan can fill. */
const walkRows = (): readonly number[] =>
  EXPERT_LINEAR_WALK_ROWS.filter((rows) => rows <= EXPERT_LINEAR_GROUP_ROWS);

/**
 * The fewest words of a weight row (16 bytes each: 32 i4 elements, 4 f32) the grouped form is taken
 * for. Measured on the two projections of Gemma 4 26B on an RTX 3080 Ti (perf-ledger K-104): a row
 * of 88 words gains from the shared walk, a row of 22 words does not.
 */
export const EXPERT_LINEAR_GROUP_MIN_WORDS = 64;

/** The quiet NaN written to a row whose expert id is out of range (ADR 0061). */
export const EXPERT_LINEAR_OOB_BITS = 0x7fc00000;

/** Elements per weight word (`vec4<f32>` = 4, `vec4<u32>` of i4 = 32). */
export const expertLinearUnit = (storage: ExpertLinearStorage): number =>
  storage === "f32" ? 4 : 32;

const isI4 = (storage: ExpertLinearStorage): boolean => storage !== "f32";

const groupShift = (
  storage: ExpertLinearStorage,
  groupSize: number | undefined,
): number | undefined => i4GroupShift("expert_linear", isI4(storage) ? "i4" : "f32", groupSize);

const storageKeyPart = (storage: ExpertLinearStorage): string =>
  storage === "f32" ? "" : storage === "i4" ? ":wi4" : ":wi4h";

/**
 * The most tokens of a run the grouped form is taken for: the longest run it was measured at (the
 * kernel alone at 26B's gate_up, 5 to 768 tokens, with a routing like a prompt's and with three
 * quarters of the rows on the same 8 experts as a bucket's pad rows are — perf-ledger K-104). It is
 * also the chunk length of the distributions, so no run of theirs is longer.
 */
export const EXPERT_LINEAR_GROUP_MAX_TOKENS = 768;

/**
 * Whether a dispatch takes the grouped form (module doc): a run of 2 to
 * {@link EXPERT_LINEAR_GROUP_MAX_TOKENS} tokens over weight rows of
 * {@link EXPERT_LINEAR_GROUP_MIN_WORDS} words or more. One token's picks are distinct experts, so a
 * one-token run has no rows to share a walk. The two forms write the same words, so the choice
 * changes no number.
 */
export const expertLinearGrouped = (
  storage: ExpertLinearStorage,
  tokens: number,
  k: number,
): boolean =>
  tokens > 1 &&
  tokens <= EXPERT_LINEAR_GROUP_MAX_TOKENS &&
  k / expertLinearUnit(storage) >= EXPERT_LINEAR_GROUP_MIN_WORDS;

/**
 * The pipeline key (codegen determinism: same key → byte-identical WGSL). `:rows<n>` names the
 * grouped form by the rows a scan finds ({@link EXPERT_LINEAR_GROUP_ROWS}); the walks' row counts
 * ({@link EXPERT_LINEAR_WALK_ROWS}) are not in the key, so a change to that list MUST come with a
 * change to this key's version. The pooled form's part stays last, so a key with it removed is the
 * resident key of the same form.
 */
export const expertLinearKey = (
  storage: ExpertLinearStorage,
  groupSize?: number,
  pooled = false,
  grouped = false,
): string =>
  `expert_linear:v1:f32:i32:c${EXPERT_LINEAR_COLS}u${EXPERT_LINEAR_UNROLL}${storageKeyPart(
    storage,
  )}${i4GroupKeyPart(groupSize)}${grouped ? `:rows${EXPERT_LINEAR_GROUP_ROWS}` : ""}${
    pooled ? ":pool" : ""
  }`;

const scaleBinding = (storage: ExpertLinearStorage): string =>
  storage === "i4"
    ? `\n@group(0) @binding(${EXPERT_LINEAR_SCALE_BINDING}) var<storage, read> wscale: array<f32>;`
    : storage === "i4f16"
      ? `
@group(0) @binding(${EXPERT_LINEAR_SCALE_BINDING}) var<storage, read> wscale: array<u32>;

// step s of an f16 scale: half s % 2 of word s / 2, widened to f32 (exact)
fn step_at(s: u32) -> f32 {
  let pair = unpack2x16float(wscale[s >> 1u]);
  return select(pair.x, pair.y, (s & 1u) == 1u);
}`
      : "";

/** The reads of one word: the weight word, the group step (i4) and the first x quad. */
const wordLoads = (
  storage: ExpertLinearStorage,
  slot: string,
  unitExpr: string,
  shift: number | undefined,
): string => {
  const unit = expertLinearUnit(storage);
  const step = `scale_base + ((unit${slot} * ${unit}u) >> ${shift}u)`;
  const groupStep =
    storage === "i4"
      ? `
      let ws${slot} = wscale[${step}];`
      : storage === "i4f16"
        ? `
      let ws${slot} = step_at(${step});`
        : "";
  return `      let unit${slot} = ${unitExpr};
      let pw${slot} = w[row_base + unit${slot}];${groupStep}
      let xq${slot} = x_base + unit${slot} * ${unit / 4}u;`;
};

/**
 * The multiply-adds of one word, indented for the row loop. The i4 text is `linear`'s own
 * ({@link unitMacs}); both i4 codecs decode the same bytes, only the step's source differs.
 */
const wordMacs = (storage: ExpertLinearStorage, slot: string): string =>
  unitMacs(storage === "f32" ? "f32" : "i4", slot)
    .split("\n")
    .map((line) => `  ${line}`)
    .join("\n");

/**
 * `expert_word` of the grouped form: one weight word's multiply-adds onto `initial`, the one-row
 * form's own text ({@link unitMacs}), as `linear`'s row block calls its word function per row.
 */
const wordFunction = (storage: ExpertLinearStorage): string => `
// One weight word's multiply-adds onto \`initial\`, in the one-row form's element order and spelling
fn expert_word(pwa: vec4<${storage === "f32" ? "f32" : "u32"}>, xqa: u32${
  isI4(storage) ? ", wsa: f32" : ""
}, initial: f32) -> f32 {
  var acc = initial;
${unitMacs(storage === "f32" ? "f32" : "i4", "a")}
  return acc;
}`;

/** The reads of one word in the grouped form: the weight word, the group step (i4), the x offset. */
const groupedLoads = (
  storage: ExpertLinearStorage,
  slot: string,
  unitExpr: string,
  shift: number | undefined,
): string => {
  const unit = expertLinearUnit(storage);
  const step = `scale_base + ((unit${slot} * ${unit}u) >> ${shift}u)`;
  const groupStep =
    storage === "i4"
      ? `
              let ws${slot} = wscale[${step}];`
      : storage === "i4f16"
        ? `
              let ws${slot} = step_at(${step});`
        : "";
  return `              let unit${slot} = ${unitExpr};
              let pw${slot} = w[row_base + unit${slot}];${groupStep}
              let xo${slot} = unit${slot} * ${unit / 4}u;`;
};

/** One word onto each of the walk's rows, row by row (each row's sum stays in ascending k). */
const groupedMacs = (storage: ExpertLinearStorage, slot: string, rows: number): string =>
  Array.from(
    { length: rows },
    (_, row) =>
      `              acc${row} = expert_word(pw${slot}, xr${row} + xo${slot}${
        isI4(storage) ? `, ws${slot}` : ""
      }, acc${row});`,
  ).join("\n");

/**
 * A walk of the weight row for `rows` rows of the expert, `members[start .. start + rows)`, taken
 * while that many of the scan's rows are left.
 */
const groupedWalk = (
  storage: ExpertLinearStorage,
  shift: number | undefined,
  rows: number,
): string => {
  const each = (line: (row: number) => string): string =>
    Array.from({ length: rows }, (_, row) => line(row)).join("\n");
  const slots = Array.from({ length: EXPERT_LINEAR_UNROLL }, (_, slot) => `${slot}`);
  return `          for (; count - start >= ${rows}u; start = start + ${rows}u) {
${each((row) => `            let m${row} = members[start + ${row}u];`)}
${each(
  (row) =>
    `            let xr${row} = select(m${row}, m${row} / dims.picks, dims.shared_x == 1u) * quads;`,
)}
${each((row) => `            var acc${row} = 0.0;`)}
            var unit = 0u;
            for (; unit + ${EXPERT_LINEAR_UNROLL}u <= units; unit = unit + ${EXPERT_LINEAR_UNROLL}u) {
${slots.map((slot) => groupedLoads(storage, slot, `unit + ${slot}u`, shift)).join("\n")}
${slots.map((slot) => groupedMacs(storage, slot, rows)).join("\n")}
            }
            for (; unit < units; unit = unit + 1u) {
${groupedLoads(storage, "t", "unit", shift)}
${groupedMacs(storage, "t", rows)}
            }
${each((row) => `            out[m${row} * dims.n + col] = acc${row};`)}
          }`;
};

/**
 * The body of the grouped form from a pick's weight row `wrow` on (module doc): the thread of the
 * first row that picked the expert walks the weight row for every row that picked it — the rows
 * found {@link EXPERT_LINEAR_GROUP_ROWS} at a time, each batch in walks of the largest of
 * {@link EXPERT_LINEAR_WALK_ROWS} it still fills.
 */
const groupedBody = (
  storage: ExpertLinearStorage,
  shift: number | undefined,
  scaleBase: string,
): string => `      // The rows that picked this expert share the walk of its weight row, in the thread of the
      // first of them: a row that an earlier row leads is written by that row's thread
      var leads = true;
      for (var earlier = 0u; earlier < row; earlier = earlier + 1u) {
        if (index[earlier] == pick) {
          leads = false;
          break;
        }
      }
      if (leads) {
        let row_base = wrow * units;${scaleBase.replace("\n      ", "\n        ")}
        var next = row;
        while (next < dims.rows) {
          var members: array<u32, ${EXPERT_LINEAR_GROUP_ROWS}>;
          var count = 0u;
          while (next < dims.rows && count < ${EXPERT_LINEAR_GROUP_ROWS}u) {
            if (index[next] == pick) {
              members[count] = next;
              count = count + 1u;
            }
            next = next + 1u;
          }
          var start = 0u;
${walkRows()
  .map((rows) => groupedWalk(storage, shift, rows))
  .join("\n")}
        }
      }`;

/** The head of a pick row up to its weight row `wrow`: the resident form addresses the expert. */
const RESIDENT_HEAD = `    let pick = index[row];
    // an id outside [0, E) breaks the planner's invariant: NaN instead of another expert's row
    if (pick < 0 || u32(pick) >= dims.experts) {
      out[row * dims.n + col] = bitcast<f32>(dims.oob);
    } else {
      let wrow = u32(pick) * dims.n + col;`;

/** The pooled form's head: the slot of the pick's expert, 0 for an expert not resident. */
const POOLED_HEAD = `    let pick = index[row];
    let known = pick >= 0 && u32(pick) < dims.experts;
    // the pick's slot in the pool, -1 when its expert is not resident (read only for an id in range)
    var slot = -1;
    if (known) {
      slot = table[u32(pick)];
    }
    // an id outside [0, E) or a slot outside [0, K) breaks an invariant: NaN instead of another row
    if (!known || slot >= i32(dims.slots)) {
      out[row * dims.n + col] = bitcast<f32>(dims.oob);
    } else if (slot < 0) {
      // not resident: contributes 0, and the host runs the row again once the expert is loaded
      out[row * dims.n + col] = 0.0;
    } else {
      let wrow = u32(slot) * dims.n + col;`;

/**
 * The WGSL of `expert_linear` (module doc). `groupSize` is required for i4 and refused for f32.
 * `pooled` selects the expert cache's form (module doc), which binds the slot table.
 */
export const expertLinearWgsl = (
  storage: ExpertLinearStorage,
  groupSize?: number,
  pooled = false,
  grouped = false,
): string => {
  const shift = groupShift(storage, groupSize);
  const unit = expertLinearUnit(storage);
  const slots = Array.from({ length: EXPERT_LINEAR_UNROLL }, (_, slot) => `${slot}`);
  const loads = slots.map((slot) => wordLoads(storage, slot, `unit + ${slot}u`, shift)).join("\n");
  const macs = slots.map((slot) => wordMacs(storage, slot)).join("\n");
  const scaleBase = isI4(storage)
    ? `
      let scale_base = wrow * (dims.k >> ${shift}u);`
    : "";
  const note =
    storage === "f32"
      ? ""
      : storage === "i4"
        ? ", i4 weight storage"
        : ", i4 weight storage with f16 steps";
  const oneRowBody = `      let row_base = wrow * units;${scaleBase}
      let x_base = select(row, row / dims.picks, dims.shared_x == 1u) * quads;
      var acc = 0.0;
      var unit = 0u;
      // Issue the prefetched words first, then the multiply-adds; words stay in ascending order,
      // so the addition order per output does not depend on the prefetch count
      for (; unit + ${EXPERT_LINEAR_UNROLL}u <= units; unit = unit + ${EXPERT_LINEAR_UNROLL}u) {
${loads}
${macs}
      }
      for (; unit < units; unit = unit + 1u) {
${wordLoads(storage, "t", "unit", shift)}
${wordMacs(storage, "t")}
      }
      out[row * dims.n + col] = acc;`;
  return `// karume expert_linear (y[t, j] = weight[index[t, j]] · x[t (, j)], f32${note}, ${EXPERT_LINEAR_COLS} cols / wg, ${EXPERT_LINEAR_UNROLL} words prefetched${
    grouped ? `, the rows of an expert grouped ${walkRows().join(" / ")} to a walk` : ""
  })
struct Dims {
  rows: u32,
  n: u32,
  k: u32,
  picks: u32,
  shared_x: u32,
  experts: u32,
  oob: u32,
  ${pooled ? "slots" : "pad"}: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> index: array<i32>;
// Row starts are 16 B aligned because k % ${unit} == 0 (guaranteed by expertLinearParams)
@group(0) @binding(3) var<storage, read> w: array<vec4<${storage === "f32" ? "f32" : "u32"}>>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;${scaleBinding(storage)}${
    pooled
      ? `\n@group(0) @binding(${EXPERT_LINEAR_TABLE_BINDING}) var<storage, read> table: array<i32>;`
      : ""
  }${grouped ? wordFunction(storage) : ""}

@compute @workgroup_size(${EXPERT_LINEAR_COLS})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let col = gid.x;
  // No shared memory and no barrier, so edge workgroups may return early
  if (col >= dims.n) {
    return;
  }
  let units = dims.k / ${unit}u;
  let quads = dims.k / 4u;
  var row = gid.y;
  while (row < dims.rows) {
${pooled ? POOLED_HEAD : RESIDENT_HEAD}
${grouped ? groupedBody(storage, shift, scaleBase) : oneRowBody}
    }
    row = row + nwg.y;
  }
}
`;
};

/** The shape of one `expert_linear` dispatch. */
export type ExpertLinearShape = {
  /** Tokens. */
  readonly tokens: number;
  /** Picks per token (the `topk` k). */
  readonly picks: number;
  /** Output features per expert. */
  readonly n: number;
  /** Input features. */
  readonly k: number;
  /** Experts in the weight (`E`). */
  readonly experts: number;
  /** `x` is `[T, k]`, shared by a token's picks (otherwise `[T, picks, k]`). */
  readonly sharedX: boolean;
  /** The pool's slots K of the pooled form (the weight is `[K, n, k]`); absent = resident. */
  readonly slots?: number;
};

/**
 * The uniform Dims (8 words = 32 bytes, the 16-byte unit of the project convention).
 *
 * MUST: `k` is a multiple of the storage's word (4 for f32, 32 for i4) and, for i4, of the group;
 * every flat index the kernel forms (weight words, steps, `x`, `out`) fits in u32 — checked here,
 * since the WGSL multiplies without a guard.
 */
export const expertLinearParams = (
  storage: ExpertLinearStorage,
  shape: ExpertLinearShape,
  groupSize?: number,
): Uint32Array<ArrayBuffer> => {
  const { tokens, picks, n, k, experts, sharedX, slots } = shape;
  assertU32Params("expert_linear params", { tokens, picks, n, k, experts, slots: slots ?? 0 });
  if (slots !== undefined && (slots < 1 || slots > experts)) {
    throw new CodegenError(
      `expert_linear params: ${slots} pool slots for ${experts} experts (1 to E)`,
    );
  }
  // The weight holds the pool's slots under an expert cache, every expert otherwise.
  const held = slots ?? experts;
  const shift = groupShift(storage, groupSize);
  const unit = expertLinearUnit(storage);
  if (picks < 1 || experts < 1 || n < 1 || k < 1) {
    throw new CodegenError(
      `expert_linear params: picks ${picks}, experts ${experts}, n ${n} and k ${k} must be at least 1`,
    );
  }
  if (k % unit !== 0 || (shift !== undefined && k % (1 << shift) !== 0)) {
    throw new CodegenError(
      `expert_linear params: k ${k} is not a multiple of the word (${unit})${
        shift === undefined ? "" : ` and of the group (${1 << shift})`
      }`,
    );
  }
  const rows = tokens * picks;
  const limit = 2 ** 32;
  const largest = [
    rows * n, // out
    (sharedX ? tokens : rows) * (k / 4), // x quads
    held * n * (k / unit), // weight words
    shift === undefined ? 0 : held * n * (k >> shift), // steps
  ];
  if (largest.some((count) => count >= limit)) {
    throw new CodegenError(
      `expert_linear params: an index of the shape (tokens ${tokens}, picks ${picks}, n ${n}, k ${k}, experts ${experts}) exceeds u32`,
    );
  }
  const params = new Uint32Array(8);
  params[0] = rows;
  params[1] = n;
  params[2] = k;
  params[3] = picks;
  params[4] = sharedX ? 1 : 0;
  params[5] = experts;
  params[6] = EXPERT_LINEAR_OOB_BITS;
  params[7] = slots ?? 0;
  return params;
};

/**
 * The workgroup counts: one column per thread along x, the rows along y, capped at the device's
 * per-dimension limit (the kernel grid-strides the rest).
 */
export const expertLinearDispatch = (
  rows: number,
  n: number,
  maxWorkgroupsPerDimension: number,
): readonly [number, number, number] => {
  const columns = Math.ceil(n / EXPERT_LINEAR_COLS);
  if (columns > maxWorkgroupsPerDimension) {
    throw new CodegenError(
      `expert_linear dispatch: ${columns} column workgroups exceed the device limit ${maxWorkgroupsPerDimension}`,
    );
  }
  return [columns, Math.max(1, Math.min(rows, maxWorkgroupsPerDimension)), 1];
};
