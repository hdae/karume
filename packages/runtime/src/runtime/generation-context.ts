/**
 * GenerationContext — the container that owns one generation's mutable state (ADR 0066 decisions
 * 1 / 3 / 5 / 6 / 7).
 *
 * It reads the `states{}` declarations (ADR 0066 decision 2 — format/ir.ts) from the Session
 * (immutable weights + plan cache) and allocates one physical buffer per slot plus one mutable
 * uniform that carries the logical lengths. This is **a fifth lifetime class**, not placed in any
 * of the existing four (weight arena / slot backing / run arena / {@link ResidentTensor}) — the
 * first three have their lifetime unified under `Session.dispose`, and ResidentTensor has a separate
 * dispose contract owned by the GpuContext, so neither matches the granularity of "returned per
 * context" (decision 6). A plain `createBuffer` plus its own bookkeeping is the smallest form.
 *
 * MUST: this holds only ownership, lifetime, the carrier (the logical-length uniform) and **the place
 * for the bind groups that bind the state** (the separate baking of ADR 0066 decision 5). The
 * execution side decides how to bind; the context is only the container that keeps that product
 * paired with the backing's generation id ({@link GenerationContextInternals}).
 * MUST: do not import executor.ts (keep the one-way Session → context import intact, in types too).
 * The surface borrowed from the Session is only the few structural fields of
 * {@link GenerationContextHost} (the same style as `RecipeBuilderContext`).
 */

import { evalDim, parseDim } from "../format/dims.ts";
import type { IrDim, IrGraph } from "../format/ir.ts";
import { STORAGE_USAGE } from "../gpu/arena.ts";
import { assertDeviceUsable, type GpuContext, RUNTIME_INTERNAL } from "../gpu/device.ts";
import { discardFailureScopes, popFailureScopes, pushFailureScopes } from "../gpu/error-scope.ts";
import { BUFFER_USAGE } from "../gpu/webgpu-constants.ts";
import { numel, stateWindow } from "../ops.ts";
import { ExecutionError, type SymbolBindings } from "./plan.ts";
import type { BakedGroups } from "./recipe.ts";
import type { GenerationContextSpec } from "./session-types.ts";

/**
 * Byte size of the logical-length uniform (two words, `pastLength` / `queryLength` — ADR 0066
 * addendum 4).
 *
 * MUST: exactly two words. It is the size of the kernel-side `struct { past: u32, query: u32 }` and
 * has the same form as the existing params uniform (`struct Params { rows: u32, dim: u32 }` —
 * codegen/reduce.ts), so these 8 bytes satisfy the minimum binding size.
 */
export const LENGTHS_BYTES: number = 8;

/**
 * Usage of the logical-length uniform.
 *
 * `COPY_SRC` is set to keep **the only observation point from which the contents can be read
 * back** — `queue.writeBuffer` becomes a no-op without even a warning on an invalid buffer or an
 * alignment violation, so reading back is the only way to confirm "the written value is really
 * there" (the execution path never issues this copy).
 */
const LENGTHS_USAGE = BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC;

/** State slots are f32 only (`STATE_DTYPES` — an f16 option is reserved by ADR 0066 addendum 5). */
export const STATE_ELEMENT_BYTES: number = 4;

/**
 * Upper bound of a logical length (carried in a `Uint32Array` / WGSL `u32` — ADR 0066 addendum 4).
 *
 * MUST: a safe integer is not enough. Assigning to a `Uint32Array` **silently truncates**
 * out-of-range values, so `queryLength = 2**32` becomes 0 in the uniform, and the host's logical
 * length and the value the GPU sees split with no exception or warning (`writeBuffer` does not
 * check values).
 */
const MAX_LOGICAL_LENGTH = 0xffffffff;

/**
 * The physical backing of one state slot (bindings resolved — never reallocated during execution =
 * static physical storage).
 */
export type StateSlotBacking = {
  readonly buffer: GPUBuffer;
  /** The binding-resolved concrete shape including capacity (same rank as the declared shape). */
  readonly shape: readonly number[];
  readonly byteLength: number;
};

/**
 * The surface the context borrows from the Session (a structural surface with **only the fields it
 * needs**).
 *
 * MUST: do not import `SessionState` from executor.ts (the one-way import of the module doc).
 * The Session passes this bundle, built from its own state.
 */
export type GenerationContextHost = {
  readonly gpu: GpuContext;
  readonly graph: IrGraph;
  /**
   * Drains unsubmitted encoding (flush-before-destroy — ADR 0004. Implemented by
   * `SubmitScheduler.flush`).
   */
  flush(): Promise<void>;
  /**
   * Queues onto the same chain as the Session's run / enqueue / dispose (implemented by
   * `Session.#serialize`).
   */
  serialize<T>(body: () => Promise<T>): Promise<T>;
  /**
   * Reports a settled disposal to the Session's diagnostics (removes it from the live set of
   * `stateBacking.residentBytes`).
   */
  forget(context: GenerationContext): void;
  /**
   * Allocates a new context and registers it in the Session's live set (implemented by
   * `Session.#adopt` — the single registration path shared by `createGenerationContext` and
   * {@link GenerationContext.fork}).
   *
   * MUST: check the Session's dispose **synchronously at the time of the call**, and call `allocate`
   * **synchronously** right after (no await in between). fork relies on finishing the submit of the
   * source slots' copy within this section and places no reservation on the source, so a single
   * await here makes it copy the state after the writes of a run issued to the source right after
   * (a mix-up with no exception).
   * MUST: register after the allocation settles, and check dispose once more before that (do not
   * register into a Session disposed across the allocation's await — tear the allocated context
   * down before failing).
   *
   * @param refusal the `ExecutionError` message thrown when the Session is already disposed (it adds
   *   the caller's context — the same message whether the check at call time or the one after
   *   allocation fails).
   */
  adopt(allocate: () => Promise<GenerationContext>, refusal: string): Promise<GenerationContext>;
};

/**
 * The runtime-internal surface of {@link GenerationContext} (backing and progress that do not map
 * to a user story).
 *
 * MUST: do not expose it under a plain name (the thin surface of ADR 0008). Advancing the logical
 * length is a contract that happens on a successful run (decision 6 — manual host-side addition is
 * not an API), so `advance` / `poison` / `writeLengths` are **called only by the execution
 * integration (the generation surface of `Session.run`)**.
 */
type GenerationContextInternals = {
  /** The state slots' backing (a generation run's bind groups bind from here — decision 5). */
  readonly slots: ReadonlyMap<string, StateSlotBacking>;
  /**
   * Whether this context is a **borrower** (ADR 0096, addendum of 2026-09-08 on stage 2). When
   * true, its logical length does not advance on its own, and apart from the lengths the slot
   * backings belong to the lender.
   *
   * MUST: the execution integration (`Session.run`) decides "do not advance" from this one field.
   * Silently turning the borrower's `advance` / `defer` into no-ops would be indistinguishable from
   * a form where only the lender's P moves.
   */
  readonly borrowing: boolean;
  /**
   * This context's device (the same-device check when borrowing — slots of another device cannot be
   * bound).
   */
  readonly gpu: GpuContext;
  /**
   * Sliding slot name → window width (the surface for checking **lender and borrower agree** when
   * borrowing).
   *
   * MUST: logical col → physical row uses the same formula for reads and writes (ADR 0067 decision
   * 4). `validateGraphContracts` checks agreement within a graph, but lending **spans two graphs**,
   * so comparing here is the only detection point (a wider borrower window reads rows that fell out
   * of the window as the past).
   */
  readonly slidingSlots: ReadonlyMap<string, number>;
  /** Whether it can serve as a lender (once, when a borrowing context is created). */
  assertLendable(where: string): void;
  /** Registers a borrower (while it lives, the lender's `dispose` is refused). */
  registerBorrower(borrower: GenerationContext): void;
  /** Removes a borrower's registration (the borrower's `dispose`). */
  forgetBorrower(borrower: GenerationContext): void;
  /**
   * The set of physical chunk row counts this context allows = `{1} ∪ chunkBuckets ∪ {chunkLength}`
   * (the only form read by the pre-run check `assertGenerationRun` — src/runtime/recipe.ts).
   *
   * MUST: built only once, at context creation. Rebuilding it per run would put a Set construction
   * per bucket on the decode hot path every step (although the number of execution shapes is fixed
   * here).
   */
  readonly allowedRows: ReadonlySet<number>;
  /** The logical-length uniform (recipes reference it through a fixed binding — addendum 4). */
  readonly lengths: GPUBuffer;
  /**
   * GPU bytes this context keeps resident (the source of the diagnostic
   * `stateBacking.residentBytes`).
   */
  readonly bytes: number;
  /**
   * Resolved bindings of the capacity symbols (the checked copy of `spec.bindings` — ADR 0066
   * addendum 7).
   *
   * A symbol that appears in **both** states and inputs is decided independently at two places: the
   * context side (capacity) and the run side (input shapes). The execution integration checks this
   * table against the run's resolved bindings and fails a split loudly (running with them split
   * silently reads state whose allocated capacity and plan were built from different values).
   */
  readonly bindings: SymbolBindings;
  /**
   * Accepts one in-flight generation run (taken **in the synchronous section of `Session.run`** and
   * always returned when the run settles). If it cannot be taken, it fails loudly (after a dispose
   * request, after poisoning, after device loss, and when **the same context already has one
   * unsettled run** — the same "logical length moved from the side" as in the doc of
   * {@link GenerationContext.rewind} happens by issuing the second run itself).
   */
  acquireRun(kind?: "batch"): void;
  assertCanDispose(): void;
  /**
   * Returns the in-flight generation run (always exactly once, on both the success and the failure
   * path).
   */
  releaseRun(): void;
  /**
   * The internal read of the logical length (once, at the head of a run — it is cut off **after
   * the disposal body runs**).
   *
   * MUST: do not read the user-facing `pastLength` from a run. dispose has two steps (closing
   * admission → the chained disposal body), and an accepted run executes between the first and the
   * second, so a user-facing check would let "`run(); dispose()` issued without await" kill the
   * accepted run.
   */
  pastLength(): number;
  /**
   * The context-side bind groups baked for the backing of `token` (undefined if none = bake again).
   *
   * MUST: always match `token` on lookup (**never create a lookup without a token**). A bundle also
   * holds buffers owned by the backing (slots / inputs), so dispatching a bundle baked under the
   * token of a retired backing reads destroyed buffers — the argument keeps the order of matching,
   * re-baking and dispatch in one place (`#generationGroups` in the executor).
   */
  bakedGroups(token: number): BakedGroups | undefined;
  /** Stores a baked bundle (it can be looked up only while the backing of `token` is kept). */
  setBakedGroups(token: number, groups: BakedGroups): void;
  /**
   * Drops the bundle of the backing of `token` (called when the Session retires a backing).
   * MUST: drop it at retirement — the bundle holds the retired backing, so keeping it extends that
   * lifetime by the reference.
   */
  dropBakedGroups(token: number): void;
  /**
   * Writes the logical lengths (before every run's encode — the doc of {@link GenerationContext}).
   * `pastLength` is **the value the caller captured at the head of the run** (checked against the
   * current internal value).
   */
  writeLengths(pastLength: number, queryLength: number): void;
  /**
   * Advances the logical length (**only on a successful run** — decision 6). The captured P is
   * checked as in `writeLengths`.
   */
  advance(pastLength: number, queryLength: number): void;
  /**
   * **Holds the logical length pending without advancing it** (when a `commit: "deferred"` run
   * returned without an exception).
   *
   * It hands the right to advance to the host exactly once, to {@link GenerationContext.commit}.
   * While a pending commit exists, new run leases and `rewind` are refused, so no window opens in
   * which "the logical length the GPU saw" and "the basis of progress" diverge (the ban on double
   * bookkeeping of ADR 0066 decision 6 holds in this form too — there is still only one path that
   * can move the logical length). The captured P is checked as in `advance`.
   */
  defer(pastLength: number, queryLength: number): void;
  /** Poisons the context (failure of a run containing a state-changing dispatch — addendum 3). */
  poison(reason: string): void;
};

/**
 * Sliding slot name → window width `W` (from the node attrs `window` — ADR 0067 decision 4).
 *
 * MUST: the information exists only on the node side (the `graph.states` declaration holds only the
 * capacity; the window is declared by **the referencing nodes**). `assertStateOrder` in
 * `validateGraphContracts` has already checked at Session construction that every node touching
 * the same slot agrees on `window`, so here one sliding declaration is enough to treat the slot as
 * sliding.
 */
const slidingSlotWindows = (graph: IrGraph): ReadonlyMap<string, number> => {
  const sliding = new Map<string, number>();
  graph.nodes.forEach((node, index) => {
    const slots = Object.values(node.states);
    if (slots.length === 0) return;
    const window = stateWindow(node.attrs, `nodes[${index}] (${node.op})`);
    if (window === undefined) return;
    for (const slot of slots) sliding.set(slot, window);
  });
  return sliding;
};

/**
 * The **slack** of the sliding slots (the minimum of `capacity − window` — undefined if there is no
 * sliding slot).
 *
 * Slack is the room for "rows written ahead of the logical length", and it is meaningful together
 * with the ring modulus being the capacity, not the window (`stateSlotRowWgsl` in
 * `src/kernels/state-attention.ts`). The host must not speculatively write more rows than this — if
 * it does, rejected rows overwrite live past KV.
 *
 * MUST: the capacity axis is **axis 2 of the slot shape** (`[B,Hkv,C,D]` — the contract of the
 * states-form ops. On the execution-plan side, `buildStateAttention` / `buildStateAppend` in
 * `recipe-builders/attention.ts` read the same axis). Reading another axis here would make the
 * published slack a number unrelated to the actual physical row count.
 */
const slidingSlackRows = (
  slots: ReadonlyMap<string, StateSlotBacking>,
  windows: ReadonlyMap<string, number>,
): number | undefined => {
  let slack: number | undefined;
  for (const [name, window] of windows) {
    // The IR layer has already checked referential integrity (states declarations vs the nodes'
    // states fields), so the name always resolves. The shape layer fails a shape with too low a rank
    // at run time, but silently skipping it here would publish either an undefined that cannot be
    // told from "no slack" or an oversized slack taken from another axis.
    const shape = slots.get(name)?.shape;
    if (shape === undefined || shape.length < 3) {
      throw new ExecutionError(
        `state '${name}': sliding 宣言（window ${window}）に対して容量形 ` +
          `[${shape?.join(",") ?? "?"}] から行容量 C（軸 2）が読めない`,
      );
    }
    const rows = shape[2] - window;
    if (rows < 0) {
      throw new ExecutionError(
        `state '${name}': window ${window} が行容量 ${shape[2]} を超える（ADR 0067 決定 4 ③）`,
      );
    }
    slack = slack === undefined ? rows : Math.min(slack, rows);
  }
  return slack;
};

/**
 * Range check of `chunkLength` (a GPU-independent pure function).
 *
 * MUST: the upper bound is u32 (through the `queryLength ≤ chunkLength` check this also fixes the
 * upper bound of the logical length — {@link MAX_LOGICAL_LENGTH}). The estimator (estimate.ts) goes
 * through the same check — if only the estimate returned a normal value for a spec the real
 * construction rejects, admission numbers would be given to a configuration that cannot be built.
 */
export const assertChunkLength = (chunkLength: number): void => {
  if (!Number.isSafeInteger(chunkLength) || chunkLength < 1 || chunkLength > MAX_LOGICAL_LENGTH) {
    throw new ExecutionError(
      `chunkLength ${chunkLength} が 1..${MAX_LOGICAL_LENGTH} の整数でない` +
        "（固定長 prefill chunk の行数・搬送先は u32 — ADR 0066 決定 4 / 追記 4）",
    );
  }
};

/**
 * Range and order check of `chunkBuckets` (a GPU-independent pure function — the same style as
 * {@link assertChunkLength}).
 *
 * Only "a **strictly increasing** list of integers ≥ 2 and < `chunkLength`" is accepted. Three
 * reasons for rejection: `1` is the decode shape itself and adds no execution shape; values ≥
 * `chunkLength` lie outside the `queryLength ≤ chunkLength` contract of {@link GenerationContext}
 * (rows that do not fit the declared shape appear); duplicates and descending order keep the choice
 * "smallest bucket ≥ `queryLength`" from being decided by a linear scan (if the caller kept its own
 * sorting, the set the context allows and the set it chooses from would grow apart).
 *
 * MUST: the estimate (estimate.ts) goes through the same check — if only the estimate returned a
 * normal value for a spec the real construction rejects, admission numbers would be given to a
 * configuration that cannot be built.
 */
export const assertChunkBuckets = (
  chunkBuckets: readonly number[] | undefined,
  chunkLength: number,
): void => {
  if (chunkBuckets === undefined) return;
  let previous = 1;
  for (const [index, rows] of chunkBuckets.entries()) {
    if (!Number.isSafeInteger(rows) || rows < 2 || rows >= chunkLength) {
      throw new ExecutionError(
        `chunkBuckets[${index}] ${rows} が 2..${chunkLength - 1} の整数でない` +
          "（1 は decode 形・chunkLength は prefill 形そのもの — ADR 0066 決定 4 / 追記〈バケット〉）",
      );
    }
    if (rows <= previous) {
      throw new ExecutionError(
        `chunkBuckets[${index}] ${rows} が直前の ${previous} 以下（狭義昇順でない）` +
          "。queryLength 以上の最小バケットを選ぶ側が並べ替えを持たない前提",
      );
    }
    previous = rows;
  }
};

/**
 * The set of physical chunk row counts a context allows — `{1}` (decode shape) ∪ `chunkBuckets` ∪
 * `{chunkLength}` (prefill shape). Inserted in ascending order (the diagnostic of
 * `assertChunkRowsAllowed` lists them in this iteration order).
 *
 * MUST: context construction and the estimate (estimate.ts) build the set with this one function
 * (if only one side were fixed, the estimate would return numbers for shapes a run rejects). It
 * assumes {@link assertChunkBuckets} has checked the range first.
 */
export const allowedChunkRows = (
  chunkLength: number,
  chunkBuckets: readonly number[],
): ReadonlySet<number> => new Set([1, ...chunkBuckets, chunkLength]);

/**
 * Checks `spec.bindings` and copies them into a null-prototype table.
 *
 * MUST: the bindings container has a null prototype (the same reason as `bindSymbols` in plan.ts —
 * the symbol grammar `[A-Za-z_][A-Za-z0-9_]*` matches "__proto__", and on a plain `{}` the
 * assignment turns into setting [[Prototype]] without creating an own property).
 * MUST: a symbolic capacity is decided **only by the values given here**. States are not a binding
 * source (ADR 0066 decision 2) and the context has no inputs at all, so inferring from input shapes
 * is impossible in principle.
 */
export const resolveBindings = (
  graph: IrGraph,
  bindings: SymbolBindings | undefined,
): SymbolBindings => {
  const symbols = new Set(graph.symbols);
  const resolved: Record<string, number> = Object.create(null);
  for (const [sym, value] of Object.entries(bindings ?? {})) {
    if (!symbols.has(sym)) {
      throw new ExecutionError(
        `束縛 '${sym}' はグラフの symbols [${graph.symbols.join(", ")}] に無い`,
      );
    }
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new ExecutionError(`束縛 '${sym}': ${value} が非負整数でない`);
    }
    resolved[sym] = value;
  }
  return resolved;
};

/**
 * Concretises one slot's declared shape with the bindings (the concrete shape including capacity —
 * ADR 0066 decision 2).
 *
 * MUST: an unbound symbol fails loudly. Filling it with 0 or 1 would start generation on a slot
 * without enough capacity, ending in a silent OOB.
 */
export const resolveSlotShape = (
  name: string,
  shape: readonly IrDim[],
  bindings: SymbolBindings,
): number[] =>
  shape.map((dim, index) => {
    if (typeof dim === "number") return dim;
    const expr = parseDim(dim);
    if (!Object.hasOwn(bindings, expr.sym)) {
      throw new ExecutionError(
        `state '${name}': 記号次元 '${dim}' のシンボル '${expr.sym}' が束縛されていない` +
          "（createGenerationContext の bindings で与えること — states は束縛源にならない）",
      );
    }
    const extent = evalDim(expr, bindings);
    if (extent < 1) {
      throw new ExecutionError(
        `state '${name}': 次元 ${index} の容量 ${extent} が正でない` +
          `（束縛 '${expr.sym}' = ${bindings[expr.sym]}）`,
      );
    }
    return extent;
  });

/**
 * The device's absolute limits used for the per-slot check (ADR 0089 — physical free VRAM is not
 * guessed).
 *
 * `GenerationContext.create` passes `GpuContext.limits` (both fields are granted values); the
 * estimator passes the fields of the same name in `EstimateOptions` (both optional) as they are.
 */
export type SlotLimits = {
  readonly maxStorageBufferBindingSize?: number;
  readonly maxBufferSize?: number;
};

/**
 * Checks one slot's byte size against the device's absolute limits (ADR 0066 addendum 5).
 *
 * MUST: check both limits. `maxStorageBufferBindingSize ≤ maxBufferSize` is a relation kept by the
 * side that plans the device (`planRequiredLimits` in gpu/device.ts), not a guarantee for a
 * GpuContext passed in from outside — checking only the binding limit would, on a device where the
 * relation is broken, let through "a size whose allocation itself cannot succeed" and hold an
 * invalid buffer.
 * MUST: the estimate (estimate.ts) goes through **this one function** too. A copy would bring back
 * the form where only the estimate returns numbers for a capacity the real construction rejects
 * (numbers at startup, a failure on the first turn).
 *
 * Fields without a limit are not checked — estimate callers may have no device, and inventing a
 * default there would make the estimator alone claim "a boundary that no device actually has".
 */
export const assertSlotWithinLimits = (
  name: string,
  shape: readonly number[],
  byteLength: number,
  limits: SlotLimits,
): void => {
  for (const limitName of ["maxStorageBufferBindingSize", "maxBufferSize"] as const) {
    const limit = limits[limitName];
    // MUST: a capacity whose single-slot byte size exceeds a limit fails loudly (addendum 5).
    // There is no split-and-bind form (KV is one contiguous capacity — the explicit choice of
    // decision 8), so an overrun can only fail at the call site as an error in capacity design.
    if (limit !== undefined && byteLength > limit) {
      throw new ExecutionError(
        `state '${name}': 容量 [${shape.join(",")}] の ${byteLength} バイトが ` +
          `${limitName} ${limit} バイトを超える（ADR 0066 追記 5）。` +
          "容量を下げるか、スロットを分けてグラフを組み直すこと",
      );
    }
  }
};

/**
 * The borrowing state a borrowing context holds (ADR 0096, addendum of 2026-09-08 on stage 2). It
 * holds **the lender itself** (not a copy — a copy would raise "which one is real").
 */
type BorrowState = {
  readonly lender: GenerationContext;
  /**
   * A copy of the lender's `slidingSlack` (the borrower has no slots of its own, so it cannot
   * derive it).
   */
  readonly slidingSlack: number | undefined;
};

export class GenerationContext {
  /**
   * Row count of a fixed-length prefill chunk (ADR 0066 decision 4 — a plan-time constant fixed at
   * context creation). decode is the fixed `queryLength = 1` shape, and the last prefill chunk is
   * filled with padding (cut by `queryLength`).
   */
  readonly chunkLength: number;
  /**
   * Physical chunk row counts allowed as prefill shapes in addition to `chunkLength` (strictly
   * increasing, frozen copy — ADR 0066 decision 4 / addendum 10 (buckets)). Empty if none was
   * declared.
   *
   * Per chunk, the caller picks the smallest value ≥ `queryLength` as the physical row count
   * (`chunkLength` if none). `chunkLength` **stays the maximum** when buckets are added, and the
   * bound `queryLength ≤ chunkLength` does not move either.
   */
  readonly chunkBuckets: readonly number[];
  /**
   * The **slack rows** of the sliding slots = the minimum of `capacity − window` (undefined if there
   * is no sliding slot). The ring modulus is the capacity, not the window (`stateSlotRowWgsl` in
   * `src/kernels/state-attention.ts`), so up to this many rows can be written to the physical ring
   * **ahead of the logical length** without rows that may be rejected (= not accepted by `commit`)
   * overwriting live past KV.
   *
   * This is the upper bound of `queryLength` in speculative decoding (verifying a draft before fixing
   * the accepted row count), and a `commit: "deferred"` run enforces `queryLength ≤ slidingSlack` in
   * the synchronous section of issue (`Session.run`). The reader with the lowest lower edge is the
   * **borrower** (readonly = the drafter) at `P+m−W`, and the condition for the logical column
   * `P+j−C` overwritten by rejected row `j` to fall below it is `j − m < capacity − window` = this
   * slack (`m` = committed rows, including `commit(0)`).
   */
  readonly slidingSlack: number | undefined;
  /** The runtime-internal surface (not a surface for users). */
  readonly [RUNTIME_INTERNAL]: GenerationContextInternals;
  readonly #host: GenerationContextHost;
  readonly #slots: ReadonlyMap<string, StateSlotBacking>;
  /** Borrowing state (undefined = an ordinary context with its own slots). */
  readonly #borrow: BorrowState | undefined;
  /**
   * The contexts borrowing this context (ADR 0096, addendum of 2026-09-08 on stage 2).
   *
   * MUST: `dispose()` while any is alive fails loudly. The borrowers' bind groups hold the lender's
   * slot buffers, so destroying them makes the next draft read destroyed buffers.
   */
  readonly #borrowers = new Set<GenerationContext>();
  /**
   * Sliding slot name → window width (the blanket refusal condition of
   * {@link GenerationContext.rewind} — ADR 0066 addendum 2. The widths also feed
   * {@link GenerationContext.slidingSlack}).
   */
  readonly #slidingSlots: ReadonlyMap<string, number>;
  readonly #lengths: GPUBuffer;
  /**
   * The values written for the logical lengths. **The whole range is written every time** (a partial
   * write could leave a half-updated pair, and dispatches would run with a mixed uniform: "pastLength
   * new, queryLength from the previous step").
   */
  readonly #lengthValues = new Uint32Array(2);
  #pastLength = 0;
  /**
   * The deferred run waiting for commit (kept until {@link GenerationContext.commit} folds it).
   *
   * MUST: while it is set, run leases and `rewind` are refused. The physical ring already holds
   * `queryLength` written rows while only the logical length is stopped; letting the next run or a
   * rewind through would have two places decide separately "how far the KV is committed".
   */
  #pending: { readonly pastLength: number; readonly queryLength: number } | undefined;
  /**
   * Context-side baked bind group bundles (backing generation id → bundle — ADR 0066 decision 5).
   * The Session keeps several backings (perf-ledger H-15), so there is one bundle per kept backing,
   * and the Session drops the bundles of retired backings with `dropBakedGroups`. **A GPUBindGroup
   * needs no destroy** (left to GC), but it extends the lifetime of the backing-owned buffers it
   * holds, so it is let go at retirement.
   */
  readonly #baked = new Map<number, BakedGroups>();
  /**
   * The poison reason (addendum 3). Once set, every operation except `dispose` is refused (reads
   * included).
   */
  #poisoned: string | undefined;
  /**
   * At most one use reservation. A run keeps it until it completes, a batch until the section's
   * final settlement. During a batch only the Session's execution body returns early, so unlike an
   * ordinary run, dispose is refused.
   */
  #runKind: "run" | "batch" | undefined;
  /**
   * Step 1 of dispose — **closing admission** (set synchronously). Later run admission and
   * user-facing reads and writes are refused, but the internal surface of accepted runs (carrying
   * and advancing the logical length) **still passes**.
   */
  #disposeRequested = false;
  /** Step 2 of dispose — the disposal body has started (the internal surface closes here too). */
  #disposed = false;
  #disposal: Promise<void> | undefined;

  /**
   * MUST: the only construction entry points are {@link GenerationContext.create} and
   * {@link GenerationContext.fork} (no bypassing the errorScope check).
   */
  private constructor(
    host: GenerationContextHost,
    slots: ReadonlyMap<string, StateSlotBacking>,
    slidingSlots: ReadonlyMap<string, number>,
    lengths: GPUBuffer,
    chunkLength: number,
    chunkBuckets: readonly number[],
    bindings: SymbolBindings,
    borrow?: BorrowState,
  ) {
    this.#host = host;
    this.#slots = slots;
    this.#slidingSlots = slidingSlots;
    this.#borrow = borrow;
    // A borrower's slack is **a copy of the lender's** (ADR 0096, addendum of 2026-09-08 on stage
    // 2). Deriving it locally would claim "no slack" when the lender has a sliding declaration but
    // the borrower's readonly nodes declare no window, loosening the speculation bound on the host.
    this.slidingSlack =
      borrow === undefined ? slidingSlackRows(slots, slidingSlots) : borrow.slidingSlack;
    this.#lengths = lengths;
    this.chunkLength = chunkLength;
    // Frozen copy: if the caller's array were modified later, the allowed set (below) and the
    // public surface would diverge.
    this.chunkBuckets = Object.freeze([...chunkBuckets]);
    this[RUNTIME_INTERNAL] = {
      slots,
      borrowing: borrow !== undefined,
      gpu: host.gpu,
      slidingSlots,
      allowedRows: allowedChunkRows(chunkLength, this.chunkBuckets),
      lengths,
      // The capacity is fixed (static physical storage — ADR 0066 decision 3), so it is summed once
      // here.
      // MUST: a borrower does not count the slots' bytes (the backing is owned by the lending
      // context, and if both counted, the diagnostic `stateBacking.residentBytes` would count the
      // same VRAM twice).
      bytes:
        borrow === undefined
          ? [...slots.values()].reduce((total, slot) => total + slot.byteLength, 0) + LENGTHS_BYTES
          : LENGTHS_BYTES,
      bindings,
      assertLendable: (where: string): void => {
        this.#assertUsable(where);
        if (this.#borrow !== undefined) {
          throw new ExecutionError(
            `${where}: 借り手 context は貸し手になれない（借用の連鎖は持たない — ` +
              "ADR 0096 段 2 §2.1）",
          );
        }
      },
      registerBorrower: (borrower: GenerationContext): void => {
        this.#assertUsable("createGenerationContext(borrow)");
        this.#borrowers.add(borrower);
      },
      forgetBorrower: (borrower: GenerationContext): void => {
        this.#borrowers.delete(borrower);
      },
      acquireRun: (kind): void => {
        this.#assertUsable("run");
        // MUST: while an uncommitted deferred run exists, do not issue the next one. The second
        // run would capture P before "how many of the physical rows the first one wrote are
        // committed" is decided, so once commit(rows) moves the logical length it disagrees with
        // the P the GPU saw (a position shift with no exception).
        if (this.#pending !== undefined) {
          throw new ExecutionError(
            `run: commit 待ちの generation run がある（pastLength ${this.#pending.pastLength} + ` +
              `queryLength ${this.#pending.queryLength} まで書き込み済み）。` +
              "context.commit(rows) で受理した行数を確定させてから次を発行すること",
          );
        }
        // MUST: at most one unsettled run per context. The second would build its uniform and
        // dispatches from the logical length P' the first advanced to, but position inputs (RoPE's
        // position_ids and the like) are **ordinary graph inputs the caller built at issue time**,
        // and the runtime does not look inside them. So the KV logical length stays right while the
        // positions quietly shift (silent wrong values with no exception or warning). Await each
        // run before issuing the next — generations meant to run in parallel use separate contexts.
        if (this.#runKind !== undefined) {
          throw new ExecutionError(
            "run: 進行中の generation run がある GenerationContext へ並行発行された" +
              "（2 本目は 1 本目の進行後の論理長で走る一方、位置入力は発行時の論理長のままなので、" +
              "例外の出ない位置ずれになる）。前の run の決着を await してから発行するか、" +
              "context を分けること",
          );
        }
        // MUST: a borrower's run **also takes the lender's lease** (ADR 0096, addendum of
        // 2026-09-08 on stage 2 — at the same point as the existing acquireRun). This serialises it
        // with the lender's run / commit / rewind, and the lender's poison, uncommitted deferred
        // run and in-flight run become the borrower's reasons for refusal as they are.
        // MUST: take it only after all of this context's own checks pass (failing after taking it
        // would leave a lease on the lender with nobody to return it, refusing every later rewind /
        // commit forever).
        this.#borrow?.lender[RUNTIME_INTERNAL].acquireRun(kind);
        this.#runKind = kind ?? "run";
      },
      assertCanDispose: (): void => this.#assertCanDispose(),
      releaseRun: (): void => {
        if (this.#runKind === undefined) {
          throw new ExecutionError(
            "releaseRun: 進行中の generation run が居ないのにリースを返した（簿記の破れ）",
          );
        }
        this.#runKind = undefined;
        this.#borrow?.lender[RUNTIME_INTERNAL].releaseRun();
      },
      pastLength: (): number => {
        this.#assertInternalUsable("pastLength");
        // A borrower's logical length is **a copy of the lender's P** (ADR 0096, addendum of
        // 2026-09-08 on stage 2). It is copied only at this one point, the head of the run, and the
        // later `writeLengths` checks agreement with this value (while the lease is held the
        // lender's P does not move, so the copy and the original agree until the run settles).
        if (this.#borrow !== undefined) {
          this.#pastLength = this.#borrow.lender[RUNTIME_INTERNAL].pastLength();
        }
        return this.#pastLength;
      },
      bakedGroups: (token: number): BakedGroups | undefined => this.#baked.get(token),
      setBakedGroups: (token: number, groups: BakedGroups): void => {
        this.#baked.set(token, groups);
      },
      dropBakedGroups: (token: number): void => {
        this.#baked.delete(token);
      },
      writeLengths: (pastLength: number, queryLength: number): void =>
        this.#writeLengths(pastLength, queryLength),
      advance: (pastLength: number, queryLength: number): void =>
        this.#advance(pastLength, queryLength),
      defer: (pastLength: number, queryLength: number): void =>
        this.#defer(pastLength, queryLength),
      poison: (reason: string): void => this.#poison(reason),
    };
  }

  /**
   * Fixes the slot capacities and `chunkLength` and allocates physically (ADR 0066 decision 6).
   *
   * MUST: it is async in order to wrap the allocation in an errorScope (decision 6's "an allocation
   * failure fails loudly through the out-of-memory errorScope"). A `createBuffer` over a limit or out
   * of room throws no synchronous exception and **returns an invalid buffer**, so without the scope
   * the generation loop would run with empty KV bound.
   * MUST: the limit check (addendum 5) runs **before** allocation. After it, an overrun would become
   * a second-hand diagnostic, "validation from an invalid buffer", and which slot was too large would
   * be lost.
   */
  static async create(
    host: GenerationContextHost,
    spec: GenerationContextSpec,
  ): Promise<GenerationContext> {
    const { gpu, graph } = host;
    // MUST: check that the device is usable before allocating (decision 7). createBuffer on a lost
    // device just returns an invalid buffer, so letting it through yields "a context with empty KV"
    // (the errorScope reports no failure after device loss either).
    assertDeviceUsable(gpu, "createGenerationContext");
    const names = Object.keys(graph.states);
    // MUST: refuse graphs without state. A context is "the owner of one generation's mutable
    // state", so for a model with zero states declarations the container itself is meaningless —
    // this is the detection line for a mix-up (created from another model's Session).
    if (names.length === 0) {
      throw new ExecutionError(
        "このグラフは states 宣言を持たない（GenerationContext は state スロットの所有者なので、" +
          "state の無いモデルでは作れない — 1-shot 実行は Session.run / enqueue をそのまま使う）",
      );
    }
    // MUST: carry the checked values themselves (do not read `spec` again). Between the check and
    // the constructor call below lies the allocation's await (`raceDeviceLost`); if the caller
    // modified the spec it passed in that window, unchecked values would pass (TOCTOU) — use only
    // what is captured here, **arrays and lengths alike** (reading `chunkLength` again would let
    // "checked with M=8 but the context has M=1" stand with no exception).
    const chunkLength = spec.chunkLength;
    assertChunkLength(chunkLength);
    const chunkBuckets = Object.freeze([...(spec.chunkBuckets ?? [])]);
    assertChunkBuckets(chunkBuckets, chunkLength);
    // Borrowed slots (external — ADR 0096, addendum of 2026-09-08 on stage 2) and `borrow` come
    // **in pairs**. Either one alone fails loudly in both directions (allocating locally despite
    // external slots reads an empty past; `borrow` alone grabs slots from the lender that nobody
    // reads).
    const external = names.filter((name) => graph.states[name].external);
    if (external.length > 0 && spec.borrow === undefined) {
      throw new ExecutionError(
        `このグラフは external な state スロット [${external.join(", ")}] を持つ` +
          "（借り物の実体は貸し手 context にあるので createGenerationContext({ borrow }) が要る" +
          " — ADR 0096 段 2 §2.1）",
      );
    }
    if (spec.borrow !== undefined) {
      if (external.length === 0) {
        throw new ExecutionError(
          "borrow を指定できるのは全スロットが external なグラフだけ" +
            `（自前スロット [${names.join(", ")}] を持つ — ADR 0096 段 2 §2.1）`,
        );
      }
      return await GenerationContext.#createBorrowed(host, spec, spec.borrow, names);
    }
    const bindings = resolveBindings(graph, spec.bindings);
    const planned = names.map((name) => {
      const shape = resolveSlotShape(name, graph.states[name].shape, bindings);
      const byteLength = numel(shape) * STATE_ELEMENT_BYTES;
      assertSlotWithinLimits(name, shape, byteLength, gpu.limits);
      return { name, shape, byteLength };
    });

    // MUST: no await between push and **issuing** the pop (the basis that prevents interleaving on
    // the per-device LIFO — the third rule of "Invariants of errorScope sections" at the top of
    // GpuContext. It completes in a synchronous section, so no lock is needed).
    pushFailureScopes(gpu.device);
    const where = "GenerationContext の state 確保";
    const created: GPUBuffer[] = [];
    const slots = new Map<string, StateSlotBacking>();
    let popped = false;
    let context: GenerationContext | undefined;
    try {
      for (const { name, shape, byteLength } of planned) {
        const buffer = gpu.device.createBuffer({
          label: `state '${name}'`,
          size: byteLength,
          usage: STORAGE_USAGE,
        });
        created.push(buffer);
        slots.set(name, { buffer, shape, byteLength });
      }
      const lengths = gpu.device.createBuffer({
        label: "generation lengths",
        size: LENGTHS_BYTES,
        usage: LENGTHS_USAGE,
      });
      created.push(lengths);
      const pending = popFailureScopes(gpu.device, where);
      popped = true;
      // MUST: race it, because an implementation may never resolve popErrorScope after loss (the
      // same reason as mapAsync in ResidentTensor.read) — this turns a hang into a failure and
      // normalises loss to GpuDeviceLostError.
      const failure = await gpu[RUNTIME_INTERNAL].raceDeviceLost(pending, where);
      if (failure !== undefined) throw failure;
      context = new GenerationContext(
        host,
        slots,
        slidingSlotWindows(graph),
        lengths,
        chunkLength,
        chunkBuckets,
        bindings,
      );
      return context;
    } finally {
      // MUST: a cleanup failure must not overwrite the body's exception (the same rule as Session
      // construction). The two pushed scopes are always popped (only when leaving before the pop
      // was issued — a double pop would take someone else's scope).
      if (!popped) await discardFailureScopes(gpu.device);
      // MUST: **unless** the context is returned, return every allocated buffer (leak none on a
      // synchronous throw, a rejected pop or device loss. A backing leaked in this window is
      // unreachable from dispose too). destroy runs only in this finally, so it is never called
      // twice.
      if (context === undefined) {
        for (const buffer of created) buffer.destroy();
      }
    }
  }

  /**
   * Creates a **borrowing context** (ADR 0096, addendum of 2026-09-08 on stage 2 — the drafter
   * reads the target's KV).
   *
   * It allocates only one logical-length uniform and binds the lender's slot backings as they are.
   * Binding goes by **name** (the borrower's declared name = the lender's declared name, MUST), and
   * the shape must be such that "the borrower's declared shape solved with **the lender's
   * bindings** (same-named symbols, MUST — the capacity `C`)" equals the actual shape of the
   * lender's slot.
   *
   * MUST: do not accept `bindings` (they are inherited from the lender — with two binding points,
   * "reading the lender's 131072-row slot as 512 rows" would stand with the capacity symbol split
   * and no exception).
   * MUST: `chunkLength` is exactly 1 (the borrower's only execution shape is decode = readonly
   * attention has M fixed at 1 — paired with the readonly attention shape check of the same
   * addendum). The ban on `chunkBuckets` is not written here — with `chunkLength = 1` the existing
   * {@link assertChunkBuckets} (elements in `2..chunkLength−1`) already fails anything non-empty, so
   * writing it again would be an unreachable branch.
   */
  static async #createBorrowed(
    host: GenerationContextHost,
    spec: GenerationContextSpec,
    lender: GenerationContext,
    names: readonly string[],
  ): Promise<GenerationContext> {
    const { gpu, graph } = host;
    const internals = lender[RUNTIME_INTERNAL];
    internals.assertLendable("createGenerationContext(borrow)");
    if (internals.gpu !== gpu) {
      throw new ExecutionError(
        "createGenerationContext(borrow): 貸し手 context と GpuContext（device）が別" +
          "（別 device のスロットは束縛できない）",
      );
    }
    if (spec.chunkLength !== 1) {
      throw new ExecutionError(
        `createGenerationContext(borrow): chunkLength ${spec.chunkLength} は 1 ちょうど` +
          "（借り手の実行形は decode 1 本だけ — ADR 0096 段 2 §2.1）",
      );
    }
    if (spec.bindings !== undefined) {
      throw new ExecutionError(
        "createGenerationContext(borrow): bindings は貸し手のものを継承する（渡せない）" +
          "— 2 つの束縛点を持つと容量記号が割れたまま別容量のスロットを読む",
      );
    }
    const bindings = internals.bindings;
    const windows = slidingSlotWindows(graph);
    const slots = new Map<string, StateSlotBacking>();
    for (const name of names) {
      const backing = internals.slots.get(name);
      if (backing === undefined) {
        throw new ExecutionError(
          `state '${name}': 貸し手 context に同名のスロットが無い` +
            "（external スロットは貸し手と同じ名前で宣言する MUST — ADR 0096 段 2 §2.1）",
        );
      }
      // MUST: the window agrees with the lender's **in presence and value** (the same read/write
      // formula — ADR 0067 decision 4 extended to lending). A wider borrower window alone reads rows
      // the lender has already overwritten as the past.
      const window = windows.get(name);
      const lent = internals.slidingSlots.get(name);
      if (window !== lent) {
        const show = (value: number | undefined): string => value?.toString() ?? "宣言なし";
        throw new ExecutionError(
          `state '${name}': attrs.window が貸し手と食い違う（貸し手 ${show(lent)} / 借り手 ${show(
            window,
          )}）— 論理 col → 物理 row の写像は読み書き同式 MUST（ADR 0067 決定 4）`,
        );
      }
      const shape = resolveSlotShape(name, graph.states[name].shape, bindings);
      if (shape.length !== backing.shape.length || shape.some((d, i) => d !== backing.shape[i])) {
        throw new ExecutionError(
          `state '${name}': 借り手の宣言 [${shape.join(",")}]（貸し手の bindings で解決）が` +
            `貸し手スロットの実形 [${backing.shape.join(",")}] と違う`,
        );
      }
      slots.set(name, backing);
    }

    // Only the lengths buffer is allocated (the slots are borrowed). The errorScope rules are the
    // same as on the own-allocation path — createBuffer throws no synchronous exception even over a
    // limit, so without the scope it becomes a silent no-op that keeps writing logical lengths into
    // an invalid buffer.
    pushFailureScopes(gpu.device);
    const where = "GenerationContext（借り手）の論理長確保";
    let popped = false;
    let created: GPUBuffer | undefined;
    let context: GenerationContext | undefined;
    try {
      created = gpu.device.createBuffer({
        label: "generation lengths (borrowed)",
        size: LENGTHS_BYTES,
        usage: LENGTHS_USAGE,
      });
      const pending = popFailureScopes(gpu.device, where);
      popped = true;
      const failure = await gpu[RUNTIME_INTERNAL].raceDeviceLost(pending, where);
      if (failure !== undefined) throw failure;
      const borrowed = new GenerationContext(
        host,
        slots,
        slidingSlotWindows(graph),
        created,
        // MUST: pass the value that passed the check above (exactly 1) as a constant — even if
        // `spec` is modified between the check and the await, the borrower's execution shape stays
        // 1 (the same TOCTOU closure as the own-allocation path).
        1,
        [],
        bindings,
        { lender, slidingSlack: lender.slidingSlack },
      );
      // MUST: register **after** construction (block the lender's dispose once the borrower
      // exists). If this fails (the lender was disposed in the await window), return the lengths
      // buffer instead of leaking it.
      internals.registerBorrower(borrowed);
      context = borrowed;
      return context;
    } finally {
      if (!popped) await discardFailureScopes(gpu.device);
      if (context === undefined) created?.destroy();
    }
  }

  /**
   * The body of {@link GenerationContext.fork} (allocate → submit the copy → issue the pop).
   *
   * MUST: go from push to **issuing** the pop without await (the same form as `create` and
   * `ResidentTensor.read` — the third rule of "Invariants of errorScope sections" at the top of
   * GpuContext). Finishing the copy's submit in this synchronous section is exactly why no
   * reservation on the source is needed (the doc of {@link GenerationContext.fork}). The copies go
   * in one encoder and one submit and are not put on the `SubmitScheduler`'s pending list (not mixed
   * into a list that another run's failure discards).
   * MUST: copy **every byte** of every slot (including the slack rows of the sliding ring — unless
   * rows written ahead of the logical length are copied with their physical layout, the copy's ring
   * would have a different layout from the source). Baked bind groups are not copied (they bind the
   * source slots — copying them would make the copy's runs write into the source KV).
   */
  static async #forkFrom(
    host: GenerationContextHost,
    source: GenerationContext,
    pastLength: number,
  ): Promise<GenerationContext> {
    const { gpu } = host;
    pushFailureScopes(gpu.device);
    const where = "GenerationContext の fork（state の複製）";
    const created: GPUBuffer[] = [];
    const slots = new Map<string, StateSlotBacking>();
    let popped = false;
    let context: GenerationContext | undefined;
    try {
      const encoder = gpu.device.createCommandEncoder({ label: "generation context fork" });
      for (const [name, slot] of source.#slots) {
        const buffer = gpu.device.createBuffer({
          label: `state '${name}'`,
          size: slot.byteLength,
          usage: STORAGE_USAGE,
        });
        created.push(buffer);
        slots.set(name, { buffer, shape: slot.shape, byteLength: slot.byteLength });
        encoder.copyBufferToBuffer(slot.buffer, 0, buffer, 0, slot.byteLength);
      }
      const lengths = gpu.device.createBuffer({
        label: "generation lengths",
        size: LENGTHS_BYTES,
        usage: LENGTHS_USAGE,
      });
      created.push(lengths);
      gpu.device.queue.submit([encoder.finish()]);
      const pending = popFailureScopes(gpu.device, where);
      popped = true;
      const failure = await gpu[RUNTIME_INTERNAL].raceDeviceLost(pending, where);
      if (failure !== undefined) throw failure;
      const forked = new GenerationContext(
        host,
        slots,
        source.#slidingSlots,
        lengths,
        source.chunkLength,
        source.chunkBuckets,
        source[RUNTIME_INTERNAL].bindings,
      );
      // The logical length is the value captured in fork()'s synchronous section (the same moment
      // as the copied bytes).
      forked.#pastLength = pastLength;
      context = forked;
      return context;
    } finally {
      // MUST: the cleanup rules are the same as `create` (discard only when leaving before the pop
      // was issued; when not returning, return every allocated buffer). destroy is safe even while
      // a submitted copy references it (the implementation delays the real release until completion).
      if (!popped) await discardFailureScopes(gpu.device);
      if (context === undefined) {
        for (const buffer of created) buffer.destroy();
      }
    }
  }

  /**
   * The logical length of the committed KV (ADR 0066 decision 6).
   *
   * It advances **only on a successful run** (manual host-side addition is not an API — no double
   * bookkeeping). It cannot be read after poisoning or device loss (addendum 3's "every later
   * operation fails loudly" and decision 7 — in both cases the physical state behind it is
   * unrecoverable, so this number must not be read as "you can resume from here". What it tells is
   * "how far it got", not "whether it can continue from there", and returned in a form that cannot
   * tell the two apart, the host would always use it as the latter).
   *
   * On a borrowing context (created with `createGenerationContext({ borrow })`), this value is a
   * snapshot of **the logical length copied from the lender when a borrower run started**, and it is
   * not updated when the lender advances (`0` if no run has been executed). Where the current value
   * is needed, read the lender's `pastLength` — the speculative decoding generation surface does
   * that too (ADR 0096, addendum of 2026-09-08 on stage 2).
   */
  get pastLength(): number {
    this.#assertUsable("pastLength");
    return this.#pastLength;
  }

  /**
   * The deferred run waiting for commit (`GenerationRun.commit: "deferred"` — undefined if none).
   *
   * `pastLength` is the logical length the run captured and `queryLength` the number of rows written
   * to the physical ring; the upper bound of the `rows` {@link GenerationContext.commit} accepts is
   * exactly `queryLength`.
   *
   * The returned object is **frozen** (`Object.freeze`) — the bound check trusts these values, so a
   * write throws `TypeError` in strict mode (modules are always strict).
   *
   * On a borrowing context it is **always undefined** (a borrower is refused both deferred runs and
   * commit, so it never holds a pending commit). A lender's pending commit is not visible from the
   * borrower, so do not use the borrower's value to decide "may I draft" — read the lender's
   * `pendingCommit` (a borrower run against a lender with a pending commit is refused by the
   * lender-side lease acquisition).
   */
  get pendingCommit(): { readonly pastLength: number; readonly queryLength: number } | undefined {
    this.#assertUsable("pendingCommit");
    return this.#pending;
  }

  /**
   * Advances the logical length of the pending deferred run **by the accepted rows only** (ADR 0066
   * decision 6's "the logical length advances on a successful run" widened to the verify shape of
   * speculative decoding).
   *
   * `rows` is an integer with `0 ≤ rows ≤ pendingCommit.queryLength`; `0` means "accept no row"
   * (the logical length does not move; only the pending commit is folded). The physical ring keeps
   * the `queryLength` written rows, but the logical columns overwritten by rows that were not
   * accepted fall outside the sliding slack ({@link GenerationContext.slidingSlack}) — a deferred
   * run whose `queryLength` exceeds the slack has already failed in the synchronous section of
   * `Session.run` (`queryLength ≤ slidingSlack`).
   *
   * MUST: **fail loudly while a generation run is in flight** (the same reason as `rewind` — a run
   * fixes its uniform and dispatch counts from the P captured at its head, so moving it from the
   * side splits the logical length the GPU saw from the basis of progress).
   * MUST: a call without a pending commit fails loudly (silently turning a commit called after an
   * immediate run into a no-op would leave the host believing "I reported the accepted rows" while
   * it keeps running on a logical length advanced twice).
   */
  commit(rows: number): void {
    this.#assertUsable("commit");
    // A borrower has no logical length (the lender advances and commits — ADR 0096, addendum of
    // 2026-09-08 on stage 2).
    this.#assertNotBorrowing("commit");
    if (this.#runKind !== undefined) {
      throw new ExecutionError(
        `commit: 進行中の generation run が 1 本ある間は確定できない` +
          "（run の決着を await してから呼ぶこと）",
      );
    }
    const pending = this.#pending;
    if (pending === undefined) {
      throw new ExecutionError(
        "commit: 確定待ちの generation run が無い" +
          "（commit を要するのは GenerationRun.commit を 'deferred' で発行した run だけ）",
      );
    }
    if (!Number.isSafeInteger(rows) || rows < 0) {
      throw new ExecutionError(`commit: 受理行数 ${rows} が非負整数でない`);
    }
    if (rows > pending.queryLength) {
      throw new ExecutionError(
        `commit: 受理行数 ${rows} が保留中の run の queryLength ${pending.queryLength} を超える` +
          "（書いていない行は確定できない）",
      );
    }
    // The logical length has not moved since the run that set the pending commit (with leases and
    // rewind blocked, a split is a broken implementation invariant — do not let it continue
    // silently).
    this.#assertCapturedPast(pending.pastLength, "commit");
    const next = pending.pastLength + rows;
    // MUST: check the sum's u32 bound here too, for the same reason as `#advance` (it overflows
    // even with both terms ≤ u32).
    if (next > MAX_LOGICAL_LENGTH) {
      throw new ExecutionError(
        `commit: pastLength ${pending.pastLength} + 受理行数 ${rows} = ${next} が ` +
          `u32 の上限 ${MAX_LOGICAL_LENGTH} を超える（論理長の搬送先は u32 — ADR 0066 追記 4）`,
      );
    }
    // MUST: updating the logical length and clearing the pending commit are indivisible (clearing
    // first would, after a commit that failed the checks above, leave a state with "no pending
    // commit and no advance", and the written rows would be unreachable from anywhere).
    this.#pastLength = next;
    this.#pending = undefined;
  }

  /**
   * Truncates the logical position (ADR 0066 decision 6). Integers with
   * `0 ≤ position ≤ pastLength` only.
   *
   * MUST: **fail loudly while a generation run is in flight**. A run writes its uniform, computes
   * its dispatch counts and `advance`s on success from the `P` captured at its head, so moving the
   * logical length from the side in the middle splits "the P the GPU read" from "the P progress was
   * based on" (no exception — only the KV's logical position quietly shifts). The lease is set in
   * the **synchronous section** of `Session.run`, so a call right after a `run()` that was not
   * awaited is caught too.
   * MUST: **a context that contains even one sliding slot refuses every rewind** (ADR 0066 addendum
   * 2). After an eviction, the ring's physical layout and logical range disagree even for a rewind
   * to a resident position (there is no left compaction). It is the same contract as ORT GenAI,
   * which refuses every rewind below current for the same reason; relaxing it goes together with an
   * implementation of compaction.
   */
  rewind(position: number): void {
    this.#assertUsable("rewind");
    this.#assertNotBorrowing("rewind");
    if (this.#runKind !== undefined) {
      throw new ExecutionError(
        `rewind: 進行中の generation run が 1 本ある間は巻き戻せない` +
          "（run は頭で捕捉した pastLength で uniform と dispatch 数を決めるので、横から動かすと" +
          "GPU が見た論理長と進行の基準が分裂する）。run の決着を await してから呼ぶこと",
      );
    }
    if (this.#pending !== undefined) {
      throw new ExecutionError(
        `rewind: commit 待ちの generation run がある間は巻き戻せない` +
          `（pastLength ${this.#pending.pastLength} + queryLength ${this.#pending.queryLength} ` +
          "まで物理 ring へ書き込み済み）。context.commit(rows) で確定させてから呼ぶこと",
      );
    }
    if (this.#slidingSlots.size > 0) {
      throw new ExecutionError(
        `rewind: sliding スロット [${[...this.#slidingSlots.keys()].join(
          ", ",
        )}] を含む context は` +
          "巻き戻せない（ring はエビクト後に物理配置と論理範囲が一致しないため — ADR 0066 " +
          "追記 2）。有効なのは全スロットが非 sliding の context だけで、復旧は新しい context",
      );
    }
    if (!Number.isSafeInteger(position) || position < 0) {
      throw new ExecutionError(`rewind: 位置 ${position} が非負整数でない`);
    }
    if (position > this.#pastLength) {
      throw new ExecutionError(
        `rewind: 位置 ${position} が現在の pastLength ${this.#pastLength} を超えている` +
          "（前進は run の成功でのみ起きる）",
      );
    }
    this.#pastLength = position;
  }

  /**
   * **Duplicates a whole context** of the same Session and capacity (ADR 0066 addendum of
   * 2026-10-11 — forking the conversation). It returns an independent context with every byte of
   * every slot and `pastLength` copied; the caller owns it and disposes it separately (there is no
   * parent-child dependency on the source, and the source may be disposed first).
   *
   * The following are decided in the synchronous section of the call and reject the returned
   * Promise: after a dispose request, after device loss (`GpuDeviceLostError`), after poisoning, an
   * unsettled run (including while a borrower's run holds the lender's reservation), during a
   * batch's use reservation, while a deferred commit is pending, a borrowing context, and a Session
   * already disposed. A lender with live borrowers and a context with sliding slots are not refused
   * (borrowers write no KV row — the source's borrowers stay bound to the source slots, and the copy
   * has no borrowers).
   *
   * MUST: finish the checks, the capture of `pastLength`, the allocation and the copy's submit **in
   * this synchronous section, without await** (the MUSTs of `host.adopt` and `#forkFrom`). No
   * reservation is placed on the source, so correctness rests on just two facts: "the source is
   * quiescent (no in-flight run and no pending commit = no unsubmitted writes)" and "the copy is
   * submitted here (queue order guarantees after the earlier writes and before the later ones)".
   * That is why a run / `rewind` / `dispose` / second `fork` issued to the source without awaiting
   * `fork()` all go through, and the copy holds the state at the time of the call.
   * NOTE: the copy is issued with its own encoder and submit (on neither the Session's
   * serialisation chain nor the `SubmitScheduler` — like `createGenerationContext`, it adds no edge
   * to the waits on batch sections). Its GPU time does not appear in the scheduler's statistics. An
   * allocation failure fails loudly through the out-of-memory errorScope, and the source stays
   * intact.
   */
  fork(): Promise<GenerationContext> {
    let pastLength: number;
    try {
      this.#assertUsable("fork");
      if (this.#borrow !== undefined) {
        throw new ExecutionError(
          "fork: 借り手 context は複製できない（スロットは貸し手の所有物 — 分岐するなら貸し手を " +
            "fork し、借り手は複製に対して開き直す。ADR 0066 追記〈2026-10-11〉）",
        );
      }
      if (this.#runKind === "batch") {
        throw new ExecutionError(
          "fork: GenerationContext は batch の最終決着待ち（finish 後に複製すること）",
        );
      }
      if (this.#runKind !== undefined) {
        throw new ExecutionError(
          "fork: 進行中の generation run がある間は複製できない（借り手の run も貸し手の予約を" +
            "取る）。run の決着を await してから呼ぶこと",
        );
      }
      if (this.#pending !== undefined) {
        throw new ExecutionError(
          `fork: commit 待ちの generation run がある（pastLength ${this.#pending.pastLength} + ` +
            `queryLength ${this.#pending.queryLength} まで書き込み済み）。` +
            "context.commit(rows) で確定させてから呼ぶこと",
        );
      }
      pastLength = this.#pastLength;
    } catch (cause) {
      return Promise.reject(cause);
    }
    const host = this.#host;
    return host.adopt(
      () => GenerationContext.#forkFrom(host, this, pastLength),
      "fork: dispose 済みの Session では GenerationContext を複製できない" +
        "（Session の dispose が fork より先か、複製の確保の途中に発行された）",
    );
  }

  /**
   * Returns the state slots and the logical-length uniform (ADR 0066 decision 6 —
   * flush-before-destroy). During a batch's use reservation it refuses without closing admission;
   * it can be retried after finish. The two-step disposal below is the contract for ordinary runs.
   *
   * MUST: use **two steps**. Only step 1 is set synchronously (closing admission — later run
   * admission and user-facing reads and writes are refused); the internal surface (carrying and
   * advancing the logical length) is cut off in step 2, when the disposal body queued on the
   * Session chain starts. The dispose body runs **after** earlier runs, so an already accepted run
   * is not killed here and completes (`run(); context.dispose();` without await has the same
   * semantics as the Session's `run(); session.dispose();`). Closing in one step would make the
   * accepted run's `writeLengths` / `advance` fail as "disposed".
   * MUST: return the same completion on the second and later calls (the same reason as
   * `Session.dispose` — returning early would let the caller treat it as "disposed" and proceed to
   * `device.destroy()`, breaking flush-before-destroy).
   * MUST: even if flush fails (mainly device loss), always destroy the buffers and return the
   * bookkeeping, and propagate the failure itself after the cleanup instead of swallowing it (the
   * same rule as `RunArena.#destroyOnce`).
   * NOTE: it may outlive the Session. This path touches only the injected flush and its own buffers,
   * never the Session's weights or plan cache (it creates no ordering dependency).
   */
  dispose(): Promise<void> {
    // During a batch's use reservation, refuse without closing admission. A retry after finish
    // proceeds to the usual disposal.
    try {
      this.#assertCanDispose();
    } catch (cause) {
      return Promise.reject(cause);
    }
    // MUST: do not dispose while borrowers are alive (ADR 0096, addendum of 2026-09-08 on stage 2).
    // The borrowers' bind groups hold these slot buffers, and destroying them makes the next draft
    // read destroyed buffers. This is a **deliberate departure** from the "idempotent, non-throwing"
    // contract of `dispose`, and it returns before setting the admission-closed flag (failing after
    // setting it would leave a context through which no operation passes).
    if (this.#borrowers.size > 0 && this.#disposal === undefined) {
      return Promise.reject(
        new ExecutionError(
          `dispose: この GenerationContext を借りている context が ${this.#borrowers.size} 本ある` +
            "（借り手の bind group が貸し手のスロットを掴んでいる）。借り手を先に dispose すること",
        ),
      );
    }
    this.#disposeRequested = true;
    this.#disposal ??= this.#host.serialize(async () => {
      this.#disposed = true;
      try {
        await this.#host.flush();
      } finally {
        // MUST: do not destroy borrowed slots (the lending context owns them). If the borrower
        // destroyed them, the lender's next run would write into destroyed buffers.
        if (this.#borrow === undefined) {
          for (const slot of this.#slots.values()) slot.buffer.destroy();
        } else {
          this.#borrow.lender[RUNTIME_INTERNAL].forgetBorrower(this);
        }
        this.#lengths.destroy();
        // MUST: let go of the baked bundles here too. No run comes afterwards (`#assertUsable`
        // fails it), so it does not affect correctness, but holding them keeps bind groups that
        // reference destroyed buffers alive for as long as the context references them.
        this.#baked.clear();
        // Drop an uncommitted pending commit here (the physical buffers are torn down, so there is
        // nothing left to commit).
        this.#pending = undefined;
        this.#host.forget(this);
      }
    });
    return this.#disposal;
  }

  /**
   * Refuses during a batch: queuing a pending disposal on the Session chain would wait on itself
   * through later enqueues.
   */
  #assertCanDispose(): void {
    if (this.#runKind === "batch") {
      throw new ExecutionError(
        "dispose: GenerationContext は batch の最終決着待ち（finish 後に破棄すること）",
      );
    }
  }

  /**
   * Writes the logical-length uniform (the carrier of ADR 0066 addendum 4).
   *
   * MUST: **do not put it** in the content-addressed params cache (ADR 0042). Putting a value that
   * changes every step into a content address hits both "unbounded cache growth" and "cannot be
   * updated, because derivation does not run on a PreparedPlan hit". The backing is this one
   * context-owned buffer, and recipes reference it through a fixed binding (`lengths` of
   * `BindingSource`).
   * MUST: call it **before every run's encode** (`queue.writeBuffer` enters the queue timeline in
   * issue order, so it does not overtake submitted dispatches — ADR 0004 invariant ④).
   */
  #writeLengths(pastLength: number, queryLength: number): void {
    this.#assertInternalUsable("writeLengths");
    this.#assertCapturedPast(pastLength, "writeLengths");
    this.#assertQueryLength(queryLength, "writeLengths");
    this.#lengthValues[0] = this.#pastLength;
    this.#lengthValues[1] = queryLength;
    this.#host.gpu.device.queue.writeBuffer(this.#lengths, 0, this.#lengthValues);
  }

  /**
   * Advances the logical length (ADR 0066 decision 6 — call it **only on a successful run**).
   *
   * WHY decision 6's "the logical length advances on a successful run" now has two forms:
   * `commit: "immediate"` (default, the original form) advances here on a successful run, and
   * `commit: "deferred"` hands the right to {@link GenerationContext.commit} on success
   * ({@link #defer}). In both, **only one path at a time** moves the logical length, and while a
   * commit is pending both the next run and rewind are refused, so no double bookkeeping (a second
   * logical length on the host side) arises.
   *
   * NOTE: the run-time check of full slots, `pastLength + queryLength ≤ capacity` (item ④ of ADR
   * 0067 decision 4), sits **before the run's encode** (`assertGenerationRun` —
   * src/runtime/recipe.ts). The op contract decides which dimension is the capacity axis, so the
   * {@link GenerationLimits} collected by the derivation phase are the source of truth, and this
   * does not check again (checking at the time of advancing would be too late, after the write).
   */
  #advance(pastLength: number, queryLength: number): void {
    this.#assertInternalUsable("advance");
    this.#assertNotBorrowing("advance");
    this.#assertCapturedPast(pastLength, "advance");
    this.#assertQueryLength(queryLength, "advance");
    const next = this.#pastLength + queryLength;
    // MUST: the logical length after the addition also fits u32. The sum overflows even with both
    // terms ≤ u32, so this is the only check point (assigning an overrun would make the next
    // writeLengths silently put a truncated past on the GPU — {@link MAX_LOGICAL_LENGTH}).
    if (next > MAX_LOGICAL_LENGTH) {
      throw new ExecutionError(
        `advance: pastLength ${this.#pastLength} + queryLength ${queryLength} = ${next} が ` +
          `u32 の上限 ${MAX_LOGICAL_LENGTH} を超える（論理長の搬送先は u32 — ADR 0066 追記 4）`,
      );
    }
    this.#pastLength = next;
  }

  /**
   * Holds the logical length pending without advancing it (only on a successful `commit:
   * "deferred"` run — the pair of {@link #advance}).
   *
   * MUST: at most one pending commit (while a commit is pending, `acquireRun` refuses the next run,
   * so no path reaches a second `defer`). Reaching it means a broken runtime-internal invariant, so
   * fail immediately instead of overwriting — overwriting would make the rows written by the first
   * unreachable from anywhere.
   */
  #defer(pastLength: number, queryLength: number): void {
    this.#assertInternalUsable("defer");
    this.#assertNotBorrowing("defer");
    this.#assertCapturedPast(pastLength, "defer");
    this.#assertQueryLength(queryLength, "defer");
    if (this.#pending !== undefined) {
      throw new ExecutionError(
        `defer: commit 待ちの generation run が既にある（pastLength ${this.#pending.pastLength} / ` +
          `queryLength ${this.#pending.queryLength}）— 内部の不変条件破れ`,
      );
    }
    // MUST: set it frozen (the getter returns this object as is, and `commit` trusts its
    // `queryLength` as the bound on accepted rows, so an outside write would let "committing rows
    // that were never written" slip past the bound check).
    this.#pending = Object.freeze({ pastLength, queryLength });
  }

  /**
   * Poisons the context (ADR 0066 addendum 3 — failure of a run containing a state-changing
   * dispatch).
   *
   * The logical length does not advance, but the physical ring may have been overwritten, so in a
   * design without rollback / staging "refuse every later operation" is the only way to claim
   * consistency (recovery = a new context + a host-side rebuild). The trigger is the run's failure
   * path (`#poisonOnStateWrite` in the executor).
   *
   * MUST: keep the first reason on later calls (do not overwrite the real cause with later failures).
   */
  #poison(reason: string): void {
    this.#poisoned ??= reason;
  }

  /**
   * Usability of the **user-facing surface** (every operation including reads — the only exception
   * is the cleanup {@link GenerationContext.dispose}). Fails a dispose request, device loss and
   * poisoning here.
   *
   * MUST: do not split the condition between reads and writes. Poisoning and device loss are both
   * states where "the physical state behind it is lost", and letting the logical length alone be
   * returned would make an unrecoverable context look like a container holding a normal value
   * (addendum 3's "every later operation fails loudly").
   * MUST: check **step 1** of dispose (admission closed). Run admission is on this surface too, so
   * a run issued after `dispose()` is refused here.
   */
  #assertUsable(where: string): void {
    if (this.#disposeRequested) {
      throw new ExecutionError(`${where}: dispose 済みの GenerationContext は使えない`);
    }
    this.#assertLive(where);
  }

  /**
   * Usability of the **internal surface** (carrying, advancing and reading the logical length). It
   * checks **step 2** of dispose (whether the disposal body has started), letting accepted runs
   * between step 1 and step 2 through.
   */
  #assertInternalUsable(where: string): void {
    if (this.#disposed) {
      throw new ExecutionError(`${where}: dispose 済みの GenerationContext は使えない`);
    }
    this.#assertLive(where);
  }

  /**
   * The surface a borrowing context cannot touch (the four operations that move the logical length)
   * — `commit` / `rewind` are user-facing refusals, `advance` / `defer` are broken runtime-internal
   * invariants (by contract the execution integration checks
   * {@link GenerationContextInternals.borrowing} and does not call them — ADR 0096, addendum of
   * 2026-09-08 on stage 2).
   */
  #assertNotBorrowing(where: string): void {
    if (this.#borrow === undefined) return;
    throw new ExecutionError(
      `${where}: 借り手 context は論理長を持たない（進行も巻き戻しも貸し手 context の側で` +
        "起きる — ADR 0096 段 2 §2.1）",
    );
  }

  /** "Is the physical state behind it alive", shared by the two cut-off surfaces. */
  #assertLive(where: string): void {
    assertDeviceUsable(this.#host.gpu, where);
    if (this.#poisoned !== undefined) {
      throw new ExecutionError(
        `${where}: 汚染された GenerationContext は使えない（${this.#poisoned}）。` +
          "復旧は新しい context + ホスト側の state 再構築",
      );
    }
  }

  /**
   * The `pastLength` a run captured at its head equals the current logical length.
   *
   * MUST: the states form assumes that the value written to the uniform, the dispatch-count
   * computation, the capacity check and the advance all come from **one and the same P** (ADR 0066
   * decision 3 / addendum 4). With a lease, every path that moves it from the side is blocked, so a
   * split here is a broken implementation invariant — fail immediately instead of continuing
   * silently.
   */
  #assertCapturedPast(pastLength: number, where: string): void {
    if (pastLength !== this.#pastLength) {
      throw new ExecutionError(
        `${where}: run が捕捉した pastLength ${pastLength} が context の現在値 ` +
          `${this.#pastLength} と食い違う（進行中 run の論理長が横から動いた — 内部の不変条件破れ）`,
      );
    }
  }

  /**
   * Check of this step's real token count (common to the two execution shapes of ADR 0066 decision
   * 4).
   *
   * prefill-chunk has `queryLength ≤ chunkLength` (rows beyond it do not fit the physical shape),
   * decode has `queryLength = 1`. 0 is not allowed because "a run that advances nothing" empties the
   * state's write range, and the correspondence between progress and physical contents can no
   * longer be observed.
   *
   * NOTE: the u32 bound ({@link MAX_LOGICAL_LENGTH}) is not checked again here — `chunkLength` is
   * narrowed to ≤ u32 in create, so `queryLength ≤ chunkLength` is the bound as is (`rewind`'s
   * `position ≤ pastLength` passes the bound on in the same way).
   */
  #assertQueryLength(queryLength: number, where: string): void {
    if (!Number.isSafeInteger(queryLength) || queryLength < 1) {
      throw new ExecutionError(`${where}: queryLength ${queryLength} が 1 以上の整数でない`);
    }
    if (queryLength > this.chunkLength) {
      throw new ExecutionError(
        `${where}: queryLength ${queryLength} が chunkLength ${this.chunkLength} を超えている` +
          "（prefill は固定長 chunk + pad・decode は 1 — ADR 0066 決定 4）",
      );
    }
  }
}
