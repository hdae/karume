/**
 * **Static liveness packing** of intermediate buffers (node outputs and node temporaries) (ADR 0093).
 *
 * Replays the allocation program ({@link TransientProgram} — built from a recipe list or from the
 * pre-fusion node list) in the same event order as the execution phase, takes the live interval and
 * the real size of each logical tensor, and places them at offsets in region buffers so that tensors
 * whose time intervals do not overlap can share the same bytes. It returns the number and size of the
 * regions, `(region, offset, byteLength)` per slot, and the slot assignment per step.
 *
 * MUST: a pure function (creates no GPU resources and does not modify its inputs). Execution
 * (recipe.ts), baking (backing) and the estimate (estimate.ts) share **this one function** — rules
 * split across two places let allocation and estimate (or allocation and binding) claim different
 * numbers with no exception.
 * MUST: the replay order is — allocate the step outputs (ascending slot) → per dispatch, allocate
 * temporaries (`allocBefore`) → dispatch → release temporaries (`releaseAfter`, reverse allocation
 * order) → release inputs (the release list) → release the definition's own count (ascending output
 * slot). A different order silently produces "an input that is still read gets reassigned as an
 * output", with no exception.
 * MUST: **two tensors where one is read and the other written in the same dispatch go into
 * different regions**. WebGPU's usage-scope validation works per GPUBuffer (it ignores the binding
 * range), and binding the same buffer as both `read` (read-only-storage) and `read_write` (storage)
 * in one dispatch fails validation. Sharing among reads, or among writes, is legal.
 * MUST: the real size is {@link toSizeClass} (4-byte alignment, 4-byte floor) — the binding `size`
 * and the offset use this unit. Offsets are aligned to `offsetAlignment` (the device's
 * `minStorageBufferOffsetAlignment`, spec default 256).
 * MUST: over-limit cases (slot real size > `maxStorageBufferBindingSize`, a slot that does not fit
 * in a region ≤ `maxBufferSize`) are **listed in full before placement** and fail (ADR 0093
 * decision 5 — a deterministic check ahead of the errorScope).
 *
 * Cost: for n slots, one placement is O(n · slots placed so far) — each slot scans every region for
 * the least growth — and it runs in two orders; 100–150 ms for the largest real assets (n ≈ 2,500).
 * Determinism: both orders are total (real size descending, then allocation event or lifetime, then
 * allocation event) and a tie between the two totals keeps the first, so the same input always gives
 * the same plan.
 */

import { toSizeClass } from "../gpu/arena.ts";
import { ExecutionError } from "./plan.ts";

/** A reference to a binding target in the replay (value name = step output, graph input or weight / temp = index within the same step). */
export type TransientRef =
  | { readonly kind: "value"; readonly name: string }
  | { readonly kind: "temp"; readonly id: number };

/** The allocation spec of one step output (a planning-oriented copy of `StepOutput`). */
export type TransientOutputSpec = {
  readonly name: string;
  /** `alias` makes the input's backing the output as is (reshape / identity expand) — no allocation. */
  readonly kind: "alloc" | "alias";
  /** The real size of `alloc` (before alignment). Not read for `alias`. */
  readonly byteLength: number;
  /** The source value name of `alias` (values without a slot, such as graph inputs / weights, become undefined in the replay). */
  readonly source?: string;
  /** Future consumption count of the output value (the exact count over `node.ins`). */
  readonly uses: number;
  /** Graph output (alive to the end, shared with no other value). */
  readonly pinned: boolean;
};

/** The allocation spec of a node temporary (same shape as `TempRecipe`). */
export type TransientTempSpec = {
  readonly byteLength: number;
  readonly allocBefore: number;
  readonly releaseAfter: number;
};

/**
 * The read / write roles of one dispatch (used for the usage-scope constraint). The roles come from
 * the kernel's WGSL declarations (`var<storage, read>` / `read_write`) — no hand-written table.
 */
export type TransientDispatchSpec = {
  readonly reads: readonly TransientRef[];
  readonly writes: readonly TransientRef[];
};

/** The allocation program of one execution step. */
export type TransientStepSpec = {
  readonly outputs: readonly TransientOutputSpec[];
  readonly temps: readonly TransientTempSpec[];
  readonly dispatches: readonly TransientDispatchSpec[];
  /** The **release list** of inputs released at the end of the step (one entry per use). */
  readonly releases: readonly string[];
};

export type TransientProgram = readonly TransientStepSpec[];

/**
 * The device limits the plan follows (granted values). The estimate uses the granted values the
 * caller passed and fills unspecified fields and the offset alignment with WebGPU core defaults.
 */
export type TransientLimits = {
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly offsetAlignment: number;
};

/** A placed slot (region index, offset within the region, binding size = aligned real size). */
export type TransientSlot = {
  readonly region: number;
  readonly offset: number;
  readonly byteLength: number;
};

/** The slot assignment per step (undefined for `alias` outputs). */
export type TransientStepSlots = {
  readonly outputs: readonly (number | undefined)[];
  readonly temps: readonly number[];
};

export type TransientPlan = {
  /** Bytes per region (the size passed to `createBuffer`, 4-byte aligned). */
  readonly regions: readonly number[];
  readonly slots: readonly TransientSlot[];
  readonly steps: readonly TransientStepSlots[];
  /** Slots of graph outputs (the set allowed for readback). */
  readonly pinned: ReadonlySet<number>;
  /** Sum of the regions (= GPU bytes needed for intermediates). */
  readonly totalBytes: number;
  /** Maximum of simultaneously live bytes (a lower bound — zero fragmentation when the placement matches it). */
  readonly peakLiveBytes: number;
  /**
   * Number of slots that share a byte range with another slot placed earlier (= how many times the
   * same bytes were reassigned because the live intervals do not overlap). A diagnostic value: the
   * observation point where the fault-injection test of full-write (ADR 0014) confirms that
   * reassignment actually happened.
   */
  readonly sharedSlots: number;
};

/** The guaranteed WebGPU core defaults (used by the estimate without a device). */
export const CORE_TRANSIENT_LIMITS: TransientLimits = {
  maxBufferSize: 256 * 1024 * 1024,
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  offsetAlignment: 256,
};

type Interval = {
  readonly byteLength: number;
  readonly start: number;
  end: number;
  readonly label: string;
  pinned: boolean;
};

const overlaps = (a: Interval, b: Interval): boolean => a.start < b.end && b.start < a.end;

const pairKey = (a: number, b: number): number => (a < b ? a * 0x100000 + b : b * 0x100000 + a);

const assertLimits = (limits: TransientLimits): void => {
  for (const [name, value] of Object.entries(limits)) {
    if (!Number.isSafeInteger(value) || value < 4) {
      throw new ExecutionError(`invalid transient plan limit ${name}: ${value}`);
    }
  }
  if (limits.offsetAlignment % 4 !== 0) {
    throw new ExecutionError(
      `transient plan offsetAlignment must be a multiple of 4: ${limits.offsetAlignment}`,
    );
  }
};

type Placement = {
  readonly regions: readonly number[];
  readonly slots: readonly TransientSlot[];
  readonly sharedSlots: number;
  readonly totalBytes: number;
};

/**
 * Places the slots one by one in `order`: each goes to the conflict-free region it grows least, at the
 * lowest offset that fits there. First-fit took the first region that could hold the slot even when
 * that region had to grow and a later one had room, so removing one node could stack a long-lived
 * tensor on a large one and grow the total with the live peak unchanged.
 * A new region opens when every candidate would grow by the slot's full size: the bytes are the same,
 * and the new region carries no read / write conflicts, so later slots keep more places to go
 * (growing the existing region instead sums to about 20% more over the shipped models — ADR 0093
 * addendum 2026-10-26).
 */
const placeInOrder = (
  order: readonly number[],
  intervals: readonly Interval[],
  conflicts: ReadonlySet<number>,
  limits: TransientLimits,
): Placement => {
  const regions: { placed: number[]; size: number }[] = [];
  const slots: TransientSlot[] = new Array(intervals.length);
  let sharedSlots = 0;
  const align = (value: number): number =>
    Math.ceil(value / limits.offsetAlignment) * limits.offsetAlignment;
  for (const id of order) {
    const iv = intervals[id];
    let best: { region: number; offset: number; growth: number } | undefined;
    for (let region = 0; region < regions.length; region += 1) {
      const entry = regions[region];
      if (entry.placed.some((other) => conflicts.has(pairKey(id, other)))) continue;
      const busy = entry.placed
        .filter((other) => overlaps(iv, intervals[other]))
        .map((other) => slots[other])
        .sort((a, b) => a.offset - b.offset);
      let offset = 0;
      for (const b of busy) {
        if (offset + iv.byteLength <= b.offset) break;
        offset = Math.max(offset, align(b.offset + b.byteLength));
      }
      if (offset + iv.byteLength > limits.maxBufferSize) continue;
      const growth = Math.max(0, offset + iv.byteLength - entry.size);
      // Equal growth keeps the lower region index (determinism).
      if (best === undefined || growth < best.growth) best = { region, offset, growth };
    }
    if (best === undefined || best.growth >= iv.byteLength) {
      slots[id] = { region: regions.length, offset: 0, byteLength: iv.byteLength };
      regions.push({ placed: [id], size: iv.byteLength });
      continue;
    }
    const { region, offset } = best;
    const entry = regions[region];
    slots[id] = { region, offset, byteLength: iv.byteLength };
    // Shares a byte range with an earlier slot (legal because the lifetimes do not overlap) = one reassignment.
    const end = offset + iv.byteLength;
    if (
      entry.placed.some(
        (other) =>
          slots[other].offset < end && offset < slots[other].offset + slots[other].byteLength,
      )
    )
      sharedSlots += 1;
    entry.placed.push(id);
    entry.size = Math.max(entry.size, end);
  }
  const regionBytes = regions.map((entry) => toSizeClass(entry.size));
  return {
    regions: regionBytes,
    slots,
    sharedSlots,
    totalBytes: regionBytes.reduce((total, size) => total + size, 0),
  };
};

/**
 * Replays the allocation program and places it (the MUSTs of the module doc).
 *
 * Replay bookkeeping (reference counts): retain 1 for the definition + `uses`, and decrement on input
 * release and on release of the definition's own count. Going negative is a consumption-count error,
 * remaining at the end is a release leak — both fail loudly (pinned slots may remain).
 */
export const planTransients = (
  program: TransientProgram,
  limits: TransientLimits,
): TransientPlan => {
  assertLimits(limits);
  const intervals: Interval[] = [];
  const refs = new Map<number, number>();
  const env = new Map<string, number>();
  const conflicts = new Set<number>();
  const steps: TransientStepSlots[] = [];
  let clock = 0;

  const alloc = (byteLength: number, label: string): number => {
    const id = intervals.length;
    intervals.push({
      byteLength: toSizeClass(byteLength),
      start: clock,
      end: Infinity,
      label,
      pinned: false,
    });
    clock += 1;
    return id;
  };
  const retain = (slot: number | undefined, count: number, pinned: boolean): void => {
    if (slot === undefined) return;
    if (pinned) intervals[slot].pinned = true;
    refs.set(slot, (refs.get(slot) ?? 0) + count + 1);
  };
  const release = (slot: number | undefined): void => {
    if (slot === undefined) return;
    const left = (refs.get(slot) ?? 0) - 1;
    if (left < 0) {
      throw new ExecutionError(
        `transient plan: '${
          intervals[slot].label
        }' has a negative reference count (released too often — wrong consumer count)`,
      );
    }
    refs.set(slot, left);
    if (left > 0 || intervals[slot].pinned) return;
    intervals[slot].end = clock;
    clock += 1;
  };
  const resolve = (ref: TransientRef, temps: readonly number[]): number | undefined =>
    ref.kind === "temp" ? temps[ref.id] : env.get(ref.name);

  program.forEach((step, stepIndex) => {
    const outputs = step.outputs.map((output) => {
      const slot =
        output.kind === "alias"
          ? output.source === undefined
            ? undefined
            : env.get(output.source)
          : alloc(output.byteLength, output.name);
      retain(slot, output.uses, output.pinned);
      if (slot === undefined) env.delete(output.name);
      else env.set(output.name, slot);
      return slot;
    });
    const temps: number[] = [];
    step.dispatches.forEach((dispatch, index) => {
      step.temps.forEach((temp, id) => {
        if (temp.allocBefore !== index) return;
        const slot = alloc(temp.byteLength, `step ${stepIndex} temporary ${id}`);
        retain(slot, 0, false);
        temps[id] = slot;
      });
      // Usage scope: a slot read in this dispatch and a slot written in it cannot share a region.
      const reads = dispatch.reads
        .map((ref) => resolve(ref, temps))
        .filter((s): s is number => s !== undefined);
      const writes = dispatch.writes
        .map((ref) => resolve(ref, temps))
        .filter((s): s is number => s !== undefined);
      for (const r of reads) for (const w of writes) if (r !== w) conflicts.add(pairKey(r, w));
      for (let id = step.temps.length - 1; id >= 0; id -= 1) {
        if (step.temps[id].releaseAfter === index) release(temps[id]);
      }
    });
    for (const name of step.releases) release(env.get(name));
    for (const slot of outputs) release(slot);
    // Alias outputs have no slot (the binding resolves the ValueSource's origin directly).
    steps.push({
      outputs: step.outputs.map((output, index) =>
        output.kind === "alias" ? undefined : outputs[index],
      ),
      temps,
    });
  });

  const leaked = [...refs].filter(([slot, count]) => count > 0 && !intervals[slot].pinned);
  if (leaked.length > 0) {
    throw new ExecutionError(
      `transient plan: intermediates are left unreleased (wrong consumer count): ${leaked
        .map(([slot, count]) => `'${intervals[slot].label}' (refs=${count})`)
        .join(" / ")}`,
    );
  }

  // Limits: list every violation before placement and fail (ADR 0093 decision 5).
  const bindingCap = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
  const oversized = intervals.filter((iv) => iv.byteLength > bindingCap);
  if (oversized.length > 0) {
    throw new ExecutionError(
      `intermediate buffers exceed the device limits (maxStorageBufferBindingSize ${limits.maxStorageBufferBindingSize} / ` +
        `maxBufferSize ${limits.maxBufferSize}): ${oversized
          .map((iv) => `'${iv.label}' ${iv.byteLength}B`)
          .join(" / ")}`,
    );
  }

  // Placement runs in two orders of the slots and keeps the smaller region total (the first order
  // on a tie). Both orders take the slots in descending real size; equal sizes go in allocation order
  // in the first, the longer-lived first in the second. A single greedy order is sensitive to how
  // equal sizes are ordered — a DiT repeats the same sizes in every block — and two graphs with the
  // same live intervals in another order then land on different totals: taking the better of the two
  // keeps the total from moving with an unrelated node (ADR 0093 addendum 2026-10-26).
  const bySize = (a: number, b: number): number =>
    intervals[b].byteLength - intervals[a].byteLength;
  // A pinned slot lives to the end: `clock` is past every finite end.
  const lifetime = (id: number): number =>
    (Number.isFinite(intervals[id].end) ? intervals[id].end : clock) - intervals[id].start;
  const ids = intervals.map((_, id) => id);
  const inAllocationOrder = placeInOrder(
    [...ids].sort((a, b) => bySize(a, b) || intervals[a].start - intervals[b].start || a - b),
    intervals,
    conflicts,
    limits,
  );
  const longerLivedFirst = placeInOrder(
    [...ids].sort(
      (a, b) =>
        bySize(a, b) ||
        lifetime(b) - lifetime(a) ||
        intervals[a].start - intervals[b].start ||
        a - b,
    ),
    intervals,
    conflicts,
    limits,
  );
  const placement =
    longerLivedFirst.totalBytes < inAllocationOrder.totalBytes
      ? longerLivedFirst
      : inAllocationOrder;

  // Live peak (lower bound).
  let peakLiveBytes = 0;
  for (const iv of intervals) {
    let live = 0;
    for (const other of intervals) {
      if (other.start <= iv.start && iv.start < other.end) live += other.byteLength;
    }
    peakLiveBytes = Math.max(peakLiveBytes, live);
  }

  return {
    regions: placement.regions,
    slots: placement.slots,
    steps,
    pinned: new Set(intervals.flatMap((iv, id) => (iv.pinned ? [id] : []))),
    totalBytes: placement.totalBytes,
    peakLiveBytes,
    sharedSlots: placement.sharedSlots,
  };
};
