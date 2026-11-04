/**
 * Structural tests of the recipes' **allocation program** (`buildTransientProgram` →
 * `planRecipes` of src/runtime/recipe.ts) and of the **static validation of declarations**
 * (`validateStepRecipe` of the same file). They touch no GPU at all, so they run in an environment
 * without an adapter.
 *
 * What is checked is "whether allocation, retain, release and the read / write roles carry over
 * correctly from the {@link StepRecipe.outputs} sequence and the dispatch bindings into the plan
 * (src/runtime/transient-plan.ts — ADR 0093)" — from a different angle than the real-GPU footprint
 * check (tests/gpu_plan_backing_test.ts compares "the backed region total = the region allocation
 * of a miss run"), **pinning the expected values by hand calculation**.
 * The placement rule itself (least region growth in two orders, usage scope in separate regions,
 * limits) is covered by tests/runtime_transient_plan_test.ts. The hand derivations here follow it:
 * each slot (real size descending) goes to the conflict-free region it grows least (equal growth ->
 * lower region index), a new region opens when every candidate would grow by the slot's full size,
 * and of the two orders (equal sizes by allocation start / by longer lifetime) the smaller total wins
 * (order 1 on a tie).
 *
 * The first test here is the acceptance condition of ADR 0068 decision 1, "the recipe form of a
 * single-output node is equivalent to the form before outputs became a sequence".
 */

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { GpuDeviceLostError } from "../src/gpu/context.ts";
import type { StorageRoles } from "../src/gpu/pipeline-cache.ts";
import type { SubmitScheduler } from "../src/gpu/submit.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import {
  assertGenerationRun,
  bakeAllBindGroups,
  type BakedGeneration,
  type BindingRecipe,
  type BindingSource,
  buildTransientProgram,
  type DispatchWorkgroups,
  executeBakedPlan,
  executeBakedPlanPaced,
  type GenerationLimits,
  planRecipes,
  type StepOutput,
  type StepRecipe,
  type TempRecipe,
  validateStepRecipe,
} from "../src/runtime/recipe.ts";
import { CORE_TRANSIENT_LIMITS } from "../src/runtime/transient-plan.ts";

/**
 * A step with only an output sequence (dispatches and temporaries are a separate axis of the
 * allocation bookkeeping, so empty suffices).
 */
const step = (outputs: readonly StepOutput[], releases: readonly string[] = []): StepRecipe => ({
  outputs,
  temps: [],
  dispatches: [],
  releases,
  writesState: false,
});

/**
 * An output allocated from a region (`byteLength` is kept a multiple of 4 = equal to its size
 * class).
 */
const alloc = (name: string, byteLength: number, uses: number, pinned = false): StepOutput => ({
  kind: "alloc",
  byteLength,
  name,
  uses,
  pinned,
});

/**
 * An alias output (a reshape / identity expand that makes the input's data the output as is — ADR
 * 0011).
 */
const alias = (name: string, source: string, uses: number): StepOutput => ({
  kind: "alias",
  source: { kind: "value", name: source },
  name,
  uses,
  pinned: false,
});

/**
 * `DispatchRecipe` is a non-exported type, so it is taken as the element type of the recipe
 * sequence.
 */
type Dispatch = StepRecipe["dispatches"][number];

/** A stub of the roles taken from the WGSL declarations (binding number → read / read-write). */
const roles = (reads: readonly number[], writes: readonly number[]): StorageRoles => ({
  reads: new Set(reads),
  writes: new Set(writes),
});

/**
 * A dispatch stub. The static validation of declarations and the allocation program read only
 * `key` / `bindings` / `roles`; the pipeline and the bind group layout are GPU resources, so the
 * stub does not carry them (avoiding instantiation just to satisfy the type).
 */
const dispatch = (
  key: string,
  bindings: readonly BindingRecipe[] = [],
  access: StorageRoles = roles([], []),
): Dispatch => ({ key, bindings, roles: access }) as unknown as Dispatch;

test("the allocation bookkeeping of single-output steps (alloc / alias / pin / consumption) carries over into the allocation program", () => {
  const program = buildTransientProgram([
    step([alloc("a", 64, 1)]),
    step([alloc("b", 64, 0)], []),
    step([alloc("c", 64, 0, true)], ["a"]),
    // An alias carries its root's name and allocates nothing.
    step([alias("d", "c", 0)]),
    // If the alias source is resident (a weight), there is no root (a value without a slot).
    step([
      {
        kind: "alias",
        source: { kind: "resident", buffer: {} as unknown as GPUBuffer },
        name: "e",
        uses: 0,
        pinned: false,
      },
    ]),
  ]);

  assertEquals(
    program.map((s) => s.outputs),
    [
      [{ name: "a", kind: "alloc", byteLength: 64, source: undefined, uses: 1, pinned: false }],
      [{ name: "b", kind: "alloc", byteLength: 64, source: undefined, uses: 0, pinned: false }],
      [{ name: "c", kind: "alloc", byteLength: 64, source: undefined, uses: 0, pinned: true }],
      [{ name: "d", kind: "alias", byteLength: 0, source: "c", uses: 0, pinned: false }],
      [{ name: "e", kind: "alias", byteLength: 0, source: undefined, uses: 0, pinned: false }],
    ],
  );
  assertEquals(
    program.map((s) => s.releases),
    [[], [], ["a"], [], []],
  );

  // Plan (offsets aligned to 256): a ends its life after c is allocated (step 2 releases a after
  // allocating c), so a and c overlap, while b (no consumer) has already ended. Stacking on a would
  // grow region 0 by 256 + 64 - 64 = 256 >= 64, so the second slot opens region 1 and the third
  // takes its bytes with growth 0. Order 1 (a, b, c): b opens region 1, c lands on b's range —
  // 128 = the live peak (a + c), so the second order does not run (it would also give 128).
  const plan = planRecipes(
    [
      step([alloc("a", 64, 1)]),
      step([alloc("b", 64, 0)], []),
      step([alloc("c", 64, 0, true)], ["a"]),
      step([alias("d", "c", 0)]),
    ],
    CORE_TRANSIENT_LIMITS,
  );
  assertEquals(plan.regions, [64, 64]);
  assertEquals(
    plan.steps.map((assigned) => assigned.outputs),
    [[0], [1], [2], [undefined]],
  );
  assertEquals(
    plan.slots.map((slot) => [slot.region, slot.offset]),
    [
      [0, 0],
      [1, 0],
      [1, 0],
    ],
  );
  assertEquals(plan.pinned, new Set([2]));
  assertEquals(plan.sharedSlots, 1, "c is handed b's bytes again");
});

/**
 * The real shape of `topk` (the first occupant of ADR 0068 decision 3). Pins by hand calculation,
 * from the pure-function side that touches no GPU, the plan of **the same graph** as the real-GPU
 * lifetime check ("per-slot uses / pin of a multi-output node" in tests/runtime_executor_test.ts):
 *
 * ```
 * topk(x[3,8], k=2) → v[3,2] f32 (consumed once by neg) + i[3,2] i32 (graph output)
 * neg(v) → nv[3,2] (consumed once by exp)
 * exp(nv) → e[3,2] (graph output)
 * ```
 *
 * MUST: the two outputs have the **same byte count** (both 3·2 words of 4-byte elements = 24) — the
 * byte total cannot tell them apart. So the expectation pins **which byte range (region, offset)
 * each slot takes** (`e` takes the range of the consumed `v`, not the range of the pinned index
 * `i`).
 */
test("the plan of topk's real shape (values + indices, the same size) matches the hand calculation", () => {
  const bytes = 3 * 2 * 4;
  const plan = planRecipes(
    [
      step([alloc("v", bytes, 1), alloc("i", bytes, 0, true)]),
      step([alloc("nv", bytes, 1)], ["v"]),
      step([alloc("e", bytes, 0, true)], ["nv"]),
    ],
    CORE_TRANSIENT_LIMITS,
  );

  // Intervals: v [0, 3), i [1, end), nv [2, 5), e [4, end). Stacking a 24-byte slot on a live one
  // grows a region by 256 + 24 - 24 = 256 >= 24, so v, i and nv each open a region; e starts after
  // v ends, so it takes v's range in region 0 with growth 0 (i and nv are still live).
  // Order 1 (v, i, nv, e) gives v 0 / i 1 / nv 2 / e 0 (regions, all at offset 0); order 2 (i, v,
  // nv, e by lifetime 5 / 3 / 3 / 2) would put e on v as well; order 1 already totals 3 · 24 = 72 =
  // the live peak (v, i and nv together), so the second order does not run.
  assertEquals(
    plan.slots.map((slot) => [slot.region, slot.offset]),
    [
      [0, 0],
      [1, 0],
      [2, 0],
      [0, 0],
    ],
  );
  assertEquals(
    plan.steps.map((assigned) => assigned.outputs),
    [[0, 1], [2], [3]],
  );
  assertEquals(plan.pinned, new Set([1, 3]));
  // Live peak = v / i / nv together (matches the real-GPU side's `peakTransientBytes` of 72).
  assertEquals(plan.peakLiveBytes, 72);
});

test("a multi-output step allocates in ascending output-slot order and retains / pins each slot independently", () => {
  const plan = planRecipes(
    [step([alloc("v", 64, 1), alloc("i", 32, 0, true)]), step([alloc("y", 64, 0, true)], ["v"])],
    CORE_TRANSIENT_LIMITS,
  );

  assertEquals(
    plan.slots.map((slot) => slot.byteLength),
    [64, 32, 64],
  );
  assertEquals(
    plan.steps.map((assigned) => assigned.outputs),
    [[0, 1], [2]],
  );
  assertEquals(plan.pinned, new Set([1, 2]));
});

// ---------------------------------------------------------------------------
// Read / write roles (WGSL declarations → plan)
// ---------------------------------------------------------------------------

test("a dispatch's reads / writes come from the WGSL roles, and resident values, state and logical lengths do not appear in the plan", () => {
  const program = buildTransientProgram([
    {
      outputs: [alloc("y", 64, 0, true)],
      temps: [{ byteLength: 64, allocBefore: 0, releaseAfter: 0 }],
      dispatches: [
        dispatch(
          "k",
          [
            { binding: 1, source: { kind: "value", name: "x" } },
            { binding: 2, source: { kind: "resident", buffer: {} as unknown as GPUBuffer } },
            { binding: 3, source: { kind: "temp", id: 0 } },
            { binding: 4, source: { kind: "state", name: "kv.k" } },
            { binding: 5, source: { kind: "lengths" } },
            { binding: 6, source: { kind: "value", name: "y" } },
          ],
          roles([1, 2, 4, 5], [3, 6]),
        ),
      ],
      releases: [],
      writesState: false,
    },
  ]);
  assertEquals(program[0].dispatches, [
    {
      reads: [{ kind: "value", name: "x" }],
      writes: [
        { kind: "temp", id: 0 },
        { kind: "value", name: "y" },
      ],
    },
  ]);
});

test("a binding without a role (a binding with no storage declaration in the WGSL) fails loudly", () => {
  const recipes = [
    {
      outputs: [alloc("y", 64, 0, true)],
      temps: [],
      dispatches: [
        dispatch("gemm", [{ binding: 7, source: { kind: "value", name: "y" } }], roles([1], [2])),
      ],
      releases: [],
      writesState: false,
    },
  ];
  const error = assertThrows(() => buildTransientProgram(recipes), ExecutionError);
  assert(
    error.message.includes("dispatch 'gemm' binding 7 has no WGSL storage declaration"),
    error.message,
  );
});

test("an input read and an output written by the same dispatch go to different regions (taken through to the plan)", () => {
  const plan = planRecipes(
    [
      step([alloc("h", 64, 1)]),
      {
        outputs: [alloc("y", 64, 0, true)],
        temps: [],
        dispatches: [
          dispatch(
            "neg",
            [
              { binding: 1, source: { kind: "value", name: "h" } },
              { binding: 2, source: { kind: "value", name: "y" } },
            ],
            roles([1], [2]),
          ),
        ],
        releases: ["h"],
        writesState: false,
      },
    ],
    CORE_TRANSIENT_LIMITS,
  );
  assertEquals(plan.regions, [64, 64]);
  assertEquals(
    plan.slots.map((slot) => slot.region),
    [0, 1],
  );
});

// ---------------------------------------------------------------------------
// Static validation of declarations (validateStepRecipe) and the plan's closure check
// ---------------------------------------------------------------------------

/**
 * A step in which only the temporaries and the dispatches vary (the output bookkeeping is a
 * separate axis of the declaration check, so it is fixed).
 */
const tempStep = (temps: readonly TempRecipe[], dispatches: readonly Dispatch[]): StepRecipe => ({
  outputs: [alloc("y", 64, 0)],
  temps,
  dispatches,
  releases: [],
  writesState: false,
});

test("a temporary's lifetime declaration that does not close inside the dispatch sequence fails at the declaration entry", () => {
  const two = [dispatch("k0"), dispatch("k1")];
  const broken: readonly (readonly [TempRecipe, string])[] = [
    // A temporary built without a release (a forgotten `releaseTemp` = leaks in the execution
    // phase).
    [
      { byteLength: 64, allocBefore: 0, releaseAfter: -1 },
      "temporary 0 has a declared lifetime [0, -1]",
    ],
    // Released at a dispatch before its allocation (the binding dispatch comes before the
    // allocation).
    [
      { byteLength: 64, allocBefore: 1, releaseAfter: 0 },
      "temporary 0 has a declared lifetime [1, 0]",
    ],
    // allocBefore outside the sequence (negative).
    [
      { byteLength: 64, allocBefore: -1, releaseAfter: 1 },
      "temporary 0 has a declared lifetime [-1, 1]",
    ],
    // releaseAfter outside the sequence (index 2 for 2 dispatches).
    [
      { byteLength: 64, allocBefore: 0, releaseAfter: 2 },
      "temporary 0 has a declared lifetime [0, 2]",
    ],
    // A byte count that makes no sense as an allocation spec.
    [{ byteLength: 0, allocBefore: 0, releaseAfter: 1 }, "(0B)"],
  ];
  for (const [temp, message] of broken) {
    assertThrows(() => validateStepRecipe(tempStep([temp], two)), ExecutionError, message);
  }
  // Control: a declaration closed inside the sequence passes (the 5 above differ only in the
  // lifetime declaration).
  validateStepRecipe(tempStep([{ byteLength: 64, allocBefore: 0, releaseAfter: 1 }], two));
});

test("a dispatch that binds an undeclared temporary fails at the declaration entry", () => {
  const temps: readonly TempRecipe[] = [{ byteLength: 64, allocBefore: 0, releaseAfter: 0 }];
  const bind = (id: number): readonly Dispatch[] => [
    dispatch("gemm", [{ binding: 1, source: { kind: "temp", id } }]),
  ];
  assertThrows(
    () => validateStepRecipe(tempStep(temps, bind(1))),
    ExecutionError,
    "dispatch 'gemm' binding 1 points to temporary 1, but 1 are declared",
  );
  // Control: a declared index passes (the same bindings, only the index differs).
  validateStepRecipe(tempStep(temps, bind(0)));
});

test("a dispatch that binds a temporary outside its live interval fails at the declaration entry", () => {
  const read = (id: number): BindingRecipe => ({ binding: 1, source: { kind: "temp", id } });
  const write = (id: number): BindingRecipe => ({ binding: 2, source: { kind: "temp", id } });
  // A is written and released at dispatch 0, B is written and released at dispatch 1, and then
  // dispatch 2 reads A. The plan overlays A and B on the same bytes, so letting it pass would read
  // the value B overwrote.
  const afterRelease = tempStep(
    [
      { byteLength: 64, allocBefore: 0, releaseAfter: 0 },
      { byteLength: 64, allocBefore: 1, releaseAfter: 1 },
    ],
    [dispatch("w0", [write(0)]), dispatch("w1", [write(1)]), dispatch("r2", [read(0)])],
  );
  assertThrows(
    () => validateStepRecipe(afterRelease),
    ExecutionError,
    "dispatch 2 'r2' binding 1 references temporary 0 outside its live range [0, 0]",
  );
  // A dispatch before the allocation binds it (it would read the value of the region's previous
  // owner).
  const beforeAlloc = tempStep(
    [{ byteLength: 64, allocBefore: 1, releaseAfter: 1 }],
    [dispatch("r0", [read(0)]), dispatch("w1", [write(0)])],
  );
  assertThrows(
    () => validateStepRecipe(beforeAlloc),
    ExecutionError,
    "dispatch 0 'r0' binding 1 references temporary 0 outside its live range [1, 1]",
  );
  // Control: binding only inside the interval passes (the first case above with dispatch 2's read
  // moved to B).
  validateStepRecipe(
    tempStep(
      [
        { byteLength: 64, allocBefore: 0, releaseAfter: 0 },
        { byteLength: 64, allocBefore: 1, releaseAfter: 2 },
      ],
      [dispatch("w0", [write(0)]), dispatch("w1", [write(1)]), dispatch("r2", [read(1)])],
    ),
  );
});

test("non-integer lifetime boundaries and temporary indices fail at the declaration entry", () => {
  const two = [dispatch("k0"), dispatch("k1")];
  // Values that comparisons alone let through (a fraction looks inside the interval, and NaN is
  // false in every ordering comparison).
  const broken: readonly (readonly [TempRecipe, string])[] = [
    [
      { byteLength: 64, allocBefore: 0.5, releaseAfter: 1 },
      "temporary 0 has a declared lifetime [0.5, 1]",
    ],
    [
      { byteLength: 64, allocBefore: Number.NaN, releaseAfter: 1 },
      "temporary 0 has a declared lifetime [NaN, 1]",
    ],
    [
      { byteLength: 64, allocBefore: 0, releaseAfter: Number.NaN },
      "temporary 0 has a declared lifetime [0, NaN]",
    ],
  ];
  for (const [temp, message] of broken) {
    assertThrows(() => validateStepRecipe(tempStep([temp], two)), ExecutionError, message);
  }
  const temps: readonly TempRecipe[] = [{ byteLength: 64, allocBefore: 0, releaseAfter: 0 }];
  for (const id of [0.5, Number.NaN]) {
    assertThrows(
      () =>
        validateStepRecipe(
          tempStep(temps, [dispatch("gemm", [{ binding: 1, source: { kind: "temp", id } }])]),
        ),
      ExecutionError,
      `dispatch 'gemm' binding 1 has a temporary index ${id} that is not an integer`,
    );
  }
});

test("a recipe sequence whose consumer count exceeds the actual releases fails the plan's closure check", () => {
  assertThrows(
    // One consumer is declared, but no step anywhere releases the value.
    () => planRecipes([step([alloc("a", 64, 1)])], CORE_TRANSIENT_LIMITS),
    ExecutionError,
    "intermediates are left unreleased",
  );
  // Control: it closes once the consumers' releases come (just one consuming step added to the
  // same sequence).
  assertEquals(
    planRecipes([step([alloc("a", 64, 1)]), step([], ["a"])], CORE_TRANSIENT_LIMITS).regions,
    [64],
  );
});

test("a recipe sequence that releases the same value twice fails on a negative reference count", () => {
  assertThrows(
    () => planRecipes([step([alloc("a", 64, 0)]), step([], ["a", "a"])], CORE_TRANSIENT_LIMITS),
    ExecutionError,
    "has a negative reference count",
  );
});

// ---------------------------------------------------------------------------
// Intervals and reuse of temporaries (temps)
// ---------------------------------------------------------------------------

/**
 * A step with temporaries (only the number of `dispatches` boundaries matters, so stubs suffice).
 *
 * `bakeGroups` uses `steps[].temps` as is to resolve the bind groups' resources, so a temp → slot
 * mapping off by one **binds a different interval of the same size** (the layout is satisfied, so
 * it passes validation, and only the values change silently). The argument the topk comment above
 * makes for outputs is applied to temporaries as is.
 */
const tempSlotsStep = (
  temps: readonly TempRecipe[],
  dispatchCount: number,
  outputs: readonly StepOutput[] = [],
  releases: readonly string[] = [],
): StepRecipe => ({
  outputs,
  temps,
  dispatches: Array.from({ length: dispatchCount }, (_, index) => dispatch(`k${index}`)),
  releases,
  writesState: false,
});

test("temporaries with nested lifetimes that are live at the same time each take a separate interval", () => {
  // The same shape as i8a8 attention: "temporaries outside the loop (0 and 2) enclose a temporary
  // inside the loop (1)".
  const plan = planRecipes(
    [
      tempSlotsStep(
        [
          { byteLength: 64, allocBefore: 0, releaseAfter: 2 },
          { byteLength: 64, allocBefore: 1, releaseAfter: 1 },
          { byteLength: 64, allocBefore: 0, releaseAfter: 2 },
        ],
        3,
      ),
    ],
    CORE_TRANSIENT_LIMITS,
  );

  // All 3 are live at the same time, so no reuse happens (live peak = all 3, sharing 0).
  assertEquals(plan.peakLiveBytes, 192);
  assertEquals(plan.sharedSlots, 0);
  // The allocation order is "declaration order per boundary" — at boundary 0, id 0 / id 2 take
  // intervals 0 / 1, and id 1, born at boundary 1, gets interval 2 (declaration order is not
  // interval order).
  assertEquals(plan.steps[0].temps, [0, 2, 1]);
  // Intervals [0, 5) / [1, 4) / [2, 3): stacking on a live slot grows a region by 256 + 64 - 64 =
  // 256 >= 64, so each slot opens its own region (192 = the live peak, so the second order does not run).
  assertEquals(
    plan.slots.map((slot) => [slot.region, slot.offset]),
    [
      [0, 0],
      [1, 0],
      [2, 0],
    ],
    "each temporary holds its own byte range",
  );
});

test("the bytes of temporaries released at the same boundary are handed to the next step's temporaries", () => {
  const plan = planRecipes(
    [
      tempSlotsStep(
        [
          { byteLength: 64, allocBefore: 0, releaseAfter: 0 },
          { byteLength: 64, allocBefore: 0, releaseAfter: 0 },
        ],
        2,
      ),
      tempSlotsStep([{ byteLength: 64, allocBefore: 0, releaseAfter: 0 }], 1),
    ],
    CORE_TRANSIENT_LIMITS,
  );

  // The second step adds no new bytes (the region stays the size of 2).
  // Intervals [0, 3) / [1, 2) / [4, 5): the second temporary of step 0 overlaps the first (stacking
  // grows by 256 >= 64) and opens region 1; step 1's temporary overlaps neither, both regions give
  // growth 0 and the lower index wins. 128 = the live peak, so the second order does not run.
  assertEquals(plan.regions, [64, 64]);
  assertEquals(plan.steps[0].temps, [0, 1]);
  assertEquals(plan.steps[1].temps, [2]);
  assertEquals(
    [plan.slots[2].region, plan.slots[2].offset],
    [0, 0],
    "equal growth hands out the freed range of the lowest region",
  );
  assertEquals(plan.sharedSlots, 1);
});

test("outputs and temporaries share the same region, and only pinned intervals are not handed out again", () => {
  // An output with zero consumers stops living at the end of its step, so the next step's
  // temporary takes its bytes.
  const shared = planRecipes(
    [
      tempSlotsStep([], 0, [alloc("a", 64, 0)]),
      tempSlotsStep([{ byteLength: 64, allocBefore: 0, releaseAfter: 0 }], 1),
    ],
    CORE_TRANSIENT_LIMITS,
  );
  assertEquals(shared.regions, [64]);
  assertEquals(shared.steps[1].temps, [1]);
  assertEquals(shared.slots[1].offset, 0);

  // A pin (graph output) keeps living even when refs reaches 0.
  const pinnedPlan = planRecipes(
    [
      tempSlotsStep([], 0, [alloc("y", 64, 0, true)]),
      tempSlotsStep([{ byteLength: 64, allocBefore: 0, releaseAfter: 0 }], 1),
    ],
    CORE_TRANSIENT_LIMITS,
  );
  // y [0, end) is still live when the temporary [1, 2) comes: stacking on it grows region 0 by
  // 256 >= 64, so the temporary opens region 1 (128 = the live peak, so the second order does not run).
  assertEquals(pinnedPlan.regions, [64, 64]);
  assertEquals([pinnedPlan.slots[1].region, pinnedPlan.slots[1].offset], [1, 0]);
  assertEquals(pinnedPlan.pinned, new Set([0]));
});

// ---------------------------------------------------------------------------
// Fail loudly on execution without the generation surface (no GPU resource is created)
// ---------------------------------------------------------------------------

/**
 * A stub dispatch that reaches the execution phase (`workgroups` is the field the baked execution
 * reads).
 */
const runnableDispatch = (
  key: string,
  workgroups: DispatchWorkgroups,
  bindings: readonly BindingRecipe[] = [],
): Dispatch => ({ key, bindings, workgroups, roles: roles([1], []) }) as unknown as Dispatch;

/**
 * A step with dispatches only (it has no outputs or temporaries, so the bookkeeping does not move).
 */
const dispatchStep = (dispatches: readonly Dispatch[]): StepRecipe => ({
  outputs: [],
  temps: [],
  dispatches,
  releases: [],
  writesState: false,
});

/** A scheduler that only records the enqueued dispatches (it holds no GPU resources). */
const recordingScheduler = (log: string[]): SubmitScheduler =>
  ({
    dispatch: (
      _pipeline: unknown,
      _group: unknown,
      workgroups: readonly [number, number, number],
      key: string,
    ) => {
      log.push(`${key} [${workgroups.join(",")}]`);
    },
  }) as unknown as SubmitScheduler;

const stubGroup = (): GPUBindGroup => ({}) as unknown as GPUBindGroup;

test("the baked path fails on a dispatch that has not been through the context-side baking", () => {
  const recipes = [dispatchStep([runnableDispatch("attn.qk", [1, 1, 1])])];
  const log: string[] = [];

  const error = assertThrows(
    () => executeBakedPlan(recipes, [[undefined]], recordingScheduler(log)),
    ExecutionError,
  );
  assert(error.message.includes("dispatch 'attn.qk'"), error.message);
  assert(error.message.includes("ADR 0066 decision 5"), error.message);
  assertEquals(log, []);

  // Control: with the hole filled, it does not throw and one dispatch is enqueued (proof that the
  // case above is not a false negative).
  executeBakedPlan(recipes, [[stubGroup()]], recordingScheduler(log));
  assertEquals(log, ["attn.qk [1,1,1]"]);
});

test("a dispatch that computes its workgroup count from the logical length fails without a GenerationContext", () => {
  const log: string[] = [];
  const derived = [dispatchStep([runnableDispatch("state.pv", (past, query) => [past, query, 1])])];

  const error = assertThrows(
    () => executeBakedPlan(derived, [[stubGroup()]], recordingScheduler(log)),
    ExecutionError,
    "a dispatch whose workgroup count is derived from the logical length without a GenerationContext",
  );
  assert(error.message.includes("dispatch 'state.pv'"), error.message);

  // Control: a static triple is enqueued even without generation. **Enqueueing even 0** is the
  // existing contract (the path of 0-element tensors does not silently skip a dispatch with 0
  // elements).
  executeBakedPlan(
    [dispatchStep([runnableDispatch("elementwise", [0, 1, 1])])],
    [[stubGroup()]],
    recordingScheduler(log),
  );
  assertEquals(log, ["elementwise [0,1,1]"]);
});

test("only a dispatch with an axis of 0 workgroups computed from the logical length is not enqueued", () => {
  const log: string[] = [];
  const recipes = [dispatchStep([runnableDispatch("state.pv", (past, query) => [past, query, 1])])];
  const generation = (past: number, query: number): BakedGeneration => ({
    groups: [[undefined]],
    encoding: {
      slots: new Map(),
      lengths: {} as unknown as GPUBuffer,
      past,
      query,
    },
    onStep: () => {},
  });

  // Zero valid rows (this row block is entirely pad rows) is not enqueued.
  executeBakedPlan(recipes, [[stubGroup()]], recordingScheduler(log), generation(0, 4));
  assertEquals(log, []);
  // Control: with valid rows, the same recipe is enqueued.
  executeBakedPlan(recipes, [[stubGroup()]], recordingScheduler(log), generation(2, 4));
  assertEquals(log, ["state.pv [2,4,1]"]);
});

/**
 * A scheduler that records the keys of enqueued dispatches and returns `backpressure` in the order
 * of the given sequence (the wait of the in-flight submit bound — ADR 0123).
 */
const pacedScheduler = (
  log: string[],
  waits: readonly (Promise<void> | undefined)[],
): SubmitScheduler => {
  let calls = 0;
  return {
    dispatch: (_pipeline: unknown, _group: unknown, _workgroups: unknown, key: string) => {
      log.push(key);
    },
    backpressure: () => waits[calls++],
  } as unknown as SubmitScheduler;
};

test("a run's baked execution does not enqueue the next step's dispatches while waiting on the submit bound", async () => {
  const log: string[] = [];
  const gate = Promise.withResolvers<void>();
  const recipes = [
    dispatchStep([runnableDispatch("a", [1, 1, 1])]),
    dispatchStep([runnableDispatch("b", [1, 1, 1])]),
  ];

  const running = executeBakedPlanPaced(
    recipes,
    [[stubGroup()], [stubGroup()]],
    pacedScheduler(log, [gate.promise]),
  );
  assert(running !== undefined, "returns a Promise when a wait is needed");
  await new Promise((resolve) => setTimeout(resolve, 0));
  assertEquals(log, ["a"], "does not enqueue the next step until the wait resolves");

  gate.resolve();
  await running;
  assertEquals(log, ["a", "b"]);
});

test("a run's baked execution returns the rejection without enqueueing the next step when the submit-bound wait rejects", async () => {
  const log: string[] = [];
  const lost = new GpuDeviceLostError(
    "simulated: device lost while waiting on a completion marker",
  );
  const recipes = [
    dispatchStep([runnableDispatch("a", [1, 1, 1])]),
    dispatchStep([runnableDispatch("b", [1, 1, 1])]),
  ];

  const running = executeBakedPlanPaced(
    recipes,
    [[stubGroup()], [stubGroup()]],
    pacedScheduler(log, [Promise.reject(lost)]),
  );
  assert(running !== undefined, "returns a Promise when a wait is needed");
  const error = await assertRejects(() => running);
  // A failed wait is returned unwrapped (the run's catch picks the cleanup by the loss error type).
  assertStrictEquals(error, lost);
  assertEquals(log, ["a"], "enqueued the next step after the wait failed");
});

test("a run's baked execution delivers a throw in the continuation after the wait as a rejection of the returned Promise, and enqueues no next step", async () => {
  const log: string[] = [];
  const recipes = [
    dispatchStep([runnableDispatch("a", [1, 1, 1])]),
    dispatchStep([runnableDispatch("b", [1, 1, 1])]),
  ];

  // The second step has no bind group (and no generation surface) — it throws inside the
  // continuation after the wait. What this pins is that the throw arrives as a rejection of the
  // Promise returned to the caller (detaching the continuation from the returned chain would make
  // it an unhandled rejection and turn this red) and that nothing is enqueued after the throw. The
  // continuation runs after the wait, so the catch below (a synchronous throw) is not reached in
  // the current form — it only guards against a regression that moves the continuation before the
  // wait.
  let running: Promise<void> | undefined;
  try {
    running = executeBakedPlanPaced(
      recipes,
      [[stubGroup()], []],
      pacedScheduler(log, [Promise.resolve()]),
    );
  } catch (cause) {
    throw new Error(`the throw after the wait was thrown synchronously from the call: ${cause}`);
  }
  assert(running !== undefined, "returns a Promise when a wait is needed");
  await assertRejects(() => running, ExecutionError, "dispatch 'b'");
  assertEquals(log, ["a"]);
});

test("a run's baked execution within the bound inserts no wait and enqueues the same sequence as the synchronous execution", () => {
  const recipes = [
    dispatchStep([runnableDispatch("a", [1, 1, 1]), runnableDispatch("b", [1, 1, 1])]),
    dispatchStep([runnableDispatch("c", [1, 1, 1])]),
  ];
  const groups = [[stubGroup(), stubGroup()], [stubGroup()]];
  const paced: string[] = [];
  const plain: string[] = [];

  // A run that needs no wait returns no Promise (the caller's await does not yield even one
  // microtask).
  assertEquals(executeBakedPlanPaced(recipes, groups, pacedScheduler(paced, [])), undefined);
  assertEquals(paced, ["a", "b", "c"], "everything is enqueued by the time the call returns");

  // The enqueue path (synchronous execution) never asks about the submit-bound wait (ADR 0123
  // decision 4).
  const neverAsked = {
    dispatch: (_pipeline: unknown, _group: unknown, _workgroups: unknown, key: string) => {
      plain.push(key);
    },
    backpressure: () => {
      throw new Error("the enqueue path called backpressure");
    },
  } as unknown as SubmitScheduler;
  executeBakedPlan(recipes, groups, neverAsked);
  assertEquals(paced, plain);
});

/** Runs only the resolution of bind group resources (a device stub that holds no GPU resources). */
const bakeOne = (
  source: BindingSource,
  generation?: { readonly slots: ReadonlyMap<string, GPUBuffer> },
): void => {
  const recipes = [
    dispatchStep([runnableDispatch("attn.pv", [1, 1, 1], [{ binding: 1, source }])]),
  ];
  bakeAllBindGroups(
    recipes,
    {
      device: { createBindGroup: stubGroup } as unknown as GPUDevice,
      plan: planRecipes(recipes, CORE_TRANSIENT_LIMITS),
      regions: [],
      inputs: new Map(),
    },
    generation === undefined
      ? undefined
      : {
          slots: generation.slots,
          lengths: {} as unknown as GPUBuffer,
          past: 0,
          query: 1,
        },
  );
};

test("execution without the generation surface fails loudly on state / logical-length bindings", () => {
  const lengths = assertThrows(() => bakeOne({ kind: "lengths" }), ExecutionError);
  assert(
    lengths.message.includes(
      "logical-length uniform: tried to run a dispatch that binds it without a GenerationContext",
    ),
    lengths.message,
  );

  const state = assertThrows(() => bakeOne({ kind: "state", name: "kv.k" }), ExecutionError);
  assert(
    state.message.includes(
      "state slot 'kv.k': tried to run a dispatch that binds it without a GenerationContext",
    ),
    state.message,
  );
});

test("a slot name the GenerationContext does not have fails loudly at resource resolution", () => {
  const error = assertThrows(
    () =>
      bakeOne(
        { kind: "state", name: "kv.v" },
        {
          slots: new Map([["kv.k", {} as unknown as GPUBuffer]]),
        },
      ),
    ExecutionError,
  );
  assert(
    error.message.includes("state slot 'kv.v' has no buffer in the GenerationContext"),
    error.message,
  );
});

/**
 * The check of the run shape (`assertGenerationRun` — ADR 0066 decision 4 / addendum on buckets).
 *
 * The second argument is the **allowed set** `{1} ∪ chunkBuckets ∪ {chunkLength}` the context has
 * already built, and this checks only "pass if in the set, fail if not" (how the set is built is
 * `assertChunkBuckets` in runtime_generation_context_test.ts; the real-GPU wiring is
 * gpu_state_execution_test.ts).
 */
const chunkLimits = (rows: readonly number[]): GenerationLimits => ({
  chunkRows: new Set(rows),
  fullCapacities: new Map(),
});

test("assertGenerationRun passes only physical chunk row counts in the allowed set", () => {
  // The set a context with chunkLength=8 / buckets [2,4] builds.
  const allowed = new Set([1, 2, 4, 8]);

  // The 4 of the decode shape, the 2 buckets and the prefill shape pass as they are.
  for (const rows of [1, 2, 4, 8]) {
    assertGenerationRun(chunkLimits([rows]), allowed, 0, 1);
  }
  // Even with several state nodes in one run, it passes when every node is in the set.
  assertGenerationRun(chunkLimits([4]), allowed, 3, 4);

  // Outside the set fails (3 in the gap between buckets / 5 above the buckets / 9 over
  // chunkLength).
  for (const rows of [3, 5, 9]) {
    const error = assertThrows(
      () => assertGenerationRun(chunkLimits([rows]), allowed, 0, 1),
      ExecutionError,
    );
    assert(error.message.includes("fixed chunk contract"), error.message);
    // The diagnostic lists the allowed set (showing what is allowed at the call site).
    assert(error.message.includes("{2, 4, 8}"), error.message);
  }

  // With a set without buckets, the same 2 / 4 fail (the converse showing the permission comes from
  // the buckets).
  const plain = new Set([1, 8]);
  for (const rows of [2, 4]) {
    const error = assertThrows(
      () => assertGenerationRun(chunkLimits([rows]), plain, 0, 1),
      ExecutionError,
    );
    assert(error.message.includes("{8}"), error.message);
  }
});

test("assertGenerationRun rejects a run whose queryLength exceeds the physical chunk row count", () => {
  const allowed = new Set([1, 4, 8]);
  // The limit of the bucket rows (4) is 4 — not chunkLength's 8.
  assertGenerationRun(chunkLimits([4]), allowed, 0, 4);
  const error = assertThrows(
    () => assertGenerationRun(chunkLimits([4]), allowed, 0, 5),
    ExecutionError,
  );
  assert(error.message.includes("exceeds the physical chunk row count 4"), error.message);
});

test("assertGenerationRun rejects exceeding a full slot's capacity regardless of the allowed set", () => {
  const limits: GenerationLimits = {
    chunkRows: new Set([4]),
    fullCapacities: new Map([["kslot", 6]]),
  };
  const allowed = new Set([1, 4, 8]);
  assertGenerationRun(limits, allowed, 2, 4);
  const error = assertThrows(() => assertGenerationRun(limits, allowed, 3, 4), ExecutionError);
  assert(error.message.includes("state 'kslot'"), error.message);
});
