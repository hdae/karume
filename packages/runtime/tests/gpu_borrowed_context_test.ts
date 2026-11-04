// **Execution integration** checks of the borrowing context (drafter) and shared weights (ADR 0096 stage 2 §2.1–§2.3).
//
// The declaration and contract checks live in tests/runtime_external_states_test.ts (GPU-independent).
// This file looks at the wiring itself: "two Sessions run sharing one KV and one set of weights":
//
//   (1) after the lender has run for a few steps, the borrower's run is within the band of an f64
//       reference (`referenceStateAttentionReadonly`) built from **the actual contents of the lender's
//       slots** (through 3 phases: P < W / P = W / after the ring has wrapped once)
//   (2) serialization: the lender's run in progress / awaiting commit / poisoned are the three reasons
//       to reject a borrower run
//   (3) lifetime: disposing the lender context while a borrower is alive is an ExecutionError; it
//       succeeds after the borrower is disposed
//   (4) plan key: two borrowing contexts bound to lenders of different capacities get **different
//       keys** (the detector that the external slot's capacity is part of the key)
//   (5) shared weights: the borrower reads the lender's i8 embedding table and the values are
//       **bit-identical** to the lender's. A mismatch of shape / semantic dtype / residency class /
//       row axis fails loudly; disposing the lender while a borrower is alive is also rejected
//   (6) fork (ADR 0066 addendum <2026-10-11>, ruling 36): a lender with a live borrower can be forked
//       (rejected only while the borrower's run is running). After the fork the original borrower
//       still reads the original slots, and the copy has no borrower. A borrowing context itself
//       cannot be forked
//
// MUST: the slots handed to the reference are **read back from the GPU** (not a reproduction by a
// host-side oracle). The borrower reads the rows the lender actually wrote, so only a readback can
// show that "they see the same storage".

import { test } from "vite-plus/test";
import {
  assert,
  assertEquals,
  assertInstanceOf,
  assertNotEquals,
  assertRejects,
  assertThrows,
  fail,
} from "@std/assert";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { referenceStateAttentionReadonly } from "../src/reference/state-attention.ts";
import { decodeI8 } from "../src/format/i8.ts";
import { openMemoryContainer } from "../src/format/container/memory.ts";
import { type IrDeclaration, parseIrDeclarationValue } from "../src/format/ir.ts";
import { acquireGpu, type GpuContext, RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import { createSessionFromContainer, type Session, type Tensor } from "../src/runtime/executor.ts";
import type { GenerationContext } from "../src/runtime/generation-context.ts";
import { LENGTHS_BYTES } from "../src/runtime/generation-context.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { type DeclarationJson, f32Bytes, fill, openGraphModel } from "./helpers/model-fixture.ts";
import { halfScale, seeded } from "./helpers/state-dispatch.ts";
import { i8BytesFrom } from "./helpers/i8.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** The band is the same as, and for the same reasons as, the states-form execution integration check (the reference accumulates in f64, the GPU accumulates sequentially in f32). */
const STATE_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

const HEADS = 4;
const KV_HEADS = 1;
const DEPTH = 8;
/** The sliding window (MUST be the same for lender and borrower — reads and writes use the same formula). */
const WINDOW = 8;

const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

const SCALE = halfScale(DEPTH);

/** The lender (the target's role): 2 sliding slots of its own + a states-form attention + 2 appends. */
const lenderGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["attention", "state_append"] },
  symbols: ["M", "C"],
  inputs: [
    { name: "q", dtype: "f32", shape: [1, HEADS, "M", DEPTH] },
    { name: "k", dtype: "f32", shape: [1, KV_HEADS, "M", DEPTH] },
    { name: "v", dtype: "f32", shape: [1, KV_HEADS, "M", DEPTH] },
  ],
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, HEADS, "M", DEPTH] } },
  states: {
    kslot: { dtype: "f32", shape: [1, KV_HEADS, "C", DEPTH] },
    vslot: { dtype: "f32", shape: [1, KV_HEADS, "C", DEPTH] },
  },
  nodes: [
    {
      op: "attention",
      ins: ["q", "k", "v"],
      outs: ["o"],
      attrs: { scale: SCALE, window: WINDOW },
      states: { k: "kslot", v: "vslot" },
    },
    {
      op: "state_append",
      ins: ["k"],
      outs: [],
      attrs: { window: WINDOW },
      states: { slot: "kslot" },
    },
    {
      op: "state_append",
      ins: ["v"],
      outs: [],
      attrs: { window: WINDOW },
      states: { slot: "vslot" },
    },
  ],
});

/** The borrower (the drafter's role): one readonly attention reads external slots with **the same names** as the lender's. */
const borrowerGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["attention"] },
  symbols: ["C"],
  inputs: [{ name: "q", dtype: "f32", shape: [1, HEADS, 1, DEPTH] }],
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, HEADS, 1, DEPTH] } },
  states: {
    kslot: { dtype: "f32", shape: [1, KV_HEADS, "C", DEPTH], external: true },
    vslot: { dtype: "f32", shape: [1, KV_HEADS, "C", DEPTH], external: true },
  },
  nodes: [
    {
      op: "attention",
      ins: ["q"],
      outs: ["o"],
      attrs: { scale: SCALE, window: WINDOW, readonly: true },
      states: { k: "kslot", v: "vslot" },
    },
  ],
});

const tensor = (shape: readonly number[], data: Float32Array<ArrayBuffer>): Tensor => ({
  dtype: "f32",
  shape,
  data,
});

/** Reads a GPU buffer back as f32 (same shape as readRaw in helpers/state-dispatch.ts). */
const readSlot = async (
  device: GPUDevice,
  buffer: GPUBuffer,
  count: number,
): Promise<Float32Array<ArrayBuffer>> => {
  const size = Math.max(4, count * 4);
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return new Float32Array(copy, 0, count);
};

const internals = (context: GenerationContext) => context[RUNTIME_INTERNAL];

/** Reads back the 2 slots of the lender context (the very storage the borrower reads). */
const readLenderSlots = async (
  gpu: GpuContext,
  context: GenerationContext,
  capacity: number,
): Promise<{ slotK: Float32Array<ArrayBuffer>; slotV: Float32Array<ArrayBuffer> }> => {
  const count = KV_HEADS * capacity * DEPTH;
  const slots = internals(context).slots;
  const k = slots.get("kslot");
  const v = slots.get("vslot");
  assert(k !== undefined && v !== undefined, "the lender's slots cannot be looked up");
  return {
    slotK: await readSlot(gpu.device, k.buffer, count),
    slotV: await readSlot(gpu.device, v.buffer, count),
  };
};

/** Advances the lender by 1 step (`chunkRows` rows x `query` valid rows). */
const lenderStep = async (
  session: Session,
  context: GenerationContext,
  chunkRows: number,
  query: number,
  salt: number,
  commit: "immediate" | "deferred" = "immediate",
): Promise<void> => {
  await session.run(
    {
      q: tensor(
        [1, HEADS, chunkRows, DEPTH],
        seeded(HEADS * chunkRows * DEPTH, (i) => QUERY(i + salt)),
      ),
      k: tensor(
        [1, KV_HEADS, chunkRows, DEPTH],
        seeded(KV_HEADS * chunkRows * DEPTH, (i) => KEY(i + salt)),
      ),
      v: tensor(
        [1, KV_HEADS, chunkRows, DEPTH],
        seeded(KV_HEADS * chunkRows * DEPTH, (i) => VALUE(i + salt)),
      ),
    },
    {},
    { context, queryLength: query, commit },
  );
};

/** Runs the borrower once and compares it with a reference built from the actual contents of the lender's slots. */
const assertDraft = async (
  gpu: GpuContext,
  borrower: Session,
  borrowed: GenerationContext,
  lender: GenerationContext,
  capacity: number,
  salt: number,
  label: string,
): Promise<void> => {
  const past = lender.pastLength;
  const q = seeded(HEADS * DEPTH, (i) => QUERY(i + salt));
  const { slotK, slotV } = await readLenderSlots(gpu, lender, capacity);
  const outputs = await borrower.run(
    { q: tensor([1, HEADS, 1, DEPTH], q) },
    {},
    { context: borrowed, queryLength: 1 },
  );
  const actual = outputs["o"].data as Float32Array<ArrayBuffer>;
  const expected = referenceStateAttentionReadonly({
    batch: 1,
    heads: HEADS,
    kvHeads: KV_HEADS,
    depth: DEPTH,
    capacity,
    window: WINDOW,
    past,
    q,
    slotK,
    slotV,
    scale: SCALE,
  });
  // MUST: check every time that the expected value is not trivial (non-zero except at P = 0 — avoids a comparison that always holds).
  assert(
    (expected.data as Float32Array).some((value) => Math.abs(value) > 1e-3),
    `${label}: the expected output is trivial (all ~0)`,
  );
  const report = compareTensors(
    { dtype: "f32", data: actual },
    { dtype: "f32", data: expected.data as Float32Array<ArrayBuffer> },
    STATE_TOLERANCE,
  );
  assertEquals(report.pass, true, `${label}: ${formatAllclose(report)}`);
  // The borrower moves the logical length by not a single step (only the lender's run advances it).
  assertEquals(
    lender.pastLength,
    past,
    `${label}: a borrower run moved the lender's logical length`,
  );
  assertEquals(
    borrowed.pastLength,
    past,
    `${label}: the borrower's pastLength is the last P it copied`,
  );
};

test.skipIf(!GPU_AVAILABLE)(
  "a run of the borrowing context matches the actual contents of the lender's slots (P < W / = W / after the ring wraps, real GPU)",
  async () => {
    const capacity = 16;
    const gpu = await acquireGpu();
    const lender = await createSessionFromContainer(
      gpu,
      await openGraphModel(lenderGraph()),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(borrowerGraph()),
      "model",
    );
    const lenderContext = await lender.createGenerationContext({
      chunkLength: 4,
      bindings: { C: capacity },
    });
    const borrowed = await borrower.createGenerationContext({
      chunkLength: 1,
      borrow: lenderContext,
    });
    try {
      // P = 0 (0 columns = an empty row -> exactly 0). The reference is 0 too, so the triviality check is not applied.
      const empty = await borrower.run(
        { q: tensor([1, HEADS, 1, DEPTH], seeded(HEADS * DEPTH, QUERY)) },
        {},
        { context: borrowed, queryLength: 1 },
      );
      for (const value of empty["o"].data as Float32Array) {
        assert(Object.is(value, 0), `the output at P = 0 is not exactly 0 (${value})`);
      }

      // prefill (M=4) -> stacked decodes go through the 3 phases: P < W (4) / P = W (8) / after the ring wraps (18).
      await lenderStep(lender, lenderContext, 4, 4, 0);
      await assertDraft(gpu, borrower, borrowed, lenderContext, capacity, 101, "P=4 < W");
      for (let step = 0; step < 4; step += 1) {
        await lenderStep(lender, lenderContext, 1, 1, 11 + step * 7);
      }
      assertEquals(lenderContext.pastLength, 8);
      await assertDraft(gpu, borrower, borrowed, lenderContext, capacity, 202, "P=8 = W");
      for (let step = 0; step < 10; step += 1) {
        await lenderStep(lender, lenderContext, 1, 1, 71 + step * 5);
      }
      assertEquals(lenderContext.pastLength, 18, "the ring has wrapped once (C=16)");
      await assertDraft(
        gpu,
        borrower,
        borrowed,
        lenderContext,
        capacity,
        303,
        "P=18 (after wrapping)",
      );

      // The second and later drafts hit the derived plan (the borrower is in the ordinary plan cache too).
      assertEquals(borrower.diagnostics().lastRunPrepared?.hit, true);
      // The borrower writes not a single row of state, so drafts do not move the lender's logical length.
      assertEquals(lenderContext.pastLength, 18);

      // Diagnostics are split per Session (a borrower run does not touch the lender's stateBacking).
      // The borrower holds only the 8 bytes of the logical-length uniform — the slots belong to the lender,
      // so counting them on both sides would count the same VRAM twice.
      const lenderState = lender.diagnostics().stateBacking;
      const borrowerState = borrower.diagnostics().stateBacking;
      assertEquals(borrowerState.contextCount, 1);
      assertEquals(borrowerState.residentBytes, LENGTHS_BYTES);
      assertEquals(lenderState.contextCount, 1);
      assertEquals(
        lenderState.residentBytes,
        2 * KV_HEADS * capacity * DEPTH * 4 + LENGTHS_BYTES,
        "the lender counts its 2 slots + the logical length",
      );
    } finally {
      await borrowed.dispose();
      await lenderContext.dispose();
      await borrower.dispose();
      await lender.dispose();
      gpu.destroy();
    }
  },
);

/**
 * Proof that the parity check above is **not vacuous** (fault injection). It breaks the reference
 * side in 3 ways and checks that each falls outside the band:
 *
 *   (1) make P one step stale — the detector that the borrower copies "the lender's **current** P"
 *   (2) swap the k / v slots — the detector of the binding direction (both have the same shape, so
 *       shape does not catch it)
 *   (3) wipe one row inside the live window — the detector that the borrower actually reads that row
 *
 * MUST: all 3 break "only the reference side" (the GPU side stays correct). Breaking both sides by
 * the same amount cancels the error and makes the check always hold.
 */
test.skipIf(!GPU_AVAILABLE)(
  "fault injection into the borrower parity (stale P, swapped k/v, a missing live row) fails the band (real GPU)",
  async () => {
    const capacity = 16;
    const gpu = await acquireGpu();
    const lender = await createSessionFromContainer(
      gpu,
      await openGraphModel(lenderGraph()),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(borrowerGraph()),
      "model",
    );
    const lenderContext = await lender.createGenerationContext({
      chunkLength: 4,
      bindings: { C: capacity },
    });
    const borrowed = await borrower.createGenerationContext({
      chunkLength: 1,
      borrow: lenderContext,
    });
    try {
      await lenderStep(lender, lenderContext, 4, 4, 0);
      for (let step = 0; step < 3; step += 1) {
        await lenderStep(lender, lenderContext, 1, 1, 21 + step * 3);
      }
      const past = lenderContext.pastLength;
      assertEquals(past, 7);
      const q = seeded(HEADS * DEPTH, (i) => QUERY(i + 55));
      const { slotK, slotV } = await readLenderSlots(gpu, lenderContext, capacity);
      const outputs = await borrower.run(
        { q: tensor([1, HEADS, 1, DEPTH], q) },
        {},
        { context: borrowed, queryLength: 1 },
      );
      const actual = outputs["o"].data as Float32Array<ArrayBuffer>;
      const base = {
        batch: 1,
        heads: HEADS,
        kvHeads: KV_HEADS,
        depth: DEPTH,
        capacity,
        window: WINDOW,
        q,
        scale: SCALE,
      };
      const inBand = (expected: Float32Array<ArrayBuffer>): boolean =>
        compareTensors(
          { dtype: "f32", data: actual },
          { dtype: "f32", data: expected },
          STATE_TOLERANCE,
        ).pass;

      // The correct reference is inside the band (the premise that the check holds).
      assert(
        inBand(
          referenceStateAttentionReadonly({ ...base, past, slotK, slotV })
            .data as Float32Array<ArrayBuffer>,
        ),
        "the correct reference is outside the band (the check itself is broken)",
      );
      // (1) stale P
      assert(
        !inBand(
          referenceStateAttentionReadonly({ ...base, past: past - 1, slotK, slotV })
            .data as Float32Array<ArrayBuffer>,
        ),
        "still inside the band with P one step stale (the borrower may not see the lender's current P)",
      );
      // (2) swapped k / v
      assert(
        !inBand(
          referenceStateAttentionReadonly({ ...base, past, slotK: slotV, slotV: slotK })
            .data as Float32Array<ArrayBuffer>,
        ),
        "still inside the band with k / v swapped (the binding direction is not detected)",
      );
      // (3) wipe one row inside the live window (logical position P−1 = the row of the latest committed token)
      const damagedK = new Float32Array(slotK);
      const row = (past - 1) % capacity;
      for (let d = 0; d < DEPTH; d += 1) damagedK[row * DEPTH + d] = 0;
      assert(
        !inBand(
          referenceStateAttentionReadonly({ ...base, past, slotK: damagedK, slotV })
            .data as Float32Array<ArrayBuffer>,
        ),
        "still inside the band with the live row wiped (that row may not be read)",
      );
    } finally {
      await borrowed.dispose();
      await lenderContext.dispose();
      await borrower.dispose();
      await lender.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a borrower run is rejected when the lender has a run in progress / awaits commit / is poisoned (real GPU)",
  async () => {
    const capacity = 16;
    const gpu = await acquireGpu();
    const lender = await createSessionFromContainer(
      gpu,
      await openGraphModel(lenderGraph()),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(borrowerGraph()),
      "model",
    );
    const lenderContext = await lender.createGenerationContext({
      chunkLength: 4,
      bindings: { C: capacity },
    });
    const borrowed = await borrower.createGenerationContext({
      chunkLength: 1,
      borrow: lenderContext,
    });
    const draft = (): Promise<unknown> =>
      borrower.run(
        { q: tensor([1, HEADS, 1, DEPTH], seeded(HEADS * DEPTH, QUERY)) },
        {},
        { context: borrowed, queryLength: 1 },
      );
    try {
      await lenderStep(lender, lenderContext, 4, 4, 0);
      await draft();

      // (1) A lender run in progress (the lease is taken in the synchronous section of the issue, so an order without await hits it).
      const running = lenderStep(lender, lenderContext, 1, 1, 5);
      const concurrent = await assertRejects(draft, ExecutionError);
      assert(concurrent.message.includes("generation run in progress"), concurrent.message);
      await running;

      // (2) Awaiting commit (a deferred lender run — drafts come after the commit).
      await lenderStep(lender, lenderContext, 1, 1, 9, "deferred");
      const pending = await assertRejects(draft, ExecutionError);
      assert(pending.message.includes("awaiting commit"), pending.message);
      lenderContext.commit(1);
      await draft();

      // (3) A borrower run cannot specify deferred (there is nothing for it to commit).
      const deferred = await assertRejects(
        () =>
          borrower.run(
            { q: tensor([1, HEADS, 1, DEPTH], seeded(HEADS * DEPTH, QUERY)) },
            {},
            { context: borrowed, queryLength: 1, commit: "deferred" },
          ),
        ExecutionError,
      );
      assert(deferred.message.includes("borrowing context"), deferred.message);

      // (4) The borrower's commit / rewind is always rejected (it has no logical length).
      assertThrows(() => borrowed.commit(0), ExecutionError, "borrowing context");
      assertThrows(() => borrowed.rewind(0), ExecutionError, "borrowing context");

      // (5) A poisoned lender is a reason to reject a borrower run (fault injection).
      internals(lenderContext).poison("fault injection by the test");
      const poisoned = await assertRejects(draft, ExecutionError);
      assert(poisoned.message.includes("poisoned"), poisoned.message);
    } finally {
      await borrowed.dispose();
      await lenderContext.dispose();
      await borrower.dispose();
      await lender.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "disposing the lender context is rejected while a borrower is alive and succeeds after the borrower is disposed (real GPU)",
  async () => {
    const capacity = 16;
    const gpu = await acquireGpu();
    const lender = await createSessionFromContainer(
      gpu,
      await openGraphModel(lenderGraph()),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(borrowerGraph()),
      "model",
    );
    const lenderContext = await lender.createGenerationContext({
      chunkLength: 4,
      bindings: { C: capacity },
    });
    const borrowed = await borrower.createGenerationContext({
      chunkLength: 1,
      borrow: lenderContext,
    });
    try {
      await lenderStep(lender, lenderContext, 4, 4, 0);
      const refused = await assertRejects(() => lenderContext.dispose(), ExecutionError);
      assert(refused.message.includes("contexts are borrowing"), refused.message);
      // A rejected dispose does not break the lender (the end-of-intake flag is not set either).
      assertEquals(lenderContext.pastLength, 4);
      await assertDraft(
        gpu,
        borrower,
        borrowed,
        lenderContext,
        capacity,
        404,
        "after the rejection",
      );

      await borrowed.dispose();
      // Once the borrower has gone, it succeeds.
      await lenderContext.dispose();
    } finally {
      await borrower.dispose();
      await lender.dispose();
      gpu.destroy();
    }
  },
);

/**
 * The output of one borrower run (u32 — it checks "which storage it read", not how close the values are).
 *
 * MUST: `borrower.run` is issued in the synchronous section of the call (the reservation is taken synchronously — the rejection check depends on this).
 */
const draftWords = async (
  borrower: Session,
  borrowed: GenerationContext,
  salt: number,
): Promise<number[]> => {
  const outputs = await borrower.run(
    {
      q: tensor(
        [1, HEADS, 1, DEPTH],
        seeded(HEADS * DEPTH, (i) => QUERY(i + salt)),
      ),
    },
    {},
    { context: borrowed, queryLength: 1 },
  );
  const data = outputs["o"].data;
  return Array.from(new Uint32Array(data.buffer, data.byteOffset, data.length));
};

test.skipIf(!GPU_AVAILABLE)(
  "fork: a lender with a borrower can be copied, and the original borrower keeps reading the original slots (real GPU)",
  async () => {
    const capacity = 16;
    const gpu = await acquireGpu();
    const lender = await createSessionFromContainer(
      gpu,
      await openGraphModel(lenderGraph()),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(borrowerGraph()),
      "model",
    );
    const lenderContext = await lender.createGenerationContext({
      chunkLength: 4,
      bindings: { C: capacity },
    });
    const borrowed = await borrower.createGenerationContext({
      chunkLength: 1,
      borrow: lenderContext,
    });
    /** Contexts created later (cleaned up in the reverse order of creation). */
    const extra: GenerationContext[] = [];
    try {
      await lenderStep(lender, lenderContext, 4, 4, 0);
      for (let step = 0; step < 3; step += 1) {
        await lenderStep(lender, lenderContext, 1, 1, 11 + step * 7);
      }

      // Rejected while the borrower's run is running (the borrower's run also takes the lender's reservation — rejection (d)).
      const drafting = draftWords(borrower, borrowed, 0);
      await assertRejects(
        () => lenderContext.fork(),
        ExecutionError,
        "generation run is in progress",
      );
      await drafting;
      // A borrowing context itself cannot be copied (rejection (g) — the slots belong to the lender).
      await assertRejects(
        () => borrowed.fork(),
        ExecutionError,
        "a borrowing context cannot be forked",
      );

      // A lender with a live borrower can be forked (ruling 36 = a). A context that includes sliding is not rejected either.
      const forked = await lenderContext.fork();
      extra.push(forked);
      assertEquals(forked.pastLength, lenderContext.pastLength);
      assertEquals(forked.slidingSlack, lenderContext.slidingSlack);

      // Advance the original and the copy by one row each with **different inputs** (different KV lands at the same logical position).
      await lenderStep(lender, lenderContext, 1, 1, 101);
      await lenderStep(lender, forked, 1, 1, 202);

      // The original borrower stays bound to the original slots: its output matches, as u32, "a borrower
      // newly created on the original" and differs from "a borrower created on the copy". An implementation
      // that re-points the original at new buffers (swap style) fails here, because the original borrower
      // would then read the copy's storage.
      const onLender = await borrower.createGenerationContext({
        chunkLength: 1,
        borrow: lenderContext,
      });
      extra.push(onLender);
      const onFork = await borrower.createGenerationContext({ chunkLength: 1, borrow: forked });
      extra.push(onFork);
      const live = await draftWords(borrower, borrowed, 303);
      assertEquals(
        live,
        await draftWords(borrower, onLender, 303),
        "the original borrower does not read the original slots",
      );
      assertNotEquals(
        live,
        await draftWords(borrower, onFork, 303),
        "the original borrower read the copy's slots (a mix-up of the re-pointing kind)",
      );

      // Lifetime: the copy has none of the original's borrowers (the copy can be returned even while the
      // original borrower is alive). The original is still rejected while it has a borrower, as before.
      await onFork.dispose();
      await forked.dispose();
      const refused = await assertRejects(() => lenderContext.dispose(), ExecutionError);
      assert(refused.message.includes("2 contexts are borrowing"), refused.message);
    } finally {
      for (const context of extra.reverse()) await context.dispose();
      await borrowed.dispose();
      await lenderContext.dispose();
      await borrower.dispose();
      await lender.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "spec checks of the borrowing context (borrow and external go together, chunkLength 1, inherited bindings, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const lender = await createSessionFromContainer(
      gpu,
      await openGraphModel(lenderGraph()),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(borrowerGraph()),
      "model",
    );
    const lenderContext = await lender.createGenerationContext({
      chunkLength: 4,
      bindings: { C: 16 },
    });
    try {
      // (1) external without borrow (allocating the borrowed slots itself would read an empty past)
      const noBorrow = await assertRejects(
        () => borrower.createGenerationContext({ chunkLength: 1 }),
        ExecutionError,
      );
      assert(noBorrow.message.includes("external state slots"), noBorrow.message);

      // (2) borrow on a graph with slots of its own (grabs slots from the lender that nobody reads)
      const ownSlots = await assertRejects(
        () => lender.createGenerationContext({ chunkLength: 4, borrow: lenderContext }),
        ExecutionError,
      );
      assert(ownSlots.message.includes("slots are all external"), ownSlots.message);

      // (3) chunkLength is exactly 1 (the borrower's execution shape is a single decode)
      const chunk = await assertRejects(
        () => borrower.createGenerationContext({ chunkLength: 2, borrow: lenderContext }),
        ExecutionError,
      );
      assert(chunk.message.includes("chunkLength"), chunk.message);

      // (4) bindings are inherited (passing them would make a second binding point)
      const bindings = await assertRejects(
        () =>
          borrower.createGenerationContext({
            chunkLength: 1,
            borrow: lenderContext,
            bindings: { C: 16 },
          }),
        ExecutionError,
      );
      assert(bindings.message.includes("inherited"), bindings.message);

      // (5) chunkBuckets cannot be declared (`chunkLength = 1`, so the existing bucket check rejects it)
      const buckets = await assertRejects(
        () =>
          borrower.createGenerationContext({
            chunkLength: 1,
            borrow: lenderContext,
            chunkBuckets: [2],
          }),
        ExecutionError,
      );
      assert(buckets.message.includes("chunkBuckets"), buckets.message);

      // (6) a borrower whose window disagrees with the lender's (breaks the same-formula rule for reads and writes — rows outside the window are read as past)
      const wideWindow = borrowerGraph();
      wideWindow.nodes[0].attrs = { scale: SCALE, window: WINDOW + 2, readonly: true };
      const wide = await createSessionFromContainer(gpu, await openGraphModel(wideWindow), "model");
      try {
        const mismatch = await assertRejects(
          () => wide.createGenerationContext({ chunkLength: 1, borrow: lenderContext }),
          ExecutionError,
        );
        assert(mismatch.message.includes("disagrees with the lender"), mismatch.message);
      } finally {
        await wide.dispose();
      }

      // Normal case: inherits the lender's bindings (C=16) and copies the slack too.
      const borrowed = await borrower.createGenerationContext({
        chunkLength: 1,
        borrow: lenderContext,
      });
      assertEquals(borrowed.chunkLength, 1);
      assertEquals(borrowed.chunkBuckets, []);
      assertEquals(
        borrowed.slidingSlack,
        lenderContext.slidingSlack,
        "the slack is a copy of the lender's",
      );
      assertEquals(borrowed.slidingSlack, 16 - WINDOW);
      await borrowed.dispose();
    } finally {
      await lenderContext.dispose();
      await borrower.dispose();
      await lender.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "rewriting chunkLength across the allocation await does not reach the borrowing context (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const lender = await createSessionFromContainer(
      gpu,
      await openGraphModel(lenderGraph()),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(borrowerGraph()),
      "model",
    );
    const lenderContext = await lender.createGenerationContext({
      chunkLength: 4,
      bindings: { C: 16 },
    });
    try {
      // Between the "chunkLength is exactly 1" check and the hand-off to the constructor there is the
      // await of the logical-length allocation. A form that re-reads `spec` would build a borrower whose
      // check passed with 1 but whose execution shape is M=4 (readonly attention has M fixed at 1 — it is
      // the counterpart of the shape check of ADR 0096 stage 2 §1.2, so this form breaks without any
      // exception).
      const spec = { chunkLength: 1, borrow: lenderContext };
      const pending = borrower.createGenerationContext(spec);
      spec.chunkLength = 4;
      const borrowed = await pending;
      try {
        assertEquals(borrowed.chunkLength, 1, "stays at the 1 that passed the check");
        assertEquals(
          [...internals(borrowed).allowedRows],
          [1],
          "the only allowed physical shape is a single decode",
        );
      } finally {
        await borrowed.dispose();
      }
    } finally {
      await lenderContext.dispose();
      await borrower.dispose();
      await lender.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the external slot's capacity is part of the plan key (two bound to lenders of different capacities get different keys, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const lender = await createSessionFromContainer(
      gpu,
      await openGraphModel(lenderGraph()),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(borrowerGraph()),
      "model",
    );
    const small = await lender.createGenerationContext({ chunkLength: 4, bindings: { C: 16 } });
    const large = await lender.createGenerationContext({ chunkLength: 4, bindings: { C: 32 } });
    const borrowedSmall = await borrower.createGenerationContext({
      chunkLength: 1,
      borrow: small,
    });
    const borrowedLarge = await borrower.createGenerationContext({
      chunkLength: 1,
      borrow: large,
    });
    const draft = (context: GenerationContext): Promise<unknown> =>
      borrower.run(
        { q: tensor([1, HEADS, 1, DEPTH], seeded(HEADS * DEPTH, QUERY)) },
        {},
        { context, queryLength: 1 },
      );
    try {
      await draft(borrowedSmall);
      assertEquals(
        borrower.diagnostics().lastRunPrepared?.hit,
        false,
        "the first is a deriving run",
      );
      await draft(borrowedSmall);
      assertEquals(
        borrower.diagnostics().lastRunPrepared?.hit,
        true,
        "the same capacity gives the same key",
      );
      await draft(borrowedLarge);
      const stats = borrower.diagnostics().lastRunPrepared;
      assertEquals(
        stats?.hit,
        false,
        "a different capacity gives a different key (the external capacity is not part of the key)",
      );
      assertEquals(stats?.cachedPlans, 2, "one plan is cached per capacity");
      await draft(borrowedLarge);
      assertEquals(borrower.diagnostics().lastRunPrepared?.hit, true);
    } finally {
      await borrowedSmall.dispose();
      await borrowedLarge.dispose();
      await small.dispose();
      await large.dispose();
      await borrower.dispose();
      await lender.dispose();
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// Shared initializers (§2.2) — the borrower reads the lender's i8 embedding table
// ---------------------------------------------------------------------------

const VOCAB = 8;
const HIDDEN = 4;

/** One i8 embedding table (a different scale per row — a row mix-up shows in the values). */
const embedTable = (): {
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly scale: Float32Array<ArrayBuffer>;
  readonly values: Float32Array<ArrayBuffer>;
} => {
  const quantized: number[] = [];
  for (let row = 0; row < VOCAB; row += 1) {
    for (let col = 0; col < HIDDEN; col += 1) quantized.push(((row * 7 + col * 13) % 251) - 125);
  }
  const bytes = i8BytesFrom(quantized);
  const scale = Float32Array.from({ length: VOCAB }, (_, row) => 0.1 + row * 0.003);
  return { bytes, scale, values: decodeI8(bytes, [VOCAB, HIDDEN], scale, [VOCAB, 1]) };
};

/** The lender (owner of the weights): the smallest graph, holding a single i8 embedding table. */
const weightLenderGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["embedding"] },
  symbols: [],
  inputs: [{ name: "index", dtype: "i32", shape: [VOCAB] }],
  outputs: ["y"],
  initializers: { embed: {} },
  values: {
    embed: { dtype: "f32", shape: [VOCAB, HIDDEN] },
    y: { dtype: "f32", shape: [VOCAB, HIDDEN] },
  },
  nodes: [{ op: "embedding", ins: ["embed", "index"], outs: ["y"], attrs: { padding_idx: -1 } }],
});

/** The borrower (the side that holds no bytes): borrows the same table with a `shared` declaration and reads it. */
const weightBorrowerGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["embedding"] },
  symbols: [],
  inputs: [{ name: "index", dtype: "i32", shape: [VOCAB] }],
  outputs: ["y"],
  initializers: { embed: { shared: true } },
  values: {
    embed: { dtype: "f32", shape: [VOCAB, HIDDEN] },
    y: { dtype: "f32", shape: [VOCAB, HIDDEN] },
  },
  nodes: [{ op: "embedding", ins: ["embed", "index"], outs: ["y"], attrs: { padding_idx: -1 } }],
});

/**
 * Material for the row-axis check (lender): the conv1d weight slot = row axis **0** (`[Cout,Cin,K]`).
 * The values are not looked at (the check looks only at the residency class and the axis), so the
 * scale is the identity and the payload is filled with a constant.
 */
const convLenderGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["conv1d"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [1, 2, 6] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [4, 2, 3] },
    b: { dtype: "f32", shape: [4] },
    y: { dtype: "f32", shape: [1, 4, 4] },
  },
  nodes: [
    {
      op: "conv1d",
      ins: ["x", "w", "b"],
      outs: ["y"],
      attrs: { stride: 1, padding: 0, dilation: 1, groups: 1 },
    },
  ],
});

/** A borrower that consumes the same bytes with conv_transpose1d (`[Cin,Cout,K]` = row axis **1**). */
const convBorrowerGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["conv_transpose1d"] },
  symbols: [],
  inputs: [{ name: "z", dtype: "f32", shape: [1, 4, 6] }],
  outputs: ["u"],
  initializers: { w: { shared: true }, c: {} },
  values: {
    w: { dtype: "f32", shape: [4, 2, 3] },
    c: { dtype: "f32", shape: [2] },
    u: { dtype: "f32", shape: [1, 2, 8] },
  },
  nodes: [
    {
      op: "conv_transpose1d",
      ins: ["z", "w", "c"],
      outs: ["u"],
      attrs: { stride: 1, padding: 0 },
    },
  ],
});

const openConvLenderModel = () =>
  openGraphModel(convLenderGraph(), [
    {
      graph: "model",
      initializer: "w",
      bytes: i8BytesFrom(new Array(24).fill(1)),
      // The rows of the per-channel scale run along the consuming op's channel axis (conv1d = 0); row length 6 = groupSize.
      encoding: {
        codec: "int8-sym",
        rowAxis: 0,
        groupSize: 6,
        scale: { bytes: f32Bytes(Float32Array.from([1, 1, 1, 1])), dtype: "f32" },
      },
    },
    {
      graph: "model",
      initializer: "b",
      bytes: f32Bytes(new Float32Array(4)),
      encoding: { codec: "f32" },
    },
  ]);

const openConvBorrowerModel = () =>
  openGraphModel(convBorrowerGraph(), [
    {
      graph: "model",
      initializer: "c",
      bytes: f32Bytes(new Float32Array(2)),
      encoding: { codec: "f32" },
    },
  ]);

const openWeightLenderModel = (table: ReturnType<typeof embedTable>) =>
  openGraphModel(weightLenderGraph(), [
    {
      graph: "model",
      initializer: "embed",
      bytes: table.bytes,
      // The embedding's channel axis is 0; row length HIDDEN = groupSize (per-channel).
      encoding: {
        codec: "int8-sym",
        rowAxis: 0,
        groupSize: HIDDEN,
        scale: { bytes: f32Bytes(table.scale), dtype: "f32" },
      },
    },
  ]);

test.skipIf(!GPU_AVAILABLE)(
  "shared initializer: the borrower reads the lender's i8 table and the values are bit-identical (real GPU)",
  async () => {
    const table = embedTable();
    const gpu = await acquireGpu();
    const lender = await createSessionFromContainer(
      gpu,
      await openWeightLenderModel(table),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(weightBorrowerGraph()),
      "model",
      { sharedWeights: { embed: lender.exportWeight("embed") } },
    );
    try {
      const index = fill([VOCAB], (i) => VOCAB - 1 - i, "i32");
      const lent = await lender.run({ index });
      const borrowedOut = await borrower.run({ index });
      const a = lent["y"].data as Float32Array<ArrayBuffer>;
      const b = borrowedOut["y"].data as Float32Array<ArrayBuffer>;
      const aBits = new Uint32Array(a.buffer, a.byteOffset, a.length);
      const bBits = new Uint32Array(b.buffer, b.byteOffset, b.length);
      for (let i = 0; i < aBits.length; i += 1) {
        assertEquals(bBits[i], aBits[i], `element ${i}: lender ${a[i]} / borrower ${b[i]}`);
      }
      // It also matches the expected values themselves (the row gather of decodeI8) = the whole table is read.
      for (let row = 0; row < VOCAB; row += 1) {
        for (let col = 0; col < HIDDEN; col += 1) {
          const source = (VOCAB - 1 - row) * HIDDEN + col;
          assertEquals(b[row * HIDDEN + col], table.values[source], `row ${row} column ${col}`);
        }
      }
      // The borrower keeps not a single byte resident (only the lender holds bytes).
      assertEquals(borrower.diagnostics().storage.residentCompressedBytes, 0);
      assertEquals(borrower.diagnostics().storage.hostExpandedBytes, 0);

      // Disposing the lender while the borrower is alive is rejected.
      const refused = await assertRejects(() => lender.dispose(), ExecutionError);
      assert(refused.message.includes("lent out"), refused.message);
      // A rejected dispose does not break the lender.
      await lender.run({ index });
    } finally {
      await borrower.dispose();
      // Once the borrower has gone, it succeeds.
      await lender.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "shared initializer checks: a missing or surplus entry, or a mismatch of shape, residency class or row axis fails loudly (real GPU)",
  async () => {
    const table = embedTable();
    const gpu = await acquireGpu();
    const lender = await createSessionFromContainer(
      gpu,
      await openWeightLenderModel(table),
      "model",
    );
    try {
      const shared = lender.exportWeight("embed");
      const build = async (
        graph: DeclarationJson,
        weights?: Record<string, typeof shared>,
      ): Promise<Session> =>
        await createSessionFromContainer(
          gpu,
          await openGraphModel(graph),
          "model",
          weights === undefined ? {} : { sharedWeights: weights },
        );

      // (1) forgotten (0 entries for 1 declaration)
      const missing = await assertRejects(() => build(weightBorrowerGraph()), ExecutionError);
      assert(missing.message.includes("missing [embed]"), missing.message);

      // (2) surplus (a name not in the declarations)
      const surplus = await assertRejects(
        () => build(weightBorrowerGraph(), { embed: shared, unknown: shared }),
        ExecutionError,
      );
      assert(surplus.message.includes("extra [unknown]"), surplus.message);

      // (3) mismatch of the declared shape
      const wrongShape = weightBorrowerGraph();
      wrongShape.values["embed"].shape = [HIDDEN, VOCAB];
      wrongShape.values["y"].shape = [VOCAB, VOCAB];
      const shapeError = await assertRejects(
        () => build(wrongShape, { embed: shared }),
        ExecutionError,
      );
      assert(shapeError.message.includes("declared shape"), shapeError.message);

      // (4) mismatch of the row axis. The borrower declares no storage (the expected residency class is
      // derived from the lender's codec), so this axis carries any disagreement in how the bytes are read:
      // the axis of the per-channel scale is set by the **consuming op**, and for the lender's conv1d
      // (axis 0) and the borrower's conv_transpose1d (axis 1) the scales of different rows land on the
      // same bytes (a silently wrong value with not a single exception).
      const axisLender = await createSessionFromContainer(
        gpu,
        await openConvLenderModel(),
        "model",
      );
      try {
        const axisError = await assertRejects(
          async () =>
            await createSessionFromContainer(gpu, await openConvBorrowerModel(), "model", {
              sharedWeights: { w: axisLender.exportWeight("w") },
            }),
          ExecutionError,
        );
        assert(axisError.message.includes("row axis"), axisError.message);
      } finally {
        await axisLender.dispose();
      }

      // (5) mismatch of the consuming residency class. A borrower that consumes the same table **outside a
      // weight slot** (as an elementwise operand) is not eligible for compressed residency = residency class
      // `expanded`, which cannot pair with the lender's i8 class. Letting it through would be exactly the
      // silently wrong value of "reading packed i8 bytes as f32".
      const wrongSeat: DeclarationJson = {
        format: "karume-ir",
        version: 2,
        requires: { ops: ["add"] },
        symbols: [],
        inputs: [{ name: "x", dtype: "f32", shape: [VOCAB, HIDDEN] }],
        outputs: ["y"],
        initializers: { embed: { shared: true } },
        values: {
          embed: { dtype: "f32", shape: [VOCAB, HIDDEN] },
          y: { dtype: "f32", shape: [VOCAB, HIDDEN] },
        },
        nodes: [{ op: "add", ins: ["x", "embed"], outs: ["y"], attrs: {} }],
      };
      const seatError = await assertRejects(
        () => build(wrongSeat, { embed: shared }),
        ExecutionError,
      );
      assert(
        seatError.message.includes(
          "the consuming residency class is not compatible with the lender's",
        ),
        seatError.message,
      );

      // (6) Re-exporting a borrowed weight is rejected (there are no borrowing chains).
      const chained = await build(weightBorrowerGraph(), { embed: shared });
      try {
        assertThrows(
          () => chained.exportWeight("embed"),
          ExecutionError,
          "borrowed weights are not re-exported",
        );
      } finally {
        await chained.dispose();
      }
    } finally {
      await lender.dispose();
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// Shared initializer checks: semantic dtype (§2.2)
// ---------------------------------------------------------------------------

/** The element count of the material for the dtype check (values are not looked at — the check looks only at declarations). */
const DTYPE_LENGTH = 4;

/** The lender: holds one initializer `w` whose semantic dtype is `dtype`, casts it to f32 and returns it. */
const dtypeLenderGraph = (dtype: "f32" | "i32"): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["cast"] },
  symbols: [],
  inputs: [],
  outputs: ["y"],
  initializers: { w: {} },
  values: {
    w: { dtype, shape: [DTYPE_LENGTH] },
    y: { dtype: "f32", shape: [DTYPE_LENGTH] },
  },
  nodes: [{ op: "cast", ins: ["w"], outs: ["y"], attrs: { to: "f32" } }],
});

const openDtypeLenderModel = (dtype: "f32" | "i32") =>
  openGraphModel(dtypeLenderGraph(dtype), [
    {
      graph: "model",
      initializer: "w",
      bytes:
        dtype === "f32"
          ? f32Bytes([1, 2, -3, 0.5])
          : new Uint8Array(Int32Array.from([1, 2, -3, 5]).buffer),
      encoding: { codec: dtype },
    },
  ]);

/**
 * The borrower: borrows `w` with `shared` (declaring its semantic dtype as `dtype`) and also holds
 * one f32 initializer `bias` of its own. `bias` is the material that guarantees "if the check sat
 * after the weight upload, the borrower side would always make a GPU allocation" (so that the
 * observation of zero allocations is not vacuous).
 *
 * The IR parser rejects bool as outside the initializer vocabulary, so it is built by rewriting a
 * parsed declaration — the public `openMemoryContainer` takes an `IrDeclaration` as a typed value
 * and does not go through the parser.
 */
const dtypeBorrowerContainer = (dtype: "f32" | "i32" | "bool") => {
  const parsed = parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["cast", "add"] },
    symbols: [],
    inputs: [],
    outputs: ["y"],
    initializers: { w: { shared: true }, bias: {} },
    values: {
      w: { dtype: dtype === "bool" ? "f32" : dtype, shape: [DTYPE_LENGTH] },
      bias: { dtype: "f32", shape: [DTYPE_LENGTH] },
      c: { dtype: "f32", shape: [DTYPE_LENGTH] },
      y: { dtype: "f32", shape: [DTYPE_LENGTH] },
    },
    nodes: [
      { op: "cast", ins: ["w"], outs: ["c"], attrs: { to: "f32" } },
      { op: "add", ins: ["c", "bias"], outs: ["y"], attrs: {} },
    ],
  });
  const declaration: IrDeclaration =
    dtype === "bool"
      ? { ...parsed, values: { ...parsed.values, w: { ...parsed.values["w"], dtype: "bool" } } }
      : parsed;
  return openMemoryContainer({
    graphs: { model: declaration },
    tensors: {
      model: {
        bias: { bytes: f32Bytes(new Float32Array(DTYPE_LENGTH)), encoding: { codec: "f32" } },
      },
    },
  });
};

/** Counts the `createBuffer` calls the device received during `fn` and returns them with `fn`'s result (restores the original afterwards). */
const countCreateBuffer = async <T>(
  gpu: GpuContext,
  fn: () => Promise<T>,
): Promise<{ allocated: number; result: T }> => {
  const create = gpu.device.createBuffer.bind(gpu.device);
  let allocated = 0;
  gpu.device.createBuffer = (descriptor): GPUBuffer => {
    allocated += 1;
    return create(descriptor);
  };
  try {
    const result = await fn();
    return { allocated, result };
  } finally {
    gpu.device.createBuffer = create;
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "shared initializer checks: a semantic dtype mismatch between lender and borrower fails loudly before the borrower's GPU allocation (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    // The lenders are built inside the try too; only those that were built are pushed here and returned
    // in finally — a failure to build the second does not skip cleaning up the first and the device.
    const lenders: Session[] = [];
    let disposed: PromiseSettledResult<void>[] = [];
    try {
      const f32Lender = await createSessionFromContainer(
        gpu,
        await openDtypeLenderModel("f32"),
        "model",
      );
      lenders.push(f32Lender);
      const i32Lender = await createSessionFromContainer(
        gpu,
        await openDtypeLenderModel("i32"),
        "model",
      );
      lenders.push(i32Lender);
      const build = (lender: Session, dtype: "f32" | "i32" | "bool"): Promise<Session> =>
        createSessionFromContainer(gpu, dtypeBorrowerContainer(dtype), "model", {
          sharedWeights: { w: lender.exportWeight("w") },
        });
      const rejectsBeforeAllocation = async (
        lender: Session,
        dtype: "f32" | "i32" | "bool",
        lenderDtype: string,
      ): Promise<void> => {
        const { allocated, result: outcome } = await countCreateBuffer(gpu, () =>
          build(lender, dtype).then(
            (borrower) => ({ borrower }),
            (error: unknown) => ({ error }),
          ),
        );
        if ("borrower" in outcome) {
          // Return the borrower first — while it is held, the lender dispose in finally would be rejected as
          // "on loan" and the real failure (the check not taking effect) would hide behind that rejection.
          // fail sits in finally so that, even if the borrower's own dispose is rejected, the message of the
          // real failure surfaces rather than that rejection.
          try {
            await outcome.borrower.dispose();
          } finally {
            fail(
              `the borrower (${dtype}) was built with the lender's (${lenderDtype}) weights — the semantic dtype check does not take effect`,
            );
          }
        }
        assertInstanceOf(outcome.error, ExecutionError);
        const message = outcome.error.message;
        assert(message.includes("semantic dtype"), message);
        assert(message.includes(`'${dtype}'`) && message.includes(`'${lenderDtype}'`), message);
        // This checks a stronger condition than the spec (before the borrower's weight upload): 0 allocations
        // over the whole build. If it turns red in the future, decide by looking at what the allocations
        // before the check are (weights or something else).
        assertEquals(
          allocated,
          0,
          "the check rejects before uploading a single byte of the borrower's weights",
        );
      };

      // Control: when the dtypes match, the build succeeds and the borrower allocates its own `bias` (= the
      // observation of zero allocations is not vacuous).
      const { allocated: matchedAllocations, result: matched } = await countCreateBuffer(gpu, () =>
        build(f32Lender, "f32"),
      );
      await matched.dispose();
      assert(matchedAllocations > 0, `allocations of the matching borrower ${matchedAllocations}`);

      // (1) lender f32, borrower i32 — letting it through would read the same 4 bytes as an integer (1.0 -> 1065353216).
      await rejectsBeforeAllocation(f32Lender, "i32", "f32");
      // (2) lender i32, borrower f32 — the same mix-up the other way round.
      await rejectsBeforeAllocation(i32Lender, "f32", "i32");
      // (3) borrower bool — the residency format (raw) is the same, so the residency-class check does not tell them apart.
      await rejectsBeforeAllocation(f32Lender, "bool", "f32");
    } finally {
      // A rejected dispose of one does not skip the rest of the cleanup (the other lender and the device).
      disposed = await Promise.allSettled(lenders.map((lender) => lender.dispose()));
      gpu.destroy();
    }
    // A dispose rejection is raised here only when the body passed (throwing in finally would overwrite the body's failure).
    for (const settled of disposed) {
      if (settled.status === "rejected") throw settled.reason;
    }
  },
);
