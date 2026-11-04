// **Bit-identity check** of GQA (exact-division broadcast — ADR 0067 decisions 1–3).
//
// For the same random inputs,
//
//   ① the GQA-form `attention` that takes `k` / `v` as `[B,Hkv,N,D]` as is
//   ② the conventional-form `attention` with kv-heads **materialized** to H on the host (repeat_kv)
//
// are run on **the same real GPU** and the outputs are compared as **f32 bits**. The GQA variant
// differs only in one line of the K / V batch base and one uniform word (ADR 0067 decision 2), so
// neither the shared tile fill nor the K reduction order moves = bit-for-bit equality **is the
// expected value itself**. With allclose, a mixed-up head mapping hides inside the tolerance.
//
// MUST: do not make it tautological. ① and ② run **separate graphs** (different input shapes) in
// separate Sessions. The materialization of ② is a naive host-side copy, written independently of
// the kernel's mapping formula (`wid.z / r`).
// MUST: besides equality, the same test sees **fault injection produce a mismatch** (if it also
// matches a materialization with a mixed-up head mapping, this check sees nothing).
// MUST: B / H / Hkv / M / N / D run over several shapes (remainders, v4, spanning 2 row tiles). The
// kernel folds B·H into one batch axis, so B=1 alone does not verify the identity `b·Hkv + h/r`.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { attentionPvKey, attentionQkKey } from "../src/kernels/attention.ts";
import { gemmUsesVec4 } from "../src/kernels/gemm.ts";
import { attentionScoreUsesF16 } from "../src/kernels/score-storage.ts";
import {
  createSessionFromContainer,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { GRAPH_NAME, openGraphModel, singleOpDeclaration } from "./helpers/model-fixture.ts";
import { deviceGeometryProfile } from "./helpers/geometry.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

/**
 * The half scale (`√scale_factor` of torch's math decomposition). The value derived from D per the
 * contract.
 */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

/** Deterministic data sequences (no random numbers — failures would not reproduce). */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

type F32Tensor = {
  readonly dtype: "f32";
  readonly shape: readonly number[];
  readonly data: Float32Array<ArrayBuffer>;
};

const f32 = (shape: readonly number[], generator: (index: number) => number): F32Tensor => {
  const count = shape.reduce((total, dim) => total * dim, 1);
  const data = new Float32Array(count);
  for (let index = 0; index < count; index += 1) data[index] = generator(index);
  return { dtype: "f32", shape, data };
};

/** The f32 bits (a form in which a 1-ulp difference and `-0.0` are visible). */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

type GqaShape = {
  readonly name: string;
  readonly b: number;
  /** Head count H of q. */
  readonly h: number;
  /** Head count Hkv of k / v (`h % kv == 0` — r = h / kv). */
  readonly kv: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

const SHAPES: readonly GqaShape[] = [
  // r=1 (control — a shape where the GQA bit is not set; keys and generated code stay as before)
  { name: "r1 B2 H4 M5 N7 D4", b: 2, h: 4, kv: 4, m: 5, n: 7, d: 4 },
  // r=2 with remainders on every tile axis (D%4 ≠ 0 drops to the scalar variant)
  { name: "r2 B2 H4 M5 N7 D6", b: 2, h: 4, kv: 2, m: 5, n: 7, d: 6 },
  // r=8 MQA (8:1 of Gemma 4 E2B — Hkv=1 follows the same exact-division formula). ① and ③ both on
  // the v4 path
  { name: "r8 B2 H8 M3 N8 D8", b: 2, h: 8, kv: 1, m: 3, n: 8, d: 8 },
  // r=4 × B=3 spans 2 row tiles (a shape where the b term of `b·Hkv + h/r` matters)
  { name: "r4 B3 H8 M68 N20 D12", b: 3, h: 8, kv: 2, m: 68, n: 20, d: 12 },
  // A shrunk version of plain GQA 16:2 (MiniCPM5-1B). M / N both multiples of 64 = the real model's
  // path
  { name: "r8 B1 H16 M64 N64 D64", b: 1, h: 16, kv: 2, m: 64, n: 64, d: 64 },
];

/**
 * The "shape spanning r=4" used for fault injection (data differs per head, so a mapping error
 * shows in the values).
 */
const INJECT_SHAPE: GqaShape = SHAPES[3];

/**
 * GQA shapes spanning the **per-stage eligibility** of i8a8 (①QK needs `D % 4`, ③PV needs `N % 4` —
 * src/runtime/recipe-builder.ts). The rejection of GQA × i8a8 is decided by **the requested mode**
 * (ADR 0067 decision 3), so whether one stage or both are ineligible, it fails with **the same
 * single error**.
 *
 * MUST: do not test only eligible shapes (SHAPES[2] has D=8 / N=8, eligible for both ① and ③) — a
 * regression that misuses per-stage eligibility as the rejection condition
 * (`gqa && qkI8a8 && pvI8a8` and the like) shows up as **only the ineligible shapes silently
 * degrading to f32 / mixed**, so a check that only touches eligible shapes stays green.
 */
const I8A8_INELIGIBLE: readonly GqaShape[] = [
  { name: "r2 D%4≠0 (①QK ineligible, ③PV eligible)", b: 2, h: 4, kv: 2, m: 5, n: 8, d: 6 },
  { name: "r2 N%4≠0 (①QK eligible, ③PV ineligible)", b: 2, h: 4, kv: 2, m: 5, n: 7, d: 4 },
  { name: "r2 D%4≠0 and N%4≠0 (both ①③ ineligible)", b: 2, h: 4, kv: 2, m: 5, n: 7, d: 6 },
];

/**
 * The **invariant part** of the rejection reason (only `${where}` and the display of `H / Hkv / r`
 * change per shape).
 */
const rejectionReason = (message: string, where: string): string => {
  const at = message.indexOf("× i8a8");
  assert(at >= 0, `${where}: the wording of the GQA × i8a8 rejection changed: ${message}`);
  return message.slice(at);
};

/**
 * **Materializes** the kv-heads to H (repeat_kv).
 *
 * `pick` is the head → kv-head mapping; the default ({@link BLOCK_REPEAT}) is the block repeat
 * `⌊h/r⌋`. Fault injection plugs another mapping in here (the point is to build it on the host,
 * independently of the kernel's mapping formula).
 */
const materializeKv = (
  source: F32Tensor,
  heads: number,
  pick: (head: number, kvHeads: number, repeat: number) => number,
): F32Tensor => {
  const [batches, kvHeads, cols, depth] = source.shape;
  const repeat = heads / kvHeads;
  const block = cols * depth;
  const data = new Float32Array(batches * heads * block);
  for (let batch = 0; batch < batches; batch += 1) {
    for (let head = 0; head < heads; head += 1) {
      const kvHead = pick(head, kvHeads, repeat);
      const from = (batch * kvHeads + kvHead) * block;
      data.set(source.data.subarray(from, from + block), (batch * heads + head) * block);
    }
  }
  return { dtype: "f32", shape: [batches, heads, cols, depth], data };
};

/**
 * The mapping per the contract (integer division `h/r` — the host-side copy of the identity of ADR
 * 0067 decision 2).
 */
const BLOCK_REPEAT = (head: number, _kvHeads: number, repeat: number): number =>
  Math.floor(head / repeat);

/**
 * The measured band mask (an additive term of `[1,1,M,N]` — an axis independent of GQA).
 *
 * MUST: **every row has at least one unmasked column** (if a whole row is masked, amax is −inf,
 * `exp(−inf − (−inf))` becomes NaN, both sides produce the same NaN and the bit comparison sees
 * nothing).
 */
const bandMask = (m: number, n: number, width: number, blocked: number): F32Tensor => {
  const data = new Float32Array(m * n);
  for (let row = 0; row < m; row += 1) {
    const center = Math.floor((row * n) / m);
    for (let col = 0; col < n; col += 1) {
      data[row * n + col] = Math.abs(col - center) <= width ? 0 : blocked;
    }
  }
  return { dtype: "f32", shape: [1, 1, m, n], data };
};

type RunResult = {
  readonly output: Tensor;
  /** Pipeline keys of the last run (census — independent of timing). */
  readonly keys: readonly string[];
};

/**
 * Runs a 1-node `attention`. If the shape of `kv` differs from `q` it is the GQA form, if equal the
 * conventional form (both go through **the same single path** — only the shapes differ).
 */
const runAttention = async (
  gpu: GpuContext,
  q: F32Tensor,
  k: F32Tensor,
  v: F32Tensor,
  mask?: F32Tensor,
  options: SessionOptions = {},
): Promise<RunResult> => {
  const out = [q.shape[0], q.shape[1], q.shape[2], q.shape[3]];
  const shapes =
    mask === undefined ? [q.shape, k.shape, v.shape] : [q.shape, k.shape, v.shape, mask.shape];
  const graph = singleOpDeclaration("attention", shapes, [out], {
    attrs: { scale: halfScale(q.shape[3]) },
  });
  const inputs: Record<string, F32Tensor> = { x0: q, x1: k, x2: v };
  if (mask !== undefined) inputs["x3"] = mask;
  const session = await createSessionFromContainer(
    gpu,
    await openGraphModel(graph),
    GRAPH_NAME,
    options,
  );
  try {
    const output = (await session.run(inputs))["y"];
    const census = requireCensus(session.diagnostics().lastRunPipelines, "attention");
    return { output, keys: census.map((row) => row.key) };
  } finally {
    await session.dispose();
  }
};

/**
 * The 3 inputs of the GQA form and the materialized form (swapping `pick` makes it fault
 * injection).
 */
const inputsFor = (
  shape: GqaShape,
  pick: (head: number, kvHeads: number, repeat: number) => number = BLOCK_REPEAT,
): {
  readonly q: F32Tensor;
  readonly k: F32Tensor;
  readonly v: F32Tensor;
  readonly kFull: F32Tensor;
  readonly vFull: F32Tensor;
} => {
  const { b, h, kv, m, n, d } = shape;
  const q = f32([b, h, m, d], QUERY);
  const k = f32([b, kv, n, d], KEY);
  const v = f32([b, kv, n, d], VALUE);
  return {
    q,
    k,
    v,
    kFull: materializeKv(k, h, pick),
    vFull: materializeKv(v, h, pick),
  };
};

/** Bit mismatches (only the first 4 are collected). */
const mismatches = (left: Tensor, right: Tensor): readonly number[] => {
  const a = bits(left);
  const c = bits(right);
  const found: number[] = [];
  for (let i = 0; i < a.length && found.length < 4; i += 1) {
    if (a[i] !== c[i]) found.push(i);
  }
  return found;
};

const assertBitEqual = (fused: Tensor, split: Tensor, where: string): void => {
  assertEquals(fused.shape, split.shape, where);
  assertEquals(
    mismatches(fused, split),
    [],
    `${where}: bits differ from the materialized version (first mismatches: ${mismatches(
      fused,
      split,
    )
      .map((i) => `[${i}] ${fused.data[i]} vs ${split.data[i]}`)
      .join(" / ")})`,
  );
  // Tautology guard: if the output is constant, "equal" verifies nothing
  assert(
    new Set(bits(fused)).size > 1,
    `${where}: the output is constant (bit equality is a tautology)`,
  );
};

test.skipIf(!GPU_AVAILABLE)(
  "GQA-form attention output is **bit-for-bit equal** to the repeat_kv materialized version (r ∈ {1,2,4,8}, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of SHAPES) {
        const { q, k, v, kFull, vFull } = inputsFor(shape);
        const gqa = await runAttention(gpu, q, k, v);
        const split = await runAttention(gpu, q, kFull, vFull);
        assertBitEqual(gqa.output, split.output, shape.name);
        // The output has q's H (pin on the shape side too that it has the same shape as the
        // materialized version)
        assertEquals(gqa.output.shape, [shape.b, shape.h, shape.m, shape.d], shape.name);
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * **Orthogonality** to mask / s16 (GQA is an axis of the K / V base only). The mask stays
 * `[1,1,M,N]` regardless of Hkv, and the f16 storage of S only switches writer and reader together
 * without touching the mapping.
 */
test.skipIf(!GPU_AVAILABLE)(
  "GQA stays bit-identical orthogonally to an additive mask and f16 storage of S (s16) (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // With a mask (r=4, spanning 2 row tiles)
      const masked = SHAPES[3];
      {
        const { q, k, v, kFull, vFull } = inputsFor(masked);
        const mask = bandMask(masked.m, masked.n, 2, -3.4028234663852886e38);
        const gqa = await runAttention(gpu, q, k, v, mask);
        const split = await runAttention(gpu, q, kFull, vFull, mask);
        assertBitEqual(gqa.output, split.output, `${masked.name} + mask`);
        // Tautology guard: the mask affects the result (values differ from no mask)
        const plain = await runAttention(gpu, q, k, v);
        assert(
          bits(gqa.output).some((word, i) => word !== bits(plain.output)[i]),
          `${masked.name}: the outputs with and without the mask are identical (the mask has no effect)`,
        );
        assert(
          gqa.output.data.every((value) => Number.isFinite(value)),
          `${masked.name}: non-finite values in the output (a whole row is masked)`,
        );
      }
      // s16 (f16 storage of S — run only after the predicate confirms the shape is eligible)
      const scored = SHAPES[2];
      {
        assertEquals(
          attentionScoreUsesF16(scored.d, scored.n),
          true,
          `${scored.name}: should be an s16-eligible shape`,
        );
        const { q, k, v, kFull, vFull } = inputsFor(scored);
        const options: SessionOptions = { attentionScoreStorage: "f16" };
        const gqa = await runAttention(gpu, q, k, v, undefined, options);
        const split = await runAttention(gpu, q, kFull, vFull, undefined, options);
        assertBitEqual(gqa.output, split.output, `${scored.name} + s16`);
        // Tautology guard: s16 has an effect (values move from f32 storage = one more rounding
        // step)
        const f32Score = await runAttention(gpu, q, k, v);
        assert(
          bits(gqa.output).some((word, i) => word !== bits(f32Score.output)[i]),
          `${scored.name}: the s16 and f32-storage outputs are identical (the opt-in has no effect)`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * **Fault injection** (the acceptance condition of ADR 0067 decision 2). Compares against a
 * materialization with a mixed-up head mapping and confirms **a mismatch shows** — if not, the
 * bit-identity check above sees nothing.
 *
 * ① Wrong `r`: materialize with `⌊h/2⌋ mod Hkv` (the r'=2 mapping). Against the true r=4 the
 *    mapping of heads 0..7 differs: 0,0,1,1,0,0,1,1 vs 0,0,0,0,1,1,1,1.
 * ② A missing mapping on the ③PV side: K is materialized correctly, V is **stuck to kv-head 0**
 *    (the same shape as the values a kernel that maps ①QK but forgets ③PV would produce — S is
 *    right, only V comes from another head).
 */
test.skipIf(!GPU_AVAILABLE)(
  "the GQA bit-identity check detects head mapping errors (wrong r, missing ③PV mapping) (real GPU)",
  async () => {
    const shape = INJECT_SHAPE;
    const gpu = await acquireGpu();
    try {
      const { q, k, v, kFull, vFull } = inputsFor(shape);
      const truth = await runAttention(gpu, q, k, v);
      // Equal to the correct materialization (control for the mismatch side — evidence that the
      // mapping, not the environment, is what is checked)
      assertBitEqual(truth.output, (await runAttention(gpu, q, kFull, vFull)).output, shape.name);

      // ① Wrong r (K / V both materialized with the r'=2 mapping)
      const wrongR = inputsFor(shape, (head, kvHeads) => Math.floor(head / 2) % kvHeads);
      const injectedR = await runAttention(gpu, q, wrongR.kFull, wrongR.vFull);
      assert(
        mismatches(truth.output, injectedR.output).length > 0,
        `${shape.name}: matches the wrong-r (r'=2) materialization (the check does not see the mapping)`,
      );

      // ② Missing mapping on the ③PV side (K correct, V stuck to kv-head 0)
      const stuckV = materializeKv(v, shape.h, () => 0);
      const injectedPv = await runAttention(gpu, q, kFull, stuckV);
      assert(
        mismatches(truth.output, injectedPv.output).length > 0,
        `${shape.name}: matches the missing V mapping (the ③PV side is not checked)`,
      );
      // The opposite side (only K stuck) is detected too — a missing mapping on either ①QK or ③PV
      // is visible
      const stuckK = materializeKv(k, shape.h, () => 0);
      const injectedQk = await runAttention(gpu, q, stuckK, vFull);
      assert(
        mismatches(truth.output, injectedQk.output).length > 0,
        `${shape.name}: matches the missing K mapping (the ①QK side is not checked)`,
      );
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * **census** (ADR 0058 decision 4). With r > 1 the GQA variant's keys **actually ran**; with r = 1
 * no `:gqa` appears at all (the execution-side counterpart of decision 2's "r=1 is
 * byte-identical").
 *
 * MUST: the source is `lastRunPipelines`, which does not depend on timing (runs on machines without
 * timing too). An empty census FAILs unconditionally (guarding with "if entries is empty, go to the
 * next shape" makes every case green without checking anything — `requireCensus` in
 * tests/helpers/pipeline-census.ts).
 */
test.skipIf(!GPU_AVAILABLE)(
  "GQA keys are set only for r > 1 (r=1 keeps the conventional keys, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      for (const shape of SHAPES) {
        const { q, k, v } = inputsFor(shape);
        const { keys } = await runAttention(gpu, q, k, v);
        assert(keys.length > 0, `${shape.name}: the breakdown is empty (the key check is a no-op)`);
        const gqa = shape.h !== shape.kv;
        const attention = keys.filter(
          (key) => key.startsWith("attention_qk") || key.startsWith("attention_pv"),
        );
        assertEquals(attention.length, 2, `${shape.name}: ① and ③ run once each`);
        assertEquals(
          keys.includes(
            attentionQkKey(
              gemmUsesVec4(shape.d, shape.n),
              "f32",
              "f32",
              false,
              gqa,
              false,
              profile.attention.qk,
            ),
          ),
          true,
          `${shape.name}: the expected ①QK key is missing (keys that ran: ${attention.join(" / ")})`,
        );
        assertEquals(
          keys.includes(
            attentionPvKey(
              gemmUsesVec4(shape.n, shape.d),
              "f32",
              "f32",
              gqa,
              false,
              profile.attention.pv,
            ),
          ),
          true,
          `${shape.name}: the expected ③PV key is missing (keys that ran: ${attention.join(" / ")})`,
        );
        // r=1 has no `:gqa` at all / r>1 has it on both ① and ③
        assertEquals(
          attention.filter((key) => key.endsWith(":gqa")).length,
          gqa ? 2 : 0,
          `${shape.name}: number of :gqa keys`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * **GQA × i8a8 fails loudly** (ADR 0067 decision 3 — silently dropping to the f32 path would
 * quietly change performance. i8a8 has 5 head bases and its K / V quantization and allocation
 * assume `B·H`, so a GQA shape would be mixed up).
 *
 * Tautology guard: **the non-GQA version of the same shape (H = Hkv) runs with i8a8** (pins that
 * the cause of the rejection is GQA, not "that shape does not run with i8a8").
 *
 * MUST: check down to the ineligible shapes spanning per-stage eligibility
 * ({@link I8A8_INELIGIBLE}) failing with **the same** error — in an implementation where the
 * rejection is decided by stage eligibility instead of the requested mode, exactly those degrade
 * silently.
 */
test.skipIf(!GPU_AVAILABLE)(
  "GQA × attentionCompute 'a8' fails loudly (regardless of per-stage eligibility; non-GQA of the same shape runs; real GPU)",
  async () => {
    const shape = SHAPES[2];
    const gpu = await acquireGpu();
    try {
      const { q, k, v, kFull, vFull } = inputsFor(shape);
      const options: SessionOptions = { attentionCompute: "a8" };
      const error = await assertRejects(
        () => runAttention(gpu, q, k, v, undefined, options),
        ExecutionError,
        "GQA",
      );
      assert(error.message.includes("ADR 0067"), `the guidance is insufficient: ${error.message}`);
      // The rejection reason on a shape eligible for i8a8 in both ① and ③ (D=8 / N=8). The
      // ineligible shapes below fail with **the identical** reason (only the display of H / Hkv / r
      // changes per shape).
      const reason = rejectionReason(error.message, shape.name);
      for (const ineligible of I8A8_INELIGIBLE) {
        // First pin that the subject really is ineligible (an eligible shape slipping in removes
        // the coverage)
        assert(
          ineligible.d % 4 !== 0 || ineligible.n % 4 !== 0,
          `${ineligible.name}: eligible for i8a8 in both ①③ = not covering an ineligible shape`,
        );
        const inputs = inputsFor(ineligible);
        const rejected = await assertRejects(
          () => runAttention(gpu, inputs.q, inputs.k, inputs.v, undefined, options),
          ExecutionError,
          "GQA",
        );
        assertEquals(
          rejectionReason(rejected.message, ineligible.name),
          reason,
          `${ineligible.name}: rejection reason differs from the eligible shape (per-stage eligibility leaks into the rejection condition)`,
        );
      }
      // Non-GQA (the same values materialized to H = Hkv) runs with i8a8
      const plain = await runAttention(gpu, q, kFull, vFull, undefined, options);
      assertEquals(plain.output.shape, [shape.b, shape.h, shape.m, shape.d], shape.name);
      assert(
        plain.output.data.every((value) => Number.isFinite(value)),
        `${shape.name}: non-finite values in the non-GQA i8a8 output`,
      );
    } finally {
      gpu.destroy();
    }
  },
);
