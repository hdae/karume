import { assert, assertEquals, assertMatch, assertThrows } from "@std/assert";
import { acquireGpu } from "../mod.ts";
import { parseStorageRoles } from "../src/gpu/pipeline-cache.ts";
import { gemmMTileGeometry } from "../src/kernels/gemm.ts";
import { defaultGemmGeometry } from "../src/kernels/gemm-geometry.ts";
import { defaultI8a8Geometry } from "../src/kernels/i8a8-geometry.ts";
import { linearWgsl } from "../src/kernels/linear.ts";
import { calibrateReps, TARGET_PASS_MS } from "../src/tune/measurement.ts";
import {
  type AttentionCase,
  type BmmCase,
  type Conv2dCase,
  type LinearCase,
  type MatmulCase,
  SWEEP_CASES,
  type SweepCase,
} from "../src/tune/cases.ts";
import {
  conv2dCandidate,
  gemmCandidate,
  type GeometryCandidate,
  i8a8Candidate,
} from "../src/tune/geometries.ts";
import {
  assertBindingRoles,
  casePlan,
  type CaseResult,
  createSweepContext,
  destroySweepContext,
  joinChunkDigests,
  resourceWordStream,
  SWEEP_MAX_REPS,
  sweepCase,
} from "../src/tune/harness.ts";

// 形状表の DiT 形は 1 本 数 ms 〜 数百 ms なので、GPU テストは同じ経路の小さい形で回す
// （M ≥ 513 で既定幾何 128×128 のバケットに入る最小の丸い形）。
const LINEAR: LinearCase = {
  id: "test-linear-m1024-n512-k256",
  op: "linear",
  m: 1024,
  n: 512,
  k: 256,
  censusCount: 1,
  source: "test",
};
const I8A8_LINEAR: LinearCase = {
  ...LINEAR,
  id: "test-i8a8-linear-m1024-n512-k256",
  op: "i8a8-linear",
};

/** 融合 attention の小形（B·H 2・M = N = 128・D 128 — 既定幾何 128×128 の 1 タイル）。 */
const attentionCase = (
  op: AttentionCase["op"],
  stage: AttentionCase["stage"],
): AttentionCase => ({
  id: `test-${op}-${stage}-bh2-m128-n128-d128`,
  op,
  stage,
  batchHeads: 2,
  m: 128,
  n: 128,
  d: 128,
  scale: 0.2973017692565918,
  score: op === "attention" ? "f32" : "f16",
  censusCount: 1,
  source: "test",
});

/** conv2d の小形（Cin = Cout = 64・16²・3×3 — m タイル 64 行の本番幾何）。 */
const CONV2D: Conv2dCase = {
  id: "test-conv2d-c64-16x16",
  op: "conv2d",
  channelsIn: 64,
  channelsOut: 64,
  height: 16,
  width: 16,
  censusCount: 1,
  source: "test",
};

/** 64×64 / 256 スレッド（quick の 2 番目）。 */
const GEMM_64X64 = gemmCandidate({ regM: 4, regN: 4, wgX: 16, wgY: 16 });
const I8A8_64X64 = i8a8Candidate({ regM: 4, regN: 4, wgX: 16, wgY: 16, tileK: 16 });
const CONV2D_64X64 = conv2dCandidate({ regM: 4, regN: 4, wgX: 16, wgY: 16 });

const adapter = navigator.gpu === undefined ? null : await navigator.gpu.requestAdapter();
const timestampQuery = adapter !== null && adapter.features.has("timestamp-query");

/**
 * 2 行（既定 → 別幾何）が測れて出力 digest が一致し、ケースの末尾で既定幾何を測り直した比が
 * 残ったこと。キーは幾何の名前を含む（名前は v4 抜きの綴りなので、キーの `v4` を除いて見る）。
 */
const assertMeasuredPair = (result: CaseResult, defaultName: string, other: string): void => {
  const { rows, summary } = result;
  assertEquals(rows.length, 2);
  const [first, second] = rows;
  assertEquals(first.isDefault, true);
  assertEquals(first.geometry, defaultName);
  assertEquals(second.geometry, other);
  for (const row of rows) {
    assertEquals(row.error, undefined, row.error);
    assert((row.perDispatch ?? 0) > 0, `${row.geometry}: ${row.perDispatch}`);
    assert((row.reps ?? 0) >= 1);
    assertMatch(row.outputSha256 ?? "", /^[0-9a-f]{64}$/);
    const key = (row.key ?? "").replaceAll("v4", "");
    assert(key.includes(row.geometry), `${row.key} に ${row.geometry} が無い`);
  }
  assertEquals(second.identicalToDefault, true);
  assert((second.speedupVsDefault ?? 0) > 0);
  assertEquals(summary.caseId, first.caseId);
  assertEquals(summary.defaultRepeatError, undefined, summary.defaultRepeatError);
  assert((summary.defaultRepeat?.perDispatch ?? 0) > 0);
  assert((summary.defaultRepeat?.driftRatio ?? 0) > 0);
};

/** 実 GPU で 1 ケースを既定 + 1 幾何で回す（rounds 1）。 */
const sweepPair = async (
  target: SweepCase,
  other: GeometryCandidate,
  gpuTiming = timestampQuery,
): Promise<CaseResult> => {
  const gpu = await acquireGpu(gpuTiming ? { gpuTiming: true } : {});
  const context = await createSweepContext(gpu, "deno-raw-tick");
  try {
    assertEquals(context.unit, gpuTiming ? "deno-raw-tick" : "wall");
    return await sweepCase(context, target, [other], { rounds: 1 });
  } finally {
    destroySweepContext(context);
    gpu.destroy();
  }
};

Deno.test("束縛表は WGSL の storage の役割（本数と書き込み先の位置）と突き合わせて通る", () => {
  const roles = parseStorageRoles(linearWgsl("f16", true, "f32", 1024));
  assertBindingRoles(
    { key: "linear", bindings: ["x", "w", "bias", "out"], writes: ["out"] },
    roles,
  );
});

Deno.test("束縛表の書き込み先の位置が WGSL と違えば落ちる（位置束縛の取り違えの検出）", () => {
  const roles = parseStorageRoles(linearWgsl("f16", true, "f32", 1024));
  assertThrows(
    () =>
      assertBindingRoles(
        { key: "linear", bindings: ["x", "w", "out", "bias"], writes: ["out"] },
        roles,
      ),
    Error,
    "書き込み束縛",
  );
});

Deno.test("matmul / bmm の全ケースの束縛表は、既定幾何の WGSL の storage の役割と噛み合う", () => {
  const dense = SWEEP_CASES.filter((target) => target.op === "matmul" || target.op === "bmm");
  assertEquals(new Set(dense.map((target) => target.op)), new Set(["matmul", "bmm"]));
  for (const target of dense) {
    const plan = casePlan(target, 65535, false);
    const launch = plan.launch(plan.defaultCandidate);
    assertBindingRoles(launch, parseStorageRoles(launch.wgsl));
    // 束縛の名前は全て資源表にあり、出力は書き込み先
    const names = new Set(plan.resources.map((spec) => spec.name));
    for (const name of launch.bindings) assert(names.has(name), `${target.id}: ${name}`);
    assert(launch.writes.includes(plan.output), target.id);
  }
});

Deno.test("束縛表の本数が WGSL と違えば落ちる", () => {
  const roles = parseStorageRoles(linearWgsl("f16", true, "f32", 1024));
  assertThrows(
    () =>
      assertBindingRoles({ key: "linear", bindings: ["x", "w", "out"], writes: ["out"] }, roles),
    Error,
    "storage 束縛",
  );
});

Deno.test("1 dispatch 15 µs の小さいケースでも、掃引の反復は opbench の上限 1024 を超えて pass が目標長に届く", () => {
  const nsPerDispatch = 15e3;
  const reps = calibrateReps(nsPerDispatch, TARGET_PASS_MS, SWEEP_MAX_REPS);
  assertEquals(reps, 5334);
  assert(reps > 1024);
  assert(reps * nsPerDispatch >= TARGET_PASS_MS * 1e6);
});

Deno.test({
  name: "実 GPU: f32 linear を既定と 64×64/256 で測ると両方の時間が正で出力が一致する",
  ignore: adapter === null,
  fn: async () => {
    assertMeasuredPair(
      await sweepPair(LINEAR, GEMM_64X64),
      gemmCandidate(defaultGemmGeometry()).name,
      GEMM_64X64.name,
    );
  },
});

Deno.test({
  name: "実 GPU: i8a8 linear を既定と 64×64/256 で測ると両方の時間が正で出力が一致する",
  ignore: adapter === null,
  fn: async () => {
    assertMeasuredPair(
      await sweepPair(I8A8_LINEAR, I8A8_64X64),
      i8a8Candidate(defaultI8a8Geometry("linear")).name,
      I8A8_64X64.name,
    );
  },
});

Deno.test({
  name: "実 GPU: timestamp-query の無い device は壁時計（wall）で同じ規約を回す",
  ignore: adapter === null,
  fn: async () => {
    const result = await sweepPair(LINEAR, GEMM_64X64, false);
    assertMeasuredPair(result, gemmCandidate(defaultGemmGeometry()).name, GEMM_64X64.name);
    // 単位が wall なら round の値は壁時計そのもの
    for (const row of result.rows) assertEquals(row.rounds, row.wallRounds);
  },
});

Deno.test({
  name: "実 GPU: f32 融合 attention の ①QK を既定と 64×64/256 で測ると出力が一致する",
  ignore: adapter === null,
  fn: async () => {
    assertMeasuredPair(
      await sweepPair(attentionCase("attention", "qk"), GEMM_64X64),
      gemmCandidate(defaultGemmGeometry()).name,
      GEMM_64X64.name,
    );
  },
});

Deno.test({
  name:
    "実 GPU: f32 融合 attention の ③PV（前段 ①QK → ② 行統計）を既定と 64×64/256 で測ると出力が一致する",
  ignore: adapter === null,
  fn: async () => {
    assertMeasuredPair(
      await sweepPair(attentionCase("attention", "pv"), GEMM_64X64),
      gemmCandidate(defaultGemmGeometry()).name,
      GEMM_64X64.name,
    );
  },
});

Deno.test({
  name: "実 GPU: i8a8 融合 attention の ①QK を既定と 64×64/256 で測ると出力が一致する",
  ignore: adapter === null,
  fn: async () => {
    assertMeasuredPair(
      await sweepPair(attentionCase("i8a8-attention", "qk"), I8A8_64X64),
      i8a8Candidate(defaultI8a8Geometry("attention_qk")).name,
      I8A8_64X64.name,
    );
  },
});

Deno.test({
  name: "実 GPU: i8a8 融合 attention の ③PV を既定と 64×64/256 で測ると出力が一致する",
  ignore: adapter === null,
  fn: async () => {
    assertMeasuredPair(
      await sweepPair(attentionCase("i8a8-attention", "pv"), I8A8_64X64),
      i8a8Candidate(defaultI8a8Geometry("attention_pv")).name,
      I8A8_64X64.name,
    );
  },
});

Deno.test({
  name: "実 GPU: conv2d の implicit GEMM を既定と 64×64/256 で測ると出力が一致する",
  ignore: adapter === null,
  fn: async () => {
    assertMeasuredPair(
      await sweepPair(CONV2D, CONV2D_64X64),
      conv2dCandidate(gemmMTileGeometry(64)).name,
      CONV2D_64X64.name,
    );
  },
});

// 幾何どうしの一致は「全幾何が同じ取り違え（束縛の入れ替わり・params の誤り）をしている」場合を
// 検出できないので、linear の既定幾何の出力だけは CPU の参照と突き合わせる。参照の丸め順は
// src/kernels/gemm.ts の契約（K 昇順・acc へ `acc + a * b` の逐次加算・fma を明示しない・bias は
// 末尾で 1 度）に合わせ、ビット同一を要求する。

/** f16 のビット列 → 値（正規数・非正規数。掃引の入力は正規数だけを作る）。 */
const f16ToNumber = (bits: number): number => {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
};

/** f32 の隣（`up` なら +∞ 側）。 */
const adjacentF32 = (value: number, up: boolean): number => {
  const bits = new Uint32Array(new Float32Array([value]).buffer);
  if (value === 0) {
    bits[0] = 1;
    const tiny = new Float32Array(bits.buffer)[0];
    return up ? tiny : -tiny;
  }
  bits[0] += (value > 0) === up ? 1 : -1;
  return new Float32Array(bits.buffer)[0];
};

/**
 * f32 の fma（`a·b + c` を 1 回だけ丸める）。f32 どうしの積は f64 で厳密なので、和の誤差を
 * TwoSum で採り、f32 の丸めの中点に乗ったときだけ誤差の符号で向きを決める。失敗時の診断用
 * （契約は fma を明示しないが、backend が縮約するかどうかを失敗の文言で見分けられるように）。
 */
const fmaF32 = (a: number, b: number, c: number): number => {
  const product = a * b;
  const sum = product + c;
  const virtual = sum - product;
  const error = (product - (sum - virtual)) + (c - virtual);
  const rounded = Math.fround(sum);
  if (error === 0 || rounded === sum) return rounded;
  const other = adjacentF32(rounded, rounded < sum);
  if ((rounded + other) / 2 !== sum) return rounded;
  return error > 0 ? Math.max(rounded, other) : Math.min(rounded, other);
};

const digestOf = async (values: Float32Array<ArrayBuffer>): Promise<string> =>
  await joinChunkDigests([await crypto.subtle.digest("SHA-256", new Uint8Array(values.buffer))]);

/** 資源表の 1 本を、harness が GPU へ書いたのと同じ語列で作る。 */
const resourceWords = (target: SweepCase, name: string): Uint32Array<ArrayBuffer> => {
  const specs = casePlan(target, 65535, false).resources;
  const index = specs.findIndex((spec) => spec.name === name);
  const spec = specs[index];
  if (spec === undefined || spec.fill === "none") throw new Error(`資源 ${name} が埋まらない`);
  const words = new Uint32Array(Math.max(4, Math.ceil(spec.bytes / 4) * 4) / 4);
  const next = resourceWordStream(spec.fill, index);
  for (let at = 0; at < words.length; at += 1) words[at] = next();
  return words;
};

Deno.test({
  name:
    "実 GPU: f32 linear の既定幾何の出力は CPU 参照（K 昇順の逐次積和 + 末尾の bias・非融合か fma 縮約のどちらか）とビット同一",
  ignore: adapter === null,
  fn: async () => {
    const target: LinearCase = {
      id: "test-linear-cpu-m64-n64-k64",
      op: "linear",
      m: 64,
      n: 64,
      k: 64,
      censusCount: 1,
      source: "test",
    };
    const { m, n, k } = target;
    const x = new Float32Array(resourceWords(target, "x").buffer);
    const weightWords = resourceWords(target, "w");
    const bias = new Float32Array(resourceWords(target, "bias").buffer);
    // 重みの f16 は 1 語に 2 つ（下位 = 偶数添字 — kernels/weight-storage.ts）
    const weight = (index: number): number =>
      f16ToNumber((weightWords[index >> 1] >>> ((index & 1) * 16)) & 0xffff);
    const expected = new Float32Array(m * n);
    // 同じ順序で積和だけを fma に縮約した参照。src/kernels/gemm.ts の契約は「`acc + a * b` の字面のまま」で、
    // fma に縮約するかは backend が決める（B570 / Vulkan は縮約する・RTX / Vulkan は縮約しなかった —
    // docs/limitations.md の環境依存の軸）。束縛・params・K の順序の取り違えはどちらの参照とも一致しないので、
    // 「どちらか一方とビット同一」でこのテストの目的（配線の検出）は保たれる。
    const contracted = new Float32Array(m * n);
    for (let row = 0; row < m; row += 1) {
      for (let col = 0; col < n; col += 1) {
        let acc = 0;
        let fused = 0;
        for (let at = 0; at < k; at += 1) {
          const a = x[row * k + at];
          const b = weight(col * k + at);
          acc = Math.fround(acc + Math.fround(a * b));
          fused = fmaF32(a, b, fused);
        }
        expected[row * n + col] = Math.fround(acc + bias[col]);
        contracted[row * n + col] = Math.fround(fused + bias[col]);
      }
    }
    const reference = await digestOf(expected);
    const contractedReference = await digestOf(contracted);
    const gpu = await acquireGpu(timestampQuery ? { gpuTiming: true } : {});
    const context = await createSweepContext(gpu, "deno-raw-tick");
    try {
      const { rows } = await sweepCase(context, target, [], { rounds: 1 });
      assertEquals(rows.length, 1);
      assertEquals(rows[0].error, undefined, rows[0].error);
      const digest = rows[0].outputSha256;
      assert(
        digest === reference || digest === contractedReference,
        `出力 ${digest} が非融合の CPU 参照 ${reference} とも fma 縮約の CPU 参照 ${contractedReference} とも一致しない（束縛・params・K の順序のどれかが違う）`,
      );
    } finally {
      destroySweepContext(context);
      gpu.destroy();
    }
  },
});

/**
 * dense の GEMM（matmul / bmm — bias 無し・B 側 `[K,N]`）の CPU 参照 2 本（非融合 / fma 縮約）。
 * 丸め順は linear の参照と同じ（K 昇順の逐次積和）。bmm は行列 `batch` 枚を連続に並べる。
 */
const denseReferences = async (
  target: MatmulCase | BmmCase,
): Promise<{ readonly plain: string; readonly contracted: string }> => {
  const batch = target.op === "bmm" ? target.batch : 1;
  const { m, n, k } = target;
  const a = new Float32Array(resourceWords(target, "a").buffer);
  const b = new Float32Array(resourceWords(target, "b").buffer);
  const plain = new Float32Array(batch * m * n);
  const contracted = new Float32Array(batch * m * n);
  for (let z = 0; z < batch; z += 1) {
    for (let row = 0; row < m; row += 1) {
      for (let col = 0; col < n; col += 1) {
        let acc = 0;
        let fused = 0;
        for (let at = 0; at < k; at += 1) {
          const left = a[(z * m + row) * k + at];
          const right = b[(z * k + at) * n + col];
          acc = Math.fround(acc + Math.fround(left * right));
          fused = fmaF32(left, right, fused);
        }
        plain[(z * m + row) * n + col] = acc;
        contracted[(z * m + row) * n + col] = fused;
      }
    }
  }
  return { plain: await digestOf(plain), contracted: await digestOf(contracted) };
};

/**
 * 既定幾何と 64×64/256 の 2 行を測り、既定の出力が CPU 参照のどちらかとビット同一で、もう 1 行が
 * 既定と一致すること（束縛・params・dispatch の取り違えは参照と食い違う）。
 */
const assertDenseMatchesReference = async (target: MatmulCase | BmmCase): Promise<void> => {
  const { plain, contracted } = await denseReferences(target);
  const gpu = await acquireGpu(timestampQuery ? { gpuTiming: true } : {});
  const context = await createSweepContext(gpu, "deno-raw-tick");
  try {
    const { rows } = await sweepCase(context, target, [GEMM_64X64], { rounds: 1 });
    assertEquals(rows.length, 2);
    for (const row of rows) assertEquals(row.error, undefined, row.error);
    const digest = rows[0].outputSha256;
    assert(
      digest === plain || digest === contracted,
      `${target.op}: 出力 ${digest} が非融合の CPU 参照 ${plain} とも fma 縮約の CPU 参照 ${contracted} とも一致しない（束縛・params・dispatch のどれかが違う）`,
    );
    assertEquals(rows[1].identicalToDefault, true);
  } finally {
    destroySweepContext(context);
    gpu.destroy();
  }
};

Deno.test({
  name:
    "実 GPU: f32 matmul の出力は CPU 参照（K 昇順の逐次積和・非融合か fma 縮約のどちらか）とビット同一",
  ignore: adapter === null,
  fn: async () => {
    // M / N / K を全て違う値にする（m・n・k の取り違えが寸法の食い違いとして出る）
    await assertDenseMatchesReference({
      id: "test-matmul-cpu-m64-n48-k32",
      op: "matmul",
      m: 64,
      n: 48,
      k: 32,
      censusCount: 0,
      mirrorOf: "test",
      source: "test",
    });
  },
});

Deno.test({
  name:
    "実 GPU: f32 bmm（batch 3）の出力は CPU 参照とビット同一（バッチの取り違えは参照と食い違う）",
  ignore: adapter === null,
  fn: async () => {
    // 入力は資源ごとの擬似乱数列なのでバッチごとに値が違う — z のオフセットを取り違えると
    // （全バッチが 0 枚目を読む・書き先が重なる・書き残しが 0 のまま）参照と一致しない
    await assertDenseMatchesReference({
      id: "test-bmm-cpu-b3-m64-n48-k32",
      op: "bmm",
      batch: 3,
      m: 64,
      n: 48,
      k: 32,
      censusCount: 1,
      source: "test",
    });
  },
});
