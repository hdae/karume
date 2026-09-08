// states 形 attention の **readonly 変種**（①' ro / ② ro / ③' ro — ADR 0096 決定 1・段 2）の実 GPU 門。
//
// drafter（MTP head）は自前の K/V を持たず、貸し手 context のスロットだけを読む。query は論理位置
// P−1 の 1 行で、列 `[P − min(P, W), P)`（full は `[0, P)`）を見る。門は 3 本:
//
// ① f64 参照（`referenceStateAttentionReadonly`）との帯（states 形と同じ atol）
// ② **states 形との同値**: readonly(P) は「past = P−1・Q = 1・ins = 位置 P−1 の行」の ①'/③' と
//    **ビット同一**（同じ列を同じ順で足すので、値が変わる理由が無い — 変わったら述語か写像が違う）
// ③ 構造的保証: P = 0 は厳密 0（空行）・決定性・棄却行（col ≥ P）と窓外（col < P−W）を**読まない**
//    （スロットの当該行に毒値を置いて値が動かないこと）・故障注入（live を 1 列広げる / 窓の先頭を
//    1 列前へずらす / 写像の法を window へ戻す）が赤になること

import { assert, assertEquals } from "@std/assert";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import {
  referenceStateAttentionReadonly,
  type StateAttentionReadonlyRefInput,
} from "../src/reference/state-attention.ts";
import { stateColumnBaseReadonly, stateSliding } from "../src/kernels/state-attention.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  halfScale,
  runStateAttention,
  runStateAttentionReadonly,
  seeded,
  type StateCase,
  type StateInputs,
  type StateMutation,
  type StatePipelineCache,
  type StateReadonlyInputs,
} from "./helpers/state-dispatch.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** f64 参照との許容差（gpu_state_attention_test.ts の `STATE_TOLERANCE` と同値・同根拠）。 */
const STATE_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

const QUERY = (i: number): number => Math.sin(i * 0.37 + 0.2) * 0.9;
const KEY = (i: number): number => Math.cos(i * 0.53 + 1.1) * 0.8;
const VALUE = (i: number): number => Math.sin(i * 0.71 + 2.3) * 1.2;
/** live の外（棄却行・窓外・未初期化）に置く毒値（読んだら参照との帯を必ず超える）。 */
const SLOT_POISON = 2 ** 60;

/** M = Q = 1 固定の readonly ケース（`past` = 貸し手の P・容量 C・窓 W）。 */
const readonlyCase = (
  name: string,
  shape: { heads: number; kvHeads: number; depth: number; capacity: number; window: number },
  past: number,
): StateCase => ({
  name,
  batch: 1,
  heads: shape.heads,
  kvHeads: shape.kvHeads,
  chunkRows: 1,
  depth: shape.depth,
  capacity: shape.capacity,
  window: shape.window,
  past,
  query: 1,
});

/** drafter の実形に近い GQA 4:1（sliding は W=8 / C=16 = 余裕 8 の縮小版）。 */
const SLIDING = { heads: 4, kvHeads: 1, depth: 16, capacity: 16, window: 8 };
const FULL = { heads: 4, kvHeads: 1, depth: 16, capacity: 24, window: 0 };
const PLAIN = { heads: 2, kvHeads: 2, depth: 8, capacity: 12, window: 0 };

const CASES: readonly StateCase[] = [
  readonlyCase("full P=0（空行）", FULL, 0),
  readonlyCase("full P=1", FULL, 1),
  readonlyCase("full P=7", FULL, 7),
  readonlyCase("full P=C（容量いっぱい）", FULL, 24),
  readonlyCase("plain P=5", PLAIN, 5),
  readonlyCase("sliding P=0（空行）", SLIDING, 0),
  readonlyCase("sliding P=3 < W", SLIDING, 3),
  readonlyCase("sliding P=8 = W", SLIDING, 8),
  readonlyCase("sliding P=9（窓が動き始める）", SLIDING, 9),
  readonlyCase("sliding P=17（ring 一周後）", SLIDING, 17),
  readonlyCase("sliding P=40（ring 複数周）", SLIDING, 40),
];

/**
 * 入力: q 1 行と貸し手スロット。live の外の物理行（棄却行 = 論理 col ≥ P が写る行・窓外・
 * 未初期化）は**毒値**で埋める — readonly が 1 語でも読めば参照との帯を超える。
 */
const makeInputs = (spec: StateCase): StateReadonlyInputs => {
  const { batch, heads, kvHeads, depth, capacity, window, past } = spec;
  const kvPlanes = batch * kvHeads;
  const q = seeded(batch * heads * depth, QUERY);
  const slotK = seeded(kvPlanes * capacity * depth, (i) => KEY(i + 3));
  const slotV = seeded(kvPlanes * capacity * depth, (i) => VALUE(i + 5));
  const resident = new Set<number>();
  for (let col = stateColumnBaseReadonly(window, past); col < past; col += 1) {
    resident.add(stateSliding(window) ? col % capacity : col);
  }
  for (let plane = 0; plane < kvPlanes; plane += 1) {
    for (let row = 0; row < capacity; row += 1) {
      if (resident.has(row)) continue;
      for (let d = 0; d < depth; d += 1) {
        slotK[(plane * capacity + row) * depth + d] = SLOT_POISON;
        slotV[(plane * capacity + row) * depth + d] = SLOT_POISON;
      }
    }
  }
  return { q, slotK, slotV };
};

const refInput = (
  spec: StateCase,
  inputs: StateReadonlyInputs,
): StateAttentionReadonlyRefInput => ({
  batch: spec.batch,
  heads: spec.heads,
  kvHeads: spec.kvHeads,
  depth: spec.depth,
  capacity: spec.capacity,
  window: spec.window,
  past: spec.past,
  q: inputs.q,
  slotK: inputs.slotK,
  slotV: inputs.slotV,
  scale: halfScale(spec.depth),
});

/**
 * 門 ②の相手: 同じスロットに対する states 形（①'/③'）を「past = P−1・Q = 1・ins = 位置 P−1 の
 * 行」で組む。ins に置くのは貸し手スロットの物理行 `slot_row(P−1)` の中身そのもの。
 */
const equivalentStatesRun = (spec: StateCase, inputs: StateReadonlyInputs): {
  readonly spec: StateCase;
  readonly inputs: StateInputs;
} => {
  const { batch, kvHeads, depth, capacity, window, past } = spec;
  const kvPlanes = batch * kvHeads;
  const last = past - 1;
  const physical = stateSliding(window) ? last % capacity : last;
  const insK = new Float32Array(kvPlanes * depth);
  const insV = new Float32Array(kvPlanes * depth);
  for (let plane = 0; plane < kvPlanes; plane += 1) {
    for (let d = 0; d < depth; d += 1) {
      insK[plane * depth + d] = inputs.slotK[(plane * capacity + physical) * depth + d];
      insV[plane * depth + d] = inputs.slotV[(plane * capacity + physical) * depth + d];
    }
  }
  return {
    spec: { ...spec, past: last, query: 1 },
    inputs: { q: inputs.q, insK, insV, slotK: inputs.slotK, slotV: inputs.slotV },
  };
};

const bitsOf = (data: Float32Array<ArrayBuffer>): Uint32Array =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

const assertBitIdentical = (
  a: Float32Array<ArrayBuffer>,
  b: Float32Array<ArrayBuffer>,
  label: string,
): void => {
  assertEquals(a.length, b.length, `${label}: 長さ`);
  const ab = bitsOf(a);
  const bb = bitsOf(b);
  for (let i = 0; i < ab.length; i += 1) {
    if (ab[i] !== bb[i]) {
      throw new Error(`${label}: 要素 ${i} がビット不一致（${a[i]} vs ${b[i]}）`);
    }
  }
};

Deno.test({
  name:
    "states 形 readonly 変種: 参照の帯・states 形（P−1・Q=1）とビット同一・空行と決定性（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      let worst = 0;
      for (const spec of CASES) {
        const inputs = makeInputs(spec);
        const expected = referenceStateAttentionReadonly(refInput(spec, inputs));
        const got = await runStateAttentionReadonly(gpu.device, spec, inputs, { cache });

        // ① f64 参照との帯（毒値を 1 語でも読めば必ず外れる）
        const ref = compareTensors({ dtype: "f32", data: got.out }, expected, STATE_TOLERANCE);
        assert(ref.pass, `${spec.name} vs 参照: ${formatAllclose(ref)}`);
        worst = Math.max(worst, ref.maxAbsError);

        if (spec.past === 0) {
          // ③ 空行は厳密 0（Object.is — −0 も NaN も通さない・毒値の初期化が残っていない）
          for (let i = 0; i < got.out.length; i += 1) {
            assert(
              Object.is(got.out[i], 0),
              `${spec.name}: 空行の出力 ${i} が厳密 0 でない（${got.out[i]}）`,
            );
          }
        } else {
          // ② states 形（past = P−1・Q = 1・ins = 位置 P−1 の行）と**ビット同一**
          const eq = equivalentStatesRun(spec, inputs);
          const states = await runStateAttention(gpu.device, eq.spec, eq.inputs, {
            cache,
            qkReduce: "parallel",
            pvReduce: "parallel",
          });
          assertBitIdentical(got.out, states.out, `${spec.name}: readonly vs states 形（①'/③'）`);
        }

        // ③ 決定性
        const again = await runStateAttentionReadonly(gpu.device, spec, inputs, { cache });
        assertBitIdentical(got.out, again.out, `${spec.name}: 決定性`);
      }
      console.log(
        `readonly 実測最悪: vs 参照 maxAbs ${worst.toExponential(2)}（帯 ${STATE_TOLERANCE.atol}）`,
      );
    } finally {
      gpu.destroy();
    }
  },
});

/** 故障注入 — live の式と物理行の写像を崩した変異が参照の帯で落ちること（門が空振りでない証明）。 */
const MUTATIONS: readonly { readonly name: string; readonly mutate: StateMutation }[] = [
  {
    // live を 1 列広げる（位置 P = 棄却行〈毒値〉を読む）
    name: "live = min(past, window) → min(past, window + 1u)",
    mutate: (_kernel, wgsl) =>
      wgsl.replaceAll("return min(past, params.window);", "return min(past, params.window + 1u);"),
  },
  {
    // 窓の先頭を 1 列前へ（窓外 P−W−1〈毒値〉を読み、P−1 を落とす）
    name: "column_base = past − min(past, window) → past − min(past, window + 1u)",
    mutate: (_kernel, wgsl) =>
      wgsl.replaceAll(
        "return past - min(past, params.window);",
        "return past - min(past, params.window + 1u);",
      ),
  },
  {
    // 読み側の写像を window の法へ戻す（読み書き同式を破る — ring 一周後は別の物理行を指す）
    name: "slot_row = col % capacity → col % window",
    mutate: (_kernel, wgsl) =>
      wgsl.replaceAll("return col % params.capacity;", "return col % params.window;"),
  },
];

Deno.test({
  name:
    "states 形 readonly 変種: 故障注入（live のずれ・窓の先頭のずれ・写像の法）は参照の帯で落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      // 変異が値に出る形: 位置 P の物理行と窓外の行に毒値が居て、ring が一周した sliding のケース
      const targets = CASES.filter((spec) => spec.past >= 9 && stateSliding(spec.window));
      assert(targets.length >= 2, "故障注入の対象ケースが足りない");
      for (const mutation of MUTATIONS) {
        let caught = 0;
        for (const spec of targets) {
          const inputs = makeInputs(spec);
          const expected = referenceStateAttentionReadonly(refInput(spec, inputs));
          let changed = 0;
          const mutated = await runStateAttentionReadonly(gpu.device, spec, inputs, {
            cache,
            mutate: (_kernel, wgsl) => {
              const after = mutation.mutate(_kernel, wgsl);
              if (after !== wgsl) changed += 1;
              return after;
            },
          });
          assert(
            changed > 0,
            `故障注入「${mutation.name}」が 1 本のカーネルも書き換えていない（文言が変わった）`,
          );
          const cmp = compareTensors(
            { dtype: "f32", data: mutated.out },
            expected,
            STATE_TOLERANCE,
          );
          if (!cmp.pass) caught += 1;
        }
        assert(caught > 0, `故障注入「${mutation.name}」が 1 ケースも赤にならない（門が空振り）`);
      }
    } finally {
      gpu.destroy();
    }
  },
});
