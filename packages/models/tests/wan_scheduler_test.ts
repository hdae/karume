// Wan2.1 のホストの UniPC と CFG（`src/wan/scheduler.ts`）を、diffusers 0.39.0 の実クラスを駆動して焼いた
// fixture と突き合わせる。GPU も実重みも要らない純関数のテスト。
//
// fixture は `fixtures/wan-scheduler/`（生成: tools/export-recipes/wan/scheduler_ref.py — 期待値は上流の
// `UniPCMultistepScheduler.set_timesteps` / `step` と `pipeline_wan.py` の CFG の式の出力で、式の写しではない）。
//
// - σ 列: **ビット一致**（Uint32）。f64 の中間列も JSON の値と完全一致（f64 で計算 → 最後に f32 — 決定 5）。
// - timestep 列: 完全一致（最初は 999 — σ[0] の 1e-6 の補正込み）。
// - 軌跡（固定のモデル出力列 50 本）: atol（{@link TRAJECTORY_ATOL} の doc に実測）。
// - CFG: **ビット一致**。
//
// 故障注入は 2 つの門に分かれる。σ 列に出る故障（σ[0] の補正を落とす・σ 列を 1 つずらす・shift 5.0）は
// σ / timestep のビット一致の門が落とし、更新則の故障（bh2 → bh1・1 次に落とす・step の順序の取り違え）は
// 軌跡の atol の門が落とす。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import {
  flowShiftedSigmas,
  WAN_UNIPC_CONFIG,
  wanClassifierFreeGuidance,
  type WanUniPcConfig,
  WanUniPcSampler,
  wanUniPcSchedule,
} from "../src/wan/scheduler.ts";

/**
 * 軌跡（50 step・潜在 `[16,3,16,16]`）の TS と上流（torch CPU f32）の許容差（絶対）。
 *
 * 実測（2026-10-02・`atol = rtol = 0` の素の突合）: 最大絶対差 **3.815e-6**（step 38・参照の最大絶対値
 * 4.29）。不一致は 614,400 要素中 364,883 件で、step 0〜3 はビット一致。
 *
 * 差の出所は f32 の `log` の 1 ULP: 上流の 0-d テンソルの `torch.log`（MKL の VML）と `Math.log` を f32 へ
 * 丸めた値（正しい丸め）が、50 step の σ のうち 6 本（step 19 / 21 / 23 / 32 / 42 / 43 の `log(1 − σ)` か
 * `log(σ)`）で 1 ULP 割れ、λ → h → `expm1` の係数へ伝わる（係数を 16 進で突き合わせて確かめた）。式の取り違え
 * ではない — 割れた係数以外は全てビット一致する。
 *
 * atol 2e-5 は実測最悪の約 5.2 倍（決定 8 の「実測最悪の約 5 倍」）。
 */
const TRAJECTORY_ATOL = 2e-5;

/**
 * 更新則の故障注入が帯から離れているべき倍率（下限）。
 *
 * ADR 0118 検収 段 6 の目安（Anima の前例）は 1e4 倍だが、50 step・乱数のモデル出力の軌跡では実測で
 * 2.0e3〜1.05e4 倍だった（bh1 7.96e3・1 次 1.05e4・step 0 / 1 の取り違え 2.04e3・step 10 / 11 の取り違え
 * 4.71e3 — 2026-10-02）。帯（実測最悪の 5 倍）は動かさず、門は観測の最小を下回る 1e3 倍に置く（開示）。
 * bh1 は有限の要素の差（最後の step は上流と同じく NaN — 下の故障注入の注記）。
 *
 * σ 列に出る故障（σ[0] の補正・1 つずらす・shift 5.0）は軌跡では 1.5e-3〜2.7e-1（atol の 77〜1.35e4 倍）で、
 * 門はビット一致の σ / timestep の側が持つ（σ[0] の補正は max で効く入力が 1e-6 しか動かないので、軌跡の
 * atol では倍率が出ない）。
 */
const UPDATE_FAULT_MARGIN = 1e3;

const FIXTURE_DIR = new URL("./fixtures/wan-scheduler/", import.meta.url);

type Schedule = {
  readonly steps: number;
  readonly sigmas_f64: readonly number[];
  readonly sigmas_f32: readonly number[];
  readonly timesteps: readonly number[];
};

type Meta = {
  readonly scheduler_config: {
    readonly num_train_timesteps: number;
    readonly solver_order: number;
    readonly solver_type: string;
    readonly flow_shift: number;
    readonly prediction_type: string;
    readonly predict_x0: boolean;
    readonly lower_order_final: boolean;
    readonly final_sigmas_type: string;
    readonly use_flow_sigmas: boolean;
    readonly use_dynamic_shifting: boolean;
    readonly thresholding: boolean;
    readonly disable_corrector: readonly number[];
  };
  readonly latent_shape: readonly number[];
  readonly guidance_scale: number;
  readonly orders: readonly number[];
  readonly correctors: readonly boolean[];
  readonly schedules: Readonly<Record<string, Schedule>>;
};

const meta = JSON.parse(await Deno.readTextFile(new URL("unipc.json", FIXTURE_DIR))) as Meta;
const fixture: SafetensorsFile = await (async () => {
  const bytes = await Deno.readFile(new URL("unipc.safetensors", FIXTURE_DIR));
  return parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
  );
})();

const floats = (key: string): Float32Array => {
  const view = fixture.tensors.get(key);
  if (view === undefined || view.dtype !== "F32") throw new Error(`fixture '${key}' が F32 で無い`);
  return new Float32Array(fixture.buffer, view.byteOffset, view.byteLength / 4);
};

const bitsOf = (values: Float32Array): Uint32Array =>
  new Uint32Array(values.buffer, values.byteOffset, values.length);

const SHIFT = meta.scheduler_config.flow_shift;
const STEPS = 50;
const ELEMENTS = meta.latent_shape.reduce((count, dim) => count * dim, 1);

/** 上流の軌跡のループ（`for t in timesteps: latents = step(out[i], t, latents)`）を TS で回す。 */
const runTrajectory = (
  sigmas: Float32Array,
  config: WanUniPcConfig = WAN_UNIPC_CONFIG,
  order: readonly number[] = [...Array(STEPS).keys()],
): Float32Array[] => {
  const outputs = floats("model_outputs");
  const sampler = new WanUniPcSampler({ sigmas }, config);
  let latents: Float32Array = floats("latents_init").slice();
  const states: Float32Array[] = [];
  for (const step of order) {
    latents = sampler.step(outputs.subarray(step * ELEMENTS, (step + 1) * ELEMENTS), latents);
    states.push(latents);
  }
  return states;
};

/**
 * 軌跡全体の差（fixture の `trajectory` と）: 有限の要素どうしの最大絶対差と、非有限の要素の数。
 *
 * MUST: 非有限を最大値の計算に混ぜない — `Math.max` は NaN を返し、比較の門が「超えない」側に倒れる。
 */
const trajectoryDifference = (
  states: readonly Float32Array[],
): { readonly maxAbs: number; readonly nonFinite: number } => {
  const trajectory = floats("trajectory");
  let maxAbs = 0;
  let nonFinite = 0;
  states.forEach((state, step) => {
    const want = trajectory.subarray(step * ELEMENTS, (step + 1) * ELEMENTS);
    for (let element = 0; element < ELEMENTS; element += 1) {
      if (!Number.isFinite(state[element])) {
        nonFinite += 1;
        continue;
      }
      maxAbs = Math.max(maxAbs, Math.abs(state[element] - want[element]));
    }
  });
  return { maxAbs, nonFinite };
};

Deno.test("UniPC: fixture の scheduler config がこの移植の対象の分岐（flow・bh2・2 次・x0 予測）", () => {
  const config = meta.scheduler_config;
  assertEquals(
    {
      numTrainTimesteps: config.num_train_timesteps,
      solverOrder: config.solver_order,
      solverType: config.solver_type,
    },
    { ...WAN_UNIPC_CONFIG },
  );
  assertEquals(
    [
      config.prediction_type,
      config.predict_x0,
      config.lower_order_final,
      config.final_sigmas_type,
      config.use_flow_sigmas,
      config.use_dynamic_shifting,
      config.thresholding,
      config.disable_corrector.length,
    ],
    ["flow_prediction", true, true, "zero", true, false, false, 0],
  );
  assertEquals(SHIFT, 3);
});

for (const steps of [2, STEPS]) {
  Deno.test(`UniPC: ${steps} step の σ 列は上流とビット一致・timestep は完全一致`, () => {
    const expected = meta.schedules[String(steps)];
    const schedule = wanUniPcSchedule(steps, SHIFT, WAN_UNIPC_CONFIG.numTrainTimesteps);
    assertEquals(
      [...bitsOf(schedule.sigmas)],
      [...bitsOf(Float32Array.from(expected.sigmas_f32))],
    );
    assertEquals(schedule.timesteps, expected.timesteps);
    assertEquals(schedule.timesteps[0], 999, "最初の timestep は σ[0] の 1e-6 の補正で 999");
    // f64 の中間列（補正込み・最後の 0 を足した形）も上流の numpy の式と完全一致する。
    const sigmas64 = flowShiftedSigmas(steps, SHIFT, WAN_UNIPC_CONFIG.numTrainTimesteps);
    sigmas64[0] -= 1e-6;
    assertEquals([...sigmas64, 0], expected.sigmas_f64);
  });
}

Deno.test("UniPC: σ 列と timestep は safetensors の F32 / I64 の列とも一致する（JSON と二重の出所）", () => {
  const schedule = wanUniPcSchedule(STEPS, SHIFT, WAN_UNIPC_CONFIG.numTrainTimesteps);
  assertEquals([...bitsOf(schedule.sigmas)], [...bitsOf(floats("sigmas"))]);
  const view = fixture.tensors.get("timesteps");
  assert(view !== undefined && view.dtype === "I64", "fixture の timesteps が I64 で無い");
  const timesteps = new BigInt64Array(fixture.buffer, view.byteOffset, view.byteLength / 8);
  assertEquals(schedule.timesteps, [...timesteps].map(Number));
});

Deno.test("UniPC: 固定のモデル出力列の軌跡が上流の step と atol の内（次数 1 → 2 → 1・修正子は 2 step 目から）", () => {
  // 次数と修正子の有無が上流の記録と同じ形であることは fixture のメタで先に見る（数値の門の前提）。
  assertEquals(meta.orders, [1, ...Array(STEPS - 2).fill(2), 1]);
  assertEquals(meta.correctors, [false, ...Array(STEPS - 1).fill(true)]);
  const schedule = wanUniPcSchedule(STEPS, SHIFT, WAN_UNIPC_CONFIG.numTrainTimesteps);
  const states = runTrajectory(schedule.sigmas);
  const { maxAbs, nonFinite } = trajectoryDifference(states);
  assertEquals(nonFinite, 0, "非有限");
  console.log(`[wan-scheduler] trajectory maxAbs=${maxAbs}（atol ${TRAJECTORY_ATOL}）`);
  assert(maxAbs <= TRAJECTORY_ATOL, `軌跡の最大絶対差 ${maxAbs} が atol ${TRAJECTORY_ATOL} の外`);
  // 最初の 4 step は係数の 1 ULP がまだ現れないのでビット一致する（atol が緩みを隠していないことの陽性対照）。
  const trajectory = floats("trajectory");
  for (let step = 0; step < 4; step += 1) {
    assertEquals(
      [...bitsOf(states[step])],
      [...bitsOf(trajectory.subarray(step * ELEMENTS, (step + 1) * ELEMENTS))],
      `step ${step}`,
    );
  }
});

Deno.test("UniPC 故障注入（σ 列の門）: σ[0] の補正を落とす・σ 列を 1 つずらす・shift 5.0 は σ / timestep の一致が割れる", () => {
  const expected = meta.schedules[String(STEPS)];
  const want = [...bitsOf(Float32Array.from(expected.sigmas_f32))];
  const noEpsilon = Float32Array.from([
    ...flowShiftedSigmas(STEPS, SHIFT, WAN_UNIPC_CONFIG.numTrainTimesteps),
    0,
  ]);
  const faults: readonly { readonly label: string; readonly sigmas: Float32Array }[] = [
    { label: "σ[0] の 1e-6 の補正を落とす", sigmas: noEpsilon },
    {
      label: "σ 列を 1 つずらす（51 step の列の 2 本目から）",
      sigmas: wanUniPcSchedule(STEPS + 1, SHIFT, WAN_UNIPC_CONFIG.numTrainTimesteps).sigmas
        .slice(1),
    },
    {
      label: "shift 5.0",
      sigmas: wanUniPcSchedule(STEPS, 5, WAN_UNIPC_CONFIG.numTrainTimesteps).sigmas,
    },
  ];
  for (const { label, sigmas } of faults) {
    assertEquals(sigmas.length, want.length, label);
    const mismatched = [...bitsOf(sigmas)].filter((bits, index) => bits !== want[index]).length;
    // 軌跡にどれだけ響くかは記録だけ（σ 列の故障は σ の門が落とす — ファイル冒頭）。
    console.log(
      `[wan-scheduler] fault ${label}: σ の不一致 ${mismatched} 本・軌跡 maxAbs=${
        trajectoryDifference(runTrajectory(sigmas)).maxAbs
      }`,
    );
    assert(mismatched > 0, `${label}: σ 列が上流とビット一致した`);
  }
  // σ[0] の補正は timestep の門でも落ちる（補正なしなら 1000）。
  assertEquals(Math.trunc(noEpsilon[0] * WAN_UNIPC_CONFIG.numTrainTimesteps), 1000);
  assertEquals(expected.timesteps[0], 999);
});

Deno.test("UniPC 故障注入（軌跡の門）: bh2 → bh1・1 次に落とす・step の順序の取り違えは atol の外", () => {
  const sigmas = wanUniPcSchedule(STEPS, SHIFT, WAN_UNIPC_CONFIG.numTrainTimesteps).sigmas;
  const swapped = (a: number, b: number): number[] => {
    const order = [...Array(STEPS).keys()];
    [order[a], order[b]] = [order[b], order[a]];
    return order;
  };
  const faults: readonly { readonly label: string; readonly states: () => Float32Array[] }[] = [
    {
      label: "bh2 → bh1",
      states: () => runTrajectory(sigmas, { ...WAN_UNIPC_CONFIG, solverType: "bh1" }),
    },
    {
      label: "1 次に落とす",
      states: () => runTrajectory(sigmas, { ...WAN_UNIPC_CONFIG, solverOrder: 1 }),
    },
    {
      label: "step 0 / 1 のモデル出力の取り違え",
      states: () => runTrajectory(sigmas, WAN_UNIPC_CONFIG, swapped(0, 1)),
    },
    {
      label: "step 10 / 11 のモデル出力の取り違え",
      states: () => runTrajectory(sigmas, WAN_UNIPC_CONFIG, swapped(10, 11)),
    },
  ];
  for (const { label, states } of faults) {
    // bh1 は最後の step（σ = 0 で h = ∞・`B(h) = hh = −∞`）の `−∞ × 0` で上流と同じく NaN になる
    // （上流の `x_t_ − alpha_t·B_h·0`）。門は有限の要素の差で見る。
    const { maxAbs, nonFinite } = trajectoryDifference(states());
    console.log(
      `[wan-scheduler] fault ${label}: maxAbs=${maxAbs}（atol の ${
        (maxAbs / TRAJECTORY_ATOL).toExponential(2)
      } 倍）・非有限 ${nonFinite}`,
    );
    assert(
      maxAbs > UPDATE_FAULT_MARGIN * TRAJECTORY_ATOL,
      `${label}: 最大絶対差 ${maxAbs} が atol の ${UPDATE_FAULT_MARGIN} 倍を超えない`,
    );
  }
});

Deno.test("UniPC: σ 列の外の step と長さ違いの入力は fail loudly", () => {
  const schedule = wanUniPcSchedule(1, SHIFT, WAN_UNIPC_CONFIG.numTrainTimesteps);
  assertEquals(schedule.timesteps, [999]);
  const sampler = new WanUniPcSampler(schedule, WAN_UNIPC_CONFIG);
  assertThrows(() => sampler.step(new Float32Array(3), new Float32Array(4)), RangeError, "要素");
  const once = sampler.step(new Float32Array([1, -1]), new Float32Array([0.5, 0.25]));
  // 1 step（σ: 0.999999 → 0）は 1 次で x0 の予測そのものになる: x0 = x − σ·v。
  assertEquals([...once], [
    Math.fround(0.5 - schedule.sigmas[0]),
    Math.fround(0.25 + schedule.sigmas[0]),
  ]);
  assertThrows(() => sampler.step(new Float32Array(2), new Float32Array(2)), RangeError, "σ 列");
  assertThrows(() => wanUniPcSchedule(0, SHIFT, 1000), RangeError, "steps");
  assertThrows(() => wanUniPcSchedule(2, 0, 1000), RangeError, "shift");
  assertThrows(() => wanUniPcSchedule(2, Number.NaN, 1000), RangeError, "shift");
});

Deno.test("CFG: uncond + g·(cond − uncond) は上流の式とビット一致（guide 5.0）", () => {
  assertEquals(meta.guidance_scale, 5);
  const got = wanClassifierFreeGuidance(
    floats("cfg.cond"),
    floats("cfg.uncond"),
    meta.guidance_scale,
  );
  assertEquals([...bitsOf(got)], [...bitsOf(floats("cfg.out"))]);
  // 演算順の故障注入: `cond·g + uncond·(1 − g)` に変えると最終ビットが割れる（式の順が門で縛られている）。
  const cond = floats("cfg.cond");
  const uncond = floats("cfg.uncond");
  const reordered = cond.map((value, index) =>
    Math.fround(Math.fround(value * 5) + Math.fround(uncond[index] * -4))
  );
  assert(
    bitsOf(reordered).some((bits, index) => bits !== bitsOf(got)[index]),
    "演算順を変えてもビット一致した（門が式の順を縛っていない）",
  );
  assertThrows(
    () => wanClassifierFreeGuidance(new Float32Array(2), new Float32Array(3), 5),
    RangeError,
  );
});
