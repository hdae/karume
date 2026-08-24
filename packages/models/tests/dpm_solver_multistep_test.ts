// DPM++ 2M（`src/generation/dpm-solver-multistep.ts`）の挙動テスト。GPU も資産も要らない純関数。
//
// 期待値の由来は `tools/export-recipes/anima/dpmsolver_ref.py` —— diffusers 0.39.0 の
// `DPMSolverMultistepScheduler` **実クラス**に karume の sigma 梯子を差し込み、合成 model 出力
// （シード固定）を食わせて印字した golden をそのまま inline している。再生成は
//
//     cd tools/export-recipes
//     uv run --no-sync python -m anima.dpmsolver_ref --steps 8 --elements 4   # SERIES_8
//     uv run --no-sync python -m anima.dpmsolver_ref --steps 2 --elements 3   # SERIES_2
//
// tolerance（{@link TOLERANCE}）の導出根拠はモジュール側の doc（「参照とビット一致しない」の
// MUST 節）が正本 —— ここに数値の由来を二重に書かない。

import { assert, assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import { dpmSolverMultistepStep, needsUncond } from "../src/generation/dpm-solver-multistep.ts";

/**
 * パリティの許容差。モジュール doc の MUST 節の実測から導出（フィクスチャ実測最悪
 * 3.576e-07 の約 11 倍・掃引最悪 1.192e-06 の約 3.4 倍）。実装バグの差は最小でも 1.36e-01
 * （= この閾値の 4.5 桁上）なので、丸め差を許しても検出力は残る。
 */
const TOLERANCE = 4e-6;

/** `models/karume-anima-turbo/karume.json` の `pipelineConfig.scheduler.shift` と同じ値。 */
const SHIFT = 3;

type Series = {
  readonly sigmas: readonly number[];
  readonly initial: readonly number[];
  readonly steps: readonly {
    readonly velocity: readonly number[];
    readonly x0: readonly number[];
    readonly sample: readonly number[];
  }[];
};

/** `--steps 8 --elements 4`（既定 seed）。1 次 → 2 次 ×6 → 1 次 の全分岐を 1 本で踏む。 */
const SERIES_8: Series = {
  sigmas: [
    1,
    0.9545454382896423,
    0.8999999761581421,
    0.8333333134651184,
    0.75,
    0.6428571343421936,
    0.5,
    0.30000001192092896,
    0,
  ],
  initial: [0.6157209873199463, -1.2011234760284424, 0.2345673143863678, -0.9813034534454346],
  steps: [
    {
      velocity: [0.1726374477148056, -1.3876512050628662, -0.6250362992286682, 1.5988130569458008],
      x0: [0.4430835247039795, 0.18652772903442383, 0.8596036434173584, -2.5801165103912354],
      sample: [0.6078738570213318, -1.1380484104156494, 0.26297807693481445, -1.0539767742156982],
    },
    {
      velocity: [
        -0.5767762660980225,
        -0.8768693208694458,
        0.5082118511199951,
        -0.7355096936225891,
      ],
      x0: [1.158432960510254, -0.3010367751121521, -0.22213321924209595, -0.35189932584762573],
      sample: [0.6393344402313232, -1.090219259262085, 0.23525743186473846, -1.0138580799102783],
    },
    {
      velocity: [-1.2072668075561523, 1.2542095184326172, 1.3473405838012695, 0.2711418569087982],
      x0: [1.7258745431900024, -2.219007968902588, -0.9773491024971008, -1.2578856945037842],
      sample: [0.734398365020752, -1.2231121063232422, 0.12603075802326202, -1.0552119016647339],
    },
    {
      velocity: [1.2115410566329956, 1.3144892454147339, 0.2535824775695801, 1.3913758993148804],
      x0: [-0.275219202041626, -2.3185198307037354, -0.08528797328472137, -2.214691638946533],
      sample: [0.5464824438095093, -1.3369770050048828, 0.14366188645362854, -1.2127362489700317],
    },
    {
      velocity: [
        0.402450829744339,
        -0.6969248056411743,
        -0.20336712896823883,
        0.4934312701225281,
      ],
      x0: [0.24464431405067444, -0.8142833709716797, 0.29618722200393677, -1.5828096866607666],
      sample: [0.5404958128929138, -1.1548610925674438, 0.19269946217536926, -1.2204694747924805],
    },
    {
      velocity: [
        -1.5521337985992432,
        -0.8093506097793579,
        -0.22987085580825806,
        -1.7819559574127197,
      ],
      x0: [1.538296103477478, -0.6345642805099487, 0.34047359228134155, -0.07492637634277344],
      sample: [0.9276240468025208, -1.016262412071228, 0.2312002331018448, -0.7731197476387024],
    },
    {
      velocity: [
        0.4386770725250244,
        -0.9429063200950623,
        -0.23527345061302185,
        -0.5406302809715271,
      ],
      x0: [0.7082855105400085, -0.5448092222213745, 0.3488369584083557, -0.5028046369552612],
      sample: [0.6005957126617432, -0.8018047213554382, 0.2806660830974579, -0.7883514761924744],
    },
    {
      velocity: [-0.5566991567611694, 0.9431769251823425, 1.243248701095581, 0.281678169965744],
      x0: [0.7676054835319519, -1.0847578048706055, -0.09230855107307434, -0.8728549480438232],
      sample: [0.7676054835319519, -1.0847578048706055, -0.09230855107307434, -0.8728549480438232],
    },
  ],
};

/** `--steps 2 --elements 3`（既定 seed）。梯子の最小長 — 2 step とも 1 次に落ちる。 */
const SERIES_2: Series = {
  sigmas: [1, 0.75, 0],
  initial: [0.6157209873199463, -1.2011234760284424, 0.2345673143863678],
  steps: [
    {
      velocity: [-0.9813034534454346, 0.1726374477148056, -1.3876512050628662],
      x0: [1.5970244407653809, -1.3737609386444092, 1.6222184896469116],
      sample: [0.8610468506813049, -1.244282841682434, 0.5814801454544067],
    },
    {
      velocity: [-0.6250362992286682, 1.5988130569458008, -0.5767762660980225],
      x0: [1.3298240900039673, -2.443392753601074, 1.0140624046325684],
      sample: [1.3298240900039673, -2.443392753601074, 1.0140624046325684],
    },
  ],
};

/**
 * 系列を最後まで回して、各 step の `sample` / `x0` と golden の**最大絶対差**を返す。
 *
 * `velocityOrder` は step 順序の故障注入口 — 与えた並びで velocity を食わせる（既定は素直な
 * 0,1,2,… の順）。**状態（x と直前 x0）は素直に繋ぐ**ので、注入した誤りは以降の step へ
 * そのまま伝播する。
 */
const worstDeviation = (series: Series, velocityOrder?: readonly number[]): number => {
  const sigmas = Float32Array.from(series.sigmas);
  let sample = Float32Array.from(series.initial);
  let previousX0: Float32Array<ArrayBuffer> | undefined = undefined;
  let worst = 0;
  for (let index = 0; index < series.steps.length; index += 1) {
    const fed = series.steps[velocityOrder === undefined ? index : velocityOrder[index]];
    const update = dpmSolverMultistepStep({
      sample,
      cond: Float32Array.from(fed.velocity),
      uncond: undefined,
      guidance: 1,
      sigmas,
      index,
      previousX0,
    });
    const expected = series.steps[index];
    for (let element = 0; element < expected.sample.length; element += 1) {
      worst = Math.max(worst, Math.abs(update.sample[element] - expected.sample[element]));
      worst = Math.max(worst, Math.abs(update.x0[element] - expected.x0[element]));
    }
    sample = update.sample;
    previousX0 = update.x0;
  }
  return worst;
};

Deno.test("dpmSolverMultistepStep: リグ golden とパリティする（steps=8・1 次/2 次/1 次 の全分岐）", () => {
  // step ごとに突き合わせるので「最後だけ合っている」では通らない（途中の x0 も見る）。
  const sigmas = Float32Array.from(SERIES_8.sigmas);
  let sample = Float32Array.from(SERIES_8.initial);
  let previousX0: Float32Array<ArrayBuffer> | undefined = undefined;
  for (let index = 0; index < SERIES_8.steps.length; index += 1) {
    const expected = SERIES_8.steps[index];
    const update = dpmSolverMultistepStep({
      sample,
      cond: Float32Array.from(expected.velocity),
      uncond: undefined,
      guidance: 1,
      sigmas,
      index,
      previousX0,
    });
    for (let element = 0; element < expected.sample.length; element += 1) {
      assertAlmostEquals(
        update.x0[element],
        expected.x0[element],
        TOLERANCE,
        `x0 が参照と割れた（step=${index} element=${element}）`,
      );
      assertAlmostEquals(
        update.sample[element],
        expected.sample[element],
        TOLERANCE,
        `状態が参照と割れた（step=${index} element=${element}）`,
      );
    }
    sample = update.sample;
    previousX0 = update.x0;
  }
});

Deno.test("dpmSolverMultistepStep: 最終 step は x0 をそのまま返す（終端 σ=0 の 1 次落ち）", () => {
  // `final_sigmas_type="zero"` の帰結。係数が (0, −1) になるのでビット同一で一致する。
  const sigmas = Float32Array.from(SERIES_8.sigmas);
  const last = SERIES_8.steps.length - 1;
  const update = dpmSolverMultistepStep({
    sample: Float32Array.from(SERIES_8.steps[last - 1].sample),
    cond: Float32Array.from(SERIES_8.steps[last].velocity),
    uncond: undefined,
    guidance: 1,
    sigmas,
    index: last,
    previousX0: Float32Array.from(SERIES_8.steps[last - 1].x0),
  });
  assertEquals([...update.sample], [...update.x0]);
});

Deno.test("dpmSolverMultistepStep: 境界 — 梯子が最小長（steps=2）でも参照と一致する", () => {
  // 2 step しか無いと 2 次の分岐に一度も入らない（step 0 と最終 step が全て）。この経路が
  // 落ちていると 2 次側だけ見ているテストは素通りする。
  assert(
    worstDeviation(SERIES_2) <= TOLERANCE,
    `steps=2 の系列が参照と割れた（最大差 ${worstDeviation(SERIES_2)}）`,
  );
});

Deno.test("dpmSolverMultistepStep: 境界 — guidance=1 は cond を速度としてそのまま使う", () => {
  // σ0=1・σ1=0 の 1 step 梯子なら更新は x0 = x − 1·v そのもの。
  const sigmas = Float32Array.from([1, 0]);
  const update = dpmSolverMultistepStep({
    sample: Float32Array.from([1, 2, 3]),
    cond: Float32Array.from([0.25, -0.5, 0.125]),
    uncond: undefined,
    guidance: 1,
    sigmas,
    index: 0,
    previousX0: undefined,
  });
  assertEquals([...update.x0], [0.75, 2.5, 2.875]);
  assertEquals([...update.sample], [0.75, 2.5, 2.875]);
});

Deno.test("dpmSolverMultistepStep: guidance≠1 は uncond + scale·(cond − uncond) を速度にする", () => {
  const update = dpmSolverMultistepStep({
    sample: Float32Array.from([0]),
    cond: Float32Array.from([1]),
    uncond: Float32Array.from([0.5]),
    guidance: 4,
    sigmas: Float32Array.from([1, 0]),
    index: 0,
    previousX0: undefined,
  });
  // v = 0.5 + 4·(1 − 0.5) = 2.5 → x0 = 0 − 1·2.5 = −2.5
  assertEquals([...update.x0], [-2.5]);
});

Deno.test("needsUncond: CFG=1 だけが uncond 不要", () => {
  assertEquals(needsUncond(1), false);
  assertEquals(needsUncond(4), true);
  assertEquals(needsUncond(0), true);
});

Deno.test("dpmSolverMultistepStep: uncond の有無が needsUncond と食い違ったら落とす", () => {
  const common = {
    sample: Float32Array.from([0, 0]),
    cond: Float32Array.from([1, 1]),
    sigmas: Float32Array.from([1, 0]),
    index: 0,
    previousX0: undefined,
  };
  assertThrows(
    () => dpmSolverMultistepStep({ ...common, uncond: undefined, guidance: 4 }),
    Error,
    "uncond 側のモデル出力を要求する",
  );
  assertThrows(
    () => dpmSolverMultistepStep({ ...common, uncond: Float32Array.from([2, 2]), guidance: 1 }),
    Error,
    "uncond 分岐を計算しない経路",
  );
});

Deno.test("dpmSolverMultistepStep: 直前 x0 の有無が step 0 かどうかと食い違ったら落とす", () => {
  // 状態の受け渡しが切れたまま黙って 1 次へ落ちるのを禁じる（1 次と 2 次は別の絵になる）。
  const sigmas = Float32Array.from(SERIES_8.sigmas);
  const common = {
    sample: Float32Array.from([0, 0, 0, 0]),
    cond: Float32Array.from([1, 1, 1, 1]),
    uncond: undefined,
    guidance: 1,
    sigmas,
  };
  assertThrows(
    () => dpmSolverMultistepStep({ ...common, index: 3, previousX0: undefined }),
    Error,
    "直前 x0 の有無が食い違う",
  );
  assertThrows(
    () =>
      dpmSolverMultistepStep({
        ...common,
        index: 0,
        previousX0: Float32Array.from([0, 0, 0, 0]),
      }),
    Error,
    "直前 x0 の有無が食い違う",
  );
});

Deno.test("dpmSolverMultistepStep: 梯子と step 添字の受理集合", () => {
  const sample = Float32Array.from([0]);
  const cond = Float32Array.from([1]);
  const common = { sample, cond, uncond: undefined, guidance: 1, previousX0: undefined };
  assertThrows(
    () => dpmSolverMultistepStep({ ...common, sigmas: Float32Array.from([0]), index: 0 }),
    RangeError,
    "step が 1 つも無い",
  );
  // 終端が 0 でない梯子は最終 step の 1 次落ちが成り立たない（`final_sigmas_type="zero"` 前提）。
  assertThrows(
    () => dpmSolverMultistepStep({ ...common, sigmas: Float32Array.from([1, 0.5]), index: 0 }),
    RangeError,
    "0 でない",
  );
  assertThrows(
    () => dpmSolverMultistepStep({ ...common, sigmas: Float32Array.from([1, 0]), index: 1 }),
    RangeError,
    "の整数でない",
  );
  assertThrows(
    () => dpmSolverMultistepStep({ ...common, sigmas: Float32Array.from([1, 0]), index: -1 }),
    RangeError,
    "の整数でない",
  );
});

Deno.test("dpmSolverMultistepStep: 故障注入 — step 順序の入れ替えはパリティ閾値を大きく超える", () => {
  // 検出力の証明。閾値（4e-6）に丸め差を吸わせても、順序を 1 組入れ替えるだけで
  // 最大絶対差は 2.03e+00（実測）まで跳ね、5.7 桁の余裕で赤くなる。
  const flipped = [0, 2, 1, 3, 4, 5, 6, 7];
  const deviation = worstDeviation(SERIES_8, flipped);
  assert(
    deviation > TOLERANCE * 1e4,
    `step 順序の入れ替えが検出できていない（最大差 ${deviation}）`,
  );
});

Deno.test("dpmSolverMultistepStep: 故障注入 — shift の取り違えはパリティ閾値を大きく超える", () => {
  // 梯子そのものの取り違え（`sigmaSchedule` の shift 違い）も同じ閾値で割れることの確認。
  // 中身の係数は全て梯子から導くので、梯子が 1 段ずれれば全 step の係数がずれる。
  // 実測（shift 3 → 4）: 最大絶対差 1.568e-01 = 閾値の 4.6 桁上。
  const wrongLadder: Series = {
    ...SERIES_8,
    sigmas: sigmaLadder(SERIES_8.steps.length, SHIFT + 1),
  };
  const deviation = worstDeviation(wrongLadder);
  assert(
    deviation > TOLERANCE * 1e4,
    `shift の取り違えが検出できていない（最大差 ${deviation}）`,
  );
});

/**
 * 故障注入用に別 shift の梯子を組む（`sigmaSchedule` の写し — テストの中だけで使う）。
 *
 * MUST: ファミリ側（`src/anima/sampler.ts`）から import しない。テスト対象のモジュールは
 * ファミリ非依存で、その依存の向きをテストが先に作ると規約の破れが検出できなくなる。
 */
function sigmaLadder(steps: number, shift: number): number[] {
  const f32 = Math.fround;
  const stop = 1 / steps;
  const delta = (stop - 1) / (steps - 1);
  const ladder: number[] = [];
  for (let index = 0; index < steps; index += 1) {
    const sigma = f32(index === steps - 1 ? stop : index * delta + 1);
    ladder.push(f32(f32(shift * sigma) / f32(f32((shift - 1) * sigma) + 1)));
  }
  ladder.push(0);
  return ladder;
}
