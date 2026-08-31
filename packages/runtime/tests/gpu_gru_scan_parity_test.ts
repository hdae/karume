// gru_scan / gru_scan_reverse（ADR 0056）の**ビット同一の門**。
//
// 同じ入力に対し、
//
//   ① 既存 GPU op 列で組んだ「torch の GRU 分解を逐語に写した」参照グラフ
//      （`linear → slice → add → sigmoid / mul → add → tanh / sub → mul → add` を T 回）
//   ② `gru_scan` op 1 ノード
//
// を**同じ実 GPU**で流し、出力を **f32 のビット列**で突き合わせる。allclose ではなく
// `Uint32Array` の完全一致で見るのが要点で、tolerance に隠れる丸め列の変化（隠れ側縮約が
// `linear` の GEMM からずれる / 融合側だけ fma へ縮約される / 更新式を `(1−z)·n + z·h` に
// 書き換える）はここでしか検出できない。
//
// **この op は attention（ADR 0023 決定 2）と同じ「本物の A/B オラクル」を持つ**。T を固定
// すれば torch の分解グラフがそのまま完全な参照実装になるので、新規原子（deform_conv2d）が
// 逃げた退化オラクルは要らない。
//
// MUST: 恒真化しないこと。①② は**別々のグラフ**を別々の Session で走らせ、さらに
//   - 出力が定数列でないこと（時間方向に値が動いていること）
//   - `gru_scan` と `gru_scan_reverse` の出力が**違う**こと（方向が効いていること）
// を毎ケース確かめる（走査方向を無視した実装は前者だけでは緑のまま通る）。

import { assert, assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import type { GraphJson } from "./helpers/format.ts";
import { fill, type FilledTensor, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

type NodeJson = GraphJson["nodes"][number];
type ValueSpec = GraphJson["values"][string];

type Geometry = {
  readonly time: number;
  readonly batch: number;
  readonly hidden: number;
};

/**
 * torch の GRU 分解を既存語彙へ逐語に写した参照グラフ（T は静的に展開する）。
 *
 * | torch の分解ノード列              | ここ                                        |
 * | --------------------------------- | ------------------------------------------- |
 * | `linear(h, w_hh, b_hh)`           | `linear`（bias は GEMM の epilogue = last） |
 * | `sigmoid(add(gh_r, gi_r))`        | `add` → `sigmoid`                           |
 * | `tanh(add(gi_n, mul(gh_n, r)))`   | `mul` → `add` → `tanh`                      |
 * | `add(mul(sub(h, n), z), n)`       | `sub` → `mul` → `add`                       |
 * | 各ステップの h を時間方向へ積む   | `reshape` → `cat`(dim 0)                    |
 *
 * MUST: 引数の順序を入れ替えない。有限 f32 では加算・乗算は可換だが、**この列が丸め列の
 * 正本**なので、順序を変えると「何と一致させたいのか」が変わる（ADR 0056 決定 3）。
 * MUST: 逆方向は**走査順だけ**を反転し、`cat` へ渡す並びは順方向の時間順のままにする
 * （`flip` を挟まない = op が畳んでいる形と同じ意味論）。
 */
const decomposedGraph = (geometry: Geometry, reverse: boolean): GraphJson => {
  const { time, batch, hidden } = geometry;
  const gates = 3 * hidden;
  const nodes: NodeJson[] = [];
  const values: Record<string, ValueSpec> = {};
  const declare = (name: string, shape: readonly number[]): string => {
    values[name] = { dtype: "f32", shape: [...shape] };
    return name;
  };
  /** `[N, 3H]` のゲート束から 1 ゲート（`0 = r / 1 = z / 2 = n`）を切り出す。 */
  const gate = (source: string, name: string, index: number): string => {
    nodes.push({
      op: "slice",
      ins: [source],
      outs: [declare(name, [batch, hidden])],
      attrs: { dim: 1, start: index * hidden, end: (index + 1) * hidden },
    });
    return name;
  };
  const unary = (op: string, source: string, name: string): string => {
    nodes.push({ op, ins: [source], outs: [declare(name, [batch, hidden])], attrs: {} });
    return name;
  };
  const binary = (op: string, a: string, b: string, name: string): string => {
    nodes.push({ op, ins: [a, b], outs: [declare(name, [batch, hidden])], attrs: {} });
    return name;
  };

  // 走査順（逆方向は t が降順）。書き出し先の時間添字は常に t そのもの。
  const order = Array.from({ length: time }, (_unused, step) => reverse ? time - 1 - step : step);
  const stacked: string[] = [];
  let state = "h0";
  for (const t of order) {
    const gh = declare(`gh${t}`, [batch, gates]);
    nodes.push({ op: "linear", ins: [state, "w", "b"], outs: [gh], attrs: {} });
    nodes.push({
      op: "slice",
      ins: ["gi"],
      outs: [declare(`giStep${t}`, [1, batch, gates])],
      attrs: { dim: 0, start: t, end: t + 1 },
    });
    const inputGates = declare(`giFlat${t}`, [batch, gates]);
    nodes.push({ op: "reshape", ins: [`giStep${t}`], outs: [inputGates], attrs: {} });

    const reset = unary(
      "sigmoid",
      binary("add", gate(gh, `ghR${t}`, 0), gate(inputGates, `giR${t}`, 0), `sumR${t}`),
      `r${t}`,
    );
    const update = unary(
      "sigmoid",
      binary("add", gate(gh, `ghZ${t}`, 1), gate(inputGates, `giZ${t}`, 1), `sumZ${t}`),
      `z${t}`,
    );
    const candidate = unary(
      "tanh",
      binary(
        "add",
        gate(inputGates, `giN${t}`, 2),
        binary("mul", gate(gh, `ghN${t}`, 2), reset, `prodN${t}`),
        `sumN${t}`,
      ),
      `n${t}`,
    );
    const next = binary(
      "add",
      binary("mul", binary("sub", state, candidate, `diff${t}`), update, `decay${t}`),
      candidate,
      // MUST: 入力名 `h0` と衝突させない（t = 0 の状態を `h0` と綴ると SSA 違反になる）
      `state${t}`,
    );
    nodes.push({
      op: "reshape",
      ins: [next],
      outs: [declare(`step${t}`, [1, batch, hidden])],
      attrs: {},
    });
    stacked[t] = `step${t}`;
    state = next;
  }
  nodes.push({
    op: "cat",
    ins: stacked,
    outs: [declare("y", [time, batch, hidden])],
    attrs: { dim: 0 },
  });

  return {
    format: "karume-ir",
    version: 1,
    requires: {
      ops: ["linear", "slice", "reshape", "add", "sub", "mul", "sigmoid", "tanh", "cat"],
    },
    symbols: [],
    inputs: [
      { name: "gi", dtype: "f32", shape: [time, batch, gates] },
      { name: "h0", dtype: "f32", shape: [batch, hidden] },
      { name: "w", dtype: "f32", shape: [gates, hidden] },
      { name: "b", dtype: "f32", shape: [gates] },
    ],
    outputs: ["y"],
    initializers: {},
    values,
    nodes,
  };
};

const run = async (
  gpu: GpuContext,
  graph: GraphJson,
  inputs: Readonly<Record<string, FilledTensor>>,
): Promise<Tensor> => {
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
  try {
    return (await session.run(inputs))["y"];
  } finally {
    await session.dispose();
  }
};

/** f32 のビット列（末尾 1 ulp の差も `0.0` と `-0.0` の差も見える形）。 */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

// 決定的な列（乱数は使わない — 失敗が再現しないため）。値域はゲートが飽和しきらない範囲に
// 採る: sigmoid が 0 / 1 に張り付くと更新式の丸め差が値に出ず、門が恒真化する。
const GATE_INPUT = (i: number): number => (((i * 7) % 23) - 11) * 0.21;
const STATE_INIT = (i: number): number => (((i * 5) % 17) - 8) * 0.13;
const WEIGHT = (i: number): number => (((i * 11) % 29) - 14) * 0.037;
const BIAS = (i: number): number => (((i * 3) % 13) - 6) * 0.11;

const CASES: readonly Geometry[] = [
  // 最小形（T が 2 = 状態が 1 度は運ばれる最小）
  { time: 2, batch: 1, hidden: 3 },
  // vowel-detector の隠れ幅（H = 128）を実長で踏む
  { time: 4, batch: 1, hidden: 128 },
  // H が workgroup 幅ちょうど（1 lane = 1 隠れユニットの上限 — 端の lane が effective）
  { time: 3, batch: 1, hidden: 256 },
  // バッチ > 1（workgroup 間で状態が混ざらないことの門）
  { time: 5, batch: 3, hidden: 7 },
  // 縮約長が GEMM の K タイル（16）の倍数でない形（端タイルのゼロ詰めが効く）
  { time: 3, batch: 2, hidden: 19 },
];

const inputsFor = (geometry: Geometry): Record<string, FilledTensor> => {
  const { time, batch, hidden } = geometry;
  const gates = 3 * hidden;
  return {
    gi: fill([time, batch, gates], GATE_INPUT),
    h0: fill([batch, hidden], STATE_INIT),
    w: fill([gates, hidden], WEIGHT),
    b: fill([gates], BIAS),
  };
};

/**
 * 飽和域の門（gru_scan v2 = tanh_stable 共有の回帰検出）。n ゲート入力を飽和域
 * （|x| > 9.5）〜 exp(2x) の f32 オーバーフロー域（|x| > 44.36）に置く。素の組込 `tanh` へ
 * 戻す退行は、IEEE 忠実な実装ではビット同一のまま緑だが、`(exp(2x)−1)/(exp(2x)+1)` で
 * 計算する実装（Metal fast-math）では下の Number.isFinite 門が赤くなる。r / z ゲートは
 * 従来の値域のまま（sigmoid が 0 / 1 に張り付くと更新式の丸め差が値に出ない）。
 */
const SATURATED_N_GATE = [12, -15, 50, -50, 120, -120, 9.4, -9.6] as const;

const saturatedInputsFor = (geometry: Geometry): Record<string, FilledTensor> => {
  const { time, batch, hidden } = geometry;
  const gates = 3 * hidden;
  const gi = (i: number): number =>
    (i % gates) >= 2 * hidden ? SATURATED_N_GATE[i % SATURATED_N_GATE.length] : GATE_INPUT(i);
  return {
    gi: fill([time, batch, gates], gi),
    h0: fill([batch, hidden], STATE_INIT),
    w: fill([gates, hidden], WEIGHT),
    b: fill([gates], BIAS),
  };
};

const scanGraph = (op: string, geometry: Geometry): GraphJson => {
  const { time, batch, hidden } = geometry;
  const gates = 3 * hidden;
  return singleOpGraph(
    op,
    [[time, batch, gates], [batch, hidden], [gates, hidden], [gates]],
    [[time, batch, hidden]],
  );
};

/** 単一ノードグラフの入力名（`x0`…）へ付け替える。 */
const asSingleOpInputs = (
  named: Readonly<Record<string, FilledTensor>>,
): Record<string, FilledTensor> => ({
  x0: named["gi"],
  x1: named["h0"],
  x2: named["w"],
  x3: named["b"],
});

for (const op of ["gru_scan", "gru_scan_reverse"] as const) {
  Deno.test({
    name:
      `${op} 1 ノードの出力が torch 分解の逐語（linear + elementwise 列）と**ビット単位で一致**する（実 GPU）`,
    ignore: !GPU_AVAILABLE,
    fn: async () => {
      const gpu = await acquireGpu();
      try {
        const parityCases = [
          ...CASES.map((geometry) => ({ geometry, named: inputsFor(geometry), tag: "" })),
          // 飽和域ケース（H = 実測形の 128 — 候補ゲートの引数が ±9.5 を跨ぐ）
          {
            geometry: { time: 3, batch: 2, hidden: 128 },
            named: saturatedInputsFor({ time: 3, batch: 2, hidden: 128 }),
            tag: " 飽和域",
          },
        ];
        for (const { geometry, named, tag } of parityCases) {
          const label = `${op} T${geometry.time} N${geometry.batch} H${geometry.hidden}${tag}`;
          const fused = await run(gpu, scanGraph(op, geometry), asSingleOpInputs(named));
          const decomposed = await run(
            gpu,
            decomposedGraph(geometry, op === "gru_scan_reverse"),
            named,
          );

          assertEquals(fused.shape, decomposed.shape, label);
          const a = bits(fused);
          const b = bits(decomposed);
          const mismatches: number[] = [];
          for (let i = 0; i < a.length; i += 1) {
            if (a[i] !== b[i]) mismatches.push(i);
          }
          assertEquals(
            mismatches.length,
            0,
            `${label}: ${mismatches.length} / ${a.length} 要素がビット不一致（先頭 ${
              mismatches.slice(0, 4).map((i) =>
                `#${i} fused=0x${a[i].toString(16)} decomposed=0x${b[i].toString(16)}`
              ).join(" / ")
            }）`,
          );

          // 恒真化の確認: 出力が定数列でない（分解側が全て同じ値なら一致は何も語らない）
          const values = fused.data as Float32Array;
          assert(values.every(Number.isFinite), `${label}: 非有限値がある`);
          assert(new Set(values).size > 1, `${label}: 出力が定数列（門が恒真化している）`);
        }
      } finally {
        gpu.destroy();
      }
    },
  });
}

// 走査方向が本当に効いていることの門。逆方向を「順方向と同じ走査」で実装しても上の parity は
// **両側が同じ誤りを持つ**ので緑になりうる（参照グラフも同じ関数から順序を受け取るため）。
Deno.test({
  name: "gru_scan と gru_scan_reverse は同じ入力で違う出力を出す（走査方向が効いている）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const geometry of CASES) {
        const named = asSingleOpInputs(inputsFor(geometry));
        const forward = await run(gpu, scanGraph("gru_scan", geometry), named);
        const reverse = await run(gpu, scanGraph("gru_scan_reverse", geometry), named);
        const a = bits(forward);
        const b = bits(reverse);
        let same = 0;
        for (let i = 0; i < a.length; i += 1) {
          if (a[i] === b[i]) same += 1;
        }
        assert(
          same < a.length,
          `T${geometry.time} N${geometry.batch} H${geometry.hidden}: 順方向と逆方向の出力が完全一致（走査方向が無視されている）`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});
