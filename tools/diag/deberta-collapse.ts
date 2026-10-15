// DeBERTa w8a8 の崩壊上限（`packages/runtime/tests/e2e_deberta_w8a8_test.ts` の `COLLAPSE_CEILINGS`）を
// 導き直す手順のうち、GPU の dump と故障注入の 2 段（手順の全体は ADR 0026 追記「導き直しの手順」・
// CPU の段は `tools/export-recipes/deberta/collapse_ceilings.py`）。
//
// 使い方（リポ直下から・<作業 dir> は Python の段と同じ dir）:
//   deno run -A tools/diag/deberta-collapse.ts dump <作業 dir> case0 case1 case2 padded
//     a8 / f32 経路の全出力（output.0..24 を連結した f32）を <作業 dir>/gpu_{a8,f32}.<case>.f32 へ書く。
//     GPU を使う — 学習が使っていないことを先に `tools/diag/gpu-busy.zsh` で確かめる。
//   deno run -A tools/diag/deberta-collapse.ts faults <作業 dir>
//     gpu_a8 の dump を写して壊し、門と同じ判定 `judgeCollapse` と出力ごとの上限 `collapseCeilingOf`
//     （テストファイルの export をそのまま使う）へ通す。参考に gpu_f32 の dump（壊さない）も通す。
//     結果は <作業 dir>/fault_inject.json。

import {
  acquireGpu,
  parseSafetensors,
  prepareContainer,
  type Tensor,
} from "../../packages/runtime/mod.ts";
import { ioTensor } from "../../packages/runtime/tests/helpers/golden-io.ts";
import { openSeriesContainer } from "../../packages/runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../packages/runtime/tests/helpers/series-graphs.ts";
import {
  collapseCeilingOf,
  judgeCollapse,
} from "../../packages/runtime/tests/e2e_deberta_w8a8_test.ts";

const SERIES = new URL("../../outputs/series/deberta-i8/full-24layer/", import.meta.url);
const CASES = ["case0", "case1", "case2", "padded"] as const;
const OUTPUTS = 25;
const FIRST = 2;

const dump = async (work: string, cases: readonly string[]): Promise<void> => {
  const opened = await openSeriesContainer(new URL("model.krm", SERIES));
  const parsed = prepareContainer(opened, seriesGraph("deberta-i8", "full-24layer"));
  for (const caseName of cases) {
    const bytes = await Deno.readFile(new URL(`io-i8a8.${caseName}.safetensors`, SERIES));
    const mirror = parseSafetensors(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    );
    const inputs: Record<string, Tensor> = {};
    for (const spec of parsed.graph.inputs) {
      inputs[spec.name] = ioTensor(mirror, mirror.tensors.get(`input.${spec.name}`)!, spec.dtype);
    }
    for (const [label, options] of [["a8", { linearCompute: "a8" }], ["f32", {}]] as const) {
      const gpu = await acquireGpu();
      const session = await parsed.createContainerSession(gpu, options);
      const outs = await session.run(inputs);
      await session.dispose();
      gpu.destroy();
      const parts = parsed.graph.outputs.map((name) => {
        const data = outs[name].data;
        if (!(data instanceof Float32Array)) throw new Error(`${name} が f32 でない`);
        return data;
      });
      const joined = new Float32Array(parts.reduce((sum, part) => sum + part.length, 0));
      let offset = 0;
      for (const part of parts) {
        joined.set(part, offset);
        offset += part.length;
      }
      await Deno.writeFile(`${work}/gpu_${label}.${caseName}.f32`, new Uint8Array(joined.buffer));
      console.log(caseName, label, joined.length, parsed.graph.outputs.length);
    }
  }
};

type Fault = { readonly name: string; readonly apply: (x: Float32Array) => Float32Array };
const map = (f: (v: number) => number) => (x: Float32Array) => Float32Array.from(x, f);
const one = (value: number) => (x: Float32Array) => {
  const y = x.slice();
  y[Math.floor(y.length / 2)] = value;
  return y;
};
const FAULTS: readonly Fault[] = [
  { name: "none", apply: (x) => x },
  { name: "zero", apply: map(() => 0) },
  { name: "negate", apply: map((v) => -v) },
  { name: "nan1", apply: one(Number.NaN) },
  { name: "posinf1", apply: one(Number.POSITIVE_INFINITY) },
  { name: "x2", apply: map((v) => v * 2) },
  { name: "x1.5", apply: map((v) => v * 1.5) },
  { name: "x1.25", apply: map((v) => v * 1.25) },
];

const faults = async (work: string): Promise<void> => {
  const summary: Record<
    string,
    { red: number; total: number; minRelRms: number; maxRelRms: number }
  > = {};
  /** 出力ごとの赤の数（4 ケースのうち）。 */
  const perOutputRed: Record<string, Record<string, number>> = {};
  const rows: unknown[] = [];
  for (const caseName of CASES) {
    const bytes = await Deno.readFile(new URL(`io-i8a8.${caseName}.safetensors`, SERIES));
    const io = parseSafetensors(bytes.buffer);
    const dumped = new Float32Array((await Deno.readFile(`${work}/gpu_a8.${caseName}.f32`)).buffer);
    const f32 = new Float32Array((await Deno.readFile(`${work}/gpu_f32.${caseName}.f32`)).buffer);
    const n = dumped.length / OUTPUTS;
    if (f32.length !== dumped.length) throw new Error(`${caseName}: f32 dump の長さ ${f32.length}`);
    for (let k = FIRST; k < OUTPUTS; k += 1) {
      const view = io.tensors.get(`output.${k}`)!;
      const golden = new Float32Array(io.buffer, view.byteOffset, view.byteLength / 4);
      if (golden.length !== n) {
        throw new Error(`${caseName} output.${k}: 長さ ${golden.length} vs ${n}`);
      }
      const gpu = dumped.subarray(k * n, (k + 1) * n);
      const ceiling = collapseCeilingOf(`output.${k}`);
      const trials = [
        ...FAULTS.map((fault) => ({ name: fault.name, actual: fault.apply(gpu) })),
        { name: "f32path", actual: f32.subarray(k * n, (k + 1) * n) },
      ];
      for (const fault of trials) {
        const report = judgeCollapse(fault.actual, golden, ceiling);
        const red = perOutputRed[fault.name] ??= {};
        red[`output.${k}`] = (red[`output.${k}`] ?? 0) + (report.pass ? 0 : 1);
        const s = summary[fault.name] ??= {
          red: 0,
          total: 0,
          minRelRms: Number.POSITIVE_INFINITY,
          maxRelRms: 0,
        };
        s.total += 1;
        if (!report.pass) s.red += 1;
        s.minRelRms = Math.min(s.minRelRms, report.relRms);
        s.maxRelRms = Math.max(s.maxRelRms, report.relRms);
        rows.push({ case: caseName, output: k, fault: fault.name, ceiling, ...report });
      }
    }
  }
  for (const name of ["x1.25", "f32path"]) {
    console.log(`${name} 出力ごとの赤（/4）:`, JSON.stringify(perOutputRed[name]));
  }
  for (const [name, s] of Object.entries(summary)) {
    console.log(
      `${name.padEnd(8)} red ${s.red}/${s.total}  relRms min ${s.minRelRms.toFixed(4)} max ${
        s.maxRelRms.toFixed(4)
      }`,
    );
  }
  await Deno.writeTextFile(
    `${work}/fault_inject.json`,
    JSON.stringify({ summary, perOutputRed, rows }, null, 1),
  );
};

const [command, work, ...rest] = Deno.args;
if (work === undefined || !(await Deno.stat(work).catch(() => undefined))?.isDirectory) {
  throw new Error("2 つ目の引数に既存の作業 dir を渡す");
}
if (command === "dump" && rest.length > 0) await dump(work, rest);
else if (command === "faults" && rest.length === 0) await faults(work);
else throw new Error("使い方: dump <作業 dir> <case>... | faults <作業 dir>");
