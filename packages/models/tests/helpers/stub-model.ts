// 構築ゲート（グラフ宣言と `pipelineConfig` の突合）を GPU も実資産も無しで叩くための
// 最小 `KarumeModel`。
//
// 門が読むのは `graph` の**宣言**（symbols / inputs / outputs / values）だけで、テンソル実体
// にもノード列にも触らない。ここで組むのはその宣言だけを持つ形で、IR パーサ（`openModel`）は
// 通していない — 通すと「壊れた宣言」がパーサ側で先に落ち、門そのものを踏めなくなる。
//
// MUST: 正常系の宣言はここで作らず、各テストが**正しい形を 1 つ書いてから 1 点だけ壊す**
// （`helpers/safetensors.ts` と同じ流儀）。共有の既定値を置くと、壊し方が既定へ吸われる。

import type { KarumeModel } from "@karume/runtime";

/** 宣言できる次元（記号次元は `format/dims.ts` の正準表記の文字列）。 */
export type StubDim = number | string;

export type StubGraph = {
  readonly symbols?: readonly string[];
  readonly inputs: readonly { readonly name: string; readonly shape: readonly StubDim[] }[];
  readonly outputs: readonly string[];
  /** 値名 → 宣言 shape（門が見るのは `outputs[0]` の 1 本だけ）。 */
  readonly values: Readonly<Record<string, readonly StubDim[]>>;
};

export const stubModel = (graph: StubGraph): KarumeModel => ({
  graph: {
    format: "karume-ir",
    version: 1,
    requires: { ops: [] },
    symbols: [...(graph.symbols ?? [])],
    inputs: graph.inputs.map((input) => ({
      name: input.name,
      dtype: "f32" as const,
      shape: [...input.shape],
    })),
    outputs: [...graph.outputs],
    initializers: {},
    values: Object.fromEntries(
      Object.entries(graph.values).map(([name, shape]) => [
        name,
        { dtype: "f32" as const, shape: [...shape] },
      ]),
    ),
    states: {},
    nodes: [],
  },
  file: { buffer: new ArrayBuffer(0), metadata: new Map(), tensors: new Map() },
});
