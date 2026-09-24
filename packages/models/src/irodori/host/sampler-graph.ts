/**
 * Euler サンプラの GPU 側 2 グラフ（**ホストで組む** IR v2 の宣言 — exporter は通さない）。
 *
 * DiT ループを 1 batch に束ねる（H-5）と、CFG 合成と Euler 更新だけがホストに残って
 * 「forward ごとに readback → 再アップロード」を強いる。この 2 演算は要素ごとの
 * add / sub / mul でしか無いので、配布形に足すのではなく**その場で組んだ小さな IR**を
 * 別 Session で回し、潜在は常駐テンソルのまま GPU に置いたままにする。
 *
 * ## MUST: 組んだ宣言は `parseIrDeclarationValue` に通す
 *
 * ここは exporter も容器の読み手も経由しないので、通さないとグラフ単体で決まる規則
 * （SSA・前方参照・トポロジカル順・`requires.ops` と実使用 op の一致・未知キー）が 1 つも
 * 検査されない。合流層（宣言 × 供給）と `prepareContainer`（capability + op 契約）はその
 * 種の壊れ方を見ないので、編集の誤りは Session 構築か実行まで落ちてこない。
 *
 * MUST NOT: 下のリテラルに省略可能な `states` を書き足さない。書かないことで、リテラルは
 * {@link IrDeclaration} として型が付かず（`states` が欠ける）、`parseIrDeclarationValue` の
 * 戻り値だけがこの関数の返り値になれる。包みを外した編集は `deno check` で落ちる。
 *
 * ## MUST: 演算の結合順・引数順は TS 正本（`sampler.ts`）と 1 演算ずつ一致させる
 *
 * f32 の加算は非結合なので、積む順が変われば最終桁が変わる。正本は
 * {@link combineCfg}（`f32(value + f32(scale * f32(base − velocity)))` を変種順に畳む）と
 * {@link eulerStep}（`f32(x + f32(deltaT * velocity))`）で、下のノード列はその写しである
 * （`sub` → `mul` → `add` / `mul` → `add`）。ホスト側の関数は数値の正本として残してある。
 *
 * ## NOTE: 記号次元を使わない（S は生成ごとに確定している）
 *
 * 呼ばれるのは S が決まった後なので、グラフは具体次元で組む。常駐テンソルは shape を持たない
 * （バイト列と大きさだけ）ため記号の束縛源になれず、記号を置くと束縛を毎 enqueue 渡す羽目に
 * なる。具体次元なら束縛は空のままで、実体の大きさとの突合は runtime が宣言 shape に対して
 * 行う。
 */

import { type IrDeclaration, parseIrDeclarationValue } from "@karume/runtime";

/** 潜在 1 本の shape（DiT の `x_t` / 速度場と同じ `[1, S, latentDim]`）。 */
const latentShape = (frames: number, latentDim: number): readonly number[] => [
  1,
  frames,
  latentDim,
];

/** 入力名（呼び出し側と 1 箇所で共有する）。 */
export const COMBINE_INPUTS = {
  /** 直前までの合成結果（k = 0 では cond そのもの）。 */
  accumulator: "acc_in",
  /** cond 側の速度場（差の基準は**常に** cond）。 */
  cond: "cond",
  /** その変種の uncond 速度場。 */
  variant: "v_k",
  /** その変種の強さ（shape `[1]` — 右詰め broadcast で全要素へ効く）。 */
  scale: "s",
} as const;

/** {@link combineGraph} の出力名。 */
export const COMBINE_OUTPUT = "acc_out";

export const EULER_INPUTS = {
  /** 現在の潜在。 */
  x: "x",
  /** 速度場（CFG 有効 step は合成後・無効 step は cond そのもの）。 */
  velocity: "v",
  /** 刻み幅 `t_next − t`（shape `[1]`・負）。 */
  deltaT: "dt",
} as const;

/** {@link eulerGraph} の出力名。 */
export const EULER_OUTPUT = "x_next";

/**
 * CFG 合成 1 変種ぶん `acc_out = acc_in + s·(cond − v_k)`（3 ノード）。
 *
 * 変種は 1 本ずつこのグラフに通す — 正本 {@link combineCfg} が変種順に 1 本ずつ畳むのと
 * 同じ積み方にするため。強さ `s` は shape `[1]` の入力で、値ごとにグラフを組み直さない。
 */
export const combineGraph = (frames: number, latentDim: number): IrDeclaration => {
  const rows = latentShape(frames, latentDim);
  return parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["sub", "mul", "add"] },
    symbols: [],
    inputs: [
      { name: COMBINE_INPUTS.accumulator, dtype: "f32", shape: rows },
      { name: COMBINE_INPUTS.cond, dtype: "f32", shape: rows },
      { name: COMBINE_INPUTS.variant, dtype: "f32", shape: rows },
      { name: COMBINE_INPUTS.scale, dtype: "f32", shape: [1] },
    ],
    outputs: [COMBINE_OUTPUT],
    initializers: {},
    values: {
      diff: { dtype: "f32", shape: rows },
      scaled: { dtype: "f32", shape: rows },
      [COMBINE_OUTPUT]: { dtype: "f32", shape: rows },
    },
    nodes: [
      { op: "sub", ins: [COMBINE_INPUTS.cond, COMBINE_INPUTS.variant], outs: ["diff"], attrs: {} },
      { op: "mul", ins: [COMBINE_INPUTS.scale, "diff"], outs: ["scaled"], attrs: {} },
      {
        op: "add",
        ins: [COMBINE_INPUTS.accumulator, "scaled"],
        outs: [COMBINE_OUTPUT],
        attrs: {},
      },
    ],
  });
};

/** Euler 更新 `x_next = x + dt·v`（2 ノード — 正本 {@link eulerStep} と同順）。 */
export const eulerGraph = (frames: number, latentDim: number): IrDeclaration => {
  const rows = latentShape(frames, latentDim);
  return parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["mul", "add"] },
    symbols: [],
    inputs: [
      { name: EULER_INPUTS.x, dtype: "f32", shape: rows },
      { name: EULER_INPUTS.velocity, dtype: "f32", shape: rows },
      { name: EULER_INPUTS.deltaT, dtype: "f32", shape: [1] },
    ],
    outputs: [EULER_OUTPUT],
    initializers: {},
    values: {
      step: { dtype: "f32", shape: rows },
      [EULER_OUTPUT]: { dtype: "f32", shape: rows },
    },
    nodes: [
      { op: "mul", ins: [EULER_INPUTS.deltaT, EULER_INPUTS.velocity], outs: ["step"], attrs: {} },
      { op: "add", ins: [EULER_INPUTS.x, "step"], outs: [EULER_OUTPUT], attrs: {} },
    ],
  });
};
