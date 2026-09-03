/**
 * Gemma 4 の RoPE（回転位置エンコーディング）の cos / sin 行を**ホスト側で生成**する実装。
 *
 * ## なぜホストが作るのか
 *
 * 配布形は以前、位置 0..maxPosition−1 の cos / sin 表 4 本（層種別 2 × cos / sin）を IR の
 * 初期化子として焼き、グラフ内の `embedding`（行 gather）が `position_ids` で引いていた。
 * 表は位置ごとに独立な純関数 `f(position)` なので、chunk に載る位置のぶんだけをホストが
 * 作って**通常のグラフ入力**として渡せば、表は VRAM にも配布物にも要らず、会話の最大長
 * （`capacity`）は表の行数から解放されて実行時ノブになる。PLE の host gather（`ple.ts`）と
 * 同じ「chunk ぶんだけホストが作る派生入力」の席に嵌まる。
 *
 * ## 式（正本は**この実装**。exporter 側 `tools/export-recipes/gemma4/rope.py` は鏡像）
 *
 * 層種別 `{theta, headDim, rotaryDim}` に対し、`half = headDim / 2`・`rotated = rotaryDim / 2`:
 *
 *     invFreq[i] = theta ** (−(2·i) / headDim)   (0 ≤ i < rotated)
 *     invFreq[i] = 0                              (rotated ≤ i < half — 回さない次元)
 *     angle[j]   = position × invFreq[j mod half] (0 ≤ j < headDim — 前半と後半は同じ角度)
 *     cos[j] = cos(angle[j]) / sin[j] = sin(angle[j])
 *
 * 上流（transformers `Gemma4TextRotaryEmbedding`）の `emb = cat(freqs, freqs)` と同じ並びで、
 * default（sliding: rotaryDim = headDim）と proportional（full: rotaryDim = partial_rotary_factor
 * ぶん）は**同じ 1 式**に畳まれる。attention_scaling は両 rope_type とも 1 なので掛けない。
 *
 * ## 数値契約
 *
 * - 計算は **f64**、格納時に 1 度だけ f32 へ丸める。上流は全経路 f32（角度 `position × invFreq`
 *   を f32 で積むため大きい位置ほど角度が粗い — 131,071 で ULP ≈ 0.008 rad）なので、
 *   **上流の表とはビット同一にならない**（ADR 0034 知見 2 と同じ壁 — torch の f32 三角関数は
 *   SLEEF の 1 ULP 誤差も持つ）。数学的に正確な側がこちらで、突合は位置比例の許容差で行う
 *   （`tests/gemma_rope_test.ts` — 上流モジュールの実出力 fixture との比較 + 故障注入）。
 * - 決定性: f64 の加減乗除は IEEE 754 で engine 非依存。`Math.cos` / `Math.sin` だけは engine 間で
 *   f64 の 1 ULP 差がありうるが、f32 へ丸めた後に残る確率は要素あたり約 2⁻²⁹（128K × 1,536 要素の
 *   全表でも期待 0.4 要素）で、GPU 側の 1 ULP 群（known-issues の Metal 節）と同じ扱いにする。
 * - 回さない次元（`invFreq = 0`）は角度 0 なので cos = 1 / sin = 0 が**厳密**に出る。
 *
 * MUST: pad 行（`position = 0`）も通常行と同じ式で埋める（cos = 1 / sin = 0）。pad 行の値は
 * 契約上無意味（ADR 0066 追記 6）だが、未初期化のまま渡すと NaN が pad 行の attention に入り
 * 「空行 → 厳密 0」の構造保証（ADR 0067 決定 6）の外側で NaN 分類が動きうる。
 */

import type { Tensor } from "@karume/runtime";

/** 層種別の名前（上流 `layer_types` の綴りそのまま — グラフ入力名の一部になる）。 */
export const GEMMA4_ROPE_LAYER_TYPES = ["sliding_attention", "full_attention"] as const;

export type Gemma4RopeLayerType = (typeof GEMMA4_ROPE_LAYER_TYPES)[number];

/** 層種別 1 つぶんの RoPE パラメータ（配布形 `pipelineConfig.rope.<layerType>` が宣言する）。 */
export type Gemma4RopeLayerSpec = {
  /** 周波数の底（上流 `rope_theta`）。 */
  readonly theta: number;
  /** head 次元 `D`（= 生成する行の幅）。 */
  readonly headDim: number;
  /** 回す次元数（default は `headDim`・proportional は `2 · int(factor · headDim / 2)`）。偶数。 */
  readonly rotaryDim: number;
};

/** 層種別 2 本ぶん。 */
export type Gemma4RopeSpec = {
  readonly [layerType in Gemma4RopeLayerType]: Gemma4RopeLayerSpec;
};

/** 生成する cos / sin の 2 部。 */
export const GEMMA4_ROPE_PARTS = ["cos", "sin"] as const;

export type Gemma4RopePart = (typeof GEMMA4_ROPE_PARTS)[number];

/**
 * グラフ入力名（exporter `tools/export-recipes/gemma4/export_decode.py` と**同じ綴り**）。
 *
 * MUST: ここが唯一の生成箇所（配布側の門 `GEMMA4_GRAPH_INPUTS` と TS の派生入力名は同じ文字列
 * から出す）。
 */
export const gemma4RopeInputName = (layerType: Gemma4RopeLayerType, part: Gemma4RopePart): string =>
  `rope_${layerType}_${part}`;

/** 4 本のグラフ入力名（層種別 × 部の固定順）。 */
export const gemma4RopeInputNames = (): string[] =>
  GEMMA4_ROPE_LAYER_TYPES.flatMap((layerType) =>
    GEMMA4_ROPE_PARTS.map((part) => gemma4RopeInputName(layerType, part))
  );

/**
 * 層種別パラメータの値域門。
 *
 * MUST: `rotaryDim` は偶数かつ `2 ≤ rotaryDim ≤ headDim`、`headDim` は偶数。奇数だと
 * 「前半 = 後半」の並びが崩れ、上流と別の表を黙って作る。`theta` は正の有限値
 * （`theta ** 負` が 0 / Inf に落ちる形を弾く）。
 */
export const assertGemma4RopeLayerSpec = (where: string, spec: Gemma4RopeLayerSpec): void => {
  const { theta, headDim, rotaryDim } = spec;
  if (!Number.isFinite(theta) || theta <= 0) {
    throw new Error(`${where}: theta は正の有限値（${theta}）`);
  }
  if (!Number.isInteger(headDim) || headDim < 2 || headDim % 2 !== 0) {
    throw new Error(`${where}: headDim は 2 以上の偶数（${headDim}）`);
  }
  if (!Number.isInteger(rotaryDim) || rotaryDim < 2 || rotaryDim % 2 !== 0 || rotaryDim > headDim) {
    throw new Error(
      `${where}: rotaryDim は 2 以上 headDim 以下の偶数（${rotaryDim} / headDim ${headDim}）`,
    );
  }
};

export const assertGemma4RopeSpec = (where: string, spec: Gemma4RopeSpec): void => {
  for (const layerType of GEMMA4_ROPE_LAYER_TYPES) {
    assertGemma4RopeLayerSpec(`${where}.${layerType}`, spec[layerType]);
  }
};

/**
 * 逆周波数 `invFreq[0 .. headDim/2)`（f64）。回さない次元は 0。
 *
 * NOTE: `theta ** x` は `Math.pow` の f64。上流は f32 で `1 / base ** t` を計算するので相対 1e-7
 * 級の差があるが、これは意図した差（上の数値契約）。
 */
export const gemma4RopeInverseFrequencies = (
  spec: Gemma4RopeLayerSpec,
): Float64Array<ArrayBuffer> => {
  assertGemma4RopeLayerSpec("gemma4 rope", spec);
  const half = spec.headDim / 2;
  const rotated = spec.rotaryDim / 2;
  const invFreq = new Float64Array(half);
  for (let i = 0; i < rotated; i += 1) {
    invFreq[i] = Math.pow(spec.theta, -(2 * i) / spec.headDim);
  }
  return invFreq;
};

/** `[rows, headDim]` 行優先の cos / sin（f32）。 */
export type Gemma4RopeRows = {
  readonly cos: Float32Array<ArrayBuffer>;
  readonly sin: Float32Array<ArrayBuffer>;
};

/**
 * 位置列 → cos / sin 行（`[positions.length, headDim]`）。
 *
 * MUST: 位置は非負整数（u32 の論理位置）。負や非整数は式が定義されないので fail loudly。
 */
export const gemma4RopeRows = (
  spec: Gemma4RopeLayerSpec,
  positions: ArrayLike<number>,
): Gemma4RopeRows => {
  const invFreq = gemma4RopeInverseFrequencies(spec);
  const width = spec.headDim;
  const half = width / 2;
  const rows = positions.length;
  const cos = new Float32Array(rows * width);
  const sin = new Float32Array(rows * width);
  for (let row = 0; row < rows; row += 1) {
    const position = positions[row];
    if (!Number.isInteger(position) || position < 0) {
      throw new Error(`gemma4 rope: 位置は非負整数（row ${row}: ${position}）`);
    }
    const base = row * width;
    for (let i = 0; i < half; i += 1) {
      const angle = position * invFreq[i];
      // f64 → f32 の丸めは代入時の 1 回だけ（前半 j = i と後半 j = i + half は同じ値）
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      cos[base + i] = c;
      cos[base + half + i] = c;
      sin[base + i] = s;
      sin[base + half + i] = s;
    }
  }
  return { cos, sin };
};

/**
 * 物理 chunk 1 本ぶんの位置列 → グラフ入力 4 本（`[1, rows, headDim]` f32）。
 *
 * `positions` は物理行数ぶん（pad 行は 0 — `sequence.ts` が `input_ids` と同じ規約で埋める）。
 */
export const gemma4RopeInputs = (
  spec: Gemma4RopeSpec,
  positions: ArrayLike<number>,
): Record<string, Tensor> => {
  const inputs: Record<string, Tensor> = {};
  for (const layerType of GEMMA4_ROPE_LAYER_TYPES) {
    const layer = spec[layerType];
    const rows = gemma4RopeRows(layer, positions);
    const shape = [1, positions.length, layer.headDim];
    inputs[gemma4RopeInputName(layerType, "cos")] = { dtype: "f32", shape, data: rows.cos };
    inputs[gemma4RopeInputName(layerType, "sin")] = { dtype: "f32", shape, data: rows.sin };
  }
  return inputs;
};
