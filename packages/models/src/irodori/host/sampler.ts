/**
 * Euler サンプラのホスト残置（ADR 0047 決定 4）— t スケジュール・CFG 合成・x の更新。
 *
 * MUST: f32 の丸めを 1 演算ずつ `Math.fround` で踏む（`anima/sampler.ts` の同節）。
 */

const f32 = Math.fround;

/**
 * `[steps+1]` の t 列（**閉形式** `initScale·(1 − i/steps)`）。
 *
 * NOTE: 上流は `(1 − linspace(0,1,n+1))·initScale` で作る。両者はビット一致しない
 * （`linspace` は `start + i·step` を f32 で積むので最終 ulp が割れる点がある — exporter 側の
 * 実測 `tScheduleClosedFormMaxAbs` = 5.96e-8 @ steps=40）。t が入るのは `t_embed` と刻み幅
 * `t_next − t` だけで、この差が最終 latent に効く量は golden の突合閾値の 3 桁下。
 * **閉形式を採るのは添字と値が 1 対 1 で読めるため**で、丸めの実測に基づく制約ではない。
 */
export const tSchedule = (steps: number, initScale: number): Float32Array<ArrayBuffer> => {
  if (!Number.isInteger(steps) || steps < 1) {
    throw new RangeError(`steps ${steps} が 1 以上の整数でない`);
  }
  const schedule = new Float32Array(steps + 1);
  for (let index = 0; index <= steps; index += 1) {
    schedule[index] = f32(initScale * f32(1 - index / steps));
  }
  return schedule;
};

/** CFG の 1 変種（uncond forward の結果とその強さ）。 */
export type CfgVariant = {
  readonly scale: number;
  /** その条件だけを落としたマスクで回した速度場。 */
  readonly velocity: Float32Array<ArrayBuffer>;
};

/**
 * CFG independent の合成 `v = v_cond + Σ_k s_k·(v_cond − v_k)`。
 *
 * MUST: 変種の順（text → speaker → caption）を保つ。加算は結合的でないので、順が変われば
 * 最終桁が変わる（上流はこの順に 1 変種ずつテンソル演算を積む）。
 * MUST: 変種が 0 本でも新しい配列を返す（呼び出し側が Session の出力バッファを掴んだまま
 * 次の run へ入るのを避ける）。
 */
export const combineCfg = (
  cond: Float32Array<ArrayBuffer>,
  variants: readonly CfgVariant[],
): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(cond.length);
  for (const variant of variants) {
    if (variant.velocity.length !== cond.length) {
      throw new Error(
        `CFG 変種の長さ ${variant.velocity.length} が cond ${cond.length} と違う`,
      );
    }
  }
  for (let index = 0; index < cond.length; index += 1) {
    const base = cond[index];
    let value = base;
    for (const variant of variants) {
      value = f32(value + f32(variant.scale * f32(base - variant.velocity[index])));
    }
    out[index] = value;
  }
  return out;
};

/** Euler 更新 `x + v·Δt`（Δt = `t_next − t` で負）。 */
export const eulerStep = (
  x: Float32Array<ArrayBuffer>,
  velocity: Float32Array<ArrayBuffer>,
  deltaT: number,
): Float32Array<ArrayBuffer> => {
  if (velocity.length !== x.length) {
    throw new Error(`速度場の長さ ${velocity.length} が x ${x.length} と違う`);
  }
  const out = new Float32Array(x.length);
  for (let index = 0; index < x.length; index += 1) {
    out[index] = f32(x[index] + f32(deltaT * velocity[index]));
  }
  return out;
};
