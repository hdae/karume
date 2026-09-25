/**
 * マイクロベンチの計測規約（`opbench single` / `graph` が共有する — 実装として内蔵する）。
 *
 * 規約の出典 = docs/research/2026-08-10-kernel-variant-sweep.md §5（散文で残っていたものをここへ
 * 移す）。**クロックの張り付けは軽い dispatch の反復では起きない** — M=1 の GEMV（48 workgroup）を
 * 80ms 積んでも RTX 3080 Ti は P8 / P5（SM 210〜900MHz・mem 405〜810MHz）に留まり、同じ形が P0 の
 * 20 倍遅く測れた（2026-09-04 実測・nvidia-smi 並走）。そのため計測の前と各 round の間に**重い filler**
 * （{@link Heater}）を挟んで P0 を維持する。規約:
 * 1. 1 タイムドパスが ≈80ms になるまで同一 dispatch を積む（duty cycle を保ってクロックを
 *    張り付かせる — アイドルから測ると同じ変種が 2 倍揺れた）
 * 2. 代表値は **min**（熱ドリフトは min で吸う。中央値は decode 壁時計など E2E 側の規約）
 * 3. 絶対 ms は別日・別機と比較しない — 同一リグ内の比だけが有効
 *
 * timing（GPU timestamp・1 dispatch = 1 pass の ns）と wall（submit → fence 決着の壁時計）は
 * **同じ量を測っていない**（timing はフェンス構造を変える）。両方を 1 レコードの別欄に持ち、
 * 割った倍率を主張しない — 混ぜて倍率を出した過去の誤りを再生産しないための規律。
 */

import type { RunInputs, Session } from "../../packages/runtime/mod.ts";

/** 1 タイムドパスの目標長（ms）。研究 §5-1 の実測値をそのまま既定にする。 */
export const TARGET_PASS_MS = 80;

/** 反復の上限。出力の readback / メモリを線形に増やすので、目標長に届かなくても打ち切る。 */
export const MAX_REPS = 1024;

/**
 * 計測前にクロックを張り付かせる空回しの下限（GPU 実時間の累計 ns）。研究 §5-2 の
 * 「対の前にメモリクロックが張り付くまで空回し」を、nvidia-smi に依らず時間で置き換えたもの。
 * アイドルから 1 パス ≈80ms では張り付かない（同一変種が 2 倍揺れた実測）。
 */
export const WARMUP_NS = 500e6;

/** 空回しの回数下限（累計時間が先に満ちても、パイプライン生成直後の 1 回だけでは終えない）。 */
export const WARMUP_MIN_RUNS = 3;

/**
 * クロックを張り付かせる filler。`run()` は重い dispatch 列を 1 回流して壁時計 ms を返す
 * （計測対象ではない — 計測の直前と round の間に挟むだけ）。実体は single.ts の f32 linear。
 */
export type Heater = {
  run(): Promise<number>;
};

/**
 * filler を回してクロックを張り付かせる。「連続 3 回の壁時計が最小値の +5% 以内」を張り付いた
 * 印にし、累計 {@link WARMUP_NS} 以上・{@link WARMUP_MIN_RUNS} 回以上を下限、64 回で打ち切る
 * （nvidia-smi に依らない — 張り付いたかは最終的に計測値の再現性で判断する）。
 */
export const pinClocks = async (heater: Heater): Promise<{ runs: number; ms: number }> => {
  let total = 0;
  let min = Number.POSITIVE_INFINITY;
  let stable = 0;
  let runs = 0;
  for (; runs < 64; runs += 1) {
    const ms = await heater.run();
    total += ms;
    if (ms <= min * 1.05) stable += 1;
    else stable = 0;
    min = Math.min(min, ms);
    if (runs + 1 >= WARMUP_MIN_RUNS && total * 1e6 >= WARMUP_NS && stable >= 3) break;
  }
  return { runs, ms: total };
};

/** 代表値を取る回数（min を採るので偶奇や対は要らない — 同一ケースの反復）。 */
export const ROUNDS = 5;

/**
 * 1 dispatch の推定 ns から、目標長に足りる反復数を決める（1 以上・上限で打ち切り）。
 * 推定が 0 / 非有限のときは上限（測れないほど速い = 積めるだけ積む）。
 */
export const calibrateReps = (
  nsPerDispatch: number,
  targetMs: number = TARGET_PASS_MS,
  maxReps: number = MAX_REPS,
): number => {
  if (!Number.isFinite(nsPerDispatch) || nsPerDispatch <= 0) return maxReps;
  const reps = Math.ceil((targetMs * 1e6) / nsPerDispatch);
  return Math.max(1, Math.min(maxReps, reps));
};

/** 1 回の timing 計測（1 run = 反復ぶんの dispatch を積んだグラフ）。 */
export type TimingSample = {
  /** run 全体の GPU 実時間（ns・全 dispatch の pass 合計）。 */
  readonly totalNs: number;
  readonly dispatchCount: number;
  /** その run で立ったパイプラインキー（ns 降順）。 */
  readonly keys: readonly string[];
  /** 非単調 timestamp を 0 に丸めたサンプル数（0 でないなら読みが疑わしい — 黙って捨てない）。 */
  readonly clampedNegativeSamples: number;
  /** 同じ run の壁時計（ns・`run` の呼び出しから決着まで）。{@link timingWallWarning} が読む。 */
  readonly wallNs: number;
};

/**
 * gpuTiming 有効 device で 1 run を測る。`lastRunTiming` が無い（計測無効）なら fail loudly —
 * 空表で「測れた」と読ませない。
 */
export const sampleTiming = async (session: Session, inputs: RunInputs): Promise<TimingSample> => {
  const started = performance.now();
  await session.run(inputs);
  const wallNs = (performance.now() - started) * 1e6;
  const timing = session.diagnostics().lastRunTiming;
  if (timing === undefined) {
    throw new Error("lastRunTiming が無い — acquireGpu({ gpuTiming: true }) の device で測ること");
  }
  return {
    totalNs: timing.totalNs,
    dispatchCount: timing.dispatchCount,
    keys: timing.entries.map((entry) => entry.key),
    clampedNegativeSamples: timing.clampedNegativeSamples,
    wallNs,
  };
};

/** GPU 実時間 / 壁時計 の上限。GPU の pass は run の壁時計の内側で走るので、1 を超えない。 */
export const MAX_GPU_WALL_RATIO = 1.05;

/**
 * GPU 実時間 / 壁時計 の下限（壁時計が目標長以上の run だけに掛ける）。
 *
 * 反復数は GPU 実時間で目標長（{@link TARGET_PASS_MS}）に合わせるので、ns が正しければ
 * 壁時計との差は submit とフェンスの床（≈11ms — research 2026-08-30 §7）と encode の費用だけで、
 * 比は 1 に近い。Deno × `timestampPeriod` ≠ 1 の GPU（B570 は 52.08 ns/tick）では ns が raw tick の
 * ままなので、比は 1/period 前後まで落ち、反復数は period 倍に膨らむ（known-issues「Intel Arc
 * B570」節）。1/8 はその 2 つの間に置いた線で、壁時計が目標長に届かない run（床が支配する短い
 * run）では比が正しくても小さくなるので判定しない。
 */
export const MIN_GPU_WALL_RATIO = 1 / 8;

/**
 * timing の ns が壁時計と矛盾するときの警告文（矛盾しなければ `undefined`）。
 *
 * timestamp の周期は WebGPU の API に出ないので、ns を自動で補正することはできない。ここで
 * できるのは、同じ run の壁時計と突き合わせて「この ns は信用できない」を記録に残すことだけで、
 * 黙って誤った ms と反復数を出さないための欄である。
 */
export const timingWallWarning = (totalNs: number, wallNs: number): string | undefined => {
  const ratio = totalNs / wallNs;
  if (ratio > MAX_GPU_WALL_RATIO) {
    return `GPU 実時間が壁時計を超えた（比 ${ratio.toFixed(3)}）— timestamp の ns 換算が過大`;
  }
  if (wallNs >= TARGET_PASS_MS * 1e6 && ratio < MIN_GPU_WALL_RATIO) {
    return `GPU 実時間が壁時計の ${ratio.toFixed(4)} 倍 — timestamp が ns に換算されていない疑い` +
      "（Deno × timestampPeriod ≠ 1）。ns / weighted_ms / 反復数を信用しない";
  }
  return undefined;
};

/** timing モードの代表値（rounds 回の min）。 */
export type TimingResult = {
  readonly mode: "timing";
  readonly reps: number;
  readonly rounds: number;
  /**
   * min(run 全体の ns / 反復数) = **ノード 1 本ぶん**の GPU 時間。1 ノードが複数 dispatch を出す op
   * （a8 linear = quantize_rows + linear）はその合計 — census 加重（ノード本数）に掛ける量はこれ。
   */
  readonly nsPerNodeMin: number;
  /** 1 ノードが出した dispatch 数（run の dispatch 数 / 反復数）。 */
  readonly dispatchesPerNode: number;
  readonly keys: readonly string[];
  readonly clampedNegativeSamples: number;
  /** 代表値（min）を採った round の GPU 実時間 / 壁時計。 */
  readonly gpuWallRatio: number;
  /** その round の {@link timingWallWarning}（矛盾が無ければ `undefined`）。 */
  readonly wallWarning: string | undefined;
};

export const measureTiming = async (
  session: Session,
  inputs: RunInputs,
  reps: number,
  rounds: number = ROUNDS,
  heater?: Heater,
): Promise<TimingResult> => {
  // パイプライン生成・初回 bind group を計測外へ（1 回）。クロックは filler が張り付かせる。
  await sampleTiming(session, inputs);
  if (heater !== undefined) await pinClocks(heater);
  let min = Number.POSITIVE_INFINITY;
  let dispatchesPerNode = 0;
  let keys: readonly string[] = [];
  let clamped = 0;
  let representative: TimingSample | undefined;
  for (let round = 0; round < rounds; round += 1) {
    // MUST: round の間に filler を挟む — 計測パス（≈80ms）だけでは P8 へ落ちる（モジュール doc）。
    if (heater !== undefined) await heater.run();
    const sample = await sampleTiming(session, inputs);
    if (sample.dispatchCount === 0) throw new Error("dispatch が 0 本の run を測った");
    const perNode = sample.totalNs / reps;
    if (perNode < min) {
      min = perNode;
      representative = sample;
    }
    dispatchesPerNode = sample.dispatchCount / reps;
    keys = sample.keys;
    clamped += sample.clampedNegativeSamples;
  }
  if (representative === undefined) {
    throw new Error(`timing の round が 1 本も無い（rounds=${rounds}）`);
  }
  return {
    mode: "timing",
    reps,
    rounds,
    nsPerNodeMin: min,
    dispatchesPerNode,
    keys,
    clampedNegativeSamples: clamped,
    gpuWallRatio: representative.totalNs / representative.wallNs,
    wallWarning: timingWallWarning(representative.totalNs, representative.wallNs),
  };
};

/** wall モードの代表値（rounds 回の min）。 */
export type WallResult = {
  readonly mode: "wall";
  readonly reps: number;
  readonly rounds: number;
  /** min(区間の壁時計 / 反復数) — フェンス 1 本ぶんの床を反復で償却した値。 */
  readonly msPerRepMin: number;
};

/**
 * gpuTiming 無効 device で、反復ぶんの enqueue を **フェンス 1 本の区間**（`beginBatch`）に束ねて
 * 壁時計を測る。1 submit ≈11ms のフェンス床（research 2026-08-30 §7）を反復で償却するため、
 * `run` を反復回数ぶん呼ぶ形は採らない（床が反復回数ぶん乗る）。
 * `gpu` / `session` は使う面だけを受ける（偽の device で区間ロックの返却を検査するため）。
 */
export const measureWall = async <Batch extends { finish(): Promise<unknown> }>(
  gpu: { beginBatch(): Promise<Batch> },
  session: { enqueue(inputs: RunInputs, options: { readonly batch: Batch }): Promise<unknown> },
  inputs: RunInputs,
  reps: number,
  rounds: number = ROUNDS,
  heater?: Heater,
): Promise<WallResult> => {
  const once = async (): Promise<number> => {
    const batch = await gpu.beginBatch();
    // MUST: batch は device 単位で排他の区間ロックを持ち、finish でしか返らない。enqueue の
    // reject で finish を飛ばすと次の beginBatch / Session の dispose が永久に待ち、single の
    // 「1 行の失敗で掃引を止めない」が掃引全体のハングに化ける（finish は何度呼んでも同じ決着）。
    try {
      const started = performance.now();
      const pending: Promise<unknown>[] = [];
      for (let rep = 0; rep < reps; rep += 1) pending.push(session.enqueue(inputs, { batch }));
      await Promise.all(pending);
      await batch.finish();
      return performance.now() - started;
    } finally {
      await batch.finish();
    }
  };
  await once();
  if (heater !== undefined) await pinClocks(heater);
  let min = Number.POSITIVE_INFINITY;
  for (let round = 0; round < rounds; round += 1) {
    if (heater !== undefined) await heater.run();
    min = Math.min(min, (await once()) / reps);
  }
  return { mode: "wall", reps, rounds, msPerRepMin: min };
};
