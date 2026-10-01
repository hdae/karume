/**
 * マイクロベンチの計測規約の定数と反復数の見積り（掃引の計測核 `harness.ts` と、リポ内の道具
 * `tools/opbench` が共有する — ADR 0117 決定 1 で道具側からここへ移した。定義はここ 1 本）。
 *
 * 規約の出典 = docs/research/2026-08-10-kernel-variant-sweep.md §5:
 * 1. 1 タイムドパスが ≈80ms になるまで同一 dispatch を積む（duty cycle を保ってクロックを
 *    張り付かせる — アイドルから測ると同じ変種が 2 倍揺れた）
 * 2. 代表値は **min**（熱ドリフトは min で吸う）
 * 3. 絶対 ms は別日・別機と比較しない — 同一リグ内の比だけが有効
 */

/** 1 タイムドパスの目標長（ms）。研究 §5-1 の実測値をそのまま既定にする。 */
export const TARGET_PASS_MS = 80;

/**
 * 計測前にクロックを張り付かせる空回しの下限（GPU 実時間の累計 ns）。研究 §5-2 の
 * 「対の前にメモリクロックが張り付くまで空回し」を、nvidia-smi に依らず時間で置き換えたもの。
 * アイドルから 1 パス ≈80ms では張り付かない（同一変種が 2 倍揺れた実測）。
 */
export const WARMUP_NS = 500e6;

/** 空回しの回数下限（累計時間が先に満ちても、パイプライン生成直後の 1 回だけでは終えない）。 */
export const WARMUP_MIN_RUNS = 3;

/** 代表値を取る回数（min を採るので偶奇や対は要らない — 同一ケースの反復）。 */
export const ROUNDS = 5;

/**
 * 1 dispatch の推定 ns から、目標長に足りる反復数を決める（1 以上・上限で打ち切り）。
 * 推定が 0 / 非有限のときは上限（測れないほど速い = 積めるだけ積む）。
 *
 * 上限は呼び手が渡す — 掃引は同じバッファへ重ね打ちするので大きく取り（harness.ts の
 * `SWEEP_MAX_REPS`）、opbench は出力の readback が反復に比例するので小さく取る（`MAX_REPS`）。
 */
export const calibrateReps = (
  nsPerDispatch: number,
  targetMs: number,
  maxReps: number,
): number => {
  if (!Number.isFinite(nsPerDispatch) || nsPerDispatch <= 0) return maxReps;
  const reps = Math.ceil((targetMs * 1e6) / nsPerDispatch);
  return Math.max(1, Math.min(maxReps, reps));
};
