/**
 * 後始末の段を「途中で失敗しても残りを必ず通す」形で並べる道具（**パイプライン非依存の共通
 * 処理** — 常駐 Session を持つ家族の `dispose()` が同じ形で使う）。
 *
 * MUST: barrel には出さない（同居する `options.ts` / `gpu-features.ts` と同じ理由 — 利用者が
 * 触る面ではない内部機構）。
 *
 * ## なぜ直列 `await` ではだめか
 *
 * `dispose()` は `await session.dispose(); if (ownsGpu) gpu.destroy();` の形で書きたくなるが、
 * `Session.dispose()` は reject しうる（`RunArena` が flush の失敗を握り潰さずに伝播させる）。
 * 直列に並べると前段の失敗で後段へ到達せず、各家族の `openXState` が catch で掲げている
 * 「**内部で取った GPU は必ず返す**」という不変条件が dispose 経路でだけ破れる。しかも
 * `#disposal` に rejected promise が居座るので、2 度目の `dispose()` は再試行ではなく同じ
 * 失敗を返すだけになる（`#state` は private なので呼び手に他の把手が無い）。
 *
 * `try { … } finally { gpu.destroy(); }` も劣る — `finally` の中で例外が出ると**元の失敗を
 * 上書き**して、最初に何が壊れたかが消える。
 */

/**
 * 段を宣言順に走らせ、失敗を集めてから投げ直す（破棄の順序は呼び手が決める）。
 *
 * 各段は独立に catch するので、途中の 1 本が投げても残りの段は必ず走る。失敗が 1 件なら
 * **その例外をそのまま**投げ（呼び手の `assertRejects(…, Error, "device lost")` が素直に
 * 書ける）、2 件以上なら `AggregateError` で運ぶ（どの段が落ちたかを消さない）。
 */
export const disposeSteps = async (
  steps: readonly (() => Promise<void> | void)[],
): Promise<void> => {
  const failures: unknown[] = [];
  for (const step of steps) {
    try {
      await step();
    } catch (error) {
      failures.push(error);
    }
  }
  if (failures.length === 1) throw failures[0];
  if (failures.length > 1) {
    throw new AggregateError(failures, `dispose: 後始末が ${failures.length} 件失敗した`);
  }
};
