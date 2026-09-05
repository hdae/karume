/**
 * 構築の中断を段の境目で観測する道具（**パイプライン非依存の共通処理**）。
 *
 * MUST: barrel には出さない。中断のノブは各パイプラインの `options.signal` で、これはその
 * 検査の置き方を 1 本に揃えるための内部機構。
 */

/**
 * 段の境目で中断を**観測できる形**に検査する（イベントループへ 1 度譲ってから `throwIfAborted`）。
 *
 * MUST: 譲り先は**マクロタスク**でなければならない。`abort()` の届き方はクリック・timer・
 * worker メッセージなどの**タスク**配送なので、`await Promise.resolve()`（マイクロタスク）
 * では現在のタスクの中に留まったままで、まだ実行されていない中断タスクを観測できない。
 * 譲らない検査は「呼ばれた時点で既に中断済み」しか拾えない死文になる。
 *
 * NOTE: 譲り先は **`setTimeout(0)`** に固定する（`MessageChannel` ではない）。ブラウザの非表示タブでは
 * timer throttling がそのまま生成速度になる（1 token ごとに 1 timer）が、`MessageChannel` に
 * 替えると **timer で届く中断（`setTimeout(() => abort())` / `AbortSignal.timeout`）との順序が
 * 失われる** — 先に仕掛けた timer より message タスクが先に配送されうるので、「実行開始後に
 * 届いた中断は最初の段境界で効く」（`anima_pipeline_test` / `irodori_pipeline_test` の門）が
 * 負荷次第で破れる（2026-09-05 フル verify で実測）。中断の確実な観測を速度より優先する。
 * 非表示タブの throttling は limitations に記録する（W-M6-3）。
 *
 * `signal` 未指定なら譲らない（購読していない呼び出しにタスク 1 往復のコストを乗せない）。
 */
export const settleAbort = async (signal: AbortSignal | undefined): Promise<void> => {
  if (signal === undefined) return;
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  signal.throwIfAborted();
};
