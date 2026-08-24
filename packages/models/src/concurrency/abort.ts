/**
 * 構築の中断を段の境目で観測する道具（**パイプライン非依存の共通処理**）。
 *
 * MUST: barrel には出さない。中断のノブは各パイプラインの `options.signal` で、これはその
 * 検査の置き方を 1 本に揃えるための内部機構。
 */

/**
 * 段の境目で中断を**観測できる形**に検査する（イベントループへ 1 度譲ってから `throwIfAborted`）。
 *
 * MUST: 譲り先は**マクロタスク**（`setTimeout`）でなければならない。`abort()` の届き方は
 * クリック・timer・worker メッセージなどの**タスク**配送なので、`await Promise.resolve()`
 * （マイクロタスク）では現在のタスクの中に留まったままで、まだ実行されていない中断タスクを
 * 観測できない。譲らない検査は「呼ばれた時点で既に中断済み」しか拾えない死文になる。
 *
 * `signal` 未指定なら譲らない（購読していない呼び出しにタスク 1 往復のコストを乗せない）。
 */
export const settleAbort = async (signal: AbortSignal | undefined): Promise<void> => {
  if (signal === undefined) return;
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  signal.throwIfAborted();
};
