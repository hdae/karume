/**
 * 非同期操作を 1 本の promise 鎖に載せる直列化席（**パイプライン非依存の共通処理** — 3 家族の
 * `generate` / `dispose` が同じ形で使う）。
 *
 * MUST: barrel には出さない。これはパイプラインが「グラフは 1 本ずつ」を公開 API 側でも守る
 * ための内部機構で、利用者が触る面ではない。
 *
 * MUST: 鎖そのものは決して reject しない — 前段の失敗（成功も）**決着だけ**を次へ渡す。
 * ここが reject を伝播すると 1 回の生成失敗で以後の生成が全て道連れになる。呼び出し側には
 * 操作の promise をそのまま返すので、失敗は fail loudly のまま届く。
 *
 * NOTE: runtime の Session も同じ形で run / dispose を直列化している（`executor.ts` の
 * `#enqueue`）が、あちらは Session 1 本の内側だけを守る。段ごとに Session を張り替える
 * パイプラインは、その外側（generate 1 回ぶん）をここで守る。
 */

/**
 * 直前までに積まれた操作の決着後に `operation` を走らせ、その結果 / 例外をそのまま返す。
 * 同期の操作（破棄など）も積める — 順番待ちに載ることだけが要件で、非同期である必要は無い。
 */
type OperationChain = <T>(operation: () => T | Promise<T>) => Promise<T>;

/** 空の（何も積まれていない）直列化鎖を作る。 */
export const createOperationChain = (): OperationChain => {
  let chain: Promise<void> = Promise.resolve();
  return <T>(operation: () => T | Promise<T>): Promise<T> => {
    const result = chain.then(operation);
    chain = result.then(() => undefined, () => undefined);
    return result;
  };
};
