/**
 * 全量面 {@link ../src/fetch.ts fetchAssets} の送出を律速する小さな並行プリミティブ。
 *
 * 呼び手は 1 箇所しかないが、破れても「たまに詰まる」としか見えない不変条件（lost wakeup）を
 * 持つ。単体で観測できる形に切り出して、テストで挙動そのものを凍結するために独立させてある。
 *
 * MUST: モジュール副作用ゼロ — 状態は必ず create* の呼び出しごとに閉じる（top-level の可変
 * 状態を置くと、無関係な 2 つの取得が同じ予算を共有してしまう）。
 */

/** {@link createByteAdmission} が返す席の貸し借り。 */
export type ByteAdmission = {
  /** `bytes` ぶんの席が空くまで待ち、確保できたら解決する。 */
  readonly admit: (bytes: number) => Promise<void>;
  /** 取得の決着（成否は問わない）で席を返す。 */
  readonly release: (bytes: number) => void;
};

/**
 * in-flight の合計バイト数に上限を課すアドミッション。要求は**先着順**に通し、追い越させない
 * （後続の小さいファイルが巨大ファイルを飛び越すと、送出順が実行時のサイズ分布で変わって
 * 再現性が失われる）。
 *
 * MUST: `admit` を呼ぶのは**同時に 1 本の送出ループだけ**。待機席は 1 つしか無いので、複数から
 * 待つと `release` の起こしを取りこぼす（この制約があるから head-of-line blocking も自明に
 * 成立する）。
 *
 * in-flight が 0 のときは予算を超える単独要求も必ず通す — 誰も走っていないのに待つと、席を
 * 返す相手が居ないまま永久に詰まる。
 *
 * lost wakeup が無いことの理由: ①判定から待機登録までの間に await が無いので、`release` が
 * その隙間へ割り込めない。②起こされた待機者が再判定するまでに追加の `release` が来ても、
 * その時点で in-flight が 0 なら条件が偽になって進み、0 でなければ必ず未決着の取得が残って
 * いる＝将来の `release` が来る。
 */
export const createByteAdmission = (maxBytes: number): ByteAdmission => {
  let inFlight = 0;
  let waiting: (() => void) | undefined;
  return {
    admit: async (bytes: number): Promise<void> => {
      while (inFlight > 0 && inFlight + bytes > maxBytes) {
        await new Promise<void>((resume) => {
          waiting = resume;
        });
      }
      inFlight += bytes;
    },
    release: (bytes: number): void => {
      inFlight -= bytes;
      const resume = waiting;
      waiting = undefined;
      resume?.();
    },
  };
};
