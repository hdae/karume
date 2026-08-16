/**
 * 取得層へ差し込むバイト数の門（ADR 0038 §2「取得時は AbortController を渡し、受信バイトが
 * size を超えた時点・content-length が size と食い違った時点で abort する」）。
 *
 * MUST: 全量読了後の判定に頼らない — 数 GB を撃ち終わってから「多すぎた」と分かる設計は
 * 防波堤として機能しない。`fetch` を包む形にしてあるのは、取得層（`@hdae/fetch-cache`）が
 * network に出た時だけこの門を通り、キャッシュヒットは素通りするため。
 */

/** URL 1 本に課すバイト数の門。 */
export type ByteBudget = {
  /** 受信を許す上限。`exact` が真なら content-length もこの値と一致しなければならない。 */
  readonly maxBytes: number;
  readonly exact: boolean;
  /** 違反時に投げるエラー（エラー型の選択は呼び出し側の責務）。 */
  readonly violation: (actual: number, where: "content-length" | "body") => Error;
  /** network 取得が実際に発火したことの通知（キャッシュヒットとの区別に使う）。 */
  readonly onRequest?: () => void;
};

const urlOf = (input: string | URL | Request): string =>
  typeof input === "string" ? input : input instanceof URL ? input.href : input.url;

/**
 * 予算付きの `fetch` を作る。予算の無い URL（revision 解決 API 等）は素通しする。
 */
export const createGuardedFetch = (
  base: typeof globalThis.fetch,
  budgets: ReadonlyMap<string, ByteBudget>,
): typeof globalThis.fetch =>
async (input, init) => {
  const url = urlOf(input);
  const budget = budgets.get(url);
  if (budget === undefined) return await base(input, init);
  budget.onRequest?.();

  // 上流の signal と合成する。違反時にこの controller を落とせば、上流を巻き込まずに
  // この 1 本だけを止められる。
  const controller = new AbortController();
  // `??` が null を落とすので outer は AbortSignal | undefined（明示 `signal: null` は
  // Request 側の signal へフォールバックする — 従来挙動のまま）。
  const outer = init?.signal ?? (input instanceof Request ? input.signal : undefined);
  const signal = outer === undefined
    ? controller.signal
    : AbortSignal.any([controller.signal, outer]);
  const response = await base(input, { ...init, signal });
  if (!response.ok || response.body === null) return response;

  const declared = response.headers.get("content-length");
  if (declared !== null) {
    const length = Number(declared);
    const mismatch = budget.exact ? length !== budget.maxBytes : length > budget.maxBytes;
    if (!Number.isSafeInteger(length) || mismatch) {
      const error = budget.violation(length, "content-length");
      controller.abort(error);
      throw error;
    }
  }

  let received = 0;
  const guarded = response.body.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform: (chunk, stream) => {
        received += chunk.byteLength;
        if (received > budget.maxBytes) {
          const error = budget.violation(received, "body");
          controller.abort(error);
          stream.error(error);
          return;
        }
        stream.enqueue(chunk);
      },
    }),
  );
  return new Response(guarded, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
};
