/**
 * A serialisation hook that puts asynchronous operations on one promise chain
 * (**pipeline-independent shared processing** — used the same way by the families' `generate` /
 * `dispose`).
 *
 * MUST: not exported from the barrel. This is an internal mechanism with which a pipeline keeps "one
 * graph at a time" on the public API side too, not a surface the user touches.
 *
 * MUST: the chain itself never rejects — it passes on **only the settlement** of the previous
 * operation (failure or success). If it propagated a rejection, one failed generation would drag
 * down every later generation. The caller gets the operation's promise as is, so failures still
 * arrive loudly.
 *
 * NOTE: the runtime's Session serialises run / dispose the same way (`#serialize` in `executor.ts`),
 * but that guards only the inside of one Session. A pipeline that swaps Sessions per stage guards
 * the outside of it (one generate) here.
 */

/**
 * Runs `operation` after everything queued before it has settled, and returns its result /
 * exception as is. Synchronous operations (disposal etc.) can be queued too — the only requirement is
 * to wait in line; being asynchronous is not needed.
 */
type OperationChain = <T>(operation: () => T | Promise<T>) => Promise<T>;

/** Creates an empty serialisation chain (nothing queued yet). */
export const createOperationChain = (): OperationChain => {
  let chain: Promise<void> = Promise.resolve();
  return <T>(operation: () => T | Promise<T>): Promise<T> => {
    const result = chain.then(operation);
    chain = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  };
};
