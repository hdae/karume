/**
 * 未開始の return / throw も所有者へ通知する。native generator は最初の next より前に
 * 閉じると本体の finally を実行しないため、発行時に確保した状態を別に解放する必要がある。
 * 開始済みの終了処理と、next / return の順序は native generator に任せる。
 */
export const closeableGenerator = <T>(
  generator: AsyncGenerator<T, void, undefined>,
  onUnstartedClose: (failure?: { readonly error: unknown }) => void | Promise<void>,
): AsyncGenerator<T, void, undefined> => {
  let started = false;
  let closing: Promise<void> | undefined;
  const close = (
    result: Promise<IteratorResult<T, void>>,
  ): Promise<IteratorResult<T, void>> => {
    // return に渡された Promise の拒否も所有者へ通知する。終了値の解決と要求の順序は
    // native generator が担い、所有者の後始末だけを最初の終了に一度重ねる。
    closing ??= result.then(
      () => onUnstartedClose(),
      (error: unknown) => onUnstartedClose({ error }),
    );
    return result.then(
      async (value) => {
        await closing;
        return value;
      },
      async (error: unknown) => {
        await closing;
        throw error;
      },
    );
  };
  const iterator: AsyncGenerator<T, void, undefined> = {
    next: (...args) => {
      if (closing !== undefined) return closing.then(() => generator.next(...args));
      started = true;
      return generator.next(...args);
    },
    return: (value) => started ? generator.return(value) : close(generator.return(value)),
    throw: (error) => started ? generator.throw(error) : close(generator.throw(error)),
    [Symbol.asyncIterator]: () => iterator,
    [Symbol.asyncDispose]: async () => {
      await iterator.return(undefined);
    },
  };
  return iterator;
};
