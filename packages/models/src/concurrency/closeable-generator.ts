/**
 * Notifies the owner of a return / throw before start as well. A native generator closed before its
 * first next does not run the body's finally, so state acquired when the generator was issued has to
 * be released separately. Cleanup after start, and the ordering of next / return, are left to the
 * native generator.
 */
export const closeableGenerator = <T>(
  generator: AsyncGenerator<T, void, undefined>,
  onUnstartedClose: (failure?: { readonly error: unknown }) => void | Promise<void>,
): AsyncGenerator<T, void, undefined> => {
  let started = false;
  let closing: Promise<void> | undefined;
  const close = (result: Promise<IteratorResult<T, void>>): Promise<IteratorResult<T, void>> => {
    // A rejection of the Promise passed to return is also reported to the owner. Resolving the
    // completion value and ordering the requests are the native generator's job; only the owner's
    // cleanup is layered once onto the first close.
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
    return: (value) => (started ? generator.return(value) : close(generator.return(value))),
    throw: (error) => (started ? generator.throw(error) : close(generator.throw(error))),
    [Symbol.asyncIterator]: () => iterator,
    [Symbol.asyncDispose]: async () => {
      await iterator.return(undefined);
    },
  };
  return iterator;
};
