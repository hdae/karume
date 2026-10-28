/**
 * The reading dictionary of the text analyser, loaded once when the page opens (from Cache Storage
 * after the first visit) and kept while the page is open. `retry` loads it again after a failure.
 */

import { useCallback, useEffect, useState } from "react";

import { loadReadingDictionary, type ReadingDictionary } from "./analyzer";

export type ReadingDictionaryState =
  | { readonly kind: "loading"; readonly loaded: number; readonly total: number | undefined }
  | { readonly kind: "ready"; readonly dictionary: ReadingDictionary; readonly seconds: number }
  | { readonly kind: "error"; readonly error: unknown };

export const useReadingDictionary = (): {
  readonly state: ReadingDictionaryState;
  readonly retry: () => void;
} => {
  const [state, setState] = useState<ReadingDictionaryState>({
    kind: "loading",
    loaded: 0,
    total: undefined,
  });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    // `getDictionary` takes no AbortSignal: a load the page no longer wants runs to its end and is
    // dropped here.
    let disposed = false;
    const started = performance.now();
    setState({ kind: "loading", loaded: 0, total: undefined });
    loadReadingDictionary((loaded, total) => {
      if (!disposed) setState({ kind: "loading", loaded, total });
    }).then(
      (dictionary) => {
        if (!disposed) {
          setState({ kind: "ready", dictionary, seconds: (performance.now() - started) / 1000 });
        }
      },
      (error: unknown) => {
        if (!disposed) setState({ kind: "error", error });
      },
    );
    return () => {
      disposed = true;
    };
  }, [attempt]);

  const retry = useCallback(() => setAttempt((count) => count + 1), []);
  return { state, retry };
};
