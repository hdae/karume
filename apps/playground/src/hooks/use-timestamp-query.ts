import { useEffect, useState } from "react";

export type TimestampQueryState =
  | { readonly kind: "probing" }
  | { readonly kind: "ready"; readonly available: boolean }
  | { readonly kind: "error"; readonly error: unknown };

/**
 * Whether this browser's adapter lists `timestamp-query` (probed once per mount). A device acquired
 * without `gpuTiming` does not enable the feature, so its `GpuContext` cannot answer this.
 */
export const useTimestampQuery = (): TimestampQueryState => {
  const [state, setState] = useState<TimestampQueryState>({ kind: "probing" });
  useEffect(() => {
    let disposed = false;
    const gpu: GPU | undefined = navigator.gpu;
    if (gpu === undefined) {
      setState({ kind: "error", error: new Error("This browser has no navigator.gpu") });
      return undefined;
    }
    gpu.requestAdapter().then(
      (adapter) => {
        if (!disposed) {
          setState({ kind: "ready", available: adapter?.features.has("timestamp-query") === true });
        }
      },
      (error: unknown) => {
        if (!disposed) setState({ kind: "error", error });
      },
    );
    return () => {
      disposed = true;
    };
  }, []);
  return state;
};
