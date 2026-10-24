import { acquireGpu, type AcquireGpuOptions, type GpuContext } from "@karume/runtime";
import { useCallback, useEffect, useState } from "react";

export type GpuState =
  | { readonly kind: "acquiring" }
  | { readonly kind: "ready"; readonly gpu: GpuContext }
  | { readonly kind: "lost"; readonly reason: string; readonly message: string }
  | { readonly kind: "error"; readonly error: unknown };

/**
 * One GPU per page: acquired on mount, destroyed on unmount. `reacquire` destroys the current one
 * and acquires again (after a device loss). `options` MUST be stable (a module constant) — a new
 * object would re-acquire on every render.
 */
export const useGpu = (
  options: Omit<AcquireGpuOptions, "onDeviceLost">,
): { readonly state: GpuState; readonly reacquire: () => void } => {
  const [state, setState] = useState<GpuState>({ kind: "acquiring" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let disposed = false;
    let acquired: GpuContext | undefined;
    setState({ kind: "acquiring" });
    acquireGpu({
      ...options,
      onDeviceLost: (info) => {
        if (!disposed) setState({ kind: "lost", reason: info.reason, message: info.message });
      },
    }).then(
      (gpu) => {
        // The page may have unmounted (or React re-run the effect) while acquiring.
        if (disposed) gpu.destroy();
        else {
          acquired = gpu;
          setState({ kind: "ready", gpu });
        }
      },
      (error: unknown) => {
        if (!disposed) setState({ kind: "error", error });
      },
    );
    return () => {
      disposed = true;
      acquired?.destroy();
    };
  }, [options, attempt]);

  const reacquire = useCallback(() => setAttempt((count) => count + 1), []);
  return { state, reacquire };
};
