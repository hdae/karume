import { useEffect, useState } from "react";

import { type DistributionIndex, fetchDistributionIndex } from "@/lib/distributions";

export type DistributionIndexState =
  | { readonly kind: "loading" }
  | { readonly kind: "ready"; readonly index: DistributionIndex }
  | { readonly kind: "error"; readonly error: unknown };

/** Which `/models/<name>/` mounts the dev server has (fetched once per mount of the component). */
export const useDistributionIndex = (): DistributionIndexState => {
  const [state, setState] = useState<DistributionIndexState>({ kind: "loading" });
  useEffect(() => {
    const controller = new AbortController();
    fetchDistributionIndex(controller.signal).then(
      (index) => setState({ kind: "ready", index }),
      (error: unknown) => {
        if (!controller.signal.aborted) setState({ kind: "error", error });
      },
    );
    return () => controller.abort();
  }, []);
  return state;
};
