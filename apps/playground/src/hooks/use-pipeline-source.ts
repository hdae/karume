import type { HubRepoRef } from "@karume/hub";
import { useCallback, useMemo, useState } from "react";

import {
  type DistributionStatus,
  hfRepoRef,
  type PipelineSource,
  type SourceKind,
} from "@/lib/distributions";

export type PipelineSourceChoice = {
  /** The source to build from (stable until the choice, the mount or the cache generation changes). */
  readonly source: PipelineSource;
  /** The cached route of the mount; `undefined` when the dev server serves none (direct only). */
  readonly hfRef: HubRepoRef | undefined;
  /** Choose a kind; `cached` falls back to `direct` while the mount has no HF-compatible route. */
  readonly choose: (kind: SourceKind) => void;
  /** Call after clearing the hub's cache: the next build gets fresh readers. */
  readonly cacheCleared: () => void;
};

/**
 * The cached / direct choice for one mount (`status` from `/models/index.json`). Cached is the
 * default. A page that switches mounts (a model of another distribution) passes the new status and
 * keeps the chosen kind.
 */
export const usePipelineSource = (status: DistributionStatus | undefined): PipelineSourceChoice => {
  const [chosen, setChosen] = useState<SourceKind>("cached");
  const hf = status?.hf;
  const hfRef = useMemo(
    () => (hf === undefined ? undefined : hfRepoRef(hf, document.baseURI)),
    [hf],
  );
  const [cacheGeneration, setCacheGeneration] = useState(0);
  const source = useMemo<PipelineSource>(
    () =>
      chosen === "cached" && hfRef !== undefined
        ? { kind: "cached", ref: hfRef, cacheGeneration }
        : { kind: "direct" },
    [chosen, hfRef, cacheGeneration],
  );
  const cacheCleared = useCallback(() => setCacheGeneration((count) => count + 1), []);
  return { source, hfRef, choose: setChosen, cacheCleared };
};
