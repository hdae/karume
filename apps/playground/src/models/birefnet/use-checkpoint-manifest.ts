import { useState } from "react";

import type { DistributionStatus } from "@/lib/distributions";
import { type ManifestState, useManifest } from "@/hooks/use-manifest";

import type { CheckpointId } from "./settings";

const LOADING: ManifestState = { kind: "loading" };

/**
 * `useManifest` for a checkpoint the user switches. `useManifest` resets to `loading` in its effect,
 * so for the one commit after a switch it still returns the previous checkpoint's state: a Run in
 * that window would pair the new checkpoint with the old manifest's selection and sha256. This hook
 * remembers the state that was current when the checkpoint changed (adjusting state while rendering,
 * which React re-renders before committing) and reports `loading` until `useManifest` returns a new
 * state.
 */
export const useCheckpointManifest = (
  checkpoint: CheckpointId,
  status: DistributionStatus | undefined,
): ManifestState => {
  const manifest = useManifest(checkpoint, status);
  const [switched, setSwitched] = useState<{
    readonly checkpoint: CheckpointId;
    readonly stale: ManifestState | undefined;
  }>({ checkpoint, stale: undefined });
  let stale = switched.stale;
  if (switched.checkpoint !== checkpoint) {
    stale = manifest;
    setSwitched({ checkpoint, stale: manifest });
  }
  return manifest === stale ? LOADING : manifest;
};
