import { MANIFEST_FILENAME, type Manifest, parseManifest } from "@karume/hub";
import { useEffect, useState } from "react";

import type { DistributionStatus } from "@/lib/distributions";
import { sha256Hex } from "@/lib/record";
import type { MountName } from "@/mounts";

export type ManifestState =
  | { readonly kind: "loading" }
  | { readonly kind: "missing"; readonly directory: string }
  | {
      readonly kind: "ready";
      readonly manifest: Manifest;
      /** sha256 of the mount's `karume.json` bytes (what a record names the distribution by). */
      readonly sha256: string;
    }
  | { readonly kind: "error"; readonly error: unknown };

/**
 * The manifest of mount `name`, read over `/models/` once `/models/index.json` says the directory is
 * there (`status`; `loading` until then). The manifest describes every preset, so a page reads it once
 * and builds its selects from it whatever source the pipeline later reads. The bytes are fetched once:
 * the manifest is parsed from the same bytes the sha256 is taken of, so the two always agree.
 */
export const useManifest = (
  name: MountName,
  status: DistributionStatus | undefined,
): ManifestState => {
  const [state, setState] = useState<ManifestState>({ kind: "loading" });
  useEffect(() => {
    if (status === undefined) {
      setState({ kind: "loading" });
      return undefined;
    }
    if (!status.available) {
      setState({ kind: "missing", directory: status.directory });
      return undefined;
    }
    const controller = new AbortController();
    setState({ kind: "loading" });
    const load = async (): Promise<ManifestState> => {
      const url = `models/${name}/${MANIFEST_FILENAME}`;
      const response = await fetch(url, { signal: controller.signal });
      if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      const manifest = parseManifest(new TextDecoder().decode(bytes));
      return { kind: "ready", manifest, sha256: await sha256Hex(bytes) };
    };
    load().then(
      (next) => {
        if (!controller.signal.aborted) setState(next);
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setState({ kind: "error", error });
      },
    );
    return () => controller.abort();
  }, [name, status]);
  return state;
};
