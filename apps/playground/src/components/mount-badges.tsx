import { Badge } from "@/components/ui/badge";
import type { DistributionIndexState } from "@/hooks/use-distribution-index";
import type { MountName } from "@/mounts";

/**
 * One badge per mount: its directory, marked "(missing)" when the dev server has no `karume.json`
 * there (just the mount name until `/models/index.json` has answered).
 */
export const MountBadges = ({
  index,
  mounts,
}: {
  index: DistributionIndexState;
  mounts: readonly MountName[];
}) => (
  <div className="flex flex-wrap gap-2">
    {mounts.map((name) => {
      const status = index.kind === "ready" ? index.index[name] : undefined;
      return (
        <Badge key={name} variant={status?.available === true ? "secondary" : "outline"}>
          {status === undefined
            ? name
            : `${status.directory}${status.available ? "" : " (missing)"}`}
        </Badge>
      );
    })}
  </div>
);
