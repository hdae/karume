import { MountBadges } from "@/components/mount-badges";
import { ErrorAlert } from "@/components/error-alert";
import { useDistributionIndex } from "@/hooks/use-distribution-index";

/** Placeholder until the page is implemented: the title and whether its mounts are there. */
export const Sbv2Page = () => {
  const index = useDistributionIndex();
  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-2xl font-semibold">Style-Bert-VITS2 — text → speech</h1>
      {index.kind === "error" ? (
        <ErrorAlert title="Distribution index" error={index.error} />
      ) : null}
      <MountBadges index={index} mounts={["sbv2"]} />
      <p className="text-sm text-muted-foreground">This page is not implemented yet.</p>
    </div>
  );
};
