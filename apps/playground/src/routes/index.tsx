import { createFileRoute, Link } from "@tanstack/react-router";

import { Badge } from "@/components/ui/badge";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useDistributionIndex } from "@/hooks/use-distribution-index";
import { MODEL_PAGES } from "@/models/registry";

const IndexPage = () => {
  const index = useDistributionIndex();
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold">Model pages</h1>
        <p className="text-sm text-muted-foreground">
          Each page runs a Karume pipeline on this browser's GPU, reading the distributions the dev
          server mounts under <code>/models/</code>.
        </p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {MODEL_PAGES.map((page) => (
          <Link key={page.path} to={page.path} className="block">
            <Card className="h-full transition-colors hover:bg-muted/50">
              <CardHeader>
                <CardTitle>{page.title}</CardTitle>
                <CardDescription>{page.summary}</CardDescription>
                <div className="flex flex-wrap gap-2 pt-2">
                  {page.distributions.map((name) => {
                    const status = index.kind === "ready" ? index.index[name] : undefined;
                    return (
                      <Badge
                        key={name}
                        variant={status?.available === true ? "secondary" : "outline"}
                      >
                        {status === undefined
                          ? name
                          : `${status.directory}${status.available ? "" : " (missing)"}`}
                      </Badge>
                    );
                  })}
                </div>
              </CardHeader>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
};

export const Route = createFileRoute("/")({ component: IndexPage });
