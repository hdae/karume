import { createFileRoute, Link } from "@tanstack/react-router";

import { MountBadges } from "@/components/mount-badges";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useDistributionIndex } from "@/hooks/use-distribution-index";
import { pagesByGroup } from "@/models/registry";

const IndexPage = () => {
  const index = useDistributionIndex();
  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-2xl font-semibold">Model pages</h1>
        <p className="text-sm text-muted-foreground">
          Each page runs a Karume pipeline on this browser's GPU, reading the distributions the dev
          server mounts under <code>/models/</code>.
        </p>
      </div>
      {pagesByGroup().map(({ group, pages }) => (
        <section key={group} className="flex flex-col gap-3">
          <h2 className="text-sm font-medium text-muted-foreground">{group}</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {pages.map((page) => (
              <Link key={page.path} to={page.path} className="block">
                <Card className="h-full transition-colors hover:bg-muted/50">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <page.icon className="size-4 shrink-0" />
                      {page.title}
                    </CardTitle>
                    <CardDescription>{page.summary}</CardDescription>
                    {page.mounts.length === 0 ? null : (
                      <div className="pt-2">
                        <MountBadges index={index} mounts={page.mounts} />
                      </div>
                    )}
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
};

export const Route = createFileRoute("/")({ component: IndexPage });
