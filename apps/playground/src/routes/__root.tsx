import { createRootRoute, Outlet } from "@tanstack/react-router";

import { AppSidebar } from "@/components/app-sidebar";
import { Separator } from "@/components/ui/separator";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useCurrentRouteId } from "@/hooks/use-current-route-id";
import { pageAt } from "@/models/registry";

/** The shell: the sidebar menu, a header with the current page's title, the page in the inset. */
const RootLayout = () => {
  const routeId = useCurrentRouteId();
  const title = (routeId === undefined ? undefined : pageAt(routeId))?.title ?? "Model pages";
  return (
    <TooltipProvider>
      <SidebarProvider>
        <AppSidebar />
        <SidebarInset className="min-w-0">
          <header className="sticky top-0 z-10 flex h-12 shrink-0 items-center gap-2 border-b bg-background px-4">
            <SidebarTrigger className="-ml-1" />
            <Separator orientation="vertical" className="mr-1 data-vertical:h-4" />
            <span className="truncate text-sm font-medium">{title}</span>
          </header>
          <div className="mx-auto w-full max-w-7xl min-w-0 px-4 py-6">
            <Outlet />
          </div>
        </SidebarInset>
      </SidebarProvider>
    </TooltipProvider>
  );
};

export const Route = createRootRoute({ component: RootLayout });
