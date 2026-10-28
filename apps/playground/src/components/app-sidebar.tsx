import { Link } from "@tanstack/react-router";
import { FlaskConicalIcon } from "lucide-react";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  useSidebar,
} from "@/components/ui/sidebar";
import { useCurrentRouteId } from "@/hooks/use-current-route-id";
import { pagesByGroup } from "@/models/registry";

/** The left menu: the pages of the registry by group, the current route highlighted. */
export const AppSidebar = () => {
  const routeId = useCurrentRouteId();
  // On a phone the sidebar is a sheet: close it once a page is chosen.
  const { setOpenMobile } = useSidebar();
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              isActive={routeId === "/"}
              tooltip="Karume playground"
              render={<Link to="/" />}
              onClick={() => setOpenMobile(false)}
            >
              <FlaskConicalIcon />
              <span className="font-semibold">Karume playground</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        {pagesByGroup().map(({ group, pages }) => (
          <SidebarGroup key={group}>
            <SidebarGroupLabel>{group}</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {pages.map((page) => (
                  <SidebarMenuItem key={page.path}>
                    <SidebarMenuButton
                      isActive={routeId === page.path}
                      tooltip={page.title}
                      render={<Link to={page.path} />}
                      onClick={() => setOpenMobile(false)}
                    >
                      <page.icon />
                      <span>{page.title}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>
      <SidebarRail />
    </Sidebar>
  );
};
