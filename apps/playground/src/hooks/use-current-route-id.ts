import { useRouterState } from "@tanstack/react-router";

/**
 * The id of the route the router matched for the current location (`"/wan22"`, `"/"` for home;
 * `undefined` before the first match). The sidebar highlight and the header title both key on it, so
 * they agree with the router's own matching (`/wan22/` matches `/wan22`, as the links' `aria-current`
 * does) instead of comparing raw pathnames.
 */
export const useCurrentRouteId = (): string | undefined =>
  useRouterState({ select: (state) => state.matches.at(-1)?.routeId });
