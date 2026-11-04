// Snapshot check of the public surface of `@karume/hub` (ADR
// [0008](../../../docs/decisions/0008-public-api.md) addendum 2026-09-20). Needs neither the
// network nor real assets.
//
// The hub exposes two surfaces: `.` (Web standard APIs only) and `./deno` (the Deno-only adapter)
// (ADR 0080 / 0086). The ways the **split** between the two can break — a Deno-dependent name
// leaking into `.`, or `./deno` thinning out so that consumers reach for internal paths directly —
// do not turn the type check red. Watching additions and removals per surface shows such a move as
// a diff.
//
// The check itself lives once, on the runtime side (a cross-package relative import, like
// `container-files.ts` and others).

import { test } from "vite-plus/test";
import { assertPublicSurface } from "../../runtime/tests/helpers/public-surface.ts";

test("public surface: the hub's exports (. and ./deno) match the snapshot", async () => {
  await assertPublicSurface(new URL("../", import.meta.url));
});
