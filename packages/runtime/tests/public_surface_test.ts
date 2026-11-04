// Snapshot check of the public surface of `@karume/runtime` (ADR
// [0008](../../../docs/decisions/0008-public-api.md) addendum 2026-09-20). Needs neither a GPU nor
// real assets.
//
// It looks only at symbols added to or removed from what the `exports` of `package.json` expose
// (= the two surfaces `./mod.ts` and `./tune.ts`); it does not say "what should be exported". The
// "thin surface" of ADR 0008 is guarded by human review, but adding one line that re-exports an
// internal module fattens the surface at once, and a removal is a breaking change for consumers.
// Neither turns the type check red, so the check that looks at the difference lives here.
//
// The failure message carries the update procedure for when the surface changes
// (`KARUME_SURFACE=write`).

import { test } from "vite-plus/test";
import { assertPublicSurface } from "./helpers/public-surface.ts";

test("public surface: the runtime exports match the snapshot", async () => {
  await assertPublicSurface(new URL("../", import.meta.url));
});
