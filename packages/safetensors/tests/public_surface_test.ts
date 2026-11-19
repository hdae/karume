// Snapshot check of the public surface of `@karume/safetensors`. Needs neither a GPU nor real
// assets.
//
// It detects names added to or removed from the one entry (`.`), type exports included — neither
// shows in the type check, and a removal is a breaking change for consumers. It does not say what
// should be exported.
//
// The check itself lives once on the runtime side (a cross-package relative import used by the
// tests only: the package itself depends on nothing).

import { test } from "vite-plus/test";
import { assertPublicSurface } from "../../runtime/tests/helpers/public-surface.ts";

test("public surface: the safetensors exports match the snapshot", async () => {
  await assertPublicSurface(new URL("../", import.meta.url));
});
