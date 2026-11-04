// Snapshot check of the public surface of `@karume/models` (ADR
// [0008](../../../docs/decisions/0008-public-api.md) addendum 2026-09-20). Needs neither a GPU nor
// real assets.
//
// Its role differs from `models_barrel_surface_test.ts` in the same directory:
//
// - The named check (that file) = a **declaration of intent**. A person writes "`gemma.Pipeline` is
//   exported" and "`createGemma4Ple` is not". Additions or removals of names nobody wrote pass
//   through.
// - This check = **detection of additions and removals**. It does not say what is correct; it only
//   looks at the difference from the previous snapshot for the barrel (`.`) and each of the 11
//   subpaths. It also captures type exports, so it can pin `export type` re-exports that
//   `Object.keys` cannot observe (the gap named in that file's opening NOTE).
//
// With 11 surfaces (the barrel plus per-family subpaths — ADR 0037), one surface alone shrinking or
// growing is not caught by eye.
//
// The check itself lives once on the runtime side (the same cross-package relative import as
// `container-files.ts` and others).

import { test } from "vite-plus/test";
import { assertPublicSurface } from "../../runtime/tests/helpers/public-surface.ts";

test("public surface: the models exports (barrel + 11 subpaths) match the snapshot", async () => {
  await assertPublicSurface(new URL("../", import.meta.url));
});
