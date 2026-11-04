// The check that the barrel (`mod.ts`) **does not export certain names as values** (the same
// pattern as models' `models_barrel_surface_test.ts`). Needs neither a GPU nor real assets.
//
// `GpuContext` / `BatchScope` / `ResidentTensor` are exported only with `export type` to "narrow
// construction to a single entry point" (the doc in `mod.ts` is the source of truth for the reason
// — constructing them directly would create instances that bypass the planRequiredLimits /
// assertLimitsGranted and errorScope checks). Changing `export type` to `export` **still passes the
// type check**, and the public-surface snapshot check shows no diff either (`deno doc --json` does
// not distinguish a type-only re-export from a value re-export in `kind` or in `declarationKind` —
// measured 2026-09-21). That they are not exported as values is only observable at run time, so the
// check that names them lives here.

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import * as runtime from "../mod.ts";

/**
 * Names exported only as types (surfaces whose values are not exported, to close off construction).
 */
const TYPE_ONLY = ["BatchScope", "GpuContext", "ResidentTensor"];

test("barrel: surfaces that close off construction are not exported as values", () => {
  const surface = Object.keys(runtime);
  // Positive control — confirm first that it is not green because it is looking at an empty
  // namespace.
  assert(surface.includes("acquireGpu"), `the barrel import is broken (${surface.length} names)`);
  assertEquals(
    surface.filter((name) => TYPE_ONLY.includes(name)),
    [],
    "a name exported only as a type is exported as a value (construction no longer has a single entry point)",
  );
});
