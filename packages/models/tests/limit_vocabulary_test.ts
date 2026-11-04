/**
 * Check that the **vocabulary of limit names** in `requiredLimits` matches between hub and runtime.
 *
 * hub's `RequiredLimitName` (the manifest-owned allowlist — ADR 0038 §3 / §7) and runtime's
 * `REQUIRED_LIMIT_KEYS` (the requiredLimits keys `acquireGpu` passes to `requestDevice` —
 * src/gpu/device.ts) are **copies of the same vocabulary**, but hub does not depend on runtime,
 * so neither can reference the other. When a name is added to or removed from only one side, both
 * type checks stay green, and the drift shows up only as "a limit the manifest declared is
 * silently not requested" = a **silent cap**. `@karume/models` is the only place that can depend
 * on both, so the detection lives here.
 *
 * MUST: detection has two stages, shaped so that a change on either side **demands the matching
 * change on the other**.
 * ① {@link HUB_LIMIT_NAMES} is type-checked as `Record<RequiredLimitName, true>`, so adding or
 *    removing a name on the hub side fails this file's **type check**.
 * ② The table's key set is compared at run time with runtime's `REQUIRED_LIMIT_KEYS`, so adding
 *    or removing one on the runtime side fails **this test**.
 *
 * NOTE: the table is spelled out by hand because a union cannot be enumerated at run time (this
 * is not duplicated derived state — ① pins the spelling through the type, so it cannot drift
 * silently). The runtime vocabulary is not on mod.ts's public surface (the thin surface — ADR
 * 0008), so it is imported relatively from src. This checks without widening the public surface;
 * cross-package relative imports from tests follow an existing precedent
 * (`../../runtime/tests/helpers/`).
 */

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { RequiredLimitName } from "@karume/hub";
import { REQUIRED_LIMIT_KEYS } from "../../runtime/src/gpu/device.ts";

/** hub's vocabulary as a value table (the type enforces that it covers `RequiredLimitName`). */
const HUB_LIMIT_NAMES = {
  maxBufferSize: true,
  maxStorageBufferBindingSize: true,
  maxUniformBufferBindingSize: true,
  maxStorageBuffersPerShaderStage: true,
  maxUniformBuffersPerShaderStage: true,
  maxComputeWorkgroupStorageSize: true,
  maxComputeInvocationsPerWorkgroup: true,
  maxComputeWorkgroupSizeX: true,
  maxComputeWorkgroupSizeY: true,
  maxComputeWorkgroupSizeZ: true,
  maxComputeWorkgroupsPerDimension: true,
} as const satisfies Record<RequiredLimitName, true>;

test("requiredLimits limit names match exactly between hub's allowlist and runtime's requested keys", () => {
  assertEquals(
    Object.keys(HUB_LIMIT_NAMES).toSorted(),
    [...REQUIRED_LIMIT_KEYS].toSorted(),
    "hub's RequiredLimitName and runtime's REQUIRED_LIMIT_KEYS have drifted apart — " +
      "copy the new name to the other side (added only on the hub side, a request the manifest accepted " +
      "never reaches the device; added only on the runtime side, the distribution cannot declare that limit)",
  );
});
