/**
 * The public reading of the online attention's device facts (`attentionDeviceFacts` of mod.ts —
 * ADR 0132 decision 5 and its addenda of 2026-10-28 and 2026-11-12): the device's subgroup and
 * workgroup-storage facts and the key-axis form of the geometry profile a Session on the device runs
 * with. Session construction, the memory estimate's callers, the attention route bench and the
 * models' headroom checks read the same facts through it, so an estimate cannot size a route the
 * Session does not take.
 *
 * NOTE: its own module because the structural core (`readAttentionDeviceFacts` of
 * attention-route.ts) stays free of GPU imports (the estimate imports that module), while this one
 * needs the profile selection of geometry-selection.ts.
 */
import type { GpuContext } from "../gpu/device.ts";
import { type AttentionDeviceFacts, readAttentionDeviceFacts } from "./attention-route.ts";
import { sessionGeometryProfile } from "./geometry-selection.ts";

/**
 * The device facts the online attention kernel's route reads on `gpu`: its enabled subgroup
 * features and measured lane mapping, its granted `maxComputeWorkgroupStorageSize`, and the
 * key-axis form chosen by the geometry profile a Session on `gpu` runs with (the injected table,
 * else the embedded one the adapter selects).
 */
export const attentionDeviceFacts = (gpu: GpuContext): AttentionDeviceFacts =>
  readAttentionDeviceFacts(gpu, sessionGeometryProfile(gpu));
