import { type GpuContext, RUNTIME_INTERNAL } from "../../src/gpu/context.ts";
import { type GeometryProfile, selectGeometryProfile } from "../../src/kernels/geometry-profile.ts";

/**
 * The geometry profile Sessions on this device run with. Mirrors the selection Session
 * construction makes (the injected table, else the embedded table matched by the adapter — ADR
 * 0115), so expected pipeline keys built from it hold on any machine that matches an embedded table.
 */
export const deviceGeometryProfile = (gpu: GpuContext): GeometryProfile =>
  gpu[RUNTIME_INTERNAL].geometryProfile ?? selectGeometryProfile(gpu.adapterInfo);
