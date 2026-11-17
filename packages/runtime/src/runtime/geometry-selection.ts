/**
 * The one selection point of the GEMM geometry profile a Session runs with (ADR 0115). Session
 * construction (session-build.ts), the attention route bench (tune/attention-bench.ts) and tests that
 * must name the geometry a Session ran all call {@link sessionGeometryProfile}, so the table the dp4a
 * canary verifies and the table the kernels run cannot diverge.
 * NOTE: internal — not on the public mod.ts surface. Kept in its own module so the `./tune` entry does
 * not pull Session construction in through the bench.
 */
import { type GpuContext, RUNTIME_INTERNAL } from "../gpu/device.ts";
import {
  type ResolvedGeometryProfile,
  resolveGeometryProfile,
  selectGeometryProfile,
} from "../kernels/geometry-profile.ts";

/**
 * The GEMM geometry profile a Session on `gpu` runs with: the table injected with
 * `acquireGpu({ geometryProfile })`, otherwise the embedded table {@link selectGeometryProfile}
 * picks for the adapter — resolved here, the one place the optional fields that fall back to the
 * default profile's are filled in ({@link resolveGeometryProfile}: a table without the GEMV choices
 * runs with the default's).
 */
export const sessionGeometryProfile = (gpu: GpuContext): ResolvedGeometryProfile =>
  resolveGeometryProfile(
    gpu[RUNTIME_INTERNAL].geometryProfile ?? selectGeometryProfile(gpu.adapterInfo),
  );
