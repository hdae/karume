import { type GpuContext, RUNTIME_INTERNAL } from "../../src/gpu/context.ts";
import { type GeometryProfile, selectGeometryProfile } from "../../src/kernels/geometry-profile.ts";
import { defaultI8a8Geometry, type I8a8Geometry } from "../../src/kernels/i8a8-geometry.ts";

/**
 * The geometry profile Sessions on this device run with. Mirrors the selection Session
 * construction makes (the injected table, else the embedded table matched by the adapter — ADR
 * 0115), so expected pipeline keys built from it hold on any machine that matches an embedded table.
 */
export const deviceGeometryProfile = (gpu: GpuContext): GeometryProfile =>
  gpu[RUNTIME_INTERNAL].geometryProfile ?? selectGeometryProfile(gpu.adapterInfo);

/**
 * The geometry the table's `i8a8.linear` rules give an i8 linear of `rows` rows: the first rule
 * whose `maxRows` is at least `rows` (ADR 0115 addendum decision 14). Looked up here, not with the
 * runtime's selector (`i8a8LinearGeometry`) or the recipe builder's `linearI8a8Geometry`, so an
 * expected key built from it catches a wrong rule choice in either.
 */
export const expectedI8a8LinearGeometry = (
  profile: GeometryProfile,
  rows: number,
): I8a8Geometry => {
  const rule = profile.i8a8.linear.find(({ maxRows }) => maxRows >= rows);
  if (rule === undefined) {
    throw new Error(`Geometry profile '${profile.id}': no i8a8.linear rule covers ${rows} rows`);
  }
  return rule.geometry;
};

/**
 * The geometry a w4a8 linear (i4 resident weights of group length `groupSize`) of `rows` rows runs:
 * the rule's geometry ({@link expectedI8a8LinearGeometry}), or the default i8a8 linear geometry when
 * the group length is not a multiple of the rule's K tile (the w4a8 fallback of ADR 0115 addendum
 * decision 14).
 */
export const expectedW4a8LinearGeometry = (
  profile: GeometryProfile,
  rows: number,
  groupSize: number,
): I8a8Geometry => {
  const ruled = expectedI8a8LinearGeometry(profile, rows);
  return groupSize % ruled.tileK === 0 ? ruled : defaultI8a8Geometry("linear");
};
