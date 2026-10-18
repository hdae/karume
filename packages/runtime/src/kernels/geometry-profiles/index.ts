/**
 * The list of embedded geometry profiles (DECIDED: ADR 0115 — selection is `selectGeometryProfile`
 * in src/kernels/geometry-profile.ts).
 *
 * MUST: the default profile (`DEFAULT_GEOMETRY_PROFILE`) is **not included** — the default is the
 * fallback for an adapter that matches nothing, and listing it would give two tables with an empty
 * `match` (selection rejects that).
 * MUST: list only the constants of the `./<id>.ts` modules generated from sweeps (no side effects, a
 * single array literal). The order does not affect selection (selection rejects a list where two
 * tables match at the same rank).
 *
 * The list leaves the public surface (mod.ts), so the array and every table are deeply frozen
 * **copies** — if an application rewrote a table of the list, automatic selection for every Session in
 * the same realm would silently switch to another geometry. The freezing is not an import-time side
 * effect: it does not touch the constants of each `./<id>.ts`; it is a pure construction that builds
 * new objects and freezes them.
 */

import type { GemmGeometry } from "../gemm-geometry.ts";
import type { GeometryProfile } from "../geometry-profile.ts";
import type { I8a8Geometry } from "../i8a8-geometry.ts";
import { APPLE_METAL_3 } from "./apple-metal-3.ts";
import { NVIDIA_BLACKWELL } from "./nvidia-blackwell.ts";

/**
 * A deeply frozen copy of one table (a pure function that does not modify its argument).
 *
 * NOTE: it does not live in geometry-profile.ts because of a cycle: that module imports this list as a
 * value (the default argument of selection) — with an import order that evaluates this module first,
 * that module's `const` is not yet initialised and access throws. Every geometry is a flat object of
 * numbers only, so a one-level copy is already deep. `provenance` nests one level in `adapter` and
 * `userAgent` (an array of strings), so those are copied and frozen too.
 */
const freezeGeometryProfile = (profile: GeometryProfile): GeometryProfile => {
  const geometry = <T extends GemmGeometry | I8a8Geometry>(value: T): T =>
    Object.freeze({ ...value });
  return Object.freeze({
    id: profile.id,
    ...(profile.match === undefined ? {} : { match: Object.freeze({ ...profile.match }) }),
    gemmRows: Object.freeze(
      profile.gemmRows.map((rule) =>
        Object.freeze({ maxRows: rule.maxRows, geometry: geometry(rule.geometry) }),
      ),
    ),
    attention: Object.freeze({
      qk: geometry(profile.attention.qk),
      pv: geometry(profile.attention.pv),
    }),
    conv2d: Object.freeze({
      rows64: geometry(profile.conv2d.rows64),
      rows32: geometry(profile.conv2d.rows32),
    }),
    i8a8: Object.freeze({
      linear: geometry(profile.i8a8.linear),
      attentionQk: geometry(profile.i8a8.attentionQk),
      attentionPv: geometry(profile.i8a8.attentionPv),
    }),
    ...(profile.provenance === undefined
      ? {}
      : {
          provenance: Object.freeze({
            ...profile.provenance,
            ...(profile.provenance.userAgent === undefined
              ? {}
              : { userAgent: Object.freeze([...profile.provenance.userAgent]) }),
            adapter: Object.freeze({ ...profile.provenance.adapter }),
          }),
        }),
  });
};

export const BUILTIN_GEOMETRY_PROFILES: readonly GeometryProfile[] = Object.freeze(
  [APPLE_METAL_3, NVIDIA_BLACKWELL].map(freezeGeometryProfile),
);
