// Selection of GEMM geometry profiles (src/kernels/geometry-profile.ts — ADR 0115) and identity of
// the defaults (no GPU needed).
//
// The focus of the checks is 3 points:
//
// 1. **Selection is deterministic** (an exact match down to description > (vendor, architecture) >
//    vendor only > default; 2 tables of the same rank fail regardless of the adapter; a table
//    without `match` is never selected and is only used by injection). If the selection wobbles
//    with the order of the list or the needs of other machines, the correspondence between the
//    reference sha rows per environment key and "which geometry it ran with" breaks.
// 2. **The default profile has the same values as the existing selection**, and even when the
//    derivation phase passes it explicitly, keys and WGSL are byte-identical to omitting it (= not
//    a single byte moves on a default machine).
// 3. **A broken table fails at the check of Session construction** (rules not in ascending order, a
//    last rule that is not Infinity, a geometry that breaks exact division). A table injected
//    through `acquireGpu({ geometryProfile })` fails **before the device is created**
//    (navigator.gpu is swapped to see that requestDevice is never reached). The callback form (ADR
//    0117 decision 6) is called exactly once with the adapter's information, its return goes
//    through the same check, and neither a thrown exception nor a Promise return creates a device.
// 4. **The check is a type predicate that narrows unknown to a table**, and names a missing field
//    or a wrong type with the check's error instead of a TypeError.
// 5. **An injected table fails against the device's workgroup limits** (thread count, the
//    components of `@workgroup_size`, shared memory — against the adapter's limits, before the
//    device is created). The derivation of shared memory (a pure function) is compared with the
//    total counted from the `var<workgroup>` declarations codegen writes, and the embedded tables
//    and the default table pass under WebGPU's default limits.

import { platformObject } from "@karume/test-support/webgpu";
import {
  assert,
  assertEquals,
  assertNotEquals,
  assertRejects,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import { describe, it } from "vite-plus/test";
// Test only: a way to create a Promise of another realm (the runtime body uses only Web standard
// APIs — this is a dependency of the tests)
import { createContext, runInContext } from "node:vm";
import { CodegenError } from "../src/codegen/errors.ts";
import {
  acquireGpu,
  GpuFeatureError,
  REQUIRED_LIMIT_KEYS,
  RUNTIME_INTERNAL,
} from "../src/gpu/device.ts";
import { fakeDevice } from "./helpers/fake-gpu.ts";
import {
  assertGeometryProfile,
  assertGeometryProfileWithinLimits,
  conv2dProfileGeometry,
  DEFAULT_GEOMETRY_PROFILE,
  gemmRowsGeometry,
  type GeometryProfile,
  selectGeometryProfile,
  type WorkgroupLimits,
} from "../src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../src/kernels/geometry-profiles/index.ts";
import { APPLE_METAL_3 } from "../src/kernels/geometry-profiles/apple-metal-3.ts";
import { NVIDIA_AMPERE } from "../src/kernels/geometry-profiles/nvidia-ampere.ts";
import { NVIDIA_BLACKWELL } from "../src/kernels/geometry-profiles/nvidia-blackwell.ts";
import {
  assertGemmGeometry,
  defaultGemmGeometry,
  GEMM_TILE_K,
  type GemmGeometry,
  gemmGeometryForRows,
  gemmQuadsPerThread,
  gemmTileM,
  gemmTileN,
} from "../src/kernels/gemm-geometry.ts";
import {
  defaultI8a8Geometry,
  type I8a8Geometry,
  i8a8WorkgroupStorageBytes,
} from "../src/kernels/i8a8-geometry.ts";
import {
  type GemmCompute,
  gemmMTileGeometry,
  gemmWorkgroupStorageBytes,
} from "../src/kernels/gemm.ts";
import { matmulKey, matmulWgsl } from "../src/kernels/matmul.ts";
import { bmmKey, bmmWgsl } from "../src/kernels/bmm.ts";
import { linearKey, linearWgsl } from "../src/kernels/linear.ts";
import {
  attentionPvKey,
  attentionPvWgsl,
  attentionQkKey,
  attentionQkWgsl,
} from "../src/kernels/attention.ts";
import { conv2dIgemmKey, conv2dIgemmWgsl } from "../src/kernels/conv2d.ts";
import { linearI8a8Key, linearI8a8Wgsl } from "../src/kernels/linear-i8a8.ts";
import { conv3dI8a8Key, conv3dI8a8Wgsl } from "../src/kernels/conv3d-i8a8.ts";
import {
  attentionPvI8a8Key,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
  attentionQkI8a8Wgsl,
} from "../src/kernels/attention-i8a8.ts";

/**
 * A geometry distinguishable from the default (`reg64x32r4x4w8` / `reg64x64r8x4w16`, which won the
 * M2 sweep).
 */
const M64N32: GemmGeometry = { regM: 4, regN: 4, wgX: 8, wgY: 16 };
const M64N64: GemmGeometry = { regM: 8, regN: 4, wgX: 16, wgY: 8 };

/**
 * A table that replaces only id and match of the default profile (the geometry's content does not
 * matter for the selection checks). If `match` is undefined, **the key itself is omitted** (the
 * shape of an injection-only table, not a table with a `match: undefined` field).
 */
const profileOf = (
  id: string,
  match: GeometryProfile["match"],
  overrides: Partial<GeometryProfile> = {},
): GeometryProfile => {
  const { match: _defaultMatch, ...geometry } = DEFAULT_GEOMETRY_PROFILE;
  return match === undefined
    ? { ...geometry, id, ...overrides }
    : { ...geometry, id, match, ...overrides };
};

const APPLE = profileOf("apple", { vendor: "apple" });
const APPLE_METAL3 = profileOf("apple-metal-3", { vendor: "apple", architecture: "metal-3" });
const APPLE_M2 = profileOf("apple-m2", {
  vendor: "apple",
  architecture: "metal-3",
  description: "Apple M2",
});
/** An injection-only table without `match` (outside automatic selection). */
const OPT_IN = profileOf("opt-in", undefined);

/**
 * Representative points of the table (for each bucket, "exactly on the boundary" and "1 past the
 * boundary").
 */
const BUCKET_ROWS = [1, 64, 65, 512, 513, 4096] as const;

describe("selection order of selectGeometryProfile", () => {
  it("an exact (vendor, architecture) match wins over a vendor-only rule, regardless of the list order", () => {
    const adapter = { vendor: "apple", architecture: "metal-3" };
    assertStrictEquals(selectGeometryProfile(adapter, [APPLE, APPLE_METAL3]), APPLE_METAL3);
    assertStrictEquals(selectGeometryProfile(adapter, [APPLE_METAL3, APPLE]), APPLE_METAL3);
  });

  it("a vendor-only rule matches adapters with a different architecture (including the empty string = Deno)", () => {
    const profiles = [APPLE, APPLE_METAL3];
    assertStrictEquals(
      selectGeometryProfile({ vendor: "apple", architecture: "metal-2" }, profiles),
      APPLE,
    );
    assertStrictEquals(
      selectGeometryProfile({ vendor: "apple", architecture: "" }, profiles),
      APPLE,
    );
  });

  it("a rule with an architecture does not match a different architecture even if the vendor matches", () => {
    assertStrictEquals(
      selectGeometryProfile({ vendor: "apple", architecture: "metal-2" }, [APPLE_METAL3]),
      DEFAULT_GEOMETRY_PROFILE,
    );
  });

  it("an adapter matching nothing falls back to the default profile (matching is exact, with no case folding)", () => {
    const profiles = [APPLE, APPLE_METAL3];
    for (const adapter of [
      { vendor: "32902", architecture: "" },
      { vendor: "Apple", architecture: "metal-3" },
      { vendor: "", architecture: "" },
    ]) {
      assertStrictEquals(
        selectGeometryProfile(adapter, profiles),
        DEFAULT_GEOMETRY_PROFILE,
        JSON.stringify(adapter),
      );
    }
    assertStrictEquals(
      selectGeometryProfile({ vendor: "apple", architecture: "metal-3" }, []),
      DEFAULT_GEOMETRY_PROFILE,
    );
  });

  it("a list with 2 matches at the same rank fails loudly even on a machine other than that match", () => {
    // It fails even when selecting on another machine (intel here) — not something visible only in
    // the CI of the colliding machine
    const unrelated = { vendor: "intel", architecture: "xe2" };
    assertThrows(
      () =>
        selectGeometryProfile(unrelated, [
          APPLE_METAL3,
          profileOf("apple-metal-3-copy", { vendor: "apple", architecture: "metal-3" }),
        ]),
      CodegenError,
      "two tables would match at the same rank",
    );
    assertThrows(
      () => selectGeometryProfile(unrelated, [APPLE, profileOf("apple-2", { vendor: "apple" })]),
      CodegenError,
      "two tables would match at the same rank",
    );
  });

  it("a table matching every adapter, an architecture without a vendor, an empty-string match and a duplicate id fail loudly", () => {
    const adapter = { vendor: "apple", architecture: "metal-3" };
    const cases: readonly (readonly [string, GeometryProfile, string])[] = [
      ["empty match", profileOf("everything", {}), "only the default profile"],
      [
        "architecture only",
        profileOf("arch-only", { architecture: "metal-3" }),
        "together with vendor",
      ],
      [
        "empty-string vendor",
        profileOf("empty-vendor", { vendor: "" }),
        "must not be an empty string",
      ],
      [
        "empty-string architecture",
        profileOf("empty-arch", { vendor: "apple", architecture: "" }),
        "must not be an empty string",
      ],
      ["same id as the default", profileOf("default", { vendor: "apple" }), "duplicate id"],
    ];
    for (const [name, profile, message] of cases) {
      assertThrows(() => selectGeometryProfile(adapter, [profile]), CodegenError, message, name);
    }
    assertThrows(
      () => selectGeometryProfile(adapter, [APPLE, profileOf("apple", { vendor: "intel" })]),
      CodegenError,
      "duplicate id",
    );
  });

  it("every embedded table passes the check, the default is not included, and each table with a match is selected on its own match's adapter", () => {
    // An empty adapter (missing adapter.info in old Chromium) falls back to the default
    assertStrictEquals(
      selectGeometryProfile({ vendor: "", architecture: "", description: "" }),
      DEFAULT_GEOMETRY_PROFILE,
    );
    for (const profile of BUILTIN_GEOMETRY_PROFILES) {
      assertGeometryProfile(profile);
      assert(
        profile.id !== DEFAULT_GEOMETRY_PROFILE.id,
        `${profile.id}: the default is in the list`,
      );
      // A table without match is injection-only (outside automatic selection — that it is never
      // selected is checked by the selection-order tests)
      const { match } = profile;
      if (match === undefined) continue;
      const vendor = match.vendor;
      assertEquals(typeof vendor, "string", `${profile.id}: an embedded table without a vendor`);
      // A table without an architecture matches adapters that return no architecture (Deno)
      const adapter = {
        vendor: vendor ?? "",
        architecture: match.architecture ?? "",
        description: match.description ?? "",
      };
      assertStrictEquals(selectGeometryProfile(adapter), profile, profile.id);
      // A table matched by description does not match adapters whose description is unavailable
      // (Chrome without the flag)
      if (match.description !== undefined) {
        assert(
          selectGeometryProfile({ ...adapter, description: "" }) !== profile,
          `${profile.id}: matched an adapter without a description`,
        );
      }
    }
  });
});

/**
 * Whether the value and every object / array reachable from it is frozen (the first path that is
 * not frozen).
 */
const unfrozenPath = (value: unknown, path: string): string | undefined => {
  if (typeof value !== "object" || value === null) return undefined;
  if (!Object.isFrozen(value)) return path;
  for (const [key, child] of Object.entries(value)) {
    const found = unfrozenPath(child, `${path}.${key}`);
    if (found !== undefined) return found;
  }
  return undefined;
};

describe("the public list BUILTIN_GEOMETRY_PROFILES is a deeply frozen copy", () => {
  it("the list and every nesting of each table are frozen, and writes fail with TypeError", () => {
    assertEquals(unfrozenPath(BUILTIN_GEOMETRY_PROFILES, "BUILTIN_GEOMETRY_PROFILES"), undefined);
    const [first] = BUILTIN_GEOMETRY_PROFILES;
    assertThrows(() => {
      (first.gemmRows[0].geometry as { regM: number }).regM = 1;
    }, TypeError);
    assertThrows(() => {
      (BUILTIN_GEOMETRY_PROFILES as GeometryProfile[]).push(DEFAULT_GEOMETRY_PROFILE);
    }, TypeError);
  });

  it("the values equal the generated constants, and the constants themselves are not frozen (loading does not rewrite another module's values)", () => {
    // A copy missing a field (the frozen copy forgetting to copy a new field of the type) also
    // fails here
    assertEquals(BUILTIN_GEOMETRY_PROFILES, [APPLE_METAL_3, NVIDIA_AMPERE, NVIDIA_BLACKWELL]);
    for (const profile of [APPLE_METAL_3, NVIDIA_AMPERE, NVIDIA_BLACKWELL]) {
      assert(!Object.isFrozen(profile), `${profile.id}: the generated constant was frozen`);
      assert(
        !Object.isFrozen(profile.gemmRows[0].geometry),
        `${profile.id}: the geometry was frozen`,
      );
    }
  });
});

describe("matching by description and injection-only tables without match", () => {
  it("a rule matching down to description wins over a rule of (vendor, architecture) only, regardless of the list order", () => {
    const adapter = { vendor: "apple", architecture: "metal-3", description: "Apple M2" };
    for (const profiles of [
      [APPLE, APPLE_METAL3, APPLE_M2],
      [APPLE_M2, APPLE_METAL3, APPLE],
      [APPLE_METAL3, APPLE_M2],
    ]) {
      assertStrictEquals(selectGeometryProfile(adapter, profiles), APPLE_M2);
    }
  });

  it("a description rule does not match adapters whose description is empty, missing or different, and falls to the next rank", () => {
    const profiles = [APPLE, APPLE_METAL3, APPLE_M2];
    for (const adapter of [
      { vendor: "apple", architecture: "metal-3", description: "" },
      { vendor: "apple", architecture: "metal-3" },
      { vendor: "apple", architecture: "metal-3", description: "Apple M5" },
      // No prefix match, no substring match, no case folding
      { vendor: "apple", architecture: "metal-3", description: "Apple M2 Pro" },
      { vendor: "apple", architecture: "metal-3", description: "apple m2" },
    ]) {
      assertStrictEquals(
        selectGeometryProfile(adapter, profiles),
        APPLE_METAL3,
        JSON.stringify(adapter),
      );
    }
    // Without a (vendor, architecture) rule it falls to the vendor-only rule → the default, in that
    // order
    const m5 = { vendor: "apple", architecture: "metal-3", description: "Apple M5" };
    assertStrictEquals(selectGeometryProfile(m5, [APPLE, APPLE_M2]), APPLE);
    assertStrictEquals(selectGeometryProfile(m5, [APPLE_M2]), DEFAULT_GEOMETRY_PROFILE);
    // Even with the same description, a different architecture does not match
    assertStrictEquals(
      selectGeometryProfile({ vendor: "apple", architecture: "metal-2", description: "Apple M2" }, [
        APPLE_M2,
      ]),
      DEFAULT_GEOMETRY_PROFILE,
    );
  });

  it("a table without match matches no adapter (it falls to the default even if every field it would compare matches)", () => {
    assert(!Object.hasOwn(OPT_IN, "match"), "the table under test has the match key");
    // Even if the omitted table's content equals another rule's table, it stays outside the
    // selection
    const adapters = [
      { vendor: "apple", architecture: "metal-3", description: "Apple M2" },
      { vendor: "", architecture: "", description: "" },
      { vendor: "nvidia", architecture: "blackwell" },
    ];
    for (const adapter of adapters) {
      assertStrictEquals(
        selectGeometryProfile(adapter, [OPT_IN]),
        DEFAULT_GEOMETRY_PROFILE,
        JSON.stringify(adapter),
      );
    }
    // Even placed next to other rules, the selection is decided by the other rules alone
    assertStrictEquals(selectGeometryProfile(adapters[0], [OPT_IN, APPLE]), APPLE);
  });

  it("a table without match still goes through the check and the duplicate-id check (a broken table left in the list must not go unnoticed)", () => {
    const adapter = { vendor: "apple", architecture: "metal-3" };
    assertThrows(
      () => selectGeometryProfile(adapter, [OPT_IN, profileOf("opt-in", { vendor: "apple" })]),
      CodegenError,
      "duplicate id",
    );
    assertThrows(
      () =>
        selectGeometryProfile(adapter, [profileOf("opt-in-broken", undefined, { gemmRows: [] })]),
      CodegenError,
      "gemmRows is empty",
    );
    // Omitted tables do not count as same-rank collisions (tables without match never compete in
    // matching)
    assertStrictEquals(
      selectGeometryProfile(adapter, [OPT_IN, profileOf("opt-in-2", undefined)]),
      DEFAULT_GEOMETRY_PROFILE,
    );
  });

  it("a match of description only, of vendor and description only, or with an empty-string description fails the check", () => {
    const adapter = { vendor: "apple", architecture: "metal-3", description: "Apple M2" };
    const cases: readonly (readonly [string, GeometryProfile, string])[] = [
      [
        "description only",
        profileOf("desc-only", { description: "Apple M2" }),
        "together with both vendor and architecture",
      ],
      [
        "vendor and description only",
        profileOf("vendor-desc", { vendor: "apple", description: "Apple M2" }),
        "together with both vendor and architecture",
      ],
      [
        "architecture and description only",
        profileOf("arch-desc", { architecture: "metal-3", description: "Apple M2" }),
        "together with vendor",
      ],
      [
        "empty-string description",
        profileOf("empty-desc", { vendor: "apple", architecture: "metal-3", description: "" }),
        "must not be an empty string",
      ],
    ];
    for (const [name, profile, message] of cases) {
      assertThrows(() => assertGeometryProfile(profile), CodegenError, message, name);
      assertThrows(() => selectGeometryProfile(adapter, [profile]), CodegenError, message, name);
    }
  });

  it("2 tables with the same (vendor, architecture, description) fail loudly even on another machine, and can sit side by side if the description differs", () => {
    const unrelated = { vendor: "intel", architecture: "xe2", description: "Intel Arc B570" };
    assertThrows(
      () =>
        selectGeometryProfile(unrelated, [
          APPLE_M2,
          profileOf("apple-m2-copy", {
            vendor: "apple",
            architecture: "metal-3",
            description: "Apple M2",
          }),
        ]),
      CodegenError,
      "two tables would match at the same rank",
    );
    const m5 = profileOf("apple-m5", {
      vendor: "apple",
      architecture: "metal-3",
      description: "Apple M5",
    });
    const profiles = [APPLE_METAL3, APPLE_M2, m5];
    assertStrictEquals(
      selectGeometryProfile(
        { vendor: "apple", architecture: "metal-3", description: "Apple M5" },
        profiles,
      ),
      m5,
    );
    assertStrictEquals(selectGeometryProfile(unrelated, profiles), DEFAULT_GEOMETRY_PROFILE);
  });
});

describe("DEFAULT_GEOMETRY_PROFILE has the same values as the existing selection", () => {
  it("match is empty, id is default, and it passes its own check", () => {
    assertEquals(DEFAULT_GEOMETRY_PROFILE.id, "default");
    assertEquals(DEFAULT_GEOMETRY_PROFILE.match, {});
    assertEquals(DEFAULT_GEOMETRY_PROFILE.provenance, undefined);
    assertGeometryProfile(DEFAULT_GEOMETRY_PROFILE);
  });

  it("the row-count buckets match gemmGeometryForRows at rows 1 / 64 / 65 / 512 / 513 / 4096", () => {
    for (const rows of [0, ...BUCKET_ROWS, 0xffff_ffff]) {
      assertEquals(
        gemmRowsGeometry(DEFAULT_GEOMETRY_PROFILE, rows),
        gemmGeometryForRows(rows),
        `M=${rows}`,
      );
    }
  });

  it("the attention / conv2d / i8a8 fields match the existing defaults (defaultGemmGeometry / gemmMTileGeometry / defaultI8a8Geometry)", () => {
    assertEquals(DEFAULT_GEOMETRY_PROFILE.attention, {
      qk: defaultGemmGeometry(),
      pv: defaultGemmGeometry(),
    });
    assertEquals(DEFAULT_GEOMETRY_PROFILE.conv2d, {
      rows64: gemmMTileGeometry(64),
      rows32: gemmMTileGeometry(32),
    });
    assertEquals(DEFAULT_GEOMETRY_PROFILE.i8a8, {
      linear: defaultI8a8Geometry("linear"),
      attentionQk: defaultI8a8Geometry("attention_qk"),
      attentionPv: defaultI8a8Geometry("attention_pv"),
      conv3d: defaultI8a8Geometry("conv3d"),
    });
  });
});

// The conv3d w8a8 field (ADR 0115 addendum 2026-10-26). Before it existed, conv3d ran linear's
// default geometry whatever the adapter; the default profile keeps that geometry, so a machine
// without a matching table sees the same key and WGSL as before.
describe("i8a8.conv3d: the conv3d w8a8 geometry is a required profile field", () => {
  it("the default profile's conv3d is the geometry conv3d ran before the field (M128N64 r8×8 wg8×16 K16) — key and WGSL unchanged", () => {
    const geometry = DEFAULT_GEOMETRY_PROFILE.i8a8.conv3d;
    assertEquals(geometry, { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 });
    assertEquals(geometry, defaultI8a8Geometry("linear"));
    assertEquals(
      conv3dI8a8Key(true, true, geometry),
      "conv3d:v1:i8a8:tile128x64r8x8w8x16k16v4:dp4a",
    );
    assertEquals(
      conv3dI8a8Key(false, false, geometry),
      "conv3d:v1:i8a8:tile128x64r8x8w8x16k16:dp4aEmu",
    );
    // The WGSL of the default profile is the one built from linear's default geometry (the
    // codegen snapshots conv3d_i8a8*.wgsl freeze its bytes)
    assertEquals(
      conv3dI8a8Wgsl(true, true, geometry),
      conv3dI8a8Wgsl(true, true, defaultI8a8Geometry("linear")),
    );
  });

  it("a table without i8a8.conv3d, or with a broken one, is rejected naming the field (selection and injection)", async () => {
    const { conv3d: _conv3d, ...withoutConv3d } = DEFAULT_GEOMETRY_PROFILE.i8a8;
    const missing = { ...OPT_IN, id: "no-conv3d", i8a8: withoutConv3d };
    assertThrows(
      () => assertGeometryProfile(missing),
      CodegenError,
      "i8a8.conv3d is not an object",
    );
    const broken = profileOf(
      "conv3d-hole",
      { vendor: "test" },
      {
        i8a8: {
          ...DEFAULT_GEOMETRY_PROFILE.i8a8,
          conv3d: { regM: 8, regN: 8, wgX: 16, wgY: 8, tileK: 6 },
        },
      },
    );
    assertThrows(
      () => selectGeometryProfile({ vendor: "test", architecture: "" }, [broken]),
      CodegenError,
      "i8a8.conv3d",
    );
    const requests = await withCountingGpu(async () => {
      await assertRejects(
        () => acquireGpu({ geometryProfile: broken }),
        GpuFeatureError,
        "i8a8.conv3d",
      );
    });
    assertEquals(requests, 0, "a broken conv3d geometry reached requestDevice");
  });

  it("the limit guard checks i8a8.conv3d and names it", () => {
    const profile = profileOf("conv3d-storage", undefined, {
      i8a8: { ...DEFAULT_GEOMETRY_PROFILE.i8a8, conv3d: I8A8_STORAGE_32K },
    });
    assertThrows(
      () => assertGeometryProfileWithinLimits(profile, WEBGPU_DEFAULT_LIMITS),
      CodegenError,
      "i8a8.conv3d (regM=8 regN=8 wgX=16 wgY=16 tileK=128): shared memory (bytes) 32768 exceeds maxComputeWorkgroupStorageSize 16384",
    );
    assertGeometryProfileWithinLimits(profile, {
      ...WEBGPU_DEFAULT_LIMITS,
      maxComputeWorkgroupStorageSize: 32_768,
    });
  });

  it("every embedded table carries i8a8.conv3d, and selection returns it", () => {
    for (const source of [APPLE_METAL_3, NVIDIA_AMPERE, NVIDIA_BLACKWELL]) {
      const { match } = source;
      assert(match?.vendor !== undefined, source.id);
      const selected = selectGeometryProfile({
        vendor: match.vendor,
        architecture: match.architecture ?? "",
        ...(match.description === undefined ? {} : { description: match.description }),
      });
      assertEquals(selected.id, source.id);
      assertEquals(selected.i8a8.conv3d, source.i8a8.conv3d, source.id);
      assert(Object.isFrozen(selected.i8a8.conv3d), `${source.id}: conv3d is not frozen`);
    }
  });
});

// The derivation phase (src/runtime/recipe-builders/) now reads the geometry from the profile and
// passes it **explicitly**. The unit version of "not a single byte moves on a default machine" is
// "keys and WGSL with the default profile's values passed explicitly are byte-identical to the call
// before its introduction (omitted)". The codegen snapshots pin the omitted form, so this connects
// the two.
describe("passing the default profile's geometry explicitly keeps keys and WGSL byte-identical to omitting it", () => {
  const profile = DEFAULT_GEOMETRY_PROFILE;

  it("matmul / bmm / linear (row-count buckets)", () => {
    for (const rows of BUCKET_ROWS) {
      const geometry = gemmRowsGeometry(profile, rows);
      for (const v4 of [true, false]) {
        assertEquals(matmulKey(v4, rows, geometry), matmulKey(v4, rows), `matmul M=${rows}`);
        assertEquals(matmulWgsl(v4, rows, geometry), matmulWgsl(v4, rows), `matmul M=${rows}`);
        assertEquals(bmmKey(v4, rows, undefined, geometry), bmmKey(v4, rows), `bmm M=${rows}`);
        assertEquals(bmmWgsl(v4, rows, undefined, geometry), bmmWgsl(v4, rows), `bmm M=${rows}`);
      }
      for (const [weight, v4, compute] of [
        ["f32", true, "f32"],
        ["f16", false, "f32"],
        ["f16", true, "f16"],
        ["i8", true, "f32"],
      ] as const) {
        const where = `linear ${weight}/${v4}/${compute} M=${rows}`;
        assertEquals(
          linearKey(weight, v4, compute, rows, undefined, geometry),
          linearKey(weight, v4, compute, rows),
          where,
        );
        assertEquals(
          linearWgsl(weight, v4, compute, rows, undefined, geometry),
          linearWgsl(weight, v4, compute, rows),
          where,
        );
      }
      assertEquals(
        linearKey("i4", true, "f32", rows, 32, geometry),
        linearKey("i4", true, "f32", rows, 32),
      );
      assertEquals(
        linearWgsl("i4", true, "f32", rows, 32, geometry),
        linearWgsl("i4", true, "f32", rows, 32),
      );
    }
  });

  it("①QK / ③PV of fused attention f32 (across the variant axes)", () => {
    const { qk, pv } = profile.attention;
    for (const [v4, compute, score, mask, gqa, rowWindow] of [
      [true, "f32", "f32", false, false, false],
      [false, "f32", "f32", false, false, false],
      [true, "f16", "f32", false, false, false],
      [true, "f32", "f16", true, false, false],
      [true, "f32", "f32", true, true, true],
    ] as const) {
      const where = `v4=${v4} ${compute} s=${score} mask=${mask} gqa=${gqa} rw=${rowWindow}`;
      assertEquals(
        attentionQkKey(v4, compute, score, mask, gqa, rowWindow, qk),
        attentionQkKey(v4, compute, score, mask, gqa, rowWindow),
        where,
      );
      assertEquals(
        attentionQkWgsl(v4, compute, score, mask, gqa, rowWindow, qk),
        attentionQkWgsl(v4, compute, score, mask, gqa, rowWindow),
        where,
      );
      assertEquals(
        attentionPvKey(v4, compute, score, gqa, rowWindow, pv),
        attentionPvKey(v4, compute, score, gqa, rowWindow),
        where,
      );
      assertEquals(
        attentionPvWgsl(v4, compute, score, gqa, rowWindow, pv),
        attentionPvWgsl(v4, compute, score, gqa, rowWindow),
        where,
      );
    }
  });

  it("implicit GEMM of conv2d (the 64-row / 32-row classes of the m tile)", () => {
    for (const mTile of [64, 32]) {
      const geometry = conv2dProfileGeometry(profile, mTile);
      for (const [weight, v4] of [
        ["f32", true],
        ["f16", false],
        ["i8", true],
      ] as const) {
        assertEquals(
          conv2dIgemmKey(weight, v4, mTile, geometry),
          conv2dIgemmKey(weight, v4, mTile),
          `${weight} mTile=${mTile}`,
        );
        assertEquals(
          conv2dIgemmWgsl(weight, v4, mTile, geometry),
          conv2dIgemmWgsl(weight, v4, mTile),
          `${weight} mTile=${mTile}`,
        );
      }
    }
  });

  it("i8a8 linear / fused attention ①QK / ③PV", () => {
    const { linear, attentionQk, attentionPv } = profile.i8a8;
    for (const [v4, dp4a] of [
      [true, true],
      [false, false],
    ] as const) {
      assertEquals(linearI8a8Key(v4, dp4a, linear), linearI8a8Key(v4, dp4a));
      assertEquals(linearI8a8Wgsl(v4, dp4a, linear), linearI8a8Wgsl(v4, dp4a));
      assertEquals(
        linearI8a8Key(v4, dp4a, linear, "i4", 32),
        linearI8a8Key(v4, dp4a, undefined, "i4", 32),
      );
      // f16 storage of S (s16) is v4-path only (the generation check rejects it)
      for (const score of v4 ? (["f32", "f16"] as const) : (["f32"] as const)) {
        assertEquals(
          attentionQkI8a8Key(v4, dp4a, score, attentionQk),
          attentionQkI8a8Key(v4, dp4a, score),
        );
        assertEquals(
          attentionQkI8a8Wgsl(v4, dp4a, score, attentionQk),
          attentionQkI8a8Wgsl(v4, dp4a, score),
        );
        assertEquals(
          attentionPvI8a8Key(v4, dp4a, score, attentionPv),
          attentionPvI8a8Key(v4, dp4a, score),
        );
        assertEquals(
          attentionPvI8a8Wgsl(v4, dp4a, score, attentionPv),
          attentionPvI8a8Wgsl(v4, dp4a, score),
        );
      }
    }
  });
});

// Selection of the tile geometry for skinny M (`gemmGeometryForRows` of
// src/kernels/gemm-geometry.ts) and how it flows into the keys and generated code of matmul / bmm /
// linear (no GPU needed).
//
// The focus of the checks is 3 points:
//
// 1. **The bucket boundaries** follow the table, and M ≥ 128 stays on the default geometry (the
//    boundary that does not move the existing keys and existing performance of Anima / SBV2).
// 2. **Keys and generated code come from the same geometry**. If only one of them forgets to pass
//    the row count, the WGSL of another geometry in the cache does not mesh with the executor's
//    dispatch count, and output tiles go missing without an exception.
// 3. **Changing the geometry does not move the numeric contract** (K tile 16, outer t ascending,
//    inner kk ascending, the text of the multiply-add). This pins the text; the bit equality of the
//    values itself is checked on a real GPU by tests/gpu_gemm_skinny_test.ts.

/**
 * Representative points of the table (for each bucket, "inside the boundary" and "1 past the
 * boundary").
 */
const ROW_SAMPLES = [1, 4, 16, 17, 64, 65, 128, 4096] as const;

describe("gemmGeometryForRows buckets and the keys / WGSL they produce", () => {
  it("the bucket boundaries of gemmGeometryForRows are 64 / 512, and M >= 513 falls to the default geometry", () => {
    // Small-M bucket (2026-08-11 sweep: M16N16 was best at ×3.0–3.2 against the default for
    // M=1/4/64. M=64 is 4 row tiles = still a win in workgroup count even reading the weights 4
    // times)
    for (const rows of [0, 1, 2, 4, 16, 17, 32, 63, 64]) {
      assertEquals(gemmGeometryForRows(rows), { regM: 1, regN: 4, wgX: 4, wgY: 16 }, `M=${rows}`);
    }
    // Medium-M bucket (sweep M=318 ×1.67 / M=512 ×1.28. Adopted together with the E2E A/B of
    // Anima/SBV2 in iteration ①)
    for (const rows of [65, 127, 128, 129, 318, 512]) {
      assertEquals(gemmGeometryForRows(rows), { regM: 4, regN: 4, wgX: 8, wgY: 16 }, `M=${rows}`);
    }
    // MUST: above this stays default (outside the sweep's measured points — do not move the DiT
    // selection measured on 2026-08-10)
    for (const rows of [513, 1024, 4096, 65536]) {
      assertEquals(gemmGeometryForRows(rows), defaultGemmGeometry(), `M=${rows}`);
    }
  });

  it("every bucket geometry meets the exact-division condition of the fill (a tile shape with holes cannot be put in the table)", () => {
    for (const rows of ROW_SAMPLES) {
      const geometry = gemmGeometryForRows(rows);
      // A failing combination is "words the fill never reaches enter the dot product as 0" = a
      // wrong value without an exception
      assertGemmGeometry(geometry, `M=${rows}`);
      // Shared memory is within the WebGPU default limit of 16,384 B (sa = tileM·16 f32 + sb =
      // 16·(tileN/4) vec4)
      const shared =
        gemmTileM(geometry) * GEMM_TILE_K * 4 + GEMM_TILE_K * (gemmTileN(geometry) / 4) * 16;
      assertEquals(shared <= 16384, true, `M=${rows}: shared ${shared} B`);
      // The small-M bucket stays within 4 row tiles (= reading the weights 4 times). The sweep's
      // conclusion is "adding workgroups wins even at the cost of more re-reads", but growing
      // without bound goes back to being transfer-bound
      if (rows <= 64) {
        assertEquals(
          Math.ceil(Math.max(rows, 1) / gemmTileM(geometry)) <= 4,
          true,
          `M=${rows}: 5 or more row tiles`,
        );
      }
    }
  });

  it("gemmGeometryForRows fails loudly on anything but a non-negative integer", () => {
    for (const rows of [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      assertThrows(
        () => gemmGeometryForRows(rows),
        CodegenError,
        "row count M must be a non-negative u32 integer",
      );
    }
  });

  it("the row count goes into the key, and different buckets make different pipelines", () => {
    // The geometry discriminator `r{regM}x{regN}w{wgX}` determines the geometry uniquely together
    // with the tile edges (wgY is tileM/regM)
    assertEquals(linearKey("f32", true, "f32", 4), "linear:v2:f32:reg16x16r1x4w4v4");
    assertEquals(linearKey("f32", true, "f32", 64), "linear:v2:f32:reg16x16r1x4w4v4");
    assertEquals(linearKey("f32", true, "f32", 318), "linear:v2:f32:reg64x32r4x4w8v4");
    assertEquals(matmulKey(false, 4), "matmul:v2:f32:reg16x16r1x4w4");
    assertEquals(bmmKey(true, 17), "bmm:v2:f32:reg16x16r1x4w4v4");
    // The keys of the 3 buckets do not collide with each other
    const keys = [4, 318, 4096].map((rows) => linearKey("f32", true, "f32", rows));
    assertEquals(new Set(keys).size, keys.length, keys.join(", "));
  });

  it("calls without a row count and M >= 513 keep the existing keys and generated code byte for byte", () => {
    // The path fused attention and the snapshots go through (no row count)
    assertEquals(linearKey("f32", true), "linear:v2:f32:reg128x128r8x8w16v4");
    assertEquals(matmulKey(true), "matmul:v2:f32:reg128x128r8x8w16v4");
    assertEquals(bmmKey(false), "bmm:v2:f32:reg128x128r8x8w16");
    for (const rows of [513, 1024, 4096]) {
      assertEquals(linearKey("i8", false, "f32", rows), linearKey("i8", false));
      assertEquals(linearWgsl("f16", true, "f16", rows), linearWgsl("f16", true, "f16"));
      assertEquals(matmulWgsl(true, rows), matmulWgsl(true));
      assertEquals(bmmWgsl(false, rows), bmmWgsl(false));
    }
  });

  it("the workgroup shape, shared tiles and output tile origin of the generated code match the key's geometry", () => {
    for (const rows of ROW_SAMPLES) {
      const geometry = gemmGeometryForRows(rows);
      const tileM = gemmTileM(geometry);
      const tileN = gemmTileN(geometry);
      const quads = gemmQuadsPerThread(geometry);
      const variants: readonly (readonly [string, string])[] = [
        [`linear M=${rows}`, linearWgsl("f32", true, "f32", rows)],
        [`matmul M=${rows}`, matmulWgsl(true, rows)],
        [`bmm M=${rows}`, bmmWgsl(true, rows)],
      ];
      for (const [where, wgsl] of variants) {
        assertEquals(
          wgsl.includes(`@compute @workgroup_size(${geometry.wgX}, ${geometry.wgY})`),
          true,
          `${where}: workgroup shape`,
        );
        assertEquals(
          wgsl.includes(`var<workgroup> sa: array<f32, ${tileM * GEMM_TILE_K}>;`),
          true,
          `${where}: shared A tile`,
        );
        assertEquals(
          wgsl.includes(`var<workgroup> sb: array<vec4<f32>, ${GEMM_TILE_K * (tileN / 4)}>;`),
          true,
          `${where}: shared B tile`,
        );
        assertEquals(
          wgsl.includes(`let orow0 = wid.y * ${tileM}u + lid.y * ${geometry.regM}u;`),
          true,
          `${where}: row origin of the output tile`,
        );
        // acc is exactly regM × (regN/4) variables (one more / fewer is a mistake in the static
        // unrolling)
        assertEquals(
          wgsl.includes(`var acc${geometry.regM - 1}_${quads - 1} = vec4<f32>(`),
          true,
          `${where}: bottom-right acc`,
        );
        assertEquals(
          wgsl.includes(`var acc${geometry.regM}_0 =`),
          false,
          `${where}: acc rows overflow`,
        );
        assertEquals(wgsl.includes(`var acc0_${quads} =`), false, `${where}: acc columns overflow`);
      }
    }
  });

  it("changing the geometry does not move the K reduction step or the text of the multiply-add (the numeric contract of ADR 0022)", () => {
    for (const rows of ROW_SAMPLES) {
      const wgsl = linearWgsl("f32", true, "f32", rows);
      const where = `linear M=${rows}`;
      // Outer t ascending / inner kk ascending / K tile 16 — the very order of additions per output
      // element
      assertEquals(wgsl.includes(`let tiles = (dims.k + 15u) / ${GEMM_TILE_K}u;`), true, where);
      assertEquals(wgsl.includes("for (var t = 0u; t < tiles; t = t + 1u) {"), true, where);
      assertEquals(
        wgsl.includes(`for (var kk = 0u; kk < ${GEMM_TILE_K}u; kk = kk + 1u) {`),
        true,
        where,
      );
      // MUST: the multiply-add stays the text `acc = acc + a * b` (no explicit fma)
      assertEquals(wgsl.includes("acc0_0 = acc0_0 + "), true, where);
      assertEquals(wgsl.includes("fma("), false, `${where}: explicit fma`);
    }
  });

  it("the same row count always produces byte-identical WGSL (determinism)", () => {
    for (const rows of ROW_SAMPLES) {
      assertEquals(linearWgsl("f32", true, "f32", rows), linearWgsl("f32", true, "f32", rows));
      assertEquals(matmulWgsl(false, rows), matmulWgsl(false, rows));
    }
    // Different buckets give different generated code (keys split while the content is the same =
    // the geometry is not flowing), and within the same bucket a different row count gives the same
    // generated code (the key's cardinality does not explode with the value of M)
    assertNotEquals(matmulWgsl(true, 64), matmulWgsl(true, 65));
    assertNotEquals(matmulWgsl(true, 512), matmulWgsl(true, 513));
    assertEquals(matmulWgsl(true, 4), matmulWgsl(true, 32));
    assertEquals(matmulWgsl(true, 65), matmulWgsl(true, 512));
  });
});

// Explicit geometry (`GemmSpec.geometry` of src/kernels/gemm.ts — the override point for
// measurement in tools/geometry-sweep). It protects 2 points: the generated code without it does
// not move by a single byte (the production path does not pass it), and when passed, keys and WGSL
// name **the same geometry** (if only one of them misses it, the WGSL of another geometry in the
// cache does not mesh with the dispatch count and tiles go missing). The byte-identity half is
// covered by the default-profile describe above; this block keeps the override half (a non-default
// geometry wins over the per-op resolution in both key and WGSL).
describe("an explicit geometry overrides the per-op resolution", () => {
  it("an explicit geometry wins over the per-op resolution, and keys and WGSL name the same geometry", () => {
    const explicit: GemmGeometry = { regM: 4, regN: 4, wgX: 8, wgY: 8 };
    const workgroup = `@compute @workgroup_size(${explicit.wgX}, ${explicit.wgY})`;
    // linear: overrides the M=4096 bucket (default 128×128)
    assertEquals(
      linearKey("f16", true, "f32", 4096, undefined, explicit),
      "linear:v2:f32:reg32x32r4x4w8v4:wf16",
    );
    assertEquals(
      linearWgsl("f16", true, "f32", 4096, undefined, explicit).includes(workgroup),
      true,
    );
    // Fused attention: overrides the fixed default
    assertEquals(
      attentionQkKey(true, "f32", "f32", false, false, false, explicit),
      "attention_qk:v1:f32:reg32x32r4x4w8v4",
    );
    assertEquals(
      attentionPvWgsl(true, "f32", "f32", false, false, explicit).includes(workgroup),
      true,
    );
    // conv2d: overrides the m-tile resolution (the spelling where the edges and the workgroup shape
    // determine the geometry)
    assertEquals(
      conv2dIgemmKey("f16", true, 64, explicit),
      "conv2d:v3:f32:igemm32x32v4:wg8x8:wf16",
    );
    assertEquals(conv2dIgemmWgsl("f16", true, 64, explicit).includes(workgroup), true);
    // The check is the same when explicit (a tile shape with holes fails at generation)
    assertThrows(
      () => linearWgsl("f16", true, "f32", 4096, undefined, { regM: 3, regN: 4, wgX: 8, wgY: 8 }),
      CodegenError,
    );
  });
});

describe("gemmRowsGeometry", () => {
  const rules = (...entries: (readonly [number, GemmGeometry])[]): GeometryProfile =>
    profileOf(
      "rules",
      { vendor: "test" },
      {
        gemmRows: entries.map(([maxRows, geometry]) => ({ maxRows, geometry })),
      },
    );
  const small = gemmGeometryForRows(1);

  it("exactly maxRows takes that rule, 1 past takes the next rule (the final Infinity matches up to the top of u32)", () => {
    const profile = rules([10, small], [100, M64N32], [Number.POSITIVE_INFINITY, M64N64]);
    const expected: readonly (readonly [number, GemmGeometry])[] = [
      [0, small],
      [10, small],
      [11, M64N32],
      [100, M64N32],
      [101, M64N64],
      [0xffff_ffff, M64N64],
    ];
    for (const [rows, geometry] of expected) {
      assertEquals(gemmRowsGeometry(profile, rows), geometry, `M=${rows}`);
    }
  });

  it("rules not strictly ascending, a last rule that is not Infinity, and an empty table fail loudly both at selection and at lookup", () => {
    const adapter = { vendor: "test", architecture: "" };
    const broken: readonly (readonly [string, GeometryProfile, string])[] = [
      [
        "descending",
        rules([100, small], [10, M64N32], [Number.POSITIVE_INFINITY, M64N64]),
        "strictly increasing",
      ],
      [
        "equal values",
        rules([10, small], [10, M64N32], [Number.POSITIVE_INFINITY, M64N64]),
        "strictly increasing",
      ],
      ["finite last rule", rules([10, small], [100, M64N32]), "Infinity"],
      ["empty", rules(), "empty"],
    ];
    for (const [name, profile, message] of broken) {
      assertThrows(() => gemmRowsGeometry(profile, 1), CodegenError, message, name);
      assertThrows(() => selectGeometryProfile(adapter, [profile]), CodegenError, message, name);
    }
  });

  it("the row count fails loudly on anything but a non-negative u32 integer (the same check as gemmGeometryForRows)", () => {
    for (const rows of [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      assertThrows(() => gemmRowsGeometry(DEFAULT_GEOMETRY_PROFILE, rows), CodegenError);
    }
  });
});

describe("conv2dProfileGeometry and the geometry check", () => {
  it("m-tile classes 64 / 32 look up rows64 / rows32, and anything else fails loudly", () => {
    const profile = profileOf(
      "conv",
      { vendor: "test" },
      {
        conv2d: { rows64: M64N64, rows32: M64N32 },
      },
    );
    assertStrictEquals(conv2dProfileGeometry(profile, 64), M64N64);
    assertStrictEquals(conv2dProfileGeometry(profile, 32), M64N32);
    assertThrows(() => conv2dProfileGeometry(profile, 16), CodegenError, "no class matches");
  });

  it("a table holding a geometry that breaks exact division fails before reaching that op (at selection time)", () => {
    const adapter = { vendor: "test", architecture: "" };
    const hole: GemmGeometry = { regM: 3, regN: 4, wgX: 8, wgY: 8 };
    const broken: readonly (readonly [string, GeometryProfile])[] = [
      [
        "attention.qk",
        profileOf("qk", { vendor: "test" }, { attention: { qk: hole, pv: M64N64 } }),
      ],
      [
        "conv2d.rows32",
        profileOf("conv", { vendor: "test" }, { conv2d: { rows64: M64N64, rows32: hole } }),
      ],
      [
        "gemmRows",
        profileOf(
          "rows",
          { vendor: "test" },
          {
            gemmRows: [{ maxRows: Number.POSITIVE_INFINITY, geometry: hole }],
          },
        ),
      ],
      [
        "i8a8.attentionPv",
        profileOf(
          "i8a8",
          { vendor: "test" },
          {
            i8a8: {
              ...DEFAULT_GEOMETRY_PROFILE.i8a8,
              attentionPv: { regM: 8, regN: 8, wgX: 16, wgY: 8, tileK: 6 },
            },
          },
        ),
      ],
    ];
    for (const [name, profile] of broken) {
      assertThrows(() => selectGeometryProfile(adapter, [profile]), CodegenError, name, name);
    }
  });
});

/** A sentinel showing requestDevice was reached (no real device is created). */
class DeviceRequested extends Error {
  override readonly name = "DeviceRequested";
}

/**
 * Runs `body` with `navigator.gpu` swapped for a fake that counts requestDevice calls and fails
 * with the sentinel (no GPU needed; no device is created even on a real machine). The swap is
 * always removed in finally. The adapter's limits are all 1 << 20, and only the fields of `limits`
 * are overridden (for the limit checks).
 */
const withCountingGpu = async (
  body: () => Promise<void>,
  limits: Partial<WorkgroupLimits> = {},
): Promise<number> => {
  let deviceRequests = 0;
  const adapter = {
    limits: { ...Object.fromEntries(REQUIRED_LIMIT_KEYS.map((key) => [key, 1 << 20])), ...limits },
    features: new Set<string>(),
    requestDevice: (): Promise<never> => {
      deviceRequests += 1;
      return Promise.reject(new DeviceRequested("requestDevice was reached"));
    },
  };
  Object.defineProperty(navigator, "gpu", {
    value: { requestAdapter: () => Promise.resolve(adapter) },
    configurable: true,
  });
  try {
    await body();
  } finally {
    Reflect.deleteProperty(navigator, "gpu");
  }
  return deviceRequests;
};

describe("the geometry profile injection point of acquireGpu", () => {
  it("a valid table (including an injection-only table without match and a table with a description) passes the check and proceeds to requestDevice (the control showing the swap is on the path)", async () => {
    for (const profile of [APPLE_METAL3, OPT_IN, APPLE_M2]) {
      const requests = await withCountingGpu(async () => {
        await assertRejects(
          () => acquireGpu({ geometryProfile: profile }),
          DeviceRequested,
          undefined,
          profile.id,
        );
      });
      assertEquals(requests, 1, profile.id);
    }
  });

  it("a frozen table from the public list can be injected as is, and the copy it keeps (structuredClone) does not carry the freeze over", async () => {
    for (const profile of BUILTIN_GEOMETRY_PROFILES) {
      const requests = await withCountingGpu(async () => {
        await assertRejects(
          () => acquireGpu({ geometryProfile: profile }),
          DeviceRequested,
          undefined,
          profile.id,
        );
      });
      assertEquals(requests, 1, profile.id);
      // The copy on the injection path is structuredClone (acquire.ts) — the freeze is not copied
      assertEquals(unfrozenPath(structuredClone(profile), profile.id), profile.id);
    }
  });

  it("a broken table (empty id, a geometry breaking exact division) fails loudly before the device is created", async () => {
    const hole: GemmGeometry = { regM: 3, regN: 4, wgX: 8, wgY: 8 };
    const broken: readonly (readonly [string, GeometryProfile, string])[] = [
      ["empty id", profileOf("", { vendor: "test" }), "id is empty"],
      [
        "match of description only",
        profileOf("desc-only", { description: "Apple M2" }),
        "together with both vendor and architecture",
      ],
      [
        "geometry of gemmRows",
        profileOf(
          "rows",
          { vendor: "test" },
          {
            gemmRows: [{ maxRows: Number.POSITIVE_INFINITY, geometry: hole }],
          },
        ),
        "gemmRows",
      ],
      [
        "geometry of attention.pv",
        profileOf("pv", { vendor: "test" }, { attention: { qk: M64N64, pv: hole } }),
        "attention.pv",
      ],
      [
        "geometry of i8a8.linear",
        profileOf(
          "i8a8",
          { vendor: "test" },
          {
            i8a8: {
              ...DEFAULT_GEOMETRY_PROFILE.i8a8,
              linear: { regM: 8, regN: 8, wgX: 16, wgY: 8, tileK: 6 },
            },
          },
        ),
        "i8a8.linear",
      ],
    ];
    for (const [name, profile, message] of broken) {
      const requests = await withCountingGpu(async () => {
        await assertRejects(
          () => acquireGpu({ geometryProfile: profile }),
          GpuFeatureError,
          message,
          name,
        );
      });
      assertEquals(requests, 0, `${name}: a broken table reached requestDevice`);
    }
  });
});

/**
 * The information the fake adapter returns (the 4 fields apple-metal-3 matches — the result of
 * automatic selection is distinguishable from the default).
 */
const FAKE_INFO: GPUAdapterInfo = platformObject<GPUAdapterInfo>({
  vendor: "apple",
  architecture: "metal-3",
  device: "",
  description: "Apple M2",
  subgroupMinSize: 0,
  subgroupMaxSize: 0,
  isFallbackAdapter: false,
});

/**
 * Runs `body` with `navigator.gpu` swapped for a fake that counts requestDevice calls and returns a
 * fake device meeting the limits (no GPU needed — the GpuContext is built for real). Omitting
 * `info` gives an adapter without `adapter.info` (old Chromium). The swap is always removed in
 * finally.
 */
const withFakeAdapter = async (
  info: GPUAdapterInfo | undefined,
  body: () => Promise<void>,
): Promise<number> => {
  let deviceRequests = 0;
  const limits = Object.fromEntries(REQUIRED_LIMIT_KEYS.map((key) => [key, 1 << 20]));
  const adapter = {
    ...(info === undefined ? {} : { info }),
    limits,
    features: new Set<string>(),
    requestDevice: (): Promise<GPUDevice> => {
      deviceRequests += 1;
      return Promise.resolve(Object.assign(fakeDevice(), { limits }));
    },
  };
  Object.defineProperty(navigator, "gpu", {
    value: { requestAdapter: () => Promise.resolve(adapter) },
    configurable: true,
  });
  try {
    await body();
  } finally {
    Reflect.deleteProperty(navigator, "gpu");
  }
  return deviceRequests;
};

/**
 * A gate for passing values that fall outside the callback's return type (JS callers, async
 * functions) without going through type checking.
 */
const untypedCallback = (
  returned: unknown,
): ((adapterInfo: GPUAdapterInfo) => GeometryProfile | undefined) =>
  // Test-only boundary: reproduces a return the type rejects (a Promise and the like) as a value
  // coming from outside the type
  (() => returned) as unknown as (adapterInfo: GPUAdapterInfo) => GeometryProfile | undefined;

describe("the geometry profile injection point of acquireGpu (callback form — ADR 0117 decision 6)", () => {
  it("is called exactly once with the adapter's information (the same value as GpuContext.adapterInfo), and a returned table is injected as a copy", async () => {
    const seen: GPUAdapterInfo[] = [];
    const requests = await withFakeAdapter(FAKE_INFO, async () => {
      const gpu = await acquireGpu({
        geometryProfile: (adapterInfo) => {
          seen.push(adapterInfo);
          return OPT_IN;
        },
      });
      try {
        assertEquals(seen.length, 1);
        assertStrictEquals(seen[0], gpu.adapterInfo);
        assertStrictEquals(seen[0], FAKE_INFO);
        const injected = gpu[RUNTIME_INTERNAL].geometryProfile;
        assertEquals(injected, OPT_IN);
        assert(injected !== OPT_IN, "the injected table was not copied");
      } finally {
        gpu.destroy();
      }
    });
    assertEquals(requests, 1);
  });

  it("returning undefined injects nothing (Session construction selects an embedded table from the adapter — the same as not specifying one)", async () => {
    await withFakeAdapter(FAKE_INFO, async () => {
      let calls = 0;
      const gpu = await acquireGpu({
        geometryProfile: () => {
          calls += 1;
          return undefined;
        },
      });
      try {
        assertEquals(calls, 1);
        assertEquals(gpu[RUNTIME_INTERNAL].geometryProfile, undefined);
        // Control for automatic selection: an embedded table matches this adapter (not the default)
        assertEquals(selectGeometryProfile(gpu.adapterInfo).id, APPLE_METAL_3.id);
      } finally {
        gpu.destroy();
      }
    });
  });

  it("an adapter without adapter.info calls it with the information normalized to empty values", async () => {
    const seen: GPUAdapterInfo[] = [];
    await withFakeAdapter(undefined, async () => {
      const gpu = await acquireGpu({
        geometryProfile: (adapterInfo) => {
          seen.push(adapterInfo);
          return undefined;
        },
      });
      try {
        assertStrictEquals(seen[0], gpu.adapterInfo);
        assertEquals(
          [seen[0].vendor, seen[0].architecture, seen[0].device, seen[0].description],
          ["", "", "", ""],
        );
      } finally {
        gpu.destroy();
      }
    });
  });

  it("returning a broken table gives GpuFeatureError before the device is created (the same check as a table passed directly)", async () => {
    const broken = profileOf("", { vendor: "test" });
    const requests = await withFakeAdapter(FAKE_INFO, async () => {
      await assertRejects(
        () => acquireGpu({ geometryProfile: () => broken }),
        GpuFeatureError,
        "geometryProfile (callback return value): ",
      );
    });
    assertEquals(requests, 0, "a broken table reached requestDevice");
  });

  it("an exception thrown by the callback propagates as is, and no device is created", async () => {
    class AppError extends Error {}
    const thrown = new AppError("the saved table cannot be read");
    const requests = await withFakeAdapter(FAKE_INFO, async () => {
      const caught = await assertRejects(() =>
        acquireGpu({
          geometryProfile: () => {
            throw thrown;
          },
        }),
      );
      assertStrictEquals(caught, thrown);
    });
    assertEquals(requests, 0, "requestDevice was reached although the callback threw");
  });

  it("returning a Promise (thenable) is rejected with GpuFeatureError without awaiting it, and no device is created", async () => {
    let awaited = false;
    const thenable = {
      // oxlint-disable-next-line unicorn/no-thenable -- a fake thenable is the input under test
      then: (resolve: (value: GeometryProfile) => void): void => {
        awaited = true;
        resolve(OPT_IN);
      },
    };
    for (const returned of [Promise.resolve(OPT_IN), thenable]) {
      const requests = await withFakeAdapter(FAKE_INFO, async () => {
        await assertRejects(
          () => acquireGpu({ geometryProfile: untypedCallback(returned) }),
          GpuFeatureError,
          "the callback returned a Promise",
        );
      });
      assertEquals(requests, 0, "requestDevice was reached although a Promise was returned");
    }
    assertEquals(awaited, false, "the thenable was awaited");
  });

  it("an async function that rejects is refused with GpuFeatureError, and its rejection does not become an unhandled rejection", async () => {
    const unhandled: unknown[] = [];
    // Node reports unhandled rejections through process events (no globalThis event target).
    const onUnhandled = (reason: unknown): void => {
      unhandled.push(reason);
    };
    // The async function itself is the callback (a Promise already rejected at call time is born —
    // the shape of a JS caller).
    // Test-only boundary: passes an async function the type rejects as a value coming from outside
    // the type
    // deno-lint-ignore require-await -- the rejection without await is what is under test
    const asyncCallback = (async () => {
      throw new Error("x");
    }) as unknown as (adapterInfo: GPUAdapterInfo) => GeometryProfile | undefined;
    process.on("unhandledRejection", onUnhandled);
    try {
      const requests = await withFakeAdapter(FAKE_INFO, async () => {
        await assertRejects(
          () => acquireGpu({ geometryProfile: asyncCallback }),
          GpuFeatureError,
          "the callback returned a Promise",
        );
      });
      assertEquals(requests, 0, "requestDevice was reached although a Promise was returned");
      // The unhandled-rejection decision runs at the task boundary after microtasks are drained —
      // wait 1 task before looking
      await new Promise((resolve) => setTimeout(resolve, 0));
      assertEquals(
        unhandled,
        [],
        "the rejection of the async callback became an unhandled rejection",
      );
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  it("an async function and an already rejected Promise from another realm (iframe, node:vm) are also refused with GpuFeatureError, and their rejection does not become an unhandled rejection", async () => {
    const unhandled: unknown[] = [];
    // Node reports unhandled rejections through process events (no globalThis event target).
    const onUnhandled = (reason: unknown): void => {
      unhandled.push(reason);
    };
    const realm = createContext({});
    // A Promise of another realm is not an instance of this realm's Promise (instanceof cannot tell
    // them apart)
    const foreignAsync: unknown = runInContext(
      "(async () => { throw new Error('foreign async'); })",
      realm,
    );
    const foreignRejected: unknown = runInContext(
      "Promise.reject(new Error('foreign rejected'))",
      realm,
    );
    assert(typeof foreignAsync === "function");
    process.on("unhandledRejection", onUnhandled);
    try {
      for (const [name, callback] of [
        // Test-only boundary: passes an async function the type rejects as a value coming from
        // outside the type
        [
          "async function",
          foreignAsync as (adapterInfo: GPUAdapterInfo) => GeometryProfile | undefined,
        ],
        ["already rejected Promise", untypedCallback(foreignRejected)],
      ] as const) {
        const requests = await withFakeAdapter(FAKE_INFO, async () => {
          await assertRejects(
            () => acquireGpu({ geometryProfile: callback }),
            GpuFeatureError,
            "the callback returned a Promise",
            name,
          );
        });
        assertEquals(
          requests,
          0,
          `${name}: requestDevice was reached although a Promise was returned`,
        );
      }
      // The unhandled-rejection decision runs at the task boundary after microtasks are drained —
      // wait 1 task before looking
      await new Promise((resolve) => setTimeout(resolve, 0));
      assertEquals(
        unhandled,
        [],
        "the rejection of a Promise from another realm became an unhandled rejection",
      );
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  it("a hand-made thenable posing as a Promise (a then getter, a faked toStringTag) is also refused without calling then", async () => {
    let called = 0;
    const then = (): void => {
      called += 1;
    };
    const thenables: readonly (readonly [string, unknown])[] = [
      [
        "then getter",
        {
          // oxlint-disable-next-line unicorn/no-thenable -- a fake thenable is the input under test
          get then() {
            return then;
          },
        },
      ],
      // oxlint-disable-next-line unicorn/no-thenable -- a fake thenable is the input under test
      ["toStringTag of Promise", { [Symbol.toStringTag]: "Promise", then }],
    ];
    for (const [name, thenable] of thenables) {
      const requests = await withFakeAdapter(FAKE_INFO, async () => {
        await assertRejects(
          () => acquireGpu({ geometryProfile: untypedCallback(thenable) }),
          GpuFeatureError,
          "the callback returned a Promise",
          name,
        );
      });
      assertEquals(
        requests,
        0,
        `${name}: requestDevice was reached although a Promise was returned`,
      );
    }
    assertEquals(called, 0, "then of the hand-made thenable was called");
  });

  it("the empty-value information passed for an adapter without adapter.info is frozen (the callback cannot rewrite the shared value)", async () => {
    const seen: GPUAdapterInfo[] = [];
    await withFakeAdapter(undefined, async () => {
      const gpu = await acquireGpu({
        geometryProfile: (adapterInfo) => {
          seen.push(adapterInfo);
          return undefined;
        },
      });
      gpu.destroy();
    });
    assert(Object.isFrozen(seen[0]), "the empty-value adapter information is not frozen");
  });

  it("a table passed directly still goes through the check before the adapter is requested (adding the callback form did not move the order)", async () => {
    let adapterRequests = 0;
    Object.defineProperty(navigator, "gpu", {
      value: {
        requestAdapter: () => {
          adapterRequests += 1;
          return Promise.resolve(null);
        },
      },
      configurable: true,
    });
    try {
      await assertRejects(
        () => acquireGpu({ geometryProfile: profileOf("", { vendor: "test" }) }),
        GpuFeatureError,
        "geometryProfile: ",
      );
    } finally {
      Reflect.deleteProperty(navigator, "gpu");
    }
    assertEquals(adapterRequests, 0);
  });
});

describe("assertGeometryProfile is a type predicate narrowing unknown to a table", () => {
  it("an untyped value (unknown) that passes the check can be read as a GeometryProfile", () => {
    const value: unknown = structuredClone(APPLE_M2);
    assertGeometryProfile(value);
    // If the type predicate's narrowing did not work, type checking would fail here
    const profile: GeometryProfile = value;
    assertEquals(profile, APPLE_M2);
  });

  it("a missing field or a wrong type names the field with CodegenError instead of TypeError", () => {
    const { gemmRows: _gemmRows, ...withoutRows } = OPT_IN;
    const cases: readonly (readonly [string, unknown, string])[] = [
      ["null", null, "table is not an object"],
      ["id is a number", { ...OPT_IN, id: 5 }, "id is not a string"],
      [
        "match.vendor is a number",
        { ...OPT_IN, match: { vendor: 1 } },
        "match.vendor is not a string",
      ],
      ["gemmRows missing", withoutRows, "gemmRows is not an array"],
      [
        "a gemmRows rule without geometry",
        { ...OPT_IN, gemmRows: [{ maxRows: Number.POSITIVE_INFINITY }] },
        "gemmRows[0].geometry is not an object",
      ],
      [
        "maxRows is a string",
        { ...OPT_IN, gemmRows: [{ maxRows: "64", geometry: M64N64 }] },
        "maxRows of gemmRows[0] is not a number",
      ],
      ["attention missing", { ...OPT_IN, attention: undefined }, "attention is not an object"],
      [
        "a geometry field is a string",
        { ...OPT_IN, attention: { qk: { ...M64N64, wgX: "16" }, pv: M64N64 } },
        "attention.qk: geometry wgX must be a positive integer (16)",
      ],
      [
        "tileK of i8a8 missing",
        { ...OPT_IN, i8a8: { ...OPT_IN.i8a8, linear: M64N64 } },
        "i8a8.linear: geometry tileK must be a positive integer (undefined)",
      ],
      [
        "a provenance field missing",
        { ...OPT_IN, provenance: {} },
        "provenance.sweep is not a string (missing)",
      ],
      [
        "provenance.userAgent is a string",
        {
          ...OPT_IN,
          provenance: {
            sweep: "s",
            sha256: "h",
            date: "d",
            candidateSet: "quick",
            kernels: "k",
            caseSet: "c",
            adapter: { vendor: "", architecture: "", device: "", description: "" },
            userAgent: "Deno/2",
          },
        },
        "provenance.userAgent is not an array",
      ],
    ];
    for (const [name, value, message] of cases) {
      assertThrows(() => assertGeometryProfile(value), CodegenError, message, name);
    }
  });

  it("holes of a sparse array (head, middle and tail of gemmRows and provenance.userAgent) name the index with CodegenError instead of TypeError", () => {
    /**
     * A sparse array of `length` with only the indices of `entries` filled (the shape a JS caller
     * builds with `new Array(n)`).
     */
    const sparse = (length: number, entries: Readonly<Record<number, unknown>>): unknown[] => {
      const array: unknown[] = new Array(length);
      for (const [index, value] of Object.entries(entries)) array[Number(index)] = value;
      return array;
    };
    const [r64, r512, rest] = DEFAULT_GEOMETRY_PROFILE.gemmRows;
    const provenance = {
      sweep: "s",
      sha256: "h",
      date: "d",
      candidateSet: "quick",
      kernels: "k",
      caseSet: "c",
      adapter: { vendor: "", architecture: "", device: "", description: "" },
    };
    const cases: readonly (readonly [string, unknown, string])[] = [
      [
        "gemmRows is new Array(1)",
        { ...OPT_IN, gemmRows: sparse(1, {}) },
        "gemmRows[0] is missing",
      ],
      [
        "hole at the head of gemmRows",
        { ...OPT_IN, gemmRows: sparse(4, { 1: r64, 2: r512, 3: rest }) },
        "gemmRows[0] is missing",
      ],
      [
        "hole in the middle of gemmRows",
        { ...OPT_IN, gemmRows: sparse(4, { 0: r64, 1: r512, 3: rest }) },
        "gemmRows[2] is missing",
      ],
      [
        "hole at the tail of gemmRows",
        { ...OPT_IN, gemmRows: sparse(4, { 0: r64, 1: r512, 2: rest }) },
        "gemmRows[3] is missing",
      ],
      [
        "userAgent is new Array(1)",
        { ...OPT_IN, provenance: { ...provenance, userAgent: sparse(1, {}) } },
        "provenance.userAgent[0] is missing",
      ],
      [
        "hole in the middle of userAgent",
        { ...OPT_IN, provenance: { ...provenance, userAgent: sparse(3, { 0: "a", 2: "b" }) } },
        "provenance.userAgent[1] is missing",
      ],
    ];
    for (const [name, value, message] of cases) {
      assertThrows(() => assertGeometryProfile(value), CodegenError, message, name);
    }
  });

  it("a table with a sparse array coming into acquireGpu from outside the type gives GpuFeatureError before the device is created (not a TypeError)", async () => {
    const holed: unknown[] = new Array(3);
    holed[0] = DEFAULT_GEOMETRY_PROFILE.gemmRows[0];
    holed[2] = DEFAULT_GEOMETRY_PROFILE.gemmRows[2];
    const requests = await withCountingGpu(async () => {
      await assertRejects(
        () => acquireGpu({ geometryProfile: untypedCallback({ ...OPT_IN, gemmRows: holed }) }),
        GpuFeatureError,
        "geometryProfile (callback return value): Geometry profile 'opt-in': gemmRows[1] is missing",
      );
    });
    assertEquals(requests, 0, "a table with a sparse array reached requestDevice");
  });

  it("an incomplete table coming into acquireGpu from outside the type gives GpuFeatureError before the device is created (not a TypeError)", async () => {
    const requests = await withCountingGpu(async () => {
      await assertRejects(
        () => acquireGpu({ geometryProfile: untypedCallback({ id: "broken" }) }),
        GpuFeatureError,
        "geometryProfile (callback return value): Geometry profile 'broken': gemmRows is not an array",
      );
    });
    assertEquals(requests, 0, "an incomplete table reached requestDevice");
  });
});

/**
 * WebGPU's default limits (every device offers at least these — the default values of the spec's
 * limits).
 */
const WEBGPU_DEFAULT_LIMITS: WorkgroupLimits = {
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupStorageSize: 16_384,
};

/** 512 threads (wgX 32 × wgY 16) — meets exact division, with shared memory at exactly 16,384 B. */
const THREADS_512: GemmGeometry = { regM: 8, regN: 4, wgX: 32, wgY: 16 };
/** 20,480 B of shared memory (M64N256 — 64 · (64 + 256)) — 256 threads. */
const STORAGE_20K: GemmGeometry = { regM: 8, regN: 8, wgX: 32, wgY: 8 };
/** 32,768 B of i8a8 shared memory (tileK 128 · (128 + 128)) — 256 threads. */
const I8A8_STORAGE_32K: I8a8Geometry = { regM: 8, regN: 8, wgX: 16, wgY: 16, tileK: 128 };
/**
 * A tiny i8a8 geometry where the per-variable round-up to 16 bytes matters (sa goes from 4 B to 16
 * B).
 */
const I8A8_TINY: I8a8Geometry = { regM: 1, regN: 4, wgX: 1, wgY: 1, tileK: 4 };

/**
 * Bytes per WGSL element type (only the types the shared tiles use — an unknown type fails instead
 * of being left uncounted).
 */
const WGSL_TYPE_BYTES: Readonly<Record<string, number>> = {
  f32: 4,
  f16: 2,
  u32: 4,
  "vec4<f32>": 16,
  "vec4<f16>": 8,
};

/**
 * Counts, from the `var<workgroup>` declarations of the generated WGSL, the quantity WebGPU
 * compares against the limit (the sum of `roundUp(16, SizeOf(T))` per variable). Fails if even one
 * declaration cannot be read (an uncounted declaration must not pass as a match).
 */
const declaredWorkgroupBytes = (wgsl: string): number => {
  const declarations = [
    ...wgsl.matchAll(/var<workgroup>\s+\w+\s*:\s*array<\s*(.+?)\s*,\s*(\d+)\s*>\s*;/g),
  ];
  assertEquals(
    declarations.length,
    wgsl.match(/var<workgroup>/g)?.length ?? 0,
    "there is a var<workgroup> declaration that cannot be read",
  );
  assert(declarations.length > 0, "no var<workgroup> declaration");
  return declarations.reduce((total, [, type, count]) => {
    const bytes = WGSL_TYPE_BYTES[type];
    assert(bytes !== undefined, `unknown element type ${type}`);
    return total + Math.ceil((bytes * Number(count)) / 16) * 16;
  }, 0);
};

/**
 * The f32 / f16 GEMM geometries to compare (the default, every field of the embedded tables,
 * brought-in geometries exceeding the limits).
 */
const gemmGeometriesOf = (profile: GeometryProfile): GemmGeometry[] => [
  ...profile.gemmRows.map((rule) => rule.geometry),
  profile.attention.qk,
  profile.attention.pv,
  profile.conv2d.rows64,
  profile.conv2d.rows32,
];
const SAMPLE_PROFILES = [DEFAULT_GEOMETRY_PROFILE, ...BUILTIN_GEOMETRY_PROFILES];
const SAMPLE_GEMM = [...SAMPLE_PROFILES.flatMap(gemmGeometriesOf), THREADS_512, STORAGE_20K];
const SAMPLE_I8A8 = [
  ...SAMPLE_PROFILES.flatMap((profile) => Object.values(profile.i8a8)),
  I8A8_STORAGE_32K,
  I8A8_TINY,
];

const label = (geometry: GemmGeometry | I8a8Geometry): string => JSON.stringify(geometry);

describe("the derivation of shared memory (a pure function) matches the total of codegen's var<workgroup> declarations", () => {
  it("f32 / f16 GEMM skeletons (linear, fused attention ①QK / ③PV, conv2d, matmul)", () => {
    for (const geometry of SAMPLE_GEMM) {
      const cases: readonly (readonly [string, GemmCompute, string])[] = [
        ["matmul", "f32", matmulWgsl(true, 128, geometry)],
        ["linear f32", "f32", linearWgsl("f32", true, "f32", 128, undefined, geometry)],
        ["linear f16 compute", "f16", linearWgsl("f32", true, "f16", 128, undefined, geometry)],
        [
          "attention_qk f32",
          "f32",
          attentionQkWgsl(true, "f32", "f32", false, false, false, geometry),
        ],
        [
          "attention_qk f16 compute",
          "f16",
          attentionQkWgsl(true, "f16", "f32", false, false, false, geometry),
        ],
        ["attention_pv f32", "f32", attentionPvWgsl(true, "f32", "f32", false, false, geometry)],
        [
          "attention_pv f16 compute",
          "f16",
          attentionPvWgsl(true, "f16", "f32", false, false, geometry),
        ],
        ["conv2d", "f32", conv2dIgemmWgsl("f32", true, 64, geometry)],
      ];
      for (const [name, compute, wgsl] of cases) {
        assertEquals(
          gemmWorkgroupStorageBytes(geometry, compute),
          declaredWorkgroupBytes(wgsl),
          `${name} ${label(geometry)}`,
        );
      }
    }
  });

  it("the f32 skeleton forms and the conv2d counter decode (perf-ledger K-88 — linear, matmul, bmm, conv2d v4): the shared tiles are declared as without them", () => {
    for (const base of SAMPLE_GEMM) {
      for (const form of ["tb-un", "tas-tb-un"] as const) {
        const geometry: GemmGeometry = { ...base, form };
        assertEquals(
          gemmWorkgroupStorageBytes(geometry, "f32"),
          gemmWorkgroupStorageBytes(base, "f32"),
        );
        const cases: readonly (readonly [string, string])[] = [
          ["linear f32", linearWgsl("f32", true, "f32", 128, undefined, geometry)],
          ["linear i4", linearWgsl("i4", true, "f32", 128, 32, geometry)],
          ["matmul", matmulWgsl(true, 128, geometry)],
          ["bmm", bmmWgsl(true, 128, undefined, geometry)],
          ["conv2d", conv2dIgemmWgsl("f32", true, 64, geometry)],
          ["conv2d decode", conv2dIgemmWgsl("i8", true, 64, { ...geometry, decode: "counter" })],
        ];
        for (const [name, wgsl] of cases) {
          assertEquals(
            gemmWorkgroupStorageBytes(geometry, "f32"),
            declaredWorkgroupBytes(wgsl),
            `${name} ${label(geometry)}`,
          );
        }
      }
      assertEquals(
        gemmWorkgroupStorageBytes(base, "f32"),
        declaredWorkgroupBytes(conv2dIgemmWgsl("f16", true, 64, { ...base, decode: "counter" })),
        `conv2d decode alone ${label(base)}`,
      );
    }
  });

  it("i8a8 (linear, fused attention ①QK / ③PV and conv3d — including the per-variable 16-byte round-up)", () => {
    for (const geometry of SAMPLE_I8A8) {
      for (const [name, wgsl] of [
        ["linear", linearI8a8Wgsl(true, true, geometry)],
        ["attention_qk", attentionQkI8a8Wgsl(true, true, "f32", geometry)],
        ["attention_pv", attentionPvI8a8Wgsl(true, true, "f32", geometry)],
        ["conv3d", conv3dI8a8Wgsl(true, true, geometry)],
      ] as const) {
        assertEquals(
          i8a8WorkgroupStorageBytes(geometry),
          declaredWorkgroupBytes(wgsl),
          `${name} ${label(geometry)}`,
        );
      }
    }
    // Control for the round-up: a value different from the naive sum of element count × 4 (4 + 16 =
    // 20)
    assertEquals(i8a8WorkgroupStorageBytes(I8A8_TINY), 32);
  });

  it("i8a8 with the vec4 skeleton (perf-ledger K-85 — both fill forms of linear, w4a8, ①QK / ③PV, conv3d): the shared tiles are declared as without it", () => {
    const skeletons: readonly I8a8Geometry[] = [
      { regM: 16, regN: 4, wgX: 16, wgY: 8, tileK: 32, skeleton: "vec4" },
      { regM: 12, regN: 4, wgX: 16, wgY: 8, tileK: 64, skeleton: "vec4" },
      { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 64, skeleton: "vec4" },
      { regM: 12, regN: 4, wgX: 16, wgY: 8, tileK: 32, skeleton: "vec4" },
      { ...I8A8_STORAGE_32K, skeleton: "vec4" },
    ];
    for (const geometry of skeletons) {
      const { skeleton: _skeleton, ...word } = geometry;
      assertEquals(i8a8WorkgroupStorageBytes(geometry), i8a8WorkgroupStorageBytes(word));
      for (const [name, wgsl] of [
        ["linear fill16", linearI8a8Wgsl(true, true, geometry, "i8", undefined, true)],
        ["linear word", linearI8a8Wgsl(false, false, geometry)],
        ["linear w4a8", linearI8a8Wgsl(true, true, geometry, "i4", 128)],
        ["attention_qk", attentionQkI8a8Wgsl(true, true, "f32", geometry)],
        ["attention_pv", attentionPvI8a8Wgsl(true, true, "f32", geometry)],
        ["conv3d", conv3dI8a8Wgsl(true, true, geometry)],
      ] as const) {
        assertEquals(
          i8a8WorkgroupStorageBytes(geometry),
          declaredWorkgroupBytes(wgsl),
          `${name} ${label(geometry)}`,
        );
      }
    }
  });
});

describe("the limit check of injected tables (the device's workgroup limits — against the adapter's limits, before the device is created)", () => {
  it("the default table and every embedded table pass under WebGPU's default limits (the reason the automatic selection path need not go through the limit check)", () => {
    assert(BUILTIN_GEOMETRY_PROFILES.length > 0);
    for (const profile of SAMPLE_PROFILES) {
      assertGeometryProfileWithinLimits(profile, WEBGPU_DEFAULT_LIMITS);
    }
  });

  it("a table exactly at the limits (the default geometry with 256 threads, 16,384 B) proceeds to requestDevice", async () => {
    const profile = profileOf("at-limit", undefined);
    const requests = await withCountingGpu(async () => {
      await assertRejects(() => acquireGpu({ geometryProfile: profile }), DeviceRequested);
    }, WEBGPU_DEFAULT_LIMITS);
    assertEquals(requests, 1);
  });

  it("thread count wgX × wgY over the limit → GpuFeatureError (naming the field, the geometry and the limit) with 0 requestDevice calls", async () => {
    const profile = profileOf("threads", undefined, {
      attention: { qk: THREADS_512, pv: defaultGemmGeometry() },
    });
    const requests = await withCountingGpu(async () => {
      await assertRejects(
        () => acquireGpu({ geometryProfile: profile }),
        GpuFeatureError,
        "geometryProfile: Geometry profile 'threads': geometry exceeds the device limits (attention.qk" +
          " (regM=8 regN=4 wgX=32 wgY=16): thread count wgX × wgY 512 exceeds maxComputeInvocationsPerWorkgroup 256)",
      );
    }, WEBGPU_DEFAULT_LIMITS);
    assertEquals(requests, 0, "a table over the limit reached requestDevice");
    // Control: the limit is not a constant but the adapter's limits — it passes on an adapter with
    // 1024 invocations
    const allowed = await withCountingGpu(
      async () => {
        await assertRejects(() => acquireGpu({ geometryProfile: profile }), DeviceRequested);
      },
      { ...WEBGPU_DEFAULT_LIMITS, maxComputeInvocationsPerWorkgroup: 1024 },
    );
    assertEquals(allowed, 1);
  });

  it("a component of @workgroup_size over the limit (thread count fits) → GpuFeatureError naming the component's limit", async () => {
    const profile = profileOf("size-x", undefined, {
      attention: { qk: THREADS_512, pv: defaultGemmGeometry() },
    });
    const requests = await withCountingGpu(
      async () => {
        await assertRejects(
          () => acquireGpu({ geometryProfile: profile }),
          GpuFeatureError,
          "attention.qk (regM=8 regN=4 wgX=32 wgY=16): wgX 32 exceeds maxComputeWorkgroupSizeX 16",
        );
      },
      {
        ...WEBGPU_DEFAULT_LIMITS,
        maxComputeInvocationsPerWorkgroup: 1024,
        maxComputeWorkgroupSizeX: 16,
      },
    );
    assertEquals(requests, 0);
  });

  it("only the Y component of @workgroup_size over the limit → GpuFeatureError naming the Y limit", async () => {
    const profile = profileOf("size-y", undefined, {
      attention: { qk: THREADS_512, pv: defaultGemmGeometry() },
    });
    const requests = await withCountingGpu(
      async () => {
        await assertRejects(
          () => acquireGpu({ geometryProfile: profile }),
          GpuFeatureError,
          "attention.qk (regM=8 regN=4 wgX=32 wgY=16): wgY 16 exceeds maxComputeWorkgroupSizeY 8",
        );
      },
      {
        ...WEBGPU_DEFAULT_LIMITS,
        maxComputeInvocationsPerWorkgroup: 1024,
        maxComputeWorkgroupSizeY: 8,
      },
    );
    assertEquals(requests, 0);
  });

  it("shared memory over the limit → GpuFeatureError (naming every gemmRows step and i8a8 field) with 0 requestDevice calls", async () => {
    const profile = profileOf("storage", undefined, {
      gemmRows: [
        DEFAULT_GEOMETRY_PROFILE.gemmRows[0],
        { maxRows: Number.POSITIVE_INFINITY, geometry: STORAGE_20K },
      ],
      i8a8: { ...DEFAULT_GEOMETRY_PROFILE.i8a8, attentionPv: I8A8_STORAGE_32K },
    });
    const requests = await withCountingGpu(async () => {
      const error = await assertRejects(
        () => acquireGpu({ geometryProfile: profile }),
        GpuFeatureError,
        "gemmRows[1] (regM=8 regN=8 wgX=32 wgY=8): shared memory (bytes) 20480 exceeds maxComputeWorkgroupStorageSize 16384",
      );
      assert(
        error.message.includes(
          "i8a8.attentionPv (regM=8 regN=8 wgX=16 wgY=16 tileK=128): shared memory (bytes) 32768 exceeds " +
            "maxComputeWorkgroupStorageSize 16384",
        ),
        error.message,
      );
    }, WEBGPU_DEFAULT_LIMITS);
    assertEquals(requests, 0, "a table over the limit reached requestDevice");
    const allowed = await withCountingGpu(
      async () => {
        await assertRejects(() => acquireGpu({ geometryProfile: profile }), DeviceRequested);
      },
      { ...WEBGPU_DEFAULT_LIMITS, maxComputeWorkgroupStorageSize: 32_768 },
    );
    assertEquals(allowed, 1);
  });

  it("the same check applies to the callback's return (after the structural check, before the device is created)", async () => {
    const profile = profileOf("callback", undefined, {
      conv2d: { rows64: STORAGE_20K, rows32: gemmMTileGeometry(32) },
    });
    let calls = 0;
    const requests = await withCountingGpu(async () => {
      await assertRejects(
        () =>
          acquireGpu({
            geometryProfile: () => {
              calls += 1;
              return profile;
            },
          }),
        GpuFeatureError,
        "geometryProfile (callback return value): Geometry profile 'callback': geometry exceeds the device limits (conv2d.rows64",
      );
    }, WEBGPU_DEFAULT_LIMITS);
    assertEquals(calls, 1);
    assertEquals(requests, 0, "a table over the limit reached requestDevice");
  });

  it("the structural check comes first (a table that is broken and also over the limits fails with the structural wording)", async () => {
    const hole: GemmGeometry = { regM: 3, regN: 4, wgX: 32, wgY: 16 };
    const profile = profileOf("order", undefined, { attention: { qk: hole, pv: THREADS_512 } });
    const requests = await withCountingGpu(async () => {
      await assertRejects(
        () => acquireGpu({ geometryProfile: profile }),
        GpuFeatureError,
        "geometryProfile: Geometry profile 'order': attention.qk",
      );
    }, WEBGPU_DEFAULT_LIMITS);
    assertEquals(requests, 0);
  });
});

describe("the optional vec4 skeleton of the i8a8 fields (perf-ledger K-85)", () => {
  const skeleton: I8a8Geometry = {
    regM: 16,
    regN: 4,
    wgX: 16,
    wgY: 8,
    tileK: 32,
    skeleton: "vec4",
  };

  it("a table with the skeleton in an i8a8 field passes the check and the limit check under WebGPU's default limits", () => {
    const profile: unknown = {
      ...OPT_IN,
      i8a8: { ...OPT_IN.i8a8, linear: skeleton, conv3d: { ...skeleton, tileK: 64 } },
    };
    assertGeometryProfile(profile);
    assertEquals(profile.i8a8.linear.skeleton, "vec4");
    assertGeometryProfileWithinLimits(profile, WEBGPU_DEFAULT_LIMITS);
  });

  it("an unknown skeleton value and a skeleton geometry the vec4 fill cannot cover are rejected naming the field (CodegenError)", () => {
    const cases: readonly (readonly [string, unknown, string])[] = [
      [
        "an unknown value",
        { ...OPT_IN, i8a8: { ...OPT_IN.i8a8, attentionPv: { ...skeleton, skeleton: "vec8" } } },
        'i8a8.attentionPv: geometry skeleton must be "vec4" or absent ("vec8")',
      ],
      [
        "a non-string value",
        { ...OPT_IN, i8a8: { ...OPT_IN.i8a8, linear: { ...skeleton, skeleton: 4 } } },
        'i8a8.linear: geometry skeleton must be "vec4" or absent (4)',
      ],
      [
        "tileK 24",
        {
          ...OPT_IN,
          i8a8: {
            ...OPT_IN.i8a8,
            linear: { regM: 4, regN: 4, wgX: 8, wgY: 12, tileK: 24, skeleton: "vec4" },
          },
        },
        "i8a8.linear: the vec4 skeleton needs tileK to be a multiple of 16 (24)",
      ],
    ];
    for (const [name, value, message] of cases) {
      assertThrows(() => assertGeometryProfile(value), CodegenError, message, name);
    }
  });
});

describe("the optional skeleton form and counter decode of the f32 fields (perf-ledger K-88)", () => {
  const FORMS: GeometryProfile = {
    ...OPT_IN,
    id: "k88-fields",
    gemmRows: OPT_IN.gemmRows.map((rule, index) => ({
      ...rule,
      geometry: { ...rule.geometry, form: index % 2 === 0 ? "tb-un" : "tas-tb-un" },
    })),
    conv2d: {
      rows64: { ...OPT_IN.conv2d.rows64, form: "tas-tb-un", decode: "counter" },
      rows32: { ...OPT_IN.conv2d.rows32, decode: "counter" },
    },
  };

  it("a table with the form in its row-count buckets and the form and the decode in conv2d passes the check and the limit check under WebGPU's default limits, and the selection reads them", () => {
    const profile: unknown = FORMS;
    assertGeometryProfile(profile);
    assertGeometryProfileWithinLimits(profile, WEBGPU_DEFAULT_LIMITS);
    assertEquals(gemmRowsGeometry(profile, 4096).form, FORMS.gemmRows.at(-1)?.geometry.form);
    assertEquals(conv2dProfileGeometry(profile, 64), FORMS.conv2d.rows64);
    assertEquals(conv2dProfileGeometry(profile, 32).decode, "counter");
  });

  it("acquireGpu injects such a table as a copy that keeps the fields (no device is needed)", async () => {
    await withFakeAdapter(FAKE_INFO, async () => {
      const gpu = await acquireGpu({ geometryProfile: FORMS });
      try {
        const injected = gpu[RUNTIME_INTERNAL].geometryProfile;
        assertEquals(injected, FORMS);
        assert(injected !== FORMS, "the injected table was not copied");
        assertEquals(injected?.conv2d.rows64.decode, "counter");
      } finally {
        gpu.destroy();
      }
    });
  });

  it("an unknown value, the decode outside conv2d and either field in the fused attention are rejected naming the field (CodegenError, and GpuFeatureError at injection)", async () => {
    const raw = (geometry: GemmGeometry, field: string, value: unknown): GemmGeometry => {
      const copy: GemmGeometry = { ...geometry };
      Reflect.set(copy, field, value);
      return copy;
    };
    const rows = (geometry: GemmGeometry): GeometryProfile["gemmRows"] =>
      OPT_IN.gemmRows.map((rule, index) => (index === 0 ? { ...rule, geometry } : rule));
    const first = OPT_IN.gemmRows[0].geometry;
    const cases: readonly (readonly [string, unknown, string])[] = [
      [
        "an unknown form",
        { ...OPT_IN, gemmRows: rows(raw(first, "form", "tb")) },
        'gemmRows[0].geometry: geometry form must be "tb-un", "tas-tb-un" or absent ("tb")',
      ],
      [
        "a non-string form",
        { ...OPT_IN, conv2d: { ...OPT_IN.conv2d, rows32: raw(OPT_IN.conv2d.rows32, "form", 1) } },
        'conv2d.rows32: geometry form must be "tb-un", "tas-tb-un" or absent (1)',
      ],
      [
        "an unknown decode",
        {
          ...OPT_IN,
          conv2d: { ...OPT_IN.conv2d, rows64: raw(OPT_IN.conv2d.rows64, "decode", "table") },
        },
        'conv2d.rows64: geometry decode must be "counter" or absent ("table")',
      ],
      [
        "the decode in a row-count bucket",
        { ...OPT_IN, gemmRows: rows({ ...first, decode: "counter" }) },
        'gemmRows[0].geometry: the counter decode is conv2d\'s alone (decode "counter")',
      ],
      [
        "the form in the fused attention",
        {
          ...OPT_IN,
          attention: { ...OPT_IN.attention, pv: { ...OPT_IN.attention.pv, form: "tb-un" } },
        },
        'attention.pv: the skeleton form is taken only by linear / matmul / bmm and conv2d (form "tb-un"',
      ],
      [
        "the decode in the fused attention",
        {
          ...OPT_IN,
          attention: { ...OPT_IN.attention, qk: { ...OPT_IN.attention.qk, decode: "counter" } },
        },
        'attention.qk: the counter decode is conv2d\'s alone (decode "counter")',
      ],
    ];
    for (const [name, value, message] of cases) {
      assertThrows(() => assertGeometryProfile(value), CodegenError, message, name);
      const requests = await withFakeAdapter(FAKE_INFO, async () => {
        await assertRejects(
          () => acquireGpu({ geometryProfile: untypedCallback(value) }),
          GpuFeatureError,
          `geometryProfile (callback return value): Geometry profile 'opt-in': ${message}`,
          name,
        );
      });
      assertEquals(requests, 0, `${name}: a rejected table reached requestDevice`);
    }
  });
});
