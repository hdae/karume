// Real-GPU check that the GEMM geometry profile (ADR 0115 — src/kernels/geometry-profile.ts) is
// wired through to the Session's derivation phase. The behaviour of the pure functions is in
// geometry_profile_test.ts.
//
// It looks at two things:
//
// 1. The name of the selected profile appears in the diagnostics
//    (`SessionDiagnostics.geometryProfile`).
// 2. The output of a Session built with each built-in profile (src/kernels/geometry-profiles/)
//    **matches the default profile's Session as Uint32 for the same inputs**. The geometry may
//    change only the work split (the numeric contract of ADR 0022); if the derivation phase passes
//    a different geometry to only one of the key, the WGSL or the dispatch, output tiles go missing
//    and show up here as a difference in values.
//    MUST: matching values alone are no evidence of the wiring (they also match if the profile is
//    ignored), so it also checks that the profile's geometry discriminator appeared in **the
//    pipeline keys that actually ran**.
//
// A profile is selected by the adapter's (vendor, architecture, description), so 2 builds the
// Session with **a GpuContext that keeps the real device and only sets adapterInfo to that
// profile's match** (the GPU running here is not the machine the profile was made on, but that the
// geometry does not move bits is a claim independent of the machine — ADR 0022 addendum). A table
// for injection only, with `match` omitted, is never selected by an adapter, so it is passed
// **through the injection field** of the same GpuContext without the disguise (every table runs —
// unless a table that is never selected automatically is also bit-identical to the default, the
// output of a user who injects it moves). The default side is also built with a GpuContext whose
// adapterInfo is empty, so that the baseline of the comparison stays the default even if a profile
// matches the real adapter in the future.
// While there are no built-in profiles at all, 2 is SKIPped explicitly (an empty list is not
// green).
//
// 3. A table injected with `acquireGpu({ geometryProfile })` is used without looking at the
//    adapter's (vendor, architecture, description) (nor at `match`). Besides the same 3 points as 2
//    (the name in the diagnostics, the geometry discriminator of the keys that ran, Uint32
//    agreement with the default), it checks in the WGSL the canary compiled that the dp4a canary of
//    the i8a8 attention fires with that table's geometry.
//    A table returned by the callback form (ADR 0117 decision 6) is used with the same 3 points,
//    and `undefined` means automatic selection.

import { platformObject } from "@karume/test-support/webgpu";
import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu, GpuContext } from "../src/gpu/device.ts";
import {
  DEFAULT_GEOMETRY_PROFILE,
  gemmRowsGeometry,
  type GemmRowsRule,
  type GeometryProfile,
  selectGeometryProfile,
} from "../src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../src/kernels/geometry-profiles/index.ts";
import { APPLE_METAL_3 } from "../src/kernels/geometry-profiles/apple-metal-3.ts";
import { NVIDIA_BLACKWELL } from "../src/kernels/geometry-profiles/nvidia-blackwell.ts";
import { attentionPvI8a8Wgsl, attentionQkI8a8Wgsl } from "../src/kernels/attention-i8a8.ts";
import { dp4aAvailable } from "../src/kernels/linear-i8a8.ts";
import {
  type GemmGeometry,
  gemmGeometryTileKeyPart,
  gemmTileM,
  gemmTileN,
} from "../src/kernels/gemm-geometry.ts";
import { i8a8GeometryKeyPart } from "../src/kernels/i8a8-geometry.ts";
import { perChannelGroupSize } from "../src/format/container/codecs.ts";
import type { OpenedContainer } from "../src/format/container/open.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { I8A8_DOT, type SessionOptions } from "../src/runtime/session-types.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import {
  f32Bytes,
  fill,
  GRAPH_NAME,
  openGraphModel,
  openModelBytes,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";

/**
 * Deterministic sequences with spread signs and magnitudes (no random numbers — a failure would not
 * reproduce).
 */
const SIGNED = (index: number): number => ((index % 11) - 5) * 0.375 + 0.125;
const WEIGHT = (index: number): number => ((index % 7) - 3) * 0.5 - 0.0625;
const BIAS = (index: number): number => ((index % 5) - 2) * 0.25;

const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

/**
 * Asserts that one of the keys that actually ran starts with `prefix` and contains every one of
 * `parts`.
 */
type ExpectedKey = { readonly prefix: string; readonly parts: readonly string[] };

type ProfileCase = {
  readonly name: string;
  readonly model: () => Promise<OpenedContainer>;
  readonly inputs: Readonly<Record<string, Tensor>>;
  readonly options?: SessionOptions;
  /** The discriminator of the geometry that should run with that profile. */
  readonly expected: (profile: GeometryProfile) => readonly ExpectedKey[];
};

const rowsKey = (prefix: string, rows: number) => (profile: GeometryProfile) => [
  { prefix, parts: [gemmGeometryTileKeyPart(gemmRowsGeometry(profile, rows))] },
];

const linearCase = (m: number): ProfileCase => {
  const [k, n] = [64, 96];
  return {
    name: `linear M=${m}`,
    model: async () =>
      await openGraphModel(singleOpDeclaration("linear", [[m, k], [n, k], [n]], [[m, n]])),
    inputs: { x0: fill([m, k], SIGNED), x1: fill([n, k], WEIGHT), x2: fill([n], BIAS) },
    expected: rowsKey("linear:", m),
  };
};

/**
 * The representative M of each `gemmRows` rule (DECIDED: ADR 0116 decision 7). A finite rule uses
 * its upper end (= `maxRows` — the upper end of a bucket is a point measured by the sweep); the
 * last (Infinity) rule uses the previous boundary + 512. Derived from the table, so the cases
 * follow the number of rules when the bucket count changes.
 */
const representativeRows = (rules: readonly GemmRowsRule[]): readonly number[] =>
  rules.map((rule, index) =>
    rule.maxRows === Number.POSITIVE_INFINITY
      ? (rules[index - 1]?.maxRows ?? 0) + 512
      : rule.maxRows,
  );

/**
 * One case per `gemmRows` rule of the table (a linear with the representative M). The expected
 * discriminator is taken from **that rule's geometry**, not looked up again with `gemmRowsGeometry`
 * — a derivation error that lets the representative M fall into a neighbouring rule also turns red
 * through a key mismatch. For a rule with the same geometry as the default, it checks that the
 * discriminator is the default geometry's (the flip side of "a different geometry gives a
 * different key").
 */
const gemmRowsCases = (rules: readonly GemmRowsRule[]): readonly ProfileCase[] =>
  representativeRows(rules).map((m, index) => {
    const { maxRows, geometry } = rules[index];
    const range =
      maxRows === Number.POSITIVE_INFINITY ? `> ${rules[index - 1]?.maxRows ?? 0}` : `≤ ${maxRows}`;
    return {
      ...linearCase(m),
      name: `linear M=${m}（gemmRows ${range}）`,
      expected: () => [{ prefix: "linear:", parts: [gemmGeometryTileKeyPart(geometry)] }],
    };
  });

/**
 * The implicit GEMM key of conv2d names its geometry with `igemm{tileM}x{tileN}` and `:wg{x}x{y}`.
 */
const conv2dParts = (geometry: GemmGeometry): readonly string[] => [
  `igemm${gemmTileM(geometry)}x${gemmTileN(geometry)}`,
  `:wg${geometry.wgX}x${geometry.wgY}`,
];

const conv2dCase = (channelsOut: number, rows: "rows64" | "rows32"): ProfileCase => {
  const channelsIn = 8;
  return {
    name: `conv2d Cout=${channelsOut} (${rows} class)`,
    model: async () =>
      await openGraphModel(
        singleOpDeclaration(
          "conv2d",
          [[1, channelsIn, 20, 20], [channelsOut, channelsIn, 3, 3], [channelsOut]],
          [[1, channelsOut, 20, 20]],
          { attrs: { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 } },
        ),
      ),
    inputs: {
      x0: fill([1, channelsIn, 20, 20], SIGNED),
      x1: fill([channelsOut, channelsIn, 3, 3], WEIGHT),
      x2: fill([channelsOut], BIAS),
    },
    expected: (profile) => [
      {
        prefix: "conv2d:v3:f32:igemm",
        parts: conv2dParts(profile.conv2d[rows]),
      },
    ],
  };
};

const attentionCase = (compute: "f32" | "a8"): ProfileCase => {
  const [b, h, m, n, d] = [1, 2, 300, 200, 64];
  return {
    name: `attention ${compute}`,
    model: async () =>
      await openGraphModel(
        singleOpDeclaration(
          "attention",
          [
            [b, h, m, d],
            [b, h, n, d],
            [b, h, n, d],
          ],
          [[b, h, m, d]],
          { attrs: { scale: halfScale(d) } },
        ),
      ),
    inputs: {
      x0: fill([b, h, m, d], SIGNED),
      x1: fill([b, h, n, d], WEIGHT),
      x2: fill([b, h, n, d], BIAS),
    },
    // The dot-product variant is the same in both Sessions (the per-device canary's verdict stays
    // out of the comparison).
    options: compute === "a8" ? { attentionCompute: "a8", [I8A8_DOT]: "emu" } : {},
    expected: (profile) =>
      compute === "a8"
        ? [
            {
              prefix: "attention_qk:v3:i8a8:",
              parts: [i8a8GeometryKeyPart(profile.i8a8.attentionQk, false)],
            },
            {
              prefix: "attention_pv:v3:i8a8:",
              parts: [i8a8GeometryKeyPart(profile.i8a8.attentionPv, false)],
            },
          ]
        : [
            {
              prefix: "attention_qk:v1:f32:",
              parts: [gemmGeometryTileKeyPart(profile.attention.qk)],
            },
            {
              prefix: "attention_pv:v1:f32:",
              parts: [gemmGeometryTileKeyPart(profile.attention.pv)],
            },
          ],
  };
};

/**
 * A linear with i8 resident weights (the form that goes to the i8a8 path with `linearCompute:
 * "a8"`).
 */
const linearI8a8Case = (): ProfileCase => {
  const [m, k, n] = [300, 64, 96];
  const weight = fill([n, k], (i) => WEIGHT(i) * (1 + (Math.floor(i / k) % 7) * 0.25));
  const quantized = quantizeI8(weight.data, [n, k], 0);
  const bias = fill([n], BIAS);
  return {
    name: "linear i8a8",
    model: () =>
      openModelBytes(
        {
          format: "karume-ir",
          version: 2,
          requires: { ops: ["linear"] },
          symbols: [],
          inputs: [{ name: "x0", dtype: "f32", shape: [m, k] }],
          outputs: ["y"],
          initializers: { w: {}, b: {} },
          values: {
            w: { dtype: "f32", shape: [n, k] },
            b: { dtype: "f32", shape: [n] },
            y: { dtype: "f32", shape: [m, n] },
          },
          nodes: [{ op: "linear", ins: ["x0", "w", "b"], outs: ["y"], attrs: {} }],
        },
        [
          {
            graph: GRAPH_NAME,
            initializer: "b",
            bytes: f32Bytes(bias.data),
            encoding: { codec: "f32" },
          },
          {
            graph: GRAPH_NAME,
            initializer: "w",
            bytes: quantized.bytes,
            encoding: {
              codec: "int8-sym",
              groupSize: perChannelGroupSize(k),
              scale: { bytes: f32Bytes(quantized.scale), dtype: "f32" },
            },
          },
        ],
      ),
    inputs: { x0: fill([m, k], SIGNED) },
    options: { linearCompute: "a8", [I8A8_DOT]: "emu" },
    expected: (profile) => [
      {
        prefix: "linear:v4:i8a8:",
        parts: [i8a8GeometryKeyPart(profile.i8a8.linear, false)],
      },
    ],
  };
};

/**
 * conv3d w8a8 (`convCompute: "a8"` with an i8-resident weight — ADR 0133): the GEMM runs the profile's
 * `i8a8.conv3d` (ADR 0115 addendum 2026-10-26). Cout 96 over P = 96 output pixels, Cin 32, 3×3×3 with
 * spatial padding.
 */
const conv3dI8a8Case = (): ProfileCase => {
  const [channelsIn, channelsOut] = [32, 96];
  const input = [3, 8, 12] as const;
  const output = [1, 8, 12] as const;
  const rowLength = channelsIn * 27;
  const weight = fill(
    [channelsOut, channelsIn, 3, 3, 3],
    (i) => WEIGHT(i) * (1 + (Math.floor(i / rowLength) % 7) * 0.25),
  );
  const quantized = quantizeI8(weight.data, [channelsOut, rowLength], 0);
  const bias = fill([channelsOut], BIAS);
  return {
    name: "conv3d i8a8",
    model: () =>
      openModelBytes(
        {
          format: "karume-ir",
          version: 2,
          requires: { ops: ["conv3d"] },
          symbols: [],
          inputs: [{ name: "x0", dtype: "f32", shape: [channelsIn, ...input] }],
          outputs: ["y"],
          initializers: { w: {}, b: {} },
          values: {
            w: { dtype: "f32", shape: [channelsOut, channelsIn, 3, 3, 3] },
            b: { dtype: "f32", shape: [channelsOut] },
            y: { dtype: "f32", shape: [channelsOut, ...output] },
          },
          nodes: [
            {
              op: "conv3d",
              ins: ["x0", "w", "b"],
              outs: ["y"],
              attrs: { stride: [1, 1, 1], padding: [0, 1, 1], dilation: [1, 1, 1], groups: 1 },
            },
          ],
        },
        [
          {
            graph: GRAPH_NAME,
            initializer: "b",
            bytes: f32Bytes(bias.data),
            encoding: { codec: "f32" },
          },
          {
            graph: GRAPH_NAME,
            initializer: "w",
            bytes: quantized.bytes,
            encoding: {
              codec: "int8-sym",
              groupSize: perChannelGroupSize(rowLength),
              scale: { bytes: f32Bytes(quantized.scale), dtype: "f32" },
            },
          },
        ],
      ),
    inputs: { x0: fill([channelsIn, ...input], SIGNED) },
    options: { convCompute: "a8" },
    expected: (profile) => [
      {
        prefix: "conv3d:v1:i8a8:",
        parts: [i8a8GeometryKeyPart(profile.i8a8.conv3d, false)],
      },
    ],
  };
};

/**
 * Cases other than the linear row-count buckets (the per-profile check builds its linears from the
 * table's rules — `gemmRowsCases`).
 */
const OTHER_CASES: readonly ProfileCase[] = [
  {
    name: "matmul M=700",
    model: async () =>
      await openGraphModel(
        singleOpDeclaration(
          "matmul",
          [
            [700, 64],
            [64, 80],
          ],
          [[700, 80]],
        ),
      ),
    inputs: { x0: fill([700, 64], SIGNED), x1: fill([64, 80], WEIGHT) },
    expected: rowsKey("matmul:", 700),
  },
  {
    name: "bmm M=600",
    model: async () =>
      await openGraphModel(
        singleOpDeclaration(
          "bmm",
          [
            [2, 600, 64],
            [2, 64, 80],
          ],
          [[2, 600, 80]],
        ),
      ),
    inputs: { x0: fill([2, 600, 64], SIGNED), x1: fill([2, 64, 80], WEIGHT) },
    expected: rowsKey("bmm:", 600),
  },
  attentionCase("f32"),
  attentionCase("a8"),
  // Cout = 96 has M%64 = 32, so it is the 32-row class; 128 is the 64-row class
  // (`conv2dIgemmMTile`)
  conv2dCase(96, "rows32"),
  conv2dCase(128, "rows64"),
  linearI8a8Case(),
  conv3dI8a8Case(),
];

const CASES: readonly ProfileCase[] = [
  // One each for the 3 row-count buckets (≤ 64 / 65–512 / 513–)
  linearCase(40),
  linearCase(300),
  linearCase(1024),
  ...OTHER_CASES,
];

/**
 * A GpuContext that keeps the real device and replaces only the vendor / architecture /
 * description of adapterInfo (the profile is selected once, from adapterInfo, while the Session is
 * built). The description is also passed explicitly — copying the real machine's value would make a
 * table that matches on description hit or miss depending on the machine it runs on.
 * `injected` is the same injection field that `acquireGpu({ geometryProfile })` passes to the
 * GpuContext.
 *
 * MUST: do not destroy the GpuContext made here — the device is shared with the original
 * GpuContext, and it is destroyed in one place on the original side (a double destroy sends the
 * loss notification to a side that does not expect it).
 */
const contextAs = (
  gpu: GpuContext,
  vendor: string,
  architecture: string,
  description: string,
  injected?: GeometryProfile,
): GpuContext =>
  new GpuContext(
    gpu.device,
    platformObject<GPUAdapterInfo>({
      vendor,
      architecture,
      device: gpu.adapterInfo.device,
      description,
      subgroupMinSize: gpu.adapterInfo.subgroupMinSize,
      subgroupMaxSize: gpu.adapterInfo.subgroupMaxSize,
      isFallbackAdapter: gpu.adapterInfo.isFallbackAdapter,
    }),
    gpu.limits,
    gpu.wgslLanguageFeatures,
    undefined,
    injected,
  );

const words = (tensor: Tensor): Uint32Array<ArrayBuffer> =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength / 4);

type CaseRun = {
  readonly y: Uint32Array<ArrayBuffer>;
  readonly profile: string;
  readonly keys: readonly string[];
};

const runCase = async (gpu: GpuContext, testCase: ProfileCase): Promise<CaseRun> => {
  const session = await createSessionFromContainer(
    gpu,
    await testCase.model(),
    GRAPH_NAME,
    testCase.options ?? {},
  );
  try {
    const y = words((await session.run(testCase.inputs))["y"]);
    const diagnostics = session.diagnostics();
    const census = diagnostics.lastRunPipelines;
    assert(census !== undefined, `${testCase.name}: no pipeline breakdown for the latest run`);
    return { y, profile: diagnostics.geometryProfile, keys: census.map((row) => row.key) };
  } finally {
    await session.dispose();
  }
};

/**
 * Compares 1 case with the default Session and checks `profile`'s name, the keys with the geometry
 * discriminator and Uint32 agreement.
 */
const assertRunsWithProfile = async (
  baseline: GpuContext,
  subject: GpuContext,
  profile: GeometryProfile,
  testCase: ProfileCase,
): Promise<void> => {
  const expected = await runCase(baseline, testCase);
  assertEquals(expected.profile, DEFAULT_GEOMETRY_PROFILE.id, testCase.name);
  const actual = await runCase(subject, testCase);
  assertEquals(actual.profile, profile.id, testCase.name);
  for (const { prefix, parts } of testCase.expected(profile)) {
    assert(
      actual.keys.some(
        (key) => key.startsWith(prefix) && parts.every((part) => key.includes(part)),
      ),
      `${testCase.name}: did not run with a ${prefix}…${parts.join("…")} key` +
        ` (actual: ${actual.keys.join(" / ")})`,
    );
  }
  assertEquals(actual.y, expected.y, `${testCase.name}: bits differ from the default profile`);
};

describe("representative M of gemmRows", () => {
  const rulesOf = (bounds: readonly number[]): GemmRowsRule[] =>
    bounds.map((maxRows) => ({ maxRows, geometry: DEFAULT_GEOMETRY_PROFILE.gemmRows[0].geometry }));

  it("with the 3-bucket table (the default 64 / 512 / ∞) gives each bucket's upper end, and the previous boundary + 512 for the last", () => {
    assertEquals(representativeRows(DEFAULT_GEOMETRY_PROFILE.gemmRows), [64, 512, 1024]);
  });

  it("with the 7-bucket table (16 / 32 / 64 / 128 / 256 / 512 / ∞) gives as many representative M as there are rules", () => {
    const rules = rulesOf([16, 32, 64, 128, 256, 512, Number.POSITIVE_INFINITY]);
    assertEquals(representativeRows(rules), [16, 32, 64, 128, 256, 512, 1024]);
  });

  it("each rule's case expects the discriminator of that rule's geometry", () => {
    const cases = gemmRowsCases(DEFAULT_GEOMETRY_PROFILE.gemmRows);
    assertEquals(
      cases.map((testCase) => testCase.expected(DEFAULT_GEOMETRY_PROFILE)[0].parts),
      DEFAULT_GEOMETRY_PROFILE.gemmRows.map((rule) => [gemmGeometryTileKeyPart(rule.geometry)]),
    );
  });
});

describe.skipIf(!GPU_AVAILABLE)("geometry profile diagnostics (real GPU)", () => {
  it("the Session reports in its diagnostics the name of the profile selected from the real adapter", async () => {
    const gpu = await acquireGpu();
    try {
      const run = await runCase(gpu, linearCase(40));
      assertEquals(run.profile, selectGeometryProfile(gpu.adapterInfo).id);
      // A machine with empty adapterInfo (adapter.info missing in old Chromium) falls back to the
      // default
      const blank = await runCase(contextAs(gpu, "", "", ""), linearCase(40));
      assertEquals(blank.profile, DEFAULT_GEOMETRY_PROFILE.id);
    } finally {
      gpu.destroy();
    }
  });
});

describe.skipIf(!GPU_AVAILABLE || BUILTIN_GEOMETRY_PROFILES.length === 0)(
  "the geometry of the built-in profiles is bit-identical to the default profile (real GPU)",
  () => {
    for (const profile of BUILTIN_GEOMETRY_PROFILES) {
      const route = profile.match === undefined ? "injection" : "adapter selection";
      it(`the Session of '${profile.id}' (${route}) matches the default as Uint32 in every case and runs with that geometry's keys`, async () => {
        const gpu = await acquireGpu();
        try {
          const baseline = contextAs(gpu, "", "", "");
          // A table with match is built through the selection path with a disguised adapter
          // (description included); a table for injection only, with match omitted, is built
          // through the injection field (the adapter is empty — selection would only fall back to
          // the default).
          const subject =
            profile.match === undefined
              ? contextAs(gpu, "", "", "", profile)
              : contextAs(
                  gpu,
                  profile.match.vendor ?? "",
                  profile.match.architecture ?? "",
                  profile.match.description ?? "",
                );
          const rowCases = gemmRowsCases(profile.gemmRows);
          assertEquals(rowCases.length, profile.gemmRows.length, "missed some gemmRows rules");
          for (const testCase of [...rowCases, ...OTHER_CASES]) {
            await assertRunsWithProfile(baseline, subject, profile, testCase);
          }
        } finally {
          gpu.destroy();
        }
      });
    }
  },
);

describe.skipIf(!GPU_AVAILABLE)("geometry profile injected with acquireGpu (real GPU)", () => {
  it("the Sessions of a device with apple-metal-3 injected run with that table's name and geometry keys in every case and match the default as Uint32", async () => {
    const plain = await acquireGpu();
    // The passed table is copied at acquire time — changing it afterwards does not change the
    // device's table (id)
    const handedOver = structuredClone(APPLE_METAL_3);
    const injected = await acquireGpu({ geometryProfile: handedOver });
    (handedOver as { id: string }).id = "changed-after-acquire";
    try {
      const baseline = contextAs(plain, "", "", "");
      for (const testCase of CASES) {
        await assertRunsWithProfile(baseline, injected, APPLE_METAL_3, testCase);
      }
    } finally {
      injected.destroy();
      plain.destroy();
    }
  });

  it("callback form (ADR 0117 decision 6): called once with the device's adapter information, the Session runs with the returned built-in table and matches the default as Uint32; undefined means automatic selection", async () => {
    const plain = await acquireGpu();
    const seen: GPUAdapterInfo[] = [];
    const injected = await acquireGpu({
      geometryProfile: (adapterInfo) => {
        seen.push(adapterInfo);
        return APPLE_METAL_3;
      },
    });
    const automatic = await acquireGpu({ geometryProfile: () => undefined });
    try {
      assertEquals(seen.length, 1);
      assert(
        seen[0] === injected.adapterInfo,
        "the callback's adapterInfo is a different object from the GpuContext's",
      );
      const baseline = contextAs(plain, "", "", "");
      for (const testCase of [linearCase(40), linearCase(300), linearI8a8Case()]) {
        await assertRunsWithProfile(baseline, injected, APPLE_METAL_3, testCase);
      }
      const run = await runCase(automatic, linearCase(40));
      assertEquals(run.profile, selectGeometryProfile(automatic.adapterInfo).id);
    } finally {
      automatic.destroy();
      injected.destroy();
      plain.destroy();
    }
  });

  it("an injected table ignores the adapter and match (a table of another vendor is used even on an adapter that apple-metal-3 matches)", async () => {
    const gpu = await acquireGpu();
    try {
      const baseline = contextAs(gpu, "", "", "");
      // Inject a table whose match is nvidia into an adapter that the built-in selection would
      // match with apple-metal-3
      const subject = contextAs(
        gpu,
        "apple",
        "metal-3",
        APPLE_METAL_3.match?.description ?? "",
        NVIDIA_BLACKWELL,
      );
      assertEquals(selectGeometryProfile(subject.adapterInfo).id, APPLE_METAL_3.id);
      for (const testCase of [linearCase(40), linearCase(300), linearI8a8Case()]) {
        await assertRunsWithProfile(baseline, subject, NVIDIA_BLACKWELL, testCase);
      }
    } finally {
      gpu.destroy();
    }
  });
});

describe.skipIf(!GPU_AVAILABLE)(
  "the dp4a canary fires with the injected table's i8a8 attention geometry (real GPU)",
  () => {
    it("while building a Session with attentionCompute a8, the canary compiles WGSL with the injected table's ①QK / ③PV geometry", async (context) => {
      const gpu = await acquireGpu({ geometryProfile: APPLE_METAL_3 });
      try {
        if (!dp4aAvailable(gpu.wgslLanguageFeatures)) {
          // On a device that does not advertise it the canary does not run (straight to emu) —
          // nothing to check, so SKIP explicitly
          console.warn(`SKIP ${context.task.name}: dot4I8Packed is not advertised`);
          return;
        }
        const compiled: string[] = [];
        const device = gpu.device;
        const createShaderModule = device.createShaderModule.bind(device);
        device.createShaderModule = (descriptor) => {
          compiled.push(descriptor.code);
          return createShaderModule(descriptor);
        };
        const testCase = attentionCase("a8");
        const session = await createSessionFromContainer(gpu, await testCase.model(), GRAPH_NAME, {
          attentionCompute: "a8",
        });
        try {
          assertEquals(session.diagnostics().geometryProfile, APPLE_METAL_3.id);
        } finally {
          await session.dispose();
        }
        // The canary always fires the dp4a arm first (including the v4 / S = f32 pair)
        const injectedQk = attentionQkI8a8Wgsl(true, true, "f32", APPLE_METAL_3.i8a8.attentionQk);
        const injectedPv = attentionPvI8a8Wgsl(true, true, "f32", APPLE_METAL_3.i8a8.attentionPv);
        const defaultQk = attentionQkI8a8Wgsl(
          true,
          true,
          "f32",
          DEFAULT_GEOMETRY_PROFILE.i8a8.attentionQk,
        );
        assert(
          injectedQk !== defaultQk,
          "the injected table's ①QK geometry equals the default (the check would miss)",
        );
        assert(
          compiled.includes(injectedQk),
          "the canary did not fire with the injected table's ①QK geometry",
        );
        assert(
          compiled.includes(injectedPv),
          "the canary did not fire with the injected table's ③PV geometry",
        );
        assert(!compiled.includes(defaultQk), "the canary fired with the default ①QK geometry");
      } finally {
        gpu.destroy();
      }
    });
  },
);
