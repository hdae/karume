// 生成器の純関数（derive.ts）のうち、注入の表（buildGeometryProfile）と注入の JSON（profileJson）の門。
// 規則の抽出そのものは profile_test.ts が固定する。GPU を使わない。

import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  defaultGemmGeometry,
  type GemmGeometry,
  gemmGeometryForRows,
} from "../../packages/runtime/src/kernels/gemm-geometry.ts";
import {
  assertGeometryProfile,
  DEFAULT_GEOMETRY_PROFILE,
  selectGeometryProfile,
} from "../../packages/runtime/src/kernels/geometry-profile.ts";
import {
  buildGeometryProfile,
  deriveProfile,
  parseSweepReport,
  profileJson,
  renderAppProfileSource,
  renderProfileSource,
  type SlotVerdict,
  type SweepSource,
} from "./derive.ts";
import { gemmCandidate } from "./geometries.ts";
import { REPORT_FORMAT } from "./report.ts";

const DEFAULT = defaultGemmGeometry();
const FAST: GemmGeometry = { regM: 4, regN: 4, wgX: 8, wgY: 16 };

/** linear M = 1024 の 1 ケース（既定と、既定より 1.2 倍速く出力の一致する幾何 1 つ）。 */
const sweep = (name: string, sha256: string): SweepSource =>
  parseSweepReport({
    format: REPORT_FORMAT,
    date: "2026-09-29T00:00:00.000Z",
    adapter: { vendor: "apple", architecture: "metal-3", device: "", description: "Test GPU" },
    gpuTiming: { feature: true, unit: "ns", quantized: false },
    cases: [{ caseId: "linear-m1024", defaultRepeat: { perDispatch: 1, driftRatio: 1 } }],
    rows: [
      {
        caseId: "linear-m1024",
        op: "linear",
        shape: "M1024 N64 K64",
        geometry: gemmCandidate(DEFAULT).name,
        geometryParams: DEFAULT,
        isDefault: true,
        speedupVsDefault: 1,
        identicalToDefault: true,
      },
      {
        caseId: "linear-m1024",
        op: "linear",
        shape: "M1024 N64 K64",
        geometry: gemmCandidate(FAST).name,
        geometryParams: FAST,
        isDefault: false,
        speedupVsDefault: 1.2,
        identicalToDefault: true,
      },
    ],
  }, { path: name, sha256 });

const SPEC = {
  from: ["a.json", "b.json"],
  id: "test-gpu",
  vendor: "apple",
  architecture: "metal-3",
  out: "profiles/test-gpu.ts",
  minSpeedup: 1.05,
} as const;

const SOURCES = [sweep("a.json", "sha-a"), sweep("b.json", "sha-b")];

/** 段の上端 → 範囲の綴り（7 段の期待値を手で書いたもの — 生成器の綴りの正本ではない）。 */
const RANGES: Readonly<Record<number, string>> = {
  16: "≤ 16",
  32: "17〜32",
  64: "33〜64",
  128: "65〜128",
  256: "129〜256",
  512: "257〜512",
};
const rangeOf = (maxRows: number): string => RANGES[maxRows] ?? "> 512";

const verdictOf = (verdicts: readonly SlotVerdict[], slot: SlotVerdict["slot"]): SlotVerdict => {
  const found = verdicts.find((verdict) => verdict.slot === slot);
  if (found === undefined) throw new Error(`${slot} が無い`);
  return found;
};

describe("buildGeometryProfile", () => {
  it("欄ごとの値は採否の幾何そのもので、runtime の門（acquireGpu が注入を受ける門）を通る", () => {
    const verdicts = deriveProfile(SOURCES, SPEC);
    const profile = buildGeometryProfile(SPEC, SOURCES, verdicts);
    assertGeometryProfile(profile);
    assertEquals(verdictOf(verdicts, "gemmRows > 512").outcome.kind, "adopted");
    assertEquals(profile.gemmRows.at(-1)?.geometry, FAST);
    assertEquals(
      profile.gemmRows.map((rule) => rule.maxRows),
      [16, 32, 64, 128, 256, 512, Infinity],
    );
    for (const rule of profile.gemmRows) {
      const slot = verdictOf(verdicts, `gemmRows ${rangeOf(rule.maxRows)}`);
      assertEquals(rule.geometry, slot.outcome.geometry, slot.slot);
    }
    const slots = [
      ["attention.qk", profile.attention.qk],
      ["attention.pv", profile.attention.pv],
      ["conv2d.rows64", profile.conv2d.rows64],
      ["conv2d.rows32", profile.conv2d.rows32],
      ["i8a8.linear", profile.i8a8.linear],
      ["i8a8.attentionQk", profile.i8a8.attentionQk],
      ["i8a8.attentionPv", profile.i8a8.attentionPv],
    ] as const;
    for (const [slot, geometry] of slots) {
      assertEquals(geometry, verdictOf(verdicts, slot).outcome.geometry, slot);
    }
  });

  it("掃引にケースが無い欄は既定の表（DEFAULT_GEOMETRY_PROFILE）と同じ値になる（gemmRows の段は範囲を覆う既定の段）", () => {
    const profile = buildGeometryProfile(SPEC, SOURCES, deriveProfile(SOURCES, SPEC));
    // 段の上端の M を runtime の既定の表で引いた幾何 = 段の範囲を覆う既定の段の幾何（段は既定の細分）
    for (const rule of profile.gemmRows.slice(0, -1)) {
      assertEquals(rule.geometry, gemmGeometryForRows(rule.maxRows), String(rule.maxRows));
    }
    assertEquals(profile.attention, DEFAULT_GEOMETRY_PROFILE.attention);
    assertEquals(profile.conv2d, DEFAULT_GEOMETRY_PROFILE.conv2d);
    assertEquals(profile.i8a8, DEFAULT_GEOMETRY_PROFILE.i8a8);
  });

  it("match と provenance は spec と掃引の順で決まる（architecture を省けば vendor だけ）", () => {
    const verdicts = deriveProfile(SOURCES, SPEC);
    const profile = buildGeometryProfile(SPEC, SOURCES, verdicts);
    assertEquals(profile.id, "test-gpu");
    assertEquals(profile.match, { vendor: "apple", architecture: "metal-3" });
    assertEquals(profile.provenance, {
      sweep: "a.json, b.json",
      sha256: "sha-a, sha-b",
      date: "2026-09-29T00:00:00.000Z, 2026-09-29T00:00:00.000Z",
      adapter: "apple / metal-3 / Test GPU",
    });
    const { architecture: _, ...vendorOnly } = SPEC;
    assertEquals(buildGeometryProfile(vendorOnly, SOURCES, verdicts).match, { vendor: "apple" });
  });

  it("TS の生成物は注入の表と同じ値を書く（7 規則・段の既定・採用した幾何と末尾の Infinity）", () => {
    const verdicts = deriveProfile(SOURCES, SPEC);
    const source = renderProfileSource(SPEC, SOURCES, verdicts);
    assertEquals(source.match(/\{ maxRows: /g)?.length, 7);
    assertStringIncludes(
      source,
      "{ maxRows: 16, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },",
    );
    assertStringIncludes(
      source,
      "{ maxRows: 128, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },",
    );
    assertStringIncludes(
      source,
      "{ maxRows: Number.POSITIVE_INFINITY, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },",
    );
    assertStringIncludes(source, 'provenance: { sweep: "a.json, b.json", sha256: "sha-a, sha-b"');
  });
});

describe("buildGeometryProfile: description と注入専用", () => {
  const described = { ...SPEC, description: "Test GPU" };
  const optIn = {
    from: SPEC.from,
    id: SPEC.id,
    optIn: true,
    out: SPEC.out,
    minSpeedup: SPEC.minSpeedup,
  } as const;
  const adapter = { vendor: "apple", architecture: "metal-3", description: "Test GPU" };

  it("description を指定した表は match に description を持ち、その機種にだけ当たる", () => {
    const profile = buildGeometryProfile(described, SOURCES, deriveProfile(SOURCES, described));
    assertEquals(profile.match, {
      vendor: "apple",
      architecture: "metal-3",
      description: "Test GPU",
    });
    assertEquals(selectGeometryProfile(adapter, [profile]), profile);
    assertEquals(
      selectGeometryProfile({ ...adapter, description: "" }, [profile]),
      DEFAULT_GEOMETRY_PROFILE,
    );
  });

  it("注入専用の表は match の欄ごと無く、門を通り、同じ adapter でも自動では選ばれない", () => {
    const profile = buildGeometryProfile(optIn, SOURCES, deriveProfile(SOURCES, optIn));
    assert(!("match" in profile), JSON.stringify(profile));
    assertGeometryProfile(profile);
    assertEquals(selectGeometryProfile(adapter, [profile]), DEFAULT_GEOMETRY_PROFILE);
    assert(!("match" in JSON.parse(profileJson(profile))));
  });

  it("TS の生成物: description は match に書き、注入専用は match の行を書かない", () => {
    const withDescription = renderProfileSource(
      described,
      SOURCES,
      deriveProfile(SOURCES, described),
    );
    assertStringIncludes(
      withDescription,
      'match: { vendor: "apple", architecture: "metal-3", description: "Test GPU" },',
    );
    assertStringIncludes(withDescription, "--description 'Test GPU'");
    const injected = renderProfileSource(optIn, SOURCES, deriveProfile(SOURCES, optIn));
    assert(!injected.includes("match:"), injected);
    assertStringIncludes(injected, "--id test-gpu --opt-in \\");
    assertStringIncludes(injected, "**注入専用**");
  });
});

describe("renderAppProfileSource", () => {
  it("公開 API の型を import し、末尾の maxRows を Number.POSITIVE_INFINITY と書く", () => {
    const profile = buildGeometryProfile(SPEC, SOURCES, deriveProfile(SOURCES, SPEC));
    const source = renderAppProfileSource(profile);
    assertStringIncludes(source, 'import type { GeometryProfile } from "@karume/runtime";\n');
    assertStringIncludes(source, "acquireGpu({ geometryProfile: TEST_GPU })");
    assertStringIncludes(source, "export const TEST_GPU: GeometryProfile = {");
    assertStringIncludes(
      source,
      "{ maxRows: Number.POSITIVE_INFINITY, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },",
    );
    assert(!source.includes("Infinity,"), source);
    assertStringIncludes(source, 'provenance: { sweep: "a.json, b.json", sha256: "sha-a, sha-b"');
  });
});

describe("profileJson", () => {
  it("JSON.parse で同じ表に戻り、末尾の maxRows は Infinity のまま（null に落ちない）", () => {
    const profile = buildGeometryProfile(SPEC, SOURCES, deriveProfile(SOURCES, SPEC));
    const text = profileJson(profile);
    assertStringIncludes(text, '"maxRows": 1e999');
    assert(!text.includes("null"), text);
    const parsed = JSON.parse(text);
    assertEquals(parsed, profile);
    assertGeometryProfile(parsed);
  });

  it("既定の表も同じ形で書ける", () => {
    assertEquals(JSON.parse(profileJson(DEFAULT_GEOMETRY_PROFILE)), DEFAULT_GEOMETRY_PROFILE);
  });
});
