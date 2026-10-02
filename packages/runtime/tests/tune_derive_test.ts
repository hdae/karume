// 生成器の純関数（src/tune/derive.ts）のうち、注入の表（buildGeometryProfile）と注入の JSON
// （geometryProfileJson）の門。規則の抽出そのものは tune_profile_test.ts、生成物の TS の描画は
// tools/geometry-sweep/render_test.ts が固定する。GPU を使わない。

import {
  assert,
  assertEquals,
  assertInstanceOf,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { gemmGeometryForRows } from "../src/kernels/gemm-geometry.ts";
import {
  assertGeometryProfile,
  DEFAULT_GEOMETRY_PROFILE,
  selectGeometryProfile,
} from "../src/kernels/geometry-profile.ts";
import { buildGeometryProfile, deriveProfile, geometryProfileJson } from "../src/tune/derive.ts";
import {
  geometryProfileKernelsId,
  KernelsIdError,
  sweepCaseSetId,
} from "../src/tune/fingerprint.ts";
import { FAST, SOURCES, SPEC, verdictOf } from "./helpers/sweep-records.ts";

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

  it("match と provenance は spec と掃引の順で決まり、adapter は 4 欄をそのまま・指紋とケース集合の版は今の runtime の値（architecture を省けば match は vendor だけ）", () => {
    const verdicts = deriveProfile(SOURCES, SPEC);
    const profile = buildGeometryProfile(SPEC, SOURCES, verdicts);
    assertEquals(profile.id, "test-gpu");
    assertEquals(profile.match, { vendor: "apple", architecture: "metal-3" });
    const { provenance: _provenance, ...table } = profile;
    assertEquals(profile.provenance, {
      sweep: "a.json, b.json",
      sha256: "sha-a, sha-b",
      date: "2026-09-29T00:00:00.000Z, 2026-09-29T00:00:00.000Z",
      candidateSet: "full, full",
      adapter: { vendor: "apple", architecture: "metal-3", device: "", description: "Test GPU" },
      kernels: geometryProfileKernelsId(table),
      caseSet: sweepCaseSetId(),
    });
    // 指紋は表の幾何から導いた値（既定の表の指紋とは違う — 採用した幾何が入っている）
    assert(profile.provenance.kernels !== geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE));
    const { architecture: _, ...vendorOnly } = SPEC;
    assertEquals(buildGeometryProfile(vendorOnly, SOURCES, verdicts).match, { vendor: "apple" });
  });
});

describe("buildGeometryProfile: 指紋を導けない表", () => {
  it("採った幾何で掃引の shape の dispatch が 65535 を超える表は作らず、表・ケース・欄・幾何を名指して投げる", () => {
    // 上限の大きい device の掃引なら通りうる幾何（tileN 4 — conv2d 512² の N = 262144 が 65536 workgroup）
    const narrow = { regM: 1, regN: 4, wgX: 1, wgY: 16 };
    const verdicts = deriveProfile(SOURCES, SPEC).map((verdict) =>
      verdict.slot === "conv2d.rows32"
        ? { ...verdict, outcome: { ...verdict.outcome, geometry: narrow } }
        : verdict
    );
    const error = assertThrows(() => buildGeometryProfile(SPEC, SOURCES, verdicts), Error);
    assertInstanceOf(error.cause, KernelsIdError);
    for (
      const part of [
        "生成した表 'test-gpu' の provenance.kernels（カーネルの指紋）を導けない",
        "conv2d-c96-512x512",
        "conv2d.rows32",
        JSON.stringify(narrow),
        "65535",
      ]
    ) {
      assertStringIncludes(error.message, part);
    }
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
    assert(!("match" in JSON.parse(geometryProfileJson(profile))));
  });
});

describe("geometryProfileJson", () => {
  it("JSON.parse で同じ表に戻り、末尾の maxRows は Infinity のまま（null に落ちない）", () => {
    const profile = buildGeometryProfile(SPEC, SOURCES, deriveProfile(SOURCES, SPEC));
    const text = geometryProfileJson(profile);
    assertStringIncludes(text, '"maxRows": 1e999');
    assert(!text.includes("null"), text);
    const parsed = JSON.parse(text);
    assertEquals(parsed, profile);
    assertGeometryProfile(parsed);
  });

  it("既定の表も同じ形で書ける", () => {
    assertEquals(
      JSON.parse(geometryProfileJson(DEFAULT_GEOMETRY_PROFILE)),
      DEFAULT_GEOMETRY_PROFILE,
    );
  });
});
