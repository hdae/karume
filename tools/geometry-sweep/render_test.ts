// 生成物の TS とアプリ用の TS の描画（render.ts）の門 — 注入の表（runtime の buildGeometryProfile）と同じ値を
// 書くこと。表の値そのものは packages/runtime/tests/tune_derive_test.ts が固定する。GPU を使わない。

import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { buildGeometryProfile, deriveProfile } from "../../packages/runtime/src/tune/derive.ts";
import { SOURCES, SPEC } from "../../packages/runtime/tests/helpers/sweep-records.ts";
import { renderAppProfileSource, renderProfileSource } from "./render.ts";

describe("buildGeometryProfile", () => {
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
