// 保存した表の JSON の unknown 境界 parseGeometryProfileJson（src/tune/profile-json.ts — ADR 0117 決定 7）の門。
// 関数と失敗の型は公開面（../tune.ts）から引く。GPU を使わない。
//
// 固定するのは ① geometryProfileJson との往復で構造が一致する（末尾の Infinity を含む）② 未知の欄を全ての
// 階層で拒む ③ 型の誤り・欠落・末尾の null（Infinity の欠落）・runtime の門に落ちる表を、欄の path を名指す
// GeometryProfileParseError で拒む。

import {
  assert,
  assertEquals,
  assertInstanceOf,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  geometryProfileJson,
  GeometryProfileParseError,
  parseGeometryProfileJson,
} from "../tune.ts";
import { CodegenError } from "../src/codegen/errors.ts";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import { APPLE_METAL_3 } from "../src/kernels/geometry-profiles/apple-metal-3.ts";
import { NVIDIA_BLACKWELL } from "../src/kernels/geometry-profiles/nvidia-blackwell.ts";
import { buildGeometryProfile, deriveProfile, infinityJson } from "../src/tune/derive.ts";
import { SOURCES, SPEC } from "./helpers/sweep-records.ts";

const BASE = APPLE_METAL_3;

/** parse が GeometryProfileParseError で拒み、文言が `parts` を全て含むこと。 */
const rejects = (text: string, ...parts: readonly string[]): GeometryProfileParseError => {
  const error = assertThrows(() => parseGeometryProfileJson(text), GeometryProfileParseError);
  assertEquals(error.name, "GeometryProfileParseError");
  for (const part of parts) assertStringIncludes(error.message, part);
  return error;
};

/** 表の値を JSON にする（Infinity は 1e999 — 書き手と同じ綴りで、壊したい欄だけを壊す）。 */
const json = (value: unknown): string => infinityJson(value);

describe("parseGeometryProfileJson: geometryProfileJson との往復", () => {
  const optIn = buildGeometryProfile(
    { id: "opt-in", optIn: true },
    SOURCES,
    deriveProfile(SOURCES, { optIn: true, minSpeedup: 1.05 }),
  );
  const generated = buildGeometryProfile(SPEC, SOURCES, deriveProfile(SOURCES, SPEC));
  const cases: readonly (readonly [string, GeometryProfile])[] = [
    ["埋め込みの表（provenance 付き・description の match）", APPLE_METAL_3],
    ["埋め込みの表（vendor / architecture の match）", NVIDIA_BLACKWELL],
    ["既定の表（provenance 無し・空の match・3 段）", DEFAULT_GEOMETRY_PROFILE],
    ["生成器の表", generated],
    ["注入専用の表（match 無し）", optIn],
  ];
  it("i8a8.conv3d survives the round trip with its own value (not the default)", () => {
    const conv3d = { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 };
    const profile = { ...BASE, i8a8: { ...BASE.i8a8, conv3d } };
    assertEquals(parseGeometryProfileJson(geometryProfileJson(profile)).i8a8.conv3d, conv3d);
  });

  for (const [name, profile] of cases) {
    it(`${name}は同じ構造に戻り、末尾の maxRows は Infinity`, () => {
      const parsed = parseGeometryProfileJson(geometryProfileJson(profile));
      assertEquals(parsed, profile);
      assertEquals(parsed.gemmRows.at(-1)?.maxRows, Number.POSITIVE_INFINITY);
      assertEquals("match" in parsed, "match" in profile);
      assertEquals("provenance" in parsed, "provenance" in profile);
    });
  }
});

describe("parseGeometryProfileJson: 未知の欄は全ての階層で拒む", () => {
  const rows = (index: number, change: (rule: GeometryProfile["gemmRows"][number]) => unknown) =>
    BASE.gemmRows.map((rule, at) => (at === index ? change(rule) : rule));
  const provenance = BASE.provenance;
  assert(provenance !== undefined);
  const cases: readonly (readonly [string, unknown, string])[] = [
    ["top level", { ...BASE, version: 2 }, "version"],
    ["match", { ...BASE, match: { ...BASE.match, device: "0x0000" } }, "device"],
    ["gemmRows[2]", { ...BASE, gemmRows: rows(2, (rule) => ({ ...rule, note: "x" })) }, "note"],
    [
      "gemmRows[6].geometry",
      {
        ...BASE,
        gemmRows: rows(6, (rule) => ({ ...rule, geometry: { ...rule.geometry, tileK: 16 } })),
      },
      "tileK",
    ],
    ["attention", { ...BASE, attention: { ...BASE.attention, rows: 1 } }, "rows"],
    [
      "attention.pv",
      {
        ...BASE,
        attention: { ...BASE.attention, pv: { ...BASE.attention.pv, k: 1 } },
      },
      "k",
    ],
    ["conv2d", { ...BASE, conv2d: { ...BASE.conv2d, rows16: BASE.conv2d.rows32 } }, "rows16"],
    [
      "conv2d.rows32",
      { ...BASE, conv2d: { ...BASE.conv2d, rows32: { ...BASE.conv2d.rows32, tileK: 16 } } },
      "tileK",
    ],
    ["i8a8", { ...BASE, i8a8: { ...BASE.i8a8, w4a8: BASE.i8a8.linear } }, "w4a8"],
    [
      "i8a8.attentionPv",
      { ...BASE, i8a8: { ...BASE.i8a8, attentionPv: { ...BASE.i8a8.attentionPv, groupSize: 32 } } },
      "groupSize",
    ],
    ["provenance", { ...BASE, provenance: { ...provenance, runtime: "0.14.0" } }, "runtime"],
    [
      "provenance.adapter",
      {
        ...BASE,
        provenance: { ...provenance, adapter: { ...provenance.adapter, subgroupMinSize: 4 } },
      },
      "subgroupMinSize",
    ],
  ];
  for (const [path, value, key] of cases) {
    it(`${path} の未知の欄 ${key}`, () => {
      rejects(
        json(value),
        `${path} has unknown fields ${key}`,
        "a table of another format version is not read",
      );
    });
  }

  it("前の形の provenance（adapter が連結文字列・指紋とケース集合の版が無い）は欠落で拒む（移行しない）", () => {
    const { candidateSet: _c, kernels: _k, caseSet: _s, ...older } = provenance;
    rejects(
      json({ ...BASE, provenance: { ...older, adapter: "apple / metal-3 / Apple M2" } }),
      "provenance is missing fields candidateSet, kernels, caseSet",
    );
    rejects(
      json({ ...BASE, provenance: { ...provenance, adapter: "apple / metal-3 / Apple M2" } }),
      "provenance.adapter is not an object",
    );
  });
});

describe("parseGeometryProfileJson: 型の誤りと欠落", () => {
  const provenance = BASE.provenance;
  assert(provenance !== undefined);
  const cases: readonly (readonly [string, unknown, string])[] = [
    ["最上位が配列", [BASE], "top level is not an object"],
    ["id が数", { ...BASE, id: 3 }, "id is not a string (3)"],
    ["match.vendor が数", { ...BASE, match: { vendor: 1 } }, "match.vendor is not a string"],
    ["gemmRows が配列でない", { ...BASE, gemmRows: {} }, "gemmRows is not an array"],
    [
      "幾何の欄が小数",
      { ...BASE, attention: { ...BASE.attention, qk: { ...BASE.attention.qk, regM: 1.5 } } },
      "attention.qk.regM is not an integer (1.5)",
    ],
    [
      "幾何の欄が文字列",
      { ...BASE, i8a8: { ...BASE.i8a8, linear: { ...BASE.i8a8.linear, tileK: "16" } } },
      'i8a8.linear.tileK is not an integer ("16")',
    ],
    [
      "末尾以外の maxRows が文字列",
      {
        ...BASE,
        gemmRows: BASE.gemmRows.map((rule, at) => (at === 0 ? { ...rule, maxRows: "16" } : rule)),
      },
      "gemmRows[0].maxRows is not a number",
    ],
    [
      "provenance.kernels が数",
      { ...BASE, provenance: { ...provenance, kernels: 1 } },
      "provenance.kernels is not a string",
    ],
    [
      "provenance.userAgent が { deno } のまま（文字列化していない）",
      { ...BASE, provenance: { ...provenance, userAgent: { deno: "2.9.6" } } },
      'provenance.userAgent is not an array ({"deno":"2.9.6"})',
    ],
    [
      "provenance.userAgent が連結した文字列（配列でない）",
      { ...BASE, provenance: { ...provenance, userAgent: "Chrome/153, Chrome/154" } },
      'provenance.userAgent is not an array ("Chrome/153, Chrome/154")',
    ],
    [
      "provenance.userAgent の要素が文字列でない",
      { ...BASE, provenance: { ...provenance, userAgent: ["Chrome/154", { deno: "2.9.6" }] } },
      'provenance.userAgent[1] is not a string ({"deno":"2.9.6"})',
    ],
    [
      "provenance.userAgent が空の配列",
      { ...BASE, provenance: { ...provenance, userAgent: [] } },
      "provenance.userAgent is an empty array",
    ],
    [
      "provenance.adapter.device が null",
      { ...BASE, provenance: { ...provenance, adapter: { ...provenance.adapter, device: null } } },
      "provenance.adapter.device is not a string (null)",
    ],
    [
      "i8a8 の欄が無い",
      { ...BASE, i8a8: { linear: BASE.i8a8.linear } },
      "i8a8 is missing fields attentionQk, attentionPv",
    ],
    [
      "a table from before i8a8.conv3d (the field is required — ADR 0115 addendum 2026-10-26)",
      {
        ...BASE,
        i8a8: {
          linear: BASE.i8a8.linear,
          attentionQk: BASE.i8a8.attentionQk,
          attentionPv: BASE.i8a8.attentionPv,
        },
      },
      "i8a8 is missing fields conv3d",
    ],
    [
      "表の欄が無い",
      { id: "x", gemmRows: BASE.gemmRows },
      "top level is missing fields attention, conv2d, i8a8",
    ],
  ];
  for (const [name, value, message] of cases) {
    it(name, () => {
      rejects(json(value), message);
    });
  }

  it("JSON として読めない文字列は JSON.parse の理由を添えて拒む", () => {
    const error = rejects("{ id: ", "cannot be parsed as JSON");
    assertInstanceOf(error.cause, SyntaxError);
  });
});

describe("parseGeometryProfileJson: gemmRows の末尾の Infinity", () => {
  it("末尾の maxRows が null（素の JSON.stringify で書いた表）は Infinity の欠落と名指す", () => {
    rejects(
      JSON.stringify(BASE),
      "gemmRows[6].maxRows is null (a lost Infinity",
      "with geometryProfileJson",
    );
  });

  it("末尾の maxRows が有限の数なら拒む（どの行数にも当たる規則が無い）", () => {
    rejects(
      json({
        ...BASE,
        gemmRows: BASE.gemmRows.map((rule, at) => (at === 6 ? { ...rule, maxRows: 4096 } : rule)),
      }),
      "gemmRows[6].maxRows is not Infinity (4096",
    );
  });

  it("1e999 と書いた末尾は Infinity に戻る（geometryProfileJson の綴り）", () => {
    const text = geometryProfileJson(BASE);
    assertStringIncludes(text, '"maxRows": 1e999');
    assertEquals(parseGeometryProfileJson(text).gemmRows[6].maxRows, Number.POSITIVE_INFINITY);
  });
});

describe("parseGeometryProfileJson: 最後に runtime の注入の門（assertGeometryProfile）を通す", () => {
  const cases: readonly (readonly [string, unknown, string])[] = [
    [
      "gemmRows が狭義昇順でない",
      {
        ...BASE,
        gemmRows: BASE.gemmRows.map((rule, at) => (at === 1 ? { ...rule, maxRows: 16 } : rule)),
      },
      "is not strictly increasing",
    ],
    [
      "整除条件を満たさない幾何",
      { ...BASE, attention: { ...BASE.attention, qk: { regM: 3, regN: 4, wgX: 16, wgY: 16 } } },
      "attention.qk",
    ],
    [
      "description だけの match",
      { ...BASE, match: { description: "Apple M2" } },
      "description must be given together with both vendor and architecture",
    ],
    ["空の id", { ...BASE, id: "" }, "id is empty"],
    [
      "a conv3d geometry that breaks the divisibility conditions",
      { ...BASE, i8a8: { ...BASE.i8a8, conv3d: { regM: 8, regN: 6, wgX: 16, wgY: 8, tileK: 16 } } },
      "i8a8.conv3d",
    ],
  ];
  for (const [name, value, message] of cases) {
    it(`${name}は門の文言のまま GeometryProfileParseError で拒む（cause は門の CodegenError）`, () => {
      const error = rejects(json(value), message);
      assertInstanceOf(error.cause, CodegenError);
    });
  }
});
