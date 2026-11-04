// The check of parseGeometryProfileJson, the unknown boundary of a saved table's JSON
// (src/tune/profile-json.ts — ADR 0117 decision 7). The function and the failure type come from the
// public surface (../tune.ts). No GPU is used.
//
// What this pins: ① the structure survives a round trip with geometryProfileJson (the trailing
// Infinity included) ② unknown fields are rejected at every level ③ type errors, missing fields, a
// trailing null (a lost Infinity) and tables that fail the runtime's check are rejected with a
// GeometryProfileParseError that names the field's path.

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

/**
 * Asserts that parsing rejects with a GeometryProfileParseError whose message contains every one of
 * `parts`.
 */
const rejects = (text: string, ...parts: readonly string[]): GeometryProfileParseError => {
  const error = assertThrows(() => parseGeometryProfileJson(text), GeometryProfileParseError);
  assertEquals(error.name, "GeometryProfileParseError");
  for (const part of parts) assertStringIncludes(error.message, part);
  return error;
};

/**
 * Turns a table value into JSON (Infinity as 1e999 — the writer's spelling, so only the field meant
 * to be broken is broken).
 */
const json = (value: unknown): string => infinityJson(value);

describe("parseGeometryProfileJson: round trip with geometryProfileJson", () => {
  const optIn = buildGeometryProfile(
    { id: "opt-in", optIn: true },
    SOURCES,
    deriveProfile(SOURCES, { optIn: true, minSpeedup: 1.05 }),
  );
  const generated = buildGeometryProfile(SPEC, SOURCES, deriveProfile(SOURCES, SPEC));
  const cases: readonly (readonly [string, GeometryProfile])[] = [
    ["a built-in table (with provenance, description match)", APPLE_METAL_3],
    ["a built-in table (vendor / architecture match)", NVIDIA_BLACKWELL],
    ["the default table (no provenance, empty match, 3 segments)", DEFAULT_GEOMETRY_PROFILE],
    ["a generated table", generated],
    ["an injection-only table (no match)", optIn],
  ];
  it("i8a8.conv3d survives the round trip with its own value (not the default)", () => {
    const conv3d = { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 };
    const profile = { ...BASE, i8a8: { ...BASE.i8a8, conv3d } };
    assertEquals(parseGeometryProfileJson(geometryProfileJson(profile)).i8a8.conv3d, conv3d);
  });

  for (const [name, profile] of cases) {
    it(`${name} comes back with the same structure, and the trailing maxRows is Infinity`, () => {
      const parsed = parseGeometryProfileJson(geometryProfileJson(profile));
      assertEquals(parsed, profile);
      assertEquals(parsed.gemmRows.at(-1)?.maxRows, Number.POSITIVE_INFINITY);
      assertEquals("match" in parsed, "match" in profile);
      assertEquals("provenance" in parsed, "provenance" in profile);
    });
  }
});

describe("parseGeometryProfileJson: unknown fields are rejected at every level", () => {
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
    it(`${path}: unknown field ${key}`, () => {
      rejects(
        json(value),
        `${path} has unknown fields ${key}`,
        "a table of another format version is not read",
      );
    });
  }

  it("rejects a provenance of the earlier form (adapter as a joined string, no fingerprint or case-set version) as missing fields (no migration)", () => {
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

describe("parseGeometryProfileJson: type errors and missing fields", () => {
  const provenance = BASE.provenance;
  assert(provenance !== undefined);
  const cases: readonly (readonly [string, unknown, string])[] = [
    ["the top level is an array", [BASE], "top level is not an object"],
    ["id is a number", { ...BASE, id: 3 }, "id is not a string (3)"],
    ["match.vendor is a number", { ...BASE, match: { vendor: 1 } }, "match.vendor is not a string"],
    ["gemmRows is not an array", { ...BASE, gemmRows: {} }, "gemmRows is not an array"],
    [
      "a geometry field is fractional",
      { ...BASE, attention: { ...BASE.attention, qk: { ...BASE.attention.qk, regM: 1.5 } } },
      "attention.qk.regM is not an integer (1.5)",
    ],
    [
      "a geometry field is a string",
      { ...BASE, i8a8: { ...BASE.i8a8, linear: { ...BASE.i8a8.linear, tileK: "16" } } },
      'i8a8.linear.tileK is not an integer ("16")',
    ],
    [
      "a non-trailing maxRows is a string",
      {
        ...BASE,
        gemmRows: BASE.gemmRows.map((rule, at) => (at === 0 ? { ...rule, maxRows: "16" } : rule)),
      },
      "gemmRows[0].maxRows is not a number",
    ],
    [
      "provenance.kernels is a number",
      { ...BASE, provenance: { ...provenance, kernels: 1 } },
      "provenance.kernels is not a string",
    ],
    [
      "provenance.userAgent is still { deno } (not turned into a string)",
      { ...BASE, provenance: { ...provenance, userAgent: { deno: "2.9.6" } } },
      'provenance.userAgent is not an array ({"deno":"2.9.6"})',
    ],
    [
      "provenance.userAgent is a joined string (not an array)",
      { ...BASE, provenance: { ...provenance, userAgent: "Chrome/153, Chrome/154" } },
      'provenance.userAgent is not an array ("Chrome/153, Chrome/154")',
    ],
    [
      "an element of provenance.userAgent is not a string",
      { ...BASE, provenance: { ...provenance, userAgent: ["Chrome/154", { deno: "2.9.6" }] } },
      'provenance.userAgent[1] is not a string ({"deno":"2.9.6"})',
    ],
    [
      "provenance.userAgent is an empty array",
      { ...BASE, provenance: { ...provenance, userAgent: [] } },
      "provenance.userAgent is an empty array",
    ],
    [
      "provenance.adapter.device is null",
      { ...BASE, provenance: { ...provenance, adapter: { ...provenance.adapter, device: null } } },
      "provenance.adapter.device is not a string (null)",
    ],
    [
      "i8a8 fields are missing",
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
      "table fields are missing",
      { id: "x", gemmRows: BASE.gemmRows },
      "top level is missing fields attention, conv2d, i8a8",
    ],
  ];
  for (const [name, value, message] of cases) {
    it(name, () => {
      rejects(json(value), message);
    });
  }

  it("rejects a string that cannot be read as JSON, with JSON.parse's reason attached", () => {
    const error = rejects("{ id: ", "cannot be parsed as JSON");
    assertInstanceOf(error.cause, SyntaxError);
  });
});

describe("parseGeometryProfileJson: the trailing Infinity of gemmRows", () => {
  it("names a trailing maxRows of null (a table written with plain JSON.stringify) as a lost Infinity", () => {
    rejects(
      JSON.stringify(BASE),
      "gemmRows[6].maxRows is null (a lost Infinity",
      "with geometryProfileJson",
    );
  });

  it("rejects a finite trailing maxRows (no rule would match every row count)", () => {
    rejects(
      json({
        ...BASE,
        gemmRows: BASE.gemmRows.map((rule, at) => (at === 6 ? { ...rule, maxRows: 4096 } : rule)),
      }),
      "gemmRows[6].maxRows is not Infinity (4096",
    );
  });

  it("a trailing value written as 1e999 comes back as Infinity (geometryProfileJson's spelling)", () => {
    const text = geometryProfileJson(BASE);
    assertStringIncludes(text, '"maxRows": 1e999');
    assertEquals(parseGeometryProfileJson(text).gemmRows[6].maxRows, Number.POSITIVE_INFINITY);
  });
});

describe("parseGeometryProfileJson: finally runs the runtime's injection check (assertGeometryProfile)", () => {
  const cases: readonly (readonly [string, unknown, string])[] = [
    [
      "gemmRows is not strictly ascending",
      {
        ...BASE,
        gemmRows: BASE.gemmRows.map((rule, at) => (at === 1 ? { ...rule, maxRows: 16 } : rule)),
      },
      "is not strictly increasing",
    ],
    [
      "a geometry that fails the divisibility conditions",
      { ...BASE, attention: { ...BASE.attention, qk: { regM: 3, regN: 4, wgX: 16, wgY: 16 } } },
      "attention.qk",
    ],
    [
      "a match with description only",
      { ...BASE, match: { description: "Apple M2" } },
      "description must be given together with both vendor and architecture",
    ],
    ["an empty id", { ...BASE, id: "" }, "id is empty"],
    [
      "a conv3d geometry that breaks the divisibility conditions",
      { ...BASE, i8a8: { ...BASE.i8a8, conv3d: { regM: 8, regN: 6, wgX: 16, wgY: 8, tileK: 16 } } },
      "i8a8.conv3d",
    ],
  ];
  for (const [name, value, message] of cases) {
    it(`${name} is rejected with a GeometryProfileParseError carrying the check's wording (cause is the check's CodegenError)`, () => {
      const error = rejects(json(value), message);
      assertInstanceOf(error.cause, CodegenError);
    });
  }
});
