// The check of the per-bundle census table (`helpers/census-table.ts`) (needs neither a GPU nor
// weights — it reads only the manifests and the table).
//
// ① Consistency of the table itself: each row's bundle is made of seats the family accepts and holds
//   no reference value, and the counts it writes are only for the bundle's non-reference seats (and
//   the built-in fusions, which apply in every bundle).
// ② Where the key comes from: the table's key is determined by the **effective** SessionOptions
//   obtained by passing the manifest's `session` through the family's resolution (before
//   stateAttentionReduce was promoted, gemma added models' defaults outside the resolution, so a
//   table looked up by the manifest's literal text mistook which bundle actually ran — a regression
//   test for that hole).
// ③ Coverage of the distribution mirrors' bundles: every quant preset that has at least one
//   value-changing non-reference value has a row in the table (mechanically stops bundles without a
//   census from growing in the distribution — ADR 0110 decision 5 ①).
//
// On a machine without the mirrors only ③ is **explicitly SKIPped** with a reason (the same
// discipline as `e2e_session_options_mirror_test.ts` — the case where the mirrors are missing
// altogether is made a FAIL by `runtime/tests/distribution_gate_test.ts`).

import { isNotFound, readDir, readTextFile } from "@karume/test-support/fs";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { type Manifest, parseManifest, type SessionSpec } from "@karume/hub";
import { resolveGemmaSessionOptions } from "../src/gemma/session-options.ts";
import { censusSeats, SEAT_REFERENCE } from "../../runtime/tests/helpers/pipeline-census.ts";
import {
  type BundleCensusRow,
  bundleKey,
  CENSUS_TABLE,
  censusRowOf,
  effectiveSessionOptions,
  phaseExpectations,
  SESSION_POLICIES,
} from "./helpers/census-table.ts";

const MODELS_ROOT = new URL("../../../models/", import.meta.url);

/**
 * The seats a level of `numerics` resolves to (the rest are approximation options, each its own key
 * of the manifest vocabulary).
 */
const RESOLVED_SEATS: ReadonlySet<string> = new Set([
  "linearGemvReduce",
  "stateAttentionReduce",
  "linearGemvPlanes",
  "rmsNormReduce",
]);

/** Lists every (component, phase) of the rows. */
const phasesOf = (row: BundleCensusRow): readonly (readonly [string, string])[] =>
  Object.entries(row.census).flatMap(([component, phases]) =>
    Object.keys(phases ?? {}).map((phase) => [component, phase] as const),
  );

describe("consistency of the census table rows (no GPU, no assets)", () => {
  it("each row's bundle is made of seats the family accepts, holds no reference value, and is not the reference bundle (empty)", () => {
    for (const row of CENSUS_TABLE) {
      const where = `${row.family} [${bundleKey(row.seats)}]`;
      const policy: Readonly<Record<string, boolean>> = SESSION_POLICIES[row.family];
      for (const [seat, value] of Object.entries(row.seats)) {
        // A resolved form comes from the family's `numerics`, an approximation from its own key.
        const key = RESOLVED_SEATS.has(seat) ? "numerics" : seat;
        assert(
          Object.hasOwn(policy, key) && policy[key],
          `${where}: the family does not accept ${key} (seat ${seat})`,
        );
        assert(
          value !== (SEAT_REFERENCE as Readonly<Record<string, unknown>>)[seat],
          `${where}: seat ${seat} is written at its reference value`,
        );
      }
      assert(bundleKey(row.seats) !== "", `${where}: the reference bundle needs no census row`);
    }
  });

  it("(series, model, bundle) does not repeat", () => {
    const seen = new Set<string>();
    for (const row of CENSUS_TABLE) {
      for (const model of row.models) {
        const key = `${row.family} / ${model} / ${bundleKey(row.seats)}`;
        assert(!seen.has(key), `the census table has a duplicate row: ${key}`);
        seen.add(key);
      }
    }
  });

  it("each row has at least one phase, expects only the bundle's non-reference seats, and its counts are non-negative integers", () => {
    for (const row of CENSUS_TABLE) {
      const where = `${row.family} [${bundleKey(row.seats)}]`;
      const phases = phasesOf(row);
      assert(
        phases.length > 0,
        `${where}: no phase at all (if not yet derived, do not write the row)`,
      );
      for (const [component, phase] of phases) {
        const expected = phaseExpectations(row, component, phase);
        assert(expected !== undefined && expected.length > 0, `${where} ${component}/${phase}`);
        for (const one of expected) {
          assert(
            "count" in one && Number.isSafeInteger(one.count) && one.count >= 0,
            `${where} ${component}/${phase}: count of ${one.label}`,
          );
        }
      }
    }
  });

  it("a row that writes a count for a seat not in the bundle fails when the expectation is built", () => {
    const broken: BundleCensusRow = {
      family: "sbv2",
      models: ["F1"],
      seats: { linearCompute: "a8" },
      census: { front: { run: { linearGemvReduce: { variant: 1, reference: 0 } } } },
    };
    assertThrows(() => phaseExpectations(broken, "front", "run"), Error, "linearGemvReduce");
  });

  it("an unfilled phase returns no expectation (it does not turn into an assertion of 0)", () => {
    const row = censusRowOf("gemma4", "12b", { numerics: { level: "reorder" } });
    assert(row !== undefined);
    assertEquals(phaseExpectations(row, "target", "verify"), undefined);
  });
});

describe("the table's key is determined by the effective settings obtained by passing the manifest's session through the family's resolution", () => {
  it("gemma's declaration resolves to the same value the entry resolves, and a preset without a declaration is the reference bundle", () => {
    const where = "census: gemma4 e2b";
    const declared: SessionSpec = { numerics: { level: "reorder" } };
    const effective = effectiveSessionOptions("gemma4", declared, {}, where);
    // The resolution the table uses and the value gemma's entry (`resolveGemmaSessionOptions`)
    // passes to the Session are the same single path.
    assertEquals(effective, resolveGemmaSessionOptions(declared, {}, where));
    assertEquals(
      bundleKey(censusSeats(effective)),
      "linearGemvReduce=parallel,stateAttentionReduce=parallel-fused",
    );
    // A preset without a declaration (`i4`) is the reference bundle — if the family's code defaults
    // crept into the resolution this would not be empty.
    assertEquals(bundleKey(censusSeats(effectiveSessionOptions("gemma4", {}, {}, where))), "");
    // An explicit reference level beats the declaration for its op kind and drops out of the key
    // (i4-fast, with attention set back to exact, is a GEMV-only bundle).
    const reverted = effectiveSessionOptions(
      "gemma4",
      declared,
      {
        numerics: { ops: { attention: "exact" } },
      },
      where,
    );
    assertEquals(bundleKey(censusSeats(reverted)), "linearGemvReduce=parallel");
    assertEquals(censusRowOf("gemma4", "e2b", reverted)?.seats, { linearGemvReduce: "parallel" });
  });

  it("a bundle without a row and an unknown series cannot be looked up", () => {
    assertEquals(censusRowOf("irodori", "v4-small", { linearCompute: "f16" }), undefined);
    assertEquals(
      censusRowOf("gemma4", "e4b", { numerics: { ops: { linear: "reorder" } } }),
      undefined,
    );
    assertThrows(() => effectiveSessionOptions("unknown-family", {}), Error, "unknown-family");
  });
});

/**
 * Reads one mirror's manifest (`undefined` when it is missing).
 *
 * MUST: propagate everything except NotFound — rounding an I/O failure to "no mirror" would make a
 * check that never ran turn green under the guise of a SKIP.
 */
const readMirror = async (dir: URL): Promise<Manifest | undefined> => {
  try {
    return parseManifest(await readTextFile(new URL("karume.json", dir)));
  } catch (cause) {
    if (isNotFound(cause)) return undefined;
    throw cause;
  }
};

const listMirrors = async (): Promise<readonly (readonly [string, Manifest])[]> => {
  const mirrors: (readonly [string, Manifest])[] = [];
  try {
    for await (const entry of readDir(MODELS_ROOT)) {
      if (!entry.isDirectory) continue;
      const manifest = await readMirror(new URL(`${entry.name}/`, MODELS_ROOT));
      if (manifest !== undefined) mirrors.push([entry.name, manifest]);
    }
  } catch (cause) {
    if (!isNotFound(cause)) throw cause;
  }
  return mirrors.toSorted(([a], [b]) => a.localeCompare(b));
};

const MIRRORS = await listMirrors();
if (MIRRORS.length === 0) {
  console.warn(
    `[karume] ${MODELS_ROOT.pathname} has no karume.json of a distribution mirror, so ` +
      "the check that every distributed bundle has a row in the census table is SKIPped",
  );
}

describe.skipIf(MIRRORS.length === 0)(
  "the distribution mirrors' bundles have rows in the census table",
  () => {
    it("every quant preset with non-reference values has a row for its (series, model, effective SessionOptions)", () => {
      const missing: string[] = [];
      let bundles = 0;
      for (const [mirror, manifest] of MIRRORS) {
        for (const [modelName, entry] of Object.entries(manifest.models)) {
          const family = entry.pipeline.name;
          for (const [quantName, quant] of Object.entries(entry.quants)) {
            const where = `${mirror} ${modelName} '${quantName}'`;
            const effective = effectiveSessionOptions(family, quant.session, {}, where);
            const key = bundleKey(censusSeats(effective));
            if (key === "") continue;
            bundles++;
            if (censusRowOf(family, modelName, effective) === undefined) {
              missing.push(`${where}: ${family} [${key}]`);
            }
          }
        }
      }
      assertEquals(
        missing,
        [],
        "bundles without a row in the census table (helpers/census-table.ts)",
      );
      // Does not let through the shape that is green with zero items (mirrors exist but not a single
      // non-reference bundle was read).
      assert(bundles > 0, "mirrors exist but not a single non-reference bundle was checked");
    });
  },
);
