/**
 * The header of a generated table: the op families derived now with one line per record (how the
 * record was made), the families carried over with the base's commit, no "Carried over" without a
 * base, and the same text for the same inputs (no date, machine or absolute path). No GPU.
 */

import { assert, assertEquals, assertThrows } from "@std/assert";
import type { GeometryProfile } from "@karume/runtime";
import { deriveGeometryProfile } from "@karume/runtime/tune";
import { describe, it } from "vite-plus/test";
import { REPOSITORY_ROOT } from "../format.ts";
import { browserOf, HeaderError, type HeaderInput, profileHeader } from "../header.ts";
import { loadRecord } from "../records.ts";
import { ADAPTER, bytesOf, sweepRecord, workbenchRecord } from "./helpers/records.ts";

const DATE = "2026-11-16T01:00:00.000Z";
const BASE_COMMIT = "fedcba9876543210fedcba9876543210fedcba98";

/** A derivation of one workbench record, and the header input around it (built afresh per call). */
const input = (options: { readonly carried?: boolean } = {}): HeaderInput => {
  const record = loadRecord(
    bytesOf(workbenchRecord("geometry-sweep", sweepRecord(DATE), { dirty: true })),
    "outputs/workbench/browser-test-gpu/geometry-sweep_a.json",
    "records",
  );
  const { profile, verdicts } = deriveGeometryProfile([record], {
    id: "test-gpu",
    vendor: ADAPTER.vendor,
    architecture: ADAPTER.architecture,
    minSpeedup: 1.05,
  });
  const provenance = profile.provenance;
  const gemm = provenance?.families.gemm;
  if (provenance === undefined || gemm === undefined) {
    throw new Error("a table derived from a linear case has gemm provenance");
  }
  // A carried family's entry cites the base's records, none of the records given now.
  const carried: GeometryProfile = {
    ...profile,
    provenance: {
      ...provenance,
      families: {
        ...provenance.families,
        conv2d: { ...gemm, sweep: "outputs/old.json", sha256: "c0ffee" },
      },
    },
  };
  return {
    profile: options.carried === true ? carried : profile,
    verdicts,
    records: [record],
    attentionBench: [],
    minSpeedup: 1.05,
    ...(options.carried === true
      ? {
          base: {
            commit: BASE_COMMIT,
            path: "packages/runtime/src/kernels/geometry-profiles/test-gpu.ts",
          },
        }
      : {}),
  };
};

describe("profileHeader", () => {
  it("names the table, its match and the families derived now, with one line per record", () => {
    const header = profileHeader(input());
    const lines = header.split("\n");
    assertEquals(lines[0], "/**");
    assertEquals(lines[1], " * Geometry profile `test-gpu` (**generated — do not edit by hand**).");
    assertEquals(header.includes("vendor testvendor, architecture testarch"), true);
    assertEquals(header.includes(" * Derived in this generation: gemm.\n"), true);
    const recordLine = lines.findIndex((line) =>
      line.startsWith(" * - outputs/workbench/browser-test-gpu/geometry-sweep_a.json (sha256 "),
    );
    assertEquals(lines[recordLine].endsWith(`, ${DATE})`), true);
    assertEquals(
      lines[recordLine + 1],
      " *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 0123456789abcdef0123456789abcdef01234567 (dirty)",
    );
    assertEquals(header.includes("at least ×1.050 faster"), true);
    assertEquals(header.endsWith(" */\n"), true);
  });

  it("states the families carried over and the commit of the base", () => {
    const header = profileHeader(input({ carried: true }));
    assertEquals(header.includes(" * Derived in this generation: gemm.\n"), true);
    assertEquals(
      header.includes(`Carried over from the registered table at commit ${BASE_COMMIT}: conv2d.`),
      true,
    );
    assertEquals(
      header.includes(
        `\`git show ${BASE_COMMIT}:packages/runtime/src/kernels/geometry-profiles/test-gpu.ts\``,
      ),
      true,
    );
  });

  it("has no Carried over section without a base", () => {
    assertEquals(profileHeader(input()).includes("Carried over"), false);
  });

  it("gives the same text for the same inputs, with no absolute path", () => {
    const header = profileHeader(input({ carried: true }));
    assertEquals(profileHeader(input({ carried: true })), header);
    assertEquals(header.includes(REPOSITORY_ROOT), false);
  });

  it("describes a bare record from what it carries", () => {
    const base = input();
    const bare = loadRecord(
      bytesOf({ ...sweepRecord(DATE), userAgent: "Node.js/24" }),
      "outputs/diag/bare.json",
      "records",
    );
    const header = profileHeader({ ...base, records: [bare] });
    assertEquals(
      header.includes(
        " *   bare record; Node.js/24; timestamps not quantised; checkout not recorded\n",
      ),
      true,
    );
  });

  it("keeps a verdict that quotes a multi-line error on one comment line", () => {
    const header = profileHeader({
      ...input(),
      verdicts: [
        "- field: failed in case-a (Error: device lost: timeout\n - While handling it\n    at f (x.mm:1)\n)",
      ],
    });
    for (const line of header.split("\n").slice(1, -2)) {
      assert(
        line === " *" || line.startsWith(" * "),
        `a header line left the comment prefix: ${line}`,
      );
    }
    assert(
      header.includes(
        "failed in case-a (Error: device lost: timeout - While handling it at f (x.mm:1) )",
      ),
    );
  });

  it("refuses a line that would close the comment", () => {
    const base = input();
    assertThrows(
      () => profileHeader({ ...base, verdicts: ["- a */ b"] }),
      HeaderError,
      "close the comment",
    );
  });
});

describe("browserOf", () => {
  it("names the browser and the system of a user agent", () => {
    assertEquals(
      browserOf(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Safari/605.1.15",
      ),
      "Safari 26 on macOS",
    );
    assertEquals(
      browserOf("Mozilla/5.0 (X11; Linux x86_64; rv:145.0) Gecko/20100101 Firefox/145.0"),
      "Firefox 145 on Linux",
    );
  });
});
