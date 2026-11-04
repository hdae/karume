// Reading and writing the per-environment reference fixtures (the models-side e2e uses
// `packages/models/tests/fixtures/references/*.json`, the runtime-side e2e uses
// `packages/runtime/tests/fixtures/references/*.json`).
//
// What this file binds is **the semantics of the modes** themselves: unset only compares, `write`
// only adds missing rows, and `rewrite` overwrites only the current environment's rows. If any of
// the three loosens, the check turns into either "green by comparing against another machine's
// reference" or "green by silently re-recording a broken value".
//
// Needs neither a real GPU nor environment variables — the environment and the mode are injected as
// arguments.

import {
  makeTempDirSync,
  readFile,
  readTextFileSync,
  removeSync,
  writeTextFileSync,
} from "@karume/test-support/fs";
import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import type { Environment } from "./helpers/environment.ts";
import { parseSafetensors } from "../mod.ts";
import {
  f32ArtifactBytes,
  openReferences,
  parseReferenceMode,
  referenceEntryFields,
  referenceGateFindings,
  referenceGatePasses,
  type ReferenceHolds,
  referenceMismatchMessage,
  type ReferenceSettlement,
  settleOrObserve,
  settleReference,
  sha256Hex,
} from "./helpers/reference.ts";
import { openResults, type Results } from "./helpers/results.ts";

const CURRENT = "deno-intel-graphics-bmg-g21";
const OTHER = "deno-nvidia-geforce-rtx-3080-ti";

const environment = (key: string): Environment => ({
  key,
  runtime: { name: "deno", version: "2.9.6", v8: "15.0.245.2", typescript: "6.0.3" },
  adapter: { vendor: "32902", architecture: "", device: "57868", description: "test" },
  os: { platform: "linux", arch: "x86_64" },
});

/** Creates, in a temporary directory, a fixture that holds a single row of another environment. */
const withFixture = (
  cases: Record<string, Record<string, string>>,
  body: (fixtureUrl: URL) => void,
): void => {
  const dir = makeTempDirSync({ prefix: "karume-reference-" });
  try {
    const fixtureUrl = new URL("references.json", `file://${dir}/`);
    writeTextFileSync(
      fixtureUrl,
      `${JSON.stringify({ schema: 1, kind: "sha256", cases }, undefined, 2)}\n`,
    );
    body(fixtureUrl);
  } finally {
    removeSync(dir, { recursive: true });
  }
};

const SHA_A = "a".repeat(64);
const SHA_B = "b".repeat(64);

test("reference fixture: lookup returns only the current environment's rows", () => {
  withFixture({ "case-1": { [OTHER]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    assertEquals(
      references.lookup("case-1"),
      undefined,
      "another environment's row is visible as the current one",
    );
    assertEquals(referenceGatePasses(references, ["case-1"]), false);
    assertEquals(references.lacksReference("case-1"), true);
    const other = openReferences(fixtureUrl, { environment: environment(OTHER), mode: undefined });
    assertEquals(other.lookup("case-1"), SHA_A);
    assertEquals(referenceGatePasses(other, ["case-1"]), true);
    assertEquals(other.lacksReference("case-1"), false);
  });
});

test("reference fixture: write adds missing rows and does not overwrite existing ones", () => {
  withFixture({ "case-1": { [OTHER]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: "write",
    });
    // No row ⇒ add the measurement and pass.
    assertEquals(references.check("case-1", SHA_B), { status: "written" });
    assertEquals(references.lookup("case-1"), SHA_B);
    // A row exists ⇒ compare as usual (write does not overwrite either).
    assertEquals(references.check("case-1", SHA_B), { status: "pass", expected: SHA_B });
    assertEquals(references.check("case-1", SHA_A), { status: "fail", expected: SHA_B });
    assertEquals(references.lookup("case-1"), SHA_B, "write mode rewrote an existing row");
    // What was written back is the same on the next read (another environment's rows stay alive).
    const reopened = openReferences(fixtureUrl, {
      environment: environment(OTHER),
      mode: undefined,
    });
    assertEquals(reopened.lookup("case-1"), SHA_A, "write moved another environment's row");
  });
});

test("reference fixture: rewrite overwrites only the current environment's rows", () => {
  withFixture({ "case-1": { [CURRENT]: SHA_A, [OTHER]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: "rewrite",
    });
    assertEquals(references.check("case-1", SHA_B), { status: "rewritten", previous: SHA_A });
    assertEquals(references.lookup("case-1"), SHA_B);
    const other = openReferences(fixtureUrl, { environment: environment(OTHER), mode: undefined });
    assertEquals(other.lookup("case-1"), SHA_A, "rewrite dragged in another environment's row");
  });
});

test("reference fixture: opening two handles first and writing in turn keeps both environments' rows", () => {
  withFixture({}, (fixtureUrl) => {
    // Open both **before writing** (writing back the copy taken at open time would make the later
    // writer erase the earlier rows).
    const current = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: "write",
    });
    const other = openReferences(fixtureUrl, { environment: environment(OTHER), mode: "write" });
    current.check("case-1", SHA_A);
    other.check("case-1", SHA_B);
    const document = JSON.parse(readTextFileSync(fixtureUrl)) as {
      cases: Record<string, Record<string, string>>;
    };
    assertEquals(
      document.cases["case-1"],
      { [CURRENT]: SHA_A, [OTHER]: SHA_B },
      "the environment written later dragged in the row another environment wrote earlier",
    );
  });
});

test("reference fixture: even with two handles of the same environment key, the later write does not overwrite the earlier row", () => {
  withFixture({}, (fixtureUrl) => {
    // Open both **before writing**. Deciding from the copy taken at open time, the later one would
    // see "no row" and overwrite the earlier row (the semantics of `write` would turn into
    // rewrite).
    const first = openReferences(fixtureUrl, { environment: environment(CURRENT), mode: "write" });
    const second = openReferences(fixtureUrl, { environment: environment(CURRENT), mode: "write" });
    assertEquals(first.check("case-1", SHA_A), { status: "written" });
    // The later one falls to comparing against the latest row on disk (red for a different
    // measurement).
    assertEquals(second.check("case-1", SHA_B), { status: "fail", expected: SHA_A });
    const document = JSON.parse(readTextFileSync(fixtureUrl)) as {
      cases: Record<string, Record<string, string>>;
    };
    assertEquals(
      document.cases["case-1"],
      { [CURRENT]: SHA_A },
      "the later write overwrote the earlier row",
    );
    // The other side: the same measurement passes (pins that this is not passing by "always red").
    assertEquals(second.check("case-1", SHA_A), { status: "pass", expected: SHA_A });
  });
});

// --- When the reference check passes (counted over the registered set of cases) ----------
//
// What the check guards against is a silent pass for "the sha check of a case with no reference for
// this environment was skipped". Counting across the whole fixture lets orphan rows left by renames
// and removals hide that state.

test("reference check: does not pass when only orphan rows hold the current environment's rows", () => {
  // `retired` is an unregistered case (left over from a rename or removal). The two live cases have
  // no rows.
  withFixture({ retired: { [CURRENT]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    assertEquals(
      referenceGatePasses(references, ["case-1", "case-2"]),
      false,
      "orphan rows hid that every live case was skipped",
    );
  });
});

test("reference check: does not pass when no live case has a row (the recording mode is separate)", () => {
  withFixture({ "case-1": { [OTHER]: SHA_A } }, (fixtureUrl) => {
    const cases = ["case-1", "case-2"];
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    assertEquals(referenceGatePasses(references, cases), false);
    // A run that records references naturally has no rows, so it passes (ADR 0106).
    const writing = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: "write",
    });
    assertEquals(referenceGatePasses(writing, cases), true);
  });
});

// With "pass if any one case has a row", a forgotten row of a newly added case would pass as a
// skip with a warning. The check requires a row for every registered case, or an explicit held row
// for a case without one.

test("reference check: red if even one registered case has no row for the current environment (names the missing case)", () => {
  withFixture({ "case-1": { [CURRENT]: SHA_A }, "case-2": { [OTHER]: SHA_B } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    const cases = ["case-1", "case-2"];
    assertEquals(referenceGatePasses(references, cases), false, "a forgotten row passed");
    assertEquals(referenceGateFindings(references, cases), { missing: ["case-2"], staleHolds: [] });
  });
});

test("reference check: passes when every registered case has a row for the current environment", () => {
  withFixture({ "case-1": { [CURRENT]: SHA_A }, "case-2": { [CURRENT]: SHA_B } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    assertEquals(referenceGatePasses(references, ["case-1", "case-2"]), true);
  });
});

test("reference check: passes when a case without a row has a held row for the current environment; another environment's held row does not count", () => {
  withFixture({ "case-1": { [CURRENT]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    const cases = ["case-1", "case-2"];
    const heldHere: ReferenceHolds = { "case-2": { [CURRENT]: "device lost on this machine" } };
    assertEquals(referenceGatePasses(references, cases, heldHere), true);
    // Another machine's held row does not excuse the gap on this machine (holding it on every
    // machine would also remove verification on machines that can run it).
    const heldElsewhere: ReferenceHolds = { "case-2": { [OTHER]: "device lost on that machine" } };
    assertEquals(referenceGatePasses(references, cases, heldElsewhere), false);
    assertEquals(referenceGateFindings(references, cases, heldElsewhere).missing, ["case-2"]);
  });
});

test("reference check: red for a case with both a held row and a current-environment row (names the stale held row)", () => {
  withFixture({ "case-1": { [CURRENT]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    const holds: ReferenceHolds = { "case-1": { [CURRENT]: "should be resolved" } };
    assertEquals(referenceGatePasses(references, ["case-1"], holds), false);
    assertEquals(referenceGateFindings(references, ["case-1"], holds), {
      missing: [],
      staleHolds: ["case-1"],
    });
  });
});

test("reference check: an opt-in case is not counted in a run that leaves it out of the registration", () => {
  // `opt-in` is a case enabled by an environment variable (its row exists only on another
  // machine). In a run where it is disabled the caller does not register it (the conditional
  // registration of CASE_IDS in e2e_wan_pipeline_test.ts) — the check counts only the registered
  // set, not the whole fixture.
  withFixture({ "case-1": { [CURRENT]: SHA_A }, "opt-in": { [OTHER]: SHA_B } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    assertEquals(referenceGatePasses(references, ["case-1"]), true, "counted a disabled opt-in");
    assertEquals(referenceGatePasses(references, ["case-1", "opt-in"]), false);
  });
});

test("reference check: red when a check that can run has zero registered cases", () => {
  withFixture({ "case-1": { [CURRENT]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    assertEquals(referenceGatePasses(references, []), false);
  });
});

test("reference fixture: matching a case without a row in the unset mode throws", () => {
  withFixture({}, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, {
      environment: environment(CURRENT),
      mode: undefined,
    });
    assertThrows(() => references.check("case-1", SHA_A), Error, "KARUME_REFERENCE=write");
  });
});

test("reference fixture: output is stable in lexicographic order (written twice, byte-identical)", () => {
  withFixture({ "case-2": { [OTHER]: SHA_A }, "case-1": { [OTHER]: SHA_A } }, (fixtureUrl) => {
    openReferences(fixtureUrl, { environment: environment(CURRENT), mode: "write" }).check(
      "case-2",
      SHA_B,
    );
    const first = readTextFileSync(fixtureUrl);
    openReferences(fixtureUrl, { environment: environment(CURRENT), mode: "rewrite" }).check(
      "case-2",
      SHA_B,
    );
    const second = readTextFileSync(fixtureUrl);
    assertEquals(second, first, "rewriting the same content moved bytes");
    // JSON keys read back in insertion order, so checking the parsed key list is enough to check
    // the order.
    const document = JSON.parse(first) as { cases: Record<string, Record<string, string>> };
    assertEquals(
      Object.keys(document.cases),
      ["case-1", "case-2"],
      "case names are not in lexicographic order",
    );
    assertEquals(
      Object.keys(document.cases["case-2"]),
      [CURRENT, OTHER].sort(),
      "environment keys are not in lexicographic order",
    );
    assert(first.endsWith("}\n"), "no trailing newline");
  });
});

test("reference fixture: an unreadable fixture throws instead of being treated as empty", () => {
  withFixture({}, (fixtureUrl) => {
    assertThrows(
      () =>
        openReferences(new URL("missing.json", fixtureUrl), {
          environment: environment(CURRENT),
          mode: undefined,
        }),
      Error,
      "fixture cannot be read",
    );
    writeTextFileSync(fixtureUrl, "{ broken");
    assertThrows(
      () => openReferences(fixtureUrl, { environment: environment(CURRENT), mode: undefined }),
      Error,
      "is broken as JSON",
    );
    writeTextFileSync(fixtureUrl, JSON.stringify({ schema: 2, kind: "sha256", cases: {} }));
    assertThrows(
      () => openReferences(fixtureUrl, { environment: environment(CURRENT), mode: undefined }),
      Error,
      "schema of the reference-value fixture is 2",
    );
  });
});

test("reference fixture: an unknown spelling of KARUME_REFERENCE throws", () => {
  assertEquals(parseReferenceMode(undefined), undefined);
  assertEquals(parseReferenceMode("write"), "write");
  assertEquals(parseReferenceMode("rewrite"), "rewrite");
  assertThrows(() => parseReferenceMode("1"), Error, "KARUME_REFERENCE");
  assertThrows(() => parseReferenceMode(""), Error, "KARUME_REFERENCE");
});

// --- Settling a reference (settleReference, and copying its result into results and diagnostics)
// ---
//
// What this binds is threefold: "the artifact stays in the results location whether or not the
// match succeeds", "the sha is taken from the artifact's bytes", and "the copy of the settlement
// does not mistake whether expected is present". If any breaks, the material for an A/B of a
// mismatching run disappears, or the results JSON misleads readers into thinking a run that
// recorded a reference "was matched against something".

/**
 * Puts the fixture and the results location in the same temporary directory (needs neither a real
 * GPU nor outputs/).
 */
const withSeats = async (
  cases: Record<string, Record<string, string>>,
  body: (fixtureUrl: URL, results: Results) => Promise<void>,
): Promise<void> => {
  const dir = makeTempDirSync({ prefix: "karume-reference-settle-" });
  try {
    const base = new URL(`file://${dir}/`);
    const fixtureUrl = new URL("references.json", base);
    writeTextFileSync(
      fixtureUrl,
      `${JSON.stringify({ schema: 1, kind: "sha256", cases }, undefined, 2)}\n`,
    );
    const results = openResults("settle", {
      root: new URL("verify/", base),
      environment: environment(CURRENT),
    });
    await body(fixtureUrl, results);
  } finally {
    removeSync(dir, { recursive: true });
  }
};

const BYTES = new TextEncoder().encode("karume-artifact");

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

/** Whether this has the shape of a fixture document's `cases` (case → environment key → sha). */
const isCasesDocument = (
  value: unknown,
): value is { readonly cases: Record<string, Record<string, string>> } =>
  isRecord(value) &&
  isRecord(value.cases) &&
  Object.values(value.cases).every(
    (rows) => isRecord(rows) && Object.values(rows).every((sha) => typeof sha === "string"),
  );

/**
 * Reads the `cases` of the fixture on disk (narrowed by a type guard — fails if the shape is
 * broken).
 */
const readCases = (fixtureUrl: URL): Record<string, Record<string, string>> => {
  const document: unknown = JSON.parse(readTextFileSync(fixtureUrl));
  assert(isCasesDocument(document), `${fixtureUrl.pathname} does not have the shape { cases }`);
  return document.cases;
};

describe("sha256Hex", () => {
  it("returns 64 lowercase hex digits matching the known FIPS 180-2 vector ('abc')", async () => {
    assertEquals(
      await sha256Hex(new TextEncoder().encode("abc")),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });
});

describe("settleReference", () => {
  it("in write mode without a row, writes the measurement as the row and keeps the artifact in the results location", async () => {
    await withSeats({}, async (fixtureUrl, results) => {
      const references = openReferences(fixtureUrl, {
        environment: environment(CURRENT),
        mode: "write",
      });
      const settlement = await settleReference(references, results, {
        id: "case-1",
        artifact: "case-1.png",
        bytes: BYTES,
      });
      const expectedSha = await sha256Hex(BYTES);
      assertEquals(settlement.check, { status: "written" });
      assertEquals(settlement.sha256, expectedSha);
      assertEquals(settlement.artifactUrl.href, new URL("case-1.png", results.dir).href);
      // The sha matches the bytes of the artifact file (the result's actual and the artifact do not
      // disagree).
      const onDisk = await readFile(settlement.artifactUrl);
      assertEquals(onDisk, BYTES);
      assertEquals(await sha256Hex(onDisk), settlement.sha256);
      assertEquals(readCases(fixtureUrl)["case-1"], { [CURRENT]: expectedSha });
    });
  });

  it("in compare mode, a mismatch with the row returns fail, writes the artifact, and leaves the row unchanged", async () => {
    await withSeats({ "case-1": { [CURRENT]: SHA_A } }, async (fixtureUrl, results) => {
      const before = readTextFileSync(fixtureUrl);
      const references = openReferences(fixtureUrl, {
        environment: environment(CURRENT),
        mode: undefined,
      });
      const settlement = await settleReference(references, results, {
        id: "case-1",
        artifact: "case-1.safetensors",
        bytes: BYTES,
      });
      assertEquals(settlement.check, { status: "fail", expected: SHA_A });
      assertEquals(
        await readFile(settlement.artifactUrl),
        BYTES,
        "the artifact was not kept on the mismatching run",
      );
      assertEquals(readTextFileSync(fixtureUrl), before, "compare mode rewrote the reference");
    });
  });

  it("in compare mode, a match returns pass", async () => {
    const sha = await sha256Hex(BYTES);
    await withSeats({ "case-1": { [CURRENT]: sha } }, async (fixtureUrl, results) => {
      const references = openReferences(fixtureUrl, {
        environment: environment(CURRENT),
        mode: undefined,
      });
      const settlement = await settleReference(references, results, {
        id: "case-1",
        artifact: "case-1.png",
        bytes: BYTES,
      });
      assertEquals(settlement.check, { status: "pass", expected: sha });
    });
  });
});

// The end point of the sha as an additional check. This function's job is to give a single shape
// to the settlement of a case without a row; if it breaks, "suites that record the measured sha and
// suites that record nothing" mix again (the cross-environment comparison of actual would work only
// in some suites).
describe("settleOrObserve", () => {
  it("in compare mode without a row, skips the match, writes the artifact and returns pass + actual + artifact (no expected)", async () => {
    await withSeats({ "case-1": { [OTHER]: SHA_A } }, async (fixtureUrl, results) => {
      const before = readTextFileSync(fixtureUrl);
      const references = openReferences(fixtureUrl, {
        environment: environment(CURRENT),
        mode: undefined,
      });
      const outcome = await settleOrObserve(references, results, {
        id: "case-1",
        artifact: "case-1.safetensors",
        bytes: BYTES,
      });
      assertEquals(outcome.settlement, undefined);
      assertEquals(outcome.fields, {
        status: "pass",
        actual: await sha256Hex(BYTES),
        artifact: "case-1.safetensors",
      });
      assertEquals(Object.keys(outcome.fields), ["status", "actual", "artifact"]);
      assertEquals(
        await readFile(new URL("case-1.safetensors", results.dir)),
        BYTES,
        "the artifact was not kept on the run that skipped the match",
      );
      // It neither matches against another environment's row (OTHER) nor creates a row for this
      // environment.
      assertEquals(readTextFileSync(fixtureUrl), before, "compare mode rewrote the reference");
    });
  });

  it("in compare mode with a row, matches against it and returns the settlement and the fields of referenceEntryFields", async () => {
    await withSeats({ "case-1": { [CURRENT]: SHA_A } }, async (fixtureUrl, results) => {
      const references = openReferences(fixtureUrl, {
        environment: environment(CURRENT),
        mode: undefined,
      });
      const outcome = await settleOrObserve(references, results, {
        id: "case-1",
        artifact: "case-1.safetensors",
        bytes: BYTES,
      });
      assertEquals(outcome.settlement?.check, { status: "fail", expected: SHA_A });
      assertEquals(outcome.fields, {
        status: "fail",
        expected: SHA_A,
        actual: await sha256Hex(BYTES),
        artifact: "case-1.safetensors",
      });
    });
  });

  it("in write mode without a row, creates the row and returns the written fields", async () => {
    await withSeats({}, async (fixtureUrl, results) => {
      const references = openReferences(fixtureUrl, {
        environment: environment(CURRENT),
        mode: "write",
      });
      const outcome = await settleOrObserve(references, results, {
        id: "case-1",
        artifact: "case-1.safetensors",
        bytes: BYTES,
      });
      const sha = await sha256Hex(BYTES);
      assertEquals(outcome.settlement?.check, { status: "written" });
      assertEquals(outcome.fields, {
        status: "written",
        actual: sha,
        artifact: "case-1.safetensors",
      });
      assertEquals(readCases(fixtureUrl)["case-1"], { [CURRENT]: sha });
    });
  });
});

describe("f32ArtifactBytes", () => {
  const data = Float32Array.from([0.5, -1, 2.25, Number.MIN_VALUE, -0, 3]);

  it("holds the tensor name, F32, the shape and the bit pattern of the values as is, with no metadata", () => {
    const bytes = f32ArtifactBytes("logits", [1, 2, 3], data);
    const parsed = parseSafetensors(bytes.buffer);
    assertEquals([...parsed.tensors.keys()], ["logits"]);
    assertEquals(parsed.metadata.size, 0);
    const view = parsed.tensors.get("logits");
    assert(view !== undefined);
    assertEquals(view.dtype, "F32");
    assertEquals(view.shape, [1, 2, 3]);
    // Compare by bit pattern (do not treat -0 and 0 as the same).
    assertEquals(
      new Uint8Array(parsed.buffer, view.byteOffset, view.byteLength),
      new Uint8Array(data.buffer),
    );
  });

  it("builds the same bytes from the same input (the sha does not move between runs)", () => {
    assertEquals(
      f32ArtifactBytes("logits", [1, 2, 3], data),
      f32ArtifactBytes("logits", [1, 2, 3], data),
    );
  });

  it("throws when the shape's element count and the data length disagree", () => {
    assertThrows(
      () => f32ArtifactBytes("logits", [1, 2, 2], data),
      Error,
      "disagrees with the data length",
    );
  });
});

/** A settlement for checking the copy (where the artifact lives plays no part in the copy). */
const settlementOf = (check: ReferenceSettlement["check"]): ReferenceSettlement => ({
  sha256: SHA_B,
  check,
  artifactUrl: new URL("file:///verify/case-1.png"),
});

describe("referenceEntryFields", () => {
  it("written has no expected field at all", () => {
    const fields = referenceEntryFields(settlementOf({ status: "written" }), "case-1.png");
    assertEquals(fields, { status: "written", actual: SHA_B, artifact: "case-1.png" });
    assertEquals(Object.keys(fields), ["status", "actual", "artifact"]);
  });

  it("rewritten holds the value before re-recording in expected", () => {
    const fields = referenceEntryFields(
      settlementOf({ status: "rewritten", previous: SHA_A }),
      "case-1.png",
    );
    assertEquals(Object.keys(fields), ["status", "expected", "actual", "artifact"]);
    assertEquals(fields, {
      status: "rewritten",
      expected: SHA_A,
      actual: SHA_B,
      artifact: "case-1.png",
    });
  });

  it("pass / fail hold the matched row in expected", () => {
    for (const status of ["pass", "fail"] as const) {
      assertEquals(referenceEntryFields(settlementOf({ status, expected: SHA_A }), "case-1.png"), {
        status,
        expected: SHA_A,
        actual: SHA_B,
        artifact: "case-1.png",
      });
    }
  });
});

describe("referenceMismatchMessage", () => {
  const fixtureUrl = new URL("file:///repo/fixtures/references/family.json");

  it("includes the expected / actual sha, where the artifact is and where the fixture is", () => {
    const message = referenceMismatchMessage(
      "case-1",
      settlementOf({ status: "fail", expected: SHA_A }),
      { fixtureUrl },
    );
    assertStringIncludes(message, "case-1");
    assertStringIncludes(message, `expected ${SHA_A}`);
    assertStringIncludes(message, `actual ${SHA_B}`);
    assertStringIncludes(message, "/verify/case-1.png");
    assertStringIncludes(message, "/repo/fixtures/references/family.json");
    assertStringIncludes(message, "do not fall back to a tolerance");
  });

  it("throws when given a settlement that is not a mismatch (does not silently turn a caller error into a sentence)", () => {
    assertThrows(
      () => referenceMismatchMessage("case-1", settlementOf({ status: "written" }), { fixtureUrl }),
      Error,
      "'written'",
    );
  });
});
