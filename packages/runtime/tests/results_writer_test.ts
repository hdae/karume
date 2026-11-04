// Writing out verification results (`outputs/verify/<environment key>/<date>_<series>/`).
//
// Three things are checked here: results directories are separated by environment key; every
// `record` rewrites `results.json` with **all entries up to that point** (a crash midway keeps
// everything before it); and artifacts stay inside the same directory. Neither a real GPU nor
// `outputs/` is needed — the root and the environment are injected.

import {
  makeTempDirSync,
  mkdirSync,
  readTextFile,
  removeSync,
  writeFile,
  writeTextFileSync,
} from "@karume/test-support/fs";
import {
  assert,
  assertEquals,
  assertRejects,
  assertStrictEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import { parseResultsDocument } from "../../../tools/verify-diff/diff.ts";
import type { Environment } from "./helpers/environment.ts";
import {
  type Comparison,
  type Measurement,
  openResults,
  recordFailure,
  type ResultEntry,
  type Results,
  runRecordedCase,
} from "./helpers/results.ts";

const KEY = "deno-intel-graphics-bmg-g21";

const ENVIRONMENT: Environment = {
  key: KEY,
  runtime: { name: "deno", version: "2.9.6", v8: "15.0.245.2", typescript: "6.0.3" },
  adapter: { vendor: "32902", architecture: "", device: "57868", description: "test" },
  os: { platform: "linux", arch: "x86_64" },
};

type ResultsDocument = {
  readonly schema: number;
  readonly family: string;
  readonly environment: Environment;
  readonly checkout: { readonly sha: string; readonly dirty: boolean } | null;
  readonly startedAt: string;
  readonly cases: readonly ResultEntry[];
};

test("results writer: every record rewrites all entries, and artifacts go in the same directory", async () => {
  const temporary = makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const root = new URL(`file://${temporary}/`);
    const results = openResults("irodori", { root, environment: ENVIRONMENT });

    const artifact = results.artifact("voice-clone.wav");
    assert(
      artifact.href.startsWith(results.dir.href),
      "the artifact points outside the results directory",
    );
    await writeFile(artifact, new Uint8Array([1, 2, 3]));

    await results.record({
      id: "voice-clone",
      status: "pass",
      expected: "a".repeat(64),
      actual: "a".repeat(64),
      artifact: "voice-clone.wav",
      elapsedMs: 12,
    });
    const afterFirst = JSON.parse(
      await readTextFile(new URL("results.json", results.dir)),
    ) as ResultsDocument;
    assertEquals(afterFirst.cases.length, 1, "the first entry is missing");
    // The directory's date is compared against the same time source as the implementation: the date
    // part (UTC) of the startedAt fixed when the run started. Reading the clock again in the test
    // would fail spuriously only on runs that cross UTC midnight between the start and here.
    assert(/^\d{4}-\d{2}-\d{2}T/.test(afterFirst.startedAt), "startedAt is not an ISO date-time");
    const today = afterFirst.startedAt.slice(0, 10);
    assertEquals(results.dir.href, new URL(`${KEY}/${today}_irodori/`, root).href);

    await results.record({ id: "no-ref", status: "written", actual: "b".repeat(64), elapsedMs: 8 });
    const document = JSON.parse(
      await readTextFile(new URL("results.json", results.dir)),
    ) as ResultsDocument;
    assertEquals(document.schema, 1);
    assertEquals(document.family, "irodori");
    assertEquals(document.environment.key, KEY);
    assertEquals(
      document.cases.map((entry) => entry.id),
      ["voice-clone", "no-ref"],
    );
    assertEquals(document.cases[1].status, "written");
    // A run that created the reference value has no expected (JSON.stringify drops undefined
    // fields).
    assertEquals(Object.hasOwn(document.cases[1], "expected"), false);
    assert(document.startedAt.endsWith("Z"), "startedAt is not an ISO string");
    // This repository is under git, so the checkout can be read (null only on a machine without
    // git).
    assert(
      document.checkout === null || /^[0-9a-f]{40}$/.test(document.checkout.sha),
      `the checkout's sha is not a commit SHA: ${JSON.stringify(document.checkout)}`,
    );
  } finally {
    removeSync(temporary, { recursive: true });
  }
});

test("results writer: creating the directory places an in-progress results.json (cases empty)", async () => {
  const temporary = makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const root = new URL(`file://${temporary}/`);
    const results = openResults("golden", { root, environment: ENVIRONMENT });
    // The outcomes a previous run left in the same directory (same day, same series).
    mkdirSync(results.dir, { recursive: true });
    writeTextFileSync(
      new URL("results.json", results.dir),
      `${JSON.stringify({ schema: 1, cases: [{ id: "activations", status: "pass" }] })}\n`,
    );
    // Only an artifact is placed and record is never reached = a run that exited with an exception
    // before recording.
    await writeFile(results.artifact("activations.png"), new Uint8Array([1, 2, 3]));
    const document = JSON.parse(
      await readTextFile(new URL("results.json", results.dir)),
    ) as ResultsDocument;
    assertEquals(
      document.cases,
      [],
      "the previous run's outcomes remain in the same directory as this run's artifacts",
    );
    assertEquals(document.family, "golden");
    assert(
      document.startedAt.endsWith("Z"),
      "the in-progress marker does not carry this run's startedAt",
    );
  } finally {
    removeSync(temporary, { recursive: true });
  }
});

test("results writer: tolerance measurements are kept in the shape given to record", async () => {
  const temporary = makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const root = new URL(`file://${temporary}/`);
    const results = openResults("golden", { root, environment: ENVIRONMENT });
    // Measurements of a passing run (without them, "how large the difference was" is unknown until
    // it goes red).
    const measurements: readonly Measurement[] = [
      {
        output: "sin",
        maxAbs: 2.68e-5,
        maxRel: 1.72e-5,
        tolerance: { atol: 2 ** -11, rtol: 0 },
        stage: "spec",
      },
      {
        output: "gelu",
        maxAbs: 1.19e-7,
        maxRel: 1.83e-6,
        tolerance: { atol: 1e-6, rtol: 1e-5 },
        stage: "karume",
      },
    ];
    await results.record({ id: "activations", status: "pass", elapsedMs: 31, measurements });
    const document = JSON.parse(
      await readTextFile(new URL("results.json", results.dir)),
    ) as ResultsDocument;
    assertEquals(document.cases[0].measurements, measurements);
  } finally {
    removeSync(temporary, { recursive: true });
  }
});

test("results writer: non-finite measurements are written as null", async () => {
  const temporary = makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const root = new URL(`file://${temporary}/`);
    const results = openResults("golden", { root, environment: ENVIRONMENT });
    // NaN / ±Inf (measurements of a run whose output contained non-finite values). JSON has no
    // spelling for them, so they become null — the reader (tools/verify-diff) takes this null as
    // "could not be measured".
    await results.record({
      id: "activations",
      status: "fail",
      elapsedMs: 9,
      measurements: [
        {
          output: "sin",
          maxAbs: Number.POSITIVE_INFINITY,
          maxRel: Number.NaN,
          tolerance: { atol: 1e-6, rtol: 1e-5 },
          stage: "karume",
        },
      ],
    });
    const text = await readTextFile(new URL("results.json", results.dir));
    assertStringIncludes(text, `"maxAbs": null`);
    assertStringIncludes(text, `"maxRel": null`);
  } finally {
    removeSync(temporary, { recursive: true });
  }
});

// Keeping the outcome of a failed run is "the job", but throwing when that write fails would
// replace the original verification exception held by the caller's catch (the only diagnostic
// saying what broke) with an I/O exception.
test("results writer: recordFailure neither throws nor stays silent when it cannot record", async () => {
  const temporary = makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const root = new URL(`file://${temporary}/`);
    // Fault injection: make the directory's parent (the environment key directory) a regular file =
    // a place where the directory cannot be created.
    writeTextFileSync(new URL(KEY, root), "");
    const results = openResults("golden", { root, environment: ENVIRONMENT });
    const entry: ResultEntry = { id: "activations", status: "fail", elapsedMs: 3 };
    // Pin first that the fault injection works (if a plain record passes, the check below says
    // nothing).
    await assertRejects(() => results.record(entry));

    const said: string[] = [];
    const original = console.error;
    console.error = (...args: unknown[]): void => {
      said.push(args.map(String).join(" "));
    };
    try {
      // It does not replace the original verification exception, so it returns here without
      // throwing.
      await recordFailure(results, entry);
    } finally {
      console.error = original;
    }
    assertEquals(said.length, 1, "the failure to record was silently swallowed");
    assertStringIncludes(said[0], "activations");
  } finally {
    removeSync(temporary, { recursive: true });
  }
});

test("results writer: an artifact name containing a directory separator throws", () => {
  const temporary = makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const results = openResults("anima", {
      root: new URL(`file://${temporary}/`),
      environment: ENVIRONMENT,
    });
    assertThrows(() => results.artifact("F1/i8.wav"), Error, "ディレクトリ区切り");
  } finally {
    removeSync(temporary, { recursive: true });
  }
});

test("results writer: artifact names that can point outside the directory or to its parent (\\ / . / .. / % / ? / #) throw", () => {
  // In a file: URL a backslash also resolves as a separator, and `..` or `%2e%2e` points to the
  // parent directory. Rejecting only `/` would let these write outside the results directory (or to
  // the directory itself).
  const temporary = makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const results = openResults("anima", {
      root: new URL(`file://${temporary}/`),
      environment: ENVIRONMENT,
    });
    // What follows `?` / `#` goes to the query / fragment, so the recorded name and the write
    // target (`a`) diverge.
    for (const name of [
      "..\\outside.wav",
      "..",
      ".",
      "%2e%2e",
      "%2E%2E\\x.wav",
      "",
      "a?b.wav",
      "a#b.wav",
    ]) {
      assertThrows(() => results.artifact(name), Error, "実物の名前", JSON.stringify(name));
    }
    // Control: ordinary names (extensions with dots, sequence numbers) resolve directly under the
    // directory.
    for (const name of ["F1.i8.wav", "frame-0001.png"]) {
      assertEquals(results.artifact(name).href, new URL(name, results.dir).href);
    }
  } finally {
    removeSync(temporary, { recursive: true });
  }
});

// runRecordedCase (the wrapper around each real-weight golden case). The writing itself is checked
// above, so here recording is faked and only "every path appends one outcome of the same shape" is
// checked.

/** A fake that keeps the appended outcomes (with fault injection that makes `record` fail). */
const fakeResults = (failRecord = false): { results: Results; entries: ResultEntry[] } => {
  const entries: ResultEntry[] = [];
  const results: Results = {
    dir: new URL("file:///fake/"),
    artifact: (name: string): URL => new URL(name, "file:///fake/"),
    record: (entry: ResultEntry): Promise<void> => {
      if (failRecord) return Promise.reject(new Error("cannot write results.json"));
      entries.push(entry);
      return Promise.resolve();
    },
  };
  return { results, entries };
};

const MEASUREMENT: Measurement = {
  output: "logits",
  maxAbs: 1e-4,
  maxRel: 2e-4,
  tolerance: { atol: 1e-3, rtol: 1e-3 },
  stage: "karume",
};

/** Waits for `action` while capturing console.error. */
const captureConsoleError = async (action: () => Promise<void>): Promise<string[]> => {
  const said: string[] = [];
  const original = console.error;
  console.error = (...args: unknown[]): void => {
    said.push(args.map(String).join(" "));
  };
  try {
    await action();
  } finally {
    console.error = original;
  }
  return said;
};

test("runRecordedCase: a run whose body returns keeps one entry with pass and the appended measurements", async () => {
  const { results, entries } = fakeResults();
  await runRecordedCase(results, { id: "case0" }, ({ measurements }) => {
    measurements.push(MEASUREMENT);
    return Promise.resolve(undefined);
  });
  assertEquals(entries.length, 1);
  const [entry] = entries;
  assertEquals(Object.keys(entry), ["id", "status", "elapsedMs", "measurements"]);
  assertEquals(entry.id, "case0");
  assertEquals(entry.status, "pass");
  assertEquals(entry.measurements, [MEASUREMENT]);
  assert(
    Number.isInteger(entry.elapsedMs) && entry.elapsedMs >= 0,
    "elapsedMs is not an integer number of ms",
  );
});

test("runRecordedCase: a run whose body throws keeps fail and then rethrows the same exception", async () => {
  const { results, entries } = fakeResults();
  const cause = new Error("output.0: allclose mismatch");
  const order: string[] = [];
  const thrown = await assertRejects(() =>
    runRecordedCase(
      results,
      {
        id: "case1",
        onFailure: () => {
          order.push(`onFailure (entries appended: ${entries.length})`);
        },
      },
      ({ measurements }) => {
        // What was measured before the failure is part of the outcome (how far it got is the
        // failure's diagnostic).
        measurements.push(MEASUREMENT);
        return Promise.reject(cause);
      },
    ),
  );
  assertStrictEquals(
    thrown,
    cause,
    "the original verification exception was replaced by another exception",
  );
  assertEquals(entries.length, 1);
  assertEquals(Object.keys(entries[0]), ["id", "status", "elapsedMs", "measurements"]);
  assertEquals(entries[0].status, "fail");
  assertEquals(entries[0].measurements, [MEASUREMENT]);
  // The series' "left in the case's directory" mark is set before recording.
  assertEquals(order, ["onFailure (entries appended: 0)"]);
});

test("runRecordedCase: a run whose body returns fail keeps it with its note without throwing", async () => {
  const { results, entries } = fakeResults();
  await runRecordedCase(results, { id: "gelu" }, () =>
    Promise.resolve({ status: "fail", note: "accepted within the specified band: sin" }),
  );
  assertEquals(entries.length, 1);
  assertEquals(Object.keys(entries[0]), ["id", "status", "elapsedMs", "note", "measurements"]);
  assertEquals(entries[0].status, "fail");
  assertEquals(entries[0].note, "accepted within the specified band: sin");
  assertEquals(entries[0].measurements, []);
});

test("runRecordedCase: the note of a run that threw is built from the exception by failureNote", async () => {
  const { results, entries } = fakeResults();
  await assertRejects(
    () =>
      runRecordedCase(
        results,
        {
          id: "gelu",
          failureNote: (cause) =>
            `exception: ${cause instanceof Error ? cause.message : String(cause)}`,
        },
        () => Promise.reject(new Error("createSession failed")),
      ),
    Error,
    "createSession failed",
  );
  assertEquals(Object.keys(entries[0]), ["id", "status", "elapsedMs", "note", "measurements"]);
  assertEquals(entries[0].note, "exception: createSession failed");
});

test("runRecordedCase: when recording fails on a run that threw, it throws the original exception and says it could not record", async () => {
  const { results } = fakeResults(true);
  const cause = new Error("shape mismatch");
  let thrown: unknown;
  const said = await captureConsoleError(async () => {
    thrown = await assertRejects(() =>
      runRecordedCase(results, { id: "case2" }, () => Promise.reject(cause)),
    );
  });
  assertStrictEquals(
    thrown,
    cause,
    "the original verification exception was replaced by an I/O exception",
  );
  assertEquals(said.length, 1, "the failure to record was silently swallowed");
  assertStringIncludes(said[0], "case2");
});

test("runRecordedCase: when recording fails on a run that returned, it throws and does not call onFailure", async () => {
  const { results } = fakeResults(true);
  let failed = false;
  await assertRejects(
    () =>
      runRecordedCase(
        results,
        {
          id: "case3",
          onFailure: () => {
            failed = true;
          },
        },
        () => Promise.resolve(undefined),
      ),
    Error,
    "cannot write results.json",
  );
  // The case's outcome is not kept — if the series mistook it for "left in the case's directory",
  // it would be kept nowhere.
  assertEquals(failed, false);
});

// The writer (helpers/results.ts) and the reader (tools/verify-diff) have separate types. The ADR
// 0106 addendum's "schema stays 1" depends on the two agreeing, so the document the writer actually
// wrote is passed to the reader.
test("round trip: the verify-diff reader accepts the results.json written by runRecordedCase as is", async () => {
  const temporary = makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const root = new URL(`file://${temporary}/`);
    const results = openResults("golden", { root, environment: ENVIRONMENT });
    await runRecordedCase(results, { id: "measured" }, ({ measurements }) => {
      measurements.push({ ...MEASUREMENT, maxAbs: Number.NaN });
      return Promise.resolve(undefined);
    });
    const cause = new Error("failed before measuring");
    await assertRejects(() =>
      runRecordedCase(results, { id: "thrown", failureNote: String }, () => Promise.reject(cause)),
    );

    const text = await readTextFile(new URL("results.json", results.dir));
    const document = parseResultsDocument(JSON.parse(text));
    assertEquals(document.family, "golden");
    assertEquals(document.environment.key, KEY);
    assertEquals(
      document.cases.map((entry) => [entry.id, entry.status]),
      [
        ["measured", "pass"],
        ["thrown", "fail"],
      ],
    );
    // Non-finite values become null through the writer's JSON.stringify, and the reader takes them
    // as "could not be measured".
    assertEquals(document.cases[0].measurements, [{ ...MEASUREMENT, maxAbs: null }]);
    // Through runRecordedCase, a run that threw before measuring also has the field (an empty
    // array).
    assertEquals(document.cases[1].measurements, []);
    assertEquals(document.cases[1].note, "Error: failed before measuring");
  } finally {
    removeSync(temporary, { recursive: true });
  }
});

// Measurements of the own-machine A/B check (ADR 0110 decision 5) go in the optional field
// `comparisons`. Only cases that appended them have the field; non-finite values are null and a
// band not yet derived drops out with its field, and the reader (tools/verify-diff) accepts this as
// is.
test("round trip: A/B measurements appended by runRecordedCase reach the reader as comparisons", async () => {
  const temporary = makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const root = new URL(`file://${temporary}/`);
    const results = openResults("anima-ab", { root, environment: ENVIRONMENT });
    const comparison: Comparison = {
      output: "step1-latent",
      reference: "f16+dit8",
      practical: "f16+dit8-a8-attn8-s16",
      relRms: 1.2e-2,
      maxAbs: 0.31,
      band: { metric: "relRms", floor: 1e-3, ceiling: 2.5e-2 },
    };
    await runRecordedCase(results, { id: "banded" }, ({ comparisons }) => {
      comparisons.push(comparison);
      return Promise.resolve(undefined);
    });
    await runRecordedCase(results, { id: "unbanded" }, ({ comparisons }) => {
      comparisons.push({ ...comparison, relRms: Number.NaN, band: undefined });
      return Promise.resolve({ status: "fail", note: "band not derived" });
    });
    // A case that appended none has no field (the outcome shape of a case that is not an A/B check
    // is unchanged).
    await runRecordedCase(results, { id: "plain" }, () => Promise.resolve(undefined));

    const text = await readTextFile(new URL("results.json", results.dir));
    const document = parseResultsDocument(JSON.parse(text));
    assertEquals(document.cases[0].comparisons, [comparison]);
    const { band: _dropped, ...unbanded } = comparison;
    assertEquals(document.cases[1].comparisons, [{ ...unbanded, relRms: null }]);
    assertEquals(Object.hasOwn(document.cases[2], "comparisons"), false);
    assertEquals(Object.keys(JSON.parse(text).cases[0]), [
      "id",
      "status",
      "elapsedMs",
      "measurements",
      "comparisons",
    ]);
  } finally {
    removeSync(temporary, { recursive: true });
  }
});

// Cases that double as the sha check (the outcome of reference.ts's settleReference goes into the
// verdict). The fields follow ResultEntry's declaration order, and the outcome shape of a case that
// carries none stays as before.
describe("the reference-value fields carried in runRecordedCase's outcome", () => {
  it("a written outcome copies status / actual / artifact and has no expected", async () => {
    const { results, entries } = fakeResults();
    await runRecordedCase(results, { id: "golden-0" }, ({ measurements }) => {
      measurements.push(MEASUREMENT);
      return Promise.resolve({
        status: "written",
        actual: "b".repeat(64),
        artifact: "golden-0.safetensors",
      });
    });
    assertEquals(entries.length, 1);
    assertEquals(Object.keys(entries[0]), [
      "id",
      "status",
      "actual",
      "artifact",
      "elapsedMs",
      "measurements",
    ]);
    assertEquals(entries[0].status, "written");
    assertEquals(entries[0].actual, "b".repeat(64));
    assertEquals(entries[0].artifact, "golden-0.safetensors");
  });

  it("copies expected / actual / artifact / note in ResultEntry's declaration order", async () => {
    const { results, entries } = fakeResults();
    await runRecordedCase(results, { id: "golden-1" }, () =>
      Promise.resolve({
        note: "sha mismatch",
        artifact: "golden-1.safetensors",
        actual: "b".repeat(64),
        expected: "a".repeat(64),
        status: "fail",
      }),
    );
    assertEquals(Object.keys(entries[0]), [
      "id",
      "status",
      "expected",
      "actual",
      "artifact",
      "elapsedMs",
      "note",
      "measurements",
    ]);
    assertEquals(entries[0].status, "fail");
    assertEquals(entries[0].expected, "a".repeat(64));
  });

  it("the entry of an outcome without reference-value fields keeps its previous shape", async () => {
    const { results, entries } = fakeResults();
    await runRecordedCase(results, { id: "plain" }, () => Promise.resolve({ status: "pass" }));
    assertEquals(Object.keys(entries[0]), ["id", "status", "elapsedMs", "measurements"]);
  });

  it("a rewritten outcome reaches the verify-diff reader", async () => {
    const temporary = makeTempDirSync({ prefix: "karume-verify-" });
    try {
      const root = new URL(`file://${temporary}/`);
      const results = openResults("golden", { root, environment: ENVIRONMENT });
      await runRecordedCase(results, { id: "rewritten" }, () =>
        Promise.resolve({
          status: "rewritten",
          expected: "a".repeat(64),
          actual: "b".repeat(64),
          artifact: "rewritten.safetensors",
        }),
      );
      const text = await readTextFile(new URL("results.json", results.dir));
      const [entry] = parseResultsDocument(JSON.parse(text)).cases;
      assertEquals(
        [entry.id, entry.status, entry.expected, entry.actual, entry.artifact],
        ["rewritten", "rewritten", "a".repeat(64), "b".repeat(64), "rewritten.safetensors"],
      );
    } finally {
      removeSync(temporary, { recursive: true });
    }
  });
});
