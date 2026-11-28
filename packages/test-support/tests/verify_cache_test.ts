/**
 * verify-cache: when a result must not be reused, how a started command enters the key, the
 * judgement of a prediction against the run that followed it, and the `reuse` mode — the decision
 * before a run, the mode variable, the stub of a reused file and the files a command line names —
 * and the code layer's import closure.
 */

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  cacheModeOf,
  type CacheRecord,
  decideBeforeRun,
  filterNamesFile,
  hashOf,
  judgePrediction,
  keyOf,
  keyVariablesOf,
  type Layers,
  reasonsNotReusable,
  RECORD_SCHEMA,
  roleOfCommand,
  stubModuleText,
  summarize,
  sha256Of,
  summarizeReuse,
} from "../src/verify-cache.ts";
import { ImportClosures } from "../src/verify-cache-reporter.ts";

const PASSED = { passed: 3, failed: 0, skipped: 1 };

/** A module reader over in-memory files (absolute path → source text). */
const filesOf =
  (files: Readonly<Record<string, string>>) =>
  (path: string): Uint8Array<ArrayBuffer> | undefined =>
    Object.hasOwn(files, path) ? new TextEncoder().encode(files[path]) : undefined;
const LAYERS: Layers = { environment: "e", data: "d", code: "c" };

describe("reasonsNotReusable", () => {
  it("lets an all-pass run without special variables be reused", () => {
    assertEquals(reasonsNotReusable(PASSED, { KARUME_NO_DAWN: "1" }, []), []);
  });

  it("refuses a failure, a run in which nothing ran, a recording run, an opt-out, unpinned inputs and an unknown process", () => {
    assertEquals(reasonsNotReusable({ passed: 2, failed: 1, skipped: 0 }, {}, []), ["1 failed"]);
    assertEquals(reasonsNotReusable({ passed: 0, failed: 0, skipped: 4 }, {}, []), ["no case ran"]);
    assertEquals(
      reasonsNotReusable(
        PASSED,
        { KARUME_REFERENCE: "write", KARUME_SERIES: "1", KARUME_ALLOW_NO_GPU: "1" },
        ["nvidia-smi --query"],
      ),
      [
        "KARUME_REFERENCE is set (a recording run)",
        "KARUME_SERIES is set (inputs without a pin)",
        "KARUME_ALLOW_NO_GPU is set (an opt-out)",
        "starts a process: nvidia-smi --query",
      ],
    );
  });
});

describe("a run with a test name filter", () => {
  it("is not reusable although every case that ran passed: the other cases did not run", () => {
    assertEquals(reasonsNotReusable(PASSED, {}, [], "sbv2"), [
      "a test name filter was set (sbv2): only the cases it matches ran",
    ]);
  });
});

describe("roleOfCommand", () => {
  it("ignores the checkout queries, takes a file listing as an input and knows nothing else", () => {
    assertEquals(roleOfCommand(["git", "rev-parse", "HEAD"]), "ignored");
    assertEquals(roleOfCommand(["git", "status", "--porcelain"]), "ignored");
    assertEquals(roleOfCommand(["git", "-C", "/repo", "ls-files", "-z", "--"]), "input");
    assertEquals(roleOfCommand(["git", "show", "HEAD:a.ts"]), "unknown");
    assertEquals(roleOfCommand(["nvidia-smi"]), "unknown");
  });

  it("ignores nvidia-smi sampling the GPU's state, which only feeds a test's printed notes", () => {
    const sampling = [
      "nvidia-smi",
      "--query-gpu=memory.used,temperature.gpu,clocks_event_reasons.active",
      "--format=csv,noheader,nounits",
      "-lms",
      "100",
    ];
    assertEquals(roleOfCommand(sampling), "ignored");
    assertEquals(roleOfCommand(["nvidia-smi", "-pl", "200"]), "unknown");
  });
});

describe("decideBeforeRun", () => {
  const record = { schema: RECORD_SCHEMA, layers: LAYERS, notReusable: [] };
  const unchanged = (): Layers => LAYERS;

  it("reuses a reusable record whose three layers still hold", () => {
    assertEquals(decideBeforeRun(record, unchanged), { reuse: true, changed: [], why: [] });
  });

  it("runs a file without a record under this environment", () => {
    assertEquals(decideBeforeRun(undefined, unchanged), {
      reuse: false,
      changed: [],
      why: ["no record under this environment"],
    });
  });

  it("runs a file whose record this harness cannot read, without computing its layers", () => {
    const decision = decideBeforeRun({ ...record, schema: RECORD_SCHEMA + 1 }, () => {
      throw new Error("the layers of an unreadable record were computed");
    });
    assertEquals(decision.reuse, false);
    assertEquals(decision.why, [
      `the record's schema is ${RECORD_SCHEMA + 1}, this harness writes ${RECORD_SCHEMA}`,
    ]);
  });

  it("runs a file whose record is not reusable by its own terms", () => {
    const failed = { ...record, notReusable: ["1 failed", "starts a process: curl"] };
    assertEquals(decideBeforeRun(failed, unchanged).why, [
      "the record is not reusable: 1 failed",
      "the record is not reusable: starts a process: curl",
    ]);
  });

  it("runs a file when any layer computed now differs, and names each", () => {
    for (const name of ["environment", "data", "code"] as const) {
      const decision = decideBeforeRun(record, () => ({ ...LAYERS, [name]: "moved" }));
      assertEquals(decision, { reuse: false, changed: [name], why: [`the ${name} layer changed`] });
    }
  });

  it("decides as the judgement predicts: one rule before and after a run", () => {
    const cases: Layers[] = [LAYERS, { ...LAYERS, data: "d2" }, { ...LAYERS, code: "c2" }];
    for (const notReusable of [[], ["1 failed"]]) {
      for (const now of cases) {
        const previous = { ...record, notReusable, outcome: PASSED, kernelFingerprint: "k" };
        const prediction = judgePrediction(previous, now, previous);
        const decision = decideBeforeRun(previous, () => now);
        assertEquals([decision.reuse, decision.changed], [prediction.reuse, prediction.changed]);
      }
    }
  });
});

describe("cacheModeOf", () => {
  it("records when the variable is unset or `record`, and reuses on `reuse`", () => {
    assertEquals(cacheModeOf({}), "record");
    assertEquals(cacheModeOf({ KARUME_VERIFY_CACHE: "record" }), "record");
    assertEquals(cacheModeOf({ KARUME_VERIFY_CACHE: "reuse" }), "reuse");
  });

  it("refuses any other value, naming the value and the two modes", () => {
    for (const value of ["", "on", "Reuse", "1"]) {
      const error = assertThrows(() => cacheModeOf({ KARUME_VERIFY_CACHE: value }), Error);
      assertStringIncludes(error.message, `KARUME_VERIFY_CACHE=${JSON.stringify(value)}`);
      assertStringIncludes(error.message, '"record"');
      assertStringIncludes(error.message, '"reuse"');
    }
  });
});

describe("keyVariablesOf", () => {
  it("keys on every KARUME_* variable but the mode, so both modes share one environment layer", () => {
    const env = { KARUME_NO_DAWN: "1", KARUME_ALL_FAMILIES: "1", HOME: "/home/x", PATH: "/bin" };
    assertEquals(keyVariablesOf(env), { KARUME_ALL_FAMILIES: "1", KARUME_NO_DAWN: "1" });
    const recorded = hashOf(keyVariablesOf({ ...env, KARUME_VERIFY_CACHE: "record" }));
    const reused = hashOf(keyVariablesOf({ ...env, KARUME_VERIFY_CACHE: "reuse" }));
    assertEquals(recorded, hashOf(keyVariablesOf(env)));
    assertEquals(reused, recorded);
    assert(hashOf(keyVariablesOf({ ...env, KARUME_SERIES: "1" })) !== recorded);
  });
});

describe("stubModuleText", () => {
  const reused = {
    commit: "0123456789abcdef0123",
    dirty: false,
    outcome: { passed: 12, failed: 0, skipped: 2 },
    durationMs: 3450,
  };

  it("is one skipped case that says the file was reused, from which commit, and what passed", () => {
    assertEquals(
      stubModuleText(reused),
      [
        "// Served by the verification cache in place of a test file whose recorded result is reused.",
        'import { it } from "vite-plus/test";',
        'it.skip("reused (KARUME_VERIFY_CACHE=reuse): the record of commit 0123456789ab passed 12 cases (2 skipped) in 3.5 s", () => {});',
        "",
      ].join("\n"),
    );
  });

  it("says when the record came from a checkout with uncommitted changes, or from no known commit", () => {
    assertStringIncludes(
      stubModuleText({ ...reused, dirty: true, outcome: { passed: 1, failed: 0, skipped: 0 } }),
      "the record of commit 0123456789ab with uncommitted changes passed 1 case in",
    );
    assertStringIncludes(
      stubModuleText({ ...reused, commit: undefined }),
      "the record of an unknown commit passed",
    );
  });
});

describe("filterNamesFile", () => {
  const root = "/repo";
  const file = "/repo/packages/hub/tests/fetch_test.ts";

  it("is named by its path from the working directory, an absolute path, or a path suffix", () => {
    assert(filterNamesFile("packages/hub/tests/fetch_test.ts", file, root, root));
    assert(filterNamesFile("./tests/fetch_test.ts", file, "/repo/packages/hub", root));
    assert(filterNamesFile(file, file, "/elsewhere", root));
    assert(filterNamesFile("tests/fetch_test.ts", file, root, root));
    assert(filterNamesFile("fetch_test.ts", file, root, root));
  });

  it("is not named by a directory, a fragment of its name or another file", () => {
    assertEquals(filterNamesFile("packages/hub", file, root, root), false);
    assertEquals(filterNamesFile("packages/hub/tests/", file, root, root), false);
    assertEquals(filterNamesFile("fetch", file, root, root), false);
    assertEquals(filterNamesFile("h_test.ts", file, root, root), false);
    assertEquals(filterNamesFile("other_fetch_test.ts", file, root, root), false);
  });
});

describe("summarizeReuse", () => {
  it("counts the reused files with their records' seconds, and the files that ran", () => {
    assertEquals(
      summarizeReuse(
        [
          { file: "a_test.ts", commit: "c1", durationMs: 1240 },
          { file: "b_test.ts", commit: undefined, durationMs: 3000 },
        ],
        5,
      ),
      {
        reused: 2,
        reusedSeconds: 4.2,
        ran: 5,
        reusedFiles: [
          { file: "a_test.ts", commit: "c1", seconds: 1.2 },
          { file: "b_test.ts", commit: undefined, seconds: 3 },
        ],
      },
    );
  });
});

describe("judgePrediction", () => {
  const previous = { layers: LAYERS, notReusable: [], outcome: PASSED, kernelFingerprint: "k1" };

  it("predicts a reuse when every layer holds, and the same run confirms it", () => {
    assertEquals(judgePrediction(previous, LAYERS, previous), {
      reuse: true,
      changed: [],
      contradictions: [],
    });
  });

  it("predicts a re-run and names the layers that moved", () => {
    assertEquals(
      judgePrediction(previous, { ...LAYERS, data: "d2", code: "c2" }, previous).changed,
      ["data", "code"],
    );
  });

  it("never predicts a reuse of a record that must not be reused", () => {
    const failed = { ...previous, notReusable: ["1 failed"] };
    assertEquals(judgePrediction(failed, LAYERS, previous).reuse, false);
  });

  it("reports a predicted reuse that the run contradicts: other kernels, or another outcome", () => {
    const moved = judgePrediction(previous, LAYERS, { ...previous, kernelFingerprint: "k2" });
    assertEquals(moved.reuse, true);
    assertEquals(moved.contradictions, ["the kernels dispatched differ from the record's"]);
    const broke = judgePrediction(previous, LAYERS, {
      ...previous,
      outcome: { passed: 2, failed: 1, skipped: 1 },
    });
    assertEquals(broke.contradictions.length, 1);
  });
});

describe("keyOf / summarize", () => {
  it("gives one key per triple of layers", () => {
    assertEquals(keyOf(LAYERS), keyOf({ ...LAYERS }));
    assertEquals(keyOf(LAYERS) === keyOf({ ...LAYERS, code: "c2" }), false);
  });

  it("totals the files, the predicted reuses with their seconds, and the contradicted ones", () => {
    const record = (
      file: string,
      durationMs: number,
      prediction: CacheRecord["prediction"],
    ): CacheRecord => ({
      schema: 1,
      file,
      lane: "core",
      commit: undefined,
      dirty: undefined,
      durationMs,
      outcome: PASSED,
      notReusable: [],
      driver: undefined,
      layers: LAYERS,
      key: keyOf(LAYERS),
      environment: {},
      data: [],
      code: [],
      kernels: [],
      kernelFingerprint: "k1",
      prediction,
    });
    const reuse = { reuse: true, changed: [], contradictions: [] };
    assertEquals(
      summarize([
        record("first.ts", 1500, undefined),
        record("held.ts", 2000, reuse),
        record("moved.ts", 500, { reuse: false, changed: ["code"], contradictions: [] }),
        record("wrong.ts", 4000, { ...reuse, contradictions: ["other kernels"] }),
      ]),
      {
        files: 4,
        judged: 3,
        reuse: 2,
        reuseSeconds: 6,
        totalSeconds: 8,
        contradicted: [{ file: "wrong.ts", why: ["other kernels"] }],
      },
    );
  });
});

describe("ImportClosures", () => {
  it("takes a module imported with a query (`?raw`) as its file, identified by the file's bytes", () => {
    const wgsl = "@compute @workgroup_size(1) fn main() {}\n";
    const closures = new ImportClosures(
      new Map(),
      filesOf({ "/repo/a.ts": 'import text from "./k.wgsl?raw";\n', "/repo/k.wgsl": wgsl }),
    );
    assertEquals(closures.closureOf(["/repo/a.ts"]), ["/repo/a.ts", "/repo/k.wgsl"]);
    assertEquals(closures.hashOfModule("/repo/k.wgsl"), sha256Of(wgsl));
  });
});
