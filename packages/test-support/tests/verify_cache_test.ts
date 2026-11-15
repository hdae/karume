/**
 * verify-cache: when a result must not be reused, how a started command enters the key, and the
 * judgement of a prediction against the run that followed it.
 */

import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  type CacheRecord,
  judgePrediction,
  keyOf,
  type Layers,
  reasonsNotReusable,
  roleOfCommand,
  summarize,
} from "../src/verify-cache.ts";

const PASSED = { passed: 3, failed: 0, skipped: 1 };
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

describe("roleOfCommand", () => {
  it("ignores the checkout queries, takes a file listing as an input and knows nothing else", () => {
    assertEquals(roleOfCommand(["git", "rev-parse", "HEAD"]), "ignored");
    assertEquals(roleOfCommand(["git", "status", "--porcelain"]), "ignored");
    assertEquals(roleOfCommand(["git", "-C", "/repo", "ls-files", "-z", "--"]), "input");
    assertEquals(roleOfCommand(["git", "show", "HEAD:a.ts"]), "unknown");
    assertEquals(roleOfCommand(["nvidia-smi"]), "unknown");
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
