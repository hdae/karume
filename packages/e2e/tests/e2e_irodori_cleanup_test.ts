// A check that every resource release is attempted even when the resident loop's body, the batch
// finish and several cleanups fail. Faults are injected after the real release, so the test
// itself leaves no VRAM or references behind. The prototype patches stay inside this test and are
// always restored. Other files run in separate Deno workers.
//
// Stage 6' (the condition-side K/V projection `dit_context` — ADR 0114) runs to completion in the
// preceding batch and tears its Session down, and only then are `dit` / combine / euler opened
// (the VRAM-peak MUST — `projectContextOnGpu` in `dit-loop.ts`). The fault is injected into the
// loop body (`dit`'s first enqueue), the preceding stage passes through, and the order and the
// release counts are checked in the same single generation.
import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import { Pipeline } from "../../models/irodori.ts";
import {
  DIST_COMMAND,
  hasQuantSeat,
  loadLocalAssets,
  MODEL,
  readManifest,
} from "./helpers/irodori-assets.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { Session } from "../../runtime/src/runtime/executor.ts";
import { BatchScope, ResidentTensor } from "../../runtime/src/gpu/device.ts";

if (!hasQuantSeat("i8")) {
  console.warn(
    `[karume] the i8 assets for the Irodori cleanup test are absent, so it SKIPs. Generate: ${DIST_COMMAND}`,
  );
}
test.skipIf(!GPU_AVAILABLE || !hasQuantSeat("i8"))(
  "Irodori resident loop: keeps the failures of the body and of multiple cleanups and releases every Session and resident (real GPU)",
  async () => {
    const manifest = readManifest();
    const assets = await loadLocalAssets(manifest, "i8");
    const pipeline = await Pipeline.fromAssets(
      { manifest, assets },
      {
        model: MODEL,
        quant: "i8",
      },
    );
    // The originals are kept unbound on purpose: each patch re-invokes them with `.call(this)` /
    // `.apply(this)` and the finally block puts the same function objects back.
    // oxlint-disable-next-line typescript/unbound-method -- re-invoked with an explicit receiver, restored as-is
    const build = Session.build;
    // oxlint-disable-next-line typescript/unbound-method -- re-invoked with an explicit receiver, restored as-is
    const enqueue = Session.prototype.enqueue;
    // oxlint-disable-next-line typescript/unbound-method -- re-invoked with an explicit receiver, restored as-is
    const sessionDispose = Session.prototype.dispose;
    // oxlint-disable-next-line typescript/unbound-method -- re-invoked with an explicit receiver, restored as-is
    const residentDispose = ResidentTensor.prototype.dispose;
    // oxlint-disable-next-line typescript/unbound-method -- re-invoked with an explicit receiver, restored as-is
    const finish = BatchScope.prototype.finish;
    const bodyFault = new Error("probe: enqueue failed");
    const finishFault = new Error("probe: finish failed");
    const sessionFaults = [
      new Error("probe: dispose session 1"),
      new Error("probe: dispose session 2"),
    ];
    const residentFault = new Error("probe: dispose resident 1");
    // Stage progression: the preceding stage (after seeing `dit_context`'s enqueue) → the loop
    // (the fault on `dit`'s enqueue).
    let phase: "before" | "context" | "loop" = "before";
    let sessions = 0;
    let loopSessions = 0;
    let residents = 0;
    let finishes = 0;
    const order: string[] = [];
    try {
      // Every Session build goes through `Session.build` (container components and the host's
      // small graphs alike).
      Session.build = function (...args: Parameters<typeof build>) {
        order.push("open");
        return build.apply(this, args);
      };
      Session.prototype.enqueue = function (inputs, options): Promise<void> {
        // Only `dit` takes `x_t`, and only the preceding `dit_context` takes `speaker_state` (the
        // small graphs use other names).
        if (Object.hasOwn(inputs, "x_t")) {
          phase = "loop";
          order.push("enqueue:dit");
          return Promise.reject(bodyFault);
        }
        if (Object.hasOwn(inputs, "speaker_state")) {
          phase = "context";
          order.push("enqueue:dit_context");
        }
        return enqueue.call(this, inputs, options);
      };
      BatchScope.prototype.finish = async function () {
        await finish.call(this);
        if (phase === "context") order.push("finish:dit_context");
        if (phase === "loop") {
          finishes++;
          order.push("finish");
          throw finishFault;
        }
      };
      Session.prototype.dispose = async function () {
        await sessionDispose.call(this);
        if (phase === "before") return;
        sessions++;
        order.push("dispose");
        if (phase === "loop") {
          loopSessions++;
          if (loopSessions <= sessionFaults.length) throw sessionFaults[loopSessions - 1];
        }
      };
      ResidentTensor.prototype.dispose = function () {
        residentDispose.call(this);
        if (phase === "loop") {
          residents++;
          order.push(`resident${residents}`);
          if (residents === 1) throw residentFault;
        }
      };
      const error = await assertRejects(() =>
        pipeline.generateLatent({ text: "こんにちは。", durationSeconds: 0.5, seed: 7 }),
      );
      const flatten = (error: unknown): unknown[] =>
        error instanceof AggregateError ? error.errors.flatMap(flatten) : [error];
      const failures = flatten(error);
      assertEquals(failures, [bodyFault, finishFault, ...sessionFaults, residentFault]);
      assertEquals(finishes, 1);
      // dit_context (released in the preceding stage) + dit + combine + euler.
      assertEquals(sessions, 4);
      // 4 working tensors + 24 condition-side K / V (12 blocks × K / V).
      assertEquals(residents, 28);
      // The preceding stage's Session is torn down **before** `dit` is opened (the VRAM-peak MUST).
      const start = order.lastIndexOf("open", order.indexOf("enqueue:dit_context"));
      assert(start >= 0, `no Session build of dit_context was observed (${order.join(" / ")})`);
      assertEquals(order.slice(start, start + 12), [
        "open",
        "enqueue:dit_context",
        "finish:dit_context",
        "dispose",
        "open",
        "open",
        "open",
        "enqueue:dit",
        "finish",
        "dispose",
        "dispose",
        "dispose",
      ]);
      assert(order.slice(start + 12).every((value) => value.startsWith("resident")));
      console.log(
        JSON.stringify({
          finishes,
          sessions,
          residents,
          order: order.slice(start),
          failures: failures.map((error) =>
            error instanceof Error ? error.message : String(error),
          ),
        }),
      );
    } finally {
      Session.build = build;
      Session.prototype.enqueue = enqueue;
      Session.prototype.dispose = sessionDispose;
      ResidentTensor.prototype.dispose = residentDispose;
      BatchScope.prototype.finish = finish;
      await pipeline.dispose();
    }
  },
);
