/**
 * Irodori's **check of stage 6' (the condition-side K/V projection `dit_context` — ADR 0114)**
 * (real GPU).
 *
 * It pins 2 points:
 *
 * - **(a) Run counts** — in one generation `dit_context` runs exactly once and `dit` runs once per
 *   forward. Checked on both the resident path (`enqueue`) and the host path (`run` — with an
 *   `onEvent` subscription). The counts catch the shapes where the hoist came off and it runs per
 *   forward, where the preceding stage does not run, and where only one of the 2 paths runs it a
 *   different way (the WAV check stays green as long as the numbers do not change, so a hoist
 *   coming off is invisible to it).
 * - **(b) Confirming detection power (fault injection)** — swapping the `context_k_0` and
 *   `context_v_0` passed to `dit` **changes** the sha256 of the generated WAV from that of the
 *   unpatched generation. If it did not, the K / V wiring would be invisible to the WAV sha256 check
 *   (`e2e_irodori_wav_test.ts`) — and "the reference sha is unchanged after splitting the graph"
 *   could no longer serve as evidence that the split is correct.
 *
 * The prototype patches stay inside this test and are always restored (the same practice as
 * `e2e_irodori_cleanup_test.ts` — other files run in separate Deno workers).
 *
 * MUST: the assets are the `i8` seat of `models/karume-irodori-v4-small/` (untracked, local
 * assets of the real-GPU machine). An environment missing them or without a GPU **SKIPs
 * explicitly** with the generation command (ADR 0005).
 */

import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assert, assertEquals, assertNotEquals } from "@std/assert";
import type { RunInputs } from "@karume/runtime";
import { type GenerateRequest, Pipeline, type RunComponent } from "../../models/irodori.ts";
import { encodeWav } from "../../models/mod.ts";
import {
  DIST_COMMAND,
  hasQuantSeat,
  loadLocalAssets,
  MODEL,
  readManifest,
} from "./helpers/irodori-assets.ts";
import { sha256Hex } from "../../models/tests/helpers/container-fixture.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { Session } from "../../runtime/src/runtime/executor.ts";

const QUANT = "i8";

if (!hasQuantSeat(QUANT)) {
  console.warn(
    `[karume] the ${QUANT} assets for Irodori's dit_context check are absent, so it SKIPs. Generate: ${DIST_COMMAND}`,
  );
}
const RUNNABLE = GPU_AVAILABLE && hasQuantSeat(QUANT);

/**
 * A request kept short — `durationSeconds` fixes S so `duration` does not run, and neither a
 * reference nor a caption is passed (only the text CFG is active, so the forward count exceeds the
 * step count).
 */
const REQUEST = {
  text: "こんにちは、これはテストです。",
  seed: 11,
  durationSeconds: 1,
} as const satisfies GenerateRequest;

/**
 * The kind of one call into a Session (only `dit` takes `x_t`, only `dit_context` takes
 * `speaker_state`).
 */
const kindOf = (inputs: RunInputs): "context" | "dit" | undefined =>
  Object.hasOwn(inputs, "x_t")
    ? "dit"
    : Object.hasOwn(inputs, "speaker_state")
      ? "context"
      : undefined;

type Calls = { context: number; dit: number };

const wavSha = async (pipeline: Pipeline): Promise<string> => {
  const audio = await pipeline.generate(REQUEST);
  return await sha256Hex(encodeWav(audio.data, audio.sampleRate));
};

describe.skipIf(!RUNNABLE)(
  "e2e (real GPU): dit_context runs exactly once per generation, and the K / V wiring is visible to the WAV sha",
  () => {
    const observed: RunComponent[] = [];
    let pipeline: Pipeline;
    // The originals are kept unbound on purpose: each patch re-invokes them with `.call(this)` and
    // the cleanup puts the same function objects back.
    // oxlint-disable-next-line typescript/unbound-method -- re-invoked with an explicit receiver, restored as-is
    const run = Session.prototype.run;
    // oxlint-disable-next-line typescript/unbound-method -- re-invoked with an explicit receiver, restored as-is
    const enqueue = Session.prototype.enqueue;

    beforeAll(async () => {
      const manifest = readManifest();
      const assets = await loadLocalAssets(manifest, QUANT);
      pipeline = await Pipeline.fromAssets(
        { manifest, assets },
        {
          model: MODEL,
          quant: QUANT,
          onRunDiagnostics: (component) => observed.push(component),
        },
      );
    });
    // Replaces the former `await using`: the pipeline is released after both its.
    afterAll(() => pipeline?.[Symbol.asyncDispose]());

    it("(a) run counts — on both the resident path and the host path, dit_context once and dit once per forward", async () => {
      const calls: { run: Calls; enqueue: Calls } = {
        run: { context: 0, dit: 0 },
        enqueue: { context: 0, dit: 0 },
      };
      const reset = (): void => {
        calls.run = { context: 0, dit: 0 };
        calls.enqueue = { context: 0, dit: 0 };
        observed.length = 0;
      };
      const count = (via: keyof typeof calls, inputs: RunInputs): void => {
        const kind = kindOf(inputs);
        if (kind !== undefined) calls[via][kind] += 1;
      };
      const observedCount = (component: RunComponent): number =>
        observed.filter((name) => name === component).length;
      try {
        Session.prototype.run = function (inputs, bindings, generation) {
          count("run", inputs);
          return run.call(this, inputs, bindings, generation);
        };
        Session.prototype.enqueue = function (inputs, options) {
          count("enqueue", inputs);
          return enqueue.call(this, inputs, options);
        };

        // The resident path (a device without timing, no onEvent) — both the preceding stage and
        // the body run through enqueue only.
        reset();
        const resident = await pipeline.generateLatent(REQUEST);
        assert(
          resident.forwards > 1,
          `forwards ${resident.forwards} cannot be told apart from "once"`,
        );
        assertEquals(
          calls.enqueue,
          { context: 1, dit: resident.forwards },
          "enqueue count on the resident path",
        );
        assertEquals(calls.run, { context: 0, dit: 0 }, "run was called on the resident path");
        // The observer (`onRunDiagnostics`) reports the same counts — the census splits components
        // by these reports.
        assertEquals(observedCount("dit-context"), 1);
        assertEquals(observedCount("dit"), resident.forwards);

        // The host path (onEvent subscription) — both the preceding stage and the body run through
        // run, and the stage opens and closes before dit.
        reset();
        const stages: string[] = [];
        const host = await pipeline.generateLatent({
          ...REQUEST,
          onEvent: (event) => {
            if (event.kind === "stage") stages.push(`${event.component}:${event.at}`);
          },
        });
        assertEquals(host.forwards, resident.forwards, "the 2 paths have different forward counts");
        assertEquals(calls.run, { context: 1, dit: host.forwards }, "run count on the host path");
        assertEquals(calls.enqueue, { context: 0, dit: 0 }, "enqueue was called on the host path");
        assertEquals(observedCount("dit-context"), 1);
        assertEquals(observedCount("dit"), host.forwards);
        const ditStart = stages.indexOf("dit:start");
        assertEquals(
          stages.slice(ditStart - 2, ditStart + 1),
          ["dit-context:start", "dit-context:end", "dit:start"],
          `the stage order differs (${stages.join(" / ")})`,
        );
        // The 2 paths' outputs are bit-identical (the MUST of `runDitLoopResident` — including the
        // difference in how stage 6' is copied).
        const bytes = (data: Float32Array<ArrayBuffer>): Uint8Array<ArrayBuffer> =>
          new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        assertEquals(
          bytes(host.data),
          bytes(resident.data),
          "the latents of the resident path and the host path are not bit-identical",
        );
      } finally {
        Session.prototype.run = run;
        Session.prototype.enqueue = enqueue;
      }
    });

    it("(b) fault injection — swapping context_k_0 and context_v_0 changes the WAV sha", async () => {
      const baseline = await wavSha(pipeline);
      // Without determinism, "the swap changed it" cannot be read, so two unpatched runs are
      // checked first.
      assertEquals(
        await wavSha(pipeline),
        baseline,
        "the WAVs of two unpatched generations differ",
      );
      let swapped = 0;
      try {
        Session.prototype.enqueue = function (inputs, options) {
          if (kindOf(inputs) !== "dit") return enqueue.call(this, inputs, options);
          if (!Object.hasOwn(inputs, "context_k_0") || !Object.hasOwn(inputs, "context_v_0")) {
            throw new Error(
              `dit's inputs have no context_k_0 / context_v_0 (${Object.keys(inputs).join(", ")})`,
            );
          }
          swapped += 1;
          return enqueue.call(
            this,
            {
              ...inputs,
              context_k_0: inputs["context_v_0"],
              context_v_0: inputs["context_k_0"],
            },
            options,
          );
        };
        const faulted = await wavSha(pipeline);
        assert(swapped > 0, "no dit enqueue was patched (it did not go through the resident path)");
        assertNotEquals(
          faulted,
          baseline,
          "swapping K / V does not change the WAV — the WAV sha256 check does not see the condition-side K/V wiring",
        );
      } finally {
        Session.prototype.enqueue = enqueue;
      }
    });
  },
);
