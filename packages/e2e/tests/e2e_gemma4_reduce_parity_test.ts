// **③PV reduction order A/B** on the real Gemma 4 E2B weights (K-12; the model side of ADR 0058
// decision 4 ②).
//
// [limitations](../../../docs/limitations.md) says "the `Pipeline` default is the KV-parallel
// reduction (`"parallel"`), whose reduction order differs from the runtime's reference path
// `"sequential"`, but **the gemma4 goldens are identical under both**". That claim is the basis for
// promoting K-12 to the default, so the check that guards it mechanically lives here.
//
// The existing checks do not guarantee it: the per-kernel A/B band
// (`gpu_state_attention_parallel_test.ts`) only checks that it stays inside the band, the census
// (`e2e_gemma4_pretrained_test.ts`) only checks **whether a key is present**, and the two goldens
// (greedy = through sequential / chat = through parallel) each go green against their own
// expectation, which does not establish "the two are identical".
//
// MUST: this check does not ride along with the census. The census requires `timestamp-query`, and
// Metal **advertises** that feature, so the census runs there and the device is lost once many
// query sets are alive at once (measured on M2 on 2026-09-03 — the `--diagnostics` section of
// `docs/known-issues.md`). Riding along would take the equality check down with it on a Mac
// (W-G7-4 / Pass2). This file requires no measurement, so its SKIP conditions are **only the asset
// and the GPU**.
//
// ## Asset and path
//
// The q4_0 QAT line's mirror `models/karume-gemma4-qat/` (E2B, its `i4-fast` seat — the `gemma4`
// pipeline's int4 seats since the regular line ships `bf16` alone, 2026-11-04), read directly
// through `nodeDirectory` (no HTTP, no persistent cache — ADR 0086). Without the mirror the cases
// SKIP explicitly.

import { isNotFound, readTextFile } from "@karume/test-support/fs";
import { assert, assertEquals } from "@std/assert";
import { nodeDirectory } from "@karume/hub/node";
import { beforeAll, describe, it } from "vite-plus/test";
import { Pipeline } from "../../models/gemma.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { allResidentPleBytesOfMirror } from "../../models/tests/helpers/ple-budget.ts";
import { GEMMA4_QAT_MIRROR, mirrorAvailable } from "../../models/tests/helpers/gemma-mirror.ts";

const MIRROR_DIR = GEMMA4_QAT_MIRROR;
/**
 * The seat these cases read: the q4_0 line's `i4-fast` (E2B is the mirror's default model), the
 * fused-parallel seat the regular line's `i4-fast` used to be. Named because the line's default is
 * the reference seat `i4`; the reducers below are passed explicitly either way.
 */
const INT4_SEAT = { quant: "i4-fast" } as const;

/** The assembly command, ready to paste when the cases SKIP. */
const ASSEMBLE_COMMAND = "cd tools/export-recipes && uv run python dist.py --pipeline gemma4-qat";

/**
 * The two reduction orders compared. **Both are passed explicitly** (the effective value when it
 * is omitted — quant declaration → reference path — is the census's job; this file only checks
 * that "the two paths produce the same sequence").
 */
const REDUCERS = ["parallel", "sequential"] as const;

/**
 * Acceptance cases. The prompt is the token sequence of the chat fixture as is (the source of
 * truth for the rendering is `gemma4_chat_test.ts`) — this check asks only about the **reduction
 * order**, not the rendering or the decoding, so it runs token sequence in → token sequence out
 * without a tokenizer. The Japanese case is mixed in to lengthen the decode steps.
 */
const CASES = [
  { fixture: "single-user", maxNewTokens: 24 },
  { fixture: "japanese", maxNewTokens: 24 },
] as const;

type ChatFixture = {
  readonly chat: { readonly name: string; readonly ids: number[] }[];
};

const fixture = JSON.parse(
  await readTextFile(
    new URL("../../models/tests/fixtures/gemma-text/gemma4-chat.json", import.meta.url),
  ),
) as ChatFixture;

const promptOf = (name: string): readonly number[] => {
  const found = fixture.chat.find((row) => row.name === name);
  assert(found !== undefined, `the fixture has no chat case '${name}'`);
  return found.ids;
};

const AVAILABLE = ((): boolean => {
  try {
    return mirrorAvailable(MIRROR_DIR);
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
})();

if (!AVAILABLE) {
  console.warn(
    `[karume] the distribution mirror models/karume-gemma4-qat/ is missing, so the ③PV reduction order A/B is SKIPPED. ` +
      `Assemble it: ${ASSEMBLE_COMMAND}`,
  );
}

/**
 * The generation result of one conversation (the id sequence of the `token` events and the stop
 * reason).
 */
type Run = {
  readonly ids: number[];
  readonly stop: { readonly reason: string; readonly tokens: number };
};

/**
 * Runs one case on one sequence. At **temperature 0** there is no sampling noise, so if the
 * sequence moves, the only cause is the reduction order.
 */
const runCase = async (
  pipeline: Pipeline,
  prompt: readonly number[],
  maxNewTokens: number,
): Promise<Run> => {
  const sequence = await pipeline.sequence();
  try {
    const stream = sequence.generate({ prompt, maxNewTokens, sampler: { temperature: 0 } });
    const ids: number[] = [];
    for await (const event of stream) {
      if (event.kind === "token") ids.push(event.id);
    }
    const stop = await stream.done;
    return { ids, stop: { reason: stop.reason, tokens: stop.tokens } };
  } finally {
    await sequence.dispose();
  }
};

describe.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  "gemma4: the ③PV reduction order (parallel / sequential) does not move a single token (real GPU)",
  () => {
    let maxResidentPleBytes: number;
    const runs: Run[][] = [];

    beforeAll(async () => {
      maxResidentPleBytes = await allResidentPleBytesOfMirror(MIRROR_DIR, INT4_SEAT);
    });

    for (const reduce of REDUCERS) {
      it(`stateAttentionReduce = ${reduce}`, async () => {
        const pipeline = await Pipeline.fromPretrained(nodeDirectory(MIRROR_DIR), {
          ...INT4_SEAT,
          maxResidentPleBytes,
          stateAttentionReduce: reduce,
        });
        try {
          const collected: Run[] = [];
          for (const { fixture: name, maxNewTokens } of CASES) {
            const started = performance.now();
            const run = await runCase(pipeline, promptOf(name), maxNewTokens);
            collected.push(run);
            console.log(
              `[e2e] gemma4 ${reduce} ${name}: ${run.ids.length} token / ` +
                `${JSON.stringify(run.stop)} / ${(performance.now() - started).toFixed(0)}ms`,
            );
          }
          runs.push(collected);
        } finally {
          await pipeline.dispose();
        }
      });
    }

    // The comparison was the tail of the former test body, after both steps; it runs after the
    // per-reducer its (Vitest runs a file's its in order and keeps going past a failure).
    it("the token sequences and stop reasons of both reduction orders match", () => {
      const [parallel, sequential] = runs;
      assertEquals(runs.length, REDUCERS.length, "did not run under both reduction orders");
      for (const [index, { fixture: name }] of CASES.entries()) {
        // Positive control — rules out the equality holding because "both were empty".
        assert(parallel[index].ids.length > 0, `${name}: generated not a single token`);
        assertEquals(
          parallel[index].ids,
          sequential[index].ids,
          `${name}: token sequence split by the reduction order`,
        );
        assertEquals(
          parallel[index].stop,
          sequential[index].stop,
          `${name}: stop reason split by the reduction order`,
        );
      }
    });
  },
);
