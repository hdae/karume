/**
 * The session A/B of `@karume/runtime/tune` on a real decode (opt-in: `KARUME_SESSION_AB=<dir>`):
 * Gemma 4 26B A4B from the distribution mirror at preset `i4`, under the exact expert cache as the
 * 2026-11-10 measurement of the candidate rule had it (64 slots per layer, every expert in the host
 * tier — research 2026-11-10 §8.1), two arms: `base` (no rule) and `rule` (the candidate rule
 * `candidate/gelu_mul` of `@karume/candidate-rules`, passed through the gemma pipeline's
 * experimental `fusionRules`). A run is one greedy generation of {@link TOKENS} tokens from one
 * fixed prompt with a fixed seed, and its output is the generated token ids, so the A/B's identity
 * is token identity. Two 26B pipelines do not fit this machine's GPU together, so every visit opens
 * and closes its pipeline (`openPerVisit`).
 *
 * NOTE: closing an arm ends with one empty submit and its fence ({@link releaseFence}). Dawn frees
 * the memory of destroyed buffers only once a submit made after the destroy has completed, so on a
 * device the caller keeps, a disposed 26B pipeline (about 7.2 GiB) stays allocated until the next
 * submit, and the next visit's build fails with VK_ERROR_OUT_OF_DEVICE_MEMORY before its first
 * submit (observed 2026-11-11, Dawn on Vulkan). The fence runs outside the timed span.
 *
 * It asserts token identity (every run of both arms equals the base's first) and the rule's hit
 * count in the decode plan (60: the 30 dense and the 30 routed-expert chains — the count of
 * e2e_gemma4_26b_external_rule_test.ts), never a verdict on times: the verdict and its numbers go
 * to the record and the log. The record (`karume-session-ab/1`, inside an envelope with the request)
 * is written under the directory the variable names.
 *
 * Each arm's counts come from `onRunDiagnostics`: the fusion counts and the planned dispatches
 * (`lastRunPipelines`) of the generation's first decode run. Under the expert cache the callback
 * does not change the decode path (it runs on the host sampler either way).
 *
 * MUST: never run it while other work uses the GPU (the times would measure the contention); run it
 * under `flock /tmp/karume-gpu.lock`. Without the variable, without the mirror's `26b` model or
 * without an adapter, the case SKIPs with the reason (ADR 0005).
 */

import { describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { mkdir, writeTextFile } from "@karume/test-support/fs";
import { acquireGpu, type GpuContext, type SessionOptions } from "@karume/runtime";
import { chatPrompt, Pipeline } from "@karume/models/gemma";
import {
  judgeSessionAb,
  runSessionAb,
  type SessionAbArm,
  type SessionAbReport,
} from "@karume/runtime/tune";
import { GELU_MUL_RULE, GELU_MUL_RULE_NAME } from "@karume/candidate-rules";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  GEMMA4_26B_IN_MIRROR,
  GEMMA4_QAT_MIRROR,
} from "../../models/tests/helpers/gemma-mirror.ts";
import { mirrorDirectory } from "../../models/tests/helpers/mirror.ts";

const DIRECTORY = process.env["KARUME_SESSION_AB"];

const skipReason =
  DIRECTORY === undefined || DIRECTORY === ""
    ? "KARUME_SESSION_AB is not set (opt-in: KARUME_SESSION_AB=<dir> writes the record there)"
    : !GEMMA4_26B_IN_MIRROR
      ? `${GEMMA4_QAT_MIRROR.pathname} has no 26b model`
      : !GPU_AVAILABLE
        ? "no GPU adapter"
        : undefined;
if (skipReason !== undefined) {
  console.warn(`[karume] the session A/B on Gemma 4 26B A4B SKIPs: ${skipReason}`);
}

/** The request the A/B runs (written into the record's envelope). */
const REQUEST = {
  model: "26b",
  quant: "i4",
  expertCache: { slots: 64, hostBytes: Number.MAX_SAFE_INTEGER },
  prompt:
    "Explain in detail how a refrigerator keeps food cold, step by step, for a curious teenager.",
  tokens: 96,
  sampler: { temperature: 0, seed: 0 },
  /** The prompt and the tokens fit in it; the expert cache's chunk rows (8) are below it. */
  capacity: 1024,
  rounds: 4,
  openPerVisit: true,
  minGain: 0.01,
} as const;
const TOKENS = REQUEST.tokens;

/** The 30 layers' dense and routed-expert chains, one hit each, in the decode plan. */
const RULE_HITS = 60;

/** The counters a run reported that are present (an external rule's key may be absent). */
const presentCounts = (counts: Readonly<Record<string, number | undefined>>) =>
  Object.fromEntries(
    Object.entries(counts).flatMap(([name, count]) =>
      count === undefined ? [] : [[name, count] as const],
    ),
  );

/**
 * One empty submit and its fence: lets Dawn free what the disposed pipeline destroyed (the NOTE at
 * the top). Bounded to the arm's close; the pipeline itself is not changed.
 */
const releaseFence = async (gpu: GpuContext): Promise<void> => {
  gpu.device.queue.submit([gpu.device.createCommandEncoder().finish()]);
  await gpu.device.queue.onSubmittedWorkDone();
};

/**
 * One arm: open builds the pipeline (the run's counts are read from its diagnostics callback),
 * a run is one generation returning its token ids, the report is the first decode run's counts,
 * close disposes the pipeline and lets the device free its memory (the GPU stays the test's).
 */
const pipelineArm = (
  name: string,
  gpu: GpuContext,
  fusionRules: SessionOptions["fusionRules"],
): SessionAbArm => {
  let pipeline: Pipeline | undefined;
  let counts: SessionAbReport | undefined;
  const opened = (): Pipeline => {
    if (pipeline === undefined) throw new Error(`arm ${name}: used before open`);
    return pipeline;
  };
  return {
    name,
    open: async () => {
      pipeline = await Pipeline.fromPretrained(mirrorDirectory(GEMMA4_QAT_MIRROR), {
        model: REQUEST.model,
        quant: REQUEST.quant,
        gpu,
        expertCache: REQUEST.expertCache,
        ...(fusionRules === undefined ? {} : { fusionRules }),
        onRunDiagnostics: (diagnostics, phase) => {
          if (phase.kind !== "decode" || counts !== undefined) return;
          counts = {
            fusions: presentCounts(diagnostics.lastRunFusions ?? {}),
            dispatches: (diagnostics.lastRunPipelines ?? []).reduce(
              (sum, { dispatchCount }) => sum + dispatchCount,
              0,
            ),
          };
        },
      });
    },
    run: async () => {
      counts = undefined;
      const current = opened();
      const sequence = await current.sequence({ capacity: REQUEST.capacity });
      try {
        const prompt = chatPrompt(current.tokenizer, [{ role: "user", content: REQUEST.prompt }]);
        const ids: number[] = [];
        const stream = sequence.generate({
          prompt,
          maxNewTokens: TOKENS,
          sampler: REQUEST.sampler,
        });
        for await (const event of stream) if (event.kind === "token") ids.push(event.id);
        await stream.done;
        return { outputs: { tokens: Int32Array.from(ids) } };
      } finally {
        await sequence.dispose();
      }
    },
    report: () => {
      if (counts === undefined) throw new Error(`arm ${name}: the run had no decode run`);
      return Promise.resolve(counts);
    },
    close: async () => {
      const current = pipeline;
      pipeline = undefined;
      await current?.dispose();
      await releaseFence(gpu);
    },
  };
};

describe.skipIf(skipReason !== undefined)("the session A/B on Gemma 4 26B A4B (opt-in)", () => {
  it("the candidate rule keeps every token and takes its 60 chains in the decode plan", async () => {
    const gpu = await acquireGpu();
    try {
      const record = await runSessionAb({
        arms: [pipelineArm("base", gpu, undefined), pipelineArm("rule", gpu, [GELU_MUL_RULE])],
        rounds: REQUEST.rounds,
        openPerVisit: REQUEST.openPerVisit,
        environment: {
          adapter: gpu.adapterInfo,
          userAgent: navigator.userAgent,
          backend: "vulkan",
          date: new Date().toISOString(),
        },
      });
      const judgements = judgeSessionAb(record, { minGain: REQUEST.minGain });
      const directory = DIRECTORY ?? "";
      await mkdir(directory, { recursive: true });
      const file = `${directory}/session-ab_${record.environment.date.replaceAll(":", "-")}.json`;
      await writeTextFile(
        file,
        `${JSON.stringify({ format: "karume-e2e-session-ab/1", request: REQUEST, record, judgements }, null, 2)}\n`,
      );
      console.log(`[session-ab] record: ${file}`);
      for (const arm of record.arms) {
        console.log(
          `[session-ab] ${arm.name}: hits ${arm.fusions?.[GELU_MUL_RULE_NAME] ?? 0}, dispatches ${arm.dispatches}, times ${JSON.stringify(arm.timesMs)}`,
        );
      }
      for (const judgement of judgements) {
        console.log(`[session-ab] ${judgement.arm}: ${judgement.verdict} — ${judgement.reason}`);
      }

      const [base, rule] = record.arms;
      assertEquals(
        [base.identity.worst.failures, rule.identity.worst.failures],
        [0, 0],
        "every run's tokens equal the base's first",
      );
      assertEquals(
        [base.fusions?.[GELU_MUL_RULE_NAME] ?? 0, rule.fusions?.[GELU_MUL_RULE_NAME]],
        [0, RULE_HITS],
        "the rule's hits in the decode plan",
      );
    } finally {
      gpu.destroy();
    }
  });
});
