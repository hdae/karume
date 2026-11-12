/**
 * The tiled key axis of the online attention against the form N gives, on this machine (opt-in:
 * `KARUME_ATTN_TILED_BENCH=<dir>`; perf-ledger K-106): `runAttentionRouteBench` of
 * `@karume/runtime/tune` with `onlineTiled` at the bench's shape set (`ATTENTION_BENCH_SHAPES`:
 * Wan2.2 TI2V-5B's self-attention at 33 and 121 frames and cross-attention at 121 frames, and
 * MOSS-Image's attention at 1024²), one call per shape with idle time between them for cooling
 * (`KARUME_ATTN_TILED_IDLE_SECONDS`, default 60). `KARUME_ATTN_TILED_SHAPES=<id>,<id>` narrows the
 * shapes.
 *
 * It needs no model asset: the shapes are sizes, and the bench fills them with seeded values. Each
 * per-shape record is written under the directory the variable names — the records are the second
 * input of the profile derivation (`deriveGeometryProfile`'s `attentionBench`, ADR 0117 addendum
 * 2026-11-12) — with a summary of the two ratios per shape: tiled ÷ online for the kernel alone (the
 * median of the per-round ratios, each round timing A B B A on one device, so a thermal drift within
 * the round cancels) and for kernel + producer per node from the arms' minimums (the statistic the
 * derivation reads), each with its gain (online ÷ tiled) and whether it clears ×1.05.
 *
 * It asserts what the run is the evidence for: GPU timestamps, the candidate ran at every shape
 * without a failed stage, and its identity block is identical (both producers on the same f32 Vᵀ,
 * both kernels on their outputs, every O and scale word equal); never a verdict on times (they go to
 * the summary and the log).
 *
 * MUST: never run it while other work uses the GPU (the times would measure the contention); run it
 * under `flock /tmp/karume-gpu.lock`. Without the variable or without an adapter, the case SKIPs with
 * the reason (ADR 0005).
 */

import { describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { mkdir, writeTextFile } from "@karume/test-support/fs";
import { acquireGpu } from "@karume/runtime";
import {
  ATTENTION_BENCH_SHAPES,
  type AttentionBenchReport,
  runAttentionRouteBench,
} from "@karume/runtime/tune";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";

const DIRECTORY = process.env["KARUME_ATTN_TILED_BENCH"];
const IDLE_SECONDS = Number(process.env["KARUME_ATTN_TILED_IDLE_SECONDS"] ?? "60");
const SHAPE_FILTER = (process.env["KARUME_ATTN_TILED_SHAPES"] ?? "")
  .split(",")
  .map((id) => id.trim())
  .filter((id) => id !== "");

const skipReason =
  DIRECTORY === undefined || DIRECTORY === ""
    ? "KARUME_ATTN_TILED_BENCH is not set (opt-in: KARUME_ATTN_TILED_BENCH=<dir> writes the records there)"
    : !Number.isInteger(IDLE_SECONDS) || IDLE_SECONDS < 0
      ? `KARUME_ATTN_TILED_IDLE_SECONDS is not a non-negative integer (${String(IDLE_SECONDS)})`
      : !GPU_AVAILABLE
        ? "no GPU adapter"
        : undefined;
if (skipReason !== undefined) {
  console.warn(`[karume] the tiled key axis bench SKIPs: ${skipReason}`);
}

const ROUNDS = 5;
/** The adoption line: a gain of ×1.05 (the derivation's default `minSpeedup`). */
const ADOPTION_GAIN = 1.05;

const sleep = (seconds: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, seconds * 1000));

const gainOf = (ratio: number | undefined): number | undefined =>
  ratio === undefined || !(ratio > 0) ? undefined : 1 / ratio;

describe.skipIf(skipReason !== undefined)(
  "the tiled key axis candidate on this machine (opt-in)",
  () => {
    it("the candidate runs at every shape of the bench's set with an identical O, timed in the same rounds as online", async () => {
      const directory = DIRECTORY ?? "";
      await mkdir(directory, { recursive: true });
      const shapes = ATTENTION_BENCH_SHAPES.filter(
        ({ id }) => SHAPE_FILTER.length === 0 || SHAPE_FILTER.includes(id),
      );
      assert(
        shapes.length > 0,
        `KARUME_ATTN_TILED_SHAPES names no shape (${SHAPE_FILTER.join(", ")})`,
      );
      const reports: AttentionBenchReport[] = [];
      for (const [index, shape] of shapes.entries()) {
        if (index > 0 && IDLE_SECONDS > 0) await sleep(IDLE_SECONDS);
        const gpu = await acquireGpu({ subgroups: "available", gpuTiming: true });
        try {
          const report = await runAttentionRouteBench(gpu, {
            shapes: [shape],
            rounds: ROUNDS,
            quantization: true,
            onlineTiled: true,
            onProgress: (progress) => {
              if (progress.kind === "status") console.log(`[attn-tiled] ${progress.message}`);
            },
          });
          await writeTextFile(
            `${directory}/attention-bench_tiled_${shape.id}.json`,
            `${JSON.stringify(report, null, 2)}\n`,
          );
          reports.push(report);
        } finally {
          gpu.destroy();
        }
      }

      const perShape = reports.flatMap((report) =>
        report.shapes.map((result) => {
          const tiled = result.onlineTiled;
          const kernel = tiled?.eligible === true ? tiled.kernel : undefined;
          const producer = result.quantization?.tiledVt;
          const kernelGain = gainOf(result.tiledVsOnline);
          const withProducerGain = gainOf(result.tiledVsOnlineWithProducer);
          return {
            shape: result.shape.id,
            rows: tiled?.eligible === true ? tiled.rows : undefined,
            reduce: result.online.eligible ? result.online.reduce : undefined,
            notRun: tiled?.eligible === false ? tiled.reason : undefined,
            identical: tiled?.eligible === true ? tiled.identity.identical : undefined,
            identity: tiled?.eligible === true ? tiled.identity : undefined,
            kernel: {
              tiledVsOnline: result.tiledVsOnline,
              gain: kernelGain,
              clearsAdoptionLine:
                kernelGain === undefined ? undefined : kernelGain >= ADOPTION_GAIN,
              perRoundRatios: kernel?.rounds?.map((round) => round.ratio),
              // The ratio of the arms' minimums, beside the median: a burst of slow passes in more
              // than half of the rounds moves the median, not the minimums
              ratioOfMinimums:
                kernel?.aMsPerDispatch === undefined || kernel.bMsPerDispatch === undefined
                  ? undefined
                  : kernel.bMsPerDispatch / kernel.aMsPerDispatch,
              onlineMs: kernel?.aMsPerDispatch,
              tiledMs: kernel?.bMsPerDispatch,
              reps: kernel?.reps,
              error: kernel?.error,
            },
            producer: {
              tiledVsToday: producer?.ratio,
              todayMs: producer?.aMsPerDispatch,
              tiledMs: producer?.bMsPerDispatch,
              perRoundRatios: producer?.rounds?.map((round) => round.ratio),
              error: producer?.error,
            },
            withProducer: {
              tiledVsOnline: result.tiledVsOnlineWithProducer,
              gain: withProducerGain,
              clearsAdoptionLine:
                withProducerGain === undefined ? undefined : withProducerGain >= ADOPTION_GAIN,
              onlineNodeMs: tiled?.eligible === true ? tiled.onlineNodeMs : undefined,
              tiledNodeMs: tiled?.eligible === true ? tiled.tiledNodeMs : undefined,
            },
          };
        }),
      );
      const [first] = reports;
      assert(first !== undefined);
      const summary = {
        format: "karume-e2e-attention-tiled-bench/1",
        adapter: first.adapter,
        userAgent: first.userAgent,
        timing: first.timing,
        dp4a: first.variants.dp4a.dot,
        geometryProfile: first.variants.geometryProfile,
        rounds: ROUNDS,
        idleSeconds: IDLE_SECONDS,
        adoptionGain: ADOPTION_GAIN,
        what: "tiled ÷ online: below 1 = the candidate faster; gain = online ÷ tiled; the profile derivation reads withProducer (kernel + producer per node from the arms' minimums)",
        shapes: perShape,
      };
      await writeTextFile(
        `${directory}/attn-tiled-summary.json`,
        `${JSON.stringify(summary, null, 2)}\n`,
      );
      for (const entry of perShape) {
        console.log(
          `[attn-tiled] ${entry.shape} (${entry.reduce ?? "?"}, ${entry.rows ?? "?"} rows): kernel tiled ÷ online ×${entry.kernel.tiledVsOnline?.toFixed(4) ?? "?"} (of minimums ×${entry.kernel.ratioOfMinimums?.toFixed(4) ?? "?"}, gain ×${entry.kernel.gain?.toFixed(4) ?? "?"}, rounds ${entry.kernel.perRoundRatios?.map((ratio) => ratio?.toFixed(4) ?? "-").join(" ") ?? "?"}), kernel + producer ×${entry.withProducer.tiledVsOnline?.toFixed(4) ?? "?"} (gain ×${entry.withProducer.gain?.toFixed(4) ?? "?"}), producer tiled ÷ today ×${entry.producer.tiledVsToday?.toFixed(4) ?? "?"}, identical ${String(entry.identical)}`,
        );
      }

      for (const report of reports) assertEquals(report.timing.unit, "ns", "GPU timestamps");
      for (const entry of perShape) {
        assertEquals(entry.notRun, undefined, `${entry.shape}: the candidate did not run`);
        assertEquals(entry.kernel.error, undefined, `${entry.shape}: kernel pair failed`);
        assertEquals(entry.producer.error, undefined, `${entry.shape}: producer pair failed`);
        assertEquals(
          entry.identical,
          true,
          `${entry.shape}: identity ${JSON.stringify(entry.identity)}`,
        );
        assert(
          entry.kernel.tiledVsOnline !== undefined &&
            entry.withProducer.tiledVsOnline !== undefined,
          `${entry.shape}: a ratio is missing`,
        );
      }
    });
  },
);
