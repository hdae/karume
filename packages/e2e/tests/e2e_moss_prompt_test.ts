/**
 * MOSS-Image from prompt strings on a real GPU — the prompt form of `generateMoss` (ADR 0138
 * decision 9): the chat template and the tokenizer on the host, the Qwen3.5 text encoder (ADR 0137)
 * opened, run for the positive and the negative prompt and closed, then `dit_text` → the DiT pair →
 * the VAE. Over the distribution's i8 text encoder (exported as the series
 * `qwen3.5-9b-text-moss-image-sft-i8-dyn`) with its tokenizer asset, the i8 DiT and the f32 VAE,
 * with the reference preset `f32+dit8`.
 *
 * ## Checks
 *
 * 1. **Token ids** (host, no GPU — MUST): for every fixed prompt of upstream's text-embedding asset
 *    (`moss.text_embeds`), the acceptance check's ids are exactly the ids upstream's
 *    `encode_prompt` fed its encoder (the asset's metadata — taken with a forward pre-hook, not
 *    re-derived).
 * 2. **The text stage on the fixed prompts** (GPU): `encodeMossPrompts` once per positive prompt,
 *    each with the empty negative. Asserted: the rows of `fridge-note` and of the negative are bit
 *    for bit this environment's rows of the encoder lane (`fixtures/references/qwen3_5.json` — the
 *    rows `e2e_qwen3_5_encoder_test.ts` holds within its band of Karume's CPU reference), so the
 *    stage feeds the DiT exactly the verified encoder's output; a missing row there fails the case
 *    (run the `models:qwen3_5` lane with `KARUME_REFERENCE=write` first). Recorded, not asserted:
 *    each prompt's distance from upstream's bf16 embeddings (relative L2 `‖K − U‖₂ / ‖U‖₂`,
 *    max |Δ|, max |Δ| ÷ max |U|, the smallest row cosine — the metrics of the recipe's record
 *    `full.json` of the export check of 2026-10-31, kept as the golden directory
 *    `qwen3_5-export-2026-10-31`), and the stage's time and memory. The
 *    distance is the i8 export's (about 4 % relative L2 in that record), not a band: whether it
 *    matters is the image A/B's question (ADR 0137 decision 5).
 * 3. **Prompt → image** (GPU): 256², 2 steps, `fridge-note` with the empty negative, guidance 3,
 *    shift 5, seed 42. The Sessions open and close as text_encoder → dit_text → the DiT pair → the
 *    VAE (each stage's Sessions disposed — completed — before the next stage opens), the image is
 *    finite, and its sha256 row (ADR 0106 — `fixtures/references/moss.json`, its own ID: the
 *    embeddings are Karume's i8 encoder's, not upstream's, so the row is another route's) pins it.
 *    MUST: a writing run (`KARUME_REFERENCE=write`) settles the row only after check 2's assertion
 *    held earlier in the same run (Vitest runs a file's tests in declaration order) — a row is
 *    written the moment it is settled, and shape and finiteness alone would pin a wrong text stage.
 *
 * Results and the measured numbers go to `outputs/verify/<environment key>/<date>_moss-prompt/`
 * (safe to delete); the numbers of 2026-10-31 are in `docs/research/2026-10-31-moss-text-stage.md`.
 *
 * Without a GPU or an asset the file skips explicitly, naming the command that gets it (ADR 0005).
 *
 * ## Where the assets come from (ADR 0139)
 *
 * The containers (text encoder, DiT, VAE) and the tokenizer are read from the distribution mirror
 * `models/karume-moss-image/` through its manifest; upstream's text-embedding asset from the
 * test-asset root (`goldenDir("moss-image-sft-text-embeds")`, fetched with
 * `node tools/test-assets/fetch.ts`). The encoder rows' case IDs keep the series name verbatim
 * (ADR 0139 decision 7).
 */

import { afterAll, beforeAll, describe, expect, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import type { GpuContext, Session, SessionDiagnostics } from "@karume/runtime";
import type { ModelComponent } from "../../models/src/hub/components.ts";
import {
  mossDitContract,
  mossDitGraphContract,
  readMossDitAssets,
} from "../../models/src/moss/dit-contract.ts";
import {
  generateMoss,
  type MossGenerateRequest,
  type MossGenerateState,
  planMossGenerate,
} from "../../models/src/moss/generate.ts";
import { MOSS_REFERENCE_PRESET, mossPreset } from "../../models/src/moss/preset.ts";
import {
  encodeMossPrompts,
  MOSS_TEXT_ENCODER,
  mossTextEncoderStageNeed,
  type MossTextEncoderState,
} from "../../models/src/moss/text-stage.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  MOSS_FETCH_GOLDENS,
  MOSS_GET_MIRROR,
  MOSS_TEXT_EMBEDS,
  testAssetsRunnable,
  mossMirrorPresent,
  openMossDit,
  syntheticTrainingGrid,
} from "../../models/tests/helpers/moss-dit.ts";
import {
  MOSS_TEXT_SERIES_NAME,
  type MossUpstreamPrompt,
  openMossTextEncoder,
  openMossTokenizer,
  readMossUpstreamPrompts,
} from "../../models/tests/helpers/moss-text.ts";
import { openMossVae } from "../../models/tests/helpers/moss-vae.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import { verifyRunningGpu } from "../../runtime/tests/helpers/environment.ts";
import {
  f32ArtifactBytes,
  openReferences,
  type ReferenceSettlement,
  referenceMismatchMessage,
  registerReferenceGate,
  settleOrObserve,
  sha256Hex,
} from "../../runtime/tests/helpers/reference.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";

/** This file's sha256 row (`moss.json` — shared with the other MOSS e2e files, one row each). */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/moss.json", import.meta.url),
);
/** The encoder lane's rows (read only — `e2e_qwen3_5_encoder_test.ts` writes them). */
const encoderReferences = openReferences(
  new URL("../../models/tests/fixtures/references/qwen3_5.json", import.meta.url),
);
/** Artifacts and verdicts (`outputs/verify/<environment key>/<date>_moss-prompt/`). */
const results = openResults("moss-prompt");

/** The mirror carries every container and the tokenizer this file reads. */
const MIRROR_PRESENT = mossMirrorPresent();
/** Present, or absent from a complete fetch of the pin (then its read FAILs naming it). */
const EMBEDS_RUNNABLE = testAssetsRunnable([MOSS_TEXT_EMBEDS]);
const RUNNABLE = GPU_AVAILABLE && MIRROR_PRESENT && EMBEDS_RUNNABLE;

const skipped = (ok: boolean, what: string, command: string): string =>
  ok ? "" : ` (SKIP: no ${what} — ${command})`;

const MIRROR_SKIP = skipped(MIRROR_PRESENT, "mirror models/karume-moss-image/", MOSS_GET_MIRROR);
const EMBEDS_SKIP = skipped(
  EMBEDS_RUNNABLE,
  `upstream text embeddings ${MOSS_TEXT_EMBEDS.pathname}`,
  MOSS_FETCH_GOLDENS,
);

/** The generation of check 3 (the other MOSS e2e files' seed path, from prompt strings). */
const PROMPT = "fridge-note";
const SIZE = { width: 256, height: 256 } as const;
const STEPS = 2;
const GUIDANCE = 3;
const SHIFT = 5;
const SEED = 42;
const PRESET = mossPreset(MOSS_REFERENCE_PRESET);

/** Check 3's row: the preset first (ADR 0138 decision 11), the text route after the prompt. */
const IMAGE_CASE_ID =
  `${PRESET.name}-${STEPS}step-${PROMPT}-qwen3_5-i8-seed${SEED}-` +
  `${SIZE.width}x${SIZE.height}-g${GUIDANCE}-shift${SHIFT}`;
const CASE_IDS: readonly string[] = [IMAGE_CASE_ID];

/** The encoder lane's row of a prompt (`e2e_qwen3_5_encoder_test.ts`'s case ID). */
const encoderCaseId = (name: string): string => `${MOSS_TEXT_SERIES_NAME}/${name}`;
/** The prompts whose rows the encoder lane pins (the others are recorded only). */
const PINNED = ["fridge-note", "negative"] as const;

/** Distance of Karume's rows from upstream's (f64 — the recipe's record's metrics). */
type Distance = {
  readonly relL2: number;
  readonly maxAbs: number;
  readonly ratio: number;
  readonly rowCosineMin: number;
};

const distance = (
  actual: Float32Array<ArrayBuffer>,
  expected: Float32Array<ArrayBuffer>,
  width: number,
): Distance => {
  assertEquals(actual.length, expected.length, "element count");
  let difference = 0;
  let norm = 0;
  let maxAbs = 0;
  let scale = 0;
  let rowCosineMin = 1;
  for (let row = 0; row < actual.length / width; row += 1) {
    let dot = 0;
    let actualNorm = 0;
    let expectedNorm = 0;
    for (let column = 0; column < width; column += 1) {
      const a = actual[row * width + column];
      const e = expected[row * width + column];
      difference += (a - e) ** 2;
      norm += e * e;
      maxAbs = Math.max(maxAbs, Math.abs(a - e));
      scale = Math.max(scale, Math.abs(e));
      dot += a * e;
      actualNorm += a * a;
      expectedNorm += e * e;
    }
    rowCosineMin = Math.min(rowCosineMin, dot / Math.sqrt(actualNorm * expectedNorm));
  }
  return {
    relL2: Math.sqrt(difference / norm),
    maxAbs,
    ratio: maxAbs / scale,
    rowCosineMin,
  };
};

const formatDistance = ({ relL2, maxAbs, ratio, rowCosineMin }: Distance): string =>
  `rel L2 ${relL2.toExponential(3)}, max |Δ| ${maxAbs.toExponential(3)}, ratio ` +
  `${ratio.toExponential(3)}, min row cosine ${rowCosineMin.toFixed(6)}`;

const countNonFinite = (values: Float32Array): number =>
  values.reduce((count, value) => (Number.isFinite(value) ? count : count + 1), 0);

/** One Session event (a monotonic instant; per run its elapsed time; the diagnostics at dispose). */
type SessionEvent = {
  readonly event: string;
  readonly at: number;
  readonly ms?: number;
  readonly diagnostics?: SessionDiagnostics;
};

/**
 * A component whose Sessions log `open <name>` (once built — the elapsed build), `run <name>` (each
 * run's elapsed time) and `dispose <name>` (once the dispose has completed, with the diagnostics
 * taken just before it).
 */
const observed = (
  component: ModelComponent,
  name: string,
  events: SessionEvent[],
): ModelComponent => ({
  ...component,
  createSession: async (gpu, options) => {
    const started = performance.now();
    const session: Session = await component.createSession(gpu, options);
    const at = performance.now();
    events.push({ event: `open ${name}`, at, ms: at - started });
    const run = session.run.bind(session);
    Object.defineProperty(session, "run", {
      configurable: true,
      value: async (...args: Parameters<Session["run"]>) => {
        const begun = performance.now();
        const outputs = await run(...args);
        const ended = performance.now();
        events.push({ event: `run ${name}`, at: ended, ms: ended - begun });
        return outputs;
      },
    });
    const dispose = session.dispose.bind(session);
    Object.defineProperty(session, "dispose", {
      configurable: true,
      value: async () => {
        const diagnostics = session.diagnostics();
        const begun = performance.now();
        await dispose();
        const ended = performance.now();
        events.push({ event: `dispose ${name}`, at: ended, ms: ended - begun, diagnostics });
      },
    });
    return session;
  },
});

/** The label of `fitsHeadroom`'s trial allocations (`packages/runtime/src/gpu/headroom.ts`). */
const PROBE_LABEL = "karume-headroom-probe";

/**
 * Runs `body` with `device.createBuffer` wrapped (restored afterwards): the peak bytes of the live
 * WebGPU buffers that are not headroom probes, and what is still held when `body` returns — the
 * buffers' own sizes (no driver overhead).
 */
const withHeldPeak = async <T>(
  device: GPUDevice,
  body: () => Promise<T>,
): Promise<{ readonly result: T; readonly peak: number; readonly after: number }> => {
  const create = device.createBuffer.bind(device);
  let held = 0;
  let peak = 0;
  device.createBuffer = (descriptor: GPUBufferDescriptor): GPUBuffer => {
    const buffer = create(descriptor);
    if (descriptor.label === PROBE_LABEL) return buffer;
    held += descriptor.size;
    peak = Math.max(peak, held);
    const destroy = buffer.destroy.bind(buffer);
    let destroyed = false;
    Object.defineProperty(buffer, "destroy", {
      configurable: true,
      value: () => {
        if (!destroyed) held -= descriptor.size;
        destroyed = true;
        destroy();
      },
    });
    return buffer;
  };
  try {
    const result = await body();
    return { result, peak, after: held };
  } finally {
    device.createBuffer = create;
  }
};

const MIB = 2 ** 20;
const mib = (bytes: number): string => `${(bytes / MIB).toFixed(0)} MiB`;

/** A request of the prompt form (the generation of check 3 with `prompt`'s text). */
const promptRequest = (prompt: string, negativePrompt: string): MossGenerateRequest => ({
  ...SIZE,
  prompt,
  negativePrompt,
  steps: STEPS,
  guidance: GUIDANCE,
  shift: SHIFT,
  seed: SEED,
});

describe.skipIf(!MIRROR_PRESENT || !EMBEDS_RUNNABLE)(
  `MOSS-Image prompt form: token ids${MIRROR_SKIP}${EMBEDS_SKIP}`,
  () => {
    test("the acceptance check gives exactly the ids upstream's encode_prompt fed its encoder, for every fixed prompt", async () => {
      const tokenizer = await openMossTokenizer();
      const prompts = await readMossUpstreamPrompts();
      const [negative] = prompts.filter(({ role }) => role === "negative");
      const positives = prompts.filter(({ role }) => role === "positive");
      assert(negative !== undefined && positives.length === 3, "the asset's four fixed prompts");
      for (const positive of positives) {
        const plan = planMossGenerate(promptRequest(positive.text, negative.text), {
          dit: { latentChannels: 128, trainingGrid: syntheticTrainingGrid(1000) },
          vae: { bytesPerCell: 49_152, spatialScale: 8 },
          limits: { maxStorageBufferBindingSize: 2 ** 31 - 4, maxBufferSize: 2 ** 31 - 4 },
          text: { tokenizer },
        });

        assert(plan.text === "prompts");
        assertEquals([...plan.prompts.positive], [...positive.ids], positive.name);
        assertEquals([...(plan.prompts.negative ?? [])], [...negative.ids], negative.name);
      }
    });
  },
);

describe.skipIf(!RUNNABLE)(
  `MOSS-Image prompt form on the GPU${skipped(GPU_AVAILABLE, "GPU adapter", "a WebGPU device")}${MIRROR_SKIP}${EMBEDS_SKIP}`,
  () => {
    let gpu: GpuContext | undefined;
    let base: Omit<MossGenerateState, "gpu" | "text"> | undefined;
    let text: (MossTextEncoderState & { readonly component: ModelComponent }) | undefined;
    let upstream: ReadonlyMap<string, MossUpstreamPrompt> | undefined;
    /** Check 2's assertion held in this run (the condition of writing check 3's row). */
    let textStageVerified = false;

    beforeAll(async () => {
      const series = await openMossDit();
      const assets = await readMossDitAssets(series.components.dit);
      base = {
        preset: PRESET,
        dit: {
          components: series.components,
          contract: mossDitContract(
            mossDitGraphContract(series.components, "MOSS-Image"),
            assets.ropeBase,
            assets.trainingGrid,
            "MOSS-Image",
          ),
        },
        vae: await openMossVae(),
      };
      text = await openMossTextEncoder();
      upstream = new Map((await readMossUpstreamPrompts()).map((entry) => [entry.name, entry]));
      // The rows may be written and compared with the encoder lane's: the adapter that runs must
      // be the one the key names, acquired as that lane acquires it (no feature asked for).
      gpu = verifyRunningGpu(await acquireTestGpu());
    });
    afterAll(async () => {
      if (gpu === undefined) return;
      await settleReleases(gpu);
      gpu.destroy();
    });
    const opened = () => {
      assert(
        gpu !== undefined && base !== undefined && text !== undefined && upstream !== undefined,
      );
      return { gpu, base, text, upstream };
    };
    const upstreamPrompt = (name: string): MossUpstreamPrompt => {
      const found = opened().upstream.get(name);
      if (found === undefined) throw new Error(`upstream's asset has no prompt '${name}'`);
      return found;
    };

    it("the text stage on the fixed prompts: the pinned prompts' rows are the encoder lane's bit for bit; the distance from upstream, the time and the memory are recorded", async () => {
      const { gpu: device, text: series } = opened();
      const negative = upstreamPrompt("negative");
      const positives = ["fridge-note", "guide-whiteboard", "guide-fridge-note"].map(
        upstreamPrompt,
      );
      const failures: string[] = [];
      for (const positive of positives) {
        await runRecordedCase(
          results,
          { id: `text-stage/${positive.name}` },
          async ({ comparisons }) => {
            const prompts = {
              positive: Int32Array.from(positive.ids),
              negative: Int32Array.from(negative.ids),
            };
            const need = mossTextEncoderStageNeed({ ...series, limits: device.limits }, prompts);
            const events: SessionEvent[] = [];
            const component = observed(series.component, MOSS_TEXT_ENCODER, events);
            const started = performance.now();
            const {
              result: texts,
              peak,
              after,
            } = await withHeldPeak(device.device, () =>
              encodeMossPrompts({ gpu: device, component, contract: series.contract }, prompts),
            );
            const stageMs = performance.now() - started;
            const lines: string[] = [];
            for (const [entry, rows] of [
              [positive, texts.positive.embeddings.data],
              [negative, texts.negative?.embeddings.data],
            ] as const) {
              assert(rows !== undefined, `${entry.name}: no rows`);
              expect(rows.length).toBe(entry.ids.length * 4096);
              const measured = distance(rows, entry.embeddings.data, 4096);
              comparisons.push({
                output: `${entry.name} (T = ${entry.ids.length})`,
                reference: "upstream-bf16 (moss.text_embeds)",
                practical: `${MOSS_TEXT_SERIES_NAME} (GPU)`,
                relRms: measured.relL2,
                maxAbs: measured.maxAbs,
                band: undefined,
              });
              lines.push(
                `${entry.name} (T = ${entry.ids.length}) vs upstream: ${formatDistance(measured)}`,
              );
              if (PINNED.some((name) => name === entry.name)) {
                const id = encoderCaseId(entry.name);
                const expected = encoderReferences.lookup(id);
                const actual = await sha256Hex(
                  f32ArtifactBytes("last_hidden_state", [1, entry.ids.length, 4096], rows),
                );
                if (expected === undefined) {
                  failures.push(`${id}: no row of this environment in qwen3_5.json`);
                } else if (actual !== expected) {
                  failures.push(
                    `${id}: the stage's rows ${actual} are not the encoder lane's ${expected}`,
                  );
                }
              }
            }
            const at = (event: string) => events.filter((entry) => entry.event === event);
            const [open] = at(`open ${MOSS_TEXT_ENCODER}`);
            const runs = at(`run ${MOSS_TEXT_ENCODER}`);
            const [disposed] = at(`dispose ${MOSS_TEXT_ENCODER}`);
            assert(open !== undefined && disposed !== undefined && runs.length === 2);
            const diagnostics = disposed.diagnostics;
            assert(diagnostics !== undefined);
            lines.push(
              `stage ${(stageMs / 1000).toFixed(2)} s: Session build ${(open.ms ?? 0).toFixed(0)} ms,` +
                ` encodings ${runs.map(({ ms }) => `${(ms ?? 0).toFixed(1)} ms`).join(" / ")},` +
                ` dispose ${(disposed.ms ?? 0).toFixed(1)} ms`,
              `memory: need ${need} B (${mib(need)}), peak held ${peak} B (${mib(peak)}), held after` +
                ` ${after} B; weights ${diagnostics.weights.allocatedBytes} B, plan backing` +
                ` ${diagnostics.planBacking.residentBytes} B, input buffers` +
                ` ${diagnostics.planBacking.inputBytes} B`,
            );
            console.info(
              `[moss-prompt] text stage ${positive.name} + negative\n  ${lines.join("\n  ")}`,
            );
            // Nothing of the stage stays on the device once it returns (module doc, check 2).
            expect(after).toBe(0);
            expect(peak).toBeLessThanOrEqual(need);
            return { status: "pass", note: lines.join(" | ") };
          },
        );
        await settleReleases(device);
      }
      assert(failures.length === 0, failures.join("\n"));
      textStageVerified = true;
    });

    it(`${IMAGE_CASE_ID}: the stages open and close in order from the text encoder to the VAE, a finite image, pinned by its sha256 row`, async () => {
      const { gpu: device, base: series, text: textSeries } = opened();
      const events: SessionEvent[] = [];
      const state: MossGenerateState = {
        gpu: device,
        ...series,
        dit: {
          ...series.dit,
          components: {
            dit: observed(series.dit.components.dit, "dit", events),
            ditSkip22: observed(series.dit.components.ditSkip22, "dit_skip22", events),
            ditText: observed(series.dit.components.ditText, "dit_text", events),
          },
        },
        vae: {
          ...series.vae,
          components: {
            pre: observed(series.vae.components.pre, "vae_decoder_pre", events),
            attn: observed(series.vae.components.attn, "vae_decoder_attn", events),
            post: observed(series.vae.components.post, "vae_decoder_post", events),
          },
        },
        text: {
          ...textSeries,
          component: observed(textSeries.component, MOSS_TEXT_ENCODER, events),
        },
      };
      let settlement: ReferenceSettlement | undefined;
      await runRecordedCase(results, { id: IMAGE_CASE_ID }, async () => {
        const started = performance.now();
        const generated = await generateMoss(
          state,
          promptRequest(upstreamPrompt(PROMPT).text, upstreamPrompt("negative").text),
        );
        const seconds = (performance.now() - started) / 1000;
        await settleReleases(device);

        expect(
          events.filter(({ event }) => !event.startsWith("run ")).map(({ event }) => event),
        ).toEqual([
          `open ${MOSS_TEXT_ENCODER}`,
          `dispose ${MOSS_TEXT_ENCODER}`,
          "open dit_text",
          "dispose dit_text",
          "open dit",
          "open dit_skip22",
          "dispose dit_skip22",
          "dispose dit",
          "open vae_decoder_pre",
          "open vae_decoder_attn",
          "open vae_decoder_post",
          "dispose vae_decoder_pre",
          "dispose vae_decoder_attn",
          "dispose vae_decoder_post",
        ]);
        expect(generated.image.length).toBe(3 * SIZE.width * SIZE.height);
        expect(countNonFinite(generated.image)).toBe(0);
        if (references.mode !== undefined && !textStageVerified) {
          throw new Error(
            `${IMAGE_CASE_ID}: KARUME_REFERENCE=${references.mode} writes this row only after the` +
              " text stage's rows matched the encoder lane's earlier in the same run (rerun the whole" +
              " file, with the models:qwen3_5 rows of this environment present)",
          );
        }
        const instant = (event: string): number => {
          const found = events.find((entry) => entry.event === event);
          assert(found !== undefined, `no event '${event}'`);
          return found.at;
        };
        const textStage =
          instant(`dispose ${MOSS_TEXT_ENCODER}`) - instant(`open ${MOSS_TEXT_ENCODER}`);
        const note =
          `generate ${seconds.toFixed(1)} s; the text encoder open → disposed ${(textStage / 1000).toFixed(1)} s` +
          ` (+ its Session build ${((events[0]?.ms ?? 0) / 1000).toFixed(1)} s before)`;
        console.info(`[moss-prompt] ${IMAGE_CASE_ID}: ${note}`);
        const outcome = await settleOrObserve(references, results, {
          id: IMAGE_CASE_ID,
          artifact: `${IMAGE_CASE_ID}.safetensors`,
          bytes: f32ArtifactBytes("image", [3, SIZE.height, SIZE.width], generated.image),
        });
        settlement = outcome.settlement;
        return { ...outcome.fields, note };
      });
      if (settlement?.check.status === "fail") {
        throw new Error(referenceMismatchMessage(IMAGE_CASE_ID, settlement, references));
      }
    });
  },
);

if (RUNNABLE) references.warnMissing(CASE_IDS);
// A missing row of this environment is not a silent pass (ADR 0005's gatekeeper form).
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
