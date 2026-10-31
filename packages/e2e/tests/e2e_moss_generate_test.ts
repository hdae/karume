/**
 * MOSS-Image text-to-image end to end on a real GPU — the generate path (`generateMoss`: the one
 * acceptance check, the first headroom check, `dit_text` → the DiT pair with the sampler → the VAE —
 * ADR 0138 decisions 5, 9 and 11, stage 4) over the i8 DiT series and
 * the f32 VAE series, with the reference preset `f32+dit8` (i8 weights × f32 compute, no session
 * knob) unless a case names the practical preset `f32+dit8-a8-attn8-online`
 * (`../../models/src/moss/preset.ts`; its self A/B check is `e2e_moss_ab_test.ts`). The device asks
 * for subgroups when available (the practical preset's online kernel takes its subgroup form; no
 * kernel of the reference preset reads them). Sizes are written width × height here (the recipe's records name them height × width — `dit-256x384` is
 * 384 px wide).
 *
 * ## Default run
 *
 * - **Order and lifetimes** (ADR 0138 decision 5 — asserted here, not left to the modules'
 *   comments): the Sessions open and close as `dit_text` → closed → `dit` + `dit_skip22`
 *   → both closed → the three VAE graphs → closed, so every DiT Session is disposed before the first
 *   VAE Session opens. A dispose is logged when it has **completed** (a dispose started but not
 *   awaited would log after the next stage's open). Every resident tensor the generation made (the
 *   text contexts, the VAE's hand-over tensors) is disposed at the end, and each Session built its
 *   plan backing at most once (`buildCount` 1 for the DiT pair and the VAE — their runs repeat a
 *   shape; 0 for `dit_text`, whose two runs have two text lengths and the run path builds a backing
 *   only on a repeated shape). Guidance 1 opens no `dit_skip22` and still pins its image. A failure
 *   inside the DiT loop has closed every Session and every text context when generate rejects, and
 *   opens no VAE.
 * - **The first headroom check** (ADR 0138 decision 5): with the VAE's need — or the pair's — made
 *   larger than any device, generate refuses before any Session opens, naming the first check and
 *   the stage. Without the first check `dit_text` and the whole DiT would run before the VAE stage
 *   (or the pair) refused. The bytes it checks are the largest stage's own need, not the sum of the
 *   three stages (the stages never hold their Sessions together).
 * - **sha256 rows** (ADR 0106, ADR 0138 decision 11 — `fixtures/references/moss.json`, one row per
 *   environment key): the clamped f32 image `[3, H, W]` (an F32 safetensors without metadata —
 *   `f32ArtifactBytes`) of the seed path at 256² and 384×256, 2 steps, `fridge-note` with the empty
 *   negative, guidance 3, shift 5, seed 42, and at 256² guidance 1 without a negative, for the
 *   reference preset; the practical preset's at 256² and 384×256, guidance 3 ({@link SEED_CASES} —
 *   a row's ID starts with its preset's name). A row is settled only after the run's own checks held (shape, finite).
 *   This machine's rows are written with `KARUME_REFERENCE=write` in this lane; another machine's
 *   rows are never touched. MUST: a writing run settles a row only after both sizes of the upstream
 *   parity below (`KARUME_UPSTREAM_PARITY=1` with its goldens — {@link PARITY_IDS}) passed earlier
 *   in the same run — the parity is declared before the row-settling tests (Vitest runs a file's
 *   tests in declaration order), and otherwise the row's test fails before writing, naming the
 *   parity cases that did not pass (`assertMossRowWritable`): a row is written the moment it is
 *   settled, and shape and finiteness alone would pin a finite but wrong build. Compare mode reads
 *   no parity.
 *
 * ## Opt-in upstream parity (`KARUME_UPSTREAM_PARITY=1` — ADR 0129 addendum (2))
 *
 * The reference is upstream's own `pipe(...)` on CPU in f32 with the DiT's linear weights rounded
 * to the series' i8 values and checked bit-identical to the containers, and upstream's f32 VAE
 * (`tools/export-recipes/moss/few_step_ref.py` — the Wan2.2 precedent `wan.ti2v_few_step_ref`): the
 * GPU and the reference run the same weights, so the band measures the GPU's f32 arithmetic and the
 * generate path's wiring, not the i8 rounding of the weights. The goldens record the part-0
 * sha256 of the six containers, the embedding asset's sha256, the token counts and the schedule
 * upstream ran; a golden whose containers, embeddings, token counts or σ / timestep columns differ
 * from this run's is refused as stale.
 *
 * Per size (256² and 384×256): three band cases and two acceptance cases ({@link PARITY_CASES} —
 * other seeds and prompts than the band cases), the reference's initial latents (torch's `randn`)
 * injected through `latents`, 2 steps, guidance 3, shift 5. Observation points: the final latents
 * (the VAE's input) and the clamped image. Metric m = max |GPU − reference| ÷ max |reference|. The
 * band per size and observation point is the worst band case × 5, rounded up to two significant
 * digits (ADR 0118 decision 8 — an f32 reference, the ratio, about 5 × the measured; the 256² band
 * is not carried to 384×256), and the acceptance cases must hold it ({@link BANDS} has the
 * derivation). MUST: the bands, the cases and the metric are not changed after seeing the
 * acceptance results; an acceptance case outside its band is investigated, not absorbed.
 *
 * Faults (the first acceptance case of each size; each must land outside 2 × the band — ADR 0118's
 * addendum): the two branches' prompts swapped, guidance 3 → 3.3, shift 5 → 3 (another 2-step
 * schedule: σ₁ snaps to grid index 991 instead of 976), another seed's initial latents, and the
 * subtle one — the first step at the next training-grid point (index 1: σ 0.999 and its timestep
 * row instead of σ 1; the second step unchanged), the precedent of Wan's timestep + 1 — and a
 * deliberately small one, guidance 3 → 3.003 (a 0.1 % error of the scale), the fault closest to the
 * band: it shows the band would catch a wiring slip far smaller than the others.
 *
 * The finer wiring of the sampler — the σ / timestep indexing, the guidance value, the step order —
 * is checked bit for bit on the CPU (`moss_generate_host_test.ts`: the generate path's own loop on
 * upstream's recorded DiT outputs reproduces upstream's latents).
 *
 * ## Opt-in 1024² × 50 steps (`KARUME_MOSS_FULL_PIPELINE=1`)
 *
 * Upstream's default request (1024², 50 steps, guidance 3, shift 5) on the seed path, once per
 * preset: it runs to completion on this device with a finite image, and its sha256 row is settled (registered with the
 * reference gate only under the opt-in, so the default lane's gate does not count it). The test logs
 * each stage's wall time from its Sessions' open and dispose (wall-clock instants, to line up with an
 * outside VRAM sampler), each Session's held bytes (plan backing + input buffers + weights) and each
 * stage's headroom need, and asserts one backing build per DiT and VAE Session. It also logs the
 * WebGPU buffer bytes the process holds, exactly, from `device.createBuffer` / `destroy` (wrapped
 * for the run): per stage the peak of the buffers held (every transient of a run included) and,
 * separately, each headroom probe with what was held when it was made — a sampler of the process
 * VRAM (`nvidia-smi`, outside the test) can miss a transient between two samples and mixes the
 * probes into the stage — and asserts that each stage's peak stays within that stage's headroom
 * need (the estimate leaves the i8a8 linears' quantised intermediates out; the need's margin must
 * cover them). The measurements are recorded in `docs/research/2026-10-28-moss-image-dit-vae.md`
 * §4.3 and §5.5.
 *
 * ## Opt-in VRAM at larger sizes (`KARUME_MOSS_VRAM=1`)
 *
 * The same measurement on 2-step generations ({@link VRAM_SIZES}: upstream's 2M-pixel example
 * 1456², and 1696×1648, the area cap of a 2 GiB binding) for both presets. Whether a size fits is a
 * fact of the device and the moment, not of the code: when the largest stage need does not fit the
 * device now, the case asserts the product's refusal instead — loud, at the first headroom check,
 * before any Session opens — and logs it (on the 12 GiB RTX 3080 Ti the reference preset's DiT pair
 * is refused at the cap: `docs/research/2026-10-28-moss-image-dit-vae.md` §5.5).
 *
 * Without a GPU adapter, the series or the stage-0 embeddings the file skips explicitly; the parity
 * group also without its goldens, naming the command that writes them (ADR 0005).
 */

import { afterAll, beforeAll, describe, expect, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  fitsHeadroom,
  type GpuContext,
  parseSafetensors,
  type Session,
  type SessionDiagnostics,
} from "@karume/runtime";
import type { ModelComponent } from "../../models/src/hub/components.ts";
import {
  mossDitContract,
  mossDitGraphContract,
  readMossDitAssets,
} from "../../models/src/moss/dit-contract.ts";
import { mossDeviceFacts, mossDitStageNeeds } from "../../models/src/moss/dit-stage.ts";
import { type MossTrainingGrid, mossTimestep } from "../../models/src/moss/dit-timestep.ts";
import {
  generateMoss,
  type MossGenerateRequest,
  type MossGenerateState,
  planMossGenerate,
} from "../../models/src/moss/generate.ts";
import {
  MOSS_PRACTICAL_PRESET,
  MOSS_REFERENCE_PRESET,
  mossAcquireGpuOptions,
  type MossPreset,
  mossPreset,
} from "../../models/src/moss/preset.ts";
import type { MossTextEmbeddings } from "../../models/src/moss/request.ts";
import { mossVaeStageNeeds } from "../../models/src/moss/vae-stage.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  containerDigests,
  filePresent,
  floatsOf,
  MOSS_GENERATE_EMBEDS,
  MOSS_GENERATE_SERIES,
  MOSS_REFERENCE_DIR,
  MOSS_TEXT_EMBEDS,
  assertMossRowWritable,
  mossSeriesPresent,
  openMossSeries,
  readBuffer,
  withResidentLog,
} from "../../models/tests/helpers/moss-dit.ts";
import {
  MOSS_GENERATE_VAE_SERIES,
  mossVaeContainerDigests,
  mossVaeSeriesPresent,
  openMossVaeSeries,
} from "../../models/tests/helpers/moss-vae.ts";
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
import { openResults, type Results, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/** The sha256 rows (environment rows — ADR 0106; shared with `e2e_moss_dit_test.ts`). */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/moss.json", import.meta.url),
);
/** Artifacts and verdicts (`outputs/verify/<environment key>/<date>_moss-generate/` — safe to delete). */
const results = openResults("moss-generate");
/** The opt-in full run's (its own directory: `results.json` is rewritten per run). */
const fullResults = openResults("moss-generate-full");

const FULL = process.env.KARUME_MOSS_FULL_PIPELINE === "1";
const VRAM = process.env.KARUME_MOSS_VRAM === "1";

const SERIES_PRESENT = mossSeriesPresent() && mossVaeSeriesPresent();
const EMBEDS_PRESENT = filePresent(MOSS_TEXT_EMBEDS);
const RUNNABLE = GPU_AVAILABLE && SERIES_PRESENT && EMBEDS_PRESENT;

const skipped = (ok: boolean, what: string, command: string): string =>
  ok ? "" : ` (SKIP: no ${what} — ${command})`;

/** The seed path's prompt, seed and knobs (upstream's defaults but the step count). */
const PROMPT = "fridge-note";
const SEED = 42;
const GUIDANCE = 3;
const SHIFT = 5;

type Size = { readonly width: number; readonly height: number };

const REFERENCE = mossPreset(MOSS_REFERENCE_PRESET);
const PRACTICAL = mossPreset(MOSS_PRACTICAL_PRESET);

/** A sha row's ID: the preset, steps, prompt, noise, size (W×H), guidance, shift. */
const caseIdOf = (
  preset: MossPreset,
  steps: number,
  { width, height }: Size,
  guidance = GUIDANCE,
): string =>
  `${preset.name}-${steps}step-${PROMPT}-seed${SEED}-${width}x${height}-g${guidance}-shift${SHIFT}`;

type SeedCase = Size & {
  readonly id: string;
  readonly guidance: number;
  readonly preset: MossPreset;
};

/**
 * The default lane's sha rows (ADR 0138 decision 11; guidance 1 — no negative branch — at 256²;
 * the practical preset at the two sizes with guidance 3 — stage 5).
 */
const SEED_CASES: readonly SeedCase[] = [
  { width: 256, height: 256, guidance: GUIDANCE, preset: REFERENCE },
  { width: 384, height: 256, guidance: GUIDANCE, preset: REFERENCE },
  { width: 256, height: 256, guidance: 1, preset: REFERENCE },
  { width: 256, height: 256, guidance: GUIDANCE, preset: PRACTICAL },
  { width: 384, height: 256, guidance: GUIDANCE, preset: PRACTICAL },
].map((testCase) => ({
  ...testCase,
  id: caseIdOf(testCase.preset, 2, testCase, testCase.guidance),
}));

/** The opt-in full run (upstream's default request), once per preset. */
const FULL_SIZE: Size = { width: 1024, height: 1024 };
const FULL_STEPS = 50;
const FULL_CASES: readonly { readonly preset: MossPreset; readonly id: string }[] = [
  REFERENCE,
  PRACTICAL,
].map((preset) => ({ preset, id: caseIdOf(preset, FULL_STEPS, FULL_SIZE) }));

/**
 * The opt-in VRAM sizes (module doc): upstream's 2M-pixel example (1456² — the GitHub README's;
 * upstream recommends 1M / 1.5M / 2M pixels) and the area cap of a 2 GiB binding (1696×1648).
 */
const VRAM_SIZES: readonly Size[] = [
  { width: 1456, height: 1456 },
  { width: 1696, height: 1648 },
];
const VRAM_STEPS = 2;

/** One case of the upstream parity (`few_step_ref.CASES`, fixed here — not enumerated from files). */
type ParityCase = Size & {
  readonly name: string;
  readonly role: "band" | "accept";
  readonly prompt: string;
  readonly seed: number;
};

const PARITY_PLAN = [
  ["band", "fridge-note", 20261201],
  ["band", "guide-whiteboard", 20261202],
  ["band", "guide-fridge-note", 20261203],
  ["accept", "guide-whiteboard", 20261204],
  ["accept", "fridge-note", 20261205],
] as const;

const PARITY_SIZES: readonly Size[] = [
  { width: 256, height: 256 },
  { width: 384, height: 256 },
];

/** The size key of the bands (W×H). */
const sizeKey = ({ width, height }: Size): string => `${width}x${height}`;

const PARITY_CASES: readonly ParityCase[] = PARITY_SIZES.flatMap((size) =>
  PARITY_PLAN.map(([role, prompt, seed]) => ({
    ...size,
    // The recipe names a golden height × width.
    name: `pipeline-${size.height}x${size.width}-${role}-${prompt}-s${seed}`,
    role,
    prompt,
    seed,
  })),
);
/** The ids of the parity tests (one per size) a writing run needs passed (module doc). */
const PARITY_IDS: readonly string[] = PARITY_SIZES.map(sizeKey);

const MOSS_GENERATE_PIPELINE_GOLDENS =
  "cd tools/export-recipes && uv run --group moss --inexact python -m moss.few_step_ref";

const parityGoldenUrl = (testCase: ParityCase): URL =>
  new URL(`${testCase.name}.safetensors`, MOSS_REFERENCE_DIR);
const PARITY_PRESENT = PARITY_CASES.every((testCase) => filePresent(parityGoldenUrl(testCase)));

/**
 * The bands per size (W×H) and observation point. `undefined` = not derived: every case reports
 * its m and the size fails, printing the candidate.
 *
 * Derivation (2026-10-27, RTX 3080 Ti, Node + Dawn, the goldens of `moss.few_step_ref` — DiT
 * containers `dit` 41745e20…, `dit_skip22` 260d33b8…, `dit_text` 822efba9…): m of the band
 * cases (seeds 20261201 fridge-note / 20261202 guide-whiteboard / 20261203 guide-fridge-note), the
 * worst × 5 rounded up to two significant digits, and the acceptance cases of the same run (seeds
 * 20261204 guide-whiteboard / 20261205 fridge-note):
 *
 * | Size    | Point   | Band cases                     | Worst × 5  | Band   | Acceptance          |
 * | ------- | ------- | ------------------------------ | ---------- | ------ | ------------------- |
 * | 256²    | latents | 9.643e-6, 9.444e-6, 1.054e-5   | 5.270e-5   | 5.3e-5 | 1.846e-5, 1.156e-5  |
 * | 256²    | image   | 2.571e-5, 1.749e-5, 1.776e-5   | 1.286e-4   | 1.3e-4 | 1.735e-5, 1.481e-5  |
 * | 384×256 | latents | 1.821e-5, 2.204e-5, 1.082e-5   | 1.102e-4   | 1.2e-4 | 1.194e-5, 8.928e-6  |
 * | 384×256 | image   | 9.288e-6, 1.242e-5, 1.291e-5   | 6.455e-5   | 6.5e-5 | 7.544e-6, 4.580e-6  |
 *
 * The faults on the first acceptance case (latents / image): prompts swapped 1.559 / 2.244 (256²),
 * 1.582 / 2.169 (384×256); guidance 3.3 9.747e-2 / 1.072e-1, 9.544e-2 / 7.561e-2; shift 3
 * 2.324e-2 / 3.580e-2, 2.288e-2 / 1.655e-2; another seed's latents 3.611e-1 / 3.115e-1, 4.077e-1 /
 * 2.492e-1 — the smallest (shift 3) is 191–439 × its band, far outside the margin of 2. Two more
 * faults, with the bands unchanged: the first step at grid index 1 1.255e-1 /
 * 1.764e-1, 2.074e-1 / 1.077e-1; guidance 3.003 9.751e-4 / 1.060e-3, 9.556e-4 / 8.329e-4 — 8.0–18.4
 * × its band, the fault closest to it.
 *
 * NOTE (a recorded deviation — ADR 0138 decision 11): the band cases and the acceptance cases were
 * measured in the same run, where ADR 0118's addendum makes the acceptance cases after the band is
 * fixed. The rule leaves no choice to adjust: recomputing each band from the band-case m above
 * (worst × 5, rounded up to two significant digits) gives the same four values.
 *
 * MUST: not changed after seeing the acceptance results.
 */
const BANDS: Readonly<
  Record<string, { readonly latents: number | undefined; readonly image: number | undefined }>
> = {
  "256x256": { latents: 5.3e-5, image: 1.3e-4 },
  "384x256": { latents: 1.2e-4, image: 6.5e-5 },
};

/** How far outside its band a fault must land. */
const FAULT_MARGIN = 2;

/** The stage-0 embeddings by prompt name (`moss.text_embeds`). */
const readEmbeddings = async (): Promise<ReadonlyMap<string, MossTextEmbeddings>> => {
  const file = parseSafetensors(await readBuffer(MOSS_TEXT_EMBEDS));
  return new Map(
    [...file.tensors.values()].map((view) => [
      view.name,
      { data: floatsOf(file, view.name, "text embeds").slice(), shape: view.shape },
    ]),
  );
};

/** One Session event of a stage (wall-clock instant; the diagnostics at dispose). */
type SessionEvent = {
  readonly event: string;
  readonly at: number;
  readonly diagnostics?: SessionDiagnostics;
};

/**
 * A component whose Sessions log `open <name>` (once made) and `dispose <name>` (once the dispose
 * has completed) into `events` in order; the last diagnostics are taken just before the dispose.
 */
const observed = (
  component: ModelComponent,
  name: string,
  events: SessionEvent[],
): ModelComponent => ({
  ...component,
  createSession: async (gpu, options) => {
    const session: Session = await component.createSession(gpu, options);
    events.push({ event: `open ${name}`, at: Date.now() });
    const dispose = session.dispose.bind(session);
    Object.defineProperty(session, "dispose", {
      configurable: true,
      value: async () => {
        const diagnostics = session.diagnostics();
        await dispose();
        events.push({ event: `dispose ${name}`, at: Date.now(), diagnostics });
      },
    });
    return session;
  },
});

/** The plan backing builds of each Session of a generation (the diagnostics at its dispose). */
const buildCounts = (events: readonly SessionEvent[]): Readonly<Record<string, number>> =>
  Object.fromEntries(
    events.flatMap(({ event, diagnostics }) =>
      diagnostics === undefined
        ? []
        : [[event.replace("dispose ", ""), diagnostics.planBacking.buildCount]],
    ),
  );

/** The label of `fitsHeadroom`'s trial allocations (`packages/runtime/src/gpu/headroom.ts`). */
const PROBE_LABEL = "karume-headroom-probe";

/** One change of the process's WebGPU buffers (`after` = the Session events logged before it). */
type BufferChange = {
  readonly after: number;
  /** Bytes of the live buffers that are not headroom probes. */
  readonly held: number;
  /** Bytes of the live headroom probes. */
  readonly probe: number;
};

/**
 * Runs `body` with `device.createBuffer` and each created buffer's `destroy` wrapped (restored
 * afterwards), logging the live bytes after every change — exact, unlike a sampler of the process
 * VRAM, but only the buffers' own sizes (no driver overhead, no allocation Dawn makes for itself).
 */
const withBufferLog = async <T>(
  device: GPUDevice,
  events: readonly SessionEvent[],
  body: () => Promise<T>,
): Promise<{ readonly result: T; readonly changes: readonly BufferChange[] }> => {
  const create = device.createBuffer.bind(device);
  const changes: BufferChange[] = [];
  let held = 0;
  let probe = 0;
  const change = (bytes: number, isProbe: boolean): void => {
    if (isProbe) probe += bytes;
    else held += bytes;
    changes.push({ after: events.length, held, probe });
  };
  device.createBuffer = (descriptor: GPUBufferDescriptor): GPUBuffer => {
    const buffer = create(descriptor);
    const isProbe = descriptor.label === PROBE_LABEL;
    change(descriptor.size, isProbe);
    const destroy = buffer.destroy.bind(buffer);
    let destroyed = false;
    Object.defineProperty(buffer, "destroy", {
      configurable: true,
      value: () => {
        if (!destroyed) change(-descriptor.size, isProbe);
        destroyed = true;
        destroy();
      },
    });
    return buffer;
  };
  try {
    return { result: await body(), changes };
  } finally {
    device.createBuffer = create;
  }
};

/** One stage's buffers: the peak bytes held and each headroom probe with what was held then. */
type StageBuffers = {
  readonly name: string;
  readonly held: number;
  readonly probes: readonly { readonly probe: number; readonly held: number }[];
};

/**
 * Per stage (each ends before the change whose `after` reaches its end): the peak bytes held, and
 * each headroom probe (its pieces together — one probe ends when the probes return to 0) with the
 * bytes held while it existed.
 */
const stageBuffers = (
  changes: readonly BufferChange[],
  stages: readonly (readonly [name: string, end: number])[],
): StageBuffers[] =>
  stages.map(([name, end], at) => {
    const start = at === 0 ? 0 : stages[at - 1][1];
    const inStage = changes.filter(({ after }) => after >= start && after < end);
    const held = Math.max(0, ...inStage.map((entry) => entry.held));
    const probes: { probe: number; held: number }[] = [];
    let open: { probe: number; held: number } | undefined;
    for (const entry of inStage) {
      if (entry.probe > 0) {
        open ??= { probe: 0, held: entry.held };
        if (entry.probe > open.probe) open = { probe: entry.probe, held: entry.held };
      } else if (open !== undefined) {
        probes.push(open);
        open = undefined;
      }
    }
    if (open !== undefined) probes.push(open);
    return { name, held, probes };
  });

/** The log lines of {@link stageBuffers}, then what is held after the last change. */
const stageBufferLines = (
  stages: readonly StageBuffers[],
  changes: readonly BufferChange[],
): string[] => {
  const mib = (bytes: number): string => `${(bytes / 2 ** 20).toFixed(0)} MiB`;
  const lines = stages.map(({ name, held, probes }) => {
    const probeText = probes
      .map((probe) => `; probe ${probe.probe} B (${mib(probe.probe)}) over ${mib(probe.held)} held`)
      .join("");
    return `${name}: peak held ${held} B (${mib(held)})${probeText}`;
  });
  const last = changes.at(-1);
  lines.push(`after the generation: held ${last?.held ?? 0} B, probes ${last?.probe ?? 0} B`);
  return lines;
};

/** JSON with the keys of every object sorted (the recipe and this file may order them otherwise). */
const canonicalJson = (value: unknown): string =>
  JSON.stringify(value, (_key, item: unknown) =>
    typeof item === "object" && item !== null && !Array.isArray(item)
      ? Object.fromEntries(Object.entries(item).sort(([left], [right]) => (left < right ? -1 : 1)))
      : item,
  );

/** A training grid with every point moved one index up (the last repeated): the "index + 1" fault. */
const nextGridPoint = (grid: MossTrainingGrid): MossTrainingGrid => {
  const shifted = (values: Float32Array<ArrayBuffer>, width: number): Float32Array<ArrayBuffer> => {
    const out = new Float32Array(values.length);
    out.set(values.subarray(width));
    out.set(values.subarray(values.length - width), values.length - width);
    return out;
  };
  return {
    ...grid,
    sigmas: shifted(grid.sigmas, 1),
    timesteps: shifted(grid.timesteps, 1),
    sinusoid: shifted(grid.sinusoid, grid.width),
  };
};

/** The state with all six components observed into `events`. */
const observedState = (state: MossGenerateState, events: SessionEvent[]): MossGenerateState => ({
  ...state,
  dit: {
    ...state.dit,
    components: {
      dit: observed(state.dit.components.dit, "dit", events),
      ditSkip22: observed(state.dit.components.ditSkip22, "dit_skip22", events),
      ditText: observed(state.dit.components.ditText, "dit_text", events),
    },
  },
  vae: {
    ...state.vae,
    components: {
      pre: observed(state.vae.components.pre, "vae_decoder_pre", events),
      attn: observed(state.vae.components.attn, "vae_decoder_attn", events),
      post: observed(state.vae.components.post, "vae_decoder_post", events),
    },
  },
});

/** A component whose estimate exceeds any device (the first check's fault). */
const huge = (component: ModelComponent): ModelComponent => ({
  ...component,
  estimate: (options) => ({ ...component.estimate(options), peakAccountedBytes: 2 ** 50 }),
});

/** The order of a generation at guidance 1 (no `dit_skip22`). */
const STAGE_ORDER_G1 = [
  "open dit_text",
  "dispose dit_text",
  "open dit",
  "dispose dit",
  "open vae_decoder_pre",
  "open vae_decoder_attn",
  "open vae_decoder_post",
  "dispose vae_decoder_pre",
  "dispose vae_decoder_attn",
  "dispose vae_decoder_post",
];

/** The plan backing builds of a whole generation (module doc "Order and lifetimes"). */
const BUILD_COUNTS = {
  dit_text: 0,
  dit: 1,
  dit_skip22: 1,
  vae_decoder_pre: 1,
  vae_decoder_attn: 1,
  vae_decoder_post: 1,
};

/** The order of a generation that ran to the end (module doc "Order and lifetimes"). */
const STAGE_ORDER = [
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
];

/** m = max |a − b| ÷ max |b| (in f64) and the count of non-finite values of `a`. */
const ratioOf = (actual: Float32Array, expected: Float32Array) => {
  assertEquals(actual.length, expected.length, "element count");
  let difference = 0;
  let scale = 0;
  let nonFinite = 0;
  for (let index = 0; index < expected.length; index += 1) {
    if (!Number.isFinite(actual[index])) nonFinite += 1;
    difference = Math.max(difference, Math.abs(actual[index] - expected[index]));
    scale = Math.max(scale, Math.abs(expected[index]));
  }
  return { m: difference / scale, nonFinite };
};

/** Worst band m × 5, rounded up to two significant digits. */
const bandCandidate = (worst: number): number => {
  const unit = 10 ** (Math.floor(Math.log10(worst * 5)) - 1);
  return Number((Math.ceil((worst * 5) / unit) * unit).toPrecision(2));
};

const countNonFinite = (values: Float32Array): number =>
  values.reduce((count, value) => (Number.isFinite(value) ? count : count + 1), 0);

describe.skipIf(!RUNNABLE)(
  `MOSS-Image generate on the GPU${skipped(GPU_AVAILABLE, "GPU adapter", "a WebGPU device")}${skipped(mossSeriesPresent(), "DiT series", MOSS_GENERATE_SERIES)}${skipped(mossVaeSeriesPresent(), "VAE series", MOSS_GENERATE_VAE_SERIES)}${skipped(EMBEDS_PRESENT, "stage-0 embeddings", MOSS_GENERATE_EMBEDS)}`,
  () => {
    let gpu: GpuContext | undefined;
    let base: Omit<MossGenerateState, "gpu"> | undefined;
    let embeds: ReadonlyMap<string, MossTextEmbeddings> | undefined;
    beforeAll(async () => {
      const series = await openMossSeries();
      const assets = await readMossDitAssets(series.components.dit);
      const dit = {
        components: series.components,
        contract: mossDitContract(
          mossDitGraphContract(series.components, "MOSS-Image"),
          assets.ropeBase,
          assets.trainingGrid,
          "MOSS-Image",
        ),
      };
      base = { dit, vae: await openMossVaeSeries(), preset: mossPreset(MOSS_REFERENCE_PRESET) };
      embeds = await readEmbeddings();
      // The sha rows may be written: the adapter that runs must be the one the row key names.
      // Subgroups when available (module doc — the practical preset's online kernel).
      gpu = verifyRunningGpu(await acquireTestGpu(mossAcquireGpuOptions(PRACTICAL)));
    });
    afterAll(async () => {
      if (gpu === undefined) return;
      await settleReleases(gpu);
      gpu.destroy();
    });
    const state = (): MossGenerateState => {
      assert(gpu !== undefined && base !== undefined, "beforeAll did not finish");
      return { gpu, ...base };
    };
    const prompt = (name: string): MossTextEmbeddings => {
      assert(embeds !== undefined, "beforeAll did not finish");
      const found = embeds.get(name);
      if (found === undefined) throw new Error(`no stage-0 embedding '${name}'`);
      return found;
    };
    /** The seed path's request (module doc "sha256 rows"; no negative at guidance 1). */
    const seedRequest = (
      { width, height }: Size,
      steps = 2,
      guidance = GUIDANCE,
    ): MossGenerateRequest => ({
      width,
      height,
      positive: prompt(PROMPT),
      ...(guidance === 1 ? {} : { negative: prompt("negative") }),
      steps,
      guidance,
      shift: SHIFT,
      seed: SEED,
    });
    /** Runs `body`, then waits for the release of what it freed (the next case's probe sees it). */
    const settled = async <T>(body: () => Promise<T>): Promise<T> => {
      try {
        return await body();
      } finally {
        await settleReleases(state().gpu);
      }
    };
    /** The parity tests ({@link PARITY_IDS}) that passed in this run — each adds its id last. */
    const parityPassed = new Set<string>();
    /** Settles a case's sha row on `sink` after the case's own checks (module doc). */
    const settleImage = async (
      sink: Results,
      id: string,
      size: Size,
      body: () => Promise<{ readonly image: Float32Array<ArrayBuffer>; readonly note: string }>,
    ): Promise<void> => {
      let settlement: ReferenceSettlement | undefined;
      await runRecordedCase(sink, { id }, async () => {
        const { image, note } = await body();
        expect(image.length).toBe(3 * size.width * size.height);
        expect(countNonFinite(image)).toBe(0);
        assertMossRowWritable(references.mode, { expected: PARITY_IDS, passed: parityPassed }, id);
        const outcome = await settleOrObserve(references, sink, {
          id,
          artifact: `${id}.safetensors`,
          bytes: f32ArtifactBytes("image", [3, size.height, size.width], image),
        });
        settlement = outcome.settlement;
        return { ...outcome.fields, note };
      });
      if (settlement?.check.status === "fail") {
        throw new Error(referenceMismatchMessage(id, settlement, references));
      }
    };

    describe.skipIf(!UPSTREAM_PARITY || !PARITY_PRESENT)(
      `against upstream's 2-step pipe(...) with the series' weights${UPSTREAM_PARITY_OPT_IN}${skipped(PARITY_PRESENT, "end-to-end goldens", MOSS_GENERATE_PIPELINE_GOLDENS)}`,
      () => {
        const METADATA_KEY = "karume.moss.few_step_ref";
        let digests: Readonly<Record<string, string>> | undefined;
        let embedsSha: string | undefined;
        /** The case's request (the golden's initial latents injected — module doc). */
        const requestOf = (
          testCase: ParityCase,
          latents: Float32Array<ArrayBuffer>,
        ): MossGenerateRequest => ({
          width: testCase.width,
          height: testCase.height,
          positive: prompt(testCase.prompt),
          negative: prompt("negative"),
          steps: 2,
          guidance: GUIDANCE,
          shift: SHIFT,
          latents,
        });
        /**
         * A golden, refused when stale (other containers, embeddings, token counts or schedule) or
         * of another case.
         */
        const loadGolden = async (testCase: ParityCase) => {
          // Key order as the recipe writes it (JSON with sorted keys), for the comparison below.
          digests ??= Object.fromEntries(
            Object.entries({
              ...(await containerDigests()),
              ...(await mossVaeContainerDigests()),
            }).sort(([left], [right]) => (left < right ? -1 : 1)),
          );
          const file = parseSafetensors(await readBuffer(parityGoldenUrl(testCase)));
          const raw = file.metadata.get(METADATA_KEY);
          if (raw === undefined) throw new Error(`${testCase.name}: no '${METADATA_KEY}' metadata`);
          const metadata: unknown = JSON.parse(raw);
          assert(typeof metadata === "object" && metadata !== null);
          const fields = Object.fromEntries(Object.entries(metadata));
          const expected: Readonly<Record<string, unknown>> = {
            case: testCase.name,
            role: testCase.role,
            seed: testCase.seed,
            prompt: testCase.prompt,
            size: [testCase.height, testCase.width],
            steps: 2,
            guidance_scale: GUIDANCE,
            uncond_skip_layers: [22],
            containers: digests,
          };
          for (const [key, value] of Object.entries(expected)) {
            if (JSON.stringify(fields[key]) !== JSON.stringify(value)) {
              throw new Error(
                `${testCase.name}: golden '${key}' is ${JSON.stringify(fields[key])} (expected` +
                  ` ${JSON.stringify(value)}) — regenerate: ${MOSS_GENERATE_PIPELINE_GOLDENS}`,
              );
            }
          }
          const where = `golden ${testCase.name}`;
          const latentsInit = floatsOf(file, "latents_init", where).slice();
          // The inputs and the schedule upstream ran, against this run's: a regenerated embedding
          // asset or another scheduler config is a stale golden, not a band failure.
          embedsSha ??= await sha256Hex(new Uint8Array(await readBuffer(MOSS_TEXT_EMBEDS)));
          const current = state();
          const { schedule } = planMossGenerate(requestOf(testCase, latentsInit), {
            dit: current.dit.contract,
            vae: current.vae.contract,
            limits: current.gpu.limits,
          });
          const grid = current.dit.contract.trainingGrid;
          const inputs: Readonly<Record<string, unknown>> = {
            text_embeds_sha256: embedsSha,
            tokens: {
              negative: prompt("negative").shape[0],
              positive: prompt(testCase.prompt).shape[0],
            },
            sigmas: [...schedule.sigmas],
            timesteps: schedule.gridIndices.map((index) => mossTimestep(grid, index)),
          };
          for (const [key, value] of Object.entries(inputs)) {
            if (canonicalJson(fields[key]) !== canonicalJson(value)) {
              throw new Error(
                `${testCase.name}: golden '${key}' is ${canonicalJson(fields[key])} (this run:` +
                  ` ${canonicalJson(value)}) — regenerate: ${MOSS_GENERATE_PIPELINE_GOLDENS}`,
              );
            }
          }
          return {
            latentsInit,
            latents: floatsOf(file, "latents.1", where),
            image: floatsOf(file, "image", where),
          };
        };
        type Golden = Awaited<ReturnType<typeof loadGolden>>;
        type Measured = { readonly latents: number; readonly image: number };
        /** A fault: an edit of the request, of the state, or of both. */
        type Fault = {
          readonly name: string;
          readonly request?: (request: MossGenerateRequest) => MossGenerateRequest;
          readonly state?: (state: MossGenerateState) => MossGenerateState;
        };
        /** One generation of `testCase`'s request (edited by `fault`) against its golden. */
        const measure = async (
          testCase: ParityCase,
          golden: Golden,
          fault: Omit<Fault, "name"> = {},
        ): Promise<Measured> => {
          const request = (fault.request ?? ((same) => same))(
            requestOf(testCase, golden.latentsInit),
          );
          const current = (fault.state ?? ((same) => same))(state());
          const generated = await settled(() => generateMoss(current, request));
          const latents = ratioOf(generated.latents, golden.latents);
          const image = ratioOf(generated.image, golden.image);
          assertEquals(latents.nonFinite + image.nonFinite, 0, `${testCase.name}: non-finite`);
          return { latents: latents.m, image: image.m };
        };
        const format = (measured: Measured): string =>
          `latents ${measured.latents.toExponential(3)} / image ${measured.image.toExponential(3)}`;

        for (const size of PARITY_SIZES) {
          const key = sizeKey(size);
          const cases = PARITY_CASES.filter(
            (c) => c.width === size.width && c.height === size.height,
          );
          it(`${key}: each case within the band of its three band cases; the six faults leave it`, async () => {
            const rows: { readonly testCase: ParityCase; readonly measured: Measured }[] = [];
            const goldens = new Map<string, Golden>();
            for (const testCase of cases) {
              const golden = await loadGolden(testCase);
              goldens.set(testCase.name, golden);
              const measured = await measure(testCase, golden);
              rows.push({ testCase, measured });
              console.log(
                `[moss-generate] ${testCase.name} (${testCase.role}): ${format(measured)}`,
              );
            }
            const [accepted] = cases.filter(({ role }) => role === "accept");
            const [other] = cases.filter(({ role }) => role === "band");
            const acceptedGolden = goldens.get(accepted.name);
            const otherGolden = goldens.get(other.name);
            assert(acceptedGolden !== undefined && otherGolden !== undefined);
            const faults: { readonly name: string; readonly measured: Measured }[] = [];
            const firstStepAtNextPoint = (current: MossGenerateState): MossGenerateState => {
              const { contract } = current.dit;
              const moved = { ...contract, trainingGrid: nextGridPoint(contract.trainingGrid) };
              // The fault is what it claims: only the first step moves (σ 1 → the next point).
              const request = requestOf(accepted, acceptedGolden.latentsInit);
              const limits = { vae: current.vae.contract, limits: current.gpu.limits };
              const before = planMossGenerate(request, { ...limits, dit: contract }).schedule;
              const after = planMossGenerate(request, { ...limits, dit: moved }).schedule;
              assertEquals(
                [...after.sigmas],
                [contract.trainingGrid.sigmas[1], ...before.sigmas.slice(1)],
              );
              return { ...current, dit: { ...current.dit, contract: moved } };
            };
            for (const fault of [
              {
                name: "prompts swapped",
                request: (request: MossGenerateRequest) => ({
                  ...request,
                  positive: prompt("negative"),
                  negative: prompt(accepted.prompt),
                }),
              },
              {
                name: "guidance 3.3",
                request: (request: MossGenerateRequest) => ({ ...request, guidance: 3.3 }),
              },
              {
                name: "guidance 3.003",
                request: (request: MossGenerateRequest) => ({ ...request, guidance: 3.003 }),
              },
              {
                name: "shift 3",
                request: (request: MossGenerateRequest) => ({ ...request, shift: 3 }),
              },
              {
                name: `latents of seed ${other.seed}`,
                request: (request: MossGenerateRequest) => ({
                  ...request,
                  latents: otherGolden.latentsInit,
                }),
              },
              { name: "first step at grid index 1", state: firstStepAtNextPoint },
            ] satisfies readonly Fault[]) {
              const measured = await measure(accepted, acceptedGolden, fault);
              faults.push({ name: fault.name, measured });
              console.log(
                `[moss-generate] ${accepted.name} fault ${fault.name}: ${format(measured)}`,
              );
            }
            const failures: string[] = [];
            for (const point of ["latents", "image"] as const) {
              const worst = Math.max(
                ...rows
                  .filter(({ testCase }) => testCase.role === "band")
                  .map(({ measured }) => measured[point]),
              );
              const band = BANDS[key][point];
              console.log(
                `[moss-generate] ${key} ${point}: worst band m ${worst.toExponential(3)} → candidate` +
                  ` ${bandCandidate(worst)}${band === undefined ? "" : ` (band ${band})`}`,
              );
              if (band === undefined) {
                failures.push(
                  `${key} ${point}: the band is not derived — candidate ${bandCandidate(worst)}`,
                );
                continue;
              }
              for (const { testCase, measured } of rows) {
                if (!(measured[point] <= band)) {
                  failures.push(
                    `${testCase.name} ${point}: m ${measured[point]} outside the band ${band}`,
                  );
                }
              }
              for (const { name, measured } of faults) {
                if (!(measured[point] > FAULT_MARGIN * band)) {
                  failures.push(
                    `${key} fault ${name} ${point}: m ${measured[point]} not outside ${FAULT_MARGIN} × the band ${band}`,
                  );
                }
              }
            }
            assert(failures.length === 0, failures.join("\n"));
            parityPassed.add(key);
          });
        }
      },
    );

    describe("default run", () => {
      it("opens and closes the stages in order: dit_text, then the DiT pair, then the VAE (every DiT Session disposed before the VAE opens)", async () => {
        const events: SessionEvent[] = [];

        const { generated, residents } = await withResidentLog(state().gpu, async (residents) => ({
          generated: await settled(() =>
            generateMoss(observedState(state(), events), seedRequest(SEED_CASES[0])),
          ),
          residents,
        }));

        expect(events.map(({ event }) => event)).toEqual(STAGE_ORDER);
        expect(buildCounts(events)).toEqual(BUILD_COUNTS);
        // The two text contexts and the VAE's three hand-over tensors.
        expect(residents.length).toBe(5);
        assert(residents.every((resident) => resident.disposed));
        expect(countNonFinite(generated.image)).toBe(0);
      });

      it("opens no dit_skip22 at guidance 1, the same stage order otherwise", async () => {
        const events: SessionEvent[] = [];

        const { generated, residents } = await withResidentLog(state().gpu, async (residents) => ({
          generated: await settled(() =>
            generateMoss(observedState(state(), events), seedRequest(SEED_CASES[0], 2, 1)),
          ),
          residents,
        }));

        expect(events.map(({ event }) => event)).toEqual(STAGE_ORDER_G1);
        const { dit_skip22: _borrower, ...withoutBorrower } = BUILD_COUNTS;
        expect(buildCounts(events)).toEqual(withoutBorrower);
        // One text context and the VAE's three hand-over tensors.
        expect(residents.length).toBe(4);
        assert(residents.every((resident) => resident.disposed));
        expect(countNonFinite(generated.image)).toBe(0);
      });

      it("refuses before any Session opens when the VAE stage would not fit (the first headroom check)", async () => {
        const events: SessionEvent[] = [];
        const watched = observedState(state(), events);
        const refusing: MossGenerateState = {
          ...watched,
          vae: {
            ...watched.vae,
            components: { ...watched.vae.components, post: huge(watched.vae.components.post) },
          },
        };

        await assertRejects(
          () => settled(() => generateMoss(refusing, seedRequest(SEED_CASES[0]))),
          Error,
          "at its largest stage (VAE) that this device cannot allocate now (checked before any stage opens)",
        );
        expect(events).toEqual([]);
      });

      it("refuses before any Session opens when the DiT pair would not fit (the first headroom check)", async () => {
        const events: SessionEvent[] = [];
        const watched = observedState(state(), events);
        const refusing: MossGenerateState = {
          ...watched,
          dit: {
            ...watched.dit,
            components: { ...watched.dit.components, dit: huge(watched.dit.components.dit) },
          },
        };

        await assertRejects(
          () => settled(() => generateMoss(refusing, seedRequest(SEED_CASES[0]))),
          Error,
          "at its largest stage (dit + dit_skip22) that this device cannot allocate now (checked before any stage opens)",
        );
        expect(events).toEqual([]);
      });

      it("takes the largest stage's own need for the first check, not the sum of the stages", async () => {
        const events: SessionEvent[] = [];
        const watched = observedState(state(), events);
        const refusing: MossGenerateState = {
          ...watched,
          vae: {
            ...watched.vae,
            components: { ...watched.vae.components, post: huge(watched.vae.components.post) },
          },
        };
        const request = seedRequest(SEED_CASES[0]);
        const { gpu } = refusing;
        const plan = planMossGenerate(request, {
          dit: refusing.dit.contract,
          vae: refusing.vae.contract,
          limits: gpu.limits,
        });
        const vaeNeed = mossVaeStageNeeds({ ...refusing.vae, limits: gpu.limits }, plan.vae);

        await assertRejects(
          () => settled(() => generateMoss(refusing, request)),
          Error,
          `needs up to ${vaeNeed} bytes of VRAM at its largest stage (VAE)`,
        );
        expect(events).toEqual([]);
      });

      it("has closed every Session and text context when generate rejects on a failing DiT step, and opens no VAE", async () => {
        const events: SessionEvent[] = [];
        const watched = observedState(state(), events);
        const failing: MossGenerateState = {
          ...watched,
          dit: {
            ...watched.dit,
            contract: {
              ...watched.dit.contract,
              outputs: { ...watched.dit.contract.outputs, dit: "no_such_output" },
            },
          },
        };
        let atRejection: readonly string[] = [];

        const residents = await withResidentLog(state().gpu, async (residents) => {
          // Not through `settled`: the log is read the moment generate rejects.
          await assertRejects(
            () => generateMoss(failing, seedRequest(SEED_CASES[0])),
            Error,
            "dit returned no 'no_such_output'",
          );
          atRejection = events.map(({ event }) => event);
          return residents;
        });
        await settleReleases(state().gpu);

        expect(atRejection).toEqual(STAGE_ORDER.slice(0, 6));
        expect(residents.length).toBe(2);
        assert(residents.every((resident) => resident.disposed));
      });

      for (const testCase of SEED_CASES) {
        it(`${testCase.id}: a finite image, pinned by its sha256 row`, async () => {
          await settleImage(results, testCase.id, testCase, async () => {
            const started = performance.now();
            const generated = await settled(() =>
              generateMoss(
                { ...state(), preset: testCase.preset },
                seedRequest(testCase, 2, testCase.guidance),
              ),
            );
            const seconds = ((performance.now() - started) / 1000).toFixed(1);
            return { image: generated.image, note: `generate ${seconds} s` };
          });
        });
      }
    });

    /**
     * One generation of `preset` at `size` with the stage timeline, the held bytes and the WebGPU
     * buffer log (module doc "Opt-in 1024² × 50 steps"): asserts the stage order, one backing build
     * per DiT and VAE Session, and each stage's peak held within its headroom need; returns the
     * image and the first log line.
     */
    const measuredGeneration = async (
      preset: MossPreset,
      size: Size,
      steps: number,
      label: string,
    ): Promise<{ readonly image: Float32Array<ArrayBuffer>; readonly note: string }> => {
      const current = { ...state(), preset };
      const request = seedRequest(size, steps);
      const plan = planMossGenerate(request, {
        dit: current.dit.contract,
        vae: current.vae.contract,
        limits: current.gpu.limits,
      });
      const ditNeeds = mossDitStageNeeds(
        { ...current.dit, preset, device: mossDeviceFacts(current.gpu) },
        plan.dit,
      );
      const vaeNeed = mossVaeStageNeeds({ ...current.vae, limits: current.gpu.limits }, plan.vae);
      console.info(
        `[${label}] ${preset.name} ${sizeKey(size)} headroom needs: dit_text ${ditNeeds.text} B,` +
          ` pair ${ditNeeds.pair} B, VAE ${vaeNeed} B; schedule ${plan.schedule.gridIndices.length} steps`,
      );
      const events: SessionEvent[] = [];
      const started = Date.now();
      console.info(`[${label}] generate start ${new Date(started).toISOString()}`);
      const { result: generated, changes } = await withBufferLog(current.gpu.device, events, () =>
        settled(() => generateMoss(observedState(current, events), request)),
      );
      const ended = Date.now();
      console.info(`[${label}] generate end ${new Date(ended).toISOString()}`);
      expect(events.map(({ event }) => event)).toEqual(STAGE_ORDER);
      expect(buildCounts(events)).toEqual(BUILD_COUNTS);
      const index = (event: string): number => {
        const found = events.findIndex((entry) => entry.event === event);
        assert(found !== -1, `no event '${event}'`);
        return found + 1;
      };
      const stages = stageBuffers(changes, [
        // A stage ends when its last Session's dispose has completed (`after` counts it).
        ["text stage (the first check included)", index("dispose dit_text")],
        ["DiT stage", index("dispose dit")],
        ["VAE stage", Number.POSITIVE_INFINITY],
      ]);
      console.info(
        `[${label}] WebGPU buffers\n  ${stageBufferLines(stages, changes).join("\n  ")}`,
      );
      const at = (event: string): number => {
        const found = events.find((entry) => entry.event === event);
        assert(found !== undefined, `no event '${event}'`);
        return found.at;
      };
      const lines = [
        `wall ${((ended - started) / 1000).toFixed(1)} s`,
        ...events.map(({ event, at: when }) => `${new Date(when).toISOString()} ${event}`),
        `text stage ${((at("dispose dit_text") - at("open dit_text")) / 1000).toFixed(1)} s`,
        `DiT stage (dit opened → dit disposed) ${((at("dispose dit") - at("open dit")) / 1000).toFixed(1)} s`,
        `VAE stage ${((at("dispose vae_decoder_post") - at("open vae_decoder_pre")) / 1000).toFixed(1)} s`,
        ...events
          .filter(({ diagnostics }) => diagnostics !== undefined)
          .map(({ event, diagnostics }) => {
            assert(diagnostics !== undefined);
            return (
              `${event.replace("dispose ", "")}: plan backing ${diagnostics.planBacking.residentBytes} B,` +
              ` input buffers ${diagnostics.planBacking.inputBytes} B,` +
              ` weights ${diagnostics.weights.allocatedBytes} B,` +
              ` backing builds ${diagnostics.planBacking.buildCount}`
            );
          }),
      ];
      console.info(`[${label}]\n  ${lines.join("\n  ")}`);
      // Each stage's peak within its headroom need (module doc — the margin covers what the
      // estimate leaves out). The text stage's held peak excludes the first check's probe.
      const [text, dit, vae] = stages;
      expect(text.held).toBeLessThanOrEqual(ditNeeds.text);
      expect(dit.held).toBeLessThanOrEqual(ditNeeds.pair);
      expect(vae.held).toBeLessThanOrEqual(vaeNeed);
      return { image: generated.image, note: lines.slice(0, 1).join(" / ") };
    };

    describe.skipIf(!FULL)("1024² × 50 steps (opt-in KARUME_MOSS_FULL_PIPELINE=1)", () => {
      for (const { preset, id } of FULL_CASES) {
        it(`${id}: runs to completion with a finite image, the stage timeline and held bytes logged, each stage within its need, pinned by its sha256 row`, async () => {
          await settleImage(fullResults, id, FULL_SIZE, () =>
            measuredGeneration(preset, FULL_SIZE, FULL_STEPS, "moss-generate full"),
          );
        });
      }
    });

    describe.skipIf(!VRAM)("VRAM at 1456² and the area cap (opt-in KARUME_MOSS_VRAM=1)", () => {
      for (const preset of [REFERENCE, PRACTICAL]) {
        for (const size of VRAM_SIZES) {
          it(`${preset.name} ${sizeKey(size)} × ${VRAM_STEPS} steps: each stage within its need, or the product's refusal before any Session opens`, async () => {
            const current = { ...state(), preset };
            const request = seedRequest(size, VRAM_STEPS);
            const plan = planMossGenerate(request, {
              dit: current.dit.contract,
              vae: current.vae.contract,
              limits: current.gpu.limits,
            });
            const ditNeeds = mossDitStageNeeds(
              { ...current.dit, preset, device: mossDeviceFacts(current.gpu) },
              plan.dit,
            );
            const vaeNeed = mossVaeStageNeeds(
              { ...current.vae, limits: current.gpu.limits },
              plan.vae,
            );
            const largest = Math.max(ditNeeds.text, ditNeeds.pair, vaeNeed);
            await settleReleases(current.gpu);
            if (await fitsHeadroom(current.gpu, largest)) {
              const { image } = await measuredGeneration(preset, size, VRAM_STEPS, "moss-vram");
              expect(countNonFinite(image)).toBe(0);
              return;
            }
            // A fact of the device and the moment (module doc): the refusal is the product's.
            console.info(
              `[moss-vram] ${preset.name} ${sizeKey(size)}: this device refuses the largest stage need` +
                ` ${largest} B now (pair ${ditNeeds.pair} B, VAE ${vaeNeed} B)`,
            );
            const events: SessionEvent[] = [];
            await assertRejects(
              () => settled(() => generateMoss(observedState(current, events), request)),
              Error,
              "checked before any stage opens",
            );
            expect(events).toEqual([]);
          });
        }
      }
    });
  },
);

/** The sha rows this file settles (the full run's only under its opt-in — module doc). */
const CASE_IDS: readonly string[] = [
  ...SEED_CASES.map(({ id }) => id),
  ...(FULL ? FULL_CASES.map(({ id }) => id) : []),
];
if (RUNNABLE) references.warnMissing(CASE_IDS);
// A missing row of this environment is not a silent pass (ADR 0005's gatekeeper form).
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
