/**
 * Irodori's **WAV sha256 check** (real GPU). It goes all the way from the distribution through
 * `Pipeline.fromAssets` → `generate` → `encodeWav` and looks only at whether the output WAV's
 * sha256 is **bit-identical** to the reference value (Irodori's version of the same philosophy as
 * SBV2's `e2e_sbv2_wav_test.ts` / Anima's PNG check).
 *
 * ## Split with the existing Irodori checks (they coexist because their roles differ)
 *
 * - The latent check (`e2e_irodori_latent_test.ts`) — looks at **agreement with upstream**. It
 *   injects the golden's initial noise, compares z within a tolerance, and pins only S and the
 *   forward count exactly. The difference in reduction order between torch and WGSL remains
 *   structurally, so bit identity is impossible in principle there.
 * - The tile-equivalence check (`e2e_irodori_codec_test.ts`) — that decode's **performance knobs
 *   do not change the output**.
 * - The reference-preprocessing parity check (`e2e_irodori_reference_test.ts`) — agreement of the
 *   reference audio → latent chain with upstream.
 * - **This check** — does not look at upstream at all. It pins the whole chain from text to
 *   waveform (the internal `Randn` random sequence / the duration's choice of S / Euler + CFG /
 *   the tail trim / codec tile decode / the cut to seconds / `encodeWav`'s quantization) **to one
 *   bit, as a regression detector in this execution environment**. Even with the 3 checks above
 *   all green, one changed line of host glue moves the waveform — this is the final check that
 *   catches it.
 *
 * ## MUST: when it breaks, do not switch to a tolerance; find the cause before updating the digest
 *
 * If a value moves by even one bit, something in the chain changed. **Switching to a tolerance is
 * forbidden**: stop red and print the differences (WAV byte length / sample count / S / forward
 * count / effective knobs / first differing position / the WAV itself).
 *
 * ## The reference digests are rows per environment (no bit identity across devices is promised)
 *
 * The reference values are in `fixtures/references/irodori.json` as **rows per environment key**
 * (`helpers/reference.ts`). All this check can claim is "the values did not move on that machine";
 * not matching on other backends (Metal etc.) is by design — the mechanism and how to check
 * soundness on another backend (self A/B) are in the "sha256 reference checks are specific to the
 * reference environment" section of [limitations](../../../docs/limitations.md). The RTX 3080 Ti
 * row was measured on 2026-08-12 and confirmed by baking it **twice in separate processes**.
 *
 * - **Adding another machine's row** is `KARUME_REFERENCE=write` (existing rows are not touched).
 * - **Re-baking your own machine's row** is `KARUME_REFERENCE=rewrite` only, and only once you can
 *   say "what changed" first (old → new goes to the log). Other environments' rows are left alone.
 * - On a machine with no reference value the case **SKIPs explicitly**, and the reference gate (at
 *   the end of this file) reports "all SKIP for lack of references" in red.
 * - After re-baking, run it **twice in separate processes** to confirm agreement, and write what
 *   changed in the commit message.
 *
 * ## Two cases
 *
 * - **voice-clone** — with reference audio + caption. The canonical case, including the
 *   `{ audio }` path (truncation to 120 s → LUFS −16 → reflect pad → `codec_encoder`), with all 3
 *   CFG branches active (100 forwards / S 170 / a 6.80 s waveform). Generation + WAV encoding take
 *   **13.5 s** (reference environment, 2026-08-12).
 * - **no-ref** — no reference / no caption. The path where the speaker's zero short-circuit and the
 *   caption's CFG off take effect (60 forwards / S 116 / a 4.64 s waveform). **7.6 s**.
 *
 * One test takes about 25 s of wall time — the 21 s above plus reading the 3.3 GB distribution and
 * `openContainer` (the 2 cases share one pipeline to keep that to once).
 *
 * Neither passes `initialNoise` — pinning the determinism of the internal `Randn` is this check's
 * job (the main difference from the latent check, which injects the golden noise). No knob is
 * passed either, so the effective values are the distribution's `pipelineConfig` as is (drift is
 * checked first with {@link REFERENCE_KNOBS}).
 *
 * MUST: the assets are the distribution `models/karume-irodori-v4-small/` and the reference audio
 * `inputs/irodori/v4-small/samples/clone_ref1.wav` (both untracked, local assets of a real-GPU
 * machine). Environments missing them and environments without a GPU **SKIP explicitly** with a
 * reason (no deleting tests to go silently green — ADR 0005).
 *
 * NOTE: only the tests read local files (`readFile` of `@karume/test-support/fs`). The package
 * itself uses Web standard APIs only (a cross-cutting invariant), and `fromAssets` is a surface that
 * only takes bytes.
 */

import { readFile, writeFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assert, assertEquals, assertFalse, assertRejects, assertStrictEquals } from "@std/assert";
import { type FileRef, parseManifest, resolveSelection } from "@karume/hub";
import type { Manifest, ModelEntry } from "@karume/hub";
import type { SessionDiagnostics } from "@karume/runtime";
import {
  type GeneratedAudio,
  type GenerateEvent,
  type GenerateRequest,
  Pipeline,
  type RunComponent,
} from "../../models/irodori.ts";
import { type DecodedWav, decodeWav, encodeWav } from "../../models/mod.ts";
import {
  type IrodoriRfPipelineConfig,
  parseIrodoriPipelineConfig,
} from "../../models/src/irodori/config.ts";
import { tSchedule } from "../../models/src/irodori/host/sampler.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { assertRunningAdapter } from "../../runtime/tests/helpers/environment.ts";
import {
  announceCheck,
  expectedOf,
  openReferences,
  registerReferenceGate,
} from "../../runtime/tests/helpers/reference.ts";
import { openResults } from "../../runtime/tests/helpers/results.ts";
import {
  readFileIfPresent,
  readTextIfPresent,
} from "../../models/tests/helpers/read-if-present.ts";

/** Where the distribution is (the default output of `karume dist --pipeline irodori`). */
const ASSETS_DIR = new URL("../../../models/karume-irodori-v4-small/", import.meta.url);
/** The reference audio (a hand-placed real asset — 48 kHz mono PCM16). */
const REFERENCE_AUDIO = new URL(
  "../../../inputs/irodori/v4-small/samples/clone_ref1.wav",
  import.meta.url,
);
/** Where the reference digests live (rows per environment key — tracked). */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/irodori.json", import.meta.url),
);
/**
 * Where the artifacts and verdicts go (`outputs/verify/<environment key>/<date>_irodori/` — safe to
 * delete).
 */
const results = openResults("irodori");

/** The generation command, ready to paste on a SKIP. */
const DIST_COMMAND = "cd tools/exporter && uv run karume dist --pipeline irodori";

const MODEL = "v4-small";
const QUANT = "f32";

/**
 * The length of the RIFF header `encodeWav` writes (needed to turn the first differing position
 * into a sample index).
 */
const HEADER_BYTES = 44;

/** The conditions and reference value of one check. */
type WavCase = {
  readonly name: string;
  /** "What this one exercises", shown in the step name. */
  readonly why: string;
  readonly text: string;
  readonly caption?: string;
  /**
   * Whether the reference audio is passed as `{ audio }` (whether the wav path = LUFS
   * normalization + `codec_encoder` is included).
   */
  readonly withReference: boolean;
  readonly seed: number;
};

/**
 * Two cases. The text and caption are borrowed from the full-loop golden's `meta.json` (comparing
 * the same sentences makes it easier to line up with the latent check's measurements when one
 * breaks). But **this check does not read the golden** — the reference digest is a measurement
 * taken all the way to the waveform, closed independently of the golden's generation conditions.
 *
 * `name` is the reference value's case ID as is (the key in `fixtures/references/irodori.json`), and
 * also the artifact's file name.
 */
const CASES: readonly WavCase[] = [
  {
    name: "voice-clone",
    why: "reference audio + caption (wav path = LUFS normalization + codec_encoder, 3 CFG branches)",
    text: "今日は近くの店まで歩いて行きました。とても良い天気でしたね。",
    caption:
      "若く元気な女性の声。カフェの店員のように、明るくハキハキとした少し高めのトーンで話している。",
    withReference: true,
    seed: 1234,
  },
  {
    name: "no-ref",
    why: "no reference / no caption (the path where the speaker's zero short-circuit and the caption's CFG off take effect)",
    text: "本日はお越しいただき、誠にありがとうございます。",
    withReference: false,
    seed: 1235,
  },
];

/**
 * The effective knobs when the reference digests were baked (they should match the
 * distribution's `pipelineConfig`).
 *
 * Only the numbers that decide the waveform are copied — the sampler (`steps` / `initScale` / CFG
 * strength and interval) and the latent → waveform geometry (`frameRate` / `sampleRate` /
 * `hopLength` / `codecHaloFrames`).
 */
const REFERENCE_KNOBS = {
  steps: 40,
  initScale: 0.999,
  cfgMinT: 0.5,
  cfgMaxT: 1,
  cfgText: 3,
  cfgSpeaker: 5,
  cfgCaption: 3,
  frameRate: 25,
  sampleRate: 48000,
  hopLength: 1920,
  codecHaloFrames: 8,
} as const;

const manifestText = await readTextIfPresent(new URL("karume.json", ASSETS_DIR));
const referenceBytes = await readFileIfPresent(REFERENCE_AUDIO);
const ASSETS_AVAILABLE = manifestText !== undefined && referenceBytes !== undefined;
if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] skipping Irodori's WAV sha256 check (it needs both the distribution ${ASSETS_DIR.pathname} and the reference audio ` +
      `${REFERENCE_AUDIO.pathname}). ` +
      (manifestText === undefined
        ? `Generate: ${DIST_COMMAND}`
        : "The reference audio is a hand-placed real asset (the inputs root of docs/assets-layout.md)"),
  );
}

const RUNNABLE = GPU_AVAILABLE && ASSETS_AVAILABLE;

/** Every case ID of this check (seen by the registration warning and the reference gate). */
const CASE_IDS: readonly string[] = CASES.map((item) => item.name);

// Cases without a reference value for this environment SKIP explicitly (the warning says how to
// make one).
if (RUNNABLE) references.warnMissing(CASE_IDS);

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const readManifest = (): Manifest => parseManifest(manifestText as string);

const modelEntry = (manifest: Manifest): ModelEntry => {
  if (!Object.hasOwn(manifest.models, MODEL)) {
    throw new Error(
      `the distribution has no model '${MODEL}' (present: ${manifest.available.models.join(" / ")})`,
    );
  }
  return manifest.models[MODEL];
};

/**
 * Reads the assets the distribution requires from local files (a local version of `fetchAssets` —
 * it bypasses the fetch layer).
 */
const loadLocalAssets = async (
  manifest: Manifest,
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const selection = resolveSelection(manifest, { model: MODEL, quant: QUANT });
  const byPath = new Map<string, Uint8Array<ArrayBuffer>>();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  const read = async (key: string, ref: FileRef): Promise<void> => {
    const cached = byPath.get(ref.path);
    const bytes = cached ?? (await readFile(new URL(ref.path, ASSETS_DIR)));
    if (cached === undefined) byPath.set(ref.path, bytes);
    assets = { ...assets, [key]: bytes };
  };
  // MUST: zero-length parts are listed too (the index is the id inside the container — skipping
  // one would shift every later one up by one).
  for (const name of Object.keys(selection.containers)) {
    const { parts } = selection.containers[name];
    for (const [index, ref] of parts.entries()) await read(`${name}[${index}]`, ref);
  }
  for (const name of Object.keys(selection.assets)) await read(name, selection.assets[name]);
  return assets;
};

/**
 * Checks the check's **preconditions** (not the check itself). The reference digests were baked
 * with the knobs as the distribution declares them, so a distribution whose declaration moved
 * fails with "the conditions differ" rather than "the sha256 differs" (when a re-baked distribution
 * breaks the digest, it says first whether the cause is on the asset side or the implementation
 * side).
 */
const assertReferenceKnobs = (entry: ModelEntry): IrodoriRfPipelineConfig => {
  const config = parseIrodoriPipelineConfig(entry.pipelineConfig);
  if (config.flowParameterization !== "rf_velocity") {
    throw new Error("the distribution does not declare the RF sampler the digests were baked with");
  }
  const drifted: string[] = [];
  const check = (key: string, expected: number, actual: number): void => {
    if (expected !== actual) drifted.push(`${key} ${expected} → ${actual}`);
  };
  check("steps", REFERENCE_KNOBS.steps, config.steps);
  check("initScale", REFERENCE_KNOBS.initScale, config.initScale);
  check("cfgMinT", REFERENCE_KNOBS.cfgMinT, config.cfgMinT);
  check("cfgMaxT", REFERENCE_KNOBS.cfgMaxT, config.cfgMaxT);
  check("cfgScales.text", REFERENCE_KNOBS.cfgText, config.cfgScales.text);
  check("cfgScales.speaker", REFERENCE_KNOBS.cfgSpeaker, config.cfgScales.speaker);
  check("cfgScales.caption", REFERENCE_KNOBS.cfgCaption, config.cfgScales.caption);
  check("frameRate", REFERENCE_KNOBS.frameRate, config.frameRate);
  check("sampleRate", REFERENCE_KNOBS.sampleRate, config.sampleRate);
  check("hopLength", REFERENCE_KNOBS.hopLength, config.hopLength);
  check("codecHaloFrames", REFERENCE_KNOBS.codecHaloFrames, config.codecHaloFrames);
  if (drifted.length > 0) {
    throw new Error(
      `the distribution's pipelineConfig differs from the reference digests' generation conditions: ${drifted.join(" / ")}\n` +
        "(the check's preconditions broke — find out why the conditions changed first, instead of replacing the reference sha256)",
    );
  }
  return config;
};

/**
 * A one-line rendering of the effective knobs (keeps "what was baked" in the report when the
 * comparison breaks).
 */
const describeKnobs = (item: WavCase, config: IrodoriRfPipelineConfig): string =>
  `model ${MODEL} / quant ${QUANT} / seed ${item.seed} / text ${JSON.stringify(item.text)} / ` +
  `caption ${JSON.stringify(item.caption ?? "")} / reference audio ${
    item.withReference ? "yes" : "no"
  } / ` +
  `steps ${config.steps} / initScale ${config.initScale} / ` +
  `cfg ${config.cfgScales.text}/${config.cfgScales.speaker}/${config.cfgScales.caption} ` +
  `on [${config.cfgMinT}, ${config.cfgMaxT}] / ${config.frameRate}fps / ` +
  `${config.sampleRate}Hz / hop ${config.hopLength} / halo ${config.codecHaloFrames}`;

/**
 * Where the reference WAV **file** is placed, if present (optional). Every run leaves its artifact
 * in the results slot, so placing the WAV of the run to compare against in the same slot as
 * `<case>-reference.wav` makes the next run print the first differing position too. The check is
 * closed by the sha256 alone; this is only **material for it**.
 */
const referenceWavPath = (item: WavCase): URL => results.artifact(`${item.name}-reference.wav`);

/**
 * If the reference WAV file is at hand, prints **the first differing position and the
 * neighbouring values** too.
 *
 * MUST: check the file's sha256 before using it — diffing against a WAV baked under other
 * conditions would make the diagnosis itself a lie.
 */
const describeFirstDifference = async (
  item: WavCase,
  expected: string,
  actual: Uint8Array<ArrayBuffer>,
): Promise<string> => {
  const path = referenceWavPath(item);
  const reference = await readFileIfPresent(path);
  if (reference === undefined) {
    return `no reference WAV file (${path.pathname}) — the first differing position cannot be shown`;
  }
  const sha = await sha256Hex(reference);
  if (sha !== expected) {
    return `${path.pathname} is not the reference WAV (sha256 ${sha}) — the first differing position cannot be shown`;
  }
  const shared = Math.min(reference.length, actual.length);
  let at = 0;
  while (at < shared && reference[at] === actual[at]) at += 1;
  if (at === shared) {
    return `the first ${shared} bytes match (only the length differs: reference ${reference.length} / measured ${actual.length})`;
  }
  if (at < HEADER_BYTES) {
    return `the first difference is inside the RIFF header (offset ${at}): byte ${reference[at]} → ${actual[at]}`;
  }
  const i16 = (bytes: Uint8Array<ArrayBuffer>, sample: number): number =>
    new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getInt16(
      HEADER_BYTES + sample * 2,
      true,
    );
  const sample = Math.floor((at - HEADER_BYTES) / 2);
  const last = Math.floor((shared - HEADER_BYTES) / 2) - 1;
  const neighbours: string[] = [];
  for (let index = Math.max(0, sample - 2); index <= Math.min(last, sample + 2); index += 1) {
    neighbours.push(
      `${index === sample ? "*" : ""}${index} ${i16(reference, index)} → ${i16(actual, index)}`,
    );
  }
  return `the first difference is at sample ${sample} (offset ${at}) / neighbouring i16 ${neighbours.join(" , ")}`;
};

/**
 * The report when the reference sha disagrees.
 *
 * MUST: do not escape into a tolerance here. The artifact stays in the results slot
 * (`outputs/verify/...`), in a form a person can listen to and compare.
 */
const mismatchReport = async (
  item: WavCase,
  config: IrodoriRfPipelineConfig,
  audio: GeneratedAudio,
  wav: Uint8Array<ArrayBuffer>,
  expected: string,
  actual: string,
  dumped: URL,
): Promise<string> =>
  `${item.name}: the output WAV's sha256 does not match the reference\n` +
  `  expected ${expected}\n  actual ${actual}\n` +
  `  WAV ${wav.length} bytes / samples ${audio.data.length} / ${audio.sampleRate}Hz / ` +
  `${(audio.data.length / audio.sampleRate).toFixed(3)}s / S ${audio.frames} / ` +
  `forwards ${audio.forwards}\n` +
  `  effective knobs ${describeKnobs(item, config)}\n` +
  `  ${await describeFirstDifference(item, expected, wav)}\n` +
  `  artifact ${dumped.pathname}`;

/**
 * Generates one case and compares its digest.
 *
 * A call with `onEvent` is also compared against **the same reference value** (= the row of the
 * same case ID) — direct evidence that the observation hook does not touch a single bit of the
 * values, and that the resident path and the host path give identical output (the onEvent check
 * below). The results JSON keeps it as a separate case.
 *
 * MUST: bit identity between the paths is compared **measurement against measurement** (the same
 * form as ADR 0106 decision 6). Relying only on the shared reference row, a
 * `KARUME_REFERENCE=rewrite` run would just re-bake the row from the plain path's measurement and
 * then overwrite the same row with the `onEvent` path's measurement, ending green even with A≠B.
 */
const runCase = async (
  pipeline: Pipeline,
  config: IrodoriRfPipelineConfig,
  item: WavCase,
  reference: DecodedWav,
  /**
   * Present only on a call with the observation hook (`onEvent`). `twinSha` is the plain path's
   * **measured** sha.
   */
  observed?: {
    readonly onEvent: (event: GenerateEvent) => void;
    readonly twinSha: string;
  },
): Promise<{ readonly audio: GeneratedAudio; readonly sha: string }> => {
  const onEvent = observed?.onEvent;
  // Pass neither `initialNoise` nor knobs — pin the internal `Randn` and the `pipelineConfig`
  // defaults as well.
  const request: GenerateRequest = {
    text: item.text,
    ...(item.caption === undefined ? {} : { caption: item.caption }),
    ...(item.withReference ? { speaker: { audio: reference } } : {}),
    seed: item.seed,
    ...(onEvent === undefined ? {} : { onEvent }),
  };
  const started = performance.now();
  const audio = await pipeline.generate(request);
  const wav = encodeWav(audio.data, audio.sampleRate);
  const actual = await sha256Hex(wav);
  const elapsedMs = performance.now() - started;
  const resultId = onEvent === undefined ? item.name : `${item.name}-onEvent`;
  // Keep the artifact every time, pass or fail (the A/B material for the next break is needed
  // before it breaks).
  const dumped = results.artifact(`${resultId}.wav`);
  await writeFile(dumped, wav);
  console.log(
    `[e2e] irodori ${MODEL}/${QUANT} ${item.name}${onEvent === undefined ? "" : "(onEvent)"}: ${(
      elapsedMs / 1000
    ).toFixed(1)}s / S ${audio.frames} / ` +
      `forwards ${audio.forwards} / WAV ${wav.length}B / ` +
      `${(audio.data.length / audio.sampleRate).toFixed(2)}s / sha256 ${actual}`,
  );
  // The bit-identity decision between paths (MUST) does not go through the reference row; it
  // compares directly with the plain path's measurement.
  const twinMismatch =
    observed !== undefined && observed.twinSha !== actual
      ? `not bit-identical to the plain path's WAV (plain ${observed.twinSha} / onEvent ${actual}) — ` +
        "the observation hook touches the values, or the host path and the resident path diverge"
      : undefined;
  const check = references.check(item.name, actual);
  announceCheck(item.name, check, actual);
  const expected = expectedOf(check);
  await results.record({
    id: resultId,
    status: twinMismatch === undefined ? check.status : "fail",
    ...(expected === undefined ? {} : { expected }),
    actual,
    artifact: `${resultId}.wav`,
    elapsedMs,
    ...(twinMismatch === undefined ? {} : { note: twinMismatch }),
  });
  if (twinMismatch !== undefined) throw new Error(`${item.name}: ${twinMismatch}`);
  if (check.status === "fail") {
    throw new Error(await mismatchReport(item, config, audio, wav, check.expected, actual, dumped));
  }
  return { audio, sha: actual };
};

describe.skipIf(!RUNNABLE)(
  `e2e (real GPU): the WAV of distribution ${MODEL} / quant ${QUANT} matches the reference sha256`,
  () => {
    let config: IrodoriRfPipelineConfig;
    let reference: DecodedWav;
    // Observations per run. Whether `lastRun` (the arena record) is present is **independent
    // evidence** of "host path or resident path" — the resident path's `enqueue` creates neither
    // an arena nor a timing window, so it is undefined.
    const runs: { component: RunComponent; hasArena: boolean }[] = [];
    // All cases share one pipeline so the 3.3 GB of assets are not read twice.
    let pipeline: Pipeline | undefined;
    /**
     * The plain path's measured sha (what the onEvent path's bit identity between paths is
     * compared with, without going through the reference row).
     */
    const plainSha = new Map<string, string>();

    // The its below run only after beforeAll succeeded; this narrows the shared pipeline.
    const opened = (): Pipeline => {
      assert(pipeline !== undefined, "the pipeline is not open (beforeAll failed)");
      return pipeline;
    };

    beforeAll(async () => {
      // This run may write reference values, so first check that the adapter the key was taken
      // from is the one running.
      await assertRunningAdapter();
      const manifest = readManifest();
      // Check the preconditions first (if the effective knobs moved, say "the conditions differ"
      // before generating).
      config = assertReferenceKnobs(modelEntry(manifest));
      const assets = await loadLocalAssets(manifest);
      // The reference audio is 48 kHz mono PCM16. If its rate differs from the distribution,
      // `generate` fails loudly (there is no resampling), so it is only read here.
      reference = decodeWav(referenceBytes as Uint8Array<ArrayBuffer>);
      pipeline = await Pipeline.fromAssets(
        { manifest, assets },
        {
          model: MODEL,
          quant: QUANT,
          onRunDiagnostics: (component: RunComponent, diagnostics: SessionDiagnostics) =>
            runs.push({ component, hasArena: diagnostics.lastRun !== undefined }),
        },
      );
    });
    // This also intends to check the release through [Symbol.asyncDispose] on this real-GPU path.
    // (Formerly `await using` around the body: it still runs after the last it has called dispose().)
    afterAll(() => pipeline?.[Symbol.asyncDispose]());

    for (const item of CASES) {
      // A case with no reference value for this environment, and not in a mode that makes one,
      // SKIPs explicitly (the reason and how to make one are in the registration warning).
      it.skipIf(references.lacksReference(item.name))(`${item.name}: ${item.why}`, async () => {
        plainSha.set(item.name, (await runCase(opened(), config, item, reference)).sha);
      });
    }

    // --- onEvent (the observation hook for generation events) ----------------
    //
    // Four points are pinned:
    //  ① Passing `onEvent` switches the DiT loop to the **host path** (the resident path is one
    //     batch + a single fence and cannot be observed midway — ADR 0054). The evidence is
    //     whether the diagnostics have `lastRun`, taken independently of the events arriving.
    //  ② Even so, **the WAV's sha256 is the same value as voice-clone above** (the only place
    //     where the MUST of bit identity between the 2 paths is observable from the public API).
    //     The subscriber even destroys its copy with NaN on every step.
    //  ③ `denoise-step` fires `config.steps` times, **per step** (not the forward count inside
    //     CFG).
    //  ④ A throw from the callback takes the whole generation down (the means of aborting at step
    //     granularity).
    const withEvents = CASES[0];
    // The sha identity is half of this stage's claim, so it SKIPs as a whole where there is no
    // reference value.
    it.skipIf(references.lacksReference(withEvents.name))(
      `onEvent: the WAV sha is identical on the host path too (${REFERENCE_KNOBS.steps} step events)`,
      async () => {
        const log: string[] = [];
        const shapes = new Set<string>();
        const lengths = new Set<number>();
        let firstStep: (() => { data: Float32Array<ArrayBuffer> }) | undefined;
        let firstStepHead: number[] = [];
        const before = runs.length;
        const twinSha = plainSha.get(withEvents.name);
        assert(
          twinSha !== undefined,
          `no measured sha of the plain path (the ${withEvents.name} step did not run)`,
        );
        const { audio } = await runCase(opened(), config, withEvents, reference, {
          twinSha,
          onEvent: (event) => {
            if (event.kind === "stage") {
              log.push(`stage:${event.component}:${event.at}`);
              return;
            }
            log.push(`step:${event.step}/${event.steps}@${event.t}`);
            const snapshot = event.copyLatents();
            shapes.add(snapshot.shape.join("x"));
            lengths.add(snapshot.data.length);
            // Destroy the copy. Had the internal array been handed out, later steps would be
            // contaminated with NaN and the WAV sha would break (the waveform's non-finite check
            // would fail first).
            snapshot.data.fill(Number.NaN);
            if (event.step === 1) {
              firstStep = event.copyLatents;
              firstStepHead = Array.from(event.copyLatents().data.slice(0, 8));
            }
          },
        });

        // ① Evidence of the host path (every dit run goes through an arena = not `enqueue`).
        const ditRuns = runs.slice(before).filter((run) => run.component === "dit");
        assertEquals(ditRuns.length, audio.forwards, "the dit run count does not match forwards");
        assertFalse(
          ditRuns.some((run) => !run.hasArena),
          "a dit run has no lastRun = still the resident path (onEvent did not switch to the host path)",
        );

        // ③ The event sequence matches the execution structure exactly (around each stage + one
        // event per step).
        const schedule = tSchedule(config.steps, config.initScale);
        assertEquals(
          log,
          [
            "stage:backbone:start",
            "stage:backbone:end",
            "stage:text-proj:start",
            "stage:text-proj:end",
            "stage:caption-proj:start",
            "stage:caption-proj:end",
            "stage:codec-encoder:start",
            "stage:codec-encoder:end",
            "stage:speaker:start",
            "stage:speaker:end",
            "stage:duration:start",
            "stage:duration:end",
            // Stage 6' (the condition-side K/V projection — ADR 0114) opens and closes only once
            // per generation, before the `dit` stage.
            "stage:dit-context:start",
            "stage:dit-context:end",
            "stage:dit:start",
            ...Array.from(
              { length: config.steps },
              (_, at) => `step:${at + 1}/${config.steps}@${schedule[at]}`,
            ),
            "stage:dit:end",
            "stage:codec-decoder:start",
            "stage:codec-decoder:end",
          ],
          "the event sequence does not match the execution structure (around stages / step count / t values)",
        );
        // Inside CFG one step runs 3 forwards — events fire per step, not per forward.
        assertFalse(
          audio.forwards <= config.steps,
          `forwards ${audio.forwards} is at most steps ${config.steps} (without CFG active this does` +
            ' not prove "per step")',
        );

        // ② The intermediate latents have the shape `[frames, latentDim]`.
        assertEquals(shapes, new Set([`${audio.frames}x${config.latentDim}`]));
        assertEquals(lengths, new Set([audio.frames * config.latentDim]));
        // Called after generation finished, it still returns step 1's values (lazy, but bound when
        // created).
        assertEquals(
          Array.from((firstStep as () => { data: Float32Array<ArrayBuffer> })().data.slice(0, 8)),
          firstStepHead,
          "copyLatents called after generation does not return step 1's values",
        );
      },
    );

    it("onEvent: a throw from the callback takes the whole generation down (abort at step granularity)", async () => {
      const seen: string[] = [];
      const abortAt = 2;
      await assertRejects(
        () =>
          opened().generate({
            text: CASES[1].text,
            seed: CASES[1].seed,
            onEvent: (event) => {
              seen.push(
                event.kind === "stage"
                  ? `stage:${event.component}:${event.at}`
                  : `step:${event.step}`,
              );
              if (event.kind === "denoise-step" && event.step === abortAt) {
                throw new Error("aborted by the subscriber");
              }
            },
          }),
        Error,
        "aborted by the subscriber",
      );
      // The exception is not swallowed, so it stops in the DiT stage — the codec never opens.
      assertEquals(seen.at(-1), `step:${abortAt}`, "the aborted step is not the last event");
      assertFalse(seen.includes("stage:dit:end"), "aborted, yet the stage ended normally");
      assertFalse(
        seen.some((entry) => entry.startsWith("stage:codec-decoder")),
        "aborted, yet codec decode was entered",
      );
      // This checks that the later `await using` release still works after an aborted
      // generation.
    });

    // --- Serialized dispose (flush-before-destroy) ----------------------------
    //
    // `serial_test.ts` pins `concurrency/serial.ts` on its own, but that `Pipeline` sits on that
    // chain correctly is visible only from the public surface. **This step goes last** (it
    // releases the pipeline).
    it("dispose waits for the in-flight work to finish, and a second call returns the same completion", async () => {
      const pipeline = opened();
      const order: string[] = [];
      const running = pipeline.generate({ text: CASES[1].text, seed: CASES[1].seed }).then(() => {
        order.push("generate");
      });
      const disposal = pipeline.dispose();
      assertStrictEquals(
        pipeline.dispose(),
        disposal,
        "the second dispose returns a different completion",
      );
      await Promise.all([running, disposal.then(() => order.push("dispose"))]);
      // If the disposal resolved first, the consumer would treat it as "released" and move on
      // (while the GPU is still generating).
      assertEquals(order, ["generate", "dispose"]);

      // The disposed check happens **before entering the chain** (at call time) — it fails
      // without waiting for the earlier generation to settle. It is an `async` method, so what is
      // observed is a rejection (not a synchronous throw).
      await assertRejects(
        () => pipeline.generate({ text: CASES[1].text }),
        Error,
        "cannot generate after dispose",
      );
      await assertRejects(
        () => pipeline.generateLatent({ text: CASES[1].text }),
        Error,
        "cannot generate after dispose",
      );
    });
  },
);

// The guard that keeps "this environment has no reference values yet" from going silently green
// (the same approach as ADR 0005). Only the live cases above are counted (rows of retired cases
// left behind do not make it green).
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
