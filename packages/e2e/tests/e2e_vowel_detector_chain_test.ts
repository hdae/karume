// Check of the **full real-audio chain** of the real-weight vowel-recognition CRNN (WAV → feature
// extraction → real GPU → `.lab`) (step 3 of ADR 0005).
//
// Both ends of the chain (`src/audio/wav.ts` / `src/vowel-detector/features.ts` /
// `src/vowel-detector/postprocess.ts`) are models implementations, so this check lives on the
// models side. It used to sit in `packages/runtime/tests/e2e_vowel_detector_test.ts`, but there a
// runtime test imported models implementations by relative path — a **reversed dependency** — so
// the whole chain moved here (the split SigLIP2 went through first — the head of
// `e2e_siglip2_image_preprocess_real_test.ts`).
//
// ## Split with the runtime side
//
// - **Numeric comparison of the logits** (4 synthetic golden cases, torch CPU expected values) was
//   removed in the 0.15 test selection (`e2e_vowel_detector_test.ts` — the chain's sha rows cover
//   it). With `.lab` alone, a change where "the logits moved but the labels did not change" would
//   slip through, so that test pinned it with a tolerance of the 1e-5 order.
// - **Exact match of `.lab`** (real audio, full chain) is **here**. It is a discrete label
//   sequence, so it has no tolerance (the same philosophy as the PNG / WAV sha256 checks — pin the
//   output artifact itself).
//
// The middle of the chain (host-side feature extraction and post-processing) is covered by
// `vowel_detector_host_test.ts` through parity with the canonical Python implementation; this file
// checks that **a meaningful `.lab` comes out** when those two are connected through real GPU
// inference.
//
// ## One series (one graph; the length is a symbol)
//
// The time axis is the symbol `T` (20ms grid) and the graph input is `[1, 2T, 83]` (ADR 0056 /
// 0057). So **all 4 audio clips run the same graph**, each at its real length. The binding gets
// no explicit seed and is **solved from the input shape** (binding through the derived dimension
// `2T` = the path of ADR 0057; this check is the only place that exercises it in a real run. The
// distribution pipeline, conversely, passes an explicit seed).
//
// **There is not a single pad element** (only the one leftover frame of an odd frame count is
// cut off). Earlier, only the distribution used length buckets + right zero padding, and since the
// pad moved the trailing logits by O(1), its `.lab` disagreed with this check (measured: max abs
// diff 2.2–3.2 even with 2 pad frames = 40ms). Now that the buckets are gone, **the
// distribution's `.lab` matches this one exactly as well** (each case below checks this as step
// ②': the distribution pipeline's `.lab` must equal this direct real-length path's `.lab`).
// ## Assets
//
// This is an implementation-time check (ADR 0139 decision 6): the vowel detector has no published
// distribution and its WAVs are not in the test-asset dataset, so the whole chain reads the series
// and the local corpus below and runs only with `KARUME_SERIES=1` (`SERIES_CHECKS`); the default
// lanes skip it explicitly.
//
// - Graph: `outputs/series/vowel-detector-crnn-epoch3/` (`outputs/` of `.gitignore`). The
//   generation command is canonically {@link GENERATE_COMMAND} as written.
// - Real audio: `outputs/misc/corpus/vowel-<case>.wav` (16kHz mono). The generation script is
//   `examples/irodori/eval-audio.ts` (the canonical text / seed / resampling live there): 48kHz
//   audio rendered by the Japanese TTS (Irodori), decimated to 1/3. The script writes to
//   `outputs/bench/vowel-detector/<date>_eval-audio/`, and the adopted clips, **frozen by a manual
//   copy into `misc/corpus/`**, are the input of this check. **The identity of the material is
//   pinned by sha256** — once the audio is regenerated and frozen again, the expected `.lab` has to
//   be recorded again too.
// - mel basis: `tests/fixtures/vowel-detector/parity.json` (git-tracked; upstream
//   `assets/feature_config.json` as is). The distribution ships the same matrix as a safetensors
//   in the `assets` slot (`dist.py --pipeline vowel-detector`), but this check does not go through
//   the distribution.
// - Distribution form: `models/karume-vowel-detector/` (untracked). Only step ②' reads it, through
//   `Pipeline.fromAssets` → `detect`. The direct path keeps the fixture mel basis, so
//   ②' also pins that the distributed mel basis yields the same `.lab`. When it is missing, ②'
//   alone is skipped with a warning (the direct-path gates still run).
//
// Where the assets are missing it **SKIPs explicitly** (independently per series and per audio
// clip). The "all SKIP is an explicit FAIL" guard of ADR 0005 looks only at *whether a GPU adapter
// exists* and is independent of this SKIP. The enumeration and completeness of the golden cases (a
// half-present asset set FAILs) belong to the "asset completeness" test on the runtime side — this
// file needs only the graph itself and the WAVs, so the series' presence is also judged by
// `model.krm`.

import { isNotFound, readFile, readTextFile, statSync } from "@karume/test-support/fs";
import { describe, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { parseManifest, resolveSelection } from "@karume/hub";
import type { Manifest } from "@karume/hub";
import { acquireGpu, prepareContainer, type PreparedModel } from "@karume/runtime";
import { decodeWav } from "../../models/src/audio/wav.ts";
import {
  extractFeatures,
  FEATURE_DIM,
  SAMPLE_RATE,
} from "../../models/src/vowel-detector/features.ts";
import { logitsToSegments, toLab } from "../../models/src/vowel-detector/postprocess.ts";
import { Pipeline } from "../../models/src/vowel-detector/pipeline.ts";
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { openResults, recordFailure } from "../../runtime/tests/helpers/results.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import {
  f32ArtifactBytes,
  openReferences,
  referenceMismatchMessage,
  type ReferenceOutcome,
  registerReferenceGate,
  settleOrObserve,
} from "../../runtime/tests/helpers/reference.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";

/**
 * Declaration of one real audio clip (the graph is shared by all 4 — only the length differs).
 *
 * `frames` / `length` / `sha256` / `lab` are **all measured values**; when the audio is
 * regenerated, all four have to be recorded again ({@link AUDIO_COMMAND} →
 * {@link GENERATE_COMMAND}, in that order).
 */
type Case = {
  /** Audio case name (1:1 with `CASES` of `examples/irodori/eval-audio.ts`). */
  readonly name: string;
  /** What the material measures (a field for people to read — a copy of the script's `why`). */
  readonly why: string;
  /** Number of 10ms frames that `extractFeatures` returns (determined by the waveform length). */
  readonly frames: number;
  /** Input length T10 passed to the graph (= `frames` rounded down to an even number). */
  readonly length: number;
  /** sha256 of the 16kHz WAV (checked before the run: it is the material the golden came from). */
  readonly sha256: string;
  /** The full expected `.lab` (confirmed to match both the real GPU and the torch CPU output). */
  readonly lab: string;
};

/**
 * The cases that run. **They are pinned here, not enumerated** — picking up whatever has been
 * generated would make an environment that forgot to generate some of them "green but unverified".
 *
 * The lengths are spread over 2.88s / 3.48s / 8.04s / 12.40s (**the same 18-node graph** runs
 * under 4 bindings — back when the graph was unrolled, these were 4 separate graphs of 10,812 /
 * 13,092 / 30,420 / 46,988 nodes). Only `vowels` also has its content checked
 * ({@link VOWEL_SEQUENCE}).
 */
const CASES: readonly Case[] = [
  {
    name: "short",
    why: "2-second class, 1 sentence (「ありがとうございます。」)",
    frames: 285,
    length: 284,
    sha256: "0b48165fd11bf34c1f2b7904e4d53b5efa3f52c9baefc6a54566f850a9a30dd8",
    lab: `0.0000000 0.2000000 a
0.2000000 0.5000000 o
0.5000000 0.6400000 a
0.6400000 0.7000000 i
0.7000000 1.7400000 a
1.7400000 2.3800000 u
2.3800000 2.8000000 i
2.8000000 2.8400000 pau
`,
  },
  {
    name: "vowels",
    why: "3-second class, 5 vowels read one by one (「あ、い、う、え、お。」)",
    frames: 345,
    length: 344,
    sha256: "ae3712e01da6906a214b46c52490d106e1dfa598528e611e8f0f85ce6207a829",
    lab: `0.0000000 0.0400000 pau
0.0400000 0.4000000 a
0.4000000 0.9600000 i
0.9600000 1.6000000 u
1.6000000 2.3000000 e
2.3000000 3.4400000 o
`,
  },
  {
    name: "mid",
    why: "8-second class, 3 sentences",
    frames: 801,
    length: 800,
    sha256: "b63d35c12d9d60af44754db209b907f03e3fd5f46f25e6bb0a7ccb5195df734a",
    lab: `0.0000000 0.0400000 pau
0.0400000 0.1600000 o
0.1600000 0.2200000 N
0.2200000 0.5400000 i
0.5400000 0.8800000 a
0.8800000 0.9200000 pau
0.9200000 1.8000000 o
1.8000000 2.0200000 a
2.0200000 2.1800000 o
2.1800000 2.3200000 e
2.3200000 2.5200000 o
2.5200000 2.7200000 i
2.7200000 2.9000000 e
2.9000000 2.9600000 N
2.9600000 3.0800000 i
3.0800000 3.2000000 e
3.2000000 3.3400000 u
3.3400000 3.7400000 e
3.7400000 4.6000000 a
4.6000000 4.6600000 N
4.6600000 4.8200000 o
4.8200000 4.9600000 i
4.9600000 5.0800000 e
5.0800000 5.2000000 a
5.2000000 5.3400000 e
5.3400000 5.4400000 u
5.4400000 5.5600000 i
5.5600000 5.8600000 a
5.8600000 5.9400000 pau
5.9400000 6.5600000 o
6.5600000 6.9000000 i
6.9000000 7.0200000 e
7.0200000 7.1400000 u
7.1400000 7.2600000 a
7.2600000 7.6000000 o
7.6000000 7.6600000 i
7.6600000 7.9000000 a
7.9000000 8.0000000 pau
`,
  },
  {
    name: "long",
    why: "12-second class, 2 sentences (the longest binding in this check, T=618)",
    frames: 1237,
    length: 1236,
    sha256: "28013dab4551c353d3f95e51733e69d71de43a719bf3c4f120eaa1082013f59b",
    lab: `0.0000000 0.0400000 pau
0.0400000 0.2400000 o
0.2400000 0.5600000 e
0.5600000 0.9400000 a
0.9400000 1.0000000 pau
1.0000000 1.3800000 o
1.3800000 1.5000000 i
1.5000000 1.6000000 N
1.6000000 1.7400000 o
1.7400000 2.0200000 a
2.0200000 2.1600000 i
2.1600000 2.3800000 o
2.3800000 2.4800000 i
2.4800000 2.6200000 a
2.6200000 2.7400000 i
2.7400000 3.0200000 e
3.0200000 3.0800000 pau
3.0800000 3.4000000 u
3.4000000 3.5400000 i
3.5400000 3.6600000 o
3.6600000 3.9200000 a
3.9200000 4.0400000 i
4.0400000 4.3400000 a
4.3400000 4.4000000 i
4.4000000 4.5200000 u
4.5200000 4.6600000 a
4.6600000 4.7600000 i
4.7600000 4.9000000 o
4.9000000 5.1200000 u
5.1200000 5.2200000 i
5.2200000 5.3800000 a
5.3800000 7.1800000 i
7.1800000 7.3400000 a
7.3400000 7.4400000 i
7.4400000 7.6000000 u
7.6000000 7.6600000 N
7.6600000 7.8000000 e
7.8000000 7.9800000 o
7.9800000 8.3600000 a
8.3600000 8.4600000 i
8.4600000 8.6000000 u
8.6000000 8.6800000 N
8.6800000 8.8200000 e
8.8200000 9.1000000 o
9.1000000 9.1600000 pau
9.1600000 9.5400000 o
9.5400000 9.6800000 a
9.6800000 9.8200000 i
9.8200000 9.9600000 e
9.9600000 10.1400000 u
10.1400000 10.1800000 N
10.1800000 10.3400000 e
10.3400000 10.5000000 o
10.5000000 10.6000000 i
10.6000000 10.7400000 e
10.7400000 10.8400000 i
10.8400000 10.9400000 u
10.9400000 11.3400000 o
11.3400000 11.4800000 a
11.4800000 11.5800000 i
11.5800000 11.7000000 a
11.7000000 11.9400000 e
11.9400000 12.0000000 i
12.0000000 12.2600000 a
12.2600000 12.3600000 pau
`,
  },
];

/**
 * The label sequence of the `vowels` case's `.lab` with silence removed.
 *
 * This **claims something different** from the exact match of the expected `.lab` — that one says
 * "the same values as last time come out", this one says "reading the 5 vowels in order yields the
 * 5 vowels in order" = the claim that the chain means something. When the audio is regenerated and
 * the expected `.lab` is recorded again, this one stops the accident of pinning a broken output.
 */
const VOWEL_SEQUENCE: readonly string[] = ["a", "i", "u", "e", "o"];
const VOWEL_CASE = "vowels";

/** 8 classes (the count of `LIPSYNC_CLASSES` — the last axis of the output). */
const CLASS_COUNT = 8;

const SERIES_NAME = "vowel-detector-crnn-epoch3";
const SERIES_ROOT = new URL(`../../../outputs/series/${SERIES_NAME}/`, import.meta.url);
/** Real-audio corpus (a frozen copy of a host asset — once deleted, regenerate and refreeze it). */
const CORPUS_DIR = new URL("../../../outputs/misc/corpus/", import.meta.url);
const FIXTURE_PATH = new URL(
  "../../models/tests/fixtures/vowel-detector/parity.json",
  import.meta.url,
);
const MODEL_FILE = "model.krm";
/**
 * The graph name in the container (`vowel_detector.export` writes `crnn`). Spelled here, not
 * looked up in the series-graph table, which ADR 0139 decision 5 retires.
 */
const MODEL_GRAPH = "crnn";
const INPUT_NAME = "features";

const audioFile = (entry: Case): string => `vowel-${entry.name}.wav`;

/**
 * Where the logits and the outcomes go (`outputs/verify/<environment key>/<date>_vowel-detector/`
 * — safe to delete).
 */
const results = openResults("vowel-detector");
/**
 * sha256 reference values of the logits (rows per environment key — `helpers/reference.ts`). The
 * case ID is the case name itself.
 *
 * This **claims something different** from the exact match of `.lab` — `.lab` is a discrete
 * sequence after argmax, so it passes as long as the labels do not change even when the logits
 * move. This pins the logits themselves (the f32 bit pattern) and says "not a single bit moved on
 * the same machine". The artifact is a safetensors holding the logits alone ({@link logitsArtifact}
 * — `f32ArtifactBytes`: tensor name = the graph's output name, no metadata),
 * and the sha is taken over the bytes of that file.
 */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/vowel-detector.json", import.meta.url),
);

/** File name of the logits artifact (relative to the results directory). */
const logitsArtifact = (entry: Case): string => `${entry.name}-logits.safetensors`;

/** Distribution form (output of `karume dist --pipeline vowel-detector`) read by step ②'. */
const DIST_DIR = new URL("../../../models/karume-vowel-detector/", import.meta.url);
/** Command that builds the distribution form (pasted into the skip warning). */
const DIST_COMMAND =
  "cd tools/export-recipes && uv run python -m vowel_detector.export && " +
  "uv run python dist.py --pipeline vowel-detector";

/** Generation command to paste as is on SKIP (one graph — no length needs to be given). */
const GENERATE_COMMAND = "cd tools/export-recipes && uv run python -m vowel_detector.export";

/**
 * Command that regenerates the real audio itself (the canonical text / seed live in the script).
 * The script writes to `outputs/bench/vowel-detector/<date>_eval-audio/`, so the adopted clips are
 * **frozen by a manual copy** into {@link CORPUS_DIR}.
 */
const AUDIO_COMMAND = "deno task demo:eval-audio --source <path to the Irodori distribution>";

/** The mel basis fixture (fields other than `melBasis` are not used by this check). */
type Fixture = { readonly melBasis: number[][] };

const melBasis: Float32Array<ArrayBuffer> = Float32Array.from(
  (JSON.parse(await readTextFile(FIXTURE_PATH)) as Fixture).melBasis.flat(),
);

/**
 * Whether the file exists.
 * MUST: propagate anything other than NotFound — reading a permission error or the like as "the
 * asset is missing" would let a check that never ran turn quietly green as a SKIP.
 */
const exists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/**
 * Checks that the graph's time axis was baked as **symbols** (input `2T` / output `T`).
 *
 * MUST: keep this. The old form baked at a fixed length has the same input / output names and
 * ranks, so even after a replacement "only the case of that one length" stays green (the other 3
 * look as if they failed for some other reason).
 *
 * NOTE: the runtime-side e2e has the same check (it exercises it on the synthetic golden side). It
 * is **not shared** so that one of them disappearing never lets the other pass silently.
 */
const assertSymbolicTimeAxis = (parsed: PreparedModel): void => {
  assertEquals(parsed.graph.symbols, ["T"], `${SERIES_NAME} symbolic dimensions`);
  assertEquals(
    parsed.graph.inputs[0].shape,
    [1, "2T", FEATURE_DIM],
    `${SERIES_NAME} declaration of input '${INPUT_NAME}'`,
  );
  assertEquals(
    parsed.graph.values[parsed.graph.outputs[0]].shape,
    [1, "T", CLASS_COUNT],
    `${SERIES_NAME} output declaration`,
  );
};

/**
 * Reads the distribution assets from disk (local stand-in for `fetchAssets` — no fetch layer).
 * MUST: zero-length parts are listed too (the index is the id inside the container; skipping one
 * shifts every later part).
 */
const loadDistAssets = async (
  manifest: Manifest,
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const selection = resolveSelection(manifest, {});
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const name of Object.keys(selection.containers)) {
    const { parts } = selection.containers[name];
    for (const [index, ref] of parts.entries()) {
      assets = { ...assets, [`${name}[${index}]`]: await readFile(new URL(ref.path, DIST_DIR)) };
    }
  }
  for (const name of Object.keys(selection.assets)) {
    const { path } = selection.assets[name];
    assets = { ...assets, [name]: await readFile(new URL(path, DIST_DIR)) };
  }
  return assets;
};

/**
 * Whether the series is present (read synchronously, since registration needs it). MUST: only under
 * `SERIES_CHECKS` — the default lanes do not read the series (ADR 0139 decision 6).
 */
const available = SERIES_CHECKS && modelPresent(new URL(MODEL_FILE, SERIES_ROOT));

if (!SERIES_CHECKS) {
  console.warn(
    "[karume] The vowel-detector real-audio chain E2E is an implementation-time series check " +
      "(no published distribution), so it is skipped. Run it with KARUME_SERIES=1 on a machine " +
      `that holds ${SERIES_ROOT.pathname}.`,
  );
} else if (!available) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} has no exported assets, so the full real-audio chain E2E is ` +
      `SKIPPED (the weights are outside the repository). Generate: ${GENERATE_COMMAND}`,
  );
}

/** Distribution manifest (`undefined` = absent: step ②' is skipped explicitly, see the warning). */
const distManifestText = await readTextIfPresent(new URL("karume.json", DIST_DIR));
if (available && distManifestText === undefined) {
  console.warn(
    `[karume] ${DIST_DIR.pathname} is missing, so the distribution .lab check (step ②') of the ` +
      `vowel-detector chain E2E is skipped. Build: ${DIST_COMMAND}`,
  );
}

/** The cases that run (series, WAV and GPU all present — the range the reference guard counts). */
const RUNNABLE_CASES: Case[] = [];

for (const entry of CASES) {
  /** Real-audio check. It runs only when **both** the series and the WAV are present. */
  const audioAvailable = available && exists(new URL(audioFile(entry), CORPUS_DIR));
  if (audioAvailable && GPU_AVAILABLE) RUNNABLE_CASES.push(entry);

  if (available && !audioAvailable) {
    console.warn(
      `[karume] Real-audio case (${entry.name}) is SKIPPED (${audioFile(entry)} is missing). ` +
        `Generate: ${AUDIO_COMMAND}`,
    );
  }

  test.skipIf(!audioAvailable || !GPU_AVAILABLE)(
    `Vowel detection, full real-audio chain: ${entry.name} — WAV → features → real GPU → .lab (${entry.why})${SERIES_CHECKS_OPT_IN}`,
    async () => {
      // The elapsed time runs from reading the WAV to just before recording the outcome (a failed
      // run is recorded over the same range).
      const started = performance.now();
      const elapsedMs = (): number => Math.round(performance.now() - started);
      /**
       * Outcome of ③ (taken only after `.lab` passed — kept outside so a mismatch fails after the
       * record).
       */
      let outcome: ReferenceOutcome;
      try {
        const wavBytes = await readFile(new URL(audioFile(entry), CORPUS_DIR));
        // ① The audio the expected `.lab` came from and the audio read now are identical. A
        // **difference no tolerance absorbs** (the script was rerun but the expected values were
        // not recorded again) fails by name, before the run.
        assertEquals(
          await sha256Hex(wavBytes),
          entry.sha256,
          `${audioFile(entry)} differs from the audio the expected .lab came from (record again: ${AUDIO_COMMAND})`,
        );

        const wav = decodeWav(wavBytes);
        assertEquals(wav.sampleRate, SAMPLE_RATE, `${audioFile(entry)} sample rate`);
        const features = extractFeatures(wav.data, melBasis);
        assertEquals(features.frames, entry.frames, `${entry.name} number of 10ms frames`);
        // The output is on a 20ms grid, so the one leftover frame of an odd frame count is dropped
        // and the run uses an even length (a truncation, not a pad — the "not a single pad
        // element" of the header).
        const usable = features.frames - (features.frames % 2);
        assertEquals(usable, entry.length, `${entry.name} frame count rounded to even`);

        const opened = await openSeriesContainer(new URL(MODEL_FILE, SERIES_ROOT));
        const parsed = prepareContainer(opened, MODEL_GRAPH);
        assertSymbolicTimeAxis(parsed);
        const [outputName] = parsed.graph.outputs;

        const gpu = await acquireGpu();
        let logits: Float32Array<ArrayBuffer>;
        try {
          // This path may write the logits' sha as this machine's row, so check first that the
          // adapter the key came from is the one running (mixing them up on a multi-GPU machine
          // would write another machine's row).
          assertAdapterMatchesEnvironment(gpu);
          const session = await parsed.createContainerSession(gpu);
          try {
            const output = (
              await session.run({
                [INPUT_NAME]: {
                  dtype: "f32",
                  shape: [1, usable, FEATURE_DIM],
                  data: features.data.slice(0, usable * FEATURE_DIM),
                },
              })
            )[outputName];
            // Narrow by the discriminant (a cast to Float32Array passes silently on a dtype drift).
            assert(output.dtype === "f32", `${entry.name}: logits dtype is ${output.dtype}`);
            assertEquals(output.shape, [1, usable / 2, CLASS_COUNT], `${entry.name}: output shape`);
            logits = output.data;
          } finally {
            await session.dispose();
          }
        } finally {
          gpu.destroy();
        }

        // ② Exact match of `.lab` (a discrete check with no tolerance).
        const segments = logitsToSegments(logits, usable / 2);
        assertEquals(toLab(segments), entry.lab, `${entry.name} .lab`);

        if (entry.name === VOWEL_CASE) {
          // A claim separate from the match of the expected `.lab` ({@link VOWEL_SEQUENCE}).
          assertEquals(
            segments.map((segment) => segment.label).filter((label) => label !== "pau"),
            [...VOWEL_SEQUENCE],
            "「あ、い、う、え、お。」 did not yield the 5 vowels in order",
          );
        }

        // ②' The distribution pipeline (`fromAssets` → `detect`) must give exactly this direct
        // real-length `.lab` — no pad, no length bucket. No tolerance: `.lab` is the discrete
        // artifact itself. On mismatch suspect the distribution side (pad or rounding reintroduced
        // in the pipeline, or `pipelineConfig` drifting from the graph).
        if (distManifestText !== undefined) {
          const manifest = parseManifest(distManifestText);
          await using pipeline = await Pipeline.fromAssets({
            manifest,
            assets: await loadDistAssets(manifest),
          });
          assertEquals(
            (await pipeline.detect(wav.data)).lab,
            toLab(segments),
            `${entry.name}: the .lab via the distribution differs from the direct real-length path`,
          );
        }

        // ③ The logits' sha. Taken only after ① and ② passed (no reference row is made from an
        // output whose `.lab` broke). In compare mode, when this machine has no row, only the
        // comparison is skipped and the measured sha is recorded (settleOrObserve).
        outcome = await settleOrObserve(references, results, {
          id: entry.name,
          artifact: logitsArtifact(entry),
          bytes: f32ArtifactBytes(outputName, [1, usable / 2, CLASS_COUNT], logits),
        });
      } catch (cause) {
        // Record the outcome before failing (leaving without one lets the outcome of an earlier
        // run of the same day stay in its place). An I/O failure of the record does not replace
        // the original exception (recordFailure).
        await recordFailure(results, {
          id: entry.name,
          status: "fail",
          elapsedMs: elapsedMs(),
          note: cause instanceof Error ? cause.message : String(cause),
        });
        throw cause;
      }
      await results.record({ id: entry.name, ...outcome.fields, elapsedMs: elapsedMs() });
      // Fail after the outcome (measured sha) is recorded — throwing first would leave the
      // measured sha out of the results.
      if (outcome.settlement?.check.status === "fail") {
        throw new Error(referenceMismatchMessage(entry.name, outcome.settlement, references));
      }
    },
  );
}

/**
 * Cases the sha check counts (series, WAV and GPU all present — decided synchronously at
 * registration).
 */
const CASE_IDS: readonly string[] = RUNNABLE_CASES.map((entry) => entry.name);
const RUNNABLE = CASE_IDS.length > 0;
if (RUNNABLE) references.warnMissing(CASE_IDS);
// The guard that keeps "this environment has no reference values yet" from being a silent green
// (the same practice as ADR 0005).
// The gate counts the implementation-time cases above, so it carries their opt-in suffix: a skipped
// gate in the default lanes says why.
describe(`Vowel-detector sha256 rows${SERIES_CHECKS_OPT_IN}`, () => {
  registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
});
