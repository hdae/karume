/**
 * E2E that runs a **component replacement** (`components` — ADR 0122 decision 7, ADR 0108 decision
 * 19) of Wan2.1's umT5 (`text_encoder`) through to real-GPU generation with local distribution
 * mirrors.
 *
 * The host test (`wan_components_test.ts`) covers admission and fetching with a fake HF and an
 * in-memory directory, but when the replacement is a **local directory**, the weight blocks are
 * read by range during Session construction (the text stage of `generate`), which the host test
 * cannot observe (the remainder of the ADR 0122 addendum "result of step d"). Four points are
 * covered here:
 *
 * ① **Construction**: `Pipeline.fromPretrained` with `text_encoder` replaced passes without giving
 *    the source of the Wan distribution a single `crossRepo` mapping for umT5's cross-repository
 *    target.
 * ② **Provenance**: the replacement's source is wrapped in a reader that records the paths and
 *    ranges it read. The ranges of all blocks bound by the replacement's `text_encoder` graph (data,
 *    scale, zero point — taken from the replacement's descriptor) are covered without gaps by reads
 *    from the replacement inside the text stage; during construction only the manifest and part 0
 *    (the descriptor) are read from the replacement, and after the text stage it is never read. The
 *    Wan source never reads the path of a text-encoder part (= nothing but the replacement supplied
 *    the text encoder's bytes — the cross-repository target has no mapping, so it cannot be opened
 *    at all). Assumption: every weight part of the replacement that holds data (part 1 onward)
 *    holds blocks the graph binds (a replacement with asset-only parts is out of scope — checked
 *    and failed before touching the GPU).
 * ③ **Generation**: the same request as the first GPU-path case of `e2e_wan_pipeline_test.ts` (the
 *    original text of the fixed prompt `boxing-cats`, seed 42, 2 steps, 832×480, 33 frames, the
 *    default negative, the `f16` preset) completes, with 0 non-finite values and the declared
 *    shape.
 * ④ **Control** (no GPU — fake GPU): with the same source, no replacement and no mapping, it fails
 *    in the prefetch of umT5's part 0 with a failure whose cause carries hub's "no cross-repository
 *    target" guidance (confirming that ① did not pass through a leftover mapping).
 *
 * The replacement is a local distribution the user chooses (a local experimental mirror), whose
 * directory comes in through the environment variable {@link SWAP_ENV} (tracked code does not name
 * a particular mirror — the TS test of ADR 0122 "affected files"). When it is unset, ①–③ SKIP
 * explicitly (opt-in). ④ does not depend on the replacement and needs only construction, so it runs
 * in the default lanes whenever the distribution is present.
 *
 * ## Observation (not a check)
 *
 * The output sha256 is not a check (the user chooses the replacement, so there can be no row per
 * environment). The sha256, the construction and stage times and the number of bytes read from the
 * replacement go to `outputs/verify/<environment key>/<date>_wan-swap/`, and the match / mismatch
 * with the row of the same request run with the original umT5 ({@link ORIGINAL_CASE_ID} of
 * `fixtures/references/wan.json`) is recorded too (the row is only looked up — not written, and not
 * used as the outcome of a comparison).
 *
 * Environments without the distribution SKIP explicitly (the all-skip guard
 * `distribution_gate_test.ts` turns an all-SKIP into a FAIL).
 */

import {
  isNotFound,
  readFile,
  readRange,
  readTextFile,
  realPathSync,
  statSync,
  writeFile,
} from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import {
  assert,
  assertEquals,
  assertInstanceOf,
  assertRejects,
  assertStringIncludes,
} from "@std/assert";
import {
  type DistributionSource,
  type FileRef,
  loadManifest,
  localDirectory,
  openContainerSource,
  parseManifest,
  resolveSelection,
} from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";
import { openContainer, type SessionDiagnostics } from "@karume/runtime";
import {
  type GeneratedVideo,
  frameToRgba,
  type GenerateEvent,
  Pipeline,
  type RunComponent,
} from "../../models/wan.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import { WAN_ASSEMBLE_COMMAND } from "../../models/tests/helpers/wan-distribution.ts";
import { verifyRunningGpu } from "../../runtime/tests/helpers/environment.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import { openReferences, sha256Hex } from "../../runtime/tests/helpers/reference.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";

/**
 * Environment variable that receives the replacement (a local distribution — a directory with
 * `karume.json`) (opt-in).
 */
const SWAP_ENV = "KARUME_WAN_SWAP_TEXT_ENCODER";
const SWAP_RAW = process.env[SWAP_ENV];
/** Directory of the replacement (unset or empty string → undefined = ①–③ SKIP). */
const SWAP_DIRECTORY = SWAP_RAW === undefined || SWAP_RAW === "" ? undefined : SWAP_RAW;

/** The distribution mirror (the default output directory of `dist.py --pipeline wan`). */
const DIST_ROOT = new URL("../../../models/karume-wan2.1/", import.meta.url);
/** Component name in the manifest (`TEXT_ENCODER` of `src/wan/pipeline.ts`). */
const TEXT_ENCODER = "text_encoder";
/**
 * Preset (explicit — when omitted it silently moves to the manifest's `defaultQuant`. It is the
 * `f16` preset, as on the GPU path of `e2e_wan_pipeline_test.ts`, so the request equals that of the
 * original umT5's row).
 */
const F16_QUANT = "f16";
const STEPS = 2;
/**
 * Defaults when `frames` / dimensions are omitted (`src/wan/pipeline.ts`) — used only for the shape
 * check.
 */
const DEFAULT_FRAMES = 33;
const DEFAULT_WIDTH = 832;
const DEFAULT_HEIGHT = 480;

/**
 * The one replacement generation. The request is the same as the first of `GPU_TEXT_CASES` in
 * `e2e_wan_pipeline_test.ts` (the original text of the fixed prompt, seed 42, 2 steps, default
 * negative / dimensions / frame count) — only the text encoder's weights differ.
 *
 * MUST: when the text, seed or step count changes, change the ID too and stop the comparison with
 * {@link ORIGINAL_CASE_ID} (it would compare the sha of different conditions).
 */
const SWAP_CASE = { id: "swap-text-2step-boxing-cats-seed42", prompt: "boxing-cats", seed: 42 };
/** sha row of the same request run with the original umT5 (only compared — an observation). */
const ORIGINAL_CASE_ID = "gpu-text-2step-boxing-cats-seed42";

const fileExists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const DIST_PRESENT = fileExists(new URL("karume.json", DIST_ROOT));
if (!DIST_PRESENT) {
  console.warn(
    `[karume] The distribution mirror ${DIST_ROOT.pathname} is missing, so the Wan text_encoder replacement e2e is SKIPPED. ` +
      `Assemble: ${WAN_ASSEMBLE_COMMAND} (the all-skip guard distribution_gate_test.ts turns an all-SKIP into a FAIL)`,
  );
}
if (SWAP_DIRECTORY === undefined) {
  console.warn(
    `[karume] The Wan text_encoder replacement generation (real GPU) is opt-in, so it is SKIPPED` +
      ` (run it with ${SWAP_ENV}=<directory of the local replacement distribution>)`,
  );
}

const references = openReferences(
  new URL("../../models/tests/fixtures/references/wan.json", import.meta.url),
);
const results = openResults("wan-swap");

/** Parts that hold data (zero-length parts are not fetched — hub does not fetch them either). */
const nonEmpty = (parts: readonly FileRef[]): readonly FileRef[] =>
  parts.filter((part) => part.size > 0);

/** The part list of the manifest's `text_encoder` (preset `quant`, default model). */
const textEncoderParts = async (root: URL, quant?: string): Promise<readonly FileRef[]> => {
  const manifest = parseManifest(await readTextFile(new URL("karume.json", root)));
  const container = resolveSelection(manifest, {
    ...(quant === undefined ? {} : { quant }),
    weights: [TEXT_ENCODER],
  }).containers[TEXT_ENCODER];
  assert(
    container !== undefined,
    `${root.pathname}: the manifest has no '${TEXT_ENCODER}' container`,
  );
  return container.parts;
};

/**
 * The part list of the Wan distribution's `text_encoder`. The precondition (that it is a
 * cross-repository reference) is checked here too — with a container in the same repository,
 * "passes without a mapping" would be trivial and ① and ④ would confirm nothing.
 */
const wanTextEncoderParts = async (): Promise<readonly FileRef[]> => {
  const parts = await textEncoderParts(DIST_ROOT, F16_QUANT);
  assert(
    parts.length > 0 && parts.every((part) => part.repo !== undefined),
    `the distribution ${DIST_ROOT.pathname}: '${TEXT_ENCODER}' is not a cross-repository reference to the umT5 distribution repository` +
      ` (a precondition of this e2e — reassemble: ${WAN_ASSEMBLE_COMMAND})`,
  );
  return parts;
};

/**
 * Location of one block the graph binds (the container's part number = index into the manifest's
 * part list, and the range inside the part).
 */
type BoundBlock = {
  readonly id: string;
  readonly part: number;
  readonly offset: number;
  readonly length: number;
};

/**
 * Opens the replacement's `text_encoder` container through range reads (only the 2 documents of
 * part 0 are read — no weights) and returns every block the graph binds (data, scale, zero point).
 * The selection is the same as how {@link textEncoderParts} is called for the replacement (default
 * model / quant). Graph name = the manifest's weights key (the writer's convention — container-v1
 * §2.1).
 */
const textEncoderBlocks = async (root: URL): Promise<readonly BoundBlock[]> => {
  const loaded = await loadManifest(nodeDirectory(root.pathname));
  const container = resolveSelection(loaded.manifest, { weights: [TEXT_ENCODER] }).containers[
    TEXT_ENCODER
  ];
  assert(
    container !== undefined,
    `${root.pathname}: the manifest has no '${TEXT_ENCODER}' container`,
  );
  const opened = await openContainer(
    { kind: "source", source: openContainerSource(loaded, container) },
    container.descriptor,
  );
  const graph = opened.graphs[TEXT_ENCODER];
  assert(
    graph !== undefined,
    `${root.pathname}: the '${TEXT_ENCODER}' container has no graph '${TEXT_ENCODER}'`,
  );
  return [...graph.supplies.values()].flatMap((supply) => [
    ...supply.blocks,
    ...(supply.scale === undefined ? [] : [supply.scale]),
    ...(supply.zeroPoint === undefined ? [] : [supply.zeroPoint]),
  ]);
};

/** When a read happened (during construction / inside the text stage / after the text stage). */
type Phase = "construction" | "text_encoder" | "after_text_encoder";

/**
 * One read the source's reader received (a whole-file read is offset 0, length = bytes returned).
 */
type Read = {
  readonly path: string;
  readonly offset: number;
  readonly length: number;
  readonly phase: Phase;
};

/**
 * Whether `reads` (reads of the same path — any order, overlaps allowed) cover
 * `[offset, offset + length)` without gaps.
 */
const covers = (reads: readonly Read[], offset: number, length: number): boolean => {
  const end = offset + length;
  let reached = offset;
  for (const read of [...reads].sort((a, b) => a.offset - b.offset)) {
    if (read.offset > reached) break;
    reached = Math.max(reached, read.offset + read.length);
    if (reached >= end) return true;
  }
  return reached >= end;
};

/**
 * Uses a local directory as the source and pushes the paths and ranges read onto `reads`. The
 * reader has the same 2 methods as `nodeDirectory` of `@karume/hub/node` (whole-file read +
 * positional read) — without the positional read, Session construction falls back from range reads
 * (seek) to whole-part reads (scan), and a different way of reading from the user's path would be
 * observed.
 *
 * WHY wrap the reader (`DirectoryAdapter`) rather than the source handle (`DistributionSource`):
 * the handle is an opaque value without public members (the MUST of hub's `src/source.ts`), so no
 * delegating wrapper can be built around it from outside. Wrapping the reader and passing it to
 * `localDirectory` is the only delegation that hub's public surface allows (`nodeDirectory` itself
 * has the same shape). NOTE: the reader's body is a copy of the 2 methods of `nodeDirectory`; the
 * wording of the wrapper for unreadable files, the abort handling of `readFile` and the label are
 * not copied (the source semantics — range reads, size bounds, cross-repository resolution — go
 * through the real `localDirectory`). Once hub exposes a delegation point for source handles, go
 * back to wrapping the handle.
 */
const recordingDirectory = (
  root: URL,
  label: string,
  reads: Read[],
  phase: () => Phase,
): DistributionSource =>
  localDirectory(
    {
      readFile: async (path, { signal }) => {
        const abort = signal === undefined ? {} : { signal };
        const bytes = await readFile(new URL(path, root), abort);
        reads.push({ path, offset: 0, length: bytes.byteLength, phase: phase() });
        return bytes;
      },
      // MUST: return exactly `length` bytes as a view spanning the whole buffer (the contract of
      // `DirectoryAdapter.readFileRange`).
      readFileRange: async (path, offset, length, { signal }) => {
        signal?.throwIfAborted();
        reads.push({ path, offset, length, phase: phase() });
        const at = new URL(path, root);
        return readRange(at, offset, length);
      },
    },
    { label },
  );

/**
 * Path from the environment variable → `file:` URL of the directory (with a trailing `/` — without
 * it `new URL(path, root)` points at a sibling).
 */
const directoryUrl = (path: string): URL => {
  let resolved: string;
  try {
    resolved = realPathSync(path);
  } catch (cause) {
    throw new Error(`${SWAP_ENV}='${path}': cannot resolve the directory`, { cause });
  }
  const url = new URL("file:///");
  url.pathname = `${resolved}/`;
  return url;
};

/**
 * The uint8 RGB of all output frames concatenated (the sha256 artifact — the rule of
 * `frameToRgba`).
 */
const rgbBytes = (video: GeneratedVideo): Uint8Array<ArrayBuffer> => {
  const plane = video.width * video.height;
  const out = new Uint8Array(video.frames * plane * 3);
  for (let frame = 0; frame < video.frames; frame += 1) {
    const rgba = frameToRgba(video, frame);
    for (let index = 0; index < plane; index += 1) {
      out.set(rgba.subarray(index * 4, index * 4 + 3), (frame * plane + index) * 3);
    }
  }
  return out;
};

const textOf = (pipeline: Pipeline, name: string): string => {
  const entry = pipeline.prompts.find((candidate) => candidate.name === name);
  assert(entry !== undefined, `the embedded assets have no '${name}'`);
  return entry.prompt;
};

test.skipIf(!DIST_PRESENT)(
  "Wan text_encoder replacement control (no GPU, fake GPU): without a replacement and without a mapping for umT5's cross-repository target, " +
    "the prefetch of umT5's part 0 fails with the 'no cross-repository target' guidance as its cause, and the Wan source does not read umT5's parts",
  async () => {
    const wanParts = await wanTextEncoderParts();
    const [part0] = wanParts;
    const reads: Read[] = [];
    const error = await assertRejects(
      () =>
        Pipeline.fromPretrained(
          recordingDirectory(DIST_ROOT, "wan-swap-control", reads, () => "construction"),
          // The failure is in the descriptor prefetch (before admission) — it never reaches the
          // stage that touches the GPU.
          { gpu: fakeGpuContext(fakeDevice()), textEncoder: "gpu", quant: F16_QUANT },
        ),
      Error,
      `${part0.path}: prefetch failed (repo ${part0.repo} @ ${part0.revision})`,
    );
    // The fetch layer keeps the source's guidance as the cause (hub's local.ts — it does not guess
    // a neighbouring directory).
    assertInstanceOf(error.cause, Error);
    assertStringIncludes(error.cause.message, `no cross-repo target for repo '${part0.repo}'`);
    assertStringIncludes(error.cause.message, "crossRepo");
    // NOTE: a safety net (almost no independent detection power) — cross-repository resolution
    // throws before any read, so as long as the failure-message check above passes, no read
    // happens. It turns red independently only on a regression where hub goes looking for the
    // cross-repository target in the Wan distribution's directory.
    const textEncoderPaths = new Set(wanParts.map((part) => part.path));
    assertEquals(
      reads.filter((read) => textEncoderPaths.has(read.path)).map((read) => read.path),
      [],
      "tried to read umT5's parts from the Wan source",
    );
  },
);

test.skipIf(SWAP_DIRECTORY === undefined || !DIST_PRESENT || !GPU_AVAILABLE)(
  `Wan text_encoder replacement (real GPU, opt-in ${SWAP_ENV}): construction passes without a mapping for umT5's cross-repository target, ` +
    "every bound block of the replacement is read in full from the replacement in the text stage (no other origin of the text encoder's bytes), " +
    "and the 2-step generation has 0 non-finite values and the declared shape",
  async () => {
    assert(SWAP_DIRECTORY !== undefined);
    const swapRoot = directoryUrl(SWAP_DIRECTORY);
    assert(
      fileExists(new URL("karume.json", swapRoot)),
      `${SWAP_ENV} directory ${swapRoot.pathname} has no karume.json (it must point at a local distribution)`,
    );
    const wanParts = await wanTextEncoderParts();
    // The replacement is `{ source }` only (model / quant are the replacement's defaults) — the
    // part list is taken with the same selection.
    const swapParts = await textEncoderParts(swapRoot);
    assert(
      swapParts.every((part) => part.repo === undefined),
      `the replacement ${swapRoot.pathname}: '${TEXT_ENCODER}' has a cross-repository reference (this e2e only covers a replacement directory` +
        " that holds the data)",
    );
    const swapWeightParts = nonEmpty(swapParts.slice(1));
    assert(swapWeightParts.length > 0, `the replacement's '${TEXT_ENCODER}' has no weight part`);
    // Precondition (checked before touching the GPU): every weight part holds blocks the graph
    // binds. If this breaks, the block coverage below misses "that part was never read" (asset-only
    // parts are not read by Session construction).
    const swapBlocks = await textEncoderBlocks(swapRoot);
    const boundPaths = new Set(swapBlocks.map((block) => swapParts[block.part].path));
    assertEquals(
      swapWeightParts.map((part) => part.path).filter((path) => !boundPaths.has(path)),
      [],
      `the replacement's '${TEXT_ENCODER}' has a weight part without any block the graph binds` +
        " (a replacement with asset-only parts is out of scope for this e2e)",
    );

    let deviceLost: string | undefined;
    const gpu = verifyRunningGpu(
      await acquireTestGpu({
        onDeviceLost: (info) => {
          deviceLost = `${info.reason}: ${info.message}`;
        },
      }),
    );
    try {
      await runRecordedCase(results, { id: SWAP_CASE.id }, async () => {
        let phase: Phase = "construction";
        const currentPhase = (): Phase => phase;
        const wanReads: Read[] = [];
        const swapReads: Read[] = [];
        const diagnostics = new Map<RunComponent, SessionDiagnostics>();

        // ① Give the Wan source not a single crossRepo (umT5's cross-repository target cannot be
        // opened from anywhere).
        const constructionStarted = performance.now();
        await using pipeline = await Pipeline.fromPretrained(
          recordingDirectory(DIST_ROOT, "wan-swap", wanReads, currentPhase),
          {
            gpu,
            textEncoder: "gpu",
            quant: F16_QUANT,
            components: {
              [TEXT_ENCODER]: {
                source: recordingDirectory(swapRoot, swapRoot.pathname, swapReads, currentPhase),
              },
            },
            onRunDiagnostics: (component, diagnosed) => diagnostics.set(component, diagnosed),
          },
        );
        const constructionMs = performance.now() - constructionStarted;

        const stageStarted = new Map<string, number>();
        const stageMs: Record<string, number> = {};
        const onEvent = (event: GenerateEvent): void => {
          if (event.kind !== "stage") return;
          if (event.component === TEXT_ENCODER) {
            phase = event.at === "start" ? "text_encoder" : "after_text_encoder";
          }
          const now = performance.now();
          if (event.at === "start") stageStarted.set(event.component, now);
          else stageMs[event.component] = now - (stageStarted.get(event.component) ?? now);
        };
        const generateStarted = performance.now();
        const video = await pipeline.generate({
          prompt: textOf(pipeline, SWAP_CASE.prompt),
          seed: SWAP_CASE.seed,
          steps: STEPS,
          onEvent,
        });
        const generateMs = performance.now() - generateStarted;
        assertEquals(phase, "after_text_encoder", "the text stage's start / end did not arrive");
        assert(diagnostics.has(TEXT_ENCODER), "the diagnostics of the umT5 run did not arrive");

        // ② What was read from the replacement.
        const swapPathSet = new Set(swapParts.map((part) => part.path));
        assertEquals(
          [...new Set(swapReads.map((read) => read.path))].filter(
            (path) => path !== "karume.json" && !swapPathSet.has(path),
          ),
          [],
          "read something other than the text_encoder parts and the manifest from the replacement",
        );
        assertEquals(
          [
            ...new Set(
              swapReads.filter((read) => read.phase === "construction").map((read) => read.path),
            ),
          ].filter((path) => path !== "karume.json" && path !== swapParts[0].path),
          [],
          "read beyond part 0 (descriptor) of the replacement during construction (by contract the weights are read in Session construction)",
        );
        // Byte coverage: every range of the bound blocks is covered by reads from the replacement
        // in the text stage (together with the precondition above, every weight part is read in the
        // text stage).
        const textStageReads = new Map<string, Read[]>();
        for (const read of swapReads.filter((candidate) => candidate.phase === "text_encoder")) {
          textStageReads.set(read.path, [...(textStageReads.get(read.path) ?? []), read]);
        }
        assertEquals(
          swapBlocks
            .filter(
              (block) =>
                !covers(
                  textStageReads.get(swapParts[block.part].path) ?? [],
                  block.offset,
                  block.length,
                ),
            )
            .map((block) => `${block.id}（part ${block.part}）`),
          [],
          "replacement blocks not read in full from the replacement in the text stage",
        );
        assertEquals(
          swapReads.filter((read) => read.phase === "after_text_encoder").map((read) => read.path),
          [],
          "read the replacement after the text stage",
        );
        // The Wan source never read the path of a text-encoder part (in either spelling — the
        // original cross-repository declaration or the replacement's declaration) — nothing but the
        // replacement supplied the text encoder's bytes. NOTE: a safety net (almost no independent
        // detection power) — a regression that opens the original preset is stopped earlier by ①'s
        // construction in cross-repository resolution, since there is no mapping. It turns red
        // independently only on a regression where hub goes looking for the cross-repository target
        // in the Wan distribution's directory.
        const textEncoderPaths = new Set([...wanParts, ...swapParts].map((part) => part.path));
        assertEquals(
          wanReads.filter((read) => textEncoderPaths.has(read.path)).map((read) => read.path),
          [],
          "read a text-encoder part from the Wan source",
        );

        // ③ Shape and finiteness of the generation.
        assertEquals(
          [video.frames, video.height, video.width],
          [DEFAULT_FRAMES, DEFAULT_HEIGHT, DEFAULT_WIDTH],
        );
        assertEquals(
          video.data.length,
          3 * video.frames * video.height * video.width,
          "element count",
        );
        const nonFinite = video.data.reduce(
          (count, value) => count + (Number.isFinite(value) ? 0 : 1),
          0,
        );
        assertEquals(nonFinite, 0, `${SWAP_CASE.id}: non-finite`);
        assertEquals(deviceLost, undefined, "device lost");

        // Observation: sha256 (not a check), match with the original umT5's row, times, amount read
        // from the replacement.
        const bytes = rgbBytes(video);
        const sha256 = await sha256Hex(bytes);
        const artifact = `${SWAP_CASE.id}.rgb`;
        await writeFile(results.artifact(artifact), bytes);
        const original = references.lookup(ORIGINAL_CASE_ID);
        const swapBytes = swapReads.reduce((total, read) => total + read.length, 0);
        const weightBytes = swapWeightParts.reduce((total, part) => total + part.size, 0);
        const gib = (value: number) => `${(value / 2 ** 30).toFixed(3)} GiB`;
        const seconds = (ms: number) => `${(ms / 1000).toFixed(1)} s`;
        const verdict = original === sha256 ? "matches" : "does not match";
        const stages = Object.entries(stageMs).map(([stage, ms]) => `${stage} ${seconds(ms)}`);
        const notes = [
          `replacement: ${swapRoot.pathname}`,
          `${swapWeightParts.length} weight parts, ${gib(weightBytes)}`,
          `read from the replacement ${gib(swapBytes)} (${swapReads.length} reads)`,
          original === undefined
            ? `the original umT5's row ${ORIGINAL_CASE_ID} is not in this environment`
            : `the original umT5's row ${ORIGINAL_CASE_ID}: ${verdict} (${original})`,
          `construction ${seconds(constructionMs)}, generation ${seconds(generateMs)}, stages ${stages.join(", ")}`,
          `non-finite ${nonFinite}`,
          `sha256 ${sha256} (an observation — not a check)`,
        ];
        console.log(`[wan-swap] ${SWAP_CASE.id}:\n  ${notes.join("\n  ")}`);
        return { status: "pass", actual: sha256, artifact, note: notes.join(" / ") };
      });
    } finally {
      // MUST: wait for the releases before dropping the device (the B570's destroy lag — leaves
      // budget for later tests).
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
);
