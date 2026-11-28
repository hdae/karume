// Gemma 4 26B A4B's QAT **MTP drafter** (ADR [0096](../../../docs/decisions/0096-speculative-decoding.md))
// over the 26B model cut after its first 12 layers (ADR 0145 decision 6), on this machine's GPU.
//
// This machine's GPU (12 GB) cannot hold the whole 26B model, and the drafter the distribution ships
// borrows the full model's last owner layers (`l28` / `l29`), which the cut does not have. So these
// checks run a test-only drafter series exported against the cut (`gemma4-26b-cut12-qat-drafter`:
// the same pinned assistant, its external slots the cut's last sliding / full owner layers `l10` /
// `l11`, its goldens taken over the cut's reference). The pair goes through the gemma pipeline from a
// manifest this file writes in memory (26B's entry as the q4_0 QAT line writes it, with the cut's
// containers in place of the model's and the drafter's), read through a local source.
//
// **The cut's drafts are never accepted.** The cut's hidden state is not the final layer's, which is
// off the drafter's training distribution: the series' goldens accept nothing (0 per step, 1.000
// token per cycle on all three cases — ① holds the record to that). So speculation's accept path is
// not exercised here: every cycle here drafts, verifies k + 1 rows and rejects at the first draft.
// The accept path on the real model is left to the browser run on a 16 GB GPU. The cases:
//
// ① assets and provenance (no GPU): the cut drafter's record names the assistant the shipped
//    drafter's record names (repository and revision), the target checkpoint the cut series was
//    exported from (sha256 of each file), the cut's layer count (12), and goldens that accept no
//    draft
// ② the pairing (no GPU): the admission the pipeline runs (`admitGemma4`) pairs the cut with the cut
//    drafter — the four external slots are the cut's `l10` [1,8,1032,256] / `l11` [1,2,C,512] with
//    the cut's shapes, the one shared initializer is the cut's `model.lm_head.weight` (no bytes of
//    its own), the `hidden` input is 2816 wide — and `Pipeline.fromPretrained` with `speculative`
//    refuses the **shipped** drafter on the cut with the admission's message (its slots `l28` /
//    `l29` are not in the cut), before any weight part is read
// ③ the drafter's drafts (real GPU), teacher-forced along each golden's continuation (3 cases × 200
//    cycles × 3 steps) on the seat `i4` (session {}): a sha256 row per case
//    (`gemma4-26b-cut12-qat-drafter/<case>/drafts` in `fixtures/references/gemma4-26b-cut.json` —
//    ADR 0106), and drafts that are not stuck
// ④ draft parity (real GPU, upstream parity — opt-in KARUME_UPSTREAM_PARITY=1): ③'s drafts match the
//    torch drafter's goldens in at least 99% of the positions (`e2e_gemma4_drafter_test.ts`'s floor —
//    MUST NOT be moved to the measurement)
// ⑤ speculation through the pipeline (real GPU), per seat (`i4`, `i4-fast`): greedy decode of 200
//    tokens from each golden prompt with `speculative: "always"` gives the same token ids (u32
//    equality) and positions as the same decode without it, the speculative path ran (cycles and
//    drafts), and its ids are pinned as a sha256 row per seat and case
//    (`gemma4-26b-cut12+drafter/<seat>/<case>`). All three golden prompts fit the cut's budget: the
//    longest (readme-exporter, 4163 tokens) runs at a capacity of 4363 rows
//
// The drafter's outputs are its three i32 draft tokens: it exits no logits and no hidden state, so
// the drafts are what ③ pins.
//
// ## Assets
//
// - the cut series `outputs/series/gemma4-26b-cut12/` (`e2e_gemma4_26b_cut_test.ts`) and the 26B
//   tokenizer series `outputs/series/gemma4-26b-tokenizer/`;
// - the cut drafter series `outputs/series/gemma4-26b-cut12-qat-drafter/`
//   (`gemma4.export_drafter_qat --model 26b --layers 12`, the cut series as its lender);
// - the shipped drafter series `outputs/series/gemma4-26b-qat-drafter/` (its container's graph —
//   ②'s refusal — and its record).
//
// A cut is not shipped, so every case is an implementation-time series check (ADR 0139 decision 6):
// without KARUME_SERIES=1 every case SKIPs explicitly and nothing under `outputs/` is read. With it,
// an absent asset SKIPs with the command that makes it.
//
// ## Memory
//
// One cut on the GPU at a time: about 6 GiB of weights, the drafter's 0.25 GiB, the sliding slots and
// two full layers' slots at a capacity of at most 4363 rows — within a 12 GB GPU. ③'s Sessions are
// disposed before ⑤ builds its pipelines, and each seat's pipeline is disposed before the next.

import { isNotFound, readFile, readTextFileSync, statSync } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { localDirectory, MANIFEST_FILENAME, parseManifest } from "@karume/hub";
import {
  acquireGpu,
  type GenerationContext,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
  type SafetensorsFile,
  type Session,
  type Tensor,
} from "@karume/runtime";
import { type GenerationStop, Pipeline, parsePipelineConfig } from "../../models/gemma.ts";
import { admitGemma4 } from "../../models/src/gemma/admission.ts";
import { gemma4RopeInputNames, ropeInputs } from "../../models/src/gemma/rope.ts";
import {
  draftOnce,
  GEMMA4_DRAFT_STEPS,
  type Gemma4Drafter,
  openDrafterContext,
} from "../../models/src/gemma/speculative.ts";
import { planPrefillChunks } from "../../models/src/generation/greedy.ts";
import { seatArtifactBytes } from "../../models/tests/helpers/gemma-reference.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";
import { seriesGraph } from "../../models/tests/helpers/series-graphs.ts";
import {
  modelPresent,
  openSeriesContainer,
  resolveParts,
} from "../../runtime/tests/helpers/container-files.ts";
import { assertRunningAdapter } from "../../runtime/tests/helpers/environment.ts";
import {
  openReferences,
  referenceEntryFields,
  referenceMismatchMessage,
  type ReferenceSettlement,
  registerReferenceGate,
  settleReference,
  sha256Hex,
} from "../../runtime/tests/helpers/reference.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

const SERIES_ROOT = new URL("../../../outputs/series/", import.meta.url);
/** The cut target (the lender), the cut drafter and the drafter the distribution ships. */
const CUT_SERIES = "gemma4-26b-cut12";
const CUT_DRAFTER_SERIES = "gemma4-26b-cut12-qat-drafter";
const SHIPPED_DRAFTER_SERIES = "gemma4-26b-qat-drafter";
const CUT = new URL(`${CUT_SERIES}/`, SERIES_ROOT);
const CUT_DRAFTER = new URL(`${CUT_DRAFTER_SERIES}/`, SERIES_ROOT);
const SHIPPED_DRAFTER = new URL(`${SHIPPED_DRAFTER_SERIES}/`, SERIES_ROOT);
const TOKENIZER = new URL("gemma4-26b-tokenizer/tokenizer.json", SERIES_ROOT);
/** The representative of each series container (its parts are `model-N-of-M.krm`). */
const CONTAINER_FILE = "model.krm";
const REFERENCE_FILE = "reference.json";

const EXPORT_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' python -m" +
  " gemma4.export_drafter_qat --model 26b (the shipped drafter) and the same with --layers 12 (the" +
  " cut drafter), and python -m gemma4.export_moe --layers --routing-outputs for the cut";

/** The roles of the manifest (`gemma4/distribution.py`) — the graph names of the two containers. */
const MODEL = "model";
const DRAFTER = "drafter";
/** The model name of the in-memory manifest. */
const MODEL_NAME = "26b-cut12";
/** The one storage of the q4_0 QAT line, and 26B's two seats over it. */
const STORAGE = "i4";
const SEATS = ["i4", "i4-fast"] as const;
type Seat = (typeof SEATS)[number];
const SEAT_SESSIONS: Record<Seat, Readonly<Record<string, unknown>>> = {
  i4: {},
  "i4-fast": { numerics: { level: "reorder" } },
};
/** The seat ③ and ④ run on (session {} — the reference summation order). */
const REFERENCE_SEAT: Seat = "i4";
/** 26B's device limits as the line declares them (`requiredLimits` of both seats). */
const REQUIRED_LIMITS = { maxBufferSize: 1073741824, maxStorageBufferBindingSize: 1073741824 };

/** google/gemma-4-26B-A4B-it config.json at 4d7ae498 (`text_config`). */
const HIDDEN = 2816;
const CAPACITY_SYMBOL = "C";
/** The cut's last sliding / full owner layers, which the cut drafter borrows. */
const CUT_SLOTS: Record<string, readonly (number | string)[]> = {
  "l10.k": [1, 8, 1032, 256],
  "l10.v": [1, 8, 1032, 256],
  "l11.k": [1, 2, CAPACITY_SYMBOL, 512],
  "l11.v": [1, 2, CAPACITY_SYMBOL, 512],
};
/** The full model's last owner layers, which the shipped drafter borrows. */
const SHIPPED_SLOT_LAYERS = ["l28", "l29"] as const;
/** The table the drafter borrows (its name is the lender's initializer name — docs/ir-v2.md). */
const SHARED_WEIGHT = "model.lm_head.weight";
/** The cut's layer count (the cut series' `reference.cut.layers`). */
const CUT_LAYERS = 12;
/** The checkpoint files the cut's record and the cut drafter's record must agree on. */
const LENDER_CHECKPOINT_FILES = [
  "model.safetensors.index.json",
  "model-00001-of-00002.safetensors",
  "model-00002-of-00002.safetensors",
  "config.json",
] as const;

/** 26B's declaration as the q4_0 QAT line writes it (`e2e_gemma4_26b_cut_test.ts`). */
const PIPELINE_CONFIG_JSON = {
  chunkLength: 768,
  maxChunkLength: 768,
  maxPosition: 262144,
  capacity: 4096,
  rope: {
    sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
    full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
  },
  sampler: { temperature: 1, topK: 64, topP: 0.95 },
  chatDialect: "gemma4-unified",
} as const;
const PIPELINE_CONFIG = parsePipelineConfig(PIPELINE_CONFIG_JSON);

/** Graph inputs of the product (`gemma4/export_product.py`). */
const INPUT_IDS = "input_ids";
const SOFT_EMBEDS = "soft_embeds";
const SOFT_MASK = "soft_mask";
const LAST_ROW = "last_row";

/** The drafter golden cases (`gemma4/export_drafter.py`). */
const CASES = ["short-en", "readme-recipes", "readme-exporter"] as const;
type CaseName = (typeof CASES)[number];
const GOLDEN_PREFIX = "drafter-golden.";
const SUFFIX = ".safetensors";
/** ④'s draft-match floor (`e2e_gemma4_drafter_test.ts` — the denominator is cases × cycles × k). */
const MATCH_FLOOR = 0.99;
/** ⑤'s generated tokens per case (the goldens' cycle count). */
const MAX_NEW_TOKENS = 200;

const isFile = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const containerOf = (root: URL): URL => new URL(CONTAINER_FILE, root);
const goldenUrl = (name: CaseName): URL => new URL(`${GOLDEN_PREFIX}${name}${SUFFIX}`, CUT_DRAFTER);

/** Every asset is present (read only with KARUME_SERIES=1 — the default lanes touch nothing). */
const PRESENT =
  SERIES_CHECKS &&
  modelPresent(containerOf(CUT)) &&
  modelPresent(containerOf(CUT_DRAFTER)) &&
  modelPresent(containerOf(SHIPPED_DRAFTER)) &&
  isFile(TOKENIZER) &&
  [CUT, CUT_DRAFTER, SHIPPED_DRAFTER].every((root) => isFile(new URL(REFERENCE_FILE, root))) &&
  CASES.every((name) => isFile(goldenUrl(name)));
if (SERIES_CHECKS && !PRESENT) {
  console.warn(
    `[karume] the 26B cut, its drafter series or the shipped drafter series under ${SERIES_ROOT.pathname}` +
      ` is absent or incomplete, so the Gemma 4 26B cut drafter cases SKIP. Export them: ${EXPORT_COMMAND}`,
  );
}

type ObjectRecord = Record<string, unknown>;
const objectAt = (value: unknown, where: string): ObjectRecord => {
  assert(
    typeof value === "object" && value !== null && !Array.isArray(value),
    `${where} is not an object`,
  );
  return value as ObjectRecord;
};
const readRecord = (root: URL): ObjectRecord =>
  objectAt(JSON.parse(readTextFileSync(new URL(REFERENCE_FILE, root))), root.pathname);

const bufferOf = (bytes: Uint8Array<ArrayBuffer>): ArrayBuffer =>
  bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? bytes.buffer
    : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

// ---------------------------------------------------------------------------
// The in-memory distribution: 26B's entry with the cut's containers
// ---------------------------------------------------------------------------

/** The fixed header of a container (docs/container-v1.md §1): magic, version, two u64 lengths. */
const CONTAINER_HEADER_BYTES = 24;

/**
 * One container's manifest entry (`descriptor` + `parts`) from its series files, and the map from the
 * manifest paths to those files. The descriptor expectations are the two documents of part 0, which
 * follow the header (docs/container-v1.md §1 / §2).
 */
const containerEntry = async (
  root: URL,
  directory: string,
): Promise<{
  readonly entry: ObjectRecord;
  readonly files: ReadonlyMap<string, URL>;
  /** The manifest paths of the weight parts (every part after part 0, the descriptor file). */
  readonly weightParts: readonly string[];
}> => {
  const parts = resolveParts(containerOf(root));
  const head = await readFile(parts[0]);
  const view = new DataView(head.buffer, head.byteOffset, head.byteLength);
  const graphLength = Number(view.getBigUint64(8, true));
  const modelLength = Number(view.getBigUint64(16, true));
  const graphStart = CONTAINER_HEADER_BYTES;
  const modelStart = graphStart + graphLength;
  const files = new Map<string, URL>();
  const refs: ObjectRecord[] = [];
  for (const part of parts) {
    const name = decodeURIComponent(part.pathname.split("/").at(-1) ?? "");
    const path = `${directory}/${name}`;
    const bytes = await readFile(part);
    refs.push({ path, size: bytes.byteLength, sha256: await sha256Hex(bytes) });
    files.set(path, part);
  }
  return {
    entry: {
      container: {
        descriptor: {
          graph: {
            length: graphLength,
            sha256: await sha256Hex(head.slice(graphStart, modelStart)),
          },
          model: {
            length: modelLength,
            sha256: await sha256Hex(head.slice(modelStart, modelStart + modelLength)),
          },
        },
        parts: refs,
      },
    },
    files,
    weightParts: [...files.keys()].slice(1),
  };
};

/** A source over an in-memory manifest, and the manifest paths it was asked for. */
type CutSource = {
  readonly source: ReturnType<typeof localDirectory>;
  readonly requested: readonly string[];
  /** The manifest paths of the weight parts (every part after part 0) of both containers. */
  readonly weightParts: ReadonlySet<string>;
};

/**
 * A local source holding one model, `26b-cut12`: 26B's entry as the q4_0 QAT line writes it — the
 * roles `model` and `drafter` at `i4`, the seats `i4` and `i4-fast` naming both, its declaration and
 * limits — with the cut as `model` and `drafter` from `drafterRoot`. The manifest is never written to
 * disk; every other path is a series file.
 */
const cutSource = async (drafterRoot: URL): Promise<CutSource> => {
  const model = await containerEntry(CUT, `${MODEL_NAME}/model`);
  const drafter = await containerEntry(drafterRoot, `${MODEL_NAME}/drafter`);
  const tokenizer = await readFile(TOKENIZER);
  const tokenizerPath = `${MODEL_NAME}/tokenizer/tokenizer.json`;
  const quant = (seat: Seat) => ({
    weights: { [MODEL]: STORAGE, [DRAFTER]: STORAGE },
    session: SEAT_SESSIONS[seat],
    requiredLimits: REQUIRED_LIMITS,
  });
  const manifest = {
    format: "karume/6",
    generator: "e2e_gemma4_26b_cut_drafter_test",
    defaultModel: MODEL_NAME,
    models: {
      [MODEL_NAME]: {
        pipeline: "gemma4/1",
        weights: {
          [MODEL]: { [STORAGE]: model.entry },
          [DRAFTER]: { [STORAGE]: drafter.entry },
        },
        assets: {
          tokenizer: {
            path: tokenizerPath,
            size: tokenizer.byteLength,
            sha256: await sha256Hex(tokenizer),
          },
        },
        quants: Object.fromEntries(SEATS.map((seat) => [seat, quant(seat)])),
        defaultQuant: REFERENCE_SEAT,
        pipelineConfig: PIPELINE_CONFIG_JSON,
      },
    },
  };
  const manifestBytes = new TextEncoder().encode(JSON.stringify(manifest));
  // The manifest this file writes is one the hub reads (a parse failure names the field here).
  parseManifest(new TextDecoder().decode(manifestBytes));
  const files = new Map([...model.files, ...drafter.files, [tokenizerPath, TOKENIZER]]);
  const weightParts = new Set([...model.weightParts, ...drafter.weightParts]);
  const requested: string[] = [];
  const source = localDirectory(
    {
      readFile: async (path) => {
        requested.push(path);
        if (path === MANIFEST_FILENAME) return manifestBytes.slice();
        const url = files.get(path);
        if (url === undefined) throw new Error(`test: the cut source has no file '${path}'`);
        return await readFile(url);
      },
    },
    { label: `${CUT_SERIES} + ${drafterRoot.pathname}` },
  );
  return { source, requested, weightParts };
};

const openGraph = async (root: URL, series: string): Promise<PreparedModel> =>
  prepareContainer(await openSeriesContainer(containerOf(root)), seriesGraph(series));

// ---------------------------------------------------------------------------
// ① ② provenance and the pairing (no GPU)
// ---------------------------------------------------------------------------

test.skipIf(!PRESENT)(
  `gemma4 26b cut drafter: the record names the shipped drafter's assistant, the cut's checkpoint and layers, and goldens that accept nothing (no GPU)${SERIES_CHECKS_OPT_IN}`,
  () => {
    const record = readRecord(CUT_DRAFTER);
    assertEquals(record.series, CUT_DRAFTER_SERIES, "the record's series");
    const assistant = objectAt(record.drafter, "drafter");
    const shipped = objectAt(readRecord(SHIPPED_DRAFTER).drafter, "shipped drafter");
    assertEquals(
      [assistant.repo, assistant.revision],
      [shipped.repo, shipped.revision],
      "the assistant (the shipped drafter's pin)",
    );
    const cut = readRecord(CUT);
    assertEquals(
      objectAt(record.cut, "cut").layers,
      CUT_LAYERS,
      "the cut drafter's target layer count",
    );
    assertEquals(
      objectAt(objectAt(cut.reference, "reference").cut, "reference.cut").layers,
      CUT_LAYERS,
      "the cut series' layer count",
    );
    const drafterFiles = objectAt(objectAt(record.checkpoint, "checkpoint").files, "files");
    const cutFiles = objectAt(objectAt(cut.checkpoint, "checkpoint").files, "files");
    for (const name of LENDER_CHECKPOINT_FILES) {
      assertEquals(
        objectAt(drafterFiles[name], name).sha256,
        objectAt(cutFiles[name], name).sha256,
        `${name}: the goldens' target is the cut's checkpoint`,
      );
    }
    // The module comment's premise: the cut's hidden state is off the drafter's training
    // distribution, and torch accepts no draft on any case. A record that accepts drafts is another
    // export, and the comment no longer describes it.
    const goldens = objectAt(record.goldens, "goldens");
    for (const name of CASES) {
      const golden = objectAt(goldens[name], `goldens.${name}`);
      assertEquals(golden.per_step, [0, 0, 0], `${name}: per-step acceptance`);
      assertEquals(golden.tokens_per_cycle, 1, `${name}: tokens per cycle`);
    }
  },
);

test.skipIf(!PRESENT)(
  `gemma4 26b cut drafter: the pipeline pairs the cut with its drafter — l10 / l11, the shared table, hidden 2816 — and refuses the shipped drafter (no GPU)${SERIES_CHECKS_OPT_IN}`,
  async () => {
    const target = await openGraph(CUT, CUT_SERIES);
    const drafter = await openGraph(CUT_DRAFTER, CUT_DRAFTER_SERIES);
    assertEquals(
      drafter.graph.inputs.map((input) => input.name),
      ["token", "hidden", ...gemma4RopeInputNames()],
      "drafter inputs (order included)",
    );
    assertEquals(Object.keys(drafter.graph.states).sort(), Object.keys(CUT_SLOTS).sort(), "slots");
    for (const [name, shape] of Object.entries(CUT_SLOTS)) {
      assert(drafter.graph.states[name].external === true, `${name}: not external`);
      assertEquals(drafter.graph.states[name].shape, shape, `${name}: the drafter's shape`);
      assertEquals(target.graph.states[name].shape, shape, `${name}: the cut's shape`);
    }
    // The admission `Pipeline.fromPretrained` runs before any weight part is fetched.
    const admitted = admitGemma4(
      { graph: target.graph },
      PIPELINE_CONFIG,
      { graph: drafter.graph },
      "gemma.Pipeline",
    );
    assert(admitted.drafter !== undefined, "no drafter admission");
    assertEquals(admitted.drafter.admission.hiddenSize, HIDDEN, "hidden width");
    assertEquals(
      admitted.drafter.admission.sharedWeights,
      { [SHARED_WEIGHT]: SHARED_WEIGHT },
      "shared weights",
    );
    assertEquals(drafter.graph.initializers[SHARED_WEIGHT].storage, undefined, "no bytes borrowed");

    // The shipped drafter reads the full model's last owner layers, which the cut does not have:
    // the pipeline refuses it with the admission's message, and reads no weight part first.
    const shipped = await openGraph(SHIPPED_DRAFTER, SHIPPED_DRAFTER_SERIES);
    assertEquals(
      [...new Set(Object.keys(shipped.graph.states).map((name) => name.split(".")[0]))].sort(),
      [...SHIPPED_SLOT_LAYERS],
      "the shipped drafter's slot layers",
    );
    const { source, requested, weightParts } = await cutSource(SHIPPED_DRAFTER);
    let refusal: unknown;
    try {
      const pipeline = await Pipeline.fromPretrained(source, { speculative: {} });
      await pipeline.dispose();
    } catch (cause) {
      refusal = cause;
    }
    assert(refusal instanceof Error, "the pipeline admitted the shipped drafter on the cut");
    assert(
      /external slot 'l2[89]\.[kv]' that the drafter reads is not in the target graph/.test(
        refusal.message,
      ),
      `the refusal is not the admission's: ${refusal.message}`,
    );
    console.log(`[e2e] gemma4 26b cut drafter: the shipped drafter refused — ${refusal.message}`);
    assertEquals(
      requested.filter((path) => weightParts.has(path)),
      [],
      "weight parts read before the refusal",
    );
  },
);

// ---------------------------------------------------------------------------
// ③ ④ the drafter's drafts, teacher-forced (real GPU)
// ---------------------------------------------------------------------------

/** sha256 rows (rows per environment key — `runtime/tests/helpers/reference.ts`). */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/gemma4-26b-cut.json", import.meta.url),
);
/** Artifacts and settlements (`outputs/verify/<environment key>/<date>_gemma4-26b-cut-drafter/`). */
const results = openResults("gemma4-26b-cut-drafter");
const RUNNABLE = PRESENT && GPU_AVAILABLE;
const draftRowId = (name: CaseName): string => `${CUT_DRAFTER_SERIES}/${name}/drafts`;
const idsRowId = (seat: Seat, name: CaseName): string => `${CUT_SERIES}+drafter/${seat}/${name}`;

/** One drafter golden (`prompt [T]` / `tokens [N + k]` / `draft [N, k]`). */
type DrafterGolden = {
  readonly prompt: Int32Array<ArrayBuffer>;
  readonly tokens: Int32Array<ArrayBuffer>;
  readonly draft: Int32Array<ArrayBuffer>;
  readonly cycles: number;
};

const readGolden = async (name: CaseName): Promise<DrafterGolden> => {
  const file: SafetensorsFile = parseSafetensors(bufferOf(await readFile(goldenUrl(name))));
  const i32 = (key: string): Int32Array<ArrayBuffer> => {
    const view = file.tensors.get(key);
    assert(view !== undefined && view.dtype === "I32", `${name}: golden '${key}' is not I32`);
    return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
  };
  const draft = file.tensors.get("draft");
  assert(draft !== undefined, `${name}: no 'draft'`);
  assertEquals(draft.shape[1], GEMMA4_DRAFT_STEPS, `${name}: draft steps`);
  const tokens = i32("tokens");
  assertEquals(draft.shape[0] + GEMMA4_DRAFT_STEPS, tokens.length, `${name}: cycles`);
  return { prompt: i32("prompt"), tokens, draft: i32("draft"), cycles: draft.shape[0] };
};

const f32Of = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  assert(tensor.data instanceof Float32Array, `${where}: not f32`);
  return tensor.data;
};

/** The bytes of ③'s artifact: the drafts in cycle order, `k` per cycle. */
const draftsArtifactBytes = (drafts: readonly number[]): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(
    `${JSON.stringify({ steps: GEMMA4_DRAFT_STEPS, drafts }, undefined, 2)}\n`,
  );

describe.skipIf(!RUNNABLE)(
  `gemma4 26b cut drafter: the drafts teacher-forced along the goldens (real GPU)${SERIES_CHECKS_OPT_IN}`,
  () => {
    let target: Session | undefined;
    let drafterSession: Session | undefined;
    let drafter: Gemma4Drafter | undefined;
    let targetOutputs: readonly string[] = [];
    let destroy: (() => void) | undefined;
    const drafts = new Map<CaseName, Int32Array<ArrayBuffer>>();

    beforeAll(async () => {
      const targetModel = await openGraph(CUT, CUT_SERIES);
      const drafterModel = await openGraph(CUT_DRAFTER, CUT_DRAFTER_SERIES);
      const admitted = admitGemma4(
        { graph: targetModel.graph },
        PIPELINE_CONFIG,
        { graph: drafterModel.graph },
        "test",
      );
      assert(admitted.drafter !== undefined, "no drafter admission");
      const { sharedWeights, outputs, hiddenSize } = admitted.drafter.admission;
      targetOutputs = targetModel.graph.outputs;
      const gpu = await acquireGpu();
      destroy = () => gpu.destroy();
      // The line's reference seat (session {}).
      const lender = await targetModel.createContainerSession(gpu, {});
      target = lender;
      drafterSession = await drafterModel.createContainerSession(gpu, {
        sharedWeights: Object.fromEntries(
          Object.keys(sharedWeights).map((borrower) => [
            borrower,
            lender.exportWeight(sharedWeights[borrower]),
          ]),
        ),
      });
      drafter = {
        session: drafterSession,
        graph: drafterModel.graph,
        outputs,
        rope: PIPELINE_CONFIG.rope,
        hiddenSize,
      };
    });

    afterAll(async () => {
      // MUST: the borrower first (the lender refuses while its table is lent out).
      await drafterSession?.dispose();
      await target?.dispose();
      destroy?.();
    });

    /** One target run; returns the selected row's hidden. */
    const runTarget = async (
      context: GenerationContext,
      ids: Int32Array<ArrayBuffer>,
      positions: Int32Array<ArrayBuffer>,
      queryLength: number,
    ): Promise<Float32Array<ArrayBuffer>> => {
      assert(target !== undefined, "the target is not open");
      // The graph takes the soft inputs on every run: these rows hold no image.
      const outputs = await target.run(
        {
          [INPUT_IDS]: { dtype: "i32", shape: [1, ids.length], data: ids },
          [SOFT_EMBEDS]: {
            dtype: "f32",
            shape: [1, ids.length, HIDDEN],
            data: new Float32Array(ids.length * HIDDEN),
          },
          [SOFT_MASK]: {
            dtype: "bool",
            shape: [1, ids.length, 1],
            data: new Uint32Array(ids.length),
          },
          ...ropeInputs(PIPELINE_CONFIG.rope, positions),
          [LAST_ROW]: { dtype: "i32", shape: [1], data: Int32Array.of(queryLength - 1) },
        },
        undefined,
        { context, queryLength },
      );
      assertEquals(outputs[targetOutputs[1]].shape, [1, 1, HIDDEN], "hidden shape");
      return f32Of(outputs[targetOutputs[1]], "hidden");
    };

    for (const name of CASES) {
      it(`③ ${name}: 200 cycles × 3 steps, the drafts against this machine's row`, async () => {
        assert(target !== undefined && drafter !== undefined, "the pair is not open");
        const golden = await readGolden(name);
        const chunkLength = PIPELINE_CONFIG.chunkLength;
        const capacity = Math.max(chunkLength, golden.prompt.length + golden.cycles);
        const id = draftRowId(name);
        const settles = !references.lacksReference(id);
        let settlement: ReferenceSettlement | undefined;
        await runRecordedCase(
          results,
          {
            id,
            failureNote: (cause) => (cause instanceof Error ? cause.message : String(cause)),
          },
          async () => {
            assert(target !== undefined && drafter !== undefined, "the pair is not open");
            await assertRunningAdapter();
            const started = performance.now();
            const context = await target.createGenerationContext({
              bindings: { [CAPACITY_SYMBOL]: capacity },
              chunkLength,
            });
            const borrowed = await openDrafterContext(drafter, context);
            const out = new Int32Array(golden.cycles * GEMMA4_DRAFT_STEPS);
            try {
              let hidden!: Float32Array<ArrayBuffer>;
              for (const chunk of planPrefillChunks(golden.prompt.length, chunkLength)) {
                const ids = new Int32Array(new ArrayBuffer(chunkLength * 4));
                const positions = new Int32Array(new ArrayBuffer(chunkLength * 4));
                for (let row = 0; row < chunk.queryLength; row += 1) {
                  ids[row] = golden.prompt[chunk.position + row];
                  positions[row] = chunk.position + row;
                }
                hidden = await runTarget(context, ids, positions, chunk.queryLength);
              }
              for (let cycle = 0; cycle < golden.cycles; cycle += 1) {
                const position = golden.prompt.length + cycle;
                if (cycle > 0) {
                  hidden = await runTarget(
                    context,
                    Int32Array.of(golden.tokens[cycle - 1]),
                    Int32Array.of(position - 1),
                    1,
                  );
                }
                const draft = await draftOnce(drafter, borrowed, {
                  token: golden.tokens[cycle],
                  hidden,
                  position,
                });
                out.set(draft, cycle * GEMMA4_DRAFT_STEPS);
              }
            } finally {
              // MUST: borrower → lender (the runtime refuses the reverse order).
              await borrowed.dispose();
              await context.dispose();
            }
            drafts.set(name, out);
            // A stuck drafter exit emits a handful of ids: 32 distinct, or the torch drafter's own
            // count where its golden repeats itself.
            const distinct = new Set(out).size;
            const floor = Math.min(32, new Set(golden.draft).size);
            assert(distinct >= floor, `${name}: only ${distinct} distinct drafts (floor ${floor})`);
            const note =
              `T=${golden.prompt.length} / ${golden.cycles} cycles / ${distinct} distinct drafts` +
              ` / ${(performance.now() - started).toFixed(0)} ms`;
            console.log(`[e2e] gemma4 26b cut drafter ③ ${name}: ${note}`);
            const artifact = `${CUT_DRAFTER_SERIES}-${name}-drafts.json`;
            if (!settles) return { status: "pass", note };
            settlement = await settleReference(references, results, {
              id,
              artifact,
              bytes: draftsArtifactBytes([...out]),
            });
            return { ...referenceEntryFields(settlement, artifact), note };
          },
        );
        if (settlement?.check.status === "fail") {
          throw new Error(referenceMismatchMessage(id, settlement, references));
        }
      });
    }

    it.skipIf(!UPSTREAM_PARITY)(
      `④ the drafts match the torch drafter's goldens in ≥ 99%${UPSTREAM_PARITY_OPT_IN}`,
      async () => {
        let matched = 0;
        let total = 0;
        const lines: string[] = [];
        for (const name of CASES) {
          const ours = drafts.get(name);
          assert(ours !== undefined, `${name}: ③ did not run`);
          const golden = await readGolden(name);
          const perStep = Array.from({ length: GEMMA4_DRAFT_STEPS }, () => 0);
          for (let at = 0; at < ours.length; at += 1) {
            if (ours[at] === golden.draft[at]) perStep[at % GEMMA4_DRAFT_STEPS] += 1;
          }
          const hits = perStep.reduce((sum, value) => sum + value, 0);
          matched += hits;
          total += ours.length;
          lines.push(
            `${name}: ${hits}/${ours.length} (per step ${perStep
              .map((hit) => `${((hit / golden.cycles) * 100).toFixed(1)}%`)
              .join(" / ")})`,
          );
        }
        const rate = matched / total;
        console.log(
          `[e2e] gemma4 26b cut drafter ④ draft match:\n  ${lines.join("\n  ")}\n  total` +
            ` ${matched}/${total} = ${(rate * 100).toFixed(2)}% (floor ${(MATCH_FLOOR * 100).toFixed(0)}%)`,
        );
        assert(
          rate >= MATCH_FLOOR,
          `the draft match ${(rate * 100).toFixed(2)}% is under the ${MATCH_FLOOR * 100}% floor` +
            ` (${lines.join(" / ")})`,
        );
      },
    );
  },
);

// ---------------------------------------------------------------------------
// ⑤ speculation through the pipeline (real GPU)
// ---------------------------------------------------------------------------

/** One greedy decode of `MAX_NEW_TOKENS` from `prompt` (ids and absolute positions). */
const decode = async (
  pipeline: Pipeline,
  prompt: readonly number[],
  speculative: "always" | false,
): Promise<{ ids: number[]; positions: number[]; stop: GenerationStop; ms: number }> => {
  const capacity = Math.max(pipeline.program.chunkLength, prompt.length + MAX_NEW_TOKENS);
  const sequence = await pipeline.sequence({ speculative, capacity });
  try {
    const started = performance.now();
    const stream = sequence.generate({
      prompt: [...prompt],
      maxNewTokens: MAX_NEW_TOKENS,
      sampler: { temperature: 0 },
    });
    const ids: number[] = [];
    const positions: number[] = [];
    for await (const event of stream) {
      if (event.kind !== "token") continue;
      ids.push(event.id);
      positions.push(event.position);
    }
    return { ids, positions, stop: await stream.done, ms: performance.now() - started };
  } finally {
    await sequence.dispose();
  }
};

for (const seat of SEATS) {
  describe.skipIf(!RUNNABLE)(
    `gemma4 26b cut drafter ${seat}: speculation through the pipeline gives plain decode's tokens (real GPU)${SERIES_CHECKS_OPT_IN}`,
    () => {
      let pipeline: Pipeline | undefined;
      beforeAll(async () => {
        const { source } = await cutSource(CUT_DRAFTER);
        const started = performance.now();
        pipeline = await Pipeline.fromPretrained(source, {
          quant: seat,
          speculative: { k: GEMMA4_DRAFT_STEPS },
        });
        console.log(
          `[e2e] gemma4 26b cut drafter ${seat}: load with the drafter` +
            ` ${(performance.now() - started).toFixed(0)} ms`,
        );
      });
      afterAll(async () => {
        await pipeline?.dispose();
      });

      for (const name of CASES) {
        it(`⑤ ${seat} ${name}: ${MAX_NEW_TOKENS} greedy tokens speculative = plain, and the ids match this machine's row`, async () => {
          assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
          const opened = pipeline;
          const golden = await readGolden(name);
          const prompt = [...golden.prompt];
          const id = idsRowId(seat, name);
          const settles = !references.lacksReference(id);
          let settlement: ReferenceSettlement | undefined;
          await runRecordedCase(
            results,
            {
              id,
              failureNote: (cause) => (cause instanceof Error ? cause.message : String(cause)),
            },
            async () => {
              await assertRunningAdapter();
              const speculative = await decode(opened, prompt, "always");
              const plain = await decode(opened, prompt, false);
              // Speculation's one contract: it never changes the output (u32 equality).
              assertEquals(
                Uint32Array.from(speculative.ids),
                Uint32Array.from(plain.ids),
                `${name}: token ids`,
              );
              assertEquals(speculative.positions, plain.positions, `${name}: positions`);
              assertEquals(speculative.stop.reason, plain.stop.reason, `${name}: stop reason`);
              assertEquals(speculative.stop.tokens, plain.stop.tokens, `${name}: token count`);
              assertEquals(
                plain.stop.speculation,
                undefined,
                "a plain decode carries no speculation",
              );
              const speculation = speculative.stop.speculation;
              assert(
                speculation !== undefined,
                `${name}: the speculative decode did not speculate`,
              );
              assert(
                speculation.cycles > 0 && speculation.drafted > 0,
                `${name}: no cycle drafted (${speculation.cycles} cycles, ${speculation.drafted} drafted)`,
              );
              const note =
                `T=${prompt.length} / ${speculative.ids.length} tokens (stop ${speculative.stop.reason})` +
                ` / ${speculation.cycles} cycles / accepted ${speculation.accepted}/${speculation.drafted}` +
                ` / ${(speculation.delivered / speculation.cycles).toFixed(3)} tokens per cycle` +
                ` / speculative ${speculative.ms.toFixed(0)} ms vs plain ${plain.ms.toFixed(0)} ms`;
              console.log(`[e2e] gemma4 26b cut drafter ⑤ ${seat} ${name}: ${note}`);
              const artifact = `${CUT_SERIES}+drafter-${seat}-${name}.json`;
              if (!settles) return { status: "pass", note };
              settlement = await settleReference(references, results, {
                id,
                artifact,
                bytes: seatArtifactBytes(speculative.ids),
              });
              return { ...referenceEntryFields(settlement, artifact), note };
            },
          );
          if (settlement?.check.status === "fail") {
            throw new Error(referenceMismatchMessage(id, settlement, references));
          }
        });
      }
    },
  );
}

const ROW_IDS: readonly string[] = [
  ...CASES.map(draftRowId),
  ...SEATS.flatMap((seat) => CASES.map((name) => idsRowId(seat, name))),
];
// The rows are implementation-time checks: registered only on a run that can run them
// (`referenceGatePasses` counts the registered cases alone).
if (RUNNABLE) {
  references.warnMissing(ROW_IDS);
  registerReferenceGate(references, { runnable: RUNNABLE, caseIds: ROW_IDS });
}
