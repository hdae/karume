// Real-weight Gemma 4 26B A4B (the mixture of experts — ADR 0145) cut after its first layers, on this
// machine's GPU.
//
// This machine's GPU (12 GB) cannot hold the 26B model (about 13.4 GiB of text weights), so its
// checks run the model cut after N layers (ADR 0145 decision 6): the same product graph as the
// distribution's — 12B's contract (input_ids / soft_embeds / soft_mask / four RoPE inputs /
// last_row → logits [1,R,V], hidden [1,R,H]) — over the first N layers, every layer a dense GeGLU
// MLP plus 128 experts of which a `topk` picks 8 per token through `expert_linear`. The cases:
//
// ① assets and provenance (no GPU): the cut container, the routing container, the tokenizer and
//    one golden per case of the record; the record binds those goldens (bytes and sha256), names
//    the experts implementation the reference ran (`karume`, summing in pick order — ADR 0145
//    decision 6), the cut's layer count and its router top (9), and that the cut reference's last
//    layer equals the full reference's (`torch.equal` — the goldens are the full model's first
//    layers, not another model's)
// ② the graph form (no GPU): 12B's product contract, 2 × N `expert_linear` over `[128, out, in]`
//    weights in `int4-sym-g-f16`, each index the second output of a `topk` (k = 8) over the 128
//    experts, 2 + N embeddings (the token table, one per-expert scale per layer, the row
//    selection), no PLE, sliding slots of 8 KV heads × 256 and full slots of 2 × 512, and the
//    family admission (`admitGemma4` / `admitGemma4Ple`) accepting it with 26B's declaration
// ③ routing first (real GPU — ADR 0145 decision 6): the routing container (the same graph with
//    each layer's `topk` index as an extra output), teacher-forced on each golden's prompt, gives
//    the golden's expert ids at every row of every layer in pick order (top-k order, ties to the
//    smaller id), up to near-ties (below)
// ④ the product container (real GPU), teacher-forced the same way: sha256 regression rows of the
//    logits and the hidden state at the golden rows (ADR 0106 — this machine's rows), the census
//    (every `expert_linear` ran on its i4 + f16-step key, no weight decoded on the host), and the
//    largest resident weight — above the WebGPU defaults — held through a device acquired the way
//    the pipelines acquire theirs; and the argmax equal to the golden's at every checked golden row
//    whose torch top1 − top2 margin exceeds 2 × {@link ATOL} (token identity, not a tolerance band,
//    like `e2e_gemma4_12b_test.ts` ⑤ — a row with a smaller margin is reported, not judged)
// ⑤ upstream parity (real GPU — opt-in KARUME_UPSTREAM_PARITY=1): the logits and the hidden state
//    at the checked golden rows within {@link ATOL} of the torch goldens
// ⑥ the gemma pipeline over the cut container (real GPU): `Pipeline.fromAssets` with 26B's
//    declaration admits the graph, derives 26B-it's stop set from the tokenizer, estimates the
//    weights at their stored size (the experts' f16 steps included), and its first greedy token of
//    each golden prompt is ④'s argmax at the prompt's last row (the same graph, options and chunk
//    rows on the same device — so the pipeline's wiring of the MoE graph is held to the
//    teacher-forced run); one decode step follows, which runs `expert_linear` at one token
// ⑦ the exact expert cache (real GPU — ADR 0146 decision 7), at 16 and 64 slots per layer, whose
//    chunks hold ⌊slots / 8⌋ rows (2 and 8 — every pick of a chunk fits in the slots, and a chunk
//    is a deferred run within the sliding slack): (7a) the product container with
//    `SessionOptions.expertCache`, teacher-forced like ④ through the re-run face (a run that missed
//    is dropped with `commit(0)`, its experts loaded and the same chunk run again; only a run with
//    no miss is committed), gives at the golden rows the very bytes of the logits and the hidden
//    state that a resident Session gives at the same chunk rows on this device (word for word, no
//    tolerance — a chunk of other rows is another shape, so ④'s rows are not the reference); and
//    (7b) `Pipeline.fromAssets` with `expertCache` generates each golden prompt's first greedy
//    tokens equal to the pipeline's without it at the chunk rows the cache settles. Both require the
//    counters to show misses, loads and re-runs (a run with no miss would prove nothing). Both run
//    the host tier too (`expertCache.hostBytes` — experts held in RAM in front of the container,
//    which changes where a load's bytes come from, not which): 7a at 16 slots with every expert held
//    (no load reads the container) and at 64 slots with a quarter of them (RAM hits, container reads
//    and RAM evictions all happen), 7b at 64 slots with every expert held — each against the same
//    resident reference, with the same misses and loads as the run at the same slots without a tier
//
// A cut model's hidden state is not the final layer's, and a short cut's logits saturate at the
// softcap (±30): on cut 6, 7 to 757 entries of each golden row are exactly 30.0, so every margin
// is 0 and the argmax is the smallest id among the tied entries — ④ decides no row there and fails
// as vacuous. Cut 12 leaves the softcap: 15 of its 16 golden rows have a margin above 2 × atol.
//
// ## Near-ties (ADR 0145 decision 6)
//
// Routing is discrete: where two experts' router probabilities are closer than the numeric
// difference between this GPU and torch, either order is a correct answer, and a whole expert can
// swap at the 8th pick. Each row is judged against the golden's top 9 (`route_probs`):
//
// - `exact` — the 8 ids in the golden's order;
// - `order` — the same 8 ids, where every moved id sits at a golden position whose probability is
//   within {@link NEAR_TIE_FLOOR} of the probability at the position it moved from; the sum over the
//   picks runs in pick order, so only the rounding of that sum differs;
// - `swap` — as `order`, but the golden's 9th expert replaced a golden pick whose probability is
//   within the floor of the 9th's: one expert of the row is another one;
// - a failure — anything else (an id outside the golden's top 9, or a move between positions whose
//   probabilities are at least the floor apart).
//
// A swap at layer l, row p changes that row's output of layer l, so every later layer sees other
// keys and values from row p on: rows p and later of the layers after l are **downstream** of it,
// and whatever they give is counted apart, not judged (a failure taints the same region, so the
// counts name root causes only). The golden rows at p or later are downstream too: their logits and
// hidden state can legitimately differ, so ④'s argmax and ⑤'s tolerance skip them and say so, while the
// sha256 rows — a regression check against this machine's own earlier output — cover every row.
// `order` taints nothing.
//
// ## Assets
//
// The cut series `outputs/series/gemma4-26b-cut12/` ({@link SERIES_ROOT} — the one line to point
// this file at another cut), written by `gemma4.export_moe --layers --routing-outputs`, and the
// tokenizer series `outputs/series/gemma4-26b-tokenizer/` (`gemma4.tokenizer --model 26b`). A cut
// is not shipped, so every case is an implementation-time series check (ADR 0139 decision 6): it
// runs only with KARUME_SERIES=1 and SKIPs explicitly otherwise. A series that holds some of its
// files but not all FAILs instead of SKIPping.
//
// ## Memory
//
// One container on the GPU at a time: cut 6 is 3.0 GiB of weights and a cut of 12 layers about
// 6 GiB (the routing container is another copy, built after the product's Session is gone), the
// sliding slots 16 MiB per layer (k and v) and the full slots 32 MiB per full layer at capacity
// 4096 — within a 12 GB GPU. ⑥ holds the container's parts in host RAM (`fromAssets` takes bytes).
// ⑦ keeps the same rule: the resident Session (or pipeline) is disposed before the cached one is
// built, whose experts take 16 / 128 or 64 / 128 of the resident ones.

import { isNotFound, readFile, readTextFileSync, statSync } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  codecLayout,
  type GenerationContext,
  type GpuContext,
  type OpenedContainer,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
  type RunOutputs,
  type SafetensorsFile,
  type Tensor,
} from "@karume/runtime";
import {
  type ExpertCacheReport,
  Pipeline,
  parsePipelineConfig,
  type PipelineConfig,
} from "../../models/gemma.ts";
import { admitGemma4, admitGemma4Ple } from "../../models/src/gemma/admission.ts";
import { createExpertCacheSession } from "../../models/src/generation/expert-cache-session.ts";
import type { GenerationSession } from "../../models/src/generation/sequence.ts";
import { gemma4PleAssetSource } from "../../models/src/gemma/ple-index.ts";
import { gemma4RopeInputNames, ropeInputs, type RopeSpec } from "../../models/src/gemma/rope.ts";
import { planPrefillChunks, type PrefillChunk } from "../../models/src/generation/greedy.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";
import { seriesGraph } from "../../models/tests/helpers/series-graphs.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import {
  modelPresent,
  openSeriesContainer,
  resolveParts,
} from "../../runtime/tests/helpers/container-files.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import { requireCensus } from "../../runtime/tests/helpers/pipeline-census.ts";
import {
  f32ArtifactBytes,
  openReferences,
  referenceMismatchMessage,
  registerReferenceGate,
  settleOrObserve,
  sha256Hex,
} from "../../runtime/tests/helpers/reference.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/** The cut series (its graph names are in `helpers/series-graphs.ts`). */
const SERIES = "gemma4-26b-cut12";
/** Where the cut series is read from — the one line to point this file at another cut. */
const SERIES_ROOT = new URL(`../../../outputs/series/${SERIES}/`, import.meta.url);
const MODEL_FILE = "model.krm";
const ROUTING_DIR = "routing";
const MODEL_GRAPH = seriesGraph(SERIES);
const ROUTING_GRAPH = seriesGraph(SERIES, ROUTING_DIR);
const MODEL = new URL(MODEL_FILE, SERIES_ROOT);
const ROUTING = new URL(`${ROUTING_DIR}/${MODEL_FILE}`, SERIES_ROOT);
/** 26B's tokenizer series (`gemma4.tokenizer --model 26b`). */
const TOKENIZER = new URL(
  "../../../outputs/series/gemma4-26b-tokenizer/tokenizer.json",
  import.meta.url,
);
const REFERENCE_FILE = "reference.json";
const REFERENCE = new URL(REFERENCE_FILE, SERIES_ROOT);
/** The schema of the provenance record (spelling source: `gemma4/provenance.py`). */
const REFERENCE_SCHEMA = 1;
/** One golden per case (spelling source: `export_moe.CUT_PREFIX` + `decode_series.GREEDY_SUFFIX`). */
const CUT_PREFIX = "cut.";
const SUFFIX = ".safetensors";
/** The tensors of one cut golden (`export_moe.write_cut_goldens`). */
const GOLDEN_KEYS = [
  "hidden",
  "layer_hidden",
  "logits",
  "prompt",
  "route_ids",
  "route_probs",
  "rows",
] as const;
/** The experts implementation the goldens must name (`gemma4.experts.EXPERTS_RECORD`). */
const EXPERTS_IMPLEMENTATION = "karume";
const EXPERTS_SUM = "over the k picks in pick order";
/** The rule the record states for the cut's last layer against the full reference. */
const CONSISTENCY_RULE = "torch.equal";
/**
 * The case whose prompt is longer than the sliding ring (1032 rows) and than one prefill chunk
 * (768 rows), so the ring evicts and the prefill runs in two chunks. The record MUST hold it.
 */
const WINDOW_CASE = "window-en";
/** The commands that write the cut series and the tokenizer series. */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --no-sync python -m gemma4.export_moe --layers --routing-outputs" +
  " (and python -m gemma4.tokenizer --model 26b)";

/** google/gemma-4-26B-A4B-it config.json at 4d7ae498 (`text_config`). */
const HIDDEN = 2816;
const VOCAB = 262144;
const EXPERTS = 128;
const TOP_K = 8;
/** `router_top`: the golden's ids and probabilities per row (one more than the picks). */
const ROUTER_TOP = TOP_K + 1;
const MOE_INTERMEDIATE = 704;
const SLIDING_WINDOW = 1024;
/** Rows a sliding slot holds beyond the window (`export_decode.SLIDING_SLACK_ROWS`). */
const SLIDING_SLACK_ROWS = 8;
const SLIDING_KV_HEADS = 8;
const FULL_KV_HEADS = 2;
const SLIDING_HEAD_DIM = 256;
const FULL_HEAD_DIM = 512;
/** Full-attention layers are every sixth (5, 11, …, 29); a cut ends on one. */
const isFullLayer = (layer: number): boolean => layer % 6 === 5;
const ROPE: RopeSpec = {
  sliding_attention: { theta: 10000, headDim: SLIDING_HEAD_DIM, rotaryDim: SLIDING_HEAD_DIM },
  full_attention: { theta: 1000000, headDim: FULL_HEAD_DIM, rotaryDim: 128 },
};
/**
 * 26B's declaration as the q4_0 QAT line writes it (`gemma4.distribution.gemma4_pipeline_config`):
 * the line's chunk rows and capacity, `max_position_embeddings`, the RoPE above, the sampler of
 * `generation_config.json`, 12B's chat template (ADR 0145 decision 5) and no suppressed ids (26B's
 * generation defaults declare none). The cut graph is the product graph over fewer layers, so it
 * takes the same declaration.
 */
const PIPELINE_CONFIG: PipelineConfig = parsePipelineConfig({
  chunkLength: 768,
  maxChunkLength: 768,
  maxPosition: 262144,
  capacity: 4096,
  rope: ROPE,
  sampler: { temperature: 1, topK: 64, topP: 0.95 },
  chatDialect: "gemma4-unified",
});
const CHUNK_LENGTH = PIPELINE_CONFIG.chunkLength;
const CAPACITY = PIPELINE_CONFIG.capacity;
/** google/gemma-4-26B-A4B-it generation_config.json `eos_token_id`, sorted. */
const STOP_TOKENS = [1, 50, 106] as const;
/** The WebGPU default of `maxStorageBufferBindingSize` (128 MiB — `karume.limits`). */
const DEFAULT_BINDING_BYTES = 134_217_728;

/** Graph input names (source of truth: the constants of `export_product.py`). */
const INPUT_IDS = "input_ids";
const SOFT_EMBEDS = "soft_embeds";
const SOFT_MASK = "soft_mask";
const LAST_ROW = "last_row";
const ROW_SYMBOL = "R";
const CAPACITY_SYMBOL = "C";
const slotName = (layer: number, part: "k" | "v"): string => `l${layer}.${part}`;

/**
 * The floor under which two of the golden's router probabilities are a near-tie (ADR 0145 decision
 * 6): a move between them is reported apart, not failed. 1e-4 is the starting value, set before
 * measuring. Measured 2026-11-08 (capital-en / capital-ja / context-en / window-en): on cut 6 every
 * row exact, 2 / 3 / 189 / 457 rows with an 8th − 9th gap under the floor (the smallest 3.7e-7,
 * window-en — exact too); on cut 12 no failure and no swap, 2 rows of window-en in another pick
 * order within the floor, 7 / 12 / 422 / 935 rows with an 8th − 9th gap under it.
 */
const NEAR_TIE_FLOOR = 1e-4;
/**
 * The gemma4 logits tolerance (`e2e_gemma4_test.ts`, rtol 0): ⑤'s band for the logits and the
 * hidden state, and ④'s argmax gate (a golden margin above 2 × atol cannot flip within it).
 * Measured 2026-11-08, worst over the four cases: logits 6.6e-4 / hidden 1.7e-3 on cut 6, logits
 * 2.6e-4 / hidden 1.3e-3 on cut 12 (the hidden state's entries reach about 1.4e3).
 */
const ATOL = 1e-2;

const exists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/** The last path segment of a directory URL (the name the record states). */
const seriesName = (root: URL): string =>
  root.pathname
    .split("/")
    .filter((part) => part !== "")
    .at(-1) ?? "";

/** Narrows the provenance JSON one level (unknown shapes fail with a clear message). */
const objectAt = (value: unknown, where: string): Map<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${REFERENCE_FILE}: ${where} is not an object`);
  }
  return new Map(Object.entries(value));
};

const readRecord = (): Map<string, unknown> =>
  objectAt(JSON.parse(readTextFileSync(REFERENCE)), "root");
const referenceOf = (record: Map<string, unknown>): Map<string, unknown> =>
  objectAt(record.get("reference"), "reference");

const goldenFile = (name: string): string => `${CUT_PREFIX}${name}${SUFFIX}`;
const goldenUrl = (name: string): URL => new URL(goldenFile(name), SERIES_ROOT);

/**
 * The cases the record binds (`reference.goldens` keys), sorted; empty without KARUME_SERIES=1 or
 * without the record (the completeness case then names what is missing). Read from the record, so a
 * case the recipe adds runs without editing this file.
 */
const CASES: readonly string[] = (() => {
  if (!SERIES_CHECKS || !exists(REFERENCE)) return [];
  return [...objectAt(referenceOf(readRecord()).get("goldens"), "reference.goldens").keys()]
    .map((file) => {
      if (!file.startsWith(CUT_PREFIX) || !file.endsWith(SUFFIX)) {
        throw new Error(
          `${REFERENCE_FILE}: reference.goldens key '${file}' is not ${goldenFile("<case>")}`,
        );
      }
      return file.slice(CUT_PREFIX.length, file.length - SUFFIX.length);
    })
    .sort();
})();

/** Every file the cases read. */
const ASSET_FILES: readonly URL[] = [REFERENCE, MODEL, ROUTING, TOKENIZER, ...CASES.map(goldenUrl)];
const present = (url: URL): boolean =>
  url.pathname.endsWith(".krm") ? modelPresent(url) : exists(url);

/** Whether every asset is here (the cases past ① need all of them). */
const AVAILABLE = CASES.length > 0 && ASSET_FILES.every(present);
/** Whether anything of the series is here (a partial series FAILs the completeness case). */
const ANY_PRESENT =
  SERIES_CHECKS && (exists(REFERENCE) || modelPresent(MODEL) || modelPresent(ROUTING));

if (SERIES_CHECKS && !AVAILABLE) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} or the 26B tokenizer series is absent or incomplete, so the ` +
      `Gemma 4 26B cut cases SKIP. Generate: ${GENERATE_COMMAND}`,
  );
}

const stringAt = (entries: Map<string, unknown>, key: string, where: string): string => {
  const value = entries.get(key);
  if (typeof value !== "string" || value === "") {
    throw new Error(`${REFERENCE_FILE}: ${where}.${key} is not a non-empty string`);
  }
  return value;
};

const integerAt = (entries: Map<string, unknown>, key: string, where: string): number => {
  const value = entries.get(key);
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${REFERENCE_FILE}: ${where}.${key} is not a positive integer`);
  }
  return value;
};

/** The record's `reference.cut` section. */
const cutRecord = (): Map<string, unknown> =>
  objectAt(referenceOf(readRecord()).get("cut"), "reference.cut");

/** The cut's layer count as the record states it (② and ③ hold the graphs and goldens to it). */
const cutLayers = (): number => integerAt(cutRecord(), "layers", "reference.cut");

test.skipIf(!ANY_PRESENT)(
  `Gemma 4 26B cut assets: the containers, the tokenizer and every bound golden are present${SERIES_CHECKS_OPT_IN}`,
  () => {
    const missing = ASSET_FILES.filter((url) => !present(url)).map((url) => url.pathname);
    assertEquals(missing, [], `missing files (${GENERATE_COMMAND})`);
    assert(CASES.length > 0, `${REFERENCE.pathname} binds no golden`);
  },
);

test.skipIf(!AVAILABLE)(
  `Gemma 4 26B cut provenance: the record binds the goldens, the experts implementation and the cut (no GPU)${SERIES_CHECKS_OPT_IN}`,
  async () => {
    const record = readRecord();
    assertEquals(record.get("schema"), REFERENCE_SCHEMA, `${REFERENCE_FILE} schema`);
    assertEquals(
      stringAt(record, "series", "root"),
      seriesName(SERIES_ROOT),
      "the record's series",
    );
    const reference = referenceOf(record);
    assertEquals(
      stringAt(reference, "series", "reference"),
      seriesName(SERIES_ROOT),
      "the bound series",
    );
    assert(
      CASES.includes(WINDOW_CASE),
      `the bound cases [${CASES.join(", ")}] lack '${WINDOW_CASE}'`,
    );
    const goldens = objectAt(reference.get("goldens"), "reference.goldens");
    for (const name of CASES) {
      const file = goldenFile(name);
      const digest = objectAt(goldens.get(file), `reference.goldens.${file}`);
      const bytes = await readFile(goldenUrl(name));
      assertEquals(bytes.byteLength, digest.get("bytes"), `${file}: bytes`);
      assertEquals(
        await sha256Hex(bytes),
        digest.get("sha256"),
        `${file}: sha256 (the goldens are of different runs — re-export the cut series)`,
      );
    }

    // The goldens name the experts implementation that ran (ADR 0145 decision 6): Karume's, whose
    // eager body is upstream's per-expert math summed over the picks in pick order — the order the
    // runtime sums in (decision 2).
    const experts = objectAt(reference.get("experts"), "reference.experts");
    assertEquals(experts.get("implementation"), EXPERTS_IMPLEMENTATION, "experts implementation");
    assertEquals(experts.get("sum"), EXPERTS_SUM, "experts summation order");

    const cut = cutRecord();
    const layers = integerAt(cut, "layers", "reference.cut");
    assert(isFullLayer(layers - 1), `the cut ends on layer ${layers - 1}, not a full layer`);
    assertEquals(cut.get("routerTop"), ROUTER_TOP, "routerTop (8 picks + the 9th)");
    assertEquals(cut.get("routingOutputs"), true, "routingOutputs (the routing container)");
    // The window case crosses the sliding ring and one prefill chunk (the ring evicts, the prefill
    // runs in two chunks).
    const { prompt } = await readGolden(WINDOW_CASE, layers);
    const ring = SLIDING_WINDOW + SLIDING_SLACK_ROWS;
    assert(prompt.length > ring, `${WINDOW_CASE}: T=${prompt.length} does not exceed the ring`);
    assert(prompt.length > CHUNK_LENGTH, `${WINDOW_CASE}: T=${prompt.length} fits one chunk`);
    // MUST: the cut reference's last layer equals the full reference's (`assert_cut_consistency`),
    // or the goldens describe another model than the first layers of 26B.
    const consistency = objectAt(cut.get("consistency"), "reference.cut.consistency");
    assertEquals(
      consistency.get("rule"),
      CONSISTENCY_RULE,
      "the cut's consistency with the full reference (a cut written with --no-full-check is not " +
        "bound to the full model)",
    );
  },
);

const goldenView = (file: SafetensorsFile, name: string, dtype: "I32" | "F32", where: string) => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `${where}: no '${name}'`);
  assertEquals(view.dtype, dtype, `${where}: '${name}' dtype`);
  return view;
};

/** One cut golden (`export_moe` layout), its shapes held to the record's cut. */
type CutGolden = {
  readonly prompt: Int32Array<ArrayBuffer>;
  /** Absolute prompt rows of `logits` / `hidden` (ascending; the last is the prompt's last row). */
  readonly rows: readonly number[];
  /** `[rows, V]`. */
  readonly logits: Float32Array<ArrayBuffer>;
  /** `[rows, H]`. */
  readonly hidden: Float32Array<ArrayBuffer>;
  /** `[layers, T, 9]` — the router's top 9 ids per row, in top-k order. */
  readonly routeIds: Int32Array<ArrayBuffer>;
  /** `[layers, T, 9]` — their probabilities. */
  readonly routeProbs: Float32Array<ArrayBuffer>;
};

const readGolden = async (name: string, layers: number): Promise<CutGolden> => {
  const bytes = await readFile(goldenUrl(name));
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const where = goldenFile(name);
  assertEquals([...file.tensors.keys()].sort(), [...GOLDEN_KEYS], `${where}: tensors`);
  const i32 = (key: string) => {
    const view = goldenView(file, key, "I32", where);
    return {
      shape: view.shape,
      data: new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4),
    };
  };
  const f32 = (key: string) => {
    const view = goldenView(file, key, "F32", where);
    return {
      shape: view.shape,
      data: new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4),
    };
  };
  const prompt = i32("prompt").data;
  const rows = [...i32("rows").data];
  const logits = f32("logits");
  const hidden = f32("hidden");
  const routeIds = i32("route_ids");
  const routeProbs = f32("route_probs");
  const tokens = prompt.length;
  assertEquals(logits.shape, [rows.length, VOCAB], `${where}: logits shape`);
  assertEquals(hidden.shape, [rows.length, HIDDEN], `${where}: hidden shape`);
  assertEquals(routeIds.shape, [layers, tokens, ROUTER_TOP], `${where}: route_ids shape`);
  assertEquals(routeProbs.shape, [layers, tokens, ROUTER_TOP], `${where}: route_probs shape`);
  assert(
    rows.length > 0 &&
      rows.at(-1) === tokens - 1 &&
      rows.every((row, index) => row >= 0 && (index === 0 || row > rows[index - 1])),
    `${where}: rows [${rows.join(", ")}] are not ascending prompt rows ending on row ${tokens - 1}`,
  );
  return {
    prompt,
    rows,
    logits: logits.data,
    hidden: hidden.data,
    routeIds: routeIds.data,
    routeProbs: routeProbs.data,
  };
};

const openPrepared = async (representative: URL, graph: string): Promise<PreparedModel> =>
  prepareContainer(await openSeriesContainer(representative), graph);

/**
 * The largest single resident weight buffer of `graph` — the weight side of the demand assembly
 * bakes into `requiredLimits` (`karume.dist.component_demand_bytes`: the payload of one merged
 * supply, its blocks joined back into one buffer, or of its companion scale).
 */
const largestResidentWeight = (
  opened: OpenedContainer,
  graph: string,
): { readonly name: string; readonly bytes: number } => {
  let largest = { name: "", bytes: 0 };
  for (const [name, supply] of opened.graphs[graph].supplies) {
    const bytes = Math.max(
      supply.blocks.reduce((total, block) => total + block.payloadBytes, 0),
      supply.scale?.payloadBytes ?? 0,
    );
    if (bytes > largest.bytes) largest = { name, bytes };
  }
  return largest;
};

/**
 * ② The cut graph's form. Returns the outputs after logits and hidden (the routing container's
 * per-layer indices; none on the product container).
 *
 * MUST: the expert block is checked here, not only through the numbers — a graph whose experts were
 * written as F32, or whose `expert_linear` index came from anything but a `topk` over the 128
 * experts, gives the right numbers on these cases and is still not the form ADR 0145 ships.
 */
const assertCutForm = (model: PreparedModel, layers: number, where: string): readonly string[] => {
  const { graph } = model;
  assertEquals(
    graph.inputs.map((spec) => spec.name).sort(),
    [INPUT_IDS, LAST_ROW, SOFT_EMBEDS, SOFT_MASK, ...gemma4RopeInputNames()].sort(),
    `${where}: graph inputs (12B's product contract)`,
  );
  assertEquals([...graph.symbols].sort(), [CAPACITY_SYMBOL, "M", ROW_SYMBOL], `${where}: symbols`);
  assertEquals(
    graph.values[graph.outputs[0]].shape,
    [1, ROW_SYMBOL, VOCAB],
    `${where}: output 0 (logits)`,
  );
  assertEquals(
    graph.values[graph.outputs[1]].shape,
    [1, ROW_SYMBOL, HIDDEN],
    `${where}: output 1 (hidden)`,
  );

  const ops = (op: string) => graph.nodes.filter((node) => node.op === op);
  assertEquals(ops("attention").length, layers, `${where}: attention nodes (the cut's layers)`);
  const topks = ops("topk");
  assertEquals(topks.length, layers, `${where}: router topk nodes (one per layer)`);
  const routerIndices = new Set<string>();
  for (const node of topks) {
    assertEquals(node.attrs["k"], TOP_K, `${where}: topk k`);
    assertEquals(
      graph.values[node.ins[0]].shape,
      ["M", EXPERTS],
      `${where}: topk input (the router probabilities)`,
    );
    routerIndices.add(node.outs[1]);
  }
  const experts = ops("expert_linear");
  assertEquals(experts.length, 2 * layers, `${where}: expert_linear nodes (gate_up, down)`);
  const shapes = new Set<string>();
  for (const node of experts) {
    assert(
      routerIndices.has(node.ins[1]),
      `${where}: expert_linear index '${node.ins[1]}' is not a router topk's index`,
    );
    const weight = node.ins[2];
    assertEquals(
      graph.initializers[weight]?.storage?.codec,
      "int4-sym-g-f16",
      `${where}: '${weight}' codec (i4 codes, f16 steps — ADR 0145 decision 3)`,
    );
    shapes.add(JSON.stringify(graph.values[weight].shape));
  }
  assertEquals(
    [...shapes].sort(),
    [
      JSON.stringify([EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN]),
      JSON.stringify([EXPERTS, HIDDEN, MOE_INTERMEDIATE]),
    ].sort(),
    `${where}: expert weight shapes (gate_up, down)`,
  );
  // The token table, one per-expert scale lookup per layer (indexed by the router's picks) and the
  // row selection by `last_row`.
  const embeddings = ops("embedding");
  assertEquals(embeddings.length, 2 + layers, `${where}: embedding nodes (2 + layers)`);
  assertEquals(
    embeddings.filter((node) => routerIndices.has(node.ins[1])).length,
    layers,
    `${where}: per-expert scale lookups`,
  );
  // The rest of the weights of rank ≥ 2 are on the grid (i4): none was written as F32.
  const census: Record<string, number> = {};
  for (const initializer of Object.values(graph.initializers)) {
    if (initializer.storage === undefined) continue;
    const layout = codecLayout(initializer.storage.codec);
    census[layout] = (census[layout] ?? 0) + 1;
  }
  assertEquals(census["i4f16"], 2 * layers, `${where}: i4f16 weights (${JSON.stringify(census)})`);
  assert((census["i4"] ?? 0) > 0, `${where}: no i4 weight (${JSON.stringify(census)})`);

  // Every layer owns its KV; full layers keep K = V in two slots of 2 heads.
  const expectedSlots = Array.from({ length: layers }, (_, layer) => [
    slotName(layer, "k"),
    slotName(layer, "v"),
  ])
    .flat()
    .sort();
  assertEquals(Object.keys(graph.states).sort(), expectedSlots, `${where}: state slots`);
  for (let layer = 0; layer < layers; layer += 1) {
    const expected = isFullLayer(layer)
      ? [1, FULL_KV_HEADS, CAPACITY_SYMBOL, FULL_HEAD_DIM]
      : [1, SLIDING_KV_HEADS, SLIDING_WINDOW + SLIDING_SLACK_ROWS, SLIDING_HEAD_DIM];
    for (const part of ["k", "v"] as const) {
      assertEquals(
        graph.states[slotName(layer, part)].shape,
        expected,
        `${where}: ${slotName(layer, part)}`,
      );
    }
  }
  return graph.outputs.slice(2);
};

test.skipIf(!AVAILABLE)(
  `Gemma 4 26B cut form: 12B's product contract with the expert block, and the family admission takes it (no GPU)${SERIES_CHECKS_OPT_IN}`,
  async () => {
    const layers = cutLayers();
    const product = await openPrepared(MODEL, MODEL_GRAPH);
    assertEquals(assertCutForm(product, layers, "product"), [], "product outputs past hidden");
    const routing = await openPrepared(ROUTING, ROUTING_GRAPH);
    const indices = assertCutForm(routing, layers, "routing");
    assertEquals(indices.length, layers, "routing outputs (one topk index per layer)");
    for (const name of indices) {
      assertEquals(
        routing.graph.values[name],
        { dtype: "i32", shape: ["M", TOP_K] },
        `routing output '${name}'`,
      );
    }
    // The family admission over 26B's declaration: the graph and the container settle the facts
    // (no PLE, the capacity symbol, the vocabulary), with nothing named after the model.
    const component = {
      graph: product.graph,
      ...gemma4PleAssetSource(await openSeriesContainer(MODEL)),
    };
    assertEquals(admitGemma4Ple(component, "test"), false, "26B has no per-layer inputs");
    const admitted = admitGemma4(component, PIPELINE_CONFIG, undefined, "test");
    assertEquals(admitted.vocabSize, VOCAB, "vocabulary");
    assertEquals(admitted.capacitySymbol, CAPACITY_SYMBOL, "capacity symbol");
  },
);

// ---------------------------------------------------------------------------
// Teacher-forced runs (③ – ⑥)
// ---------------------------------------------------------------------------

/** One chunk of a teacher-forced prompt and the golden rows inside it (relative to the chunk). */
type ChunkRun = {
  readonly chunk: PrefillChunk;
  readonly selected: readonly number[];
  readonly outputs: RunOutputs;
};

/**
 * Feeds `prompt` through `session` in prefill chunks of `chunkLength` rows (default: the
 * distribution's) and hands each chunk's outputs to `onChunk`, selecting through `last_row` the
 * golden `rows` that fall in the chunk (a chunk without one selects its last row, which nothing
 * reads). `session` is a Session or a face over one (⑦'s re-run face of the expert cache).
 *
 * The soft inputs are bound as text-only runs bind them (zeros, false — the graph takes them on
 * every run until 26B's vision lands).
 */
const teacherForce = async (
  session: GenerationSession,
  prompt: Int32Array<ArrayBuffer>,
  rows: readonly number[],
  onChunk: (run: ChunkRun) => void,
  chunkLength: number = CHUNK_LENGTH,
): Promise<void> => {
  const context: GenerationContext = await session.createGenerationContext({
    bindings: { [CAPACITY_SYMBOL]: CAPACITY },
    chunkLength,
  });
  try {
    for (const chunk of planPrefillChunks(prompt.length, chunkLength)) {
      const ids = new Int32Array(chunkLength);
      const positions = new Int32Array(chunkLength);
      for (let row = 0; row < chunk.queryLength; row += 1) {
        ids[row] = prompt[chunk.position + row];
        positions[row] = chunk.position + row;
      }
      const selected = rows
        .filter((row) => row >= chunk.position && row < chunk.position + chunk.queryLength)
        .map((row) => row - chunk.position);
      const lastRow = selected.length > 0 ? selected : [chunk.queryLength - 1];
      const outputs = await session.run(
        {
          [INPUT_IDS]: { dtype: "i32", shape: [1, chunkLength], data: ids },
          [SOFT_EMBEDS]: {
            dtype: "f32",
            shape: [1, chunkLength, HIDDEN],
            data: new Float32Array(chunkLength * HIDDEN),
          },
          [SOFT_MASK]: {
            dtype: "bool",
            shape: [1, chunkLength, 1],
            data: new Uint32Array(chunkLength),
          },
          ...ropeInputs(ROPE, positions),
          [LAST_ROW]: { dtype: "i32", shape: [lastRow.length], data: Int32Array.from(lastRow) },
        },
        undefined,
        { context, queryLength: chunk.queryLength },
      );
      onChunk({ chunk, selected, outputs });
    }
  } finally {
    // MUST: released on a failure midway too (the full slots hold the capacity's rows).
    await context.dispose();
  }
};

/** A row judged against the golden's top 9 (module doc, "Near-ties"). */
type RowVerdict = "exact" | "order" | "swap" | "failure";

/**
 * Judges one row: `ids` are the GPU's 8 picks, `golden` / `probs` the golden's top 9 ids and their
 * probabilities, in top-k order.
 */
const judgeRoutingRow = (
  ids: ArrayLike<number>,
  golden: ArrayLike<number>,
  probs: ArrayLike<number>,
): RowVerdict => {
  let exact = true;
  const taken = new Set<number>();
  for (let pick = 0; pick < TOP_K; pick += 1) {
    const id = ids[pick];
    if (taken.has(id)) return "failure";
    taken.add(id);
    if (id === golden[pick]) continue;
    exact = false;
    let at = -1;
    for (let position = 0; position < ROUTER_TOP; position += 1) {
      if (golden[position] === id) at = position;
    }
    if (at < 0 || Math.abs(probs[pick] - probs[at]) >= NEAR_TIE_FLOOR) return "failure";
  }
  if (exact) return "exact";
  let dropped = false;
  for (let position = 0; position < TOP_K; position += 1) {
    if (taken.has(golden[position])) continue;
    // A golden pick left out: only the 9th can have taken its place, and only from within the floor.
    if (Math.abs(probs[position] - probs[TOP_K]) >= NEAR_TIE_FLOOR) return "failure";
    dropped = true;
  }
  return dropped ? "swap" : "order";
};

/** The routing comparison of one case (③), with what ④ and ⑤ need to skip downstream rows. */
type RoutingReport = {
  readonly rows: number;
  /** Rows whose golden 8th − 9th gap is below the floor. */
  readonly underFloor: number;
  readonly order: number;
  readonly swaps: number;
  readonly failures: readonly string[];
  /** Rows downstream of a swap or a failure that were not exact (counted, not judged). */
  readonly downstream: number;
  /** The first prompt row whose final outputs are downstream (`T` = none). */
  readonly downstreamFrom: number;
};

const judgeRouting = (
  golden: CutGolden,
  picks: readonly Int32Array<ArrayBuffer>[],
): RoutingReport => {
  const tokens = golden.prompt.length;
  let underFloor = 0;
  let order = 0;
  let swaps = 0;
  let downstream = 0;
  const failures: string[] = [];
  // The first row of this layer's input that is downstream of an earlier layer's swap or failure.
  let from = tokens;
  for (const [layer, layerPicks] of picks.entries()) {
    let next = from;
    for (let row = 0; row < tokens; row += 1) {
      const base = (layer * tokens + row) * ROUTER_TOP;
      const goldenIds = golden.routeIds.subarray(base, base + ROUTER_TOP);
      const probs = golden.routeProbs.subarray(base, base + ROUTER_TOP);
      if (probs[TOP_K - 1] - probs[TOP_K] < NEAR_TIE_FLOOR) underFloor += 1;
      const ids = layerPicks.subarray(row * TOP_K, (row + 1) * TOP_K);
      const verdict = judgeRoutingRow(ids, goldenIds, probs);
      if (row >= from) {
        if (verdict !== "exact") downstream += 1;
        continue;
      }
      if (verdict === "order") order += 1;
      if (verdict === "swap") swaps += 1;
      if (verdict === "failure") {
        failures.push(
          `layer ${layer} row ${row}: GPU [${[...ids].join(", ")}] vs golden ` +
            `[${[...goldenIds].join(", ")}] (probabilities ` +
            `${[...probs].map((value) => value.toExponential(4)).join(", ")})`,
        );
      }
      if (verdict === "swap" || verdict === "failure") next = Math.min(next, row);
    }
    from = next;
  }
  return {
    rows: picks.length * tokens,
    underFloor,
    order,
    swaps,
    failures,
    downstream,
    downstreamFrom: from,
  };
};

/** The product container's outputs at the golden rows of one case. */
type ProductRows = {
  readonly logits: Float32Array<ArrayBuffer>;
  readonly hidden: Float32Array<ArrayBuffer>;
};

/** Copies output rows `[1,R,width]` into `into` at the golden rows they select. */
const placeRows = (
  tensor: Tensor,
  width: number,
  run: ChunkRun,
  rows: readonly number[],
  into: Float32Array<ArrayBuffer>,
  where: string,
): void => {
  assert(tensor.dtype === "f32", `${where}: dtype ${tensor.dtype}`);
  assertEquals(tensor.shape, [1, run.selected.length, width], `${where}: shape`);
  run.selected.forEach((relative, index) => {
    into.set(
      tensor.data.subarray(index * width, (index + 1) * width),
      rows.indexOf(run.chunk.position + relative) * width,
    );
  });
};

/** The argmax and the top1 − top2 margin of one row (ties to the smaller id, as torch's argmax). */
const topOf = (row: Float32Array): { readonly top: number; readonly margin: number } => {
  let top = 0;
  let best = row[0];
  let second = Number.NEGATIVE_INFINITY;
  for (let index = 1; index < row.length; index += 1) {
    const value = row[index];
    if (value > best) {
      second = best;
      best = value;
      top = index;
    } else if (value > second) {
      second = value;
    }
  }
  assert(Number.isFinite(best), `the row's maximum is not finite (${best})`);
  return { top, margin: best - second };
};

const rowOf = (data: Float32Array<ArrayBuffer>, index: number, width: number) =>
  data.subarray(index * width, (index + 1) * width);

/** The largest |actual − expected| over the given rows (NaN when any element is NaN). */
const maxAbs = (
  actual: Float32Array<ArrayBuffer>,
  expected: Float32Array<ArrayBuffer>,
  width: number,
  indices: readonly number[],
): number => {
  let worst = 0;
  for (const index of indices) {
    const a = rowOf(actual, index, width);
    const e = rowOf(expected, index, width);
    for (let column = 0; column < width; column += 1) {
      const difference = Math.abs(a[column] - e[column]);
      worst = Number.isNaN(difference) ? Number.NaN : Math.max(worst, difference);
      if (Number.isNaN(worst)) return worst;
    }
  }
  return worst;
};

/**
 * sha256 reference rows of the logits and the hidden state at the golden rows (ADR 0106). The case
 * ids carry the series' name, so the rows of one cut never stand for another's.
 */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/gemma4-26b-cut.json", import.meta.url),
);
/** Settlements and artifacts (`outputs/verify/<environment key>/<date>_gemma4-26b-cut/`). */
const results = openResults("gemma4-26b-cut");
const OUTPUTS = ["logits", "hidden"] as const;
const referenceId = (name: string, output: (typeof OUTPUTS)[number]): string =>
  `${seriesName(SERIES_ROOT)}/${name}/${output}`;
const REFERENCE_IDS: readonly string[] = CASES.flatMap((name) =>
  OUTPUTS.map((output) => referenceId(name, output)),
);

describe.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  `Gemma 4 26B cut, teacher-forced on the golden prompts (real GPU)${SERIES_CHECKS_OPT_IN}`,
  () => {
    let gpu: GpuContext | undefined;
    let layers = 0;
    const goldens = new Map<string, CutGolden>();
    const routing = new Map<string, RoutingReport>();
    const product = new Map<string, ProductRows>();
    let census:
      | {
          readonly experts: readonly string[];
          readonly dispatches: number;
          readonly expanded: number;
        }
      | undefined;
    let demand: { readonly name: string; readonly bytes: number } | undefined;
    let residentPayload = 0;

    // The GPU runs happen once here, the routing container's first (ADR 0145 decision 6); the cases
    // below judge what they recorded.
    beforeAll(async () => {
      layers = cutLayers();
      for (const name of CASES) goldens.set(name, await readGolden(name, layers));
      gpu = await acquireGpu();
      // These runs write this machine's sha rows: the adapter that runs is the one the key names.
      assertAdapterMatchesEnvironment(gpu);

      const routingModel = await openPrepared(ROUTING, ROUTING_GRAPH);
      const indexOutputs = routingModel.graph.outputs.slice(2);
      assertEquals(indexOutputs.length, layers, "routing outputs (one topk index per layer)");
      const routingSession = await routingModel.createContainerSession(gpu);
      try {
        for (const [name, golden] of goldens) {
          const picks = indexOutputs.map(() => new Int32Array(golden.prompt.length * TOP_K));
          await teacherForce(routingSession, golden.prompt, [], (run) => {
            indexOutputs.forEach((output, layer) => {
              const tensor = run.outputs[output];
              assert(tensor.dtype === "i32", `${name}: routing output '${output}' is not i32`);
              assertEquals(tensor.shape, [CHUNK_LENGTH, TOP_K], `${name}: '${output}' shape`);
              // Only the chunk's prompt rows (its pad rows run too, and are never read).
              picks[layer].set(
                tensor.data.subarray(0, run.chunk.queryLength * TOP_K),
                run.chunk.position * TOP_K,
              );
            });
          });
          routing.set(name, judgeRouting(golden, picks));
        }
      } finally {
        await routingSession.dispose();
        await settleReleases(gpu);
      }

      const opened = await openSeriesContainer(MODEL);
      demand = largestResidentWeight(opened, MODEL_GRAPH);
      for (const supply of opened.graphs[MODEL_GRAPH].supplies.values()) {
        residentPayload += supply.blocks.reduce((total, block) => total + block.payloadBytes, 0);
        residentPayload += supply.scale?.payloadBytes ?? 0;
      }
      const productModel = prepareContainer(opened, MODEL_GRAPH);
      const [logitsName, hiddenName] = productModel.graph.outputs;
      const session = await productModel.createContainerSession(gpu);
      try {
        for (const [name, golden] of goldens) {
          const logits = new Float32Array(golden.rows.length * VOCAB);
          const hidden = new Float32Array(golden.rows.length * HIDDEN);
          await teacherForce(session, golden.prompt, golden.rows, (run) => {
            if (run.selected.length === 0) return;
            placeRows(run.outputs[logitsName], VOCAB, run, golden.rows, logits, `${name} logits`);
            placeRows(run.outputs[hiddenName], HIDDEN, run, golden.rows, hidden, `${name} hidden`);
          });
          product.set(name, { logits, hidden });
          if (census === undefined) {
            const rows = requireCensus(session.diagnostics().lastRunPipelines, name);
            const experts = rows.filter((row) => row.key.startsWith("expert_linear:"));
            census = {
              experts: experts.map((row) => row.key),
              dispatches: experts.reduce((sum, row) => sum + row.dispatchCount, 0),
              expanded: session.diagnostics().storage?.hostExpandedBytes ?? Number.NaN,
            };
          }
        }
      } finally {
        await session.dispose();
        await settleReleases(gpu);
      }
    });
    afterAll(() => {
      gpu?.destroy();
    });

    /** The golden rows ④ and ⑤ judge: those before the first downstream row. */
    const checkedRows = (name: string): readonly number[] => {
      const golden = goldens.get(name);
      const report = routing.get(name);
      assert(golden !== undefined && report !== undefined, `${name}: not run`);
      return golden.rows.flatMap((row, index) => (row < report.downstreamFrom ? [index] : []));
    };

    it("③ routing: every row's 8 expert ids equal the golden's in pick order, near-ties apart", () => {
      const lines: string[] = [];
      const failures: string[] = [];
      for (const name of CASES) {
        const report = routing.get(name);
        assert(report !== undefined, `${name}: the routing run did not record this case`);
        lines.push(
          `${name}: ${report.rows} rows, ${report.underFloor} with 8th − 9th < ${NEAR_TIE_FLOOR}; ` +
            `near-ties ${report.order} order / ${report.swaps} swap; failures ` +
            `${report.failures.length}; downstream ${report.downstream} not exact; final rows ` +
            `downstream from row ${report.downstreamFrom} of ${goldens.get(name)?.prompt.length}`,
        );
        failures.push(...report.failures.map((line) => `${name}: ${line}`));
      }
      console.log(`[e2e] gemma4 26b cut routing (${layers} layers):\n  ${lines.join("\n  ")}`);
      assertEquals(failures, [], "routing mismatches outside the near-ties");
    });

    it("④ product: sha256 rows at the golden rows, the census and the raised binding limit", async () => {
      assert(
        gpu !== undefined && census !== undefined && demand !== undefined,
        "the product runs did not happen",
      );
      // The cut holds weights above the WebGPU defaults (the tied embedding, then each layer's
      // gate_up), resident through a device acquired the way the pipelines acquire theirs:
      // `acquireGpu` requests the adapter's limits, and the distribution's `requiredLimits` —
      // which assembly derives from this same demand (`karume.dist.component_demand_bytes`) — only
      // refuses an adapter below them before the download.
      assert(
        demand.bytes > DEFAULT_BINDING_BYTES,
        `the largest resident weight '${demand.name}' is ${demand.bytes} B, within the defaults`,
      );
      for (const limit of ["maxStorageBufferBindingSize", "maxBufferSize"] as const) {
        assert(
          gpu.limits[limit] >= demand.bytes,
          `the device's ${limit} ${gpu.limits[limit]} B < '${demand.name}' ${demand.bytes} B`,
        );
      }
      console.log(
        `[e2e] gemma4 26b cut: largest resident weight '${demand.name}' ${demand.bytes} B`,
      );
      assertEquals(census.expanded, 0, "hostExpandedBytes (no weight decoded on the host)");
      assertEquals(
        census.experts.filter((key) => !key.includes(":wi4h")),
        [],
        "expert_linear keys without the i4 + f16-step storage",
      );
      assert(census.dispatches >= 2 * layers, `expert_linear dispatches ${census.dispatches}`);

      const mismatches: string[] = [];
      for (const name of CASES) {
        const golden = goldens.get(name);
        const rows = product.get(name);
        assert(golden !== undefined && rows !== undefined, `${name}: not run`);
        await runRecordedCase(results, { id: name }, async () => {
          for (const output of OUTPUTS) {
            const id = referenceId(name, output);
            const outcome = await settleOrObserve(references, results, {
              id,
              artifact: `${name}-${output}.safetensors`,
              bytes: f32ArtifactBytes(
                output,
                [golden.rows.length, output === "logits" ? VOCAB : HIDDEN],
                rows[output],
              ),
            });
            if (outcome.settlement?.check.status === "fail") {
              mismatches.push(referenceMismatchMessage(id, outcome.settlement, references));
            }
          }
          return undefined;
        });
      }
      if (mismatches.length > 0) throw new Error(mismatches.join("\n"));
    });

    it("④ argmax: equal to the golden's at every checked golden row whose margin decides it", () => {
      const lines: string[] = [];
      let decided = 0;
      for (const name of CASES) {
        const golden = goldens.get(name);
        const rows = product.get(name);
        assert(golden !== undefined && rows !== undefined, `${name}: not run`);
        const checked = new Set(checkedRows(name));
        const verdicts = golden.rows.map((row, index) => {
          const expected = topOf(rowOf(golden.logits, index, VOCAB));
          const observed = topOf(rowOf(rows.logits, index, VOCAB));
          const shown = `@${row} ${observed.top === expected.top ? "=" : "≠"} (margin ${expected.margin.toExponential(2)}`;
          if (!checked.has(index)) return `${shown}, downstream of a near-tie swap — not judged)`;
          if (expected.margin <= 2 * ATOL) return `${shown}, ≤ 2 × atol — not judged)`;
          decided += 1;
          assertEquals(
            observed.top,
            expected.top,
            `${name}@${row}: argmax (GPU margin ${observed.margin} / golden margin ${expected.margin})`,
          );
          return `${shown})`;
        });
        lines.push(`${name}: ${verdicts.join(" ")}`);
      }
      console.log(`[e2e] gemma4 26b cut argmax:\n  ${lines.join("\n  ")}`);
      // A cut too short to leave the softcap decides no row (cut 6 — module doc): the check would
      // be vacuous, so it fails instead of passing on nothing.
      assert(
        decided > 0,
        "no golden row has a margin above 2 × atol — the argmax check decides nothing",
      );
    });

    it.skipIf(!UPSTREAM_PARITY)(
      `⑤ upstream parity: logits and hidden within atol ${ATOL} at the checked golden rows${UPSTREAM_PARITY_OPT_IN}`,
      () => {
        const lines: string[] = [];
        for (const name of CASES) {
          const golden = goldens.get(name);
          const rows = product.get(name);
          assert(golden !== undefined && rows !== undefined, `${name}: not run`);
          const checked = checkedRows(name);
          const logits = maxAbs(rows.logits, golden.logits, VOCAB, checked);
          const hidden = maxAbs(rows.hidden, golden.hidden, HIDDEN, checked);
          lines.push(
            `${name}: ${checked.length}/${golden.rows.length} rows checked; max |Δ| logits ` +
              `${logits.toExponential(3)}, hidden ${hidden.toExponential(3)}`,
          );
          assert(logits <= ATOL, `${name}: logits max |Δ| ${logits} > ${ATOL}`);
          assert(hidden <= ATOL, `${name}: hidden max |Δ| ${hidden} > ${ATOL}`);
        }
        console.log(`[e2e] gemma4 26b cut parity:\n  ${lines.join("\n  ")}`);
      },
    );

    it("⑥ gemma.Pipeline: fromAssets admits the cut with 26B's declaration, and its first greedy token is ④'s argmax", async () => {
      assert(gpu !== undefined, "no GPU");
      const parts = await Promise.all(resolveParts(MODEL).map((part) => readFile(part)));
      // The teacher-forced runs' chunk rows: no buckets, so every prefill is one shape of
      // `chunkLength` rows and the last row's logits are those ④ took.
      const pipeline = await Pipeline.fromAssets(
        { config: PIPELINE_CONFIG, model: parts, tokenizer: await readFile(TOKENIZER) },
        { gpu, chunkBuckets: [] },
      );
      const lines: string[] = [];
      try {
        assertEquals(pipeline.program.vocabSize, VOCAB, "vocabulary");
        assertEquals(pipeline.tokenizer.chatDialect, "gemma4-unified", "chat dialect");
        // The stop set the tokenizer asset gives is 26B-it's `generation_config.json` eos_token_id.
        assertEquals(
          [...pipeline.program.stopTokens].sort((a, b) => a - b),
          [...STOP_TOKENS],
          "stop tokens",
        );
        const estimate = pipeline.estimateSessionMemory();
        // The estimate counts each resident weight at its stored size — the f16 steps of the
        // experts at 2 bytes, not as the f32 steps of `int4-sym-g` (285 MB more on cut 6).
        assertEquals(
          estimate.resident.weights.totalBytes,
          residentPayload,
          "estimated weights against the container's resident payload",
        );
        lines.push(
          `estimated peak ${estimate.peakAccountedBytes} B (weights ` +
            `${estimate.resident.weights.totalBytes} B; the container's resident payload ` +
            `${residentPayload} B) at capacity ${CAPACITY}`,
        );
        for (const name of CASES) {
          const golden = goldens.get(name);
          const rows = product.get(name);
          assert(golden !== undefined && rows !== undefined, `${name}: not run`);
          const expected = topOf(rowOf(rows.logits, golden.rows.length - 1, VOCAB)).top;
          const sequence = await pipeline.sequence();
          try {
            const ids: number[] = [];
            // Two tokens: the prefill's, then one decode step (`expert_linear` at one token).
            const stream = sequence.generate({ prompt: [...golden.prompt], maxNewTokens: 2 });
            for await (const event of stream) {
              if (event.kind === "token") ids.push(event.id);
            }
            const stop = await stream.done;
            if (stop.reason === "eos") ids.push(stop.token);
            assert(ids.length > 0, `${name}: no token (stop '${stop.reason}')`);
            assertEquals(ids[0], expected, `${name}: the first greedy token against ④'s argmax`);
            lines.push(`${name}: tokens [${ids.join(", ")}] (stop '${stop.reason}')`);
          } finally {
            await sequence.dispose();
          }
        }
      } finally {
        await pipeline.dispose();
        await settleReleases(gpu);
      }
      console.log(`[e2e] gemma4 26b cut pipeline:\n  ${lines.join("\n  ")}`);
    });
  },
);

// ---------------------------------------------------------------------------
// ⑦ The exact expert cache (ADR 0146)
// ---------------------------------------------------------------------------

/**
 * The rows of one prefill chunk under the cache at `slots`: a chunk's picks fit in the slots at once,
 * and a chunk is a deferred run within the sliding slack (ADR 0146 decision 4) — 2 rows at 16 slots,
 * 8 at 64.
 */
const cacheChunkRows = (slots: number): number =>
  Math.min(SLIDING_SLACK_ROWS, Math.floor(slots / TOP_K));
/** Tokens ⑦b generates per golden prompt. */
const CACHE_NEW_TOKENS = 8;
/**
 * A run of ⑦: the slots, and the host tier's budget (absent = no tier; `all` = every expert, as
 * `Number.MAX_SAFE_INTEGER`; `quarter` = a quarter of every expert's bytes, read off the `all` run's
 * tier right after its construction, so the `all` run comes first).
 */
type CacheRun = { readonly slots: number; readonly host?: "all" | "quarter" };
/**
 * ⑦a's runs, at slot counts small enough that runs miss and re-run (ADR 0146 decision 7). The host-tier runs repeat a run without a tier at the same slots, so their misses and
 * loads must be the same (the tier changes where a load's bytes come from, not which experts).
 */
const SESSION_CACHE_RUNS: readonly CacheRun[] = [
  { slots: 16 },
  { slots: 64 },
  { slots: 16, host: "all" },
  { slots: 64, host: "quarter" },
];
/** ⑦b's runs (likewise). */
const PIPELINE_CACHE_RUNS: readonly CacheRun[] = [
  { slots: 16 },
  { slots: 64 },
  { slots: 64, host: "all" },
];
/** A run's name in the log and the failures. */
const cacheRunName = ({ slots, host }: CacheRun): string =>
  `slots ${slots}${host === undefined ? "" : ` host ${host}`}`;

/** The number of 32-bit words in which two arrays differ (0 = the same bytes). */
const differingWords = (actual: Float32Array<ArrayBuffer>, expected: Float32Array<ArrayBuffer>) => {
  assertEquals(actual.length, expected.length, "row lengths");
  const a = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const e = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  let differing = 0;
  for (let at = 0; at < a.length; at += 1) if (a[at] !== e[at]) differing += 1;
  return differing;
};

/** One line of the cache's counters (the runtime's merged with the re-run face's). */
const reportLine = (report: ExpertCacheReport): string =>
  `runs ${report.runs}, missedRuns ${report.missedRuns}, loads ${report.loads}, loadedExperts ` +
  `${report.loadedExperts}, evictedExperts ${report.evictedExperts}, bytesRead ${report.bytesRead}, ` +
  `loadMs ${report.loadMs.toFixed(1)}, committedRuns ${report.committedRuns}, retries ${report.retries}` +
  `, hostExperts ${report.hostExperts}, containerExperts ${report.containerExperts}, bytesUploaded ` +
  `${report.bytesUploaded}, readMs ${report.readMs.toFixed(1)}, uploadMs ${report.uploadMs.toFixed(1)}` +
  `, hostHeldExperts ${report.hostHeldExperts}, hostHeldBytes ${report.hostHeldBytes}`;

/** The counters a host tier must leave as they are: which experts missed, loaded and were evicted. */
const residencyCounters = (report: ExpertCacheReport) => ({
  runs: report.runs,
  missedRuns: report.missedRuns,
  loads: report.loads,
  loadedExperts: report.loadedExperts,
  evictedExperts: report.evictedExperts,
  committedRuns: report.committedRuns,
  retries: report.retries,
});

/**
 * The host tier's counters against its budget (`entryBytes` = one expert's bytes, every layer's
 * the same on 26B): every loaded expert came from RAM or the container and was uploaded whole, the
 * container's bytes are its experts', and the tier holds at most its budget. With every expert held
 * no load reads the container; with a part of them, RAM hits and container reads both happen, and the
 * tier evicted (each container read is put into it, so it held more than it holds now).
 */
const hostTierFailures = (
  where: string,
  run: CacheRun,
  report: ExpertCacheReport,
  entryBytes: number,
  hostBytes: number,
  heldAtBuild: number,
): string[] => {
  const failures: string[] = [];
  const fail = (what: string): void => {
    failures.push(`${where}: ${what}`);
  };
  if (report.hostExperts + report.containerExperts !== report.loadedExperts) {
    fail(
      `${report.hostExperts} from RAM + ${report.containerExperts} from the container ≠ ${report.loadedExperts} loaded`,
    );
  }
  if (report.bytesUploaded !== report.loadedExperts * entryBytes) {
    fail(`${report.bytesUploaded} bytes uploaded for ${report.loadedExperts} experts`);
  }
  if (report.bytesRead !== report.containerExperts * entryBytes) {
    fail(`${report.bytesRead} bytes read for ${report.containerExperts} container experts`);
  }
  if (report.hostHeldBytes > hostBytes) {
    fail(`the tier holds ${report.hostHeldBytes} bytes, above its budget ${hostBytes}`);
  }
  if (report.hostExperts === 0) fail("no expert came from RAM — the case decides nothing");
  if (run.host === "all") {
    if (report.containerExperts !== 0 || report.bytesRead !== 0) {
      fail(
        `${report.containerExperts} experts (${report.bytesRead} bytes) read from the container with every expert held`,
      );
    }
  } else {
    if (report.containerExperts === 0) {
      fail("no expert read from the container — the budget held every miss");
    }
    if (heldAtBuild + report.containerExperts <= report.hostHeldExperts) {
      fail(
        `no RAM eviction (${heldAtBuild} held at build + ${report.containerExperts} put, ${report.hostHeldExperts} held now)`,
      );
    }
  }
  return failures;
};

describe.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  `Gemma 4 26B cut, ⑦ the exact expert cache (real GPU)${SERIES_CHECKS_OPT_IN}`,
  () => {
    let gpu: GpuContext | undefined;
    const goldens = new Map<string, CutGolden>();

    beforeAll(async () => {
      const layers = cutLayers();
      for (const name of CASES) goldens.set(name, await readGolden(name, layers));
      gpu = await acquireGpu();
    });
    afterAll(() => {
      gpu?.destroy();
    });

    /** The logits and hidden rows of one golden prompt fed through `session` in `chunkLength` rows. */
    const goldenRows = async (
      session: GenerationSession,
      golden: CutGolden,
      chunkLength: number,
      outputs: readonly string[],
      where: string,
    ): Promise<{ readonly rows: ProductRows; readonly chunks: number }> => {
      const [logitsName, hiddenName] = outputs;
      const logits = new Float32Array(golden.rows.length * VOCAB);
      const hidden = new Float32Array(golden.rows.length * HIDDEN);
      let chunks = 0;
      await teacherForce(
        session,
        golden.prompt,
        golden.rows,
        (run) => {
          chunks += 1;
          if (run.selected.length === 0) return;
          placeRows(run.outputs[logitsName], VOCAB, run, golden.rows, logits, `${where} logits`);
          placeRows(run.outputs[hiddenName], HIDDEN, run, golden.rows, hidden, `${where} hidden`);
        },
        chunkLength,
      );
      return { rows: { logits, hidden }, chunks };
    };

    it("⑦a Session: under the cache at 16 and 64 slots, with and without a host tier, the golden rows are the bytes of a resident Session at the same chunk rows, with misses and loads", async () => {
      assert(gpu !== undefined, "no GPU");
      const model = prepareContainer(await openSeriesContainer(MODEL), MODEL_GRAPH);
      const outputs = model.graph.outputs;
      const key = (chunkLength: number, name: string): string => `${chunkLength}/${name}`;
      // The resident references first, all from one Session (one large container on the GPU at a
      // time); the cache's chunk rows change the numbers (other shapes), so each has its own.
      const resident = new Map<string, ProductRows>();
      const session = await model.createContainerSession(gpu);
      try {
        for (const chunkLength of new Set(
          SESSION_CACHE_RUNS.map(({ slots }) => cacheChunkRows(slots)),
        )) {
          for (const [name, golden] of goldens) {
            const { rows } = await goldenRows(session, golden, chunkLength, outputs, name);
            resident.set(key(chunkLength, name), rows);
          }
        }
      } finally {
        await session.dispose();
        await settleReleases(gpu);
      }

      const lines: string[] = [];
      const failures: string[] = [];
      /** Per slots, the residency counters of the run without a tier (the host-tier runs repeat them). */
      const untiered = new Map<number, ReturnType<typeof residencyCounters>>();
      /** One expert's bytes and every expert's, read off the `all` run's tier after construction. */
      let everyExpert: { readonly entryBytes: number; readonly totalBytes: number } | undefined;
      for (const cacheRun of SESSION_CACHE_RUNS) {
        const { slots, host } = cacheRun;
        const where = cacheRunName(cacheRun);
        const chunkLength = cacheChunkRows(slots);
        let hostBytes: number | undefined;
        if (host === "all") hostBytes = Number.MAX_SAFE_INTEGER;
        if (host === "quarter") {
          assert(everyExpert !== undefined, `${where}: the 'all' run must come first`);
          hostBytes = Math.floor(everyExpert.totalBytes / 4);
        }
        const started = performance.now();
        const cached = await model.createContainerSession(gpu, {
          expertCache: { slots, ...(hostBytes === undefined ? {} : { hostBytes }) },
        });
        try {
          const built = cached.expertCache?.stats();
          assert(built !== undefined, `${where}: no expert cache`);
          const heldAtBuild = built.hostHeldExperts;
          if (host === "all") {
            // Every routed expert of every layer, filled from the bytes the build read.
            const experts = cutLayers() * EXPERTS;
            assertEquals(heldAtBuild, experts, `${where}: experts held after construction`);
            assertEquals(built.hostHeldBytes % experts, 0, `${where}: one expert's bytes`);
            everyExpert = {
              entryBytes: built.hostHeldBytes / experts,
              totalBytes: built.hostHeldBytes,
            };
          }
          const face = createExpertCacheSession(cached);
          let chunks = 0;
          for (const [name, golden] of goldens) {
            const run = await goldenRows(face, golden, chunkLength, outputs, name);
            chunks += run.chunks;
            const reference = resident.get(key(chunkLength, name));
            assert(reference !== undefined, `${name}: no resident reference`);
            for (const output of OUTPUTS) {
              const differing = differingWords(run.rows[output], reference[output]);
              if (differing > 0) {
                failures.push(
                  `${where} ${name} ${output}: ${differing} of ${reference[output].length} words differ from the resident Session's`,
                );
              }
            }
          }
          const report = face.report();
          lines.push(
            `${where} (chunks of ${chunkLength} rows, ${chunks} chunks, ${((performance.now() - started) / 1000).toFixed(1)} s${
              hostBytes === undefined
                ? ""
                : `, hostBytes ${hostBytes}, ${heldAtBuild} held at build`
            }): ${reportLine(report)}`,
          );
          // A run with no miss would prove nothing.
          if (report.missedRuns === 0 || report.loads === 0) {
            failures.push(`${where}: no missed run or no load — the case decides nothing`);
          }
          if (report.committedRuns !== chunks || report.retries !== report.missedRuns) {
            failures.push(
              `${where}: ${report.committedRuns} committed runs for ${chunks} chunks, ${report.retries} retries for ${report.missedRuns} missed runs`,
            );
          }
          if (hostBytes === undefined) {
            untiered.set(slots, residencyCounters(report));
          } else {
            assert(everyExpert !== undefined, `${where}: no expert size`);
            failures.push(
              ...hostTierFailures(
                where,
                cacheRun,
                report,
                everyExpert.entryBytes,
                hostBytes,
                heldAtBuild,
              ),
            );
            const expected = untiered.get(slots);
            assert(expected !== undefined, `${where}: no run without a tier at ${slots} slots`);
            if (JSON.stringify(residencyCounters(report)) !== JSON.stringify(expected)) {
              failures.push(
                `${where}: residency ${JSON.stringify(residencyCounters(report))}, without a tier ${JSON.stringify(expected)}`,
              );
            }
          }
        } finally {
          await cached.dispose();
          await settleReleases(gpu);
        }
      }
      console.log(`[e2e] gemma4 26b cut expert cache (Session):\n  ${lines.join("\n  ")}`);
      assertEquals(failures, [], "the expert cache against the resident Session");
    });

    it("⑦b gemma.Pipeline: with the cache at 16 and 64 slots, and with every expert in the host tier at 64, the greedy tokens are those without it at the same chunk rows, with re-runs", async () => {
      assert(gpu !== undefined, "no GPU");
      const assets = {
        config: PIPELINE_CONFIG,
        model: await Promise.all(resolveParts(MODEL).map((part) => readFile(part))),
        tokenizer: await readFile(TOKENIZER),
      };
      /** Each golden prompt's greedy tokens (a stop token included) and stop reason. */
      const generate = async (pipeline: Pipeline): Promise<Map<string, string>> => {
        const generated = new Map<string, string>();
        for (const [name, golden] of goldens) {
          const sequence = await pipeline.sequence();
          try {
            const ids: number[] = [];
            const stream = sequence.generate({
              prompt: [...golden.prompt],
              maxNewTokens: CACHE_NEW_TOKENS,
              sampler: { temperature: 0 },
            });
            for await (const event of stream) {
              if (event.kind === "token") ids.push(event.id);
            }
            const stop = await stream.done;
            if (stop.reason === "eos") ids.push(stop.token);
            generated.set(name, `[${ids.join(", ")}] (stop '${stop.reason}')`);
          } finally {
            await sequence.dispose();
          }
        }
        return generated;
      };

      const lines: string[] = [];
      const failures: string[] = [];
      /** Per chunk rows, the tokens without the cache (each computed once). */
      const withoutCache = new Map<number, Map<string, string>>();
      const untiered = new Map<number, ReturnType<typeof residencyCounters>>();
      for (const cacheRun of PIPELINE_CACHE_RUNS) {
        const { slots, host } = cacheRun;
        const where = cacheRunName(cacheRun);
        const chunkLength = cacheChunkRows(slots);
        let expected = withoutCache.get(chunkLength);
        if (expected === undefined) {
          // Without the cache at the chunk rows the cache settles (the same shapes, so the same
          // numbers); one large container on the GPU at a time.
          const resident = await Pipeline.fromAssets(assets, { gpu, chunkLength });
          try {
            expected = await generate(resident);
          } finally {
            await resident.dispose();
            await settleReleases(gpu);
          }
          withoutCache.set(chunkLength, expected);
        }
        const started = performance.now();
        const cached = await Pipeline.fromAssets(assets, {
          gpu,
          expertCache: {
            slots,
            ...(host === "all" ? { hostBytes: Number.MAX_SAFE_INTEGER } : {}),
          },
        });
        try {
          assertEquals(cached.program.chunkLength, chunkLength, `${where}: the chunk rows`);
          const actual = await generate(cached);
          const report = cached.expertCacheReport();
          assert(report !== undefined, "a pipeline with the cache has no report");
          lines.push(
            `${where} (chunks of ${chunkLength} rows, ${((performance.now() - started) / 1000).toFixed(1)} s): ${reportLine(report)}`,
          );
          for (const [name, tokens] of expected) {
            lines.push(`  ${name}: ${tokens}`);
            if (actual.get(name) !== tokens) {
              failures.push(
                `${where} ${name}: ${actual.get(name)} with the cache, ${tokens} without`,
              );
            }
          }
          if (report.retries === 0) {
            failures.push(`${where}: no re-run — the case decides nothing`);
          }
          if (host === undefined) {
            untiered.set(slots, residencyCounters(report));
          } else {
            // Every expert held: each load came from RAM, none from the distribution.
            if (
              report.hostExperts === 0 ||
              report.hostExperts !== report.loadedExperts ||
              report.containerExperts !== 0 ||
              report.bytesRead !== 0
            ) {
              failures.push(
                `${where}: ${report.hostExperts} of ${report.loadedExperts} loads from RAM, ${report.containerExperts} (${report.bytesRead} bytes) from the distribution`,
              );
            }
            const residency = JSON.stringify(residencyCounters(report));
            const expectedResidency = JSON.stringify(untiered.get(slots));
            if (residency !== expectedResidency) {
              failures.push(
                `${where}: residency ${residency}, without a tier ${expectedResidency}`,
              );
            }
          }
        } finally {
          await cached.dispose();
          await settleReleases(gpu);
        }
      }
      console.log(`[e2e] gemma4 26b cut expert cache (Pipeline):\n  ${lines.join("\n  ")}`);
      assertEquals(failures, [], "the expert cache against the pipeline without it");
    });
  },
);

const RUNNABLE = AVAILABLE && GPU_AVAILABLE;
if (RUNNABLE) references.warnMissing(REFERENCE_IDS);
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: REFERENCE_IDS });
