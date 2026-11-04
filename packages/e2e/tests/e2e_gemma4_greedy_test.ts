// Real-GPU acceptance check of the real-weight Gemma 4 E2B (states-form decode graph) — **parity of
// fixed token id sequences**.
//
// `packages/runtime/tests/e2e_gemma4_test.ts` covers the numerics of MQA and mixed storage in the
// 1-shot form (a single pass, like prefill); this file covers that **the whole generation loop**
// (the prefill-chunk and decode forms of ADR 0066 decision 4 + the GenerationContext's KV slots)
// **emits the same tokens in the same order as torch** on the real weights. The acceptance form is
// **discrete agreement**, not a tolerance — the same philosophy as the `.lab` / PNG sha256 checks
// (pin the exit artifact itself — however much the numbers move, green while the sequence does not
// split, red when it does).
//
// The model it mirrors is `e2e_minicpm5_greedy_test.ts` (no KV sharing, full attention only,
// uniform f32 storage); what this one adds are three points specific to Gemma 4:
//
// - **Two kinds, sliding + full** (the 28 sliding layers are ring slots with a 512 window, the 7
//   full layers the whole context). The longest case T=598 exceeds the window, so the ring eviction
//   and the window predicate are actually evaluated on the real weights.
// - **KV sharing** (the last 20 of the 35 layers have no slots of their own and read the slots of
//   layer 13 (sliding) / layer 14 (full) — only the 30 slots of the owning layers exist).
// - **Mixed storage** (embedding i8 × linear i4 — ADR 0069 decision 5) holds on the decode path
//   too.
//
// Five checks:
//
// ① Form premises (5 inputs = input_ids + 4 derived RoPE inputs / 2 outputs / output 1 comes from
//    argmax / 30 states slots and the sharing rule / the window per layer kind / symbols M, C)
// ② **greedy parity** (3 cases × K=16 steps exactly match torch's expected sequences) — the claim
//    of this check
// ③ Logits tolerance of the last prefill chunk (assembled by hand, not through `generateGreedy`; a
//    diagnostic line)
// ④ census (the states-form kernel families run in both the sliding and the full variants, only the
//    q side carries `:gqa`, and linear / embedding run on compressed storage)
// ⑤ Plan properties of the decode path (PreparedPlan settles on two plans, the prefill form and the
//    decode form)
//
// The host-side generation loop (`src/generation/greedy.ts`) is the models implementation, so this
// check lives on the models side (runtime tests do not create the reverse dependency of a relative
// import of the models implementation).
//
// ## Assets
//
// `outputs/series/gemma4-e2b-decode/` (about 4 GiB even with mixed storage, so outside the
// repository — `outputs/` in `.gitignore`). The generation script is
// `tools/export-recipes/gemma4/export_decode.py`; the {@link GENERATE_COMMAND} below is the source
// of truth for the command.
//
// - `io.<case>.safetensors` — inputs and outputs of **one unpadded full-length pass** (3 cases).
//   It is not a reference from running the chunk-form graph once over the full length but a table
//   of "the values torch produces with the same weights at the same positions"; this side splits
//   into chunks and compares **only the valid rows** (only the states-form attention output is
//   pinned to 0 on pad rows — ADR 0066 addendum 8. The later MLP / lm_head write meaningless values
//   on pad rows too, so the graph output of pad rows **must not be read** — the reference has no
//   counterpart either).
// - `greedy.<case>.safetensors` — the source of truth of the decode acceptance (3 cases). `prompt`
//   i32 `[T]` / `expected` i32 `[K]` / `margin` f32 `[K]`. The expected sequences were taken with a
//   **full-length full re-forward** (not through the KV cache path — the MUST of
//   `export_decode.greedy_continuation`), so the mechanism under acceptance and the one that makes
//   the expectations are independent.
//
// The decode graph (PLE and RoPE inside, argmax exit) is not shipped by any distribution, so every
// case here is an implementation-time series check (ADR 0139 decision 6): it runs only with
// KARUME_SERIES=1 and SKIPs explicitly otherwise. The shipped product graph's generation loop and
// census are covered by e2e_gemma4_sequence_test.ts / e2e_gemma4_quant_test.ts.
//
// Where the asset is missing the cases **SKIP explicitly**. ADR 0005's "an all-SKIP run is an
// explicit FAIL" guard looks only at *whether a GPU adapter exists* and is independent of this
// SKIP. An asset that exists **only partly** (missing cases) FAILs rather than SKIPs (the "asset
// completeness" test below) — there a SKIP would be a silent apparent success.

import { type DirEntry, isNotFound, readDirSync, readFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  codecLayout,
  type GenerationContext,
  type GpuContext,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
  type RunOutputs,
  type SafetensorsFile,
  type Session,
  type SessionDiagnostics,
  type Tensor,
} from "@karume/runtime";
import { generateGreedy } from "../../models/src/generation/greedy.ts";
import { ropeInputs, type RopeSpec } from "../../models/src/gemma/rope.ts";
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../models/tests/helpers/series-graphs.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";
import { requireCensus } from "../../runtime/tests/helpers/pipeline-census.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

const SERIES_NAME = "gemma4-e2b-decode";
const SERIES_ROOT = new URL(`../../../outputs/series/${SERIES_NAME}/`, import.meta.url);
const MODEL_FILE = "model.krm";
/**
 * The graph name inside the container (the one table in helpers/series-graphs.ts — looked up from
 * the same source of truth as the guard).
 */
const MODEL_GRAPH = seriesGraph(SERIES_NAME);
const IO_PREFIX = "io.";
const GREEDY_PREFIX = "greedy.";
const SUFFIX = ".safetensors";

/**
 * The generate command, ready to paste when the cases SKIP (the same spelling as
 * tools/export-recipes/gemma4/README.md).
 */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' python -m gemma4.export_decode";

/**
 * The cases that should have been generated. **Pinned here rather than taken from the listing** —
 * relying on the listing alone, an environment that ran only part of the generation would lose
 * tests silently and end up "green but unchecked". The source of truth is `GOLDEN_CASES` in
 * `gemma4/export.py` (io, T = 6 / 10 / 598) and `GREEDY_CASES` in `export_decode.py`.
 *
 * NOTE: on the greedy side the script too **provisionally adopts every case** (the real run settles
 * which cases pass the margin check). If a real run drops a case, fix the script's `GREEDY_CASES`
 * and this list together — do not fall back to the listing to follow automatically (a gap would go
 * green silently).
 */
const EXPECTED_IO_CASES = ["capital-en", "capital-ja", "context-en"] as const;
const EXPECTED_GREEDY_CASES = ["capital-en", "capital-ja", "context-en"] as const;

/** The step count of the greedy golden (source of truth: `export_decode.GREEDY_STEPS`). */
const GREEDY_STEPS = 16;

/**
 * The row count of the fixed-length prefill chunk (the plan-time constant of ADR 0066 decision 4).
 *
 * **32 is the value that "always runs a multi-chunk prefill"**. The two short cases (T = 6 / 10)
 * fit in one chunk, and `context-en` (T=598) splits into 19 chunks, where the second and later are
 * prefills with `pastLength > 0` (`M > 1` with a past — a path distinct both from decode's `M = 1`
 * and from the first chunk's `P = 0`). With a value at which every case fits in one chunk, the
 * states-form attention's "two-sided predicate over the past + the current chunk" would never run
 * on the real weights.
 *
 * Only `context-en` crosses the 512 window too: chunk 17 onward (absolute positions 512 and up)
 * actually runs the sliding slots' ring wrap-around (`col % window`) and eviction.
 *
 * NOTE: the distribution's default is 768 (ADR 0091 decision 3), but this check stays at 32 — at
 * 768 all three cases fit in one chunk, and the two paths above (prefill with a past and the ring
 * wrap-around) never run. The expected sequences were taken with torch's full-length full
 * re-forward and do not depend on the chunk length, so the values do not move (the chunk length
 * can change how f32 partial sums fold, but ②'s margin check covers that width — the "changing
 * chunkLength can move the token sequence" section of limitations).
 */
const CHUNK_LENGTH = 32;

/**
 * The capacity symbol of the KV slots (`export_decode.CAPACITY_SYMBOL`) and the value this check
 * binds.
 *
 * 640 covers `T + K` of every case (the longest is context-en's 598 + 16 = 614). Whether it covers
 * them is measured per case (the {@link CAPACITY} check below) — this is where a re-export with a
 * longer prompt that runs out of capacity fails instead of being silently truncated.
 *
 * NOTE: only the 6 full slots carry the symbol C — the 24 sliding slots have the actual `window`
 * (512) baked in as their capacity and are not affected by this binding (ADR 0066 addendum 9).
 */
const CAPACITY_SYMBOL = "C";
const CAPACITY = 640;

/**
 * The exclusive upper bound of absolute positions that can be looked up = **the position limit the
 * model declares** (upstream `max_position_embeddings`).
 *
 * The RoPE tables are no longer baked into the graph (ADR 0091 decision 1 — cos / sin are derived
 * inputs the host builds per chunk), so the limit is the model's declaration, not the table's row
 * count. Attention at an untrained position does not throw but folds into "a plausible token id",
 * so `generateGreedy` requires this value and fails at the entry.
 */
const MAX_POSITION = 131_072;

/** The graph input name (source of truth: `export_decode.INPUT_IDS`). */
const INPUT_IDS = "input_ids";

/**
 * The names and order of the 4 derived RoPE inputs (source of truth: `export_decode.ROPE_INPUTS` =
 * layer kind × cos / sin).
 *
 * MUST: not derived from the TS-side generator (`gemma4RopeInputName`). This check looks at
 * **whether the asset's graph inputs follow the exporter's spelling**; building them from the TS
 * spelling would go green even if both sides drifted together.
 */
const ROPE_INPUTS = [
  "rope_sliding_attention_cos",
  "rope_sliding_attention_sin",
  "rope_full_attention_cos",
  "rope_full_attention_sin",
] as const;

/** The layer count (`num_hidden_layers` in the config). */
const LAYERS = 35;

/**
 * The layer where KV sharing starts (`num_hidden_layers − num_kv_shared_layers` = 35 − 20). Only
 * layers 0..14 have their own k / v slots; layers 15..34 read the owning layers' slots.
 */
const OWNED_LAYERS = 15;
/** The number of state slots (k / v for each owning layer). */
const SLOTS = OWNED_LAYERS * 2;

/**
 * The full-attention layer indices (the rest are sliding). Source of truth: `config.layer_types`.
 * The layer kind decides three things: head_dim (sliding 256 / full 512), whether there is a
 * window, and which slots are shared.
 */
const FULL_LAYERS: ReadonlySet<number> = new Set([4, 9, 14, 19, 24, 29, 34]);

/**
 * The KV owning layer per layer kind (`export_decode.kv_owner_layers` — the last layer of the same
 * kind before sharing starts).
 */
const SLIDING_OWNER = 13;
const FULL_OWNER = 14;

/** head_dim per layer kind (sliding = `head_dim` / full = `global_head_dim`). */
const SLIDING_HEAD_DIM = 256;
const FULL_HEAD_DIM = 512;

/**
 * This series' RoPE parameters (two layer kinds — the declaration of the formula the host uses to
 * build cos / sin).
 *
 * A distribution declares them in `pipelineConfig.rope` of `karume.json`, but the decode series is
 * a bare IR for acceptance and has no `karume.json` (the script only derives them from the config
 * for the example inputs — `rope_specs` in `gemma4/rope.py`). So they are constants here. The
 * values come from the E2B config and equal the distribution's declaration
 * (`pipelineConfig.rope` in `models/karume-gemma4/karume.json`):
 * sliding = `rope_local_base_freq` 10,000 / `head_dim` 256 (rotates every dimension),
 * full = `rope_theta` 1,000,000 / `global_head_dim` 512 / `partial_rotary_factor` 0.25 → 128.
 *
 * If the width (`headDim`) disagrees with the layer kind's head_dim, the run fails on the shape
 * rather than ①'s D-axis check.
 */
const ROPE_SPEC: RopeSpec = {
  sliding_attention: { theta: 10_000, headDim: SLIDING_HEAD_DIM, rotaryDim: SLIDING_HEAD_DIM },
  full_attention: { theta: 1_000_000, headDim: FULL_HEAD_DIM, rotaryDim: 128 },
};

/**
 * The window width of the sliding layers (`sliding_window` in the config — declared as is in the
 * `window` attr).
 */
const WINDOW = 512;

/**
 * The head counts on the q side / the k·v side (8:1 = true MQA; the basis of the census's `:gqa`).
 */
const HEADS = 8;
const KV_HEADS = 1;

/**
 * Layer → the owning layer of the KV slots that layer reads (the same rule as
 * `export_decode.slot_layers`).
 */
const slotOwner = (layer: number): number =>
  layer < OWNED_LAYERS ? layer : FULL_LAYERS.has(layer) ? FULL_OWNER : SLIDING_OWNER;

/** The slot name (spelling source of truth: `export_decode.slot_name`). */
const slotName = (layer: number, part: "k" | "v"): string => `l${layer}.${part}`;

/**
 * The tolerance for comparing the prefill logits (`[1,M,262144]`) against the torch CPU
 * expectations.
 *
 * **The same threshold** as the 1-shot check (`GEMMA4_TOLERANCE` in `e2e_gemma4_test.ts`) — what
 * is compared is the same (the logits of the same weights at the same positions), and only the path
 * differs (a single pass vs chunks + KV slots), so a different threshold would make "is it the path
 * or the threshold?" unreadable. The measured worst case on the 1-shot side is 2.232e-3 over all
 * rows of context-en (atol is about 4.5 times that).
 *
 * Measured (first real run, 2026-08-19) — the error the chunk + KV slot path adds is of the same
 * order as the 1-shot one, with no gap of orders of magnitude that would point to a broken ring
 * wrap-around or position mapping:
 *
 * | Target                    | Valid rows | maxAbs   | Same case in the 1-shot check (all rows) |
 * | ------------------------- | ---------- | -------- | ---------------------------------------- |
 * | last chunk of context-en  | 22         | 1.092e-3 | 2.232e-3                                 |
 *
 * **rtol is 0** (the verdict is atol alone — the reason is the same as in the 1-shot docstring).
 */
const PREFILL_ATOL = 1e-2;

/**
 * Lists the directory. It falls back to empty only when the directory does not exist.
 * MUST: anything but NotFound propagates — reading a permission error or the like as "no asset"
 * would turn a check that never ran into a quiet green SKIP.
 */
const listDir = (url: URL): readonly DirEntry[] => {
  try {
    return [...readDirSync(url)];
  } catch (cause) {
    if (isNotFound(cause)) {
      return [];
    }
    throw cause;
  }
};

const discoverCases = (prefix: string): readonly string[] =>
  listDir(SERIES_ROOT)
    .filter((entry) => entry.isFile && entry.name.startsWith(prefix) && entry.name.endsWith(SUFFIX))
    .map((entry) => entry.name.slice(prefix.length, entry.name.length - SUFFIX.length))
    .sort();

/**
 * Listed synchronously because it is needed at registration (the same reason as `Deno.test`'s
 * ignore decision). Without KARUME_SERIES=1 the series is not read at all.
 */
const IO_CASES = SERIES_CHECKS ? discoverCases(IO_PREFIX) : [];
const GREEDY_CASES = SERIES_CHECKS ? discoverCases(GREEDY_PREFIX) : [];

/**
 * Whether the asset exists. None at all = an environment that has not generated it, so every case
 * SKIPs (a partial gap goes to the FAIL side).
 */
const AVAILABLE = IO_CASES.length > 0 || GREEDY_CASES.length > 0;
/**
 * Whether **anything at all** is left (the SKIP predicate of the completeness test — Codex round H
 * finding H-02). A gap where every golden is gone and only the model remains makes `AVAILABLE`
 * false, and the completeness test itself would SKIP — the predicate that turns a gap into a FAIL
 * may stay dormant only when the directory is "completely empty".
 */
const ANY_PRESENT = SERIES_CHECKS && (AVAILABLE || modelPresent(new URL(MODEL_FILE, SERIES_ROOT)));

if (SERIES_CHECKS && !AVAILABLE) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} has no exported asset, so the Gemma 4 E2B decode acceptance is ` +
      `SKIPPED (about 4 GiB even with mixed storage, kept outside the repository). Generate: ${GENERATE_COMMAND}`,
  );
}

/**
 * Reads one file as an `ArrayBuffer`.
 *
 * MUST: do not slice when the view covers the whole buffer — io.context-en is 627 MB, and an
 * unconditional `slice` would double the peak (the same reason as the 1-shot check).
 */
const readBuffer = async (file: string): Promise<ArrayBuffer> => {
  const bytes = await readFile(new URL(file, SERIES_ROOT));
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return bytes.buffer;
  }
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

/**
 * Takes one tensor of the golden as a view of the declared dtype (no copy).
 *
 * MUST: fail when the storage dtype disagrees — f32 and i32 elements are both 4 bytes, so reading
 * one as the other silently would let a reinterpretation of the bits "pass". The runtime tests'
 * helper of the same name (`helpers/golden-io.ts`) is not imported (a package's tests do not depend
 * on another package's test internals — the same split as the NOTE in `helpers/gpu.ts`).
 */
const goldenF32 = (file: SafetensorsFile, name: string): Float32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `the golden has no '${name}'`);
  assertEquals(view.dtype, "F32", `storage dtype of golden '${name}'`);
  return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

const goldenI32 = (file: SafetensorsFile, name: string): Int32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `the golden has no '${name}'`);
  assertEquals(view.dtype, "I32", `storage dtype of golden '${name}'`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

/** `greedy.<case>.safetensors` (the source of truth of the decode acceptance). */
type GreedyGolden = {
  readonly prompt: Int32Array<ArrayBuffer>;
  readonly expected: Int32Array<ArrayBuffer>;
  readonly margin: Float32Array<ArrayBuffer>;
};

const loadGreedy = async (caseName: string): Promise<GreedyGolden> => {
  const file = parseSafetensors(await readBuffer(`${GREEDY_PREFIX}${caseName}${SUFFIX}`));
  assertEquals(
    [...file.tensors.keys()].sort(),
    ["expected", "margin", "prompt"],
    `tensor keys of ${GREEDY_PREFIX}${caseName}${SUFFIX}`,
  );
  return {
    prompt: goldenI32(file, "prompt"),
    expected: goldenI32(file, "expected"),
    margin: goldenF32(file, "margin"),
  };
};

/**
 * `io.<case>.safetensors` (inputs and outputs of one unpadded full-length pass).
 *
 * The 4 derived RoPE inputs are recorded too, but this side **does not read them** — the Deno side
 * rebuilds them with `ropeInputs` from the same positions (feeding the tables the script built as
 * they are would keep ③ green even if the TS formula broke). The comparison of the tables
 * themselves belongs to `gemma4_rope_test.ts` (parity with the upstream module's actual output).
 */
type IoGolden = {
  readonly ids: Int32Array<ArrayBuffer>;
  readonly logits: Float32Array<ArrayBuffer>;
  readonly tokens: Int32Array<ArrayBuffer>;
};

const loadIo = async (caseName: string): Promise<IoGolden> => {
  const file = parseSafetensors(await readBuffer(`${IO_PREFIX}${caseName}${SUFFIX}`));
  assertEquals(
    [...file.tensors.keys()].sort(),
    [
      `input.${INPUT_IDS}`,
      ...ROPE_INPUTS.map((name) => `input.${name}`),
      "output.0",
      "output.1",
    ].sort(),
    `tensor keys of ${IO_PREFIX}${caseName}${SUFFIX}`,
  );
  return {
    ids: goldenI32(file, `input.${INPUT_IDS}`),
    logits: goldenF32(file, "output.0"),
    tokens: goldenI32(file, "output.1"),
  };
};

/**
 * Form premises of the states-form decode graph (checks only the properties that **break silently
 * while the numbers still match**).
 *
 * The full form check belongs to the writing side (`export_decode.assert_ir_form_decode`). The
 * reading side needs only to confirm "the two forms this check runs are a graph with the expected
 * entry, exit and slots"; if this loosened, **a different graph would be accepted** (e.g. the old
 * form with a mask input left in becomes chunk-local causal, and the numbers match only for the
 * cases whose prompt fits in one chunk).
 *
 * What Gemma 4 adds are KV sharing and layer kinds:
 *
 * - There are **30 slots, for the owning layers**, and layers 15..34 point at the names of the
 *   owning layers (sliding→13 / full→14). Mixing up the assignment of even one layer would let a
 *   form that "reads another layer's KV" pass **with matching shapes and types**.
 * - If the presence of `window` disagrees with the layer kind, a full layer drops what is outside
 *   the window / a sliding layer ignores the window.
 * - The order of the D axis (256 / 512) also proves **node order = layer order** (if the order
 *   broke, the premise of the slot check above would break too).
 */
const assertDecodeForm = (parsed: PreparedModel): void => {
  const graph = parsed.graph;
  // The inputs are the token ids + the 4 derived RoPE inputs (ADR 0091 decision 1 —
  // `position_ids` is gone).
  assertEquals(
    graph.inputs.map((spec) => spec.name),
    [INPUT_IDS, ...ROPE_INPUTS],
    "graph inputs (position_ids left in means an old-generation asset with the tables baked in)",
  );
  // The width of a derived input is the layer kind's head_dim itself (the row count is the
  // physical chunk symbol M). If this disagreed, "a table of another width with a matching row
  // count" could be passed, and the declaration would pass without fitting the attention's D axis.
  for (const spec of graph.inputs.filter((input) => input.name !== INPUT_IDS)) {
    assertEquals(spec.dtype, "f32", `dtype of the derived RoPE input '${spec.name}'`);
    assertEquals(
      spec.shape,
      [1, "M", spec.name.includes("full") ? FULL_HEAD_DIM : SLIDING_HEAD_DIM],
      `shape of the derived RoPE input '${spec.name}'`,
    );
  }
  assertEquals(graph.outputs.length, 2, "number of graph.outputs (logits / token)");

  // Output 1 comes from argmax (the decode exit of ADR 0068 decision 4). `generateGreedy` feeds
  // this output straight into the next step, so if it came from another op (e.g. the 1-shot form's
  // logits), "a token sequence read from another tensor of a matching shape" would be built
  // without an exception.
  const producer = new Map<string, string>();
  for (const node of graph.nodes) {
    for (const out of node.outs) producer.set(out, node.op);
  }
  assertEquals(
    producer.get(graph.outputs[1]),
    "argmax",
    `producer of output 1 ('${graph.outputs[1]}')`,
  );

  // states are k / v for each owning layer (shared layers have no slots of their own).
  const expectedSlots = Array.from({ length: OWNED_LAYERS }, (_, layer) => layer)
    .flatMap((layer) => [slotName(layer, "k"), slotName(layer, "v")])
    .sort();
  assertEquals(Object.keys(graph.states).sort(), expectedSlots, "names of the states slots");
  assertEquals(Object.keys(graph.states).length, SLOTS, "number of states slots");

  // The slot capacity splits by layer kind: sliding owning layers = the actual window (the ring
  // closes exactly at the window) / full owning layers = the symbol C (chosen at run time). The
  // full check is on the writing side (`export_decode.assert_ir_form_decode`) — this only confirms
  // the asset read has that form.
  for (let layer = 0; layer < OWNED_LAYERS; layer += 1) {
    const capacity = FULL_LAYERS.has(layer) ? CAPACITY_SYMBOL : WINDOW;
    for (const part of ["k", "v"] as const) {
      const name = slotName(layer, part);
      assertEquals(
        graph.states[name].shape[2],
        capacity,
        `capacity of states['${name}'] (sliding = the actual window / full = the symbol)`,
      );
    }
  }

  const attentions = graph.nodes.filter((node) => node.op === "attention");
  assertEquals(attentions.length, LAYERS, "number of attention nodes (= layer count)");
  attentions.forEach((node, layer) => {
    const full = FULL_LAYERS.has(layer);
    const where = `attention[${layer}] (${full ? "full" : "sliding"})`;
    assertEquals(
      node.ins.length,
      3,
      `${where}: number of ins (the states form has 3: q / k / v — 4 means the old form with a mask)`,
    );
    const owner = slotOwner(layer);
    assertEquals(
      node.states,
      { k: slotName(owner, "k"), v: slotName(owner, "v") },
      `${where}: states field (KV sharing assignment — the owning layer is ${owner})`,
    );
    assertEquals(
      Object.hasOwn(node.attrs, "window") ? node.attrs.window : undefined,
      full ? undefined : WINDOW,
      `${where}: window attr (layer kind vs window)`,
    );
    const shapes = node.ins.map((name) => graph.values[name].shape);
    assertEquals(
      shapes.map((shape) => shape[1]),
      [HEADS, KV_HEADS, KV_HEADS],
      `${where}: head axis`,
    );
    const depth = full ? FULL_HEAD_DIM : SLIDING_HEAD_DIM;
    assertEquals(
      shapes.map((shape) => shape[3]),
      [depth, depth, depth],
      `${where}: D axis (the layer kind's head_dim — also proves node order is layer order)`,
    );
  });
  assertEquals(
    graph.nodes.filter((node) => node.op === "state_append").length,
    SLOTS,
    "number of state_append nodes (one per slot)",
  );

  // Two symbols: `M` (the physical chunk row count — solvable from the input shapes) and `C` (the
  // capacity — a dedicated symbol that appears only in states, bound only at
  // createGenerationContext, ADR 0066 addendum 7).
  assertEquals([...graph.symbols].sort(), ["C", "M"], "graph.symbols");

  // The declared breakdown of the mixed storage (the same exact match as the 1-shot check — a
  // weight that failed the eligibility check silently stays f32). Source of truth: the summary
  // `form.storage` of export_decode.py (i8 = the main embedding + 35 PLE / i4 = every linear −
  // the tied lm_head).
  const storage: Record<string, number> = {};
  for (const initializer of Object.values(graph.initializers)) {
    // A shared initializer (borrowed) has no storage. Counted as `shared` rather than dropped
    // silently.
    const layout =
      initializer.storage === undefined ? "shared" : codecLayout(initializer.storage.codec);
    storage[layout] = (storage[layout] ?? 0) + 1;
  }
  for (const [layout, expected] of Object.entries({ i8: 36, i4: 276 })) {
    assertEquals(
      Object.hasOwn(storage, layout) ? storage[layout] : 0,
      expected,
      `number of initializers with storage ${layout} (full breakdown ${JSON.stringify(storage)})`,
    );
  }
};

/** The i32 input of one chunk (both the token ids and the absolute positions are `[1, rows]`). */
const i32Row = (rows: number, data: Int32Array<ArrayBuffer>): Tensor => ({
  dtype: "i32",
  shape: [1, rows],
  data,
});

/** Takes PreparedPlan's record from the diagnostics after a run (fails loudly if nothing ran). */
const preparedOf = (diagnostics: SessionDiagnostics, where: string) => {
  const prepared = diagnostics.lastRunPrepared;
  assert(prepared !== undefined, `${where}: no lastRunPrepared (the run has not settled)`);
  return prepared;
};

/**
 * Runs the greedy loop on one golden prompt with this gate's plan constants (chunk 32, capacity C).
 * Shared by the default plan-property case and the opt-in parity case so both drive the same loop.
 */
const generateCase = async (
  session: Session,
  tokenName: string,
  caseName: string,
  prompt: readonly number[],
): Promise<{ readonly generated: readonly number[]; readonly elapsed: number }> => {
  // The capacity covers prompt + continuation (otherwise the state slots overflow).
  assert(
    prompt.length + GREEDY_STEPS <= CAPACITY,
    `${caseName}: T ${prompt.length} + K ${GREEDY_STEPS} exceeds the capacity ${CAPACITY}`,
  );
  const started = performance.now();
  const generated = await generateGreedy({
    session,
    inputIds: INPUT_IDS,
    // The host builds RoPE's cos / sin per chunk (ADR 0091 decision 1).
    derive: (_ids, positions) => ropeInputs(ROPE_SPEC, positions),
    token: tokenName,
    chunkLength: CHUNK_LENGTH,
    maxPosition: MAX_POSITION,
    // The capacity symbol reaches only the context creation (it does not appear in the run's
    // bindings — addendum 7).
    bindings: { [CAPACITY_SYMBOL]: CAPACITY },
    prompt,
    maxNewTokens: GREEDY_STEPS,
  });
  return { generated, elapsed: performance.now() - started };
};

/**
 * One prefill chunk assembled by hand (pad rows 0, absolute positions — not through
 * `generateGreedy`). Shared by the census case and the opt-in first-token check so both read the
 * same run shape.
 */
const runSingleChunkPrefill = (
  session: Session,
  context: GenerationContext,
  prompt: Int32Array<ArrayBuffer>,
): Promise<RunOutputs> => {
  assert(prompt.length <= CHUNK_LENGTH, "the census uses a case that fits in one chunk");
  const ids = new Int32Array(CHUNK_LENGTH);
  const positions = new Int32Array(CHUNK_LENGTH);
  for (let row = 0; row < prompt.length; row += 1) {
    ids[row] = prompt[row];
    positions[row] = row;
  }
  return session.run(
    {
      [INPUT_IDS]: i32Row(CHUNK_LENGTH, ids),
      ...ropeInputs(ROPE_SPEC, positions),
    },
    undefined,
    { context, queryLength: prompt.length },
  );
};

// Only a completely empty environment SKIPs. If **anything at all** is there, a gap FAILs (the
// JSDoc of `ANY_PRESENT`).
test.skipIf(!ANY_PRESENT)(
  `Gemma 4 E2B decode asset: the expected cases and the model itself are all present${SERIES_CHECKS_OPT_IN}`,
  () => {
    assertEquals(IO_CASES, [...EXPECTED_IO_CASES], `io cases of ${SERIES_ROOT.pathname}`);
    assertEquals(
      GREEDY_CASES,
      [...EXPECTED_GREEDY_CASES],
      `greedy cases of ${SERIES_ROOT.pathname}`,
    );
    assert(modelPresent(new URL(MODEL_FILE, SERIES_ROOT)), `${MODEL_FILE} is missing`);
  },
);

/**
 * Default case: ① the graph form and ⑤ the plan properties of the decode path. The greedy loop
 * runs on the golden prompts (inputs only); its token lists are compared with torch only in the
 * opt-in parity case below, so this case asserts nothing against upstream values. Session and
 * GenerationContext follow the MUST on the opt-in case (`generateGreedy` owns one context per
 * generation).
 */
describe.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  `Gemma 4 E2B decode acceptance: form premises and plan properties of the decode path (real GPU)${SERIES_CHECKS_OPT_IN}`,
  () => {
    // Assigned in beforeAll; the its do not run when beforeAll throws.
    let parsed!: PreparedModel;
    let tokenName!: string;
    let gpu: GpuContext | undefined;
    let session!: Session;

    beforeAll(async () => {
      const opened = await openSeriesContainer(new URL(MODEL_FILE, SERIES_ROOT));
      parsed = prepareContainer(opened, MODEL_GRAPH);
      [, tokenName] = parsed.graph.outputs;
      gpu = await acquireGpu();
      session = await parsed.createContainerSession(gpu);
    });

    afterAll(async () => {
      // Same order as the former finally block: the session first, then the device.
      await session?.dispose();
      gpu?.destroy();
    });

    it("① form premises: a states-form chunk graph", () => {
      assertDecodeForm(parsed);
    });

    it("⑤ plan properties of the decode path: generating 3 cases × 16 steps uses two plans, the prefill form and the decode form", async () => {
      /** The prefill chunk count per case (the measurement that a multi-chunk prefill ran). */
      const chunkCounts: number[] = [];
      for (const caseName of EXPECTED_GREEDY_CASES) {
        const golden = await loadGreedy(caseName);
        const prompt = [...golden.prompt];
        assertEquals(golden.expected.length, GREEDY_STEPS, `${caseName}: golden step count`);

        const { elapsed } = await generateCase(session, tokenName, caseName, prompt);

        const chunks = Math.ceil(prompt.length / CHUNK_LENGTH);
        chunkCounts.push(chunks);
        console.log(
          `[e2e] gemma4 decode greedy ${caseName}: T=${prompt.length} ` +
            `(prefill ${chunks} chunk) + K=${GREEDY_STEPS} step / ${elapsed.toFixed(0)}ms`,
        );

        // ⑤ Plan properties of the decode path: the plans derived for one generation are **only
        // two, the prefill form (M=32) and the decode form (M=1)** (ADR 0066 decision 4). A form
        // where this grows with the step count (e.g. queryLength mixed into the plan key) would
        // re-derive every step while the values stay correct. It stays at two across cases too (the
        // prompt length does not affect the plan key).
        const prepared = preparedOf(session.diagnostics(), caseName);
        assertEquals(prepared.cachedPlans, 2, `${caseName}: number of derived plans`);
        assertEquals(prepared.hit, true, `${caseName}: the last decode step hits the plan cache`);
      }
      // At least one case ran a multi-chunk prefill (`pastLength > 0` and `M > 1`).
      assert(
        Math.max(...chunkCounts) >= 2,
        `every case fits in one chunk (chunk counts ${chunkCounts}) — ` +
          `at chunkLength ${CHUNK_LENGTH} a prefill with a past never runs`,
      );
    });
  },
);

/**
 * ① form / ② greedy parity / ③ prefill logits / ⑤ plan properties.
 *
 * MUST: **one** Session is built and **shared by every step**. The asset is about 4 GiB, and
 * rebuilding per step would pay the same upload again and again (the same reason the 1-shot check
 * used one Session for its 3 cases). Only the Session (immutable weights and the plan cache) is
 * shared between steps; **the GenerationContext is made separately per use** — one generation =
 * one context (ADR 0066 decision 6), and reusing one would start prefill with the previous
 * generation's KV still in place.
 *
 * Opt-in case (upstream parity): ② and ③ compare with the torch goldens, and ④ holds the census
 * prompt's first-token check against the greedy golden (moved out of the census case, which stays
 * in the default run). ① and ⑤ run in the default case above.
 */
describe.skipIf(!UPSTREAM_PARITY || !AVAILABLE || !GPU_AVAILABLE)(
  `Gemma 4 E2B decode acceptance: parity of fixed token id sequences (real GPU / torch CPU expectations)${UPSTREAM_PARITY_OPT_IN}${SERIES_CHECKS_OPT_IN}`,
  () => {
    // Assigned in beforeAll; the its do not run when beforeAll throws.
    let logitsName!: string;
    let tokenName!: string;
    let gpu: GpuContext | undefined;
    let session!: Session;

    beforeAll(async () => {
      const opened = await openSeriesContainer(new URL(MODEL_FILE, SERIES_ROOT));
      const parsed = prepareContainer(opened, MODEL_GRAPH);
      [logitsName, tokenName] = parsed.graph.outputs;
      gpu = await acquireGpu();
      session = await parsed.createContainerSession(gpu);
    });

    afterAll(async () => {
      // Same order as the former finally block: the session first, then the device.
      await session?.dispose();
      gpu?.destroy();
    });

    it("② greedy parity: 3 cases × 16 steps exactly match torch's expected sequences", async () => {
      for (const caseName of EXPECTED_GREEDY_CASES) {
        const golden = await loadGreedy(caseName);
        const prompt = [...golden.prompt];
        const expected = [...golden.expected];
        assertEquals(expected.length, GREEDY_STEPS, `${caseName}: golden step count`);

        // The golden's qualification (**not a tautology**): if the gap between torch's first and
        // second candidates exceeds twice atol, a numeric difference within ③'s tolerance cannot
        // move the first = this sequence cannot split on numeric differences. The script picks its
        // cases with `MARGIN_FLOOR = 2.5e-2` (> 2 × atol — the producer's floor is above the
        // consumer's premise), so this fails only when the script and the asset disagree.
        const minMargin = Math.min(...golden.margin);
        assert(
          minMargin > 2 * PREFILL_ATOL,
          `${caseName}: the golden's smallest margin ${minMargin} is on the order of atol — ` +
            `in this form the token-sequence agreement could flip on numeric differences (not a valid check)`,
        );

        const { generated, elapsed } = await generateCase(session, tokenName, caseName, prompt);

        // The acceptance itself. A discrete agreement with no tolerance: a wrong MQA head mapping,
        // mixed-up layer kinds, a wrong sliding ring wrap-around, a wrong KV sharing assignment
        // and pad rows leaking in all split the sequence here.
        assertEquals(generated, expected, `${caseName}: generated token sequence`);

        console.log(
          `[e2e] gemma4 decode greedy parity ${caseName}: T=${prompt.length} ` +
            `+ K=${GREEDY_STEPS} step / smallest margin ${minMargin.toExponential(3)} / ` +
            `${elapsed.toFixed(0)}ms`,
        );
      }
    });

    it("③ prefill logits: splits context-en into chunks and compares the valid rows of the last chunk (diagnostic line)", async () => {
      // ② looks only at discrete agreement; this looks at **continuous values**. A change that
      // shifts the numbers slowly without splitting the sequence (a fusion dropping out, a changed
      // reduction order) slips past ②, and this is the line that covers it. It does not go
      // through `generateGreedy`; the chunk assembly (pad 0, absolute positions, fixed physical
      // row count) is **written out here** — using the same helper as the generation loop would
      // put a split error into the expectations in the same way and make the check a tautology.
      //
      // Only **the valid rows of the last chunk** are compared. They are absolute positions
      // 576..597 = beyond the 512 window, values that have gone through the sliding layers' ring
      // wrap-around and eviction, and through the KV of all 18 earlier chunks (a broken position
      // mapping in an earlier chunk would break these). Not scanning all 598 rows is not only
      // about cost (598 × 262144 elements) but also because the argmax of intermediate rows has
      // no margin guarantee (the golden's margins are taken only for the K continuation tokens
      // starting from the last row).
      const io = await loadIo("context-en");
      const total = io.ids.length;
      const vocab = io.logits.length / total;
      assertEquals(io.tokens.length, total, "io token row count");
      assert(
        Number.isSafeInteger(vocab),
        `io logits length ${io.logits.length} is not divisible by the row count`,
      );

      const context = await session.createGenerationContext({
        bindings: { [CAPACITY_SYMBOL]: CAPACITY },
        chunkLength: CHUNK_LENGTH,
      });
      let worst = 0;
      let worstAt = "";
      let chunks = 0;
      let totalQueryLength = 0;
      try {
        for (let position = 0; position < total; position += CHUNK_LENGTH) {
          const queryLength = Math.min(CHUNK_LENGTH, total - position);
          totalQueryLength += queryLength;
          // Pad rows stay 0 (the value contract of ADR 0066 addendum 6) — `Int32Array`'s initial
          // value is that contract.
          const ids = new Int32Array(CHUNK_LENGTH);
          const positions = new Int32Array(CHUNK_LENGTH);
          for (let row = 0; row < queryLength; row += 1) {
            ids[row] = io.ids[position + row];
            positions[row] = position + row;
          }
          const outputs = await session.run(
            {
              [INPUT_IDS]: i32Row(CHUNK_LENGTH, ids),
              ...ropeInputs(ROPE_SPEC, positions),
            },
            undefined,
            { context, queryLength },
          );
          chunks += 1;
          const last = position + CHUNK_LENGTH >= total;

          const logits = outputs[logitsName];
          assertEquals(logits.dtype, "f32", "logits dtype");
          assertEquals(
            logits.shape,
            [1, CHUNK_LENGTH, vocab],
            "logits shape (physical chunk shape)",
          );
          const tokens = outputs[tokenName];
          assertEquals(tokens.dtype, "i32", "token dtype");
          assertEquals(tokens.shape, [1, CHUNK_LENGTH, 1], "token shape");
          if (!last) continue;

          for (let row = 0; row < queryLength; row += 1) {
            const where = `position ${position + row}`;
            // Tokens are discrete, so exact match (the same verdict as ②, applied to every row of
            // the last chunk too).
            assertEquals(tokens.data[row], io.tokens[position + row], `${where}: argmax token`);
            const base = row * vocab;
            const reference = (position + row) * vocab;
            for (let column = 0; column < vocab; column += 1) {
              const delta = Math.abs(logits.data[base + column] - io.logits[reference + column]);
              // MUST: written so that NaN **stays** as "the worst value" (with
              // `!(delta <= worst)` a NaN caught once would be overwritten by the next finite
              // value, and the non-finite value would vanish silently).
              if (Number.isNaN(delta) || delta > worst) {
                worst = delta;
                worstAt = `${where} / column ${column}`;
              }
            }
          }
        }
      } finally {
        // MUST: released even on a failure midway (do not hold on to the VRAM of the KV capacity).
        await context.dispose();
      }
      // Avoiding a tautology: `chunks` is the loop iteration count itself, so comparing it would
      // be self-comparison (G4 L-4). Effectively check that the sum of each chunk's queryLength
      // equals the total row count.
      assertEquals(totalQueryLength, total, "sum of the chunks' queryLength");
      console.log(
        `[e2e] gemma4 decode prefill context-en: ${chunks} chunk / ${total} rows in all / ` +
          `last chunk logits maxAbs ${worst.toExponential(3)} (worst ${worstAt})`,
      );
      // A non-finite value (NaN / inf) always fails here (if `worst` is NaN, `<=` is false).
      assert(
        worst <= PREFILL_ATOL,
        `context-en: prefill logits maxAbs ${worst} exceeds atol ${PREFILL_ATOL}` +
          ` (worst ${worstAt})`,
      );
    });

    it("④ census premise: the last valid row of capital-en's hand-built prefill is the greedy golden's first token", async () => {
      const golden = await loadGreedy("capital-en");
      const context = await session.createGenerationContext({
        bindings: { [CAPACITY_SYMBOL]: CAPACITY },
        chunkLength: CHUNK_LENGTH,
      });
      try {
        const prefill = await runSingleChunkPrefill(session, context, golden.prompt);
        const first = prefill[tokenName].data[golden.prompt.length - 1];
        assertEquals(
          first,
          golden.expected[0],
          "the last valid prefill row is the greedy golden's first token",
        );
      } finally {
        await context.dispose();
      }
    });
  },
);

/** The breakdown of one census (dispatch counts of the states-form kernel families). */
type StateCensus = {
  readonly qk: number;
  readonly stats: number;
  readonly pv: number;
  readonly append: number;
};

/**
 * ④ census (ADR 0058 decision 4 / 0067 acceptance condition ③ / 0069 decision 5). Checks from the
 * pipeline keys that the 4 states-form families **actually ran**, that the 2 q-side families carry
 * `:gqa`, that **both** layer kinds ran, and that linear / embedding ran **on compressed storage**.
 *
 * The numbers (②③) do not tell keys apart — a form with `repeat_kv` materialised and a weight
 * expanded to f32 after failing eligibility give the same values when implemented correctly.
 * **Only the keys make this distinction**.
 *
 * NOTE: only ①QK and ③PV carry `:gqa`. ② row statistics and `state_append` have no GQA axis (the
 * key is built by `stateVariantKeyPart(sliding, false)` — src/kernels/state-attention.ts), so "all
 * 4 families carry `:gqa`" does not hold. This checks both sides: **present on the 2 families that
 * should carry it, absent on the 2 that should not**. `:sliding`, by contrast, can appear on all 4
 * families (split by layer kind).
 */
const assertStateCensus = (diagnostics: SessionDiagnostics, where: string): StateCensus => {
  // MUST: a run without a census is not counted as "0" — the key checks would silently check
  // nothing.
  // NOTE: the source is the planned count (a states-form dispatch with zero work counts as 1 too).
  // Every count check here is a lower bound, so a difference toward more than issued does not
  // loosen the checks.
  const entries = requireCensus(diagnostics.lastRunPipelines, where);

  const family = (prefix: string) => entries.filter((entry) => entry.key.startsWith(prefix));
  const qk = family("attention_state_qk");
  const stats = family("attention_state_stats");
  const pv = family("attention_state_pv");
  const append = family("state_append");
  const dispatches = (of: readonly { readonly dispatchCount: number }[]): number =>
    of.reduce((total, entry) => total + entry.dispatchCount, 0);
  const census: StateCensus = {
    qk: dispatches(qk),
    stats: dispatches(stats),
    pv: dispatches(pv),
    append: dispatches(append),
  };

  const shown = entries.map((entry) => `${entry.key}×${entry.dispatchCount}`).join(" / ");
  // ① The 4 families are in effect (a family with 0 means that stage ran on another path or did
  // not run).
  assert(
    census.qk > 0 && census.stats > 0 && census.pv > 0 && census.append > 0,
    `${where}: a states-form family has 0 dispatches ${JSON.stringify(census)} (ran: ${shown})`,
  );
  // Every layer is covered (row-block splitting — ADR 0060 / 0067 decision 7 — only adds, so a
  // lower bound).
  assert(
    census.qk >= LAYERS && census.stats >= LAYERS && census.pv >= LAYERS,
    `${where}: the attention dispatches fall short of the layer count ${LAYERS} ${JSON.stringify(census)}`,
  );
  assert(
    census.append >= SLOTS,
    `${where}: the state_append dispatches fall short of the slot count ${SLOTS} ${JSON.stringify(census)}`,
  );

  // ② 8:1 MQA, so ①QK / ③PV are all `:gqa` variants (`:sliding:gqa` also ends in `:gqa`). If even
  // one plain key shows up, that layer ran with k / v expanded to H, or the GQA decision dropped.
  assertEquals(
    [...qk, ...pv].filter((entry) => !entry.key.endsWith(":gqa")).map((entry) => entry.key),
    [],
    `${where}: non-GQA keys ran despite 8:1 MQA (ran: ${shown})`,
  );
  // The other side — row statistics and append carry no `:gqa` (they have no GQA axis).
  assertEquals(
    [...stats, ...append].filter((entry) => entry.key.includes(":gqa")).map((entry) => entry.key),
    [],
    `${where}: GQA variants appeared for row statistics / state_append (ran: ${shown})`,
  );
  // ③ **Both** layer kinds ran. There are 28 sliding / 7 full layers, so if only one side shows
  // up, either the `window` attr was missed (all treated as full = no window) or the reverse.
  assert(
    qk.some((entry) => entry.key.includes(":sliding")) &&
      qk.some((entry) => !entry.key.includes(":sliding")),
    `${where}: attention_state_qk ran for only one layer kind (ran: ${shown})`,
  );
  // ④ No non-states attention (it did not fall into the separate fused-attention family).
  assertEquals(
    entries
      .filter(
        (entry) => entry.key.startsWith("attention") && !entry.key.startsWith("attention_state"),
      )
      .map((entry) => entry.key),
    [],
    `${where}: keys of non-states-form attention are mixed in (ran: ${shown})`,
  );

  // ⑤ Mixed storage holds on the decode path too (ADR 0069 decision 5 — the same 2 conditions as
  // the 1-shot check). A weight that fails eligibility is expanded to f32 without an exception,
  // so only the keys show it. The i4 linears at M=1 run in the GEMV family `linear_gemv:`
  // (ADR 0082), so both families are collected — with only the `linear:` prefix the decode run's
  // 276 would drop out of the check (the caveat in ADR 0082's consequences).
  const linear = entries.filter(
    (entry) => entry.key.startsWith("linear:") || entry.key.startsWith("linear_gemv:"),
  );
  assert(linear.length > 0, `${where}: no linear in the breakdown (ran: ${shown})`);
  assertEquals(
    linear
      .filter((entry) => !entry.key.includes(":wi4g32") && !entry.key.includes(":wi8"))
      .map((entry) => entry.key),
    [],
    `${where}: a linear not on compressed storage ran (a weight expanded to f32 after failing eligibility — ${shown})`,
  );
  // The embeddings are **only the quantized group** (the main embedding + 35 PLE = `:wi8`). The
  // generation that baked the RoPE tables mixed in 4 f32 table lookups (cos/sin × 2 layer kinds),
  // but the tables became derived inputs the host builds (ADR 0091 decision 1), so **not a single**
  // plain f32 key appears — if one does, it is a weight expanded to f32 after dropping out of
  // quantization (no exception, so only the keys show it) or an old-generation asset with the
  // tables baked in.
  const embedding = entries.filter((entry) => entry.key.startsWith("embedding:"));
  assert(embedding.length > 0, `${where}: no embedding in the breakdown (ran: ${shown})`);
  const plainEmbedding = embedding.filter((entry) => !entry.key.includes(":wi8"));
  assertEquals(
    dispatches(plainEmbedding),
    0,
    `${where}: an f32-stored embedding ran (dropped out of quantization, or an old-generation asset with the RoPE tables baked in —` +
      ` ran: ${shown})`,
  );
  assert(
    dispatches(embedding.filter((entry) => entry.key.includes(":wi8"))) >= 36,
    `${where}: the i8-stored embeddings fall short of 36 (the main embedding + 35 PLE)` +
      ` (ran: ${shown})`,
  );
  return census;
};

/**
 * ④ census (real GPU).
 *
 * MUST: the source is `lastRunPipelines`, which does not depend on measurement (it runs on a device
 * without measurement too). undefined / empty is failed by {@link assertStateCensus} above.
 */
test.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  `Gemma 4 E2B decode census: keys of the states-form kernel families and the mixed storage (real GPU)${SERIES_CHECKS_OPT_IN}`,
  async () => {
    const opened = await openSeriesContainer(new URL(MODEL_FILE, SERIES_ROOT));
    const parsed = prepareContainer(opened, MODEL_GRAPH);
    const [, tokenName] = parsed.graph.outputs;
    // The shortest case (T=6) is enough — what is checked is the kinds and counts of the pipelines
    // that ran.
    const golden = await loadGreedy("capital-en");

    const gpu = await acquireGpu();
    const session = await parsed.createContainerSession(gpu);
    // The residency of the mixed storage itself (ADR 0069's acceptance condition — a measured line
    // independent of the key checks). A weight that fails eligibility is only expanded to f32 on
    // the CPU without an exception, so hostExpandedBytes is the only direct observation.
    const storage = session.diagnostics().storage;
    assert(storage !== undefined, "diagnostics.storage is missing");
    assertEquals(
      storage.hostExpandedBytes,
      0,
      "hostExpandedBytes (CPU expansion of ineligible weights)",
    );
    assert(
      storage.residentCompressedBytes > 3_000_000_000,
      `residentCompressedBytes ${storage.residentCompressedBytes} is not the size of a mixed residency`,
    );
    const context = await session.createGenerationContext({
      bindings: { [CAPACITY_SYMBOL]: CAPACITY },
      chunkLength: CHUNK_LENGTH,
    });
    try {
      const prefill = await runSingleChunkPrefill(session, context, golden.prompt);
      const prefillCensus = assertStateCensus(session.diagnostics(), "prefill");
      // A record of one run's total dispatches (planned) and fusion hit counts (not a check — the
      // fusion expectations belong to the static check on the assets_fusion_counts_test side). When
      // a fusion drops out, the dispatches grow while the values stay correct, so the total is kept
      // from the same run as the census. GPU time needs a measuring device, so it is not part of
      // the census check (measurement can take the device down on Metal's query set limit).
      const runRecord = (diagnostics: SessionDiagnostics): string => {
        const census = requireCensus(diagnostics.lastRunPipelines, "record");
        const total = census.reduce((sum, row) => sum + row.dispatchCount, 0);
        return `dispatch ${total} fusions ${JSON.stringify(diagnostics.lastRunFusions)}`;
      };
      const prefillRecord = runRecord(session.diagnostics());

      // The decode form (M=1) is a plan separate from prefill, so the key checks are needed once
      // for each.
      // The first token is compared with the greedy golden only in the opt-in parity case (④).
      const first = prefill[tokenName].data[golden.prompt.length - 1];
      await session.run(
        {
          [INPUT_IDS]: i32Row(1, Int32Array.of(first)),
          ...ropeInputs(ROPE_SPEC, Int32Array.of(golden.prompt.length)),
        },
        undefined,
        { context, queryLength: 1 },
      );
      const decodeCensus = assertStateCensus(session.diagnostics(), "decode");
      // The 277 linears of decode (M=1) went onto the GEMV family (ADR 0082 — 276 i4 + the i8
      // lm_head (perf-ledger K-16)). The per-key check is gpu_linear_gemv_test — this checks that
      // the real graph's run actually ran on that family, the line that detects a regression where
      // the branch predicate silently narrows (the values stay correct, only the speed goes back).
      const gemvDispatches = requireCensus(session.diagnostics().lastRunPipelines, "decode")
        .filter((entry) => entry.key.startsWith("linear_gemv:"))
        .reduce((total, entry) => total + entry.dispatchCount, 0);
      assertEquals(
        gemvDispatches,
        277,
        "decode linear_gemv dispatch count (276 i4 linears + the i8 lm_head)",
      );
      console.log(
        `[e2e] gemma4 decode census: prefill ${JSON.stringify(prefillCensus)} / ` +
          `decode ${JSON.stringify(decodeCensus)}`,
      );
      console.log(`[e2e] gemma4 decode prefill breakdown: ${prefillRecord}`);
      console.log(`[e2e] gemma4 decode decode breakdown: ${runRecord(session.diagnostics())}`);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);
