// Real-GPU acceptance check of the real-weight MiniCPM5-1B (states-form decode graph) — **parity
// of a fixed token id sequence**.
//
// Where wave A's `e2e_minicpm5_test.ts` (removed in the 0.15 triage) covered GQA's numerics in the
// 1-shot form (a single pass, equivalent to prefill), this one covers that **the whole generation
// loop** (the two forms of ADR 0066 decision 4, prefill-chunk and decode, + the GenerationContext's
// KV slots) **emits the same tokens in the same order as torch** on real weights. The acceptance is
// not a tolerance but **discrete agreement**, the same philosophy as the `.lab` / PNG sha256 checks
// (pin the output artifact itself — however much the values move, green as long as the sequence
// does not diverge, red once it does).
//
// There are 5 checks:
//
// ① form preconditions (2 inputs / 2 outputs / output 1 produced by argmax / 48 state slots /
//   symbols M and C)
// ② **greedy parity** (3 cases × K=16 steps exactly match torch's expected sequence) — this
//   check's claim
// ③ tolerance comparison of the prefill logits (3 hand-built chunks that bypass `generateGreedy` —
//   a diagnostic line)
// ④ census (the states-form kernel families actually run, and only the q side gets `:gqa`)
// ⑤ the plan properties of the decode path (PreparedPlan settles at 2: the prefill form and the
//   decode form)
//
// The host-side generation loop (`src/generation/greedy.ts`) is a models implementation, so this
// check sits on the models side (the same split as `e2e_vowel_detector_chain_test.ts` — no
// backwards dependency of a runtime test importing the models implementation by a relative path).
// The runtime side, in turn, holds the acceptance conditions of the states-form kernels on
// synthetic graphs (`tests/gpu_state_acceptance_test.ts`), which need no real weights.
//
// ## Assets
//
// `outputs/series/minicpm5-1b-decode/` (outside the repository because the f32 weights are
// 4.03 GiB — `outputs/` in `.gitignore`). The generation script is
// `tools/export-recipes/minicpm5/export_decode.py`, and {@link GENERATE_COMMAND} below is the
// command's source of truth as is.
//
// - `io.<case>.safetensors` — inputs and outputs of **one unpadded full-length pass** (4 cases).
//   Not a reference that ran the chunk-form graph over the full length in one go but a table of
//   "the values torch gives with the same weights at the same positions"; this side splits it into
//   chunks and compares **only the valid rows** (only the states-form attention's output is pinned
//   to 0 on pad rows — ADR 0066 addendum 8. The later MLP / lm_head write meaningless values to pad
//   rows as well, so the graph outputs of pad rows **must not be read** — the reference side has no
//   counterpart either).
// - `greedy.<case>.safetensors` — the source of truth for the decode acceptance (3 cases).
//   `prompt` i32 `[T]` / `expected` i32 `[K]` / `margin` f32 `[K]`. The expected sequences were
//   taken with a **full-length full re-forward** (bypassing the KV cache path — the MUST of
//   `export_decode.greedy_continuation`), so the mechanism under acceptance and the mechanism that
//   makes the expected values are independent.
//
// MiniCPM5 has no distribution, so every case here is an implementation-time series check
// (ADR 0139 decision 6 and Rulings D2): it runs only with KARUME_SERIES=1 (on top of the
// low-priority lane's KARUME_ALL_FAMILIES=1) and SKIPs explicitly otherwise.
//
// Where the assets are absent it **SKIPs explicitly**. ADR 0005's "all-SKIP is an explicit FAIL"
// guard looks only at *whether a GPU adapter exists*, independent of this SKIP. Assets that are
// **partially** present (missing cases) FAIL rather than SKIP (the "asset completeness" test below)
// — that would be a silent apparent success.

import { type DirEntry, isNotFound, readDirSync, readFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  type GpuContext,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
  type SafetensorsFile,
  type Session,
  type SessionDiagnostics,
  type Tensor,
} from "@karume/runtime";
import { generateGreedy } from "../../models/src/generation/greedy.ts";
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../models/tests/helpers/series-graphs.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";
import { requireCensus } from "../../runtime/tests/helpers/pipeline-census.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";

const SERIES_NAME = "minicpm5-1b-decode";
const SERIES_ROOT = new URL(`../../../outputs/series/${SERIES_NAME}/`, import.meta.url);
const MODEL_FILE = "model.krm";
/**
 * The graph name in the container (the table is the one in helpers/series-graphs.ts — looked up
 * from the same source of truth as the guard).
 */
const MODEL_GRAPH = seriesGraph(SERIES_NAME);
const IO_PREFIX = "io.";
const GREEDY_PREFIX = "greedy.";
const SUFFIX = ".safetensors";

/**
 * The generation command, ready to paste on a SKIP (spelled the same as
 * tools/export-recipes/minicpm5/README.md).
 */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' python -m minicpm5.export_decode";

/**
 * The cases that should have been generated. **Fixed here rather than enumerated** — relying on an
 * enumeration alone, tests would silently vanish in an environment that ran only part of the
 * generation, giving "green but unchecked". The source of truth is `GOLDEN_CASES` (io, T = 6 / 12
 * / 87 / 61) and `GREEDY_CASES` (greedy — `capital-ja` is excluded for lack of continuation
 * margin) in `export_decode.py`.
 */
const EXPECTED_IO_CASES = ["capital-en", "capital-ja", "context-en", "context-ja"] as const;
const EXPECTED_GREEDY_CASES = ["capital-en", "context-en", "context-ja"] as const;

/** The step count of the greedy goldens (the source of truth is `export_decode.GREEDY_STEPS`). */
const GREEDY_STEPS = 16;

/**
 * The row count of a fixed-length prefill chunk (the plan-time constant of ADR 0066 decision 4).
 *
 * **32 is the value that "always exercises multi-chunk prefill"**. The context cases (T = 87 / 61)
 * split into 3 / 2 chunks, and the 2nd chunk onwards is a prefill with `pastLength > 0` (`M > 1`
 * with a past — a path distinct from decode's `M = 1` and from the 1st chunk's `P = 0`). With a
 * value that fits every case in 1 chunk, the states-form attention's "two-sided predicate over the
 * past + the current chunk" would never be exercised on real weights.
 */
const CHUNK_LENGTH = 32;

/**
 * The capacity symbol of the KV slots (`export_decode.CAPACITY_SYMBOL`) and the value this check
 * binds.
 *
 * 128 covers every case's `T + K` (the longest is context-en's 87 + 16 = 103). Whether it covers
 * them is measured per case (the {@link CAPACITY} check below) — when a re-export with a longer
 * prompt runs out of capacity, this is where it fails instead of being silently rounded.
 */
const CAPACITY_SYMBOL = "C";
const CAPACITY = 128;

/**
 * The exclusive upper bound of the absolute positions this series can look up (the source of truth
 * is `export_decode.ROPE_TABLE_POSITIONS` = the RoPE table's row count). A gather outside the table
 * returns non-finite values that argmax folds into a silently wrong token, so `generateGreedy`
 * requires this value and fails at the entry point.
 */
const MAX_POSITION = 512;

/** The graph input names (the source of truth is `export_decode.INPUT_IDS` / `POSITION_IDS`). */
const INPUT_IDS = "input_ids";
const POSITION_IDS = "position_ids";

/**
 * The layer count (the config's `num_hidden_layers`) and the state slot count (2 per layer, k /
 * v).
 */
const LAYERS = 24;
const SLOTS = LAYERS * 2;

/**
 * The head counts of the q side / the k and v side (16:2 = true GQA form; the grounds for `:gqa`
 * in the census).
 */
const HEADS = 16;
const KV_HEADS = 2;

/**
 * The tolerance for comparing the prefill logits (`[1,M,130560]`) against the torch CPU expected
 * values.
 *
 * It is **the same threshold** as wave A's 1-shot check (`MINICPM5_TOLERANCE` in
 * `e2e_minicpm5_test.ts`) — what is compared is the same (logits of the same weights at the same
 * positions), and only the path differs (a single pass vs chunks + KV slots), so a different
 * threshold would make it impossible to tell "a path difference or a threshold difference".
 *
 * Measured (2026-08-18, the 87 valid rows of context-en / T=87 run in 3 chunks): **maxAbs
 * 7.439e-5**, **the same value** as the 7.44e-5 the 1-shot check measured on the same case (1/13
 * of atol 1e-3). What the measurement says: chunk splitting + the KV slots give no more error than
 * a single pass.
 *
 * **rtol is 0** (atol alone decides — the reason written in the 1-shot side's docstring, "a few
 * elements have a reference value of exactly 0.0", applies as is).
 */
const PREFILL_ATOL = 1e-3;

/**
 * Lists a directory. Only a missing directory degrades to empty.
 * MUST: anything other than NotFound propagates — reading a permission error or the like as "no
 * asset" would quietly turn a check that never ran green as a SKIP.
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
 * Enumerated synchronously, as registration needs it (the same reason as `Deno.test`'s ignore).
 * Without KARUME_SERIES=1 the series is not read at all.
 */
const IO_CASES = SERIES_CHECKS ? discoverCases(IO_PREFIX) : [];
const GREEDY_CASES = SERIES_CHECKS ? discoverCases(GREEDY_PREFIX) : [];

/**
 * Whether assets are present. None at all = an environment that did not generate them, so
 * everything SKIPs (a partial gap goes to FAIL).
 */
const AVAILABLE = IO_CASES.length > 0 || GREEDY_CASES.length > 0;
/**
 * Whether **anything at all** is left (the SKIP predicate of the completeness test — Codex wave H
 * finding H-02). A gap where every golden is gone and only the model is left makes `AVAILABLE`
 * false, which would SKIP the completeness test itself — the predicate that FAILs a gap may sleep
 * only when everything is empty.
 */
const ANY_PRESENT = SERIES_CHECKS && (AVAILABLE || modelPresent(new URL(MODEL_FILE, SERIES_ROOT)));

if (SERIES_CHECKS && !AVAILABLE) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} has no exported assets, so the MiniCPM5-1B decode acceptance ` +
      `is SKIPped (outside the repository because the f32 weights are 4.03 GiB). Generate: ${GENERATE_COMMAND}`,
  );
}

/**
 * Reads one file as an `ArrayBuffer`.
 *
 * MUST: do not slice when the view covers the whole buffer — one golden is GB-sized, so an
 * unconditional `slice` doubles the peak (the same reason as the 1-shot check).
 */
const readBuffer = async (file: string): Promise<ArrayBuffer> => {
  const bytes = await readFile(new URL(file, SERIES_ROOT));
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return bytes.buffer;
  }
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

/**
 * Takes one golden tensor as a view of the declared dtype (no copy).
 *
 * MUST: fail when the storage dtype disagrees — f32 and i32 elements are both 4 bytes, so silently
 * reading one as the other would let a reinterpretation of the bits "pass". The runtime-side test
 * helper of the same name (`helpers/golden-io.ts`) is not imported (a package's tests do not
 * depend on another package's test internals — the same split as the NOTE in `helpers/gpu.ts`).
 */
const goldenF32 = (file: SafetensorsFile, name: string): Float32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `'${name}' is not in the golden`);
  assertEquals(view.dtype, "F32", `golden '${name}' storage dtype`);
  return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

const goldenI32 = (file: SafetensorsFile, name: string): Int32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `'${name}' is not in the golden`);
  assertEquals(view.dtype, "I32", `golden '${name}' storage dtype`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

/** `greedy.<case>.safetensors` (the source of truth for the decode acceptance). */
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
    `${GREEDY_PREFIX}${caseName}${SUFFIX} tensor keys`,
  );
  return {
    prompt: goldenI32(file, "prompt"),
    expected: goldenI32(file, "expected"),
    margin: goldenF32(file, "margin"),
  };
};

/** `io.<case>.safetensors` (inputs and outputs of one unpadded full-length pass). */
type IoGolden = {
  readonly ids: Int32Array<ArrayBuffer>;
  readonly positions: Int32Array<ArrayBuffer>;
  readonly logits: Float32Array<ArrayBuffer>;
  readonly tokens: Int32Array<ArrayBuffer>;
};

const loadIo = async (caseName: string): Promise<IoGolden> => {
  const file = parseSafetensors(await readBuffer(`${IO_PREFIX}${caseName}${SUFFIX}`));
  assertEquals(
    [...file.tensors.keys()].sort(),
    [`input.${INPUT_IDS}`, `input.${POSITION_IDS}`, "output.0", "output.1"].sort(),
    `${IO_PREFIX}${caseName}${SUFFIX} tensor keys`,
  );
  return {
    ids: goldenI32(file, `input.${INPUT_IDS}`),
    positions: goldenI32(file, `input.${POSITION_IDS}`),
    logits: goldenF32(file, "output.0"),
    tokens: goldenI32(file, "output.1"),
  };
};

/**
 * The form preconditions of the states-form decode graph (it looks only at properties that **break
 * silently while the values still match**).
 *
 * The complete form check is held by the writer side (`export_decode.assert_ir_form_decode`). The
 * reader side only needs to confirm "the 2 forms this check runs are graphs with the expected entry
 * points, exits and slots"; if this loosened, **another graph would get accepted** (e.g. the older
 * form with a mask input left over becomes chunk-local causal, and the values match only for cases
 * whose prompt fits in 1 chunk).
 */
const assertDecodeForm = (parsed: PreparedModel): void => {
  const graph = parsed.graph;
  assertEquals(
    graph.inputs.map((spec) => spec.name),
    [INPUT_IDS, POSITION_IDS],
    "graph inputs",
  );
  assertEquals(graph.outputs.length, 2, "graph.outputs count (2: logits / token)");

  // Output 1 is produced by argmax (the decode exit of ADR 0068 decision 4). `generateGreedy`
  // feeds this output straight into the next step's input, so if this were another op (e.g. the
  // 1-shot form's logits), "a token sequence read from another tensor of a matching shape" would
  // come out without any exception.
  const producer = new Map<string, string>();
  for (const node of graph.nodes) {
    for (const out of node.outs) producer.set(out, node.op);
  }
  assertEquals(
    producer.get(graph.outputs[1]),
    "argmax",
    `producer of output 1 ('${graph.outputs[1]}')`,
  );

  // The states are 2 per layer, k / v. Every attention has a states field, in true GQA form
  // (16:2).
  assertEquals(Object.keys(graph.states).length, SLOTS, "states slot count");
  const attentions = graph.nodes.filter((node) => node.op === "attention");
  assertEquals(attentions.length, LAYERS, "attention node count (= layer count)");
  attentions.forEach((node, index) => {
    assertEquals(
      Object.keys(node.states).sort(),
      ["k", "v"],
      `attention[${index}]: states field (no older form with a mask left over)`,
    );
    const heads = node.ins.map((name) => graph.values[name].shape[1]);
    assertEquals(heads, [HEADS, KV_HEADS, KV_HEADS], `attention[${index}]: head axis`);
  });
  assertEquals(
    graph.nodes.filter((node) => node.op === "state_append").length,
    SLOTS,
    "state_append node count (one per slot)",
  );

  // Two symbols: `M` (the physical chunk row count — solvable from the input shape) and `C` (the
  // capacity — a dedicated symbol that appears only in the states, bound only by
  // createGenerationContext; ADR 0066 addendum 7).
  assertEquals([...graph.symbols].sort(), ["C", "M"], "graph.symbols");
};

/** The i32 inputs of one chunk (both the token ids and the absolute positions are `[1, rows]`). */
const i32Row = (rows: number, data: Int32Array<ArrayBuffer>): Tensor => ({
  dtype: "i32",
  shape: [1, rows],
  data,
});

/** Takes the PreparedPlan record from the diagnostics after a run (fails loudly if none ran). */
const preparedOf = (diagnostics: SessionDiagnostics, where: string) => {
  const prepared = diagnostics.lastRunPrepared;
  assert(prepared !== undefined, `${where}: no lastRunPrepared (the run has not settled)`);
  return prepared;
};

// Only a completely empty environment SKIPs. If **anything at all** is present, a gap is a FAIL
// (the JSDoc of `ANY_PRESENT`).
test.skipIf(!ANY_PRESENT)(
  `MiniCPM5 decode assets: the expected cases and the model are complete${SERIES_CHECKS_OPT_IN}`,
  () => {
    assertEquals(IO_CASES, [...EXPECTED_IO_CASES], `${SERIES_ROOT.pathname}: io cases`);
    assertEquals(GREEDY_CASES, [...EXPECTED_GREEDY_CASES], `${SERIES_ROOT.pathname}: greedy cases`);
    assert(modelPresent(new URL(MODEL_FILE, SERIES_ROOT)), `${MODEL_FILE} is missing`);
  },
);

/**
 * ① form / ② greedy parity / ③ prefill logits / ⑤ plan properties.
 *
 * MUST: **build only one Session and share it across all steps**. This asset is 4.03 GiB of f32
 * weights, and rebuilding per step would pay the same upload again and again (the same reason the
 * 1-shot check used 1 Session for 4 cases). Only the Session (the immutable weights and the plan
 * cache) is shared between steps; **a GenerationContext is made per use** — one generation = one
 * context (ADR 0066 decision 6), and reusing one would start prefill with the previous
 * generation's KV still in it.
 */
describe.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  `MiniCPM5 decode acceptance: parity of a fixed token id sequence (real GPU / torch CPU expected values)${SERIES_CHECKS_OPT_IN}`,
  () => {
    // Assigned in beforeAll; the its do not run when beforeAll throws.
    let parsed!: PreparedModel;
    let logitsName!: string;
    let tokenName!: string;
    let gpu: GpuContext | undefined;
    let session!: Session;

    beforeAll(async () => {
      const opened = await openSeriesContainer(new URL(MODEL_FILE, SERIES_ROOT));
      parsed = prepareContainer(opened, MODEL_GRAPH);
      [logitsName, tokenName] = parsed.graph.outputs;
      gpu = await acquireGpu();
      session = await parsed.createContainerSession(gpu);
    });

    afterAll(async () => {
      // Same order as the former finally block: the session first, then the device.
      await session?.dispose();
      gpu?.destroy();
    });

    it("① form preconditions: it is a states-form chunk graph", () => {
      assertDecodeForm(parsed);
    });

    it("② greedy parity: 3 cases × 16 steps exactly match torch's expected sequence", async () => {
      /** The prefill chunk count per case (the measurement that multi-chunk prefill was exercised). */
      const chunkCounts: number[] = [];
      for (const caseName of EXPECTED_GREEDY_CASES) {
        const golden = await loadGreedy(caseName);
        const prompt = [...golden.prompt];
        const expected = [...golden.expected];
        assertEquals(expected.length, GREEDY_STEPS, `${caseName}: golden step count`);

        // The golden side's eligibility (**not vacuous**): if the gap between torch's top 1 and
        // top 2 exceeds twice atol, a numerical difference within ③'s tolerance cannot move the
        // top 1 = numerical differences cannot split this sequence. The script picks cases with
        // `MARGIN_FLOOR = 1e-2`; the smallest measured margin is context-ja's 0.0261 (step 15) =
        // 26× atol 1e-3.
        const minMargin = Math.min(...golden.margin);
        assert(
          minMargin > 2 * PREFILL_ATOL,
          `${caseName}: the golden's smallest margin ${minMargin} is on the order of atol — ` +
            `in this form a numerical difference could flip the token sequence's agreement (it does not work as a check)`,
        );
        // The capacity covers the prompt + the continuation (otherwise the state slots overflow).
        assert(
          prompt.length + GREEDY_STEPS <= CAPACITY,
          `${caseName}: T ${prompt.length} + K ${GREEDY_STEPS} exceeds the capacity ${CAPACITY}`,
        );

        const started = performance.now();
        const generated = await generateGreedy({
          session,
          inputIds: INPUT_IDS,
          // This series still has the RoPE table baked in, so the graph input is still the one
          // `position_ids`, but the supply of positions was unified into `GreedySpec.derive` (ADR
          // 0091 decision 1 — the gemma4 side builds cos / sin from the positions, and the
          // `positionIds` field was retired).
          derive: (_ids, positions) => ({ [POSITION_IDS]: i32Row(positions.length, positions) }),
          token: tokenName,
          chunkLength: CHUNK_LENGTH,
          maxPosition: MAX_POSITION,
          // The capacity symbol goes only to the context creation (it does not appear in the
          // run's bindings — addendum 7).
          bindings: { [CAPACITY_SYMBOL]: CAPACITY },
          prompt,
          maxNewTokens: GREEDY_STEPS,
        });
        const elapsed = performance.now() - started;

        // The acceptance itself. A discrete agreement without a tolerance: a wrong GQA head
        // mapping, a RoPE position shift, a wrong KV-slot row mapping, or pad rows leaking in all
        // split the sequence here.
        assertEquals(generated, expected, `${caseName}: generated token sequence`);

        const chunks = Math.ceil(prompt.length / CHUNK_LENGTH);
        chunkCounts.push(chunks);
        console.log(
          `[e2e] minicpm5 decode greedy ${caseName}: T=${prompt.length} ` +
            `(prefill ${chunks} chunk) + K=${GREEDY_STEPS} step / smallest margin ` +
            `${minMargin.toExponential(3)} / ${elapsed.toFixed(0)}ms`,
        );

        // ⑤ The plan properties of the decode path: one generation derives **only 2 plans, the
        // prefill form (M=32) and the decode form (M=1)** (ADR 0066 decision 4). A form where this
        // grows with the step count (e.g. queryLength mixed into the plan key) re-derives every
        // step while the values stay correct. It stays at 2 across cases too (the prompt length
        // does not affect the plan key).
        const prepared = preparedOf(session.diagnostics(), caseName);
        assertEquals(prepared.cachedPlans, 2, `${caseName}: derived plan count`);
        assertEquals(prepared.hit, true, `${caseName}: the final decode step hits the plan cache`);
      }
      // At least one case exercised multi-chunk prefill (`pastLength > 0` and `M > 1`).
      assert(
        Math.max(...chunkCounts) >= 2,
        `every case fits in 1 chunk (chunk counts ${chunkCounts}) — ` +
          `with chunkLength ${CHUNK_LENGTH} a prefill with a past never runs`,
      );
    });

    it("③ prefill logits: compare only the valid rows of context-en's 3 chunks (diagnostic line)", async () => {
      // Where ② looks only at discrete agreement, this looks at **continuous values**. A change
      // that drifts the values without splitting the sequence (a fusion falling off, a change of
      // reduction order) passes ② untouched, so this line covers it. It does not go through
      // `generateGreedy` and writes the chunk assembly (pad 0, absolute positions, a fixed
      // physical row count) **right here** — using the same helper as the generation loop would
      // put a splitting error into the expected side the same way and make it vacuous.
      //
      // Measured fault injection (2026-08-18): adding +1 to the absolute positions of the 2nd
      // chunk onwards only breaks the argmax of the first shifted row (position 33) at once. On
      // the other hand, **+1 on every row uniformly** passes untouched at maxAbs 8.9e-5 — RoPE
      // looks only at relative positions, so this is not a weakness of the check but the
      // invariance itself (what this path can catch is a relative shift between chunks, mixed-up
      // rows and pad rows leaking in, not a uniform offset of the absolute positions).
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
          // Pad rows stay 0 (the value contract of ADR 0066 addendum 6) — the initial value of
          // `Int32Array` is that contract.
          const ids = new Int32Array(CHUNK_LENGTH);
          const positions = new Int32Array(CHUNK_LENGTH);
          for (let row = 0; row < queryLength; row += 1) {
            ids[row] = io.ids[position + row];
            positions[row] = position + row;
          }
          const outputs = await session.run(
            {
              [INPUT_IDS]: i32Row(CHUNK_LENGTH, ids),
              [POSITION_IDS]: i32Row(CHUNK_LENGTH, positions),
            },
            undefined,
            { context, queryLength },
          );
          chunks += 1;

          const logits = outputs[logitsName];
          assertEquals(logits.dtype, "f32", "logits dtype");
          assertEquals(
            logits.shape,
            [1, CHUNK_LENGTH, vocab],
            "logits shape (the physical chunk form)",
          );
          const tokens = outputs[tokenName];
          assertEquals(tokens.dtype, "i32", "token dtype");
          assertEquals(tokens.shape, [1, CHUNK_LENGTH, 1], "token shape");

          for (let row = 0; row < queryLength; row += 1) {
            const where = `position ${position + row}`;
            // Tokens are discrete, so exact agreement (the same decision as ② applied to the
            // rows in the middle of a chunk too).
            assertEquals(tokens.data[row], io.tokens[position + row], `${where}: argmax token`);
            const base = row * vocab;
            const reference = (position + row) * vocab;
            for (let column = 0; column < vocab; column += 1) {
              const delta = Math.abs(logits.data[base + column] - io.logits[reference + column]);
              // MUST: written so a NaN **stays** as "the worst value" (with `!(delta <= worst)`,
              // a NaN caught once would be overwritten by the next finite value and the
              // non-finite would silently vanish).
              if (Number.isNaN(delta) || delta > worst) {
                worst = delta;
                worstAt = `${where} / column ${column}`;
              }
            }
          }
        }
      } finally {
        // MUST: release it even on a failure midway (do not keep holding the VRAM of the KV
        // capacity).
        await context.dispose();
      }
      // Avoiding a vacuous check: `chunks` is the loop's own iteration count, so comparing it
      // would be self-comparison (G4 L-4). Check effectively that the sum of each chunk's
      // queryLength equals the valid row count total.
      assertEquals(totalQueryLength, total, "sum of the chunks' queryLength");
      console.log(
        `[e2e] minicpm5 decode prefill context-en: ${chunks} chunk / ${total} valid rows / ` +
          `logits maxAbs ${worst.toExponential(3)} (worst ${worstAt})`,
      );
      // Non-finite values (NaN / inf) always fail here (if `worst` is NaN, `<=` is false).
      assert(
        worst <= PREFILL_ATOL,
        `context-en: prefill logits maxAbs ${worst} exceeded atol ${PREFILL_ATOL}` +
          ` (worst ${worstAt})`,
      );
    });
  },
);

/** The breakdown of one census (the dispatch counts of the states-form kernel families). */
type StateCensus = {
  readonly qk: number;
  readonly stats: number;
  readonly pv: number;
  readonly append: number;
};

/**
 * ④ census (ADR 0058 decision 4 / 0067 acceptance condition ③). It checks, from the pipeline
 * keys, that the 4 states-form families **actually ran** and that the 2 q-side families got
 * `:gqa`.
 *
 * The values (② ③) do not tell the keys apart — the form that materializes `repeat_kv` and the
 * older fused-attention form give the same values when implemented correctly. **Only the keys make
 * this distinction**.
 *
 * NOTE: only ①QK and ③PV get `:gqa`. ② the row statistics and `state_append` have no GQA axis
 * (their key comes from `stateVariantKeyPart(sliding, false)` — src/kernels/state-attention.ts),
 * so "`:gqa` on all 4 families" does not hold. This checks both sides: **present on the 2 families
 * that should have it, absent on the 2 that should not**.
 */
const assertStateCensus = (diagnostics: SessionDiagnostics, where: string): StateCensus => {
  // MUST: do not count a run without a census as "0" — the key checks would silently do nothing.
  // NOTE: the source is the planned count (a states-form dispatch with zero work also counts as
  // 1). Every count check here is a lower bound, so a difference toward more than issued does not
  // loosen the check.
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
  // ① The 4 families are in effect (a family with 0 means that stage ran another path or did not
  // run).
  assert(
    census.qk > 0 && census.stats > 0 && census.pv > 0 && census.append > 0,
    `${where}: a states-form family has 0 ${JSON.stringify(census)} (what ran: ${shown})`,
  );
  // Every layer is there (row-block splitting — ADR 0060 / 0067 decision 7 — only adds, so lower
  // bounds).
  assert(
    census.qk >= LAYERS && census.stats >= LAYERS && census.pv >= LAYERS,
    `${where}: attention dispatches fall short of the layer count ${LAYERS} ${JSON.stringify(census)}`,
  );
  assert(
    census.append >= SLOTS,
    `${where}: state_append dispatches fall short of the slot count ${SLOTS} ${JSON.stringify(census)}`,
  );

  // ② With 16:2, ①QK / ③PV are all `:gqa` variants. A single bare key means that layer ran with
  // k / v widened to H, or the GQA detection failed.
  assertEquals(
    [...qk, ...pv].filter((entry) => !entry.key.endsWith(":gqa")).map((entry) => entry.key),
    [],
    `${where}: non-GQA keys ran despite 16:2 GQA (what ran: ${shown})`,
  );
  // The other side — the row statistics and append do not get `:gqa` (they have no GQA axis).
  assertEquals(
    [...stats, ...append].filter((entry) => entry.key.includes(":gqa")).map((entry) => entry.key),
    [],
    `${where}: GQA variants appeared in the row statistics / state_append (what ran: ${shown})`,
  );
  // The full form without `window` attrs, so no sliding variant appears (one would cut outside
  // the window).
  assertEquals(
    entries.filter((entry) => entry.key.includes(":sliding")).map((entry) => entry.key),
    [],
    `${where}: sliding variants ran despite the full form (what ran: ${shown})`,
  );
  // ③ Zero non-states attention kernels (it did not fall to another fused-attention family).
  assertEquals(
    entries
      .filter(
        (entry) => entry.key.startsWith("attention") && !entry.key.startsWith("attention_state"),
      )
      .map((entry) => entry.key),
    [],
    `${where}: attention keys that are not states-form are mixed in (what ran: ${shown})`,
  );
  return census;
};

/**
 * ④ census (real GPU).
 *
 * MUST: the source is `lastRunPipelines`, which does not depend on timing (it works on a device
 * without timing). {@link assertStateCensus} above fails on undefined / empty.
 */
test.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  `MiniCPM5 decode census: the states-form kernel families run as GQA variants (real GPU)${SERIES_CHECKS_OPT_IN}`,
  async () => {
    const opened = await openSeriesContainer(new URL(MODEL_FILE, SERIES_ROOT));
    const parsed = prepareContainer(opened, MODEL_GRAPH);
    const [, tokenName] = parsed.graph.outputs;
    // The shortest case (T=6) is enough — what is checked is the kinds and counts of pipelines that
    // ran.
    const golden = await loadGreedy("capital-en");
    assert(golden.prompt.length <= CHUNK_LENGTH, "the census uses a case exercised in 1 chunk");

    const gpu = await acquireGpu();
    const session = await parsed.createContainerSession(gpu);
    const context = await session.createGenerationContext({
      bindings: { [CAPACITY_SYMBOL]: CAPACITY },
      chunkLength: CHUNK_LENGTH,
    });
    try {
      const ids = new Int32Array(CHUNK_LENGTH);
      const positions = new Int32Array(CHUNK_LENGTH);
      for (let row = 0; row < golden.prompt.length; row += 1) {
        ids[row] = golden.prompt[row];
        positions[row] = row;
      }
      const prefill = await session.run(
        {
          [INPUT_IDS]: i32Row(CHUNK_LENGTH, ids),
          [POSITION_IDS]: i32Row(CHUNK_LENGTH, positions),
        },
        undefined,
        { context, queryLength: golden.prompt.length },
      );
      const prefillCensus = assertStateCensus(session.diagnostics(), "prefill");

      // The decode form (M=1) is a plan separate from prefill, so the key check is needed once
      // for each.
      const first = prefill[tokenName].data[golden.prompt.length - 1];
      assertEquals(
        first,
        golden.expected[0],
        "prefill's last valid row is the greedy golden's first token",
      );
      await session.run(
        {
          [INPUT_IDS]: i32Row(1, Int32Array.of(first)),
          [POSITION_IDS]: i32Row(1, Int32Array.of(golden.prompt.length)),
        },
        undefined,
        { context, queryLength: 1 },
      );
      const decodeCensus = assertStateCensus(session.diagnostics(), "decode");
      console.log(
        `[e2e] minicpm5 decode census: prefill ${JSON.stringify(prefillCensus)} / ` +
          `decode ${JSON.stringify(decodeCensus)}`,
      );
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);
