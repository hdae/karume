// Acceptance check of the **product graph + host PLE + GenerationSequence** of real-weight Gemma 4
// E2B — the pass line of stage 3.
//
// What it accepts is the generation surface of ADR
// [0083](../../../docs/decisions/0083-generation-api-surface.md) decisions 1–5
// (`src/generation/program.ts` / `sequence.ts`), with 3 checks:
//
// ① **First-turn parity**: one turn run through a sequence at temperature 0 (this layer's default)
//    matches `greedy.<case>.safetensors` of the logits opt-in series (the expected lists of a torch
//    full re-forward) exactly on 3 cases × K=16. It re-takes, **through the sequence**, the same
//    lists the stage-2 check (`e2e_gemma4_sequence_test.ts`) took with **a loop written on the
//    spot** — proof that the two layers (program / sequence) and the slot for host-derived inputs
//    do not move the lists.
// ② **Multi-turn drops no token** (decision 4's prefill concatenating `pendingToken`): break the
//    first turn midway with `break`, emit the rest as a second turn, and the concatenated list
//    **equals exactly the reference run streamed in one go without interruption (= the golden of
//    ①)**. A dropped token raises no exception — "the last token of the previous assistant turn
//    disappears from the history" is all — so only this direct check catches it.
//    NOTE: the match is **exact** because a chunk with a single valid row runs in decode form (M=1)
//    (the corresponding MUST in `sequence.ts`) — the single run of the resume is identical to the
//    decode run of the uninterrupted run.
// ③ **Real stop on the EOS set** (decision 8): a program that declares stop tokens stops on the real
//    GPU output at a per-case fixed index (2 / 3 / 2, the same as the stage-2 check), and the stop
//    token itself is not emitted as a `token` event but carried by `done`.
// ④ **A stop token the request adds** (ADR 0083 addendum 2026-09-02): even with the program's set
//    **empty**, `GenerationRequest.stopTokens` alone stops at the same index with the same list as
//    ③ (the reason is `stop-token`). If the union check looks only at the program side, this case
//    alone runs to max-tokens.
// ⑤ **Prefill buckets** (ADR 0066 addendum, buckets): streaming the last chunk with a smaller
//    physical row count and no pad still yields the same golden token list as ①. Changing the
//    physical row count changes the GEMM geometry (the skeleton switches at M thresholds), so
//    **bit identity is not part of the contract** — the token list not moving on real weights is
//    this optimisation's pass line, and a miss leaves the values correct (pad rows are 0 and
//    harmless), so no other check exists.
// ⑥ **Independence of forks** (`fork` — ADR 0066 / 0083 addenda 2026-10-11, ruling 8): two branches
//    forked through the public surface (`Pipeline.sequence().fork()`) do not share KV. In 3 modes —
//    no speculation / `"always"` / gated — the token lists equal those of the same sequence of
//    operations without fork (the baseline). The order is what matters (the doc of the section
//    below).
//
// NOTE: the stage-2 sampler gate (formerly `e2e_gemma4_sequence_test.ts`, a hand-written host loop
// over `createSampler`) was folded into this file. Its temperature-0 parity used the same 3 goldens
// as ① and its EOS stop indices are the ones ③ pins, so only its unique part survives here: the
// step right after ① checks that `topK: 1` at temperature 1 reproduces the greedy golden on the
// real vocabulary (V = 262,144). The sampler unit tests use vocabularies of 4 to 4,096, so this is
// the only place candidate selection runs at full size.
//
// NOTE: the check that compares the goldens' provenance (the digests `reference.json` bundles) is
// held by ④ of `e2e_gemma4_product_test.ts` for the same 3 files — one asset does not get two.
//
// ## Assets
//
// The container is the distribution mirror's `model` component (`models/karume-gemma4/`, model
// e2b, quant bf16 — the regular line's one preset since 2026-11-04: the checkpoint's own weights
// and its bf16 PLE assets in one container; byte-identical to the bf16 product series), opened
// through the manifest (ADR 0139 decision 1); the expected lists are that series' own greedy
// goldens in the test-asset root (`goldens/gemma4-e2b-product-bf16/`, decision 2 — the same 3
// cases × 16 steps on the same prompts as the decode series' goldens the int4 seat read before).
// Both are outside git; without the mirror ①〜⑤ SKIP explicitly. The goldens follow ADR 0139's
// fetched rule: one absent from a complete fetch of the pinned test-asset dataset FAILs ①〜⑤ (the
// dataset at the pin lacks it); without a complete fetch ①〜⑤ SKIP with the missing files and
// `node tools/test-assets/fetch.ts` in the title. ⑥ alone uses the distribution mirror with the
// drafter, `models/karume-gemma4/` (to run both speculative modes on the same pipeline) — without
// the mirror, ⑥ alone SKIPs explicitly.

import { isNotFound, readFile, readTextFileSync, statSync } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assert, assertEquals, assertNotEquals, assertRejects, assertThrows } from "@std/assert";
import { MANIFEST_FILENAME, parseManifest } from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";
import {
  acquireGpu,
  type GpuContext,
  parseSafetensors,
  type PreparedModel,
  prepareContainer,
  type SafetensorsFile,
  type Session,
} from "@karume/runtime";
import { createGemma4Ple, type Gemma4Ple } from "../../models/src/gemma/ple.ts";
import { gemma4RopeInputNames, ropeInputs, type RopeSpec } from "../../models/src/gemma/rope.ts";
import {
  createGenerationProgram,
  type GenerationWiring,
} from "../../models/src/generation/program.ts";
import {
  createGenerationSequence,
  type GenerationEvent,
  type GenerationSequence,
  physicalChunkRows,
} from "../../models/src/generation/sequence.ts";
import {
  goldenDir,
  TEST_ASSETS,
  TEST_ASSETS_ROOT,
  testAssetsFetched,
} from "@karume/test-support/test-assets";
import {
  mirrorRoot,
  openSelectedContainer,
  selectMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
// Builds the PLE index and the block reader from the opened container (ADR 0109 decision 4).
import { seriesPleHandle } from "../../models/tests/helpers/ple-series.ts";
import { gemma4PleTotalBytes } from "../../models/src/gemma/ple-index.ts";
import {
  chatPrompt,
  chatTurn,
  Pipeline,
  type GenerationStop,
  type SamplerSpec,
} from "../../models/gemma.ts";
import { GEMMA4_DRAFT_STEPS } from "../../models/src/gemma/speculative.ts";
import { mirrorAvailable } from "../../models/tests/helpers/gemma-mirror.ts";
import { allResidentPleBytesOfMirror } from "../../models/tests/helpers/ple-budget.ts";

/**
 * The mirror seat ①〜⑤ read: e2b with the bf16 weights. MUST: name the quant — the session below
 * is built with the options this gate always used (none, the reference summation order the `bf16`
 * preset declares too), so the token lists it checks stay those of the named seat whatever the
 * manifest's default becomes.
 */
const PRODUCT_CHOICE = { model: "e2b", quant: "bf16" } as const;
/** The component key, which is also the container's graph name (container-v1 §2.1). */
const MODEL_GRAPH = "model";
/** The bf16 product series' greedy goldens (the expected lists). */
const GOLDEN_ROOT = goldenDir("gemma4-e2b-product-bf16");
const GREEDY_PREFIX = "greedy.";
const SUFFIX = ".safetensors";

/** The command that restores the goldens (ADR 0139 decision 2). */
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";

/**
 * The expected list of each case (canonical in `export_decode.GREEDY_CASES`) and the index where
 * the EOS set first holds.
 *
 * MUST: the stop indices are fixed by hand (as in the stage-2 check) — deriving them from the
 * golden takes the shape "the index moves along when the set shrinks", and ③ becomes a tautology.
 *
 * The indices were re-read by hand on 2026-11-04 from the bf16 series' goldens when the gate moved
 * there from the int4 seat (a new target): capital-en [9079, 236761, 106, …], capital-ja
 * [31414, 3652, 236924, 106, …] and context-en [9079, 236761, 106, …] first meet the stop set at
 * 2 / 3 / 2 — the indices the decode series' goldens had.
 */
const EXPECTED_CASES = [
  { name: "capital-en", firstStop: 2 },
  { name: "capital-ja", firstStop: 3 },
  { name: "context-en", firstStop: 2 },
] as const;
const GREEDY_STEPS = 16;
/** The token count after which ② cuts the first turn (the second turn emits the rest). */
const BREAK_AFTER = 5;

/** `eos_token_id` of gemma-4-E2B-it's `generation_config.json` (ADR 0083 decision 8). */
const STOP_TOKENS = [1, 106, 50] as const;

/**
 * The row count of the fixed-length prefill chunk. **32 is the value that "exercises the
 * sequence layer's multi-chunk path on real weights"**, not the distribution's default (768 — ADR
 * 0091 decision 3).
 *
 * With 768, all 3 cases (the longest is T=598) fit in one chunk, and two shapes specific to this
 * layer never run on real weights: ① position generation based at `base = context.pastLength`
 * running alongside a second or later chunk; ② the frontier commit at `index === 0` (the prefix of
 * `pendingToken`) running alongside following chunks. The synthetic fake
 * (`generation_sequence_test.ts`) looks at the position lists too, so what is lost is "the recheck
 * on real weights" — but the parity checks ① and ② below compare only means something once it
 * passes through there.
 *
 * NOTE: prefill with a past itself and the sliding ring's wrap-around are covered by
 * `e2e_gemma4_greedy_test.ts` (also in steps of 32, T=598 as 19 chunks), not here. The expected
 * lists were taken by torch's full-length full re-forward and do not depend on the step.
 */
const CHUNK_LENGTH = 32;

/**
 * The prefill buckets ⑤ declares (only the steps below `chunkLength` — ADR 0066 addendum,
 * buckets).
 *
 * The 3 cases' last chunks have 6 / 10 / 22 valid rows, so with this list they land on 8 / 16 / 24
 * rows. **Chosen to straddle 16** — the states-form attention switches to the tiled path (K-13's
 * ①ₜ / ③ₜ) at M ≥ 16, so both sides run once each.
 */
const CHUNK_BUCKETS = [4, 8, 16, 24] as const;

/**
 * The capacity-related run conditions equal the existing checks' (the ruling for the same asset
 * generation is used as is).
 */
const CAPACITY_SYMBOL = "C";
const CAPACITY = 4096;
const MAX_POSITION = 131072;

/**
 * The RoPE parameters (the same values the distribution declares). The tables are no longer in the
 * assets; `ropeInputs` builds cos / sin from the position list (the formula's source of truth is
 * `src/gemma/rope.ts`).
 */
const ROPE: RopeSpec = {
  sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
  full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
};

/**
 * The graph input names (canonical in `export_product`'s constants — `position_ids` is gone).
 */
const INPUT_IDS = "input_ids";
const PER_LAYER_INPUTS = "per_layer_inputs";
const LAST_ROW = "last_row";

const VOCAB = 262144;

/** The distribution mirror: ①〜⑤ read its product container, ⑥ its drafter pipeline. */
const MIRROR_DIR = mirrorRoot("karume-gemma4");
/**
 * The seat ⑥ loads: the regular line's `bf16` preset with its int8 drafter (the same seat as
 * ①〜⑤). Named so a later default does not move it.
 */
const BF16_SEAT = { quant: "bf16" } as const;

const exists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const MODEL_PRESENT = mirrorAvailable(MIRROR_DIR);
/** The goldens of `files` that are absent, as paths under the test-asset root. */
const missingGoldens = (files: readonly URL[]): string[] =>
  files
    .filter((file) => !exists(file))
    .map((file) => file.href.slice(TEST_ASSETS_ROOT.href.length));
const goldenUrl = (name: string): URL => new URL(`${GREEDY_PREFIX}${name}${SUFFIX}`, GOLDEN_ROOT);
const MISSING_GOLDENS = missingGoldens(EXPECTED_CASES.map(({ name }) => goldenUrl(name)));
/** ADR 0139's fetched rule: a missing golden SKIPs only without a complete fetch of the pin. */
const GOLDENS_SKIP = MISSING_GOLDENS.length > 0 && !testAssetsFetched();
const AVAILABLE = MODEL_PRESENT && !GOLDENS_SKIP;

if (!MODEL_PRESENT) {
  console.warn(
    `[karume] the distribution mirror ${MIRROR_DIR.pathname} is absent (or not karume/5), so the ` +
      "Gemma 4 E2B sequence cases ①〜⑤ SKIP",
  );
}

/**
 * Reads one file as an `ArrayBuffer`.
 * MUST: do not slice when the view covers the whole buffer (the goldens here are large).
 */
const readBuffer = async (root: URL, file: string): Promise<ArrayBuffer> => {
  const bytes = await readFile(new URL(file, root));
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return bytes.buffer;
  }
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

const goldenI32 = (file: SafetensorsFile, name: string): Int32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `the golden has no '${name}'`);
  assertEquals(view.dtype, "I32", `storage dtype of golden '${name}'`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

const readCase = async (
  name: string,
): Promise<{ readonly prompt: number[]; readonly expected: number[] }> => {
  // Reached with a golden absent only when the test assets are a complete fetch of the pin.
  const missing = missingGoldens([goldenUrl(name)]);
  assert(
    missing.length === 0,
    `${missing.join(" / ")} absent although outputs/test-assets/ holds a complete fetch of ` +
      `${TEST_ASSETS.repo}@${TEST_ASSETS.revision} — the test-asset dataset at the pin lacks it`,
  );
  const golden = parseSafetensors(
    await readBuffer(GOLDEN_ROOT, `${GREEDY_PREFIX}${name}${SUFFIX}`),
  );
  return {
    prompt: [...goldenI32(golden, "prompt")],
    expected: [...goldenI32(golden, "expected")],
  };
};

const tokenIds = (events: readonly GenerationEvent[]): number[] =>
  events.filter((event) => event.kind === "token").map((event) => event.id);

describe.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  `Gemma 4 E2B sequence acceptance: one-turn parity, multi-turn concatenation, two kinds of stop token (real GPU)${
    GOLDENS_SKIP
      ? ` (SKIP: ${MISSING_GOLDENS.join(" / ")} absent and outputs/test-assets/ is not a complete ` +
        `fetch of the pin — ${FETCH_COMMAND})`
      : ""
  }`,
  () => {
    let parsed: PreparedModel;
    let ple: Gemma4Ple;
    // Undefined until beforeAll creates them: afterAll releases only what was created.
    let gpu: GpuContext | undefined;
    let session: Session | undefined;

    beforeAll(async () => {
      // The container is opened once (the weights and the PLE are both in the same `model`
      // container).
      const opened = await openSelectedContainer(
        await selectMirrorComponent(MIRROR_DIR, MODEL_GRAPH, PRODUCT_CHOICE),
      );
      parsed = prepareContainer(opened, MODEL_GRAPH);

      const { index, openBlock } = await seriesPleHandle(opened, `test: ${MIRROR_DIR.pathname}`);
      ple = createGemma4Ple({
        index,
        openBlock,
        vocabSize: VOCAB,
        // Every block resident (no re-reads across generation round trips) = a budget of the
        // whole PLE.
        maxResidentBytes: gemma4PleTotalBytes(index),
      });

      gpu = await acquireGpu();
      session = await parsed.createContainerSession(gpu);
    });

    afterAll(async () => {
      await session?.dispose();
      gpu?.destroy();
    });

    /**
     * The static wiring (built varying only the stop set and the buckets — the rest is the same
     * wiring of the same asset).
     */
    const programOf = (
      stopTokens: readonly number[],
      chunkBuckets: readonly number[] = [],
    ): GenerationWiring =>
      createGenerationProgram({
        graph: parsed.graph,
        inputIds: INPUT_IDS,
        lastRow: LAST_ROW,
        // Two outputs (the order is the contract — output 0 = logits, output 1 = the hidden state
        // after the final norm).
        logits: parsed.graph.outputs[0],
        hidden: parsed.graph.outputs[1],
        chunkLength: CHUNK_LENGTH,
        chunkBuckets,
        maxPosition: MAX_POSITION,
        capacity: CAPACITY,
        vocabSize: VOCAB,
        stopTokens,
        capacitySymbol: CAPACITY_SYMBOL,
        // Plugs the PLE gather and RoPE into the slot for host-derived per-chunk inputs (ADR 0085 /
        // the variable-capacity iteration — `ple.ts` / `rope.ts` unchanged).
        derivedInputs: {
          names: [PER_LAYER_INPUTS, ...gemma4RopeInputNames()],
          derive: async (ids, positions) => ({
            [PER_LAYER_INPUTS]: await ple.gather(ids),
            ...ropeInputs(ROPE, positions),
          }),
        },
      });

    /**
     * One conversation = one sequence (returned when done, so no context is held on to).
     */
    const withSequence = async <T>(
      stopTokens: readonly number[],
      body: (sequence: GenerationSequence) => Promise<T>,
      chunkBuckets: readonly number[] = [],
    ): Promise<T> => {
      assert(session !== undefined, "the session is created in beforeAll");
      const sequence = await createGenerationSequence({
        session,
        program: programOf(stopTokens, chunkBuckets),
      });
      try {
        return await body(sequence);
      } finally {
        await sequence.dispose();
      }
    };

    it("① a temperature-0 sequence matches the expected lists on 3 cases × 16 steps", async () => {
      /**
       * Whether a prefill split into 2 or more chunks ran at least once (detects a regression of
       * the step — the doc of `CHUNK_LENGTH`).
       */
      let sawMultipleChunks = false;
      for (const { name } of EXPECTED_CASES) {
        const { prompt, expected } = await readCase(name);
        assertEquals(expected.length, GREEDY_STEPS, `${name}: step count of the golden`);

        const started = performance.now();
        const { events, stop } = await withSequence([], async (sequence) => {
          const stream = sequence.generate({ prompt, maxNewTokens: GREEDY_STEPS });
          const collected: GenerationEvent[] = [];
          for await (const event of stream) collected.push(event);
          return { events: collected, stop: await stream.done };
        });

        assertEquals(tokenIds(events), expected, `${name}: token list generated via the sequence`);
        assertEquals(
          stop,
          { reason: "max-tokens", tokens: GREEDY_STEPS },
          `${name}: stop reason and generated token count`,
        );
        // Prefill events follow the chunk split (1 for the two short cases, 19 for the longest).
        const prefills = events.filter((event) => event.kind === "prefill");
        assertEquals(
          prefills.length,
          Math.ceil(prompt.length / CHUNK_LENGTH),
          `${name}: prefill event count`,
        );
        if (prefills.length > 1) sawMultipleChunks = true;
        // Positions continue from the prompt one by one (the sequence keeps no counter; it derives
        // them from the context).
        assertEquals(
          events.filter((event) => event.kind === "token").map((event) => event.position),
          Array.from({ length: GREEDY_STEPS }, (_unused, step) => prompt.length + step),
          `${name}: absolute positions of the token events`,
        );
        console.log(
          `[e2e] gemma4 sequence ${name}: T=${prompt.length} + K=${GREEDY_STEPS} step / ` +
            `${(performance.now() - started).toFixed(0)}ms`,
        );
      }
      // Setting the step back to "a value at which every case fits in one chunk" makes the
      // sequence layer's multi-chunk path this check guards vanish without an exception or a
      // difference (the doc of `CHUNK_LENGTH`). Fail here.
      assert(
        sawMultipleChunks,
        `with step ${CHUNK_LENGTH} every case fits in one chunk (no multi-chunk prefill runs)`,
      );
    });

    it("① topK = 1 also matches greedy on the full-size vocabulary (V = 262,144)", async () => {
      const { name } = EXPECTED_CASES[0];
      const { prompt, expected } = await readCase(name);
      const generated = await withSequence([], async (sequence) => {
        const stream = sequence.generate({
          prompt,
          maxNewTokens: GREEDY_STEPS,
          // Temperature 1 keeps the sampler off the greedy fast path, so every step runs
          // candidate selection over the full logits row. The candidates collapse to the top
          // one, so the seed cannot move the sequence.
          sampler: { temperature: 1, topK: 1, seed: 20_260_831 },
        });
        const ids: number[] = [];
        for await (const event of stream) {
          if (event.kind === "token") ids.push(event.id);
        }
        return ids;
      });
      assertEquals(
        generated,
        expected,
        `${name}: token list generated with topK 1 / temperature 1 (must equal greedy)`,
      );
    });

    it("② break → the continuing second turn matches the reference run exactly (drops no token)", async () => {
      const { name } = EXPECTED_CASES[0];
      const { prompt, expected } = await readCase(name);

      const generated = await withSequence([], async (sequence) => {
        // First turn: break midway (the path into finally via `return()`).
        const first = sequence.generate({ prompt, maxNewTokens: GREEDY_STEPS });
        const head: number[] = [];
        for await (const event of first) {
          if (event.kind !== "token") continue;
          head.push(event.id);
          if (head.length === BREAK_AFTER) break;
        }
        assertEquals(
          await first.done,
          { reason: "closed", tokens: BREAK_AFTER },
          "stop reason and generated token count of the first turn",
        );
        assertEquals(head, expected.slice(0, BREAK_AFTER), "token list of the first turn");

        // Second turn: only the continuation, adding no new token (pendingToken is concatenated
        // at the front).
        const second = sequence.generate({
          prompt: [],
          maxNewTokens: GREEDY_STEPS - BREAK_AFTER,
        });
        const tail: number[] = [];
        for await (const event of second) {
          if (event.kind === "token") tail.push(event.id);
        }
        assertEquals(
          await second.done,
          { reason: "max-tokens", tokens: GREEDY_STEPS - BREAK_AFTER },
          "stop reason and generated token count of the second turn",
        );
        return [...head, ...tail];
      });

      assertEquals(
        generated,
        expected,
        `${name}: concatenated list of interrupt → resume (must equal exactly the reference run ` +
          `streamed in one go — if it differs, pendingToken was dropped or the resume's run shape differs)`,
      );
      console.log(
        `[e2e] gemma4 sequence multi-turn: ${BREAK_AFTER} + ${GREEDY_STEPS - BREAK_AFTER} tokens ` +
          `match the single run`,
      );
    });

    it("③ a program that declares the EOS set stops on the real output", async () => {
      for (const { name, firstStop } of EXPECTED_CASES) {
        const { prompt, expected } = await readCase(name);
        const { events, stop } = await withSequence(STOP_TOKENS, async (sequence) => {
          const stream = sequence.generate({ prompt, maxNewTokens: GREEDY_STEPS });
          const collected: GenerationEvent[] = [];
          for await (const event of stream) collected.push(event);
          return { events: collected, stop: await stream.done };
        });

        // The stop token itself is not emitted as a `token` event (it is a terminator, not text).
        assertEquals(
          tokenIds(events),
          expected.slice(0, firstStop),
          `${name}: token list emitted before the stop`,
        );
        assertEquals(
          stop,
          // The stop token itself counts as one generated token (the doc of
          // `GenerationStop.tokens`) — the text is firstStop tokens.
          { reason: "eos", token: expected[firstStop], tokens: firstStop + 1 },
          `${name}: stop on the EOS set [${STOP_TOKENS.join(",")}]`,
        );
        console.log(`[e2e] gemma4 sequence ${name}: EOS stop @${firstStop}`);
      }
    });

    it("④ a stop token the request adds also stops on the real output (the distribution's set is empty)", async () => {
      // Stops at the same index as ③, but the stop set comes from elsewhere — the program's is
      // **empty** and only the request declares the stop token. If this differs, the union check
      // looks only at the program side.
      for (const { name, firstStop } of EXPECTED_CASES) {
        const { prompt, expected } = await readCase(name);
        const { events, stop } = await withSequence([], async (sequence) => {
          const stream = sequence.generate({
            prompt,
            maxNewTokens: GREEDY_STEPS,
            stopTokens: [expected[firstStop]],
          });
          const collected: GenerationEvent[] = [];
          for await (const event of stream) collected.push(event);
          return { events: collected, stop: await stream.done };
        });

        assertEquals(
          tokenIds(events),
          expected.slice(0, firstStop),
          `${name}: token list emitted before the stop (must be the same list as ③)`,
        );
        assertEquals(
          stop,
          // Only the reason differs from ③ (stopped for this request's own purpose, not on the
          // distribution's terminator).
          { reason: "stop-token", token: expected[firstStop], tokens: firstStop + 1 },
          `${name}: stop on the request's stop token [${expected[firstStop]}]`,
        );
        console.log(`[e2e] gemma4 sequence ${name}: request stop token @${firstStop}`);
      }
    });
    it("⑤ declaring prefill buckets does not move the token lists", async () => {
      // Streams the last chunk with a smaller physical row count and no pad (ADR 0066 addendum,
      // buckets). Row-local ops (linear / pointwise / norm) do not change the valid rows' values
      // when the physical row count changes, but the GEMM geometry changes with M (the thresholds
      // of `gemm-geometry.ts`), so **bit identity is not part of the contract** — what is pinned
      // is the same golden token list as ①.
      let sawBucketRows = false;
      for (const { name } of EXPECTED_CASES) {
        const { prompt, expected } = await readCase(name);
        const tail = prompt.length % CHUNK_LENGTH;
        // Whether the last chunk landed on a bucket (with a step where it does not, this step
        // turns into a re-run of ①).
        const rows = physicalChunkRows(
          tail === 0 ? CHUNK_LENGTH : tail,
          programOf([], CHUNK_BUCKETS),
        );
        // M=1 is the decode form, not a bucket (`physicalChunkRows` always returns 1 for a chunk
        // with one valid row), so it is excluded from the positive control.
        if (rows > 1 && rows < CHUNK_LENGTH) sawBucketRows = true;

        const started = performance.now();
        const { events, stop } = await withSequence(
          [],
          async (sequence) => {
            const stream = sequence.generate({ prompt, maxNewTokens: GREEDY_STEPS });
            const collected: GenerationEvent[] = [];
            for await (const event of stream) collected.push(event);
            return { events: collected, stop: await stream.done };
          },
          CHUNK_BUCKETS,
        );

        assertEquals(tokenIds(events), expected, `${name}: token list generated with buckets`);
        assertEquals(
          stop,
          { reason: "max-tokens", tokens: GREEDY_STEPS },
          `${name}: stop reason and generated token count`,
        );
        // The chunk **split** stays chunkLength (only the row count the last chunk lands on
        // changes).
        assertEquals(
          events.filter((event) => event.kind === "prefill").length,
          Math.ceil(prompt.length / CHUNK_LENGTH),
          `${name}: prefill event count (buckets do not change the split)`,
        );
        console.log(
          `[e2e] gemma4 sequence buckets ${name}: T=${prompt.length} / last chunk ${rows} rows / ` +
            `${(performance.now() - started).toFixed(0)}ms`,
        );
      }
      assert(
        sawBucketRows,
        `no case's last chunk lands on the buckets [${CHUNK_BUCKETS.join(",")}]` +
          ` (this step has become a re-run of ①)`,
      );
    });
  },
);

// ---------------------------------------------------------------------------
// ⑥ Independence of forks (fork — ADR 0066 / 0083 addenda 2026-10-11, ruling 8)
// ---------------------------------------------------------------------------
//
// The order is what matters: a full layer (attention without a window) always reads the rows it
// wrote later on, but never reads rows ahead that it has not written itself. In the order "the
// fork first runs another turn → X on the source", a bug that shares KV may overwrite the source's
// rows, yet X's prefill rewrites them before reading, so the lists still match. The order that
// detects it is "the source advances with X → the fork overwrites **the same positions** with Y →
// the source reads X's rows with Z": with a sharing bug Z's list changes (a missed copy changes
// Y's list). The baseline is built from the same sequence of operations without fork.
//
// NOTE: "bit identity with a re-run" is not looked at, as no check guarantees it. The gated system relies on
// decode (M=1) and verify (M=4) giving the same tokens — the gate picks M=1 or M=4 by wall clock,
// so on a device where the two differ the baseline and fork lists can split (ADR 0066 addendum
// 2026-10-11, the determinism section). On the `bf16` preset read here (the reference summation
// order) that premise is checked by `e2e_gemma4_bf16_mirror_test.ts` ③ (a speculative greedy turn
// equals the plain one); the u32 gate of `e2e_gemma4_speculative_test.ts` ⑤ covers the q4_0 line's
// int4 seats.

/**
 * Whether the mirror is a `karume/5` one with the drafter (⑥'s premise — it runs both speculative
 * modes).
 *
 * MUST: judge by the manifest's declaration, not by file names (as
 * `e2e_gemma4_speculative_test.ts` does).
 */
const MIRROR_WITH_DRAFTER = ((): boolean => {
  if (!mirrorAvailable(MIRROR_DIR)) return false;
  const manifest = parseManifest(readTextFileSync(new URL(MANIFEST_FILENAME, MIRROR_DIR)));
  return Object.hasOwn(manifest.models[manifest.defaultModel].weights, "drafter");
})();

if (!MIRROR_WITH_DRAFTER) {
  console.warn(
    `[karume] the distribution mirror with the drafter (${MIRROR_DIR.pathname}) is absent, so ` +
      "the Gemma 4 E2B fork acceptance SKIPs. Build it: cd tools/export-recipes && uv run python dist.py " +
      "--pipeline gemma4",
  );
}

/**
 * The token count generated per turn (short — what matters is the lists matching, not the
 * acceptance rate).
 *
 * Being short, a turn can stop at the limit before EOS (`<turn|>`, which closes the previous turn).
 * The response is then unclosed, and the next delta is stacked on the continuation of the unclosed
 * response (the last token is prefixed as pendingToken). Unnatural as a conversation, but it holds
 * because the check only compares the lists against the same sequence of operations in the
 * baseline. Which way it stopped is logged as each turn's stop reason.
 */
const FORK_TURN_TOKENS = 12;

/**
 * Sampling at temperature > 0 with a fixed seed (the RNG is not copied into the fork but built from
 * the seed on every `generate` — this also checks that the same spec and the same state give the
 * same list).
 */
const FORK_SAMPLER: SamplerSpec = { temperature: 0.8, topK: 64, seed: 20261011 };

/**
 * The observation of one turn (the token id list and the stop without the speculation accounting —
 * the accounting can vary with wall-clock time).
 */
type ForkTurn = {
  readonly ids: readonly number[];
  readonly stop: Omit<GenerationStop, "speculation">;
};

const forkTurn = async (
  sequence: GenerationSequence,
  prompt: readonly number[],
): Promise<ForkTurn> => {
  const stream = sequence.generate({
    prompt: [...prompt],
    maxNewTokens: FORK_TURN_TOKENS,
    sampler: FORK_SAMPLER,
  });
  const ids: number[] = [];
  for await (const event of stream) {
    if (event.kind === "token") ids.push(event.id);
  }
  const { speculation: _speculation, ...stop } = await stream.done;
  return { ids, stop };
};

describe.skipIf(!MIRROR_WITH_DRAFTER || !GPU_AVAILABLE)(
  "Gemma 4 E2B sequence acceptance ⑥: two forked branches do not share KV (no speculation / always / gated, real GPU)",
  () => {
    // Undefined until beforeAll loads it: afterAll releases only what was created.
    let loaded: Pipeline | undefined;
    let p0: number[];
    let x: number[];
    let y: number[];
    let z: number[];
    const pipelineOf = (): Pipeline => {
      assert(loaded !== undefined, "the pipeline is loaded in beforeAll");
      return loaded;
    };

    beforeAll(async () => {
      const started = performance.now();
      loaded = await Pipeline.fromPretrained(nodeDirectory(MIRROR_DIR), {
        ...BF16_SEAT,
        speculative: { k: GEMMA4_DRAFT_STEPS },
        maxResidentPleBytes: await allResidentPleBytesOfMirror(MIRROR_DIR, BF16_SEAT),
      });
      console.log(
        `[e2e] gemma4 sequence ⑥: pipeline load ${(performance.now() - started).toFixed(0)}ms`,
      );
      const tokenizer = loaded.tokenizer;
      // P0 is a full render, X / Y / Z are deltas (if the previous turn stopped on EOS,
      // pendingToken prefixes the closing marker — it may also have stopped at the limit unclosed
      // ({@link FORK_TURN_TOKENS})). X and Y differ in content, and Z is a question that depends on
      // the previous turn's content (it reads X's rows).
      p0 = chatPrompt(tokenizer, [
        { role: "user", content: "Write one short sentence about the sea." },
      ]);
      x = chatTurn(tokenizer, {
        role: "user",
        content: "Now write one about mountains.",
      });
      y = chatTurn(tokenizer, {
        role: "user",
        content: "Now write one about a desert at night.",
      });
      z = chatTurn(tokenizer, {
        role: "user",
        content: "Repeat your previous sentence word for word.",
      });
    });

    afterAll(async () => {
      await loaded?.dispose();
    });

    const systems = [
      ["no speculation", false],
      ["always", "always"],
      ["gated", true],
    ] as const;
    for (const [label, speculative] of systems) {
      it(`${label}: X on S → Y on the fork → Z on S matches the baseline`, async () => {
        const pipeline = pipelineOf();
        const stepStarted = performance.now();
        /** The sequence of operations without fork (the baseline). */
        const baseline = async (prompts: readonly (readonly number[])[]): Promise<ForkTurn[]> => {
          const sequence = await pipeline.sequence({ speculative });
          try {
            const turns: ForkTurn[] = [];
            for (const prompt of prompts) turns.push(await forkTurn(sequence, prompt));
            return turns;
          } finally {
            await sequence.dispose();
          }
        };
        // Baseline: B = P0 → X → Z, B2 = P0 → Y.
        const [, expectedX, expectedZ] = await baseline([p0, x, z]);
        const [, expectedY] = await baseline([p0, y]);
        // Positive control: Z reads the previous turn's rows (Z of P0 → Y → Z differs from the
        // baseline's Z). If they were equal, the comparison of Z below could not detect the fork's
        // Y overwriting the source's X rows.
        const [, , afterY] = await baseline([p0, y, z]);
        assertNotEquals(
          afterY.ids,
          expectedZ.ids,
          `${label}: Z does not depend on the previous turn (no power to detect an overwrite)`,
        );

        // Under test: S = P0 → F = S.fork() → X on S → Y on F → Z on S.
        const source = await pipeline.sequence({ speculative });
        let forked: GenerationSequence | undefined;
        /**
         * The stop reason of each turn under test (closed on EOS or stopped at the limit — logged).
         */
        const reasons: string[] = [];
        const turn = async (
          name: string,
          sequence: GenerationSequence,
          prompt: readonly number[],
        ): Promise<ForkTurn> => {
          const observed = await forkTurn(sequence, prompt);
          reasons.push(`${name}=${observed.stop.reason}`);
          return observed;
        };
        try {
          await turn("P0", source, p0);
          forked = await source.fork();
          assertEquals(
            forked.used,
            source.used,
            `${label}: the fork did not inherit the conversation's position`,
          );
          assertEquals(
            await turn("X", source, x),
            expectedX,
            `${label}: the source's X differs from the baseline`,
          );
          assertEquals(
            await turn("Y", forked, y),
            expectedY,
            `${label}: the fork's Y differs from the baseline (the copy did not reflect the source's state)`,
          );
          assertEquals(
            await turn("Z", source, z),
            expectedZ,
            `${label}: the source's Z differs from the baseline (the fork's Y overwrote the source's KV)`,
          );
        } finally {
          // The stop reasons are needed precisely when a match check fails (they tell whether the
          // premise "the previous turn was closed" held).
          console.log(`[e2e] gemma4 sequence ⑥ ${label}: stops ${reasons.join(" ")}`);
          await forked?.dispose();
          await source.dispose();
        }
        console.log(
          `[e2e] gemma4 sequence ⑥ ${label}: 3 baselines + 1 fork pair match / ` +
            `${(performance.now() - stepStarted).toFixed(0)}ms`,
        );
      });
    }

    // Placed last (it tears the pipeline down). Pipeline.sequence's wrapper records forks in
    // `#handed` and the pipeline's dispose collects them — a Session's dispose does not tear
    // contexts down, so if this leaks, the fork's KV and the drafter's borrower outlive the
    // pipeline. Building a pipeline needs a real Session and a GPU, so no GPU-free check of this
    // wrapper property can be built and it is checked here.
    it("the pipeline's dispose also collects forks not yet disposed", async () => {
      const pipeline = pipelineOf();
      const source = await pipeline.sequence({ speculative: "always" });
      await forkTurn(source, p0);
      const forked = await source.fork();
      // Hand-outs still allocating when dispose is issued: dispose waits for them before closing
      // the Sessions they borrow (Pipeline.#inFlight), so it resolves, and each of them is closed
      // by the hand-out itself and rejects. Without the wait, the target Session's dispose could run
      // while their contexts still borrow it and dispose would reject with "still lent out".
      const allocating = pipeline.sequence({ speculative: "always" });
      const copying = forked.fork();
      // Both reject while dispose still waits for them, and dispose then awaits GPU work: the
      // handlers are attached before dispose, or Node reports the rejections as unhandled.
      const allocatingRejects = assertRejects(
        () => allocating,
        Error,
        "cannot create a sequence after dispose",
      );
      const copyingRejects = assertRejects(
        () => copying,
        Error,
        "cannot create a sequence after dispose",
      );
      await pipeline.dispose();
      await allocatingRejects;
      await copyingRejects;
      // The fork's sequence is disposed (= it returned the drafter's borrower, then the copied
      // context). Had it not been collected, this would not throw synchronously but fail later
      // in a run against the disposed Session.
      assertThrows(
        () => forked.generate({ prompt: [...x], maxNewTokens: 1 }),
        Error,
        "GenerationSequence: cannot generate after dispose",
      );
      // A fork of the fork after the pipeline's dispose is failed synchronously by the wrapper.
      assertThrows(() => forked.fork(), Error, "cannot create a sequence after dispose");
    });
  },
);
