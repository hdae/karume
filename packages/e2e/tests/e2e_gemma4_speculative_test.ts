// Acceptance check of the **speculative loop** of real-weight Gemma 4 E2B (speculative decoding
// with the MTP drafter) — ADR [0096](../../../docs/decisions/0096-speculative-decoding.md) stage 3.
//
// The stage-2 check (`e2e_gemma4_drafter_test.ts`) goes as far as "the drafter emits the same
// drafts as torch"; this file owns the speculative loop (verify, accept, reject, commit). It looks
// at 5 things:
//
// ① **Identity** (`stateAttentionReduce: "sequential"`) … speculation's only contract is "it only
//    gets faster; the output does not change". 200 tokens are generated from each of the 3 cases'
//    golden prompts, and the token id lists and absolute position lists with and without
//    speculation **match exactly**. A floor is also put on the common prefix with the torch
//    continuation ("speculative == plain" alone lets a regression that moves both sides to another
//    list together pass green). The acceptance rate (token/cycle) gets a floor on the 2 README
//    cases (breaking the floor = a regression of the drafter or of the acceptance decision; the
//    shape "it works but is not faster" is caught here).
//    NOTE: the seat is set to `"sequential"` so that the contract and the floor are taken on the
//    runtime's reference path (decode (M=1) and verify (M=4) both go through the same reduction
//    kernels ① + ③). The default seat's `"parallel-fused"` also has ①′ and fusion apply under
//    M ≤ 8, so decode and verify go through the same reduction order; the M=1 / M=4 match is pinned
//    separately by ② below (token lists) and ⑤ (logits as u32).
//    NOTE: the checks run with `speculative: "always"` (always speculating, no gate) — both the floor
//    and the exact match are contracts of "always speculating". The default self-paying gate
//    (stage 4-B ④, `speculative: true`) is looked at separately by the last step (only the list
//    match is a check; the number of dropped steps depends on wall-clock time, so it is only
//    logged).
// ② **Identity on the default seat** (the fused-parallel seat `i4-fast` (GEMV parallel + RMS→add
//    fusion + attention `"parallel-fused"` — all 3 declared by the quant), named because the q4_0
//    line's default is the reference seat `i4`) … the same 3 cases as ① run on that seat, and the
//    check is that the speculative / plain token lists match (0 differences). The same seat also
//    checks that the list with the self-paying gate (`speculative: true`) equals always — the gate
//    switches between M=1 and M=4 by wall-clock time, so that choice not changing the output (=
//    in-device determinism of the practical tier) holds through the bit identity of these 2 paths
//    (general rule 3 of the ADR 0058 addendum 2026-08-29, ADR 0096 addendum 2026-09-26). Measured
//    on the B570 on 2026-09-26: 0 differences on all 3 cases.
// ③ **Phase observation** (`onRunDiagnostics`) … the run notifications delivered while generating
//    one sequence match `speculation.draftRuns` / `cycles` down to the count, and the plan of the
//    verify form (M=4 / R=4) **stays** in the LRU from the second cycle on
//    (`lastRunPrepared.hit`). The same window also checks that the state bind group rebuilds
//    (`stateBacking.rebindCount`) do not grow in proportion to the cycle count.
// ④ **Estimate** (`estimateSessionMemory`) … the resident weights **match exactly** the
//    diagnostics of the 2 Sessions target + drafter, the state is the lender's share + the
//    borrower's 8-byte logical-length uniform, and the verify scenario's `ioBytes` is larger than
//    decode's by exactly "3 more rows". Forgetting the drafter in the accounting breaks all of them.
// ⑤ **Kernel equivalence** (u32) … the very premise for the identity of ① and ②: the logits of a
//    row with the same token at the same position, run in verify form (row 0 of an M=4 deferred
//    run) and in decode form (an M=1 immediate run), **match exactly as u32**. This check separates
//    "a defect in the acceptance decision" from "a numeric difference of the kernels" when ① or ②
//    breaks, and it drives the `Session` directly (the generation surface does not expose
//    `commit(0)`). All 12 combinations of the seat — 3 attention variants × 2 GEMV kinds × RMS→add
//    fusion — (and the linear→SRQ setting) are pinned by the same u32 check (all combinations
//    matched in the B570 measurement of 2026-09-26).
//
// ## Assets
//
// The q4_0 QAT line's mirror `models/karume-gemma4-qat/` (E2B with its int4 QAT drafter — the
// `gemma4` pipeline's int4 seats since the regular line ships `bf16` alone, 2026-11-04) and the QAT
// drafter goldens in the test-asset root, `goldens/gemma4-e2b-qat-drafter/` (the prompts and the
// torch continuations — ADR 0139 decision 2). Both are
// outside the repository, and without the mirror every case SKIPs explicitly. The goldens follow ADR 0139's
// fetched rule: a golden absent from a complete fetch of the pinned test-asset dataset FAILs the
// cases that read it (the dataset at the pin lacks it); without a complete fetch those cases SKIP
// with the missing files and `node tools/test-assets/fetch.ts` in their titles.

import { isNotFound, readFile, readTextFileSync, statSync } from "@karume/test-support/fs";
import { assert, assertEquals } from "@std/assert";
import { type Manifest, MANIFEST_FILENAME, parseManifest } from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";
import {
  acquireGpu,
  type GenerationContext,
  type GpuContext,
  parseSafetensors,
  type SafetensorsFile,
  type Session,
  type SessionDiagnostics,
  type StateAttentionReduce,
  type Tensor,
} from "@karume/runtime";
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { Pipeline, type GenerationRunPhase, type GenerationStop } from "../../models/gemma.ts";
import { type PipelineConfig, parsePipelineConfig } from "../../models/src/gemma/config.ts";
import { createGemma4Ple, type Gemma4Ple } from "../../models/src/gemma/ple.ts";
import { ropeInputs, type RopeSpec } from "../../models/src/gemma/rope.ts";
import { GEMMA4_DRAFT_STEPS } from "../../models/src/gemma/speculative.ts";
import type { SamplerSpec } from "../../models/src/generation/sampler.ts";
import {
  GEMMA4_QAT_MIRROR,
  mirrorAvailable,
  openGemma4Ple,
  openMirrorComponent,
} from "../../models/tests/helpers/gemma-mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  goldenDir,
  TEST_ASSETS,
  TEST_ASSETS_ROOT,
  testAssetsFetched,
} from "@karume/test-support/test-assets";
import {
  allResidentBytes,
  allResidentPleBytesOfMirror,
} from "../../models/tests/helpers/ple-budget.ts";

const MIRROR_DIR = GEMMA4_QAT_MIRROR;
/**
 * The seat these cases read: the q4_0 line's fused-parallel `i4-fast` over its E2B (the mirror's
 * default model) and its QAT drafter — the seat the regular line's `i4-fast` was before the regular
 * line went `bf16`-only (2026-11-04). Named because the line's default is the reference `i4`.
 */
const INT4_SEAT = { quant: "i4-fast" } as const;
const GOLDEN_ROOT = goldenDir("gemma4-e2b-qat-drafter");

/** The build command, ready to paste on a SKIP (the same text as the stage-2 check). */
const ASSEMBLE_COMMAND =
  "cd tools/export-recipes && uv run python dist.py --pipeline gemma4-qat" +
  " (the drafter series and the goldens: … python -m gemma4.export_drafter_qat)";

/** The golden cases (canonical in `export_drafter.py`'s GOLDEN_CASES). */
const CASES = ["short-en", "readme-recipes", "readme-exporter"] as const;
type CaseName = (typeof CASES)[number];
const GOLDEN_PREFIX = "drafter-golden.";
const SUFFIX = ".safetensors";

/**
 * The token count generated per case (long enough to build the acceptance rate's denominator up
 * to a practical range).
 */
const MAX_NEW_TOKENS = 200;

/**
 * The floor of the acceptance rate (token/cycle = `delivered / cycles` — the note to ADR 0096
 * decision 7 is the source of truth).
 *
 * MUST: **do not fit it to the measurement**. The value is a regression line placed **below** the
 * end-to-end measurement when stage 3 landed (2.010 / 2.140 token/cycle on the 2 README cases,
 * 2026-09-08, RTX 3080 Ti); it breaks when the drafter's quality or the acceptance decision
 * regresses (a different quantity from the golden side's teacher-forcing values 2.125 / 2.110).
 * `short-en` (a short 25-token text) has too much variance for one case, so it is only logged (no
 * check).
 *
 * Re-collected 2026-11-04 on the RTX 3080 Ti (Node + Dawn) when the gate moved from the regular
 * `i4-fast` + int8 drafter to the q4_0 `i4-fast` + QAT drafter (a new target): 2.095 / 2.117
 * token/cycle on the README cases (short-en 1.605, log only). The same rule as the first floor —
 * below the smaller measurement, rounded down to one decimal — gives 1.9 again, so the value stays.
 *
 * NOTE: the baseline is a measurement on the **RTX 3080 Ti / Vulkan backend**. Even on the
 * `"sequential"` seat, ① does not pin the reduction's **execution order** (the grid-stride split
 * changes with the device's limits), so on another GPU / driver the accepted lists can shift and
 * break the floor — that is an environment difference, not a regression, so lower the floor.
 */
const ACCEPTANCE_FLOOR: Partial<Record<CaseName, number>> = {
  "readme-recipes": 1.9,
  "readme-exporter": 1.9,
};

/**
 * The short turn ③ and ④ run (they only compare the phase list and the estimate, so no length is
 * needed).
 */
const PHASE_TOKENS = 40;

/**
 * The real asset's shape (the config's `hidden_size`; the lender's embedding table is borrowed by
 * the borrower).
 */
const HIDDEN = 1536;
const VOCAB = 262144;

/**
 * ⑤'s prefill step (this check only feeds a 25-token prompt as one chunk, so the distribution's
 * 768 would make the pad rows 30 times as many).
 */
const CHUNK_LENGTH = 32;
/** The physical row count of a verify run (`k+1` — ⑤ declares it as a bucket). */
const VERIFY_ROWS = GEMMA4_DRAFT_STEPS + 1;

/** The graph input names (canonical in `export_product.py`'s constants). */
const INPUT_IDS = "input_ids";
const PER_LAYER_INPUTS = "per_layer_inputs";
const LAST_ROW = "last_row";
const CAPACITY_SYMBOL = "C";

/**
 * The bytes a borrower context holds on the GPU (only the logical-length uniform — borrowed slots
 * allocate no byte at all). The source of truth is the runtime's `LENGTHS_BYTES` (copied here, as
 * it is not on the public surface).
 */
const LENGTHS_BYTES = 16;

const exists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const goldenPath = (name: string): URL => new URL(`${GOLDEN_PREFIX}${name}${SUFFIX}`, GOLDEN_ROOT);

/**
 * Whether the mirror is a `karume/5` one with the drafter.
 *
 * MUST: judge by **the manifest's declaration**, not by file names — a container part's path is
 * the writer's convention (`<stem>-NNNNN-of-NNNNN.krm`) and changes when the part count changes
 * (ADR 0109 decision 3).
 */
const MIRROR_PRESENT = ((): boolean => {
  if (!mirrorAvailable(MIRROR_DIR)) return false;
  const manifest = parseManifest(readTextFileSync(new URL(MANIFEST_FILENAME, MIRROR_DIR)));
  return Object.hasOwn(manifest.models[manifest.defaultModel].weights, "drafter");
})();
/** The goldens of `files` that are absent, as paths under the test-asset root. */
const missingGoldens = (files: readonly URL[]): string[] =>
  files
    .filter((file) => !exists(file))
    .map((file) => file.href.slice(TEST_ASSETS_ROOT.href.length));
const MISSING_GOLDENS = missingGoldens(CASES.map(goldenPath));
/** ADR 0139's fetched rule: a missing golden SKIPs only without a complete fetch of the pin. */
const GOLDENS_SKIP = MISSING_GOLDENS.length > 0 && !testAssetsFetched();
const AVAILABLE = MIRROR_PRESENT && !GOLDENS_SKIP;
/** The title suffix that names a golden SKIP (empty otherwise). */
const GOLDENS_SKIP_REASON = GOLDENS_SKIP
  ? ` (SKIP: ${MISSING_GOLDENS.join(" / ")} absent and outputs/test-assets/ is not a complete ` +
    "fetch of the pin — node tools/test-assets/fetch.ts)"
  : "";

if (!MIRROR_PRESENT) {
  console.warn(
    `[karume] the distribution mirror with the drafter (${MIRROR_DIR.pathname}) is absent, so the speculative loop` +
      ` acceptance SKIPs. Build it: ${ASSEMBLE_COMMAND}`,
  );
}

const goldenI32 = (file: SafetensorsFile, name: string): Int32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `the golden has no '${name}'`);
  assertEquals(view.dtype, "I32", `storage dtype of golden '${name}'`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

/**
 * One golden case (`prompt` and the torch greedy continuation — the continuation is a reference
 * value, not a check).
 */
type GoldenCase = {
  readonly prompt: number[];
  readonly continuation: number[];
};

const readGoldenCase = async (name: string): Promise<GoldenCase> => {
  // Reached with a golden absent only when the test assets are a complete fetch of the pin.
  assert(
    missingGoldens([goldenPath(name)]).length === 0,
    `${goldenPath(name).href.slice(TEST_ASSETS_ROOT.href.length)} absent although ` +
      `outputs/test-assets/ holds a complete fetch of ${TEST_ASSETS.repo}@${TEST_ASSETS.revision} — ` +
      "the test-asset dataset at the pin lacks it",
  );
  const bytes = await readFile(goldenPath(name));
  const file = parseSafetensors(
    bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
      ? bytes.buffer
      : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  return {
    prompt: [...goldenI32(file, "prompt")],
    continuation: [...goldenI32(file, "tokens")],
  };
};

/** The mirror's manifest / `pipelineConfig` (where RoPE and the capacity come from). */
const mirrorManifest = (): Manifest =>
  parseManifest(readTextFileSync(new URL(MANIFEST_FILENAME, MIRROR_DIR)));

const pipelineConfig = () => {
  const manifest = mirrorManifest();
  return parsePipelineConfig(manifest.models[manifest.defaultModel].pipelineConfig);
};

// ---------------------------------------------------------------------------
// One run (open a sequence, draw one turn, close it)
// ---------------------------------------------------------------------------

/**
 * The observation of one turn (token id list, absolute position list, stop reason, duration).
 */
type Turn = {
  readonly ids: number[];
  readonly positions: number[];
  readonly stop: GenerationStop;
  readonly ms: number;
};

const runTurn = async (
  pipeline: Pipeline,
  prompt: readonly number[],
  options: {
    /**
     * `"always"` = always speculating, no gate (the checks use this — both the floor and the
     * exact match are contracts of "always speculating"). `true` is the default seat with the
     * self-paying gate; only ①'s last step runs it.
     */
    readonly speculative: boolean | "always";
    readonly capacity?: number;
    readonly tokens: number;
    readonly sampler?: SamplerSpec;
  },
): Promise<Turn> => {
  const sequence = await pipeline.sequence({
    speculative: options.speculative,
    ...(options.capacity === undefined ? {} : { capacity: options.capacity }),
  });
  try {
    const started = performance.now();
    // The 3-case comparison passes no sampler (the lower layer's default = temperature 0).
    // Speculation is set up whatever the temperature, so the speculative / plain lists match even
    // with sampling, but ① also compares against the torch continuation — the RNG is splitmix64,
    // a different thing from torch's Philox (the same seed does not give the same list), so parity
    // can only be taken at temperature 0. Identity at temperature > 0 is checked separately with
    // one run of the recommended sampler.
    const stream = sequence.generate({
      prompt: [...prompt],
      maxNewTokens: options.tokens,
      ...(options.sampler === undefined ? {} : { sampler: options.sampler }),
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

/**
 * The capacity this conversation needs (raised to the lower bound, as less than `chunkLength`
 * cannot be given).
 */
const capacityFor = (pipeline: Pipeline, promptLength: number): number =>
  Math.max(pipeline.program.chunkLength, promptLength + MAX_NEW_TOKENS);

/** The common prefix length of two lists. */
const commonPrefix = (left: readonly number[], right: readonly number[]): number => {
  let index = 0;
  while (index < left.length && index < right.length && left[index] === right[index]) index += 1;
  return index;
};

/**
 * The prefix length required to match the torch continuation (①'s **external** reference point).
 *
 * ①'s main check "speculative == plain" cannot catch a regression that moves speculative and plain
 * **together** to another list (a regression on the target side, a PLE / RoPE mix-up) — if both
 * break by the same amount, they still match. Also checking that the prefix matches torch's greedy
 * continuation catches that shape.
 *
 * MUST: do not fit it to the measurement (the 3 cases matched 200/200 on 2026-09-08). The i4
 * target is not bit-equivalent to torch, so a long list can eventually split midway — the floor is
 * placed where it only catches the "switched to another list" class. Re-measured 2026-11-04 on the new target (q4_0 E2B against its QAT drafter
 * golden, whose continuation is the torch QAT target's): 200 / 200 on all 3 cases, so 16 stays.
 */
const GOLDEN_PREFIX_FLOOR = 16;

/**
 * Turns the speculation accounting into "tokens committed per cycle" (= `delivered / cycles` — the
 * note to ADR 0096 decision 7 is the source of truth).
 *
 * The old form `(accepted + cycles) / cycles` assumes "a cycle always commits 1 + a tokens", but a
 * cycle whose accepted draft was a stop token cuts the enumeration there and commits only a — the
 * numerator is 1 too large for each such cycle.
 */
const tokensPerCycle = (stop: GenerationStop): number => {
  const speculation = stop.speculation;
  assert(speculation !== undefined, `a turn without speculation: ${JSON.stringify(stop)}`);
  // A turn where no cycle ran cannot be divided (do not let a division by 0 slip through as NaN).
  assert(speculation.cycles > 0, `a turn where no cycle ran: ${JSON.stringify(stop)}`);
  return speculation.delivered / speculation.cycles;
};

// ---------------------------------------------------------------------------
// ① Identity (sequential) — the lists with and without speculation match exactly
// ---------------------------------------------------------------------------

describe.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  `gemma4 speculation ①: the speculative / plain token lists match exactly on the sequential seat (real GPU)${GOLDENS_SKIP_REASON}`,
  () => {
    let pipeline: Pipeline;

    beforeAll(async () => {
      const started = performance.now();
      pipeline = await Pipeline.fromPretrained(nodeDirectory(MIRROR_DIR), {
        ...INT4_SEAT,
        speculative: { k: GEMMA4_DRAFT_STEPS },
        // The reference path (the runtime's default) — the seat where decode and verify go
        // through the same reduction kernels.
        stateAttentionReduce: "sequential",
        maxResidentPleBytes: await allResidentPleBytesOfMirror(MIRROR_DIR, INT4_SEAT),
      });
      console.log(
        `[e2e] gemma4 speculation ①: pipeline load ${(performance.now() - started).toFixed(0)}ms`,
      );
    });

    afterAll(async () => {
      // Undefined when the load in beforeAll threw.
      await pipeline?.dispose();
    });

    for (const name of CASES) {
      it(`${name}: ${MAX_NEW_TOKENS} tokens speculative / plain`, async () => {
        const golden = await readGoldenCase(name);
        const capacity = capacityFor(pipeline, golden.prompt.length);
        const speculative = await runTurn(pipeline, golden.prompt, {
          speculative: "always",
          capacity,
          tokens: MAX_NEW_TOKENS,
        });
        const plain = await runTurn(pipeline, golden.prompt, {
          speculative: false,
          capacity,
          tokens: MAX_NEW_TOKENS,
        });

        // Speculation's only contract (it only gets faster; the list does not change).
        assertEquals(speculative.ids, plain.ids, `${name}: speculative / plain token id lists`);
        assertEquals(speculative.positions, plain.positions, `${name}: absolute position lists`);
        assertEquals(speculative.stop.reason, plain.stop.reason, `${name}: stop reason`);
        assertEquals(speculative.stop.tokens, plain.stop.tokens, `${name}: generated token count`);
        // A plain sequence has no accounting field at all (if the field appeared, it went through
        // the speculative path).
        assertEquals(
          plain.stop.speculation,
          undefined,
          `${name}: speculation field of the plain turn`,
        );

        const perCycle = tokensPerCycle(speculative.stop);
        const speculation = speculative.stop.speculation;
        assert(speculation !== undefined, "the speculative turn carries no accounting");
        const goldenPrefix = commonPrefix(speculative.ids, golden.continuation);
        console.log(
          `[e2e] gemma4 speculation ① ${name}: T=${golden.prompt.length} / ${speculative.ids.length} ` +
            `token / ${speculation.cycles} cycle / accepted ${speculation.accepted}/` +
            `${speculation.drafted} / ${perCycle.toFixed(3)} token/cycle / histogram [${speculation.acceptedHistogram.join(
              ",",
            )}] / speculative ${speculative.ms.toFixed(0)}ms vs plain ${plain.ms.toFixed(0)}ms / ` +
            `common prefix with the torch continuation ${goldenPrefix}`,
        );

        // External reference point (catches the regression of both sides together that ①'s
        // "speculative == plain" alone cannot).
        assert(
          goldenPrefix >= GOLDEN_PREFIX_FLOOR,
          `${name}: the common prefix with the torch continuation is only ${goldenPrefix} tokens (floor ` +
            `${GOLDEN_PREFIX_FLOOR}) — speculative and plain have moved to another list together`,
        );

        const floor = ACCEPTANCE_FLOOR[name];
        if (floor !== undefined) {
          assert(
            perCycle >= floor,
            `${name}: acceptance rate ${perCycle.toFixed(3)} token/cycle is below the floor ${floor.toFixed(2)}` +
              ` (accepted ${speculation.accepted} / ${speculation.cycles} cycles, histogram [${speculation.acceptedHistogram.join(
                ",",
              )}])`,
          );
        }
      });
    }

    it("readme-recipes (the distribution's recommended sampler, temperature 1): the speculative / plain lists are identical", async () => {
      // Speculation is set up whatever the temperature — acceptance sampling calls
      // `sampler.next` per row with the same logits, history and order as plain decoding, so even
      // the RNG consumption matches. Parity with torch cannot be taken (the RNG differs), but the
      // speculative / plain identity is checked here. The sampler is the distribution's
      // declaration (the recommended values) itself.
      const sampler: SamplerSpec = { ...pipelineConfig().sampler, seed: 1 };
      assert(
        (sampler.temperature ?? 0) > 0,
        "the distribution's recommended sampler is at temperature 0 (this check's premise is broken)",
      );
      const golden = await readGoldenCase("readme-recipes");
      const capacity = capacityFor(pipeline, golden.prompt.length);
      const speculative = await runTurn(pipeline, golden.prompt, {
        speculative: "always",
        capacity,
        tokens: MAX_NEW_TOKENS,
        sampler,
      });
      const plain = await runTurn(pipeline, golden.prompt, {
        speculative: false,
        capacity,
        tokens: MAX_NEW_TOKENS,
        sampler,
      });
      assertEquals(speculative.ids, plain.ids, "temperature 1: speculative / plain token id lists");
      assertEquals(
        speculative.positions,
        plain.positions,
        "temperature 1: absolute position lists",
      );
      assertEquals(
        speculative.stop.tokens,
        plain.stop.tokens,
        "temperature 1: generated token count",
      );
      // Sampling actually runs (if the list matched the temperature-0 list over its own length,
      // the RNG was not consumed). NOTE: a raw-text continuation at temperature 1 draws a stop
      // token early (3 tokens in the 2026-09-08 measurement), so the acceptance rate is not looked
      // at here — the acceptance rate with the recommended sampler is a stage-4 measurement item.
      const greedy = await runTurn(pipeline, golden.prompt, {
        speculative: false,
        capacity,
        tokens: MAX_NEW_TOKENS,
      });
      assert(
        commonPrefix(plain.ids, greedy.ids) < plain.ids.length,
        "the temperature-1 list equals the temperature-0 list (sampling has no effect)",
      );
      const speculation = speculative.stop.speculation;
      assert(speculation !== undefined && speculation.cycles >= 1, "speculation is not set up");
      console.log(
        `[e2e] gemma4 speculation ① readme-recipes (temperature ${sampler.temperature}, topK ${sampler.topK}, ` +
          `topP ${sampler.topP}): ${plain.ids.length} token / ${speculation.cycles} cycle / ${tokensPerCycle(
            speculative.stop,
          ).toFixed(
            3,
          )} token/cycle / common prefix with temperature 0 ${commonPrefix(plain.ids, greedy.ids)}`,
      );
    });

    it("with the self-paying gate (the default speculative: true): emits the same list as always", async () => {
      // The gate picks "speculative / decode form" per cycle by wall-clock time. On the
      // sequential seat the two are bit-identical (⑤'s check), so wherever it switches, the list
      // does not change — if this differs, the switch itself (the hidden hand-over, the frontier
      // commit) is broken.
      const golden = await readGoldenCase("readme-recipes");
      const capacity = capacityFor(pipeline, golden.prompt.length);
      const gated = await runTurn(pipeline, golden.prompt, {
        speculative: true,
        capacity,
        tokens: MAX_NEW_TOKENS,
      });
      const always = await runTurn(pipeline, golden.prompt, {
        speculative: "always",
        capacity,
        tokens: MAX_NEW_TOKENS,
      });
      assertEquals(gated.ids, always.ids, "gated / always token id lists");
      assertEquals(gated.positions, always.positions, "absolute position lists");
      assertEquals(gated.stop.tokens, always.stop.tokens, "generated token count");
      const speculation = gated.stop.speculation;
      assert(speculation !== undefined, "the gated turn carries no accounting");
      // MUST: the number of dropped steps is **not a check** (it depends on wall-clock time and
      // changes with the host and the load). This only records how it came out on this machine.
      console.log(
        `[e2e] gemma4 speculation ① gate: ${gated.ids.length} token / ${speculation.cycles} cycle / ` +
          `plain step ${speculation.plainSteps} / switches ${speculation.switches} / ` +
          `gated ${gated.ms.toFixed(0)}ms vs always ${always.ms.toFixed(0)}ms`,
      );
    });
  },
);

// ---------------------------------------------------------------------------
// ②③④ Default seat (i4-fast = GEMV parallel + RMS→add fusion + attention parallel-fused) — exact checks of identity, phase observation and the estimate
// ---------------------------------------------------------------------------

/**
 * The observation of one run (the phase + the diagnostics read in that run's synchronous section).
 */
type PhaseRecord = {
  readonly phase: GenerationRunPhase;
  readonly preparedHit: boolean | undefined;
  readonly rebindCount: number;
  readonly residentCompressedBytes: number;
  readonly stateResidentBytes: number;
};

const recordOf = (diagnostics: SessionDiagnostics, phase: GenerationRunPhase): PhaseRecord => ({
  phase,
  preparedHit: diagnostics.lastRunPrepared?.hit,
  rebindCount: diagnostics.stateBacking.rebindCount,
  residentCompressedBytes: diagnostics.storage.residentCompressedBytes,
  stateResidentBytes: diagnostics.stateBacking.residentBytes,
});

/** Picks one kind of phase out of the records. */
const ofKind = (records: readonly PhaseRecord[], kind: GenerationRunPhase["kind"]) =>
  records.filter((record) => record.phase.kind === kind);

describe.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  `gemma4 speculation ②③④: identity on the default seat, phase counts, the estimate's sum (real GPU)${GOLDENS_SKIP_REASON}`,
  () => {
    /** The window that collects observations (filled only while a step keeps it open). */
    let records: PhaseRecord[] | undefined;
    let pipeline: Pipeline;
    const k = GEMMA4_DRAFT_STEPS;

    beforeAll(async () => {
      const started = performance.now();
      pipeline = await Pipeline.fromPretrained(nodeDirectory(MIRROR_DIR), {
        ...INT4_SEAT,
        speculative: { k: GEMMA4_DRAFT_STEPS },
        maxResidentPleBytes: await allResidentPleBytesOfMirror(MIRROR_DIR, INT4_SEAT),
        onRunDiagnostics: (diagnostics, phase) => {
          records?.push(recordOf(diagnostics, phase));
        },
      });
      console.log(
        `[e2e] gemma4 speculation ②: pipeline load ${(performance.now() - started).toFixed(0)}ms`,
      );
    });

    afterAll(async () => {
      // Undefined when the load in beforeAll threw.
      await pipeline?.dispose();
    });

    it("② the speculative / plain token lists match on the default seat (i4-fast)", async () => {
      for (const name of CASES) {
        const golden = await readGoldenCase(name);
        const capacity = capacityFor(pipeline, golden.prompt.length);
        const speculative = await runTurn(pipeline, golden.prompt, {
          speculative: "always",
          capacity,
          tokens: MAX_NEW_TOKENS,
        });
        const plain = await runTurn(pipeline, golden.prompt, {
          speculative: false,
          capacity,
          tokens: MAX_NEW_TOKENS,
        });
        // On the default seat too, M=1 (decode) and M=4 (verify) are bit-identical (⑤'s check), so
        // the list match is a check as in ① — this is the premise that the self-paying gate's
        // wall-clock choice does not change the output.
        assertEquals(
          speculative.ids.length,
          MAX_NEW_TOKENS,
          `${name}: token count, speculative side`,
        );
        assertEquals(plain.ids.length, MAX_NEW_TOKENS, `${name}: token count, plain side`);
        const first = commonPrefix(speculative.ids, plain.ids);
        let diverged = 0;
        for (let index = 0; index < MAX_NEW_TOKENS; index += 1) {
          if (speculative.ids[index] !== plain.ids[index]) diverged += 1;
        }
        console.log(
          `[e2e] gemma4 speculation ② ${name}: ${diverged}/${MAX_NEW_TOKENS} differ, first mismatch ` +
            `${first === MAX_NEW_TOKENS ? "none" : `@${first}`} / ${tokensPerCycle(
              speculative.stop,
            ).toFixed(3)} token/cycle / speculative ${speculative.ms.toFixed(0)}ms vs plain ` +
            `${plain.ms.toFixed(0)}ms`,
        );
        // MUST: log before the check (so that the position and count of the differences are on
        // record even on a machine where it breaks).
        assertEquals(
          diverged,
          0,
          `${name}: on the default seat the speculative / plain token lists differ at ${diverged}/${MAX_NEW_TOKENS}` +
            ` (first @${first}) — M=1 and M=4 are not bit-identical on this machine`,
        );
        assertEquals(first, MAX_NEW_TOKENS, `${name}: common prefix of speculative / plain`);
      }
    });

    it("② with the self-paying gate on the default seat (speculative: true): emits the same list as always", async () => {
      // The gate picks "speculative / decode form" per cycle by wall-clock time. On the default
      // seat too the two are bit-identical (⑤'s check), so wherever it switches, the list does not
      // change — if this differs, the switch itself (the hidden hand-over, the frontier commit) or
      // the default seat's M=1 / M=4 identity is broken.
      const golden = await readGoldenCase("readme-recipes");
      const capacity = capacityFor(pipeline, golden.prompt.length);
      const gated = await runTurn(pipeline, golden.prompt, {
        speculative: true,
        capacity,
        tokens: MAX_NEW_TOKENS,
      });
      const always = await runTurn(pipeline, golden.prompt, {
        speculative: "always",
        capacity,
        tokens: MAX_NEW_TOKENS,
      });
      assertEquals(gated.ids, always.ids, "default seat: gated / always token id lists");
      assertEquals(gated.positions, always.positions, "default seat: absolute position lists");
      assertEquals(gated.stop.tokens, always.stop.tokens, "default seat: generated token count");
      const speculation = gated.stop.speculation;
      assert(speculation !== undefined, "the gated turn carries no accounting");
      // MUST: the number of dropped steps is **not a check** (it depends on wall-clock time and
      // changes with the host and the load). This only records how it came out on this machine.
      console.log(
        `[e2e] gemma4 speculation ② gate: ${gated.ids.length} token / ${speculation.cycles} cycle / ` +
          `plain step ${speculation.plainSteps} / switches ${speculation.switches} / ` +
          `gated ${gated.ms.toFixed(0)}ms vs always ${always.ms.toFixed(0)}ms`,
      );
    });

    // ③ and ④ share the observation of one turn (the estimate is compared with "what that run
    // actually held resident", so the capacity stays the default = the same shape
    // `estimateSessionMemory` estimates by default).
    // The turn ran between ② and ③ in the former test body. It is taken lazily by whichever of
    // ③ / ④ runs first, so it still runs after ② (not in beforeAll, which would move it before ②
    // and change the pipeline state the observation sees), and a failure fails both ③ and ④.
    let observation: Promise<{ turn: Turn; observed: PhaseRecord[] }> | undefined;
    const observeTurn = () =>
      (observation ??= (async () => {
        const golden = await readGoldenCase("short-en");
        const observed: PhaseRecord[] = [];
        records = observed;
        const turn = await runTurn(pipeline, golden.prompt, {
          speculative: "always",
          tokens: PHASE_TOKENS,
        });
        records = undefined;
        return { turn, observed };
      })());

    it("③ the count and kinds of run notifications match the speculation accounting", async () => {
      const { turn, observed } = await observeTurn();
      const speculation = turn.stop.speculation;
      assert(speculation !== undefined, "speculation is not set up");
      const drafts = ofKind(observed, "draft");
      const verifies = ofKind(observed, "verify");
      const prefills = ofKind(observed, "prefill");
      // Not a single plain decode run is mixed in (a speculative turn's decode is emitted as a
      // `k'=0` verify — if kinds mix, "a cycle that dropped to decode while speculating" goes
      // silent).
      //
      // MUST: these 0 runs hold only for the `speculative: "always"` (no gate) turn above. With
      // the default `auto`, plain steps the self-paying gate drops are **legitimately** emitted as
      // `decode` notifications (count = `speculation.plainSteps`; the run's shape is M=1, R=1,
      // decode itself, so it names the same branch), so neither this check nor the total formula
      // below (which has no `plainSteps` term) applies to a gated turn.
      assertEquals(
        ofKind(observed, "decode").length,
        0,
        "a decode notification got mixed into the speculative turn",
      );
      assertEquals(drafts.length, speculation.draftRuns, "draft notification count = draftRuns");
      assertEquals(verifies.length, speculation.cycles, "verify notification count = cycles");
      assertEquals(
        observed.length,
        prefills.length + speculation.cycles + speculation.draftRuns,
        "total notification count (prefill chunks + cycles + draft runs)",
      );
      // Cycles are numbered consecutively from 1 (a cycle's draft and verify carry the same
      // number).
      assertEquals(
        verifies.map((record) => (record.phase.kind === "verify" ? record.phase.cycle : -1)),
        Array.from({ length: speculation.cycles }, (_unused, index) => index + 1),
        "verify cycle numbers",
      );
      // The sum of accepted counts matches the accounting (the `accepted` the phases carry and
      // the sum in `GenerationStop` are not counted separately).
      const accepted = verifies.reduce(
        (sum, record) => sum + (record.phase.kind === "verify" ? record.phase.accepted : 0),
        0,
      );
      assertEquals(accepted, speculation.accepted, "sum of the phases' accepted");

      // The plan of the verify form (M=k+1 / R=k+1) stays in the LRU from the second one on (if it
      // fell back to derivation every cycle, plan derivation would return to the decode hot path
      // without an exception).
      // MUST: like the rebind check below, only the verifies on the k+1-row bucket are held to it.
      // A `k' = 0` cycle at the end of the budget is a decode-form run (M = 1), a plan this context
      // derives the first time it meets it, and whether the turn ends there depends on the
      // acceptance series (GPU / driver / target). The q4_0 E2B's turn of 2026-11-04 did: verify
      // rows [4 × 24, 3, 2, 1], hit on every one but the first and the closing M = 1 run.
      const bucketedAfterFirst = verifies
        .slice(1)
        .filter((record) => record.phase.kind === "verify" && record.phase.rows > 1);
      const missed = bucketedAfterFirst.filter((record) => record.preparedHit !== true).length;
      assertEquals(
        missed,
        0,
        `of the ${bucketedAfterFirst.length} M=k+1 verifies from the second on, ${missed} missed the PreparedPlan` +
          ` (hit: [${verifies.map((record) => String(record.preparedHit)).join(",")}])`,
      );
      // State bind group rebuilds are **not proportional to the cycle count**. A rebuild happens
      // only when the (context, backing) pair changes, and over a stretch of verifies with the
      // same physical row count it does not increase once.
      const rowsSeries = verifies.map((record) =>
        record.phase.kind === "verify" ? record.phase.rows : -1,
      );
      // MUST: what is pinned is "no rebuild **over the stretch of M=k+1 verifies**". Only the
      // `k' = 0` cycle at the end of the budget becomes a decode-form (M=1) run, and if that
      // physical row count is new to this context the bind group is rebuilt once — whether the
      // end lands on `k'=0` is decided by the accepted list (GPU / driver dependent), so mixing it
      // in makes a check that breaks on environment differences.
      const bucketed = verifies.filter(
        (record) => record.phase.kind === "verify" && record.phase.rows > 1,
      );
      const rebinds = bucketed.map((record) => record.rebindCount);
      // In the measurement (2026-09-08) a single rebuild for the switch from prefill form to
      // verify form appears by the second verify and it never moves after that (a cycle where
      // `k'` shrinks to 2 / 1 only has fewer valid rows; the physical row count stays the bucket
      // `k+1`, so no rebuild happens).
      assert(
        bucketed.length >= 4,
        `${bucketed.length} M=k+1 verifies are too few to see proportionality`,
      );
      const settled = new Set(rebinds.slice(1));
      assertEquals(
        settled.size,
        1,
        `rebindCount moved in the verifies from the second on (rebuilding in proportion to the cycle count): [${rebinds.join(
          ",",
        )}]`,
      );
      assert(
        rebinds[1] - rebinds[0] <= 1,
        `${rebinds[1] - rebinds[0]} rebuilds before settling on the verify form: [${rebinds.join(
          ",",
        )}]`,
      );
      console.log(
        `[e2e] gemma4 speculation ③: prefill ${prefills.length} + draft ${drafts.length} + verify ` +
          `${verifies.length} notifications (${bucketed.length} of them M=k+1) / delivered ` +
          `${turn.stop.tokens} token / verify rows [${rowsSeries.join(",")}] / rebind [${rebinds.join(
            ",",
          )}]`,
      );
    });

    it("④ the estimate sums the drafter's resident weights, the borrower state and the verify form", async () => {
      const { observed } = await observeTurn();
      // The diagnostics come from a different Session per phase (draft = the borrower, everything
      // else = the lender).
      const lender = ofKind(observed, "verify")[0];
      const borrower = ofKind(observed, "draft")[0];
      assert(
        lender !== undefined && borrower !== undefined,
        "the lender / borrower diagnostics were not taken",
      );
      // MUST: emit the observed values **before** taking the estimate (so the material for the
      // comparison remains even if the estimate fails).
      console.log(
        `[e2e] gemma4 speculation ④ observed: resident weights target ${lender.residentCompressedBytes} B + ` +
          `drafter ${borrower.residentCompressedBytes} B / state lender ` +
          `${lender.stateResidentBytes} B + borrower ${borrower.stateResidentBytes} B`,
      );
      const report = pipeline.estimateSessionMemory();

      // 1. Resident weights: exactly the sum of the bytes the 2 Sessions actually put on the GPU.
      assertEquals(
        report.resident.weights.compressedBytes,
        lender.residentCompressedBytes + borrower.residentCompressedBytes,
        `estimate of the compressed resident bytes (target ${lender.residentCompressedBytes} + drafter ` +
          `${borrower.residentCompressedBytes}）`,
      );
      // 2. The borrower context holds only the logical-length uniform (borrowed slots allocate no
      // byte at all).
      assertEquals(
        borrower.stateResidentBytes,
        LENGTHS_BYTES,
        "resident bytes of the borrower context",
      );
      // 3. The state is the lender's share + the borrower's 8 bytes.
      assertEquals(
        report.resident.stateBytes,
        lender.stateResidentBytes + LENGTHS_BYTES,
        `estimate of the state (lender ${lender.stateResidentBytes} + borrower ${LENGTHS_BYTES})`,
      );

      // 4. The verify scenario's inputs and outputs are larger than decode's by exactly "k more
      //    rows". The **input** bytes of one row are derived from the same report (prefill and
      //    decode have the same shape at R=1 and differ only in the chunk row count, so dividing
      //    their difference by the row count gives one row). It closes inside the report without
      //    copying the formula, so a change of `chunkLength` or the PLE width needs no follow-up.
      const named = (name: string) => {
        const scenario = report.scenarios.find((entry) => entry.name === name);
        assert(scenario !== undefined, `the report has no scenario '${name}'`);
        return scenario;
      };
      const prefill = named("prefill");
      const decode = named("decode");
      const verify = named("verify");
      const chunkLength = pipeline.program.chunkLength;
      const perRowInputBytes = (prefill.ioBytes - decode.ioBytes) / (chunkLength - 1);
      assert(
        Number.isSafeInteger(perRowInputBytes),
        `the input bytes of one row are not an integer (prefill ${prefill.ioBytes} − decode ` +
          `${decode.ioBytes} divided by ${chunkLength - 1} rows = ${perRowInputBytes})`,
      );
      const rows = k + 1;
      assertEquals(
        verify.ioBytes - decode.ioBytes,
        // Inputs: k more rows (token / PLE / RoPE) + k more elements of the row selection
        // `last_row`.
        k * perRowInputBytes +
          k * 4 +
          // Outputs: the selected rows' logits and hidden are read back for k extra rows.
          k * (VOCAB + HIDDEN) * 4,
        `ioBytes difference between verify (${rows} rows) and decode (1 row)`,
      );
      console.log(
        `[e2e] gemma4 speculation ④: resident weights ${report.resident.weights.compressedBytes} B (target ` +
          `${lender.residentCompressedBytes} + drafter ${borrower.residentCompressedBytes}）/ ` +
          `state ${report.resident.stateBytes} B / io prefill ${prefill.ioBytes}, decode ` +
          `${decode.ioBytes}, verify ${verify.ioBytes} B / 1 row ${perRowInputBytes} B`,
      );
    });
  },
);

// ---------------------------------------------------------------------------
// ⑤ Kernel equivalence (u32) — verify's row 0 and decode's single row
// ---------------------------------------------------------------------------

/** Takes an f32 output tensor (a dtype mix-up silently becomes another list). */
const f32Of = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  assert(tensor.data instanceof Float32Array, `${where}: reading a non-f32 output as f32`);
  return tensor.data;
};

/**
 * The pair of logits of the same row run in verify form (row 0 of an M=k+1 deferred run) and in
 * decode form (M=1).
 */
type RowPair = {
  readonly deferred: Float32Array<ArrayBuffer>;
  readonly immediate: Float32Array<ArrayBuffer>;
};

describe.skipIf(!AVAILABLE || !GPU_AVAILABLE)(
  `gemma4 speculation ⑤: the logits of verify's row 0 and of a single decode row match as u32 (real GPU)${GOLDENS_SKIP_REASON}`,
  () => {
    let config: PipelineConfig;
    let rope: RopeSpec;
    let golden: GoldenCase;
    let prompt: number[];
    let ple: Gemma4Ple;
    let component: Awaited<ReturnType<typeof openMirrorComponent>>;
    let gpu: GpuContext;

    beforeAll(async () => {
      config = pipelineConfig();
      rope = config.rope;
      golden = await readGoldenCase("short-en");
      prompt = golden.prompt;
      const { index, openBlock } = await openGemma4Ple(MIRROR_DIR, INT4_SEAT);
      ple = createGemma4Ple({
        index,
        openBlock,
        vocabSize: VOCAB,
        maxResidentBytes: allResidentBytes(index),
      });
      component = await openMirrorComponent(MIRROR_DIR, "model", INT4_SEAT);
      gpu = await acquireGpu();
    });

    afterAll(() => {
      // Each is undefined when beforeAll threw before creating it.
      gpu?.destroy();
      ple?.dispose();
    });

    /**
     * One run (the element count of `last_row` binds R). It returns the logits of the selected
     * rows.
     *
     * The physical row count `rows` is one of the 3 kinds the fixed-chunk contract (ADR 0066
     * decision 4) allows — `chunkLength` / a declared bucket / the decode form's 1 — and the valid
     * rows are its first `ids.length` (the rest are pad = token 0, position 0, outside
     * `queryLength`, so they do not enter the KV).
     */
    const run = async (
      session: Session,
      logitsName: string,
      context: GenerationContext,
      spec: {
        readonly rows: number;
        readonly ids: readonly number[];
        readonly position: number;
        readonly lastRow: readonly number[];
        readonly commit: "immediate" | "deferred";
      },
      where: string,
    ): Promise<Float32Array<ArrayBuffer>> => {
      const tokens = new Int32Array(new ArrayBuffer(spec.rows * 4));
      const positions = new Int32Array(new ArrayBuffer(spec.rows * 4));
      for (let row = 0; row < spec.ids.length; row += 1) {
        tokens[row] = spec.ids[row];
        positions[row] = spec.position + row;
      }
      const outputs = await session.run(
        {
          [INPUT_IDS]: { dtype: "i32", shape: [1, spec.rows], data: tokens },
          [PER_LAYER_INPUTS]: await ple.gather([...tokens]),
          ...ropeInputs(rope, positions),
          [LAST_ROW]: {
            dtype: "i32",
            shape: [spec.lastRow.length],
            data: Int32Array.from(spec.lastRow),
          },
        },
        undefined,
        { context, queryLength: spec.ids.length, commit: spec.commit },
      );
      assertEquals(
        outputs[logitsName].shape,
        [1, spec.lastRow.length, VOCAB],
        `${where}: logits shape`,
      );
      return f32Of(outputs[logitsName], where);
    };

    /**
     * The measurement of one seat (prefill → a 4-row deferred run → `commit(0)` → a 1-row run of
     * the same row).
     *
     * The rejection (`commit(0)`) sits in between because that is the order the speculative loop
     * actually goes through (the point of this check is that after a cycle that accepted 0, the
     * next decode emits the same values).
     */
    const measure = async (
      reduce: StateAttentionReduce,
      linearGemvReduce: "sequential" | "parallel" = "sequential",
      fuseRmsNormAdd = false,
      fuseLinearStaticQuantize = false,
    ): Promise<RowPair> => {
      // The order of the two outputs is the contract (output 0 = logits, output 1 = the hidden
      // state after the final norm).
      const logitsName = component.graph.outputs[0];
      const session = await component.createSession(gpu, {
        stateAttentionReduce: reduce,
        linearGemvReduce,
        fuseRmsNormAdd,
        fuseLinearStaticQuantize,
      });
      try {
        const context = await session.createGenerationContext({
          bindings: { [CAPACITY_SYMBOL]: config.capacity },
          chunkLength: CHUNK_LENGTH,
          // The run shape that streams the verify form (k+1 rows) without pad. It is the same
          // physical row count the generation surface (`sequence.ts`) picks when speculating.
          chunkBuckets: [VERIFY_ROWS],
        });
        try {
          await run(
            session,
            logitsName,
            context,
            {
              rows: CHUNK_LENGTH,
              ids: prompt,
              position: 0,
              lastRow: [prompt.length - 1],
              commit: "immediate",
            },
            "prefill",
          );
          assertEquals(context.pastLength, prompt.length, "logical length after prefill");
          // The frontier `b` and the 3 drafts placed after it (any values — only row 0 is looked
          // at; the following rows matter only by "being in the same run"). Taken from the
          // golden's continuation.
          const frontier = golden.continuation[0];
          const drafts = golden.continuation.slice(1, 1 + GEMMA4_DRAFT_STEPS);
          const deferred = await run(
            session,
            logitsName,
            context,
            {
              rows: VERIFY_ROWS,
              ids: [frontier, ...drafts],
              position: prompt.length,
              lastRow: Array.from({ length: VERIFY_ROWS }, (_unused, row) => row),
              commit: "deferred",
            },
            "verify",
          );
          // Accept no row (= a cycle that rejects everything). The logical length stays as right
          // after prefill.
          context.commit(0);
          assertEquals(context.pastLength, prompt.length, "logical length after commit(0)");
          const immediate = await run(
            session,
            logitsName,
            context,
            {
              rows: 1,
              ids: [frontier],
              position: prompt.length,
              lastRow: [0],
              commit: "immediate",
            },
            "decode",
          );
          assertEquals(session.diagnostics().lastRunFusions?.rmsNormAdd, fuseRmsNormAdd ? 106 : 0);
          return {
            // Copy row 0 only (`deferred` actually holds 4 rows).
            deferred: deferred.slice(0, VOCAB),
            immediate: immediate.slice(0, VOCAB),
          };
        } finally {
          await context.dispose();
        }
      } finally {
        await session.dispose();
      }
    };

    /**
     * Compares two lists as u32 (a summary of the mismatches — so a failure does not print a diff
     * of 260,000 elements).
     */
    const compare = (pair: RowPair) => {
      const left = new Uint32Array(pair.deferred.buffer);
      const right = new Uint32Array(pair.immediate.buffer);
      let mismatches = 0;
      let firstIndex = -1;
      let maxAbsDiff = 0;
      for (let index = 0; index < VOCAB; index += 1) {
        if (left[index] !== right[index]) {
          if (mismatches === 0) firstIndex = index;
          mismatches += 1;
        }
        const diff = Math.abs(pair.deferred[index] - pair.immediate[index]);
        if (diff > maxAbsDiff) maxAbsDiff = diff;
      }
      return { mismatches, firstIndex, maxAbsDiff, left, right };
    };

    it("① matches exactly as u32 on the sequential seat", async () => {
      const started = performance.now();
      const pair = await measure("sequential");
      const { mismatches, firstIndex, maxAbsDiff, left, right } = compare(pair);
      assertEquals(
        mismatches,
        0,
        `verify's row 0 and the single decode row differ in ${mismatches}/${VOCAB} entries (first @${firstIndex}: ` +
          `${pair.deferred[firstIndex]} vs ${pair.immediate[firstIndex]}, max absolute difference ` +
          `${maxAbsDiff}）`,
      );
      // The summary has not become a tautology (the same comparison runs once more as a strict
      // typed comparison).
      assertEquals(left, right, "strict u32 comparison");
      console.log(
        `[e2e] gemma4 speculation ⑤ sequential: u32 match (${VOCAB} entries) / ` +
          `${(performance.now() - started).toFixed(0)}ms`,
      );
    });

    // Every combination of 3 attention variants × 2 GEMV kinds × RMS→add fusion (the 11 besides
    // ①'s sequential × sequential × no fusion) is pinned by the same u32 check — the self-paying
    // gate picks M=1 / M=4 by wall-clock time, so unless the 2 paths are bit-identical on every
    // selectable seat, the practical tier's in-device determinism (ADR 0110 decision 6) does not
    // hold. Thinning the combinations leaves the hole "a combination outside the check claims no
    // identity" (extended to every combination on 2026-09-26).
    for (const attention of ["sequential", "parallel", "parallel-fused"] as const) {
      for (const gemv of ["sequential", "parallel"] as const) {
        for (const fuse of [false, true] as const) {
          if (attention === "sequential" && gemv === "sequential" && !fuse) continue;
          it(`attention=${attention} / GEMV=${gemv} / RMS fusion=${fuse}: verify row 0 and decode match as u32`, async () => {
            const started = performance.now();
            const { mismatches, firstIndex, maxAbsDiff, left, right } = compare(
              await measure(attention, gemv, fuse),
            );
            // MUST: log before the check (so the size of the difference is on record even on a
            // machine where it breaks).
            console.log(
              `[e2e] gemma4 speculation ⑤ ${attention}/${gemv}/fuse=${fuse}: u32 differences ${mismatches}/${VOCAB} entries` +
                ` (first @${firstIndex}) / max absolute difference ${maxAbsDiff} / ${(
                  performance.now() - started
                ).toFixed(0)}ms`,
            );
            assertEquals(mismatches, 0);
            assertEquals(left, right);
          });
        }
      }
    }

    it("linear→SRQ requested keeps the MTP verify row 0 / decode u32 identity", async () => {
      const { mismatches, left, right } = compare(
        await measure("parallel", "parallel", true, true),
      );
      assertEquals(mismatches, 0);
      assertEquals(left, right);
    });
  },
);
