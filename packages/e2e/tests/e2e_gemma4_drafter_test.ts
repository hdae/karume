// Acceptance check of the real-weight Gemma 4 E2B **MTP drafter** (the draft side of speculative
// decoding) — ADR [0096](../../../docs/decisions/0096-speculative-decoding.md) step 2.
//
// What it accepts is "the drafter runs by **borrowing** the target's KV and embedding table, and
// produces the same drafts as the torch drafter". The check of the speculative loop itself
// (verify, accept, reject, commit) belongs to step 3's `e2e_gemma4_speculative_test.ts`, so this
// file looks at one cycle's drafts and at the distribution's pipeline using that drafter **by
// default**:
//
// ① Distribution (no GPU) … the manifest's weights hold `drafter`, and the default quant points at
//    both roles. The `ResolveOptions.weights` filter works, and **a load without speculation does
//    not keep the drafter's container in the selection** (if it stayed, its descriptor and weight
//    parts would be downloaded too). The PLE index is also consistent, entry for entry, with the
//    assets of the `model` container (ADR 0109 decision 4 — the successor of the old "deferred
//    assets = exactly the PLE shards" cross-check; only these two run without a GPU)
// ② Container shape (no GPU) … 6 inputs (order included) / 3 outputs / 4 external slots with the
//    same names and shapes as the target / 1 shared initializer that resolves to the target's
//    initializer
// ③ **Draft match** (real GPU) … 3 cases × 200 cycles × 3 steps = 1,800 draft tokens match the
//    torch drafter's golden **at 99% or more**. The hidden on the karume side comes from the bf16
//    target (the golden's hidden comes from an f32 proxy target), so bit identity cannot be asked
//    for — it is a match-rate check. The target's token sequence is teacher-forced with the
//    golden's continuation.
//    The golden's proxy target is the product series as it stores the weights — the checkpoint's
//    bf16 values, nothing rounded (`export_drafter.write_goldens`) — so it is the `bf16` seat this
//    file opens, the drafter's one target.
// ④ Lifetime (real GPU) … while the borrower is alive, neither the lender context nor the lender
//    Session can be disposed, and in the right order (borrower → lender) it goes through
//
// MUST: the match-rate check is **not fitted to the measurement**. If it fails, report the
// distribution per step and per position and leave it red (the drafter's quality is the speed of
// speculation itself; loosening the check would turn "works but is not faster" green).
//
// ## Assets
//
// The distribution mirror `models/karume-gemma4/` (assembled by `dist.py --pipeline gemma4` —
// drafter included, its `bf16` preset — the int8 drafter borrows the bf16 target's table) and
// the drafter goldens in the test-asset root, `goldens/gemma4-e2b-drafter/` (ADR 0139 decision 2).
// Both are outside the repository; without the mirror the cases **SKIP explicitly**. Their
// completeness case below runs in the default lane under ADR 0139's fetched rule: a golden absent
// from a complete fetch of the pinned test-asset dataset FAILs (the dataset at the pin lacks it);
// without a complete fetch the case SKIPs explicitly (the assets gate is red then).
// (`e2e_gemma4_speculative_test.ts` reads the q4_0 QAT drafter's goldens since 2026-11-04, so ③ is
// the only reader of these.)

import {
  isNotFound,
  readFile,
  readTextFile,
  readTextFileSync,
  statSync,
} from "@karume/test-support/fs";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import {
  type Manifest,
  MANIFEST_FILENAME,
  ManifestReferenceError,
  parseManifest,
  type ResolvedSelection,
  resolveSelection,
} from "@karume/hub";
import {
  acquireGpu,
  ExecutionError,
  type GenerationContext,
  parseSafetensors,
  type SafetensorsFile,
  type Session,
  type Tensor,
} from "@karume/runtime";
import { nodeDirectory } from "@karume/hub/node";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
// MUST: the entry point is taken from the **public surface** (grabbing `src/...` directly would
// keep the check green even if the surface shrank).
import { type ChatMessage, Pipeline } from "../../models/gemma.ts";
import { type PipelineConfig, parsePipelineConfig } from "../../models/src/gemma/config.ts";
import { createGemma4Ple, type Gemma4Ple } from "../../models/src/gemma/ple.ts";
import { gemma4RopeInputNames, ropeInputs, type RopeSpec } from "../../models/src/gemma/rope.ts";
import {
  admitGemma4Drafter,
  draftOnce,
  GEMMA4_DRAFT_STEPS,
  type Gemma4Drafter,
  openDrafterContext,
} from "../../models/src/gemma/speculative.ts";
import { planPrefillChunks } from "../../models/src/generation/greedy.ts";
import {
  mirrorAvailable,
  openGemma4Ple,
  openMirrorComponent,
} from "../../models/tests/helpers/gemma-mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  allResidentBytes,
  allResidentPleBytesOfMirror,
} from "../../models/tests/helpers/ple-budget.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";
import {
  goldenDir,
  TEST_ASSETS,
  TEST_ASSETS_ROOT,
  testAssetsFetched,
} from "@karume/test-support/test-assets";

const MIRROR_DIR = new URL("../../../models/karume-gemma4/", import.meta.url);
/**
 * The seat these cases read: the regular line's `bf16` preset (its default and, since 2026-11-04,
 * its one preset — the int8 drafter's only target). Named so a later default does not move them.
 */
const BF16_SEAT = { quant: "bf16" } as const;
const GOLDEN_ROOT = goldenDir("gemma4-e2b-drafter");

/** The assembly command, ready to paste when the cases SKIP. */
const ASSEMBLE_COMMAND =
  "cd tools/export-recipes && uv run python dist.py --pipeline gemma4" +
  " (drafter series and goldens: … python -m gemma4.export_drafter)";

/**
 * The distribution's role names (the same spelling as `Pipeline`'s constants of the same name — the
 * writing side is `distribution.py`).
 */
const MODEL = "model";
const DRAFTER = "drafter";
/** The golden cases (source of truth: GOLDEN_CASES in `export_drafter.py`). */
const CASES = ["short-en", "readme-recipes", "readme-exporter"] as const;
const GOLDEN_PREFIX = "drafter-golden.";
const SUFFIX = ".safetensors";

/** The match-rate floor (denominator = 3 cases × 200 cycles × 3 steps). */
const MATCH_FLOOR = 0.99;

/**
 * The real asset's shape (`hidden_size` / `vocab_size` from the config, and the shared
 * initializer's name).
 *
 * The shared initializer has the same name in the borrower and the lender (the name is the key of
 * the tensor — docs/ir-v2.md), so one constant is enough.
 */
const HIDDEN = 1536;
const VOCAB = 262144;
const SHARED_WEIGHT = "model.lm_head.weight";

/** Graph input names (source of truth: the constants in `export_product.py`). */
const INPUT_IDS = "input_ids";
const PER_LAYER_INPUTS = "per_layer_inputs";
const LAST_ROW = "last_row";
const CAPACITY_SYMBOL = "C";
/**
 * The prefill chunk length (equal to the distribution's declared `chunkLength` — stated explicitly
 * because it is a run-time knob).
 */
const CHUNK_LENGTH = 768;

const exists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const goldenPath = (name: string): URL => new URL(`${GOLDEN_PREFIX}${name}${SUFFIX}`, GOLDEN_ROOT);

/** The goldens of `files` that are absent, as paths under the test-asset root. */
const missingGoldens = (files: readonly URL[]): string[] =>
  files
    .filter((file) => !exists(file))
    .map((file) => file.href.slice(TEST_ASSETS_ROOT.href.length));

/**
 * Whether this is a `karume/5` mirror with the drafter.
 *
 * MUST: decided by **the manifest's declaration**, not by file names — a container part's path is
 * the writer's convention (`<stem>-NNNNN-of-NNNNN.krm`), and the names move when the part count
 * changes (ADR 0109 decision 3).
 */
const MIRROR_PRESENT = ((): boolean => {
  if (!mirrorAvailable(MIRROR_DIR)) return false;
  const manifest = parseManifest(readTextFileSync(new URL(MANIFEST_FILENAME, MIRROR_DIR)));
  return Object.hasOwn(manifest.models[manifest.defaultModel].weights, "drafter");
})();
const GOLDENS_PRESENT = CASES.every((name) => exists(goldenPath(name)));
const AVAILABLE = MIRROR_PRESENT && GOLDENS_PRESENT;

if (!MIRROR_PRESENT) {
  console.warn(
    `[karume] the distribution mirror with the drafter (${MIRROR_DIR.pathname}) is missing, so the MTP drafter acceptance is` +
      ` SKIPPED. Assemble it: ${ASSEMBLE_COMMAND}`,
  );
}

const manifest: Manifest | undefined = MIRROR_PRESENT
  ? parseManifest(readTextFileSync(new URL(MANIFEST_FILENAME, MIRROR_DIR)))
  : undefined;

/** The mirror's manifest (called only where the cases do not SKIP). */
const mirrorManifest = (): Manifest => {
  assert(manifest !== undefined, "the distribution mirror's manifest was not read");
  return manifest;
};

/**
 * Completeness of the dependent assets (a gap is not folded into a SKIP). The goldens are read by ③
 * here; this case follows ADR 0139's fetched rule: a golden absent from a complete fetch of the pin
 * is a defect of the dataset at the pin and FAILs; without a complete fetch the case SKIPs with the
 * reason in its title (the assets gate is red then).
 */
const MISSING_AT_LOAD = missingGoldens(CASES.map(goldenPath));
const GOLDENS_SKIP = MISSING_AT_LOAD.length > 0 && !testAssetsFetched();
test.skipIf(GOLDENS_SKIP)(
  `gemma4 drafter assets: the 3 golden cases are all present${
    GOLDENS_SKIP
      ? ` (SKIP: ${MISSING_AT_LOAD.join(" / ")} absent and outputs/test-assets/ is not a complete ` +
        "fetch of the pin — node tools/test-assets/fetch.ts)"
      : ""
  }`,
  () => {
    const missing = missingGoldens(CASES.map(goldenPath));
    assert(
      missing.length === 0,
      `${missing.join(" / ")} absent although outputs/test-assets/ holds a complete fetch of ` +
        `${TEST_ASSETS.repo}@${TEST_ASSETS.revision} — the test-asset dataset at the pin lacks it`,
    );
  },
);

// ---------------------------------------------------------------------------
// ① Distribution (no GPU) — the manifest's roles and the `ResolveOptions.weights` filter
// ---------------------------------------------------------------------------

test.skipIf(!MIRROR_PRESENT)(
  "gemma4 drafter distribution: the manifest's weights hold the drafter and the quant points at both",
  () => {
    const entry = mirrorManifest().models[mirrorManifest().defaultModel];
    assertEquals(Object.keys(entry.weights).sort(), [DRAFTER, MODEL], "weights roles");
    const quant = entry.quants[BF16_SEAT.quant];
    // A complete mapping (the parse guarantees it): the target is the checkpoint's own "bf16", the
    // drafter **int8 throughout** (its linears too — int4 g32 drops the acceptance by 10–30 %, ADR
    // 0096 addendum).
    assertEquals(
      Object.keys(quant.weights).sort(),
      [DRAFTER, MODEL],
      "the quant's weights mapping",
    );
    assertEquals(quant.weights[MODEL], "bf16", "the target's dtype label");
    assertEquals(quant.weights[DRAFTER], "i8", "the drafter's dtype label");
    // One requiredLimits per seat (the bf16 target's buffers bound it; the drafter adds none).
    assert(quant.requiredLimits !== undefined, "the default quant declares no requiredLimits");
  },
);

test.skipIf(!MIRROR_PRESENT)(
  "gemma4 drafter distribution: the weights filter drops the drafter's container from the selection",
  () => {
    const target = resolveSelection(mirrorManifest(), { weights: [MODEL] });
    const both = resolveSelection(mirrorManifest(), { weights: [MODEL, DRAFTER] });

    const components = (selection: ResolvedSelection) => Object.keys(selection.containers).sort();
    assertEquals(components(target), [MODEL], "the drafter's container remains despite the filter");
    assertEquals(components(both), [DRAFTER, MODEL], "the selection with the drafter added");
    // Filtering does not move a single part or asset on the target side.
    assertEquals(both.containers[MODEL], target.containers[MODEL], "the model container reference");
    assertEquals(Object.keys(both.assets).sort(), Object.keys(target.assets).sort(), "assets");
    // The PLE is a container asset (ADR 0109 decision 4), so only the tokenizer is left in the
    // manifest's assets.
    assertEquals(Object.keys(target.assets), ["tokenizer"], "the manifest's assets");

    assertThrows(
      () => resolveSelection(mirrorManifest(), { weights: [MODEL, "mtp"] }),
      ManifestReferenceError,
      "weights 'mtp' is not in the manifest",
    );
  },
);

test.skipIf(!MIRROR_PRESENT)(
  "gemma4 drafter distribution: the PLE index is consistent, entry for entry, with the model container's assets",
  async () => {
    // `openGemma4Ple` goes through `readGemma4PleIndex` (every entry of index × container assets)
    // — with or without the drafter, the PLE stays inside the `model` container (the successor of
    // the old `deferred` cross-check).
    const { index } = await openGemma4Ple(MIRROR_DIR, BF16_SEAT);
    assertEquals(index.tokens, VOCAB, "index row count (= the main embedding's vocab)");
    assert(index.values.blocks.length > 1, "values has one block (no split by vocab range)");
  },
);

// ---------------------------------------------------------------------------
// ② Container shape (no GPU)
// ---------------------------------------------------------------------------

const targetComponent = () => openMirrorComponent(MIRROR_DIR, MODEL, BF16_SEAT);
const drafterComponent = () => openMirrorComponent(MIRROR_DIR, DRAFTER, BF16_SEAT);

/** The mirror's `pipelineConfig` (where the RoPE declaration and the capacity come from). */
const pipelineConfig = () => {
  const entry = mirrorManifest().models[mirrorManifest().defaultModel];
  return parsePipelineConfig(entry.pipelineConfig);
};

/**
 * Takes the declarations from each component's descriptor (not a single byte of the data section is
 * needed).
 */
const readGraphs = async () => ({
  target: (await targetComponent()).graph,
  drafter: (await drafterComponent()).graph,
});

test.skipIf(!MIRROR_PRESENT)(
  "gemma4 drafter container: the inputs/outputs, external slots and shared initializer fit the target",
  async () => {
    const { target, drafter } = await readGraphs();
    const config = pipelineConfig();

    // 6 inputs (the order is the contract), 3 outputs (in step order).
    assertEquals(
      drafter.inputs.map((input) => input.name),
      ["token", "hidden", ...gemma4RopeInputNames()],
      "drafter graph inputs (order included)",
    );
    assertEquals(drafter.outputs.length, GEMMA4_DRAFT_STEPS, "number of draft outputs (= k)");
    assertEquals(
      drafter.symbols,
      [CAPACITY_SYMBOL],
      "symbols (one capacity symbol inherited from the lender)",
    );

    // 4 external slots (k/v of l13 sliding / l14 full) with the same names and shapes as the
    // target.
    const external = Object.keys(drafter.states).filter((name) => drafter.states[name].external);
    assertEquals(external.length, 4, "number of external slots");
    assertEquals(external.length, Object.keys(drafter.states).length, "own slots are mixed in");
    for (const name of external) {
      assertEquals(drafter.states[name].shape, target.states[name].shape, `shape of ${name}`);
    }

    // There is one shared initializer, and it holds no bytes (no storage). Its name is the
    // lender's initializer name as is (= the key of the tensor).
    const shared = Object.keys(drafter.initializers).filter(
      (name) => drafter.initializers[name].shared !== undefined,
    );
    assertEquals(shared.length, 1, "number of shared initializers");
    assertEquals(
      drafter.initializers[shared[0]].storage,
      undefined,
      "the shared declaration holds bytes",
    );
    assertEquals(
      shared[0],
      SHARED_WEIGHT,
      "the lender initializer name the shared initializer gives",
    );

    // The check itself (the one `Pipeline` runs at admission). It resolves as far as the lender's
    // initializer name.
    const admitted = admitGemma4Drafter("test", drafter, {
      graph: target,
      rope: config.rope,
      hiddenSize: HIDDEN,
      capacitySymbol: CAPACITY_SYMBOL,
    });
    assertEquals(
      admitted.outputs,
      [...drafter.outputs],
      "the outputs admission returns (step order)",
    );
    assertEquals(admitted.hiddenSize, HIDDEN, "hidden width");
    assertEquals(admitted.sharedWeights, { [shared[0]]: SHARED_WEIGHT }, "the shared weight map");
  },
);

test.skipIf(!MIRROR_PRESENT)(
  "gemma4 drafter container: a declaration paired with a target of another generation fails at admission",
  async () => {
    const { target, drafter } = await readGraphs();
    const config = pipelineConfig();
    const admit = (hiddenSize: number, capacitySymbol: string) =>
      admitGemma4Drafter("test", drafter, {
        graph: target,
        rope: config.rope,
        hiddenSize,
        capacitySymbol,
      });

    // It cannot pair with a target of a different hidden width (= another generation).
    assertThrows(() => admit(HIDDEN + 1, CAPACITY_SYMBOL), Error, "drafter input 'hidden'");
    // With a differently spelled capacity symbol the borrower context has nothing to bind.
    assertThrows(() => admit(HIDDEN, "K"), Error, "the drafter graph's symbols are");
    // A lender without the shared tensor (a target with the initializer dropped).
    const stripped = {
      ...target,
      initializers: Object.fromEntries(
        Object.entries(target.initializers).filter(([name]) => name !== SHARED_WEIGHT),
      ),
    };
    assertThrows(
      () =>
        admitGemma4Drafter("test", drafter, {
          graph: stripped,
          rope: config.rope,
          hiddenSize: HIDDEN,
          capacitySymbol: CAPACITY_SYMBOL,
        }),
      Error,
      "shared initializer",
    );
  },
);

// NOTE: the asset-free check of the `speculative` option's accepted set lives in
// `packages/models/tests/speculative_options_test.ts`.

// ---------------------------------------------------------------------------
// ③④ Real GPU — comparison with the golden, and lifetime
// ---------------------------------------------------------------------------

/** One golden case (`prompt [T]` / `tokens [N + k]` / `draft [N, k]`). */
type Golden = {
  readonly prompt: Int32Array<ArrayBuffer>;
  readonly tokens: Int32Array<ArrayBuffer>;
  readonly draft: Int32Array<ArrayBuffer>;
  readonly cycles: number;
};

const goldenI32 = (file: SafetensorsFile, name: string): Int32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `the golden has no '${name}'`);
  assertEquals(view.dtype, "I32", `storage dtype of golden '${name}'`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

const readGolden = async (name: string): Promise<Golden> => {
  const bytes = await readFile(goldenPath(name));
  const file = parseSafetensors(
    bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
      ? bytes.buffer
      : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const draftView = file.tensors.get("draft");
  assert(draftView !== undefined, `golden ${name} has no 'draft'`);
  assertEquals(draftView.shape[1], GEMMA4_DRAFT_STEPS, `${name}: draft step count`);
  const tokens = goldenI32(file, "tokens");
  // The continuation is longer by **what the drafts are compared against** (+ k) — up to the
  // position the last cycle's d_k predicts.
  assertEquals(
    draftView.shape[0] + GEMMA4_DRAFT_STEPS,
    tokens.length,
    `${name}: draft cycle count`,
  );
  return {
    prompt: goldenI32(file, "prompt"),
    tokens,
    draft: goldenI32(file, "draft"),
    cycles: draftView.shape[0],
  };
};

/** Takes out an f32 output tensor (a dtype mix-up would silently give a different sequence). */
const f32Of = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  assert(tensor.data instanceof Float32Array, `${where}: reading a non-f32 output as f32`);
  return tensor.data;
};

/** The target and drafter Sessions of the mirror on one GPU. */
type DrafterPair = {
  readonly target: Session;
  readonly drafterSession: Session;
  readonly drafter: Gemma4Drafter;
  readonly targetGraph: Awaited<ReturnType<typeof readGraphs>>["target"];
  /**
   * Disposes in the lifetime order (borrower -> lender) and checks that the lender refuses to go
   * first.
   */
  readonly close: () => Promise<void>;
};

/**
 * Opens the target Session and a drafter Session that borrows the target's embedding table; the
 * caller's teardown calls `close`. Shared by the default lifetime case and the opt-in draft-match
 * case.
 */
const openDrafterPair = async (): Promise<DrafterPair> => {
  const rope: RopeSpec = pipelineConfig().rope;
  const { target: targetGraph, drafter: drafterGraph } = await readGraphs();
  const admitted = admitGemma4Drafter("test", drafterGraph, {
    graph: targetGraph,
    rope,
    hiddenSize: HIDDEN,
    capacitySymbol: CAPACITY_SYMBOL,
  });

  const targetOpened = await targetComponent();
  const drafterOpened = await drafterComponent();
  const gpu = await acquireGpu();
  // ③PV runs with the parallel reduction — the order the parallel presets decode with
  // (`i4-gemvpar`'s `"parallel"`; `i4-fast`'s `"parallel-fused"` adds in the same order — ADR
  // 0102).
  const sessionOptions = { stateAttentionReduce: "parallel" } as const;
  const target: Session = await targetOpened.createSession(gpu, sessionOptions);
  let drafterSession: Session;
  try {
    // What is borrowed is the one embedding table (not a single byte is copied).
    const sharedWeights = Object.fromEntries(
      Object.keys(admitted.sharedWeights).map((borrower) => [
        borrower,
        target.exportWeight(admitted.sharedWeights[borrower]),
      ]),
    );
    drafterSession = await drafterOpened.createSession(gpu, {
      ...sessionOptions,
      sharedWeights,
    });
  } catch (error) {
    // No borrower was created, so the lender goes alone.
    await target.dispose();
    gpu.destroy();
    throw error;
  }
  const drafter: Gemma4Drafter = {
    session: drafterSession,
    graph: drafterGraph,
    outputs: admitted.outputs,
    rope,
    hiddenSize: admitted.hiddenSize,
  };
  const close = async (): Promise<void> => {
    // MUST: the borrower Session is torn down first (disposing the lender while its weights are
    // borrowed is refused).
    const refused = await assertRejects(() => target.dispose(), ExecutionError);
    assert(refused.message.includes("lent out"), refused.message);
    await drafterSession.dispose();
    await target.dispose();
    gpu.destroy();
  };
  return { target, drafterSession, drafter, targetGraph, close };
};

describe.skipIf(!MIRROR_PRESENT || !GPU_AVAILABLE)(
  "gemma4 drafter lifetime: the drafter stays resident on the target's borrowed table, and the lender cannot be torn down while the borrower lives (real GPU)",
  () => {
    let pair: DrafterPair | undefined;

    beforeAll(async () => {
      pair = await openDrafterPair();
      const { target, drafterSession } = pair;
      // The residency check was the head of the former test body, before its step: it stays a
      // precondition here (a failure fails this describe and skips the step's it).
      // It is not that the drafter keeps **no** bytes resident (it holds its own 4 layers), but it
      // does not hold the borrowed table = it stays under 1/20 of the target.
      const drafterStorage = drafterSession.diagnostics().storage;
      const targetStorage = target.diagnostics().storage;
      assert(
        drafterStorage.residentCompressedBytes * 10 < targetStorage.residentCompressedBytes,
        `the drafter's resident ${drafterStorage.residentCompressedBytes} bytes are too many against the target's ` +
          `${targetStorage.residentCompressedBytes} bytes (it is not borrowing the table)`,
      );
    });

    afterAll(async () => {
      await pair?.close();
    });

    it("④ lifetime: the lender cannot be torn down while the borrower lives", async () => {
      assert(pair !== undefined, "the target / drafter pair is not open");
      const { target, drafter } = pair;
      const context = await target.createGenerationContext({
        bindings: { [CAPACITY_SYMBOL]: CHUNK_LENGTH },
        chunkLength: CHUNK_LENGTH,
      });
      const borrowed = await openDrafterContext(drafter, context);
      // The lender context cannot be disposed while a borrower exists (the borrower's bind group
      // holds it).
      const refusedContext = await assertRejects(() => context.dispose(), ExecutionError);
      assert(refusedContext.message.includes("are borrowing"), refusedContext.message);
      // In the right order it goes through.
      await borrowed.dispose();
      await context.dispose();
    });
  },
);

/** The match counts of one case (per step). */
type MatchCounts = { readonly perStep: number[]; readonly total: number; readonly cycles: number };

// The draft-match gate compares Karume's drafts with the torch drafter's goldens under a match-rate
// floor (upstream parity), so it is opt-in.
describe.skipIf(!UPSTREAM_PARITY || !AVAILABLE || !GPU_AVAILABLE)(
  `gemma4 drafter acceptance: the drafts match the torch golden at 99% or more (real GPU)${UPSTREAM_PARITY_OPT_IN}`,
  () => {
    let config: PipelineConfig;
    let ple: Gemma4Ple | undefined;
    let pair: DrafterPair | undefined;
    const counts: MatchCounts[] = [];

    beforeAll(async () => {
      config = pipelineConfig();
      const { index, openBlock } = await openGemma4Ple(MIRROR_DIR, BF16_SEAT);
      ple = createGemma4Ple({
        index,
        openBlock,
        vocabSize: VOCAB,
        // Fully resident (zero re-reads — the budget is derived from the index).
        maxResidentBytes: allResidentBytes(index),
      });
      pair = await openDrafterPair();
    });

    afterAll(async () => {
      try {
        await pair?.close();
      } finally {
        ple?.dispose();
      }
    });

    /** One run of the target (it returns the hidden of the selected row). */
    const runTarget = async (
      context: GenerationContext,
      ids: Int32Array<ArrayBuffer>,
      positions: Int32Array<ArrayBuffer>,
      queryLength: number,
      where: string,
    ): Promise<Float32Array<ArrayBuffer>> => {
      assert(pair !== undefined && ple !== undefined, "the target / drafter pair is not open");
      const { target, targetGraph } = pair;
      const logitsName = targetGraph.outputs[0];
      const hiddenName = targetGraph.outputs[1];
      const perLayer = await ple.gather([...ids]);
      const outputs = await target.run(
        {
          [INPUT_IDS]: { dtype: "i32", shape: [1, ids.length], data: ids },
          [PER_LAYER_INPUTS]: perLayer,
          ...ropeInputs(config.rope, positions),
          [LAST_ROW]: { dtype: "i32", shape: [1], data: Int32Array.of(queryLength - 1) },
        },
        undefined,
        // MUST: `commit: "deferred"` is not used (a draft is refused while an uncommitted run
        // remains — the draft is taken after the commit).
        { context, queryLength },
      );
      assertEquals(outputs[logitsName].shape, [1, 1, VOCAB], `${where}: logits shape`);
      assertEquals(outputs[hiddenName].shape, [1, 1, HIDDEN], `${where}: hidden shape`);
      return f32Of(outputs[hiddenName], `${where}: hidden`);
    };

    for (const name of CASES) {
      it(`${name}: compares 200 cycles × 3 steps of drafts with the golden`, async () => {
        assert(pair !== undefined, "the target / drafter pair is not open");
        const { target, drafter } = pair;
        const golden = await readGolden(name);
        const prompt = golden.prompt;
        const promptLength = prompt.length;
        // The capacity this conversation needs (prompt + continuation — some cases do not fit
        // the distribution's default of 4096).
        const capacity = promptLength + golden.cycles;
        assert(
          capacity <= config.maxPosition,
          `${name}: capacity ${capacity} exceeds maxPosition ${config.maxPosition}`,
        );
        const started = performance.now();
        const context = await target.createGenerationContext({
          bindings: { [CAPACITY_SYMBOL]: capacity },
          chunkLength: CHUNK_LENGTH,
        });
        const borrowed = await openDrafterContext(drafter, context);
        const perStep = Array.from({ length: GEMMA4_DRAFT_STEPS }, () => 0);
        /**
         * The set of draft tokens produced (a field that shows on every run that the comparison is
         * not trivial).
         */
        const distinct = new Set<number>();
        try {
          // Phase 1: prefill the prompt (the row of the last chunk gives the hidden at position
          // T−1).
          let hidden!: Float32Array<ArrayBuffer>;
          const chunks = planPrefillChunks(promptLength, CHUNK_LENGTH);
          for (const chunk of chunks) {
            const ids = new Int32Array(new ArrayBuffer(CHUNK_LENGTH * 4));
            const positions = new Int32Array(new ArrayBuffer(CHUNK_LENGTH * 4));
            for (let row = 0; row < chunk.queryLength; row += 1) {
              ids[row] = prompt[chunk.position + row];
              positions[row] = chunk.position + row;
            }
            hidden = await runTarget(
              context,
              ids,
              positions,
              chunk.queryLength,
              `${name} prefill@${chunk.position}`,
            );
          }

          // Phase 2: the draft of cycle t. The token passed is **the last one confirmed** (the
          // continuation's `tokens[t]` — at position P = T+t but not yet in the KV), and the hidden
          // is that of the row that produced it (position P−1). For t>0 the previous continuation
          // token is fed and that row's hidden is taken (teacher forcing).
          for (let cycle = 0; cycle < golden.cycles; cycle += 1) {
            const position = promptLength + cycle;
            const token = golden.tokens[cycle];
            if (cycle > 0) {
              hidden = await runTarget(
                context,
                Int32Array.of(golden.tokens[cycle - 1]),
                Int32Array.of(position - 1),
                1,
                `${name} decode@${cycle}`,
              );
            }
            assertEquals(context.pastLength, position, `${name}: logical length at cycle ${cycle}`);
            const draft = await draftOnce(drafter, borrowed, { token, hidden, position });
            assertEquals(draft.length, GEMMA4_DRAFT_STEPS, `${name}: number of drafts`);
            for (let step = 0; step < GEMMA4_DRAFT_STEPS; step += 1) {
              distinct.add(draft[step]);
              if (draft[step] === golden.draft[cycle * GEMMA4_DRAFT_STEPS + step]) {
                perStep[step] += 1;
              }
            }
          }
        } finally {
          // MUST: borrower → lender (the runtime refuses the reverse order).
          await borrowed.dispose();
          await context.dispose();
        }
        // MUST: check that the comparison is not trivial (if the drafter kept emitting one word,
        // the match rate would fall to the golden's acceptance rate, but a stuck `argmax` shows up
        // first as "the same word at every position").
        assert(
          distinct.size >= 32,
          `${name}: only ${distinct.size} distinct draft tokens (the drafter's output is stuck)`,
        );
        const total = perStep.reduce((sum, value) => sum + value, 0);
        counts.push({ perStep, total, cycles: golden.cycles });
        console.log(
          `[e2e] gemma4 drafter ${name}: T=${promptLength} / ${golden.cycles} cycles / matched ` +
            `${total}/${golden.cycles * GEMMA4_DRAFT_STEPS} (per step ${perStep
              .map((hit, step) => `${step + 1}: ${((hit / golden.cycles) * 100).toFixed(1)}%`)
              .join(" / ")}) / ${(performance.now() - started).toFixed(0)}ms`,
        );
      });
    }

    it("① the match rate over the 3 cases meets the floor", () => {
      const matched = counts.reduce((sum, entry) => sum + entry.total, 0);
      const drafts = counts.reduce((sum, entry) => sum + entry.cycles * GEMMA4_DRAFT_STEPS, 0);
      assertEquals(counts.length, CASES.length, "number of cases compared");
      const rate = matched / drafts;
      console.log(
        `[e2e] gemma4 drafter total: ${matched}/${drafts} = ${(rate * 100).toFixed(2)}%` +
          ` (floor ${(MATCH_FLOOR * 100).toFixed(0)}%)`,
      );
      assert(
        rate >= MATCH_FLOOR,
        `the draft match rate ${(rate * 100).toFixed(2)}% is below the floor ${(
          MATCH_FLOOR * 100
        ).toFixed(0)}% (per step: ${counts
          .map(
            (entry, index) =>
              `${CASES[index]} ${entry.perStep.map((hit) => `${hit}/${entry.cycles}`).join(",")}`,
          )
          .join(" / ")})`,
      );
    });
  },
);

// ---------------------------------------------------------------------------
// ⑤ The pipeline's speculative option (real GPU) — builds as far as the drafter Session, and not
//   one generated token changes
// ---------------------------------------------------------------------------

/**
 * The chat acceptance case (**the same golden** as `e2e_gemma4_pretrained_test.ts` — it does not
 * move with or without speculation).
 *
 * `stop` has the same value too (speculation **only adds** the `speculation` field and does not
 * move the reason, the stop token or the generated token count at all — if they moved, the
 * speculative path would be turning the conversation into a different sequence).
 */
const CHAT_CASE = {
  fixture: "single-user",
  maxNewTokens: 24,
  expected: "The capital of France is **Paris**.",
  stop: { reason: "eos", token: 106, tokens: 9 },
} as const;

type ChatFixture = {
  readonly chat: { readonly name: string; readonly messages: ChatMessage[] }[];
};

describe.skipIf(!MIRROR_PRESENT || !GPU_AVAILABLE)(
  "gemma.Pipeline: passing speculative loads the drafter too, and the chat golden does not move (real GPU)",
  () => {
    let chatCase: ChatFixture["chat"][number];
    let pipeline: Pipeline | undefined;

    beforeAll(async () => {
      const fixture: ChatFixture = JSON.parse(
        await readTextFile(
          new URL("../../models/tests/fixtures/gemma-text/gemma4-chat.json", import.meta.url),
        ),
      );
      const found = fixture.chat.find((row) => row.name === CHAT_CASE.fixture);
      assert(found !== undefined, `the fixture has no chat case '${CHAT_CASE.fixture}'`);
      chatCase = found;

      const started = performance.now();
      pipeline = await Pipeline.fromPretrained(nodeDirectory(MIRROR_DIR), {
        ...BF16_SEAT,
        speculative: { k: GEMMA4_DRAFT_STEPS },
        maxResidentPleBytes: await allResidentPleBytesOfMirror(MIRROR_DIR, BF16_SEAT),
      });
      console.log(
        `[e2e] gemma4 drafter pipeline: load with speculation ${(
          performance.now() - started
        ).toFixed(0)}ms`,
      );
    });

    afterAll(async () => {
      // If the order (drafter Session → target Session) is right, this call returns successfully
      // (in the reverse order it hits the lending check and `disposeSteps` carries the failure).
      await pipeline?.dispose();
    });

    // NOTE: this pipeline has the drafter, so `chat` **speculates by default** (step 3). The strict
    // checks of the acceptance rate, token-sequence agreement with non-speculative decoding, the
    // phases and the estimate belong to `e2e_gemma4_speculative_test.ts` — this file checks that
    // "speculation is actually set up on the default path (chat), and still the golden does not
    // move by a single character".
    it("① even with speculation the chat golden does not move by a single character", async () => {
      assert(pipeline !== undefined, "the pipeline is not open");
      const stream = pipeline.chat(chatCase.messages, {
        maxNewTokens: CHAT_CASE.maxNewTokens,
        sampler: { temperature: 0 },
      });
      assertEquals(await stream.text(), CHAT_CASE.expected, "output at temperature 0");
      const stop = await stream.done;
      assertEquals(
        stop.tokens,
        CHAT_CASE.stop.tokens,
        "generated token count (stop token included)",
      );
      assert(stop.reason === "eos", `the stop reason is not eos: ${JSON.stringify(stop)}`);
      assertEquals(stop.token, CHAT_CASE.stop.token, "stop token");
      // MUST: check here that speculation was **actually set up** (if not, ① would turn into "a
      // re-run of the non-speculative golden" and the speculative path would never run).
      // `cycles > 0` means verify runs happened, `accepted > 0` that at least one draft was
      // accepted — without the latter it cannot be told apart from "every cycle rejected
      // everything = the same sequence as decode, by a detour".
      const speculation = stop.speculation;
      assert(
        speculation !== undefined,
        `speculation is not set up by default: ${JSON.stringify(stop)}`,
      );
      assert(speculation.cycles > 0, "not a single verify run happened");
      assert(
        speculation.accepted > 0,
        `not a single draft was accepted（${speculation.cycles} cycle / ` +
          `drafted ${speculation.drafted}）`,
      );
      console.log(
        `[e2e] gemma4 drafter chat: ${speculation.cycles} cycle / accepted ` +
          `${speculation.accepted}/${speculation.drafted} / delivered ${stop.tokens} token`,
      );
    });

    it("② the estimate adds up the residency of target + drafter", () => {
      assert(pipeline !== undefined, "the pipeline is not open");
      const opened = pipeline;
      const report = opened.estimateSessionMemory();
      assert(report.resident.weights.totalBytes > 0, "the resident weight estimate is 0 bytes");
      assert(
        report.auxiliaryBytes !== undefined && report.auxiliaryBytes > 0,
        "the accounting of the auxiliary output, usable in non-speculative conversations too, is missing",
      );
      // The inner max is the maximum over the public scenarios plus the prefill bucket shapes the
      // sequence runs (which are not listed among the scenarios) (the estimate's spec changed in
      // ADR 0095 addendum 2026-10-10 — an intermediate bucket shape can exceed the prefill shape
      // at a row-block count boundary, so the max of the public scenarios alone is not the
      // formula). The value of a bucket shape is taken as "the prefill shape estimated with that
      // row count as the chunk length" (the same bindings and computation; only the physical row
      // count differs).
      // NOTE: on the real gemma4 E2B asset the bucket shapes do not dominate the peak, so this
      // equality does not discriminate the wiring — gemma4_estimate_test.ts (no GPU) pins the
      // wiring.
      const { chunkBuckets } = opened.program;
      assert(
        chunkBuckets.length > 0,
        "no buckets (the check would not look at the bucket-shape accounting)",
      );
      const bucketShapes = chunkBuckets.map((rows) => {
        const prefill = opened
          .estimateSessionMemory({ chunkLength: rows })
          .scenarios.find((one) => one.name === "prefill");
        assert(prefill !== undefined, `the report has no prefill shape for the ${rows}-row bucket`);
        return prefill.ioBytes + prefill.workspaceBytes;
      });
      assertEquals(
        report.peakAccountedBytes,
        report.resident.weights.totalBytes +
          report.resident.stateBytes +
          Math.max(
            report.planBackingBudgetBytes,
            ...report.scenarios.map((one) => one.ioBytes + one.workspaceBytes),
            ...bucketShapes,
          ) +
          report.auxiliaryBytes,
        "after adding the drafter, the extra accounting and the peak breakdown still agree",
      );
      // The exact-match checks (comparison with the two Sessions' diagnostics, the borrower
      // state's 8 bytes, the verify scenario's ioBytes) are added by step 3's real-GPU leg.
      assertEquals(
        report.scenarios.map((scenario) => scenario.name),
        ["prefill", "decode", "verify"],
        "the verify scenario is not listed",
      );
    });
  },
);
