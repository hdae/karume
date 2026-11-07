// Real-weight Gemma 4 E2B / E4B / 12B of the q4_0 QAT line through its distribution mirror
// (`models/karume-gemma4-qat/` — `dist.py --pipeline gemma4-qat`): the regular gemma pipeline opens
// it with the QAT MTP drafters (ADR 0096) borrowing the int4 embedding table of their target.
//
// Every weight of the line is the checkpoint's own 4-bit grid (int4 g32), the drafters' included,
// so the drafter of this line is int4 where the regular E2B drafter is int8, and it borrows an int4
// table whose scale is split on the table's pieces (ADR 0108 addendum 15). The cases, per model:
//
// ① the manifest entry (no GPU): weights `model` + `drafter`, one storage `i4` each; the default
//    seat `i4` (session {}) and the regular line's parallel and fused seats
//    `i4-gemvpar` / `i4-fast` (their session flags over the same containers), every seat naming
//    both roles as `i4`; 12B keeps its dialect and suppressed ids
// ② the drafter container (no GPU): the admission the pipeline runs pairs it with its target —
//    the four external slots are the target's last sliding / full owner layers with the target's
//    shapes (12B: `l46` [1,8,1032,256] / `l47` [1,1,C,512]), the one shared initializer is the
//    target's `model.lm_head.weight`, and the `hidden` input is the target's width
// ③ speculation through the mirror (real GPU): `fromPretrained` with `speculative` builds the
//    drafter Session on the borrowed table, and a greedy chat turn with `speculative: "always"`
//    gives the same text, stop token and token count as the same turn without it, with at least
//    one draft accepted (the speculative path really ran) — for every seat of the model, the
//    parallel and fused ones included (the regular line's check of its `i4-fast` seat: the
//    verify rows M = k + 1 and the decode row M = 1 take the same summation order)
// ④ greedy parity (real GPU): the greedy chain of each golden case of the product series, run
//    through the mirror both without and with speculation, equals the torch golden token for token
//    (each golden's top1 − top2 margin exceeds 2 × atol, so the comparison does not hinge on
//    numeric noise — token identity, not a tolerance band, like `e2e_gemma4_12b_test.ts` ⑤)
// ⑤ draft parity (real GPU, upstream parity — opt-in KARUME_UPSTREAM_PARITY=1 like
//    `e2e_gemma4_drafter_test.ts`): the drafter's drafts, teacher-forced along the golden
//    continuation of the drafter series (3 cases × 200 cycles × 3 steps), match the torch drafter's
//    in at least 99% of the positions (`e2e_gemma4_drafter_test.ts`'s floor — MUST NOT be moved to
//    the measurement), and the drafts are not stuck (at least 32 distinct ids per case, or the torch
//    drafter's own count where its golden repeats itself — 12B's short-en)
//
// ## Assets
//
// The weights are the local mirror (ADR 0139 decision 1); without it every case SKIPs explicitly.
// The goldens are the recipe outputs in the test-asset root (decision 2): ④ reads
// `goldens/gemma4-<model>-qat-product/` (12B `goldens/gemma4-12b-product/`, shared with
// `e2e_gemma4_12b_test.ts`) — `greedy.<case>.safetensors` and the `reference.json` that binds
// them — and ⑤ reads `goldens/gemma4-<model>-qat-drafter/drafter-golden.<case>.safetensors`, under
// the fetched rule: absent from a complete fetch of the pinned dataset ⇒ FAIL (the per-model asset
// case); otherwise SKIP with the missing files and `node tools/test-assets/fetch.ts` in the title.
//
// ## GPU memory
//
// One pipeline (or one target + drafter pair) at a time; 12B with its drafter is about 7.2 GiB of
// weights + 0.8 GiB of slots at the default capacity 4096 — within a 12 GB GPU.

import { isNotFound, readFile, readTextFileSync, statSync } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { type Manifest, MANIFEST_FILENAME, parseManifest } from "@karume/hub";
import {
  acquireGpu,
  type GenerationContext,
  parseSafetensors,
  type SafetensorsFile,
  type Session,
  type Tensor,
} from "@karume/runtime";
import {
  type ChatMessage,
  type GenerationSequence,
  Pipeline,
  parsePipelineConfig,
  type PipelineConfig,
} from "../../models/gemma.ts";
import { createGemma4Ple, type Gemma4Ple } from "../../models/src/gemma/ple.ts";
import { gemma4RopeInputNames, ropeInputs } from "../../models/src/gemma/rope.ts";
import {
  admitGemma4Drafter,
  draftOnce,
  GEMMA4_DRAFT_STEPS,
  type Gemma4Drafter,
  openDrafterContext,
} from "../../models/src/gemma/speculative.ts";
import { planPrefillChunks } from "../../models/src/generation/greedy.ts";
import { openGemma4Ple } from "../../models/tests/helpers/gemma-mirror.ts";
import {
  mirrorDirectory,
  mirrorHasModel,
  mirrorRoot,
  openMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { allResidentBytes } from "../../models/tests/helpers/ple-budget.ts";
import {
  goldenDir,
  TEST_ASSETS,
  TEST_ASSETS_ROOT,
  testAssetsFetched,
} from "@karume/test-support/test-assets";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

const MIRROR = mirrorRoot("karume-gemma4-qat");
const ASSEMBLE =
  "cd tools/export-recipes && uv run python dist.py --pipeline gemma4-qat" +
  " --model e2b --model e4b --model 12b";
/** The command that restores the goldens (ADR 0139 decision 2). */
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";

const MODELS = ["e2b", "e4b", "12b"] as const;
type Model = (typeof MODELS)[number];

/** The roles of the manifest (`gemma4/distribution.py`). */
const MODEL = "model";
const DRAFTER = "drafter";
/** The line's default seat and its one dtype label. */
const QUANT = "i4";
/**
 * The seats per model (`gemma4/distribution.py` — `gemma4_qat_q4_0_quants`): the reference seat
 * `i4` and the parallel and fused seats over the same containers.
 */
const SEATS: Record<Model, readonly string[]> = {
  e2b: [QUANT, "i4-gemvpar", "i4-fast"],
  e4b: [QUANT, "i4-gemvpar", "i4-fast"],
  "12b": [QUANT, "i4-gemvpar", "i4-fast"],
};
/** The session each seat declares (`GEMMA4_QAT_Q4_0_PARALLEL_QUANTS`). */
const SEAT_SESSIONS: Readonly<Record<string, Readonly<Record<string, unknown>>>> = {
  [QUANT]: {},
  "i4-gemvpar": { linearGemvReduce: "parallel", stateAttentionReduce: "parallel" },
  "i4-fast": {
    linearGemvReduce: "parallel",
    fuseRmsNormAdd: true,
    stateAttentionReduce: "parallel-fused",
  },
};
/** The table the drafter borrows (its name is the lender's initializer name — docs/ir-v2.md). */
const SHARED_WEIGHT = "model.lm_head.weight";
const CAPACITY_SYMBOL = "C";
const VOCAB = 262144;

/**
 * Per model: the target's hidden width, whether it has per-layer embeddings, its product series
 * (E2B / E4B their QAT series, 12B its one line — the goldens' folder in the test-asset root) and
 * the slots its drafter borrows (the last sliding / full KV owner layers).
 */
const SHAPES: Record<
  Model,
  {
    readonly hidden: number;
    readonly ple: boolean;
    readonly product: string;
    readonly slots: Record<string, readonly (number | string)[]>;
  }
> = {
  e2b: {
    hidden: 1536,
    ple: true,
    product: "gemma4-e2b-qat-product",
    slots: {
      "l13.k": [1, 1, 520, 256],
      "l13.v": [1, 1, 520, 256],
      "l14.k": [1, 1, CAPACITY_SYMBOL, 512],
      "l14.v": [1, 1, CAPACITY_SYMBOL, 512],
    },
  },
  e4b: {
    hidden: 2560,
    ple: true,
    product: "gemma4-e4b-qat-product",
    slots: {
      "l22.k": [1, 2, 520, 256],
      "l22.v": [1, 2, 520, 256],
      "l23.k": [1, 2, CAPACITY_SYMBOL, 512],
      "l23.v": [1, 2, CAPACITY_SYMBOL, 512],
    },
  },
  "12b": {
    hidden: 3840,
    ple: false,
    product: "gemma4-12b-product",
    slots: {
      "l46.k": [1, 8, 1032, 256],
      "l46.v": [1, 8, 1032, 256],
      "l47.k": [1, 1, CAPACITY_SYMBOL, 512],
      "l47.v": [1, 1, CAPACITY_SYMBOL, 512],
    },
  },
};

/** Graph inputs of the product (`gemma4/export_product.py`). */
const INPUT_IDS = "input_ids";
const PER_LAYER_INPUTS = "per_layer_inputs";
const LAST_ROW = "last_row";

/** The gemma4 logits tolerance the margin gate is stated against (`e2e_gemma4_12b_test.ts`). */
const ATOL = 1e-2;
/** The draft-match floor (`e2e_gemma4_drafter_test.ts` — the denominator is cases × cycles × k). */
const MATCH_FLOOR = 0.99;
/** The drafter golden cases (`gemma4/export_drafter.py` GOLDEN_CASES). */
const DRAFTER_CASES = ["short-en", "readme-recipes", "readme-exporter"] as const;
const GREEDY_PREFIX = "greedy.";
const DRAFTER_GOLDEN_PREFIX = "drafter-golden.";
const SUFFIX = ".safetensors";
const REFERENCE_FILE = "reference.json";

/** The chat turn of ③ (an English question whose greedy answer runs past a few tokens). */
const CHAT: readonly ChatMessage[] = [
  { role: "user", content: "Explain in about five sentences why the sky is blue." },
];
const CHAT_TOKENS = 64;

const isFile = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/** The goldens of `files` that are absent, as paths under the test-asset root. */
const missingGoldens = (files: readonly URL[]): string[] =>
  files
    .filter((file) => !isFile(file))
    .map((file) => file.href.slice(TEST_ASSETS_ROOT.href.length));

/**
 * ADR 0139's fetched rule for the goldens a case reads: absent with a complete fetch of the pin ⇒
 * the case runs and {@link assertGoldensPresent} FAILs naming them; absent without one ⇒ the case
 * SKIPs, and `title` says why.
 */
const goldensGate = (files: readonly URL[]): { readonly skip: boolean; readonly title: string } => {
  const missing = missingGoldens(files);
  const skip = missing.length > 0 && !testAssetsFetched();
  const title = skip
    ? ` (SKIP: ${missing.join(" / ")} absent and outputs/test-assets/ is not a complete fetch of ` +
      `the pin — ${FETCH_COMMAND})`
    : "";
  return { skip, title };
};

/** FAILs naming each absent golden of `files` (the dataset at the pin lacks it). */
const assertGoldensPresent = (files: readonly URL[]): void => {
  const missing = missingGoldens(files);
  assert(
    missing.length === 0,
    `${missing.join(" / ")} absent although outputs/test-assets/ holds a complete fetch of ` +
      `${TEST_ASSETS.repo}@${TEST_ASSETS.revision} — the test-asset dataset at the pin lacks it`,
  );
};

const PRESENT: Record<Model, boolean> = {
  e2b: mirrorHasModel(MIRROR, "e2b"),
  e4b: mirrorHasModel(MIRROR, "e4b"),
  "12b": mirrorHasModel(MIRROR, "12b"),
};
if (MODELS.some((model) => !PRESENT[model])) {
  console.warn(
    `[karume] ${MIRROR.pathname} lacks ${MODELS.filter((model) => !PRESENT[model]).join(" / ")}` +
      ` (or is not karume/5), so those gemma4-qat mirror cases SKIP. Assemble: ${ASSEMBLE}`,
  );
}

const manifest: Manifest | undefined = MODELS.some((model) => PRESENT[model])
  ? parseManifest(readTextFileSync(new URL(MANIFEST_FILENAME, MIRROR)))
  : undefined;

const entryOf = (model: Model) => {
  assert(manifest !== undefined, "the mirror's manifest is not read");
  return manifest.models[model];
};

const configOf = (model: Model): PipelineConfig =>
  parsePipelineConfig(entryOf(model).pipelineConfig);

/** The options of a load: the whole PLE resident for a PLE model (none for 12B — it refuses). */
const loadOptions = async (model: Model) =>
  SHAPES[model].ple
    ? {
        model,
        maxResidentPleBytes: allResidentBytes((await openGemma4Ple(MIRROR, { model })).index),
      }
    : { model };

// ---------------------------------------------------------------------------
// ① ② the manifest entry and the drafter container (no GPU)
// ---------------------------------------------------------------------------

for (const model of MODELS) {
  test.skipIf(!PRESENT[model])(
    `gemma4-qat ${model}: the manifest ships model and drafter as i4 under its seats (no GPU)`,
    () => {
      const entry = entryOf(model);
      assertEquals(Object.keys(entry.weights).sort(), [DRAFTER, MODEL], "weights roles");
      for (const role of [MODEL, DRAFTER]) {
        assertEquals(Object.keys(entry.weights[role]), [QUANT], `${role}: one storage`);
      }
      assertEquals(Object.keys(entry.quants), [...SEATS[model]], "quants");
      assertEquals(entry.defaultQuant, QUANT, "default quant");
      const quant = entry.quants[QUANT];
      assertEquals(quant.weights, { [MODEL]: QUANT, [DRAFTER]: QUANT }, "the quant's weights");
      assertEquals(quant.session ?? {}, {}, "the reference summation order (session {})");
      for (const name of SEATS[model]) {
        assertEquals(
          entry.quants[name].weights,
          { [MODEL]: QUANT, [DRAFTER]: QUANT },
          `${name}: the same containers`,
        );
        assertEquals(entry.quants[name].session ?? {}, SEAT_SESSIONS[name], `${name}: session`);
      }
      const config = configOf(model);
      if (model === "12b") {
        assertEquals(config.chatDialect, "gemma4-unified", "12B's chat dialect");
        assertEquals(config.suppressTokens, [258882, 258883], "12B's suppressed ids");
      } else {
        assertEquals(config.chatDialect, undefined, `${model}: chat dialect`);
        assertEquals(config.suppressTokens, undefined, `${model}: suppressed ids`);
      }
    },
  );

  test.skipIf(!PRESENT[model])(
    `gemma4-qat ${model}: the drafter pairs with its target — slots, shared table, hidden width (no GPU)`,
    async () => {
      const target = (await openMirrorComponent(MIRROR, MODEL, { model })).graph;
      const drafter = (await openMirrorComponent(MIRROR, DRAFTER, { model })).graph;
      const config = configOf(model);
      assertEquals(
        drafter.inputs.map((input) => input.name),
        ["token", "hidden", ...gemma4RopeInputNames()],
        "drafter inputs (order included)",
      );
      assertEquals(drafter.symbols, [CAPACITY_SYMBOL], "the one symbol, inherited from the lender");
      const slots = SHAPES[model].slots;
      assertEquals(Object.keys(drafter.states).sort(), Object.keys(slots).sort(), "borrowed slots");
      for (const [name, shape] of Object.entries(slots)) {
        assert(drafter.states[name].external === true, `${name}: not external`);
        assertEquals(drafter.states[name].shape, shape, `${name}: the drafter's shape`);
        assertEquals(target.states[name].shape, shape, `${name}: the target's shape`);
      }
      const admitted = admitGemma4Drafter("test", drafter, {
        graph: target,
        rope: config.rope,
        hiddenSize: SHAPES[model].hidden,
        capacitySymbol: CAPACITY_SYMBOL,
      });
      assertEquals(admitted.sharedWeights, { [SHARED_WEIGHT]: SHARED_WEIGHT }, "shared weights");
      assertEquals(admitted.outputs.length, GEMMA4_DRAFT_STEPS, "draft exits");
      assertEquals(drafter.initializers[SHARED_WEIGHT].storage, undefined, "no bytes borrowed");
    },
  );
}

// ---------------------------------------------------------------------------
// ③ speculation through the mirror (real GPU)
// ---------------------------------------------------------------------------

/**
 * ③'s check on one opened pipeline: a greedy chat turn with `speculative: "always"` gives the same
 * text, stop and token count as the plain turn, with at least one draft accepted.
 */
const assertSpeculativeTurn = async (opened: Pipeline, label: string): Promise<void> => {
  const turn = async (speculative: "always" | false) => {
    const started = performance.now();
    const stream = opened.chat(CHAT, {
      maxNewTokens: CHAT_TOKENS,
      sampler: { temperature: 0 },
      speculative,
    });
    const text = await stream.text();
    return { text, stop: await stream.done, ms: performance.now() - started };
  };
  const speculative = await turn("always");
  const plain = await turn(false);
  assertEquals(speculative.text, plain.text, "the text");
  assertEquals(speculative.stop.reason, plain.stop.reason, "the stop reason");
  assertEquals(speculative.stop.tokens, plain.stop.tokens, "the token count");
  assertEquals(plain.stop.speculation, undefined, "a plain turn carries no speculation");
  const speculation = speculative.stop.speculation;
  assert(speculation !== undefined, "the speculative turn did not speculate");
  assert(speculation.cycles > 0, "no verify run");
  assert(
    speculation.accepted > 0,
    `no draft accepted (${speculation.cycles} cycles, ${speculation.drafted} drafted)`,
  );
  console.log(
    `[e2e] gemma4-qat ${label} chat: ${speculative.stop.tokens} tokens /` +
      ` ${speculation.cycles} cycles / accepted ${speculation.accepted}/${speculation.drafted}` +
      ` / ${(speculation.delivered / speculation.cycles).toFixed(3)} tokens per cycle /` +
      ` speculative ${speculative.ms.toFixed(0)} ms vs plain ${plain.ms.toFixed(0)} ms`,
  );
};

for (const model of MODELS) {
  describe.skipIf(!PRESENT[model] || !GPU_AVAILABLE)(
    `gemma4-qat ${model}: a speculative greedy chat turn equals the plain one (real GPU)`,
    () => {
      let pipeline: Pipeline | undefined;
      beforeAll(async () => {
        const started = performance.now();
        pipeline = await Pipeline.fromPretrained(mirrorDirectory(MIRROR), {
          ...(await loadOptions(model)),
          speculative: { k: GEMMA4_DRAFT_STEPS },
        });
        console.log(
          `[e2e] gemma4-qat ${model}: speculative load ${(performance.now() - started).toFixed(0)} ms`,
        );
      });
      afterAll(async () => {
        await pipeline?.dispose();
      });

      it("③ the same text, stop and token count, with drafts accepted", async () => {
        assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
        await assertSpeculativeTurn(pipeline, model);
      });
    },
  );

  // The parallel and fused seats: one load each, the seat named (the default is `i4`).
  for (const quant of SEATS[model].filter((name) => name !== QUANT)) {
    describe.skipIf(!PRESENT[model] || !GPU_AVAILABLE)(
      `gemma4-qat ${model} ${quant}: a speculative greedy chat turn equals the plain one (real GPU)`,
      () => {
        let pipeline: Pipeline | undefined;
        beforeAll(async () => {
          pipeline = await Pipeline.fromPretrained(mirrorDirectory(MIRROR), {
            ...(await loadOptions(model)),
            quant,
            speculative: { k: GEMMA4_DRAFT_STEPS },
          });
        });
        afterAll(async () => {
          await pipeline?.dispose();
        });

        it(`③ ${quant}: the same text, stop and token count, with drafts accepted`, async () => {
          assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
          await assertSpeculativeTurn(pipeline, `${model} ${quant}`);
        });
      },
    );
  }
}

// ---------------------------------------------------------------------------
// ④ greedy parity through the mirror (real GPU, implementation-time)
// ---------------------------------------------------------------------------

type ObjectRecord = Record<string, unknown>;

const objectAt = (value: unknown, where: string): ObjectRecord => {
  assert(
    typeof value === "object" && value !== null && !Array.isArray(value),
    `${REFERENCE_FILE}: ${where} is not an object`,
  );
  return value as ObjectRecord;
};

const bufferOf = (bytes: Uint8Array<ArrayBuffer>): ArrayBuffer =>
  bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? bytes.buffer
    : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

const i32Of = (file: SafetensorsFile, name: string): Int32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined && view.dtype === "I32", `golden '${name}' is not I32`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/** The greedy goldens the product series' record binds (sorted case names). */
const boundCases = (root: URL): string[] =>
  Object.keys(
    objectAt(
      objectAt(
        objectAt(JSON.parse(readTextFileSync(new URL(REFERENCE_FILE, root))), "root").reference,
        "reference",
      ).goldens,
      "reference.goldens",
    ),
  )
    .map((file) => file.slice(GREEDY_PREFIX.length, -SUFFIX.length))
    .sort();

/** One greedy golden, checked against the digest its record binds. */
const readGreedy = async (
  root: URL,
  name: string,
): Promise<{ prompt: number[]; expected: number[]; margin: number[] }> => {
  const file = `${GREEDY_PREFIX}${name}${SUFFIX}`;
  const bytes = await readFile(new URL(file, root));
  const record = objectAt(JSON.parse(readTextFileSync(new URL(REFERENCE_FILE, root))), "root");
  const digest = objectAt(
    objectAt(objectAt(record.reference, "reference").goldens, "reference.goldens")[file],
    file,
  );
  assertEquals(await sha256Hex(bytes), digest.sha256, `${file}: the sha256 its record binds`);
  const golden = parseSafetensors(bufferOf(bytes));
  const margin = golden.tensors.get("margin");
  assert(margin !== undefined && margin.dtype === "F32", `${name}: 'margin' is not F32`);
  return {
    prompt: [...i32Of(golden, "prompt")],
    expected: [...i32Of(golden, "expected")],
    margin: [...new Float32Array(golden.buffer, margin.byteOffset, margin.byteLength / 4)],
  };
};

/**
 * The greedy chain of `steps` tokens from `prompt`, stop tokens included (the chain of
 * `e2e_gemma4_qat_product_test.ts`): a stop token ends a turn and stays the pending frontier, so
 * the next `generate` with an empty prompt continues one uninterrupted greedy run.
 */
const greedyChain = async (
  sequence: GenerationSequence,
  prompt: readonly number[],
  steps: number,
): Promise<number[]> => {
  const ids: number[] = [];
  let next: readonly number[] = prompt;
  while (ids.length < steps) {
    const stream = sequence.generate({ prompt: [...next], maxNewTokens: steps - ids.length });
    for await (const event of stream) {
      if (event.kind === "token") ids.push(event.id);
    }
    const stop = await stream.done;
    if (stop.reason === "max-tokens") break;
    assert(stop.reason === "eos", `unexpected stop '${stop.reason}' after ${ids.length} tokens`);
    ids.push(stop.token);
    next = [];
  }
  return ids;
};

for (const model of MODELS) {
  const root = goldenDir(SHAPES[model].product);
  const record = new URL(REFERENCE_FILE, root);
  const cases = isFile(record) ? boundCases(root) : [];
  const files = [record, ...cases.map((name) => new URL(`${GREEDY_PREFIX}${name}${SUFFIX}`, root))];
  const gate = goldensGate(files);
  test.skipIf(!PRESENT[model] || gate.skip)(
    `gemma4-qat ${model}: the greedy goldens are in the test-asset root${gate.title}`,
    () => {
      assertGoldensPresent(files);
      assert(cases.length > 0, `${record.pathname}: the record binds no greedy golden`);
    },
  );
  describe.skipIf(!PRESENT[model] || gate.skip || cases.length === 0 || !GPU_AVAILABLE)(
    `gemma4-qat ${model}: greedy parity through the mirror, plain and speculative (real GPU)${gate.title}`,
    () => {
      let pipeline: Pipeline | undefined;
      beforeAll(async () => {
        pipeline = await Pipeline.fromPretrained(mirrorDirectory(MIRROR), {
          ...(await loadOptions(model)),
          speculative: { k: GEMMA4_DRAFT_STEPS },
        });
      });
      afterAll(async () => {
        await pipeline?.dispose();
      });

      for (const name of cases) {
        it(`④ ${name}: the greedy chain equals the golden, with and without speculation`, async () => {
          assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
          const { prompt, expected, margin } = await readGreedy(root, name);
          const minMargin = Math.min(...margin);
          assert(
            minMargin > 2 * ATOL,
            `${name}: the golden's smallest margin ${minMargin} is comparable to atol (not a valid gate)`,
          );
          for (const speculative of [false, "always"] as const) {
            const sequence = await pipeline.sequence({ speculative });
            try {
              const started = performance.now();
              const ids = await greedyChain(sequence, prompt, expected.length);
              assertEquals(ids, expected, `${name} (speculative ${speculative}): greedy tokens`);
              console.log(
                `[e2e] gemma4-qat ${model} ${name} (speculative ${speculative}):` +
                  ` T=${prompt.length} + K=${expected.length} / min margin` +
                  ` ${minMargin.toExponential(3)} / ${(performance.now() - started).toFixed(0)} ms`,
              );
            } finally {
              await sequence.dispose();
            }
          }
        });
      }
    },
  );
}

// ---------------------------------------------------------------------------
// ⑤ draft parity (real GPU, implementation-time)
// ---------------------------------------------------------------------------

/** One drafter golden (`prompt [T]` / `tokens [N + k]` / `draft [N, k]`). */
type DrafterGolden = {
  readonly prompt: Int32Array<ArrayBuffer>;
  readonly tokens: Int32Array<ArrayBuffer>;
  readonly draft: Int32Array<ArrayBuffer>;
  readonly cycles: number;
};

const readDrafterGolden = async (root: URL, name: string): Promise<DrafterGolden> => {
  const file = parseSafetensors(
    bufferOf(await readFile(new URL(`${DRAFTER_GOLDEN_PREFIX}${name}${SUFFIX}`, root))),
  );
  const draft = file.tensors.get("draft");
  assert(draft !== undefined, `${name}: no 'draft'`);
  assertEquals(draft.shape[1], GEMMA4_DRAFT_STEPS, `${name}: draft steps`);
  const tokens = i32Of(file, "tokens");
  assertEquals(draft.shape[0] + GEMMA4_DRAFT_STEPS, tokens.length, `${name}: cycles`);
  return {
    prompt: i32Of(file, "prompt"),
    tokens,
    draft: i32Of(file, "draft"),
    cycles: draft.shape[0],
  };
};

const f32Of = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  assert(tensor.data instanceof Float32Array, `${where}: not f32`);
  return tensor.data;
};

for (const model of MODELS) {
  const root = goldenDir(`gemma4-${model}-qat-drafter`);
  const files = DRAFTER_CASES.map(
    (name) => new URL(`${DRAFTER_GOLDEN_PREFIX}${name}${SUFFIX}`, root),
  );
  const gate = goldensGate(files);
  test.skipIf(!UPSTREAM_PARITY || !PRESENT[model] || gate.skip)(
    `gemma4-qat ${model}: the drafter goldens are in the test-asset root${gate.title}${UPSTREAM_PARITY_OPT_IN}`,
    () => assertGoldensPresent(files),
  );
  describe.skipIf(!UPSTREAM_PARITY || !PRESENT[model] || gate.skip || !GPU_AVAILABLE)(
    `gemma4-qat ${model}: drafts match the torch drafter in ≥ 99% (real GPU)${gate.title}${UPSTREAM_PARITY_OPT_IN}`,
    () => {
      const shape = SHAPES[model];
      let config: PipelineConfig;
      let ple: Gemma4Ple | undefined;
      let target: Session | undefined;
      let drafterSession: Session | undefined;
      let drafter: Gemma4Drafter | undefined;
      let targetOutputs: readonly string[] = [];
      let destroy: (() => void) | undefined;
      const counts: { perStep: number[]; total: number; cycles: number }[] = [];

      beforeAll(async () => {
        config = configOf(model);
        if (shape.ple) {
          const { index, openBlock } = await openGemma4Ple(MIRROR, { model });
          ple = createGemma4Ple({
            index,
            openBlock,
            vocabSize: VOCAB,
            maxResidentBytes: allResidentBytes(index),
          });
        }
        const targetOpened = await openMirrorComponent(MIRROR, MODEL, { model });
        const drafterOpened = await openMirrorComponent(MIRROR, DRAFTER, { model });
        const admitted = admitGemma4Drafter("test", drafterOpened.graph, {
          graph: targetOpened.graph,
          rope: config.rope,
          hiddenSize: shape.hidden,
          capacitySymbol: CAPACITY_SYMBOL,
        });
        targetOutputs = targetOpened.graph.outputs;
        const gpu = await acquireGpu();
        destroy = () => gpu.destroy();
        // The line's one seat runs the reference summation order (session {}).
        const lender = await targetOpened.createSession(gpu, {});
        target = lender;
        drafterSession = await drafterOpened.createSession(gpu, {
          sharedWeights: Object.fromEntries(
            Object.keys(admitted.sharedWeights).map((borrower) => [
              borrower,
              lender.exportWeight(admitted.sharedWeights[borrower]),
            ]),
          ),
        });
        drafter = {
          session: drafterSession,
          graph: drafterOpened.graph,
          outputs: admitted.outputs,
          rope: config.rope,
          hiddenSize: admitted.hiddenSize,
        };
      });

      afterAll(async () => {
        try {
          // MUST: the borrower first (the lender refuses while its table is lent out).
          await drafterSession?.dispose();
          await target?.dispose();
          destroy?.();
        } finally {
          ple?.dispose();
        }
      });

      /** One target run; returns the selected row's hidden. */
      const runTarget = async (
        context: GenerationContext,
        ids: Int32Array<ArrayBuffer>,
        positions: Int32Array<ArrayBuffer>,
        queryLength: number,
      ): Promise<Float32Array<ArrayBuffer>> => {
        assert(target !== undefined, "the target is not open");
        const inputs: Record<string, Tensor> = {
          [INPUT_IDS]: { dtype: "i32", shape: [1, ids.length], data: ids },
          ...ropeInputs(config.rope, positions),
          [LAST_ROW]: { dtype: "i32", shape: [1], data: Int32Array.of(queryLength - 1) },
        };
        if (ple !== undefined) inputs[PER_LAYER_INPUTS] = await ple.gather([...ids]);
        const outputs = await target.run(inputs, undefined, { context, queryLength });
        return f32Of(outputs[targetOutputs[1]], "hidden");
      };

      for (const name of DRAFTER_CASES) {
        it(`⑤ ${name}: 200 cycles × 3 steps against the golden`, async () => {
          assert(target !== undefined && drafter !== undefined, "the pair is not open");
          const golden = await readDrafterGolden(root, name);
          const chunkLength = config.chunkLength;
          const capacity = Math.max(chunkLength, golden.prompt.length + golden.cycles);
          const started = performance.now();
          const context = await target.createGenerationContext({
            bindings: { [CAPACITY_SYMBOL]: capacity },
            chunkLength,
          });
          const borrowed = await openDrafterContext(drafter, context);
          const perStep = Array.from({ length: GEMMA4_DRAFT_STEPS }, () => 0);
          const distinct = new Set<number>();
          // The floor of distinct drafts (a stuck drafter exit emits a handful of ids and could
          // still match a degenerate continuation): 32, or the torch drafter's own count when its
          // golden repeats itself — 12B's short-en continuation cycles through 14 ids and its
          // torch drafts hold 21, so 32 would reject the reference itself.
          const distinctFloor = Math.min(32, new Set(golden.draft).size);
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
              for (let step = 0; step < GEMMA4_DRAFT_STEPS; step += 1) {
                distinct.add(draft[step]);
                if (draft[step] === golden.draft[cycle * GEMMA4_DRAFT_STEPS + step]) {
                  perStep[step] += 1;
                }
              }
            }
          } finally {
            await borrowed.dispose();
            await context.dispose();
          }
          assert(
            distinct.size >= distinctFloor,
            `${name}: only ${distinct.size} distinct draft tokens (floor ${distinctFloor})`,
          );
          const total = perStep.reduce((sum, value) => sum + value, 0);
          counts.push({ perStep, total, cycles: golden.cycles });
          console.log(
            `[e2e] gemma4-qat ${model} drafter ${name}: T=${golden.prompt.length} /` +
              ` match ${total}/${golden.cycles * GEMMA4_DRAFT_STEPS} (per step ${perStep
                .map((hit) => `${((hit / golden.cycles) * 100).toFixed(1)}%`)
                .join(" / ")}) / ${(performance.now() - started).toFixed(0)} ms`,
          );
        });
      }

      it("⑤ the three cases together reach the floor", () => {
        assertEquals(counts.length, DRAFTER_CASES.length, "cases matched");
        const matched = counts.reduce((sum, entry) => sum + entry.total, 0);
        const drafts = counts.reduce((sum, entry) => sum + entry.cycles * GEMMA4_DRAFT_STEPS, 0);
        const rate = matched / drafts;
        console.log(
          `[e2e] gemma4-qat ${model} drafter total: ${matched}/${drafts} =` +
            ` ${(rate * 100).toFixed(2)}%`,
        );
        assert(
          rate >= MATCH_FLOOR,
          `${model}: draft match ${(rate * 100).toFixed(2)}% under the ${MATCH_FLOOR * 100}% floor` +
            ` (per case ${counts
              .map((entry, index) => `${DRAFTER_CASES[index]} ${entry.perStep.join(",")}`)
              .join(" / ")})`,
        );
      });
    },
  );
}
