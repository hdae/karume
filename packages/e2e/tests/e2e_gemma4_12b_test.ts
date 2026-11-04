// Real-weight Gemma 4 12B (gemma-4-12B-it) — the distribution entry, the PLE-free product graph,
// the provenance of its goldens and the greedy parity through the public pipeline.
//
// 12B is not "E4B but bigger": upstream ships it as `gemma4_unified` with no PLE (per-layer
// inputs), no KV sharing, a 1024 window and 8 / 1 KV heads on sliding / full layers, and its own
// tokenizer and chat template. The cases:
//
// ① the manifest entry: model `12b` in the q4_0 QAT line's distribution (`karume-gemma4-qat`, the
//    one distribution that ships 12B), one quant `i4` (session {}), the 12B pipelineConfig
//    (chatDialect "gemma4-unified", suppressTokens [258882, 258883] — upstream generation_config's
//    suppress_tokens, sorted — and maxPosition 262144), its own tokenizer asset and its QAT drafter
//    (`e2e_gemma4_qat_mirror_test.ts` pins the drafter itself); the E2B / E4B entries keep their
//    configs without either key
// ② the product graph without PLE: inputs = input_ids / four RoPE / last_row (no per_layer_inputs),
//    outputs = logits [1,R,V] then hidden [1,R,3840], 96 state slots (48 layers × k / v — no KV
//    sharing; sliding [1,8,1032,256], full [1,1,C,512]) and no PLE asset in the container
// ③ the PLE placement options are refused at admission (before the GPU; that no weight part is
//    fetched first is pinned GPU-free and asset-free by `models/tests/gemma4_ple_free_test.ts`)
// ④ the provenance record binds the greedy goldens the parity case reads, records the suppressed
//    ids the goldens were taken with, and its case list is the one ⑤ runs (read from the record, not
//    written here); the list holds the window case `window-en`, whose prompt is longer than the
//    sliding ring (1032 rows) and than one prefill chunk (768 rows)
// ⑤ greedy parity: Pipeline.fromPretrained on model 12b / quant i4, the greedy chain of each case
//    of the record through `sequence()` equals the golden's expected list token for token. The
//    goldens are torch greedy over logits with the suppressed ids at -inf (upstream generate()), and
//    the pipeline masks the same ids, so a model that would emit `<image|>` (258882) agrees with
//    them. Each golden's torch top1 − top2 margin (over the masked logits) exceeds 2 × atol, so the
//    comparison does not hinge on numeric noise. A stop token of the distribution's EOS set ends a
//    turn; the chain goes on with an empty prompt (the stop token is the pending frontier), so the
//    whole list is compared, not only the part before the first stop.
//
// ## Assets
//
// The model is the distribution mirror's `model` component of model `12b` (`models/karume-gemma4-qat/`,
// quant i4 — about 6.9 GiB of weights), opened through the manifest (ADR 0139 decision 1). Without
// the mirror, or with a mirror that has no model `12b`, every case SKIPs explicitly. The goldens are
// the product series' outputs in the test-asset root (`goldens/gemma4-12b-product/` —
// `greedy.<case>.safetensors` with `prompt` / `expected` i32 and `margin` f32, and `reference.json`) under ADR 0139's
// fetched rule: absent from a complete fetch of the pinned dataset ⇒ FAIL; otherwise SKIP with the
// missing files and `node tools/test-assets/fetch.ts` in the title.
//
// ## GPU memory
//
// ⑤ holds one 12B pipeline at a time: about 6.9 GiB of weights + 645 MiB of sliding slots + 128 MiB
// of full slots at the default capacity 4096 + transients — about 9 GiB, which a 12 GB GPU holds.
// No other case of this file touches the GPU. The sha256 regression row of the i4 seat (`12b/i4`) is
// `e2e_gemma4_reference_test.ts`'s, which enumerates the mirror's seats (ADR 0106).

import {
  isNotFound,
  readFile,
  readTextFile,
  readTextFileSync,
  statSync,
} from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertNotEquals, assertRejects } from "@std/assert";
import { type Manifest, MANIFEST_FILENAME, parseManifest } from "@karume/hub";
import { parseSafetensors, prepareContainer, type SafetensorsFile } from "@karume/runtime";
import {
  type GenerationSequence,
  Pipeline,
  parsePipelineConfig,
  type RopeSpec,
} from "../../models/gemma.ts";
import { ModelInputError } from "../../models/src/errors.ts";
import { gemma4RopeInputNames } from "../../models/src/gemma/rope.ts";
import { declaredGemma4PleAssets, gemma4PleAssetSource } from "../../models/src/gemma/ple-index.ts";
import {
  goldenDir,
  TEST_ASSETS,
  TEST_ASSETS_ROOT,
  testAssetsFetched,
} from "@karume/test-support/test-assets";
import {
  mirrorDirectory,
  mirrorPresent,
  mirrorRoot,
  openSelectedContainer,
  selectMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";

/** The q4_0 QAT line's distribution mirror (E2B, E4B and 12B — the one that ships 12B). */
const MIRROR = mirrorRoot("karume-gemma4-qat");
/** The seat this file reads (the only quant of model 12b). */
const CHOICE = { model: "12b", quant: "i4" } as const;
/** The component key, which is also the container's graph name (container-v1 §2.1). */
const MODEL_GRAPH = "model";
/** The product series' goldens (the greedy expected lists and the provenance record). */
const GOLDEN_ROOT = goldenDir("gemma4-12b-product");
const GREEDY_PREFIX = "greedy.";
const SUFFIX = ".safetensors";
const REFERENCE_FILE = "reference.json";
/** The schema of the provenance record (spelling source: `gemma4/provenance.py`). */
const REFERENCE_SCHEMA = 1;
/**
 * The key of `reference` that records the suppressed ids the goldens were taken with (spelling
 * source: `gemma4/export_unified.py`).
 */
const REFERENCE_SUPPRESS_KEY = "suppressTokens";
/**
 * The case that crosses the sliding window: its prompt is longer than the sliding ring and than one
 * prefill chunk (the ring evicts, the prefill runs in more than one chunk). The record's case list
 * MUST hold it — the standard cases are all shorter than 12B's window (`context-en` is 598 tokens).
 */
const WINDOW_CASE = "window-en";
/** Steps of every greedy golden (`export_decode.GREEDY_STEPS["12b"]`). */
const GREEDY_STEPS = 16;
/** The tensors of one greedy golden (`export_decode` layout). */
const GOLDEN_KEYS = ["expected", "margin", "prompt"] as const;
/**
 * The tolerance the margin gate is stated against: the gemma4 logits tolerance (`atol` 1e-2, rtol 0
 * — `e2e_gemma4_test.ts` / `e2e_gemma4_greedy_test.ts`). A golden whose torch top1 − top2 margin is
 * not above 2 × atol could flip within that tolerance; the recipe's floor (`MARGIN_FLOOR` 2.5e-2)
 * is above it, so the gate fails only when the recipe and the assets disagree.
 */
const ATOL = 1e-2;
/**
 * google/gemma-4-12B-it generation_config.json `suppress_tokens` (`<image|>` 258882 / `<audio|>`
 * 258883), sorted as the distribution writes them.
 */
const SUPPRESS_TOKENS = [258882, 258883] as const;

/** The command that restores the goldens (ADR 0139 decision 2). */
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";

/** google/gemma-4-12B-it config.json at 43c9d014 (`text_config`). */
const LAYERS = 48;
const HIDDEN = 3840;
const VOCAB = 262144;
const SLIDING_WINDOW = 1024;
/** Rows a sliding slot holds beyond the window (`export_decode.SLIDING_SLACK_ROWS`). */
const SLIDING_SLACK_ROWS = 8;
const SLIDING_KV_HEADS = 8;
const FULL_KV_HEADS = 1;
const SLIDING_HEAD_DIM = 256;
const FULL_HEAD_DIM = 512;
/** Full-attention layers are every sixth (5, 11, …, 47 — the last layer is full). */
const isFullLayer = (layer: number): boolean => layer % 6 === 5;
const MAX_POSITION = 262144;
/** The distribution's default capacity and chunk rows (runtime knobs; the manifest declares them). */
const CAPACITY = 4096;
const CHUNK_LENGTH = 768;
/** RoPE (sliding θ 1e4 default; full θ 1e6 proportional with partial rotary 0.25). */
const ROPE: RopeSpec = {
  sliding_attention: { theta: 10000, headDim: SLIDING_HEAD_DIM, rotaryDim: SLIDING_HEAD_DIM },
  full_attention: { theta: 1000000, headDim: FULL_HEAD_DIM, rotaryDim: 128 },
};
/** generation_config.json (temperature / top-k / top-p — the same as E2B / E4B). */
const SAMPLER = { temperature: 1, topK: 64, topP: 0.95 } as const;

/** Graph input names (source of truth: the constants of `export_product.py`). */
const INPUT_IDS = "input_ids";
const LAST_ROW = "last_row";
/** The row-selection symbol bound by `last_row[R]`. */
const ROW_SYMBOL = "R";
/** The capacity symbol of the full slots. */
const CAPACITY_SYMBOL = "C";
/** The state slot of one layer (`l<layer>.<k|v>` — the exporter's `states_plan`). */
const slotName = (layer: number, part: "k" | "v"): string => `l${layer}.${part}`;

const exists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/** The mirror's manifest when it is readable by this version (undefined otherwise). */
const MANIFEST: Manifest | undefined = mirrorPresent(MIRROR)
  ? parseManifest(readTextFileSync(new URL(MANIFEST_FILENAME, MIRROR)))
  : undefined;
const MODEL_PRESENT = MANIFEST !== undefined && Object.hasOwn(MANIFEST.models, CHOICE.model);

if (!MODEL_PRESENT) {
  console.warn(
    `[karume] the distribution mirror ${MIRROR.pathname} is absent, not karume/5, or has no model ` +
      `'${CHOICE.model}', so the Gemma 4 12B cases SKIP. Assemble: cd tools/export-recipes && uv run ` +
      "python dist.py --pipeline gemma4-qat",
  );
}

/** Narrows the provenance JSON one level (unknown shapes fail with a clear message). */
const objectAt = (value: unknown, where: string): Map<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${REFERENCE_FILE}: ${where} is not an object`);
  }
  return new Map(Object.entries(value));
};

const goldenFile = (name: string): string => `${GREEDY_PREFIX}${name}${SUFFIX}`;
const goldenUrl = (name: string): URL => new URL(goldenFile(name), GOLDEN_ROOT);
const REFERENCE = new URL(REFERENCE_FILE, GOLDEN_ROOT);

/**
 * The greedy cases the record binds (`reference.goldens` — `greedy.<case>.safetensors` keys), in
 * sorted order; empty while the record is absent (the gates below then name it). The list is read
 * from the record, not written here, so a case the recipe adds is run without editing this file.
 */
const CASES: readonly string[] = (() => {
  if (!exists(REFERENCE)) return [];
  const record: unknown = JSON.parse(readTextFileSync(REFERENCE));
  const reference = objectAt(objectAt(record, "root").get("reference"), "reference");
  return [...objectAt(reference.get("goldens"), "reference.goldens").keys()]
    .map((file) => {
      if (!file.startsWith(GREEDY_PREFIX) || !file.endsWith(SUFFIX)) {
        throw new Error(
          `${REFERENCE_FILE}: reference.goldens key '${file}' is not ${goldenFile("<case>")}`,
        );
      }
      return file.slice(GREEDY_PREFIX.length, file.length - SUFFIX.length);
    })
    .sort();
})();
const GREEDY_GOLDENS = CASES.map(goldenUrl);

/** The goldens of `files` that are absent, as paths under the test-asset root. */
const missingGoldens = (files: readonly URL[]): string[] =>
  files
    .filter((file) => !exists(file))
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

const PROVENANCE_GATE = goldensGate([REFERENCE, ...GREEDY_GOLDENS]);
const GREEDY_GATE = goldensGate([REFERENCE, ...GREEDY_GOLDENS]);

/** The manifest entry of model 12b (only called by cases guarded by {@link MODEL_PRESENT}). */
const entryOf = (manifest: Manifest | undefined, model: string) => {
  assert(manifest !== undefined && Object.hasOwn(manifest.models, model), `no model '${model}'`);
  return manifest.models[model];
};

test.skipIf(!MODEL_PRESENT)(
  "Gemma 4 12B distribution: the manifest entry declares the 12B pipeline, one i4 seat and its own tokenizer (no GPU)",
  () => {
    const entry = entryOf(MANIFEST, CHOICE.model);
    assertEquals(entry.pipeline, { name: "gemma4", major: 1 }, "pipeline contract");
    assertEquals(entry.available.quants, [CHOICE.quant], "quants (only i4 this round)");
    assertEquals(entry.defaultQuant, CHOICE.quant, "defaultQuant");
    assertEquals(entry.quants[CHOICE.quant].session, {}, "the i4 seat carries no session knobs");
    // The product graph and its QAT drafter; PLE would be assets of `model`, and there are none (②).
    assertEquals(
      Object.keys(entry.weights).sort(),
      ["drafter", MODEL_GRAPH],
      "weights (the product graph and its drafter)",
    );
    assertEquals(Object.keys(entry.assets), ["tokenizer"], "assets (the tokenizer only)");

    const config = parsePipelineConfig(entry.pipelineConfig);
    assertEquals(config.chatDialect, "gemma4-unified", "chatDialect (the 12B chat template)");
    assertEquals(
      config.suppressTokens,
      [...SUPPRESS_TOKENS],
      "suppressTokens (generation_config.json suppress_tokens, sorted)",
    );
    assertEquals(config.maxPosition, MAX_POSITION, "maxPosition (max_position_embeddings)");
    assertEquals(config.capacity, CAPACITY, "default capacity");
    assertEquals(config.chunkLength, CHUNK_LENGTH, "default chunkLength");
    assertEquals(config.maxChunkLength, CHUNK_LENGTH, "maxChunkLength (the trace range of M)");
    assertEquals(config.rope, ROPE, "rope (from config.json)");
    assertEquals(config.sampler, SAMPLER, "sampler (from generation_config.json)");

    // 12B's glitch-token fix drops other BPE merges than E2B's and E4B's — sharing their asset
    // would give other tokens for some inputs.
    const others = ["e2b", "e4b"].filter((model) => Object.hasOwn(MANIFEST?.models ?? {}, model));
    assert(others.includes("e2b"), "the mirror has no e2b (the family's default model)");
    for (const other of others) {
      const theirs = entryOf(MANIFEST, other);
      assertNotEquals(
        entry.assets["tokenizer"].sha256,
        theirs.assets["tokenizer"].sha256,
        `the 12b tokenizer is not ${other}'s`,
      );
      // The E2B / E4B declarations stay as they were (absent = "gemma4" / nothing suppressed).
      for (const key of ["chatDialect", "suppressTokens"]) {
        assertEquals(
          Object.hasOwn(theirs.pipelineConfig, key),
          false,
          `${other}'s pipelineConfig has no ${key}`,
        );
      }
    }
  },
);

/** Opens the mirror's `model` container of model 12b with range reads (part 0 only until a Session). */
const openProductContainer = async () =>
  await openSelectedContainer(await selectMirrorComponent(MIRROR, MODEL_GRAPH, CHOICE));

test.skipIf(!MODEL_PRESENT)(
  "Gemma 4 12B product graph: no per-layer inputs, logits + hidden exits, 96 owned slots, no PLE assets (no GPU)",
  async () => {
    const opened = await openProductContainer();
    assertEquals(
      declaredGemma4PleAssets(gemma4PleAssetSource(opened)),
      [],
      "PLE assets in the container (12B has no per-layer inputs)",
    );
    const graph = prepareContainer(opened, MODEL_GRAPH).graph;

    assertEquals(
      graph.inputs.map((spec) => spec.name).sort(),
      [INPUT_IDS, LAST_ROW, ...gemma4RopeInputNames()].sort(),
      "graph inputs (the E2B / E4B product contract minus per_layer_inputs)",
    );
    for (const spec of graph.inputs.filter((input) => input.name.startsWith("rope_"))) {
      assertEquals(
        spec.shape,
        [1, "M", spec.name.includes("full") ? FULL_HEAD_DIM : SLIDING_HEAD_DIM],
        `'${spec.name}' shape`,
      );
    }
    const lastRow = graph.inputs.find((spec) => spec.name === LAST_ROW);
    assert(lastRow !== undefined, `'${LAST_ROW}' is missing`);
    assertEquals(lastRow.shape, [ROW_SYMBOL], `'${LAST_ROW}' shape (the only binding of R)`);

    // Two exits, order fixed: logits of the selected rows, then their final-norm hidden.
    assertEquals(graph.outputs.length, 2, "graph.outputs (logits + hidden)");
    assertEquals(graph.values[graph.outputs[0]].shape, [1, ROW_SYMBOL, VOCAB], "output 0 (logits)");
    assertEquals(
      graph.values[graph.outputs[1]].shape,
      [1, ROW_SYMBOL, HIDDEN],
      "output 1 (hidden)",
    );
    assertEquals(
      graph.nodes.filter((node) => node.op === "argmax").length,
      0,
      "argmax nodes (sampling is the host's — ADR 0083 decision 6)",
    );
    assertEquals([...graph.symbols].sort(), [CAPACITY_SYMBOL, "M", ROW_SYMBOL], "graph.symbols");

    // Every layer owns its KV (num_kv_shared_layers = 0). Slot form `[B, Hkv, C, D]` (ir-v2).
    const expectedSlots = Array.from({ length: LAYERS }, (_, layer) => layer)
      .flatMap((layer) => [slotName(layer, "k"), slotName(layer, "v")])
      .sort();
    assertEquals(
      Object.keys(graph.states).sort(),
      expectedSlots,
      "state slots (48 layers × k / v)",
    );
    for (let layer = 0; layer < LAYERS; layer += 1) {
      const expected = isFullLayer(layer)
        ? [1, FULL_KV_HEADS, CAPACITY_SYMBOL, FULL_HEAD_DIM]
        : [1, SLIDING_KV_HEADS, SLIDING_WINDOW + SLIDING_SLACK_ROWS, SLIDING_HEAD_DIM];
      for (const part of ["k", "v"] as const) {
        const name = slotName(layer, part);
        assertEquals(
          graph.states[name].shape,
          expected,
          `states['${name}'] (${isFullLayer(layer) ? "full" : "sliding"})`,
        );
      }
    }
  },
);

test.skipIf(!MODEL_PRESENT)(
  "Gemma 4 12B admission: the PLE placement options are refused before any weight part (no GPU)",
  async () => {
    for (const options of [
      { pleResidency: "host" },
      { pleResidency: "gpu" },
      { maxResidentPleBytes: 0 },
    ] as const) {
      const error = await assertRejects(
        () => Pipeline.fromPretrained(mirrorDirectory(MIRROR), { ...CHOICE, ...options }),
        ModelInputError,
        "cannot be used with this model",
      );
      const [key] = Object.keys(options);
      assert(error.message.includes(key), error.message);
    }
  },
);

const stringAt = (entries: Map<string, unknown>, key: string, where: string): string => {
  const value = entries.get(key);
  if (typeof value !== "string" || value === "") {
    throw new Error(`${REFERENCE_FILE}: ${where}.${key} is not a non-empty string`);
  }
  return value;
};

/** One `{bytes, sha256}` entry (sha256 as 64 lowercase hex digits). */
const digestAt = (
  entries: Map<string, unknown>,
  key: string,
  where: string,
): { bytes: number; sha256: string } => {
  const digest = objectAt(entries.get(key), `${where}.${key}`);
  const bytes = digest.get("bytes");
  if (typeof bytes !== "number" || !Number.isSafeInteger(bytes) || bytes <= 0) {
    throw new Error(`${REFERENCE_FILE}: ${where}.${key}.bytes is not a positive integer`);
  }
  const sha256 = stringAt(digest, "sha256", `${where}.${key}`);
  if (!/^[0-9a-f]{64}$/.test(sha256)) {
    throw new Error(`${REFERENCE_FILE}: ${where}.${key}.sha256 is not 64 hex digits`);
  }
  return { bytes, sha256 };
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/** The last path segment of a series directory (the name the record states). */
const seriesName = (root: URL): string =>
  root.pathname
    .split("/")
    .filter((part) => part !== "")
    .at(-1) ?? "";

/**
 * ④ The provenance record binds the goldens ⑤ reads: re-taking one golden without the others (or
 * without re-exporting) moves a digest, and this case fails instead of ⑤ comparing against a
 * golden of another run.
 */
test.skipIf(PROVENANCE_GATE.skip)(
  `Gemma 4 12B goldens: the provenance record binds the greedy goldens (no GPU)${PROVENANCE_GATE.title}`,
  async () => {
    assertGoldensPresent([REFERENCE, ...GREEDY_GOLDENS]);
    const record = objectAt(JSON.parse(await readTextFile(REFERENCE)), "root");
    assertEquals(record.get("schema"), REFERENCE_SCHEMA, `${REFERENCE_FILE} schema`);
    assertEquals(
      stringAt(record, "series", "root"),
      seriesName(GOLDEN_ROOT),
      "the record's series",
    );
    const checkpoint = objectAt(record.get("checkpoint"), "checkpoint");
    stringAt(checkpoint, "dir", "checkpoint");
    const fingerprint = objectAt(checkpoint.get("files"), "checkpoint.files");
    assert(fingerprint.size > 0, `${REFERENCE_FILE}: checkpoint.files is empty`);
    for (const file of fingerprint.keys()) digestAt(fingerprint, file, "checkpoint.files");

    const reference = objectAt(record.get("reference"), "reference");
    // The greedy goldens are this series' own outputs (the 12B product series takes them itself).
    assertEquals(
      stringAt(reference, "series", "reference"),
      seriesName(GOLDEN_ROOT),
      "the bound series",
    );
    // The goldens were taken over logits with these ids at -inf — the set the pipeline masks.
    assertEquals(
      reference.get(REFERENCE_SUPPRESS_KEY),
      [...SUPPRESS_TOKENS],
      `${REFERENCE_FILE}: reference.${REFERENCE_SUPPRESS_KEY} (the masking the goldens were taken with)`,
    );
    const goldens = objectAt(reference.get("goldens"), "reference.goldens");
    assert(
      CASES.includes(WINDOW_CASE),
      `the bound cases [${CASES.join(", ")}] lack '${WINDOW_CASE}'`,
    );
    assertEquals(
      [...goldens.keys()].sort(),
      CASES.map(goldenFile).sort(),
      "the bound goldens (exactly the cases ⑤ reads)",
    );
    for (const name of CASES) {
      const file = goldenFile(name);
      const digest = digestAt(goldens, file, "reference.goldens");
      const bytes = await readFile(new URL(file, GOLDEN_ROOT));
      assertEquals(bytes.byteLength, digest.bytes, `${file}: bytes`);
      assertEquals(
        await sha256Hex(bytes),
        digest.sha256,
        `${file}: sha256 (the dataset at the pin holds goldens of different runs — re-export the ` +
          `12B product series, upload its goldens to ${TEST_ASSETS.repo} and move the pin)`,
      );
    }
  },
);

const goldenTensor = (file: SafetensorsFile, name: string, dtype: "I32" | "F32") => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `golden has no '${name}'`);
  assertEquals(view.dtype, dtype, `golden '${name}' dtype`);
  return view;
};

/**
 * One greedy case (`export_decode` layout): the prompt ids, the expected continuation and the torch
 * top1 − top2 margin of each step. The tensor set and the step count are pinned, so a golden of
 * another layout or length fails here instead of comparing a shorter list.
 */
const readCase = async (
  name: string,
): Promise<{
  readonly prompt: number[];
  readonly expected: number[];
  readonly margin: number[];
}> => {
  assertGoldensPresent([goldenUrl(name)]);
  const bytes = await readFile(goldenUrl(name));
  const golden = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  assertEquals([...golden.tensors.keys()].sort(), [...GOLDEN_KEYS], `${goldenFile(name)}: tensors`);
  const prompt = goldenTensor(golden, "prompt", "I32");
  const expected = goldenTensor(golden, "expected", "I32");
  const margin = goldenTensor(golden, "margin", "F32");
  const read = {
    prompt: [...new Int32Array(golden.buffer, prompt.byteOffset, prompt.byteLength / 4)],
    expected: [...new Int32Array(golden.buffer, expected.byteOffset, expected.byteLength / 4)],
    margin: [...new Float32Array(golden.buffer, margin.byteOffset, margin.byteLength / 4)],
  };
  assertEquals(read.expected.length, GREEDY_STEPS, `${name}: golden steps`);
  assertEquals(read.margin.length, GREEDY_STEPS, `${name}: margin steps`);
  return read;
};

/**
 * The greedy chain of `steps` tokens from `prompt`, stop tokens included.
 *
 * A turn stops at the distribution's EOS set (the stop token is not a `token` event; `done` carries
 * it and the sequence keeps it as the pending frontier). The next `generate` with an empty prompt
 * feeds that token and continues, so the chain is the same as one uninterrupted greedy run — the
 * goldens are the torch greedy continuation, which does not stop.
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

describe.skipIf(!MODEL_PRESENT || GREEDY_GATE.skip || !GPU_AVAILABLE)(
  `Gemma 4 12B greedy parity: the pipeline's greedy chains equal the goldens (real GPU)${GREEDY_GATE.title}`,
  () => {
    let pipeline: Pipeline | undefined;
    beforeAll(async () => {
      pipeline = await Pipeline.fromPretrained(mirrorDirectory(MIRROR), CHOICE);
    });
    afterAll(async () => {
      await pipeline?.dispose();
    });

    it("the pipeline carries the 12B chat dialect and the 12B wiring", () => {
      assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
      assertEquals(pipeline.tokenizer.chatDialect, "gemma4-unified");
      assertEquals(pipeline.program.maxPosition, MAX_POSITION);
      assertEquals(pipeline.program.capacity, CAPACITY);
      assertEquals(pipeline.program.vocabSize, VOCAB);
      const estimate = pipeline.estimateSessionMemory();
      console.log(
        `[e2e] gemma4 12b i4: estimated peak ${(estimate.peakAccountedBytes / 2 ** 30).toFixed(2)} GiB` +
          ` (weights ${(estimate.resident.weights.totalBytes / 2 ** 30).toFixed(2)} GiB, state` +
          ` ${(estimate.resident.stateBytes / 2 ** 20).toFixed(0)} MiB) at capacity ${CAPACITY}`,
      );
    });

    it(`the record's cases include the window case '${WINDOW_CASE}', longer than the ring and a chunk`, async () => {
      assertGoldensPresent([REFERENCE, ...GREEDY_GOLDENS]);
      assert(CASES.includes(WINDOW_CASE), `the record's cases [${CASES.join(", ")}] lack it`);
      const { prompt } = await readCase(WINDOW_CASE);
      const ring = SLIDING_WINDOW + SLIDING_SLACK_ROWS;
      assert(
        prompt.length > ring,
        `${WINDOW_CASE}: T=${prompt.length} does not exceed the ring ${ring}`,
      );
      assert(prompt.length > CHUNK_LENGTH, `${WINDOW_CASE}: T=${prompt.length} fits one chunk`);
    });

    for (const name of CASES) {
      it(`${name}: the greedy chain equals the golden token for token`, async () => {
        assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
        const { prompt, expected, margin } = await readCase(name);
        // The golden's qualification (not always true): a torch top1 − top2 margin above 2 × atol
        // cannot flip within the tolerance, so this list does not split on numeric noise.
        const minMargin = Math.min(...margin);
        assert(
          minMargin > 2 * ATOL,
          `${name}: the golden's smallest margin ${minMargin} is comparable to atol — the token` +
            " match could flip on numeric differences (not a valid gate)",
        );
        const sequence = await pipeline.sequence();
        try {
          const started = performance.now();
          const ids = await greedyChain(sequence, prompt, expected.length);
          assertEquals(ids, expected, `${name}: greedy tokens (against the torch golden)`);
          console.log(
            `[e2e] gemma4 12b ${name}: T=${prompt.length} + K=${expected.length} /` +
              ` min margin ${minMargin.toExponential(3)} / ${(performance.now() - started).toFixed(0)} ms`,
          );
        } finally {
          await sequence.dispose();
        }
      });
    }
  },
);
