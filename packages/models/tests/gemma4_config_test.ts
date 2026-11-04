// Checks of the gemma4 `pipelineConfig` schema (`src/gemma/config.ts`) — needs neither a GPU nor assets.
//
// The hub passes `pipelineConfig` through, so this parser on the models side is the only source of truth for its
// shape. Four things are pinned here:
//
//  ① **The accepted set of fields** (unknown keys, missing fields, types, ranges) — if a misspelling silently fell
//     back to a default, the distributor's declaration and the execution would disagree unnoticed
//  ② **The numeric relations** (`chunkLength ≤ maxChunkLength` and `chunkLength ≤ capacity ≤ maxPosition`) — a
//     conversation that fills the capacity advances to position `capacity - 1`, so a declaration beyond the
//     position limit fails **only in long conversations**. `maxChunkLength` is the upper end of the trace range of
//     symbol `M` and cannot be read from the assets (the IR's `symbols` is only a list of names), so the
//     declaration is its only source
//  ③ **Agreement with the writing side** — when a local distribution mirror exists, the declaration in its
//     `karume.json` passes this parser, and the recommended sampler is exactly the upstream
//     `generation_config.json` values (a measurement of "the defaults are declared by the distribution" in
//     ADR 0083 decision 7)
//  ④ **The comparison with the graph declaration** (`rope.<layer type>.headDim` = the last dimension of the
//     `rope_*` inputs) — even a correct declaration gives tables of the wrong width if it disagrees with the
//     graph. This moves the failure point back from the first `run` to admission
//
// The assets of ③ (`models/karume-gemma4/`) are outside the repository, so in an environment without them **only
// that one test** is skipped explicitly (① and ② always run).

import { isNotFound, readTextFileSync } from "@karume/test-support/fs";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { ExecutionError } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import { type PipelineConfig, parsePipelineConfig } from "../src/gemma/config.ts";
import {
  admitGemma4,
  assertChunkLength,
  assertGemma4ChunkBuckets,
  assertRopeInputShapes,
  gemmaEntryName,
} from "../src/gemma/admission.ts";
import { buildGemma4Program, CHUNK_BUCKETS, Pipeline } from "../src/gemma/pipeline.ts";
import type { GenerationGraph } from "../src/generation/program.ts";
import type { GraphOwner, ModelComponent } from "../src/hub/components.ts";
import { stubModel } from "./helpers/stub-model.ts";
import { beforeAll, describe, it, test } from "vite-plus/test";

/**
 * The minimal accepted shape (4 numbers + `rope` — only `sampler` is optional).
 *
 * `rope` is Gemma 4 E2B's actual values (sliding is the default rope; full is proportional with partial rotary
 * 0.25). The tables are not in the distribution; the host builds cos / sin from this declaration.
 */
const ROPE = {
  sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
  full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
} as const;
const MINIMAL = {
  chunkLength: 32,
  maxChunkLength: 128,
  maxPosition: 1024,
  capacity: 640,
  rope: ROPE,
} as const;
/** The defaults declared by the distribution mirror (capacity / chunkLength are both **defaults** of runtime knobs). */
const SHIPPED_CHUNK_LENGTH = 768;
/** The trace upper bound of symbol `M` (the writing side's `gemma4.export.SYM_MAX` — equal to the default, but a separate fact). */
const SHIPPED_MAX_CHUNK_LENGTH = 768;
const SHIPPED_CAPACITY = 4096;
/** Upstream `max_position_embeddings` (= the position limit the model declares). */
const SHIPPED_MAX_POSITION = 131072;

/** The recommendation in upstream `gemma-4-E2B-it`'s `generation_config.json` (ADR 0083 decision 7). */
const RECOMMENDED = { temperature: 1, topK: 64, topP: 0.95 } as const;

const MIRROR = new URL("../../../models/karume-gemma4/karume.json", import.meta.url);
/** The major of the distribution manifest this version reads (older versions are not read — ADR 0109 decision 1). */
const MANIFEST_FORMAT = "karume/5";

/**
 * The distribution mirror's `karume.json` (skipped on a machine without it / with an old major).
 *
 * MUST: decide by "is the format `karume/5`". On a machine where a pre-migration mirror remains, the parse itself
 * fails, so conditioning on "the file exists" would only turn the check red and tell nothing (the same discipline
 * as `distributionFormat` in `tools/_shared` — the coexistence period of ADR 0109 decision 9).
 */
const readMirror = (): string | undefined => {
  let text: string;
  try {
    text = readTextFileSync(MIRROR);
  } catch (cause) {
    if (isNotFound(cause)) return undefined;
    throw cause;
  }
  const format = (JSON.parse(text) as { readonly format?: unknown }).format;
  if (format === MANIFEST_FORMAT) return text;
  console.warn(
    `[karume] models/karume-gemma4/ has format ${String(format)} (not ${MANIFEST_FORMAT}),` +
      " so the check of the distribution's declaration is skipped.",
  );
  return undefined;
};

test("gemma4 pipelineConfig: reads 3 numbers + rope and the optional sampler", () => {
  assertEquals(parsePipelineConfig(MINIMAL), MINIMAL);
  assertEquals(parsePipelineConfig({ ...MINIMAL, sampler: { ...RECOMMENDED } }), {
    ...MINIMAL,
    sampler: RECOMMENDED,
  });
});

describe("gemma4 pipelineConfig: unknown keys and missing fields fail loudly", () => {
  it("a declaration that is not an object (the boundary parser of `unknown`)", () => {
    for (const raw of [undefined, null, 42, "chunkLength=32", [MINIMAL]]) {
      assertThrows(() => parsePipelineConfig(raw), Error, "not an object");
    }
  });
  it("an unknown key at the root (a misspelling does not fall back to a default)", () => {
    assertThrows(
      () => parsePipelineConfig({ ...MINIMAL, capasity: 640 }),
      Error,
      "unknown key 'capasity'",
    );
  });
  it("an unknown key in sampler (only 3 fields can be declared)", () => {
    assertThrows(
      () => parsePipelineConfig({ ...MINIMAL, sampler: { ...RECOMMENDED, seed: 7 } }),
      Error,
      "unknown key 'seed'",
    );
  });
  it("a missing field", () => {
    const { capacity: _dropped, ...rest } = MINIMAL;
    assertThrows(() => parsePipelineConfig(rest), Error, "pipelineConfig.capacity: missing");
  });
  it("maxChunkLength is required (without it nobody can reject what is outside the trace range)", () => {
    // Passing a missing value as "no limit" lets a chunkLength of 1024 given to assets with a limit of 768 run
    // without an exception (measured before the check existed — 2026-09-03). The declaration is the only source, so a missing value fails loudly.
    const { maxChunkLength: _dropped, ...rest } = MINIMAL;
    assertThrows(() => parsePipelineConfig(rest), Error, "pipelineConfig.maxChunkLength: missing");
  });
  it("rope is required (running without it would silently run attention without rotation)", () => {
    const { rope: _dropped, ...rest } = MINIMAL;
    assertThrows(() => parsePipelineConfig(rest), Error, "pipelineConfig.rope: missing");
  });
  it("passes when sampler is absent as a whole (= the lower layer's default = greedy)", () => {
    assertEquals(parsePipelineConfig(MINIMAL).sampler, undefined);
  });
});

describe("gemma4 pipelineConfig: the accepted set of rope", () => {
  const withRope = (rope: unknown): unknown => ({ ...MINIMAL, rope });

  it("exactly 2 layer types (an unknown layer type, a missing one)", () => {
    assertThrows(
      () => parsePipelineConfig(withRope({ ...ROPE, local_attention: ROPE.full_attention })),
      Error,
      "unknown key 'local_attention'",
    );
    const { full_attention: _dropped, ...sliding } = ROPE;
    assertThrows(
      () => parsePipelineConfig(withRope(sliding)),
      Error,
      "pipelineConfig.rope.full_attention: missing",
    );
  });

  it("exactly 3 fields per layer type (a misspelling does not fall back to a default)", () => {
    assertThrows(
      () =>
        parsePipelineConfig(
          withRope({ ...ROPE, full_attention: { ...ROPE.full_attention, ropeType: "linear" } }),
        ),
      Error,
      "unknown key 'ropeType'",
    );
    assertThrows(
      () =>
        parsePipelineConfig(
          withRope({ ...ROPE, sliding_attention: { theta: 10000, headDim: 256 } }),
        ),
      Error,
      "pipelineConfig.rope.sliding_attention.rotaryDim: missing",
    );
    assertThrows(
      () =>
        parsePipelineConfig(
          withRope({ ...ROPE, sliding_attention: { ...ROPE.sliding_attention, theta: "10000" } }),
        ),
      Error,
      "pipelineConfig.rope.sliding_attention.theta: not a number",
    );
  });

  it("the range check is owned by rope.ts (the same formula is not implemented twice)", () => {
    // An odd rotaryDim breaks the "first half = second half" layout = silently builds tables different from upstream.
    assertThrows(
      () =>
        parsePipelineConfig(
          withRope({ ...ROPE, full_attention: { ...ROPE.full_attention, rotaryDim: 129 } }),
        ),
      Error,
      "rotaryDim must be an even number in 2..headDim",
    );
    assertThrows(
      () =>
        parsePipelineConfig(
          withRope({ ...ROPE, full_attention: { ...ROPE.full_attention, rotaryDim: 1024 } }),
        ),
      Error,
      "rotaryDim must be an even number in 2..headDim",
    );
    assertThrows(
      () =>
        parsePipelineConfig(
          withRope({ ...ROPE, sliding_attention: { ...ROPE.sliding_attention, theta: 0 } }),
        ),
      Error,
      "theta must be a positive finite number",
    );
    assertThrows(
      () =>
        parsePipelineConfig(
          withRope({ ...ROPE, sliding_attention: { ...ROPE.sliding_attention, headDim: 255 } }),
        ),
      Error,
      "headDim must be an even number ≥ 2",
    );
  });
});

describe("gemma4 pipelineConfig: ranges", () => {
  it("numbers are integers of 1 or more", () => {
    assertThrows(() => parsePipelineConfig({ ...MINIMAL, chunkLength: 0 }), Error);
    assertThrows(() => parsePipelineConfig({ ...MINIMAL, maxPosition: 1.5 }), Error);
  });
  it("chunk rows are 2 or more (1 row is reserved for the decode shape — a plain Error as an asset inconsistency)", () => {
    for (const key of ["chunkLength", "maxChunkLength"]) {
      const error = assertThrows(
        () => parsePipelineConfig({ ...MINIMAL, [key]: 1 }),
        Error,
        `pipelineConfig.${key}: not an integer ≥ 2`,
      );
      assert(!(error instanceof ModelInputError), error.message);
    }
  });
  it("the sampler ranges are the same as assertSpec in generation/sampler.ts", () => {
    const sampler = { ...RECOMMENDED };
    assertThrows(
      () => parsePipelineConfig({ ...MINIMAL, sampler: { ...sampler, temperature: -1 } }),
      Error,
      "not a finite number ≥ 0",
    );
    assertThrows(
      () => parsePipelineConfig({ ...MINIMAL, sampler: { ...sampler, topK: 0 } }),
      Error,
      "not a positive integer",
    );
    assertThrows(
      () => parsePipelineConfig({ ...MINIMAL, sampler: { ...sampler, topP: 0 } }),
      Error,
      "not in the range 0 < topP ≤ 1",
    );
  });
});

describe("gemma4 pipelineConfig: chunkLength ≤ maxChunkLength / capacity ≤ maxPosition", () => {
  it("the default chunkLength is outside the trace range of symbol M", () => {
    assertThrows(
      () => parsePipelineConfig({ ...MINIMAL, chunkLength: 129 }),
      Error,
      "exceeds maxChunkLength 128",
    );
  });
  it("exactly the upper bound passes (the counterpart showing the check is not too wide)", () => {
    const config = parsePipelineConfig({ ...MINIMAL, chunkLength: 128 });
    assertEquals(config.chunkLength, 128);
  });
  it("a capacity that cannot hold even one chunk", () => {
    assertThrows(
      () => parsePipelineConfig({ ...MINIMAL, chunkLength: 64, capacity: 32 }),
      Error,
      "exceeds capacity 32",
    );
  });
  it("a conversation that fills the capacity reaches beyond the model's position limit", () => {
    assertThrows(
      () => parsePipelineConfig({ ...MINIMAL, capacity: 2048, maxPosition: 1024 }),
      Error,
      "exceeds maxPosition 1024",
    );
  });
});

// ---- Wiring with the entry points ----------------------------------------------------------
//
// `fromPretrained` runs this parser while reading the manifest (`gemma4ManifestConfig`). `fromAssets` takes it
// through TS types, but types check neither unknown keys nor ranges — without the check, a declaration of
// `temperature: -1` would first fail **at the first `chat` after reading all 3.7GiB**. What is checked here is
// "the same parser runs, and before a single byte stream is opened", so `model` is given garbage (if it were
// opened, it would fail with the container side's message = this check is missing).

describe("gemma4 fromAssets: config goes through the same check as fromPretrained, before the bytes are opened", () => {
  const junk = new Uint8Array<ArrayBuffer>(new ArrayBuffer(8));
  const fromAssets = (config: unknown): Promise<Pipeline> =>
    Pipeline.fromAssets({
      // The shape of a call from untyped JS (in TS the type check rejects unknown keys first).
      config: config as PipelineConfig,
      model: [junk],
      tokenizer: junk,
    });

  it("an unknown key", async () => {
    await assertRejects(
      () => fromAssets({ ...MINIMAL, capasity: 640 }),
      Error,
      "unknown key 'capasity'",
    );
  });
  it("the relation of the 3 numbers", async () => {
    await assertRejects(
      () => fromAssets({ ...MINIMAL, capacity: 2048 }),
      Error,
      "exceeds maxPosition 1024",
    );
  });
  it("the sampler ranges (the very asymmetry V2-b named)", async () => {
    await assertRejects(
      () => fromAssets({ ...MINIMAL, sampler: { ...RECOMMENDED, temperature: -1 } }),
      Error,
      "not a finite number ≥ 0",
    );
  });
  it("a correct declaration passes the check and fails with the container side's message (negative control)", async () => {
    // The counterpart showing it is not "green because the config check rejects everything".
    const error = await assertRejects(() => fromAssets({ ...MINIMAL }), Error);
    assertEquals(error.message.includes("pipelineConfig"), false, error.message);
  });
});

test("gemma4 fromAssets: an explicit quant execution knob also fails with the same type before the assets are opened", async () => {
  // This surface has no manifest, so it cannot go through `resolveGemmaSessionOptions`. Without the check, the same
  // wrong setting would go down to Session building and turn into the runtime's `ExecutionError`, splitting the
  // classification from `fromPretrained` (ADR 0107 decision 2). `model` is given garbage, so a failure that is not
  // this check would carry the container side's message.
  const junk = new Uint8Array<ArrayBuffer>(new ArrayBuffer(8));
  const error = await assertRejects(
    () =>
      Pipeline.fromAssets(
        {
          config: MINIMAL as unknown as PipelineConfig,
          model: [junk],
          tokenizer: junk,
        },
        { fuseLinearStaticQuantize: true, linearGemvReduce: "sequential" },
      ),
    ModelInputError,
    "fuseLinearStaticQuantize supports only linearGemvReduce: parallel / linearCompute: f32",
  );
  assertEquals(error.message.includes("fromAssets"), true, error.message);
});

// ---- Comparison with the graph declaration (RoPE width) ----------------------------------------
//
// `pipelineConfig.rope.<layer type>.headDim` is **the width of the tables the host builds**, so if it disagrees
// with the graph input's declaration (`[1, M, headDim]`) the host goes all the way through and the failure comes at
// the first `run` (**after** the 3.7GiB load, with the message "element count does not match shape" = it cannot
// tell which declaration is wrong). The comparison is done by family admission (`admitGemma4` = before a single
// byte of the weight parts is fetched). The writing side's mirror is `assert_rope_inputs` in
// `tools/export-recipes/gemma4/export_decode.py`.

/** The same declaration as the real distribution (sliding 256 / full 512). `patch` breaks exactly one. */
const ropeOwner = (
  patch: Readonly<Record<string, readonly (number | string)[]>> = {},
): GraphOwner =>
  stubModel({
    symbols: ["C", "M"],
    inputs: [
      { name: "input_ids", shape: [1, "M"] },
      { name: "last_row", shape: [1] },
      { name: "per_layer_inputs", shape: [1, "M", 35, 256] },
      { name: "rope_sliding_attention_cos", shape: patch.slidingCos ?? [1, "M", 256] },
      { name: "rope_sliding_attention_sin", shape: patch.slidingSin ?? [1, "M", 256] },
      { name: "rope_full_attention_cos", shape: patch.fullCos ?? [1, "M", 512] },
      { name: "rope_full_attention_sin", shape: patch.fullSin ?? [1, "M", 512] },
    ].filter((input) => input.shape.length > 0),
    outputs: ["logits"],
    values: { logits: [1, 1, 262144] },
  });

const ropeGraph = (
  patch: Readonly<Record<string, readonly (number | string)[]>> = {},
): GenerationGraph => ropeOwner(patch).graph;

describe("gemma4 rope comparison: a graph as declared passes, and a width mismatch fails by name", () => {
  let config: PipelineConfig;
  beforeAll(() => {
    config = parsePipelineConfig(MINIMAL);
  });

  it("as declared (negative control — the check is not green by rejecting everything)", () => {
    assertRopeInputShapes(ropeGraph(), config);
  });

  it("swapped layer types (sliding's width in full's place)", () => {
    // The very mistake the exporter's `rope.py` acknowledges in its `head_dim` / `global_head_dim` branch.
    const error = assertThrows(
      () => assertRopeInputShapes(ropeGraph({ fullCos: [1, "M", 256] }), config),
      Error,
      "rope_full_attention_cos",
    );
    assert(error.message.includes("headDim 512"), error.message);
  });

  it("a declaration where the distribution misstates headDim (the other side of the same mismatch)", () => {
    const lying = parsePipelineConfig({
      ...MINIMAL,
      rope: { ...ROPE, sliding_attention: { theta: 10000, headDim: 128, rotaryDim: 128 } },
    });
    assertThrows(
      () => assertRopeInputShapes(ropeGraph(), lying),
      Error,
      "rope_sliding_attention_cos",
    );
  });

  it("a declaration with a different number of dimensions (not `[1, M, headDim]`)", () => {
    assertThrows(
      () => assertRopeInputShapes(ropeGraph({ fullSin: [1, 512] }), config),
      Error,
      "rope_full_attention_sin",
    );
  });

  it("assets where RoPE is not host-supplied (the input itself is missing)", () => {
    assertThrows(
      () => assertRopeInputShapes(ropeGraph({ slidingSin: [] }), config),
      Error,
      "graph input 'rope_sliding_attention_sin' is missing",
    );
  });
});

describe("gemma4 family admission: the messages of the checks below it name the calling family's entry point", () => {
  let config: PipelineConfig;
  let qat: string;
  beforeAll(() => {
    config = parsePipelineConfig(MINIMAL);
    qat = gemmaEntryName("gemma4-qat");
  });

  it("a RoPE width mismatch (from the QAT entry point)", () => {
    const error = assertThrows(
      () => admitGemma4(ropeOwner({ fullCos: [1, "M", 256] }), config, undefined, qat),
      Error,
      "rope_full_attention_cos",
    );
    assert(error.message.startsWith("gemma4QatMobile.Pipeline: "), error.message);
  });

  it("a mismatch in the number of outputs (from the QAT entry point)", () => {
    // `ropeGraph` has one output, logits — it fails on the product graph's contract (2 outputs, logits + hidden).
    const error = assertThrows(
      () => admitGemma4(ropeOwner(), config, undefined, qat),
      Error,
      "the graph has 1 outputs",
    );
    assert(error.message.startsWith("gemma4QatMobile.Pipeline: "), error.message);
  });
});

test("gemma4 pipelineConfig: the distribution mirror's declaration passes this parser", () => {
  const text = readMirror();
  if (text === undefined) {
    console.warn(
      "[karume] models/karume-gemma4/ is missing, so the check of the distribution's declaration is skipped. " +
        "Generate it: cd tools/export-recipes && uv run python dist.py --pipeline gemma4",
    );
    return;
  }
  const manifest = parseManifest(text);
  const entry = manifest.models[manifest.defaultModel];
  const config = parsePipelineConfig(entry.pipelineConfig);
  // The recommended sampler is **the upstream declaration itself** (a hand copy would let the values move).
  assertEquals(config.sampler, RECOMMENDED, "the sampler defaults declared by the distribution");
  assertEquals(config.chunkLength, SHIPPED_CHUNK_LENGTH);
  // The trace upper bound of symbol `M` does not remain in the assets (the IR's symbols are names only), so the
  // distribution's declaration is the only source. Equality with the writing side (`gemma4.export.SYM_MAX`) is checked by the recipe's pytest.
  assertEquals(config.maxChunkLength, SHIPPED_MAX_CHUNK_LENGTH);
  // capacity / chunkLength are runtime knobs; what the distribution declares is their **default**.
  assertEquals(config.capacity, SHIPPED_CAPACITY);
  // The position limit is upstream `max_position_embeddings` (not the tables' row count — the tables left the distribution).
  assertEquals(config.maxPosition, SHIPPED_MAX_POSITION);
  // The RoPE parameters are the upstream config's actual values (a bad copy would make the rotation something else).
  assertEquals(config.rope, ROPE, "the rope declared by the distribution");
});

// ---- Checks of the runtime knob chunkLength --------------------------------------------
//
// While `assertChunkLength`'s 3 branches were non-exported, the only paths reaching them were `fromAssets` /
// `fromPretrained` / `estimateSessionMemory`, and every existing e2e passed **valid values** only = there was not
// even a negative control showing the checks were not green by rejecting everything. The check guards exactly the
// regression the doc describes (before the check existed, passing a chunkLength of 1024 to assets with a limit of
// 768 ran without an exception — measured 2026-09-03), so it is called directly here.

describe("assertChunkLength: the accepted set of the runtime knob (all 3 relations to the declaration)", () => {
  let config: PipelineConfig;
  beforeAll(() => {
    config = parsePipelineConfig(MINIMAL);
  });

  it("the value reserved for the decode shape (1) is not passed as a prefill shape", () => {
    assertThrows(
      () => assertChunkLength(1, config),
      ModelInputError,
      "chunkLength 1 is not an integer ≥ 2",
    );
  });

  it("rejects non-integers", () => {
    assertThrows(
      () => assertChunkLength(3.5, config),
      ModelInputError,
      "chunkLength 3.5 is not an integer ≥ 2",
    );
  });

  it("exactly the declared maxChunkLength passes (the counterpart showing the check is not too wide)", () => {
    assertEquals(assertChunkLength(config.maxChunkLength, config), config.maxChunkLength);
  });

  it("beyond the declared maxChunkLength (outside the trace range symbol M was exported with)", () => {
    assertThrows(
      () => assertChunkLength(config.maxChunkLength + 1, config),
      ModelInputError,
      `chunkLength 129 exceeds the distribution's declared maxChunkLength ${config.maxChunkLength}`,
    );
  });

  it("beyond maxPosition (confirms the third check is reachable)", () => {
    // A branch reachable only with a distribution that declares maxChunkLength larger than maxPosition.
    const wide = parsePipelineConfig({
      ...MINIMAL,
      maxChunkLength: 4096,
      maxPosition: 1024,
      capacity: 1024,
    });
    assertThrows(
      () => assertChunkLength(1025, wide),
      ModelInputError,
      "chunkLength 1025 exceeds maxPosition 1024",
    );
  });
});

// ---- Checks of the runtime knob chunkBuckets -------------------------------------------
//
// `assertGemma4ChunkBuckets` is a layer that "passes through to the runtime's accepted set and only adds the entry
// point's name", and holds not a single line of the rules itself (the source of truth is the runtime's
// `assertChunkBuckets`). As checks, two things are pinned here — ① the pass-through actually takes effect (the
// rejection names `Pipeline:`, and the original diagnostic stays in `cause`) ② the default ladder satisfies its own
// rules. If the default broke the rules, it would be the shape "fromAssets fails without any explicit setting", and
// the failure point would be far from the entry point.

describe("assertGemma4ChunkBuckets: the accepted set is delegated to the runtime and only the entry point's name is added", () => {
  it("a setting that fits the distribution's default chunkLength is returned as is", () => {
    const buckets: readonly number[] = [32, 64];
    assertEquals(assertGemma4ChunkBuckets(buckets, SHIPPED_CHUNK_LENGTH), buckets);
  });

  it("exactly chunkLength overlaps the prefill shape and fails (the original diagnostic in cause)", () => {
    const error = assertThrows(
      () => assertGemma4ChunkBuckets([SHIPPED_CHUNK_LENGTH], SHIPPED_CHUNK_LENGTH),
      ModelInputError,
    );
    assert(
      error.message.startsWith("gemma.Pipeline: "),
      `does not start with the entry point's name: ${error.message}`,
    );
    assert(
      error.cause instanceof ExecutionError,
      "keeps the runtime's diagnostic in cause (rewrapping does not erase its origin)",
    );
  });

  it("the default ladder gemma.CHUNK_BUCKETS itself is inside the accepted set", () => {
    assert(Object.isFrozen(CHUNK_BUCKETS), "the module-scope shared array is frozen");
    for (const [index, rows] of CHUNK_BUCKETS.entries()) {
      if (index > 0) {
        assert(
          rows > CHUNK_BUCKETS[index - 1],
          `not strictly ascending: ${CHUNK_BUCKETS.join(",")}`,
        );
      }
      // Not a single bucket is cut off by the distribution's default chunkLength = the defaults do not disagree.
      assert(
        rows < SHIPPED_CHUNK_LENGTH,
        `default ${rows} is at least the distribution's default chunkLength`,
      );
    }
    // The buckets for verify (the ladder that fits the draft limit of 8 rows + 1 confirmed row). If these 2 buckets
    // were dropped, the speculative verify run would be padded to 32 rows and pay back the saved work in padding —
    // the values stay correct, so it is a regression that only shows in performance.
    for (const rows of [4, 8]) {
      assert(
        CHUNK_BUCKETS.includes(rows),
        `verify bucket ${rows} is missing from the default ladder: ${CHUNK_BUCKETS.join(",")}`,
      );
    }
    assertEquals(assertGemma4ChunkBuckets(CHUNK_BUCKETS, SHIPPED_CHUNK_LENGTH), CHUNK_BUCKETS);
  });
});

// ---- chatDialect (the chat template the model follows) --------------------------------------

describe("gemma4 pipelineConfig: chatDialect", () => {
  it("absent: the parsed value has no chatDialect field (the E2B / E4B manifests as they are)", () => {
    const config = parsePipelineConfig(MINIMAL);
    assertEquals(Object.hasOwn(config, "chatDialect"), false);
  });

  it("each dialect of the closed set is read as declared", () => {
    for (const chatDialect of ["gemma4", "gemma4-unified"] as const) {
      assertEquals(parsePipelineConfig({ ...MINIMAL, chatDialect }), { ...MINIMAL, chatDialect });
    }
  });

  it("a dialect next to a sampler keeps both", () => {
    assertEquals(
      parsePipelineConfig({
        ...MINIMAL,
        sampler: { ...RECOMMENDED },
        chatDialect: "gemma4-unified",
      }),
      { ...MINIMAL, sampler: RECOMMENDED, chatDialect: "gemma4-unified" },
    );
  });

  it("an unknown dialect fails loudly naming the accepted set (no fallback to the default)", () => {
    for (const chatDialect of ["gemma4-12b", "Gemma4-unified", "", 1, null, ["gemma4"]]) {
      const error = assertThrows(
        () => parsePipelineConfig({ ...MINIMAL, chatDialect }),
        Error,
        "pipelineConfig.chatDialect",
      );
      assert(error.message.includes("gemma4 / gemma4-unified"), error.message);
    }
  });

  it("fromAssets runs the same check before opening a byte", async () => {
    const junk = new Uint8Array<ArrayBuffer>(new ArrayBuffer(8));
    await assertRejects(
      () =>
        Pipeline.fromAssets({
          config: { ...MINIMAL, chatDialect: "gemma4-12b" } as unknown as PipelineConfig,
          model: [junk],
          tokenizer: junk,
        }),
      Error,
      "pipelineConfig.chatDialect",
    );
  });
});

// ---- suppressTokens (ids the model never generates) ----------------------------------------
//
// gemma-4-12B-it's generation_config.json declares suppress_tokens [258883, 258882] (`<audio|>` /
// `<image|>`); the distribution writes them sorted. The selection side is in the generation tests
// (`generation_sampler_test.ts` / `generation_sequence_test.ts`); here: the declaration's shape and
// the vocabulary bound at admission.

/** The 12B declaration as the distribution writes it. */
const SUPPRESS_12B = [258882, 258883] as const;

describe("gemma4 pipelineConfig: suppressTokens", () => {
  it("absent: the parsed value has no suppressTokens field (the E2B / E4B manifests as they are)", () => {
    assertEquals(Object.hasOwn(parsePipelineConfig(MINIMAL), "suppressTokens"), false);
  });

  it("a sorted list of distinct ids is read as declared, next to the other optional keys", () => {
    assertEquals(parsePipelineConfig({ ...MINIMAL, suppressTokens: [...SUPPRESS_12B] }), {
      ...MINIMAL,
      suppressTokens: [...SUPPRESS_12B],
    });
    assertEquals(
      parsePipelineConfig({
        ...MINIMAL,
        sampler: { ...RECOMMENDED },
        chatDialect: "gemma4-unified",
        suppressTokens: [0],
      }),
      { ...MINIMAL, sampler: RECOMMENDED, chatDialect: "gemma4-unified", suppressTokens: [0] },
    );
  });

  it("an invalid declaration fails loudly naming the field", () => {
    const cases: readonly (readonly [unknown, string])[] = [
      // Empty would be a declaration that does nothing; the writer omits the key instead.
      [[], "is not a non-empty array of token ids"],
      ["258882", "is not a non-empty array of token ids"],
      [{ 0: 258882 }, "is not a non-empty array of token ids"],
      [null, "is not a non-empty array of token ids"],
      [[258882, "258883"], '[1]: "258883" is not a non-negative integer'],
      [[1.5], "[0]: 1.5 is not a non-negative integer"],
      [[-1], "[0]: -1 is not a non-negative integer"],
      [[null], "[0]: null is not a non-negative integer"],
      // The upstream order is not normalized here — the writer sorts; anything else was not written by it.
      [[258883, 258882], "[1]: 258882 does not follow 258883 in strictly ascending order"],
      [[7, 7], "[1]: 7 does not follow 7 in strictly ascending order"],
    ];
    for (const [suppressTokens, message] of cases) {
      const error = assertThrows(
        () => parsePipelineConfig({ ...MINIMAL, suppressTokens }),
        Error,
        message,
      );
      assert(error.message.startsWith("pipelineConfig.suppressTokens"), error.message);
    }
  });
});

describe("gemma4 admission: the suppressed ids are checked against the graph's vocabulary", () => {
  /** A product graph declaration with a vocabulary of `vocab` (the rest of the shipped shape). */
  const productOwner = (vocab: number): GraphOwner =>
    stubModel({
      symbols: ["C", "M", "R"],
      inputs: [
        { name: "input_ids", shape: [1, "M"], dtype: "i32" },
        { name: "rope_sliding_attention_cos", shape: [1, "M", 256] },
        { name: "rope_sliding_attention_sin", shape: [1, "M", 256] },
        { name: "rope_full_attention_cos", shape: [1, "M", 512] },
        { name: "rope_full_attention_sin", shape: [1, "M", 512] },
        { name: "last_row", shape: ["R"], dtype: "i32" },
      ],
      outputs: ["logits", "hidden"],
      values: { logits: [1, "R", vocab], hidden: [1, "R", 3840] },
    });
  const entry = gemmaEntryName("gemma4");

  it("ids inside the vocabulary pass (and no declaration passes)", () => {
    const config = parsePipelineConfig({ ...MINIMAL, suppressTokens: [...SUPPRESS_12B] });
    assertEquals(admitGemma4(productOwner(262144), config, undefined, entry).vocabSize, 262144);
    admitGemma4(productOwner(262144), parsePipelineConfig(MINIMAL), undefined, entry);
  });

  it("an id at or past the vocabulary fails before any weight part, naming the entry", () => {
    const config = parsePipelineConfig({ ...MINIMAL, suppressTokens: [5, 16] });
    const error = assertThrows(
      () => admitGemma4(productOwner(16), config, undefined, entry),
      Error,
      "pipelineConfig.suppressTokens[1] 16 is outside the vocabulary 0..15",
    );
    assert(error.message.startsWith(`${entry}: `), error.message);
    // The distribution's declaration, not the caller's request (ADR 0107 decision 2).
    assert(!(error instanceof ModelInputError));
  });
});

// ---- The declarations reach the build (GPU-free) --------------------------------------------
//
// `buildGemma4Program` is the step both construction surfaces share: it builds the tokenizer and
// the generation wiring every sequence of the pipeline is made from (`Pipeline.sequence` / `chat`
// pass `state.wiring` and render with `state.tokenizer`). A Session needs a real device, so this is
// the GPU-free point that shows `pipelineConfig.chatDialect` reaching the tokenizer and
// `pipelineConfig.suppressTokens` reaching the selection's wiring; the selection itself masking
// the wiring's set is `generation_sequence_test.ts`'s.

describe("gemma4 build: chatDialect reaches the tokenizer, suppressTokens the wiring", () => {
  const VOCAB = 300;
  const byteToken = (byte: number): string =>
    `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;
  /** A compiled tokenizer asset with the added tokens the stop set needs (`gemma4StopTokens`). */
  const tokenizerBytes = (): Uint8Array<ArrayBuffer> => {
    const added = ["<turn|>", "<|tool_response>", "<|channel>", "<channel|>"];
    const vocab = [
      "<unk>",
      "<bos>",
      "<eos>",
      ...Array.from({ length: 256 }, (_, byte) => byteToken(byte)),
      ...added,
    ];
    const text = JSON.stringify({
      format: "karume-gemma-tokenizer/1",
      spec: {
        normalizer: "replace-space-with-metaspace",
        preTokenizer: "split-space-merged-with-previous",
        decoder: "metaspace-byte-fallback-fuse",
        postProcessor: "none",
      },
      vocab,
      mergesText: "",
      mergesCount: 0,
      byteIds: Array.from({ length: 256 }, (_, byte) => vocab.indexOf(byteToken(byte))),
      addedTokens: [
        ["<unk>", 0],
        ["<bos>", 1],
        ["<eos>", 2],
        ...added.map((token) => [token, vocab.indexOf(token)]),
      ],
      specialIds: [0, 1, 2],
      unkId: 0,
      bosId: 1,
      eosId: 2,
    });
    const bytes = new TextEncoder().encode(text);
    const out = new Uint8Array(new ArrayBuffer(bytes.byteLength));
    out.set(bytes);
    return out;
  };
  /** A PLE-free product graph declaration (gemma-4-12B-it's input set) of vocabulary {@link VOCAB}. */
  const component = (): ModelComponent => ({
    ...stubModel({
      symbols: ["C", "M", "R"],
      inputs: [
        { name: "input_ids", shape: [1, "M"], dtype: "i32" },
        { name: "rope_sliding_attention_cos", shape: [1, "M", 256] },
        { name: "rope_sliding_attention_sin", shape: [1, "M", 256] },
        { name: "rope_full_attention_cos", shape: [1, "M", 512] },
        { name: "rope_full_attention_sin", shape: [1, "M", 512] },
        { name: "last_row", shape: ["R"], dtype: "i32" },
      ],
      outputs: ["logits", "hidden"],
      values: { logits: [1, "R", VOCAB], hidden: [1, "R", 16] },
    }),
    assets: {},
    // The build of a PLE-free model reads no asset and builds no Session.
    asset: () => {
      throw new Error("the build read an asset");
    },
    createSession: () => Promise.reject(new Error("the build created a Session")),
    estimate: () => {
      throw new Error("the build estimated");
    },
    maxPartBytes: 0,
  });
  const build = (raw: Record<string, unknown>) => {
    const config = parsePipelineConfig(raw);
    const model = component();
    const entry = gemmaEntryName("gemma4");
    return buildGemma4Program(
      admitGemma4(model, config, undefined, entry),
      () => model,
      tokenizerBytes(),
      {},
      { family: "gemma4" },
      { residency: "none" },
    );
  };

  it("the 12B declarations: the tokenizer renders gemma4-unified, the wiring masks the ids", () => {
    const built = build({ ...MINIMAL, chatDialect: "gemma4-unified", suppressTokens: [7, 260] });
    assertEquals(built.tokenizer.chatDialect, "gemma4-unified");
    assertEquals(built.wiring.suppressTokens, [7, 260]);
    assertEquals(built.ple, undefined, "a PLE-free model builds no host gather");
  });

  it("no declaration (E2B / E4B): the default dialect and an empty set", () => {
    const built = build({ ...MINIMAL });
    assertEquals(built.tokenizer.chatDialect, "gemma4");
    assertEquals(built.wiring.suppressTokens, []);
  });
});
