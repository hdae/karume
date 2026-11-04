/**
 * **Construction checks** of `Encoder` (`@karume/models/umt5`) and the contract of umT5's graph declarations
 * (`umt5GraphContract`). Only what needs neither a GPU nor real assets (real-GPU encoding is in
 * `e2e_wan_umt5_test.ts`).
 *
 * Three points are pinned:
 *
 * ① **Family admission** (pipeline name / major, `pipelineConfig`, quant, the tokenizer asset declaration, the
 *    quant's `session`, the graph declarations) fails **before touching a GPU**. On the full-assets surface
 *    (`fromAssets`), with the containers present, it fails with that violation's message (without containers it
 *    fails with the entry diagnostic — the control).
 * ② On the fetch surface (`fromPretrained` — a fake HF), an admission violation fails **without fetching a single
 *    weight part** (only the manifest and part 0 (the descriptor) were fetched). The positive path stops at
 *    "parsing the tokenizer asset" — having passed admission, the weight-part fetch and the asset fetch (the same
 *    "observe passing through by the next stage's message" form as `wan_components_test.ts`). A shared fake GPU is
 *    passed, and no adapter is touched either.
 * ③ There is no default source (`SOURCES` is an opt-in table the caller passes, not a default — ADR 0073. The hint
 *    message for a missing ref is in `repo_ref_test.ts`).
 */

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { type FileRef, parseManifest } from "@karume/hub";
import { ModelInputError } from "../src/errors.ts";
import type { GraphOwner } from "../src/hub/components.ts";
import { Encoder, type FromPretrainedOptions } from "../src/umt5/encoder.ts";
import { umt5GraphContract } from "../src/umt5/graph-contract.ts";
import { UMT5_TOKENIZER_FORMAT } from "../src/umt5/tokenizer.ts";
import type { ModelInput, TensorInput } from "../../runtime/tests/helpers/container-write.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import {
  parseIrDeclarationValue,
  partAssets,
  writeContainer,
} from "./helpers/container-fixture.ts";
import {
  fileRef,
  HUB_URL,
  MANIFEST_PATH,
  REPO,
  serveContainer,
  serveRepos,
  SHA,
} from "./helpers/container-loading-fixture.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";

/** The shape of umT5's vocabulary embedding (2 vocabulary rows × width 32 — the width is read from the declaration). */
const EMBED_ROWS = 2;
const HIDDEN = 32;

const f32Bytes = (values: readonly number[]): Uint8Array<ArrayBuffer> =>
  new Uint8Array(Float32Array.from(values).buffer);

/**
 * Storage of the vocabulary embedding (i8 = per-channel, i4 = a group per 32 elements — the control for the storage
 * check).
 */
const embedWeight = (storage: "i8" | "i4"): TensorInput => {
  const elements = EMBED_ROWS * HIDDEN;
  if (storage === "i8") {
    return {
      graph: "text_encoder",
      initializer: "embed",
      bytes: new Uint8Array(elements).fill(1),
      encoding: {
        codec: "int8-sym",
        groupSize: HIDDEN,
        scale: { bytes: f32Bytes(Array(EMBED_ROWS).fill(0.5)), dtype: "f32" },
      },
    };
  }
  const groupSize = 32;
  return {
    graph: "text_encoder",
    initializer: "embed",
    bytes: new Uint8Array(elements / 2).fill(1),
    encoding: {
      codec: "int4-sym-g",
      groupSize,
      scale: {
        bytes: f32Bytes(Array(EMBED_ROWS * (HIDDEN / groupSize)).fill(0.5)),
        dtype: "f32",
      },
    },
  };
};

/**
 * umT5's declarations (the contract's normal form — two i32 inputs, symbol L, output `[1, L, 32]`, i8 weights).
 * Never executed.
 */
const umt5Input = (storage: "i8" | "i4" = "i8"): ModelInput => ({
  graphs: {
    text_encoder: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["embedding"] },
      symbols: ["L"],
      inputs: [
        { name: "input_ids", dtype: "i32", shape: [1, "L"] },
        { name: "relative_position_buckets", dtype: "i32", shape: ["L", "L"] },
      ],
      outputs: ["last_hidden_state"],
      initializers: { embed: {} },
      values: {
        embed: { dtype: "f32", shape: [EMBED_ROWS, HIDDEN] },
        last_hidden_state: { dtype: "f32", shape: [1, "L", HIDDEN] },
      },
      states: {},
      nodes: [
        {
          op: "embedding",
          ins: ["embed", "input_ids"],
          outs: ["last_hidden_state"],
          attrs: { padding_idx: -1 },
        },
      ],
    }),
  },
  consts: [],
  weights: [embedWeight(storage)],
  assets: [],
  provenance: { license: "test" },
});

/** A minimal tokenizer asset (format `karume-umt5-tokenizer/1` — 6 vocabulary rows, limit 512). */
const tokenizerJson = (patch: Record<string, unknown> = {}): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(
    JSON.stringify({
      format: UMT5_TOKENIZER_FORMAT,
      maxLength: 512,
      unkId: 2,
      eosId: 1,
      addedTokens: [
        ["<pad>", 0],
        ["</s>", 1],
        ["<s>", 2],
        ["<unk>", 3],
      ],
      space: [
        [0x09, 0x0d],
        [0x20, 0x20],
      ],
      vocabText: "<pad>\n</s>\n<s>\n<unk>\n▁cat\n▁",
      scores: [0, 0, 0, 0, -1, -2],
      ...patch,
    }),
  );

/**
 * The tokenizer asset declaration (the `fromAssets` checks do not read the contents, so the three fields only match in
 * spelling).
 */
const TOKENIZER_REF = { path: "xxl/tokenizer/tokenizer.json", size: 8, sha256: "f".repeat(64) };

/**
 * Skeleton of the distribution `karume-umt5-xxl` (the same fields as `models/karume-umt5-xxl/karume.json` — only
 * what the checks need). `patch` overrides the contents of `models["xxl"]`.
 */
const manifestOf = (textEncoder: Record<string, unknown>, patch: Record<string, unknown> = {}) =>
  parseManifest(
    JSON.stringify({
      format: "karume/5",
      generator: "karume/0.1.0",
      defaultModel: "xxl",
      models: {
        xxl: {
          pipeline: "umt5-encoder/1",
          weights: { text_encoder: { i8: textEncoder } },
          assets: { tokenizer: TOKENIZER_REF },
          quants: { i8: { weights: { text_encoder: "i8" }, session: {} } },
          defaultQuant: "i8",
          pipelineConfig: {},
          ...patch,
        },
      },
    }),
  );

const written = await writeContainer(umt5Input());
const writtenI4 = await writeContainer(umt5Input("i4"));
/**
 * The manifest declaration (the full-assets checks do not read the parts' contents, so only the descriptor and the
 * parts' three fields).
 */
const DECLARED = {
  container: {
    descriptor: written.descriptor,
    parts: await Promise.all(
      written.parts.map((bytes, index) => fileRef(`xxl/text_encoder/model.i8-${index}.krm`, bytes)),
    ),
  },
};
const COMPONENTS = partAssets("text_encoder", written);

/**
 * A shared fake GPU (never reads adapter limits, cannot set up a Session — touching the GPU would fail with a different
 * message).
 */
const fakeGpu = () => fakeGpuContext(fakeDevice());

describe("fromAssets: family admission fails before touching a GPU (with the containers present)", () => {
  const rejected: readonly [string, Record<string, unknown>, string][] = [
    ["wrong pipeline name", { pipeline: "wan/1" }, "the manifest's pipeline is 'wan/1'"],
    ["unknown major", { pipeline: "umt5-encoder/2" }, "unsupported major"],
    ["a pipelineConfig field", { pipelineConfig: { maxLength: 512 } }, "unknown key 'maxLength'"],
    [
      "no tokenizer asset declaration (the shape of a mirror made by import)",
      { assets: {} },
      "have no tokenizer asset 'tokenizer'",
    ],
    [
      "an execution knob in the quant's session",
      { quants: { i8: { weights: { text_encoder: "i8" }, session: { linearCompute: "a8" } } } },
      "session.linearCompute is not supported",
    ],
  ];
  for (const [label, patch, message] of rejected) {
    it(label, async () => {
      const error = await assertRejects(
        () =>
          Encoder.fromAssets(
            {
              manifest: manifestOf(DECLARED, patch),
              assets: { ...COMPONENTS, tokenizer: tokenizerJson() },
            },
            { gpu: fakeGpu() },
          ),
        Error,
        message,
      );
      assert(
        !(error instanceof ModelInputError),
        `an asset inconsistency is not an input error: ${label}`,
      );
    });
  }

  it("a misspelt model / quant fails with the list of available ones", async () => {
    const input = { manifest: manifestOf(DECLARED), assets: COMPONENTS };
    await assertRejects(
      () => Encoder.fromAssets(input, { model: "base", gpu: fakeGpu() }),
      Error,
      "model 'base' is not in the manifest (available: xxl)",
    );
    await assertRejects(
      () => Encoder.fromAssets(input, { quant: "f16", gpu: fakeGpu() }),
      Error,
      "quant 'f16' is not in the manifest (available: i8)",
    );
  });

  it("rejects a container whose graph declaration storage is not i8 / f32 (i4)", async () => {
    await assertRejects(
      () =>
        Encoder.fromAssets(
          {
            manifest: manifestOf(DECLARED),
            assets: { ...partAssets("text_encoder", writtenI4), tokenizer: tokenizerJson() },
          },
          { gpu: fakeGpu() },
        ),
      Error,
      "text_encoder initializer 'embed' has storage int4-sym-g, which is not accepted",
    );
  });

  it("control: without containers it fails with the entry diagnostic, not a manifest check", async () => {
    await assertRejects(
      () => Encoder.fromAssets({ manifest: manifestOf(DECLARED), assets: {} }),
      Error,
      "component 'text_encoder' has no container",
    );
  });
});

describe("fromAssets: the tokenizer asset's contents are checked before touching a GPU", () => {
  const build = (tokenizer: Uint8Array<ArrayBuffer>) =>
    Encoder.fromAssets(
      { manifest: manifestOf(DECLARED), assets: { ...COMPONENTS, tokenizer } },
      { gpu: fakeGpu() },
    );

  it("fails, listing the keys that are present, when the asset is declared but its bytes are missing", async () => {
    await assertRejects(
      () =>
        Encoder.fromAssets(
          { manifest: manifestOf(DECLARED), assets: COMPONENTS },
          { gpu: fakeGpu() },
        ),
      Error,
      "asset 'tokenizer' is missing",
    );
  });

  it("does not read Wan's tokenizer asset (a different format with preprocessing tables)", async () => {
    await assertRejects(
      () => build(tokenizerJson({ format: "karume-wan-umt5-tokenizer/1" })),
      Error,
      "cannot read format karume-wan-umt5-tokenizer/1",
    );
  });

  it("rejects an asset whose maximum length differs from the graph's valid-length upper limit 512", async () => {
    await assertRejects(
      () => build(tokenizerJson({ maxLength: 256 })),
      Error,
      "maxLength 256 differs from umT5's valid-length upper limit 512",
    );
  });
});

// ---- The fetch surface (fake HF) ---------------------------------------------

/** Parts with contents (a part of length 0 is not fetched — hub does not fetch it either). */
const nonEmpty = (parts: readonly FileRef[]): readonly FileRef[] =>
  parts.filter((part) => part.size > 0);

/** The umT5 distribution served on a fake HF (`patch` replaces fields of `models["test"]`). */
const serveUmt5 = async (patch: Record<string, unknown>, tokenizer: Uint8Array<ArrayBuffer>) => {
  const served = await serveContainer("xxl/text_encoder/model.i8", umt5Input(), {
    partBytes: 1024,
    blockBytes: 512,
  });
  const tokenizerRef = await fileRef("xxl/tokenizer/tokenizer.json", tokenizer);
  const mock = serveRepos([
    {
      repo: REPO,
      models: {
        test: {
          pipeline: "umt5-encoder/1",
          weights: { text_encoder: { i8: served.entry } },
          assets: { tokenizer: tokenizerRef },
          quants: { i8: { weights: { text_encoder: "i8" }, session: {} } },
          defaultQuant: "i8",
          pipelineConfig: {},
          ...patch,
        },
      },
      files: [...served.files, [tokenizerRef.path, tokenizer]],
    },
  ]);
  const load = (options: FromPretrainedOptions = {}) =>
    Encoder.fromPretrained(
      { repo: REPO, revision: SHA, hubUrl: HUB_URL },
      {
        gpu: fakeGpu(),
        fetch: mock.fetch,
        caches: new MemoryCacheStorage(),
        ...options,
      },
    );
  return { load, paths: mock.paths, parts: served.parts, tokenizerRef };
};

describe("fromPretrained: an admission violation fails without fetching a single weight part", () => {
  const rejected: readonly [string, Record<string, unknown>, string][] = [
    ["no tokenizer asset declaration", { assets: {} }, "have no tokenizer asset 'tokenizer'"],
    ["wrong pipeline name", { pipeline: "siglip2/1" }, "the manifest's pipeline is 'siglip2/1'"],
    ["unknown major", { pipeline: "umt5-encoder/2" }, "unsupported major"],
  ];
  for (const [label, patch, message] of rejected) {
    it(label, async () => {
      const rig = await serveUmt5(patch, tokenizerJson());
      await assertRejects(() => rig.load(), Error, message);
      // Only the manifest and part 0 (the descriptor) were fetched — not the weight parts nor the tokenizer asset.
      assertEquals(rig.paths, [MANIFEST_PATH, rig.parts[0].path]);
    });
  }

  it("positive path: passes admission, fetches the weight parts and the tokenizer asset, then stops at parsing the asset", async () => {
    // The observation set-up: the tokenizer asset is JSON without a format field. Construction runs admission →
    // weight-part fetch → asset fetch → asset parsing, so failing with this message means every earlier stage passed
    // (no GPU is touched).
    const rig = await serveUmt5({}, new TextEncoder().encode("{}"));
    await assertRejects(() => rig.load(), Error, "tokenizer.format: not a string");
    for (const part of nonEmpty(rig.parts)) {
      assert(rig.paths.includes(part.path), `weight part ${part.path} was not fetched`);
    }
    assert(rig.paths.includes(rig.tokenizerRef.path), "the tokenizer asset was not fetched");
  });
});

describe("fromPretrained: there is no default source", () => {
  it("omitting ref (a JS call outside the type check) fails, saying that repo is required", async () => {
    await assertRejects(
      // oxlint-disable-next-line typescript/unbound-method -- Reflect.apply supplies Encoder as `this`
      () => Reflect.apply(Encoder.fromPretrained, Encoder, []),
      Error,
      "umt5.Encoder.fromPretrained: repo is required (there is no default source)",
    );
  });
});

// ---- The graph declaration contract (calling the check directly) -----------

/** A declaration-only `GraphOwner` (only the output shape varies — a single i8 weight). */
const ownerOf = (outputShape: readonly (number | string)[]): GraphOwner => ({
  graph: {
    format: "karume-ir",
    version: 2,
    requires: { ops: [] },
    symbols: ["L", "W"],
    inputs: [
      { name: "input_ids", dtype: "i32", shape: [1, "L"] },
      { name: "relative_position_buckets", dtype: "i32", shape: ["L", "L"] },
    ],
    outputs: ["out"],
    initializers: { w: { storage: { codec: "int8-sym" } } },
    values: { out: { dtype: "f32", shape: [...outputShape] } },
    states: {},
    nodes: [],
  },
});

describe("umt5GraphContract: the output width", () => {
  it("reads the declaration's static width when no width is passed (the standalone encoder)", () => {
    assertEquals(umt5GraphContract(ownerOf([1, "L", 4096]), "umt5.Encoder"), {
      output: "out",
      hiddenSize: 4096,
    });
  });

  it("fails, saying it cannot be read, when the declared width is symbolic (the host reads the width from the declaration)", () => {
    assertThrows(
      () => umt5GraphContract(ownerOf([1, "L", "W"]), "umt5.Encoder"),
      Error,
      "axis 2 is not a static positive integer",
    );
  });

  it("checks the declared width against a width that is passed (Wan passes the DiT's context width)", () => {
    assertEquals(umt5GraphContract(ownerOf([1, "L", 4096]), "wan.Pipeline", 4096).hiddenSize, 4096);
    assertThrows(
      () => umt5GraphContract(ownerOf([1, "L", 2048]), "wan.Pipeline", 4096),
      Error,
      "output 'out' shape [1, L, 2048] differs from the host's [1, L, 4096]",
    );
  });
});
