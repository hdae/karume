// The wiring of the generation settings (`gemma4EstimateGenerations`) that the gemma4 estimate
// (`Pipeline.estimateSessionMemory`) passes to `estimateGraphMemory`. Uses no GPU.
//
// On the real gemma4 E2B assets the bucket shapes do not dominate the peak, so the check on `peakAccountedBytes`
// (`e2e_gemma4_drafter_test.ts` ②) does not break even if the bucket wiring is removed. The wiring is pinned here directly.

import { assert, assertEquals, assertThrows } from "@std/assert";
import { afterEach, beforeEach, describe, it, vi } from "vite-plus/test";
import type { Session, SessionOptions } from "@karume/runtime";
import { GpuContext, readAdapterInfo } from "../../runtime/src/gpu/device.ts";
import { PreparedModel } from "../../runtime/src/runtime/executor.ts";
import { fakeDevice } from "../../runtime/tests/helpers/fake-gpu.ts";
import type { ModelInput } from "../../runtime/tests/helpers/container-write.ts";
import {
  CHUNK_BUCKETS,
  gemma4EstimateGenerations,
  Pipeline,
  type PipelineOptions,
} from "../src/gemma/pipeline.ts";
import { parseIrDeclarationValue } from "./helpers/container-fixture.ts";
import {
  fileRef,
  HUB_URL,
  REPO,
  serveContainer,
  serveRepo,
  SHA,
} from "./helpers/container-loading-fixture.ts";
import { fakeSession, programOf } from "./helpers/generation-fake.ts";

/** A pipeline with chunk length 16 and bucket ladder [4, 8] (capacity symbol `C`, row symbol `R`). */
const wiring = () =>
  programOf(fakeSession(), { chunkLength: 16, chunkBuckets: [4, 8], capacity: 64 });

describe("the generation settings the gemma4 estimate passes to the main model", () => {
  it("passes the pipeline's chunkBuckets that are below the estimated chunkLength", () => {
    const at = (chunkLength: number) =>
      gemma4EstimateGenerations(wiring(), { capacity: 64, chunkLength }).target.chunkBuckets;
    assertEquals(
      at(16),
      [4, 8],
      "at the same chunk length as the pipeline the ladder is passed as is",
    );
    assertEquals(
      at(8),
      [4],
      "a bucket equal to the chunk length is not in the accepted set (below the chunk length)",
    );
    assertEquals(at(4), [], "an empty accepted set passes no bucket");
  });

  it("even with an assumed chunkLength larger than the pipeline's, the buckets passed stay the pipeline's ladder", () => {
    const { target } = gemma4EstimateGenerations(wiring(), { capacity: 64, chunkLength: 32 });
    assertEquals(target.chunkBuckets, [4, 8]);
  });

  it("passes the estimated chunk length and capacity as is", () => {
    const { target } = gemma4EstimateGenerations(wiring(), { capacity: 48, chunkLength: 16 });
    assertEquals(target.chunkLength, 16);
    assertEquals(target.bindings, { C: 48 });
  });

  it("adds no extra scenario for a pipeline without a drafter", () => {
    const { target } = gemma4EstimateGenerations(wiring(), { capacity: 64, chunkLength: 16 });
    assertEquals(target.scenarios, undefined);
  });

  it("with speculation, adds verify with k+1 rows and physical rows rounded up to a bucket", () => {
    const { target } = gemma4EstimateGenerations(wiring(), {
      capacity: 64,
      chunkLength: 16,
      verifyQueryLength: 3,
    });
    assertEquals(target.scenarios, [{ name: "verify", chunkLength: 4, bindings: { R: 3 } }]);
  });

  it("with an assumed chunkLength larger than the pipeline's and k+1 beyond the ladder's largest bucket, verify runs at the assumed chunk length", () => {
    // Rounding with the pipeline's chunk length 16 would pass 16, which is not in the allowed set {1, 4, 8, 32} of
    // the estimated context (chunk length 32, ladder [4, 8]), and the estimate would become an ExecutionError.
    const { target } = gemma4EstimateGenerations(wiring(), {
      capacity: 64,
      chunkLength: 32,
      verifyQueryLength: 10,
    });
    assertEquals(target.scenarios, [{ name: "verify", chunkLength: 32, bindings: { R: 10 } }]);
  });
});

describe("the generation settings the gemma4 estimate passes to the drafter (the borrowing context)", () => {
  it("chunk length 1, the same capacity as the lender, and no buckets", () => {
    const { borrower } = gemma4EstimateGenerations(wiring(), {
      capacity: 48,
      chunkLength: 16,
      verifyQueryLength: 4,
    });
    assertEquals(borrower, { chunkLength: 1, bindings: { C: 48 } });
  });
});

// ---- The row-block limit (`attentionRowBlockMaxBytes`, ADR 0124) from the options to the Sessions and the estimate ----
//
// A real pipeline is built from a declaration-only distribution on a fake Hugging Face. Only
// Session construction is replaced (it needs a real device): everything before it (admission, the
// wiring) and the estimator after it are the production code, so the numbers below are what the
// runtime computes for these declarations.

describe("gemma.Pipeline attentionRowBlockMaxBytes", () => {
  const MIB = 1024 * 1024;
  /** The capacity of the fixture: the prefill score matrix below is 512 MiB at `CHUNK_ROWS`. */
  const CAPACITY = 131072;
  const CHUNK_ROWS = 128;
  /** Well above the score matrix, so the binding limit alone would not split it. */
  const BINDING_LIMIT = 2048 * MIB;
  const VOCAB = 300;
  const HIDDEN = 16;

  const rope = (name: string, width: number, rows: number | string) => ({
    name,
    dtype: "f32",
    shape: [1, rows, width],
  });
  const ropeInputs = (rows: number | string) => [
    rope("rope_sliding_attention_cos", 256, rows),
    rope("rope_sliding_attention_sin", 256, rows),
    rope("rope_full_attention_cos", 512, rows),
    rope("rope_full_attention_sin", 512, rows),
  ];
  const f32Weight = (graph: string, initializer: string, elements: number) => ({
    graph,
    initializer,
    bytes: new Uint8Array(new Float32Array(elements).buffer),
    encoding: { codec: "f32" as const },
  });

  /**
   * One states-form attention of 8 heads over k / v slots `[1,8,C,32]`: the prefill score matrix is
   * `8·M·C·4` bytes and one of its rows is `8·C·4` = 4 MiB. The rest of the graph only satisfies
   * admission (two exits by row, an embedding table the drafter can borrow).
   */
  const targetComponent = (): ModelInput => ({
    graphs: {
      model: parseIrDeclarationValue({
        format: "karume-ir",
        version: 2,
        requires: { ops: ["attention", "state_append", "reshape", "embedding"] },
        symbols: ["C", "M", "R"],
        inputs: [
          { name: "input_ids", dtype: "i32", shape: [1, "M"] },
          ...ropeInputs("M"),
          { name: "last_row", dtype: "i32", shape: ["R"] },
        ],
        outputs: ["logits", "hidden"],
        initializers: { table: {}, hidden_table: {} },
        values: {
          table: { dtype: "f32", shape: [8, VOCAB] },
          hidden_table: { dtype: "f32", shape: [8, HIDDEN] },
          q: { dtype: "f32", shape: [1, 8, "M", 32] },
          k: { dtype: "f32", shape: [1, 8, "M", 32] },
          o: { dtype: "f32", shape: [1, 8, "M", 32] },
          logits_rows: { dtype: "f32", shape: ["R", VOCAB] },
          hidden_rows: { dtype: "f32", shape: ["R", HIDDEN] },
          logits: { dtype: "f32", shape: [1, "R", VOCAB] },
          hidden: { dtype: "f32", shape: [1, "R", HIDDEN] },
        },
        states: {
          kslot: { dtype: "f32", shape: [1, 8, "C", 32] },
          vslot: { dtype: "f32", shape: [1, 8, "C", 32] },
        },
        nodes: [
          { op: "reshape", ins: ["rope_sliding_attention_cos"], outs: ["q"], attrs: {} },
          { op: "reshape", ins: ["rope_sliding_attention_sin"], outs: ["k"], attrs: {} },
          {
            op: "attention",
            ins: ["q", "k", "k"],
            outs: ["o"],
            attrs: { scale: 0.5 },
            states: { k: "kslot", v: "vslot" },
          },
          { op: "state_append", ins: ["k"], outs: [], attrs: {}, states: { slot: "kslot" } },
          { op: "state_append", ins: ["k"], outs: [], attrs: {}, states: { slot: "vslot" } },
          {
            op: "embedding",
            ins: ["table", "last_row"],
            outs: ["logits_rows"],
            attrs: { padding_idx: -1 },
          },
          {
            op: "embedding",
            ins: ["hidden_table", "last_row"],
            outs: ["hidden_rows"],
            attrs: { padding_idx: -1 },
          },
          { op: "reshape", ins: ["logits_rows"], outs: ["logits"], attrs: {} },
          { op: "reshape", ins: ["hidden_rows"], outs: ["hidden"], attrs: {} },
        ],
      }),
    },
    consts: [],
    weights: [
      f32Weight("model", "table", 8 * VOCAB),
      f32Weight("model", "hidden_table", 8 * HIDDEN),
    ],
    assets: [],
    provenance: { license: "test" },
  });

  /**
   * The drafter's readonly attention has **twice the heads** of the target's, so one of its rows is
   * 8 MiB: a limit between 4 and 8 MiB can only fail on the drafter's side.
   */
  const drafterComponent = (): ModelInput => ({
    graphs: {
      drafter: parseIrDeclarationValue({
        format: "karume-ir",
        version: 2,
        requires: { ops: ["attention", "reshape", "embedding", "argmax"] },
        symbols: ["C"],
        inputs: [
          { name: "token", dtype: "i32", shape: [1, 1] },
          { name: "hidden", dtype: "f32", shape: [1, HIDDEN] },
          ...ropeInputs(1),
        ],
        outputs: ["draft_0", "draft_1", "draft_2"],
        initializers: { table: { shared: true } },
        values: {
          table: { dtype: "f32", shape: [8, VOCAB] },
          q: { dtype: "f32", shape: [1, 16, 1, 32] },
          o: { dtype: "f32", shape: [1, 16, 1, 32] },
          embedded: { dtype: "f32", shape: [1, 1, VOCAB] },
          draft_0: { dtype: "i32", shape: [1, 1, 1] },
          draft_1: { dtype: "i32", shape: [1, 1, 1] },
          draft_2: { dtype: "i32", shape: [1, 1, 1] },
        },
        states: {
          kslot: { dtype: "f32", shape: [1, 8, "C", 32], external: true },
          vslot: { dtype: "f32", shape: [1, 8, "C", 32], external: true },
        },
        nodes: [
          { op: "reshape", ins: ["rope_full_attention_cos"], outs: ["q"], attrs: {} },
          {
            op: "attention",
            ins: ["q"],
            outs: ["o"],
            attrs: { scale: 0.5, readonly: true },
            states: { k: "kslot", v: "vslot" },
          },
          {
            op: "embedding",
            ins: ["table", "token"],
            outs: ["embedded"],
            attrs: { padding_idx: -1 },
          },
          { op: "argmax", ins: ["embedded"], outs: ["draft_0"], attrs: {} },
          { op: "argmax", ins: ["embedded"], outs: ["draft_1"], attrs: {} },
          { op: "argmax", ins: ["embedded"], outs: ["draft_2"], attrs: {} },
        ],
      }),
    },
    consts: [],
    weights: [],
    assets: [],
    provenance: { license: "test" },
  });

  /** A compiled tokenizer asset with the added tokens the stop set needs (`gemma4StopTokens`). */
  const tokenizerBytes = (): Uint8Array<ArrayBuffer> => {
    const byteToken = (byte: number): string =>
      `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;
    const added = ["<turn|>", "<|tool_response>", "<|channel>", "<channel|>"];
    const vocab = [
      "<unk>",
      "<bos>",
      "<eos>",
      ...Array.from({ length: 256 }, (_, byte) => byteToken(byte)),
      ...added,
    ];
    const bytes = new TextEncoder().encode(
      JSON.stringify({
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
      }),
    );
    const out = new Uint8Array(new ArrayBuffer(bytes.byteLength));
    out.set(bytes);
    return out;
  };

  /** A real GpuContext on a fake device, with the limits the estimator reads. */
  const gpuContext = (): GpuContext =>
    new GpuContext(
      fakeDevice(),
      readAdapterInfo({}),
      {
        maxBufferSize: BINDING_LIMIT,
        maxStorageBufferBindingSize: BINDING_LIMIT,
        maxUniformBufferBindingSize: 65536,
        maxStorageBuffersPerShaderStage: 10,
        maxUniformBuffersPerShaderStage: 12,
        maxComputeWorkgroupStorageSize: 16384,
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxComputeWorkgroupSizeY: 256,
        maxComputeWorkgroupSizeZ: 64,
        maxComputeWorkgroupsPerDimension: 65535,
      },
      new Set(),
    );

  /** The options each `createContainerSession` received, in construction order (target first). */
  let sessionOptions: SessionOptions[] = [];
  /** The prepared models Sessions were built from, in the same order. */
  let prepared: PreparedModel[] = [];

  beforeEach(() => {
    sessionOptions = [];
    prepared = [];
    vi.spyOn(PreparedModel.prototype, "createContainerSession").mockImplementation(function (
      this: PreparedModel,
      _gpu,
      options = {},
    ) {
      sessionOptions.push(options);
      prepared.push(this);
      return Promise.resolve({
        exportWeight: () => ({}),
        dispose: () => Promise.resolve(),
      } as unknown as Session);
    });
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  /** One row of the fixture's tables has to fit a block. */
  const BLOCKS = { partBytes: 1024 * 1024, blockBytes: 64 * 1024 };

  const load = async (options: PipelineOptions): Promise<Pipeline> => {
    const model = await serveContainer("model/model.f32", targetComponent(), BLOCKS);
    const drafter = await serveContainer("drafter/model.f32", drafterComponent(), BLOCKS);
    const tokenizer = tokenizerBytes();
    const rig = await serveRepo(
      {
        test: {
          pipeline: "gemma4/1",
          weights: { model: { f32: model.entry }, drafter: { f32: drafter.entry } },
          assets: { tokenizer: await fileRef("tokenizer.json", tokenizer) },
          quants: { f32: { weights: { model: "f32", drafter: "f32" }, session: {} } },
          defaultQuant: "f32",
          pipelineConfig: {
            chunkLength: CHUNK_ROWS,
            maxChunkLength: CHUNK_ROWS,
            maxPosition: CAPACITY,
            capacity: CAPACITY,
            rope: {
              sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
              full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
            },
          },
        },
      },
      new Map([...model.files, ...drafter.files, ["tokenizer.json", tokenizer]]),
      { model: model.parts, drafter: drafter.parts },
    );
    return await Pipeline.fromPretrained(
      { repo: REPO, revision: SHA, hubUrl: HUB_URL },
      {
        ...options,
        fetch: rig.mock.fetch,
        caches: rig.hubOptions.caches,
        gpu: gpuContext(),
        // Without a diagnostics callback the small-output decode would build its GPU resources.
        onRunDiagnostics: () => undefined,
      },
    );
  };

  describe("the Sessions", () => {
    it("the target Session and the drafter Session both get the value", async () => {
      await load({ speculative: {}, attentionRowBlockMaxBytes: 16 * MIB });
      assertEquals(
        sessionOptions.map((options) => options.attentionRowBlockMaxBytes),
        [16 * MIB, 16 * MIB],
      );
    });

    it("unset, the field does not exist on either Session's options", async () => {
      await load({ speculative: {} });
      assertEquals(sessionOptions.length, 2);
      for (const options of sessionOptions) {
        assertEquals(Object.hasOwn(options, "attentionRowBlockMaxBytes"), false);
      }
    });
  });

  // The neighbouring knob on the same fixture: a real pipeline, a real estimator.
  describe("fullStateStorage", () => {
    /** The fixture's two full slots `[1,8,C,32]` in f32. Written out, not derived from the code. */
    const FULL_SLOTS_F32 = 2 * (8 * CAPACITY * 32 * 4);

    it("the target Session and the drafter Session both get the value", async () => {
      await load({ speculative: {}, fullStateStorage: "f16" });
      assertEquals(
        sessionOptions.map((options) => options.fullStateStorage),
        ["f16", "f16"],
      );
    });

    it("unset, the field does not exist on either Session's options", async () => {
      await load({ speculative: {} });
      for (const options of sessionOptions) {
        assertEquals(Object.hasOwn(options, "fullStateStorage"), false);
      }
    });

    it("estimateSessionMemory reports the full slots halved, with and without a drafter", async () => {
      for (const options of [{}, { speculative: {} }]) {
        const f32 = (await load(options)).estimateSessionMemory();
        const f16 = (await load({ ...options, fullStateStorage: "f16" })).estimateSessionMemory();
        assertEquals(f32.resident.stateBytes - f16.resident.stateBytes, FULL_SLOTS_F32 / 2);
        // What `fitCapacity` reads: the peak less the weights moves by the same bytes.
        assertEquals(f32.peakAccountedBytes - f16.peakAccountedBytes, FULL_SLOTS_F32 / 2);
      }
    });

    it("the report is the runtime's own estimate, with the option passed down", async () => {
      assertEquals(
        (await load({ fullStateStorage: "f32" })).estimateSessionMemory(),
        (await load({})).estimateSessionMemory(),
      );
    });
  });

  describe("estimateSessionMemory", () => {
    it("a limit below the score matrix lowers the peak", async () => {
      const unset = (await load({})).estimateSessionMemory().peakAccountedBytes;
      const limited = (await load({ attentionRowBlockMaxBytes: 16 * MIB })).estimateSessionMemory()
        .peakAccountedBytes;
      assert(limited < unset, `limited ${limited} is not below unset ${unset}`);
    });

    it("a limit above the score matrix (no split) reports the same as unset", async () => {
      assertEquals(
        (await load({ attentionRowBlockMaxBytes: BINDING_LIMIT })).estimateSessionMemory(),
        (await load({})).estimateSessionMemory(),
      );
    });

    it("the report is the runtime's own estimate, with the limit when set and without the field when unset", async () => {
      const estimateOf = async (options: PipelineOptions) => {
        const report = (await load(options)).estimateSessionMemory();
        // The options the pipeline passes for this graph, spelled out without the limit.
        const direct = {
          bindings: { R: 1 },
          generation: {
            chunkLength: CHUNK_ROWS,
            bindings: { C: CAPACITY },
            chunkBuckets: CHUNK_BUCKETS.filter((rows) => rows < CHUNK_ROWS),
          },
          maxStorageBufferBindingSize: BINDING_LIMIT,
          maxBufferSize: BINDING_LIMIT,
        };
        return { report, direct, model: prepared[prepared.length - 1] };
      };
      const unset = await estimateOf({});
      assertEquals(unset.report, unset.model.estimate(unset.direct));
      // The score matrix of the fixture alone is above the default slot-backing budget of 256 MiB,
      // so an unset limit leaves it whole in the peak.
      assert(unset.report.peakAccountedBytes > 8 * CHUNK_ROWS * CAPACITY * 4);
      const limited = await estimateOf({ attentionRowBlockMaxBytes: 16 * MIB });
      assertEquals(
        limited.report,
        limited.model.estimate({ ...limited.direct, attentionRowBlockMaxBytes: 16 * MIB }),
      );
    });

    it("a limit below one row of the target's scores fails loudly, naming the limit", async () => {
      const pipeline = await load({ attentionRowBlockMaxBytes: 2 * MIB });
      assertThrows(() => pipeline.estimateSessionMemory(), Error, "attentionRowBlockMaxBytes");
    });

    it("the drafter's estimate names the same limit: one row fits the target's budget but not the drafter's", async () => {
      const limit = 6 * MIB;
      // The control: without the drafter the same limit passes.
      (await load({ attentionRowBlockMaxBytes: limit })).estimateSessionMemory();
      const speculative = await load({ speculative: {}, attentionRowBlockMaxBytes: limit });
      assertThrows(() => speculative.estimateSessionMemory(), Error, "attentionRowBlockMaxBytes");
    });
  });
});
