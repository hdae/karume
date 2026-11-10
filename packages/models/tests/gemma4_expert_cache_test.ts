// Gemma 4's `PipelineOptions.expertCache` (ADR 0146) — the parts that need no GPU: the option's
// acceptance and the combinations it refuses, the prefill chunk rows it settles from the graph, and
// that each refusal sits before any weight part is fetched. The re-run protocol is
// `generation_expert_cache_session_test.ts`; the real-weight check is case ⑦ of
// `packages/e2e/tests/e2e_gemma4_26b_cut_test.ts`.

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { ModelInputError } from "../src/errors.ts";
import {
  assertGemma4ExpertCacheOptions,
  resolveGemma4ExpertCache,
} from "../src/gemma/expert-cache.ts";
import type { PipelineOptions } from "../src/gemma/pipeline.ts";
import { Pipeline } from "../gemma.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import type { ModelInput } from "../../runtime/tests/helpers/container-write.ts";
import { parseIrDeclarationValue } from "./helpers/container-fixture.ts";
import {
  HUB_URL,
  REPO,
  serveContainer,
  serveRepo,
  SHA,
} from "./helpers/container-loading-fixture.ts";

/** 26B's routing: 128 experts per layer, 8 picks per token. */
const EXPERTS = 128;
const TOP_K = 8;
const SLIDING_WINDOW = 1024;
/** Rows a sliding slot holds beyond the window (8 on the published graphs). */
const SLACK = 8;
/** The distribution's chunk rows (`pipelineConfig.chunkLength` of 26B). */
const CHUNK = 768;

/**
 * The parts of a 26B-shaped product graph the checks read: per layer an attention over its own k /
 * v slots (sliding layers with a window and a numeric ring of window + slack rows, the last a full
 * layer over the capacity symbol), a router `topk` and the two `expert_linear` it feeds.
 */
const moeGraph = (
  options: {
    readonly layers?: number;
    readonly picks?: number;
    readonly sliding?: boolean;
    readonly experts?: boolean;
  } = {},
) => {
  const layers = options.layers ?? 3;
  const picks = options.picks ?? TOP_K;
  const nodes: {
    readonly op: string;
    readonly ins: readonly string[];
    readonly outs: readonly string[];
    readonly attrs: Readonly<Record<string, unknown>>;
    readonly states: Readonly<Record<string, string>>;
  }[] = [];
  const values: Record<string, { readonly shape: readonly (number | string)[] }> = {};
  const states: Record<string, { readonly shape: readonly (number | string)[] }> = {};
  for (let layer = 0; layer < layers; layer += 1) {
    const full = layer === layers - 1 || options.sliding === false;
    const slot = (part: string): string => `l${layer}.${part}`;
    for (const part of ["k", "v"]) {
      states[slot(part)] = {
        shape: full ? [1, 2, "C", 512] : [1, 8, SLIDING_WINDOW + SLACK, 256],
      };
    }
    nodes.push({
      op: "attention",
      ins: [slot("q")],
      outs: [slot("attn")],
      attrs: full ? {} : { window: SLIDING_WINDOW },
      states: { k: slot("k"), v: slot("v") },
    });
    if (options.experts === false) continue;
    nodes.push({
      op: "topk",
      ins: [slot("router")],
      outs: [slot("route_weights"), slot("route_ids")],
      attrs: { k: picks },
      states: {},
    });
    for (const weight of ["gate_up", "down"]) {
      values[slot(weight)] = { shape: [EXPERTS, 64, 32] };
      nodes.push({
        op: "expert_linear",
        ins: [slot("x"), slot("route_ids"), slot(weight)],
        outs: [slot(`${weight}_out`)],
        attrs: {},
        states: {},
      });
    }
  }
  return { nodes, values, states };
};

const resolve = (options: PipelineOptions, graph = moeGraph(), chunk = CHUNK) =>
  resolveGemma4ExpertCache("T", graph, options, chunk);

describe("expertCache: the options alone", () => {
  it("is absent without the option and returns the slots with it", () => {
    assertEquals(assertGemma4ExpertCacheOptions("T", {}), undefined);
    assertEquals(assertGemma4ExpertCacheOptions("T", { expertCache: { slots: 64 } }), 64);
  });

  it("refuses slots that are not a positive integer, and a value that is not { slots }, as input errors", () => {
    for (const expertCache of [
      { slots: 0 },
      { slots: -16 },
      { slots: 16.5 },
      { slots: Number.NaN },
      { slots: "16" },
      {},
      null,
      16,
    ]) {
      assertThrows(
        () =>
          assertGemma4ExpertCacheOptions("T", {
            expertCache: expertCache as unknown as { readonly slots: number },
          }),
        ModelInputError,
        "expertCache must be { slots: <a positive integer> }",
      );
    }
  });

  it("accepts a host tier of a positive integer of bytes, Number.MAX_SAFE_INTEGER (every expert) included", () => {
    for (const hostBytes of [1, 3_345_408, Number.MAX_SAFE_INTEGER]) {
      assertEquals(
        assertGemma4ExpertCacheOptions("T", { expertCache: { slots: 64, hostBytes } }),
        64,
      );
    }
  });

  it("refuses a host tier that is not a positive safe integer as an input error, naming the field", () => {
    const cases: readonly (readonly [unknown, string])[] = [
      [0, "0"],
      [-1, "-1"],
      [1.5, "1.5"],
      [Number.NaN, "NaN"],
      [Number.MAX_SAFE_INTEGER + 1, "9007199254740992"],
      ["1", '"1"'],
      [null, "null"],
    ];
    for (const [hostBytes, shown] of cases) {
      const error = assertThrows(
        () =>
          assertGemma4ExpertCacheOptions("T", {
            expertCache: { slots: 64, hostBytes } as unknown as {
              readonly slots: number;
              readonly hostBytes?: number;
            },
          }),
        ModelInputError,
        "expertCache.hostBytes must be a positive integer",
      );
      assert(error.message.includes(`(hostBytes ${shown})`), error.message);
    }
  });

  it("refuses a depth that is not a positive safe integer as an input error, naming the field", () => {
    const cases: readonly (readonly [unknown, string])[] = [
      [0, "0"],
      [-1, "-1"],
      [1.5, "1.5"],
      [Number.NaN, "NaN"],
      ["4", '"4"'],
      [null, "null"],
    ];
    for (const [depth, shown] of cases) {
      const error = assertThrows(
        () =>
          assertGemma4ExpertCacheOptions("T", {
            expertCache: { slots: 64, depth } as unknown as {
              readonly slots: number;
              readonly depth?: number;
            },
          }),
        ModelInputError,
        "expertCache.depth must be a positive integer",
      );
      assert(error.message.includes(`(depth ${shown})`), error.message);
    }
    assertEquals(assertGemma4ExpertCacheOptions("T", { expertCache: { slots: 64, depth: 4 } }), 64);
  });

  it("refuses every combination that runs the target in a batch or an immediate run, naming it", () => {
    const cases: readonly (readonly [PipelineOptions, string])[] = [
      [{ decodeAhead: 2 }, "decodeAhead 2"],
      [{ speculative: {} }, "speculative"],
      [{ pleResidency: "gpu" }, 'pleResidency: "gpu"'],
      [{ vision: true }, "vision"],
    ];
    for (const [other, named] of cases) {
      const error = assertThrows(
        () => assertGemma4ExpertCacheOptions("T", { expertCache: { slots: 64 }, ...other }),
        ModelInputError,
        `expertCache cannot be combined with ${named}`,
      );
      assert(error.message.includes("ADR 0146"), error.message);
    }
  });

  it("accepts the values of those options that keep runs alone and deferred (negative control)", () => {
    assertEquals(
      assertGemma4ExpertCacheOptions("T", {
        expertCache: { slots: 64 },
        decodeAhead: 1,
        pleResidency: "host",
        vision: false,
      }),
      64,
    );
  });

  it("leaves those options alone without the cache", () => {
    assertEquals(
      assertGemma4ExpertCacheOptions("T", { decodeAhead: 4, speculative: {}, vision: true }),
      undefined,
    );
  });
});

describe("expertCache: the prefill chunk rows", () => {
  it("is absent without the option", () => {
    assertEquals(resolve({}), undefined);
  });

  it("defaults to min(sliding slack, ⌊slots / k⌋, the distribution's chunkLength)", () => {
    // ⌊slots / 8⌋ decides below the slack, the slack above it, and a smaller distribution chunk last.
    assertEquals(resolve({ expertCache: { slots: 16 } }), { slots: 16, chunkLength: 2 });
    assertEquals(resolve({ expertCache: { slots: 31 } }), { slots: 31, chunkLength: 3 });
    assertEquals(resolve({ expertCache: { slots: 64 } }), { slots: 64, chunkLength: 8 });
    assertEquals(resolve({ expertCache: { slots: 128 } }), { slots: 128, chunkLength: SLACK });
    assertEquals(resolve({ expertCache: { slots: 128 } }, moeGraph(), 4), {
      slots: 128,
      chunkLength: 4,
    });
  });

  it("carries the host tier into the settled option, and leaves it out without one", () => {
    assertEquals(resolve({ expertCache: { slots: 64, hostBytes: Number.MAX_SAFE_INTEGER } }), {
      slots: 64,
      hostBytes: Number.MAX_SAFE_INTEGER,
      chunkLength: 8,
    });
    assert(!Object.hasOwn(resolve({ expertCache: { slots: 64 } }) ?? {}, "hostBytes"));
  });

  it("takes no slack bound from a graph without sliding slots", () => {
    assertEquals(resolve({ expertCache: { slots: 128 } }, moeGraph({ sliding: false })), {
      slots: 128,
      chunkLength: EXPERTS / TOP_K,
    });
  });

  it("keeps a requested chunkLength within the bound and refuses one above it, naming the bound", () => {
    assertEquals(resolve({ expertCache: { slots: 64 }, chunkLength: 4 }), {
      slots: 64,
      chunkLength: 4,
    });
    for (const chunkLength of [9, CHUNK]) {
      const error = assertThrows(
        () => resolve({ expertCache: { slots: 64 }, chunkLength }),
        ModelInputError,
        `chunkLength ${chunkLength} exceeds 8`,
      );
      assert(error.message.includes("sliding slack 8"), error.message);
    }
  });

  it("refuses slots below 2 × k, saying the minimum, and slots above the experts of a layer", () => {
    for (const slots of [TOP_K, 2 * TOP_K - 1]) {
      assertThrows(
        () => resolve({ expertCache: { slots } }),
        ModelInputError,
        `expertCache.slots ${slots} is below the minimum ${2 * TOP_K}`,
      );
    }
    assertThrows(
      () => resolve({ expertCache: { slots: EXPERTS + 1 } }),
      ModelInputError,
      `exceeds the ${EXPERTS} experts of a layer`,
    );
    // The minimum follows the routing's k.
    assertEquals(resolve({ expertCache: { slots: 8 } }, moeGraph({ picks: 4 })), {
      slots: 8,
      chunkLength: 2,
    });
  });

  it("refuses a graph without expert_linear as an input error", () => {
    assertThrows(
      () => resolve({ expertCache: { slots: 64 } }, moeGraph({ experts: false })),
      ModelInputError,
      "expertCache needs a mixture-of-experts model",
    );
  });

  it("carries self-speculation's depth into the settled option up to its bound, and leaves it out without one", () => {
    // depth + 1 ≤ min(⌊slots / k⌋, slack, chunk rows): 7 at 64 slots, 1 at 16, 3 with 4 chunk rows.
    for (const [options, depth] of [
      [{ expertCache: { slots: 64, depth: 7 } }, 7],
      [{ expertCache: { slots: 64, depth: 4 } }, 4],
      [{ expertCache: { slots: 16, depth: 1 } }, 1],
      [{ expertCache: { slots: 64, depth: 3 }, chunkLength: 4 }, 3],
    ] as const) {
      assertEquals(resolve(options)?.depth, depth, JSON.stringify(options));
    }
    assertEquals(resolve({ expertCache: { slots: 128, depth: 4 } }, moeGraph({ sliding: false })), {
      slots: 128,
      depth: 4,
      chunkLength: EXPERTS / TOP_K,
    });
    assert(!Object.hasOwn(resolve({ expertCache: { slots: 64 } }) ?? {}, "depth"));
  });

  it("refuses a depth whose verify does not fit the slots, the slack or the chunk rows, naming the bound", () => {
    const cases: readonly (readonly [PipelineOptions, number])[] = [
      // ⌊16 / 8⌋ = 2 rows of picks: one draft.
      [{ expertCache: { slots: 16, depth: 2 } }, 1],
      // The slack (8) bounds one deferred run of depth + 1 rows.
      [{ expertCache: { slots: 128, depth: 8 } }, 7],
      [{ expertCache: { slots: 64, depth: 4 }, chunkLength: 4 }, 3],
    ];
    for (const [options, bound] of cases) {
      const depth = options.expertCache?.depth;
      const error = assertThrows(
        () => resolve(options),
        ModelInputError,
        `expertCache.depth ${depth} exceeds ${bound}`,
      );
      assert(error.message.includes("depth + 1"), error.message);
    }
  });

  it("fails loudly on routed layers that do not share one k", () => {
    const graph = moeGraph();
    const mixed = {
      ...graph,
      nodes: graph.nodes.map((node, at) =>
        node.op === "topk" && at === 1 ? { ...node, attrs: { k: 4 } } : node,
      ),
    };
    assertThrows(() => resolve({ expertCache: { slots: 64 } }, mixed), Error, "one topk k");
  });
});

describe("expertCache: one slot count per routed layer", () => {
  /** The research's fitted allocation at a mean of 64, its first 12 layers (smallest: 52 at 10). */
  const FITTED = [96, 108, 84, 88, 84, 68, 72, 72, 68, 60, 52, 56] as const;
  /** Small and large layers mixed (smallest: 16, first at 0). */
  const STRESS = [16, 128, 32, 96, 16, 64, 128, 16, 48, 32, 112, 16] as const;
  const twelve = moeGraph({ layers: 12 });

  it("accepts an array of positive integers and returns a frozen copy of it", () => {
    const given = [64, 32, 96];
    const slots = assertGemma4ExpertCacheOptions("T", { expertCache: { slots: given } });
    assertEquals(slots, [64, 32, 96]);
    assert(slots !== given && Object.isFrozen(slots), "the settled slots are the caller's array");
  });

  it("refuses an entry that is not a positive integer as an input error, naming it", () => {
    const cases: readonly (readonly [readonly unknown[], string])[] = [
      [[64, 0], "slots[1] 0"],
      [[-1, 64], "slots[0] -1"],
      [[64, 64, 16.5], "slots[2] 16.5"],
      [[64, "16"], 'slots[1] "16"'],
      [[64, null], "slots[1] null"],
      // oxlint-disable-next-line no-sparse-arrays -- the hole is the input under test (read as undefined)
      [[64, , 64], "slots[1] undefined"],
    ];
    for (const [slots, shown] of cases) {
      const error = assertThrows(
        () =>
          assertGemma4ExpertCacheOptions("T", {
            expertCache: { slots } as unknown as { readonly slots: readonly number[] },
          }),
        ModelInputError,
        `expertCache.${shown.split(" ")[0]} must be a positive integer, one per routed layer`,
      );
      assert(error.message.includes(`(${shown})`), error.message);
    }
  });

  it("settles the chunk rows from the smallest layer, and carries the array as given", () => {
    // ⌊52 / 8⌋ = 6 below the slack (8); ⌊16 / 8⌋ = 2.
    assertEquals(resolve({ expertCache: { slots: FITTED } }, twelve), {
      slots: FITTED,
      chunkLength: 6,
    });
    assertEquals(resolve({ expertCache: { slots: STRESS } }, twelve), {
      slots: STRESS,
      chunkLength: 2,
    });
    // Every layer at 128 is the slack's bound, as the number 128 is.
    assertEquals(
      resolve({ expertCache: { slots: new Array<number>(12).fill(128) } }, twelve)?.chunkLength,
      resolve({ expertCache: { slots: 128 } }, twelve)?.chunkLength,
    );
  });

  it("refuses a chunkLength above the smallest layer's bound, naming that layer", () => {
    const error = assertThrows(
      () => resolve({ expertCache: { slots: FITTED }, chunkLength: 7 }, twelve),
      ModelInputError,
      "chunkLength 7 exceeds 6",
    );
    assert(error.message.includes("the smallest layer, slots[10], has 52"), error.message);
    assert(error.message.includes("⌊52 / 8⌋ = 6 of the smallest layer, slots[10]"), error.message);
  });

  it("bounds the depth by the smallest layer, naming it", () => {
    assertEquals(resolve({ expertCache: { slots: STRESS, depth: 1 } }, twelve)?.depth, 1);
    assertEquals(resolve({ expertCache: { slots: FITTED, depth: 5 } }, twelve)?.depth, 5);
    for (const [slots, depth, bound, named] of [
      [STRESS, 2, 1, "slots[0], has 16"],
      [FITTED, 6, 5, "slots[10], has 52"],
    ] as const) {
      const error = assertThrows(
        () => resolve({ expertCache: { slots, depth } }, twelve),
        ModelInputError,
        `expertCache.depth ${depth} exceeds ${bound}`,
      );
      assert(error.message.includes(`the smallest layer, ${named}`), error.message);
    }
  });

  it("refuses an array that is not one entry per routed layer, naming both counts", () => {
    for (const slots of [FITTED.slice(0, 11), [...FITTED, 64], []]) {
      assertThrows(
        () => resolve({ expertCache: { slots } }, twelve),
        ModelInputError,
        `expertCache.slots has ${slots.length} entries, but the model has 12 routed layers`,
      );
    }
  });

  it("refuses an entry below 2 × k or above the experts of a layer, naming it", () => {
    const below: number[] = [...FITTED];
    below[4] = 2 * TOP_K - 1;
    assertThrows(
      () => resolve({ expertCache: { slots: below } }, twelve),
      ModelInputError,
      `expertCache.slots[4] ${2 * TOP_K - 1} is below the minimum ${2 * TOP_K}`,
    );
    const above: number[] = [...FITTED];
    above[11] = EXPERTS + 1;
    assertThrows(
      () => resolve({ expertCache: { slots: above } }, twelve),
      ModelInputError,
      `expertCache.slots[11] ${EXPERTS + 1} exceeds the ${EXPERTS} experts of a layer`,
    );
  });
});

// ---- fromAssets / fromPretrained: where the refusals sit ---------------------------------------
//
// The option checks come before a byte of the assets is read (fromAssets with no parts and a
// broken declaration, fromPretrained with a fetch that counts requests). The graph check comes
// after the descriptor and before any weight part: a distribution without `expert_linear` on a fake
// Hugging Face (`helpers/container-loading-fixture.ts`, which records every fetched path) is refused
// with no weight part fetched, while the same load without the option fetches it and fails later at
// the missing tokenizer — so "no weight part" is a property of where the refusal sits.

describe("expertCache: the refusals come before the weights", () => {
  it("fromAssets refuses a combination before reading the declaration or a part", async () => {
    await assertRejects(
      () =>
        Pipeline.fromAssets(
          { config: {} as never, model: [], tokenizer: new Uint8Array(0) },
          { expertCache: { slots: 64 }, decodeAhead: 2 },
        ),
      ModelInputError,
      "expertCache cannot be combined with decodeAhead 2",
    );
  });

  it("fromPretrained refuses an invalid option and a combination with no request sent", async () => {
    for (const options of [
      { expertCache: { slots: 0 } },
      { expertCache: { slots: 64, depth: 0 } },
      { expertCache: { slots: 64 }, pleResidency: "gpu" },
    ] as const) {
      let requests = 0;
      await assertRejects(
        () =>
          Pipeline.fromPretrained("karume/gemma4-26b", {
            ...options,
            fetch: () => {
              requests += 1;
              return Promise.reject(new Error("the check should have failed first"));
            },
          }),
        ModelInputError,
        "expertCache",
      );
      assertEquals(requests, 0, "a request went out before the check");
    }
  });

  it("fromPretrained refuses a bad entry of per-layer slots with no request sent", async () => {
    let requests = 0;
    await assertRejects(
      () =>
        Pipeline.fromPretrained("karume/gemma4-26b", {
          expertCache: { slots: [64, 0, 64] },
          fetch: () => {
            requests += 1;
            return Promise.reject(new Error("the check should have failed first"));
          },
        }),
      ModelInputError,
      "expertCache.slots[1] must be a positive integer",
    );
    assertEquals(requests, 0, "a request went out before the check");
  });

  const MODEL = "model";
  const rope = (name: string, width: number) => ({ name, dtype: "f32", shape: [1, "M", width] });
  /** The admission-passing PLE-free declaration of `gemma4_ple_free_test.ts`: no expert_linear. */
  const component = (): ModelInput => ({
    graphs: {
      [MODEL]: parseIrDeclarationValue({
        format: "karume-ir",
        version: 2,
        requires: { ops: ["mul", "state_append"] },
        symbols: ["C", "M"],
        inputs: [
          rope("rope_sliding_attention_cos", 256),
          rope("rope_sliding_attention_sin", 256),
          rope("rope_full_attention_cos", 512),
          rope("rope_full_attention_sin", 512),
        ],
        outputs: ["logits", "hidden"],
        initializers: { w: {} },
        values: {
          w: { dtype: "f32", shape: [256] },
          logits: { dtype: "f32", shape: [1, "M", 8] },
          hidden: { dtype: "f32", shape: [1, "M", 4] },
        },
        states: { kv: { dtype: "f32", shape: [1, "C", 4] } },
        nodes: [
          { op: "mul", ins: ["rope_sliding_attention_cos", "w"], outs: ["logits"], attrs: {} },
          {
            op: "mul",
            ins: ["rope_full_attention_cos", "rope_full_attention_sin"],
            outs: ["hidden"],
            attrs: {},
          },
          {
            op: "state_append",
            ins: ["rope_sliding_attention_cos"],
            outs: [],
            attrs: {},
            states: { slot: "kv" },
          },
        ],
      }),
    },
    consts: [],
    weights: [
      {
        graph: MODEL,
        initializer: "w",
        bytes: new Uint8Array(new Float32Array(256).fill(0.5).buffer),
        encoding: { codec: "f32" },
      },
    ],
    assets: [],
    provenance: { license: "test" },
  });

  const load = async (options: PipelineOptions) => {
    const served = await serveContainer(`${MODEL}/model.f32`, component());
    const rig = await serveRepo(
      {
        test: {
          pipeline: "gemma4/1",
          weights: { [MODEL]: { f32: served.entry } },
          assets: {},
          quants: { f32: { weights: { [MODEL]: "f32" }, session: {} } },
          defaultQuant: "f32",
          pipelineConfig: {
            chunkLength: 32,
            maxChunkLength: 128,
            maxPosition: 1024,
            capacity: 640,
            rope: {
              sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
              full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
            },
          },
        },
      },
      new Map(served.files),
      { [MODEL]: served.parts },
    );
    const [descriptor, ...weightParts] = served.parts.map((part) => part.path);
    const result = Pipeline.fromPretrained(
      { repo: REPO, revision: SHA, hubUrl: HUB_URL },
      {
        ...options,
        fetch: rig.mock.fetch,
        caches: rig.hubOptions.caches,
        gpu: fakeGpuContext(fakeDevice()),
      },
    );
    return {
      result,
      descriptor,
      weightParts,
      fetchedWeightParts: (): string[] =>
        rig.mock.paths.filter((path) => weightParts.includes(path)),
      paths: rig.mock.paths,
    };
  };

  it("fromPretrained refuses a model without expert_linear after the descriptor, with no weight part fetched", async () => {
    const loaded = await load({ expertCache: { slots: 64 } });
    assert(loaded.weightParts.length > 0, "the fixture has no weight part (the case is vacuous)");
    await assertRejects(
      () => loaded.result,
      ModelInputError,
      "expertCache needs a mixture-of-experts model",
    );
    assert(loaded.paths.includes(loaded.descriptor), "admission did not read the descriptor");
    assertEquals(loaded.fetchedWeightParts(), [], "weight parts fetched before the refusal");
  });

  it("control: without the option the same load fetches the weight part and fails later", async () => {
    const loaded = await load({});
    const error = await assertRejects(() => loaded.result, Error);
    assert(error.message.includes("asset 'tokenizer' is missing"), error.message);
    assert(loaded.fetchedWeightParts().length > 0, "the control fetched no weight part");
  });
});
