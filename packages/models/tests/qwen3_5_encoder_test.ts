// The Qwen3.5 text encoder without a GPU (src/qwen3_5/graph-contract.ts, src/qwen3_5/encoder.ts —
// ADR 0137): the graph-declaration contract on a declaration-only graph shaped like the recipe's
// export, and the encoder's lifecycle over a stand-in Session (serialisation, dispose, cancellation,
// the checks before the run). Numerical correctness is the GPU e2e's (e2e_qwen3_5_encoder_test.ts).

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import type { RunInputs, RunOutputs, SessionOptions, Tensor } from "@karume/runtime";
import type { GraphOwner } from "../src/hub/components.ts";
import {
  QWEN3_5_TEXT_ENCODER_SESSION_POLICY,
  type Qwen3_5Session,
  type Qwen3_5TextComponent,
  Qwen3_5TextEncoder,
} from "../src/qwen3_5/encoder.ts";
import { qwen3_5GraphContract } from "../src/qwen3_5/graph-contract.ts";
import { QWEN3_5_ROTARY_DIM } from "../src/qwen3_5/rope.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";

type Graph = GraphOwner["graph"];

const WIDTH = 8;
const VOCABULARY = 16;
const OUTPUT = "rms_norm_default_104";

/**
 * A declaration-only graph with the recipe's interface (`tools/export-recipes/qwen3_5/patch.py`):
 * four inputs over one symbol T, one embedding of `input_ids`, i8 linear weights next to f32 norms,
 * one f32 output `[1, T, W]` (the real container's output is named as here).
 */
const recipeGraph = (): Graph => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: [] },
  symbols: ["T"],
  inputs: [
    { name: "input_ids", dtype: "i32", shape: [1, "T"] },
    { name: "positions", dtype: "f32", shape: ["T"] },
    { name: "rope_cos", dtype: "f32", shape: [1, "T", QWEN3_5_ROTARY_DIM] },
    { name: "rope_sin", dtype: "f32", shape: [1, "T", QWEN3_5_ROTARY_DIM] },
  ],
  outputs: [OUTPUT],
  initializers: {
    "embed_tokens.weight": { storage: { codec: "int8-sym" } },
    "layers.0.mlp.down_proj.weight": { storage: { codec: "int8-sym" } },
    "norm.weight": { storage: { codec: "f32" } },
  },
  values: {
    "embed_tokens.weight": { dtype: "f32", shape: [VOCABULARY, WIDTH] },
    embedding: { dtype: "f32", shape: [1, "T", WIDTH] },
    [OUTPUT]: { dtype: "f32", shape: [1, "T", WIDTH] },
  },
  states: {},
  nodes: [
    {
      op: "embedding",
      ins: ["embed_tokens.weight", "input_ids"],
      outs: ["embedding"],
      attrs: { padding_idx: -1 },
      states: {},
    },
  ],
});

const ownerOf = (graph: Graph): GraphOwner => ({ graph });

const contractOf = (edit: (graph: Graph) => Graph) =>
  qwen3_5GraphContract(ownerOf(edit(recipeGraph())), "test");

const withInputs = (graph: Graph, inputs: Graph["inputs"]): Graph => ({ ...graph, inputs });

const replaceInput = (graph: Graph, name: string, spec: Graph["inputs"][number]): Graph =>
  withInputs(
    graph,
    graph.inputs.map((input) => (input.name === name ? spec : input)),
  );

describe("qwen3_5GraphContract: the recipe's declaration", () => {
  it("accepts it and reads the output name, the width and the embedding's row count", () => {
    assertEquals(qwen3_5GraphContract(ownerOf(recipeGraph()), "test"), {
      output: OUTPUT,
      hiddenSize: WIDTH,
      vocabularySize: VOCABULARY,
    });
  });
});

describe("qwen3_5GraphContract: inputs other than the host builds fail loudly", () => {
  it("a missing input, naming the declared ones", () => {
    assertThrows(
      () => contractOf((graph) => withInputs(graph, graph.inputs.slice(0, 3))),
      Error,
      "graph input 'rope_sin' is missing (declared: input_ids / positions / rope_cos)",
    );
  });

  it("an input the host does not build", () => {
    assertThrows(
      () =>
        contractOf((graph) =>
          withInputs(graph, [
            ...graph.inputs,
            { name: "attention_mask", dtype: "f32", shape: ["T"] },
          ]),
        ),
      Error,
      "has 5 graph inputs",
    );
  });

  it("a static length on input_ids", () => {
    assertThrows(
      () =>
        contractOf((graph) =>
          replaceInput(graph, "input_ids", { name: "input_ids", dtype: "i32", shape: [1, 64] }),
        ),
      Error,
      "axis 1 is not a symbolic dimension (64)",
    );
  });

  it("positions on another symbol than input_ids", () => {
    assertThrows(
      () =>
        contractOf((graph) =>
          replaceInput(graph, "positions", { name: "positions", dtype: "f32", shape: ["S"] }),
        ),
      Error,
      "'positions' shape [S] differs from the host's [T]",
    );
  });

  it("RoPE rows of another width than the pinned rotary width", () => {
    assertThrows(
      () =>
        contractOf((graph) =>
          replaceInput(graph, "rope_cos", { name: "rope_cos", dtype: "f32", shape: [1, "T", 128] }),
        ),
      Error,
      "'rope_cos' shape [1, T, 128] differs from the host's [1, T, 64]",
    );
  });

  it("an input of another dtype", () => {
    assertThrows(
      () =>
        contractOf((graph) =>
          replaceInput(graph, "positions", { name: "positions", dtype: "i32", shape: ["T"] }),
        ),
      Error,
      "'positions' has dtype i32, not f32",
    );
  });
});

describe("qwen3_5GraphContract: an output other than [1, T, W] f32 fails loudly", () => {
  it("two outputs", () => {
    assertThrows(
      () => contractOf((graph) => ({ ...graph, outputs: [OUTPUT, "embedding"] })),
      Error,
      "has 2 outputs",
    );
  });

  it("a symbolic width", () => {
    assertThrows(
      () =>
        contractOf((graph) => ({
          ...graph,
          values: { ...graph.values, [OUTPUT]: { dtype: "f32", shape: [1, "T", "W"] } },
        })),
      Error,
      "axis 2 is not a static positive integer",
    );
  });

  it("rows on another symbol than the inputs", () => {
    assertThrows(
      () =>
        contractOf((graph) => ({
          ...graph,
          values: { ...graph.values, [OUTPUT]: { dtype: "f32", shape: [1, "S", WIDTH] } },
        })),
      Error,
      `output '${OUTPUT}' shape [1, S, 8] differs from the host's [1, T, 8]`,
    );
  });
});

describe("qwen3_5GraphContract: the id range needs one embedding with a static table", () => {
  it("no embedding reads input_ids", () => {
    assertThrows(
      () => contractOf((graph) => ({ ...graph, nodes: [] })),
      Error,
      "has 0 embedding nodes that read 'input_ids'",
    );
  });

  it("a symbolic row count", () => {
    assertThrows(
      () =>
        contractOf((graph) => ({
          ...graph,
          values: { ...graph.values, "embed_tokens.weight": { dtype: "f32", shape: ["V", WIDTH] } },
        })),
      Error,
      "embedding table 'embed_tokens.weight' has shape [V, 8]",
    );
  });
});

describe("qwen3_5GraphContract: only the verified storage (i8 per-channel + f32) is accepted", () => {
  const withStorage = (graph: Graph, name: string, initializer: Graph["initializers"][string]) => ({
    ...graph,
    initializers: { ...graph.initializers, [name]: initializer },
  });

  it("BF16 weights", () => {
    assertThrows(
      () =>
        contractOf((graph) =>
          withStorage(graph, "layers.0.mlp.down_proj.weight", { storage: { codec: "bf16" } }),
        ),
      Error,
      "'layers.0.mlp.down_proj.weight' has storage bf16, which is not accepted",
    );
  });

  it("a container without any i8 weight", () => {
    assertThrows(
      () =>
        contractOf((graph) =>
          withStorage(
            withStorage(graph, "embed_tokens.weight", { storage: { codec: "f32" } }),
            "layers.0.mlp.down_proj.weight",
            { storage: { codec: "f32" } },
          ),
        ),
      Error,
      "has no i8 weights at all",
    );
  });

  it("a shared declaration (borrowed weights)", () => {
    assertThrows(
      () => contractOf((graph) => withStorage(graph, "norm.weight", { shared: true })),
      Error,
      "'norm.weight' is a shared declaration",
    );
  });
});

// ---- the encoder over a stand-in Session ----------------------------------------------------

/** A promise whose settlement the test drives. */
const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
};

/** The host tensor a run received as `input_ids` (the encoder passes host tensors only). */
const idsInput = (inputs: RunInputs): Tensor => {
  const input = inputs["input_ids"];
  if (input === undefined || !("data" in input)) {
    throw new Error("the run received no host tensor as input_ids");
  }
  return input;
};

type Stand = {
  readonly component: Qwen3_5TextComponent;
  /** Every `run`'s inputs, in call order. */
  readonly runs: RunInputs[];
  /** Calls of `createSession` with their options. */
  readonly sessions: SessionOptions[];
  disposed: number;
  /** The most runs in flight at once. */
  maxInFlight: number;
  /** When set, each run waits for it before answering. */
  gate?: Promise<void>;
  /** The output value written at row 1, column 2 (to inject a non-finite output). */
  poison?: number;
};

/** A component whose Session answers each run with `[1, T, W]` = the 1-based running index. */
const standIn = (graph: Graph = recipeGraph()): Stand => {
  let inFlight = 0;
  const stand: Stand = {
    runs: [],
    sessions: [],
    disposed: 0,
    maxInFlight: 0,
    component: {
      graph,
      createSession: async (_gpu, options = {}) => {
        stand.sessions.push(options);
        const session: Qwen3_5Session = {
          run: async (inputs): Promise<RunOutputs> => {
            stand.runs.push(inputs);
            inFlight += 1;
            stand.maxInFlight = Math.max(stand.maxInFlight, inFlight);
            try {
              await stand.gate;
              const tokens = idsInput(inputs).shape[1] ?? 0;
              const data = Float32Array.from({ length: tokens * WIDTH }, (_, index) => index + 1);
              if (stand.poison !== undefined) data[1 * WIDTH + 2] = stand.poison;
              return { [OUTPUT]: { dtype: "f32", shape: [1, tokens, WIDTH], data } };
            } finally {
              inFlight -= 1;
            }
          },
          diagnostics: () => {
            throw new Error("the stand-in Session has no diagnostics");
          },
          dispose: async () => {
            stand.disposed += 1;
          },
        };
        return session;
      },
    },
  };
  return stand;
};

const gpu = fakeGpuContext(fakeDevice());
const idsOf = (...values: number[]): Int32Array<ArrayBuffer> => Int32Array.from(values);

describe("Qwen3_5TextEncoder.open", () => {
  it("builds one Session with no execution knob set", async () => {
    const stand = standIn();
    const encoder = await Qwen3_5TextEncoder.open(stand.component, gpu);
    assertEquals(stand.sessions, [{}]);
    assertEquals(encoder.hiddenSize, WIDTH);
  });

  it("rejects a graph the host cannot feed before building a Session", async () => {
    const stand = standIn(withInputs(recipeGraph(), recipeGraph().inputs.slice(1)));
    await assertRejects(
      () => Qwen3_5TextEncoder.open(stand.component, gpu),
      Error,
      "Qwen3_5TextEncoder: text_encoder graph input 'input_ids' is missing",
    );
    assertEquals(stand.sessions, []);
  });

  it("rejects every session declaration (no knob is verified) before building a Session", async () => {
    assert(Object.values(QWEN3_5_TEXT_ENCODER_SESSION_POLICY).every((accepted) => !accepted));
    const stand = standIn();
    await assertRejects(
      () => Qwen3_5TextEncoder.open(stand.component, gpu, { session: { linearCompute: "a8" } }),
      Error,
      "session.linearCompute is not supported",
    );
    assertEquals(stand.sessions, []);
  });
});

describe("Qwen3_5TextEncoder.encode", () => {
  it("runs the four host inputs once and returns the rows [T, W] of the graph output", async () => {
    const stand = standIn();
    const encoder = await Qwen3_5TextEncoder.open(stand.component, gpu);
    const ids = idsOf(3, 1, 4);

    const rows = await encoder.encode(ids);

    assertEquals(stand.runs.length, 1);
    assertEquals(Object.keys(stand.runs[0]).sort(), [
      "input_ids",
      "positions",
      "rope_cos",
      "rope_sin",
    ]);
    assertStrictEquals(idsInput(stand.runs[0]).data, ids);
    assertEquals(rows.length, 3 * WIDTH);
    assertEquals([rows[0], rows[3 * WIDTH - 1]], [1, 3 * WIDTH]);
  });

  it("rejects an id outside the embedding before the run", async () => {
    const stand = standIn();
    const encoder = await Qwen3_5TextEncoder.open(stand.component, gpu);
    await assertRejects(
      () => encoder.encode(idsOf(1, VOCABULARY)),
      Error,
      `id ${VOCABULARY} at index 1`,
    );
    await assertRejects(() => encoder.encode(idsOf()), Error, "length 0 is outside 1..4096");
    assertEquals(stand.runs.length, 0);
  });

  it("rejects non-finite rows, naming the row and column", async () => {
    const stand = standIn();
    stand.poison = Number.NaN;
    const encoder = await Qwen3_5TextEncoder.open(stand.component, gpu);
    await assertRejects(
      () => encoder.encode(idsOf(1, 2, 3)),
      Error,
      "output (3 tokens) row 1, column 2 is non-finite (NaN)",
    );
  });

  it("runs concurrent calls one at a time, in call order", async () => {
    const stand = standIn();
    const gate = deferred<void>();
    stand.gate = gate.promise;
    const encoder = await Qwen3_5TextEncoder.open(stand.component, gpu);

    const first = encoder.encode(idsOf(1));
    const second = encoder.encode(idsOf(2, 2));
    gate.resolve();
    const [a, b] = await Promise.all([first, second]);

    assertEquals(stand.maxInFlight, 1);
    assertEquals(
      stand.runs.map((inputs) => idsInput(inputs).shape),
      [
        [1, 1],
        [1, 2],
      ],
    );
    assertEquals([a.length, b.length], [WIDTH, 2 * WIDTH]);
  });

  it("throws an aborted signal's reason as is, without running", async () => {
    const stand = standIn();
    const encoder = await Qwen3_5TextEncoder.open(stand.component, gpu);
    const controller = new AbortController();
    const reason = new Error("stop");
    controller.abort(reason);
    const caught = await encoder
      .encode(idsOf(1), { signal: controller.signal })
      .catch((error: unknown) => error);
    assertStrictEquals(caught, reason);
    assertEquals(stand.runs.length, 0);
    assertEquals((await encoder.encode(idsOf(1))).length, WIDTH);
  });
});

describe("Qwen3_5TextEncoder.dispose", () => {
  it("waits for the encoding in flight, folds the Session once, and refuses later encodings", async () => {
    const stand = standIn();
    const gate = deferred<void>();
    stand.gate = gate.promise;
    const encoder = await Qwen3_5TextEncoder.open(stand.component, gpu);

    const inFlight = encoder.encode(idsOf(1, 2));
    const disposal = encoder.dispose();
    assertEquals(stand.disposed, 0, "folded before the run settled");
    gate.resolve();
    assertEquals((await inFlight).length, 2 * WIDTH);
    await disposal;

    assertEquals(stand.disposed, 1);
    assertStrictEquals(encoder.dispose(), disposal);
    await assertRejects(() => encoder.encode(idsOf(1)), Error, "cannot encode after dispose");
    assertEquals(stand.disposed, 1);
  });
});
