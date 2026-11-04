/**
 * The check against the runtime support table (`assertRuntimeSupport`) — the capability check that
 * `PreparedModel` runs **before fetching a single weight block** (the two-stage boundary of ADR
 * 0070 decision 5).
 *
 * What this file pins is the design claim itself, "unsupported items are **all enumerated** and
 * fail at once": if the count (deduplicated per declaration), the axes (op / semantic dtype / attrs
 * / storage) or the per-slot accepted sets break, model authors are back to a loop where filling in
 * the support table only reveals the next item.
 *
 * The graph must be **merged** (`IrGraph`), so it is built with `mergedGraph` from the supply side
 * of the in-memory container (the same single path as `krm` — the source is not the subject here,
 * so no container is written).
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { mergedGraph } from "../src/format/container/bind.ts";
import type { CodecLayout } from "../src/format/container/codecs.ts";
import type { IrDtype, IrGraph } from "../src/format/ir.ts";
import { RUNTIME_SUPPORT } from "../src/ops.ts";
import {
  assertRuntimeSupport,
  type OpSupport,
  type RuntimeSupport,
  RuntimeSupportError,
} from "../src/ops/support.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import {
  baseDeclaration,
  baseTensors,
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  memoryModel,
} from "./helpers/model-fixture.ts";

/**
 * The minimal support table with the same shape as M0 (f32 only, no attrs, binary, single output).
 */
const f32Only: OpSupport = {
  dtypes: new Set<IrDtype>(["f32"]),
  slotDtypes: [new Set<IrDtype>(["f32"]), new Set<IrDtype>(["f32"])],
  outDtypes: [new Set<IrDtype>(["f32"])],
  attrKeys: new Set<string>(),
};

const M0_SUPPORT: RuntimeSupport = {
  ops: new Map([
    ["matmul", f32Only],
    ["add", f32Only],
  ]),
  storage: new Set<CodecLayout>(["f32"]),
  io: new Set<IrDtype>(["f32"]),
};

/** Declaration + supply → the merged graph (the form `assertRuntimeSupport` takes). */
const graphOf = (
  declaration: DeclarationJson,
  tensors: readonly TensorInput[] = baseTensors(),
): IrGraph => mergedGraph(memoryModel(declaration, tensors).graphs[GRAPH_NAME], GRAPH_NAME);

/** A declaration without weights (cases that ask only about the graph side, like gather / topk). */
const weightless = (): DeclarationJson => {
  const graph = baseDeclaration();
  graph.initializers = {};
  return graph;
};

describe("assertRuntimeSupport: the op axis", () => {
  it("enumerates unsupported ops and fails", () => {
    const graph = baseDeclaration();
    graph.requires.ops = ["gelu", "matmul", "tanh"];
    graph.nodes[1] = { op: "gelu", ins: ["h"], outs: ["y"], attrs: {} };
    graph.nodes.push({ op: "tanh", ins: ["y"], outs: ["z"], attrs: {} });
    graph.values["z"] = { dtype: "f32", shape: ["T", 3] };

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph), M0_SUPPORT),
      RuntimeSupportError,
      "unsupported op (2)",
    );
    assertEquals(error.message.includes("gelu, tanh"), true, error.message);
  });

  // Matching by op name alone gives "in the support table but fails at run time" (recon §3-9).
  it("enumerates, per declaration, semantic dtypes a supported op cannot run", () => {
    const graph = baseDeclaration();
    graph.inputs = [{ name: "x", dtype: "i32", shape: ["T", 4] }];
    graph.values["h"] = { dtype: "bool", shape: ["T", 3] };

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph), M0_SUPPORT),
      RuntimeSupportError,
      // The count is "the number of declarations to fix" — h appears twice (matmul outs and add
      // ins) but counts once
      "unsupported semantic dtype (2)",
    );
    assertEquals(error.message.includes("value 'x': i32"), true, error.message);
    assertEquals(error.message.includes("value 'h': bool"), true, error.message);
  });

  // An aliased input (a node that takes the same value twice) does not inflate the count.
  it("does not enumerate a dtype violation of the same declaration twice", () => {
    const graph = baseDeclaration();
    graph.values["h"] = { dtype: "bool", shape: ["T", 3] };
    graph.values["y"] = { dtype: "f32", shape: ["T", 3] };
    graph.nodes[1] = { op: "add", ins: ["h", "h"], outs: ["y"], attrs: {} };

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph), M0_SUPPORT),
      RuntimeSupportError,
      "unsupported semantic dtype (1)",
    );
    assertEquals(error.message.includes("value 'h': bool"), true, error.message);
  });

  // Matching only from the nodes lets a dtype violation of an input no node consumes slip through
  // the check (the executor transfers every graph.inputs, so the constraints of the transfer layer
  // are real whether or not the input is used).
  it("also enumerates a dtype violation of an input no node uses", () => {
    const graph = baseDeclaration();
    graph.inputs.push({ name: "z", dtype: "i32", shape: ["T"] });

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph), M0_SUPPORT),
      RuntimeSupportError,
      "unsupported semantic dtype (1)",
    );
    assertEquals(error.message.includes("value 'z': i32"), true, error.message);
  });
});

describe("assertRuntimeSupport: per-slot accepted sets", () => {
  // If the per-slot dtype contracts (gather / embedding / masked_fill) are matched against the
  // **union** of the accepted sets, "values and indices passed the wrong way round" is in the union
  // either way and slips through the enumerating check. Falling through to the contract check
  // (plan.ts), which stops at one item at a time, breaks the check's intent of "enumerate every
  // unsupported item and show them at once".
  it("enumerates a perSlot op whose slots are mixed up", () => {
    const graph = weightless();
    graph.requires = { ops: ["gather"] };
    // Values f32 and indices i32 passed **the wrong way round**. Matching against the union ({f32,
    // i32}) alone lets both through.
    graph.inputs = [
      { name: "src", dtype: "i32", shape: ["T", 4] },
      { name: "idx", dtype: "f32", shape: ["T", 3] },
    ];
    graph.outputs = ["y"];
    graph.values = { y: { dtype: "f32", shape: ["T", 3] } };
    graph.nodes = [{ op: "gather", ins: ["src", "idx"], outs: ["y"], attrs: {} }];

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph, []), RUNTIME_SUPPORT),
      RuntimeSupportError,
      "unsupported semantic dtype (2)",
    );
    assertEquals(error.message.includes("value 'src': i32"), true, error.message);
    assertEquals(error.message.includes("value 'idx': f32"), true, error.message);
  });

  // The output must also have the same type as slot 0 (the values side) to run — seen through the
  // union, gather's i32 output would pass.
  it("checks a perSlot op's output dtype against the accepted set of the values side", () => {
    const graph = weightless();
    graph.requires = { ops: ["gather"] };
    graph.inputs = [
      { name: "src", dtype: "f32", shape: ["T", 4] },
      { name: "idx", dtype: "i32", shape: ["T", 3] },
    ];
    graph.outputs = ["y"];
    graph.values = { y: { dtype: "i32", shape: ["T", 3] } };
    graph.nodes = [{ op: "gather", ins: ["src", "idx"], outs: ["y"], attrs: {} }];

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph, []), RUNTIME_SUPPORT),
      RuntimeSupportError,
      "unsupported semantic dtype (1)",
    );
    assertEquals(error.message.includes("value 'y': i32"), true, error.message);
  });

  // The output declarations of a multi-output op (topk) **swapped** between slots. With a
  // regression that checks every output against slot 0's accepted set (= an implementation that
  // collapsed the per-output-slot columns), both the f32 of the values side and the i32 of the
  // indices side are judged against "slot 0's set", and only one of them is enumerated (or both
  // slip through).
  // MUST: keep this **the same shape** as
  // `test_swapped_output_slots_of_a_multi_output_op_are_enumerated` in
  // tools/exporter/tests/test_verify.py (detecting the same regression on both sides is this
  // check's symmetry).
  it("enumerates both declarations when a multi-output op's output slots are mixed up", () => {
    const graph = weightless();
    graph.requires = { ops: ["topk"] };
    graph.inputs = [{ name: "x", dtype: "f32", shape: ["T", 4] }];
    graph.outputs = ["v", "i"];
    // Slot 0 is values (f32) and slot 1 is indices (i32), so **both** of these are unsupported
    graph.values = {
      v: { dtype: "i32", shape: ["T", 2] },
      i: { dtype: "f32", shape: ["T", 2] },
    };
    graph.nodes = [{ op: "topk", ins: ["x"], outs: ["v", "i"], attrs: { k: 2 } }];

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph, []), RUNTIME_SUPPORT),
      RuntimeSupportError,
      "unsupported semantic dtype (2)",
    );
    assertEquals(error.message.includes("value 'v': i32"), true, error.message);
    assertEquals(error.message.includes("value 'i': f32"), true, error.message);
  });
});

describe("assertRuntimeSupport: the attrs and storage axes", () => {
  it("enumerates, per node, unimplemented attrs on a supported op", () => {
    const graph = baseDeclaration();
    graph.nodes[1].attrs = { alpha: 1, approximate: "tanh" };

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph), M0_SUPPORT),
      RuntimeSupportError,
      "unimplemented attrs (1)",
    );
    assertEquals(error.message.includes("nodes[1] (add): alpha, approximate"), true, error.message);
  });

  it("enumerates unsupported storage with the initializer name", () => {
    // The semantic dtype stays f32; only the storage codec is f16 / bf16 (the expansion path is
    // seen through layout).
    const tensors: readonly TensorInput[] = [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes: new Uint8Array(new ArrayBuffer(24)),
        encoding: { codec: "f16" },
      },
      {
        graph: GRAPH_NAME,
        initializer: "b",
        bytes: new Uint8Array(new ArrayBuffer(6)),
        encoding: { codec: "bf16" },
      },
    ];

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(baseDeclaration(), tensors), M0_SUPPORT),
      RuntimeSupportError,
      "Missing runtime capabilities",
    );
    assertEquals(error.message.includes("unsupported storage 'bf16' (1): b"), true, error.message);
    assertEquals(error.message.includes("unsupported storage 'f16' (1): w"), true, error.message);
  });

  it("passes storage listed in the support table (the control showing the check is not a tautology)", () => {
    const tensors: readonly TensorInput[] = [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes: new Uint8Array(new ArrayBuffer(24)),
        encoding: { codec: "f16" },
      },
      {
        graph: GRAPH_NAME,
        initializer: "b",
        bytes: f32Bytes([1, 2, 3]),
        encoding: { codec: "f32" },
      },
    ];
    assertRuntimeSupport(graphOf(baseDeclaration(), tensors), {
      ...M0_SUPPORT,
      storage: new Set<CodecLayout>(["f32", "f16"]),
    });
  });
});
