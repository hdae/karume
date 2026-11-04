// Cross-check of the 4 paths that consume the companion scale shape (`groupScaleShape` —
// src/format/container/codecs.ts). Uses no GPU at all.
//
// Four paths assume the scale shape — the merge layer (scale block length), the residency planner
// (scale bytes), the scale shape handed from the container to Session construction
// (`containerBatches`) and the check in the CPU decode `decodeI4`. If any one of them has its own
// formula, the accepted shape and the shape the decode reads silently disagree (a silent wrong
// value where the group scale is handed out as one value per channel).
//
// Each table row is checked in 2 steps:
// ① The shared function returns the hand-written shape (pins the formula itself — the expected
//    value is written by hand; deriving it from the function would be a tautology).
// ② What the 4 paths accept / count / shape matches that function's value (changing the formula
//    moves all 4 paths together, and reverting one path to a different formula turns its row red).
// Row axis 1 (`[Cin,Cout,K]` of conv_transpose1d) and the degenerate row length 0 are always in the
// table, as the shapes on which a local formula most easily breaks.

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { mergedGraph } from "../src/format/container/bind.ts";
import { type CodecName, groupScaleShape, payloadBytes } from "../src/format/container/codecs.ts";
import { ContainerFormatError } from "../src/format/container/header.ts";
import { decodeI4, I4Error } from "../src/format/i4.ts";
import { containerBatches } from "../src/runtime/executor.ts";
import type { ReadyScale } from "../src/runtime/session-build.ts";
import { planWeightResidency } from "../src/runtime/weight-residency.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { type DeclarationJson, GRAPH_NAME, memoryModel } from "./helpers/model-fixture.ts";

const numel = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

const zeros = (length: number): Uint8Array<ArrayBuffer> => new Uint8Array(new ArrayBuffer(length));

/** An op consuming the single weight `w` (what the paths treat as consuming a weight slot). */
type Consumer = "linear" | "conv1d" | "conv_transpose1d";

type Case = {
  readonly label: string;
  readonly consumer: Consumer;
  readonly codec: CodecName;
  readonly shape: readonly number[];
  readonly rowAxis: 0 | 1;
  readonly groupSize: number;
  /** The hand-written expected shape (not derived from the shared function). */
  readonly want: readonly [number, number];
};

const CASES: readonly Case[] = [
  {
    label: "i4 linear [8,64], group 16",
    consumer: "linear",
    codec: "int4-sym-g",
    shape: [8, 64],
    rowAxis: 0,
    groupSize: 16,
    want: [8, 4],
  },
  {
    label: "i4 linear [8,64], group 64 (row length = group length)",
    consumer: "linear",
    codec: "int4-sym-g",
    shape: [8, 64],
    rowAxis: 0,
    groupSize: 64,
    want: [8, 1],
  },
  {
    label: "i4 conv1d [4,16,4], group 16 (row length flattened to Cin·K = 64)",
    consumer: "conv1d",
    codec: "int4-sym-g",
    shape: [4, 16, 4],
    rowAxis: 0,
    groupSize: 16,
    want: [4, 4],
  },
  {
    label: "i4 linear [4,0], group 16 (the degenerate row length 0 has group count 1)",
    consumer: "linear",
    codec: "int4-sym-g",
    shape: [4, 0],
    rowAxis: 0,
    groupSize: 16,
    want: [4, 1],
  },
  {
    label: "i8 linear [6,16], per-channel",
    consumer: "linear",
    codec: "int8-sym",
    shape: [6, 16],
    rowAxis: 0,
    groupSize: 16,
    want: [6, 1],
  },
  {
    label: "i8 conv_transpose1d [4,6,2], row axis 1 (rows are Cout = 6, row length 8)",
    consumer: "conv_transpose1d",
    codec: "int8-sym",
    shape: [4, 6, 2],
    rowAxis: 1,
    groupSize: 8,
    want: [6, 1],
  },
  {
    // The per-channel element count is `numel / groupSize` whatever the row axis, so on the paths
    // that only look at bytes (merge layer, residency planner) a mixed-up axis shows in the count
    // only for a row length of 0 (reading it as axis 0 gives 0 rows).
    label:
      "i8 conv_transpose1d [0,6,2], row axis 1, row length 0 (read as axis 0: 0 scale elements)",
    consumer: "conv_transpose1d",
    codec: "int8-sym",
    shape: [0, 6, 2],
    rowAxis: 1,
    groupSize: 1,
    want: [6, 1],
  },
];

/** A graph consuming `w` with one op (bias `b` is f32). */
const declarationOf = (testCase: Case): DeclarationJson => {
  const { consumer, shape } = testCase;
  const common = {
    format: "karume-ir",
    version: 2,
    requires: { ops: [consumer] },
    symbols: [],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
  };
  if (consumer === "linear") {
    const [rows, width] = shape;
    return {
      ...common,
      inputs: [{ name: "x", dtype: "f32", shape: [2, width] }],
      values: {
        w: { dtype: "f32", shape: [...shape] },
        b: { dtype: "f32", shape: [rows] },
        y: { dtype: "f32", shape: [2, rows] },
      },
      nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
    };
  }
  if (consumer === "conv1d") {
    const [channelsOut, channelsIn, kernel] = shape;
    const length = 6;
    return {
      ...common,
      inputs: [{ name: "x", dtype: "f32", shape: [1, channelsIn, length] }],
      values: {
        w: { dtype: "f32", shape: [...shape] },
        b: { dtype: "f32", shape: [channelsOut] },
        y: { dtype: "f32", shape: [1, channelsOut, length - kernel + 1] },
      },
      nodes: [
        {
          op: "conv1d",
          ins: ["x", "w", "b"],
          outs: ["y"],
          attrs: { stride: 1, padding: 0, dilation: 1, groups: 1 },
        },
      ],
    };
  }
  const [channelsIn, channelsOut, kernel] = shape;
  const length = 5;
  return {
    ...common,
    inputs: [{ name: "x", dtype: "f32", shape: [1, channelsIn, length] }],
    values: {
      w: { dtype: "f32", shape: [...shape] },
      b: { dtype: "f32", shape: [channelsOut] },
      y: { dtype: "f32", shape: [1, channelsOut, length - 1 + kernel] },
    },
    nodes: [
      {
        op: "conv_transpose1d",
        ins: ["x", "w", "b"],
        outs: ["y"],
        attrs: { stride: 1, padding: 0 },
      },
    ],
  };
};

/**
 * Supplies with `scaleElements` elements of scale for `w` (zero-filled, the contents are not read).
 */
const tensorsOf = (testCase: Case, scaleElements: number): readonly TensorInput[] => {
  const { codec, shape, rowAxis, groupSize } = testCase;
  const rows = shape[rowAxis];
  return [
    {
      graph: GRAPH_NAME,
      initializer: "w",
      bytes: zeros(payloadBytes(codec, numel(shape), "w")),
      encoding: {
        codec,
        ...(rowAxis === 0 ? {} : { rowAxis }),
        groupSize,
        scale: { bytes: zeros(scaleElements * 4), dtype: "f32" },
      },
    },
    {
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: zeros(rows * 4),
      encoding: { codec: "f32" },
    },
  ];
};

/**
 * The scale that `containerBatches` carries along with the `w` item (the GPU-free entry of Session
 * construction).
 */
const yieldedScale = async (testCase: Case): Promise<ReadyScale | undefined> => {
  const expected = groupScaleShape(testCase.shape, testCase.rowAxis, testCase.groupSize);
  const opened = memoryModel(declarationOf(testCase), tensorsOf(testCase, numel(expected)));
  for await (const batch of containerBatches(opened, GRAPH_NAME)) {
    for await (const item of batch.items) {
      if (item.name === "w") return item.scale;
    }
  }
  return undefined;
};

describe("companion scale shape: the 4 paths follow the single groupScaleShape", () => {
  for (const testCase of CASES) {
    const { shape, rowAxis, groupSize } = testCase;
    const expected = groupScaleShape(shape, rowAxis, groupSize);

    describe(testCase.label, () => {
      it("the shared function returns the hand-written shape", () => {
        assertEquals(expected, testCase.want);
      });

      it("the merge layer accepts a scale block of that shape and rejects one 1 element longer", () => {
        const bound = memoryModel(declarationOf(testCase), tensorsOf(testCase, numel(expected)))
          .graphs[GRAPH_NAME];
        assertEquals(bound.supplies.get("w")?.scale?.payloadBytes, numel(expected) * 4);
        assertThrows(
          () => memoryModel(declarationOf(testCase), tensorsOf(testCase, numel(expected) + 1)),
          ContainerFormatError,
          // The type alone stays green if the scale length check dies but another rule rejects —
          // look at the diagnostic specific to the rule.
          `initializer 'w' scale: block 'model/w#scale' length ${(numel(expected) + 1) * 4} is not ` +
            `payload ${numel(expected) * 4} bytes`,
        );
      });

      it("the residency planner's scale bytes are that shape's element count × 4", () => {
        const bound = memoryModel(declarationOf(testCase), tensorsOf(testCase, numel(expected)))
          .graphs[GRAPH_NAME];
        const seat = planWeightResidency(mergedGraph(bound, GRAPH_NAME)).get("w");
        // Every weight in the table, including the i4 one with row length 0, lands in a class
        // resident in compressed form (the class decision belongs to runtime_weight_residency_test
        // — here only the scale bytes are checked).
        if (seat?.seat !== "i4" && seat?.seat !== "i8") {
          throw new Error(`w is not in a compressed residency class: ${seat?.seat}`);
        }
        assertEquals(seat.scaleBytes, numel(expected) * 4);
      });

      it("the scale shape handed from the container to Session construction is that shape", async () => {
        const scale = await yieldedScale(testCase);
        assertEquals(scale?.shape, expected);
        assertEquals(scale?.bytes.byteLength, numel(expected) * 4);
      });

      if (testCase.codec !== "int4-sym-g") return;

      it("decodeI4 accepts that shape and rejects one whose group count is off by 1", () => {
        const bytes = zeros(numel(shape) / 2);
        const decoded = decodeI4(
          bytes,
          shape,
          new Float32Array(numel(expected)),
          expected,
          groupSize,
        );
        assertEquals(decoded.length, numel(shape));
        const wider: readonly [number, number] = [expected[0], expected[1] + 1];
        assertThrows(
          () => decodeI4(bytes, shape, new Float32Array(numel(wider)), wider, groupSize),
          I4Error,
          "is not the group shape",
        );
      });
    });
  }
});
