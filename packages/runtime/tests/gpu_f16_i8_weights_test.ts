// Weight storages shared by the five fused kernels: f16 (ADR 0018), bf16 (ADR 0131) and per-channel
// i8 (ADR 0019). The storages mirror each other test-for-test, so the mirrored tests run once per
// entry of STORAGES (each keeps its own title); storage-specific parts (host decoding, the exhaustive
// unpack check, the i8 channel axis and rowAxis admission) stay separate. bf16 shares f16's packing
// (two elements per u32 word, the same odd-length traps) and differs only in the unpack (a bitcast).
//
// The f16 storage execution path (ADR 0018) — end-to-end checks of eligibility, the load path, the
// kernel variants and the diagnostics.
// The i8 (per-channel symmetric int8) storage execution path (ADR 0019) — end-to-end checks of
// load-time validation, the shared eligibility decision, the kernel variants and the diagnostics.
//
// Both paths introduced in this iteration are exercised:
//   eligible (consumed only in the weight slot of an op listed in WEIGHT_SLOTS) -> resident on the
//     GPU as raw bytes, dequant inside the kernel (`unpack2x16float`)
//   ineligible (bias / mixed consumption / anything else) -> expanded to f32 on the CPU at load
//     (zero VRAM saving)
// (i8 dequantizes in the kernel with `unpack4xI8` + the per-channel scale.)
//
// MUST: the numeric cases make **in-features / K / Cin·Kh·Kw odd**. f16 packs 2 elements into one
// word, so with an even row length the start of a weight row always lands on a word boundary — the
// error of "choosing the pair by the index relative to the row" never shows in the values with even
// lengths (demonstrated by the fault injection of iteration Q1).
// MUST: the total element count of the weights is odd too (this exercises the path where writeBuffer
// fails validation without the zero fill of the last 2 bytes).
// MUST: the numeric cases make **the weight row length not a multiple of 4**. i8 packs 4 elements
// into one word, so with a row length that is a multiple of 4 the start of a weight row always lands
// on a word boundary — the error of "choosing the word and the lane by the index relative to the
// row" never shows in the values with such lengths (the same trap as the f16 parity MUST).
// MUST: the total element count of the weights is not a multiple of 4 either (this exercises the
// path where writeBuffer fails validation without the trailing zero fill).
// MUST: the expected values are built from **the rounded weights** (fake-quant — ADR 0006).
// Comparing with the f32 before rounding would mix quantization error with implementation error
// and create pressure to loosen the tolerance.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { type CodecName, perChannelGroupSize } from "../src/format/container/codecs.ts";
import { ContainerFormatError } from "../src/format/container/header.ts";
import type { OpenedContainer } from "../src/format/container/open.ts";
import { bf16BitsToF32, decodeBf16 } from "../src/format/bf16.ts";
import { decodeF16, f16BitsToF32 } from "../src/format/f16.ts";
import { alignI8Payload, decodeI8, I8Error } from "../src/format/i8.ts";
import type { IrGraph } from "../src/format/ir.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { GEMM_TOLERANCE } from "./helpers/op-tolerance.ts";
import { applyReferenceOp, type RefTensor, refTensor } from "../src/reference/ops.ts";
import { RUNTIME_SUPPORT, WEIGHT_SLOTS } from "../src/ops.ts";
import { assertRuntimeSupport } from "../src/ops/support.ts";
import {
  eligibleCompressedInitializers,
  ExecutionError,
  weightChannelAxes,
} from "../src/runtime/plan.ts";
import {
  createSessionFromContainer,
  prepareContainer,
  type Tensor,
} from "../src/runtime/executor.ts";
import type { EncodingInput, TensorInput } from "./helpers/container-write.ts";
import { bf16BytesFromBits, quantizeBf16 } from "./helpers/bf16.ts";
import { f16BytesFromBits, quantizeF16 } from "./helpers/f16.ts";
import { i8BytesFrom, quantizeI8 } from "./helpers/i8.ts";
import { mergeGraph, mergeTensors } from "./helpers/merged-graph.ts";
import {
  type DeclarationJson,
  f32Bytes,
  fill,
  type FilledTensor,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const SIGNED = (i: number): number => ((i % 13) - 6) * 0.75;
const POSITIVE = (i: number): number => 0.125 + (i % 17) * 0.5;

/**
 * An `IrGraph` (synchronous) merged with empty bytes of the given codec on every initializer of the
 * declaration. Eligibility looks only at the graph structure, so the actual contents are not needed
 * — only the merge layer is passed through.
 *
 * The default is `f16` (what this series looks at is the eligibility of f16 storage, so initializers
 * not named explicitly are loaded as f16 too — that bias drops out of eligibility while still f16 is
 * the very claim here).
 * `fallback` replaces that default for the other storage (the i8 run loads every unnamed
 * initializer, bias included, as int8-sym).
 */
const mergedFor = (
  declaration: DeclarationJson,
  codecs: Readonly<Record<string, CodecName>> = {},
  fallback: CodecName = "f16",
): IrGraph =>
  mergeGraph(
    declaration,
    Object.fromEntries(
      Object.keys(declaration.initializers).map((name) => [name, codecs[name] ?? fallback]),
    ),
  );

/** The declaration of a single `linear(x, w, b)`. `extra` can add consumers of w (to build mixed consumption). */
const linearDeclaration = (
  extra: DeclarationJson["nodes"] = [],
  extraValues: DeclarationJson["values"] = {},
  extraOutputs: readonly string[] = [],
): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear", ...extra.map((node) => node.op)] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 4] }],
  outputs: ["y", ...extraOutputs],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [3, 4] },
    b: { dtype: "f32", shape: [3] },
    y: { dtype: "f32", shape: [2, 3] },
    ...extraValues,
  },
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }, ...extra],
});

type WeightedCase = {
  readonly name: string;
  readonly op: string;
  /** Graph inputs (x / index). */
  readonly inputs: readonly (readonly [string, FilledTensor])[];
  readonly weight: FilledTensor;
  /** The channel axis of the per-channel scale (1 only for conv_transpose1d). */
  readonly channelAxis: number;
  /** The bias is **always f32-stored** (ADR 0006). undefined for ops without a bias, such as embedding. */
  readonly bias?: FilledTensor;
  readonly outShape: readonly number[];
  readonly attrs?: Record<string, unknown>;
};

/** A weight rounded to one storage: the bytes to load and the values the CPU reference uses. */
type QuantizedWeight = {
  readonly bytes: Uint8Array<ArrayBuffer>;
  /** The values after quantization (fake-quant — the CPU reference uses these as the weights). */
  readonly values: Float32Array<ArrayBuffer>;
  readonly encoding: EncodingInput;
  /** The per-channel scale (i8 only); it is resident next to the payload. */
  readonly scale?: Float32Array<ArrayBuffer>;
};

/** One storage under test, with the expectations that differ between the storages. */
type WeightStorageCase = {
  readonly storage: "f16" | "bf16" | "i8";
  readonly codec: CodecName;
  readonly quantize: (weight: FilledTensor, channelAxis: number) => QuantizedWeight;
  /** linear cases on the GEMM tile edges (v4 path and scalar path). */
  readonly tileCases: readonly WeightedCase[];
  /** Resident bytes of the [3,4] linear weight on the eligible path (literal oracle). */
  readonly mixedResidentBytes: number;
  /** Payload / resident bytes of the [3,7] weight in the diagnostics graph (literal oracles). */
  readonly diagnostics: { readonly payloadBytes: number; readonly residentBytes: number };
  readonly titles: {
    readonly eligibility: string;
    readonly graphOutputEligibility: string;
    readonly variants: string;
    readonly tiles: string;
    readonly poison: string;
    readonly mixed: string;
    readonly diagnostics: string;
    readonly graphOutput: string;
  };
};

// Weights whose magnitude differs per row (= output channel). With POSITIVE as is, every row's
// maximum would be the same, the scale would be the same for every channel, and an axis mix-up
// would not show in the numbers.
const rowVarying =
  (cols: number) =>
  (i: number): number =>
    POSITIVE(i) * (1 + (Math.floor(i / cols) % 7) * 0.25);

const STORAGES: readonly WeightStorageCase[] = [
  {
    storage: "f16",
    codec: "f16",
    quantize: (weight) => {
      const quantized = quantizeF16(weight.data);
      return { ...quantized, encoding: { codec: "f16" } };
    },
    tileCases: [
      {
        // k=20 (a multiple of 4, not of 16) / n=68 (1 of the 16 quads of the last tile is valid) /
        // m=65 (2 row tiles). m/n/k all have different lengths.
        name: "linear v4 [65,20] × W[68,20] + b[68]",
        op: "linear",
        inputs: [["x", fill([65, 20], SIGNED)]],
        weight: fill([68, 20], POSITIVE),
        channelAxis: 0,
        bias: fill([68], SIGNED),
        outShape: [65, 68],
      },
      {
        // k=37 (not a multiple of 2 = the detector for the parity of the flat index) / n=23 / m=70 (2 row tiles)
        name: "linear scalar [70,37] × W[23,37] + b[23]",
        op: "linear",
        inputs: [["x", fill([70, 37], SIGNED)]],
        weight: fill([23, 37], POSITIVE),
        channelAxis: 0,
        bias: fill([23], SIGNED),
        outShape: [70, 23],
      },
    ],
    // 12 elements × 2 bytes, already 4-byte aligned.
    mixedResidentBytes: 24,
    // Eligible: 21 elements × 2 bytes = 42 -> 44 with 4-byte alignment
    diagnostics: { payloadBytes: 42, residentBytes: 44 },
    titles: {
      eligibility: "eligibility admits only consumption in the weight slot of a WEIGHT_SLOTS op",
      graphOutputEligibility:
        "an initializer that became a graph output is ineligible even when consumed only in a weight slot",
      variants:
        "the 5 w=f16 variants match the CPU reference (rounded weights) (odd lengths, real GPU)",
      tiles:
        "w=f16 linear matches the CPU reference at the GEMM tile boundaries (v4 / scalar) (real GPU)",
      poison:
        "the w=f16 variants also leave not a single poison word in pool-reused buffers (full-write / real GPU)",
      mixed:
        "f16 with mixed consumption falls back to CPU expansion and gives bit-identical values to the eligible path (real GPU)",
      diagnostics:
        "the diagnostics tell eligible / ineligible / uncompressed apart by actual byte counts (real GPU)",
      graphOutput:
        "an f16 initializer made a graph output falls back to CPU expansion, and the rounded values read as f32 (real GPU)",
    },
  },
  {
    // bf16 = the upper 16 bits of f32 (ADR 0131). The packing is the same as f16 (2 elements / word),
    // so it shares the odd-length traps and the alignment padding as is. The rounding is the RNE that
    // drops the lower 16 bits of f32 (helpers/bf16.ts).
    storage: "bf16",
    codec: "bf16",
    quantize: (weight) => {
      const quantized = quantizeBf16(weight.data);
      return { ...quantized, encoding: { codec: "bf16" } };
    },
    tileCases: [
      {
        // The same shapes as f16 (the v4 path with k=20 / the scalar path with k=37 — the parity detector)
        name: "linear v4 [65,20] × W[68,20] + b[68]",
        op: "linear",
        inputs: [["x", fill([65, 20], SIGNED)]],
        weight: fill([68, 20], POSITIVE),
        channelAxis: 0,
        bias: fill([68], SIGNED),
        outShape: [65, 68],
      },
      {
        name: "linear scalar [70,37] × W[23,37] + b[23]",
        op: "linear",
        inputs: [["x", fill([70, 37], SIGNED)]],
        weight: fill([23, 37], POSITIVE),
        channelAxis: 0,
        bias: fill([23], SIGNED),
        outShape: [70, 23],
      },
    ],
    // 12 elements × 2 bytes, already 4-byte aligned.
    mixedResidentBytes: 24,
    // Eligible: 21 elements × 2 bytes = 42 -> 44 with 4-byte alignment
    diagnostics: { payloadBytes: 42, residentBytes: 44 },
    titles: {
      eligibility: "bf16 eligibility goes through the same single decision as f16 (no new one)",
      graphOutputEligibility:
        "a bf16 initializer that became a graph output is ineligible by the same single decision",
      variants:
        "the 5 w=bf16 variants match the CPU reference (rounded weights) (odd lengths, real GPU)",
      tiles:
        "w=bf16 linear matches the CPU reference at the GEMM tile boundaries (v4 / scalar) (real GPU)",
      poison:
        "the w=bf16 variants also leave not a single poison word in pool-reused buffers (full-write / real GPU)",
      mixed:
        "bf16 with mixed consumption falls back to CPU expansion and gives bit-identical values to the eligible path (real GPU)",
      diagnostics:
        "the diagnostics tell eligible / ineligible / uncompressed apart by actual byte counts (bf16, real GPU)",
      graphOutput:
        "a bf16 initializer made a graph output falls back to CPU expansion, and the rounded values read as f32 (real GPU)",
    },
  },
  {
    storage: "i8",
    codec: "int8-sym",
    quantize: (weight, channelAxis) => {
      const quantized = quantizeI8(weight.data, weight.shape, channelAxis);
      // The per-channel groupSize is taken from the ledger (`codecs.ts`) (row length = numel /
      // shape[rowAxis], groups is 1) — recounting it in the test would make the supplied length and the
      // expected length share the same error.
      const numel = weight.shape.reduce((count, dim) => count * dim, 1);
      return {
        bytes: quantized.bytes,
        values: quantized.values,
        scale: quantized.scale,
        encoding: {
          codec: "int8-sym",
          rowAxis: channelAxis === 1 ? 1 : 0,
          groupSize: perChannelGroupSize(numel / weight.shape[channelAxis]),
          scale: { bytes: f32Bytes(quantized.scale), dtype: "f32" },
        },
      };
    },
    tileCases: [
      {
        // k=20 (a multiple of 4, not of 16) / n=68 (1 of the 16 quads of the last tile is valid) /
        // m=65 (2 row tiles). m/n/k all have different lengths, and the scale covers 68 channels.
        name: "linear v4 [65,20] × W[68,20] + b[68]",
        op: "linear",
        inputs: [["x", fill([65, 20], SIGNED)]],
        weight: fill([68, 20], rowVarying(20)),
        channelAxis: 0,
        bias: fill([68], SIGNED),
        outShape: [65, 68],
      },
      {
        // n=19 (not a multiple of 4 -> the scalar path; exercises the trap of the remainder mod 4 within a row) / k=20 / m=17
        name: "linear scalar [17,20] × W[19,20] + b[19]",
        op: "linear",
        inputs: [["x", fill([17, 20], SIGNED)]],
        weight: fill([19, 20], rowVarying(20)),
        channelAxis: 0,
        bias: fill([19], SIGNED),
        outShape: [17, 19],
      },
    ],
    // The eligible side: 12 bytes (a multiple of 4, no zero fill) + 3 scale elements, 12 bytes
    mixedResidentBytes: 12 + 12,
    // Eligible: 21 bytes -> 24 with 4-byte alignment, + 3 scale elements, 12 bytes
    diagnostics: { payloadBytes: 21, residentBytes: 24 + 12 },
    titles: {
      eligibility: "i8 eligibility goes through the same single decision as f16 (no new one)",
      graphOutputEligibility:
        "an i8 initializer that became a graph output is ineligible by the same single decision",
      variants:
        "the 5 w=i8 variants match the CPU reference (rounded weights) (row length not a multiple of 4, real GPU)",
      tiles:
        "w=i8 linear matches the CPU reference at the GEMM tile boundaries (v4 / scalar) (real GPU)",
      poison:
        "the w=i8 variants also leave not a single poison word in pool-reused buffers (full-write / real GPU)",
      mixed:
        "i8 with mixed consumption falls back to CPU expansion and gives bit-identical values to the eligible path (real GPU)",
      diagnostics:
        "the diagnostics tell eligible / ineligible / uncompressed apart by actual byte counts (scale included, real GPU)",
      graphOutput:
        "an i8 initializer made a graph output falls back to CPU expansion, and the rounded values read as f32 (real GPU)",
    },
  },
];

// ---------------------------------------------------------------------------
// Eligibility (GPU-independent — decided by the graph structure alone)
// ---------------------------------------------------------------------------

for (const storage of STORAGES) {
  test(storage.titles.eligibility, () => {
    const eligible = (declaration: DeclarationJson): readonly string[] =>
      [...eligibleCompressedInitializers(mergedFor(declaration, {}, storage.codec))].sort();

    // A sole weight consumption is eligible. The bias is **never eligible**, even in the same graph
    // (the reverse of the prototype's f16 demotion bug — the rule ADR 0006 names).
    assertEquals(eligible(linearDeclaration()), ["w"]);

    // MUST: if the same initializer is also consumed outside a weight slot, it loses eligibility
    // (mixed consumption). Uploading it compressed would make the elementwise kernel read u32 as f32, a
    // silently wrong value.
    assertEquals(
      eligible(
        linearDeclaration(
          [{ op: "add", ins: ["w", "w"], outs: ["z"], attrs: {} }],
          { z: { dtype: "f32", shape: [3, 4] } },
          ["z"],
        ),
      ),
      [],
    );

    // An initializer with no consumers is ineligible too (bytes not used by execution are not counted as "resident compressed")
    const unused = linearDeclaration();
    unused.initializers["dead"] = {};
    unused.values["dead"] = { dtype: "f32", shape: [2] };
    assertEquals(eligible(unused), ["w"]);
  });
}

test("eligibility looks at the weight-slot position of each WEIGHT_SLOTS op (bias / index are not made eligible)", () => {
  // op -> (order of ins, position of the weight). Only embedding has slot 0 (the others 1).
  const cases: readonly (readonly [string, readonly string[], readonly string[]])[] = [
    ["linear", ["x", "w", "b"], ["w"]],
    ["conv1d", ["x", "w", "b"], ["w"]],
    ["conv2d", ["x", "w", "b"], ["w"]],
    ["conv3d", ["x", "w", "b"], ["w"]],
    ["conv_transpose1d", ["x", "w", "b"], ["w"]],
    ["embedding", ["w", "x"], ["w"]],
    // Ops without a weight slot make no slot eligible
    ["layer_norm", ["x", "w", "b"], []],
    ["rms_norm", ["x", "w"], []],
    ["masked_fill", ["w", "x"], []],
  ];
  // Pin that the table covers every op in WEIGHT_SLOTS (keeps it from staying green with a stale table after an op is added)
  assertEquals(
    new Set(cases.filter(([, , expected]) => expected.length > 0).map(([op]) => op)),
    new Set(WEIGHT_SLOTS.keys()),
  );
  for (const [op, ins, expected] of cases) {
    // Eligibility looks only at the graph structure (op name x slot position), so the shape / dtype
    // declarations need not be consistent here (the contract checks are a separate layer —
    // validateGraphContracts in plan.ts).
    const graph = {
      initializers: {
        w: { storage: { codec: "f16" } },
        b: { storage: { codec: "f16" } },
      },
      nodes: [{ op, ins: [...ins], outs: ["y"], attrs: {} }],
    };
    assertEquals(
      [...eligibleCompressedInitializers(graph as never)].sort(),
      [...expected],
      `${op}(${ins.join(", ")})`,
    );
  }
});

for (const storage of STORAGES) {
  test(storage.titles.graphOutputEligibility, () => {
    const eligible = (declaration: DeclarationJson): readonly string[] =>
      [...eligibleCompressedInitializers(mergedFor(declaration, {}, storage.codec))].sort();

    // MUST: even when consumed only in a weight slot, the value loses eligibility if it is itself a
    // graph output. Readback copies from the weight buffer assuming semantic f32 (4 bytes / element), so
    // keeping it resident compressed would make the copy overrun the actual buffer (at tiny sizes a
    // reinterpretation of the bits comes back silently).
    assertEquals(eligible(linearDeclaration([], {}, ["w"])), []);

    // Disqualification applies **only to the name that is an output** (the other weight of the same graph stays eligible)
    const twoWeights = linearDeclaration(
      [{ op: "linear", ins: ["y", "w2", "b"], outs: ["y2"], attrs: {} }],
      { y2: { dtype: "f32", shape: [2, 3] } },
      ["y2", "w"],
    );
    // requires.ops rejects duplicates (one declaration covers the 2 linears)
    twoWeights.requires.ops = ["linear"];
    twoWeights.initializers["w2"] = {};
    twoWeights.values["w2"] = { dtype: "f32", shape: [3, 3] };
    assertEquals(eligible(twoWeights), ["w2"]);
  });
}

test("the channel axis is decided by the consuming op (1 only for conv_transpose1d)", () => {
  const axisOf = (op: string, ins: readonly string[]): number | undefined => {
    const graph = {
      initializers: { w: { storage: { codec: "int8-sym", groupSize: 4, rowAxis: 0 } } },
      nodes: [{ op, ins: [...ins], outs: ["y"], attrs: {} }],
    };
    return weightChannelAxes(graph as never).get("w");
  };
  assertEquals(axisOf("linear", ["x", "w", "b"]), 0);
  assertEquals(axisOf("conv1d", ["x", "w", "b"]), 0);
  assertEquals(axisOf("conv2d", ["x", "w", "b"]), 0);
  // MUST: the weight [Cin, Cout, K] is in a transposed layout, so the output channel is axis 1
  assertEquals(axisOf("conv_transpose1d", ["x", "w", "b"]), 1);
  assertEquals(axisOf("embedding", ["w", "x"]), 0);
  // Ops without a weight slot / positions other than the weight slot have no axis
  assertEquals(axisOf("layer_norm", ["x", "w", "b"]), undefined);
  assertEquals(axisOf("embedding", ["x", "w"]), undefined);
});

test("the same weight consumed by 2 ops with different axes fails loudly", () => {
  const graph = {
    initializers: { w: { storage: { codec: "int8-sym", groupSize: 4, rowAxis: 0 } } },
    nodes: [
      { op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} },
      { op: "conv_transpose1d", ins: ["x", "w", "b"], outs: ["z"], attrs: {} },
    ],
  };
  assertThrows(
    () => weightChannelAxes(graph as never),
    ExecutionError,
    "consumers disagree on the channel axis",
  );
});

// ---------------------------------------------------------------------------
// Validation of the merge layer (declaration x supply — GPU-independent)
// ---------------------------------------------------------------------------

/** The weight [3,4] of {@link linearDeclaration}, quantized to i8 (3 scales, one per row). */
const LINEAR_W = quantizeI8(fill([3, 4], POSITIVE).data, [3, 4], 0);

/** The i8 supply for {@link linearDeclaration}. Replacing `encoding` breaks it at exactly one point. */
const linearTensors = (encoding: EncodingInput): TensorInput[] => [
  { graph: GRAPH_NAME, initializer: "w", bytes: LINEAR_W.bytes, encoding },
  {
    graph: GRAPH_NAME,
    initializer: "b",
    bytes: f32Bytes([0.5, -0.25, 1]),
    encoding: { codec: "f32" },
  },
];

// The only codec that accepts group quantization is int4-sym-g (ADR 0069 decision 2) — a
// per-channel codec with a groupSize different from the row length is rejected by the merge layer
// (reading per-group scales as per-channel would be a silently wrong value).
test("a per-channel codec with a groupSize different from the row length is rejected at merge (ADR 0069)", () => {
  const error = assertThrows(
    () =>
      mergeTensors(
        linearDeclaration(),
        linearTensors({
          codec: "int8-sym",
          groupSize: 2,
          scale: { bytes: f32Bytes(LINEAR_W.scale), dtype: "f32" },
        }),
      ),
    ContainerFormatError,
    "is per-channel, so groupSize MUST equal the row length 4",
  );
  assertEquals(error.message.includes("initializer 'w'"), true, error.message);
});

/**
 * f16 / bf16 / i8 go through the same capability check (whether it is eligible is a separate axis
 * from whether it can run). Until ADR 0131, bf16 was a storage that "stays valid as a declaration
 * but cannot run"; it now runs on the same path as f16.
 */
test("f16 / bf16 / i8 pass the capability check (whether it is eligible is a separate axis from whether it can run)", () => {
  for (const storage of STORAGES) {
    // Compressed weights + an f32 bias (leaving out `b` would give `mergedFor`'s default f16 and change the fixture's intent).
    assertRuntimeSupport(
      mergedFor(linearDeclaration(), { b: "f32" }, storage.codec),
      RUNTIME_SUPPORT,
    );
  }
});

/**
 * The scale of the eligible path **must have a shape addressable by the flat index** (ADR 0019). A
 * shape that is merely broadcastable (`[1,4]` for the weight `[3,4]`) is read by the kernel as
 * `wscale[col]`, a silently wrong value — this pins that the admission which checks it against the
 * consuming op's axis (`planWeightResidency`) rejects it.
 */
test("a rowAxis that disagrees with the consuming axis fails loudly at admission (prepareContainer)", async () => {
  // A shape that declares axis 1 of the weight [3,4] as the rows. It is consistent as a container,
  // but the consuming op's (linear's) channel axis is 0 — letting it through would make the kernel
  // take the scale of another axis, a silently wrong value.
  const opened = await openModelBytes(
    linearDeclaration(),
    linearTensors({
      codec: "int8-sym",
      rowAxis: 1,
      groupSize: 3,
      scale: { bytes: f32Bytes([1, 1, 1, 1]), dtype: "f32" },
    }),
  );
  const error = assertThrows(
    () => prepareContainer(opened, GRAPH_NAME),
    ExecutionError,
    "the declared rowAxis 1 differs from the consuming op's channel axis 0",
  );
  assertEquals(error.message.includes("initializer 'w'"), true, error.message);
});

// ---------------------------------------------------------------------------
// CPU-side expansion (the spec of the host mirror)
// ---------------------------------------------------------------------------

test("f16 -> f32 expansion handles ±0 / subnormal / ±Inf / NaN / the largest and smallest normal correctly", () => {
  assert(Object.is(f16BitsToF32(0x0000), 0), "+0");
  assert(Object.is(f16BitsToF32(0x8000), -0), "-0");
  // subnormal: the smallest (2^-24) and the largest (1 ulp short of 2^-14)
  assertEquals(f16BitsToF32(0x0001), 2 ** -24);
  assertEquals(f16BitsToF32(0x03ff), 1023 * 2 ** -24);
  assertEquals(f16BitsToF32(0x8001), -(2 ** -24));
  // normal: the smallest (2^-14) and the largest (65504)
  assertEquals(f16BitsToF32(0x0400), 2 ** -14);
  assertEquals(f16BitsToF32(0x7bff), 65504);
  assertEquals(f16BitsToF32(0xfbff), -65504);
  assertEquals(f16BitsToF32(0x3c00), 1);
  assertEquals(f16BitsToF32(0xbc00), -1);
  assertEquals(f16BitsToF32(0x7c00), Infinity);
  assertEquals(f16BitsToF32(0xfc00), -Infinity);
  assert(Number.isNaN(f16BitsToF32(0x7e00)), "quiet NaN");
  assert(Number.isNaN(f16BitsToF32(0xfc01)), "signaling NaN (signed)");
  // After expansion the value is always exact in f32 (no rounding ever happens)
  for (const bits of [0x0001, 0x03ff, 0x0400, 0x7bff, 0x3555]) {
    assertEquals(Math.fround(f16BitsToF32(bits)), f16BitsToF32(bits), `bits=${bits}`);
  }
});

test("bf16 -> f32 expansion copies the upper 16 bits (±0 / subnormal / ±Inf / NaN / the largest and smallest normal)", () => {
  assert(Object.is(bf16BitsToF32(0x0000), 0), "+0");
  assert(Object.is(bf16BitsToF32(0x8000), -0), "-0");
  // subnormal: the smallest (2^-133) and the largest (1 ulp short of 2^-126) — they map directly onto f32 subnormals
  assertEquals(bf16BitsToF32(0x0001), 2 ** -133);
  assertEquals(bf16BitsToF32(0x007f), 127 * 2 ** -133);
  // normal: the smallest (2^-126) and the largest ((2 - 2^-7) · 2^127)
  assertEquals(bf16BitsToF32(0x0080), 2 ** -126);
  assertEquals(bf16BitsToF32(0x7f7f), (2 - 2 ** -7) * 2 ** 127);
  assertEquals(bf16BitsToF32(0xff7f), -(2 - 2 ** -7) * 2 ** 127);
  assertEquals(bf16BitsToF32(0x3f80), 1);
  assertEquals(bf16BitsToF32(0xbf80), -1);
  assertEquals(bf16BitsToF32(0x7f80), Infinity);
  assertEquals(bf16BitsToF32(0xff80), -Infinity);
  assert(Number.isNaN(bf16BitsToF32(0x7fc0)), "quiet NaN");
  assert(Number.isNaN(bf16BitsToF32(0xff81)), "signaling NaN (signed)");
  // The expansion is a copy of bits, so the f32 -> bf16 round trip (values whose lower 16 bits are 0) is the identity
  for (const bits of [0x0001, 0x007f, 0x0080, 0x7f7f, 0x3555]) {
    const value = bf16BitsToF32(bits);
    assertEquals(Math.fround(value), value, `bits=${bits}`);
    assertEquals(decodeBf16(bf16BytesFromBits([bits]))[0], value, `bits=${bits}`);
  }
});

test("decodeI8: expands signed 8-bit values with the per-channel scale (keepdim broadcast)", () => {
  // A [2,3] weight. The scale [2,1] on channel axis 0 applies per row.
  const bytes = i8BytesFrom([127, -128, 0, 1, -1, 64]);
  const scale = Float32Array.from([0.5, 0.25]);
  assertEquals([...decodeI8(bytes, [2, 3], scale, [2, 1])], [63.5, -64, 0, 0.25, -0.25, 16]);
  // Channel axis 1 (the conv_transpose1d shape). Applies per column.
  assertEquals(
    [...decodeI8(bytes, [2, 3], Float32Array.from([1, 2, 4]), [1, 3])],
    [127, -256, 0, 1, -2, 256],
  );
  // A scale with every axis 1 (per-tensor shape) also holds as a broadcast
  assertEquals(
    [...decodeI8(bytes, [2, 3], Float32Array.from([2]), [1, 1])],
    [254, -256, 0, 2, -2, 128],
  );
});

test("decodeI8: broadcasts correctly per axis at rank 3 and above too", () => {
  // [2,3,2] (the [Cout,Cin,K] shape of conv1d). Channel axis 0 = output channels.
  const q = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6];
  const bytes = i8BytesFrom(q);
  const scale = Float32Array.from([0.5, 4]);
  const decoded = [...decodeI8(bytes, [2, 3, 2], scale, [2, 1, 1])];
  assertEquals(decoded, [
    ...q.slice(0, 6).map((value) => value * 0.5),
    ...q.slice(6).map((value) => value * 4),
  ]);
  // The middle axis (the [Cin,Cout,K] shape of conv_transpose1d — axis 1 is the output channels)
  const middle = [...decodeI8(bytes, [2, 3, 2], Float32Array.from([1, 2, 4]), [1, 3, 1])];
  assertEquals(middle, [1, 2, 6, 8, 20, 24, -1, -2, -6, -8, -20, -24]);
});

test("decodeI8: the expansion rounds only once, in f32 (the same as the GPU's f32 multiplication)", () => {
  // 0.1 is a value that f32 rounds. q·s is exact in f64 (7 bits x 24 bits), so the result rounded
  // once to f32 matches the correct rounding of the GPU's f32 multiplication.
  const scale = Float32Array.from([0.1]);
  const decoded = decodeI8(i8BytesFrom([3, 7, 127]), [3, 1], scale, [1, 1]);
  for (const [index, q] of [3, 7, 127].entries()) {
    assertEquals(decoded[index], Math.fround(q * scale[0]), `q=${q}`);
    assertEquals(Math.fround(decoded[index]), decoded[index], `q=${q}: exact in f32`);
  }
});

test("decodeI8: rejects a scale that cannot broadcast and an element-count mismatch", () => {
  const bytes = i8BytesFrom([1, 2, 3, 4, 5, 6]);
  assertThrows(() => decodeI8(bytes, [2, 3], Float32Array.from([1, 2]), [2]), I8Error, "rank");
  // Axis 1 is 2 (unlike the weight's 3) — the element count is consistent with the shape, so the broadcast check catches it
  assertThrows(
    () => decodeI8(bytes, [2, 3], Float32Array.from([1, 2, 3, 4]), [2, 2]),
    I8Error,
    "cannot be broadcast",
  );
  assertThrows(
    () => decodeI8(bytes, [2, 4], Float32Array.from([1, 2]), [2, 1]),
    I8Error,
    "payload",
  );
});

test("alignI8Payload: zero-fills up to a 4-byte boundary and does not copy an already aligned payload", () => {
  const aligned = i8BytesFrom([1, 2, 3, 4]);
  assertEquals(alignI8Payload(aligned), aligned);
  assert(alignI8Payload(aligned) === aligned, "an aligned payload returns the same view");
  for (const length of [1, 2, 3, 5, 6, 7]) {
    const padded = alignI8Payload(i8BytesFrom(new Array(length).fill(7)));
    assertEquals(padded.byteLength % 4, 0, `${length} bytes`);
    assertEquals(padded.byteLength, length + (4 - (length % 4)), `${length} bytes`);
    assertEquals([...padded.slice(length)], new Array(padded.byteLength - length).fill(0));
  }
});

// ---------------------------------------------------------------------------
// GPU execution (variant kernels)
// ---------------------------------------------------------------------------

/** The declaration of a single node with the weight as a compressed initializer + the opened container. */
const weightedModel = (
  testCase: WeightedCase,
  quantized: QuantizedWeight,
): Promise<OpenedContainer> => {
  const values: DeclarationJson["values"] = {
    w: { dtype: "f32", shape: [...testCase.weight.shape] },
    y: { dtype: "f32", shape: [...testCase.outShape] },
  };
  const initializers: DeclarationJson["initializers"] = { w: {} };
  // NOTE: even for an f16 with an odd element count (42 bytes), the order of placement does not
  // matter — the writer bakes 64 B alignment at the start of each block + the trailing zero fill into
  // the container, so the alignment holds whatever sits next to it.
  const tensors: TensorInput[] = [];
  if (testCase.bias !== undefined) {
    values["b"] = { dtype: "f32", shape: [...testCase.bias.shape] };
    initializers["b"] = {};
    tensors.push({
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: new Uint8Array(testCase.bias.data.buffer.slice(0)),
      encoding: { codec: "f32" },
    });
  }
  tensors.push({
    graph: GRAPH_NAME,
    initializer: "w",
    bytes: quantized.bytes,
    encoding: quantized.encoding,
  });
  // The order of ins follows the contract (embedding has the weight first, the others after x)
  const ins =
    testCase.op === "embedding"
      ? ["w", ...testCase.inputs.map(([name]) => name)]
      : [
          ...testCase.inputs.map(([name]) => name),
          "w",
          ...(testCase.bias === undefined ? [] : ["b"]),
        ];
  const declaration: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: [testCase.op] },
    symbols: [],
    inputs: testCase.inputs.map(([name, tensor]) => ({
      name,
      dtype: tensor.dtype,
      shape: [...tensor.shape],
    })),
    outputs: ["y"],
    initializers,
    values,
    nodes: [{ op: testCase.op, ins, outs: ["y"], attrs: { ...testCase.attrs } }],
  };
  return openModelBytes(declaration, tensors);
};

const runWeighted = async (
  gpu: GpuContext,
  testCase: WeightedCase,
  quantized: QuantizedWeight,
): Promise<{ readonly output: Tensor; readonly residentBytes: number }> => {
  const session = await createSessionFromContainer(
    gpu,
    await weightedModel(testCase, quantized),
    GRAPH_NAME,
  );
  try {
    const named: Record<string, Tensor> = {};
    for (const [name, tensor] of testCase.inputs) named[name] = tensor;
    const outputs = await session.run(named);
    return {
      output: outputs["y"],
      residentBytes: session.diagnostics().storage.residentCompressedBytes,
    };
  } finally {
    await session.dispose();
  }
};

/**
 * Numeric cases of the 5 variants. **The total element count of every weight is odd and not a
 * multiple of 4** (the zero-fill path), and **the weight row length is also odd and not a multiple of
 * 4** (the detector that the pair choice / the word and lane choice uses the flat index).
 */
const WEIGHTED_CASES: readonly WeightedCase[] = [
  {
    // k = 7 (odd) / weight 3·7 = 21 elements (odd)
    name: "linear [3,7] × W[3,7] + b[3]",
    op: "linear",
    inputs: [["x", fill([3, 7], SIGNED)]],
    weight: fill([3, 7], POSITIVE),
    channelAxis: 0,
    bias: fill([3], SIGNED),
    outShape: [3, 3],
  },
  {
    // Cin·K = 9 (odd) / weight 5·3·3 = 45 elements (odd)
    name: "conv1d [1,3,9] * W[5,3,3] stride=1 padding=1",
    op: "conv1d",
    inputs: [["x", fill([1, 3, 9], SIGNED)]],
    weight: fill([5, 3, 3], POSITIVE),
    channelAxis: 0,
    bias: fill([5], SIGNED),
    outShape: [1, 5, 9],
    attrs: { stride: 1, padding: 1, dilation: 1, groups: 1 },
  },
  {
    // Cin·Kh·Kw = 9 (odd) / weight 5·3·3·1 = 45 elements (odd). Kh ≠ Kw turns an axis mix-up red too
    name: "conv2d [1,3,5,4] * W[5,3,3,1]",
    op: "conv2d",
    inputs: [["x", fill([1, 3, 5, 4], SIGNED)]],
    weight: fill([5, 3, 3, 1], POSITIVE),
    channelAxis: 0,
    bias: fill([5], SIGNED),
    outShape: [1, 5, 5, 4],
    attrs: { stride: [1, 1], padding: [1, 0], dilation: [1, 1], groups: 1 },
  },
  {
    // K = 3 (odd; stride 1 satisfies the contract 2·padding == K − stride) / weight 3·5·3 = 45 elements
    // **The channel axis is 1** (the transposed [Cin,Cout,K] layout), and Cin ≠ Cout, so mistaking the
    // axis for 0 turns the shape check red.
    name: "conv_transpose1d [1,3,7] * W[3,5,3] stride=1 padding=1",
    op: "conv_transpose1d",
    inputs: [["x", fill([1, 3, 7], SIGNED)]],
    weight: fill([3, 5, 3], POSITIVE),
    channelAxis: 1,
    bias: fill([5], SIGNED),
    outShape: [1, 5, 7],
    attrs: { stride: 1, padding: 1 },
  },
  {
    // H = 3 (odd) / weight 5·3 = 15 elements (odd)
    name: "embedding W[5,3] × index [2,4]",
    op: "embedding",
    inputs: [["x", fill([2, 4], (i) => (i * 3) % 5, "i32")]],
    weight: fill([5, 3], SIGNED),
    channelAxis: 0,
    outShape: [2, 4, 3],
    attrs: { padding_idx: -1 },
  },
];

for (const storage of STORAGES) {
  test.skipIf(!GPU_AVAILABLE)(storage.titles.variants, async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of WEIGHTED_CASES) {
        const quantized = storage.quantize(testCase.weight, testCase.channelAxis);
        // A payload that is not 4-byte aligned needs the zero padding (f16: an odd element
        // count; i8: an element count that is not a multiple of 4).
        assert(
          quantized.bytes.byteLength % 4 !== 0,
          `${testCase.name}: payload is 4-byte aligned (the zero-padding path is not taken)`,
        );
        const { output, residentBytes } = await runWeighted(gpu, testCase, quantized);
        // Resident compressed (= not fallen back to CPU expansion). Includes the zero fill and the scale.
        const padded = quantized.bytes.byteLength + (4 - (quantized.bytes.byteLength % 4));
        assertEquals(
          residentBytes,
          padded + (quantized.scale?.byteLength ?? 0),
          `${testCase.name}: compressed bytes resident on the GPU (scale included)`,
        );
        const reference = refTensor(testCase.weight.shape, quantized.values);
        const operands: RefTensor[] =
          testCase.op === "embedding"
            ? [reference, ...testCase.inputs.map(([, tensor]) => tensor as RefTensor)]
            : [
                ...testCase.inputs.map(([, tensor]) => tensor as RefTensor),
                reference,
                ...(testCase.bias === undefined ? [] : [testCase.bias as RefTensor]),
              ];
        const expected = applyReferenceOp(
          testCase.op,
          operands,
          testCase.attrs ?? {},
          testCase.outShape,
        );
        assertEquals(output.shape, expected.shape, testCase.name);
        const report = compareTensors(output, expected, GEMM_TOLERANCE);
        assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
      }
    } finally {
      gpu.destroy();
    }
  });
}

/**
 * Exercises linear's GEMM tile boundaries (64×64 register tiles — src/kernels/gemm.ts) with
 * compressed weights.
 *
 * f16 — MUST: **have the v4 path and the scalar path as a pair**.
 * - v4 (`k%4==0 && n%4==0`) builds 4 elements from **2 u32 words**; an offset error in the second
 *   word always shows in the values for shapes where k is a multiple of 4 (k=20).
 * - The scalar path selects by parity, 2 elements per word; together with {@link WEIGHTED_CASES}
 *   (k=7) above, the shape "row length not a multiple of 2" is the only detector (k=37). On the v4
 *   path a row always starts on a word boundary, so this trap **cannot be hit** — removing the scalar
 *   case would remove the detection power.
 *
 * i8 — MUST: **have the v4 path and the scalar path as a pair, both with different n / k lengths**.
 * The scale is **per output channel** (`wscale[wcol]`), and the error of taking it from another axis
 * (row / k / anything other than the output column) shows in the values only when "the scale differs
 * per channel". quantizeI8 sets the scale from each row's maximum, so weights whose values differ per
 * row give a scale that differs per row too.
 * MUST: even on the v4 path the scale is **an f32 multiplication per component** (`vec4 * scalar`) —
 * moving it outside the reduction would run into ADR 0019's "(Σ x·q)·s loses bit identity with CPU
 * expansion".
 */
for (const storage of STORAGES) {
  test.skipIf(!GPU_AVAILABLE)(storage.titles.tiles, async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of storage.tileCases) {
        const quantized = storage.quantize(testCase.weight, testCase.channelAxis);
        // The scale differs per channel (if it were the same, an axis mix-up would not show in the numbers — the vacuity check)
        if (quantized.scale !== undefined) {
          assert(
            new Set(quantized.scale).size > 1,
            `${testCase.name}: the scale is the same for every channel`,
          );
        }
        const { output } = await runWeighted(gpu, testCase, quantized);
        const expected = applyReferenceOp(
          testCase.op,
          [
            testCase.inputs[0][1] as RefTensor,
            refTensor(testCase.weight.shape, quantized.values),
            testCase.bias as RefTensor,
          ],
          {},
          testCase.outShape,
        );
        assertEquals(output.shape, expected.shape, testCase.name);
        const report = compareTensors(output, expected, GEMM_TOLERANCE);
        assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
      }
    } finally {
      gpu.destroy();
    }
  });
}

/**
 * The GPU's `unpack2x16float` and the CPU's {@link f16BitsToF32} are **bit-identical over all 65,536
 * patterns** (ADR 0018's verification condition).
 *
 * How: the embedding weight is a [256, 256] f16 laid out as `bits = row·256 + col`, covering every
 * pattern at once. embedding is a row gather with not a single arithmetic operation in between, so
 * the output is the dequant result itself.
 *
 * MUST: NaN alone is checked as **NaN on both sides**. WGSL does not guarantee NaN payload
 * propagation, so demanding bit equality would make the test depend on a property outside the spec.
 */
test.skipIf(!GPU_AVAILABLE)(
  "the GPU's unpack2x16float and CPU expansion are bit-identical over all 65,536 patterns (real GPU)",
  async () => {
    const vocab = 256;
    const hidden = 256;
    const bytes = f16BytesFromBits(Array.from({ length: vocab * hidden }, (_, i) => i));
    const declaration: DeclarationJson = {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["embedding"] },
      symbols: [],
      inputs: [{ name: "index", dtype: "i32", shape: [vocab] }],
      outputs: ["y"],
      initializers: { w: {} },
      values: {
        w: { dtype: "f32", shape: [vocab, hidden] },
        y: { dtype: "f32", shape: [vocab, hidden] },
      },
      nodes: [
        {
          op: "embedding",
          ins: ["w", "index"],
          outs: ["y"],
          attrs: { padding_idx: -1 },
        },
      ],
    };
    const opened = await openModelBytes(declaration, [
      { graph: GRAPH_NAME, initializer: "w", bytes, encoding: { codec: "f16" } },
    ]);
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME);
    let actual: Float32Array<ArrayBuffer>;
    try {
      const outputs = await session.run({ index: fill([vocab], (i) => i, "i32") });
      actual = outputs["y"].data as Float32Array<ArrayBuffer>;
      // If it is not resident compressed, the test is not looking at "the GPU's expansion"
      assertEquals(session.diagnostics().storage.residentCompressedBytes, bytes.byteLength);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
    const expected = decodeF16(bytes);
    const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
    const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
    let nan = 0;
    let subnormal = 0;
    for (let i = 0; i < expected.length; i += 1) {
      if (Number.isNaN(expected[i])) {
        nan += 1;
        assert(Number.isNaN(actual[i]), `pattern 0x${i.toString(16)}: the GPU side is not NaN`);
        continue;
      }
      if (((i >> 10) & 0x1f) === 0 && (i & 0x3ff) !== 0) subnormal += 1;
      assertEquals(
        actualBits[i],
        expectedBits[i],
        `pattern 0x${i.toString(16)}: GPU ${actual[i]} / CPU ${expected[i]}`,
      );
    }
    // Evidence of exhaustiveness (2046 NaN and 2046 subnormal patterns are actually exercised)
    assertEquals(nan, 2046, "number of NaN patterns");
    assertEquals(subnormal, 2046, "number of subnormal patterns");
  },
);

/**
 * The GPU's bf16 expansion (`bitcast<f32>` — ADR 0131) and the CPU's {@link bf16BitsToF32} **agree
 * over all 65,536 patterns**. The method is the same as the f16 check (embedding [256, 256],
 * `bits = row·256 + col`).
 *
 * MUST: NaN is checked as **NaN on both sides** (WGSL does not guarantee NaN payload propagation —
 * the same as f16).
 * MUST: patterns that map onto f32 subnormals (exponent 0, mantissa ≠ 0) **may be flushed to 0 by
 * the GPU** (WGSL does not guarantee that subnormals are kept) — everything else must be
 * bit-identical.
 */
test.skipIf(!GPU_AVAILABLE)(
  "the GPU's bf16 bitcast expansion and CPU expansion agree over all 65,536 patterns (real GPU, subnormal flush allowed)",
  async () => {
    const vocab = 256;
    const hidden = 256;
    const bytes = bf16BytesFromBits(Array.from({ length: vocab * hidden }, (_, i) => i));
    const declaration: DeclarationJson = {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["embedding"] },
      symbols: [],
      inputs: [{ name: "index", dtype: "i32", shape: [vocab] }],
      outputs: ["y"],
      initializers: { w: {} },
      values: {
        w: { dtype: "f32", shape: [vocab, hidden] },
        y: { dtype: "f32", shape: [vocab, hidden] },
      },
      nodes: [
        {
          op: "embedding",
          ins: ["w", "index"],
          outs: ["y"],
          attrs: { padding_idx: -1 },
        },
      ],
    };
    const opened = await openModelBytes(declaration, [
      { graph: GRAPH_NAME, initializer: "w", bytes, encoding: { codec: "bf16" } },
    ]);
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME);
    let actual: Float32Array<ArrayBuffer>;
    try {
      const outputs = await session.run({ index: fill([vocab], (i) => i, "i32") });
      actual = outputs["y"].data as Float32Array<ArrayBuffer>;
      // If it is not resident compressed, the test is not looking at "the GPU's expansion"
      assertEquals(session.diagnostics().storage.residentCompressedBytes, bytes.byteLength);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
    const expected = decodeBf16(bytes);
    const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
    const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
    let nan = 0;
    let subnormal = 0;
    let flushed = 0;
    for (let i = 0; i < expected.length; i += 1) {
      if (Number.isNaN(expected[i])) {
        nan += 1;
        assert(Number.isNaN(actual[i]), `pattern 0x${i.toString(16)}: the GPU side is not NaN`);
        continue;
      }
      if (((i >> 7) & 0xff) === 0 && (i & 0x7f) !== 0) {
        subnormal += 1;
        if (actual[i] === 0) {
          flushed += 1;
          continue;
        }
      }
      assertEquals(
        actualBits[i],
        expectedBits[i],
        `pattern 0x${i.toString(16)}: GPU ${actual[i]} / CPU ${expected[i]}`,
      );
    }
    // Evidence of exhaustiveness (254 NaN and 254 subnormal patterns are actually exercised). Whether
    // flushing happens is a fact of each machine, not a condition of the check (only the counts are
    // kept here).
    assertEquals(nan, 254, "number of NaN patterns");
    assertEquals(subnormal, 254, "number of subnormal patterns");
    assert(flushed <= subnormal, "the flush count exceeds the subnormal count");
  },
);

/**
 * The GPU's `unpack4xI8` + scale multiplication and the CPU's {@link decodeI8} are **bit-identical
 * over all 256 values x 4 lane positions** (ADR 0019's verification condition).
 *
 * How: the embedding weight is a [256, 4] i8 with the value `r − 128` in all 4 elements of row r.
 * The flat index is `r·4 + col`, so the same value appears in all 4 lane positions. embedding is a
 * row gather with not a single arithmetic operation in between, so the output is the dequant result
 * itself.
 *
 * The scale is a different non-power-of-2 value per row (the detector for forgetting to multiply by
 * the scale and for mixing up rows, and at the same time a check that "one rounding of the f32
 * multiplication" matches the host).
 */
test.skipIf(!GPU_AVAILABLE)(
  "the GPU's unpack4xI8 and CPU expansion are bit-identical over all 256 values x 4 lanes (real GPU)",
  async () => {
    const vocab = 256;
    const hidden = 4;
    const quantized: number[] = [];
    for (let row = 0; row < vocab; row += 1) {
      for (let col = 0; col < hidden; col += 1) quantized.push(row - 128);
    }
    const bytes = i8BytesFrom(quantized);
    const scale = Float32Array.from({ length: vocab }, (_, row) => 0.1 + row * 0.003);
    const declaration: DeclarationJson = {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["embedding"] },
      symbols: [],
      inputs: [{ name: "index", dtype: "i32", shape: [vocab] }],
      outputs: ["y"],
      initializers: { w: {} },
      values: {
        w: { dtype: "f32", shape: [vocab, hidden] },
        y: { dtype: "f32", shape: [vocab, hidden] },
      },
      nodes: [{ op: "embedding", ins: ["w", "index"], outs: ["y"], attrs: { padding_idx: -1 } }],
    };
    const opened = await openModelBytes(declaration, [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes,
        encoding: {
          codec: "int8-sym",
          groupSize: hidden,
          scale: { bytes: f32Bytes(scale), dtype: "f32" },
        },
      },
    ]);
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME);
    let actual: Float32Array<ArrayBuffer>;
    try {
      const outputs = await session.run({ index: fill([vocab], (i) => i, "i32") });
      actual = outputs["y"].data as Float32Array<ArrayBuffer>;
      // If it is not resident compressed, the test is not looking at "the GPU's expansion"
      assertEquals(
        session.diagnostics().storage.residentCompressedBytes,
        bytes.byteLength + scale.byteLength,
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
    const expected = decodeI8(bytes, [vocab, hidden], scale, [vocab, 1]);
    const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
    const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
    const seen = new Set<string>();
    for (let i = 0; i < expected.length; i += 1) {
      assertEquals(
        actualBits[i],
        expectedBits[i],
        `q=${quantized[i]} lane=${i & 3}: GPU ${actual[i]} / CPU ${expected[i]}`,
      );
      seen.add(`${quantized[i]}:${i & 3}`);
    }
    // Evidence of exhaustiveness (256 values x 4 lanes are actually exercised)
    assertEquals(seen.size, 256 * 4, "number of value x lane combinations covered");
  },
);

/**
 * full-write of the variant paths (ADR 0014). Poison values are planted in pool-reused buffers
 * before running the compressed variants, and the test checks that not a single word remains.
 *
 * They are planted the same way as in tests/gpu_full_write_test.ts (the output of an identity cast
 * with no consumers returns to the pool at the node boundary and is handed out again by the next
 * allocation).
 */
const poisonedGraph = (
  op: string,
  node: { readonly ins: readonly string[]; readonly attrs: Record<string, unknown> },
  inputs: readonly (readonly [string, FilledTensor])[],
  weightShape: readonly number[],
  outShape: readonly number[],
  bias?: readonly number[],
): DeclarationJson => {
  const count = outShape.reduce((total, dim) => total * dim, 1);
  const values: DeclarationJson["values"] = {
    poison: { dtype: "f32", shape: [count] },
    w: { dtype: "f32", shape: [...weightShape] },
    y: { dtype: "f32", shape: [...outShape] },
  };
  const initializers: DeclarationJson["initializers"] = { w: {} };
  if (bias !== undefined) {
    values["b"] = { dtype: "f32", shape: [...bias] };
    initializers["b"] = {};
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["cast", op] },
    symbols: [],
    inputs: [
      { name: "seed", dtype: "f32", shape: [count] },
      ...inputs.map(([name, tensor]) => ({
        name,
        dtype: tensor.dtype,
        shape: [...tensor.shape],
      })),
    ],
    outputs: ["y"],
    initializers,
    values,
    nodes: [
      { op: "cast", ins: ["seed"], outs: ["poison"], attrs: { to: "f32" } },
      { op, ins: [...node.ins], outs: ["y"], attrs: { ...node.attrs } },
    ],
  };
};

/** The poison value 0xDEADBEEF read as f32 (a non-zero finite value). */
const POISON = new Float32Array(new Uint32Array([0xdeadbeef]).buffer)[0];

for (const storage of STORAGES) {
  test.skipIf(!GPU_AVAILABLE)(storage.titles.poison, async () => {
    const gpu = await acquireGpu();
    try {
      // Exercised with 2 ops: linear (tiled) and embedding (grid-stride + out-of-range branch)
      const cases: readonly {
        readonly name: string;
        readonly graph: DeclarationJson;
        readonly tensors: readonly TensorInput[];
        readonly inputs: Record<string, Tensor>;
        readonly count: number;
      }[] = [
        (() => {
          const quantized = storage.quantize(fill([3, 7], POSITIVE), 0);
          const bias = fill([3], SIGNED);
          return {
            name: "linear",
            graph: poisonedGraph(
              "linear",
              { ins: ["x", "w", "b"], attrs: {} },
              [["x", fill([4, 7], SIGNED)]],
              [3, 7],
              [4, 3],
              [3],
            ),
            tensors: [
              {
                graph: GRAPH_NAME,
                initializer: "b",
                bytes: new Uint8Array(bias.data.buffer.slice(0)),
                encoding: { codec: "f32" },
              },
              {
                graph: GRAPH_NAME,
                initializer: "w",
                bytes: quantized.bytes,
                encoding: quantized.encoding,
              },
            ],
            inputs: { x: fill([4, 7], SIGNED) },
            count: 12,
          };
        })(),
        (() => {
          const quantized = storage.quantize(fill([5, 3], SIGNED), 0);
          return {
            name: "embedding",
            graph: poisonedGraph(
              "embedding",
              { ins: ["w", "idx"], attrs: { padding_idx: -1 } },
              [["idx", fill([4], (i) => i % 5, "i32")]],
              [5, 3],
              [4, 3],
            ),
            tensors: [
              {
                graph: GRAPH_NAME,
                initializer: "w",
                bytes: quantized.bytes,
                encoding: quantized.encoding,
              },
            ],
            inputs: { idx: fill([4], (i) => i % 5, "i32") },
            count: 12,
          };
        })(),
      ];
      for (const testCase of cases) {
        const opened = await openModelBytes(testCase.graph, testCase.tensors);
        const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME);
        try {
          const outputs = await session.run({
            seed: fill([testCase.count], () => POISON),
            ...testCase.inputs,
          });
          const reuseCount = session.diagnostics().lastRun?.reuseCount ?? 0;
          // If no reuse happened, the poison check sees nothing (the vacuity check)
          assert(reuseCount >= 1, `${testCase.name}: no pool reuse happened`);
          assertEquals(
            [...outputs["y"].data].filter((value) => value === POISON),
            [],
            `${testCase.name}: poison values remain`,
          );
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  });
}

/**
 * The ineligible case (mixed consumption) is expanded to f32 on the CPU and gives **the same values
 * as the eligible path** (for i8, ADR 0019's "per-element dequant" is the reason they agree).
 *
 * MUST: compare the outputs of the 2 paths. Comparing only one of them with the CPU reference would
 * miss the case where both paths are off by the same amount (e.g. the expansion table is wrong and
 * the GPU side has the same error).
 */
for (const storage of STORAGES) {
  test.skipIf(!GPU_AVAILABLE)(storage.titles.mixed, async () => {
    const weight = fill([3, 4], POSITIVE);
    const bias = fill([3], SIGNED);
    const quantized = storage.quantize(weight, 0);
    const tensors: readonly TensorInput[] = [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes: quantized.bytes,
        encoding: quantized.encoding,
      },
      {
        graph: GRAPH_NAME,
        initializer: "b",
        bytes: new Uint8Array(bias.data.buffer.slice(0)),
        encoding: { codec: "f32" },
      },
    ];
    const x = fill([2, 4], SIGNED);
    const run = async (
      gpu: GpuContext,
      declaration: DeclarationJson,
    ): Promise<{
      readonly y: Tensor;
      readonly resident: number;
      readonly expanded: number;
    }> => {
      const session = await createSessionFromContainer(
        gpu,
        await openModelBytes(declaration, tensors),
        GRAPH_NAME,
      );
      try {
        const outputs = await session.run({ x });
        const diagnostics = session.diagnostics().storage;
        return {
          y: outputs["y"],
          resident: diagnostics.residentCompressedBytes,
          expanded: diagnostics.hostExpandedBytes,
        };
      } finally {
        await session.dispose();
      }
    };
    const gpu = await acquireGpu();
    try {
      const eligible = await run(gpu, linearDeclaration());
      // w is also consumed by add = mixed consumption (the same linear node stays)
      const mixed = await run(
        gpu,
        linearDeclaration(
          [{ op: "add", ins: ["w", "w"], outs: ["z"], attrs: {} }],
          { z: { dtype: "f32", shape: [3, 4] } },
          ["z"],
        ),
      );
      assertEquals(
        eligible.resident,
        storage.mixedResidentBytes,
        "the eligible side stays resident compressed (scale included)",
      );
      assertEquals(eligible.expanded, 0);
      assertEquals(
        mixed.resident,
        0,
        "mixed consumption keeps not a single byte resident compressed",
      );
      assertEquals(mixed.expanded, 48, "CPU-expanded bytes (12 elements as f32)");
      // Same weights, same computation, so the outputs are bit-identical (no rounding difference either)
      assertEquals([...mixed.y.data], [...eligible.y.data]);
      // Oracle: the CPU reference with the rounded weights
      const expected = applyReferenceOp(
        "linear",
        [x as RefTensor, refTensor([3, 4], quantized.values), bias as RefTensor],
        {},
        [2, 3],
      );
      const report = compareTensors(eligible.y, expected, GEMM_TOLERANCE);
      assertEquals(report.pass, true, formatAllclose(report));
    } finally {
      gpu.destroy();
    }
  });
}

/**
 * A cross-check of the diagnostics (a standing obligation of ADR 0006). Eligible, ineligible and
 * uncompressed are mixed into one graph, and both numbers must match **the actual byte counts**.
 *
 * Even when the bias is declared with compressed storage it **always falls back to CPU expansion**
 * (the bias slot is not part of eligibility — the reverse of the prototype's demotion bug).
 */
for (const storage of STORAGES) {
  test.skipIf(!GPU_AVAILABLE)(storage.titles.diagnostics, async () => {
    // Weight 3·7 = 21 elements (a length that needs the zero fill) / bias of 3 elements **declared with compressed storage**
    const weight = storage.quantize(fill([3, 7], POSITIVE), 0);
    const bias = storage.quantize(fill([3], SIGNED), 0);
    const scale = fill([3], POSITIVE);
    const declaration: DeclarationJson = {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["linear", "mul"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [2, 7] }],
      outputs: ["y"],
      initializers: { w: {}, b: {}, s: {} },
      values: {
        w: { dtype: "f32", shape: [3, 7] },
        b: { dtype: "f32", shape: [3] },
        s: { dtype: "f32", shape: [3] },
        h: { dtype: "f32", shape: [2, 3] },
        y: { dtype: "f32", shape: [2, 3] },
      },
      nodes: [
        { op: "linear", ins: ["x", "w", "b"], outs: ["h"], attrs: {} },
        { op: "mul", ins: ["h", "s"], outs: ["y"], attrs: {} },
      ],
    };
    const opened = await openModelBytes(declaration, [
      { graph: GRAPH_NAME, initializer: "w", bytes: weight.bytes, encoding: weight.encoding },
      // i8: the bias is rank 1 (rows = 3, row length 1) — the per-channel groupSize is the row length 1.
      { graph: GRAPH_NAME, initializer: "b", bytes: bias.bytes, encoding: bias.encoding },
      {
        graph: GRAPH_NAME,
        initializer: "s",
        bytes: new Uint8Array(scale.data.buffer.slice(0)),
        encoding: { codec: "f32" },
      },
    ]);
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME);
    try {
      const diagnostics = session.diagnostics().storage;
      assertEquals(weight.bytes.byteLength, storage.diagnostics.payloadBytes);
      assertEquals(diagnostics.residentCompressedBytes, storage.diagnostics.residentBytes);
      // Ineligible: the 3 bias elements expanded to f32 = 12 bytes (the f32-stored s goes **into
      // neither**. The bias's scale is not uploaded to the GPU, so it counts as neither resident nor
      // expanded)
      assertEquals(diagnostics.hostExpandedBytes, 12);
      // The expanded bias is also correct as values (closes off the case where only the diagnostics match and the contents are broken)
      const x = fill([2, 7], SIGNED);
      const outputs = await session.run({ x });
      const linear = applyReferenceOp(
        "linear",
        [x as RefTensor, refTensor([3, 7], weight.values), refTensor([3], bias.values)],
        {},
        [2, 3],
      );
      const expected = applyReferenceOp("mul", [linear, scale as RefTensor], {}, [2, 3]);
      const report = compareTensors(outputs["y"], expected, GEMM_TOLERANCE);
      assertEquals(report.pass, true, formatAllclose(report));
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  });
}

/**
 * A shape that makes an initializer itself a graph output (the IR allows it — the definition check in
 * format/ir.ts).
 *
 * Readback copies the declared dtype's semantic f32 from the weight buffer at 4 bytes / element, so
 * if it stayed resident compressed, the copy would overrun the actual buffer (f16 is 2 bytes /
 * element, i8 is 1 byte / element). If eligibility looks at `graph.outputs`, it falls back to CPU
 * expansion and the rounded weights read as they are.
 */
for (const storage of STORAGES) {
  test.skipIf(!GPU_AVAILABLE)(storage.titles.graphOutput, async () => {
    const weight = fill([3, 4], POSITIVE);
    const bias = fill([3], SIGNED);
    const quantized = storage.quantize(weight, 0);
    const opened = await openModelBytes(linearDeclaration([], {}, ["w"]), [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes: quantized.bytes,
        encoding: quantized.encoding,
      },
      {
        graph: GRAPH_NAME,
        initializer: "b",
        bytes: new Uint8Array(bias.data.buffer.slice(0)),
        encoding: { codec: "f32" },
      },
    ]);
    const x = fill([2, 4], SIGNED);
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME);
    try {
      const diagnostics = session.diagnostics().storage;
      assertEquals(
        diagnostics.residentCompressedBytes,
        0,
        "a weight that is a graph output keeps not a single byte resident compressed",
      );
      assertEquals(diagnostics.hostExpandedBytes, 48, "CPU-expanded bytes (12 elements as f32)");
      const outputs = await session.run({ x });
      // The weights are constants independent of execution, so they are bit-identical to the rounded values
      assertEquals(outputs["w"].shape, [3, 4]);
      assertEquals([...outputs["w"].data], [...quantized.values]);
      // The computing side of the same run is as before (the values do not change on the expansion path either)
      const expected = applyReferenceOp(
        "linear",
        [x as RefTensor, refTensor([3, 4], quantized.values), bias as RefTensor],
        {},
        [2, 3],
      );
      const report = compareTensors(outputs["y"], expected, GEMM_TOLERANCE);
      assertEquals(report.pass, true, formatAllclose(report));
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  });
}
