// Real-GPU check of conv3d with symbolic spatial lengths -> asymmetric pad (permute + a pad of the last axis) -> conv2d
// with stride 2 (ADR 0121 stage 9a — a shrunk version of the down sequence of chunk 0 of the Wan2.2 VAE encoder).
//
// The encoder's pre / post graphs keep the spatial lengths as the symbols h, w (makes the assets independent of the
// resolution — ADR 0038 §4). The runtime binds first and then plans with concrete shapes, so the conv shape rules accept
// symbols, but the encoder is the first to run conv on the GPU over symbolic space. Here **the same Session** receives
// bindings 3 times (size A -> size B -> size A), and the test checks that
//
// 1. each run's outputs (the conv3d exit and the exit after pad -> conv2d) are within the band of the CPU reference (the
//    same band as the conv parity tests)
// 2. the binding changes which conv3d variant runs (v4 if the output width 2w is a multiple of 4, scalar otherwise)
// 3. the returning run matches the first as Uint32 (the plan cache carries over neither the shape nor the variant of the
//    previous binding — at both exits)
//
// The encoder's full-size widths (8, 4 and 2 times the latent 80 / 44 / 16) are all multiples of 4, so the e2e exercises
// only v4 — switching the variant by binding is visible only here.
//
// NOTE: running the other nodes of pre / post (the space-to-depth reshape with symbolic division, slice / cat on the
// channel axis, the sum reduction of RMS norm, clamp_min / sqrt / div) and f16 weight storage under changing symbol
// bindings is left to the real-weight e2e (`packages/models/tests/e2e_wan_ti2v_vae_encoder_test.ts`) — on the default
// lane it rebinds the same Session to 3 sizes (1280×704 -> 704×1280 -> 256×160) and compares each binding against a
// golden.

import { assert, assertEquals } from "@std/assert";
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import { conv2dIgemmMTile } from "../src/kernels/conv2d.ts";
import { conv3dIgemmKey } from "../src/kernels/conv3d.ts";
import { allclose } from "../src/reference/allclose.ts";
import {
  referenceConv2d,
  referenceConv3d,
  referencePad,
  referencePermute,
  type RefTensor,
  refTensor,
} from "../src/reference/ops.ts";
import { createSessionFromContainer, type Session } from "../src/runtime/executor.ts";
import type { Tensor } from "../src/runtime/session-types.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

/** The band of the comparison with the CPU reference (the same as the conv2d / conv3d parity tests — the reference reduces in f64). */
const CPU_REFERENCE_TOLERANCE = { atol: 1e-5, rtol: 1e-5 } as const;

/** Channel counts (conv3d's kFlat = 4·1·3·3 = 36 is a multiple of 4 — whether it rides v4 is decided by the output width alone). */
const CHANNELS_IN = 4;
const CHANNELS_MID = 6;
const CHANNELS_OUT = 5;

const CONV3D_ATTRS = { stride: [1, 1, 1], padding: [0, 1, 1], dilation: [1, 1, 1], groups: 1 };
const CONV2D_ATTRS = { stride: [2, 2], padding: [0, 0], dilation: [1, 1], groups: 1 };
const PAD_RIGHT = { left: 0, right: 1 };
const TO_BATCH = [1, 0, 2, 3];
const SWAP_LAST = [0, 1, 3, 2];

/**
 * The encoder's down sequence (`LastSliceConv3d` -> `downsample` of the recipe `wan/vae_encoder_patch.py`):
 * `x [Cin,1,2h,2w]` -> conv3d (1 time frame, spatial pad 1) -> frames into the batch -> pad on the W side -> rotate the
 * H side to the last axis with permute and pad -> rotate back -> conv2d stride 2 -> `[1,Cout,h,w]`.
 */
const downGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["conv2d", "conv3d", "pad", "permute"] },
  symbols: ["h", "w"],
  inputs: [
    { name: "x", dtype: "f32", shape: [CHANNELS_IN, 1, "2h", "2w"] },
    { name: "w3", dtype: "f32", shape: [CHANNELS_MID, CHANNELS_IN, 1, 3, 3] },
    { name: "b3", dtype: "f32", shape: [CHANNELS_MID] },
    { name: "w2", dtype: "f32", shape: [CHANNELS_OUT, CHANNELS_MID, 3, 3] },
    { name: "b2", dtype: "f32", shape: [CHANNELS_OUT] },
  ],
  outputs: ["y3", "y"],
  initializers: {},
  values: {
    y3: { dtype: "f32", shape: [CHANNELS_MID, 1, "2h", "2w"] },
    p0: { dtype: "f32", shape: [1, CHANNELS_MID, "2h", "2w"] },
    q0: { dtype: "f32", shape: [1, CHANNELS_MID, "2h", "2w+1"] },
    p1: { dtype: "f32", shape: [1, CHANNELS_MID, "2w+1", "2h"] },
    q1: { dtype: "f32", shape: [1, CHANNELS_MID, "2w+1", "2h+1"] },
    p2: { dtype: "f32", shape: [1, CHANNELS_MID, "2h+1", "2w+1"] },
    y: { dtype: "f32", shape: [1, CHANNELS_OUT, "h", "w"] },
  },
  nodes: [
    { op: "conv3d", ins: ["x", "w3", "b3"], outs: ["y3"], attrs: CONV3D_ATTRS },
    { op: "permute", ins: ["y3"], outs: ["p0"], attrs: { dims: TO_BATCH } },
    { op: "pad", ins: ["p0"], outs: ["q0"], attrs: PAD_RIGHT },
    { op: "permute", ins: ["q0"], outs: ["p1"], attrs: { dims: SWAP_LAST } },
    { op: "pad", ins: ["p1"], outs: ["q1"], attrs: PAD_RIGHT },
    { op: "permute", ins: ["q1"], outs: ["p2"], attrs: { dims: SWAP_LAST } },
    { op: "conv2d", ins: ["p2", "w2", "b2"], outs: ["y"], attrs: CONV2D_ATTRS },
  ],
});

/** Deterministic data sequences (no random numbers — a failure would not reproduce). The bias is non-zero (avoids the signed-zero region). */
const SIGNED = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const WEIGHT = (i: number): number => (((i * 11) % 19) - 9) * 0.031;
const BIAS = (i: number): number => 0.125 + i * 0.25;

const WEIGHTS = {
  w3: fill([CHANNELS_MID, CHANNELS_IN, 1, 3, 3], WEIGHT),
  b3: fill([CHANNELS_MID], BIAS),
  w2: fill([CHANNELS_OUT, CHANNELS_MID, 3, 3], (i) => WEIGHT(i + 5)),
  b2: fill([CHANNELS_OUT], (i) => BIAS(i) - 0.5),
} as const;

const refOf = (tensor: Tensor): RefTensor => {
  if (tensor.dtype !== "f32") throw new Error(`accepts only f32 (${tensor.dtype})`);
  return refTensor(tensor.shape, tensor.data);
};

/** The CPU reference (builds the same sequence from reference ops). */
const referenceDown = (x: Tensor): { readonly y3: RefTensor; readonly y: RefTensor } => {
  const y3 = referenceConv3d(refOf(x), refOf(WEIGHTS.w3), refOf(WEIGHTS.b3), CONV3D_ATTRS);
  const widened = referencePad(referencePermute(y3, TO_BATCH), PAD_RIGHT);
  const heightened = referencePad(referencePermute(widened, SWAP_LAST), PAD_RIGHT);
  const y = referenceConv2d(
    referencePermute(heightened, SWAP_LAST),
    refOf(WEIGHTS.w2),
    refOf(WEIGHTS.b2),
    CONV2D_ATTRS,
  );
  return { y3, y };
};

/**
 * The sequence of bindings: output width 2w = 10 (not a multiple of 4 — conv3d is scalar) -> 2w = 12 (v4) -> back to
 * the first. Height and width get different values so that a symbol mix-up shows up in the shapes.
 */
const BINDINGS: readonly { readonly h: number; readonly w: number; readonly v4: boolean }[] = [
  { h: 3, w: 5, v4: false },
  { h: 4, w: 6, v4: true },
  { h: 3, w: 5, v4: false },
];

describe.skipIf(!GPU_AVAILABLE)(
  "conv3d over symbolic space -> asymmetric pad -> conv2d with stride 2 matches the CPU reference over 3 bindings of the same Session, " +
    "the binding switches the conv3d variant, and the returning run is bit-identical to the first (real GPU)",
  () => {
    let gpu: Awaited<ReturnType<typeof acquireGpu>> | undefined;
    let session: Session;
    const mTile = conv2dIgemmMTile(CHANNELS_MID);
    const outputsByRun: { readonly y3: Float32Array; readonly y: Float32Array }[] = [];

    beforeAll(async () => {
      gpu = await acquireGpu();
      session = await createSessionFromContainer(
        gpu,
        await openModelBytes(downGraph(), []),
        GRAPH_NAME,
      );
    });

    afterAll(async () => {
      // A failed beforeAll leaves session unassigned; the GPU is released either way.
      await session?.dispose();
      gpu?.destroy();
    });

    for (const [index, { h, w, v4 }] of BINDINGS.entries()) {
      it(`run ${index + 1}: h = ${h}, w = ${w} (conv3d is ${v4 ? "v4" : "scalar"})`, async () => {
        const x = fill([CHANNELS_IN, 1, 2 * h, 2 * w], (i) => SIGNED(i + h * 31 + w));
        const outputs = await session.run({ x, ...WEIGHTS });
        const reference = referenceDown(x);
        assertEquals(
          outputs["y3"].shape,
          [CHANNELS_MID, 1, 2 * h, 2 * w],
          "shape of the conv3d exit",
        );
        assertEquals(outputs["y"].shape, [1, CHANNELS_OUT, h, w], "shape of the conv2d exit");
        for (const [name, expected] of [
          ["y3", reference.y3],
          ["y", reference.y],
        ] as const) {
          const got = outputs[name].data;
          if (!(got instanceof Float32Array) || !(expected.data instanceof Float32Array)) {
            throw new Error(`${name}: not f32`);
          }
          const report = allclose(got, expected.data, CPU_REFERENCE_TOLERANCE);
          assert(
            report.pass,
            `${name}: outside the band of the CPU reference (maxAbs ${report.maxAbsError}, maxRel ${report.maxRelError})`,
          );
          // The output is not constant (the check has not become vacuous).
          assert(new Set(got).size > 1, `${name}: the output is constant`);
        }
        const conv3dKeys = requireCensus(session.diagnostics().lastRunPipelines, `run ${index + 1}`)
          .map((row) => row.key)
          .filter((key) => key.startsWith("conv3d:"));
        assertEquals(
          conv3dKeys,
          [conv3dIgemmKey("f32", v4, mTile)],
          "the conv3d variant exercised",
        );
        const [y3, y] = [outputs["y3"].data, outputs["y"].data];
        if (!(y3 instanceof Float32Array) || !(y instanceof Float32Array)) {
          throw new Error("y3 / y is not f32");
        }
        outputsByRun.push({ y3, y });
      });
    }

    it("the returning run (run 3) matches the first as Uint32 (the conv3d exit y3 and the final exit y)", () => {
      const [first, , back] = outputsByRun;
      assert(first !== undefined && back !== undefined, "no outputs of run 1 / 3");
      const bits = (values: Float32Array): number[] => [
        ...new Uint32Array(values.buffer, values.byteOffset, values.length),
      ];
      assertEquals(
        bits(back.y3),
        bits(first.y3),
        "y3 (the conv3d variant switches scalar -> v4 -> scalar)",
      );
      assertEquals(bits(back.y), bits(first.y), "y");
    });
  },
);
