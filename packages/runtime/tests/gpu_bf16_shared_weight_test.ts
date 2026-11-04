// Borrowing a bf16-resident weight (ADR 0131): the borrower's expected residency class is derived
// from its own consumption and the lender's codec, and for bf16 it MUST follow the lender's plan
// (resident `bf16` when eligible), the same rule as f16 — a drafter's `lm_head` borrows the bf16
// target's table this way.
//
// MUST: the weight's row length and element count are odd (bf16 packs two elements per u32 word,
// the same traps as f16 — see gpu_f16_i8_weights_test.ts).
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { quantizeBf16 } from "./helpers/bf16.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";

const N = 5;
const K = 7;

const linearDeclaration = (shared: boolean): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [1, K] }],
  outputs: ["y"],
  initializers: { w: shared ? { shared: true } : {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [N, K] },
    b: { dtype: "f32", shape: [N] },
    y: { dtype: "f32", shape: [1, N] },
  },
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});

const bits = (tensor: Tensor): Uint32Array<ArrayBuffer> =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength / 4);

test.skipIf(!GPU_AVAILABLE)(
  "a linear borrows a bf16-resident lender's weight and computes the lender's result bit for bit",
  async () => {
    const gpu = await acquireGpu();
    const weight = quantizeBf16(Array.from({ length: N * K }, (_, i) => ((i % 11) - 5) * 0.37));
    const bias: TensorInput = {
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: f32Bytes(Array.from({ length: N }, (_, i) => (i - 2) * 0.25)),
      encoding: { codec: "f32" },
    };
    try {
      const lender = await createSessionFromContainer(
        gpu,
        await openModelBytes(linearDeclaration(false), [
          bias,
          { graph: GRAPH_NAME, initializer: "w", bytes: weight.bytes, encoding: { codec: "bf16" } },
        ]),
        GRAPH_NAME,
      );
      try {
        // The lender really holds the weight in the bf16 resident class (its 35 values, 2 bytes
        // each, padded to 4-byte alignment) and expanded nothing to f32 — otherwise both sides
        // would expand and match without exercising the bf16-resident borrow.
        const lent = lender.diagnostics().storage;
        assertEquals(lent.residentCompressedBytes, Math.ceil((N * K * 2) / 4) * 4);
        assertEquals(lent.hostExpandedBytes, 0);
        const borrower = await createSessionFromContainer(
          gpu,
          await openModelBytes(linearDeclaration(true), [bias]),
          GRAPH_NAME,
          { sharedWeights: { w: lender.exportWeight("w") } },
        );
        try {
          const x = {
            dtype: "f32",
            shape: [1, K],
            data: Float32Array.from({ length: K }, (_, i) => Math.sin(i + 1) * 0.5),
          } satisfies Tensor;
          assertEquals(bits((await borrower.run({ x })).y), bits((await lender.run({ x })).y));
          assertEquals(borrower.diagnostics().storage.residentCompressedBytes, 0);
        } finally {
          await borrower.dispose();
        }
      } finally {
        await lender.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);
