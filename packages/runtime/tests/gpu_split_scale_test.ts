/**
 * Split scales on the GPU — a container whose scale is split on the weight's pieces (ADR 0108
 * addendum 15) builds a Session with **the same output bytes** as the same weights with the scale in
 * one block. The runtime is handed one contiguous scale either way (`readBlock` of the joined id), so
 * both residency paths must not see a difference: compressed residency (i4 linear — the scale is
 * written to the GPU whole with piece 1) and CPU decoding (i4 consumed by matmul — the scale is
 * carried over the piece list and sliced per piece).
 */

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { openContainer } from "../src/format/container/open.ts";
import { parseIrDeclaration } from "../src/format/ir.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { type ModelInput, writeModelContainer } from "./helpers/container-write.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { quantizeI4 } from "./helpers/i4.ts";
import { f32Bytes, fill } from "./helpers/model-fixture.ts";

const M = 4;
const ROWS = 32;
const COLS = 32;
const GROUP = 16;

/**
 * One i4 weight `w` [32, 32] (group 16) consumed by `op`: `linear` keeps it compressed and resident,
 * `matmul` (not a weight slot) decodes it on the CPU.
 */
const model = (op: "linear" | "matmul"): ModelInput => {
  const w = quantizeI4(
    fill([ROWS, COLS], (i) => Math.sin(i * 0.37) * 0.5).data as Float32Array<ArrayBuffer>,
    [ROWS, COLS],
    GROUP,
  );
  const ins = op === "linear" ? ["x", "w", "b"] : ["x", "w"];
  return {
    graphs: {
      main: parseIrDeclaration(
        JSON.stringify({
          format: "karume-ir",
          version: 2,
          requires: { ops: [op] },
          symbols: [],
          inputs: [{ name: "x", dtype: "f32", shape: [M, op === "linear" ? COLS : ROWS] }],
          outputs: ["y"],
          initializers: op === "linear" ? { w: {}, b: {} } : { w: {} },
          values: {
            w: { dtype: "f32", shape: [ROWS, COLS] },
            ...(op === "linear" ? { b: { dtype: "f32", shape: [ROWS] } } : {}),
            y: { dtype: "f32", shape: [M, op === "linear" ? ROWS : COLS] },
          },
          nodes: [{ op, ins, outs: ["y"], attrs: {} }],
        }),
      ),
    },
    consts: [],
    weights: [
      {
        graph: "main",
        initializer: "w",
        bytes: w.bytes,
        encoding: {
          codec: "int4-sym-g",
          groupSize: GROUP,
          scale: { bytes: f32Bytes(w.scale), dtype: "f32" },
        },
      },
      ...(op === "linear"
        ? [
            {
              graph: "main",
              initializer: "b",
              bytes: f32Bytes(Array.from({ length: ROWS }, (_, i) => (i % 3) * 0.05)),
              encoding: { codec: "f32" as const },
            },
          ]
        : []),
    ],
    assets: [],
    provenance: { license: "test" },
  };
};

/**
 * 512 B of i4 data → 4 pieces at a 128 B block limit; its 256 B scale (32 rows × 2 groups × 4 B) is
 * over the limit, so it is split into 4 pieces of 64 B.
 */
const SPLIT = { partBytes: 1024, blockBytes: 128 } as const;

describe.skipIf(!GPU_AVAILABLE)("split scale session", () => {
  for (const [op, path] of [
    ["linear", "compressed residency"],
    ["matmul", "CPU decoding"],
  ] as const) {
    it(`gives the output bytes of a one-block scale (${op} — ${path})`, async () => {
      const input = model(op);
      const split = await openContainer({
        kind: "parts",
        parts: [...(await writeModelContainer(input, SPLIT)).parts],
      });
      const whole = await openContainer({
        kind: "parts",
        parts: [...(await writeModelContainer(input)).parts],
      });
      // Preconditions: the split container really splits both the weight and its scale, and the
      // reference has neither (otherwise the A/B compares a form with itself).
      assertEquals(split.graphs["main"].supplies.get("w")?.scale?.pieces?.length, 4);
      assertEquals(split.graphs["main"].supplies.get("w")?.blocks.length, 4);
      assertEquals(whole.graphs["main"].supplies.get("w")?.scale?.pieces, undefined);
      assertEquals(whole.graphs["main"].supplies.get("w")?.blocks.length, 1);

      const x = fill([M, op === "linear" ? COLS : ROWS], (i) => ((i * 7) % 11) / 11 - 0.5);
      const gpu = await acquireGpu();
      try {
        const outputs: Uint8Array[] = [];
        const resident: number[] = [];
        for (const container of [whole, split]) {
          const session = await createSessionFromContainer(gpu, container, "main");
          try {
            const y = (await session.run({ x }))["y"].data as Float32Array<ArrayBuffer>;
            outputs.push(new Uint8Array(y.buffer.slice(0)));
            resident.push(session.diagnostics().storage.residentCompressedBytes);
          } finally {
            await session.dispose();
          }
        }

        assertEquals(outputs[1], outputs[0]);
        assertEquals(resident[1], resident[0]);
        // The residency path is the one this case is named after.
        assert(op === "linear" ? resident[0] > 0 : resident[0] === 0, `resident ${resident[0]}`);
      } finally {
        gpu.destroy();
      }
    });
  }
});
