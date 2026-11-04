/**
 * Gemma 4's decode run-ahead (`PipelineOptions.decodeAhead`) — the parts that need no GPU: the
 * option's acceptance, the on-GPU index expansion of the PLE's resident-id form, and the assembly
 * of the run-ahead in the greedy output. The token ids on a real GPU are
 * `packages/e2e/tests/e2e_gemma4_qat_mobile_decode_ahead_test.ts`.
 */
import { describe, expect, it } from "vite-plus/test";
import {
  type GpuContext,
  prepareContainer,
  type ResidentTensor,
  type Session,
} from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import { createGemmaGreedyOutput } from "../src/gemma/greedy-output.ts";
import { resolveGemma4DecodeAhead } from "../src/gemma/pipeline.ts";
import {
  buildGemma4PleExpandContainer,
  gemma4PleExpandGraph,
  type Gemma4PleResident,
} from "../src/gemma/ple-gpu.ts";
import { pleFixture } from "./helpers/ple-fixture.ts";

/** A model with PLE whose distribution suppresses no token (E2B / E4B). */
const PLE_MODEL = { ple: true, suppressTokens: [] };

const index = pleFixture({ storage: "i4", tokens: 6, layers: 3, dim: 16, embedScale: 16 }).index;

describe("decodeAhead", () => {
  describe("the option", () => {
    it("is 1 (off) when absent and accepts a depth with the GPU-resident PLE", () => {
      expect(resolveGemma4DecodeAhead("T", {}, PLE_MODEL)).toBe(1);
      expect(resolveGemma4DecodeAhead("T", { decodeAhead: 1 }, PLE_MODEL)).toBe(1);
      expect(
        resolveGemma4DecodeAhead("T", { decodeAhead: 4, pleResidency: "gpu" }, PLE_MODEL),
      ).toBe(4);
    });

    it("refuses a value that is not an integer ≥ 1 as an input error", () => {
      for (const decodeAhead of [0, -2, 2.5, Number.NaN, "4" as unknown as number]) {
        expect(() =>
          resolveGemma4DecodeAhead("T", { decodeAhead, pleResidency: "gpu" }, PLE_MODEL),
        ).toThrow(ModelInputError);
      }
    });

    it("refuses a depth ≥ 2 without the GPU-resident PLE, saying why", () => {
      for (const pleResidency of [undefined, "host"] as const) {
        expect(() =>
          resolveGemma4DecodeAhead(
            "T",
            {
              decodeAhead: 2,
              ...(pleResidency === undefined ? {} : { pleResidency }),
            },
            PLE_MODEL,
          ),
        ).toThrow(/needs pleResidency: "gpu".*on the GPU/);
      }
    });

    it("refuses a depth ≥ 2 next to speculative decoding", () => {
      expect(() =>
        resolveGemma4DecodeAhead("T", { decodeAhead: 4, speculative: {} }, PLE_MODEL),
      ).toThrow(ModelInputError);
      expect(() =>
        resolveGemma4DecodeAhead("T", { decodeAhead: 4, speculative: {} }, PLE_MODEL),
      ).toThrow("speculative");
    });

    it("refuses a depth ≥ 2 for a model without PLE, even with pleResidency unset", () => {
      // 12B: the run-ahead rides the GPU-resident PLE, which a PLE-free model does not have.
      const model = { ple: false, suppressTokens: [] };
      expect(resolveGemma4DecodeAhead("T", {}, model)).toBe(1);
      expect(() => resolveGemma4DecodeAhead("T", { decodeAhead: 2 }, model)).toThrow(
        ModelInputError,
      );
      expect(() => resolveGemma4DecodeAhead("T", { decodeAhead: 2 }, model)).toThrow(/has no PLE/);
    });

    it("refuses a depth ≥ 2 for a distribution that suppresses tokens", () => {
      // The GPU selection reads the raw maximum, which can be a suppressed id.
      const model = { ple: true, suppressTokens: [258882] };
      expect(resolveGemma4DecodeAhead("T", { pleResidency: "gpu" }, model)).toBe(1);
      expect(() =>
        resolveGemma4DecodeAhead("T", { decodeAhead: 4, pleResidency: "gpu" }, model),
      ).toThrow(/suppresses tokens/);
    });
  });

  describe("the index expansion of the PLE's resident-id form", () => {
    it("maps one token id to the gather's rows id × layers + layer with i32 mul and sub", () => {
      const graph = gemma4PleExpandGraph(index);
      expect(graph.nodes.map((node) => node.op)).toEqual(["mul", "sub"]);
      expect(graph.inputs).toEqual([{ name: "ple_token", dtype: "i32", shape: [1, 1, 1] }]);
      expect(graph.values[graph.outputs[0]]).toEqual({ dtype: "i32", shape: [1, 1, index.layers] });
    });

    it("carries the layer count and the negated layer offsets as i32 constants", async () => {
      const bound = buildGemma4PleExpandContainer(index);
      const words = async (name: string): Promise<number[]> => {
        const supply = bound.graphs.ple_expand.supplies.get(name);
        expect(supply?.encoding.codec).toBe("i32");
        const block = supply?.blocks[0];
        if (block === undefined) throw new Error(`no supply for '${name}'`);
        return [...new Int32Array((await bound.readBlock(block.id)).slice().buffer)];
      };
      expect(await words("ple_layers")).toEqual([3]);
      // id × 3 − (−layer) = id × 3 + layer for layer 0, 1, 2.
      expect(await words("ple_negated_layer")).toEqual([0, -1, -2]);
    });

    it("passes the runtime's capability check and its shape planning", () => {
      const prepared = prepareContainer(buildGemma4PleExpandContainer(index), "ple_expand");
      expect(prepared.estimate({ planBackingBudgetBytes: 0 }).peakAccountedBytes).toBeGreaterThan(
        0,
      );
    });
  });

  describe("the greedy output", () => {
    /**
     * Assembly touches neither the GPU nor the target (they are used from the first call), so
     * stand-ins of the right type suffice for these checks.
     */
    const gpu = {} as GpuContext;
    const target = {} as Session;
    const ple = (residentIds: boolean): Gemma4PleResident => ({
      inputName: "per_layer_inputs",
      idsName: "input_ids",
      extraBytes: 0,
      residentBytes: 0,
      enqueue: () => Promise.reject(new Error("not reached")),
      dispose: () => Promise.resolve(),
      ...(residentIds
        ? {
            residentIds: {
              ids: {} as ResidentTensor,
              enqueue: () => Promise.reject(new Error("not reached")),
            },
          }
        : {}),
    });

    it("builds the run-ahead only for a depth ≥ 2", () => {
      expect(createGemmaGreedyOutput(gpu, target, "logits", 32, ple(true)).ahead).toBeUndefined();
      const output = createGemmaGreedyOutput(gpu, target, "logits", 32, ple(true), 4);
      expect(output.ahead?.depth).toBe(4);
      // One value + index readback pair per step on top of the per-step greedy output.
      const base = createGemmaGreedyOutput(gpu, target, "logits", 32, ple(true)).extraBytes;
      expect(output.extraBytes).toBeGreaterThanOrEqual(base + 4 * 8);
    });

    it("refuses a depth ≥ 2 without the PLE's resident-id form", () => {
      expect(() => createGemmaGreedyOutput(gpu, target, "logits", 32, ple(false), 2)).toThrow(
        "resident-id form",
      );
      expect(() => createGemmaGreedyOutput(gpu, target, "logits", 32, undefined, 2)).toThrow(
        "resident-id form",
      );
    });
  });
});
