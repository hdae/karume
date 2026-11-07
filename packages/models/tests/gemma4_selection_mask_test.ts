/**
 * The GPU selection of a distribution that suppresses tokens: on a real GPU it returns what the
 * host's masked argmax returns (`createSampler(...).next`), at the real vocabulary, where the
 * suppressed ids are the raw maximum, on a tie, and next to −0. Without the mask the same logits
 * return the suppressed id, so the comparison can fail.
 */
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { acquireGpu, type GpuContext, type Session } from "@karume/runtime";
import { gemmaSelectionModel } from "../src/gemma/greedy-output.ts";
import { createSampler } from "../src/generation/sampler.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const VOCABULARY = 262144;
const SUPPRESSED = [258882, 258883] as const;

/** Row 0 is the baseline every case raises from. */
const logitsWith = (entries: readonly (readonly [number, number])[]): Float32Array<ArrayBuffer> => {
  const logits = new Float32Array(VOCABULARY).fill(-5);
  for (const [token, value] of entries) logits[token] = value;
  return logits;
};

const inputOf = (logits: Float32Array<ArrayBuffer>) => ({
  logits: { dtype: "f32" as const, shape: [1, 1, VOCABULARY], data: logits },
});

const CASES: readonly {
  readonly name: string;
  readonly logits: Float32Array<ArrayBuffer>;
  readonly raw: number;
}[] = [
  {
    name: "a suppressed id is the raw maximum",
    logits: logitsWith([
      [258882, 40],
      [17, 3],
    ]),
    raw: 258882,
  },
  {
    name: "both suppressed ids lie above a tie, which keeps the smaller id",
    logits: logitsWith([
      [258882, 40],
      [258883, 41],
      [200000, 7],
      [90000, 7],
    ]),
    raw: 258883,
  },
  {
    name: "the maximum after the mask is −0 beside +0",
    logits: new Float32Array(VOCABULARY).fill(-1).fill(0, 258882, 258884).fill(-0, 12, 13),
    raw: 12,
  },
  {
    name: "no suppressed id is near the maximum",
    logits: logitsWith([[262143, 9]]),
    raw: 262143,
  },
];

describe.skipIf(!GPU_AVAILABLE)("the masked GPU selection (real GPU)", () => {
  let gpu: GpuContext;
  let masked: Session;
  let raw: Session;

  beforeAll(async () => {
    gpu = await acquireGpu();
    masked = await gemmaSelectionModel(VOCABULARY, false, SUPPRESSED).createContainerSession(gpu);
    raw = await gemmaSelectionModel(VOCABULARY, false, []).createContainerSession(gpu);
  });

  afterAll(async () => {
    await masked.dispose();
    await raw.dispose();
    gpu.destroy();
  });

  for (const { name, logits, raw: rawMaximum } of CASES) {
    it(`matches the host's masked argmax: ${name}`, async () => {
      const host = createSampler({}, SUPPRESSED).next(logits, []);
      const out = await masked.run(inputOf(logits));
      assertEquals(out.index.data[0], host);
      const unmasked = await raw.run(inputOf(logits));
      assertEquals(unmasked.index.data[0], rawMaximum, "the selection without it");
    });
  }

  it("passes a finite maximum to the sampler's check, which accepts it", async () => {
    const logits = logitsWith([[258882, 40]]);
    const out = await masked.run(inputOf(logits));
    const value = out.value.data[0];
    const index = out.index.data[0];
    assertEquals(createSampler({}, SUPPRESSED, true).greedy?.(value, index), 0);
  });
});
