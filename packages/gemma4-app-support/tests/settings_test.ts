import { parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import {
  EXPERT_CACHE_FITTED,
  EXPERT_CACHE_SLOTS,
  EXPERT_HOST_TIER_ALL,
  EXPERT_SPECULATION_DEPTH,
  entryPointOf,
  FITTED_EXPERT_SLOTS,
  hasExpertCache,
  isQatModel,
  modelConfig,
  parseExpertCacheSlots,
  parseExpertHostTier,
} from "../settings.ts";

const GIB = 2 ** 30;

const hex = (character: string): string => character.repeat(64);

/** A `karume/5` manifest shaped like a Gemma 4 distribution: the one model `e2b`. */
const MANIFEST = parseManifest(
  JSON.stringify({
    format: "karume/5",
    generator: "test",
    defaultModel: "e2b",
    models: {
      e2b: {
        pipeline: "gemma4/1",
        assets: {},
        weights: {
          model: {
            i4: {
              container: {
                descriptor: {
                  graph: { length: 8, sha256: hex("d") },
                  model: { length: 8, sha256: hex("e") },
                },
                parts: [
                  { path: "e2b/model.krm-00001", size: 40, sha256: hex("a") },
                  { path: "e2b/model.krm-00002", size: 8, sha256: hex("b") },
                ],
              },
            },
          },
        },
        quants: { i4: { weights: { model: "i4" }, session: {} } },
        defaultQuant: "i4",
        pipelineConfig: {
          chunkLength: 768,
          maxChunkLength: 768,
          maxPosition: 131072,
          capacity: 4096,
          rope: {
            sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
            full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
          },
          sampler: { temperature: 1, topK: 64, topP: 0.95 },
        },
      },
    },
  }),
);

describe("model keys", () => {
  it("accepts only the models gemma4QatMobile.Pipeline takes", () => {
    expect(isQatModel("e2b")).toBe(true);
    expect(isQatModel("e4b")).toBe(true);
    expect(isQatModel("12b")).toBe(false);
  });
});

describe("entryPointOf", () => {
  it("builds only the mobile QAT mount with the mobile entry point", () => {
    expect(entryPointOf("gemma4")).toBe("gemma.Pipeline");
    expect(entryPointOf("gemma4-qat")).toBe("gemma.Pipeline");
    expect(entryPointOf("gemma4-qat-mobile")).toBe("gemma4QatMobile.Pipeline");
  });
});

describe("modelConfig", () => {
  it("reads the validated pipelineConfig of the model", () => {
    const config = modelConfig(MANIFEST, "e2b");
    expect(config.capacity).toBe(4096);
    expect(config.sampler).toEqual({ temperature: 1, topK: 64, topP: 0.95 });
  });

  it("fails loudly for a model the manifest does not have", () => {
    expect(() => modelConfig(MANIFEST, "e4b")).toThrow(/no model 'e4b'/);
  });
});

describe("hasExpertCache", () => {
  it("offers the cache for 26B of the q4_0 QAT line", () => {
    expect(hasExpertCache({ mount: "gemma4-qat", model: "26b" })).toBe(true);
  });

  it("does not offer it for the dense models of any mount", () => {
    expect(hasExpertCache({ mount: "gemma4-qat", model: "12b" })).toBe(false);
    expect(hasExpertCache({ mount: "gemma4-qat", model: "e4b" })).toBe(false);
    expect(hasExpertCache({ mount: "gemma4", model: "e2b" })).toBe(false);
    expect(hasExpertCache({ mount: "gemma4-qat-mobile", model: "26b" })).toBe(false);
  });
});

describe("parseExpertCacheSlots", () => {
  it("reads an empty field as no cache", () => {
    expect(parseExpertCacheSlots("")).toEqual({ ok: true, value: undefined });
    expect(parseExpertCacheSlots("   ")).toEqual({ ok: true, value: undefined });
  });

  it("accepts the bounds and the integers between them", () => {
    expect(parseExpertCacheSlots(String(EXPERT_CACHE_SLOTS.min))).toEqual({ ok: true, value: 16 });
    expect(parseExpertCacheSlots(" 64 ")).toEqual({ ok: true, value: 64 });
    expect(parseExpertCacheSlots(String(EXPERT_CACHE_SLOTS.max))).toEqual({
      ok: true,
      value: 128,
    });
  });

  it("refuses a count just outside the bounds", () => {
    expect(parseExpertCacheSlots("15").ok).toBe(false);
    expect(parseExpertCacheSlots("129").ok).toBe(false);
    expect(parseExpertCacheSlots("0").ok).toBe(false);
  });

  it("refuses what is not a plain integer, naming the forms", () => {
    for (const text of ["64.5", "-32", "1e2", "sixty", "32 slots", "fitted64"]) {
      const parsed = parseExpertCacheSlots(text);
      expect(parsed.ok).toBe(false);
      if (parsed.ok) continue;
      expect(parsed.errors).toEqual([
        'Expert cache slots must be an integer from 16 to 128, a comma-separated list of such integers (one per routed layer, layer 0 first), "fitted" for the fitted allocation of 26B, or empty for none (every expert resident).',
      ]);
    }
  });

  it("reads a comma-separated list as one count per layer, in order, spaces allowed", () => {
    expect(parseExpertCacheSlots("96, 40,108 ,64")).toEqual({ ok: true, value: [96, 40, 108, 64] });
    expect(parseExpertCacheSlots("16,128")).toEqual({ ok: true, value: [16, 128] });
  });

  it("refuses a list naming every layer whose count is wrong, an empty entry included", () => {
    const parsed = parseExpertCacheSlots("64, 15, 64,, 129, x");
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.errors).toEqual([
      'Expert cache slots must be an integer from 16 to 128, a comma-separated list of such integers (one per routed layer, layer 0 first), "fitted" for the fitted allocation of 26B, or empty for none (every expert resident).',
      'Layer 1 ("15") is not.',
      'Layer 3 ("") is not.',
      'Layer 4 ("129") is not.',
      'Layer 5 ("x") is not.',
    ]);
    expect(parseExpertCacheSlots("64,").ok).toBe(false);
  });

  it("reads the preset word, in any case, as the fitted allocation", () => {
    for (const text of [EXPERT_CACHE_FITTED, " Fitted ", "FITTED"]) {
      expect(parseExpertCacheSlots(text)).toEqual({ ok: true, value: FITTED_EXPERT_SLOTS });
    }
  });
});

describe("FITTED_EXPERT_SLOTS", () => {
  it("has one count per routed layer of 26B, at the 64 slots' total VRAM", () => {
    expect(FITTED_EXPERT_SLOTS).toHaveLength(30);
    expect(FITTED_EXPERT_SLOTS.reduce((sum, count) => sum + count, 0)).toBe(30 * 64);
  });

  it("keeps every layer in the range the slots' text form accepts", () => {
    for (const count of FITTED_EXPERT_SLOTS) {
      expect(parseExpertCacheSlots(String(count))).toEqual({ ok: true, value: count });
    }
  });

  it("leaves the fewest layer room for a verify at the largest depth (depth + 1 ≤ ⌊slots / 8⌋)", () => {
    expect(EXPERT_SPECULATION_DEPTH.max + 1).toBeLessThanOrEqual(
      Math.floor(Math.min(...FITTED_EXPERT_SLOTS) / 8),
    );
  });
});

describe("parseExpertHostTier", () => {
  it("reads an empty field as no host tier", () => {
    expect(parseExpertHostTier("")).toEqual({ ok: true, value: undefined });
    expect(parseExpertHostTier("  ")).toEqual({ ok: true, value: undefined });
  });

  it("reads the word for every expert, in any case, as the largest budget", () => {
    for (const text of [EXPERT_HOST_TIER_ALL, " ALL ", "All"]) {
      expect(parseExpertHostTier(text)).toEqual({ ok: true, value: Number.MAX_SAFE_INTEGER });
    }
  });

  it("reads a positive number of GiB as whole bytes", () => {
    expect(parseExpertHostTier("12")).toEqual({ ok: true, value: 12 * GIB });
    expect(parseExpertHostTier(" 0.5 ")).toEqual({ ok: true, value: GIB / 2 });
    // Taken down to whole bytes, never up past what was asked.
    expect(parseExpertHostTier("0.3")).toEqual({ ok: true, value: Math.floor(0.3 * GIB) });
  });

  it("refuses zero, a negative or non-decimal number and other words, naming the forms", () => {
    for (const text of ["0", "0.0", "-1", "1e2", ".5", "1.", "twelve", "12 GiB", "everything"]) {
      const parsed = parseExpertHostTier(text);
      expect(parsed.ok).toBe(false);
      if (parsed.ok) continue;
      expect(parsed.errors).toEqual([
        'Host tier must be a positive number of GiB, "all" for every expert, or empty for none (every miss reads the distribution).',
      ]);
    }
  });

  it("refuses a budget beyond a safe integer of bytes", () => {
    expect(parseExpertHostTier("9000000").ok).toBe(false);
  });
});
