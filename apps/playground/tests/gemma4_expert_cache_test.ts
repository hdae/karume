import type { ExpertCacheReport } from "@karume/models/gemma";
import { describe, expect, it } from "vite-plus/test";

import {
  expertCacheFigures,
  expertCacheText,
  expertHostTierText,
} from "../src/models/gemma4/expert-cache.ts";
import {
  EXPERT_CACHE_SLOTS,
  EXPERT_HOST_TIER_ALL,
  hasExpertCache,
  parseExpertCacheSlots,
  parseExpertHostTier,
} from "../src/models/gemma4/settings.ts";

const MIB = 2 ** 20;
const GIB = 2 ** 30;

const report = (patch: Partial<ExpertCacheReport> = {}): ExpertCacheReport => ({
  runs: 30,
  missedRuns: 10,
  loads: 10,
  loadedExperts: 400,
  evictedExperts: 150,
  hostExperts: 300,
  containerExperts: 100,
  bytesRead: 1280 * MIB,
  bytesUploaded: 5120 * MIB,
  loadMs: 900,
  readMs: 600,
  uploadMs: 250,
  hostHeldExperts: 1536,
  hostHeldBytes: 12 * GIB,
  slots: 64,
  committedRuns: 20,
  retries: 10,
  ...patch,
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

  it("refuses what is not a plain integer, naming the range", () => {
    for (const text of ["64.5", "-32", "1e2", "sixty", "32 slots"]) {
      const parsed = parseExpertCacheSlots(text);
      expect(parsed.ok).toBe(false);
      if (parsed.ok) continue;
      expect(parsed.errors).toEqual([
        "Expert cache slots must be an integer from 16 to 128, or empty for none (every expert resident).",
      ]);
    }
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

describe("expertHostTierText", () => {
  it("names every expert as all and a budget in GiB", () => {
    expect(expertHostTierText(Number.MAX_SAFE_INTEGER)).toBe("host tier all");
    expect(expertHostTierText(12 * GIB)).toBe("host tier 12.00 GiB");
  });
});

describe("expertCacheFigures", () => {
  it("divides the retries by the committed runs and the load time by the loaded experts", () => {
    expect(expertCacheFigures(report())).toEqual({
      slots: 64,
      committedRuns: 20,
      retriesPerCommittedRun: 0.5,
      loadedExperts: 400,
      hostExperts: 300,
      containerExperts: 100,
      evictedExperts: 150,
      readMiB: 1280,
      loadMs: 900,
      loadMsPerExpert: 2.25,
      readMs: 600,
      uploadMs: 250,
      hostHeldExperts: 1536,
      hostHeldGiB: 12,
    });
  });

  it("leaves out a ratio with nothing to divide by (before the first turn, a run that missed nothing)", () => {
    const figures = expertCacheFigures(
      report({ committedRuns: 0, retries: 0, loadedExperts: 0, bytesRead: 0, loadMs: 0 }),
    );
    expect(figures.retriesPerCommittedRun).toBeUndefined();
    expect(figures.loadMsPerExpert).toBeUndefined();
    expect(figures.readMiB).toBe(0);
  });
});

describe("expertCacheText", () => {
  it("shows every figure of the status row", () => {
    expect(expertCacheText(report())).toBe(
      "64 slots · 20 committed runs · 0.50 retries / committed run · 400 loaded (300 from the host tier, 100 from the distribution) · 150 evicted · 1280.0 MiB read · load 900 ms (2.25 ms / expert; read 600 ms, upload 250 ms) · host tier holds 1536 experts (12.00 GiB)",
    );
  });

  it("shows a dash for a ratio over zero", () => {
    expect(
      expertCacheText(
        report({
          committedRuns: 0,
          retries: 0,
          loadedExperts: 0,
          hostExperts: 0,
          containerExperts: 0,
          loadMs: 0,
          readMs: 0,
          uploadMs: 0,
        }),
      ),
    ).toBe(
      "64 slots · 0 committed runs · — retries / committed run · 0 loaded (0 from the host tier, 0 from the distribution) · 150 evicted · 1280.0 MiB read · load 0 ms (— ms / expert; read 0 ms, upload 0 ms) · host tier holds 1536 experts (12.00 GiB)",
    );
  });
});
