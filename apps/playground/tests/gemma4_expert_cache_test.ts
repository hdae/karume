import { EXPERT_SPECULATION_DEPTH, FITTED_EXPERT_SLOTS } from "@karume/gemma4-app-support/settings";
import type { ExpertCacheReport } from "@karume/models/gemma";
import { describe, expect, it } from "vite-plus/test";

import {
  expertCacheFigures,
  expertCacheRequestText,
  expertCacheText,
  expertHostTierText,
} from "../src/models/gemma4/expert-cache.ts";
import { parseExpertSpeculationDepth, sameExpertSlots } from "../src/models/gemma4/settings.ts";

const MIB = 2 ** 20;
const GIB = 2 ** 30;
/** 64 slots in every one of 26B's 30 routed layers (`ExpertCacheReport.slots`). */
const SLOTS_64: readonly number[] = new Array<number>(30).fill(64);

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
  slots: SLOTS_64,
  committedRuns: 20,
  retries: 10,
  ...patch,
});

describe("sameExpertSlots", () => {
  it("compares lists by value", () => {
    expect(sameExpertSlots([96, 40], [96, 40])).toBe(true);
    expect(sameExpertSlots([...FITTED_EXPERT_SLOTS], FITTED_EXPERT_SLOTS)).toBe(true);
    expect(sameExpertSlots([96, 40], [40, 96])).toBe(false);
    expect(sameExpertSlots([64, 64], [64, 64, 64])).toBe(false);
  });

  it("compares counts and no cache by value, and never a count with a list", () => {
    expect(sameExpertSlots(64, 64)).toBe(true);
    expect(sameExpertSlots(64, 32)).toBe(false);
    expect(sameExpertSlots(undefined, undefined)).toBe(true);
    expect(sameExpertSlots(undefined, 64)).toBe(false);
    expect(sameExpertSlots([64], undefined)).toBe(false);
    expect(sameExpertSlots(64, [64])).toBe(false);
  });
});

describe("parseExpertSpeculationDepth", () => {
  it("reads an empty field as no self-speculation", () => {
    expect(parseExpertSpeculationDepth("")).toEqual({ ok: true, value: undefined });
    expect(parseExpertSpeculationDepth("  ")).toEqual({ ok: true, value: undefined });
  });

  it("accepts the bounds and the integers between them", () => {
    expect(parseExpertSpeculationDepth(String(EXPERT_SPECULATION_DEPTH.min))).toEqual({
      ok: true,
      value: 1,
    });
    expect(parseExpertSpeculationDepth(" 3 ")).toEqual({ ok: true, value: 3 });
    expect(parseExpertSpeculationDepth(String(EXPERT_SPECULATION_DEPTH.max))).toEqual({
      ok: true,
      value: 4,
    });
  });

  it("refuses a depth just outside the bounds and what is not a plain integer, naming the range", () => {
    for (const text of ["0", "5", "-1", "2.5", "1e0", "four", "4 drafts"]) {
      const parsed = parseExpertSpeculationDepth(text);
      expect(parsed.ok).toBe(false);
      if (parsed.ok) continue;
      expect(parsed.errors).toEqual([
        "Self-speculation depth must be an integer from 1 to 4, or empty for none (one run per decode token).",
      ]);
    }
  });
});

describe("expertCacheRequestText", () => {
  it("names the slots, then the host tier and the depth only when they were asked for", () => {
    expect(expertCacheRequestText({ slots: 64 })).toBe("expert cache 64 slots");
    expect(expertCacheRequestText({ slots: 64, hostBytes: Number.MAX_SAFE_INTEGER })).toBe(
      "expert cache 64 slots · host tier all",
    );
    expect(expertCacheRequestText({ slots: 64, depth: 4 })).toBe("expert cache 64 slots · depth 4");
    expect(
      expertCacheRequestText({ slots: 64, hostBytes: Number.MAX_SAFE_INTEGER, depth: 4 }),
    ).toBe("expert cache 64 slots · host tier all · depth 4");
  });

  it("names a list by its mean and range, and a list of one count as that count", () => {
    expect(expertCacheRequestText({ slots: [96, 40, 108, 64], depth: 4 })).toBe(
      "expert cache 77 slots per layer on average (40–108) · depth 4",
    );
    expect(expertCacheRequestText({ slots: [16, 17] })).toBe(
      "expert cache 16.5 slots per layer on average (16–17)",
    );
    expect(expertCacheRequestText({ slots: [32, 32, 32] })).toBe("expert cache 32 slots");
  });

  it("names the fitted allocation as fitted", () => {
    expect(
      expertCacheRequestText({ slots: FITTED_EXPERT_SLOTS, hostBytes: Number.MAX_SAFE_INTEGER }),
    ).toBe("expert cache fitted, 64 slots per layer on average (40–108) · host tier all");
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
      slots: SLOTS_64,
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

  it("has no self-speculation figures for a pipeline built without a depth", () => {
    expect(expertCacheFigures(report())).not.toHaveProperty("speculation");
  });

  it("divides the accepted drafts by the cycles when the pipeline was built with a depth", () => {
    const figures = expertCacheFigures(
      report({
        speculation: {
          depth: 4,
          cycles: 40,
          draftRuns: 150,
          acceptedDrafts: 120,
          verifyRetries: 3,
          partialCommits: 7,
        },
      }),
    );
    expect(figures.speculation).toEqual({
      cycles: 40,
      acceptedDraftsPerCycle: 3,
      partialCommits: 7,
      verifyRetries: 3,
    });
  });

  it("leaves out the accepted drafts per cycle before the first cycle", () => {
    const figures = expertCacheFigures(
      report({
        speculation: {
          depth: 4,
          cycles: 0,
          draftRuns: 0,
          acceptedDrafts: 0,
          verifyRetries: 0,
          partialCommits: 0,
        },
      }),
    );
    expect(figures.speculation).toEqual({ cycles: 0, partialCommits: 0, verifyRetries: 0 });
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

  it("names the mean and range of the slots when the layers have counts of their own", () => {
    expect(expertCacheText(report({ slots: [96, 40, 108, 64] }))).toMatch(
      /^77 slots per layer on average \(40–108\) · 20 committed runs · /,
    );
    expect(expertCacheText(report({ slots: [...FITTED_EXPERT_SLOTS] }))).toMatch(
      /^fitted, 64 slots per layer on average \(40–108\) · 20 committed runs · /,
    );
  });

  it("adds the self-speculation cycles after the committed runs when the pipeline was built with a depth", () => {
    expect(
      expertCacheText(
        report({
          speculation: {
            depth: 4,
            cycles: 40,
            draftRuns: 150,
            acceptedDrafts: 125,
            verifyRetries: 3,
            partialCommits: 7,
          },
        }),
      ),
    ).toBe(
      "64 slots · 20 committed runs · 0.50 retries / committed run · 40 cycles · 3.13 accepted drafts / cycle · 7 partial commits · 3 verify retries · 400 loaded (300 from the host tier, 100 from the distribution) · 150 evicted · 1280.0 MiB read · load 900 ms (2.25 ms / expert; read 600 ms, upload 250 ms) · host tier holds 1536 experts (12.00 GiB)",
    );
  });

  it("shows a dash for the accepted drafts per cycle before the first cycle", () => {
    expect(
      expertCacheText(
        report({
          speculation: {
            depth: 2,
            cycles: 0,
            draftRuns: 0,
            acceptedDrafts: 0,
            verifyRetries: 0,
            partialCommits: 0,
          },
        }),
      ),
    ).toContain(
      " · 0 cycles · — accepted drafts / cycle · 0 partial commits · 0 verify retries · ",
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
