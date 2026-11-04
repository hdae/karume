// Checks the pure-function part of the self A/B check (ADR 0110 decision 5) — no GPU and no assets;
// every input is synthetic.
//
// Every reason the real-GPU checks (e2e_*_ab_test.ts) turn red goes through the verdicts here, so
// what this file pins is that the check does not pass vacuously: identical arrays are red on the
// floor, non-finite values are red on the ceiling side, an underived band prints the measurement
// and is red, and zero or two reference-partner candidates throw.

import { assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { linearKey } from "../../runtime/src/kernels/linear.ts";
import { linearI8a8Key } from "../../runtime/src/kernels/linear-i8a8.ts";
import {
  type AbBand,
  assertBitIdentical,
  captureThenAbort,
  describeBundleCensus,
  judgeAb,
  maxAbs,
  measureAb,
  overrideQuantSession,
  type QuantTable,
  referencePartnerOf,
  relRms,
} from "./helpers/ab-gate.ts";

const I8_DIT = { text_encoder: "f16", transformer: "i8", vae_decoder: "f16" };
const F16_DIT = { text_encoder: "f16", transformer: "f16", vae_decoder: "f16" };

/**
 * The presets laid out as in the anima distribution (a reference partner, practical presets and
 * presets with other weights).
 */
const ANIMA_LIKE: QuantTable = {
  models: {
    turbo: {
      quants: {
        f16: { weights: F16_DIT, session: {} },
        "f16+dit8": { weights: I8_DIT, session: {} },
        "f16+dit8-a8": { weights: I8_DIT, session: { linearCompute: "a8" } },
        "f16+dit8-a8-attn8-s16": {
          weights: I8_DIT,
          session: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "f16" },
        },
        "f16-c16": {
          weights: F16_DIT,
          session: { linearCompute: "f16", attentionCompute: "f16" },
          gpuFeatures: { shaderF16: true },
        },
      },
    },
  },
};

const BAND: AbBand = { floor: 1e-3, ceiling: 0.1 };

describe("referencePartnerOf: derives a practical preset's reference partner from the manifest", () => {
  it("returns the one preset with the same weights and an empty session when there is exactly one", () => {
    assertEquals(referencePartnerOf(ANIMA_LIKE, "turbo", "f16+dit8-a8-attn8-s16"), "f16+dit8");
    assertEquals(referencePartnerOf(ANIMA_LIKE, "turbo", "f16+dit8-a8"), "f16+dit8");
  });

  it("pairs a reference partner whose gpuFeatures are a subset of the practical preset's (f16-c16 ↔ f16)", () => {
    assertEquals(referencePartnerOf(ANIMA_LIKE, "turbo", "f16-c16"), "f16");
  });

  it("does not pair when only the reference partner needs a feature (zero candidates, throws)", () => {
    const table: QuantTable = {
      models: {
        m: {
          quants: {
            ref: { weights: I8_DIT, session: {}, gpuFeatures: { shaderF16: true } },
            fast: { weights: I8_DIT, session: { linearCompute: "a8" } },
          },
        },
      },
    };
    assertThrows(() => referencePartnerOf(table, "m", "fast"), Error, "number 0 (none)");
  });

  it("throws on zero candidates (no reference partner with the same weights is shipped)", () => {
    const table: QuantTable = {
      models: {
        m: {
          quants: {
            f16: { weights: F16_DIT, session: {} },
            "i8-a8": { weights: I8_DIT, session: { linearCompute: "a8" } },
          },
        },
      },
    };
    assertThrows(() => referencePartnerOf(table, "m", "i8-a8"), Error, "number 0");
  });

  it("throws on two candidates (the meaning of the floor depends on which one is compared)", () => {
    const table: QuantTable = {
      models: {
        m: {
          quants: {
            i8: { weights: I8_DIT, session: {} },
            "i8-copy": { weights: I8_DIT, session: {} },
            "i8-a8": { weights: I8_DIT, session: { linearCompute: "a8" } },
          },
        },
      },
    };
    assertThrows(() => referencePartnerOf(table, "m", "i8-a8"), Error, "number 2 (i8 / i8-copy)");
  });

  it("throws when a preset with an empty session is passed as the practical preset", () => {
    assertThrows(
      () => referencePartnerOf(ANIMA_LIKE, "turbo", "f16+dit8"),
      Error,
      "has an empty session",
    );
  });

  it("throws on an unknown model / quant", () => {
    assertThrows(() => referencePartnerOf(ANIMA_LIKE, "base", "f16"), Error, "has no model 'base'");
    assertThrows(() => referencePartnerOf(ANIMA_LIKE, "turbo", "i4"), Error, "has no quant 'i4'");
  });
});

describe("relRms / maxAbs", () => {
  it("returns ‖p − r‖₂ / ‖r‖₂ and the largest absolute difference", () => {
    const reference = new Float32Array([3, 4]);
    const practical = new Float32Array([3, 3]);
    assertEquals(relRms(practical, reference), 1 / 5);
    assertEquals(maxAbs(practical, reference), 1);
  });

  it("does not let NaN through as 0", () => {
    const reference = new Float32Array([1, 2, 3]);
    const practical = new Float32Array([1, Number.NaN, 3]);
    assertEquals(Number.isNaN(relRms(practical, reference)), true);
    assertEquals(Number.isNaN(maxAbs(practical, reference)), true);
  });

  it("throws when the element counts differ", () => {
    assertThrows(
      () => relRms(new Float32Array(2), new Float32Array(3)),
      Error,
      "the observation points differ in element count",
    );
  });
});

describe("judgeAb: the A/B verdict", () => {
  const reference = new Float32Array([1, -2, 3, -4]);

  it("is red on the floor for identical arrays, band or no band (the bundle did not reach the Session)", () => {
    const measured = measureAb(new Float32Array(reference), reference);
    assertThrows(
      () => judgeAb(measured, BAND, "case"),
      Error,
      "floor failure — the practical preset does not differ from the reference partner by a single bit",
    );
    assertThrows(() => judgeAb(measured, undefined, "case"), Error, "floor failure");
  });

  it("is red on the ceiling side for non-finite values", () => {
    const measured = measureAb(new Float32Array([1, Number.NaN, 3, -4]), reference);
    assertThrows(() => judgeAb(measured, BAND, "case"), Error, "ceiling-side failure — non-finite");
    const infinite = measureAb(new Float32Array([1, -2, Number.POSITIVE_INFINITY, -4]), reference);
    assertThrows(() => judgeAb(infinite, BAND, "case"), Error, "ceiling-side failure — non-finite");
  });

  it("prints the measurement and stops red when the band is not derived yet", () => {
    const measured = measureAb(new Float32Array([1, -2, 3, -4.5]), reference);
    const error = assertThrows(() => judgeAb(measured, undefined, "anima 512"), Error);
    assertStringIncludes(error.message, "anima 512: band not derived yet (measured relRMS ");
    assertStringIncludes(error.message, measured.relRms.toExponential(4));
    assertStringIncludes(error.message, "maxAbs 5.0000e-1");
  });

  it("is green inside the band and red below the floor or above the ceiling", () => {
    const measured = measureAb(new Float32Array([1, -2, 3, -4.1]), reference);
    judgeAb(measured, { floor: measured.relRms / 2, ceiling: measured.relRms * 2 }, "case");
    assertThrows(
      () => judgeAb(measured, { floor: measured.relRms * 2, ceiling: measured.relRms * 4 }, "case"),
      Error,
      "floor failure — relRMS",
    );
    assertThrows(
      () => judgeAb(measured, { floor: 0, ceiling: measured.relRms / 2 }, "case"),
      Error,
      "ceiling failure",
    );
  });

  it("throws on a malformed band declaration", () => {
    const measured = measureAb(new Float32Array([1, -2, 3, -4.1]), reference);
    assertThrows(
      () => judgeAb(measured, { floor: 0.2, ceiling: 0.1 }, "case"),
      Error,
      "the band declaration is broken",
    );
  });
});

describe("assertBitIdentical: in-device determinism", () => {
  it("passes when bit-identical", () => {
    assertBitIdentical(new Float32Array([1, 2]), new Float32Array([1, 2]), "twice");
  });

  it("is red on a single differing element, naming the count and position (±0 differ in bits)", () => {
    assertThrows(
      () => assertBitIdentical(new Float32Array([1, 0, 3]), new Float32Array([1, -0, 3]), "twice"),
      Error,
      "twice: two runs with the same preset and the same input are not bit-identical",
    );
    const error = assertThrows(
      () => assertBitIdentical(new Float32Array([1, 2, 3, 4]), new Float32Array([1, 5, 3, 6]), "x"),
      Error,
    );
    assertStringIncludes(error.message, "2 / 4 elements differ. First at 1: 2 / 5");
  });
});

describe("captureThenAbort: stops the generation at the observation point", () => {
  it("returns the value at the observation point and runs nothing after it", async () => {
    const seen: number[] = [];
    const value = await captureThenAbort<number>("gen", async (capture) => {
      for (let step = 1; step <= 3; step += 1) {
        await Promise.resolve();
        seen.push(step);
        if (step === 1) capture(step * 10);
      }
    });
    assertEquals(value, 10);
    assertEquals(seen, [1]);
  });

  it("rethrows any exception other than the abort as is", async () => {
    const cause = new Error("failed to build the DiT");
    const thrown = await assertRejects(() =>
      captureThenAbort<number>("gen", () => Promise.reject(cause)),
    );
    assertEquals(thrown, cause);
  });

  it("throws when the generation ends without reaching the observation point", async () => {
    await assertRejects(
      () => captureThenAbort<number>("gen", () => Promise.resolve()),
      Error,
      "gen: generation ended without ever reaching the observation point",
    );
  });
});

describe("overrideQuantSession: the temporary manifest of a fault injection", () => {
  it("replaces only the named preset's session and leaves the original untouched", () => {
    const raw = {
      format: "karume/5",
      models: {
        m: {
          defaultQuant: "i8-a8",
          quants: {
            i8: { weights: { dit: "i8" }, session: {} },
            "i8-a8": { weights: { dit: "i8" }, session: { linearCompute: "a8" } },
          },
        },
      },
    };
    const before = structuredClone(raw);
    const injected = overrideQuantSession(raw, "m", "i8-a8", {});
    assertEquals(injected, {
      ...raw,
      models: {
        m: {
          ...raw.models.m,
          quants: { ...raw.models.m.quants, "i8-a8": { weights: { dit: "i8" }, session: {} } },
        },
      },
    });
    assertEquals(raw, before);
    assertThrows(() => overrideQuantSession(raw, "m", "i4", {}), Error, "quants.i4");
  });
});

describe("describeBundleCensus: the dispatch count of each knob of the bundle", () => {
  it("lists only the knobs with a non-reference value, with the dispatch count of that variant", () => {
    const rows = [
      { key: linearI8a8Key(true, true), dispatchCount: 3 },
      { key: linearKey("f32", true), dispatchCount: 5 },
    ];
    assertEquals(
      describeBundleCensus(rows, { linearCompute: "a8", attentionCompute: "f32" }),
      "linearCompute=a8 ×3",
    );
    assertEquals(describeBundleCensus(rows, { attentionCompute: "a8" }), "attentionCompute=a8 ×0");
  });
});
