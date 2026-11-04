// The collapse-ceiling judgement of the DeBERTa w8a8 mirror gate (`e2e_deberta_w8a8_test.ts`,
// output.2 onwards — ADR 0026 addendum 2026-10-07) on fixed synthetic data: needs neither a GPU nor
// model assets.

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { AllcloseError } from "../../runtime/src/reference/allclose.ts";
import {
  COLLAPSE_CEILINGS,
  collapseCeilingOf,
  GRAPH_OUTPUTS,
  judgeCollapse,
  STRICT_OUTPUTS,
} from "./helpers/collapse-ceilings.ts";

describe("the collapse-ceiling judgement (output.2 onwards — needs neither a GPU nor assets)", () => {
  // Fixed synthetic data: a golden with a single outlier channel an order of magnitude larger (the
  // shape of the real channel 686).
  const SIZE = 1024;
  const OUTLIER = 686;
  const golden = Float32Array.from({ length: SIZE }, (_, index) =>
    index === OUTLIER ? 28.8 : Math.sin(index * 0.37 + 0.1) * (1 + (index % 7) * 0.3),
  );
  const normOf = (values: ArrayLike<number>): number =>
    Math.sqrt(Array.from(values, (value) => value * value).reduce((sum, value) => sum + value, 0));
  /**
   * A different sample: the golden plus a perturbation (with a period other than the golden's)
   * whose relative RMS error is exactly `target`.
   */
  const sampleAt = (target: number): Float32Array => {
    const noise = Array.from({ length: SIZE }, (_, index) => Math.cos(index * 1.13 + 0.7));
    const scale = (target * normOf(golden)) / normOf(noise);
    return Float32Array.from(golden, (value, index) => value + noise[index] * scale);
  };
  const map = (values: Float32Array, f: (value: number) => number): Float32Array =>
    Float32Array.from(values, f);
  const withElement = (values: Float32Array, index: number, value: number): Float32Array => {
    const copy = values.slice();
    copy[index] = value;
    return copy;
  };
  /** Every output in the table (output.2–24) and its ceiling. */
  const ceilings = Object.entries(COLLAPSE_CEILINGS);
  /**
   * A different sample with a relative RMS of half each output's ceiling. A ceiling is at least × 2
   * the worst case used to derive it, so this sample perturbs at least as much as the legitimate
   * worst case used in the derivation (a representative of the legitimate samples).
   */
  const legitimateFor = (ceiling: number): Float32Array => sampleAt(ceiling / 2);
  const deepest = collapseCeilingOf("output.24");
  /**
   * The last output whose ceiling is narrower than output.24's legitimate perturbation (half its
   * ceiling ≈ 0.165 — about the derivation's worst case 0.1620) (the table of the ADR 0026 addendum
   * 2026-10-07: output.14's ceiling 0.15 < 0.165 ≤ output.15's 0.17).
   */
  const LAST_TIGHTER_THAN_DEEPEST = 14;
  const indexOf = (output: string): number => Number(output.slice("output.".length));

  it("the ceiling table covers exactly output.2–24 and throws for an output not in the table", () => {
    assertEquals(
      ceilings.map(([output]) => output),
      Array.from(
        { length: GRAPH_OUTPUTS - STRICT_OUTPUTS },
        (_, k) => `output.${k + STRICT_OUTPUTS}`,
      ),
    );
    for (const output of ["output.1", `output.${GRAPH_OUTPUTS}`]) {
      assertThrows(() => collapseCeilingOf(output), Error, output);
    }
  });

  it("the ceilings are non-decreasing in output order (activation-quantisation error accumulates down the layers — the derivation's worst cases are monotonic too)", () => {
    ceilings.slice(1).forEach(([output, ceiling], position) => {
      const [previousOutput, previous] = ceilings[position];
      assert(previous <= ceiling, `${previousOutput} at ${previous} > ${output} at ${ceiling}`);
    });
  });

  it("a different sample with output.24's legitimate perturbation size is red on output.2–14 and passes on output.15–24 (the ceiling is per output)", () => {
    const deepLegitimate = legitimateFor(deepest);
    for (const [output, ceiling] of ceilings) {
      const report = judgeCollapse(deepLegitimate, golden, ceiling);
      assertEquals(
        report.pass,
        indexOf(output) > LAST_TIGHTER_THAN_DEEPEST,
        `${output}: relRms=${report.relRms} ceiling ${ceiling}`,
      );
    }
  });

  it("on each output, a different sample at half the ceiling (at least the derivation's worst case) passes and one 1% over the ceiling is red", () => {
    for (const [output, ceiling] of ceilings) {
      const inside = judgeCollapse(legitimateFor(ceiling), golden, ceiling);
      assert(inside.pass, `${output}: relRms=${inside.relRms} ceiling ${ceiling}`);
      const outside = judgeCollapse(sampleAt(ceiling * 1.01), golden, ceiling);
      assertEquals(outside.pass, false, `${output}: relRms=${outside.relRms} ceiling ${ceiling}`);
    }
  });

  it("even when one element of the outlier channel moves by 7 (a shape that failed under maxAbs), it passes output.19's ceiling if the whole is close", () => {
    const ceiling = collapseCeilingOf("output.19");
    const report = judgeCollapse(withElement(golden, OUTLIER, 28.8 - 7.17), golden, ceiling);
    assert(report.pass, `relRms=${report.relRms} ceiling ${ceiling}`);
    assert(report.maxAbs > 7, `maxAbs=${report.maxAbs}`);
  });

  it("all-zero elements are red on every output (relative RMS 1.0)", () => {
    for (const [output, ceiling] of ceilings) {
      const report = judgeCollapse(new Float32Array(SIZE), golden, ceiling);
      assertEquals([report.pass, report.relRms], [false, 1], output);
    }
  });

  it("a sign flip is red on every output", () => {
    for (const [output, ceiling] of ceilings) {
      const flipped = map(legitimateFor(ceiling), (value) => -value);
      assertEquals(judgeCollapse(flipped, golden, ceiling).pass, false, output);
    }
  });

  it("scaling every element × 2 and × 1.5 is red on every output (relative RMS 1.0 / 0.5)", () => {
    for (const factor of [2, 1.5]) {
      const scaled = map(golden, (value) => value * factor);
      for (const [output, ceiling] of ceilings) {
        const report = judgeCollapse(scaled, golden, ceiling);
        assertEquals(
          report.pass,
          false,
          `× ${factor} ${output}: relRms=${report.relRms} ceiling ${ceiling}`,
        );
      }
    }
  });

  it("a single NaN / +Inf / -Inf element is red on both the actual side and the expected side", () => {
    const legitimate = legitimateFor(deepest);
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      const onActual = judgeCollapse(withElement(legitimate, 3, bad), golden, deepest);
      assertEquals([onActual.pass, onActual.nonFiniteCount], [false, 1], `${bad} in actual`);
      const onExpected = judgeCollapse(legitimate, withElement(golden, 3, bad), deepest);
      assertEquals([onExpected.pass, onExpected.nonFiniteCount], [false, 1], `${bad} in expected`);
    }
  });

  it("throws without judging when the lengths differ (a mix-up — not a matter of error)", () => {
    assertThrows(() => judgeCollapse(golden.subarray(1), golden, deepest), AllcloseError);
  });
});
