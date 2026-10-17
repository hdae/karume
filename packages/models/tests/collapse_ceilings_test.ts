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

describe("崩壊上限の判定（output.2 以降 — GPU も資産も要らない）", () => {
  // 固定の合成データ: 1 チャネルだけ桁の大きい外れ値チャネルを持つ golden（実物の channel 686 の形）。
  const SIZE = 1024;
  const OUTLIER = 686;
  const golden = Float32Array.from({ length: SIZE }, (_, index) =>
    index === OUTLIER ? 28.8 : Math.sin(index * 0.37 + 0.1) * (1 + (index % 7) * 0.3),
  );
  const normOf = (values: ArrayLike<number>): number =>
    Math.sqrt(Array.from(values, (value) => value * value).reduce((sum, value) => sum + value, 0));
  /** golden に、相対 RMS 誤差がちょうど `target` になる揺れ（golden と別周期）を足した別標本。 */
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
  /** 表の全出力（output.2〜24）と、その上限。 */
  const ceilings = Object.entries(COLLAPSE_CEILINGS);
  /**
   * 各出力の上限の半分の相対 RMS を持つ別標本。上限は導出に使った最悪の × 2 以上なので、
   * これは導出に使った正当な最悪以上の揺れを持つ（正当な標本の代表）。
   */
  const legitimateFor = (ceiling: number): Float32Array => sampleAt(ceiling / 2);
  const deepest = collapseCeilingOf("output.24");
  /**
   * 上限が output.24 の正当な揺れ（上限の半分 ≈ 0.165 — 導出の最悪 0.1620 程度）より狭い最後の
   * 出力（ADR 0026 追記 2026-10-07 の表: output.14 の上限 0.15 < 0.165 ≤ output.15 の 0.17）。
   */
  const LAST_TIGHTER_THAN_DEEPEST = 14;
  const indexOf = (output: string): number => Number(output.slice("output.".length));

  it("上限の表は output.2〜24 をちょうど覆い、表に無い出力は投げる", () => {
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

  it("上限は出力の順に非減少（活性量子化の誤差は層を下るほど積もる — 導出の最悪も単調）", () => {
    ceilings.slice(1).forEach(([output, ceiling], position) => {
      const [previousOutput, previous] = ceilings[position];
      assert(previous <= ceiling, `${previousOutput} の ${previous} > ${output} の ${ceiling}`);
    });
  });

  it("output.24 の正当な揺れの大きさの別標本は、output.2〜14 では赤・output.15〜24 では通す（上限は出力ごと）", () => {
    const deepLegitimate = legitimateFor(deepest);
    for (const [output, ceiling] of ceilings) {
      const report = judgeCollapse(deepLegitimate, golden, ceiling);
      assertEquals(
        report.pass,
        indexOf(output) > LAST_TIGHTER_THAN_DEEPEST,
        `${output}: relRms=${report.relRms} 上限 ${ceiling}`,
      );
    }
  });

  it("各出力で、上限の半分（導出の最悪以上）の別標本は通し、上限を 1% 超えた別標本は赤にする", () => {
    for (const [output, ceiling] of ceilings) {
      const inside = judgeCollapse(legitimateFor(ceiling), golden, ceiling);
      assert(inside.pass, `${output}: relRms=${inside.relRms} 上限 ${ceiling}`);
      const outside = judgeCollapse(sampleAt(ceiling * 1.01), golden, ceiling);
      assertEquals(outside.pass, false, `${output}: relRms=${outside.relRms} 上限 ${ceiling}`);
    }
  });

  it("外れ値チャネルの 1 要素が 7 動いても（maxAbs では落ちた形）全体が近ければ output.19 の上限で通す", () => {
    const ceiling = collapseCeilingOf("output.19");
    const report = judgeCollapse(withElement(golden, OUTLIER, 28.8 - 7.17), golden, ceiling);
    assert(report.pass, `relRms=${report.relRms} 上限 ${ceiling}`);
    assert(report.maxAbs > 7, `maxAbs=${report.maxAbs}`);
  });

  it("全要素 0 を全出力で赤にする（相対 RMS 1.0）", () => {
    for (const [output, ceiling] of ceilings) {
      const report = judgeCollapse(new Float32Array(SIZE), golden, ceiling);
      assertEquals([report.pass, report.relRms], [false, 1], output);
    }
  });

  it("符号反転を全出力で赤にする", () => {
    for (const [output, ceiling] of ceilings) {
      const flipped = map(legitimateFor(ceiling), (value) => -value);
      assertEquals(judgeCollapse(flipped, golden, ceiling).pass, false, output);
    }
  });

  it("全要素 × 2 と × 1.5 の拡大を全出力で赤にする（相対 RMS 1.0 / 0.5）", () => {
    for (const factor of [2, 1.5]) {
      const scaled = map(golden, (value) => value * factor);
      for (const [output, ceiling] of ceilings) {
        const report = judgeCollapse(scaled, golden, ceiling);
        assertEquals(
          report.pass,
          false,
          `× ${factor} ${output}: relRms=${report.relRms} 上限 ${ceiling}`,
        );
      }
    }
  });

  it("1 要素だけの NaN / +Inf / -Inf を、実行側でも期待値側でも赤にする", () => {
    const legitimate = legitimateFor(deepest);
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      const onActual = judgeCollapse(withElement(legitimate, 3, bad), golden, deepest);
      assertEquals([onActual.pass, onActual.nonFiniteCount], [false, 1], `actual に ${bad}`);
      const onExpected = judgeCollapse(legitimate, withElement(golden, 3, bad), deepest);
      assertEquals([onExpected.pass, onExpected.nonFiniteCount], [false, 1], `expected に ${bad}`);
    }
  });

  it("長さが違えば判定せずに投げる（取り違え — 誤差の問題ではない）", () => {
    assertThrows(() => judgeCollapse(golden.subarray(1), golden, deepest), AllcloseError);
  });
});
