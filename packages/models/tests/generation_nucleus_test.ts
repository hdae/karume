import { assertEquals } from "@std/assert";
import { samplerDistribution } from "../src/generation/sampler.ts";

/** 比較ソートによる参照。順位・加算順・境界を固定し、radix の添字算術とは独立に検証する。 */
const orderedWeights = (logits: Float32Array<ArrayBuffer>) => {
  const tokens = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a] || a - b);
  const weights = tokens.map((token) => Math.exp(logits[token] - logits[tokens[0]]));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  return { tokens, weights, total };
};

const assertNucleus = (logits: Float32Array<ArrayBuffer>, topP: number): void => {
  const { tokens, weights, total } = orderedWeights(logits);
  let cumulative = 0;
  const crossing = weights.findIndex((weight) => {
    cumulative += weight / total;
    return cumulative >= topP;
  });
  const kept = crossing < 0 ? tokens.length : crossing + 1;
  const retained = weights.slice(0, kept);
  const keptTotal = retained.reduce((sum, weight) => sum + weight, 0);
  const probabilities = Float64Array.from(retained, (weight) => weight / keptTotal);
  const before = new Uint32Array(logits.buffer, logits.byteOffset, logits.length).slice();
  const actual = samplerDistribution(logits, { temperature: 1, topP }, []);
  assertEquals(actual.tokens, Int32Array.from(tokens.slice(0, kept)));
  assertEquals(
    new BigUint64Array(actual.probabilities.buffer),
    new BigUint64Array(probabilities.buffer),
  );
  assertEquals(new Uint32Array(logits.buffer, logits.byteOffset, logits.length), before);
};

Deno.test("top-p 単独指定の全語彙順位", async (t) => {
  await t.step("正負の同値、符号付きゼロ、subnormal、禁止 token を比較ソートと同じ順に残す", () => {
    const bits = Uint32Array.of(
      0x7fc00000, // 入力 view の外の NaN を読まない。
      0x80000000,
      0x00000000,
      0x80000000,
      0x00000000,
      0x00000001,
      0x80000001,
      0x00800000,
      0x80800000,
      0x3f800000,
      0xbf800000,
      0x3f800000,
      0xbf800000,
      0x7f7fffff,
      0xff7fffff,
      0xff800000,
      0x7fc00000,
    );
    const all = new Float32Array(bits.buffer);
    // 巨大な正値に確率が集中する場合と、それを除いた同値の境界の両方を通す。
    for (const logits of [all.subarray(1, 16), all.subarray(1, 13)]) {
      for (const topP of [0.01, 0.25, 0.5, 0.95, 1 - Number.EPSILON]) {
        assertNucleus(logits, topP);
      }
    }
  });

  await t.step("全件ソートで得た累積確率の直前・一致・直後でも候補と確率を変えない", () => {
    let state = 20260910;
    const logits = Float32Array.from({ length: 8192 }, () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 2 ** 32 * 16 - 8;
    });
    const { weights, total } = orderedWeights(logits);
    let cumulative = 0;
    for (let index = 0; index < 1024; index += 1) cumulative += weights[index] / total;
    const boundary = Float64Array.of(cumulative);
    const bits = new BigUint64Array(boundary.buffer);
    const exact = bits[0];
    for (const delta of [-1n, 0n, 1n]) {
      bits[0] = exact + delta;
      assertNucleus(logits, boundary[0]);
    }
  });

  await t.step("昇順・降順・同値が多い大語彙でも分布の全ビットが参照と一致する", () => {
    for (const shape of ["ascending", "descending", "ties"] as const) {
      const logits = Float32Array.from(
        { length: 32768 },
        (_, token) =>
          shape === "ascending"
            ? token / 4096 - 4
            : shape === "descending"
            ? 4 - token / 4096
            : token % 17 - 8,
      );
      for (const topP of [0.1, 0.9, 0.999]) assertNucleus(logits, topP);
    }
  });
});
