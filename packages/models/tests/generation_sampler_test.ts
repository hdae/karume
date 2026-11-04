// Behaviour tests of host-side sampling (`src/generation/sampler.ts`) and the stop decision over
// the EOS set. Neither a GPU nor real assets are needed.
//
// What this pins is the **order and boundaries of the logit processing** from "one logits row → one
// token" — temperature 0 collapsing to greedy, the sets top-k / top-p keep, the sign convention of
// the repetition penalty, logit bias, and seed determinism. On a real GPU these show up only as
// "the text is a little odd" (sampling raises no exception), and the acceptance e2e only takes the
// temperature-0 path, so a boundary off by one would **stay green forever**.
//
// MUST: token-sequence parity with HF's sampling output is impossible (the RNG is splitmix64, not
// torch's Philox — ADR 0083 decision 7). So every expected value is **a theoretical value this
// check wrote down by hand**; not a single fixture was taken from the implementation. The
// distribution itself is pinned by the χ² check (below).

import { assert, assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import { ModelInputError } from "../src/errors.ts";
import {
  createSampler,
  isStopToken,
  samplerDistribution,
  type SamplerSpec,
  snapshotSpec,
} from "../src/generation/sampler.ts";

const f32 = (values: readonly number[]): Float32Array<ArrayBuffer> => {
  const data = new Float32Array(values.length);
  data.set(values);
  return data;
};

/**
 * The softmax on the expected side (written **in a different form from the implementation** — the
 * implementation subtracts the maximum before taking exp, so writing the plain formula avoids the
 * tautology of "calling the same function on both sides and checking they agree").
 */
const softmax = (logits: readonly number[]): number[] => {
  const weights = logits.map((logit) => Math.exp(logit));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  return weights.map((weight) => weight / total);
};

const assertProbabilities = (
  actual: Float64Array<ArrayBuffer>,
  expected: readonly number[],
  where: string,
): void => {
  assertEquals(actual.length, expected.length, `${where}: candidate count`);
  expected.forEach((want, index) => {
    assertAlmostEquals(actual[index], want, 1e-6, `${where}: probability[${index}]`);
  });
};

/**
 * The subject of the distribution check (`logits = ln(p)`, so the temperature-1 softmax gives back
 * `p`).
 */
const BASE_PROBABILITIES = [0.5, 0.3, 0.15, 0.05] as const;
const BASE_LOGITS = f32(BASE_PROBABILITIES.map((probability) => Math.log(probability)));

// --------------------------------------------------------------------------- Temperature 0 =
// collapse to greedy (ADR 0083 decision 7 — the path the parity check keeps alive)
// ---------------------------------------------------------------------------

test("sampler: the default (nothing set) is temperature-0 greedy and matches argmax exactly", () => {
  const logits = f32([-1, 4.5, 0, 4.25, -3]);
  assertEquals(createSampler().next(logits, []), 1);
  assertEquals(createSampler({}).next(logits, []), 1);
  const distribution = samplerDistribution(logits, {}, []);
  assertEquals(
    [...distribution.tokens],
    [1],
    "temperature 0 collapses to a one-point distribution",
  );
  assertEquals([...distribution.probabilities], [1]);
});

test("sampler: the temperature-0 tie-break takes the earlier one (the same as torch's argmax)", () => {
  // A row of equal values fails an implementation that updates on `>=` (the later one wins).
  assertEquals(createSampler().next(f32([1, 3, 3, 2]), []), 1);
  assertEquals(createSampler().next(f32([3, 3, 3]), []), 0);
});

test("sampler: temperature 0 gives the same sequence whatever the seed (no draw runs)", () => {
  const logits = f32([0.1, 0.9, 0.4]);
  const first = createSampler({ seed: 1 });
  const second = createSampler({ seed: 999 });
  for (let step = 0; step < 8; step += 1) {
    assertEquals(first.next(logits, []), second.next(logits, []), `step ${step}`);
  }
});

test("sampler: NaN fails even when it is not the maximum (a value that makes every comparison false is not passed)", () => {
  // Every comparison with NaN is false, so passing it through would ① make argmax return another
  // token, ② reject every candidate after the top-k threshold once became NaN, and ③ make the
  // softmax denominator NaN. None of these raises an exception.
  const poisoned = f32([1, NaN, 3, 2]);
  assertThrows(() => createSampler().next(poisoned, []), Error, "NaN");
  assertThrows(() => samplerDistribution(poisoned, { temperature: 1, topK: 2 }, []), Error, "NaN");
  assertThrows(
    () => samplerDistribution(poisoned, { temperature: 1, topP: 0.9 }, []),
    Error,
    "NaN",
  );
});

test("sampler: greedy rejects the first NaN in range ahead of the maximum", () => {
  for (const values of [
    [NaN, 3, 2],
    [Infinity, 3, NaN],
    [NaN, NaN, Infinity],
  ]) {
    // The NaNs outside are not part of the view. The check runs to the end even after the maximum
    // is found.
    const backing = f32([NaN, ...values, NaN]);
    const logits = backing.subarray(1, backing.length - 1);
    const first = values.findIndex(Number.isNaN);
    const error = assertThrows(() => createSampler().next(logits, []), Error);
    assertEquals(
      error.message,
      `logits[${first}] is NaN (non-finite) — failing here instead of folding it into a token id`,
    );
  }
  assertEquals(createSampler().next(f32([-Infinity, -0, 0, -Infinity]), []), 1);
  assertThrows(
    () => createSampler().next(f32([0, Infinity, -1]), []),
    Error,
    "maximum of the logits is non-finite",
  );
});

test("sampler: topK rejects the first NaN even outside the candidate range", () => {
  for (const topK of [1, 2, 4, 8]) {
    for (const values of [
      [NaN, 3, 2, 1],
      [3, NaN, 2, NaN],
      [Infinity, 3, 2, NaN],
    ]) {
      const backing = f32([NaN, ...values, NaN]);
      const logits = backing.subarray(1, backing.length - 1);
      const first = values.findIndex(Number.isNaN);
      const error = assertThrows(
        () => samplerDistribution(logits, { temperature: 1, topK }, []),
        Error,
      );
      assertEquals(
        error.message,
        `logits[${first}] is NaN (non-finite) — failing here instead of folding it into a token id`,
      );
    }
  }
  // A NaN produced by the processing is rejected too, even at a position outside the top
  // candidates.
  assertThrows(
    () =>
      samplerDistribution(
        f32([3, 2, Infinity]),
        {
          temperature: 1,
          topK: 1,
          logitBias: [[2, -Infinity]],
        },
        [],
      ),
    Error,
    "logits[2] is NaN",
  );
});

test("sampler: a non-finite maximum logit fails instead of being folded into a token id", () => {
  // A gather outside the position table poisons the whole row with NaN (known-issues), so the last
  // row's logits become non-finite **entirely**. argmax would fold that into "a plausible token id"
  // (token 0 at the front, since NaN comparisons are always false), so this is the only place
  // before the folding.
  assertThrows(() => createSampler().next(f32([NaN, NaN, NaN]), []), Error, "non-finite");
  assertThrows(() => createSampler().next(f32([-Infinity, -Infinity]), []), Error, "non-finite");
  // The temperature > 0 side fails at the same step (before the softmax goes all 0 and breaks the
  // draw).
  assertThrows(
    () => createSampler({ temperature: 1 }).next(f32([NaN, NaN]), []),
    Error,
    "non-finite",
  );
});

// --------------------------------------------------------------------------- top-k (no upper limit
// — because this is the last-row logits exit; the flip side of ADR 0083 decision 6)
// ---------------------------------------------------------------------------

test("sampler: topK = 1 matches greedy whatever the temperature (one candidate makes the draw trivial)", () => {
  const spec = { temperature: 1, topK: 1, seed: 3 } satisfies SamplerSpec;
  const distribution = samplerDistribution(BASE_LOGITS, spec, []);
  assertEquals([...distribution.tokens], [0]);
  assertEquals([...distribution.probabilities], [1]);
  // Changing the seed does not move it (the filtering has collapsed to the top one).
  const first = createSampler(spec);
  const second = createSampler({ ...spec, seed: 4242 });
  for (let step = 0; step < 16; step += 1) {
    assertEquals(first.next(BASE_LOGITS, []), 0, `step ${step}`);
    assertEquals(second.next(BASE_LOGITS, []), 0, `step ${step}`);
  }
});

test("sampler: topK keeps only the top k and normalizes the probabilities among them", () => {
  const distribution = samplerDistribution(BASE_LOGITS, { temperature: 1, topK: 2 }, []);
  assertEquals([...distribution.tokens], [0, 1], "top 2 (descending probability)");
  // The dropped mass of 0.15 + 0.05 leaves the denominator.
  assertProbabilities(distribution.probabilities, [0.5 / 0.8, 0.3 / 0.8], "topK 2");
});

test("sampler: a topK tie keeps the smaller token id", () => {
  const distribution = samplerDistribution(f32([2, 2, 1]), { temperature: 1, topK: 1 }, []);
  assertEquals([...distribution.tokens], [0]);
});

test("sampler: topK at or above the vocabulary size keeps everything (HF's min(top_k, vocab))", () => {
  for (const topK of [4, 5, 1_000_000]) {
    const distribution = samplerDistribution(BASE_LOGITS, { temperature: 1, topK }, []);
    assertEquals([...distribution.tokens], [0, 1, 2, 3], `topK ${topK}`);
    assertProbabilities(distribution.probabilities, [...BASE_PROBABILITIES], `topK ${topK}`);
  }
});

/**
 * The reference top k — **literally as the contract says**, "sort everything by descending value
 * (ties by ascending token id) and take the first k". Written in a different form from the
 * implementation (a bounded heap), so it is not a tautology (it sorts everything).
 */
const referenceTopK = (logits: Float32Array<ArrayBuffer>, k: number): number[] =>
  Array.from({ length: logits.length }, (_unused, token) => token)
    .sort((left, right) => {
      if (logits[left] === logits[right]) return left - right;
      return logits[left] > logits[right] ? -1 : 1;
    })
    .slice(0, k);

/**
 * Three distributions that change the shape of the selection (vocabulary 8,192). **Ascending** is
 * the worst case where "the value keeps improving", and **ties** is the case where, without an
 * effective cutoff, candidates are swapped every time — both are inputs that break the order before
 * they break the contents of the top k.
 */
const SELECTION_LOGITS = (() => {
  const size = 8_192;
  let state = 20_260_907;
  const random = (): number => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  const build = (value: (token: number) => number): Float32Array<ArrayBuffer> => {
    const logits = new Float32Array(size);
    for (let token = 0; token < size; token += 1) logits[token] = value(token);
    return logits;
  };
  return {
    uniform: build(() => random() * 16 - 8),
    ascending: build((token) => token / size),
    ties: build(() => Math.floor(random() * 8)),
  };
})();

test("sampler: topK matches the reference (the first k of everything sorted) in contents and order for every k", () => {
  // To pin against a regression to a selection that switches with k (where order or tie-break could
  // change around a k boundary), distant k values and two adjacent ones (1024 / 1025) are checked
  // against the same reference.
  for (const [shape, logits] of Object.entries(SELECTION_LOGITS)) {
    for (const topK of [64, 1024, 1025, 4096]) {
      const distribution = samplerDistribution(logits, { temperature: 1, topK }, []);
      assertEquals(
        [...distribution.tokens],
        referenceTopK(logits, topK),
        `${shape} / topK ${topK}`,
      );
    }
  }
});

test("sampler: the probabilities of a large topK are normalized over the reference top k only", () => {
  const logits = SELECTION_LOGITS.uniform;
  const topK = 1025;
  const distribution = samplerDistribution(logits, { temperature: 1, topK }, []);
  const expected = softmax(referenceTopK(logits, topK).map((token) => logits[token]));
  assertProbabilities(distribution.probabilities, expected, `topK ${topK}`);
});

// --------------------------------------------------------------------------- top-p (nucleus — the
// smallest set whose cumulative mass reaches p; the top one is always kept)
// ---------------------------------------------------------------------------

test("sampler: topP = 1 does not filter", () => {
  const distribution = samplerDistribution(BASE_LOGITS, { temperature: 1, topP: 1 }, []);
  assertEquals([...distribution.tokens], [0, 1, 2, 3]);
  assertProbabilities(distribution.probabilities, [...BASE_PROBABILITIES], "topP 1");
});

test("sampler: topP keeps up to the one at which the cumulative mass reaches p", () => {
  // The cumulative masses are 0.5 / 0.8 / 0.95 / 1.0. This fixes how the boundary is read
  // ("including the one that reaches it").
  // NOTE: a p exactly equal to a cumulative mass (here 0.5 / 0.8 / 0.95) can swing either way
  // through f32 rounding, so it is not checked — what is fixed is the reading "including the one
  // that reaches it", not the rounding at the boundary.
  const cases = [
    { topP: 0.4, tokens: [0] },
    { topP: 0.75, tokens: [0, 1] },
    { topP: 0.85, tokens: [0, 1, 2] },
    { topP: 0.99, tokens: [0, 1, 2, 3] },
  ] as const;
  for (const { topP, tokens } of cases) {
    const distribution = samplerDistribution(BASE_LOGITS, { temperature: 1, topP }, []);
    assertEquals([...distribution.tokens], [...tokens], `topP ${topP}`);
    const mass = tokens.reduce<number>((sum, token) => sum + BASE_PROBABILITIES[token], 0);
    assertProbabilities(
      distribution.probabilities,
      tokens.map((token) => BASE_PROBABILITIES[token] / mass),
      `topP ${topP}`,
    );
  }
});

test("sampler: topP applies after topK (the cumulative mass is taken within the remaining set)", () => {
  // topK 3 leaves {0.5, 0.3, 0.15}, whose normalized probabilities are 0.5263 / 0.3158 / 0.1579.
  // The top cumulative mass 0.5263 reaches 0.52, so only one remains — an implementation that takes
  // the cumulative mass over the whole vocabulary (0.5 < 0.52) keeps two and fails.
  const distribution = samplerDistribution(
    BASE_LOGITS,
    { temperature: 1, topK: 3, topP: 0.52 },
    [],
  );
  assertEquals([...distribution.tokens], [0]);
});

/**
 * A reference by comparison sort. It fixes rank, summation order and boundary, and verifies
 * independently of the radix index arithmetic.
 */
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

describe("whole-vocabulary ranking with top-p alone", () => {
  it("keeps positive and negative ties, signed zeros, subnormals and banned tokens in the same order as a comparison sort", () => {
    const bits = Uint32Array.of(
      0x7fc00000, // The NaN outside the input view is not read.
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
    // Covers both the case where the probability concentrates on a huge positive value and the tie
    // boundary without it.
    for (const logits of [all.subarray(1, 16), all.subarray(1, 13)]) {
      for (const topP of [0.01, 0.25, 0.5, 0.95, 1 - Number.EPSILON]) {
        assertNucleus(logits, topP);
      }
    }
  });

  it("does not change candidates or probabilities just before, at, or just after a cumulative probability from a full sort", () => {
    let state = 20260910;
    const logits = Float32Array.from({ length: 8192 }, () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return (state / 2 ** 32) * 16 - 8;
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

  it("every bit of the distribution matches the reference on large vocabularies that are ascending, descending or tie-heavy", () => {
    for (const shape of ["ascending", "descending", "ties"] as const) {
      const logits = Float32Array.from({ length: 32768 }, (_, token) =>
        shape === "ascending"
          ? token / 4096 - 4
          : shape === "descending"
            ? 4 - token / 4096
            : (token % 17) - 8,
      );
      for (const topP of [0.1, 0.9, 0.999]) assertNucleus(logits, topP);
    }
  });
});

// --------------------------------------------------------------------------- repetition penalty
// (the sign convention of HF's `RepetitionPenaltyLogitsProcessor`)
// ---------------------------------------------------------------------------

test("sampler: the repetition penalty multiplies negative logits and divides non-negative ones", () => {
  const logits = f32([2, -2, 0, 1]);
  const distribution = samplerDistribution(
    logits,
    { temperature: 1, repetitionPenalty: 2 },
    [0, 1, 2],
  );
  // 2 → 2/2 = 1 / −2 → −2×2 = −4 / 0 → 0/2 = 0 (the non-negative side) / 1, not in the history,
  // passes through.
  assertEquals([...distribution.tokens], [0, 1, 2, 3]);
  assertProbabilities(distribution.probabilities, softmax([1, -4, 0, 1]), "penalty 2");
});

test("sampler: a repetition penalty < 1 works the other way (it lifts tokens already seen)", () => {
  const distribution = samplerDistribution(
    f32([2, -2, 0]),
    { temperature: 1, repetitionPenalty: 0.5 },
    [0, 1],
  );
  assertProbabilities(distribution.probabilities, softmax([4, -1, 0]), "penalty 0.5");
});

test("sampler: the repetition penalty applies once however often a token appears in the history", () => {
  const once = samplerDistribution(f32([2, 0]), { temperature: 1, repetitionPenalty: 2 }, [0]);
  const many = samplerDistribution(
    f32([2, 0]),
    { temperature: 1, repetitionPenalty: 2 },
    [0, 0, 0, 0],
  );
  assertProbabilities(many.probabilities, [...once.probabilities], "repeated history");
  assertProbabilities(once.probabilities, softmax([1, 0]), "one application");
});

test("sampler: repetitionPenalty = 1 (the default) does not read the history", () => {
  const plain = samplerDistribution(BASE_LOGITS, { temperature: 1 }, []);
  const withHistory = samplerDistribution(BASE_LOGITS, { temperature: 1 }, [0, 1, 2, 3]);
  assertProbabilities(withHistory.probabilities, [...plain.probabilities], "with history");
});

test("sampler: a history token id outside the vocabulary fails loudly", () => {
  assertThrows(
    () => samplerDistribution(f32([1, 2]), { temperature: 1, repetitionPenalty: 2 }, [5]),
    RangeError,
    "vocabulary",
  );
});

// ---------------------------------------------------------------------------
// logit bias
// ---------------------------------------------------------------------------

test("sampler: the logit bias is added before the temperature", () => {
  const logits = f32([2, 1, 0]);
  const distribution = samplerDistribution(logits, { temperature: 2, logitBias: [[2, 6]] }, []);
  // Add, then divide ([2,1,6] / 2 = [1,0.5,3]). An implementation that adds after the temperature
  // does not give [1,0.5,3].
  assertProbabilities(distribution.probabilities, softmax([1, 0.5, 3]), "bias then temperature");
  assertEquals(
    createSampler({ logitBias: [[2, 6]] }).next(logits, []),
    2,
    "applies at temperature 0 too",
  );
});

test("sampler: a logit bias of -Infinity removes that token from the candidates", () => {
  const logits = f32([5, 1, 2]);
  const spec = { logitBias: [[0, Number.NEGATIVE_INFINITY]] } satisfies SamplerSpec;
  assertEquals(createSampler(spec).next(logits, []), 2, "temperature 0");
  const distribution = samplerDistribution(logits, { ...spec, temperature: 1 }, []);
  assertEquals([...distribution.tokens], [0, 1, 2]);
  assertEquals(distribution.probabilities[0], 0, "probability of the banned token");
});

test("sampler: banning every token fails loudly (it does not silently return 0)", () => {
  const banned: SamplerSpec["logitBias"] = [
    [0, -Infinity],
    [1, -Infinity],
  ];
  // The maximum of the logits before processing is finite, so it was the caller's `logitBias` that
  // removed the candidates = an input error (ADR 0107 decision 2 — the remedy is "drop the ban").
  // Both paths, temperature 0 and with a temperature.
  assertThrows(
    () => samplerDistribution(f32([1, 2]), { temperature: 1, logitBias: banned }, []),
    ModelInputError,
    "every token is banned",
  );
  assertThrows(
    () => samplerDistribution(f32([1, 2]), { temperature: 0, logitBias: banned }, []),
    ModelInputError,
    "The maximum of the logits is non-finite",
  );
  assertThrows(
    () => samplerDistribution(f32([1, 2]), { temperature: 1, topK: 1, logitBias: banned }, []),
    ModelInputError,
    "every token is banned",
  );
});

test("sampler: logits already non-finite before processing are not an input error (the source is told apart from bias)", () => {
  // The same two lines also pass when "the model output is all −Infinity to begin with". That is a
  // wiring or weight breakage no change to `logitBias` by the caller can fix, so it stays a plain
  // `Error` (ADR 0107 decision 2).
  for (const spec of [{ temperature: 0 }, { temperature: 1 }, { temperature: 1, topK: 1 }]) {
    const error = assertThrows(() => samplerDistribution(f32([-Infinity, -Infinity]), spec, []));
    assert(
      !(error instanceof ModelInputError),
      `${JSON.stringify(spec)}: a model-output failure is classified as an input error`,
    );
  }
  // Even with a ban set, when the logits are non-finite before processing the source stays the
  // model output.
  const withBias = assertThrows(() =>
    samplerDistribution(
      f32([-Infinity, -Infinity]),
      {
        temperature: 1,
        logitBias: [[0, -Infinity]],
      },
      [],
    ),
  );
  assert(!(withBias instanceof ModelInputError));
});

test("sampler: a logit bias token outside the vocabulary fails loudly", () => {
  assertThrows(
    () => createSampler({ logitBias: [[9, 1]] }).next(f32([1, 2]), []),
    ModelInputError,
    "vocabulary",
  );
});

test("sampler: writing a logit bias for the same token twice fails loudly (it is not folded with the last one winning)", () => {
  // A `Map` would have silently folded this (`[[2,6],[2,-1]]`) with the last one winning — with an
  // array it would mean "only one of the two biases written applies", so it fails. It fails even
  // when the values are equal (the question of how to fold them is the same).
  assertThrows(
    () =>
      createSampler({
        logitBias: [
          [2, 6],
          [2, -1],
        ],
      }),
    ModelInputError,
    "Token 2 appears twice",
  );
  assertThrows(
    () =>
      samplerDistribution(
        f32([1, 2, 3]),
        {
          temperature: 1,
          logitBias: [
            [0, 1],
            [0, 1],
          ],
        },
        [],
      ),
    ModelInputError,
    "Token 0 appears twice",
  );
});

test("sampler: changing the logitBias array after creation does not change the draws (defensive copy)", () => {
  const logits = f32([5, 1, 2]);
  // Create it with a mutable array the caller keeps holding (both the outer array and the inner
  // tuples can be changed).
  const bias: [number, number][] = [[0, Number.NEGATIVE_INFINITY]];
  const sampler = createSampler({ logitBias: bias });
  assertEquals(sampler.next(logits, []), 2, "the spec at creation (token 0 banned)");

  bias[0][1] = 0; // lift the ban
  assertEquals(sampler.next(logits, []), 2, "changing the tuple has no effect");
  bias.push([2, Number.NEGATIVE_INFINITY]); // ban the winning token
  assertEquals(sampler.next(logits, []), 2, "appending to the array has no effect either");
  bias.length = 0;
  assertEquals(sampler.next(logits, []), 2, "emptying it has no effect either");
});

// --------------------------------------------------------------------------- The accepted set of
// specs (nothing is rounded silently)
// ---------------------------------------------------------------------------

test("sampler: a spec that cannot be accepted fails at construction", () => {
  const rejected: readonly SamplerSpec[] = [
    { temperature: -1 },
    { temperature: Number.NaN },
    { topK: 0 },
    { topK: 1.5 },
    { topP: 0 },
    { topP: 1.5 },
    { repetitionPenalty: 0 },
    { repetitionPenalty: -1 },
    { seed: -1 },
    { seed: 0.5 },
    { logitBias: [[-1, 1]] },
    { logitBias: [[0, Number.POSITIVE_INFINITY]] },
    { logitBias: [[0, Number.NaN]] },
    {
      logitBias: [
        [0, 1],
        [0, 2],
      ],
    },
  ];
  for (const spec of rejected) {
    assertThrows(
      () => createSampler(spec),
      ModelInputError,
      undefined,
      JSON.stringify(Object.keys(spec)),
    );
  }
});

/** An untyped spec from JS / JSON (to bring values the TS type check would reject to the check). */
const untypedSpec = (value: unknown): SamplerSpec => value as SamplerSpec;

test("sampler: an unknown key is not silently ignored; it fails listing the allowed keys", () => {
  // Even when `minP: 1` was meant to narrow to the most probable candidate, if it is ignored the
  // draw comes from the whole distribution. A misspelling (`top_p`) likewise stays quietly as "a
  // knob that has no effect".
  for (const spec of [{ temperature: 1, minP: 1, seed: 1 }, { top_p: 0.9 }]) {
    const error = assertThrows(
      () => createSampler(untypedSpec(spec)),
      ModelInputError,
      "Unknown key",
      JSON.stringify(spec),
    );
    for (const key of ["temperature", "topK", "topP", "repetitionPenalty", "logitBias", "seed"]) {
      assert(
        error.message.includes(key),
        `allowed key ${key} missing from the message: ${error.message}`,
      );
    }
  }
  // The snapshot at creation goes through the same check (the upper layers snapshot before creating
  // the sampler).
  assertThrows(() => snapshotSpec(untypedSpec({ minP: 1 })), ModelInputError, "minP");
  // Listing every known key passes (seed was not forgotten in the allowed set).
  createSampler({
    temperature: 1,
    topK: 2,
    topP: 0.5,
    repetitionPenalty: 1.1,
    logitBias: [[0, 1]],
    seed: 3,
  });
});

describe("sampler: an untyped logitBias fails as an input error without being added", () => {
  it("a string bias does not become string concatenation (it does not change the temperature-0 choice)", () => {
    // `1 + "0"` is `"10"` — written to f32 as 10, it would turn token 1, chosen without the bias,
    // into 0.
    assertThrows(
      () => createSampler(untypedSpec({ temperature: 0, logitBias: [[0, "0"]] })),
      ModelInputError,
      "logitBias[0]",
    );
  });

  it("a null / boolean bias is not silently taken as 0 / 1", () => {
    for (const bias of [null, true, false, undefined]) {
      assertThrows(
        () => createSampler(untypedSpec({ logitBias: [[0, bias]] })),
        ModelInputError,
        "logitBias[0]",
        String(bias),
      );
    }
  });

  it("a logitBias that is not an array, or a pair without 2 elements, is a shape error", () => {
    for (const logitBias of [new Map([[0, 1]]), { 0: 1 }, [[0]], [[0, 1, 2]], [0]]) {
      assertThrows(
        () => createSampler(untypedSpec({ logitBias })),
        ModelInputError,
        "logitBias",
        JSON.stringify(logitBias),
      );
    }
  });

  it("the snapshot at creation (snapshotSpec) goes through the same check first", () => {
    // The upper layers (chat / ChatSession) snapshot before creating the sampler, so if the
    // snapshot turned a shape error into a TypeError it would not look like an input error.
    assertThrows(
      () => snapshotSpec(untypedSpec({ logitBias: new Map([[0, 1]]) })),
      ModelInputError,
      "logitBias",
    );
  });

  it("-Infinity (ban) and finite numbers are accepted as before", () => {
    createSampler({
      logitBias: [
        [0, Number.NEGATIVE_INFINITY],
        [1, -2.5],
      ],
    });
  });
});

// --------------------------------------------------------------------------- Failure
// classification (ADR 0107 — the type is split by where the caller branches)
// ---------------------------------------------------------------------------

describe("sampler: the failure classification is set by where the caller branches", () => {
  it("the range of a spec value is an input error (the caller can fix the value to pass)", () => {
    assertThrows(() => createSampler({ topP: 0 }), ModelInputError, "topP 0");
    // The single owner of seed's accepted set is `request-gates.ts` (ADR 0107 decision 6). That
    // this fails with `ModelInputError` is itself the counterpart that `Randu` has no condition of
    // its own.
    assertThrows(() => createSampler({ seed: -1 }), ModelInputError, "seed -1");
  });

  it("a logitBias outside the vocabulary is an input error too (the vocabulary is only known at the first draw)", () => {
    assertThrows(
      () => createSampler({ logitBias: [[9, 1]] }).next(f32([1, 2]), []),
      ModelInputError,
      "logitBias token 9 is outside the vocabulary 0..1",
    );
  });

  it("a NaN in the logits is not an input error (the caller cannot fix it by changing the request)", () => {
    // A weight or wiring breakage; an HTTP server would return 500. Caught as the same type, it
    // would come back as "fix your input".
    const error = assertThrows(() => createSampler().next(f32([Number.NaN, 1]), []), Error, "NaN");
    assert(
      !(error instanceof ModelInputError),
      "a NaN in the logits is classified as an input error",
    );
  });

  it("a history token id outside the vocabulary is not an input error either (the only caller is already checked)", () => {
    // `createSampler` is not on the public surface, and only `GenerationSequence.generate` passes
    // `history` (a checked prompt + the tokens it chose itself). A failure here is a wiring
    // breakage.
    const error = assertThrows(
      () => samplerDistribution(f32([1, 2]), { temperature: 1, repetitionPenalty: 2 }, [5]),
      RangeError,
      "vocabulary",
    );
    assert(
      !(error instanceof ModelInputError),
      "a history token outside the vocabulary is classified as an input error",
    );
  });
});

// --------------------------------------------------------------------------- Seed determinism and
// the distribution check
//
// Design of the distribution check: draw N = 20,000 times from the known distribution p = (0.5,
// 0.3, 0.15, 0.05) over a vocabulary of 4 and compare Pearson's χ² statistic (3 degrees of freedom)
// against the **theoretical values** (p / p², written by hand in this check). The threshold 21.108
// is the one-sided α = 1e-4 quantile, and there are 3 checks (temperature 1 × 2 seeds + temperature
// 0.5), so the probability of failing with a correct implementation is ≤ 3e-4 by the union bound.
// The seeds are fixed, so once green it stays green deterministically (it does not become "CI that
// fails by chance").
//
// Why it is not a tautology: an implementation that draws uniformly gives χ² ≈ 20,000, and one that
// ignores the temperature (draws from p as is) gives χ² ≈ 9,200 in the temperature-0.5 check; both
// land 2–3 orders of magnitude above the threshold.
// ---------------------------------------------------------------------------

/** The χ² quantile at 3 degrees of freedom, one-sided α = 1e-4. */
const CHI_SQUARE_DF3_1E4 = 21.108;
const DRAWS = 20_000;

const drawCounts = (spec: SamplerSpec, draws: number): number[] => {
  const sampler = createSampler(spec);
  const counts = [0, 0, 0, 0];
  for (let draw = 0; draw < draws; draw += 1) counts[sampler.next(BASE_LOGITS, [])] += 1;
  return counts;
};

const chiSquare = (counts: readonly number[], probabilities: readonly number[]): number => {
  const total = counts.reduce((sum, count) => sum + count, 0);
  return counts.reduce((statistic, observed, index) => {
    const expected = probabilities[index] * total;
    return statistic + (observed - expected) ** 2 / expected;
  }, 0);
};

test("sampler: temperature-1 draws follow the theoretical distribution (χ² check, 3 degrees of freedom, α = 1e-4)", () => {
  for (const seed of [20_260_831, 7]) {
    const counts = drawCounts({ temperature: 1, seed }, DRAWS);
    assertEquals(
      counts.reduce((sum, count) => sum + count, 0),
      DRAWS,
      `seed ${seed}: total`,
    );
    const statistic = chiSquare(counts, BASE_PROBABILITIES);
    assert(
      statistic < CHI_SQUARE_DF3_1E4,
      `seed ${seed}: χ² = ${statistic.toFixed(3)} is at or above ${CHI_SQUARE_DF3_1E4}` +
        ` (observed ${counts.join("/")} / expected ${BASE_PROBABILITIES.map((p) => p * DRAWS).join("/")})`,
    );
  }
});

test("sampler: the temperature sharpens the distribution (matches the temperature-0.5 theoretical distribution)", () => {
  // The theoretical distribution at temperature T is p^(1/T) normalized. At T = 0.5 it is p² / Σp².
  const squared = BASE_PROBABILITIES.map((probability) => probability ** 2);
  const total = squared.reduce((sum, weight) => sum + weight, 0);
  const expected = squared.map((weight) => weight / total);
  const counts = drawCounts({ temperature: 0.5, seed: 11 }, DRAWS);
  const statistic = chiSquare(counts, expected);
  assert(
    statistic < CHI_SQUARE_DF3_1E4,
    `χ² = ${statistic.toFixed(3)} is at or above ${CHI_SQUARE_DF3_1E4}` +
      ` (observed ${counts.join("/")} / expected ${expected
        .map((p) => (p * DRAWS).toFixed(1))
        .join("/")})`,
  );
});

test("sampler: the same seed gives the same sequence and different seeds different ones", () => {
  const draw = (seed: number): number[] => {
    const sampler = createSampler({ temperature: 1, seed });
    return Array.from({ length: 64 }, () => sampler.next(BASE_LOGITS, []));
  };
  assertEquals(draw(5), draw(5), "same seed");
  assert(draw(5).join() !== draw(6).join(), "different seeds gave the same sequence");
});

test("sampler: a token dropped by topK / topP is never drawn", () => {
  // The counterpart of the distribution check (it looks at the **set**, not the frequency). It
  // always fails if the boundary implementation loosens.
  const sampler = createSampler({ temperature: 1, topK: 2, seed: 31 });
  const seen = new Set<number>();
  for (let draw = 0; draw < 5_000; draw += 1) seen.add(sampler.next(BASE_LOGITS, []));
  assertEquals(
    [...seen].sort((a, b) => a - b),
    [0, 1],
    "tokens observed with topK 2",
  );
});

// --------------------------------------------------------------------------- Stop decision (ADR
// 0083 decision 8 — EOS is a set, not a single token)
// ---------------------------------------------------------------------------

test("isStopToken: stops on any member of the set and not outside it", () => {
  // The set from real assets (`generation_config.json` of gemma-4-E2B-it has eos_token_id = [1,
  // 106, 50]).
  const stopTokens = [1, 106, 50] as const;
  for (const token of stopTokens) {
    assert(isStopToken(token, stopTokens), `did not stop on token ${token}`);
  }
  for (const token of [0, 2, 49, 51, 105, 107, 262_143]) {
    assert(!isStopToken(token, stopTokens), `stopped on token ${token}`);
  }
});

test("isStopToken: an empty set never stops (assets that declare no EOS)", () => {
  for (const token of [0, 1, 106]) assert(!isStopToken(token, []));
});

// ---------------------------------------------------------------------------
// Suppressed tokens (the distribution's `suppress_tokens` — HF `SuppressTokensLogitsProcessor`)
// ---------------------------------------------------------------------------
//
// gemma-4-12B-it's generation_config declares suppress_tokens [258883, 258882]; upstream
// `generate()` sets their logits to -inf before selection in greedy and in sampling. The cases put
// a suppressed id at the raw maximum, so an implementation that skips the mask on any path picks
// it and fails.

describe("sampler: suppressed tokens are never selected", () => {
  // Raw argmax = 1 (logit 9); the masked argmax is 3 (logit 4).
  const RAW = f32([0, 9, 1, 4, 2]);
  const SUPPRESSED = [1] as const;

  it("greedy: the suppressed raw maximum is skipped for the masked maximum", () => {
    assertEquals(createSampler().next(RAW, []), 1, "control: unmasked greedy takes the raw max");
    assertEquals(createSampler({}, SUPPRESSED).next(RAW, []), 3);
    assertEquals(samplerDistribution(RAW, {}, [], SUPPRESSED).tokens, Int32Array.of(3));
  });

  it("greedy: a tie after masking keeps the smaller id (argmax's tie-break over the masked row)", () => {
    // The two highest are suppressed; 2 and 4 tie at 5 behind them.
    assertEquals(createSampler({}, [0, 1]).next(f32([9, 9, 5, 1, 5]), []), 2);
  });

  it("sampling with a seed never draws a suppressed id, however likely it was", () => {
    // Without the mask id 1 carries ~99% of the mass (logit 9 vs ≤ 4).
    for (const spec of [
      { temperature: 1, seed: 7 },
      { temperature: 1, topK: 2, seed: 7 },
      { temperature: 1, topP: 0.9, seed: 7 },
      { temperature: 0.7, topK: 5, topP: 0.95, seed: 11 },
    ] satisfies SamplerSpec[]) {
      const masked = createSampler(spec, SUPPRESSED);
      const control = createSampler(spec);
      const seen = new Set<number>();
      let controlHits = 0;
      for (let draw = 0; draw < 2_000; draw += 1) {
        seen.add(masked.next(RAW, []));
        if (control.next(RAW, []) === 1) controlHits += 1;
      }
      assert(!seen.has(1), `${JSON.stringify(spec)}: a suppressed id was drawn`);
      assert(
        controlHits > 1_000,
        `${JSON.stringify(spec)}: control drew id 1 ${controlHits} times`,
      );
    }
  });

  it("the distribution renormalizes over the rest (the suppressed id has probability 0)", () => {
    const { tokens, probabilities } = samplerDistribution(RAW, { temperature: 1 }, [], SUPPRESSED);
    // Ascending id order without narrowing; the suppressed entry keeps its slot with weight 0.
    assertEquals([...tokens], [0, 1, 2, 3, 4]);
    const expected = softmax([0, Number.NEGATIVE_INFINITY, 1, 4, 2]);
    assertProbabilities(probabilities, expected, "masked softmax");
    assertEquals(probabilities[1], 0);
  });

  it("a logit bias cannot bring a suppressed id back (the mask comes after the bias)", () => {
    const spec = { logitBias: [[1, 100]] } satisfies SamplerSpec;
    assertEquals(createSampler(spec).next(RAW, []), 1, "control: the bias alone keeps id 1");
    assertEquals(createSampler(spec, SUPPRESSED).next(RAW, []), 3);
  });

  it("the GPU selection capability is withdrawn while a set is declared", () => {
    assert(createSampler({}).greedy !== undefined, "control: greedy without a set");
    assert(createSampler({}, []).greedy !== undefined, "an empty set is no set");
    assertEquals(createSampler({}, SUPPRESSED).greedy, undefined);
  });

  it("an empty set leaves selection exactly as without the argument", () => {
    const logits = f32([0.3, -1, 2.5, 2.5, 0.1, -4]);
    for (const spec of [
      {},
      { temperature: 1, seed: 3 },
      { temperature: 0.8, topK: 3, seed: 5 },
      { temperature: 1, topP: 0.7, seed: 9 },
    ] satisfies SamplerSpec[]) {
      const without = createSampler(spec);
      const empty = createSampler(spec, []);
      for (let step = 0; step < 200; step += 1) {
        assertEquals(empty.next(logits, []), without.next(logits, []), JSON.stringify(spec));
      }
      assertEquals(
        samplerDistribution(logits, spec, [], []),
        samplerDistribution(logits, spec, []),
      );
    }
  });

  it("the sampler keeps its own copy of the set (a later edit of the caller's array is not seen)", () => {
    const set = [1];
    const sampler = createSampler({}, set);
    set.length = 0;
    assertEquals(sampler.next(RAW, []), 3);
  });

  it("an id outside the logits width is broken wiring (RangeError, not an input error)", () => {
    const error = assertThrows(() => createSampler({}, [5]).next(RAW, []), RangeError);
    assert(error.message.includes("Suppressed token id 5"), error.message);
  });

  it("only the suppressed ids finite: a broken model output, not the request's fault", () => {
    // Every id the request could get is -inf; the mask (not the request) removed the finite one.
    const row = f32([Number.NEGATIVE_INFINITY, 3, Number.NEGATIVE_INFINITY]);
    for (const spec of [{}, { temperature: 1 }] satisfies SamplerSpec[]) {
      const error = assertThrows(() => createSampler(spec, [1]).next(row, []), Error);
      assert(!(error instanceof ModelInputError), `${JSON.stringify(spec)}: ${error.message}`);
    }
    // Control: the caller's ban of the last finite id is still the caller's (input error).
    assertThrows(
      () =>
        createSampler({ logitBias: [[2, Number.NEGATIVE_INFINITY]] }, [1]).next(
          f32([Number.NEGATIVE_INFINITY, 3, 1]),
          [],
        ),
      ModelInputError,
    );
  });
});
