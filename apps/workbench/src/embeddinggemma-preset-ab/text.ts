/**
 * The text an EmbeddingGemma preset A/B embeds: a prefix of one golden case of the EmbeddingGemma 2
 * recipe, cut so that the pipeline's token ids (`Pipeline.tokenize` with the case's prompt — the
 * ids `embed` runs, `<bos>` and `<eos>` included) are exactly the asked length. The cases are the
 * tracked fixture the recipe writes, so the long documents are not spelled a second time here; it
 * is read from the models package's test fixtures on purpose — this app is private, and those are
 * the cases the reference rows were taken with. Pure, so `vp test` covers it with a stand-in
 * tokenizer.
 */

import goldenCases from "../../../../packages/models/tests/fixtures/embeddinggemma2/golden-cases.json" with { type: "json" };

/** One golden case: its name, the prompt it is embedded with (none: no prefix), its text and its pinned length. */
export type GoldenCase = {
  readonly name: string;
  readonly prompt?: string;
  readonly text: string;
  /** The length the recipe pinned: the ids of the prompt's prefix and the text, `<bos>` and `<eos>` included. */
  readonly tokens: number;
};

/** The golden cases in the fixture's order. */
export const GOLDEN_CASES: readonly GoldenCase[] = goldenCases.cases;

/** The longest golden case's length: no text of the cases is longer. */
export const LONGEST_CASE_TOKENS: number = Math.max(...GOLDEN_CASES.map(({ tokens }) => tokens));

/**
 * The case a text of `tokens` is cut from: the shortest case of at least that many tokens, the
 * first in the fixture among equals. Throws past the longest case.
 */
export const sourceCase = (cases: readonly GoldenCase[], tokens: number): GoldenCase => {
  const fitting = cases.filter((candidate) => candidate.tokens >= tokens);
  const shortest = fitting.reduce<GoldenCase | undefined>(
    (best, candidate) => (best === undefined || candidate.tokens < best.tokens ? candidate : best),
    undefined,
  );
  if (shortest === undefined) {
    throw new Error(
      `No golden case has ${tokens} tokens or more (the longest has ${Math.max(...cases.map((c) => c.tokens))})`,
    );
  }
  return shortest;
};

/** The text that was cut, and where from. */
export type CutText = {
  /** The golden case it is a prefix of. */
  readonly case: string;
  /** The prompt it is embedded with (absent: no prefix), the case's. */
  readonly prompt?: string;
  readonly text: string;
  /** The prefix's length in code points, of the case's {@link GoldenCase.text}. */
  readonly codePoints: number;
  /** The ids `embed` runs for it: exactly the asked length. */
  readonly tokens: number;
};

/**
 * How far past the prefix the bisection settles on the cut looks for one of exactly the length: one
 * code point can add more than one id (a merge that splits, or byte-fallback ids).
 */
const SCAN_CODE_POINTS = 64;

/**
 * A short prefix of `source`'s text (in whole code points) whose ids are `tokens` long, by `count`
 * (the ids' length of a text with the case's prompt). The prefix is found by bisection over the
 * code points, which assumes the count mostly grows with the prefix (a merge can lower it, and one
 * code point can add several ids), then by a scan of at most {@link SCAN_CODE_POINTS} more code
 * points; the result's length is checked exactly, so it is always `tokens` ids, though not
 * necessarily the shortest such prefix. Throws when the scan finds no prefix of exactly that
 * length: one shorter than the prompt's prefix and `<bos>` `<eos>` alone, or a length the scan
 * steps over.
 */
export const cutText = (
  source: GoldenCase,
  tokens: number,
  count: (text: string) => number,
): CutText => {
  const points = Array.from(source.text);
  const prefix = (length: number): string => points.slice(0, length).join("");
  const whole = count(source.text);
  if (whole < tokens) {
    throw new Error(
      `The golden case ${source.name} is ${whole} tokens, fewer than the ${tokens} asked for`,
    );
  }
  // A prefix whose count reaches `tokens` (the whole text reaches it): the smallest one when the
  // count grows with the prefix.
  let low = 0;
  let high = points.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (count(prefix(middle)) >= tokens) high = middle;
    else low = middle + 1;
  }
  const last = Math.min(points.length, low + SCAN_CODE_POINTS);
  for (let length = low; length <= last; length += 1) {
    const text = prefix(length);
    if (count(text) === tokens) {
      return {
        case: source.name,
        ...(source.prompt === undefined ? {} : { prompt: source.prompt }),
        text,
        codePoints: length,
        tokens,
      };
    }
  }
  throw new Error(
    `No prefix of the golden case ${source.name} is exactly ${tokens} tokens: the prefix of ${low}` +
      ` code points is ${count(prefix(low))}, and none of the next ${last - low} is ${tokens}`,
  );
};
