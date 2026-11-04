/**
 * The allow-list of the language gate (ADR 0130, step 2-9h): the tracked files that may carry
 * Japanese text, each with the reason. Everything else in the repository is English, and
 * `packages/runtime/tests/verify_language_test.ts` (the core lane) fails on Japanese outside this
 * list.
 *
 * An entry names one file (exact path) or a directory (a path ending in `/`, a prefix). An exact
 * entry wins over a prefix; among prefixes the longest wins. `maxLines` caps the number of lines
 * with Japanese in that file: data that stays by nature is pinned at its current count, so new
 * Japanese in the file is caught. `until` marks an area that this ADR's staging leaves for a later
 * release; the entry goes when the area is translated.
 *
 * MUST: an entry that no longer matches any Japanese line is reported by the gate as stale — the
 * list ratchets down as the areas are translated and never silently keeps a dead exception.
 */

export type Allowance = {
  /** Repository-relative path; a trailing `/` makes it a directory prefix. */
  readonly path: string;
  /** Why Japanese stays here. */
  readonly why: string;
  /** The release that translates the area (a staging exception), or absent (by nature). */
  readonly until?: "0.15.x";
  /** The number of lines with Japanese this file may carry (only for an exact path). */
  readonly maxLines?: number;
};

const SNAPSHOT_HEADER =
  "a dated snapshot kept behind an English header line (ADR 0130 decision 3 (d))";
const EXCLUDED_ADR =
  "a superseded ADR excluded from translation; carries an English status line (ADR 0130, 2-9b)";
const RESEARCH_ANCHORS = "links into the Japanese headings of docs/research snapshots";
const TEST_COMMENTS = "test comments, names and the tests' own messages (ADR 0130 staging: 0.15.x)";

export const ALLOW_LIST: readonly Allowance[] = [
  // --- by nature: documents -----------------------------------------------------------------
  { path: "docs/research/", why: SNAPSHOT_HEADER },
  { path: "docs/GLOSSARY.ja.md", why: "the Japanese term table (ADR 0130 decision 5)" },
  { path: "docs/decisions/0038-manifest-v1.md", why: EXCLUDED_ADR },
  { path: "docs/decisions/0043-op-addition-layers.md", why: EXCLUDED_ADR },
  { path: "docs/decisions/0071-manifest-v3-shards.md", why: EXCLUDED_ADR },
  { path: "docs/decisions/0072-sbv2-text-injection.md", why: EXCLUDED_ADR },
  {
    path: "docs/decisions/0130-project-language.md",
    why: "the Japanese terms the ADR rules on, as examples",
    maxLines: 7,
  },
  {
    path: "docs/decisions/0128-default-preset-original-weights.md",
    why: "quotes the backlog item it closed",
    maxLines: 1,
  },
  {
    path: "docs/decisions/0131-bf16-weight-execution.md",
    why: "quotes the Japanese title of the preset the ADR replaced",
    maxLines: 1,
  },
  { path: "docs/perf-ledger.md", why: `${RESEARCH_ANCHORS}; quoted commit subjects`, maxLines: 24 },
  { path: "docs/backlog.md", why: RESEARCH_ANCHORS, maxLines: 3 },
  { path: ".claude/ACTIVE_DESIGN.md", why: RESEARCH_ANCHORS, maxLines: 2 },
  {
    path: "docs/known-issues.md",
    why: "an error message quoted as it was observed (2026-10) and test-name filters of the tests still in Japanese",
    maxLines: 4,
  },
  {
    path: "docs/limitations.md",
    why: "an error message quoted as it was observed (2026-10)",
    maxLines: 1,
  },
  {
    path: "docs/release-runbook.md",
    why: "a test-name filter of a test still in Japanese; an rg pattern for an old wording",
    maxLines: 2,
  },
  // --- by nature: data in sources and tests ---------------------------------------------------
  {
    path: "packages/models/src/wan/text-stage.ts",
    why: "the upstream default negative prompt of Wan (Chinese)",
    maxLines: 4,
  },
  {
    path: "tools/export-recipes/moss/prompts.py",
    why: "the fixed prompts of MOSS-Image, quoted verbatim from upstream (Chinese text in them)",
    maxLines: 11,
  },
  {
    path: "packages/models/src/irodori/text/normalize.ts",
    why: "the normalisation tables of the Japanese TTS",
    maxLines: 3,
  },
  {
    path: "packages/models/src/sbv2/text/bert-text.ts",
    why: "tokenizer examples in comments (the Japanese TTS)",
    maxLines: 2,
  },
  {
    path: "packages/models/src/text/detokenizer.ts",
    why: "a detokenizer example in a comment",
    maxLines: 1,
  },
  {
    path: "packages/models/tests/fixtures/",
    why: "vocabularies, sample texts and expected values of the tokenizer and TTS parity fixtures; the notes of the recipe-generated fixtures are translated with the recipes (0.15.x)",
  },
  {
    path: "tools/export-recipes/qwen3_5/tokenizer_cases.py",
    why: "Japanese and Chinese sample prompts of the Qwen3.5 tokenizer fixture",
    maxLines: 5,
  },
  { path: "tools/exporter/tests/test_artifacts.py", why: "a fixture payload", maxLines: 1 },
  {
    path: "tools/exporter/tests/test_dist.py",
    why: "a fixture payload (a LICENSE.md body)",
    maxLines: 1,
  },
  {
    path: "tools/exporter/tests/test_migrate_manifest.py",
    why: "a fixture payload (a stray file)",
    maxLines: 1,
  },
  {
    path: "apps/playground/tests/zip_test.ts",
    why: "a test payload (a non-ASCII entry name for the UTF-8 flag)",
    maxLines: 1,
  },
  {
    path: "apps/playground/src/models/irodori/samples.ts",
    why: "the Japanese TTS inputs of the Irodori page (default text and voice-design examples)",
    maxLines: 8,
  },
  {
    path: "apps/playground/src/models/sbv2/samples.ts",
    why: "the Japanese TTS inputs of the SBV2 page (the regression sentence and demo sentences)",
    maxLines: 3,
  },
  {
    path: "apps/playground/src/models/gemma4/samples.ts",
    why: "a Japanese sample chat message (checks multi-byte streaming detokenization by eye)",
    maxLines: 1,
  },
  { path: "tools/lang/scan-japanese.ts", why: "the JAPANESE character class", maxLines: 1 },
  { path: "tools/lang/py_skeleton.py", why: "the JAPANESE character class", maxLines: 1 },
  {
    path: "tools/lang/check-doc-links.ts",
    why: "the Japanese punctuation a heading slug keeps",
    maxLines: 1,
  },
  { path: "tools/lang/tests/", why: "inputs and expected values of the language tools' own tests" },
  // --- staging: translated in 0.15.x -----------------------------------------------------------
  { path: "packages/runtime/tests/", why: TEST_COMMENTS, until: "0.15.x" },
  { path: "packages/hub/tests/", why: TEST_COMMENTS, until: "0.15.x" },
  { path: "packages/models/tests/", why: TEST_COMMENTS, until: "0.15.x" },
  { path: "packages/e2e/tests/", why: TEST_COMMENTS, until: "0.15.x" },
  {
    path: "tools/export-recipes/",
    why: "the per-model export recipes, their tests and READMEs (ADR 0130 staging: 0.15.x)",
    until: "0.15.x",
  },
];

/** The entry that covers `file`: an exact path first, then the longest directory prefix. */
export const allowanceOf = (
  file: string,
  list: readonly Allowance[] = ALLOW_LIST,
): Allowance | undefined => {
  let best: Allowance | undefined;
  for (const entry of list) {
    if (entry.path.endsWith("/")) {
      if (
        file.startsWith(entry.path) &&
        (best === undefined || entry.path.length > best.path.length)
      ) {
        best = entry;
      }
    } else if (entry.path === file) {
      return entry;
    }
  }
  return best;
};
