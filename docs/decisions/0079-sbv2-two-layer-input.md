# 0079: SBV2 synthesis input in two layers (phrase layer → mora layer) — separating the yomi dependency

- Status: accepted (2026-08-25, the maintainer's ruling after a staged design discussion; naming
  ruled the same day; implemented in `7cd82c8`)
- Date: 2026-08-25
- Related: [0072](0072-sbv2-text-injection.md) (injection fields — superseded by this ADR) /
  [0008](0008-public-api.md) (thin public surface) / backlog later (barrel and family-prefix review)
- Implementation: `packages/models/src/sbv2/` (`text/utterance.ts`, `text/model-input.ts`,
  `pipeline.ts`)

## Context

`@karume/models` ran SBV2 text analysis on `@hdae/yomi` internally (the text field, a 19 MB dictionary
behind a Promise cache, overlays, `analyzeProsody` — the only network path bypassing the hub). Dropping
yomi was settled for 0.5.0 (backlog) but deferred until helpers sparing callers extra steps existed.

The model's effective input is three parallel sequences (phonemes, tones, word2ph × BERT), valid only
if ① `sum(word2ph) === phoneme count`, ② phonemes and tones have the same length, ③ all come from
one analysis. ADR 0072's prosody-draft round trip kept them in sync (no derived fields, nucleus
clamping, "return it paired with the same analysis", the phoneme-content check) but was hard to grasp.

## Decision

**models owns two yomi-independent representations (layers) and one conversion between them.**

```
text ──(yomi: analyzeWithWords)──▶ phrase layer ──(toSbv2Utterance)──▶ mora layer ──(generate)──▶ audio
                                   edit = accent nucleus (main UI)     edit = set tone high/low directly
```

1. **Phrase layer = `Sbv2Phrases`**: a karume-owned type that yomi's `analyzeWithWords` result
   (`{ result, words }`) satisfies structurally. Accents are edited here (fix
   `accentPhrases[i].accentNucleus`, split a phrase); the caller then converts again and never
   reimplements the nucleus → tone rule.
2. **Conversion = `toSbv2Utterance(phrases)`**: one pure function. It expands nucleus → tone with the
   upstream g2p rule (upper clamp k = min(nucleus, mora count); flat / head-high / mid-high /
   tail-high; a negative nucleus fails loudly), moves phrase punctuations to the last mora, and folds
   the upstream geminate spelling `cl` into the phoneme `q` (`Sbv2Mora.vowel` is the phoneme). Not
   `sbv2Utterance`, whose "2" reads as the x2y idiom (`toUtterance` once the barrel review drops the
   prefix — filed as later).
3. **Mora layer = `Sbv2Utterance`**, flat (no nested phrases):
   `{ leadingPunctuations, moras: [{ kana, consonant?, vowel, tone: 0|1, punctuations? }], words }`.
   **`tone` is the only effective edit point** (any 0/1 pattern a nucleus cannot express; this
   absorbs the old `givenTone`). `words` is read-only, the source of word2ph and of the BERT text
   (derived from `surface`, so there is no original-text field).
4. **`generate(utterance, options?)`** — one path. The utterance is the only required argument; the
   knobs (speaker / style / … / seed) are an all-optional bag, as in `fromPretrained(ref, options)`.
5. **Removed** (0.6.0 breaking, no shim): the text field, the `dictionary` / `overlay` injection
   fields, `analyzeProsody`, `Sbv2Prosody` and the draft types, `givenTone`. No preview function
   (`sbv2PhoneTone`): tones and phonemes are visible in the object, and one can be added later.
6. **All checks stay in models**: phonemes derived from `moras` match `words` position by position,
   `sum(word2ph) === P`, `tone` is 0 or 1. A reading (phonemes) changes by one route: fix the
   dictionary with an overlay and analyze again (`adjust_word2ph` stays rejected, ADR 0072 decision 8).

## Design history (rejected intermediate designs)

Rejected: three fields (analysis + prosody / givenTone) — several confusing paths left over from
re-analysing text in the pipeline; a flat single layer — the maintainer ruled that expelling the
phrase structure, the main editing surface, goes too far; one-mora phrases for any tone pattern — a
hack that breaks the phrase display (direct tones belong in the mora layer).

## Consequences

- Breaking only in the SBV2 API of models (0.6.0); distribution, manifest and pins are unchanged.
  "Return it paired with the same analysis" disappears: one object is one analysis by construction.
- **Proof that the absorption is bit-identical: the three e2e WAV sha256 checks did not move** (i8 /
  i8+bert4 / i4; expected values untouched; all matched on the real GPU).
- Rewriting `surface` in `words` (it changes the BERT input) is a declared freedom the checks cannot
  detect; `readonly` types and the documentation carry the discipline.
- Preparing the dictionary and the overlay is the caller's job; a glue package can be split out on
  real demand (2026-08-25 ruling). The example `examples/sbv2/` was removed with every Deno example
  (ADR [0127](0127-node-dawn-toolchain.md) decision 5, `bb35fe2a`); the repository has none today.
- `packages/models/src` imports nothing from yomi. Current fact (2026-10-18):
  `packages/models/package.json:29` lists `@hdae/yomi` under `dependencies` (`cc22c086`) though only
  tests use it (`packages/models/tests/sbv2_tone_injection_test.ts`): npm consumers install yomi,
  contrary to this decision.
