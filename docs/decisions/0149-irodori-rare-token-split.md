# 0149: Irodori rare-token split — upstream's table in the tokenizer asset, on by default for the text

- Status: proposed (2026-11-14 — follows upstream Irodori-TTS commit `82c9980b`; the recipe, the TS tokenizer and the
  request option are implemented and pass the parity fixture; no distribution carries the table yet)
- Date: 2026-11-14
- Related: [0048](0048-irodori-host-port.md) (the Irodori host port and its shared Unigram layer) /
  [0107](0107-model-input-error.md) (input-caused failures) / [0005](0005-verification.md) (addendum of 2026-10-11:
  parity with upstream at the level of the numbers) / [0139](0139-tests-on-distributions-and-test-assets.md) (where
  goldens live) / [0038](0038-manifest-v1.md) §7 (cross-repository references, through which v4.1-small reads
  v4-small's tokenizer)

## Context

The text tokenizer of Irodori-TTS v4 / v4.1 Small is the Unigram vocabulary of `sbintuitions/modernbert-ja-310m`
(102,400 entries). It keeps many long phrases as one token (upstream's examples are a football club's name, a game
title and "free shipping").
They are rare in the TTS training data, so the model often misreads or skips them.

Upstream commit `82c9980b` ("Avoid rare long tokens of the v4 Small text tokenizer at inference", 2026-10-25) changes
inference only:

- The text is encoded with a copy of the Unigram tokenizer in which selected tokens carry the score `-1e9`, so the
  lattice falls back to shorter, well-trained pieces. A token is selected when it has at least 4 characters and a
  log-probability score of at most -13, unless it is hiragana-only (splitting it does not change the reading but breaks
  prosody) or in `KEEP_TOKENS` (misread once split). `FORCE_SPLITS` lists frequent tokens above the threshold that are
  still misread; after the lattice is solved, their ids are replaced by fixed pieces.
- The threshold is on this tokenizer's score scale, so it applies only when `text_tokenizer_repo` is
  `sbintuitions/modernbert-ja-310m`. It is on by default (`RuntimeKey.split_rare_tokens = True`). The caption tokenizer
  is a separate instance that is never switched. Truncation applies after the split; BOS, padding and mask are
  unchanged.
- Measured effect as upstream states it, from listening tests on v4.1 Small: sentences with hard words went from 41%
  to 75% read correctly; on ordinary sentences there are 8% more tokens and 3% more audio, and no new misreadings.

On the real vocabulary the splitter disables 36,798 tokens and forces 70.

A copy with replaced scores is a different Unigram model from the original with entries removed: the unknown-piece
score of the HF `tokenizers` Unigram is "minimum score − 10", and the minimum of the copy is `-1e9`. An unknown piece
then costs more than a disabled token, so a disabled token whose characters have no other piece is still emitted,
where a vocabulary without it would fall back to bytes.

## Decision

1. **The asset carries upstream's result, not its rule.** The recipe (`tools/export-recipes/irodori/tokenizer_ref.py`)
   runs upstream's `RareTokenSplitter` on the real tokenizer and writes what it did into the tokenizer asset as one
   optional key, written last:

   ```json
   "rareTokenSplit": {
     "disabledScore": -1e9,
     "disabledIds": [ascending ids whose score the splitter replaced],
     "forcedSplits": [[tokenId, [pieceId, ...]], ... ascending by tokenId]
   }
   ```

   The threshold, the length rule, the hiragana exception, `KEEP_TOKENS` and `FORCE_SPLITS` are not restated in
   Karume, so the table follows upstream's lists by construction. Before writing, the recipe checks that the table is
   exactly what the splitter did (every id whose score differs and nothing else, one replaced value, `forcedSplits`
   equal to the splitter's `forced`). It also checks that a tokenizer rebuilt from `tokenizer.json` with only the table
   applied reproduces the splitter on every split case and on the spelling of every disabled and forced token. The
   asset without the key is byte-identical to the asset before the key existed. The key is emitted only when the
   checkpoint's `text_tokenizer_repo` is the one upstream splits.

2. **The TS tokenizer solves the copy as its own Unigram model.** `parseIrodoriTokenizerAsset` reads the key with the
   same strictness as the rest of the asset: ids inside the vocabulary, ascending and unique, every forced expansion
   non-empty, a finite `disabledScore`. A malformed table is an asset error. It computes the minimum score of the
   replaced-score whole vocabulary. `IrodoriTokenizer` builds the second model once at construction: the same
   vocabulary with the disabled tokens' scores replaced and its own minimum. Removing the entries instead would change
   the unknown-piece weight. After the lattice, it replaces every id found in the forced table by its pieces. `encode`
   and `encodePadded` take the path as an explicit, required flag (`{ splitRareTokens }`). The shared Unigram core
   (`src/text/unigram.ts`) is unchanged.

3. **The request option, text only.** `irodori.GenerateRequest.splitRareTokens?: boolean`. Omitted, the split is on
   when the tokenizer asset carries the table and off when it does not, the same default as upstream, where other
   tokenizers are unaffected. An explicit `true` with an asset without the table is a `ModelInputError` (ADR 0107) at
   the entry of `generate` / `generateLatent`, before anything runs. Encoding unsplit there would hide that the
   distribution cannot do what was asked. An explicit `false` always holds. One resolver decides the flag for both
   the entry check and the text stage. The caption path never splits.

4. **The golden.** The parity fixture (`packages/models/tests/fixtures/irodori-text/parity.json`, tracked in git like
   the rest of the tokenizer fixture — it is not a test asset under ADR 0139) carries the table cut to its vocabulary
   subset. `minScore` is that of the replaced-score whole vocabulary. Eight split cases, taken through upstream's
   `enable_rare_token_split()` with the split on and off, cover upstream's examples, `FORCE_SPLITS` (including a piece
   that is not itself a token), `KEEP_TOKENS`, a hiragana-only token below the threshold, an added token with both
   emoji paths, a text that truncates only once split, and a long passage. Four of them also carry the padded
   `batch_encode` form.

## Consequences

- With a distribution whose tokenizer asset carries the table, the default output changes for every text that contains
  a disabled or forced token, as it did upstream. No e2e sha256 row changes: the texts of the Irodori sha rows and of
  the full-loop goldens contain no such token (checked on 2026-11-14). The texts of the i4 calibration corpus
  (`calib_cases.py`) do, and the recipe's reference loops (`pipeline_ref.py`, `calib.py`, `export.py`) encode unsplit.
- The tokenizer asset gains one key (261,160 bytes on 3,232,154). A distribution has to be reassembled to carry it, and
  v4.1-small reads v4-small's tokenizer through a cross-repository reference that pins its bytes, so it picks up the
  table only when it is assembled against the v4-small revision that carries it.
- Older distribution revisions keep working: without the key there is no split path, the default is off, and only an
  explicit `true` is refused.
- The recipe's step 3 needs an upstream clone that includes `82c9980b` (it imports `token_split.py`).
