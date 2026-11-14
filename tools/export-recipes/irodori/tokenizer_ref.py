r"""Asset and goldens of the Irodori-TTS v4 text preprocessing layer (`normalize_text` + Unigram).

`irodori/export.py` emits the **graphs**; this script handles only "raw text → token id list".
It touches neither the model graph nor the weights (all it reads from the checkpoint is
`bos_token_id` / `pad_token_id` in `__metadata__`).

    cd tools/export-recipes
    uv run --with 'transformers==5.14.1' python -m irodori.tokenizer_ref

Four outputs (default `outputs/series/irodori-v4-small/tokenizer/`, under `.gitignore`):

    tokenizer.json         the runtime asset (vocabText / scores / addedTokens / bosId / padId /
                           unkId / byteBaseId, + rareTokenSplit for the tokenizer upstream
                           splits — below)
    golden.encode.json     fixed `{raw, normalized, ids}` cases, batch_encode equivalence cases,
                           the cases of the **caption path** (strip only) and the rare-token
                           split cases (split off / on)
    golden.normalize.json  `{raw, normalized}` of `normalize_text` (every replacement rule fires
                           at least once)
    nfkc-diff.json         the map `{cp: normalized}` of single code points on which NFKC is
                           **not the identity**

Plus one fixture for the TS-side parity check (**tracked in git**, outside `.gitignore`):

    packages/models/tests/fixtures/irodori-text/parity.json
        the same content as the four above + a **subset** of the vocabulary (split off so that
        the 102,400 entries are not committed; {@link check_subset_covers_lattice} checks that
        the lattice is the same as with the full vocabulary)

After generating, run `deno fmt packages/models/tests/fixtures/irodori-text/parity.json`
(the formatter's output is the committed form — the `fmt --check` of `deno task verify` also
covers the fixtures).

## Shape of the asset (follows anima's T5 asset)

Folded the same way as `parseT5Asset` in `packages/models/src/anima/text/tokenizer.ts`: the
vocabulary is **one string joined with newlines** (0-based line number = id) + a score array. No
token of this Unigram vocabulary contains a newline, a tab or a raw space
({@link check_upstream_shape} checks it every time — a single one would break the line split and
shift every id).

Unlike anima, **there are no `normalizer` / `space` tables**:

- The `normalizer` of `tokenizer.json` is `null` (normalization is done by upstream's
  `normalize_text`, not by the model side — the language's built-in `String.prototype.normalize`
  is enough for NFKC. Only the single code points that may diverge are baked by
  {@link nfkc_diff_table} as material for the comparison)
- The pre_tokenizer is `Metaspace{replacement=▁, prepend_scheme=never, split=false}`, which does
  **not go through** `WhitespaceSplit` and its whitespace table (it only replaces U+0020 with ▁)

## byte_fallback

`unk_id=0`, but `byte_fallback=true`, so a character missing from the vocabulary falls back to a
**byte sequence** of `<0xNN>` (id = `byteBaseId` + byte value; that `byteBaseId` is 15 is checked
every time). The TS side only has to implement "expand to UTF-8 bytes when not in the
vocabulary"; unk stays only as the path for the case without byte_fallback. The c / d cases of
the golden pin this path.

## The rare-token split (ADR 0149)

Upstream encodes the text (never the caption) of models whose `text_tokenizer_repo` is
`sbintuitions/modernbert-ja-310m` with a copy of the Unigram tokenizer in which rare long tokens
carry the score -1e9, then replaces the ids of its FORCE_SPLITS words by fixed pieces
(`irodori_tts/token_split.py`, on by default). The rule is **not** restated here:
{@link RareTokenSplit} runs upstream's `RareTokenSplitter` on the real tokenizer and
{@link derive_rare_token_split} writes what it did into the asset as one key, written last so the
asset without it is byte-identical to the asset before the key existed:

    "rareTokenSplit": {"disabledScore": -1e9,
                       "disabledIds": [ascending ids whose score the splitter replaced],
                       "forcedSplits": [[tokenId, [pieceId, ...]], ... ascending by tokenId]}

The TS side solves the replaced-score vocabulary as its own Unigram model (its own minimum score,
hence its own unknown-piece weight) and then expands the forced ids.

## What is checked

Everything is checked before emitting, and if any check fails **nothing is written** (fail loudly,
ADR 0005):

- The shape of upstream's `tokenizer.json` (model type / unk_id / byte_fallback / normalizer /
  pre_tokenizer / the added_tokens flags / the ids of the 256 byte tokens)
- No vocabulary token contains a newline, a tab or a raw space (the premise of the line split)
- `bos_token_id` / `pad_token_id` **agree between the checkpoint config and added_tokens**
- **The two call paths produce the same id list** — `tokenizers.Tokenizer` (used by `build_cases`
  in `irodori/export.py`) and `transformers.AutoTokenizer` (used by upstream's
  `PretrainedTextTokenizer`). The golden agrees with itself even when only one of them is right,
  so they are compared here
- The byte_fallback case **really** expands to bytes (it matches the expected byte id list)
- The in-vocabulary emoji case **really** is in the vocabulary (otherwise b and c become the
  same test)
- Every replacement rule of `normalize_text` **fires at least once** somewhere in the golden
- The caption cases are **sensitive to normalization** (the strip and normalize id lists
  differ) — upstream applies only strip on the caption path, so with insensitive cases alone a
  mix-up would pass the check unnoticed
- The split table is exactly what the splitter did (every id whose score differs, one replaced
  value, `forcedSplits` equal to the splitter's `forced`), and **the table alone reproduces the
  splitter**: a tokenizer rebuilt from `tokenizer.json` with only the table applied gives the
  splitter's id list on every split case and on the spelling of every disabled and forced token
- The split golden is taken through upstream's entry (`enable_rare_token_split`) and agrees with
  the splitter itself, and every split case exercises the role its reason names
  ({@link check_split_coverage})
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from _shared.paths import REPO_ROOT, SERIES_ROOT

from .export import DEFAULT_MODEL_DIR, DEFAULT_SOURCE_DIR, TOKENIZER_FILE, read_configs

ASSET_FILE = "tokenizer.json"
ENCODE_GOLDEN_FILE = "golden.encode.json"
NORMALIZE_GOLDEN_FILE = "golden.normalize.json"
NFKC_DIFF_FILE = "nfkc-diff.json"

#: The parity fixture (**tracked in git**). Read on the Deno side by
#: `packages/models/tests/irodori_text_test.ts`.
DEFAULT_FIXTURE_PATH = (
    REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "irodori-text" / "parity.json"
)

#: The Metaspace replacement character (U+2581). With `split=false` / `prepend_scheme=never`,
#: pre-tokenization does nothing besides "replace U+0020 with this".
METASPACE = "▁"

#: The base id of the asset's byte tokens (the id of `<0x00>`). **Checked every time instead of
#: trusted** ({@link check_upstream_shape}) — one extra special token upstream shifts all 256.
EXPECTED_BYTE_BASE_ID = 15

#: The expected shape of upstream's pre_tokenizer. `split=false` means "do not split into pieces
#: at ▁", and `prepend_scheme=never` means "do not prepend ▁" (pinned by case i).
EXPECTED_PRE_TOKENIZER = {
    "type": "Metaspace",
    "replacement": "▁",
    "prepend_scheme": "never",
    "split": False,
}

#: The `max_length` of the `batch_encode` equivalence cases. `max_text_len` (256) for text,
#: `max_caption_len` (512) for the caption — the checkpoint config is the source of truth for
#: both, so {@link build_batch_cases} takes the values from the config. They are not hard-coded
#: here.
TEXT_MAX_LENGTH_KEY = "max_text_len"
CAPTION_MAX_LENGTH_KEY = "max_caption_len"

#: The emoji checked to be in the vocabulary (case b).
IN_VOCAB_EMOJI = "\U0001f60a"

#: The emoji checked to be **outside** the vocabulary (cases c / d). It is 4 bytes in UTF-8, so
#: byte_fallback expands it to 4 tokens.
OUT_OF_VOCAB_EMOJI = "\U0001f442"

#: The fixed cases of `golden.encode.json`, `(name, reason, raw text)`.
#:
#: MUST: invisible characters are always written as `\uXXXX` (otherwise they vanish or get
#: swapped for something else during editing). Not exhaustive: one case per boundary where a
#: naive port fails (the same discipline as anima's `PROMPT_CASES`), and the a–l classes are
#: directly the acceptance criteria of the W3 host implementation.
ENCODE_CASES: tuple[tuple[str, str, str], ...] = (
    ("ja-plain", "a: kanji mixed with kana + punctuation", "今日は近くの店まで歩いて行きました。"),
    (
        "ja-formal",
        "a: a longer sentence in polite form (two commas)",
        "本日はお越しいただき、誠にありがとうございます。どうぞごゆっくりお過ごしください。",
    ),
    (
        "ja-question",
        "a: question and exclamation marks (normalization makes them half-width)",
        "本当にそれでいいの？！",
    ),
    ("emoji-in-vocab", "b: in-vocabulary emoji U+1F60A", f"それは嬉しい{IN_VOCAB_EMOJI}ね。"),
    (
        "emoji-byte-fallback",
        "c: a lone out-of-vocabulary emoji U+1F442 → byte_fallback expands it to 4 bytes",
        OUT_OF_VOCAB_EMOJI,
    ),
    (
        "emoji-byte-fallback-pair",
        "d: two out-of-vocabulary emoji in a row — fuse and byte_fallback interact (pinned as is)",
        OUT_OF_VOCAB_EMOJI * 2,
    ),
    (
        "emoji-zwj",
        "e: ZWJ emoji sequence U+1F62E U+200D U+1F4A8",
        "\U0001f62e\u200d\U0001f4a8",
    ),
    ("emoji-vs16", "f: with a variation selector U+23F8 U+FE0F", "\u23f8\ufe0f"),
    (
        "added-token-in-text",
        "g: an added token's string appears in user input (split out before normalization)",
        "設定で <|user|> と書くとどうなるのか。",
    ),
    ("with-spaces", "h: half-width space U+0020 → replaced by U+2581", "hello world foo bar"),
    (
        "leading-space",
        "i: leading space (prepend_scheme=never — ▁ comes from replacement, not prepending)",
        " leading space matters",
    ),
    (
        "latin-digits",
        "j: mixed with Latin letters and digits",
        "GPU 4 台で 2026 年に v1.5 を出す。",
    ),
    (
        "combining-dakuten",
        "k: the interaction of the combining mark U+3099 and NFKC (U+304B + U+3099 → U+304C)",
        "か\u3099っこいいし\u3099ゃんけん",
    ),
    (
        "halfwidth-kana",
        "l: half-width kana + voiced sound mark (NFKC turns them into composed full-width kana)",
        "ｶﾞﾝﾊﾞｬﾁ",
    ),
)

#: The `batch_encode` equivalence cases `(name, reason, raw text, max_length key)`.
#:
#: MUST: one case is long enough that **truncation really fires** (a body over 255 against
#: `max_text_len` = 256). With only cases that do not fire, "the truncation order (before or
#: after the BOS)" never shows in the golden, and a mix-up on the TS side passes unnoticed.
BATCH_CASES: tuple[tuple[str, str, str, str], ...] = (
    (
        "batch-short",
        "BOS prepended + right padding (truncation does not fire)",
        "今日は近くの店まで歩いて行きました。",
        TEXT_MAX_LENGTH_KEY,
    ),
    (
        "batch-caption",
        "the same steps with the caption-side length (max_caption_len)",
        "若く元気な女性の声。カフェの店員のように、明るくハキハキとした少し高めのトーンで話している。",
        CAPTION_MAX_LENGTH_KEY,
    ),
)

#: The preprocessing cases of the **caption side** `(name, reason, raw text)`.
#:
#: Upstream's `inference_runtime._synthesize` applies only `str(...).strip()` to the caption
#: (`normalize_text` is for the text only). Every spelling listed here is **sensitive to
#: normalization**: applying `normalize_text` to the caption path by mistake changes the id list,
#: so the check of {@link build_caption_cases} that "the strip and normalize versions differ"
#: holds.
CAPTION_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "caption-brackets",
        "outer brackets: normalize strips the corner brackets U+300C U+300D / caption keeps them",
        "「若く元気な女性の声」",
    ),
    (
        "caption-fullwidth",
        "NFKC: normalize folds Ａ to A / the caption keeps it full-width",
        "Ａの声",
    ),
    (
        "caption-circled",
        "symbol removal: normalize removes ① / the caption keeps it",
        "①の声",
    ),
    (
        "caption-semicolon",
        "symbol removal: normalize removes ; / the caption keeps it",
        "声;高め",
    ),
    (
        "caption-surrounding-space",
        "strip trims only the ends — caption keeps inner ideographic spaces (normalize drops them)",
        "\u3000明るい\u3000声\u3000",
    ),
    (
        "caption-official",
        "the official card's Voice Design text (insensitive to normalization, like the old golden)",
        "若く元気な女性の声。カフェの店員のように、明るくハキハキとした少し高めのトーンで話している。",
    ),
)

#: A long passage (the body of `batch-truncated`). The same style as `LONG_PASSAGE_A` in
#: `irodori/export.py`; **one paragraph is not enough** (T=144), so it is three paragraphs long —
#: the point is to really hit the truncation at `max_text_len` = 256.
TRUNCATING_PASSAGE = (
    "秋の朝、駅前の商店街はまだ静かで、シャッターの下りた店の前を、通勤の人がまばらに"
    "歩いていた。パン屋の換気口から流れてくる甘い匂いだけが、これから一日が始まることを"
    "教えている。私は改札の手前で立ち止まり、鞄の中をもう一度だけ確かめた。定期券、財布、"
    "それから昨日の夜にようやく書き上げた原稿の束。指先にざらついた紙の感触が伝わってきて、"
    "そこで初めて、これが現実の予定なのだという実感が戻ってきた。ホームに上がると、向かいの"
    "線路の向こうに、朝日を受けた高いビルが並んでいるのが見えた。ガラスの壁面がひとつずつ"
    "順番に光り始め、まるで街全体がゆっくりと目を覚ましていくようだった。遠くから電車の"
    "近づいてくる音が聞こえてくる。私は深く息を吸い込み、今日これから会う人たちのことを"
    "順番に思い浮かべた。打ち合わせの場所は、線路沿いの古い建物の三階だった。エレベーターが"
    "無いので階段を上がるしかなく、途中の踊り場で一度だけ立ち止まって呼吸を整えた。扉を"
    "開けると、細長い部屋の奥に大きな机がひとつ置かれていて、その上には前回の打ち合わせで"
    "使った資料がそのまま積まれていた。窓は北向きで、季節に関わらず光は柔らかい。"
    "担当の人はまだ来ていなかったので、私は椅子に腰を下ろし、持ってきた原稿をもう一度"
    "読み返した。読み返すたびに直したい箇所が見つかるのは毎回のことで、けれども今日は、"
    "書いたときに考えていたことがそのまま残っているように思えた。階段を上がってくる足音が"
    "近づいてきて、私は原稿を閉じ、顔を上げて扉のほうを見た。"
)

#: The asset key of the rare-token split table (ADR 0149). Present only for a tokenizer upstream
#: applies its `RareTokenSplitter` to; the asset without it stays byte-identical to the asset
#: before the key existed.
RARE_TOKEN_SPLIT_KEY = "rareTokenSplit"

#: The cases of the **rare-token split** golden `(name, reason, raw text)`, each solved by upstream
#: with the split on (`PretrainedTextTokenizer.enable_rare_token_split()`) and off.
#:
#: MUST: {@link check_split_coverage} checks that every role named in a reason really happens on
#: the real vocabulary (a disabled token falls back, a forced id expands, a KEEP / hiragana-only
#: token below the threshold stays one token, truncation fires only on the split side). A rule
#: change upstream that turns one of them into a no-op fails the emit instead of leaving a case
#: that tests nothing.
SPLIT_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "split-commit-examples",
        "disabled tokens fall back to shorter pieces and a forced token expands"
        " (the examples of upstream's commit message)",
        "浦和レッズの試合を見ながら、ゼルダの伝説を遊んだ。送料無料です。",
    ),
    (
        "split-forced",
        "FORCE_SPLITS entries above the threshold, two of them into three pieces",
        "セブンイレブンで厚生労働省の代表取締役社長に会った。",
    ),
    (
        "split-forced-solved-piece",
        "a FORCE_SPLITS piece that is not a token itself: the table holds what the splitter solved"
        " it into",
        "リハビリテーションに通う",
    ),
    (
        "split-keep",
        "KEEP_TOKENS below the threshold stay one token (both paths agree)",
        "心筋梗塞と瀬戸内海",
    ),
    (
        "split-hiragana-only",
        "a hiragana-only token below the threshold is never disabled; the next word is",
        "わたくしが参ります。",
    ),
    (
        "split-added-token-emoji",
        "an added token and both emoji paths (in vocabulary / byte_fallback) next to a disabled"
        " token",
        f"設定で <|user|> と書くと{IN_VOCAB_EMOJI}浦和レッズ{OUT_OF_VOCAB_EMOJI}",
    ),
    (
        "split-truncates",
        "fits max_text_len unsplit but not split: truncation applies after the split",
        "浦和レッズとゼルダの伝説、" * 40,
    ),
    (
        "split-long-passage",
        "a long passage that truncates on both paths",
        TRUNCATING_PASSAGE,
    ),
)

#: The split cases whose padded `batch_encode` form (BOS + right padding + mask, at
#: `max_text_len`) also goes into the golden. The others carry the plain id lists only (the
#: padded form of every case would multiply the fixture for no extra boundary).
SPLIT_BATCH_CASES = frozenset(
    {"split-commit-examples", "split-added-token-emoji", "split-truncates", "split-long-passage"}
)

#: Token spellings the coverage check looks for, per role (they come from upstream's lists and
#: commit message; {@link check_split_coverage} reads their ids from the vocabulary).
SPLIT_KEEP_TOKEN = "心筋梗塞"
SPLIT_HIRAGANA_TOKEN = "わたくし"

#: The fixed cases of `golden.normalize.json` `(name, reason, raw text)`.
#:
#: `normalize_text` has five stages, SIMPLE_REPLACE_MAP → REGEX_REPLACE_MAP →
#: `strip_outer_brackets` → NFKC → `...`/`..` to `…`, and the list includes cases where **the
#: order itself is observable** (e.g. `①` is removed before NFKC and so does not become `1` / the
#: full-width `；` slips past the removal rule and becomes `;` under NFKC).
NORMALIZE_CASES: tuple[tuple[str, str, str], ...] = (
    ("plain", "a plain sentence where nothing happens", "今日は近くの店まで歩いて行きました。"),
    ("tab", "SIMPLE: tab removal", "前\t後ろ"),
    ("bracket-n-plain", "SIMPLE: [n] removal", "改行[n]の指示"),
    (
        "bracket-n-escaped",
        r"SIMPLE: \[n\] removal (a separate rule for the escaped spelling)",
        r"改行\[n\]の指示",
    ),
    (
        "ideographic-space",
        "SIMPLE: ideographic space U+3000 removal (before NFKC, so it is not made half-width)",
        "前\u3000後ろ",
    ),
    ("question-bang", "SIMPLE: ？ → ? / ！ → !", "本当？そうだ！"),
    ("heart", "SIMPLE: ♥ → ♡", "だいすき♥"),
    ("circles", "SIMPLE: ● ◯ 〇 → ○", "●と◯と〇"),
    ("regex-symbols", "REGEX: removal of ; ▼ ♀ ♂ 《》 ≪≫ ①〜⑥", "あ;い▼う♀え♂お《か》き≪く≫け①こ⑥"),
    (
        "regex-symbols-before-nfkc",
        "order: ① is removed before NFKC (no `1`) / full-width ； survives and NFKC makes it ;",
        "①番目；ここ",
    ),
    (
        "dashes",
        "REGEX: removal of dashes (U+2010〜U+2015 / U+2212 / U+2500 and others)",
        "あ‐い―う−え─お",
    ),
    (
        "dash-modifier",
        "REGEX: removal of U+02D7 / U+2043 / U+23AF / U+23E4 / U+2E3A / U+2E3B",
        "あ˗い⁃う⎯え⏤お⸺か⸻き",
    ),
    ("wave-dash", "REGEX: ～(U+FF5E) / 〜(U+301C) → ー", "あ～い〜う"),
    ("ellipsis-run", "REGEX: …{3,} → …… (only for three or more)", "ええ………そうですか"),
    (
        "ellipsis-two",
        "REGEX: two … are not contracted (the lower side of the boundary)",
        "ええ……そうですか",
    ),
    (
        "brackets-kagi",
        "bracket stripping: corner brackets U+300C U+300D enclose the whole text",
        "「こんにちは」",
    ),
    (
        "brackets-nested",
        "bracket stripping: nested (stripped twice, the corner brackets first, then （）)",
        "「（こんにちは）」",
    ),
    ("brackets-double", "bracket stripping: 『』", "『こんにちは』"),
    ("brackets-lenticular", "bracket stripping: 【】", "【こんにちは】"),
    ("brackets-ascii", "bracket stripping: ASCII ()", "(hello)"),
    (
        "brackets-not-enclosing",
        "bracket stripping: not stripped, since they do not enclose the whole text",
        "「あ」と「い」",
    ),
    ("brackets-asymmetric", "bracket stripping: a mismatched pair is not stripped", "「あ）"),
    ("brackets-inner", "bracket stripping: brackets in the middle stay", "（あ（い）う）"),
    ("nfkc-fullwidth", "NFKC: full-width letters and digits → half-width", "ＡＢＣ１２３ｄｅｆ"),
    (
        "nfkc-halfwidth-kana",
        "NFKC: half-width kana + voiced sound mark → composed full-width",
        "ｶﾞﾝﾊﾞｬﾁ",
    ),
    ("nfkc-square", "NFKC: decomposition of the compatibility characters ㈱ ℃ U+3314", "㈱と℃と㌔"),
    ("nfkc-roman", "NFKC: the Roman numeral Ⅻ, the ligature ﬁ and the fraction ½", "Ⅻとﬁと½"),
    (
        "nfkc-combining",
        "NFKC: U+304B + U+3099 → U+304C (composition with a combining mark)",
        "か\u3099っこいい",
    ),
    (
        "nfkc-dakuten-alone",
        "NFKC: a lone voiced sound mark U+309B decomposes into a space + a combining mark",
        "あ゛い",
    ),
    ("dots-three", "trailing: ... → …", "そうですね..."),
    ("dots-two", "trailing: .. → …", "そうですね.."),
    ("dots-four", "trailing: order (... matches first, so .... becomes ….)", "そうですね...."),
    (
        "combined",
        "combined: ideographic space + ？ + dash + brackets + NFKC + ... overlap in one sentence",
        "「本当に\u3000そうなの？―ＡＢＣ...」",
    ),
)


def _tokenizer_json(model_dir: Path) -> dict[str, Any]:
    path = model_dir / TOKENIZER_FILE
    if not path.is_file():
        raise SystemExit(f"tokenizer.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_upstream_shape(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Check that upstream's `tokenizer.json` matches the premises of how the asset is folded.

    MUST: do not leave this as an assumption — emitting with a premise broken lets the id list
    alone silently change on the TS side (a broken line split / a shifted byte_fallback base / a
    different splitting rule for the added tokens).
    """
    model = raw["model"]
    if model.get("type") != "Unigram":
        raise SystemExit(f"model.type is not Unigram: {model.get('type')}")
    if int(model.get("unk_id", -1)) != 0:
        raise SystemExit(f"unk_id is not 0: {model.get('unk_id')}")
    if not bool(model.get("byte_fallback")):
        raise SystemExit(
            "byte_fallback is false — out-of-vocabulary characters would be handled differently"
        )
    if raw.get("normalizer") is not None:
        raise SystemExit(
            f"normalizer is not null: {raw['normalizer']} — a normalization table would be needed"
        )
    if raw.get("pre_tokenizer") != EXPECTED_PRE_TOKENIZER:
        raise SystemExit(
            f"pre_tokenizer differs from {EXPECTED_PRE_TOKENIZER}: {raw.get('pre_tokenizer')}"
        )
    # The flags of the added tokens should all be at their defaults (if any of normalized /
    # lstrip / rstrip / single_word is set, the meaning of the copied rule "split them out
    # leftmost-longest before normalization" changes).
    for entry in raw["added_tokens"]:
        flags = {key: entry[key] for key in ("lstrip", "rstrip", "single_word", "normalized")}
        if any(flags.values()):
            raise SystemExit(f"added token {entry['content']!r} has non-default flags: {flags}")

    vocab: list[list[Any]] = model["vocab"]
    for index, (token, _score) in enumerate(vocab):
        if "\n" in token or "\t" in token or " " in token:
            raise SystemExit(
                f"vocabulary id {index}: the token {token!r} contains a newline / tab / space"
            )
    byte_ids = {
        int(token[3:5], 16): index
        for index, (token, _score) in enumerate(vocab)
        if len(token) == 6 and token.startswith("<0x") and token.endswith(">")
    }
    if len(byte_ids) != 256:
        raise SystemExit(f"{len(byte_ids)} byte tokens (expected 256)")
    if any(index != EXPECTED_BYTE_BASE_ID + value for value, index in byte_ids.items()):
        raise SystemExit(
            f"the byte token ids are not laid out as {EXPECTED_BYTE_BASE_ID} + byte value"
        )
    return {
        "vocab": len(vocab),
        "addedTokens": len(raw["added_tokens"]),
        "byteBaseId": EXPECTED_BYTE_BASE_ID,
    }


def build_asset(
    raw: Mapping[str, Any],
    text_config: Mapping[str, Any],
    rare_token_split: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The asset the Deno side uses by lookup alone (folded like anima's T5 asset).

    MUST: `bosId` / `padId` come from the **checkpoint config** and are checked against the
    spelling of the added tokens. Looking only at the tokenizer's `added_tokens` breaks silently
    when the config points at another id: a different id from the BOS the model expects gets
    prepended.

    `rare_token_split` (the table of {@link derive_rare_token_split}) becomes the **last** key, so
    the asset of a tokenizer without it keeps the exact bytes it had before the key existed and
    the asset with it differs from that by the one key only.
    """
    asset = _base_asset(raw, text_config)
    if rare_token_split is not None:
        asset[RARE_TOKEN_SPLIT_KEY] = dict(rare_token_split)
    return asset


def _base_asset(raw: Mapping[str, Any], text_config: Mapping[str, Any]) -> dict[str, Any]:
    """The keys of the asset that every tokenizer has (the body of {@link build_asset})."""
    vocab: list[list[Any]] = raw["model"]["vocab"]
    added = {entry["content"]: int(entry["id"]) for entry in raw["added_tokens"]}
    bos_id = int(text_config["bos_token_id"])
    pad_id = int(text_config["pad_token_id"])
    for label, token, want in (("bos", "<s>", bos_id), ("pad", "<pad>", pad_id)):
        if added.get(token) != want:
            raise SystemExit(
                f"{label}: the config id {want} differs from the added token {token!r}, whose id"
                f" is {added.get(token)}"
            )
    return {
        "vocabText": "\n".join(token for token, _score in vocab),
        "scores": [float(score) for _token, score in vocab],
        # `[string, id]` pairs (the shape of `parseAddedTokens` on the TS side), in ascending id.
        "addedTokens": [
            [content, value] for content, value in sorted(added.items(), key=lambda item: item[1])
        ],
        "bosId": bos_id,
        "padId": pad_id,
        "unkId": int(raw["model"]["unk_id"]),
        "byteBaseId": EXPECTED_BYTE_BASE_ID,
    }


def byte_fallback_ids(text: str) -> list[int]:
    """The id list of `text` written with byte_fallback alone (computes case c's expectation
    independently)."""
    return [EXPECTED_BYTE_BASE_ID + byte for byte in text.encode("utf-8")]


class TokenizerPair:
    """Holds the **two paths** to an id list (raw `tokenizers` / through `transformers`) and
    compares them.

    `build_cases` in `irodori/export.py` uses the former, upstream's `PretrainedTextTokenizer` the
    latter. The golden can only be taken from one of them, so **both producing the same list** is
    checked per case before it is taken (if only one breaks, the golden still agrees with itself).
    """

    def __init__(self, model_dir: Path) -> None:
        from irodori_tts.tokenizer import PretrainedTextTokenizer
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer

        self.raw = Tokenizer.from_file(str(model_dir / TOKENIZER_FILE))
        self.hf = AutoTokenizer.from_pretrained(str((model_dir / TOKENIZER_FILE).parent))
        self.wrapper = PretrainedTextTokenizer(self.hf, add_bos=True)

    def encode(self, text: str, where: str) -> list[int]:
        """The plain id list without special tokens (called like upstream's
        `PretrainedTextTokenizer.encode`)."""
        raw_ids = list(self.raw.encode(text, add_special_tokens=False).ids)
        hf_ids = list(self.hf.encode(text, add_special_tokens=False))
        if raw_ids != hf_ids:
            raise SystemExit(
                f"{where}: the id lists of tokenizers and transformers differ\n"
                f"  tokenizers={raw_ids}\n  transformers={hf_ids}"
            )
        return raw_ids


def derive_rare_token_split(
    original_vocab: Sequence[Sequence[Any]],
    split_vocab: Sequence[Sequence[Any]],
    forced: Mapping[int, Sequence[int]],
) -> dict[str, Any]:
    """The asset's rare-token split table, read from what upstream's splitter **did**.

    `original_vocab` / `split_vocab` are the `model.vocab` lists of the tokenizer before and after
    upstream's `RareTokenSplitter` replaced the scores, and `forced` is the splitter's `forced` map
    (token id → piece ids). The rule itself (threshold, length, hiragana exception, KEEP_TOKENS,
    FORCE_SPLITS) is not restated here, so the table follows upstream's lists by construction.

    MUST: the table holds every id whose score differs and nothing else, and all the replaced
    scores are one value. A replaced-score copy whose vocabulary moved, a second replacement value
    or an empty result means the splitter changed shape — fail rather than write a table the TS
    side would reproduce wrongly.
    """
    if len(original_vocab) != len(split_vocab):
        raise SystemExit(
            f"the split vocabulary has {len(split_vocab)} entries, the original"
            f" {len(original_vocab)}"
        )
    disabled: list[int] = []
    replaced: set[float] = set()
    for index, (before, after) in enumerate(zip(original_vocab, split_vocab, strict=True)):
        if before[0] != after[0]:
            raise SystemExit(
                f"vocabulary id {index}: the split copy holds {after[0]!r} instead of {before[0]!r}"
            )
        if float(before[1]) != float(after[1]):
            disabled.append(index)
            replaced.add(float(after[1]))
    if not disabled:
        raise SystemExit("the splitter replaced no score (an empty table would split nothing)")
    if len(replaced) != 1:
        raise SystemExit(f"the splitter replaced scores with {len(replaced)} different values")
    (disabled_score,) = replaced
    if not math.isfinite(disabled_score):
        raise SystemExit(f"the replaced score is not finite: {disabled_score}")

    size = len(original_vocab)
    disabled_set = set(disabled)
    forced_rows: list[list[Any]] = []
    for token_id in sorted(forced):
        pieces = [int(piece) for piece in forced[token_id]]
        if not pieces:
            raise SystemExit(f"forced token {token_id} has no pieces")
        if not 0 <= int(token_id) < size or any(not 0 <= piece < size for piece in pieces):
            raise SystemExit(f"forced token {token_id} → {pieces} leaves the vocabulary")
        if token_id in disabled_set:
            raise SystemExit(
                f"forced token {token_id} is also disabled (it could never be emitted)"
            )
        forced_rows.append([int(token_id), pieces])
    return {
        "disabledScore": disabled_score,
        "disabledIds": disabled,
        "forcedSplits": forced_rows,
    }


def split_min_score(raw_vocab: Sequence[Sequence[Any]], table: Mapping[str, Any]) -> float:
    """The minimum score of the **replaced-score** vocabulary (the weight of its unknown nodes).

    MUST: computed over the whole vocabulary with the scores replaced, not taken from the original
    — the Unigram model of the split copy derives its unknown-piece score from its own minimum.
    """
    disabled = set(table["disabledIds"])
    return min(
        float(table["disabledScore"]) if index in disabled else float(score)
        for index, (_token, score) in enumerate(raw_vocab)
    )


class RareTokenSplit:
    """Upstream's rare-token split on the real tokenizer, and the asset table derived from it.

    The splitter is built exactly as upstream builds it (`RareTokenSplitter` over the
    `transformers` backend tokenizer), and the golden is taken through upstream's own entry
    (`PretrainedTextTokenizer.enable_rare_token_split()`); {@link RareTokenSplit.encode} checks
    that the two agree, the way {@link TokenizerPair.encode} does for the plain path.
    """

    def __init__(self, pair: TokenizerPair, raw_vocab: Sequence[Sequence[Any]]) -> None:
        from irodori_tts.token_split import FORCE_SPLITS, SCORE_THRESHOLD, RareTokenSplitter
        from irodori_tts.tokenizer import PretrainedTextTokenizer

        backend = pair.hf.backend_tokenizer
        original = json.loads(backend.to_str())["model"]["vocab"]
        if [(token, float(score)) for token, score in original] != [
            (token, float(score)) for token, score in raw_vocab
        ]:
            raise SystemExit("the transformers backend's vocabulary differs from tokenizer.json")
        self.splitter = RareTokenSplitter(backend)
        split_vocab = json.loads(self.splitter.tokenizer.to_str())["model"]["vocab"]
        self.table = derive_rare_token_split(original, split_vocab, self.splitter.forced)
        if self.table["forcedSplits"] != [
            [token_id, list(pieces)] for token_id, pieces in sorted(self.splitter.forced.items())
        ]:
            raise SystemExit("forcedSplits differs from the splitter's forced map")
        self.force_splits: Mapping[str, Sequence[str]] = FORCE_SPLITS
        self.threshold = float(SCORE_THRESHOLD)
        self.wrapper = PretrainedTextTokenizer(pair.hf, add_bos=True)
        self.wrapper.enable_rare_token_split()

    def encode(self, text: str, where: str) -> list[int]:
        """The split id list without special tokens (upstream's `encode(text, add_bos=False)`)."""
        wrapper_ids = [int(value) for value in self.wrapper.encode(text, add_bos=False).tolist()]
        splitter_ids = list(self.splitter.encode(text))
        if wrapper_ids != splitter_ids:
            raise SystemExit(
                f"{where}: the split id lists of the wrapper and the splitter differ\n"
                f"  wrapper={wrapper_ids}\n  splitter={splitter_ids}"
            )
        return wrapper_ids


def rare_token_split_for(
    pair: TokenizerPair, raw: Mapping[str, Any], model_config: Mapping[str, Any]
) -> RareTokenSplit | None:
    """The split for a model whose text tokenizer upstream splits, `None` for any other.

    Upstream enables the split only when `text_tokenizer_repo` is the tokenizer whose score scale
    the threshold was measured on (`inference_runtime.InferenceRuntime` / `SPLIT_TOKENIZER_REPO`),
    so the same condition decides whether the asset gets the table.
    """
    from irodori_tts.token_split import SPLIT_TOKENIZER_REPO

    repo = model_config.get("text_tokenizer_repo")
    if repo is None:
        raise SystemExit("the checkpoint config has no text_tokenizer_repo")
    if repo != SPLIT_TOKENIZER_REPO:
        return None
    return RareTokenSplit(pair, raw["model"]["vocab"])


def check_table_reproduces_splitter(
    raw: Mapping[str, Any],
    table: Mapping[str, Any],
    split_encode: Any,
    texts: Sequence[str],
) -> int:
    """Check that the **table alone** reproduces the splitter (what the TS side relies on).

    A tokenizer is rebuilt from `tokenizer.json` with only the table's ids set to its score (its
    own Unigram model, so its own minimum score), followed by the forced-id expansion, and its id
    lists are compared with `split_encode` (upstream's splitter) over `texts`. Not tautological:
    the rebuilt side never sees upstream's rule or lists.
    """
    from tokenizers import Tokenizer

    data = copy.deepcopy(dict(raw))
    vocab = data["model"]["vocab"]
    for index in table["disabledIds"]:
        vocab[index][1] = table["disabledScore"]
    rebuilt = Tokenizer.from_str(json.dumps(data, ensure_ascii=False))
    rebuilt.no_padding()
    rebuilt.no_truncation()
    forced = {int(token_id): list(pieces) for token_id, pieces in table["forcedSplits"]}
    for text in texts:
        ids: list[int] = []
        for token_id in rebuilt.encode(text, add_special_tokens=False).ids:
            ids.extend(forced.get(token_id, (token_id,)))
        want = list(split_encode(text))
        if ids != want:
            raise SystemExit(
                f"the table does not reproduce the splitter on {text!r}\n"
                f"  table={ids}\n  splitter={want}"
            )
    return len(texts)


def reproduction_texts(
    raw_vocab: Sequence[Sequence[Any]],
    table: Mapping[str, Any],
    split_cases: Sequence[Mapping[str, Any]],
) -> list[str]:
    """The texts {@link check_table_reproduces_splitter} runs: every split case, and the spelling
    of every disabled and every forced token on its own (each one is a lattice the replaced scores
    decide)."""
    ids = [*table["disabledIds"], *(token_id for token_id, _pieces in table["forcedSplits"])]
    return [*(case["normalized"] for case in split_cases), *(raw_vocab[index][0] for index in ids)]


def build_split_cases(
    pair: TokenizerPair, split: RareTokenSplit, normalize_text: Any, model_config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The rare-token split golden: each case with the split off (`ids`) and on (`idsSplit`).

    The body is `normalize_text(raw).strip()`, the string the pipeline tokenizes. The cases of
    {@link SPLIT_BATCH_CASES} also carry upstream's padded `batch_encode` form on both paths
    (truncation of the body at `max_text_len - 1` happens **after** the split upstream).
    """
    max_length = int(model_config[TEXT_MAX_LENGTH_KEY])
    cases: list[dict[str, Any]] = []
    for name, why, text in SPLIT_CASES:
        normalized = normalize_text(text).strip()
        case: dict[str, Any] = {
            "name": name,
            "why": why,
            "raw": text,
            "normalized": normalized,
            "ids": pair.encode(normalized, name),
            "idsSplit": split.encode(normalized, name),
        }
        if name in SPLIT_BATCH_CASES:
            case["maxLength"] = max_length
            for suffix, wrapper, body in (
                ("", pair.wrapper, case["ids"]),
                ("Split", split.wrapper, case["idsSplit"]),
            ):
                ids, mask = wrapper.batch_encode([normalized], max_length=max_length)
                used = int(mask[0].sum())
                if used != min(len(body) + 1, max_length):
                    raise SystemExit(
                        f"{name}: the mask's valid length {used} differs from BOS + body ({suffix})"
                    )
                if [int(value) for value in ids[0, 1:used].tolist()] != body[: used - 1]:
                    raise SystemExit(
                        f"{name}: the padded body is not the encoded prefix ({suffix})"
                    )
                case[f"idsPadded{suffix}"] = [int(value) for value in ids[0].tolist()]
                case[f"mask{suffix}"] = [bool(value) for value in mask[0].tolist()]
        cases.append(case)
    return cases


def check_split_coverage(
    cases: Sequence[Mapping[str, Any]],
    table: Mapping[str, Any],
    raw_vocab: Sequence[Sequence[Any]],
    force_splits: Mapping[str, Sequence[str]],
    threshold: float,
    max_text_len: int,
) -> dict[str, Any]:
    """Check that every split case exercises the role its reason names (the MUST of
    {@link SPLIT_CASES}).

    Judged on the measured id lists against the table and the vocabulary, so a case that stopped
    testing its boundary (upstream moved a token into KEEP_TOKENS, a forced split now matches the
    vocabulary, …) fails here.
    """
    by_name = {case["name"]: case for case in cases}
    disabled = set(table["disabledIds"])
    forced = {token_id: pieces for token_id, pieces in table["forcedSplits"]}
    ids_of = {token: index for index, (token, _score) in enumerate(raw_vocab)}
    problems: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    commit = by_name["split-commit-examples"]
    require(any(i in disabled for i in commit["ids"]), "commit examples: no disabled token")
    require(any(i in forced for i in commit["ids"]), "commit examples: no forced token")
    require(
        not any(i in disabled for i in commit["idsSplit"]),
        "commit examples: a disabled token survived the split",
    )

    forced_case = [i for i in by_name["split-forced"]["ids"] if i in forced]
    require(len(forced_case) >= 2, "forced: fewer than two forced tokens")
    require(any(len(forced[i]) >= 3 for i in forced_case), "forced: no three-piece expansion")

    solved = [i for i in by_name["split-forced-solved-piece"]["ids"] if i in forced]
    require(
        any(len(forced[i]) > len(force_splits[raw_vocab[i][0]]) for i in solved),
        "forced-solved-piece: no forced token whose pieces had to be solved",
    )

    for name, token in (
        ("split-keep", SPLIT_KEEP_TOKEN),
        ("split-hiragana-only", SPLIT_HIRAGANA_TOKEN),
    ):
        case = by_name[name]
        token_id = ids_of[token]
        require(
            float(raw_vocab[token_id][1]) <= threshold, f"{name}: {token!r} is above the threshold"
        )
        require(token_id not in disabled, f"{name}: {token!r} is disabled")
        require(
            token_id in case["ids"] and token_id in case["idsSplit"],
            f"{name}: {token!r} is not one token on both paths",
        )
    require(
        by_name["split-keep"]["ids"] == by_name["split-keep"]["idsSplit"],
        "keep: the split changed the case",
    )
    require(
        by_name["split-hiragana-only"]["ids"] != by_name["split-hiragana-only"]["idsSplit"],
        "hiragana-only: nothing next to the kept token was split",
    )

    added = by_name["split-added-token-emoji"]
    require(any(i in disabled for i in added["ids"]), "added-token-emoji: no disabled token")
    require(ids_of["<|user|>"] in added["idsSplit"], "added-token-emoji: the added token is lost")
    require(
        byte_fallback_ids(OUT_OF_VOCAB_EMOJI)
        == [
            i for i in added["idsSplit"] if EXPECTED_BYTE_BASE_ID <= i < EXPECTED_BYTE_BASE_ID + 256
        ],
        "added-token-emoji: the byte_fallback emoji is not the only byte run",
    )

    truncates = by_name["split-truncates"]
    require(
        len(truncates["ids"]) + 1 <= max_text_len < len(truncates["idsSplit"]) + 1,
        "truncates: truncation does not fire on the split side only",
    )
    require(
        len(by_name["split-long-passage"]["ids"]) + 1 > max_text_len,
        "long passage: truncation does not fire unsplit",
    )
    if problems:
        raise SystemExit(f"the split cases lost their roles: {problems}")
    return {
        "changed": sum(1 for case in cases if case["ids"] != case["idsSplit"]),
        "cases": len(cases),
    }


def build_encode_cases(pair: TokenizerPair, normalize_text: Any) -> list[dict[str, Any]]:
    """The fixed `{raw, normalized, ids}` cases.

    `normalized` is the string with only `normalize_text` applied, **without `strip()`** — strip
    is applied by upstream's pipeline stage (`inference_runtime._synthesize`) and is not part of
    the tokenizer's contract; applying it would leave the leading-space case (i) nothing to test.
    `ids` is the plain list **before the BOS is prepended** (prepending is the contract of
    {@link build_batch_cases}).
    """
    vocab = set(pair.raw.get_vocab())
    if IN_VOCAB_EMOJI not in vocab:
        raise SystemExit(
            f"{IN_VOCAB_EMOJI!r} is not in the vocabulary — case b would test the same as case c"
        )
    if OUT_OF_VOCAB_EMOJI in vocab:
        raise SystemExit(
            f"{OUT_OF_VOCAB_EMOJI!r} is in the vocabulary — cases c / d do not take byte_fallback"
        )

    cases: list[dict[str, Any]] = []
    for name, why, text in ENCODE_CASES:
        normalized = normalize_text(text)
        cases.append(
            {
                "name": name,
                "why": why,
                "raw": text,
                "normalized": normalized,
                "ids": pair.encode(normalized, name),
            }
        )
    by_name = {case["name"]: case for case in cases}
    # Check that byte_fallback **really** fires (not tautological — the expected byte list is
    # computed independently from UTF-8, without looking at the vocabulary).
    want = byte_fallback_ids(OUT_OF_VOCAB_EMOJI)
    got = by_name["emoji-byte-fallback"]["ids"]
    if got != want:
        raise SystemExit(f"case c did not go through byte_fallback: {got} != {want}")
    return cases


def build_batch_cases(
    pair: TokenizerPair, normalize_text: Any, model_config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The **final form** of upstream's `batch_encode` (BOS prepended + right padding + mask).

    MUST: **call** upstream's `PretrainedTextTokenizer.batch_encode` instead of copying its steps —
    the truncation order (cut the body at `max_length-1`, then add the BOS) is the point a copy
    most easily gets wrong, and a copied formula keeps passing silently after upstream changes.
    """
    specs = [
        *BATCH_CASES,
        (
            "batch-truncated",
            "long enough that truncation really fires (body > max_length-1)",
            TRUNCATING_PASSAGE,
            TEXT_MAX_LENGTH_KEY,
        ),
    ]
    cases: list[dict[str, Any]] = []
    truncated = 0
    for name, why, text, length_key in specs:
        max_length = int(model_config[length_key])
        normalized = normalize_text(text).strip()
        ids, mask = pair.wrapper.batch_encode([normalized], max_length=max_length)
        body = pair.encode(normalized, name)
        used = int(mask[0].sum())
        if used != min(len(body) + 1, max_length):
            raise SystemExit(f"{name}: the mask's valid length {used} differs from BOS + body")
        truncated += int(len(body) + 1 > max_length)
        cases.append(
            {
                "name": name,
                "why": why,
                "raw": text,
                "normalized": normalized,
                "maxLength": max_length,
                "idsPadded": [int(value) for value in ids[0].tolist()],
                "mask": [bool(value) for value in mask[0].tolist()],
            }
        )
    if truncated == 0:
        raise SystemExit("no case makes truncation fire (the truncation order would not be pinned)")
    return cases


def _packed_upstream_ids(pair: TokenizerPair, body: str, max_length: int) -> list[int]:
    """The **packed** id list: the final form of upstream's `batch_encode` without the padding.

    The mask is always a contiguous prefix from the start, so its sum is the valid length (the
    same shape as the list the host builds in the static mode —
    `packages/models/src/irodori/host/pack.ts`).
    """
    ids, mask = pair.wrapper.batch_encode([body], max_length=max_length)
    used = int(mask[0].sum())
    return [int(value) for value in ids[0, :used].tolist()]


def build_caption_cases(
    pair: TokenizerPair, normalize_text: Any, model_config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The packed id list of the **caption path** (strip only) and the same list of the **text
    path** (normalize).

    Both are taken by **calling** upstream's entry point (`PretrainedTextTokenizer.batch_encode`),
    and only the prefix without the padding is kept (the host on the TS side builds this packed
    list — `host/pack.ts`). `normalizedIdsPacked` is the list "if `normalize_text` were applied to
    the caption too", and the TS-side check goes as far as these two lists **differing** (with
    only matching cases, a mix-up of the caption preprocessing would keep the check green).

    MUST: cases sensitive to normalization (the two lists differ) make up most of
    {@link CAPTION_CASES}. The only insensitive one is the official Voice Design sentence, and
    that fact is written out to `normalizeSensitive` too (the caption of the existing golden is on
    that side = fixing the preprocessing does not move the golden).
    """
    max_length = int(model_config[CAPTION_MAX_LENGTH_KEY])
    cases: list[dict[str, Any]] = []
    sensitive = 0
    for name, why, text in CAPTION_CASES:
        stripped = text.strip()
        if not stripped:
            raise SystemExit(f"{name}: the caption is empty after strip")
        normalized = normalize_text(text).strip()
        stripped_ids = _packed_upstream_ids(pair, stripped, max_length)
        normalized_ids = _packed_upstream_ids(pair, normalized, max_length)
        differs = stripped_ids != normalized_ids
        sensitive += int(differs)
        cases.append(
            {
                "name": name,
                "why": why,
                "raw": text,
                "stripped": stripped,
                "normalized": normalized,
                "maxLength": max_length,
                "idsPacked": stripped_ids,
                "normalizedIdsPacked": normalized_ids,
                "normalizeSensitive": differs,
            }
        )
    if sensitive < len(CAPTION_CASES) - 1:
        raise SystemExit(
            f"only {sensitive} caption cases are sensitive to normalization"
            " (a mix-up of the caption preprocessing would keep the check green)"
        )
    return cases


def build_normalize_cases(normalize_text: Any) -> list[dict[str, Any]]:
    """The `{raw, normalized}` golden of `normalize_text` (fires every replacement rule at least
    once)."""
    return [
        {"name": name, "why": why, "raw": text, "normalized": normalize_text(text)}
        for name, why, text in NORMALIZE_CASES
    ]


def check_rule_coverage(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check that every upstream replacement rule **fires** somewhere in the golden.

    MUST: not tautological — beyond "the rule's spelling appears in a case's raw text", it checks
    that the rule **really changes the output** (the string differs before and after it fires).
    When upstream adds a rule this fails, and the hole in the golden shows.
    """
    from irodori_tts.text_normalization import REGEX_REPLACE_MAP, SIMPLE_REPLACE_MAP

    raws = [case["raw"] for case in cases]
    fired: dict[str, int] = {}
    missing: list[str] = []
    for old, new in SIMPLE_REPLACE_MAP.items():
        hits = sum(1 for raw in raws if raw.replace(old, new) != raw)
        fired[f"simple:{old}"] = hits
        if hits == 0:
            missing.append(f"SIMPLE_REPLACE_MAP[{old!r}]")
    for pattern, replacement in REGEX_REPLACE_MAP.items():
        hits = sum(1 for raw in raws if pattern.sub(replacement, raw) != raw)
        fired[f"regex:{pattern.pattern}"] = hits
        if hits == 0:
            missing.append(f"REGEX_REPLACE_MAP[{pattern.pattern!r}]")
    if missing:
        raise SystemExit(f"the golden does not fire these rules: {missing}")

    stripped = sum(1 for case in cases if case["why"].startswith("bracket stripping"))
    changed_by_nfkc = sum(
        1 for case in cases if unicodedata.normalize("NFKC", case["raw"]) != case["raw"]
    )
    dots = sum(1 for case in cases if ".." in case["raw"])
    if stripped < 3 or changed_by_nfkc < 3 or dots < 2:
        raise SystemExit(
            f"the golden's coverage is thin (brackets {stripped} / NFKC {changed_by_nfkc} /"
            f" dots {dots})"
        )
    return {"rules": len(fired), "brackets": stripped, "nfkc": changed_by_nfkc, "dots": dots}


#: Every code point except the surrogates (which a Python str cannot hold).
ALL_CODEPOINTS = [cp for cp in range(0x110000) if not (0xD800 <= cp <= 0xDFFF)]


def nfkc_diff_table() -> dict[str, str]:
    """The map of single code points on which NFKC is **not the identity**,
    `{cp (decimal string): normalized}`.

    The TS side does not implement NFKC itself but defers to `String.prototype.normalize("NFKC")`
    (the source of truth of `normalize_text` is Python's `unicodedata`, so the moment the two
    Unicode versions diverge the normalization result changes **with no exception or warning**).
    This table is the material for that comparison: the TS-side test runs every entry through its
    own `normalize` and checks that they agree.

    NOTE: a single-cp table cannot catch **differences spanning several code points** (canonical
    ordering, composition). Those are covered by the combining-character cases of
    `golden.normalize.json` (か + U+3099 and the like).
    """
    return {
        str(cp): normalized
        for cp in ALL_CODEPOINTS
        if (normalized := unicodedata.normalize("NFKC", chr(cp))) != chr(cp)
    }


def lattice_texts(
    encode_cases: Sequence[Mapping[str, Any]],
    batch_cases: Sequence[Mapping[str, Any]],
    caption_cases: Sequence[Mapping[str, Any]],
    split_cases: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    """The strings the Viterbi lattice really sees (the golden's bodies with ▁ replaced).

    The caption cases contribute **both the strip and the normalize versions** — the TS-side
    check runs the same raw text through two paths and compares the id lists, so with only one
    of them the vocabulary subset would come up short.

    Splitting out the added tokens cuts the text into pieces, but **a substring of a piece is a
    substring of the original string**, so passing the whole text before the split is enough to
    extract the vocabulary (it only yields a superset; the lattice does not change).

    The split cases solve the same match set with replaced scores, so their texts join the same
    extraction.
    """
    texts = [
        case["normalized"] for case in (*encode_cases, *batch_cases, *caption_cases, *split_cases)
    ]
    texts.extend(case["stripped"] for case in caption_cases)
    return [text.replace(" ", METASPACE) for text in texts]


def vocab_subset(
    vocab: Sequence[Sequence[Any]], texts: Sequence[str], added: Mapping[str, int]
) -> list[list[Any]]:
    """The **subset** of the vocabulary that goes into the fixture, `[token, id, score]` (in
    ascending id).

    It holds ① every token that matches a substring of any of the texts (= everything the
    lattice can look up), ② the 256 byte tokens (the TS side tests the consecutive byte_fallback
    ids themselves), ③ the added tokens. Split off so that the 102,400-entry vocabulary is not
    committed whole; {@link check_subset_covers_lattice} checks **in the opposite direction** that
    the lattice is the same.
    """
    entry = {token: (index, score) for index, (token, score) in enumerate(vocab)}
    width = max(len(token) for token, _score in vocab)
    picked: set[str] = set()
    for text in texts:
        for start in range(len(text)):
            for stop in range(start + 1, min(start + width, len(text)) + 1):
                candidate = text[start:stop]
                if candidate in entry:
                    picked.add(candidate)
    picked.update(token for token in entry if len(token) == 6 and token.startswith("<0x"))
    picked.update(added)
    return sorted(
        ([token, entry[token][0], entry[token][1]] for token in picked), key=lambda row: row[1]
    )


def check_subset_covers_lattice(
    vocab: Sequence[Sequence[Any]], texts: Sequence[str], subset: Sequence[Sequence[Any]]
) -> int:
    """Check that Viterbi gives the same result with the subset as with the full vocabulary.

    The lattice only looks up "vocabulary entries that match a substring of the piece", so **the
    same match set means the same lattice**. This scans the **full** vocabulary and counts "tokens
    that are substrings of the texts but missing from the subset" — the opposite direction from
    the extraction (substring → vocabulary lookup), so it is not a tautological test that would
    copy a miss of the extraction.
    """
    present = {row[0] for row in subset}
    missing = [
        token
        for token, _score in vocab
        if token not in present and any(token in text for text in texts)
    ]
    if missing:
        raise SystemExit(
            f"the subset does not cover the lattice's match set: {missing[:8]}"
            f" ({len(missing)} in total)"
        )
    return sum(1 for token, _score in vocab if any(token in text for text in texts))


def build_parity_fixture(
    asset: Mapping[str, Any],
    raw_vocab: Sequence[Sequence[Any]],
    encode_cases: Sequence[Mapping[str, Any]],
    batch_cases: Sequence[Mapping[str, Any]],
    caption_cases: Sequence[Mapping[str, Any]],
    normalize_cases: Sequence[Mapping[str, Any]],
    diff: Mapping[str, str],
    split_cases: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """The fixture of the TS-side parity check (**tracked in git**, runs without the real assets).

    MUST: `minScore` / `maxTokenLength` come from the **full** vocabulary. The score of unknown
    nodes and the search width of the prefix match are determined by the whole, so deriving them
    from the subset gives a different segmentation (the same trap as anima's T5).

    When the asset carries the rare-token split table, the fixture carries it too, cut down the
    same way: `disabledIds` only within the subset (an id outside it never enters a case's
    lattice), `forcedSplits` whole, and `minScore` of the **replaced-score full** vocabulary
    ({@link split_min_score}), with the split golden under `encode.split`.
    """
    texts = lattice_texts(encode_cases, batch_cases, caption_cases, split_cases)
    subset = vocab_subset(raw_vocab, texts, dict(asset["addedTokens"]))
    matched = check_subset_covers_lattice(raw_vocab, texts, subset)
    split_fixture: dict[str, Any] = {}
    table = asset.get(RARE_TOKEN_SPLIT_KEY)
    if table is not None:
        in_subset = {row[1] for row in subset}
        split_fixture = {
            RARE_TOKEN_SPLIT_KEY: {
                "disabledScore": table["disabledScore"],
                "disabledIds": [index for index in table["disabledIds"] if index in in_subset],
                "forcedSplits": table["forcedSplits"],
                "minScore": split_min_score(raw_vocab, table),
                "disabledTotal": len(table["disabledIds"]),
            }
        }
    return {
        "_doc": [
            "Parity fixture of the Irodori-TTS v4 text layer",
            "(generated by tools/export-recipes/irodori/tokenizer_ref.py). The source of truth is",
            "upstream normalize_text and tokenizers / transformers; the id lists were measured",
            "there. The vocabulary is the **subset** every case needs (so as not to commit 102,400",
            "entries), but minScore / maxTokenLength come from the **full** vocabulary. The NFKC",
            "diff table is complete (the moment TS's normalize('NFKC') and the Unicode version",
            "drift apart, normalization changes silently). encode.caption has the id lists of the",
            "caption path (upstream: strip only) and the text path (normalize) for one raw text.",
            *(
                [
                    "asset.rareTokenSplit / encode.split: upstream's rare-token split (ADR 0149);",
                    "disabledIds is cut to the subset; minScore is the replaced-score minimum.",
                ]
                if split_fixture
                else []
            ),
        ],
        "source": {"repo": "Aratako/Irodori-TTS", "reference": "tokenizers + transformers"},
        "asset": {
            "vocab": subset,
            "minScore": min(float(score) for _token, score in raw_vocab),
            "maxTokenLength": max(len(token) for token, _score in raw_vocab),
            "addedTokens": asset["addedTokens"],
            "bosId": asset["bosId"],
            "padId": asset["padId"],
            "unkId": asset["unkId"],
            "byteBaseId": asset["byteBaseId"],
            **split_fixture,
        },
        "encode": {
            "cases": list(encode_cases),
            "batch": list(batch_cases),
            "caption": list(caption_cases),
            **({"split": list(split_cases)} if split_fixture else {}),
        },
        "normalize": {"cases": list(normalize_cases)},
        "nfkcDiff": dict(diff),
        "stats": {"vocabTotal": len(raw_vocab), "latticeMatches": matched},
    }


def emit(model_dir: Path, source_dir: Path, out_dir: Path, fixture_path: Path) -> dict[str, Any]:
    """Write the asset, the goldens and the fixture and return a summary (nothing is written when
    a check fails)."""
    if not (source_dir / "irodori_tts" / "text_normalization.py").is_file():
        raise SystemExit(
            f"model implementation not found: {source_dir}"
            " (pass the checkout of `git clone https://github.com/Aratako/Irodori-TTS`"
            " as --source-dir)"
        )
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from irodori_tts.text_normalization import normalize_text

    text_config, model_config = read_configs(model_dir)
    raw = _tokenizer_json(model_dir)
    shape = check_upstream_shape(raw)

    pair = TokenizerPair(model_dir)
    split = rare_token_split_for(pair, raw, model_config)
    encode_cases = build_encode_cases(pair, normalize_text)
    batch_cases = build_batch_cases(pair, normalize_text, model_config)
    caption_cases = build_caption_cases(pair, normalize_text, model_config)
    split_cases: list[dict[str, Any]] = []
    split_summary: dict[str, Any] | None = None
    if split is not None:
        split_cases = build_split_cases(pair, split, normalize_text, model_config)
        split_summary = {
            "disabledScore": split.table["disabledScore"],
            "disabled": len(split.table["disabledIds"]),
            "forced": len(split.table["forcedSplits"]),
            "coverage": check_split_coverage(
                split_cases,
                split.table,
                raw["model"]["vocab"],
                split.force_splits,
                split.threshold,
                int(model_config[TEXT_MAX_LENGTH_KEY]),
            ),
            "reproduced": check_table_reproduces_splitter(
                raw,
                split.table,
                split.splitter.encode,
                reproduction_texts(raw["model"]["vocab"], split.table, split_cases),
            ),
        }
    asset = build_asset(raw, text_config, None if split is None else split.table)
    normalize_cases = build_normalize_cases(normalize_text)
    coverage = check_rule_coverage(normalize_cases)
    diff = nfkc_diff_table()
    fixture = build_parity_fixture(
        asset,
        raw["model"]["vocab"],
        encode_cases,
        batch_cases,
        caption_cases,
        normalize_cases,
        diff,
        split_cases,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for filename, payload, indent in (
        (ASSET_FILE, asset, None),
        (
            ENCODE_GOLDEN_FILE,
            {
                "cases": encode_cases,
                "batch": batch_cases,
                "caption": caption_cases,
                **({"split": split_cases} if split is not None else {}),
            },
            1,
        ),
        (NORMALIZE_GOLDEN_FILE, {"cases": normalize_cases}, 1),
        (NFKC_DIFF_FILE, diff, 1),
    ):
        path = out_dir / filename
        text = json.dumps(payload, ensure_ascii=False, indent=indent)
        path.write_text(text if indent is None else text + "\n", encoding="utf-8")
        written[filename] = path.stat().st_size
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return {
        "dir": str(out_dir),
        "bytes": written,
        "fixture": {
            "path": str(fixture_path),
            "bytes": fixture_path.stat().st_size,
            "vocabSubset": len(fixture["asset"]["vocab"]),
            "latticeMatches": fixture["stats"]["latticeMatches"],
        },
        "upstream": shape,
        "asset": {
            "bosId": asset["bosId"],
            "padId": asset["padId"],
            "unkId": asset["unkId"],
            "byteBaseId": asset["byteBaseId"],
            "addedTokens": len(asset["addedTokens"]),
            "vocab": len(asset["scores"]),
        },
        "encodeCases": {case["name"]: len(case["ids"]) for case in encode_cases},
        "batchCases": {
            case["name"]: {"maxLength": case["maxLength"], "used": sum(case["mask"])}
            for case in batch_cases
        },
        "captionCases": {
            case["name"]: {
                "tokens": len(case["idsPacked"]),
                "normalizeSensitive": case["normalizeSensitive"],
            }
            for case in caption_cases
        },
        "rareTokenSplit": split_summary,
        "splitCases": {
            case["name"]: {"tokens": len(case["ids"]), "tokensSplit": len(case["idsSplit"])}
            for case in split_cases
        },
        "normalizeCases": len(normalize_cases),
        "ruleCoverage": coverage,
        "nfkcDiffEntries": len(diff),
    }


def default_out_dir(model_dir: Path) -> Path:
    """The default location (`outputs/series/irodori-<real weight directory name>/tokenizer/`)."""
    return SERIES_ROOT / f"irodori-{model_dir.name}" / "tokenizer"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fixture-out", type=Path, default=DEFAULT_FIXTURE_PATH)
    args = parser.parse_args(argv)
    out_dir = default_out_dir(args.model_dir) if args.out is None else args.out
    summary = emit(args.model_dir, args.source_dir, out_dir, args.fixture_out)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
