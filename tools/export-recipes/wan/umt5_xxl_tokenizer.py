"""Bakes the umT5 tokenizer asset that `karume-umt5-xxl` distributes on its own from the pin of the
original `google/umt5-xxl`, and writes the TS-side fixture (asset format `karume-umt5-tokenizer/1`).

    uv run --group wan --inexact python -m wan.umt5_xxl_tokenizer [--fetch]

Two outputs (**always both in one run** — unless they are built from the same tables, the runtime
asset and the fixture go stale separately. The same practice as `wan.umt5_tokenizer`):

  ① Runtime asset `outputs/series/umt5-xxl-tokenizer/tokenizer.json` (untracked, about 8 MB).
       The asset `tokenizer` of model `xxl` in the distribution `karume-umt5-xxl`
       (`wan.umt5_distribution`).
  ② Fixture `packages/models/tests/fixtures/umt5/tokenizer-cases.json` (tracked).

After generating, run `deno fmt packages/models/tests/fixtures/umt5/` (the committed form is what
the formatter produces). With `--fetch`, the pinned tokenizer files are downloaded when they are not
in the HF cache (without it, a missing file fails).

## Asset (`karume-umt5-tokenizer/1`)

**Exactly the T5 table part** of Wan's asset (`karume-wan-umt5-tokenizer/1` —
`wan.umt5_tokenizer.build_asset`); it does not hold Wan's own preprocessing tables (`promptClean`).
The T5 table fields are baked by the same single function as Wan's
(`wan.umt5_tokenizer.t5_table_fields` — the field names and meanings are what TS's `parseT5Tables`
reads):

    format      "karume-umt5-tokenizer/1"
    source      {"repo", "revision", "files": {file name: sha256}}
                (the row of `wan.sources.UMT5_SOURCES`)
    versions    {"tokenizers", "transformers"} (the versions that decide the id sequence)
    maxLength   512 (maximum length of the id sequence including `</s>` — same value and meaning
                as Wan)
    unkId / eosId / addedTokens / space / vocabText / scores (the same fields as Wan's asset)

## Source of truth for the tokenizer (measured on 2026-10-15)

As for Wan, the source of truth for the id sequence is `AutoTokenizer` of the pinned transformers
5.14.1, and generation always checks that it **matches the `tokenizer.json` path**
(`tokenizers.Tokenizer.from_file`) **for every accepted input** (sharing
{@link wan.umt5_tokenizer.tokenized_expectation}). The source of truth is read from a location
holding only the two sha256-verified files `tokenizer.json` and `tokenizer_config.json`
(`wan.sources.pinned_umt5_tokenizer`).

- The original `tokenizer.json` and the `tokenizer/tokenizer.json` of Wan2.1 T2V 1.3B Diffusers
  differ in two points only: the old spelling `add_prefix_space: true` of Metaspace (pre_tokenizer
  and decoder) (tokenizers reads it as `prepend_scheme: always`, `split: true` — the configuration
  after reading is the same as Wan's), and the decimal notation of the scores of 65,654 of the
  256,300 vocabulary entries (up to 3.6e-15 apart in f64 — the original writes the f32 values with
  all f64 digits, Wan uses the short notation). The token order, the 304 added tokens (ids
  included), the normalizer (`Replace(" {2,}", " ")`) and the post_processor are identical.
- The backend that transformers 5.14.1 rebuilds has the same shape as Wan's (no normalizer,
  `WhitespaceSplit` → `Metaspace`, Unigram with unk_id 2 — what
  `wan.umt5_tokenizer.check_upstream_shape` checks); the only difference is the scores above. TS's
  T5 layer (`parseT5Tables` / `T5UnigramTokenizer`, no `normalize`) reproduces it as is — no
  normalization is needed.
- The backend read from the two-file location is identical to the backend read from all the
  tokenizer files of the repository (including `special_tokens_map.json` and `spiece.model`).
  Reading from the repository root makes transformers warn about "an incorrect mistral regex", but
  that is a false positive caused by `config.json` carrying a transformers version below 5 and no
  `model_type`, and no correction is applied (the backend is identical). The two-file location
  does not trigger it.
- Id sequences agree with Wan's tokenizer: for all 50 accepted cases of this fixture, the 218
  accepted cases of Wan's parity fixture (the strings after preprocessing) and the 4,563 accepted
  strings of the sweep ({@link SWEEP_CASES} strings), both the transformers 5.14.1 path and the
  `tokenizer.json` path gave the same id sequences as Wan's (the score notation difference did not
  show up in the id sequences). `wan/tests/test_umt5_xxl_tokenizer.py` pins the accepted cases of
  the fixture on a machine that has both tokenizers locally.

## Rejections added by the absence of preprocessing (`whitespace`)

Wan hands the tokenizer only strings that passed preprocessing (`prompt_clean` — whitespace
collapsing and strip), so the two paths diverged in only three shapes: out-of-vocabulary, added
tokens inside the text and `▁` right after whitespace (the docstring of `wan.umt5_tokenizer`). The
standalone tokenizer passes its input through unchanged, so the paths diverge in two more shapes
(measured):

- **Whitespace other than U+0020** (24 of the 25 characters of `WhitespaceSplit`'s whitespace set,
  all but U+0020 — tab, newline, NBSP, ideographic space and so on; the set the asset's `space`
  holds) — transformers 5 drops them as separators, `tokenizer.json` keeps them as characters
  (`a\\tb` gives `[289, 748, 1]` and `[289, 3, 485, 1]`).
- **A trailing U+0020** (including whitespace-only input) — transformers 5 drops it,
  `tokenizer.json` keeps a `▁` (273) (`a ` gives `[289, 1]` and `[289, 273, 1]`).

Leading U+0020 and runs of U+0020 between words do not diverge (the normalizer of `tokenizer.json`
collapses them). In line with "the same id sequence as the reference of any version" of ADR 0119
decision 1, these two are rejected too (reason `whitespace`). Trimming or collapsing is not added,
because it would be a second preprocessing step separate from Wan's.

Rejection order (TS decides in the same order): added token inside the text `added-token` → `▁`
right after whitespace `metaspace` → `whitespace` → out-of-vocabulary `oov` / over the upper limit
`too-long` (whichever is hit first when accumulating from the start) → under the lower limit
`too-short` (fewer than 2 tokens — the empty string).

## Fixture schema (`tokenizer-cases.json`)

    _doc            description of the generator and how to read the file (array of strings)
    assetFormat     "karume-umt5-tokenizer/1" (the asset format the expectations of these cases
                    assume)
    source          the same as the asset's `source` (repo, revision, sha256 of the files)
    versions        the same as the asset's `versions` (the transformers version = the version of
                    the source of truth for id sequences)
    maxLength       512 / minTokens 2
    t5              the same shape as `t5` of Wan's parity fixture — addedTokens, unkId, eosId,
                    minScore and maxTokenLength (values of the **whole vocabulary**), space, vocab
                    (the subset `[token, id, score]` of the vocabulary that are substrings of the
                    pieces of all cases — the lattice is the same as with the full vocabulary)
    cases[]         id, why (purpose), text (the input passed as is), and one of:
                    accepted — `ids` (the i32 id sequence, ending with `</s>` = 1) and
                    `tokenizerJsonIds` (the same sequence)
                    rejected — `reject` (one of the 6 reasons above). Only `whitespace` also carries
                    the record of the divergence `split: {transformers, tokenizerJson}` (the id
                    sequences of the two paths)

MUST: import transformers / tokenizers / huggingface_hub inside functions (the `wan` group is not in
the default sync — `tests/test_optional_group_imports.py`). Do not import torch either.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from _shared.paths import REPO_ROOT, SERIES_ROOT
from wan import prompt_clean as pc
from wan import umt5_tokenizer as ut
from wan.sources import UMT5_SOURCES, pinned_umt5_tokenizer

#: Location in the series (`<name>-tokenizer/tokenizer.json` — docs/assets-layout.md).
SERIES_NAME = "umt5-xxl-tokenizer"
ASSET_FILE = "tokenizer.json"

#: Version of the asset format (so TS does not silently read a version it does not know).
ASSET_FORMAT = "karume-umt5-tokenizer/1"

#: The default model (a key of `wan.sources.UMT5_SOURCES` — the same spelling as the distribution's
#: model name `wan.umt5_distribution.UMT5_DEFAULT_MODEL`. That module imports this one, so it cannot
#: be imported from here — `wan/tests/test_umt5_xxl_tokenizer.py` checks that they agree).
DEFAULT_MODEL = "xxl"

#: Where the fixture goes (consumed by the umT5 tokenizer tests under `packages/models/tests/`).
FIXTURE_DIR = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "umt5"
CASES_FIXTURE = "tokenizer-cases.json"

#: The versions recorded in the asset's `versions` (those that decide the id sequence — there is no
#: preprocessing, so ftfy and the like are not recorded).
ASSET_VERSIONS: tuple[str, ...] = ("tokenizers", "transformers")

#: U+0020 (the only accepted whitespace).
SPACE = " "

#: A longer prompt (one line, around 100 tokens — a length in the middle of the accepted range).
LONG_PROMPT = (
    "A wide shot of a quiet harbor at dawn: fishing boats rest on calm water, gulls circle above"
    " the masts, a lighthouse blinks on the rocky point, and soft orange light spreads across the"
    " clouds while a small dog runs along the wooden pier toward an old man mending his nets."
)

#: Boundary and multilingual cases (id, input, purpose). Always write invisible characters as
#: \\uXXXX.
CASES: tuple[tuple[str, str, str], ...] = (
    ("english", "A cat walks on the grass, realistic style.", "英語"),
    ("japanese", "夕焼けの海と猫", "日本語"),
    ("chinese", "一只猫在草地上散步", "中国語"),
    ("korean", "고양이가 잔디 위를 걷는다", "韓国語"),
    ("accented-latin", "café naïve résumé Zürich São Paulo", "アクセント付きの欧文"),
    ("emoji", "a cat \U0001f431 and a dog \U0001f436", "絵文字"),
    ("digits-punctuation", "Price: $1,299.99 (save 15%) - call 555-0100! #42 @home", "数字と記号"),
    ("long-prompt", LONG_PROMPT, "長めのプロンプト"),
    ("leading-spaces", "  a cat", "先頭の U+0020 は 2 経路で割れない（受ける）"),
    ("inner-spaces", "a   cat  sits", "語の間の連続した U+0020 は 2 経路で割れない（受ける）"),
    ("decomposed", "café", "正規化はしない（NFD のまま — `café` の合成済みとは別の id 列）"),
    ("fullwidth", "ＡＢＣ１２３", "全角英数も正規化しない"),
    ("user-defined", "[web] page", "user-defined のピース `[web]` は普通の語彙"),
    ("metaspace-inner", "a▁b cat", "空白の直後でない `▁` は 2 経路で一致する"),
    ("min-two", "cat", "受理の下限ちょうど（`▁cat` + `</s>` の 2 トークン）"),
    ("max-512", "cat " * 510 + "cat", "受理の上限ちょうど（511 + `</s>` = 512）"),
    ("over-513", "cat " * 511 + "cat", "上限 + 1 は拒む（上流は黙って切り詰める）"),
    ("empty", "", "空文字は `</s>` だけの長さ 1 で拒む"),
    (
        "oov",
        "\U0002000b\U0002123d",
        "語彙外の文字は拒む（transformers 5 は id 2・tokenizer.json は id 3）",
    ),
    ("oov-inner", "a cat \U0002000b here", "語彙外が途中にあっても拒む"),
    ("added-eos", "a </s> b", "本文中の追加語彙は拒む（2 経路で id 列が割れる）"),
    ("added-extra", "<extra_id_0> cat", "追加語彙 `<extra_id_*>` も拒む"),
    ("metaspace", "a ▁b", "空白の直後の `▁` は拒む（2 経路で断片化が割れる）"),
    ("tab", "a\tcat", "タブは拒む（U+0020 以外の空白）"),
    ("newline", "a cat\nsits", "改行は拒む（U+0020 以外の空白）"),
    ("crlf", "a cat\r\nsits", "CR LF も拒む"),
    ("nbsp", "a cat", "NBSP は拒む（U+0020 以外の空白）"),
    ("ideographic-space", "猫　犬", "全角空白は拒む（U+0020 以外の空白）"),
    ("trailing-space", "a cat ", "末尾の U+0020 は拒む（tokenizer.json は `▁` を残す）"),
    ("only-space", "   ", "空白だけの入力も末尾の U+0020 として拒む"),
)

#: The intended outcomes (a case that misses its purpose is not baked — so that we notice the day
#: upstream behaviour changes).
EXPECTED_OUTCOMES: Mapping[str, str] = {
    "english": "accepted",
    "japanese": "accepted",
    "chinese": "accepted",
    "korean": "accepted",
    "accented-latin": "accepted",
    "emoji": "accepted",
    "digits-punctuation": "accepted",
    "long-prompt": "accepted",
    "leading-spaces": "accepted",
    "inner-spaces": "accepted",
    "decomposed": "accepted",
    "fullwidth": "accepted",
    "user-defined": "accepted",
    "metaspace-inner": "accepted",
    "min-two": "accepted",
    "max-512": "accepted",
    "over-513": "too-long",
    "empty": "too-short",
    "oov": "oov",
    "oov-inner": "oov",
    "added-eos": "added-token",
    "added-extra": "added-token",
    "metaspace": "metaspace",
    "tab": "whitespace",
    "newline": "whitespace",
    "crlf": "whitespace",
    "nbsp": "whitespace",
    "ideographic-space": "whitespace",
    "trailing-space": "whitespace",
    "only-space": "whitespace",
}

#: Random cases (the ones written into the fixture — fixed seed).
RANDOM_CASES = 40
RANDOM_SEED = 7
RANDOM_MAX_LENGTH = 40
#: Weight of the U+0020 mixed into the random pool (creates word breaks — leading, repeated and
#: trailing U+0020 also occur).
RANDOM_SPACE_WEIGHT = 2000

#: Sweep of the agreement of the two paths (not written into the fixture — a check that runs on
#: every generation).
SWEEP_CASES = 5000
SWEEP_SEED = 11


def whitespace_split(space: pc.RangeSet, text: str) -> bool:
    """Whether the text has a whitespace shape on which the two paths diverge without preprocessing
    (whitespace other than U+0020, a trailing U+0020)."""
    return text.endswith(SPACE) or any(ch != SPACE and ord(ch) in space for ch in text)


def expectation(
    truth: ut.Truth, added: Sequence[str], space: pc.RangeSet, text: str
) -> dict[str, Any]:
    """The expectation for an input passed as is (an id sequence or a rejection reason).

    The rejection order is the same as in TS: added token → `▁` right after whitespace (up to here
    the same single function as Wan — {@link ut.version_split}) → whitespace shape →
    out-of-vocabulary / over the upper limit → under the lower limit (the same single function as
    Wan — {@link ut.tokenized_expectation}, which also holds the check on how the two paths
    diverge).
    """
    reason = ut.version_split(added, text)
    if reason is not None:
        return {"reject": reason}
    if whitespace_split(space, text):
        return {
            "reject": "whitespace",
            "split": {
                "transformers": truth.transformers(text),
                "tokenizerJson": truth.tokenizer_json(text),
            },
        }
    return ut.tokenized_expectation(truth, text)


def outcome(case: Mapping[str, Any]) -> str:
    """The outcome of a case (`accepted` or the rejection reason)."""
    return "accepted" if "ids" in case else case["reject"]


def random_pool(truth: ut.Truth, space: pc.RangeSet) -> list[int]:
    """The random pool — the single-character vocabulary entries that are not whitespace + U+0020
    (weighted)."""
    singles = sorted(
        {
            ord(token)
            for token, _ in truth.backend["model"]["vocab"]
            if len(token) == 1 and ord(token) not in space
        }
    )
    return [*singles, *([ord(SPACE)] * RANDOM_SPACE_WEIGHT)]


def random_texts(pool: Sequence[int], seed: int, count: int) -> Iterator[str]:
    return pc.fuzz_strings(pool, seed, count, RANDOM_MAX_LENGTH)


def build_cases(truth: ut.Truth, *, random_count: int = RANDOM_CASES) -> list[dict[str, Any]]:
    """The expectations of all cases (boundary and multilingual + random). Fails if a boundary case
    misses its intended outcome."""
    added = [entry["content"] for entry in truth.backend["added_tokens"]]
    space = pc.RangeSet(ut.whitespace_split_ranges())
    cases = [
        {"id": case_id, "why": why, "text": text, **expectation(truth, added, space, text)}
        for case_id, text, why in CASES
    ]
    wrong = {
        case["id"]: outcome(case)
        for case in cases
        if outcome(case) != EXPECTED_OUTCOMES[case["id"]]
    }
    if wrong:
        raise ValueError(
            f"boundary cases do not end with their intended outcome (actual outcome): {wrong}"
        )
    if len(next(case for case in cases if case["id"] == "max-512")["ids"]) != ut.MAX_LENGTH:
        raise ValueError("max-512 is not exactly at the upper limit")
    pool = random_pool(truth, space)
    for index, text in enumerate(random_texts(pool, RANDOM_SEED, random_count)):
        cases.append(
            {
                "id": f"random-{index}",
                "why": "乱択（語彙の 1 文字 + U+0020）",
                "text": text,
                **expectation(truth, added, space, text),
            }
        )
    return cases


def sweep(truth: ut.Truth, count: int = SWEEP_CASES) -> dict[str, int]:
    """The counts of the outcomes of `count` random strings ({@link ut.tokenized_expectation} fails
    if the two paths diverge in an unknown shape — the check for the part not written into the
    fixture)."""
    added = [entry["content"] for entry in truth.backend["added_tokens"]]
    space = pc.RangeSet(ut.whitespace_split_ranges())
    counts: dict[str, int] = {}
    for text in random_texts(random_pool(truth, space), SWEEP_SEED, count):
        result = outcome(expectation(truth, added, space, text))
        counts[result] = counts.get(result, 0) + 1
    return dict(sorted(counts.items()))


def build_asset(truth: ut.Truth, model: str, versions: Mapping[str, str]) -> dict[str, Any]:
    """The asset TS only looks up (format {@link ASSET_FORMAT} — the T5 table fields come from the
    same single function as Wan's asset)."""
    row = UMT5_SOURCES[model]
    return {
        "format": ASSET_FORMAT,
        "source": {
            "repo": row.source.repo,
            "revision": row.source.revision,
            "files": dict(sorted(row.tokenizer.items())),
        },
        "versions": {name: versions[name] for name in ASSET_VERSIONS},
        **ut.t5_table_fields(truth),
    }


def asset_versions() -> dict[str, str]:
    import tokenizers
    import transformers

    return {"tokenizers": tokenizers.__version__, "transformers": transformers.__version__}


def build_fixture(
    truth: ut.Truth, asset: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The TS-side fixture (the vocabulary is a subset — the schema is in the module docstring)."""
    scores = asset["scores"]
    tokens = asset["vocabText"].split("\n")
    return {
        "_doc": [
            "umT5-XXL（google/umt5-xxl）のトークナイザの期待（生成:",
            "tools/export-recipes/wan/umt5_xxl_tokenizer.py — schema はその docstring）。",
            "id 列の正本は transformers 5.14.1 の AutoTokenizer。受理したケースは",
            "tokenizer.json の経路の id 列も持つ（2 経路の一致は生成時に確認済み）。",
            "前処理はしない — text をそのまま渡す。語彙は全ケースの断片の部分文字列だけ",
            "（minScore / maxTokenLength は語彙全体の値）。",
        ],
        "assetFormat": asset["format"],
        "source": asset["source"],
        "versions": asset["versions"],
        "maxLength": asset["maxLength"],
        "minTokens": ut.MIN_TOKENS,
        "t5": {
            "addedTokens": asset["addedTokens"],
            "unkId": asset["unkId"],
            "eosId": asset["eosId"],
            # MUST: the unknown-node score and the search width come from the **whole vocabulary**
            # (never derived from the subset).
            "minScore": min(scores),
            "maxTokenLength": max(len(token) for token in tokens),
            "space": asset["space"],
            "vocab": ut.vocab_subset(truth, [case["text"] for case in cases]),
        },
        "cases": list(cases),
    }


def load_pinned_truth(model: str, directory: Path, *, fetch: bool = False) -> ut.Truth:
    """Verifies the pinned tokenizer files, copies them into `directory` and reads the source of
    truth from there."""
    pinned_umt5_tokenizer(model, directory, fetch=fetch)
    return ut.load_truth(directory)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(UMT5_SOURCES))
    parser.add_argument(
        "--fetch", action="store_true", help="download the pinned tokenizer files from HF"
    )
    parser.add_argument("--assets-out", type=Path, default=SERIES_ROOT / SERIES_NAME)
    parser.add_argument("--fixtures-out", type=Path, default=FIXTURE_DIR)
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="umt5-tokenizer-") as scratch:
        truth = load_pinned_truth(args.model, Path(scratch), fetch=args.fetch)
        asset = build_asset(truth, args.model, asset_versions())
        log(f"[asset] vocabulary {len(asset['scores'])} ({time.perf_counter() - started:.1f}s)")
        cases = build_cases(truth)
        log(f"[cases] {len(cases)} cases ({sorted({outcome(case) for case in cases})})")
        swept = sweep(truth)
        log(f"[sweep] {SWEEP_CASES} strings: {swept}")
        fixture = build_fixture(truth, asset, cases)

    # Write only after every check has passed (ADR 0005's fail loudly — a failure midway writes
    # not a single byte).
    written = {
        "asset": ut.write_text(args.assets_out / ASSET_FILE, json.dumps(asset, ensure_ascii=False)),
        "fixture": ut.write_text(
            args.fixtures_out / CASES_FIXTURE, ut.fixture_json(fixture) + "\n"
        ),
    }
    log(
        json.dumps(
            {
                "bytes": written,
                "vocabSubset": len(fixture["t5"]["vocab"]),
                "seconds": round(time.perf_counter() - started, 1),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
