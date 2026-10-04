"""Wan2.1 の umT5 トークナイザと前処理を TS が引くだけの資産へ焼き、パリティ用フィクスチャを作る
（ADR 0119 段 10a — 決定 1・2・4）。

    uv run --group wan --inexact python -m wan.umt5_tokenizer

出力は 3 本（**1 回の実行で必ず全部**出す — 同じ表から作らないと、実行時資産とフィクスチャが
別々に古びて「テストは緑だが実行時だけ別の id 列」になる。anima の `anima/demo.py` と同じ流儀）:

  ① 実行時資産 `outputs/series/wan2.1-umt5-tokenizer/tokenizer.json`（git 追跡外・約 8 MB）
       語彙とスコア（行番号 = id）・追加語彙・空白集合・前処理の表（`promptClean`）。
       配布形への組み込みは段 10d。
  ② パリティ用フィクスチャ `packages/models/tests/fixtures/wan-text/parity.json`（git 管理）
       ケースごとの期待（id 列か拒否の理由）と、その再現に要る語彙の**部分集合**・前処理の表の全体。
  ③ 前処理の掃引フィクスチャ `packages/models/tests/fixtures/wan-text/prompt-clean-sweep.json`
       （git 管理）— 乱択と全コードポイント掃引の期待を digest で持つ（TS が同じ列を再生する）。

生成後は `deno fmt packages/models/tests/fixtures/wan-text/` を掛ける（commit 形は
フォーマッタが正）。

## トークナイザの正本（ADR 0119 決定 1）

id 列の正本は pin した transformers 5.14.1 の `AutoTokenizer` の実挙動（Wan の pin した revision の
`tokenizer/`）。transformers 5 は `tokenizer.json` を読んだ上で T5 の経路を組み直す
（`models/t5/tokenization_t5.py:112-127` — 正規化なし・`WhitespaceSplit` + `Metaspace`・
Unigram の unk_id 2）。資産は組み直した後の backend（`backend_tokenizer.to_str()`）から焼き、
`tokenizer.json` と語彙・スコア・追加語彙が一致することを確かめる。

語彙外の文字は拒む（決定 1）。語彙外の id は transformers 5（id 2 = `<s>`）と `tokenizer.json`
（id 3 = `<unk>`）で割れるので、**受理した入力では 2 経路の id 列が一致する**ことを fixture の
生成時に必ず確かめる（版に依らないことの検査）。逆に 2 経路が割れたケースは語彙外として
記録する（割れ方が「同じ位置で 2 と 3」以外なら落とす）。

2 経路は語彙外のほかに 2 つの形でも割れる（2026-10-03 の実測 — 乱択 60,000 本で割れた全件が
「語彙外・追加語彙・空白の直後の U+2581」のどれかを含み、どれも含まないものは全件一致）:

  * **本文中の追加語彙**（`</s>`・`<extra_id_*>` など 304 個）— transformers 5 は空白で先に割るので
    `a </s> b` が `[289, 1, 748, 1]`、`tokenizer.json` は追加語彙の直前の空白を `▁`（273）として
    残すので `[289, 273, 1, 748, 1]`。
  * **空白の直後の `▁`（U+2581）** — transformers 5 は空白を捨ててから `▁` で割る（`a ▁b` →
    `▁a` `▁b`）、`tokenizer.json` は空白そのものを `▁` に置き換える（→ `▁a` `▁` `▁b`）。

決定 1 の「守るもの」（どの版の参照とも同じ id 列）に合わせ、この 2 つも語彙外と同じく拒む
（ADR 0119 決定 1 は本文中の追加語彙を「上流どおり切り出す」と書くが、調査が 1 文字の掃引しか
見ておらず、版で割れることを見落としていた — 拒む側へ倒した）。

上限 512 は上流の `max_sequence_length`（`wan.pipeline_ref.MAX_SEQUENCE_LENGTH`）で、上流は
黙って切り詰めるが karume は拒む（決定 4）。下限 2（空・空白だけ = `</s>` だけ）も拒む。

MUST: transformers / tokenizers / huggingface_hub は関数の中で import する（`wan` グループは既定の
sync に入らない — `tests/test_optional_group_imports.py`）。torch も import しない（トークナイザ
だけの台本に重い依存を持ち込まない — `wan.pipeline_ref` を import しないのもそのため）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _shared.paths import REPO_ROOT, SERIES_ROOT
from wan import prompt_clean as pc
from wan.prompts import FIXED_PROMPTS
from wan.sources import DEFAULT_MODEL, SOURCES, WAN21_MODELS, WanSourceError

#: 系列の席（`<名前>-tokenizer/tokenizer.json` — docs/assets-layout.md）。
SERIES_NAME = "wan2.1-umt5-tokenizer"
ASSET_FILE = "tokenizer.json"

#: 資産の形式の版（TS 側が知らない版を黙って読まない）。
ASSET_FORMAT = "karume-wan-umt5-tokenizer/1"

#: フィクスチャの置き場（消費者は `packages/models/tests/wan_text_*_test.ts`）。
FIXTURE_DIR = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "wan-text"
PARITY_FIXTURE = "parity.json"
SWEEP_FIXTURE = "prompt-clean-sweep.json"

#: 上流の `max_sequence_length`（`wan.pipeline_ref.MAX_SEQUENCE_LENGTH` と同じ値 — torch を
#: 読まないためにここで持ち、pytest が一致を見る）。
MAX_LENGTH = 512

#: 受理集合の下限（`torch.export` の 0 / 1 特殊化を避ける — anima の `PROMPT_MIN_TOKENS` と同じ）。
MIN_TOKENS = 2

#: pin した transformers の版（id 列の正本）。
TRANSFORMERS_VERSION = "5.14.1"

#: Metaspace の置換文字（U+2581）。
METASPACE = "▁"

#: 受理の固定 4 本の長さ（調査 §1.4・ADR 0118 段 6 の値）。
FIXED_PROMPT_LENGTHS = {"boxing-cats": 28, "ferret": 118, "cat-dog-baking": 50, "negative": 126}

#: 境界のケース（id・入力・狙い）。目に見えない文字は必ず \\uXXXX で書く。
BOUNDARY_CASES: tuple[tuple[str, str, str], ...] = (
    ("min-two", "cat", "受理の下限ちょうど（`▁cat` + `</s>` の 2 トークン）"),
    ("max-512", "cat " * 511, "受理の上限ちょうど（511 + `</s>` = 512）"),
    ("over-513", "cat " * 512, "上限 + 1 は拒む（上流は黙って切り詰める）"),
    ("empty", "", "空文字は `</s>` だけの長さ 1 で拒む"),
    ("only-space", " \n\t　 ", "空白だけ（全角空白も畳まれる）は長さ 1 で拒む"),
    ("oov", "𠀋𡈽", "語彙外の文字は拒む（transformers 5 は id 2・tokenizer.json は id 3）"),
    ("oov-inner", "a cat 𠀋 here", "語彙外が途中にあっても拒む"),
    ("added-eos", "a </s> b", "本文中の追加語彙は拒む（2 経路で id 列が割れる）"),
    ("added-extra", "<extra_id_0> cat", "追加語彙 `<extra_id_*>` も拒む"),
    ("metaspace", "a ▁b", "空白の直後の `▁` は拒む（2 経路で断片化が割れる）"),
    ("metaspace-inner", "a▁b cat", "空白の直後でない `▁` は 2 経路で一致する"),
    ("user-defined", "[web] page", "user-defined のピース `[web]` は普通の語彙（id 5）"),
    ("fullwidth-comma", "色调艳丽，过曝", "公式 negative の全角読点は `,` になる"),
    ("curly", "“quoted” it’s", "曲がった引用符は ASCII へ"),
    ("ligature", "ﬁne ŉ", "合字の分解（`ŉ` は `ʼn` を経て `'n`）"),
    ("ansi", "\x1b[31mred\x1b[0m", "端末エスケープを消す"),
    (
        "ansi-unicode-digit",
        "\x1b[\u0661mred",
        "端末エスケープの `\\d` は Unicode の Nd（ASCII だけではない）",
    ),
    ("line-breaks", "a\r\nb\rc d", "改行の正規化と空白の畳み込み"),
    ("nfc", "é café", "NFC の合成"),
    ("accents", "café naïve résumé Zürich São Paulo", "アクセント付きの欧文は受ける"),
    ("width", "ＡＢＣ　ｶﾀｶﾅ", "全角英字・半角カナは標準形へ"),
    ("controls", "a\x07b﻿c", "制御文字と BOM は消える"),
    ("zero-width", "x​y", "ゼロ幅空白は空白ではない（残る）"),
    ("japanese", "夕焼けの海と猫", "日本語"),
    ("ampersand-spaced", "salt & pepper", "空白を挟んだ `&` は受ける"),
    ("mojibake", "cafÃ©", "mojibake の判定が真なら拒む（上流は推定で修復する）"),
    ("entity-named", "salt &amp; pepper", "HTML の文字参照は拒む"),
    ("entity-bare", "R&D", "entity にならない並びも `&` の直後が文字なら拒む"),
    ("c1", "a\x85b", "C1 制御文字は拒む"),
    ("unassigned", "a͸b", "UCD 16.0.0 で未割り当ての文字は拒む（U+0378 は語彙にある）"),
)

#: 乱択ケース（受理集合の文字だけ — seed 固定）。
RANDOM_CASES = 200
RANDOM_SEED = 5
RANDOM_MAX_LENGTH = 40
#: 乱択の池に混ぜる空白の重み（語の切れ目を作る）。
RANDOM_SPACE_WEIGHT = 2000


def tokenizer_snapshot(model: str = DEFAULT_MODEL) -> Path:
    """pin した revision の `tokenizer/` だけの snapshot（ネットワークに出ない）。"""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    source = SOURCES[model]
    try:
        snapshot = Path(
            snapshot_download(
                source.repo,
                revision=source.revision,
                allow_patterns=["tokenizer/*"],
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError as error:
        raise WanSourceError(
            f"{source.repo}@{source.revision} の tokenizer/ が HF キャッシュに無い — 先に"
            " `uv run --group wan --inexact python -m wan.text_embeds --fetch` で取得する"
        ) from error
    if not (snapshot / "tokenizer" / "tokenizer.json").is_file():
        raise WanSourceError(f"{snapshot}/tokenizer/tokenizer.json が無い")
    return snapshot


@dataclass(frozen=True)
class Truth:
    """2 経路の正本（transformers 5.14.1 の `AutoTokenizer` と `tokenizer.json`）。"""

    #: `tokenizer([text], add_special_tokens=True)` の id 列（切り詰めなし）。
    transformers: Callable[[str], list[int]]
    #: 上流の呼び方（`padding="max_length"`・`truncation=True`）のマスクの長さ。
    masked_length: Callable[[str], int]
    #: `tokenizers.Tokenizer.from_file(tokenizer.json)` の id 列。
    tokenizer_json: Callable[[str], list[int]]
    #: transformers が組み直した後の backend（`to_str()` の JSON）。
    backend: dict[str, Any]
    #: `tokenizer.json` の中身。
    raw: dict[str, Any]
    #: 断片化（`WhitespaceSplit` → `Metaspace`）だけを回す口。
    pre_tokenize: Callable[[str], list[str]]


def load_truth(snapshot: Path) -> Truth:
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ValueError(
            f"transformers {transformers.__version__} — id 列の正本は {TRANSFORMERS_VERSION}"
        )
    path = snapshot / "tokenizer" / "tokenizer.json"
    auto = AutoTokenizer.from_pretrained(snapshot, subfolder="tokenizer")
    plain = Tokenizer.from_file(str(path))
    backend = auto.backend_tokenizer

    def masked_length(text: str) -> int:
        encoded = auto(
            [text],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        return sum(encoded["attention_mask"][0])

    return Truth(
        transformers=lambda text: list(auto([text], add_special_tokens=True)["input_ids"][0]),
        masked_length=masked_length,
        tokenizer_json=lambda text: list(plain.encode(text).ids),
        backend=json.loads(backend.to_str()),
        raw=json.loads(path.read_text(encoding="utf-8")),
        pre_tokenize=lambda text: [
            piece for piece, _ in backend.pre_tokenizer.pre_tokenize_str(text)
        ],
    )


def check_upstream_shape(truth: Truth) -> None:
    """焼いた表が前提にしている backend の構造を検査する（fail loudly）。

    TS は「追加語彙の切り出し → WhitespaceSplit → Metaspace（▁・always・split）→ Unigram →
    末尾 `</s>`」だけを持つ。backend がこの形から 1 欄でも外れたら、TS の id 列は黙って別物になる。
    """
    backend = truth.backend
    if backend["normalizer"] is not None:
        raise ValueError(f"backend に正規化がある: {backend['normalizer']}")
    expected_pre = {
        "type": "Sequence",
        "pretokenizers": [
            {"type": "WhitespaceSplit"},
            {
                "type": "Metaspace",
                "replacement": METASPACE,
                "prepend_scheme": "always",
                "split": True,
            },
        ],
    }
    if backend["pre_tokenizer"] != expected_pre:
        raise ValueError(f"backend の pre_tokenizer が想定と違う: {backend['pre_tokenizer']}")
    model = backend["model"]
    if model["type"] != "Unigram" or model["byte_fallback"] or model["unk_id"] != 2:
        raise ValueError(f"backend のモデルが Unigram・unk_id 2・byte_fallback なしでない: {model}")
    post = backend["post_processor"]
    specials = [entry["SpecialToken"]["id"] for entry in post["single"] if "SpecialToken" in entry]
    if post["type"] != "TemplateProcessing" or specials != ["</s>"]:
        raise ValueError(f"post_processor が `</s>` 1 つだけを足す形でない: {post}")
    if post["single"][-1] != {"SpecialToken": {"id": "</s>", "type_id": 0}}:
        raise ValueError(f"`</s>` が末尾に足されない: {post['single']}")
    # 追加語彙のフラグが既定値でないと「正規化の前に leftmost-longest で切り出す」写経の意味が
    # 変わる。
    for entry in backend["added_tokens"]:
        flags = {key: entry[key] for key in ("lstrip", "rstrip", "single_word", "normalized")}
        if any(flags.values()):
            raise ValueError(f"追加語彙 {entry['content']!r} のフラグが既定値でない: {flags}")
    # 語彙・スコア・追加語彙は tokenizer.json と同一（transformers 5 は構成だけを組み直す）。
    if backend["model"]["vocab"] != truth.raw["model"]["vocab"]:
        raise ValueError("backend の語彙・スコアが tokenizer.json と違う")
    if backend["added_tokens"] != truth.raw["added_tokens"]:
        raise ValueError("backend の追加語彙が tokenizer.json と違う")


def whitespace_split_ranges() -> pc.Ranges:
    """`WhitespaceSplit` が空白と見なす文字（Rust の `char::is_whitespace`）を畳む。

    全コードポイントを `a` + 文字 + `b` に入れて、2 断片に割れるかを正本に聞く。
    """
    from tokenizers import pre_tokenizers

    splitter = pre_tokenizers.WhitespaceSplit()
    return pc.ranges_where(lambda ch: len(splitter.pre_tokenize_str("a" + ch + "b")) > 1)


def build_asset(
    truth: Truth, tables: pc.PromptCleanTables, versions: Mapping[str, str], model: str
) -> dict[str, Any]:
    """TS が引くだけの資産（`parseWanTokenizerAsset` が読む形）。"""
    check_upstream_shape(truth)
    vocab = [(token, float(score)) for token, score in truth.backend["model"]["vocab"]]
    for token, _ in vocab:
        if "\n" in token or token == "":
            raise ValueError(f"語彙に改行を含む / 空のトークン {token!r} — 行区切りが壊れる")
    added = sorted(
        ((entry["content"], entry["id"]) for entry in truth.backend["added_tokens"]),
        key=lambda pair: pair[1],
    )
    eos = dict(added)["</s>"]
    source = SOURCES[model]
    return {
        "format": ASSET_FORMAT,
        "source": {"repo": source.repo, "revision": source.revision, "subfolder": "tokenizer"},
        "versions": dict(sorted(versions.items())),
        "maxLength": MAX_LENGTH,
        "unkId": truth.backend["model"]["unk_id"],
        "eosId": eos,
        "addedTokens": [[content, token_id] for content, token_id in added],
        "space": whitespace_split_ranges(),
        # 行番号 0-origin = id（改行・空のトークンが無いことは上で検査済み）。
        "vocabText": "\n".join(token for token, _ in vocab),
        "scores": [score for _, score in vocab],
        "promptClean": tables.to_json(),
    }


def split_added(text: str, added: Sequence[str]) -> list[tuple[str, bool]]:
    """追加語彙を leftmost-longest で切り出す（TS の `splitAddedTokens` の鏡像）。"""
    out: list[tuple[str, bool]] = []
    buffer = ""
    i = 0
    while i < len(text):
        hit = max((token for token in added if text.startswith(token, i)), key=len, default=None)
        if hit is None:
            buffer += text[i]
            i += 1
            continue
        if buffer:
            out.append((buffer, False))
            buffer = ""
        out.append((hit, True))
        i += len(hit)
    if buffer:
        out.append((buffer, False))
    return out


def pieces_of(truth: Truth, added: Sequence[str], text: str) -> list[str]:
    """追加語彙の外の断片（Unigram に渡る単位）。"""
    return [
        piece
        for chunk, is_added in split_added(text, added)
        if not is_added
        for piece in truth.pre_tokenize(chunk)
    ]


def expectation(truth: Truth, added: Sequence[str], cleaned: str) -> dict[str, Any]:
    """前処理を通った文字列の期待（id 列か、トークナイザ側の拒否の理由）。

    拒否の順は TS と同じ: 追加語彙 → 空白の直後の `▁` → 語彙外 / 上限超え → 下限割れ。

    MUST: 受理するのは 2 経路の id 列が一致したときだけ。追加語彙と空白の直後の `▁` を除いた後に
    割れたら語彙外で、割れ方は「同じ長さ・同じ位置で transformers 5 が 2・tokenizer.json が 3」
    以外にありえない（それ以外は経路の構成が違う = 落とす）。
    """
    if any(is_added for _, is_added in split_added(cleaned, added)):
        return {"reject": "added-token"}
    if " " + METASPACE in cleaned:
        return {"reject": "metaspace"}
    ids = truth.transformers(cleaned)
    plain = truth.tokenizer_json(cleaned)
    if ids != plain:
        if len(ids) != len(plain) or any(
            (a, b) != (2, 3) for a, b in zip(ids, plain, strict=True) if a != b
        ):
            raise ValueError(
                f"2 経路の割れ方が語彙外（2 と 3）でない: {cleaned!r}\n  tf5={ids}\n  json={plain}"
            )
        if len(ids) > MAX_LENGTH:
            raise ValueError(f"語彙外と上限超えが重なるケースは置かない: {cleaned!r}")
        return {"reject": "oov"}
    if len(ids) > MAX_LENGTH:
        return {"reject": "too-long"}
    if len(ids) < MIN_TOKENS:
        return {"reject": "too-short"}
    if truth.masked_length(cleaned) != len(ids):
        raise ValueError(f"上流の呼び方のマスク長が id 列の長さと違う: {cleaned!r}")
    return {"ids": ids, "tokenizerJsonIds": plain}


def build_case(
    truth: Truth,
    compiled: pc.CompiledTables,
    clean_truth: Callable[[str], str],
    case_id: str,
    text: str,
    why: str,
) -> dict[str, Any]:
    """1 ケースの期待。前処理の拒否 → トークナイザの拒否 → 受理の順（TS と同じ）。"""
    outcome = pc.record(compiled, clean_truth, text)
    base = {"id": case_id, "why": why, "text": text}
    if outcome.startswith("R"):
        return {**base, "reject": outcome[1:]}
    cleaned = outcome[1:]
    added = [entry["content"] for entry in truth.backend["added_tokens"]]
    return {**base, "cleaned": cleaned, **expectation(truth, added, cleaned)}


def random_case_pool(truth: Truth, tables: pc.PromptCleanTables) -> list[int]:
    """乱択ケースの池 — 1 文字として引ける語彙のうち、前処理の門と表引きに触れない文字 + 空白。"""
    singles = {ord(token) for token, _ in truth.backend["model"]["vocab"] if len(token) == 1}
    assigned = pc.RangeSet(tables.assigned)
    literals = set(pc.badness_literals())
    control = pc.RangeSet(tables.control_chars)
    space = pc.RangeSet(tables.collapse_space)
    chars = sorted(
        cp
        for cp in singles
        if cp in assigned
        and cp not in literals
        and cp != ord("&")
        and cp not in tables.width
        and cp not in tables.ligatures
        and cp not in control
        and cp not in space
        and not pc.C1_FIRST <= cp <= pc.C1_LAST
    )
    return [*chars, *([0x20] * RANDOM_SPACE_WEIGHT)]


def build_cases(
    truth: Truth,
    tables: pc.PromptCleanTables,
    clean_truth: Callable[[str], str],
    *,
    random_count: int = RANDOM_CASES,
) -> list[dict[str, Any]]:
    """全ケースの期待を作る（固定 4 本 + 境界 + 乱択）。"""
    compiled = pc.CompiledTables(tables)
    cases = [
        build_case(truth, compiled, clean_truth, f"fixed-{p.name}", p.text, f"固定 {p.role}")
        for p in FIXED_PROMPTS
    ]
    for case in cases:
        name = case["id"].removeprefix("fixed-")
        if len(case.get("ids", [])) != FIXED_PROMPT_LENGTHS[name]:
            raise ValueError(f"固定プロンプト {name} の長さが調査の値と違う: {case}")
    cases += [
        build_case(truth, compiled, clean_truth, case_id, text, why)
        for case_id, text, why in BOUNDARY_CASES
    ]
    pool = random_case_pool(truth, tables)
    for index, text in enumerate(
        pc.fuzz_strings(pool, RANDOM_SEED, random_count, RANDOM_MAX_LENGTH)
    ):
        cases.append(
            build_case(
                truth, compiled, clean_truth, f"random-{index}", text, "乱択（受理集合の文字）"
            )
        )
    check_boundaries(cases)
    return cases


def check_boundaries(cases: Sequence[Mapping[str, Any]]) -> None:
    """境界のケースが狙いどおりの決着になっていることを確かめる（狙いが外れたケースは焼かない）。"""
    by_id = {case["id"]: case for case in cases}
    wants = {
        "min-two": lambda c: len(c.get("ids", [])) == MIN_TOKENS,
        "max-512": lambda c: len(c.get("ids", [])) == MAX_LENGTH,
        "over-513": lambda c: c.get("reject") == "too-long",
        "empty": lambda c: c.get("reject") == "too-short",
        "only-space": lambda c: c.get("reject") == "too-short",
        "oov": lambda c: c.get("reject") == "oov",
        "oov-inner": lambda c: c.get("reject") == "oov",
        "ansi-unicode-digit": lambda c: c.get("cleaned") == "red",
        "added-eos": lambda c: c.get("reject") == "added-token",
        "added-extra": lambda c: c.get("reject") == "added-token",
        "metaspace": lambda c: c.get("reject") == "metaspace",
        "metaspace-inner": lambda c: "ids" in c,
        "fullwidth-comma": lambda c: c.get("cleaned") == "色调艳丽,过曝",
        "mojibake": lambda c: c.get("reject") == "mojibake",
        "entity-named": lambda c: c.get("reject") == "entity",
        "entity-bare": lambda c: c.get("reject") == "entity",
        "c1": lambda c: c.get("reject") == "c1",
        "unassigned": lambda c: c.get("reject") == "unassigned",
    }
    for case_id, want in wants.items():
        if not want(by_id[case_id]):
            raise ValueError(f"境界ケース {case_id} が狙いの決着でない: {by_id[case_id]}")
    accepted = [case for case in cases if case["id"].startswith("random-") and "ids" in case]
    if len(accepted) < len([c for c in cases if c["id"].startswith("random-")]) // 2:
        raise ValueError("乱択ケースの過半が拒否 — 受理集合の池になっていない")


def vocab_subset(truth: Truth, cases: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    """全ケースの断片の**部分文字列**に当たる語彙（各断片の格子が全語彙のときと同じになる）。

    分割に使われたピースだけを載せると、未知ノードの入り方や同点の勝ち方が部分集合の格子で
    変わりうる。断片の部分文字列を全部載せれば格子そのものが一致する（gemma の
    `_shared/gemma_tokenizer.py` の `subset_keys` と同じ考え）。
    """
    vocab = {
        token: (token_id, float(score))
        for token_id, (token, score) in enumerate(truth.backend["model"]["vocab"])
    }
    added = [entry["content"] for entry in truth.backend["added_tokens"]]
    longest = max(len(token) for token in vocab)
    keep: set[str] = set()
    for case in cases:
        if "cleaned" not in case:
            continue
        for piece in pieces_of(truth, added, case["cleaned"]):
            for start in range(len(piece)):
                for end in range(start + 1, min(len(piece), start + longest) + 1):
                    if piece[start:end] in vocab:
                        keep.add(piece[start:end])
    return [[token, *vocab[token]] for token in sorted(keep, key=lambda t: vocab[t][0])]


def build_parity_fixture(
    truth: Truth,
    asset: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """id 列のパリティ用フィクスチャ（語彙は部分集合・前処理の表は全体）。"""
    scores = asset["scores"]
    tokens = asset["vocabText"].split("\n")
    return {
        "_doc": [
            "Wan2.1 の umT5 トークナイザと前処理のパリティ用フィクスチャ"
            "（生成: tools/export-recipes/wan/umt5_tokenizer.py）。",
            "id 列の正本は transformers 5.14.1 の AutoTokenizer。受理したケースは",
            "tokenizer.json の経路の id 列も持つ（2 経路の一致は生成時に確認済み）。",
            "語彙は全ケースの断片の部分文字列だけ、前処理の表（promptClean）は",
            "畳み込みの成果物そのものなので全体を載せる。",
        ],
        "source": asset["source"],
        "versions": asset["versions"],
        "maxLength": asset["maxLength"],
        "minTokens": MIN_TOKENS,
        "t5": {
            "addedTokens": asset["addedTokens"],
            "unkId": asset["unkId"],
            "eosId": asset["eosId"],
            # MUST: 未知ノードのスコアと探索幅は**語彙全体**から決まる（部分集合から導かない）。
            "minScore": min(scores),
            "maxTokenLength": max(len(token) for token in tokens),
            "space": asset["space"],
            "vocab": vocab_subset(truth, cases),
        },
        "promptClean": asset["promptClean"],
        "cases": list(cases),
    }


def build_sweep_fixture(
    tables: pc.PromptCleanTables,
    clean_truth: Callable[[str], str],
    versions: Mapping[str, str],
    *,
    clean_runs: Sequence[pc.FuzzRun] = pc.CLEAN_RUNS,
    badness_runs: Sequence[pc.FuzzRun] = pc.BADNESS_RUNS,
    badness_probes: Sequence[tuple[str, str]] = pc.BADNESS_PROBES,
    clean_sweeps: Sequence[tuple[str, str]] = pc.CLEAN_SWEEPS,
    nfc_sweeps: Sequence[tuple[str, str]] = pc.NFC_SWEEPS,
    log: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """前処理の乱択・全コードポイント掃引の期待（digest）。

    期待は常に上流（`prompt_clean`・`BADNESS_RE`・`unicodedata`）から採る。鏡像は拒む入力を
    決めるだけで、受理した入力では上流と一致しなければ {@link pc.record} が落ちる。
    """
    import unicodedata

    from ftfy.badness import BADNESS_RE

    compiled = pc.CompiledTables(tables)
    pools = pc.fuzz_pools(tables)
    expanded = {name: pc.expand(ranges) for name, ranges in pools.items()}
    rng = pc.XorShift32(1)
    first = [rng.next() for _ in range(8)]

    clean = []
    for run in clean_runs:
        started = time.perf_counter()
        clean.append(pc.verify_fuzz(compiled, clean_truth, run, expanded[run.pool]))
        seconds = time.perf_counter() - started
        log(f"[clean-fuzz] {run.name}: {clean[-1]['outcomes']}（{seconds:.1f}s）")

    badness = []
    for run in badness_runs:
        texts = list(pc.fuzz_strings(expanded[run.pool], run.seed, run.count, run.max_length))
        bits = pc.badness_bits(BADNESS_RE, texts)
        if bits != pc.badness_bits(compiled.badness, texts):
            raise ValueError(f"BADNESS_RE の中間表現が上流と割れる（乱択 {run.name}）")
        badness.append(
            {
                "name": run.name,
                "pool": run.pool,
                "seed": run.seed,
                "count": run.count,
                "maxLength": run.max_length,
                "digest": pc.bits_digest(bits),
                "hits": bits.count("1"),
            }
        )
    for name, template in badness_probes:
        texts = [template.replace("{}", chr(cp)) for cp in pc.ALL_CODEPOINTS]
        bits = pc.badness_bits(BADNESS_RE, texts)
        if bits != pc.badness_bits(compiled.badness, texts):
            raise ValueError(f"BADNESS_RE の中間表現が上流と割れる（掃引 {name}）")
        badness.append(
            {
                "name": name,
                "template": template,
                "digest": pc.bits_digest(bits),
                "hits": bits.count("1"),
            }
        )
    log(f"[badness] {[(entry['name'], entry['hits']) for entry in badness]}")

    clean_blocks = []
    for name, template in clean_sweeps:
        started = time.perf_counter()
        blocks = pc.sweep_records(compiled, clean_truth, template)
        clean_blocks.append({"name": name, "template": template, "blocks": blocks})
        log(f"[clean-sweep] {name}（{time.perf_counter() - started:.1f}s）")

    assigned = pc.RangeSet(tables.assigned)
    domain = [cp for cp in pc.ALL_CODEPOINTS if cp in assigned]
    nfc = [
        {
            "name": name,
            "template": template,
            "blocks": pc.sweep_blocks(
                domain,
                lambda cp, template=template: unicodedata.normalize(
                    "NFC", template.replace("{}", chr(cp))
                ),
            ),
        }
        for name, template in nfc_sweeps
    ]
    log(f"[nfc-sweep] {len(nfc)} 文脈 × {len(domain)} 文字")

    return {
        "_doc": [
            "Wan2.1 の前処理（prompt_clean の鏡像）の乱択・全コードポイント掃引の期待"
            "（生成: tools/export-recipes/wan/umt5_tokenizer.py）。",
            "期待は上流（diffusers 0.39.0 の prompt_clean・ftfy 6.3.1 の BADNESS_RE・",
            "unicodedata 16.0.0 の NFC）から採り、記録の列の SHA-256 を塊ごとに持つ。",
            "TS は同じ PRNG（xorshift32）と池から同じ列を再生して突き合わせる。記録は",
            "受理なら A + 出力、拒否なら R + 理由で、各記録の後に U+0000 を置いた UTF-8 を",
            "塊ごとにハッシュする。",
        ],
        "versions": dict(sorted(versions.items())),
        "prng": {"name": "xorshift32", "seed": 1, "first": first},
        "block": pc.SWEEP_BLOCK,
        "pools": pools,
        "clean": clean,
        "badness": badness,
        "cleanSweeps": clean_blocks,
        "nfcSweeps": nfc,
    }


#: `deno fmt` の行幅（deno.json の `fmt.lineWidth`）。
FMT_LINE_WIDTH = 100


def _display_width(text: str) -> int:
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def fixture_json(value: Any, indent: int = 0) -> str:
    """フィクスチャの JSON（`deno fmt` の形に寄せる — 行幅に収まる配列は 1 行、他は 1 要素 1 行）。

    素の `indent` 付き `json.dumps` は区間 `[start, end]` や id 列まで 1 要素 1 行に展開し、
    フィクスチャが数倍に膨らむ。`deno fmt` は 1 行で書かれた配列を行幅に収まる限り 1 行のまま
    残すので、ここで 1 行にしておけば commit 形（フォーマッタの出力）も小さく保てる。
    """
    pad = " " * (indent + 2)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [
            f"{pad}{json.dumps(key, ensure_ascii=False)}: {fixture_json(item, indent + 2)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    if isinstance(value, list):
        single = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        flat = not any(isinstance(item, dict) for item in value)
        if flat and indent + _display_width(single) + 1 <= FMT_LINE_WIDTH:
            return single
        items = [f"{pad}{fixture_json(item, indent + 2)}" for item in value]
        return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"
    return json.dumps(value, ensure_ascii=False)


def write_text(path: Path, text: str) -> int:
    """書いてバイト数を返す（staging → 置換 — 途中で落ちても古いファイルが残る）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".staging")
    staging.write_text(text, encoding="utf-8")
    staging.replace(path)
    return path.stat().st_size


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=WAN21_MODELS)
    parser.add_argument("--assets-out", type=Path, default=SERIES_ROOT / SERIES_NAME)
    parser.add_argument("--fixtures-out", type=Path, default=FIXTURE_DIR)
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    started = time.perf_counter()
    snapshot = tokenizer_snapshot(args.model)
    truth = load_truth(snapshot)
    tables = pc.build_tables()
    import tokenizers
    import transformers

    versions = {
        **pc.check_versions(),
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
    }
    asset = build_asset(truth, tables, versions, args.model)
    log(f"[asset] 語彙 {len(asset['scores'])}（{time.perf_counter() - started:.1f}s）")

    clean_truth = pc.upstream_prompt_clean()
    cases = build_cases(truth, tables, clean_truth)
    outcomes = sorted({case.get("reject", "accepted") for case in cases})
    log(f"[cases] {len(cases)} 件（{outcomes}）")
    parity = build_parity_fixture(truth, asset, cases)
    sweep = build_sweep_fixture(tables, clean_truth, versions, log=log)

    # 検査を全部通ってから書く（ADR 0005 の fail loudly — 途中で落ちたら 1 バイトも書かない）。
    written = {
        "asset": write_text(args.assets_out / ASSET_FILE, json.dumps(asset, ensure_ascii=False)),
        "parity": write_text(args.fixtures_out / PARITY_FIXTURE, fixture_json(parity) + "\n"),
        "sweep": write_text(args.fixtures_out / SWEEP_FIXTURE, fixture_json(sweep) + "\n"),
    }
    log(
        json.dumps(
            {
                "bytes": written,
                "vocabSubset": len(parity["t5"]["vocab"]),
                "clean": [{run["name"]: run["outcomes"]} for run in sweep["clean"]],
                "maxRounds": max(run["maxRounds"] for run in sweep["clean"]),
                "seconds": round(time.perf_counter() - started, 1),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
