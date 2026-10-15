"""`karume-umt5-xxl` が単体で配る umT5 のトークナイザ資産を本家 `google/umt5-xxl` の pin から焼き、
TS 側のフィクスチャを作る（資産の形式 `karume-umt5-tokenizer/1`）。

    uv run --group wan --inexact python -m wan.umt5_xxl_tokenizer [--fetch]

出力は 2 本（**1 回の実行で必ず両方**出す — 同じ表から作らないと、実行時資産とフィクスチャが
別々に古びる。`wan.umt5_tokenizer` と同じ流儀）:

  ① 実行時資産 `outputs/series/umt5-xxl-tokenizer/tokenizer.json`（git 追跡外・約 8 MB）。
       配布形 `karume-umt5-xxl` のモデル `xxl` の資産 `tokenizer`（`wan.umt5_distribution`）。
  ② フィクスチャ `packages/models/tests/fixtures/umt5/tokenizer-cases.json`（git 管理）。

生成後は `deno fmt packages/models/tests/fixtures/umt5/` を掛ける（commit 形はフォーマッタが正）。
`--fetch` を付けると、pin したトークナイザのファイルが HF キャッシュに無いときに取りに行く
（無ければ付けないと落ちる）。

## 資産（`karume-umt5-tokenizer/1`）

Wan の資産（`karume-wan-umt5-tokenizer/1` — `wan.umt5_tokenizer.build_asset`）の **T5 の表の部分
そのもの**で、Wan 固有の前処理の表（`promptClean`）は持たない。T5 の表の欄は Wan と同じ 1 本
（`wan.umt5_tokenizer.t5_table_fields` — 欄の名前と意味は TS の `parseT5Tables` が読むもの）から
焼く:

    format      "karume-umt5-tokenizer/1"
    source      {"repo", "revision", "files": {ファイル名: sha256}}
                （`wan.sources.UMT5_SOURCES` の行）
    versions    {"tokenizers", "transformers"}（id 列を決める版）
    maxLength   512（`</s>` を含む id 列の最大長 — Wan と同じ値・同じ意味）
    unkId / eosId / addedTokens / space / vocabText / scores（Wan の資産と同じ欄）

## トークナイザの正本（2026-10-15 の実測）

id 列の正本は Wan と同じく pin した transformers 5.14.1 の `AutoTokenizer` で、`tokenizer.json` の
経路（`tokenizers.Tokenizer.from_file`）と**受理した入力で一致する**ことを生成時に必ず確かめる
（{@link wan.umt5_tokenizer.tokenized_expectation} を共有）。正本は sha256 を照合した
`tokenizer.json` と `tokenizer_config.json` の 2 本だけを写した置き場から読む
（`wan.sources.pinned_umt5_tokenizer`）。

- 本家の `tokenizer.json` と Wan2.1 T2V 1.3B Diffusers の `tokenizer/tokenizer.json` の違いは
  2 点だけ: Metaspace（pre_tokenizer と decoder）の旧綴り `add_prefix_space: true`（tokenizers は
  `prepend_scheme: always`・`split: true` と読む — 読んだ後の構成は Wan と同じ）と、語彙
  256,300 個のうち 65,654 個のスコアの 10 進表記（f64 で最大 3.6e-15 の差 — 本家は f32 の値を
  f64 の全桁で書き、Wan は短い表記）。トークンの並び・追加語彙 304 個（id 込み）・normalizer
  （`Replace(" {2,}", " ")`）・post_processor は同一。
- transformers 5.14.1 が組み直した backend は Wan と同じ形（正規化なし・`WhitespaceSplit` →
  `Metaspace`・Unigram の unk_id 2 — `wan.umt5_tokenizer.check_upstream_shape` が見る）で、違いは
  上のスコアだけ。TS の T5 層（`parseT5Tables` / `T5UnigramTokenizer`・`normalize` なし）がそのまま
  再現できる — 正規化を要しない。
- 2 本だけの置き場から読んだ backend は、リポのトークナイザのファイル全部
  （`special_tokens_map.json`・`spiece.model` を含む）から読んだ backend と同一。リポ直下から
  読むと transformers が「mistral の誤った regex」の警告を出すが、これは `config.json` に
  transformers 5 未満の版があって `model_type` が無いことによる誤検知で、補正は掛からない
  （backend は同一）。2 本だけの置き場では出ない。
- Wan のトークナイザとの id 列の一致: このフィクスチャの受理したケース 50 件・Wan の parity
  フィクスチャの受理したケース（前処理の後の文字列）218 件・掃引（{@link SWEEP_CASES} 本）のうち
  受理した 4,563 本の全部で、transformers 5.14.1 の経路も `tokenizer.json` の経路も Wan のものと
  同じ id 列だった（スコアの表記の差は id 列に現れなかった）。`wan/tests/test_umt5_xxl_tokenizer.py`
  が両方のトークナイザが手元にある機でフィクスチャの受理したケースを縛る。

## 前処理が無いことで増える拒否（`whitespace`）

Wan は前処理（`prompt_clean` — 空白の畳み込みと strip）を通った文字列だけをトークナイザに渡すので、
2 経路の割れ方は語彙外・本文中の追加語彙・空白の直後の `▁` の 3 つに限られていた
（`wan.umt5_tokenizer` の docstring）。単体のトークナイザは入力をそのまま渡すので、さらに 2 つの
形で割れる（実測）:

- **U+0020 以外の空白**（`WhitespaceSplit` の空白集合 25 文字のうち U+0020 を除く 24 文字 — タブ・
  改行・NBSP・全角空白など。資産の `space` が持つ集合）— transformers 5 は区切りとして捨て、
  `tokenizer.json` は文字として残す（`a\\tb` は `[289, 748, 1]` と `[289, 3, 485, 1]`）。
- **末尾の U+0020**（空白だけの入力を含む）— transformers 5 は捨て、`tokenizer.json` は `▁`（273）を
  残す（`a ` は `[289, 1]` と `[289, 273, 1]`）。

先頭の U+0020 と語の間の連続した U+0020 は割れない（`tokenizer.json` の normalizer が畳む）。
ADR 0119 決定 1 の「どの版の参照とも同じ id 列」に合わせて、この 2 つも拒む（理由 `whitespace`）。
trim や畳み込みを足すと Wan の前処理とは別の第 2 の前処理になるので足さない。

拒否の順（TS も同じ順で判定する）: 本文中の追加語彙 `added-token` → 空白の直後の `▁` `metaspace` →
`whitespace` → 語彙外 `oov` / 上限超え `too-long`（先頭から積んで先に当たった方）→ 下限割れ
`too-short`（2 トークン未満 — 空文字）。

## フィクスチャの schema（`tokenizer-cases.json`）

    _doc            生成元と読み方の説明（文字列の配列）
    assetFormat     "karume-umt5-tokenizer/1"（このケースの期待が前提にする資産の形式）
    source          資産の `source` と同じ（repo・revision・files の sha256）
    versions        資産の `versions` と同じ（transformers の版 = id 列の正本の版）
    maxLength       512 / minTokens 2
    t5              Wan の parity フィクスチャの `t5` と同じ形 — addedTokens・unkId・eosId・
                    minScore と maxTokenLength（**語彙全体**の値）・space・vocab（全ケースの断片の
                    部分文字列に当たる語彙の部分集合 `[token, id, score]` — 格子は全語彙と同じ）
    cases[]         id・why（狙い）・text（そのまま渡す入力）と、次のどちらか:
                    受理 — `ids`（i32 の id 列・末尾は `</s>` = 1）と `tokenizerJsonIds`（同じ列）
                    拒否 — `reject`（上の 6 つの理由）。`whitespace` だけは割れ方の記録
                    `split: {transformers, tokenizerJson}`（2 経路の id 列）を併せて持つ

MUST: transformers / tokenizers / huggingface_hub は関数の中で import する（`wan` グループは既定の
sync に入らない — `tests/test_optional_group_imports.py`）。torch も import しない。
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

#: 系列の席（`<名前>-tokenizer/tokenizer.json` — docs/assets-layout.md）。
SERIES_NAME = "umt5-xxl-tokenizer"
ASSET_FILE = "tokenizer.json"

#: 資産の形式の版（TS 側が知らない版を黙って読まない）。
ASSET_FORMAT = "karume-umt5-tokenizer/1"

#: 既定のモデル（`wan.sources.UMT5_SOURCES` のキー — 配布形のモデル名
#: `wan.umt5_distribution.UMT5_DEFAULT_MODEL` と同じ綴り。あちらがこのモジュールを import するので
#: こちらからは import できない — 一致は `wan/tests/test_umt5_xxl_tokenizer.py` が見る）。
DEFAULT_MODEL = "xxl"

#: フィクスチャの置き場（消費者は `packages/models/tests/` の umT5 のトークナイザのテスト）。
FIXTURE_DIR = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "umt5"
CASES_FIXTURE = "tokenizer-cases.json"

#: 資産の `versions` に載せる版（id 列を決めるもの — 前処理が無いので ftfy などは載せない）。
ASSET_VERSIONS: tuple[str, ...] = ("tokenizers", "transformers")

#: U+0020（受理する唯一の空白）。
SPACE = " "

#: 長めのプロンプト（1 行・100 トークン前後 — 受理の中ほどの長さ）。
LONG_PROMPT = (
    "A wide shot of a quiet harbor at dawn: fishing boats rest on calm water, gulls circle above"
    " the masts, a lighthouse blinks on the rocky point, and soft orange light spreads across the"
    " clouds while a small dog runs along the wooden pier toward an old man mending his nets."
)

#: 境界と多言語のケース（id・入力・狙い）。目に見えない文字は必ず \\uXXXX で書く。
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

#: 狙いの決着（狙いが外れたケースは焼かない — 上流の挙動が変わった日に気づくため）。
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

#: 乱択ケース（フィクスチャに載せる分 — seed 固定）。
RANDOM_CASES = 40
RANDOM_SEED = 7
RANDOM_MAX_LENGTH = 40
#: 乱択の池に混ぜる U+0020 の重み（語の切れ目を作る — 先頭・連続・末尾の U+0020 も出る）。
RANDOM_SPACE_WEIGHT = 2000

#: 2 経路の一致の掃引（フィクスチャに載せない — 生成のたびに走る門）。
SWEEP_CASES = 5000
SWEEP_SEED = 11


def whitespace_split(space: pc.RangeSet, text: str) -> bool:
    """前処理なしで 2 経路が割れる空白の形か（U+0020 以外の空白・末尾の U+0020）。"""
    return text.endswith(SPACE) or any(ch != SPACE and ord(ch) in space for ch in text)


def expectation(
    truth: ut.Truth, added: Sequence[str], space: pc.RangeSet, text: str
) -> dict[str, Any]:
    """そのまま渡す入力の期待（id 列か拒否の理由）。

    拒否の順は TS と同じ: 追加語彙 → 空白の直後の `▁`（ここまで Wan と同じ 1 本 —
    {@link ut.version_split}）→ 空白の形 → 語彙外 / 上限超え → 下限割れ（Wan と同じ 1 本 —
    {@link ut.tokenized_expectation}。2 経路の割れ方の門もそちらが持つ）。
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
    """ケースの決着（`accepted` か拒否の理由）。"""
    return "accepted" if "ids" in case else case["reject"]


def random_pool(truth: ut.Truth, space: pc.RangeSet) -> list[int]:
    """乱択の池 — 1 文字として引ける語彙のうち空白でない文字 + U+0020（重みつき）。"""
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
    """全ケースの期待（境界・多言語 + 乱択）。狙いが外れた境界のケースがあれば落とす。"""
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
        raise ValueError(f"境界のケースが狙いの決着でない（実際の決着）: {wrong}")
    if len(next(case for case in cases if case["id"] == "max-512")["ids"]) != ut.MAX_LENGTH:
        raise ValueError("max-512 が上限ちょうどの長さでない")
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
    """乱択 `count` 本の決着の数（2 経路の割れ方が既知の形でなければ
    {@link ut.tokenized_expectation} が落とす — フィクスチャに載せない分の門）。"""
    added = [entry["content"] for entry in truth.backend["added_tokens"]]
    space = pc.RangeSet(ut.whitespace_split_ranges())
    counts: dict[str, int] = {}
    for text in random_texts(random_pool(truth, space), SWEEP_SEED, count):
        result = outcome(expectation(truth, added, space, text))
        counts[result] = counts.get(result, 0) + 1
    return dict(sorted(counts.items()))


def build_asset(truth: ut.Truth, model: str, versions: Mapping[str, str]) -> dict[str, Any]:
    """TS が引くだけの資産（形式 {@link ASSET_FORMAT} — T5 の表の欄は Wan の資産と同じ 1 本）。"""
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
    """TS 側のフィクスチャ（語彙は部分集合 — schema はモジュールの docstring）。"""
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
            # MUST: 未知ノードのスコアと探索幅は**語彙全体**から決まる（部分集合から導かない）。
            "minScore": min(scores),
            "maxTokenLength": max(len(token) for token in tokens),
            "space": asset["space"],
            "vocab": ut.vocab_subset(truth, [case["text"] for case in cases]),
        },
        "cases": list(cases),
    }


def load_pinned_truth(model: str, directory: Path, *, fetch: bool = False) -> ut.Truth:
    """pin したトークナイザのファイルを照合して `directory` へ写し、そこから正本を読む。"""
    pinned_umt5_tokenizer(model, directory, fetch=fetch)
    return ut.load_truth(directory)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(UMT5_SOURCES))
    parser.add_argument(
        "--fetch", action="store_true", help="pin したトークナイザのファイルを HF から取得する"
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
        log(f"[asset] 語彙 {len(asset['scores'])}（{time.perf_counter() - started:.1f}s）")
        cases = build_cases(truth)
        log(f"[cases] {len(cases)} 件（{sorted({outcome(case) for case in cases})}）")
        swept = sweep(truth)
        log(f"[sweep] {SWEEP_CASES} 本: {swept}")
        fixture = build_fixture(truth, asset, cases)

    # 検査を全部通ってから書く（ADR 0005 の fail loudly — 途中で落ちたら 1 バイトも書かない）。
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
