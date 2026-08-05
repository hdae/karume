r"""examples/anima デモのプロンプト層（トークナイザ）の資産 prep（表焼き + フィクスチャ）。

`export_anima.py` が**グラフ**を、`anima_pipeline.py` が**ホスト側の数の正**を出すのに対し、
こちらが扱うのは「プロンプト文字列 → トークン id 列」だけ。モデルグラフには触らない。

    uv run --group anima python anima_demo.py

出力は 2 系統（**1 回の実行で必ず両方**出す — 同じ表から作らないと、実行時資産と
フィクスチャが別々に古びて「テストは緑だがデモだけ別の id 列」になる）:

  ① 実行時資産 `models/anima-demo/text/`（`.gitignore` 配下・計 4.6MB 級）
       qwen2-tokenizer.json   語彙 / merges / 文字クラス表 / NFC 分節表 / 追加語彙
       t5-tokenizer.json      語彙 / スコア / 正規化表（Precompiled の畳み込み）/ 追加語彙
  ② パリティ用フィクスチャ `packages/runtime/tests/fixtures/anima-text/parity.json`
     （**git 管理**・470KB 級）
       全ケース（PROMPT_CASES）の参照 id 列と、その再現に要る語彙の**部分集合**
       （151k / 32k の語彙を commit しないための分離）、および NFC の実測対

MUST: 資産を `models/anima/` 直下に置かない。あちらは `packages/runtime/tests/e2e_anima_test.ts` が
**ターゲット全量の等値検査**をしているディレクトリで、別種の資産を混ぜると資産完全性の
テストが赤くなる（`models/sbv2-demo/` / `models/anima-pipeline/` を分けたのと同じ理由）。

畳み方の根拠と検証は `karume/anima_text.py` の docstring が正本。要点だけ:
**Unicode 判定は TS で再実装しないし標準 API にも委ねない** — 正本（Rust の `tokenizers` /
その正規表現エンジン / `unicode-normalization`）を Python から全コードポイントに当てて
閉区間表・写像表へ畳む。畳み込みの同値は emit のたびに網羅 + 乱択で検査し、加えて参照実装と
`AutoTokenizer` の id 列を乱択 2,000 件で突き合わせる。外れたら**何も書かない**（ADR 0005）。

生成後は `deno fmt packages/runtime/tests/fixtures/anima-text/parity.json` を掛ける
（commit 形はフォーマッタが正 — `deno task verify` の `fmt --check` が fixtures も見る）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from karume import anima_text as at

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"

#: 実行時資産の置き場（`models/` は `.gitignore` 済み）。Deno 側 `examples/anima/text/tokenizer.ts`
#: が同じ名前を読む。
DEFAULT_ASSETS_DIR = REPO_ROOT / "models" / "anima-demo" / "text"
QWEN_ASSET_FILE = "qwen2-tokenizer.json"
T5_ASSET_FILE = "t5-tokenizer.json"

#: パリティ用フィクスチャ（**git 管理**）。Deno 側
#: `packages/runtime/tests/anima_tokenizer_test.ts` が読む。
DEFAULT_FIXTURE_PATH = (
    REPO_ROOT / "packages" / "runtime" / "tests" / "fixtures" / "anima-text" / "parity.json"
)

#: 正本の呼び出しに合わせた切り詰め長。`anima_pipeline.py` の `--max-sequence-length` の既定
#: （両トークナイザ共通で `max_length=512, truncation=True` として渡される）。
MAX_LENGTH = 512

#: パリティ検証のプロンプト集。**素朴な移植が落ちる境界を 1 件ずつ**持たせる設計で、
#: 目に見えない文字は必ず \uXXXX で書く（編集の途中で消えたり別物が混ざるため）。
PROMPT_CASES: list[tuple[str, str, str]] = [
    (
        "tags",
        "1girl, solo, long hair, blue eyes, school uniform, cherry blossoms, "
        "outdoors, smile, upper body, masterpiece, best quality",
        "既定プロンプト — danbooru タグ列",
    ),
    (
        "negative",
        "low quality, worst quality, blurry, bad anatomy, jpeg artifacts",
        "既定ネガティブ",
    ),
    (
        "digits",
        "1girl, 2boys, 3 cats, 2024, 1.5x, 100%",
        "Qwen2 の `\\p{N}` は 1 文字ずつ切れる",
    ),
    (
        "ascii_punct",
        "a<b>c&d\"e'f`g~h|i\\j/k",
        "記号の連続 — `[^\\s\\p{L}\\p{N}]+` の貪欲さ",
    ),
    (
        "apostrophe",
        "it's a cat's toy, I'M HERE, they'll, we've, don't, I'd",
        "`(?i:'s|'t|'re|'ve|'m|'ll|'d)` の大小無視分岐",
    ),
    (
        "apostrophe_fold",
        "it'\u017fs and it'S",
        "`(?i:)` は simple case folding — U+017F も `'S` も s に畳まれる",
    ),
    (
        "long_s",
        "itſs and it's",
        "U+017F 単体（アポストロフィ無し — 短縮形分岐には入らない）",
    ),
    (
        "japanese",
        "夕焼けの空、桜が舞う。少女がひとり立っている。",
        "日本語（T5 は unk 融合 / Qwen2 は byte-level）",
    ),
    (
        "mixed",
        "1girl, 猫耳, ロングヘア, masterpiece",
        "日英混在 — 語 / 非語の切り替わり",
    ),
    (
        "emoji",
        "a cat \U0001f408 and a flag \U0001f1ef\U0001f1f5 and ❤\ufe0f love",
        "絵文字・地域表示子・異体字セレクタ",
    ),
    (
        "zwj",
        "family \U0001f468\u200d\U0001f469\u200d\U0001f466 photo",
        "ZWJ 連結列 — T5 正規化は ZWJ を空白へ写す",
    ),
    (
        "combining",
        "café vs cafe\u0301, A\u0301\u0301, ᴬ\u0300, e\u0303\u033c",
        "結合文字 — Precompiled の「最短接頭辞が勝ち残りは捨てる」を直接叩く",
    ),
    (
        "odd_spaces",
        "wide\u00a0space and\u3000ideographic\u2028line\u000bvtab",
        "各種空白（NBSP / 全角 / 行区切り / 垂直タブ）",
    ),
    (
        "spaces",
        "  leading and   inner   and trailing  ",
        "空白の連続・先頭末尾空白 — `\\s+(?!\\S)` の後戻り",
    ),
    (
        "newlines",
        "line1\nline2\r\nline3\n\n\nline4",
        "改行と CRLF（`\\s*[\\r\\n]+` と GB3）",
    ),
    (
        "only_space",
        "   ",
        "空白のみ — T5 は `</s>` だけになり Dim(min=2) を外す",
    ),
    (
        "single_char",
        "a",
        "1 文字 — Qwen2 が 1 トークンになり Dim(min=2) を外す",
    ),
    (
        "qwen_special",
        "<|endoftext|>a<|im_start|>b<|vision_pad|>",
        "Qwen2 の追加語彙は正規化前に leftmost-longest で切り出す",
    ),
    (
        "t5_special",
        "<extra_id_0> foo <extra_id_99> </s> <pad>",
        "T5 の追加語彙（<extra_id_*> 群）",
    ),
    (
        "special_lookalike",
        "<|endoftext|> <|endoftext|x <|not_a_token|>",
        "特殊トークン「らしき」文字列 — 途中まで一致しても切り出さない",
    ),
    (
        "control",
        "abc\ufeffd\u200be\u00adf\u0001g",
        "BOM・ゼロ幅空白・soft hyphen・制御文字",
    ),
    (
        "metaspace",
        "under\u2581score",
        "入力に U+2581（Metaspace の置換文字）が来る場合",
    ),
    (
        "long",
        ", ".join(["masterpiece", "best quality", "highly detailed", "1girl", "solo"] * 40),
        "非常に長い列 — 512 での切り詰め",
    ),
    (
        "very_long_word",
        "a" * 600,
        "長い 1 語 — BPE の反復と切り詰め",
    ),
    (
        "cjk_punct",
        "「猫」と『犬』、あるいは…！？",
        "日本語の約物（全角記号の正規化）",
    ),
    (
        "accents",
        "naïve café résumé Ångström ﬁ ﬂ ½ Ⅻ",
        "合字・分数・ローマ数字（NFKC 系の写像）",
    ),
    (
        "thai_indic",
        "กำ क\u093c\u094dष ൎक",
        "タイ / デーヴァナーガリー（SpacingMark・Prepend）",
    ),
    (
        "hangul",
        "각 각",
        "ハングル合成済み音節と字母列",
    ),
]


def _resolve(repo: str, filename: str, snapshot: Path | None) -> Path:
    if snapshot is not None:
        return snapshot / filename
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename))


def check_upstream_shape(qwen_raw: dict[str, Any], t5_raw: dict[str, Any]) -> None:
    """焼いた表が前提にしている上流 `tokenizer.json` の構造を検査する（fail loudly）。

    ここで見る項目はどれも「外れると id 列だけが静かに変わる」形で、`build_cases` の
    AutoTokenizer 突合は **PROMPT_CASES が叩く範囲でしか**拾えない（NFC のエンジン差が
    28 ケースをすり抜けたのと同じ穴）。前提そのものを構造で固定する。
    """
    if qwen_raw["normalizer"]["type"] != "NFC":
        raise ValueError(f"Qwen2 の normalizer が NFC でない: {qwen_raw['normalizer']['type']}")
    pre = qwen_raw["pre_tokenizer"]["pretokenizers"]
    if pre[0]["type"] != "Split":
        raise ValueError(f"Qwen2 の pre_tokenizer 先頭が Split でない: {pre[0]['type']}")
    byte_level = [entry for entry in pre if entry["type"] == "ByteLevel"]
    if len(byte_level) != 1 or byte_level[0]["add_prefix_space"]:
        raise ValueError("Qwen2 の ByteLevel が 1 段・add_prefix_space=false でない")
    if t5_raw["model"].get("byte_fallback"):
        raise ValueError("T5 の Unigram が byte_fallback を使っている（unk 融合の前提が崩れる）")

    # post_processor: Qwen2 は**何も足さない**（だから切り詰めが単純な先頭切りでよい）、
    # T5 は末尾に `</s>` を 1 つだけ足す（参照実装が自前で足している分）。
    qwen_post = qwen_raw["post_processor"]
    if qwen_post["type"] != "TemplateProcessing" or any(
        "SpecialToken" in entry for entry in qwen_post["single"]
    ):
        raise ValueError(f"Qwen2 の post_processor が特殊トークンを足す: {qwen_post}")
    t5_post = t5_raw["post_processor"]
    t5_specials = [
        entry["SpecialToken"]["id"] for entry in t5_post["single"] if "SpecialToken" in entry
    ]
    if t5_post["type"] != "TemplateProcessing" or t5_specials != ["</s>"]:
        raise ValueError(f"T5 の post_processor が `</s>` 1 つだけを足す形でない: {t5_post}")

    # 追加語彙のフラグは全て既定値のはず（`normalized: true` は正規化**後**に照合される /
    # `lstrip` `rstrip` は前後の空白を食う / `single_word` は語境界を要求する — どれも
    # 「正規化の前に leftmost-longest で切り出す」という写経の意味を変える）。
    for name, raw in (("Qwen2", qwen_raw), ("T5", t5_raw)):
        for entry in raw["added_tokens"]:
            flags = {key: entry[key] for key in ("lstrip", "rstrip", "single_word", "normalized")}
            if any(flags.values()):
                raise ValueError(
                    f"{name} の追加語彙 {entry['content']!r} のフラグが既定値でない: {flags}"
                )


def build_assets(repo: str, snapshot: Path | None, fuzz: int) -> dict[str, Any]:
    """`tokenizer.json` 2 本から、Deno 側が引くだけの表を作って検証する。

    検証（畳み込みの同値）を通らなければ例外で落ちる = **1 バイトも書かない**。畳んだ根拠
    そのものなので、`--fuzz 0` で薄めることはできても網羅の 3 本は外せない。
    """
    qwen_path = _resolve(repo, "tokenizer/tokenizer.json", snapshot)
    t5_path = _resolve(repo, "t5_tokenizer/tokenizer.json", snapshot)
    source = at.AnimaTokenizerSource(qwen_path, t5_path)
    qwen_raw, t5_raw = source.qwen_json(), source.t5_json()

    if qwen_raw["model"]["type"] != "BPE" or t5_raw["model"]["type"] != "Unigram":
        raise ValueError("model.type が想定（BPE / Unigram）と違う")
    check_upstream_shape(qwen_raw, t5_raw)

    started = time.perf_counter()
    classes = at.build_char_classes()
    class_stats = at.verify_char_classes(classes)
    case_fold = at.build_case_fold()
    class_stats["caseFold"] = len(case_fold)
    print(f"[classes] {class_stats}（{time.perf_counter() - started:.1f}s）", flush=True)

    started = time.perf_counter()
    nfc_segments = at.build_nfc_segments()
    nfc_stats = at.verify_nfc_segments(nfc_segments, fuzz=fuzz)
    print(f"[nfc] {nfc_stats}（{time.perf_counter() - started:.1f}s）", flush=True)

    started = time.perf_counter()
    pattern = qwen_raw["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    scan_stats = at.verify_qwen_pre_tokenize(pattern, classes, case_fold)
    print(f"[pre-token] {scan_stats}（{time.perf_counter() - started:.1f}s）", flush=True)

    started = time.perf_counter()
    charsmap = at.t5_charsmap(t5_raw)
    tables = at.build_spm_tables(charsmap)
    table_stats = at.verify_spm_tables(tables, charsmap, fuzz=fuzz)
    print(f"[normalizer] {table_stats}（{time.perf_counter() - started:.1f}s）", flush=True)

    qwen_vocab: dict[str, int] = qwen_raw["model"]["vocab"]
    qwen_merges = [(left, right) for left, right in qwen_raw["model"]["merges"]]
    # 行区切り形式（行番号 = id / rank）が壊れないことを emit の側で検査する。上流の語彙が
    # 変わったときに黙って壊れる形を塞ぐ（TS 側は行を split するだけで、異常を検出できない）。
    for token in qwen_vocab:
        if "\n" in token or " " in token:
            raise ValueError(f"Qwen2 語彙に改行 / 空白を含むトークン {token!r} — 行区切りが壊れる")
    t5_vocab = [(token, float(score)) for token, score in t5_raw["model"]["vocab"]]
    for token, _ in t5_vocab:
        if "\n" in token:
            raise ValueError(f"T5 語彙に改行を含むトークン {token!r} — 行区切りが壊れる")

    ordered_qwen = [""] * len(qwen_vocab)
    for token, tid in qwen_vocab.items():
        ordered_qwen[tid] = token
    if any(token == "" for token in ordered_qwen):
        raise ValueError("Qwen2 語彙の id に穴がある（行番号 = id の前提が崩れている）")

    qwen_asset = {
        "source": f"{repo}#tokenizer",
        "maxLength": MAX_LENGTH,
        "addedTokens": sorted(at.added_token_map(qwen_raw).items(), key=lambda kv: kv[1]),
        "classes": classes,
        # 短縮形 `(?i:'s|…)` の大小無視。ASCII 相当ではない（U+017F → s）。
        "caseFold": sorted([cp, target] for cp, target in case_fold.items()),
        # 正本の NFC と素の NFC が割れる cp（ここで分節してから正規化する）。
        "nfcSegments": nfc_segments,
        # 行番号 0-origin = id。byte-level 語彙は改行も空白も含まない（上で検査済み）。
        "vocabText": "\n".join(ordered_qwen),
        # 行番号 0-origin = rank。
        "mergesText": "\n".join(f"{left} {right}" for left, right in qwen_merges),
        # 行数の期待値。欠けた `mergesText` は「BPE の分割だけが変わる」沈黙誤値になるので、
        # TS 側が split の結果と突き合わせられるようにする。
        "vocabCount": len(ordered_qwen),
        "mergesCount": len(qwen_merges),
    }
    t5_asset = {
        "source": f"{repo}#t5_tokenizer",
        "maxLength": MAX_LENGTH,
        "unkId": t5_raw["model"]["unk_id"],
        "eosId": at.added_token_map(t5_raw)["</s>"],
        "addedTokens": sorted(at.added_token_map(t5_raw).items(), key=lambda kv: kv[1]),
        # T5 の WhitespaceSplit は Qwen2 の `\s` と同じ集合（`verify_char_classes` が実測済み）。
        "space": classes["space"],
        "normalizer": tables.to_json(),
        "vocabText": "\n".join(token for token, _ in t5_vocab),
        "scores": [score for _, score in t5_vocab],
    }
    return {
        "qwen": qwen_asset,
        "t5": t5_asset,
        "classes": classes,
        "caseFold": case_fold,
        "nfcSegments": nfc_segments,
        "tables": tables,
        "qwenVocab": qwen_vocab,
        "qwenMerges": qwen_merges,
        "t5Vocab": t5_vocab,
        "stats": {
            "classes": class_stats,
            "nfc": nfc_stats,
            "preTokenize": scan_stats,
            "normalizer": table_stats,
        },
        "paths": {"qwen": str(qwen_path), "t5": str(t5_path)},
    }


def references(built: dict[str, Any]) -> tuple[at.Qwen2Reference, at.T5Reference]:
    """焼いた表**だけ**を見る参照実装の対（TS 実装の鏡像）。"""
    qwen = at.Qwen2Reference(
        vocab=built["qwenVocab"],
        merges={pair: rank for rank, pair in enumerate(built["qwenMerges"])},
        added=dict(built["qwen"]["addedTokens"]),
        classes=built["classes"],
        case_fold=built["caseFold"],
        nfc_segments=built["nfcSegments"],
    )
    t5_vocab = {token: (i, score) for i, (token, score) in enumerate(built["t5Vocab"])}
    t5 = at.T5Reference(
        vocab=t5_vocab,
        min_score=min(score for _, score in built["t5Vocab"]),
        unk_id=built["t5"]["unkId"],
        eos_id=built["t5"]["eosId"],
        added=dict(built["t5"]["addedTokens"]),
        tables=built["tables"],
        space=built["classes"]["space"],
    )
    return qwen, t5


def reference_ids(tokenizer: Any, text: str) -> list[int]:
    """`anima_pipeline.encode_text` と**同じ呼び方**で正本の id 列を取る。

    MUST: chat template を通さない（素の `__call__`）。`tokenizer/chat_template.jinja` は
    同梱されているが Anima の経路では使わず、`apply_chat_template` を通すと `<|im_start|>`
    系が前置されて id 列が丸ごと別物になる。
    """
    encoded = tokenizer([text], padding="longest", max_length=MAX_LENGTH, truncation=True)
    return list(encoded["input_ids"][0])


def build_cases(repo: str, built: dict[str, Any]) -> list[dict[str, Any]]:
    """全ケースの正本 id 列を採り、参照実装と一致することを確かめる。

    正本は `transformers` の `AutoTokenizer`。ここが緑であることが「焼いた表に必要な情報が
    全部入っている」ことの機械証明で、TS 実装はこの写経になる。1 件でも違えば **emit しない**
    （フィクスチャに「参照実装では再現できない id 列」を焼き付けない）。同じ突合を
    `tests/test_anima_demo.py` が**commit 済みフィクスチャ**に対しても走らせる（上流の
    tokenizer.json が動いたときに、再生成しなくても気づけるようにするため）。
    """
    from transformers import AutoTokenizer

    truth = {
        "qwen": AutoTokenizer.from_pretrained(repo, subfolder="tokenizer"),
        "t5": AutoTokenizer.from_pretrained(repo, subfolder="t5_tokenizer"),
    }
    qwen_ref, t5_ref = references(built)

    cases: list[dict[str, Any]] = []
    for name, text, why in PROMPT_CASES:
        want_q = reference_ids(truth["qwen"], text)
        want_t = reference_ids(truth["t5"], text)
        got_q = qwen_ref.encode(text, MAX_LENGTH)
        got_t = t5_ref.encode(text, MAX_LENGTH)
        if got_q != want_q:
            raise ValueError(f"[{name}] Qwen2 id 列が正本と違う\n  正本={want_q}\n  参照={got_q}")
        if got_t != want_t:
            raise ValueError(f"[{name}] T5 id 列が正本と違う\n  正本={want_t}\n  参照={got_t}")
        cases.append(
            {
                "id": name,
                "why": why,
                "text": text,
                "qwenIds": want_q,
                "t5Ids": want_t,
                "t5Normalized": at.normalize_with_tables(built["tables"], text),
            }
        )
    return cases


#: id 列の乱択突合に混ぜる文字（絵文字 / 地域表示子 / 異体字 / 結合文字 / ハングル字母 /
#: 各種空白・制御 / 珍しい面の文字）。`PROMPT_CASES` が叩かない組み合わせを機械的に踏ませる。
_FUZZ_EXOTIC = (
    0x000A, 0x000D, 0x0009, 0x0020, 0x00A0, 0x2028, 0x3000, 0x200B, 0x200D, 0xFEFF,
    0x00AD, 0x0001, 0x0027, 0x0060, 0x005C, 0x002F, 0x0025, 0x2581, 0x017F, 0x00DF,
    0x0300, 0x0301, 0x0302, 0x0303, 0x0316, 0x0328, 0x0334, 0x033C, 0x093C, 0x094D,
    0x0E33, 0x0EB3, 0x0F71, 0x0F72, 0x3099, 0xFE0F, 0xE0100, 0x1100, 0x1161, 0x11A8,
    0xAC00, 0x304B, 0x3053, 0x4E16, 0x732B, 0xFF21, 0xFB01, 0x00BD, 0x216B, 0x2026,
    0x1F408, 0x1F600, 0x1F1EF, 0x1F1F5, 0x1F3FB, 0x1F468, 0x2F929, 0x0818, 0x089A,
)  # fmt: skip

#: 乱択の素材にする普通の語（実プロンプトに近い列も混ぜる — 全部が exotic だと BPE の
#: 通常経路が薄くなる）。
_FUZZ_WORDS = ("1girl", "solo", "masterpiece", "cat's", "café", "夕焼け", "2024", "<|endoftext|>")


def verify_reference_fuzz(
    repo: str, built: dict[str, Any], *, count: int = 2_000, seed: int = 0
) -> dict[str, Any]:
    """乱択プロンプトで参照実装と正本（AutoTokenizer）の id 列を突き合わせる。

    `PROMPT_CASES` は「素朴な移植が落ちる境界」を人手で 1 件ずつ置いたもので、**そこに
    無い文字は 1 つも検査されない**（NFC のエンジン差はこの穴を通って id 列を割った）。
    ここは網羅ではないが、seed 固定の乱択で「人手の台本が思いつかなかった組み合わせ」を
    機械的に踏ませる恒久の門にする。1 件でも違えば emit しない。
    """
    import random

    from transformers import AutoTokenizer

    truth = {
        "qwen": AutoTokenizer.from_pretrained(repo, subfolder="tokenizer"),
        "t5": AutoTokenizer.from_pretrained(repo, subfolder="t5_tokenizer"),
    }
    # 参照実装は `used_vocab` を溜めるので、フィクスチャの部分集合を膨らませないよう
    # **この検査専用のインスタンス**を作る（build_fixture は別に組み直す）。
    qwen_ref, t5_ref = references(built)
    alphabet = [chr(cp) for cp in _FUZZ_EXOTIC]
    alphabet += [chr(cp) for start, end in built["nfcSegments"] for cp in range(start, end + 1)]
    alphabet += [chr(cp) for cp in range(0x20, 0x7F)]
    alphabet += list(_FUZZ_WORDS)
    rng = random.Random(seed)
    for index in range(count):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 16)))
        for which, ref, tokenizer in (
            ("Qwen2", qwen_ref, truth["qwen"]),
            ("T5", t5_ref, truth["t5"]),
        ):
            want = reference_ids(tokenizer, text)
            got = ref.encode(text, MAX_LENGTH)
            if got != want:
                raise ValueError(
                    f"乱択 {index} 件目の {which} id 列が正本と違う\n"
                    f"  入力={[hex(ord(c)) for c in text]}\n  正本={want}\n  参照={got}"
                )
    return {"count": count, "seed": seed, "alphabet": len(alphabet)}


def build_fixture(built: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """フィクスチャ = 参照 id 列 + それを再現できる**最小の**語彙・merges。

    語彙（151k / 32k）を丸ごと commit しない。正規化表と文字クラス表は**畳み込みの成果物
    そのもの**（= 検証対象）なので削らない。
    """
    qwen_ref, t5_ref = references(built)
    for case in cases:
        qwen_ref.encode(case["text"], MAX_LENGTH)
        t5_ref.encode(case["text"], MAX_LENGTH)

    merge_rank = {pair: rank for rank, pair in enumerate(built["qwenMerges"])}
    t5_vocab = {token: (i, score) for i, (token, score) in enumerate(built["t5Vocab"])}
    return {
        "_doc": [
            "Anima プロンプト層のパリティ用フィクスチャ（生成: tools/exporter/anima_demo.py）。",
            "正本は transformers の AutoTokenizer で、id 列はそこから採った実測値。語彙と",
            "merges は全ケースの再現に要る**部分集合**だけを載せる（151k / 32k を commit",
            "しないため）。正規化表・文字クラス表は畳み込みの成果物そのものなので全体を載せる。",
        ],
        "source": {"repo": built["qwen"]["source"].split("#")[0], "reference": "tokenizers"},
        "maxLength": MAX_LENGTH,
        "qwen": {
            "addedTokens": built["qwen"]["addedTokens"],
            "classes": built["classes"],
            "caseFold": built["qwen"]["caseFold"],
            "nfcSegments": built["nfcSegments"],
            # `[入力, 正本の NFC 出力]` — TS 側は素の `normalize("NFC")` では再現できない
            # （分節表を引かない実装がここで落ちる）。
            "nfcCases": at.nfc_fixture_cases(built["nfcSegments"]),
            "vocab": {t: built["qwenVocab"][t] for t in sorted(qwen_ref.used_vocab)},
            "merges": sorted(
                ([left, right, merge_rank[(left, right)]] for left, right in qwen_ref.used_merges),
                key=lambda entry: entry[2],
            ),
        },
        "t5": {
            "addedTokens": built["t5"]["addedTokens"],
            "unkId": built["t5"]["unkId"],
            "eosId": built["t5"]["eosId"],
            # MUST: 未知ノードのスコアと Viterbi の探索幅は**語彙全体**から決まる。部分集合
            # から計算すると別の分割になるので、部分集合とは別に持つ。
            "minScore": min(score for _, score in built["t5Vocab"]),
            "maxTokenLength": max(len(token) for token, _ in built["t5Vocab"]),
            "space": built["classes"]["space"],
            "normalizer": built["tables"].to_json(),
            "vocab": [
                [token, t5_vocab[token][0], t5_vocab[token][1]]
                for token in sorted(t5_ref.used_vocab)
            ],
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--snapshot", type=Path, default=None, help="HF スナップショット直指定")
    parser.add_argument("--assets-out", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--fixtures-out", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--fuzz", type=int, default=200_000, help="畳み込み検査の乱択件数")
    parser.add_argument("--id-fuzz", type=int, default=2_000, help="id 列の乱択突合の件数")
    args = parser.parse_args()

    built = build_assets(args.repo, args.snapshot, args.fuzz)
    cases = build_cases(args.repo, built)
    print(f"[cases] {len(cases)} 件が正本と一致", flush=True)

    started = time.perf_counter()
    fuzz_stats = verify_reference_fuzz(args.repo, built, count=args.id_fuzz)
    print(f"[id-fuzz] {fuzz_stats}（{time.perf_counter() - started:.1f}s）", flush=True)

    args.assets_out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}
    for filename, payload in ((QWEN_ASSET_FILE, built["qwen"]), (T5_ASSET_FILE, built["t5"])):
        path = args.assets_out / filename
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        written[filename] = path.stat().st_size

    fixture = build_fixture(built, cases)
    args.fixtures_out.parent.mkdir(parents=True, exist_ok=True)
    args.fixtures_out.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "stats": {**built["stats"], "idFuzz": fuzz_stats},
                "sources": built["paths"],
                "assets": {"dir": str(args.assets_out), "bytes": written},
                "fixture": {
                    "path": str(args.fixtures_out),
                    "bytes": args.fixtures_out.stat().st_size,
                    "cases": len(cases),
                    "qwenVocabSubset": len(fixture["qwen"]["vocab"]),
                    "qwenMergeSubset": len(fixture["qwen"]["merges"]),
                    "t5VocabSubset": len(fixture["t5"]["vocab"]),
                },
            },
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
