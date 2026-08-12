r"""Irodori-TTS v4 のテキスト前処理層（`normalize_text` + Unigram トークナイザ）の資産と golden。

`export_irodori.py` が**グラフ**を出すのに対し、こちらが扱うのは「生テキスト → トークン id 列」
だけ。モデルグラフにも重みにも触らない（読むのはチェックポイントの `__metadata__` の
`bos_token_id` / `pad_token_id` だけ）。

    cd tools/exporter
    uv run --with 'transformers==5.14.1' python irodori_tokenizer.py

出力は 4 本（既定 `outputs/series/irodori-v4-small/tokenizer/`・`.gitignore` 配下）:

    tokenizer.json         実行時資産（vocabText / scores / addedTokens / bosId / padId /
                           unkId / byteBaseId）
    golden.encode.json     `{raw, normalized, ids}` の固定ケースと batch_encode 同値ケース
    golden.normalize.json  `normalize_text` の `{raw, normalized}`（各置換規則を最低 1 回発火）
    nfkc-diff.json         NFKC が**恒等でない**単一コードポイントの写像表 `{cp: 正規化後}`

加えて TS 側パリティ門のフィクスチャ 1 本（**git 管理**・`.gitignore` の外）:

    packages/models/tests/fixtures/irodori-text/parity.json
        上の 4 本と同じ内容 + 語彙の**部分集合**（102,400 本を commit しないための分離。
        格子が全語彙と同じであることは {@link check_subset_covers_lattice} が実測する）

生成後は `deno fmt packages/models/tests/fixtures/irodori-text/parity.json` を掛ける
（commit 形はフォーマッタが正 — `deno task verify` の `fmt --check` が fixtures も見る）。

## 資産の形（anima の T5 資産に倣う）

`packages/models/src/anima/text/tokenizer.ts` の `parseT5Asset` と同じ畳み方: 語彙は
**改行 join の 1 本の文字列**（行番号 0-origin = id）+ スコア配列。この Unigram の語彙には
改行・タブ・生の空白を含むトークンが 1 つも無い（{@link check_upstream_shape} が毎回実測 —
1 つでもあれば行区切りが壊れて id が総ずれする）。

anima と違って **`normalizer` / `space` の表は持たない**:

- `tokenizer.json` の `normalizer` は `null`（正規化はモデル側ではなく上流の
  `normalize_text` が担う — NFKC は言語標準の `String.prototype.normalize` で足りる。
  ずれる可能性のある単一 cp だけ {@link nfkc_diff_table} が突合材料として焼く）
- pre_tokenizer は `Metaspace{replacement=▁, prepend_scheme=never, split=false}` で、
  空白の判定表が要る `WhitespaceSplit` を**通らない**（U+0020 を ▁ に置換するだけ）

## byte_fallback

`unk_id=0` だが `byte_fallback=true` なので、語彙に無い文字は `<0xNN>` の**バイト列**へ落ちる
（id = `byteBaseId` + バイト値。`byteBaseId` が 15 であることは毎回実測する）。TS 側は
「語彙に無ければ UTF-8 バイトへ展開」を実装すれば良く、unk は byte_fallback が無い場合の
経路として残るだけ。golden の c / d ケースがこの経路を固定する。

## 何を門にするか

emit の前に全て実測し、1 つでも外れたら**何も書かない**（ADR 0005 の fail loudly）:

- 上流 `tokenizer.json` の形（model 種別 / unk_id / byte_fallback / normalizer / pre_tokenizer /
  added_tokens のフラグ / 256 本のバイトトークンの id）
- 語彙に改行・タブ・生空白を含むトークンが無いこと（行区切りの前提）
- `bos_token_id` / `pad_token_id` が**チェックポイントの config と added_tokens で一致**すること
- **2 つの呼び出し経路が同じ id 列を出すこと** — `tokenizers.Tokenizer`（`export_irodori.py` の
  `build_cases` が使う）と `transformers.AutoTokenizer`（上流 `PretrainedTextTokenizer` が
  使う）。片方だけが正しくても golden は自己一致するので、ここで突き合わせる
- byte_fallback ケースが**実際に**バイト展開になっていること（期待バイト id 列との一致）
- 語彙内絵文字ケースが**実際に**語彙内であること（外れると b と c が同じ検査になる）
- `normalize_text` の各置換規則が golden のどこかで**最低 1 回発火**すること
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from export_irodori import DEFAULT_MODEL_DIR, DEFAULT_SOURCE_DIR, TOKENIZER_FILE, read_configs
from karume.paths import REPO_ROOT, SERIES_ROOT

ASSET_FILE = "tokenizer.json"
ENCODE_GOLDEN_FILE = "golden.encode.json"
NORMALIZE_GOLDEN_FILE = "golden.normalize.json"
NFKC_DIFF_FILE = "nfkc-diff.json"

#: パリティ用フィクスチャ（**git 管理**）。Deno 側
#: `packages/models/tests/irodori_text_test.ts` が読む。
DEFAULT_FIXTURE_PATH = (
    REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "irodori-text" / "parity.json"
)

#: Metaspace の置換文字（U+2581）。`split=false` / `prepend_scheme=never` なので、
#: pre-tokenize は「U+0020 をこれに置き換える」以外に何もしない。
METASPACE = "▁"

#: 資産のバイトトークンの id 起点（`<0x00>` の id）。**直書きせず毎回実測する**
#: （{@link check_upstream_shape}）— 上流が特殊トークンを 1 本足すだけで 256 本全部がずれる。
EXPECTED_BYTE_BASE_ID = 15

#: 上流の pre_tokenizer に期待する形。`split=false` は「▁ で断片に割らない」形で、
#: `prepend_scheme=never` は「先頭に ▁ を足さない」形（i ケースが固定する）。
EXPECTED_PRE_TOKENIZER = {
    "type": "Metaspace",
    "replacement": "▁",
    "prepend_scheme": "never",
    "split": False,
}

#: `batch_encode` 同値ケースの `max_length`。text 側は `max_text_len`（256）、caption 側は
#: `max_caption_len`（512）— どちらもチェックポイントの config が正本なので、値は
#: {@link build_batch_cases} が config から採る。ここに直書きはしない。
TEXT_MAX_LENGTH_KEY = "max_text_len"
CAPTION_MAX_LENGTH_KEY = "max_caption_len"

#: 語彙内であることを実測する絵文字（b ケース）。
IN_VOCAB_EMOJI = "\U0001f60a"

#: 語彙**外**であることを実測する絵文字（c / d ケース）。UTF-8 4 バイトなので
#: byte_fallback が 4 トークンに展開する。
OUT_OF_VOCAB_EMOJI = "\U0001f442"

#: `golden.encode.json` の固定ケース `(名前, 理由, 生テキスト)`。
#:
#: MUST: 目に見えない文字は必ず `\uXXXX` で書く（編集の途中で消えたり別物が混ざる）。
#: 網羅ではなく「素朴な移植が落ちる境界」を 1 件ずつ置く設計（anima の `PROMPT_CASES` と
#: 同じ規律）で、a〜l の分類はそのまま W3 のホスト実装の受け入れ条件になる。
ENCODE_CASES: tuple[tuple[str, str, str], ...] = (
    ("ja-plain", "a: 漢字かな混じり + 句読点", "今日は近くの店まで歩いて行きました。"),
    (
        "ja-formal",
        "a: 敬体の長め文（読点 2 つ）",
        "本日はお越しいただき、誠にありがとうございます。どうぞごゆっくりお過ごしください。",
    ),
    ("ja-question", "a: 疑問符・感嘆符（正規化で半角化される）", "本当にそれでいいの？！"),
    ("emoji-in-vocab", "b: 語彙内絵文字 U+1F60A", f"それは嬉しい{IN_VOCAB_EMOJI}ね。"),
    (
        "emoji-byte-fallback",
        "c: 語彙外絵文字 U+1F442 単体 → byte_fallback で 4 バイトへ展開",
        OUT_OF_VOCAB_EMOJI,
    ),
    (
        "emoji-byte-fallback-pair",
        "d: 語彙外絵文字 2 連 — fuse と byte_fallback の相互作用（実挙動をそのまま固定）",
        OUT_OF_VOCAB_EMOJI * 2,
    ),
    (
        "emoji-zwj",
        "e: ZWJ 絵文字列 U+1F62E U+200D U+1F4A8",
        "\U0001f62e\u200d\U0001f4a8",
    ),
    ("emoji-vs16", "f: 異体字セレクタ付き U+23F8 U+FE0F", "\u23f8\ufe0f"),
    (
        "added-token-in-text",
        "g: 追加語彙の文字列がユーザ入力に現れる（正規化より前に切り出される）",
        "設定で <|user|> と書くとどうなるのか。",
    ),
    ("with-spaces", "h: 半角空白 U+0020 → U+2581 置換", "hello world foo bar"),
    (
        "leading-space",
        "i: 先頭が空白（prepend_scheme=never — 足されるのではなく置換で ▁ が付く）",
        " leading space matters",
    ),
    ("latin-digits", "j: 英数字 Latin 混じり", "GPU 4 台で 2026 年に v1.5 を出す。"),
    (
        "combining-dakuten",
        "k: 結合文字 U+3099 と NFKC の相互作用（か + U+3099 → が）",
        "がっこいいじゃんけん",
    ),
    (
        "halfwidth-kana",
        "l: 半角カナ + 濁点（NFKC で全角合成へ変わる）",
        "ｶﾞﾝﾊﾞｬﾁ",
    ),
)

#: `batch_encode` 同値ケース `(名前, 理由, 生テキスト, max_length のキー)`。
#:
#: MUST: 1 件は**実際に truncation が発火する**長さにする（`max_text_len` = 256 に対し
#: body が 255 を超える）。発火しないケースだけだと「切り詰めの順序（BOS の前か後か）」が
#: golden に現れず、TS 側の取り違えが素通りする。
BATCH_CASES: tuple[tuple[str, str, str, str], ...] = (
    (
        "batch-short",
        "BOS 前置 + 右詰め pad（truncation は発火しない）",
        "今日は近くの店まで歩いて行きました。",
        TEXT_MAX_LENGTH_KEY,
    ),
    (
        "batch-caption",
        "caption 側の長さ（max_caption_len）で同じ手順を踏む",
        "若く元気な女性の声。カフェの店員のように、明るくハキハキとした少し高めのトーンで話している。",
        CAPTION_MAX_LENGTH_KEY,
    ),
)

#: 長文（`batch-truncated` の本文）。`export_irodori.py` の `LONG_PASSAGE_A` と同じ文体で、
#: **1 段落では足りない**（T=144）ので 3 段落ぶんの長さを持たせる — `max_text_len` = 256 の
#: truncation を実際に踏ませるのが目的。
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

#: `golden.normalize.json` の固定ケース `(名前, 理由, 生テキスト)`。
#:
#: `normalize_text` の段は SIMPLE_REPLACE_MAP → REGEX_REPLACE_MAP → `strip_outer_brackets`
#: → NFKC → `...`/`..` の `…` 化 の 5 段で、**順序そのものが観測できる**ケースを混ぜてある
#: （例: `①` は NFKC の前に削除されるので `1` にはならない / 全角 `；` は削除規則を
#: すり抜けて NFKC で `;` になる）。
NORMALIZE_CASES: tuple[tuple[str, str, str], ...] = (
    ("plain", "何も起きない素の文", "今日は近くの店まで歩いて行きました。"),
    ("tab", "SIMPLE: タブ削除", "前\t後ろ"),
    ("bracket-n-plain", "SIMPLE: [n] 削除", "改行[n]の指示"),
    ("bracket-n-escaped", r"SIMPLE: \[n\] 削除（エスケープ綴りの別規則）", r"改行\[n\]の指示"),
    (
        "ideographic-space",
        "SIMPLE: 全角空白 U+3000 削除（NFKC の前なので半角化されない）",
        "前　後ろ",
    ),
    ("question-bang", "SIMPLE: ？ → ? / ！ → !", "本当？そうだ！"),
    ("heart", "SIMPLE: ♥ → ♡", "だいすき♥"),
    ("circles", "SIMPLE: ● ◯ 〇 → ○", "●と◯と〇"),
    ("regex-symbols", "REGEX: ; ▼ ♀ ♂ 《》 ≪≫ ①〜⑥ の削除", "あ;い▼う♀え♂お《か》き≪く≫け①こ⑥"),
    (
        "regex-symbols-before-nfkc",
        "順序: ① は NFKC の前に消えるので `1` にならない / 全角 ； は残って NFKC で ; になる",
        "①番目；ここ",
    ),
    (
        "dashes",
        "REGEX: ダッシュ類（U+2010〜U+2015 / U+2212 / U+2500 など）の削除",
        "あ‐い―う−え─お",
    ),
    (
        "dash-modifier",
        "REGEX: U+02D7 / U+2043 / U+23AF / U+23E4 / U+2E3A / U+2E3B の削除",
        "あ˗い⁃う⎯え⏤お⸺か⸻き",
    ),
    ("wave-dash", "REGEX: ～(U+FF5E) / 〜(U+301C) → ー", "あ～い〜う"),
    ("ellipsis-run", "REGEX: …{3,} → ……（3 つ以上のときだけ）", "ええ………そうですか"),
    ("ellipsis-two", "REGEX: … 2 つは縮約されない（境界の下側）", "ええ……そうですか"),
    ("brackets-kagi", "括弧剥がし: 「」が全体を囲む", "「こんにちは」"),
    ("brackets-nested", "括弧剥がし: 多重（「」→（）の順に 2 回剥がれる）", "「（こんにちは）」"),
    ("brackets-double", "括弧剥がし: 『』", "『こんにちは』"),
    ("brackets-lenticular", "括弧剥がし: 【】", "【こんにちは】"),
    ("brackets-ascii", "括弧剥がし: ASCII ()", "(hello)"),
    ("brackets-not-enclosing", "括弧剥がし: 全体を囲んでいないので剥がれない", "「あ」と「い」"),
    ("brackets-asymmetric", "括弧剥がし: 対応しない組は剥がれない", "「あ）"),
    ("brackets-inner", "括弧剥がし: 中間に現れる括弧は残る", "（あ（い）う）"),
    ("nfkc-fullwidth", "NFKC: 全角英数 → 半角", "ＡＢＣ１２３ｄｅｆ"),
    ("nfkc-halfwidth-kana", "NFKC: 半角カナ + 濁点 → 全角合成", "ｶﾞﾝﾊﾞｬﾁ"),
    ("nfkc-square", "NFKC: 組文字 ㈱ ℃ ㌔ の分解", "㈱と℃と㌔"),
    ("nfkc-roman", "NFKC: ローマ数字 Ⅻ と合字 ﬁ と分数 ½", "Ⅻとﬁと½"),
    ("nfkc-combining", "NFKC: か + U+3099 → が（結合文字の合成）", "がっこいい"),
    ("nfkc-dakuten-alone", "NFKC: 単独の濁点 U+309B は空白 + 結合文字へ分解される", "あ゛い"),
    ("dots-three", "末尾: ... → …", "そうですね..."),
    ("dots-two", "末尾: .. → …", "そうですね.."),
    ("dots-four", "末尾: 順序（... が先に食うので .... は ….）", "そうですね...."),
    (
        "combined",
        "複合: 全角空白 + ？ + ダッシュ + 括弧 + NFKC + ... が 1 本の文で重なる",
        "「本当に　そうなの？―ＡＢＣ...」",
    ),
)


def _tokenizer_json(model_dir: Path) -> dict[str, Any]:
    path = model_dir / TOKENIZER_FILE
    if not path.is_file():
        raise SystemExit(f"tokenizer.json が見つからない: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_upstream_shape(raw: Mapping[str, Any]) -> dict[str, Any]:
    """上流 `tokenizer.json` が資産の畳み方の前提どおりであることを実測する。

    MUST: 主張のままにしない — ここが外れたまま emit すると、id 列だけが静かに別物になる
    形（行区切りの崩れ / byte_fallback の起点ずれ / 追加語彙の切り出し規則の違い）で
    TS 側へ流れ込む。
    """
    model = raw["model"]
    if model.get("type") != "Unigram":
        raise SystemExit(f"model.type が Unigram でない: {model.get('type')}")
    if int(model.get("unk_id", -1)) != 0:
        raise SystemExit(f"unk_id が 0 でない: {model.get('unk_id')}")
    if not bool(model.get("byte_fallback")):
        raise SystemExit("byte_fallback が false — 語彙外文字の扱いが別物になる")
    if raw.get("normalizer") is not None:
        raise SystemExit(f"normalizer が null でない: {raw['normalizer']} — 正規化表が要る")
    if raw.get("pre_tokenizer") != EXPECTED_PRE_TOKENIZER:
        raise SystemExit(
            f"pre_tokenizer が {EXPECTED_PRE_TOKENIZER} と違う: {raw.get('pre_tokenizer')}"
        )
    # 追加語彙のフラグは全て既定値のはず（normalized / lstrip / rstrip / single_word のどれかが
    # 立つと「正規化の前に leftmost-longest で切り出す」という写経の意味が変わる）。
    for entry in raw["added_tokens"]:
        flags = {key: entry[key] for key in ("lstrip", "rstrip", "single_word", "normalized")}
        if any(flags.values()):
            raise SystemExit(f"追加語彙 {entry['content']!r} のフラグが既定値でない: {flags}")

    vocab: list[list[Any]] = model["vocab"]
    for index, (token, _score) in enumerate(vocab):
        if "\n" in token or "\t" in token or " " in token:
            raise SystemExit(f"語彙 id {index} のトークン {token!r} が改行 / タブ / 空白を含む")
    byte_ids = {
        int(token[3:5], 16): index
        for index, (token, _score) in enumerate(vocab)
        if len(token) == 6 and token.startswith("<0x") and token.endswith(">")
    }
    if len(byte_ids) != 256:
        raise SystemExit(f"バイトトークンが {len(byte_ids)} 本（256 本のはず）")
    if any(index != EXPECTED_BYTE_BASE_ID + value for value, index in byte_ids.items()):
        raise SystemExit(f"バイトトークンの id が {EXPECTED_BYTE_BASE_ID} + バイト値で並んでいない")
    return {
        "vocab": len(vocab),
        "addedTokens": len(raw["added_tokens"]),
        "byteBaseId": EXPECTED_BYTE_BASE_ID,
    }


def build_asset(raw: Mapping[str, Any], text_config: Mapping[str, Any]) -> dict[str, Any]:
    """Deno 側が「引くだけ」で使える資産（anima の T5 資産と同じ畳み方）。

    MUST: `bosId` / `padId` は**チェックポイントの config** から採り、追加語彙の綴りと
    突き合わせる。トークナイザ側の `added_tokens` だけを見ると、config が別の id を
    指していた場合に「モデルが期待する BOS と違う id を前置する」形で静かに割れる。
    """
    vocab: list[list[Any]] = raw["model"]["vocab"]
    added = {entry["content"]: int(entry["id"]) for entry in raw["added_tokens"]}
    bos_id = int(text_config["bos_token_id"])
    pad_id = int(text_config["pad_token_id"])
    for label, token, want in (("bos", "<s>", bos_id), ("pad", "<pad>", pad_id)):
        if added.get(token) != want:
            raise SystemExit(
                f"{label}: config の id {want} と追加語彙 {token!r} の id {added.get(token)} が違う"
            )
    return {
        "vocabText": "\n".join(token for token, _score in vocab),
        "scores": [float(score) for _token, score in vocab],
        # `[文字列, id]` の対（TS 側 `parseAddedTokens` の形）。id 昇順に並べる。
        "addedTokens": [
            [content, value] for content, value in sorted(added.items(), key=lambda item: item[1])
        ],
        "bosId": bos_id,
        "padId": pad_id,
        "unkId": int(raw["model"]["unk_id"]),
        "byteBaseId": EXPECTED_BYTE_BASE_ID,
    }


def byte_fallback_ids(text: str) -> list[int]:
    """`text` を byte_fallback だけで表したときの id 列（c ケースの期待値の独立計算）。"""
    return [EXPECTED_BYTE_BASE_ID + byte for byte in text.encode("utf-8")]


class TokenizerPair:
    """id 列の**2 経路**（`tokenizers` 生 / `transformers` 経由）を束ねて突き合わせる。

    `export_irodori.py` の `build_cases` は前者を、上流の `PretrainedTextTokenizer` は
    後者を使う。golden はどちらか一方から採るしかないので、**両方が同じ列を出すこと**を
    ケースごとに実測してから採る（片方だけ壊れても golden は自己一致してしまう）。
    """

    def __init__(self, model_dir: Path) -> None:
        from irodori_tts.tokenizer import PretrainedTextTokenizer
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer

        self.raw = Tokenizer.from_file(str(model_dir / TOKENIZER_FILE))
        self.hf = AutoTokenizer.from_pretrained(str((model_dir / TOKENIZER_FILE).parent))
        self.wrapper = PretrainedTextTokenizer(self.hf, add_bos=True)

    def encode(self, text: str, where: str) -> list[int]:
        """特殊トークン無しの素の id 列（上流 `PretrainedTextTokenizer.encode` と同じ呼び方）。"""
        raw_ids = list(self.raw.encode(text, add_special_tokens=False).ids)
        hf_ids = list(self.hf.encode(text, add_special_tokens=False))
        if raw_ids != hf_ids:
            raise SystemExit(
                f"{where}: tokenizers と transformers の id 列が違う\n"
                f"  tokenizers={raw_ids}\n  transformers={hf_ids}"
            )
        return raw_ids


def build_encode_cases(pair: TokenizerPair, normalize_text: Any) -> list[dict[str, Any]]:
    """`{raw, normalized, ids}` の固定ケース。

    `normalized` は `normalize_text` を掛けただけの文字列で、**`strip()` は掛けない** —
    strip は上流のパイプライン段（`inference_runtime._synthesize`）が掛けるもので
    トークナイザの契約ではなく、掛けると先頭空白ケース（i）が検査対象を失う。
    `ids` は **BOS 前置前**の素の列（前置は {@link build_batch_cases} 側の契約）。
    """
    vocab = set(pair.raw.get_vocab())
    if IN_VOCAB_EMOJI not in vocab:
        raise SystemExit(f"{IN_VOCAB_EMOJI!r} が語彙に無い — b ケースが c ケースと同じ検査になる")
    if OUT_OF_VOCAB_EMOJI in vocab:
        raise SystemExit(
            f"{OUT_OF_VOCAB_EMOJI!r} が語彙にある — c / d ケースが byte_fallback を踏まない"
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
    # byte_fallback が**実際に**発火していることの実測（恒真にしない — 期待バイト列は
    # 語彙を見ずに UTF-8 から独立に計算する）。
    want = byte_fallback_ids(OUT_OF_VOCAB_EMOJI)
    got = by_name["emoji-byte-fallback"]["ids"]
    if got != want:
        raise SystemExit(f"c ケースが byte_fallback になっていない: {got} != {want}")
    return cases


def build_batch_cases(
    pair: TokenizerPair, normalize_text: Any, model_config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """上流 `batch_encode` の**最終形**（BOS 前置 + 右詰め pad + マスク）。

    MUST: 手順を写さず上流の `PretrainedTextTokenizer.batch_encode` を**呼ぶ** — 切り詰めの
    順序（body を `max_length-1` で切ってから BOS を足す）は写経が最も外しやすい点で、
    写した式は上流が変わっても黙って古いまま通る。
    """
    specs = [
        *BATCH_CASES,
        (
            "batch-truncated",
            "truncation が実際に発火する長さ（body > max_length-1）",
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
            raise SystemExit(f"{name}: マスクの有効長 {used} が BOS + body の想定と違う")
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
        raise SystemExit("truncation が発火するケースが 1 件も無い（切り詰めの順序が固定されない）")
    return cases


def build_normalize_cases(normalize_text: Any) -> list[dict[str, Any]]:
    """`normalize_text` の `{raw, normalized}` golden（各置換規則を最低 1 回発火させる）。"""
    return [
        {"name": name, "why": why, "raw": text, "normalized": normalize_text(text)}
        for name, why, text in NORMALIZE_CASES
    ]


def check_rule_coverage(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """上流の置換規則が golden のどこかで**発火している**ことを実測する。

    MUST: 恒真にしない — 「規則の綴りがケースの生テキストに含まれる」だけでなく、
    その規則が**実際に出力を変えている**（発火前後で文字列が違う）ことまで見る。
    上流が規則を足したらここが落ちて、golden の穴に気づける。
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
        raise SystemExit(f"golden がどの規則も発火させていない: {missing}")

    stripped = sum(1 for case in cases if case["why"].startswith("括弧剥がし"))
    changed_by_nfkc = sum(
        1 for case in cases if unicodedata.normalize("NFKC", case["raw"]) != case["raw"]
    )
    dots = sum(1 for case in cases if ".." in case["raw"])
    if stripped < 3 or changed_by_nfkc < 3 or dots < 2:
        raise SystemExit(
            f"golden の被覆が薄い（括弧 {stripped} / NFKC {changed_by_nfkc} / ドット {dots}）"
        )
    return {"rules": len(fired), "brackets": stripped, "nfkc": changed_by_nfkc, "dots": dots}


#: サロゲート（Python の str に載らない）を除く全コードポイント。
ALL_CODEPOINTS = [cp for cp in range(0x110000) if not (0xD800 <= cp <= 0xDFFF)]


def nfkc_diff_table() -> dict[str, str]:
    """NFKC が**恒等でない**単一コードポイントの写像表 `{cp（10 進の文字列）: 正規化後}`。

    TS 側は NFKC を自前実装せず `String.prototype.normalize("NFKC")` に委ねる（`normalize_text`
    の正本が Python の `unicodedata` なので、両者の Unicode 版がずれた瞬間に**例外も警告も
    出さずに**正規化結果が変わる）。この表はその突合材料で、TS 側のテストが全エントリを
    自分の `normalize` に通して一致を確かめる。

    NOTE: 単一 cp の表なので**複数 cp にまたがる差**（正準順序付け・合成）は捕まえられない。
    そちらは `golden.normalize.json` の結合文字ケース（か + U+3099 など）が受け持つ。
    """
    return {
        str(cp): normalized
        for cp in ALL_CODEPOINTS
        if (normalized := unicodedata.normalize("NFKC", chr(cp))) != chr(cp)
    }


def lattice_texts(
    encode_cases: Sequence[Mapping[str, Any]], batch_cases: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Viterbi の格子が実際に見る文字列（golden の `normalized` を ▁ 置換したもの）。

    追加語彙の切り出しで断片に割れるが、**割った断片の部分文字列は元の文字列の部分文字列**
    なので、割る前の全体を渡せば語彙の抽出には十分（超集合になるだけで格子は変わらない）。
    """
    return [case["normalized"].replace(" ", METASPACE) for case in (*encode_cases, *batch_cases)]


def vocab_subset(
    vocab: Sequence[Sequence[Any]], texts: Sequence[str], added: Mapping[str, int]
) -> list[list[Any]]:
    """フィクスチャに載せる語彙の**部分集合** `[トークン, id, スコア]`（id 昇順）。

    載せるのは ①texts のいずれかの部分文字列に一致するトークン全件（= 格子が引きうる全て）
    ②バイトトークン 256 本（byte_fallback の id 連番そのものを TS 側が検査する）③追加語彙。
    102,400 本の語彙を丸ごと commit しないための分離で、格子が同じであることは
    {@link check_subset_covers_lattice} が**逆向き**に実測する。
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
    """部分集合でも Viterbi の結果が語彙全体と一致することを実測する。

    格子は「その断片の部分文字列に一致する語彙エントリ」しか引かないので、**一致集合が同じ
    なら格子も同じ**。ここは語彙**全体**を走査して「texts の部分文字列であるのに部分集合に
    無いトークン」を数える — 抽出（部分文字列 → 語彙引き）とは向きが逆なので、抽出側の
    取りこぼしをそのまま写す恒真検査にならない。
    """
    present = {row[0] for row in subset}
    missing = [
        token
        for token, _score in vocab
        if token not in present and any(token in text for text in texts)
    ]
    if missing:
        raise SystemExit(
            f"部分集合が格子の一致集合を覆えていない: {missing[:8]}（計 {len(missing)} 件）"
        )
    return sum(1 for token, _score in vocab if any(token in text for text in texts))


def build_parity_fixture(
    asset: Mapping[str, Any],
    raw_vocab: Sequence[Sequence[Any]],
    encode_cases: Sequence[Mapping[str, Any]],
    batch_cases: Sequence[Mapping[str, Any]],
    normalize_cases: Sequence[Mapping[str, Any]],
    diff: Mapping[str, str],
) -> dict[str, Any]:
    """TS 側パリティ門のフィクスチャ（**git 管理**・実資産無しで走る形）。

    MUST: `minScore` / `maxTokenLength` は語彙**全体**から採る。未知ノードのスコアと前方一致の
    探索幅は全体で決まるので、部分集合から導くと別の分割になる（anima の T5 と同じ罠）。
    """
    texts = lattice_texts(encode_cases, batch_cases)
    subset = vocab_subset(raw_vocab, texts, dict(asset["addedTokens"]))
    matched = check_subset_covers_lattice(raw_vocab, texts, subset)
    return {
        "_doc": [
            "Irodori-TTS v4 テキスト層のパリティ用フィクスチャ",
            "（生成: tools/exporter/irodori_tokenizer.py）。",
            "正本は上流 normalize_text と tokenizers / transformers で、id 列はそこから採った",
            "実測値。語彙は全ケースの再現に要る**部分集合**（102,400 本を commit しないため）",
            "だが、minScore / maxTokenLength は語彙**全体**の値。NFKC 差分表は全体を載せる",
            "（TS の normalize('NFKC') と Unicode 版がずれた瞬間に正規化が静かに変わるため）。",
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
        },
        "encode": {"cases": list(encode_cases), "batch": list(batch_cases)},
        "normalize": {"cases": list(normalize_cases)},
        "nfkcDiff": dict(diff),
        "stats": {"vocabTotal": len(raw_vocab), "latticeMatches": matched},
    }


def emit(model_dir: Path, source_dir: Path, out_dir: Path, fixture_path: Path) -> dict[str, Any]:
    """資産・golden・フィクスチャを書き、要約を返す（検証に落ちたら 1 バイトも書かない）。"""
    if not (source_dir / "irodori_tts" / "text_normalization.py").is_file():
        raise SystemExit(
            f"モデル実装が見つからない: {source_dir}"
            "（`git clone https://github.com/Aratako/Irodori-TTS` の展開先を"
            " --source-dir に指定する）"
        )
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from irodori_tts.text_normalization import normalize_text

    text_config, model_config = read_configs(model_dir)
    raw = _tokenizer_json(model_dir)
    shape = check_upstream_shape(raw)
    asset = build_asset(raw, text_config)

    pair = TokenizerPair(model_dir)
    encode_cases = build_encode_cases(pair, normalize_text)
    batch_cases = build_batch_cases(pair, normalize_text, model_config)
    normalize_cases = build_normalize_cases(normalize_text)
    coverage = check_rule_coverage(normalize_cases)
    diff = nfkc_diff_table()
    fixture = build_parity_fixture(
        asset, raw["model"]["vocab"], encode_cases, batch_cases, normalize_cases, diff
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for filename, payload, indent in (
        (ASSET_FILE, asset, None),
        (ENCODE_GOLDEN_FILE, {"cases": encode_cases, "batch": batch_cases}, 1),
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
        "normalizeCases": len(normalize_cases),
        "ruleCoverage": coverage,
        "nfkcDiffEntries": len(diff),
    }


def default_out_dir(model_dir: Path) -> Path:
    """既定の置き場（`outputs/series/irodori-<実重みのディレクトリ名>/tokenizer/`）。"""
    return SERIES_ROOT / f"irodori-{model_dir.name}" / "tokenizer"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
