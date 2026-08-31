"""Gemma 系（gemma4 / EmbeddingGemma）の SPM-BPE トークナイザを compile-to-asset する
（ADR 0084 決定 1・複数 family で共有する補助なので `_shared` — ADR 0065 決定 2）。

持つのは 3 つ:

  ① **compile** — 上流 `tokenizer.json`（32.2MB）を読み、Deno 側が引くだけで済む小さな
     schema へ畳む。**未知の構成は compile 時に落とす**（{@link assert_supported}）—
     実行時に散らさないのがこの形を採る理由そのもの（ADR 0084 検討した代替案 b）。
     畳んだ先が持つのは「id 順の vocab 行 / merges の id 対 / byteIds 256 本 /
     added tokens / special id 集合」だけで、Qwen2 で要った焼き表（`code-ranges` /
     `caseFold` / `nfcSegments`）は 1 つも要らない — Gemma の normalizer は
     `Replace(" " → "▁")` 1 本、pre_tokenizer は `Split(" ", MergedWithPrevious)` だけで、
     Unicode 分類にも NFC にも触らないため（ADR 0084 Context 2）。

  ② **parity fixture** — TS 実装の期待値。**採取は `tokenizers.Tokenizer` を独立に呼ぶ経路
     だけ**で、①の畳み込みも Python 側の前処理 helper も一切通さない（ADR 0084 決定 7・
     ADR 0048 の追記 = 前処理を TS と Python で共有すると parity が恒真化する）。

  ③ **資産の部分集合** — フィクスチャは git 管理なので、262,144 行の語彙と 514,906 本の
     merges を丸ごとは載せられない。ケース本文の**部分文字列**に当たるものだけへ絞る
     （{@link subset_keys}）。絞り方が誤っていても危険側には倒れない: 期待値は常に上流
     （full 資産）から採るので、部分集合が足りなければ TS 側の突合が**落ちる**。

MUST: 家族固有の綴り（入力の置き場・系列名・ケース）は**呼び手（family recipe）が持つ**。
ここが知っているのは Gemma の tokenizer 構成だけで、gemma4 と EmbeddingGemma のどちらでも
同じ関数がそのまま通る（ADR 0084 決定 6 の「1 実装 2 資産」を実際に成り立たせる分割）。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _shared.paths import REPO_ROOT

#: 資産 schema の版（Deno 側が知らない版を黙って読まないための欄）。
ASSET_FORMAT = "karume-gemma-tokenizer/1"

#: SentencePiece の空白置換文字（U+2581）。normalizer と decoder の両方が使う。
METASPACE = "▁"

#: 受理する上流構成（**exact-match** — 1 欄でも違えば別のトークナイザ）。
EXPECTED_NORMALIZER = {"type": "Replace", "pattern": {"String": " "}, "content": METASPACE}
EXPECTED_PRE_TOKENIZER = {
    "type": "Split",
    "pattern": {"String": " "},
    "behavior": "MergedWithPrevious",
    "invert": False,
}
EXPECTED_DECODER = {
    "type": "Sequence",
    "decoders": [
        {"type": "Replace", "pattern": {"String": METASPACE}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
    ],
}

#: `spec` 欄が採る綴り（Deno 側が同じ文字列で exact-match する）。
SPEC_NORMALIZER = "replace-space-with-metaspace"
SPEC_PRE_TOKENIZER = "split-space-merged-with-previous"
SPEC_DECODER = "metaspace-byte-fallback-fuse"

#: post_processor の 2 形（実資産の実測 — gemma4 は素通し / EmbeddingGemma は bos…eos）。
POST_NONE = "none"
POST_BOS_EOS = "bos-eos"

#: 特殊トークンの綴り（両資産共通）。
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


def byte_token(byte: int) -> str:
    """byte_fallback 語彙の綴り（`tokenizers` の `format!("<{:#04X}>")` と同じ）。"""
    return f"<0x{byte:02X}>"


class UnsupportedTokenizerError(ValueError):
    """上流 `tokenizer.json` が受理集合の外（compile 時に落とす）。"""


@dataclass(frozen=True)
class CompiledTokenizer:
    """畳んだ結果（JSON へ落とす手前の形）。"""

    #: id 順の語彙（行番号 = id）。
    vocab: list[str]
    #: merges を `(左 id, 右 id)` で持つ。並び順が rank そのもの。
    merges: list[tuple[int, int]]
    #: byte 0..255 に対応する id（**256 本明示** — ADR 0084 決定 1 の MUST）。
    byte_ids: list[int]
    #: 追加語彙 `(綴り, id)`（正規化の手前で切り出される）。
    added_tokens: list[tuple[str, int]]
    #: `special: true` の追加語彙の id（decode の skip 対象）。
    special_ids: list[int]
    unk_id: int
    bos_id: int
    eos_id: int
    #: post_processor の形（{@link POST_NONE} / {@link POST_BOS_EOS}）。
    post_processor: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UnsupportedTokenizerError(message)


def assert_supported(raw: Mapping[str, Any]) -> str:
    """上流 `tokenizer.json` が受理集合に入っていることを検め、post_processor の形を返す。

    MUST: **exact-match で見る**（欄の有無だけを見ない）。normalizer が 1 文字違えば
    metaspace の入り方が変わり、pre_tokenizer が変われば BPE に渡る単位が変わる —
    どちらも例外にならず id 列だけが静かに別物になる。
    """
    model = raw.get("model")
    _require(isinstance(model, Mapping), "model 欄が無い")
    _require(model.get("type") == "BPE", f"model.type が BPE でない（{model.get('type')!r}）")
    _require(model.get("dropout") is None, "model.dropout が null でない（学習時の枝刈り）")
    _require(
        model.get("continuing_subword_prefix") is None,
        "model.continuing_subword_prefix が null でない",
    )
    _require(model.get("end_of_word_suffix") is None, "model.end_of_word_suffix が null でない")
    _require(model.get("unk_token") == UNK_TOKEN, f"model.unk_token が {UNK_TOKEN!r} でない")
    _require(model.get("byte_fallback") is True, "model.byte_fallback が true でない")
    _require(model.get("fuse_unk") is True, "model.fuse_unk が true でない")
    _require(model.get("ignore_merges") is False, "model.ignore_merges が false でない")
    _require(raw.get("truncation") is None, "truncation が null でない")
    _require(raw.get("padding") is None, "padding が null でない")
    _require(
        raw.get("normalizer") == EXPECTED_NORMALIZER,
        f"normalizer が {EXPECTED_NORMALIZER} でない（{raw.get('normalizer')!r}）",
    )
    _require(
        raw.get("pre_tokenizer") == EXPECTED_PRE_TOKENIZER,
        f"pre_tokenizer が {EXPECTED_PRE_TOKENIZER} でない（{raw.get('pre_tokenizer')!r}）",
    )
    _require(
        raw.get("decoder") == EXPECTED_DECODER,
        f"decoder が {EXPECTED_DECODER} でない（{raw.get('decoder')!r}）",
    )
    return _post_processor_kind(raw.get("post_processor"))


def _post_processor_kind(post: Any) -> str:
    """post_processor を 2 形のどちらかへ判定する（それ以外は落とす）。

    見るのは `single` だけ — 対（`pair`）を組む経路はこの移植の射程外で、そちらの綴りが
    違っても single の振る舞いは変わらないため（受理を無闇に狭めない）。
    """
    _require(isinstance(post, Mapping), "post_processor が無い")
    _require(
        post.get("type") == "TemplateProcessing",
        f"post_processor.type が TemplateProcessing でない（{post.get('type')!r}）",
    )
    sequence_a = {"Sequence": {"id": "A", "type_id": 0}}
    single = post.get("single")
    if single == [sequence_a]:
        return POST_NONE
    wrapped = [
        {"SpecialToken": {"id": BOS_TOKEN, "type_id": 0}},
        sequence_a,
        {"SpecialToken": {"id": EOS_TOKEN, "type_id": 0}},
    ]
    if single == wrapped:
        return POST_BOS_EOS
    raise UnsupportedTokenizerError(f"post_processor.single が未知の形（{single!r}）")


def compile_tokenizer(raw: Mapping[str, Any]) -> CompiledTokenizer:
    """上流 `tokenizer.json` を畳む（受理集合の検査つき）。"""
    post_processor = assert_supported(raw)
    model = raw["model"]
    vocab_map: Mapping[str, int] = model["vocab"]
    _require(len(set(vocab_map.values())) == len(vocab_map), "語彙の id に重複がある")
    vocab = [""] * len(vocab_map)
    for token, token_id in vocab_map.items():
        _require(
            isinstance(token_id, int) and 0 <= token_id < len(vocab_map),
            f"語彙 {token!r} の id {token_id!r} が 0..{len(vocab_map) - 1} の外",
        )
        vocab[token_id] = token
    # MUST: byte 256 本を**明示的に引く**。実資産では `<0x00>`..`<0xFF>` が連番だが、それは
    # この資産の実測事実であって schema の保証ではない（ADR 0084 決定 1）。
    byte_ids: list[int] = []
    for byte in range(256):
        spelling = byte_token(byte)
        if spelling not in vocab_map:
            raise UnsupportedTokenizerError(f"byte_fallback 語彙 {spelling!r} が語彙に無い")
        byte_ids.append(vocab_map[spelling])
    _require(len(set(byte_ids)) == 256, "byte_fallback 語彙の id に重複がある")
    # decode 側は「id が byteIds に入っているか」で byte run を判定する。上流の
    # `ByteFallback` デコーダは**綴りの形**で判定するので、形が一致する語彙が 256 本ちょうど
    # であることを compile 時に確かめておかないと 2 つの判定が食い違いうる。
    shaped = [t for t in vocab_map if len(t) == 6 and t.startswith("<0x") and t.endswith(">")]
    _require(
        len(shaped) == 256,
        f"`<0xHH>` の形の語彙が 256 本でない（{len(shaped)} 本 — decode の byte 判定が割れる）",
    )

    merges: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for entry in model["merges"]:
        # 旧 format の `"左 右"` 文字列は受けない（Gemma の語彙は空白を含まないので一見
        # 分割できるが、綴りに改行を含む語彙がある以上「行と対」の対応が別物になる）。
        _require(
            isinstance(entry, list) and len(entry) == 2,
            f"merge の要素が [左, 右] でない（{entry!r}）",
        )
        left, right = entry
        for side in (left, right, left + right):
            if side not in vocab_map:
                raise UnsupportedTokenizerError(f"merge の {side!r} が語彙に無い")
        pair = (vocab_map[left], vocab_map[right])
        # MUST: 後勝ちを許さない。上流は HashMap なので重複対は静かに片方だけが残り、
        # 分割規則だけが別物になる（`asset-gates.ts` の `setUnique` と同じ理由）。
        _require(pair not in seen_pairs, f"merge の対 {(left, right)!r} が重複している")
        seen_pairs.add(pair)
        merges.append(pair)

    added_tokens: list[tuple[str, int]] = []
    special_ids: list[int] = []
    for entry in raw.get("added_tokens", []):
        for flag in ("single_word", "lstrip", "rstrip", "normalized"):
            _require(
                entry.get(flag) is False,
                f"追加語彙 {entry.get('content')!r} の {flag} が false でない"
                "（切り出し規則が別物になる）",
            )
        content = entry["content"]
        added_tokens.append((content, entry["id"]))
        if entry.get("special") is True:
            special_ids.append(entry["id"])
    _require(
        len({token for token, _ in added_tokens}) == len(added_tokens),
        "追加語彙の綴りに重複がある",
    )

    ids_by_token = dict(added_tokens)
    for name in (UNK_TOKEN, BOS_TOKEN, EOS_TOKEN):
        _require(name in vocab_map, f"語彙に {name!r} が無い")
        if name in ids_by_token:
            _require(
                ids_by_token[name] == vocab_map[name],
                f"{name!r} の追加語彙 id と語彙 id が食い違う",
            )
    return CompiledTokenizer(
        vocab=vocab,
        merges=merges,
        byte_ids=byte_ids,
        added_tokens=added_tokens,
        special_ids=special_ids,
        unk_id=vocab_map[UNK_TOKEN],
        bos_id=vocab_map[BOS_TOKEN],
        eos_id=vocab_map[EOS_TOKEN],
        post_processor=post_processor,
    )


def spec_of(compiled: CompiledTokenizer) -> dict[str, str]:
    """Deno 側が exact-match する構成の宣言（畳んだ資産は上流 JSON を持たないため）。"""
    return {
        "normalizer": SPEC_NORMALIZER,
        "preTokenizer": SPEC_PRE_TOKENIZER,
        "decoder": SPEC_DECODER,
        "postProcessor": compiled.post_processor,
    }


def asset_payload(compiled: CompiledTokenizer, *, source: Mapping[str, Any]) -> dict[str, Any]:
    """配布形の JSON（`mergesText` は行番号 = rank・`vocab` は行番号 = id）。

    merges を綴りでなく **id 対**で持つのは、Gemma の語彙が改行を含む（実測 31 本）ため —
    Qwen2 の byte-level 語彙のように「`左 右`」の行へは畳めない。
    """
    return {
        "format": ASSET_FORMAT,
        "source": dict(source),
        "spec": spec_of(compiled),
        "vocab": compiled.vocab,
        "mergesText": "\n".join(f"{left} {right}" for left, right in compiled.merges),
        "mergesCount": len(compiled.merges),
        "byteIds": compiled.byte_ids,
        "addedTokens": [[token, token_id] for token, token_id in compiled.added_tokens],
        "specialIds": compiled.special_ids,
        "unkId": compiled.unk_id,
        "bosId": compiled.bos_id,
        "eosId": compiled.eos_id,
    }


def file_digest(path: Path) -> dict[str, Any]:
    """`path` の `{bytes, sha256}`（出所記録）。"""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def write_json(path: Path, payload: Mapping[str, Any]) -> int:
    """UTF-8 の JSON を書いてバイト数を返す（親ディレクトリは作る）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


# ---------------------------------------------------------------------------
# parity fixture（期待値は上流の `tokenizers` から独立に採る）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodeCase:
    """符号化ケース（`why` は「素朴な移植が落ちる境界」を 1 行で述べる）。"""

    name: str
    why: str
    text: str


@dataclass(frozen=True)
class DecodeCase:
    """復号ケース（id 列を直接叩く — 符号化では作れない byte run を踏むため）。"""

    name: str
    why: str
    ids: list[int]


#: 両資産で共通に流す符号化ケース。**素朴な移植が落ちる境界を 1 件ずつ**持たせる設計で、
#: 目に見えない文字は必ず \uXXXX で書く（編集の途中で消えたり別物が混ざるため）。
#: 同じ電池を 2 資産へ流すことが「1 実装 2 資産」（ADR 0084 決定 6）の実証そのもの。
SHARED_ENCODE_CASES: tuple[EncodeCase, ...] = (
    EncodeCase("english", "普通の英文 — 空白が metaspace へ落ちる基本形", "The quick brown fox."),
    EncodeCase(
        "english-long",
        "空白があっても pre_tokenizer は正規化後に切れ目を見つけられない（1 pre-token が全長）"
        " — merge の rank / 位置の同点が多発する長さ",
        "Machine learning models trained on large corpora can generate text that is "
        "surprisingly coherent, but they still fail at arithmetic and long-horizon planning.",
    ),
    EncodeCase(
        "japanese-long",
        "空白の無い長文 — 1 pre-token が入力全長になり merge queue が要る（ADR 0084 決定 3）",
        "むかしむかし、あるところにおじいさんとおばあさんが住んでいました。"
        "おじいさんは山へしばかりに、おばあさんは川へせんたくに行きました。"
        "おばあさんが川でせんたくをしていると、大きな桃がどんぶらこどんぶらこと"
        "流れてきました。おばあさんはその桃を家に持ち帰り、おじいさんと二人で"
        "切ってみると、中から元気な男の子が飛び出しました。二人はたいそう喜んで、"
        "その子を桃太郎と名づけ、たいせつに育てることにしたのでした。",
    ),
    EncodeCase(
        "cyrillic",
        "非ラテン・非 CJK（語彙内トークンと byte fallback の境目が別の位置に来る）",
        "Большая часть текста написана на русском языке.",
    ),
    EncodeCase("leading-space", "先頭の空白（metaspace が先頭に来る）", " leading"),
    EncodeCase("trailing-space", "末尾の空白（merge の右端が metaspace）", "trailing "),
    EncodeCase("double-space", "空白の連続（▁▁ 系の語彙へ畳まれる）", "a  b   c"),
    EncodeCase(
        "whitespace-run",
        "空白 30 連 — 長い metaspace 語彙（merges 表の先頭付近の rank）を踏む",
        "x" + " " * 30 + "y",
    ),
    EncodeCase(
        "tab-newline",
        "normalizer が触らない空白類（\\t / \\n は metaspace にならない）",
        "a\tb\nc\r\nd",
    ),
    EncodeCase(
        "newline-run",
        "改行 30 連 — 語彙に改行を含む綴りがある（行区切り資産が使えない証）",
        "\n" * 30,
    ),
    EncodeCase("digits", "数字列 — SPM 系は 1 桁ずつ切れる語彙を持つ", "1girl 2024 3.14159 100%"),
    EncodeCase(
        "url",
        "空白の無い長い ASCII（URL / base64 も 1 pre-token 全長になる）",
        "https://example.com/a/b/c?q=1&r=2#frag",
    ),
    EncodeCase("mixed", "日英混在 — 語彙の切り替わり", "1girl, 猫耳, ロングヘア, masterpiece"),
    EncodeCase(
        "emoji",
        "非 BMP + 地域表示子 + 異体字セレクタ（byte_fallback を踏む）",
        "a cat \U0001f408 and a flag \U0001f1ef\U0001f1f5 and ❤️ love",
    ),
    EncodeCase(
        "zwj",
        "ZWJ 連結（1 書記素が複数コードポイント — 記号は**コードポイント**単位で始まる）",
        "family \U0001f468‍\U0001f469‍\U0001f466 photo",
    ),
    EncodeCase(
        "combining",
        "結合文字 — NFC を掛けない（掛けると id 列が変わる）",
        "Á vs Á and Ấ",
    ),
    EncodeCase("thai", "結合文字を持つタイ語（語彙内と byte fallback が混ざる）", "กำลังทดสอบ"),
    EncodeCase(
        "control-bytes",
        "制御文字 — 語彙に無く byte_fallback へ落ちる（複数バイト run も含む）",
        "a\x00b\x1fcd",
    ),
    EncodeCase(
        "metaspace-literal",
        "入力に U+2581 が直接ある（正規化の出力と同じ文字が最初から入っている形）",
        "raw ▁ metaspace ▁▁ run",
    ),
    EncodeCase("empty", "空文字（記号が 0 個 — 走査の初期状態）", ""),
    EncodeCase("single-space", "空白 1 文字だけ（正規化後は metaspace 1 文字）", " "),
)


def subset_keys(texts: Iterable[str], *, max_token_chars: int) -> set[str]:
    """ケース本文から、語彙・merges の**部分集合**を切り出すための鍵集合を作る。

    BPE の記号はどの時点でも「正規化後の文字列の連続した部分列」なので、`max_token_chars`
    文字までの部分文字列を全部集めれば、走査中に現れうる記号と対を必ず覆う（byte_fallback
    の記号だけは部分列にならないが、実資産では byte 語彙を含む merge が 1 本も無いことを
    {@link assert_no_byte_merges} が確かめる）。

    NOTE: ここで使う正規化は「空白 → metaspace」だけ。追加語彙の切り出しを通す前の全文を
    使うのは**わざと**で、切り出し後の各断片は必ずこの文字列の部分列になる（Replace は
    文字ごとの置換なので）— 覆いを広い側へ倒しておけば、絞り方の誤りが「フィクスチャが
    足りず TS の突合が落ちる」形にしか出ない。
    """
    keys: set[str] = set()
    for text in texts:
        chars = list(text.replace(" ", METASPACE))
        for start in range(len(chars)):
            piece = ""
            for length in range(1, min(max_token_chars, len(chars) - start) + 1):
                piece += chars[start + length - 1]
                keys.add(piece)
    return keys


def assert_no_byte_merges(compiled: CompiledTokenizer) -> None:
    """byte_fallback 語彙を含む merge が無いことを確かめる（部分集合の覆いの前提）。"""
    byte_id_set = set(compiled.byte_ids)
    for left, right in compiled.merges:
        if left in byte_id_set or right in byte_id_set:
            raise UnsupportedTokenizerError(
                "byte_fallback 語彙を含む merge がある — フィクスチャの部分集合が覆えない",
            )


def asset_subset(
    compiled: CompiledTokenizer,
    *,
    texts: Sequence[str],
    required_ids: Iterable[int],
) -> dict[str, Any]:
    """フィクスチャへ載せる語彙・merges の部分集合を作る。

    NOTE: 公開名なのは chat のフィクスチャ（`gemma4/chat.py`）も同じ切り方を使うため —
    絞り方を 2 実装持つと、片方だけが「足りない部分集合」を作っても気づけない。
    """
    assert_no_byte_merges(compiled)
    max_token_chars = max(len(list(token)) for token in compiled.vocab)
    keys = subset_keys(texts, max_token_chars=max_token_chars)
    ids_by_token = {token: index for index, token in enumerate(compiled.vocab)}

    required_kept = set(required_ids)
    kept: set[int] = set(compiled.byte_ids)
    kept.add(compiled.unk_id)
    kept.add(compiled.bos_id)
    kept.add(compiled.eos_id)
    for token in keys:
        if token in ids_by_token:
            kept.add(ids_by_token[token])
    for token_id in required_kept:
        if token_id < len(compiled.vocab):
            kept.add(token_id)

    merges: list[list[int]] = []
    for rank, (left, right) in enumerate(compiled.merges):
        joined = compiled.vocab[left] + compiled.vocab[right]
        if joined in keys:
            merges.append([left, right, rank])
            kept.update((left, right, ids_by_token[joined]))

    # 追加語彙は「ケース本文に現れうるもの」だけへ絞る。落とした綴りが本文に無い以上、
    # leftmost-longest の切り出し結果は変わらない。
    added = [
        [token, token_id]
        for token, token_id in compiled.added_tokens
        if token in keys or token_id in required_kept
    ]
    special = set(compiled.special_ids)
    return {
        "vocab": sorted([token_id, compiled.vocab[token_id]] for token_id in kept),
        "merges": merges,
        "byteIds": compiled.byte_ids,
        "addedTokens": added,
        "specialIds": sorted(token_id for _, token_id in added if token_id in special),
        "unkId": compiled.unk_id,
        "bosId": compiled.bos_id,
        "eosId": compiled.eos_id,
    }


def build_fixture(
    *,
    tokenizer_json: Path,
    compiled: CompiledTokenizer,
    encode_cases: Sequence[EncodeCase],
    decode_cases: Sequence[DecodeCase],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """TS 実装の期待値を上流から独立に採る（ADR 0084 決定 7）。

    MUST: 期待値の経路に**このモジュールの畳み込みを 1 行も通さない** — 通すと「畳み方が
    間違っていても両方が同じだけ間違う」形になり、突合が恒真化する（ADR 0048 追記の実例）。
    """
    from tokenizers import Tokenizer  # 遅延 import（compile だけなら不要な依存）

    reference = Tokenizer.from_file(str(tokenizer_json))

    encode: list[dict[str, Any]] = []
    for case in encode_cases:
        encode.append(
            {
                "name": case.name,
                "why": case.why,
                "text": case.text,
                "ids": reference.encode(case.text, add_special_tokens=False).ids,
                "idsWithSpecials": reference.encode(case.text, add_special_tokens=True).ids,
            }
        )
    # 往復（符号化の出力をそのまま復号へ）は自動で足す — 手で書き写すと片方だけ古くなる。
    all_decode = [
        DecodeCase(
            name=f"roundtrip-{row['name']}",
            why=f"符号化 {row['name']} の出力（特殊トークン込み）をそのまま復号する",
            ids=list(row["idsWithSpecials"]),
        )
        for row in encode
    ] + list(decode_cases)
    decode: list[dict[str, Any]] = []
    for case in all_decode:
        decode.append(
            {
                "name": case.name,
                "why": case.why,
                "ids": case.ids,
                "text": reference.decode(case.ids, skip_special_tokens=False),
                "textSkipSpecials": reference.decode(case.ids, skip_special_tokens=True),
            }
        )

    required = {token_id for row in encode for token_id in row["idsWithSpecials"]}
    required.update(token_id for row in decode for token_id in row["ids"])
    return {
        "source": dict(source),
        "spec": spec_of(compiled),
        "asset": asset_subset(
            compiled,
            texts=[case.text for case in encode_cases],
            required_ids=required,
        ),
        "encode": encode,
        "decode": decode,
    }


def emit(
    *,
    tokenizer_json: Path,
    asset_path: Path,
    fixture_path: Path,
    extra_cases: Sequence[EncodeCase] = (),
) -> None:
    """1 資産ぶんの compile とフィクスチャ採取を**必ず両方**行う。

    片方だけ出す口を開けないのは、実行時資産とフィクスチャが別々に古びると「テストは緑だが
    実行だけ別の id 列」になるため（`anima/demo.py` と同じ理由）。
    """
    if not tokenizer_json.is_file():
        raise SystemExit(f"[karume] 上流の tokenizer.json が無い: {tokenizer_json}")
    raw = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    compiled = compile_tokenizer(raw)
    source = {
        # フィクスチャは git 管理なので、出所はリポジトリ相対で書く（絶対パスを commit しない）。
        "path": str(tokenizer_json.relative_to(REPO_ROOT)),
        **file_digest(tokenizer_json),
        "vocabCount": len(compiled.vocab),
        "mergesCount": len(compiled.merges),
        "addedCount": len(compiled.added_tokens),
    }
    asset_bytes = write_json(asset_path, asset_payload(compiled, source=source))
    fixture = build_fixture(
        tokenizer_json=tokenizer_json,
        compiled=compiled,
        encode_cases=[*SHARED_ENCODE_CASES, *extra_cases],
        decode_cases=byte_run_decode_cases(compiled),
        source=source,
    )
    fixture_bytes = write_json(fixture_path, fixture)
    print(f"[karume] 資産      {asset_path} ({asset_bytes / 1e6:.1f} MB)")
    print(
        f"[karume] フィクスチャ {fixture_path} ({fixture_bytes / 1e3:.0f} KB・"
        f"語彙 {len(fixture['asset']['vocab'])} / merges {len(fixture['asset']['merges'])} / "
        f"符号化 {len(fixture['encode'])} / 復号 {len(fixture['decode'])} ケース)"
    )
    # commit 形はフォーマッタが正（`deno task verify` の `fmt --check` が fixtures も見る）。
    print(
        f"[karume] 生成後はリポジトリ直下で `deno fmt {fixture_path.relative_to(REPO_ROOT)}` "
        "を掛ける"
    )


def byte_run_decode_cases(compiled: CompiledTokenizer) -> list[DecodeCase]:
    """byte run を直接叩く復号ケース（符号化からは作れない形を含む）。"""
    byte_ids = compiled.byte_ids
    return [
        DecodeCase(
            name="byte-all-256",
            why="byte 0..255 を 1 続きの run で流す（全体が不正 UTF-8 = 1 バイト 1 置換文字）",
            ids=list(byte_ids),
        ),
        DecodeCase(
            name="byte-run-truncated",
            why="3 バイト文字の頭 2 バイトだけ（不完全 run — run 全体が置換文字になる）",
            ids=[byte_ids[0xE3], byte_ids[0x81]],
        ),
        DecodeCase(
            name="byte-run-valid",
            why="3 バイト文字がちょうど揃う run（1 文字に畳まれる）",
            ids=[byte_ids[0xE3], byte_ids[0x81], byte_ids[0x82]],
        ),
        DecodeCase(
            name="byte-run-then-token",
            why="不完全 run の直後に非 byte トークン（run の切れ目が確定する位置）",
            ids=[byte_ids[0xE3], byte_ids[0x81], compiled.eos_id],
        ),
        DecodeCase(
            name="byte-run-split-by-token",
            why="run が非 byte トークンで 2 つに割れる（片方だけ有効）",
            ids=[
                byte_ids[0xE3],
                byte_ids[0x81],
                byte_ids[0x82],
                compiled.bos_id,
                byte_ids[0xF0],
                byte_ids[0x9F],
            ],
        ),
        DecodeCase(
            name="specials-only",
            why="特殊トークンだけ（skip_special_tokens の分岐が出る唯一の形）",
            ids=[compiled.bos_id, compiled.eos_id],
        ),
    ]
