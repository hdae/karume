"""gemma4 E2B の chat フォーマットの parity フィクスチャ採取（ADR 0084 決定 5 / 決定 7）。

    uv run --with 'transformers==5.14.1' python -m gemma4.chat

出力は 1 本（**git 管理**）:

    packages/models/tests/fixtures/gemma-text/gemma4-chat.json

TS 側の実装は `packages/models/src/gemma/text/chat.ts` の純関数 1 本
（`gemma4ChatPrompt`）で、汎用テンプレートレンダラは作らない（ADR 0084 決定 5 —
「ランタイム依存は Web 標準のみ」と「投機的な一般化をしない」の両方に反する）。ここが
持つのはその**期待値**だけである。

MUST: 期待値は上流の `apply_chat_template` を**独立に**呼んで採る（ADR 0084 決定 7）—
自前のレンダラも compile 台本の畳み込みも 1 行も通さない。共有すると parity が恒真化する
（ADR 0048 追記の実例）。

## 射程は「素の会話」だけ（採らないケースの記録）

初版の射程は **role が `system` / `developer` / `user` / `assistant` の、content が文字列の
会話**だけで、次は**フィクスチャに採らない**（TS 側は fail loudly で拒否する — ADR 0084
決定 5・6.3）。射程を広げるのは実需が出てからで、そのときはここへケースを足す:

- `tools`（関数宣言）— `chat_template.jinja` の `format_function_declaration` /
  `format_parameters` が要る（tool 宣言のフォーマッタだけで 150 行級）
- `reasoning` / `reasoning_content`（thinking チャネル）と `enable_thinking` —
  `<|channel>thought` の順序制御と `strip_thinking` の巻き戻しが要る
- `tool_calls` / `tool_responses` / role `tool` — 引数の直列化と OpenAI 形の前方走査
- 画像 / 音声 / 動画パート（`<|image|>` / `<|audio|>` / `<|video|>`）

**黙って無視しない**のが要点である（無視すると「tool を渡したのに使われない」が例外なしで
通る）。

## 素の会話の射程（実測した分岐）

綴りは Gemma 3 系ではない（ADR 0084 Context 3 — `<start_of_turn>` は**この語彙に無い**）。
実際に踏む分岐は 3 本だけで、{@link CASES} が 1 本ずつ踏む:

1. 先頭が `system` / `developer` なら **`<|turn>system\\n` の system ブロック**へ落ちる
   （`developer` でも綴りは `system` — 2 番目以降の `developer` は `<|turn>developer\\n`）
2. `assistant` は `model` へ写像され、**直前の非 tool ロールも `assistant` なら turn を
   開き直さない**（連続 assistant は 1 つの model turn へ畳まれる）
3. 末尾に生成プロンプト `<|turn>model\\n` を必ず出す

`<bos>` は template が出す = **ホストの仕事**（ADR 0084 決定 5 の MUST — `encode` は
付けない。分けないと chat 導入時に double-BOS になる）。

## EOS 集合を同じ束で採る（ADR 0083 決定 8 / 0084 決定 5）

`generation_config.json` の `eos_token_id` と、その id の綴りをフィクスチャへ載せる。
TS 側は**トークナイザ資産の追加語彙から**同じ集合を導出するので、ここが「導出が上流宣言と
一致すること」の突合先になる。chat 形式と EOS 集合を別々の場所から拾うと片方だけ古くなる
ので、出所記録（{@link _source}）は 4 ファイルの digest を 1 つの束として書く。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _shared.gemma_tokenizer import (
    CompiledTokenizer,
    asset_subset,
    compile_tokenizer,
    file_digest,
    spec_of,
    write_json,
)
from _shared.paths import INPUTS_ROOT, REPO_ROOT

#: 上流の手置き資産（`inputs/<family>/<name>/` — assets-layout）。
CHECKPOINT_DIR = INPUTS_ROOT / "gemma4" / "gemma-4-E2B-it"

#: 出所記録に束ねる 4 ファイル（chat 形式と EOS 集合が**同じ digest set** から来ることの記録）。
SOURCE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)

#: パリティ用フィクスチャ（git 管理 — Deno 側 `gemma_chat_test.ts` が読む）。
FIXTURE_PATH = (
    REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "gemma-text" / "gemma4-chat.json"
)

#: 初版の射程（ADR 0084 決定 5）。ここに無い role は TS 側が fail loudly で拒否する。
PLAIN_ROLES = ("system", "developer", "user", "assistant")

#: メッセージが持ってよい欄。**許可リスト**にするのは、`tools` / `reasoning` /
#: `tool_calls` のような射程外の欄が「渡したのに効かない」形で素通りしないため。
ALLOWED_MESSAGE_KEYS = frozenset({"role", "content"})


class OutOfScopeChatError(ValueError):
    """初版の射程（素の会話）から外れた入力。"""


@dataclass(frozen=True)
class ChatCase:
    """chat ケース（`why` は「素朴な移植が落ちる境界」を 1 行で述べる）。"""

    name: str
    why: str
    messages: tuple[Mapping[str, Any], ...]


def assert_plain_conversation(messages: Sequence[Mapping[str, Any]]) -> None:
    """素の会話であることを確かめる（射程外は採らない）。

    MUST: 上流と同じところで落とす — `apply_chat_template` は空の会話を
    `ValueError` で拒否するので、ここも拒否する（TS 側の受理集合を上流より広げない）。
    """
    if not messages:
        raise OutOfScopeChatError("会話が空（上流の apply_chat_template も拒否する）")
    for index, message in enumerate(messages):
        extra = sorted(set(message) - ALLOWED_MESSAGE_KEYS)
        if extra:
            raise OutOfScopeChatError(
                f"messages[{index}]: 射程外の欄 {extra}"
                "（tools / reasoning / tool_calls などは初版では拒否する — ADR 0084 決定 5）"
            )
        role = message.get("role")
        if role not in PLAIN_ROLES:
            raise OutOfScopeChatError(
                f"messages[{index}].role {role!r} が {list(PLAIN_ROLES)} のどれでもない"
            )
        if not isinstance(message.get("content"), str):
            raise OutOfScopeChatError(
                f"messages[{index}].content が文字列でない"
                "（画像 / 音声パートの配列は初版では拒否する）"
            )


#: 素の会話の電池。**分岐と境界を 1 件ずつ**踏ませる（同じ形を 2 度採らない）。
CASES: tuple[ChatCase, ...] = (
    ChatCase(
        "single-user",
        "最小形 — `<bos>` + user turn + 生成プロンプト",
        ({"role": "user", "content": "What is the capital of France?"},),
    ),
    ChatCase(
        "system-first",
        "先頭 system は system ブロックへ落ち、content は trim される",
        (
            {"role": "system", "content": "  You are a terse assistant.  "},
            {"role": "user", "content": "Hi"},
        ),
    ),
    ChatCase(
        "developer-first",
        "先頭 developer も**綴りは system**（role をそのまま出さない唯一の位置）",
        (
            {"role": "developer", "content": "Follow the house style."},
            {"role": "user", "content": "Hi"},
        ),
    ),
    ChatCase(
        "developer-late",
        "2 番目以降の developer は `<|turn>developer` （先頭との非対称）",
        (
            {"role": "user", "content": "Hi"},
            {"role": "developer", "content": "Be brief."},
        ),
    ),
    ChatCase(
        "round-trip",
        "user → assistant → user の往復（assistant は `model` へ写像される）",
        (
            {"role": "user", "content": "Who wrote Hamlet?"},
            {"role": "assistant", "content": "William Shakespeare."},
            {"role": "user", "content": "When?"},
        ),
    ),
    ChatCase(
        "assistant-last",
        "末尾が assistant でも turn は閉じ、生成プロンプトは**新しい** model turn を開く",
        (
            {"role": "user", "content": "Who wrote Hamlet?"},
            {"role": "assistant", "content": "William Shakespeare."},
        ),
    ),
    ChatCase(
        "assistant-twice",
        "連続 assistant は 1 つの model turn へ畳まれる（turn を開き直さない・閉じない）",
        (
            {"role": "user", "content": "Continue."},
            {"role": "assistant", "content": "First half."},
            {"role": "assistant", "content": "Second half."},
        ),
    ),
    ChatCase(
        "assistant-first",
        "assistant で始まる会話（直前ロールが無いので継続にならない）",
        ({"role": "assistant", "content": "I already answered."},),
    ),
    ChatCase(
        "user-twice",
        "連続 user は畳まれない（継続の判定は model turn 側だけ）",
        (
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ),
    ),
    ChatCase(
        "system-only",
        "system だけの会話（ループが 1 度も回らず生成プロンプトへ抜ける）",
        ({"role": "system", "content": "You are a calculator."},),
    ),
    ChatCase(
        "empty-content",
        "空の content（turn は開いて即閉じる — 空文字を落とさない）",
        ({"role": "user", "content": ""},),
    ),
    ChatCase(
        "whitespace-content",
        "trim は Python の `str.strip()` — 全角空白 U+3000 も改行もタブも落ちる",
        ({"role": "user", "content": "\n\t hi 　\n"},),
    ),
    ChatCase(
        "assistant-whitespace",
        "assistant 側も trim される（`strip_thinking` は thinking 綴りが無ければ trim と同値）",
        (
            {"role": "user", "content": "A?"},
            {"role": "assistant", "content": "   B.   "},
            {"role": "user", "content": "C?"},
        ),
    ),
    ChatCase(
        "japanese",
        "日本語の多ターン（空白が無いので 1 pre-token が turn 本文の全長になる）",
        (
            {"role": "system", "content": "あなたは簡潔に答える助手です。"},
            {"role": "user", "content": "日本の首都はどこですか？"},
            {"role": "assistant", "content": "東京です。"},
            {"role": "user", "content": "その人口は？"},
        ),
    ),
    ChatCase(
        "special-in-content",
        "本文中の chat 綴りは**エスケープされない**（追加語彙として切り出される）",
        ({"role": "user", "content": "abc<|turn>def"},),
    ),
    ChatCase(
        "multiline-content",
        "改行を含む content（turn の区切りは改行ではなく `<turn|>`）",
        ({"role": "user", "content": "line one\nline two\n\nline four"},),
    ),
)


def _source(checkpoint_dir: Path) -> dict[str, Any]:
    """出所記録（chat 形式と EOS 集合を束ねる 4 ファイルの digest — ADR 0084 決定 5）。"""
    return {
        "dir": str(checkpoint_dir.relative_to(REPO_ROOT)),
        "files": {name: file_digest(checkpoint_dir / name) for name in SOURCE_FILES},
    }


def stop_tokens(checkpoint_dir: Path) -> list[int]:
    """`generation_config.json` の `eos_token_id`（単数でも集合として読む）。"""
    raw = json.loads((checkpoint_dir / "generation_config.json").read_text(encoding="utf-8"))
    declared = raw.get("eos_token_id")
    ids = [declared] if isinstance(declared, int) else declared
    if not isinstance(ids, list) or not ids or not all(isinstance(one, int) for one in ids):
        raise ValueError(
            f"generation_config.json の eos_token_id が整数 / 整数列でない: {declared!r}"
        )
    return list(ids)


def harvest_cases(tokenizer: Any, cases: Sequence[ChatCase]) -> list[dict[str, Any]]:
    """上流の `apply_chat_template` から期待値を採る（自前のレンダラを 1 行も通さない）。

    MUST: 「レンダリング済み文字列」と「id 列」の**両方**を採る。片方だけだと、割れたときに
    レンダリングの問題か符号化の問題かが読み手に伝わらない（TS 側は同じ 2 段で組む）。
    MUST: `ids` が「レンダリング結果を `add_special_tokens=False` で符号化したもの」と一致する
    ことをここで確かめる — TS 側の実装（描画 → `encode`）が成立する前提そのものなので、
    崩れたら期待値を採る側で落とす。
    """
    rows: list[dict[str, Any]] = []
    for case in cases:
        assert_plain_conversation(case.messages)
        messages = [dict(message) for message in case.messages]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = list(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)[
                "input_ids"
            ]
        )
        direct = list(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if ids != direct:
            raise AssertionError(
                f"{case.name}: apply_chat_template(tokenize=True) が"
                "「描画 → add_special_tokens=False の符号化」と一致しない"
                "（TS 側の 2 段構成の前提が崩れている）"
            )
        rows.append(
            {
                "name": case.name,
                "why": case.why,
                "messages": messages,
                "rendered": rendered,
                "ids": ids,
            }
        )
    return rows


def build_chat_fixture(
    *,
    tokenizer: Any,
    compiled: CompiledTokenizer,
    cases: Sequence[ChatCase],
    stops: Sequence[int],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """chat フィクスチャ 1 本を組む（期待値は上流・部分集合は compile 済み資産から）。

    語彙・merges の部分集合の切り方は符号化フィクスチャと同じ 1 本
    （{@link _shared.gemma_tokenizer.asset_subset}）を通す — 絞り方を 2 実装持たない。
    足りなければ TS 側の突合が**落ちる**ので、誤りは危険側に倒れない。
    """
    chat = harvest_cases(tokenizer, cases)
    required = {token_id for row in chat for token_id in row["ids"]}
    # EOS 集合の id は本文に出ないもの（`<|tool_response>`）を含むので、明示的に残す —
    # TS 側は追加語彙からこの集合を導出し、ここの `stopTokens` と突き合わせる。
    required.update(stops)
    required.add(compiled.bos_id)
    added_by_id = {token_id: token for token, token_id in compiled.added_tokens}
    return {
        "source": dict(source),
        "spec": spec_of(compiled),
        "asset": asset_subset(
            compiled,
            texts=[row["rendered"] for row in chat],
            required_ids=required,
        ),
        "stopTokens": list(stops),
        "stopTokenSpellings": [added_by_id.get(token_id) for token_id in stops],
        "chat": chat,
    }


def main() -> None:
    tokenizer_json = CHECKPOINT_DIR / "tokenizer.json"
    for name in SOURCE_FILES:
        if not (CHECKPOINT_DIR / name).is_file():
            raise SystemExit(f"[karume] 上流の {name} が無い: {CHECKPOINT_DIR / name}")
    from transformers import AutoTokenizer  # 遅延 import（採取のときだけ要る）

    compiled = compile_tokenizer(json.loads(tokenizer_json.read_text(encoding="utf-8")))
    fixture = build_chat_fixture(
        tokenizer=AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR)),
        compiled=compiled,
        cases=CASES,
        stops=stop_tokens(CHECKPOINT_DIR),
        source=_source(CHECKPOINT_DIR),
    )
    written = write_json(FIXTURE_PATH, fixture)
    print(
        f"[karume] フィクスチャ {FIXTURE_PATH} ({written / 1e3:.0f} KB・"
        f"chat {len(fixture['chat'])} ケース / 語彙 {len(fixture['asset']['vocab'])} / "
        f"merges {len(fixture['asset']['merges'])} / 停止 {fixture['stopTokens']})"
    )
    print(
        f"[karume] 生成後はリポジトリ直下で `deno fmt {FIXTURE_PATH.relative_to(REPO_ROOT)}` "
        "を掛ける"
    )


if __name__ == "__main__":
    main()
