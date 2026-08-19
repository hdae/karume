"""**自分の golden を採らない系列**の出所記録（`reference.json`）— 書き手側の正本。

token-only 系列（{@link gemma4.export_token}）は greedy 期待列を自分では採らず、logits opt-in
系列（{@link gemma4.export_decode}）の `greedy.<case>.safetensors` を検収門
（`packages/models/tests/e2e_gemma4_token_exit_test.ts`）へ流用する。安いのはこの形だが、
**2 系列が同じチェックポイントから出た**という前提だけが機械可読でなかった（全体レビュー
CX-2.3）— 資産ディレクトリの存在確認しか無いので、片方だけ古い組み合わせでも門は緑になる。

そこで token-only の export が、据える資産の隣へこの記録を書く:

- `checkpoint` — **元チェックポイントの指紋**（{@link FINGERPRINT_FILES} の sha256 と byte 数）。
  この容器を生んだ重み・config・トークナイザがどれだったかを名指しする。
- `reference` — **流用する golden 系列の識別**（系列ディレクトリ名と、参照する
  `greedy.<case>.safetensors` 1 本ずつの sha256 と byte 数）。

束縛の意味論: 「この容器は、checkpoint X から出て、**そこに書いた digest の golden そのもの**
に対して検収されるべきものだ」。golden を採り直せば（= 別のチェックポイントか別の丸めで
logits 系列を作り直せば）digest が動き、TS 側の門が **fail loudly** になる — 再 export を
強制されるので、そのとき `checkpoint` も一緒に更新される。逆に token-only 側だけ作り直しても
記録は容器と同じ据え替えで書かれるため、記録が容器より古くなる形が作れない。

MUST: **logits 系列の再 export を要求しない**（greedy 期待列の採り直しは 1 実走数十分）。
ここが読むのは既存 golden のバイト列だけで、書き足しも上書きもしない。

MUST: この記録は「同一チェックポイント由来」の**主張**であって証明ではない。数値としての
同一性は TS 側の系列間交差 parity（3 ケース × 16 step の厳密一致）が見る — ここが担うのは
「どの資産の組み合わせで parity を見るべきか」を機械可読に固定する側。export 時にも
{@link assert_reference_goldens} が golden の `prompt` を今回組んだケースと突合するので、
トークナイザ・ケース定義が食い違った組み合わせは記録を書く前に落ちる。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from _shared.decode_series import GREEDY_PREFIX, GREEDY_SUFFIX, PROMPT_KEY

#: 記録のファイル名（系列ディレクトリ直下・TS 側の検収門が同じ綴りで読む）。
REFERENCE_FILE = "reference.json"

#: 記録の版（読み手が知らない版を黙って読まないための欄）。
SCHEMA = 1

#: 指紋を採るチェックポイント側のファイル。重み本体だけでなく config と tokenizer も採る —
#: 同じ重みでも `config.json` の層構成やトークナイザが違えば別の資産になる。
FINGERPRINT_FILES = ("model.safetensors", "config.json", "tokenizer.json")

#: sha256 を採るときの読み込み単位（10GB のチェックポイントを丸ごとメモリに載せない）。
DIGEST_BLOCK = 8 << 20


def file_digest(path: Path) -> dict[str, Any]:
    """`path` の `{bytes, sha256}`（ブロック読み — 実体をメモリに載せない）。"""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(DIGEST_BLOCK)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def checkpoint_fingerprint(model_dir: Path) -> dict[str, Any]:
    """元チェックポイントの指紋（{@link FINGERPRINT_FILES} を 1 本ずつ）。"""
    files: dict[str, Any] = {}
    for name in FINGERPRINT_FILES:
        path = model_dir / name
        if not path.is_file():
            raise AssertionError(f"チェックポイントに指紋対象 '{name}' が無い: {model_dir}")
        files[name] = file_digest(path)
    return {"dir": model_dir.name, "files": files}


def _golden_prompt(path: Path) -> torch.Tensor:
    """golden の `prompt`（i32 の 1 次元）を読む。"""
    with safe_open(str(path), framework="pt") as handle:
        stored = handle.keys()
        if PROMPT_KEY not in stored:
            raise AssertionError(f"golden {path.name} に '{PROMPT_KEY}' が無い（{sorted(stored)}）")
        return handle.get_tensor(PROMPT_KEY)


def assert_reference_goldens(
    reference_dir: Path, cases: Sequence[tuple[str, torch.Tensor]]
) -> dict[str, Any]:
    """流用する golden 1 本ずつを検め、`{ファイル名: {bytes, sha256}}` を返す。

    MUST: `prompt` が**今回組んだケースと同一**であることを見る（トークナイザ・BOS の付け方・
    ケース本文のどれかが食い違った組み合わせを、digest を書く前に落とす）。golden の
    `expected` は読まない — 期待列との突合は実 GPU の検収門の仕事で、ここでその一部を
    torch で焼き直すと「同じ向きに間違った 2 つ」を作る余地が生まれる。
    """
    if not reference_dir.is_dir():
        raise AssertionError(
            f"参照 golden 系列 {reference_dir} が無い"
            " — 先に logits opt-in 系列（python -m gemma4.export_decode）を書き出すこと"
        )
    goldens: dict[str, Any] = {}
    for name, ids in cases:
        path = reference_dir / f"{GREEDY_PREFIX}{name}{GREEDY_SUFFIX}"
        if not path.is_file():
            raise AssertionError(f"参照 golden {path} が無い（流用先の系列に {name} が欠けている）")
        prompt = _golden_prompt(path).to(torch.int64)
        if not torch.equal(prompt, ids[0].to(torch.int64)):
            raise AssertionError(
                f"参照 golden {path.name} の prompt が今回のケース '{name}' と違う"
                f"（golden {tuple(prompt.shape)} / 今回 {tuple(ids[0].shape)}）"
                " — 別のトークナイザかケース本文で採られた golden を流用しようとしている"
            )
        goldens[path.name] = file_digest(path)
    return {"series": reference_dir.name, "goldens": goldens}


def build_record(series_dir: Path, model_dir: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    """`reference.json` の中身（{@link assert_reference_goldens} の結果を指紋と束ねる）。"""
    return {
        "schema": SCHEMA,
        "series": series_dir.name,
        "checkpoint": checkpoint_fingerprint(model_dir),
        "reference": dict(reference),
    }


def write_record(series_dir: Path, record: Mapping[str, Any]) -> None:
    """記録を系列ディレクトリへ書く。

    MUST: 呼び手は**容器と同じ据え替え単位**（`staged_publication` の作業席）へ書くこと —
    別々に据えると「新しい容器 + 古い記録」が一瞬でも作れてしまい、束縛が意味を失う。
    """
    path = series_dir / REFERENCE_FILE
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
