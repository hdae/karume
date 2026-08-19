"""decode 系列の台本が共有する門とファイル操作（ADR 0065 決定 2 — core へは昇格できない）。

ここに置くのは **states 形 decode 台本のうちモデル知識を 1 つも持たない層**だけ:
greedy 継続の採り方（{@link greedy_continuation}）・その余裕門（{@link assert_greedy_margins}）
・golden の書き出し（{@link _write_greedy}）・位置表の容量門（{@link assert_case_room}）・
staging → final の入れ替え（{@link _publish}）。config も layer_type も読まないので、family
ごとに引数を一般化する必要が無い。

MUST: **これらの規律の正本はここ 1 本**（family 側の台本は import して使う）。門を片方の
family でだけ強めると、もう片方が黙って弱くなる — 実際に逐語コピーだった時期に margin 床の
MUST が gemma4 側にしか無い状態が生まれた（2026-08-19 レビュー G3-03）。

ここへ**上げないもの**: `_write_container`（量子化引数の有無で分岐する）と `_write_io`
（出力本数と入力名が family の形）。一般化するとかえって読めなくなる境界を越えない。

依存方向は recipe → core の一方向だけ（`_shared` は family を import しない — 綴りが要る
ものはここが持つ）。
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

import torch
from safetensors.torch import save_file
from torch import nn

from karume.convert import normalize_boundary_tensor

#: greedy golden のファイル名（`greedy.<case>.safetensors`）。読み手は TS 側の検収門と
#: `minicpm5/sweep_w4.py`。拡張子は 1-shot の `io.*` と同じだが、`_shared` は family を
#: import しないので綴りはここが持つ。
GREEDY_PREFIX = "greedy."
GREEDY_SUFFIX = ".safetensors"

#: `greedy.<case>.safetensors` のテンソルキー。
PROMPT_KEY = "prompt"
EXPECTED_KEY = "expected"
MARGIN_KEY = "margin"


def positions_for(ids: torch.Tensor) -> torch.Tensor:
    """無 pad 全長の位置列 `[[0, 1, …, T-1]]`（1 chunk で全 prompt を食う形）。"""
    return torch.arange(int(ids.shape[1])).unsqueeze(0)


def greedy_continuation(
    wrapper: nn.Module, ids: torch.Tensor, steps: int, *, label: str = ""
) -> tuple[list[int], list[float]]:
    """chunk ラッパを**全長で呼び直す** greedy 継続 `steps` step の `(token 列, margin 列)`。

    MUST: full re-forward（毎 step 先頭から標準 causal で計算）で採る。KV cache 経路で採ると
    「ランタイムが検収したい機構」と同じ機構で期待値を作ることになり、両方が同じ向きに
    間違っていても緑になる。遅いのは承知の上（1 ケース `steps` 回の全長 forward）。

    `margin` は各 step の top1 − top2 の logit 差。GPU 実行の偏差でこの列が割れないことの
    保証で、閾値検査は {@link assert_greedy_margins} が持つ。
    """
    tokens: list[int] = []
    margins: list[float] = []
    current = ids
    for step in range(steps):
        with torch.no_grad():
            logits, _ = wrapper(current, positions_for(current))
        best = torch.topk(logits[0, -1], 2)
        token = int(best.indices[0])
        tokens.append(token)
        margins.append(float(best.values[0] - best.values[1]))
        current = torch.cat([current, best.indices[:1].unsqueeze(0)], dim=1)
        print(
            f"[greedy] {label} step {step + 1}/{steps} token={token} margin={margins[-1]:.4g}",
            file=sys.stderr,
            flush=True,
        )
    return tokens, margins


def assert_greedy_margins(margins: Mapping[str, Sequence[float]], floor: float) -> None:
    """全ケース・全 step の margin が下限を超えることを見る（波 A の余裕保証門の多 step 版）。

    MUST: fail loudly。margin の小さい step を golden に混ぜると「GPU 側が正しくても偏差で
    1 位が入れ替わる」門になり、以後どの赤も信用できなくなる。
    MUST: **全ケースを測り終えてから 1 度に**掛ける（1 ケース目で止めない）。門の役目は
    「どのケースを golden にできるか」の判断材料で、最初の 1 件で止めると除外のたびに実走を
    やり直すことになる（実重みの 1 実走は数十分かかる）。
    """
    weak = {
        name: {step: margin for step, margin in enumerate(values) if margin <= floor}
        for name, values in margins.items()
    }
    offenders = {name: steps for name, steps in weak.items() if steps}
    if offenders:
        raise AssertionError(
            f"margin が下限 {floor} 以下の step がある {offenders}"
            "（K を下げず、当該ケースを GREEDY_CASES から外すこと）"
        )


def _write_greedy(
    wrapper: nn.Module,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
    *,
    steps: int,
    floor: float,
) -> tuple[list[str], dict[str, list[int]], dict[str, list[float]]]:
    """採用ケースの greedy 継続を測り、**全ケース測ってから**門を掛けて書く。

    門より前に 1 本も書かないので、落ちたときに「一部のケースだけ新しい」golden が残らない。
    """
    tokens: dict[str, list[int]] = {}
    margins: dict[str, list[float]] = {}
    for name, ids in cases:
        tokens[name], margins[name] = greedy_continuation(wrapper, ids, steps, label=name)
    assert_greedy_margins(margins, floor)

    written: list[str] = []
    for name, ids in cases:
        continuation, margin = tokens[name], margins[name]
        tensors = {
            PROMPT_KEY: normalize_boundary_tensor(ids[0], f"{name} の prompt"),
            EXPECTED_KEY: normalize_boundary_tensor(
                torch.tensor(continuation, dtype=torch.int64), f"{name} の expected"
            ),
            MARGIN_KEY: torch.tensor(margin, dtype=torch.float32).contiguous(),
        }
        path = out_dir / f"{GREEDY_PREFIX}{name}{GREEDY_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written, tokens, margins


def assert_case_room(cases: Sequence[tuple[str, torch.Tensor]], steps: int, positions: int) -> None:
    """prompt + 継続が RoPE 表の位置数に収まることを見る（表の外は引けない）。"""
    for name, ids in cases:
        total = int(ids.shape[1]) + steps
        if total > positions:
            raise AssertionError(
                f"{name}: prompt {int(ids.shape[1])} + {steps} step = {total} が"
                f" RoPE 表の位置数 {positions} を超える"
            )


def _publish(staging: Path, final: Path) -> None:
    """全ての門を通した staging ディレクトリを final へ**丸ごと**入れ替える。

    MUST: 公開は形検査・margin 門・sanity の**全部の後**（呼び手 = 各 family の
    `export_series` の構造で保証）。ファイル単位で final へ書いていく形だと、途中の門で
    落ちたときに「新しい model + 古い greedy」の**混ざった正規資産**が残り、検収門が
    拒否済みの資産で緑になれる。

    完全な原子性は狙わない — 退避 → 昇格の 2 rename の間で落ちると final は**不在**になるが、
    不在は読み手が確実に検出できる（fail loudly）。作れてはいけないのは「静かに読めてしまう
    混成」で、この手順はそれを構造的に作れない。昇格に失敗したら旧資産を戻す。
    """
    retired: Path | None = None
    if final.exists():
        retired = final.with_name(f"{final.name}.retired-{uuid4().hex}")
        os.replace(final, retired)
    try:
        os.replace(staging, final)
    except BaseException:
        if retired is not None:
            os.replace(retired, final)
        raise
    if retired is not None:
        shutil.rmtree(retired)
