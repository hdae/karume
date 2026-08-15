"""`karume::` 名前空間の custom op（torch.export で 1 ノードのまま残す層 — ADR 0056）。

`torch.library.custom_op` は本体を `register_fake` の裏へ隠すので、トレースが Python ループへ
入らない。そのため **記号次元（時間軸 T）を保ったまま単一ノードで残り**、
`run_decompositions(curated_decompositions())` を通しても展開されない（`karume::` は
Core ATen の分解表に無い — `torchvision::deform_conv2d` と同じ理由）。

これが解いている問題: `aten.gru.input` の分解は Python ループで **T 回展開**されるので、
`torch.export` が長さ軸を specialize してしまう（`Specializations unexpectedly required (T)`）。
vowel-detector では T10=200 で 8,434 ノードになり、長さごとに別のグラフが要る。

MUST: 本体（eager 実装）は **torch の GRU 分解の逐語**で書く。golden の期待値はこの実装を
eager で回して採るので、ここがずれると「エクスポータとランタイムが一致して両方間違っている」
状態が緑になる。特に更新式は `(h − n)·z + n` であって `(1 − z)·n + z·h` ではない
（f32 で 10 万要素中 44,345 件がビット不一致 — ADR 0056 決定 3）。

NOTE: `torch.library.custom_op` はプロセス全域のグローバル登録なので、この import は
**副作用を持つ**（`packages/*` の「全モジュール副作用ゼロ」は TS 側の tree-shaking 不変条件で、
Python 側 exporter には掛からない）。`convert.py` はハンドラのキー
（`torch.ops.karume.gru_scan.default`）を書くためにこのモジュールを import する。
"""

from __future__ import annotations

import torch

#: ゲートの本数（r / z / n）。gi と w_hh / b_hh の第 1 軸はこの倍数。
GATE_COUNT = 3


def _scan(
    gi: torch.Tensor,
    h0: torch.Tensor,
    w_hh: torch.Tensor,
    b_hh: torch.Tensor,
    *,
    reverse: bool,
) -> torch.Tensor:
    """隠れ側スキャン本体（入力側 GEMM は呼び手の `linear` が済ませている前提）。

    MUST: 演算の並びと括り方を変えない。ゲートの足し順は**隠れ側が第 1 引数**、`n` の積は
    `h_n · r`（reset は **b_hh 込みの隠れ側積**に掛かる）、更新は `(h − n)·z + n`。
    MUST: 逆方向は**走査順だけ**を反転し、書き出しは順方向の時間添字（出力に `flip` を
    掛けない — ランタイム側カーネルと同じ契約）。
    """
    length = gi.shape[0]
    hidden = h0.shape[1]
    order = range(length - 1, -1, -1) if reverse else range(length)
    state = h0
    steps: list[torch.Tensor] = [h0] * length
    for step in order:
        gh = torch.nn.functional.linear(state, w_hh, b_hh)
        input_r, input_z, input_n = gi[step].split(hidden, dim=1)
        hidden_r, hidden_z, hidden_n = gh.split(hidden, dim=1)
        reset = torch.sigmoid(hidden_r + input_r)
        update = torch.sigmoid(hidden_z + input_z)
        candidate = torch.tanh(input_n + hidden_n * reset)
        state = (state - candidate) * update + candidate
        steps[step] = state
    return torch.stack(steps, dim=0)


@torch.library.custom_op("karume::gru_scan", mutates_args=())
def gru_scan(
    gi: torch.Tensor, h0: torch.Tensor, w_hh: torch.Tensor, b_hh: torch.Tensor
) -> torch.Tensor:
    """順方向の隠れ側スキャン: gi[T,N,3H] / h0[N,H] / w_hh[3H,H] / b_hh[3H] -> y[T,N,H]。"""
    return _scan(gi, h0, w_hh, b_hh, reverse=False)


@torch.library.custom_op("karume::gru_scan_reverse", mutates_args=())
def gru_scan_reverse(
    gi: torch.Tensor, h0: torch.Tensor, w_hh: torch.Tensor, b_hh: torch.Tensor
) -> torch.Tensor:
    """逆方向の隠れ側スキャン（**出力は順方向の時間順**）。"""
    return _scan(gi, h0, w_hh, b_hh, reverse=True)


def _fake(
    gi: torch.Tensor, h0: torch.Tensor, w_hh: torch.Tensor, b_hh: torch.Tensor
) -> torch.Tensor:
    """メタ実装。時間軸は `gi.shape[0]` を**そのまま**伝播させる（記号のまま残る要）。"""
    del w_hh, b_hh
    return gi.new_empty((gi.shape[0], gi.shape[1], h0.shape[1]))


gru_scan.register_fake(_fake)
gru_scan_reverse.register_fake(_fake)
