"""`nn.GRU` を `karume::gru_scan` の並びへ割る patch 層（ADR 0056 決定 7）。

`aten.gru.input` の分解は Python ループで**時間方向へ完全展開**されるので、`torch.export` が
長さ軸を specialize する（`Specializations unexpectedly required (T)`）。差し替え先の
`karume::gru_scan{,_reverse}` は本体が `register_fake` の裏に隠れるため T を記号のまま通す
（`karume/custom_ops.py`）。

## 割り方 — 多層 / 双方向は**ノードを並べて**表す

IR v1 の可変アリティは `cat` だけで、`aten.gru` の `Tensor[16]`（層 × 方向 × 4 本の重み）は
構造的に載らない。したがって層と方向はグラフの構造で表す:

    層 k: gi = linear(x, W_ih_k, b_ih_k)         ← 入力側 GEMM は既存 `linear`（時間一括 1 本）
          y_fwd = gru_scan(gi, h0, W_hh_k, b_hh_k)
          y_rev = gru_scan_reverse(gi_rev, h0, W_hh_k_rev, b_hh_k_rev)
          x     = cat([y_fwd, y_rev], dim=2)     ← 次の層の入力

**この分割は `nn.GRU` と `torch.equal` でビット一致する**（`tests/test_patch_vowel_detector.py`
が実測で固定）。一致の根拠は torch の GRU 分解の構造をそのまま写していること:

- 入力側 linear は**時間一括 1 本**（`[T,N,3H]` を作ってから t ごとに切る）で、隠れ側だけが
  毎ステップ。したがって入力側を素の `linear` へ出しても丸め列は変わらない。
- 逆方向は**走査順だけ**が逆で、書き出しは順方向の時間添字（`flip` は記号軸を拒否するので
  出力側の反転は op の中へ畳んである）。
- ゲートの式と更新式は `karume/custom_ops.py` の本体が持つ（`(h − n)·z + n`）。

## 対応する `nn.GRU` の形

母音認識 CRNN が持つ形（batch_first・双方向・bias あり・dropout なし）だけを受ける。
欄を増やさないのは ADR 0023 決定 4 の規律で、**対応外は fail loudly**（黙って別の経路を
選ぶと「グラフは焼けるが数値が別物」が沈黙で出る）。
"""

from __future__ import annotations

import torch
from torch import nn

# NOTE: import は `karume::` 名前空間への**登録の副作用**も兼ねる（`torch.ops.karume` は
# 登録済みでなければ引けない）。
from karume import custom_ops  # noqa: F401  — torch.ops.karume の登録

#: 方向ごとの (`state_dict` の接尾辞, 走査する op)。並びは `cat` の並び順そのもので、
#: `nn.GRU` の出力レイアウト（前半が順方向・後半が逆方向）と一致していなければならない。
_DIRECTIONS = (
    ("", torch.ops.karume.gru_scan),
    ("_reverse", torch.ops.karume.gru_scan_reverse),
)


def assert_supported(gru: nn.GRU) -> None:
    """差し替えが**同値**になる形かを見る（対応外は fail loudly）。

    見ているのはどれも「通してしまうと値が静かに変わる」軸:
    `batch_first=False` は時間軸と batch 軸の取り違え、`bidirectional=False` は逆方向ぶんの
    `cat` が余る形、`bias=False` は `b_ih` / `b_hh` が存在しない形（`gru_scan` の契約は
    bias 必須 — ADR 0056 決定 6）、`dropout` は推論では無効でも学習時と式が違う宣言。
    """
    if not isinstance(gru, nn.GRU):
        raise TypeError(f"nn.GRU でない（{type(gru).__name__}）")
    if not gru.batch_first:
        raise ValueError("batch_first=False は差し替え版が持たない経路（時間軸の位置が違う）")
    if not gru.bidirectional:
        raise ValueError("bidirectional=False は差し替え版が持たない経路（実測に出た形は双方向）")
    if not gru.bias:
        raise ValueError("bias=False は差し替え版が持たない経路（gru_scan の契約は bias 必須）")
    if gru.dropout != 0:
        raise ValueError(f"dropout={gru.dropout} は差し替え版が持たない経路")


def gru_forward(gru: nn.GRU, x: torch.Tensor) -> torch.Tensor:
    """`nn.GRU.forward` の**出力 `y` だけ**を `gru_scan` の並びで組む。

    `[N, T, input_size]` → `[N, T, 2·hidden_size]`（batch_first の入出力そのもの）。

    MUST: `h_n` を返さない — IR v1 は実質単一出力で、`gru_scan` は最終状態を持ち帰れない
    （ADR 0056 決定 5）。呼び手が `h_n` を消費する形はこの層では表せない。
    """
    assert_supported(gru)
    hidden_size = gru.hidden_size
    # 全層・全方向の初期状態はゼロ（`nn.GRU` の `h_0` 省略時と同じ）。定数なので変換段で
    # 畳み込まれ、IR には initializer 1 本として載る。
    initial = torch.zeros(x.shape[0], hidden_size)
    sequence = x.transpose(0, 1)
    for layer in range(gru.num_layers):
        directions = []
        for suffix, scan in _DIRECTIONS:
            gates = nn.functional.linear(
                sequence,
                getattr(gru, f"weight_ih_l{layer}{suffix}"),
                getattr(gru, f"bias_ih_l{layer}{suffix}"),
            )
            directions.append(
                scan(
                    gates,
                    initial,
                    getattr(gru, f"weight_hh_l{layer}{suffix}"),
                    getattr(gru, f"bias_hh_l{layer}{suffix}"),
                )
            )
        # 次の層は両方向の連結を受ける（`nn.GRU` の層間の受け渡しと同じ）。
        sequence = torch.cat(directions, dim=2)
    return sequence.transpose(0, 1)
