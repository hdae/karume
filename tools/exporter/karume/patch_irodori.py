"""Irodori-TTS v4 を torch.export 可能にするパッチ層。

現状の対象は**テキスト条件エンコーダ**（同梱 ModernBERT-ja-310m backbone）だけで、パッチも
1 種類しかない:

1. **ModernBertAttention の qkv 取り出しを rank ≤ 4 に落とす** — 原実装は
   `Wqkv(x).view(B, T, 3, H, D)` の **rank 5** を作ってから `unbind(dim=-3)` で 3 本に割る。
   分解後は rank 5 の `slice` になり、strided カーネルの rank 上限（`ops.STRIDED_RANK` = 4）に
   当たって export できない。最終次元を 3 等分する slice に置き換えると、同じ要素が同じ順序で
   同じ 3 本に落ちる（下の `_flat_qkv_attention_forward` の同値の根拠）。

MUST: パッチ後のモジュールはパッチ前と **eager 同値**であること（`export_irodori.py` が
実重み・全 golden ケースで実測し、差が出たら export ごと落とす）。

MUST: パッチはクラス属性の**プロセス全域**差し替えなので、「パッチ前の参照」を採れるのは
1 プロセスにつき 1 回だけ。適用済みかどうかは {@link patches_applied} が答え、順序違反は
呼び出し側（`export_irodori.py`）が fail loudly で拒否する（恒真化 = 偽 PASS の遮断）。

NOTE: transformers のモデリングコードを差し替えるので、版は `transformers==5.14.1` に
ピンして使う（`export_irodori.py` の docstring）。
"""

from __future__ import annotations

from typing import Any

import torch

#: パッチ適用済みフラグ。プロセス全域差し替えの副作用を可視化するためだけに持つ
#: （パッチ後に「パッチ前の参照」を採ると同値検証が恒真化する）。
_APPLIED = False


def patches_applied() -> bool:
    """このプロセスで既にパッチを当てたか。

    参照値を採る側（同値検証）が「まだ当てていない」ことを assert するための門。
    """
    return _APPLIED


def _flat_qkv_attention_forward(
    self: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`ModernBertAttention.forward` の同値実装（rank 5 を作らない）。

    原実装との差は **qkv の割り方 1 点だけ**で、残りは逐語（RoPE の適用・attention interface
    への引数・出力の reshape と `Wo` / `out_drop`）。

    同値の根拠: 原実装の `view(B, T, 3, H, D)` は最終次元 `3·H·D` を C 順で `(3, H, D)` に
    括り直すだけなので、`unbind(dim=-3)` の第 i 本は元の最終次元の `[i·H·D, (i+1)·H·D)` 区間
    そのもの。ここでの `slice` + `view(B, T, H, D)` は同じ区間を同じ順序で取り出す
    （コピーの有無しか違わない）。
    """
    batch, seq = hidden_states.shape[0], hidden_states.shape[1]
    heads = self.config.num_attention_heads
    width = heads * self.head_dim
    qkv = self.Wqkv(hidden_states)
    query_states, key_states, value_states = (
        qkv[..., start : start + width].view(batch, seq, heads, self.head_dim).transpose(1, 2)
        for start in (0, width, 2 * width)
    )

    from transformers.models.modernbert import modeling_modernbert as modernbert

    cos, sin = position_embeddings
    query_states, key_states = modernbert.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    attention_interface = modernbert.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, modernbert.eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.head_dim**-0.5,
        sliding_window=self.sliding_window,
        deterministic=self.deterministic_flash_attn,
        **kwargs,
    )

    attn_output = attn_output.reshape(batch, seq, width).contiguous()
    return self.out_drop(self.Wo(attn_output)), attn_weights


def apply_patches() -> None:
    """ModernBERT のパッチをプロセス全域へ当てる（冪等）。

    MUST: 呼び出し側は「パッチ前の参照」を採り終えてから呼ぶ（{@link patches_applied}）。
    """
    global _APPLIED
    if _APPLIED:
        return
    from transformers.models.modernbert.modeling_modernbert import ModernBertAttention

    ModernBertAttention.forward = _flat_qkv_attention_forward
    _APPLIED = True
