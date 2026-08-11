"""DeBERTa-v2 の相対位置の添字表を**グラフ入力**へ出すパッチ層（ADR 0045 波 3）。

`DisentangledSelfAttention.disentangled_attention_bias` は `relative_pos`（`[1, T, T]`）から
`c2p_pos` / `p2c_pos` を clamp で作る。この 2 表は **T だけで決まる定数**なので、torch.export は
Tmax=512 で畳み込んで `[1,512,512]` の i32 を 2 本焼き込む（実測 2,097,152 B = 配布形の 0.68%）。
中身は Toeplitz でしかも互いに転置なので、実体は 1,023 要素の i32 ベクトルに等しい。

ADR 0013 が flow / voice の `(T,T)` 表で採ったのと同じ理由でこれを外部供給へ移す。副次的に
**Tmax=512 の上限が消える**（焼き込みが無くなるので、表の大きさが export 時の定数でなくなる）。

MUST: パッチ後は **eager 同値**であること — 表を外から渡す以外の違いを持ち込まない。同値性は
`export_deberta.py` が採る golden（torch CPU の期待値）と、TS 側パリティテストが実測で縛る。

MUST: 差し替え版は表をタプルで受ける形しか通さない（**fallback を持たない**）。元の実装へ黙って
落ちると「2MiB を焼いたまま緑」になり、パッチが外れたことが誰にも見えなくなる。

## この表が何であるか（式の正本）

`build_relative_position(q, k, bucket_size=256, max_position=512)` は `rel[i,j] = i − j` を
`make_log_bucket_position` で対数バケット化したもの。`att_span = pos_ebd_size = position_buckets`
= 256 なので:

    c2p_pos[i,j] = clamp(bucket(i − j) + 256, 0, 511)
    p2c_pos[i,j] = clamp(bucket(j − i) + 256, 0, 511) = c2p_pos[j,i]

`p2c` 側が転置になるのは、`build_rpos` が query_size == key_size のとき `relative_pos` をその
まま返し、`clamp(-rel + att_span, …)` を掛けるから（バケット化は奇関数）。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

#: 差し替えの適用済みフラグ（プロセス全域のクラス属性差し替えなので 1 回だけ）。
_applied = False


def patch_applied() -> bool:
    """差し替えが適用済みか（呼び出し側の順序違反を fail loudly にするための問い合わせ）。"""
    return _applied


def build_rel_pos_tables(
    length: int, *, position_buckets: int, max_position: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(c2p_pos, p2c_pos)` を実長 `length` で作る（**TS 鏡像の Python 側正本**）。

    どちらも `[length, length]` の int64。`disentangled_attention_bias` が `expand` で
    `[B*H, T, T]` へ広げるので、先頭に 1 の軸は付けない（元実装の `squeeze(0)` に相当）。
    """
    from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position

    ids = torch.arange(length, dtype=torch.long)
    rel = ids[:, None] - ids[None, :]
    bucket = make_log_bucket_position(rel, position_buckets, max_position).to(torch.long)
    span = position_buckets
    c2p_pos = torch.clamp(bucket + span, 0, span * 2 - 1)
    p2c_pos = torch.clamp(-bucket + span, 0, span * 2 - 1)
    return c2p_pos, p2c_pos


def _external_disentangled_attention_bias(
    self: nn.Module,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    relative_pos: Any,
    rel_embeddings: torch.Tensor,
    scale_factor: int,
) -> torch.Tensor:
    """`relative_pos` の位置で `(c2p_pos, p2c_pos)` を受け取る同値実装。

    元実装との差は「表を作らずに受け取る」ことだけ。clamp / build_rpos / squeeze が消え、
    残りの bmm → gather → scale は 1 対 1 で対応する。
    """
    if not isinstance(relative_pos, tuple) or len(relative_pos) != 2:
        raise TypeError(
            "外部供給の (c2p_pos, p2c_pos) 以外は受け付けない — パッチが中途半端に当たっている"
            f"（受け取ったのは {type(relative_pos).__name__}）"
        )
    from transformers.models.deberta_v2.modeling_deberta_v2 import scaled_size_sqrt

    c2p_pos, p2c_pos = relative_pos
    att_span = self.pos_ebd_size
    rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)
    heads = self.num_attention_heads
    repeats = query_layer.size(0) // heads
    pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), heads).repeat(
        repeats, 1, 1
    )
    pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), heads).repeat(
        repeats, 1, 1
    )

    # content -> position
    scale = scaled_size_sqrt(pos_key_layer, scale_factor)
    c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
    c2p_att = torch.gather(
        c2p_att,
        dim=-1,
        index=c2p_pos.expand([query_layer.size(0), query_layer.size(1), c2p_pos.size(-1)]),
    )
    score = c2p_att / scale.to(dtype=c2p_att.dtype)

    # position -> content
    scale = scaled_size_sqrt(pos_query_layer, scale_factor)
    p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
    p2c_att = torch.gather(
        p2c_att,
        dim=-1,
        index=p2c_pos.expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]),
    ).transpose(-1, -2)
    return score + p2c_att / scale.to(dtype=p2c_att.dtype)


def apply_external_rel_pos_patch() -> None:
    """`disentangled_attention_bias` を外部供給版へ差し替える（プロセス全域・1 回だけ）。

    MUST: モデルを構築する**前**に呼ぶ必要は無い（クラス属性の差し替えなので後でも効く）が、
    golden を採る前でなければならない — 採ってから差し替えると期待値だけが元の経路で計算される。
    """
    global _applied
    if _applied:
        return
    from transformers.models.deberta_v2 import modeling_deberta_v2

    modeling_deberta_v2.DisentangledSelfAttention.disentangled_attention_bias = (
        _external_disentangled_attention_bias
    )
    _applied = True


def assert_supported(config: Any) -> None:
    """差し替え版が前提にしている config を検査する（外れたら黙って別の式になる）。

    元実装は `pos_att_type` / `share_att_key` で 4 通りに分岐するが、差し替え版は
    ku-nlp の char-wwm が使う 1 通り（c2p + p2c・共有射影）しか書いていない。
    """
    pos_att_type = set(getattr(config, "pos_att_type", ()) or ())
    if pos_att_type != {"c2p", "p2c"}:
        raise ValueError(f"pos_att_type が c2p+p2c でない（{sorted(pos_att_type)}）")
    if not getattr(config, "share_att_key", False):
        raise ValueError("share_att_key=False は差し替え版が持たない経路")
    if int(getattr(config, "position_buckets", 0)) <= 0:
        raise ValueError(f"position_buckets が正でない（{config.position_buckets}）")
