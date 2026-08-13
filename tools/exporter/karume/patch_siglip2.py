"""SigLIP2 vision tower を torch.export 可能にするパッチ層。

対象は `google/siglip2-base-patch16-224` の画像側だけ（text tower は載せない）。config が
`model_type="siglip"` なので transformers が組むのは **`Siglip*`** のクラス群
（`SiglipVisionModel` / `SiglipVisionEmbeddings` / `SiglipMultiheadAttentionPoolingHead`）で、
`Siglip2*`（naflex 系）ではない。transformers 5.x では `vision_model` の中間ラッパが無く、
`SiglipVisionModel` が `embeddings` / `encoder` / `post_layernorm` / `head` を直接持つ。

パッチは**同値の強さで 2 段**に分かれる:

- **形の畳み込み（{@link apply_shape_patches}）— ビット同一**
  ① `patch_embedding` の `padding="valid"`（文字列）は `aten.conv2d.padding` 過負荷を選ぶ。
     変換段の HANDLERS が持つのは `aten.conv2d.default` だけなので、同値な `(0, 0)` に直す。
  ② `position_embedding(position_ids)` は `position_ids = arange(N)` の i64 バッファ相手の
     行 gather なので、出力は weight そのもの。直加算へ書き換えると **i64 の initializer**
     （IR v1 の initializer は意味論 f32 のみ）と `aten.embedding` ノードが同時に消える。
- **MAP head の q/k/v 明示化（{@link apply_map_head_patch}）— ビット同一ではない**
  ③ `SiglipMultiheadAttentionPoolingHead` は `torch.nn.MultiheadAttention` をそのまま持ち、
     `need_weights` 既定 True で呼ぶので手動経路（q に 1/√d を掛けてから bmm）へ落ちる。
     パック `in_proj_weight` からアクティベーション側の k/v を取り出す形が長さ 2 の軸の
     `aten.select.int` になり、`normalize._select_to_squeeze` は「静的で長さ 2 以上の軸は
     書き換えずに残す」ので未対応 op として残る。`probe.repeat(B,1,1)` も同様に
     `aten.repeat.default` として残る（`probe` は Parameter なので lifted 定数にならない）。
     `in_proj_weight` を 3 等分した明示 linear + `scaled_dot_product_attention` に書き直すと
     両方が同時に消え、12 層のエンコーダ層と同じ分解経路（mul×2 + bmm + softmax + bmm）に揃う。

MUST: パッチはどちらも fallback を持たない（`patch_deberta` と同じ規律）。前提を外した
モデルは黙って別の数値経路へ落ちるのではなく、その場で落ちる。

MUST: ③ は式としては同値だが**ビット一致ではない** — `nn.MultiheadAttention` は q にだけ
1/√d を掛けるのに対し、SDPA は q と k に √(1/√d) を対称に割る。実測の差は pooler_output で
7.75e-07〜2.38e-06（`export_siglip2.py` の golden 4 ケース・ベクトルの L2 ノルムは 12.7〜13.1
なので相対 ~1e-7）。golden の期待値は**差し替え後**のモジュールから採る（`export_siglip2.py`）
ので、e2e の門はこの差を跨がない。差し替え前後の実測は `export_siglip2.py --verify` が
2 点評価で毎回採る。

MUST: パッチはクラス属性の**プロセス全域**差し替えなので、「パッチ前の参照」を採れるのは
1 プロセスにつき 1 回だけ。適用済みかどうかは {@link patches_applied} が答え、順序違反は
呼び出し側（`export_siglip2.py`）が fail loudly で拒否する（恒真化 = 偽 PASS の遮断）。

NOTE: transformers のモデリングコードを差し替えるので、版は `transformers==5.14.1` に
ピンして使う（pyproject.toml の `siglip2` グループ）。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

#: 期待する活性（`hidden_act`）。MAP head の MLP もエンコーダの MLP もここから決まるので、
#: 外れると台本が約束する op 表（`gelu_tanh`）が黙って別の op に変わる。
EXPECTED_HIDDEN_ACT = "gelu_pytorch_tanh"

#: 形の畳み込み（① + ②）を当てたか。
_SHAPE_APPLIED = False
#: MAP head の q/k/v 明示化（③）を当てたか。
_MAP_HEAD_APPLIED = False


def patches_applied() -> bool:
    """このプロセスで（どちらの段でも）クラス属性の差し替えを当てたか。

    参照値を採る側（`export_siglip2.py --verify`）が「まだ当てていない」ことを assert する
    ための門 — 当てた後に「当てる前の参照」を採ると、同値検証が差 0 で恒真化する。
    """
    return _SHAPE_APPLIED or _MAP_HEAD_APPLIED


def assert_supported(config: Any) -> None:
    """差し替え版が前提にしている vision config を検査する。

    ここで見るのは**外れても shape エラーにならずに別の数値経路へ落ちる**ものだけ。
    """
    if config.hidden_act != EXPECTED_HIDDEN_ACT:
        raise ValueError(
            f"hidden_act が {EXPECTED_HIDDEN_ACT} でない（{config.hidden_act}）—"
            " 台本が約束する op 表が変わる"
        )
    if config.num_channels != 3:
        raise ValueError(f"num_channels が 3 でない（{config.num_channels}）— 前処理の規約外")
    if config.hidden_size % config.num_attention_heads:
        raise ValueError(
            f"hidden_size {config.hidden_size} が num_attention_heads"
            f" {config.num_attention_heads} で割り切れない"
        )


def _folded_embeddings_forward(
    self: nn.Module, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
) -> torch.Tensor:
    """`SiglipVisionEmbeddings.forward` の同値実装（位置埋め込みを weight の直加算にする）。

    元実装との差は `self.position_embedding(self.position_ids)` を
    `self.position_embedding.weight` へ置き換えた 1 点だけ（先頭の 1 軸は broadcast で同値）。
    `position_ids` が `arange(N)` であることは {@link apply_shape_patches} が実測で確かめる。
    """
    if interpolate_pos_encoding:
        raise NotImplementedError(
            "interpolate_pos_encoding は差し替え版が持たない経路 —"
            " weight の直加算が gather と同値なのは固定解像度のときだけ"
        )
    target_dtype = self.patch_embedding.weight.dtype
    patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
    return patch_embeds.flatten(2).transpose(1, 2) + self.position_embedding.weight


def _map_head_forward(self: nn.Module, hidden_state: torch.Tensor) -> torch.Tensor:
    """`SiglipMultiheadAttentionPoolingHead.forward` の同値実装（q/k/v を明示 linear で持つ）。

    元実装との差は attention の呼び方だけで、残り（residual + layernorm + mlp と先頭行の
    取り出し）は逐語。`nn.MultiheadAttention` の内部と同じく q は probe（長さ 1 の系列）、
    k / v は入力系列から採る。
    """
    batch = hidden_state.shape[0]
    # MUST: `repeat` ではなく `expand` — probe は Parameter なので lifted 定数にならず、
    # `aten.repeat.default` は畳まれずに batch>1 で未対応 op のまま残る（実測）。
    probe = self.probe.expand(batch, -1, -1)
    shape = (batch, -1, self.num_heads, self.head_dim)
    query = self.q_proj(probe).reshape(shape).transpose(1, 2)
    key = self.k_proj(hidden_state).reshape(shape).transpose(1, 2)
    value = self.v_proj(hidden_state).reshape(shape).transpose(1, 2)
    attended = nn.functional.scaled_dot_product_attention(query, key, value)
    attended = attended.transpose(1, 2).reshape(batch, -1, self.num_heads * self.head_dim)
    hidden_state = self.o_proj(attended)

    residual = hidden_state
    hidden_state = self.layernorm(hidden_state)
    hidden_state = residual + self.mlp(hidden_state)
    return hidden_state[:, 0]


def _assert_position_ids_are_arange(embeddings: nn.Module) -> None:
    """`position_ids` が `arange(N)` を `[1, N]` に広げたものであることを実測で確かめる。

    weight の直加算が行 gather と同値なのはこの形のときだけ — 並びが違えば、書き換えは
    通ったまま**別の位置埋め込み**になる（shape は変わらないので何も落ちない）。

    NOTE: `image_size` が `patch_size` で割り切れる必要は無い。行数は上流が
    `(image_size // patch_size) ** 2` で決め、padding='valid' の conv も
    `floor((image_size - patch_size) / patch_size) + 1 = image_size // patch_size` なので、
    端が捨てられても両側が同じ floor を踏む（実例 = so400m の 384 / 14 = 27 グリッド・
    右下 6 画素は上流ごと捨てられる）。行数が食い違えば直加算が shape エラーで落ちる。
    """
    positions = int(embeddings.position_embedding.weight.shape[0])
    expected = torch.arange(positions).reshape(1, positions)
    actual = embeddings.position_ids
    if actual.shape != expected.shape or not torch.equal(actual.cpu(), expected):
        raise ValueError(
            f"position_ids が arange({positions}) の [1, N] 形でない（shape"
            f" {tuple(actual.shape)}）— weight の直加算が gather と同値にならない"
        )


def apply_shape_patches(embeddings: nn.Module) -> None:
    """① conv の文字列 padding と ② 位置埋め込みの畳み込みを当てる（**ビット同一**の段）。

    ① はインスタンス属性、② はクラス属性の差し替え。どちらも演算列が 1 対 1 で対応するので
    pooler_output はビット同一（`export_siglip2.py --verify` の 1 点目が毎回実測する）。
    """
    global _SHAPE_APPLIED
    padding = embeddings.patch_embedding.padding
    if padding != "valid":
        raise ValueError(f"patch_embedding.padding が 'valid' でない（{padding!r}）")
    embeddings.patch_embedding.padding = (0, 0)
    _assert_position_ids_are_arange(embeddings)
    type(embeddings).forward = _folded_embeddings_forward
    _SHAPE_APPLIED = True


def apply_map_head_patch(head: nn.Module) -> None:
    """③ MAP head の `nn.MultiheadAttention` を q/k/v 明示 linear へ割る（**非**ビット同一）。

    パック `in_proj_weight` `[3E, E]` / `in_proj_bias` `[3E]` を E ごとに 3 等分して
    `nn.Linear(E, E)` 3 本に載せ替え、`out_proj` を `o_proj` として引き継ぐ。等分の順序
    （q → k → v）は `nn.MultiheadAttention` の `_in_projection_packed` の逐語。
    """
    global _MAP_HEAD_APPLIED
    attention = head.attention
    if not isinstance(attention, nn.MultiheadAttention):
        raise TypeError(
            f"head.attention が nn.MultiheadAttention でない（{type(attention).__name__}）"
        )
    if attention.in_proj_weight is None or attention.in_proj_bias is None:
        raise ValueError("q/k/v が別々の重みを持つ形は差し替え版が持たない経路")
    if not attention.batch_first:
        raise ValueError("batch_first=False は差し替え版が持たない経路（形が seq-first になる）")
    if attention.bias_k is not None or attention.bias_v is not None or attention.add_zero_attn:
        raise ValueError("add_bias_kv / add_zero_attn は差し替え版が持たない経路（式が変わる）")
    embed_dim = int(attention.embed_dim)
    num_heads = int(attention.num_heads)
    if embed_dim % num_heads:
        raise ValueError(f"embed_dim {embed_dim} が num_heads {num_heads} で割り切れない")

    weight = attention.in_proj_weight.detach()
    bias = attention.in_proj_bias.detach()
    for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
        rows = slice(index * embed_dim, (index + 1) * embed_dim)
        projection = nn.Linear(embed_dim, embed_dim, bias=True)
        projection.weight = nn.Parameter(weight[rows].clone())
        projection.bias = nn.Parameter(bias[rows].clone())
        setattr(head, name, projection)
    head.o_proj = attention.out_proj
    head.num_heads = num_heads
    head.head_dim = embed_dim // num_heads
    del head.attention

    type(head).forward = _map_head_forward
    _MAP_HEAD_APPLIED = True
