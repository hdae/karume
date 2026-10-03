"""Wan2.1 の DiT（diffusers `WanTransformer3DModel`）を S 形の IR にするパッチ層（ADR 0118）。

グラフは Anima と同じ **S 形**（ADR 0034）: 入口は patchify の後・出口は unpatchify の前で、
次元はトークン長 S の 1 シンボルだけ。上流 forward のうち次の段をホストへ出す。

- **patchify / patch 埋め込み**: ホストが潜在 `[1,C,F,H,W]` を `tokens [1,S,C·pt·ph·pw]`（最終次元の
  並びは `(c, pt, ph, pw)`）にする（{@link dit_patchify}）。`Conv3d(k = s = patch)` は窓が重ならない
  ので、重みを平坦化した `Linear` と同じ計算になる（{@link patch_embedding_linear}）。
- **unpatchify**: 出力は `proj_out` の直後 `[1,S,pt·ph·pw·C]`。並びは `(pt, ph, pw, c)` で patchify
  と **違う**（{@link dit_unpatchify} — 上流の rank 8 の permute をグラフに出さない）。
- **時刻埋め込みの sinusoidal**: ホストがグラフ入力 `timesteps_proj [1,256]` にする
  （{@link dit_timesteps_proj}）。MLP（256→1536→1536→9216）はグラフの中。
- **3D RoPE の表**: ホストが軸別の素表（資産 `rope_base` — {@link dit_rope_base_tables}）
  から並べ替えてグラフ入力 `rope_cos` / `rope_sin [1,S,1,head_dim]` にする
  （{@link dit_rope_tables}）。

グラフの中に残る書き換えは 1 つだけで、interleave 形の RoPE の適用を「隣接ペアの入れ替え + cos / sin
の要素積」で rank 4 のまま書く（{@link real_pair_rotary} — irodori の実数化と同じ形）。上流は
`unflatten(-1, (-1, 2)).unbind(-1)` で rank 5 を作り、`out[..., 0::2] = …` の strided 代入で戻す。

adaLN・qk-norm（1536 次元全体の `nn.RMSNorm`）・テキスト文脈（`[1,512,4096]`
をマスク無しで全行に射影し、cross-attn もマスク無し）は**上流のまま**通す（ADR 0118 決定 3 の表）。

MUST: パッチ後の経路は上流と **eager 同値**であること。RoPE の書き換えはビット一致
（`a − b = a + (−b)` と加算の可換 — {@link real_pair_rotary}）。patch 埋め込みの Linear 化は縮約順が
conv3d と違いうるので差を記録する（実測 2026-10-02: torch 2.13 CPU では 8 ケースとも差 0 —
`wan/tests/test_dit_patch.py` が実重みで縮約順の差の上界と突き合わせる）。

MUST: 上流の素の経路を変えない。attn1 の processor を差し替えるが、差し替え後の processor は
{@link RealPairRope} で呼ばれたときだけ実数形を通り、上流の forward（素の tuple の表）が来たら上流の
実装へそのまま委ねる。だから差し替えの前後どちらで参照を採っても参照は上流の値のまま（Anima の VAE
パッチのような「参照は差し替えの前に採る」順序の門は要らない）。

MUST: diffusers は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

#: ホストの軸別素表のテンソルキー（`t → h → w` のブロック順）。Anima の資産 `rope_base` と同じ綴り・
#: 同じ意味（軸 × 位置 × 周波数の cos / sin。値は上流の表そのもの）。
ROPE_BASE_KEYS = ("cos_t", "sin_t", "cos_h", "sin_h", "cos_w", "sin_w")


class RealPairRope(NamedTuple):
    """実数形 RoPE の表（ホストが組んだグラフ入力）。

    `cos` / `sin` は `[1,S,1,head_dim]` で、隣接ペアの 2 成分に同じ値が入る（上流
    `WanRotaryPosEmbed` の `repeat_interleave` 形そのもの）。型で包むのは、差し替えた processor が
    「上流の素の tuple」と区別するため（{@link WanRealPairAttnProcessor}）。
    """

    cos: torch.Tensor
    sin: torch.Tensor


def real_pair_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """interleave 形の RoPE を rank 4 のまま適用する（`x [B,S,H,D]`・表 `[1,S,1,D]`）。

    上流（`WanAttnProcessor.__call__` の `apply_rotary_emb`）は `x1 = x[..., 0::2]`・
    `x2 = x[..., 1::2]` に対し `out[0::2] = x1·cos − x2·sin`・`out[1::2] = x1·sin + x2·cos` を書く
    （表は偶数列の cos と奇数列の sin — 隣接ペアで同じ値）。ここでは同じ `x` に対し

    - `x · cos` が `(x1·c, x2·c)`
    - 隣接ペアを入れ替えて前半を符号反転した `(−x2, x1)` に `sin` を掛けたものが `(−x2·s, x1·s)`

    で、和は `(x1·c − x2·s, x2·c + x1·s)`。f32 の丸めも同じ: 乗算は正確丸めなので
    `(−x2)·s == −(x2·s)`、`a + (−b)` は `a − b` と同じ丸め、`x2·c + x1·s` は加算の可換で
    `x1·s + x2·c` と同じ値になる（**ビット一致**が主張 — `wan/tests/test_dit_patch.py`）。

    MUST: 中間を rank 5 にしない — strided カーネルの rank 上限（`ops.STRIDED_RANK` = 4）に当たる。
    head 軸を潰した `[B,S,H·D/2,2]` で入れ替えてから元形へ戻す（要素順は変わらない）。
    MUST: 長さ 2 の軸の取り出しは長さ 1 の `slice`（`select` は IR 語彙に無い — irodori と同じ）。
    """
    pairs = x.reshape(x.shape[0], x.shape[1], -1, 2)
    swapped = torch.cat([-pairs[..., 1:2], pairs[..., 0:1]], dim=-1).reshape(x.shape)
    return x * cos + swapped * sin


def _processor_base() -> type:
    """上流の `WanAttnProcessor`（diffusers は関数の中で import する — モジュール docstring）。"""
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor

    return WanAttnProcessor


# Third-party code notice. `WanRealPairAttnProcessor.__call__` below is adapted from
# `WanAttnProcessor.__call__` in huggingface/diffusers
# (`src/diffusers/models/transformers/transformer_wan.py`, `diffusers==0.39.0`). It is verbatim
# except for how the rotary embedding is applied (real-pair form) and the image-conditioning branch,
# which is rejected instead of carried over. License: Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Upstream copyright notice, copied verbatim from the
# header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def _real_pair_call(
    self: Any,
    attn: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    rotary_emb: Any = None,
) -> torch.Tensor:
    """`WanAttnProcessor.__call__` の同値実装（RoPE を {@link real_pair_rotary} で掛ける）。

    {@link RealPairRope} 以外の表（上流の素の tuple・None）で呼ばれたら**上流の実装へそのまま委ね
    る**。実数形の経路は self-attention（`encoder_hidden_states is None`）だけで、上流との差は RoPE
    の掛け方 1 点だけ（q / k / v の射影・qk-norm・head 分割・attention の呼び方・出力の射影は逐
    語）。
    """
    base = _processor_base()
    if not isinstance(rotary_emb, RealPairRope):
        return base.__call__(
            self, attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb
        )
    if encoder_hidden_states is not None or attn.add_k_proj is not None:
        raise NotImplementedError(
            "実数形 RoPE は self-attention（image 条件の add_k_proj 無し）だけ"
            " — Wan2.1 T2V の attn1"
        )
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_wan import _get_qkv_projections

    query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    query = real_pair_rotary(query, rotary_emb.cos, rotary_emb.sin)
    key = real_pair_rotary(key, rotary_emb.cos, rotary_emb.sin)

    hidden_states = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        backend=self._attention_backend,
        parallel_config=self._parallel_config,
    )
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.type_as(query)

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


@cache
def _real_pair_processor_class() -> type:
    """`WanAttnProcessor` の派生（`__call__` だけを {@link _real_pair_call} にしたもの）。

    派生にするのは、上流のクラス属性（`_attention_backend` / `_parallel_config`）
    をそのまま引き継ぐため。クラスは diffusers を import してから作る（モジュール docstring の
    MUST）。1 プロセスで 1 つだけ作る（差し替えの冪等の判定がクラスの同一性で済むように）。
    """
    base = _processor_base()
    return type("WanRealPairAttnProcessor", (base,), {"__call__": _real_pair_call})


def install_real_pair_processors(blocks: nn.ModuleList) -> None:
    """各ブロックの attn1（self-attention）の processor を実数形 RoPE の版へ差し替える（冪等）。

    差し替えはインスタンス単位（クラス属性には触らない）。差し替え後も上流の素の forward
    は上流の値を返す（{@link _real_pair_call} の委譲）。
    """
    processor_class = _real_pair_processor_class()
    for block in blocks:
        if type(block.attn1.processor) is not processor_class:
            block.attn1.set_processor(processor_class())


def patch_embedding_linear(conv: nn.Conv3d) -> nn.Linear:
    """窓が重ならない `Conv3d(k = s = patch)` と同じ計算の `Linear(C·pt·ph·pw → Cout)`。

    重みは conv の重み `[Cout, C, pt, ph, pw]` を `[Cout, C·pt·ph·pw]` に平坦化した
    **view**（記憶域を共有する）で、bias は同じ Parameter。共有にするのは、f16 の丸め（fake-quant —
    ADR 0006）を S 形のラッパへ掛けたときに、参照を採る上流の conv にも同じ丸めが届くようにするため
    （別の複製だと参照だけが元の重みで計算され、差に量子化誤差が混ざる）。

    入力の並び `(c, pt, ph, pw)` は平坦化の C 順そのもの（{@link dit_patchify}）。
    MUST: k = s・padding 0・dilation 1・groups 1 以外は fail loudly（窓が重なる / 欠ける conv は
    reshape + Linear で書けない）。
    """
    kernel = tuple(int(size) for size in conv.kernel_size)
    if (
        tuple(int(size) for size in conv.stride) != kernel
        or any(int(size) != 0 for size in conv.padding)
        or any(int(size) != 1 for size in conv.dilation)
        or conv.groups != 1
        or conv.padding_mode != "zeros"
    ):
        raise NotImplementedError(
            f"patch 埋め込みの Conv3d が k = s・padding 0・dilation 1・groups 1 の外"
            f"（k={kernel} s={tuple(conv.stride)} p={conv.padding} d={conv.dilation}"
            f" g={conv.groups}）— Linear と同じ計算にならない"
        )
    weight = conv.weight
    if not weight.is_contiguous():
        raise NotImplementedError("patch 埋め込みの重みが連続でない（平坦化 view が作れない）")
    out_channels = int(weight.shape[0])
    linear = nn.Linear(int(weight[0].numel()), out_channels, bias=conv.bias is not None)
    linear.weight = nn.Parameter(weight.view(out_channels, -1), requires_grad=False)
    if conv.bias is not None:
        linear.bias = conv.bias
    return linear


def _assert_supported_config(model: nn.Module) -> None:
    """S 形のラッパが前提にする構成（Wan2.1 T2V）を確かめる。外れたら fail loudly。"""
    config = model.config
    if config.image_dim is not None or config.added_kv_proj_dim is not None:
        raise NotImplementedError("image 条件（I2V）の構成は未対応 — Wan2.1 T2V だけ")
    if config.pos_embed_seq_len is not None:
        raise NotImplementedError("pos_embed_seq_len 付きの構成は未対応")
    if config.qk_norm != "rms_norm_across_heads":
        raise NotImplementedError(
            f"qk_norm={config.qk_norm} は未対応（rms_norm_across_heads だけ）"
        )


# Third-party code notice. `WanDitTokens.forward` below is adapted from
# `WanTransformer3DModel.forward` and `WanTimeTextImageEmbedding.forward` in huggingface/diffusers
# (`src/diffusers/models/transformers/transformer_wan.py`, `diffusers==0.39.0`). The block loop, the
# condition embedding (minus the sinusoidal step) and the output norm / projection are verbatim; the
# patchify, the sinusoidal timestep projection, the RoPE table construction and the unpatchify were
# moved to the host. License: Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Upstream copyright notice, copied verbatim from the
# header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
class WanDitTokens(nn.Module):
    """`WanTransformer3DModel` の 1 forward の**トークン形**（S 形グラフ — ADR 0118 決定 3）。

    上流 forward との差は入口と出口だけで、30 ブロックの本体は上流のモジュールをそのまま呼ぶ:

    - 入力が `hidden_states [1,C,F,H,W]` ではなく **`tokens [1,S,C·pt·ph·pw]`**（ホストが patchify
      したもの — {@link dit_patchify}）。patch 埋め込みは平坦化した Linear
      （{@link patch_embedding_linear}）
    - 時刻は整数の `timestep` ではなく **`timesteps_proj [1,freq_dim]`**（ホストの sinusoidal —
      {@link dit_timesteps_proj}）
    - RoPE の cos / sin 表を**グラフ入力**にする（`[1,S,1,head_dim]` — ホストが素表から組む）。
      attn1 の processor は実数形の版（{@link install_real_pair_processors}）
    - 出力は unpatchify **前**のトークン `[1,S,pt·ph·pw·C]`（逆並べ替えはホスト —
      {@link dit_unpatchify}）

    結果としてグラフ内に F / H / W が 1 つも現れず、トークン長 1 シンボル `S` だけで書ける。

    `encoder_hidden_states` は umT5 の出力を有効長で切ってからゼロで 512 行まで埋めた `[1,512,4096]`
    （ホスト — `wan.pipeline_ref.pad_text_embeds`）。`text_embedder` は 512 行すべてに掛かり、
    cross-attn はマスク無し（上流と同じ — パディング位置は射影後に非ゼロの定数ベクトルになる）。

    保持するのは上流の部品への参照だけ（`patch_embedding` の Linear は conv の重みの view）。`rope`
    は持たない — 表はグラフ入力なので、`rope` のバッファ（float64 から落とした f32 の表）
    を初期値として運ばない。

    `blocks` を渡すと、その部分列だけを回す（層数に対する誤差の伸びを測る切り詰め用。既定は全層）。
    """

    def __init__(self, model: nn.Module, blocks: nn.ModuleList | None = None) -> None:
        super().__init__()
        _assert_supported_config(model)
        self.patch_size = tuple(int(size) for size in model.config.patch_size)
        self.patch_embedding = patch_embedding_linear(model.patch_embedding)
        self.condition_embedder = model.condition_embedder
        self.blocks = model.blocks if blocks is None else blocks
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out
        self.scale_shift_table = model.scale_shift_table
        install_real_pair_processors(self.blocks)

    def forward(
        self,
        tokens: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_hidden(
            self.patch_embedding(tokens),
            timesteps_proj,
            encoder_hidden_states,
            rope_cos,
            rope_sin,
        )

    def forward_hidden(
        self,
        hidden_states: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        """patch 埋め込みの**後**から（`hidden_states [1,S,dim]`）。

        {@link forward} との差は patch 埋め込み 1 段だけ。上流の conv3d の出力をここへ入れると、
        上流の forward と**ビット一致**する（patch 埋め込み以外の書き換えがビット一致であることの実
        測口 — `wan/tests/test_dit_patch.py`）。
        """
        output, _ = self._trunk(
            hidden_states, timesteps_proj, encoder_hidden_states, rope_cos, rope_sin, collect=False
        )
        return output

    def _trunk(
        self,
        hidden_states: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        *,
        collect: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """本体（条件の埋め込み → ブロック列 → 出力の norm / 射影）。

        `collect` で各ブロックの出力も返す。
        """
        embedder = self.condition_embedder
        # 上流 `WanTimeTextImageEmbedding.forward`（sinusoidal の段だけがホストへ出た）。
        time_embedder_dtype = next(iter(embedder.time_embedder.parameters())).dtype
        if timesteps_proj.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timesteps_proj = timesteps_proj.to(time_embedder_dtype)
        temb = embedder.time_embedder(timesteps_proj).type_as(encoder_hidden_states)
        timestep_proj = embedder.time_proj(embedder.act_fn(temb))
        encoder_hidden_states = embedder.text_embedder(encoder_hidden_states)

        # batch_size, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        rotary_emb = RealPairRope(rope_cos, rope_sin)
        collected: list[torch.Tensor] = []
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            if collect:
                collected.append(hidden_states)

        # batch_size, inner_dim
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(
            hidden_states
        )
        return self.proj_out(hidden_states), collected


class WanDitTokensLayers(WanDitTokens):
    """{@link WanDitTokens} に**層別の出口を足した**計測用の形（層数に対する誤差の伸び — 決定 8）。

    出力は各ブロックの出力 `[1,S,dim]`（ブロック順）と、最後に {@link WanDitTokens} と同じ最終出力
    `[1,S,pt·ph·pw·C]`。製品のグラフではない（配布しない）— 入力も重みも同じなので、同じ golden
    の入力で回して、上流の各ブロックの出力（forward hook で採った値）と層ごとに突き合わせる。
    """

    def forward(  # type: ignore[override]
        self,
        tokens: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        output, collected = self._trunk(
            self.patch_embedding(tokens),
            timesteps_proj,
            encoder_hidden_states,
            rope_cos,
            rope_sin,
            collect=True,
        )
        return (*collected, output)


# ---- ホストへ出した段の Python 側の正本（TS の鏡像の突き合わせ相手） ----------------


def dit_patchify(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """`latents [1,C,F,H,W]` → `tokens [1,S,C·pt·ph·pw]`（最終次元は `(c, pt, ph, pw)`）。

    トークン添字は `(f·H' + h)·W' + w`（上流の `conv3d(…).flatten(2).transpose(1, 2)` の並び）、
    最終次元は conv3d の重み `[Cout, C, pt, ph, pw]` の平坦化と同じ C 順
    （{@link patch_embedding_linear} と対）。
    """
    patch_t, patch_h, patch_w = patch_size
    batch, channels, frames, height, width = latents.shape
    patches = latents.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    return patches.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)


# Third-party code notice. The body of `dit_unpatchify` below reproduces the unpatchify lines of
# `WanTransformer3DModel.forward` in huggingface/diffusers (`transformer_wan.py`,
# `diffusers==0.39.0`) verbatim. License: Apache License, Version 2.0. Upstream copyright notice,
# copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def dit_unpatchify(
    tokens: torch.Tensor,
    latent_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """`tokens [1,S,pt·ph·pw·C]` → `latents [1,C,F,H,W]`（上流の出口の逐語）。

    MUST: **patchify の逆順ではない** — 最終次元の並びは `(pt, ph, pw, c)` で、patchify 側の
    `(c, pt, ph, pw)` と別物（上流 `reshape(…, p_t, p_h, p_w, -1)` →
    `permute(0, 7, 1, 4, 2, 5, 3, 6)`）。
    """
    frames, height, width = latent_shape
    p_t, p_h, p_w = patch_size
    batch_size = tokens.shape[0]
    post_patch_num_frames = frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    hidden_states = tokens.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def dit_timesteps_proj(model: nn.Module, timestep: torch.Tensor) -> torch.Tensor:
    """ホストで計算する時刻の sinusoidal（上流 `condition_embedder.timesteps_proj` そのもの）。

    `timestep` は上流のパイプラインが渡す整数の timestep（`[1]`・int64）。式は写さず上流のモジュール
    を呼ぶ（`Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)` — cos 先・sin 後）。
    """
    return model.condition_embedder.timesteps_proj(timestep)


def dit_rope_tables(
    rope: nn.Module, latent_shape: tuple[int, int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """S 形グラフへ渡す RoPE の cos / sin 表 `[1,S,1,head_dim]`（上流 `WanRotaryPosEmbed` の出力）。

    `rope` は上流の `model.rope`、`latent_shape` は潜在の `(F, H, W)`。上流の `rope` は入力の形しか
    見ないので、ゼロの probe を渡す。
    MUST: MPS が見える機では呼ばない（表が float32 で作られて機に依って変わる —
    `pipeline_ref.assert_no_mps` が組む前に止める）。
    """
    probe = torch.zeros(1, 1, *latent_shape)
    return rope(probe)


def dit_rope_base_tables(rope: nn.Module) -> dict[str, torch.Tensor]:
    """ホスト（TS）が RoPE の表を組むための**軸別の素表**を上流の表（`model.rope`）から切り出す。

    返すのは `cos_t` / `sin_t` / `cos_h` / … の 6 本で、各 `[rows, 軸のブロック幅 / 2]`（行 =
    位置・列 = その軸の周波数）。値は上流 `model.rope` の出力そのもので、上流の表は隣接ペアに同じ値
    を `repeat_interleave` した形なので、偶数列だけを取れば素表になる（奇数列が偶数列と同じことは
    {@link _assert_rope_base} が確かめる）。

    ## なぜ表を焼くのか

    上流は float64 で角度と三角関数を計算してから f32 に落とす（MPS の無い機 —
    `transformer_wan.py:375`）。TS で式を写すと、`pow` / 除算 / `cos` の f64
    の実装差で最終ビットが割れうる。素表を焼いてホストは並べ替えだけをすれば、
    表は上流とビット同一になる（Anima の `rope-base.ts` と同じ判断 — TS で三角関数を計算しない
    MUST）。

    ## 行数

    上流の位置表の天井 `rope_max_seq_len`（1024）そのもの。F' / H' / W' のどれかがこれを超える潜在は
    上流でも組めない。

    MUST: 切り出しは `model.rope` の**出力から**行う（式を写さない）。1 軸だけを `rows`
    位置ぶん動かし、他の 2 軸を位置 0 に固定した probe を軸ごとに 1 本ずつ回す。
    """
    rows = int(rope.max_seq_len)
    patch_t, patch_h, patch_w = (int(size) for size in rope.patch_size)
    dims = (int(rope.t_dim), int(rope.h_dim), int(rope.w_dim))
    starts = (0, dims[0], dims[0] + dims[1])

    def probe(frames: int, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(ppf, pph, ppw) = (frames, height, width)` になる形で上流の rope をそのまま呼ぶ。"""
        sample = torch.zeros(1, 1, frames * patch_t, height * patch_h, width * patch_w)
        cos, sin = rope(sample)
        return cos[0, :, 0], sin[0, :, 0]

    tables: dict[str, torch.Tensor] = {}
    for axis, (frames, height, width), start, dim in zip(
        ("t", "h", "w"),
        ((rows, 1, 1), (1, rows, 1), (1, 1, rows)),
        starts,
        dims,
        strict=True,
    ):
        cos, sin = probe(frames, height, width)
        block_cos = cos[:, start : start + dim]
        block_sin = sin[:, start : start + dim]
        if not (
            torch.equal(block_cos[:, 0::2], block_cos[:, 1::2])
            and torch.equal(block_sin[:, 0::2], block_sin[:, 1::2])
        ):
            raise AssertionError(
                f"上流の RoPE 表の {axis} ブロックが隣接ペアで同じ値になっていない"
                "（repeat_interleave 形の前提が崩れた — 偶数列を素表にできない）"
            )
        tables[f"cos_{axis}"] = block_cos[:, 0::2].contiguous()
        tables[f"sin_{axis}"] = block_sin[:, 0::2].contiguous()
    _assert_rope_base(tables, rows, tuple(dim // 2 for dim in dims))
    return tables


def _assert_rope_base(tables: dict[str, torch.Tensor], rows: int, widths: tuple[int, ...]) -> None:
    """素表の切り出しが上流のブロック境界と合っていることを確かめる（恒真化の門）。

    切り出しが 1 ブロックずれても shape は合う（h と w のブロック幅は等しい）。
    そこで**その軸だけが位置で動く**ことを見る: 位置 0 の行は角度 0 なので cos = 1 / sin = 0、位置
    1 の行は sin が 0 でない。別のブロックを切っていれば位置に対して定数のままになり、後者で破れる。
    """
    if set(tables) != set(ROPE_BASE_KEYS):
        raise AssertionError(
            f"rope 素表のキーが {sorted(tables)}（期待は {sorted(ROPE_BASE_KEYS)}）"
        )
    for axis, width in zip(("t", "h", "w"), widths, strict=True):
        for kind in ("cos", "sin"):
            table = tables[f"{kind}_{axis}"]
            if table.dtype != torch.float32 or tuple(table.shape) != (rows, width):
                raise AssertionError(
                    f"rope 素表 {kind}_{axis} が {table.dtype} {tuple(table.shape)}"
                    f"（期待は float32 {(rows, width)}）"
                )
        if not torch.equal(tables[f"cos_{axis}"][0], torch.ones(width)):
            raise AssertionError(f"rope 素表 cos_{axis} の位置 0 が全 1 でない（切り出しずれ）")
        if not torch.equal(tables[f"sin_{axis}"][0], torch.zeros(width)):
            raise AssertionError(f"rope 素表 sin_{axis} の位置 0 が全 0 でない（切り出しずれ）")
        if torch.equal(tables[f"sin_{axis}"][1], torch.zeros(width)):
            raise AssertionError(
                f"rope 素表 sin_{axis} が位置 1 でも全 0（この軸で動いていない = 切り出しずれ）"
            )


@contextmanager
def flash_attention_only() -> Iterator[None]:
    """この間の SDPA を CPU の flash 経路（`aten._scaled_dot_product_flash_attention_for_cpu`）に
    固定する（CPU の参照とパッチ後の eager — export のトレースには掛けない）。

    WHY: 実寸 81 フレーム（S = 32,760）の参照が MATH backend に落ちると、スコア行列
    `[1,12,S,S]` 1 枚で 51.5 GB（f32）/ 103 GB（f64）を確保して OOM で死ぬ（それまでの数十分の
    計算ごと）。torch 2.13.0+cpu は既定で flash を選ぶ（`torch._fused_sdp_choice` —
    `wan/tests/test_dit_patch.py` が見る）ので、固定しても数値は変わらない。固定しておけば、flash が
    受けない入力（最終次元が非連続など）は MATH へ黙って落ちずに RuntimeError で止まる。
    """
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        yield


def reference_dit(
    model: nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """**上流の素の** `WanTransformer3DModel.forward` の出力 `[1,C,F,H,W]`（参照値）。

    attention は {@link flash_attention_only} の下で回す（f32 / f64 の参照とも）。
    """
    with flash_attention_only():
        return model(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


def reference_dit_layers(
    model: nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """{@link reference_dit} の出力と、上流の各ブロックの出力 `[1,S,dim]`（forward hook で採る）。

    hook は呼び出しの間だけ張る（S 形のラッパも同じブロックを回すので、張りっぱなしにすると向こうの
    呼び出しまで拾う）。
    """
    collected: list[torch.Tensor] = []

    def capture(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        collected.append(output.detach().clone())

    handles = [block.register_forward_hook(capture) for block in model.blocks]
    try:
        output = reference_dit(model, latents, timestep, encoder_hidden_states)
    finally:
        for handle in handles:
            handle.remove()
    if len(collected) != len(model.blocks):
        raise AssertionError(
            f"ブロックの出力を {len(collected)} 本拾った（{len(model.blocks)} 本のはず）"
        )
    return output, collected


class _NarrowFloatWatch(TorchDispatchMode):
    """f64 の forward の中で、f64 でない浮動小数の値（要素数が `limit` を超えるもの）を記録する。"""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit
        self.found: list[str] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        result = func(*args, **(kwargs or {}))
        for leaf in tree_leaves(result):
            if (
                isinstance(leaf, torch.Tensor)
                and leaf.is_floating_point()
                and leaf.dtype != torch.float64
                and leaf.numel() > self.limit
            ):
                self.found.append(f"{func}: {leaf.dtype} {tuple(leaf.shape)}")
        return result


@contextmanager
def _float_keeps_float64() -> Iterator[None]:
    """この間だけ `Tensor.float()` を f64 のテンソルでは素通しにする（f64 以外は元のまま）。

    WHY: 上流は FP32LayerNorm と block の変調・残差を `.float()` で f32 へ寄せる（bf16 / f16 で
    回すときに norm を f32 で計算するための寄せ）。f32 のモデルではどれも恒等で計算の意味に含まれ
    ないが、f64 のモデルでそのまま通すと norm と残差だけが f32 に落ち、「活性を f64 で回した参照」に
    ならない。
    """
    original = vars(torch.Tensor).get("float")
    narrow = torch.Tensor.float

    def keep_float64(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if tensor.dtype == torch.float64:
            return tensor
        return narrow(tensor, *args, **kwargs)

    torch.Tensor.float = keep_float64  # type: ignore[method-assign]
    try:
        yield
    finally:
        if original is None:
            del torch.Tensor.float
        else:
            torch.Tensor.float = original  # type: ignore[method-assign]


@contextmanager
def float64_forward(model: nn.Module) -> Iterator[None]:
    """この間の上流の forward を**活性も f64** で回す（ADR 0118 段 3 の誤差の帰属・実寸の参照）。

    `model` は呼び手が `double()` しておく（重みは f16 へ丸めた値をそのまま f64 に広げる — GPU と
    同じ重み）。入力（f32 の潜在・テキスト文脈）も呼び手が値を変えずに f64 へ広げる。上流の素の
    forward を回すのは f32 の参照と同じで、違いは次の 2 点だけ:

    - 上流の `.float()` の寄せを f64 では素通しにする（{@link _float_keeps_float64}）。
    - f64 でない浮動小数の値が作られたら抜けるときに止める。例外は時刻の sinusoid（上流の
      `Timesteps` が f32 で組む `[1,freq_dim]` — GPU にも同じ f32 の値が `timesteps_proj` として
      入り、`time_embedder` の入口で f64 へ広がる）だけで、要素数 `freq_dim` 以下として許す。
    """
    narrow = sorted(
        {str(tensor.dtype) for tensor in (*model.parameters(), *model.buffers())}
        - {str(torch.float64)}
    )
    if narrow:
        raise AssertionError(f"f64 の参照なのにモデルに {narrow} の重み / バッファが残っている")
    watch = _NarrowFloatWatch(int(model.config.freq_dim))
    with _float_keeps_float64(), watch:
        yield
    if watch.found:
        raise AssertionError(f"f64 の参照の中で f64 でない値が作られた: {watch.found[:5]}")


def reference_dit_f64(
    model: nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """{@link reference_dit} を {@link float64_forward} の下で回した出力 `[1,C,F,H,W]`（f64 のまま —
    丸めるかどうかは格納する側が決める）。"""
    with float64_forward(model):
        return reference_dit(model, latents.double(), timestep, encoder_hidden_states.double())
