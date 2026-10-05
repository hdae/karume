"""Wan の動画 VAE decoder（2.1 と 2.2）を **chunk グラフ**（unbatched の rank 4）へ書き直す
（ADR 0118・ADR 0121 決定 6）。

上流 `AutoencoderKLWan`（diffusers 0.39.0 `autoencoder_kl_wan.py`）の decode は、潜在 1 フレームずつ
decoder を回し、因果キャッシュ（feat_cache）を Python のリストと番兵 `'Rep'` で持ち回す。これを
「潜在 1 フレーム + cache テンソル群 → フレーム + 更新後の cache」の静的なグラフ 2 種
（最初の chunk 用 = {@link WanVaeChunkDecoder} の `first=True`・それ以降用 = `first=False`）に
組み替える。書き直しは 6 つで、どれも **上流のモジュールの重みをそのまま読む関数**として書く:

1. **cache の正規化 + 因果パディングの cat 化**（{@link causal_conv3d}）— cache は常に 2 フレーム・
   初期値ゼロ・更新は `cat(cache, x)` の末尾 2 フレーム。時間カーネル 3 の CausalConv3d は
   どの chunk でも `conv3d(cat(cache, x))`（時間 padding 0・空間 padding は attrs）になる。
2. **RMS_norm の rank 4 化**（{@link rms_norm}）— `F.normalize` × √C × gamma を `sum` → `sqrt` →
   `clamp_min` → 除算で書く（`linalg_vector_norm` を語彙に入れない — ADR 0017）。
3. **upsample3d の時間インターリーブの rank 4 化**（{@link _shared.vae_rank4.interleave_frames}）—
   上流の rank 6（`reshape(b,2,c,t,h,w)` → `stack(dim=3)`）を rank 4 の reshape / permute で書く。
4. **フレームごとの 2D 処理を (b t) に畳む**（{@link resample} / {@link attention_block}）—
   `[C,T,H,W]` → permute → `[T,C,H,W]` で、フレームを conv2d / attention のバッチに置く。
5. **f32 の normalize**: diffusers は fp16 / bf16 入力のとき normalize を f32 で行う
   （`autoencoder_kl_wan.py:202-210`）。karume の活性は常に f32（格納だけを圧縮 — ADR 0006）なので
   この分岐は常に「f32 のまま」側で、書き直しは要らない（{@link rms_norm} は f32 以外を拒む）。
6. **残差の外側のショートカットの rank 4 化**（Wan2.2 の `DupUp3D` — {@link avg_shortcut} →
   {@link _shared.vae_rank4.dup_up_3d}）— 上流の rank 8 の view / permute を、複製と軸ごとの
   depth-to-space（各段 rank 4）で書く。最初の chunk で時間の先頭 `ft − 1` 枚を捨てる slice は
   cache の仕組みの側（ここ）に置く。

受ける世代は {@link SUPPORTED_GENERATIONS} の 2 つに閉じる: Wan2.1（`WanUpBlock`・出口は画素の
3 ch）と Wan2.2（`WanResidualUpBlock`・出口は patchify 空間の 12 ch）。2.2 の出口は unpatchify
せずに返す（unpatchify と clamp はホスト — ADR 0121 決定 6）。2.1 の分岐は 2.2 の部品を通らない
（{@link WanVaeChunkDecoder.forward}）。

cache の正規化が上流と値で一致する根拠（決定 2 の導出）: 上流が `F.pad` で時間の先頭に詰める
ゼロと、ここで cache に置くゼロは同じ値なので、各 conv3d の入力テンソルは要素ごとに上流と同一に
なる（潜在解像度の conv なら上流 `[0,0,x0]` → `[0,x0,x1]` → `[x0,x1,x2]`・こちらは cache が
`[0,0]` → `[0,x0]` → `[x0,x1]` と進んで同じ並び）。`time_conv` は最初の chunk で走らない
（上流の `'Rep'`）ので、その cache はゼロのまま残り、chunk 2 が `[0,0,x1]` を読む — 上流の `'Rep'`
の経路（ゼロ 2 枚の pad）と同じ。この同値は {@link normalized_cache_decode} が**上流のコードの
まま** cache だけを正規化した形で実測する（2.1 は `tests/test_vae_patch.py`・2.2 は
`tests/test_vae_patch_ti2v.py`）。

テンソルは unbatched の `[C, T, H, W]`（B = 1 を落とす — 決定 2）。conv3d の重みだけが rank 5 で、
値は全て rank 4 以下に収まる（strided コピー族の上限 — ADR 0011 / 0014 / 0016）。

MUST: モジュールの差し替え（クラス属性の monkeypatch）はしない。関数は上流モジュールを**読むだけ**
で、パッチ前の参照（{@link reference_decode_unclamped}）はいつでも採れる — anima の VAE パッチが
持つ「適用済みフラグ」の門（パッチ後に参照を採ると同値検証が恒真化する）が要らない形にした。

MUST: diffusers は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional

from _shared.vae_rank4 import (
    UPSAMPLE_SCALE,
    dup_up_3d,
    interleave_frames,
    l2_normalize,
    nearest_exact_2x,
)

if TYPE_CHECKING:
    from diffusers import AutoencoderKLWan

#: 各 CausalConv3d が次の chunk へ持ち越すフレーム数（上流の `CACHE_T` —
#: `autoencoder_kl_wan.py:31`）。
CACHE_FRAMES = 2

#: cache の入力の名前の接頭辞（グラフ入力は `cache_00` … — dict 入力を `torch.export` が平らにした
#: 名前）。
CACHE_INPUT = "cache"

#: 書き直しが受ける世代の閉じた表: 上流 config の `(is_residual, patch_size)` → up block の型名
#: （`diffusers.models.autoencoders.autoencoder_kl_wan` の名前 — diffusers は関数の中で
#: import する）。
#: Wan2.1 は `WanUpBlock`、Wan2.2 は `WanResidualUpBlock`（`avg_shortcut` の DupUp3D を持ち、
#: upsampler は dim → dim の 1 本）。表に無い組（`(True, None)`・`(False, 2)`・patch 4 など）は、
#: ショートカットの有無と出口の並びの前提を確かめていないので fail loudly。
SUPPORTED_GENERATIONS: dict[tuple[bool, int | None], str] = {
    (False, None): "WanUpBlock",
    (True, 2): "WanResidualUpBlock",
}

#: 出口の画素のチャネル数（RGB）。patchify する世代は `× patch_size²` のチャネル（patchify 空間 —
#: 2.2 は 12）のまま返す。
SAMPLE_CHANNELS = 3

#: upsampler の mode → 時間の倍率（upsample3d は `time_conv` の時間インターリーブで 2 倍）。
TIME_FACTORS = {"upsample3d": 2, "upsample2d": 1}


class UnsupportedVaeError(NotImplementedError):
    """書き直しが前提とする構成（{@link SUPPORTED_GENERATIONS} の decoder）から外れた VAE を
    受け取った。"""


# ---- cache の表 ------------------------------------------------------------------


@dataclass(frozen=True)
class CacheSlot:
    """因果キャッシュ 1 本（時間カーネル 3 の CausalConv3d 1 つに対応）。

    `index` は上流の `feat_idx` の順（decoder の forward で CausalConv3d を通る順）で、名前と
    グラフの入出力の並びはこの順に揃える。`scale` は潜在タイルの辺に対する空間倍率
    （up block の nearest ×2 を通るたびに 2 倍）。
    """

    index: int
    channels: int
    scale: int
    #: upsample3d の `time_conv`（最初の chunk では走らない — 上流の `'Rep'`）。
    time_conv: bool

    @property
    def key(self) -> str:
        """dict 入力のキー（グラフ入力名は `cache_<key>`）。"""
        return f"{self.index:02d}"

    @property
    def name(self) -> str:
        """グラフ入力名（`torch.export` が dict 入力 `cache` を平らにした綴り）。"""
        return f"{CACHE_INPUT}_{self.key}"

    def shape(self, tile: int) -> tuple[int, int, int, int]:
        """潜在タイル辺 `tile` のときの形 `[C, 2, h, w]`。"""
        side = tile * self.scale
        return (self.channels, CACHE_FRAMES, side, side)


def assert_supported(vae: AutoencoderKLWan) -> None:
    """書き直しが前提とする decoder の構成かを確かめる（外れたら fail loudly）。

    見るのは書き直しが**構造として**仮定している点だけ: 世代が {@link SUPPORTED_GENERATIONS} の
    表にあり、up block が全てその世代の型であること・mid の attention が 1 本であること・出口の
    チャネル数が `3 × patch_size²` であること。Wan2.2 は加えて、各 up block のショートカットと
    upsampler の有無が揃い、ショートカットの倍率が upsampler と同じ（空間 ×2・時間は upsample3d なら
    ×2・upsample2d なら ×1）であること — 最初の chunk 1 枚・以降 4 枚の取り決めは、残差の経路と
    ショートカットの時間の倍率が揃っていることに依る。
    """
    from diffusers.models.autoencoders import autoencoder_kl_wan

    config = vae.config
    generation = (bool(config.is_residual), config.patch_size)
    if generation not in SUPPORTED_GENERATIONS:
        raise UnsupportedVaeError(
            f"(is_residual, patch_size) = {generation} は未対応"
            f"（受けるのは {list(SUPPORTED_GENERATIONS)} だけ）"
        )
    decoder = vae.decoder
    if len(decoder.mid_block.attentions) != 1 or len(decoder.mid_block.resnets) != 2:
        raise UnsupportedVaeError("mid block は Res → Attention → Res の 1 層だけを書いている")
    up_block_type = getattr(autoencoder_kl_wan, SUPPORTED_GENERATIONS[generation])
    for block in decoder.up_blocks:
        if not isinstance(block, up_block_type):
            raise UnsupportedVaeError(
                f"up block の型 {type(block).__name__} は世代 {generation} の"
                f" {up_block_type.__name__} でない"
            )
        if config.is_residual:
            _assert_shortcut_follows_the_upsampler(block)
    channels = SAMPLE_CHANNELS * (config.patch_size or 1) ** 2
    if decoder.conv_out.out_channels != channels:
        raise UnsupportedVaeError(
            f"出口のチャネル {decoder.conv_out.out_channels} が"
            f" {SAMPLE_CHANNELS} × patch_size² = {channels} でない"
        )


def _assert_shortcut_follows_the_upsampler(block: nn.Module) -> None:
    """`WanResidualUpBlock` のショートカット（DupUp3D）が upsampler と同じ倍率で同じ有無か。"""
    shortcut, upsampler = block.avg_shortcut, block.upsampler
    if (shortcut is None) != (upsampler is None):
        raise UnsupportedVaeError("up block のショートカットと upsampler の有無が揃っていない")
    if shortcut is None:
        return
    expected = (TIME_FACTORS.get(upsampler.mode), UPSAMPLE_SCALE)
    if (shortcut.factor_t, shortcut.factor_s) != expected:
        raise UnsupportedVaeError(
            f"ショートカットの倍率 (時間 {shortcut.factor_t}, 空間 {shortcut.factor_s}) が"
            f" upsampler（{upsampler.mode}）の {expected} と違う"
        )


def _upsampler_of(block: nn.Module) -> nn.Module | None:
    """up block の upsampler（`WanResidualUpBlock` は `upsampler` 1 本・`WanUpBlock` は
    `upsamplers[0]`・無ければ None）。"""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualUpBlock, WanUpBlock

    if isinstance(block, WanResidualUpBlock):
        return block.upsampler
    if isinstance(block, WanUpBlock):
        return None if block.upsamplers is None else block.upsamplers[0]
    raise UnsupportedVaeError(f"up block の型 {type(block).__name__} は未対応")


def cache_slots(vae: AutoencoderKLWan) -> tuple[CacheSlot, ...]:
    """decoder の forward 順に因果キャッシュの表を作る（上流の `feat_idx` と同じ順）。

    実モジュールを歩いて導く（表を手で持たない）: conv_in → mid の Res ×2 → 各 up block の
    Res ×3 と upsample3d の `time_conv` → head。`shortcut` と post-quant の 1×1×1 は cache を
    持たない（時間 padding 0 — 決定 2）。Wan2.2 のショートカット（DupUp3D）もデータ移動だけで
    cache を持たない。
    """
    assert_supported(vae)
    decoder = vae.decoder
    slots: list[CacheSlot] = []
    scale = 1

    def add(conv: nn.Module, *, time_conv: bool = False) -> None:
        _assert_causal(conv)
        slots.append(CacheSlot(len(slots), int(conv.in_channels), scale, time_conv))

    add(decoder.conv_in)
    for resnet in decoder.mid_block.resnets:
        add(resnet.conv1)
        add(resnet.conv2)
    for block in decoder.up_blocks:
        for resnet in block.resnets:
            add(resnet.conv1)
            add(resnet.conv2)
        upsampler = _upsampler_of(block)
        if upsampler is not None:
            if upsampler.mode == "upsample3d":
                add(upsampler.time_conv, time_conv=True)
            elif upsampler.mode != "upsample2d":
                raise UnsupportedVaeError(f"decoder に現れない resample mode: {upsampler.mode}")
            scale *= UPSAMPLE_SCALE
    add(decoder.conv_out)
    return tuple(slots)


def graph_slots(slots: Sequence[CacheSlot], *, first: bool) -> tuple[CacheSlot, ...]:
    """そのグラフが入出力に持つ cache（first は `time_conv` の 2 本を持たない — 決定 2 の表）。"""
    return tuple(slot for slot in slots if not (first and slot.time_conv))


def zero_caches(slots: Sequence[CacheSlot], tile: int, *, first: bool) -> dict[str, torch.Tensor]:
    """そのグラフの cache 入力をゼロで揃えた dict（キーの順 = グラフ入力の順）。"""
    return {slot.key: torch.zeros(slot.shape(tile)) for slot in graph_slots(slots, first=first)}


# ---- 書き直しの部品（上流モジュールを読むだけ）------------------------------------


def _assert_causal(conv: nn.Module) -> None:
    """時間カーネル 3・因果パディング 2・stride / dilation 1 の CausalConv3d か。"""
    pad_w, pad_w_after, pad_h, pad_h_after, pad_t, pad_t_after = conv._padding
    kernel_t = int(conv.weight.shape[2])
    if pad_t != CACHE_FRAMES or pad_t_after != 0 or kernel_t != CACHE_FRAMES + 1:
        raise UnsupportedVaeError(
            f"因果パディング {conv._padding}・時間カーネル {kernel_t} は未対応"
            f"（時間の先頭に {CACHE_FRAMES} 枚・カーネル {CACHE_FRAMES + 1} だけ）"
        )
    if pad_w != pad_w_after or pad_h != pad_h_after:
        raise UnsupportedVaeError(
            f"空間の非対称 padding {conv._padding} は conv3d の attrs に畳めない"
        )
    if tuple(conv.stride) != (1, 1, 1) or tuple(conv.dilation) != (1, 1, 1):
        raise UnsupportedVaeError(
            f"stride {conv.stride}・dilation {conv.dilation} は未対応（1 だけ）"
        )


def causal_conv3d(
    conv: nn.Module, x: torch.Tensor, cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """時間カーネル 3 の CausalConv3d を「cache を時間の先頭に cat → 時間 padding 0 の conv3d」
    で回す。

    `x` は `[Cin, T, H, W]`、`cache` は前の chunk の入力の末尾 2 フレーム `[Cin, 2, H, W]`
    （最初の chunk はゼロ）。戻りは出力 `[Cout, T, H, W]` と、次の chunk へ渡す
    `cat(cache, x)` の末尾 2 フレーム（上流の `cache_x` の更新規則を T = 1 / T ≥ 2 で 1 本に
    した形）。
    空間 padding は conv3d の attrs へ畳む（IR の pad は最終次元・定数 0 だけ — 決定 1）。
    """
    _assert_causal(conv)
    expected = (x.shape[0], CACHE_FRAMES, x.shape[2], x.shape[3])
    if tuple(cache.shape) != expected:
        raise ValueError(
            f"cache の形 {tuple(cache.shape)} が入力 {tuple(x.shape)} と合わない（{expected}）"
        )
    pad_w, _, pad_h, _, _, _ = conv._padding
    joined = torch.cat([cache, x], dim=1)
    out = functional.conv3d(joined, conv.weight, conv.bias, padding=(0, pad_h, pad_w))
    return out, joined[:, -CACHE_FRAMES:]


def pointwise_conv3d(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """cache を持たない 1×1×1 の CausalConv3d（post-quant・shortcut）を unbatched の conv3d で
    回す。"""
    if tuple(conv.weight.shape[2:]) != (1, 1, 1) or any(conv._padding):
        raise UnsupportedVaeError(
            f"1×1×1 以外（カーネル {tuple(conv.weight.shape[2:])}・padding {conv._padding}）は"
            " cache 無しで回せない"
        )
    return functional.conv3d(x, conv.weight, conv.bias)


def rms_norm(norm: nn.Module, x: torch.Tensor, dim: int) -> torch.Tensor:
    """`WanRMS_norm` の rank 4 版（チャネル軸 `dim` — `[C,T,H,W]` は 0・`[T,C,H,W]` は 1）。

    gamma は上流の格納形のまま右詰めで当たる（`images=False` の `(C,1,1,1)` は `[C,T,H,W]` に、
    `images=True` の `(C,1,1)` は `[T,C,H,W]` の後ろ 3 軸に揃う）。上流の `+ self.bias`
    （`bias=False` のとき Python の `0.0`）は落とす — 値を変えずノードだけ増やす加算で、符号付き
    ゼロ以外に観測できる差は無い（anima の VAE パッチと同じ判断）。
    """
    if isinstance(norm.bias, torch.Tensor):
        raise UnsupportedVaeError("学習された bias 付きの RMS_norm は未対応")
    if not norm.channel_first:
        raise UnsupportedVaeError("channel_first=False の RMS_norm は未対応")
    if x.dtype != torch.float32:
        # 上流の fp16 / bf16 分岐（normalize だけ f32 で行う）を書いていない — 活性は常に f32。
        raise UnsupportedVaeError(f"RMS_norm の入力は f32 だけ（{x.dtype}）")
    return l2_normalize(x, dim) * norm.scale * norm.gamma


# Third-party code notice. `attention_block` below is adapted from `WanAttentionBlock.forward` in
# huggingface/diffusers
# (`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`, `diffusers==0.39.0`).
# It is verbatim except that the time axis is folded by a rank-4 permute instead of the rank-5
# permute / reshape, and the RMS norm is the rank-4 rewrite (`rms_norm`).
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def attention_block(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """mid の単一 head attention（`WanAttentionBlock`）を `[C,T,H,W]` のまま回す。

    上流の `rearrange(b c t h w → (b t) c h w)` を permute `[C,T,H,W] → [T,C,H,W]` で写し、
    フレームを
    attention のバッチに置く（`[T,1,H·W,C]` — B = T・H = 1・D = C）。chunk の mid は潜在 1 フレーム
    なので T = 1。中身は上流の forward から時間軸の出入りを外しただけ（anima の
    `_attention_block_forward` と同じ形）。
    """
    identity = x
    channels, frames, height, width = x.shape
    hidden = rms_norm(block.norm, x.permute(1, 0, 2, 3), dim=1)
    qkv = block.to_qkv(hidden)
    qkv = qkv.reshape(frames, 1, channels * 3, -1).permute(0, 1, 3, 2).contiguous()
    query, key, value = qkv.chunk(3, dim=-1)
    attended = functional.scaled_dot_product_attention(query, key, value)
    attended = attended.squeeze(1).permute(0, 2, 1).reshape(frames, channels, height, width)
    return block.proj(attended).permute(1, 0, 2, 3) + identity


class _CacheCursor:
    """decoder の forward 順に cache を配り、更新後の cache を同じ順で集める
    （上流の `feat_idx`）。"""

    def __init__(
        self, slots: Sequence[CacheSlot], cache: Mapping[str, torch.Tensor], *, first: bool
    ) -> None:
        expected = [slot.key for slot in graph_slots(slots, first=first)]
        if list(cache) != expected:
            raise ValueError(f"cache のキー {list(cache)} が {expected} と違う（順も見る）")
        self._slots: Iterator[CacheSlot] = iter(slots)
        self._cache = cache
        self._first = first
        self._updated: list[torch.Tensor] = []

    def causal(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        slot = next(self._slots)
        if slot.time_conv:
            raise AssertionError(f"slot {slot.key} は time_conv（forward の順が表とずれている）")
        return self._apply(slot, conv, x)

    def time_conv(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor | None:
        """`time_conv` を回す。最初の chunk では走らせず None（上流の `'Rep'` の経路）。"""
        slot = next(self._slots)
        if not slot.time_conv:
            raise AssertionError(
                f"slot {slot.key} は time_conv でない（forward の順が表とずれている）"
            )
        return None if self._first else self._apply(slot, conv, x)

    def finish(self) -> list[torch.Tensor]:
        """更新後の cache（入力と同じ順）。表を使い切っていなければ fail loudly。"""
        rest = list(self._slots)
        if rest:
            raise AssertionError(
                f"forward が cache を使い切っていない: {[slot.key for slot in rest]}"
            )
        return self._updated

    def _apply(self, slot: CacheSlot, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out, updated = causal_conv3d(conv, x, self._cache[slot.key])
        self._updated.append(updated)
        return out


# Third-party code notice. `residual_block` and `resample` below are adapted from
# `WanResidualBlock.forward` and `WanResample.forward` in
# huggingface/diffusers
# (`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`, `diffusers==0.39.0`).
# The order of operations is kept; the feat_cache bookkeeping is replaced by the normalized
# cache (`causal_conv3d`) and the rank-6 interleave by `interleave_frames`.
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def residual_block(block: nn.Module, x: torch.Tensor, cursor: _CacheCursor) -> torch.Tensor:
    """`WanResidualBlock` の rank 4 版（shortcut → norm1 → SiLU → conv1 → norm2 → SiLU → conv2
    → 加算）。

    dropout は eval で恒等（`nn.Dropout` を素通しする — 上流と同じモジュールを呼ぶ）。
    """
    shortcut = x if isinstance(block.conv_shortcut, nn.Identity) else None
    if shortcut is None:
        shortcut = pointwise_conv3d(block.conv_shortcut, x)
    hidden = block.nonlinearity(rms_norm(block.norm1, x, dim=0))
    hidden = cursor.causal(block.conv1, hidden)
    hidden = block.dropout(block.nonlinearity(rms_norm(block.norm2, hidden, dim=0)))
    hidden = cursor.causal(block.conv2, hidden)
    return hidden + shortcut


def resample(module: nn.Module, x: torch.Tensor, cursor: _CacheCursor) -> torch.Tensor:
    """`WanResample`（upsample3d / upsample2d）の rank 4 版。

    upsample3d は `time_conv`（next だけ）→ 時間インターリーブ → フレームごとの nearest ×2 +
    Conv2d 3×3。フレームごとの 2D 処理は permute `[C,T,H,W] → [T,C,H,W]` でフレームをバッチに置く
    （上流の `permute(0,2,1,3,4).reshape(b*t, …)` と op が 1 対 1）。
    """
    if module.mode == "upsample3d":
        doubled = cursor.time_conv(module.time_conv, x)
        if doubled is not None:
            x = interleave_frames(doubled)
    elif module.mode != "upsample2d":
        raise UnsupportedVaeError(f"decoder に現れない resample mode: {module.mode}")
    upsample, conv = module.resample
    frames = conv(nearest_exact_2x(upsample, x.permute(1, 0, 2, 3)))
    return frames.permute(1, 0, 2, 3)


# Third-party code notice. `avg_shortcut` and `residual_up_block` below are adapted from
# `DupUp3D.forward` and `WanResidualUpBlock.forward` in
# huggingface/diffusers
# (`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`, `diffusers==0.39.0`).
# The order of operations and the first-chunk slice are kept; the rank-8 view / permute is replaced
# by the rank-4 `dup_up_3d`, the defensive `x.clone()` is left out (it does not change values) and
# the feat_cache bookkeeping is replaced by the normalized cache (`causal_conv3d`).
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def avg_shortcut(module: nn.Module, x: torch.Tensor, *, first_chunk: bool) -> torch.Tensor:
    """Wan2.2 のショートカット `DupUp3D` の rank 4 版（`x` は `[Cin, T, H, W]`）。

    本体は {@link _shared.vae_rank4.dup_up_3d}。最初の chunk では時間の先頭 `ft − 1` 枚を捨てる
    （上流の `x[:, :, ft-1:]` — 残差の経路の upsample3d が最初の chunk で `time_conv` を走らせず
    1 枚のまま進むのに揃える）。ft = 1 の上流の slice は恒等なので書かない。
    """
    if int(module.in_channels) != x.shape[0]:
        raise UnsupportedVaeError(
            f"ショートカットの入力チャネル {module.in_channels} が入力 {tuple(x.shape)} と合わない"
        )
    factor_t = int(module.factor_t)
    out = dup_up_3d(
        x,
        out_channels=int(module.out_channels),
        factor_t=factor_t,
        factor_s=int(module.factor_s),
    )
    return out[:, factor_t - 1 :] if first_chunk and factor_t > 1 else out


def residual_up_block(
    block: nn.Module, x: torch.Tensor, cursor: _CacheCursor, *, first: bool
) -> torch.Tensor:
    """`WanResidualUpBlock` の rank 4 版（Res ×3 → upsampler → ショートカットを加算）。

    ショートカットは block の入力から作り、残差の経路の出力に足す（上流の
    `x + avg_shortcut(x_copy, first_chunk)` と同じ順）。
    """
    shortcut_input = x
    for resnet in block.resnets:
        x = residual_block(resnet, x, cursor)
    if block.upsampler is not None:
        x = resample(block.upsampler, x, cursor)
    if block.avg_shortcut is not None:
        x = x + avg_shortcut(block.avg_shortcut, shortcut_input, first_chunk=first)
    return x


# Third-party code notice. `WanVaeChunkDecoder.forward` below is adapted from
# `WanDecoder3d.forward` in
# huggingface/diffusers
# (`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`, `diffusers==0.39.0`).
# The order of the blocks is kept; the feat_cache list and the `first_chunk` flag are replaced by
# graph inputs / outputs and two graph variants.
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
class WanVaeChunkDecoder(nn.Module):
    """VAE decode の chunk グラフ 1 本（`first=True` が最初の chunk 用・`False` がそれ以降用）。

    入力は潜在 1 フレーム `[z, 1, h, w]`（z は 2.1 が 16・2.2 が 48。逆正規化 `z·std + mean` は
    ホスト — 決定 2）と cache の dict（キーは {@link CacheSlot.key}・順は表の順）。出力は
    **クランプ前**のフレーム（first は `[C, 1, 8h, 8w]`・next は `[C, 4, 8h, 8w]` — C は 2.1 が
    画素の 3・2.2 が patchify 空間の 12）と、更新後の cache（入力と同じ順）。`clamp(-1, 1)`（と
    2.2 の unpatchify）はホスト（タイルのブレンドの後 — 上流 `tiled_decode` と同じ位置）。

    post-quant の 1×1×1 conv はグラフの中でフレームごとに掛ける（上流の非タイル decode は全フレーム
    に 1 度掛けるが、1×1×1 なのでフレームごとと同じ値 — `tiled_decode` はフレームごとに掛ける）。
    """

    def __init__(self, vae: AutoencoderKLWan, *, first: bool) -> None:
        super().__init__()
        self.slots = cache_slots(vae)
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder
        self.first = first
        #: Wan2.2 の up block（`WanResidualUpBlock` — {@link assert_supported} が型を揃えてある）。
        self.residual = bool(vae.config.is_residual)

    @property
    def cache_slots(self) -> tuple[CacheSlot, ...]:
        """このグラフが入出力に持つ cache（入力の順 = 出力の順）。"""
        return graph_slots(self.slots, first=self.first)

    def forward(
        self, latent: torch.Tensor, cache: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        if latent.dim() != 4 or latent.shape[1] != 1:
            raise ValueError(
                f"潜在は unbatched の 1 フレーム [C, 1, h, w] で渡す（{tuple(latent.shape)}）"
            )
        decoder = self.decoder
        cursor = _CacheCursor(self.slots, cache, first=self.first)
        x = pointwise_conv3d(self.post_quant_conv, latent)
        x = cursor.causal(decoder.conv_in, x)
        mid = decoder.mid_block
        x = residual_block(mid.resnets[0], x, cursor)
        x = attention_block(mid.attentions[0], x)
        x = residual_block(mid.resnets[1], x, cursor)
        for block in decoder.up_blocks:
            if self.residual:
                x = residual_up_block(block, x, cursor, first=self.first)
                continue
            for resnet in block.resnets:
                x = residual_block(resnet, x, cursor)
            if block.upsamplers is not None:
                x = resample(block.upsamplers[0], x, cursor)
        x = decoder.nonlinearity(rms_norm(decoder.norm_out, x, dim=0))
        x = cursor.causal(decoder.conv_out, x)
        return (x, *cursor.finish())


# ---- 参照と chunk ループ（eager）----------------------------------------------------


# Third-party code notice. The loop of `reference_decode_unclamped` below reproduces the chunk loop
# of `AutoencoderKLWan._decode` in
# huggingface/diffusers
# (`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`, `diffusers==0.39.0`).
# verbatim; the unpatchify branch and the final clamp after it are left out.
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def reference_decode_unclamped(vae: AutoencoderKLWan, latents: torch.Tensor) -> torch.Tensor:
    """上流の非タイル `_decode` の chunk ループそのもの（その後の unpatchify と `clamp` を外した
    もの）。

    `latents` は `[1, z, F, h, w]`（逆正規化済み）。戻りは `[1, C, 1 + 4(F−1), 8h, 8w]`（C は 2.1 が
    画素の 3・2.2 が patchify 空間の 12 — chunk グラフの出口と同じ空間）。上流の
    `AutoencoderKLWan._decode`（`autoencoder_kl_wan.py:1187-1217`）の chunk ループを逐語で写し、
    unpatchify（2.1 には無い）と `clamp` の手前で止める — クランプは飽和した要素の差を隠すので、
    照合はクランプ前で行う（unpatchify とクランプを当てると上流の `_decode` とビット一致することは
    テストが固定する）。
    """
    assert_supported(vae)
    vae.clear_cache()
    hidden = vae.post_quant_conv(latents)
    frames: list[torch.Tensor] = []
    for index in range(latents.shape[2]):
        vae._conv_idx = [0]
        frames.append(
            vae.decoder(
                hidden[:, :, index : index + 1],
                feat_cache=vae._feat_map,
                feat_idx=vae._conv_idx,
                first_chunk=index == 0,
            )
        )
    vae.clear_cache()
    return torch.cat(frames, dim=2)


def normalized_cache_decode(
    vae: AutoencoderKLWan, latents: torch.Tensor, tile: int
) -> torch.Tensor:
    """**cache の正規化だけ**を当てた chunk ループ（上流の decoder のコードのまま・rank 5）。

    上流の decoder をそのまま呼び、feat_cache の初期値だけを「2 フレームのゼロ」に変える
    （`time_conv` の 2 本は最初の chunk で `None` = 上流の `'Rep'` の経路・2 chunk 目の前に
    ゼロへ差し替える）。上流は 2 フレームの cache を受けると時間の pad を 0 にして cat だけで
    回すので、ここが上流の `_decode` とビット一致すれば「cache の正規化は値を変えない」が
    言える（決定 2 の導出の実測。空間の pad は上流のまま明示）。
    """
    slots = cache_slots(vae)
    side = latents.shape[3]
    if latents.shape[4] != side or side != tile:
        raise ValueError(f"潜在の空間 {tuple(latents.shape[3:])} がタイル {tile} の正方でない")

    def zero(slot: CacheSlot) -> torch.Tensor:
        return torch.zeros(1, *slot.shape(tile))

    hidden = vae.post_quant_conv(latents)
    feat: list[torch.Tensor | str | None] = [
        None if slot.time_conv else zero(slot) for slot in slots
    ]
    frames: list[torch.Tensor] = []
    for index in range(latents.shape[2]):
        frames.append(
            vae.decoder(
                hidden[:, :, index : index + 1],
                feat_cache=feat,
                feat_idx=[0],
                first_chunk=index == 0,
            )
        )
        if index == 0:
            if any(feat[slot.index] != "Rep" for slot in slots if slot.time_conv):
                raise AssertionError("最初の chunk で time_conv が 'Rep' の経路を通っていない")
            feat = [zero(slot) if slot.time_conv else feat[slot.index] for slot in slots]
        for slot in slots:
            cached = feat[slot.index]
            if not isinstance(cached, torch.Tensor) or cached.shape[2] != CACHE_FRAMES:
                raise AssertionError(f"chunk {index} の後の cache {slot.key} が 2 フレームでない")
    return torch.cat(frames, dim=2)


def chunk_decode(
    first: WanVaeChunkDecoder, following: WanVaeChunkDecoder, latents: torch.Tensor
) -> torch.Tensor:
    """書き直した chunk グラフ 2 本を eager で回す（ホストのタイル 1 枚ぶんの chunk ループ）。

    `latents` は unbatched の `[z, F, h, w]`。cache はタイルの頭でゼロに作り直し、first の後は
    `time_conv` の 2 本だけゼロのまま足して next へ渡す（first は触らない — 決定 2）。戻りは
    クランプ前の `[C, 1 + 4(F−1), 8h, 8w]`（C は {@link WanVaeChunkDecoder} の出口と同じ）。
    """
    if not first.first or following.first:
        raise ValueError("first には first=True・following には first=False のグラフを渡す")
    tile = latents.shape[2]
    zeros = zero_caches(first.slots, tile, first=False)
    cache = {slot.key: zeros[slot.key] for slot in first.cache_slots}
    frames: list[torch.Tensor] = []
    for index in range(latents.shape[1]):
        graph = first if index == 0 else following
        latent = latents[:, index : index + 1]
        out, *updated = graph(latent, cache)
        frames.append(out)
        cache = dict(zeros)
        cache.update(
            (slot.key, value) for slot, value in zip(graph.cache_slots, updated, strict=True)
        )
    return torch.cat(frames, dim=1)
