"""BiRefNet_HR（同梱 `birefnet.py`）を torch.export → IR v1 まで通すパッチ層。

対象は HF `ZhengPeng7/BiRefNet_HR` が `trust_remote_code` で読み込む動的モジュール
（`transformers_modules.…birefnet`）。差し替えるのはそのモジュールのクラス属性と
グローバル関数だけで、transformers / timm / torchvision 側には一切触らない。

差し替えは**全て第 0 層**（新しい op を足さない）で、性格は 2 群に分かれる:

- **ビット同一（データの並べ替えだけ — 演算を 1 つも増減させない）**
  ① `window_partition` / `window_reverse` の rank-6 view+permute を rank-4 の隣接転置へ
     （IR の strided 表現は rank ≤ 4 — `normalize._lower_reshape_permute` が rank5 以上の
     端点を見つけられずに落ちる）
  ② `WindowAttention` の qkv を rank-5 permute から**最終次元 slice 3 本**へ（同じ理由）。
     相対位置バイアスの表引き `table[index.view(-1)]`（`aten.index.Tensor` = 未対応）は
     同値な `nn.functional.embedding(index, table)` へ。窓マスクの加算は rank-5 view +
     broadcast から rank-4 の broadcast へ（**B = 1 でのみ同値** — 外れたら fail loudly）
  ③ `SwinTransformerBlock` の H/W 方向 `F.pad`（IR の pad は最終次元のみ）をゼロ定数との
     `cat` へ、`torch.roll`（ハンドラ無し・分解形は `index_select` + `fmod`）を静的
     シフトの slice + cat へ
  ④ `BasicLayer` の shifted-window マスク（`zeros` への in-place 代入 = `slice_scatter` /
     `select` / `copy` の列）を**焼いた定数バッファ**へ。値の作り方は元実装の逐語
     （{@link _build_attn_mask}）で、実行の場所が export の外に移るだけ
  ⑤ `PatchMerging` の `x[:, 0::2, 0::2, :]`（step 2 の slice = IR の slice は連続窓のみ）を
     reshape + 連続 slice へ
  ⑥ `image2patches` の einops `rearrange`（rank-6）を rank-4 の隣接転置へ
- **ビット同一ではない（式は同値・丸めだけ違う）— 実測値は下記**
  ⑦ `nn.BatchNorm2d`（推論）を per-channel の `x·α + β` へ。α/β の作り方は ATen の
     推論経路（`alpha = weight·rsqrt(var+eps)` / `beta = bias − mean·alpha`）と同式だが、
     ATen のカーネルは積和を 1 命令（FMA）で畳むので**最下位ビットが違う**
     （実測 max 2.4e-07 — `birefnet/tests/test_patch.py`）。分解形を FMA で書き直しても
     torch 側と一致させる手段は無い（torch の eager `x*a+b` 自体が一致しない）
  ⑧ `nn.AdaptiveAvgPool2d((1, 1))` を `sum(dim=3) → sum(dim=2) → div(H·W)` へ。
     縮約の順序が torch の 2 軸同時縮約と違うぶん最下位ビットが違う（実測 max 1.1e-08）
  ⑨ `Decoder` 末尾の「全解像度へ伸ばす → `cat` → 1×1 conv」を「1×1 conv → 伸ばす →
     もう片側の 1×1 conv を足す」へ組み替える（{@link TailConv}）。1×1 conv も bilinear
     upsample も線形なので実数の上で可換で、全解像度の巨大な中間 2 本（2048² で
     `[1,192,2048,2048]` 3.2GB と `[1,240,2048,2048]` 4.0GB）が消える。縮約の順序が
     変わるぶん最下位ビットが違うが、寄与は ⑦⑧ の丸めに埋もれる大きさ（1024² を端から
     端まで通した実測 maxdiff は ⑦⑧ まで 5.3e-05 → ⑨ まで 5.8e-05・`export.py --verify`）

入口は 3 群に対応して分かれている（{@link apply_layout_patches} / {@link apply_module_patches}
/ {@link apply_tail_patches}・全て当てるのが {@link apply}）— 一括でしか当てられないと、
①〜⑥ のビット同一を単体で実測できず、⑦⑧ と ⑨ の丸めの寄与も分けて採れない。

MUST: どのパッチも fallback を持たない（`siglip2.patch` / `deberta.patch` と同じ規律）。
前提（B = 1 / 偶数解像度 / `dec_att='ASPPDeformable'` 等）を外したモデルは黙って別の数値
経路へ落ちるのではなく、その場で落ちる。

MUST: {@link apply} の後に {@link prepare} を**必ず**通す。④ のマスクは解像度依存の定数で、
export の中では作れない（`register_buffer` がグラフに乗らない）。`prepare` は eager の
1 forward でマスクを焼き、そのまま**パッチ後の参照出力**を返す。

NOTE: 動的モジュールのクラス属性をプロセス全域で差し替えるので、「パッチ前の参照」を採れる
のは 1 プロセスにつき 1 回だけ（{@link patches_applied} が門）。
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

#: 差し替えを当てたか（パッチ前の参照を採る側の恒真化を防ぐ門）。
_APPLIED = False

#: {@link prepare} の実行中だけ True。窓マスクの構築を許す唯一の窓で、export 中に
#: `register_buffer` が走る（= グラフに乗らない定数が黙って生える）ことを防ぐ。
_MASK_BUILD_ENABLED = False


def patches_applied() -> bool:
    """このプロセスでクラス属性の差し替えを当てたか。"""
    return _APPLIED


def assert_supported(model: nn.Module) -> None:
    """差し替え版が前提にしている構成を検査する。

    見るのは**外れても shape エラーにならずに別の経路へ落ちる**ものだけ。
    """
    config = model.config
    expected = {
        "bb": "swin_v1_l",
        "dec_att": "ASPPDeformable",
        "dec_blk": "BasicDecBlk",
        "mul_scl_ipt": "cat",
        "squeeze_block": "BasicDecBlk_x1",
        "refine": "",
    }
    for key, want in expected.items():
        got = getattr(config, key)
        if got != want:
            raise ValueError(f"config.{key} が {want!r} でない（{got!r}）— 差し替え版の対象外")
    if not config.dec_ipt or not config.dec_ipt_split:
        raise ValueError(
            "dec_ipt / dec_ipt_split が False — image2patches の書き換えが噛み合わない"
        )
    if config.batch_size <= 1:
        raise ValueError("batch_size ≤ 1 では BatchNorm2d が Identity になる — 重みの前提と違う")
    if model.bb.ape:
        raise ValueError("ape=True は差し替え版が持たない経路（bicubic の位置埋め込み補間が出る）")


# ---- ①窓の切り出し / 復元（rank ≤ 4） -------------------------------------


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """`[B, H, W, C]` → `[B·nH·nW, ws, ws, C]`（元実装の rank-6 版とビット同一）。

    元実装は `(b, hb, hi, wb, wi, c)` を `(b, hb, wb, hi, wi, c)` へ並べ替える。入れ替える
    のは**隣接する 2 軸**（`hi` と `wb`）だけなので、先頭 `(b, hb)` と末尾 `(wi, c)` を
    それぞれ 1 軸に畳めば rank-4 の permute 1 本で書ける。
    """
    batch, height, width, channels = x.shape
    x = x.reshape(
        batch * (height // window_size),
        window_size,
        width // window_size,
        window_size * channels,
    )
    x = x.permute(0, 2, 1, 3)
    return x.reshape(-1, window_size, window_size, channels)


def _window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int
) -> torch.Tensor:
    """`[B·nH·nW, ws, ws, C]` → `[B, H, W, C]`（{@link _window_partition} の逆写像）。"""
    channels = windows.shape[-1]
    batch = windows.shape[0] // ((height // window_size) * (width // window_size))
    x = windows.reshape(
        batch * (height // window_size),
        width // window_size,
        window_size,
        window_size * channels,
    )
    x = x.permute(0, 2, 1, 3)
    return x.reshape(batch, height, width, channels)


# ---- ②窓 attention ---------------------------------------------------------


def _window_attention_forward(
    self: nn.Module, x: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """`WindowAttention.forward` の同値実装（rank ≤ 4・表引きは embedding）。

    元実装との差は 3 点だけで、いずれも値の並べ替え方が違うだけ:

    1. qkv は `reshape(B_, N, 3, nH, hd).permute(2, 0, 3, 1, 4)` の代わりに最終次元を
       3 等分する（`(3, nH, hd)` の入れ子なので先頭 C 要素が q）。
    2. 相対位置バイアスは `table[index.view(-1)]`（`aten.index.Tensor`）ではなく
       `nn.functional.embedding(index, table)`。どちらも行 gather そのもの。
    3. 窓マスクの加算は `attn.view(B_//nW, nW, …) + mask[None, :, None]` ではなく
       `attn + mask[:, None]`。**B_ == nW（= バッチ 1）でのみ同値**なので明示に落とす。
    """
    batch_windows, tokens, channels = x.shape
    heads = self.num_heads
    head_dim = channels // heads
    window_tokens = self.window_size[0] * self.window_size[1]
    if tokens != window_tokens:
        raise ValueError(f"窓の要素数 {tokens} が window_size {self.window_size} と合わない")

    qkv = self.qkv(x)
    projections = []
    for index in range(3):
        part = qkv[:, :, index * channels : (index + 1) * channels]
        projections.append(part.reshape(batch_windows, tokens, heads, head_dim).permute(0, 2, 1, 3))
    query, key, value = projections

    query = query * self.scale
    attn = query @ key.transpose(-2, -1)

    bias = nn.functional.embedding(
        self.relative_position_index.reshape(-1), self.relative_position_bias_table
    )
    bias = bias.reshape(window_tokens, window_tokens, heads).permute(2, 0, 1)
    attn = attn + bias.unsqueeze(0)

    if mask is not None:
        windows = mask.shape[0]
        if batch_windows != windows:
            raise NotImplementedError(
                f"窓数 {windows} と第 0 軸 {batch_windows} が違う（バッチ > 1）—"
                " 差し替え版の broadcast は B = 1 でのみ元実装と同値"
            )
        attn = attn + mask.unsqueeze(1)
    attn = self.softmax(attn)
    attn = self.attn_drop(attn)

    out = (attn @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
    out = self.proj(out)
    return self.proj_drop(out)


# ---- ③Swin ブロック（pad → cat / roll → slice+cat） ------------------------


def _pad_zero(x: torch.Tensor, dim: int, width: int) -> torch.Tensor:
    """`x` の軸 `dim` の末尾へゼロを `width` 本足す（`F.pad` の同値形）。

    IR の `pad` は最終次元専業なので、H/W 方向の padding は**ゼロ定数との `cat`** で書く。
    ゼロ定数は入力に依存しないため、エクスポータの定数畳み込みが initializer へ焼く。
    """
    if width == 0:
        return x
    shape = list(x.shape)
    shape[dim] = width
    zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, zeros], dim=dim)


def _roll(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """`torch.roll(x, shift, dim)` の同値形（静的シフトの slice + cat）。

    `torch.roll` は curated decomp 後に `arange` + `fmod` + `index_select` へ落ちる
    （`index_select` は語彙外）。シフト量が静的なら連結順を入れ替えるだけで書ける。
    """
    if shift == 0:
        return x
    size = x.shape[dim]
    offset = shift % size
    head = x.narrow(dim, size - offset, offset)
    tail = x.narrow(dim, 0, size - offset)
    return torch.cat([head, tail], dim=dim)


def _swin_block_forward(
    self: nn.Module, x: torch.Tensor, mask_matrix: torch.Tensor | None
) -> torch.Tensor:
    """`SwinTransformerBlock.forward` の同値実装（pad と roll だけ書き換え）。"""
    batch, length, channels = x.shape
    height, width = self.H, self.W
    if length != height * width:
        raise ValueError(f"入力長 {length} が {height}×{width} と合わない")

    shortcut = x
    x = self.norm1(x)
    x = x.view(batch, height, width, channels)

    pad_bottom = (self.window_size - height % self.window_size) % self.window_size
    pad_right = (self.window_size - width % self.window_size) % self.window_size
    x = _pad_zero(x, 1, pad_bottom)
    x = _pad_zero(x, 2, pad_right)
    padded_height, padded_width = height + pad_bottom, width + pad_right

    if self.shift_size > 0:
        shifted_x = _roll(_roll(x, -self.shift_size, 1), -self.shift_size, 2)
        attn_mask = mask_matrix
    else:
        shifted_x = x
        attn_mask = None

    x_windows = _window_partition(shifted_x, self.window_size)
    x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)
    attn_windows = self.attn(x_windows, mask=attn_mask)

    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)
    shifted_x = _window_reverse(attn_windows, self.window_size, padded_height, padded_width)

    if self.shift_size > 0:
        x = _roll(_roll(shifted_x, self.shift_size, 1), self.shift_size, 2)
    else:
        x = shifted_x

    if pad_right > 0 or pad_bottom > 0:
        x = x[:, :height, :width, :].contiguous()

    x = x.view(batch, height * width, channels)
    x = shortcut + self.drop_path(x)
    return x + self.drop_path(self.mlp(self.norm2(x)))


# ---- ④shifted-window マスク（焼いた定数） ----------------------------------


def _build_attn_mask(
    padded_height: int,
    padded_width: int,
    window_size: int,
    shift_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """`BasicLayer.forward` のマスク生成の逐語移植（eager 実行専用）。

    元実装との差は ①`Hp` / `Wp` を tensor ではなく python int で受ける ②`window_partition`
    が {@link _window_partition}（ビット同一）である の 2 点だけ。値の作り方は同じ。
    """
    img_mask = torch.zeros((1, padded_height, padded_width, 1), device=device)
    slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    count = 0
    for rows in slices:
        for columns in slices:
            img_mask[:, rows, columns, :] = count
            count += 1

    mask_windows = _window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0).to(dtype)


def _basic_layer_forward(
    self: nn.Module, x: torch.Tensor, height: int, width: int
) -> tuple[Any, ...]:
    """`BasicLayer.forward` の同値実装（マスクは焼いたバッファから引く）。"""
    window_size = self.window_size
    padded_height = -(-height // window_size) * window_size
    padded_width = -(-width // window_size) * window_size
    name = f"karume_attn_mask_{padded_height}x{padded_width}"
    attn_mask = getattr(self, name, None)
    if attn_mask is None:
        if not _MASK_BUILD_ENABLED:
            raise RuntimeError(
                f"{padded_height}×{padded_width} の窓マスクが焼かれていない —"
                " export の前に birefnet.patch.prepare(model, x) を通すこと"
            )
        attn_mask = _build_attn_mask(
            padded_height, padded_width, window_size, self.shift_size, x.dtype, x.device
        )
        self.register_buffer(name, attn_mask)

    if self.use_checkpoint:
        raise NotImplementedError("use_checkpoint=True は差し替え版が持たない経路")
    for block in self.blocks:
        block.H, block.W = height, width
        x = block(x, attn_mask)
    if self.downsample is not None:
        x_down = self.downsample(x, height, width)
        return x, height, width, x_down, (height + 1) // 2, (width + 1) // 2
    return x, height, width, x, height, width


# ---- ⑤PatchMerging（step 2 の slice を消す） -------------------------------


def _patch_merging_forward(
    self: nn.Module, x: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """`PatchMerging.forward` の同値実装（`x[:, 0::2, 0::2, :]` を連続 slice へ）。

    `(hb, hi, wb, wi)` に割り直すと、元実装の 4 本の step-2 slice は「軸 `hi` と `wi` を
    それぞれ 0 / 1 で切る」だけになる。`hi` は 2 番目の軸、`wi` は最終次元の前半 / 後半。
    """
    batch, length, channels = x.shape
    if length != height * width:
        raise ValueError(f"入力長 {length} が {height}×{width} と合わない")
    if height % 2 or width % 2:
        raise NotImplementedError(
            f"奇数解像度 {height}×{width} は差し替え版が持たない経路（元実装の pad 分岐）"
        )
    x = x.view(batch, height, width, channels)
    x = x.reshape(batch * (height // 2), 2, width // 2, 2 * channels)

    quadrants = []
    for row in range(2):
        picked = x[:, row : row + 1, :, :]
        for column in range(2):
            part = picked[:, :, :, column * channels : (column + 1) * channels]
            quadrants.append(part.reshape(batch, height // 2, width // 2, channels))
    # 元実装の並び（even/even, odd/even, even/odd, odd/odd）へ組み直す。
    even_even, even_odd, odd_even, odd_odd = quadrants
    x = torch.cat([even_even, odd_even, even_odd, odd_odd], -1)
    x = x.view(batch, -1, 4 * channels)

    return self.reduction(self.norm(x))


# ---- ⑥image2patches（einops の rank-6 rearrange） --------------------------

#: 差し替え版が持つ唯一の変換式（`Decoder.forward` / `BiRefNet` の実測形）。
_SUPPORTED_TRANSFORMATION = "b c (hg h) (wg w) -> b (c hg wg) h w"


def _image2patches(
    image: torch.Tensor,
    grid_h: int = 2,
    grid_w: int = 2,
    patch_ref: torch.Tensor | None = None,
    transformation: str = _SUPPORTED_TRANSFORMATION,
) -> torch.Tensor:
    """`image2patches` の同値実装（rank-4 の隣接転置 1 本）。

    `(b, c, hg, h, wg, w)` → `(b, c, hg, wg, h, w)` は隣接 2 軸（`h` と `wg`）の入れ替え
    なので、先頭 `(b, c, hg)` を 1 軸に畳めば rank-4 で書ける。
    """
    if transformation != _SUPPORTED_TRANSFORMATION:
        raise NotImplementedError(f"変換式 {transformation!r} は差し替え版が持たない経路")
    if patch_ref is not None:
        grid_h = image.shape[-2] // patch_ref.shape[-2]
        grid_w = image.shape[-1] // patch_ref.shape[-1]
    batch, channels, height, width = image.shape
    patch_h, patch_w = height // grid_h, width // grid_w
    patches = image.reshape(batch * channels * grid_h, patch_h, grid_w, patch_w)
    patches = patches.permute(0, 2, 1, 3)
    return patches.reshape(batch, channels * grid_h * grid_w, patch_h, patch_w)


# ---- ⑦⑧ モジュールの差し替え（BatchNorm2d / AdaptiveAvgPool2d） -----------


class ChannelAffine(nn.Module):
    """推論時 `nn.BatchNorm2d` の同値形 `x·α + β`（α / β は per-channel の定数）。

    ATen の推論経路と同式（`alpha = weight·rsqrt(var+eps)` / `beta = bias − mean·alpha`）だが、
    ATen 側は積和を FMA で畳むので**ビット一致はしない**（実測 max 2.4e-07）。
    """

    def __init__(self, source: nn.BatchNorm2d) -> None:
        super().__init__()
        if source.training:
            raise ValueError("学習モードの BatchNorm2d は差し替え対象外（統計が動く）")
        if source.running_mean is None or source.running_var is None:
            raise ValueError("track_running_stats=False の BatchNorm2d は推論の統計を持たない")
        weight = source.weight if source.affine else torch.ones_like(source.running_var)
        bias = source.bias if source.affine else torch.zeros_like(source.running_var)
        alpha = weight.detach() * torch.rsqrt(source.running_var.detach() + source.eps)
        beta = bias.detach() - source.running_mean.detach() * alpha
        self.register_buffer("alpha", alpha.reshape(1, -1, 1, 1).clone())
        self.register_buffer("beta", beta.reshape(1, -1, 1, 1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha + self.beta


class SpatialMean(nn.Module):
    """`nn.AdaptiveAvgPool2d((1, 1))` の同値形（1 軸ずつの `sum` + `div`）。

    IR の reduce は 1 軸専業で `keepdim` を持たないので、`sum(dim=3) → sum(dim=2)` の 2 段に
    割ってから `[B, C, 1, 1]` へ戻す。縮約順序が torch の 2 軸同時縮約と違うぶん最下位ビット
    が違う（実測 max 1.1e-08）。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        total = x.sum(dim=3).sum(dim=2)
        return (total / (height * width)).reshape(batch, channels, 1, 1)


def _lift_relative_position_index(model: nn.Module) -> int:
    """`relative_position_index` をバッファから**素のテンソル属性**へ降ろす。

    IR の initializer は意味論 f32 のみで、i64 バッファはそのままでは載らない
    （`convert._materialize_initializer` が fail loudly）。素の属性にすると torch.export は
    lifted tensor constant として扱い、エクスポータの定数畳み込み経路が i32 へ境界正規化して
    焼く（`convert._add_const` — ADR 0009 / 0010）。値は 1 要素も変えない。
    """
    lifted = 0
    for module in model.modules():
        index = module._buffers.get("relative_position_index")
        if index is None:
            continue
        del module._buffers["relative_position_index"]
        object.__setattr__(module, "relative_position_index", index)
        lifted += 1
    return lifted


def _replace_modules(model: nn.Module) -> dict[str, int]:
    """`BatchNorm2d` → {@link ChannelAffine} / `AdaptiveAvgPool2d` → {@link SpatialMean}。"""
    counts = {"batch_norm": 0, "adaptive_avg_pool": 0}
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                setattr(parent, name, ChannelAffine(child))
                counts["batch_norm"] += 1
            elif isinstance(child, nn.AdaptiveAvgPool2d):
                if tuple(child.output_size) != (1, 1):
                    raise NotImplementedError(
                        f"output_size={child.output_size} の AdaptiveAvgPool2d は差し替え対象外"
                    )
                setattr(parent, name, SpatialMean())
                counts["adaptive_avg_pool"] += 1
    return counts


# ---- ⑨decoder 末尾（1×1 conv と bilinear upsample の順序交換） -------------


def _resize(x: torch.Tensor, size: Any) -> torch.Tensor:
    """上流の `F.interpolate(..., mode='bilinear', align_corners=True)` の綴り違い。"""
    return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=True)


class TailConv(nn.Module):
    """`Decoder.conv_out1`（1×1 conv 240→1）を「先に潰してから伸ばす」順序へ組み替えた同値形。

    上流の末尾は全解像度の中間を 2 本作る（`decoder_block1` の出力を伸ばした `[1,192,S,S]` と、
    そこへ ipt 枝 48 チャネルを継いだ `[1,240,S,S]`）。S = 2048 では 3.2GB / 4.0GB になり、
    どちらも NVIDIA の束縛上限 2GiB を超える。

    1×1 conv はチャネル方向の線形結合、bilinear upsample は空間方向の線形結合で、掛かる軸が
    互いに素なので**実数の上で可換**。さらに `cat` の後の 1×1 conv は 2 つの部分和に割れる
    ので、重みを head 側（`[1,192,1,1]`）と ipt 側（`[1,48,1,1]`）へ分けて

        p1_out = interpolate(conv1x1(head, W_head)) + conv1x1(ipt, W_ipt, b)

    と書ける。全解像度で残る中間は `[1,1,S,S]`（4MB）と ipt 枝の内部だけになる。

    式は同値だが縮約の順序が変わる（192 チャネルの積和を伸ばす前に畳む）ので**ビット一致は
    しない** — ⑦⑧ と同じ「丸めだけ違う」群（単体の実測 max 2.0e-05 = 出力の絶対値 ~55 に
    対する相対 3.5e-07・`birefnet/tests/test_patch.py`）。
    """

    def __init__(self, source: nn.Module, head_channels: int) -> None:
        super().__init__()
        conv = _tail_conv2d(source)
        if not 0 < head_channels < conv.in_channels:
            raise NotImplementedError(
                f"head 側 {head_channels} チャネルが conv_out1 の入力 {conv.in_channels} と"
                " 噛み合わない（重みをチャネルで割れない）"
            )
        weight = conv.weight.detach()
        self.register_buffer("head_weight", weight[:, :head_channels].clone())
        self.register_buffer("ipt_weight", weight[:, head_channels:].clone())
        self.register_buffer("bias", conv.bias.detach().clone())

    def forward(self, head: torch.Tensor, size: Any, ipt: torch.Tensor) -> torch.Tensor:
        """`head`（低解像度）を `size` へ伸ばした結果と `ipt`（全解像度）の寄与を足す。

        bias は片側にだけ足す（両方に足すと 2 回効く）。
        """
        pooled = nn.functional.conv2d(head, self.head_weight)
        return _resize(pooled, size) + nn.functional.conv2d(ipt, self.ipt_weight, self.bias)


def _tail_conv2d(source: nn.Module) -> nn.Conv2d:
    """`conv_out1` から 1×1 conv を取り出す（前提を外れたら落とす — fallback 無し）。"""
    if not isinstance(source, nn.Sequential) or len(source) != 1:
        raise NotImplementedError(f"conv_out1 が長さ 1 の nn.Sequential でない（{source!r}）")
    conv = source[0]
    if not isinstance(conv, nn.Conv2d):
        raise NotImplementedError(f"conv_out1[0] が nn.Conv2d でない（{type(conv).__name__}）")
    if (
        tuple(conv.kernel_size) != (1, 1)
        or tuple(conv.stride) != (1, 1)
        or tuple(conv.padding) != (0, 0)
        or tuple(conv.dilation) != (1, 1)
        or conv.groups != 1
    ):
        raise NotImplementedError(
            "conv_out1[0] が 1×1・stride 1・pad 0・groups 1 の conv でない"
            f"（kernel={conv.kernel_size} stride={conv.stride} padding={conv.padding}"
            f" dilation={conv.dilation} groups={conv.groups}）— upsample と可換にならない"
        )
    if conv.bias is None:
        raise NotImplementedError("conv_out1[0] が bias を持たない — 差し替え版の前提と違う")
    return conv


def _tail_head_channels(decoder: nn.Module) -> int:
    """`conv_out1` の入力のうち `decoder_block1` 由来の本数（残りが ipt 枝）。

    同じ数が 2 通りに導ける（`decoder_block1` の出力 conv / `conv_out1` の入力から ipt 枝の
    出力を引く）ので両方を採って突き合わせる — 食い違ったら重みをチャネルで割る位置が
    前提と違うので、黙って別の分け方をせずに落とす。
    """
    from_block = decoder.decoder_block1.conv_out.out_channels
    from_split = decoder.conv_out1[0].in_channels - decoder.ipt_blk1.conv_out.out_channels
    if from_block != from_split:
        raise NotImplementedError(
            f"decoder_block1 の出力 {from_block} と conv_out1 の残り {from_split} が食い違う"
            " — 末尾の重みをチャネルで割れない"
        )
    return from_block


def _decoder_forward(self: nn.Module, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """`Decoder.forward` の eval 経路の同値実装（末尾だけ {@link TailConv} へ差し替え）。

    上流 2145〜2224 行（x4 → p4 → p3 → p2 → _p1 の 4 段）は逐語で、`config.out_ref` の gdt
    attention（`p4 = p4 * gdt_attn_4` など）は eval でも走るのでそのまま写す。書き換えたのは
    末尾 2225〜2231 行だけ（理屈は {@link TailConv}）。

    学習経路（`ms_supervision` の中間予測 append / gdt の GT 側）は写していないので、
    `self.training` が真ならその場で落とす。
    """
    if self.training:
        raise RuntimeError(
            "学習モードの Decoder.forward は差し替え版が持たない経路"
            "（ms_supervision の中間予測と gdt の GT を写していない）"
        )
    x, x1, x2, x3, x4 = features

    if self.config.dec_ipt:
        patches_batch = _image2patches(x, patch_ref=x4) if self.split else x
        x4 = torch.cat((x4, self.ipt_blk5(_resize(patches_batch, x4.shape[2:]))), 1)
    p4 = self.decoder_block4(x4)
    if self.config.out_ref:
        p4_gdt = self.gdt_convs_4(p4)
        gdt_attn_4 = self.gdt_convs_attn_4(p4_gdt).sigmoid()
        p4 = p4 * gdt_attn_4
    _p4 = _resize(p4, x3.shape[2:])
    _p3 = _p4 + self.lateral_block4(x3)

    if self.config.dec_ipt:
        patches_batch = _image2patches(x, patch_ref=_p3) if self.split else x
        _p3 = torch.cat((_p3, self.ipt_blk4(_resize(patches_batch, x3.shape[2:]))), 1)
    p3 = self.decoder_block3(_p3)
    if self.config.out_ref:
        p3_gdt = self.gdt_convs_3(p3)
        gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
        p3 = p3 * gdt_attn_3
    _p3 = _resize(p3, x2.shape[2:])
    _p2 = _p3 + self.lateral_block3(x2)

    if self.config.dec_ipt:
        patches_batch = _image2patches(x, patch_ref=_p2) if self.split else x
        _p2 = torch.cat((_p2, self.ipt_blk3(_resize(patches_batch, x2.shape[2:]))), 1)
    p2 = self.decoder_block2(_p2)
    if self.config.out_ref:
        p2_gdt = self.gdt_convs_2(p2)
        gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
        p2 = p2 * gdt_attn_2
    _p2 = _resize(p2, x1.shape[2:])
    _p1 = _p2 + self.lateral_block2(x1)

    if self.config.dec_ipt:
        patches_batch = _image2patches(x, patch_ref=_p1) if self.split else x
        _p1 = torch.cat((_p1, self.ipt_blk2(_resize(patches_batch, x1.shape[2:]))), 1)
    # 以下が書き換えた末尾。上流は head を全解像度へ伸ばして `_p1` に代入し（2226 行）、その
    # `[1, 192, S, S]` を次の `patch_ref` に渡す（2229 行）。`image2patches` が `patch_ref` から
    # 読むのは空間サイズだけで、伸ばした後の形は `x` と同じ `(S, S)` — 格子は 1×1 になるので、
    # 伸ばす前に `x` 自身を渡しても上流と同じ `patches_batch` になる。
    head = self.decoder_block1(_p1)
    patches_batch = _image2patches(x, patch_ref=x) if self.split else x
    ipt = self.ipt_blk1(_resize(patches_batch, x.shape[2:]))
    p1_out = self.conv_out1(head, x.shape[2:], ipt)
    return [p1_out]


# ---- 適用と準備 ------------------------------------------------------------


def apply_layout_patches(model: nn.Module) -> dict[str, int]:
    """並べ替えだけの差し替え ①〜⑥ を当てる（クラス属性 5 本 + モジュールグローバル 1 本）。

    この段だけならパッチ前 eager と**ビット同一**で、`birefnet.export.py --verify` の 1 点目
    がその主張を実測する。⑦⑧ と分けて公開しているのはそのため（一括で当てると、丸めが動く
    書き換えと混ざってビット一致の主張が採れない）。
    """
    global _APPLIED
    assert_supported(model)
    module = sys.modules[type(model).__module__]
    module.image2patches = _image2patches
    module.WindowAttention.forward = _window_attention_forward
    module.SwinTransformerBlock.forward = _swin_block_forward
    module.BasicLayer.forward = _basic_layer_forward
    module.PatchMerging.forward = _patch_merging_forward
    _APPLIED = True
    return {"relative_position_index": _lift_relative_position_index(model)}


def apply_module_patches(model: nn.Module) -> dict[str, int]:
    """丸めが動く差し替え ⑦⑧ を当てる（`BatchNorm2d` / `AdaptiveAvgPool2d` のインスタンス走査）。"""
    return _replace_modules(model)


def apply_tail_patches(model: nn.Module) -> dict[str, int]:
    """decoder 末尾の代数書き換え ⑨ を当てる（`conv_out1` の差し替え + `Decoder.forward`）。

    `dec_ipt` / `dec_ipt_split` / `mul_scl_ipt == 'cat'` は {@link assert_supported} が保証
    済みなので、ここで足すのは `conv_out1` の形の検査だけ（{@link _tail_conv2d}）。⑦⑧ と
    別の入口にしているのは、`birefnet.export.py --verify` が段ごとに maxdiff を採るため。
    """
    decoder = model.decoder
    decoder.conv_out1 = TailConv(decoder.conv_out1, _tail_head_channels(decoder))
    module = sys.modules[type(model).__module__]
    module.Decoder.forward = _decoder_forward
    return {"tail_conv": 1}


def apply(model: nn.Module) -> dict[str, int]:
    """差し替えを全て当てる（export 経路 — ①〜⑥ + ⑦⑧ + ⑨）。

    戻り値は差し替えたモジュール数。
    """
    return {
        **apply_layout_patches(model),
        **apply_module_patches(model),
        **apply_tail_patches(model),
    }


def prepare(model: nn.Module, sample: torch.Tensor) -> Any:
    """解像度依存の定数（shifted-window マスク）を焼き、パッチ後の参照出力を返す。

    MUST: export の直前に、**export と同じ形の入力**で通すこと。マスクは
    `(padded_height, padded_width)` ごとの別バッファで、形が違えば焼かれていない。
    """
    global _MASK_BUILD_ENABLED
    if not _APPLIED:
        raise RuntimeError("apply(model) が先 — 素の forward はマスクバッファを引かない")
    _MASK_BUILD_ENABLED = True
    try:
        with torch.no_grad():
            return model(sample)
    finally:
        _MASK_BUILD_ENABLED = False
