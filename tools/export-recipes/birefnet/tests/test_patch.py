"""`birefnet.patch` の書き換えが原実装と同値であることの回帰テスト（実重み不要分）。

原実装（HF `ZhengPeng7/BiRefNet_HR` 同梱の `birefnet.py`）を**逐語で写した参照**をこの
ファイル内に置き、パッチ側と突き合わせる。実重み 1024² での端から端までの同値は
export 台本側の責務で、ここは「どの書き換えがどの強さで同値か」を単体で固定する。

主張の強さは 2 段（`birefnet.patch` の docstring と対応）:

- **データの並べ替えだけの書き換え**（窓の切り出し / qkv の 3 等分 / roll / pad / PatchMerging /
  image2patches / 窓マスク）は **`torch.equal` = ビット一致**。演算を 1 つも増減させないので、
  「差が小さい」で通す形にすると H/W の取り違えや窓の並び違いが素通りする。
- **式は同値・丸めだけ違う書き換え**（BatchNorm2d の per-channel affine 化 / AdaptiveAvgPool2d
  の 2 段 sum 化 / decoder 末尾の 1×1 conv と upsample の順序交換）は近い値まで。ATen 側が
  積和を FMA で畳む / 2 軸を同時に縮約する / 縮約を伸ばす前に済ませるぶんだけ最下位ビットが
  動く。tolerance は f32 の丸め幅に留めて、α/β の取り違えやチャネルの割り方の取り違えが
  通らないようにする（割り方の取り違えが実際に落ちることは故障注入で実測する）。

参照は**非正方**（H ≠ W）・**窓で割り切れない**形を混ぜる — 正方形だけで見ると H/W の
入れ替えがビット一致のまま通る。
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from birefnet import patch as patch_birefnet

#: BatchNorm2d の affine 化の許容差（実測 max 2.4e-07 — ATen の FMA との差）。
BATCH_NORM_ATOL = 1e-6

#: AdaptiveAvgPool2d の 2 段 sum 化の許容差（実測 max 1.1e-08 — 縮約順序の差）。
MEAN_ATOL = 1e-7

#: decoder 末尾の順序交換の許容差（実測 max 2.0e-5）。他の 2 つより桁が大きいのは**出力の
#: 絶対値が大きい**ため — 下のテストは重みを `normal_()` で振るので 240 チャネルの積和が
#: 絶対値 ~55 まで出る（相対では 3.5e-07 = f32 の丸め幅の数倍で、⑦の 2.4e-07 と同じ桁）。
TAIL_ATOL = 5e-5


# ---- 原実装の逐語写し ------------------------------------------------------


def ref_window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """原実装 `window_partition`（rank-6 view + permute）。"""
    batch, height, width, channels = x.shape
    x = x.view(
        batch, height // window_size, window_size, width // window_size, window_size, channels
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)


def ref_window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int
) -> torch.Tensor:
    """原実装 `window_reverse`（rank-6 view + permute）。"""
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(
        batch, height // window_size, width // window_size, window_size, window_size, -1
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


def ref_image2patches(image: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """原実装 `image2patches`（einops `b c (hg h) (wg w) -> b (c hg wg) h w`）を素の torch で。

    einops を持たない環境でも回る独立オラクル（添字を素で写す）。einops 本体との突合は
    {@link test_image2patches_matches_einops} が einops のある環境でだけ行う。
    """
    batch, channels, height, width = image.shape
    patch_h, patch_w = height // grid_h, width // grid_w
    out = torch.empty((batch, channels * grid_h * grid_w, patch_h, patch_w), dtype=image.dtype)
    for b in range(batch):
        for c in range(channels):
            for hg in range(grid_h):
                for wg in range(grid_w):
                    target = (c * grid_h + hg) * grid_w + wg
                    out[b, target] = image[
                        b, c, hg * patch_h : (hg + 1) * patch_h, wg * patch_w : (wg + 1) * patch_w
                    ]
    return out


def ref_attn_mask(
    padded_height: int, padded_width: int, window_size: int, shift_size: int
) -> torch.Tensor:
    """原実装 `BasicLayer.forward` の窓マスク生成（in-place 代入 + masked_fill）。"""
    img_mask = torch.zeros((1, padded_height, padded_width, 1))
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
    mask_windows = ref_window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


def ref_decoder_tail(
    conv: nn.Conv2d, head: torch.Tensor, size: tuple[int, int], ipt: torch.Tensor
) -> torch.Tensor:
    """原実装 `Decoder.forward` の末尾（2226〜2231 行）— 伸ばす → `cat` → 1×1 conv。"""
    upsampled = nn.functional.interpolate(head, size=size, mode="bilinear", align_corners=True)
    return conv(torch.cat((upsampled, ipt), 1))


class RefWindowAttention(nn.Module):
    """原実装 `WindowAttention` の骨格（forward は逐語写し・`SDPA_enabled=False` 側）。"""

    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_windows, tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = query * self.scale
        attn = query @ key.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            windows = mask.shape[0]
            attn = attn.view(
                batch_windows // windows, windows, self.num_heads, tokens, tokens
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, tokens, tokens)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        out = (attn @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj_drop(self.proj(out))


class RefSwinBlock(nn.Module):
    """原実装 `SwinTransformerBlock` の骨格（forward は逐語写し）。"""

    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RefWindowAttention(dim, window_size, num_heads)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.H: int | None = None
        self.W: int | None = None

    def forward(self, x: torch.Tensor, mask_matrix: torch.Tensor | None) -> torch.Tensor:
        batch, length, channels = x.shape
        height, width = self.H, self.W
        assert length == height * width
        shortcut = x
        x = self.norm1(x)
        x = x.view(batch, height, width, channels)

        pad_r = (self.window_size - width % self.window_size) % self.window_size
        pad_b = (self.window_size - height % self.window_size) % self.window_size
        x = nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, padded_height, padded_width, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = ref_window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)
        shifted_x = ref_window_reverse(attn_windows, self.window_size, padded_height, padded_width)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :height, :width, :].contiguous()
        x = x.view(batch, height * width, channels)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class RefPatchMerging(nn.Module):
    """原実装 `PatchMerging` の骨格（forward は逐語写し）。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, length, channels = x.shape
        assert length == height * width
        x = x.view(batch, height, width, channels)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(batch, -1, 4 * channels)
        return self.reduction(self.norm(x))


# ---- ①窓の切り出し / 復元 --------------------------------------------------


@pytest.mark.parametrize(
    ("batch", "height", "width", "channels", "window"),
    [(1, 12, 12, 6, 6), (1, 24, 12, 4, 12), (2, 12, 24, 3, 4), (1, 264, 132, 2, 12)],
)
def test_window_partition_matches_reference(
    batch: int, height: int, width: int, channels: int, window: int
) -> None:
    x = torch.randn(batch, height, width, channels)
    got = patch_birefnet._window_partition(x, window)
    assert torch.equal(got, ref_window_partition(x, window))


@pytest.mark.parametrize(
    ("batch", "height", "width", "channels", "window"),
    [(1, 12, 12, 6, 6), (1, 24, 12, 4, 12), (2, 12, 24, 3, 4)],
)
def test_window_reverse_matches_reference(
    batch: int, height: int, width: int, channels: int, window: int
) -> None:
    windows = torch.randn(batch * (height // window) * (width // window), window, window, channels)
    assert torch.equal(
        patch_birefnet._window_reverse(windows, window, height, width),
        ref_window_reverse(windows, window, height, width),
    )


def test_window_roundtrip_is_identity() -> None:
    x = torch.randn(2, 24, 36, 5)
    windows = patch_birefnet._window_partition(x, 12)
    assert torch.equal(patch_birefnet._window_reverse(windows, 12, 24, 36), x)


# ---- ③roll / pad -----------------------------------------------------------


@pytest.mark.parametrize("shift", [-6, -1, 0, 1, 6])
@pytest.mark.parametrize("dim", [1, 2])
def test_roll_matches_torch(shift: int, dim: int) -> None:
    x = torch.randn(1, 13, 17, 3)
    assert torch.equal(patch_birefnet._roll(x, shift, dim), torch.roll(x, shift, dim))


@pytest.mark.parametrize(("dim", "width"), [(1, 0), (1, 5), (2, 3)])
def test_pad_zero_matches_functional_pad(dim: int, width: int) -> None:
    x = torch.randn(1, 7, 11, 3)
    pads = [0, 0, 0, 0, 0, 0]
    # F.pad の並びは最終次元から（dim 2 = 位置 2..3 / dim 1 = 位置 4..5）
    pads[2 * (3 - dim) + 1] = width
    assert torch.equal(patch_birefnet._pad_zero(x, dim, width), nn.functional.pad(x, pads))


# ---- ②窓 attention ---------------------------------------------------------


@pytest.mark.parametrize("with_mask", [False, True])
def test_window_attention_matches_reference(with_mask: bool) -> None:
    torch.manual_seed(0)
    window, heads, dim = 4, 3, 12
    attention = RefWindowAttention(dim, window, heads).eval()
    windows = 6
    x = torch.randn(windows, window * window, dim)
    mask = ref_attn_mask(12, 8, window, window // 2) if with_mask else None
    if with_mask:
        assert mask.shape[0] == windows
    with torch.no_grad():
        expected = attention(x, mask=mask)
        got = patch_birefnet._window_attention_forward(attention, x, mask=mask)
    assert torch.equal(expected, got)


def test_window_attention_rejects_batched_mask() -> None:
    """窓数と第 0 軸が食い違う形（バッチ > 1）は fail loudly。"""
    window, heads, dim = 4, 3, 12
    attention = RefWindowAttention(dim, window, heads).eval()
    windows = 6
    x = torch.randn(2 * windows, window * window, dim)
    mask = ref_attn_mask(12, 8, window, window // 2)
    with pytest.raises(NotImplementedError, match="B = 1"):
        patch_birefnet._window_attention_forward(attention, x, mask=mask)


# ---- ③Swin ブロック --------------------------------------------------------


@pytest.mark.parametrize(("height", "width"), [(8, 8), (10, 6), (13, 9)])
@pytest.mark.parametrize("shift", [0, 2])
def test_swin_block_matches_reference(height: int, width: int, shift: int) -> None:
    torch.manual_seed(1)
    window, heads, dim = 4, 3, 12
    block = RefSwinBlock(dim, heads, window, shift).eval()
    block.H, block.W = height, width
    padded_height = -(-height // window) * window
    padded_width = -(-width // window) * window
    mask = ref_attn_mask(padded_height, padded_width, window, window // 2)
    x = torch.randn(1, height * width, dim)
    with torch.no_grad():
        expected = block(x, mask)
        got = patch_birefnet._swin_block_forward(block, x, mask)
    assert torch.equal(expected, got)


# ---- ④窓マスク -------------------------------------------------------------


@pytest.mark.parametrize(
    ("padded_height", "padded_width", "window"), [(12, 12, 4), (24, 12, 12), (264, 132, 12)]
)
def test_attn_mask_matches_reference(padded_height: int, padded_width: int, window: int) -> None:
    got = patch_birefnet._build_attn_mask(
        padded_height, padded_width, window, window // 2, torch.float32, torch.device("cpu")
    )
    assert torch.equal(got, ref_attn_mask(padded_height, padded_width, window, window // 2))


# ---- ⑤PatchMerging ---------------------------------------------------------


@pytest.mark.parametrize(("height", "width"), [(4, 4), (8, 6), (6, 10)])
def test_patch_merging_matches_reference(height: int, width: int) -> None:
    torch.manual_seed(2)
    merging = RefPatchMerging(5).eval()
    x = torch.randn(1, height * width, 5)
    with torch.no_grad():
        expected = merging(x, height, width)
        got = patch_birefnet._patch_merging_forward(merging, x, height, width)
    assert torch.equal(expected, got)


def test_patch_merging_rejects_odd_resolution() -> None:
    """原実装の pad 分岐（奇数解像度）は差し替え版が持たない — 黙って別の形にしない。"""
    merging = RefPatchMerging(5).eval()
    x = torch.randn(1, 5 * 4, 5)
    with pytest.raises(NotImplementedError, match="奇数解像度"):
        patch_birefnet._patch_merging_forward(merging, x, 5, 4)


# ---- ⑥image2patches --------------------------------------------------------


@pytest.mark.parametrize(("grid_h", "grid_w"), [(2, 2), (4, 2), (8, 8)])
def test_image2patches_matches_reference(grid_h: int, grid_w: int) -> None:
    image = torch.randn(1, 3, 8 * grid_h, 4 * grid_w)
    reference = torch.randn(1, 1, 8, 4)
    got = patch_birefnet._image2patches(image, patch_ref=reference)
    assert torch.equal(got, ref_image2patches(image, grid_h, grid_w))


def test_image2patches_matches_einops() -> None:
    """einops のある環境では本家の `rearrange` とも突き合わせる。"""
    rearrange = pytest.importorskip("einops").rearrange
    image = torch.randn(1, 3, 16, 12)
    got = patch_birefnet._image2patches(image, grid_h=2, grid_w=3)
    expected = rearrange(image, patch_birefnet._SUPPORTED_TRANSFORMATION, hg=2, wg=3)
    assert torch.equal(got, expected)


def test_image2patches_rejects_other_transformation() -> None:
    with pytest.raises(NotImplementedError, match="変換式"):
        patch_birefnet._image2patches(torch.randn(1, 3, 4, 4), transformation="b c h w -> b c w h")


# ---- ⑦⑧ モジュールの差し替え ---------------------------------------------


@pytest.mark.parametrize(("channels", "height", "width"), [(4, 5, 3), (16, 8, 8)])
def test_channel_affine_matches_batch_norm(channels: int, height: int, width: int) -> None:
    torch.manual_seed(3)
    norm = nn.BatchNorm2d(channels)
    norm.weight.data.normal_()
    norm.bias.data.normal_()
    norm.running_mean.data.normal_()
    norm.running_var.data.uniform_(0.1, 3.0)
    norm.eval()
    x = torch.randn(1, channels, height, width)
    with torch.no_grad():
        expected = norm(x)
        got = patch_birefnet.ChannelAffine(norm)(x)
    # ビット一致ではない（ATen の推論カーネルが積和を FMA で畳む）— docstring 参照。
    assert torch.allclose(expected, got, rtol=0, atol=BATCH_NORM_ATOL)


def test_channel_affine_rejects_training_mode() -> None:
    with pytest.raises(ValueError, match="学習モード"):
        patch_birefnet.ChannelAffine(nn.BatchNorm2d(4))


@pytest.mark.parametrize(("channels", "height", "width"), [(4, 5, 3), (8, 32, 32)])
def test_spatial_mean_matches_adaptive_avg_pool(channels: int, height: int, width: int) -> None:
    torch.manual_seed(4)
    x = torch.randn(1, channels, height, width)
    with torch.no_grad():
        expected = nn.AdaptiveAvgPool2d((1, 1))(x)
        got = patch_birefnet.SpatialMean()(x)
    assert got.shape == expected.shape
    # ビット一致ではない（2 軸同時縮約と 1 軸ずつの縮約の順序差）— docstring 参照。
    assert torch.allclose(expected, got, rtol=0, atol=MEAN_ATOL)


def test_replace_modules_swaps_every_instance() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 4, 1),
        nn.BatchNorm2d(4),
        nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.BatchNorm2d(4)),
    ).eval()
    x = torch.randn(1, 3, 6, 6)
    with torch.no_grad():
        expected = model(x)
    counts = patch_birefnet._replace_modules(model)
    assert counts == {"batch_norm": 2, "adaptive_avg_pool": 1}
    assert not [m for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.AdaptiveAvgPool2d))]
    with torch.no_grad():
        got = model(x)
    assert torch.allclose(expected, got, rtol=0, atol=BATCH_NORM_ATOL)


def test_replace_modules_rejects_non_unit_pooling() -> None:
    model = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)))
    with pytest.raises(NotImplementedError, match="AdaptiveAvgPool2d"):
        patch_birefnet._replace_modules(model)


def test_lift_relative_position_index_keeps_values() -> None:
    """バッファ → 素の属性（lifted 定数）への降ろしで値が 1 要素も変わらないこと。"""
    attention = RefWindowAttention(12, 4, 3)
    before = copy.deepcopy(attention.relative_position_index)
    assert "relative_position_index" in dict(attention.named_buffers())
    assert patch_birefnet._lift_relative_position_index(attention) == 1
    assert "relative_position_index" not in dict(attention.named_buffers())
    assert torch.equal(attention.relative_position_index, before)
    assert patch_birefnet._lift_relative_position_index(attention) == 0


# ---- ⑨decoder 末尾（1×1 conv と upsample の順序交換） ----------------------


def _tail_source(in_channels: int) -> nn.Sequential:
    """`Decoder.conv_out1` の席（`nn.Sequential(nn.Conv2d(C, 1, 1, 1, 0))`）。

    重みは `normal_()` で振る — 既定初期化のままだと絶対値が小さすぎて、チャネルの割り方の
    取り違え（下の故障注入）が丸め差に埋もれかねない。
    """
    conv = nn.Conv2d(in_channels, 1, 1, 1, 0)
    conv.weight.data.normal_()
    conv.bias.data.normal_()
    return nn.Sequential(conv).eval()


class TestTailConv:
    """末尾の代数書き換えが原実装の末尾と同値 / 前提を外れたら落ちる。"""

    @pytest.mark.parametrize(
        ("head_channels", "ipt_channels", "small", "size"),
        [
            (192, 48, (16, 16), (64, 64)),  # 実物の割り方（channels[3]//2 と channels[3]//8）
            (12, 5, (5, 3), (20, 12)),  # 非正方・割り切れない拡大率
            (8, 8, (7, 9), (7, 9)),  # 伸ばさない（size が head と同じ）
        ],
    )
    def test_it_matches_the_upstream_tail(
        self,
        head_channels: int,
        ipt_channels: int,
        small: tuple[int, int],
        size: tuple[int, int],
    ) -> None:
        torch.manual_seed(6)
        source = _tail_source(head_channels + ipt_channels)
        head = torch.randn(1, head_channels, *small)
        ipt = torch.randn(1, ipt_channels, *size)

        with torch.no_grad():
            expected = ref_decoder_tail(source[0], head, size, ipt)
            got = patch_birefnet.TailConv(source, head_channels)(head, size, ipt)

        assert got.shape == expected.shape
        # ビット一致ではない（縮約を伸ばす前に済ませる順序差）— docstring 参照。
        assert torch.allclose(expected, got, rtol=0, atol=TAIL_ATOL)

    def test_a_swapped_channel_split_does_not_match(self) -> None:
        """故障注入: head / ipt の重みを入れ替えると参照から外れる（許容差が空振りでない）。"""
        torch.manual_seed(7)
        channels = 6
        source = _tail_source(2 * channels)
        head = torch.randn(1, channels, 4, 4)
        ipt = torch.randn(1, channels, 8, 8)
        size = (8, 8)
        swapped = patch_birefnet.TailConv(source, channels)
        swapped.head_weight, swapped.ipt_weight = (
            swapped.ipt_weight.clone(),
            swapped.head_weight.clone(),
        )

        with torch.no_grad():
            expected = ref_decoder_tail(source[0], head, size, ipt)
            intact = patch_birefnet.TailConv(source, channels)(head, size, ipt)
            got = swapped(head, size, ipt)

        assert torch.allclose(expected, intact, rtol=0, atol=TAIL_ATOL)
        assert not torch.allclose(expected, got, rtol=0, atol=TAIL_ATOL)

    def test_a_head_split_outside_the_input_is_rejected(self) -> None:
        with pytest.raises(NotImplementedError, match="噛み合わない"):
            patch_birefnet.TailConv(_tail_source(8), 8)

    @pytest.mark.parametrize(
        ("source", "match"),
        [
            (nn.Conv2d(8, 1, 1), "nn.Sequential"),
            (nn.Sequential(nn.Conv2d(8, 1, 1), nn.ReLU()), "nn.Sequential"),
            (nn.Sequential(nn.ReLU()), "nn.Conv2d"),
            (nn.Sequential(nn.Conv2d(8, 1, 3, 1, 1)), "1×1"),
            (nn.Sequential(nn.Conv2d(8, 1, 1, 2, 0)), "1×1"),
            (nn.Sequential(nn.Conv2d(8, 1, 1, 1, 0, bias=False)), "bias"),
        ],
    )
    def test_an_unsupported_tail_shape_fails_loudly(self, source: nn.Module, match: str) -> None:
        """1×1・stride 1・pad 0・groups 1・bias あり以外は upsample と可換にならない。"""
        with pytest.raises(NotImplementedError, match=match):
            patch_birefnet.TailConv(source, 4)


def _decoder_shape_stub(
    head_channels: int = 192, ipt_channels: int = 48, tail_in_channels: int = 240
) -> SimpleNamespace:
    """`_tail_head_channels` が触る 3 つの数だけを持つ decoder の席。"""
    return SimpleNamespace(
        decoder_block1=SimpleNamespace(conv_out=SimpleNamespace(out_channels=head_channels)),
        ipt_blk1=SimpleNamespace(conv_out=SimpleNamespace(out_channels=ipt_channels)),
        conv_out1=[SimpleNamespace(in_channels=tail_in_channels)],
    )


class TestTailHeadChannels:
    """head 側のチャネル数は 2 通りに導けるので、突き合わせてから使う。"""

    def test_the_two_derivations_agree_on_the_reference_shape(self) -> None:
        """正常系（これが無いと以下の否定形が恒真になる）。"""
        assert patch_birefnet._tail_head_channels(_decoder_shape_stub()) == 192

    def test_a_disagreement_fails_loudly(self) -> None:
        """故障注入: ipt 枝の出力を 1 本ずらすと 2 つの導出が食い違って落ちる。"""
        with pytest.raises(NotImplementedError, match="食い違う"):
            patch_birefnet._tail_head_channels(_decoder_shape_stub(ipt_channels=47))


class TestDecoderForward:
    """差し替えた `Decoder.forward` は eval 経路しか持たない。"""

    def test_training_mode_is_rejected(self) -> None:
        """学習経路（ms_supervision の append / gdt の GT）を写していないので落とす。"""
        decoder = nn.Module()
        assert decoder.training

        with pytest.raises(RuntimeError, match="学習モード"):
            patch_birefnet._decoder_forward(decoder, ())

    def test_eval_mode_gets_past_the_gate(self) -> None:
        """上の否定形が恒真でないこと — eval では門を通り、features の分解まで進む。"""
        decoder = nn.Module().eval()

        with pytest.raises(ValueError, match="unpack"):
            patch_birefnet._decoder_forward(decoder, ())


# ---- 構成の検査（外れても shape エラーにならない欄） ------------------------


def _supported_config(**overrides: object) -> SimpleNamespace:
    """`assert_supported` が通す構成（上流クラスは `trust_remote_code` 由来で import 不可）。"""
    config = SimpleNamespace(
        bb="swin_v1_l",
        dec_att="ASPPDeformable",
        dec_blk="BasicDecBlk",
        mul_scl_ipt="cat",
        squeeze_block="BasicDecBlk_x1",
        refine="",
        dec_ipt=True,
        dec_ipt_split=True,
        batch_size=2,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _supported_model(ape: bool = False, **overrides: object) -> SimpleNamespace:
    """`assert_supported` が触るのは `config.<key>` と `model.bb.ape` だけ。"""
    return SimpleNamespace(config=_supported_config(**overrides), bb=SimpleNamespace(ape=ape))


class TestAssertSupported:
    """差し替え版が前提にする構成の検査（外れても別の**数値経路**へ落ちるものだけ）。"""

    def test_the_reference_configuration_passes(self) -> None:
        """正常系（これが無いと以下の否定形が恒真になる）。"""
        patch_birefnet.assert_supported(_supported_model())

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("bb", "swin_v1_b"),
            ("dec_att", "ASPP"),
            ("dec_blk", "ResBlk"),
            ("mul_scl_ipt", "add"),
            ("squeeze_block", "BasicDecBlk_x3"),
            ("refine", "RefUNet"),
        ],
    )
    def test_a_different_component_is_rejected(self, key: str, value: str) -> None:
        """6 欄はどれも「外れると別の重み・別のグラフ」— 文言に欄名と期待値が出る。"""
        with pytest.raises(ValueError, match=f"config.{key}"):
            patch_birefnet.assert_supported(_supported_model(**{key: value}))

    def test_the_expected_backbone_is_named_in_the_message(self) -> None:
        with pytest.raises(ValueError, match="swin_v1_l"):
            patch_birefnet.assert_supported(_supported_model(bb="swin_v1_b"))

    @pytest.mark.parametrize("key", ["dec_ipt", "dec_ipt_split"])
    def test_the_patch_split_input_path_is_required(self, key: str) -> None:
        """`dec_ipt` 系が False だと image2patches の書き換えが噛み合わない。"""
        with pytest.raises(ValueError, match="image2patches"):
            patch_birefnet.assert_supported(_supported_model(**{key: False}))

    def test_a_batch_size_of_one_is_rejected(self) -> None:
        """`batch_size <= 1` では上流の `BatchNorm2d` が Identity になり重みの前提と違う。"""
        with pytest.raises(ValueError, match="BatchNorm2d"):
            patch_birefnet.assert_supported(_supported_model(batch_size=1))

    def test_absolute_position_embeddings_are_rejected(self) -> None:
        """`ape=True` は bicubic の位置埋め込み補間（語彙に無い）を出す経路。"""
        with pytest.raises(ValueError, match="ape"):
            patch_birefnet.assert_supported(_supported_model(ape=True))


# ---- BasicLayer.forward（焼いたマスクを引く） ------------------------------


class _NoopDownsample(nn.Module):
    """`BasicLayer.downsample` の席（戻り値の形は問わない — 見るのは末尾の 2 数）。"""

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return x


class _MinimalBasicLayer(nn.Module):
    """`BasicLayer` の最小の骨格（`_basic_layer_forward` が触る属性だけ）。"""

    def __init__(self, use_checkpoint: bool = False, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.window_size = 4
        self.shift_size = 2
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([RefSwinBlock(12, 3, 4, 0), RefSwinBlock(12, 3, 4, 2)])
        self.downsample = downsample


class TestBasicLayerForward:
    """マスクは**焼いたバッファ**からしか引かない / 分岐が正しく並ぶ。"""

    HEIGHT = 10
    WIDTH = 6
    PADDED_HEIGHT = 12
    PADDED_WIDTH = 8

    @staticmethod
    def _input(height: int = HEIGHT, width: int = WIDTH) -> torch.Tensor:
        torch.manual_seed(5)
        return torch.randn(1, height * width, 12)

    def test_an_unbaked_mask_fails_loudly(self) -> None:
        """MUST: export 中に `register_buffer` が走る（定数が黙って生える）形を作らない。"""
        layer = _MinimalBasicLayer()

        with pytest.raises(RuntimeError, match=r"12×8 の窓マスクが焼かれていない"):
            patch_birefnet._basic_layer_forward(layer, self._input(), self.HEIGHT, self.WIDTH)

        with pytest.raises(RuntimeError, match="prepare"):
            patch_birefnet._basic_layer_forward(layer, self._input(), self.HEIGHT, self.WIDTH)

    def test_the_build_window_bakes_the_reference_mask(self, monkeypatch) -> None:
        """ビルド窓を開けた 1 回だけバッファが生え、値は `_build_attn_mask` と一致する。"""
        monkeypatch.setattr(patch_birefnet, "_MASK_BUILD_ENABLED", True)
        layer = _MinimalBasicLayer()

        with torch.no_grad():
            patch_birefnet._basic_layer_forward(layer, self._input(), self.HEIGHT, self.WIDTH)

        name = f"karume_attn_mask_{self.PADDED_HEIGHT}x{self.PADDED_WIDTH}"
        baked = getattr(layer, name)
        assert torch.equal(
            baked,
            patch_birefnet._build_attn_mask(
                self.PADDED_HEIGHT,
                self.PADDED_WIDTH,
                layer.window_size,
                layer.shift_size,
                baked.dtype,
                baked.device,
            ),
        )

    def test_checkpointing_is_rejected(self, monkeypatch) -> None:
        """`use_checkpoint=True` は差し替え版が持たない経路（黙って通さない）。"""
        monkeypatch.setattr(patch_birefnet, "_MASK_BUILD_ENABLED", True)
        layer = _MinimalBasicLayer(use_checkpoint=True)

        with pytest.raises(NotImplementedError, match="use_checkpoint"):
            patch_birefnet._basic_layer_forward(layer, self._input(), self.HEIGHT, self.WIDTH)

    def test_without_a_downsample_the_six_slots_repeat_the_input_resolution(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(patch_birefnet, "_MASK_BUILD_ENABLED", True)
        layer = _MinimalBasicLayer()

        with torch.no_grad():
            got = patch_birefnet._basic_layer_forward(layer, self._input(), self.HEIGHT, self.WIDTH)

        assert got[1:3] == (self.HEIGHT, self.WIDTH)
        assert got[4:] == (self.HEIGHT, self.WIDTH)
        assert got[0] is got[3]

    def test_a_downsample_halves_the_trailing_resolution(self, monkeypatch) -> None:
        """`(H+1)//2` / `(W+1)//2` — 奇数側の切り上げが落ちると次の段の形がずれる。"""
        monkeypatch.setattr(patch_birefnet, "_MASK_BUILD_ENABLED", True)
        layer = _MinimalBasicLayer(downsample=_NoopDownsample())

        with torch.no_grad():
            got = patch_birefnet._basic_layer_forward(layer, self._input(9, 7), 9, 7)

        assert got[1:3] == (9, 7)
        assert got[4:] == (5, 4)


# ---- prepare / apply の門 ---------------------------------------------------


class _ExplodingModel(nn.Module):
    """`prepare` の中で落ちるモデル（`finally` の復元を観測するための席）。"""

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("上流の forward が落ちた")


class TestPrepareGate:
    """`apply` より先に `prepare` を呼べない / ビルド窓は例外経路でも閉じる。"""

    def test_prepare_before_apply_fails_loudly(self, monkeypatch) -> None:
        """素の forward はマスクバッファを引かないので、焼く前に落とす。"""
        monkeypatch.setattr(patch_birefnet, "_APPLIED", False)

        with pytest.raises(RuntimeError, match=r"apply\(model\)"):
            patch_birefnet.prepare(_ExplodingModel(), torch.zeros(1))

    def test_the_build_window_closes_even_when_prepare_raises(self, monkeypatch) -> None:
        """MUST: 窓が開いたままだと、以後どこでもマスクが黙って焼ける。"""
        monkeypatch.setattr(patch_birefnet, "_APPLIED", True)
        monkeypatch.setattr(patch_birefnet, "_MASK_BUILD_ENABLED", False)

        with pytest.raises(RuntimeError, match="上流の forward が落ちた"):
            patch_birefnet.prepare(_ExplodingModel(), torch.zeros(1))

        layer = _MinimalBasicLayer()
        with pytest.raises(RuntimeError, match="焼かれていない"):
            patch_birefnet._basic_layer_forward(layer, TestBasicLayerForward._input(), 10, 6)
