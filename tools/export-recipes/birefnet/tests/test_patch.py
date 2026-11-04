"""Regression tests that the `birefnet.patch` rewrites are equivalent to the original
implementation (the part that needs no real weights).

A **verbatim copy** of the original implementation (`birefnet.py` shipped with HF
`ZhengPeng7/BiRefNet_HR`) lives in this file as the reference, and the patched side is compared
with it. End-to-end equivalence with the real weights at 1024² is the export script's job; this
file pins, unit by unit, "which rewrite is equivalent and how strongly".

The strength of the claim has two levels (matching the `birefnet.patch` docstring):

- **Rewrites that only reorder data** (window partition / splitting qkv in three / roll / pad /
  PatchMerging / image2patches / window mask) are **`torch.equal` = bit-identical**. They add or
  remove no operation, so accepting them on "the difference is small" would let an H/W swap or a
  wrong window order slip through.
- **Rewrites with an equivalent formula that differ only in rounding** (BatchNorm2d as a
  per-channel affine / AdaptiveAvgPool2d as a two-step sum / swapping the order of the 1×1 conv
  and the upsample at the end of the decoder) are checked to close values. The last bits move by
  as much as ATen folds the multiply-add into an FMA / reduces both axes at once / finishes the
  reduction before the upsample. The tolerance stays at the f32 rounding width, so that an α/β
  swap or a wrong channel split does not pass (fault injection measures that a wrong split
  actually fails).

The references mix **non-square** (H ≠ W) shapes and shapes **not divisible by the window** —
with squares only, an H/W swap stays bit-identical and passes.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from birefnet import patch as patch_birefnet

#: Tolerance of BatchNorm2d as an affine (measured max 2.4e-07 — the difference from ATen's FMA).
BATCH_NORM_ATOL = 1e-6

#: Tolerance of AdaptiveAvgPool2d as a two-step sum (measured max 1.1e-08 — reduction order).
MEAN_ATOL = 1e-7

#: Tolerance of the reordered decoder tail (measured max 2.0e-5). It is an order of magnitude
#: above the other two because **the outputs are large in absolute value** — the test below draws
#: the weights with `normal_()`, so the 240-channel multiply-add reaches ~55 in absolute value
#: (relative 3.5e-07 = a few f32 rounding widths, the same order as ⑦'s 2.4e-07).
TAIL_ATOL = 5e-5


# ---- Verbatim copy of the original implementation --------------------------


def ref_window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """The original `window_partition` (rank-6 view + permute)."""
    batch, height, width, channels = x.shape
    x = x.view(
        batch, height // window_size, window_size, width // window_size, window_size, channels
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)


def ref_window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int
) -> torch.Tensor:
    """The original `window_reverse` (rank-6 view + permute)."""
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(
        batch, height // window_size, width // window_size, window_size, window_size, -1
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


def ref_image2patches(image: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """The original `image2patches` (einops `b c (hg h) (wg w) -> b (c hg wg) h w`) in plain torch.

    An independent oracle that runs without einops (it spells the indices out). The comparison
    with einops itself is {@link test_image2patches_matches_einops}, only where einops exists.
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
    """The window-mask construction of the original `BasicLayer.forward` (in-place assignment +
    masked_fill)."""
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


def _bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Same dtype, shape and bits (`torch.equal` treats `0.0` and `-0.0` as equal)."""
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
    )


def ref_decoder_tail(
    conv: nn.Conv2d, head: torch.Tensor, size: tuple[int, int], ipt: torch.Tensor
) -> torch.Tensor:
    """The tail of the original `Decoder.forward` (lines 2226–2231) — upsample → `cat` → 1×1
    conv."""
    upsampled = nn.functional.interpolate(head, size=size, mode="bilinear", align_corners=True)
    return conv(torch.cat((upsampled, ipt), 1))


class RefWindowAttention(nn.Module):
    """A skeleton of the original `WindowAttention` (forward copied verbatim, the
    `SDPA_enabled=False` side)."""

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
    """A skeleton of the original `SwinTransformerBlock` (forward copied verbatim)."""

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
    """A skeleton of the original `PatchMerging` (forward copied verbatim)."""

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


# ---- ① Window partition / reverse ------------------------------------------


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


# ---- ③ roll / pad ----------------------------------------------------------


@pytest.mark.parametrize("shift", [-6, -1, 0, 1, 6])
@pytest.mark.parametrize("dim", [1, 2])
def test_roll_matches_torch(shift: int, dim: int) -> None:
    x = torch.randn(1, 13, 17, 3)
    assert torch.equal(patch_birefnet._roll(x, shift, dim), torch.roll(x, shift, dim))


@pytest.mark.parametrize(("dim", "width"), [(1, 0), (1, 5), (2, 3)])
def test_pad_zero_matches_functional_pad(dim: int, width: int) -> None:
    x = torch.randn(1, 7, 11, 3)
    pads = [0, 0, 0, 0, 0, 0]
    # F.pad lists dimensions from the last one (dim 2 = positions 2..3 / dim 1 = positions 4..5)
    pads[2 * (3 - dim) + 1] = width
    assert torch.equal(patch_birefnet._pad_zero(x, dim, width), nn.functional.pad(x, pads))


class TestPadZeroAtExportSizes:
    """③'s zero pad is bit-identical to `F.pad` in values and shape at real sizes too, and bakes
    no huge constant into the export."""

    @pytest.mark.parametrize(
        ("shape", "dim", "width"),
        [
            ((1, 256, 256, 2), 1, 8),  # stage 0 of 1024² (256 → 264)
            ((1, 264, 256, 2), 2, 8),
            ((1, 512, 512, 1), 1, 4),  # stage 0 of 2048² (512 → 516)
            ((1, 516, 512, 1), 2, 4),
        ],
    )
    def test_the_padding_matches_functional_pad_bit_for_bit(
        self, shape: tuple[int, ...], dim: int, width: int
    ) -> None:
        x = torch.randn(shape)
        pads = [0, 0, 0, 0, 0, 0]
        pads[2 * (3 - dim) + 1] = width
        assert _bits_equal(patch_birefnet._pad_zero(x, dim, width), nn.functional.pad(x, pads))

    def test_the_exported_graph_bakes_no_zero_block(self) -> None:
        """The zeros are a one-element constant widened by `expand` at run time (no all-zero
        initializer is baked).

        The earlier `torch.zeros(shape)` was baked by constant folding into an all-zero
        `[1, pad, W, C]`, one cause of the long zero runs in the 2048² container (fragmenting HF
        xet transfers) (the `birefnet.patch` docstring).
        """
        from karume.pipeline import export_module

        class Padded(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return patch_birefnet._pad_zero(patch_birefnet._pad_zero(x, 1, 8), 2, 8)

        graph, tensors = export_module(
            Padded().eval(), (torch.randn(1, 64, 56, 4),), symbol_names=()
        )

        assert [node.op for node in graph.nodes].count("expand") == 2
        largest = max(tensors[init.tensor].numel() for init in graph.initializers.values())
        assert largest == 1


# ---- ② Window attention ----------------------------------------------------


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
    """A shape whose axis 0 disagrees with the window count (batch > 1) fails loudly."""
    window, heads, dim = 4, 3, 12
    attention = RefWindowAttention(dim, window, heads).eval()
    windows = 6
    x = torch.randn(2 * windows, window * window, dim)
    mask = ref_attn_mask(12, 8, window, window // 2)
    with pytest.raises(NotImplementedError, match="B = 1"):
        patch_birefnet._window_attention_forward(attention, x, mask=mask)


# ---- ③ Swin block ----------------------------------------------------------


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


# ---- ④ Window mask ---------------------------------------------------------


@pytest.mark.parametrize(
    ("padded_height", "padded_width", "window"), [(12, 12, 4), (24, 12, 12), (264, 132, 12)]
)
def test_attn_mask_matches_reference(padded_height: int, padded_width: int, window: int) -> None:
    got = patch_birefnet._build_attn_mask(
        padded_height, padded_width, window, window // 2, torch.float32, torch.device("cpu")
    )
    assert torch.equal(got, ref_attn_mask(padded_height, padded_width, window, window // 2))


#: The padded sides at which the real model's `BasicLayer` looks up a window mask (window 12,
#: shift 6). The main body rounds S/4, S/8, S/16, S/32, and the half-resolution branch of
#: `mul_scl_ipt='cat'` rounds S/8 to S/64, up to a multiple of 12. 1024² = {264, 132, 72, 36, 24},
#: 2048² = {516, 264, 132, 72, 36} (measured from the initializer names of a re-exported
#: container — 2026-10-16).
EXPORT_PADDED_SIZES = (516, 264, 132, 72, 36, 24)

#: The real model's window (`swin_v1_l`'s window_size). The shift is half of it.
EXPORT_WINDOW = 12


class TestCompactAttnMask:
    """④'s table + index rebuild the upstream mask without changing a single bit / there are at
    most 4 kinds."""

    @staticmethod
    def _compact(height: int, width: int, window: int) -> tuple[torch.Tensor, torch.Tensor]:
        mask = patch_birefnet._build_attn_mask(
            height, width, window, window // 2, torch.float32, torch.device("cpu")
        )
        return patch_birefnet._compact_attn_mask(mask, (height // window, width // window))

    @pytest.mark.parametrize("size", EXPORT_PADDED_SIZES)
    def test_it_reproduces_the_upstream_mask_at_export_sizes(self, size: int) -> None:
        table, index = self._compact(size, size, EXPORT_WINDOW)

        got = patch_birefnet._expand_attn_mask(table, index)

        assert _bits_equal(got, ref_attn_mask(size, size, EXPORT_WINDOW, EXPORT_WINDOW // 2))

    @pytest.mark.parametrize(("height", "width"), [(24, 36), (36, 24), (24, 60)])
    def test_it_reproduces_the_upstream_mask_off_square(self, height: int, width: int) -> None:
        """Non-square (an H/W swap and a swap of the grid's rows / columns fail only here)."""
        table, index = self._compact(height, width, EXPORT_WINDOW)

        got = patch_birefnet._expand_attn_mask(table, index)

        assert _bits_equal(got, ref_attn_mask(height, width, EXPORT_WINDOW, EXPORT_WINDOW // 2))

    @pytest.mark.parametrize("size", EXPORT_PADDED_SIZES)
    def test_the_baked_tensors_are_small(self, size: int) -> None:
        """Only a `[K ≤ 4, N·N]` f32 table and an i64 index over the window grid are baked
        (N = 144)."""
        table, index = self._compact(size, size, EXPORT_WINDOW)
        tokens = EXPORT_WINDOW * EXPORT_WINDOW

        assert table.dtype is torch.float32
        assert table.shape[0] <= patch_birefnet.MAX_MASK_WINDOW_KINDS
        assert table.shape[1] == tokens * tokens
        assert index.dtype is torch.int64
        assert tuple(index.shape) == (size // EXPORT_WINDOW, size // EXPORT_WINDOW)

    def test_table_rows_follow_the_upstream_window_order(self) -> None:
        """The table rows are in order of first appearance in row-major window order (interior →
        right column → bottom row → bottom-right corner)."""
        table, index = self._compact(36, 48, EXPORT_WINDOW)

        flat = index.reshape(-1).tolist()
        first_seen = sorted(range(table.shape[0]), key=flat.index)
        assert first_seen == list(range(table.shape[0]))
        assert index.tolist() == [[0, 0, 0, 1], [0, 0, 0, 1], [2, 2, 2, 3]]
        assert not table[0].any()

    def test_more_than_four_window_kinds_fail_loudly(self) -> None:
        """A mask outside the three-band split (5 or more kinds) is not folded silently."""
        mask = torch.zeros(5, 4, 4)
        for window in range(5):
            mask[window, 0, window % 4] = -100.0
        mask[4, 1, 0] = -100.0

        with pytest.raises(NotImplementedError, match="5 kinds"):
            patch_birefnet._compact_attn_mask(mask, (1, 5))

    def test_the_exported_graph_bakes_the_table_and_an_i32_index(self) -> None:
        """The export bakes only the table and an i32 index, not an `[nW, N, N]` block.

        The rebuild is one `embedding` (the same shape as ②'s relative-position bias). The
        `reshape` that flattens the index from the grid goes through constant folding and becomes
        an i32 constant.
        """
        from karume.pipeline import export_module

        size = 36
        table, index = self._compact(size, size, EXPORT_WINDOW)
        windows = index.numel()
        tokens = EXPORT_WINDOW * EXPORT_WINDOW

        class Masked(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("table", table)
                object.__setattr__(self, "index", index)

            def forward(self, attn: torch.Tensor) -> torch.Tensor:
                mask = patch_birefnet._expand_attn_mask(self.table, self.index)
                return attn + mask.unsqueeze(1)

        graph, tensors = export_module(
            Masked().eval(), (torch.randn(windows, 2, tokens, tokens),), symbol_names=()
        )

        assert [node.op for node in graph.nodes].count("embedding") == 1
        baked = {name: tensors[init.tensor] for name, init in graph.initializers.items()}
        shapes = sorted((str(t.dtype), tuple(t.shape)) for t in baked.values())
        assert shapes == [
            ("torch.float32", tuple(table.shape)),
            ("torch.int32", (windows,)),
        ]


# ---- ⑤ PatchMerging --------------------------------------------------------


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
    """The replacement does not have the original's pad branch (odd resolution) — it does not
    silently produce another shape."""
    merging = RefPatchMerging(5).eval()
    x = torch.randn(1, 5 * 4, 5)
    with pytest.raises(NotImplementedError, match="odd resolution"):
        patch_birefnet._patch_merging_forward(merging, x, 5, 4)


# ---- ⑥ image2patches -------------------------------------------------------


@pytest.mark.parametrize(("grid_h", "grid_w"), [(2, 2), (4, 2), (8, 8)])
def test_image2patches_matches_reference(grid_h: int, grid_w: int) -> None:
    image = torch.randn(1, 3, 8 * grid_h, 4 * grid_w)
    reference = torch.randn(1, 1, 8, 4)
    got = patch_birefnet._image2patches(image, patch_ref=reference)
    assert torch.equal(got, ref_image2patches(image, grid_h, grid_w))


def test_image2patches_matches_einops() -> None:
    """Where einops exists, also compare with the real `rearrange`."""
    rearrange = pytest.importorskip("einops").rearrange
    image = torch.randn(1, 3, 16, 12)
    got = patch_birefnet._image2patches(image, grid_h=2, grid_w=3)
    expected = rearrange(image, patch_birefnet._SUPPORTED_TRANSFORMATION, hg=2, wg=3)
    assert torch.equal(got, expected)


def test_image2patches_rejects_other_transformation() -> None:
    with pytest.raises(NotImplementedError, match="transformation"):
        patch_birefnet._image2patches(torch.randn(1, 3, 4, 4), transformation="b c h w -> b c w h")


# ---- ⑦⑧ Module replacement -------------------------------------------------


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
    # Not bit-identical (ATen's inference kernel folds the multiply-add into an FMA) — see the
    # docstring.
    assert torch.allclose(expected, got, rtol=0, atol=BATCH_NORM_ATOL)


def test_channel_affine_rejects_training_mode() -> None:
    with pytest.raises(ValueError, match="training-mode"):
        patch_birefnet.ChannelAffine(nn.BatchNorm2d(4))


@pytest.mark.parametrize(("channels", "height", "width"), [(4, 5, 3), (8, 32, 32)])
def test_spatial_mean_matches_adaptive_avg_pool(channels: int, height: int, width: int) -> None:
    torch.manual_seed(4)
    x = torch.randn(1, channels, height, width)
    with torch.no_grad():
        expected = nn.AdaptiveAvgPool2d((1, 1))(x)
        got = patch_birefnet.SpatialMean()(x)
    assert got.shape == expected.shape
    # Not bit-identical (reducing both axes at once vs one axis at a time) — see the docstring.
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
    """Lowering a buffer to a plain attribute (a lifted constant) changes not a single element."""
    attention = RefWindowAttention(12, 4, 3)
    before = copy.deepcopy(attention.relative_position_index)
    assert "relative_position_index" in dict(attention.named_buffers())
    assert patch_birefnet._lift_relative_position_index(attention) == 1
    assert "relative_position_index" not in dict(attention.named_buffers())
    assert torch.equal(attention.relative_position_index, before)
    assert patch_birefnet._lift_relative_position_index(attention) == 0


# ---- ⑨ Decoder tail (1×1 conv and upsample in swapped order) ---------------


def _tail_source(in_channels: int) -> nn.Sequential:
    """The slot of `Decoder.conv_out1` (`nn.Sequential(nn.Conv2d(C, 1, 1, 1, 0))`).

    The weights are drawn with `normal_()` — with the default initialization the absolute values
    are so small that a wrong channel split (the fault injection below) could drown in rounding
    differences.
    """
    conv = nn.Conv2d(in_channels, 1, 1, 1, 0)
    conv.weight.data.normal_()
    conv.bias.data.normal_()
    return nn.Sequential(conv).eval()


class TestTailConv:
    """The algebraic rewrite of the tail is equivalent to the original tail / fails outside its
    assumptions."""

    @pytest.mark.parametrize(
        ("head_channels", "ipt_channels", "small", "size"),
        [
            (192, 48, (16, 16), (64, 64)),  # the real split (channels[3]//2 and channels[3]//8)
            (12, 5, (5, 3), (20, 12)),  # non-square, non-integer scale factor
            (8, 8, (7, 9), (7, 9)),  # no upsample (size equals head's)
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
        # Not bit-identical (the reduction is finished before the upsample) — see the docstring.
        assert torch.allclose(expected, got, rtol=0, atol=TAIL_ATOL)

    def test_a_swapped_channel_split_does_not_match(self) -> None:
        """Fault injection: swapping the head / ipt weights departs from the reference (the
        tolerance is not vacuous)."""
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
        with pytest.raises(NotImplementedError, match="do not fit"):
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
        """Anything but 1×1, stride 1, pad 0, groups 1 with a bias does not commute with the
        upsample."""
        with pytest.raises(NotImplementedError, match=match):
            patch_birefnet.TailConv(source, 4)


def _decoder_shape_stub(
    head_channels: int = 192, ipt_channels: int = 48, tail_in_channels: int = 240
) -> SimpleNamespace:
    """A decoder stand-in holding only the three numbers `_tail_head_channels` touches."""
    return SimpleNamespace(
        decoder_block1=SimpleNamespace(conv_out=SimpleNamespace(out_channels=head_channels)),
        ipt_blk1=SimpleNamespace(conv_out=SimpleNamespace(out_channels=ipt_channels)),
        conv_out1=[SimpleNamespace(in_channels=tail_in_channels)],
    )


class TestTailHeadChannels:
    """The head-side channel count can be derived two ways, so they are compared before use."""

    def test_the_two_derivations_agree_on_the_reference_shape(self) -> None:
        """The normal case (without it the negative cases below would hold trivially)."""
        assert patch_birefnet._tail_head_channels(_decoder_shape_stub()) == 192

    def test_a_disagreement_fails_loudly(self) -> None:
        """Fault injection: shifting the ipt branch's output by one channel makes the two
        derivations disagree, and it fails."""
        with pytest.raises(NotImplementedError, match="disagree"):
            patch_birefnet._tail_head_channels(_decoder_shape_stub(ipt_channels=47))


class TestDecoderForward:
    """The replaced `Decoder.forward` has only the eval path."""

    def test_training_mode_is_rejected(self) -> None:
        """The training path (ms_supervision's append / gdt's GT) is not copied, so it fails."""
        decoder = nn.Module()
        assert decoder.training

        with pytest.raises(RuntimeError, match="training-mode"):
            patch_birefnet._decoder_forward(decoder, ())

    def test_eval_mode_gets_past_the_gate(self) -> None:
        """The negative case above is not trivial — in eval it passes the gate and gets as far
        as unpacking the features."""
        decoder = nn.Module().eval()

        with pytest.raises(ValueError, match="unpack"):
            patch_birefnet._decoder_forward(decoder, ())


# ---- Configuration check (fields whose mismatch raises no shape error) -----


def _supported_config(**overrides: object) -> SimpleNamespace:
    """A configuration `assert_supported` accepts (the upstream class comes from
    `trust_remote_code` and cannot be imported)."""
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
    """`assert_supported` touches only `config.<key>` and `model.bb.ape`."""
    return SimpleNamespace(config=_supported_config(**overrides), bb=SimpleNamespace(ape=ape))


class TestAssertSupported:
    """The check of the configuration the replacement assumes (only fields whose mismatch falls
    into another **numeric path**)."""

    def test_the_reference_configuration_passes(self) -> None:
        """The normal case (without it the negative cases below would hold trivially)."""
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
        """For each of the 6 fields a mismatch means other weights and another graph — the
        message names the field and the expected value."""
        with pytest.raises(ValueError, match=f"config.{key}"):
            patch_birefnet.assert_supported(_supported_model(**{key: value}))

    def test_the_expected_backbone_is_named_in_the_message(self) -> None:
        with pytest.raises(ValueError, match="swin_v1_l"):
            patch_birefnet.assert_supported(_supported_model(bb="swin_v1_b"))

    @pytest.mark.parametrize("key", ["dec_ipt", "dec_ipt_split"])
    def test_the_patch_split_input_path_is_required(self, key: str) -> None:
        """With a `dec_ipt` field False, the image2patches rewrite does not fit."""
        with pytest.raises(ValueError, match="image2patches"):
            patch_birefnet.assert_supported(_supported_model(**{key: False}))

    def test_a_batch_size_of_one_is_rejected(self) -> None:
        """With `batch_size <= 1` the upstream `BatchNorm2d` becomes an Identity, unlike what the
        weights assume."""
        with pytest.raises(ValueError, match="BatchNorm2d"):
            patch_birefnet.assert_supported(_supported_model(batch_size=1))

    def test_absolute_position_embeddings_are_rejected(self) -> None:
        """`ape=True` is the path that emits a bicubic interpolation of the position embedding
        (not in the vocabulary)."""
        with pytest.raises(ValueError, match="ape"):
            patch_birefnet.assert_supported(_supported_model(ape=True))


# ---- BasicLayer.forward (looks up the baked mask) --------------------------


class _NoopDownsample(nn.Module):
    """The slot of `BasicLayer.downsample` (the shape of the return value does not matter — the
    test looks at the trailing two numbers)."""

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return x


class _MinimalBasicLayer(nn.Module):
    """A minimal skeleton of `BasicLayer` (only the attributes `_basic_layer_forward` touches)."""

    def __init__(self, use_checkpoint: bool = False, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.window_size = 4
        self.shift_size = 2
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([RefSwinBlock(12, 3, 4, 0), RefSwinBlock(12, 3, 4, 2)])
        self.downsample = downsample


class TestBasicLayerForward:
    """The mask is looked up only from **the baked buffer** / the branches line up correctly."""

    HEIGHT = 10
    WIDTH = 6
    PADDED_HEIGHT = 12
    PADDED_WIDTH = 8

    @staticmethod
    def _input(height: int = HEIGHT, width: int = WIDTH) -> torch.Tensor:
        torch.manual_seed(5)
        return torch.randn(1, height * width, 12)

    def test_an_unbaked_mask_fails_loudly(self) -> None:
        """MUST: never let `register_buffer` run during export (a constant appearing silently)."""
        layer = _MinimalBasicLayer()

        with pytest.raises(RuntimeError, match=r"the 12×8 window mask is not baked"):
            patch_birefnet._basic_layer_forward(layer, self._input(), self.HEIGHT, self.WIDTH)

        with pytest.raises(RuntimeError, match="prepare"):
            patch_birefnet._basic_layer_forward(layer, self._input(), self.HEIGHT, self.WIDTH)

    def test_the_build_window_bakes_the_reference_mask(self, monkeypatch) -> None:
        """Only the one call with the build window open adds the table (a buffer) and the index
        (a plain attribute), and the rebuilt values are bit-identical to `_build_attn_mask`."""
        monkeypatch.setattr(patch_birefnet, "_MASK_BUILD_ENABLED", True)
        layer = _MinimalBasicLayer()

        with torch.no_grad():
            patch_birefnet._basic_layer_forward(layer, self._input(), self.HEIGHT, self.WIDTH)

        suffix = f"{self.PADDED_HEIGHT}x{self.PADDED_WIDTH}"
        buffers = dict(layer.named_buffers())
        table = buffers[f"karume_attn_mask_table_{suffix}"]
        # The index is not a buffer (an i64 buffer cannot be an initializer — it travels as a
        # lifted constant).
        assert f"karume_attn_mask_index_{suffix}" not in buffers
        index = getattr(layer, f"karume_attn_mask_index_{suffix}")
        expected = patch_birefnet._build_attn_mask(
            self.PADDED_HEIGHT,
            self.PADDED_WIDTH,
            layer.window_size,
            layer.shift_size,
            table.dtype,
            table.device,
        )
        assert _bits_equal(patch_birefnet._expand_attn_mask(table, index), expected)

    def test_checkpointing_is_rejected(self, monkeypatch) -> None:
        """`use_checkpoint=True` is a path the replacement does not have (not passed silently)."""
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
        """`(H+1)//2` / `(W+1)//2` — dropping the round-up on the odd side shifts the next
        stage's shape."""
        monkeypatch.setattr(patch_birefnet, "_MASK_BUILD_ENABLED", True)
        layer = _MinimalBasicLayer(downsample=_NoopDownsample())

        with torch.no_grad():
            got = patch_birefnet._basic_layer_forward(layer, self._input(9, 7), 9, 7)

        assert got[1:3] == (9, 7)
        assert got[4:] == (5, 4)


# ---- The prepare / apply gate -----------------------------------------------


class _ExplodingModel(nn.Module):
    """A model that fails inside `prepare` (a stand-in for observing the `finally` restore)."""

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("the upstream forward failed")


class TestPrepareGate:
    """`prepare` cannot be called before `apply` / the build window closes on the exception path
    too."""

    def test_prepare_before_apply_fails_loudly(self, monkeypatch) -> None:
        """The plain forward does not look up the mask buffers, so it fails before baking."""
        monkeypatch.setattr(patch_birefnet, "_APPLIED", False)

        with pytest.raises(RuntimeError, match=r"apply\(model\)"):
            patch_birefnet.prepare(_ExplodingModel(), torch.zeros(1))

    def test_the_build_window_closes_even_when_prepare_raises(self, monkeypatch) -> None:
        """MUST: a window left open would let masks be baked silently anywhere afterwards."""
        monkeypatch.setattr(patch_birefnet, "_APPLIED", True)
        monkeypatch.setattr(patch_birefnet, "_MASK_BUILD_ENABLED", False)

        with pytest.raises(RuntimeError, match="the upstream forward failed"):
            patch_birefnet.prepare(_ExplodingModel(), torch.zeros(1))

        layer = _MinimalBasicLayer()
        with pytest.raises(RuntimeError, match="is not baked"):
            patch_birefnet._basic_layer_forward(layer, TestBasicLayerForward._input(), 10, 6)
