"""Patch layer that takes BiRefNet_HR (the bundled `birefnet.py`) through torch.export → IR
(docs/ir-v2.md).

The target is the dynamic module HF `ZhengPeng7/BiRefNet_HR` loads with `trust_remote_code`
(`transformers_modules.…birefnet`). Only that module's class attributes and global functions are
replaced; nothing on the transformers / timm / torchvision side is touched.

The replacements are **all layer 0** (no new ops are added) and fall into 2 groups by nature:

- **Bit-identical (data reordering only — not a single operation added or removed)**
  ① The rank-6 view+permute of `window_partition` / `window_reverse` becomes an adjacent
     transpose at rank 4 (the IR's strided representation is rank ≤ 4 —
     `normalize._lower_reshape_permute` cannot find the endpoints at rank 5 or more and fails)
  ② The qkv of `WindowAttention` goes from a rank-5 permute to **3 slices of the last dimension**
     (the same reason). The relative-position-bias table lookup `table[index.view(-1)]`
     (`aten.index.Tensor` = unsupported) becomes the equivalent
     `nn.functional.embedding(index, table)`. The window-mask addition goes from a rank-5 view +
     broadcast to a rank-4 broadcast (**equivalent only for B = 1** — fails loudly otherwise)
  ③ The H/W `F.pad` of `SwinTransformerBlock` (the IR's pad is last-dimension only) becomes a
     `cat` with a single zero element widened by `expand` ({@link _pad_zero}), and `torch.roll` (no
     handler; its decomposition is `index_select` + `fmod`) becomes slice + cat with a static shift
  ④ The shifted-window mask of `BasicLayer` (in-place assignment into `zeros` = a sequence of
     `slice_scatter` / `select` / `copy`) becomes **2 small baked tensors** (a table of window
     kinds + per-window indices) assembled in the graph with `embedding`
     ({@link _compact_attn_mask}). The values are made verbatim as in the original
     ({@link _build_attn_mask}); only where it runs moves outside the export

③ and ④ take a form that bakes no huge constants for the sake of transferring the distribution
(investigation record `outputs/diag/birefnet-xet-2026-10-16/`). Previously ④'s mask was baked whole
as f32 `[nW, 144, 144]` (248 MiB in total at 2048², values only {0, −100}, 4 window kinds per
mask), and ③'s zero pad was also baked by the exporter's constant folding into all-zero
initializers (~29 MB in total at 2048²). The 2048² container became ~86% zero bytes, and since HF
xet's content-defined chunking cannot cut inside long runs of zeros, the terms referencing the same
128 KiB zero chunk split into hundreds (fragmented transfer). Both contents are redundant bytes, so
the amount baked is itself reduced, without changing a single bit of the values.
  ⑤ The `x[:, 0::2, 0::2, :]` of `PatchMerging` (a step-2 slice = the IR's slice is contiguous
     windows only) becomes reshape + contiguous slices
  ⑥ The einops `rearrange` (rank 6) of `image2patches` becomes an adjacent transpose at rank 4
- **Not bit-identical (equivalent formulas, only the rounding differs) — measurements below**
  ⑦ `nn.BatchNorm2d` (inference) becomes a per-channel `x·α + β`. α/β are computed with the same
     formula as ATen's inference path (`alpha = weight·rsqrt(var+eps)` /
     `beta = bias − mean·alpha`), but ATen's kernel folds the multiply-add into one instruction
     (FMA), so **the lowest bits differ** (measured max 2.4e-07 — `birefnet/tests/test_patch.py`).
     Rewriting the decomposition with FMA gives no way to match the torch side (torch's own eager
     `x*a+b` does not match)
  ⑧ `nn.AdaptiveAvgPool2d((1, 1))` becomes `sum(dim=3) → sum(dim=2) → div(H·W)`. The reduction
     order differs from torch's simultaneous 2-axis reduction, so the lowest bits differ (measured
     max 1.1e-08)
  ⑨ The tail of `Decoder`, "stretch to full resolution → `cat` → 1×1 conv", is rearranged into
     "1×1 conv → stretch → add the other side's 1×1 conv" ({@link TailConv}). Both the 1×1 conv
     and the bilinear upsample are linear, so they commute over the reals, and the 2 huge
     full-resolution intermediates (at 2048², `[1,192,2048,2048]` 3.2GB and `[1,240,2048,2048]`
     4.0GB) disappear. The changed reduction order makes the lowest bits differ, but the
     contribution is small enough to be buried in ⑦⑧'s rounding (measured end to end at 1024²,
     maxdiff 5.3e-05 up to ⑦⑧ → 5.8e-05 up to ⑨, `export.py --verify`)

The entry points are split to match the 3 groups ({@link apply_layout_patches} /
{@link apply_module_patches} / {@link apply_tail_patches}; {@link apply} applies them all) — if
they could only be applied together, the bit identity of ①–⑥ could not be measured on its own, nor
could the rounding contributions of ⑦⑧ and ⑨ be taken separately.

MUST: no patch has a fallback (the same discipline as `siglip2.patch` / `deberta.patch`). A model
that breaks a premise (B = 1 / even resolution / `dec_att='ASPPDeformable'` and so on) fails on the
spot rather than silently falling to another numeric path.

MUST: {@link prepare} **always** runs after {@link apply}. ④'s table and indices are
resolution-dependent constants that cannot be made inside export (`register_buffer` does not get
into the graph). `prepare` bakes the table and indices in one eager forward and returns **the
patched reference output** as it is.

NOTE: class attributes of the dynamic module are replaced process-wide, so "the reference before
the patches" can be taken only once per process ({@link patches_applied} is the check).
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

#: Whether the replacements were applied (the check that keeps the side taking the pre-patch
#: reference from becoming trivially true).
_APPLIED = False

#: True only while {@link prepare} runs. The only window in which building the window mask is
#: allowed; it keeps `register_buffer` from running during export (= constants that are not in the
#: graph silently appearing).
_MASK_BUILD_ENABLED = False


def patches_applied() -> bool:
    """Whether this process applied the class-attribute replacements."""
    return _APPLIED


def assert_supported(model: nn.Module) -> None:
    """Check the configuration the replacement version assumes.

    Only what **falls to another path without a shape error when it differs** is checked.
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
            raise ValueError(
                f"config.{key} is not {want!r} ({got!r}) — not covered by the replacement version"
            )
    if not config.dec_ipt or not config.dec_ipt_split:
        raise ValueError(
            "dec_ipt / dec_ipt_split is False — the image2patches rewrite does not fit"
        )
    if config.batch_size <= 1:
        raise ValueError(
            "with batch_size ≤ 1 BatchNorm2d becomes Identity — not what the weights assume"
        )
    if model.bb.ape:
        raise ValueError(
            "ape=True is a path the replacement version does not have (bicubic interpolation of the"
            " position embedding appears)"
        )


# ---- ① Window partition / reverse (rank ≤ 4) -------------------------------


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """`[B, H, W, C]` → `[B·nH·nW, ws, ws, C]` (bit-identical to the original rank-6 version).

    The original reorders `(b, hb, hi, wb, wi, c)` into `(b, hb, wb, hi, wi, c)`. Only **2 adjacent
    axes** (`hi` and `wb`) are swapped, so folding the leading `(b, hb)` and the trailing `(wi, c)`
    into one axis each lets it be written as a single rank-4 permute.
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
    """`[B·nH·nW, ws, ws, C]` → `[B, H, W, C]` (the inverse of {@link _window_partition})."""
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


# ---- ② Window attention -----------------------------------------------------


def _window_attention_forward(
    self: nn.Module, x: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Equivalent implementation of `WindowAttention.forward` (rank ≤ 4, table lookup via
    embedding).

    It differs from the original in only 3 points, each only in how values are reordered:

    1. Instead of `reshape(B_, N, 3, nH, hd).permute(2, 0, 3, 1, 4)`, qkv splits the last
       dimension into 3 equal parts (it is nested as `(3, nH, hd)`, so the first C elements are q).
    2. The relative position bias is `nn.functional.embedding(index, table)` rather than
       `table[index.view(-1)]` (`aten.index.Tensor`). Both are a plain row gather.
    3. The window-mask addition is `attn + mask[:, None]` rather than
       `attn.view(B_//nW, nW, …) + mask[None, :, None]`. It is **equivalent only for B_ == nW
       (= batch 1)**, so anything else fails explicitly.
    """
    batch_windows, tokens, channels = x.shape
    heads = self.num_heads
    head_dim = channels // heads
    window_tokens = self.window_size[0] * self.window_size[1]
    if tokens != window_tokens:
        raise ValueError(
            f"window element count {tokens} does not match window_size {self.window_size}"
        )

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
                f"window count {windows} differs from axis 0, {batch_windows} (batch > 1) —"
                " the replacement's broadcast is equivalent to the original only for B = 1"
            )
        attn = attn + mask.unsqueeze(1)
    attn = self.softmax(attn)
    attn = self.attn_drop(attn)

    out = (attn @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
    out = self.proj(out)
    return self.proj_drop(out)


# ---- ③ Swin block (pad → cat / roll → slice+cat) ----------------------------


def _pad_zero(x: torch.Tensor, dim: int, width: int) -> torch.Tensor:
    """Append `width` zeros at the end of axis `dim` of `x` (the equivalent of `F.pad`).

    The IR's `pad` handles only the last dimension, so padding along H/W is written as **a `cat`
    with zeros**.

    The zeros are "a single zero element widened by `expand`". Written as `torch.zeros(shape)`, the
    exporter's constant folding would bake an all-zero `[1, pad, W, C]` into an initializer as a
    subtree independent of the input (~29 MB in total at 2048² — long runs of zeros fragment the HF
    xet transfer; the investigation record in the module docstring). A shape-changing `expand` is a
    frontier of constant folding (`_classify_foldable` of `karume.convert` stops there), so only a
    one-element constant is baked and the widening happens at run time. The values are the same as
    `F.pad`'s zeros (+0.0).
    """
    if width == 0:
        return x
    shape = list(x.shape)
    shape[dim] = width
    zeros = x.new_zeros((1,) * x.dim()).expand(shape)
    return torch.cat([x, zeros], dim=dim)


def _roll(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """The equivalent of `torch.roll(x, shift, dim)` (slice + cat with a static shift).

    After the curated decomposition `torch.roll` falls to `arange` + `fmod` + `index_select`
    (`index_select` is outside the vocabulary). With a static shift it can be written just by
    swapping the order of concatenation.
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
    """Equivalent implementation of `SwinTransformerBlock.forward` (only pad and roll rewritten)."""
    batch, length, channels = x.shape
    height, width = self.H, self.W
    if length != height * width:
        raise ValueError(f"input length {length} does not match {height}×{width}")

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


# ---- ④ Shifted-window mask (table of window kinds + per-window indices) ----

#: Upper bound of the window kinds one mask can have. The mask comes from the numbers of the 9
#: regions made by cutting the image into 3 bands in both rows and columns (`slice(0, -ws)` /
#: `slice(-ws, -shift)` / `slice(-shift, None)`), and the window grid straddles only the last 2
#: bands — a window's contents can only be one of 4 kinds: "interior / right column / bottom row /
#: bottom-right corner".
MAX_MASK_WINDOW_KINDS = 4


def _build_attn_mask(
    padded_height: int,
    padded_width: int,
    window_size: int,
    shift_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Verbatim port of the mask generation of `BasicLayer.forward` (eager execution only).

    It differs from the original in only 2 points: ① `Hp` / `Wp` are taken as python ints rather
    than tensors, and ② `window_partition` is {@link _window_partition} (bit-identical). The values
    are made the same way.
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


def _expand_attn_mask(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Build the `[nH·nW, N, N]` mask from the table of window kinds `[K, N·N]` and the indices on
    the window grid `[nH, nW]`.

    It is a plain row gather and becomes a single `embedding` in the IR (the same shape as ②'s
    relative position bias — the indices a lifted constant, the table a buffer).

    The indices are held in grid shape and flattened row-major here for the same reason as ②:
    `reshape` is subject to constant folding, so the exporter bakes the flattened indices as an i32
    constant (`karume.convert._add_const` — ADR 0009 / 0010). Passing a lifted constant straight to
    `embedding` falls to the initializer path (semantic f32 only) and stops at i64 (measured). The
    window order is row-major, as in {@link _window_partition}.
    """
    tokens = math.isqrt(table.shape[1])
    if tokens * tokens != table.shape[1]:
        raise ValueError(
            f"table row length {table.shape[1]} is not the square of a window's element count"
        )
    return nn.functional.embedding(index.reshape(-1), table).view(index.numel(), tokens, tokens)


def _compact_attn_mask(
    mask: torch.Tensor, grid: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold a `[nW, N, N]` mask into (a table of window kinds `[K, N·N]`, i64 indices on the window
    grid `grid`).

    Why fold: baking the mask whole comes to 248 MiB in total at 2048², but its contents are only
    {0, −100} and there are at most {@link MAX_MASK_WINDOW_KINDS} window kinds — the rest repeats
    the same windows, and also forms long runs of zeros that fragment the HF xet transfer (the
    investigation record in the module docstring).

    The table's rows are in **order of first appearance** in the window order of the upstream mask
    (`window_partition`'s row-major). The table's dtype is the same as `mask`. The indices' dtype is
    i64 — like ②'s `relative_position_index` they are held as a plain attribute (a lifted constant),
    which the exporter boundary-normalizes to i32 and bakes (an i64 buffer cannot be an initializer
    — {@link _lift_relative_position_index}).

    MUST: fail when the kinds exceed the bound or the reassembled values do not match the original
    mask bit for bit (a mask outside the assumed 3-band split is not silently turned into other
    values).
    """
    windows = mask.shape[0]
    if grid[0] * grid[1] != windows:
        raise ValueError(f"window grid {grid} does not match the mask's window count {windows}")
    flat = mask.reshape(windows, -1)
    # Compare bit patterns (value `==` treats `0.0` and `-0.0` as the same).
    bits = flat.contiguous().view(torch.uint8)
    kinds: list[int] = []
    index = torch.empty(windows, dtype=torch.long)
    for window in range(windows):
        for kind, first in enumerate(kinds):
            if torch.equal(bits[first], bits[window]):
                index[window] = kind
                break
        else:
            index[window] = len(kinds)
            kinds.append(window)
    if len(kinds) > MAX_MASK_WINDOW_KINDS:
        raise NotImplementedError(
            f"the window mask has {len(kinds)} kinds (at most {MAX_MASK_WINDOW_KINDS}) —"
            " not a shifted-window mask from a 3-band split"
        )
    table = flat[kinds]
    index = index.reshape(grid)
    rebuilt = _expand_attn_mask(table, index)
    if not torch.equal(rebuilt.contiguous().view(torch.uint8), mask.contiguous().view(torch.uint8)):
        raise AssertionError(
            "the window mask reassembled from the table and indices is not bit-identical to the"
            " original"
        )
    return table, index


def _basic_layer_forward(
    self: nn.Module, x: torch.Tensor, height: int, width: int
) -> tuple[Any, ...]:
    """Equivalent implementation of `BasicLayer.forward` (the mask is assembled in the graph from
    the baked table and indices)."""
    window_size = self.window_size
    padded_height = -(-height // window_size) * window_size
    padded_width = -(-width // window_size) * window_size
    table_name = f"karume_attn_mask_table_{padded_height}x{padded_width}"
    index_name = f"karume_attn_mask_index_{padded_height}x{padded_width}"
    table = getattr(self, table_name, None)
    if table is None:
        if not _MASK_BUILD_ENABLED:
            raise RuntimeError(
                f"the {padded_height}×{padded_width} window mask is not baked —"
                " run birefnet.patch.prepare(model, x) before export"
            )
        table, index = _compact_attn_mask(
            _build_attn_mask(
                padded_height, padded_width, window_size, self.shift_size, x.dtype, x.device
            ),
            (padded_height // window_size, padded_width // window_size),
        )
        self.register_buffer(table_name, table)
        # The indices go on a plain attribute, not a buffer (a lifted constant — the reason is in
        # _compact_attn_mask).
        object.__setattr__(self, index_name, index)
    attn_mask = _expand_attn_mask(table, getattr(self, index_name))

    if self.use_checkpoint:
        raise NotImplementedError(
            "use_checkpoint=True is a path the replacement version does not have"
        )
    for block in self.blocks:
        block.H, block.W = height, width
        x = block(x, attn_mask)
    if self.downsample is not None:
        x_down = self.downsample(x, height, width)
        return x, height, width, x_down, (height + 1) // 2, (width + 1) // 2
    return x, height, width, x, height, width


# ---- ⑤ PatchMerging (removes the step-2 slices) -----------------------------


def _patch_merging_forward(
    self: nn.Module, x: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Equivalent implementation of `PatchMerging.forward` (`x[:, 0::2, 0::2, :]` to contiguous
    slices).

    Re-split into `(hb, hi, wb, wi)`, the original's 4 step-2 slices become just "cut axes `hi` and
    `wi` at 0 / 1 each". `hi` is the 2nd axis, `wi` the first / second half of the last dimension.
    """
    batch, length, channels = x.shape
    if length != height * width:
        raise ValueError(f"input length {length} does not match {height}×{width}")
    if height % 2 or width % 2:
        raise NotImplementedError(
            f"odd resolution {height}×{width} is a path the replacement version does not have (the"
            " original's pad branch)"
        )
    x = x.view(batch, height, width, channels)
    x = x.reshape(batch * (height // 2), 2, width // 2, 2 * channels)

    quadrants = []
    for row in range(2):
        picked = x[:, row : row + 1, :, :]
        for column in range(2):
            part = picked[:, :, :, column * channels : (column + 1) * channels]
            quadrants.append(part.reshape(batch, height // 2, width // 2, channels))
    # Reassemble into the original's order (even/even, odd/even, even/odd, odd/odd).
    even_even, even_odd, odd_even, odd_odd = quadrants
    x = torch.cat([even_even, odd_even, even_odd, odd_odd], -1)
    x = x.view(batch, -1, 4 * channels)

    return self.reduction(self.norm(x))


# ---- ⑥ image2patches (einops rank-6 rearrange) ------------------------------

#: The only transformation the replacement version has (the measured form in `Decoder.forward` /
#: `BiRefNet`).
_SUPPORTED_TRANSFORMATION = "b c (hg h) (wg w) -> b (c hg wg) h w"


def _image2patches(
    image: torch.Tensor,
    grid_h: int = 2,
    grid_w: int = 2,
    patch_ref: torch.Tensor | None = None,
    transformation: str = _SUPPORTED_TRANSFORMATION,
) -> torch.Tensor:
    """Equivalent implementation of `image2patches` (a single rank-4 adjacent transpose).

    `(b, c, hg, h, wg, w)` → `(b, c, hg, wg, h, w)` swaps 2 adjacent axes (`h` and `wg`), so folding
    the leading `(b, c, hg)` into one axis lets it be written at rank 4.
    """
    if transformation != _SUPPORTED_TRANSFORMATION:
        raise NotImplementedError(
            f"transformation {transformation!r} is a path the replacement version does not have"
        )
    if patch_ref is not None:
        grid_h = image.shape[-2] // patch_ref.shape[-2]
        grid_w = image.shape[-1] // patch_ref.shape[-1]
    batch, channels, height, width = image.shape
    patch_h, patch_w = height // grid_h, width // grid_w
    patches = image.reshape(batch * channels * grid_h, patch_h, grid_w, patch_w)
    patches = patches.permute(0, 2, 1, 3)
    return patches.reshape(batch, channels * grid_h * grid_w, patch_h, patch_w)


# ---- ⑦⑧ Module replacement (BatchNorm2d / AdaptiveAvgPool2d) --------------


class ChannelAffine(nn.Module):
    """The equivalent `x·α + β` of inference-time `nn.BatchNorm2d` (α / β are per-channel
    constants).

    The same formula as ATen's inference path (`alpha = weight·rsqrt(var+eps)` /
    `beta = bias − mean·alpha`), but ATen folds the multiply-add with FMA, so it is **not
    bit-identical** (measured max 2.4e-07).
    """

    def __init__(self, source: nn.BatchNorm2d) -> None:
        super().__init__()
        if source.training:
            raise ValueError("a training-mode BatchNorm2d is not replaced (its statistics move)")
        if source.running_mean is None or source.running_var is None:
            raise ValueError(
                "a BatchNorm2d with track_running_stats=False has no inference statistics"
            )
        weight = source.weight if source.affine else torch.ones_like(source.running_var)
        bias = source.bias if source.affine else torch.zeros_like(source.running_var)
        alpha = weight.detach() * torch.rsqrt(source.running_var.detach() + source.eps)
        beta = bias.detach() - source.running_mean.detach() * alpha
        self.register_buffer("alpha", alpha.reshape(1, -1, 1, 1).clone())
        self.register_buffer("beta", beta.reshape(1, -1, 1, 1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha + self.beta


class SpatialMean(nn.Module):
    """The equivalent of `nn.AdaptiveAvgPool2d((1, 1))` (`sum` one axis at a time + `div`).

    The IR's reduce handles one axis only and has no `keepdim`, so it is split into the 2 stages
    `sum(dim=3) → sum(dim=2)` and then reshaped back to `[B, C, 1, 1]`. The reduction order differs
    from torch's simultaneous 2-axis reduction, so the lowest bits differ (measured max 1.1e-08).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        total = x.sum(dim=3).sum(dim=2)
        return (total / (height * width)).reshape(batch, channels, 1, 1)


def _lift_relative_position_index(model: nn.Module) -> int:
    """Move `relative_position_index` from a buffer down to a **plain tensor attribute**.

    The IR's initializers are semantic f32 only, so an i64 buffer cannot go in as it is
    (`convert._materialize_initializer` fails loudly). As a plain attribute, torch.export treats it
    as a lifted tensor constant, and the exporter's constant-folding path boundary-normalizes it to
    i32 and bakes it (`convert._add_const` — ADR 0009 / 0010). Not a single element's value changes.
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
    """`BatchNorm2d` → {@link ChannelAffine} / `AdaptiveAvgPool2d` → {@link SpatialMean}."""
    counts = {"batch_norm": 0, "adaptive_avg_pool": 0}
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                setattr(parent, name, ChannelAffine(child))
                counts["batch_norm"] += 1
            elif isinstance(child, nn.AdaptiveAvgPool2d):
                if tuple(child.output_size) != (1, 1):
                    raise NotImplementedError(
                        f"an AdaptiveAvgPool2d with output_size={child.output_size} is not replaced"
                    )
                setattr(parent, name, SpatialMean())
                counts["adaptive_avg_pool"] += 1
    return counts


# ---- ⑨ Decoder tail (swapping the order of the 1×1 conv and bilinear upsample) -


def _resize(x: torch.Tensor, size: Any) -> torch.Tensor:
    """Another spelling of upstream's `F.interpolate(..., mode='bilinear', align_corners=True)`."""
    return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=True)


class TailConv(nn.Module):
    """The equivalent of `Decoder.conv_out1` (1×1 conv 240→1) rearranged into the order "reduce
    first, then stretch".

    Upstream's tail makes 2 full-resolution intermediates (`[1,192,S,S]`, the stretched output of
    `decoder_block1`, and `[1,240,S,S]`, with the ipt branch's 48 channels concatenated onto it). At
    S = 2048 they are 3.2GB / 4.0GB, both above NVIDIA's 2GiB binding limit.

    A 1×1 conv is a linear combination along channels and a bilinear upsample a linear combination
    in space; the axes they act on are disjoint, so they **commute over the reals**. Moreover the
    1×1 conv after `cat` splits into 2 partial sums, so splitting the weights into the head side
    (`[1,192,1,1]`) and the ipt side (`[1,48,1,1]`) lets it be written as

        p1_out = interpolate(conv1x1(head, W_head)) + conv1x1(ipt, W_ipt, b)

    The only intermediates left at full resolution are `[1,1,S,S]` (4MB) and the inside of the ipt
    branch.

    The formulas are equivalent but the reduction order changes (the multiply-add over 192 channels
    is folded before stretching), so it is **not bit-identical** — the same "only the rounding
    differs" group as ⑦⑧ (measured on its own max 2.0e-05 = relative 3.5e-07 against output
    magnitudes of ~55, `birefnet/tests/test_patch.py`).
    """

    def __init__(self, source: nn.Module, head_channels: int) -> None:
        super().__init__()
        conv = _tail_conv2d(source)
        if not 0 < head_channels < conv.in_channels:
            raise NotImplementedError(
                f"the head side's {head_channels} channels do not fit conv_out1's input"
                f" {conv.in_channels} (the weights cannot be split by channel)"
            )
        weight = conv.weight.detach()
        self.register_buffer("head_weight", weight[:, :head_channels].clone())
        self.register_buffer("ipt_weight", weight[:, head_channels:].clone())
        self.register_buffer("bias", conv.bias.detach().clone())

    def forward(self, head: torch.Tensor, size: Any, ipt: torch.Tensor) -> torch.Tensor:
        """Add the result of stretching `head` (low resolution) to `size` and the contribution of
        `ipt` (full resolution).

        The bias is added on one side only (adding it on both would apply it twice).
        """
        pooled = nn.functional.conv2d(head, self.head_weight)
        return _resize(pooled, size) + nn.functional.conv2d(ipt, self.ipt_weight, self.bias)


def _tail_conv2d(source: nn.Module) -> nn.Conv2d:
    """Take the 1×1 conv out of `conv_out1` (fails when a premise breaks — no fallback)."""
    if not isinstance(source, nn.Sequential) or len(source) != 1:
        raise NotImplementedError(f"conv_out1 is not an nn.Sequential of length 1 ({source!r})")
    conv = source[0]
    if not isinstance(conv, nn.Conv2d):
        raise NotImplementedError(f"conv_out1[0] is not an nn.Conv2d ({type(conv).__name__})")
    if (
        tuple(conv.kernel_size) != (1, 1)
        or tuple(conv.stride) != (1, 1)
        or tuple(conv.padding) != (0, 0)
        or tuple(conv.dilation) != (1, 1)
        or conv.groups != 1
    ):
        raise NotImplementedError(
            "conv_out1[0] is not a 1×1, stride 1, pad 0, groups 1 conv"
            f" (kernel={conv.kernel_size} stride={conv.stride} padding={conv.padding}"
            f" dilation={conv.dilation} groups={conv.groups}) — it would not commute with the"
            " upsample"
        )
    if conv.bias is None:
        raise NotImplementedError(
            "conv_out1[0] has no bias — not what the replacement version assumes"
        )
    return conv


def _tail_head_channels(decoder: nn.Module) -> int:
    """The number of `conv_out1`'s input channels that come from `decoder_block1` (the rest are the
    ipt branch).

    The same number can be derived 2 ways (`decoder_block1`'s output conv / `conv_out1`'s input
    minus the ipt branch's output), so both are taken and compared — if they disagree, the position
    to split the weights by channel is not the assumed one, so it fails instead of silently
    splitting another way.
    """
    from_block = decoder.decoder_block1.conv_out.out_channels
    from_split = decoder.conv_out1[0].in_channels - decoder.ipt_blk1.conv_out.out_channels
    if from_block != from_split:
        raise NotImplementedError(
            f"decoder_block1's output {from_block} and conv_out1's remainder {from_split} disagree"
            " — the tail's weights cannot be split by channel"
        )
    return from_block


def _decoder_forward(self: nn.Module, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Equivalent implementation of the eval path of `Decoder.forward` (only the tail replaced with
    {@link TailConv}).

    Upstream lines 2145–2224 (the 4 stages x4 → p4 → p3 → p2 → _p1) are verbatim, and the gdt
    attention of `config.out_ref` (`p4 = p4 * gdt_attn_4` and so on) also runs in eval, so it is
    copied as is. Only the tail, lines 2225–2231, is rewritten (the reasoning is in
    {@link TailConv}).

    The training path (the `ms_supervision` intermediate-prediction appends / the GT side of gdt) is
    not copied, so it fails on the spot when `self.training` is true.
    """
    if self.training:
        raise RuntimeError(
            "a training-mode Decoder.forward is a path the replacement version does not"
            " have (the ms_supervision intermediate predictions and gdt's GT are not copied)"
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
    # Below is the rewritten tail. Upstream stretches head to full resolution, assigns it to `_p1`
    # (line 2226) and passes that `[1, 192, S, S]` as the next `patch_ref` (line 2229).
    # `image2patches` reads only the spatial size from `patch_ref`, and after stretching the shape
    # is `(S, S)`, the same as `x` — the grid becomes 1×1, so passing `x` itself before stretching
    # gives the same `patches_batch` as upstream.
    head = self.decoder_block1(_p1)
    patches_batch = _image2patches(x, patch_ref=x) if self.split else x
    ipt = self.ipt_blk1(_resize(patches_batch, x.shape[2:]))
    p1_out = self.conv_out1(head, x.shape[2:], ipt)
    return [p1_out]


# ---- Applying and preparing -------------------------------------------------


def apply_layout_patches(model: nn.Module) -> dict[str, int]:
    """Apply the reorder-only replacements ①–⑥ (5 class attributes + 1 module global).

    This stage alone is **bit-identical** to the pre-patch eager, and the first point of
    `birefnet.export.py --verify` measures that claim. That is why it is exposed separately from
    ⑦⑧ (applied together, it would mix with rewrites that move the rounding and the bit-identity
    claim could not be taken).
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
    """Apply the replacements ⑦⑧ that move the rounding (an instance scan for `BatchNorm2d` /
    `AdaptiveAvgPool2d`)."""
    return _replace_modules(model)


def apply_tail_patches(model: nn.Module) -> dict[str, int]:
    """Apply ⑨, the algebraic rewrite of the decoder tail (replacing `conv_out1` +
    `Decoder.forward`).

    `dec_ipt` / `dec_ipt_split` / `mul_scl_ipt == 'cat'` are already guaranteed by
    {@link assert_supported}, so the only check added here is the shape of `conv_out1`
    ({@link _tail_conv2d}). It is a separate entry point from ⑦⑧ so that
    `birefnet.export.py --verify` can take the maxdiff per stage.
    """
    decoder = model.decoder
    decoder.conv_out1 = TailConv(decoder.conv_out1, _tail_head_channels(decoder))
    module = sys.modules[type(model).__module__]
    module.Decoder.forward = _decoder_forward
    return {"tail_conv": 1}


def apply(model: nn.Module) -> dict[str, int]:
    """Apply all replacements (the export path — ①–⑥ + ⑦⑧ + ⑨).

    The return value is the number of replaced modules.
    """
    return {
        **apply_layout_patches(model),
        **apply_module_patches(model),
        **apply_tail_patches(model),
    }


def prepare(model: nn.Module, sample: torch.Tensor) -> Any:
    """Bake the resolution-dependent constants (the shifted-window mask) and return the patched
    reference output.

    MUST: run it right before export, **with an input of the same shape as the export**. The mask is
    a separate buffer per `(padded_height, padded_width)`, and for another shape it is not baked.
    """
    global _MASK_BUILD_ENABLED
    if not _APPLIED:
        raise RuntimeError(
            "apply(model) comes first — the plain forward does not read the mask buffer"
        )
    _MASK_BUILD_ENABLED = True
    try:
        with torch.no_grad():
            return model(sample)
    finally:
        _MASK_BUILD_ENABLED = False
