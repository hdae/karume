"""The patch layer that turns the MOSS-Image DiT (`MOSSImageTransformer` of the pinned
`pipeline_moss_image.py`) into three IR graphs with one length symbol each (ADR 0138 decisions 2–4
and 6, stage 1).

## The graphs

- `dit` ({@link MossDit}, symbol S = N + T): inputs `latent_tokens [1,S,128]` (the patchified
  latents in the image rows, zeros in the text rows), `text_context [1,S,3072]` (the refined text in
  the text rows, zeros in the image rows), `is_image [1,S,1]` (bool), `rope_cos` / `rope_sin
  [1,1,S,128]` and `timesteps_proj [1,256]`; output `[1,S,128]`. Upstream concatenates
  `[video_embedder(image tokens), text]` (`cat` along two symbols has no IR form), so the graph
  selects `where(is_image, video_embedder(latent_tokens), text_context)` instead, row by row. The
  time MLP and both modulations run inside at M = 1; the head runs over all S rows and the host
  keeps the first N ({@link head_rows}).
- `dit_skip22` ({@link MossDit} with `skip_layers={22}`): the same graph without block 22 —
  upstream's default negative branch (`uncond_skip_layers=(22,)`). Its blocks keep upstream's
  indices as module names (`layers.23.…`), so every tensor key is the key of the same tensor in
  `dit` and the export can declare all of them shared (`moss.export_dit`).
- `dit_text` ({@link MossDitText}, symbol T): `text_embeds [1,T,4096]`, `rope_cos` / `rope_sin
  [1,1,T,128]` → `[1,T,3072]` — `text_embedder` (the projection and the two refiner blocks), which
  has no modulation and so runs once per branch and generation.

## What changes against upstream

Only the RoPE operand. Upstream builds the rotation tables inside every `_rotate` call from the
`[1,S,128]` table `[sin 64 | cos 64]` (`sin, cos = rope.chunk(2, -1)`, each repeated to 128 and
unsqueezed to `[1,S,1,128]`). Here the host passes the repeated tables as graph inputs
(`[1,1,S,128]` — the host assembles them from the asset `rope_base`, {@link rope_rows}), and {@link
install_rope_pair} replaces upstream's module-level `_rotate` by a version that takes a {@link
RopePair} and computes upstream's expression `x * cos + cat((-b, a), -1) * sin` on it. Called with
upstream's own table (a tensor), the replacement hands over to upstream's `_rotate` unchanged, so
upstream's forward keeps its values whether or not the patch is installed. Everything else — the
blocks, `_attention`, `_situ` (so `h / 25` stays a division — ADR 0138 decision 6), the RMSNorms,
the time MLP, the modulations and the head — is upstream's code, called as it is.

The `[1,1,S,128]` input is turned into `[1,S,1,128]` by a transpose of a length-1 axis (no data
movement), the operand shape upstream's `_rotate` multiplies with.

## Eager equivalence (the step's gate — `moss.export_dit`)

The comparison partner is upstream `MOSSImageTransformer.forward` itself. Two linears run on a
different row count here (`video_embedder` on S rows instead of N; `final_linear_video` on S rows
instead of N), and torch's CPU GEMM may round the last bit differently with M. {@link
padded_rows} swaps exactly those two modules of the upstream model, for the duration of one call,
for a wrapper that zero-pads the input to S rows, calls the same linear and returns upstream's rows
(ADR 0138 decision 11). Everything else in the comparison runs as upstream's code.

MUST: diffusers is imported inside the functions (`tests/test_optional_group_imports.py`); the
upstream classes come from the hash-checked pipeline module (`moss.sources.load_pipeline_module`).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import ModuleType
from typing import Any, NamedTuple

import torch
from torch import nn

from moss import pipeline_ref

#: The graph names (= the container graph names and the series part directories).
DIT = "dit"
DIT_SKIP22 = "dit_skip22"
DIT_TEXT = "dit_text"

#: The block the negative branch skips (zero-based) — upstream's `uncond_skip_layers` default, taken
#: from {@link moss.pipeline_ref.UNCOND_SKIP_LAYERS} (the one source; a pytest pins it to the pinned
#: upstream signature). MUST: one source — the eager comparison feeds this value to both sides, so a
#: wrong value would pass it. The unpacking fails loudly if the default names other than one block
#: (`dit_skip22` skips exactly one).
(SKIP_BLOCK,) = pipeline_ref.UNCOND_SKIP_LAYERS

#: The graph inputs in forward-argument order (the argument names become the IR input names).
DIT_INPUTS = ("latent_tokens", "text_context", "is_image", "rope_cos", "rope_sin", "timesteps_proj")
DIT_TEXT_INPUTS = ("text_embeds", "rope_cos", "rope_sin")

#: The RoPE axes in upstream's column order (`_RoPE.forward`: coordinates `(r, f, h, w)`).
ROPE_AXES = ("r", "f", "h", "w")


class RopePair(NamedTuple):
    """The rotation tables as graph inputs, in the operand shape of upstream's `_rotate`.

    `cos` / `sin` are `[1,S,1,head_dim]`: upstream's `cat((cos, cos), -1).unsqueeze(-2)` of the
    `[sin | cos]` table. Wrapping them in a type lets the patched `_rotate` tell them from
    upstream's own `[1,S,head_dim]` table ({@link install_rope_pair}).
    """

    cos: torch.Tensor
    sin: torch.Tensor


def upstream_module(model: nn.Module) -> ModuleType:
    """The loaded pipeline module that defines `model`'s class (`moss.sources.PIPELINE_MODULE_NAME`
    — one instance per process)."""
    return sys.modules[type(model).__module__]


#: The attribute that marks the patched `_rotate` (keeps {@link install_rope_pair} idempotent and
#: holds the upstream function it hands over to).
_UPSTREAM_ROTATE = "_karume_upstream_rotate"


def install_rope_pair(module: ModuleType) -> None:
    """Replace the module-level `_rotate` of the pipeline module by the {@link RopePair} form
    (idempotent).

    `_attention` looks `_rotate` up in the module's globals at call time, so both block classes
    (`_Block` and `_TextBlock`) reach the replacement without any other change. A tensor table —
    upstream's own forward — goes to upstream's `_rotate` unchanged.
    """
    current = module._rotate
    if hasattr(current, _UPSTREAM_ROTATE):
        return
    upstream = current

    # Third-party code notice. The `RopePair` branch below is adapted from `_rotate` in
    # `pipeline_moss_image.py` of OpenMOSS-Team/MOSS-Image-SFT (revision pinned in
    # `moss.sources`), licensed under the Apache License, Version 2.0: upstream's expression on
    # tables that the host built (upstream builds them from the `[sin | cos]` table here).
    def rotate(x: torch.Tensor, rope: Any) -> torch.Tensor:
        if not isinstance(rope, RopePair):
            return upstream(x, rope)
        a, b = x.chunk(2, -1)
        return x * rope.cos + torch.cat((-b, a), -1) * rope.sin

    setattr(rotate, _UPSTREAM_ROTATE, upstream)
    module._rotate = rotate


def upstream_rotate(module: ModuleType) -> Any:
    """Upstream's own `_rotate` of the pipeline module, whether or not the patch is installed."""
    current = module._rotate
    return getattr(current, _UPSTREAM_ROTATE, current)


def rope_pair(rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> RopePair:
    """`[1,1,S,D]` graph inputs → the `[1,S,1,D]` operands (a transpose of a length-1 axis)."""
    return RopePair(rope_cos.transpose(1, 2), rope_sin.transpose(1, 2))


def _assert_supported(model: nn.Module) -> None:
    """The wrappers rely on batch 1, the 4-axis RoPE and the module layout of the pinned code."""
    config = model.config
    head_dim = int(config["dim"]) // int(config["num_heads"])
    if model.rope.num_bands * 2 * len(ROPE_AXES) != head_dim:
        raise NotImplementedError(
            f"rope bands {model.rope.num_bands} × 2 × {len(ROPE_AXES)} axes != head_dim {head_dim}"
        )
    if len(model.text_embedder.blocks) != 2:
        raise NotImplementedError(f"{len(model.text_embedder.blocks)} refiner blocks (2 expected)")


# Third-party code notice. `MossDit.forward` below is adapted from `MOSSImageTransformer.forward`
# in `pipeline_moss_image.py` of OpenMOSS-Team/MOSS-Image-SFT (revision pinned in `moss.sources`),
# licensed under the Apache License, Version 2.0. The time MLP, the modulations, the block loop and
# the head are upstream's statements for a batch of one; the patchify, the sinusoid, the RoPE table,
# the text refiner and the concatenation moved out (the host and `dit_text`), the concatenation
# becoming a row selection, and the head runs over every row.
class MossDit(nn.Module):
    """The S-form graph `dit` (or `dit_skip22` with `skip_layers`) — module docstring.

    It holds references to upstream's modules only; the blocks are kept under their upstream
    indices (`layers.<j>`), so the tensor keys are upstream's state-dict keys.
    """

    def __init__(self, model: nn.Module, *, skip_layers: frozenset[int] = frozenset()) -> None:
        super().__init__()
        _assert_supported(model)
        depth = len(model.layers)
        if any(not 0 <= index < depth for index in skip_layers):
            raise ValueError(f"skip_layers {sorted(skip_layers)} outside [0, {depth})")
        install_rope_pair(upstream_module(model))
        self.skip_layers = frozenset(skip_layers)
        self.video_embedder = model.video_embedder
        self.time_embedding = model.time_embedding
        self.shared_block_modulation = model.shared_block_modulation
        self.layers = nn.ModuleDict(
            {
                str(index): layer
                for index, layer in enumerate(model.layers)
                if index not in skip_layers
            }
        )
        self.final_norm_video = model.final_norm_video
        self.final_linear_video = model.final_linear_video
        self.video_head_modulation = model.video_head_modulation

    def forward(
        self,
        latent_tokens: torch.Tensor,
        text_context: torch.Tensor,
        is_image: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        timesteps_proj: torch.Tensor,
    ) -> torch.Tensor:
        time = self.time_embedding(timesteps_proj.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(
            1, 6, -1
        )
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(
            1, 2, -1
        )
        video = self.video_embedder(latent_tokens.to(self.video_embedder.weight.dtype).contiguous())
        x = torch.where(is_image, video, text_context)
        rope = rope_pair(rope_cos, rope_sin)
        for layer in self.layers.values():
            x = layer(x, rope, mod)
        x = self.final_norm_video(x.to(self.final_norm_video.weight.dtype))
        x = x * (1 + head[0:1, 1, None]) + head[0:1, 0, None]
        return self.final_linear_video(x.to(self.final_linear_video.weight.dtype))


# Third-party code notice. `MossDitText.forward` below is the `text_embedder` call of
# `MOSSImageTransformer.forward` in `pipeline_moss_image.py` of OpenMOSS-Team/MOSS-Image-SFT
# (revision pinned in `moss.sources`), licensed under the Apache License, Version 2.0, with
# `_TextProjection.forward`'s statements inlined so the RoPE operand can be the host's tables.
class MossDitText(nn.Module):
    """The T-form graph `dit_text` — module docstring. The module is upstream's `text_embedder`
    (tensor keys `text_embedder.…`, upstream's state-dict keys)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        _assert_supported(model)
        install_rope_pair(upstream_module(model))
        self.text_embedder = model.text_embedder

    def forward(
        self, text_embeds: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor
    ) -> torch.Tensor:
        embedder = self.text_embedder
        rope = rope_pair(rope_cos, rope_sin)
        x = embedder.proj(text_embeds.to(embedder.proj.weight.dtype))
        for block in embedder.blocks:
            x = block(x, rope)
        return x


# ---------------------------------------------------------------------------
# The RoPE base tables (asset `rope_base`) and their assembly
# ---------------------------------------------------------------------------

#: The rows of the `f` table: integer positions 0 … {@link MAX_TEXT_TOKENS} (text tokens sit at
#: `f = i < T`, image tokens at `f = T`, and upstream accepts T ≤ 4,096 — `max_sequence_length`).
MAX_TEXT_TOKENS = 4096

#: The accepted longest image side: 4,096 latent cells = 65,536 px — an explicit acceptance limit
#: (ADR 0138 decision 9). MUST: the TS acceptance check (stage 2) rejects a longer side with a
#: loud error that names this limit; {@link rope_rows} refuses one here.
#:
#: Why a limit: the `h` / `w` tables of `rope_base` have one row per position an image token can
#: take. A token sits at `h − (H−1)/2` (half-integers when H is even), so the tables hold the
#: half-integer grid `±(MAX_SIDE − 1)/2`, 2·MAX_SIDE − 1 rows (row `2h + MAX_SIDE − H` is position
#: `h − (H−1)/2`), and their row count fixes the longest side. Upstream has no ceiling (only memory
#: bounds an image), but with the area cap of the acceptance check (the VAE binding — about 2.80M px
#: on the 3080 Ti) a side over 65,536 px means a short side under about 43 px, which no use needs.
MAX_SIDE = 4096


def rope_columns(num_bands: int, axis: str, kind: str) -> slice:
    """The columns of one axis in upstream's `[1,S,8·num_bands]` table (`[sin r f h w | cos r f h
    w]`, `num_bands` columns per axis — `_RoPE.forward`'s `cat((sin, cos), -2).flatten(-2)`)."""
    offset = (0 if kind == "sin" else len(ROPE_AXES)) + ROPE_AXES.index(axis)
    return slice(offset * num_bands, (offset + 1) * num_bands)


def rope_base_tables(
    rope: nn.Module, *, max_text: int = MAX_TEXT_TOKENS, max_side: int = MAX_SIDE
) -> dict[str, torch.Tensor]:
    """The per-axis sin / cos base tables, cut out of upstream `_RoPE.forward` outputs (ADR 0034
    decision 3 — the host only reorders them and never computes a trigonometric function).

    Keys `sin_<axis>` / `cos_<axis>` `[rows, num_bands]` f32:

    - `r`: 1 row (every token has `r = 0`).
    - `f`: `max_text + 1` rows, position = row. From one call `forward(1, 1, max_text)`: its text
      tokens are at `f = 0 … max_text − 1` and its image token at `f = max_text`.
    - `h` / `w`: `2·max_side − 1` rows over the half-integer grid ({@link MAX_SIDE}). From two calls
      per axis, the side `max_side` and `max_side − 1` (one parity each); the image rows of the call
      with side `H` land at rows `2h + max_side − H`.

    MUST: the values are upstream's outputs — no formula is transcribed. {@link
    assert_rope_base} checks the slices land on their axis (position 0 is cos 1 / sin 0; position 1
    moves).
    """
    bands = int(rope.num_bands)
    tables: dict[str, torch.Tensor] = {}

    def cut(table: torch.Tensor, axis: str) -> dict[str, torch.Tensor]:
        return {
            f"{kind}_{axis}": table[:, rope_columns(bands, axis, kind)].contiguous()
            for kind in ("sin", "cos")
        }

    with torch.no_grad():
        text_call = rope(1, 1, max_text, torch.device("cpu"))[0]
        tables.update(cut(text_call[:1], "r"))
        # Rows 1 … max_text are the text tokens 0 … max_text − 1; row 0 is the image token at f = T.
        tables.update(cut(torch.cat((text_call[1:], text_call[:1])), "f"))
        for axis in ("h", "w"):
            grid = torch.empty(2 * max_side - 1, 8 * bands)
            for side in (max_side, max_side - 1):
                height, width = (side, 1) if axis == "h" else (1, side)
                image = rope(height, width, 1, torch.device("cpu"))[0, :side]
                rows = 2 * torch.arange(side) + (max_side - side)
                grid[rows] = image
            tables.update(cut(grid, axis))
    assert_rope_base(tables, bands, max_text=max_text, max_side=max_side)
    return tables


def assert_rope_base(
    tables: Mapping[str, torch.Tensor], bands: int, *, max_text: int, max_side: int
) -> None:
    """The cut is on the right columns and rows (fails loudly otherwise).

    Shapes alone cannot tell the four axes apart (the same width). So: the position-0 row of each
    axis is cos 1 / sin 0, and for f / h / w the next position moves sin off 0 (a slice of another
    axis would stay at its own position 0). For h / w the middle row is position 0 and the next row
    is position 1/2.
    """
    expected_rows = {"r": 1, "f": max_text + 1, "h": 2 * max_side - 1, "w": 2 * max_side - 1}
    keys = {f"{kind}_{axis}" for axis in ROPE_AXES for kind in ("sin", "cos")}
    if set(tables) != keys:
        raise AssertionError(f"rope base keys {sorted(tables)} (expected {sorted(keys)})")
    for axis, rows in expected_rows.items():
        for kind in ("sin", "cos"):
            table = tables[f"{kind}_{axis}"]
            if table.dtype != torch.float32 or tuple(table.shape) != (rows, bands):
                raise AssertionError(
                    f"rope base {kind}_{axis}: {table.dtype} {tuple(table.shape)}"
                    f" (expected float32 {(rows, bands)})"
                )
        zero = {"r": 0, "f": 0, "h": max_side - 1, "w": max_side - 1}[axis]
        if not torch.equal(tables[f"cos_{axis}"][zero], torch.ones(bands)):
            raise AssertionError(f"rope base cos_{axis}: position 0 is not all ones (misaligned)")
        if not torch.equal(tables[f"sin_{axis}"][zero], torch.zeros(bands)):
            raise AssertionError(f"rope base sin_{axis}: position 0 is not all zeros (misaligned)")
        if axis != "r" and torch.equal(tables[f"sin_{axis}"][zero + 1], torch.zeros(bands)):
            raise AssertionError(f"rope base sin_{axis}: does not move with its position")


def rope_rows(
    base: Mapping[str, torch.Tensor], kind: str, height: int, width: int, text_length: int
) -> torch.Tensor:
    """One kind (`sin` / `cos`) of the `[S, 4·num_bands]` table for the joint sequence `[image
    (h-major), text]`, gathered from the base tables (the host function TS transcribes in step 2).

    Image token (h, w): `(r 0, f T, h − (H−1)/2, w − (W−1)/2)`; text token i: `(r 0, f i, h 0, w
    0)` — `_RoPE.forward`'s coordinates.
    """
    side_rows = int(base[f"{kind}_h"].shape[0])
    max_side = (side_rows + 1) // 2
    if not (1 <= height <= max_side and 1 <= width <= max_side):
        raise ValueError(f"{height}×{width} cells: a side outside 1 … {max_side}")
    if not 1 <= text_length < int(base[f"{kind}_f"].shape[0]):
        raise ValueError(f"T = {text_length} outside 1 … {int(base[f'{kind}_f'].shape[0]) - 1}")
    h = torch.arange(height).repeat_interleave(width)
    w = torch.arange(width).repeat(height)
    image = torch.cat(
        (
            base[f"{kind}_r"][0].expand(height * width, -1),
            base[f"{kind}_f"][text_length].expand(height * width, -1),
            base[f"{kind}_h"][2 * h + max_side - height],
            base[f"{kind}_w"][2 * w + max_side - width],
        ),
        -1,
    )
    centre = max_side - 1
    text = torch.cat(
        (
            base[f"{kind}_r"][0].expand(text_length, -1),
            base[f"{kind}_f"][:text_length],
            base[f"{kind}_h"][centre].expand(text_length, -1),
            base[f"{kind}_w"][centre].expand(text_length, -1),
        ),
        -1,
    )
    return torch.cat((image, text))


def rope_table(
    base: Mapping[str, torch.Tensor], height: int, width: int, text_length: int
) -> torch.Tensor:
    """Upstream's own table `[1, S, 8·num_bands]` (`[sin | cos]` — what `_RoPE.forward(height,
    width, text_length)` returns), assembled from the base tables (the check of the asset)."""
    return torch.cat(
        tuple(rope_rows(base, kind, height, width, text_length) for kind in ("sin", "cos")), -1
    )[None]


def split_rope(table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream's `[1, S, D]` table `[sin | cos]` → the graph inputs `rope_cos` / `rope_sin
    [1,1,S,D]` (each half repeated to D — the values upstream's `_rotate` multiplies with)."""
    sin, cos = table.chunk(2, -1)
    return (
        torch.cat((cos, cos), -1)[:, None].contiguous(),
        torch.cat((sin, sin), -1)[:, None].contiguous(),
    )


# ---------------------------------------------------------------------------
# Graph inputs and outputs (the host functions TS transcribes in step 2)
# ---------------------------------------------------------------------------


def patchified(latents: torch.Tensor) -> torch.Tensor:
    """`[1, C, H, W]` → `[1, N, C]` (upstream's `hidden_states.flatten(2).transpose(1, 2)`,
    h-major)."""
    return latents.flatten(2).transpose(1, 2).contiguous()


def dit_inputs(
    latents: torch.Tensor,
    text_context: torch.Tensor,
    rope: torch.Tensor,
    timesteps_proj: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """The six inputs of `dit` / `dit_skip22` in {@link DIT_INPUTS} order.

    `latents` `[1,128,H,W]`; `text_context` the `dit_text` output `[1,T,3072]`; `rope` upstream's
    `[1,S,128]` table for `(H, W, T)`; `timesteps_proj` `[1,256]` (the sinusoid upstream feeds its
    time MLP).
    """
    tokens = patchified(latents)
    image, text = int(tokens.shape[1]), int(text_context.shape[1])
    rope_cos, rope_sin = split_rope(rope)
    return (
        torch.cat((tokens, tokens.new_zeros(1, text, tokens.shape[2])), 1),
        torch.cat((text_context.new_zeros(1, image, text_context.shape[2]), text_context), 1),
        torch.cat(
            (torch.ones(1, image, 1, dtype=torch.bool), torch.zeros(1, text, 1, dtype=torch.bool)),
            1,
        ),
        rope_cos,
        rope_sin,
        timesteps_proj,
    )


def head_rows(output: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """`[1,S,C]` graph output → upstream's sample `[1,C,H,W]` (the first N rows, upstream's
    `x.transpose(1, 2).reshape(1, -1, height, width)`)."""
    return output[:, : height * width].transpose(1, 2).reshape(1, -1, height, width)


# ---------------------------------------------------------------------------
# The M-aligned upstream forward (eager equivalence)
# ---------------------------------------------------------------------------


class PaddedRows(nn.Module):
    """A linear called on `rows` rows: the input `[1, n, c]` is zero-padded to `[1, rows, c]`, the
    same linear runs, and upstream's n rows come back (module docstring "Eager equivalence")."""

    def __init__(self, linear: nn.Linear, rows: int) -> None:
        super().__init__()
        self.linear = linear
        self.rows = rows

    @property
    def weight(self) -> torch.Tensor:
        """Upstream reads `<module>.weight.dtype` before calling it."""
        return self.linear.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        count = int(x.shape[1])
        if count > self.rows:
            raise ValueError(f"{count} rows do not fit {self.rows}")
        padded = torch.cat((x, x.new_zeros(1, self.rows - count, x.shape[2])), 1)
        return self.linear(padded)[:, :count]


#: The upstream modules {@link padded_rows} swaps (the two linears whose M differs — ADR 0138,
#: Context).
PADDED_MODULES = ("video_embedder", "final_linear_video")


@contextmanager
def padded_rows(
    model: nn.Module, rows: int, names: tuple[str, ...] = PADDED_MODULES
) -> Iterator[None]:
    """For the duration of the block, upstream's `names` run on `rows` rows ({@link PaddedRows})."""
    originals = {name: getattr(model, name) for name in names}
    try:
        for name, linear in originals.items():
            setattr(model, name, PaddedRows(linear, rows))
        yield
    finally:
        for name, linear in originals.items():
            setattr(model, name, linear)
