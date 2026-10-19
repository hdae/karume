"""The contents of the golden fixtures — the tiny nn.Module classes and their weight
initialisation and input generation helpers.

The ledger ({@link karume.goldens.GOLDEN_SPECS}) and the generation driver are in
{@link karume.goldens}. The import goes one way only, goldens → golden_models.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.ops import deform_conv2d


def _uniform(generator: torch.Generator, *shape: int, low: float, high: float) -> torch.Tensor:
    return torch.rand(shape, generator=generator) * (high - low) + low


def _mask(*shape: int, zero_at: int) -> torch.Tensor:
    """An i64 0/1 mask with exactly one 0 (torch's default integer dtype = int64)."""
    mask = torch.ones(shape, dtype=torch.int64).reshape(-1)
    mask[zero_at] = 0
    return mask.reshape(shape)


def _band_mask(length: int, *, blocked_row: int) -> torch.Tensor:
    """An i64 mask `[1,1,T,T]` for attention (1 = keep / 0 = mask).

    A band mask that keeps only the neighbourhood of the diagonal, with the row `blocked_row` set
    **entirely to 0** (fully masked). In a fully masked row masked_fill makes every element
    -3.4e38, which gives a uniform distribution under safe-softmax and NaN under the naive form —
    a shape that exercises the interaction with "the identity is ±F32_MAX" in limitations.md
    (section "Non-finite values (NaN / ±Infinity) are not checked").
    """
    rows = torch.arange(length).reshape(length, 1)
    cols = torch.arange(length).reshape(1, length)
    mask = ((rows - cols).abs() <= 1).to(torch.int64)
    mask[blocked_row] = 0
    return mask.reshape(1, 1, length, length)


class UnaryChain(nn.Module):
    """neg / abs / sqrt / log / exp (the input generator guarantees the domains of log and sqrt)."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude = torch.abs(torch.neg(x))
        logged = torch.log(torch.sqrt(magnitude))
        return logged, torch.exp(logged)


class Activations(nn.Module):
    """tanh / sigmoid / relu / gelu / gelu_tanh / sin (the two gelu variants are kept from
    decomposing and carried as one node).

    `sin` is not an activation, but it has the same shape as a unary f32 elementwise op on a
    **run-time value**, so it rides along here (`sin` on a constant input is folded by
    FOLDABLE_OPS and leaves no node — exercising the emit path needs a value that comes from a
    placeholder).
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            torch.tanh(x),
            torch.sigmoid(x),
            torch.relu(x),
            nn.functional.gelu(x),
            nn.functional.gelu(x, approximate="tanh"),
            torch.sin(x),
        )


class BroadcastBinary(nn.Module):
    """Exercise add / sub / mul / div with torch's right-aligned broadcast (rank 2 × rank 1).

    `offset` is a plain attribute, neither a Parameter nor a buffer — it goes through the path where
    torch.export lifts it as a lifted tensor constant (becoming an initializer).
    """

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.offset = _uniform(generator, 1, 4, low=-0.5, high=0.5)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return x + y, x - self.offset, x * y, x / y


class Mlp(nn.Module):
    """A 2-layer MLP of rank-2 matmuls. Every weight is carried through initializers."""

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(8, 16, generator=generator) * 0.3)
        self.b1 = nn.Parameter(torch.randn(16, generator=generator) * 0.1)
        self.w2 = nn.Parameter(torch.randn(16, 4, generator=generator) * 0.3)
        self.b2 = nn.Parameter(torch.randn(4, generator=generator) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(x @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2


class RowReduce(nn.Module):
    """Row reduce over the last dimension (no keepdim) — sum / amax / amin."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return x.sum(dim=-1), x.amax(dim=-1), x.amin(dim=-1)


class MaskChain(nn.Module):
    """A shrunk version of the DeBERTa front's mask path (recon §3-8).

    mask outer product mul(i64) → cast to truth values → bitwise_not → cast to weights → f32
    weight multiplication. To exercise the outer product without unsqueeze, the column / row
    masks are taken as [T,1] / [1,T]. One bool output puts the non-f32 readback path into the
    golden too.
    """

    def forward(
        self, mask_col: torch.Tensor, mask_row: torch.Tensor, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair = mask_col * mask_row
        keep = pair.to(torch.bool)
        drop = torch.bitwise_not(keep)
        return scores * keep.to(torch.float32), drop


class IntCast(nn.Module):
    """The f32 → integer truncation convention, i32 sub / mul, and the readback of an i32 output.

    `x.to(torch.int64)` is torch's truncate (rounding toward 0). In the IR it is declared as i32
    (boundary normalisation — ADR 0009).
    """

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = x.to(torch.int64)
        scaled = (truncated - k) * k
        return scaled, scaled.to(torch.float32)


class LayoutChain(nn.Module):
    """The chain permute → clone → view → unsqueeze → squeeze (ADR 0011 / recon §2).

    The same sequence as the observed attention head reshaping (a contiguous clone → view right
    after permute [0,2,1,3]). `reshape(3, -1)` applies to a non-contiguous permute result, so
    torch inserts a clone — that clone is the identity in the IR (values are always contiguous).

    MUST: the permutation is **a 3-cycle** ([1,2,0]). The forms seen in practice ([0,2,1,3] /
    [0,2,1]) are all involutions whose inverse is themselves, so a stride table built from the
    inverse permutation by mistake cannot be caught even by a torch comparison (the involution
    forms themselves are covered by the per-op cases of packages/runtime/tests/gpu_ops_test.ts).

    The second output is the **alias chain** of the permute output itself (3 reshape steps), which
    exercises "readback is correct even when a graph output is an alias" from the golden side.
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        heads = x.permute(1, 2, 0)  # [T,3,4] -> [3,4,T]
        merged = heads.reshape(3, -1)  # -> [3,4T] (clone + view)
        wide = merged.unsqueeze(0)  # -> [1,3,4T]
        return heads, torch.squeeze(wide, dim=[0])


class ExpandMask(nn.Module):
    """expand of a bool mask and i32 indices (the same shape as the 5 observed cases of recon §2).

    A shrunk form of the conv path's bool mask `[1,T,1] → [1,T,C]` and the gather indices
    `[1,T,T] → [16,T,T]`. One expand output is read back as i32, taking a non-f32 strided copy
    all the way to readback.
    """

    def forward(
        self, mask: torch.Tensor, index: torch.Tensor, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keep = mask.to(torch.bool)  # bool[T,1]
        wide = keep.expand(-1, 3)  # bool[T,3]
        spread = index.expand(3, -1)  # i64[3,T] (i32 in the IR)
        return scores * spread.to(torch.float32), torch.bitwise_not(wide), spread


class BatchMatmul(nn.Module):
    """Rank-3 batched matmul (the same shape as the 8 observed cases of recon §2).

    MUST: **B / M / K / N all have different lengths** (a pitfall once listed in ACTIVE_DESIGN). A
    batch stride built from the product of the wrong neighbouring dimensions (confusing M·K, K·N
    and M·N) does not show in the numbers for shapes where two axes have the same length. Here
    B=2 / M=T=6 / K=3 / N=5, and the element counts of one matrix, 18 / 15 / 30, all differ too.

    The second output multiplies the same multiplicand by a transposed partner, exercising a read
    with K and N swapped (building b's stride from k instead of n) from another shape too.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.bmm(x, y), torch.bmm(y.permute(0, 2, 1), x.permute(0, 2, 1))


class GatherLastDim(nn.Module):
    """gather over the last dimension (observed: dim=-1 / src f32[16,T,512] / index
    i32[16,T,T]).

    index is passed as **an i32 graph input** (torch's default i64 is normalised to i32 at the
    boundary — ADR 0009). An i32 initializer can be baked (the gather indices of
    {@link SymbolicTable}), but this golden makes it a graph input to exercise indices that arrive
    at run time.

    The second output is a downstream operation that uses the looked-up values, putting into the
    golden that the gather result flows on correctly as values.
    """

    def forward(self, src: torch.Tensor, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        picked = torch.gather(src, -1, index)
        return picked, picked.sum(dim=-1)


def _fill_param(generator: torch.Generator, tensor: torch.Tensor, scale: float) -> None:
    """Overwrite a parameter from the fixed-seed generator.

    MUST: the default initialisation of `nn.Linear` / `nn.Conv1d` / `nn.Embedding` uses **the
    global RNG**. Without the overwrite the golden does not reproduce byte for byte, and the
    regeneration cross-check (TestDeterminism) fails.
    """
    with torch.no_grad():
        tensor.copy_(torch.randn(tensor.shape, generator=generator) * scale)


class ArgmaxPick(nn.Module):
    """The exit of greedy decoding (ADR 0068 decision 2) — argmax over the last dimension = the
    vocabulary axis, **rank preserved**.

    The first output is the real shape (argmax over the logits of a `linear` = the lm_head
    equivalent; the position axis is the symbol T). The second is argmax over **equal-value
    ties**: `cat([t, t], dim=-1)` always creates equal values at symmetric positions, so "the
    smallest index" is required **by construction**, not by chance — an implementation that returns
    the largest index answers `argmax(t) + n`, and the golden comparison turns red on the spot (the
    axis on which llama.cpp disagreed between GPU and CPU — survey §2).

    MUST: the tie input is built from **a graph input** (as a constant, FOLDABLE_OPS would fold it
    and the node would disappear).
    """

    HIDDEN = 5
    VOCAB = 7

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.head = nn.Linear(self.HIDDEN, self.VOCAB)
        _fill_param(generator, self.head.weight, 0.7)
        _fill_param(generator, self.head.bias, 0.3)

    def forward(self, x: torch.Tensor, tied: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.head(x)
        doubled = torch.cat([tied, tied], dim=-1)
        return (
            torch.argmax(logits, dim=-1, keepdim=True),
            torch.argmax(doubled, dim=-1, keepdim=True),
        )


class AttentionBlock(nn.Module):
    """linear → bmm → softmax → bmm → linear → residual + layer_norm (a shrunk form of the
    sequence of recon §2).

    The same order as the observed attention, including the head split
    `view → permute[0,2,1,3] → reshape`. It exercises the fused ops linear (always with bias) /
    softmax (dim=-1) / layer_norm ([H] + affine) in one graph.

    The second output is **softmax over large negative values** (where the naive form's `exp`
    collapses to 0 in f32 and gives 0/0 = NaN). The golden side deliberately plants a shape on
    which the golden comparison turns red the moment safe-softmax's amax subtraction is dropped
    (ADR 0012's counterexamples).
    """

    HIDDEN = 8
    HEADS = 2
    HEAD_DIM = 4

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.query = nn.Linear(self.HIDDEN, self.HIDDEN)
        self.key = nn.Linear(self.HIDDEN, self.HIDDEN)
        self.value = nn.Linear(self.HIDDEN, self.HIDDEN)
        self.dense = nn.Linear(self.HIDDEN, self.HIDDEN)
        # Use the observed layer_norm_eps (1e-07) as is
        self.norm = nn.LayerNorm(self.HIDDEN, eps=1e-07)
        for projection in (self.query, self.key, self.value, self.dense):
            _fill_param(generator, projection.weight, 0.3)
            _fill_param(generator, projection.bias, 0.1)
        _fill_param(generator, self.norm.weight, 0.2)
        _fill_param(generator, self.norm.bias, 0.1)

    def _split_heads(self, x: torch.Tensor, length: Any) -> torch.Tensor:
        return (
            x.view(1, length, self.HEADS, self.HEAD_DIM)
            .permute(0, 2, 1, 3)
            .reshape(self.HEADS, length, self.HEAD_DIM)
        )

    def forward(self, x: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = x.shape[1]
        query = self._split_heads(self.query(x), length)
        key = self._split_heads(self.key(x), length)
        value = self._split_heads(self.value(x), length)
        scores = torch.bmm(query, key.permute(0, 2, 1))
        context = torch.bmm(torch.softmax(scores, dim=-1), value)
        merged = (
            context.view(1, self.HEADS, length, self.HEAD_DIM)
            .permute(0, 2, 1, 3)
            .reshape(1, length, self.HIDDEN)
        )
        return self.norm(self.dense(merged) + x), torch.softmax(logits, dim=-1)


class FusedAttention(nn.Module):
    """The golden that preserves SDPA and makes it one fused `attention` node (ADR 0023).

    MUST: **B / H / M / N / D all have different lengths**. The kernel folds B and H into one batch
    axis, so with B=1 (the observed shape) or B==H a mix-up does not show in the values (detection
    limit ① of the design recon §4.6).
    NOTE: the shape is **static**. The attention of DiT / VAE has a fixed resolution and no
    symbolic dimensions, so the golden does not anticipate shapes not seen in practice (symbolic
    M / N).
    """

    BATCH = 2
    HEADS = 3
    QUERIES = 5
    KEYS = 7
    DEPTH = 4

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)


def _fused_attention_inputs(generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    """Inputs of `FusedAttention`. **Only the last query row has its logits dropped uniformly to
    −195**.

    It makes the golden always exercise the region where, if the amax subtraction of the row
    statistics (②) is dropped, `exp(−195)` collapses to 0 in f32 and the denominator 0 gives
    0/0 = NaN (the same device `masked_scores` / `attention_block` use for softmax alone, placed
    inside the fused path).

    MUST: that row's logits are **the same bits for every n** (`q = [−c,0,0,0]` × `k[…,0]` is a
    constant independent of n). When they are identical, `S − amax` is exactly 0 and the weights
    land exactly on `1/N`. **Uneven logits push the golden past the tolerance** — a dot product
    with |S| ≈ 190 carries an absolute error of ≈1e-5 in f32, which `exp(S − amax)` turns
    directly into a **relative** weight error of 1e-5 (observed: with logits spread by 12.7 within
    the row, maxAbs 4.7e-6 / maxRel 9.6e-4 — beyond the atol 1e-6 of GOLDEN_TOLERANCE in
    packages/runtime/tests/e2e_golden_test.ts). This is not an error of the fusion but **the
    condition number of the input**, and the decomposed path gives the same values (bit identity
    of fused and decomposed is pinned separately by
    packages/runtime/tests/gpu_attention_parity_test.ts).
    """
    shape = FusedAttention
    query = _uniform(
        generator, shape.BATCH, shape.HEADS, shape.QUERIES, shape.DEPTH, low=-1.0, high=1.0
    )
    key = _uniform(generator, shape.BATCH, shape.HEADS, shape.KEYS, shape.DEPTH, low=-1.0, high=1.0)
    value = _uniform(
        generator, shape.BATCH, shape.HEADS, shape.KEYS, shape.DEPTH, low=-1.0, high=1.0
    )
    # Only channel 0 is a constant independent of n. The square of the half scale √(1/√D) =
    # 1/√D = 0.5 multiplies the dot product, so the row q = [−60,0,0,0] has
    # S = 0.5 · (−60 · 6.5) = −195 (the same value for every n).
    key[:, :, :, 0] = 6.5
    query[:, :, -1, :] = 0.0
    query[:, :, -1, 0] = -60.0
    return (query, key, value)


class EmbeddingLookup(nn.Module):
    """embedding (row gather) — weight f32[V,H] × indices i32[…].

    MUST: the `padding_idx` row is **overwritten with non-zero values before** export.
    `nn.Embedding` initialises the padding_idx row to 0, so left as is, a wrong implementation
    that "zeroes the padding row" would agree with torch's output and the golden would be
    tautological (ADR 0012's "accepted and inert" could not actually be verified).
    """

    VOCAB = 5
    HIDDEN = 3

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.table = nn.Embedding(self.VOCAB, self.HIDDEN, padding_idx=0)
        _fill_param(generator, self.table.weight, 1.0)

    def forward(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self.table(index)
        return rows, rows.sum(dim=-1)


class MaskedScores(nn.Module):
    """The 2 observed forms of masked_fill (recon §2) — attention's -3.4e38 fill and the conv
    path's 0 fill.

    - `mask bool[1,1,T,T]` is applied to `scores f32[1,2,T,T]` with **right-aligned broadcast**.
    - The fill value is the smallest finite f32 (if a JSON round trip moved it by an ulp, this
      would turn red).
    - The downstream softmax gives a uniform distribution on a row whose elements are all masked
      (exercising, in the padded case, the interaction with "the identity of amax/amin is
      ±F32_MAX" in limitations.md).
    """

    NEG_INF = -3.4028234663852886e38

    def forward(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        hidden: torch.Tensor,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        drop = torch.bitwise_not(mask.to(torch.bool))
        weights = torch.softmax(scores.masked_fill(drop, self.NEG_INF), dim=-1)
        return weights, hidden.masked_fill(gate.to(torch.bool), 0.0)


class RuntimeMaskedAttention(nn.Module):
    """SDPA with a run-time bool mask (ADR 0044) — the form in which the guard is baked as
    `safe_softmax`.

    It exercises ADR 0044's whole chain: **a bool graph input** (ADR 0009 — IO as u32 0/1) →
    `where(mask, 0, -inf)` emitted by torch's SDPA decomposition → `add` onto the scores →
    `softmax` + the safe-softmax guard → the value proof of `_drop_safe_softmax_guard` does not
    hold for a run-time input, so a constructive replacement with one `safe_softmax` node.

    MUST: the mask contains **an all-False row** (`_runtime_masked_attention_inputs`). After the
    addition every element of that row is -inf and torch returns 0 — if safe_softmax could not
    write an empty row as 0, it would give 0/0 = NaN and the comparison would turn red. This is
    why the golden exists.
    NOTE: SDPA is not preserved (the default `PRESERVED_OP_PREFIXES` as is). The mask contract of
    fused attention is only a static additive f32 `[1,1,M,N]`; a run-time mask runs on the
    decomposed path (ADR 0044 decision 1).
    """

    HEADS = 2
    QUERIES = 4
    DEPTH = 3

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)


def _runtime_masked_attention_inputs(generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    """Inputs of `RuntimeMaskedAttention` (q / k / v and **a bool mask**).

    The mask is `[1,1,M,M]` with True = keep. The kept side is a checkerboard so that the rows are
    not only "all True", and row 2 is **all False** (the row on which torch's guard fires).
    """
    shape = RuntimeMaskedAttention
    length = shape.QUERIES
    qkv = tuple(
        _uniform(generator, 1, shape.HEADS, length, shape.DEPTH, low=-1.0, high=1.0)
        for _ in range(3)
    )
    rows = torch.arange(length).reshape(length, 1)
    cols = torch.arange(length).reshape(1, length)
    mask = ((rows + cols) % 3) != 2
    mask[2] = False
    return (*qkv, mask.reshape(1, 1, length, length))


class ConvBlock(nn.Module):
    """The observed form of conv1d (kernel 3 / stride 1 / padding 1 / groups 1 / dilation 1 / with
    bias).

    As observed (recon §2), it includes the round trip `[B,T,C] → permute → conv1d → permute`. A
    bug that treats padding as 0 shows not in the output length but in **the values at both ends**
    (the output length is decided up front from attrs).
    """

    CHANNELS = 4

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.conv = nn.Conv1d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1)
        _fill_param(generator, self.conv.weight, 0.3)
        _fill_param(generator, self.conv.bias, 0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moved = self.conv(x.permute(0, 2, 1))
        return moved, moved.permute(0, 2, 1)


class SymbolicTable(nn.Module):
    """Symbolic-length table (ADR 0010) — the T-dependent subtree is baked at Tmax and carried
    through sym_prefix_slice.

    - Outer product of the relative positions `arange(T)` → clamp → **gather indices (an i32
      initializer)**. Both axes are symbolic, so there are 2 slices.
    - f32 scaling of `arange(T)` (an f32 sym_prefix_slice, 1 slice).

    The baked constants span the full Tmax (= the upper bound of `_SMALL_DYNAMIC_T` in
    goldens.py), and the golden's actual length T is shorter. So **a bug that builds the read
    stride from the bound shape shows in the values here** (it would agree only at T = Tmax, so a
    golden with a short actual length is the detector).
    """

    BUCKETS = 5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = x.shape[0]
        positions = torch.arange(length)
        buckets = torch.clamp(
            positions.unsqueeze(1) - positions.unsqueeze(0) + 2, 0, self.BUCKETS - 1
        )
        scale = torch.arange(length).to(torch.float32) * 0.5
        return torch.gather(x, -1, buckets), x + scale.unsqueeze(-1)


class ScalarOperands(nn.Module):
    """Constant promotion of scalar operands (recon §4-4) — 4 f32 forms and an i32 form with a
    reversed scalar.

    `1 - mask` is the non-commutative reversed form (as in the observed conv path). A bug that
    puts the constant on the right gives `mask - 1`, flips the sign, and **changes the values** in
    the downstream weight multiplication.

    MUST: the result of `1 - mask` **flows downstream as a value** (multiplied into x as a
    weight). With mask ∈ {0,1}, `(1 - mask) * mask` is identically 0 even reversed, a tautological
    golden on which the bug does not show.
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scaled = (x + 2.5 - 0.5) * 3.0 / 2.0
        return scaled, (1 - mask).to(torch.float32) * x


class SplinePieces(nn.Module):
    """A shrunk version of the skeleton of sdp's rational-quadratic spline (recon §2).

    From the in-interval test `(x >= -b) & (x <= b)` (ge_scalar / le_scalar / bitwise_and) →
    `cumsum` of the interval boundaries → searchsorted_free's `sum(x[…,None] >= bl, dim=-1)` (the
    Tensor form of ge and **a bool row sum → i32**) → the `where` that restores out-of-interval
    values and the `log1p` of the decomposed softplus, it exercises the math ops added in phase 3
    along one continuous path.

    MUST: the inputs straddle the interval boundaries (±TAIL) and both ends of the clamp. If every
    element fell inside (or outside) the interval, only one side of the where would be exercised
    and a mix-up would not show in the values.
    """

    TAIL = 1.0
    CLAMP_MIN = -0.75
    CLAMP_MAX = 0.5

    def forward(self, inputs: torch.Tensor, widths: torch.Tensor) -> tuple[torch.Tensor, ...]:
        inside = (inputs >= -self.TAIL) & (inputs <= self.TAIL)
        cumwidths = torch.cumsum(widths, dim=-1)
        # searchsorted_free: count how many boundaries each element is at or above (a bool row
        # sum = an i32 count)
        bins = torch.sum(inputs.unsqueeze(-1) >= cumwidths, dim=-1)
        clamped = torch.clamp(inputs, self.CLAMP_MIN, self.CLAMP_MAX)
        softplus = torch.log1p(torch.exp(clamped))
        return (
            torch.where(inside, softplus, inputs),
            bins,
            inside,
            clamped > 0.25,
            cumwidths,
        )


class DecoderTail(nn.Module):
    """The final stage of dec (recon §2) — **2 leaky_relu slopes** and an f32 expand.

    ups / ResBlock use `LRELU_SLOPE=0.1`, the final stage torch's default 0.01 (the positional
    argument omitted altogether). The golden pins the form in which one of them silently goes
    wrong unless attrs carry the slope (ADR 0015). The expand of `gain` is **an f32 expand** of the
    same shape as turning the relative position embedding 4-D (enc_p / flow).
    """

    CHANNELS = 4
    OUT_CHANNELS = 3
    SLOPE = 0.1

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.conv = nn.Conv1d(self.CHANNELS, self.OUT_CHANNELS, kernel_size=3, padding=1)
        _fill_param(generator, self.conv.weight, 0.3)
        _fill_param(generator, self.conv.bias, 0.1)

    def forward(self, x: torch.Tensor, gain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wide = gain.expand(-1, -1, x.shape[2])
        hidden = nn.functional.leaky_relu(x * wide, self.SLOPE)
        # The form with the positional argument omitted (torch default 0.01) — the exporter makes
        # it explicit in attrs at the boundary
        return hidden, torch.tanh(nn.functional.leaky_relu(self.conv(hidden)))


class CouplingSplit(nn.Module):
    """A shrunk version of the coupling reverse's split → transform → cat → flip path (recon §2).

    ① `torch.split(x, [half]*2, 1)` — opened into 2 slices by normalize's split→slice pass
    ② Transform only one side and join back with `torch.cat([x0, x1], 1)` (a form where **the
       input order shows in the values**)
    ③ Finally `Flip` (reversal of the channel axis)

    MUST: the 2 inputs of cat have **different value ranges** (one transformed, one plain). With
    the same range, a mix-up of the write positions does not show in the values.
    MUST: the channel count is 6 (= 3 + 3), so the flip axis has length 3 or more. Reversing 2
    channels cancels an off-by-one symmetrically, so it is no detector (sdp's observed 2 channels
    are covered separately by the per-op cases of packages/runtime/tests/gpu_ops_test.ts).

    The second output is **pad + slice of the last dimension** (a shrunk form of the value side of
    relative-position attention, `F.pad(p_attn, [w,w])`, and the spline's
    `bin_locations[..., :-1]`). A pad fill value other than 0 and an off-by-one slice start show
    directly in the values.
    """

    WINDOW = 2

    def forward(self, x: torch.Tensor, attn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first, second = torch.split(x, [3, 3], 1)
        transformed = torch.tanh(first) * 2.0
        joined = torch.cat([transformed, second], 1)
        flipped = torch.flip(joined, [1])
        padded = nn.functional.pad(attn, [self.WINDOW, self.WINDOW])
        return flipped, padded[..., 1:] * 3.0


class DilatedConvStack(nn.Module):
    """conv1d groups / dilation (ADR 0015) — a shrunk form of sdp's DDSConv and dec's ResBlock1.

    ① **depthwise dilated** (`groups = C`, k 5, dilation 1/3/9, padding = d·(k−1)/2) in 3 steps.
       The observed form is exactly sdp's DDSConv (`dilation = kernel_size**i`). A bug that
       ignores groups and runs a plain convolution always shows in the values, as **values of
       other channels mixing in**.
    ② **pointwise 1x1** (asymmetric Cin 6 → Cout 4) exercises DDSConv's `conv_1x1`. With k=1,
       `d·(K−1) = 0`, so it is also the boundary where the general dilation form agrees with the
       old formula.
    ③ **intermediate groups** (1 < g < C: Cin 4 / Cout 6 / g 2) — the form where the weight's
       second axis is Cin/groups = 2. A bug that drops the leading offset of the group band
       vanishes under depthwise, so this step is the detector.
    ④ **ResBlock1's residual** (k 3, dilation 3 and 5, padding = d) — `x + conv(leaky_relu(x))`.

    MUST: the steps do not share a channel count. If B/Cin/Cout/K had the same length, an index
    mix-up would not show in the values.
    """

    DEPTH_CHANNELS = 6
    DEPTH_KERNEL = 5
    DEPTH_DILATIONS = (1, 3, 9)
    MID_CHANNELS = 4
    GROUPED_CHANNELS = 6
    GROUPS = 2
    RES_KERNEL = 3
    RES_DILATIONS = (3, 5)
    SLOPE = 0.1

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.depthwise = nn.ModuleList(
            nn.Conv1d(
                self.DEPTH_CHANNELS,
                self.DEPTH_CHANNELS,
                kernel_size=self.DEPTH_KERNEL,
                padding=dilation * (self.DEPTH_KERNEL - 1) // 2,
                dilation=dilation,
                groups=self.DEPTH_CHANNELS,
            )
            for dilation in self.DEPTH_DILATIONS
        )
        self.pointwise = nn.Conv1d(self.DEPTH_CHANNELS, self.MID_CHANNELS, kernel_size=1)
        self.grouped = nn.Conv1d(
            self.MID_CHANNELS,
            self.GROUPED_CHANNELS,
            kernel_size=3,
            padding=1,
            groups=self.GROUPS,
        )
        self.res = nn.ModuleList(
            nn.Conv1d(
                self.GROUPED_CHANNELS,
                self.GROUPED_CHANNELS,
                kernel_size=self.RES_KERNEL,
                padding=dilation,
                dilation=dilation,
            )
            for dilation in self.RES_DILATIONS
        )
        for conv in [*self.depthwise, self.pointwise, self.grouped, *self.res]:
            _fill_param(generator, conv.weight, 0.4)
            _fill_param(generator, conv.bias, 0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = x
        for conv in self.depthwise:
            hidden = nn.functional.leaky_relu(conv(hidden), self.SLOPE)
        depthwise_out = hidden
        hidden = self.grouped(self.pointwise(hidden))
        for conv in self.res:
            hidden = hidden + conv(nn.functional.leaky_relu(hidden, self.SLOPE))
        return depthwise_out, hidden


class ConvTransposeStack(nn.Module):
    """conv_transpose1d (ADR 0015) — a shrunk form of dec's ups and the bias-less conv_post.

    - The observed ups are up_rates [8,8,2,2,2] / up_k [16,16,8,2,2]. Here there is one each of
      **up 2 (k 2 / pad 0)** and **up 8 (k 16 / pad 4)** (forms where `2·pad == k − u` holds for
      both).
    - **The channel counts are asymmetric at every step** (5 → 3 → 2 → 1). Reading the weight
      `[Cin, Cout, K]` as `[Cout, Cin, K]` by mistake matches the element count when
      Cin == Cout, and passes both the shape check and the golden (recon §4 / ADR 0015).
    - The final step is **a conv1d without bias** (`Conv1d(2, 1, 7, padding=3, bias=False)` — the
      same shape as the observed conv_post). The exporter's zero-bias synthesis runs through here.
    """

    IN_CHANNELS = 5
    MID_CHANNELS = 3
    UP_CHANNELS = 2
    POST_KERNEL = 7
    SLOPE = 0.1

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        # (k, stride, padding) are the 2 observed forms satisfying 2·padding == k − stride
        self.up_small = nn.ConvTranspose1d(
            self.IN_CHANNELS, self.MID_CHANNELS, kernel_size=2, stride=2, padding=0
        )
        self.up_large = nn.ConvTranspose1d(
            self.MID_CHANNELS, self.UP_CHANNELS, kernel_size=16, stride=8, padding=4
        )
        self.post = nn.Conv1d(
            self.UP_CHANNELS, 1, kernel_size=self.POST_KERNEL, padding=3, bias=False
        )
        for conv in (self.up_small, self.up_large):
            _fill_param(generator, conv.weight, 0.4)
            _fill_param(generator, conv.bias, 0.1)
        _fill_param(generator, self.post.weight, 0.4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = nn.functional.leaky_relu(self.up_small(x), self.SLOPE)
        hidden = nn.functional.leaky_relu(self.up_large(hidden), self.SLOPE)
        return hidden, torch.tanh(self.post(hidden))


class RmsNormBlock(nn.Module):
    """RMSNorm's **2 supply routes** (ADR 0017) and synthesis of omitted slots (ADR 0016).

    - **The hand-written decomposed form** `x · rsqrt(mean(x²)+eps) · weight` — the path that
      `normalize._fold_rms_norm` folds into 1 node (Qwen3 / DiT use this form). Without folding,
      mean / rsqrt would fail as unsupported ops.
    - **`nn.RMSNorm`** — the path that keeps `aten.rms_norm` as PRESERVED (from diffusers). Both
      with and without affine are present, and the side without exercises **ones synthesis**
      (normalisation to arity 2).
    - **linear without bias** and **layer_norm without affine** — zero bias / ones and zeros
      synthesis.

    MUST: all 3 eps are **different values**. A bug that drops eps from attrs and falls back to
    the default does not show in the values when every eps is the same (the same shape as Anima,
    where layer_norm's observed 1e-7 and rms's 1e-6 live together).
    MUST: weight is away from 1 (left as `ones`, a mix-up in ones synthesis would not show in the
    values).
    """

    HIDDEN = 8
    FOLD_EPS = 1e-6
    RMS_EPS = 1e-5
    NORM_EPS = 1e-7

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.weight = nn.Parameter(_uniform(generator, self.HIDDEN, low=0.4, high=1.6))
        self.affine = nn.RMSNorm(self.HIDDEN, eps=self.RMS_EPS)
        self.plain = nn.RMSNorm(self.HIDDEN, eps=self.RMS_EPS, elementwise_affine=False)
        self.dense = nn.Linear(self.HIDDEN, self.HIDDEN, bias=False)
        self.layer_norm = nn.LayerNorm(self.HIDDEN, eps=self.NORM_EPS, elementwise_affine=False)
        _fill_param(generator, self.affine.weight, 0.5)
        _fill_param(generator, self.dense.weight, 0.3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        squared = x.pow(2).mean(-1, keepdim=True)
        folded = x * torch.rsqrt(squared + self.FOLD_EPS) * self.weight
        projected = self.dense(folded)
        return self.affine(projected), self.plain(projected), self.layer_norm(projected)


class Conv2dBlock(nn.Module):
    """conv2d (ADR 0017) and the `clamp_min` of channel L2 normalisation.

    MUST (the test requirement of ADR 0017 — a weight layout mix-up matches the element count and
    passes the shape check): exercises all of **Kh≠Kw / H/W-asymmetric stride and padding / Cin
    and Cout both 2 or more and different from each other**. It also has one each of groups
    (1 < g < C), an asymmetric dilation and no bias (zero-bias synthesis). The input has H≠W too,
    so a height / width mix-up shows in the values.

    MUST: the L2 input has **a position whose norm is exactly 0** (`example_inputs` zeroes one
    point). If the floor of `clamp_min` is dropped, it gives 0/0 = NaN and the golden comparison
    turns red there — the only point that pins, non-tautologically, that the floor takes effect.

    NOTE: the channel L2 mirrors `_shared.vae_rank4.l2_normalize` (`dim=1`) and is **an axis sum
    without permute** (attrs `dim=1`). It is also the only path that exercises **the axis-reduce
    variant** with a torch reference in a real-GPU golden (the `sum` of every other golden is over
    the last dimension = the row kernel).
    """

    EPS = 1e-12

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        # Kh≠Kw / stride (1,2) / padding (1,0) / Cin 4 → Cout 6
        self.wide = nn.Conv2d(4, 6, kernel_size=(3, 1), stride=(1, 2), padding=(1, 0))
        # groups 3 (Cin/groups = 2), no bias → zero-bias synthesis
        self.grouped = nn.Conv2d(6, 3, kernel_size=(1, 3), padding=(0, 1), groups=3, bias=False)
        # H/W-asymmetric dilation ((2,1)) and padding (2,1)
        self.dilated = nn.Conv2d(3, 2, kernel_size=(2, 2), padding=(2, 1), dilation=(2, 1))
        for conv in (self.wide, self.dilated):
            _fill_param(generator, conv.weight, 0.4)
            _fill_param(generator, conv.bias, 0.1)
        _fill_param(generator, self.grouped.weight, 0.4)

    def forward(self, x: torch.Tensor, channels: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.dilated(self.grouped(self.wide(x)))
        norm = torch.sqrt(torch.sum(channels * channels, dim=1)).clamp(min=self.EPS).unsqueeze(1)
        return hidden, channels / norm


class Conv3dBlock(nn.Module):
    """torch comparison of unbatched conv3d (extended composite layer — ADR 0118 decision 1).

    MUST (a weight layout mix-up matches the element count and passes the shape check):
    exercises all of **Kt≠Kh≠Kw / stride, padding and dilation asymmetric on the 3 axes / Cin and
    Cout both 2 or more and different from each other**. It also has one without bias (zero-bias
    synthesis). The input has T≠H≠W too, so an axis mix-up shows in the values.

    MUST: the input is **unbatched rank 4 `[Cin, T, H, W]`** (the unbatched form of `nn.Conv3d` —
    `aten.conv3d.default` stays in the trace with its rank-4 input). Batched rank 5 is outside the
    contract.
    MUST: groups is 1 only. The golden is compared on a real GPU, and the GPU's implemented subset
    is groups == 1 (groups > 1 fails loudly at plan time — the general groups semantics are held by
    the CPU reference and the conformance table).
    """

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        # Kt≠Kh≠Kw (3,2,1) / stride (1,2,1) / padding (0,1,0) / Cin 4 → Cout 6
        self.wide = nn.Conv3d(4, 6, kernel_size=(3, 2, 1), stride=(1, 2, 1), padding=(0, 1, 0))
        # No bias → zero-bias synthesis (kernel 3 on W only, padding 1 on W only)
        self.plain = nn.Conv3d(6, 3, kernel_size=(1, 1, 3), padding=(0, 0, 1), bias=False)
        # Dilation asymmetric on the 3 axes (1,2,1), temporal stride 2, padding (1,2,1)
        self.dilated = nn.Conv3d(
            3, 2, kernel_size=(2, 2, 2), stride=(2, 1, 1), padding=(1, 2, 1), dilation=(1, 2, 1)
        )
        for conv in (self.wide, self.dilated):
            _fill_param(generator, conv.weight, 0.4)
            _fill_param(generator, conv.bias, 0.1)
        _fill_param(generator, self.plain.weight, 0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dilated(self.plain(self.wide(x)))


class DeformConvBlock(nn.Module):
    """torch comparison of deform_conv2d (DCNv2, extended atomic layer — ADR 0055).

    A new atomic op has no existing path to pair with, so **this golden, with the torchvision
    output baked in, is the main check** (the other check is "offset all 0, mask all 1 → bit
    identical to conv2d", held by the real-GPU test side). The shapes to exercise:

    MUST: **Kh ≠ Kw / Cin ≠ Cout / padding with H ≠ W**. A mix-up of the offset channel order
    (even = y / odd = x) can give matching values with a square kernel and symmetric padding.
    MUST: a range where **the offset points outside the input plane** (±2.5). Both levels of zero
    fill are exercised (if the centre is out of range the whole tap is 0; inside, an out-of-range
    corner is 0 for that corner only) — a border-clamp implementation turns red only here.
    MUST: offset is **a graph input** (as a Parameter it would fall into an initializer and miss
    this op's premise of "an offset that is a run-time value").
    MUST: the mask's range is **[0, 2]** (BiRefNet's `2·sigmoid(...)`). Narrowed to [0,1] it could
    become a form where "forgetting to apply the mask makes little difference", and it would not
    separate the bug of applying it **before** interpolation either.
    NOTE: the `point` branch is k=1 (ASPP's `aspp1`) **without bias** — torchvision's Python
    wrapper inserts `aten.full([Cout], 0)`, exercising the path where constant folding of the
    export-eliminated layer turns it into an initializer.
    """

    IN_CHANNELS = 3
    WIDE_CHANNELS = 5
    POINT_CHANNELS = 4

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        # Kh=3 / Kw=2 / padding (1,0) — Cin 3 → Cout 5
        self.weight = nn.Parameter(
            _uniform(generator, self.WIDE_CHANNELS, self.IN_CHANNELS, 3, 2, low=-1.0, high=1.0)
        )
        self.bias = nn.Parameter(_uniform(generator, self.WIDE_CHANNELS, low=-0.5, high=0.5))
        # k=1 / padding 0 / no bias
        self.point = nn.Parameter(
            _uniform(generator, self.POINT_CHANNELS, self.IN_CHANNELS, 1, 1, low=-1.0, high=1.0)
        )

    def forward(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
        point_offset: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        wide = deform_conv2d(
            input=x, offset=offset, weight=self.weight, bias=self.bias, padding=(1, 0), mask=mask
        )
        point = deform_conv2d(
            input=x, offset=point_offset, weight=self.point, padding=(0, 0), mask=point_mask
        )
        return wide, point


class GruScanBlock(nn.Module):
    """torch comparison of gru_scan / gru_scan_reverse (extended composite layer — ADR 0056).

    The input-side GEMM is a plain `nn.Linear` (= the IR's `linear`); the op handles only the
    hidden-side recurrence. The eager expected values come from the body in `karume/custom_ops.py`
    (bit-exact agreement with a unidirectional 1-layer `nn.GRU` under `torch.equal` is the
    observation of ADR 0056 decision 6). The shapes to exercise:

    MUST: **the time axis is symbolic**. The reason this op exists is "passing T through as a
    symbol", so a golden with static T shows no difference from the decomposed form
    (`aten.gru.input` is unrolled T times and specialised).
    MUST: **both directions are output**. The reverse direction is "only the scan order is
    reversed; the writes follow the forward time order", and the bug of inserting an output-side
    `flip` turns red only here.
    MUST: **h0 is exercised two ways** — the forward direction uses `torch.zeros` (the path where
    constant folding of the export-eliminated layer turns it into an initializer), the reverse
    direction **a graph input** (an initial state that is a run-time value). With a zero h0 alone,
    "a kernel that skips reading h0" stays green.
    MUST: the hidden-side weights are smallish (if the gates saturated over the H-term
    contraction, rounding differences of the update formula would not show in the values and the
    comparison would become tautological).
    """

    IN_FEATURES = 4
    HIDDEN = 5

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.project = nn.Linear(self.IN_FEATURES, 3 * self.HIDDEN)
        _fill_param(generator, self.project.weight, 0.4)
        _fill_param(generator, self.project.bias, 0.2)
        self.weight_hh = nn.Parameter(
            _uniform(generator, 3 * self.HIDDEN, self.HIDDEN, low=-0.5, high=0.5)
        )
        self.bias_hh = nn.Parameter(_uniform(generator, 3 * self.HIDDEN, low=-0.3, high=0.3))

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        gates = self.project(x)
        zeros = torch.zeros(x.shape[1], self.HIDDEN)
        return (
            torch.ops.karume.gru_scan(gates, zeros, self.weight_hh, self.bias_hh),
            torch.ops.karume.gru_scan_reverse(gates, state, self.weight_hh, self.bias_hh),
        )


class BilinearResize(nn.Module):
    """torch comparison of upsample_bilinear2d (Core ATen layer, align_corners=True only).

    A new atomic op has no existing path to pair with (a decomposed form / another kernel), so
    **this golden, with the torch output baked in, is the main check**. The shapes to exercise:

    MUST: **a non-integer factor** (4×5 → 7×9 has scales 3/6 and 4/8, and the weights differ per
    output position). Pinning only integer factors would let "a weight table hard-coded with
    period 2" pass.
    MUST: **downscaling** (4×5 → 2×3). A form with 8 occurrences in practice (BiRefNet's
    `forward_enc` / `cxt`), pinning that it goes through the same formula and the same 2 taps as
    upscaling.
    MUST: **H ≠ W and Hout ≠ Wout**. With a square, an H / W mix-up does not show in the values.
    MUST: **an input height of 1** (`narrow`). The align_corners scale is `(in−1)/(out−1)`, so an
    axis with in = 1 has scale 0 = row duplication (the only form in which the tail special case
    `index1 = index0` fires for every output). The width of the same tensor is downscaled, so the
    2 axes exercise different branches at once.
    """

    def forward(self, x: torch.Tensor, narrow: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            nn.functional.interpolate(x, size=(7, 9), mode="bilinear", align_corners=True),
            nn.functional.interpolate(x, size=(2, 3), mode="bilinear", align_corners=True),
            nn.functional.interpolate(narrow, size=(3, 4), mode="bilinear", align_corners=True),
        )


class Int8Weights(nn.Module):
    """The golden that exercises, with i8 storage (per-channel scale), 5 of the 6 ops of
    `WEIGHT_SLOTS`, all but conv3d (ADR 0019; i8 conv3d is held by the runtime's GPU tests).

    MUST: neither **the row length nor the total element count of a weight is a multiple of 4**.
    i8 packs 4 elements into 1 u32, so the bug of "deriving word and lane from the index relative
    to the row" **agrees by chance only when the row length is a multiple of 4** (the same trap
    as f16's even / odd — ADR 0019). Pinning only multiples of 4 would let this bug pass the
    golden. Actual shapes: embedding `[7,5]` / linear `[3,5]` / conv1d `[5,3,3]` /
    conv_transpose1d `[5,2,3]` / conv2d `[3,2,3,1]` — neither the element count nor the block
    length per output channel is a multiple of 4 in any of them.

    MUST: one embedding row is **all zero** (a channel with `amax == 0`). If the lower clamp of the
    scale is dropped, it gives `0/0 = NaN`, and this one row alone turns the golden red — the only
    point that pins, non-tautologically, that the lower bound takes effect.

    MUST: conv_transpose1d is mixed in (its weight has the transposed layout `[Cin,Cout,K]` and
    per-channel axis **1**). A mix-up of the axis table does not show in the values for the other
    4 ops.
    """

    VOCAB = 7
    EMBED = 5
    HIDDEN = 3
    SIGNAL_IN = 3
    SIGNAL_MID = 5
    SIGNAL_OUT = 2
    UP_STRIDE = 3
    IMAGE_IN = 2
    IMAGE_OUT = 3

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.table = nn.Embedding(self.VOCAB, self.EMBED)
        self.dense = nn.Linear(self.EMBED, self.HIDDEN)
        self.conv = nn.Conv1d(self.SIGNAL_IN, self.SIGNAL_MID, kernel_size=3, padding=1)
        # 2·padding == k − stride (the form with output length L·stride that the runtime accepts
        # — ADR 0015)
        self.up = nn.ConvTranspose1d(
            self.SIGNAL_MID, self.SIGNAL_OUT, kernel_size=self.UP_STRIDE, stride=self.UP_STRIDE
        )
        self.image = nn.Conv2d(self.IMAGE_IN, self.IMAGE_OUT, kernel_size=(3, 1), padding=(1, 0))
        for module in (self.dense, self.conv, self.up, self.image):
            _fill_param(generator, module.weight, 0.5)
            _fill_param(generator, module.bias, 0.2)
        _fill_param(generator, self.table.weight, 1.0)
        with torch.no_grad():
            self.table.weight[0].zero_()

    def forward(
        self, index: torch.Tensor, signal: torch.Tensor, image: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return (
            self.dense(self.table(index)),
            self.up(torch.tanh(self.conv(signal))),
            self.image(image),
        )


def _zeroed_column(generator: torch.Generator, *shape: int, at: tuple[int, int]) -> torch.Tensor:
    """`[1,C,H,W]` with every channel 0 at a single spatial position (exercises the floor of
    `clamp_min`)."""
    tensor = _uniform(generator, *shape, low=-1.5, high=1.5)
    tensor[:, :, at[0], at[1]] = 0.0
    return tensor


class StaticQuantizeBlock(nn.Module):
    """A tiny golden that runs the preserved SRQ and the identity behaviour at scale=0 (ADR
    0097)."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.ops.karume.static_quantize(x, 1.0),
            torch.ops.karume.static_quantize(x, 0.09940945357084274),
            torch.ops.karume.static_quantize(x, 0.0),
        )
