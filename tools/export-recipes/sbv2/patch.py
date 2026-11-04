"""The patch layer that makes SBV2 (Style-Bert-VITS2 JP-Extra) exportable with torch.export (ADR
0013).

The targets are the **front** (one graph fusing enc_p + dp + sdp reverse) and, in the second half
of the chain, the **flow** (the reverse of TransformerCouplingBlock) / the **voice** (flow + dec
fused). The patches replace attributes of imported classes (monkeypatch) and add wrappers; the
`style_bert_vits2` package itself is not touched (**dec** needs no patch — `remove_weight_norm` is
the only preprocessing).

3 patches + 2 wrappers:

1. **Branch-free, non-destructive spline** — an equivalent implementation of `transforms.py`
   without boolean-mask indexing / Python branches on tensor values / in-place ops. The original
   "extracts only the elements inside the interval into 1 dimension", so the element count depends
   on the data and `torch.export` fails with GuardOnDataDependentSymNode.
2. **Relative-position attention as a gather** — replaces the quadratic shape expressions
   `2P−1` / `2P²` / `P(2P−1)` created by the rel⇄abs shift (144 places on the front side) with a
   P-independent index table + a gather on the last dimension. A quadratic expression cannot be
   rescued even by an affine extension of the dimension language (`coeff·sym+offset`), so this is
   **a structural precondition for a dynamic P**, not an optimization.
3. **FFN's explicit pad → conv's padding argument** — folds `constant_pad_nd` away (exactly
   equivalent only for an odd kernel that is not causal — see `_patched_ffn_forward` below).

MUST: the patched modules are **eager-equivalent** to the unpatched ones. A change that is not
equivalent MUST NOT be placed here (`sbv2.export --verify` measures it with the real weights and
several P).

MUST: the patches replace class attributes **process-wide**, so "the unpatched reference" can be
taken only once per process. {@link patches_applied} answers whether they are applied, and the
caller (`sbv2.export`) rejects an order violation by failing loudly (blocking vacuous passes =
false PASSes).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional

#: The spline defaults (the same values as `style_bert_vits2.models.transforms`). They appear as
#: defaults in the signature of the replacement function, so if they disagree with the original
#: implementation a call that omits them gets a different spline (a silently wrong value).
DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

#: The patches-applied flag. Held only to make the side effect of the process-wide replacement
#: visible (taking "the unpatched reference" after patching makes the equivalence check vacuous —
#: ADR 0013).
_APPLIED = False


def patches_applied() -> bool:
    """Whether this process has already applied the patches.

    The gate the reference-taking side (the equivalence check) uses to assert "not applied yet".
    """
    return _APPLIED


# ---- ① Branch-free, non-destructive spline --------------------------------------


def _constant_column(t: torch.Tensor, value: float) -> torch.Tensor:
    """Builds a constant column of the same shape as `t[..., :1]` **using only the existing
    vocabulary**.

    `full_like` / `zeros_like` are not in the IR vocabulary (adding a constant-generating op would
    bring a new concept into the IR: a constant that depends on the runtime shape). `x * 0.0 +
    value` is exactly `value` as long as t is finite — the values reaching the spline are clamped
    and finite, so the precondition holds.
    """
    return t[..., :1] * 0.0 + value


def searchsorted_free(
    bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """The equivalent of the original `transforms.searchsorted` without its in-place mutation.

    The original **mutates its argument** with `bin_locations[..., -1] += eps` before comparing.
    The non-destructive form (rebuilding with cat a column with eps added to the last entry only) is
    equivalent because the mutated last element is **never referenced** by the later gather — the
    return value is `sum(...) - 1`, at most `num_bins - 1`, so the gather's index never reaches
    beyond the element just before the last one. Without carrying this reason it looks like "a
    change that is not equivalent".
    """
    last = bin_locations[..., -1:] + eps
    shifted = torch.cat([bin_locations[..., :-1], last], dim=-1)
    return torch.sum(inputs[..., None] >= shifted, dim=-1) - 1


def rational_quadratic_spline_free(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    left: float = 0.0,
    right: float = 1.0,
    bottom: float = 0.0,
    top: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The branch-free, non-destructive version of `transforms.rational_quadratic_spline`
    (numerically equivalent).

    Only 3 differences from the original:

    - **The domain check is removed** — `if torch.min(inputs) < left ...: raise` is a Python
      branch on tensor values and becomes a data-dependent guard at export. The caller
      (`unconstrained_rqs_free`) only passes values already clamped to the domain, so the check is
      always true. **Now that the safety valve is gone, this function MUST NOT be called without
      the clamp** (this is why the `tails is None` branch of the replacement `piecewise_free` is a
      RuntimeError).
    - **Building the cumulative bounds** — the form that `pad`s and then slice-assigns
      `cumwidths[..., 0] = left` / `[..., -1] = right` is replaced by a construction with cat
      (removing the in-place op). The values are identical.
    - **The non-destructive searchsorted** ({@link searchsorted_free}).
    """
    num_bins = unnormalized_widths.shape[-1]
    # A precondition decided by shapes alone (independent of tensor values), so it can stay a
    # Python branch.
    if min_bin_width * num_bins > 1.0:
        raise ValueError(f"min_bin_width {min_bin_width} is too large for {num_bins} bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError(f"min_bin_height {min_bin_height} is too large for {num_bins} bins")

    widths = functional.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    scaled_widths = (right - left) * torch.cumsum(widths, dim=-1) + left
    # The same column as the original's "pad a 0 at the start → scale → overwrite both ends with
    # left / right". The last element is dropped (the final value of the cumulative sum is the one
    # that gets overwritten).
    cumwidths = torch.cat(
        [
            _constant_column(scaled_widths, left),
            scaled_widths[..., :-1],
            _constant_column(scaled_widths, right),
        ],
        dim=-1,
    )
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + functional.softplus(unnormalized_derivatives)

    heights = functional.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    scaled_heights = (top - bottom) * torch.cumsum(heights, dim=-1) + bottom
    cumheights = torch.cat(
        [
            _constant_column(scaled_heights, bottom),
            scaled_heights[..., :-1],
            _constant_column(scaled_heights, top),
        ],
        dim=-1,
    )
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    locations = cumheights if inverse else cumwidths
    bin_idx = searchsorted_free(locations, inputs)[..., None]

    # The original `gather(...)[..., 0]` becomes `aten.select.int` (an indexing access that drops
    # an axis). The IR has no select, and this form, which drops an axis of length 1, is a pure
    # meta operation that does not change the element order, so it is written as `squeeze(-1)`
    # (= reshape). Values and layout are identical.
    input_cumwidths = cumwidths.gather(-1, bin_idx).squeeze(-1)
    input_bin_widths = widths.gather(-1, bin_idx).squeeze(-1)
    input_cumheights = cumheights.gather(-1, bin_idx).squeeze(-1)
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx).squeeze(-1)
    input_derivatives = derivatives.gather(-1, bin_idx).squeeze(-1)
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx).squeeze(-1)
    input_heights = heights.gather(-1, bin_idx).squeeze(-1)

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)
        # The original's `assert (discriminant >= 0).all()` is also a Python branch on tensor
        # values, so it is dropped (inside the domain it is mathematically non-negative — the
        # inverse of a monotonic rational-quadratic).
        discriminant = b.pow(2) - 4 * a * c
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)
    numerator = input_heights * (
        input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / denominator
    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
    return outputs, logabsdet


def unconstrained_rqs_free(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tails: str = "linear",
    tail_bound: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The equivalent version with boolean-mask indexing replaced by clamp + where.

    The original extracts only the elements inside the interval into 1 dimension with
    `inputs[inside_interval_mask]`, so **the element count is decided by tensor values** (not
    exportable). Every element is clamped to the domain and passed through the spline, and the
    elements outside the interval are put back to the input itself with where — for elements inside
    the interval the clamp is the identity (they are already in the domain), so the chosen values
    match the original exactly, and for elements outside the interval the spline's result is
    discarded, so whatever the clamp returns has no effect.
    """
    if tails != "linear":
        raise RuntimeError(f"{tails} tails are not implemented.")

    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)

    # The same column as the original's "pad both ends → overwrite the 2 end elements with
    # constant" (removing the in-place op).
    constant = float(np.log(np.exp(1 - min_derivative) - 1))
    tail_column = _constant_column(unnormalized_derivatives, constant)
    padded_derivatives = torch.cat([tail_column, unnormalized_derivatives, tail_column], dim=-1)

    spline_outputs, spline_logabsdet = rational_quadratic_spline_free(
        inputs=torch.clamp(inputs, -tail_bound, tail_bound),
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=padded_derivatives,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    outputs = torch.where(inside, spline_outputs, inputs)
    # The logabsdet outside the interval is 0 (exactly the original's `torch.zeros_like` initial
    # value). The constant 0 is built as `x * 0.0` — the standard technique to keep zeros_like out
    # of the IR.
    logabsdet = torch.where(inside, spline_logabsdet, spline_logabsdet * 0.0)
    return outputs, logabsdet


def apply_spline_patch() -> None:
    """Replaces the spline implementation with the branch-free version.

    MUST: the replacement target is **`style_bert_vits2.models.modules`** — `modules.py` has
    already bound the function object into its own namespace with
    `from ...transforms import piecewise_rational_quadratic_transform`, so replacing it on the
    `transforms` side has no effect on the calls (the original would run while we think it is
    replaced, and only the export would fail).
    """
    from style_bert_vits2.models import modules

    def piecewise_free(
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=False,
        tails=None,
        tail_bound=1.0,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
    ):
        if tails is None:
            # MUST: do not swallow this. `rational_quadratic_spline_free` removes the original's
            # domain check (InputOutsideDomain), so tails=None would be a path straight into that
            # spline without the clamp = **the form with only the safety valve removed**. SBV2's
            # caller (ConvFlow) always uses tails="linear", so it is unreachable, but if it is
            # reached it fails.
            raise RuntimeError(
                "tails=None is not supported (the spline is only called through the domain clamp)"
            )
        return unconstrained_rqs_free(
            inputs,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=inverse,
            tails=tails,
            tail_bound=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )

    modules.piecewise_rational_quadratic_transform = piecewise_free


# ---- ② Relative-position attention as a gather -------------------------------------------


def build_relattn_tables(
    length: int,
    window_size: int = 4,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds the `(T, T)` tables `(idx_k, valid)` of the gather attention at the real length (the
    Python-side source of truth).

    The front (P ≤ 512) builds the same formula **in-graph** and lets the exporter's constant
    folding bake in Pmax + create `sym_prefix_slice`, but the flow's Ty is orders of magnitude
    larger and baking in would be O(Tmax²) (134MB at sym_max=4096), so the tables are **promoted to
    graph inputs** (ADR 0013). Promotion moves the responsibility for building the table values to
    the host, so this is where the golden comes from,
    `packages/models/src/sbv2/relattn-tables.ts` is the TS-side mirror, and
    `packages/models/tests/sbv2_relattn_parity_test.ts` pins that the two are byte-identical with
    the golden's real data.

    MUST: the front's in-graph construction also **calls this function** (the
    `relattn_tables is None` path of {@link mha_gather_forward}). Writing the formula in 2 places
    means that when only one is fixed, the front (baked in) and the flow (input) get different
    tables, and since both shapes still match it silently becomes a different model (the silently
    wrong value class). `device` / `dtype` are arguments that keep the in-graph construction during
    tracing from producing an extra `_to_copy`; the host-supply side uses the defaults (CPU / f32).
    """
    positions = torch.arange(length, device=device)
    rel = positions.unsqueeze(0) - positions.unsqueeze(1)
    idx_k = torch.clamp(rel + window_size, 0, 2 * window_size)
    valid = (torch.abs(rel) <= window_size).to(dtype)
    return idx_k, valid


def mha_gather_forward(
    self,
    x: torch.Tensor,
    c: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    relattn_tables: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """The gather version of `MultiHeadAttention.forward` for windowed relative-position attention
    (equivalent to the original).

    The original maps the relative logits `(b,h,P,2P−1)` to absolute positions `(b,h,P,P)` with a
    pad / view shift. That path creates **quadratic shape expressions** `2P−1` / `2P²` /
    `P(2P−1)`, so export does not work with a dynamic P. Using the fact that everything outside the
    window width `2w+1` is identically 0, the same values are built with index tables and gather:

    - key side: `scores_local[i,j] = (q_i·Ek)[w + (j−i)] · [|j−i| ≤ w]`
      → a gather with the index table `idx_k[i,j] = clamp(w + j − i, 0, 2w)` × a 0/1 mask multiply.
      Outside the window the clamped index **reads another embedding**, but the multiply by the
      valid mask always makes it 0 (a form that does not rely on gather's out-of-range convention
      — ADR 0013).
    - value side: `out_i += Σ_j p[i,j]·Ev[w + (j−i)]` (0 outside the window)
      → zero-pad p by `[w,w]` and gather `idx_v[i,c] = i + c`. For the padded length `P+2w`, `i+c`
      is always in range, so outside the window it reads exactly 0 (no clamp needed).

    The tables are decided by `i` and `j` (and the constant w) alone and **do not depend on P**, so
    the exporter's constant folding bakes them in at Pmax + lowers them to `sym_prefix_slice` (the
    2-point evaluation check passes too).

    Passing `relattn_tables=(idx_k, valid)` switches only the key-side `(T, T)` table to **external
    supply** (the flow path — {@link FlowReverse} threads the graph inputs down to here). With None
    it is built in-graph (the front path — eligible for baking in). The values are identical on
    both paths (the same formula as {@link build_relattn_tables}). **The value-side `idx_v` is a
    small `(T, 2w+1)`; baked in it is only 150KB at Tmax=4096**, so it stays in-graph on the flow
    too (every extra graph input adds to the responsibilities of the host-side mirror
    implementation).

    MUST: the embedding side is **expanded to 4D and multiplied 4D×4D**. A matmul kept in 2D is
    decomposed into a `(b·h·P, kc)` view, a symbolic dimension with a coefficient, `2P`, appears,
    and it looks as if the affine extension of the dimension language were needed (it is not —
    recon §4).
    """
    if self.window_size is None:
        raise ValueError(
            "the gather form targets windowed relative-position attention (window_size=None)"
        )
    if not self.heads_share:
        raise ValueError(
            "heads_share=False is not supported — expanding the embedding's head axis (length 1)"
            " to every head no longer works"
        )
    if self.block_length is not None or self.proximal_bias:
        raise ValueError(
            f"unsupported attention configuration (block_length={self.block_length} /"
            f" proximal_bias={self.proximal_bias})"
        )

    q = self.conv_q(x)
    k = self.conv_k(c)
    v = self.conv_v(c)

    b, d, t_s = k.size()
    t_t = q.size(2)
    if t_s != t_t:
        raise ValueError(
            "the gather form targets self-attention (a square relative-position table)"
        )
    h = self.n_heads
    kc = self.k_channels
    w = self.window_size
    q = q.view(b, h, kc, t_t).transpose(2, 3)
    k = k.view(b, h, kc, t_s).transpose(2, 3)
    v = v.view(b, h, kc, t_s).transpose(2, 3)

    scaled_q = q / math.sqrt(kc)
    scores = torch.matmul(scaled_q, k.transpose(-2, -1))

    # rel[i,j] = j − i (the same direction as the original's rel→abs mapping
    # `x_final[i,j] = x[i, j−i+P−1]`).
    positions = torch.arange(t_t, device=x.device)
    if relattn_tables is None:
        idx_k, valid = build_relattn_tables(t_t, w, device=x.device, dtype=scores.dtype)
    else:
        idx_k, valid = relattn_tables

    # emb_rel_k is [1, 2w+1, kc] (heads_share, so the head axis has length 1). The head axis is
    # used as the broadcast source as is to build [1, 1, kc, 2w+1] (without going through the
    # select of `emb[0]`).
    ek = self.emb_rel_k.permute(0, 2, 1).unsqueeze(0)
    rel_local = torch.matmul(scaled_q, ek.expand(b, h, kc, 2 * w + 1))
    scores_local = torch.gather(
        rel_local, -1, idx_k.unsqueeze(0).unsqueeze(0).expand(b, h, t_t, t_t)
    ) * valid.unsqueeze(0).unsqueeze(0)
    scores = scores + scores_local

    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, -1e4)
    p_attn = functional.softmax(scores, dim=-1)
    p_attn = self.drop(p_attn)
    output = torch.matmul(p_attn, v)

    p_pad = functional.pad(p_attn, [w, w])
    window = torch.arange(2 * w + 1, device=x.device)
    idx_v = positions.unsqueeze(1) + window.unsqueeze(0)
    p_local = torch.gather(p_pad, -1, idx_v.unsqueeze(0).unsqueeze(0).expand(b, h, t_t, 2 * w + 1))
    ev = self.emb_rel_v.unsqueeze(0)
    output = output + torch.matmul(p_local, ev.expand(b, h, 2 * w + 1, kc))

    output = output.transpose(2, 3).contiguous().view(b, d, t_t)
    return self.conv_o(output)


def apply_gather_relattn_patch() -> None:
    """Replaces `MultiHeadAttention.forward` with the gather version.

    The original **assigns to the module as a side effect** with `x, self.attn =
    self.attention(...)`, which by itself is a warning at export. The replacement removes the side
    effect along with it (nobody on the inference path reads `self.attn`).
    """
    from style_bert_vits2.models.attentions import MultiHeadAttention

    MultiHeadAttention.forward = mha_gather_forward


# ---- ③ FFN's explicit pad → conv's padding ------------------------------------


def _patched_ffn_forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
    """`FFN.forward` with the explicit `F.pad` removed and folded into conv's padding argument.

    MUST: it is equivalent **only for an odd kernel that is not causal**. The original's
    `_same_padding` pads asymmetrically, left `(k−1)//2` / right `k//2`, so for an even kernel left
    and right differ by 1 (conv1d's padding argument can only express symmetric padding). The causal
    side pads only the left by `k−1`, so it is a different thing altogether. Letting either through
    silently would give a different model whose output is shifted by 1 sample, so they fail.
    """
    if self.causal:
        raise ValueError("a causal FFN is outside this patch (asymmetric left/right padding)")
    if self.kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size={self.kernel_size} is even — the explicit pad and conv's padding are not"
            " equivalent"
        )
    padding = (self.kernel_size - 1) // 2
    x = functional.conv1d(x * x_mask, self.conv_1.weight, self.conv_1.bias, padding=padding)
    x = x * torch.sigmoid(1.702 * x) if self.activation == "gelu" else torch.relu(x)
    x = self.drop(x)
    x = functional.conv1d(x * x_mask, self.conv_2.weight, self.conv_2.bias, padding=padding)
    return x * x_mask


def apply_ffn_conv_padding_patch() -> None:
    """Replaces `FFN.forward` with the folded-pad version."""
    from style_bert_vits2.models.attentions import FFN

    FFN.forward = _patched_ffn_forward


def apply_all_patches() -> None:
    """Applies every patch SBV2's export needs (idempotent). Called by each of the front / flow /
    voice targets and by `demo` / `measure_quant` alike.

    MUST: call it **after** whatever takes "the unpatched reference". It replaces class attributes
    process-wide, so applying it first turns every later reference value into a patched value, and
    the equivalence check becomes vacuous and gives a false PASS (ADR 0013).
    """
    global _APPLIED
    apply_spline_patch()
    apply_gather_relattn_patch()
    apply_ffn_conv_padding_patch()
    _APPLIED = True


# ---- ④ Externalizing the random numbers + the fused front wrapper -----------------------------


class SdpReverseNoiseIn(nn.Module):
    """The reverse path of `StochasticDurationPredictor` — with the random numbers promoted to an
    external input.

    Equivalent to the original `forward(..., reverse=True)`; the only difference is that
    `torch.randn(B,2,P) * noise_scale` is received as the external input `z_noise` (the multiply by
    `noise_scale` happens on the host — runtime knobs are not baked into the graph).
    `torch.detach` is the identity on the eager inference path, so it is not copied.

    MUST: **hold only the submodules that are used** (do not hold the whole sdp). The `post_*`
    modules the reverse path does not touch (the training-only posterior) and the dropped ConvFlow
    would, if in the tree, be counted by the fake-quant census while torch.export never visits them
    and they never become initializers — the i4 override spelling check (which prevents emit from
    silently dropping entries) fails on "a key not in the graph". Making ownership = use makes the
    census and the graph agree structurally (surfaced in wave J-5b). The selection of the flows
    (dropping the second from the end in reversed order = the unused vflow — as the original does)
    is moved into `__init__` for the same reason, and only the 7 used ones are registered.
    """

    def __init__(self, sdp: nn.Module) -> None:
        super().__init__()
        self.pre = sdp.pre
        self.cond = sdp.cond
        self.convs = sdp.convs
        self.proj = sdp.proj
        flows = list(reversed(sdp.flows))
        self.flows = nn.ModuleList([*flows[:-2], flows[-1]])

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor,
        z_noise: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pre(x)
        x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask
        z = z_noise
        for flow in self.flows:
            z = flow(z, x_mask, g=x, reverse=True)
        # The original takes the first element of `torch.split(z, [1, 1], 1)`. One outlet is
        # enough for the split.
        return z[:, :1]


class Sbv2Front(nn.Module):
    """The fused front graph of enc_p + dp + sdp(reverse) (an emit target of ADR 0013).

    - `x_mask` is an external input (the `sequence_mask` generation of the original
      `TextEncoder.forward` moves to the host). Putting the part that builds the mask from lengths
      into the graph would not make it a target of constant folding, and would take the form that
      uses `x_lengths` as a value (a path that turns symbols into values).
    - `TextEncoder.forward` is copied here — the original generates x_mask internally and splits
      m/logs with `torch.split`, so both the input shape and the split shape are fixed here (the
      values are identical to the original).
    - The outputs are `(logw_sdp, logw_dp, m_p, logs_p)`. Mixing with sdp_ratio and turning them
      into durations happen on the host.
    """

    def __init__(self, net_g: nn.Module) -> None:
        super().__init__()
        self.enc_p = net_g.enc_p
        self.sdp_rev = SdpReverseNoiseIn(net_g.sdp)
        self.dp = net_g.dp

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        style_vec: torch.Tensor,
        g: torch.Tensor,
        z_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.enc_p
        bert_emb = enc.bert_proj(bert).transpose(1, 2)
        style_emb = enc.style_proj(style_vec.unsqueeze(1))
        h = (
            enc.emb(x) + enc.tone_emb(tone) + enc.language_emb(language) + bert_emb + style_emb
        ) * math.sqrt(enc.hidden_channels)
        h = torch.transpose(h, 1, -1)
        h = enc.encoder(h * x_mask, x_mask, g=g)
        stats = enc.proj(h) * x_mask
        channels = enc.out_channels
        m_p, logs_p = stats[:, :channels], stats[:, channels:]

        logw_sdp = self.sdp_rev(h, x_mask, g, z_noise)
        logw_dp = self.dp(h, x_mask, g=g)
        return logw_sdp, logw_dp, m_p, logs_p


# ---- ⑤ flow reverse (tables promoted to inputs) and the fused voice ---------------------------


def _encoder_forward(
    enc: nn.Module,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    g: torch.Tensor,
    tables: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """A copy of `attentions.Encoder.forward` + threading of the relative-position tables.

    The front calls `enc_p.encoder(...)` **as is** (the tables are built in-graph = eligible for
    baking in). The flow has to carry the tables from the graph inputs to the attention layers, but
    the arguments of `Encoder.forward` have no slot for them. The alternative of **also replacing
    the class attribute Encoder.forward so it takes the tables** would share a branch with the front
    path (no tables) and create a path where "the front grabs the flow's tables by mistake", so it
    is not taken — keeping one copy and calling it separately keeps the paths from crossing.

    MUST: keep a line-by-line correspondence with the original (`Encoder.forward` in
    `attentions.py`), including the conditioning of the `isflow` configuration (adding
    `spk_emb_linear(g)` at layer `cond_layer_idx`). `drop` is the identity in eval, but it is kept
    so the comparison with the original can be followed line by line.
    """
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    x = x * x_mask
    for i in range(enc.n_layers):
        if i == enc.cond_layer_idx and g is not None:
            gl = enc.spk_emb_linear(g.transpose(1, 2))
            gl = gl.transpose(1, 2)
            x = x + gl
            x = x * x_mask
        y = mha_gather_forward(enc.attn_layers[i], x, x, attn_mask, tables)
        y = enc.drop(y)
        x = enc.norm_layers_1[i](x + y)

        y = enc.ffn_layers[i](x, x_mask)
        y = enc.drop(y)
        x = enc.norm_layers_2[i](x + y)
    x = x * x_mask
    return x


def _coupling_reverse(
    layer: nn.Module,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    g: torch.Tensor,
    tables: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """A copy of `modules.TransformerCouplingLayer.forward(reverse=True)` (numerically
    equivalent).

    2 differences from the original:

    - `torch.split(x, [half] * 2, 1)` → 2 explicit slices (`split_with_sizes` → `getitem` is a
      form the IR does not have, and there is no path that normalizes it into a slice — the same
      convention as the front's `stats` split).
    - **The multiply by `exp(-logs)` is folded.** With `mean_only=True` the original's `logs` is
      `torch.zeros_like(m)`, `exp(-0.0)` is **exactly 1.0** in IEEE 754, and `t * 1.0 == t` is
      bit-identical including NaN/Inf, so `(x1 − m) * exp(−logs) * x_mask` gives the same numbers
      as `(x1 − m) * x_mask`. The standard technique for keeping `zeros_like` / `full_like` out of
      the IR (recon §4), not an approximation.
    """
    if not layer.mean_only:
        raise ValueError(
            "a coupling with mean_only=False is not supported — logs is no longer 0 and"
            " exp(−logs) cannot be folded"
        )
    half = layer.half_channels
    x0, x1 = x[:, :half], x[:, half:]
    h = layer.pre(x0) * x_mask
    h = _encoder_forward(layer.enc, h, x_mask, g, tables)
    m = layer.post(h) * x_mask
    x1 = (x1 - m) * x_mask
    return torch.cat([x0, x1], 1)


class FlowReverse(nn.Module):
    """The reverse path of `TransformerCouplingBlock` — with the relative-position tables promoted
    to inputs.

    `forward(z_p, y_mask, g, idx_k, valid) -> z`. Equivalent to the original
    `TransformerCouplingBlock.forward(..., reverse=True)`; the differences are

    - the tables `idx_k` `(T, T)` i64 / `valid` `(T, T)` f32 are received as **graph inputs**
      (baked in they would be 134MB at sym_max=4096 — ADR 0013). The host builds the values with
      the mirror of {@link build_relattn_tables}.
    - `Flip` is `torch.flip(x, [1])` (the reverse branch of `modules.Flip.forward` only omits the
      logdet; the values are the same).
    - the coupling is {@link _coupling_reverse} (split → slice, folding `exp(−logs)=1`).

    The loop over the flows is unrolled on the Python side, so the IR is a fully static serial
    graph.
    """

    def __init__(self, net_g: nn.Module) -> None:
        super().__init__()
        self.flow = net_g.flow

    def forward(
        self,
        z_p: torch.Tensor,
        y_mask: torch.Tensor,
        g: torch.Tensor,
        idx_k: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        from style_bert_vits2.models import modules

        x = z_p
        for flow in reversed(self.flow.flows):
            if isinstance(flow, modules.Flip):
                x = torch.flip(x, [1])
            else:
                x = _coupling_reverse(flow, x, y_mask, g, (idx_k, valid))
        return x


class Sbv2Voice(nn.Module):
    """The fused graph of flow reverse + dec (the emit target `voice` of ADR 0013).

    `forward(z_p, y_mask, g, idx_k, valid) -> audio [1, 1, 512T]`. The same order as the end of the
    reference infer (`z = flow(z_p, y_mask, g, reverse=True)` →
    `o = dec((z * y_mask)[:, :, :max_len], g=g)` in `models_jp_extra.py`). `max_len` is always
    `None` on the inference path (= an identity slice), so it is not brought into the graph.

    MUST: `dec` has **already had `remove_weight_norm` applied** (`sbv2.export.ensure_dec_plain`).
    If weight_norm is still in place, `dec.weight` is not the effective weight, and writing it to
    the IR as is gives a different model. The f16 / i8 / i4 rounding (`sbv2.export._fake_quant`) is
    applied to the effective weights **after** the removal (ADR 0013).
    """

    def __init__(self, net_g: nn.Module) -> None:
        super().__init__()
        self.flow_rev = FlowReverse(net_g)
        self.dec = net_g.dec

    def forward(
        self,
        z_p: torch.Tensor,
        y_mask: torch.Tensor,
        g: torch.Tensor,
        idx_k: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        z = self.flow_rev(z_p, y_mask, g, idx_k, valid)
        return self.dec(z * y_mask, g=g)
