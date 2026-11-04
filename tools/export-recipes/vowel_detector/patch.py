"""Patch layer that splits `nn.GRU` into a sequence of `karume::gru_scan` (ADR 0056 decision 7).

The decomposition of `aten.gru.input` is **fully unrolled along time** by a Python loop, so
`torch.export` specializes the length axis (`Specializations unexpectedly required (T)`). The
replacement `karume::gru_scan{,_reverse}` hides its body behind `register_fake` and so passes T
through as a symbol (`karume/custom_ops.py`).

## How it is split — multiple layers / both directions are expressed **as a sequence of nodes**

The only variable-arity op in the IR is `cat`, and `gru_scan` has a fixed arity of 4 (ADR 0056
decision 1), so the `Tensor[16]` of `aten.gru` (layer × direction × 4 weights) does not fit
structurally. Layers and directions are therefore expressed by the structure of the graph:

    layer k: gi    = linear(x, W_ih_k, b_ih_k)     ← input GEMM: plain `linear`, one call for all t
             y_fwd = gru_scan(gi, h0, W_hh_k, b_hh_k)
             y_rev = gru_scan_reverse(gi_rev, h0, W_hh_k_rev, b_hh_k_rev)
             x     = cat([y_fwd, y_rev], dim=2)    ← input of the next layer

**This split is bit-identical to `nn.GRU` under `torch.equal`**
(`vowel_detector/tests/test_patch.py` pins it by measurement). It matches because it copies the
structure of torch's GRU decomposition as is:

- The input-side linear is **one call over all time** (it builds `[T,N,3H]` and then slices it per
  t), and only the hidden side runs per step. Emitting the input side as a plain `linear` therefore
  leaves the rounding sequence unchanged.
- The reverse direction reverses **only the scan order**; it writes at the forward time index
  (`flip` rejects symbolic axes, so the output-side reversal is folded into the op).
- The gate and update equations are held by the body in `karume/custom_ops.py` (`(h − n)·z + n`).

## Supported `nn.GRU` shapes

Only the shape the vowel-recognition CRNN has (batch_first, bidirectional, with bias, no dropout)
is accepted. Not adding fields follows the discipline of ADR 0023 decision 4, and **anything
unsupported fails loudly** (silently choosing another path would produce, without a sound, "the
graph bakes but the numbers are something else").
"""

from __future__ import annotations

import torch
from torch import nn

# NOTE: the import also serves the **side effect of registering** the `karume::` namespace
# (`torch.ops.karume` cannot be looked up unless it is registered).
from karume import custom_ops  # noqa: F401  — registers torch.ops.karume

#: Per direction, (`state_dict` suffix, scan op). The order is the order of `cat` itself and must
#: match the output layout of `nn.GRU` (forward direction first, reverse direction second).
_DIRECTIONS = (
    ("", torch.ops.karume.gru_scan),
    ("_reverse", torch.ops.karume.gru_scan_reverse),
)


def assert_supported(gru: nn.GRU) -> None:
    """Check that the replacement is **equivalent** for this shape (unsupported fails loudly).

    Every axis checked here is one where letting it through would change values silently:
    `batch_first=False` swaps the time and batch axes, `bidirectional=False` leaves the reverse
    half of `cat` over, `bias=False` has no `b_ih` / `b_hh` (the `gru_scan` contract requires bias
    — ADR 0056 decision 6), and `dropout` declares a different equation at training time even if
    it is a no-op at inference.
    """
    if not isinstance(gru, nn.GRU):
        raise TypeError(f"not an nn.GRU ({type(gru).__name__})")
    if not gru.batch_first:
        raise ValueError(
            "batch_first=False is a path the replacement does not have (the time axis is elsewhere)"
        )
    if not gru.bidirectional:
        raise ValueError(
            "bidirectional=False is a path the replacement does not have"
            " (the shape seen in practice is bidirectional)"
        )
    if not gru.bias:
        raise ValueError(
            "bias=False is a path the replacement does not have"
            " (the gru_scan contract requires bias)"
        )
    if gru.dropout != 0:
        raise ValueError(f"dropout={gru.dropout} is a path the replacement does not have")


def gru_forward(gru: nn.GRU, x: torch.Tensor) -> torch.Tensor:
    """Build **only the output `y`** of `nn.GRU.forward` from a sequence of `gru_scan`.

    `[N, T, input_size]` → `[N, T, 2·hidden_size]` (the batch_first input and output as they are).

    MUST: do not return `h_n` — the `gru_scan` contract has one output (the `h` of every step) and
    does not bring back the final state (ADR 0056 decisions 1 and 7). A caller that consumes `h_n`
    cannot be expressed by this layer.
    """
    assert_supported(gru)
    hidden_size = gru.hidden_size
    # The initial state of every layer and direction is zero (the same as `nn.GRU` with `h_0`
    # omitted). It is a constant, so the conversion folds it and it lands in the IR as one
    # initializer.
    initial = torch.zeros(x.shape[0], hidden_size)
    sequence = x.transpose(0, 1)
    for layer in range(gru.num_layers):
        directions = []
        for suffix, scan in _DIRECTIONS:
            gates = nn.functional.linear(
                sequence,
                getattr(gru, f"weight_ih_l{layer}{suffix}"),
                getattr(gru, f"bias_ih_l{layer}{suffix}"),
            )
            directions.append(
                scan(
                    gates,
                    initial,
                    getattr(gru, f"weight_hh_l{layer}{suffix}"),
                    getattr(gru, f"bias_hh_l{layer}{suffix}"),
                )
            )
        # The next layer takes the concatenation of both directions (the same hand-off between
        # layers as `nn.GRU`).
        sequence = torch.cat(directions, dim=2)
    return sequence.transpose(0, 1)
