"""Custom ops in the `karume::` namespace (the layer torch.export keeps as one node — ADR 0056).

`torch.library.custom_op` hides the body behind `register_fake`, so tracing does not enter the
Python loop. The op therefore **stays a single node with its symbolic dimension (the time axis T)
intact**, and is not expanded by `run_decompositions(curated_decompositions())` either (`karume::`
is not in the Core ATen decomposition table — the same reason as `torchvision::deform_conv2d`).

The problem this solves: the decomposition of `aten.gru.input` is **unrolled T times** by a Python
loop, so `torch.export` specializes the length axis (`Specializations unexpectedly required (T)`).
In vowel-detector, T10=200 gives 8,434 nodes, and every length needs its own graph.

MUST: the body (the eager implementation) is written as **a verbatim copy of torch's GRU
decomposition**. The golden expectations are taken by running this implementation eagerly, so a
drift here turns green the state "the exporter and the runtime agree and are both wrong". In
particular the update is `(h − n)·z + n`, not `(1 − z)·n + z·h` (44,345 of 100,000 f32 elements
differ in bits — ADR 0056 decision 3).

NOTE: `torch.library.custom_op` is a process-wide global registration, so this import **has a side
effect** (the "zero side effects in every module" of `packages/*` is a TS-side tree-shaking
invariant and does not apply to the Python exporter). `aten_handlers.py` imports this module to
write the handler key (`torch.ops.karume.gru_scan.default`).
"""

from __future__ import annotations

import torch

#: Number of gates (r / z / n). The first axis of gi and w_hh / b_hh is a multiple of this.
GATE_COUNT = 3


def _scan(
    gi: torch.Tensor,
    h0: torch.Tensor,
    w_hh: torch.Tensor,
    b_hh: torch.Tensor,
    *,
    reverse: bool,
) -> torch.Tensor:
    """The hidden-side scan body (assumes the caller's `linear` has done the input-side GEMM).

    MUST: do not change the order or grouping of operations. Gate sums take **the hidden side as
    the first operand**, the product in `n` is `h_n · r` (reset multiplies **the hidden-side
    product including b_hh**), and the update is `(h − n)·z + n`.
    MUST: the reverse direction flips **only the scan order**, and writes at the forward time index
    (no `flip` on the output — the same contract as the runtime kernel).
    """
    length = gi.shape[0]
    hidden = h0.shape[1]
    order = range(length - 1, -1, -1) if reverse else range(length)
    state = h0
    steps: list[torch.Tensor] = [h0] * length
    for step in order:
        gh = torch.nn.functional.linear(state, w_hh, b_hh)
        input_r, input_z, input_n = gi[step].split(hidden, dim=1)
        hidden_r, hidden_z, hidden_n = gh.split(hidden, dim=1)
        reset = torch.sigmoid(hidden_r + input_r)
        update = torch.sigmoid(hidden_z + input_z)
        candidate = torch.tanh(input_n + hidden_n * reset)
        state = (state - candidate) * update + candidate
        steps[step] = state
    return torch.stack(steps, dim=0)


@torch.library.custom_op("karume::gru_scan", mutates_args=())
def gru_scan(
    gi: torch.Tensor, h0: torch.Tensor, w_hh: torch.Tensor, b_hh: torch.Tensor
) -> torch.Tensor:
    """Forward hidden-side scan: gi[T,N,3H] / h0[N,H] / w_hh[3H,H] / b_hh[3H] -> y[T,N,H]."""
    return _scan(gi, h0, w_hh, b_hh, reverse=False)


@torch.library.custom_op("karume::gru_scan_reverse", mutates_args=())
def gru_scan_reverse(
    gi: torch.Tensor, h0: torch.Tensor, w_hh: torch.Tensor, b_hh: torch.Tensor
) -> torch.Tensor:
    """Reverse hidden-side scan (**the output is in forward time order**)."""
    return _scan(gi, h0, w_hh, b_hh, reverse=True)


def _fake(
    gi: torch.Tensor, h0: torch.Tensor, w_hh: torch.Tensor, b_hh: torch.Tensor
) -> torch.Tensor:
    """Meta implementation: the time axis passes `gi.shape[0]` **as is** (so it stays symbolic)."""
    del w_hh, b_hh
    return gi.new_empty((gi.shape[0], gi.shape[1], h0.shape[1]))


gru_scan.register_fake(_fake)
gru_scan_reverse.register_fake(_fake)


@torch.library.custom_op("karume::static_quantize", mutates_args=())
def static_quantize(x: torch.Tensor, scale: float) -> torch.Tensor:
    """SRQ with a fixed f32 scale. The generic core does not depend on Transformers (ADR 0097)."""
    from karume.ops import assert_static_quantize_scale

    assert_static_quantize_scale(scale, "static_quantize")
    if x.dtype != torch.float32:
        raise ValueError("static_quantize supports f32 input only")
    if scale == 0:
        return x.clone()
    return torch.clamp(torch.round(x / scale), -128, 127) * scale


@static_quantize.register_fake
def _static_quantize_fake(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x.new_empty(x.shape)
