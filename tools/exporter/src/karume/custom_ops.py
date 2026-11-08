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

`karume::gated_delta_scan` (ADR 0135 decision 5) solves the same problem for the Gated DeltaNet
recurrence: both upstream functions loop in Python (over tokens, or over 64-token chunks with a
T-dependent padding), so `torch.export` specialises the length axis. Its eager body is a copy of
upstream's prefill function (the docstring of {@link gated_delta_scan}).

`karume::expert_linear` (ADR 0145 decision 5) is the routed expert projection of a
mixture-of-experts layer: upstream's experts loop over the experts hit in Python, and gathering
the picked weights as a tensor would materialise `[T, k, out, in]`. Its eager body is upstream's
per-expert math (the docstring of {@link expert_linear}).

NOTE: `torch.library.custom_op` is a process-wide global registration, so this import **has a side
effect** (the "zero side effects in every module" of `packages/*` is a TS-side tree-shaking
invariant and does not apply to the Python exporter). `aten_handlers.py` imports this module to
write the handler key (`torch.ops.karume.gru_scan.default`).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812  -- the spelling of the upstream copy below

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


#: The upstream version the gated delta rule body below is copied from (ADR 0135 decision 5).
GATED_DELTA_RULE_UPSTREAM = "transformers 5.14.1, models/qwen3_5/modeling_qwen3_5.py"


# --- copied from upstream (GATED_DELTA_RULE_UPSTREAM), token for token ---------------------------
# MUST: `l2norm` and `torch_chunk_gated_delta_rule` below are a **verbatim copy** of upstream's
# functions (ADR 0135 decision 5): the goldens are taken from this eager body, so a drift here turns
# green "the exporter and the runtime agree and are both wrong". The only differences are the
# leading underscore of the names and the line breaks and quotes `ruff format` imposes; the tokens
# are upstream's (tests/test_gated_delta_scan.py compares the output bit for bit with upstream's
# function when transformers is installed). It is a copy, not an import, because the exporter core
# depends only on torch, torchvision, numpy and safetensors (ADR 0065).
#
# Third-party code notice. `_l2norm` and `_torch_chunk_gated_delta_rule` are copied from `l2norm`
# and `torch_chunk_gated_delta_rule` in huggingface/transformers
# (`src/transformers/models/qwen3_5/modeling_qwen3_5.py`, transformers 5.14.1), licensed under the
# Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0); the changes are the
# ones listed above. Upstream copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
def _l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# --- end of the upstream copy ---------------------------------------------------------------------


@torch.library.custom_op("karume::gated_delta_scan", mutates_args=())
def gated_delta_scan(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """The gated delta rule scan, value form (ADR 0135): q[B,T,Hk,dk] / k[B,T,Hk,dk] (both
    l2-normalised) / v[B,T,Hv,dv] / g[B,T,Hv] / beta[B,T,Hv] -> o[B,T,Hv,dv], from a zero state.

    MUST: the body is upstream's prefill function `torch_chunk_gated_delta_rule` (the copy above),
    called as `Qwen3_5GatedDeltaNet.forward` calls it after its `repeat_interleave` of q / k to the
    value heads, with `use_qk_l2norm_in_kernel=False` (q / k arrive normalised),
    `initial_state=None` and `output_final_state=False` (ADR 0135 decision 5). The head repeat
    `r = Hv / Hk` is the op's derived broadcast (value head h reads key head floor(h / r)).
    """
    key_heads, value_heads = q.shape[2], v.shape[2]
    if key_heads < 1 or value_heads % key_heads != 0:
        raise ValueError(
            f"gated_delta_scan: Hv {value_heads} is not a positive multiple of Hk {key_heads}"
        )
    ratio = value_heads // key_heads
    query = q.repeat_interleave(ratio, dim=2) if ratio > 1 else q
    key = k.repeat_interleave(ratio, dim=2) if ratio > 1 else k
    out, _ = _torch_chunk_gated_delta_rule(
        query,
        key,
        v,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    )
    return out


@gated_delta_scan.register_fake
def _gated_delta_scan_fake(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Meta implementation: the output is shaped like `v` (T passes **as is**, so it stays
    symbolic)."""
    del q, k, g, beta
    return v.new_empty(v.shape)


#: The upstream version the expert_linear body below follows (ADR 0145 decision 5).
EXPERT_LINEAR_UPSTREAM = "transformers 5.14.1, models/gemma4/modeling_gemma4.py Gemma4TextExperts"


def expert_linear_body(
    x: torch.Tensor,
    index: torch.Tensor,
    weight_of: Callable[[int], torch.Tensor],
    experts: int,
    out_features: int,
) -> torch.Tensor:
    """The per-pick math of {@link expert_linear}, with the weight of expert `e` read through
    `weight_of(e)` (an `[out, in]` F32 tensor) — `[T, k, out]` for x `[T, in]` / `[T, k, in]` and
    index `[T, k]`.

    MUST: this is the one body of the op's eager form and of any caller that holds the experts in
    another form (a recipe's CPU reference over packed bytes dequantises one expert per call), so
    the trace's op and such a reference share one per-pick summation (ADR 0145 decision 5).
    `weight_of` is called once per expert the index hits, in ascending id order, and never for an
    expert it does not hit. MUST: an id outside `[0, experts)` raises (torch would wrap a negative
    one) — the CPU reference's rule (ADR 0061).
    """
    if index.numel() > 0 and (int(index.min()) < 0 or int(index.max()) >= experts):
        raise ValueError(
            f"expert_linear: index ids span [{int(index.min())}, {int(index.max())}],"
            f" outside the {experts} experts"
        )
    tokens, picks = index.shape
    out = x.new_zeros((tokens, picks, out_features))
    for expert in torch.unique(index).tolist():
        token, pick = torch.where(index == expert)
        rows = x[token] if x.dim() == 2 else x[token, pick]
        out[token, pick] = F.linear(rows, weight_of(expert))
    return out


@torch.library.custom_op("karume::expert_linear", mutates_args=())
def expert_linear(x: torch.Tensor, index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The routed expert linear (ADR 0145 decision 2): x[T,in] (shared by the k picks) or
    x[T,k,in] (one row per pick) / index[T,k] / weight[E,out,in] -> y[T,k,out], with
    `y[t, j] = weight[index[t, j]] · x[t (, j)]`.

    MUST: the body is upstream's eager experts math (`Gemma4TextExperts.forward`): for each expert
    the index hits, the rows routed to it go through `F.linear` with that expert's weight
    ({@link expert_linear_body} — the body a reference over another weight form shares). Only
    the picked experts are read (`weight[expert]`), so a weight that dequantises on read
    dequantises only those. The op stops at one projection: the activation, the gate × up product,
    the routing weight and the sum over k are the caller's ops (ADR 0145 decision 2).
    MUST: an id outside `[0, E)` raises (torch would wrap a negative one) — the CPU reference's
    rule (ADR 0061); in a graph the index is a topk over E (the runtime's plan-time rule).
    """
    return expert_linear_body(x, index, weight.__getitem__, weight.shape[0], weight.shape[1])


@expert_linear.register_fake
def _expert_linear_fake(x: torch.Tensor, index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Meta implementation: `[T, k, out]` with T and k from index (T stays symbolic).

    The ranks are not checked here: a malformed call still traces, and the handler refuses it by
    rank (`aten_handlers._h_expert_linear`) with the op named in the message.
    """
    del x
    out = weight.shape[-2] if weight.dim() >= 2 else 0
    return weight.new_empty((*index.shape, out))


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
