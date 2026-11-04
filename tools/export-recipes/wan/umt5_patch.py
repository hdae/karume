"""The wrapper that turns Wan2.1 umT5 (`UMT5EncoderModel`) into an IR with a single symbol for the
valid length `L`, and the Python-side source of truth for the relative-position bucket table
(ADR 0119 decisions 3 to 5, step 10c).

The graph inputs are the token ids `[1, L]` and the bucket table `[L, L]` ({@link INPUT_NAMES});
the output is `[1, L, d_model]`. There is no mask input (decision 4 — it runs on the valid length
only).

## Differences from upstream (all inside {@link Umt5EncoderTokens})

- Relative-position buckets: upstream, `compute_bias` builds them inside the graph with `arange` →
  difference → `log` → `where`. The wrapper takes a host-built table as an input and looks up the
  per-layer table `[32, heads]` with `embedding` (decision 3).
- Mask: upstream adds an additive mask that is 1 over the valid length (`(1 − m)·finfo.min`) to the
  bias of every layer. The wrapper has none (decision 4).
- FFN activation: upstream uses `gelu_new` (the hand-written formula
  `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`). The wrapper uses `nn.GELU(approximate="tanh")`
  (`aten.gelu` with the same formula — see the section below).
- dropout: upstream it is the identity in eval. The wrapper has none ({@link Umt5EncoderTokens}
  requires eval).

The remaining parts (token embedding, q/k/v/o, RMSNorm, softmax taken in f32, the gated FFN, the
final RMSNorm) either call the upstream modules as they are or spell out the upstream lines in the
same order. With the activation left as upstream (`activation="upstream"`), the wrapper is
**bit-exact** with the upstream forward in f32 (pinned by pytest — so that a mistake in the spelled-
out code is told apart from the rounding of the activation replacement).

## Why `gelu_new` is replaced (measured in the step 10c preliminary check)

The exporter cannot take the `pow(x, 3.0)` in `gelu_new` (`aten.pow.Tensor_Scalar` ends up in the
list of unsupported ops — the `_pow2_to_mul` normalization handles exponent 2 only).
`nn.GELU(approximate="tanh")` writes the mathematically same function (the tanh-approximated GELU)
as a single `aten.gelu`, which lowers to the IR's `gelu_tanh`. The only difference is f32 rounding;
the per-element measurement is `activation_rounding` in {@link wan.umt5_probe}.

## The bucket table (decision 3)

{@link relative_position_buckets} calls upstream `UMT5Attention._relative_position_bucket` as is
(the formula is not copied). The table's orientation is the same as upstream `compute_bias`:
`table[i][j] = bucket(j − i)` (key position − query position). The TS generator
(`packages/models/src/umt5/relative-position.ts`) is byte-identical to this table (the fixture is
{@link wan.umt5_host_fixture}).

## What is quantized (decision 5)

The default targets of `karume.quantize.fake_quant_int8` include `nn.Embedding`, so the per-layer
relative-position tables (`relative_attention_bias` — `nn.Embedding [32, heads]`) would become i8
too. {@link fake_quant_i8} excludes these tables through include, and emit is told F32 explicitly
through {@link storage_overrides} (the tables go into the weight slot of `embedding`, so they are in
emit's eligible set, and without a scale emit fails with `has no per-channel scale`).

MUST: transformers is imported inside functions (the `wan` group is not in the default sync —
`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import torch
from torch import nn
from torch.export import Dim
from torch.nn import functional

from karume.quantize import QUANT_MODULE_TYPES, Int8Report, fake_quant_int8, iter_quant_targets
from wan.umt5_tokenizer import MAX_LENGTH, MIN_TOKENS

#: Names of the graph inputs (in the order of the forward arguments — the same spelling as TS
#: `src/umt5/session-io.ts`).
INPUT_NAMES = ("input_ids", "relative_position_buckets")

#: Name of the symbolic dimension for the valid length (decision 4 — a single symbol).
SYMBOL = "L"

#: The upstream FFN activation (`config.dense_act_fn`). The wrapper handles no other configuration.
UPSTREAM_ACTIVATION = "gelu_new"

#: The wrapper's activation choice (`tanh` = the export form, `upstream` = the form that checks the
#: spelled-out code is equivalent).
Activation = Literal["tanh", "upstream"]

#: Trailing part of the relative-position table's module name (the attribute name in upstream
#: `UMT5Attention`).
RELATIVE_BIAS_ATTRIBUTE = "relative_attention_bias"


class Umt5ConfigError(ValueError):
    """The upstream configuration differs from what the wrapper assumes (encoder, gated `gelu_new`,
    per-layer tables)."""


def check_supported(config: Any) -> None:
    """Check that the configuration is one the wrapper's spelled-out code holds for (it never
    silently computes something else when it does not)."""
    problems = []
    if config.is_decoder:
        problems.append(
            "is_decoder is true (only bidirectional buckets and self-attention are spelled out)"
        )
    if not config.is_gated_act:
        problems.append("the FFN is not gated (only the `wi_0`, `wi_1`, `wo` form is spelled out)")
    if config.dense_act_fn != UPSTREAM_ACTIVATION:
        problems.append(
            f"the activation is {config.dense_act_fn!r} (only {UPSTREAM_ACTIVATION!r} is replaced)"
        )
    if problems:
        raise Umt5ConfigError("; ".join(problems))


def bucket_attention(config: Any) -> nn.Module:
    """An upstream `UMT5Attention` that only carries the bucket formula (weights on meta — no
    storage is allocated).

    The entry point for calling the upstream formula without loading the real model (fixture
    generation — {@link wan.umt5_host_fixture}). The q/k/v/o Linears are built on meta, so even the
    full-size configuration (4,096 × 4,096 × 4) takes no memory.
    """
    from transformers.models.umt5.modeling_umt5 import UMT5Attention

    check_supported(config)
    with torch.device("meta"):
        return UMT5Attention(config, has_relative_attention_bias=False)


def relative_position_buckets(length: int, attention: nn.Module) -> torch.Tensor:
    """The bucket table `[L, L]` for valid length `length` (int64 — exactly the intermediate in
    upstream `compute_bias`).

    `attention` is an upstream `UMT5Attention` (a layer of the real model or
    {@link bucket_attention}). The same lines as the first half of upstream `compute_bias`
    (position difference → `_relative_position_bucket`); the table's orientation is
    `table[i][j] = bucket(j − i)`. Looking up a layer's table with `embedding` and permuting to
    `[heads, L, L]` is bit-exact with upstream `compute_bias` (pytest).

    The accepted lengths are 2 to 512, the same as the graph's symbolic dimension (decision 4).
    The TS generator was checked against this range only (f64 `Math.log` matching torch's f32 `log`
    was measured on every distance up to ±511 — research §4.3). The caller applies the i32
    conversion at the boundary with `karume.convert.normalize_boundary_tensor` (ADR 0009).
    """
    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError(f"pass the valid length as an int ({type(length).__name__})")
    if not MIN_TOKENS <= length <= MAX_LENGTH:
        raise ValueError(
            f"valid length {length} is outside {MIN_TOKENS} to {MAX_LENGTH} (decision 4)"
        )
    if attention.is_decoder:
        raise Umt5ConfigError(
            "decoder (unidirectional) buckets are not handled (encoder bidirectional only)"
        )
    context_position = torch.arange(length, dtype=torch.long)[:, None]
    memory_position = torch.arange(length, dtype=torch.long)[None, :]
    return attention._relative_position_bucket(memory_position - context_position)


class Umt5EncoderTokens(nn.Module):
    """Turn `UMT5EncoderModel` into "token ids `[1, L]` + bucket table `[L, L]` →
    `[1, L, d_model]`".

    The upstream modules are shared (rounding the weights — fake-quant — through the wrapper reaches
    upstream too). `self.encoder` holds only the upstream `UMT5Stack` so that the tied `shared`
    stays out of the name space (the weight key is then only `encoder.embed_tokens.weight`).

    MUST: wrap an upstream in eval only (the wrapper has no dropout, so it would compute something
    other than an upstream in training mode).
    MUST: f32 weights (the dtype cast before `wo` in the upstream FFN is not spelled out).
    """

    def __init__(self, model: nn.Module, *, activation: Activation = "tanh") -> None:
        super().__init__()
        check_supported(model.config)
        if model.training:
            raise Umt5ConfigError(
                "an upstream in training mode is not wrapped (dropout is not the identity)"
                " — pass it after eval()"
            )
        dtypes = {parameter.dtype for parameter in model.parameters()}
        if dtypes != {torch.float32}:
            raise Umt5ConfigError(f"weight dtypes {sorted(map(str, dtypes))} — only f32 is wrapped")
        if activation not in ("tanh", "upstream"):
            raise ValueError(f"activation {activation!r} must be tanh or upstream")
        self.encoder = model.encoder
        self.activation = activation
        self.gelu = nn.GELU(approximate="tanh")

    def forward(
        self, input_ids: torch.Tensor, relative_position_buckets: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.encoder.embed_tokens(input_ids)
        for block in self.encoder.block:
            hidden = self._self_attention(block.layer[0], hidden, relative_position_buckets)
            hidden = self._feed_forward(block.layer[-1], hidden)
        return self.encoder.final_layer_norm(hidden)

    @staticmethod
    def _self_attention(
        layer: nn.Module, hidden: torch.Tensor, buckets: torch.Tensor
    ) -> torch.Tensor:
        """`UMT5LayerSelfAttention` + `UMT5Attention.forward` spelled out (no cache, no mask).

        No scale on the scores, and softmax taken in f32 and cast back to the original dtype — both
        as in the upstream lines. Only the bias is the second half of `compute_bias` (gather from
        the table → `[1, heads, L, L]`), with the table taken from the input.
        """
        attention = layer.SelfAttention
        normed = layer.layer_norm(hidden)
        batch_size, seq_length = normed.shape[:2]
        heads, width = attention.n_heads, attention.key_value_proj_dim
        query = attention.q(normed).view(batch_size, -1, heads, width).transpose(1, 2)
        key = attention.k(normed).view(batch_size, -1, heads, width).transpose(1, 2)
        value = attention.v(normed).view(batch_size, -1, heads, width).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(3, 2))
        bias = attention.relative_attention_bias(buckets).permute([2, 0, 1]).unsqueeze(0)
        scores = scores + bias
        weights = functional.softmax(scores.float(), dim=-1).type_as(scores)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_length, -1)
        return hidden + attention.o(output)

    def _feed_forward(self, layer: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        """`UMT5LayerFF` + `UMT5DenseGatedActDense.forward` spelled out (only the activation is
        selectable)."""
        dense = layer.DenseReluDense
        normed = layer.layer_norm(hidden)
        activation = self.gelu if self.activation == "tanh" else dense.act
        gated = activation(dense.wi_0(normed)) * dense.wi_1(normed)
        return hidden + dense.wo(gated)


def dynamic_shapes() -> tuple[dict[int, Any], dict[int, Any]]:
    """The symbolic dimensions per input. Both axes of the bucket table are declared with the same
    `L` as the token ids (separate symbols would accept a table whose length differs from the body's
    and silently give wrong values — the same reason as the DiT's RoPE tables)."""
    length = Dim(SYMBOL, min=MIN_TOKENS, max=MAX_LENGTH)
    return ({1: length}, {0: length, 1: length})


def relative_bias_modules(wrapper: nn.Module) -> tuple[str, ...]:
    """Module FQNs of the per-layer relative-position tables (`nn.Embedding [num_buckets, heads]`).

    MUST: the count matches the layer count (every umT5 layer has its own table — research §1.1).
    If any are missing, a table that failed to be excluded from quantization is written as i8.
    """
    names = tuple(
        name
        for name, module in wrapper.named_modules()
        if name.rsplit(".", 1)[-1] == RELATIVE_BIAS_ATTRIBUTE and isinstance(module, nn.Embedding)
    )
    layers = len(wrapper.encoder.block)
    if len(names) != layers:
        raise Umt5ConfigError(
            f"{len(names)} relative-position tables differ from the layer count {layers}"
        )
    return names


def quant_include(wrapper: nn.Module) -> Callable[[str], bool]:
    """Predicate for the i8 targets (module FQN — excludes only the relative-position tables)."""
    tables = frozenset(relative_bias_modules(wrapper))
    return lambda name: name not in tables


def storage_overrides(wrapper: nn.Module) -> dict[str, str]:
    """Explicit storage passed to emit (tensor key → `"f32"` — the relative-position tables are
    written as F32).

    The tables are consumed only by the weight slot of `embedding`, so they are in emit's eligible
    set. Left at the default `i8` without a scale, emit fails loudly, so F32 is stated here
    (decision 5).
    """
    return {f"{name}.weight": "f32" for name in relative_bias_modules(wrapper)}


def fake_quant_i8(wrapper: nn.Module) -> Int8Report:
    """Round the linear and token-embedding weights to values representable in per-channel
    symmetric i8 (decision 5).

    The relative-position tables are not rounded (they stay F32 — decision 5). MUST: check that the
    number of rounded weights equals "every weight of a quantizable type − the relative-position
    tables" and that the table values did not move by a single bit — if a table were not excluded,
    both the golden and the storage would hold the i8 table and no check would see it.
    """
    include = quant_include(wrapper)
    tables = relative_bias_modules(wrapper)
    modules = dict(wrapper.named_modules())
    before = {name: modules[name].weight.detach().clone() for name in tables}
    expected = sorted(
        fqn
        for fqn, _, _ in iter_quant_targets(wrapper, op_types=QUANT_MODULE_TYPES, include=include)
    )
    report = fake_quant_int8(wrapper, include=include)
    if sorted(report.scales) != expected:
        raise AssertionError(
            f"the weights rounded to i8 differ from the expected ones: {sorted(report.scales)}"
            f" vs {expected}"
        )
    moved = [name for name in tables if not torch.equal(modules[name].weight, before[name])]
    if moved:
        raise AssertionError(f"relative-position tables were rounded: {moved}")
    return report


def padded_output(
    model: nn.Module, input_ids: torch.Tensor, max_length: int = MAX_LENGTH
) -> torch.Tensor:
    """The output of the upstream way of calling (`_get_t5_prompt_embeds` — padded with the pad id
    to 512 and masked), cut to the valid length.

    `model` is an upstream `UMT5EncoderModel` (any dtype — also used for the bf16 comparison). The
    mask is int64 0 / 1, the same as the upstream tokenizer's.
    """
    length = int(input_ids.shape[1])
    if input_ids.shape[0] != 1 or not 0 < length <= max_length:
        raise ValueError(f"pass the token ids as [1, 1 to {max_length}] ({tuple(input_ids.shape)})")
    padded = torch.full((1, max_length), model.config.pad_token_id, dtype=input_ids.dtype)
    padded[:, :length] = input_ids
    mask = torch.zeros((1, max_length), dtype=torch.long)
    mask[:, :length] = 1
    with torch.no_grad():
        output = model(input_ids=padded, attention_mask=mask).last_hidden_state
    return output[:, :length]


def valid_output(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """The upstream output run on the valid length only (no mask passed — upstream builds an
    all-ones mask itself)."""
    with torch.no_grad():
        return model(input_ids=input_ids).last_hidden_state


def max_ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Max absolute difference ÷ max absolute value of the reference (in f64 — the same form as the
    band metric of `wan.export_dit`)."""
    expected = expected.double()
    return float((actual.double() - expected).abs().max() / expected.abs().max())


def compare(actual: torch.Tensor, expected: torch.Tensor) -> Mapping[str, Any]:
    """How closely two outputs agree (bit-exact, ratio, max absolute difference, fraction of
    differing elements)."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(
            f"shape or dtype differs: {tuple(actual.shape)} {actual.dtype} vs "
            f"{tuple(expected.shape)} {expected.dtype}"
        )
    return {
        "bit_exact": bool(torch.equal(actual, expected)),
        "ratio": max_ratio(actual, expected),
        "max_abs": float((actual.double() - expected.double()).abs().max()),
        "differing": float((actual != expected).double().mean()),
    }
