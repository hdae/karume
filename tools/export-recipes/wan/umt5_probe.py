"""Script that checks, on CPU with a small random umT5, the dynamic-shape export over the valid
length `L` and the assumptions of step 10c (ADR 0119 step 10c preliminary check — decisions 3 to
5). No real weights are read (the configuration is hand-written in {@link TINY_CONFIG}).

    uv run --group wan --inexact python -m wan.umt5_probe   # measure and print JSON

What it checks (each one a JSON field — pytest `wan/tests/test_umt5_patch.py` pins them with the
same functions):

- `export`: {@link wan.umt5_patch.Umt5EncoderTokens} goes through the exporter (karume) and passes
  the i8 storage plan and the checks `publish_model` applies before writing
  (`assert_runtime_support` / `assert_op_contracts`). Returns the IR op counts, the inputs (i32
  token ids `[1, L]` and bucket table `[L, L]`), the output `[1, L, d_model]`, the storage of the
  relative-position tables (F32) and the shapes of the golden (the wrapper's CPU f32 output with
  the weights after i8 fake-quant). Writing the container file and reading it back is done by
  pytest (the reason is in {@link export_probe}).
- `export.emit_without_overrides`: excluding the relative-position tables from the i8 targets
  without also passing F32 explicitly makes emit fail (decision 5's "form of the instruction that
  keeps them F32").
- `quant_targets`: whether the exporter's default i8 targets include the relative-position tables
  (`nn.Embedding [32, heads]`).
- `activation_rounding`: the per-element difference between the hand-written `gelu_new` formula
  (upstream) and `nn.GELU(approximate="tanh")` (the replacement). For reference, the formula with
  `pow(x, 3)` rewritten as `x·x·x` (a form written only with mul / add / tanh from the exporter's
  vocabulary) is listed alongside.
- `wrapper`: the wrapper with the upstream activation is bit-exact with the upstream forward (the
  spelled-out code is equivalent), and the difference between the replaced wrapper and upstream
  (the activation rounding only).
- `mask`: agreement of "valid length only" and "padded to 512 and masked" (the upstream
  `_get_t5_prompt_embeds` way of calling) in f32 and bf16 (reproducing research §2.3 — agreement
  on the real model is the step 10c acceptance).

MUST: transformers is imported inside functions (the same reason as `wan/umt5_patch.py`).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from karume.container import Provenance
from karume.convert import normalize_boundary_tensor
from karume.emit import EmitError, storage_breakdown, stored_model
from karume.ir import IrGraph
from karume.pipeline import export_module
from karume.verify import assert_op_contracts, assert_runtime_support
from wan import umt5_patch

#: Configuration of the small umT5. The **structural fields** match the real model
#: (`text_encoder/config.json`: gated `gelu_new`, 32 buckets, max_distance 128, eps 1e-6,
#: pad 0 / eos 1); only the sizes shrink. heads and d_kv differ (so that mixing up the axes of
#: `[L, L, heads]` → `[1, heads, L, L]` fails on the shape).
TINY_CONFIG: Mapping[str, Any] = {
    "vocab_size": 384,
    "d_model": 64,
    "d_kv": 16,
    "num_heads": 4,
    "d_ff": 160,
    "num_layers": 4,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "feed_forward_proj": "gated-gelu",
    "layer_norm_epsilon": 1e-6,
    "dropout_rate": 0.0,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "is_encoder_decoder": False,
}

#: Seed of the random initialization (weights, token ids).
SEED = 20261003

#: Valid lengths compared: the lower bound 2, the lengths of the four fixed prompts
#: (28 / 118 / 50 / 126), a length beyond max_distance 128, and the upper bound and the one just
#: below it (decision 4).
LENGTHS: tuple[int, ...] = (2, 28, 50, 118, 126, 300, 511, 512)

#: Valid lengths the container and golden are taken at (a small one and a large one — the golden
#: shows the same container runs at two values of L).
EXPORT_LENGTHS: tuple[int, ...] = (28, 126)

#: Graph name of the preliminary container pytest writes (a candidate for the distribution's
#: component name — decided in step 10d).
TARGET = "text_encoder"
MODEL_FILE = "model.krm"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."
IO_PREFIX = "io."
CASE_SUFFIX = ".safetensors"

#: Sample count for the activation comparison (standard normal, per scale) and the scales (umT5's
#: FFN pre-activations are O(1) to O(10)).
ACTIVATION_SAMPLES = 1_000_000
ACTIVATION_SCALES: tuple[float, ...] = (0.1, 1.0, 3.0, 10.0)

#: Lower bound of the reference's absolute value for counting ULP differences
#: ({@link _elementwise}).
ULP_FLOOR = 2.0**-8


def tiny_model(seed: int = SEED) -> nn.Module:
    """`UMT5EncoderModel` of {@link TINY_CONFIG} (upstream random initialization, eval, f32)."""
    from transformers import UMT5Config, UMT5EncoderModel

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return UMT5EncoderModel(UMT5Config(**TINY_CONFIG)).eval()


def tiny_ids(length: int, seed: int = SEED) -> torch.Tensor:
    """Token ids `[1, L]` of valid length `length` (random non-special tokens, ending in `</s>`)."""
    generator = torch.Generator().manual_seed(seed + length)
    body = torch.randint(
        3, int(TINY_CONFIG["vocab_size"]), (1, length - 1), generator=generator, dtype=torch.long
    )
    eos = torch.full((1, 1), int(TINY_CONFIG["eos_token_id"]), dtype=torch.long)
    return torch.cat([body, eos], dim=1)


def case_inputs(wrapper: umt5_patch.Umt5EncoderTokens, length: int) -> tuple[torch.Tensor, ...]:
    """The wrapper's inputs (token ids and the bucket table — built with layer 0's upstream
    formula)."""
    attention = wrapper.encoder.block[0].layer[0].SelfAttention
    return tiny_ids(length), umt5_patch.relative_position_buckets(length, attention)


def activation_rounding(seed: int = SEED) -> dict[str, Any]:
    """Per-element differences between upstream `gelu_new` and the two rewrites (same f32 input)."""
    from transformers.activations import NewGELUActivation

    generator = torch.Generator().manual_seed(seed)
    inputs = torch.cat(
        [
            torch.randn(ACTIVATION_SAMPLES, generator=generator) * scale
            for scale in ACTIVATION_SCALES
        ]
    )
    upstream = NewGELUActivation()(inputs)
    tanh = nn.GELU(approximate="tanh")(inputs)
    inner = math.sqrt(2.0 / math.pi) * (inputs + 0.044715 * (inputs * inputs * inputs))
    multiplied = 0.5 * inputs * (1.0 + torch.tanh(inner))
    return {
        "samples": int(inputs.numel()),
        "pow3_equals_mul": bool(torch.equal(torch.pow(inputs, 3.0), inputs * inputs * inputs)),
        "tanh": _elementwise(tanh, upstream),
        "mul": _elementwise(multiplied, upstream),
    }


def _elementwise(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """Fraction of differing elements, max absolute difference and max ULP difference (measured in
    ULPs of the reference value).

    The ULP difference only looks at elements whose reference absolute value is at least
    {@link ULP_FLOOR} — near 0 (where GELU is around −1e-20 for large negative inputs) the ULP is
    extremely small, and even an absolute difference around 1e-9 becomes 1e7 ULPs, which says
    nothing about the size of the rounding.
    """
    differing = actual != expected
    magnitude = expected.abs()
    spacing = (torch.nextafter(magnitude, torch.tensor(float("inf"))) - magnitude).double()
    ulps = (actual.double() - expected.double()).abs() / spacing
    return {
        "differing": float(differing.double().mean()),
        "max_abs": float((actual - expected).abs().max()),
        "max_ulp": float(ulps[magnitude >= ULP_FLOOR].max()),
    }


def wrapper_comparison(model: nn.Module, lengths: Sequence[int] = LENGTHS) -> list[dict[str, Any]]:
    """Agreement with the upstream forward of the wrapper with the upstream activation and of the
    wrapper with the replaced one."""
    exact = umt5_patch.Umt5EncoderTokens(model, activation="upstream")
    tanh = umt5_patch.Umt5EncoderTokens(model, activation="tanh")
    rows = []
    for length in lengths:
        inputs = case_inputs(exact, length)
        reference = umt5_patch.valid_output(model, inputs[0])
        with torch.no_grad():
            rows.append(
                {
                    "length": length,
                    "upstream_activation": dict(umt5_patch.compare(exact(*inputs), reference)),
                    "tanh_activation": dict(umt5_patch.compare(tanh(*inputs), reference)),
                }
            )
    return rows


def mask_equivalence(
    model: nn.Module, dtype: torch.dtype, lengths: Sequence[int] = LENGTHS
) -> list[dict[str, Any]]:
    """Agreement of "valid length only" and "512 + mask" (upstream forward against itself — run in
    `dtype`)."""
    cast = copy.deepcopy(model).to(dtype)
    rows = []
    for length in lengths:
        ids = tiny_ids(length)
        valid = umt5_patch.valid_output(cast, ids)
        padded = umt5_patch.padded_output(cast, ids)
        rows.append({"length": length, **umt5_patch.compare(valid, padded)})
    return rows


def quant_targets(model: nn.Module) -> dict[str, Any]:
    """The exporter's default i8 targets and the targets without the relative-position tables (both
    as counts after rounding)."""
    from karume.quantize import fake_quant_int8

    default = fake_quant_int8(umt5_patch.Umt5EncoderTokens(copy.deepcopy(model)))
    wrapper = umt5_patch.Umt5EncoderTokens(copy.deepcopy(model))
    tables = umt5_patch.relative_bias_modules(wrapper)
    selected = umt5_patch.fake_quant_i8(wrapper)
    return {
        "relative_bias_tables": len(tables),
        "default": report_targets(default.scales),
        "default_includes_relative_bias": all(
            f"{name}.weight" in default.scales for name in tables
        ),
        "selected": report_targets(selected.scales),
        "selected_includes_relative_bias": any(
            f"{name}.weight" in selected.scales for name in tables
        ),
    }


def report_targets(scales: Mapping[str, torch.Tensor]) -> dict[str, int]:
    """Count the rounded weights per module kind (the last name of the FQN)."""
    return dict(sorted(Counter(key.rsplit(".", 2)[-2] for key in scales).items()))


def provenance() -> Provenance:
    """Provenance of the preliminary container (random weights, not distributed — records the
    version of the transformers UMT5 implementation)."""
    import transformers

    return Provenance(
        license="apache-2.0", writer=f"wan.umt5_probe (transformers {transformers.__version__})"
    )


@dataclass(frozen=True)
class ProbeExport:
    """Result of the preliminary export (no container file is written — {@link export_probe}
    below)."""

    #: The raw graph and stored tensors of `export_module` (in the form `publish_model` takes as
    #: is).
    graph: IrGraph
    tensors: Mapping[str, torch.Tensor]
    #: The graph with the storage declarations committed (passed through the same two checks as
    #: `publish_model`).
    stored: IrGraph
    #: The i8 scale ledger and the explicit F32 for the relative-position tables (arguments of
    #: `publish_model`).
    scales: Mapping[str, torch.Tensor]
    overrides: Mapping[str, str]
    #: golden (file name `io.lNNN.safetensors` → graph inputs `input.*` and output `output.0`).
    golden: Mapping[str, Mapping[str, torch.Tensor]]
    #: The text of emit's rejection when the explicit F32 is not passed.
    emit_without_overrides: str


def export_probe(model: nn.Module) -> ProbeExport:
    """Run through to i8 storage (the relative-position tables in F32) and take the golden (no
    container file is written).

    `model` is copied before rounding (the caller's upstream weights are not changed). The golden is
    the CPU f32 output of the wrapper after rounding (ADR 0026 decision 2 / 0119 decision 8 — the
    reference is taken with the same fake-quant weights). The checks `publish_model` applies before
    writing (`assert_runtime_support` / `assert_op_contracts`) are applied here to the same target
    (the graph with the storage declarations committed).

    The code that writes the container (`publish_model` with `graph_name=`) lives only in pytest's
    scratch area: when a recipe script passes `graph_name=`, the cross-family check
    (`tests/test_graph_names.py`) requires it to match a component name of a distribution (a key
    of `wan.distribution.WAN_WEIGHTS`). The preliminary container has random weights and is not
    distributed, so it has no component name to use (umT5's component name and the distribution's
    layout are step 10d — ADR 0119 ruling 1).
    """
    wrapper = umt5_patch.Umt5EncoderTokens(copy.deepcopy(model))
    report = umt5_patch.fake_quant_i8(wrapper)
    overrides = umt5_patch.storage_overrides(wrapper)
    cases = {length: case_inputs(wrapper, length) for length in EXPORT_LENGTHS}
    graph, tensors = export_module(
        wrapper,
        cases[EXPORT_LENGTHS[0]],
        dynamic_shapes=umt5_patch.dynamic_shapes(),
        symbol_names=(umt5_patch.SYMBOL,),
    )
    try:
        stored_model(graph, tensors, weight_dtype="i8", weight_scales=report.scales)
    except EmitError as error:
        without_overrides = str(error)
    else:
        raise AssertionError(
            "emit passed although the relative-position tables have neither a scale nor an"
            " explicit F32"
        )
    stored = stored_model(
        graph,
        tensors,
        weight_dtype="i8",
        weight_scales=report.scales,
        weight_dtype_overrides=overrides,
    ).graph
    assert_runtime_support(stored)
    assert_op_contracts(stored)
    declared = [entry.name for entry in stored.inputs]
    if declared != list(umt5_patch.INPUT_NAMES):
        raise AssertionError(f"graph input names do not match the declaration: {declared}")
    golden: dict[str, dict[str, torch.Tensor]] = {}
    for length, inputs in cases.items():
        with torch.no_grad():
            output = wrapper(*inputs)
        io = {
            f"{INPUT_PREFIX}{name}": normalize_boundary_tensor(value, f"L={length} input '{name}'")
            for name, value in zip(umt5_patch.INPUT_NAMES, inputs, strict=True)
        }
        io[f"{OUTPUT_PREFIX}0"] = normalize_boundary_tensor(
            output.contiguous(), f"L={length} output"
        )
        golden[f"{IO_PREFIX}l{length:03d}{CASE_SUFFIX}"] = io
    return ProbeExport(
        graph=graph,
        tensors=tensors,
        stored=stored,
        scales=report.scales,
        overrides=overrides,
        golden=golden,
        emit_without_overrides=without_overrides,
    )


def summarize(export: ProbeExport) -> dict[str, Any]:
    """Summary of the export (op counts, symbols, inputs and outputs, storage, golden shapes)."""
    stored = export.stored
    breakdown = storage_breakdown(stored)
    return {
        "ops": dict(sorted(Counter(node.op for node in stored.nodes).items())),
        "symbols": list(stored.symbols),
        "inputs": [[entry.name, list(entry.shape), entry.dtype] for entry in stored.inputs],
        "outputs": [list(stored.values[name].shape) for name in stored.outputs],
        "storage": storage_by_kind(stored, export.overrides),
        "compressed_tensors": breakdown.compressed_tensors,
        "plain_tensors": breakdown.plain_tensors,
        "emit_without_overrides": export.emit_without_overrides,
        "golden": {
            name: {key: list(value.shape) for key, value in io.items()}
            for name, io in export.golden.items()
        },
    }


def storage_by_kind(graph: Any, tables: Mapping[str, str]) -> dict[str, list[str]]:
    """The set of storage dtypes per weight kind (relative-position tables, token embedding,
    linear, other)."""
    kinds: dict[str, set[str]] = {}
    for initializer in graph.initializers.values():
        key = initializer.tensor or ""
        if key in tables:
            kind = umt5_patch.RELATIVE_BIAS_ATTRIBUTE
        elif key == "encoder.embed_tokens.weight":
            kind = "embed_tokens"
        elif (
            key.endswith(".weight") and ".layer_norm." not in key and "final_layer_norm" not in key
        ):
            kind = "linear"
        else:
            kind = "other"
        kinds.setdefault(kind, set()).add(initializer.storage.dtype)
    return {kind: sorted(dtypes) for kind, dtypes in sorted(kinds.items())}


def run() -> dict[str, Any]:
    """All the measurements (the module docstring gives the meaning of the JSON fields)."""
    model = tiny_model()
    return {
        "config": dict(TINY_CONFIG),
        "export": summarize(export_probe(model)),
        "quant_targets": quant_targets(model),
        "activation_rounding": activation_rounding(),
        "wrapper": wrapper_comparison(model),
        "mask": {
            "f32": mask_equivalence(model, torch.float32),
            "bf16": mask_equivalence(model, torch.bfloat16),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.parse_args(argv)
    started = time.perf_counter()
    summary = run()
    summary["seconds"] = round(time.perf_counter() - started, 1)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
