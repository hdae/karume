"""Generate the tiny golden fixtures (read downstream by the runtime's E2E test,
`packages/runtime/tests/e2e_golden_test.ts`).

Small fixed-seed models go through torch.export → container (`krm`), and the expected outputs on
torch CPU are placed in the same directory. Every op of the contract table is always covered by
the models combined (the COVERAGE check).

    uv run python -m karume.goldens --out ../../packages/runtime/tests/fixtures/golden

The destination is **always taken as an argument** (`--out` is required) — which repository's
`packages/runtime/tests/fixtures/` it lands in is repo topology, not knowledge of the generic
exporter (ADR 0065 Consequences). The invocation above uses paths relative to `tools/exporter/`.

The layout and tensor key naming are documented in README.md "Golden layout".

The model definitions (the tiny nn.Module classes) and the weight initialisation and input
generation helpers are {@link karume.golden_models} — this module is the ledger
({@link GOLDEN_SPECS}) and the generation driver.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from karume.artifacts import staged_publication
from karume.container import Provenance
from karume.convert import (
    PRESERVED_OP_PREFIXES,
    PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    normalize_boundary_tensor,
)

# Bindings of the models and input generation helpers moved out to the fixture side
# ({@link karume.golden_models}). The import goes one way only, goldens → golden_models (an edge
# the other way would let model definitions creep into the ledger).
from karume.golden_models import (
    Activations,
    ArgmaxPick,
    AttentionBlock,
    BatchMatmul,
    BilinearResize,
    BroadcastBinary,
    Conv2dBlock,
    Conv3dBlock,
    ConvBlock,
    ConvTransposeStack,
    CouplingSplit,
    DecoderTail,
    DeformConvBlock,
    DilatedConvStack,
    EmbeddingLookup,
    ExpandMask,
    FusedAttention,
    GatedDeltaScanBlock,
    GatherLastDim,
    GroupNormBlock,
    GruScanBlock,
    Int8Weights,
    IntCast,
    LayoutChain,
    MaskChain,
    MaskedScores,
    Mlp,
    RmsNormBlock,
    RowReduce,
    RuntimeMaskedAttention,
    ScalarOperands,
    SplinePieces,
    StaticQuantizeBlock,
    SymbolicTable,
    UnaryChain,
    _band_mask,
    _fused_attention_inputs,
    _mask,
    _runtime_masked_attention_inputs,
    _uniform,
    _unit_rows,
    _zeroed_column,
)
from karume.ir import IrGraph
from karume.ops import EMITTABLE_OPS
from karume.pipeline import export_to_file
from karume.quantize import fake_quant_int8

MODEL_FILE = "model.krm"
IO_FILE = "io.safetensors"

#: Provenance of the goldens. `writer` is **not written** — baking in a generator tag would move
#: the bytes of every golden on a mere package version bump, and break the premise that a
#: regeneration that changes bytes reads as "the asset changed".
GOLDEN_PROVENANCE = Provenance(license="mit")
#: Key convention of io.safetensors. Inputs are keyed by graph input name, outputs by position in
#: graph.outputs.
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: Every random number is drawn from here (an explicit torch.Generator — no dependence on the
#: global seed).
SEED = 20260802
#: The actual golden length of the symbolic dimension T. At least 2 to avoid torch.export's 0/1
#: specialisation.
GOLDEN_T = 6
_DYNAMIC_T = Dim("T", min=2, max=64)
#: A small upper bound for goldens that exercise Tmax folding (baked constants grow with Tmax² —
#: keeps the fixture under 16KiB). T = GOLDEN_T is well below it, so the prefix really is shorter.
_SMALL_DYNAMIC_T = Dim("T", min=2, max=24)


def _rng() -> torch.Generator:
    return torch.Generator().manual_seed(SEED)


@dataclass(frozen=True)
class GoldenSpec:
    name: str
    build: Callable[[torch.Generator], nn.Module]
    example_inputs: Callable[[torch.Generator], tuple[torch.Tensor, ...]]
    dynamic_shapes: Any = None
    symbol_names: Sequence[str] = field(default=("T",))
    #: Storage dtype of the weights. With `"i8"`, `generate_golden` applies fake-quant **before
    #: export and before capturing the reference** (ADR 0006 / 0019).
    weight_dtype: str = "f32"
    #: Set of prefixes of high-level ops whose decomposition is stopped (ADR 0023). Only the
    #: golden that preserves SDPA to emit a fused `attention` replaces the default — adding it to
    #: the default would make SDPA with a mask a preservation target too.
    preserved: Sequence[str] = field(default=PRESERVED_OP_PREFIXES)


GOLDEN_SPECS: tuple[GoldenSpec, ...] = (
    GoldenSpec(
        name="static_quantize_block",
        build=lambda g: StaticQuantizeBlock(),
        example_inputs=lambda g: (
            torch.tensor(
                [
                    [
                        -1000.0,
                        -128.5,
                        -127.5,
                        -2.5,
                        -1.5,
                        -0.5,
                        -0.0,
                        0.0,
                        0.5,
                        1.5,
                        2.5,
                        126.5,
                        127.5,
                        128.5,
                        1000.0,
                    ]
                ],
                dtype=torch.float32,
            ),
        ),
    ),
    GoldenSpec(
        name="unary_chain",
        build=lambda _: UnaryChain(),
        # Only positive values away from 0, to stay inside the domains of sqrt / log.
        example_inputs=lambda g: (_uniform(g, 3, 4, low=0.5, high=2.0),),
    ),
    GoldenSpec(
        name="activations",
        build=lambda _: Activations(),
        example_inputs=lambda g: (_uniform(g, 2, 5, low=-2.0, high=2.0),),
    ),
    GoldenSpec(
        name="broadcast_binary",
        build=BroadcastBinary,
        # The divisor y does not straddle 0 (so the div golden does not diverge).
        example_inputs=lambda g: (
            _uniform(g, GOLDEN_T, 4, low=-1.0, high=1.0),
            _uniform(g, 4, low=0.5, high=1.5),
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, None),
    ),
    GoldenSpec(
        name="mlp",
        build=Mlp,
        example_inputs=lambda g: (_uniform(g, GOLDEN_T, 8, low=-1.0, high=1.0),),
        dynamic_shapes=({0: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="row_reduce",
        build=lambda _: RowReduce(),
        example_inputs=lambda g: (_uniform(g, GOLDEN_T, 5, low=-2.0, high=2.0),),
        dynamic_shapes=({0: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="argmax_pick",
        build=ArgmaxPick,
        # x[1,T,HIDDEN] is symbolic only on the position axis (the real shape at the decode
        # exit). tied is **the seed that creates equal-value ties**, with the position of the
        # maximum changing per row (if every row were the same, a "reads another row" bug would
        # not show in the values).
        example_inputs=lambda g: (
            _uniform(g, 1, GOLDEN_T, ArgmaxPick.HIDDEN, low=-1.5, high=1.5),
            torch.tensor([[0.5, 2.0, -1.0, 0.25], [1.0, -0.5, 3.0, 0.75], [4.0, 1.5, 0.0, -2.0]]),
        ),
        dynamic_shapes=({1: _DYNAMIC_T}, None),
    ),
    GoldenSpec(
        name="mask_chain",
        build=lambda _: MaskChain(),
        # mask is torch's default i64 (normalised to i32 at the boundary). It always contains 0
        # so that both branches, bitwise_not and the conversion to truth values, are exercised.
        example_inputs=lambda g: (
            _mask(GOLDEN_T, 1, zero_at=2),
            _mask(1, GOLDEN_T, zero_at=5),
            _uniform(g, GOLDEN_T, GOLDEN_T, low=-2.0, high=2.0),
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, {1: _DYNAMIC_T}, {0: _DYNAMIC_T, 1: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="int_cast",
        build=lambda _: IntCast(),
        # Values on which truncation is distinguishable from round / floor (straddling ±.5 /
        # ±.7).
        example_inputs=lambda g: (
            _uniform(g, GOLDEN_T, 3, low=-3.0, high=3.0),
            torch.arange(GOLDEN_T * 3, dtype=torch.int64).reshape(GOLDEN_T, 3) % 4 - 1,
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, {0: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="layout_chain",
        build=lambda _: LayoutChain(),
        example_inputs=lambda g: (_uniform(g, GOLDEN_T, 3, 4, low=-1.0, high=1.0),),
        dynamic_shapes=({0: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="expand_mask",
        # mask / index are torch's default i64 (normalised to i32 at the boundary). mask always
        # contains 0 so that both branches of bitwise_not are exercised.
        example_inputs=lambda g: (
            _mask(GOLDEN_T, 1, zero_at=2),
            torch.arange(GOLDEN_T, dtype=torch.int64).reshape(1, GOLDEN_T) - 2,
            _uniform(g, 3, GOLDEN_T, low=-2.0, high=2.0),
        ),
        build=lambda _: ExpandMask(),
        dynamic_shapes=({0: _DYNAMIC_T}, {1: _DYNAMIC_T}, {1: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="batch_matmul",
        build=lambda _: BatchMatmul(),
        # B=2 / M=T / K=3 / N=5 — all different lengths (so a swapped axis shows in the numbers)
        example_inputs=lambda g: (
            _uniform(g, 2, GOLDEN_T, 3, low=-1.0, high=1.0),
            _uniform(g, 2, 3, 5, low=-1.0, high=1.0),
        ),
        dynamic_shapes=({1: _DYNAMIC_T}, None),
    ),
    GoldenSpec(
        name="gather_last_dim",
        build=lambda _: GatherLastDim(),
        # index is torch's default i64 (normalised to i32 at the boundary). 3i mod 4 visits all
        # 4 entries of src's last dimension in a sequence that is neither the identity nor
        # monotonic (an off-by-one index always shows).
        example_inputs=lambda g: (
            _uniform(g, 2, GOLDEN_T, 4, low=-2.0, high=2.0),
            (torch.arange(2 * GOLDEN_T * 3, dtype=torch.int64) * 3 % 4).reshape(2, GOLDEN_T, 3),
        ),
        dynamic_shapes=({1: _DYNAMIC_T}, {1: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="attention_block",
        build=AttentionBlock,
        # The second logits tensor lies where a naive softmax's exp collapses to 0 in f32
        # (-205..-180)
        example_inputs=lambda g: (
            _uniform(g, 1, GOLDEN_T, AttentionBlock.HIDDEN, low=-1.0, high=1.0),
            _uniform(g, 3, 5, low=-205.0, high=-180.0),
        ),
        dynamic_shapes=({1: _DYNAMIC_T}, None),
    ),
    GoldenSpec(
        name="fused_attention",
        build=lambda _: FusedAttention(),
        example_inputs=_fused_attention_inputs,
        # Only this golden preserves SDPA (adding it to the default 13 ops of
        # `PRESERVED_OP_PREFIXES` would make graphs with masked SDPA unexportable — ADR 0023).
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    ),
    GoldenSpec(
        name="embedding_lookup",
        build=EmbeddingLookup,
        # Indices are torch's default i64 (normalised to i32 at the boundary). The sequence
        # always looks up padding_idx=0 and cycles through the 5 vocabulary rows.
        example_inputs=lambda _: (
            (torch.arange(GOLDEN_T, dtype=torch.int64) % EmbeddingLookup.VOCAB).reshape(
                1, GOLDEN_T
            ),
        ),
        dynamic_shapes=({1: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="masked_scores",
        build=lambda _: MaskedScores(),
        # mask is torch's default i64 (cast to bool at the boundary). **One row is fully
        # masked**, so the softmax of a row whose elements are all -3.4e38 (a uniform
        # distribution) is exercised too.
        example_inputs=lambda g: (
            _uniform(g, 1, 2, GOLDEN_T, GOLDEN_T, low=-2.0, high=2.0),
            _band_mask(GOLDEN_T, blocked_row=2),
            _uniform(g, 1, GOLDEN_T, 4, low=-1.0, high=1.0),
            _mask(1, GOLDEN_T, 4, zero_at=5),
        ),
        dynamic_shapes=(
            {2: _DYNAMIC_T, 3: _DYNAMIC_T},
            {2: _DYNAMIC_T, 3: _DYNAMIC_T},
            {1: _DYNAMIC_T},
            {1: _DYNAMIC_T},
        ),
    ),
    GoldenSpec(
        name="runtime_masked_attention",
        build=lambda _: RuntimeMaskedAttention(),
        # mask is **a bool graph input** (not via i64 — the IR boundary of ADR 0044 decision 1).
        example_inputs=_runtime_masked_attention_inputs,
    ),
    GoldenSpec(
        name="conv_block",
        build=ConvBlock,
        example_inputs=lambda g: (
            _uniform(g, 1, GOLDEN_T, ConvBlock.CHANNELS, low=-1.0, high=1.0),
        ),
        dynamic_shapes=({1: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="symbolic_table",
        build=lambda _: SymbolicTable(),
        example_inputs=lambda g: (
            _uniform(g, GOLDEN_T, SymbolicTable.BUCKETS, low=-1.0, high=1.0),
        ),
        dynamic_shapes=({0: _SMALL_DYNAMIC_T},),
    ),
    GoldenSpec(
        name="scalar_operands",
        build=lambda _: ScalarOperands(),
        # mask is torch's default i64 (normalised to i32 at the boundary). It contains 0 so that
        # both values of `1 - mask` are exercised.
        example_inputs=lambda g: (
            _uniform(g, GOLDEN_T, 4, low=-1.0, high=1.0),
            _mask(GOLDEN_T, 4, zero_at=7),
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, {0: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="spline_pieces",
        build=lambda _: SplinePieces(),
        # A sequence that straddles both ±TAIL (1.0) and the two ends of the clamp (-0.75 / 0.5).
        # One point exactly on the boundary (-1.0) exercises where ge and gt part.
        example_inputs=lambda g: (
            torch.tensor([-2.0, -1.0, -0.4, 0.3, 1.2, 2.0]),
            _uniform(g, GOLDEN_T, 4, low=0.2, high=0.8),
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, {0: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="coupling_split",
        build=lambda _: CouplingSplit(),
        # x[1,6,T] (T symbolic while the channel axis stays static) and attn[2,4] (the last
        # dimension is padded)
        example_inputs=lambda g: (
            _uniform(g, 1, 6, GOLDEN_T, low=-1.5, high=1.5),
            _uniform(g, 2, 4, low=-1.0, high=1.0),
        ),
        dynamic_shapes=({2: _DYNAMIC_T}, None),
    ),
    GoldenSpec(
        name="dilated_conv",
        build=DilatedConvStack,
        example_inputs=lambda g: (
            _uniform(g, 1, DilatedConvStack.DEPTH_CHANNELS, GOLDEN_T, low=-1.5, high=1.5),
        ),
        dynamic_shapes=({2: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="conv_transpose",
        build=ConvTransposeStack,
        example_inputs=lambda g: (
            _uniform(g, 1, ConvTransposeStack.IN_CHANNELS, GOLDEN_T, low=-1.5, high=1.5),
        ),
        dynamic_shapes=({2: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="decoder_tail",
        build=DecoderTail,
        example_inputs=lambda g: (
            _uniform(g, 1, DecoderTail.CHANNELS, GOLDEN_T, low=-1.5, high=1.5),
            _uniform(g, 1, DecoderTail.CHANNELS, 1, low=0.5, high=1.5),
        ),
        dynamic_shapes=({2: _DYNAMIC_T}, None),
    ),
    GoldenSpec(
        name="rms_norm_block",
        build=RmsNormBlock,
        # A range that straddles 0 (rms's denominator is a sum of squares and ignores the sign,
        # but the sign of the weight multiplication shows)
        example_inputs=lambda g: (
            _uniform(g, 1, GOLDEN_T, RmsNormBlock.HIDDEN, low=-1.5, high=1.5),
        ),
        dynamic_shapes=({1: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="conv2d_block",
        build=Conv2dBlock,
        # x has H≠W (6×5). channels has a norm of 0 at one point only, exercising the floor of
        # clamp_min.
        example_inputs=lambda g: (
            _uniform(g, 1, 4, 6, 5, low=-1.5, high=1.5),
            _zeroed_column(g, 1, 4, 3, 5, at=(1, 2)),
        ),
    ),
    GoldenSpec(
        name="conv3d_block",
        build=Conv3dBlock,
        # x is unbatched [Cin, T, H, W] = [4, 5, 6, 5] (T≠H≠W — a swapped axis shows in the
        # values)
        example_inputs=lambda g: (_uniform(g, 4, 5, 6, 5, low=-1.5, high=1.5),),
    ),
    GoldenSpec(
        name="deform_conv2d_block",
        build=DeformConvBlock,
        # x has H≠W (4×5). offset swings ±2.5, out beyond the input plane, and mask is [0, 2] as
        # in BiRefNet. The wide branch's output space is 4×4 (H: 4+2−2 / W: 5+0−1), the point
        # branch's 4×5.
        example_inputs=lambda g: (
            _uniform(g, 1, DeformConvBlock.IN_CHANNELS, 4, 5, low=-1.5, high=1.5),
            _uniform(g, 1, 2 * 3 * 2, 4, 4, low=-2.5, high=2.5),
            _uniform(g, 1, 3 * 2, 4, 4, low=0.0, high=2.0),
            _uniform(g, 1, 2, 4, 5, low=-2.5, high=2.5),
            _uniform(g, 1, 1, 4, 5, low=0.0, high=2.0),
        ),
    ),
    GoldenSpec(
        name="gru_scan_block",
        build=GruScanBlock,
        # x[T, N, IN] is symbolic only on the time axis (the reason this op exists). state is the
        # initial state of the reverse direction, in a range that excludes 0 (so an
        # implementation that skips reading `h0` does not go green).
        example_inputs=lambda g: (
            _uniform(g, GOLDEN_T, 1, GruScanBlock.IN_FEATURES, low=-1.5, high=1.5),
            _uniform(g, 1, GruScanBlock.HIDDEN, low=0.2, high=0.9),
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, None),
    ),
    GoldenSpec(
        name="gated_delta_scan_block",
        build=lambda _: GatedDeltaScanBlock(),
        # q / k [1, T, 1, 128] (l2-normalised) / g / beta [1, T, 2]; only the length axis is
        # symbolic. g covers fast decay (−3) to long memory (−0.01).
        example_inputs=lambda g: (
            _unit_rows(g, 1, GOLDEN_T, 1, GatedDeltaScanBlock.HEAD_DIM),
            _unit_rows(g, 1, GOLDEN_T, 1, GatedDeltaScanBlock.HEAD_DIM),
            _uniform(g, 1, GOLDEN_T, 2, low=-3.0, high=-0.01),
            _uniform(g, 1, GOLDEN_T, 2, low=0.1, high=0.9),
        ),
        dynamic_shapes=({1: _DYNAMIC_T}, {1: _DYNAMIC_T}, {1: _DYNAMIC_T}, {1: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="bilinear_resize",
        build=lambda _: BilinearResize(),
        # x has H≠W (4×5). narrow has height 1 (the axis with scale 0) and shrinks the width.
        example_inputs=lambda g: (
            _uniform(g, 1, 3, 4, 5, low=-1.5, high=1.5),
            _uniform(g, 1, 2, 1, 6, low=-1.5, high=1.5),
        ),
    ),
    GoldenSpec(
        name="group_norm_block",
        build=GroupNormBlock,
        # x straddles 0 with a symbolic third axis; narrow sits at a mean offset of 3 (a one-pass
        # variance would lose digits there) with 520 elements per channel (its group spans two
        # partial chunks).
        example_inputs=lambda g: (
            _uniform(g, 2, GroupNormBlock.CHANNELS, GOLDEN_T, 5, low=-1.5, high=1.5),
            _uniform(g, 1, GroupNormBlock.NARROW_CHANNELS, 520, low=2.0, high=4.0),
        ),
        dynamic_shapes=({2: _DYNAMIC_T}, None),
    ),
    GoldenSpec(
        name="i8_weights",
        build=Int8Weights,
        # index is torch's default i64 (normalised to i32 at the boundary). **It always contains
        # 0** — this is the only case that looks up the all-zero row (the lower clamp of the
        # scale).
        example_inputs=lambda g: (
            torch.arange(GOLDEN_T, dtype=torch.int64) % Int8Weights.VOCAB,
            _uniform(g, 1, Int8Weights.SIGNAL_IN, GOLDEN_T, low=-1.5, high=1.5),
            _uniform(g, 1, Int8Weights.IMAGE_IN, 5, 3, low=-1.5, high=1.5),
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, {2: _DYNAMIC_T}, None),
        weight_dtype="i8",
    ),
)


def _assert_not_trivial(where: str, tensor: torch.Tensor) -> None:
    """Pin at generation time that a golden output is **not a trivial value**.

    An output with 2 or more elements that are all the same value is a tautological expectation
    on which a computation error does not show (real case: `(1 - mask) * mask` over mask ∈ {0,1}
    is identically 0 even with the sub reversed). A green comparison would verify nothing, so it
    fails before the fixture is written.
    """
    if tensor.numel() > 1 and tensor.unique().numel() == 1:
        raise AssertionError(
            f"{where}: a trivial expectation with every element the same value"
            f" ({tensor.flatten()[0].item()}) — the golden would not show a computation error"
        )


def generate_golden(spec: GoldenSpec, root: Path) -> IrGraph:
    """Write one model's container (the part list of `model.krm`) and io.safetensors, and return
    the graph.

    MUST: generating and verifying the model and io finish **inside a staging directory**, and the
    destination is swapped in as a whole only on a passing run
    ({@link karume.artifacts.staged_publication}). Writing to the final destination directly would
    leave "a new model + an old io" behind when verification fails after the model is in place or
    swapping in a part fails.
    """
    generator = _rng()
    module = spec.build(generator).eval()
    args = spec.example_inputs(generator)
    # MUST: fake-quant comes before **both export and the reference capture below** (ADR 0006).
    # Moved after them, only the expected values would be computed with the original weights,
    # and the E2E difference would be quantization error and implementation error combined.
    scales = fake_quant_int8(module).scales if spec.weight_dtype == "i8" else None
    with staged_publication(root / spec.name) as out_dir:
        out_dir.mkdir(parents=True)
        graph = export_to_file(
            module,
            args,
            out_dir / MODEL_FILE,
            provenance=GOLDEN_PROVENANCE,
            # The graph name is **the component name** (= the destination directory name = the
            # name the TS side looks up). There is no default (the MUST of
            # `pipeline._assert_graph_name`), so it is spelled here.
            graph_name=spec.name,
            dynamic_shapes=spec.dynamic_shapes,
            symbol_names=spec.symbol_names,
            weight_dtype=spec.weight_dtype,
            weight_scales=scales,
            preserved=spec.preserved,
        )

        with torch.no_grad():
            result = module(*args)
        outputs = result if isinstance(result, tuple) else (result,)
        if len(outputs) != len(graph.outputs):
            raise AssertionError(
                f"{spec.name}: {len(outputs)} torch outputs ≠ {len(graph.outputs)} IR outputs"
            )
        for index, tensor in enumerate(outputs):
            _assert_not_trivial(f"{spec.name} output {index}", tensor)
        # MUST: io is lowered to the actual representation of the IR's semantic dtypes too (i64 →
        # I32 / bool → U32 0/1). Unless it matches the form the runtime receives, the E2E on the TS
        # side cannot read the golden.
        io_tensors = {
            f"{INPUT_PREFIX}{declared.name}": normalize_boundary_tensor(
                tensor.detach(), f"{spec.name} input '{declared.name}'"
            )
            for declared, tensor in zip(graph.inputs, args, strict=True)
        }
        for index, tensor in enumerate(outputs):
            io_tensors[f"{OUTPUT_PREFIX}{index}"] = normalize_boundary_tensor(
                tensor.detach(), f"{spec.name} output {index}"
            )
        save_file(io_tensors, str(out_dir / IO_FILE))
    return graph


def generate_all(root: Path) -> dict[str, IrGraph]:
    """Generate every golden and check the op coverage of the contract table.

    The coverage check sits on the generator side to back mechanically the implementation
    contract of adding a golden for every op added (ADR 0005) — it fails the moment a model is
    trimmed and leaves a hole.
    """
    graphs = {spec.name: generate_golden(spec, root) for spec in GOLDEN_SPECS}
    covered = {op for graph in graphs.values() for op in graph.required_ops}
    uncovered = sorted(EMITTABLE_OPS - covered)
    if uncovered:
        raise AssertionError(f"ops not covered by any golden: {uncovered}")
    return graphs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory (required — e.g. ../../packages/runtime/tests/fixtures/golden)",
    )
    args = parser.parse_args()
    for name, graph in generate_all(args.out).items():
        print(f"{name}: ops={graph.required_ops} symbols={graph.symbols}")
    print(f"wrote {len(GOLDEN_SPECS)} models to {args.out}")


if __name__ == "__main__":
    main()
