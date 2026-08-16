"""tiny golden fixtures の生成（後段の Deno E2E テストがこれを読む）。

固定 seed の小モデルを torch.export → IR v1 コンテナ化し、torch CPU での期待出力を
同じディレクトリに置く。契約表の全 op を全モデル合計で必ず被覆する（COVERAGE 検査）。

    uv run python -m karume.goldens --out ../../packages/runtime/tests/fixtures/golden

置き場は**必ず引数で受ける**（`--out` 必須）— どのリポの `packages/runtime/tests/fixtures/`
へ落とすかは repo topology で、汎用 exporter の知識ではない（ADR 0065 Consequences）。
上の起動例は `tools/exporter/` から走らせたときの相対 path。

レイアウトとテンソルキー命名は README.md「golden レイアウト」に明記。

モデル定義（tiny な nn.Module 群）と重み初期化・入力生成ヘルパは
{@link karume.golden_models} — ここは台帳（{@link GOLDEN_SPECS}）と生成ドライバ側。
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

from karume.convert import (
    PRESERVED_OP_PREFIXES,
    PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    normalize_boundary_tensor,
)

# fixture 側（{@link karume.golden_models}）へ出したモデルと入力生成ヘルパの束縛。
# import は goldens → golden_models の一方向のみ（逆辺を張るとモデル定義が台帳へ食い込む）。
from karume.golden_models import (
    Activations,
    AttentionBlock,
    BatchMatmul,
    BilinearResize,
    BroadcastBinary,
    Conv2dBlock,
    ConvBlock,
    ConvTransposeStack,
    CouplingSplit,
    DecoderTail,
    DeformConvBlock,
    DilatedConvStack,
    EmbeddingLookup,
    ExpandMask,
    FusedAttention,
    GatherLastDim,
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
    SymbolicTable,
    UnaryChain,
    _band_mask,
    _fused_attention_inputs,
    _mask,
    _runtime_masked_attention_inputs,
    _uniform,
    _zeroed_column,
)
from karume.ir import IrGraph
from karume.ops import EMITTABLE_OPS
from karume.pipeline import export_to_file
from karume.quantize import fake_quant_int8

MODEL_FILE = "model.safetensors"
IO_FILE = "io.safetensors"
#: io.safetensors のキー規約。入力はグラフ入力名、出力は graph.outputs の位置で引く。
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: 乱数は全てここから引く（torch.Generator を明示 — グローバル seed に依存しない）。
SEED = 20260802
#: 記号次元 T の golden 実長。torch.export の 0/1 特殊化を避けるため 2 以上。
GOLDEN_T = 6
_DYNAMIC_T = Dim("T", min=2, max=64)
#: Tmax 畳み込みを踏む golden 用の小さい上限（焼いた定数は Tmax² で膨らむ — fixture を
#: 16KiB 未満に保つ）。T = GOLDEN_T はこれより十分小さく、prefix が本当に短くなる。
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
    #: 重みの格納 dtype。`"i8"` のとき `generate_golden` が **export と参照採取の前に**
    #: fake-quant を掛ける（ADR 0006 / 0019）。
    weight_dtype: str = "f32"
    #: 分解を止める高位 op の接頭辞集合（ADR 0023）。SDPA を保存して融合 `attention` を
    #: 出す golden だけが既定を差し替える — 既定へ足すと mask 付き SDPA まで保存対象になる。
    preserved: Sequence[str] = field(default=PRESERVED_OP_PREFIXES)


GOLDEN_SPECS: tuple[GoldenSpec, ...] = (
    GoldenSpec(
        name="unary_chain",
        build=lambda _: UnaryChain(),
        # sqrt / log の定義域を外さないよう、0 から離した正値だけを与える。
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
        # 除数 y は 0 を跨がせない（div の golden が発散しないように）。
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
        name="mask_chain",
        build=lambda _: MaskChain(),
        # mask は torch 既定の i64（境界で i32 へ正規化される）。0 を必ず含めて
        # bitwise_not / 真偽化の両方の分岐を踏む。
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
        # 切り捨てが round / floor と区別できる値（±.5 / ±.7 を跨ぐ）を与える。
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
        # mask / index は torch 既定の i64（境界で i32 へ正規化される）。mask は 0 を必ず
        # 含めて bitwise_not の両分岐を踏む。
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
        # B=2 / M=T / K=3 / N=5 — 全て違う長さ（軸の取り違えが数値に出る形）
        example_inputs=lambda g: (
            _uniform(g, 2, GOLDEN_T, 3, low=-1.0, high=1.0),
            _uniform(g, 2, 3, 5, low=-1.0, high=1.0),
        ),
        dynamic_shapes=({1: _DYNAMIC_T}, None),
    ),
    GoldenSpec(
        name="gather_last_dim",
        build=lambda _: GatherLastDim(),
        # index は torch 既定の i64（境界で i32 へ正規化される）。3i mod 4 で src の
        # 最終次元 4 を全部踏み、恒等でも単調でもない列にする（添字 1 ずれが必ず出る）。
        example_inputs=lambda g: (
            _uniform(g, 2, GOLDEN_T, 4, low=-2.0, high=2.0),
            (torch.arange(2 * GOLDEN_T * 3, dtype=torch.int64) * 3 % 4).reshape(2, GOLDEN_T, 3),
        ),
        dynamic_shapes=({1: _DYNAMIC_T}, {1: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="attention_block",
        build=AttentionBlock,
        # 2 本目の logits は素朴 softmax なら f32 で exp が 0 に潰れる領域（-205..-180）
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
        # SDPA を保存するのはこの golden だけ（既定の 11 op に足すと mask 付き SDPA を持つ
        # グラフが export 不能になる — ADR 0023）。
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    ),
    GoldenSpec(
        name="embedding_lookup",
        build=EmbeddingLookup,
        # 添字は torch 既定の i64（境界で i32 へ正規化される）。padding_idx=0 を必ず引く列で、
        # 語彙 5 行を一巡する。
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
        # mask は torch 既定の i64（境界で bool へ cast される）。**1 行を全マスク**にして、
        # 全要素が -3.4e38 になった行の softmax（一様分布）まで踏む。
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
        # mask は **bool のグラフ入力**（i64 経由にしない — ADR 0044 決定 1 の IR 境界）。
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
        # mask は torch 既定の i64（境界で i32 へ正規化される）。0 を含めて `1 - mask` の
        # 両方の値を踏む。
        example_inputs=lambda g: (
            _uniform(g, GOLDEN_T, 4, low=-1.0, high=1.0),
            _mask(GOLDEN_T, 4, zero_at=7),
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, {0: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="spline_pieces",
        build=lambda _: SplinePieces(),
        # ±TAIL（1.0）と clamp の両端（-0.75 / 0.5）をどちらも跨ぐ列。境界ちょうど（-1.0）も
        # 1 点入れて ge と gt の分かれ目を踏む。
        example_inputs=lambda g: (
            torch.tensor([-2.0, -1.0, -0.4, 0.3, 1.2, 2.0]),
            _uniform(g, GOLDEN_T, 4, low=0.2, high=0.8),
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, {0: _DYNAMIC_T}),
    ),
    GoldenSpec(
        name="coupling_split",
        build=lambda _: CouplingSplit(),
        # x[1,6,T]（チャネル軸を静的に保ったまま T は記号）と attn[2,4]（最終次元を pad）
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
        # 0 を跨ぐ値域（rms の分母は二乗和なので符号に依らないが、weight 掛けの符号は出る）
        example_inputs=lambda g: (
            _uniform(g, 1, GOLDEN_T, RmsNormBlock.HIDDEN, low=-1.5, high=1.5),
        ),
        dynamic_shapes=({1: _DYNAMIC_T},),
    ),
    GoldenSpec(
        name="conv2d_block",
        build=Conv2dBlock,
        # x は H≠W（6×5）。channels は 1 点だけノルム 0 にして clamp_min の床を踏む。
        example_inputs=lambda g: (
            _uniform(g, 1, 4, 6, 5, low=-1.5, high=1.5),
            _zeroed_column(g, 1, 4, 3, 5, at=(1, 2)),
        ),
    ),
    GoldenSpec(
        name="deform_conv2d_block",
        build=DeformConvBlock,
        # x は H≠W（4×5）。offset は ±2.5 で入力平面の外まで振り、mask は BiRefNet と同じ
        # [0, 2]。wide 分岐の出力空間は 4×4（H: 4+2−2 / W: 5+0−1）、point 分岐は 4×5。
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
        # x[T, N, IN] は時間軸だけ記号（この op の存在理由）。state は逆方向の初期状態で、
        # 0 を含めない値域にする（`h0` を読み飛ばす実装が緑にならない形）。
        example_inputs=lambda g: (
            _uniform(g, GOLDEN_T, 1, GruScanBlock.IN_FEATURES, low=-1.5, high=1.5),
            _uniform(g, 1, GruScanBlock.HIDDEN, low=0.2, high=0.9),
        ),
        dynamic_shapes=({0: _DYNAMIC_T}, None),
    ),
    GoldenSpec(
        name="bilinear_resize",
        build=lambda _: BilinearResize(),
        # x は H≠W（4×5）。narrow は高さ 1（scale 0 の軸）で幅は縮小させる。
        example_inputs=lambda g: (
            _uniform(g, 1, 3, 4, 5, low=-1.5, high=1.5),
            _uniform(g, 1, 2, 1, 6, low=-1.5, high=1.5),
        ),
    ),
    GoldenSpec(
        name="i8_weights",
        build=Int8Weights,
        # index は torch 既定の i64（境界で i32 へ正規化される）。**0 を必ず含める** —
        # 全ゼロ行（scale の下限 clamp）を引くのはこのケースだけ。
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
    """golden の出力が**自明値でない**ことを生成時に固定する。

    2 要素以上あるのに全要素が同じ値の出力は、演算の誤りが値に出ない恒真な期待値
    （実例: mask ∈ {0,1} に対する `(1 - mask) * mask` は逆順 sub でも恒等 0）。突合が
    緑でも何も検証していないので、fixture を書き出す前に落とす。
    """
    if tensor.numel() > 1 and tensor.unique().numel() == 1:
        raise AssertionError(
            f"{where}: 全要素が同じ値（{tensor.flatten()[0].item()}）の自明な期待値 —"
            " 演算の誤りが値に出ない golden になっている"
        )


def generate_golden(spec: GoldenSpec, root: Path) -> IrGraph:
    """1 モデル分の model.safetensors と io.safetensors を書き、グラフを返す。"""
    generator = _rng()
    module = spec.build(generator).eval()
    args = spec.example_inputs(generator)
    # MUST: fake-quant は export と**下の参照採取の両方より前**（ADR 0006）。後ろへ動かすと
    # 期待値だけが元の重みで計算され、E2E の差が量子化誤差と実装誤差の合成になる。
    scales = fake_quant_int8(module).scales if spec.weight_dtype == "i8" else None
    out_dir = root / spec.name
    graph = export_to_file(
        module,
        args,
        out_dir / MODEL_FILE,
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
            f"{spec.name}: torch 出力 {len(outputs)} 本 ≠ IR 出力 {len(graph.outputs)} 本"
        )
    for index, tensor in enumerate(outputs):
        _assert_not_trivial(f"{spec.name} の出力 {index}", tensor)
    # MUST: io も IR の意味論 dtype の実表現へ落とす（i64 → I32 / bool → U32 の 0/1）。
    # ランタイムが受け取る形と揃っていないと、Deno 側 E2E が golden を読めない。
    io_tensors = {
        f"{INPUT_PREFIX}{declared.name}": normalize_boundary_tensor(
            tensor.detach(), f"{spec.name} の入力 '{declared.name}'"
        )
        for declared, tensor in zip(graph.inputs, args, strict=True)
    }
    for index, tensor in enumerate(outputs):
        io_tensors[f"{OUTPUT_PREFIX}{index}"] = normalize_boundary_tensor(
            tensor.detach(), f"{spec.name} の出力 {index}"
        )
    save_file(io_tensors, str(out_dir / IO_FILE))
    return graph


def generate_all(root: Path) -> dict[str, IrGraph]:
    """全 golden を生成し、契約表の op 被覆を検査する。

    被覆検査を生成側に置くのは、op を 1 個足すたびに golden を足す実装契約
    （ADR 0005）を機械で担保するため — モデルを削って穴を空けた瞬間に落ちる。
    """
    graphs = {spec.name: generate_golden(spec, root) for spec in GOLDEN_SPECS}
    covered = {op for graph in graphs.values() for op in graph.required_ops}
    uncovered = sorted(EMITTABLE_OPS - covered)
    if uncovered:
        raise AssertionError(f"golden がカバーしない op: {uncovered}")
    return graphs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="生成先ディレクトリ（必須 — 例: ../../packages/runtime/tests/fixtures/golden）",
    )
    args = parser.parse_args()
    for name, graph in generate_all(args.out).items():
        print(f"{name}: ops={graph.required_ops} symbols={graph.symbols}")
    print(f"wrote {len(GOLDEN_SPECS)} models to {args.out}")


if __name__ == "__main__":
    main()
