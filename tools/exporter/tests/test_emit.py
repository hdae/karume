"""配布形の書き出し。宣言（グラフ JSON）と格納テンソルの対応が崩れたら書かない。"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import weakref
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from karume.emit import (
    INT4_OFFSET,
    EmitError,
    eligible_compressed_initializers,
    i4_eligible_initializers,
    pack_int4,
    storage_breakdown,
    unpack_int4,
    weight_channel_axes,
    write_model,
)
from karume.ir import (
    IR_METADATA_KEY,
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrStorage,
    IrValue,
)
from karume.ops import OpContractError
from karume.quantize import (
    channel_scale,
    dequantize_int4,
    group_scale,
    quantize_to_int4,
    quantize_to_int8,
)
from karume.shards import (
    ShardError,
    parse_piece_key,
    resolve_shards,
    shard_path,
    shard_siblings,
)
from karume.verify import (
    ContainerError,
    assert_reader_layout,
    parse_ir_graph,
    verify_model,
    verify_shards,
)


def sample_graph() -> tuple[IrGraph, dict[str, torch.Tensor]]:
    graph = IrGraph(
        symbols=["T"],
        inputs=[IrInput(name="x", dtype="f32", shape=["T", 4])],
        outputs=["y"],
        initializers={"w": IrInitializer(tensor="enc.w", storage=IrStorage(dtype="f32"))},
        values={
            "w": IrValue(dtype="f32", shape=[4]),
            "y": IrValue(dtype="f32", shape=["T", 4]),
        },
        nodes=[IrNode(op="add", ins=["x", "w"], outs=["y"], attrs={})],
    )
    return graph, {"enc.w": torch.ones(4)}


class TestRoundTrip:
    def test_the_graph_is_embedded_under_the_metadata_key(self, tmp_path):
        """グラフはグラフ shard の `__metadata__` に載り、テンソルは weight shard に載る。"""
        graph, tensors = sample_graph()

        path = write_component(tmp_path / "model.safetensors", graph, tensors)

        with safe_open(str(graph_shard(path)), framework="pt") as handle:
            assert json.loads(handle.metadata()[IR_METADATA_KEY]) == graph.to_dict()
            assert set(handle.keys()) == set()
        with safe_open(str(weight_shard(path)), framework="pt") as handle:
            assert handle.metadata() == {}
            assert set(handle.keys()) == {"enc.w"}

    def test_the_written_file_passes_the_full_verification(self, tmp_path):
        graph, tensors = sample_graph()

        path = write_component(tmp_path / "model.safetensors", graph, tensors)

        assert verify_model(path).to_dict() == graph.to_dict()


class TestDeclarationAgreement:
    def test_missing_tensor_fails_loudly(self, tmp_path):
        graph, _ = sample_graph()

        with pytest.raises(EmitError, match="欠落"):
            write_model(tmp_path / "model.safetensors", graph, {})

    def test_unreferenced_tensor_fails_loudly(self, tmp_path):
        graph, tensors = sample_graph()

        with pytest.raises(EmitError, match="余剰"):
            write_model(tmp_path / "model.safetensors", graph, {**tensors, "stray": torch.zeros(2)})


def rounded(*shape: int) -> torch.Tensor:
    """f16 表現可能値へ丸めた乱数（= fake-quant 済みの重み）。"""
    return torch.randn(*shape).to(torch.float16).to(torch.float32)


def weight_graph(
    *, share_weight: bool = False, unused: bool = False, output_weight: bool = False
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """linear（重み偶数要素）+ embedding（重み **奇数要素**）を持つグラフ。

    `share_weight` は linear の重みを elementwise でも消費する形（適格外になるはず）、
    `unused` は誰も消費しない initializer を 1 本足す（同じく適格外）、
    `output_weight` は linear の重みをそのままグラフ出力にする形（同じく適格外 —
    ランタイムの readback は semantic f32 を仮定する）。
    """
    nodes = [
        IrNode(op="linear", ins=["x", "w", "b"], outs=["h"], attrs={}),
        IrNode(op="embedding", ins=["emb", "idx"], outs=["e"], attrs={"padding_idx": -1}),
    ]
    initializers = {
        "w": IrInitializer(tensor="enc.w", storage=IrStorage(dtype="f32")),
        "b": IrInitializer(tensor="enc.b", storage=IrStorage(dtype="f32")),
        "emb": IrInitializer(tensor="enc.emb", storage=IrStorage(dtype="f32")),
    }
    values = {
        "w": IrValue(dtype="f32", shape=[3, 4]),
        "b": IrValue(dtype="f32", shape=[3]),
        "emb": IrValue(dtype="f32", shape=[3, 5]),
        "h": IrValue(dtype="f32", shape=["T", 3]),
        "e": IrValue(dtype="f32", shape=["T", 5]),
    }
    tensors = {"enc.w": rounded(3, 4), "enc.b": rounded(3), "enc.emb": rounded(3, 5)}
    outputs = ["h", "e"]
    if share_weight:
        # 同じテンソルを linear の重みと mul の被演算子の両方で使う（後者は f32 として読む）。
        nodes.append(IrNode(op="mul", ins=["w", "w"], outs=["w2"], attrs={}))
        values["w2"] = IrValue(dtype="f32", shape=[3, 4])
        outputs.append("w2")
    if unused:
        initializers["dead"] = IrInitializer(tensor="enc.dead", storage=IrStorage(dtype="f32"))
        values["dead"] = IrValue(dtype="f32", shape=[6])
        tensors["enc.dead"] = rounded(6)
    if output_weight:
        # IR は initializer 名をそのままグラフ出力に書くことを許す（値は実行に依らない定数）。
        outputs.append("w")
    graph = IrGraph(
        symbols=["T"],
        inputs=[
            IrInput(name="x", dtype="f32", shape=["T", 4]),
            IrInput(name="idx", dtype="i32", shape=["T"]),
        ],
        outputs=outputs,
        initializers=initializers,
        values=values,
        nodes=nodes,
    )
    return graph, tensors


def compressed_view(
    graph: IrGraph, dtypes: dict[str, str], *, group_size: int | None = None
) -> IrGraph:
    """`write_model` がヘッダへ載せるはずの宣言ビュー（呼び手の graph は書き換わらない）。

    companion scale のキー規則（`karume.scale.<テンソルキー>`）は実装を呼ばずテスト側で
    書き下す — 実装から引くと突合が恒真になる。`group_size` は i4 の宣言欄
    （ADR 0069 決定 2）で、i8 / f16 の宣言には載らない。
    """
    initializers = dict(graph.initializers)
    for name, dtype in dtypes.items():
        key = initializers[name].tensor
        scale = f"karume.scale.{key}" if dtype in ("i8", "i4") else None
        initializers[name] = IrInitializer(
            tensor=key,
            storage=IrStorage(
                dtype=dtype, scale=scale, group_size=group_size if dtype == "i4" else None
            ),
        )
    return replace(graph, initializers=initializers)


class TestEligibility:
    """適格判定は `packages/runtime/src/runtime/plan.ts` の鏡像 —
    ずれると VRAM 削減が黙って消える。
    """

    def test_only_weight_slots_are_eligible(self):
        graph, _ = weight_graph()

        assert eligible_compressed_initializers(graph) == {"w", "emb"}

    def test_a_weight_consumed_elsewhere_is_disqualified(self):
        """1 つでも重みスロット以外の消費があれば適格外（そちらは f32 として読む）。"""
        graph, _ = weight_graph(share_weight=True)

        assert eligible_compressed_initializers(graph) == {"emb"}

    def test_an_unconsumed_initializer_is_disqualified(self):
        """実行に使われないバイトを「GPU 常駐圧縮」と数えると診断が実態からずれる。"""
        graph, _ = weight_graph(unused=True)

        assert "dead" not in eligible_compressed_initializers(graph)

    def test_an_initializer_in_the_graph_outputs_is_disqualified(self):
        """readback は semantic f32（4 バイト / 要素）を仮定して重みバッファから写す。

        圧縮のまま常駐させると copy が実バッファをはみ出して validation で落ちるか、
        極小サイズではビット列の読み替えが黙って返る。
        """
        graph, _ = weight_graph(output_weight=True)

        # 失格は出力に載った名前だけ（同じグラフの embedding 表は適格のまま）
        assert eligible_compressed_initializers(graph) == {"emb"}


class TestF16Storage:
    def test_eligible_weights_are_stored_as_f16_and_bias_stays_f32(self, tmp_path):
        graph, tensors = weight_graph()

        path = write_component(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        # 宣言の観測点はファイル側（`write_model` は呼び手の graph を書き換えない）。
        declared = verify_model(path).initializers
        assert declared["w"].storage.dtype == "f16"
        assert declared["emb"].storage.dtype == "f16"
        # MUST: bias は常に f32（プロトタイプの f16 降格バグの根治形 — ADR 0006）。
        assert declared["b"].storage.dtype == "f32"
        with safe_open(str(weight_shard(path)), framework="pt") as handle:
            assert handle.get_slice("enc.w").get_dtype() == "F16"
            assert handle.get_slice("enc.b").get_dtype() == "F32"

    def test_a_weight_in_the_graph_outputs_is_not_stored_compressed(self, tmp_path):
        """グラフ出力の重みは f32 のまま書く（ランタイム側の適格判定と対）。

        鏡像がずれると「exporter は f16 で書き、ランタイムは f32 として読み戻す」の
        食い違いになり、ロードした瞬間に読めないモデルが配布形として通ってしまう。
        """
        graph, tensors = weight_graph(output_weight=True)

        path = write_component(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        declared = verify_model(path).initializers
        assert declared["w"].storage.dtype == "f32"
        assert declared["emb"].storage.dtype == "f16"
        with safe_open(str(weight_shard(path)), framework="pt") as handle:
            assert handle.get_slice("enc.w").get_dtype() == "F32"
            assert handle.get_slice("enc.emb").get_dtype() == "F16"
        assert verify_model(path).to_dict() == compressed_view(graph, {"emb": "f16"}).to_dict()

    def test_the_f16_file_passes_the_full_verification(self, tmp_path):
        graph, tensors = weight_graph()

        path = write_component(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        expected = compressed_view(graph, {"w": "f16", "emb": "f16"})
        assert verify_model(path).to_dict() == expected.to_dict()

    def test_the_stored_values_match_the_rounded_weights_bit_for_bit(self, tmp_path):
        graph, tensors = weight_graph()
        expected = tensors["enc.w"].clone()

        path = write_component(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        with safe_open(str(weight_shard(path)), framework="pt") as handle:
            stored = handle.get_tensor("enc.w").to(torch.float32)
        assert torch.equal(stored, expected)

    def test_an_unrounded_eligible_weight_fails_loudly(self, tmp_path):
        """丸めの掛け忘れ / 掛ける順序の誤りは黙って通さない（ADR 0006）。

        通すと「golden は元値・実行は丸め値」になり、E2E の差が量子化誤差と実装誤差の
        合成になる — tolerance を緩める方向にしか効かないので、緑のまま検出力だけが落ちる。
        """
        graph, tensors = weight_graph()
        tensors["enc.w"] = torch.full((3, 4), 1.0 / 3.0)

        with pytest.raises(EmitError, match="fake-quant"):
            write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

    def test_a_graph_without_eligible_weights_fails_loudly(self, tmp_path):
        """「f16 指定なのに適格 0MB」を沈黙させない（ADR 0006 の常設診断の書き出し側）。"""
        graph, tensors = sample_graph()

        with pytest.raises(EmitError, match="圧縮格納が 1 本も計画されなかった"):
            write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

    def test_an_unknown_weight_dtype_fails_loudly(self, tmp_path):
        graph, tensors = weight_graph()

        # bf16 は IR の語彙にはあるが実行経路が無い（ADR 0006）。
        with pytest.raises(EmitError, match="書き出せない"):
            write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="bf16")

    def test_the_breakdown_counts_eligible_and_plain_bytes(self, tmp_path):
        graph, tensors = weight_graph()

        path = write_component(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")
        # 内訳は**書いたファイルの宣言**から数える（呼び手の graph は圧縮宣言を持たない）。
        breakdown = storage_breakdown(verify_model(path))

        assert breakdown.compressed_tensors == 2
        assert breakdown.compressed_bytes == (3 * 4 + 3 * 5) * 2
        assert breakdown.plain_tensors == 1
        assert breakdown.plain_bytes == 3 * 4

    def test_the_breakdown_counts_a_declared_bf16_initializer(self):
        """bf16 は「宣言だけ受理」の格納 dtype（実行可否は verify が単独で持つ）。

        内訳はバイト数を数えるだけの層なので、**読めるグラフを数えられない**状態
        （語彙にある dtype で KeyError）を作らない。エクスポータは bf16 を書かないので、
        到達経路は `parse_ir_graph` で読んだグラフと手組みグラフ。
        """
        graph, _ = weight_graph()
        graph.initializers["w"] = IrInitializer(tensor="enc.w", storage=IrStorage(dtype="bf16"))

        breakdown = storage_breakdown(graph)

        assert breakdown.compressed_tensors == 1
        assert breakdown.compressed_bytes == 3 * 4 * 2
        assert breakdown.plain_tensors == 2
        assert breakdown.plain_bytes == (3 + 3 * 5) * 4
        assert breakdown.scale_bytes == 0


def int8_weight_graph() -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """`weight_graph` を i8 の fake-quant 済みにしたもの（グラフ / テンソル / scale 台帳）。

    `enc.w` は linear の重み（軸 0・`[3,4]`）、`enc.emb` は embedding の表（軸 0・`[3,5]`）。
    conv_transpose1d の軸 1 は `TestI8ChannelAxes` が別に踏む。
    """
    graph, tensors = weight_graph()
    scales: dict[str, torch.Tensor] = {}
    for key in ("enc.w", "enc.emb"):
        scale = channel_scale(tensors[key], 0)
        tensors[key] = quantize_to_int8(tensors[key], scale).to(torch.float32) * scale
        scales[key] = scale
    return graph, tensors, scales


class TestI8ChannelAxes:
    """per-channel 軸は**消費側の op** から引く（重みの shape だけでは決まらない）。"""

    def test_the_axis_comes_from_the_consuming_op(self):
        graph, _ = weight_graph()

        assert weight_channel_axes(graph) == {"w": 0, "emb": 0}

    def test_conv_transpose1d_uses_the_second_axis(self):
        graph, _ = weight_graph()
        graph.nodes[0] = IrNode(op="conv_transpose1d", ins=["x", "w", "b"], outs=["h"], attrs={})

        assert weight_channel_axes(graph)["w"] == 1

    def test_conflicting_axes_fail_loudly(self):
        """同じ重みを軸の違う op が消費する形は 1 本の scale で表せない。"""
        graph, _ = weight_graph()
        graph.nodes.append(
            IrNode(op="conv_transpose1d", ins=["x", "w", "b"], outs=["h2"], attrs={})
        )
        graph.values["h2"] = IrValue(dtype="f32", shape=["T", 3])
        graph.outputs.append("h2")

        with pytest.raises(EmitError, match="per-channel 軸が違う"):
            weight_channel_axes(graph)


class TestI8Storage:
    def test_eligible_weights_are_stored_as_i8_with_a_scale_and_bias_stays_f32(self, tmp_path):
        graph, tensors, scales = int8_weight_graph()

        path = write_component(
            tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8", weight_scales=scales
        )

        declared = verify_model(path).initializers
        assert declared["w"].storage.dtype == "i8"
        assert declared["w"].storage.scale == "karume.scale.enc.w"
        assert declared["emb"].storage.dtype == "i8"
        # MUST: bias は常に f32（プロトタイプの降格バグの根治形 — ADR 0006）。
        assert declared["b"].storage.dtype == "f32"
        with safe_open(str(weight_shard(path)), framework="pt") as handle:
            assert handle.get_slice("enc.w").get_dtype() == "I8"
            assert handle.get_slice("enc.b").get_dtype() == "F32"
            assert handle.get_slice("karume.scale.enc.w").get_dtype() == "F32"
            assert list(handle.get_slice("karume.scale.enc.w").get_shape()) == [3, 1]

    def test_the_i8_file_passes_the_full_verification(self, tmp_path):
        graph, tensors, scales = int8_weight_graph()

        path = write_component(
            tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8", weight_scales=scales
        )

        expected = compressed_view(graph, {"w": "i8", "emb": "i8"})
        assert verify_model(path).to_dict() == expected.to_dict()

    def test_the_stored_values_reconstruct_the_weights_bit_for_bit(self, tmp_path):
        """`q8·scale` が fake-quant 済みの重みと**ビット一致**する（格納の意味そのもの）。"""
        graph, tensors, scales = int8_weight_graph()
        expected = tensors["enc.w"].clone()

        path = write_component(
            tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8", weight_scales=scales
        )

        with safe_open(str(weight_shard(path)), framework="pt") as handle:
            stored = handle.get_tensor("enc.w").to(torch.float32)
            scale = handle.get_tensor("karume.scale.enc.w")
        assert torch.equal(stored * scale, expected)

    def test_an_unrounded_eligible_weight_fails_loudly(self, tmp_path):
        """丸めの掛け忘れ / 順序の誤りは黙って通さない（ADR 0006）。"""
        graph, tensors, scales = int8_weight_graph()
        tensors["enc.w"] = torch.full((3, 4), 1.0 / 3.0)

        with pytest.raises(EmitError, match="ビット一致しない"):
            write_model(
                tmp_path / "model.safetensors",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=scales,
            )

    def test_a_scale_that_is_not_the_one_fake_quant_used_fails_loudly(self, tmp_path):
        """fake-quant が使ったのと**別の** scale で書こうとすると逆変換ゲートが落ちる。

        ADR 0019 の「scale は再計算せずそのまま渡す」を守れなかったときの一般形。1ulp でも
        違えば全要素の格納値が golden と対応しなくなり、差は例外にならず値にだけ出るので、
        ここで書き出しごと止める。
        """
        graph, tensors, scales = int8_weight_graph()
        drifted = {key: value * 1.0000002 for key, value in scales.items()}

        with pytest.raises(EmitError, match="ビット一致しない"):
            write_model(
                tmp_path / "model.safetensors",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=drifted,
            )

    def test_recomputing_the_scale_from_a_quantized_weight_is_a_fixed_point(self):
        """NOTE（実測・期待と違う）: **正しく fake-quant 済みの重みからの再計算は同値**。

        `q` を ±127 に閉じているので最大絶対値要素は必ず `q = ±127` に乗り、
        `amax(|q·s|)/127 = fl(fl(127·s)/127) = s`（f32 の不動点 — 乱数 8.9e7 サンプルで
        反例ゼロ）。つまり ADR 0019 の「再計算禁止」は**逆変換ゲートでは検出できない**規律で、
        守る理由は「emit の時点の重みが実効重みでない / 軸が違う / 式が違う」ときに黙って別の
        scale になることのほうにある。この不動点性は冪等性（test_quantize.py）と同じ性質なので、
        ここでも固定しておく（崩れたら両方が同時に崩れる）。
        """
        _, tensors, scales = int8_weight_graph()

        for key, scale in scales.items():
            assert torch.equal(channel_scale(tensors[key], 0), scale), key

    def test_a_scale_that_is_not_f32_fails_loudly(self, tmp_path):
        """companion scale は F32 固定（ADR 0019）。writer は F16 もそのまま直列化できる。

        逆変換の等値検査は「同じ f16 scale で fake-quant 済み」なら通ってしまうので、
        検出は計画段の dtype 検査でしか掛からない（後段の `verify_model` は書いた後）。
        """
        graph, tensors, scales = int8_weight_graph()
        half = {key: value.to(torch.float16) for key, value in scales.items()}
        for key, scale in half.items():
            tensors[key] = quantize_to_int8(tensors[key], scale).to(torch.float32) * scale

        with pytest.raises(EmitError, match="scale の dtype"):
            write_model(
                tmp_path / "model.safetensors",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=half,
            )

    def test_a_missing_scale_fails_loudly(self, tmp_path):
        """適格なのに scale が無い = fake-quant が届いていない重み（f32 へ落とさない）。"""
        graph, tensors, scales = int8_weight_graph()
        del scales["enc.emb"]

        with pytest.raises(EmitError, match="per-channel scale が無い"):
            write_model(
                tmp_path / "model.safetensors",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=scales,
            )

    def test_a_scale_with_the_wrong_shape_fails_loudly(self, tmp_path):
        """broadcast できるだけの形（`[1,4]`）はカーネルの読み方と食い違う沈黙誤値になる。"""
        graph, tensors, scales = int8_weight_graph()
        scales["enc.w"] = torch.ones(1, 4)

        with pytest.raises(EmitError, match="keepdim 形"):
            write_model(
                tmp_path / "model.safetensors",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=scales,
            )

    def test_a_scale_key_colliding_with_a_real_tensor_fails_loudly(self, tmp_path):
        """scale のキーが実テンソルと衝突すると「別の重みを scale として読む」形になる。"""
        graph, tensors, scales = int8_weight_graph()
        graph.initializers["b"] = IrInitializer(
            tensor="karume.scale.enc.w", storage=IrStorage(dtype="f32")
        )
        tensors["karume.scale.enc.w"] = tensors.pop("enc.b")

        with pytest.raises(EmitError, match="衝突"):
            write_model(
                tmp_path / "model.safetensors",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=scales,
            )

    def test_a_graph_without_eligible_weights_fails_loudly(self, tmp_path):
        graph, tensors = sample_graph()

        with pytest.raises(EmitError, match="圧縮格納が 1 本も計画されなかった"):
            write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8")

    def test_the_breakdown_counts_i8_bytes_and_the_scale_overhead(self, tmp_path):
        graph, tensors, scales = int8_weight_graph()

        path = write_component(
            tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8", weight_scales=scales
        )
        # 内訳は**書いたファイルの宣言**から数える（呼び手の graph は圧縮宣言を持たない）。
        breakdown = storage_breakdown(verify_model(path))

        assert breakdown.compressed_tensors == 2
        assert breakdown.compressed_bytes == 3 * 4 + 3 * 5  # i8 = 1 バイト/要素
        # scale は出力チャネル数 × 4 バイト（w も emb も軸 0 の長さ 3）。
        assert breakdown.scale_bytes == (3 + 3) * 4
        assert breakdown.plain_tensors == 1
        assert breakdown.plain_bytes == 3 * 4


def write_component(path, *args, **kwargs) -> Path:
    """書いて**代表 path** を返す（現物は連番の shard 列 — ADR 0081）。

    配布形は常に「グラフ shard + weight shard 列」なので、書いた `path` 自身は存在しない。
    観測点は席で違う — 宣言（`karume_ir`）は {@link graph_shard}、テンソルは
    {@link weight_shard}、両者をまたぐ検証は代表 path のまま `verify_model` が解決する。
    """
    write_model(path, *args, **kwargs)
    return Path(path)


def graph_shard(path) -> Path:
    """コンポーネントの先頭 shard（`karume_ir` を載せる器・データ節は空）。"""
    return resolve_shards(Path(path))[0]


def weight_shard(path) -> Path:
    """コンポーネントの weight shard（合成のテンソルは小さいので常に 1 本）。"""
    shards = resolve_shards(Path(path))
    assert len(shards) == 2, f"weight shard が 1 本でない: {[p.name for p in shards]}"
    return shards[1]


def container_header(path) -> dict:
    """safetensors のヘッダ JSON を直に読む。

    `safetensors` ライブラリ（0.8.0）は `I4` を知らないので、packed 4bit を含む配布形は
    `safe_open` では開けない（ADR 0069 決定 2）— 観測はヘッダ JSON から行う。
    """
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    return json.loads(raw[8 : 8 + length])


def stored_payload(path, name: str) -> torch.Tensor:
    """データ節から 1 本ぶんの生バイトを uint8 テンソルで取り出す（代表 path を受ける）。"""
    raw = weight_shard(path).read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    begin, end = json.loads(raw[8 : 8 + length])[name]["data_offsets"]
    start = 8 + length
    return torch.frombuffer(bytearray(raw[start + begin : start + end]), dtype=torch.uint8)


def written_graph(path) -> IrGraph:
    """書いたファイルに載った宣言（ヘッダの埋め込みグラフを `parse_ir_graph` で読み直す）。

    i4 は実行 capability が未開放（ADR 0069 の実行波で開く）なので `verify_model` は最後まで
    通らない — 宣言の観測点をここに置く。読むのは verify の正規のパーサなので、i4 の宣言規則
    （scale + group_size 必須・2 冪 ≥ 16・量子化軸の整除）はこの経路でも掛かる。
    """
    return parse_ir_graph(container_header(graph_shard(path))["__metadata__"][IR_METADATA_KEY])


def asymmetric_nibbles(count: int) -> torch.Tensor:
    """隣接要素が全て異なる非対称パターン `[1,−2,3,−4,…]`（値域 ±7）。

    pack 順の検出器（ADR 0069 決定 4 ①）— 対称なパターンだと上下 nibble を取り違えても
    往復が通ってしまう。
    """
    index = torch.arange(count)
    return (((index % 7) + 1) * torch.where(index % 2 == 0, 1, -1)).to(torch.int8)


class TestPackedFourBitOrder:
    """pack 順の正本（ADR 0069 決定 4）— 「要素 2i が下位 nibble / 2i+1 が上位 nibble」。

    上流でも割れている自由パラメータで、間違えても**形も型も合う沈黙誤値**にしかならない。
    往復（pack → unpack）だけでは pack と unpack が同じ向きに間違った形を検出できないので、
    バイト値を手で書き下した固定と対で置く。
    """

    def test_the_even_element_goes_to_the_low_nibble(self):
        """`q = [1, 2]` → `u = [9, 10]` → 1 バイト `0xA9`（上下を逆に詰めると `0x9A`）。"""
        packed = pack_int4(torch.tensor([1, 2], dtype=torch.int8))

        assert packed.dtype is torch.uint8
        assert packed.tolist() == [0xA9]

    def test_the_stored_nibbles_are_offset_by_eight_and_never_zero(self):
        """格納値は `u = q + 8`（値域 [1,15]・0 は未使用の 15 準位 — ADR 0069 決定 3）。"""
        packed = pack_int4(torch.tensor([-7, 7, 0, -7], dtype=torch.int8))

        assert INT4_OFFSET == 8
        assert packed.tolist() == [0xF1, 0x18]
        nibbles = [byte & 0x0F for byte in packed.tolist()] + [
            byte >> 4 for byte in packed.tolist()
        ]
        assert 0 not in nibbles

    @pytest.mark.parametrize(
        "shape",
        [(1, 16), (3, 16), (2, 48), (5, 20), (7, 4), (2, 3, 6)],
        ids=["one-row", "two-words", "six-words", "row-straddles", "short-rows", "rank3"],
    )
    def test_the_round_trip_preserves_every_position(self, shape):
        """行長が語（4 バイト = 8 要素）境界と一致しない形も含めて、位置が完全に一致する。

        平坦添字の罠（ADR 0019 の同型検出器の 4bit 版）— 行ごとに詰め直す実装だと
        `(5,20)` / `(7,4)` のように行が語の途中で終わる形で位置がずれる。
        """
        count = 1
        for dim in shape:
            count *= dim
        quantized = asymmetric_nibbles(count).reshape(shape)

        packed = pack_int4(quantized)

        assert packed.numel() == count // 2
        assert torch.equal(unpack_int4(packed, shape), quantized)

    def test_a_split_half_pack_does_not_survive_the_round_trip(self):
        """故障注入: llama.cpp Q4_0 型の split-half 順（前半 = 下位 / 後半 = 上位）で詰めると、
        こちらの unpack は**形も型も合ったまま**別の列を返す。

        検出器が実際に検出力を持つことの確認 — ここが通ってしまうなら往復テストは恒真。
        """
        quantized = asymmetric_nibbles(16)
        nibbles = (quantized + INT4_OFFSET).to(torch.uint8)
        half = nibbles.numel() // 2

        split = nibbles[:half] | (nibbles[half:] << 4)

        restored = unpack_int4(split, (16,))
        assert restored.shape == quantized.shape
        assert not torch.equal(restored, quantized)

    def test_an_odd_element_count_fails_loudly(self):
        """末尾要素が半バイトだけ突き出す形は詰められない（バイト長が宣言から決まらない）。"""
        with pytest.raises(EmitError, match="奇数"):
            pack_int4(torch.zeros(3, dtype=torch.int8))


def conv1d_attrs(groups: int = 1) -> dict:
    """conv1d の attrs（4 つとも宣言必須 — `ops.CONV1D_ATTRS`）。"""
    return {"stride": 1, "padding": 0, "dilation": 1, "groups": groups}


def int4_weight_graph(
    *,
    embedding: bool = False,
    conv: bool = False,
    wide: bool = False,
    group_size: int = 16,
    conv_shape: tuple[int, ...] = (3, 2, 16),
    conv_groups: int = 1,
    conv_op: str = "conv1d",
) -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """linear の重み `[3,32]` を group 対称 i4 で fake-quant 済みにしたグラフ。

    in 軸 32 / group 16 なので 1 行に group が 2 つ入る（scale の group 形 `[3,2]`）。
    `embedding` を立てると **embedding 表**（i4 適格の 2 つ目の重みスロット — ADR 0069
    決定 5 の追補）が、`conv` を立てると **conv の重み**（既定は `groups == 1` の conv1d
    `[3,2,16]` = i4 適格の 3 つ目。`conv_shape` / `conv_groups` / `conv_op` で適格から
    外れる枝も踏める）が増える。`wide` を立てると **行長 48 の linear**（出荷 g 16 では
    割り切れるが `quantize.DEFAULT_GROUP_SIZE` = 32 では割り切れない形）が増える。

    conv の重みは**適格な形のときだけ**丸めて台帳に載せる（適格外は emit が f32 のまま
    残すので、丸めても台帳が使われない）。
    """
    nodes = [IrNode(op="linear", ins=["x", "w", "b"], outs=["h"], attrs={})]
    inputs = [IrInput(name="x", dtype="f32", shape=["T", 32])]
    initializers = {
        "w": IrInitializer(tensor="enc.w", storage=IrStorage(dtype="f32")),
        "b": IrInitializer(tensor="enc.b", storage=IrStorage(dtype="f32")),
    }
    values = {
        "w": IrValue(dtype="f32", shape=[3, 32]),
        "b": IrValue(dtype="f32", shape=[3]),
        "h": IrValue(dtype="f32", shape=["T", 3]),
    }
    tensors = {"enc.w": torch.randn(3, 32), "enc.b": torch.randn(3)}
    outputs = ["h"]
    if embedding:
        nodes.append(
            IrNode(op="embedding", ins=["emb", "idx"], outs=["e"], attrs={"padding_idx": -1})
        )
        inputs.append(IrInput(name="idx", dtype="i32", shape=["T"]))
        initializers["emb"] = IrInitializer(tensor="enc.emb", storage=IrStorage(dtype="f32"))
        values["emb"] = IrValue(dtype="f32", shape=[3, 32])
        values["e"] = IrValue(dtype="f32", shape=["T", 32])
        tensors["enc.emb"] = torch.randn(3, 32)
        outputs.append("e")
    if wide:
        nodes.append(IrNode(op="linear", ins=["xw", "ww", "bw"], outs=["hw"], attrs={}))
        inputs.append(IrInput(name="xw", dtype="f32", shape=["T", 48]))
        initializers["ww"] = IrInitializer(tensor="enc.ww", storage=IrStorage(dtype="f32"))
        initializers["bw"] = IrInitializer(tensor="enc.bw", storage=IrStorage(dtype="f32"))
        values["ww"] = IrValue(dtype="f32", shape=[3, 48])
        values["bw"] = IrValue(dtype="f32", shape=[3])
        values["hw"] = IrValue(dtype="f32", shape=["T", 3])
        tensors["enc.ww"] = torch.randn(3, 48)
        tensors["enc.bw"] = torch.randn(3)
        outputs.append("hw")
    if conv:
        # conv の入力は rank 3（`x[B,Cin,L]`）— L = K + 1 なので出力長は 2
        # （stride / dilation 1・padding 0）。verify の shape 推論まで通す形にしておく。
        channels, kernel = conv_shape[1] * conv_groups, conv_shape[2]
        attrs = conv1d_attrs(conv_groups) if conv_op == "conv1d" else {"stride": 1, "padding": 0}
        nodes.append(IrNode(op=conv_op, ins=["cx", "cw", "cb"], outs=["c"], attrs=attrs))
        inputs.append(IrInput(name="cx", dtype="f32", shape=[1, channels, kernel + 1]))
        initializers["cw"] = IrInitializer(tensor="enc.cw", storage=IrStorage(dtype="f32"))
        initializers["cb"] = IrInitializer(tensor="enc.cb", storage=IrStorage(dtype="f32"))
        values["cw"] = IrValue(dtype="f32", shape=list(conv_shape))
        values["cb"] = IrValue(dtype="f32", shape=[conv_shape[0]])
        values["c"] = IrValue(dtype="f32", shape=[1, conv_shape[0], 2])
        tensors["enc.cw"] = torch.randn(*conv_shape)
        tensors["enc.cb"] = torch.randn(conv_shape[0])
        outputs.append("c")
    graph = IrGraph(
        symbols=["T"],
        inputs=inputs,
        outputs=outputs,
        initializers=initializers,
        values=values,
        nodes=nodes,
    )
    scales = {}
    for key in ["enc.w"] + (["enc.ww"] if wide else []) + (["enc.cw"] if conv else []):
        weight = tensors[key]
        if weight.numel() // weight.shape[0] % group_size:
            continue  # 端数 group（ADR 0069 決定 2）— 丸められない = 適格でもない
        scale = group_scale(weight, group_size)
        tensors[key] = dequantize_int4(quantize_to_int4(weight, scale), scale)
        scales[key] = scale
    return graph, tensors, scales


def write_int4(path, graph: IrGraph, tensors, scales):
    return write_component(path, graph, tensors, weight_dtype="i4", weight_scales=scales)


def int4_embedding_graph() -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """linear と embedding の**両方**を group 対称 i4 で丸めたグラフ（embedding 追補の正の形）。

    embedding 表 `[V,D]` は `channel_rows` が恒等なので、group scale も `[V, D/group]` の
    「重みと同 rank・最終次元だけ group 数」— linear とまったく同じ形で格納できる。
    """
    graph, tensors, scales = int4_weight_graph(embedding=True)
    emb_scale = group_scale(tensors["enc.emb"], 16)
    tensors["enc.emb"] = dequantize_int4(quantize_to_int4(tensors["enc.emb"], emb_scale), emb_scale)
    scales["enc.emb"] = emb_scale
    return graph, tensors, scales


class TestI4Eligibility:
    """i4 の適格は **linear / embedding / conv1d の重みスロット限定**（ADR 0069 決定 5 +
    embedding / conv1d 追補 — i4 の展開経路を持つカーネルはこの 3 つだけ）。

    conv1d は展開できるのが igemm 変種だけなので `groups == 1` が要り、さらに格納行
    （`Cin·K` の平坦化）が group 長で割り切れることが要る。
    """

    def test_linear_and_embedding_weight_slots_are_listed(self):
        graph, _, _ = int4_weight_graph(embedding=True)

        assert eligible_compressed_initializers(graph) == {"w", "emb"}
        assert i4_eligible_initializers(graph) == {"w", "emb"}

    def test_a_dense_conv1d_weight_is_i4_eligible(self):
        """`groups == 1` の conv1d は i4 の適格（波 J-5b の追補）。

        行長は最終次元（K = 16）ではなく**受容野の平坦化**（`Cin·K` = 32）— group 16 で
        2 group に割れる。K だけを見ていると `[3,2,16]` が「16 % 16 == 0」で通ってしまい、
        受容野をまたぐ group が別の scale で読まれる沈黙誤値になる。
        """
        graph, _, _ = int4_weight_graph(conv=True)

        assert eligible_compressed_initializers(graph) == {"w", "cw"}
        assert i4_eligible_initializers(graph, group_size=16) == {"w", "cw"}

    def test_a_grouped_conv1d_weight_is_not_i4_eligible(self):
        """`groups > 1` は direct カーネルで実行され、i4 の展開経路が無い。"""
        graph, _, _ = int4_weight_graph(conv=True, conv_groups=2)

        assert eligible_compressed_initializers(graph) == {"w", "cw"}
        assert i4_eligible_initializers(graph, group_size=16) == {"w"}

    def test_a_conv1d_without_a_declared_groups_attr_fails_loudly(self):
        """`groups` の宣言が無い形に既定値 1 を補わない（ADR 0012・ランタイム側と同じ扱い）。

        黙って 1 を仮定すると depthwise の重みが i4 で常駐し、direct カーネルが packed バイトを
        f32 として読む（例外の出ない沈黙誤値）。
        """
        graph, _, _ = int4_weight_graph(conv=True)
        without_groups = {key: value for key, value in conv1d_attrs().items() if key != "groups"}
        graph.nodes[-1] = replace(graph.nodes[-1], attrs=without_groups)

        with pytest.raises(OpContractError, match=re.escape("attrs.groups")):
            i4_eligible_initializers(graph, group_size=16)

    def test_a_conv1d_whose_flattened_row_is_not_divisible_is_not_i4_eligible(self):
        """行長 `Cin·K` = 24 は group 16 で端数 group を作る（ADR 0069 決定 2）。

        適格に入れてしまうと「fake-quant が届いていない重み」として export 全体が落ちる —
        linear と割り切れない conv1d を持つ普通のグラフが i4 で書けなくなる。
        """
        graph, _, _ = int4_weight_graph(conv=True, conv_shape=(3, 3, 8))

        assert eligible_compressed_initializers(graph) == {"w", "cw"}
        assert i4_eligible_initializers(graph, group_size=16) == {"w"}

    def test_a_conv_transpose1d_weight_is_not_i4_eligible(self):
        """転置レイアウト `[Cin,Cout,K]` は行軸が先頭でない = pack 順と行の並びが食い違う。

        2026-08-20 のユーザー裁定で permuted pack は買わないので、展開経路ごと持たない。
        """
        graph, _, _ = int4_weight_graph(conv=True, conv_op="conv_transpose1d")

        assert eligible_compressed_initializers(graph) == {"w", "cw"}
        assert i4_eligible_initializers(graph, group_size=16) == {"w"}

    def test_a_weight_shared_with_another_weight_slot_is_excluded(self):
        """i4 の経路を持たない重みスロットとも共有される重みは適格に入らない
        （展開経路が 1 つに決まらない）。"""
        graph, _, _ = int4_weight_graph()
        graph.nodes.append(
            IrNode(op="conv1d", ins=["x", "w", "b"], outs=["c"], attrs=conv1d_attrs(groups=2))
        )

        assert i4_eligible_initializers(graph, group_size=16) == set()


class TestI4Storage:
    def test_eligible_linear_weights_are_stored_as_i4_with_a_group_scale(self, tmp_path):
        graph, tensors, scales = int4_weight_graph()

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

        declared = written_graph(path).initializers
        assert declared["w"].storage.dtype == "i4"
        assert declared["w"].storage.scale == "karume.scale.enc.w"
        assert declared["w"].storage.group_size == 16
        # MUST: bias は常に f32（プロトタイプの降格バグの根治形 — ADR 0006）。
        assert declared["b"].storage.dtype == "f32"
        header = container_header(weight_shard(path))
        assert header["enc.w"]["dtype"] == "I4"
        assert header["enc.w"]["shape"] == [3, 32]  # shape は論理形のまま
        begin, end = header["enc.w"]["data_offsets"]
        assert end - begin == 3 * 32 // 2  # バイト長だけが bit 幅から決まる
        assert header["karume.scale.enc.w"]["shape"] == [3, 2]
        assert header["karume.scale.enc.w"]["dtype"] == "F32"

    def test_the_stored_bytes_reconstruct_the_weight_bit_for_bit(self, tmp_path):
        """`dequant(unpack(格納バイト))` が fake-quant 済みの重みと**ビット一致**する。

        emit の門（`_convert_for_storage`）と同じ式だが、こちらは**ファイルに出たバイト**から
        辿る — 書き出しの経路（順序・offset・memoryview の cast）まで含めて対応を固定する。
        """
        graph, tensors, scales = int4_weight_graph()
        expected = tensors["enc.w"].clone()

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

        packed = stored_payload(path, "enc.w")
        restored = dequantize_int4(unpack_int4(packed, (3, 32)), scales["enc.w"])
        assert torch.equal(restored, expected)

    def test_the_emitted_i4_file_satisfies_the_reader_layout_rules(self, tmp_path):
        graph, tensors, scales = int4_weight_graph()

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

        # 例外が出なければ合格（I4 の先頭 4 バイト整列を含む）。
        assert_reader_layout(weight_shard(path))

    def test_the_i4_file_passes_the_full_verification(self, tmp_path):
        """emit → verify_model の往復が i4 で最後まで通る（実行 capability は第 3 便で開放済み）。

        ここが緑なのは、自前リーダ（`verify._read_container`）が I4 コンテナを読めていること
        そのもの — `safetensors` の `safe_open` はこのファイルを開けない。宣言・shape・
        group 形 scale・runtime support・op 契約の全規則を通る。
        """
        graph, tensors, scales = int4_weight_graph()

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

        expected = compressed_view(graph, {"w": "i4"}, group_size=16)
        assert verify_model(path).to_dict() == expected.to_dict()

    def test_the_breakdown_counts_i4_bytes_and_the_group_scale_overhead(self, tmp_path):
        graph, tensors, scales = int4_weight_graph()

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)
        breakdown = storage_breakdown(written_graph(path))

        assert breakdown.compressed_tensors == 1
        assert breakdown.compressed_bytes == 3 * 32 // 2  # i4 = 0.5 バイト/要素
        # scale は group 数（3 行 × 2 group）× 4 バイト。
        assert breakdown.scale_bytes == 3 * 2 * 4
        assert breakdown.plain_tensors == 1
        assert breakdown.plain_bytes == 3 * 4

    def test_a_dense_conv1d_weight_is_stored_as_i4_with_a_rank2_group_scale(self, tmp_path):
        """`groups == 1` の conv1d `[3,2,16]` は i4 格納・scale は **rank2**（波 J-5b）。

        scale の形は rank に依らず「行数 = 先頭次元・最終次元 = 行長 / group 長」
        （ADR 0069 決定 3 の rank 非依存規則）— 重みと同 rank の `[3,2,1]` ではない。
        """
        graph, tensors, scales = int4_weight_graph(conv=True)

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

        declared = written_graph(path).initializers
        assert declared["cw"].storage.dtype == "i4"
        assert declared["cw"].storage.group_size == 16
        header = container_header(weight_shard(path))
        assert header["enc.cw"]["dtype"] == "I4"
        assert header["enc.cw"]["shape"] == [3, 2, 16]  # shape は論理形のまま
        begin, end = header["enc.cw"]["data_offsets"]
        assert end - begin == 3 * 2 * 16 // 2
        assert header["karume.scale.enc.cw"]["shape"] == [3, 2]
        assert declared["cb"].storage.dtype == "f32", "bias は常に f32（ADR 0006）"

    def test_the_stored_conv1d_bytes_reconstruct_the_rank3_weight_bit_for_bit(self, tmp_path):
        """rank3 でも `dequant(unpack(格納バイト))` が丸め済みの重みと**ビット一致**する。

        payload の nibble 順は重みの連続メモリ順（`[O,Cin,K]` の row-major = `[O,Cin·K]` の
        平坦と同一バイト列）— 行の畳み方を取り違えても形も型もバイト長も合うので、この門が
        唯一の検出点（ADR 0069 決定 4 ③）。
        """
        graph, tensors, scales = int4_weight_graph(conv=True)
        expected = tensors["enc.cw"].clone()

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

        packed = stored_payload(path, "enc.cw")
        restored = dequantize_int4(unpack_int4(packed, (3, 2, 16)), scales["enc.cw"])
        assert torch.equal(restored, expected)

    def test_the_conv1d_i4_file_passes_the_full_verification(self, tmp_path):
        """emit → verify_model の往復が rank3 の i4 でも最後まで通る。

        group 形の検査（`_check_group_quantized_shape` / `_assert_scale_tensor`）が rank2 の
        scale を rank3 の重みに対して受理する — ここが同 rank を要求したままだと、書けた
        ファイルが読めない。
        """
        graph, tensors, scales = int4_weight_graph(conv=True)

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

        declared = verify_model(path).initializers
        assert declared["cw"].storage.dtype == "i4"
        assert declared["w"].storage.dtype == "i4"

    def test_a_grouped_conv1d_weight_stays_f32(self, tmp_path):
        """適格外の conv1d は i4 にせず f32 のまま残す（Codex 波 F 指摘 I4-ELIG-01）。

        ここで export 全体を落とすと、linear + depthwise conv を持つ普通のグラフが i4 で
        書けなくなる。ランタイム側の受け皿（eligible ∩ i4Eligible の外は CPU 展開）と対の
        設計は「展開経路のある op の重みだけ圧縮・他は f32」で、i8 の適格外と同じ。
        """
        graph, tensors, scales = int4_weight_graph(conv=True, conv_groups=2)

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

        declared = written_graph(path).initializers
        assert declared["w"].storage.dtype == "i4"
        assert declared["cw"].storage.dtype == "f32"

    def test_a_row_only_the_shipped_group_length_divides_is_still_stored_as_i4(self, tmp_path):
        """g16 で丸めた行長 48 の linear も i4 で格納される（除数は出荷の実 g）。

        適格判定の除数を既定 g（32）に固定すると、`48 % 32 != 0` でこの重みだけが適格から
        外れ、値は i4 グリッドへ丸められたまま **f32 で格納**される — 行長 32 の linear が
        1 本でもあれば「適格 0 本」の門も鳴らないので、品質劣化だけ乗ってサイズが縮まない
        配布形が例外も診断も無しに出る（ADR 0006 の「圧縮指定なのに実質 f32」）。
        """
        graph, tensors, scales = int4_weight_graph(wide=True)

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

        declared = verify_model(path).initializers
        assert declared["w"].storage.dtype == "i4", "行長 32 側（既定 g でも割り切れる）"
        assert declared["ww"].storage.dtype == "i4", "行長 48 側（出荷 g 16 でだけ割り切れる）"
        assert declared["ww"].storage.group_size == 16
        # scale は 48 / 16 = 3 group（既定 g を使っていれば宣言そのものが立たない）。
        assert container_header(weight_shard(path))["karume.scale.enc.ww"]["shape"] == [3, 3]

    def test_an_embedding_table_is_stored_as_i4_with_a_group_scale(self, tmp_path):
        """embedding 表 `[V,D]` も i4 で格納される（ADR 0069 決定 5 の embedding 追補）。

        scale は linear とまったく同じ group 形（同 rank・最終次元だけ group 数）で、
        宣言も `storage.group_size` を持つ — 席が増えただけで格納規則は 1 つ。
        """
        graph, tensors, scales = int4_embedding_graph()

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

        declared = written_graph(path).initializers
        assert declared["emb"].storage.dtype == "i4"
        assert declared["emb"].storage.scale == "karume.scale.enc.emb"
        assert declared["emb"].storage.group_size == 16
        header = container_header(weight_shard(path))
        assert header["enc.emb"]["dtype"] == "I4"
        assert header["enc.emb"]["shape"] == [3, 32]  # shape は論理形のまま
        begin, end = header["enc.emb"]["data_offsets"]
        assert end - begin == 3 * 32 // 2
        assert header["karume.scale.enc.emb"]["shape"] == [3, 2]
        assert header["karume.scale.enc.emb"]["dtype"] == "F32"

    def test_the_stored_embedding_bytes_reconstruct_the_table_bit_for_bit(self, tmp_path):
        """embedding 表でも `dequant(unpack(格納バイト))` が丸め済みの重みとビット一致する。"""
        graph, tensors, scales = int4_embedding_graph()
        expected = tensors["enc.emb"].clone()

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

        packed = stored_payload(path, "enc.emb")
        restored = dequantize_int4(unpack_int4(packed, (3, 32)), scales["enc.emb"])
        assert torch.equal(restored, expected)

    def test_the_emitted_embedding_i4_file_satisfies_the_reader_layout_rules(self, tmp_path):
        """I4 が 2 本並んでも「隙間なく・4 バイト整列」が保たれる（ADR 0069 追記 2 の並び）。"""
        graph, tensors, scales = int4_embedding_graph()

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

        assert_reader_layout(weight_shard(path))

    def test_the_embedding_i4_file_passes_the_full_verification(self, tmp_path):
        """emit → verify_model の往復が embedding の i4 でも最後まで通る。

        group 形の検査（`_check_group_quantized_shape` — 最終次元 % group_size）は
        embedding `[V,D]` の D 軸でそのまま成立する。
        """
        graph, tensors, scales = int4_embedding_graph()

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

        expected = compressed_view(graph, {"w": "i4", "emb": "i4"}, group_size=16)
        assert verify_model(path).to_dict() == expected.to_dict()

    def test_the_breakdown_counts_both_i4_tensors(self, tmp_path):
        """内訳の i4 バイトは linear と embedding の 2 本ぶん（0.5 バイト / 要素）。"""
        graph, tensors, scales = int4_embedding_graph()

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)
        breakdown = storage_breakdown(written_graph(path))

        assert breakdown.compressed_tensors == 2
        assert breakdown.compressed_bytes == 2 * (3 * 32 // 2)
        assert breakdown.scale_bytes == 2 * (3 * 2 * 4)
        assert breakdown.plain_tensors == 1
        assert breakdown.plain_bytes == 3 * 4

    def test_a_graph_whose_only_weight_is_shared_with_conv_fails_loudly(self, tmp_path):
        """linear と depthwise conv で共有された重みしか無い形は「適格 0 本」で落ちる。"""
        graph, tensors, scales = int4_weight_graph()
        graph.nodes.append(
            IrNode(op="conv1d", ins=["x", "w", "b"], outs=["c"], attrs=conv1d_attrs(groups=2))
        )
        graph.values["c"] = IrValue(dtype="f32", shape=["T", 3])
        graph.outputs.append("c")

        with pytest.raises(EmitError, match="圧縮格納が 1 本も計画されなかった"):
            write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

    def test_a_missing_scale_fails_loudly(self, tmp_path):
        """適格なのに scale が無い = fake-quant が届いていない重み（ADR 0006）。"""
        graph, tensors, _ = int4_weight_graph()

        with pytest.raises(EmitError, match="group scale が無い"):
            write_int4(tmp_path / "model.safetensors", graph, tensors, {})

    def test_a_scale_that_is_not_in_group_form_fails_loudly(self, tmp_path):
        graph, tensors, _ = int4_weight_graph()

        with pytest.raises(EmitError, match="group 形"):
            write_int4(tmp_path / "model.safetensors", graph, tensors, {"enc.w": torch.ones(1, 2)})

    def test_a_group_size_outside_the_accepted_set_fails_loudly(self, tmp_path):
        """scale `[3,4]` は group 8 を意味する — 2 冪だが 16 未満（ADR 0069 決定 2）。

        書いてから `verify` で落とすのでは「読めないファイルを配布形に残す」ので、
        書く前の計画段で落とす。
        """
        graph, tensors, _ = int4_weight_graph()

        with pytest.raises(EmitError, match="2 冪かつ 16 以上でない"):
            write_int4(tmp_path / "model.safetensors", graph, tensors, {"enc.w": torch.ones(3, 4)})

    def test_a_scale_that_is_not_f32_fails_loudly(self, tmp_path):
        """companion scale は F32 固定（i8 と同じ理由 — 逆変換の等値検査では検出できない）。"""
        graph, tensors, scales = int4_weight_graph()
        half = {key: value.to(torch.float16) for key, value in scales.items()}

        with pytest.raises(EmitError, match="scale の dtype"):
            write_int4(tmp_path / "model.safetensors", graph, tensors, half)

    def test_an_unrounded_eligible_weight_fails_loudly(self, tmp_path):
        """丸めの掛け忘れ / 順序の誤りは黙って通さない（ADR 0006）。"""
        graph, tensors, scales = int4_weight_graph()
        tensors["enc.w"] = torch.full((3, 32), 1.0 / 3.0)

        with pytest.raises(EmitError, match="ビット一致しない"):
            write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

    def test_a_scale_that_is_not_the_one_fake_quant_used_fails_loudly(self, tmp_path):
        """fake-quant が使ったのと**別の** scale で書こうとすると逆変換ゲートが落ちる。"""
        graph, tensors, scales = int4_weight_graph()
        drifted = {key: value * 1.0000002 for key, value in scales.items()}

        with pytest.raises(EmitError, match="ビット一致しない"):
            write_int4(tmp_path / "model.safetensors", graph, tensors, drifted)

    def test_a_tampered_packed_byte_is_caught_by_the_round_trip_gate(self, tmp_path, monkeypatch):
        """故障注入: packed バイトを 1 個だけ書き換えると逆変換ビット一致門が発火する
        （ADR 0069 決定 4 ③）。

        shape も dtype もバイト長も変わらないので、この門以外に検出点は無い。
        """
        from karume import emit

        pack = emit.pack_int4

        def tamper(quantized):
            packed = pack(quantized)
            packed[0] ^= 0x10  # 上位 nibble を 1 段ずらす
            return packed

        monkeypatch.setattr(emit, "pack_int4", tamper)
        graph, tensors, scales = int4_weight_graph()

        with pytest.raises(EmitError, match="ビット一致しない"):
            write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

    def test_a_graph_without_eligible_weights_fails_loudly(self, tmp_path):
        graph, tensors = sample_graph()

        with pytest.raises(EmitError, match="圧縮格納が 1 本も計画されなかった"):
            write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="i4")


class TestFailureLeavesTheGraphUntouched:
    """1 本でも落ちたら宣言は 1 つも書き換えない（commit は全バイトを書き終えた後）。

    適格な重みは 2 本（`emb` / `w`）あり、以下はどれも「片方は通る・もう片方で落ちる」形。
    落ちた時点で書き換えていると、書き出しが例外で止まっても呼び出し側の `graph` に片方だけ
    圧縮宣言が残り、その graph を使い回す経路（同じ graph を別 dtype で書き直す / 内訳を
    数える）が黙って宣言と実体の食い違った答えを出す。
    """

    def test_a_failed_f16_pass_leaves_no_partial_declaration(self, tmp_path):
        graph, tensors = weight_graph()
        tensors["enc.w"] = torch.full((3, 4), 1.0 / 3.0)
        before = json.loads(graph.to_json())

        with pytest.raises(EmitError, match="fake-quant"):
            write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        assert graph.initializers["emb"].storage.dtype == "f32"
        assert json.loads(graph.to_json()) == before

    def test_a_failed_i8_pass_leaves_no_partial_declaration(self, tmp_path):
        graph, tensors, scales = int8_weight_graph()
        del scales["enc.w"]
        before = json.loads(graph.to_json())

        with pytest.raises(EmitError, match="per-channel scale が無い"):
            write_model(
                tmp_path / "model.safetensors",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=scales,
            )

        assert graph.initializers["emb"].storage.dtype == "f32"
        assert json.loads(graph.to_json()) == before

    def test_a_failure_during_the_data_section_leaves_no_declaration(self, tmp_path):
        """逆変換ゲートは**データ節を書きながら**踏む（他の本は既にファイルへ出ている）。

        書きかけのファイルは残るが（配布物の原子性は `pipeline.export_to_file` の一時ファイル
        層が持つ）、呼び出し側の `graph` は 1 つも書き換わっていてはいけない。
        """
        graph, tensors, scales = int8_weight_graph()
        tensors["enc.w"] = torch.full((3, 4), 1.0 / 3.0)
        before = json.loads(graph.to_json())
        path = tmp_path / "model.safetensors"

        with pytest.raises(EmitError, match="ビット一致しない"):
            write_model(path, graph, tensors, weight_dtype="i8", weight_scales=scales)

        # 書きかけの shard は残る — 捨てるのは呼び出し側の層（代表 path 自身は書かれない）。
        assert shard_siblings(path)
        assert not path.exists()
        assert json.loads(graph.to_json()) == before


class TestSuccessLeavesTheGraphUntouched:
    """書き出しが成功しても呼び出し側の `graph` は 1 バイトも変わらない。

    commit を呼び手の graph へ書き戻していると、同じ graph を別 dtype で書き直す経路
    （`write_model` の docstring が再利用経路として名指ししている形）で前回の圧縮宣言が
    残る — f32 の計画は空プランで宣言を**復元しない**ので、2 本目は「宣言 f16 / 実体 F32」の
    壊れたコンテナになる。
    """

    def test_a_successful_f16_write_leaves_the_declarations_alone(self, tmp_path):
        graph, tensors = weight_graph()
        before = json.loads(graph.to_json())

        write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        assert json.loads(graph.to_json()) == before

    def test_writing_the_same_graph_as_f32_after_f16_stays_readable(self, tmp_path):
        graph, tensors = weight_graph()

        write_model(tmp_path / "half.safetensors", graph, tensors, weight_dtype="f16")
        plain = write_component(tmp_path / "plain.safetensors", graph, tensors)

        with safe_open(str(weight_shard(plain)), framework="pt") as handle:
            assert handle.get_slice("enc.w").get_dtype() == "F32"
        # 宣言と実体が食い違っていれば verify_model がここで落ちる。
        assert verify_model(plain).to_dict() == graph.to_dict()


class TestStreamingConversion:
    """圧縮変換は**書く直前に 1 本ずつ**掛ける（ADR 0018 / 0019 の格納そのものは不変）。

    全件を先に変換すると、圧縮側の集合が呼び出し側の f32 集合と同時に生きてピーク RAM が
    両者の和になる（Irodori 規模で f32 3.44GB に f16 1.72GB / i8 0.87GB が重なる）。
    """

    def test_only_one_converted_tensor_is_alive_at_a_time(self, tmp_path, monkeypatch):
        """故障注入: 変換済みを溜めてから書く実装へ戻すと、前の 1 本が生き残って落ちる。"""
        from karume import emit

        alive: list[weakref.ReferenceType] = []
        convert = emit._convert_for_storage

        def spy(key, tensor, conversion):
            # 次の 1 本へ入る時点で、前の変換済みは解放されているはず。
            assert [ref for ref in alive if ref() is not None] == []
            converted = convert(key, tensor, conversion)
            alive.append(weakref.ref(converted))
            return converted

        monkeypatch.setattr(emit, "_convert_for_storage", spy)
        graph, tensors = weight_graph()

        write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        assert len(alive) == 2  # enc.w / enc.emb

    def test_the_streamed_f16_bytes_match_a_pre_converted_write(self, tmp_path):
        """流しながら書いたバイト列 = 全件を先に変換してから素で書いたバイト列。

        ヘッダ（dtype / shape / data_offsets）を**変換前**の形と計画だけから導いているので、
        導出を間違えるとここで offset ごとずれる。
        """
        from karume.emit import _save_ordered, _write_order

        graph, tensors = weight_graph()
        compressed = {"enc.w", "enc.emb"}
        pre = {
            key: value.to(torch.float16) if key in compressed else value
            for key, value in tensors.items()
        }

        streamed = write_component(
            tmp_path / "streamed.safetensors", graph, tensors, weight_dtype="f16"
        )

        # 対照は shard 2 本ぶん — グラフ shard（commit 済みの宣言ビューだけ）と weight shard
        # （テンソルだけ）。呼び手の graph は f32 のままなので、宣言は書き出し側が commit する。
        committed = compressed_view(graph, {"w": "f16", "emb": "f16"})
        reference_graph = tmp_path / "reference-graph.safetensors"
        reference = tmp_path / "reference.safetensors"
        _save_ordered(reference_graph, {}, [], {IR_METADATA_KEY: committed.to_json()})
        _save_ordered(reference, pre, _write_order(pre), {})
        assert graph_shard(streamed).read_bytes() == reference_graph.read_bytes()
        assert weight_shard(streamed).read_bytes() == reference.read_bytes()

    def test_the_streamed_i8_bytes_match_a_pre_converted_write(self, tmp_path):
        """i8 も同じ（companion scale が増える形と I8 群の順序を踏む）。"""
        from karume.emit import _save_ordered, _write_order

        graph, tensors, scales = int8_weight_graph()
        pre = dict(tensors)
        for key, scale in scales.items():
            pre[key] = quantize_to_int8(tensors[key], scale)
            pre[f"karume.scale.{key}"] = scale

        streamed = write_component(
            tmp_path / "streamed.safetensors",
            graph,
            tensors,
            weight_dtype="i8",
            weight_scales=scales,
        )

        committed = compressed_view(graph, {"w": "i8", "emb": "i8"})
        reference_graph = tmp_path / "reference-graph.safetensors"
        reference = tmp_path / "reference.safetensors"
        _save_ordered(reference_graph, {}, [], {IR_METADATA_KEY: committed.to_json()})
        _save_ordered(reference, pre, _write_order(pre), {})
        assert graph_shard(streamed).read_bytes() == reference_graph.read_bytes()
        assert weight_shard(streamed).read_bytes() == reference.read_bytes()


def data_layout(path) -> list[tuple[int, str, str]]:
    """safetensors のデータ節に並ぶ順（開始 offset / 名前 / dtype）。"""
    header = container_header(weight_shard(path))
    entries = [
        (entry["data_offsets"][0], name, entry["dtype"])
        for name, entry in header.items()
        if name != "__metadata__"
    ]
    return sorted(entries)


class TestWriteOrder:
    """奇数要素の F16 の**後ろ**に 4 バイト型を置くと Karume のリーダが読めない。

    並べ替えはエクスポータの責務（docs/limitations.md）なので、書き出し順で担保する。
    """

    def test_odd_element_f16_tensors_are_written_last(self, tmp_path):
        graph, tensors = weight_graph()

        path = write_component(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        # enc.emb は 15 要素 = 30 バイト（≡ 2 mod 4）なので末尾へ寄る。
        assert [name for _, name, _ in data_layout(path)][-1] == "enc.emb"

    def test_i8_tensors_are_written_after_everything_else(self, tmp_path):
        """I8 は要素サイズ 1 = 任意のバイト長を作るので、群の**末尾**に置く（ADR 0019）。

        前に置くと、その後ろの F32 / I32 / F16 の絶対 offset が要素サイズの倍数から外れて
        Karume のリーダが読めなくなる（HF の `safe_open` は読めてしまう）。
        """
        graph, tensors, scales = int8_weight_graph()

        path = write_component(
            tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8", weight_scales=scales
        )

        dtypes = [dtype for _, _, dtype in data_layout(path)]
        assert dtypes[-2:] == ["I8", "I8"]
        assert "I8" not in dtypes[:-2]

    def test_i4_tensors_are_written_with_the_four_byte_aligned_group(self, tmp_path):
        """並びは**整列単位の降順** — F32 / I32 / I4 → F16 → I8（ADR 0069 追記 2）。

        I4 は要素整列の概念を持たず「テンソル先頭が 4 バイト整列」を要求するので、F16 / I8 の
        後ろへ置くと絶対 offset が 4 の倍数から外れる。逆に I4 節のバイト長は必ず 8 の倍数
        （量子化軸が 2 冪 ≥ 16 の group で割り切れる ⇒ 要素数は 16 の倍数）なので、F32 / I32 と
        同じ群に前置しても後続の整列を崩さない。
        """
        from karume.emit import _Conversion, _write_order

        tensors = {
            "plain": torch.randn(4),
            "sym": torch.zeros(2, dtype=torch.int32),
            "packed": torch.randn(32),
            "odd": torch.randn(3).to(torch.float16),
            "bytes": torch.zeros(3, dtype=torch.int8),
        }
        conversions = {"packed": _Conversion(dtype="i4", name="w", scale=torch.ones(2))}

        order = _write_order(tensors, conversions)

        assert order == ["plain", "sym", "packed", "odd", "bytes"]

    def test_the_emitted_i4_file_places_the_packed_weight_after_the_f32_tensors(self, tmp_path):
        graph, tensors, scales = int4_weight_graph()

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

        dtypes = [dtype for _, _, dtype in data_layout(path)]
        assert dtypes == ["F32", "F32", "I4"]
        assert_reader_layout(weight_shard(path))

    def test_the_emitted_file_satisfies_the_reader_layout_rules(self, tmp_path):
        graph, tensors = weight_graph()

        path = write_component(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        assert_reader_layout(weight_shard(path))  # 例外が出なければ合格

    def test_the_emitted_i8_file_satisfies_the_reader_layout_rules(self, tmp_path):
        graph, tensors, scales = int8_weight_graph()

        path = write_component(
            tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8", weight_scales=scales
        )

        assert_reader_layout(weight_shard(path))  # 例外が出なければ合格

    def test_an_i8_tensor_placed_first_is_caught_by_the_reader_rules(self, tmp_path):
        """故障注入: 奇数長の I8 を先頭へ置くと後続 F32 の絶対 offset が 4 の倍数から外れる。"""
        from karume.emit import _save_ordered

        path = tmp_path / "broken.safetensors"
        tensors = {"packed": torch.zeros(3, dtype=torch.int8), "plain": torch.randn(4)}
        _save_ordered(path, tensors, ["packed", "plain"], {})

        with safe_open(str(path), framework="pt") as handle:  # 他のリーダは読める
            assert handle.get_tensor("plain").shape == (4,)
        with pytest.raises(ContainerError, match="整列していない"):
            assert_reader_layout(path)

    def test_a_deliberately_broken_order_is_caught_by_the_reader_rules(self, tmp_path):
        """故障注入: 奇数 F16 を先頭へ置くと絶対 offset が 4 の倍数から外れる。

        HF の `safe_open` はこの並びを**読めてしまう**ので、エクスポータ側で
        Karume のリーダ規則を写して検査する必要がある（片方だけでは検出できない）。
        """
        from karume.emit import _save_ordered

        path = tmp_path / "broken.safetensors"
        tensors = {"odd": torch.randn(3).to(torch.float16), "plain": torch.randn(4)}
        _save_ordered(path, tensors, ["odd", "plain"], {})

        with safe_open(str(path), framework="pt") as handle:  # 他のリーダは読める
            assert handle.get_tensor("plain").shape == (4,)
        with pytest.raises(ContainerError, match="整列していない"):
            assert_reader_layout(path)

    def test_f32_only_files_keep_the_bytes_that_save_file_produces(self, tmp_path):
        """f16 を含まない資産は writer を差し替えてもバイト列が変わらない。

        f32 系列（outputs/series/anima/ 等）を再生成したときに、順序の変更だけで全ファイルが
        別バイトになるのを避けるための不変条件。
        """
        graph, tensors = weight_graph()
        reference = tmp_path / "reference.safetensors"
        # 対照は weight shard（テンソルだけ）— 宣言はグラフ shard 側に居る（ADR 0081）。
        # `metadata={}` は weight shard と同じ空の `__metadata__` 欄を作る。
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(reference), metadata={})

        path = write_component(tmp_path / "model.safetensors", graph, tensors)

        assert weight_shard(path).read_bytes() == reference.read_bytes()


class TestJsonCompliance:
    def test_non_finite_values_are_refused_at_serialization(self):
        """NaN / Infinity は JSON の標準リテラルに無い — 書き出しの時点で失敗させる。"""
        graph, _ = sample_graph()
        graph.nodes[0] = IrNode(op="add", ins=["x", "w"], outs=["y"], attrs={"value": float("inf")})

        with pytest.raises(ValueError, match="JSON compliant"):
            graph.to_json()

    def test_ordinary_graphs_serialize_to_standard_json(self):
        graph, _ = sample_graph()

        restored = json.loads(graph.to_json(), parse_constant=lambda lit: pytest.fail(lit))

        assert restored["requires"]["ops"] == ["add"]


def mixed_weight_graph() -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """linear = i4 / embedding = i8 の混成向けに両方を fake-quant 済みにしたグラフ。

    LLM の「embedding / tied lm_head は i8・linear 本体は i4」（Gemma 4 E2B が初出）の最小形。
    """
    graph, tensors, scales = int4_weight_graph(embedding=True)
    emb_scale = channel_scale(tensors["enc.emb"], 0)
    tensors["enc.emb"] = quantize_to_int8(tensors["enc.emb"], emb_scale).to(torch.float32) * (
        emb_scale
    )
    scales["enc.emb"] = emb_scale
    return graph, tensors, scales


class TestMixedStorage:
    """`weight_dtype_overrides` — 1 本単位の明示指定（既定に優先・満たせなければ fail loudly）。"""

    def test_an_override_mixes_i8_into_an_i4_default(self, tmp_path):
        graph, tensors, scales = mixed_weight_graph()

        path = write_component(
            tmp_path / "model.safetensors",
            graph,
            tensors,
            weight_dtype="i4",
            weight_scales=scales,
            weight_dtype_overrides={"enc.emb": "i8"},
        )

        declared = verify_model(path).initializers
        assert declared["w"].storage.dtype == "i4"
        assert declared["w"].storage.group_size == 16
        assert declared["emb"].storage.dtype == "i8"
        assert declared["emb"].storage.scale == "karume.scale.enc.emb"
        assert declared["b"].storage.dtype == "f32"
        header = container_header(weight_shard(path))
        assert header["enc.w"]["dtype"] == "I4"
        assert header["enc.emb"]["dtype"] == "I8"

    def test_overrides_compress_even_with_a_f32_default(self, tmp_path):
        """既定 f32 は従来「空プランで即返し」だった — 明示指定だけの圧縮も通ること。"""
        graph, tensors, scales = mixed_weight_graph()

        path = write_component(
            tmp_path / "model.safetensors",
            graph,
            tensors,
            weight_scales=scales,
            weight_dtype_overrides={"enc.emb": "i8"},
        )

        declared = verify_model(path).initializers
        assert declared["emb"].storage.dtype == "i8"
        assert declared["w"].storage.dtype == "f32"

    def test_an_override_to_f32_exempts_a_weight_from_the_default(self, tmp_path):
        graph, tensors = weight_graph()

        path = write_component(
            tmp_path / "model.safetensors",
            graph,
            tensors,
            weight_dtype="f16",
            weight_dtype_overrides={"enc.w": "f32"},
        )

        declared = verify_model(path).initializers
        assert declared["w"].storage.dtype == "f32"
        assert declared["emb"].storage.dtype == "f16"

    def test_an_unknown_override_key_fails_loudly(self, tmp_path):
        graph, tensors = weight_graph()

        with pytest.raises(EmitError, match="どの initializer のテンソルでもない"):
            write_model(
                tmp_path / "model.safetensors",
                graph,
                tensors,
                weight_dtype="f16",
                weight_dtype_overrides={"enc.typo": "f16"},
            )

    def test_an_override_on_an_ineligible_weight_fails_loudly(self, tmp_path):
        """既定 dtype は適格外を静かに f32 へ残すが、明示指定は黙って別の格納にしない。"""
        graph, tensors = weight_graph(share_weight=True)

        with pytest.raises(EmitError, match="適格でない"):
            write_model(
                tmp_path / "model.safetensors",
                graph,
                tensors,
                weight_dtype_overrides={"enc.w": "f16"},
            )

    def test_an_explicit_i4_on_a_conv_weight_fails_loudly(self, tmp_path):
        """適格外の conv（ここでは depthwise）は一般適格でも i4 の展開経路が無い。

        既定は静かに f32 へ残すが、明示指定は fail loudly（`_plan_weight_dtype` の線引き）。
        """
        graph, tensors, scales = int4_weight_graph(conv=True, conv_groups=2)

        with pytest.raises(EmitError, match="の重みスロットだけ"):
            write_model(
                tmp_path / "model.safetensors",
                graph,
                tensors,
                weight_scales=scales,
                weight_dtype_overrides={"enc.cw": "i4"},
            )

    def test_an_unknown_override_dtype_fails_loudly(self, tmp_path):
        graph, tensors = weight_graph()

        with pytest.raises(EmitError, match="書き出せない"):
            write_model(
                tmp_path / "model.safetensors",
                graph,
                tensors,
                weight_dtype_overrides={"enc.w": "f64"},
            )

    def test_an_override_on_a_non_f32_tensor_fails_loudly(self, tmp_path):
        """既定は非 f32 実体を静かに飛ばすが、明示指定は「圧縮格納は f32 実体のみ」で落ちる。

        i32 格納（記号依存定数 — ADR 0010）が重みスロットへ流れた形。意味論ごと違うので
        黙って圧縮格納にしない側の枝（他の 3 枝と対）。
        """
        graph, tensors, scales = mixed_weight_graph()
        tensors["enc.emb"] = torch.zeros(3, 32, dtype=torch.int32)

        with pytest.raises(EmitError, match="圧縮格納は f32 実体のみ"):
            write_model(
                tmp_path / "model.safetensors",
                graph,
                tensors,
                weight_scales=scales,
                weight_dtype_overrides={"enc.emb": "i8"},
            )


def conv2d_only_graph() -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """重みスロットは在るが **i4 の展開経路が無い**グラフ（conv2d 1 本 — VAE の最小形）。"""
    graph = IrGraph(
        symbols=[],
        inputs=[IrInput(name="x", dtype="f32", shape=[1, 2, 4, 4])],
        outputs=["y"],
        initializers={
            "w": IrInitializer(tensor="enc.w", storage=IrStorage(dtype="f32")),
            "b": IrInitializer(tensor="enc.b", storage=IrStorage(dtype="f32")),
        },
        values={
            "w": IrValue(dtype="f32", shape=[3, 2, 3, 3]),
            "b": IrValue(dtype="f32", shape=[3]),
            "y": IrValue(dtype="f32", shape=[1, 3, 2, 2]),
        },
        nodes=[
            IrNode(
                op="conv2d",
                ins=["x", "w", "b"],
                outs=["y"],
                attrs={"stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 1},
            )
        ],
    )
    return graph, {"enc.w": rounded(3, 2, 3, 3), "enc.b": rounded(3)}


class TestThePlannedCompressionGate:
    """「圧縮指定なのに適格 0MB」の門は **圧縮格納が 1 本も計画されなかったか**で見る
    （ADR 0006 の常設診断）。

    既定 dtype と同じ格納 dtype が 0 本でも、明示指定が適格な全件を別 dtype で覆っていれば
    意図は果たされている — その形まで落とすと、混成格納を 1 本単位で書き切った配布形が
    書けなくなる。真に 0 本の形で従来どおり落ちることは
    {@link TestF16Storage.test_a_graph_without_eligible_weights_fails_loudly} 以下 3 本が持つ。
    """

    def test_overrides_covering_every_eligible_weight_pass_the_gate(self, tmp_path):
        """既定 i8 の宣言が 1 本も無くても、明示指定の i4 が計画されていれば通る。"""
        graph, tensors, scales = int4_weight_graph()

        path = write_component(
            tmp_path / "model.safetensors",
            graph,
            tensors,
            weight_dtype="i8",
            weight_scales=scales,
            weight_dtype_overrides={"enc.w": "i4"},
        )

        assert verify_model(path).initializers["w"].storage.dtype == "i4"

    def test_the_i4_message_names_the_expansion_path(self, tmp_path):
        """i4 適格が 0 本のとき、原因は「重みスロットが無い」ではなく「展開経路が無い」。

        conv2d の重みは一般適格でも i4 で展開できるカーネルが無い（ADR 0069 決定 5）。
        「融合 op の重みを持たないグラフに i4 を指定していないか」ではグラフの構造を疑って
        空振りする。
        """
        graph, tensors = conv2d_only_graph()

        with pytest.raises(EmitError) as err:
            write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="i4")

        assert "i4 の展開経路（linear / embedding / conv1d" in str(err.value)

    def test_the_message_reports_the_default_and_the_explicit_dtypes(self, tmp_path):
        """診断は既定と明示指定の内訳を出す（どちらの指定が空振りしたかを読ませる）。"""
        graph, tensors = sample_graph()

        with pytest.raises(EmitError) as err:
            write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        assert "既定 f16" in str(err.value)
        assert "明示指定 なし" in str(err.value)


class TestInitializerKeyInjectivity:
    """宣言は名前単位・実体はキー単位 — 名前 ↔ テンソルキーが 1:1 でなければ計画段で落とす。"""

    def test_two_initializers_sharing_one_tensor_key_fail_loudly(self, tmp_path):
        """片方だけが適格だと「実体は packed / 宣言は f32」の沈黙誤値になる形。"""
        graph, tensors = weight_graph()
        graph = replace(
            graph,
            initializers={
                **graph.initializers,
                "w_alias": IrInitializer(tensor="enc.w", storage=IrStorage(dtype="f32")),
            },
            values={
                **graph.values,
                "w_alias": IrValue(dtype="f32", shape=[3, 4]),
                "a": IrValue(dtype="f32", shape=[3, 4]),
            },
            nodes=[
                *graph.nodes,
                IrNode(op="mul", ins=["w_alias", "w_alias"], outs=["a"], attrs={}),
            ],
        )

        with pytest.raises(EmitError, match="1:1 でない"):
            write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")


def fixed_int8_weight_graph() -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """`int8_weight_graph` の**決定的**版（乱数を使わない — バイト固定の門が引く）。

    値は f32 で厳密に表せる等差列で、fake-quant（`channel_scale` → `quantize_to_int8` →
    dequant）まで含めて実行のたびに同じバイト列になる。
    """
    graph, tensors = weight_graph()
    tensors["enc.w"] = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 8
    tensors["enc.b"] = torch.arange(3, dtype=torch.float32) / 4
    tensors["enc.emb"] = torch.arange(15, dtype=torch.float32).reshape(3, 5) / 16
    scales: dict[str, torch.Tensor] = {}
    for key in ("enc.w", "enc.emb"):
        scale = channel_scale(tensors[key], 0)
        tensors[key] = quantize_to_int8(tensors[key], scale).to(torch.float32) * scale
        scales[key] = scale
    return graph, tensors, scales


class TestTheWrittenShardSet:
    """コンポーネントは**常に**連番の shard 列として書かれる（ADR 0081 — 常時分割）。

    バイト列は配布物そのものなので、上限に遠く及ばない最小のコンポーネントの sha256 を
    ここで固定する（**新規則の実測値** — 旧規則の「単一ファイルでバイト不変」の pin は
    ADR 0081 が放棄した。ここが割れたら、配布リポの全ファイルが再ハッシュ・再アップロード
    になるという性格は変わらない）。
    """

    #: `sample_graph()` の (グラフ shard, weight shard) の sha256。
    PLAIN_SHA256 = (
        "5e6dee6f15f9bc6aa40319a79e7dc7b63a087c86bdd8e17b8c720c82de9c55b7",
        "638caa270359830d8bcb9d500c6fbcd9cea3018069a40d4afbef059bc5f5437f",
    )
    #: 同・`fixed_int8_weight_graph()`（i8 + companion scale の並びまで含む）。
    INT8_SHA256 = (
        "4c78a499c3d5fc5c22327fc3bfbcf4540bf81a42581e9432831741b867dff337",
        "5d41fd9468c68eb5630bbf18fe7579414a534f736573c849a9a9f889f7b8f2f1",
    )

    def test_a_small_model_is_written_as_a_numbered_pair(self, tmp_path):
        graph, tensors = sample_graph()
        final = tmp_path / "model.safetensors"

        written = write_model(final, graph, tensors)

        # 代表 path 自身は 1 バイトも書かれない（現物は連番だけ）。
        assert written == [shard_path(final, 1, 2), shard_path(final, 2, 2)]
        assert sorted(entry.name for entry in tmp_path.iterdir()) == [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]

    def test_the_plain_bytes_stay_what_they_are(self, tmp_path):
        graph, tensors = sample_graph()

        written = write_model(tmp_path / "model.safetensors", graph, tensors)

        assert tuple(hashlib.sha256(p.read_bytes()).hexdigest() for p in written) == (
            self.PLAIN_SHA256
        )

    def test_the_int8_bytes_stay_what_they_are(self, tmp_path):
        graph, tensors, scales = fixed_int8_weight_graph()

        written = write_model(
            tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8", weight_scales=scales
        )

        assert tuple(hashlib.sha256(p.read_bytes()).hexdigest() for p in written) == (
            self.INT8_SHA256
        )

    def test_a_limit_that_the_payload_exactly_fills_keeps_one_weight_shard(self, tmp_path):
        """境界は「超えたら分ける」— ちょうど収まるコンポーネントは weight shard 1 本のまま。"""
        graph, tensors = sample_graph()
        final = tmp_path / "model.safetensors"

        # `enc.w` は f32 4 要素 = 16 バイト（容量はデータ節に対する値で、ヘッダは別枠）。
        written = write_model(final, graph, tensors, _shard_capacity=16)

        assert written == [shard_path(final, 1, 2), shard_path(final, 2, 2)]
        assert tuple(hashlib.sha256(p.read_bytes()).hexdigest() for p in written) == (
            self.PLAIN_SHA256
        )


class TestShardSplitting:
    """weight shard は決定的に割り付けられる（規則の正本は `karume.shards`）。

    容量は合成の小テンソルへ人工的に下げて踏む（`_shard_capacity` — テスト専用の差し込み。
    実データで 256MiB を踏むテストは書けない）。`fixed_int8_weight_graph` の payload は
    F32 群が `enc.b` 12B → `karume.scale.enc.emb` 12B → `karume.scale.enc.w` 12B、
    I8 群が `enc.emb` 15B → `enc.w` 12B（並びは ADR 0063 の書き出し順）で、跨げない単位は
    `enc.b` 12B / `scale.emb + enc.emb` 27B / `scale.w + enc.w` 24B の 3 つ。
    """

    def split(self, tmp_path, capacity: int) -> list:
        graph, tensors, scales = fixed_int8_weight_graph()
        return write_model(
            tmp_path / "model.safetensors",
            graph,
            tensors,
            weight_dtype="i8",
            weight_scales=scales,
            _shard_capacity=capacity,
        )

    def tensors_of(self, path) -> set[str]:
        return set(container_header(path)) - {"__metadata__"}

    def test_it_writes_the_numbered_sequence_and_not_the_plain_name(self, tmp_path):
        written = self.split(tmp_path, 40)

        assert [path.name for path in written] == [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
        assert not (tmp_path / "model.safetensors").exists()

    def test_only_the_first_shard_carries_the_graph(self, tmp_path):
        """先頭 = グラフ shard・後続への `karume_ir` 再登場は ADR 0070 決定 3 で禁止。"""
        first, *rest = self.split(tmp_path, 40)

        assert IR_METADATA_KEY in container_header(first)["__metadata__"]
        # グラフ shard はテンソルを 1 本も持たない（ADR 0081 の読み手契約 1）。
        assert self.tensors_of(first) == set()
        # データ節そのものが空（ファイルは 8 バイトのヘッダ長 + ヘッダ JSON で終わる）。
        assert first.stat().st_size == 8 + struct.unpack("<Q", first.read_bytes()[:8])[0]
        for shard in rest:
            assert container_header(shard)["__metadata__"] == {}

    def test_every_shard_is_a_standalone_safetensors(self, tmp_path):
        """各 shard は自分のテンソルだけを宣言し、単体でリーダ規則を満たす（ADR 0063）。"""
        for shard in self.split(tmp_path, 40):
            assert_reader_layout(shard)

    def test_the_shards_declare_every_tensor_exactly_once(self, tmp_path):
        """宣言完全性: 全 shard の和 = 元の全テンソル（欠け・重複なし）。"""
        written = self.split(tmp_path, 40)

        declared = [name for shard in written for name in self.tensors_of(shard)]
        assert sorted(declared) == [
            "enc.b",
            "enc.emb",
            "enc.w",
            "karume.scale.enc.emb",
            "karume.scale.enc.w",
        ]
        assert len(declared) == len(set(declared))

    def test_each_weight_shares_its_shard_with_its_scale(self, tmp_path):
        """co-shard MUST（ADR 0070 決定 1）— 逐次消費は weight と scale を同時に要求する。"""
        owner = {
            name: index
            for index, shard in enumerate(self.split(tmp_path, 40))
            for name in self.tensors_of(shard)
        }

        assert owner["karume.scale.enc.w"] == owner["enc.w"]
        assert owner["karume.scale.enc.emb"] == owner["enc.emb"]

    def test_the_scale_is_pulled_forward_into_the_weights_shard(self, tmp_path):
        """対は原子 — 書き出し順で離れていても（scale は F32 群・重みは I8 群）同居する。"""
        _, first, second = self.split(tmp_path, 40)

        assert self.tensors_of(first) == {"enc.b", "karume.scale.enc.emb", "enc.emb"}
        assert self.tensors_of(second) == {"karume.scale.enc.w", "enc.w"}

    def test_a_tighter_limit_opens_more_shards(self, tmp_path):
        """対 1 つずつまで詰まる（先頭のグラフ shard は空のまま）。"""
        written = self.split(tmp_path, 30)

        assert [self.tensors_of(shard) for shard in written] == [
            set(),
            {"enc.b"},
            {"karume.scale.enc.emb", "enc.emb"},
            {"karume.scale.enc.w", "enc.w"},
        ]

    def test_the_same_input_produces_the_same_shards_byte_for_byte(self, tmp_path):
        """決定的（同入力 → 同分割・同バイト）— 再 dist が sha256 を揺らさない前提。"""
        first = self.split(tmp_path / "a", 40)
        second = self.split(tmp_path / "b", 40)

        assert [path.name for path in first] == [path.name for path in second]
        assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]

    def test_the_split_container_passes_the_full_verification(self, tmp_path):
        """読み返しの門は分割前と同じ集合を見る（宣言・scale・余剰・op 契約）。"""
        graph, _, _ = fixed_int8_weight_graph()
        written = self.split(tmp_path, 40)

        expected = compressed_view(graph, {"w": "i8", "emb": "i8"})
        assert verify_shards(written).to_dict() == expected.to_dict()

    def test_a_pair_whose_weight_cannot_be_split_fails_loudly(self, tmp_path):
        """行で割っても入らない対は fail loudly — 黙って容量を破らない。

        `enc.emb` は I8 `[3,5]` = 1 行 5 バイトなので、4 バイト整列の刻みは 4 行 = 20 バイト
        （読み手契約 5）。相方の scale 12B を差し引いた 8 バイトには 1 ブロックも入らない。
        """
        with pytest.raises(ShardError, match="これ以上細かく割れない"):
            self.split(tmp_path, 20)

    def test_it_refuses_to_read_a_shard_set_that_split_a_pair(self, tmp_path):
        """規則が作れない形も**読み返し**では受け止める（手組み / 別実装の shard 列）。

        合成の作り方は「割った上で scale だけを別の weight shard へ移す」— 書き手の割り付けを
        迂回するため、`_save_ordered` 相当をテスト側で組まずに**片方の shard から scale を
        落とし、もう片方へ足す**形にする（グラフ shard はそのまま使う）。
        """
        head, first, second = self.split(tmp_path, 40)
        # `enc.w` は second に居る。その scale だけを first と同じ集合へ動かした列を作る。
        moved = tmp_path / "moved.safetensors"
        with safe_open(str(second), framework="pt") as handle:
            payload = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
        save_file({"enc.w": payload["enc.w"]}, str(moved))
        merged = tmp_path / "merged.safetensors"
        with safe_open(str(first), framework="pt") as handle:
            kept = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
        kept["karume.scale.enc.w"] = payload["karume.scale.enc.w"]
        save_file(kept, str(merged))

        with pytest.raises(ContainerError, match="co-shard MUST"):
            verify_shards([head, merged, moved])

    def test_it_refuses_a_shard_set_that_repeats_the_graph(self, tmp_path):
        """`karume_ir` を持つ shard が 2 本ある列は「どちらのグラフか」が決まらない。"""
        head, *_ = self.split(tmp_path, 40)

        with pytest.raises(ContainerError, match="グラフ shard は先頭 1 本だけ"):
            verify_shards([head, head])


def piece_graph(
    storage: str, rows: int = 8, cols: int = 32
) -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """`[rows, cols]` の linear 重み 1 本を `storage` で丸めたグラフ（行分割の被験体）。

    値は決定的な等差列（`rounded` の乱数を使わない — 分割の前後でバイト列が同じであることを
    突き合わせるので、同じ引数から同じバイトが出る必要がある）。`cols` は 32 なので 1 行は
    F16 で 64B・I8 で 32B・I4（g16）で 16B と、どれも 4 の倍数になる。
    """
    weight = ((torch.arange(rows * cols, dtype=torch.float32) % 7) - 3).reshape(rows, cols)
    scales: dict[str, torch.Tensor] = {}
    if storage == "f16":
        weight = weight.to(torch.float16).to(torch.float32)
    elif storage == "i8":
        scale = channel_scale(weight, 0)
        weight = quantize_to_int8(weight, scale).to(torch.float32) * scale
        scales["enc.w"] = scale
    else:
        scale = group_scale(weight, 16)
        weight = dequantize_int4(quantize_to_int4(weight, scale), scale)
        scales["enc.w"] = scale
    graph = IrGraph(
        symbols=["T"],
        inputs=[IrInput(name="x", dtype="f32", shape=["T", cols])],
        outputs=["h"],
        initializers={
            "w": IrInitializer(tensor="enc.w", storage=IrStorage(dtype="f32")),
            "b": IrInitializer(tensor="enc.b", storage=IrStorage(dtype="f32")),
        },
        values={
            "w": IrValue(dtype="f32", shape=[rows, cols]),
            "b": IrValue(dtype="f32", shape=[rows]),
            "h": IrValue(dtype="f32", shape=["T", rows]),
        },
        nodes=[IrNode(op="linear", ins=["x", "w", "b"], outs=["h"], attrs={})],
    )
    tensors = {"enc.w": weight, "enc.b": (torch.arange(rows, dtype=torch.float32) % 5) - 2}
    return graph, tensors, scales


def stored_tensors(paths) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    """shard 列の全テンソル（キー → dtype・shape・生バイト）。piece は畳まずそのまま。"""
    found: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for path in paths:
        raw = path.read_bytes()
        length = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + length])
        start = 8 + length
        for key, spec in header.items():
            if key == "__metadata__":
                continue
            begin, end = spec["data_offsets"]
            found[key] = (spec["dtype"], tuple(spec["shape"]), raw[start + begin : start + end])
    return found


def joined_payload(paths, name: str) -> bytes:
    """`name` の生バイト（分割されていれば piece を index 順に連結して親へ畳む）。"""
    found = stored_tensors(paths)
    if name in found:
        return found[name][2]
    numbered = []
    for key, (_dtype, _shape, payload) in found.items():
        parsed = parse_piece_key(key)
        if parsed is not None and parsed[0] == name:
            numbered.append((parsed[1], payload))
    assert numbered, f"テンソル '{name}' が shard 列に無い: {sorted(found)}"
    return b"".join(payload for _index, payload in sorted(numbered))


class TestTensorPieces:
    """容量に収まらないテンソルは**行**で割って連続 shard へ配る（ADR 0090 決定 1）。

    被験体は `[8,32]` の linear 重み 1 本（{@link piece_graph}）。容量は f16 / i8 が 200B・
    i4 が 150B — どれも「重み（+ scale）が 1 shard に入らないが、行ブロックなら入る」帯に
    取ってある（実データで 256MiB を踏むテストは書けない）。
    """

    CAPACITY: ClassVar[dict[str, int]] = {"f16": 200, "i8": 200, "i4": 150}

    def write(self, tmp_path, storage: str, capacity: int | None) -> list:
        graph, tensors, scales = piece_graph(storage)
        return write_model(
            tmp_path / "model.safetensors",
            graph,
            tensors,
            weight_dtype=storage,
            weight_scales=scales,
            _shard_capacity=capacity,
        )

    def pieces_of(self, paths, name: str) -> list[tuple[int, int, int, str, tuple[int, ...]]]:
        """`name` の piece を `(index, count, shard, dtype, shape)` で index 順に。"""
        found = []
        for shard, path in enumerate(paths):
            for key, (dtype, shape, _payload) in stored_tensors([path]).items():
                parsed = parse_piece_key(key)
                if parsed is not None and parsed[0] == name:
                    found.append((parsed[1], parsed[2], shard, dtype, shape))
        return sorted(found)

    @pytest.mark.parametrize("storage", ["f16", "i8", "i4"])
    def test_the_weight_becomes_a_run_of_pieces_on_consecutive_shards(self, tmp_path, storage):
        """index は 1..n で、shard は 1 本ずつ進む（読み手契約 5）。"""
        written = self.write(tmp_path, storage, self.CAPACITY[storage])

        pieces = self.pieces_of(written, "enc.w")
        assert len(pieces) >= 2
        assert [index for index, *_ in pieces] == list(range(1, len(pieces) + 1))
        assert {count for _index, count, *_ in pieces} == {len(pieces)}
        assert [shard for _index, _count, shard, *_ in pieces] == list(
            range(pieces[0][2], pieces[0][2] + len(pieces))
        )

    @pytest.mark.parametrize("storage", ["f16", "i8", "i4"])
    def test_each_piece_declares_its_row_range_with_the_parent_dtype(self, tmp_path, storage):
        """dtype は親と同じ・shape は先頭次元だけが行数（残りの次元は親のまま）。"""
        written = self.write(tmp_path, storage, self.CAPACITY[storage])

        pieces = self.pieces_of(written, "enc.w")
        assert {dtype for *_head, dtype, _shape in pieces} == {storage.upper()}
        assert all(shape[1:] == (32,) for *_head, shape in pieces)
        assert sum(shape[0] for *_head, shape in pieces) == 8

    @pytest.mark.parametrize("storage", ["f16", "i8", "i4"])
    def test_the_split_container_passes_the_full_verification(self, tmp_path, storage):
        """読み返しの門は piece を親へ畳んでから、分割前と同じ集合を見る。"""
        written = self.write(tmp_path, storage, self.CAPACITY[storage])

        graph = verify_shards(written)

        assert graph.initializers["w"].storage.dtype == storage
        assert graph.values["w"].shape == [8, 32]

    @pytest.mark.parametrize("storage", ["f16", "i8", "i4"])
    def test_the_parent_bytes_match_an_unsplit_write(self, tmp_path, storage):
        """piece を連結すると分割前と**ビット同一**（切ってから変換 = 変換してから切る）。

        i8 の per-channel scale も i4 の group scale も先頭次元が行なので、断片ごとに切った
        scale で変換しても親の格納バイトと 1 バイトも変わらない。
        """
        whole = self.write(tmp_path / "whole", storage, None)
        split = self.write(tmp_path / "split", storage, self.CAPACITY[storage])

        assert len(split) > len(whole)
        assert joined_payload(split, "enc.w") == joined_payload(whole, "enc.w")

    @pytest.mark.parametrize("storage", ["i8", "i4"])
    def test_the_scale_shares_the_shard_of_the_first_piece(self, tmp_path, storage):
        """companion scale は割らず piece 1 と同居する（co-shard MUST の piece 版）。"""
        written = self.write(tmp_path, storage, self.CAPACITY[storage])

        first_shard = self.pieces_of(written, "enc.w")[0][2]
        assert "karume.scale.enc.w" in stored_tensors([written[first_shard]])
        assert joined_payload(written, "karume.scale.enc.w") == joined_payload(
            self.write(tmp_path / "whole", storage, None), "karume.scale.enc.w"
        )

    @pytest.mark.parametrize("storage", ["f16", "i8", "i4"])
    def test_a_capacity_that_fits_writes_no_piece_at_all(self, tmp_path, storage):
        """容量に収まるなら丸ごと 1 本のまま（小さいテンソルが piece に化けない）。"""
        written = self.write(tmp_path, storage, None)

        assert self.pieces_of(written, "enc.w") == []
        assert "enc.w" in stored_tensors(written)
