"""配布形の書き出し。宣言（グラフ JSON）と格納テンソルの対応が崩れたら書かない。"""

from __future__ import annotations

import json
import struct
import weakref

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from karume.emit import (
    EmitError,
    eligible_compressed_initializers,
    storage_breakdown,
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
from karume.quantize import channel_scale, quantize_to_int8
from karume.verify import ContainerError, assert_reader_layout, verify_model


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
        graph, tensors = sample_graph()

        path = write_model(tmp_path / "model.safetensors", graph, tensors)

        with safe_open(str(path), framework="pt") as handle:
            assert json.loads(handle.metadata()[IR_METADATA_KEY]) == graph.to_dict()
            assert set(handle.keys()) == {"enc.w"}

    def test_the_written_file_passes_the_full_verification(self, tmp_path):
        graph, tensors = sample_graph()

        path = write_model(tmp_path / "model.safetensors", graph, tensors)

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
    *, share_weight: bool = False, unused: bool = False
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """linear（重み偶数要素）+ embedding（重み **奇数要素**）を持つグラフ。

    `share_weight` は linear の重みを elementwise でも消費する形（適格外になるはず）、
    `unused` は誰も消費しない initializer を 1 本足す（同じく適格外）。
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


class TestF16Storage:
    def test_eligible_weights_are_stored_as_f16_and_bias_stays_f32(self, tmp_path):
        graph, tensors = weight_graph()

        path = write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        assert graph.initializers["w"].storage.dtype == "f16"
        assert graph.initializers["emb"].storage.dtype == "f16"
        # MUST: bias は常に f32（プロトタイプの f16 降格バグの根治形 — ADR 0006）。
        assert graph.initializers["b"].storage.dtype == "f32"
        with safe_open(str(path), framework="pt") as handle:
            assert handle.get_slice("enc.w").get_dtype() == "F16"
            assert handle.get_slice("enc.b").get_dtype() == "F32"

    def test_the_f16_file_passes_the_full_verification(self, tmp_path):
        graph, tensors = weight_graph()

        path = write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        assert verify_model(path).to_dict() == graph.to_dict()

    def test_the_stored_values_match_the_rounded_weights_bit_for_bit(self, tmp_path):
        graph, tensors = weight_graph()
        expected = tensors["enc.w"].clone()

        path = write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        with safe_open(str(path), framework="pt") as handle:
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

        with pytest.raises(EmitError, match="適格な重みスロットが 1 本も無い"):
            write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

    def test_an_unknown_weight_dtype_fails_loudly(self, tmp_path):
        graph, tensors = weight_graph()

        # bf16 は IR の語彙にはあるが実行経路が無い（ADR 0006）。
        with pytest.raises(EmitError, match="書き出せない"):
            write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="bf16")

    def test_the_breakdown_counts_eligible_and_plain_bytes(self, tmp_path):
        graph, tensors = weight_graph()

        write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")
        breakdown = storage_breakdown(graph)

        assert breakdown.compressed_tensors == 2
        assert breakdown.compressed_bytes == (3 * 4 + 3 * 5) * 2
        assert breakdown.plain_tensors == 1
        assert breakdown.plain_bytes == 3 * 4


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

        path = write_model(
            tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8", weight_scales=scales
        )

        assert graph.initializers["w"].storage.dtype == "i8"
        assert graph.initializers["w"].storage.scale == "karume.scale.enc.w"
        assert graph.initializers["emb"].storage.dtype == "i8"
        # MUST: bias は常に f32（プロトタイプの降格バグの根治形 — ADR 0006）。
        assert graph.initializers["b"].storage.dtype == "f32"
        with safe_open(str(path), framework="pt") as handle:
            assert handle.get_slice("enc.w").get_dtype() == "I8"
            assert handle.get_slice("enc.b").get_dtype() == "F32"
            assert handle.get_slice("karume.scale.enc.w").get_dtype() == "F32"
            assert list(handle.get_slice("karume.scale.enc.w").get_shape()) == [3, 1]

    def test_the_i8_file_passes_the_full_verification(self, tmp_path):
        graph, tensors, scales = int8_weight_graph()

        path = write_model(
            tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8", weight_scales=scales
        )

        assert verify_model(path).to_dict() == graph.to_dict()

    def test_the_stored_values_reconstruct_the_weights_bit_for_bit(self, tmp_path):
        """`q8·scale` が fake-quant 済みの重みと**ビット一致**する（格納の意味そのもの）。"""
        graph, tensors, scales = int8_weight_graph()
        expected = tensors["enc.w"].clone()

        path = write_model(
            tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8", weight_scales=scales
        )

        with safe_open(str(path), framework="pt") as handle:
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

        with pytest.raises(EmitError, match="適格な重みスロットが 1 本も無い"):
            write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8")

    def test_the_breakdown_counts_i8_bytes_and_the_scale_overhead(self, tmp_path):
        graph, tensors, scales = int8_weight_graph()

        write_model(
            tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8", weight_scales=scales
        )
        breakdown = storage_breakdown(graph)

        assert breakdown.compressed_tensors == 2
        assert breakdown.compressed_bytes == 3 * 4 + 3 * 5  # i8 = 1 バイト/要素
        # scale は出力チャネル数 × 4 バイト（w も emb も軸 0 の長さ 3）。
        assert breakdown.scale_bytes == (3 + 3) * 4
        assert breakdown.plain_tensors == 1
        assert breakdown.plain_bytes == 3 * 4


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

        assert path.exists()  # 書きかけ — 捨てるのは呼び出し側の層
        assert json.loads(graph.to_json()) == before


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

        streamed = write_model(
            tmp_path / "streamed.safetensors", graph, tensors, weight_dtype="f16"
        )

        reference = tmp_path / "reference.safetensors"
        # graph は書き出し後の commit 済み（= ヘッダに載ったのと同じ宣言）。
        _save_ordered(reference, pre, _write_order(pre), {IR_METADATA_KEY: graph.to_json()})
        assert streamed.read_bytes() == reference.read_bytes()

    def test_the_streamed_i8_bytes_match_a_pre_converted_write(self, tmp_path):
        """i8 も同じ（companion scale が増える形と I8 群の順序を踏む）。"""
        from karume.emit import _save_ordered, _write_order

        graph, tensors, scales = int8_weight_graph()
        pre = dict(tensors)
        for key, scale in scales.items():
            pre[key] = quantize_to_int8(tensors[key], scale)
            pre[f"karume.scale.{key}"] = scale

        streamed = write_model(
            tmp_path / "streamed.safetensors",
            graph,
            tensors,
            weight_dtype="i8",
            weight_scales=scales,
        )

        reference = tmp_path / "reference.safetensors"
        _save_ordered(reference, pre, _write_order(pre), {IR_METADATA_KEY: graph.to_json()})
        assert streamed.read_bytes() == reference.read_bytes()


def data_layout(path) -> list[tuple[int, str, str]]:
    """safetensors のデータ節に並ぶ順（開始 offset / 名前 / dtype）。"""
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + length])
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

        path = write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        # enc.emb は 15 要素 = 30 バイト（≡ 2 mod 4）なので末尾へ寄る。
        assert [name for _, name, _ in data_layout(path)][-1] == "enc.emb"

    def test_i8_tensors_are_written_after_everything_else(self, tmp_path):
        """I8 は要素サイズ 1 = 任意のバイト長を作るので、群の**末尾**に置く（ADR 0019）。

        前に置くと、その後ろの F32 / I32 / F16 の絶対 offset が要素サイズの倍数から外れて
        Karume のリーダが読めなくなる（HF の `safe_open` は読めてしまう）。
        """
        graph, tensors, scales = int8_weight_graph()

        path = write_model(
            tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8", weight_scales=scales
        )

        dtypes = [dtype for _, _, dtype in data_layout(path)]
        assert dtypes[-2:] == ["I8", "I8"]
        assert "I8" not in dtypes[:-2]

    def test_the_emitted_file_satisfies_the_reader_layout_rules(self, tmp_path):
        graph, tensors = weight_graph()

        path = write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        assert_reader_layout(path)  # 例外が出なければ合格

    def test_the_emitted_i8_file_satisfies_the_reader_layout_rules(self, tmp_path):
        graph, tensors, scales = int8_weight_graph()

        path = write_model(
            tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8", weight_scales=scales
        )

        assert_reader_layout(path)  # 例外が出なければ合格

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
        metadata = {IR_METADATA_KEY: graph.to_json()}
        reference = tmp_path / "reference.safetensors"
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(reference), metadata)

        path = write_model(tmp_path / "model.safetensors", graph, tensors)

        assert path.read_bytes() == reference.read_bytes()


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
