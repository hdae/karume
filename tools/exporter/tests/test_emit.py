"""配布形の書き出し。宣言（グラフ JSON）と格納テンソルの対応が崩れたら書かない。"""

from __future__ import annotations

import json
import struct
import weakref
from dataclasses import replace

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
from karume.quantize import (
    channel_scale,
    dequantize_int4,
    group_scale,
    quantize_to_int4,
    quantize_to_int8,
)
from karume.verify import ContainerError, assert_reader_layout, parse_ir_graph, verify_model


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

        path = write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        # 宣言の観測点はファイル側（`write_model` は呼び手の graph を書き換えない）。
        declared = verify_model(path).initializers
        assert declared["w"].storage.dtype == "f16"
        assert declared["emb"].storage.dtype == "f16"
        # MUST: bias は常に f32（プロトタイプの f16 降格バグの根治形 — ADR 0006）。
        assert declared["b"].storage.dtype == "f32"
        with safe_open(str(path), framework="pt") as handle:
            assert handle.get_slice("enc.w").get_dtype() == "F16"
            assert handle.get_slice("enc.b").get_dtype() == "F32"

    def test_a_weight_in_the_graph_outputs_is_not_stored_compressed(self, tmp_path):
        """グラフ出力の重みは f32 のまま書く（ランタイム側の適格判定と対）。

        鏡像がずれると「exporter は f16 で書き、ランタイムは f32 として読み戻す」の
        食い違いになり、ロードした瞬間に読めないモデルが配布形として通ってしまう。
        """
        graph, tensors = weight_graph(output_weight=True)

        path = write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        declared = verify_model(path).initializers
        assert declared["w"].storage.dtype == "f32"
        assert declared["emb"].storage.dtype == "f16"
        with safe_open(str(path), framework="pt") as handle:
            assert handle.get_slice("enc.w").get_dtype() == "F32"
            assert handle.get_slice("enc.emb").get_dtype() == "F16"
        assert verify_model(path).to_dict() == compressed_view(graph, {"emb": "f16"}).to_dict()

    def test_the_f16_file_passes_the_full_verification(self, tmp_path):
        graph, tensors = weight_graph()

        path = write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")

        expected = compressed_view(graph, {"w": "f16", "emb": "f16"})
        assert verify_model(path).to_dict() == expected.to_dict()

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

        path = write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="f16")
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

        path = write_model(
            tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8", weight_scales=scales
        )

        declared = verify_model(path).initializers
        assert declared["w"].storage.dtype == "i8"
        assert declared["w"].storage.scale == "karume.scale.enc.w"
        assert declared["emb"].storage.dtype == "i8"
        # MUST: bias は常に f32（プロトタイプの降格バグの根治形 — ADR 0006）。
        assert declared["b"].storage.dtype == "f32"
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

        expected = compressed_view(graph, {"w": "i8", "emb": "i8"})
        assert verify_model(path).to_dict() == expected.to_dict()

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

        with pytest.raises(EmitError, match="適格な重みスロットが 1 本も無い"):
            write_model(tmp_path / "model.safetensors", graph, tensors, weight_dtype="i8")

    def test_the_breakdown_counts_i8_bytes_and_the_scale_overhead(self, tmp_path):
        graph, tensors, scales = int8_weight_graph()

        path = write_model(
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


def container_header(path) -> dict:
    """safetensors のヘッダ JSON を直に読む。

    `safetensors` ライブラリ（0.8.0）は `I4` を知らないので、packed 4bit を含む配布形は
    `safe_open` では開けない（ADR 0069 決定 2）— 観測はヘッダ JSON から行う。
    """
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    return json.loads(raw[8 : 8 + length])


def stored_payload(path, name: str) -> torch.Tensor:
    """データ節から 1 本ぶんの生バイトを uint8 テンソルで取り出す。"""
    raw = path.read_bytes()
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
    return parse_ir_graph(container_header(path)["__metadata__"][IR_METADATA_KEY])


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


def int4_weight_graph(
    *, embedding: bool = False, conv: bool = False, group_size: int = 16
) -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """linear の重み `[3,32]` を group 対称 i4 で fake-quant 済みにしたグラフ。

    in 軸 32 / group 16 なので 1 行に group が 2 つ入る（scale の group 形 `[3,2]`）。
    `embedding` を立てると **embedding 表**（i4 適格の 2 つ目の重みスロット — ADR 0069
    決定 5 の追補）が、`conv` を立てると **conv1d の重み**（重みスロット適格だが i4 の
    展開経路が無い = i4 が拒否すべき形）が増える。
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
    if conv:
        nodes.append(IrNode(op="conv1d", ins=["x", "cw", "cb"], outs=["c"], attrs={}))
        initializers["cw"] = IrInitializer(tensor="enc.cw", storage=IrStorage(dtype="f32"))
        initializers["cb"] = IrInitializer(tensor="enc.cb", storage=IrStorage(dtype="f32"))
        values["cw"] = IrValue(dtype="f32", shape=[3, 2, 16])
        values["cb"] = IrValue(dtype="f32", shape=[3])
        values["c"] = IrValue(dtype="f32", shape=["T", 3])
        tensors["enc.cw"] = torch.randn(3, 2, 16)
        tensors["enc.cb"] = torch.randn(3)
        outputs.append("c")
    graph = IrGraph(
        symbols=["T"],
        inputs=inputs,
        outputs=outputs,
        initializers=initializers,
        values=values,
        nodes=nodes,
    )
    scale = group_scale(tensors["enc.w"], group_size)
    tensors["enc.w"] = dequantize_int4(quantize_to_int4(tensors["enc.w"], scale), scale)
    return graph, tensors, {"enc.w": scale}


def write_int4(path, graph: IrGraph, tensors, scales):
    return write_model(path, graph, tensors, weight_dtype="i4", weight_scales=scales)


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
    """i4 の適格は **linear / embedding の重みスロット限定**（ADR 0069 決定 5 + embedding 追補
    — i4 の展開経路を持つカーネルはこの 2 つだけ）。
    """

    def test_linear_and_embedding_weight_slots_are_listed(self):
        graph, _, _ = int4_weight_graph(embedding=True)

        assert eligible_compressed_initializers(graph) == {"w", "emb"}
        assert i4_eligible_initializers(graph) == {"w", "emb"}

    def test_a_conv_weight_is_eligible_but_not_i4_eligible(self):
        """conv 系の重みは一般適格でも i4 の適格には入らない（展開経路が無い）。"""
        graph, _, _ = int4_weight_graph(conv=True)

        assert eligible_compressed_initializers(graph) == {"w", "cw"}
        assert i4_eligible_initializers(graph) == {"w"}

    def test_a_weight_shared_with_another_weight_slot_is_excluded(self):
        """conv とも共有される重みは i4 の適格に入らない（展開経路が 1 つに決まらない）。"""
        graph, _, _ = int4_weight_graph()
        graph.nodes.append(IrNode(op="conv1d", ins=["x", "w", "b"], outs=["c"], attrs={}))

        assert i4_eligible_initializers(graph) == set()


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
        header = container_header(path)
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

        assert_reader_layout(path)  # 例外が出なければ合格（I4 の先頭 4 バイト整列を含む）

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

    def test_a_conv_weight_stays_f32(self, tmp_path):
        """conv の重みは一般適格でも i4 にせず f32 のまま残す（Codex 波 F 指摘 I4-ELIG-01）。

        ここで export 全体を落とすと、linear + conv を持つ普通のグラフが i4 で書けなくなる。
        ランタイム側の受け皿（eligible ∩ i4Eligible の外は CPU 展開）と対の設計は
        「展開経路のある op の重みだけ圧縮・他は f32」で、i8 の適格外が f32 で残るのと同じ。
        """
        graph, tensors, scales = int4_weight_graph(conv=True)

        path = write_int4(tmp_path / "model.safetensors", graph, tensors, scales)

        declared = written_graph(path).initializers
        assert declared["w"].storage.dtype == "i4"
        assert declared["cw"].storage.dtype == "f32"

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
        header = container_header(path)
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

        assert_reader_layout(path)

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
        """linear と conv で共有された重みしか無い形は「適格 0 本」で落ちる（沈黙させない）。"""
        graph, tensors, scales = int4_weight_graph()
        graph.nodes.append(IrNode(op="conv1d", ins=["x", "w", "b"], outs=["c"], attrs={}))
        graph.values["c"] = IrValue(dtype="f32", shape=["T", 3])
        graph.outputs.append("c")

        with pytest.raises(EmitError, match="適格な重みスロットが 1 本も無い"):
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

        with pytest.raises(EmitError, match="適格な重みスロットが 1 本も無い"):
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

        assert path.exists()  # 書きかけ — 捨てるのは呼び出し側の層
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
        plain = write_model(tmp_path / "plain.safetensors", graph, tensors)

        with safe_open(str(plain), framework="pt") as handle:
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

        streamed = write_model(
            tmp_path / "streamed.safetensors", graph, tensors, weight_dtype="f16"
        )

        reference = tmp_path / "reference.safetensors"
        # ヘッダに載るのは commit 済みの宣言ビュー（呼び手の graph は f32 のまま）。
        committed = compressed_view(graph, {"w": "f16", "emb": "f16"})
        _save_ordered(reference, pre, _write_order(pre), {IR_METADATA_KEY: committed.to_json()})
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
        committed = compressed_view(graph, {"w": "i8", "emb": "i8"})
        _save_ordered(reference, pre, _write_order(pre), {IR_METADATA_KEY: committed.to_json()})
        assert streamed.read_bytes() == reference.read_bytes()


def data_layout(path) -> list[tuple[int, str, str]]:
    """safetensors のデータ節に並ぶ順（開始 offset / 名前 / dtype）。"""
    header = container_header(path)
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
        assert_reader_layout(path)

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

        path = write_model(
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
        header = container_header(path)
        assert header["enc.w"]["dtype"] == "I4"
        assert header["enc.emb"]["dtype"] == "I8"

    def test_overrides_compress_even_with_a_f32_default(self, tmp_path):
        """既定 f32 は従来「空プランで即返し」だった — 明示指定だけの圧縮も通ること。"""
        graph, tensors, scales = mixed_weight_graph()

        path = write_model(
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

        path = write_model(
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
        """conv の重みは一般適格でも i4 の展開経路が無い。

        既定は静かに f32 へ残すが、明示指定は fail loudly（`_plan_weight_dtype` の線引き）。
        """
        graph, tensors, scales = int4_weight_graph(conv=True)

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
