"""`irodori/dacvae/export.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（既存の export 台本のテストと同じ規律）。ここで固定するのは、壊れると
**偽 PASS** になる側の規律だけ:

- 参照（golden の期待値）は `remove_weight_norm` の**前**にしか採れない（採れてしまうと
  畳み込みの同値検証が恒真化する）
- 透かしバイパスが②形になっていない（出力が波形でない）と参照採取で落ちる
- 主経路の抽出が `Decoder` 本体と食い違ったら落ちる
- `in_proj` の切り詰めが後半（scale）を取っていたら落ちる
- `remove_weight_norm` が値を変えたら落ちる
- 往復（encoder → decoder）が波形として妥当でなければ落ちる
- Snake の定数部分木が畳まれず `reciprocal` が残ったら落ちる
- `_write_io` が 1 入力 1 出力でないグラフを書かない
- ケースの長さが記号次元の上限を超えたら落ちる
"""

from __future__ import annotations

import types

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from irodori.dacvae import export as ex
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.pipeline import export_to_file
from karume.quantize import quantize_to_int8


class Scale(nn.Module):
    """定数倍の層（並び順の違いが値に出る代役）。"""

    def __init__(self, gain: float) -> None:
        super().__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gain


class TinyBlock(nn.Module):
    """`DecoderBlock` の代役（`_chunk_size` ごとに切って倍数のかたまりだけ通す）。"""

    def __init__(self) -> None:
        super().__init__()
        self._chunk_size = 2
        self.block = nn.ModuleList([Scale(g) for g in (2.0, 3.0, 5.0, 7.0, 11.0, 13.0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # チャンク 0（gain 2,3）とチャンク 2（gain 11,13）だけが主経路。
        return x * (2.0 * 3.0 * 11.0 * 13.0)


class TinyWatermark(nn.Module):
    """`wm_model.encoder_block` の代役（`pre[3]` は②形で Identity に差し替わる位置）。"""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Sequential(Scale(17.0), Scale(19.0), Scale(23.0), Scale(29.0))

    def forward_no_conv(self, x: torch.Tensor) -> torch.Tensor:
        return self.pre[2](self.pre[1](self.pre[0](x)))


class TinyDecoder(nn.Module):
    """`Decoder` の代役。`extra` を与えると主経路から外れた層を混ぜる（故障注入）。"""

    def __init__(self, extra: float = 1.0) -> None:
        super().__init__()
        self.model = nn.ModuleList([Scale(1.5), TinyBlock()])
        self.wm_model = nn.Module()
        self.wm_model.encoder_block = TinyWatermark()
        self.extra = extra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model:
            x = layer(x)
        return self.wm_model.encoder_block.forward_no_conv(x) * self.extra


class TinyQuantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out_proj = nn.Conv1d(2, 2, 1)
        self.in_proj = nn.Conv1d(2, 4, 1)


class TinyCodec(nn.Module):
    """`_pristine_outputs` / `_remove_weight_norm_evidence` が読む属性だけを持つ代役。"""

    def __init__(self, extra: float = 1.0, weight_norm: bool = True) -> None:
        super().__init__()
        self.decoder = TinyDecoder(extra)
        self.quantizer = TinyQuantizer()
        self.encoder = nn.Conv1d(1, 2, 1)
        if weight_norm:
            nn.utils.weight_norm(self.quantizer.in_proj)
        self.eval()


DECODER_CASES = (("a", torch.ones(1, 3, 2)),)
ENCODER_CASES = (("b", torch.ones(1, 3, 4)),)


class SingleChannelDecoder(TinyDecoder):
    """出力が 1 チャネル（波形）になる代役 — 実物の②形バイパスと同じ性質。"""


class MultiChannelDecoder(nn.Module):
    """②形バイパスが効いていない代役（96ch 相当がそのまま返る故障注入）。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _codec_with_decoder(decoder: nn.Module) -> nn.Module:
    codec = TinyCodec()
    codec.decoder = decoder
    return codec


class TestPristineOrdering:
    """MUST: 参照は `remove_weight_norm` の前にしか採れない（恒真化の遮断）。"""

    def test_taking_a_reference_after_folding_fails_loudly(self):
        codec = TinyCodec(weight_norm=False)
        codec.decoder = _one_channel_decoder()

        with pytest.raises(AssertionError, match="weight_norm を畳んだ後に参照を採ろうとした"):
            ex._pristine_outputs(codec, DECODER_CASES, ENCODER_CASES)

    def test_taking_a_reference_before_folding_is_allowed(self):
        codec = TinyCodec()
        codec.decoder = _one_channel_decoder()

        pristine = ex._pristine_outputs(codec, DECODER_CASES, ENCODER_CASES)

        assert set(pristine) == {ex.TARGET_DECODER, ex.TARGET_ENCODER}
        assert tuple(pristine[ex.TARGET_DECODER]["a"].shape) == (1, 1, 3)
        # in_proj は 4 出力なので mean 側は 2 チャネル → 転置して [1,T,2]。
        assert tuple(pristine[ex.TARGET_ENCODER]["b"].shape) == (1, 12, 2)

    def test_a_multi_channel_decode_fails_loudly(self):
        """MUST: `alpha = 0.0` だけの片手落ちバイパスは shape でしか気づけない。"""
        codec = _codec_with_decoder(MultiChannelDecoder())

        with pytest.raises(AssertionError, match="透かしバイパス"):
            ex._pristine_outputs(codec, DECODER_CASES, ENCODER_CASES)


def _one_channel_decoder() -> nn.Module:
    """`[1,2,T]` を受けて `[1,1,T]` を返す代役（波形ヘッドと同じ形）。"""

    class Head(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :1] * 2.0

    return Head()


class TestMainPathEvidence:
    """MUST: 「透かし枝は decode に寄与しない」を主張のままにしない。"""

    def test_a_faithful_decoder_measures_zero(self):
        decoder = TinyDecoder()
        out_proj = nn.Identity()

        evidence = ex._main_path_evidence(decoder, DECODER_CASES, out_proj)

        assert evidence == {"a": 0.0}

    def test_a_decoder_that_takes_another_route_fails_loudly(self):
        decoder = TinyDecoder(extra=1.0001)

        with pytest.raises(AssertionError, match="主経路の抽出"):
            ex._main_path_evidence(decoder, DECODER_CASES, nn.Identity())

    def test_the_extraction_keeps_only_the_chunk_multiples(self):
        extracted = ex.main_path(TinyDecoder())

        # head(1.5) + チャンク 0（2,3）+ チャンク 2（11,13）+ pre[0..2]（17,19,23）。
        assert [float(layer.gain) for layer in extracted] == [
            1.5,
            2.0,
            3.0,
            11.0,
            13.0,
            17.0,
            19.0,
            23.0,
        ]


class TestInProjTruncation:
    def test_the_first_half_measures_within_the_gate_tolerance(self):
        """NOTE: 期待は 0.0 ぴったりではない — 比べる 2 辺は conv のカーネル選択が違い得て、
        ビット一致は torch の仕様保証ではない（`IN_PROJ_TRUNCATION_ATOL` の導出表）。実測でも
        テスト追加による global RNG 列のずれで 1 ulp 差が出た（2026-08-13）。"""
        codec = TinyCodec(weight_norm=False)
        trimmed = ex.truncated_in_proj(codec.quantizer.in_proj)

        evidence = ex._in_proj_truncation_evidence(codec, trimmed, ENCODER_CASES)

        assert set(evidence) == {"b"}
        assert 0.0 <= evidence["b"] <= ex.IN_PROJ_TRUNCATION_ATOL

    def test_taking_the_second_half_fails_loudly(self):
        """MUST: mean と scale の取り違えは shape も dtype も一致するのでここでしか出ない。"""
        codec = TinyCodec(weight_norm=False)
        in_proj = codec.quantizer.in_proj
        half = in_proj.out_channels // 2
        wrong = nn.Conv1d(in_proj.in_channels, half, 1)
        with torch.no_grad():
            wrong.weight.copy_(in_proj.weight[half:])
            wrong.bias.copy_(in_proj.bias[half:])

        with pytest.raises(AssertionError, match="chunk\\(2\\)\\[0\\]"):
            ex._in_proj_truncation_evidence(codec, wrong.eval(), ENCODER_CASES)

    def test_truncating_before_folding_fails_loudly(self):
        codec = TinyCodec()

        with pytest.raises(SystemExit, match="weight_norm が畳まれていない"):
            ex.truncated_in_proj(codec.quantizer.in_proj)

    def test_an_odd_output_count_fails_loudly(self):
        with pytest.raises(SystemExit, match="偶数でない"):
            ex.truncated_in_proj(nn.Conv1d(2, 3, 1))


class TestFoldEvidence:
    def test_an_unchanged_output_measures_zero(self):
        codec = TinyCodec()
        codec.decoder = _one_channel_decoder()
        pristine = ex._pristine_outputs(codec, DECODER_CASES, ENCODER_CASES)
        ex.fold_weight_norm(codec)

        evidence = ex._remove_weight_norm_evidence(codec, DECODER_CASES, ENCODER_CASES, pristine)

        assert evidence[ex.TARGET_DECODER] == {"a": 0.0}
        assert evidence[ex.TARGET_ENCODER] == {"b": 0.0}

    def test_a_changed_output_fails_loudly(self):
        codec = TinyCodec()
        codec.decoder = _one_channel_decoder()
        pristine = ex._pristine_outputs(codec, DECODER_CASES, ENCODER_CASES)
        ex.fold_weight_norm(codec)
        with torch.no_grad():
            codec.quantizer.in_proj.bias.add_(1e-3)

        with pytest.raises(AssertionError, match="remove_weight_norm の前後で出力が変わった"):
            ex._remove_weight_norm_evidence(codec, DECODER_CASES, ENCODER_CASES, pristine)

    def test_measuring_before_folding_fails_loudly(self):
        codec = TinyCodec()
        codec.decoder = _one_channel_decoder()
        pristine = ex._pristine_outputs(codec, DECODER_CASES, ENCODER_CASES)

        with pytest.raises(AssertionError, match="weight_norm がまだ残っている"):
            ex._remove_weight_norm_evidence(codec, DECODER_CASES, ENCODER_CASES, pristine)

    def test_a_model_without_weight_norm_fails_loudly(self):
        with pytest.raises(SystemExit, match="weight_norm を持つモジュールが 1 本も無い"):
            ex.fold_weight_norm(TinyCodec(weight_norm=False))


class FakeSnake(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 2, 1))


class TestLiftSnakeAlphas:
    def test_alphas_become_plain_attributes(self):
        model = nn.Sequential(FakeSnake(), FakeSnake())
        source = types.SimpleNamespace(snake_cls=FakeSnake)

        assert ex.lift_snake_alphas(source, model) == 2
        assert [name for name, _ in model.named_parameters()] == []
        assert all(isinstance(module.alpha, torch.Tensor) for module in model)

    def test_an_empty_scan_fails_loudly(self):
        source = types.SimpleNamespace(snake_cls=FakeSnake)

        with pytest.raises(SystemExit, match="Snake1d が 1 本も見つからない"):
            ex.lift_snake_alphas(source, nn.Sequential(nn.Identity()))


class Passthrough(nn.Module):
    """往復の代役（`gain` で相関を、`shift` で長さの取り違えを注入する）。"""

    def __init__(self, gain: float = 1.0, noise: float = 0.0, trim: int = 0) -> None:
        super().__init__()
        self.gain, self.noise, self.trim = gain, noise, trim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(1, 1, -1) * self.gain
        if self.noise:
            generator = torch.Generator().manual_seed(0)
            flat = flat + torch.randn(flat.shape, generator=generator) * self.noise
        return flat[..., : flat.shape[-1] - self.trim]


class TestRoundtripEvidence:
    @pytest.fixture
    def wave(self):
        return torch.sin(torch.linspace(0, 40, 800)).reshape(1, 10, 80)

    def test_a_faithful_roundtrip_reports_the_correlation(self, wave):
        evidence = ex._roundtrip_evidence(nn.Identity(), Passthrough(), wave)

        assert evidence["correlation"] == 1.0

    def test_an_uncorrelated_roundtrip_fails_loudly(self, wave):
        with pytest.raises(AssertionError, match="往復の相関"):
            ex._roundtrip_evidence(nn.Identity(), Passthrough(gain=0.0, noise=1.0), wave)

    def test_a_length_mismatch_fails_loudly(self, wave):
        with pytest.raises(AssertionError, match="往復の長さ"):
            ex._roundtrip_evidence(nn.Identity(), Passthrough(trim=8), wave)

    def test_a_non_finite_output_fails_loudly(self, wave):
        with pytest.raises(AssertionError, match="非有限"):
            ex._roundtrip_evidence(nn.Identity(), Passthrough(gain=float("inf")), wave)


class TestWrapperEquivalence:
    def test_an_identical_output_measures_zero(self):
        value = torch.ones(1, 2, 3)

        assert ex._check_wrapper_equivalence(nn.Identity(), value, value, "テスト", 0.0) == 0.0

    def test_a_shape_mismatch_fails_loudly(self):
        with pytest.raises(AssertionError, match="ラッパの出力 shape"):
            ex._check_wrapper_equivalence(
                nn.Flatten(), torch.ones(1, 2, 3), torch.ones(1, 2, 3), "テスト", 0.0
            )

    def test_a_difference_over_atol_fails_loudly(self):
        with pytest.raises(AssertionError, match="eager 同値が崩れた"):
            ex._check_wrapper_equivalence(
                Scale(1.5), torch.ones(1, 2, 3), torch.ones(1, 2, 3), "テスト", 0.0
            )


def _graph_with_ops(*ops: str) -> IrGraph:
    return IrGraph(
        symbols=["S"],
        inputs=[IrInput(name="latent", dtype="f32", shape=[1, "S", 2])],
        outputs=["out"],
        initializers={"w": IrInitializer(tensor="w", storage=IrStorage(dtype="f32"))},
        values={"out": IrValue(dtype="f32", shape=[1, "S", 2])},
        nodes=[IrNode(op=op, ins=["latent"], outs=["out"], attrs={}) for op in ops],
    )


class TestSnakeFolded:
    def test_a_folded_graph_passes(self):
        ex.assert_snake_folded(_graph_with_ops("sin", "mul", "add"), "テスト")

    def test_a_leaked_reciprocal_fails_loudly(self):
        with pytest.raises(AssertionError, match="定数部分木が畳まれず"):
            ex.assert_snake_folded(_graph_with_ops("sin", "reciprocal"), "テスト")

    def test_a_graph_without_sin_fails_loudly(self):
        with pytest.raises(AssertionError, match="sin がグラフに無い"):
            ex.assert_snake_folded(_graph_with_ops("mul", "add"), "テスト")


class TinyGraph(nn.Module):
    """`_write_io` を回すための最小グラフ（入力名 `latent` の 1 入力 1 出力）。"""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.fc(latent)


class TestWriteIo:
    @pytest.fixture
    def exported(self, tmp_path):
        torch.manual_seed(0)
        module = TinyGraph()
        graph = export_to_file(module, (torch.randn(1, 3, 4),), tmp_path / ex.MODEL_FILE)
        return module, graph, tmp_path

    def test_writes_one_file_per_case(self, exported):
        module, graph, out_dir = exported
        latent = torch.randn(1, 3, 4)
        with torch.no_grad():
            expected = {"a": module(latent)}

        written = ex._write_io(graph, {"a": latent}, expected, out_dir)

        assert written == [f"{ex.IO_PREFIX}a{ex.IO_SUFFIX}"]
        tensors = load_file(str(out_dir / written[0]))
        assert sorted(tensors) == ["input.latent", "output.0"]
        assert torch.equal(tensors["output.0"], expected["a"])

    def test_a_multi_output_graph_fails_loudly(self, exported):
        """MUST: 1 入力 1 出力の規約から外れた io を黙って書かない。"""
        _module, graph, out_dir = exported
        two = IrGraph(
            inputs=graph.inputs,
            outputs=[*graph.outputs, "second"],
            nodes=graph.nodes,
            values={**graph.values, "second": IrValue(dtype="f32", shape=[1, 3, 2])},
            initializers=graph.initializers,
            symbols=graph.symbols,
        )

        with pytest.raises(AssertionError, match="1 本ずつでない"):
            ex._write_io(two, {"a": torch.randn(1, 3, 4)}, {"a": torch.zeros(1, 3, 2)}, out_dir)


class TestDecoderCases:
    @pytest.fixture
    def latent_dir(self, tmp_path):
        for name, frames in (("full", 5), ("no-ref", 3)):
            latent = torch.arange(frames * 2, dtype=torch.float32).reshape(1, frames, 2)
            save_file(
                {ex.LATENT_KEY: latent},
                str(tmp_path / f"{ex.LATENT_CASE_PREFIX}{name}{ex.IO_SUFFIX}"),
            )
        return tmp_path

    def test_lengths_follow_the_table_and_are_deterministic(self, latent_dir):
        first = ex.build_decoder_cases(latent_dir, 750)
        second = ex.build_decoder_cases(latent_dir, 750)

        assert [name for name, _ in first] == [name for name, _s, _l in ex.DECODER_CASES]
        assert {name: int(z.shape[1]) for name, z in first} == {
            "z-min": ex.MIN_SYM_LENGTH,
            "z-no-ref": 3,
            "z-full": 5,
            "z-max": 750,
        }
        for (_name, lhs), (_same, rhs) in zip(first, second, strict=True):
            assert torch.equal(lhs, rhs)

    def test_the_long_case_repeats_the_real_latent(self, latent_dir):
        cases = dict(ex.build_decoder_cases(latent_dir, 750))

        source = cases["z-full"]
        assert torch.equal(cases["z-max"][:, : source.shape[1]], source)
        assert torch.equal(cases["z-max"][:, source.shape[1] : 2 * source.shape[1]], source)

    def test_a_case_over_the_symbolic_cap_fails_loudly(self, latent_dir):
        with pytest.raises(SystemExit, match="記号次元の範囲"):
            ex.build_decoder_cases(latent_dir, 8)

    def test_a_missing_latent_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="実 latent が無い"):
            ex.build_decoder_cases(tmp_path, 750)


class TestBypassWatermark:
    def test_both_halves_of_the_bypass_are_applied(self):
        decoder = TinyDecoder()

        ex.bypass_watermark(decoder)

        assert decoder.alpha == 0.0
        assert torch.equal(
            decoder.watermark(torch.ones(1, 1, 4)), torch.full((1, 1, 4), 17.0 * 19.0 * 23.0)
        )


class TestTargetCli:
    def test_all_targets_by_default(self, monkeypatch):
        seen: dict[str, object] = {}
        monkeypatch.setattr(ex, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})
        ex.main([])

        assert seen["targets"] == ex.TARGETS

    def test_a_single_target_is_forwarded(self, monkeypatch):
        seen: dict[str, object] = {}
        monkeypatch.setattr(ex, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})
        ex.main(["--target", ex.TARGET_ENCODER])

        assert seen["targets"] == (ex.TARGET_ENCODER,)

    def test_the_default_out_root_is_the_weight_directory_name(self, tmp_path):
        assert ex.default_out_root(tmp_path / "dacvae-32dim").name == "dacvae-32dim"

    def test_the_dtype_is_forwarded(self, monkeypatch):
        seen: dict[str, object] = {}
        monkeypatch.setattr(ex, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})
        ex.main(["--dtype", "f16"])

        assert seen["dtype"] == "f16"


def _is_f16_exact(tensor: torch.Tensor) -> bool:
    """f16 の格子に乗っているか（emit の適格判定と同じ述語）。"""
    return bool(torch.equal(tensor, tensor.to(torch.float16).to(torch.float32)))


class TestWeightDtypeSeries:
    """格納 dtype の系列（ADR 0018 / 0019 / 0027 / 0050）— 壊れると**偽 PASS** になる側だけ。

    実重みは使わない（配線の規律はモデルに依らない）。数値そのものの検証は Deno 側の E2E と
    emit の適格判定が持つ。
    """

    def test_each_dtype_gets_its_own_series_root(self):
        """MUST: 圧縮系列は別ディレクトリ（同居させると f32 の網が圧縮資産へ掛かる）。"""
        roots = {
            dtype: ex.default_out_root(ex.DEFAULT_MODEL_DIR, dtype) for dtype in ex.WEIGHT_DTYPES
        }

        assert set(roots) == {"f32", "f16", "i8"}
        assert len(set(roots.values())) == len(roots)
        assert roots["f32"].name == "dacvae-32dim"
        assert roots["f16"].name == "dacvae-32dim-f16"
        assert roots["i8"].name == "dacvae-32dim-i8"

    def test_f32_leaves_the_weights_untouched(self):
        # MUST: 種を蒔いたら元へ戻す（`fork_rng`）— 大域 RNG を置き去りにすると、後続の
        # テストが**別の乱数**でモジュールを組むことになり、この追加が離れた場所の実測門を動かす。
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model = TinyCodec(weight_norm=False)
            before = model.encoder.weight.clone()

            quantized = ex._fake_quant("f32", model)

            assert quantized.report is None
            assert quantized.scales == {}
            assert torch.equal(model.encoder.weight, before)

    def test_f16_rounds_every_parameter_of_the_exported_model(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model = TinyCodec(weight_norm=False)
            assert not _is_f16_exact(model.encoder.weight), "丸め前から格子に乗っては検出力が無い"

            quantized = ex._fake_quant("f16", model)

            assert quantized.report is not None
            assert quantized.scales == {}
            assert all(_is_f16_exact(tensor) for tensor in model.parameters())

    def test_i8_quantizes_the_convolutions_and_hands_out_scales(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model = TinyCodec(weight_norm=False)

            quantized = ex._fake_quant("i8", model)

            assert sorted(quantized.scales) == [
                "encoder.weight",
                "quantizer.in_proj.weight",
                "quantizer.out_proj.weight",
            ]
            for key, scale in quantized.scales.items():
                weight = model.get_parameter(key)
                restored = quantize_to_int8(weight, scale).to(torch.float32) * scale
                assert torch.equal(restored, weight), "i8 の格子に乗っていない"

    def test_a_lifted_snake_alpha_stays_f32(self):
        """降格済みの `alpha` は重みスロットではない（lifted 定数）ので**丸めない**。

        golden も f32 の alpha で計算されるので、丸めた側だけが動くと両者が食い違う。
        """
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model = nn.Sequential(FakeSnake(), nn.Conv1d(2, 2, 1))
            with torch.no_grad():
                model[0].alpha.copy_(torch.full((1, 2, 1), 1.0 + 2.0**-13))
            source = types.SimpleNamespace(snake_cls=FakeSnake)
            ex.lift_snake_alphas(source, model)
            alpha = model[0].alpha.clone()
            assert not _is_f16_exact(alpha), "丸め前から格子に乗っては検出力が無い"

            ex._fake_quant("f16", model)

            assert torch.equal(model[0].alpha, alpha)
            assert _is_f16_exact(model[1].weight)


class TestTargetScales:
    """i8 の scale 台帳を**ラッパ内 FQN**へ張り替える表（`TARGET_SCALE_SOURCES`）。

    MUST: ここが崩れると emit が「適格なのに scale が無い」で落ちる（値が壊れる形では
    通らない）。落ちる場所が遠いので、表とラッパの対応はここで固定する。
    """

    def _quantized(self) -> tuple[nn.Module, dict[str, torch.Tensor]]:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model = TinyCodec(weight_norm=False)
            return model, dict(ex._fake_quant("i8", model).scales)

    def test_the_table_covers_every_target(self):
        assert sorted(ex.TARGET_SCALE_SOURCES) == sorted(ex.TARGETS)

    def test_the_decoder_takes_only_its_own_branch(self):
        """`quantizer.out_proj` は `out_proj` へ張り替え、encoder / in_proj は落とす。"""
        model, scales = self._quantized()
        wrapper = ex.DecoderGraph(model.quantizer.out_proj, model.decoder)

        picked = ex._target_scales(ex.TARGET_DECODER, wrapper, scales)

        assert sorted(picked) == ["out_proj.weight"]
        assert torch.equal(picked["out_proj.weight"], scales["quantizer.out_proj.weight"])

    def test_the_encoder_gets_the_front_rows_of_the_in_proj_scale(self):
        """MUST: 切り詰めた `in_proj` の scale は本体の**前半行そのもの**（引き直さない）。

        引き直すと f32 の割り算で 1ulp 動きうるので、emit の逆変換ビット一致検査が落ちる。
        ここでは「emit が実際にする検査」を同じ式で踏んで、通ることまで確かめる。
        """
        model, scales = self._quantized()
        trimmed = ex.truncated_in_proj(model.quantizer.in_proj)
        wrapper = ex.EncoderGraph(model.encoder, trimmed)

        picked = ex._target_scales(ex.TARGET_ENCODER, wrapper, scales)

        assert sorted(picked) == ["encoder.weight", "in_proj.weight"]
        scale = picked["in_proj.weight"]
        assert torch.equal(scale, scales["quantizer.in_proj.weight"][: trimmed.out_channels])
        restored = quantize_to_int8(trimmed.weight, scale).to(torch.float32) * scale
        assert torch.equal(restored, trimmed.weight), "emit の逆変換ビット一致検査に落ちる形"

    def test_a_missing_in_proj_scale_fails_loudly(self):
        model, scales = self._quantized()
        wrapper = ex.EncoderGraph(model.encoder, ex.truncated_in_proj(model.quantizer.in_proj))
        del scales[ex.FULL_IN_PROJ_KEY]

        with pytest.raises(SystemExit, match="切り詰めの対応が取れない"):
            ex._target_scales(ex.TARGET_ENCODER, wrapper, scales)

    def test_f16_hands_no_scales_to_emit(self):
        model = TinyCodec(weight_norm=False)
        wrapper = ex.DecoderGraph(model.quantizer.out_proj, model.decoder)

        assert ex._target_scales(ex.TARGET_DECODER, wrapper, {}) == {}
