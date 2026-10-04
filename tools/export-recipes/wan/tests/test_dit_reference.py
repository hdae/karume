"""`wan/dit_reference.py`（DiT の層逐次の f64 / f32 参照）と `wan/dit_probe.py`（5B の RAM の
下見）の約束事（ADR 0121 決定 7・段 0）。

固定するのは、壊れても例外が出ず**数だけが静かにずれる**側:

- 層逐次の f32 / f64 が、上流のモデル全体で回した参照（f32 は `reference_dit_layers`・f64 は
  `model.double()` + `reference_dit_f64` — Wan2.1 の golden の作り方）とビット一致すること（f16 と
  i8 の両系列の丸め・読み口は in-memory のモデル / 容器 / checkpoint の 3 つ）
- 容器の i8 の逆量子化が f32 で行われること（f64 で行うと既存の golden と割れる — 値は
  `test_it_reads_every_weight_at_the_fake_quant_value`、f64 の値を返す読み口の拒否は
  `test_a_source_returning_f64_is_rejected`）
- 同時に実体を持つブロックが 1 個であること（5B の f64 の RAM の見積りの前提）
- f64 の経路で f64 以外の浮動小数を作ったら止まること（重みの読み込みは監視の外）
- 上流の mmap のずれの表が `from_pretrained` の実際の番地と一致すること
- 突き合わせ（`compare_tensors`）が 1 ビットの違いを拾うこと
- 5B の下見の meta trace のグラフが実重みの export と同じ構造で、行の塊ごとの i8 が fake-quant と
  同じ値になること

合成モデル（乱数初期化の小さな `WanTransformer3DModel` — `test_dit_patch.TINY_DIT`）だけで回す。
1.3B の実物の golden との突き合わせは時間がかかるので
`python -m wan.dit_reference compare`（runbook）。
"""

from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from wan import dit_patch, dit_reference, export_dit
from wan.tests.test_dit_patch import TINY_DIT, TINY_LATENT

#: 合成の層数（TINY_DIT の num_layers）。
LAYERS = int(TINY_DIT["num_layers"])


def _tiny(seed: int = 20261004) -> nn.Module:
    transformer_wan = pytest.importorskip("diffusers.models.transformers.transformer_wan")
    torch.manual_seed(seed)
    return transformer_wan.WanTransformer3DModel(**TINY_DIT).to(torch.float32).eval()


def _inputs(seed: int = 11) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, TINY_DIT["in_channels"], *TINY_LATENT, generator=generator)
    timestep = torch.tensor([750], dtype=torch.int64)
    embeds = torch.randn(1, 5, TINY_DIT["text_dim"], generator=generator)
    return latents, timestep, embeds


def _rounded(dtype: str, seed: int = 20261004) -> nn.Module:
    model = _tiny(seed)
    export_dit.fake_quant(dtype, model, dit_patch.WanDitTokens(model))
    return model


def _whole_f64(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Wan2.1 の f64 の参照の作り方（モデル全体を f64 にして上流の forward）。"""
    with torch.no_grad():
        return dit_patch.reference_dit_f64(copy.deepcopy(model).double(), *inputs)


@pytest.mark.parametrize("dtype", ["f16", "i8"])
class TestLayerwiseMatchesTheWholeModel:
    """in-memory のモデルを読み口にした層逐次 = モデル全体で回した上流の forward（ビット一致）。"""

    def test_f32_output_and_every_block_are_bit_identical(self, dtype: str) -> None:
        model = _rounded(dtype)
        inputs = _inputs()
        with torch.no_grad():
            expected, blocks = dit_patch.reference_dit_layers(model, *inputs)

        got = dit_reference.LayerwiseDit(model.config).forward(
            dit_reference.ModuleWeights(model), torch.float32, *inputs, collect_blocks=True
        )

        assert torch.equal(got.output, expected)
        assert len(got.blocks) == LAYERS
        assert all(torch.equal(a, b) for a, b in zip(got.blocks, blocks, strict=True))

    def test_f64_output_is_bit_identical_to_the_double_model(self, dtype: str) -> None:
        model = _rounded(dtype)
        inputs = _inputs()

        got = dit_reference.LayerwiseDit(model.config).forward(
            dit_reference.ModuleWeights(model), torch.float64, *inputs
        )

        assert got.output.dtype == torch.float64
        assert got.blocks == []
        assert torch.equal(got.output, _whole_f64(model, inputs))

    def test_a_different_weight_breaks_the_identity(self, dtype: str) -> None:
        """故障注入: 1 本の重みを 1 ULP ずらすと一致しない（恒真の門でないこと）。"""
        model = _rounded(dtype)
        inputs = _inputs()
        expected = _whole_f64(model, inputs)

        class Nudged:
            def tensor(self, key: str) -> torch.Tensor:
                value = dit_reference.ModuleWeights(model).tensor(key)
                if key == "blocks.1.ffn.net.2.weight":
                    return torch.nextafter(value, torch.full_like(value, float("inf")))
                return value

        got = dit_reference.LayerwiseDit(model.config).forward(Nudged(), torch.float64, *inputs)

        assert not torch.equal(got.output, expected)


class TestTheFloat64Path:
    def test_an_upstream_f32_cast_left_in_place_is_caught(self, monkeypatch) -> None:
        """`.float()` の素通しを外すと（上流の norm と残差の f32 化がそのまま走ると）止まる。"""
        model = _rounded("i8")
        monkeypatch.setattr(dit_patch, "_float_keeps_float64", nullcontext)

        with pytest.raises(dit_reference.DitReferenceError, match="f64 でない値"):
            dit_reference.LayerwiseDit(model.config).forward(
                dit_reference.ModuleWeights(model), torch.float64, *_inputs()
            )

    def test_the_tensor_float_method_is_restored(self) -> None:
        model = _rounded("f16")
        dit_reference.LayerwiseDit(model.config).forward(
            dit_reference.ModuleWeights(model), torch.float64, *_inputs()
        )

        assert torch.ones(2, dtype=torch.float64).float().dtype == torch.float32
        assert "float" not in vars(torch.Tensor)

    def test_a_per_token_timestep_stops_at_the_entrance_naming_stage_1(self) -> None:
        """TI2V のトークンごとの timestep（`[1,S]`）の f64 は段 1 の仕事 — 監視の奥で落ちる前に、
        段 1 で対応する形だと分かる文言で止まる。"""
        model = _rounded("i8")
        latents, _, embeds = _inputs()
        tokens = TINY_LATENT[0] * (TINY_LATENT[1] // 2) * (TINY_LATENT[2] // 2)
        per_token = torch.full((1, tokens), 750, dtype=torch.int64)

        with pytest.raises(dit_reference.DitReferenceError, match="段 1"):
            dit_reference.LayerwiseDit(model.config).forward(
                dit_reference.ModuleWeights(model), torch.float64, latents, per_token, embeds
            )

    def test_a_source_returning_f64_is_rejected(self) -> None:
        """読み口は f32 を返す — f64 の値を受けると「f64 で逆量子化した重み」が素通りしうる。"""
        model = _rounded("i8")

        class Wide:
            def tensor(self, key: str) -> torch.Tensor:
                return dit_reference.ModuleWeights(model).tensor(key).double()

        with pytest.raises(dit_reference.DitReferenceError, match="float64"):
            dit_reference.LayerwiseDit(model.config).forward(Wide(), torch.float64, *_inputs())

    def test_a_wrongly_shaped_weight_fails_loudly(self) -> None:
        """形を戻してよいのは patch 埋め込み（Linear 化の形）だけ。"""
        model = _rounded("i8")

        class Flat:
            def tensor(self, key: str) -> torch.Tensor:
                value = dit_reference.ModuleWeights(model).tensor(key)
                return value.reshape(-1) if key == "proj_out.weight" else value

        with pytest.raises(dit_reference.DitReferenceError, match=r"proj_out\.weight"):
            dit_reference.LayerwiseDit(model.config).forward(Flat(), torch.float32, *_inputs())


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    """上流の `save_pretrained` の形（config + 単一の safetensors）の合成 DiT。"""
    directory = tmp_path_factory.mktemp("dit-tiny")
    _tiny().save_pretrained(directory)
    return directory


def _upstream(checkpoint: Path) -> nn.Module:
    from diffusers import WanTransformer3DModel

    return WanTransformer3DModel.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()


def _container(model: nn.Module, dtype: str, directory: Path) -> Path:
    """合成のモデルを丸めて、系列と同じ形の容器を書く（`export_dit.emit` と同じ引数の export）。"""
    from karume.container import Provenance
    from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
    from karume.pipeline import export_to_file

    wrapper = dit_patch.WanDitTokens(model)
    _, scales = export_dit.fake_quant(dtype, model, wrapper)
    latents, timestep, embeds = _inputs()
    inputs = (
        dit_patch.dit_patchify(latents, (1, 2, 2)),
        dit_patch.dit_timesteps_proj(model, timestep),
        embeds,
        *dit_patch.dit_rope_tables(model.rope, TINY_LATENT),
    )
    path = directory / export_dit.MODEL_FILE
    export_to_file(
        wrapper,
        inputs,
        path,
        provenance=Provenance(license="apache-2.0"),
        graph_name=export_dit.TARGET,
        dynamic_shapes=export_dit.dynamic_shapes(),
        symbol_names=("S",),
        weight_dtype=dtype,
        weight_scales=scales,
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )
    return path


@pytest.mark.parametrize("dtype", ["f16", "i8"])
class TestTheContainerSource:
    """系列の容器から読む層逐次（5B の経路）= 上流のモデル全体の参照。"""

    def test_it_reads_every_weight_at_the_fake_quant_value(
        self, dtype: str, checkpoint: Path, tmp_path: Path
    ) -> None:
        """i8 は packed × scale を f32 で掛けた値（fake-quant と同じ）・f16 は f16 → f32。"""
        model = _upstream(checkpoint)
        path = _container(model, dtype, tmp_path)
        source = dit_reference.ContainerDitWeights(path)
        writer = dit_reference.LayerwiseDit(model.config)
        state = model.state_dict()

        for key in writer.weight_keys():
            assert torch.equal(source.tensor(key).reshape(state[key].shape), state[key]), key

    def test_f64_and_aligned_f32_are_bit_identical_to_the_upstream_reference(
        self, dtype: str, checkpoint: Path, tmp_path: Path
    ) -> None:
        model = _upstream(checkpoint)
        path = _container(model, dtype, tmp_path)
        inputs = _inputs()
        with torch.no_grad():
            expected32, blocks = dit_patch.reference_dit_layers(model, *inputs)
        writer = dit_reference.LayerwiseDit(
            model.config, alignment=dit_reference.upstream_alignment(checkpoint)
        )
        source = dit_reference.ContainerDitWeights(path)

        got32 = writer.forward(source, torch.float32, *inputs, collect_blocks=True)
        got64 = writer.forward(source, torch.float64, *inputs)

        assert torch.equal(got32.output, expected32)
        assert all(torch.equal(a, b) for a, b in zip(got32.blocks, blocks, strict=True))
        assert torch.equal(got64.output, _whole_f64(model, inputs))


class TestOneBlockAtATime:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
    def test_the_previous_block_is_released_before_the_next_is_read(
        self, dtype: torch.dtype
    ) -> None:
        """次のブロックの重みを読み始める時点で、実体を持つ（meta でない）層逐次のブロックは 0 個。

        上流の `for block in self.blocks` のループ変数は前のブロックを握ったまま次を求めるので、
        前のブロックの重みを手放さないと 2 個が同時に生きる（5B の f64 で 1.22 → 2.44 GiB）。
        """
        import gc

        from diffusers.models.transformers.transformer_wan import WanTransformerBlock

        model = _rounded("i8")
        upstream_blocks = {id(block) for block in model.blocks}
        writer = dit_reference.LayerwiseDit(model.config)
        first = next(iter(writer.block_shapes))
        weights = dit_reference.ModuleWeights(model)
        live: list[int] = []

        class Counting:
            def tensor(self, key: str) -> torch.Tensor:
                if key.startswith("blocks.") and key.endswith(f".{first}"):
                    gc.collect()  # 到達できない循環の残り（回収待ち）は「生きている」に数えない
                    live.append(
                        sum(
                            type(item) is WanTransformerBlock
                            and id(item) not in upstream_blocks
                            and any(not p.is_meta for p in item.parameters())
                            for item in gc.get_objects()
                        )
                    )
                return weights.tensor(key)

        writer.forward(Counting(), dtype, *_inputs())

        assert live == [0] * LAYERS


class TestTheUpstreamAlignment:
    def test_it_predicts_where_from_pretrained_puts_every_weight(self, checkpoint: Path) -> None:
        """ずれの表 = `from_pretrained` で読んだ重みの実際の番地の 64 バイト境界からのずれ。"""
        model = _upstream(checkpoint)
        table = dit_reference.upstream_alignment(checkpoint)

        actual = {
            key: value.data_ptr() % dit_reference.ALIGNMENT_BYTES
            for key, value in model.state_dict().items()
        }
        assert table == actual

    @pytest.mark.parametrize("offset", [0, 8, 36, 60])
    def test_placing_keeps_the_value_and_lands_on_the_offset(self, offset: int) -> None:
        value = torch.randn(5, 7)

        placed = dit_reference._placed(value, offset)

        assert placed.data_ptr() % dit_reference.ALIGNMENT_BYTES == offset
        assert placed.is_contiguous()
        assert torch.equal(placed, value)

    def test_a_key_outside_the_table_fails_loudly(self) -> None:
        model = _rounded("i8")
        writer = dit_reference.LayerwiseDit(model.config, alignment={})

        with pytest.raises(dit_reference.DitReferenceError, match="ずれが無い"):
            writer.forward(dit_reference.ModuleWeights(model), torch.float32, *_inputs())


class TestTheCheckpointSource:
    """checkpoint から 1 本ずつ丸める読み口 = モデル全体に掛けた丸め（ビット一致）。"""

    @pytest.mark.parametrize("dtype", ["f16", "i8"])
    def test_each_weight_equals_the_whole_model_rounding(
        self, dtype: str, checkpoint: Path
    ) -> None:
        model = _upstream(checkpoint)
        export_dit.fake_quant(dtype, model, dit_patch.WanDitTokens(model))
        source = dit_reference.CheckpointDitWeights(
            checkpoint, dtype, dit_reference.quant_keys(model.config)
        )

        for key, value in model.state_dict().items():
            assert torch.equal(source.tensor(key), value), key

    def test_the_unrounded_source_returns_the_checkpoint_values(self, checkpoint: Path) -> None:
        model = _upstream(checkpoint)
        source = dit_reference.CheckpointDitWeights(checkpoint, "none")

        for key, value in model.state_dict().items():
            assert torch.equal(source.tensor(key), value), key

    def test_the_quant_keys_are_the_weights_the_export_rounds(self) -> None:
        """量子化の対象 = export の `fake_quant_i8` の scale 台帳のキー（linear と patch
        埋め込み）。"""
        model = _tiny()
        report = export_dit.fake_quant_i8(model, dit_patch.WanDitTokens(model))

        assert dit_reference.quant_keys(model.config) == frozenset(report.scales)
        assert "patch_embedding.weight" in report.scales

    def test_i8_without_targets_is_refused(self, checkpoint: Path) -> None:
        with pytest.raises(dit_reference.DitReferenceError, match="quant_keys"):
            dit_reference.CheckpointDitWeights(checkpoint, "i8")


class TestCompareTensors:
    def _file(self, tmp_path: Path, tensors: dict[str, torch.Tensor]) -> Path:
        from safetensors.torch import save_file

        path = tmp_path / "reference.case.safetensors"
        save_file(tensors, str(path))
        return path

    def test_identical_tensors_have_no_mismatch(self, tmp_path: Path) -> None:
        tensors = {"output": torch.randn(3, 4), "timestep": torch.tensor([7], dtype=torch.int32)}

        result = dit_reference.compare_tensors(tensors, self._file(tmp_path, tensors))

        assert result == {"keys": ["output", "timestep"], "mismatched": {}}

    def test_one_flipped_bit_and_a_missing_key_are_named(self, tmp_path: Path) -> None:
        tensors = {"output": torch.randn(3, 4), "block.00": torch.randn(2)}
        path = self._file(tmp_path, tensors)
        flipped = tensors["output"].clone()
        flipped.view(torch.int32)[1, 2] ^= 1

        result = dit_reference.compare_tensors({"output": flipped, "extra": torch.ones(1)}, path)

        assert set(result["mismatched"]) == {"output", "block.00", "extra"}
        assert "ビットが違う" in result["mismatched"]["output"]
        assert result["mismatched"]["block.00"] == "書いた側に無い"
        assert result["mismatched"]["extra"] == "既存に無い"

    def test_negative_zero_is_not_equal_to_zero(self, tmp_path: Path) -> None:
        path = self._file(tmp_path, {"output": torch.zeros(2)})

        result = dit_reference.compare_tensors({"output": torch.tensor([0.0, -0.0])}, path)

        assert "output" in result["mismatched"]

    def test_only_the_named_keys_are_compared(self, tmp_path: Path) -> None:
        path = self._file(tmp_path, {"output": torch.zeros(2), "output.f64": torch.ones(2)})

        result = dit_reference.compare_tensors({"output.f64": torch.ones(2)}, path, ["output.f64"])

        assert result == {"keys": ["output.f64"], "mismatched": {}}


def _row(case: str, mode: str, version: int = dit_reference.COMPARE_VERSION) -> dict[str, Any]:
    return {"case": case, "status": "equal", "mode": mode, "compare_version": version}


class TestTheResumeMarker:
    """`results.jsonl` の行のうち、この実行で「済み」に数えるもの（再開の目印）。"""

    def _write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / dit_reference.RESULTS_FILE
        path.write_text(text, encoding="utf-8")
        return path

    def _lines(self, *rows: dict[str, Any]) -> str:
        return "".join(json.dumps(row) + "\n" for row in rows)

    def test_an_f64_only_row_does_not_count_for_a_full_run(self, tmp_path: Path) -> None:
        """f64 だけの行で全量の実行を飛ばすと、f32 の output と block を比べないまま緑になる。"""
        path = self._write(tmp_path, self._lines(_row("a", "f64-only"), _row("b", "full")))

        assert sorted(dit_reference._done(path, "full")) == ["b"]
        assert sorted(dit_reference._done(path, "f64-only")) == ["a", "b"]

    def test_a_row_of_another_compare_version_does_not_count(self, tmp_path: Path) -> None:
        old = {"case": "a", "status": "equal"}  # 版の欄を持たない行（版 1 の形）
        path = self._write(
            tmp_path,
            self._lines(
                old, _row("b", "full", dit_reference.COMPARE_VERSION - 1), _row("c", "full")
            ),
        )

        assert sorted(dit_reference._done(path, "full")) == ["c"]

    def test_a_half_written_last_line_is_dropped_and_cut_off(self, tmp_path: Path) -> None:
        whole = self._lines(_row("a", "full"))
        path = self._write(tmp_path, whole + json.dumps(_row("b", "full"))[:17])

        assert sorted(dit_reference._done(path, "full")) == ["a"]
        assert path.read_text(encoding="utf-8") == whole

    def test_an_unreadable_line_before_the_last_fails_loudly(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, '{"case": \n' + self._lines(_row("a", "full")))

        with pytest.raises(dit_reference.DitReferenceError, match=":1 の行が読めない"):
            dit_reference._done(path, "full")

    def test_the_numeric_environment_records_the_mkl_switches(self, monkeypatch) -> None:
        monkeypatch.setenv("MKL_CBWR", "COMPATIBLE")
        monkeypatch.setenv("OMP_NUM_THREADS", "3")

        environment = dit_reference.numeric_environment()

        assert environment["env"]["MKL_CBWR"] == "COMPATIBLE"
        assert environment["env"]["OMP_NUM_THREADS"] == "3"
        assert environment["threads"] == torch.get_num_threads()
        assert environment["torch"] == torch.__version__


class TestTheExportDitRoute:
    """`export_dit.float64_references` は層逐次を通り、Wan2.1 の全体 f64 の値を変えない。"""

    def test_it_reproduces_the_whole_model_f64_reference(self, monkeypatch) -> None:
        model = _rounded("i8")
        spec = export_dit.CaseSpec("full-band", TINY_LATENT, 999, 5, 2, blocks=False)
        monkeypatch.setattr(export_dit, "TEXT_DIM", TINY_DIT["text_dim"])
        monkeypatch.setattr(export_dit, "pad_text_embeds", lambda embeds: embeds.unsqueeze(0))
        inputs = export_dit.case_inputs(model, spec)

        references = export_dit.float64_references(model, [spec], "i8")

        (reference,) = references.values()
        assert list(references) == [spec.name((1, 2, 2))]
        assert torch.equal(reference.output, _whole_f64(model, inputs))

    @pytest.mark.parametrize("dtype", ["f16", "i8"])
    def test_a_model_before_the_fake_quant_is_refused(self, dtype: str) -> None:
        """呼ぶ順が崩れて丸める前のモデルが渡ると止まる（丸めていない f64 golden を書かない）。"""
        spec = export_dit.CaseSpec("full-band", TINY_LATENT, 999, 5, 2, blocks=False)

        with pytest.raises(AssertionError, match="丸めの前の重み"):
            export_dit.float64_references(_tiny(), [spec], dtype)

    def test_cases_without_an_f64_reference_take_nothing(self) -> None:
        spec = export_dit.CaseSpec("band", TINY_LATENT, 999, 5, 2)

        assert export_dit.float64_references(_rounded("f16"), [spec], "f16") == {}


# ---- 5B の下見（`wan.dit_probe`）--------------------------------------------------------


def _stage(name: str) -> Any:
    from wan.umt5_export import StageRecord

    return nullcontext(StageRecord(name))


class TestTheProbe:
    def test_the_meta_trace_has_the_structure_of_the_real_weight_export(self) -> None:
        """meta の重み + meta の例示入力の trace = 実重みの export と同じ op・配線・値の形。

        値の名前（FX の連番）だけは違う — 構造で比べる（ノードの出力を生産者の位置へ置き換える）。
        """
        from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
        from karume.pipeline import export_module
        from wan import dit_probe

        model = _tiny()
        config = dict(model.config)
        real_inputs = tuple(
            torch.randn(tensor.shape) for tensor in dit_probe.meta_inputs(config, _PROBE_SPEC)
        )
        meta_graph, tensors, targets = dit_probe.trace(config, sym_max=256)
        real_graph, _ = export_module(
            dit_patch.WanDitTokens(model),
            real_inputs,
            dynamic_shapes=dit_probe.dynamic_shapes(256),
            symbol_names=("S",),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )

        assert _structure(meta_graph) == _structure(real_graph)
        assert len(targets) == len(
            export_dit.fake_quant_i8(model, dit_patch.WanDitTokens(model)).scales
        )
        assert sorted(key for key, value in tensors.items() if value.is_meta) == sorted(
            dict(model.named_parameters())
        )

    def test_prepare_rounds_like_the_fake_quant_and_stores_i8_linears(self, checkpoint: Path):
        """行の塊ごとの i8（patch 埋め込みは Conv3d を 2 次元に見て）= モデル全体の fake-quant。"""
        from wan import dit_probe

        export, summary = dit_probe.prepare(checkpoint, stage=_stage, sym_max=256, chunk_rows=5)
        model = _upstream(checkpoint)
        report = export_dit.fake_quant_i8(model, dit_patch.WanDitTokens(model))
        state = model.state_dict()

        assert sorted(export.fixed) == sorted(report.scales)
        for key, weight in export.fixed.items():
            restored = (weight.packed.to(torch.float32) * weight.scale).reshape(state[key].shape)
            assert torch.equal(restored, state[key]), key
        assert all(torch.equal(export.tensors[key], state[key]) for key in export.plain)
        stored = dit_probe.store(export)
        assert stored["storage"] == {
            "f32": len(export.graph.initializers) - len(export.fixed),
            "i8": len(export.fixed),
        }
        assert summary["packed_bytes"] == sum(state[key].numel() for key in export.fixed)


class TestTheProbeMemory:
    def test_the_estimate_counts_the_real_parameters(self) -> None:
        """f32 = 上流の全パラメータ・層逐次の f64 = ブロックの外 + 1 ブロック（meta の雛形から
        数えた値が、実体のモデルの数と同じ）。"""
        from wan import dit_probe

        model = _tiny()
        parameters = dict(model.named_parameters())
        outer = sum(p.numel() for name, p in parameters.items() if not name.startswith("blocks."))
        block = sum(p.numel() for name, p in parameters.items() if name.startswith("blocks.0."))

        estimate = dit_probe.eager_memory(dict(model.config))

        assert estimate["f32"] == sum(p.numel() for p in parameters.values()) * 4
        assert estimate["layerwise_f64"] == (outer + block) * 8
        assert estimate["peak"] == (
            estimate["f32"] + estimate["layerwise_f64"] + dit_probe.EAGER_BASE_BYTES
        )

    def test_eager_stops_before_loading_when_memory_is_short(
        self, checkpoint: Path, tmp_path: Path, monkeypatch
    ) -> None:
        from wan import dit_probe

        def never(*_args: Any) -> None:
            raise AssertionError("RAM が足りないのに上流を読んだ")

        monkeypatch.setattr(dit_probe, "transformer_dir", lambda _model: checkpoint)
        monkeypatch.setattr(dit_probe, "mem_available", lambda: 2**30)
        monkeypatch.setattr(export_dit, "load_transformer", never)

        with pytest.raises(dit_probe.DitProbeError, match=r"load の前の MemAvailable 1\.00 GiB"):
            dit_probe.run_eager("t2v-1.3b", tmp_path)

    def test_the_default_outputs_live_under_the_bench_root(self) -> None:
        from _shared.paths import BENCH_ROOT
        from wan import dit_probe
        from wan.sources import SOURCES

        assert set(dit_probe.BENCH_NAMES) == set(SOURCES)
        for model in SOURCES:
            assert dit_probe.default_out(model).parent.parent == BENCH_ROOT
        assert dit_reference.DEFAULT_OUT.parent.parent == BENCH_ROOT


_PROBE_SPEC = export_dit.CaseSpec("band", TINY_LATENT, 999, 5, 1)


def _structure(graph: Any) -> list[tuple[str, list[str], str, list[str]]]:
    """グラフの構造（op・入力の生産者・attrs・出力の形）— 値の名前に依らない。"""
    document = json.loads(graph.to_json())
    produced: dict[str, str] = {}
    rows = []
    for position, node in enumerate(document["nodes"]):
        sources = [produced.get(name, name) for name in node["ins"]]
        shapes = [json.dumps(document["values"][name], sort_keys=True) for name in node["outs"]]
        rows.append(
            (node["op"], sources, json.dumps(node.get("attrs", {}), sort_keys=True), shapes)
        )
        for index, name in enumerate(node["outs"]):
            produced[name] = f"#{position}.{index}"
    return rows
