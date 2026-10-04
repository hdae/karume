"""`wan/ti2v_export_dit.py`（Wan2.2 TI2V-5B の DiT の i8 系列と小さい S の golden — ADR 0121
段 1）の約束事。

固定するのは、壊れても例外が出ず**数だけが静かにずれる**側と、段 1 の検収の形:

- ケースの表（決定用 6 本 = T2V 3 + I2V 3・受入れ 4 本 = T2V 2 + I2V 2・全て S = 192）
- meta trace のグラフが実重みの export と同じ構造であること（7 入力・条件マスクは bool `[1,S,1]`）
- IR の検査: linear のノード = 量子化の対象 + 時刻の MLP の 3 本・`where` がブロックあたり 6 本 +
  head の 2 本で、どれも消費者が直後の 1 ノード（故障注入: 上流の順でブロックの先頭へ寄せる・
  時刻の MLP を M = 2 の 1 回にまとめる — どちらも名指しで落ちる）
- 容器の格納: i8 = 量子化の対象ちょうど・bias / norm / `scale_shift_table` は f32
- golden: 7 入力と eager 出力・参照（f32 / f64・条件側の timestep）・I2V の条件マスク
- 作業席の規律: 門（IR の検査・eager 同値・本数の期待値）に落ちたら系列は 1 バイトも変わらない

合成モデル（`test_dit_patch.TINY_DIT` を `save_pretrained` した checkpoint）だけで回す。5B の実物は
`python -m wan.ti2v_export_dit write`（runbook）。
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from torch import nn

from wan import dit_patch, dit_probe, dit_reference, export_dit, ti2v_export_dit
from wan.tests.test_dit_patch import EXPECTED_OPS, TINY_DIT

#: 合成の層数（TINY_DIT の num_layers）。
LAYERS = int(TINY_DIT["num_layers"])

#: trace の S の上限（合成 — 5B の 27,280 だと trace が重いだけで構造は同じ）。
SYM_MAX = 256


def _tiny(seed: int = 20261004) -> nn.Module:
    transformer_wan = pytest.importorskip("diffusers.models.transformers.transformer_wan")
    torch.manual_seed(seed)
    return transformer_wan.WanTransformer3DModel(**TINY_DIT).to(torch.float32).eval()


def _stage(name: str) -> Any:
    from wan.umt5_export import StageRecord

    return nullcontext(StageRecord(name))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    """上流の `save_pretrained` の形（config + 単一の safetensors）の合成 DiT。"""
    directory = tmp_path_factory.mktemp("ti2v-tiny")
    _tiny().save_pretrained(directory)
    return directory


def _tracer(config: Any, _sym_max: int) -> Any:
    return ti2v_export_dit.trace(config, SYM_MAX)


def _tiny_text(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    """golden の文脈を合成の text_dim（8）で作る（実物は埋め込み資産の `[L, 4096]` を 512 行へ
    埋める）。固定 4 プロンプトの名前で、合成の埋め込みの資産を `directory` に書いて差し替える。"""
    from safetensors.torch import save_file

    from wan.prompts import FIXED_PROMPTS
    from wan.text_embeds import METADATA_KEY

    generator = torch.Generator().manual_seed(5)
    asset = directory / "text_embeds.safetensors"
    save_file(
        {
            prompt.name: torch.randn(3 + index, TINY_DIT["text_dim"], generator=generator)
            for index, prompt in enumerate(FIXED_PROMPTS)
        },
        str(asset),
        metadata={METADATA_KEY: "{}"},
    )
    monkeypatch.setattr(dit_probe, "TEXT_EMBEDS_ASSET", asset)
    monkeypatch.setattr(dit_probe, "pad_text_embeds", lambda embeds: embeds.unsqueeze(0))
    monkeypatch.setattr(export_dit, "TEXT_DIM", TINY_DIT["text_dim"])
    monkeypatch.setattr(export_dit, "pad_text_embeds", lambda embeds: embeds.unsqueeze(0))


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


class TestTheCases:
    PATCH = (1, 2, 2)

    def test_the_roles_and_forms_have_the_stage_2_counts(self) -> None:
        """決定用 6 本（T2V 3 + I2V 3）・受入れ 4 本（T2V 2 + I2V 2）— 段 2 の r 門の本数。"""
        counts = {
            (role, form): sum(c.role == role and c.form == form for c in ti2v_export_dit.CASES)
            for role in ("band", "accept")
            for form in ("t2v", "i2v")
        }

        assert counts == {
            ("band", "t2v"): 3,
            ("band", "i2v"): 3,
            ("accept", "t2v"): 2,
            ("accept", "i2v"): 2,
        }

    def test_every_case_has_192_tokens_and_a_unique_name_and_seed(self) -> None:
        cases = ti2v_export_dit.CASES

        assert {case.tokens(self.PATCH) for case in cases} == {192}
        assert len({case.name(self.PATCH) for case in cases}) == len(cases)
        assert len({case.seed for case in cases}) == len(cases)
        assert not {case.seed for case in cases} & {spec.seed for spec in export_dit.CASES}

    def test_every_context_is_a_fixed_prompt_and_every_prompt_is_used(self) -> None:
        """文脈は埋め込み資産の固定 4 プロンプト（合成の乱数は 5B の f32 / f64 の差を増幅）。"""
        from wan.prompts import FIXED_PROMPTS

        texts = [case.text for case in ti2v_export_dit.CASES]

        assert set(texts) == {prompt.name for prompt in FIXED_PROMPTS}

    def test_i2v_cases_keep_generated_frames_and_grids_are_not_all_square(self) -> None:
        """I2V は F' ≥ 2（F' = 1 だと全トークンが条件）。格子は非正方を含む（h / w の取り違え）。"""
        cases = ti2v_export_dit.CASES

        assert all(case.latent_shape[0] >= 2 for case in cases if case.form == "i2v")
        assert any(case.latent_shape[1] != case.latent_shape[2] for case in cases)


class TestTheTrace:
    def test_the_meta_trace_has_the_structure_of_the_real_weight_export(self) -> None:
        from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
        from karume.pipeline import export_module

        model = _tiny()
        config = dict(model.config)
        meta_graph, tensors, targets = ti2v_export_dit.trace(config, SYM_MAX)
        real_inputs = tuple(
            torch.zeros(tensor.shape, dtype=tensor.dtype)
            for tensor in ti2v_export_dit.meta_inputs(config, ti2v_export_dit.TRACE_LATENT)
        )
        real_graph, _ = export_module(
            dit_patch.WanDitTokensTi2v(model),
            real_inputs,
            dynamic_shapes=ti2v_export_dit.dynamic_shapes(SYM_MAX),
            symbol_names=("S",),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )

        assert _structure(meta_graph) == _structure(real_graph)
        assert sorted(key for key, value in tensors.items() if value.is_meta) == sorted(
            dict(model.named_parameters())
        )
        assert sorted(targets) == sorted(dit_reference.quant_keys(config))

    def test_the_graph_declares_seven_inputs_with_a_boolean_mask(self) -> None:
        graph, _, _ = ti2v_export_dit.trace(dict(_tiny().config), SYM_MAX)

        assert graph.symbols == ["S"]
        assert [[e.name, list(e.shape), e.dtype] for e in graph.inputs] == [
            ["tokens", [1, "S", 12], "f32"],
            ["timesteps_proj", [1, 16], "f32"],
            ["encoder_hidden_states", [1, 512, 8], "f32"],
            ["rope_cos", [1, "S", 1, 24], "f32"],
            ["rope_sin", [1, "S", 1, 24], "f32"],
            ["timesteps_proj_condition", [1, 16], "f32"],
            ["condition_mask", [1, "S", 1], "bool"],
        ]
        assert set(graph.required_ops) == EXPECTED_OPS | {"where"}


def _hoisted_block(
    block: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    generation: torch.Tensor,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    rotary_emb: Any,
) -> torch.Tensor:
    """故障注入用: 上流の順（ブロックの先頭で 6 成分の `where` をまとめて作る）で書いたブロック。"""
    table = block.scale_shift_table
    shift, scale, gate, c_shift, c_scale, c_gate = (
        torch.where(condition_mask, c, g)
        for c, g in zip(
            (table + condition.float()).chunk(6, dim=1),
            (table + generation.float()).chunk(6, dim=1),
            strict=True,
        )
    )
    norm = (block.norm1(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = (
        hidden_states.float() + block.attn1(norm, None, None, rotary_emb) * gate
    ).type_as(hidden_states)
    norm = block.norm2(hidden_states.float()).type_as(hidden_states)
    hidden_states = hidden_states + block.attn2(norm, encoder_hidden_states, None, None)
    norm = (block.norm3(hidden_states.float()) * (1 + c_scale) + c_shift).type_as(hidden_states)
    return (hidden_states.float() + block.ffn(norm).float() * c_gate).type_as(hidden_states)


class TestTheStructureInspection:
    def test_the_patch_passes_with_three_extra_linears_and_adjacent_wheres(self) -> None:
        graph, _, targets = ti2v_export_dit.trace(dict(_tiny().config), SYM_MAX)

        inspection = ti2v_export_dit.inspect_structure(graph, sorted(targets), LAYERS)

        assert inspection["failures"] == []
        assert inspection["linear_nodes"] == len(targets) + 3
        assert inspection["where"] == 6 * LAYERS + 2

    def test_wheres_hoisted_to_the_block_top_are_named(self, monkeypatch) -> None:
        """上流の順で 6 成分の `where` を先頭にまとめると、attention
        をまたいで生きる出力として名指す
        （値は同じなので eager 同値では見えない — IR の検査だけが拾う）。"""
        monkeypatch.setattr(dit_patch, "_ti2v_block", _hoisted_block)
        graph, _, targets = ti2v_export_dit.trace(dict(_tiny().config), SYM_MAX)

        inspection = ti2v_export_dit.inspect_structure(graph, sorted(targets), LAYERS)

        assert inspection["where"] == 6 * LAYERS + 2
        crossing = [f for f in inspection["failures"] if "attention をまたぐ" in f]
        # ブロックあたり、先頭の 2 本（shift / scale）は直後に消費されうるが、gate と FFN の 3 本は
        # attention の後まで生きる。
        assert len(crossing) >= 4 * LAYERS

    def test_a_time_mlp_run_once_for_both_times_is_named(self, monkeypatch) -> None:
        """時刻の MLP を M = 2 の 1 回にまとめると linear のノードが 3 本足りない（決定 3 の
        「M = 2 の 1 回にしない」）。"""

        def batched(
            embedder: nn.Module,
            generation: torch.Tensor,
            condition: torch.Tensor,
            like: torch.Tensor,
        ) -> Any:
            temb = embedder.time_embedder(torch.cat([generation, condition])).type_as(like)
            proj = embedder.time_proj(embedder.act_fn(temb))
            return (temb[:1], proj[:1]), (temb[1:], proj[1:])

        monkeypatch.setattr(dit_patch, "_two_time_embeddings", batched)
        model = _tiny()
        graph, _, targets = ti2v_export_dit.trace(dict(model.config), SYM_MAX)

        inspection = ti2v_export_dit.inspect_structure(graph, sorted(targets), LAYERS)

        assert any("linear のノードが" in f for f in inspection["failures"])

    def test_the_5b_counts_are_checked_only_for_the_5b(self) -> None:
        inspection = {"i8_weights": 306, "linear_nodes": 310}

        assert ti2v_export_dit.expected_counts("ti2v-5b", inspection) == [
            "i8_weights が 306（決定 2 の期待値 307）"
        ]
        assert ti2v_export_dit.expected_counts("t2v-1.3b", inspection) == []


class TestTheContainer:
    def test_linears_are_i8_and_everything_else_keeps_the_source_f32(
        self, checkpoint: Path, tmp_path: Path
    ) -> None:
        export, _ = dit_probe.prepare(checkpoint, stage=_stage, chunk_rows=5, tracer=_tracer)
        writer = dit_reference.LayerwiseDit(dit_reference.load_config(checkpoint))
        path = tmp_path / ti2v_export_dit.MODEL_FILE

        graph = ti2v_export_dit.write_container(
            export, path, ti2v_export_dit.rope_base_asset(writer)
        )

        storage = ti2v_export_dit.inspect_storage(graph, sorted(export.fixed), export.plain)
        assert storage["failures"] == []
        assert storage["i8_weights"] == len(dit_reference.quant_keys(writer.config))
        assert storage["f32_weights"] == len(export.plain)
        source = dit_reference.ContainerDitWeights(path)
        rounded = dit_reference.CheckpointDitWeights(
            checkpoint, "i8", dit_reference.quant_keys(writer.config)
        )
        for key in writer.weight_keys():
            assert torch.equal(
                source.tensor(key).reshape(rounded.tensor(key).shape), rounded.tensor(key)
            ), key


@pytest.fixture(scope="module")
def written(checkpoint: Path, tmp_path_factory) -> tuple[Path, dict[str, Any]]:
    """合成の checkpoint で系列を書く（`transformer_dir` / `SERIES` / 本数の期待値を合成へ）。"""
    series = tmp_path_factory.mktemp("series") / ti2v_export_dit.SERIES_NAME
    with pytest.MonkeyPatch.context() as patch:
        _tiny_text(patch, tmp_path_factory.mktemp("text"))
        _point_at_tiny(patch, checkpoint, series)
        summary = ti2v_export_dit.write_series()
    return series / ti2v_export_dit.TARGET, summary


def _point_at_tiny(patch: pytest.MonkeyPatch, checkpoint: Path, series: Path) -> None:
    patch.setattr(ti2v_export_dit, "transformer_dir", lambda _model=None: checkpoint)
    patch.setattr(ti2v_export_dit, "SERIES", series)
    patch.setattr(ti2v_export_dit, "SYM_MAX", SYM_MAX)
    targets = len(dit_reference.quant_keys(dit_reference.load_config(checkpoint)))
    patch.setattr(
        ti2v_export_dit, "EXPECTED_5B", {"i8_weights": targets, "linear_nodes": targets + 3}
    )


class TestTheSeriesWriter:
    def test_every_case_gets_an_io_and_a_reference_file_next_to_the_container(
        self, written
    ) -> None:
        target, summary = written
        names = [case.name((1, 2, 2)) for case in ti2v_export_dit.CASES]

        assert summary["passed"] is True
        assert summary["inspection"]["failures"] == []
        assert [report["case"] for report in summary["eager"]] == names
        assert (target / "model-00001-of-00001.krm").is_file() or any(target.glob("model-*.krm"))
        for name in names:
            assert (target / f"io.{name}.safetensors").is_file()
            assert (target / f"reference.{name}.safetensors").is_file()

    def test_the_io_file_holds_seven_inputs_and_the_patched_output(self, written) -> None:
        target, _ = written
        with safe_open(str(target / "io.band-i2v-s00192-t0999.safetensors"), "pt") as handle:
            keys = set(handle.keys())
            mask = handle.get_tensor("input.condition_mask")
        with safe_open(str(target / "io.band-t2v-s00192-t0999.safetensors"), "pt") as handle:
            t2v_mask = handle.get_tensor("input.condition_mask")

        assert keys == {f"input.{name}" for name in ti2v_export_dit.INPUT_NAMES} | {"output.0"}
        assert mask.dtype == torch.uint32
        # 潜在 (3, 16, 16) → 1 フレーム 8·8 = 64 トークンが条件フレーム。
        assert mask.flatten().to(torch.int64).tolist() == [1] * 64 + [0] * 128
        assert not t2v_mask.to(torch.int64).any()

    def test_the_reference_holds_both_references_and_the_condition_timestep(self, written) -> None:
        target, _ = written
        expected = {"latents", "timestep", "condition_timestep", "output", "output.f64"}
        for name, condition in (("band-i2v-s00192-t0750", 0), ("band-t2v-s00192-t0500", 500)):
            with safe_open(str(target / f"reference.{name}.safetensors"), "pt") as handle:
                assert set(handle.keys()) == expected
                assert handle.get_tensor("condition_timestep").tolist() == [condition]
                assert handle.get_tensor("timestep").dtype == torch.int32
                output = handle.get_tensor("output")
                f64 = handle.get_tensor("output.f64")
            assert output.dtype == f64.dtype == torch.float32
            assert not torch.equal(output, f64)

    def test_every_eager_report_passed_the_gate_and_records_the_per_token_path(
        self, written
    ) -> None:
        _, summary = written
        for report in summary["eager"]:
            assert export_dit.eager_failures(report) == [], report["case"]
            assert report["trunk_bit_exact"] is True
            observed = "per_token_timestep_vs_reference_ratio" in report
            assert observed is report["case"].startswith("band-"), report["case"]
            if observed:
                assert isinstance(report["per_token_timestep_vs_reference_ratio"], float)

    def test_a_narrowed_run_does_not_publish(
        self, checkpoint: Path, tmp_path: Path, monkeypatch
    ) -> None:
        series = tmp_path / ti2v_export_dit.SERIES_NAME
        _tiny_text(monkeypatch, tmp_path)
        _point_at_tiny(monkeypatch, checkpoint, series)

        with pytest.raises(ti2v_export_dit.Ti2vExportError, match="系列を据えない"):
            ti2v_export_dit.write_series(names=["band-t2v-s00192-t0999"])

        assert not (series / ti2v_export_dit.TARGET).exists()

    @pytest.mark.parametrize("fault", ["swapped-where", "expected-count"])
    def test_a_broken_gate_leaves_the_existing_series_untouched(
        self, fault: str, checkpoint: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """門（eager 同値・本数の期待値）に落ちたら、据わっている系列は 1 バイトも変わらない。"""
        series = tmp_path / ti2v_export_dit.SERIES_NAME
        target = series / ti2v_export_dit.TARGET
        target.mkdir(parents=True)
        (target / "marker").write_text("before", encoding="utf-8")
        _tiny_text(monkeypatch, tmp_path)
        _point_at_tiny(monkeypatch, checkpoint, series)
        if fault == "swapped-where":
            real = dit_patch._ti2v_block

            def swapped(block, hidden, text, generation, condition, mask, rope):  # type: ignore[no-untyped-def]
                return real(block, hidden, text, condition, generation, mask, rope)

            monkeypatch.setattr(dit_patch, "_ti2v_block", swapped)
            expected_error: type[Exception] = export_dit.EagerEquivalenceError
        else:
            monkeypatch.setattr(ti2v_export_dit, "EXPECTED_5B", {"linear_nodes": 1})
            expected_error = ti2v_export_dit.Ti2vExportError

        with pytest.raises(expected_error):
            ti2v_export_dit.write_series()

        assert [path.name for path in target.iterdir()] == ["marker"]
        assert (target / "marker").read_text(encoding="utf-8") == "before"


def _point_eager_full_at(patch: pytest.MonkeyPatch, checkpoint: Path, series: Path) -> None:
    """`eager-full` を合成の checkpoint（`from_pretrained` で全量）と書いた系列へ向ける。"""

    def load(_model: str) -> nn.Module:
        from diffusers import WanTransformer3DModel

        return WanTransformer3DModel.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()

    _point_at_tiny(patch, checkpoint, series)
    patch.setattr(export_dit, "load_transformer", load)


class TestTheFullEager:
    """独立した重み（f32 の全量 1 本）での eager 同値と、据えた golden との突き合わせ
    （決定 4 の形）。"""

    def test_the_full_model_agrees_with_the_patched_graph_and_the_written_golden(
        self, written, checkpoint: Path, tmp_path: Path, monkeypatch
    ) -> None:
        target, _ = written
        _tiny_text(monkeypatch, tmp_path)
        _point_eager_full_at(monkeypatch, checkpoint, target.parent)

        summary = ti2v_export_dit.eager_full()

        assert summary["passed"] is True, summary["failures"]
        assert [report["form"] for report in summary["cases"]] == [
            case.form for case in ti2v_export_dit.CASES
        ]
        for report in summary["cases"]:
            assert report["trunk_bit_exact"] is True, report["case"]
            assert report["golden_inputs_bit_exact"] is True, report["case"]
            assert report["golden_reference_bit_exact"] is True, report["case"]
            assert report["golden_output_bit_exact"] is True, report["case"]

    def test_a_container_read_error_that_the_write_gate_cannot_see_is_caught(
        self, checkpoint: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """故障注入: 容器の読み口が 1 本の重みを 1 ULP ずらす。`write` の門は両側が同じ読み口を
        通るので緑のまま golden を据える — `eager-full` だけが golden との食い違いとして拾う。"""
        series = tmp_path / ti2v_export_dit.SERIES_NAME
        _tiny_text(monkeypatch, tmp_path)
        _point_at_tiny(monkeypatch, checkpoint, series)
        honest = dit_reference.ContainerDitWeights.tensor

        def nudged(self: Any, key: str) -> torch.Tensor:
            value = honest(self, key)
            if key == "blocks.1.ffn.net.2.weight":
                return torch.nextafter(value, torch.full_like(value, float("inf")))
            return value

        with monkeypatch.context() as fault:
            fault.setattr(dit_reference.ContainerDitWeights, "tensor", nudged)
            assert ti2v_export_dit.write_series()["passed"] is True
        _point_eager_full_at(monkeypatch, checkpoint, series)

        summary = ti2v_export_dit.eager_full()

        assert summary["passed"] is False
        assert set(summary["failures"]) == {case.name((1, 2, 2)) for case in ti2v_export_dit.CASES}
        for report in summary["cases"]:
            assert report["trunk_bit_exact"] is True, report["case"]
            assert report["golden_reference_bit_exact"] is False, report["case"]
            assert report["golden_output_bit_exact"] is False, report["case"]

    def test_it_stops_before_loading_when_the_golden_is_missing(
        self, checkpoint: Path, tmp_path: Path, monkeypatch
    ) -> None:
        _point_eager_full_at(monkeypatch, checkpoint, tmp_path / ti2v_export_dit.SERIES_NAME)
        monkeypatch.setattr(export_dit, "load_transformer", _never_load)

        with pytest.raises(ti2v_export_dit.Ti2vExportError, match="据えた golden が揃っていない"):
            ti2v_export_dit.eager_full()

    def test_it_stops_before_loading_when_memory_is_short(
        self, written, checkpoint: Path, monkeypatch
    ) -> None:
        target, _ = written
        _point_eager_full_at(monkeypatch, checkpoint, target.parent)
        monkeypatch.setattr(export_dit, "load_transformer", _never_load)
        monkeypatch.setattr(dit_probe, "mem_available", lambda: 2**30)

        with pytest.raises(dit_probe.DitProbeError, match=r"load の前の MemAvailable 1\.00 GiB"):
            ti2v_export_dit.eager_full()


def _never_load(_model: str) -> nn.Module:
    raise AssertionError("止まるべきところで上流を読んだ")


class TestTheQuantObservation:
    def test_the_cases_are_the_decision_cases_of_each_model(self) -> None:
        five_b = ti2v_export_dit._observe_cases("ti2v-5b", (1, 2, 2))
        one_three = ti2v_export_dit._observe_cases("t2v-1.3b", (1, 2, 2))

        assert [form for _, _, form, _ in five_b] == ["t2v"] * 3 + ["i2v"] * 3
        assert len(one_three) == 6
        assert {form for _, _, form, _ in one_three} == {"t2v"}
        assert all(name.startswith("band-") for name, *_ in five_b + one_three)
        # 文脈の作り方を 2 つのモデルで揃える（同じ順に同じプロンプト）。
        assert [text for *_, text in five_b] == [text for *_, text in one_three]

    def test_it_measures_both_models_and_compares_their_worst_case(
        self, checkpoint: Path, tmp_path: Path, monkeypatch
    ) -> None:
        _tiny_text(monkeypatch, tmp_path)
        monkeypatch.setattr(ti2v_export_dit, "transformer_dir", lambda _model=None: checkpoint)

        summary = ti2v_export_dit.observe_quant()

        for model in ti2v_export_dit.OBSERVE_MODELS:
            rows = summary["models"][model]["cases"]
            assert len(rows) == 6
            assert all(0 < row["ratio"] < 1 for row in rows)
        factor = summary["models"]["ti2v-5b"]["worst"] / summary["models"]["t2v-1.3b"]["worst"]
        assert summary["worst_factor_5b_over_1_3b"] == factor
        assert summary["escalate"] is (factor > ti2v_export_dit.ESCALATE_FACTOR)
