"""The EmbeddingGemma 2 distribution recipe (`embeddinggemma2.distribution`) — unit tests over one
assembly pass.

No real weights are used: each storage's series (bf16 and i8) holds a **valid minimal container**
of the graph's contract (`input_ids` i32 / `pool_mask` f32 `[1, T]`, `soft_embeds` f32
`[1, T, 4]` / `soft_mask` bool `[1, T, 1]` in, one f32 `[1, 768]` out, a token table, a projection
and a constant sliced by `T` at the bound), written through the core's own writer
({@link _write_container} — the i8 one rounded first, as the export does); the tokenizer asset
and the upstream snapshot are synthetic JSON, held to a synthetic pin. Every check of the plan is
exercised by shifting one input from that passing form. The card is covered by
`test_card.py`; the cases that read the real pinned snapshot skip explicitly without it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from container_series import i2_container, placed_paths, replace_component
from upstream_fixture import FIXTURE_REVISION, OTHER_REVISION

import dist
from _shared.licenses import APACHE_LICENSE_2_0_PATH
from embeddinggemma.export import MODEL_FILE as SINGLE_COMPONENT_FILE
from embeddinggemma2 import distribution as eg2
from embeddinggemma2 import export, tokenizer
from embeddinggemma2 import sources as upstream_sources
from embeddinggemma2.card import EMBEDDINGGEMMA2_UPSTREAM
from karume.container import CODEC_LEDGER, Provenance, sequence_siblings
from karume.dist import (
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    DistError,
    assemble_family,
    ir_graph,
    storage_dtypes,
    verify_dist,
)
from karume.emit import stored_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.publish import publish_container
from karume.quantize import channel_scale, quantize_to_int8

#: The synthetic vocabulary (16 rows) and its placeholder ids — numbers other than the real ones.
VOCAB = 16
PLACEHOLDERS = {"image": 13, "audio": 14, "video": 15}
BOS, EOS = 2, 1
#: The graph's output width (the full dimension — it must be the declared one).
WIDTH = eg2.EMBEDDINGGEMMA2_DIMENSIONS[0]

#: A prompt table over the eight real prefixes (`tokenizer.PROMPT_PREFIXES` — each has a parity
#: case), with names of its own and an order that is neither sorted nor the prefixes'.
PROMPTS = {
    "zeta": tokenizer.PROMPT_PREFIXES[3],
    "query": tokenizer.PROMPT_PREFIXES[0],
    "document": tokenizer.PROMPT_PREFIXES[1],
    **{f"p{index}": prefix for index, prefix in enumerate(tokenizer.PROMPT_PREFIXES[2:], start=2)},
}


def _write_container(
    path: Path,
    *,
    weight_dtype: str = "bf16",
    overrides: Mapping[str, str] | None = None,
    inputs: tuple[tuple[str, str, tuple[int | str, ...]], ...] = eg2.EMBEDDINGGEMMA2_GRAPH_INPUTS,
    width: int = WIDTH,
    bound: int = eg2.EMBEDDINGGEMMA2_MAX_TOKENS,
    vocab: int = VOCAB,
    lookups: int = 1,
    revision: str = FIXTURE_REVISION,
    license: str = eg2.EMBEDDINGGEMMA2_LICENSE,
) -> None:
    """Write the smallest container of the graph's contract as a series component (numbered
    parts at `path`'s place), through the core's writer. Under i8 the two weights are rounded to
    per-row int8 first and stored with their scales (the export's order)."""
    graph = IrGraph(symbols=[eg2.EMBEDDINGGEMMA2_SYMBOL])
    # The trailing axes of each input, the table's width (4) in place of the hidden placeholder.
    for name, dtype, trailing in inputs:
        shape = [1, eg2.EMBEDDINGGEMMA2_SYMBOL]
        shape += [4 if axis == eg2.EMBEDDINGGEMMA2_HIDDEN else axis for axis in trailing]
        graph.inputs.append(IrInput(name=name, dtype=dtype, shape=shape))
    # Values bf16 holds exactly (multiples of 1/8), so the bf16 write stores them unchanged.
    tensors = {
        "table": torch.arange(vocab * 4, dtype=torch.float32).reshape(vocab, 4) / 8,
        "projection": torch.arange(24, dtype=torch.float32).reshape(6, 4) / 8,
        "bias": torch.zeros(6, dtype=torch.float32),
        "const.band": torch.zeros(1, 1, bound, 1, dtype=torch.float32),
        "seed": torch.ones(1, 1, dtype=torch.float32),
    }
    for name, tensor in tensors.items():
        graph.initializers[name] = IrInitializer(tensor=name, storage=IrStorage("f32"))
        graph.values[name] = IrValue("f32", list(tensor.shape))
    graph.values["band.T"] = IrValue("f32", [1, 1, "T", 1])
    graph.values["projected"] = IrValue("f32", [1, "T", 6])
    graph.values["embedding"] = IrValue("f32", [1, width])
    for index in range(lookups):
        graph.values[f"tokens{index}"] = IrValue("f32", [1, "T", 4])
        graph.nodes.append(
            IrNode(
                op="embedding",
                ins=["table", "input_ids"],
                outs=[f"tokens{index}"],
                attrs={"padding_idx": -1},
            )
        )
    graph.nodes += [
        IrNode(op="linear", ins=["tokens0", "projection", "bias"], outs=["projected"], attrs={}),
        IrNode(
            op="sym_prefix_slice",
            ins=["const.band"],
            outs=["band.T"],
            attrs={"sym": "T", "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
        ),
        IrNode(op="expand", ins=["seed"], outs=["embedding"], attrs={}),
    ]
    graph.outputs.append("embedding")
    scales: dict[str, torch.Tensor] = {}
    for name in ("table", "projection"):
        if (overrides or {}).get(name, weight_dtype) == "i8":
            scales[name] = channel_scale(tensors[name], 0)
            tensors[name] = quantize_to_int8(tensors[name], scales[name]).to(torch.float32)
            tensors[name] *= scales[name]
    stored = stored_model(
        graph,
        tensors,
        weight_dtype=weight_dtype,
        weight_scales=scales,
        weight_dtype_overrides=overrides or {},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_container(
        path,
        stored.graph,
        stored.tensors,
        stored.bindings,
        graph_name=eg2.EMBEDDINGGEMMA2_ROLE,
        provenance=Provenance(license=license, upstream_revision=revision),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


UPSTREAM_CONFIG = {
    "image_token_id": PLACEHOLDERS["image"],
    "audio_token_id": PLACEHOLDERS["audio"],
    "video_token_id": PLACEHOLDERS["video"],
    "text_config": {"bos_token_id": BOS, "eos_token_id": EOS},
}
PROMPTS_CONFIG = {"default_prompt_name": None, "prompts": PROMPTS, "similarity_fn_name": "cosine"}


def _tokenizer_asset(tokenizer_sha256: str) -> dict[str, Any]:
    return {
        "format": eg2.EMBEDDINGGEMMA2_TOKENIZER_FORMAT,
        "source": {"sha256": tokenizer_sha256},
        "spec": {"postProcessor": eg2.EMBEDDINGGEMMA2_POST_PROCESSOR},
        "vocab": [f"t{index}" for index in range(VOCAB)],
        "specialIds": [0, EOS, BOS, *PLACEHOLDERS.values()],
        "bosId": BOS,
        "eosId": EOS,
    }


class Fixture:
    """The synthetic inputs of one assembly: the series root, the sources and the pin."""

    def __init__(self, root: Path) -> None:
        self.series_root = root / "outputs" / "series"
        self.sources = eg2.EmbeddingGemma2Sources(
            series={
                dtype: self.series_root / name for dtype, name in eg2.EMBEDDINGGEMMA2_SERIES.items()
            },
            tokenizer=self.series_root / tokenizer.TOKENIZER_SERIES,
            snapshot=root / "inputs" / "embeddinggemma2" / "embeddinggemma-2",
        )
        #: Dtype label → the series' container.
        self.containers = {
            dtype: series / eg2.EMBEDDINGGEMMA2_MODEL_FILE
            for dtype, series in self.sources.series.items()
        }
        #: The original weights' container (the one the contract is read from).
        self.container = self.containers[eg2.EMBEDDINGGEMMA2_DTYPE]
        for dtype, container in self.containers.items():
            _write_container(container, weight_dtype=dtype)
            # A series also holds the goldens, which must not be distributed.
            (container.parent / "io.query-en.safetensors").write_bytes(b"io-fixture")
        _write_json(self.sources.snapshot / "config.json", UPSTREAM_CONFIG)
        _write_json(self.sources.snapshot / "config_sentence_transformers.json", PROMPTS_CONFIG)
        (self.sources.snapshot / "tokenizer.json").write_text('{"fixture": true}', encoding="utf-8")
        self.write_tokenizer(_tokenizer_asset(self.snapshot_sha("tokenizer.json")))

    def snapshot_sha(self, name: str) -> str:
        return upstream_sources.file_sha256(self.sources.snapshot / name)

    def write_tokenizer(self, asset: Mapping[str, Any]) -> None:
        _write_json(self.sources.tokenizer / eg2.EMBEDDINGGEMMA2_TOKENIZER_FILE, asset)

    def rewrite_snapshot(self, name: str, value: Any) -> None:
        """Replace a snapshot file and move the pin with it (the passing form of the pin check)."""
        _write_json(self.sources.snapshot / name, value)

    @property
    def pin(self) -> upstream_sources.SourcePin:
        return upstream_sources.SourcePin(
            repo=upstream_sources.PIN.repo,
            revision=FIXTURE_REVISION,
            sha256={name: self.snapshot_sha(name) for name in eg2.EMBEDDINGGEMMA2_READ_FILES},
        )

    def rewrite_container(self, **container: Any) -> None:
        """Replace every storage's container with one written with other arguments (each in its
        own storage — the presets keep one graph)."""
        for dtype in self.containers:
            self.rewrite_storage(dtype, **container)

    def rewrite_storage(self, dtype: str, **container: Any) -> None:
        """Replace the container of the series of `dtype` alone (written as `dtype` unless the
        arguments say otherwise)."""
        path = self.containers[dtype]
        for stale in sequence_siblings(path):
            stale.unlink()
        _write_container(path, **{"weight_dtype": dtype, **container})

    def plan(self, model: str = eg2.EMBEDDINGGEMMA2_MODEL):
        return eg2.embeddinggemma2_plan(self.sources, model, self.pin)


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path)


@pytest.fixture
def assembled(fixture: Fixture, tmp_path: Path) -> tuple[Path, dict]:
    out_dir = tmp_path / "models" / eg2.EMBEDDINGGEMMA2_REPO_NAME
    manifest = assemble_family(
        [fixture.plan()],
        out_dir,
        eg2.EMBEDDINGGEMMA2_MODEL,
        root_files=eg2.PIPELINE.root_files,
    )
    return out_dir, manifest


def _model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][eg2.EMBEDDINGGEMMA2_MODEL]


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


class TestLayout:
    def test_it_places_the_graph_and_the_tokenizer_under_the_text_model(self, assembled) -> None:
        out_dir, _ = assembled
        expected = [
            f"{eg2.EMBEDDINGGEMMA2_MODEL}/{path}"
            for path in placed_paths(eg2.EMBEDDINGGEMMA2_OUTPUT_PATHS, eg2.EMBEDDINGGEMMA2_WEIGHTS)
        ]

        assert _present(out_dir) == sorted(
            [*expected, MANIFEST_FILENAME, "LICENSE.md", "NOTICE.md"]
        )
        assert verify_dist(out_dir)

    def test_it_never_carries_the_goldens(self, assembled) -> None:
        out_dir, _ = assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_the_manifest_declares_the_contract_the_roles_and_the_two_presets(
        self, assembled
    ) -> None:
        _, manifest = assembled
        model = _model(manifest)

        assert manifest["defaultModel"] == "text"
        assert list(manifest["models"]) == ["text"]
        assert model["pipeline"] == "embeddinggemma/1"
        assert list(model["weights"]) == ["model"]
        assert list(model["weights"]["model"]) == ["bf16", "i8"]
        assert list(model["assets"]) == ["tokenizer"]
        assert list(model["quants"]) == ["bf16", "i8"]
        assert model["defaultQuant"] == "bf16"
        assert model["quants"]["bf16"]["weights"] == {"model": "bf16"}
        assert model["quants"]["bf16"]["session"] == {}
        assert model["quants"]["bf16"]["label"] == "Original weights (bf16)"
        assert model["quants"]["i8"]["weights"] == {"model": "i8"}
        assert model["quants"]["i8"]["session"] == {}
        assert model["quants"]["i8"]["label"] == "Half size (int8)"

    def test_every_preset_carries_a_label_and_a_description_within_the_ui_limits(
        self, assembled
    ) -> None:
        """Two presets, so both need both fields (ADR 0128 C3), within 64 / 200 characters."""
        _, manifest = assembled
        for quant in _model(manifest)["quants"].values():
            assert 0 < len(quant["label"].strip()) <= 64
            assert 0 < len(quant["description"].strip()) <= 200

    def test_each_storage_ships_its_own_series_container(self, assembled, fixture) -> None:
        out_dir, manifest = assembled
        containers = _model(manifest)["weights"]["model"]

        for dtype, series in fixture.containers.items():
            parts = containers[dtype]["container"]["parts"]
            assert parts[0]["path"].startswith(f"text/model/model.{dtype}-")
            sources = sorted(sequence_siblings(series))
            assert [(out_dir / part["path"]).read_bytes() for part in parts] == [
                source.read_bytes() for source in sources
            ]

    def test_the_repository_is_karume_embeddinggemma_2(self) -> None:
        assert eg2.PIPELINE.repo_name("text") == "karume-embeddinggemma-2"
        assert dist.default_out_dir(eg2.PIPELINE, ["text"]).name == "karume-embeddinggemma-2"

    def test_another_model_is_refused(self, fixture: Fixture) -> None:
        with pytest.raises(DistError, match="unknown EmbeddingGemma 2 model 'image'"):
            fixture.plan("image")


class TestStorage:
    def test_the_f32_series_is_refused(self, fixture: Fixture) -> None:
        """The f32 series has the same graph and goldens — only its storage tells it apart."""
        fixture.rewrite_storage("bf16", weight_dtype="f32")

        with pytest.raises(DistError, match=r"model_bf16: storage of .* has no bf16"):
            fixture.plan()

    def test_the_f32_series_in_the_int8_place_is_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_storage("i8", weight_dtype="f32")

        with pytest.raises(DistError, match=r"model_i8: storage of .* has no i8"):
            fixture.plan()

    def test_a_weight_in_another_compressed_storage_is_refused(self, fixture: Fixture) -> None:
        """bf16 is there (the requirement passes) — the forbidden table catches the f16 weight."""
        fixture.rewrite_storage("bf16", overrides={"projection": "f16"})

        with pytest.raises(DistError, match=r"model_bf16: .* has .*f16 \(actual"):
            fixture.plan()

    def test_an_int8_weight_in_the_original_weights_is_refused(self, fixture: Fixture) -> None:
        """bf16 is there — the forbidden table keeps an i8 series' weight out of the bf16 preset."""
        fixture.rewrite_storage("bf16", overrides={"projection": "i8"})

        with pytest.raises(DistError, match=r"model_bf16: .* has i8 \(actual"):
            fixture.plan()

    def test_a_bf16_weight_in_the_int8_series_is_refused(self, fixture: Fixture) -> None:
        """i8 is there — the forbidden table keeps the original weights out of the i8 preset."""
        fixture.rewrite_storage("i8", overrides={"table": "bf16"})

        with pytest.raises(DistError, match=r"model_i8: .* has bf16 \(actual"):
            fixture.plan()

    def test_the_two_series_swapped_are_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_storage("bf16", weight_dtype="i8")
        fixture.rewrite_storage("i8", weight_dtype="bf16")

        with pytest.raises(DistError, match=r"model_bf16: storage of .* has no bf16"):
            fixture.plan()

    def test_an_i2_series_is_refused(self, fixture: Fixture) -> None:
        replace_component(fixture.container, i2_container(named=eg2.EMBEDDINGGEMMA2_ROLE))

        with pytest.raises(DistError, match="model_bf16: "):
            fixture.plan()

    def test_an_i2_series_in_the_int8_place_is_refused(self, fixture: Fixture) -> None:
        replace_component(fixture.containers["i8"], i2_container(named=eg2.EMBEDDINGGEMMA2_ROLE))

        with pytest.raises(DistError, match="model_i8: "):
            fixture.plan()

    @pytest.mark.parametrize("dtype", sorted(eg2.EMBEDDINGGEMMA2_SERIES))
    def test_the_forbidden_table_names_every_other_compressed_layout(self, dtype: str) -> None:
        compressed = {entry.layout for entry in CODEC_LEDGER.values()} - {"f32", "i32", dtype}

        assert (
            set(eg2.EMBEDDINGGEMMA2_STORAGE_FORBIDDEN[eg2.embeddinggemma2_role(dtype)])
            == compressed
        )

    def test_every_storage_requires_itself(self) -> None:
        assert eg2.EMBEDDINGGEMMA2_STORAGE_REQUIREMENTS == {"model_bf16": "bf16", "model_i8": "i8"}

    def test_a_missing_storage_is_refused(self, fixture: Fixture) -> None:
        sources = eg2.EmbeddingGemma2Sources(
            series={"bf16": fixture.sources.series["bf16"]},
            tokenizer=fixture.sources.tokenizer,
            snapshot=fixture.sources.snapshot,
        )

        with pytest.raises(DistError, match=r"given for \['bf16'\], not for the storages"):
            eg2.embeddinggemma2_plan(sources, pin=fixture.pin)


class TestOneGraph:
    """Every preset runs one graph — the contract is read from the original weights' container."""

    def test_an_int8_container_of_another_bound_is_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_storage("i8", bound=1024)

        with pytest.raises(DistError, match="its graph is not the graph of"):
            fixture.plan()

    def test_an_int8_container_with_a_mask_input_is_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_storage(
            "i8", inputs=(*eg2.EMBEDDINGGEMMA2_GRAPH_INPUTS, ("attention_mask", "f32", ()))
        )

        with pytest.raises(DistError, match="its graph is not the graph of"):
            fixture.plan()

    def test_the_storage_alone_differs(self, fixture: Fixture) -> None:
        graphs = {dtype: ir_graph(path) for dtype, path in fixture.containers.items()}

        assert graphs["bf16"] == graphs["i8"]
        assert storage_dtypes(fixture.containers["bf16"]) != storage_dtypes(
            fixture.containers["i8"]
        )


class TestPipelineConfig:
    """`pipelineConfig` — the field names part of the TypeScript schema, every value from its
    source."""

    def test_it_declares_exactly_the_four_fields_in_order(self, assembled) -> None:
        _, manifest = assembled
        assert list(_model(manifest)["pipelineConfig"]) == [
            "prompts",
            "dimensions",
            "maxTokens",
            "placeholderTokens",
        ]

    def test_the_prompts_are_upstreams_verbatim_and_in_order(self, assembled) -> None:
        _, manifest = assembled
        prompts = _model(manifest)["pipelineConfig"]["prompts"]

        assert prompts == PROMPTS
        assert list(prompts) == list(PROMPTS)

    def test_the_dimensions_start_at_the_graphs_width(self, assembled) -> None:
        _, manifest = assembled
        assert _model(manifest)["pipelineConfig"]["dimensions"] == [768, 512, 256, 128]

    def test_the_token_bound_is_the_graphs(self, assembled) -> None:
        _, manifest = assembled
        assert _model(manifest)["pipelineConfig"]["maxTokens"] == 8192

    def test_the_placeholders_come_from_the_upstream_config(self, assembled) -> None:
        """The synthetic ids differ from the real ones — hard-coded ids would show."""
        _, manifest = assembled
        assert _model(manifest)["pipelineConfig"]["placeholderTokens"] == PLACEHOLDERS

    def test_a_graph_of_another_width_is_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_container(width=512)

        with pytest.raises(DistError, match="full dimension 768 is not the graph's output width"):
            fixture.plan()

    def test_a_graph_traced_with_another_bound_is_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_container(bound=1024)

        with pytest.raises(DistError, match="binds at most 1024 tokens"):
            fixture.plan()

    def test_a_default_prompt_is_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_snapshot(
            "config_sentence_transformers.json", {**PROMPTS_CONFIG, "default_prompt_name": "query"}
        )
        with pytest.raises(DistError, match="default_prompt_name is 'query'"):
            fixture.plan()

    def test_another_similarity_function_is_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_snapshot(
            "config_sentence_transformers.json", {**PROMPTS_CONFIG, "similarity_fn_name": "dot"}
        )
        with pytest.raises(DistError, match="similarity_fn_name is 'dot'"):
            fixture.plan()

    def test_a_prefix_without_a_tokenizer_case_is_refused(self, fixture: Fixture) -> None:
        prompts = {**PROMPTS, "new": "task: something new | query: "}
        fixture.rewrite_snapshot(
            "config_sentence_transformers.json", {**PROMPTS_CONFIG, "prompts": prompts}
        )
        with pytest.raises(DistError, match="without a case"):
            fixture.plan()

    def test_no_prompts_are_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_snapshot(
            "config_sentence_transformers.json", {**PROMPTS_CONFIG, "prompts": {}}
        )
        with pytest.raises(DistError, match="prompts is not a non-empty object"):
            fixture.plan()

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("image_token_id", VOCAB, "not an id of the 16 rows"),
            ("audio_token_id", True, "not an id of the 16 rows"),
            ("video_token_id", 3, "not a special token"),
        ],
    )
    def test_a_placeholder_id_the_tokenizer_cannot_produce_is_refused(
        self, fixture: Fixture, field: str, value: Any, message: str
    ) -> None:
        fixture.rewrite_snapshot("config.json", {**UPSTREAM_CONFIG, field: value})
        with pytest.raises(DistError, match=message):
            fixture.plan()

    def test_repeated_placeholder_ids_are_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_snapshot(
            "config.json", {**UPSTREAM_CONFIG, "video_token_id": PLACEHOLDERS["image"]}
        )
        with pytest.raises(DistError, match="placeholder ids repeat"):
            fixture.plan()


class TestGraphContract:
    def test_a_mask_left_as_an_input_is_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_container(
            inputs=(*eg2.EMBEDDINGGEMMA2_GRAPH_INPUTS, ("attention_mask", "f32", ())),
        )
        with pytest.raises(DistError, match="the graph inputs are"):
            fixture.plan()

    def test_a_pool_mask_of_another_dtype_is_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_container(
            inputs=(
                ("input_ids", "i32", ()),
                ("pool_mask", "i32", ()),
                *eg2.EMBEDDINGGEMMA2_GRAPH_INPUTS[2:],
            )
        )

        with pytest.raises(DistError, match="the graph inputs are"):
            fixture.plan()

    def test_a_graph_without_the_soft_inputs_is_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_container(inputs=eg2.EMBEDDINGGEMMA2_GRAPH_INPUTS[:2])

        with pytest.raises(DistError, match="the graph inputs are"):
            fixture.plan()

    def test_soft_rows_of_another_width_than_the_token_table_are_refused(
        self, fixture: Fixture
    ) -> None:
        fixture.rewrite_container(
            inputs=(
                *eg2.EMBEDDINGGEMMA2_GRAPH_INPUTS[:2],
                ("soft_embeds", "f32", (8,)),
                eg2.EMBEDDINGGEMMA2_GRAPH_INPUTS[3],
            )
        )

        with pytest.raises(DistError, match="the graph inputs are"):
            fixture.plan()

    def test_a_soft_mask_of_the_full_width_is_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_container(
            inputs=(
                *eg2.EMBEDDINGGEMMA2_GRAPH_INPUTS[:3],
                ("soft_mask", "bool", (eg2.EMBEDDINGGEMMA2_HIDDEN,)),
            )
        )

        with pytest.raises(DistError, match="the graph inputs are"):
            fixture.plan()

    def test_a_second_embedding_lookup_is_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_container(lookups=2)

        with pytest.raises(DistError, match="2 embedding lookups"):
            fixture.plan()

    def test_the_bound_is_read_from_every_slice_of_t(self) -> None:
        graph = {
            "values": {
                "a": {"shape": [1, 1, 64, 64]},
                "b": {"shape": [1, 1, 64, 8]},
            },
            "nodes": [
                {
                    "op": "sym_prefix_slice",
                    "ins": [name],
                    "attrs": {"sym": "T", "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
                }
                for name in ("a", "b")
            ],
        }
        assert eg2.embeddinggemma2_graph_max_tokens(graph, Path("g")) == 64

    def test_disagreeing_extents_are_refused(self) -> None:
        graph = {
            "values": {"a": {"shape": [1, 1, 64, 64]}, "b": {"shape": [1, 1, 32, 8]}},
            "nodes": [
                {
                    "op": "sym_prefix_slice",
                    "ins": [name],
                    "attrs": {"sym": "T", "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
                }
                for name in ("a", "b")
            ],
        }
        with pytest.raises(DistError, match=r"extents \[32, 64\]"):
            eg2.embeddinggemma2_graph_max_tokens(graph, Path("g"))

    def test_a_graph_without_a_slice_of_t_is_refused(self) -> None:
        with pytest.raises(DistError, match=r"extents \[\]"):
            eg2.embeddinggemma2_graph_max_tokens({"values": {}, "nodes": []}, Path("g"))

    def test_a_slice_with_an_offset_is_refused(self) -> None:
        graph = {
            "values": {"a": {"shape": [1, 1, 64, 64]}},
            "nodes": [
                {
                    "op": "sym_prefix_slice",
                    "ins": ["a"],
                    "attrs": {"sym": "T", "slices": [{"dim": 2, "coeff": 1, "offset": 1}]},
                }
            ],
        }
        with pytest.raises(DistError, match="cannot be read"):
            eg2.embeddinggemma2_graph_max_tokens(graph, Path("g"))


class TestTokenizerAsset:
    def test_an_asset_compiled_from_other_bytes_is_refused(self, fixture: Fixture) -> None:
        fixture.write_tokenizer(_tokenizer_asset("0" * 64))
        with pytest.raises(DistError, match=r"compiled from a tokenizer\.json with SHA-256 0000"):
            fixture.plan()

    def test_an_asset_of_another_vocabulary_is_refused(self, fixture: Fixture) -> None:
        asset = _tokenizer_asset(fixture.snapshot_sha("tokenizer.json"))
        fixture.write_tokenizer({**asset, "vocab": asset["vocab"][:-1]})
        with pytest.raises(DistError, match="15 vocabulary rows, not the graph's 16"):
            fixture.plan()

    def test_an_asset_without_eos_is_refused(self, fixture: Fixture) -> None:
        asset = _tokenizer_asset(fixture.snapshot_sha("tokenizer.json"))
        fixture.write_tokenizer({**asset, "spec": {"postProcessor": "bos"}})
        with pytest.raises(DistError, match="post-processor is 'bos'"):
            fixture.plan()

    def test_an_asset_with_other_bos_eos_ids_is_refused(self, fixture: Fixture) -> None:
        asset = _tokenizer_asset(fixture.snapshot_sha("tokenizer.json"))
        fixture.write_tokenizer({**asset, "eosId": 106})
        with pytest.raises(DistError, match="eosId is 106"):
            fixture.plan()

    def test_an_asset_of_another_format_is_refused(self, fixture: Fixture) -> None:
        asset = _tokenizer_asset(fixture.snapshot_sha("tokenizer.json"))
        fixture.write_tokenizer({**asset, "format": "karume-umt5-tokenizer/1"})
        with pytest.raises(DistError, match="not a compiled tokenizer asset"):
            fixture.plan()


class TestProvenance:
    def test_a_snapshot_file_off_the_pin_is_refused(self, fixture: Fixture) -> None:
        pin = fixture.pin
        fixture.rewrite_snapshot("config.json", {**UPSTREAM_CONFIG, "extra": 1})

        with pytest.raises(DistError, match=r"config\.json: SHA-256"):
            eg2.embeddinggemma2_plan(fixture.sources, pin=pin)

    def test_a_container_from_another_revision_is_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_container(revision=OTHER_REVISION)

        with pytest.raises(DistError, match="built from a different revision"):
            fixture.plan()

    def test_a_container_under_another_license_is_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_container(license="gemma")

        with pytest.raises(DistError, match=r"provenance\.license is 'gemma'"):
            fixture.plan()

    def test_an_int8_container_from_another_revision_is_refused(self, fixture: Fixture) -> None:
        """The provenance is not part of the graph — each container is held to the pin itself."""
        fixture.rewrite_storage("i8", revision=OTHER_REVISION)

        with pytest.raises(DistError, match="built from a different revision"):
            fixture.plan()

    def test_an_int8_container_under_another_license_is_refused(self, fixture: Fixture) -> None:
        fixture.rewrite_storage("i8", license="gemma")

        with pytest.raises(DistError, match=r"provenance\.license is 'gemma'"):
            fixture.plan()


class TestLegalText:
    def test_it_ships_the_license_text_byte_identical(self, assembled) -> None:
        out_dir, _ = assembled
        assert (out_dir / "LICENSE.md").read_bytes() == APACHE_LICENSE_2_0_PATH.read_bytes()

    def test_the_notice_names_the_upstream_and_states_the_changes(self, assembled) -> None:
        out_dir, manifest = assembled
        notice = (out_dir / "NOTICE.md").read_text(encoding="utf-8")

        for name in manifest["models"]:
            assert f"`{EMBEDDINGGEMMA2_UPSTREAM[name]}`" in notice
        assert "**text tower only**" in notice
        assert "**The `bf16` form keeps the checkpoint's BF16 values unchanged**" in notice
        assert "replaced by a band declared on the attention" in notice
        # The i8 form is the one quantized copy, and the notice says what was rounded.
        assert "**The\n  `i8` form is an int8-quantized copy**" in notice
        assert "rounded to symmetric int8 with one float32 scale per row" in notice
        assert notice.count("quantiz") == 1

    def test_the_notice_names_exactly_the_storages_the_manifest_ships(self, assembled) -> None:
        """Both directions: a shipped form the notice leaves out, or a form it still describes
        after it was dropped, fails."""
        out_dir, manifest = assembled
        notice = (out_dir / "NOTICE.md").read_text(encoding="utf-8")

        named = set(re.findall(r"`([a-z0-9]+)` form", notice))
        assert named == set(_model(manifest)["weights"]["model"])


class TestTheDriver:
    def test_the_driver_builds_the_mirror_from_the_series_root(
        self, fixture: Fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`dist.py --pipeline embeddinggemma-2` reads the snapshot and the pin at call time."""
        monkeypatch.setattr(upstream_sources, "SNAPSHOT_DIR", fixture.sources.snapshot)
        monkeypatch.setattr(upstream_sources, "PIN", fixture.pin)
        out_dir = tmp_path / "mirror"

        dist.main(
            [
                "--pipeline",
                "embeddinggemma-2",
                "--series",
                str(fixture.series_root),
                "--out",
                str(out_dir),
            ]
        )

        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert _model(manifest)["pipeline"] == "embeddinggemma/1"
        assert (out_dir / MODEL_CARD_FILENAME).is_file()


class TestMirroredConstants:
    """The writer's constants this torch-free module mirrors stay equal to the writer's."""

    def test_the_token_bound_is_the_exports(self) -> None:
        assert export.SYM_MAX == eg2.EMBEDDINGGEMMA2_MAX_TOKENS

    def test_the_series_are_the_exports_bf16_and_i8_series(self) -> None:
        assert list(eg2.EMBEDDINGGEMMA2_SERIES) == ["bf16", "i8"]
        for dtype, series in eg2.EMBEDDINGGEMMA2_SERIES.items():
            assert export.DEFAULT_OUT_DIRS[dtype].name == series

    def test_the_role_is_the_exports_graph_name(self) -> None:
        assert eg2.EMBEDDINGGEMMA2_ROLE == export.GRAPH_NAME

    def test_the_inputs_are_the_exports(self) -> None:
        assert tuple(name for name, *_ in eg2.EMBEDDINGGEMMA2_GRAPH_INPUTS) == export.GRAPH_INPUTS

    def test_the_file_names_are_the_writers(self) -> None:
        assert eg2.EMBEDDINGGEMMA2_MODEL_FILE == SINGLE_COMPONENT_FILE
        assert tokenizer.ASSET_PATH.name == eg2.EMBEDDINGGEMMA2_TOKENIZER_FILE

    def test_the_placeholder_fields_are_the_ones_the_export_refuses(self) -> None:
        """The export keeps these ids out of the goldens (`export.build_cases`); the reader is
        told the same three."""
        source = Path(export.__file__).read_text(encoding="utf-8")
        for _, field in eg2.EMBEDDINGGEMMA2_PLACEHOLDER_FIELDS:
            assert f"composite.{field}" in source


def _real_snapshot() -> Path:
    directory = upstream_sources.SNAPSHOT_DIR
    if not (directory / "config_sentence_transformers.json").is_file():
        pytest.skip(f"the pinned snapshot is not placed at {directory}")
    upstream_sources.assert_pinned(directory, ("config.json", "config_sentence_transformers.json"))
    return directory


class TestThePinnedSnapshot:
    """The values read from the real pinned snapshot (skipped without it)."""

    def test_it_declares_twenty_prompts_over_the_eight_prefixes(self) -> None:
        directory = _real_snapshot()
        path = directory / "config_sentence_transformers.json"
        prompts = eg2.embeddinggemma2_prompts(
            json.loads(path.read_text(encoding="utf-8")), str(path)
        )

        assert len(prompts) == 20
        assert prompts["query"] == "task: search result | query: "
        assert prompts["document"] == "title: none | text: "
        assert set(prompts.values()) == set(tokenizer.PROMPT_PREFIXES)

    def test_its_placeholder_ids_are_special_tokens_of_the_compiled_asset(self) -> None:
        directory = _real_snapshot()
        if not tokenizer.ASSET_PATH.is_file():
            pytest.skip(f"the compiled tokenizer asset is not at {tokenizer.ASSET_PATH}")
        asset = json.loads(tokenizer.ASSET_PATH.read_text(encoding="utf-8"))
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))

        placeholders = eg2.embeddinggemma2_placeholder_tokens(
            config, asset, len(asset["vocab"]), "config.json"
        )

        assert placeholders == {"image": 258880, "audio": 258881, "video": 258884}
