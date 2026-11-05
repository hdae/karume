"""The text-tower writer (`qwen3_5.export`) on a tiny checkpoint: the meta trace is the real-weight
trace, the graph has the promised form, the row-block i8 container is byte-identical to the naive
form, and the checkpoint reader refuses what it does not know.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("transformers")

import torch

from _shared.container_read import read_stored
from karume.container import Provenance, container_parts
from karume.pipeline import publish_model
from karume.quantize import fake_quant_int8
from qwen3_5 import export, patch
from qwen3_5.checkpoint import Checkpoint, CheckpointError, TextTowerSource, assert_pinned_shards
from qwen3_5.tests.text_tower_fixture import PREFIX, stock_model, write_checkpoint

PROVENANCE = Provenance(license="apache-2.0", upstream_revision="0" * 40)


@pytest.fixture(scope="module")
def model():
    return stock_model()


@pytest.fixture(scope="module")
def checkpoint_dir(model, tmp_path_factory) -> Path:
    return write_checkpoint(model, tmp_path_factory.mktemp("checkpoint"))


@pytest.fixture(scope="module")
def checkpoint(checkpoint_dir) -> Checkpoint:
    return Checkpoint(checkpoint_dir, prefix=PREFIX, ignored_prefixes=("visual.",))


@pytest.fixture(scope="module")
def config(checkpoint_dir):
    return export.text_config(checkpoint_dir)


def real_tower(checkpoint: Checkpoint, config) -> patch.Qwen35TextTower:
    """The wrapped upstream with the checkpoint's values (F32), rewrites applied."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    with torch.device("meta"):
        model = Qwen3_5TextModel(config).float()
    names = [name for name, _ in model.named_parameters()]
    model.load_state_dict({name: checkpoint.read(name) for name in names}, assign=True)
    return patch.prepare_text_tower(model.eval())


def part_digests(path: Path) -> list[str]:
    return [hashlib.sha256(part.read_bytes()).hexdigest() for part in container_parts(path)]


@pytest.fixture(scope="module")
def prepared(checkpoint, config) -> export.TowerExport:
    # 7 rows per block: every linear and the embedding is split into several blocks.
    return export.prepare(checkpoint, config, "i8", chunk_rows=7)


class TestTheTrace:
    def test_the_meta_graph_is_the_real_weight_graph(self, prepared, checkpoint, config):
        graph, _ = export.trace(real_tower(checkpoint, config))

        assert prepared.graph.to_json() == graph.to_json()

    def test_the_graph_takes_the_four_inputs_with_one_symbol(self, prepared):
        inputs = [(entry.name, entry.dtype, entry.shape) for entry in prepared.graph.inputs]

        assert inputs == [
            ("input_ids", "i32", [1, "T"]),
            ("positions", "f32", ["T"]),
            ("rope_cos", "f32", [1, "T", 8]),
            ("rope_sin", "f32", [1, "T", 8]),
        ]

    def test_each_mixer_is_one_op(self, prepared):
        ops = [node.op for node in prepared.graph.nodes]

        assert ops.count("gated_delta_scan") == 3
        assert ops.count("attention") == 1
        assert ops.count("ge") == 1
        assert "sym_prefix_slice" not in ops

    def test_a_folded_mask_is_refused(self, checkpoint, config, monkeypatch):
        """Fault injection: a mask built from `arange` folds into a Tmax×Tmax constant read through
        `sym_prefix_slice` (the 64 MiB form this recipe avoids); the form check rejects it. (Real
        weights: constant folding evaluates the mask, which meta tensors cannot.)"""
        original = patch.causal_mask

        def folded(positions: torch.Tensor) -> torch.Tensor:
            return original(torch.arange(positions.shape[0]).float())

        monkeypatch.setattr(patch, "causal_mask", folded)

        with pytest.raises(export.Qwen35ExportError, match="sym_prefix_slice"):
            export.trace(real_tower(checkpoint, config))


class TestTheI8Series:
    def test_it_matches_the_naive_form_byte_for_byte(self, prepared, checkpoint, config, tmp_path):
        """Row blocks + `fixed_weights` against F32 + `fake_quant_int8` + `weight_dtype="i8"`."""
        tower = real_tower(checkpoint, config)
        modules = dict(tower.named_modules())
        report = fake_quant_int8(
            tower, include=lambda name: isinstance(modules[name], export.QUANT_TYPES)
        )
        graph, tensors = export.trace(tower)
        publish_model(
            tmp_path / "naive" / export.MODEL_FILE,
            graph,
            tensors,
            provenance=PROVENANCE,
            graph_name=export.GRAPH_NAME,
            weight_dtype="i8",
            weight_scales=report.scales,
        )
        export.write_container(prepared, tmp_path / "rows" / export.MODEL_FILE, PROVENANCE)

        expected = part_digests(tmp_path / "naive" / export.MODEL_FILE)
        assert part_digests(tmp_path / "rows" / export.MODEL_FILE) == expected

    def test_a_scale_shared_across_rows_breaks_the_match(self, prepared, checkpoint):
        """Fault injection: a block-wide scale instead of the row's gives other packed values."""
        key = "layers.0.mlp.down_proj.weight"
        fixed = prepared.fixed[key]
        weight = checkpoint.read(key)
        block_scale = fixed.scale.max().expand_as(fixed.scale)

        assert not torch.equal(
            torch.round(weight / block_scale).clamp(-127, 127).to(torch.int8), fixed.packed
        )

    def test_linear_and_embedding_are_i8_and_the_rest_f32(self, prepared, tmp_path):
        export.write_container(prepared, tmp_path / export.MODEL_FILE, PROVENANCE)
        stored = read_stored(tmp_path / export.MODEL_FILE)
        layouts = {key: value.layout for key, value in stored.items() if not value.constant}

        assert {key for key, layout in layouts.items() if layout == "i8"} == set(prepared.fixed)
        assert len(prepared.fixed) == 3 * 5 + 4 + 4 * 3 + 1
        assert {layouts[key] for key in layouts if key not in prepared.fixed} == {"f32"}

    def test_the_zero_centred_norms_hold_one_plus_w(self, prepared, checkpoint, tmp_path):
        export.write_container(prepared, tmp_path / export.MODEL_FILE, PROVENANCE)
        stored = read_stored(tmp_path / export.MODEL_FILE)

        def value(key: str) -> torch.Tensor:
            payload = bytearray(stored[key].payload)
            return torch.frombuffer(payload, dtype=torch.float32).reshape(stored[key].shape)

        for key in prepared.unit_offset_keys:
            assert torch.equal(value(key), 1.0 + checkpoint.read(key)), key
        gated = "layers.0.linear_attn.norm.weight"
        assert torch.equal(value(gated), checkpoint.read(gated))


class TestTheBf16Series:
    def test_it_stores_the_checkpoints_own_values(
        self, checkpoint, config, checkpoint_dir, tmp_path
    ):
        material = export.prepare(checkpoint, config, "bf16")
        export.write_container(material, tmp_path / export.MODEL_FILE, PROVENANCE)
        stored = read_stored(tmp_path / export.MODEL_FILE)
        from safetensors import safe_open

        key = "layers.3.self_attn.q_proj.weight"
        index = (checkpoint_dir / "model.safetensors.index.json").read_text()
        shard = next(
            name
            for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
            if f'"{PREFIX}{key}": "{name}"' in index
        )
        with safe_open(str(checkpoint_dir / shard), framework="pt") as handle:
            raw = handle.get_tensor(f"{PREFIX}{key}").view(torch.int16).numpy().tobytes()

        assert stored[key].layout == "bf16"
        assert stored[key].payload == raw
        assert stored["norm.weight"].layout == "f32"


class TestTheCheckpoint:
    def test_the_vision_tower_is_ignored(self, checkpoint):
        assert not [key for key in checkpoint.names() if key.startswith("visual")]

    def test_an_unknown_key_outside_the_text_tower_fails_loudly(self, model, tmp_path):
        directory = write_checkpoint(model, tmp_path, extra={"mtp.fc.weight": torch.zeros(2, 2)})

        with pytest.raises(CheckpointError, match="outside"):
            Checkpoint(directory, prefix=PREFIX, ignored_prefixes=("visual.",))

    def test_a_missing_graph_weight_fails_loudly(self, config):
        with pytest.raises(export.Qwen35ExportError, match="not in the checkpoint"):
            export.assert_checkpoint_keys(
                ["norm.weight", "absent.weight"], frozenset({"norm.weight"}), 4
            )

    def test_an_unconsumed_layer_key_fails_unless_truncated_away(self):
        available = frozenset({"norm.weight", "layers.0.x", "layers.2.x"})

        export.assert_checkpoint_keys(["norm.weight", "layers.0.x"], available, 2)
        with pytest.raises(export.Qwen35ExportError, match="does not consume"):
            export.assert_checkpoint_keys(["norm.weight", "layers.0.x"], available, 3)

    def test_a_shard_with_another_digest_fails_the_pin(self, checkpoint_dir):
        digests = {
            name: hashlib.sha256((checkpoint_dir / name).read_bytes()).hexdigest()
            for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        }
        source = TextTowerSource(
            name="tiny",
            repo="tiny/tiny",
            revision="0" * 40,
            subfolder="",
            license="apache-2.0",
            prefix=PREFIX,
            ignored_prefixes=("visual.",),
            shards=digests,
        )
        assert_pinned_shards(checkpoint_dir, source, hash_contents=True)

        wrong = replace(source, shards={**digests, next(iter(digests)): "0" * 64})
        with pytest.raises(CheckpointError, match="SHA-256"):
            assert_pinned_shards(checkpoint_dir, wrong, hash_contents=True)

    def test_a_blob_that_links_on_to_another_blob_passes_the_cheap_pin(
        self, checkpoint_dir, tmp_path
    ):
        # huggingface_hub 1.33 links blobs/<sha256> on to blobs/<shard>/<another 64-hex hash>:
        # the snapshot's own link still names the SHA-256, the end of the chain does not.
        blobs = tmp_path / "blobs"
        snapshot = tmp_path / "snapshots" / ("0" * 40)
        (blobs / "5b").mkdir(parents=True)
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors.index.json").write_bytes(
            (checkpoint_dir / "model.safetensors.index.json").read_bytes()
        )
        digests = {}
        for index, name in enumerate(
            ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        ):
            payload = (checkpoint_dir / name).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            stored = f"{index:x}" * 64
            (blobs / "5b" / stored).write_bytes(payload)
            (blobs / digest).symlink_to(Path("5b") / stored)
            (snapshot / name).symlink_to(Path("..") / ".." / "blobs" / digest)
            digests[name] = digest
        source = TextTowerSource(
            name="tiny",
            repo="tiny/tiny",
            revision="0" * 40,
            subfolder="",
            license="apache-2.0",
            prefix=PREFIX,
            ignored_prefixes=("visual.",),
            shards=digests,
        )

        assert_pinned_shards(snapshot, source)
        wrong = replace(source, shards={**digests, next(iter(digests)): "0" * 64})
        with pytest.raises(CheckpointError, match="resolves to blob"):
            assert_pinned_shards(snapshot, wrong)


class TestTheWriter:
    def test_a_truncated_export_needs_its_own_out(self):
        source = export.SOURCES[export.DEFAULT_SOURCE]

        with pytest.raises(export.Qwen35ExportError, match="--out"):
            export.write(source, "i8", layers=2)

    def test_the_series_is_named_by_model_source_and_storage(self):
        source = export.SOURCES[export.DEFAULT_SOURCE]

        assert export.series_dir(source, "i8").name == "qwen3.5-9b-text-moss-image-sft-i8-dyn"
