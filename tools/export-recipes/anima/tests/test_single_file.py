"""Rebuilding a single-file checkpoint into the diffusers layout (`anima.single_file`).

The real file (4GB) is not used — what we want to observe here is only **the key renaming, and
whether the comparison before writing really fails**, and both reproduce with a fake checkpoint
of a few tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from anima.single_file import (
    UNPREFIXED_MARKERS,
    _assert_covers,
    _block_count,
    _declared_shapes,
    _file_shapes,
    _link,
    _source_provenance,
    _strip_prefix,
    _transformer_config,
)


def _state(prefix: str) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}blocks.0.weight": torch.zeros(2, 3),
        f"{prefix}llm_adapter.blocks.0.weight": torch.zeros(4),
    }


class TestStripPrefix:
    """civitai wraps checkpoints in two ways (`net.` and `model.diffusion_model.`) — both are
    accepted."""

    @pytest.mark.parametrize("prefix", ["net.", "model.diffusion_model."])
    def test_it_normalizes_a_known_prefix_to_net(self, prefix: str) -> None:
        stripped = _strip_prefix(_state(prefix))

        assert sorted(stripped) == ["net.blocks.0.weight", "net.llm_adapter.blocks.0.weight"]

    def test_it_refuses_an_unknown_prefix_with_what_it_saw(self) -> None:
        """Silently passing an unknown wrapping through would write an empty result with no entry
        of the conversion table hitting."""
        with pytest.raises(SystemExit, match="unknown key prefix"):
            _strip_prefix(_state("diffusion_model."))

    def test_it_refuses_a_mixed_checkpoint(self) -> None:
        """For a checkpoint where only some prefixes differ, no basis for stripping can be
        chosen."""
        mixed = {**_state("net."), "model.diffusion_model.blocks.1.weight": torch.zeros(1)}

        with pytest.raises(SystemExit, match="unknown key prefix"):
            _strip_prefix(mixed)

    def test_it_takes_a_file_without_a_prefix_that_holds_the_markers(self) -> None:
        """A fine-tune saved without a wrapping (keys as the DiT names them) is normalized the
        same way."""
        state = {**dict.fromkeys(UNPREFIXED_MARKERS, torch.zeros(1)), **_state("")}

        stripped = _strip_prefix(state)

        assert sorted(stripped) == sorted(f"net.{name}" for name in state)

    def test_it_refuses_a_file_without_a_prefix_that_lacks_a_marker(self) -> None:
        """Without every marker, "no prefix" is not assumed — an unknown wrapping would pass as
        one otherwise."""
        state = {**dict.fromkeys(UNPREFIXED_MARKERS[:-1], torch.zeros(1)), **_state("")}

        with pytest.raises(SystemExit, match="unknown key prefix"):
            _strip_prefix(state)


class TestBlockCount:
    """The DiT's block count comes from the checkpoint's keys (a deeper fine-tune declares it
    nowhere else)."""

    @staticmethod
    def _blocks(*indices: int) -> dict[str, torch.Tensor]:
        state = {f"net.blocks.{index}.self_attn.q_proj.weight": torch.zeros(1) for index in indices}
        # The conditioner's blocks are not the DiT's.
        state["net.llm_adapter.blocks.7.cross_attn.q_proj.weight"] = torch.zeros(1)
        return state

    def test_it_counts_the_dit_blocks(self) -> None:
        assert _block_count(self._blocks(*range(40))) == 40

    def test_it_refuses_a_gap_in_the_indices(self) -> None:
        with pytest.raises(SystemExit, match=r"missing \[2\]"):
            _block_count(self._blocks(0, 1, 3))

    def test_it_refuses_a_checkpoint_without_blocks(self) -> None:
        with pytest.raises(SystemExit, match="no DiT block"):
            _block_count(self._blocks())


class TestTransformerConfig:
    """The transformer's config is base's with the block count of the checkpoint."""

    BASE = (
        b'{\n  "_class_name": "CosmosTransformer3DModel",\n  "num_layers": 28,\n'
        b'  "in_channels": 16\n}'
    )

    def test_base_s_count_keeps_base_s_bytes(self) -> None:
        assert _transformer_config(self.BASE, 28) == self.BASE

    def test_another_count_changes_that_field_only(self) -> None:
        config = json.loads(_transformer_config(self.BASE, 40))

        assert config == {**json.loads(self.BASE), "num_layers": 40}


class TestAssertCovers:
    """The check that keeps "weights only partly carried over" from being written the day the
    conversion table moves upstream."""

    @pytest.fixture
    def reference(self, tmp_path: Path) -> dict[str, tuple[int, ...]]:
        """Base's file, read the way the rebuild reads it (shapes from its header)."""
        path = tmp_path / "reference.safetensors"
        save_file({"a": torch.zeros(2, 3), "b": torch.zeros(4)}, path)
        return _file_shapes(path)

    def test_it_passes_when_the_keys_and_shapes_match(self, reference) -> None:
        _assert_covers({"a": torch.zeros(2, 3), "b": torch.zeros(4)}, reference, "t", "base")

    def test_it_stops_when_a_key_is_missing(self, reference) -> None:
        with pytest.raises(SystemExit, match="1 missing"):
            _assert_covers({"a": torch.zeros(2, 3)}, reference, "t", "base")

    def test_it_stops_when_an_unexpected_key_is_left_over(self, reference) -> None:
        """A leftover key is a sign that "the table hit a different model", so this fails too."""
        state = {"a": torch.zeros(2, 3), "b": torch.zeros(4), "c": torch.zeros(1)}

        with pytest.raises(SystemExit, match="1 extra"):
            _assert_covers(state, reference, "t", "base")

    def test_it_stops_when_a_shape_differs(self, reference) -> None:
        """Even with matching key names, a different shape means another architecture — it loads,
        but the picture breaks."""
        with pytest.raises(SystemExit, match="shape differs from base"):
            _assert_covers({"a": torch.zeros(3, 2), "b": torch.zeros(4)}, reference, "t", "base")


#: Base's transformer config (circlestone-labs/Anima-Base-v1.0-Diffusers `transformer/config.json`)
#: with a reduced block count — the declared state is built on the meta device, so the real width
#: costs nothing.
COSMOS_CONFIG = {
    "_class_name": "CosmosTransformer3DModel",
    "adaln_lora_dim": 256,
    "attention_head_dim": 128,
    "concat_padding_mask": True,
    "controlnet_block_every_n": None,
    "crossattn_proj_in_channels": 1024,
    "encoder_hidden_states_channels": 1024,
    "extra_pos_embed_type": None,
    "img_context_dim_in": None,
    "img_context_dim_out": 2048,
    "img_context_num_tokens": 256,
    "in_channels": 16,
    "max_size": [128, 240, 240],
    "mlp_ratio": 4.0,
    "num_attention_heads": 16,
    "num_layers": 2,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "rope_scale": [1.0, 4.0, 4.0],
    "text_embed_dim": 1024,
    "use_crossattn_projection": False,
}


class TestDeclaredTransformer:
    """The transformer is compared against what a transformer of the checkpoint's block count
    declares — a deeper fine-tune passes against its own count and fails against base's."""

    @pytest.fixture(autouse=True)
    def _diffusers(self) -> None:
        pytest.importorskip("diffusers")

    @staticmethod
    def _declared(num_layers: int) -> dict[str, tuple[int, ...]]:
        base = json.dumps(COSMOS_CONFIG).encode()
        return _declared_shapes(_transformer_config(base, num_layers))

    def test_one_more_block_declares_one_more_block_of_keys(self) -> None:
        two, three = self._declared(2), self._declared(3)
        added = sorted(set(three) - set(two))

        assert added and all(name.startswith("transformer_blocks.2.") for name in added)
        assert set(two) <= set(three)

    def test_a_deeper_state_passes_against_its_own_count(self) -> None:
        state = {
            name: torch.empty(shape, device="meta") for name, shape in self._declared(3).items()
        }

        _assert_covers(state, self._declared(3), "transformer", "a transformer of 3 blocks")

    def test_a_deeper_state_fails_against_base_s_count(self) -> None:
        state = {
            name: torch.empty(shape, device="meta") for name, shape in self._declared(3).items()
        }

        with pytest.raises(SystemExit, match="extra"):
            _assert_covers(state, self._declared(2), "transformer", "a transformer of 2 blocks")


class TestLink:
    """A rerun leaves no stale link target (the rebuild is idempotent)."""

    def test_it_replaces_an_existing_link(self, tmp_path: Path) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        link = tmp_path / "link"

        _link(first, link)
        _link(second, link)

        assert link.resolve() == second.resolve()

    def test_it_replaces_a_real_directory_left_behind(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"
        link.mkdir()

        _link(target, link)

        assert link.is_symlink()
        assert link.resolve() == target.resolve()


#: The intake record `anima.civitai` writes next to the checkpoint (the essentials only).
CIVITAI_RECORD = {
    "air": "urn:air:anima:checkpoint:civitai:2544636@2983680",
    "model_id": 2544636,
    "version_id": 2983680,
    "model_name": "WAI-ANIMA",
    "version_name": "v1.0(base 1.0)",
    "derived_name": "anima-wai-v1.0",
    "base_model": "Anima",
    "file": {"name": "waiANIMA_v10Base10.safetensors", "sha256": "9d5a1e13", "size_kb": 1.0},
    "permissions": {"allowNoCredit": True, "allowDerivatives": True},
    "descriptions": {"model": "<p>WAI-ANIMA</p>", "version": None},
    "fetched_at": "2026-09-01T00:00:00+00:00",
}


class TestSourceProvenance:
    """An intake record next to the checkpoint is carried along (a hand-placed checkpoint has none,
    so the shape without one stays as before)."""

    @pytest.fixture
    def checkpoint(self, tmp_path: Path) -> Path:
        return tmp_path / "waiANIMA_v10Base10.safetensors"

    def _write_record(self, checkpoint: Path, record: object) -> None:
        (checkpoint.parent / "civitai.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )

    def test_it_keeps_the_old_shape_without_a_record(self, checkpoint: Path) -> None:
        assert _source_provenance(checkpoint, "org/base") == {
            "file": "waiANIMA_v10Base10.safetensors",
            "base_repo": "org/base",
        }

    def test_it_carries_the_civitai_section(self, checkpoint: Path) -> None:
        """The HTML of the descriptions is not carried — it is not needed to trace the
        conversion's source."""
        self._write_record(checkpoint, CIVITAI_RECORD)

        provenance = _source_provenance(checkpoint, "org/base")

        assert provenance["civitai"] == {
            "air": "urn:air:anima:checkpoint:civitai:2544636@2983680",
            "model_id": 2544636,
            "version_id": 2983680,
            "model_name": "WAI-ANIMA",
            "version_name": "v1.0(base 1.0)",
            "derived_name": "anima-wai-v1.0",
            "sha256": "9d5a1e13",
            "permissions": {"allowNoCredit": True, "allowDerivatives": True},
        }

    def test_it_stops_on_a_record_it_cannot_read(self, checkpoint: Path) -> None:
        """Silently dropping the section would ship weights of unknown source dressed up with a
        provenance."""
        (checkpoint.parent / "civitai.json").write_text("{broken", encoding="utf-8")

        with pytest.raises(SystemExit, match="cannot read"):
            _source_provenance(checkpoint, "org/base")

    def test_it_stops_on_a_record_that_is_not_an_object(self, checkpoint: Path) -> None:
        self._write_record(checkpoint, ["urn:air:anima:checkpoint:civitai:2544636@2983680"])

        with pytest.raises(SystemExit, match="is not a JSON object"):
            _source_provenance(checkpoint, "org/base")

    def test_it_stops_when_a_carried_field_is_missing(self, checkpoint: Path) -> None:
        record = {key: value for key, value in CIVITAI_RECORD.items() if key != "air"}
        record["file"] = {"name": "waiANIMA_v10Base10.safetensors"}
        self._write_record(checkpoint, record)

        with pytest.raises(SystemExit, match=r"is missing fields: \['air', 'file.sha256'\]"):
            _source_provenance(checkpoint, "org/base")


class TestCivitaiIntakeLayout:
    """A missing record means different things by location — in the intake layout a broken
    intake, elsewhere a hand-placed checkpoint."""

    def test_it_stops_when_the_intake_directory_lost_its_record(self, tmp_path: Path) -> None:
        """`civitai-<versionId>/` is the shape the intake creates, so a missing record is the
        trace of a broken chain."""
        intake = tmp_path / "civitai-123"
        intake.mkdir()

        with pytest.raises(SystemExit, match="is in the intake layout but"):
            _source_provenance(intake / "waiANIMA_v10Base10.safetensors", "org/base")

    def test_it_keeps_the_old_shape_for_a_hand_placed_checkpoint(self, tmp_path: Path) -> None:
        """A source other than Civitai just has no `civitai.json` (ADR 0088) — it MUST NOT
        fail."""
        placed = tmp_path / "anima-hand-placed"
        placed.mkdir()

        provenance = _source_provenance(placed / "waiANIMA_v10Base10.safetensors", "org/base")

        assert provenance == {"file": "waiANIMA_v10Base10.safetensors", "base_repo": "org/base"}

    def test_it_reads_the_record_inside_the_intake_directory(self, tmp_path: Path) -> None:
        intake = tmp_path / "civitai-123"
        intake.mkdir()
        (intake / "civitai.json").write_text(
            json.dumps(CIVITAI_RECORD, ensure_ascii=False), encoding="utf-8"
        )

        provenance = _source_provenance(intake / "waiANIMA_v10Base10.safetensors", "org/base")

        civitai = provenance["civitai"]
        assert isinstance(civitai, dict)
        assert civitai["sha256"] == "9d5a1e13"
