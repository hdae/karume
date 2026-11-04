"""Script-level contracts of `irodori/dacvae/convert.py` (the part that needs no real weights).

Converting the real weights is manual (a one-off utility), but **the shape checks and the
move-into-place discipline** run on a synthetic checkpoint alone (a `.pth` made with
`torch.save`). What is pinned here is only what would silently turn the input material into
something else if it broke:

- a wrong outer wrapper shape (key set, `state_dict` is a dict, every value is a tensor) fails
- a type whose metadata JSON round trip is not equal (tuple / non-string key) fails
- the byte-identity check that reads the written safetensors back with a separate reader
  implementation catches a dtype / shape / single-element mismatch each
- `convert()` writes `metadata.json` and `__metadata__` **from the same source**
- a run that fails a check does not move the canonical path by a single byte (ADR 0052's move into
  place)
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from irodori.dacvae import convert as cv

#: Contents of the synthetic checkpoint (two keys spelled like the real ones).
TENSORS = {
    "encoder.block.0.weight": torch.tensor([[1.0, -2.0], [0.5, 0.25]]),
    "quantizer.in_proj.bias": torch.tensor([0.125, -0.5, 2.0, 4.0]),
}

#: metadata shaped like the real one (plain configuration values without tensors).
METADATA = {"kwargs": {"latent_dim": 32, "encoder_rates": [2, 4, 5, 8], "name": "dacvae"}}


def _write_ckpt(tmp_path, *, state_dict=None, metadata=None, wrapper=None):
    """Write a synthetic `.pth` and return its path (passing `wrapper` replaces the outer shape
    as a whole)."""
    path = tmp_path / "weights.pth"
    payload = (
        wrapper
        if wrapper is not None
        else {
            "state_dict": TENSORS if state_dict is None else state_dict,
            "metadata": METADATA if metadata is None else metadata,
        }
    )
    torch.save(payload, path)
    return path


class TestLoadCheckpoint:
    """MUST: an addition or removal in the outer shape is a change on the publisher's side, so it
    fails loudly (no silently reading other keys)."""

    def test_the_expected_shape_returns_the_state_dict_and_metadata(self, tmp_path):
        state_dict, metadata = cv._load_checkpoint(_write_ckpt(tmp_path))

        assert sorted(state_dict) == sorted(TENSORS)
        assert metadata == METADATA

    def test_a_top_level_that_is_not_a_dict_fails_loudly(self, tmp_path):
        with pytest.raises(cv.ConvertError, match="the top level is not a dict"):
            cv._load_checkpoint(_write_ckpt(tmp_path, wrapper=[1, 2, 3]))

    def test_a_wrapper_with_other_keys_fails_loudly(self, tmp_path):
        wrapper = {"state_dict": TENSORS, "metadata": METADATA, "optimizer": {}}

        with pytest.raises(cv.ConvertError, match="outer wrapper"):
            cv._load_checkpoint(_write_ckpt(tmp_path, wrapper=wrapper))

    def test_a_state_dict_that_is_not_a_dict_fails_loudly(self, tmp_path):
        with pytest.raises(cv.ConvertError, match="state_dict is not a dict"):
            cv._load_checkpoint(_write_ckpt(tmp_path, state_dict=[1, 2]))

    def test_a_state_dict_holding_a_non_tensor_fails_loudly(self, tmp_path):
        state_dict = {**TENSORS, "encoder.steps": 4}

        with pytest.raises(cv.ConvertError, match="non-tensor"):
            cv._load_checkpoint(_write_ckpt(tmp_path, state_dict=state_dict))


class TestMetadataJson:
    def test_a_plain_config_round_trips(self):
        text = cv._metadata_json(METADATA)

        assert json.loads(text) == METADATA

    def test_a_tuple_does_not_round_trip_and_fails_loudly(self):
        """JSON turns a tuple into a list — writing a value whose round trip is not equal would
        misstate the provenance."""
        with pytest.raises(cv.ConvertError, match="round trip"):
            cv._metadata_json({"a": (1, 2)})

    def test_a_non_string_key_does_not_round_trip_and_fails_loudly(self):
        with pytest.raises(cv.ConvertError, match="round trip"):
            cv._metadata_json({1: "one"})

    def test_a_value_json_cannot_hold_fails_loudly(self):
        with pytest.raises(cv.ConvertError, match="cannot be serialized to JSON"):
            cv._metadata_json({"a": object()})


class TestByteIdentical:
    """MUST: read back with **a reader implementation separate** from the writer (do not trust the
    writer's own report)."""

    def test_an_untouched_file_matches_every_tensor(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        save_file(TENSORS, str(path))

        assert cv._assert_byte_identical(path, TENSORS) == len(TENSORS)

    def test_a_missing_key_fails_loudly(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        save_file({"quantizer.in_proj.bias": TENSORS["quantizer.in_proj.bias"]}, str(path))

        with pytest.raises(cv.ConvertError, match="key sets differ"):
            cv._assert_byte_identical(path, TENSORS)

    def test_a_different_dtype_fails_loudly(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        written = {**TENSORS, "quantizer.in_proj.bias": TENSORS["quantizer.in_proj.bias"].half()}
        save_file(written, str(path))

        with pytest.raises(cv.ConvertError, match="dtype mismatch"):
            cv._assert_byte_identical(path, TENSORS)

    def test_a_different_shape_fails_loudly(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        written = {**TENSORS, "quantizer.in_proj.bias": TENSORS["quantizer.in_proj.bias"][:2]}
        save_file(written, str(path))

        with pytest.raises(cv.ConvertError, match="shape mismatch"):
            cv._assert_byte_identical(path, TENSORS)

    def test_a_single_changed_element_fails_loudly(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        moved = TENSORS["quantizer.in_proj.bias"].clone()
        moved[0] = -0.0  # Equal to 0.0 as a value — a mismatch only the bits catch.
        save_file({**TENSORS, "quantizer.in_proj.bias": moved}, str(path))

        with pytest.raises(cv.ConvertError, match="bytes differ"):
            cv._assert_byte_identical(path, TENSORS)


class TestConvert:
    def test_it_writes_both_records_from_the_same_source(self, tmp_path):
        """MUST: `metadata.json` and `__metadata__` are written from the same source in one pass."""
        ckpt = _write_ckpt(tmp_path)

        summary = cv.convert(ckpt)

        target = tmp_path / "weights.safetensors"
        assert summary["tensors"] == len(TENSORS)
        assert summary["byte_identical"] == len(TENSORS)
        assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8")) == METADATA
        with safe_open(str(target), framework="pt") as handle:
            header = handle.metadata()
        source = json.loads(header[cv.SOURCE_KEY])
        assert source[cv.SOURCE_FILE_KEY] == ckpt.name
        assert source[cv.SOURCE_SHA256_KEY] == cv._sha256(ckpt) == summary["source_sha256"]
        assert source[cv.SOURCE_METADATA_KEY] == METADATA

    def test_the_header_carries_exactly_one_metadata_key(self, tmp_path):
        """MUST: `__metadata__` has one key (`safetensors` does not preserve the order).

        With two or more, the header bytes produced from the same input move from run to run — the
        data section is identical but only the file's sha256 moves, which reads as a diff saying
        "the asset changed".
        """
        cv.convert(_write_ckpt(tmp_path))

        with safe_open(str(tmp_path / "weights.safetensors"), framework="pt") as handle:
            header = handle.metadata()

        assert list(header) == [cv.SOURCE_KEY]

    def test_two_runs_write_the_same_bytes(self, tmp_path):
        """MUST: the same input gives **the whole file** as the same bytes (regeneration produces no
        diff).

        Back when `__metadata__` held two or more keys, the data section and the tensor
        declarations matched exactly while the sha256 differed all three times (measured).
        """
        ckpt = _write_ckpt(tmp_path)
        first = (tmp_path / "a" / "weights.safetensors").resolve()
        second = (tmp_path / "b" / "weights.safetensors").resolve()
        first.parent.mkdir()
        second.parent.mkdir()

        cv.convert(ckpt, first)
        cv.convert(ckpt, second)

        assert cv._sha256(first) == cv._sha256(second)
        # The counterpart (not a tautology): a different input moves the sha256.
        other = (tmp_path / "c" / "weights.safetensors").resolve()
        other.parent.mkdir()
        torch.save({"state_dict": {"a.weight": torch.ones(2)}, "metadata": METADATA}, ckpt)
        cv.convert(ckpt, other)

        assert cv._sha256(other) != cv._sha256(first)

    def test_a_run_that_fails_a_gate_leaves_the_previous_output_in_place(
        self, tmp_path, monkeypatch
    ):
        """MUST (ADR 0052): move into place only after the checks pass — a failed run does not
        move the canonical path."""
        ckpt = _write_ckpt(tmp_path)
        cv.convert(ckpt)
        target = tmp_path / "weights.safetensors"
        metadata_path = tmp_path / "metadata.json"
        before = (target.read_bytes(), metadata_path.read_bytes())

        def refuse(path):
            raise cv.ConvertError("test: fail the reader-layout check")

        monkeypatch.setattr(cv, "assert_reader_layout", refuse)
        torch.save({"state_dict": {"other.weight": torch.ones(3)}, "metadata": {}}, ckpt)

        with pytest.raises(cv.ConvertError, match="reader-layout check"):
            cv.convert(ckpt)

        assert (target.read_bytes(), metadata_path.read_bytes()) == before
