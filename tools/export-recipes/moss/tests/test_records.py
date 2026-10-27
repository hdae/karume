"""`moss/records.py` — the record format and the bit comparison."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from moss import records

KEY = "karume.moss.test"


class TestBitsEqual:
    def test_a_signed_zero_differs(self):
        assert not records.bits_equal(torch.tensor([0.0]), torch.tensor([-0.0]))

    def test_the_same_nan_is_equal(self):
        nan = torch.tensor([float("nan")])

        assert records.bits_equal(nan, nan.clone())

    def test_the_dtype_and_shape_count(self):
        assert not records.bits_equal(torch.zeros(2), torch.zeros(2, dtype=torch.float64))
        assert not records.bits_equal(torch.zeros(2), torch.zeros(1, 2))

    def test_a_zero_dim_tensor_compares(self):
        assert records.bits_equal(torch.tensor(3, dtype=torch.int64), torch.tensor(3))


class TestRecordFile:
    def test_names_that_share_storage_are_written_as_copies(self, tmp_path):
        """A step's output latents are the next step's input: one value, two names."""
        base = torch.arange(6, dtype=torch.float32)
        tensors = {"all": base, "first": base[:1].reshape(1), "alias": base}
        path = tmp_path / "r.safetensors"

        records.write_record(path, tensors, {"n": 1}, key=KEY)
        read, metadata = records.read_record(path, key=KEY)

        assert metadata == {"n": 1}
        assert all(torch.equal(read[name], tensors[name]) for name in tensors)

    def test_another_metadata_layout_is_refused(self, tmp_path):
        path = tmp_path / "r.safetensors"
        save_file({"a": torch.zeros(1)}, str(path), metadata={KEY: "{}", "extra": "x"})

        with pytest.raises(records.RecordError):
            records.read_record(path, key=KEY)
