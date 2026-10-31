"""The committed `gated_delta_scan` fixture is what `_shared.gated_delta_fixture` writes.

The inputs come from torch's CPU generator (deterministic across machines) and must match bit for
bit. Upstream's outputs and the errors derived from them may move in the last bits with the CPU's
kernels (the same split as the goldens' io — tools/exporter/tests/test_goldens.py), so the bands
are compared with a relative margin. transformers is a group dependency, so the test SKIPs
explicitly where it is not installed.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open

from _shared.gated_delta_fixture import CASES, DEFAULT_OUT, build


@pytest.fixture(scope="module")
def regenerated():
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    return build()


def _committed() -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(str(DEFAULT_OUT), framework="pt") as file:
        # safe_open is not a Mapping, so keys() is the only way to enumerate.
        tensors = {key: file.get_tensor(key) for key in file.keys()}  # noqa: SIM118
        return tensors, file.metadata()


def test_the_inputs_match_the_committed_fixture_bit_for_bit(regenerated):
    tensors, _ = regenerated
    committed, _ = _committed()

    assert sorted(tensors) == sorted(committed)
    for case in CASES:
        for slot in ("q", "k", "v", "g", "beta"):
            name = f"{case.name}.{slot}"
            assert torch.equal(tensors[name].view(torch.int32), committed[name].view(torch.int32))


def test_the_bands_match_the_committed_fixture(regenerated):
    _, metadata = regenerated
    _, committed = _committed()

    fresh = {record["name"]: record for record in json.loads(metadata["cases"])}
    stored = {record["name"]: record for record in json.loads(committed["cases"])}
    assert sorted(fresh) == sorted(stored) == sorted(case.name for case in CASES)
    for name, record in fresh.items():
        assert record["shape"] == stored[name]["shape"]
        for metric in ("max_abs", "rel_l2"):
            assert record["band"][metric] == pytest.approx(stored[name]["band"][metric], rel=1e-3)


def test_a_long_memory_case_keeps_t_at_512_or_more():
    """ADR 0135 decision 6 MUST: the default lane holds a long-memory case with T ≥ 512."""
    assert any(
        case.time >= 512 and all(window.regime == "long" for window in case.windows)
        for case in CASES
    )
