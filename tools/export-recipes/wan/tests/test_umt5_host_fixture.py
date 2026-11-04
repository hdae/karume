"""The contract of `wan/umt5_host_fixture.py` (the fixture the TS side checks the relative-position
bucket table against).

- The tracked fixture (`packages/models/tests/fixtures/wan-umt5/`) equals the generator's current
  output (catches a fixture left stale after the generator or the upstream formula changed — the TS
  side compares only against this fixture)
- The bucket configuration the fixture records equals `text_encoder/config.json` of the pinned
  revision (SKIP on a machine without the config — fetch it with
  `python -m wan.text_embeds --fetch`)
- The self-check of the distance column against the table's orientation stops a flipped orientation
  before anything is written

Regeneration works from the recorded configuration (it reads neither the real model nor the config —
it calls the formula on the upstream module on meta).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from wan import umt5_host_fixture as fixture

COMMITTED_META = fixture.DEFAULT_OUT / fixture.META_FILE
COMMITTED_TENSORS = fixture.DEFAULT_OUT / fixture.TENSORS_FILE


@pytest.fixture(scope="module")
def committed() -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    return (
        json.loads(COMMITTED_META.read_text(encoding="utf-8")),
        load_file(str(COMMITTED_TENSORS)),
    )


def recorded_config(meta: dict[str, Any]) -> Any:
    from transformers import UMT5Config

    return UMT5Config(
        relative_attention_num_buckets=meta["num_buckets"],
        relative_attention_max_distance=meta["max_distance"],
        is_decoder=not meta["bidirectional"],
    )


def test_committed_fixture_is_what_the_generator_writes(committed):
    pytest.importorskip("transformers")
    meta, tensors = committed
    source = {"repo": meta["source"]["repo"], "revision": meta["source"]["revision"]}

    expected_tensors, expected_meta = fixture.build_fixture(recorded_config(meta), source)

    assert meta == expected_meta
    assert sorted(tensors) == sorted(expected_tensors)
    for key, value in expected_tensors.items():
        assert torch.equal(tensors[key], value), key


def test_recorded_bucket_config_is_the_pinned_text_encoder(committed):
    pytest.importorskip("transformers")
    from wan.sources import WanSourceError

    meta, _ = committed
    try:
        config = fixture.pinned_config()
    except WanSourceError as error:
        pytest.skip(f"the config of the pinned text_encoder is not available locally: {error}")

    assert (
        config.relative_attention_num_buckets,
        config.relative_attention_max_distance,
        not config.is_decoder,
    ) == (meta["num_buckets"], meta["max_distance"], meta["bidirectional"])


def test_consistency_check_stops_a_flipped_distance_column(committed):
    """Baking the distance column reversed (query − key) disagrees with the table and fails before
    anything is written."""
    pytest.importorskip("transformers")
    meta, _ = committed
    attention = fixture.umt5_patch.bucket_attention(recorded_config(meta))
    flipped = fixture.bucket_by_distance(attention).flip(0)

    with pytest.raises(AssertionError, match="Toeplitz"):
        fixture.check_consistent(attention, flipped)
