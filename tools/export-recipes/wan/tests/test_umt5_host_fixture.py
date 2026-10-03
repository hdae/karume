"""`wan/umt5_host_fixture.py`（相対位置のバケット表の TS 照合用 fixture）の約束事。

- 追跡している fixture（`packages/models/tests/fixtures/wan-umt5/`）が生成器の今の出力と一致する
  （生成器か上流の式が変わったのに fixture が古いまま、の検出 — TS 側はこの fixture としか比べない）
- fixture が記録したバケットの構成が pin した revision の `text_encoder/config.json` と一致する
  （config が手元に無い機では SKIP — 取得は `python -m wan.text_embeds --fetch`）
- 距離の列と表の向きの自己検査が、向きの取り違えを書く前に止める

再生成は記録された構成から作る（実モデルも config も読まない — meta 上の上流のモジュールで
式を呼ぶ）。
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
        pytest.skip(f"pin した text_encoder の config が手元に無い: {error}")

    assert (
        config.relative_attention_num_buckets,
        config.relative_attention_max_distance,
        not config.is_decoder,
    ) == (meta["num_buckets"], meta["max_distance"], meta["bidirectional"])


def test_consistency_check_stops_a_flipped_distance_column(committed):
    """距離の列を逆向き（クエリ − キー）で焼くと、表と食い違って書く前に落ちる。"""
    pytest.importorskip("transformers")
    meta, _ = committed
    attention = fixture.umt5_patch.bucket_attention(recorded_config(meta))
    flipped = fixture.bucket_by_distance(attention).flip(0)

    with pytest.raises(AssertionError, match="Toeplitz"):
        fixture.check_consistent(attention, flipped)
