"""Wan2.2 TI2V-5B の逆正規化の統計の写し（ADR 0121 段 5）。

TS の `packages/models/src/wan/latents.ts` の `WAN22_LATENTS_MEAN` /
`WAN22_LATENTS_STD`（48 本ずつ）は、上流の VAE の config にしか無い値の手写し。追跡する
fixture `packages/models/tests/fixtures/wan-latents/wan22-ti2v.json` を挟んで両側から縛る:
TS のホストテスト（`wan_ti2v_vae_tiles_host_test.ts`）が定数と fixture のビット一致を、
ここが pin した revision の `vae/config.json` を `np.float32` にした値と fixture の一致を
見る。片側だけの書き換えはどちらかが赤にする。

config を読むテストは pin した snapshot が手元に無い機では SKIP
（`conftest.wan22_snapshot` — 実重みのテストと同じ規律）。fixture の出所（repo と
revision が `wan.sources` の pin と同じこと）は snapshot が無くても見る。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from _shared.paths import REPO_ROOT
from wan.sources import SOURCES

#: TS と recipe が挟む fixture。
_FIXTURE = REPO_ROOT / "packages/models/tests/fixtures/wan-latents/wan22-ti2v.json"

#: 潜在のチャネル数（上流 config の `z_dim`）。
_LATENT_CHANNELS = 48


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_the_fixture_names_the_pinned_revision(fixture) -> None:
    source = SOURCES["ti2v-5b"]
    assert fixture["source"] == {
        "repo": source.repo,
        "revision": source.revision,
        "reference": "AutoencoderKLWan.config",
    }, "wan.sources の pin を変えたら fixture の統計を新しい config から取り直す"


@pytest.mark.parametrize("key", ["mean", "std"])
def test_the_fixture_holds_float32_values(fixture, key: str) -> None:
    values = np.asarray(fixture[key], dtype=np.float64)
    assert values.shape == (_LATENT_CHANNELS,)
    np.testing.assert_array_equal(values, values.astype(np.float32).astype(np.float64))


@pytest.mark.parametrize("key", ["mean", "std"])
def test_the_fixture_matches_the_pinned_config_as_float32(
    fixture, wan22_snapshot, key: str
) -> None:
    config = json.loads((wan22_snapshot / "vae" / "config.json").read_text(encoding="utf-8"))
    upstream = np.asarray(config[f"latents_{key}"], dtype=np.float32)
    assert upstream.shape == (_LATENT_CHANNELS,)
    # f32 → f64 は厳密なので、fixture（f32 の最短 10 進表記を f64 で読んだ値）と完全一致するはず。
    np.testing.assert_array_equal(
        np.asarray(fixture[key], dtype=np.float64),
        upstream.astype(np.float64),
        err_msg=f"latents_{key}: fixture と pin の config（f32）が違う — latents.ts の定数も揃える",
    )
