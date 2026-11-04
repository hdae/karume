"""A copy of the Wan2.2 TI2V-5B inverse-normalization statistics (ADR 0121 step 5).

`WAN22_LATENTS_MEAN` / `WAN22_LATENTS_STD` (48 values each) in the TS
`packages/models/src/wan/latents.ts` are hand copies of values that exist only in the upstream VAE
config. They are bound from both sides through the tracked fixture
`packages/models/tests/fixtures/wan-latents/wan22-ti2v.json`: the TS host test
(`wan_ti2v_vae_tiles_host_test.ts`) checks that the constants and the fixture are bit-identical,
and this file checks that the fixture equals the `vae/config.json` of the pinned revision converted
to `np.float32`. Rewriting only one side turns one of them red.

The tests that read the config SKIP on a machine without the pinned snapshot
(`conftest.wan22_snapshot` — the same discipline as the real-weight tests). The provenance of the
fixture (repo and revision equal the pin of `wan.sources`) is checked even without the snapshot.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from _shared.paths import REPO_ROOT
from wan.sources import SOURCES

#: The fixture shared by the TS side and the recipe.
_FIXTURE = REPO_ROOT / "packages/models/tests/fixtures/wan-latents/wan22-ti2v.json"

#: The number of latent channels (`z_dim` of the upstream config).
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
    }, "after changing the pin of wan.sources, re-derive the fixture statistics from the new config"


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
    # f32 → f64 is exact, so this must exactly match the fixture (the shortest decimal form of the
    # f32 read as f64).
    np.testing.assert_array_equal(
        np.asarray(fixture[key], dtype=np.float64),
        upstream.astype(np.float64),
        err_msg=f"latents_{key}: the fixture and the pinned config (f32) differ"
        " — align the constants of latents.ts too",
    )
