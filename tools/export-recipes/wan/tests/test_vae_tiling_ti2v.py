"""The contract of the Wan2.2 path of `wan/vae_tiling.py` (ADR 0121 step 5 — the tile reference in
patchify space).

What is pinned here is the side where a break raises no exception and **only the values drift
quietly**:

- the start positions of the 2.2 tile plan (tile 16, overlap 4 latents) equal the TS side's
  ({@link MIRRORED_STARTS_TI2V} — frozen twice with the same values as `AXIS_STARTS_TI2V` of the TS
  side's `packages/models/tests/wan_ti2v_vae_tiles_host_test.ts`), and the tile counts and blend
  widths of the four sizes
- the unpatchify order ({@link UNPATCHIFY_SOURCE_CHANNEL} — frozen twice with the TS side's table)
  equals the upstream diffusers `unpatchify`
- the tile reference in the degenerate case (one tile) is **bit-identical** to the upstream untiled
  chunk loop, and with unpatchify → clamp applied it is **bit-identical** to the upstream `_decode`
  itself (tying the reference's origin to the upstream). Two levels: the synthetic 2.2 VAE (always
  runs) and real weights (only on a machine with the pinned 5B snapshot)
- `frames_rgb` of `accept` is `frames` with the upstream unpatchify → clamp applied, and the only
  two things added to the metadata are `patch_size` and `reference`
- the CLI pair of `--model` and `--case` (a pair not in the table fails before the VAE is read), the
  per-model defaults, and the check of the loaded VAE against the series' premises
- the whole metadata of 2.1's `write_fixture` and the set of tensor names (decode is swapped out —
  the values are R0's job)

The 2.1 path (`test_vae_tiling.py`) does not change — that 2.1's fixture is unchanged is confirmed
by rewriting `vae_tiles.{band,accept}` and comparing them with the existing ones through
`wan.series_check` (it is long on the CPU, so it is not in pytest).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from _shared.paths import SERIES_ROOT
from wan import export_vae, vae_patch, vae_tiling

#: Wan2.2's latent tile (`export_vae.TI2V_TILE`) and patchify factor (`patch_size` of the upstream
#: config).
TILE = 16
PATCH_SIZE = 2

#: Wan2.2's lower bound of the overlap (latent) = 64 px ÷ (the graph's ratio 8 × unpatchify 2) = 4.
OVERLAP = vae_tiling.min_overlap_latent(export_vae.SPATIAL_SCALE, PATCH_SIZE)

#: Start positions at tile edge 16 and overlap lower bound 4 (the latent's full length → the
#: sequence of start positions). Frozen with **the same values** as the TS side (`AXIS_STARTS_TI2V`
#: of `packages/models/tests/wan_ti2v_vae_tiles_host_test.ts`) — the only path that compares the two
#: implementations by machine is the GPU comparison (the fixture's metadata), so this catches a
#: mirror drift on a machine without a GPU too.
#:
#: - 16: degenerate (one tile — the shape of the degenerate check)
#: - 30 / 52: the two edges of 832×480 / 480×832 (latent = pixels ÷ 16)
#: - 44 / 80: the two edges of 1280×704 / 704×1280 (the blend width differs per pair)
MIRRORED_STARTS_TI2V = {
    16: (0,),
    30: (0, 7, 14),
    44: (0, 9, 19, 28),
    52: (0, 12, 24, 36),
    80: (0, 11, 21, 32, 43, 53, 64),
}

#: The four sizes (width × height in pixels) → (tile count, row blend width, column blend width).
#: The blend widths are pixels in patchify space (= the overlap in latents × the graph's ratio 8).
SIZES = {
    (832, 480): (12, [72, 72], [32, 32, 32]),
    (480, 832): (12, [32, 32, 32], [72, 72]),
    (1280, 704): (28, [56, 48, 56], [40, 48, 40, 40, 48, 40]),
    (704, 1280): (28, [40, 48, 40, 40, 48, 40], [56, 48, 56]),
}

#: The unpatchify order: `[dy][dx]` → the offset within the channel in patchify space (dy = the
#: pixel offset along the height, dx = along the width).
#: `out[c, f, y·2 + dy, x·2 + dx] = in[c·4 + table[dy][dx], f, y, x]` (the offset is dx·2 + dy).
#: The same values as the TS side (`UNPATCHIFY_SOURCE_CHANNEL` of
#: `packages/models/tests/wan_vae_tiles_test.ts`).
UNPATCHIFY_SOURCE_CHANNEL = ((0, 2), (1, 3))


def _upstream_unpatchify(frames: torch.Tensor) -> torch.Tensor:
    wan_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
    return wan_vae.unpatchify(frames, patch_size=PATCH_SIZE)


def _table_matches(table, patchified: torch.Tensor, pixels: torch.Tensor) -> bool:
    """Whether each offset of `pixels [1, C, F, 2h, 2w]` is the channel of
    `patchified [1, 4C, F, h, w]` that the table points to."""
    batch, channels, frames, height, width = patchified.shape
    grouped = patchified.view(batch, channels // 4, 4, frames, height, width)
    return all(
        torch.equal(pixels[:, :, :, dy::2, dx::2], grouped[:, :, table[dy][dx]])
        for dy in range(2)
        for dx in range(2)
    )


class TestPlanTi2v:
    def test_the_overlap_is_four_latents_and_the_tile_is_sixteen(self):
        """The values the frozen table was built from (tile 16, overlap 4) are the same as the
        recipe's declared values."""
        assert OVERLAP == 4
        assert export_vae.TI2V_TILE == TILE
        assert export_vae.VAE_SERIES["ti2v-5b"].patch_size == PATCH_SIZE

    @pytest.mark.parametrize("extent", sorted(MIRRORED_STARTS_TI2V))
    def test_starts_match_the_ts_side(self, extent: int):
        assert (
            vae_tiling.plan_tile_axis(extent, TILE, OVERLAP).starts == MIRRORED_STARTS_TI2V[extent]
        )

    @pytest.mark.parametrize(("width", "height"), sorted(SIZES))
    def test_tiles_and_blend_widths_of_the_four_sizes(self, width: int, height: int):
        tiles, rows_blend, cols_blend = SIZES[(width, height)]
        plan = vae_tiling.plan_tiles(height // 16, width // 16, TILE, OVERLAP)

        assert plan.scale == 8
        assert plan.rows.starts == MIRRORED_STARTS_TI2V[height // 16]
        assert plan.cols.starts == MIRRORED_STARTS_TI2V[width // 16]
        assert plan.tiles == tiles
        assert plan.rows.blends(plan.scale) == rows_blend
        assert plan.cols.blends(plan.scale) == cols_blend


class TestUnpatchifyOrder:
    def test_the_channel_table_matches_upstream_unpatchify(self):
        patchified = torch.arange(12 * 2 * 3 * 5, dtype=torch.float32).view(1, 12, 2, 3, 5)

        pixels = _upstream_unpatchify(patchified)

        assert pixels.shape == (1, 3, 2, 6, 10)
        assert _table_matches(UNPATCHIFY_SOURCE_CHANNEL, patchified, pixels)

    def test_the_table_with_dx_and_dy_swapped_does_not_match(self):
        """The control: a table with the vertical and horizontal offsets swapped does not match (the
        check above can catch a mix-up)."""
        patchified = torch.arange(12 * 2 * 3 * 5, dtype=torch.float32).view(1, 12, 2, 3, 5)
        swapped = ((0, 1), (2, 3))

        assert not _table_matches(swapped, patchified, _upstream_unpatchify(patchified))

    def test_unpatchify_frames_is_the_upstream_function(self):
        # The function under test also imports the upstream unpatchify — it is evaluated before the
        # importorskip of the reference it is compared with.
        pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
        frames = torch.randn(1, 12, 3, 4, 6, generator=torch.Generator().manual_seed(5))

        assert torch.equal(
            vae_tiling.unpatchify_frames(frames, PATCH_SIZE), _upstream_unpatchify(frames)
        )

    def test_a_generation_without_patchify_is_returned_as_is(self):
        frames = torch.randn(1, 3, 2, 4, 4)

        assert vae_tiling.unpatchify_frames(frames, None) is frames


class TestFixtureCasesTi2v:
    def test_the_tables_cover_the_series_of_export_vae(self):
        assert set(vae_tiling.FIXTURE_CASES_BY_MODEL) == set(export_vae.VAE_SERIES)
        assert vae_tiling.FIXTURE_CASES_BY_MODEL["t2v-1.3b"] is vae_tiling.FIXTURE_CASES
        assert vae_tiling.FIXTURE_CASES_BY_MODEL["ti2v-5b"] is vae_tiling.TI2V_FIXTURE_CASES

    def test_the_cases_are_the_three_sizes_of_the_design(self):
        """band = 832×480×81, accept = 480×832×9, wide = 1280×704×5 (latents are pixels ÷ 16, frames
        are 1 + 4(chunk − 1))."""
        cases = {case.name: case for case in vae_tiling.TI2V_FIXTURE_CASES}

        assert list(cases) == ["band", "accept", "wide"]
        band, accept, wide = cases["band"], cases["accept"], cases["wide"]
        assert (band.height, band.width, band.chunks, band.role) == (30, 52, 21, "band")
        assert (accept.height, accept.width, accept.chunks, accept.role) == (52, 30, 3, "accept")
        assert (wide.height, wide.width, wide.chunks, wide.role) == (44, 80, 2, "accept")
        # The seeds are the values of the design table (the fixture holds the latents, so e2e's
        # self-consistency survives a drift — a drift from the table is caught only here).
        assert (band.seed, accept.seed, wide.seed) == (20261051, 20261052, 20261053)

    def test_acceptance_differs_from_the_band_in_seed_chunks_and_placement(self):
        cases = {case.name: case for case in vae_tiling.TI2V_FIXTURE_CASES}
        band = cases["band"]

        for name in ("accept", "wide"):
            case = cases[name]
            assert case.chunks != band.chunks, name
            assert (case.height, case.width) != (band.height, band.width), name

    def test_seeds_are_distinct_from_every_other_fixture(self):
        """No latent is shared with the 2.1 tile reference or with step 4's chunk sequences (2.1 /
        2.2)."""
        seeds = [case.seed for case in vae_tiling.TI2V_FIXTURE_CASES]
        others = {
            case.seed
            for case in (
                *vae_tiling.FIXTURE_CASES,
                *export_vae.FIXTURE_CASES,
                *export_vae.TI2V_FIXTURE_CASES,
            )
        }

        assert len(set(seeds)) == len(seeds)
        assert not set(seeds) & others

    def test_only_accept_of_ti2v_carries_rgb_and_the_untiled_reference(self):
        """RGB is only in accept of the generation that patchifies (ruling F-11); the 2.1 table has
        no RGB."""
        flags = {
            (model, case.name): (case.full, case.rgb)
            for model, cases in vae_tiling.FIXTURE_CASES_BY_MODEL.items()
            for case in cases
        }

        assert {key for key, (_, rgb) in flags.items() if rgb} == {("ti2v-5b", "accept")}
        assert flags[("ti2v-5b", "accept")] == (True, True)
        assert flags[("ti2v-5b", "band")] == (False, False)
        assert flags[("ti2v-5b", "wide")] == (False, False)

    def test_the_wan21_reference_string_is_unchanged(self):
        """The 2.1 metadata `reference` is the same string as before the 2.2 path was added (the
        premise of R0's metadata match)."""
        assert vae_tiling.FIXTURE_REFERENCE == (
            "snapped-tile decode with upstream blend_v / blend_h before clamp (CPU f32)"
        )


def _seeded(seed: int, height: int, width: int, chunks: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 48, chunks, height, width, generator=generator)


def _assert_degenerate_tile_is_upstream(vae, latents: torch.Tensor) -> None:
    """A one-tile reference ≡ the untiled chunk loop (before clamping), clamp(unpatchify) ≡ the
    upstream `_decode`.

    `latents` is square (tile edge = latent edge).
    """
    _, _, chunks, height, width = latents.shape
    plan = vae_tiling.plan_tiles(height, width, tile=height, min_overlap=OVERLAP)
    assert plan.tiles == 1

    with torch.no_grad():
        tiled = vae_tiling.tiled_decode_unclamped(vae, latents, plan)
        untiled = vae_patch.reference_decode_unclamped(vae, latents)
        decoded = vae._decode(latents, return_dict=False)[0]

    assert tiled.shape == (1, 12, 1 + 4 * (chunks - 1), height * 8, width * 8)
    assert torch.equal(tiled, untiled)
    assert torch.equal(vae_tiling.unpatchify_frames(tiled, PATCH_SIZE).clamp(-1.0, 1.0), decoded)


class TestDegenerateTileSynthetic:
    def test_one_tile_is_the_untiled_chunk_loop_and_the_upstream_decode_bitwise(
        self, ti2v_synthetic_vae
    ):
        """The latent is exactly one tile (8×8, 3 chunks). It checks that the values do not change,
        including the difference in how the post-quant 1×1×1 conv is applied (per frame for the
        tiles, once over all frames untiled)."""
        _assert_degenerate_tile_is_upstream(ti2v_synthetic_vae, _seeded(11, 8, 8))

    def test_a_generation_the_chunk_graph_does_not_take_is_refused(self, ti2v_synthetic_vae):
        """With residuals and without patchify is a generation the chunk-graph rewrite does not
        take. Without the check, decode would go through as is (the shapes and the plan are right),
        so the exception can come only from this check."""
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_config(ti2v_synthetic_vae.config, patch_size=None).eval()
        plan = vae_tiling.plan_tiles(8, 8, tile=8, min_overlap=OVERLAP)

        with pytest.raises(vae_patch.UnsupportedVaeError, match="unsupported"):
            vae_tiling.decode_tiles(vae, _seeded(11, 8, 8, chunks=1), plan)


#: A small case written with the synthetic VAE (latent 8×12, tile 8 → 2 tiles at columns 0 / 4,
#: 2 chunks).
SYNTHETIC_CASE = vae_tiling.FixtureCase(
    "accept", seed=3, chunks=2, height=8, width=12, role="accept", full=True, rgb=True
)


@pytest.fixture(scope="module")
def written(ti2v_synthetic_vae, tmp_path_factory):
    """(summary, tensors, metadata) of {@link SYNTHETIC_CASE} written with the synthetic VAE and
    read back."""
    from safetensors import safe_open

    out = tmp_path_factory.mktemp("ti2v-tiles")
    summary = vae_tiling.write_fixture(
        ti2v_synthetic_vae, SYNTHETIC_CASE, 8, out, min_overlap=OVERLAP, patch_size=PATCH_SIZE
    )
    with safe_open(summary["path"], framework="pt") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118
        metadata = handle.metadata()
    return summary, tensors, metadata


class TestWriteFixtureTi2v:
    def test_frames_stay_in_patchify_space_and_rgb_is_three_channels(self, written):
        _, tensors, _ = written

        assert set(tensors) == {"latents", "frames", "frames_full", "frames_rgb"}
        assert tensors["frames"].shape == (12, 5, 64, 96)
        assert tensors["frames_full"].shape == (12, 5, 64, 96)
        assert tensors["frames_rgb"].shape == (3, 5, 128, 192)

    def test_frames_rgb_is_the_upstream_unpatchify_then_clamp_of_frames(self, written):
        """Rebuild independently from `frames` with the table (the order confirmed with the
        upstream `unpatchify(arange)`) and compare, and compare with the upstream function itself
        too."""
        _, tensors, _ = written
        frames, rgb = tensors["frames"], tensors["frames_rgb"]
        # Premise: some elements are outside [-1, 1] before clamping (otherwise a missing clamp
        # would not split this check).
        assert bool((frames.abs() > 1.0).any())

        assert _table_matches(UNPATCHIFY_SOURCE_CHANNEL, frames.clamp(-1.0, 1.0)[None], rgb[None])
        assert torch.equal(rgb, _upstream_unpatchify(frames[None])[0].clamp(-1.0, 1.0))

    def test_the_metadata_adds_only_patch_size_and_the_reference(self, written):
        """No statistics are added (the 48 copied values are pinned by the tracked fixture —
        ruling F-04)."""
        _, _, metadata = written

        assert metadata == {
            "seed": "3",
            "chunks": "2",
            "role": "accept",
            "weights": "f16-rounded",
            "reference": (
                "snapped-tile decode with upstream blend_v / blend_h in patchify space"
                " before unpatchify and clamp (CPU f32)"
            ),
            "tile": "8",
            "scale": "8",
            "rows_starts": "0",
            "cols_starts": "0,4",
            "rows_blend": "",
            "cols_blend": "32",
            "patch_size": "2",
            "frames_full": (
                "upstream non-tiled _decode chunk loop before unpatchify and clamp (CPU f32)"
            ),
        }

    def test_a_patch_size_that_disagrees_with_the_config_is_rejected(
        self, ti2v_synthetic_vae, tmp_path: Path
    ):
        with pytest.raises(ValueError, match="differs from the upstream config's 2"):
            vae_tiling.write_fixture(
                ti2v_synthetic_vae,
                SYNTHETIC_CASE,
                8,
                tmp_path,
                min_overlap=OVERLAP,
                patch_size=None,
            )
        assert not any(tmp_path.iterdir())

    def test_rgb_is_refused_for_a_generation_without_patchify(self, tmp_path: Path):
        vae = SimpleNamespace(config=SimpleNamespace(patch_size=None))

        with pytest.raises(ValueError, match="only for a generation that patchifies"):
            vae_tiling.write_fixture(
                vae, SYNTHETIC_CASE, 8, tmp_path, min_overlap=OVERLAP, patch_size=None
            )

    def test_the_overlap_and_the_patch_size_are_required(self, ti2v_synthetic_vae, tmp_path: Path):
        """Forgetting to pass it is a TypeError — no silent writing with the 2.1 value."""
        with pytest.raises(TypeError):
            vae_tiling.write_fixture(ti2v_synthetic_vae, SYNTHETIC_CASE, 8, tmp_path)  # type: ignore[call-arg]


#: The metadata of the 2.1 fixture (values copied from the output written before the 2.2 path was
#: added, `wan2.1-t2v-1.3b-f16-dyn/vae_tiles.{band,accept}.safetensors` under `outputs/series/`).
WAN21_REFERENCE = "snapped-tile decode with upstream blend_v / blend_h before clamp (CPU f32)"
WAN21_METADATA = {
    "band": {
        "seed": "20261012",
        "chunks": "9",
        "role": "band",
        "weights": "f16-rounded",
        "reference": WAN21_REFERENCE,
        "tile": "32",
        "scale": "8",
        "rows_starts": "0,14,28",
        "cols_starts": "0,24,48,72",
        "rows_blend": "144,144",
        "cols_blend": "64,64,64",
        "frames_full": "upstream non-tiled _decode chunk loop before clamp (CPU f32)",
    },
    "accept": {
        "seed": "20261013",
        "chunks": "3",
        "role": "accept",
        "weights": "f16-rounded",
        "reference": WAN21_REFERENCE,
        "tile": "32",
        "scale": "8",
        "rows_starts": "0,24,48,72",
        "cols_starts": "0,14,28",
        "rows_blend": "64,64,64",
        "cols_blend": "144,144",
    },
}
WAN21_TENSORS = {
    "band": {"latents", "frames", "frames_full"},
    "accept": {"latents", "frames"},
}


class TestWriteFixtureWan21:
    """Freeze the whole metadata and the set of tensor names of the 2.1 path (`patch_size` is
    None).

    decode is swapped out (that the values are unchanged is R0's job — comparing the rewritten real
    outputs with the existing ones through `wan.series_check`). What this catches is only a
    metadata regression caused by adding the 2.2 path.
    """

    @pytest.fixture
    def decode_stubbed(self, monkeypatch):
        def seeded_latents(_vae, case):
            return torch.zeros(1, 16, case.chunks, case.height, case.width)

        def decoded(_vae, latents, *_plan):
            _, _, chunks, height, width = latents.shape
            return torch.full((1, 3, 1 + 4 * (chunks - 1), height * 8, width * 8), 0.5)

        monkeypatch.setattr(vae_tiling, "seeded_latents", seeded_latents)
        monkeypatch.setattr(vae_tiling, "tiled_decode_unclamped", decoded)
        monkeypatch.setattr(vae_tiling.vae_patch, "reference_decode_unclamped", decoded)

    @pytest.mark.parametrize("case", vae_tiling.FIXTURE_CASES, ids=lambda case: case.name)
    def test_the_metadata_and_the_tensor_names_are_those_before_ti2v(
        self, decode_stubbed, case, tmp_path: Path
    ):
        from safetensors import safe_open

        vae = SimpleNamespace(config=SimpleNamespace(patch_size=None))
        summary = vae_tiling.write_fixture(
            vae,
            case,
            export_vae.DEFAULT_TILE,
            tmp_path,
            min_overlap=vae_tiling.WAN21_MIN_OVERLAP_LATENT,
            patch_size=None,
        )

        with safe_open(summary["path"], framework="pt") as handle:
            assert set(handle.keys()) == WAN21_TENSORS[case.name]
            assert handle.metadata() == WAN21_METADATA[case.name]


class _LoadedError(Exception):
    """A marker that the CLI checks got as far as reading the VAE (it stops without reading)."""


class TestCli:
    @pytest.fixture
    def recorded(self, monkeypatch):
        """Record the wiring of `main` (model, tile, location, overlap, cases) without reading the
        VAE."""
        calls: dict[str, list] = {"load": [], "vaes": [], "series": [], "write": []}

        def load_vae(model: str, *, round_f16: bool):
            vae = SimpleNamespace(use_tiling=False)
            calls["load"].append((model, round_f16))
            calls["vaes"].append(vae)
            return vae

        def assert_series_config(vae, series):
            calls["series"].append((vae, series))

        def write_fixture(_vae, case, tile, out, *, min_overlap, patch_size):
            calls["write"].append((case.name, tile, out, min_overlap, patch_size))
            return {}

        monkeypatch.setattr(vae_tiling.export_vae, "load_vae", load_vae)
        monkeypatch.setattr(vae_tiling.export_vae, "assert_series_config", assert_series_config)
        monkeypatch.setattr(vae_tiling, "write_fixture", write_fixture)
        return calls

    @staticmethod
    def _assert_the_series_config_is_checked(recorded, model: str) -> None:
        """The loaded VAE is checked against the premises of that model's series exactly once
        (without it, a different generation would be written silently)."""
        [(vae, series)] = recorded["series"]
        [loaded] = recorded["vaes"]
        assert vae is loaded
        assert series is export_vae.VAE_SERIES[model]

    def test_the_default_model_keeps_the_wan21_values(self, recorded):
        assert vae_tiling.main([]) == 0

        out = SERIES_ROOT / export_vae.SERIES_NAME
        assert recorded["load"] == [("t2v-1.3b", True)]
        self._assert_the_series_config_is_checked(recorded, "t2v-1.3b")
        assert recorded["write"] == [
            ("band", 32, out, 8, None),
            ("accept", 32, out, 8, None),
        ]

    def test_ti2v_takes_its_series_tile_overlap_and_cases(self, recorded):
        assert vae_tiling.main(["--model", "ti2v-5b"]) == 0

        out = SERIES_ROOT / export_vae.TI2V_SERIES_NAME
        assert recorded["load"] == [("ti2v-5b", True)]
        self._assert_the_series_config_is_checked(recorded, "ti2v-5b")
        assert recorded["write"] == [
            ("band", 16, out, 4, 2),
            ("accept", 16, out, 4, 2),
            ("wide", 16, out, 4, 2),
        ]

    @pytest.mark.parametrize(("model", "tile"), [("t2v-1.3b", 8), ("ti2v-5b", 4)])
    def test_a_tile_not_above_the_overlap_is_rejected_before_loading(
        self, recorded, capsys, model: str, tile: int
    ):
        """A tile exactly at the overlap's lower bound is not accepted by the plan — it fails before
        the VAE is read."""
        with pytest.raises(SystemExit) as exited:
            vae_tiling.main(["--model", model, "--tile", str(tile)])

        assert exited.value.code == 2
        assert (
            f"--tile {tile} is not above the minimum overlap {tile} (latents)"
            in capsys.readouterr().err
        )
        assert recorded["load"] == []

    def test_a_case_of_another_model_is_rejected_before_loading(self, monkeypatch, capsys):
        def load_vae(*_args, **_kwargs):
            raise _LoadedError

        monkeypatch.setattr(vae_tiling.export_vae, "load_vae", load_vae)

        with pytest.raises(SystemExit) as exited:
            vae_tiling.main(["--model", "t2v-1.3b", "--case", "wide"])

        assert exited.value.code == 2
        assert "--case ['wide'] is not in the table of --model t2v-1.3b" in capsys.readouterr().err

    def test_case_choices_are_the_union_of_the_models(self, recorded):
        """`wide` is only in the 2.2 table, but it is accepted as a `--case` choice."""
        assert vae_tiling.main(["--model", "ti2v-5b", "--case", "wide"]) == 0
        assert [call[0] for call in recorded["write"]] == ["wide"]

        with pytest.raises(SystemExit) as exited:
            vae_tiling.main(["--case", "nope"])
        assert exited.value.code == 2


# ---- Real weights (SKIP without the pinned 5B snapshot) ------------------------------------------


@pytest.fixture(scope="module")
def vae(wan22_snapshot):
    """The 5B VAE of the pinned revision (CPU f32, no rounding — the identity claim does not depend
    on the weight values)."""
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        wan22_snapshot, subfolder="vae", torch_dtype=torch.float32
    ).eval()


class TestDegenerateTileRealWeights:
    def test_one_tile_is_the_untiled_chunk_loop_and_the_upstream_decode_bitwise(self, vae):
        """The same two points with real weights (latent 8×8, 3 chunks — smaller than the real tile
        16, but the claim does not depend on the shape)."""
        mean = torch.tensor(vae.config.latents_mean).view(1, 48, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, 48, 1, 1, 1)

        _assert_degenerate_tile_is_upstream(vae, _seeded(13, 8, 8) * std + mean)
