"""The 32-latent tile references of `wan/vae_tiling.py` (ADR 0134 decision 6).

What is pinned here fails silently when it breaks:

- the file name: the default tile keeps `vae_tiles.<case>.safetensors` (every existing reader), any
  other tile adds `.t<tile>` — a missing suffix would overwrite the 16-tile references
- the plan at tile 32 with the Wan2.2 overlap (4 latents) for 1280×704 (latent 44×80)
- the CLI refusals (not a multiple of 16, not above the overlap, a named case smaller than the tile)
  happen before the VAE is read, and the default case list skips the cases a 32 tile cannot take
- the meta carries the tile the references were planned at (the TS plan test matches it)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from _shared.paths import SERIES_ROOT
from wan import export_vae, vae_tiling

#: Wan2.2's series tile, the shared pair's tile, and the overlap (64 px ÷ 16 = 4 latents).
TI2V_TILE = 16
TILE = 32
PATCH_SIZE = 2
OVERLAP = vae_tiling.min_overlap_latent(export_vae.SPATIAL_SCALE, PATCH_SIZE)


class TestFixtureName:
    @pytest.mark.parametrize(
        ("tile", "default_tile", "suffix"),
        [(16, 16, ""), (32, 16, ".t32"), (32, 32, ""), (16, 32, ".t16")],
    )
    def test_only_a_non_default_tile_gets_a_suffix(self, tile: int, default_tile: int, suffix: str):
        assert vae_tiling.fixture_tile_suffix(tile, default_tile) == suffix

    def test_the_default_name_is_unchanged_and_the_suffix_precedes_the_extension(self):
        assert vae_tiling.fixture_name("wide", 16, 16) == "vae_tiles.wide.safetensors"
        assert vae_tiling.fixture_name("wide", 32, 16) == "vae_tiles.wide.t32.safetensors"

    @pytest.mark.parametrize(("patch_size", "tile"), [(None, 32), (2, 16)])
    def test_the_default_tile_is_the_series_tile_of_the_generation(
        self, patch_size: int | None, tile: int
    ):
        assert vae_tiling.default_tile_of(patch_size) == tile

    def test_an_unknown_generation_is_refused(self):
        with pytest.raises(ValueError, match="does not name exactly one series tile"):
            vae_tiling.default_tile_of(4)


class TestPlanAtTile32:
    def test_1280x704_is_two_rows_by_three_columns(self):
        plan = vae_tiling.plan_tiles(704 // 16, 1280 // 16, TILE, OVERLAP)

        assert OVERLAP == 4
        assert plan.rows.starts == (0, 12)
        assert plan.cols.starts == (0, 24, 48)
        assert plan.tiles == 6
        # Overlap in latents = tile − step; the blend widths are those × the graph's scale 8.
        assert plan.rows.blends(plan.scale) == [160]
        assert plan.cols.blends(plan.scale) == [64, 64]
        overlaps = [
            TILE - (later - earlier)
            for axis in (plan.rows, plan.cols)
            for earlier, later in zip(axis.starts, axis.starts[1:], strict=False)
        ]
        assert overlaps == [20, 8, 8]
        assert min(overlaps) >= OVERLAP

    def test_704x1280_is_the_transpose(self):
        plan = vae_tiling.plan_tiles(1280 // 16, 704 // 16, TILE, OVERLAP)

        assert (plan.rows.starts, plan.cols.starts) == ((0, 24, 48), (0, 12))

    def test_one_tile_exactly_is_the_degenerate_plan(self):
        assert vae_tiling.plan_tiles(TILE, TILE, TILE, OVERLAP).tiles == 1

    def test_a_latent_smaller_than_the_tile_is_refused(self):
        """832×480 is 30 latents high: a 32 graph cannot take it (the TS planner refuses it)."""
        with pytest.raises(ValueError):
            vae_tiling.plan_tiles(30, 52, TILE, OVERLAP)


class _LoadedError(Exception):
    """The CLI got as far as reading the VAE."""


class TestCliTile:
    @pytest.fixture
    def recorded(self, monkeypatch):
        calls: dict[str, list] = {"load": [], "write": []}

        def load_vae(model: str, *, round_f16: bool):
            calls["load"].append(model)
            return SimpleNamespace(use_tiling=False)

        def write_fixture(_vae, case, tile, out, *, min_overlap, patch_size):
            calls["write"].append((case.name, tile, out, min_overlap, patch_size))
            return {}

        monkeypatch.setattr(vae_tiling.export_vae, "load_vae", load_vae)
        monkeypatch.setattr(vae_tiling.export_vae, "assert_series_config", lambda *_: None)
        monkeypatch.setattr(vae_tiling, "write_fixture", write_fixture)
        return calls

    def test_tile_32_writes_the_cases_that_fit_and_names_the_skipped(self, recorded, capsys):
        import json

        assert vae_tiling.main(["--model", "ti2v-5b", "--tile", "32"]) == 0

        out = SERIES_ROOT / export_vae.TI2V_SERIES_NAME
        assert recorded["write"] == [("wide", 32, out, OVERLAP, PATCH_SIZE)]
        summary = json.loads(capsys.readouterr().out)
        assert summary["tile"] == 32
        assert [skipped["case"] for skipped in summary["skipped"]] == ["band", "accept"]

    def test_the_default_tile_skips_nothing(self, recorded, capsys):
        import json

        assert vae_tiling.main(["--model", "ti2v-5b"]) == 0

        assert [call[0] for call in recorded["write"]] == ["band", "accept", "wide"]
        assert json.loads(capsys.readouterr().out)["skipped"] == []

    @pytest.mark.parametrize(
        ("model", "tile", "message"),
        [
            ("ti2v-5b", 20, "--tile 20 is not a multiple of 16"),
            ("ti2v-5b", 24, "--tile 24 is not a multiple of 16"),
            ("t2v-1.3b", 40, "--tile 40 is not a multiple of 16"),
            ("ti2v-5b", 0, "--tile は正の整数（0）"),
            ("ti2v-5b", -16, "--tile は正の整数（-16）"),
            ("ti2v-5b", 4, "--tile 4 が重なりの下限 4（潜在）以下"),
        ],
    )
    def test_a_tile_that_is_not_a_positive_multiple_of_16_above_the_overlap_is_refused(
        self, recorded, capsys, model: str, tile: int, message: str
    ):
        with pytest.raises(SystemExit) as exited:
            vae_tiling.main(["--model", model, "--tile", str(tile)])

        assert exited.value.code == 2
        assert message in capsys.readouterr().err
        assert recorded["load"] == []

    def test_a_named_case_smaller_than_the_tile_is_refused_before_loading(self, recorded, capsys):
        with pytest.raises(SystemExit) as exited:
            vae_tiling.main(["--model", "ti2v-5b", "--tile", "32", "--case", "band"])

        assert exited.value.code == 2
        assert "--case band (latent 30×52) is smaller than --tile 32" in capsys.readouterr().err
        assert recorded["load"] == []

    def test_no_case_left_is_refused_before_loading(self, recorded, capsys):
        with pytest.raises(SystemExit) as exited:
            vae_tiling.main(["--model", "ti2v-5b", "--tile", "64"])

        assert exited.value.code == 2
        assert "no case of --model ti2v-5b is at least --tile 64" in capsys.readouterr().err
        assert recorded["load"] == []


class TestWriteFixtureAtTile32:
    """The decode is stubbed: this pins the name and the meta, not the values."""

    @pytest.fixture
    def decode_stubbed(self, monkeypatch):
        def seeded_latents(_vae, case):
            return torch.zeros(1, 48, case.chunks, case.height, case.width)

        def decoded(_vae, latents, *_plan):
            _, _, chunks, height, width = latents.shape
            return torch.full((1, 12, 1 + 4 * (chunks - 1), height * 8, width * 8), 0.5)

        monkeypatch.setattr(vae_tiling, "seeded_latents", seeded_latents)
        monkeypatch.setattr(vae_tiling, "tiled_decode_unclamped", decoded)

    @staticmethod
    def _write(tile: int, out: Path) -> tuple[Path, dict[str, str]]:
        from safetensors import safe_open

        [wide] = [case for case in vae_tiling.TI2V_FIXTURE_CASES if case.name == "wide"]
        vae = SimpleNamespace(config=SimpleNamespace(patch_size=PATCH_SIZE))
        summary = vae_tiling.write_fixture(
            vae, wide, tile, out, min_overlap=OVERLAP, patch_size=PATCH_SIZE
        )
        path = Path(summary["path"])
        with safe_open(str(path), framework="pt") as handle:
            return path, handle.metadata()

    def test_tile_32_writes_the_t32_name_with_the_tile_in_the_meta(
        self, decode_stubbed, tmp_path: Path
    ):
        path, metadata = self._write(TILE, tmp_path)

        assert path == tmp_path / "vae_tiles.wide.t32.safetensors"
        assert metadata["tile"] == "32"
        assert metadata["scale"] == "8"
        assert metadata["rows_starts"] == "0,12"
        assert metadata["cols_starts"] == "0,24,48"
        assert metadata["rows_blend"] == "160"
        assert metadata["cols_blend"] == "64,64"
        assert metadata["patch_size"] == "2"

    def test_the_series_tile_keeps_its_name_beside_the_t32_file(
        self, decode_stubbed, tmp_path: Path
    ):
        """Both tiles in one root: the 32 write leaves the 16 file alone and vice versa."""
        path16, metadata16 = self._write(TI2V_TILE, tmp_path)
        before = path16.read_bytes()
        path32, _ = self._write(TILE, tmp_path)

        assert path16 == tmp_path / "vae_tiles.wide.safetensors"
        assert metadata16["tile"] == "16"
        assert path32 != path16
        assert path16.read_bytes() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "vae_tiles.wide.safetensors",
            "vae_tiles.wide.t32.safetensors",
        ]
