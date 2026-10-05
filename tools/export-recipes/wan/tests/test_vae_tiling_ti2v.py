"""`wan/vae_tiling.py` の Wan2.2 の経路の約束事（ADR 0121 段 5 — patchify 空間のタイル参照）。

ここで固定するのは、壊れても例外が出ず**値だけが静かにずれる**側:

- 2.2 のタイル計画（タイル 16・重なり潜在 4）の開始位置が TS 側と一致すること
  （{@link MIRRORED_STARTS_TI2V} — TS 側 `packages/models/tests/wan_ti2v_vae_tiles_host_test.ts` の
  `AXIS_STARTS_TI2V` と同じ値の二重凍結）と、4 寸法の枚数・ブレンド幅
- unpatchify の並び（{@link UNPATCHIFY_SOURCE_CHANNEL} — TS 側の表と二重凍結）が上流 diffusers の
  `unpatchify` と一致すること
- 縮退（タイル 1 枚）のタイル参照が上流の非タイル chunk ループと**ビット一致**し、unpatchify →
  クランプを当てると上流の `_decode` そのものと**ビット一致**すること（参照の素性を上流に結ぶ）。
  合成の 2.2 VAE（常に走る）と実重み（pin した 5B の snapshot がある機だけ）の 2 層
- `accept` の `frames_rgb` が `frames` に上流の unpatchify → クランプを当てたものであること・メタに
  足すのが `patch_size` と `reference` の 2 つだけであること
- CLI の `--model` と `--case` の組（表に無い組は VAE を読む前に落ちる）と、モデルごとの既定値・
  読んだ VAE と系列の前提の照合
- 2.1 の `write_fixture` のメタ全体とテンソル名の組（decode は差し替える — 値は R0 の担当）

2.1 の経路（`test_vae_tiling.py`）は変えない — 2.1 のフィクスチャの不変は、書き出し直した
`vae_tiles.{band,accept}` を `wan.series_check` で既存と突き合わせて確かめる（CPU で長いので
pytest には入れない）。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from _shared.paths import SERIES_ROOT
from wan import export_vae, vae_patch, vae_tiling

#: Wan2.2 の潜在タイル（`export_vae.TI2V_TILE`）と patchify の倍率（上流 config の `patch_size`）。
TILE = 16
PATCH_SIZE = 2

#: Wan2.2 の重なりの下限（潜在）= 64 px ÷ (グラフの比 8 × unpatchify 2) = 4。
OVERLAP = vae_tiling.min_overlap_latent(export_vae.SPATIAL_SCALE, PATCH_SIZE)

#: タイル辺 16・重なりの下限 4 の開始位置（潜在の全長 → 開始位置列）。TS 側
#: （`packages/models/tests/wan_ti2v_vae_tiles_host_test.ts` の `AXIS_STARTS_TI2V`）と**同じ値**で
#: 凍結する — 両実装を機械で突き合わせる経路は GPU の照合（フィクスチャのメタ）だけなので、GPU の
#: 無い機でも鏡像のずれを捕まえる。
#:
#: - 16: 縮退（1 枚 — 縮退門の形）
#: - 30 / 52: 832×480 / 480×832 の 2 辺（潜在 = 画素 ÷ 16）
#: - 44 / 80: 1280×704 / 704×1280 の 2 辺（対ごとにブレンド幅が違う）
MIRRORED_STARTS_TI2V = {
    16: (0,),
    30: (0, 7, 14),
    44: (0, 9, 19, 28),
    52: (0, 12, 24, 36),
    80: (0, 11, 21, 32, 43, 53, 64),
}

#: 4 寸法（幅 × 高さの画素）→ (枚数, 行のブレンド幅, 列のブレンド幅)。ブレンド幅は patchify 空間の
#: 画素（= 重なりの潜在 × グラフの比 8）。
SIZES = {
    (832, 480): (12, [72, 72], [32, 32, 32]),
    (480, 832): (12, [32, 32, 32], [72, 72]),
    (1280, 704): (28, [56, 48, 56], [40, 48, 40, 40, 48, 40]),
    (704, 1280): (28, [40, 48, 40, 40, 48, 40], [56, 48, 56]),
}

#: unpatchify の並び: `[dy][dx]` → patchify 空間のチャネル内のずれ（dy = 高さ方向・dx = 幅方向の
#: 画素のずれ）。`out[c, f, y·2 + dy, x·2 + dx] = in[c·4 + 表[dy][dx], f, y, x]`
#: （ずれは dx·2 + dy）。
#: TS 側（`packages/models/tests/wan_vae_tiles_test.ts` の `UNPATCHIFY_SOURCE_CHANNEL`）と同じ値。
UNPATCHIFY_SOURCE_CHANNEL = ((0, 2), (1, 3))


def _upstream_unpatchify(frames: torch.Tensor) -> torch.Tensor:
    wan_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
    return wan_vae.unpatchify(frames, patch_size=PATCH_SIZE)


def _table_matches(table, patchified: torch.Tensor, pixels: torch.Tensor) -> bool:
    """`pixels [1, C, F, 2h, 2w]` の各ずれが `patchified [1, 4C, F, h, w]` の表の示すチャネルか。"""
    batch, channels, frames, height, width = patchified.shape
    grouped = patchified.view(batch, channels // 4, 4, frames, height, width)
    return all(
        torch.equal(pixels[:, :, :, dy::2, dx::2], grouped[:, :, table[dy][dx]])
        for dy in range(2)
        for dx in range(2)
    )


class TestPlanTi2v:
    def test_the_overlap_is_four_latents_and_the_tile_is_sixteen(self):
        """凍結表を作った値（タイル 16・重なり 4）が recipe の宣言値と同じ。"""
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
        """対照: 縦横のずれを入れ替えた表は一致しない（上の照合が取り違えを捕まえられる）。"""
        patchified = torch.arange(12 * 2 * 3 * 5, dtype=torch.float32).view(1, 12, 2, 3, 5)
        swapped = ((0, 1), (2, 3))

        assert not _table_matches(swapped, patchified, _upstream_unpatchify(patchified))

    def test_unpatchify_frames_is_the_upstream_function(self):
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
        """band = 832×480×81・accept = 480×832×9・wide = 1280×704×5（潜在は画素 ÷ 16・フレームは
        1 + 4(chunk − 1)）。"""
        cases = {case.name: case for case in vae_tiling.TI2V_FIXTURE_CASES}

        assert list(cases) == ["band", "accept", "wide"]
        band, accept, wide = cases["band"], cases["accept"], cases["wide"]
        assert (band.height, band.width, band.chunks, band.role) == (30, 52, 21, "band")
        assert (accept.height, accept.width, accept.chunks, accept.role) == (52, 30, 3, "accept")
        assert (wide.height, wide.width, wide.chunks, wide.role) == (44, 80, 2, "accept")
        # seed は設計の表の値（フィクスチャが潜在を持つので e2e の自己整合はずれても保たれる —
        # 表からのずれはここでだけ捕まる）。
        assert (band.seed, accept.seed, wide.seed) == (20261051, 20261052, 20261053)

    def test_acceptance_differs_from_the_band_in_seed_chunks_and_placement(self):
        cases = {case.name: case for case in vae_tiling.TI2V_FIXTURE_CASES}
        band = cases["band"]

        for name in ("accept", "wide"):
            case = cases[name]
            assert case.chunks != band.chunks, name
            assert (case.height, case.width) != (band.height, band.width), name

    def test_seeds_are_distinct_from_every_other_fixture(self):
        """2.1 のタイル参照・段 4 の chunk 列（2.1 / 2.2）とも潜在を共有しない。"""
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
        """RGB は patchify する世代の accept だけ（裁定 F-11）・2.1 の表は RGB を持たない。"""
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
        """2.1 のメタ `reference` は 2.2 の経路を足す前と同じ文字列（R0 のメタの一致の前提）。"""
        assert vae_tiling.FIXTURE_REFERENCE == (
            "snapped-tile decode with upstream blend_v / blend_h before clamp (CPU f32)"
        )


def _seeded(seed: int, height: int, width: int, chunks: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 48, chunks, height, width, generator=generator)


def _assert_degenerate_tile_is_upstream(vae, latents: torch.Tensor) -> None:
    """1 枚のタイル参照 ≡ 非タイル chunk ループ（クランプ前）、clamp(unpatchify) ≡ 上流 `_decode`。

    `latents` は正方（タイル辺 = 潜在の辺）。
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
        """潜在がタイル 1 枚ちょうど（8×8・3 chunk）。post-quant の 1×1×1 conv の掛け方（タイルは
        フレームごと・非タイルは全フレームに 1 度）の違いも含めて値が変わらないことを見る。"""
        _assert_degenerate_tile_is_upstream(ti2v_synthetic_vae, _seeded(11, 8, 8))

    def test_a_generation_the_chunk_graph_does_not_take_is_refused(self, ti2v_synthetic_vae):
        """残差あり・patchify 無しは chunk グラフの書き直しが受けない世代。門が無ければ decode は
        そのまま通る（形も計画も正しい）ので、例外はこの門からしか出ない。"""
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_config(ti2v_synthetic_vae.config, patch_size=None).eval()
        plan = vae_tiling.plan_tiles(8, 8, tile=8, min_overlap=OVERLAP)

        with pytest.raises(vae_patch.UnsupportedVaeError, match="未対応"):
            vae_tiling.decode_tiles(vae, _seeded(11, 8, 8, chunks=1), plan)


#: 合成の VAE で書く小さなケース（潜在 8×12・タイル 8 → 列 0 / 4 の 2 枚・2 chunk）。
SYNTHETIC_CASE = vae_tiling.FixtureCase(
    "accept", seed=3, chunks=2, height=8, width=12, role="accept", full=True, rgb=True
)


@pytest.fixture(scope="module")
def written(ti2v_synthetic_vae, tmp_path_factory):
    """合成の VAE で {@link SYNTHETIC_CASE} を書いて読み戻した (要約, テンソル, メタ)。"""
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
        """表（上流の `unpatchify(arange)` で確かめた並び）で `frames` から独立に組み直して比べ、
        上流の関数そのものとも比べる。"""
        _, tensors, _ = written
        frames, rgb = tensors["frames"], tensors["frames_rgb"]
        # 前提: クランプ前に [-1, 1] の外の要素がある（無いとクランプ抜けがこの照合で割れない）。
        assert bool((frames.abs() > 1.0).any())

        assert _table_matches(UNPATCHIFY_SOURCE_CHANNEL, frames.clamp(-1.0, 1.0)[None], rgb[None])
        assert torch.equal(rgb, _upstream_unpatchify(frames[None])[0].clamp(-1.0, 1.0))

    def test_the_metadata_adds_only_patch_size_and_the_reference(self, written):
        """統計は足さない（48 本の写しは追跡する fixture が縛る — 裁定 F-04）。"""
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
        with pytest.raises(ValueError, match="上流 config の 2 と違う"):
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

        with pytest.raises(ValueError, match="patchify する世代だけ"):
            vae_tiling.write_fixture(
                vae, SYNTHETIC_CASE, 8, tmp_path, min_overlap=OVERLAP, patch_size=None
            )

    def test_the_overlap_and_the_patch_size_are_required(self, ti2v_synthetic_vae, tmp_path: Path):
        """渡し忘れは TypeError — 2.1 の値で黙って書かない。"""
        with pytest.raises(TypeError):
            vae_tiling.write_fixture(ti2v_synthetic_vae, SYNTHETIC_CASE, 8, tmp_path)  # type: ignore[call-arg]


#: 2.1 のフィクスチャのメタ（2.2 の経路を足す前の書き出し物 `outputs/series/` の
#: `wan2.1-t2v-1.3b-f16-dyn/vae_tiles.{band,accept}.safetensors` から写した値）。
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
    """2.1 の経路（`patch_size` が None）のメタ全体とテンソル名の組を凍結する。

    decode は差し替える（値の不変は R0 — 書き出し直した実物を `wan.series_check` で既存と
    突き合わせる — の担当）。ここが捕まえるのは、2.2 の経路を足したことによるメタの退行だけ。
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
    """CLI の検査が VAE を読む段まで進んだ印（読まずに止める）。"""


class TestCli:
    @pytest.fixture
    def recorded(self, monkeypatch):
        """VAE を読まずに `main` の配線（モデル・タイル・置き場・重なり・ケース）を記録する。"""
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
        """読んだ VAE がそのモデルの系列の前提と 1 回だけ照合される（外すと世代違いを黙って
        書く）。"""
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
        """重なりの下限ちょうどのタイルは計画が受けない — VAE を読む前に落ちる。"""
        with pytest.raises(SystemExit) as exited:
            vae_tiling.main(["--model", model, "--tile", str(tile)])

        assert exited.value.code == 2
        assert f"--tile {tile} が重なりの下限 {tile}（潜在）以下" in capsys.readouterr().err
        assert recorded["load"] == []

    def test_a_case_of_another_model_is_rejected_before_loading(self, monkeypatch, capsys):
        def load_vae(*_args, **_kwargs):
            raise _LoadedError

        monkeypatch.setattr(vae_tiling.export_vae, "load_vae", load_vae)

        with pytest.raises(SystemExit) as exited:
            vae_tiling.main(["--model", "t2v-1.3b", "--case", "wide"])

        assert exited.value.code == 2
        assert "--case ['wide'] は --model t2v-1.3b の表に無い" in capsys.readouterr().err

    def test_case_choices_are_the_union_of_the_models(self, recorded):
        """`wide` は 2.2 の表にだけあるが、`--case` の choices としては受ける。"""
        assert vae_tiling.main(["--model", "ti2v-5b", "--case", "wide"]) == 0
        assert [call[0] for call in recorded["write"]] == ["wide"]

        with pytest.raises(SystemExit) as exited:
            vae_tiling.main(["--case", "nope"])
        assert exited.value.code == 2


# ---- 実重み（pin した 5B の snapshot が無ければ SKIP）---------------------------------


@pytest.fixture(scope="module")
def vae(wan22_snapshot):
    """pin した revision の 5B の VAE（CPU f32・丸め無し — 恒等の主張は重みの値に依らない）。"""
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        wan22_snapshot, subfolder="vae", torch_dtype=torch.float32
    ).eval()


class TestDegenerateTileRealWeights:
    def test_one_tile_is_the_untiled_chunk_loop_and_the_upstream_decode_bitwise(self, vae):
        """実重みでも同じ 2 点（潜在 8×8・3 chunk — 実寸のタイル 16 より小さいが、主張は形に
        依らない）。"""
        mean = torch.tensor(vae.config.latents_mean).view(1, 48, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, 48, 1, 1, 1)

        _assert_degenerate_tile_is_upstream(vae, _seeded(13, 8, 8) * std + mean)
