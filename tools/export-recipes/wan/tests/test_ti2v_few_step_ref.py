"""`wan/ti2v_few_step_ref.py`（Wan2.2 の 2 ステップの通しの CPU 参照 — ADR 0121 段 6）の約束事。

参照そのもの（5B の DiT の forward 20 回 + 28 タイルの decode × 3 ケース・RAM 約 24 GiB）は
回さない。ここで固定するのは:

- ケースの表（決定用 2 本 + 受入れ 1 本・seed もプロンプトも互いに違う・3 本とも同じ形）と、
  潜在の形・shift・guidance の定数（利用者の裁定 2026-10-05 — 1280×704・17 フレーム・shift 5.0）
- 資産のメタの `name` で正 / negative を引くこと（役割の取り違えは読み込みの前に落ちる）
- CLI（`--case` で 1 本だけ・重い読み込みの前に落ちる口）
- scheduler の `flow_shift` を明示して読むこと（config の値に依らず 5.0・式が効く形でなければ
  止まる）
- 第 2 相の合成が、統計の非自明な VAE で上流の逆正規化 + 非タイルの decode と同じ値になること
- 実行の DiT の読み手（i8 の fake-quant + 実数形 RoPE の processor）で 2 ステップが完走すること
- 合成の小さな 2.2 モデル（48 ch の DiT + 合成の 2.2 VAE）で、2 相の順（DiT を手放してから
  decode する — 手放さなければ落ちる）・M = 1 / M = S の hook・書いたテンソルとメタの欄・
  記録した cond / uncond から上流の CFG + UniPC が潜在をビット一致で作り直せること

生成は `uv run --group wan --inexact python -m wan.ti2v_few_step_ref`。
"""

from __future__ import annotations

import hashlib
import json
import weakref
from pathlib import Path
from typing import Any

import pytest
import torch

from wan import (
    dit_probe,
    export_dit,
    export_vae,
    few_step_ref,
    prompts,
    text_embeds,
    ti2v_few_step_ref,
)
from wan.tests.test_dit_patch import TINY_DIT

#: 合成の潜在 `[C, F, H, W]`（48 ch・格子 2·3·5 — 上流の高さ / 幅の倍数 32 を満たす）。
SYNTHETIC_LATENT = (48, 2, 6, 10)

#: 合成のタイル辺（潜在）。重なりの下限 4 より大きく、縦 1 枚 × 横 3 枚になる。
SYNTHETIC_TILE = 6

#: 合成の 2.2 の DiT（`TINY_DIT` の 48 ch 版 — 文脈の幅は埋め込み資産と同じ 4096）。
TINY_TI2V_DIT = {**TINY_DIT, "in_channels": 48, "out_channels": 48, "text_dim": 4096}

#: pin の `scheduler/scheduler_config.json` の値（`flow_shift` だけ 3.0 に変えてある —
#: 明示の 5.0 が config の値に勝つことを見る）。
SCHEDULER_CONFIG: dict[str, Any] = {
    "_class_name": "UniPCMultistepScheduler",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 3.0,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "bh2",
    "steps_offset": 0,
    "thresholding": False,
    "time_shift_type": "exponential",
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False,
}

CASES = ti2v_few_step_ref.FIXTURE_CASES


def _asset_metadata(negative_role: str = prompts.NEGATIVE) -> dict[str, Any]:
    entries = [{"name": case.prompt, "role": prompts.POSITIVE} for case in CASES]
    return {"prompts": [*entries, {"name": "negative", "role": negative_role}]}


def _embeds() -> dict[str, torch.Tensor]:
    """正 3 本 + negative（値は互いに違う — hook の値での振り分けが決まる）。"""
    generator = torch.Generator().manual_seed(20262199)
    lengths = {"boxing-cats": 3, "cat-dog-baking": 4, "ferret": 5, "negative": 6}
    return {name: torch.randn(rows, 4096, generator=generator) for name, rows in lengths.items()}


def _write_asset(path: Path, metadata: dict[str, Any]) -> None:
    embeds = _embeds()
    for entry in metadata["prompts"]:
        entry["tokens"] = embeds[entry["name"]].shape[0]
    text_embeds.write_asset(path, embeds, metadata)


class TestCases:
    def test_two_band_cases_and_one_accept_case_differ_in_seed_and_prompt(self):
        """決定用 2 本 + 受入れ 1 本。受入れは決定用と別の seed・プロンプト。"""
        assert [(case.name, case.role, case.seed, case.prompt) for case in CASES] == [
            ("band-boxing-cats", "band", 20262101, "boxing-cats"),
            ("band-cat-dog-baking", "band", 20262102, "cat-dog-baking"),
            ("accept-ferret", "accept", 20262103, "ferret"),
        ]
        assert len({case.seed for case in CASES}) == len(CASES)
        assert len({case.prompt for case in CASES}) == len(CASES)

    def test_the_seeds_are_not_shared_with_the_wan21_reference(self):
        assert not {case.seed for case in CASES} & {
            case.seed for case in few_step_ref.FIXTURE_CASES
        }

    def test_only_the_band_cases_observe_the_m_s_path(self):
        assert [case.observes_m_s for case in CASES] == [True, True, False]

    def test_every_case_uses_a_positive_fixed_prompt(self):
        for case in CASES:
            assert text_embeds.prompt_by_name(case.prompt).role == prompts.POSITIVE

    def test_the_shape_is_1280x704_with_17_frames_and_shift_5(self):
        """3 本とも同じ形（受入れの形を変えて帯を外させない — 裁定 2026-10-05）。"""
        channels, frames, height, width = ti2v_few_step_ref.LATENT_SHAPE

        assert channels == 48
        assert (4 * (frames - 1) + 1, 16 * height, 16 * width) == (17, 704, 1280)
        assert frames * (height // 2) * (width // 2) == 4400
        assert ti2v_few_step_ref.STEPS == 2
        assert ti2v_few_step_ref.FLOW_SHIFT == 5.0
        assert ti2v_few_step_ref.GUIDANCE_SCALE == 5.0

    def test_the_decode_uses_the_28_tile_plan_of_stage_5(self):
        series = ti2v_few_step_ref.VAE_SERIES
        _, _, height, width = ti2v_few_step_ref.LATENT_SHAPE
        plan = ti2v_few_step_ref.vae_tiling.plan_tiles(
            height,
            width,
            series.tile,
            ti2v_few_step_ref.vae_tiling.min_overlap_latent(
                export_vae.SPATIAL_SCALE, series.patch_size
            ),
        )

        assert (series.tile, series.patch_size, plan.tiles) == (16, 2, 28)

    def test_the_reference_is_written_at_the_i8_dit_series(self):
        assert ti2v_few_step_ref.SERIES.name == "wan2.2-ti2v-5b-i8-dyn"


class TestPromptNames:
    def test_it_picks_the_positive_and_the_negative_row_by_name(self):
        names = ti2v_few_step_ref.prompt_names(_asset_metadata(), CASES[2])

        assert (names.positive, names.negative) == ("ferret", "negative")

    def test_a_negative_row_with_the_wrong_role_is_refused(self):
        with pytest.raises(ValueError, match="'negative' の役割"):
            ti2v_few_step_ref.prompt_names(_asset_metadata(prompts.POSITIVE), CASES[0])

    def test_a_missing_row_is_refused(self):
        metadata = {"prompts": [{"name": "negative", "role": prompts.NEGATIVE}]}

        with pytest.raises(ValueError, match="'boxing-cats' の行が無い"):
            ti2v_few_step_ref.prompt_names(metadata, CASES[0])


class TestCli:
    @pytest.fixture
    def heavy(self, monkeypatch, tmp_path):
        """重い段（RAM の確認・読み込み・2 相）を記録だけの代役に差し替える。

        代役が呼ばれた順は `calls["order"]` に積む（何も呼ばれなければ `calls` は空のまま）。
        """
        calls: dict[str, Any] = {}

        def called(name: str, value: Any) -> Any:
            calls.setdefault("order", []).append(name)
            return value

        monkeypatch.setattr(ti2v_few_step_ref, "memory_estimate", lambda _model: 123)
        monkeypatch.setattr(
            dit_probe,
            "require_available",
            lambda stage, need: calls.update(memory=called("memory", (stage, need))),
        )
        monkeypatch.setattr(
            ti2v_few_step_ref, "load_rounded_vae", lambda _model: called("vae", "vae")
        )
        monkeypatch.setattr(
            ti2v_few_step_ref, "load_scheduler", lambda _snapshot: called("scheduler", "scheduler")
        )
        monkeypatch.setattr(ti2v_few_step_ref, "local_snapshot", lambda _model: tmp_path)
        monkeypatch.setattr(
            ti2v_few_step_ref, "load_quantized_transformer", lambda _model: called("dit", "dit")
        )

        def generate(cases, **kwargs):
            called("generate", None)
            calls.update(cases=[case.name for case in cases], **kwargs)
            # main が渡す DiT の読み手が i8 の読み手へ届くこと（DiT は第 1 相の中で読む）。
            calls["dit"] = kwargs["load_transformer"]()
            return []

        monkeypatch.setattr(ti2v_few_step_ref, "generate", generate)
        return calls

    def test_help_loads_nothing(self, heavy, capsys):
        with pytest.raises(SystemExit) as raised:
            ti2v_few_step_ref.main(["--help"])

        assert raised.value.code == 0
        assert heavy == {}
        assert "--case" in capsys.readouterr().out

    def test_case_selects_one_case_and_checks_memory_before_loading(self, heavy, tmp_path):
        embeds = tmp_path / "text_embeds.safetensors"
        _write_asset(embeds, _asset_metadata())

        assert ti2v_few_step_ref.main(["--embeds", str(embeds), "--case", "accept-ferret"]) == 0

        assert heavy["cases"] == ["accept-ferret"]
        assert heavy["order"] == ["memory", "vae", "scheduler", "generate", "dit"]
        assert heavy["dit"] == "dit"
        assert heavy["memory"] == ("load", 123)
        assert (heavy["vae"], heavy["scheduler"]) == ("vae", "scheduler")
        assert heavy["tile"] == 16
        assert heavy["out_root"] == ti2v_few_step_ref.SERIES
        assert heavy["names"]["accept-ferret"].negative == "negative"
        assert (
            heavy["common"]["text_embeds_sha256"] == hashlib.sha256(embeds.read_bytes()).hexdigest()
        )
        assert heavy["common"]["source"].startswith("Wan-AI/Wan2.2-TI2V-5B-Diffusers@")

    def test_an_unknown_case_is_refused(self, heavy):
        with pytest.raises(SystemExit) as raised:
            ti2v_few_step_ref.main(["--case", "accept-otter"])

        assert raised.value.code == 2
        assert heavy == {}

    def test_a_missing_asset_is_refused_before_loading(self, heavy, tmp_path):
        with pytest.raises(SystemExit) as raised:
            ti2v_few_step_ref.main(["--embeds", str(tmp_path / "missing.safetensors")])

        assert raised.value.code == 2
        assert heavy == {}

    def test_a_wrong_role_in_the_asset_is_refused_before_the_memory_check(self, heavy, tmp_path):
        embeds = tmp_path / "text_embeds.safetensors"
        _write_asset(embeds, _asset_metadata(prompts.POSITIVE))

        with pytest.raises(ValueError, match="役割"):
            ti2v_few_step_ref.main(["--embeds", str(embeds)])

        assert heavy == {}


@pytest.fixture
def snapshot(tmp_path) -> Path:
    """scheduler の config だけを持つ snapshot（`flow_shift` は 3.0）。"""
    directory = tmp_path / "snapshot" / "scheduler"
    directory.mkdir(parents=True)
    (directory / "scheduler_config.json").write_text(json.dumps(SCHEDULER_CONFIG))
    return directory.parent


class TestScheduler:
    def test_the_explicit_shift_wins_over_the_config(self, snapshot):
        pytest.importorskip("diffusers")

        scheduler = ti2v_few_step_ref.load_scheduler(snapshot)

        assert scheduler.config.flow_shift == 5.0
        assert scheduler.config.use_dynamic_shifting is False

    def test_a_loader_that_drops_the_shift_is_refused(self, snapshot, monkeypatch):
        diffusers = pytest.importorskip("diffusers")
        original = diffusers.UniPCMultistepScheduler.from_pretrained

        def ignoring(path, subfolder, **_kwargs):
            return original(path, subfolder=subfolder)

        monkeypatch.setattr(diffusers.UniPCMultistepScheduler, "from_pretrained", ignoring)

        with pytest.raises(AssertionError, match=r"flow_shift 3\.0 が 5\.0 でない"):
            ti2v_few_step_ref.load_scheduler(snapshot)

    @pytest.mark.parametrize(
        "override", [{"use_dynamic_shifting": True}, {"use_flow_sigmas": False}]
    )
    def test_a_config_where_the_shift_does_not_apply_is_refused(self, tmp_path, override):
        """動的 shift・flow でない σ では `flow_shift` が σ に効かない — 5.0 を読めても止まる。"""
        pytest.importorskip("diffusers")
        directory = tmp_path / "snapshot" / "scheduler"
        directory.mkdir(parents=True)
        (directory / "scheduler_config.json").write_text(
            json.dumps({**SCHEDULER_CONFIG, **override})
        )

        with pytest.raises(AssertionError, match="flow_shift の式が効く形"):
            ti2v_few_step_ref.load_scheduler(directory.parent)


class TestMemoryEstimate:
    def test_the_gate_is_not_below_the_documented_peak(self, wan22_snapshot):
        """読み込み前の門（見込み + 余白）が doc / README の第 1 相の山（約 24 GiB）を
        下回らない。"""
        estimate = ti2v_few_step_ref.memory_estimate()

        assert estimate + dit_probe.MEMORY_HEADROOM_BYTES >= 24 * 2**30


def _tiny_dit() -> torch.nn.Module:
    transformer_wan = pytest.importorskip("diffusers.models.transformers.transformer_wan")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20262198)
        return transformer_wan.WanTransformer3DModel(**TINY_TI2V_DIT).to(torch.float32).eval()


class TestSyntheticRun:
    """合成の 2.2 モデルで 2 相を通して書く（5B の重みは読まない）。"""

    @pytest.fixture
    def run(self, monkeypatch, tmp_path, snapshot, ti2v_synthetic_vae):
        monkeypatch.setattr(ti2v_few_step_ref, "LATENT_SHAPE", SYNTHETIC_LATENT)
        scheduler = ti2v_few_step_ref.load_scheduler(snapshot)
        refs: list[weakref.ref] = []
        decodes: list[bool] = []

        def load() -> torch.nn.Module:
            model = _tiny_dit()
            refs.append(weakref.ref(model))
            return model

        decode_tiles = ti2v_few_step_ref.vae_tiling.decode_tiles

        def spying(vae, latents, plan):
            # decode の時点で DiT が回収済みか（2 相の順）。
            decodes.append(refs[0]() is None)
            return decode_tiles(vae, latents, plan)

        monkeypatch.setattr(ti2v_few_step_ref.vae_tiling, "decode_tiles", spying)
        embeds = _embeds()
        names = {
            case.name: ti2v_few_step_ref.prompt_names(_asset_metadata(), case) for case in CASES
        }
        summaries = ti2v_few_step_ref.generate(
            [CASES[0], CASES[2]],
            load_transformer=load,
            vae=ti2v_synthetic_vae,
            scheduler=scheduler,
            embeds=embeds,
            names=names,
            tile=SYNTHETIC_TILE,
            out_root=tmp_path / "series",
            common=ti2v_few_step_ref.common_metadata("0" * 64),
        )
        written = {}
        for summary in summaries:
            from safetensors import safe_open

            with safe_open(summary["path"], framework="pt") as handle:
                tensors = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
                written[summary["case"]] = (tensors, handle.metadata())
        return {
            "summaries": summaries,
            "written": written,
            "decodes": decodes,
            "scheduler": scheduler,
        }

    def test_the_dit_is_released_before_the_first_decode(self, run):
        """第 1 相（全ケースの DiT）→ DiT を手放す → 第 2 相（decode）。"""
        assert run["decodes"] == [True, True]

    def test_the_tensors_have_the_documented_names_and_shapes(self, run):
        latent_names = {"latents_init"} | {
            f"{kind}.{index}"
            for kind in ("noise_cond", "noise_uncond", "latents")
            for index in (0, 1)
        }
        observed = {f"observed_m_s.latents.{index}" for index in (0, 1)}
        stats = {"latents_mean", "latents_std"}

        band, _ = run["written"]["band-boxing-cats"]
        accept, _ = run["written"]["accept-ferret"]

        assert set(band) == latent_names | observed | stats | {"frames"}
        assert set(accept) == latent_names | stats | {"frames"}
        for name in latent_names | observed:
            assert band[name].shape == SYNTHETIC_LATENT, name
            assert band[name].dtype == torch.float32, name
        assert band["frames"].shape == (3, 5, 96, 160)
        assert float(band["frames"].abs().max()) <= 1.0
        assert band["latents_mean"].shape == band["latents_std"].shape == (48,)

    def test_the_noise_is_the_seeded_torch_randn(self, run):
        tensors, _ = run["written"]["accept-ferret"]
        generator = torch.Generator().manual_seed(CASES[2].seed)

        assert torch.equal(
            tensors["latents_init"], torch.randn(1, *SYNTHETIC_LATENT, generator=generator)[0]
        )

    def test_upstream_cfg_and_unipc_rebuild_the_latents_from_the_recorded_outputs(self, run):
        """cond / uncond の取り違え・step の記録のずれは、ここでビットが合わなくなる。"""
        from diffusers import UniPCMultistepScheduler

        for case, (tensors, _) in run["written"].items():
            scheduler = UniPCMultistepScheduler.from_config(run["scheduler"].config)
            scheduler.set_timesteps(ti2v_few_step_ref.STEPS)
            scheduler.set_begin_index(0)
            latents = tensors["latents_init"][None]
            for index, timestep in enumerate(scheduler.timesteps):
                uncond = tensors[f"noise_uncond.{index}"][None]
                cond = tensors[f"noise_cond.{index}"][None]
                noise = uncond + ti2v_few_step_ref.GUIDANCE_SCALE * (cond - uncond)
                latents = scheduler.step(noise, timestep, latents, return_dict=False)[0]

                assert torch.equal(latents[0], tensors[f"latents.{index}"]), (case, index)

    def test_the_metadata_carries_the_documented_fields(self, run):
        band_tensors, band = run["written"]["band-boxing-cats"]
        _, accept = run["written"]["accept-ferret"]
        fields = {
            "role", "seed", "prompt", "negative", "steps", "guidance_scale", "flow_shift",
            "expand_timesteps", "timesteps", "sigmas", "weights", "reference", "tile", "scale",
            "rows_starts", "cols_starts", "rows_blend", "cols_blend", "patch_size",
            "text_embeds_sha256", "source", "environment",
        }  # fmt: skip

        assert set(accept) == fields
        assert set(band) == fields | {"observed_m_s"}
        assert (band["role"], band["seed"], band["prompt"], band["negative"]) == (
            "band",
            "20262101",
            "boxing-cats",
            "negative",
        )
        assert (band["steps"], band["guidance_scale"], band["flow_shift"]) == ("2", "5.0", "5.0")
        assert band["expand_timesteps"] == "false"
        assert band["weights"] == "dit-i8-rtn-fake-quant+vae-f16-rounded"
        assert (band["tile"], band["patch_size"], band["cols_starts"]) == ("6", "2", "0,2,4")
        timesteps = json.loads(band["timesteps"])
        assert len(timesteps) == 2 and timesteps[0] == 999
        assert len(json.loads(band["sigmas"])) == 3
        environment = json.loads(band["environment"])
        assert environment["threads"] == torch.get_num_threads()
        assert band_tensors["latents_mean"].tolist() == [0.0] * 48

    def test_the_summary_reports_the_m_s_observation_for_the_band_cases_only(self, run):
        band, accept = run["summaries"]

        assert len(band["observed_m_s_vs_m_1"]) == 2
        assert "observed_m_s_vs_m_1" not in accept
        assert band["tiles"] == 3


def _read_fixture(path: str) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118


class TestDecodeFrames:
    """第 2 相の合成（逆正規化 → タイル decode → unpatchify → clamp）が上流の decode と同じ値に
    なること。共有の合成 VAE は mean 0・std 1 で逆正規化が恒等なので、統計の非自明な VAE で見る。"""

    def test_a_single_tile_decode_matches_the_upstream_decode_of_the_denormalized_latents(
        self, snapshot, ti2v_synthetic_vae
    ):
        from diffusers import AutoencoderKLWan

        mean = [0.1 * index for index in range(48)]
        std = [0.5 + 0.01 * index for index in range(48)]
        vae = AutoencoderKLWan.from_config(
            ti2v_synthetic_vae.config, latents_mean=mean, latents_std=std
        ).eval()
        vae.load_state_dict(ti2v_synthetic_vae.state_dict())
        pipeline = ti2v_few_step_ref.build_pipeline(
            None, vae, ti2v_few_step_ref.load_scheduler(snapshot), expand_timesteps=False
        )
        latents = torch.randn(1, 48, 2, 6, 6, generator=torch.Generator().manual_seed(20262197))
        vae_tiling = ti2v_few_step_ref.vae_tiling
        plan = vae_tiling.plan_tiles(
            6, 6, 6, vae_tiling.min_overlap_latent(export_vae.SPATIAL_SCALE, 2)
        )

        got = ti2v_few_step_ref.decode_frames(pipeline, latents, plan, 2)
        # 上流 `WanPipeline.__call__` の逆正規化（`latents / (1 / std) + mean`）→ 非タイルの decode
        # （unpatchify とクランプ込み）。
        mean_view = torch.tensor(mean).view(1, 48, 1, 1, 1)
        std_view = torch.tensor(std).view(1, 48, 1, 1, 1)
        with torch.no_grad():
            want = vae.decode(latents / (1.0 / std_view) + mean_view, return_dict=False)[0][0]
            skipped = vae.decode(latents, return_dict=False)[0][0]

        assert plan.tiles == 1
        assert got.shape == want.shape == (3, 5, 96, 96)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-6)
        # 逆正規化を飛ばした値とは区別がつく（この比較が恒等の逆正規化で素通りしない）。
        assert float((skipped - want).abs().max()) > 1e-3


class TestQuantizedDit:
    """実行が使う DiT の読み手（f32 → i8 の fake-quant と実数形 RoPE の processor）を、合成の
    48 ch の DiT で `WanPipeline` に通す（5B の重みは読まない）。"""

    @pytest.fixture
    def tiny_upstream(self, monkeypatch):
        monkeypatch.setattr(
            ti2v_few_step_ref.export_dit, "load_transformer", lambda _model: _tiny_dit()
        )

    def test_the_loaded_dit_is_on_the_i8_grid_with_the_real_pair_processors(self, tiny_upstream):
        # 乱数初期化の素の重みは i8 の格子に無い（下の空の判定が素通りでないこと）。
        assert export_dit.unrounded_weights("i8", _tiny_dit())

        model = ti2v_few_step_ref.load_quantized_transformer()

        assert export_dit.unrounded_weights("i8", model) == []
        assert {type(block.attn1.processor).__name__ for block in model.blocks} == {
            "WanRealPairAttnProcessor"
        }

    def test_the_two_step_run_completes_on_the_quantized_dit(
        self, tiny_upstream, monkeypatch, tmp_path, snapshot, ti2v_synthetic_vae
    ):
        monkeypatch.setattr(ti2v_few_step_ref, "LATENT_SHAPE", SYNTHETIC_LATENT)
        scheduler = ti2v_few_step_ref.load_scheduler(snapshot)
        names = {CASES[2].name: ti2v_few_step_ref.prompt_names(_asset_metadata(), CASES[2])}

        def run(load_transformer, directory: str) -> dict[str, torch.Tensor]:
            [summary] = ti2v_few_step_ref.generate(
                [CASES[2]],
                load_transformer=load_transformer,
                vae=ti2v_synthetic_vae,
                scheduler=scheduler,
                embeds=_embeds(),
                names=names,
                tile=SYNTHETIC_TILE,
                out_root=tmp_path / directory,
                common=ti2v_few_step_ref.common_metadata("0" * 64),
            )
            return _read_fixture(summary["path"])

        quantized = run(ti2v_few_step_ref.load_quantized_transformer, "quantized")
        unrounded = run(_tiny_dit, "unrounded")

        for index in range(ti2v_few_step_ref.STEPS):
            assert quantized[f"latents.{index}"].shape == SYNTHETIC_LATENT
        # 丸めた重みが pipeline の forward に届いている（素の重みと出力が違う）。
        assert not torch.equal(quantized["noise_cond.0"], unrounded["noise_cond.0"])


class TestFaults:
    """順と経路の検査が落ちうること（故障注入）。"""

    @pytest.fixture
    def parts(self, monkeypatch, snapshot, ti2v_synthetic_vae):
        monkeypatch.setattr(ti2v_few_step_ref, "LATENT_SHAPE", SYNTHETIC_LATENT)
        return ti2v_synthetic_vae, ti2v_few_step_ref.load_scheduler(snapshot)

    def test_a_dit_that_is_still_referenced_stops_before_the_decode(self, parts, tmp_path):
        vae, scheduler = parts
        kept = _tiny_dit()

        with pytest.raises(AssertionError, match="DiT への参照が残っている"):
            ti2v_few_step_ref.generate(
                [CASES[2]],
                load_transformer=lambda: kept,
                vae=vae,
                scheduler=scheduler,
                embeds=_embeds(),
                names={CASES[2].name: ti2v_few_step_ref.prompt_names(_asset_metadata(), CASES[2])},
                tile=SYNTHETIC_TILE,
                out_root=tmp_path / "series",
                common={},
            )
        assert not (tmp_path / "series").exists()

    def test_the_m_1_hook_refuses_a_per_token_timestep(self, parts):
        """M = S の経路を M = 1 の記録に回すと落ちる（経路の取り違え）。"""
        vae, scheduler = parts
        pipeline = ti2v_few_step_ref.build_pipeline(
            _tiny_dit(), vae, scheduler, expand_timesteps=True
        )
        names = ti2v_few_step_ref.prompt_names(_asset_metadata(), CASES[2])

        with pytest.raises(AssertionError, match="1 次元でない"):
            ti2v_few_step_ref.denoise_case(pipeline, CASES[2], _embeds(), names)

    def test_the_m_s_hook_refuses_a_scalar_timestep(self, parts):
        """M = 1 の経路を M = S の観測に回すと落ちる（別の hook が形を見る）。"""
        vae, scheduler = parts
        pipeline = ti2v_few_step_ref.build_pipeline(
            _tiny_dit(), vae, scheduler, expand_timesteps=False
        )
        names = ti2v_few_step_ref.prompt_names(_asset_metadata(), CASES[0])
        denoised = ti2v_few_step_ref.denoise_case(pipeline, CASES[0], _embeds(), names)

        with pytest.raises(AssertionError, match="M = S の経路の timestep の形"):
            ti2v_few_step_ref.observe_m_s(pipeline, denoised, _embeds())

    def test_latents_whose_channels_differ_from_the_dit_are_refused(self, parts):
        """上流は注入した潜在の形を見ない — 16 ch の潜在が黙って流れないこと。"""
        vae, scheduler = parts
        pipeline = ti2v_few_step_ref.build_pipeline(
            _tiny_dit(), vae, scheduler, expand_timesteps=False
        )

        with pytest.raises(ValueError, match="z_dim 48 と違う"):
            ti2v_few_step_ref.video_size(pipeline, (16, 2, 6, 10))
