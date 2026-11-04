"""The contract of `wan/ti2v_i2v_few_step_ref.py` (the CPU reference of a full 2-step Wan2.2 I2V run
— ADR 0121 step 9b).

The reference itself (20 forwards of the 5B DiT + a 28-tile decode × 3 cases) is not run. What is
pinned here:

- the case table (2 decision cases + 1 acceptance case; roles and images assigned the same way as
  the encoder goldens of step 9a; prompts named after the images) and the latent shape (1280×704,
  9 frames, S = 2,640, P = 880)
- **the replacement forms**: the final output of the official Wan2.2 form (replacing the scheduler
  state at the start and after each step) and of the diffusers form (replacing only the DiT input,
  once after the loop) are bit-identical (with the pinned UniPC, a pseudo-model that mixes across
  elements, and 2 / 8 / 50 steps), and that comparison tells faults apart
- the time replacement (decision 4 — M = 1 per value) is bit-identical to the DiT reference
  wrapper and is restored afterwards
- the export run on a small synthetic 2.2 model (48-ch DiT + synthetic 2.2 VAE + synthetic
  images): the order of the 2 phases, the fields of the tensors and metadata, upstream CFG + UniPC
  rebuilding the latents from the recorded cond / uncond, the condition latent equalling the
  encoder reference of step 9a, and the M = S observation
- the CLI (`--case` for a single case; the paths that fail before the heavy loading) and the
  provenance check of the condition images

Generate with `uv run --group wan --inexact python -m wan.ti2v_i2v_few_step_ref`.
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
    dit_patch,
    dit_probe,
    export_vae_encoder,
    few_step_ref,
    i2v_preprocess_ref,
    prompts,
    text_embeds,
    ti2v_few_step_ref,
    ti2v_i2v_few_step_ref,
    vae_encoder_patch,
)
from wan.tests.test_dit_patch import TINY_DIT
from wan.tests.test_ti2v_few_step_ref import SCHEDULER_CONFIG, SYNTHETIC_TILE, TINY_TI2V_DIT

REF = ti2v_i2v_few_step_ref
CASES = REF.FIXTURE_CASES

#: The synthetic latent `[C, F, H, W]` (48 ch, 9 frames, 96×160 — grid 3·3·5, P = 15).
SYNTHETIC_LATENT = (48, 3, 6, 10)

#: The original size of the synthetic condition image (width, height). It is cropped horizontally
#: to 96×160 (so the crop path is taken).
SYNTHETIC_SOURCE = (240, 120)

#: The values of the pinned `model_index.json` (without `_diffusers_version` — the fields
#: `from_pretrained` reads).
MODEL_INDEX: dict[str, Any] = {
    "_class_name": "WanPipeline",
    "boundary_ratio": None,
    "expand_timesteps": True,
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "transformer_2": [None, None],
    "vae": ["diffusers", "AutoencoderKLWan"],
}


def _asset_metadata() -> dict[str, Any]:
    entries = [{"name": case.prompt, "role": prompts.POSITIVE} for case in CASES]
    return {"prompts": [*entries, {"name": "negative", "role": prompts.NEGATIVE}]}


def _embeds() -> dict[str, torch.Tensor]:
    """3 positives + the negative (all different values — so the hook values decide the sorting)."""
    generator = torch.Generator().manual_seed(20262189)
    lengths = {"boxing-cats": 3, "cat-dog-baking": 4, "ferret": 5, "negative": 6}
    return {name: torch.randn(rows, 4096, generator=generator) for name, rows in lengths.items()}


def _tiny_dit() -> torch.nn.Module:
    transformer_wan = pytest.importorskip("diffusers.models.transformers.transformer_wan")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20262188)
        return transformer_wan.WanTransformer3DModel(**TINY_TI2V_DIT).to(torch.float32).eval()


def _source(name: str = "boxing-cats") -> REF.SourceImage:
    """A synthetic condition image (random RGB8; the sha256 is that of the raw pixel bytes — it only
    fills the provenance fields)."""
    image_module = pytest.importorskip("PIL.Image")
    width, height = SYNTHETIC_SOURCE
    generator = torch.Generator().manual_seed(20262187)
    pixels = torch.randint(0, 256, (height, width, 3), generator=generator, dtype=torch.uint8)
    digest = hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
    return REF.SourceImage(name, digest, image_module.fromarray(pixels.numpy(), mode="RGB"))


@pytest.fixture
def snapshot(tmp_path) -> Path:
    """A snapshot with only `model_index.json` and the scheduler config (`flow_shift` is 3.0)."""
    root = tmp_path / "snapshot"
    (root / "scheduler").mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps(MODEL_INDEX))
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(SCHEDULER_CONFIG))
    return root


class TestCases:
    def test_two_band_cases_and_one_accept_case_with_the_image_of_the_same_name(self):
        assert [(case.name, case.role, case.seed, case.prompt, case.image) for case in CASES] == [
            ("band-boxing-cats", "band", 20262111, "boxing-cats", "boxing-cats"),
            ("band-cat-dog-baking", "band", 20262112, "cat-dog-baking", "cat-dog-baking"),
            ("accept-ferret", "accept", 20262113, "ferret", "ferret"),
        ]
        assert len({case.seed for case in CASES}) == len(CASES)

    def test_the_seeds_are_not_shared_with_the_t2v_and_the_wan21_references(self):
        others = {case.seed for case in ti2v_few_step_ref.FIXTURE_CASES}
        others |= {case.seed for case in few_step_ref.FIXTURE_CASES}

        assert not {case.seed for case in CASES} & others

    def test_the_roles_follow_the_encoder_golden_of_stage_9a(self):
        """The decision and acceptance images are assigned the same way as the encoder goldens of
        step 9a."""
        assert tuple(case.image for case in CASES if case.role == "band") == (
            export_vae_encoder.BAND_IMAGES
        )
        assert tuple(case.image for case in CASES if case.role == "accept") == (
            export_vae_encoder.ACCEPT_IMAGES
        )
        assert [case.observes_m_s for case in CASES] == [True, True, False]

    def test_every_case_uses_a_positive_fixed_prompt(self):
        for case in CASES:
            assert text_embeds.prompt_by_name(case.prompt).role == prompts.POSITIVE

    def test_the_shape_is_1280x704_with_9_frames_and_880_condition_tokens(self):
        channels, frames, height, width = REF.LATENT_SHAPE

        assert channels == 48
        assert (4 * (frames - 1) + 1, 16 * height, 16 * width) == (9, 704, 1280)
        assert frames * (height // 2) * (width // 2) == 2640
        assert (height // 2) * (width // 2) == 880
        assert (REF.STEPS, REF.FLOW_SHIFT, REF.GUIDANCE_SCALE) == (2, 5.0, 5.0)
        assert REF.FIT == "crop"

    def test_the_reference_is_written_next_to_the_t2v_reference(self):
        path = REF.fixture_path(REF.SERIES, CASES[0])

        assert REF.SERIES.name == "wan2.2-ti2v-5b-i8-dyn"
        assert path.name == "pipeline_steps_i2v.band-boxing-cats.safetensors"
        # Does not mix with the spelling of the T2V references (`pipeline_steps.*`).
        assert not path.name.startswith(few_step_ref.FIXTURE_PREFIX)


def _pseudo_velocity(_index: int, timestep: torch.Tensor, model_input: torch.Tensor):
    """A pseudo-model that mixes across elements (rolls along the frame axis and mixes in the mean
    over channels and frames) — the values of the first frame affect the velocity of the other
    frames."""
    mixed = model_input.mean(dim=(1, 2), keepdim=True)
    return torch.tanh(0.7 * model_input.roll(1, dims=2) + 0.3 * mixed) + torch.sin(
        timestep.to(torch.float32) / 1000
    )


class TestReplacementForms:
    """Bit identity of the final outputs of the official form and the diffusers form
    ({@link REF.sample}) (with the pinned UniPC)."""

    @pytest.fixture
    def parts(self, snapshot):
        pytest.importorskip("diffusers")
        config = ti2v_few_step_ref.load_scheduler(snapshot).config
        generator = torch.Generator().manual_seed(20262186)
        shape = (1, 48, 3, 4, 6)
        latents = torch.randn(*shape, generator=generator)
        condition = torch.randn(1, 48, 1, 4, 6, generator=generator)
        return config, latents, condition, REF.first_frame_mask(shape[1:])

    def _run(self, parts, steps: int, form: str, mask: torch.Tensor | None = None):
        config, latents, condition, default_mask = parts
        return REF.sample(
            REF.fresh_scheduler(config, steps),
            latents,
            condition,
            default_mask if mask is None else mask,
            _pseudo_velocity,
            form=form,
        )

    @pytest.mark.parametrize("steps", [2, 8, 50])
    def test_the_two_forms_end_bit_identical(self, parts, steps):
        upstream = self._run(parts, steps, "diffusers")
        official = self._run(parts, steps, "official")

        assert torch.equal(upstream.final, official.final)
        for got, want in zip(official.model_inputs, upstream.model_inputs, strict=True):
            assert torch.equal(got, want)

    @pytest.mark.parametrize("steps", [2, 8])
    def test_the_scheduler_states_differ_only_in_the_first_frame(self, parts, steps):
        """Not vacuous: the intermediate states really split in the first frame and agree in the
        remaining frames."""
        upstream = self._run(parts, steps, "diffusers")
        official = self._run(parts, steps, "official")

        for got, want in zip(official.states, upstream.states, strict=True):
            assert torch.equal(got[:, :, 1:], want[:, :, 1:])
            assert not torch.equal(got[:, :, :1], want[:, :, :1])

    def test_the_first_frame_of_the_final_latents_is_the_condition(self, parts):
        _, _, condition, _ = parts
        upstream = self._run(parts, 2, "diffusers")

        assert torch.equal(upstream.final[:, :, :1], condition)
        assert torch.equal(upstream.final[:, :, 1:], upstream.states[-1][:, :, 1:])

    def test_skipping_the_last_replacement_is_caught(self, parts):
        """Fault injection: dropping the replacement after the loop splits from the official form
        in the first frame."""
        upstream = self._run(parts, 2, "diffusers")
        official = self._run(parts, 2, "official")

        assert not torch.equal(upstream.states[-1], official.final)

    def test_a_mask_on_the_wrong_frame_is_caught_beyond_that_frame(self, parts):
        """Fault injection: moving the official form's replacement to the second frame changes the
        DiT input, and because the pseudo-model mixes across frames, frames other than the first
        split too."""
        _, latents, _, _ = parts
        shifted = torch.ones(1, 1, *latents.shape[2:])
        shifted[:, :, 1] = 0
        upstream = self._run(parts, 2, "diffusers")
        official = self._run(parts, 2, "official", shifted)

        assert not torch.equal(upstream.final[:, :, 2:], official.final[:, :, 2:])

    def test_an_unknown_form_is_refused(self, parts):
        with pytest.raises(ValueError, match="unknown replacement form"):
            self._run(parts, 2, "every-other-step")


def _tiny_three_channel_dit() -> torch.nn.Module:
    transformer_wan = pytest.importorskip("diffusers.models.transformers.transformer_wan")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20262185)
        return transformer_wan.WanTransformer3DModel(**TINY_DIT).to(torch.float32).eval()


class TestTokenwiseTimeEmbedding:
    """The time replacement applied to each forward of the pipeline (decision 4 — M = 1 per
    value)."""

    LATENT = (1, 3, 2, 6, 10)

    @pytest.fixture
    def inputs(self):
        model = _tiny_three_channel_dit()
        generator = torch.Generator().manual_seed(20262184)
        latents = torch.randn(*self.LATENT, generator=generator)
        embeds = torch.randn(1, 5, TINY_DIT["text_dim"], generator=generator)
        mask = REF.condition_token_mask(model, self.LATENT[1:])
        return model, latents, embeds, mask

    def _upstream_timestep(self, mask: torch.Tensor, value: int) -> torch.Tensor:
        """The same f32 `[1, S]` as the `expand_timesteps` branch of diffusers (0 for the condition
        tokens)."""
        return torch.where(mask, 0.0, float(value)).to(torch.float32)

    def test_it_matches_the_reference_wrapper_of_the_dit_golden(self, inputs):
        model, latents, embeds, mask = inputs
        condition = dit_patch.TimestepCondition(torch.tensor([0]), mask)
        with torch.no_grad():
            want = dit_patch.reference_dit(model, latents, torch.tensor([750]), embeds, condition)
            with (
                dit_patch.flash_attention_only(),
                REF.tokenwise_time_embedding(model, mask) as seen,
            ):
                got = model(
                    hidden_states=latents,
                    timestep=self._upstream_timestep(mask, 750),
                    encoder_hidden_states=embeds,
                    return_dict=False,
                )[0]

        assert seen == [750]
        assert torch.equal(got, want)

    def test_the_time_mlp_runs_one_row_per_value(self, inputs):
        """Inside the replacement the time MLP runs one row twice; outside it (diffusers itself),
        S rows."""
        model, latents, embeds, mask = inputs
        rows: list[int] = []
        handle = model.condition_embedder.time_embedder.linear_1.register_forward_hook(
            lambda _m, args, _o: rows.append(args[0].numel() // int(args[0].shape[-1]))
        )
        timestep = self._upstream_timestep(mask, 750)
        try:
            with torch.no_grad(), dit_patch.flash_attention_only():
                with REF.tokenwise_time_embedding(model, mask):
                    model(hidden_states=latents, timestep=timestep, encoder_hidden_states=embeds)
                model(hidden_states=latents, timestep=timestep, encoder_hidden_states=embeds)
        finally:
            handle.remove()

        assert rows == [1, 1, mask.shape[1]]

    def _forward_replaced(self, inputs, timestep: torch.Tensor) -> None:
        model, latents, embeds, mask = inputs
        with torch.no_grad(), REF.tokenwise_time_embedding(model, mask):
            model(hidden_states=latents, timestep=timestep, encoder_hidden_states=embeds)

    def test_the_original_embedder_is_restored_even_on_failure(self, inputs):
        original = inputs[0].condition_embedder

        with pytest.raises(AssertionError):
            self._forward_replaced(inputs, torch.tensor([750]))

        assert inputs[0].condition_embedder is original

    def test_a_one_dimensional_timestep_is_refused(self, inputs):
        """If the T2V path (a one-dimensional timestep) reaches this replacement, it stops."""
        with pytest.raises(AssertionError, match="are not the per-token"):
            self._forward_replaced(inputs, torch.tensor([750]))

    def test_two_generation_values_are_refused(self, inputs):
        timestep = self._upstream_timestep(inputs[3], 750)
        timestep[0, -1] = 500.0

        with pytest.raises(AssertionError, match="takes 2 values"):
            self._forward_replaced(inputs, timestep)

    def test_a_nonzero_condition_token_is_refused(self, inputs):
        """If the time of a condition token is not 0, the reference wrapper stops it (a shifted
        mapping to the mask)."""
        timestep = self._upstream_timestep(inputs[3], 750)
        timestep[0, 0] = 750.0

        with pytest.raises(AssertionError, match="does not match the condition mask"):
            self._forward_replaced(inputs, timestep)


def _read_fixture(path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}, handle.metadata()  # noqa: SIM118


def _generate(monkeypatch, tmp_path, snapshot, vae, cases, load, **overrides):
    monkeypatch.setattr(REF, "LATENT_SHAPE", SYNTHETIC_LATENT)
    scheduler = ti2v_few_step_ref.load_scheduler(snapshot)
    kwargs: dict[str, Any] = {
        "load_transformer": load,
        "snapshot": snapshot,
        "vae": vae,
        "scheduler": scheduler,
        "embeds": _embeds(),
        "names": {
            case.name: ti2v_few_step_ref.prompt_names(_asset_metadata(), case) for case in cases
        },
        "sources": {case.image: _source(case.image) for case in cases},
        "tile": SYNTHETIC_TILE,
        "out_root": tmp_path / "series",
        "common": REF.common_metadata("0" * 64),
        **overrides,
    }
    return REF.generate(cases, **kwargs), scheduler


class TestSyntheticRun:
    """Writes through both phases with a synthetic 2.2 model and synthetic images (reads neither the
    5B weights nor real images)."""

    @pytest.fixture
    def run(self, monkeypatch, tmp_path, snapshot, ti2v_synthetic_vae):
        refs: list[weakref.ref] = []
        decodes: list[bool] = []

        def load() -> torch.nn.Module:
            model = _tiny_dit()
            refs.append(weakref.ref(model))
            return model

        decode_tiles = REF.vae_tiling.decode_tiles

        def spying(vae, latents, plan):
            # Whether the DiT has been collected at decode time (the order of the 2 phases).
            decodes.append(refs[0]() is None)
            return decode_tiles(vae, latents, plan)

        monkeypatch.setattr(REF.vae_tiling, "decode_tiles", spying)
        summaries, scheduler = _generate(
            monkeypatch,
            tmp_path,
            snapshot,
            ti2v_synthetic_vae,
            [CASES[0], CASES[2]],
            load,
            encoder_golden_root=tmp_path / "no-goldens",
        )
        written = {summary["case"]: _read_fixture(summary["path"]) for summary in summaries}
        return {
            "summaries": summaries,
            "written": written,
            "decodes": decodes,
            "scheduler": scheduler,
            "vae": ti2v_synthetic_vae,
        }

    def test_the_dit_is_released_before_the_first_decode(self, run):
        assert run["decodes"] == [True, True]

    def test_the_tensors_have_the_documented_names_shapes_and_types(self, run):
        latent_names = {"latents_init", "latents_final"} | {
            f"{kind}.{index}"
            for kind in ("noise_cond", "noise_uncond", "latents")
            for index in (0, 1)
        }
        observed = {"observed_m_s.latents.0", "observed_m_s.latents.1"}
        observed |= {"observed_m_s.latents_final"}
        others = {"condition_latents", "frames", "latents_mean", "latents_std", "source", "rgb8"}

        band, _ = run["written"]["band-boxing-cats"]
        accept, _ = run["written"]["accept-ferret"]

        assert set(band) == latent_names | observed | others
        assert set(accept) == latent_names | others
        for name in latent_names | observed:
            assert band[name].shape == SYNTHETIC_LATENT, name
            assert band[name].dtype == torch.float32, name
        assert band["condition_latents"].shape == (48, 1, 6, 10)
        assert band["frames"].shape == (3, 9, 96, 160)
        assert float(band["frames"].abs().max()) <= 1.0
        width, height = SYNTHETIC_SOURCE
        assert (band["source"].dtype, band["source"].shape) == (torch.uint8, (height, width, 3))
        assert (band["rgb8"].dtype, band["rgb8"].shape) == (torch.uint8, (96, 160, 3))

    def test_the_images_are_the_source_pixels_and_the_official_crop(self, run):
        tensors, metadata = run["written"]["accept-ferret"]
        source = _source("ferret")
        cropped = i2v_preprocess_ref.crop_resize(source.image, 160, 96)

        assert torch.equal(tensors["source"], i2v_preprocess_ref.rgb8(source.image))
        assert torch.equal(tensors["rgb8"], i2v_preprocess_ref.rgb8(cropped))
        assert (
            metadata["rgb8_sha256"] == hashlib.sha256(tensors["rgb8"].numpy().tobytes()).hexdigest()
        )
        assert metadata["source_rgb8_sha256"] == source.sha256

    def test_the_noise_is_the_seeded_torch_randn(self, run):
        tensors, _ = run["written"]["accept-ferret"]
        generator = torch.Generator().manual_seed(CASES[2].seed)

        assert torch.equal(
            tensors["latents_init"], torch.randn(1, *SYNTHETIC_LATENT, generator=generator)[0]
        )

    def test_the_condition_is_the_stage_9a_encoder_reference(self, run):
        """The condition latent = the value of the same function as the step 9a goldens (the mu of
        an untiled encode → normalization)."""
        tensors, _ = run["written"]["accept-ferret"]
        sample = i2v_preprocess_ref.to_signed_unit(
            i2v_preprocess_ref.crop_resize(_source("ferret").image, 160, 96)
        )
        with torch.no_grad():
            mu = vae_encoder_patch.reference_mu(run["vae"], sample)
            want = vae_encoder_patch.normalize_condition(run["vae"], mu)[0]

        assert torch.equal(tensors["condition_latents"], want)

    def test_upstream_cfg_and_unipc_rebuild_the_scheduler_latents(self, run):
        """From the recorded cond / uncond, upstream CFG + UniPC rebuild the unreplaced latents (a
        cond / uncond mix-up, a shifted step record, or passing the replaced latents to the
        scheduler breaks the bit match here)."""
        from diffusers import UniPCMultistepScheduler

        for case, (tensors, _) in run["written"].items():
            scheduler = UniPCMultistepScheduler.from_config(run["scheduler"].config)
            scheduler.set_timesteps(REF.STEPS)
            scheduler.set_begin_index(0)
            latents = tensors["latents_init"][None]
            for index, timestep in enumerate(scheduler.timesteps):
                uncond = tensors[f"noise_uncond.{index}"][None]
                cond = tensors[f"noise_cond.{index}"][None]
                noise = uncond + REF.GUIDANCE_SCALE * (cond - uncond)
                latents = scheduler.step(noise, timestep, latents, return_dict=False)[0]

                assert torch.equal(latents[0], tensors[f"latents.{index}"]), (case, index)

    def test_only_the_first_frame_of_the_final_latents_is_replaced(self, run):
        for case, (tensors, _) in run["written"].items():
            final = tensors["latents_final"]

            assert torch.equal(final[:, :1], tensors["condition_latents"]), case
            assert torch.equal(final[:, 1:], tensors["latents.1"][:, 1:]), case
            # Not vacuous: the first frame of the scheduler latents stays on the noise trajectory.
            assert not torch.equal(tensors["latents.1"][:, :1], tensors["condition_latents"])

    def test_the_official_form_replays_to_the_same_final_latents(self, run):
        for summary in run["summaries"]:
            official = summary["official_form"]

            assert official["model_inputs_bit_exact"] is True
            assert official["final_bit_exact"] is True
            assert official["final_max_abs"] == 0.0
            assert all(value > 0 for value in official["states_first_frame_max_abs"])

    def test_the_metadata_carries_the_documented_fields(self, run):
        _, band = run["written"]["band-boxing-cats"]
        _, accept = run["written"]["accept-ferret"]
        fields = {
            "role", "seed", "prompt", "negative", "steps", "guidance_scale", "flow_shift",
            "expand_timesteps", "timestep_mlp", "condition_tokens", "condition_timestep",
            "timesteps", "sigmas", "weights", "reference", "image", "image_sha256",
            "source_rgb8_sha256", "rgb8_sha256", "width", "height", "fit", "resample", "tile",
            "scale", "rows_starts", "cols_starts", "rows_blend", "cols_blend", "patch_size",
            "text_embeds_sha256", "source", "environment",
        }  # fmt: skip
        versions = {f"version_{name}" for name in ("pillow", "torch", "torchvision", "diffusers")}

        assert set(accept) == fields | versions
        assert set(band) == fields | versions | {"observed_m_s"}
        assert (band["role"], band["seed"], band["prompt"], band["negative"]) == (
            "band",
            "20262111",
            "boxing-cats",
            "negative",
        )
        assert (band["steps"], band["guidance_scale"], band["flow_shift"]) == ("2", "5.0", "5.0")
        assert (band["expand_timesteps"], band["timestep_mlp"]) == ("true", "per-value-m1")
        assert (band["condition_tokens"], band["condition_timestep"]) == ("15", "0")
        assert (band["image"], band["width"], band["height"]) == ("boxing-cats", "160", "96")
        assert (band["fit"], band["resample"]) == ("crop", "LANCZOS")
        assert band["image_sha256"] == _source("boxing-cats").sha256
        timesteps = json.loads(band["timesteps"])
        assert len(timesteps) == 2 and timesteps[0] == 999

    def test_the_summary_reports_the_m_s_observation_for_the_band_cases_only(self, run):
        band, accept = run["summaries"]

        assert band["observed_m_s_vs_m_1"]["condition_equal"] is True
        assert len(band["observed_m_s_vs_m_1"]["latents"]) == 2
        assert "observed_m_s_vs_m_1" not in accept
        assert band["condition_vs_encoder_golden"].startswith("missing ")
        assert set(band["first_frame_vs_image"]) == {"max_abs", "mean_abs"}


def _write_encoder_goldens(root: Path, vae, cases, *, perturb: str | None = None) -> None:
    """Write the step 9a encoder goldens (`source` / `rgb8` / `latent`) with synthetic images and
    the synthetic VAE. Only the key `perturb` is moved by one element (reproducing a mismatch
    between the golden and the reference)."""
    from safetensors.torch import save_file

    width, height = SYNTHETIC_LATENT[3] * 16, SYNTHETIC_LATENT[2] * 16
    for case in cases:
        source = _source(case.image)
        cropped = i2v_preprocess_ref.crop_resize(source.image, width, height)
        with torch.no_grad():
            mu = vae_encoder_patch.reference_mu(vae, i2v_preprocess_ref.to_signed_unit(cropped))
            latent = vae_encoder_patch.normalize_condition(vae, mu)[0]
        tensors = {
            "source": i2v_preprocess_ref.rgb8(source.image),
            "rgb8": i2v_preprocess_ref.rgb8(cropped),
            "latent": latent.contiguous(),
        }
        if perturb is not None:
            tensors[perturb] = tensors[perturb].clone()
            tensors[perturb].view(-1)[0] += 1
        golden = export_vae_encoder.GoldenCase(case.image, width, height, "crop", case.role)
        path = export_vae_encoder.golden_path(root, golden)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path))


class TestEncoderGolden:
    """On a machine with the step 9a encoder goldens, the condition image and the condition latent
    of the reference are bit-identical to them (a check)."""

    def _run(self, monkeypatch, tmp_path, snapshot, vae, root):
        summaries, _ = _generate(
            monkeypatch,
            tmp_path,
            snapshot,
            vae,
            [CASES[0], CASES[2]],
            _tiny_dit,
            encoder_golden_root=root,
        )
        return summaries

    def test_a_matching_golden_is_reported_bit_exact(
        self, monkeypatch, tmp_path, snapshot, ti2v_synthetic_vae
    ):
        root = tmp_path / "goldens"
        _write_encoder_goldens(root, ti2v_synthetic_vae, [CASES[0], CASES[2]])

        summaries = self._run(monkeypatch, tmp_path, snapshot, ti2v_synthetic_vae, root)

        for summary in summaries:
            golden = summary["condition_vs_encoder_golden"]
            assert golden["bit_exact"] is True, summary["case"]
            assert golden["keys"] == ["latent", "rgb8", "source"]

    @pytest.mark.parametrize("key", ["latent", "rgb8", "source"])
    def test_a_golden_that_differs_stops_before_any_fixture_is_written(
        self, monkeypatch, tmp_path, snapshot, ti2v_synthetic_vae, key
    ):
        root = tmp_path / "goldens"
        _write_encoder_goldens(root, ti2v_synthetic_vae, [CASES[0], CASES[2]], perturb=key)

        with pytest.raises(
            AssertionError, match=f"not bit-identical to the step 9a encoder golden.*'{key}'"
        ):
            self._run(monkeypatch, tmp_path, snapshot, ti2v_synthetic_vae, root)
        assert not (tmp_path / "series").exists()


class TestFaults:
    """The order and path checks can fail (fault injection)."""

    def test_a_dit_that_is_still_referenced_stops_before_the_decode(
        self, monkeypatch, tmp_path, snapshot, ti2v_synthetic_vae
    ):
        kept = _tiny_dit()

        with pytest.raises(AssertionError, match="a reference to the DiT remains"):
            _generate(monkeypatch, tmp_path, snapshot, ti2v_synthetic_vae, [CASES[2]], lambda: kept)
        assert not (tmp_path / "series").exists()

    def test_a_model_index_without_expand_timesteps_is_refused(self, snapshot, ti2v_synthetic_vae):
        """If false, it takes the Wan2.1 I2V branch (channel concatenation)."""
        (snapshot / "model_index.json").write_text(
            json.dumps({**MODEL_INDEX, "expand_timesteps": False})
        )

        with pytest.raises(AssertionError, match="expand_timesteps"):
            REF.build_pipeline(
                snapshot,
                _tiny_dit(),
                ti2v_synthetic_vae,
                ti2v_few_step_ref.load_scheduler(snapshot),
            )

    @pytest.fixture
    def pipeline(self, monkeypatch, snapshot, ti2v_synthetic_vae):
        monkeypatch.setattr(REF, "LATENT_SHAPE", SYNTHETIC_LATENT)
        return REF.build_pipeline(
            snapshot, _tiny_dit(), ti2v_synthetic_vae, ti2v_few_step_ref.load_scheduler(snapshot)
        )

    def _names(self, case):
        return ti2v_few_step_ref.prompt_names(_asset_metadata(), case)

    def test_a_non_finite_condition_stops_before_the_dit(self, monkeypatch, pipeline):
        pipeline_wan_i2v = pytest.importorskip("diffusers.pipelines.wan.pipeline_wan_i2v")
        original = pipeline_wan_i2v.retrieve_latents

        def poisoned(*args, **kwargs):
            latents = original(*args, **kwargs).clone()
            latents[0, 0, 0, 0, 0] = float("inf")
            return latents

        monkeypatch.setattr(pipeline_wan_i2v, "retrieve_latents", poisoned)
        forwards: list[int] = []
        pipeline.transformer.register_forward_hook(lambda *_: forwards.append(1))
        image = REF.condition_image(_source(), 160, 96)

        with pytest.raises(AssertionError, match="non-finite values"):
            REF.denoise_case(pipeline, CASES[2], _embeds(), self._names(CASES[2]), image)
        assert forwards == []

    def test_an_image_the_pipeline_resizes_is_refused(self, pipeline):
        """An image that is not exact-size is resized by upstream — the pixels would differ from
        the official preprocessing, so it stops."""
        source = _source()
        image = REF.ConditionImage(source=source, cropped=source.image)

        with pytest.raises(AssertionError, match="preprocessed image is not bit-identical"):
            REF.denoise_case(pipeline, CASES[2], _embeds(), self._names(CASES[2]), image)

    def test_the_m_s_observation_refuses_a_left_over_time_replacement(self, pipeline):
        image = REF.condition_image(_source(), 160, 96)
        denoised = REF.denoise_case(pipeline, CASES[0], _embeds(), self._names(CASES[0]), image)
        mask = REF.condition_token_mask(pipeline.transformer, SYNTHETIC_LATENT)

        with (
            REF.tokenwise_time_embedding(pipeline.transformer, mask),
            pytest.raises(AssertionError, match="the timestep replacement is still in place"),
        ):
            REF.observe_m_s(pipeline, denoised, _embeds())


class TestCli:
    @pytest.fixture
    def heavy(self, monkeypatch, tmp_path):
        """Replace the heavy steps (condition images, RAM check, loading, the 2 phases) with
        stand-ins that only record."""
        calls: dict[str, Any] = {}

        def called(name: str, value: Any) -> Any:
            calls.setdefault("order", []).append(name)
            return value

        monkeypatch.setattr(
            REF,
            "load_sources",
            lambda cases: called("sources", {case.image: case.image for case in cases}),
        )
        monkeypatch.setattr(ti2v_few_step_ref, "memory_estimate", lambda _model: 123)
        monkeypatch.setattr(
            dit_probe,
            "require_available",
            lambda stage, need: calls.update(memory=called("memory", (stage, need))),
        )
        monkeypatch.setattr(
            ti2v_few_step_ref, "load_rounded_vae", lambda _model: called("vae", "vae")
        )
        monkeypatch.setattr(REF, "local_snapshot", lambda _model: tmp_path)
        monkeypatch.setattr(
            ti2v_few_step_ref, "load_scheduler", lambda _snapshot: called("scheduler", "scheduler")
        )
        monkeypatch.setattr(
            ti2v_few_step_ref, "load_quantized_transformer", lambda _model: called("dit", "dit")
        )
        monkeypatch.setattr(export_vae_encoder, "library_versions", lambda: {"pillow": "12.3.0"})

        def generate(cases, **kwargs):
            called("generate", None)
            calls.update(cases=[case.name for case in cases], **kwargs)
            calls["dit"] = kwargs["load_transformer"]()
            return []

        monkeypatch.setattr(REF, "generate", generate)
        return calls

    def _asset(self, tmp_path) -> Path:
        path = tmp_path / "text_embeds.safetensors"
        metadata = _asset_metadata()
        embeds = _embeds()
        for entry in metadata["prompts"]:
            entry["tokens"] = embeds[entry["name"]].shape[0]
        text_embeds.write_asset(path, embeds, metadata)
        return path

    def test_help_loads_nothing(self, heavy, capsys):
        with pytest.raises(SystemExit) as raised:
            REF.main(["--help"])

        assert raised.value.code == 0
        assert heavy == {}
        assert "--case" in capsys.readouterr().out

    def test_case_selects_one_case_and_checks_images_and_memory_before_loading(
        self, heavy, tmp_path
    ):
        embeds = self._asset(tmp_path)

        assert REF.main(["--embeds", str(embeds), "--case", "accept-ferret"]) == 0

        assert heavy["cases"] == ["accept-ferret"]
        assert heavy["order"] == ["sources", "memory", "vae", "scheduler", "generate", "dit"]
        assert heavy["sources"] == {"ferret": "ferret"}
        assert heavy["snapshot"] == tmp_path
        assert heavy["tile"] == 16
        assert heavy["out_root"] == REF.SERIES
        assert heavy["encoder_golden_root"].name == "wan2.2-ti2v-5b-f16-dyn"
        assert heavy["names"]["accept-ferret"].negative == "negative"
        assert heavy["common"]["version_pillow"] == "12.3.0"
        assert (
            heavy["common"]["text_embeds_sha256"] == hashlib.sha256(embeds.read_bytes()).hexdigest()
        )

    def test_an_unknown_case_is_refused(self, heavy):
        with pytest.raises(SystemExit) as raised:
            REF.main(["--case", "accept-otter"])

        assert raised.value.code == 2
        assert heavy == {}

    def test_the_memory_gate_adds_the_measured_i2v_excess_to_the_t2v_estimate(
        self, heavy, tmp_path
    ):
        """The measured I2V peak (24.07 GiB) exceeded the T2V estimate (23.70 GiB), so the check
        adds an extra margin."""
        assert REF.main(["--embeds", str(self._asset(tmp_path))]) == 0

        assert heavy["memory"] == ("load", 123 + REF.I2V_EXCESS_BYTES)
        assert int(0.37 * 2**30) <= REF.I2V_EXCESS_BYTES


class TestLoadSources:
    """The provenance of the condition images (the same sha256 table as step 9a — a mismatch stops
    before the heavy loading)."""

    def test_a_missing_image_is_refused(self, monkeypatch, tmp_path):
        monkeypatch.setattr(export_vae_encoder, "IMAGE_DIR", tmp_path)

        with pytest.raises(FileNotFoundError, match="not tracked in git"):
            REF.load_sources([CASES[2]])

    def test_an_image_with_another_digest_is_refused(self, monkeypatch, tmp_path):
        image_module = pytest.importorskip("PIL.Image")
        monkeypatch.setattr(export_vae_encoder, "IMAGE_DIR", tmp_path)
        image_module.new("RGB", (832, 480)).save(tmp_path / "ferret-832x480.png")

        with pytest.raises(export_vae_encoder.EncoderGraphError, match="sha256"):
            REF.load_sources([CASES[2]])
