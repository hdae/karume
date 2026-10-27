"""`moss/few_step_ref.py` — the case table the TS parity check mirrors, and the recording on a tiny
pipeline (without the 20 GB DiT; the container check runs in the script itself).

The recording tests build a `MOSSImagePipeline` from the pinned pipeline code ({@link
moss_snapshot}; SKIP when not cached) around a 4-block `MOSSImageTransformer` and a tiny
`AutoencoderKLFlux2` (the `moss/tests/test_reference.py` form) and check that the recorded latents
are upstream's own final latents and that the image is upstream's clamped decode of them.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from moss import few_step_ref, host_fixture, pipeline_ref
from moss.prompts import POSITIVE, prompt_by_name
from moss.records import bits_equal


class TestCases:
    def test_each_size_has_three_band_and_two_acceptance_cases(self):
        per_size = Counter((case.height, case.width, case.role) for case in few_step_ref.CASES)

        assert per_size == {
            (256, 256, "band"): 3,
            (256, 256, "accept"): 2,
            (256, 384, "band"): 3,
            (256, 384, "accept"): 2,
        }

    def test_names_and_seeds_are_unique_and_every_prompt_is_a_positive_one(self):
        names = [case.name for case in few_step_ref.CASES]
        seeds_per_size = Counter((c.height, c.width, c.seed) for c in few_step_ref.CASES)

        assert len(set(names)) == len(names)
        assert set(seeds_per_size.values()) == {1}
        assert {prompt_by_name(case.prompt).role for case in few_step_ref.CASES} == {POSITIVE}

    def test_the_acceptance_cases_differ_from_the_band_cases_in_seed(self):
        band = {c.seed for c in few_step_ref.CASES if c.role == "band"}
        accept = {c.seed for c in few_step_ref.CASES if c.role == "accept"}

        assert band.isdisjoint(accept)

    def test_a_case_is_found_by_its_name_and_an_unknown_name_fails(self):
        case = few_step_ref.CASES[3]

        assert few_step_ref.case_by_name(case.name) == case
        with pytest.raises(few_step_ref.FewStepReferenceError, match="no case"):
            few_step_ref.case_by_name("pipeline-1x1-band-x-s0")


#: The tiny DiT's depth and the negative branch's skip index (a middle block, as upstream's 22).
TINY_LAYERS = 4
TINY_SKIP = (2,)

#: Pixels (2 × 3 latent cells — non-square).
HEIGHT, WIDTH = 32, 48


@pytest.fixture(scope="module")
def tiny_pipe(moss_snapshot):
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLFlux2

    from moss import sources

    module = sources.load_pipeline_module(moss_snapshot)
    transformer = host_fixture.tiny_transformer(module, 256, num_layers=TINY_LAYERS)
    torch.manual_seed(1)
    # latent_channels 1 × the 2×2 patch = the tiny DiT's 4 latent channels.
    vae = AutoencoderKLFlux2(
        block_out_channels=(4, 4, 4, 4), latent_channels=1, norm_num_groups=2, layers_per_block=1
    ).eval()
    return pipeline_ref.build_pipeline(
        module, transformer=transformer, scheduler=module.MOSSImageScheduler(), vae=vae
    )


@pytest.fixture
def inputs(tiny_pipe):
    generator = torch.Generator().manual_seed(5)
    text_dim = tiny_pipe.transformer.config.text_dim
    channels = tiny_pipe.transformer.config.latent_channels
    return {
        "positive": torch.randn(3, text_dim, generator=generator),
        "negative": torch.randn(2, text_dim, generator=generator),
        "latents_init": torch.randn(1, channels, HEIGHT // 16, WIDTH // 16, generator=generator),
    }


def _run(pipe, inputs):
    return few_step_ref.run_case(
        pipe,
        inputs["positive"],
        inputs["negative"],
        inputs["latents_init"],
        height=HEIGHT,
        width=WIDTH,
        skip_layers=TINY_SKIP,
    )


class TestRunCase:
    def test_the_last_step_is_upstreams_final_latents_and_the_image_its_clamped_decode(
        self, tiny_pipe, inputs
    ):
        golden = _run(tiny_pipe, inputs)
        with torch.no_grad():
            final = tiny_pipe(
                prompt_embeds=inputs["positive"][None],
                negative_prompt_embeds=inputs["negative"][None],
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=few_step_ref.STEPS,
                guidance_scale=few_step_ref.GUIDANCE,
                uncond_skip_layers=TINY_SKIP,
                latents=inputs["latents_init"].clone(),
                output_type="latent",
            ).images
            z = pipeline_ref.unpatchify(pipeline_ref.denormalize(tiny_pipe.vae, final))
            decoded = tiny_pipe.vae.decode(z, return_dict=False)[0].clamp(-1, 1)

        assert sorted(golden) == ["image", "latents.0", "latents.1", "latents_init"]
        assert bits_equal(golden["latents_init"], inputs["latents_init"][0])
        assert bits_equal(golden["latents.1"], final[0])
        assert not bits_equal(golden["latents.0"], golden["latents.1"])
        assert bits_equal(golden["image"], decoded[0])
        assert golden["image"].shape == (3, HEIGHT, WIDTH)

    def test_the_step_wrapper_is_removed_after_the_call(self, tiny_pipe, inputs):
        _run(tiny_pipe, inputs)

        assert "step" not in vars(tiny_pipe.scheduler)

    def test_a_scheduler_with_an_instance_level_step_is_refused(self, tiny_pipe, inputs):
        tiny_pipe.scheduler.step = tiny_pipe.scheduler.step
        try:
            with pytest.raises(few_step_ref.FewStepReferenceError, match="instance-level step"):
                _run(tiny_pipe, inputs)
        finally:
            del tiny_pipe.scheduler.step
