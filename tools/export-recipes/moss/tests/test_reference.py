"""`moss/reference.py` — the host steps it re-assembles with, against the upstream code they stand
in for, and the gate itself on a tiny pipeline (without the 20 GB DiT; the full gate runs in the
script itself).

The gate tests build a `MOSSImagePipeline` from the pinned pipeline code ({@link moss_snapshot};
SKIP when not cached) around a 4-block `MOSSImageTransformer` and a tiny `AutoencoderKLFlux2`,
record one call, and check that the unchanged recording passes and that each perturbation of it —
a sample one ULP off, two block outputs swapped, another skip index — fails the gate.
"""

from __future__ import annotations

import math

import pytest
import torch

from _shared.paths import REPO_ROOT
from moss import host_fixture, pipeline_ref, reference
from moss.prompts import NEGATIVE, POSITIVE
from moss.records import bits_equal


class TestEuler:
    def test_it_is_the_flow_match_euler_step_of_the_pinned_diffusers(self):
        """The trajectory check rests on this being the scheduler's update to the bit."""
        pytest.importorskip("diffusers")
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
        scheduler.set_timesteps(sigmas=[1.0, 0.637, 0.0241])
        generator = torch.Generator().manual_seed(3)
        sample = torch.randn(1, 128, 4, 6, generator=generator)
        velocity = torch.randn(1, 128, 4, 6, generator=generator)
        expected = sample
        ours = sample

        for step, timestep in enumerate(scheduler.timesteps):
            expected = scheduler.step(velocity, timestep, expected, return_dict=False)[0]
            ours = reference.euler(
                ours, velocity, scheduler.sigmas[step], scheduler.sigmas[step + 1]
            )
            assert bits_equal(ours, expected), step


class TestSize:
    def test_height_comes_first(self):
        assert reference.parse_size("256x384") == (256, 384)


class TestRepositoryName:
    def test_a_path_in_the_repository_is_named_relative_to_its_root(self):
        path = REPO_ROOT / "outputs" / "series" / "x" / "a.safetensors"

        assert reference.repository_name(path) == "outputs/series/x/a.safetensors"

    def test_a_path_outside_the_repository_keeps_only_its_file_name(self, tmp_path):
        assert reference.repository_name(tmp_path / "a.safetensors") == "a.safetensors"


#: The tiny DiT's depth and the negative branch's skip index (a middle block, as upstream's 22).
TINY_LAYERS = 4
TINY_SKIP = (2,)

#: Pixels (2 × 3 latent cells — non-square, as the real cases).
HEIGHT, WIDTH = 32, 48


@pytest.fixture(scope="module")
def module(moss_snapshot):
    pytest.importorskip("diffusers")
    from moss import sources

    return sources.load_pipeline_module(moss_snapshot)


@pytest.fixture(scope="module")
def tiny_pipe(module):
    from diffusers import AutoencoderKLFlux2

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
def recording(tiny_pipe):
    generator = torch.Generator().manual_seed(5)
    text_dim = tiny_pipe.transformer.config.text_dim
    embeds = {
        POSITIVE: torch.randn(3, text_dim, generator=generator),
        NEGATIVE: torch.randn(2, text_dim, generator=generator),
    }
    channels = tiny_pipe.transformer.config.latent_channels
    latents = torch.randn(1, channels, HEIGHT // 16, WIDTH // 16, generator=generator)

    return reference.record_case(
        tiny_pipe, embeds, latents, height=HEIGHT, width=WIDTH, skip_layers=TINY_SKIP
    )


def _forward(recording, step, branch):
    found = [f for f in recording.forwards[2 * step : 2 * step + 2] if f.branch == branch]
    assert len(found) == 1
    return found[0]


class TestGate:
    def test_an_unchanged_recording_passes(self, tiny_pipe, module, recording):
        tensors, gate = reference.check_recording(
            tiny_pipe, module, recording, skip_layers=TINY_SKIP
        )

        assert gate["passed"] is True
        assert gate["blocks_recorded"] == reference.STEPS * (2 * TINY_LAYERS - len(TINY_SKIP))
        assert "step1.negative.block3" in tensors
        assert "step1.negative.block2" not in tensors

    def test_the_skip_index_reaches_the_negative_branch_only(self, recording):
        """Otherwise the perturbation tests below would not exercise a skipped block."""
        assert _forward(recording, 0, POSITIVE).ran == [0, 1, 2, 3]
        assert _forward(recording, 0, NEGATIVE).ran == [0, 1, 3]

    def test_a_sample_one_ulp_off_fails(self, tiny_pipe, module, recording):
        forward = _forward(recording, 1, NEGATIVE)
        assert forward.sample is not None
        sample = forward.sample.clone()
        flat = sample.view(-1)
        flat[0] = torch.nextafter(flat[0], torch.tensor(math.inf))
        forward.sample = sample

        with pytest.raises(reference.DitReferenceError, match="step 1 negative: head"):
            reference.check_recording(tiny_pipe, module, recording, skip_layers=TINY_SKIP)

    def test_two_swapped_block_outputs_fail(self, tiny_pipe, module, recording):
        blocks = _forward(recording, 0, POSITIVE).tensors
        blocks["block1"], blocks["block2"] = blocks["block2"], blocks["block1"]

        with pytest.raises(reference.DitReferenceError, match=r"step 0 positive: chain broken"):
            reference.check_recording(tiny_pipe, module, recording, skip_layers=TINY_SKIP)

    def test_another_skip_index_fails(self, tiny_pipe, module, recording):
        with pytest.raises(reference.DitReferenceError, match="negative: skip_layers"):
            reference.check_recording(tiny_pipe, module, recording, skip_layers=(1,))
