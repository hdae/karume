"""`moss/host_fixture.py` — the invariants the fixed values must hold.

The membership lookup and the thread comparison are checked with fault injection (a value one ULP
off, a sign-flipped zero). The upstream-derived values use the pinned pipeline code
({@link moss_snapshot}; SKIP when not cached). The generated record, if present, is checked for
the facts the later steps rely on.
"""

from __future__ import annotations

import pytest
import torch

from moss import host_fixture, pipeline_ref, records


def _next_after(value: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(value, torch.tensor(2.0))


class TestGridIndices:
    def test_members_map_to_their_positions(self):
        table = torch.tensor([1.0, 0.75, 0.5, 0.25])

        index = host_fixture.grid_indices(table, torch.tensor([1.0, 0.5, 0.25]))

        assert index.tolist() == [0, 2, 3]

    def test_a_value_one_ulp_off_is_not_a_member(self):
        table = torch.tensor([1.0, 0.75, 0.5])

        with pytest.raises(host_fixture.HostFixtureError):
            host_fixture.grid_indices(table, _next_after(torch.tensor([0.75])))


class TestDisagreements:
    def test_one_ulp_and_a_signed_zero_are_disagreements(self):
        base = {"a": torch.tensor([0.5, 0.0]), "b": torch.tensor([1.0])}
        moved = {"a": torch.tensor([0.5, -0.0]), "b": _next_after(torch.tensor([1.0]))}

        assert host_fixture.disagreements(base, moved) == ["a", "b"]
        assert host_fixture.disagreements(base, dict(base)) == []

    def test_a_missing_name_is_a_disagreement(self):
        assert host_fixture.disagreements({"a": torch.ones(1)}, {}) == ["a"]


@pytest.fixture(scope="module")
def module(moss_snapshot):
    pytest.importorskip("diffusers")
    from moss import sources

    return sources.load_pipeline_module(moss_snapshot)


class TestUpstreamValues:
    def test_the_training_grid_is_a_thousand_descending_points_from_one(
        self, moss_snapshot, module
    ):
        config = host_fixture.scheduler_config(moss_snapshot, module)

        sigmas, timesteps = host_fixture.training_grid(module, config)

        assert sigmas.shape == timesteps.shape == (1000,)
        assert float(sigmas[0]) == 1.0
        assert bool((sigmas[1:] < sigmas[:-1]).all())

    def test_a_step_sequence_snaps_onto_the_training_grid(self, moss_snapshot, module):
        config = host_fixture.scheduler_config(moss_snapshot, module)
        sigmas, timesteps = host_fixture.training_grid(module, config)
        scheduler = module.MOSSImageScheduler.from_config(config)

        sequence = host_fixture.step_sequence(module, scheduler, 50)
        index = host_fixture.grid_indices(sigmas, sequence.sigmas[:-1]).long()

        assert records.bits_equal(timesteps[index], sequence.timesteps)
        assert float(sequence.sigmas[-1]) == 0.0

    def test_the_requested_grid_snaps_to_its_nearest_training_points(self, moss_snapshot, module):
        """`nearest` is the snapping itself: no training neighbour of a snapped point is closer to
        the requested point, and dropping the consecutive duplicates of the snapped values gives
        the sequence."""
        config = host_fixture.scheduler_config(moss_snapshot, module)
        scheduler = module.MOSSImageScheduler.from_config(config)

        sequence = host_fixture.step_sequence(module, scheduler, 500)
        training, nearest = sequence.training_ascending, sequence.nearest_ascending
        requested = sequence.requested
        lower = (nearest - 1).clamp(min=0)
        upper = (nearest + 1).clamp(max=training.shape[0] - 1)
        distance = (requested - training[nearest]).abs()

        assert sequence.requested.shape == (500,)
        assert float(requested[0]) == 1.0
        assert bool((distance <= (requested - training[lower]).abs()).all())
        assert bool((distance <= (requested - training[upper]).abs()).all())
        assert records.bits_equal(torch.unique_consecutive(training[nearest]), sequence.sigmas[:-1])
        assert sequence.timesteps.shape[0] < 500  # 500 requested points snap with duplicates

    def test_the_wrapped_torch_functions_are_restored(self, moss_snapshot, module):
        config = host_fixture.scheduler_config(moss_snapshot, module)
        scheduler = module.MOSSImageScheduler.from_config(config)
        before = (torch.searchsorted, torch.where)

        host_fixture.step_sequence(module, scheduler, 3)

        assert (torch.searchsorted, torch.where) == before

    def test_a_shifted_scheduler_changes_only_the_shift_and_its_requested_grid(
        self, moss_snapshot, module
    ):
        config = host_fixture.scheduler_config(moss_snapshot, module)

        shifted = host_fixture.shifted_scheduler(moss_snapshot, module, 3.7)
        pinned = module.MOSSImageScheduler.from_config(config)

        assert {**shifted.config, "shift": config["shift"]} == dict(pinned.config)
        moved = host_fixture.step_sequence(module, shifted, 7).requested
        kept = host_fixture.step_sequence(module, pinned, 7).requested
        assert float(moved[0]) == float(kept[0]) == 1.0
        assert not records.bits_equal(moved, kept)

    def test_a_guidance_step_reads_both_samples_and_upstreams_combination(
        self, moss_snapshot, module
    ):
        """Two forwards and one step per call; the skipped block moves the negative sample; scale 0
        hands the negative sample on (`uncond + 0 · …`); the step wrapper is gone afterwards."""
        config = host_fixture.scheduler_config(moss_snapshot, module)
        transformer = host_fixture.tiny_transformer(module, 256, num_layers=host_fixture.CFG_LAYERS)
        pipe = pipeline_ref.build_pipeline(
            module,
            transformer=transformer,
            scheduler=module.MOSSImageScheduler.from_config(config),
        )
        generator = torch.Generator().manual_seed(1)
        text_dim = transformer.config.text_dim
        cells = host_fixture.CFG_CELLS
        inputs = {
            "positive": torch.randn(1, 3, text_dim, generator=generator),
            "negative": torch.randn(1, 2, text_dim, generator=generator),
            "latents": torch.randn(1, 4, cells, cells, generator=generator),
        }

        cond, uncond, combined = host_fixture.guidance_step(pipe, 0.0, inputs)
        _, _, guided = host_fixture.guidance_step(pipe, 3.3, inputs)

        assert cond.shape == uncond.shape == combined.shape == (1, 4, cells, cells)
        assert not records.bits_equal(cond, uncond)
        assert torch.equal(combined, uncond)
        assert not torch.equal(guided, uncond)
        assert "step" not in vars(pipe.scheduler)

    def test_the_image_rows_of_the_rope_table_move_with_the_text_length(
        self, moss_snapshot, module
    ):
        """Upstream puts image tokens at f = T — the table is not a function of (H, W) alone."""
        config = host_fixture.transformer_config(moss_snapshot)

        short = host_fixture.rope_table(module, config, 4, 6, 10)
        long = host_fixture.rope_table(module, config, 4, 6, 11)

        assert short.shape == (1, 4 * 6 + 10, 128)
        assert not torch.equal(short[:, : 4 * 6], long[:, : 4 * 6])
        assert torch.equal(short[:, 4 * 6 :], long[:, 4 * 6 : 4 * 6 + 10])


class TestGeneratedRecord:
    @pytest.fixture
    def record(self):
        path = records.default_reference_dir().parent
        found = (
            sorted(path.glob(f"*_stage0-reference/{host_fixture.RECORD_NAME}"))
            if path.is_dir()
            else []
        )
        if not found:
            pytest.skip("no host fixture record has been generated")
        return records.read_record(found[-1], key=host_fixture.METADATA_KEY)

    def test_it_agreed_across_thread_counts(self, record):
        _, metadata = record

        assert metadata["threads_agree"] is True
        assert metadata["thread_counts"][0] == 1 and metadata["thread_counts"][1] > 1

    def test_the_sinusoid_table_has_a_row_per_training_timestep(self, record):
        tensors, _ = record

        assert tensors["time_sinusoid"].shape == (1000, 256)
        assert tensors["training.timesteps"].shape == (1000,)

    def test_every_step_count_carries_its_requested_grid_and_snapped_indices(self, record):
        """`nearest` indexes the descending training grid; its consecutive duplicates dropped, it
        is `grid`, and the training sigmas it picks are the sequence."""
        tensors, metadata = record
        training = tensors["training.sigmas"]

        for steps in metadata["step_counts"]:
            requested = tensors[f"steps.{steps}.requested"]
            nearest = tensors[f"steps.{steps}.nearest"]
            grid = tensors[f"steps.{steps}.grid"]
            assert requested.shape == nearest.shape == (steps,), steps
            assert requested.dtype == torch.float32 and nearest.dtype == torch.int32, steps
            assert records.bits_equal(torch.unique_consecutive(nearest), grid), steps
            snapped = torch.unique_consecutive(training[nearest.long()])
            assert records.bits_equal(snapped, tensors[f"steps.{steps}.sigmas"][:-1]), steps

    def test_every_shift_case_carries_its_sequence_on_the_same_training_grid(self, record):
        tensors, metadata = record
        training = tensors["training.sigmas"]
        cases = metadata["shift_cases"]

        assert cases == {
            "shifts": list(host_fixture.SHIFTS),
            "step_counts": list(host_fixture.SHIFT_STEP_COUNTS),
        }
        for index, _shift in enumerate(cases["shifts"]):
            for steps in cases["step_counts"]:
                where = f"shift.{index}.steps.{steps}"
                requested, nearest = tensors[f"{where}.requested"], tensors[f"{where}.nearest"]
                assert requested.shape == nearest.shape == (steps,), where
                assert records.bits_equal(
                    torch.unique_consecutive(nearest), tensors[f"{where}.grid"]
                )
                snapped = torch.unique_consecutive(training[nearest.long()])
                assert records.bits_equal(snapped, tensors[f"{where}.sigmas"][:-1]), where

    def test_the_cfg_record_has_one_combination_per_scale(self, record):
        tensors, metadata = record
        scales = metadata["cfg"]["scales"]

        assert scales == list(host_fixture.CFG_SCALES)
        assert tensors["cfg.cond"].shape == tensors["cfg.uncond"].shape
        for index in range(len(scales)):
            assert tensors[f"cfg.{index}.combined"].shape == tensors["cfg.cond"].shape
