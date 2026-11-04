"""The contract of the Wan2.2 TI2V form of `wan/dit_patch.py` (ADR 0121 decisions 3 / 4, step 1).

It pins the side where a breakage raises no exception and **only the numbers drift silently**:

- the T2V use of the I2V-capable graph (`WanDitTokensTi2v` — two time inputs, a condition mask,
  `where`) is **bit-identical** to the upstream forward with a one-dimensional timestep (the same
  form of computation as 2.1 — the "why" of decision 3)
- the I2V use is **bit-identical** to the reference wrapper of decision 4 (it runs the time MLP per
  value with M = 1 and passes per-token tables to the upstream blocks) (fault injection: swapping
  the two time inputs, or shifting the mask by one token, splits them)
- the reference wrapper is bit-identical to the upstream one-dimensional timestep path when every
  token has the same time (T2V) (f32 / f64, both in-memory and layer by layer)
- the reference-wrapper path of the layer-by-layer writer (`wan.dit_reference.LayerwiseDit`) is
  bit-identical to the whole-model reference (for f64, the values of whole-model `double()` +
  `float64_forward`)
- the host function of the condition mask (true for the tokens of the first latent frame)

Runs only on a synthetic model (`test_dit_patch.TINY_DIT` — a small randomly initialised
`WanTransformer3DModel`). The eager equivalence on the real 5B weights is
`python -m wan.ti2v_export_dit write` (the check of the golden export — runbook).
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from wan import dit_patch, dit_reference
from wan.tests.test_dit_patch import TINY_DIT, TINY_LATENT

#: The synthetic patch (patch_size of TINY_DIT).
PATCH = (1, 2, 2)


def _tiny(seed: int = 20261004) -> nn.Module:
    transformer_wan = pytest.importorskip("diffusers.models.transformers.transformer_wan")
    torch.manual_seed(seed)
    return transformer_wan.WanTransformer3DModel(**TINY_DIT).to(torch.float32).eval()


def _inputs(seed: int = 11) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, TINY_DIT["in_channels"], *TINY_LATENT, generator=generator)
    timestep = torch.tensor([750], dtype=torch.int64)
    embeds = torch.randn(1, 5, TINY_DIT["text_dim"], generator=generator)
    return latents, timestep, embeds


def _graph_inputs(
    model: nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    embeds: torch.Tensor,
    condition_timestep: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return (
        dit_patch.dit_patchify(latents, PATCH),
        dit_patch.dit_timesteps_proj(model, timestep),
        embeds,
        *dit_patch.dit_rope_tables(model.rope, TINY_LATENT),
        dit_patch.dit_timesteps_proj(model, condition_timestep),
        mask,
    )


def _trunk(model: nn.Module, latents: torch.Tensor, inputs: tuple[torch.Tensor, ...]):
    """The path with the patch embedding replaced by the output of the upstream conv3d (compares
    only the body of the graph — the same as 2.1)."""
    wrapper = dit_patch.WanDitTokensTi2v(model)
    with torch.no_grad():
        hidden = model.patch_embedding(latents).flatten(2).transpose(1, 2)
        tokens = wrapper.forward_hidden(hidden, *inputs[1:])
    return dit_patch.dit_unpatchify(tokens, TINY_LATENT, PATCH)


def _i2v_mask() -> torch.Tensor:
    return dit_patch.dit_condition_mask(TINY_LATENT, PATCH, conditioned=True)


ZERO = torch.tensor([0], dtype=torch.int64)


class TestTheConditionMask:
    def test_i2v_marks_exactly_the_tokens_of_the_first_latent_frame(self) -> None:
        """Token index `(f·H' + h)·W' + w` — the 15 tokens of the first latent frame come first."""
        mask = _i2v_mask()

        assert mask.dtype == torch.bool
        assert list(mask.shape) == [1, 2 * 3 * 5, 1]
        assert mask[0, :15, 0].all()
        assert not mask[0, 15:, 0].any()

    def test_t2v_marks_nothing(self) -> None:
        mask = dit_patch.dit_condition_mask(TINY_LATENT, PATCH, conditioned=False)

        assert list(mask.shape) == [1, 30, 1]
        assert not mask.any()

    def test_a_temporal_patch_other_than_one_fails_loudly(self) -> None:
        with pytest.raises(NotImplementedError, match="temporal patch 2"):
            dit_patch.dit_condition_mask((4, 6, 10), (2, 2, 2), conditioned=True)


class TestTheTi2vGraphForm:
    def test_t2v_matches_the_upstream_one_dimensional_timestep_forward_bit_exactly(self) -> None:
        """All-false mask, conditioned-side time = generated side → bit-identical to the upstream
        `[B,6,C]` path."""
        model = _tiny()
        latents, timestep, embeds = _inputs()
        no_mask = dit_patch.dit_condition_mask(TINY_LATENT, PATCH, conditioned=False)
        inputs = _graph_inputs(model, latents, timestep, embeds, timestep, no_mask)
        with torch.no_grad():
            expected = dit_patch.reference_dit(model, latents, timestep, embeds)

        assert torch.equal(_trunk(model, latents, inputs), expected)

    def test_i2v_matches_the_reference_wrapper_bit_exactly(self) -> None:
        model = _tiny()
        latents, timestep, embeds = _inputs()
        mask = _i2v_mask()
        inputs = _graph_inputs(model, latents, timestep, embeds, ZERO, mask)
        condition = dit_patch.TimestepCondition(ZERO, mask[..., 0])
        with torch.no_grad():
            expected = dit_patch.reference_dit(model, latents, timestep, embeds, condition)
            one_dimensional = dit_patch.reference_dit(model, latents, timestep, embeds)

        assert torch.equal(_trunk(model, latents, inputs), expected)
        # Not a vacuous check: t = 0 of the conditioned frame differs in value from the
        # one-dimensional path (every token at t = 750).
        assert not torch.equal(expected, one_dimensional)

    @pytest.mark.parametrize("fault", ["swapped-times", "shifted-mask"])
    def test_a_wiring_fault_breaks_the_i2v_identity(self, fault: str) -> None:
        """Fault injection: swapping the two time inputs or shifting the condition mask by one
        token → splits from the reference wrapper (the same 2 injections as for the r check of
        step 2 — they only bite in the I2V form)."""
        model = _tiny()
        latents, timestep, embeds = _inputs()
        mask = _i2v_mask()
        condition = dit_patch.TimestepCondition(ZERO, mask[..., 0])
        if fault == "swapped-times":
            inputs = _graph_inputs(model, latents, ZERO, embeds, timestep, mask)
        else:
            inputs = _graph_inputs(model, latents, timestep, embeds, ZERO, mask.roll(1, dims=1))
        with torch.no_grad():
            expected = dit_patch.reference_dit(model, latents, timestep, embeds, condition)

        assert not torch.equal(_trunk(model, latents, inputs), expected)

    def test_the_two_wrappers_share_the_patch_embedding_and_parts(self) -> None:
        """The TI2V wrapper refers to the same parts as the 2.1 wrapper (weight rounding reaches
        both and upstream)."""
        model = _tiny()
        wrapper = dit_patch.WanDitTokensTi2v(model)

        assert wrapper.patch_embedding.weight.data_ptr() == model.patch_embedding.weight.data_ptr()
        assert wrapper.condition_embedder is model.condition_embedder
        assert wrapper.blocks is model.blocks


class TestTheReferenceWrapper:
    @pytest.mark.parametrize("variant", ["no-condition-token", "same-time"])
    def test_one_time_for_every_token_is_the_one_dimensional_path_in_f32(
        self, variant: str
    ) -> None:
        """When every token has the same time, reference wrapper = upstream one-dimensional
        timestep path (bit-identical).

        Both the all-false mask, and a true mask whose conditioned-side time equals the generated
        side.
        """
        model = _tiny()
        latents, timestep, embeds = _inputs()
        mask = _i2v_mask()[..., 0]
        condition = (
            dit_patch.TimestepCondition(timestep, torch.zeros_like(mask))
            if variant == "no-condition-token"
            else dit_patch.TimestepCondition(timestep.clone(), mask)
        )
        with torch.no_grad():
            expected = dit_patch.reference_dit(model, latents, timestep, embeds)
            got = dit_patch.reference_dit(model, latents, timestep, embeds, condition)

        assert torch.equal(got, expected)

    def test_one_time_for_every_token_is_the_one_dimensional_path_in_f64(self) -> None:
        model = copy.deepcopy(_tiny()).double()
        latents, timestep, embeds = _inputs()
        condition = dit_patch.TimestepCondition(timestep, torch.zeros_like(_i2v_mask()[..., 0]))
        with torch.no_grad(), dit_patch.float64_forward(model):
            expected = dit_patch.reference_dit(model, latents.double(), timestep, embeds.double())
            got = dit_patch.reference_dit(
                model, latents.double(), timestep, embeds.double(), condition
            )

        assert got.dtype == torch.float64
        assert torch.equal(got, expected)

    def test_the_original_condition_embedder_is_restored(self) -> None:
        model = _tiny()
        original = model.condition_embedder
        latents, timestep, embeds = _inputs()
        condition = dit_patch.TimestepCondition(ZERO, _i2v_mask()[..., 0])
        with torch.no_grad():
            dit_patch.reference_dit(model, latents, timestep, embeds, condition)

        assert model.condition_embedder is original

    def test_a_per_token_timestep_out_of_line_with_the_mask_fails_loudly(self) -> None:
        """It stops if the flat timestep upstream passes differs from `where(mask, conditioned side,
        generated side)` (if the order of upstream's flattening changed, the mapping between tokens
        and modulation would drift silently)."""
        model = _tiny()
        mask = _i2v_mask()[..., 0]
        embedder = dit_patch._TokenwiseTimeEmbedder(
            model.condition_embedder, torch.tensor([750]), dit_patch.TimestepCondition(ZERO, mask)
        )
        wrong = torch.full((mask.shape[1],), 750, dtype=torch.int64)

        with pytest.raises(AssertionError, match="does not match the condition mask"):
            embedder(wrong, _inputs()[2], None, mask.shape[1])

    def test_the_condition_timestep_must_be_one_value(self) -> None:
        model = _tiny()
        latents, timestep, embeds = _inputs()
        mask = _i2v_mask()[..., 0]

        with pytest.raises(ValueError, match="both the generated and the conditioned side"):
            dit_patch.reference_dit(
                model,
                latents,
                timestep,
                embeds,
                dit_patch.TimestepCondition(torch.zeros(1, mask.shape[1], dtype=torch.int64), mask),
            )

    def test_the_diffusers_per_token_path_runs_the_time_mlp_on_every_token(self) -> None:
        """The observed counterpart (diffusers' two-dimensional timestep — a linear with M = S)
        takes a different path from the reference wrapper.

        Checked by the row count of the input of the time MLP's linear: the reference wrapper feeds
        one row twice, the two-dimensional path S rows.
        """
        model = _tiny()
        latents, timestep, embeds = _inputs()
        mask = _i2v_mask()[..., 0]
        rows: list[int] = []
        handle = model.condition_embedder.time_embedder.linear_1.register_forward_hook(
            lambda _m, args, _o: rows.append(args[0].numel() // int(args[0].shape[-1]))
        )
        try:
            with torch.no_grad():
                dit_patch.reference_dit(
                    model, latents, timestep, embeds, dit_patch.TimestepCondition(ZERO, mask)
                )
                dit_patch.reference_dit(model, latents, torch.where(mask, ZERO, timestep), embeds)
        finally:
            handle.remove()

        assert rows == [1, 1, mask.shape[1]]


class TestTheLayerwiseTokenwisePath:
    """The layer-by-layer reference-wrapper path = the whole-model reference (how the f64 / f32
    goldens of the 5B are made)."""

    def test_f32_and_f64_are_bit_identical_to_the_whole_model(self) -> None:
        model = _tiny()
        latents, timestep, embeds = _inputs()
        condition = dit_patch.TimestepCondition(ZERO, _i2v_mask()[..., 0])
        double = copy.deepcopy(model).double()
        with torch.no_grad():
            expected32 = dit_patch.reference_dit(model, latents, timestep, embeds, condition)
            with dit_patch.float64_forward(double):
                expected64 = dit_patch.reference_dit(
                    double, latents.double(), timestep, embeds.double(), condition
                )
        writer = dit_reference.LayerwiseDit(model.config)
        source = dit_reference.ModuleWeights(model)

        got32 = writer.forward(
            source, torch.float32, latents, timestep, embeds, condition=condition
        )
        got64 = writer.forward(
            source, torch.float64, latents, timestep, embeds, condition=condition
        )

        assert torch.equal(got32.output, expected32)
        assert torch.equal(got64.output, expected64)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
    def test_one_time_for_every_token_is_the_one_dimensional_layerwise_path(
        self, dtype: torch.dtype
    ) -> None:
        """With T2V (every token at the same t), reference-wrapper path = one-dimensional timestep
        path (the acceptance of step 1)."""
        model = _tiny()
        latents, timestep, embeds = _inputs()
        condition = dit_patch.TimestepCondition(timestep, torch.zeros_like(_i2v_mask()[..., 0]))
        writer = dit_reference.LayerwiseDit(model.config)
        source = dit_reference.ModuleWeights(model)

        expected = writer.forward(source, dtype, latents, timestep, embeds)
        got = writer.forward(source, dtype, latents, timestep, embeds, condition=condition)

        assert torch.equal(got.output, expected.output)

    def test_the_patched_graph_runs_on_the_layerwise_weights(self) -> None:
        """The entry point that runs the eager patched graph on the layer-by-layer weights (`run` +
        `install_processors=False`) = the patched graph run on the in-memory model (the form of the
        5B eager equivalence)."""
        model = _tiny()
        latents, timestep, embeds = _inputs()
        mask = _i2v_mask()
        inputs = _graph_inputs(model, latents, timestep, embeds, ZERO, mask)
        with torch.no_grad():
            expected = dit_patch.WanDitTokensTi2v(copy.deepcopy(model))(*inputs)
        writer = dit_reference.LayerwiseDit(model.config)

        got, _, _ = writer.run(
            dit_reference.ModuleWeights(model),
            torch.float32,
            lambda outer: dit_patch.WanDitTokensTi2v(outer, install_processors=False)(*inputs),
            prepare_block=lambda block: dit_patch.install_real_pair_processors([block]),
        )

        assert torch.equal(got, expected)

    @pytest.mark.parametrize("installed", [True, False], ids=["prepared", "not-prepared"])
    def test_the_layerwise_patched_graph_runs_the_real_pair_rope(
        self, installed: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The layer-by-layer patched eager goes through the same real-form RoPE as the graph
        (twice per block, for q and k).

        The values cannot tell them apart: the upstream processor also accepts `RealPairRope`
        (holding tables of the same form as upstream) as a tuple and produces bit-identical values.
        So if `prepare_block` were dropped, the upstream path would run while we believed the graph
        path was running — hence the binding by call count.
        """
        model = _tiny()
        latents, timestep, embeds = _inputs()
        inputs = _graph_inputs(model, latents, timestep, embeds, ZERO, _i2v_mask())
        writer = dit_reference.LayerwiseDit(model.config)
        calls: list[int] = []
        real = dit_patch.real_pair_rotary

        def counting(*args: torch.Tensor) -> torch.Tensor:
            calls.append(1)
            return real(*args)

        monkeypatch.setattr(dit_patch, "real_pair_rotary", counting)
        writer.run(
            dit_reference.ModuleWeights(model),
            torch.float32,
            lambda outer: dit_patch.WanDitTokensTi2v(outer, install_processors=False)(*inputs),
            prepare_block=(
                (lambda block: dit_patch.install_real_pair_processors([block]))
                if installed
                else None
            ),
        )

        assert len(calls) == (2 * writer.layers if installed else 0)

    @pytest.mark.parametrize("hook", ["collect_blocks", "on_block", "on_output"])
    def test_block_hooks_that_cannot_reach_the_patched_graph_fail_loudly(self, hook: str) -> None:
        """The per-block entry points fire from the blocks' forward hooks — they cannot reach the
        TI2V graph, which calls the parts directly, so it stops instead of silently returning an
        empty record."""
        model = _tiny()
        latents, timestep, embeds = _inputs()
        inputs = _graph_inputs(model, latents, timestep, embeds, ZERO, _i2v_mask())
        writer = dit_reference.LayerwiseDit(model.config)
        seen: list[int] = []
        option: dict[str, object] = {
            "collect_blocks": {"collect_blocks": True},
            "on_block": {"on_block": lambda index, _seconds: seen.append(index)},
            "on_output": {"on_output": lambda index, _output: seen.append(index)},
        }[hook]

        with pytest.raises(dit_reference.DitReferenceError, match="forward was not called"):
            writer.run(
                dit_reference.ModuleWeights(model),
                torch.float32,
                lambda outer: dit_patch.WanDitTokensTi2v(outer, install_processors=False)(*inputs),
                prepare_block=lambda block: dit_patch.install_real_pair_processors([block]),
                **option,  # type: ignore[arg-type]
            )

        assert seen == []

    def test_a_single_block_run_is_checked_at_the_end(self) -> None:
        """Running a single block never reaches "the point where the next block is requested" —
        the final check stops it."""
        model = _tiny()
        latents, timestep, embeds = _inputs()
        inputs = _graph_inputs(model, latents, timestep, embeds, ZERO, _i2v_mask())
        writer = dit_reference.LayerwiseDit(model.config)

        with pytest.raises(dit_reference.DitReferenceError, match="block 0's forward"):
            writer.run(
                dit_reference.ModuleWeights(model),
                torch.float32,
                lambda outer: dit_patch.WanDitTokensTi2v(outer, install_processors=False)(*inputs),
                prepare_block=lambda block: dit_patch.install_real_pair_processors([block]),
                on_output=lambda _index, _output: None,
                blocks=1,
            )

    def test_the_upstream_reference_wrapper_still_feeds_every_block_hook(self) -> None:
        """On the reference-wrapper path, which goes through the upstream blocks' forward, the
        entry points fire on every block."""
        model = _tiny()
        latents, timestep, embeds = _inputs()
        condition = dit_patch.TimestepCondition(ZERO, _i2v_mask()[..., 0])
        writer = dit_reference.LayerwiseDit(model.config)
        timed: list[int] = []
        outputs: list[int] = []

        got = writer.forward(
            dit_reference.ModuleWeights(model),
            torch.float32,
            latents,
            timestep,
            embeds,
            condition=condition,
            collect_blocks=True,
            on_block=lambda index, _seconds: timed.append(index),
            on_output=lambda index, _output: outputs.append(index),
        )

        assert timed == outputs == list(range(writer.layers))
        assert len(got.blocks) == writer.layers
