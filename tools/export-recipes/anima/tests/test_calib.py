"""The rules of the wiring of calibrated i4 (GPTQ) (the part that needs no real weights).

Calibration with real weights is manual (README / `anima.export --dtype i4`). What is pinned here
is only what gives a **false PASS** when broken:

- the stage-by-stage block loop is bit-identical to one forward of the wrapper (rounding on a
  drifted path still produces ordinary numbers)
- the i4-eligible set splits into "inside blocks = calibrated" and "outside blocks = plain RTN"
  **exactly and exclusively**
- the sensitivity-experiment variant (`--i4-adaln-i8`) moves adaLN + the outside of the blocks to
  i8 by role, and **the default (off) does not change the set, the path or the values by a single
  bit**
- the keys of the scale ledger live in the wrapper's FQN space (= the container's initializer
  names)
- calibration **actually produces a different rounding** (if it passed through, the storage form
  would be the same, so it could not be read from the assets)
- the calibration inputs and the graph call the blocks the same way
- the calibration inputs are taken from **weights in the same state as the shipping
  conditions** (outside the blocks after rounding, inside the blocks before rounding)
- `--no-calib`, the opt-out, really does not run calibration

The model is a tiny skeleton that copies **the same FQNs as the real one**
(`model.transformer_blocks.<i>` / `model.time_embed.t_embedder.linear_1` /
`model.norm_out.linear_1` / `model.proj_out`) and **the same call shapes** (a block takes hidden
as a positional argument and returns hidden; the quantization axis of `patch_embed.proj` is not
divisible by g32). The names are copied too so that the checks can be exercised without replacing
the constants meant for the real model ({@link anima.export.NON_STAGE_I4_WEIGHTS}).

Like the real one, the model has **two block loops** ({@link TinyRawDit} = the plain model's
forward / {@link TinyWrapper} = the exported wrapper). Merging them into one would put the capture
source and the graph on the same code path, and the very seam that
{@link anima.calib.assert_calib_batches_match_graph} guards would vanish from the tests (in the
real model the capture uses the plain model and the rounding uses the wrapper).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
from dataclasses import replace

import pytest
import torch
from torch import nn

from anima import calib, pipeline_ref
from anima import export as export_anima
from anima.distribution import (
    ANIMA_BASE_MODEL_NAME,
    ANIMA_MODELS,
    ANIMA_TURBO_MODEL_NAME,
    CALIB_SHIPPABLE_DEVICE,
)
from karume.quantize import fake_quant_int4, fake_quant_int8

#: The models that use CFG (= everything except the guidance 1 distilled ones). They are
#: **derived from the declarations instead of being listed** so that, the day models are added,
#: "a model that never goes through this check" cannot appear silently (the restructuring of
#: 2026-09-01 added the official Aesthetic and the v1.0 generation — ADR 0087). They are split by
#: the guidance of pipeline_config rather than by name because turbo now has several versions
#: (anima-turbo-v1.0 / v1.1).
CFG_MODELS = tuple(
    name
    for name, spec in ANIMA_MODELS.items()
    if spec.pipeline_config["defaults"]["guidanceScale"] != 1
)
TURBO_LIKE_MODELS = tuple(name for name in ANIMA_MODELS if name not in CFG_MODELS)

#: The quantization axis (= the group length). i4 makes no partial groups (ADR 0069 decision 2).
HIDDEN = 32
#: Input channels of the patchify entry (17ch × 2×2 = 68, as in the real DiT — not divisible by
#: g32, so not i4-eligible).
PATCH_IN = 68
#: Token count (the calibration hidden is `[1, TOKENS, HIDDEN]`).
TOKENS = 4
#: The latent side of the capture tests and the resolution that produces it (the real 512px is
#: 64² tokens; 8² is enough here — what is checked is how branches are counted, not the
#: distribution of activations).
CAPTURE_LATENT = 8
CAPTURE_RESOLUTION = CAPTURE_LATENT * pipeline_ref.SPATIAL_COMPRESSION


class TinyAdaLayerNormZero(nn.Module):
    """Stands in for a block's `norm1` / `norm2` / `norm3` (`CosmosAdaLayerNormZero`).

    Like the real one it has **two** linears, `linear_1` / `linear_2`, and its input is
    `embedded_timestep` (rank2), not hidden — this is the spelling the sensitivity-experiment
    variant (`--i4-adaln-i8`) removes from i4, and the side where the samples of `H = Σ XᵀX` that
    GPTQ sees accumulate only one row per block.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN, elementwise_affine=False)
        self.linear_1 = nn.Linear(HIDDEN, HIDDEN)
        self.linear_2 = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, hidden_states: torch.Tensor, embedded_timestep: torch.Tensor):
        scale = self.linear_2(torch.nn.functional.silu(self.linear_1(embedded_timestep)))
        return self.norm(hidden_states) * (1 + scale.unsqueeze(1))


class TinyBlock(nn.Module):
    """A block that copies only **the call shape** of `CosmosTransformerBlock`.

    It is called with 8 positional arguments (hidden + 7 companions) and returns hidden — the
    same as the real one, so the stages can be listed as they are, without wrapping
    (`anima.calib.dit_stages`).

    The modulation (`norm1` / `norm2` / `norm3`) is copied too so that the variant's split can
    be exercised without replacing the segment constants meant for the real model
    ({@link anima.calib.ADALN_SEGMENTS}) — the linears inside a block are 6 adaLN + 2 others.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = TinyAdaLayerNormZero()
        self.attn = nn.Linear(HIDDEN, HIDDEN)
        self.norm2 = TinyAdaLayerNormZero()
        self.norm3 = TinyAdaLayerNormZero()
        self.ff = nn.Linear(HIDDEN, HIDDEN)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        extra_pos_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        controlnet_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended = torch.tanh(
            self.attn(self.norm1(hidden_states, embedded_timestep))
            + self.norm2(encoder_hidden_states, embedded_timestep).mean(dim=1, keepdim=True)
        )
        return hidden_states + self.ff(self.norm3(attended, embedded_timestep)) * temb.unsqueeze(1)


class TinyTimestepEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(HIDDEN, HIDDEN)
        self.linear_2 = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, timesteps_proj: torch.Tensor) -> torch.Tensor:
        return self.linear_2(torch.nn.functional.silu(self.linear_1(timesteps_proj)))


class TinyEmbedding(nn.Module):
    """Stands in for `CosmosEmbedding` (`t_embedder` gives temb, `norm` gives
    embedded_timestep)."""

    def __init__(self) -> None:
        super().__init__()
        self.t_embedder = TinyTimestepEmbedding()
        self.norm = nn.LayerNorm(HIDDEN, elementwise_affine=False)


class TinyAdaLayerNorm(nn.Module):
    """Stands in for `CosmosAdaLayerNorm` (= `norm_out`)."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN, elementwise_affine=False)
        self.linear_1 = nn.Linear(HIDDEN, HIDDEN)
        self.linear_2 = nn.Linear(HIDDEN, 2 * HIDDEN)

    def forward(
        self, hidden_states: torch.Tensor, embedded_timestep: torch.Tensor, temb: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = self.linear_2(self.linear_1(embedded_timestep)).chunk(2, dim=-1)
        return self.norm(hidden_states) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TinyPatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(PATCH_IN, HIDDEN)


class TinyRawDit(nn.Module):
    """The skeleton of `CosmosTransformer3DModel` and **the plain forward** (the source the
    calibration inputs are captured from).

    In the real model the capture runs the plain model through the reference denoise
    (`export._round_i4_calibrated` passes `wrapper.model`), so the block loop is held as **a
    separate implementation** from the wrapper — so that the tests can exercise the check
    catching a disagreement with the wrapper the day upstream adds or removes arguments here.
    """

    def __init__(self, blocks: int) -> None:
        super().__init__()
        self.patch_embed = TinyPatchEmbed()
        self.time_embed = TinyEmbedding()
        self.transformer_blocks = nn.ModuleList(TinyBlock() for _ in range(blocks))
        self.norm_out = TinyAdaLayerNorm()
        self.proj_out = nn.Linear(HIDDEN, HIDDEN)

    def forward(
        self,
        tokens: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.patch_embed.proj(tokens)
        temb = self.time_embed.t_embedder(timesteps_proj)
        embedded_timestep = self.time_embed.norm(timesteps_proj)
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                embedded_timestep,
                temb,
                None,
                None,
                None,
                None,
            )
        hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
        return self.proj_out(hidden_states)


class DriftingRawDit(TinyRawDit):
    """The plain model on the day upstream drops one companion argument of the block (the wrapper
    keeps 8).

    If this happened to the real model, only the calibration would choose rounding targets under
    a condition that does not see `controlnet_residual`, and the numbers would come out as usual
    (the goldens stay green too).
    """

    def forward(
        self,
        tokens: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.patch_embed.proj(tokens)
        temb = self.time_embed.t_embedder(timesteps_proj)
        embedded_timestep = self.time_embed.norm(timesteps_proj)
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states, embedded_timestep, temb, None, None, None
            )
        hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
        return self.proj_out(hidden_states)


class TinyWrapper(nn.Module):
    """The skeleton of `patch.AnimaDit` (patchify → block loop → norm_out → proj_out)."""

    def __init__(self, model: TinyRawDit) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        tokens: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        hidden_states = model.patch_embed.proj(tokens)
        temb = model.time_embed.t_embedder(timesteps_proj)
        embedded_timestep = model.time_embed.norm(timesteps_proj)
        for block in model.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                embedded_timestep,
                temb,
                None,
                None,
                None,
                None,
            )
        hidden_states = model.norm_out(hidden_states, embedded_timestep, temb)
        return model.proj_out(hidden_states)


class DenoiseDit(nn.Module):
    """A miniature called **as the real thing** by the reference denoise (the way
    `CosmosTransformer3DModel` is called).

    It is needed besides {@link TinyRawDit} because {@link anima.calib.capture_stage_batches}
    runs the DiT through `pipeline_ref.reference_steps` — the entry takes 5 keywords, the latent
    is rank5 `[1, LATENT_CHANNELS, 1, h, w]`, and the output must have the same shape (the shape
    for which the Euler update `x + Δσ·pred` holds).

    MUST: do not replace `reference_steps` with a copy. Copying the CFG branch (`skip_uncond`)
    would make the cross-check of the captured batch count only check **the test's copy**, and
    the test would drift along with it the day upstream's branch changes.
    """

    def __init__(self, blocks: int = 2, seed: int = 11) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.patch_embed = nn.Linear(pipeline_ref.LATENT_CHANNELS, HIDDEN)
        self.time_embed = TinyEmbedding()
        self.transformer_blocks = nn.ModuleList(TinyBlock() for _ in range(blocks))
        self.norm_out = TinyAdaLayerNorm()
        self.proj_out = nn.Linear(HIDDEN, pipeline_ref.LATENT_CHANNELS)
        self.eval()

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        padding_mask: torch.Tensor,
        return_dict: bool = True,
    ):
        batch, channels, frames, height, width = hidden_states.shape
        tokens = hidden_states.reshape(batch, channels, height * width).transpose(1, 2)
        hidden = self.patch_embed(tokens)
        temb = self.time_embed.t_embedder(timestep.reshape(1, 1).expand(batch, HIDDEN))
        embedded_timestep = self.time_embed.norm(temb)
        for block in self.transformer_blocks:
            hidden = block(
                hidden, encoder_hidden_states, embedded_timestep, temb, None, None, None, None
            )
        hidden = self.norm_out(hidden, embedded_timestep, temb)
        out = self.proj_out(hidden).transpose(1, 2).reshape(batch, channels, frames, height, width)
        return out if return_dict else (out,)


def embed_for(text: str) -> torch.Tensor:
    """Prompt → text condition (deterministic; a different tensor per string).

    Scaffolding to tell from the captured contents that cond and uncond are **different
    embeddings** — a stub returning the same value would hide the breakage "feed cond to uncond
    as is" from the tests.
    """
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return torch.randn(1, TOKENS, HIDDEN, generator=torch.Generator().manual_seed(seed))


def make_wrapper(blocks: int = 3, seed: int = 0, dit: type[TinyRawDit] = TinyRawDit) -> TinyWrapper:
    torch.manual_seed(seed)
    wrapper = TinyWrapper(dit(blocks))
    wrapper.eval()
    return wrapper


def make_probes(count: int = 3, seed: int = 7) -> tuple[tuple[torch.Tensor, ...], ...]:
    """The wrapper's positional arguments (in the real model the 1st is the probe and the rest
    correspond to the conditions the reference denoise builds)."""
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        (
            torch.randn(1, TOKENS, PATCH_IN, generator=generator),
            torch.randn(1, HIDDEN, generator=generator),
            torch.randn(1, TOKENS, HIDDEN, generator=generator),
        )
        for _ in range(count)
    )


def capture_batches(raw: nn.Module, probes) -> tuple:
    """Run **the plain model** and take the `(args, kwargs)` passed to the first block.

    In the real model `calib.capture_stage_batches` does the same in the reference denoise (what
    gets driven is `wrapper.model` = the plain `CosmosTransformer3DModel`) — running the wrapper
    here would put the capture source and the graph on the same code path, and only inputs that
    trivially pass the companion-argument check could be built.
    """
    blocks = raw.transformer_blocks
    names = tuple(inspect.signature(blocks[0].forward).parameters)
    batches: list = []

    def catch(_module, args, kwargs):
        batches.append(((args[0].detach(),), dict(zip(names[1:], args[1:], strict=False)) | kwargs))

    handle = blocks[0].register_forward_pre_hook(catch, with_kwargs=True)
    try:
        with torch.no_grad():
            for probe in probes:
                raw(*probe)
    finally:
        handle.remove()
    return tuple(batches)


def i4_args(
    *,
    no_calib: bool = False,
    model: str = ANIMA_TURBO_MODEL_NAME,
    adaln_i8: bool = False,
    calib_device: str = CALIB_SHIPPABLE_DEVICE,
) -> argparse.Namespace:
    return argparse.Namespace(
        dtype="i4",
        no_calib=no_calib,
        calib_prompts=2,
        repo="unused",
        model=model,
        i4_adaln_i8=adaln_i8,
        calib_device=calib_device,
    )


@pytest.fixture
def stub_capture(monkeypatch):
    """Replace `capture_stage_batches` (the real one runs the text front end + the reference
    denoise).

    The replacement actually runs **the plain model** (`wrapper.model`) to capture, so both the
    shape of `(args, kwargs)` and the capture source stay the same as the real ones. It returns
    `(prompt count per call, weights at capture time)`; the latter is a copy that lets the caller
    claim "which weights were already rounded when the calibration inputs were taken" (the
    batches themselves are taken at install time, but the stub copies the weights **at the very
    moment the calibration calls it**).
    """
    calls: list[int] = []
    at_capture: dict[str, torch.Tensor] = {}

    def install(wrapper: nn.Module, probes=None) -> tuple[list[int], dict[str, torch.Tensor]]:
        batches = capture_batches(wrapper.model, probes if probes is not None else make_probes())

        def stub(model, prompts, **kwargs):
            calls.append(len(prompts))
            at_capture.update({name: p.detach().clone() for name, p in wrapper.named_parameters()})
            return batches

        monkeypatch.setattr(calib, "capture_stage_batches", stub)
        return calls, at_capture

    return install


@pytest.fixture
def stub_text_stack(monkeypatch):
    """Replace **only** the text front end (Qwen3 + conditioner) (the reference denoise runs for
    real).

    The return value is the sequence of prompts `encode_prompt` received — to check, from the
    call side too, that the uncond embedding really is built from the negative prompt.
    """
    seen: list[str] = []

    def load_text_stack(repo: str, dtype: str) -> str:
        return f"stack:{repo}:{dtype}"

    def encode_prompt(_stack: str, _max_len: int, text: str) -> dict[str, torch.Tensor]:
        seen.append(text)
        return {"encoder_hidden_states": embed_for(text)}

    monkeypatch.setattr(calib.pipeline_ref, "load_text_stack", load_text_stack)
    monkeypatch.setattr(calib.pipeline_ref, "encode_prompt", encode_prompt)
    return seen


def capture(model: nn.Module, prompts, conditions: calib.CalibConditions):
    """{@link anima.calib.capture_stage_batches} on the miniature DiT (only the resolution is
    reduced)."""
    return calib.capture_stage_batches(
        model, prompts, repo="unused", conditions=conditions, resolution=CAPTURE_RESOLUTION
    )


def quantize(wrapper: nn.Module, args: argparse.Namespace | None = None):
    return export_anima._fake_quant(
        args if args is not None else i4_args(),
        wrapper,
        "transformer",
        calib_probe=make_probes(count=1, seed=3)[0],
    )


class TestCalibConditions:
    """The calibration conditions are **derived** from the distribution's defaults (iteration J-4
    (2) — 2026-08-23).

    When they were module constants assuming turbo (8 steps / CFG 1), this one point blocked the
    base model's i4 entry. Now that they are derived, two things matter: "turbo has not moved by
    a single bit" and "the base model runs with its own defaults".
    """

    def test_turbo_derives_what_the_module_constants_used_to_hold(self) -> None:
        """MUST: the calibration conditions of the turbo line stay as they are (if the rounding
        results moved, they would disagree with the shipped assets)."""
        assert TURBO_LIKE_MODELS, "no distilled model at all (the test checks nothing)"
        for model in TURBO_LIKE_MODELS:
            conditions = calib.calib_conditions(model)

            assert (conditions.steps, conditions.guidance) == (8, 1.0), model
            assert conditions.branches == 1, "CFG=1 does not compute the uncond branch"

    def test_the_cfg_models_derive_their_cfg_conditions(self) -> None:
        """Everything but Turbo is many steps + CFG — one step is 2 forwards.

        The step counts copy the visual rulings (base line = 20 (2026-08-22) / aesthetic line =
        30 (2026-09-01)) — reading them again from the pipeline_config they are derived from
        would make the test vacuous, so the expected values are written here as constants.
        """
        expected_steps = {name: 30 if "aesthetic" in name else 20 for name in CFG_MODELS}
        assert CFG_MODELS, "no model uses CFG (the test checks nothing)"
        for model in CFG_MODELS:
            conditions = calib.calib_conditions(model)

            assert (conditions.steps, conditions.guidance) == (expected_steps[model], 4.0), model
            assert conditions.branches == 2, model
            assert conditions.negative_prompt, model

    def test_every_model_reads_its_own_pipeline_config(self) -> None:
        """MUST: the calibration conditions and the shipping defaults are not held independently
        (with a copy, one of them would go stale)."""
        for model, spec in ANIMA_MODELS.items():
            defaults = spec.pipeline_config["defaults"]
            conditions = calib.calib_conditions(model)

            assert conditions.steps == defaults["steps"], model
            assert conditions.guidance == defaults["guidanceScale"], model
            assert conditions.negative_prompt == defaults["negativePrompt"], model

    def test_it_follows_a_change_to_the_shipped_defaults(self, monkeypatch) -> None:
        """Fault injection: moving the shipping defaults moves the calibration conditions too (going
        back to constants would not be green)."""
        spec = ANIMA_MODELS[ANIMA_TURBO_MODEL_NAME]
        moved = replace(
            spec,
            pipeline_config={
                "defaults": {**spec.pipeline_config["defaults"], "steps": 12, "guidanceScale": 2.5}
            },
        )
        monkeypatch.setitem(ANIMA_MODELS, ANIMA_TURBO_MODEL_NAME, moved)

        conditions = calib.calib_conditions(ANIMA_TURBO_MODEL_NAME)

        assert (conditions.steps, conditions.guidance, conditions.branches) == (12, 2.5, 2)


class TestCaptureBranches:
    """Capture with CFG > 1 — the activations that go through the DiT when the distribution runs
    are **both** cond and uncond.

    Narrowing to cond would choose rounding targets from "half of the activations that actually
    flow", and since the counts still add up, it could be read neither from the assets nor from
    the diagnostic lines. The branches are made by `pipeline_ref.reference_steps` (run for real);
    what this checks is how the capture and the cross-check count its result.
    """

    def test_cfg_one_captures_one_batch_per_step(self, stub_text_stack) -> None:
        conditions = calib.CalibConditions(steps=3, guidance=1.0, negative_prompt="neg")

        batches = capture(DenoiseDit(), ("a", "b"), conditions)

        assert len(batches) == 2 * 3
        assert stub_text_stack == ["a", "b", "neg"], "the uncond embedding is always prepared"

    def test_cfg_above_one_captures_both_branches(self, stub_text_stack) -> None:
        conditions = calib.CalibConditions(steps=3, guidance=4.0, negative_prompt="neg")

        batches = capture(DenoiseDit(), ("a", "b"), conditions)

        assert len(batches) == 2 * 3 * 2

    def test_the_uncond_branch_carries_the_negative_prompt(self, stub_text_stack) -> None:
        """MUST: do not feed uncond the same embedding as cond.

        Doing so makes `uncond == cond`, CFG degenerates into the identity, and the same
        activations pile up in an H with **only the count doubled**. The batch-count cross-check
        passes, the rounding completes, and the record says CFG 4.
        """
        conditions = calib.CalibConditions(steps=2, guidance=4.0, negative_prompt="neg")

        batches = capture(DenoiseDit(), ("a",), conditions)

        embeds = [kwargs["encoder_hidden_states"] for _args, kwargs in batches]
        assert sum(torch.equal(embed, embed_for("a")) for embed in embeds) == 2
        assert sum(torch.equal(embed, embed_for("neg")) for embed in embeds) == 2

    def test_the_turbo_conditions_capture_the_same_batches_as_before(self, stub_text_stack) -> None:
        """After the derivation, turbo is still prompts × 8 steps × 1 branch (the same count as when
        they were constants)."""
        batches = capture(DenoiseDit(), ("a", "b"), calib.calib_conditions(ANIMA_TURBO_MODEL_NAME))

        assert len(batches) == 2 * 8

    def test_a_branch_count_that_disagrees_with_the_capture_fails_loudly(
        self, monkeypatch, stub_text_stack
    ) -> None:
        """Cross-check: it stops when the expected branch count and the actual number of forwards
        disagree.

        The breakage is "uncond stops running even with CFG > 1" — only the samples accumulated
        in H halve; the rounding completes as usual and neither the storage form nor the counts
        change.
        """
        conditions = calib.CalibConditions(steps=2, guidance=4.0, negative_prompt="neg")
        genuine = calib.pipeline_ref.reference_steps
        monkeypatch.setattr(
            calib.pipeline_ref, "reference_steps", lambda *args: genuine(*args[:-1], 1.0)
        )

        with pytest.raises(AssertionError, match="calibration batches, not prompts"):
            capture(DenoiseDit(), ("a",), conditions)


class TestModelSelection:
    """`--model` is where the calibration conditions come from (no default — it does not silently
    run with turbo's conditions)."""

    def test_the_calibration_refuses_to_run_without_a_named_model(self, monkeypatch, capsys):
        """MUST: with a default, a forgotten `--model` would quietly produce "a base model
        calibrated with turbo's conditions"."""
        monkeypatch.setattr("sys.argv", ["export.py", "--dtype", "i4"])

        with pytest.raises(SystemExit):
            export_anima.main()

        assert "needs --model" in capsys.readouterr().err

    def test_the_knob_is_refused_where_no_calibration_runs(self, monkeypatch, capsys):
        """A knob with no effect is not silently accepted (`--no-calib` has no conditions to
        take)."""
        monkeypatch.setattr(
            "sys.argv",
            ["export.py", "--dtype", "i4", "--no-calib", "--model", ANIMA_TURBO_MODEL_NAME],
        )

        with pytest.raises(SystemExit):
            export_anima.main()

        assert "--model is a knob only" in capsys.readouterr().err

    def test_the_export_path_hands_the_models_conditions_to_the_capture(self, monkeypatch):
        """MUST: the conditions the capture receives are those of `--model` (going back to fixed
        turbo values would not be green)."""
        wrapper = make_wrapper()
        batches = capture_batches(wrapper.model, make_probes())
        seen: dict = {}

        def stub(model, prompts, **kwargs):
            seen.update(kwargs)
            return batches

        monkeypatch.setattr(calib, "capture_stage_batches", stub)

        quantize(wrapper, i4_args(model=ANIMA_BASE_MODEL_NAME))

        assert seen["conditions"] == calib.calib_conditions(ANIMA_BASE_MODEL_NAME)


class TestStageSplit:
    def test_the_stage_chain_reproduces_the_block_loop_bit_exactly(self):
        """MUST: the decomposition does not differ from the real path by a single bit (otherwise
        GPTQ of another path ships)."""
        wrapper = make_wrapper()

        batch = calib.assert_stage_split(
            wrapper, make_probes(count=1)[0], calib.dit_stages(wrapper)
        )

        assert set(batch[1]) == {
            "encoder_hidden_states",
            "embedded_timestep",
            "temb",
            "image_rotary_emb",
            "extra_pos_emb",
            "attention_mask",
            "controlnet_residual",
        }

    def test_a_dropped_stage_is_caught(self):
        """Dropping one stage changes the final hidden (measures that the check does not let it
        through)."""
        wrapper = make_wrapper()
        stages = calib.dit_stages(wrapper)

        with pytest.raises(AssertionError, match="is not bit-identical"):
            calib.assert_stage_split(wrapper, make_probes(count=1)[0], stages[:-1])

    def test_a_reordered_stage_chain_is_caught(self):
        """A swapped order still has matching shapes (a breakage where only the values change,
        silently)."""
        wrapper = make_wrapper()
        stages = calib.dit_stages(wrapper)
        swapped = (stages[1], stages[0], *stages[2:])

        with pytest.raises(AssertionError, match="is not bit-identical"):
            calib.assert_stage_split(wrapper, make_probes(count=1)[0], swapped)

    def test_the_stage_prefix_lands_in_the_wrapper_fqn_space(self):
        """The prefixed local FQNs match the wrapper's real FQNs (the place that decides the space
        of the ledger keys)."""
        wrapper = make_wrapper(blocks=2)

        names = calib.stage_linear_names(calib.dit_stages(wrapper))

        assert names == {
            f"model.transformer_blocks.{index}.{child}"
            for index in range(2)
            for child in (
                "attn",
                "ff",
                *(f"norm{at}.linear_{part}" for at in (1, 2, 3) for part in (1, 2)),
            )
        }
        assert names <= {name for name, _module in wrapper.named_modules()}


class TestBatchesMatchGraph:
    def test_the_raw_model_and_the_wrapper_call_the_block_alike(self):
        """The plain model (the capture source) and the wrapper (what gets rounded) call the blocks
        with the same set of arguments.

        It also checks that the 2 paths are **separate implementations of the same function**
        (bit-identical outputs) — if this broke, the drift injection below would not mirror "a
        real disagreement".
        """
        wrapper = make_wrapper()
        probe = make_probes(count=1)[0]
        stages = calib.dit_stages(wrapper)
        graph_batch = calib.assert_stage_split(wrapper, probe, stages)

        calib.assert_calib_batches_match_graph(
            graph_batch, capture_batches(wrapper.model, make_probes())
        )

        with torch.no_grad():
            assert torch.equal(wrapper(*probe), wrapper.model(*probe))

    def test_a_batch_with_a_different_keyword_set_is_caught(self):
        """If the reference denoise and the wrapper call the blocks differently, rounding targets
        chosen under other conditions ship."""
        wrapper = make_wrapper()
        stages = calib.dit_stages(wrapper)
        graph_batch = calib.assert_stage_split(wrapper, make_probes(count=1)[0], stages)
        batches = capture_batches(wrapper.model, make_probes(count=1))
        args, kwargs = batches[0]
        drifted = ((args, {**kwargs, "img_context": None}),)

        with pytest.raises(AssertionError, match="call the blocks differently"):
            calib.assert_calib_batches_match_graph(graph_batch, drifted)

    def test_the_export_path_runs_the_gate(self, stub_capture):
        """MUST: the check is really wired into `_round_i4_calibrated` (the function alone is not
        enough).

        The breakage is "the plain model drops one companion argument of the block" — the
        numbers come out as usual, so without the check, rounding targets chosen under other
        conditions would ship as they are.
        """
        wrapper = make_wrapper(dit=DriftingRawDit)
        stub_capture(wrapper)

        with pytest.raises(AssertionError, match="call the blocks differently"):
            quantize(wrapper)


class TestCalibratedI4:
    def test_the_ledger_keys_live_in_the_wrapper_fqn_space(self, stub_capture):
        """MUST: ledger keys = the container's initializer names (emit's comparison is decided
        here)."""
        wrapper = make_wrapper()
        stub_capture(wrapper)

        scales, overrides = quantize(wrapper)

        parameters = dict(wrapper.named_parameters())
        assert set(overrides) <= set(parameters), (
            "the keys of the i4 entries are not in the wrapper's FQN space"
        )
        assert set(scales) <= set(parameters)
        assert set(overrides) == {
            f"{name}.weight" for name in export_anima._i4_module_names(wrapper)
        }

    def test_the_eligible_set_is_split_exclusively_between_the_two_paths(self):
        """Inside blocks (calibrated) and outside blocks (RTN) are **exact and exclusive** (the
        mixing MUST of `quantize.py`)."""
        wrapper = make_wrapper()
        stage_names = calib.stage_linear_names(calib.dit_stages(wrapper))
        i4_names = export_anima._i4_module_names(wrapper)

        assert i4_names - stage_names == export_anima.NON_STAGE_I4_WEIGHTS
        assert stage_names <= i4_names, (
            "a linear inside a block is missing from the i4-eligible set"
        )
        # The quantization axis of the patchify entry is not divisible by g32 — it does not go
        # to i4 (it stays with i8)
        assert "model.patch_embed.proj" not in i4_names

    def test_calibration_produces_a_different_rounding_from_plain_rtn(self, stub_capture):
        """If calibration passed through, the storage form would be the same and could not be read
        from the assets — measured by the difference in values."""
        calibrated = make_wrapper()
        plain = make_wrapper()
        stub_capture(calibrated)

        quantize(calibrated)
        quantize(plain, i4_args(no_calib=True))

        after = dict(calibrated.named_parameters())
        differing = sorted(
            name
            for name, weight in plain.named_parameters()
            if not torch.equal(weight, after[name])
        )
        assert differing, "GPTQ gave the same values as RTN (the calibration had no effect)"
        assert all(name.startswith("model.transformer_blocks.") for name in differing)

    def test_the_capture_sees_the_shipped_condition(self, stub_capture):
        """MUST: the calibration inputs are taken **after** rounding outside the blocks and
        **before** rounding inside them.

        The second half (inside the blocks still original) catches "the circularity of choosing
        a weight's rounding targets from activations those rounded weights produced"; the first
        half (outside the blocks already rounded) catches "shipping rounding targets chosen
        around f32 surroundings together with i4 surroundings".
        """
        wrapper = make_wrapper()
        before = {name: p.detach().clone() for name, p in wrapper.named_parameters()}
        _calls, at_capture = stub_capture(wrapper)

        quantize(wrapper)

        for name in export_anima.NON_STAGE_I4_WEIGHTS:
            key = f"{name}.weight"
            assert not torch.equal(before[key], at_capture[key]), (
                f"{key} was still original at capture time"
            )
        for key in (name for name in before if name.startswith("model.transformer_blocks.")):
            assert torch.equal(before[key], at_capture[key]), (
                f"{key} was rounded before the capture"
            )

    def test_the_capture_sees_the_i8_side_already_rounded(self, stub_capture):
        """MUST: the i8 side (the patchify entry) is also rounded **before** the capture.

        `patch_embed.proj` is the only linear that does not go to i4 because its quantization
        axis is not divisible by g32 (one in the real DiT too), but **it builds the very input
        of block 0**. Rounding it after the capture would ship rounding targets chosen from
        "activations that went through an f32 patchify" together with an i8 patchify — the
        reason for rounding the out-of-block i4 first (ordering MUST (2)) would fail to apply to
        this one linear only.
        """
        wrapper = make_wrapper()
        before = {name: p.detach().clone() for name, p in wrapper.named_parameters()}
        _calls, at_capture = stub_capture(wrapper)

        quantize(wrapper)

        key = "model.patch_embed.proj.weight"
        assert key in before, "the miniature has no i8-side weight (the test checks nothing)"
        assert not torch.equal(before[key], at_capture[key]), (
            f"{key} was still original at capture time"
        )

    def test_the_out_of_block_weights_are_rounded_by_the_plain_path_in_both_modes(
        self, stub_capture
    ):
        """Outside the blocks both paths use plain RTN (the rounded values are bit-identical)."""
        calibrated = make_wrapper()
        plain = make_wrapper()
        stub_capture(calibrated)

        quantize(calibrated)
        quantize(plain, i4_args(no_calib=True))

        after = dict(calibrated.named_parameters())
        for name in export_anima.NON_STAGE_I4_WEIGHTS:
            key = f"{name}.weight"
            assert torch.equal(dict(plain.named_parameters())[key], after[key]), key

    def test_an_unexpected_non_stage_eligible_fails_loudly(self, monkeypatch, stub_capture):
        """Out-of-block eligibles differing from the declaration = a sign that upstream's
        architecture moved (the classification does not change silently)."""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        monkeypatch.setattr(export_anima, "NON_STAGE_I4_WEIGHTS", frozenset())

        with pytest.raises(AssertionError, match="i4-eligible weights outside the DiT blocks"):
            quantize(wrapper)

    def test_a_block_linear_outside_the_i4_set_fails_loudly(self, monkeypatch, stub_capture):
        """A non-eligible linear inside a block makes the check "scanned count = rounded count"
        impossible."""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        full = export_anima._i4_module_names(wrapper)
        dropped = frozenset(full - {"model.transformer_blocks.0.attn"})
        monkeypatch.setattr(export_anima, "_i4_module_names", lambda model: dropped)

        with pytest.raises(AssertionError, match=r"are not i4-eligible \(quantization axis"):
            quantize(wrapper)

    def test_a_ledger_that_overlaps_the_plain_path_fails_loudly(self, monkeypatch, stub_capture):
        """Rounding the same weight on 2 paths silently corrupts only the values (the storage form
        and the counts look right)."""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        genuine = calib.calibrate_i4
        outside = f"{sorted(export_anima.NON_STAGE_I4_WEIGHTS)[0]}.weight"

        def overlapping(rig, include=None):
            report, ledger = genuine(rig, include)
            merged = dict(ledger.scales) | {outside: torch.ones(HIDDEN, 1)}
            return report, replace(ledger, scales=merged)

        monkeypatch.setattr(calib, "calibrate_i4", overlapping)

        with pytest.raises(AssertionError, match="double rounding"):
            quantize(wrapper)

    def test_a_report_short_of_the_scan_fails_loudly(self, monkeypatch, stub_capture):
        """A missed rounding makes the quality look **better** (the pass-through cannot be read from
        the numbers)."""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        genuine = calib.calibrate_i4

        def short(rig, include=None):
            report, ledger = genuine(rig, include)
            return replace(report, layers=report.layers[:-1]), ledger

        monkeypatch.setattr(calib, "calibrate_i4", short)

        with pytest.raises(AssertionError, match="of the scan"):
            quantize(wrapper)

    def test_zero_calibration_prompts_fail_loudly(self):
        """MUST: there is no branch that silently falls back to plain RTN."""
        wrapper = make_wrapper()

        with pytest.raises(AssertionError, match="No calibration prompts"):
            capture(wrapper.model, (), calib.calib_conditions(ANIMA_TURBO_MODEL_NAME))


class TestOptOut:
    def test_the_i4_series_is_calibrated_by_default(self, monkeypatch):
        """MUST: the CLI's `--dtype i4` is calibrated without any flag (`--no-calib` is the only
        opt-out)."""
        captured: dict[str, argparse.Namespace] = {}

        def stub(target, args, out_dir):
            captured[target] = args
            return {"target": target}

        monkeypatch.setattr(
            "sys.argv", ["export.py", "--dtype", "i4", "--model", ANIMA_TURBO_MODEL_NAME]
        )
        monkeypatch.setattr(export_anima, "emit_target", stub)

        export_anima.main()

        assert captured["transformer"].no_calib is False
        assert captured["transformer"].calib_prompts == export_anima.DEFAULT_CALIB_PROMPTS

    def test_no_calib_never_touches_the_calibration_path(self, stub_capture):
        """The opt-out means calibration **does not run**, not "calibration gets thinner" (both cost
        and quality are different things)."""
        wrapper = make_wrapper()
        calls, _at_capture = stub_capture(wrapper)

        quantize(wrapper, i4_args(no_calib=True))

        assert calls == []

    def test_the_calibration_knobs_are_refused_outside_i4(self, monkeypatch):
        """A knob with no effect is not silently accepted (f16 has no calibration path at all)."""
        monkeypatch.setattr("sys.argv", ["export.py", "--dtype", "f16", "--no-calib"])

        with pytest.raises(SystemExit):
            export_anima.main()

    def test_a_resolution_off_the_calibration_condition_is_refused(self, monkeypatch, capsys):
        """MUST: rounding targets chosen at 512px are not baked into a graph of another resolution
        (a breakage where the numbers come out as usual)."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "export.py",
                "--dtype",
                "i4",
                "--model",
                ANIMA_TURBO_MODEL_NAME,
                "--resolution",
                "1024",
            ],
        )

        with pytest.raises(SystemExit):
            export_anima.main()

        assert f"fixed at {calib.CALIB_RESOLUTION}px" in capsys.readouterr().err

    def test_no_calib_leaves_the_resolution_free(self, monkeypatch):
        """The constraint belongs to the calibration path — plain RTN (`--no-calib`) requires
        nothing of the resolution."""
        captured: dict[str, argparse.Namespace] = {}

        def stub(target, args, out_dir):
            captured[target] = args
            return {"target": target}

        monkeypatch.setattr(
            "sys.argv", ["export.py", "--dtype", "i4", "--no-calib", "--resolution", "1024"]
        )
        monkeypatch.setattr(export_anima, "emit_target", stub)

        export_anima.main()

        assert captured["transformer"].resolution == 1024


class TestAdalnSegments:
    """adaLN is decided by **segment match** (the elements split by `.`) — neither substring nor
    suffix."""

    def test_it_matches_a_segment_and_not_a_substring(self):
        assert calib.is_adaln("norm1.linear_1")
        assert calib.is_adaln("norm3.linear_2")
        # A substring match would pull in spellings like `norm1x`, and a suffix match would miss
        # children under adaLN — either way "the count meant to be excluded" drifts silently.
        assert not calib.is_adaln("norm1x.linear_1")
        assert not calib.is_adaln("norm_out.linear_1")
        assert not calib.is_adaln("attn")

    def test_it_answers_alike_for_stage_local_and_wrapper_fqns(self):
        """MUST: the calibration's `include` (local names) and export's set split (FQNs within the
        wrapper) share one implementation."""
        assert calib.is_adaln("norm2.linear_1")
        assert calib.is_adaln("model.transformer_blocks.7.norm2.linear_1")

    def test_it_reports_which_spellings_were_seen(self):
        """A net that catches a rename on one side only — a check that looks only at "none at all"
        lets it through."""
        names = {"model.transformer_blocks.0.norm1.linear_1", "model.transformer_blocks.0.attn"}

        assert calib.adaln_segments_seen(names) == frozenset({"norm1"})
        assert calib.adaln_segments_seen(()) == frozenset()


class TestAdalnI8Variant:
    """`--i4-adaln-i8` (off by default) = the sensitivity-experiment variant that moves adaLN + the
    outside of the blocks to i8 storage.

    Checks both that the default does not move by a single bit (set, path, values) and that the
    variant **really** splits by role. Even in the variant the storage form stays "a series that
    mixes i4 and i8", so the assets cannot tell which way it was baked — every flaw in the split
    has to be caught here.
    """

    @staticmethod
    def _split(wrapper: nn.Module) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
        """`(eligible, adaLN inside blocks, non-adaLN inside blocks)`."""
        eligible = export_anima._i4_module_names(wrapper)
        stage_names = calib.stage_linear_names(calib.dit_stages(wrapper))
        adaln = frozenset(name for name in stage_names if calib.is_adaln(name))
        return eligible, adaln, stage_names - adaln

    def test_the_default_stores_every_eligible_weight_as_i4(self, stub_capture):
        """The storage set of the default (off) = all eligibles (the variant's plumbing does not
        touch the default split)."""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        eligible, _adaln, _rest = self._split(wrapper)

        _scales, overrides = quantize(wrapper)

        assert set(overrides) == {f"{name}.weight" for name in eligible}

    def test_the_default_hands_no_include_predicate_to_the_calibration(self, stub_capture):
        """MUST: the default has **no** predicate (`None`) — a predicate that re-spells "everything"
        would, the day the scan's definition moves, follow on one side only and drift
        silently."""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        genuine = calib.calibrate_i4
        seen: list[object] = []

        def spy(rig, include=None):
            seen.append(include)
            return genuine(rig, include)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(calib, "calibrate_i4", spy)
            quantize(wrapper)

        assert seen == [None]

    def test_the_default_matches_an_independently_expressed_plain_rounding(self):
        """The default (off, `--no-calib`) is **bit-identical** to "all eligibles by plain RTN → the
        rest to i8".

        The variant's plumbing inserts just one set subtraction, and by default the subtracted
        side is empty — this measures that it holds on the values too, by comparing with a
        direct rounding that does not go through the script (the storage form is the same in the
        variant, so a drift could not be read from the assets).
        """
        through_export = make_wrapper()
        direct = make_wrapper()
        eligible = export_anima._i4_module_names(direct)

        quantize(through_export, i4_args(no_calib=True))
        fake_quant_int4(
            direct,
            include=lambda name: name in eligible,
            op_types=export_anima.I4_MODULE_TYPES,
        )
        fake_quant_int8(direct, include=lambda name: name not in eligible)

        after = dict(through_export.named_parameters())
        assert {name for name, _ in direct.named_parameters()} == set(after)
        for name, weight in direct.named_parameters():
            assert torch.equal(weight, after[name]), name

    def test_the_variant_keeps_only_the_non_adaln_block_linears_in_i4(self, stub_capture):
        wrapper = make_wrapper()
        stub_capture(wrapper)
        _eligible, _adaln, rest = self._split(wrapper)

        _scales, overrides = quantize(wrapper, i4_args(adaln_i8=True))

        assert set(overrides) == {f"{name}.weight" for name in rest}
        assert all(dtype == "i4" for dtype in overrides.values())

    def test_the_excluded_count_is_the_adaln_rows_plus_the_out_of_block_five(self):
        """The count: blocks × (3 segments × 2 linears) + the out-of-block
        {@link NON_STAGE_I4_WEIGHTS}.

        In the real model (anima-v1.0 = 28 blocks) this is 28 × 6 + 5 = **173** (adaLN 168 +
        5). The count is not hard-coded so that the same formula holds for a reduced export with
        `--num-layers`.
        """
        blocks = 4
        wrapper = make_wrapper(blocks=blocks)

        excluded = export_anima._adaln_i8_names(wrapper)

        per_block = len(calib.ADALN_SEGMENTS) * 2
        assert len(excluded) == blocks * per_block + len(export_anima.NON_STAGE_I4_WEIGHTS)
        assert excluded >= export_anima.NON_STAGE_I4_WEIGHTS
        assert len(excluded - export_anima.NON_STAGE_I4_WEIGHTS) == blocks * per_block

    def test_the_variant_rounds_the_excluded_weights_as_i8(self, stub_capture):
        """MUST: the excluded side is rounded as i8 (rules out slipping through both i4 and i8 and
        staying f32)."""
        wrapper = make_wrapper()
        before = {name: p.detach().clone() for name, p in wrapper.named_parameters()}
        stub_capture(wrapper)

        scales, overrides = quantize(wrapper, i4_args(adaln_i8=True))

        after = dict(wrapper.named_parameters())
        for name in export_anima._adaln_i8_names(wrapper):
            key = f"{name}.weight"
            assert key in scales, f"{key} has no scale in the ledger"
            assert key not in overrides, f"{key} has an i4 storage spec"
            # A per-channel scale is one per row (a group scale is rows × groups).
            assert list(scales[key].shape) == [after[key].shape[0], 1]
            assert not torch.equal(before[key], after[key]), f"{key} is not rounded"

    def test_the_calibration_rounds_only_the_non_adaln_block_linears(self, stub_capture):
        """The set that goes to GPTQ equals "inside blocks − adaLN" exactly (the predicate and the
        storage set are the same)."""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        _eligible, adaln, rest = self._split(wrapper)
        genuine = calib.calibrate_i4
        rounded: list[frozenset[str]] = []

        def spy(rig, include=None):
            report, ledger = genuine(rig, include)
            rounded.append(frozenset(layer.fqn for layer in report.layers))
            return report, ledger

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(calib, "calibrate_i4", spy)
            quantize(wrapper, i4_args(adaln_i8=True))

        assert rounded == [frozenset(f"{name}.weight" for name in rest)]
        assert adaln, "the miniature has no adaLN (the test checks nothing)"

    def test_the_capture_sees_the_adaln_side_already_rounded(self, stub_capture):
        """MUST: in the variant, adaLN is also rounded to i8 **before the capture**.

        When the distribution runs, the modulation is built by the i8 adaLN, so rounding it later
        would ship rounding targets chosen from "activations that went through an f32
        modulation" together with the i8 modulation (the same reason the outside of the blocks
        is rounded first — ordering MUST (2)).
        """
        wrapper = make_wrapper()
        before = {name: p.detach().clone() for name, p in wrapper.named_parameters()}
        _calls, at_capture = stub_capture(wrapper)
        _eligible, adaln, rest = self._split(wrapper)

        quantize(wrapper, i4_args(adaln_i8=True))

        for name in adaln:
            key = f"{name}.weight"
            assert not torch.equal(before[key], at_capture[key]), (
                f"{key} was still original at capture time"
            )
        for name in rest:
            key = f"{name}.weight"
            assert torch.equal(before[key], at_capture[key]), (
                f"{key} was rounded before the capture"
            )

    def test_the_plain_rtn_mode_splits_the_same_way(self):
        """`--no-calib` splits the same way (the variant is a storage split, orthogonal to whether
        calibration runs)."""
        wrapper = make_wrapper()
        _eligible, _adaln, rest = self._split(wrapper)

        _scales, overrides = quantize(wrapper, i4_args(no_calib=True, adaln_i8=True))

        assert set(overrides) == {f"{name}.weight" for name in rest}

    def test_a_renamed_adaln_segment_fails_loudly(self, monkeypatch, stub_capture):
        """MUST: it fails if even one spelling is not found — a rename on one side only gets past
        the count check."""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        monkeypatch.setattr(calib, "ADALN_SEGMENTS", calib.ADALN_SEGMENTS | {"norm4"})

        with pytest.raises(AssertionError, match="adaLN linear spellings with no match"):
            quantize(wrapper, i4_args(adaln_i8=True))

    def test_an_exclusion_outside_the_eligible_set_fails_loudly(self, monkeypatch, stub_capture):
        """An excluded spelling not in the eligible set = a sign that upstream's architecture moved
        (the subtraction must not silently hit nothing)."""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        monkeypatch.setattr(
            export_anima, "_adaln_i8_names", lambda model: frozenset({"model.nowhere"})
        )

        with pytest.raises(AssertionError, match="weights assigned to i8"):
            quantize(wrapper, i4_args(adaln_i8=True))

    def test_the_out_of_block_weights_leave_the_plain_i4_path_entirely(self, stub_capture):
        """In the variant the plain RTN i4 path **rounds nothing** (everything outside the blocks
        moves to i8)."""
        wrapper = make_wrapper()
        stub_capture(wrapper)
        calls: list[frozenset[str]] = []

        def spy(model, names, target, label):
            calls.append(frozenset(names))
            return export_anima._round_i4_plain(model, names, target, label)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(export_anima, "_round_i4_plain", spy)
            quantize(wrapper, i4_args(adaln_i8=True))

        assert calls == []
