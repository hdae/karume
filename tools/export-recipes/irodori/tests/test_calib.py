"""Wiring contracts of calibrated i4 (GPTQ) (the part that needs no real weights).

Calibration with the real weights is manual (`python -m irodori.export --dtype i4`). What is
pinned here is only what would turn into a **false PASS** if it broke:

- the stage-by-stage block loop is bit-identical to one forward of the plain DiT (rounding along
  a drifted path still produces ordinary-looking numbers)
- i4 eligibility is "only the linears in the shipped graph", and a weight not divisible by g32
  fails loudly
- the shipped graph's linears split **exactly and exclusively** into "inside the blocks except
  adaLN = i4 storage × calibration" and "outside the blocks + adaLN = i8 storage" (the listening
  ruling of 2026-08-23 took the outside of the blocks and adaLN out of i4)
- the adaLN match is **by segment** (`attention` does not drag in `attention_adaln`, and no child
  under adaLN is missed), and it fails loudly both on the day there is no adaLN at all **and on
  the day only one side was renamed**
- the keys of the scale ledger and of the per-weight storage overrides live in the wrappers' FQN
  space (= the safetensors tensor keys)
- calibration **actually produces a different rounding** (if it passed through untouched, the
  storage form is the same, so it cannot be read off the assets)
- calibration inputs are taken from **weights in the same state as the shipped condition** (after
  all of i8 is rounded, before the i4 side is rounded)
- a change of the arguments of upstream `DiffusionBlock.forward` fails (the cond-side slicing
  would silently come off)
- the context plan upstream builds for the whole batch is not replayed against the cond row (the
  block rebuilds it from the sliced masks)
- running with a condition K/V cache fails (one kwargs cannot be reused across all stages)
- `--no-calib` really does not run calibration, as an opt-out

The model here is a tiny skeleton that copies **the real FQNs** (`blocks.<i>` / `in_proj` /
`out_proj` / `cond_module.<i>`, the condition-side projections `wk_text` … of attention and
`k_norm`) and **the same call shape** (a block takes all 12 arguments as keywords and returns the
hidden state; the DiT holds copies of the other roles inside). Even the names are copied so that
the checks run against the predicates meant for the real model without replacing them
({@link irodori.calib.dit_i4_names} takes the intersection with the union of what the two DiT
distribution wrappers own [`dit-context` / `dit` — ADR 0114]).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from irodori import calib
from irodori import export as ir
from irodori import patch as ir_patch
from irodori.calib_cases import CALIB_CASES
from karume.artifacts import staged_publication
from karume.quantize import channel_scale, quantize_to_int8

#: Quantization axis (= group length). i4 makes no partial group (ADR 0069 decision 2).
HIDDEN = 32
#: Width of the condition state (the cross-attention side of a block — the same as hidden to
#: keep the copies few).
CONTEXT = 32
#: A width not divisible by g32 (a number that exists only to hit the eligibility check's
#: fail-loudly path).
UNALIGNED = 20
#: Number of latent tokens.
TOKENS = 4
#: Batch rows of a captured batch (a miniature of upstream independent CFG bundling cond +
#: uncond).
CFG_ROWS = 3


class TinyNorm(nn.Module):
    """The place of an `RMSNorm` (it has a weight but is not a quantized type — it goes into
    neither i8 nor i4)."""

    def __init__(self, width: int = HIDDEN) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class TinyAttention(nn.Module):
    """The children of `JointAttention` (2 on the self side + 6 condition-side projections + the
    shared `k_norm`).

    The condition-side projections and `k_norm` are there so that the model also exercises the
    ownership split after the DiT was divided into `dit-context` / `dit` (ADR 0114)
    (`irodori.export.CONTEXT_KV_PROJECTIONS`) — without the real names `DitContextGraph` cannot
    be built.
    """

    def __init__(self, width: int = HIDDEN) -> None:
        super().__init__()
        self.wq = nn.Linear(width, HIDDEN, bias=False)
        self.wo = nn.Linear(HIDDEN, HIDDEN, bias=False)
        for name in ir.CONTEXT_KV_PROJECTIONS:
            setattr(self, name, nn.Linear(CONTEXT, HIDDEN, bias=False))
        self.k_norm = TinyNorm(HIDDEN)

    def context(
        self, text_state: torch.Tensor, speaker_state: torch.Tensor, caption_state: torch.Tensor
    ) -> torch.Tensor:
        """A term that goes through all 6 condition-side projections (fails if calibration
        input is not captured for any one of them)."""
        pooled = torch.zeros(())
        for prefix, state in (
            ("text", text_state),
            ("speaker", speaker_state),
            ("caption", caption_state),
        ):
            keys = getattr(self, f"wk_{prefix}")(state)
            values = getattr(self, f"wv_{prefix}")(state)
            pooled = (
                pooled
                + self.k_norm(keys).mean(dim=1, keepdim=True)
                + values.mean(dim=1, keepdim=True)
            )
        return pooled


class TinyBlock(nn.Module):
    """A block that copies only the **call shape** of upstream `DiffusionBlock`.

    It takes all 12 arguments as keywords and returns the hidden state — the same as the real
    one, so the stages can be chained as is without wrapping (`irodori.calib.dit_stages`).

    `context_plan` stands in for upstream's `ContextAttentionPlan` by its batch-shaped content
    (the concatenated key mask `(B, S_kv)`): like the real one it belongs to the batch it was built
    for, so a plan replayed against a different number of rows fails, and `None` means "build it
    from the masks received" (upstream `JointAttention.forward`'s fallback).

    adaLN uses the real attribute names (`attention_adaln` / `mlp_adaln`) in the form that **has a
    linear as a child** (`…_adaln.1`) — so that the model side also exercises the match being by
    segment (neither by leaf nor by suffix).
    """

    def __init__(self, attention_width: int = HIDDEN) -> None:
        super().__init__()
        self.attention = TinyAttention(attention_width)
        self.mlp = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.attention_adaln = nn.Sequential(nn.SiLU(), nn.Linear(HIDDEN, HIDDEN, bias=False))
        self.mlp_adaln = nn.Sequential(nn.SiLU(), nn.Linear(HIDDEN, HIDDEN, bias=False))

    def forward(
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor,
        speaker_mask: torch.Tensor,
        caption_state: torch.Tensor,
        caption_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        context_kv: tuple[torch.Tensor, ...] | None = None,
        context_plan: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_plan is not None and context_plan.shape[0] != x.shape[0]:
            raise ValueError(
                f"a context plan built for {context_plan.shape[0]} rows was replayed against"
                f" {x.shape[0]}"
            )
        context = self.attention.context(text_state, speaker_state, caption_state)
        attended = torch.tanh(self.attention.wq(x) + context)
        return x + self.attention.wo(self.mlp(attended)) * self.modulation(cond_embed)

    def modulation(self, cond_embed: torch.Tensor) -> torch.Tensor:
        return self.attention_adaln(cond_embed) + self.mlp_adaln(cond_embed)


class TinyDit(nn.Module):
    """Skeleton of `TextToLatentRFDiT` (copied down to holding copies of the other roles
    inside)."""

    def __init__(self, blocks: int = 3, block_type: type[TinyBlock] = TinyBlock) -> None:
        super().__init__()
        self.cond_module = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN, bias=False),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN, bias=False),
        )
        self.in_proj = nn.Linear(HIDDEN, HIDDEN)
        self.blocks = nn.ModuleList(block_type() for _ in range(blocks))
        self.out_norm = TinyNorm()
        self.out_proj = nn.Linear(HIDDEN, HIDDEN)
        self.text_norm = TinyNorm(CONTEXT)
        self.caption_norm = TinyNorm(CONTEXT)
        self.head_dim = 8
        # A copy of another role held inside the DiT (not in `DitGraph` = not i4-eligible).
        self.pretrained_text_backbone = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward_with_encoded_conditions(self, x_t: torch.Tensor, **conditions: Any) -> torch.Tensor:
        """The same shape as upstream: "call every block with all 12 arguments as
        keywords"."""
        x = self.in_proj(x_t)
        for block in self.blocks:
            x = block(x=x, **conditions)
        return self.out_proj(self.out_norm(x))


class UnalignedDit(TinyDit):
    """The DiT on the day one linear inside a block became not divisible by g32."""

    def __init__(self, blocks: int = 3) -> None:
        super().__init__(blocks)
        self.blocks[0].attention.wq = nn.Linear(UNALIGNED, HIDDEN, bias=False)


class DriftingBlock(TinyBlock):
    """The block on the day upstream dropped one of the block's accompanying arguments."""

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor,
        speaker_mask: torch.Tensor,
        caption_state: torch.Tensor,
        caption_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        context_plan: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().forward(
            x,
            cond_embed,
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
            freqs_cis,
            self_mask,
            context_plan=context_plan,
        )


class DriftingDit(TinyDit):
    """Upstream with one accompanying argument fewer (the numbers come out normally, so
    without the check nobody would notice)."""

    def __init__(self, blocks: int = 3) -> None:
        super().__init__(blocks, block_type=DriftingBlock)

    def forward_with_encoded_conditions(self, x_t: torch.Tensor, **conditions: Any) -> torch.Tensor:
        conditions.pop("context_kv", None)
        return super().forward_with_encoded_conditions(x_t, **conditions)


class RenamedAdalnBlock(TinyBlock):
    """The block on the day upstream renamed the modulation attributes (no adaLN spelling
    anywhere).

    Not a single op of the computation changes — only the FQNs do, so without the check the 144
    weights would silently go back to i4 (with the storage form and the count check still right,
    the configuration the ruling rejected would ship).
    """

    def __init__(self, attention_width: int = HIDDEN) -> None:
        super().__init__(attention_width)
        self.attention_modulation = self.attention_adaln
        self.mlp_modulation = self.mlp_adaln
        del self.attention_adaln
        del self.mlp_adaln

    def modulation(self, cond_embed: torch.Tensor) -> torch.Tensor:
        return self.attention_modulation(cond_embed) + self.mlp_modulation(cond_embed)


class RenamedAdalnDit(TinyDit):
    def __init__(self, blocks: int = 3) -> None:
        super().__init__(blocks, block_type=RenamedAdalnBlock)


class HalfRenamedAdalnBlock(TinyBlock):
    """The block on the day upstream renamed **only one side** of the modulation (only
    `mlp_adaln` disappears).

    The set does not become empty, so a check that only looks for "no adaLN at all" passes, and
    the exact-coverage check of i4 builds both sides from the same classifier, so it stays
    self-consistent and green — with not a single net to catch it, half of what was moved back
    to i8 goes back to i4.
    """

    def __init__(self, attention_width: int = HIDDEN) -> None:
        super().__init__(attention_width)
        self.mlp_modulation = self.mlp_adaln
        del self.mlp_adaln

    def modulation(self, cond_embed: torch.Tensor) -> torch.Tensor:
        return self.attention_adaln(cond_embed) + self.mlp_modulation(cond_embed)


class HalfRenamedAdalnDit(TinyDit):
    def __init__(self, blocks: int = 3) -> None:
        super().__init__(blocks, block_type=HalfRenamedAdalnBlock)


class CachedKvDit(TinyDit):
    """The DiT on the day it was run with upstream's default (`use_context_kv_cache=True`).

    Each block receives a different `context_kv`, which does not mesh with the core's driver
    reusing one kwargs across all stages (and the numbers still come out normally).
    """

    def forward_with_encoded_conditions(self, x_t: torch.Tensor, **conditions: Any) -> torch.Tensor:
        x = self.in_proj(x_t)
        for index, block in enumerate(self.blocks):
            x = block(x=x, **{**conditions, "context_kv": (torch.zeros(index + 1),)})
        return self.out_proj(self.out_norm(x))


@pytest.fixture(autouse=True)
def stub_rope_table(monkeypatch):
    """`DitGraph`'s RoPE table is built from the upstream implementation — the model only needs
    the shape.

    This lets {@link irodori.calib.dit_i4_names}'s "is this weight in the shipped graph" predicate
    run regardless of whether the implementation clone (`irodori_tts`) is present.
    """
    monkeypatch.setattr(
        ir_patch,
        "real_pair_rope_table",
        lambda head_dim, length: torch.zeros(length, 2, head_dim),
    )


def make_dit(blocks: int = 3, seed: int = 0, dit_type: type[TinyDit] = TinyDit) -> TinyDit:
    torch.manual_seed(seed)
    dit = dit_type(blocks)
    dit.eval()
    return dit


def make_conditions(seed: int = 7, rows: int = CFG_ROWS) -> dict[str, Any]:
    """The full set of accompanying arguments passed to a block (the 11 besides `x`)."""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    return {
        "cond_embed": randn(rows, 1, HIDDEN),
        "text_state": randn(rows, 5, CONTEXT),
        "text_mask": torch.ones(rows, 5, dtype=torch.bool),
        "speaker_state": randn(rows, 3, CONTEXT),
        "speaker_mask": torch.ones(rows, 3, dtype=torch.bool),
        "caption_state": randn(rows, 4, CONTEXT),
        "caption_mask": torch.ones(rows, 4, dtype=torch.bool),
        "freqs_cis": randn(TOKENS, 4),
        "self_mask": None,
        "context_kv": None,
        # Self + text + speaker + caption keys, one plan for the whole batch (as upstream builds it
        # once per forward).
        "context_plan": torch.ones(rows, TOKENS + 5 + 3 + 4, dtype=torch.bool),
    }


def capture_run(dit: nn.Module, seed: int = 7, rows: int = CFG_ROWS) -> calib.CapturedRun:
    """Run the plain DiT once and build a {@link irodori.calib.CapturedRun} (a miniature of the
    real thing).

    In the real path `irodori.calib.capture_stage_batches` runs upstream `sample_euler_rf_cfg`
    and takes the same thing. Slicing goes through the genuine
    {@link irodori.calib._split_block_kwargs} as is, so the checks on taking the single cond-side
    row and on the set of arguments are in effect here too.
    """
    generator = torch.Generator().manual_seed(seed + 1)
    x_t = torch.randn(rows, TOKENS, HIDDEN, generator=generator)
    conditions = make_conditions(seed, rows)
    batches: list[Any] = []
    graph_batch: list[Any] = []
    tail: list[torch.Tensor] = []

    def on_block(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if graph_batch:
            return
        graph_batch.append(
            (
                (kwargs["x"].detach(),),
                {name: value for name, value in kwargs.items() if name != "x"},
            )
        )
        batches.append(calib._split_block_kwargs(kwargs, 1))

    def on_tail(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if not tail:
            tail.append(args[0].detach())

    handles = [
        dit.blocks[0].register_forward_pre_hook(on_block, with_kwargs=True),
        dit.out_norm.register_forward_pre_hook(on_tail),
    ]
    try:
        with torch.no_grad():
            dit.forward_with_encoded_conditions(x_t, **conditions)
    finally:
        for handle in handles:
            handle.remove()
    return calib.CapturedRun(tuple(batches), graph_batch[0], tail[0])


def make_plan(*, calibrated: bool = True, steps: int = 2) -> ir.CalibPlan:
    """The calibration scaffolding (the stub path does not use the contents of `context`)."""
    return ir.CalibPlan(
        context=calib.CalibContext(
            model_dir=Path("unused"),
            source=None,
            config=SimpleNamespace(latent_patch_size=1),
            text_config={},
            model_config={},
        ),
        cases=CALIB_CASES,
        steps=steps,
        calibrated=calibrated,
    )


@pytest.fixture
def stub_capture(monkeypatch):
    """Replace `capture_stage_batches` (the real one runs upstream's reference denoise).

    The replacement captures by **actually running the plain DiT**, so both the shape of
    `(args, kwargs)` and the capture source stay the same as the real one. It returns
    `(the step count per call, the weights at capture time)`; the latter is a copy that lets the
    caller assert "which weights were rounded when the calibration input was taken".

    MUST: it runs **at the time it is called** (as the real one does) — capturing in advance
    would make the block-internal adaLN, already rounded by i8, disagree with the weights at
    capture time, and the split-equivalence check would fail because of the mock.
    """

    def install(dit: nn.Module) -> tuple[list[int], dict[str, torch.Tensor]]:
        calls: list[int] = []
        at_capture: dict[str, torch.Tensor] = {}

        def stub(model, cases, context, *, steps=calib.CALIB_STEPS, label="calib"):
            calls.append(steps)
            at_capture.clear()
            at_capture.update({n: p.detach().clone() for n, p in model.named_parameters()})
            prepared = capture_run(model)
            return tuple(prepared for _ in cases)

        monkeypatch.setattr(calib, "capture_stage_batches", stub)
        return calls, at_capture

    return install


def quantize(dit: nn.Module, plan: ir.CalibPlan | None = None) -> ir.FakeQuantResult:
    return ir.fake_quant("i4", {ir.TARGET_DIT: dit}, calib_plan=plan or make_plan())


class TestStageSplit:
    def test_the_stage_chain_reproduces_the_block_loop_bit_exactly(self):
        """MUST: the split does not differ from the genuine path by a single bit (otherwise
        GPTQ along another path would ship)."""
        dit = make_dit()

        calib.assert_stage_split(calib.dit_stages(dit), capture_run(dit))

    def test_a_dropped_stage_is_caught(self):
        """Dropping one stage changes the final hidden state (a measurement that the check does
        not pass vacuously)."""
        dit = make_dit()
        run = capture_run(dit)

        with pytest.raises(AssertionError, match="not bit-identical"):
            calib.assert_stage_split(calib.dit_stages(dit)[:-1], run)

    def test_a_reordered_stage_chain_is_caught(self):
        """Swapped stages still fit the shapes (a breakage where only the values quietly
        change)."""
        dit = make_dit()
        stages = calib.dit_stages(dit)
        run = capture_run(dit)

        with pytest.raises(AssertionError, match="not bit-identical"):
            calib.assert_stage_split((stages[1], stages[0], *stages[2:]), run)

    def test_the_stage_prefix_lands_in_the_dit_fqn_space(self):
        """The prefixed local FQNs match the DiT's real FQNs (where the ledger's key space is
        decided)."""
        dit = make_dit(blocks=2)

        names = calib.stage_linear_names(calib.dit_stages(dit))

        # The 6 condition-side projections are inside the blocks too (the model now has the real
        # set of children — ADR 0114 split the shipped graph, but the calibration stage is still
        # the whole block).
        assert names == {
            f"blocks.{index}.{child}"
            for index in range(2)
            for child in (
                "attention.wq",
                "attention.wo",
                *(f"attention.{name}" for name in ir.CONTEXT_KV_PROJECTIONS),
                "mlp",
                "attention_adaln.1",
                "mlp_adaln.1",
            )
        }
        assert names <= {name for name, _module in dit.named_modules()}

    def test_the_export_path_runs_the_gate(self, stub_capture, monkeypatch):
        """MUST: the check is really wired into `_round_i4_calibrated` (the function alone is
        not enough)."""
        dit = make_dit()
        stub_capture(dit)
        genuine = calib.assert_stage_split

        def drifted(stages, run):
            return genuine(stages[:-1], run)

        monkeypatch.setattr(calib, "assert_stage_split", drifted)

        with pytest.raises(AssertionError, match="not bit-identical"):
            quantize(dit)


class TestCapturedKwargs:
    def test_it_keeps_only_the_cond_row_of_the_batched_arguments(self):
        """Upstream independent CFG bundles cond + uncond into one forward — only row 0 is the
        cond side."""
        dit = make_dit()

        run = capture_run(dit)

        args, kwargs = run.batches[0]
        assert args[0].shape[0] == 1
        assert run.graph_batch[0][0].shape[0] == CFG_ROWS
        for name in calib.BATCHED_BLOCK_ARGS - {"x"}:
            assert kwargs[name].shape[0] == 1, name
        # Arguments without a batch axis stay as they are (slicing would make the RoPE table 1
        # long).
        assert kwargs["freqs_cis"].shape[0] == TOKENS

    def test_the_cond_row_lets_the_block_rebuild_the_whole_batch_context_plan(self):
        """Upstream builds the context plan once for the whole `[cond, uncond…]` batch — the
        cond-side batch hands the block `None` so it builds the plan from the row-0 masks, while
        the unsliced batch keeps the captured plan for the stage-split replay."""
        dit = make_dit()

        run = capture_run(dit)

        assert run.batches[0][1]["context_plan"] is None
        assert run.graph_batch[1]["context_plan"].shape[0] == CFG_ROWS
        # The cond-side batch really replays through a block (a kept whole-batch plan raises).
        args, kwargs = run.batches[0]
        dit.blocks[0](*args, **kwargs)

    def test_a_dropped_block_argument_is_caught(self):
        """If upstream drops an argument, the cond-side slicing silently comes off (the numbers
        come out normally)."""
        dit = make_dit(dit_type=DriftingDit)

        with pytest.raises(AssertionError, match="the block call arguments"):
            capture_run(dit)

    def test_a_context_kv_cache_is_caught(self):
        """The shape where one kwargs would be reused for a cache that differs per block."""
        dit = make_dit(dit_type=CachedKvDit)

        with pytest.raises(AssertionError, match="condition K/V cache"):
            capture_run(dit)


class TestEligibleSet:
    def test_only_the_linears_the_shipped_graph_holds_are_eligible(self):
        """Copies of other roles held inside the DiT are not in i4 (decided by the intersection
        with the union of what the two distribution wrappers own)."""
        dit = make_dit(blocks=2)

        names = calib.dit_i4_names(dit, ir.dit_sym_max(SimpleNamespace(latent_patch_size=1)))

        assert names == {
            "in_proj",
            "out_proj",
            "cond_module.0",
            "cond_module.2",
            *calib.stage_linear_names(calib.dit_stages(dit)),
        }
        assert "pretrained_text_backbone" not in names

    def test_a_weight_off_the_group_grid_fails_loudly(self):
        """MUST: a weight not divisible by g32 fails rather than being **excluded** (dropping
        just one would break the exact-coverage check)."""
        dit = make_dit(dit_type=UnalignedDit)

        with pytest.raises(AssertionError, match="is not divisible"):
            calib.dit_i4_names(dit, ir.dit_sym_max(SimpleNamespace(latent_patch_size=1)))

    def test_the_eligible_set_is_split_exclusively_between_the_two_paths(self):
        """The eligible set splits **exactly and exclusively** into inside and outside the
        blocks (the basis of how storage is split).

        The mixing MUST of `quantize.py` (no weight goes through both paths). The eligible set
        itself does not depend on the adaLN ruling — only the split of the storage overrides
        below moves.
        """
        dit = make_dit()
        stage_names = calib.stage_linear_names(calib.dit_stages(dit))
        eligible = calib.dit_i4_names(dit, ir.dit_sym_max(SimpleNamespace(latent_patch_size=1)))

        assert stage_names <= eligible, (
            "a linear inside a block is missing from the i4-eligible set"
        )
        assert eligible - stage_names == {"in_proj", "out_proj", "cond_module.0", "cond_module.2"}

    def test_a_block_linear_outside_the_i4_set_fails_loudly(self, monkeypatch, stub_capture):
        """A non-eligible linear inside a block breaks the check "scanned count = rounded
        count"."""
        dit = make_dit()
        stub_capture(dit)
        genuine = calib.dit_i4_names
        monkeypatch.setattr(
            calib,
            "dit_i4_names",
            lambda model, sym_max: frozenset(genuine(model, sym_max) - {"blocks.0.mlp"}),
        )

        with pytest.raises(SystemExit, match="are not i4-eligible"):
            quantize(dit)

    def test_a_report_short_of_the_scan_fails_loudly(self, monkeypatch, stub_capture):
        """A missed rounding makes quality look **better** (passing through cannot be read off
        the numbers)."""
        from dataclasses import replace

        dit = make_dit()
        stub_capture(dit)
        genuine = calib.calibrate_i4

        def short(stages, batches, include=None):
            report, ledger = genuine(stages, batches, include)
            return replace(report, layers=report.layers[:-1]), ledger

        monkeypatch.setattr(calib, "calibrate_i4", short)

        with pytest.raises(AssertionError, match="of the scan"):
            quantize(dit)

    def test_a_ledger_that_overlaps_the_i8_path_fails_loudly(self, monkeypatch, stub_capture):
        """Rounding the same weight along two paths quietly corrupts only the values (the
        storage form and the count look right).

        The injected weight is `in_proj` outside the blocks (= already rounded by i8) — the
        double-rounding check runs before the exact-coverage check, so the diagnosis says "the
        same weight was rounded twice" rather than reporting a count mismatch.
        """
        from dataclasses import replace

        dit = make_dit()
        stub_capture(dit)
        genuine = calib.calibrate_i4

        def overlapping(stages, batches, include=None):
            report, ledger = genuine(stages, batches, include)
            merged = dict(ledger.scales) | {"in_proj.weight": torch.ones(HIDDEN, 1)}
            return report, replace(ledger, scales=merged)

        monkeypatch.setattr(calib, "calibrate_i4", overlapping)

        with pytest.raises(SystemExit, match="double rounding"):
            quantize(dit)


class TestAdalnBoundary:
    """The adaLN boundary that splits i4 from i8 (the listening ruling of 2026-08-23 — 144
    weights moved back to i8)."""

    def test_the_match_is_by_segment_not_by_substring(self):
        """`attention` does not drag in `attention_adaln`, and no child under adaLN is
        missed."""
        assert calib.is_adaln("blocks.0.attention_adaln.1")
        assert calib.is_adaln("blocks.0.mlp_adaln.1")
        # The same answer for the local FQNs inside a stage (the form calibration's `include`
        # receives).
        assert calib.is_adaln("attention_adaln.1")
        assert calib.is_adaln("mlp_adaln")
        assert not calib.is_adaln("blocks.0.attention.wq")
        assert not calib.is_adaln("blocks.0.attention.wo")
        assert not calib.is_adaln("blocks.0.mlp")
        assert not calib.is_adaln("in_proj")

    def test_a_dit_without_adaln_fails_loudly(self, stub_capture):
        """MUST: fail when upstream renames the modulation attributes.

        If the predicate matches nothing, the 144 weights silently go back to i4 — the storage
        form and the count check stay right, so the configuration the ruling rejected ships while
        green (leaving only the degraded delivery of the speech).
        """
        dit = make_dit(dit_type=RenamedAdalnDit)
        stub_capture(dit)

        with pytest.raises(SystemExit, match="no adaLN linear for the spellings"):
            quantize(dit)

    def test_the_seen_segments_are_reported_one_by_one(self):
        """MUST: return **which** segments matched, not "whether any matched" (what detects a
        one-sided rename)."""
        both = ["blocks.0.attention_adaln.1", "blocks.0.mlp_adaln.1", "blocks.0.attention.wq"]

        assert calib.adaln_segments_seen(both) == calib.ADALN_SEGMENTS
        assert calib.adaln_segments_seen(both[:1]) == {"attention_adaln"}
        assert calib.adaln_segments_seen(["blocks.0.mlp_adaln"]) == {"mlp_adaln"}
        assert calib.adaln_segments_seen(["blocks.0.attention.wq"]) == frozenset()

    def test_a_dit_that_renamed_only_one_side_fails_loudly(self, stub_capture):
        """MUST: a one-sided rename fails too (the set is not empty, so the "none at all"
        check passes).

        If only `mlp_adaln` disappears, half of the 144 weights moved back to i8 go back to i4.
        The storage form and the exact-coverage check of the count stay green (both sides are
        built from the same classifier), so this is the only net that catches it.
        """
        dit = make_dit(dit_type=HalfRenamedAdalnDit)
        stub_capture(dit)

        with pytest.raises(SystemExit, match=r"no adaLN linear for the spellings: \['mlp_adaln'\]"):
            quantize(dit)


class TestCalibratedI4:
    def test_the_ledger_keys_live_in_the_shipped_graph_fqn_space(self, stub_capture):
        """MUST: ledger keys = safetensors tensor keys (this is where emit's matching is
        decided)."""
        dit = make_dit()
        stub_capture(dit)

        result = quantize(dit)

        # The shipped graphs are the two `dit-context` / `dit` (ADR 0114) — the FQN space is the
        # union of what the two own.
        wrappers = ir.dit_wrappers(dit, ir.dit_sym_max(SimpleNamespace(latent_patch_size=1)))
        owned = {
            name for graph in wrappers.by_target().values() for name, _p in graph.named_parameters()
        }
        overrides = result.overrides[ir.TARGET_DIT]
        assert set(overrides) <= owned, "i4 override keys are not in the shipped graph's FQN space"
        assert set(overrides) == {
            f"{name}.weight"
            for name in calib.dit_i4_names(
                dit, ir.dit_sym_max(SimpleNamespace(latent_patch_size=1))
            )
        }

    def test_the_overrides_are_a_hybrid_of_i4_and_i8_by_block_and_adaln(self, stub_capture):
        """MUST: the i8 side (outside the blocks + adaLN) is named with **an explicit `i8`** —
        dropping it lets it flow to emit's default (i4).

        The default `weight_dtype` is i4, so `karume.emit._plan_weight_dtype` sends i4-eligible
        weights without an override to the i4 plan by default. When the i8 side, which has only
        a per-channel scale, arrives there, it fails with "no group scale" (= the check on the
        side that needs the explicit override).
        """
        dit = make_dit()
        stub_capture(dit)

        result = quantize(dit)

        overrides = result.overrides[ir.TARGET_DIT]
        stage_names = calib.stage_linear_names(calib.dit_stages(dit))
        adaln_names = {name for name in stage_names if calib.is_adaln(name)}
        assert adaln_names, "the model has no adaLN (this check matches nothing)"
        assert {key for key, dtype in overrides.items() if dtype == "i4"} == {
            f"{name}.weight" for name in stage_names - adaln_names
        }
        assert {key for key, dtype in overrides.items() if dtype == "i8"} == {
            "in_proj.weight",
            "out_proj.weight",
            "cond_module.0.weight",
            "cond_module.2.weight",
            *(f"{name}.weight" for name in adaln_names),
        }
        # The i8 overrides outside the blocks need a scale too (`karume.emit._plan_i8` fails
        # on a missing scale).
        scales = result.scales[ir.TARGET_DIT]
        assert set(overrides) <= set(scales), (
            "a weight with a storage override has no scale (emit would fail)"
        )

    def test_the_rebased_tables_reach_emit_with_the_same_keys(self, stub_capture):
        """MUST: the scales and storage overrides passed to emit live in **the same key
        space** (the rebasing is one implementation)."""
        dit = make_dit()
        stub_capture(dit)

        result = quantize(dit)

        # The union rebased onto the two DiT graphs (ADR 0114) matches the rounded side's
        # storage overrides exactly.
        wrappers = ir.dit_wrappers(dit, ir.dit_sym_max(SimpleNamespace(latent_patch_size=1)))
        rebased: dict[str, str] = {}
        for target, graph in wrappers.by_target().items():
            overrides = ir.target_weight_dtypes(target, graph, result.overrides)
            scales = ir.target_scales(target, graph, result.scales)
            assert set(overrides) <= set(scales), (
                f"{target}: a weight with an i4 override has no scale"
            )
            rebased.update(overrides)
        assert rebased == dict(result.overrides[ir.TARGET_DIT])

    def test_the_i4_set_is_split_by_ownership_between_the_two_dit_graphs(self, stub_capture):
        """MUST: rounding happens once, and the i4 set stays "inside the blocks − adaLN",
        split by what the two wrappers own.

        `dit-context` receives only the condition-side projections (i4) and no i8 at all; `dit`
        receives the remaining block-internal i4 and all of i8. The two are **disjoint** and their
        union is the whole rounded set — an overlap would bake the same weight into the two
        containers separately, and a gap would lose storage overrides.
        """
        dit = make_dit(blocks=2)
        stub_capture(dit)

        result = quantize(dit)

        wrappers = ir.dit_wrappers(dit, ir.dit_sym_max(SimpleNamespace(latent_patch_size=1)))
        context = ir.target_weight_dtypes(ir.TARGET_DIT_CONTEXT, wrappers.context, result.overrides)
        rest = ir.target_weight_dtypes(ir.TARGET_DIT, wrappers.dit, result.overrides)
        assert set(context) == {
            f"blocks.{index}.attention.{name}.weight"
            for index in range(2)
            for name in ir.CONTEXT_KV_PROJECTIONS
        }
        assert set(context.values()) == {"i4"}
        assert not set(context) & set(rest)
        assert {**context, **rest} == dict(result.overrides[ir.TARGET_DIT])
        assert {"i4", "i8"} <= set(rest.values())

    def test_calibration_produces_a_different_rounding_from_plain_rtn(self, stub_capture):
        """If calibration passed through, the storage form is the same and it cannot be read off
        the assets — measure it by the value difference."""
        calibrated = make_dit()
        plain = make_dit()
        stub_capture(calibrated)

        quantize(calibrated)
        quantize(plain, make_plan(calibrated=False))

        after = dict(calibrated.named_parameters())
        differing = sorted(
            name
            for name, weight in plain.named_parameters()
            if not torch.equal(weight, after[name])
        )
        assert differing, "GPTQ produced the same values as RTN (calibration had no effect)"
        assert all(name.startswith("blocks.") for name in differing)
        # MUST: adaLN is excluded by GPTQ's `include` — a difference would mean it was rounded on
        # the i4 side.
        assert not any(calib.is_adaln(name) for name in differing)

    def test_the_i8_side_weights_land_on_the_i8_grid(self, stub_capture):
        """MUST: the i8 side is on **the i8 grid** (also with `--no-calib` — calibration acts only
        on the i4 side).

        These are the outside of the blocks and adaLN that the listening ruling of 2026-08-23
        took out of i4. If they stayed on the i4 grid, switching the storage override to i8
        would still ship values that keep the 4-bit muffled sound (with shapes and counts looking
        right).
        """
        calibrated = make_dit()
        plain = make_dit()
        stub_capture(calibrated)

        quantize(calibrated)
        quantize(plain, make_plan(calibrated=False))

        after = dict(calibrated.named_parameters())
        adaln = sorted(
            name
            for name in calib.stage_linear_names(calib.dit_stages(calibrated))
            if calib.is_adaln(name)
        )
        for name in ("in_proj", "out_proj", "cond_module.0", "cond_module.2", *adaln):
            key = f"{name}.weight"
            assert torch.equal(dict(plain.named_parameters())[key], after[key]), key
            scale = channel_scale(after[key], 0)
            restored = quantize_to_int8(after[key], scale).to(torch.float32) * scale
            assert torch.equal(restored, after[key]), f"{key} is not on the i8 grid"

    def test_the_capture_sees_the_shipped_condition(self, stub_capture):
        """MUST: calibration inputs are taken **after** the i8 side (outside the blocks, adaLN,
        the inner copies of other roles) is rounded and **before** the i4 side is.

        The second half (the i4 side still unrounded) catches "the cycle of choosing a weight's
        rounding target from activations that the rounded weight itself produced", and the first
        half (the i8 side rounded) catches "shipping rounding targets chosen around f32 together
        with i8 surroundings". adaLN is i8 **inside** the blocks, so if it were deferred, only the
        modulation would choose its rounding targets from f32 activations.
        """
        dit = make_dit()
        before = {name: p.detach().clone() for name, p in dit.named_parameters()}
        _calls, at_capture = stub_capture(dit)

        quantize(dit)

        for name in ("in_proj", "out_proj", "cond_module.0", "cond_module.2"):
            key = f"{name}.weight"
            assert not torch.equal(before[key], at_capture[key]), f"{key} is unrounded at capture"
        # The i8 side (the copies held inside the DiT) is also rounded before capture (for the
        # same reason as the side that builds block 0's input).
        i8_key = "pretrained_text_backbone.weight"
        assert not torch.equal(before[i8_key], at_capture[i8_key]), (
            f"{i8_key} is unrounded at capture"
        )
        for key in (name for name in before if name.startswith("blocks.")):
            if calib.is_adaln(key):
                assert not torch.equal(before[key], at_capture[key]), (
                    f"{key} is unrounded at capture"
                )
            else:
                assert torch.equal(before[key], at_capture[key]), f"{key} is rounded before capture"

    def test_the_probe_run_precedes_every_rounding(self, stub_capture):
        """Ordering MUST ①: the split-equivalence check's probe is **1 step** and runs before
        rounding (the second call is the real capture)."""
        dit = make_dit()
        calls, _at_capture = stub_capture(dit)

        quantize(dit, make_plan(steps=5))

        assert calls == [1, 5]


class TestOptOut:
    def test_no_calib_never_touches_the_calibration_path(self, stub_capture):
        """The opt-out means calibration **does not run**, not "calibration gets thinner" (cost
        and quality differ entirely)."""
        dit = make_dit()
        calls, _at_capture = stub_capture(dit)

        quantize(dit, make_plan(calibrated=False))

        assert calls == []

    def test_i4_without_a_plan_fails_loudly(self):
        """MUST: there is no branch that silently drops to plain RTN (fail without the
        calibration scaffolding)."""
        dit = make_dit()

        with pytest.raises(SystemExit, match="calibration scaffolding"):
            ir.fake_quant("i4", {ir.TARGET_DIT: dit})

    def test_the_i4_series_is_calibrated_by_default(self, monkeypatch):
        """MUST: the CLI's `--dtype i4` is calibrated without a flag (only `--no-calib` opts
        out)."""
        seen: dict[str, Any] = {}
        monkeypatch.setattr(ir, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})

        ir.main(["--dtype", "i4"])

        assert seen["no_calib"] is False
        assert seen["calib_steps"] is None
        # The i4 series is the two DiT graphs (one `dit` before the split — ADR 0114).
        assert seen["targets"] == (ir.TARGET_DIT, ir.TARGET_DIT_CONTEXT)

    def test_the_calibration_knobs_are_refused_outside_i4(self, monkeypatch):
        """Knobs that have no effect are not silently accepted (i8 has no calibration path at
        all)."""
        with pytest.raises(SystemExit):
            ir.main(["--dtype", "i8", "--no-calib"])
        with pytest.raises(SystemExit):
            ir.main(["--dtype", "i8", "--calib-steps", "4"])

    def test_a_target_the_i4_series_does_not_write_is_refused(self):
        """MUST: do not silently write a series the distribution table does not read (an explicit
        `--target` is refused too)."""
        with pytest.raises(SystemExit):
            ir.main(["--dtype", "i4", "--target", ir.TARGET_BACKBONE])


class TestCalibProvenance:
    def test_it_records_the_shippable_method_by_default(self, tmp_path):
        name = ir._write_calib_provenance("i4", make_plan(steps=7), ir.TARGET_DIT, tmp_path)

        import json

        record = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        assert record["method"] == calib.CALIB_METHOD
        assert record["grid"] == calib.CALIB_GRID.kind
        assert record["group_size"] == calib.CALIB_GRID.group_size
        assert record["cases"] == len(CALIB_CASES)
        assert record["steps"] == 7

    def test_no_calib_is_written_as_rtn_rather_than_omitted(self, tmp_path):
        """Absence could also read as "an old export" — use a form the build side can reject by
        name."""
        import json

        name = ir._write_calib_provenance(
            "i4", make_plan(calibrated=False), ir.TARGET_DIT, tmp_path
        )

        record = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        assert record["method"] == "rtn"
        assert record["cases"] == 0
        assert record["steps"] == 0

    def test_a_stale_record_is_removed_when_the_series_is_retaken(self, tmp_path):
        """MUST: a total function (it guarantees absence for callers that do not go through
        staged too)."""
        ir._write_calib_provenance("i4", make_plan(), ir.TARGET_DIT, tmp_path)

        assert ir._write_calib_provenance("i8", None, ir.TARGET_DIT, tmp_path) is None
        assert not (tmp_path / "calib_provenance.json").exists()

    def test_the_record_does_not_survive_a_staged_retake(self, tmp_path):
        """The record is not carried over with the production call either (write into a staging
        directory and swap it in whole).

        What actually guarantees absence is the swap, not `unlink` — `export_series` always passes
        **an empty staging directory**, and `swap_into_place` carries over nothing of the old
        contents. The shape of retaking a series captured as i4 as i8 goes through here.
        """
        final = tmp_path / "irodori-v4-small-i4" / ir.TARGET_DIT

        with staged_publication(final) as staged:
            staged.mkdir(parents=True)
            ir._write_calib_provenance("i4", make_plan(), ir.TARGET_DIT, staged)
        assert (final / "calib_provenance.json").is_file()

        with staged_publication(final) as staged:
            staged.mkdir(parents=True)
            ir._write_calib_provenance("i8", None, ir.TARGET_DIT, staged)

        assert not (final / "calib_provenance.json").exists()

    def test_both_dit_targets_carry_the_same_record(self, tmp_path):
        """The two DiT graphs write out the same rounding (ADR 0114) — the build side checks
        that both agree."""
        import json

        records = []
        for target in (ir.TARGET_DIT, ir.TARGET_DIT_CONTEXT):
            directory = tmp_path / target
            directory.mkdir()
            name = ir._write_calib_provenance("i4", make_plan(steps=3), target, directory)
            records.append(json.loads((directory / name).read_text(encoding="utf-8")))

        assert records[0] == records[1]

    def test_it_is_not_written_for_other_targets(self, tmp_path):
        assert ir._write_calib_provenance("i4", make_plan(), ir.TARGET_BACKBONE, tmp_path) is None
        assert not (tmp_path / "calib_provenance.json").exists()


class CacheBuildingDit(TinyDit):
    """A DiT whose sampler would hand every block a per-block condition K/V cache (the shape of
    upstream's MeanFlow loop, which always builds one through `build_context_kv_cache`)."""

    def build_context_kv_cache(self, **_states: Any) -> list[tuple[torch.Tensor, ...]]:
        return [(torch.ones(1),) for _ in self.blocks]


class TestMeanFlowCapture:
    """The MeanFlow loop of upstream always builds the condition K/V cache; the capture runs it
    with every block projecting its own K/V instead (ADR 0150)."""

    def test_the_cache_builder_returns_nothing_while_shadowed_and_comes_back_after(self):
        dit = CacheBuildingDit()

        with calib._without_context_kv_cache(dit):
            assert dit.build_context_kv_cache(text_state=None) is None
        assert dit.build_context_kv_cache(text_state=None) == [(torch.ones(1),) for _ in dit.blocks]
        assert "build_context_kv_cache" not in vars(dit)

    def test_the_cache_builder_comes_back_after_a_failure_too(self):
        dit = CacheBuildingDit()

        with pytest.raises(RuntimeError, match="boom"), calib._without_context_kv_cache(dit):
            raise RuntimeError("boom")
        assert "build_context_kv_cache" not in vars(dit)

    def test_shadowing_twice_fails_loudly(self):
        dit = CacheBuildingDit()

        with (
            calib._without_context_kv_cache(dit),
            pytest.raises(AssertionError, match="already shadowed"),
            calib._without_context_kv_cache(dit),
        ):
            pass

    def test_a_meanflow_dit_is_driven_by_its_own_loop(self):
        """The capture picks the sampler from the DiT itself (its `delta_cond_module`), never from a
        name — the same test `irodori.export.dit_wrappers` uses."""
        rf = make_dit()
        meanflow = make_dit()
        meanflow.delta_cond_module = nn.Linear(HIDDEN, HIDDEN, bias=False)

        assert ir.dit_flow(rf) == "rf_velocity"
        assert ir.dit_flow(meanflow) == "meanflow"
