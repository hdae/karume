"""Unit tests of the SBV2 patch layer (M1-P3 iteration 6).

Each patch exists only on condition that it is "eager-equivalent to the original implementation"
(ADR 0013). The end-to-end checks with real weights belong to `test_export.py` and
`sbv2.export --verify`; here **small synthetic modules** isolate the patches one by one. SKIPs in
an environment without the `style_bert_vits2` package.

MUST: a test that replaces class attributes takes the `pristine_classes` fixture. Replacements
affect the whole process, so ① **not restoring the original implementation** on entry makes a
vacuous comparison that "grabs the patched version believing it is the original", and ② not
restoring the previous state on exit makes later tests silently check another implementation.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from sbv2 import patch as patch_sbv2

_PACKAGE_PRESENT = importlib.util.find_spec("style_bert_vits2") is not None

requires_package = pytest.mark.skipif(
    not _PACKAGE_PRESENT,
    reason="no style_bert_vits2 (requires `uv sync --group sbv2`)",
)

#: The tolerance of the equivalence check. A patch is a rewrite that "builds the same formula in
#: another order", so the difference should stay within f32 reduction-order differences (a few
#: ulp). For an O(1) value range, 1e-5 is 5 orders of magnitude away from the O(1) error of an
#: implementation mix-up (index slip, swapped axis).
EQUIVALENCE_ATOL = 1e-5

#: **The original implementation taken at collection time**. pytest collects every test module
#: before running, so this is a point where no test has run = nobody has applied a patch yet.
#:
#: MUST: saving "the value at test start" is not enough — if another test in the same process
#: (the front export of `test_export.py`) applied the patch first, it grabs **the patched
#: implementation** believing it is the original, and the equivalence test becomes a **vacuous
#: comparison** of patched vs patched (this actually happened once, green alone and green in the
#: full run).
if _PACKAGE_PRESENT:
    from style_bert_vits2.models import attentions as _sbv2_attentions
    from style_bert_vits2.models import modules as _sbv2_modules

    PRISTINE_MHA_FORWARD = _sbv2_attentions.MultiHeadAttention.forward
    PRISTINE_FFN_FORWARD = _sbv2_attentions.FFN.forward
    PRISTINE_SPLINE = _sbv2_modules.piecewise_rational_quadratic_transform


@pytest.fixture
def pristine_classes():
    """Enters the test after restoring the patch targets to **the original implementation**, and
    restores the previous state on leaving.

    Restoring the original on entry prevents a vacuous check (the MUST above). Restoring "the
    state on entry" on exit keeps this test from breaking other tests' premises (the front export
    assumes the patches are applied) — restoring to different states on entry and exit is
    intentional.
    """
    from style_bert_vits2.models import modules
    from style_bert_vits2.models.attentions import FFN, MultiHeadAttention

    entered = (
        MultiHeadAttention.forward,
        FFN.forward,
        modules.piecewise_rational_quadratic_transform,
        patch_sbv2.patches_applied(),
    )
    MultiHeadAttention.forward = PRISTINE_MHA_FORWARD
    FFN.forward = PRISTINE_FFN_FORWARD
    modules.piecewise_rational_quadratic_transform = PRISTINE_SPLINE
    patch_sbv2._APPLIED = False
    try:
        yield
    finally:
        MultiHeadAttention.forward = entered[0]
        FFN.forward = entered[1]
        modules.piecewise_rational_quadratic_transform = entered[2]
        patch_sbv2._APPLIED = entered[3]


def _spline_arguments(seed: int, num_bins: int = 10, tail_bound: float = 5.0):
    """A synthetic input of the shape `[b, c, t, ?]` that ConvFlow passes to the spline.

    `inputs` covers a value range **both outside and inside** tail_bound (so that one case goes
    through both the path that returns out-of-range inputs as they are and the spline body).
    """
    generator = torch.Generator().manual_seed(seed)
    shape = (1, 1, 7)
    inputs = torch.randn(shape, generator=generator) * tail_bound
    return {
        "inputs": inputs,
        "unnormalized_widths": torch.randn((*shape, num_bins), generator=generator),
        "unnormalized_heights": torch.randn((*shape, num_bins), generator=generator),
        "unnormalized_derivatives": torch.randn((*shape, num_bins - 1), generator=generator),
        "tail_bound": tail_bound,
    }


@requires_package
class TestSplineEquivalence:
    """① The spline made branch-free and non-destructive."""

    @pytest.mark.parametrize("inverse", [False, True])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_it_matches_the_original_transform(self, inverse, seed):
        from style_bert_vits2.models import transforms

        args = _spline_arguments(seed)
        tail_bound = args.pop("tail_bound")
        # The original rewrites unnormalized_derivatives in place, so pass clones (otherwise the
        # second and later comparisons get other inputs).
        expected = transforms.piecewise_rational_quadratic_transform(
            **{key: value.clone() for key, value in args.items()},
            inverse=inverse,
            tails="linear",
            tail_bound=tail_bound,
        )
        got = patch_sbv2.unconstrained_rqs_free(
            **{key: value.clone() for key, value in args.items()},
            inverse=inverse,
            tails="linear",
            tail_bound=tail_bound,
        )

        for index, (a, b) in enumerate(zip(got, expected, strict=True)):
            assert torch.allclose(a, b, atol=EQUIVALENCE_ATOL, rtol=0), index

    def test_the_case_actually_crosses_the_tail_bound(self):
        """Guard against a vacuous check — unless the input holds **both** in-range and out-of-range
        values, the where path is never exercised."""
        args = _spline_arguments(0)
        inside = (args["inputs"].abs() <= args["tail_bound"]).sum().item()

        assert 0 < inside < args["inputs"].numel()

    def test_it_does_not_destroy_its_arguments(self):
        """The check of non-destructiveness itself (the original destroys its input with
        `bin_locations[..., -1] += eps`)."""
        args = _spline_arguments(0)
        tail_bound = args.pop("tail_bound")
        before = {key: value.clone() for key, value in args.items()}

        patch_sbv2.unconstrained_rqs_free(
            **args, inverse=True, tails="linear", tail_bound=tail_bound
        )

        for key, value in args.items():
            assert torch.equal(value, before[key]), key

    def test_searchsorted_matches_the_original(self):
        from style_bert_vits2.models import transforms

        generator = torch.Generator().manual_seed(3)
        # The same shapes as in real use: boundaries [b, c, t, bins+1], values [b, c, t] (only the
        # last axis is the bin axis).
        locations = torch.cumsum(torch.rand(2, 5, 3, 6, generator=generator), dim=-1)
        values = torch.rand(2, 5, 3, generator=generator) * float(locations.max())

        got = patch_sbv2.searchsorted_free(locations, values)
        # The original destroys its first argument, so pass a clone for comparison (comparing with
        # the destroyed list would be a vacuous comparison "between two equally destroyed ones").
        expected = transforms.searchsorted(locations.clone(), values)

        assert torch.equal(got, expected)

    def test_tails_none_is_rejected_after_patching(self, pristine_classes):
        """`tails=None` goes straight to the spline without the original's domain check, so it is
        not accepted."""
        from style_bert_vits2.models import modules

        patch_sbv2.apply_spline_patch()
        args = _spline_arguments(0)
        args.pop("tail_bound")

        with pytest.raises(RuntimeError, match="tails=None"):
            modules.piecewise_rational_quadratic_transform(**args, tails=None)

    def test_the_patch_replaces_the_symbol_modules_actually_calls(self, pristine_classes):
        """The replaced binding is in modules (replacing the transforms side does not affect the
        binding in modules)."""
        from style_bert_vits2.models import modules, transforms

        original = modules.piecewise_rational_quadratic_transform
        assert original is transforms.piecewise_rational_quadratic_transform

        patch_sbv2.apply_spline_patch()

        assert modules.piecewise_rational_quadratic_transform is not original
        assert transforms.piecewise_rational_quadratic_transform is original


def _attention(seed: int, window_size: int = 4, channels: int = 24, heads: int = 2):
    from style_bert_vits2.models.attentions import MultiHeadAttention

    torch.manual_seed(seed)
    attn = MultiHeadAttention(channels, channels, heads, window_size=window_size)
    attn.eval()
    return attn


def _attention_case(length: int, channels: int = 24, seed: int = 11):
    generator = torch.Generator().manual_seed(seed + length)
    x = torch.randn(1, channels, length, generator=generator)
    mask = torch.ones(1, 1, length)
    attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
    return x, attn_mask


@requires_package
class TestGatherRelativeAttention:
    """② Relative-position attention rewritten with gather."""

    @pytest.mark.parametrize("length", [2, 4, 5, 9, 17])
    def test_it_matches_the_original_forward(self, pristine_classes, length):
        """For window size 4, exercises the 3 regions P < w+1 / P == w+1 / P > w+1.

        The original `_get_relative_embeddings` **cuts the embeddings from another position**
        when `P ≤ window_size` (it switches between pad and slice), so comparing only with large P
        would miss a mix-up in the cutting rule for small P.
        """
        from style_bert_vits2.models.attentions import MultiHeadAttention

        # Guard against a vacuous check: the reference is taken with **the original implementation**
        # (the fixture guarantees it, but if that broke the comparison would become patched vs
        # patched and quietly lose its meaning, so it is checked here too).
        assert MultiHeadAttention.forward is PRISTINE_MHA_FORWARD
        attn = _attention(seed=5)
        x, attn_mask = _attention_case(length)
        with torch.no_grad():
            expected = attn(x, x, attn_mask)

        patch_sbv2.apply_gather_relattn_patch()
        with torch.no_grad():
            got = attn(x, x, attn_mask)

        assert got.shape == expected.shape
        assert torch.allclose(got, expected, atol=EQUIVALENCE_ATOL, rtol=0)

    def test_the_relative_embeddings_actually_move_the_result(self, pristine_classes):
        """Guard against a vacuous check — changing the relative-position embeddings changes the
        output.

        Without it, the equivalence test above would be green even if "the gather path contributes
        exactly 0" (the real weights' emb_rel_* are small, so the values look plausible even with
        the contribution gone).
        """
        attn = _attention(seed=5)
        x, attn_mask = _attention_case(9)
        patch_sbv2.apply_gather_relattn_patch()
        with torch.no_grad():
            before = attn(x, x, attn_mask)
            attn.emb_rel_k.add_(1.0)
            attn.emb_rel_v.add_(1.0)
            after = attn(x, x, attn_mask)

        assert not torch.allclose(before, after, atol=EQUIVALENCE_ATOL, rtol=0)

    def test_the_masked_columns_are_excluded(self, pristine_classes):
        """Columns with attn_mask=0 are excluded from the softmax (the mask path did not disappear
        in the gather rewrite)."""
        attn = _attention(seed=5)
        x, _ = _attention_case(9)
        mask = torch.ones(1, 1, 9)
        mask[..., 6:] = 0.0
        attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        with torch.no_grad():
            expected = attn(x, x, attn_mask)

        patch_sbv2.apply_gather_relattn_patch()
        with torch.no_grad():
            got = attn(x, x, attn_mask)

        assert torch.allclose(got, expected, atol=EQUIVALENCE_ATOL, rtol=0)

    def test_heads_share_false_is_rejected(self, pristine_classes):
        """A configuration with separate embeddings per head cannot be expressed by expanding one
        head axis."""
        attn = _attention(seed=5)
        attn.heads_share = False
        x, attn_mask = _attention_case(9)
        patch_sbv2.apply_gather_relattn_patch()

        with pytest.raises(ValueError, match="heads_share"):
            attn(x, x, attn_mask)

    def test_a_window_free_attention_is_rejected(self, pristine_classes):
        attn = _attention(seed=5)
        attn.window_size = None
        x, attn_mask = _attention_case(9)
        patch_sbv2.apply_gather_relattn_patch()

        with pytest.raises(ValueError, match="window_size"):
            attn(x, x, attn_mask)


@requires_package
class TestRelattnTableThreading:
    """Table threading of ② (the flow / voice path — carrying the tables from the graph inputs to
    the attention layers)."""

    @pytest.mark.parametrize("length", [2, 4, 5, 9, 17])
    def test_supplied_tables_match_the_original_forward(self, pristine_classes, length):
        """Attention built from externally supplied tables is equivalent to **the original**
        (exercises the same 3 regions as front)."""
        from style_bert_vits2.models.attentions import MultiHeadAttention

        assert MultiHeadAttention.forward is PRISTINE_MHA_FORWARD  # guard against a vacuous check
        attn = _attention(seed=5)
        x, attn_mask = _attention_case(length)
        with torch.no_grad():
            expected = attn(x, x, attn_mask)

        tables = patch_sbv2.build_relattn_tables(length)
        with torch.no_grad():
            got = patch_sbv2.mha_gather_forward(attn, x, x, attn_mask, tables)

        assert got.shape == expected.shape
        assert torch.allclose(got, expected, atol=EQUIVALENCE_ATOL, rtol=0)

    @pytest.mark.parametrize("length", [2, 5, 17])
    def test_the_two_table_sources_are_bit_identical(self, pristine_classes, length):
        """front (built in-graph = baked) and flow (graph inputs) produce **the same numbers**.

        A measurement of the design where the table formula exists in one place only
        (`build_relattn_tables`). If this split, front with its baked table and flow with the
        host-built table would **both become a different model with the shapes still matching**
        (the silent-wrong-value class of ADR 0013).
        """
        attn = _attention(seed=5)
        x, attn_mask = _attention_case(length)
        with torch.no_grad():
            in_graph = patch_sbv2.mha_gather_forward(attn, x, x, attn_mask, None)
            supplied = patch_sbv2.mha_gather_forward(
                attn, x, x, attn_mask, patch_sbv2.build_relattn_tables(length)
            )

        assert torch.equal(in_graph, supplied)

    def test_a_table_shifted_by_one_changes_the_result(self, pristine_classes):
        """Fault injection — shifting the index table by 1 changes the output.

        A window-size mix-up has the same form as shifting `idx_k` by 1, and **the element count
        matches, so there is no shape error**. This pins that the equivalence test above can
        actually detect this class of error (= the table's contribution is not collapsed to
        exactly 0).
        """
        attn = _attention(seed=5)
        x, attn_mask = _attention_case(17)
        idx_k, valid = patch_sbv2.build_relattn_tables(17)
        window = 2 * 4
        shifted = torch.clamp(idx_k + 1, 0, window)
        with torch.no_grad():
            correct = patch_sbv2.mha_gather_forward(attn, x, x, attn_mask, (idx_k, valid))
            wrong = patch_sbv2.mha_gather_forward(attn, x, x, attn_mask, (shifted, valid))

        assert not torch.allclose(correct, wrong, atol=EQUIVALENCE_ATOL, rtol=0)

    def test_a_wider_window_runs_off_the_embedding(self, pristine_classes):
        """A table whose window size drifted **larger** has indices outside the embedding, and torch
        fails.

        Drifting smaller reads another row within range and is silently wrong (the shift by 1
        above is that form). Without recording that the two sides behave differently, one would
        misread "did not fail = correct".
        """
        attn = _attention(seed=5)
        x, attn_mask = _attention_case(17)
        tables = patch_sbv2.build_relattn_tables(17, window_size=5)

        with pytest.raises(RuntimeError):
            patch_sbv2.mha_gather_forward(attn, x, x, attn_mask, tables)


class TestRelattnTableBuilder:
    """The table generator itself
    (the source of truth of the host TS mirror `packages/models/src/sbv2/relattn-tables.ts`).
    """

    def test_the_tables_have_the_square_shape_and_dtypes(self):
        idx_k, valid = patch_sbv2.build_relattn_tables(7)

        assert idx_k.shape == (7, 7)
        assert valid.shape == (7, 7)
        assert idx_k.dtype == torch.int64
        assert valid.dtype == torch.float32

    def test_the_indices_stay_inside_the_embedding(self):
        """Already clamped, so the gather out-of-range rule (NaN pollution) is not violated (ADR
        0013)."""
        idx_k, _ = patch_sbv2.build_relattn_tables(64)

        assert int(idx_k.min()) == 0
        assert int(idx_k.max()) == 2 * 4

    def test_the_window_mask_matches_the_relative_distance(self):
        length, window = 12, 4
        _, valid = patch_sbv2.build_relattn_tables(length, window)
        positions = torch.arange(length)
        expected = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs() <= window

        assert torch.equal(valid, expected.to(torch.float32))

    def test_a_different_window_gives_different_tables(self):
        """Guard against a vacuous check (paired with the TS-side parity test) — measures that the
        window size takes effect."""
        narrow = patch_sbv2.build_relattn_tables(9, 4)
        wide = patch_sbv2.build_relattn_tables(9, 5)

        assert narrow[0].shape == wide[0].shape
        assert not torch.equal(narrow[0], wide[0])
        assert not torch.equal(narrow[1], wide[1])


@requires_package
class TestFfnPaddingFold:
    """③ FFN's explicit pad → conv's padding argument."""

    @pytest.mark.parametrize("kernel_size", [1, 3, 5])
    def test_it_matches_the_original_forward_for_odd_kernels(self, pristine_classes, kernel_size):
        from style_bert_vits2.models.attentions import FFN

        assert (
            FFN.forward is PRISTINE_FFN_FORWARD
        )  # guard against a vacuous check (as in gather above)
        torch.manual_seed(7)
        ffn = FFN(16, 16, 32, kernel_size)
        ffn.eval()
        generator = torch.Generator().manual_seed(8)
        x = torch.randn(1, 16, 11, generator=generator)
        mask = torch.ones(1, 1, 11)
        with torch.no_grad():
            expected = ffn(x, mask)

        patch_sbv2.apply_ffn_conv_padding_patch()
        with torch.no_grad():
            got = ffn(x, mask)

        assert torch.allclose(got, expected, atol=EQUIVALENCE_ATOL, rtol=0)

    def test_an_even_kernel_is_rejected(self, pristine_classes):
        """For an even kernel `_same_padding` is asymmetric, not equivalent to conv's padding."""
        from style_bert_vits2.models.attentions import FFN

        torch.manual_seed(7)
        ffn = FFN(16, 16, 32, 4)
        ffn.eval()
        patch_sbv2.apply_ffn_conv_padding_patch()

        with pytest.raises(ValueError, match="is even"):
            ffn(torch.randn(1, 16, 11), torch.ones(1, 1, 11))

    def test_a_causal_ffn_is_rejected(self, pristine_classes):
        from style_bert_vits2.models.attentions import FFN

        torch.manual_seed(7)
        ffn = FFN(16, 16, 32, 3, causal=True)
        ffn.eval()
        patch_sbv2.apply_ffn_conv_padding_patch()

        with pytest.raises(ValueError, match="causal"):
            ffn(torch.randn(1, 16, 11), torch.ones(1, 1, 11))

    @pytest.mark.parametrize("kernel_size", [2, 4])
    def test_an_even_kernel_really_differs(self, kernel_size):
        """Guard against a vacuous check — for an even kernel, the original's padding and conv's
        padding really differ.

        The original `_same_padding` is an **asymmetric** pad, left `(k−1)//2` / right `k//2`,
        which conv1d's padding argument (equal on both sides) cannot produce. A measurement that the
        rejection assert above stops real harm and is not "just in case" (with an even kernel even
        the output length differs).
        """
        from torch.nn import functional

        x = torch.zeros(1, 1, 11)
        explicit = functional.pad(x, [(kernel_size - 1) // 2, kernel_size // 2])
        symmetric = functional.pad(x, [(kernel_size - 1) // 2] * 2)

        assert explicit.shape != symmetric.shape


class TestConstantColumn:
    """How a constant column is built (the standard technique for keeping `full_like` out of the
    IR)."""

    def test_it_has_the_leading_shape_with_a_single_trailing_column(self):
        source = torch.randn(2, 3, 4)

        column = patch_sbv2._constant_column(source, 1.5)

        assert column.shape == (2, 3, 1)
        assert torch.equal(column, torch.full((2, 3, 1), 1.5))

    def test_a_negative_zero_source_still_yields_the_exact_constant(self):
        """`x * 0.0` is −0.0 even for x = −0.0, and `+ value` returns exactly value."""
        source = torch.full((1, 1, 2), -0.0)

        assert torch.equal(patch_sbv2._constant_column(source, -5.0), torch.full((1, 1, 1), -5.0))


@requires_package
class TestPatchApplicationFlag:
    """The flag that makes the process-wide effect of the patches visible (used to block vacuous
    checks)."""

    def test_it_flips_only_after_applying(self, pristine_classes):
        # The fixture enters with nothing applied.
        assert not patch_sbv2.patches_applied()

        patch_sbv2.apply_all_patches()

        assert patch_sbv2.patches_applied()


@requires_package
class TestWrapperShapes:
    """The form of the export wrappers (pinned as far as possible without real weights)."""

    def test_the_forward_signature_is_the_ir_input_order(self):
        import inspect

        from sbv2 import export as export_sbv2

        names = list(inspect.signature(patch_sbv2.Sbv2Front.forward).parameters)[1:]

        # The IR's input names are the forward's argument names as they are. If they disagree with
        # the script's order, dynamic_shapes attaches to another input (an input without P is baked
        # statically and breaks).
        assert names == list(export_sbv2.FRONT_INPUT_ORDER)

    def test_the_sdp_wrapper_takes_the_noise_as_an_input(self):
        import inspect

        names = list(inspect.signature(patch_sbv2.SdpReverseNoiseIn.forward).parameters)[1:]

        assert names == ["x", "x_mask", "g", "z_noise"]

    @pytest.mark.parametrize("wrapper", [patch_sbv2.FlowReverse, patch_sbv2.Sbv2Voice])
    def test_the_flow_family_takes_the_tables_as_inputs(self, wrapper):
        """flow / voice take the tables as **graph inputs** (unlike front's baking — ADR 0013)."""
        import inspect

        from sbv2 import export as export_sbv2

        names = list(inspect.signature(wrapper.forward).parameters)[1:]

        assert names == list(export_sbv2.FLOW_INPUT_ORDER)
        assert names[-2:] == ["idx_k", "valid"]


# NOTE: there is **no** test that restates the index tables themselves
# (`idx_k = clamp(w + j − i, 0, 2w)` / `idx_v = i + c`) as formulas and checks them — checking a
# formula copied from the patch's code only says the copy is right (the comparison with the
# original is the check). Both a reversed direction and an out-of-range index are caught by
# `test_it_matches_the_original_forward` above (torch itself raises on an out-of-range gather).
