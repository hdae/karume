"""`moss/dit_patch.py` — the RoPE operand, the base tables and their assembly, and the patched
graphs against upstream's own forward (on a tiny `MOSSImageTransformer` of the pinned pipeline code;
the real-size check runs in `moss.export_dit`).

The pipeline code comes from the pinned snapshot ({@link moss_snapshot}; SKIP when not cached).
"""

from __future__ import annotations

import inspect

import pytest
import torch

from moss import dit_patch, export_dit, pipeline_ref
from moss.prompts import NEGATIVE, POSITIVE
from moss.records import bits_equal
from moss.tests.tiny import TINY, tiny_model


@pytest.fixture(scope="module")
def module(moss_snapshot):
    pytest.importorskip("diffusers")
    from moss import sources

    return sources.load_pipeline_module(moss_snapshot)


@pytest.fixture(scope="module")
def tiny(module):
    return tiny_model(module)


@pytest.fixture(scope="module")
def graphs(tiny):
    return {name: export_dit.wrap(name, tiny) for name in export_dit.GRAPHS}


@pytest.fixture(scope="module")
def tiny_rope_base(module):
    head_dim = TINY["dim"] // TINY["num_heads"]
    return dit_patch.rope_base_tables(module._RoPE(head_dim, 1e4), max_text=64, max_side=32)


class TestSkippedBlock:
    def test_it_is_the_negative_branch_default_of_the_pinned_pipeline(self, module):
        """The eager comparison feeds `SKIP_BLOCK` to upstream and to `dit_skip22` alike, so it
        cannot catch a wrong value; this pins its source to upstream's `__call__` signature."""
        parameters = inspect.signature(module.MOSSImagePipeline.__call__).parameters
        default = parameters["uncond_skip_layers"].default

        assert default == (22,)
        assert default == pipeline_ref.UNCOND_SKIP_LAYERS
        assert default == (dit_patch.SKIP_BLOCK,)


class TestRotate:
    def test_upstreams_own_table_still_reaches_upstreams_rotate(self, module, tiny):
        """Installing the patch must not move upstream's forward (the references run it)."""
        dit_patch.install_rope_pair(module)
        upstream = dit_patch.upstream_rotate(module)
        dit_patch.install_rope_pair(module)  # idempotent: still one level over upstream
        x = torch.randn(1, 7, 2, 16)
        rope = tiny.rope(1, 3, 4, torch.device("cpu"))

        assert module._rotate is not upstream
        assert dit_patch.upstream_rotate(module) is upstream
        assert bits_equal(module._rotate(x, rope), upstream(x, rope))

    def test_the_pair_form_is_upstreams_expression_to_the_bit(self, module, tiny):
        dit_patch.install_rope_pair(module)
        upstream = dit_patch.upstream_rotate(module)
        x = torch.randn(1, 7, 2, 16)
        rope = tiny.rope(1, 3, 4, torch.device("cpu"))

        pair = dit_patch.rope_pair(*dit_patch.split_rope(rope))

        assert bits_equal(module._rotate(x, pair), upstream(x, rope))


#: `(H, W, T)` in latent cells and tokens: odd / even sides, a transpose pair, the minimal 1×1 with
#: T = 1, the multi-threaded 64×64, the longest sides and the longest text.
ROPE_CASES = (
    (16, 16, 10),
    (16, 24, 11),
    (24, 16, 10),
    (15, 17, 12),
    (1, 1, 1),
    (64, 64, 10),
    (64, 64, 4096),
    (128, 128, 256),
    (dit_patch.MAX_SIDE, 1, 1),
    (2, dit_patch.MAX_SIDE - 1, 33),
    (105, 104, 4000),
)


@pytest.fixture(scope="module")
def rope(module):
    """The real-size `_RoPE` (head_dim 128, base 1e4 — `transformer/config.json`)."""
    return module._RoPE(128, 1e4)


@pytest.fixture(scope="module")
def base(rope):
    return dit_patch.rope_base_tables(rope)


class TestRopeBase:
    def test_the_tables_hold_the_documented_rows(self, base):
        side_rows = 2 * dit_patch.MAX_SIDE - 1

        assert {name: tuple(table.shape) for name, table in base.items()} == {
            "sin_r": (1, 16),
            "cos_r": (1, 16),
            "sin_f": (dit_patch.MAX_TEXT_TOKENS + 1, 16),
            "cos_f": (dit_patch.MAX_TEXT_TOKENS + 1, 16),
            "sin_h": (side_rows, 16),
            "cos_h": (side_rows, 16),
            "sin_w": (side_rows, 16),
            "cos_w": (side_rows, 16),
        }

    @pytest.mark.parametrize(("height", "width", "text"), ROPE_CASES)
    def test_the_assembled_table_is_upstreams_table_to_the_bit(
        self, rope, base, height, width, text
    ):
        """The host only reorders upstream's values (ADR 0034 decision 3) — at every position the
        tables cover, whatever upstream call produced it."""
        expected = rope(height, width, text, torch.device("cpu"))

        assert bits_equal(dit_patch.rope_table(base, height, width, text), expected)

    @pytest.mark.parametrize(
        ("height", "width", "text"),
        [(dit_patch.MAX_SIDE + 1, 1, 1), (1, dit_patch.MAX_SIDE + 1, 1), (1, 1, 0), (1, 1, 4097)],
    )
    def test_a_size_beyond_the_tables_is_refused(self, base, height, width, text):
        with pytest.raises(ValueError, match="outside"):
            dit_patch.rope_rows(base, "sin", height, width, text)

    def test_a_table_cut_from_another_axis_is_rejected(self, base):
        """The gate against a column slip: an axis that does not move with its position."""
        broken = dict(base)
        broken["sin_h"] = base["sin_r"].expand_as(base["sin_h"]).contiguous()

        with pytest.raises(AssertionError, match="does not move"):
            dit_patch.assert_rope_base(
                broken, 16, max_text=dit_patch.MAX_TEXT_TOKENS, max_side=dit_patch.MAX_SIDE
            )

    def test_rows_shifted_by_one_are_rejected(self, base):
        broken = dict(base)
        broken["cos_w"] = torch.roll(base["cos_w"], 1, 0)

        with pytest.raises(AssertionError, match="position 0"):
            dit_patch.assert_rope_base(
                broken, 16, max_text=dit_patch.MAX_TEXT_TOKENS, max_side=dit_patch.MAX_SIDE
            )


def _case(seed: int, text: int, height: int = 5, width: int = 6):
    generator = torch.Generator().manual_seed(seed)
    embeds = torch.randn(text, TINY["text_dim"], generator=generator)
    latents = torch.randn(1, TINY["latent_channels"], height, width, generator=generator)
    return embeds, latents


class TestGraphsAgainstUpstream:
    """`moss.export_dit.eager_case` (gate 2) on the tiny DiT: upstream's forward with the two
    M-aligned linears ≡ the patched graphs, for both branches."""

    @pytest.mark.parametrize(("branch", "text"), [(POSITIVE, 7), (NEGATIVE, 10)])
    def test_the_graphs_reproduce_upstreams_forward(
        self, tiny, graphs, tiny_rope_base, branch, text
    ):
        embeds, latents = _case(3, text)

        report = export_dit.eager_case(
            tiny, graphs, tiny_rope_base, embeds, latents, torch.tensor(500.0), branch=branch
        )

        assert report["failures"] == []
        assert report["S"] == 30 + text
        assert report["observation"]["final_norm_input_image_rows"]["bit_identical"]

    def test_an_inverted_image_mask_is_caught(self, tiny, graphs, tiny_rope_base):
        """`where`'s direction is what the comparison checks (fault injection)."""
        embeds, latents = _case(4, 7)
        height, width = latents.shape[2:]
        image = height * width
        with torch.no_grad(), export_dit.Hooks() as hooks, dit_patch.padded_rows(tiny, image + 7):
            hooks.input_of(tiny.time_embedding[0], "time_sinusoid")
            expected = tiny(latents, torch.tensor(500.0), embeds[None]).sample
        rope = dit_patch.rope_table(tiny_rope_base, height, width, 7)
        with torch.no_grad():
            text = graphs[export_dit.DIT_TEXT](embeds[None], *dit_patch.split_rope(rope[:, image:]))
            inputs = list(dit_patch.dit_inputs(latents, text, rope, hooks.values["time_sinusoid"]))
            right = graphs[export_dit.DIT](*inputs)
            inputs[2] = ~inputs[2]
            wrong = graphs[export_dit.DIT](*inputs)

        assert bits_equal(dit_patch.head_rows(right, height, width), expected)
        assert not bits_equal(dit_patch.head_rows(wrong, height, width), expected)

    def test_skipping_another_block_is_caught(self, tiny, graphs, tiny_rope_base):
        embeds, latents = _case(5, 10)
        other = dit_patch.MossDit(tiny, skip_layers=frozenset({21}))

        report = export_dit.eager_case(
            tiny,
            {**graphs, export_dit.DIT_SKIP22: other},
            tiny_rope_base,
            embeds,
            latents,
            torch.tensor(250.0),
            branch=NEGATIVE,
            observe=False,
        )

        assert any(failure.startswith("dit:") for failure in report["failures"])

    def test_the_skip_graph_keeps_upstreams_block_names(self, graphs):
        """Its tensor keys must be `dit`'s (the shared declarations name them)."""
        names = list(graphs[export_dit.DIT_SKIP22].layers)

        assert names == [str(index) for index in range(24) if index != 22]


class TestPaddedRows:
    def test_the_linear_runs_on_the_padded_rows_and_upstreams_rows_come_back(self, tiny):
        seen: list[int] = []
        handle = tiny.video_embedder.register_forward_pre_hook(
            lambda _m, args: seen.append(int(args[0].shape[1]))
        )
        try:
            with torch.no_grad(), dit_patch.padded_rows(tiny, 9):
                out = tiny.video_embedder(torch.randn(1, 4, TINY["latent_channels"]))
        finally:
            handle.remove()

        assert seen == [9]
        assert tuple(out.shape) == (1, 4, TINY["dim"])

    def test_the_modules_are_restored_afterwards(self, tiny):
        before = {name: getattr(tiny, name) for name in dit_patch.PADDED_MODULES}
        with pytest.raises(RuntimeError), dit_patch.padded_rows(tiny, 9):
            raise RuntimeError

        assert {name: getattr(tiny, name) for name in dit_patch.PADDED_MODULES} == before

    def test_more_rows_than_the_padding_fails_loudly(self, tiny):
        with dit_patch.padded_rows(tiny, 3), pytest.raises(ValueError, match="do not fit"):
            tiny.video_embedder(torch.randn(1, 4, TINY["latent_channels"]))
