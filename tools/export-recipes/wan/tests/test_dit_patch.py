"""The contract of `wan/dit_patch.py` (the S form of the DiT) and `wan/export_dit.py`.

It pins the side where a breakage raises no exception and **only the numbers drift silently** (ADR
0118 decision 3, acceptance step 2):

- the real form of RoPE (the interleaved form kept at rank 4) is **bit-identical** to the upstream
  processor
- swapping the attn1 processor does not change upstream's plain forward (the check against a
  vacuous reference)
- the difference between the Linear form of the patch embedding and conv3d stays within the bound
  of a difference in reduction order
- the host patchify / unpatchify / base RoPE tables are bit-identical to upstream's ordering
- the token-form wrapper composed with the host steps gives back the upstream forward
- the shape of the S-form export (one symbol `S`, the input names, the op set, nothing baked in
  that depends on S)
- the attention of the CPU reference and of the patched eager is pinned to the flash path (falling
  back to MATH at S = 32,760 means OOM — it stops instead of falling back silently)
- the i8 series (ADR 0120 decisions 2 / 3): rounding reaches every upstream weight slot, bias /
  norm keep upstream's f32 values, the binding table has i8 and no f16, and the case set and the
  f64 references of every case

Some tests run on a synthetic model (a small randomly initialised `WanTransformer3DModel`), others
on the real weights (fixture `wan_snapshot` — SKIP on a machine without it). The synthetic model's
shape uses **a different value for each of the 3 axes and C** (with square shapes an axis mix-up is
an involution and stays hidden).

`importorskip("diffusers")` is taken inside the tests that depend on diffusers (the helpers and
fixtures that build the synthetic model and upstream parts) — at module level, not even the
invariants that do not depend on diffusers, such as the case table, the host ordering and the
flash pinning, would run under the default sync (without the `wan` group) (the same judgement as
`test_vae_tiling.py`).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional
from torch.nn.attention import SDPBackend

from _shared.paths import REPO_ROOT
from wan import dit_patch, export_dit

#: The shape of the synthetic DiT (head_dim 24 → RoPE t / h / w of 8 / 8 / 8 dimensions).
TINY_DIT = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 2,
    "attention_head_dim": 24,
    "in_channels": 3,
    "out_channels": 3,
    "text_dim": 8,
    "freq_dim": 16,
    "ffn_dim": 32,
    "num_layers": 2,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-6,
    "rope_max_seq_len": 32,
}
#: The synthetic latent `(F, H, W)` → grid F'·H'·W' = 2·3·5 (a different value on each of the 3
#: axes).
TINY_LATENT = (2, 6, 10)

#: The op set observed in the product export (real weights) (`ops` of the `wan.export_dit`
#: summary). The synthetic model has the same structure, so it gives the same set — a new op is a
#: change that needs a vocabulary decision, so it stops here.
EXPECTED_OPS = frozenset(
    {
        "add",
        "attention",
        "cat",
        "gelu_tanh",
        "layer_norm",
        "linear",
        "mul",
        "neg",
        "permute",
        "reshape",
        "rms_norm",
        "sigmoid",
        "slice",
    }
)


def _transformer_wan():
    """The upstream `transformer_wan` module (SKIPs the calling test when diffusers is absent)."""
    return pytest.importorskip("diffusers.models.transformers.transformer_wan")


def _tiny_dit() -> nn.Module:
    WanTransformer3DModel = _transformer_wan().WanTransformer3DModel  # noqa: N806

    torch.manual_seed(20261002)
    return WanTransformer3DModel(**TINY_DIT).to(torch.float32).eval()


def _tiny_inputs(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    latents = torch.randn(1, model.config.in_channels, *TINY_LATENT, generator=generator)
    timestep = torch.tensor([500], dtype=torch.int64)
    embeds = torch.randn(1, 7, model.config.text_dim, generator=generator)
    return latents, timestep, embeds


def _token_inputs(model: nn.Module, latents: torch.Tensor, timestep: torch.Tensor, embeds):
    patch = tuple(int(size) for size in model.config.patch_size)
    return (
        dit_patch.dit_patchify(latents, patch),
        dit_patch.dit_timesteps_proj(model, timestep),
        embeds,
        *dit_patch.dit_rope_tables(model.rope, tuple(latents.shape[2:])),
    )


class TestRealPairRotary:
    """The real form of the interleaved RoPE (`real_pair_rotary` / the swapped processor)."""

    def _attention_and_tables(self, dim: int, heads: int, latent: tuple[int, int, int]):
        upstream = _transformer_wan()
        WanAttention = upstream.WanAttention  # noqa: N806
        WanAttnProcessor = upstream.WanAttnProcessor  # noqa: N806
        WanRotaryPosEmbed = upstream.WanRotaryPosEmbed  # noqa: N806

        torch.manual_seed(11)
        attention = WanAttention(dim, heads, dim // heads, 1e-6, processor=WanAttnProcessor())
        rope = WanRotaryPosEmbed(dim // heads, (1, 2, 2), 1024)
        cos, sin = rope(torch.zeros(1, 1, *latent))
        hidden = torch.randn(1, cos.shape[1], dim)
        return attention.eval(), cos, sin, hidden

    @pytest.mark.parametrize(
        ("dim", "heads", "latent"),
        [(48, 2, (2, 6, 10)), (1536, 12, (2, 16, 24))],
        ids=["tiny", "wan2.1-1.3b-dims"],
    )
    def test_the_real_pair_form_matches_the_upstream_processor_bit_for_bit(
        self, dim: int, heads: int, latent: tuple[int, int, int]
    ) -> None:
        """**Bit-identical** by `a − b = a + (−b)` and the commutativity of addition.

        Also checked with the head_dim 128 of Wan2.1 (RoPE t / h / w = 44 / 42 / 42 dimensions).
        """
        attention, cos, sin, hidden = self._attention_and_tables(dim, heads, latent)
        with torch.no_grad():
            expected = attention(hidden, None, None, (cos, sin))
            dit_patch.install_real_pair_processors(nn.ModuleList([_Holder(attention)]))
            got = attention(hidden, None, None, dit_patch.RealPairRope(cos, sin))

        # The check against a vacuous comparison: if the swap had no effect, `RealPairRope` (a
        # tuple) would go through the upstream path as is, and the comparison would be with itself.
        assert type(attention.processor).__name__ == "WanRealPairAttnProcessor"
        assert torch.equal(got, expected)

    def test_real_pair_rotary_matches_the_upstream_formula_and_a_sign_fault_breaks_it(self) -> None:
        """Independent oracle: compares the upstream formula (strided assignment to the even / odd
        columns) and the real form bit for bit."""
        _, cos, sin, _ = self._attention_and_tables(48, 2, (2, 6, 10))
        x = torch.randn(1, cos.shape[1], 2, 24)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        expected = torch.empty_like(x)
        expected[..., 0::2] = x1 * cos[..., 0::2] - x2 * sin[..., 1::2]
        expected[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 0::2]

        assert torch.equal(dit_patch.real_pair_rotary(x, cos, sin), expected)
        assert not torch.allclose(dit_patch.real_pair_rotary(x, cos, -sin), expected)

    def test_tables_in_another_dtype_than_x_are_rejected(self) -> None:
        """Upstream converts back to the dtype of x with `.type_as(x)`, but the real form does not
        copy that conversion — a call with split dtypes stops instead of silently returning values
        in a different dtype from upstream."""
        x = torch.randn(1, 3, 2, 8, dtype=torch.bfloat16)
        table = torch.ones(1, 3, 1, 8)

        with pytest.raises(NotImplementedError, match="type_as"):
            dit_patch.real_pair_rotary(x, table, table)
        assert dit_patch.real_pair_rotary(x, table.bfloat16(), table.bfloat16()).dtype == x.dtype

    def test_the_upstream_path_is_untouched_after_the_swap(self) -> None:
        """After the swap, plain tuple tables are still delegated to the upstream implementation
        (the reason the order of taking references does not matter)."""
        attention, cos, sin, hidden = self._attention_and_tables(48, 2, (2, 6, 10))
        with torch.no_grad():
            before = attention(hidden, None, None, (cos, sin))
            dit_patch.install_real_pair_processors(nn.ModuleList([_Holder(attention)]))
            after = attention(hidden, None, None, (cos, sin))

        assert torch.equal(after, before)

    def test_the_swap_is_idempotent(self) -> None:
        attention, *_ = self._attention_and_tables(48, 2, (1, 2, 2))
        holder = nn.ModuleList([_Holder(attention)])
        dit_patch.install_real_pair_processors(holder)
        first = attention.processor
        dit_patch.install_real_pair_processors(holder)

        assert attention.processor is first

    def test_real_pair_tables_on_cross_attention_are_rejected(self) -> None:
        attention, cos, sin, hidden = self._attention_and_tables(48, 2, (1, 2, 2))
        dit_patch.install_real_pair_processors(nn.ModuleList([_Holder(attention)]))
        with pytest.raises(NotImplementedError, match="self-attention"):
            attention(hidden, hidden, None, dit_patch.RealPairRope(cos, sin))


class _Holder(nn.Module):
    """The minimal form of "a block with `attn1`" that `install_real_pair_processors` accepts."""

    def __init__(self, attention: nn.Module) -> None:
        super().__init__()
        self.attn1 = attention


class TestPatchEmbedding:
    def test_the_linear_shares_storage_with_the_conv(self) -> None:
        """Rounding (fake-quant) applied to the wrapper reaches the upstream conv too (the reference
        is not left on the original weights).

        The Linear weight is a view of the conv weight, and the bias is the same Parameter.
        """
        model = _tiny_dit()
        linear = dit_patch.patch_embedding_linear(model.patch_embedding)
        with torch.no_grad():
            linear.weight.mul_(0.5)

        assert torch.equal(model.patch_embedding.weight.reshape(linear.weight.shape), linear.weight)
        assert linear.bias is model.patch_embedding.bias

    def test_an_overlapping_conv_is_rejected(self) -> None:
        conv = nn.Conv3d(3, 8, kernel_size=(1, 2, 2), stride=(1, 1, 1))
        with pytest.raises(NotImplementedError, match="k = s"):
            dit_patch.patch_embedding_linear(conv)


class TestHostFunctions:
    def test_patchify_matches_the_upstream_patch_embedding_order(self) -> None:
        """Independent oracle: the same `Conv3d(k = s)` as upstream with 1-hot weights → the window
        order `(c,pt,ph,pw)`."""
        from wan.dit_host_fixture import onehot_patchify

        latents = torch.randn(1, 3, *TINY_LATENT)
        got = dit_patch.dit_patchify(latents, (1, 2, 2))

        assert torch.equal(got, onehot_patchify(latents, (1, 2, 2)))

    def test_unpatchify_matches_an_index_level_oracle(self) -> None:
        """The last dimension of the exit is `(pt, ph, pw, c)`, and the token index is
        `(f·H' + h)·W' + w`.

        Placed independently by the index formula (not a copy of the view chain).
        """
        frames, height, width = TINY_LATENT
        patch_t, patch_h, patch_w = (1, 2, 2)
        channels = 3
        grid = (frames // patch_t, height // patch_h, width // patch_w)
        count = grid[0] * grid[1] * grid[2]
        tokens = torch.arange(float(count * patch_t * patch_h * patch_w * channels)).reshape(
            1, count, -1
        )

        got = dit_patch.dit_unpatchify(tokens, TINY_LATENT, (patch_t, patch_h, patch_w))

        assert tuple(got.shape) == (1, channels, frames, height, width)
        for c in range(channels):
            for f in range(frames):
                for h in range(height):
                    for w in range(width):
                        token = ((f // patch_t) * grid[1] + h // patch_h) * grid[2] + w // patch_w
                        inner = ((f % patch_t) * patch_h + h % patch_h) * patch_w + w % patch_w
                        assert got[0, c, f, h, w] == tokens[0, token, inner * channels + c]

    def test_the_base_tables_rebuild_the_upstream_rope_table_bit_exactly(self) -> None:
        """Reordering the base tables (even columns) and repeating each twice gives back the table
        of upstream `WanRotaryPosEmbed`.

        The settings are full size (head_dim 128, position 1024).
        """
        WanRotaryPosEmbed = _transformer_wan().WanRotaryPosEmbed  # noqa: N806

        rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024)
        latent = (3, 10, 14)
        base = dit_patch.dit_rope_base_tables(rope)
        cos, sin = dit_patch.dit_rope_tables(rope, latent)
        rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        for f in range(latent[0]):
            for h in range(latent[1] // 2):
                for w in range(latent[2] // 2):
                    rows.append(
                        tuple(  # type: ignore[arg-type]
                            torch.cat(
                                [
                                    base[f"{kind}_t"][f],
                                    base[f"{kind}_h"][h],
                                    base[f"{kind}_w"][w],
                                ]
                            ).repeat_interleave(2)
                            for kind in ("cos", "sin")
                        )
                    )

        assert torch.equal(torch.stack([row[0] for row in rows]), cos[0, :, 0])
        assert torch.equal(torch.stack([row[1] for row in rows]), sin[0, :, 0])

    def test_the_base_tables_have_the_upstream_ceiling_and_widths(self) -> None:
        WanRotaryPosEmbed = _transformer_wan().WanRotaryPosEmbed  # noqa: N806

        base = dit_patch.dit_rope_base_tables(WanRotaryPosEmbed(128, (1, 2, 2), 1024))

        assert set(base) == set(dit_patch.ROPE_BASE_KEYS)
        assert {name: tuple(table.shape) for name, table in base.items()} == {
            "cos_t": (1024, 22),
            "sin_t": (1024, 22),
            "cos_h": (1024, 21),
            "sin_h": (1024, 21),
            "cos_w": (1024, 21),
            "sin_w": (1024, 21),
        }

    def test_a_misaligned_base_slice_is_rejected(self) -> None:
        """The check against a vacuous comparison: a slice with shifted block boundaries "does not
        move along that axis", so it fails."""
        WanRotaryPosEmbed = _transformer_wan().WanRotaryPosEmbed  # noqa: N806

        base = dit_patch.dit_rope_base_tables(WanRotaryPosEmbed(24, (1, 2, 2), 32))
        broken = dict(base)
        broken["sin_h"] = torch.zeros_like(base["sin_h"])

        with pytest.raises(AssertionError, match="does not move on this axis"):
            dit_patch._assert_rope_base(broken, 32, (4, 4, 4))


class TestDitTokens:
    def test_the_token_form_composes_back_to_the_upstream_forward_bit_exactly(self) -> None:
        """`unpatchify ∘ S form ∘ patchify` is bit-identical to the upstream forward (synthetic
        model)."""
        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        with torch.no_grad():
            expected = dit_patch.reference_dit(model, latents, timestep, embeds)
            wrapper = dit_patch.WanDitTokens(model)
            tokens = wrapper(*_token_inputs(model, latents, timestep, embeds))
            hidden = model.patch_embedding(latents).flatten(2).transpose(1, 2)
            trunk = wrapper.forward_hidden(
                hidden, *_token_inputs(model, latents, timestep, embeds)[1:]
            )

        assert torch.equal(dit_patch.dit_unpatchify(trunk, TINY_LATENT, (1, 2, 2)), expected)
        got = dit_patch.dit_unpatchify(tokens, TINY_LATENT, (1, 2, 2))
        assert torch.allclose(got, expected, rtol=0, atol=1e-5)

    def test_the_layers_form_ends_with_the_product_output(self) -> None:
        """The last output of the measurement form (per-layer exits) is the same as the product
        wrapper, and the earlier outputs are the outputs of each upstream block."""
        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        inputs = _token_inputs(model, latents, timestep, embeds)
        with torch.no_grad():
            product = dit_patch.WanDitTokens(model)(*inputs)
            layers = dit_patch.WanDitTokensLayers(model)(*inputs)
            hidden = model.patch_embedding(latents).flatten(2).transpose(1, 2)
            _, blocks = dit_patch.reference_dit_layers(model, latents, timestep, embeds)

        assert len(layers) == len(model.blocks) + 1
        assert torch.equal(layers[-1], product)
        assert hidden.shape == layers[0].shape
        for got, want in zip(layers[:-1], blocks, strict=True):
            assert torch.allclose(got, want, rtol=0, atol=1e-5)

    def test_rounding_reaches_every_parameter_but_not_the_rope_tables(self) -> None:
        """f16 rounding reaches every upstream parameter through the wrapper, but not the RoPE
        tables.

        Every parameter includes the conv patch embedding (the wrapper's Linear is a view of it).
        """
        from wan.export_dit import round_to_f16

        model = _tiny_dit()
        rope_before = {name: buffer.clone() for name, buffer in model.rope.named_buffers()}
        round_to_f16(model, dit_patch.WanDitTokens(model))

        weight = model.patch_embedding.weight
        assert torch.equal(weight, weight.half().float())
        for name, buffer in model.rope.named_buffers():
            assert torch.equal(buffer, rope_before[name]), name


class TestExport:
    def test_the_token_graph_has_one_symbol_and_no_resolution_dependent_constant(self) -> None:
        """The S-form IR: one symbol `S`, the input names and shapes, the op set, and not a single
        S-dependent baked-in value.

        The only initializers of rank ≥ 3 are the modulation tables (`scale_shift_table`
        `[1,6,dim]` / `[1,2,dim]`). If the RoPE tables or constants depending on the latent shape
        were folded in (they would be baked at the upper bound of `Dim`), this breaks.
        """
        from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
        from karume.pipeline import export_module
        from karume.verify import assert_op_contracts, assert_runtime_support
        from wan.export_dit import INPUT_NAMES, dynamic_shapes

        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        graph, tensors = export_module(
            dit_patch.WanDitTokens(model),
            _token_inputs(model, latents, timestep, embeds),
            dynamic_shapes=dynamic_shapes(),
            symbol_names=("S",),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )

        assert graph.symbols == ["S"]
        assert [entry.name for entry in graph.inputs] == list(INPUT_NAMES)
        assert [list(entry.shape) for entry in graph.inputs] == [
            [1, "S", 12],
            [1, 16],
            [1, 7, 8],
            [1, "S", 1, 24],
            [1, "S", 1, 24],
        ]
        assert [graph.values[name].shape for name in graph.outputs] == [[1, "S", 12]]
        assert set(graph.required_ops) == EXPECTED_OPS
        assert {name for name, tensor in tensors.items() if tensor.dim() >= 3} == {
            "blocks.0.scale_shift_table",
            "blocks.1.scale_shift_table",
            "scale_shift_table",
        }
        assert_runtime_support(graph)
        assert_op_contracts(graph)

    def test_case_names_are_unique_and_every_role_has_a_case(self) -> None:
        from wan.export_dit import CASES

        names = [spec.name((1, 2, 2)) for spec in CASES]
        assert len(set(names)) == len(names)
        assert {spec.role for spec in CASES} == {
            "band",
            "accept",
            "growth",
            "full-band",
            "full-accept",
        }
        # The example input (the first) is the smaller one (`torch.export` runs one forward while
        # tracing).
        assert names[0].startswith("band-s00192")

    def test_full_size_cases_have_their_own_seeds(self) -> None:
        """The full-size cases are 8 each at 832×480 with 33 frames (S = 14,040 — ADR 0118 step 3)
        and with 81 frames (S = 32,760 — step 8), and to derive the band independently per S their
        seeds differ from every step 2 case (and also between decision and acceptance, and between
        the S values)."""
        from wan.export_dit import CASES

        full = [spec for spec in CASES if spec.full_size]
        assert [spec.name((1, 2, 2)) for spec in full] == [
            "full-band-s14040-t0999",
            "full-band-s14040-t0999-2",
            "full-band-s14040-t0750",
            "full-band-s14040-t0500",
            "full-band-s14040-t0250",
            "full-band-s14040-t0113",
            "full-accept-s14040-t0999",
            "full-accept-s14040-t0600",
            "full-band-s32760-t0999",
            "full-band-s32760-t0999-2",
            "full-band-s32760-t0750",
            "full-band-s32760-t0500",
            "full-band-s32760-t0250",
            "full-band-s32760-t0113",
            "full-accept-s32760-t0999",
            "full-accept-s32760-t0600",
        ]
        others = {spec.seed for spec in CASES if not spec.full_size}
        assert len({spec.seed for spec in full}) == len(full)
        assert not others & {spec.seed for spec in full}

    def test_only_two_full_size_cases_keep_block_outputs(self) -> None:
        """The per-block outputs at full size (2.6 GB per case at S = 14,040) are kept only by one
        decision case and one acceptance case at S = 14,040 (both t = 999). S = 32,760 keeps none
        (the per-layer probe would need about 6 GB of readback staging, which does not fit on the
        B570). S = 192 / 768 keep them for every case (the counterpart of the per-layer record)."""
        from wan.export_dit import CASES

        assert [spec.name((1, 2, 2)) for spec in CASES if spec.full_size and spec.blocks] == [
            "full-band-s14040-t0999",
            "full-accept-s14040-t0999",
        ]
        assert all(spec.blocks for spec in CASES if not spec.full_size)

    def test_the_largest_case_sits_exactly_on_the_declared_ceiling_of_s(self) -> None:
        """The upper bound 32,760 of `Dim("S")` is exactly the 81-frame grid 21·30·52. Every case's
        S is within the export's range, and the largest case is the bound itself (the goldens cover
        the edge of the accepted set).

        The range is read from `range_constraints` of the ExportedProgram (the constraints the
        export actually received), not from the declared constant.
        """
        from wan.export_dit import CASES, DIT_SYM_MAX, dynamic_shapes

        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        program = torch.export.export(
            dit_patch.WanDitTokens(model),
            _token_inputs(model, latents, timestep, embeds),
            dynamic_shapes=dynamic_shapes(),
        )
        ranges = {
            (int(bound.lower), int(bound.upper)) for bound in program.range_constraints.values()
        }
        tokens = [
            frames * (height // 2) * (width // 2)
            for frames, height, width in (spec.latent_shape for spec in CASES)
        ]

        assert ranges == {(2, 32_760)}
        assert DIT_SYM_MAX == 21 * 30 * 52
        assert all(2 <= count <= 32_760 for count in tokens)
        assert max(tokens) == 32_760


class TestFlashAttentionOnly:
    """Pins the attention of the CPU reference and of the patched eager to the flash path
    (`dit_patch.flash_attention_only` — falling back to MATH at S = 32,760 makes one score matrix
    51.5 GB)."""

    @staticmethod
    def _diffusers_layout(sequence: int, dtype: torch.dtype) -> torch.Tensor:
        """The shape of the Wan2.1 attention (12 heads, head_dim 128) in the same layout as the
        diffusers native backend (`[B,H,S,D]` permuted from `[B,S,H,D]` — only the last dimension
        is contiguous). The values do not affect the selection, so they are left uninitialised (no
        pages are touched even at full size)."""
        return torch.empty(1, sequence, 12, 128, dtype=dtype).permute(0, 2, 1, 3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
    @pytest.mark.parametrize("sequence", [192, 14_040, 32_760])
    def test_the_default_choice_is_flash_for_the_wan_attention_shapes(
        self, dtype: torch.dtype, sequence: int
    ) -> None:
        """The premise that pinning does not change the numbers: even without pinning, the default
        selection is flash (self-attention `S×S` and cross-attention `S×512`). If a torch update
        broke this, the existing goldens would hold MATH values, and pinning would change the
        numbers — the goldens would need to be rebuilt."""
        query = self._diffusers_layout(sequence, dtype)
        for key in (query, self._diffusers_layout(512, dtype)):
            assert torch._fused_sdp_choice(query, key, key) == int(SDPBackend.FLASH_ATTENTION)

    def test_an_input_flash_cannot_take_raises_instead_of_falling_back_to_math(self) -> None:
        """Fault injection: a q whose last dimension is non-contiguous (which flash does not take)
        silently falls back to MATH and passes by default, and stops with a RuntimeError under the
        pinning."""
        query = torch.randn(1, 2, 24, 30).transpose(-1, -2)

        assert torch._fused_sdp_choice(query, query, query) == int(SDPBackend.MATH)
        functional.scaled_dot_product_attention(query, query, query)
        with dit_patch.flash_attention_only(), pytest.raises(RuntimeError):
            functional.scaled_dot_product_attention(query, query, query)

    def test_the_fixation_leaves_the_reference_bit_identical(self) -> None:
        """While the default is flash, the pinned reference and the plain forward are bit-identical
        (synthetic model)."""
        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        with torch.no_grad():
            plain = model(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=embeds,
                return_dict=False,
            )[0]
            fixed = dit_patch.reference_dit(model, latents, timestep, embeds)

        assert torch.equal(fixed, plain)

    def test_every_reference_and_eager_attention_runs_with_math_disabled(self, monkeypatch) -> None:
        """The SDPA calls of the f32 reference, the f64 reference and the patched eager
        (`export_dit.eager_report`) all run with MATH disabled (removing the pinning breaks
        this)."""
        from wan.export_dit import Case, CaseSpec, eager_report, round_to_f16

        states: list[bool] = []
        original = functional.scaled_dot_product_attention

        def recording(*args, **kwargs):
            states.append(torch.backends.cuda.math_sdp_enabled())
            return original(*args, **kwargs)

        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        wrapper = dit_patch.WanDitTokens(model)
        round_to_f16(model, wrapper)
        monkeypatch.setattr(functional, "scaled_dot_product_attention", recording)
        with torch.no_grad():
            reference = dit_patch.reference_dit(model, latents, timestep, embeds)
            case = Case(
                name="tiny",
                spec=CaseSpec("band", TINY_LATENT, 500, 7, 0),
                inputs=_token_inputs(model, latents, timestep, embeds),
                latents=latents,
                timestep=timestep,
                reference=reference,
                reference_blocks=[],
                reference_seconds=0.0,
            )
            eager_report(wrapper, model, case)
            dit_patch.reference_dit_f64(model.double(), latents, timestep, embeds)

        # One forward has 2 calls (self / cross) × 2 layers. 12 or more over the 3 paths (more for
        # the re-runs of the eager trunk).
        assert len(states) >= 12
        assert not any(states)
        assert torch.backends.cuda.math_sdp_enabled()


class TestFloat64Reference:
    """The full-size f64 reference (`dit_patch.float64_forward`) — all activations run in f64."""

    def _rounded_f64(self) -> nn.Module:
        from wan.export_dit import round_to_f16

        model = _tiny_dit()
        round_to_f16(model, dit_patch.WanDitTokens(model))
        return model.double()

    def test_the_f64_forward_stays_in_f64_and_restores_tensor_float(self) -> None:
        """The output is f64, differs from the f32 reference only by rounding, and after leaving,
        `Tensor.float()` is the original f32 conversion again."""
        model = self._rounded_f64()
        latents, timestep, embeds = _tiny_inputs(model)
        with torch.no_grad():
            got = dit_patch.reference_dit_f64(model, latents, timestep, embeds)
            expected = dit_patch.reference_dit(model.float(), latents, timestep, embeds)

        assert got.dtype == torch.float64
        assert 0 < float((got - expected.double()).abs().max()) < 1e-5 * float(got.abs().max())
        assert torch.ones(2, dtype=torch.float64).float().dtype == torch.float32
        assert "float" not in vars(torch.Tensor)

    def test_upstream_f32_casts_left_in_place_are_caught(self, monkeypatch) -> None:
        """Removing the `.float()` pass-through (so that upstream's FP32LayerNorm and the f32
        conversion of the residuals run as is) stops it."""
        from contextlib import nullcontext

        model = self._rounded_f64()
        latents, timestep, embeds = _tiny_inputs(model)
        monkeypatch.setattr(dit_patch, "_float_keeps_float64", nullcontext)
        with torch.no_grad(), pytest.raises(AssertionError, match="non-f64 values were created"):
            dit_patch.reference_dit_f64(model, latents, timestep, embeds)

    @pytest.mark.parametrize("shape", [(2, 3), (16,), (1, 4)], ids=["2x3", "freq_dim", "1x4"])
    def test_a_small_f32_value_outside_the_timestep_sinusoid_is_caught(
        self, shape: tuple[int, ...]
    ) -> None:
        """Only the shapes of the time sinusoid (`[1,1]` / `[half]` / `[1,half]` / `[1,freq_dim]`)
        are allowed.

        The synthetic model's freq_dim is 16 — even 16 elements or fewer stop if the shape differs
        (allowing by size would let things slip through).
        """
        model = self._rounded_f64()
        with (
            pytest.raises(AssertionError, match="non-f64 values were created"),
            dit_patch.float64_forward(model),
        ):
            torch.ones(shape)

    def test_a_model_left_in_f32_is_rejected(self) -> None:
        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        with torch.no_grad(), pytest.raises(AssertionError, match="float32"):
            dit_patch.reference_dit_f64(model, latents, timestep, embeds)


#: An eager summary that satisfies every assertion (the same fields as `export_dit.eager_report`).
_PASSING_REPORT = {
    "case": "tiny",
    "reference_max_abs": 1.5,
    "trunk_bit_exact": True,
    "trunk_max_abs_diff": 0.0,
    "patch_embedding_max_abs_diff": 0.0,
    "patch_embedding_within_bound": True,
    "output_finite": True,
    "full_max_abs_diff": 0.0,
}


class TestTheEagerGate:
    """The eager-equivalence check of the writers (`export_dit.eager_failures`) — a summary that
    fails it never reaches publication."""

    def test_a_report_that_keeps_every_claim_passes(self) -> None:
        assert export_dit.eager_failures(_PASSING_REPORT) == []
        assert (
            export_dit.eager_failures({**_PASSING_REPORT, "reference_f32_vs_f64_ratio": 3e-6}) == []
        )

    @pytest.mark.parametrize(
        ("change", "named"),
        [
            ({"trunk_bit_exact": False}, "is not bit-identical to upstream"),
            ({"patch_embedding_within_bound": False}, "exceeds the reduction-order bound"),
            ({"output_finite": False}, "non-finite values"),
            ({"full_max_abs_diff": float("nan")}, "full_max_abs_diff"),
            ({"reference_f32_vs_f64_ratio": float("inf")}, "reference_f32_vs_f64_ratio"),
            ({"trunk_bit_exact": None}, "is not bit-identical to upstream"),
        ],
        ids=["trunk", "patch-bound", "output", "nan-diff", "inf-ratio", "missing-claim"],
    )
    def test_each_broken_claim_is_named(self, change: dict, named: str) -> None:
        failures = export_dit.eager_failures({**_PASSING_REPORT, **change})

        assert len(failures) == 1
        assert named in failures[0]

    def test_a_missing_numeric_field_is_a_failure(self) -> None:
        report = {
            key: value for key, value in _PASSING_REPORT.items() if key != "trunk_max_abs_diff"
        }

        assert export_dit.eager_failures(report) == [
            "trunk_max_abs_diff is not a finite number (None)"
        ]

    def test_the_patch_embedding_bound_holds_and_a_misordered_window_breaks_it(self) -> None:
        """The difference between the Linear form and conv3d is within the bound. An input with a
        mixed-up window order exceeds the bound by far."""
        model = _tiny_dit()
        latents, *_ = _tiny_inputs(model)
        linear = dit_patch.patch_embedding_linear(model.patch_embedding)
        tokens = dit_patch.dit_patchify(latents, (1, 2, 2))
        with torch.no_grad():
            hidden = model.patch_embedding(latents).flatten(2).transpose(1, 2)
            bound = export_dit.patch_embedding_error_bound(tokens, linear)
            within = (linear(tokens) - hidden).abs() <= bound
            misordered = (linear(tokens.flip(-1)) - hidden).abs() <= bound

        assert bool(within.all())
        assert not bool(misordered.any())


#: The writer cases run on the synthetic model (one per role-specific writing path — the full-size
#: role has the f64 reference).
_TINY_CASES = (
    export_dit.CaseSpec("band", TINY_LATENT, 500, 7, 1),
    export_dit.CaseSpec("full-band", TINY_LATENT, 999, 5, 2, blocks=False),
    export_dit.CaseSpec("full-accept", TINY_LATENT, 600, 3, 3),
)
_TINY_NAMES = [spec.name((1, 2, 2)) for spec in _TINY_CASES]


def _tiny_writer(monkeypatch: pytest.MonkeyPatch, series: Path) -> None:
    """Stub `export_dit`'s emit / verify so they run on the synthetic model (no weights are read).

    Only the input sizes (text width 4096, 512 rows), the source, the location and the case table
    are replaced; the reference, eager, writing and export are the real ones.
    """
    monkeypatch.setattr(export_dit, "CASES", _TINY_CASES)
    monkeypatch.setattr(export_dit, "load_transformer", lambda _name: _tiny_dit())
    monkeypatch.setattr(export_dit, "SERIES", series)
    monkeypatch.setattr(export_dit, "PROBE_SERIES", series.with_name("probe"))
    monkeypatch.setattr(export_dit, "I8_SERIES", series.with_name("i8"))
    # The full-size grid of the i8 series (33 frames for real) is pointed at the synthetic grid —
    # one round also covers the role-specific writing of the full-size role.
    monkeypatch.setattr(export_dit, "I8_FULL_LATENTS", (TINY_LATENT,))
    monkeypatch.setattr(export_dit, "TEXT_DIM", TINY_DIT["text_dim"])
    monkeypatch.setattr(export_dit, "pad_text_embeds", lambda embeds: embeds.unsqueeze(0))


def _emit_args(dtype: str = "f16", *, no_full: bool = False) -> argparse.Namespace:
    """The arguments of `emit` (the same fields as the CLI's `main` builds)."""
    return argparse.Namespace(
        model=export_dit.DEFAULT_MODEL, dtype=dtype, layers=False, no_full=no_full, check=False
    )


def _break_the_rope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fault injection: move the values of the real-form RoPE by about 1 ULP (a stand-in for a
    porting error).

    It breaks the bit identity of the trunk.
    """
    original = dit_patch.real_pair_rotary
    monkeypatch.setattr(
        dit_patch, "real_pair_rotary", lambda x, cos, sin: original(x, cos, sin) * (1 + 2**-20)
    )


@pytest.fixture(scope="module")
def emitted(tmp_path_factory) -> tuple[Path, dict]:
    """The component directory and the summary of a series written by one emit round on the
    synthetic model."""
    series = tmp_path_factory.mktemp("emit") / "series"
    with pytest.MonkeyPatch.context() as monkeypatch:
        _tiny_writer(monkeypatch, series)
        summary = export_dit.emit(_emit_args())
    return series / export_dit.TARGET, summary


class TestTheCaseWriters:
    """The writers (`_write_case_files` / `float64_references` / `emit`) write goldens according to
    the role.

    Regressions that would first split in the GPU e2e after the full-size generation (hours) — a
    missing f64 reference, storage left in f64, `block.NN` for a `blocks=False` case — fail first in
    one round on the synthetic model.
    """

    def test_every_case_gets_an_io_and_a_reference_file(self, emitted) -> None:
        out_dir, summary = emitted

        expected = [
            f"{prefix}{name}{export_dit.CASE_SUFFIX}"
            for name in _TINY_NAMES
            for prefix in (export_dit.IO_PREFIX, export_dit.REFERENCE_PREFIX)
        ]
        assert summary["io"] == expected
        assert all((out_dir / name).is_file() for name in expected)
        assert [report["case"] for report in summary["eager"]] == _TINY_NAMES

    def test_the_io_file_holds_the_graph_inputs_and_the_patched_output(self, emitted) -> None:
        out_dir, _ = emitted
        for name in _TINY_NAMES:
            io = load_file(out_dir / f"{export_dit.IO_PREFIX}{name}{export_dit.CASE_SUFFIX}")

            assert set(io) == {
                *(f"{export_dit.INPUT_PREFIX}{key}" for key in export_dit.INPUT_NAMES),
                f"{export_dit.OUTPUT_PREFIX}0",
            }, name

    def test_block_outputs_and_the_f64_reference_follow_the_case_role(self, emitted) -> None:
        """`block.NN` only for the `blocks` cases; `output.f64` only for the cases of the full-size
        role."""
        out_dir, _ = emitted
        layers = TINY_DIT["num_layers"]
        for spec, name in zip(_TINY_CASES, _TINY_NAMES, strict=True):
            reference = load_file(
                out_dir / f"{export_dit.REFERENCE_PREFIX}{name}{export_dit.CASE_SUFFIX}"
            )
            blocks = {f"block.{index:02d}" for index in range(layers)} if spec.blocks else set()
            f64 = {export_dit.REFERENCE_F64_KEY} if spec.full_size else set()

            assert set(reference) == {"latents", "timestep", "output", *blocks, *f64}, name

    def test_the_f64_reference_is_stored_as_f32_and_differs_from_the_f32_reference(
        self, emitted
    ) -> None:
        """The TS safetensors does not read F64 — it is rounded to f32 and written. Its contents
        come from a forward whose activations are also f64, not a copy of the CPU f32 reference
        (they differ by the rounding)."""
        out_dir, _ = emitted
        for spec, name in zip(_TINY_CASES, _TINY_NAMES, strict=True):
            if not spec.full_size:
                continue
            reference = load_file(
                out_dir / f"{export_dit.REFERENCE_PREFIX}{name}{export_dit.CASE_SUFFIX}"
            )
            f64, f32 = reference[export_dit.REFERENCE_F64_KEY], reference["output"]
            distance = float((f64 - f32).abs().max())

            assert f64.dtype == torch.float32, name
            assert 0 < distance < 1e-5 * float(f64.abs().max()), name

    def test_the_f64_key_is_spelled_like_the_typescript_reader(self) -> None:
        source = (REPO_ROOT / "packages/e2e/tests/e2e_wan_dit_test.ts").read_text()
        declared = re.findall(r'^const REFERENCE_F64_KEY = "([^"]+)";$', source, re.MULTILINE)

        assert declared == [export_dit.REFERENCE_F64_KEY]

    def test_every_eager_report_passed_the_gate(self, emitted) -> None:
        _, summary = emitted

        assert all(export_dit.eager_failures(report) == [] for report in summary["eager"])


class TestTheWriterStopsBeforePublication:
    """If the eager equivalence breaks, nothing is published (emit), and it exits non-zero
    (--verify)."""

    def test_a_broken_rope_leaves_the_existing_series_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        series = tmp_path / "series"
        (series / export_dit.TARGET).mkdir(parents=True)
        (series / export_dit.TARGET / "previous.krm").write_text("last known good")
        _tiny_writer(monkeypatch, series)
        _break_the_rope(monkeypatch)

        with pytest.raises(
            export_dit.EagerEquivalenceError, match="is not bit-identical to upstream"
        ):
            export_dit.emit(_emit_args())

        assert sorted(path.name for path in series.iterdir()) == [export_dit.TARGET]
        assert [path.name for path in (series / export_dit.TARGET).iterdir()] == ["previous.krm"]

    @pytest.mark.parametrize("broken", [False, True], ids=["intact", "broken-rope"])
    def test_verify_exits_non_zero_only_when_a_claim_breaks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: bool
    ) -> None:
        _tiny_writer(monkeypatch, tmp_path / "series")
        if broken:
            _break_the_rope(monkeypatch)

        assert export_dit.main(["--verify"]) == (1 if broken else 0)


# ---- the i8 series (ADR 0120 decisions 2 / 3) --------------------------------------------


def _weight_slots(model: nn.Module) -> dict[str, torch.Tensor]:
    """The upstream weight slots (the `weight` of the Linears and of the patch-embedding Conv3d) by
    FQN."""
    return {
        f"{name}.weight": module.weight
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv3d))
    }


class TestTheI8Rounding:
    """`export_dit.fake_quant_i8` — rounding reaches every upstream weight slot, and everything else
    keeps upstream's f32 values (decision 2 — no f16 rounding mixed in)."""

    def test_it_reaches_every_weight_slot_and_leaves_everything_else_at_the_source_value(
        self,
    ) -> None:
        from karume.quantize import channel_scale, quantize_to_int8

        model = _tiny_dit()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        report = export_dit.fake_quant_i8(model, dit_patch.WanDitTokens(model))

        slots = _weight_slots(model)
        assert report.modules == len(slots)
        for name, weight in slots.items():
            scale = channel_scale(weight, 0)
            requantized = quantize_to_int8(weight, scale).to(torch.float32) * scale
            assert torch.equal(requantized, weight), name
            assert not torch.equal(weight, before[name]), name
        untouched = {name for name in before if name not in slots}
        # bias, norm weights, scale_shift_table and the RoPE tables (the rope buffers).
        assert any(name.endswith(".bias") for name in untouched)
        assert "scale_shift_table" in untouched
        for name, tensor in model.state_dict().items():
            if name in untouched:
                assert torch.equal(tensor, before[name]), name

    def test_a_patch_embedding_copied_instead_of_viewed_is_caught(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fault injection: if the wrapper's patch embedding were a copy of the conv weights, only
        the reference (the upstream conv) would stay outside the rounding."""
        original = dit_patch.patch_embedding_linear

        def copied(conv: nn.Conv3d) -> nn.Linear:
            linear = original(conv)
            linear.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
            return linear

        monkeypatch.setattr(dit_patch, "patch_embedding_linear", copied)
        model = _tiny_dit()
        with pytest.raises(AssertionError, match=r"patch_embedding\.weight"):
            export_dit.fake_quant_i8(model, dit_patch.WanDitTokens(model))


class TestTheI8Cases:
    def test_the_i8_series_retakes_the_s192_and_s14040_cases_of_the_f16_series(self) -> None:
        """Decision 3: the S = 192 set (including the one S = 128 acceptance case) and the 8 cases
        at S = 14,040, with the same rows. `growth` (S = 768) and S = 32,760 are not taken."""
        names = [spec.name((1, 2, 2)) for spec in export_dit.series_cases("i8")]

        assert names == [
            "band-s00192-t0999",
            "band-s00192-t0750",
            "band-s00192-t0500",
            "band-s00192-t0250",
            "band-s00192-t0600",
            "band-s00192-t0113",
            "accept-s00192-t0600",
            "accept-s00128-t0030",
            "accept-s00192-t0400",
            "full-band-s14040-t0999",
            "full-band-s14040-t0999-2",
            "full-band-s14040-t0750",
            "full-band-s14040-t0500",
            "full-band-s14040-t0250",
            "full-band-s14040-t0113",
            "full-accept-s14040-t0999",
            "full-accept-s14040-t0600",
        ]
        assert set(export_dit.series_cases("i8")) <= set(export_dit.CASES)

    def test_every_i8_case_has_a_float64_reference_but_only_full_size_f16_cases_do(self) -> None:
        assert all(export_dit.wants_float64("i8", spec) for spec in export_dit.series_cases("i8"))
        assert [spec for spec in export_dit.CASES if export_dit.wants_float64("f16", spec)] == [
            spec for spec in export_dit.CASES if spec.full_size
        ]

    def test_no_full_keeps_the_s192_cases_only(self) -> None:
        small = export_dit.series_cases("i8", full=False)

        assert small == export_dit.series_cases("i8")[:9]
        assert not any(spec.full_size for spec in small)
        assert export_dit.series_cases("f16") == export_dit.CASES

    @pytest.mark.parametrize(
        "argv",
        [
            ["--no-full"],
            ["--dtype", "i8", "--layers"],
            ["--dtype", "i8", "--verify", "--no-f16"],
            ["--no-f16"],
        ],
        ids=["no-full-on-f16", "i8-layers", "i8-no-f16", "no-f16-without-verify"],
    )
    def test_the_cli_refuses_flags_outside_their_series(self, argv: list[str]) -> None:
        """`--no-full` would be a way to replace the complete goldens of the f16 series with an
        incomplete set, so it is i8 only."""
        with pytest.raises(SystemExit):
            export_dit.main(argv)


@pytest.fixture(scope="module")
def emitted_i8(tmp_path_factory) -> tuple[Path, dict]:
    """The component directory and the summary of a series written by one i8 emit round on the
    synthetic model."""
    series = tmp_path_factory.mktemp("emit-i8") / "series"
    with pytest.MonkeyPatch.context() as monkeypatch:
        _tiny_writer(monkeypatch, series)
        summary = export_dit.emit(_emit_args("i8"))
    return series.with_name("i8") / export_dit.TARGET, summary


class TestTheI8Writer:
    """The writer of the i8 series (`emit --dtype i8`) — location, storage and the shape of the
    goldens."""

    def test_it_writes_its_own_series_and_leaves_the_f16_series_alone(self, emitted_i8) -> None:
        out_dir, summary = emitted_i8

        assert summary["dtype"] == "i8"
        assert summary["dir"] == str(out_dir)
        assert not (out_dir.parent.with_name("series")).exists()

    def test_the_linear_weights_are_stored_as_i8_and_nothing_as_f16(self, emitted_i8) -> None:
        """The binding table has i8 and no f16 / i4. The number of i8 entries is the number of
        linears (the scales are companions)."""
        from _shared.container_read import read_layouts

        out_dir, summary = emitted_i8
        layouts = read_layouts(out_dir / export_dit.MODEL_FILE)
        wrapper = dit_patch.WanDitTokens(_tiny_dit())
        linears = sum(isinstance(module, nn.Linear) for module in wrapper.modules())

        assert set(layouts.values()) == {"f32", "i8"}
        assert sum(layout == "i8" for layout in layouts.values()) == linears
        assert summary["compressed_tensors"] == linears

    def test_biases_keep_the_source_float32_values(self, emitted_i8) -> None:
        """Decision 2: the i8 series applies no f16 rounding — the biases hold upstream's f32
        values bit for bit."""
        from _shared.container_read import read_stored

        out_dir, _ = emitted_i8
        stored = read_stored(out_dir / export_dit.MODEL_FILE)
        source = _tiny_dit().state_dict()
        name = "blocks.0.attn1.to_q.bias"
        value = torch.frombuffer(bytearray(stored[name].payload), dtype=torch.float32)

        assert stored[name].layout == "f32"
        assert torch.equal(value, source[name])
        assert not torch.equal(value, source[name].half().float())

    def test_every_case_carries_the_float64_reference(self, emitted_i8) -> None:
        out_dir, summary = emitted_i8
        for name in _TINY_NAMES:
            reference = load_file(
                out_dir / f"{export_dit.REFERENCE_PREFIX}{name}{export_dit.CASE_SUFFIX}"
            )

            assert export_dit.REFERENCE_F64_KEY in reference, name
        assert all("reference_f32_vs_f64_ratio" in report for report in summary["eager"])
        assert all(export_dit.eager_failures(report) == [] for report in summary["eager"])

    def test_no_full_writes_the_small_cases_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _tiny_writer(monkeypatch, tmp_path / "series")

        summary = export_dit.emit(_emit_args("i8", no_full=True))

        small = [spec.name((1, 2, 2)) for spec in _TINY_CASES if not spec.full_size]
        assert [report["case"] for report in summary["eager"]] == small

    @pytest.mark.parametrize("broken", [False, True], ids=["intact", "broken-rope"])
    def test_verify_applies_the_same_gate_to_the_i8_weights(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: bool
    ) -> None:
        _tiny_writer(monkeypatch, tmp_path / "series")
        if broken:
            _break_the_rope(monkeypatch)

        assert export_dit.main(["--verify", "--dtype", "i8"]) == (1 if broken else 0)


# ---- real weights (pinned revision — SKIP on a machine without them) --------------------------


@pytest.fixture(scope="module")
def wan_transformer(wan_snapshot) -> nn.Module:
    """The real-weight DiT (CPU f32, untouched). MUST: tests do not rewrite it (rounding and so
    on)."""
    pytest.importorskip("diffusers")
    from wan.export_dit import load_transformer

    return load_transformer()


class TestRealWeights:
    def test_the_patched_path_matches_the_upstream_forward(self, wan_transformer) -> None:
        """With the real weights at S = 192 (non-square 2·8·12), every rewrite other than the patch
        embedding is **bit-identical** to upstream.

        Only the Linear form of the patch embedding may differ from conv3d in reduction order (the
        measured difference is 0 with torch CPU on this machine), so the difference is bounded by
        the reduction-order bound `2·γ_{K+1}·Σ|w·x|` (`γ_n = n·u / (1 − n·u)`, `u = 2⁻²⁴`, K = 64
        plus the bias — the same {@link wan.export_dit.patch_embedding_error_bound} as the writer's
        check) — an implementation error (a mixed-up ordering) shows in the magnitude of the values
        themselves.
        """
        from wan.pipeline_ref import pad_text_embeds

        model = wan_transformer
        generator = torch.Generator().manual_seed(1)
        latents = torch.randn(1, 16, 2, 16, 24, generator=generator)
        timestep = torch.tensor([500], dtype=torch.int64)
        embeds = pad_text_embeds(torch.randn(37, 4096, generator=generator))
        inputs = _token_inputs(model, latents, timestep, embeds)
        with torch.no_grad():
            expected = dit_patch.reference_dit(model, latents, timestep, embeds)
            wrapper = dit_patch.WanDitTokens(model)
            hidden = model.patch_embedding(latents).flatten(2).transpose(1, 2)
            trunk = wrapper.forward_hidden(hidden, *inputs[1:])
            embedded = wrapper.patch_embedding(inputs[0])
            bound = export_dit.patch_embedding_error_bound(inputs[0], wrapper.patch_embedding)

        assert torch.equal(dit_patch.dit_unpatchify(trunk, (2, 16, 24), (1, 2, 2)), expected)
        assert bool(((embedded - hidden).abs() <= bound).all())

    def test_the_i8_series_quantizes_the_307_linear_weights(self, wan_transformer) -> None:
        """The i8 targets (the wrapper's weight slots) are 307 linears with 700,480 output
        channels.

        The counts come from ADR 0120 decision 2 and survey §2.2 / §2.3 (the upstream headers). The
        patch embedding is one Linear that is a view of the conv weights, and the Conv3d is not in
        the wrapper (no double counting). It only counts and does not rewrite the weights (the MUST
        of the fixture).
        """
        from karume.quantize import QUANT_MODULE_TYPES, iter_quant_targets

        wrapper = dit_patch.WanDitTokens(wan_transformer)
        targets = list(iter_quant_targets(wrapper, op_types=QUANT_MODULE_TYPES))

        assert len(targets) == 307
        assert sum(int(weight.shape[axis]) for _, weight, axis in targets) == 700_480
        assert not any(isinstance(module, nn.Conv3d) for module in wrapper.modules())
