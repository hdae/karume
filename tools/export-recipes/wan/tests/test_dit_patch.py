"""The contract of `wan/dit_patch.py` (the S form of the DiT) and the shared export helpers of
`wan/dit_common.py` (the eager-equivalence gate, the weight rounding, the golden key names).

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
  back to MATH at a large S means OOM — it stops instead of falling back silently)
- the i8 rounding (ADR 0120 decision 2): it reaches every upstream weight slot, and bias / norm
  keep upstream's f32 values

The tests run on a synthetic model (a small randomly initialised `WanTransformer3DModel`). The
synthetic model's shape uses **a different value for each of the 3 axes and C** (with square shapes
an axis mix-up is an involution and stays hidden).

`importorskip("diffusers")` is taken inside the tests that depend on diffusers (the helpers and
fixtures that build the synthetic model and upstream parts) — at module level, not even the
invariants that do not depend on diffusers, such as the host ordering and the flash pinning, would
run under the default sync (without the `wan` group) (the same judgement as `test_vae_tiling.py`).
"""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn
from torch.nn import functional
from torch.nn.attention import SDPBackend

from _shared.paths import REPO_ROOT
from wan import dit_common, dit_patch

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

#: The op set of the 5-input S-form export (observed with the real weights). The synthetic model
#: has the same structure, so it gives the same set — a new op is a
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
        ids=["tiny", "head-dim-128"],
    )
    def test_the_real_pair_form_matches_the_upstream_processor_bit_for_bit(
        self, dim: int, heads: int, latent: tuple[int, int, int]
    ) -> None:
        """**Bit-identical** by `a − b = a + (−b)` and the commutativity of addition.

        Also checked with head_dim 128, the head size of the Wan DiTs (RoPE t / h / w = 44 / 42 / 42
        dimensions).
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

    def test_rounding_reaches_every_parameter_but_not_the_rope_tables(self) -> None:
        """f16 rounding reaches every upstream parameter through the wrapper, but not the RoPE
        tables.

        Every parameter includes the conv patch embedding (the wrapper's Linear is a view of it).
        """
        from wan.dit_common import round_to_f16

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
        from wan.dit_common import INPUT_NAMES
        from wan.dit_probe import dynamic_shapes

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


class TestFlashAttentionOnly:
    """Pins the attention of the CPU reference and of the patched eager to the flash path
    (`dit_patch.flash_attention_only` — falling back to MATH at S = 27,280 makes one score matrix
    of tens of GB)."""

    @staticmethod
    def _diffusers_layout(sequence: int, dtype: torch.dtype) -> torch.Tensor:
        """The shape of the Wan2.2 TI2V-5B attention (24 heads, head_dim 128) in the same layout as
        the diffusers native backend (`[B,H,S,D]` permuted from `[B,S,H,D]` — only the last
        dimension is contiguous). The values do not affect the selection, so they are left
        uninitialised (no pages are touched even at full size)."""
        return torch.empty(1, sequence, 24, 128, dtype=dtype).permute(0, 2, 1, 3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
    @pytest.mark.parametrize("sequence", [192, 8_190, 27_280])
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
        (`dit_common.eager_report`) all run with MATH disabled (removing the pinning breaks
        this)."""
        from wan.dit_common import Case, CaseSpec, eager_report, round_to_f16

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
        from wan.dit_common import round_to_f16

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


#: An eager summary that satisfies every assertion (the same fields as `dit_common.eager_report`).
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
    """The eager-equivalence check of the writers (`dit_common.eager_failures`) — a summary that
    fails it never reaches publication."""

    def test_a_report_that_keeps_every_claim_passes(self) -> None:
        assert dit_common.eager_failures(_PASSING_REPORT) == []
        assert (
            dit_common.eager_failures({**_PASSING_REPORT, "reference_f32_vs_f64_ratio": 3e-6}) == []
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
        failures = dit_common.eager_failures({**_PASSING_REPORT, **change})

        assert len(failures) == 1
        assert named in failures[0]

    def test_a_missing_numeric_field_is_a_failure(self) -> None:
        report = {
            key: value for key, value in _PASSING_REPORT.items() if key != "trunk_max_abs_diff"
        }

        assert dit_common.eager_failures(report) == [
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
            bound = dit_common.patch_embedding_error_bound(tokens, linear)
            within = (linear(tokens) - hidden).abs() <= bound
            misordered = (linear(tokens.flip(-1)) - hidden).abs() <= bound

        assert bool(within.all())
        assert not bool(misordered.any())


class TestTheGoldenKeys:
    """The key names of the goldens are the ones the TypeScript readers look up."""

    def test_the_f64_key_is_spelled_like_the_typescript_reader(self) -> None:
        source = (REPO_ROOT / "packages/e2e/tests/e2e_wan_ti2v_dit_test.ts").read_text()
        declared = re.findall(r'^const REFERENCE_F64_KEY = "([^"]+)";$', source, re.MULTILINE)

        assert declared == [dit_common.REFERENCE_F64_KEY]


# ---- the i8 rounding (ADR 0120 decision 2) ------------------------------------------------


def _weight_slots(model: nn.Module) -> dict[str, torch.Tensor]:
    """The upstream weight slots (the `weight` of the Linears and of the patch-embedding Conv3d) by
    FQN."""
    return {
        f"{name}.weight": module.weight
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv3d))
    }


class TestTheI8Rounding:
    """`dit_common.fake_quant_i8` — rounding reaches every upstream weight slot, and everything else
    keeps upstream's f32 values (decision 2 — no f16 rounding mixed in)."""

    def test_it_reaches_every_weight_slot_and_leaves_everything_else_at_the_source_value(
        self,
    ) -> None:
        from karume.quantize import channel_scale, quantize_to_int8

        model = _tiny_dit()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        report = dit_common.fake_quant_i8(model, dit_patch.WanDitTokens(model))

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
            dit_common.fake_quant_i8(model, dit_patch.WanDitTokens(model))
