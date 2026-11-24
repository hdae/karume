"""The shared parts of the Wan reference runs on diffusers 0.39.0 (ADR 0118 decisions 3 / 4 / 5):
the form of the text context, the reference settings and the MPS guard.

The reference scripts (`wan.ti2v_few_step_ref`, `wan.ti2v_i2v_few_step_ref`, `wan.dit_common`,
`wan.dit_probe`) build and run diffusers' pipelines and modules **unmodified** on CPU f32 without
the text encoder (umT5-XXL is not held next to the DiT in host RAM — decision 4), and feed the
precomputed text embeddings in upstream's shape ({@link pad_text_embeds}). The text-embedding
writer (`wan.text_embeds`) and the fixture generators of the TS host (`wan.scheduler_ref`,
`wan.dit_host_fixture`) read the same constants.

MUST: do not build a reference on a machine where MPS is visible ({@link assert_no_mps}). diffusers'
`WanRotaryPosEmbed` builds the RoPE table in float32 only on machines where MPS is available and in
float64 everywhere else (`transformer_wan.py:375`). It is pinned to the float64 side so that the
reference and the base tables do not depend on the machine (decision 3).
"""

from __future__ import annotations

import torch

#: The length of upstream's text context (the default `max_sequence_length` of
#: `WanPipeline.__call__`).
MAX_SEQUENCE_LENGTH = 512

#: The hidden size of umT5-XXL (`text_dim` in the transformer config).
TEXT_DIM = 4096

#: The settings for making reference outputs (decision 5). shift comes from the scheduler config,
#: so it is not here.
GUIDANCE_SCALE = 5.0
NUM_INFERENCE_STEPS = 50


class MpsVisibleError(RuntimeError):
    """A reference was about to be built on a machine where MPS is visible (the RoPE table would be
    float32 — decision 3)."""


def assert_no_mps() -> None:
    """Checks the condition under which the RoPE table is built in float64 (no MPS)."""
    if torch.backends.mps.is_available():
        raise MpsVisibleError(
            "MPS is visible — diffusers' WanRotaryPosEmbed builds the RoPE table in float32, and"
            " the reference differs from a float64 machine (ADR 0118 decision 3). Run on a"
            " machine without MPS"
        )


def pad_text_embeds(embeds: torch.Tensor) -> torch.Tensor:
    """Valid-length embeddings `[L_valid, 4096]` → upstream's `[1, 512, 4096]` (zero-padded at the
    end).

    Upstream's `_get_t5_prompt_embeds` cuts the umT5 output at the valid length and then pads it to
    512 with `new_zeros`. This builds it in the same order (the cut values → concatenated zeros).
    Anything other than f32 fails loudly (no rounding the reference does not have — decision 4).
    """
    if embeds.dtype != torch.float32:
        raise ValueError(f"pass the embeddings in f32 (got {embeds.dtype})")
    if embeds.dim() != 2 or embeds.shape[1] != TEXT_DIM:
        raise ValueError(
            f"pass the embeddings as [L_valid, {TEXT_DIM}] (got {tuple(embeds.shape)})"
        )
    length = embeds.shape[0]
    if not 0 < length <= MAX_SEQUENCE_LENGTH:
        raise ValueError(f"valid length {length} is outside 1..{MAX_SEQUENCE_LENGTH}")
    padding = embeds.new_zeros(MAX_SEQUENCE_LENGTH - length, TEXT_DIM)
    return torch.cat([embeds, padding]).unsqueeze(0)
