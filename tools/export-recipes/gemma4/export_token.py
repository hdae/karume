"""Script that writes real-weight Gemma 4 E2B / E4B (`--model`) in the states form with the
**token-only default exit** (ADR 0068 decision 4).

The **default-form** version of `export_decode.py` (the logits opt-in form — runs lm_head over all
M rows and outputs `[logits, token]`). The chunk-series path (how the source tensors are read, the
states surgery, the host-supplied RoPE, mixed quantization, the order of the gates) goes through the
core of {@link gemma4.export_decode} unchanged; this script owns only one wrapper, the variant
description ({@link VARIANT}) and the entry point. The exits differ in three points:

- The inputs gain **`last_row[1]` i32** (the index of the last valid row = `queryLength − 1`). The
  last valid row of a prefill chunk is a runtime scalar, and the graph has no way to slice it
  statically (ADR 0068 addendum 3) — taking it as an i32 graph input is the new wiring decision 4
  anticipated.
- The exit selects one row of the hidden `[1,M,H]` after the final norm with
  **`F.embedding(last_row, hidden[0])`** (the existing `embedding` op — even with a runtime index it
  is a row gather with a fixed last dimension, so no new op is needed), runs lm_head + softcap +
  argmax over that one row only, and outputs **a single `token[1,1,1]`**.
- The full-vocabulary logits are **not declared** as a graph output (the readback is the 4 B token
  only; the lm_head computation also drops to one row — a saving of `(M−1) × 262144 × 1536` MACs
  per prefill chunk).

    uv run --with 'transformers==5.14.1' python -m gemma4.export_token

## This series makes no goldens (the gate is a cross-check against the logits opt-in series)

The greedy expected sequence of the form that outputs no logits **should be identical** to
`greedy.<case>.safetensors` of the logits opt-in series **of the same model**
(`gemma4-<model>-decode/`) (same weights, same rounding, same surgery, and the exit's row
selection points at the same row). The acceptance gate (`e2e_gemma4_token_exit_test.ts`) reuses
them as they are and checks that "the token sequences of the two series match exactly" — the
cross-check between the series is itself the gate, and the torch reference is not recomputed (tens
of minutes per run). This script's sanity check goes as far as comparing each case's **first
continuation token** with the expectation table of the 1-shot script (the same text as
`export_decode`).

## The reused goldens are bound by a provenance record

The premise that makes the reuse valid (**both series come from the same checkpoint**) cannot be
protected by checking that the assets exist — the gate stays green even when only one of them is
stale (full review CX-2.3). So the fingerprint of the source checkpoint and the digest of each
reused golden are written to `reference.json`, and the acceptance gate on the TS side checks them
before it looks at parity (the source of truth is {@link gemma4.provenance}).
MUST: do **not re-export** the logits opt-in series to write the record — only read the existing
goldens.

## Output layout

    outputs/series/gemma4-<model>-decode-token/model.krm        weights, constants + 2 descriptors
    outputs/series/gemma4-<model>-decode-token/reference.json   provenance (fingerprint + goldens)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional

from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from gemma4 import ple
from gemma4.distribution import GEMMA4_DEFAULT_MODEL

#: The series suffix of this script (`gemma4-<model>-decode-token/` — a directory separate from the
#: logits opt-in series; a separate asset with a different exit).
TOKEN_SUFFIX = "decode-token"

#: Default location of the outputs for the default model (E2B).
DEFAULT_OUT_DIR = one_shot.series_dir(GEMMA4_DEFAULT_MODEL, TOKEN_SUFFIX)


class TokenOnlyChunkWrapper(decode.DecodeChunkWrapper):
    """Token-only chunk wrapper `(input_ids, 4 RoPE inputs, last_row) → token[1,1,1]`.

    MUST: a **subclass** of `DecodeChunkWrapper` (identity of the module FQN space — the
    quantization target predicate and the condition for reusing the scale ledger; the same reason
    as in the docstring of `export_decode.DecodeChunkWrapper`).
    MUST: the row selection is `F.embedding` (`aten.embedding` → the existing `embedding` op).
    Passing a tensor to `logits_to_keep` of upstream `Gemma4ForCausalLM.forward` lowers to advanced
    indexing (`aten.index.Tensor` — outside the IR vocabulary), so it cannot be used. The three
    lm_head and softcap lines mirror the upstream forward (modeling_gemma4.py:1895-1899) — same
    config values, same sequence of operations, so the numbers of the selected row match the opt-in
    form.
    """

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        rope_sliding_attention_cos: torch.Tensor,
        rope_sliding_attention_sin: torch.Tensor,
        rope_full_attention_cos: torch.Tensor,
        rope_full_attention_sin: torch.Tensor,
        last_row: torch.Tensor,
    ) -> torch.Tensor:
        length = input_ids.shape[1]
        mask = {
            one_shot.FULL_ATTENTION: one_shot.additive_causal_mask(length),
            one_shot.SLIDING_ATTENTION: one_shot.additive_sliding_mask(length, self.sliding_window),
        }
        embeds = self.model.model.embed_tokens(input_ids)
        stacked = ple.per_layer_inputs(self.per_layer, input_ids, self.per_layer_scale)
        tables = decode.bound_rope(
            rope_sliding_attention_cos,
            rope_sliding_attention_sin,
            rope_full_attention_cos,
            rope_full_attention_sin,
        )
        with self.model.model.rotary_emb.bound(tables):
            hidden = self.model.model(
                inputs_embeds=embeds,
                per_layer_inputs=stacked,
                attention_mask=mask,
                position_ids=None,
                use_cache=False,
            ).last_hidden_state
        # Lift to [1,1,H] after the row selection, then run lm_head — putting the reshape **after**
        # argmax makes a reshape the producer of the output and breaks "token output = argmax
        # directly" (ADR 0068 decision 4 / the check in assert_ir_form_decode).
        rowed = functional.embedding(last_row, hidden[0]).unsqueeze(0)
        logits = self.model.lm_head(rowed)
        cap = float(self.model.config.final_logit_softcapping)
        logits = torch.tanh(logits / cap) * cap
        return logits.argmax(-1, keepdim=True)


#: This script's series (the token-only default exit). The path is owned unchanged by
#: {@link gemma4.export_decode.export_series}; this module passes only the 4 differences.
#:
#: MUST: `goldens=False` — this series records no greedy goldens of its own (the acceptance gate
#: reuses `greedy.<case>.safetensors` of the logits opt-in series and makes the exact match of the
#: two series' token sequences the gate).
VARIANT = decode.ChunkVariant(
    suffix=TOKEN_SUFFIX, wrapper=TokenOnlyChunkWrapper, token_only=True, goldens=False
)


def main(argv: Sequence[str] | None = None) -> None:
    decode.run_variant_cli(VARIANT, __doc__.split("\n\n")[0], argv)


if __name__ == "__main__":
    main()
