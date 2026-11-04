"""Writes Gemma 4 E2B's **MTP drafter** (HF `google/gemma-4-E2B-it-assistant`) as a borrower graph.

ADR [0096](../../../docs/decisions/0096-speculative-decoding.md) step 2.

    uv run --with 'transformers==5.14.1' python -m gemma4.export_drafter

## What a borrower graph is (three declarations)

The drafter owns no KV; it **reads the k/v of the target's (the product graph's) l13 (sliding,
window 512) / l14 (full) as they are**. The embedding table it consumes before the output head is
also borrowed from the target's main table. So this graph cannot run on its own; at run time it is
bound to the lender's Session / context:

- the 4 slots of `states` are **external** (the data lives in the lender's context). No
  `state_append`
- attention is in **readonly form** (`ins` is just q; no k/v of the current step and no mask)
- the main table's initializer is **shared** (not a single byte written; it names the lender's
  tensor key)

At trace time the KV comes in as **ordinary input tensors** (torch.export cannot follow state), and
`karume.states.to_external_states_form` drops those 4 inputs and rewrites to the readonly form
(the same discipline as `to_states_form` on the target side — a pure function, checked by verify).

## What one step contains, and the k-step unrolling

    x       = cat(target_embed(token) * sqrt(1536), hidden)        [1,1,3072]
    state   = pre_projection(x)                                    [1,1,256]
    4 layers = q_proj → q_norm → RoPE (one host-supplied row) → SDPA(q, K, V, scale=1.0, mask 0)
              → o_proj / 4 RMSNorm + residual + layer_scalar / MLP gelu_tanh
    state   = norm(state)
    hidden  = post_projection(state)                               [1,1,1536]
    token   = argmax(lm_head(state))                               [1,1,1]

This is unrolled **k = 3 steps** in a Python loop (the only state between steps is
`(token, hidden)` — the drafter has no KV). Each step contributes one token, 3 in total, as graph
outputs (the IR's `cat` is f32 only and cannot concatenate i32).

MUST: load with `use_ordered_embeddings=False`. The distributed checkpoint's sparse centroid softmax
(topk → 4096 candidates → scatter) is not in the IR vocabulary (`topk` is in `NON_EMITTABLE_OPS`),
so it is lowered to **lm_head + argmax over the full vocabulary**. `masked_embedding` is created by
`__init__` from the config, so instead of dropping it after loading, **the config is switched off
before loading** (`Gemma4AssistantForCausalLM.__init__`).

## Quantization (i8 only)

| Weights | Storage | Reason |
| --- | --- | --- |
| all `nn.Linear` (incl. `drafter.lm_head`) | i8 per-channel | see below — i4 cuts acceptance |
| `target_embed` (shared, no bytes) | bf16, not rounded | declared as the lender stores it (MUST) |
| norm / q_norm / layer_scalar | f32 | `fake_quant_int8` sees only the 5 types with a `weight` |

**The drafter has no i4 option.** Measured on 2026-09-08 (E[a](k=3) taken from sequential
acceptance on the same target trajectory, N = 200 cycles × 3 cases): dropping the drafter's linears
(pre / post projection, q_proj, o_proj, gate/up/down = 22 tensors, 10.1M elements) to i4 g32
lowered E[a] to **−33% / −21% / −15%** (in case order) relative to no rounding. The same 22 tensors
in i8 per-channel stay within **−2 to −3%**, and making the shared main table i8 has an effect of
**±0**. The heavily rounded side is the output head (262,144×256 = 67.1M elements), which stays
i8 — raising the 22 linears from i4 g32 (half a byte + an f32 scale per group) to i8 per-channel
(1 byte + an f32 scale per row) grows the distribution by just under 4MB, an amount not comparable
with losing 10 to 30 percent of the acceptance rate (E[a] is exactly the number of tokens
confirmed per speculation cycle).

`target_embed` is **not rounded**: the lender (the product series, `gemma4.export_product`) stores
the checkpoint's bf16 values, which the loaded f32 table already is, so the drafter's goldens see
the values the lender's resident table decodes to. No bytes are written for it. The goldens run
the rounded `wrapper` as it is ({@link export_series}'s order), so the distributed weights and the
goldens' weights are one object (MUST).

## Calling convention of the goldens (the set the host passes in one cycle)

With the lender's logical length `P` (the rows in the KV) and the last confirmed token `b`
(position `P`, **not yet in the KV**), the drafter is passed `token = b` / `hidden = h@(P−1)` (the
post-final-norm hidden of the row that produced `b`) / one RoPE row = position `P` / the KV columns
`[P−min(P,W), P)` and `[0, P)`. This is the same pairing as the drafter's own recursion between
steps (`token` and "the hidden of the row that produced it"), and also the set the step-3
speculative loop passes in its first cycle. So `draft[t][j]` (`j` from 0) predicts
`tokens[t + 1 + j]` — `d₁` predicts the token **after** the bonus token.

## Output layout

    outputs/series/gemma4-e2b-drafter/model.krm                weights + 2 descriptors
    outputs/series/gemma4-e2b-drafter/drafter-golden.<case>.safetensors  expected draft sequences
    outputs/series/gemma4-e2b-drafter/reference.json           provenance (target / drafter)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from _shared.container_read import read_layouts
from _shared.decode_series import PROMPT_KEY
from _shared.paths import REPO_ROOT, SERIES_ROOT
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from gemma4 import export_product as product
from gemma4 import export_unified as unified
from gemma4 import ple, provenance
from gemma4.distribution import (
    GEMMA4_CHUNK_LENGTH,
    GEMMA4_DEFAULT_MODEL,
    GEMMA4_DRAFTER_ROLE,
    GEMMA4_DRAFTER_SUFFIX,
    gemma4_ple_dim,
)
from karume.artifacts import staged_publication
from karume.container import container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.dist import ir_graph
from karume.emit import bakeable_initializers
from karume.ir import IrGraph, IrInitializer, IrShared, IrStorage
from karume.ops import ARGMAX_OP, ATTENTION_OP, EMBEDDING_OP, STATE_APPEND_OP, attention_readonly
from karume.pipeline import publish_model
from karume.quantize import fake_quant_int8
from karume.states import ExternalAttentionSpec, ExternalStatesPlan, to_external_states_form

#: Default location of the outputs (a directory separate from the product series — a separate asset
#: that cannot run on its own).
DEFAULT_OUT_DIR = SERIES_ROOT / "gemma4-e2b-drafter"

#: The lender's (target's) product series. The tensor key of the shared initializer and the real
#: shapes of the external slots are **read from here** (copied by hand, only the borrower would go
#: stale on the day the target is rebuilt).
DEFAULT_PRODUCT_DIR = product.DEFAULT_OUT_DIR

#: The drafter's checkpoint (the HF hub — not hand-placed source files).
ASSISTANT_REPO = "google/gemma-4-E2B-it-assistant"

#: Number of draft steps unrolled in one forward (the distributed drafter is baked with these 3
#: steps. The run-time k can be chosen within `k <= 3`; more steps need a re-export — ADR 0096).
DRAFT_STEPS = 3

# NOTE: the layers whose KV the lender reads (E2B: sliding 13 / full 14) are **not held as
# constants** — `decode.kv_owner_layers(target_config)` derives them from the upstream layer layout
# ({@link lender_slots}).

#: Row count of the K / V inputs placed for the trace. The surgery drops them together with the
#: inputs, so they do not remain in the distribution — any small value works, but `1` could make
#: softmax's reduction axis degenerate, so a fixed value with some room is used.
TRACE_KV_ROWS = 8

#: Graph input names (**the wrapper's forward argument names themselves** — torch.export takes them
#: as the graph input names). The first 6 remain after the surgery, the last 4 are dropped.
DRAFTER_TOKEN = "token"
DRAFTER_HIDDEN = "hidden"
DRAFTER_KV_INPUTS: tuple[str, ...] = ("sliding_k", "sliding_v", "full_k", "full_v")

#: The order of the graph inputs that remain after the surgery (the host binds them by name).
DRAFTER_INPUTS: tuple[str, ...] = (DRAFTER_TOKEN, DRAFTER_HIDDEN, *decode.ROPE_INPUTS)

#: File name and tensor keys of the goldens.
GOLDEN_PREFIX = "drafter-golden."
GOLDEN_SUFFIX = ".safetensors"
TOKENS_KEY = "tokens"
DRAFT_KEY = "draft"

#: Cycles per golden case (each cycle is one `(token, hidden)` pair → k drafts).
GOLDEN_CYCLES = 200

#: Rows per chunk of the golden's prefill. **The same as karume's default chunkLength** (the
#: lender's run accumulates with the same split, so it is the only knob that aligns the order of the
#: sums). The source of truth for the spelling is the distribution recipe.
GOLDEN_PREFILL_CHUNK = GEMMA4_CHUNK_LENGTH

#: Text of the short case (the same conversation as a of E-4 — a one-turn request). A length that
#: ends **inside the 512 window**, the only case that exercises the `P < window` side (fewer
#: columns than the window).
SHORT_PROMPT = (
    "Write a short story (about 300 words) about a lighthouse keeper who discovers a message "
    "in a bottle."
)

#: The texts of the long cases are taken from **documents tracked by git inside the repository**
#: (`(case name, relative path, character limit)`).
#:
#: MUST: keep the source text inside the repository — bench's working area (`outputs/`) is not
#: tracked by git, so reading from there gives "the same script yields a different golden".
#: MUST: cut at a **paragraph boundary** (the last blank line before the limit). Cutting by token
#: count ends mid-sentence, and the continuation degenerates into repeating the same few words,
#: thinning the golden (measured: a 3-word loop "matmul, matmul, matmul" on a 2,560-token cut).
#: At a paragraph boundary the length does not move either, unless the limit is crossed.
CASE_DOCUMENTS: tuple[tuple[str, str, int], ...] = (
    ("readme-recipes", "tools/export-recipes/README.md", 12000),
    ("readme-exporter", "tools/exporter/README.md", 20000),
)

#: Preamble of the long cases (the instruction placed **before** the document), and the request
#: placed after it.
DOCUMENT_INSTRUCTION = "Read the following project document.\n\n"
DOCUMENT_QUESTION = (
    "\n\nBased on the document above, explain in your own words what this project is for and "
    "how its pieces fit together."
)


def truncate_paragraph(text: str, limit: int) -> str:
    """Truncate to the **last paragraph boundary** (a blank line) within `limit` characters."""
    if len(text) <= limit:
        return text.rstrip()
    cut = text.rfind("\n\n", 0, limit)
    if cut <= 0:
        raise AssertionError(f"no paragraph boundary before the {limit}-character limit")
    return text[:cut].rstrip()


# ---- Source material -------------------------------------------------------


def _open_mask(key: torch.Tensor) -> torch.Tensor:
    """Additive mask `[1, 1, 1, N]` allowing every column (0) (N = the column count of K).

    At trace time N is a static integer, so it is folded as **a constant** and pruned once the
    surgery removes it from the node's inputs. When the goldens are taken, K of the real length
    comes in, so the same expression returns a mask of the real length (holding a fixed length with
    `register_buffer` would mismatch the axis the moment it runs at the real length).
    """
    return torch.zeros(1, 1, 1, key.shape[2], dtype=key.dtype, device=key.device)


class DrafterWrapper(nn.Module):
    """Export wrapper `(token, hidden, 4 RoPE inputs, 4 K/V inputs) → (token_0 … token_{k-1})`.

    MUST: the forward's **argument names and order** match {@link DRAFTER_INPUTS} +
    {@link DRAFTER_KV_INPUTS} (torch.export takes the argument names as the graph input names, so a
    different spelling would leave the host unable to bind them).
    MUST: `hidden` is the target's one row **after the final norm** (output 1 of the product graph).
    Being the value before lm_head is the drafter side's input condition, and another intermediate
    value would still match the shape `[1,1536]`.
    MUST: K / V are `[1, Hkv, T, D]` (the same physical shape as the lender's slots). They are
    trace inputs that the surgery replaces with external slots.
    MUST: the mask is **a 0 row allowing every column** (bidirectional with q_len 1 means "see every
    column" — narrowing the columns is the job of the window and the logical length). The column
    count is taken from axis 2 of K ({@link _open_mask}), so it also runs as is at the real length
    when the goldens are taken — the surgery drops it, so not a byte of it remains in the
    distribution.
    MUST: `use_cache=False`, `position_ids=None`. The positions are held by the RoPE hand-over
    inputs ({@link decode.RopeInputs}), so the `arange` upstream builds is read by nobody and dies
    in the export's reachability analysis.
    """

    def __init__(self, drafter: nn.Module, target_embed: nn.Module, steps: int) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError(f"draft step count {steps} is not at least 1")
        self.drafter = drafter
        self.target_embed = target_embed
        self.steps = steps

    def forward(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        rope_sliding_attention_cos: torch.Tensor,
        rope_sliding_attention_sin: torch.Tensor,
        rope_full_attention_cos: torch.Tensor,
        rope_full_attention_sin: torch.Tensor,
        sliding_k: torch.Tensor,
        sliding_v: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        tables = decode.bound_rope(
            rope_sliding_attention_cos,
            rope_sliding_attention_sin,
            rope_full_attention_cos,
            rope_full_attention_sin,
        )
        shared = {
            one_shot.SLIDING_ATTENTION: (sliding_k, sliding_v),
            one_shot.FULL_ATTENTION: (full_k, full_v),
        }
        masks = {
            one_shot.SLIDING_ATTENTION: _open_mask(sliding_k),
            one_shot.FULL_ATTENTION: _open_mask(full_k),
        }
        state = hidden.unsqueeze(1)
        drafted: list[torch.Tensor] = []
        for _ in range(self.steps):
            embedded = self.target_embed(token)
            projected = self.drafter.pre_projection(torch.cat([embedded, state], dim=-1))
            with self.drafter.model.rotary_emb.bound(tables):
                inner = self.drafter.model(
                    inputs_embeds=projected,
                    attention_mask=masks,
                    position_ids=None,
                    use_cache=False,
                    shared_kv_states=shared,
                ).last_hidden_state
            state = self.drafter.post_projection(inner)
            token = self.drafter.lm_head(inner).argmax(-1, keepdim=True)
            drafted.append(token)
            token = token.reshape(1, 1)
        return tuple(drafted)


def target_embedding_class(config: Any) -> type[nn.Module]:
    """Upstream's scaled embedding class of the target text model `config` (its `model_type`):
    `Gemma4TextScaledWordEmbedding` for E2B / E4B (`gemma4_text`),
    `Gemma4UnifiedTextScaledWordEmbedding` for 12B (`gemma4_unified_text`). An unknown model type
    fails loudly."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedTextScaledWordEmbedding,
    )

    classes: dict[str, type[nn.Module]] = {
        "gemma4_text": Gemma4TextScaledWordEmbedding,
        "gemma4_unified_text": Gemma4UnifiedTextScaledWordEmbedding,
    }
    found = classes.get(config.model_type)
    if found is None:
        raise AssertionError(
            f"target text model type {config.model_type!r} has no known embedding"
            f" (known: {sorted(classes)})"
        )
    return found


def load_target_embed(model_dir: Path, config: Any) -> nn.Module:
    """Read the target's main embedding table in f32 and return it as the same **scaled** embedding
    as upstream.

    MUST: use upstream's scaled embedding class of the target itself ({@link
    target_embedding_class} — do not replace it with `nn.Embedding` + a hand-written scalar
    multiply) — that both the value of `embed_scale = sqrt(hidden_size)` and where it is applied are
    upstream's is the basis for the lender using the same table with the same meaning.
    """
    key = one_shot.PLE_CHECKPOINT_KEY.replace("embed_tokens_per_layer", "embed_tokens")
    with safe_open(str(model_dir / one_shot.CHECKPOINT_FILE), framework="pt") as handle:
        if key not in set(handle.keys()):
            raise AssertionError(f"the target checkpoint has no '{key}': {model_dir}")
        weight = handle.get_tensor(key).to(torch.float32)
    vocab, hidden = int(config.vocab_size), int(config.hidden_size)
    if list(weight.shape) != [vocab, hidden]:
        raise AssertionError(
            f"main embedding table is {list(weight.shape)} — not [{vocab}, {hidden}]"
        )
    module = target_embedding_class(config)(
        vocab, hidden, int(config.pad_token_id), embed_scale=hidden**0.5
    )
    module.weight.data = weight
    return module


def assistant_class(config: Any) -> type[nn.Module]:
    """Upstream's MTP assistant class of drafter config `config` (its `model_type`):
    `Gemma4AssistantForCausalLM` for E2B / E4B (`gemma4_assistant`),
    `Gemma4UnifiedAssistantForCausalLM` for 12B (`gemma4_unified_assistant` — the same code under
    another name, over a `gemma4_unified_text` decoder). An unknown model type fails loudly."""
    from transformers import Gemma4AssistantForCausalLM, Gemma4UnifiedAssistantForCausalLM

    classes: dict[str, type[nn.Module]] = {
        "gemma4_assistant": Gemma4AssistantForCausalLM,
        "gemma4_unified_assistant": Gemma4UnifiedAssistantForCausalLM,
    }
    found = classes.get(config.model_type)
    if found is None:
        raise AssertionError(
            f"drafter model type {config.model_type!r} is not an MTP assistant"
            f" (known: {sorted(classes)})"
        )
    return found


def load_drafter(
    steps: int, model_dir: Path, *, source: str | Path = ASSISTANT_REPO
) -> DrafterWrapper:
    """Read the drafter in f32 and return the wrapper with the sparse softmax removed and RoPE
    replaced by the hand-over inputs.

    `source` is what `from_pretrained` reads the drafter from: the regular E2B assistant's
    repository id by default, or a pinned snapshot directory (`gemma4.export_drafter_qat`). The
    class is the one the config names ({@link assistant_class} — 12B's assistant is
    `Gemma4UnifiedAssistantForCausalLM`).

    MUST: switch `use_ordered_embeddings` off **before loading** (`masked_embedding` is created by
    `__init__` from the config, so deleting the attribute afterwards leaves the forward's branch on
    the sparse softmax side).
    MUST: drop the inner `model.embed_tokens`. It is tied, so `lm_head` keeps holding the data;
    without dropping it, the same weight is visible from two FQNs and the i8 rounding is applied
    twice.
    """
    from transformers import AutoConfig

    one_shot.register_attention()
    config = AutoConfig.from_pretrained(source)
    config.use_ordered_embeddings = False
    config._attn_implementation = one_shot.ATTENTION_NAME
    text = config.get_text_config()
    text._attn_implementation = one_shot.ATTENTION_NAME
    drafter = (
        assistant_class(config).from_pretrained(source, config=config, dtype=torch.float32).eval()
    )
    if drafter.masked_embedding is not None:
        raise AssertionError(
            "the sparse softmax (masked_embedding) is still there"
            " — use_ordered_embeddings was not switched off before loading"
        )
    del drafter.model.embed_tokens
    drafter.model.rotary_emb = decode.RopeInputs(decode.unique_layer_types(text))
    target_config = one_shot.load_text_config(model_dir)
    return DrafterWrapper(drafter, load_target_embed(model_dir, target_config), steps).eval()


#: Module FQN of the shared main table (the spelling the assembly of {@link DrafterWrapper}
#: decides) — the one module with a weight slot that {@link quantize_wrapper} leaves unrounded.
TARGET_EMBED_MODULE = "target_embed"


def int8_seats(wrapper: nn.Module) -> set[str]:
    """FQNs of the weights that **should** be rounded to i8 — every `nn.Linear` (nothing else is
    rounded — the shared table stays the lender's bf16 values).

    The expected set, **drawn by a different rule**, for comparing with the ledger of what
    {@link quantize_wrapper} actually rounded (the rounding side selects by the 5 types of
    `quantize.QUANT_CHANNEL_AXES`). There are two situations in which the two sets disagree, and
    both show only in the values:

    - the `drafter.model.embed_tokens` that should have been dropped came back — it is tied, so it
      points at the same data as `lm_head`, and `fake_quant_int8` rounds **the same weight twice**
      in one pass (the scale ledger keeps only the last one, so a weight that does not restore with
      the declared scale goes into the distribution)
    - the drafter's structure changed and a conv or a new embedding came in — it silently drops to
      i8
    """
    return {
        f"{name}.weight"
        for name, module in wrapper.named_modules()
        if isinstance(module, nn.Linear)
    }


def quantize_wrapper(wrapper: nn.Module) -> tuple[Any, dict[str, torch.Tensor]]:
    """Round every module with a weight slot to i8 per-channel, except the shared main table, and
    return the scale ledger.

    MUST: the rounding is a single i8 pass (measured in the module docstring — i4 cuts the
    acceptance rate). Even with a single rounding method, "two FQNs pointing at the same data" are
    rounded twice in one pass, so the ledger actually rounded is compared with {@link int8_seats}
    (the detection line for the day the `del` in `load_drafter` goes missing).
    MUST: the shared table is not rounded — the lender stores the checkpoint's bf16 values, which
    the loaded f32 table already is, so the goldens see what the lender's resident table decodes to.
    """
    report = fake_quant_int8(wrapper, include=lambda name: name != TARGET_EMBED_MODULE)
    rounded, expected = set(report.scales), int8_seats(wrapper)
    if rounded != expected:
        raise AssertionError(
            f"the weights rounded to i8 differ from the expected ones"
            f" (extra {sorted(rounded - expected)} / missing {sorted(expected - rounded)})"
        )
    return report, dict(report.scales)


# ---- Read from the lender (the target's product series) --------------------


@dataclass(frozen=True)
class LenderSlots:
    """The real shapes of the lender's external slots, and what the shared initializer points at."""

    #: Slot name → declared shape (`[1, Hkv, capacity, D]`).
    shapes: Mapping[str, list[Any]]
    #: Layer type → (k slot name, v slot name).
    slots: Mapping[str, tuple[str, str]]
    #: **The lender container's tensor key** of the shared main table (in IR v2 the initializer name
    #: itself).
    shared_tensor: str


def lender_container(product_dir: Path) -> Path:
    """Representative path of the product container (resolving the part list is the reader's
    job)."""
    return product_dir / one_shot.MODEL_FILE


def lender_graph(product_dir: Path) -> Mapping[str, Any]:
    """Read the product container's graph descriptor (**the IR v2 document**) — not a byte of the
    weights is touched."""
    return ir_graph(lender_container(product_dir))


def lender_slots(
    graph: Mapping[str, Any],
    layouts: Mapping[str, str],
    config: Any,
    target_config: Any,
    *,
    shared_storage: IrStorage | None = None,
) -> LenderSlots:
    """Take the lender's real slot shapes and the shared tensor key from the product container's
    declarations.

    `shared_storage` is the borrower's declaration of the shared table (default
    {@link drafter_shared_storage} — the regular product's bf16); the lender's layout must match its
    dtype.

    MUST: read both shapes and names **from the lender** (do not derive them from the drafter's
    config). The borrower's declaration is correct only in "matching the lender"; built
    independently, only the borrower goes stale on the day the target is rebuilt — and stays silent
    as long as the shapes match.

    MUST: the shared main table is found **structurally** as "the `[V, H]` initializer consumed by
    the embedding's weight slot" (the key's spelling is not copied). If it is not exactly one, fail
    loudly.

    `layouts` is tensor key → storage layout ({@link _shared.container_read.read_layouts}) — the
    IR v2 graph descriptor holds no storage (the source of truth is the binding table), so the
    resident form is compared from there.
    """
    owners = decode.kv_owner_layers(target_config)
    states = graph.get("states") or {}
    initializers = graph.get("initializers") or {}
    values = graph.get("values") or {}
    slots: dict[str, tuple[str, str]] = {}
    shapes: dict[str, list[Any]] = {}
    for layer_type in decode.ROPE_LAYER_TYPES:
        if layer_type not in owners:
            raise AssertionError(f"the lender has no KV-owning layer of layer type '{layer_type}'")
        owner = owners[layer_type]
        pair = (decode.slot_name(owner, "k"), decode.slot_name(owner, "v"))
        for name in pair:
            slot = states.get(name)
            if slot is None:
                raise AssertionError(
                    f"the lender graph has no state slot '{name}' (declared {sorted(states)})"
                )
            shapes[name] = list(slot["shape"])
        slots[layer_type] = pair

    vocab, hidden = int(target_config.vocab_size), int(target_config.hidden_size)
    weight_slot_names = {
        node["ins"][0] for node in graph["nodes"] if node["op"] == EMBEDDING_OP and node["ins"]
    }
    candidates = sorted(
        name
        for name in weight_slot_names
        if name in initializers and list(values[name]["shape"]) == [vocab, hidden]
    )
    if len(candidates) != 1:
        raise AssertionError(
            f"main embedding tables of the lender graph (embedding weight slot,"
            f" [{vocab}, {hidden}]): {len(candidates)} found: {candidates}"
        )
    key = candidates[0]
    if initializers[key].get("shared") is True:
        raise AssertionError(
            f"the lender's main table '{key}' holds no data (a shared declaration)"
        )
    layout = layouts.get(key)
    declared = drafter_shared_storage(config) if shared_storage is None else shared_storage
    if layout != declared.dtype:
        raise AssertionError(
            f"the lender's main table is stored as '{layout}' — not the borrower's declared"
            f" '{declared.dtype}'"
        )
    return LenderSlots(shapes=shapes, slots=slots, shared_tensor=key)


def drafter_shared_storage(_config: Any) -> IrStorage:
    """Storage declaration of the shared initializer (MUST match the lender; no scale is
    written) — the regular product's bf16 table (`gemma4.export_product.BF16_STORAGE`)."""
    return IrStorage(dtype=product.BF16_STORAGE)


# ---- Surgery ---------------------------------------------------------------


def external_states_plan(
    graph: IrGraph,
    config: Any,
    lender: LenderSlots,
    *,
    capacity_symbol: str = decode.CAPACITY_SYMBOL,
) -> ExternalStatesPlan:
    """Surgery plan that assigns the attention nodes (layer order = order of appearance) to the
    lender's slots.

    The slot read is determined per layer type (3 sliding layers → the lender's sliding-owning
    layer / 1 full layer → the full-owning layer). The unrolling is k steps, so each slot gets
    `steps × layer count` readers.

    MUST: the slots' shapes are taken **from the lender's declaration** ({@link lender_slots}). The
    window is the borrower config's `sliding_window` (the window width read is the borrower's
    semantics, a different number from the lender's physical capacity).
    """
    layer_types = list(config.layer_types)
    nodes = [node for node in graph.nodes if node.op == ATTENTION_OP]
    if len(nodes) % len(layer_types):
        raise AssertionError(
            f"{len(nodes)} attention nodes — not a multiple of the layer count {len(layer_types)}"
            " (k-step unrolling)"
        )
    window = int(config.sliding_window)
    specs: list[ExternalAttentionSpec] = []
    for index, node in enumerate(nodes):
        layer_type = layer_types[index % len(layer_types)]
        k_slot, v_slot = lender.slots[layer_type]
        shape = lender.shapes[k_slot]
        sliding = layer_type == one_shot.SLIDING_ATTENTION
        k_input, v_input = (
            (DRAFTER_KV_INPUTS[0], DRAFTER_KV_INPUTS[1])
            if sliding
            else (DRAFTER_KV_INPUTS[2], DRAFTER_KV_INPUTS[3])
        )
        specs.append(
            ExternalAttentionSpec(
                output=node.outs[0],
                k_slot=k_slot,
                v_slot=v_slot,
                k_input=k_input,
                v_input=v_input,
                kv_heads=int(shape[1]),
                head_dim=int(shape[3]),
                capacity=shape[2],
                window=window if sliding else None,
            )
        )
    return ExternalStatesPlan(capacity_symbol=capacity_symbol, attentions=tuple(specs))


def share_target_embedding(
    graph: IrGraph,
    tensors: dict[str, torch.Tensor],
    lender: LenderSlots,
    config: Any,
    *,
    storage: IrStorage | None = None,
) -> tuple[IrGraph, dict[str, torch.Tensor], str]:
    """Replace the main table's initializer with a **shared declaration** and remove its data from
    the stored tensors.

    `storage` is the shared declaration's storage (default {@link drafter_shared_storage}; the
    same value the caller handed {@link lender_slots}).

    MUST: the target is exactly one "`[V, H]` initializer consumed by the embedding's weight slot"
    (found structurally on the borrower side too — the FQN-derived tensor keys depend on
    torch.export's internal order).
    MUST: drop the data from `tensors`. Left in, it fails as "extra" in `write_model`'s exact match
    of declarations / storage (= there is a detection line, but dropping it is this function's job).
    """
    vocab, hidden = int(config.vocab_size), int(config.hidden_size)
    consumed = {node.ins[0] for node in graph.nodes if node.op == EMBEDDING_OP and node.ins}
    candidates = sorted(
        name
        for name in consumed
        if name in graph.initializers and list(graph.values[name].shape) == [vocab, hidden]
    )
    if len(candidates) != 1:
        raise AssertionError(
            f"main embedding tables of the borrower graph ([{vocab}, {hidden}]): {len(candidates)}:"
            f" {candidates}"
        )
    name = candidates[0]
    key = graph.initializers[name].tensor
    initializers = dict(graph.initializers)
    initializers[name] = IrInitializer(
        shared=IrShared(tensor=lender.shared_tensor),
        storage=drafter_shared_storage(config) if storage is None else storage,
    )
    remaining = {tensor: value for tensor, value in tensors.items() if tensor != key}
    if len(remaining) != len(tensors) - 1:
        raise AssertionError(f"the main table's data '{key}' is not among the stored tensors")
    shared_graph = IrGraph(
        symbols=list(graph.symbols),
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
        initializers=initializers,
        values=dict(graph.values),
        states=dict(graph.states),
        nodes=list(graph.nodes),
    )
    return shared_graph, remaining, name


# ---- Form checks -----------------------------------------------------------


def assert_ir_form_drafter(
    graph: IrGraph,
    config: Any,
    lender: LenderSlots,
    steps: int,
    *,
    capacity_symbol: str = decode.CAPACITY_SYMBOL,
) -> dict[str, Any]:
    """Check the borrower graph's form (every property that **breaks silently while the numbers
    still match**).

    - the graph inputs are exactly `token` / `hidden` / the 4 RoPE inputs (if a K/V placeholder
      remained, no node would read the value the caller passes)
    - there are k outputs, each an i32 `[1,1,1]` from `argmax` (order = step order, MUST)
    - the state slots are **all external**, matching the lender's l13 / l14 in name and shape
    - **no** `state_append` (the borrower writes not a single row of the lender's ring)
    - every attention is in readonly form, `layer count × step count` of them
    - the initializers hold one shared declaration (no bytes) pointing at the lender's tensor key
    - the symbols are exactly the one capacity symbol (K/V were dropped, so the only binding point
      is the states' shapes)
    """
    names = [spec.name for spec in graph.inputs]
    if names != list(DRAFTER_INPUTS):
        raise AssertionError(
            f"graph inputs are {names}, not {list(DRAFTER_INPUTS)}"
            " (the K/V placeholders may not have been dropped by the surgery)"
        )
    token_spec, hidden_spec = graph.inputs[0], graph.inputs[1]
    backbone = int(config.backbone_hidden_size)
    if token_spec.dtype != "i32" or list(token_spec.shape) != [1, 1]:
        raise AssertionError(
            f"'{DRAFTER_TOKEN}' is {token_spec.dtype} {list(token_spec.shape)} — not i32 [1, 1]"
        )
    if hidden_spec.dtype != "f32" or list(hidden_spec.shape) != [1, backbone]:
        raise AssertionError(
            f"'{DRAFTER_HIDDEN}' is {hidden_spec.dtype} {list(hidden_spec.shape)} —"
            f" not f32 [1, {backbone}] (the target's post-final-norm hidden)"
        )
    text = config.get_text_config()
    specs = decode.rope_specs(text)
    for layer_type in decode.ROPE_LAYER_TYPES:
        expected = [1, 1, specs[layer_type].head_dim]
        for part in decode.ROPE_PARTS:
            name = decode.rope_input_name(layer_type, part)
            spec = next(item for item in graph.inputs if item.name == name)
            if spec.dtype != "f32" or list(spec.shape) != expected:
                raise AssertionError(
                    f"graph input '{name}' is {spec.dtype} {list(spec.shape)} — not f32 {expected}"
                )

    if len(graph.outputs) != steps:
        raise AssertionError(f"{len(graph.outputs)} IR outputs (one per draft step, {steps} steps)")
    producer = {out: node for node in graph.nodes for out in node.outs}
    for index, output in enumerate(graph.outputs):
        source = producer.get(output)
        if source is None or source.op != ARGMAX_OP:
            found = "not a node output" if source is None else source.op
            raise AssertionError(f"the producer of output {index} is {found} — not `{ARGMAX_OP}`")
        value = graph.values[output]
        if value.dtype != "i32" or list(value.shape) != [1, 1, 1]:
            raise AssertionError(
                f"output {index} is {value.dtype} {list(value.shape)} — not i32 [1, 1, 1]"
            )

    expected_slots = {name: shape for name, shape in lender.shapes.items()}
    if sorted(graph.states) != sorted(expected_slots):
        raise AssertionError(
            f"states declared are {sorted(graph.states)} — not the lender's"
            f" {sorted(expected_slots)}"
        )
    for name, shape in sorted(expected_slots.items()):
        slot = graph.states[name]
        if not slot.external:
            raise AssertionError(f"states['{name}'] is not external (the borrower holds no data)")
        if slot.dtype != "f32" or list(slot.shape) != shape:
            raise AssertionError(
                f"states['{name}'] is {slot.dtype} {list(slot.shape)} — differs from the lender's"
                f" f32 {shape}"
            )
    appends = [node for node in graph.nodes if node.op == STATE_APPEND_OP]
    if appends:
        raise AssertionError(
            f"`{STATE_APPEND_OP}` still appears {len(appends)} time(s)"
            " (the borrower writes not a single row of the lender's ring)"
        )

    attentions = [node for node in graph.nodes if node.op == ATTENTION_OP]
    expected_attentions = int(text.num_hidden_layers) * steps
    if len(attentions) != expected_attentions:
        raise AssertionError(
            f"{len(attentions)} attention nodes — not {text.num_hidden_layers} layers ×"
            f" {steps} steps = {expected_attentions}"
        )
    plain = [index for index, node in enumerate(attentions) if not attention_readonly(node.attrs)]
    if plain:
        raise AssertionError(
            f"{len(plain)} attention node(s) that are not readonly remain (indices {plain[:4]})"
        )
    window = int(text.sliding_window)
    for index, node in enumerate(attentions):
        layer_type = list(text.layer_types)[index % int(text.num_hidden_layers)]
        expected_states = dict(zip(("k", "v"), lender.slots[layer_type], strict=True))
        if node.states != expected_states:
            raise AssertionError(
                f"attention[{index}] ({layer_type}): the states field is {node.states}"
                f" (expected {expected_states})"
            )
        expected_window = window if layer_type == one_shot.SLIDING_ATTENTION else None
        if node.attrs.get("window") != expected_window:
            raise AssertionError(
                f"attention[{index}] ({layer_type}): the window in attrs is"
                f" {node.attrs.get('window')} (expected {expected_window})"
            )

    shared = sorted(name for name, init in graph.initializers.items() if init.is_shared)
    if len(shared) != 1:
        raise AssertionError(f"{len(shared)} shared initializers: {shared} (exactly 1 expected)")
    declaration = graph.initializers[shared[0]]
    if declaration.shared.tensor != lender.shared_tensor:
        raise AssertionError(
            f"the shared initializer points at '{declaration.shared.tensor}' —"
            f" not the lender's '{lender.shared_tensor}'"
        )
    if declaration.storage.scale is not None:
        raise AssertionError(
            "the shared initializer declares a scale (the lender's resident weight holds it)"
        )

    if sorted(graph.symbols) != [capacity_symbol]:
        raise AssertionError(f"symbols are {sorted(graph.symbols)} — not ['{capacity_symbol}']")
    residue = sorted(set(graph.required_ops) & set(decode.RESIDUE_OPS))
    if residue:
        raise AssertionError(f"ops that the surgery should have killed remain: {residue}")

    storage: dict[str, int] = {}
    for initializer in graph.initializers.values():
        dtype = initializer.storage.dtype
        storage[dtype] = storage.get(dtype, 0) + 1
    return {
        "inputs": names,
        "outputs": len(graph.outputs),
        "attention_nodes": len(attentions),
        "external_slots": sorted(graph.states),
        "shared_initializer": shared[0],
        "shared_tensor": declaration.shared.tensor,
        "storage": dict(sorted(storage.items())),
    }


# ---- golden -----------------------------------------------------------------


def build_cases(model_dir: Path) -> tuple[tuple[str, torch.Tensor], ...]:
    """`(name, input_ids)` of the golden cases (`<bos>` is added here — the same convention as
    `gemma4.export`).

    The source texts are one short request + 2 **documents tracked by git inside the repository**
    ({@link CASE_DOCUMENTS}). The short case ends inside the 512 window and the 2 long ones cross
    the window by far — exercising both the side where the sliding slot returns `[P − W, P)` while
    wrapping around and the side where it does not wrap.
    """
    tokenizer = one_shot.load_tokenizer(model_dir)
    bos = one_shot.bos_token_id(model_dir, tokenizer)

    def encode(text: str) -> torch.Tensor:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        return torch.tensor([[bos, *ids]], dtype=torch.int64)

    cases: list[tuple[str, torch.Tensor]] = [("short-en", encode(SHORT_PROMPT))]
    for name, relative, limit in CASE_DOCUMENTS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise AssertionError(f"case '{name}': source text {path} is missing")
        document = truncate_paragraph(path.read_text(encoding="utf-8"), limit)
        cases.append((name, encode(DOCUMENT_INSTRUCTION + document + DOCUMENT_QUESTION)))
    return tuple(cases)


@dataclass
class _Lender:
    """The lender when taking goldens (the target with real weights) — a holder that carries the
    cache and the shared KV along.

    `tables` / `scale` are the host's PLE lookup — `None` for a target without per-layer
    embeddings (12B), whose decoder takes no `per_layer_inputs` ({@link _new_lender}).
    """

    model: Any
    tables: Any
    scale: float | None
    specs: Mapping[str, Any]
    cache: Any
    shared: Any
    length: int = 0

    def step(self, ids: torch.Tensor) -> torch.Tensor:
        """Feed `ids[1, L]` into the cache and return the last row's post-norm hidden `[1, H]`."""
        positions = torch.arange(self.length, self.length + int(ids.shape[1]))
        rope = decode.rope_args(self.specs, positions.unsqueeze(0))
        device = self.model.device
        embeds = self.model.model.embed_tokens(ids.to(device))
        per_layer: dict[str, torch.Tensor] = {}
        if self.tables is not None:
            per_layer["per_layer_inputs"] = ple.per_layer_inputs(
                self.tables, ids.to("cpu"), self.scale
            ).to(device)
        with (
            torch.no_grad(),
            self.model.model.rotary_emb.bound(
                decode.bound_rope(*(table.to(device) for table in rope))
            ),
        ):
            out = self.model.model(
                inputs_embeds=embeds,
                **per_layer,
                attention_mask=None,
                position_ids=positions.unsqueeze(0).to(device),
                past_key_values=self.cache,
                use_cache=True,
                shared_kv_states=self.shared,
                return_shared_kv_states=True,
            )
        self.length += int(ids.shape[1])
        return out.last_hidden_state[:, -1]

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.lm_head(hidden)

    def kv(self, layer_type: str, columns: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Cut **exactly the last `columns` columns** out of `shared_kv_states`.

        MUST: the script owns the cut (it is not left to how many rows the cache keeps). HF's
        sliding layers return "the previous `window − 1` rows + this call's", so the row count
        differs between the step form and the chunk form, and passing them as they are would let
        only the goldens see a different set of columns. The runtime's contract is
        `[P − min(P, W), P)` (sliding) / `[0, P)` (full), so this aligns to it.
        """
        key, value = self.shared[layer_type]
        held = int(key.shape[2])
        if held < columns:
            raise AssertionError(
                f"shared KV '{layer_type}' has only {held} columns ({columns} needed)"
            )
        return key[:, :, held - columns :], value[:, :, held - columns :]


def _new_lender(wrapper: Any, specs: Mapping[str, Any]) -> _Lender:
    """The lender for one case (the cache and the shared KV are rebuilt per case).

    Whether the host's PLE lookup runs is read from the target's config
    (`gemma4.distribution.gemma4_ple_dim` — never from whether the wrapper happens to hold the
    tables): a PLE model whose wrapper lost its tables fails here, and a PLE-free one (12B) runs
    without them.
    """
    from transformers import DynamicCache

    config = wrapper.model.config
    with_ple = gemma4_ple_dim(config, "the golden target's config") > 0
    return _Lender(
        model=wrapper.model,
        tables=wrapper.per_layer if with_ple else None,
        scale=wrapper.per_layer_scale if with_ple else None,
        specs=specs,
        cache=DynamicCache(config=config),
        shared={},
    )


def draft_case(
    drafter: DrafterWrapper,
    wrapper: Any,
    specs: Mapping[str, Any],
    ids: torch.Tensor,
    cycles: int,
    *,
    device: str,
    chunk: int = GOLDEN_PREFILL_CHUNK,
    suppress: Sequence[int] = (),
) -> dict[str, torch.Tensor]:
    """Take one case's golden (`prompt` / `tokens` / `draft`).

    `suppress` are the ids the host's greedy selection never picks
    (`pipelineConfig.suppressTokens` — 12B's `generation_config.json` declares two, E2B / E4B
    none): the target's continuation `tokens` is the argmax of its logits with them at −inf
    (`gemma4.export_unified.suppressed_logits`, the product goldens' rule). The drafts are the
    drafter's own unmasked argmax — what the graph emits and what upstream's candidate generator
    takes.

    The lender is prefilled with the prompt (`chunk` rows at a time), then `cycles` greedy
    continuations are run. Each cycle starts at the lender's logical length `P` (= the rows in the
    KV), and the drafter is passed these four:

    - `token` = `b` (**the last confirmed token** — at position `P` but not yet in the KV)
    - `hidden` = `h@(P−1)` (the post-final-norm hidden of the row that produced `b`)
    - one RoPE row = position **`P`** (the position of `b`)
    - the lender's l13 / l14 columns `[P − min(P, W), P)` / `[0, P)`

    This is the same pairing as the drafter's own recursion between steps (sending `token` and "the
    hidden of the row that produced it" to the next step), and also the set the speculative loop
    passes in its first cycle. So `draft[t][j]` (`j` from 0) predicts the token at position
    `P+1+j` = `tokens[t + 1 + j]`, and `d₁` predicts the token after the bonus token.

    MUST: take `cycles + k` `tokens` (the last cycle's drafts also need something to compare with).

    MUST: the tokens fed to the lender are **the greedy continuation, not the drafter's drafts**
    (teacher forcing) — running the accept / reject loop would make the goldens depend on that
    implementation, and they would have to be retaken every time the step-3 speculative loop
    changes.
    """
    window = int(drafter.drafter.config.get_text_config().sliding_window)
    lender = _new_lender(wrapper, specs)
    for start in range(0, int(ids.shape[1]), chunk):
        hidden = lender.step(ids[:, start : start + chunk])
    tokens: list[int] = []
    drafts: list[list[int]] = []
    for cycle in range(cycles + drafter.steps):
        past = lender.length
        # The frontier token at position P (the one the lender chose from h@(P−1) — not in the KV
        # yet).
        nxt = int(unified.suppressed_logits(lender.logits(hidden), suppress).argmax(-1))
        tokens.append(nxt)
        if cycle < cycles:
            sliding = lender.kv(one_shot.SLIDING_ATTENTION, min(past, window))
            full = lender.kv(one_shot.FULL_ATTENTION, past)
            rope = decode.rope_args(specs, torch.tensor([[past]]))
            with torch.no_grad():
                drafted = drafter(
                    torch.tensor([[nxt]], dtype=torch.int64, device=device),
                    hidden,
                    *(table.to(device) for table in rope),
                    *sliding,
                    *full,
                )
            drafts.append([int(value) for value in drafted])
        hidden = lender.step(torch.tensor([[nxt]], dtype=torch.int64))
    return {
        PROMPT_KEY: ids[0].to(torch.int32).contiguous(),
        TOKENS_KEY: torch.tensor(tokens, dtype=torch.int32).contiguous(),
        DRAFT_KEY: torch.tensor(drafts, dtype=torch.int32).contiguous(),
    }


def acceptance_stats(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """The acceptance outlook the goldens show (`per_step` = per-position match rate,
    `tokens_per_cycle` = sequential acceptance).

    `draft[t][j]` predicts `tokens[t + 1 + j]` ({@link draft_case}). The per-position match rate is
    the hit rate of step `j` taken on its own, and the sequential acceptance is taken from **the
    length `a` of the run of matches from the start** — the speculative loop confirms `1 + a`
    tokens per cycle (a rejected row also carries one bonus token), so
    `tokens_per_cycle` = `1 + mean(a)` is exactly "the tokens confirmed per cycle".
    """
    tokens, draft = tensors[TOKENS_KEY], tensors[DRAFT_KEY]
    cycles, steps = int(draft.shape[0]), int(draft.shape[1])
    if int(tokens.shape[0]) != cycles + steps:
        raise AssertionError(f"{int(tokens.shape[0])} tokens — not cycles {cycles} + steps {steps}")
    per_step = [0] * steps
    prefixes = 0
    for cycle in range(cycles):
        running = True
        for step in range(steps):
            hit = int(draft[cycle][step]) == int(tokens[cycle + 1 + step])
            per_step[step] += int(hit)
            running = running and hit
            prefixes += int(running)
    return {
        "per_step": [hits / cycles for hits in per_step],
        "tokens_per_cycle": 1 + prefixes / cycles,
    }


def write_goldens(
    drafter: DrafterWrapper,
    model_dir: Path,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
    cycles: int,
    device: str,
) -> dict[str, Any]:
    """Take and write every case's golden and return the summary (the lender is loaded only here).

    MUST: the lender holds **the product series' values** (`export_product.load_wrapper` alone —
    the series stores the checkpoint's bf16 values, nothing rounded). Expectations taken against
    other values compare two things wrong in the same direction.
    """
    print("[golden] the target (the product series' bf16 values)", file=sys.stderr, flush=True)
    wrapper = product.load_wrapper(model_dir)
    return goldens_against(drafter, wrapper, cases, out_dir, cycles, device)


def goldens_against(
    drafter: DrafterWrapper,
    wrapper: Any,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
    cycles: int,
    device: str,
    *,
    suppress: Sequence[int] = (),
) -> dict[str, Any]:
    """Take and write every case's golden against `wrapper` — the target reference, a product
    wrapper (`gemma4.export_product.ProductChunkWrapper`, or 12B's
    `gemma4.export_unified.UnifiedProductWrapper`) already in the storage the lender series
    holds — and return the summary ({@link write_goldens} loads the regular product's; the QAT
    line hands its own, `gemma4.export_drafter_qat`). `suppress` is {@link draft_case}'s.
    """
    # MUST: only the lender goes back to upstream's default SDPA. `karume_gqa`
    # ({@link gemma4.export.gqa_sdpa_attention}) requires a mask because "no mask silently turns
    # non-causal", but on the path with a KV cache upstream may make the mask **None**, deciding to
    # "leave causal to is_causal" (`create_causal_mask`).
    # Building the mask here would also mean re-implementing a kv_offset matching the sliding
    # layers' cache truncation (`DynamicSlidingWindowLayer` keeps only the previous `window − 1`
    # rows).
    # The numbers come from the same SDPA; the only difference is whether repeat_kv is
    # materialized (the borrower side stays on `karume_gqa` as in the export — that is what the
    # goldens are about).
    wrapper.model.config._attn_implementation = "sdpa"
    specs = decode.rope_specs(wrapper.model.config)
    # The 35 PLE tables (9.4GB) MUST stay on the CPU — on the GPU they do not fit a 12GB-class card
    # even with the tower removed. Only `wrapper.model` is moved, so the sibling attribute
    # `per_layer` stays.
    wrapper.model.to(device)
    drafter.to(device)

    summary: dict[str, Any] = {}
    for name, ids in cases:
        print(f"[golden] {name}: prompt {int(ids.shape[1])} token", file=sys.stderr, flush=True)
        tensors = draft_case(drafter, wrapper, specs, ids, cycles, device=device, suppress=suppress)
        path = out_dir / f"{GOLDEN_PREFIX}{name}{GOLDEN_SUFFIX}"
        save_file(tensors, str(path))
        stats = acceptance_stats(tensors)
        summary[name] = {
            "file": path.name,
            "prompt": int(tensors[PROMPT_KEY].shape[0]),
            "cycles": int(tensors[DRAFT_KEY].shape[0]),
            "steps": int(tensors[DRAFT_KEY].shape[1]),
            **stats,
        }
        print(
            f"[golden] {name}: per-position match rate "
            + " / ".join(
                f"d{step + 1} {rate * 100:.1f}%" for step, rate in enumerate(stats["per_step"])
            )
            + f" / sequential acceptance {stats['tokens_per_cycle']:.3f} token/cycle",
            file=sys.stderr,
            flush=True,
        )
    drafter.to("cpu")
    return summary


# ---- Series ----------------------------------------------------------------


def drafter_reference(out_dir: Path, model_dir: Path, goldens: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance record (the fingerprints of **both** the target and the drafter + the digests of
    the goldens taken).

    The assets come from two checkpoints, the lender's and the borrower's, so both are named, so
    that a combination with only one of them replaced cannot be built (an extension of the
    single-series {@link gemma4.provenance}).
    """
    assistant = _assistant_snapshot_dir()
    return {
        "schema": provenance.SCHEMA,
        "series": out_dir.name,
        "checkpoint": provenance.checkpoint_fingerprint(model_dir),
        "drafter": {
            "repo": ASSISTANT_REPO,
            "dir": assistant.name,
            "files": {
                name: provenance.file_digest(assistant / name)
                for name in ("model.safetensors", "config.json")
            },
        },
        "goldens": dict(goldens),
    }


def _assistant_snapshot_dir() -> Path:
    """The drafter snapshot in the HF cache (where the fingerprinted data lives)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(ASSISTANT_REPO, allow_patterns=["*.json", "*.safetensors"]))


def export_series(
    model_dir: Path,
    out_dir: Path,
    *,
    steps: int = DRAFT_STEPS,
    product_dir: Path = DEFAULT_PRODUCT_DIR,
    cycles: int = GOLDEN_CYCLES,
    goldens: bool = True,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Write the borrower container + goldens + provenance record, and return the summary.

    MUST: the outputs are written into the staging directory and placed only after passing **every
    gate** (the form check, the goldens) ({@link karume.artifacts.staged_publication}).
    MUST: the rounding comes before taking the reference and the goldens (ADR 0006) — after it, only
    the goldens would run on the original weights.
    """
    print(
        "[load] drafter (no sparse softmax) + the target's main table", file=sys.stderr, flush=True
    )
    wrapper = load_drafter(steps, model_dir)
    config = wrapper.drafter.config
    text = config.get_text_config()

    print(
        "[quant] i8 per-channel (every nn.Linear; the shared table unrounded)",
        file=sys.stderr,
        flush=True,
    )
    int8, scales = quantize_wrapper(wrapper)

    lender = lender_slots(
        lender_graph(product_dir),
        read_layouts(lender_container(product_dir)),
        config,
        one_shot.load_text_config(model_dir),
    )
    specs = decode.rope_specs(text)
    example = example_inputs(wrapper, specs)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with staged_publication(out_dir) as staged:
        staged.mkdir()
        print("[export] torch.export → conversion", file=sys.stderr, flush=True)
        graph, tensors = export_module_with_attention(wrapper, example)
        print("[export] surgery → shared declaration → write", file=sys.stderr, flush=True)
        surgical = to_external_states_form(graph, external_states_plan(graph, text, lender))
        shared_graph, remaining, shared_name = share_target_embedding(
            surgical, tensors, lender, one_shot.load_text_config(model_dir)
        )
        verified = publish_model(
            staged / one_shot.MODEL_FILE,
            shared_graph,
            {name: value for name, value in remaining.items() if name in _declared(shared_graph)},
            provenance=one_shot.PROVENANCE,
            # The graph name is **the component name** (= the weights key of karume.json). It is
            # not derived from the directory name — neither the series name (`gemma4-e2b-product`)
            # nor the staging directory's name matches the component name (container-v1 §2.1).
            graph_name=GEMMA4_DRAFTER_ROLE,
            weight_dtype="i8",
            weight_scales=scales,
        )
        form = assert_ir_form_drafter(verified, config, lender, steps)

        golden_summary: dict[str, Any] = {}
        if goldens:
            cases = build_cases(model_dir)
            golden_summary = write_goldens(wrapper, model_dir, cases, staged, cycles, device)
        print("[provenance] checkpoint fingerprint → reference.json", file=sys.stderr, flush=True)
        provenance.write_record(staged, drafter_reference(out_dir, model_dir, golden_summary))

    return {
        "dir": str(out_dir),
        "steps": steps,
        "nodes": len(verified.nodes),
        "initializers": len(verified.initializers),
        "shared_initializer": shared_name,
        "model_bytes": sum(
            path.stat().st_size for path in container_parts(out_dir / one_shot.MODEL_FILE)
        ),
        "parts": [path.name for path in container_parts(out_dir / one_shot.MODEL_FILE)],
        "ops": sorted(verified.required_ops),
        "symbols": list(verified.symbols),
        "quantized": {"i8": int8.describe()},
        "form": form,
        "goldens": golden_summary,
    }


def _declared(graph: IrGraph) -> set[str]:
    """Tensor keys of the initializers whose data this container writes (excluding shared
    declarations).

    The exclusion rule itself is owned by the core ({@link karume.emit.bakeable_initializers}) —
    this only projects "name → tensor key" and does not copy the rule.
    """
    return {graph.initializers[name].tensor for name in bakeable_initializers(graph)}


def export_module_with_attention(
    wrapper: nn.Module, example: tuple[torch.Tensor, ...]
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """Export with SDPA preserved (the form where the `attention` op remains — the opt-in of
    ADR 0023).

    Not a single symbol is declared (the K/V placeholders have fixed length, and the surgery adds
    the capacity symbol).
    """
    from karume.pipeline import export_module

    return export_module(
        wrapper,
        example,
        dynamic_shapes=None,
        symbol_names=(),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )


def example_inputs(wrapper: DrafterWrapper, specs: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
    """Example inputs of the trace (K / V are placeholders with a fixed {@link TRACE_KV_ROWS}
    rows).

    The K / V placeholders carry the drafter's KV heads per layer type
    (`gemma4.export_decode.kv_heads` — upstream's rule): E2B 1 / E4B 2 on both types; 12B 8 on the
    sliding layers and 1 on the full one (`attention_k_eq_v`). The lender's slots have the same
    head counts, which the surgery takes from the lender's declaration.
    """
    backbone = int(wrapper.drafter.config.backbone_hidden_size)
    text = wrapper.drafter.config.get_text_config()
    sliding_heads = decode.kv_heads(text, one_shot.SLIDING_ATTENTION)
    full_heads = decode.kv_heads(text, one_shot.FULL_ATTENTION)
    generator = torch.Generator().manual_seed(20260908)
    rope = decode.rope_args(specs, torch.tensor([[TRACE_KV_ROWS - 1]]))
    sliding = specs[one_shot.SLIDING_ATTENTION].head_dim
    full = specs[one_shot.FULL_ATTENTION].head_dim
    return (
        torch.zeros(1, 1, dtype=torch.int64),
        torch.randn(1, backbone, generator=generator),
        *rope,
        torch.randn(1, sliding_heads, TRACE_KV_ROWS, sliding, generator=generator),
        torch.randn(1, sliding_heads, TRACE_KV_ROWS, sliding, generator=generator),
        torch.randn(1, full_heads, TRACE_KV_ROWS, full, generator=generator),
        torch.randn(1, full_heads, TRACE_KV_ROWS, full, generator=generator),
    )


def build_parser() -> argparse.ArgumentParser:
    # MUST: `--model` accepts E2B only — both the drafter's upstream ({@link ASSISTANT_REPO}) and
    # the lender's default (`--product-dir`) are spelled for E2B only, so accepting another model
    # would silently bake E2B's drafter against another model's lender (E4B's drafter is not
    # supported).
    parser = one_shot.series_parser(
        __doc__.split("\n\n")[0],
        partial(one_shot.series_dir, suffix=GEMMA4_DRAFTER_SUFFIX),
        with_sym_max=False,
        models=(GEMMA4_DEFAULT_MODEL,),
    )
    parser.add_argument("--steps", type=int, default=DRAFT_STEPS)
    parser.add_argument("--product-dir", type=Path, default=DEFAULT_PRODUCT_DIR)
    parser.add_argument("--cycles", type=int, default=GOLDEN_CYCLES)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-goldens", dest="goldens", action="store_false")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    one_shot.run_series_cli(build_parser(), export_series, argv)


if __name__ == "__main__":
    main()
