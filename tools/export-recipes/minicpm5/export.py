"""Script that writes the real-weight MiniCPM5-1B (causal LM) in **one-shot form** to an IR v2
container + golden io.

The **scaffold for accepting**
ADR [0067](../../../docs/decisions/0067-autoregressive-attention-vocabulary.md) decision 1
(GQA accepted as the divisible broadcast of `attention`) **on a real model**. No KV cache — a
single prefill-like path that takes all of `input_ids[1,T]` and returns `logits[1,T,vocab]`; the
state slots (ADR 0066) and the decode path are left to later iterations.

    uv run --with 'transformers==5.14.1' python -m minicpm5.export

transformers is **pinned to 5.14.1** (for the same reason as `embeddinggemma/export.py` — a change
in the modeling code changes the graph's shape). It is not added to pyproject.toml / uv.lock but
added temporarily with `--with`.

## What goes into the graph

The whole forward of `LlamaForCausalLM` as one graph (24 layers + `lm_head`). Tokenizing and
applying the chat template happen on the host. The output is the raw logits `[1, T, 130560]` (no
softmax and no sampling in the graph).

MUST: **`head_dim` is an independent field** of the config (128 — not the same thing as
`hidden_size / num_attention_heads` = 1536 / 16 = 96). Do not write anything that derives head_dim
from hidden/heads. Whenever it is needed, always read `config.head_dim`.

## Emitting GQA in its "true form" (the acceptance point of decision 1)

MiniCPM5-1B is 16:2 GQA. In transformers 5.14.1, `sdpa_attention_forward` has `use_gqa_in_sdpa`
(integrations/sdpa_attention.py:38) **conditioned on `attention_mask is None`**, so with this
script's form, which passes an additive mask, it is always false, and `repeat_kv` **materializes**
k / v from `[1,2,T,128]` to `[1,16,T,128]` **before** calling SDPA (lines 99-100 of the same
file). Measured as well: with plain `sdpa`, all three attention ins have H=16 and the GQA is gone.

So {@link gqa_sdpa_attention} is registered through **transformers' public extension point**
(`AttentionInterface.register` — not one line of upstream code is replaced), and it calls SDPA
with `enable_gqa=True` without going through `repeat_kv`. The IR's attention becomes
`q[1,16,T,128] / k[1,2,T,128] / v[1,2,T,128]`.

MUST: {@link assert_ir_form} **always checks** the shape of the emitted IR. The `repeat_kv`
materialized form (Hkv=16) is an asset where "the numbers agree but the acceptance has lost its
meaning", so it is never written silently.

## causal is expressed by constant folding of an additive mask

The runtime's attention has no causal field, and the exporter rejects `is_causal=True`
(`_h_attention` in aten_handlers.py). This script **structurally has no path by which `is_causal`
reaches SDPA** — {@link gqa_sdpa_attention} always passes `is_causal=False` and fails loudly when
the mask is `None` (= the form that would silently turn non-causal).

Causality is expressed by the additive f32 `[1,1,T,T]` (0 / −inf) built by
{@link additive_causal_mask}. Its leaves are only `arange` and `full`, and T appears only as a
slice length, so the exporter's constant folding lowers it to **a Tmax×Tmax f32 initializer +
`sym_prefix_slice`** (ADR 0010 — the same mechanism as EmbeddingGemma's band mask). Because it is
passed as 4D, transformers' `create_causal_mask` returns it unchanged in
`_preprocess_mask_arguments` (masking_utils.py:818) and never enters the `is_causal` shortcut of
`sdpa_mask`. As a result **the only graph input is `input_ids`**.

## Output layout

    outputs/series/minicpm5-1b/model-NNNNN-of-NNNNN.krm weights / constants + the 2 documents
    outputs/series/minicpm5-1b/io.<case>.safetensors    inputs and the expected outputs on torch CPU

The tensor key convention of io is the same as tiny golden / DeBERTa / EmbeddingGemma
(`input.<graph input name>` / `output.<position>`). The logits have a 130560 vocabulary, so one
case is `T × 522KB` — the golden T is kept at 128 or less ({@link GOLDEN_CASES}).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from _shared.upstream import snapshot_revision
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.rope import assert_rope_lifted
from karume.shapes import declared_shape

#: Where the official weights live (the target of `hf download openbmb/MiniCPM5-1B`).
DEFAULT_MODEL_DIR = INPUTS_ROOT / "minicpm5" / "MiniCPM5-1B"

#: Default location of the artifacts. The storage dtype is f32 only (f16 / i8 / w4 are matters
#: for other series).
DEFAULT_OUT_DIR = SERIES_ROOT / "minicpm5-1b"

MODEL_FILE = "model.krm"

#: The container's graph name = **the part name of the distribution** (= the key of `weights` in
#: `karume.json` — container-v1 §2.1). This recipe builds no distribution (there is no
#: `distribution.py`), but the graph name is the spelling the runtime looks up with
#: `prepareContainer(opened, <weights key>)`, so it takes **the convention of a single-part
#: series** (`model`) so that the container in place can be used as is on the day one is built.
#: It does not match the series directory name (`minicpm5-1b`).
GRAPH_NAME = "model"

#: The license identifier baked into the container (container-v1 §2.3). This recipe builds no
#: distribution (neither a card nor `distribution.py`), so its source is the upstream model card's
#: declaration itself — the value `THIRD_PARTY_NOTICES.md` records as "not verified against the
#: revision used".
LICENSE = "apache-2.0"


def upstream_provenance(model_dir: Path) -> Provenance:
    """The provenance baked into the container. The revision is derived from the download record of
    `--model-dir`.

    Not hard-coded ({@link _shared.upstream.snapshot_revision}) — with a constant, the container
    would claim the same revision even when `--model-dir` points at another snapshot. The
    decode-form script (`minicpm5.export_decode`) and the series migration (`migrate_series`) use
    the same function.
    """
    return Provenance(license=LICENSE, upstream_revision=snapshot_revision(model_dir))


IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: Name of the graph input (the wrapper's forward argument name). The spelling
#: {@link assert_ir_form} uses to check "this one and nothing else" — if a mask or position_ids
#: shows up among the inputs, the folding is not working.
INPUT_IDS = "input_ids"

#: IR op names used by the checks.
ATTENTION_OP = "attention"
SYM_PREFIX_SLICE_OP = "sym_prefix_slice"

#: Upper bound of the symbolic dimension T. The causal constant is baked at Tmax² (512² × 4B =
#: 1MiB), so raise it only knowing that cost (ADR 0010 — it is the very evaluation point of the
#: folding). The model's `max_position_embeddings` is 131072, but that is a separate matter from
#: the length the one-shot acceptance needs.
SYM_MAX = 512

#: The blocking value of the additive mask. The same value `normalize._additive_attn_mask` lowers a
#: bool mask to, exactly the mask contract of fused attention (ADR 0023).
NEG_INF = float("-inf")

#: The name registered with `AttentionInterface`. **It MUST NOT contain `sdpa` /
#: `flash_attention` / `flex_attention` as a substring** — transformers'
#: `get_correct_attn_implementation` branches into backend-specific dispatch checks by partial
#: name matching (modeling_utils.py:2086-2091).
ATTENTION_NAME = "karume_gqa"

#: Body of the longer (T=87) English case. A paragraph whose end is a "strongly determined
#: continuation", to see that the prefix slice and the RoPE table act at their real length at a T
#: far enough from the short cases.
CONTEXT_EN = (
    "France is a country in Western Europe with a long written history. Its cities grew along "
    "the rivers that carried grain and stone, and the largest of them sits on the Seine, where "
    "an island in the middle of the river held the first settlement. That city has been the "
    "seat of government since the Middle Ages, and today it holds the parliament, the "
    "ministries, and the residence of the head of state. The capital of France is"
)

#: Body of the medium (T=61) Japanese case. It shows that the same form works with CJK byte-level
#: BPE, and it exercises a T different from the English case.
CONTEXT_JA = (
    "日本は東アジアの島国で、四つの大きな島と多くの小さな島から成る。国会と中央官庁、"
    "そして皇居が置かれているのは関東平野の都市で、江戸と呼ばれていた時代から政治の"
    "中心だった。日本の首都は"
)

#: The fixed texts of the golden (`(case name, body)`). Tokenizing reads the official
#: `tokenizer.json` directly with `tokenizers` (`<s>` is added by the post_processor —
#: `add_bos_token: false` in `tokenizer_config.json` means "the template adds it", and the encode
#: result starts with id 0). T is spread over 6 / 12 / 61 / 87 (far from the 0/1 specialization of
#: symbolic dimensions, and well inside the upper bound of 512). The logits of the 4 cases total
#: 86MB (one case = T × 522KB).
#:
#: capital-ja is written as a **pair** (with the lead-in 「フランスの首都はパリ、」) — with the bare
#: 「日本の首都は」 the greedy top-1 is the comma `、` (the frequent form 「〜は、東京」) and no
#: expectation holds, and the reversed order 「日本の首都は東京、フランスの首都は」 gives the
#: Chinese `巴黎` as top-1 (measured 2026-08-17 — the OpenBMB models lean Chinese). Only this
#: word order returns `東京` as top-1.
GOLDEN_CASES: tuple[tuple[str, str], ...] = (
    ("capital-en", "The capital of France is"),
    ("capital-ja", "フランスの首都はパリ、日本の首都は"),
    ("context-en", CONTEXT_EN),
    ("context-ja", CONTEXT_JA),
)

#: The expected continuation of each case's **greedy token at the last position** (a sanity check
#: that is not vacuous — ADR 0005's fail loudly). Each is a "strongly determined continuation", and
#: if any of ① mixed-up weights ② the wrong mask direction (looking at the future) ③ a position
#: offset in the RoPE table ④ a wrong GQA head mapping happens, the top-1 at the last position
#: changes.
#: MUST: each expected continuation MUST be a **single token** (checked by
#: {@link expected_token_ids}) — with several tokens, "top-1 agreement" has no defined meaning.
GREEDY_EXPECTATIONS: Mapping[str, str] = {
    "capital-en": " Paris",
    "capital-ja": "東京",
    "context-en": " Paris",
    "context-ja": "東京",
}


def additive_causal_mask(length: int) -> torch.Tensor:
    """Build the additive causal mask `[1,1,length,length]` (0 allowed / −inf blocked).

    MUST: write it so that the leaves are only `arange` and `full`, and `length` appears **only as
    the length of an arange** (the extent position of `convert.SYMBOL_EXTENT_ARGS`). If T enters
    the values, the Tmax folding stops commuting with the prefix and fails the two-point evaluation
    of `_check_prefix_commutes` (ADR 0010 addendum).
    MUST: write the comparison as `cols <= rows` (`le.Tensor`) — `rows >= cols` (`ge.Tensor`) is
    not in `convert.FOLDABLE_OPS`, so it cannot be folded and the whole export fails on an op that
    is not in the IR vocabulary.

    NOTE: during export, `length` is a `torch.SymInt` (it behaves as an int).
    """
    rows = torch.arange(length).unsqueeze(-1)
    cols = torch.arange(length).unsqueeze(0)
    keep = torch.full((), 0.0, dtype=torch.float32)
    drop = torch.full((), NEG_INF, dtype=torch.float32)
    return torch.where(cols <= rows, keep, drop).unsqueeze(0).unsqueeze(0)


def gqa_sdpa_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """An attention implementation that calls SDPA with `enable_gqa=True` without going through
    `repeat_kv` (preserving the GQA form).

    It follows the calling convention transformers' `AttentionInterface` requires as is
    (`(module, q, k, v, attention_mask, dropout=…, scaling=…, **kwargs)` →
    `(attn_output[B,M,H,D], attn_weights)`). Only three things are dropped from upstream's
    `sdpa_attention_forward`: ① the `repeat_kv` materialization ② the automatic `is_causal`
    decision ③ the NPU / paged branches.

    MUST: `attention_mask is None` fails loudly. Upstream uses that form as the signal to "leave it
    to `is_causal=True`", and passing it through here would silently emit **non-causal** attention
    (the IR has no causal field, so no later check would catch it).
    MUST: kwargs that change numerical semantics (`is_causal` / `position_bias`) are not accepted —
    ignoring them would hide, in both the export and the golden, that the numerical path differs
    from eager.
    """
    if attention_mask is None:
        raise ValueError(
            f"{type(module).__name__}: attention was called without attn_mask"
            " (causality is expressed by an additive mask — no mask silently turns non-causal)"
        )
    if float(dropout) != 0.0:
        raise ValueError(f"attention with dropout={dropout} is not supported (inference only)")
    unsupported = sorted(
        name for name in ("is_causal", "position_bias") if kwargs.get(name) is not None
    )
    if unsupported:
        raise ValueError(
            f"attention with kwargs {unsupported} is not supported (the numerical path changes)"
        )
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=scaling,
        is_causal=False,
        enable_gqa=True,
    )
    return output.transpose(1, 2).contiguous(), None


def register_attention() -> None:
    """Register {@link gqa_sdpa_attention} in transformers' attention registry (idempotent).

    `AttentionInterface.register` updates a class-attribute dict, so calling it any number of times
    under the same name leaves the last one (upstream's `sdpa` entry is not touched).
    """
    from transformers import AttentionInterface

    AttentionInterface.register(ATTENTION_NAME, gqa_sdpa_attention)


class CausalLmWrapper(nn.Module):
    """Export wrapper that fixes `LlamaForCausalLM` to the one-shot form (`input_ids → logits`).

    MUST: `attention_mask` is **always our own 4D additive mask**. Omitting it sends transformers
    down the path that leaves things to `is_causal`, which fails at the mask check of
    {@link gqa_sdpa_attention}.
    MUST: `use_cache=False`. With True a `DynamicCache` appears and the form returns KV, which
    breaks the one-shot contract (a single output).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = additive_causal_mask(input_ids.shape[1])
        return self.model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits


def load_wrapper(model_dir: Path) -> CausalLmWrapper:
    """Load the real weights in f32 and return an exportable wrapper with the RoPE buffers
    demoted."""
    from transformers import LlamaForCausalLM

    register_attention()
    model = LlamaForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation=ATTENTION_NAME
    )
    model.eval()
    # While inv_freq stays a buffer it is not a leaf of the constant folding, and sin / cos remain
    # in the IR.
    assert_rope_lifted(model, "minicpm5")
    return CausalLmWrapper(model).eval()


def load_tokenizer(model_dir: Path) -> Any:
    """Read the official `tokenizer.json` directly with `tokenizers` (not through transformers'
    tokenizer layer).

    Returns a `tokenizers.Tokenizer`. The type is `Any` because of the lazy import (importing at
    module top would make merely importing the script require tokenizers).
    """
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(model_dir / "tokenizer.json"))


def build_cases(model_dir: Path, sym_max: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """`(name, input_ids)` of the 4 golden cases. Fails loudly when a length is outside the range
    of the symbolic dimension."""
    tokenizer = load_tokenizer(model_dir)
    cases: list[tuple[str, torch.Tensor]] = []
    for name, text in GOLDEN_CASES:
        ids = torch.tensor([tokenizer.encode(text).ids], dtype=torch.int64)
        length = int(ids.shape[1])
        if not 2 <= length <= sym_max:
            raise ValueError(
                f"{name}: T={length} is outside the symbolic dimension's range [2, {sym_max}]"
            )
        cases.append((name, ids))
    return tuple(cases)


def assert_ir_form(graph: IrGraph, config: Any, sym_max: int) -> dict[str, Any]:
    """Check that the IR has "the true GQA form" and that "causal is a constant independent of T".

    Both are properties that **break quietly while the numbers still agree**, and comparing against
    the golden io does not catch them:

    - when `repeat_kv` materializes, the attention's Hkv turns into H (= the subject of decision 1's
      acceptance disappears)
    - when the mask remains a graph input, the form becomes "the host builds T² and passes it every
      time" (against the intent of ADR 0010, and a different asset on the runtime side too)

    MUST: read `config.head_dim` (do not derive it from `hidden_size // num_attention_heads` —
    MiniCPM5-1B disagrees, 96 ≠ 128).
    """
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    depth = int(config.head_dim)
    layers = int(config.num_hidden_layers)

    names = [spec.name for spec in graph.inputs]
    if names != [INPUT_IDS]:
        raise AssertionError(
            f"graph inputs are {names} — not `{INPUT_IDS}` alone"
            " (the mask may have stayed an input instead of being folded)"
        )
    if len(graph.outputs) != 1:
        raise AssertionError(
            f"the IR has {len(graph.outputs)} outputs (the one-shot form has one: the logits)"
        )

    producer = {out: node for node in graph.nodes for out in node.outs}
    attentions = [node for node in graph.nodes if node.op == ATTENTION_OP]
    if len(attentions) != layers:
        raise AssertionError(
            f"{len(attentions)} attention nodes (does not match the {layers} layers)"
        )

    mask_constants: set[str] = set()
    for index, node in enumerate(attentions):
        where = f"attention[{index}]"
        if len(node.ins) != 4:
            raise AssertionError(f"{where}: {len(node.ins)} ins (expected 4: q / k / v / mask)")
        query, key, value, _ = (declared_shape(graph, name) for name in node.ins)
        if query[1] != heads or key[1] != kv_heads or value[1] != kv_heads:
            raise AssertionError(
                f"{where}: the head axes are {[query[1], key[1], value[1]]} —"
                f" not the true GQA form {[heads, kv_heads, kv_heads]}"
                " (if k / v are widened to H, repeat_kv has materialized)"
            )
        if query[3] != depth or key[3] != depth or value[3] != depth:
            raise AssertionError(
                f"{where}: the D axes are {[query[3], key[3], value[3]]} —"
                f" not config.head_dim {depth}"
            )
        source = producer.get(node.ins[3])
        if source is None or source.op != SYM_PREFIX_SLICE_OP:
            found = (
                "not a node output (a graph input or an initializer directly)"
                if source is None
                else source.op
            )
            raise AssertionError(
                f"{where}: the mask comes from {found} — not the {SYM_PREFIX_SLICE_OP} of a Tmax"
                " constant"
            )
        constant = source.ins[0]
        if constant not in graph.initializers:
            raise AssertionError(f"{where}: the mask's source '{constant}' is not an initializer")
        shape = declared_shape(graph, constant)
        if shape != [1, 1, sym_max, sym_max]:
            raise AssertionError(
                f"{where}: the mask constant's shape {shape} is not the Tmax shape"
                f" [1, 1, {sym_max}, {sym_max}]"
            )
        mask_constants.add(constant)
    if len(mask_constants) != 1:
        raise AssertionError(
            f"{len(mask_constants)} mask constants (all layers should share one):"
            f" {sorted(mask_constants)}"
        )
    return {
        "attention_nodes": len(attentions),
        "heads": [heads, kv_heads, kv_heads],
        "head_dim": depth,
        "mask_constant": sorted(mask_constants)[0],
    }


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Write each case's inputs and torch CPU expected outputs to `io.<case>.safetensors`.

    The second return value is the logits for the sanity check (kept as `[1, T, vocab]` — the last
    position has to be picked from it).
    """
    written: list[str] = []
    logits: dict[str, torch.Tensor] = {}
    for name, ids in cases:
        with torch.no_grad():
            output = wrapper(ids)
        args = {INPUT_IDS: ids}
        # MUST: the io is also lowered to the concrete representation of the IR's semantic dtype
        # (i64 → i32). If it does not match what the runtime receives, the Deno-side E2E cannot
        # read the golden (ADR 0009's boundary normalization).
        tensors = {
            f"{INPUT_PREFIX}{declared.name}": normalize_boundary_tensor(
                args[declared.name], f"{name}: input '{declared.name}'"
            )
            for declared in graph.inputs
        }
        tensors[f"{OUTPUT_PREFIX}0"] = normalize_boundary_tensor(
            output.detach().contiguous(), f"{name}: output 0"
        )
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
        logits[name] = output.detach()
    return written, logits


def greedy_tokens(logits: Mapping[str, torch.Tensor]) -> dict[str, int]:
    """The top-1 token id at the **last position** of each case."""
    return {name: int(value[0, -1].argmax()) for name, value in logits.items()}


def expected_token_ids(tokenizer: Any) -> dict[str, int]:
    """The expected continuations of {@link GREEDY_EXPECTATIONS} → token ids (fails unless each is
    a single token).

    Looked up with `add_special_tokens=False` — it is about a continuation, so no `<s>` is added.
    """
    expected: dict[str, int] = {}
    for name, continuation in GREEDY_EXPECTATIONS.items():
        ids = tokenizer.encode(continuation, add_special_tokens=False).ids
        if len(ids) != 1:
            raise AssertionError(
                f"{name}: the expected continuation {continuation!r} is {len(ids)} tokens {ids}"
                ' (unless it is a single token, "top-1 agreement" has no defined meaning)'
            )
        expected[name] = int(ids[0])
    return expected


def _sanity(
    greedy: Mapping[str, int], expected: Mapping[str, int], labels: Mapping[int, str]
) -> dict[str, Any]:
    """Check that the top-1 at the last position equals the expected continuation, and that it is
    not constant across cases.

    MUST: a mismatch fails. Checks of norms or shapes alone cannot detect "the numbers move but it
    is broken as a language model" (mixed-up layers, the mask direction, a RoPE offset).
    MUST: fail when the top-1 of every case is the same. It overlaps the expectation check, but it
    stands on its own as the detection line for "constant output", which remains even if the
    expectation table is loosened.
    """
    if set(greedy) != set(expected):
        raise AssertionError(
            f"the case sets differ: greedy={sorted(greedy)} expected={sorted(expected)}"
        )

    def show(token: int) -> str:
        return f"{token}({labels.get(token, '?')})"

    wrong = {
        name: (show(token), show(expected[name]))
        for name, token in greedy.items()
        if token != expected[name]
    }
    if wrong:
        raise AssertionError(
            "the top-1 at the last position differs from the expected continuation"
            f" (measured, expected): {wrong}"
        )
    if len(set(greedy.values())) == 1:
        raise AssertionError(
            f"the top-1 at the last position is the same {show(next(iter(greedy.values())))}"
            " for every case — constant output"
        )
    return {name: show(token) for name, token in sorted(greedy.items())}


def export_series(model_dir: Path, out_dir: Path, *, sym_max: int = SYM_MAX) -> dict[str, Any]:
    """Write the IR container and the golden io, and return a summary.

    MUST: the artifacts are written to a staging directory and put in place only after passing
    **every check** (the shape check and the sanity check). Placing them in final before the checks
    would let a failed run leave "an asset that can pass the acceptance check" — the io golden is
    taken from the same broken wrapper, so the two agree with each other and the TS-side comparison
    **goes green** (the spelling of "when publishing is allowed" is
    {@link _shared.decode_series._publish}; the rules of the swap and of cleanup belong to the core
    primitive {@link karume.artifacts.staged_publication}).
    """
    # The provenance is read before the weights (a checkpoint without a download record fails
    # before the weights are loaded).
    provenance = upstream_provenance(model_dir)
    wrapper = load_wrapper(model_dir)
    cases = build_cases(model_dir, sym_max)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # The example input is the longest case (the closer T is to the upper bound, the further from
    # the 0/1 specialization). min=2 avoids the 0/1 specialization, and max is the very evaluation
    # point of the Tmax folding (ADR 0010 — not managed twice through a separate knob).
    _, example_ids = max(cases, key=lambda case: case[1].shape[1])
    seq = Dim("T", min=2, max=sym_max)
    with staged_publication(out_dir) as staged:
        # The writer creates the directory itself (the primitive does not create it — it only
        # passes a path).
        staged.mkdir()
        graph = export_to_file(
            wrapper,
            (example_ids,),
            staged / MODEL_FILE,
            provenance=provenance,
            # The graph name is the **part name** ({@link GRAPH_NAME}) — not derived from the
            # directory name.
            graph_name=GRAPH_NAME,
            dynamic_shapes=({1: seq},),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )
        form = assert_ir_form(graph, wrapper.model.config, sym_max)
        written, logits = _write_io(wrapper, graph, cases, staged)
        tokenizer = load_tokenizer(model_dir)
        greedy = greedy_tokens(logits)
        labels = {token: tokenizer.id_to_token(token) for token in set(greedy.values())}
        expected = expected_token_ids(tokenizer)
        labels.update({token: tokenizer.id_to_token(token) for token in set(expected.values())})
        # MUST: evaluated before publication (the only non-vacuous check of this series — when it
        # fails, the staging directory goes with it).
        sanity = _sanity(greedy, expected, labels)
    return {
        "dir": str(out_dir),
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": sum(p.stat().st_size for p in container_parts(out_dir / MODEL_FILE)),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "case_lengths": {name: int(ids.shape[1]) for name, ids in cases},
        "form": form,
        "sanity": sanity,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sym-max", type=int, default=SYM_MAX)
    args = parser.parse_args(argv)
    summary = export_series(args.model_dir, args.out, sym_max=args.sym_max)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
