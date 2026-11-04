"""Script that writes the real-weight DeBERTa-v2 (SBV2 text front) to an IR v2 container (`krm`)
+ golden io.

Where the tiny goldens of `karume.goldens` cover "the op contracts", this script covers
**numeric agreement with real weights and real token sequences**. The outputs go under
`outputs/series/` and are kept out of commits by the repository's top-level `.gitignore` (the
weights are about 1.3GB).

    uv run --with 'transformers==5.14.1' python -m deberta.export
    uv run --with 'transformers==5.14.1' python -m deberta.export --layers 2
    uv run --with 'transformers==5.14.1' python -m deberta.export --dtype i8 --act-quant
    uv run --with 'transformers==5.14.1' python -m deberta.export --dtype i4 --layers 22

transformers is **pinned to 5.14.1** (the 5 items of
`docs/research/2026-08-02-deberta-front-recon.md` §6 — when the modelling code changes, the graph
shape changes). It is not put into pyproject.toml / uv.lock; it is added temporarily with `--with`.

Output layout (enumerated by the TS side, `packages/e2e/tests/e2e_deberta_test.ts`):

    outputs/series/deberta/<variant>/model-NNNNN-of-NNNNN.krm weights, constants + 2 descriptors
    outputs/series/deberta/<variant>/io.<case>.safetensors    inputs + torch CPU expected outputs

The container is a part sequence whose stem is `model.krm` ({@link MODEL_FILE}) (container-v1 §8).

The tensor key convention of io is the same as for the tiny goldens (`input.<graph input name>` /
`output.<position>`). The only difference is that one model has several io files.

## Storage dtypes and the w8a8 mirror (ADR 0019 / 0025)

`--dtype i8` writes to a **separate series** (`outputs/series/deberta-i8/`) — sharing a directory
with the f32 series would silently put the existing E2E net (the f32 tolerance) on different
assets. `--dtype f16` is **not added** (it is decided together with moving the SBV2 series to f16
— not filed yet).

`--dtype i4` is also a **separate series** (`outputs/series/deberta-i4/`) for the same reason, and
its contents are **mixed** (`nn.Linear` / `nn.Embedding` = i4 group32, the rest = i8). In the SBV2
distribution it is the `text_encoder` component of the `i8+bert4` quant (`sbv2/distribution.py`).
The ops eligible for i4 are the weight slots of linear / embedding / conv1d (`groups == 1`) (ADR
0069 addenda 6 / 7), and this series **does not select** conv1d for i4 ({@link I4_MODULE_TYPES}),
so the series is mixed.

The encoder linears of the i4 series are **rounded with GPTQ calibration** (the default —
perf-ledger Q-6 / `deberta.calib`). The storage form does not change by a single byte (the grid
stays RTN i4 g32); only the rounded values and the contents of the scale ledger change. The
calibration inputs are the 48 sentences of {@link deberta.calib_texts.CALIB_TEXTS} passed through
**the same input-building path as the goldens** ({@link build_graph_inputs}). A calibration failure
fails loudly; there is no branch that silently drops to plain RTN ("shipped uncalibrated while
meaning to calibrate" cannot be read from the assets).

`--act-quant` is the torch mirror of w8a8 (`SessionOptions.linearCompute: "a8"`): it additionally
writes expected values with the inputs of the eligible `nn.Linear` modules dropped to per-token i8,
under the separate prefix **`io-i8a8.<case>.safetensors`**. **The plain golden io is taken without
hooks** MUST — writing `io.<case>` with the hooks attached contaminates the expected values of the
w8 (f32 compute) E2E with activation quantization, and only the detection power disappears while
everything stays green. The prefix is separate so that the enumeration on the Deno side
(startsWith `io.`) does not mistake the mirror for a plain case.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from _shared.paths import SERIES_ROOT
from karume.act_quant import attach_act_quant, detach_act_quant
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import normalize_boundary_tensor
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    channel_rows,
    fake_quant_int4,
    fake_quant_int8,
    iter_quant_targets,
)

from . import calib, patch
from .calib_texts import CALIB_TEXTS

#: The very BERT the SBV2 text front uses (recon §1).
MODEL_ID = "ku-nlp/deberta-v2-large-japanese-char-wwm"

#: The revision (HF commit SHA) {@link MODEL_ID} is fetched at. Fetching the weights and the
#: tokenizer is pinned to this one point, and the same value is baked into the container's
#: `provenance.upstreamRevision` ({@link PROVENANCE}).
#:
#: MUST: do not drop `revision=` — without it HF's main is pulled as of that moment, so on the day
#: upstream main moves, the `text_encoder` component of the SBV2 distribution is silently baked from
#: a different revision (pinning transformers does not pin the weights). The value is the only
#: snapshot in the HF cache fetched on 2026-08-02
#: (`models--ku-nlp--deberta-v2-large-japanese-char-wwm`), and the current distribution was baked
#: from it (`refs/main` has the same value — checked against the cache on 2026-09-25).
#:
#: NOTE: when `--model` is given something other than HF (a local directory), transformers ignores
#: `revision`. A container baked that way still claims this revision, so the series used for the
#: distribution is baked with the default `--model`.
MODEL_REVISION = "547b0e8b044fba3f9b84d0ab9f990440bd130c8b"

#: The supported storage dtypes. **There is no f16** — it is decided together with moving the SBV2
#: series to f16 (not filed yet). `i4` is the name of a **mixed** series; what it holds is "the
#: eligible weights of {@link I4_MODULE_TYPES} = i4 group32, everything else (conv, weights the
#: group size does not divide) = i8 per-channel as before" ({@link BASE_WEIGHT_DTYPES} /
#: {@link _fake_quant}). conv1d has an i4 execution path since ADR 0069 addendum 7, but this series
#: does not select it for i4, so the series is mixed.
WEIGHT_DTYPES: tuple[str, ...] = ("f32", "i8", "i4")

#: The **module types** rounded to i4 group32 (a subset of `I4_WEIGHT_OPS` in `karume.emit`).
#: conv1d (`groups == 1`) is also i4-eligible since ADR 0069 addendum 7, but it is **not selected**
#: here (adding it means re-recording the SBV2 WAV reference values — a separate ruling).
I4_MODULE_TYPES: tuple[type[nn.Module], ...] = (nn.Linear, nn.Embedding)

#: FQNs of weights that sit in an `nn.Embedding` container but are **never looked up**. The
#: relative-position embedding table is used by the forward `deberta.patch` replaces as "slice the
#: table and pass it through query_proj / key_proj", so no `embedding` op appears in the graph — no
#: weight slot consumes it, so it is not eligible for compressed storage and stays f32 even in the
#: i8 series. Assigning it to i4 by type alone makes emit fail loudly with "not eligible", so it is
#: excluded from the i4 target set by name (the i8 side does the rounding as before).
#:
#: NOTE: if an upstream module rename makes the spelling miss, it does not go silent — the moment
#: the exclusion disappears, emit's check of explicit assignments fails with "not eligible" and
#: names that tensor.
NON_LOOKUP_EMBEDDINGS: frozenset[str] = frozenset({"model.encoder.rel_embeddings"})

#: The i4-eligible weights **outside** the encoder stages (= the one vocabulary table). Calibrated
#: rounding is driven stage by stage, so it can round only inside the stages; eligible weights
#: outside are handled by plain RTN i4 ({@link _round_i4_calibrated}).
#:
#: They are declared by name so that no "silent reclassification" happens — measured (on the
#: all-layer wrapper built from the config), this is the only i4-eligible weight outside the stages
#: (`embed_proj` is `None` because `embedding_size == hidden_size`, `position_embeddings` /
#: `token_type_embeddings` are `None` by config too, and `rel_embeddings` is ineligible via
#: {@link NON_LOOKUP_EMBEDDINGS}). If the upstream structure changes and the eligible set grows or
#: shrinks, the check of {@link _round_i4_calibrated} fails and names the FQN.
NON_STAGE_I4_WEIGHTS: frozenset[str] = frozenset({"model.embeddings.word_embeddings"})

#: Series name → the **default** storage dtype passed to `export_to_file`. Only the i4 series
#: defaults to i8, and the eligible linear / embedding weights are assigned to i4 one by one with
#: `weight_dtype_overrides` (the same form as gemma4's mixed i8+i4).
BASE_WEIGHT_DTYPES: Mapping[str, str] = {"f32": "f32", "i8": "i8", "i4": "i8"}

#: The default location of the outputs (one **series** per storage dtype). The parent is
#: `SERIES_ROOT` (= outputs/series/) — models/ holds distributions only (_shared.paths). Kept out of
#: commits by `outputs/` in `.gitignore`.
#: MUST: one directory per dtype (ADR 0019) — sharing one puts the f32 series' net on i8 assets.
DEFAULT_OUT_ROOTS: Mapping[str, Path] = {
    "f32": SERIES_ROOT / "deberta",
    "i8": SERIES_ROOT / "deberta-i8",
    "i4": SERIES_ROOT / "deberta-i4",
}

#: The graph inputs of one case (name → tensor).
InputArgs = Mapping[str, torch.Tensor]

#: The order of the graph inputs = the argument order of `HiddenStatesWrapper.forward`.
#: torch.export lays out `graph.inputs` in this order, so it is compared after export and **export
#: stops on a mismatch** (since the inputs are passed by position, a silent shift would swap only
#: the golden's inputs).
INPUT_ORDER: tuple[str, ...] = ("input_ids", "attention_mask", "c2p_pos", "p2c_pos")

MODEL_FILE = "model.krm"

#: The container's graph name = **the component name in the distribution** (= the key of weights in
#: `karume.json` — container-v1 §2.1). This script only bakes SBV2's `text_encoder` component, so
#: the source of truth is on the consuming side (`sbv2.distribution.SBV2_TEXT_ENCODER_COMPONENT`).
#:
#: MUST: the spelling matches the reader. A copy is kept for the same reason as {@link LICENSE} —
#: the producing side never imports the consuming side. The cross-family check
#: `test_deberta_names_the_seat_its_consumer_declares` in
#: `tests/test_graph_names.py::TestTheNamesAreTheWeightsKeys` compares them on every run
#: (`deberta/tests/test_export.py` does not look at this constant). It does not match the series
#: directory names (`full-24layer` / `dev-2layer`).
GRAPH_NAME = "text_encoder"

#: The provenance baked into the container (container-v1 §2.3). This recipe only bakes SBV2's
#: `text_encoder` component and has no card of its own, so the source of truth of the license
#: identifier is the consuming side's card (`sbv2.card.SBV2_TEXT_ENCODER_LICENSE`, checked at the
#: source on 2026-08-07).
#:
#: MUST: the spelling matches the reader. A copy is kept for the same reason as
#: {@link EXPORT_PROVENANCE_FILE} — the producing side never imports the consuming side.
#: `deberta/tests/test_export.py` compares them on every run.
LICENSE = "cc-by-sa-4.0"

PROVENANCE = Provenance(license=LICENSE, upstream_revision=MODEL_REVISION)
IO_PREFIX = "io."
#: The prefix of the w8a8 mirror io. It **does not start with `io.`** MUST — the Deno side
#: enumerates plain cases with `startsWith("io.")`, so an `io.` prefix would pick the mirror up as
#: a w8 golden.
ACT_IO_PREFIX = "io-i8a8."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: The upper bound of the symbolic dimension T. It matches the config's `max_position_embeddings`
#: (the `Dim` bound = the bound `export_provenance.json` carries — there is no baked constant, so
#: only the record carries it).
SYM_MAX = 512

#: The provenance record that writes the symbolic dimension's bound into the series
#: ({@link _write_export_provenance}).
#:
#: MUST: the spelling matches the readers (`sbv2.distribution.EXPORT_PROVENANCE_FILE` /
#: `sbv2.distribution.assert_sym_provenance`). A copy is kept because `deberta` is **not a
#: consumer** of SBV2 — the distribution's text_encoder component is a shared asset, not a coupling
#: (the module doc of `sbv2/distribution.py`), so the producing side never imports the consuming
#: side. So that the two tables cannot move independently, `deberta/tests/test_export.py` compares
#: them on every run (the same discipline as `birefnet`'s normalisation constants).
EXPORT_PROVENANCE_FILE = "export_provenance.json"


@dataclass(frozen=True)
class Variant:
    """The spelling and output form of one output directory.

    The output form is written here next to the layer count because **mixing up the distribution
    form (one output) and the verification form (every layer) goes unnoticed** — both output
    tensors of the same shape, so a mistake still loads and runs.
    """

    name: str
    #: Output only the one tensor SBV2 uses (the distribution form). False lists every layer's
    #: hidden_states.
    single_output: bool = False


#: Layer count → variant. 2 layers is for development iterations, **22 layers is the SBV2
#: distribution form**, and 24 layers is for verifying the goldens of every layer (reading how the
#: error grows per layer — the tolerance derivation of ADR 0026 decision 2).
#:
#: It is 22 because SBV2 uses `hidden_states[-3]` (= the 22nd from the start = the output of layer
#: 21), so the last 2 layers are completely dead in the distribution. The final output of the
#: truncated 22-layer model was measured to be **bit-identical** to `hidden_states[-3]` of the
#: 24-layer model (in the 3 configurations f32 / i8 / i8+a8) —
#: ADR 0045 / docs/research/2026-08-11-deberta-size-recon.md §4.
VARIANTS: dict[int, Variant] = {
    2: Variant("dev-2layer"),
    22: Variant("sbv2-22layer", single_output=True),
    24: Variant("full-24layer"),
}

#: The fixed sentences of the goldens (the tokenizer is the character-level BertJapaneseTokenizer).
#: Three sentences — short / longer / with symbols — spread T.
GOLDEN_SENTENCES: tuple[str, ...] = (
    "こんにちは、世界。",
    "音声合成のための特徴量を、この文から抽出します。",
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
)

#: The number of [PAD] tokens added in the padded case. Mixing in `attention_mask=0` exercises the
#: mask path (mul → cast → bitwise_not → masked_fill, and the zero fill of the conv path).
PAD_COUNT = 5


class HiddenStatesWrapper(nn.Module):
    """Export wrapper that returns a tuple of hidden_states instead of a `ModelOutput` (dict).

    IR v2 graph outputs are addressed by position (`output.<i>`), so every layer's hidden_states
    is laid out as an output as is. Each layer can be compared, so **how the error grows with the
    layer count** can be read directly from the goldens (this turns the justification of the
    tolerance into a measurement).

    With `single_output` set, **only the final layer's tensor** is output (the distribution form).
    The runtime reads back all of `graph.outputs`, so with 25 outputs even SBV2, which uses only
    one, would pay 25 tensors' worth of staging + mapAsync on every run (ADR 0026's "readback of
    ~52MB dominates"). In a truncated variant the final layer = the layer SBV2 uses, so narrowing
    loses no information.

    MUST: do not use `DebertaV2Model.forward`; call embeddings + encoder directly — the former has
    no way to pass `relative_pos` to the encoder, so the relative-position index tables **cannot
    be supplied externally** (`deberta.patch` — baking them adds two 2MiB constants at Tmax=512).
    The z_steps branch is permanently dead through the hard-coded `self.z_steps = 0`, so it is not
    copied.
    """

    def __init__(self, model: nn.Module, *, single_output: bool = False) -> None:
        super().__init__()
        self.model = model
        self.single_output = single_output

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        c2p_pos: torch.Tensor,
        p2c_pos: torch.Tensor,
    ) -> tuple[Any, ...]:
        model = self.model
        embedding_output = model.embeddings(
            input_ids=input_ids,
            token_type_ids=torch.zeros_like(input_ids),
            position_ids=None,
            mask=attention_mask,
            inputs_embeds=None,
        )
        hidden_states = model.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            relative_pos=(c2p_pos, p2c_pos),
            return_dict=True,
        ).hidden_states
        if self.single_output:
            return (hidden_states[-1],)
        return hidden_states


def load_model(model_id: str, num_layers: int) -> nn.Module:
    """Fetch the real weights, truncate the layer count and switch to eval.

    MUST: `attn_implementation="eager"` — the SDPA path gives a different graph shape under
    torch.export and no longer matches the recon's measurement (the classification of 27
    unsupported ops).
    """
    from transformers import DebertaV2Model

    model = DebertaV2Model.from_pretrained(
        model_id, revision=MODEL_REVISION, dtype=torch.float32, attn_implementation="eager"
    )
    if num_layers > model.config.num_hidden_layers:
        raise ValueError(
            f"--layers {num_layers} exceeds the model's layer count"
            f" {model.config.num_hidden_layers}"
        )
    model.encoder.layer = model.encoder.layer[:num_layers]
    model.config.num_hidden_layers = num_layers
    model.eval()
    return model


def encode_text(tokenizer: Any, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn one sentence into `[1, T]` `(input_ids, attention_mask)` (the mask is all ones)."""
    ids = torch.tensor([tokenizer(text)["input_ids"]], dtype=torch.int64)
    return ids, torch.ones_like(ids)


def build_graph_inputs(model: nn.Module, ids: torch.Tensor, mask: torch.Tensor) -> InputArgs:
    """Add the relative-position index tables to `(input_ids, attention_mask)` to make the set of 4
    inputs.

    The relative-position index tables are graph inputs, so they are built per case at the actual
    length (`deberta.patch` is the source of truth). The bucket size and the maximum position are
    **read from the module** — the config's `max_relative_positions` has a rule that falls back to
    `max_position_embeddings` when it is -1, so copying it would mean maintaining it twice.

    The golden cases ({@link build_cases}) and the calibration inputs ({@link build_calib_args})
    **go through this same function** MUST — built with a different spelling, the activations seen
    in calibration and those the goldens run would silently differ.
    """
    attention = model.encoder.layer[0].attention.self
    c2p_pos, p2c_pos = patch.build_rel_pos_tables(
        int(ids.shape[1]),
        position_buckets=attention.position_buckets,
        max_position=attention.max_relative_positions,
    )
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "c2p_pos": c2p_pos,
        "p2c_pos": p2c_pos,
    }


def build_cases(tokenizer: Any, model: nn.Module) -> tuple[tuple[str, InputArgs], ...]:
    """`(name, graph input name → tensor)` of the 4 golden cases (3 sentences + padded)."""
    ids_cases: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        (f"case{index}", *encode_text(tokenizer, text))
        for index, text in enumerate(GOLDEN_SENTENCES)
    ]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("the tokenizer has no pad_token_id (the padded case cannot be built)")
    base_ids = ids_cases[0][1]
    pad_ids = torch.full((1, PAD_COUNT), pad_id, dtype=torch.int64)
    ids_cases.append(
        (
            "padded",
            torch.cat([base_ids, pad_ids], dim=1),
            torch.cat([torch.ones_like(base_ids), torch.zeros_like(pad_ids)], dim=1),
        )
    )
    return tuple((name, build_graph_inputs(model, ids, mask)) for name, ids, mask in ids_cases)


def build_calib_args(
    tokenizer: Any, model: nn.Module, texts: Sequence[str]
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Turn the calibration corpus into a list of the wrapper's **positional arguments** (in the
    order of {@link INPUT_ORDER}).

    Tokenization and the index tables go through the same path as the goldens
    ({@link encode_text} / {@link build_graph_inputs}). Positional arguments let the calibration
    rig (`deberta.calib`) **not know the names** of the graph inputs (the rig only calls the
    wrapper).
    """
    if not texts:
        raise ValueError(
            "the calibration corpus is empty (calibrated i4 cannot be built from zero inputs)"
        )
    built: list[tuple[torch.Tensor, ...]] = []
    for text in texts:
        args = build_graph_inputs(model, *encode_text(tokenizer, text))
        built.append(tuple(args[key] for key in INPUT_ORDER))
    return tuple(built)


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, InputArgs]],
    out_dir: Path,
    *,
    prefix: str = IO_PREFIX,
) -> list[str]:
    """Write each case's inputs and the torch CPU expected outputs to
    `<prefix><case>.safetensors`."""
    written: list[str] = []
    for name, args in cases:
        # Passed by position, but the order comes from `graph.inputs` (export already compared it
        # with INPUT_ORDER).
        with torch.no_grad():
            outputs = wrapper(*(args[declared.name] for declared in graph.inputs))
        if len(outputs) != len(graph.outputs):
            raise AssertionError(
                f"{name}: {len(outputs)} torch outputs ≠ {len(graph.outputs)} IR outputs"
            )
        # MUST: io is also lowered to the concrete representation of the IR's semantic dtype
        # (i64 → I32). Unless it matches the form the runtime receives, the Deno-side E2E cannot
        # read the goldens (the boundary normalization of ADR 0009).
        tensors = {
            f"{INPUT_PREFIX}{declared.name}": normalize_boundary_tensor(
                args[declared.name], f"{name} input '{declared.name}'"
            )
            for declared in graph.inputs
        }
        for index, tensor in enumerate(outputs):
            tensors[f"{OUTPUT_PREFIX}{index}"] = normalize_boundary_tensor(
                tensor.detach().contiguous(), f"{name} output {index}"
            )
        path = out_dir / f"{prefix}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written


def _write_mirror_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, InputArgs]],
    out_dir: Path,
) -> tuple[list[str], int]:
    """Write the expected values with the activation-i8 hooks attached (the w8a8 mirror) to
    `io-i8a8.<case>`.

    MUST: the hooked span closes inside this function (the hooks are always removed in `finally`)
    — taking the plain `io.<case>` with them attached contaminates the expected values of the w8
    E2E. Callers come here only **after the plain io** has been written.

    The returned count is the basis of the expected constant of the diagnostic-key check on the
    Deno side (the dispatch count of `linearI8a8Key`). Zero fails loudly — "took w8 numbers while
    meaning w8a8" must not go unnoticed.
    """
    handles, attached = attach_act_quant(wrapper)
    print(f"[act-quant] applied per-token i8 to {attached} eligible linears (w8a8 mirror)")
    if attached == 0:
        raise SystemExit(
            "--act-quant was given but there are 0 eligible linears (the eligibility rule is"
            " broken)"
        )
    try:
        return _write_io(wrapper, graph, cases, out_dir, prefix=ACT_IO_PREFIX), attached
    finally:
        detach_act_quant(handles)


def _i4_module_names(wrapper: nn.Module) -> frozenset[str]:
    """The FQN set of the modules rounded to i4 group32 (the single source of the **exclusive
    split** of mixed i8+i4).

    Eligibility is the conjunction of 3 conditions: ① it is one of {@link I4_MODULE_TYPES} (emit's
    i4 eligibility = limited to the weight slots of linear / embedding — ADR 0069 decision 5;
    explicitly assigning i4 to any other tensor makes emit fail loudly) ② the quantization axis is
    divisible by the group size (i4 never creates a partial group MUST — the same ADR's decision 2;
    such a weight is **excluded from the targets rather than failing the whole configuration**, so
    it falls to the i8 side) ③ it is not in {@link NON_LOOKUP_EMBEDDINGS} (weights in an
    `nn.Embedding` container that are never looked up).

    Target enumeration goes through the core (`iter_quant_targets`) so that the set the rounding
    sees and the set counted here stay one implementation. The i8-side predicate is written as
    "not in this set" for the same reason: building the two predicates from separate spellings
    would leave weights in neither **silently in f32** (the hole on the other side of the ban on
    double rounding — the values stay correct, only the size comes back).
    """
    return frozenset(
        fqn.removesuffix(".weight")
        for fqn, weight, axis in iter_quant_targets(wrapper, I4_MODULE_TYPES)
        if channel_rows(weight, axis).shape[-1] % DEFAULT_GROUP_SIZE == 0
        and fqn.removesuffix(".weight") not in NON_LOOKUP_EMBEDDINGS
    )


def _round_i4_plain(
    wrapper: nn.Module, names: frozenset[str], label: str
) -> Mapping[str, torch.Tensor]:
    """Round the named set with plain RTN i4 g32 (the first-level grid itself, without
    calibration).

    `label` names what was rounded in the diagnostic line (it fills "rounded <label> to i4
    groups").
    """
    report = fake_quant_int4(
        wrapper,
        include=lambda name: name in names,
        op_types=I4_MODULE_TYPES,
    )
    print(f"[fake-quant] rounded {label} to i4 groups (RTN) — {report.describe()}", flush=True)
    return report.scales


def _round_i4_calibrated(
    wrapper: nn.Module,
    i4_names: frozenset[str],
    calib_args: Sequence[Sequence[torch.Tensor]],
) -> Mapping[str, torch.Tensor]:
    """Split the i4-eligible weights exclusively into "outside the stages = RTN" and "inside the
    encoder stages = GPTQ", and round them.

    Order MUST (① → ② → ③):

    1. **Round the eligible weights outside the stages (the vocabulary table) with RTN i4 first**
       — at distribution run time, what enters the encoder are activations looked up from the i4
       vocabulary table, so the calibration inputs are matched to the distribution's conditions
       (the encoder's rounding targets are picked with the vocabulary table's rounding error
       already propagated). Doing it later would ship weights calibrated on "activations looked
       up from the f32 table" paired with the i4 table.
    2. Capture of the calibration batches and the stage-split equivalence check
       (`deberta.calib.build_rig`).
    3. Round the eligible linears inside the stages with the GPTQ × RTN grid.

    MUST: the eligible weights outside the stages equal {@link NON_STAGE_I4_WEIGHTS} — otherwise
    the upstream structure has changed and the eligible set has grown or shrunk. Silently routing
    them to the RTN side (= thinner calibration) and silently dropping them to i8 (= only the size
    comes back) are both unreadable from the numbers, so it fails on the spot.
    """
    stages = calib.encoder_stages(wrapper)
    stage_names = calib.stage_linear_names(stages) & i4_names
    plain_names = i4_names - stage_names
    if plain_names != NON_STAGE_I4_WEIGHTS:
        raise AssertionError(
            f"the i4-eligible weights outside the encoder stages are {sorted(plain_names)}, which"
            f" differs from the declaration (NON_STAGE_I4_WEIGHTS = {sorted(NON_STAGE_I4_WEIGHTS)})"
            " — the upstream structure changed and moved the i4 split"
        )
    plain = _round_i4_plain(wrapper, plain_names, "the vocabulary table outside the stages")
    rig = calib.build_rig(wrapper, stages, calib_args)
    report, ledger = calib.calibrate_i4(
        rig, include=lambda local: f"{calib.STAGE_PREFIX}.{local}" in stage_names
    )
    calibrated = ledger.scales
    # MUST: the 2 paths are disjoint (an overlap would round the same weight twice, and only the
    # values would quietly go wrong).
    overlap = sorted(set(plain) & set(calibrated))
    if overlap:
        raise AssertionError(
            f"the two i4 paths round the same weights (double rounding): {overlap[:3]}"
        )
    print(
        "[fake-quant] rounded the encoder-stage linears with GPTQ calibration"
        f" — {report.describe()} / {len(rig.batches)} calibration inputs, {rig.tokens:,} tokens",
        flush=True,
    )
    return {**plain, **calibrated}


def _fake_quant(
    dtype: str,
    wrapper: nn.Module,
    *,
    calib_args: Sequence[Sequence[torch.Tensor]] | None = None,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
    """Round the weights to the representable values of the storage dtype, and return the scale
    ledger and the per-tensor storage assignments (ADR 0006).

    MUST: apply this to **the very `nn.Module` being exported** (= `HiddenStatesWrapper`). The
    scale ledger's keys are the FQNs seen here, and unless they equal the safetensors tensor keys
    (= the FQNs `torch.export` sees), the matching on the emit side finds nothing (the FQN
    discipline of `quantize.py` — matching by `id()` is forbidden).

    MUST: call it **before the golden io is taken** (the docstring of `quantize.py`) — applying it
    later computes only the expected values with the original weights, quantization error mixes
    into the E2E difference, and the tolerance loses its meaning. Calibrated rounding (GPTQ) is
    kept inside this function for the same reason.

    The i4 series is mixed (eligible linear / embedding = i4 group32, the rest = i8 per-channel),
    and the two predicates are split **exclusively** from {@link _i4_module_names} (the mixed MUST
    of `quantize.py`). The returned overrides are the mapping "assign every key of the i4 scale
    ledger to i4", and emit fails loudly if it cannot honour an explicit assignment.

    `calib_args` takes effect only for i4 (`None` = plain RTN without calibration — an opt-out for
    tests).
    """
    if dtype == "f32":
        return {}, {}
    if dtype == "i8":
        report = fake_quant_int8(wrapper)
        print(f"[fake-quant] rounded to i8 per-channel — {report.describe()}", flush=True)
        return report.scales, {}
    i4_names = _i4_module_names(wrapper)
    int4_scales = (
        _round_i4_plain(wrapper, i4_names, "linear / embedding")
        if calib_args is None
        else _round_i4_calibrated(wrapper, i4_names, calib_args)
    )
    # MUST: the rounded set = the stored set (the override keys). A mismatch has the form "counted
    # as eligible but not rounded", which stays green with i8 weights mixed into the i4 storage.
    if set(int4_scales) != {f"{name}.weight" for name in i4_names}:
        raise AssertionError(
            f"{len(i4_names)} i4-eligible weights, but {len(int4_scales)} were rounded (difference:"
            f" {sorted(set(int4_scales) ^ {f'{name}.weight' for name in i4_names})[:3]})"
        )
    int8 = fake_quant_int8(wrapper, include=lambda name: name not in i4_names)
    print(f"[fake-quant] the rest is i8 per-channel — {int8.describe()}", flush=True)
    return {**int8.scales, **int4_scales}, dict.fromkeys(int4_scales, "i4")


def _write_export_provenance(target: str, sym_max: int, out_dir: Path) -> None:
    """Record in the series **which symbolic-dimension bound** this variant was baked with.

    `--sym-max` is a research knob that accepts any value, and the value given becomes the `Dim`
    max as is. But `pipelineConfig.maxTokens` of the SBV2 distribution is baked as a constant
    (`SBV2_MAX_TOKENS` in `sbv2/distribution.py`), so plugging a series re-taken off the default
    into the text_encoder component makes only the declaration a lie while **export stays green and
    the distribution stays green**, and long inputs first fail on the user's machine.

    MUST: this fact **can be carried only by a record**. front / voice hold the bound as a baked
    constant of `sym_prefix_slice`, so it can be read back from the artifact
    (`sbv2.distribution.assert_baked_sym_max`), but DeBERTa promoted the relative-position index
    tables to graph **inputs** in ADR 0045 wave 3, so not one constant carrying the bound remains
    in the graph. `sbv2.distribution.assert_sym_provenance` does the comparison.

    `target` is the variant name (layer count × output form — {@link VARIANTS}). `deberta/export.py`
    has no `--target`, so this is the only thing that can state "which form it was baked as", and
    the distribution side looks the component up by that spelling
    (`sbv2.distribution.SBV2_TEXT_ENCODER_VARIANT`).

    MUST: it is written **inside the staging directory** (inside the `with` of
    `staged_publication`) — the swap into place happens once, so a pair of a new container and an
    old record cannot arise (the same discipline as `_staged_target` in `sbv2/export.py`;
    `tests/test_staged_publication.py` enforces mechanically via the AST that it stays inside).
    """
    record = {"target": target, "sym_max": sym_max}
    (out_dir / EXPORT_PROVENANCE_FILE).write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def export_variant(
    model_id: str,
    num_layers: int,
    out_dir: Path,
    *,
    sym_max: int = SYM_MAX,
    dtype: str = "f32",
    act_quant: bool = False,
    single_output: bool = False,
    calib_texts: Sequence[str] | None = CALIB_TEXTS,
) -> dict[str, Any]:
    """Write the IR container and the golden io for one layer count, and return a summary.

    `calib_texts` is the calibration corpus that **only the i4 series** reads (default = all 48
    sentences). `None` goes back to plain RTN i4 without calibration — an opt-out for tests that
    the CLI cannot reach (`--dtype i4` is always calibrated).

    MUST: the outputs are written to the staging directory and published only after passing
    **every check** (the input order). Placing them at the final location before the checks would
    let a failed run leave "assets that pass the acceptance checks" — the io goldens are taken from
    the same broken wrapper, so they are consistent with it and the TS-side comparison **turns
    green** (the spelling of "when publishing is allowed" is {@link _shared.decode_series._publish};
    the discipline of the swap and the cleanup is the core primitive
    {@link karume.artifacts.staged_publication}). The provenance record
    ({@link _write_export_provenance}) is written into the same staging directory just before
    leaving it.
    """
    from transformers import AutoTokenizer

    # MUST: the replacement comes before the goldens are taken (the docstring of `deberta.patch`)
    # — applying it later computes only the expected values on the original path (the tables built
    # internally), which stays green while disagreeing with the graph.
    model = load_model(model_id, num_layers)
    patch.assert_supported(model.config)
    patch.apply_external_rel_pos_patch()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=MODEL_REVISION)
    cases = build_cases(tokenizer, model)
    wrapper = HiddenStatesWrapper(model, single_output=single_output)
    # MUST: the calibration inputs run through **the wrapper after the patch** (the patch changes
    # the actual signature of a stage).
    calib_args = (
        build_calib_args(tokenizer, model, calib_texts)
        if dtype == "i4" and calib_texts is not None
        else None
    )
    scales, dtype_overrides = _fake_quant(dtype, wrapper, calib_args=calib_args)

    # The example input is the padded case (a real token sequence with zeros in the mask). min=2
    # avoids the 0/1 specialization, and max is exactly the bound `export_provenance.json` carries
    # (not maintained twice through a separate knob).
    _, example_args = cases[-1]
    seq = Dim("T", min=2, max=sym_max)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with staged_publication(out_dir) as staged:
        # The writer creates the directory (the primitive does not create it — it only hands over
        # a path).
        staged.mkdir()
        graph = export_to_file(
            wrapper,
            tuple(example_args[key] for key in INPUT_ORDER),
            staged / MODEL_FILE,
            provenance=PROVENANCE,
            # The graph name is **the component name** ({@link GRAPH_NAME}) — not derived from
            # the directory name (`full-24layer`).
            graph_name=GRAPH_NAME,
            # The index tables are `[T, T]` — both axes share one symbol (squareness is pinned at
            # the export step).
            dynamic_shapes=({1: seq}, {1: seq}, {0: seq, 1: seq}, {0: seq, 1: seq}),
            weight_dtype=BASE_WEIGHT_DTYPES[dtype],
            weight_scales=scales,
            weight_dtype_overrides=dtype_overrides,
        )
        declared = tuple(item.name for item in graph.inputs)
        if declared != INPUT_ORDER:
            raise AssertionError(
                f"the graph input order is {declared}, not the expected {INPUT_ORDER}"
            )
        # MUST: the plain golden io is taken **without hooks** (the docstring of
        # `_write_mirror_io`).
        written = _write_io(wrapper, graph, cases, staged)
        mirror: list[str] = []
        attached = 0
        if act_quant:
            mirror, attached = _write_mirror_io(wrapper, graph, cases, staged)
        _write_export_provenance(out_dir.name, sym_max, staged)

    model_bytes = sum(p.stat().st_size for p in container_parts(out_dir / MODEL_FILE))
    return {
        "layers": num_layers,
        "dir": str(out_dir),
        "dtype": dtype,
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": model_bytes,
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "act_quant_io": mirror,
        "act_quant_linears": attached,
        "calib_texts": len(calib_args) if calib_args is not None else 0,
        "case_lengths": {name: int(args["input_ids"].shape[1]) for name, args in cases},
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: the series of each --dtype —"
        " outputs/series/deberta{,-i8,-i4}/)",
    )
    parser.add_argument(
        "--dtype",
        choices=WEIGHT_DTYPES,
        default="f32",
        help="storage dtype of the weights (i8 fake-quantizes, then stores only the eligible slots"
        " compressed — ADR 0019. i4 is mixed: the eligible linear / embedding are group32 i4, the"
        " rest i8 — ADR 0069. The i4 encoder linears are rounded with GPTQ calibration"
        f" (calibration corpus of {len(CALIB_TEXTS)} sentences))",
    )
    parser.add_argument(
        "--act-quant",
        action="store_true",
        help="additionally write mirror io with the inputs of the eligible linears fake-quantized"
        " to per-token i8 (the mirror of the runtime's linearCompute:'a8' — use it with"
        " --dtype i8 for the weights)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=sorted(VARIANTS),
        help=f"layer counts to write (default {sorted(VARIANTS)})",
    )
    parser.add_argument("--sym-max", type=int, default=SYM_MAX)
    args = parser.parse_args(argv)
    # MUST: activation i8 applies only to **linears with i8-resident weights** (ADR 0025 decision
    # 1). A mirror of f32 assets is "expected values the runtime never reproduces", so it is
    # rejected mechanically.
    if args.act_quant and args.dtype != "i8":
        raise SystemExit(
            f"--act-quant requires --dtype i8 (given {args.dtype}) — "
            "activation i8 applies only to linears with i8-resident weights (ADR 0025)"
        )
    if args.out is None:
        args.out = DEFAULT_OUT_ROOTS[args.dtype]

    summaries = []
    for num_layers in args.layers:
        variant = VARIANTS.get(num_layers, Variant(f"{num_layers}layer"))
        summaries.append(
            export_variant(
                args.model,
                num_layers,
                args.out / variant.name,
                sym_max=args.sym_max,
                dtype=args.dtype,
                act_quant=args.act_quant,
                single_output=variant.single_output,
            )
        )
    print(json.dumps({"model": args.model, "variants": summaries}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
