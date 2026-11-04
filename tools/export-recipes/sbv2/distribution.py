"""SBV2 distribution recipe — series layout, output path table, quant table, card selection (ADR
0065 decision 2).

The generic assembly engine (placement, folding of shared seats, sha256, manifest, staging/swap,
verification) lives in `karume.dist`. This module holds only **the facts specific to SBV2**: what
to pick up from which series directory, at which path of the distribution and under which dtype
label to place it, and which quant is the default.

Only **the 3 graphs and 4 host assets needed to run** are shipped (the SBV2 example of ADR 0038
§2 lists 5 graphs, but that only illustrates the shape). `front` = enc_p + dp + sdp and `voice` =
flow + dec are fused, so `dp` / `flow` / `dec` are standalone graphs for golden verification only
and are not part of the distribution.
`text_encoder` (DeBERTa) has no f16 seat because `deberta/export.py` has no f16 (1.32GB of f32 is
impractical to ship). The default i8 was accepted by ADR 0026 including its listening check, and
the i4 mix (linear and embedding in i4 group32) was added later as the seat of the `i8+bert4`
quant (perf-ledger Q-1 — the default quant since 2026-08-20). `front` / `voice` also have an i4
mixed seat (eligible linear and conv1d in i4 — conv1d is the addendum of wave J-5b), and the `i4`
quant picks i4 for all 3 seats (an opt-in that favours speed / size).

Of the host assets, `style_vectors` / `speaker_embeddings` take the form **ship the table, look up
the row at run time**. The graph inputs `style_vec[1,256]` / `g[1,512,1]` of `front` / `voice` are
built from these 2 tables, and the name → row mapping is held by `styles` / `speakers` of
`pipelineConfig` (the 3 form one set).

`text_encoder` comes from a series written by the `deberta` recipe, but only **its output path**
is referenced (there is no code import — the shared seat shares assets, it does not couple the
code).

The public surface is **two, one per voice family** (`karume.dist.Pipeline`) — {@link PIPELINE}
(`--pipeline sbv2` = JVNV, public repository `karume-sbv2-jvnv`) and {@link FN_PIPELINE}
(`--pipeline sbv2-fn` = FN, publication on hold). The repository's dist driver
(`tools/export-recipes/dist.py`) merges them into the core's PIPELINES. They are split for the same
reason as anima's official / fine-tuned ones: `root_files` (`LICENSE.md` / `NOTICE.md` at the root
of the distribution repository) is one fixed set per Pipeline, and the license differs per family
(JVNV is CC BY-SA 4.0, FN is Booth's distribution terms). Folding them into one would make one
family's repository declare legal text and attribution that are not its own — still valid prose,
so `verify_dist` and the manifest checks pass it, and nobody can notice until after shipping.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from safetensors import safe_open

from _shared.licenses import cc_by_sa_4_0
from _shared.paths import INPUTS_ROOT, MISC_ROOT
from karume.dist import (
    Artifact,
    DistError,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assert_model_name,
    assert_storage,
    assert_storage_absent,
    complete_quant_weights,
    ir_graph,
    table_payload,
)
from sbv2.card import (
    SBV2_CARD_PROFILES,
    SBV2_JVNV_CORPUS_PAGE,
    SBV2_JVNV_CORPUS_PAPER_URL,
    SBV2_JVNV_LICENSE_URL,
    SBV2_JVNV_PROFILE,
    SBV2_TEXT_ENCODER_MODEL,
    render_sbv2_model_card,
)

#: The default model name — the one word that ties the series spelling (`sbv2-F1{,-f16,-i8}`) to
#: the location of the real weights. `sbv2.export.default_out_root` builds the series name from the
#: directory name of `--model-dir`, so this reading side builds from the same word.
SBV2_DEFAULT_MODEL = "F1"

#: The default model name of the FN Pipeline ({@link FN_PIPELINE}) (the speaker `sbv2/README.md`
#: uses in its steps).
SBV2_FN_DEFAULT_MODEL = "FN4"

#: Model name (= the directory name of `--model-dir`) prefix → voice family (exactly the
#: convention of the README's "the default `--model-dir` is `inputs/sbv2/F1`" and
#: "`inputs/sbv2/FN*`"). Export (`sbv2.export.sbv2_family` — the license baked into the container)
#: and assembly (the models each family's Pipeline accepts) both read this one table.
#:
#: MUST: decided by an **allow-list**. Back when it was the binary "jvnv unless it starts with
#: `FN`", an unexpected directory name (a new speaker, the name of a local copy, another spelling of
#: an FN model without `FN`) **silently** produced a distribution declaring the jvnv license
#: (`cc-by-sa-4.0`) — a silent wrong value in a legal fact, which nobody can notice until after
#: shipping.
#:
#: NOTE: the spelling of the local input directories differs from **the directory names of the
#: upstream repository** (`sbv2.card.Sbv2CardProfile.source_dirs` = `jvnv-F1-jp/` …); on the jvnv
#: side it is the speaker id alone.
SBV2_FAMILY_DIRS: Mapping[str, tuple[str, ...]] = {
    "fn": ("FN",),
    "jvnv": ("F1", "F2", "M1", "M2"),
}

#: The prefix of series and repository names (`sbv2-<model name>`).
SBV2_SERIES_PREFIX = "sbv2"

#: The pipeline contract (ADR 0041 §2 — per model).
SBV2_PIPELINE = "sbv2/1"

#: The DeBERTa series and its variant directory (the spelling of `deberta/export.py`). Independent
#: of the model name (in a family assembly every model points at the same text_encoder, which goes
#: into `shared/` once). The variant has 22 layers because the last 2 layers are dead on the SBV2
#: path ({@link SBV2_TEXT_ENCODER_LAYERS}).
#: The i4 series is **mixed** (linear and the vocabulary table = i4 group32, conv and the
#: relative-position tables = i8 — `BASE_WEIGHT_DTYPES` / `I4_MODULE_TYPES` of
#: `deberta/export.py`) and becomes the text_encoder seat of the `i8+bert4` quant.
SBV2_TEXT_ENCODER_SERIES = "deberta-i8"
SBV2_TEXT_ENCODER_I4_SERIES = "deberta-i4"
SBV2_TEXT_ENCODER_VARIANT = "sbv2-22layer"

#: The number of encoder layers left in the distributed text_encoder. It is the layer count that
#: makes `hidden_states[-3]`, which SBV2 uses (`[0]` = embedding output, `[i+1]` = output of layer
#: i, so the 22nd from the start = the output of layer 21), **the graph's final output**; the
#: reference implementation's index is resolved into this one number.
SBV2_TEXT_ENCODER_LAYERS = 22

#: The number of graph outputs of the distributed text_encoder. SBV2 reads only one, and the
#: runtime reads back **all** of `graph.outputs`, so shipping it with every layer as an output
#: would pay staging + mapAsync for 22 unused outputs on every run (measured in ADR 0045 wave 2 —
#: −10.6% at T=512).
SBV2_TEXT_ENCODER_OUTPUTS = 1

#: The order of the graph inputs of the distributed text_encoder (the same as
#: `deberta.export.INPUT_ORDER`). **Having the 2 relative-position index tables as inputs** is the
#: result of wave 3 itself; going back to baking them in brings back 2MiB of dead weight (the values
#: stay correct, so the E2E does not catch it).
SBV2_TEXT_ENCODER_INPUTS: tuple[str, ...] = (
    "input_ids",
    "attention_mask",
    "c2p_pos",
    "p2c_pos",
)

#: Picks the encoder layer number out of an initializer name (an IR v2 initializer name = the tensor
#: key `model.encoder.layer.<i>....` — docs/ir-v2.md). The layer-count check counts with it.
SBV2_LAYER_PATTERN = re.compile(r"layer[._](\d+)[._]")

#: Location and spelling of the host assets `sbv2.demo assets` writes. They are not a series (IR +
#: io), so they are not under `outputs/series/`.
SBV2_DEMO_DIRNAME = "sbv2-demo"
SBV2_SYMBOLS_FILE = "symbols.json"
SBV2_TOKENIZER_FILE = "deberta-tokenizer.json"

#: Location of the real weights and the config (the same place as `sbv2.export.DEFAULT_MODEL_DIR`).
SBV2_CONFIG_FILE = "config.json"
SBV2_STYLE_FILE = "style_vectors.npy"

#: **Model-specific fields** of `symbols.json` → keys of `data` in `config.json` (the pairs {@link
#: assert_symbols_match_model} compares). The TS side does not read `style` / `speaker` (there is no
#: such field in `JpExtraRules` of `symbols.ts`), so only these 2 fields are compared.
SBV2_SYMBOLS_MODEL_FIELDS: Mapping[str, str] = {
    "samplingRate": "sampling_rate",
    "hopLength": "hop_length",
}

#: The width of a style vector (the 256 of `front`'s graph input `style_vec[1,256]`).
#: `sbv2.export.style_vector` holds the same value — accepting a table of another width would
#: assemble a green distribution that fails against the graph input **only on the user's machine**.
SBV2_STYLE_WIDTH = 256

#: Where the speaker embeddings come from (the tensor key in the ckpt). `front` / `voice` both take
#: `g[1,512,1]` as a graph input, so **without this table the distribution alone cannot run the
#: graphs**.
SBV2_SPEAKER_TENSOR = "emb_g.weight"

#: Tensor keys of the shipped tables (the only key when moving a `.npy` / ckpt into a one-tensor
#: safetensors).
SBV2_STYLE_KEY = "style_vectors"
SBV2_SPEAKER_KEY = "speaker_embeddings"

#: Relative output paths (**within the model subtree**) — the one place shared by the placement
#: table, the conversion targets and the manifest. Only `style_vectors` / `speaker_embeddings` are
#: outputs of a conversion rather than a placement, so they do not appear in
#: {@link sbv2_placements}.
#: `text_encoder` alone keeps the i8 seat's spelling with no dtype suffix on the role name because
#: the second storage form (the i4 mix) is a seat added later — renaming an existing seat's role
#: would move the distribution's paths too, so only the added side gets a suffix.
SBV2_OUTPUT_PATHS: Mapping[str, str] = {
    "text_encoder": "text_encoder/model.i8.krm",
    "text_encoder_i4": "text_encoder/model.i4.krm",
    "front_f16": "front/model.f16.krm",
    "front_i8": "front/model.i8.krm",
    "front_i4": "front/model.i4.krm",
    "voice_f16": "voice/model.f16.krm",
    "voice_i8": "voice/model.i8.krm",
    "voice_i4": "voice/model.i4.krm",
    "tokenizer": "tokenizer/deberta-tokenizer.json",
    "symbols": "text/symbols.json",
    "style_vectors": "styles/style_vectors.safetensors",
    "speaker_embeddings": "speakers/speaker_embeddings.safetensors",
}

#: The text_encoder seats (one per storage form). MUST: the check {@link assert_bert_hidden} runs
#: **per seat** — the 2 are independent containers exported separately, so a distribution where
#: only one has a shifted layer count, output form or input order assembles without complaint (the
#: shapes still match and it runs, producing audio from another layer's BERT features — silently).
SBV2_TEXT_ENCODER_ROLES: tuple[str, ...] = ("text_encoder", "text_encoder_i4")

#: Storage dtype requirements (the same rationale as Anima's {@link STORAGE_REQUIREMENTS} — an
#: actual incident where a plain F32 asset passed assembly, loading and execution and stayed silent
#: until the reference-agreement check). tokenizer / symbols (JSON) and style_vectors (F32 written
#: here) are not listed.
#:
#: `text_encoder` is an i8 series, so I8; `text_encoder_i4` / `front_i4` / `voice_i4` are mixed (F32
#: + I8 + I4 together), so they **require I4** — {@link assert_storage} is a one-way check of "is
#: the required dtype in the header", so it names the only dtype that tells the 2 series apart
#: within the set. Requiring I8 would let the i8 series through and a swapped seat would stay
#: silent (an i8 series in an i4 seat assembles a distribution whose size alone went back, with
#: layer count and shapes still matching).
SBV2_STORAGE_REQUIREMENTS: Mapping[str, str] = {
    "text_encoder": "i8",
    "text_encoder_i4": "i4",
    "front_f16": "f16",
    "front_i8": "i8",
    "front_i4": "i4",
    "voice_f16": "f16",
    "voice_i8": "i8",
    "voice_i4": "i4",
}

#: Storage vocabulary that **must not appear** in each role's binding table
#: ({@link assert_storage_absent}). The presence check of {@link SBV2_STORAGE_REQUIREMENTS} is
#: one-way, so **a swap between compressed seats** gets through — the i4 series is mixed (F32 + I8 +
#: I4) and the weights not eligible for i4 stay i8, so it **always contains I8**. Plugging the i4
#: series into the i8 seat of `text_encoder` / `front_i8` / `voice_i8` therefore satisfies "contains
#: I8", and assembly, verify_dist and the manifest checks all pass. The result is a distribution
#: whose "seat name and path say `model.i8.krm` but the contents are the i4 mix"; the i8 seat
#: computes in f32, so execution raises nothing either, and the audio just comes out at i4 quality,
#: silently.
#: MUST: the prohibitions are held **as a set per role** (writing a single dtype means that on the
#: day a 4th series appears, the one not named silently gets through — the same discipline as anima
#: / irodori). The f16 seats are locked twice, by the absence of both I8 and I4.
SBV2_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    "text_encoder": ("i4",),
    "front_f16": ("i8", "i4"),
    "front_i8": ("i4",),
    "voice_f16": ("i8", "i4"),
    "voice_i8": ("i4",),
}

#: The component names in the distribution = the keys of the manifest's weights = **the graph names
#: of the containers** (container-v1 §2.1 — the runtime looks them up by name with
#: `prepareContainer(opened, <weights key>)`). The writers take them from here: `front` / `voice`
#: via `TARGET_*` of `sbv2/export.py`, while `text_encoder` is copied by a separate script
#: (`deberta/export.py`) (no import from the producing side to the consuming side — for the same
#: reason as {@link EXPORT_PROVENANCE_FILE}, a deberta-side test checks that they agree).
#:
#: NOTE: `dp` / `flow` / `dec` are standalone graphs for golden verification only and are not in the
#: distribution (module docstring), so they have no component names — the writer uses the target
#: name as the graph name.
SBV2_TEXT_ENCODER_COMPONENT = "text_encoder"
SBV2_FRONT_ROLE = "front"
SBV2_VOICE_ROLE = "voice"

#: The weights declaration (dtype label → role name). The dtype keys are the unified form of ADR
#: 0041 §3 (the 2 forms `{file}` / `{variants}` of v1 are gone). For every role, `i4` is a label
#: pointing at **a mixed series**, which in fact means "i4-eligible weights in i4 group32, the rest
#: i8" — the eligible range differs per script: linear + the vocabulary table for `text_encoder`
#: (`deberta/export.py`), linear + conv1d for `front` / `voice` (`sbv2/export.py` — those with
#: groups == 1 and a row length divisible by group32).
SBV2_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    SBV2_TEXT_ENCODER_COMPONENT: {
        "i8": WeightFiles("text_encoder"),
        "i4": WeightFiles("text_encoder_i4"),
    },
    SBV2_FRONT_ROLE: {
        "f16": WeightFiles("front_f16"),
        "i8": WeightFiles("front_i8"),
        "i4": WeightFiles("front_i4"),
    },
    SBV2_VOICE_ROLE: {
        "f16": WeightFiles("voice_f16"),
        "i8": WeightFiles("voice_i8"),
        "i4": WeightFiles("voice_i4"),
    },
}

#: The assets declaration (unconditional files that do not depend on the quant choice).
SBV2_ASSETS: Mapping[str, str] = {
    "tokenizer": "tokenizer",
    "symbols": "symbols",
    "style_vectors": "style_vectors",
    "speaker_embeddings": "speaker_embeddings",
}

#: Component-override token of a quant name → its weights name (ADR 0074 decision 4 — **the recipe
#: defines the abbreviations, and the quant table of the generated model card always shows the
#: mapping**). `bert` is shorter and easier to read than the role name `text_encoder`, but it
#: becomes vocabulary itself, so the mapping is always shown in what is shipped.
SBV2_QUANT_ABBREVIATIONS: Mapping[str, str] = {"bert": "text_encoder"}

#: The quant table. MUST: **for a role with 2 or more dtypes, every quant names one explicitly**
#: ({@link complete_quant_weights}) — once `text_encoder` gained an i4 seat, the path that picks a
#: default on its own (silently shipping another storage form) was closed.
#:
#: Quant names follow the grammar of ADR 0074 `<storage>[+<component><bits>]…[-<knob>]…`. In SBV2
#: every role has the same storage, so one base word suffices; only the `f16` seat has an i8
#: text_encoder, so it gets `+bert8` (matching reality — there is no f16 text_encoder for the f16
#: seat (`deberta/export.py` has no f16)).
#: `i8+bert4` is configured like `i8` except that `text_encoder` is the i4 mix (BERT's linear and
#: vocabulary table dropped to i4 group32). `i4` is **the i4 mix for all 3 seats** (session is
#: empty = still f32 compute — activations are left alone).
#: The numbers move a lot on f32-identity metrics. It passed a first listening check, but `i4` has
#: slightly lower tension than f32 (perf-ledger Q-1 / research 2026-08-19 §6 — from the RTN i4 of
#: net_g), which is the reason the default went back to `i8+bert4` ({@link SBV2_DEFAULT_QUANT}).
#:
#: `label` / `description` are display fields for selection UIs (ADR 0075 decision 1 — English,
#: limits of 64 / 200 characters). Being the default is not written there (`defaultQuant` already
#: points at it — ADR 0075 decision 3).
#:
#: NOTE: on the net_g side, i4 became **the main source of the size gain** with the conv1d addendum
#: (wave J-5b, ADR 0069 addendum 7) — the eligible conv1d weights (≈56MiB) are halved (in the
#: linear-only days there were 6 weights = under 0.1% of the whole distribution, and the gain was
#: nearly zero). Together with the `text_encoder` reduction (Q-1 / J-5a), `i4` is in name and in
#: fact "the seat that stores the whole distribution in 4 bits".
SBV2_QUANTS: Mapping[str, Any] = {
    "f16+bert8": {
        "weights": {"text_encoder": "i8", "front": "f16", "voice": "f16"},
        "session": {},
        "label": "Highest fidelity (f16 synthesis)",
        "description": "front and voice in f16 storage, with the int8 text encoder — the largest"
        " download, and the closest match to the source checkpoint's audio.",
    },
    "i8": {
        "weights": {"text_encoder": "i8", "front": "i8", "voice": "i8"},
        "session": {},
        "label": "Half size (int8)",
        "description": "Every component stored as int8 and computed in f32 — roughly half the f16"
        " download, with the execution path left unchanged.",
    },
    "i8-a8": {
        "weights": {"text_encoder": "i8", "front": "i8", "voice": "i8"},
        "session": {"linearCompute": "a8"},
        "label": "int8 with int8 linear activations",
        "description": "The int8 weights with per-token int8 activations in the linear layers —"
        " faster on GPUs with dp4a, same download as the plain int8 seat.",
    },
    "i8+bert4": {
        "weights": {"text_encoder": "i4", "front": "i8", "voice": "i8"},
        "session": {},
        "label": "Balanced (int8 + int4 text encoder)",
        "description": "int8 synthesis with the text encoder in GPTQ-calibrated int4 (group-32):"
        " a markedly smaller download that stays audibly level with f32.",
    },
    "i4": {
        "weights": {"text_encoder": "i4", "front": "i4", "voice": "i4"},
        "session": {},
        "label": "Smallest (int4)",
        "description": "Text encoder, front and voice all in int4 group-32 — the smallest download"
        " and the fastest warm start, with slightly lower tension than f32.",
    },
}

#: The default is `i8+bert4` (maintainer re-ruling of 2026-08-20 — the default **puts quality
#: first**. `i4` keeps an audible difference from f32 (slightly lower tension — from the RTN i4 of
#: net_g), so it is an opt-in seat chosen explicitly by those who want speed / size. `i8+bert4` has
#: only BERT in GPTQ-calibrated i4 and sounds nearly identical to f32. History and numbers are in
#: perf-ledger Q-1 / Q-6 and research 2026-08-20 §7). `i8` remains as the opt-in reference — the
#: fixed anchor of the WAV reference check (e2e_sbv2_wav_test).
SBV2_DEFAULT_QUANT = "i8+bert4"

#: The runtime knobs in `pipelineConfig.defaults` (from `style_bert_vits2.constants` —
#: {@link sbv2_knob_defaults}). Within the distribution, this one field of the manifest is the only
#: holder of knob defaults; they are not put into the shared seat `symbols.json` (the MUST of
#: `sbv2.demo.jp_extra_rules`).
SBV2_KNOB_KEYS: tuple[str, ...] = (
    "style",
    "styleWeight",
    "sdpRatio",
    "noiseScale",
    "noiseScaleW",
    "lengthScale",
)

#: **Operating limits** in `pipelineConfig`. They are exactly the symbolic-dimension limits of the
#: baked graphs: `maxTokens` = the DeBERTa token sequence T (`deberta.export.SYM_MAX`; the limit of
#: front's phoneme dimension P, `sbv2.export.SYM_MAX`, is the same value), `maxFrames` = the frame
#: dimension T of flow / voice (`sbv2.export.FLOW_SYM_MAX`).
#:
#: MUST: they match the scripts' values (`sbv2/tests/test_distribution.py` checks that). ADR 0045
#: moved the relative-position tables out to the host, so **the T×T allocation is on the host side**
#: (on the order of 8·T² bytes), and the distribution is the only way to know the allocation limit
#: — a mismatch shows up only on the user's machine, as "the declaration passes but the graph's
#: tables are too small".
SBV2_MAX_TOKENS = 512
SBV2_MAX_FRAMES = 4096

#: The provenance record that writes the symbolic-dimension limit into the series (written by
#: `sbv2.export` and `deberta.export` inside the staging directory). The spelling must agree between
#: the 2 writers and the 1 reader, but `deberta` is not a consumer of SBV2, so it **does not import
#: this module** (the shared seat shares assets, it does not couple the code — module doc).
#: `deberta/tests/test_export.py` checks the agreement on every run.
EXPORT_PROVENANCE_FILE = "export_provenance.json"


class Sbv2SymExpectation(NamedTuple):
    """Declaration of the symbolic dimension a seat should be baked with (the script's target name,
    the IR symbol name, the limit).

    `baked` says whether "that limit **can be read from the artifact's bytes**" — front / voice bake
    the relative-position index tables at Tmax and cut them with `sym_prefix_slice`, so it can be
    read; for text_encoder (DeBERTa), ADR 0045 wave 3 promoted those tables to graph **inputs**, so
    it has no baked constant carrying the limit at all (measured on the 4 shipped containers on
    2026-08-30 — 0 `sym_prefix_slice` nodes). For seats where it cannot be read,
    {@link assert_baked_sym_max_absent} runs instead of {@link assert_baked_sym_max}, and only
    {@link assert_sym_provenance} (the record) compares the limit itself.
    """

    target: str
    symbol: str
    sym_max: int
    baked: bool = True


#: Seat → expected symbolic dimension. `--sym-max` of `sbv2.export` / `deberta.export` is a research
#: knob that accepts any value, and the value passed becomes the max of `Dim` as is. But
#: {@link SBV2_MAX_TOKENS} / {@link SBV2_MAX_FRAMES} above are constants baked into the manifest, so
#: plugging in a series re-taken away from the default leaves **export green and distribution
#: green** while only `pipelineConfig` lies, and the consumer is the first to hit the limit (the
#: host-side `maxFrames` check passes). Two checks close it:
#: {@link assert_sym_provenance} (the record of the writing side) and {@link assert_baked_sym_max}
#: (the baked constants of **the artifact itself** — only for `baked` seats).
#:
#: MUST: the enumeration comes from {@link SBV2_WEIGHTS} — so that on the day one more dtype seat
#: appears, no seat silently gets through because it was not named.
#:
#: The limit of `text_encoder` is {@link SBV2_MAX_TOKENS} because that constant serves as the limit
#: of **both** front's phoneme dimension P and DeBERTa's token dimension T (`deberta.export.SYM_MAX`
#: is the same value). The target name is the variant name because `deberta/export.py` has no
#: `--target`, and the variant (layer count × output form) is the only thing that says what form it
#: was baked as — only this record can name the mix-up of copying dev-2layer / full-24layer into the
#: directory of the 22-layer seat name.
SBV2_SYM_EXPECTATIONS: Mapping[str, Sbv2SymExpectation] = {
    files.file: expectation
    for name, expectation in (
        (
            "text_encoder",
            Sbv2SymExpectation(SBV2_TEXT_ENCODER_VARIANT, "T", SBV2_MAX_TOKENS, baked=False),
        ),
        ("front", Sbv2SymExpectation("front", "P", SBV2_MAX_TOKENS)),
        ("voice", Sbv2SymExpectation("voice", "T", SBV2_MAX_FRAMES)),
    )
    for files in SBV2_WEIGHTS[name].values()
}


def sbv2_series_name(model: str) -> str:
    """The stem of the series name (`outputs/series/<this name>-{f16,i8}/`).

    The spelling is identical to `sbv2.export.default_out_root` — writer and reader build from the
    same word.
    """
    return f"{SBV2_SERIES_PREFIX}-{model}"


def sbv2_repo_name(family: str) -> str:
    """The distribution repository name of a voice family (`<DIST_ROOT>/<this name>/` becomes one HF
    repository).

    Repositories are **per voice family** (the 4 JVNV voices share `karume-sbv2-jvnv`); there are
    no per-speaker repositories (ruling of 2026-09-25 — repository-wide review W-RC5-2). The
    Pipeline declares this name, so the card's Usage and the default output point at the public
    repository even without an explicit `--repo` (when it was derived from the speaker name, it
    declared nonexistent repositories such as `karume-sbv2-F1`).

    It is prefixed with `karume-` (repository-name ruling of 2026-08-09 — instead of creating an HF
    org, the distribution repositories get a namespace from the `karume-` prefix). It does not apply
    to series names ({@link sbv2_series_name}).
    """
    return f"karume-{SBV2_SERIES_PREFIX}-{family}"


@dataclass(frozen=True)
class Sbv2Sources:
    """Assembly inputs. Besides the 2 series, they span 2 locations that are not series.

    `demo` is the host assets written by `sbv2.demo assets` (directly under `outputs/` — not a
    series), `model` is the ckpt and `config.json` / `style_vectors.npy` (`inputs/` — not generated
    output). Neither is under `--series`, so only the first 3 can be derived mechanically from the
    parent of the series.
    """

    series_f16: Path
    series_i8: Path
    series_i4: Path
    text_encoder: Path
    text_encoder_i4: Path
    demo: Path
    model: Path


def sbv2_sources(series_dir: Path, model: str = SBV2_DEFAULT_MODEL) -> Sbv2Sources:
    """Takes the inputs from the parent directory of the series (`outputs/series/`) and the
    spellings of `_shared.paths`."""
    return Sbv2Sources(
        series_f16=series_dir / f"{sbv2_series_name(model)}-f16",
        series_i8=series_dir / f"{sbv2_series_name(model)}-i8",
        series_i4=series_dir / f"{sbv2_series_name(model)}-i4",
        text_encoder=series_dir / SBV2_TEXT_ENCODER_SERIES / SBV2_TEXT_ENCODER_VARIANT,
        text_encoder_i4=series_dir / SBV2_TEXT_ENCODER_I4_SERIES / SBV2_TEXT_ENCODER_VARIANT,
        demo=MISC_ROOT / SBV2_DEMO_DIRNAME,
        model=INPUTS_ROOT / SBV2_SERIES_PREFIX / model,
    )


def sbv2_placements(sources: Sbv2Sources) -> dict[str, Path]:
    """Role name → source file. The output paths are held by {@link SBV2_OUTPUT_PATHS}.

    Anything not in this table does not go into the output (that is all it takes to drop
    `io.*.safetensors`). `style_vectors` (`.npy` → safetensors) and `speaker_embeddings` (one tensor
    of the ckpt → safetensors) are **conversions** rather than placements, so they do not appear
    here.
    """
    return {
        "text_encoder": sources.text_encoder / "model.krm",
        "text_encoder_i4": sources.text_encoder_i4 / "model.krm",
        "front_f16": sources.series_f16 / "front" / "model.krm",
        "front_i8": sources.series_i8 / "front" / "model.krm",
        "front_i4": sources.series_i4 / "front" / "model.krm",
        "voice_f16": sources.series_f16 / "voice" / "model.krm",
        "voice_i8": sources.series_i8 / "voice" / "model.krm",
        "voice_i4": sources.series_i4 / "voice" / "model.krm",
        "tokenizer": sources.demo / SBV2_TOKENIZER_FILE,
        "symbols": sources.demo / SBV2_SYMBOLS_FILE,
    }


def sbv2_knob_defaults() -> dict[str, Any]:
    """Takes the runtime knob defaults from `style_bert_vits2.constants` (the constants are not
    hand-copied).

    In the distribution, the one field `pipelineConfig.defaults` of `karume.json` is the only holder
    of knob defaults (the TS-side `sbv2.Pipeline` reads only that — the doc of `defaults` in
    `symbols.ts`). A copy of the same values used to be baked into the shared seat `symbols.json` as
    well and compared, but that copy had no consumer (removed together with the comparison in the
    2026-09-24 repository-wide review W-RC5-1).

    NOTE: `style_bert_vits2` is the optional `sbv2` dependency-group, so it is imported inside the
    function. Anima's assembly and the import of `karume.dist` itself do not touch this dependency.
    """
    from style_bert_vits2.constants import (
        DEFAULT_LENGTH,
        DEFAULT_NOISE,
        DEFAULT_NOISEW,
        DEFAULT_SDP_RATIO,
        DEFAULT_STYLE,
        DEFAULT_STYLE_WEIGHT,
    )

    return {
        "style": DEFAULT_STYLE,
        "styleWeight": DEFAULT_STYLE_WEIGHT,
        "sdpRatio": DEFAULT_SDP_RATIO,
        "noiseScale": DEFAULT_NOISE,
        "noiseScaleW": DEFAULT_NOISEW,
        "lengthScale": DEFAULT_LENGTH,
    }


def sbv2_ir_graph(path: Path) -> Mapping[str, Any]:
    """Reads the IR v2 graph JSON from the graph descriptor of a distribution candidate (not a
    single byte of the weights is read).

    Two checks look at the graph (layers / outputs / inputs in {@link assert_bert_hidden}, and the
    baked dimensions in {@link assert_baked_sym_max}), so reading and naming inconsistencies live in
    this one function.

    It just calls the core's {@link karume.dist.ir_graph} — this module used to hold a copy of the
    same reading, but when the distribution became always split (container-v1 §8), **only the copy
    kept opening the representative path directly** (only the core reads part 0 by name). The name
    stays because the 2 checks above refer to it by this spelling.
    """
    return ir_graph(path)


def sbv2_baked_sym_bounds(role: str, path: Path, symbol: str) -> list[int]:
    """Collects **every** limit of a symbolic dimension that the graph accepts, from the static
    dimensions of its baked constants.

    `sym_prefix_slice` is the op that "cuts the leading `coeff·sym+offset` out of a static shape
    baked at Tmax" (ADR 0010), and the constant dimension it cuts from is exactly **the sym limit it
    can accept** (going over fails in the runtime's `shapes.ts` with "Tmax exceeded"). So the
    script's `--sym-max` remains inside the asset and can be read from the artifact alone, without
    trusting the record.

    "None at all" is returned as an empty list, leaving the decision to the caller, because it means
    opposite things per seat — for front / voice a sign that the check has become vacuous
    ({@link assert_baked_sym_max}), for text_encoder **the expected state**
    ({@link assert_baked_sym_max_absent}).
    """
    graph = sbv2_ir_graph(path)
    values = graph.get("values")
    nodes = graph.get("nodes")
    if not isinstance(values, dict) or not isinstance(nodes, list):
        raise DistError(f"{role}: the IR metadata of {path} has no values / nodes")
    bounds: list[int] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        attrs = node.get("attrs")
        if node.get("op") != "sym_prefix_slice" or not isinstance(attrs, dict):
            continue
        if attrs.get("sym") != symbol:
            continue
        try:
            shape = values[node["ins"][0]]["shape"]
            for entry in attrs["slices"]:
                bounds.append((shape[entry["dim"]] - entry["offset"]) // entry["coeff"])
        except (KeyError, IndexError, TypeError, ZeroDivisionError) as error:
            raise DistError(f"{role}: unreadable sym_prefix_slice in {path} — {error}") from error
    return bounds


def sbv2_baked_sym_max(role: str, path: Path, symbol: str) -> int:
    """The limit of a symbolic dimension the graph actually accepts (the strictest, if several).

    MUST: fail if there is no target node at all (the vacuity check) — when the graph's form
    changes, for example by promoting the relative-position tables to inputs, this check silently
    degrades into "green without looking at anything".
    """
    bounds = sbv2_baked_sym_bounds(role, path, symbol)
    if not bounds:
        raise DistError(
            f"{role}: {path} has no sym_prefix_slice for symbol {symbol!r} at all — the check"
            " that reads the limit from the baked constants has become vacuous (the graph's form"
            " changed)"
        )
    return min(bounds)


def assert_baked_sym_max(role: str, path: Path, expectation: Sbv2SymExpectation) -> None:
    """Checks that the symbolic-dimension limit of the baked graph matches the declaration in
    `pipelineConfig`.

    The manifest's `maxTokens` / `maxFrames` claim to be "exactly the symbolic-dimension limits of
    the baked graphs" ({@link SBV2_MAX_TOKENS}), yet the values were baked as constants without ever
    looking at the artifact — this closes that gap. Unlike the record
    ({@link assert_sym_provenance}), this derives from **the actual bytes**, so it also works on old
    series without a record.
    """
    baked = sbv2_baked_sym_max(role, path, expectation.symbol)
    if baked != expectation.sym_max:
        raise DistError(
            f"{role}: {path} bakes symbol {expectation.symbol!r} with the limit {baked}, but"
            f" the distribution declares {expectation.sym_max}"
            f" (a series taken with a non-default `sbv2.export --target {expectation.target}"
            " --sym-max`) — export and the distribution both stay green, and only the consumer"
            " fails by exceeding Tmax"
        )


def assert_baked_sym_max_absent(role: str, path: Path, expectation: Sbv2SymExpectation) -> None:
    """Checks that a `baked=False` seat really has no baked constant carrying the limit.

    The limit of text_encoder cannot be read from the artifact (the relative-position tables are
    graph inputs — ADR 0045 wave 3), so only the record ({@link assert_sym_provenance}) compares it,
    and a series without a record is accepted. That "cannot be read" is **a fact about the graph's
    form**, not a permanent guarantee, so the seat's classification silently goes stale on the day
    the form comes back — if the tables get baked in again, `sym_prefix_slice` nodes appear, and the
    right move is to put the seat on the {@link assert_baked_sym_max} side (`baked=True`). This
    failing is the signal for that.

    NOTE: the only static dimension that looks like 512 in the unreadable seat is the `[512, 1024]`
    of `rel_embeddings`, but that is `2 × position_buckets` (= 256), a different thing whose value
    merely equals `max_position_embeddings` (the log bucketing of `deberta/patch.py`). Reading it as
    the source of the limit would **build a check on a coincidence**, so instead this only checks
    that "there is no node carrying the limit".
    """
    bounds = sbv2_baked_sym_bounds(role, path, expectation.symbol)
    if bounds:
        raise DistError(
            f"{role}: {path} has sym_prefix_slice nodes for symbol {expectation.symbol!r}:"
            f" {len(bounds)} of them (limits {sorted(set(bounds))}) — this slot compares only the"
            " record, on the premise that 'the limit cannot be read from the artifact'."
            " If the baking came back, move baked in SBV2_SYM_EXPECTATIONS to True"
        )


def assert_sym_provenance(role: str, path: Path, expectation: Sbv2SymExpectation) -> None:
    """Compares the record left by the writing side (`export_provenance.json`) with the declaration.

    NOTE: a series without a record is accepted. {@link assert_baked_sym_max}, which derives the
    same fact from the artifact, **always** runs, and this only adds "which target it was baked
    as" — turning a missing record into a rejection would force a re-export (= work that moves the
    sha of shipped distributions) on every series baked before records existed, out of proportion
    to the hole it closes.
    """
    record_path = path.parent / EXPORT_PROVENANCE_FILE
    if not record_path.is_file():
        return
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise DistError(
            f"{role}: cannot parse the provenance record: {record_path} — {error}"
        ) from error
    if not isinstance(record, Mapping):
        raise DistError(f"{role}: the provenance record {record_path} is not a top-level object")
    if record.get("target") != expectation.target:
        raise DistError(
            f"{role}: the target in the provenance record is {record.get('target')!r}, not the"
            f" {expectation.target!r} this slot requires (a series of another target is plugged"
            f" in): {record_path}"
        )
    if record.get("sym_max") != expectation.sym_max:
        raise DistError(
            f"{role}: the sym_max in the provenance record is {record.get('sym_max')!r}, not the"
            f" distribution's declared {expectation.sym_max}: {record_path}"
        )


def assert_bert_hidden(text_encoder: Path, symbols_path: Path) -> None:
    """Checks that the distributed text_encoder has the form "outputs only the one layer output SBV2
    uses".

    There is only one correct combination, **22 layers × 1 output × `bertHiddenFromEnd` 1**, but the
    layer count and output form are held by the variant of `deberta/export.py` and the take-out
    position by a constant of `sbv2/demo.py` — **separate scripts** — so a distribution where only
    one of them moved assembles without complaint.

    MUST: when they disagree, the shapes still match, loading and execution pass, and it stays
    silent, **producing audio from another layer's BERT features** (the same mechanism as the
    row-count checks of the style and speaker tables). The 3 are checked separately because each
    catches a different mix-up — the layer count "which layer's output", the output count "whether
    an all-layer-output asset for verification got mixed in", the position "whether only
    symbols.json is stale".

    """
    graph = sbv2_ir_graph(text_encoder)

    initializers = graph.get("initializers")
    if not isinstance(initializers, dict) or not initializers:
        raise DistError(f"{text_encoder}: the IR metadata has no non-empty initializers")
    layers = {match.group(1) for name in initializers if (match := SBV2_LAYER_PATTERN.search(name))}
    if len(layers) != SBV2_TEXT_ENCODER_LAYERS:
        raise DistError(
            f"the encoder of {text_encoder} has {len(layers)} layers, not the expected"
            f" {SBV2_TEXT_ENCODER_LAYERS} — making hidden_states[-3], which SBV2 uses, the final"
            " output needs the variant truncated at 22 layers (VARIANTS in deberta/export.py)"
        )

    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != SBV2_TEXT_ENCODER_OUTPUTS:
        count = len(outputs) if isinstance(outputs, list) else outputs
        raise DistError(
            f"{text_encoder} has {count} graph outputs, not the"
            f" {SBV2_TEXT_ENCODER_OUTPUTS} the distribution requires — an all-layers-out"
            " (verification) asset got mixed in"
        )

    inputs = graph.get("inputs")
    names = (
        tuple(item.get("name") for item in inputs if isinstance(item, dict))
        if isinstance(inputs, list)
        else ()
    )
    if names != SBV2_TEXT_ENCODER_INPUTS:
        raise DistError(
            f"the graph inputs of {text_encoder} are {list(names)}, not the expected"
            f" {list(SBV2_TEXT_ENCODER_INPUTS)} — when the relative-position index tables leave the"
            " inputs, Tmax worth of constants (2MiB) is baked back in (the values stay correct, so"
            " E2E does not catch it)"
        )

    shipped = json.loads(symbols_path.read_text(encoding="utf-8"))
    from_end = shipped.get("bertHiddenFromEnd") if isinstance(shipped, dict) else None
    if from_end != SBV2_TEXT_ENCODER_OUTPUTS:
        raise DistError(
            f"{symbols_path}: bertHiddenFromEnd={from_end!r} is not {SBV2_TEXT_ENCODER_OUTPUTS},"
            " the only value meaningful for a graph with one output — take `sbv2.demo assets`"
            " again"
        )


def assert_symbols_match_model(symbols_path: Path, config: Mapping[str, Any]) -> None:
    """Checks that the **model-specific fields** of `symbols.json` match the `config.json` of the
    model being shipped.

    `sbv2.demo assets` bakes `samplingRate` / `hopLength` from `hps.data` of the model passed with
    `--model-dir`, yet it writes to **one fixed place**, `outputs/misc/sbv2-demo/`, whose spelling
    does not include the model name. So assembling in a tree where another model's assets remain
    ships the symbol table (model-independent) unchanged while these 2 fields carry the other
    model's values.

    MUST: a `samplingRate` mismatch is **silent** — the TS side passes it straight into the
    `sampleRate` returned by `generate()` of `sbv2.Pipeline`, so the WAV is written at the wrong
    frequency and only the playback speed and pitch shift, with no exception (on the `hopLength`
    side the length cross-check fails loudly).
    """
    data = _sbv2_section(config, "data")
    shipped = json.loads(symbols_path.read_text(encoding="utf-8"))
    if not isinstance(shipped, dict):
        raise DistError(f"{symbols_path}: the top level is not an object")
    disagreed = [
        f"{asset_key}: symbols.json={shipped.get(asset_key)!r} / config.json={expected!r}"
        for asset_key, config_key in SBV2_SYMBOLS_MODEL_FIELDS.items()
        if shipped.get(asset_key) != (expected := _sbv2_int(data, config_key, "data"))
    ]
    if disagreed:
        raise DistError(
            f"the model-specific fields of {symbols_path} disagree with config.json"
            f" ({', '.join(disagreed)}) — demo assets of another model remain. Re-run"
            " `sbv2.demo assets` for this model"
        )


def sbv2_config(model_dir: Path) -> Mapping[str, Any]:
    """Reads `config.json` (the source of truth for styles / speakers and the tables' row and column
    counts)."""
    path = model_dir / SBV2_CONFIG_FILE
    if not path.is_file():
        raise DistError(f"assembly input missing: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path}: cannot be read as JSON") from error
    if not isinstance(config, dict):
        raise DistError(f"{path}: the top level is not an object")
    return config


def _sbv2_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Checks and reads one section of `config.json` (`data` / `model`)."""
    section = config.get(name)
    if not isinstance(section, dict):
        raise DistError(f"config.json has no '{name}' section (actual: {section!r})")
    return section


def _sbv2_id_map(data: Mapping[str, Any], key: str) -> dict[str, int]:
    """Checks and reads `data.<key>` of `config.json` (a non-empty name → ID map)."""
    table = data.get(key)
    if not isinstance(table, dict) or not table:
        raise DistError(f"data.{key} of config.json is not a non-empty map (actual: {table!r})")
    for name, value in table.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise DistError(f"data.{key}[{name!r}] of config.json is not an integer ({value!r})")
    return dict(table)


def _sbv2_int(section: Mapping[str, Any], key: str, where: str) -> int:
    """Checks and reads an integer field of `config.json` (a declared row / column count of a
    table)."""
    value = section.get(key)
    # bool derives from int. Letting `"n_speakers": true` through as 1 would loosen the row-count
    # check of the table.
    if not isinstance(value, int) or isinstance(value, bool):
        raise DistError(f"{where}.{key} of config.json is not an integer ({value!r})")
    return value


def sbv2_pipeline_config(config: Mapping[str, Any], knobs: Mapping[str, Any]) -> dict[str, Any]:
    """Builds `pipelineConfig` from config.json and the runtime knobs (the tables are not baked in).

    MUST: `styles` / `speakers` are not hard-coded. A different ckpt changes the names and their
    order (this FN4 has the 4 Neutral / high / low / NSFW, another ckpt the 7 Neutral / Angry / …),
    and shipping a copied table **keeps the shapes matching while another style's voice comes out**.
    For the same reason the numbers of `defaults` take the values looked up from `style_bert_vits2`
    ({@link sbv2_knob_defaults}).

    `defaults.speaker` is the first key of `spk2id` (the same formula as
    `sbv2.demo.resolve_style_and_speaker`). The names of `speakers` resolve to rows of the
    distribution's `speaker_embeddings` ({@link sbv2_speaker_embeddings}), those of `styles` to
    `style_vectors`.

    `maxTokens` / `maxFrames` are **numbers of the baked graphs** that the ckpt does not have, so
    they come from {@link SBV2_MAX_TOKENS} / {@link SBV2_MAX_FRAMES}.
    """
    data = _sbv2_section(config, "data")
    styles = _sbv2_id_map(data, "style2id")
    speakers = _sbv2_id_map(data, "spk2id")
    missing = [key for key in SBV2_KNOB_KEYS if key not in knobs]
    if missing:
        raise DistError(f"runtime knob defaults are missing: {missing}")
    if knobs["style"] not in styles:
        raise DistError(
            f"the default style {knobs['style']!r} is not in the config's style2id {sorted(styles)}"
            " — a distribution whose default is a style name that does not exist fails only at"
            " startup"
        )
    return {
        "styles": styles,
        "speakers": speakers,
        "maxTokens": SBV2_MAX_TOKENS,
        "maxFrames": SBV2_MAX_FRAMES,
        "defaults": {
            "speaker": next(iter(speakers)),
            **{key: knobs[key] for key in SBV2_KNOB_KEYS},
        },
    }


def sbv2_style_vectors(model_dir: Path, config: Mapping[str, Any]) -> np.ndarray:
    """Checks `style_vectors.npy` and reads it as f32 `[number of styles, 256]`.

    MUST: the row count matches **both** `data.num_styles` and `len(data.style2id)`. A style ID is
    the row number itself, so even when rows and names drift apart, loading and execution pass and
    it stays silent, **producing another style's voice** (matching the table's row count is the only
    way to detect it).

    MUST: the column count is checked too ({@link SBV2_STYLE_WIDTH}) — `sbv2.export.style_vector`,
    which reads the same asset, rejects other widths, so if this let one through, the scripts would
    disagree on which `.npy` files they accept.
    """
    data = _sbv2_section(config, "data")
    path = model_dir / SBV2_STYLE_FILE
    if not path.is_file():
        raise DistError(f"assembly input missing: {path}")
    table = np.load(path)
    if table.ndim != 2 or table.shape[1] != SBV2_STYLE_WIDTH:
        raise DistError(
            f"{path}: the shape {table.shape} is not [number of styles, {SBV2_STYLE_WIDTH}]"
        )
    styles = _sbv2_id_map(data, "style2id")
    num_styles = _sbv2_int(data, "num_styles", "data")
    if table.shape[0] != num_styles or table.shape[0] != len(styles):
        raise DistError(
            f"{path}: the row count {table.shape[0]} does not match the config's num_styles"
            f" {num_styles} / style2id ({len(styles)} entries) — a style ID is the row number, so"
            " shipping it misaligned produces another style's voice"
        )
    return np.ascontiguousarray(table, dtype=np.float32)


def sbv2_ckpt(model_dir: Path) -> Path:
    """The ckpt of the real weights. Requires exactly one `*.safetensors` (the same as
    `sbv2.export.load_net_g`).

    With several, "which one the speaker embeddings came from" would change silently.
    """
    ckpts = sorted(model_dir.glob("*.safetensors"))
    if len(ckpts) != 1:
        raise DistError(
            f"{model_dir}: the ckpt is not unique ({len(ckpts)} found: {[p.name for p in ckpts]})"
        )
    return ckpts[0]


def sbv2_speaker_embeddings(model_dir: Path, config: Mapping[str, Any]) -> np.ndarray:
    """Takes `emb_g.weight` of the ckpt as f32 `[number of speakers, gin_channels]`.

    `front` / `voice` both take the speaker embedding `g` as a **graph input**, so without this
    table the distribution alone cannot run the graphs (the demo path baked it into
    `assets.safetensors`). Like `style_vectors`, it takes the form "ship the table, look up the row
    at run time".

    MUST: the row count matches **both** `data.n_speakers` and `len(data.spk2id)` — a speaker ID is
    the row number itself, so a mismatch still loads and runs and stays silent, **producing another
    speaker's voice** (the same mechanism as the row-count check of the style table). The column
    count matches `model.gin_channels` (a value derivable from the config, so the whole shape can be
    pinned).

    MUST: the ckpt is on the order of 251MB. The lazy read of `safe_open` takes **only this one
    tensor** (`load_file` expands everything into numpy).
    """
    data = _sbv2_section(config, "data")
    model = _sbv2_section(config, "model")
    speakers = _sbv2_id_map(data, "spk2id")
    num_speakers = _sbv2_int(data, "n_speakers", "data")
    gin_channels = _sbv2_int(model, "gin_channels", "model")
    ckpt = sbv2_ckpt(model_dir)
    with safe_open(str(ckpt), framework="np") as handle:
        # `keys()` is the list of tensor names in the header (not a dict). `get_tensor` reads only
        # that tensor's byte range, so taking 2KB out of a 251MB ckpt does not load the whole file.
        available = handle.keys()
        if SBV2_SPEAKER_TENSOR not in available:
            raise DistError(
                f"{ckpt} has no {SBV2_SPEAKER_TENSOR} — the speaker embeddings moved elsewhere"
            )
        table = handle.get_tensor(SBV2_SPEAKER_TENSOR)
    if table.ndim != 2:
        raise DistError(f"{ckpt}: {SBV2_SPEAKER_TENSOR} has shape {table.shape}, which is not 2-D")

    if table.shape[0] != num_speakers or table.shape[0] != len(speakers):
        raise DistError(
            f"{ckpt}: the {SBV2_SPEAKER_TENSOR} row count {table.shape[0]} does not match the"
            f" config's n_speakers {num_speakers} / spk2id ({len(speakers)} entries) — a speaker"
            " ID is the row number, so shipping it misaligned produces another speaker's voice"
        )
    if table.shape[1] != gin_channels:
        raise DistError(
            f"{ckpt}: the {SBV2_SPEAKER_TENSOR} column count {table.shape[1]} does not match the"
            f" config's gin_channels {gin_channels} — it disagrees with the width of the graph"
            " input g"
        )
    return np.ascontiguousarray(table, dtype=np.float32)


def sbv2_plan(
    sources: Sbv2Sources, knobs: Mapping[str, Any], model: str = SBV2_DEFAULT_MODEL
) -> ModelPlan:
    """Builds the plan for one SBV2 model (all checks and table reads are done here).

    The knob defaults are taken as an argument to separate **where the values come from** (constants
    of `style_bert_vits2` — an optional dependency-group) from **assembling** the distribution.
    Resolving the source is the job of {@link sbv2_knob_defaults}.
    """
    assert_model_name(model)
    placements = sbv2_placements(sources)
    config = sbv2_config(sources.model)
    pipeline_config = sbv2_pipeline_config(config, knobs)
    style_vectors = sbv2_style_vectors(sources.model, config)
    speaker_embeddings = sbv2_speaker_embeddings(sources.model, config)
    for role, source in placements.items():
        assert_storage(role, source, SBV2_STORAGE_REQUIREMENTS)
        assert_storage_absent(role, source, SBV2_STORAGE_FORBIDDEN)
        expectation = SBV2_SYM_EXPECTATIONS.get(role)
        if expectation is not None:
            if expectation.baked:
                assert_baked_sym_max(role, source, expectation)
            else:
                assert_baked_sym_max_absent(role, source, expectation)
            assert_sym_provenance(role, source, expectation)
    for role in SBV2_TEXT_ENCODER_ROLES:
        assert_bert_hidden(placements[role], placements["symbols"])
    assert_symbols_match_model(placements["symbols"], config)
    artifacts = {
        role: Artifact(SBV2_OUTPUT_PATHS[role], source=source)
        for role, source in placements.items()
    }
    artifacts["style_vectors"] = Artifact(
        SBV2_OUTPUT_PATHS["style_vectors"],
        payload=table_payload(SBV2_STYLE_KEY, style_vectors),
    )
    artifacts["speaker_embeddings"] = Artifact(
        SBV2_OUTPUT_PATHS["speaker_embeddings"],
        payload=table_payload(SBV2_SPEAKER_KEY, speaker_embeddings),
    )
    return ModelPlan(
        name=model,
        pipeline=SBV2_PIPELINE,
        artifacts=artifacts,
        weights=SBV2_WEIGHTS,
        assets=SBV2_ASSETS,
        quants=complete_quant_weights(SBV2_WEIGHTS, SBV2_QUANTS),
        default_quant=SBV2_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


def sbv2_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """Builds the plan for one SBV2 model from the parent of `--series` (the CLI's dispatch
    target)."""
    sources = sbv2_sources(series_dir, model)
    return sbv2_plan(sources, sbv2_knob_defaults(), model)


def sbv2_family_dist_plan(series_dir: Path, model: str, family: str) -> ModelPlan:
    """Builds the plan for one model as the Pipeline of one voice family.

    MUST: anything other than that family's models ({@link SBV2_FAMILY_DIRS}) fails **before
    assembly**. The legal text, attribution and card at the repository root are fixed by the family,
    so assembling an FN voice with the JVNV Pipeline would produce a distribution in which FN
    weights declare the CC BY-SA 4.0 text and JVNV's attribution.
    """
    prefixes = SBV2_FAMILY_DIRS[family]
    if not model.startswith(prefixes):
        raise DistError(
            f"model {model!r} does not belong in the repository of voice family {family!r}"
            f" (it takes the prefixes {', '.join(prefixes)}) — the legal text and attribution at"
            " the repository root would disagree with the contents, so assembly stops (use a"
            " separate --pipeline per family)"
        )
    return sbv2_dist_plan(series_dir, model)


#: Where the JVNV voices come from (the first `base_model` entry of the card = the source of the
#: weights itself).
_JVNV_VOICES_REPO = SBV2_JVNV_PROFILE.metadata.base_model[0]


def _owner(repo: str) -> str:
    """The namespace of an HF repository (the only spelling under which upstream names itself as the
    author)."""
    return repo.split("/")[0]


#: The JVNV notice of modifications and attribution (CC BY-SA 4.0 §3(a)(1) and §3(b)). The legal
#: text is in `LICENSE.md`.
#:
#: It covers 2 works — the voices (`litagin/style_bert_vits2_jvnv`) and the text encoder every model
#: shares and redistributes (`ku-nlp/deberta-v2-large-japanese-char-wwm`). Both are CC BY-SA 4.0, so
#: ShareAlike (§3(b)) is consistent across the whole repository.
#:
#: MUST: the wording corresponds to the contents of the distribution — as a value it is valid prose,
#: so `verify_dist` and the manifest checks pass it, and a mismatch cannot be noticed until after
#: shipping. Source, training data, component names and layer count are built from the same
#: constants the card and the assembly use (the text and the card do not declare different facts).
#:
#: MUST: the voices are not named one by one (it says "speakers from the JVNV corpus, the breakdown
#: is in the model card"). `root_files` is one fixed set per Pipeline and the same wording ships
#: even for a subset of models (e.g. `--model F1 --model F2`), so naming them would also declare
#: voices that were not assembled. The card declares the breakdown from the manifest.
#:
#: NOTE: neither upstream repository ships a copyright notice (only HF metadata and a README), so
#: there is no "copyright notice to retain" under §3(a)(1)(A)(ii). The authors are shown by the
#: namespaces upstream uses.
SBV2_JVNV_NOTICE_MARKDOWN = f"""# NOTICE

This repository redistributes modified forms of two works licensed under the Creative Commons
Attribution-ShareAlike 4.0 International license (CC BY-SA 4.0 — the full text is in `LICENSE.md`
and at <{SBV2_JVNV_LICENSE_URL}>):

- **Voices**: Style-Bert-VITS2 JP-Extra models of JVNV-corpus speakers from
  <https://huggingface.co/{_JVNV_VOICES_REPO}>
  (published by `{_owner(_JVNV_VOICES_REPO)}`) — the model card
  (`README.md`) lists which ones this repository holds. They were trained on the JVNV corpus
  (<{SBV2_JVNV_CORPUS_PAPER_URL}>; corpus page: <{SBV2_JVNV_CORPUS_PAGE}>).
  The source repository states that the corpus's CC BY-SA 4.0 license carries over to the models.
- **Text encoder**: <https://huggingface.co/{SBV2_TEXT_ENCODER_MODEL}>
  (published by `{_owner(SBV2_TEXT_ENCODER_MODEL)}`).

The following changes were made:

- Each voice was re-expressed in the Karume container format (a `.krm` part sequence whose first
  part carries the graph and model descriptors) as two fused graphs: `front` (phoneme encoder +
  duration predictors) and `voice` (flow + HiFi-GAN decoder). Its style vectors and speaker
  embeddings are shipped as separate tables that the host indexes at run time.
- The text encoder was **truncated to its first {SBV2_TEXT_ENCODER_LAYERS} encoder layers** (it
  now ends at the hidden state Style-Bert-VITS2 reads; the layers after it were dropped) and
  re-expressed in the same container format as `text_encoder`.
- **The weights were quantized**: the voices are stored as `f16` / `i8` / mixed and the text
  encoder as `i8` / mixed, where the mixed form stores the group-quantizable weights as `i4` and
  everything else as `i8`.

No retraining and no fine-tuning were performed. The original checkpoints are not distributed here.
The modified forms in this repository are licensed under CC BY-SA 4.0 as well, with no further
restrictions.
"""


def sbv2_jvnv_root_files() -> dict[str, str]:
    """The legal texts placed at the root of the JVNV distribution repository
    (`karume.dist.Pipeline.root_files`).

    CC BY-SA 4.0 §3(a)(1)(C) / §3(b)(2) ask for "the text itself or a URI". The URI alone would
    satisfy it, but the full text is included so that it moves along when the repository is copied
    outside HF ({@link _shared.licenses.cc_by_sa_4_0} **verbatim** — reformatting it would make it
    no longer a copy). The attribution and the notice of modifications are in `NOTICE.md`.
    """
    return {
        "LICENSE.md": cc_by_sa_4_0(),
        "NOTICE.md": SBV2_JVNV_NOTICE_MARKDOWN,
    }


def _sbv2_pipeline(family: str, default_model: str, root_files: Mapping[str, str]) -> Pipeline:
    """Builds the Pipeline for one voice family = one distribution repository.

    The family is bound **once, here** — the accepted models (`plan`), the distribution repository
    name (`repo_name`), the card's attribution (`card_profiles`) and the legal texts at the
    repository root (`root_files`) all come from the same word, so no form exists where just one of
    them points at another family. There is a single attribution option, so `--card-profile` may be
    omitted (an explicit one passes with the same name).

    """
    return Pipeline(
        default_model=default_model,
        repo_name=lambda _model: sbv2_repo_name(family),
        plan=lambda series_dir, model: sbv2_family_dist_plan(series_dir, model, family),
        card_profiles={
            family: partial(
                render_sbv2_model_card,
                profile=SBV2_CARD_PROFILES[family],
                abbreviations=SBV2_QUANT_ABBREVIATIONS,
            )
        },
        root_files=root_files,
    )


#: The row of `--pipeline sbv2` (JVNV — public repository `karume-sbv2-jvnv`).
#:
#: MUST: a **separate Pipeline** from FN — the reason is in the last paragraph of the module doc.
PIPELINE = _sbv2_pipeline("jvnv", SBV2_DEFAULT_MODEL, sbv2_jvnv_root_files())

#: The row of `--pipeline sbv2-fn` (FN — HF publication on hold, only the card machinery is kept).
#:
#: No legal texts: upstream's written terms exist only on the Booth distribution page, and there is
#: no license text to bundle (checked on the source for `sbv2.card.SBV2_FN_METADATA`). On the day
#: publication is decided, fill this in to match those terms.
FN_PIPELINE = _sbv2_pipeline("fn", SBV2_FN_DEFAULT_MODEL, {})
