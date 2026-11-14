"""Script that writes the real-weight Irodori-TTS v4-Small to IR v2 containers (`krm`) + golden io.

This scope covers the **text condition encoders** (G1 / G1a / G1b of the recon), the **speaker
encoder / duration predictor** (G2 / G3 of the same) and the **DiT** (2 graphs, the condition-side
K/V projections G4 and one step G5 — ADR 0047 / 0114); the codec (G6 / G7) is written by a
separate script, `irodori/dacvae/export.py`.

    cd tools/export-recipes
    uv run --with 'transformers==5.14.1' python -m irodori.export
    uv run --with 'transformers==5.14.1' python -m irodori.export --target backbone
    uv run --with 'transformers==5.14.1' python -m irodori.export --dtype f16
    uv run --with 'transformers==5.14.1' python -m irodori.export --dtype i8

transformers is **pinned at 5.14.1** (the same reason as `embeddinggemma/export.py` — when the
modeling code changes, the graph shape changes; in addition `irodori.patch` replaces
`ModernBertAttention.forward` as a class attribute). It is not put in pyproject.toml / uv.lock but
added temporarily with `--with`.

The model implementation is **imported by adding to `sys.path`** (`--source-dir`) from a clone of
GitHub `Aratako/Irodori-TTS` (default `inputs/irodori/Irodori-TTS/`). Importing
`irodori_tts.model` goes through `irodori_tts/__init__.py`, but the only extra dependency it pulls
in is transformers, so no limited-import trick is needed.

## What goes into the graphs (7 targets, B=1, T / S are symbolic dimensions)

| Target         | Input       | Output        | Contents                                          |
| -------------- | ----------- | ------------- | ------------------------------------------------- |
| `backbone`     | `[1,T]` ids | `[1,T,768]`   | ModernBERT-ja-310m (25 layers, shared)            |
| `text-proj`    | `[1,T,768]` | `[1,T,512]`   | text-side projector (residual_mlp)                |
| `caption-proj` | `[1,T,768]` | `[1,T,512]`×2 | caption-side projector (+ `caption_norm`)         |
| `speaker`      | `[1,S,128]` | `[1,S,768]`   | `ReferenceLatentEncoder` + output norm            |
| `duration`     | 5 below     | `[1]`         | `text_norm` + duration (token-sum form)           |
| `dit-context`  | 3 below     | 24 below      | condition-side K/V projections (12 layers, once)  |
| `dit`          | 27 below    | `[1,S,32]`    | one DiT step (12 layers, condition K/V as inputs) |

The 5 inputs of `duration` are `text_state [1,T,512]` / `speaker_vec [1,768]` /
`has_speaker [1,1]` (bool) / `caption_vec [1,512]` / `has_caption [1,1]` (bool).

The backbone is not fused with the projectors because **the backbone is shared by text and
caption** (fusing would duplicate 1.2GB of weights into 2 series). The host runs the backbone
twice and feeds each projector.

### The second output of `caption-proj` (G1b)

The contract of the `caption_vec` that `duration` (G3) consumes is "the masked mean of the caption
sequence **with `caption_norm` applied**" (`DurationPredictor._caption_vec` — the pooling is
`masked_mean`), but the weights of `caption_norm` exist only **inside** the `dit` graph
(`text_norm` / `caption_norm` are contained by their consumer — the reasoning of ADR 0010). For the
host to take the masked mean it needs the normed sequence.

So `caption-proj` gets **a second output = the sequence with `caption_norm` (RMSNorm 512)
applied**, and only the masked mean is left to the host. The first output (the raw projector
output) stays **as is**, and both the `caption_state` input of `dit` and the `duration` chain
consume it as before (the contract of `duration` and its golden are unchanged — the first output
after the re-export matching the old golden **bit for bit** is the measurement behind this claim).

Alternatives (rejected): ① ship the weight of `caption_norm` to the host → trained values leave the
model file, and the canonical path no longer closes within one file (the same reason as
alternative 2 of ADR 0010)
② put the masked mean in the graph as well → the mask becomes a run-time input, and `caption-proj`
alone becomes a target with a run-time mask (its scheme splits from the other 2 text graphs).
`text-proj` does not need `text_norm` (`duration` applies it inside its graph), so it is **left
unchanged**.

### The boundary of `speaker` (G2)

The graph **includes** the `speaker_norm` (RMSNorm 768) that comes right after
`ReferenceLatentEncoder` — `encode_conditions` always applies the two in sequence, so cutting there
would make the RMSNorm a host-side copy of the model computation. The next step,
`_prepend_masked_mean_token` (prepending the time-mean token), **stays on the host in the current
pipeline**: the IR's `cat` accepts `1 + S → S+1` on a symbolic axis (ADR 0046), so this is not a
vocabulary constraint but an implementation boundary. Therefore

- generating and prepending the mean token happens on **the host** (the axis-1 mean of `[1,S,768]`
  + concat — a pure array operation)
- the `speaker_state[:,0]` that `duration` needs is that mean token itself

MUST: the **mask of the reference latent is not in the graph**. The implementation
(`_load_reference_latent`) always builds an all-ones mask when there is a reference, so at B=1 this
is equivalent to "calling with the packed sequence" (`_check_wrapper_equivalence` measures atol 0
on the real weights). No reference (`no_ref`) is a separate path with an **all-zero** mask, but the
torch output is **exactly 0** (SDPA's safe-softmax returns 0 for fully masked rows, and the final
`x * mask_f` makes everything 0 — measured by `_no_reference_evidence`). The host only has to build
a zero matrix; it does not need to call the graph.

### The boundary of `duration` (G3)

The recon's "no sequence input" **does not hold for these weights**: `duration_architecture` is
`token_sum_dual_adarn_zero_no_aux`, which runs over the text sequence token by token and returns
`log1p(Σ softplus(logit))` (`DurationPredictor.forward` in `model.py`). So a sequence input
`[1,T,512]` is needed. The graph covers

- the leading `text_norm` (RMSNorm 512) — so that it **chains directly** from the output of
  `text-proj` (the existing targets are unchanged, so the consumer holds the norm)
- `speaker_vec` / `caption_vec` are **supplied by the host** (the former cut out of the speaker
  mean token, the latter the masked mean of the caption sequence with `caption_norm` applied).
  Putting the caption sequence into the graph would give it 2 symbolic dimensions (T and the
  caption length), so that is not done
- **`has_speaker` / `has_caption` are bool inputs**, keeping the selection of `null_speaker` /
  `null_caption` (`torch.where`) **inside the graph**. Moving it out would turn the 2 trained
  vectors into "values taken out of the model file and handed to the host", and the canonical
  paths without a reference / without a caption would no longer close within one model file (the
  same reason alternative 2 of ADR 0010 was rejected)

### The `dit-context` → `dit` boundary (G4: condition K/V pre-projection, G5: one DiT step)

The source of truth for the design is ADR 0047 (its decision 3 is replaced by ADR 0114). The 3
inputs of `dit-context` are

| Name            | shape            | Contents                                                   |
| --------------- | ---------------- | ---------------------------------------------------------- |
| `text_state`    | `[1,256,512]`    | `text-proj` output right-padded to `max_text_len`          |
| `speaker_state` | `[1,751,768]`    | `speaker` output + mean token prepended, padded to 750+1   |
| `caption_state` | `[1,512,512]`    | `caption-proj` output right-padded to `max_caption_len`    |

and its outputs are the 24 condition-side K / V tensors of the blocks b (`context_k_<b>` /
`context_v_<b>`, each `[1,1519,20,64]`), ordered K → V in ascending b. The contents are upstream's
`JointAttention.project_context_kv` and the first-stage concatenation (text / speaker / caption —
1519 = 256 + 751 + 512); K has `k_norm` applied, and no RoPE is applied. The inputs of `dit` are

| Name            | shape            | Contents                                                   |
| --------------- | ---------------- | ---------------------------------------------------------- |
| `x_t`           | `[1,S,32]`       | patched latent (S = symbolic dimension, min 2 / max 750)   |
| `t_embed`       | `[1,512]`        | `get_timestep_embedding(t, 512)` (**moved to the host**)   |
| `mask`          | `[1,1,1,S+1519]` | bool; self, text, speaker, caption concatenated in order   |
| `context_*_<b>` | `[1,1519,20,64]` | the `dit-context` outputs as is (24, same names and order) |

**G4 is split off** (DECIDED: ADR 0114 — replaces "fold it in" of ADR 0047 decision 3): the
condition-side K/V projections depend on none of `x_t` / `t_embed` / `mask`, so they produce the
same values in every forward of one generation (60 in the typical case, 100 for voice clone). They
run only once per generation as `dit-context`, and the 24 outputs (178 MiB) are handed to `dit` as
resident tensors (ADR 0054). The premise of the old decision, "a separate graph uploads 178MB of
outputs on every run", disappeared with resident tensors. The boundary is **after** the first-stage
concatenation — what stays in `dit` starts at the symbolic-axis `cat` with the self K/V (the second
stage, `S+1519` — ADR 0046).

MUST: the split does not move the numbers by a single bit — `dit-context` calls the same module's
methods in the same order, and `dit` performs the same operations on the same values as before the
split. `export_series` measures every time that `ComposedDitGraph` (`dit(x_t, t_embed, mask,
*dit_context(condition states))`) matches upstream's unpatched `forward_with_encoded_conditions`
in every case at `EAGER_EQUIV_ATOL = 0`.

MUST: the boundary names are spelled as in `irodori.distribution.irodori_context_kv_names` (the
runtime binds resident tensors **by name** — `copyOutputs` and the inputs of `dit`). torch.export
names variadic arguments `context_<i>` and outputs by FX node name, so the IR's boundary names are
renamed after the export ({@link name_boundary}). The block count is not baked in (it is derived
from `len(model.blocks)`).

MUST: **`t_embed` is not in the graph** — `get_timestep_embedding` uses `cos`, and `cos` is not in
the IR op vocabulary (the first-layer decision that added only `sin` — ADR 0043). The host builds
it in 3 lines.

MUST: **`text_norm` / `caption_norm` are contained in `dit-context`** (its inputs are the raw
projector outputs). For the same reason that `duration` contains `text_norm`: moving them out would
turn the 2 trained RMSNorm weights into "values taken out of the model file and handed to the
host", and the canonical path would no longer close within one model file (the reason alternative
2 of ADR 0010 was rejected). Pad rows are 0 and `rms_norm(0)` is exactly 0, so the pad positions
stay 0 through the norm (and are dropped by the mask too). Only on the speaker side does the norm
sit upstream (in the `speaker` target), because prepending the mean token stays on the host (the
G2 boundary — the section above).

**uncond is expressed by the mask alone** (ADR 0047 decision 1): upstream's CFG builds another set
of "context KV with the state set to 0" for each of the text / speaker / caption unconds, but the
contribution of a segment whose mask is 0 is exactly 0 by `exp(−inf)=0`, so the result **matches,
bit for bit, the one that keeps the cond KV and only zeroes the mask**. The 3 uncond golden
variants take their reference values with upstream's uncond (state 0 + mask 0) and pass the graph
**the K/V built from the cond state + a mask with that segment at 0** — the `EAGER_EQUIV_ATOL = 0`
equivalence check passing is itself the proof of this decision (`dit-context` only has to run the
one cond set).

MUST: SDPA is **not preserved** (the default decomposition table = `PRESERVED_OP_PREFIXES`). The
mask is a run-time bool input, so it does not fit the fused-attention contract (no mask / additive
`[1,1,M,N]`) and goes through the decomposed path + `safe_softmax` (ADR 0044). That is why each
target carries its own `preserved`.

`aux_features` (14 dimensions) are **never read** in the token-sum form, so they are not an input.
So that this does not become vacuous, `_duration_aux_is_inert` measures every time that "passing a
different aux does not change the output by a single bit". `_safe_attention_mask`
(`model.py:429`) is the identity for an all-ones mask (`has_any.all()` is true and it returns its
input as is), so under the contract of B=1 + "the text always has at least 1 token" it does not
appear in the graph.

## No run-time attention_mask (the same static scheme as EmbeddingGemma)

MUST: the backbone **does not take `attention_mask` as a run-time input**. Irodori's inference
calls the backbone at a fixed length right-padded to `max_text_len` / `max_caption_len` and drops
the pad positions with a mask, but **at B=1 this is mathematically equivalent to "calling with the
packed length T"** (dropping the pad with a bidirectional mask ⇔ not passing the pad in the first
place). The host packs the sequence, receives the output as `[1,T,·]` and zero-pads later if needed
(on the eager side too, pad rows become exactly 0 through the mask multiplication). The script
**measures** the equivalence (`_static_scheme_evidence` — for every case it matches "the first T
rows of the padded call" against "the packed call").

MUST: the full_attention mask is **not passed** (a dict is passed as `attention_mask`, making it
`{"full_attention": None, ...}`). Called without a mask, ModernBERT builds an all-true band mask,
which in additive form becomes an **all-zero constant** `[1,1,Tmax,Tmax]` (f32, 1MB) — as a value
it is the identity, so it is exactly equivalent to not passing it at all, and the 1MB constant and
9 layers of mask additions disappear. The sliding_attention side (a band of half-width 64) is a
real band, so the model's own `create_bidirectional_sliding_window_mask` builds it, and it is
folded into a Tmax constant + `sym_prefix_slice` (ADR 0010).

MUST: SDPA is **preserved** (`PRESERVED_OP_PREFIXES_WITH_ATTENTION` — the fused attention of ADR
0023). This is a script-local choice and is not put in the default decomposition table (ADR 0023
decision 5).

## Order of patching and reference

The replacements of `irodori.patch` are **process-wide**, so the patches are not applied until the
golden expected values (= the unpatched eager outputs) have all been taken. `export_series` keeps
the order mechanically with `patch.patches_applied()` (breaking it gives a false PASS).

speaker / duration need 2 patches (making complex RoPE real-valued and splitting the rank-2 weight
RMSNorm — the module docstring of `irodori.patch`), and `dit` additionally needs the weightless RMS
folding of LowRankAdaLN. The RMSNorm split and the AdaLN folding are bit-exact by construction, and
the real-valued RoPE is measured to be bit-exact **at the geometry of these weights (head_dim
64)**, so the `EAGER_EQUIV_ATOL = 0` equivalence check passes with the same strictness as for the 3
text graphs (if head_dim changes it may drift by 1 ulp — it then fails at 0, so suspect the
geometry before loosening it).

## Storage dtype series (ADR 0018 / 0019 / 0027 / 0050)

`--dtype f16` / `--dtype i8` each write to a **separate series** (`irodori-v4-small-f16/` /
`irodori-v4-small-i8/`). Sharing one would silently apply the f32 series' net (per-target
tolerances derived from measurements) to the compressed assets, so the series are always kept
apart. Rounding (fake-quant) is applied right after `load_*`, **before the reference / golden is
taken** (the ordering MUST of `fake_quant`).

i8 is per-channel symmetric (ADR 0019), and the scale ledger obtained together with the rounding is
passed to the emit ({@link TARGET_SCALE_SOURCES} holds the remapping "FQN in the bare module → FQN
in the wrapper").

`--dtype i4` is **a series of the DiT's two graphs only (`dit` / `dit-context`)**
({@link DTYPE_TARGETS}): the DiT's 168 linears **inside the blocks, excluding adaLN**, are group32
symmetric int4 (ADR 0069), while adaLN (`attention_adaln` / `mlp_adaln`), everything outside the
blocks (`in_proj` / `out_proj` / `cond_module`) and the rest are i8 — a mix. Of the 168, the 72
condition-side K/V projections go into the `dit-context` container and the other 96 into the `dit`
container — the rounding is applied once to the whole `TextToLatentRFDiT`, and the 2 wrappers write
out the same rounding result, each for the part it owns (ADR 0114). By default the rounding is
**GPTQ-calibrated** (`irodori.calib` — the i4 rounding targets are re-chosen with activations taken
from reference denoising), and only `--no-calib` opts out. Calibration does not change a single
byte of the storage format, so its conditions are recorded in `calib_provenance.json`
({@link _write_calib_provenance} — the same file for both) and matched on the assembly side. The
other 7 roles share the bytes of the i8 series even in the quant preset `i8+dit4`, so no i4 series
is written for them.

MUST: `--dtype` is emit-only (the same nature as the `--verify` exclusivity of `export_sbv2.py`).
This script has no verification-only mode, so no mechanical exclusivity is needed, but the same
discipline holds: references taken with rounded weights are not used for anything other than "the
unpatched reference".

## Output layout

    outputs/series/irodori-v4-small/<target>/model.krm              tensors + 2 descriptors
    outputs/series/irodori-v4-small/<target>/io.<case>.safetensors  inputs, torch CPU outputs

The tensor-key convention of io is the same as for tiny golden / DeBERTa (`input.<graph input
name>` / `output.<position>`).
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.export import Dim

from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import (
    PRESERVED_OP_PREFIXES,
    PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    normalize_boundary_tensor,
)
from karume.dist import NOTICE_FILENAME
from karume.ir import IrGraph, IrInput
from karume.pipeline import export_module, publish_model
from karume.quantize import (
    QUANT_CHANNEL_AXES,
    fake_quant_int4,
    fake_quant_int8,
    round_weights_to_f16,
)
from karume.rope import ROPE_BUFFER_NAMES, assert_rope_lifted

from . import patch
from .card import IRODORI_LICENSE
from .distribution import (
    CALIB_PROVENANCE_FILE,
    IRODORI_CONTEXT_INPUTS,
    IRODORI_DIT_HEAD_INPUTS,
    IRODORI_FLOW_MEANFLOW,
    IRODORI_FLOW_RF,
    IRODORI_MEANFLOW_DIT_HEAD_INPUTS,
    IRODORI_SERIES_ROLES,
    irodori_context_kv_names,
    irodori_dit_head_inputs,
    irodori_flow_parameterization,
)

if TYPE_CHECKING:  # imported lazily at run time (the NOTE of {@link _fake_quant_i4} below)
    from karume.quant_calib import StageSpec

    from .calib import CalibContext

#: Where the real weights live (where `hf download Aratako/Irodori-TTS-v4-Small` unpacks to).
DEFAULT_MODEL_DIR = INPUTS_ROOT / "irodori" / "v4-small"

#: The storage dtypes that can be written. `i8` was added in iteration 2 (the branch point of ADR
#: 0050 decision 6 = the material for measuring the S drift). **w8a8 is a later iteration**
#: (activation quantization needs a dist preset and the discriminating-band E2E). `i4` is **a series
#: of the DiT's two graphs only (`dit` / `dit-context`)** ({@link DTYPE_TARGETS}), rounded with GPTQ
#: calibration by default ({@link _fake_quant_i4}).
WEIGHT_DTYPES: tuple[str, ...] = ("f32", "f16", "i8", "i4")

#: Where the model implementation (a clone of GitHub `Aratako/Irodori-TTS`) lives.
DEFAULT_SOURCE_DIR = INPUTS_ROOT / "irodori" / "Irodori-TTS"

#: The source of the real latent (the host-preprocessing golden of the reference audio that
#: `irodori/dacvae/host.py` writes). **It only reads the ready-made file there** — this script pulls
#: in neither the DACVAE weights nor its encode path (`irodori/dacvae/host.py` has measured bit
#: equality with upstream's `encode_waveform`).
REFERENCE_LATENT_DIR = SERIES_ROOT / "dacvae-32dim" / "host"
REFERENCE_LATENT_CASE = "ref-default"
REFERENCE_LATENT_PREFIX = "case."
REFERENCE_LATENT_KEY = "latent"

#: The command that runs `irodori/dacvae/host.py` (printed as is in the error when the real latent
#: is missing).
REFERENCE_LATENT_COMMAND = (
    "uv run --with descript-audiotools --with einops --with 'transformers==5.14.1'"
    " python -m irodori.dacvae.host"
)

MODEL_FILE = "model.krm"

#: The weights file name of the **upstream checkpoint** (HF's spelling — not the output container).
#: The same fact as `irodori.distribution.IRODORI_CKPT_FILE`; reader and writer are separate
#: modules, so a copy is kept (the direction is one-way, produce → consume — MUST).
CHECKPOINT_FILE = "model.safetensors"

#: The provenance baked into the container (container-v1 §2.3). The license identifier is taken from
#: the card's source of truth ({@link irodori.card.IRODORI_LICENSE}) — the 2 tables do not move
#: independently. `notice` points at `NOTICE.md` at the root of the distribution repository
#: (`irodori.distribution.irodori_root_files`) — as in the other families that ship a NOTICE, so
#: that the modification notice can still be traced when a single krm is taken out.
PROVENANCE = Provenance(license=IRODORI_LICENSE, notice=NOTICE_FILENAME)

TOKENIZER_FILE = "tokenizer/tokenizer.json"
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: The 2 configs in the checkpoint's `__metadata__` (both JSON strings). The implementation's source
#: of truth is `irodori_tts/inference_runtime.py`. **They are read from here, so no connection to HF
#: is needed**.
TEXT_CONFIG_META_KEY = "text_encoder_config_json"
MODEL_CONFIG_META_KEY = "config_json"

#: state_dict prefixes (the attribute names in `irodori_tts/model.py`).
BACKBONE_PREFIX = "pretrained_text_backbone."
TEXT_PROJ_PREFIX = "text_encoder."
CAPTION_PROJ_PREFIX = "caption_encoder."
SPEAKER_PREFIX = "speaker_encoder."
SPEAKER_NORM_PREFIX = "speaker_norm."
TEXT_NORM_PREFIX = "text_norm."
CAPTION_NORM_PREFIX = "caption_norm."
DURATION_PREFIX = "duration_predictor."

TARGET_BACKBONE = "backbone"
TARGET_TEXT_PROJ = "text-proj"
TARGET_CAPTION_PROJ = "caption-proj"
TARGET_SPEAKER = "speaker"
TARGET_DURATION = "duration"
TARGET_DIT = "dit"
#: The DiT's condition-side K/V projections (once per generation, building the inputs of `dit` —
#: ADR 0114). Spelled in hyphen form like `text-proj`; the component name (the key in the
#: manifest's weights) is `dit_context` (`irodori.distribution.IRODORI_SERIES_DIRS`).
TARGET_DIT_CONTEXT = "dit-context"
#: The 3 that share the same token-sequence golden cases (the chain in which 2 projectors consume
#: the backbone).
TEXT_TARGETS = (TARGET_BACKBONE, TARGET_TEXT_PROJ, TARGET_CAPTION_PROJ)
TARGETS = (*TEXT_TARGETS, TARGET_SPEAKER, TARGET_DURATION, TARGET_DIT, TARGET_DIT_CONTEXT)

#: The targets written per storage dtype. The i4 series has only the DiT's two graphs because the i4
#: execution path is limited to linear weight slots (ADR 0069 decision 5), and the non-DiT roles
#: **share the bytes of the i8 series as is** even in the quant preset `i8+dit4`
#: (`irodori.distribution.IRODORI_QUANT_SEATS`) — an i4 series of the other roles would not be
#: picked up by the distribution tables. `dit-context` is included because, before the split, its 72
#: linears were stored as i4 inside `dit` (dropping them to i8 would move the preset's numbers —
#: ADR 0114).
#:
#: MUST: besides narrowing the default, an explicit `--target` is rejected too ({@link main}). If "a
#: series that was written but is not picked up by the distribution tables" silently stayed around,
#: it would become material for the untrue premise "the other roles have been measured in i4 too" in
#: the next iteration.
DTYPE_TARGETS: Mapping[str, tuple[str, ...]] = {
    "f32": TARGETS,
    "f16": TARGETS,
    "i8": TARGETS,
    "i4": (TARGET_DIT, TARGET_DIT_CONTEXT),
}

#: Symbolic dimension name of the 3 text graphs (the name that goes into the IR).
TEXT_SYMBOL = "T"

#: Symbolic dimension name of `speaker`. The length of the reference latent **after patching**, an
#: axis separate from the text's T.
SPEAKER_SYMBOL = "S"

#: Symbolic dimension name of `dit` (the name that goes into the IR — the S of ADR 0047's
#: `x_t[1,S,32]`). It is a different axis of a different graph from `speaker`'s S, and IR symbol
#: names are scoped per graph, so they do not collide.
DIT_SYMBOL = "S"

#: The **torch-side** `Dim` name of `dit`, kept separately from the IR name (`DIT_SYMBOL`).
#:
#: MUST: it must not be `"S"` — `Dim.__add__` builds the derived dimension's name with
#: `sympy.sympify(name)`, so `"S"` resolves to sympy's singleton registry and gives
#: `TypeError: unsupported operand type(s) for +: 'SingletonRegistry' and 'int'`.
#: `dit` is the only target that uses a derived dimension (`S+1519` — ADR 0046) in a declaration
#: (of `mask`), so only it hits this trap (`speaker`'s `Dim("S")` creates no derived dimension, so
#: it is safe). `export_module(symbol_names=…)` renames the IR-side names from torch's internal
#: symbols (`s27` and so on), so whatever the torch name is, `DIT_SYMBOL` is what goes into the IR.
DIT_TORCH_DIM = "L"

#: Upper bound of the symbolic dimension T. **One bound for both text (256) and caption (512)** —
#: the backbone is shared by both, so separate series would make 2 copies of the same 1.2GB of
#: weights. The only constants that folding bakes in are the half-width-64 band mask
#: (`[1,1,Tmax,Tmax]` f32 = 1MB) and the RoPE tables (2 θ variants × cos/sin × `[1,Tmax,64]` =
#: 512KB), so raising Tmax from 256 to 512 adds on the order of 1MB (ADR 0010).
SYM_MAX = 512

#: DACVAE frame rate (Hz). 48kHz / hop 1920 = 25. The codec is a separate repository with separate
#: weights, and this value is not in the checkpoint's config (the only external constant needed to
#: derive the maximum reference latent length from `ref_max_seconds`).
CODEC_FRAME_RATE = 25

#: Lower bound of the symbolic dimensions. Kept at 2 to avoid torch.export's 0/1 specialization (the
#: same reason as the 2-point evaluation of ADR 0010).
MIN_SYM_LENGTH = 2

#: The utterance length (seconds) that sets the upper bound of `dit`'s symbolic dimension S. The
#: implementation's source of truth is the default 30.0 of
#: `inference_runtime.SamplingRequest.max_seconds` (`latent_steps` is cut from this many seconds as
#: `floor(max_seconds × sample_rate / hop_length)`). It is not in the checkpoint's config
#: (`ref_max_seconds` is the upper bound of the **reference latent**, a different thing), so like
#: `CODEC_FRAME_RATE` it lives here as an external constant. The decision to put the bound at 750 is
#: itself ADR 0047.
DIT_MAX_SECONDS = 30.0


def speaker_sym_max(model_config: Mapping[str, Any]) -> int:
    """Upper bound of `speaker`'s symbolic dimension S (the maximum reference latent length after
    patching).

    `ref_max_seconds` (120s) × 25Hz ÷ `speaker_patch_size` (4) = 750. **Derived from the
    checkpoint's config** — hard-coding it would silently keep the old value when the maximum
    reference length changes. The only constant baked in is the RoPE table (`[750,2,64]` f32 =
    384KB), so leaving headroom in the bound barely grows the container.
    """
    seconds = float(model_config["ref_max_seconds"])
    patch = int(model_config["speaker_patch_size"])
    steps = int(seconds * CODEC_FRAME_RATE) // patch
    if steps < MIN_SYM_LENGTH:
        raise SystemExit(
            f"The maximum reference latent length {steps} is below the symbolic dimension's"
            f" minimum {MIN_SYM_LENGTH}"
        )
    return steps


def dit_sym_max(config: Any) -> int:
    """Upper bound of `dit`'s symbolic dimension S (the maximum latent length that can be generated
    — 30s × 25Hz ÷ patch).

    `latent_patch_size` is taken **from the checkpoint's config** (the same source as the last
    dimension of x_t, `latent_dim × latent_patch_size`). The only constant baked in is the RoPE
    table (`[750,2,64]` f32 = 384KB), so leaving headroom in the bound barely grows the container.
    """
    patch = int(config.latent_patch_size)
    steps = int(DIT_MAX_SECONDS * CODEC_FRAME_RATE) // patch
    if steps < MIN_SYM_LENGTH:
        raise SystemExit(
            f"The maximum latent length {steps} is below the symbolic dimension's minimum"
            f" {MIN_SYM_LENGTH}"
        )
    return steps


#: The difference allowed between "the first T rows of the padded call" and "the packed call"
#: (`_static_scheme_evidence`). If the static scheme holds **structurally**, all that remains is
#: rounding from the different softmax reduction lengths, measured at the 1e-5 level. If the scheme
#: is broken (pad positions visible and so on), the difference shows at the scale of the state
#: values, O(10).
STATIC_SCHEME_ATOL = 1e-3

#: The difference allowed for eager equivalence between the wrappers (+ patches) and the real
#: modules. **0 = bit equality is required**. The 3 differences (how qkv is split, dropping the
#: multiplication by an all-ones mask, dropping the addition of an all-true mask) are all exact
#: identities, so there is no room for rounding differences (measured bit-exact as well). A nonzero
#: value means one of the equivalence claims has broken (e.g. SDPA picking a different kernel
#: depending on whether there is a mask) — instead of letting it pass approximately, it fails, so
#: that which one broke is found out first.
EAGER_EQUIV_ATOL = 0.0

#: Lower bound for seeing that the text and caption projectors differ (`_sanity`). Reading the same
#: weights twice would make the difference exactly 0. Measured at O(1).
PROJECTOR_DIVERGENCE_MIN = 1e-2

#: Lower bound for seeing that `text_norm` and `caption_norm` differ (`_norm_divergence`). Reading
#: the same weights twice would make the difference exactly 0. The only check guarding the contract
#: of `caption-proj`'s second output.
NORM_DIVERGENCE_MIN = 1e-3

#: The body of the long case (1 paragraph ≈ 200 tokens). It is there to see, at a T well beyond the
#: sliding_window half-width of 64, that the band mask **really works as a band** (at T ≤ 129 the
#: band covers nearly the whole range, and a mix-up of the window would not show in the golden).
LONG_PASSAGE_A = (
    "秋の朝、駅前の商店街はまだ静かで、シャッターの下りた店の前を、通勤の人がまばらに"
    "歩いていた。パン屋の換気口から流れてくる甘い匂いだけが、これから一日が始まることを"
    "教えている。私は改札の手前で立ち止まり、鞄の中をもう一度だけ確かめた。定期券、財布、"
    "それから昨日の夜にようやく書き上げた原稿の束。指先にざらついた紙の感触が伝わってきて、"
    "そこで初めて、これが現実の予定なのだという実感が戻ってきた。ホームに上がると、向かいの"
    "線路の向こうに、朝日を受けた高いビルが並んでいるのが見えた。ガラスの壁面がひとつずつ"
    "順番に光り始め、まるで街全体がゆっくりと目を覚ましていくようだった。遠くから電車の"
    "近づいてくる音が聞こえてくる。私は深く息を吸い込み、今日これから会う人たちのことを"
    "順番に思い浮かべた。"
)

#: The second and third paragraphs of the long case. Joined with `LONG_PASSAGE_A` they stretch T to
#: the 400-token range and step close to the declared upper bound Tmax = 512 (confirming that
#: `sym_prefix_slice` cuts correctly at lengths close to the bound).
LONG_PASSAGE_B = (
    "打ち合わせの場所は、線路沿いの古い建物の三階だった。エレベーターが無いので階段を"
    "上がるしかなく、途中の踊り場で一度だけ立ち止まって呼吸を整えた。扉を開けると、"
    "細長い部屋の奥に大きな机がひとつ置かれていて、その上には前回の打ち合わせで使った"
    "資料がそのまま積まれていた。窓は北向きで、季節に関わらず光は柔らかい。"
    "担当の人はまだ来ていなかったので、私は椅子に腰を下ろし、持ってきた原稿をもう一度"
    "読み返した。読み返すたびに直したい箇所が見つかるのは毎回のことで、けれども今日は、"
    "書いたときに考えていたことがそのまま残っているように思えた。階段を上がってくる足音が"
    "近づいてきて、私は原稿を閉じ、顔を上げて扉のほうを見た。"
)

LONG_PASSAGE_C = (
    "話が一区切りついたのは、昼を少し回ったころだった。窓の外では雲が薄く広がっていて、"
    "先ほどまで机の上に落ちていた光の四角形が、いつの間にか輪郭を失っていた。担当の人は"
    "資料の余白に細かく書き込みを入れながら、次の段取りを順番に確認していった。締切、"
    "分量、それから読み手として想定している人の年齢層。ひとつずつ言葉にしていくと、"
    "頭の中で漠然としていた輪郭が、少しずつ具体的な形になっていくのが分かった。帰り際、"
    "階段の途中で振り返ると、廊下の奥の窓から差し込む光が、床の埃をゆっくりと照らしていた。"
    "外に出ると風が思っていたより冷たく、私は上着の襟を立てて、来たときと同じ道を駅へ向かって"
    "歩き始めた。商店街はすでに人で埋まっていて、朝の静けさはどこにも残っていなかった。"
)

#: The fixed golden cases `(name, kind, text)`. **The kind** only decides the pad length used by the
#: static-scheme measurement (`_static_scheme_evidence`) (text = `max_text_len` / caption =
#: `max_caption_len`) and does not affect the graph (the backbone is shared by both). The texts come
#: from the examples in the official model card (read-aloud sentences / Voice Design captions / with
#: emoji) and the 3 long paragraphs above. T is 7 / 13 / 22 / 29 / 144 / 404, covering both the
#: short and the long end relative to the declared upper bound Tmax = 512.
GOLDEN_CASES: tuple[tuple[str, str, str], ...] = (
    ("text-short", "text", "今日は近くの店まで歩いて行きました。"),
    (
        "text-formal",
        "text",
        "本日はお越しいただき、誠にありがとうございます。どうぞごゆっくりお過ごしください。",
    ),
    (
        "text-emoji",
        "text",
        "あははっ🤭、それ本当に言ってるの？…😮\u200d💨まぁ、君らしいけどね。",
    ),
    (
        "caption-ja",
        "caption",
        "若く元気な女性の声。カフェの店員のように、明るくハキハキとした少し高めのトーンで話している。",
    ),
    ("text-long", "text", LONG_PASSAGE_A),
    ("caption-long", "caption", LONG_PASSAGE_A + LONG_PASSAGE_B + LONG_PASSAGE_C),
)

#: The **synthetic** golden cases of `speaker` `(name, token count S after patching, random seed)`.
#:
#: Synthetic data (`torch.randn` with a deterministic seed) is kept because these 5 carry **the
#: length coverage** (S = 2 … the declared upper bound 750) — the one real latent gives only 47 rows
#: and reaches neither the upper nor the lower bound. The character of the values (the value range
#: of a DACVAE latent of real audio) is covered by {@link SPEAKER_REAL_CASES}.
#:
#: The lengths come from seconds: 25Hz ÷ `speaker_patch_size` 4 = 6.25 tokens/s, so 6 / 31 / 187
#: correspond to 1s / 5s / 30s, plus **the declared upper bound itself** (120s = 750) and the lower
#: bound 2 of the symbolic dimension. Both ends are covered to see that the RoPE table's
#: `sym_prefix_slice` does not break at the edges.
SPEAKER_CASES: tuple[tuple[str, int, int], ...] = (
    ("ref-min", 2, 101),
    ("ref-1s", 6, 102),
    ("ref-5s", 31, 103),
    ("ref-30s", 187, 104),
    ("ref-max", 750, 105),
)

#: The **real-latent** golden cases of `speaker` `(name, token count S after patching (None = full
#: length))`.
#:
#: The source is {@link REFERENCE_LATENT_KEY}: the reference audio of the official sample (7.6
#: seconds = 190 frames) passed through upstream's deterministic encode, which after patching
#: (`speaker_patch_size` 4, upstream drops the remainder) is 47 rows. Its value range and
#: correlation structure differ from the synthetic standard normal, so these 2 are added **to tie
#: the grounds of the tolerances to the value range seen in real use** (the same reason as
#: `DECODER_CASES` in `irodori/dacvae/export.py`).
#:
#: The short one is set to 6 to line it up with **the same S as the synthetic `ref-1s`** — keeping
#: synthetic and real errors directly comparable at the same length lets the table show whether a
#: tolerance is set by "length" or by "value range".
SPEAKER_REAL_CASES: tuple[tuple[str, int | None], ...] = (
    ("ref-real-short", 6),
    ("ref-real-full", None),
)

#: The golden cases of `duration` `(name, text case name, has_speaker, has_caption)`.
#:
#: The text sequence fed in is **the torch expected value of upstream `text-proj` itself** (a chain
#: — the same values as the projector's golden become duration's input as is). The speaker /
#: caption vectors are also built from the torch expected values of the `speaker` target and of
#: `caption-proj` by **calling** the implementation's `_prepend_masked_mean_token` / `_caption_vec`
#: (the formulas are not copied).
#:
#: All 4 combinations of `(has_speaker, has_caption)` are covered — with only one side, the 2
#: `where`s kept in the graph (the null-vector selection) would become vacuous and a mix-up would
#: not show.
DURATION_CASES: tuple[tuple[str, str, bool, bool], ...] = (
    ("dur-both", "text-short", True, True),
    ("dur-speaker-only", "text-formal", True, False),
    ("dur-caption-only", "text-emoji", False, True),
    ("dur-neither", "caption-ja", False, False),
    ("dur-long", "text-long", True, True),
)

#: The source cases of the speaker / caption vectors of `duration`. There is no reason **to pick 2
#: of different lengths** (the vectors are `[1,768]` / `[1,512]` regardless of length), so each is
#: fixed to one.
DURATION_SPEAKER_SOURCE = "ref-5s"
DURATION_CAPTION_SOURCE = "caption-ja"

#: The source cases of `dit`'s condition states (a chain — the torch expected values of the upstream
#: targets themselves). The text is on the long side for Tmax = 256 (T=144), the caption is a real
#: Voice Design sentence (T=22), and the speaker is the 5s one (S=31 → 32 with the mean token
#: prepended).
DIT_TEXT_SOURCE = "text-long"
DIT_CAPTION_SOURCE = "caption-ja"
DIT_SPEAKER_SOURCE = "ref-5s"

#: The CFG variant names of `dit` (spelled as `independent_names` in `rf.py`). `None` = cond.
DIT_UNCOND_VARIANTS = ("text", "speaker", "caption")

#: The golden cases of `dit` `(name, latent length S, random seed, t, uncond segment)`.
#:
#: `x_t` is a standard normal with a deterministic seed (the x_t of inference is pure noise at t=1
#: and stays noise-like afterwards, so synthetic data does not stray far in the character of its
#: values — that real z has roughly unit variance was measured when the speaker real-latent cases
#: were introduced, which backs this choice).
#:
#: There are 4 case axes:
#:
#: - **cond / the 3 uncond variants** — uncond **keeps the cond state and only zeroes that segment
#:   of the mask**. The reference values are taken with upstream's uncond (state 0 + mask 0), so the
#:   `EAGER_EQUIV_ATOL = 0` equivalence is itself the proof of ADR 0047 decision 1. Mix-ups are
#:   prevented by `_dit_uncond_divergence` (it measures every time that the values really differ
#:   from cond).
#: - **2 points of t** (0.9 / 0.3) — upstream's CFG applies only for t ∈ [0.5, 1.0], so one point
#:   is taken inside and one outside. `t_embed` is generated on the host, so a mix-up shows here as
#:   a value.
#: - **short and long S** — the lower bound 2 of the symbolic dimension, 25 for 1s, and **750, the
#:   declared upper bound itself**. 750 is where the intermediate `scores[1,20,750,2269]` becomes
#:   130MB, the heavy end as a real-GPU check too.
#: - the 3 uncond variants **share x_t / t / the condition states** with the cond 1s case (only the
#:   mask differs).
DIT_CASES: tuple[tuple[str, int, int, float, str | None], ...] = (
    ("dit-cond-min", 2, 301, 0.9, None),
    ("dit-cond-1s", 25, 302, 0.9, None),
    ("dit-cond-late", 25, 303, 0.3, None),
    ("dit-uncond-text", 25, 302, 0.9, "text"),
    ("dit-uncond-speaker", 25, 302, 0.9, "speaker"),
    ("dit-uncond-caption", 25, 302, 0.9, "caption"),
    ("dit-cond-max", 750, 304, 0.9, None),
)

#: Lower bound for seeing that the outputs of cond and the 3 uncond variants differ **from each
#: other** (`_dit_uncond_divergence`).
#:
#: MUST: do not let it become vacuous — even if the segment layout of the mask (the order and
#: offsets of self S / text 256 / speaker 751 / caption 512) breaks, the golden still matches when
#: the wrapper and the reference break **the same way** (e.g. every variant zeroes only positions
#: that were already pad = every variant equals cond / the 3 variants drop the same segment). The
#: **minimum pairwise difference** of the 4 outputs is measured every time to catch that shape.
DIT_UNCOND_DIVERGENCE_MIN = 1e-3


class DitCase(NamedTuple):
    """One `dit` golden case (the rows of {@link DIT_CASES}, plus the interval length of a
    MeanFlow checkpoint — `None` for the RF step, which takes none)."""

    name: str
    length: int
    seed: int
    t: float
    uncond: str | None
    delta: float | None = None


#: The interval length `delta = t − t_next` of each {@link DIT_CASES} row for a **MeanFlow**
#: checkpoint (ADR 0150 — its `dit` takes `delta_embed` too). 0.25 is the interval of upstream's
#: default 4-step schedule; `dit-cond-late` takes the final interval of a step that ends at 0 (t
#: 0.3 → 0). The three uncond variants keep the interval of `dit-cond-1s`, so they still differ
#: from it by the mask alone ({@link _dit_uncond_divergence}).
MEANFLOW_DIT_DELTAS: Mapping[str, float] = MappingProxyType(
    {
        "dit-cond-min": 0.25,
        "dit-cond-1s": 0.25,
        "dit-cond-late": 0.3,
        "dit-uncond-text": 0.25,
        "dit-uncond-speaker": 0.25,
        "dit-uncond-caption": 0.25,
        "dit-cond-max": 0.25,
    }
)

#: The one case only a MeanFlow checkpoint adds: the inputs of `dit-cond-1s` (same S, seed and t)
#: with a different interval, so that "the interval moves the output" is measured
#: ({@link _dit_delta_divergence}). MUST: not vacuous — upstream initializes the last linear of
#: `delta_cond_module` to zero, and if the trained branch stayed negligible a graph that dropped
#: `delta_embed` would still pass every golden.
MEANFLOW_DIT_EXTRA_CASES: tuple[DitCase, ...] = (
    DitCase("dit-cond-1s-wide", 25, 302, 0.9, None, 0.9),
)

#: The pair {@link _dit_delta_divergence} compares (same inputs, different interval).
MEANFLOW_DIT_DELTA_PAIR = ("dit-cond-1s", "dit-cond-1s-wide")

#: Lower bound for seeing that the interval moves the output (the same floor as the mask pairs).
DIT_DELTA_DIVERGENCE_MIN = 1e-3


def dit_cases(flow: str) -> tuple[DitCase, ...]:
    """The `dit` golden cases of a checkpoint of flow parameterization `flow`.

    The RF cases are {@link DIT_CASES} as they are (no interval); a MeanFlow checkpoint gets each
    row's interval from {@link MEANFLOW_DIT_DELTAS} plus {@link MEANFLOW_DIT_EXTRA_CASES}. A row
    without an interval fails loudly — a MeanFlow step cannot run without one.
    """
    if flow == IRODORI_FLOW_RF:
        return tuple(DitCase(*row) for row in DIT_CASES)
    if flow != IRODORI_FLOW_MEANFLOW:
        raise SystemExit(f"Unknown flow parameterization {flow!r}")
    missing = sorted({name for name, *_rest in DIT_CASES} - set(MEANFLOW_DIT_DELTAS))
    if missing:
        raise SystemExit(f"The MeanFlow dit cases {missing} have no interval length")
    return (
        *(DitCase(*row, MEANFLOW_DIT_DELTAS[row[0]]) for row in DIT_CASES),
        *MEANFLOW_DIT_EXTRA_CASES,
    )


class IrodoriSource:
    """The implementation taken from the `irodori_tts` package (imported by adding to `sys.path`).

    MUST: import **from the implementation clone** (the script keeps no copy). The projector is a
    plain 4-line formula, but copying it would set up "upstream changes while only the script stays
    stale and passes silently".
    """

    def __init__(self, source_dir: Path) -> None:
        model_py = source_dir / "irodori_tts" / "model.py"
        if not model_py.is_file():
            raise SystemExit(
                f"Model implementation not found: {model_py}"
                " (pass the directory of `git clone https://github.com/Aratako/Irodori-TTS`"
                " as --source-dir)"
            )
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        from irodori_tts.config import ModelConfig
        from irodori_tts.model import (
            DurationPredictor,
            PretrainedConditionProjector,
            PretrainedTextBackbone,
            ReferenceLatentEncoder,
            RMSNorm,
            TextToLatentRFDiT,
            get_timestep_embedding,
            patch_sequence_with_mask,
        )
        from irodori_tts.text_normalization import normalize_text

        self.backbone_cls = PretrainedTextBackbone
        self.projector_cls = PretrainedConditionProjector
        self.speaker_cls = ReferenceLatentEncoder
        self.duration_cls = DurationPredictor
        self.rms_norm_cls = RMSNorm
        self.model_config_cls = ModelConfig
        self.dit_cls = TextToLatentRFDiT
        #: Prepending the mean token is a staticmethod of `TextToLatentRFDiT`. The script calls it
        #: instead of copying the formula.
        self.prepend_masked_mean_token = TextToLatentRFDiT._prepend_masked_mean_token
        #: Grouping of the reference latent (`[T,32]` → `[T//4,128]`, the remainder dropped).
        #: **Called, not copied** — if upstream changed how the remainder is handled (truncate /
        #: round up), a copied formula would silently stay stale and pass.
        self.patch_sequence_with_mask = patch_sequence_with_mask
        #: Host-side generation of `t_embed` (3 lines of sin/cos). **Called, not copied** — if
        #: upstream changed how θ is split, a copied formula would silently stay stale and pass.
        self.timestep_embedding = get_timestep_embedding
        self.normalize_text = normalize_text

    def model_config(self, raw: Mapping[str, Any]) -> Any:
        """Build a `ModelConfig` from the checkpoint's `config_json`.

        `config_json` also holds training-side values (`max_text_len` / `max_caption_len` /
        `ref_max_seconds`), so only the dataclass fields are picked. The 3 keys left out are not
        discarded; the callers read them directly from the raw dict (`speaker_sym_max` /
        `_static_scheme_evidence`).
        """
        names = {field.name for field in dataclasses.fields(self.model_config_cls)}
        return self.model_config_cls(**{key: value for key, value in raw.items() if key in names})


def read_configs(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read `(text_encoder_config, model_config)` from the checkpoint's `__metadata__`.

    MUST: do not re-fetch the config from HF — the dict embedded in the checkpoint is the source of
    truth for the shape these weights were actually built with (`text_encoder_revision` included).
    """
    with safe_open(str(model_dir / CHECKPOINT_FILE), framework="pt") as handle:
        metadata = handle.metadata()
    if metadata is None:
        raise SystemExit(f"{model_dir / CHECKPOINT_FILE} has no __metadata__")
    missing = [key for key in (TEXT_CONFIG_META_KEY, MODEL_CONFIG_META_KEY) if key not in metadata]
    if missing:
        raise SystemExit(f"The __metadata__ of {model_dir / CHECKPOINT_FILE} has no {missing}")
    return (
        json.loads(metadata[TEXT_CONFIG_META_KEY]),
        json.loads(metadata[MODEL_CONFIG_META_KEY]),
    )


def _sub_state(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    picked = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not picked:
        raise SystemExit(f"The checkpoint has no tensor with the prefix '{prefix}'")
    return picked


def load_backbone(
    source: IrodoriSource,
    state: Mapping[str, torch.Tensor],
    text_config: Mapping[str, Any],
) -> nn.Module:
    """Build `PretrainedTextBackbone` **without loading weights** and put the checkpoint's weights
    on it.

    `load_pretrained_weights=False` takes the `no_init_weights` + `from_config` path, so neither a
    connection to HF nor random initialization happens (every weight put on it comes from the
    state_dict).
    """
    backbone = source.backbone_cls(
        str(text_config.get("_name_or_path", "")),
        config_dict=dict(text_config),
        load_pretrained_weights=False,
    )
    backbone.load_state_dict(_sub_state(state, BACKBONE_PREFIX), strict=True)
    return backbone.eval()


def load_projector(
    source: IrodoriSource,
    state: Mapping[str, torch.Tensor],
    model_config: Mapping[str, Any],
    prefix: str,
    backbone_dim: int,
    output_dim: int,
) -> nn.Module:
    """Build `PretrainedConditionProjector` as the config says and put the weights of the given
    prefix on it.

    MUST: the input dimension is taken from the backbone (no copy is kept).
    `load_state_dict(strict=True)` rejects shape mismatches, so a mix-up does not pass silently.
    """
    projector = source.projector_cls(
        backbone_dim,
        output_dim,
        projector_type=str(model_config["pretrained_projector_type"]),
        hidden_ratio=float(model_config["pretrained_projector_hidden_ratio"]),
        dropout=0.0,
        norm_eps=float(model_config["norm_eps"]),
    )
    projector.load_state_dict(_sub_state(state, prefix), strict=True)
    return projector.eval()


def load_rms_norm(
    source: IrodoriSource,
    state: Mapping[str, torch.Tensor],
    prefix: str,
    dim: int,
    norm_eps: float,
) -> nn.Module:
    """Build the `RMSNorm` of one prefix and load its weights (`load_state_dict(strict=True)` is
    the check)."""
    norm = source.rms_norm_cls(dim, eps=norm_eps)
    norm.load_state_dict(_sub_state(state, prefix), strict=True)
    return norm.eval()


def load_speaker_encoder(
    source: IrodoriSource, state: Mapping[str, torch.Tensor], config: Any
) -> nn.Module:
    """Build `ReferenceLatentEncoder` as the config says and put the checkpoint's weights on it."""
    encoder = source.speaker_cls(config)
    encoder.load_state_dict(_sub_state(state, SPEAKER_PREFIX), strict=True)
    return encoder.eval()


def load_duration_predictor(
    source: IrodoriSource, state: Mapping[str, torch.Tensor], config: Any
) -> nn.Module:
    """Build `DurationPredictor` as the config says and put the checkpoint's weights on it.

    MUST: the arguments are passed from the config in the same order as in
    `TextToLatentRFDiT.__init__` (the only place that copies the construction in `model.py` — if a
    dataclass name changes, it fails with `AttributeError`).
    """
    predictor = source.duration_cls(
        text_dim=config.text_dim,
        aux_dim=config.duration_aux_dim,
        hidden_dim=config.duration_hidden_dim,
        layers=config.duration_layers,
        dropout=config.duration_dropout,
        speaker_dim=config.speaker_dim if config.use_speaker_condition_resolved else None,
        speaker_fusion=config.duration_speaker_fusion,
        caption_dim=config.caption_dim_resolved if config.use_caption_condition else None,
        caption_fusion=config.duration_caption_fusion,
        caption_pooling=config.duration_caption_pooling,
        attention_heads=config.duration_attention_heads,
        norm_eps=config.norm_eps,
        architecture=config.duration_architecture,
        token_init_frames=config.duration_token_init_frames,
    )
    predictor.load_state_dict(_sub_state(state, DURATION_PREFIX), strict=True)
    return predictor.eval()


def load_dit(
    source: IrodoriSource,
    state: Mapping[str, torch.Tensor],
    config: Any,
    text_config: Mapping[str, Any],
) -> nn.Module:
    """Build **the whole** `TextToLatentRFDiT` and load every tensor of the checkpoint.

    The DiT alone is not rebuilt partially because the golden's reference values are taken from the
    implementation's `forward_with_encoded_conditions` (= upstream's canonical path). The full
    `strict=True` load also serves as the check that "all 714 tensors were consumed, without
    exception".

    The backbone is built with `load_pretrained_backbone_weights=False` (neither a connection to HF
    nor random initialization happens — every weight put on it comes from the state_dict). **This
    target does not put the backbone in the graph** (`DitContextGraph` / `DitGraph` hold only
    subtrees of the DiT body), so the backbone built here does not become part of the reference
    either.
    """
    model = source.dit_cls(
        config,
        pretrained_backbone_config=dict(text_config),
        load_pretrained_backbone_weights=False,
    )
    model.load_state_dict(state, strict=True)
    return model.eval()


class BackboneGraph(nn.Module):
    """Export wrapper of `PretrainedTextBackbone.forward` (no mask as a run-time input).

    It differs from the original forward in 2 points, both **exact identities for an all-ones
    mask** (the MUST of the module docstring — `_check_wrapper_equivalence` measures it on the real
    weights every time):

    - the output's `state * mask` is dropped (`×1`)
    - a dict is passed as `attention_mask`, with the full_attention side set to `None` (an all-true
      band mask is all zeros in additive form, so the addition itself is the identity)
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        # While inv_freq stays a buffer it does not become a leaf of constant folding, and sin /
        # cos remain in the IR.
        assert_rope_lifted(backbone.backbone, "irodori text backbone")
        self.model = backbone.backbone

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        from transformers.masking_utils import create_bidirectional_sliding_window_mask

        model = self.model
        # Building the band mask needs only shape / dtype / device (the transformers docstring).
        # To avoid materializing anything, zeros of hidden width 1 are passed (they become a leaf
        # of constant folding).
        metadata = torch.zeros(input_ids.shape[0], input_ids.shape[1], 1, dtype=torch.float32)
        masks = {
            "full_attention": None,
            "sliding_attention": create_bidirectional_sliding_window_mask(
                config=model.config, inputs_embeds=metadata, attention_mask=None
            ),
        }
        return model(input_ids=input_ids, attention_mask=masks).last_hidden_state


class ProjectorGraph(nn.Module):
    """Export wrapper of `PretrainedConditionProjector.forward` (without the backbone).

    It differs from the original forward in 2 points, both **exact identities for an all-ones
    mask**:

    - the backbone call is moved out (`state` is taken as an argument)
    - the output's `projected * mask` is dropped (`×1`)

    Projector types other than `residual_mlp` do not occur in Irodori v4-Small, so they are not
    accepted (`export_series` checks the config and fails loudly).
    """

    def __init__(self, projector: nn.Module) -> None:
        super().__init__()
        self.projector = projector

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        projector = self.projector
        projected = projector.projector(hidden)
        residual = projector.residual_norm(hidden)
        residual = torch.nn.functional.silu(projector.residual_up(residual))
        return projected + projector.residual_down(residual)


class CaptionProjectorGraph(nn.Module):
    """Export wrapper of the caption-side projector (**2 outputs** — the G1b section of the module
    docstring).

    The first output is {@link ProjectorGraph} itself (the raw projector output), and the second
    output is that sequence with `caption_norm` (RMSNorm 512) applied. It is added only so that the
    host can take the masked mean for the `caption_vec` that `duration` needs.

    MUST: the first output is computed by **using `ProjectorGraph` as is** (the formula is not
    copied) — a copy could silently introduce a difference from the text side, and the bit equality
    of the first output (the match against the old golden) would degrade into a check that "the same
    formula was written in 2 places".
    """

    def __init__(self, projector: nn.Module, caption_norm: nn.Module) -> None:
        super().__init__()
        self.projection = ProjectorGraph(projector)
        self.caption_norm = caption_norm

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.projection(hidden)
        return projected, self.caption_norm(projected)


class SpeakerGraph(nn.Module):
    """Export wrapper of `ReferenceLatentEncoder.forward` + `speaker_norm`.

    It differs from the original 2 stages (`encode_conditions`) in 2 points, both **exact
    identities for an all-ones mask**:

    - `x * mask_f` is dropped in 3 places (right after in_proj, after each block, at the output)
      (`×1`)
    - the `key_mask` of `SelfAttention` is set to `None` (SDPA with an all-ones mask has an
      all-zero additive bias, so `scores + 0` is the identity)

    The RoPE table passed in is **a real-valued constant baked at Tmax (= the declared upper bound
    of S)** (`real_pair_rope_table` of `irodori.patch`). The `freqs_cis[:seq_len]` on the
    `SelfAttention` side becomes a symbolic prefix slice, and constant folding follows the ADR 0010
    path as is (treated the same as the backbone's band mask / RoPE table). It is **not promoted to
    a graph input** because the table is only 384KB and there is no reason to break "complete within
    one model file" (alternative 2 of ADR 0010).
    """

    def __init__(self, encoder: nn.Module, speaker_norm: nn.Module, sym_max: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.speaker_norm = speaker_norm
        # Made a plain attribute (a lifted tensor constant). A buffer / parameter does not become
        # a leaf of constant folding, and cos / sin would remain in the IR as is (the same reason
        # as karume.rope.lift_rope_buffers).
        self.rope_table = patch.real_pair_rope_table(encoder.head_dim, sym_max)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        x = encoder.in_proj(latent)
        x = x / 6.0
        for block in encoder.blocks:
            x = block(x, mask=None, freqs_cis=self.rope_table)
        return self.speaker_norm(x)


class DurationGraph(nn.Module):
    """Export wrapper of `text_norm` + `DurationPredictor` (token-sum form).

    It differs from the original `predict_duration_log_frames` in 3 points, all exact identities
    under the input contract (B=1, an all-ones mask, the text has at least 1 token):

    - `_safe_attention_mask` is dropped (with an all-ones mask it is the identity that returns its
      input as is)
    - `token_frames * text_mask` is dropped (`×1`)
    - `speaker_state[:,0]` / the caption's masked mean are taken as **results the host cut out**
      (the `null_*` selection stays in the graph together with the bool inputs `has_*` — the module
      docstring)

    `aux_features` are never read by the token-sum form, so they are not taken
    (`_duration_aux_is_inert` measures every time that they are "not read").
    """

    def __init__(self, predictor: nn.Module, text_norm: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor
        self.text_norm = text_norm

    def forward(
        self,
        text_state: torch.Tensor,
        speaker_vec: torch.Tensor,
        has_speaker: torch.Tensor,
        caption_vec: torch.Tensor,
        has_caption: torch.Tensor,
    ) -> torch.Tensor:
        predictor = self.predictor
        speaker = torch.where(has_speaker, speaker_vec, predictor.null_speaker)
        caption = torch.where(has_caption, caption_vec, predictor.null_caption)
        hidden = predictor.token_input_proj(self.text_norm(text_state))
        for block in predictor.token_blocks:
            hidden = block(hidden, cond=speaker, caption_cond=caption)
        logits = predictor.token_out_proj(predictor.token_out_norm(hidden)).squeeze(-1)
        frames = torch.nn.functional.softplus(logits.float())
        return torch.log1p(frames.sum(dim=1).clamp_min(0.0))


#: The children of `JointAttention` used **only by the condition-side K/V projections** (spelled as
#: in upstream's `project_context_kv`). `dit-context` holds these and {@link SHARED_KEY_NORM}, and
#: `dit` holds **all other children** ({@link DitContextGraph} / {@link DitGraph}) — the `dit` side
#: is defined as the complement, so a child that upstream adds to attention does not slip out of
#: `dit`'s ownership. A rename fails in `getattr`.
CONTEXT_KV_PROJECTIONS: tuple[str, ...] = (
    "wk_text",
    "wv_text",
    "wk_speaker",
    "wv_speaker",
    "wk_caption",
    "wv_caption",
)

#: The q/k norm applied to both the self-side K and the condition-side K (upstream's
#: `JointAttention.k_norm`). Both wrappers hold it, and it appears with the same values in both
#: containers.
SHARED_KEY_NORM = "k_norm"


def _submodule_view(children: Mapping[str, nn.Module]) -> nn.Module:
    """A container that only holds children of the real module **under the same attribute names**
    (it has no forward).

    Splitting the DiT into 2 graphs divides the weights of one `DiffusionBlock` / `JointAttention`
    between 2 wrappers. Holding the real module as is would make both wrappers own all the weights,
    breaking "the wrapper's owned parameters = the container's initializers" (the remapping of the
    scale ledger and the storage specification, the read-back of the i4 series and the i4
    eligibility check all rely on it) — the emit rejects a storage specification for an owned but
    unconsumed weight as an unknown key, and the read-back fails with "not in the container".

    The container only registers **the same module objects**, so the FQNs
    (`blocks.<b>.attention.wk_text.weight`) and the Parameter objects are the same as before the
    split. The computation is done by the real module's methods as is (torch.export's weight
    substitution acts on the attributes of the module objects, so going through the container or
    the real module points at the same slot).
    """
    view = nn.Module()
    for name, child in children.items():
        view.add_module(name, child)
    return view


class DitContextGraph(nn.Module):
    """Export wrapper of the condition-side K/V projections (`dit-context`, once per generation —
    ADR 0114).

    The inputs are the 3 condition states (the raw projector outputs right-padded to Tmax — the 4th
    to 6th inputs of `dit` before the split), and the outputs are the condition-side K / V of each
    block (the first-stage concatenation), 2 × the block count of them, in the order of
    `irodori.distribution.irodori_context_kv_names`.

    Correspondence with upstream: `text_norm` / `caption_norm` are applied (at the same position as
    in `encode_conditions` — `rms_norm` is a per-row reduction, so the leading rows are equivalent
    before and after padding and the pad rows stay 0), and each block's
    `JointAttention.project_context_kv` is **called as is** (the same way upstream's
    `build_context_kv_cache` calls it). The only thing copied is the order of the first-stage
    concatenation, which has the same elements in the same order as the text-onward part of
    `torch.cat(context_k, dim=1)` in upstream's `JointAttention.forward` (`cat` is associative).

    It owns only the 6 condition-side projections × blocks, the shared `k_norm`, and `text_norm` /
    `caption_norm` ({@link _submodule_view}).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.text_norm = model.text_norm
        self.caption_norm = model.caption_norm
        self.blocks = nn.ModuleList(
            _submodule_view(
                {
                    "attention": _submodule_view(
                        {
                            name: getattr(block.attention, name)
                            for name in (*CONTEXT_KV_PROJECTIONS, SHARED_KEY_NORM)
                        }
                    )
                }
            )
            for block in model.blocks
        )
        # The targets of the computation (the real modules' `JointAttention`). A tuple is not
        # registered by nn.Module, so the owned parameters sit only in the containers above.
        self.attentions: tuple[nn.Module, ...] = tuple(block.attention for block in model.blocks)

    @property
    def output_names(self) -> tuple[str, ...]:
        """The IR output names (the spelling {@link name_boundary} renames to after the export)."""
        return irodori_context_kv_names(len(self.attentions))

    def forward(
        self,
        text_state: torch.Tensor,
        speaker_state: torch.Tensor,
        caption_state: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        text = self.text_norm(text_state)
        caption = self.caption_norm(caption_state)
        outputs: list[torch.Tensor] = []
        for attention in self.attentions:
            outputs.extend(self.block_context(attention, text, speaker_state, caption))
        return tuple(outputs)

    @staticmethod
    def block_context(
        attention: nn.Module,
        text: torch.Tensor,
        speaker: torch.Tensor,
        caption: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The condition-side `(K, V)` of one block (`project_context_kv` + the first-stage
        concatenation).

        The conditions it takes are already normed (after `text_norm` / `caption_norm`).

        The stage decomposition of `irodori.measure_quant` calls this too (a calibration stage
        includes the condition-side projections inside the block, so the same single
        implementation is used instead of adding a copy).
        """
        projected = attention.project_context_kv(
            text_context=text, speaker_context=speaker, caption_context=caption
        )
        if len(projected) != 6:
            raise AssertionError(
                f"The context KV has {len(projected)} tensors (3 conditions text / speaker /"
                " caption = 6) — all 3 conditions should be active with these weights, so which"
                " conditions are present disagrees with the config"
            )
        k_text, v_text, k_speaker, v_speaker, k_caption, v_caption = projected
        return (
            torch.cat([k_text, k_speaker, k_caption], dim=1),
            torch.cat([v_text, v_speaker, v_caption], dim=1),
        )


def _dit_block_view(block: nn.Module) -> nn.Module:
    """The container of one block owned by `dit` (its attention holds only the children other than
    the condition-side projections).

    The order of the children stays as in the real module (the position of `attention` does not
    change either) — keeping the order of the owned parameters the same as before the split.
    """
    return _submodule_view(
        {
            name: (
                _submodule_view(
                    {
                        key: grandchild
                        for key, grandchild in child.named_children()
                        if key not in CONTEXT_KV_PROJECTIONS
                    }
                )
                if name == "attention"
                else child
            )
            for name, child in block.named_children()
        }
    )


class DitGraph(nn.Module):
    """Export wrapper of one DiT step (`forward_with_encoded_conditions`; the condition-side K/V are
    inputs).

    It differs from the original `forward_with_encoded_conditions` in 4 points, all exact
    identities under the input contract (B=1, conditions right-padded to Tmax, the mask
    concatenated in the order self / text / speaker / caption):

    - `t_embed` is **taken as an argument** (`get_timestep_embedding` runs on the host — `cos` is
      not in the vocabulary)
    - the condition-side K/V are **taken as arguments** (the outputs of {@link DitContextGraph}
      themselves — the same values as upstream's `context_kv_cache`; ADR 0114)
    - **the concatenation of `JointAttention`'s 4 masks is taken as one argument** (the same thing
      as upstream's `torch.cat(context_masks, dim=1)[:, None, None, :]`). The K/V concatenation is
      a **symbolic-axis cat** (`S+1519` — ADR 0046), and the 3 condition-side tensors are folded
      into the static-axis cat on the `dit-context` side (`cat` is associative, so either grouping
      lays out the same elements in the same order)
    - `nn.Dropout(p=0.0)` is dropped (an exact identity in eval)

    RoPE and the q/k norms call the real module's methods as is, so all this wrapper copies is "the
    order of the concatenation" and "how the residuals are assembled". The inputs are `x_t` /
    `t_embed` / `mask` followed by the condition-side K/V, 2 × the block count of them (**variadic
    arguments** — the block count is not baked in), and the IR input names are renamed to
    {@link input_names} after the export.

    Like `speaker`, the RoPE table is **a real-valued constant baked at Smax**
    (`real_pair_rope_table` of `irodori.patch`), and the symbolic prefix slice `[:seq_len]` follows
    the ADR 0010 path.

    It owns the subtree minus the condition-side projections ({@link _submodule_view}), and holds no
    `text_norm` / `caption_norm` (they moved to `dit-context`).
    """

    def __init__(self, model: nn.Module, sym_max: int) -> None:
        super().__init__()
        self.cond_module = model.cond_module
        self.in_proj = model.in_proj
        self.blocks = nn.ModuleList(_dit_block_view(block) for block in model.blocks)
        self.out_norm = model.out_norm
        self.out_proj = model.out_proj
        # Made a plain attribute (a lifted tensor constant) — the same reason as SpeakerGraph.
        self.rope_table = patch.real_pair_rope_table(model.head_dim, sym_max)
        # The targets of the computation (the real modules' `DiffusionBlock`). A tuple is not
        # registered by nn.Module, so the owned parameters sit only in the containers above.
        self.source_blocks: tuple[nn.Module, ...] = tuple(model.blocks)

    @property
    def input_names(self) -> tuple[str, ...]:
        """The IR input names (the spelling {@link name_boundary} renames to after the export)."""
        return (
            *IRODORI_DIT_HEAD_INPUTS,
            *irodori_context_kv_names(len(self.source_blocks)),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t_embed: torch.Tensor,
        mask: torch.Tensor,
        *context: torch.Tensor,
    ) -> torch.Tensor:
        return self._denoise(x_t, self.cond_module(t_embed), mask, context)

    def _denoise(
        self,
        x_t: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        context: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """The step after the condition embedding (`cond_embed` before its sequence axis is
        added) — shared by both flow parameterizations, which differ only in how `cond` is made
        ({@link MeanFlowDitGraph})."""
        if len(context) != 2 * len(self.source_blocks):
            raise ValueError(
                f"{len(context)} condition-side K/V tensors (should be 2 × the block count ="
                f" {2 * len(self.source_blocks)})"
            )
        cond_embed = cond[:, None, :]
        x = self.in_proj(x_t)
        # Cut to the same length as upstream's `_rope_freqs(x.shape[1])` (the
        # `freqs_cis[:seq_len]` on the `JointAttention` side becomes the identity at this point).
        freqs = self.rope_table[: x.shape[1]]
        for index, block in enumerate(self.source_blocks):
            h, attention_gate = block.attention_adaln(x, cond_embed)
            x = x + attention_gate * self._attention(
                block.attention, h, context[2 * index], context[2 * index + 1], mask, freqs
            )
            h, mlp_gate = block.mlp_adaln(x, cond_embed)
            x = x + mlp_gate * block.mlp(h)
        return self.out_proj(self.out_norm(x))

    @staticmethod
    def _attention(
        attention: nn.Module,
        x: torch.Tensor,
        context_k: torch.Tensor,
        context_v: torch.Tensor,
        mask: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """An equivalent implementation of `JointAttention.forward` (condition-side K/V already
        projected + an already concatenated mask)."""
        bsz, seq_len, _ = x.shape
        heads, head_dim = attention.heads, attention.head_dim
        q = attention.wq(x).reshape(bsz, seq_len, heads, head_dim)
        k_self = attention.wk(x).reshape(bsz, seq_len, heads, head_dim)
        v_self = attention.wv(x).reshape(bsz, seq_len, heads, head_dim)

        q = attention.q_norm(q)
        k_self = attention.k_norm(k_self)
        q = attention._apply_rotary_half(q, freqs)
        k_self = attention._apply_rotary_half(k_self, freqs)

        k = torch.cat([k_self, context_k], dim=1)
        v = torch.cat([v_self, context_v], dim=1)

        y = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, is_causal=False
        ).transpose(1, 2)
        y = y.reshape(bsz, seq_len, attention.dim)
        y = y * torch.sigmoid(attention.gate(x))
        return attention.wo(y)


class MeanFlowDitGraph(DitGraph):
    """Export wrapper of one DiT step of a **MeanFlow** checkpoint (ADR 0150).

    Upstream's MeanFlow DiT adds `delta_cond_module` (three bias-free linears, the same shape as
    `cond_module`) and makes the condition embedding
    `cond_module(t_embed) + delta_cond_module(delta_embed)`, where `delta_embed` is
    `get_timestep_embedding` of the interval length `t − t_next`
    (`TextToLatentRFDiT.forward_with_encoded_conditions`). Everything after that embedding is the
    same step as the RF one ({@link DitGraph._denoise}), so this differs from {@link DitGraph}
    only in one more input, `delta_embed` — built on the host for the same reason as `t_embed`
    (`cos` is not in the vocabulary) — and in owning `delta_cond_module`.

    The two embeddings are evaluated in upstream's order (`cond_module` first, then
    `delta_cond_module`, then the sum), so the eager equivalence check at atol 0 holds as for the
    RF step.
    """

    def __init__(self, model: nn.Module, sym_max: int) -> None:
        super().__init__(model, sym_max)
        if model.delta_cond_module is None:
            raise ValueError(
                "MeanFlowDitGraph needs a MeanFlow DiT (this one has no delta_cond_module)"
            )
        self.delta_cond_module = model.delta_cond_module

    @property
    def input_names(self) -> tuple[str, ...]:
        """The IR input names (the spelling {@link name_boundary} renames to after the export)."""
        return (
            *IRODORI_MEANFLOW_DIT_HEAD_INPUTS,
            *irodori_context_kv_names(len(self.source_blocks)),
        )

    def forward(  # type: ignore[override]  # one more positional input than the RF step
        self,
        x_t: torch.Tensor,
        t_embed: torch.Tensor,
        delta_embed: torch.Tensor,
        mask: torch.Tensor,
        *context: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.cond_module(t_embed) + self.delta_cond_module(delta_embed)
        return self._denoise(x_t, cond, mask, context)


class ComposedDitGraph(nn.Module):
    """`dit(<head inputs>, *dit_context(3 condition states))` — composed with the same inputs as
    before the split (`x_t` / `t_embed` / `mask` for the RF step, with `delta_embed` after
    `t_embed` for MeanFlow — {@link DitGraph.input_names}).

    The subject of the split's equivalence check; it is not exported ({@link export_series}
    matches it against upstream's unpatched `forward_with_encoded_conditions` at atol 0 — ADR
    0114).
    """

    def __init__(self, context: DitContextGraph, dit: DitGraph) -> None:
        super().__init__()
        self.context = context
        self.dit = dit

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """The head inputs of `dit` followed by the 3 condition states, in that order."""
        heads = len(self.dit.input_names) - 2 * len(self.dit.source_blocks)
        if len(inputs) != heads + len(IRODORI_CONTEXT_INPUTS):
            raise ValueError(
                f"{len(inputs)} inputs (should be the {heads} head inputs of dit + the"
                f" {len(IRODORI_CONTEXT_INPUTS)} condition states)"
            )
        return self.dit(*inputs[:heads], *self.context(*inputs[heads:]))


class DitWrappers(NamedTuple):
    """The 2 distribution wrappers of the DiT (the return value of {@link dit_wrappers})."""

    context: DitContextGraph
    dit: DitGraph

    def by_target(self) -> dict[str, nn.Module]:
        """Target name → wrapper (the form the writing, the remapping and the read-back look up)."""
        return {TARGET_DIT_CONTEXT: self.context, TARGET_DIT: self.dit}


def dit_wrappers(model: nn.Module, sym_max: int) -> DitWrappers:
    """Build the 2 distribution wrappers of the DiT.

    The union of the 2 sets of owned parameters is the full set of "weights that go into the DiT's
    distribution graphs", and the only overlap is `k_norm` ({@link DitContextGraph} /
    {@link DitGraph}). The i4 eligibility check (`irodori.calib.dit_i4_names`), the read-back of the
    i4 series (`irodori.pipeline_ref`) and the measurement rig (`irodori.measure_quant`) build from
    here as well — the pair of wrappers is decided in one place.

    The step wrapper follows the checkpoint's own flow parameterization: upstream builds
    `delta_cond_module` exactly when the config says `meanflow` (`TextToLatentRFDiT.__init__`), so
    its presence picks {@link MeanFlowDitGraph} (ADR 0150) — never the directory name.
    """
    meanflow = getattr(model, "delta_cond_module", None) is not None
    step = MeanFlowDitGraph(model, sym_max) if meanflow else DitGraph(model, sym_max)
    return DitWrappers(context=DitContextGraph(model), dit=step)


def dit_flow(model: nn.Module) -> str:
    """The flow parameterization of a loaded DiT (the same presence test as
    {@link dit_wrappers})."""
    return (
        IRODORI_FLOW_MEANFLOW
        if getattr(model, "delta_cond_module", None) is not None
        else IRODORI_FLOW_RF
    )


def golden_case_body(name: str, kind: str, body: str, normalize_text: Any) -> str:
    """Apply the **per-kind** preprocessing to a golden case's text (the same as upstream's
    `_synthesize`).

    MUST: `normalize_text` is **for text only**. Upstream's `inference_runtime._synthesize` applies
    only `str(...).strip()` to the caption, so adding normalization to the caption would silently
    change the conditioning by the stripped outer brackets, NFKC and removed symbols (for the same
    reason `_packed_caption_ids` in `irodori/pipeline_ref.py` applies only strip as well).
    """
    if kind == "text":
        prepared = normalize_text(body).strip()
    elif kind == "caption":
        prepared = body.strip()
    else:
        raise SystemExit(f"{name}: the kind {kind!r} is neither text nor caption")
    if not prepared:
        raise SystemExit(f"{name}: the text is empty after preprocessing")
    return prepared


def build_cases(
    model_dir: Path, source: IrodoriSource, text_config: Mapping[str, Any], sym_max: int
) -> tuple[tuple[str, str, torch.Tensor], ...]:
    """The golden cases `(name, kind, input_ids)`.

    The preprocessing follows the same order as the implementation's source of truth
    (`inference_runtime._synthesize`): per-kind preprocessing ({@link golden_case_body}) → tokenize
    without special tokens → prepend BOS. There is **no** right padding (in the static scheme the
    host packs the sequence — the module docstring).
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / TOKENIZER_FILE))
    bos_id = int(text_config["bos_token_id"])
    cases: list[tuple[str, str, torch.Tensor]] = []
    for name, kind, body in GOLDEN_CASES:
        prepared = golden_case_body(name, kind, body, source.normalize_text)
        ids = [bos_id, *tokenizer.encode(prepared, add_special_tokens=False).ids]
        if not 2 <= len(ids) <= sym_max:
            raise SystemExit(
                f"{name}: T={len(ids)} is outside the symbolic dimension's range [2, {sym_max}]"
            )
        cases.append((name, kind, torch.tensor([ids], dtype=torch.int64)))
    return tuple(cases)


def build_speaker_cases(latent_dim: int, sym_max: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """The **synthetic** golden cases of `speaker` `(name, reference latent)` (as in
    {@link SPEAKER_CASES}).

    The values are a standard normal with a deterministic seed (the table's note — these carry the
    length coverage).
    """
    cases: list[tuple[str, torch.Tensor]] = []
    for name, length, seed in SPEAKER_CASES:
        if not MIN_SYM_LENGTH <= length <= sym_max:
            raise SystemExit(
                f"{name}: S={length} is outside the symbolic dimension's range"
                f" [{MIN_SYM_LENGTH}, {sym_max}]"
            )
        generator = torch.Generator().manual_seed(seed)
        cases.append((name, torch.randn(1, length, latent_dim, generator=generator)))
    return tuple(cases)


def build_real_speaker_cases(
    patch_sequence: Any,
    latent_dim: int,
    patch_size: int,
    patched_dim: int,
    sym_max: int,
    *,
    latent_dir: Path = REFERENCE_LATENT_DIR,
) -> tuple[tuple[str, torch.Tensor], ...]:
    """The **real-latent** golden cases of `speaker` (as in the table {@link SPEAKER_REAL_CASES}).

    The source is the ready-made `[1,190,32]` that `irodori/dacvae/host.py` wrote. Patching **calls
    upstream's `patch_sequence_with_mask`** (passed in as `patch_sequence`) — copying how the
    remainder is truncated would let the script alone stay stale and pass silently when upstream
    changes.

    MUST: **fail** in an environment without the real latent (no synthetic substitute) — a
    substitute would silently remove the property these 2 carry, "the value range is that of real
    use", and break the grounds of the tolerances.
    """
    path = latent_dir / f"{REFERENCE_LATENT_PREFIX}{REFERENCE_LATENT_CASE}{IO_SUFFIX}"
    if not path.is_file():
        raise SystemExit(f"Real latent missing: {path} (created by `{REFERENCE_LATENT_COMMAND}`)")
    tensors = load_file(str(path))
    if REFERENCE_LATENT_KEY not in tensors:
        raise SystemExit(f"{path} has no '{REFERENCE_LATENT_KEY}'")
    latent = tensors[REFERENCE_LATENT_KEY].to(torch.float32)
    if latent.ndim != 3 or latent.shape[0] != 1 or int(latent.shape[2]) != latent_dim:
        raise SystemExit(f"The real latent's shape {tuple(latent.shape)} is not [1,S,{latent_dim}]")
    mask = torch.ones(latent.shape[:2], dtype=torch.bool)
    patched, _patched_mask = patch_sequence(latent, mask, patch_size)
    if int(patched.shape[2]) != patched_dim:
        raise SystemExit(
            f"The dimension after patching, {int(patched.shape[2])}, differs from speaker's input"
            f" dimension {patched_dim}"
        )
    available = int(patched.shape[1])

    cases: list[tuple[str, torch.Tensor]] = []
    for name, length in SPEAKER_REAL_CASES:
        want = available if length is None else int(length)
        if not MIN_SYM_LENGTH <= want <= sym_max:
            raise SystemExit(
                f"{name}: S={want} is outside the symbolic dimension's range"
                f" [{MIN_SYM_LENGTH}, {sym_max}]"
            )
        if want > available:
            raise SystemExit(
                f"{name}: S={want} exceeds the real latent's length after patching, {available}"
            )
        cases.append((name, patched[:, :want].contiguous()))
    return tuple(cases)


def _pristine_speaker_outputs(
    encoder: nn.Module,
    speaker_norm: nn.Module,
    cases: Sequence[tuple[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """The speaker-side eager outputs **before patching** (`[1,S,768]` with `speaker_norm`
    applied)."""
    if patch.patches_applied():
        raise AssertionError(
            "Tried to take a reference after the patches were applied (the equivalence check would"
            " become vacuous)"
        )
    outputs: dict[str, torch.Tensor] = {}
    for name, latent in cases:
        mask = torch.ones(latent.shape[:2], dtype=torch.bool)
        with torch.no_grad():
            outputs[name] = speaker_norm(encoder(latent, mask))
    return outputs


def _no_reference_evidence(encoder: nn.Module, latent_dim: int, patch_size: int) -> float:
    """Measure that the output of the no-reference (`no_ref`) path is **exactly 0**.

    `no_ref` is the only canonical path that builds `ref_mask` as **all zeros**
    (`_load_reference_latent`), and the speaker graph has no mask, so it cannot express this input.
    If the torch output is a zero matrix, the host is equivalent by just "putting zeros without
    calling the graph" — that claim is measured here (if it is not 0, the design decision to promote
    the mask to a run-time input is needed).

    MUST: do not let it become vacuous — pass a **nonzero** reference latent. With a zero input
    there is no telling whether "the mask works" or "0 came out because the input was zero".
    """
    length = max(MIN_SYM_LENGTH, patch_size)
    generator = torch.Generator().manual_seed(999)
    latent = torch.randn(1, length, latent_dim, generator=generator)
    with torch.no_grad():
        state = encoder(latent, torch.zeros((1, length), dtype=torch.bool))
    worst = float(state.abs().max())
    if worst != 0.0:
        raise AssertionError(
            f"The speaker output without a reference (all-zero mask) is not 0 (max abs {worst})"
            " — zeros supplied by the host would not be equivalent, so a run-time mask input has"
            " to be designed"
        )
    return worst


def _duration_cases(
    source: IrodoriSource,
    predictor: nn.Module,
    text_norm: nn.Module,
    caption_norm: nn.Module,
    text_proj: Mapping[str, torch.Tensor],
    caption_proj: Mapping[str, torch.Tensor],
    speaker: Mapping[str, torch.Tensor],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    """Build the graph inputs of `duration` and the full arguments **for calling the real
    module**.

    Returns `(graph inputs, kwargs of the reference call)`. The former are the 5 graph inputs
    themselves, the latter every argument of `DurationPredictor.forward` (masks and `aux_features`
    included), used to take the reference values.

    MUST: the speaker / caption vectors are built by **calling** the implementation's methods
    (`_prepend_masked_mean_token` / `_speaker_vec` / `_caption_vec`) — copying the formulas would
    let the script alone stay stale and pass silently when upstream changes.
    """
    missing = [name for _n, name, _s, _c in DURATION_CASES if name not in text_proj]
    if missing:
        raise SystemExit(f"The text cases {missing} that duration refers to are not in the golden")
    for label, table, key in (
        ("speaker", speaker, DURATION_SPEAKER_SOURCE),
        ("caption-proj", caption_proj, DURATION_CAPTION_SOURCE),
    ):
        if key not in table:
            raise SystemExit(
                f"The {label} case '{key}' that duration refers to is not in the golden"
            )

    truthy = torch.ones((1,), dtype=torch.bool)
    with torch.no_grad():
        speaker_state, speaker_mask = source.prepend_masked_mean_token(
            speaker[DURATION_SPEAKER_SOURCE],
            torch.ones(speaker[DURATION_SPEAKER_SOURCE].shape[:2], dtype=torch.bool),
        )
        caption_state = caption_norm(caption_proj[DURATION_CAPTION_SOURCE])
        caption_mask = torch.ones(caption_state.shape[:2], dtype=torch.bool)
        speaker_vec = predictor._speaker_vec(
            batch_size=1,
            device=speaker_state.device,
            dtype=speaker_state.dtype,
            speaker_state=speaker_state,
            has_speaker=truthy,
        )
        caption_vec = predictor._caption_vec(
            batch_size=1,
            device=caption_state.device,
            dtype=caption_state.dtype,
            caption_state=caption_state,
            caption_mask=caption_mask,
            has_caption=truthy,
        )

    inputs: dict[str, dict[str, torch.Tensor]] = {}
    reference: dict[str, dict[str, torch.Tensor]] = {}
    for name, text_case, has_speaker, has_caption in DURATION_CASES:
        state = text_proj[text_case]
        flags = {
            "has_speaker": torch.tensor([[has_speaker]], dtype=torch.bool),
            "has_caption": torch.tensor([[has_caption]], dtype=torch.bool),
        }
        inputs[name] = {
            "text_state": state,
            # For an absent condition **the host puts zeros** (the value is discarded by where —
            # the in-graph `null_*` is selected).
            "speaker_vec": speaker_vec if has_speaker else torch.zeros_like(speaker_vec),
            "has_speaker": flags["has_speaker"],
            "caption_vec": caption_vec if has_caption else torch.zeros_like(caption_vec),
            "has_caption": flags["has_caption"],
        }
        with torch.no_grad():
            normalized = text_norm(state)
        reference[name] = {
            "text_state": normalized,
            "text_mask": torch.ones(normalized.shape[:2], dtype=torch.bool),
            "speaker_state": speaker_state,
            "speaker_mask": speaker_mask,
            "has_speaker": torch.tensor([has_speaker], dtype=torch.bool),
            "caption_state": caption_state,
            "caption_mask": caption_mask,
            "has_caption": torch.tensor([has_caption], dtype=torch.bool),
        }
    return inputs, reference


def _call_duration(
    predictor: nn.Module, args: Mapping[str, torch.Tensor], aux: torch.Tensor
) -> torch.Tensor:
    """Call the real module's `DurationPredictor.forward` with every argument (where the reference
    values are taken)."""
    with torch.no_grad():
        return predictor(
            args["text_state"],
            text_mask=args["text_mask"],
            aux_features=aux,
            speaker_state=args["speaker_state"],
            speaker_mask=args["speaker_mask"],
            has_speaker=args["has_speaker"],
            caption_state=args["caption_state"],
            caption_mask=args["caption_mask"],
            has_caption=args["has_caption"],
        ).float()


def _pristine_duration_outputs(
    predictor: nn.Module,
    reference: Mapping[str, Mapping[str, torch.Tensor]],
    aux_dim: int,
) -> dict[str, torch.Tensor]:
    """The duration outputs **before patching** (log frames, `[1]`)."""
    if patch.patches_applied():
        raise AssertionError(
            "Tried to take a reference after the patches were applied (the equivalence check would"
            " become vacuous)"
        )
    aux = torch.zeros((1, aux_dim), dtype=torch.float32)
    return {name: _call_duration(predictor, args, aux) for name, args in reference.items()}


def _duration_aux_is_inert(
    predictor: nn.Module,
    reference: Mapping[str, Mapping[str, torch.Tensor]],
    aux_dim: int,
    pristine: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Measure that `aux_features` are **never read** in the token-sum form.

    This is the only ground for dropping the 14-dimensional input from the graph, so "it should not
    be read" is not left as a claim. It calls again with another aux (all ones) and fails unless
    the result is **bit-identical**.
    """
    alternative = torch.ones((1, aux_dim), dtype=torch.float32)
    evidence: dict[str, float] = {}
    for name, args in reference.items():
        diff = float((_call_duration(predictor, args, alternative) - pristine[name]).abs().max())
        if diff != 0.0:
            raise AssertionError(
                f"{name}: changing aux_features moved the duration output by {diff}"
                " — the token-sum form reads aux (it cannot be dropped from the graph inputs)"
            )
        evidence[name] = diff
    return evidence


def _right_pad(state: torch.Tensor, length: int, where: str) -> torch.Tensor:
    """**Right-pad** `[1,T,D]` with zeros to `[1,length,D]` (left on the host by ADR 0047)."""
    used = int(state.shape[1])
    if used > length:
        raise SystemExit(
            f"{where}: the length {used} exceeds the condition's declared length {length}"
        )
    padded = state.new_zeros((int(state.shape[0]), length, int(state.shape[2])))
    padded[:, :used] = state
    return padded


def _segment_mask(length: int, used: int) -> torch.Tensor:
    """A `[1,length]` bool mask (the first `used` are True). `used=0` is an uncond segment."""
    mask = torch.zeros((1, length), dtype=torch.bool)
    mask[0, :used] = True
    return mask


def _dit_cases(
    source: IrodoriSource,
    config: Any,
    model_config: Mapping[str, Any],
    text_norm: nn.Module,
    caption_norm: nn.Module,
    speaker_max: int,
    sym_max: int,
    text_proj: Mapping[str, torch.Tensor],
    caption_proj: Mapping[str, torch.Tensor],
    speaker: Mapping[str, torch.Tensor],
    *,
    flow: str = IRODORI_FLOW_RF,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    """Build the DiT inputs **in condition-state form** and the full arguments **for calling the
    real module**.

    `flow` is the checkpoint's flow parameterization ({@link dit_flow}): a MeanFlow case also
    carries its interval — `delta_embed` (built by the same upstream function as `t_embed`) on the
    graph side, `delta_t` on the reference side ({@link dit_cases}).

    Returns `(inputs in condition-state form, kwargs of the reference call)`. The former are the 6
    inputs of `dit` before the split (`x_t` / `t_embed` / `mask` + the 3 condition states), the
    inputs of the split's equivalence check ({@link ComposedDitGraph}) and the source of the graph
    inputs of `dit-context` / `dit` ({@link _dit_context_cases} / {@link _dit_graph_inputs}). The
    latter are the arguments of `forward_with_encoded_conditions` (the **normed** condition states
    and the segment masks), used to take the reference values.

    MUST: the condition states are chained from the torch expected values of the upstream targets,
    and the mean-token prepending and `t_embed` are built by **calling** the implementation's
    functions (the formulas are not copied).

    MUST: the reference of an uncond variant is taken the same way as upstream, "the state set to
    0 and the mask set to 0 too". The graph side keeps the cond state and only zeroes the mask, so
    the two agreeing is the proof of ADR 0047 decision 1 (not vacuous — `_dit_uncond_divergence`
    measures the mutual differences of the 4 outputs).
    """
    for label, table, key in (
        ("text-proj", text_proj, DIT_TEXT_SOURCE),
        ("caption-proj", caption_proj, DIT_CAPTION_SOURCE),
        ("speaker", speaker, DIT_SPEAKER_SOURCE),
    ):
        if key not in table:
            raise SystemExit(f"The {label} case '{key}' that dit refers to is not in the golden")

    with torch.no_grad():
        speaker_packed, _speaker_mask = source.prepend_masked_mean_token(
            speaker[DIT_SPEAKER_SOURCE],
            torch.ones(speaker[DIT_SPEAKER_SOURCE].shape[:2], dtype=torch.bool),
        )
    packed = {
        "text": text_proj[DIT_TEXT_SOURCE],
        # The length after prepending the mean token (the `speaker` graph's output + 1).
        "speaker": speaker_packed,
        "caption": caption_proj[DIT_CAPTION_SOURCE],
    }
    caps = {
        "text": int(model_config["max_text_len"]),
        "speaker": speaker_max + 1,
        "caption": int(model_config["max_caption_len"]),
    }
    padded = {
        name: _right_pad(state, caps[name], f"the {name} condition of dit")
        for name, state in packed.items()
    }
    with torch.no_grad():
        # The reference consumes normed states (`encode_conditions` applies the norms before
        # handing them to the DiT). The graph takes the raw states and applies the same norms
        # inside.
        normed = {
            "text": text_norm(padded["text"]),
            "speaker": padded["speaker"],
            "caption": caption_norm(padded["caption"]),
        }

    latent_dim = int(config.patched_latent_dim)
    embed_dim = int(config.timestep_embed_dim)
    inputs: dict[str, dict[str, torch.Tensor]] = {}
    reference: dict[str, dict[str, torch.Tensor]] = {}
    for name, length, seed, t_value, uncond, delta in dit_cases(flow):
        if uncond is not None and uncond not in DIT_UNCOND_VARIANTS:
            raise SystemExit(
                f"{name}: the uncond segment {uncond!r} is not in {DIT_UNCOND_VARIANTS}"
            )
        if not MIN_SYM_LENGTH <= length <= sym_max:
            raise SystemExit(
                f"{name}: S={length} is outside the symbolic dimension's range"
                f" [{MIN_SYM_LENGTH}, {sym_max}]"
            )
        generator = torch.Generator().manual_seed(seed)
        x_t = torch.randn(1, length, latent_dim, generator=generator)
        t = torch.tensor([t_value], dtype=torch.float32)
        delta_t = None if delta is None else torch.tensor([delta], dtype=torch.float32)
        with torch.no_grad():
            t_embed = source.timestep_embedding(t, embed_dim).to(dtype=x_t.dtype)
            interval = (
                {}
                if delta_t is None
                else {
                    "delta_embed": source.timestep_embedding(delta_t, embed_dim).to(dtype=x_t.dtype)
                }
            )
        masks = {
            segment: _segment_mask(
                caps[segment], 0 if uncond == segment else int(packed[segment].shape[1])
            )
            for segment in DIT_UNCOND_VARIANTS
        }
        # Exactly upstream's `torch.cat(context_masks, dim=1)[:, None, None, :]` in
        # `JointAttention.forward` (the self mask is all ones on every inference path — recon §1).
        # **The order is taken explicitly from `DIT_UNCOND_VARIANTS`** — the graph side
        # concatenates the K/V in this order, so relying implicitly on the dict's insertion order
        # would let a swap pass quietly.
        mask = torch.cat(
            [
                torch.ones((1, length), dtype=torch.bool),
                *(masks[segment] for segment in DIT_UNCOND_VARIANTS),
            ],
            dim=1,
        )[:, None, None, :]
        inputs[name] = {
            "x_t": x_t,
            "t_embed": t_embed,
            **interval,
            "mask": mask,
            "text_state": padded["text"],
            "speaker_state": padded["speaker"],
            "caption_state": padded["caption"],
        }
        reference[name] = {
            "x_t": x_t,
            "t": t,
            **({} if delta_t is None else {"delta_t": delta_t}),
            **{
                f"{segment}_state": (
                    torch.zeros_like(normed[segment]) if uncond == segment else normed[segment]
                )
                for segment in DIT_UNCOND_VARIANTS
            },
            **{f"{segment}_mask": value for segment, value in masks.items()},
        }
    return inputs, reference


def _call_dit(model: nn.Module, args: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Call the real module's `forward_with_encoded_conditions` (where the reference values are
    taken).

    `latent_mask=None` is upstream's inference path as is (`JointAttention` builds all ones).
    `delta_t` is present only in a MeanFlow case ({@link _dit_cases}); upstream refuses it for an
    RF checkpoint and requires it for a MeanFlow one, so passing it through as is lets upstream
    itself reject a mix-up.
    """
    with torch.no_grad():
        return model.forward_with_encoded_conditions(
            x_t=args["x_t"],
            t=args["t"],
            delta_t=args.get("delta_t"),
            text_state=args["text_state"],
            text_mask=args["text_mask"],
            speaker_state=args["speaker_state"],
            speaker_mask=args["speaker_mask"],
            caption_state=args["caption_state"],
            caption_mask=args["caption_mask"],
            latent_mask=None,
        )


def _pristine_dit_outputs(
    model: nn.Module, reference: Mapping[str, Mapping[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """The DiT outputs **before patching** (v_pred, `[1,S,32]`)."""
    if patch.patches_applied():
        raise AssertionError(
            "Tried to take a reference after the patches were applied (the equivalence check would"
            " become vacuous)"
        )
    return {name: _call_dit(model, args) for name, args in reference.items()}


def _dit_context_cases(
    dit_inputs: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, str]]:
    """The cases of `dit-context` (one per set of condition states), and DiT case → its
    representative's name.

    The inputs of `dit-context` are only the 3 condition states, so DiT cases with the same states
    produce the same K/V (the 3 uncond variants keep the cond states and differ only in the mask —
    ADR 0047 decision 1). Sameness of a set is **judged by value** (no assumption about how the case
    table is built), so if a case with other states is added to the table, it only adds one more
    representative. The representative's name is that of the first DiT case to appear in the set.
    """
    groups: dict[str, dict[str, torch.Tensor]] = {}
    owner: dict[str, str] = {}
    for name, args in dit_inputs.items():
        states = {key: args[key] for key in IRODORI_CONTEXT_INPUTS}
        found = next(
            (
                representative
                for representative, kept in groups.items()
                if all(torch.equal(kept[key], states[key]) for key in IRODORI_CONTEXT_INPUTS)
            ),
            None,
        )
        if found is None:
            groups[name] = states
            found = name
        owner[name] = found
    return groups, owner


def _pristine_dit_context_outputs(
    model: nn.Module, cases: Mapping[str, Mapping[str, torch.Tensor]]
) -> dict[str, tuple[torch.Tensor, ...]]:
    """The condition-side K/V **before patching** (the golden expected values of `dit-context` — a
    set of 2 × the block count).

    They are taken through upstream's canonical path itself: `build_context_kv_cache` (= each
    block's `project_context_kv` — the cache upstream uses with `use_context_kv_cache=True`) is
    called with the conditions after the DiT's own `text_norm` / `caption_norm`, and each block's
    text / speaker / caption are laid out in the concatenation order of upstream's
    `JointAttention.forward`. The output of the wrapper ({@link DitContextGraph}) is matched
    against this at atol 0 after patching — using the wrapper's own output as the expected value
    would be vacuous.
    """
    if patch.patches_applied():
        raise AssertionError(
            "Tried to take a reference after the patches were applied (the equivalence check would"
            " become vacuous)"
        )
    outputs: dict[str, tuple[torch.Tensor, ...]] = {}
    for name, states in cases.items():
        with torch.no_grad():
            cache = model.build_context_kv_cache(
                text_state=model.text_norm(states["text_state"]),
                speaker_state=states["speaker_state"],
                caption_state=model.caption_norm(states["caption_state"]),
            )
        flat: list[torch.Tensor] = []
        for projected in cache:
            if len(projected) != 6:
                raise AssertionError(
                    f"Upstream's context KV cache has {len(projected)} tensors (should be 3"
                    " conditions = 6)"
                )
            k_text, v_text, k_speaker, v_speaker, k_caption, v_caption = projected
            flat.append(torch.cat([k_text, k_speaker, k_caption], dim=1))
            flat.append(torch.cat([v_text, v_speaker, v_caption], dim=1))
        outputs[name] = tuple(flat)
    return outputs


def _dit_graph_inputs(
    dit_inputs: Mapping[str, Mapping[str, torch.Tensor]],
    owner: Mapping[str, str],
    context: Mapping[str, Sequence[torch.Tensor]],
    names: Sequence[str],
    heads: Sequence[str] = IRODORI_DIT_HEAD_INPUTS,
) -> dict[str, dict[str, torch.Tensor]]:
    """The graph inputs of `dit` (the head inputs `heads` — `x_t` / `t_embed` / `mask`, with
    `delta_embed` after `t_embed` for MeanFlow — + the condition-side K/V under their boundary
    names).

    The K/V are **the unpatched expected values** of `dit-context` themselves (a chain — the torch
    expected values of the upstream target become the downstream inputs, the same form as
    `duration`). The order is the same as the IR's input order (the forward's argument order) — the
    equivalence check passes the values of this dict by position.
    """
    return {
        name: {
            **{key: args[key] for key in heads},
            **dict(zip(names, context[owner[name]], strict=True)),
        }
        for name, args in dit_inputs.items()
    }


def _dit_uncond_divergence(
    pristine: Mapping[str, torch.Tensor], cases: Sequence[DitCase] | None = None
) -> dict[str, float]:
    """Measure that the outputs of the 4 that differ only in the mask (cond + the 3 uncond
    variants) differ **from each other**.

    MUST: do not let it become vacuous — even if the segment layout (the order and offsets of self
    S / text / speaker / caption) breaks, the golden still matches when the wrapper and the
    reference break the same way. Fails when the minimum pairwise difference drops below the floor.

    `cases` defaults to the RF cases ({@link dit_cases}); a MeanFlow checkpoint passes its own, and
    the interval is part of what the group shares (only the mask may differ).
    """
    table = dit_cases(IRODORI_FLOW_RF) if cases is None else cases
    keys = {
        (case.length, case.seed, case.t, case.delta) for case in table if case.uncond is not None
    }
    if len(keys) != 1:
        raise AssertionError(
            f"The uncond variants do not share x_t / t ({sorted(keys)}) — the difference would not"
            " be the mask alone"
        )
    key = keys.pop()
    group = [case.name for case in table if (case.length, case.seed, case.t, case.delta) == key]
    if len(group) != 1 + len(DIT_UNCOND_VARIANTS):
        raise AssertionError(
            f"The set of cond + the 3 uncond variants is {group} — a cond case is needed as the"
            " baseline of the comparison"
        )
    pairs = {
        f"{lhs} vs {rhs}": float((pristine[lhs] - pristine[rhs]).abs().max())
        for index, lhs in enumerate(group)
        for rhs in group[index + 1 :]
    }
    worst = min(pairs.values())
    if worst < DIT_UNCOND_DIVERGENCE_MIN:
        raise AssertionError(
            f"The minimum output difference between cond / uncond is {worst} — the segment layout"
            " of the mask is suspected not to work"
        )
    return {name: round(value, 5) for name, value in pairs.items()}


def _dit_delta_divergence(pristine: Mapping[str, torch.Tensor]) -> float:
    """Measure that the interval of a MeanFlow step moves the output ({@link
    MEANFLOW_DIT_DELTA_PAIR} — the same `x_t` / t / conditions / mask, another `delta`).

    MUST: do not let it become vacuous (the reason at {@link MEANFLOW_DIT_EXTRA_CASES}). Fails
    when the difference drops below {@link DIT_DELTA_DIVERGENCE_MIN}.
    """
    lhs, rhs = MEANFLOW_DIT_DELTA_PAIR
    difference = float((pristine[lhs] - pristine[rhs]).abs().max())
    if difference < DIT_DELTA_DIVERGENCE_MIN:
        raise AssertionError(
            f"The output difference between {lhs} and {rhs} (only the interval differs) is"
            f" {difference} — the interval embedding seems not to reach the output"
        )
    return round(difference, 5)


def _single(outputs: Mapping[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, ...]]:
    """Turn the expected values of a 1-output target into positional tuples (the form `_write_io`
    consumes)."""
    return {name: (value,) for name, value in outputs.items()}


def _first(outputs: Mapping[str, tuple[torch.Tensor, ...]]) -> dict[str, torch.Tensor]:
    """Take **only output 0** from the tuples of expected values.

    The downstream of the chain (`duration` / `dit` / the static-scheme measurement / the projector
    mix-up check) always consumes the first output; the second output of `caption-proj` (the
    sequence with `caption_norm` applied) exists only for the host to take the masked mean.
    """
    return {name: value[0] for name, value in outputs.items()}


def _pristine_outputs(
    backbone: nn.Module,
    projectors: Mapping[str, nn.Module],
    caption_norm: nn.Module,
    cases: Sequence[tuple[str, str, torch.Tensor]],
) -> dict[str, dict[str, tuple[torch.Tensor, ...]]]:
    """Take the eager outputs **before patching** (the golden expected values themselves) per
    target.

    MUST: call it before `irodori.patch` is applied (after, the equivalence check would become
    vacuous). The real modules' forward runs as is (all-ones mask) — a copying mistake in a wrapper
    shows up in `_check_wrapper_equivalence` as a difference from the values taken here.

    Only `caption-proj` has 2 expected values (the second output = the sequence with `caption_norm`
    applied).
    """
    if patch.patches_applied():
        raise AssertionError(
            "Tried to take a reference after the patches were applied (the equivalence check would"
            " become vacuous)"
        )
    outputs: dict[str, dict[str, tuple[torch.Tensor, ...]]] = {
        target: {} for target in TEXT_TARGETS
    }
    for name, _kind, ids in cases:
        mask = torch.ones_like(ids, dtype=torch.bool)
        with torch.no_grad():
            outputs[TARGET_BACKBONE][name] = (backbone(ids, mask),)
            for target, projector in projectors.items():
                projected = projector(backbone, ids, mask)
                outputs[target][name] = (
                    (projected, caption_norm(projected))
                    if target == TARGET_CAPTION_PROJ
                    else (projected,)
                )
    return outputs


def _static_scheme_evidence(
    backbone: nn.Module,
    text_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    cases: Sequence[tuple[str, str, torch.Tensor]],
    pristine: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Measure that "right padding + mask" and "the packed sequence" are equivalent (the ground of
    the static scheme).

    Irodori's inference calls the backbone right-padded to a fixed length (text 256 / caption 512).
    The static scheme **removes that padding on the host side**, so if the two did not agree the
    scheme itself would not hold. To **settle it by measurement rather than by working around it**,
    this is a check that always runs.
    """
    pad_id = int(text_config["pad_token_id"])
    caps = {
        "text": int(model_config["max_text_len"]),
        "caption": int(model_config["max_caption_len"]),
    }
    evidence: dict[str, float] = {}
    for name, kind, ids in cases:
        length = int(ids.shape[1])
        padded_len = caps[kind]
        if length > padded_len:
            raise SystemExit(f"{name}: T={length} exceeds the {kind} upper bound {padded_len}")
        padded_ids = torch.full((1, padded_len), pad_id, dtype=torch.int64)
        padded_ids[0, :length] = ids[0]
        padded_mask = torch.zeros((1, padded_len), dtype=torch.bool)
        padded_mask[0, :length] = True
        with torch.no_grad():
            padded_state = backbone(padded_ids, padded_mask)
        diff = float((padded_state[:, :length] - pristine[name]).abs().max())
        if diff > STATIC_SCHEME_ATOL:
            raise AssertionError(
                f"{name}: the first {length} rows of the padded call differ from the packed"
                f" sequence by {diff}, more than {STATIC_SCHEME_ATOL} — the static scheme (no"
                " run-time attention_mask) does not hold for this model"
            )
        evidence[name] = diff
    return evidence


def _check_wrapper_equivalence(
    wrapper: nn.Module,
    args: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
    where: str,
    atol: float,
) -> float:
    """Check that a wrapper's output matches the unpatched real module's output (multi-output
    aware)."""
    with torch.no_grad():
        actual = wrapper(*args)
    outputs = actual if isinstance(actual, tuple) else (actual,)
    if len(outputs) != len(expected):
        raise AssertionError(
            f"{where}: the wrapper has {len(outputs)} outputs, the expected values {len(expected)}"
        )
    diff = max(
        float((value - want).abs().max()) for value, want in zip(outputs, expected, strict=True)
    )
    if diff > atol:
        raise AssertionError(f"{where}: eager equivalence broke (max abs diff {diff} > {atol})")
    return diff


def _write_io(
    graph: IrGraph,
    inputs: Mapping[str, Mapping[str, torch.Tensor]],
    expected: Mapping[str, Sequence[torch.Tensor]],
    out_dir: Path,
) -> list[str]:
    """Write each case's inputs and torch CPU expected outputs to `io.<case>.safetensors`.

    The expected values come as **positional tuples** (only `caption-proj` has 2 — the G1b section
    of the module docstring). An io whose count disagrees with the IR's output count would pass
    quietly as "readable, but one output never verified", so it is matched per case.
    """
    declared = [spec.name for spec in graph.inputs]
    written: list[str] = []
    for name, args in inputs.items():
        if sorted(args) != sorted(declared):
            raise AssertionError(
                f"{name}: the input names {sorted(args)} differ from the IR's {sorted(declared)}"
            )
        outputs = expected[name]
        if len(outputs) != len(graph.outputs):
            raise AssertionError(
                f"{name}: {len(outputs)} expected outputs, but the IR has {len(graph.outputs)}"
            )
        # MUST: the io is also lowered to the concrete representation of the IR's semantic dtype
        # (i64 → i32). Unless it matches the form the runtime receives, the Deno-side E2E cannot
        # read the golden (the boundary normalization of ADR 0009).
        tensors = {
            f"{INPUT_PREFIX}{key}": normalize_boundary_tensor(value, f"{name}: input '{key}'")
            for key, value in args.items()
        }
        for index, value in enumerate(outputs):
            tensors[f"{OUTPUT_PREFIX}{index}"] = normalize_boundary_tensor(
                value.detach().contiguous(), f"{name}: output {index}"
            )
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written


def _sanity(pristine: Mapping[str, Mapping[str, Sequence[torch.Tensor]]]) -> dict[str, Any]:
    """Check that the text-side and caption-side projectors are **different** (comparing the first
    outputs).

    MUST: do not let it become vacuous — the mix-up of reading the same prefix twice matches in
    shape and dtype, so it is invisible anywhere but here (the golden would also be baked with the
    same values on both sides and match).
    """
    divergence = {
        name: float(
            (pristine[TARGET_TEXT_PROJ][name][0] - pristine[TARGET_CAPTION_PROJ][name][0])
            .abs()
            .max()
        )
        for name in pristine[TARGET_TEXT_PROJ]
    }
    worst = min(divergence.values())
    if worst < PROJECTOR_DIVERGENCE_MIN:
        raise AssertionError(
            f"The minimum output difference between the text / caption projectors is {worst} —"
            " the same weights are suspected to be read twice"
        )
    return {name: round(value, 5) for name, value in divergence.items()}


def _norm_divergence(text_norm: nn.Module, caption_norm: nn.Module) -> float:
    """Check that `text_norm` and `caption_norm` are **different** (the max abs difference of their
    weights).

    MUST: do not let it become vacuous — both are `[512]` RMSNorm weights, so the mix-up of reading
    the same prefix twice matches in shape and dtype. The whole contract of `caption-proj`'s second
    output is "the sequence with `caption_norm` applied", yet **the reference and the wrapper use
    the same module**, so after a mix-up the golden would match itself and pass quietly (the norm
    version of the hole `_sanity` closes for the projectors).
    """
    worst = float((text_norm.weight - caption_norm.weight).detach().abs().max())
    if worst < NORM_DIVERGENCE_MIN:
        raise AssertionError(
            f"The weight difference between text_norm / caption_norm is {worst} — the same"
            " weights are suspected to be read twice"
        )
    return worst


class TargetAxis(NamedTuple):
    """A target's declaration of symbolic dimensions, and how SDPA is treated.

    - `torch_dim` / `symbol`: the torch `Dim` name and the name that goes into the IR. **Kept
      separately** — `dit` cannot use `"S"` as its torch name (the MUST of `DIT_TORCH_DIM`). A
      target without a symbolic dimension (`dit-context` — its inputs are static shapes already
      right-padded to Tmax) has `None` for both.
    - `dynamic`: **input position → (axis, offset)**. Offset 0 is the plain symbol, nonzero is a
      derived dimension (`S+1519` — ADR 0046 / only `dit`'s `mask`).
    - `preserved`: the ops removed from the decomposition table. Fused attention (ADR 0023) is used
      only by targets without a mask or with an additive `[1,1,M,N]` one; `dit` has a run-time bool
      mask, so it falls to the decomposed path (+ `safe_softmax` — ADR 0044).
    """

    torch_dim: str | None
    symbol: str | None
    upper: int
    dynamic: Mapping[int, tuple[int, int]]
    preserved: tuple[str, ...]


def _dynamic_axis(found: tuple[int, int] | None, seq: Any) -> dict[int, Any] | None:
    """Lower one element of {@link TargetAxis.dynamic} to `torch.export`'s `dynamic_shapes`."""
    if found is None:
        return None
    axis, offset = found
    return {axis: seq if offset == 0 else seq + offset}


def _dynamic_shapes(wrapper: nn.Module, axis: TargetAxis, count: int) -> tuple[Any, ...]:
    """`dynamic_shapes` with the per-input-position specification laid out in the **tree
    structure** of the forward's arguments.

    torch.export takes variadic arguments (`*context` of `DitGraph`) as a subtree of one tuple, so a
    flat sequence fails with "inputs and dynamic_shapes have different structures". The positional
    arguments before the variadic ones stay as is, and the rest are bundled into one tuple.
    """
    seq = (
        None if axis.torch_dim is None else Dim(axis.torch_dim, min=MIN_SYM_LENGTH, max=axis.upper)
    )
    flat = tuple(_dynamic_axis(axis.dynamic.get(index), seq) for index in range(count))
    parameters = inspect.signature(wrapper.forward).parameters.values()
    if not any(parameter.kind is parameter.VAR_POSITIONAL for parameter in parameters):
        return flat
    fixed = sum(
        1
        for parameter in parameters
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    )
    return (*flat[:fixed], flat[fixed:])


def name_boundary(
    graph: IrGraph,
    *,
    inputs: Sequence[str] | None = None,
    outputs: Sequence[str] | None = None,
) -> IrGraph:
    """Return a new graph with the IR's boundary names (graph inputs / outputs) renamed **by
    position**.

    torch.export names the forward's variadic arguments `<argument name>_<i>` and the outputs by FX
    node name (`cat_3` and so on). The runtime binds the DiT's boundary by name (`copyOutputs` and
    the inputs of `dit` — ADR 0114), so they are renamed after the export to the contract's spelling
    (`irodori.distribution.irodori_context_kv_names`). The order is the forward's argument order /
    return order itself (torch.export keeps it), so they are matched by position. A side given as
    `None` is left alone.

    MUST: fail if a new name collides with a remaining value name (overwriting would silently drop
    one declaration). Fail if an output is a value that does not go through a node (returning an
    input / initializer as is) — renaming it would move input names and weight keys too.
    """
    rename: dict[str, str] = {}
    if inputs is not None:
        current = [spec.name for spec in graph.inputs]
        if len(current) != len(inputs):
            raise AssertionError(f"Tried to rename {len(current)} inputs with {len(inputs)} names")
        rename.update({old: new for old, new in zip(current, inputs, strict=True) if old != new})
    if outputs is not None:
        if len(graph.outputs) != len(outputs):
            raise AssertionError(
                f"Tried to rename {len(graph.outputs)} outputs with {len(outputs)} names"
            )
        produced = {name for node in graph.nodes for name in node.outs}
        for old, new in zip(graph.outputs, outputs, strict=True):
            if old not in produced:
                raise AssertionError(
                    f"Output '{old}' is not a node output (an input / initializer is returned as"
                    " is)"
                )
            if old != new:
                rename[old] = new
    occupied = {spec.name for spec in graph.inputs} | set(graph.values) | set(graph.initializers)
    targets = list(rename.values())
    clashes = sorted((occupied - set(rename)) & set(targets))
    if clashes or len(set(targets)) != len(targets):
        raise AssertionError(
            f"The new boundary names collide with existing value names: {clashes or targets}"
        )

    def renamed(name: str) -> str:
        return rename.get(name, name)

    return IrGraph(
        symbols=list(graph.symbols),
        inputs=[
            IrInput(name=renamed(spec.name), dtype=spec.dtype, shape=list(spec.shape))
            for spec in graph.inputs
        ],
        outputs=[renamed(name) for name in graph.outputs],
        initializers=dict(graph.initializers),
        values={renamed(name): value for name, value in graph.values.items()},
        states=dict(graph.states),
        nodes=[
            dataclasses.replace(
                node,
                ins=[renamed(name) for name in node.ins],
                outs=[renamed(name) for name in node.outs],
            )
            for node in graph.nodes
        ],
    )


def export_ir(
    wrapper: nn.Module,
    example: tuple[torch.Tensor, ...],
    *,
    axis: TargetAxis,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """Convert one wrapper to IR and rename its boundary names (the first half of
    `karume.pipeline.export_to_file`, split off so that {@link name_boundary} can run before
    publishing).

    Publishing (`karume.pipeline.publish_model`) is done by the caller, {@link export_series} — to
    keep the place that names the graph (the component name) in one spot (the check in
    `tests/test_graph_names.py`).
    """
    graph, tensors = export_module(
        wrapper,
        example,
        dynamic_shapes=_dynamic_shapes(wrapper, axis, len(example)),
        symbol_names=() if axis.symbol is None else (axis.symbol,),
        preserved=axis.preserved,
    )
    return name_boundary(graph, inputs=input_names, outputs=output_names), tensors


def _graph_summary(graph: IrGraph, path: Path) -> dict[str, Any]:
    return {
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "model_bytes": sum(p.stat().st_size for p in container_parts(path)),
    }


class FakeQuantResult(NamedTuple):
    """The return value of {@link fake_quant} (a summary of the rounding, the i8 / i4 scale ledger,
    and per-tensor storage specifications)."""

    #: Role name → summary of how many tensors were rounded (a count so that "`--dtype` was given
    #: but 0 tensors were rounded" is not silent).
    reports: dict[str, str]
    #: Role name → **FQN in the bare module** → scale (empty for f32 / f16). The remapping to the
    #: **FQN in the wrapper** that the emit looks up is done by {@link target_scales}.
    scales: dict[str, Mapping[str, torch.Tensor]]
    #: Role name → **FQN in the bare module** → **explicit specification** of the storage dtype
    #: (empty except for i4). The remapping is {@link target_weight_dtypes}. Whereas the default
    #: `weight_dtype` quietly leaves ineligible weights in f32, an explicit specification that
    #: cannot be met makes the emit fail loudly (`_plan_weight_dtype` of `karume.emit`) — which is
    #: why the i4 series names the 317 linears of the distribution graphs one by one (the 168
    #: inside the blocks, excluding adaLN, as `i4`; the 144 adaLN + 5 outside the blocks = 149 as
    #: `i8`. The containers: `dit` has i4 96 + i8 149, `dit-context` has i4 72). **Naming the i8
    #: side explicitly as `i8` is required**: the default `weight_dtype` is `i4`, so without the
    #: specification `_plan_weight_dtype` would route the i4-eligible linears into the i4 plan by
    #: default, and they would fail loudly for lack of a group scale (only the i8 per-channel scale
    #: exists).
    overrides: dict[str, Mapping[str, str]]


@dataclass(frozen=True)
class CalibPlan:
    """What the rounding of the i4 series needs "besides the weights" (passed only with `--dtype
    i4`).

    `calibrated` False = `--no-calib` (plain RTN, for smoke runs that are **not shipped**). There is
    no branch that swallows a calibration failure and falls back to plain RTN — the opt-out is this
    one explicit switch alone.
    """

    context: CalibContext
    #: The calibration cases (as returned, with the separation check, by
    #: `irodori.calib_cases.calibration_cases`).
    cases: tuple[Any, ...]
    #: The number of captured steps per case.
    steps: int
    calibrated: bool = True


#: Target → the sequence of sources of the scales that go into its graph `(role, FQN prefix in the
#: bare module, FQN prefix in the wrapper)` (used for i8 / i4).
#:
#: MUST: use the same spelling as the graph wrappers' constructors — the wrappers hold the modules
#: that `load_*` returned under other attribute names, so the ledger's keys (FQNs in the bare
#: modules) do not match the keys the emit looks up (FQNs in the exported module). A mismatch makes
#: either `target_scales` or the emit fail loudly (there is no path that silently falls to f32
#: storage).
TARGET_SCALE_SOURCES: Mapping[str, tuple[tuple[str, str, str], ...]] = MappingProxyType(
    {
        # `BackboneGraph` holds `backbone.backbone` (the ModernBERT body) as `model`.
        TARGET_BACKBONE: ((TARGET_BACKBONE, "backbone.", "model."),),
        TARGET_TEXT_PROJ: ((TARGET_TEXT_PROJ, "", "projector."),),
        # `CaptionProjectorGraph` contains `ProjectorGraph` as `projection` (2 levels).
        TARGET_CAPTION_PROJ: ((TARGET_CAPTION_PROJ, "", "projection.projector."),),
        TARGET_SPEAKER: ((TARGET_SPEAKER, "", "encoder."),),
        TARGET_DURATION: ((TARGET_DURATION, "", "predictor."),),
        # `DitGraph` / `DitContextGraph` hold subtrees of the DiT body under **the same attribute
        # names** (no remapping needed). Both look up the ledger of the same `dit` role (the result
        # of rounding the whole `TextToLatentRFDiT` once), and it splits naturally along what each
        # owns ({@link _submodule_view}). The ledger also holds the inner copies (the backbone and
        # so on) that `load_dit` builds as a whole and the other wrapper's weights, but those are
        # not in the wrapper's `named_parameters` and drop out.
        TARGET_DIT: ((TARGET_DIT, "", ""),),
        TARGET_DIT_CONTEXT: ((TARGET_DIT, "", ""),),
    }
)


def _has_quantizable_weights(module: nn.Module) -> bool:
    """Whether it contains at least one module of a per-channel i8 target type
    (`QUANT_CHANNEL_AXES`)."""
    types = tuple(QUANT_CHANNEL_AXES)
    return any(isinstance(child, types) for child in module.modules())


def _rebase_to_wrapper(
    target: str, wrapper: nn.Module, by_role: Mapping[str, Mapping[str, Any]], what: str
) -> dict[str, Any]:
    """Remap the per-role ledgers to the target's **FQNs in the wrapper** and merge them into one.

    The emit looks up both the scales and the per-tensor storage specifications by the
    initializers' tensor keys (= FQNs in the exported module) (`karume.emit._plan_i8` /
    `_plan_weight_dtype`), so passing the keys relative to the bare modules that `load_*` returns
    would hit nothing. The remapping table is {@link TARGET_SCALE_SOURCES}.

    Only **weights not in the wrapper** are dropped here (the `dit` ledger even includes inner
    copies that are not used). The reverse (in the wrapper but not in the ledger) makes the emit
    fail loudly — an eligible weight slot without a scale cannot be stored compressed, so there is
    no path that silently falls to f32.

    This holds the one implementation so that **the same remapping is not written twice**, once for
    scales and once for storage specifications (if the keys diverged, only the i4 specifications
    would point outside the wrapper while staying green).
    """
    owned = {name for name, _ in wrapper.named_parameters()}
    picked: dict[str, Any] = {}
    for role, source, destination in TARGET_SCALE_SOURCES[target]:
        for key, value in by_role.get(role, {}).items():
            if not key.startswith(source):
                continue
            rebased = destination + key[len(source) :]
            if rebased in owned:
                picked[rebased] = value
    if not picked:
        raise SystemExit(
            f"{target}: not a single entry of the {what} could be remapped to an FQN in the wrapper"
            f" (the prefixes of {TARGET_SCALE_SOURCES[target]} disagree with how the wrapper is"
            " built)"
        )
    return picked


def target_scales(
    target: str, wrapper: nn.Module, scales: Mapping[str, Mapping[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """Remap the per-role scale ledgers to FQNs in the wrapper and merge them into one (shared by
    i8 / i4)."""
    if not scales:
        return {}
    return _rebase_to_wrapper(target, wrapper, scales, "scale ledger")


def target_weight_dtypes(
    target: str, wrapper: nn.Module, overrides: Mapping[str, Mapping[str, str]]
) -> dict[str, str]:
    """Remap the per-role, per-tensor storage specifications to FQNs in the wrapper (non-empty only
    in the i4 series)."""
    if not overrides:
        return {}
    return _rebase_to_wrapper(target, wrapper, overrides, "explicit storage dtype specifications")


def _calib_plan(
    model_dir: Path,
    source: Any,
    config: Any,
    text_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    steps: int | None,
    no_calib: bool,
) -> CalibPlan:
    """Build the scaffolding for rounding the i4 series (only calibration cases that passed the
    separation check).

    The default step count is the whole reference loop of the checkpoint's flow parameterization
    (40 Euler steps, or the 4 steps of a MeanFlow loop — ADR 0150).
    """
    from . import calib
    from .calib_cases import calibration_cases
    from .pipeline_ref import reference_steps

    loop_steps = reference_steps(irodori_flow_parameterization(model_config))
    if steps is not None and not 1 <= steps <= loop_steps:
        raise SystemExit(
            f"--calib-steps must be 1 to {loop_steps} for this checkpoint (the step count of its"
            f" reference loop; given {steps})"
        )
    return CalibPlan(
        context=calib.CalibContext(
            model_dir=model_dir,
            source=source,
            config=config,
            text_config=text_config,
            model_config=model_config,
        ),
        cases=calibration_cases(),
        steps=loop_steps if steps is None else steps,
        calibrated=not no_calib,
    )


def _write_calib_provenance(
    dtype: str, plan: CalibPlan | None, target: str, target_dir: Path
) -> str | None:
    """Write the rounding conditions of the i4 series (method, grid, case count, step count) into
    the series and return the file name.

    Whether calibration happened **does not change a single byte of the storage format** (the grid
    stays RTN i4 g32 — only the rounded values and the scale ledger change), so it cannot be told
    from the assets, the manifest or the header checks. Yet the quality difference is large enough
    to decide a ruling, so `irodori.distribution.assert_irodori_calib_provenance` matches this
    record at assembly time and stops the output of `--no-calib` from slipping into a
    distribution.

    It is **written** even with `--no-calib` (recorded as `rtn`, not removed) — an absent record
    could also read as "an old export", and the assembly side could not refuse "an attempt to ship
    without calibration" by name.

    **The same record** is written to **both** targets that have an i4 series (the DiT's two graphs
    — {@link DTYPE_TARGETS}). The two are the same rounding written out through 2 wrappers (ADR
    0114), and the assembly side reads both and checks that they agree.

    MUST: anything other than i4 **removes an old record** (a total function). What actually
    guarantees the absence of the record, though, is the swap — the caller (`export_series`) always
    passes an empty staging slot, and `swap_into_place` is a rename of the whole directory that
    carries over none of the old contents, so re-taking a series taken as i4 with another dtype
    removes the previous record in the swap. The `unlink` below is therefore always a no-op on the
    caller's path and stays as a guard for callers that do not go through staging (tests, future
    direct writes).
    """
    from . import calib

    path = target_dir / CALIB_PROVENANCE_FILE
    if dtype != "i4" or target not in DTYPE_TARGETS["i4"] or plan is None:
        path.unlink(missing_ok=True)
        return None
    record = {
        "method": calib.CALIB_METHOD if plan.calibrated else "rtn",
        "grid": calib.CALIB_GRID.kind,
        "group_size": calib.CALIB_GRID.group_size,
        "cases": len(plan.cases) if plan.calibrated else 0,
        "steps": plan.steps if plan.calibrated else 0,
    }
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return CALIB_PROVENANCE_FILE


def _round_i4_calibrated(
    dit: nn.Module,
    stages: Sequence[StageSpec],
    i4_names: frozenset[str],
    plan: CalibPlan,
) -> tuple[Mapping[str, torch.Tensor], str]:
    """Round **the side excluding adaLN** of the linears inside the DiT's blocks (= everything
    stored as i4) with GPTQ × the RTN grid.

    Order MUST (① → ② → ③):

    1. **Pass the stage-split agreement check before rounding** (`irodori.calib.assert_stage_split`)
       — rounding with a misaligned split would ship "GPTQ of a different path", and the numbers
       would still come out normally. The forward fed to the check is **1 step of the reference
       loop before rounding** (taken the same way as the calibration inputs, so a probe with a
       fabricated shape has no room to drift from upstream).
    2. Generate the calibration inputs (capturing reference denoising). **The side shipped as i8
       is rounded first** — the caller ({@link _fake_quant_i4}) finishes all of i8 before coming
       here. At distribution run time what enters the blocks is the hidden state made by the i8
       `in_proj` and the condition embedding made by the i8 `cond_module`, and inside the blocks
       the modulation is made by the i8 adaLN too. Doing i8 later would ship rounding targets chosen
       with "activations that went through f32 surroundings" combined with i8 surroundings.
    3. Round the linears inside the blocks other than adaLN with GPTQ × the RTN grid. **The
       exclusion is made exclusive through core's `include`** (`irodori.calib.is_adaln` — one
       implementation decides both the rounded set and the set stored as i4).

    Returns `(the i4 scale ledger, one diagnostic line)`.
    """
    from . import calib

    probe = calib.capture_stage_batches(dit, plan.cases[:1], plan.context, steps=1, label="probe")[
        0
    ]
    calib.assert_stage_split(stages, probe)
    runs = calib.capture_stage_batches(dit, plan.cases, plan.context, steps=plan.steps)
    batches = tuple(batch for run in runs for batch in run.batches)
    report, ledger = calib.calibrate_i4(
        stages, batches, include=lambda local: not calib.is_adaln(local)
    )
    calib.assert_calib_covers_scan(report, i4_names)
    summary = (
        f"rounded to storage i4 — {len(i4_names)} non-adaLN linears inside the blocks, with GPTQ"
        f" calibration ({report.describe()}, "
        f"{len(plan.cases)} cases × {plan.steps} steps = {len(batches)} batches)"
    )
    return ledger.scales, summary


def _fake_quant_i4(
    modules: Mapping[str, nn.Module], plan: CalibPlan | None
) -> tuple[dict[str, str], dict[str, Mapping[str, torch.Tensor]], dict[str, Mapping[str, str]]]:
    """Rounding of the mixed i4 series (the DiT's linears **inside the blocks, excluding adaLN** =
    i4 g32; everything else i8 per-channel).

    Narrowing i4 **to inside the blocks** is the attribution result of the listening ruling of
    2026-08-23. The 5 linears outside the blocks (`in_proj` / `out_proj` / `cond_module.{0,2,4}`)
    were not covered by GPTQ, because calibration is driven per stage, and were rounded with plain
    RTN i4; measurements confirmed that this was one cause of the muffled sound of the i8+dit4
    preset (the one remaining axis is how well the calibration corpus generalizes —
    {@link irodori.calib_cases}).

    In addition, **the 144 adaLN linears go back to i8 as well** (the same listening ruling of
    2026-08-23 — in an A/B sim "the delivery clearly improved"). The scale / shift / gate of the
    modulation are highly sensitive to quantization, and left in i4 they kept the prosody off even
    after the out-of-block linears moved to i8. The cost is +13.1 MiB (+6.0% of the dit i4
    payload). The judgement is a segment match ({@link irodori.calib.is_adaln}), and the 168
    linears inside the blocks are what stays in i4.

    Even after the DiT was split into 2 graphs (ADR 0114), the rounding happens **once** on the
    whole `TextToLatentRFDiT`, and what drives calibration (stage = a whole block, the calibration
    inputs) is the same as before the split. In the containers the 168 split between `dit-context`
    (72 condition-side K/V projections) and `dit` (the other 96), but that is decided solely by the
    remapping of {@link TARGET_SCALE_SOURCES} picking up the ledger within what each wrapper owns —
    the ledger and storage specifications returned here stay a single `dit` role.

    The 2 predicates split **exclusively** at "inside the blocks − adaLN" (the mix MUST of
    `quantize.py`). `dit` is **the whole** `TextToLatentRFDiT` (it holds copies of the backbone /
    projector / speaker / duration inside), so everything outside the blocks falls to i8 here —
    the same condition as the distribution's quant preset `i8+dit4` assigning the bytes of the i8
    series to the other 7 roles, and the key to **matching the calibration inputs to the
    distribution conditions** (`irodori.distribution.IRODORI_QUANT_SEATS`).

    NOTE (deferred import): `irodori.calib` imports this module back through
    `irodori.pipeline_ref`, so reading it at module level would be circular. Also `irodori_tts` /
    `tokenizers` live in the dependency group, so this matches the convention of these recipes as
    well (import inside the function that needs it).
    """
    from . import calib

    if plan is None:
        raise SystemExit(
            "Storage i4 needs the calibration scaffolding (CalibPlan)"
            " (call it from `python -m irodori.export --dtype i4`"
            " — there is no branch that silently falls back to plain RTN)"
        )
    dit = modules[TARGET_DIT]
    graph_names = calib.dit_i4_names(dit, dit_sym_max(plan.context.config))
    stages = calib.dit_stages(dit)
    stage_names = calib.stage_linear_names(stages)
    adaln_names = frozenset(name for name in stage_names if calib.is_adaln(name))
    i4_names = stage_names - adaln_names
    # MUST: every linear inside the blocks, without exception, must be eligible on the distribution
    # graph side — if one is not, upstream's construction has changed (calibration drives whole
    # stages, so there is no escape hatch of leaving out just one).
    unaligned = sorted(stage_names - graph_names)
    if unaligned:
        raise SystemExit(
            f"The linears {unaligned[:3]} inside the DiT blocks are not i4-eligible"
            " (calibration drives whole stages, so there is no escape hatch of leaving out just"
            " one)"
        )
    # MUST: having no adaLN at all fails loudly — if the 144 linears the listening ruling moved back
    # to i8 silently returned to i4 through an upstream rename, "the configuration the ruling
    # rejected" would ship with the storage format and the count checks still correct.
    # MUST: check **per spelling** ({@link irodori.calib.adaln_segments_seen}) — a rename on one
    # side only does not empty the set, so it would slip through a check that only looks for "none
    # at all", and the later surplus/shortage check builds both sides from the same classifier, so
    # it would stay self-consistent and green (only half, 72 linears, would silently return to i4).
    missing = sorted(calib.ADALN_SEGMENTS - calib.adaln_segments_seen(stage_names))
    if missing:
        raise SystemExit(
            f"The DiT blocks have no adaLN linear for the spellings: {missing}"
            f" (spellings searched: {sorted(calib.ADALN_SEGMENTS)}; adaLN linears found:"
            f" {len(adaln_names)}) — upstream has renamed the modulation's attributes"
        )
    # The side shipped as i8 = the eligible linears of the distribution graphs minus the 168 i4
    # ones (5 outside the blocks + 144 adaLN).
    i8_names = graph_names - i4_names
    reports: dict[str, str] = {}
    scales: dict[str, Mapping[str, torch.Tensor]] = {}
    # MUST: i8 is applied **before the calibration inputs are captured** (order MUST ② —
    # `_round_i4_calibrated`). The `dit` predicate leaves out only the 168 i4 linears, so the 5
    # outside the blocks and the 144 adaLN are rounded to i8 here too.
    for name, module in sorted(modules.items()):
        include = (lambda key: key not in i4_names) if name == TARGET_DIT else None
        if not _has_quantizable_weights(module):
            reports[name] = "storage stays f32 (no per-channel target type)"
            continue
        int8 = fake_quant_int8(module, include=include)
        scales[name] = int8.scales
        reports[name] = f"rounded to storage i8 — {int8.describe()}"
    for name, report in reports.items():
        print(f"[fake-quant] {name}: {report}", flush=True)
    if plan.calibrated:
        int4_scales, summary = _round_i4_calibrated(dit, stages, i4_names, plan)
    else:
        print(
            f"[fake-quant] {TARGET_DIT}: --no-calib — plain RTN without calibration"
            " (do not ship it; quality is below the perf-ledger Q-2 baseline)",
            flush=True,
        )
        plain = fake_quant_int4(dit, include=lambda name: name in i4_names)
        int4_scales = plain.scales
        summary = f"rounded to storage i4 (RTN, no calibration) — {plain.describe()}"
    print(f"[fake-quant] {TARGET_DIT}: {summary}", flush=True)
    # The i8 summary is **kept** (`dit` goes through 2 stages — outside the blocks, adaLN and the
    # inner copies of the other roles are i8; the rest inside the blocks is i4). Overwriting it
    # would hide a regression where "the i8 targets dropped to 0" from the summary.
    reports[TARGET_DIT] = f"{reports[TARGET_DIT]} / {summary}"
    # MUST: the 2 paths are disjoint (an overlap would mean the same weight was rounded twice, and
    # only the values would quietly go wrong). Checked **before** the surplus/shortage check below
    # — an overlap is a separate diagnosis, "the same weight rounded twice", and reporting it as a
    # count mismatch would hide the cause.
    overlap = sorted(set(scales.get(TARGET_DIT, {})) & set(int4_scales))
    if overlap:
        raise SystemExit(f"i8 and i4 round the same weights (double rounding): {overlap[:3]}")
    # MUST: the rounded set = the stored set (the keys of the overrides). The mismatch that occurs
    # is "counted as i4 but not rounded", which stays green with i8 weights mixed into the i4
    # preset (only the size quietly goes back).
    expected = {f"{name}.weight" for name in i4_names}
    if set(int4_scales) != expected:
        raise SystemExit(
            f"Of the {len(i4_names)} non-adaLN linears inside the blocks, {len(int4_scales)} were"
            f" rounded to i4 (surplus/shortage: {sorted(set(int4_scales) ^ expected)[:3]})"
        )
    scales[TARGET_DIT] = {**scales.get(TARGET_DIT, {}), **int4_scales}
    # MUST: the 149 on the i8 side (5 outside the blocks + 144 adaLN) are named **explicitly as
    # `i8`** ({@link FakeQuantResult.overrides}) — the default `weight_dtype` is i4, so without
    # this the emit would route the i4-eligible linears into the i4 plan by default and fail loudly
    # for lack of a group scale.
    overrides: dict[str, str] = {
        **{f"{name}.weight": "i8" for name in sorted(i8_names)},
        **dict.fromkeys(int4_scales, "i4"),
    }
    return reports, scales, {TARGET_DIT: overrides}


def fake_quant(
    dtype: str, modules: Mapping[str, nn.Module], *, calib_plan: CalibPlan | None = None
) -> FakeQuantResult:
    """Round the **effective weights** to the values the storage dtype can represent (f32 does
    nothing).

    The fake-quant of ADR 0006 / 0018 / 0019 / 0027 / 0050. It returns the number of rounded
    tensors keyed by role name (so that "`--dtype f16` was given but 0 tensors were rounded" is not
    silent — a total of 0 fails).

    MUST (order): ① call it **right after** `load_*`. The weights of this script are effective
    weights already at the point of `irodori/dacvae/convert.py` / the checkpoint (the equivalent of
    `remove_weight_norm` is baked in at conversion, and there is no LoRA), so the point that
    corresponds to SBV2's "after remove" moves up to right after loading. ② call it **before** the
    reference / golden is taken (the MUST of `karume.quantize` — applied afterwards, only the golden
    would be computed with the original weights, the E2E differences would contain the quantization
    error, and the tolerances would lose their meaning).

    MUST: what is bundled is **all of the 6 modules that go into graphs + the 3 norms the host uses
    to build golden inputs**. `text_norm` / `caption_norm` also live as separate copies inside
    `duration` / `dit`, but both are rounded to the same values, so the chain (using the expected
    values of an upstream target as the next target's inputs) is preserved. Missing even one would
    put values computed with the original weights on that side alone into the golden.

    NOTE: f16 rounds only the f32 parameters / buffers (`round_weights_to_f16`). The graph constants
    (band mask, RoPE tables, the index tables of `sym_prefix_slice`) are not module weights, so they
    are left alone, and the emit's eligibility check keeps them in f32 storage too.

    NOTE: i8 touches only the `weight` of the `QUANT_CHANNEL_AXES` types (Linear / Conv /
    Embedding) (ADR 0019). Neither biases nor norm weights are **rounded** — the emit's eligibility
    check also stores everything except weight slots as f32, so the two agree. The 3 norms passing
    through untouched under i8 is therefore correct, and "0 targets" does not fail per role (it
    fails if the **total** is 0).

    NOTE: i4 (`--dtype i4`) is handled by {@link _fake_quant_i4} — only the DiT's linears **inside
    the blocks, excluding adaLN**, are i4 group32, and everything else, adaLN and the out-of-block
    linears included, is i8. The rounding is GPTQ-calibrated by default, so
    `calib_plan` is needed.
    """
    if dtype == "f32":
        return FakeQuantResult({}, {}, {})
    if not modules:
        raise SystemExit(f"Storage {dtype} was given but there is no module to round")
    if dtype == "i4":
        return FakeQuantResult(*_fake_quant_i4(modules, calib_plan))
    reports: dict[str, str] = {}
    scales: dict[str, Mapping[str, torch.Tensor]] = {}
    rounded = 0
    for name, module in sorted(modules.items()):
        if dtype != "i8":
            report = round_weights_to_f16(module)
            rounded += report.parameters + report.buffers
            reports[name] = f"rounded to storage f16 — {report.describe()}"
        elif _has_quantizable_weights(module):
            int8 = fake_quant_int8(module)
            scales[name] = int8.scales
            reports[name] = f"rounded to storage i8 — {int8.describe()}"
        else:
            reports[name] = "storage stays f32 (no per-channel target type)"
    for name, report_line in reports.items():
        print(f"[fake-quant] {name}: {report_line}", flush=True)
    if dtype == "i8" and not scales:
        raise SystemExit("Storage i8 was given but no module could be quantized per channel")
    if dtype != "i8" and rounded == 0:
        raise SystemExit(f"Storage {dtype} was given but not a single weight was rounded")
    return FakeQuantResult(reports, scales, {})


def export_series(
    model_dir: Path,
    source_dir: Path,
    out_dir: Path,
    *,
    targets: Sequence[str] = TARGETS,
    sym_max: int = SYM_MAX,
    dtype: str = "f32",
    calib_steps: int | None = None,
    no_calib: bool = False,
) -> dict[str, Any]:
    """Write the IR containers and golden io, and return a summary.

    `targets` narrows **only the writing** (the writing loop at the end). The prerequisite assets
    and the reference computations always run in full — the requirement for the real latent
    (`build_real_speaker_cases`) and the torch forwards of speaker / duration / DiT do not depend
    on `targets`. Not allowing only the downstream of the chain to be re-baked is intentional:
    precisely because "baking the downstream always re-bakes the upstream too", consistency
    between the goldens is kept structurally, and no execution path arises that skips the
    anti-vacuity checks (`_sanity` / `_norm_divergence` / `_dit_uncond_divergence`).
    """
    source = IrodoriSource(source_dir)
    text_config, model_config = read_configs(model_dir)
    if str(model_config["pretrained_projector_type"]) != "residual_mlp":
        raise SystemExit(
            f"The projector type {model_config['pretrained_projector_type']!r} is not supported"
            " (v4-Small uses residual_mlp)"
        )
    state = load_file(str(model_dir / CHECKPOINT_FILE))
    backbone = load_backbone(source, state, text_config)
    hidden_size = int(backbone.hidden_size)
    projectors = {
        TARGET_TEXT_PROJ: load_projector(
            source,
            state,
            model_config,
            TEXT_PROJ_PREFIX,
            hidden_size,
            int(model_config["text_dim"]),
        ),
        TARGET_CAPTION_PROJ: load_projector(
            source,
            state,
            model_config,
            CAPTION_PROJ_PREFIX,
            hidden_size,
            int(model_config["caption_dim"]),
        ),
    }
    config = source.model_config(model_config)
    speaker_encoder = load_speaker_encoder(source, state, config)
    speaker_norm = load_rms_norm(
        source, state, SPEAKER_NORM_PREFIX, int(config.speaker_dim), float(config.norm_eps)
    )
    text_norm = load_rms_norm(
        source, state, TEXT_NORM_PREFIX, int(config.text_dim), float(config.norm_eps)
    )
    caption_norm = load_rms_norm(
        source,
        state,
        CAPTION_NORM_PREFIX,
        int(config.caption_dim_resolved),
        float(config.norm_eps),
    )
    duration = load_duration_predictor(source, state, config)
    dit = load_dit(source, state, config, text_config)
    # The checkpoint's own flow parameterization decides the step graph's inputs and the golden
    # cases (ADR 0150) — read from the loaded DiT, never from the directory name.
    flow = dit_flow(dit)
    calib_plan = (
        _calib_plan(model_dir, source, config, text_config, model_config, calib_steps, no_calib)
        if dtype == "i4"
        else None
    )
    # MUST: rounding comes right after loading and before the reference is taken (the order MUST
    # of {@link fake_quant}). The reference loop of calibrated i4 also runs at this point — why it
    # must be driven by **the bare DiT** (before the reference, which can only be taken unpatched)
    # is the MUST in the module docstring of `irodori.calib`.
    quantized = fake_quant(
        dtype,
        {
            TARGET_BACKBONE: backbone,
            TARGET_TEXT_PROJ: projectors[TARGET_TEXT_PROJ],
            TARGET_CAPTION_PROJ: projectors[TARGET_CAPTION_PROJ],
            TARGET_SPEAKER: speaker_encoder,
            TARGET_DURATION: duration,
            TARGET_DIT: dit,
            "speaker_norm": speaker_norm,
            "text_norm": text_norm,
            "caption_norm": caption_norm,
        },
        calib_plan=calib_plan,
    )
    cases = build_cases(model_dir, source, text_config, sym_max)
    speaker_max = speaker_sym_max(model_config)
    speaker_cases = (
        *build_speaker_cases(int(config.speaker_patched_latent_dim), speaker_max),
        *build_real_speaker_cases(
            source.patch_sequence_with_mask,
            int(config.latent_dim),
            int(config.speaker_patch_size),
            int(config.speaker_patched_latent_dim),
            speaker_max,
        ),
    )
    dit_max = dit_sym_max(config)
    #: The total condition-side length of `dit`'s mask (the derived dimension `S+<this>` — 1519 in
    #: ADR 0047).
    dit_context_total = (
        int(model_config["max_text_len"]) + (speaker_max + 1) + int(model_config["max_caption_len"])
    )

    # ---- ① The unpatched references (the golden expected values) and measuring the scheme ----
    pristine = _pristine_outputs(backbone, projectors, caption_norm, cases)
    static_evidence = _static_scheme_evidence(
        backbone, text_config, model_config, cases, _first(pristine[TARGET_BACKBONE])
    )
    sanity = _sanity(pristine)
    norm_divergence = _norm_divergence(text_norm, caption_norm)
    pristine[TARGET_SPEAKER] = _single(
        _pristine_speaker_outputs(speaker_encoder, speaker_norm, speaker_cases)
    )
    no_reference_max_abs = _no_reference_evidence(
        speaker_encoder,
        int(config.speaker_patched_latent_dim),
        int(config.speaker_patch_size),
    )
    duration_inputs, duration_reference = _duration_cases(
        source,
        duration,
        text_norm,
        caption_norm,
        _first(pristine[TARGET_TEXT_PROJ]),
        _first(pristine[TARGET_CAPTION_PROJ]),
        _first(pristine[TARGET_SPEAKER]),
    )
    aux_dim = int(config.duration_aux_dim)
    duration_pristine = _pristine_duration_outputs(duration, duration_reference, aux_dim)
    pristine[TARGET_DURATION] = _single(duration_pristine)
    aux_inert = _duration_aux_is_inert(duration, duration_reference, aux_dim, duration_pristine)
    dit_inputs, dit_reference = _dit_cases(
        source,
        config,
        model_config,
        text_norm,
        caption_norm,
        speaker_max,
        dit_max,
        _first(pristine[TARGET_TEXT_PROJ]),
        _first(pristine[TARGET_CAPTION_PROJ]),
        _first(pristine[TARGET_SPEAKER]),
        flow=flow,
    )
    dit_pristine = _pristine_dit_outputs(dit, dit_reference)
    dit_divergence = _dit_uncond_divergence(dit_pristine, dit_cases(flow))
    dit_delta_divergence = (
        _dit_delta_divergence(dit_pristine) if flow == IRODORI_FLOW_MEANFLOW else None
    )
    # `dit-context` has one case per set of condition states (one, if every DiT case uses the same
    # set). Its expected values are taken unpatched through upstream's cache path
    # (`build_context_kv_cache`) and used as is as the K/V of `dit`'s golden inputs (a chain).
    context_inputs, context_owner = _dit_context_cases(dit_inputs)
    pristine[TARGET_DIT_CONTEXT] = _pristine_dit_context_outputs(dit, context_inputs)
    pristine[TARGET_DIT] = _single(dit_pristine)
    dit_heads = irodori_dit_head_inputs(flow)
    dit_graph_inputs = _dit_graph_inputs(
        dit_inputs,
        context_owner,
        pristine[TARGET_DIT_CONTEXT],
        irodori_context_kv_names(len(dit.blocks)),
        dit_heads,
    )

    # ---- ② Apply the patches (eager values taken from here on cannot serve as references) ----
    rope_buffers = sorted(
        name for name, _ in backbone.backbone.named_buffers() if name.endswith(ROPE_BUFFER_NAMES)
    )
    patch.apply_patches()

    # ---- ③ Eager equivalence of the wrappers (measures the patches and the dropped masks) ----
    dit_graphs = dit_wrappers(dit, dit_max)
    if dit_graphs.dit.input_names[: len(dit_heads)] != dit_heads:
        raise AssertionError(
            f"The step wrapper's head inputs {dit_graphs.dit.input_names[: len(dit_heads)]} are not"
            f" the {flow} heads {dit_heads}"
        )
    graphs = {
        TARGET_BACKBONE: BackboneGraph(backbone),
        TARGET_TEXT_PROJ: ProjectorGraph(projectors[TARGET_TEXT_PROJ]),
        TARGET_CAPTION_PROJ: CaptionProjectorGraph(projectors[TARGET_CAPTION_PROJ], caption_norm),
        TARGET_SPEAKER: SpeakerGraph(speaker_encoder, speaker_norm, speaker_max),
        TARGET_DURATION: DurationGraph(duration, text_norm),
        **dit_graphs.by_target(),
    }
    graph_args: dict[str, dict[str, dict[str, torch.Tensor]]] = {
        TARGET_BACKBONE: {name: {"input_ids": ids} for name, _kind, ids in cases},
        TARGET_SPEAKER: {name: {"latent": latent} for name, latent in speaker_cases},
        TARGET_DURATION: duration_inputs,
        TARGET_DIT: dit_graph_inputs,
        TARGET_DIT_CONTEXT: context_inputs,
    }
    for target in (TARGET_TEXT_PROJ, TARGET_CAPTION_PROJ):
        graph_args[target] = {
            name: {"hidden": pristine[TARGET_BACKBONE][name][0]} for name, _kind, _ids in cases
        }
    equivalence = {
        target: max(
            _check_wrapper_equivalence(
                module,
                tuple(args.values()),
                pristine[target][name],
                f"{target}/{name}",
                EAGER_EQUIV_ATOL,
            )
            for name, args in graph_args[target].items()
        )
        for target, module in graphs.items()
    }
    # The split's equivalence check (ADR 0114): the composition of the 2 split graphs matches
    # upstream's unpatched `forward_with_encoded_conditions` bit for bit in every case. The check of
    # `dit` alone above takes "K/V = upstream's cached values" as input, so it does not cover that
    # the `dit-context` wrapper itself can produce those values — the composition checks how the 2
    # mesh as well (by positional order, not by name).
    split_equivalence = max(
        _check_wrapper_equivalence(
            ComposedDitGraph(dit_graphs.context, dit_graphs.dit),
            tuple(args.values()),
            pristine[TARGET_DIT][name],
            f"dit-split/{name}",
            EAGER_EQUIV_ATOL,
        )
        for name, args in dit_inputs.items()
    )

    # ---- ④ Export and write the golden ----
    # The declaration of symbolic dimensions per target ({@link TargetAxis}). The 3 text graphs and
    # duration use T (512), speaker uses S (the upper bound of the reference latent after
    # patching), and dit uses S too (latent length 750 — a separate graph, so the names do not
    # collide). The 2nd to 5th inputs of duration (the condition vectors and bools) and the 2nd /
    # 4th-onward inputs of dit (t_embed and the condition-side K/V) are fixed shapes without
    # symbols, and dit-context has no symbolic dimension at all (its inputs are right-padded to
    # Tmax).
    axes: dict[str, TargetAxis] = {
        TARGET_BACKBONE: TargetAxis(
            TEXT_SYMBOL, TEXT_SYMBOL, sym_max, {0: (1, 0)}, PRESERVED_OP_PREFIXES_WITH_ATTENTION
        ),
        TARGET_TEXT_PROJ: TargetAxis(
            TEXT_SYMBOL, TEXT_SYMBOL, sym_max, {0: (1, 0)}, PRESERVED_OP_PREFIXES_WITH_ATTENTION
        ),
        TARGET_CAPTION_PROJ: TargetAxis(
            TEXT_SYMBOL, TEXT_SYMBOL, sym_max, {0: (1, 0)}, PRESERVED_OP_PREFIXES_WITH_ATTENTION
        ),
        TARGET_SPEAKER: TargetAxis(
            SPEAKER_SYMBOL,
            SPEAKER_SYMBOL,
            speaker_max,
            {0: (1, 0)},
            PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        ),
        TARGET_DURATION: TargetAxis(
            TEXT_SYMBOL, TEXT_SYMBOL, sym_max, {0: (1, 0)}, PRESERVED_OP_PREFIXES_WITH_ATTENTION
        ),
        # `mask` sits after `t_embed` (RF) or after `delta_embed` (MeanFlow) — its position is
        # read from the step's own input names.
        TARGET_DIT: TargetAxis(
            DIT_TORCH_DIM,
            DIT_SYMBOL,
            dit_max,
            {0: (1, 0), dit_heads.index("mask"): (3, dit_context_total)},
            PRESERVED_OP_PREFIXES,
        ),
        TARGET_DIT_CONTEXT: TargetAxis(None, None, 0, {}, PRESERVED_OP_PREFIXES),
    }
    written: dict[str, Any] = {}
    for target in targets:
        target_dir = out_dir / target
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        by_case = graph_args[target]
        longest = max(by_case, key=lambda name: next(iter(by_case[name].values())).shape[1])
        # MUST: the artifacts are written to a per-target staging slot and put in place once
        # complete (nothing goes into the final place before the checks — the same discipline as
        # `export_series` on the dacvae side; the primitive is
        # `karume.artifacts.staged_publication`).
        with staged_publication(target_dir) as staged:
            # The writer creates the directory slot (the primitive does not create it — it only
            # hands over a path).
            staged.mkdir()
            exported, tensors = export_ir(
                graphs[target],
                tuple(by_case[longest].values()),
                axis=axes[target],
                # Only the DiT's boundary gets its IR names renamed to the contract's spelling
                # ({@link name_boundary}).
                input_names=dit_graphs.dit.input_names if target == TARGET_DIT else None,
                output_names=(
                    dit_graphs.context.output_names if target == TARGET_DIT_CONTEXT else None
                ),
            )
            graph = publish_model(
                staged / MODEL_FILE,
                exported,
                tensors,
                provenance=PROVENANCE,
                # The graph name is **the component name** (= the key in the weights of
                # karume.json). It is spelled differently from the directory name (`caption-proj`
                # → `caption_proj`), so it goes through the reverse lookup table (container-v1
                # §2.1).
                graph_name=IRODORI_SERIES_ROLES[target],
                weight_dtype=dtype,
                weight_scales=target_scales(target, graphs[target], quantized.scales),
                weight_dtype_overrides=target_weight_dtypes(
                    target, graphs[target], quantized.overrides
                ),
            )
            io_files = _write_io(graph, by_case, pristine[target], staged)
            # The record of the calibration conditions (only the 2 DiT graphs of i4). **The stdout
            # summary is for people to read**; the machine matching relies on
            # `calib_provenance.json`.
            calib_provenance = _write_calib_provenance(dtype, calib_plan, target, staged)
        written[target] = {
            **_graph_summary(graph, target_dir / MODEL_FILE),
            "io": io_files,
            "calib_provenance": calib_provenance,
        }
    return {
        "dir": str(out_dir),
        "dtype": dtype,
        "fake_quant": quantized.reports,
        "targets": written,
        "case_lengths": {name: int(ids.shape[1]) for name, _kind, ids in cases},
        "speaker_case_lengths": {name: int(latent.shape[1]) for name, latent in speaker_cases},
        "speaker_sym_max": speaker_max,
        "dit_case_lengths": {name: int(args["x_t"].shape[1]) for name, args in dit_inputs.items()},
        "dit_sym_max": dit_max,
        "dit_context_total": dit_context_total,
        "dit_context_cases": context_owner,
        "flow_parameterization": flow,
        "dit_uncond_divergence": dit_divergence,
        **({} if dit_delta_divergence is None else {"dit_delta_divergence": dit_delta_divergence}),
        "rope_buffers_lifted": rope_buffers,
        "static_scheme_max_abs": {k: float(f"{v:.3e}") for k, v in static_evidence.items()},
        "no_reference_max_abs": no_reference_max_abs,
        "duration_aux_inert_max_abs": aux_inert,
        "eager_equivalence_max_abs": equivalence,
        "dit_split_equivalence_max_abs": split_equivalence,
        "projector_divergence": sanity,
        "norm_divergence": round(norm_divergence, 5),
    }


def default_out_root(model_dir: Path, dtype: str = "f32") -> Path:
    """The default place for the artifacts (`outputs/series/irodori-<directory name of the real
    weights>{,-f16,-i8,-i4}/`).

    `export_series` digs one more level of per-target subdirectories (the same form as
    `export_sbv2.py`).

    MUST: a separate directory per dtype (ADR 0018 / 0019 / 0027) — sharing one would silently apply
    the f32 series' net (per-target tolerances derived from measurements) to the compressed assets.
    The spelling matches `irodori_series_name` of `karume.dist` + the dtype suffix (writer and
    reader build from the same word).
    """
    suffix = "" if dtype == "f32" else f"-{dtype}"
    return SERIES_ROOT / f"irodori-{model_dir.name}{suffix}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: the series of the --dtype —"
        " outputs/series/irodori-<directory name of --model-dir>{,-f16,-i8,-i4}/)",
    )
    parser.add_argument("--sym-max", type=int, default=SYM_MAX)
    parser.add_argument(
        "--dtype",
        choices=WEIGHT_DTYPES,
        default="f32",
        help="storage dtype of the weights (f16 / i8 / i4 fake-quant first and then store only the"
        " eligible slots compressed — ADR 0018 / 0019 / 0027 / 0050 / 0069. i4 is a series of the"
        " DiT's 2 graphs only (dit / dit-context), GPTQ-calibrated by default. **emit-only**)",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=TARGETS,
        help="narrow **only the writing** (repeatable; default: everything for the --dtype). The"
        " prerequisite assets (the real latent of step 5) and the reference computations (the"
        " torch forwards of speaker / duration / DiT) always run in full — the run time is the"
        " same even for 1 target",
    )
    parser.add_argument(
        "--calib-steps",
        type=int,
        default=None,
        help="the number of steps per case that i4 calibration uses (default = every step of the"
        " reference loop). The reference loop is cut short once enough is captured, so lower this"
        " for a reduced smoke run. The value used is recorded in calib_provenance.json",
    )
    parser.add_argument(
        "--no-calib",
        action="store_true",
        help="round i4 with plain RTN without calibration (an explicit opt-out for smoke runs —"
        " **never used for distribution assets**; the assembly side refuses it by name through"
        " calib_provenance.json)",
    )
    args = parser.parse_args(argv)
    allowed = DTYPE_TARGETS[args.dtype]
    targets = tuple(args.target) if args.target else allowed
    refused = sorted(set(targets) - set(allowed))
    if refused:
        parser.error(
            f"--dtype {args.dtype} writes only {list(allowed)} (given {refused})"
            " — a series the distribution tables do not pick up is not written silently"
        )
    if args.dtype != "i4" and (args.no_calib or args.calib_steps is not None):
        parser.error(f"--calib-steps / --no-calib only apply to --dtype i4 (given {args.dtype})")
    # An upper bound is needed because an excess **fails with another diagnosis after running to
    # the end**: the cut-off sentinel only fires at `len(batches) >= steps`
    # (`irodori.calib.capture_stage_batches`), so a value beyond the full reference loop never
    # fires, and after denoising one whole case with the real weights it fails with a diagnosis
    # **about the wrong subject**: "the spelling of upstream's `sample_euler_rf_cfg` disagrees with
    # the script's assumption". The same knob of `irodori.measure_quant` has had this form (a range
    # check on both sides) from the start.
    from .pipeline_ref import NUM_STEPS

    # The longest reference loop (RF's 40 Euler steps) bounds it here without reading the
    # checkpoint; a MeanFlow checkpoint's shorter loop (4 steps — ADR 0150) is checked by
    # `_calib_plan` as soon as the config is read, before any rounding.
    if args.calib_steps is not None and not 1 <= args.calib_steps <= NUM_STEPS:
        parser.error(
            f"--calib-steps must be 1 to {NUM_STEPS} (the step count of the reference loop;"
            f" given {args.calib_steps})"
        )
    out_dir = default_out_root(args.model_dir, args.dtype) if args.out is None else args.out
    summary = export_series(
        args.model_dir,
        args.source_dir,
        out_dir,
        targets=targets,
        sym_max=args.sym_max,
        dtype=args.dtype,
        calib_steps=args.calib_steps,
        no_calib=args.no_calib,
    )
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
