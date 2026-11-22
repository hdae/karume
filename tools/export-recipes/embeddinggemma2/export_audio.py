"""Script that writes the audio tower of EmbeddingGemma 2 (an audio's log-mel features → its soft
tokens) to an IR v2 container + golden io.

    uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_audio
    uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_audio --dtype bf16

transformers is pinned to 5.23.0, as for the text tower ({@link embeddinggemma2.export}).

## Source

The pinned snapshot of `google/embeddinggemma-2` ({@link embeddinggemma2.sources.PIN}). Of
`model.safetensors` only the 751 `audio_tower.*` tensors and `embed_audio.embedding_projection` are
read, loaded strictly into upstream's `Gemma4AudioModel` (the class `audio_config` names,
`model_type` `gemma4_audio` — Gemma 4 E2B / E4B's audio tower, unchanged) and
`EmbeddingGemma2MultimodalEmbedder` ({@link load_tower}); the weights are bf16 and upcast to f32
(exact). The goldens' frames come from upstream's `Gemma4AudioFeatureExtractor` as the snapshot's
`preprocessor_config.json` configures it ({@link load_feature_extractor} checks every setting).

## The graph and its io contract

    features    [1, 4n, 128] f32   the audio's valid log-mel frames, then zero frames up to 4n
    frame_mask  [1, 4n]      f32   1 on the valid frames, 0 on the padding
    → soft      [1, n, 512]  f32   the audio's soft tokens, ready to be soft rows of the text graph

`n` is the audio's soft-token count, the graph's one symbol (1 to {@link SOFT_TOKENS_MAX} = 750).
An audio of `L` valid frames has `n = ceil(L / 4)` soft tokens (two stride-2 convolutions, each
`ceil(t / 2)` — upstream's processor counts the same from the mask), so `4n − 3 ≤ L ≤ 4n`: the host
passes the `L` frames, appends `4n − L` zero frames (0 to 3) and sets the mask to 1 on the first `L`
— upstream's own `input_features_mask`, as f32. Every output token is valid (upstream keeps the same
`n` rows, `audio_features[attention_mask]`). An audio with no valid frame (at most 160 samples)
has no soft token and no tower run.

**The frame mask is needed exactly when `L ≡ 2 (mod 4)`.** Upstream zeroes the masked frames before
each subsampling convolution (`x · mask`, then `mask[:, ::2]` for the second). At `L = 4n − 2` the
first convolution's last output frame (`2n − 1`) reads the last real frame at its left tap, so it is
not zero, and upstream masks it before the second convolution reads it into the last token (without
the mask that token moves by 4.2 on `edge`). At the other residues the frames past `L` are zeros
anyway and the mask changes no bit (checked on every golden). Every stage after the subsampling is
causal (the attention sees itself and the 11 tokens before it, the depthwise convolution is left
padded), so no padding frame reaches an earlier token. No frame count avoids the mask with exactly
`n` outputs: `L = 4n` needs all `4n` frames, while `L = 4n − 2` is exact only with `4n − 2` (the
first convolution's own zero padding then stands for the masked frame). Frame counts of the form
`4s + 2` would do without a mask — the host pads to the next such count — but then the output holds
one token more than the audio's for `L ≡ 0, 3 (mod 4)`, which the host drops; the mask keeps the
output at upstream's `n` rows.

The frames are upstream's feature extractor's (`input_features`, f32): 16 kHz mono f32 samples
(the extractor does not resample), truncated at {@link MAX_SAMPLES} = 480,000 (30 s), zero-padded on
the right to a multiple of 128 samples, 160 zeros prepended (semicausal), frames of 321 samples by a
hop of 160 (the first 320 used — preemphasis 0), times a periodic Hann window of 320 (f32),
`rfft` of length 512, the magnitude (not the power), an HTK mel bank of 128 triangles from 0 to
8 kHz without normalisation (f64, filter 0 all zero), `log(mel + 1e-3)`, no per-bin normalisation
(numpy computes the window and the FFT in f32 — complex64 — for f32 samples, the mel product and
the log in f64, and the result is cast to f32);
frame `i` is valid iff sample `160 · i + 160` of the original audio exists (its window's last
sample is real), and an invalid frame is set to zeros. So `s` samples have `L = max(0, ceil(s / 160)
− 1)` valid frames ({@link embeddinggemma2.waveforms.valid_frames}): 1 s → 99 frames → 25 soft
tokens, 30 s → 2,999 → 750.

## What the wrapper changes ({@link AudioTowerWrapper})

Upstream's modules run as they are — the subsampling's two convolutions with their layer norms and
ReLU, `input_proj_linear`, the 12 conformer layers (the two half-step feed-forward blocks, the
norms, the light convolution with its GLU and causal depthwise convolution of width 5, every
`±1e10` gradient clamp), `output_proj` and `embed_audio` (scale-free RMSNorm → linear 1536 → 512) —
except three forms the IR has no op for, each rewritten without a new op (the layers are a copy;
upstream's own modules make the goldens):

- **The clipped linears' bounds.** Upstream clamps by four scalar tensors per linear
  (`clamp.Tensor`); {@link ClippedLinear} reads them as Python floats (the checkpoint's bf16
  values, exact) — the same clamps through `clamp` with scalar bounds.
- **The chunked local attention.** Upstream cuts the sequence into blocks of 12 queries, unfolds a
  24-key context per block (rank-5 tensors, `unfold`), adds the relative-position term through a
  pad-and-view shift and masks every key but the query and the 11 before it
  (`sliding_window_mask_function((12, 0))`). {@link WindowAttention} computes the same scores per
  query over exactly those {@link WINDOW} = 12 keys: k and v are copied into 12 shifted rows by one
  depthwise `conv1d` whose weight is one-hot ({@link window_selector} — exact for finite values;
  one tensor shared by the 24 copies), the content term is a `bmm` of each query with its window,
  the relative term a `bmm` with the relative keys `relative_k_proj(position table)` folded at
  export time (window column `j` has distance `11 − j`, upstream's row `j + 1`), then upstream's
  soft cap `50 · tanh(x / 50)`, an additive `−1e9` on the window columns before the first token
  ({@link window_bias}, built from `arange` of the symbol so it folds into one `[750, 12]` constant
  cut by `sym_prefix_slice`), the softmax over the 12 and a `bmm` with the values. The masked
  context columns upstream also carries weigh exactly 0 after its softmax, so nothing else changes.
- **The constants.** `softplus(per_dim_scale)` and the relative position table are computed at
  export time by upstream's own code (`cos` is not a run-time op).

The subsampling's frame mask is upstream's: the frames times the mask before the first
convolution, every other mask value (`mask[:, ::2]`, a reshape and a slice) times its output before
the second.

## Checks before anything is published

- {@link compare_with_upstream} — on every golden waveform, the wrapper's soft tokens against
  upstream's forward on the extractor's frames with its mask (the golden) and on the graph's own
  frames, within {@link UPSTREAM_TOLERANCE}; and the wrapper with an all-ones mask, bit-identical
  unless `L ≡ 2 (mod 4)` and far off there. The goldens cover every residue and the largest count.
- {@link assert_ir_form} — the two inputs on `4n` and the one output `[1, n, 512]`, one symbol, no
  state, no fused `attention` and no `where`, 12 softmaxes on `[8, n, 12]`, 24 window copies over
  one selector and 12 depthwise convolutions of width 5, one `sym_prefix_slice` of the `[750, 12]`
  bias, the two stride-2 convolutions.
- {@link assert_storage_form} — `f32`: every initializer f32; `bf16`: every weight slot bf16 (the
  linears, the convolutions and the selector, whose 0 and 1 are bf16 values), the rest f32.

## Storage

`--dtype f32` (the default, {@link AUDIO_SERIES}) and `--dtype bf16` (`<series>-bf16` — upstream's
own dtype for the weights): every linear and convolution weight bf16 (nothing rounded — the
checkpoint is bf16), the norms, biases and folded constants f32. The goldens are the same for both.
Activations are f32 throughout, as the text tower's.

## Goldens

Upstream's feature extractor and upstream's modules over the real weights in f32, on the synthetic
waveforms of {@link embeddinggemma2.waveforms.WAVEFORMS} (every residue of `L` mod 4, 1 s, 10 s and
the 30 s maximum): `io.<waveform>.safetensors` holds `input.features` / `input.frame_mask` (the
graph's frames) and `output.0` (upstream's soft tokens `[1, n, 512]` from the extractor's frames);
`reference.json` records per waveform its length, the SHA-256 of its PCM and of its valid frames,
the extractor's frame count, the valid frames, the soft-token count and the measured differences.

## Output layout

    outputs/series/embeddinggemma-2-audio/model-NNNNN-of-NNNNN.krm  the container (graph `audio`)
    outputs/series/embeddinggemma-2-audio/io.<waveform>.safetensors the goldens
    outputs/series/embeddinggemma-2-audio/reference.json            provenance and measurements
    outputs/series/embeddinggemma-2-audio-bf16/…                    the same with `--dtype bf16`
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim
from torch.nn import functional

from _shared.paths import SERIES_ROOT
from embeddinggemma.export import INPUT_PREFIX, IO_PREFIX, IO_SUFFIX, MODEL_FILE, OUTPUT_PREFIX
from embeddinggemma2 import sources
from embeddinggemma2.export import (
    LICENSE,
    WHERE_OP,
    EmbeddingGemma2ExportError,
    assert_transformers_version,
)
from embeddinggemma2.waveforms import (
    PCM_SCALE,
    SAMPLE_RATE,
    WAVEFORMS,
    SyntheticWaveform,
    soft_tokens,
    valid_frames,
)
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import normalize_boundary_tensor
from karume.emit import eligible_compressed_initializers
from karume.ir import IrGraph
from karume.ops import ATTENTION_OP
from karume.pipeline import export_module, publish_model
from karume.shapes import declared_shape

#: The series this script writes.
AUDIO_SERIES = "embeddinggemma-2-audio"

#: The storage dtypes `--dtype` takes.
WEIGHT_DTYPES = ("f32", "bf16")

#: One series per storage (ADR 0019).
DEFAULT_OUT_DIRS: Mapping[str, Path] = {
    "f32": SERIES_ROOT / AUDIO_SERIES,
    "bf16": SERIES_ROOT / f"{AUDIO_SERIES}-bf16",
}

#: The graph name (the component name a distribution will give the tower).
GRAPH_NAME = "audio"

#: The graph's inputs, the symbol (the soft-token count) and the output's width.
FEATURES = "features"
FRAME_MASK = "frame_mask"
GRAPH_INPUTS = (FEATURES, FRAME_MASK)
SYMBOL = "n"
SOFT_WIDTH = 512

#: The mel bins of a frame (`feature_size` of the feature extractor, checked).
MEL_BINS = 128

#: The mel frames per soft token: the two stride-2 convolutions of the subsampling.
FRAMES_PER_TOKEN = 4

#: The snapshot files the script reads (each checked against the pin first). The feature
#: extractor reads `preprocessor_config.json`; `processor_config.json` holds the same settings.
READ_FILES: tuple[str, ...] = (
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
)

#: The checkpoint's key prefixes of the tower and of its projector.
TOWER_PREFIX = "audio_tower."
PROJECTION_PREFIX = "embed_audio."

#: The feature extractor's settings the graph and the host's frame layout are built for (each
#: checked against the loaded extractor — {@link load_feature_extractor}).
EXTRACTOR_SETTINGS: Mapping[str, Any] = {
    "sampling_rate": SAMPLE_RATE,
    "feature_size": MEL_BINS,
    "frame_length": 320,
    "hop_length": 160,
    "fft_length": 512,
    "min_frequency": 0.0,
    "max_frequency": 8000.0,
    "preemphasis": 0.0,
    "dither": 0.0,
    "input_scale_factor": 1.0,
    "mel_floor": 1e-3,
    "per_bin_mean": None,
    "per_bin_stddev": None,
    "fft_overdrive": False,
}

#: The longest audio upstream's feature extractor reads: its `max_length` default, 30 s at 16 kHz
#: (a longer one is truncated silently; checked against the signature).
MAX_SAMPLES = 480_000

#: The bound of the symbol: the soft tokens of {@link MAX_SAMPLES} samples (2,999 frames → 750).
SOFT_TOKENS_MAX = soft_tokens(valid_frames(MAX_SAMPLES))

#: The keys each query sees: itself and the 11 before it (`attention_context_left − 1`, nothing on
#: the right — upstream's `sliding_window_mask_function((12, 0))`; module doc).
WINDOW = 12

#: The attention implementation of upstream's own forward (the audio tower supports eager only).
UPSTREAM_ATTENTION = "eager"

#: The bound of "equal to f32 noise" between the wrapper and upstream (max abs of the soft tokens,
#: which reach 5.5 to 6.1). Measured on 2026-11-22 over the six golden waveforms: the wrapper
#: against upstream 5.1e-6 to 1.01e-4 (`tones`), upstream on the graph's frames against upstream on
#: the extractor's 0 (bit-identical). Both sit in the tower's own f32 noise — in f64 (upstream's
#: f32 casts lifted) the wrapper equals upstream to 1.3e-12, while upstream's f32 run is up to
#: 3.0e-4 from that f64 run (`long`) and the wrapper's up to 3.4e-4 (`tones`). The bound is three
#: times the worst wrapper-to-upstream difference, the size of upstream's own f32 error. The
#: breakages it guards against are far above it: the frame mask dropped moves an `L ≡ 2 (mod 4)`
#: audio's last token by 4.2 (`edge`), a window of 13 keys by about 0.4 (random frames, torch —
#: the investigation before this script).
UPSTREAM_TOLERANCE = 3e-4

#: The trace's example soft-token count (inside the symbol's range, far from 0 / 1).
EXAMPLE_TOKENS = 100

#: The buffer name of the window's one-hot selector (one tensor shared by every attention).
WINDOW_SELECTOR = "window_selector"

#: The golden file names.
REFERENCE_FILE = "reference.json"


class AudioExportError(EmbeddingGemma2ExportError):
    """An input of the script is not what the tower graph is built for (fails before writing)."""


# ---- The upstream modules ---------------------------------------------------------------------


@dataclass(frozen=True)
class Tower:
    """Upstream's audio tower and its projector over the real weights in f32, and their config."""

    tower: nn.Module
    embedder: nn.Module
    config: Any


def load_config(directory: Path) -> Any:
    """The composite config (`EmbeddingGemma2Config`), its audio part checked against what the
    graph is built for."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(directory)
    audio = config.audio_config
    facts = {
        "model_type": (audio.model_type, "gemma4_audio"),
        "attention_context_left": (int(audio.attention_context_left), WINDOW + 1),
        "attention_context_right": (int(audio.attention_context_right), 0),
        "subsampling_conv_channels": (list(audio.subsampling_conv_channels), [MEL_BINS, 32]),
        "use_clipped_linears": (bool(audio.use_clipped_linears), True),
        "text hidden_size": (int(config.text_config.hidden_size), SOFT_WIDTH),
    }
    wrong = {key: found for key, (found, expected) in facts.items() if found != expected}
    if wrong:
        raise AudioExportError(f"the audio config is not the one the graph is built for: {wrong}")
    return config


def checkpoint_state(model_file: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """`(tower state, projector state)` under upstream's module keys, upcast to f32 (exact)."""
    tower: dict[str, torch.Tensor] = {}
    projector: dict[str, torch.Tensor] = {}
    with safe_open(str(model_file), framework="pt") as handle:
        for key in handle.keys():  # noqa: SIM118 — safe_open is not a mapping
            for prefix, state in ((TOWER_PREFIX, tower), (PROJECTION_PREFIX, projector)):
                if key.startswith(prefix):
                    state[key[len(prefix) :]] = handle.get_tensor(key).to(torch.float32, copy=True)
    if not tower or not projector:
        raise AudioExportError(
            f"{model_file} lacks '{TOWER_PREFIX}*' or '{PROJECTION_PREFIX}*' tensors"
        )
    return tower, projector


def load_tower(directory: Path) -> Tower:
    """Upstream's `Gemma4AudioModel` and `EmbeddingGemma2MultimodalEmbedder` with the real weights
    in f32.

    MUST: strict loads — a key renamed upstream would otherwise leave a random tensor in the graph.
    """
    import transformers
    from transformers import AutoModel
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
        EmbeddingGemma2MultimodalEmbedder,
    )

    assert_transformers_version(transformers.__version__)
    config = load_config(directory)
    config.audio_config._attn_implementation = UPSTREAM_ATTENTION
    tower_state, projector_state = checkpoint_state(directory / "model.safetensors")
    tower = AutoModel.from_config(config.audio_config, dtype=torch.float32)
    if type(tower).__name__ != "Gemma4AudioModel":
        raise AudioExportError(f"audio_config builds {type(tower).__name__}")
    tower.load_state_dict(tower_state, strict=True)
    embedder = EmbeddingGemma2MultimodalEmbedder(config.audio_config, config.text_config)
    embedder.load_state_dict(projector_state, strict=True)
    tower = tower.to(torch.float32).eval().requires_grad_(False)
    embedder = embedder.to(torch.float32).eval().requires_grad_(False)
    return Tower(tower=tower, embedder=embedder, config=config)


def upstream_soft_tokens(
    tower: Tower, features: torch.Tensor, frame_mask: torch.Tensor
) -> torch.Tensor:
    """Upstream's soft tokens `[m, 512]` of the frames given (`get_audio_features` of the
    composite — the tower, then `embed_audio` — and the valid rows the composite keeps)."""
    with torch.no_grad():
        output = tower.tower(features, frame_mask.to(torch.bool), return_dict=True)
        soft = tower.embedder(inputs_embeds=output.last_hidden_state)
    return soft[0][output.attention_mask[0]]


# ---- The host's frames ------------------------------------------------------------------------


@dataclass(frozen=True)
class HostFrames:
    """One audio as the graph takes it: its valid mel frames zero-padded to `4n`, and the mask."""

    #: `[1, 4n, 128]` f32.
    features: torch.Tensor
    #: `[1, 4n]` f32, 1 on the first `valid` frames and 0 after.
    frame_mask: torch.Tensor
    #: The valid frames `L` (`4n − 3 ≤ L ≤ 4n`).
    valid: int

    @property
    def soft_tokens(self) -> int:
        return int(self.features.shape[1]) // FRAMES_PER_TOKEN


def host_frames(features: torch.Tensor, frame_mask: torch.Tensor) -> HostFrames:
    """The feature extractor's frames of one audio (`[1, F, 128]`, mask `[1, F]`) → the graph's.

    MUST: the valid frames are a prefix and the frames after them zeros, as upstream's extractor
    gives one audio — frames in any other form are not what the layout is defined on. An audio
    with no valid frame has no soft token (the host runs no tower for it).
    """
    valid_mask = frame_mask[0].to(torch.bool)
    valid = int(valid_mask.sum())
    if not bool(valid_mask[:valid].all()):
        raise AudioExportError("the extractor's valid frames are not the first frames")
    if bool(features[0, valid:].any()):
        raise AudioExportError("the extractor's frames after the valid ones are not zeros")
    if valid == 0:
        raise AudioExportError("an audio with no valid frame has no soft token")
    tokens = soft_tokens(valid)
    if tokens > SOFT_TOKENS_MAX:
        raise AudioExportError(f"{tokens} soft tokens, more than the graph's {SOFT_TOKENS_MAX}")
    frames = FRAMES_PER_TOKEN * tokens
    padded = torch.zeros(1, frames, MEL_BINS, dtype=torch.float32)
    padded[0, :valid] = features[0, :valid]
    mask = torch.zeros(1, frames, dtype=torch.float32)
    mask[0, :valid] = 1.0
    return HostFrames(features=padded, frame_mask=mask, valid=valid)


# ---- The wrapper ------------------------------------------------------------------------------


class ClippedLinear(nn.Module):
    """Upstream's `Gemma4ClippableLinear` with its four bounds read as Python floats: the same
    clamps through `clamp` with scalar bounds (the IR has no clamp by tensor bounds).

    MUST: the bounds are finite — the checkpoint's are (bf16 values, exact in f32 and as floats).
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.linear = module.linear
        bounds = [float(getattr(module, name)) for name in ("input_min", "input_max")]
        bounds += [float(getattr(module, name)) for name in ("output_min", "output_max")]
        if not all(math.isfinite(bound) for bound in bounds):
            raise AudioExportError(f"a clipped linear has non-finite bounds {bounds}")
        self.input_min, self.input_max, self.output_min, self.output_max = bounds

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
        return torch.clamp(self.linear(hidden_states), self.output_min, self.output_max)


def window_selector(channels: int) -> torch.Tensor:
    """The weight `[channels · WINDOW, 1, WINDOW]` of a depthwise `conv1d` whose output channel
    `c · WINDOW + j` is input channel `c` shifted by `j`: over rows left-padded by `WINDOW − 1`,
    column `t` of copy `j` is row `t − (WINDOW − 1) + j` — the `WINDOW` keys before and at `t`
    (one-hot taps, so the copies are exact)."""
    selector = torch.zeros(channels * WINDOW, 1, WINDOW, dtype=torch.float32)
    for shift in range(WINDOW):
        selector[shift::WINDOW, 0, shift] = 1.0
    return selector


def window_bias(tokens: int, invalid: float) -> torch.Tensor:
    """`[tokens, WINDOW]`: `invalid` where window column `j` of query `t` is before the first token
    (`t − (WINDOW − 1) + j < 0`), 0 elsewhere — the only keys upstream's mask removes once every
    token is valid."""
    index = torch.arange(tokens).unsqueeze(1) + torch.arange(WINDOW).unsqueeze(0) - (WINDOW - 1)
    return (index < 0).to(torch.float32) * invalid


class WindowAttention(nn.Module):
    """Upstream's chunked local attention (`Gemma4AudioAttention`) as a window of {@link WINDOW}
    keys per query (module doc, "What the wrapper changes")."""

    def __init__(
        self, attention: nn.Module, relative_positions: torch.Tensor, selector: torch.Tensor
    ) -> None:
        super().__init__()
        self.q_proj = ClippedLinear(attention.q_proj)
        self.k_proj = ClippedLinear(attention.k_proj)
        self.v_proj = ClippedLinear(attention.v_proj)
        self.post = ClippedLinear(attention.post)
        self.heads = int(attention.num_heads)
        self.head_dim = int(attention.head_dim)
        self.q_scale = float(attention.q_scale)
        self.k_scale = float(attention.k_scale)
        self.soft_cap = float(attention.attention_logits_soft_cap)
        with torch.no_grad():
            self.register_buffer(
                "query_scale", functional.softplus(attention.per_dim_scale.detach()).clone()
            )
            # Upstream's relative keys `[WINDOW + 1, H · D]`: row p holds distance WINDOW − p.
            # Window column j has distance WINDOW − 1 − j, so it reads row j + 1 (row 0, distance
            # WINDOW, is always masked upstream).
            relative = attention.relative_k_proj(relative_positions)
            relative = relative.view(-1, self.heads, self.head_dim).permute(1, 2, 0)
            if int(relative.shape[-1]) != WINDOW + 1:
                raise AudioExportError(f"{relative.shape[-1]} relative positions, not {WINDOW + 1}")
            self.register_buffer("relative_keys", relative[:, :, 1:].contiguous())
        self.register_buffer(WINDOW_SELECTOR, selector)

    def window(self, rows: torch.Tensor) -> torch.Tensor:
        """`[1, n, H · D]` → `[H, n, WINDOW, D]`: each token's window of rows (zeros before the
        first)."""
        tokens = rows.shape[1]
        padded = functional.pad(rows.transpose(1, 2), (WINDOW - 1, 0))
        copies = functional.conv1d(padded, self.window_selector, groups=self.heads * self.head_dim)
        return copies.view(self.heads, self.head_dim, WINDOW, tokens).permute(0, 3, 2, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Any = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        # `attention_mask` is the wrapper's window bias `[n, WINDOW]`; `position_embeddings` is
        # upstream's relative table, folded into `relative_keys` here.
        del position_embeddings
        if attention_mask is None:
            raise AudioExportError("the window attention runs with its window bias only")
        heads, head_dim = self.heads, self.head_dim
        tokens = hidden_states.shape[1]
        queries = self.q_proj(hidden_states).view(tokens, heads, head_dim)
        queries = (queries * self.q_scale * self.query_scale).permute(1, 0, 2)
        keys = self.window(self.k_proj(hidden_states) * self.k_scale)
        values = self.window(self.v_proj(hidden_states))
        content = torch.bmm(
            queries.reshape(heads * tokens, 1, head_dim),
            keys.reshape(heads * tokens, WINDOW, head_dim).transpose(1, 2),
        ).view(heads, tokens, WINDOW)
        position = torch.bmm(queries, self.relative_keys)
        scores = torch.tanh((content + position) / self.soft_cap) * self.soft_cap
        weights = functional.softmax(scores + attention_mask, dim=-1)
        output = torch.bmm(
            weights.reshape(heads * tokens, 1, WINDOW),
            values.reshape(heads * tokens, WINDOW, head_dim),
        ).view(heads, tokens, head_dim)
        return self.post(output.permute(1, 0, 2).reshape(1, tokens, heads * head_dim)), None


def _replace_clipped_linears(module: nn.Module) -> int:
    """Swap every `Gemma4ClippableLinear` under `module` for a {@link ClippedLinear}; returns the
    count."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

    swapped = 0
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, Gemma4ClippableLinear):
                setattr(parent, name, ClippedLinear(child))
                swapped += 1
    return swapped


class AudioTowerWrapper(nn.Module):
    """`(features [1, 4n, 128], frame_mask [1, 4n]) → soft [1, n, 512]`: upstream's audio tower and
    `embed_audio` over the host's frames (module doc).

    MUST: the graph inputs carry these names (the host binds by name).
    MUST: upstream's modules are not changed — the layers are a copy with their attention and
    clipped linears swapped (the goldens run upstream's own).
    """

    def __init__(self, tower: Tower) -> None:
        super().__init__()
        model = tower.tower
        subsampling = model.subsample_conv_projection
        self.layer0 = subsampling.layer0
        self.layer1 = subsampling.layer1
        self.input_proj_linear = subsampling.input_proj_linear
        hidden = int(model.config.hidden_size)
        with torch.no_grad():
            # Upstream's table in the tower's dtype, as its forward makes it.
            dtype = model.output_proj.weight.dtype
            relative_positions = model.rel_pos_enc(torch.zeros(1, 1, hidden, dtype=dtype))
        selector = window_selector(hidden)
        self.layers = copy.deepcopy(model.layers)
        for layer in self.layers:
            layer.self_attn = WindowAttention(layer.self_attn, relative_positions, selector)
        _replace_clipped_linears(self.layers)
        self.invalid = float(model.config.attention_invalid_logits_value)
        self.output_proj = model.output_proj
        self.embed_audio = tower.embedder

    def forward(self, features: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        # Upstream's subsampling with its frame mask: zero the masked frames before each
        # convolution; the second one reads every other frame's mask (upstream's `mask[:, ::2]`,
        # spelled as a reshape and a slice).
        hidden = (features * frame_mask.unsqueeze(-1)).unsqueeze(1)
        hidden, _ = self.layer0(hidden, None)
        half = frame_mask.reshape(1, frame_mask.shape[1] // 2, 2)[:, :, 0]
        hidden, _ = self.layer1(hidden * half[:, None, :, None], None)
        tokens = hidden.shape[2]
        hidden = self.input_proj_linear(hidden.permute(0, 2, 3, 1).reshape(1, tokens, -1))
        # Built from `arange` of the symbol, so the export folds it into one constant at the bound
        # cut by `sym_prefix_slice` (a buffer sliced by the symbol is not folded).
        bias = window_bias(tokens, self.invalid)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=bias, position_embeddings=None)
        return self.embed_audio(inputs_embeds=self.output_proj(hidden))


# ---- The feature extractor --------------------------------------------------------------------


def load_feature_extractor(directory: Path) -> Any:
    """Upstream's feature extractor as the snapshot configures it, its settings checked."""
    from transformers import AutoFeatureExtractor

    extractor = AutoFeatureExtractor.from_pretrained(directory)
    found: dict[str, Any] = {"class": type(extractor).__name__}
    for key in EXTRACTOR_SETTINGS:
        value = getattr(extractor, key)
        # `mel_floor` is held as a 0-d numpy array.
        found[key] = value.item() if isinstance(value, np.ndarray) else value
    defaults = inspect.signature(type(extractor).__call__).parameters
    found["max_length"] = defaults["max_length"].default
    expected = {
        "class": "Gemma4AudioFeatureExtractor",
        **EXTRACTOR_SETTINGS,
        "max_length": MAX_SAMPLES,
    }
    wrong = {key: value for key, value in found.items() if value != expected[key]}
    if wrong:
        raise AudioExportError(
            f"the feature extractor is not the one the graph is built for: {wrong}"
        )
    return extractor


@dataclass(frozen=True)
class ProcessedAudio:
    """One synthetic waveform through upstream's feature extractor."""

    waveform: SyntheticWaveform
    #: `[1, F, 128]` f32 and `[1, F]` bool, as the extractor gives them (F counts the padding).
    input_features: torch.Tensor
    input_features_mask: torch.Tensor
    frames: HostFrames

    @property
    def soft_tokens(self) -> int:
        return self.frames.soft_tokens


def process_audio(extractor: Any, waveform: SyntheticWaveform) -> ProcessedAudio:
    """Upstream's feature extractor on the waveform's f32 samples, handed over as a list of one —
    as upstream's processor does (`make_list_of_audio`; the extractor reads a bare 1-D array as
    one row of `samples` channels and fails)."""
    output = extractor([waveform.f32()], return_tensors="pt")
    features = output["input_features"]
    mask = output["input_features_mask"]
    if features.dtype != torch.float32 or mask.dtype != torch.bool:
        raise AudioExportError(
            f"{waveform.name}: the extractor gives {features.dtype} {mask.dtype}"
        )
    frames = host_frames(features, mask)
    if frames.valid != valid_frames(waveform.samples):
        raise AudioExportError(
            f"{waveform.name}: {frames.valid} valid frames, the rule says"
            f" {valid_frames(waveform.samples)}"
        )
    return ProcessedAudio(waveform, features, mask, frames)


# ---- Checks -----------------------------------------------------------------------------------


def _difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left, right = left.reshape(-1).double(), right.reshape(-1).double()
    return {
        "max_abs": float((left - right).abs().max()),
        "cosine": float(torch.dot(left, right) / (left.norm() * right.norm())),
        "max_abs_value": float(right.abs().max()),
    }


def compare_with_upstream(
    wrapper: AudioTowerWrapper, tower: Tower, audio: ProcessedAudio
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Upstream's soft tokens of the extractor's frames (the golden) and the measured differences:
    the wrapper against them and against upstream on the graph's own frames, upstream's two
    forwards against each other, and the wrapper with an all-ones mask (what the mask input
    guards).

    MUST: each of the first three within {@link UPSTREAM_TOLERANCE}. MUST: without the mask the
    wrapper is bit-identical unless `L ≡ 2 (mod 4)`, and then moves the last token far — the rule
    the io contract states.
    """
    frames = audio.frames
    golden = upstream_soft_tokens(tower, audio.input_features, audio.input_features_mask)
    same_frames = upstream_soft_tokens(tower, frames.features, frames.frame_mask)
    with torch.no_grad():
        ours = wrapper(frames.features, frames.frame_mask)[0]
        unmasked = wrapper(frames.features, torch.ones_like(frames.frame_mask))[0]
    expected = [frames.soft_tokens, SOFT_WIDTH]
    if list(golden.shape) != expected or ours.shape != golden.shape:
        raise AssertionError(
            f"{audio.waveform.name}: soft tokens {list(golden.shape)} / {list(ours.shape)}, not"
            f" {expected}"
        )
    measured: dict[str, Any] = {
        "wrapper_vs_upstream": _difference(ours, golden),
        "wrapper_vs_upstream_same_frames": _difference(ours, same_frames),
        "upstream_same_frames_vs_extractor_frames": _difference(same_frames, golden),
    }
    worst = max(entry["max_abs"] for entry in measured.values())
    if worst > UPSTREAM_TOLERANCE:
        raise AssertionError(
            f"{audio.waveform.name}: the wrapper differs from upstream's tower by {measured}"
            f" (> {UPSTREAM_TOLERANCE})"
        )
    measured["wrapper_without_mask"] = _difference(unmasked, ours)
    moved = measured["wrapper_without_mask"]["max_abs"]
    if frames.valid % FRAMES_PER_TOKEN == 2:
        if moved <= 100 * UPSTREAM_TOLERANCE:
            raise AssertionError(
                f"{audio.waveform.name}: L = {frames.valid} ≡ 2 (mod 4), yet the mask moves the"
                f" soft tokens by only {moved:.3e}"
            )
    elif not torch.equal(unmasked, ours):
        raise AssertionError(
            f"{audio.waveform.name}: L = {frames.valid}, yet the mask moves the soft tokens by"
            f" {moved:.3e}"
        )
    return golden, measured


def trace(wrapper: AudioTowerWrapper) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """Export with the soft-token count symbolic in `[1, SOFT_TOKENS_MAX]` and the frame axis `4n`
    (the graph does not depend on the example's values)."""
    tokens = Dim(SYMBOL, min=1, max=SOFT_TOKENS_MAX)
    frames = FRAMES_PER_TOKEN * tokens
    rows = EXAMPLE_TOKENS * FRAMES_PER_TOKEN
    return export_module(
        wrapper,
        (
            torch.zeros(1, rows, MEL_BINS, dtype=torch.float32),
            torch.ones(1, rows, dtype=torch.float32),
        ),
        dynamic_shapes=({1: frames}, {1: frames}),
        symbol_names=(SYMBOL,),
    )


def assert_ir_form(graph: IrGraph, config: Any) -> dict[str, Any]:
    """The traced form (module doc, "Checks"). Each property breaks while the numbers can still
    agree on the goldens: a fused attention or a `where` in the graph, a window of another width,
    copies of the window selector, a bias that is not the one prefix of one constant."""
    audio = config.audio_config
    frames = f"{FRAMES_PER_TOKEN}{SYMBOL}"
    expected_inputs = [[FEATURES, "f32", [1, frames, MEL_BINS]], [FRAME_MASK, "f32", [1, frames]]]
    found = [[spec.name, spec.dtype, list(spec.shape)] for spec in graph.inputs]
    if found != expected_inputs:
        raise AssertionError(f"graph inputs are {found}, not {expected_inputs}")
    if list(graph.symbols) != [SYMBOL] or graph.states:
        raise AssertionError(f"symbols {list(graph.symbols)} / states {sorted(graph.states)}")
    if len(graph.outputs) != 1:
        raise AssertionError(f"{len(graph.outputs)} outputs (the soft tokens are one)")
    output = graph.values[graph.outputs[0]]
    if output.dtype != "f32" or list(output.shape) != [1, SYMBOL, SOFT_WIDTH]:
        raise AssertionError(f"the output is {output.dtype} {list(output.shape)}")
    banned = [node.op for node in graph.nodes if node.op in (ATTENTION_OP, WHERE_OP)]
    if banned:
        raise AssertionError(f"{Counter(banned)} in the graph (the window form has neither)")
    layers = int(audio.num_hidden_layers)
    hidden = int(audio.hidden_size)
    heads = int(audio.num_attention_heads)
    softmaxes = [node for node in graph.nodes if node.op == "softmax"]
    softmax_shapes = [declared_shape(graph, node.ins[0]) for node in softmaxes]
    if softmax_shapes != [[heads, SYMBOL, WINDOW]] * layers:
        raise AssertionError(f"the softmaxes run on {softmax_shapes}, not {layers} × window")
    # Every conv1d is a window copy (k and v of each layer) or a layer's causal depthwise
    # convolution, both over rows left-padded by their kernel's width less one.
    depthwise = {"stride": 1, "padding": 0, "dilation": 1, "groups": hidden}
    kernel = int(audio.conv_kernel_size)
    window_form = (
        [1, hidden, f"{SYMBOL}+{WINDOW - 1}"],
        [hidden * WINDOW, 1, WINDOW],
        depthwise,
    )
    light_form = ([1, hidden, f"{SYMBOL}+{kernel - 1}"], [hidden, 1, kernel], depthwise)
    conv1d_forms: Counter[str] = Counter()
    windows = []
    for node in graph.nodes:
        if node.op != "conv1d":
            continue
        found_form = (
            declared_shape(graph, node.ins[0]),
            declared_shape(graph, node.ins[1]),
            node.attrs,
        )
        conv1d_forms[repr(found_form)] += 1
        if found_form == window_form:
            windows.append(node)
    expected_forms = Counter({repr(window_form): 2 * layers, repr(light_form): layers})
    selectors = {node.ins[1] for node in windows}
    if conv1d_forms != expected_forms or len(selectors) != 1:
        raise AssertionError(
            f"the conv1d nodes are {dict(conv1d_forms)} over {len(selectors)} selectors, not"
            f" {dict(expected_forms)} over one"
        )
    (selector,) = selectors
    if not graph.initializers[selector].tensor.endswith(f".self_attn.{WINDOW_SELECTOR}"):
        raise AssertionError(f"the window selector is {graph.initializers[selector].tensor}")
    prefixes = [node for node in graph.nodes if node.op == "sym_prefix_slice"]
    bias_shapes = [declared_shape(graph, node.ins[0]) for node in prefixes]
    if bias_shapes != [[SOFT_TOKENS_MAX, WINDOW]]:
        raise AssertionError(f"the prefix slices read {bias_shapes}, not one window bias")
    convolutions = [
        (declared_shape(graph, node.ins[0]), node.attrs)
        for node in graph.nodes
        if node.op == "conv2d"
    ]
    if len(convolutions) != 2 or any(
        attrs["stride"] != [2, 2] or attrs["padding"] != [1, 1] for _, attrs in convolutions
    ):
        raise AssertionError(f"the subsampling convolutions are {convolutions}")
    return {
        "inputs": found,
        "softmax_nodes": len(softmaxes),
        "window_copies": len(windows),
        "ops": dict(Counter(node.op for node in graph.nodes).most_common()),
    }


def assert_storage_form(graph: IrGraph, dtype: str) -> dict[str, int]:
    """The storage the writer committed: under `bf16`, every weight slot bf16 and the rest f32;
    under `f32`, every initializer f32 (or i32)."""
    if dtype not in WEIGHT_DTYPES:
        raise AudioExportError(f"storage {dtype!r} is not one of {WEIGHT_DTYPES}")
    eligible = eligible_compressed_initializers(graph)
    counts: dict[str, int] = {}
    wrong: list[str] = []
    for name, initializer in graph.initializers.items():
        stored = initializer.storage.dtype
        counts[stored] = counts.get(stored, 0) + 1
        expected = {"bf16"} if dtype == "bf16" and name in eligible else {"f32", "i32"}
        if stored not in expected:
            wrong.append(
                f"{initializer.tensor}: {stored} (expected {' / '.join(sorted(expected))})"
            )
    if wrong:
        raise AssertionError(f"the {dtype} storage form does not hold: {wrong[:4]}")
    return dict(sorted(counts.items()))


# ---- Writing ----------------------------------------------------------------------------------


def _sha256(array: np.ndarray | torch.Tensor) -> str:
    data = array.numpy() if isinstance(array, torch.Tensor) else array
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def write_io(graph: IrGraph, audio: ProcessedAudio, golden: torch.Tensor, out_dir: Path) -> str:
    """`io.<waveform>.safetensors`: the graph's two inputs and upstream's soft tokens
    `[1, n, 512]`."""
    args = {FEATURES: audio.frames.features, FRAME_MASK: audio.frames.frame_mask}
    tensors = {
        f"{INPUT_PREFIX}{spec.name}": normalize_boundary_tensor(
            args[spec.name], f"{audio.waveform.name}: input '{spec.name}'"
        )
        for spec in graph.inputs
    }
    tensors[f"{OUTPUT_PREFIX}0"] = normalize_boundary_tensor(
        golden.unsqueeze(0).contiguous(), f"{audio.waveform.name}: output 0"
    )
    path = out_dir / f"{IO_PREFIX}{audio.waveform.name}{IO_SUFFIX}"
    save_file(tensors, str(path))
    return path.name


def audio_record(audio: ProcessedAudio, measured: Mapping[str, Any]) -> dict[str, Any]:
    """What `reference.json` records of one golden waveform."""
    frames = audio.frames
    return {
        "name": audio.waveform.name,
        "why": audio.waveform.why,
        "samples": audio.waveform.samples,
        "pcm16Sha256": _sha256(audio.waveform.pcm16()),
        "extractorFrames": int(audio.input_features.shape[1]),
        "validFrames": frames.valid,
        "softTokens": frames.soft_tokens,
        "featuresSha256": _sha256(audio.input_features[0, : frames.valid].contiguous()),
        "measured": dict(measured),
    }


def export_series(
    directory: Path,
    out_dir: Path,
    *,
    dtype: str = "f32",
    waveforms: Collection[str] = tuple(WAVEFORMS),
) -> dict[str, Any]:
    """Write the tower's container and goldens, and return a summary (staged — nothing is placed
    unless every check passes)."""
    if dtype not in WEIGHT_DTYPES:
        raise AudioExportError(f"storage {dtype!r} is not one of {WEIGHT_DTYPES}")
    sources.assert_pinned(directory, READ_FILES)
    tower = load_tower(directory)
    extractor = load_feature_extractor(directory)
    wrapper = AudioTowerWrapper(tower).eval()
    processed = [process_audio(extractor, WAVEFORMS[name]) for name in waveforms]
    residues = {audio.frames.valid % FRAMES_PER_TOKEN for audio in processed}
    if residues != set(range(FRAMES_PER_TOKEN)):
        raise AudioExportError(f"the goldens cover the frame residues {sorted(residues)} only")
    if max(audio.soft_tokens for audio in processed) != SOFT_TOKENS_MAX:
        raise AudioExportError(f"no golden runs at the largest count {SOFT_TOKENS_MAX}")
    goldens = {
        audio.waveform.name: compare_with_upstream(wrapper, tower, audio) for audio in processed
    }
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    graph, tensors = trace(wrapper)
    form = assert_ir_form(graph, tower.config)
    with staged_publication(out_dir) as staged:
        staged.mkdir()
        graph = publish_model(
            staged / MODEL_FILE,
            graph,
            tensors,
            provenance=Provenance(license=LICENSE, upstream_revision=sources.PIN.revision),
            graph_name=GRAPH_NAME,
            weight_dtype=dtype,
        )
        storage = assert_storage_form(graph, dtype)
        written = [
            write_io(graph, audio, goldens[audio.waveform.name][0], staged) for audio in processed
        ]
        record = {
            "series": out_dir.name,
            "upstream": sources.PIN.label,
            "sampleRate": SAMPLE_RATE,
            "pcmScale": PCM_SCALE,
            "softTokensMax": SOFT_TOKENS_MAX,
            "window": WINDOW,
            "upstreamTolerance": UPSTREAM_TOLERANCE,
            "audios": [audio_record(audio, goldens[audio.waveform.name][1]) for audio in processed],
        }
        (staged / REFERENCE_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {
        "dir": str(out_dir),
        "nodes": len(graph.nodes),
        "initializers": len(graph.initializers),
        "parts": len(container_parts(out_dir / MODEL_FILE)),
        "model_bytes": sum(p.stat().st_size for p in container_parts(out_dir / MODEL_FILE)),
        "ops": sorted(graph.required_ops),
        "storage": storage,
        "form": form,
        "io": written,
        "audios": {name: measured for name, (_, measured) in goldens.items()},
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--snapshot-dir", type=Path, default=sources.SNAPSHOT_DIR)
    parser.add_argument("--dtype", choices=WEIGHT_DTYPES, default="f32")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out_dir = args.out if args.out is not None else DEFAULT_OUT_DIRS[args.dtype]
    summary = export_series(args.snapshot_dir, out_dir, dtype=args.dtype)
    print(json.dumps({"dtype": args.dtype, **summary}, indent=1))


if __name__ == "__main__":
    main()
