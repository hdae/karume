"""Script that writes the real-weight DACVAE (Semantic-DACVAE-Japanese-32dim) to IR v2 containers
+ golden io.

These are G6 (decoder) / G7 (encoder) of the recon, the **codec stage** of Irodori-TTS v4. The six
text-side targets live in `irodori/export.py`, and the host-side path up to the latent in
`irodori/pipeline_ref.py`.

    cd tools/export-recipes
    uv run --with descript-audiotools --with einops python -m irodori.dacvae.export
    uv run --with descript-audiotools --with einops python -m irodori.dacvae.export --target decoder
    uv run --with descript-audiotools --with einops python -m irodori.dacvae.export --dtype f16
    uv run --with descript-audiotools --with einops python -m irodori.dacvae.export --dtype i8

Dependencies are added with `--with` (not in pyproject.toml — the default sync runs on the base
deps only). `descript-audiotools` is needed for two reasons: ① the `dacvae` package subclasses
`audiotools.ml.BaseModel` ② the encoder's golden input is **the reference audio normalized to
−16 LUFS**, and that normalization is audiotools' real path (the IIR of ITU-R BS.1770-4).
`einops` is imported only because `dacvae/__init__.py` adds a name to `BaseModel.EXTERN`.

The model implementation is **imported by adding to `sys.path`** (`--source-dir`) from a clone of
GitHub `facebookresearch/dacvae` (default `inputs/irodori/dacvae-src/`). The pinned version is
{@link SOURCE_COMMIT}.

## What goes into the graphs (2 targets, B=1, S / T are symbolic dimensions)

| Target     | Input           | Output        | Contents                                    |
| ---------- | --------------- | ------------- | ------------------------------------------- |
| `decoder`  | `[1,S,32]`      | `[1,1,1920S]` | `quantizer.out_proj` + decoder main path    |
| `encoder`  | `[1,T,1920]`    | `[1,T,32]`    | encoder + first 32 rows of `quantizer.in_proj` |

The hop is 1920 samples (48 kHz / 25 frames per second), derived from the product of the
checkpoint's `encoder_rates` (not hard-coded).

### Boundary of `decoder` (G6)

The input is `[1,S,32]`, the same as Irodori's latent (the frame axis is the symbol S); inside the
graph it is transposed to `[1,32,S]` and then passed to `quantizer.out_proj` (conv1d 32→1024
k=1). This is exactly the two stages `out_proj` → `decoder` of upstream `DACVAE.decode`, and
`_vae_sample` is not used (Irodori uses the mean as the latent as is with
`deterministic_encode=True` — `codec.py`).

**The watermarking branch is bypassed in form ②, which the upstream README recommends**
(`decoder.alpha = 0.0`, plus replacing `decoder.watermark` with
`wm_model.encoder_block.forward_no_conv`). With `alpha = 0.0` alone, `Decoder.watermark` returns
its input (96ch) as is and **produces no waveform** — because the waveform head
(`Snake(96)` → `conv1d(96→1,k7)` → `Tanh`) sits on the `wm_model.encoder_block.pre` side.
`bypass_watermark` applies both, and the caller measures that the output has 1 channel.

### Boundary of `encoder` (G7)

`quantizer.in_proj` is a conv1d 1024→64 (k=1); upstream splits its output with
`chunk(2, dim=1)` and uses the first half as the mean and the second half as the scale. The
deterministic path discards the scale, so the graph carries **a conv1d truncated to the first 32
rows + the first 32 of the bias** (`_in_proj_truncation_evidence` measures agreement with
`chunk(2)[0]` every run — the tolerance is `IN_PROJ_TRUNCATION_ATOL`).

MUST: the input is **the framed `[1,T,1920]`** (reshaped to `[1,1,1920T]` inside the graph). A
plain `[1,1,1920T]` is not possible because the IR's binding rule requires that "a symbol appears
**bare** in at least one input shape" (`karume.verify._check_symbol_bindability`) — `1920T` alone
cannot bind T, and it fails before the container is written. Framing is **a reinterpretation that
keeps the element order** (row-major, frame t occupies samples `[1920t, 1920(t+1))`), so the host
can pass its contiguous buffer as is. At the same time, the contract "the graph only accepts
multiples of 1920" shows up in the shape.

Reflect pad (`DACVAE._pad`), LUFS normalization and resampling are **the host's job** and are not
in the graph. The reference values of those two stages are produced separately by
`irodori/dacvae/host.py`.

## Preprocessing (two steps always applied before export)

- **`nn.utils.remove_weight_norm` on every `NormConv1d` / `NormConvTranspose1d`**. This checkpoint
  holds the old-style `weight_g` / `weight_v` pairs, and without folding them `weight` is not the
  effective weight (the same preprocessing as `dec` in `export_sbv2.py`). Bit identity of the
  output is measured by `_remove_weight_norm_evidence` every run.
- **Demote `Snake1d.alpha` from a parameter to a plain attribute** (lifted tensor constant). Snake
  is `x + (α+1e-9).reciprocal() · sin(αx)²`, and `(α+1e-9).reciprocal()` is **a constant
  subtree**, but only lifted constants qualify as leaves for constant folding
  (`convert._classify_foldable` — values reached through parameters / buffers do not qualify, to
  avoid baking in huge constants). Without the demotion, `reciprocal` stays in the IR as a
  run-time op, and since it is not in the vocabulary, the export fails. `assert_snake_folded`
  looks at `required_ops` after emit and confirms every run that the folding took effect.

## Standing checks (all measured before emit; if any one fails, **nothing is written**)

1. **The main-path extraction** (`main_path`) is bit-identical to **the `Decoder` patched in form
   ②**. What goes into the graph is the `Decoder` itself (not a copy); the extraction is used only
   by the check — because the next iteration (tiling) builds on the structural claim that
   "decode actually goes through 28 layers, and the watermarking branch contributes only the 4
   layers of the waveform head".
2. **The `in_proj` truncation** agrees with `chunk(2, dim=1)[0]`
   ({@link IN_PROJ_TRUNCATION_ATOL} — bit identity depends on the conv kernel selection and is
   not guaranteed).
3. **Before and after `remove_weight_norm`**, the decoder / encoder outputs are bit-identical.
4. **Eager equivalence of the wrappers** (the difference from the real modules' output before
   patching — {@link EAGER_EQUIV_ATOL} = 0).
5. **Round-trip sanity** (reference audio → encoder → decoder is finite and correlates with the
   input waveform). `_roundtrip_evidence` enforces the correlation floor
   {@link ROUNDTRIP_CORRELATION_MIN} by measurement.

## Storage dtype series (ADR 0018 / 0019 / 0027 / 0050)

`--dtype f16` / `--dtype i8` each write to **a separate series** (`dacvae-32dim-f16/` /
`dacvae-32dim-i8/`) (sharing a directory would silently apply the f32 series' checks to compressed
assets). i8 is per-channel symmetric (axis 0 for conv1d, axis 1 only for `ConvTranspose1d` —
`karume.quantize.QUANT_CHANNEL_AXES` is the source of truth), and {@link _target_scales} remaps
the scale ledger to in-wrapper FQNs before handing it to emit.
The rounding order is the MUST of `_fake_quant` — **after** `fold_weight_norm` and
`lift_snake_alphas`, **before** the reference / golden is taken; so for compressed series the
reference is **taken again** after folding (check 3's "bit identity before and after folding" is
settled first with the reference taken before folding, and only then is the rounding applied).

**`Snake1d.alpha` is not rounded** — rounding happens after `lift_snake_alphas` demoted it from a
parameter to a plain attribute, so it does not appear in `named_parameters`, and on the emit side
it is not a weight slot either (a lifted constant), so it **stays stored as F32**. The golden is
computed with the same f32 alpha, so the two agree.

The golden inputs (the decoder's real latents / the encoder's normalized waveform) are **the same
across series** (the default of `--latent-dir` stays the full-loop golden of the f32 series) —
inputs are what the reference and the GPU read as the same bytes, and the storage dtype does not
affect them.

## Output layout

    outputs/series/dacvae-32dim/<target>/model.krm              weights, constants + 2 descriptors
    outputs/series/dacvae-32dim/<target>/io.<case>.safetensors  inputs + torch CPU expected outputs

The tensor-key convention of io is the same as the other series (`input.<graph input name>` /
`output.<position>`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.export import Dim

from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from irodori.card import IRODORI_CODEC_LICENSE
from irodori.distribution import IRODORI_CODEC_ROLES
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import normalize_boundary_tensor
from karume.dist import NOTICE_FILENAME
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.quantize import fake_quant_int8, round_weights_to_f16

#: Location of the real weights (the output of `hf download Aratako/Semantic-DACVAE-Japanese-32dim`
#: converted to safetensors by `irodori/dacvae/convert.py`).
DEFAULT_MODEL_DIR = INPUTS_ROOT / "irodori" / "dacvae-32dim"

#: Location of the model implementation (a clone of GitHub `facebookresearch/dacvae`).
DEFAULT_SOURCE_DIR = INPUTS_ROOT / "irodori" / "dacvae-src"

#: Origin and pinned version of the implementation clone. Place it with
#: `git clone <SOURCE_REPO> inputs/irodori/dacvae-src`.
SOURCE_REPO = "https://github.com/facebookresearch/dacvae"
SOURCE_COMMIT = "414c20785fc3a28373073ea8ef7a1316eeeaca6e"

#: Storage dtypes that can be written. `i8` was added in iteration 2 (there is no precedent for the
#: i8 quality of a conv-dominated network, so before it takes a slot in the distribution,
#: `i8-codec-only` of `irodori/measure_quant.py` measures it on its own — ADR 0050 decision 6 /
#: the risks of the quantization recon).
WEIGHT_DTYPES: tuple[str, ...] = ("f32", "f16", "i8")

WEIGHTS_FILE = "weights.safetensors"
METADATA_FILE = "metadata.json"
MODEL_FILE = "model.krm"

#: Provenance baked into the containers (container-v1 §2.3). The codec's upstream is
#: `Aratako/Semantic-DACVAE-Japanese-32dim`, whose original weights are the Apache 2.0
#: `facebook/dacvae-watermarked`, so the license identifier baked in is the card side's source of
#: truth {@link irodori.card.IRODORI_CODEC_LICENSE} (the MIT and Apache 2.0 chain).
#: `notice` is `NOTICE.md` at the root of the distribution repository (where the modification
#: notice of Apache 2.0 §4(b) lives).
PROVENANCE = Provenance(license=IRODORI_CODEC_LICENSE, notice=NOTICE_FILENAME)
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

TARGET_DECODER = "decoder"
TARGET_ENCODER = "encoder"
TARGETS = (TARGET_DECODER, TARGET_ENCODER)

#: Names of the symbolic dimensions in the IR (scoped per graph, so the two would not collide even
#: with the same name).
DECODER_SYMBOL = "S"
ENCODER_SYMBOL = "T"

#: Lower bound of the symbolic dimensions. Set to 2 to avoid torch.export's 0/1 specialization
#: (same as the other series).
MIN_SYM_LENGTH = 2

#: Utterance length (seconds) that sets the upper bound of the `decoder`'s symbolic dimension S.
#: It is the upper bound of the latents that can be generated (30 s, the same as
#: `irodori.export.DIT_MAX_SECONDS` — ADR 0047). **This script has no position table and no mask**
#: (a pure convolutional network), so raising the bound does not grow the container by a single
#: byte.
DECODER_MAX_SECONDS = 30.0

#: Reference audio length (seconds) that sets the upper bound of the `encoder`'s symbolic dimension
#: T. Matches the checkpoint's `ref_max_seconds` (120 s) — reference audio is never longer than
#: that.
ENCODER_MAX_SECONDS = 120.0

#: Reference audio (the voice-cloning reference of the official samples). 48 kHz mono, 7.6 s.
DEFAULT_REFERENCE_WAV = INPUTS_ROOT / "irodori" / "v4-small" / "samples" / "clone_ref1.wav"

#: Target of the reference audio's LUFS normalization (the default of
#: `inference_runtime.SamplingRequest.ref_normalize_db`).
REFERENCE_NORMALIZE_DB = -16.0

#: Source of the real latents (the final z of the full-loop golden written by
#: `irodori/pipeline_ref.py`).
DEFAULT_LATENT_DIR = SERIES_ROOT / "irodori-v4-small" / "pipeline"
LATENT_CASE_PREFIX = "case."
LATENT_KEY = "z"

#: Difference allowed in the eager equivalence between the wrappers and the real modules **before
#: patching**. **0 = bit identity is required**.
#:
#: The wrappers do four things — ① add a transpose / reshape ② truncate `in_proj`
#: ③ `remove_weight_norm` ④ demote `Snake1d.alpha` — none of which changes values (② to ④ also
#: have bit identity measured by independent checks). A non-zero value means one of these claims
#: broke — fail instead of passing it approximately, and find out first which one broke.
EAGER_EQUIV_ATOL = 0.0

#: Lower bound of the correlation between the round-trip (encoder → decoder) reconstruction and the
#: input waveform (`_roundtrip_evidence`).
#:
#: MUST: do not make it a tautology — "finite and non-zero" alone would pass output that is not
#: audio because of swapped channel order or a stray path. The measured value is 0.988 (7.6 s of
#: reference audio, 0 samples of delay), and 0.8 is four fifths of it. This is the reconstruction
#: of a trained codec, so with the correct path the correlation is close to 1.
ROUNDTRIP_CORRELATION_MIN = 0.8

#: Golden cases of `decoder` `(name, source case of the latent, length)`. Length `None` keeps the
#: source as is.
#:
#: The real latents come from the full-loop golden of `irodori/pipeline_ref.py` (`case.full` =
#: with reference audio, 161 frames / `case.no-ref` = without reference, 116 frames). **Synthetic
#: random values are not used** because Snake's `sin(αx)` is strongly affected by the input's value
#: range; the choice ties the basis of the tolerance to the value range of real use (the same
#: reason as the speaker embedding in `export_sbv2.py`).
#:
#: Both ends of the length: the symbolic dimension's lower bound 2 (the first 2 frames) and 750,
#: **the declared upper bound itself**. There are not enough real latents for 750, so it is built
#: by repeating along the frame axis (the value range stays that of the real latents).
DECODER_CASES: tuple[tuple[str, str, int | None], ...] = (
    ("z-min", "full", MIN_SYM_LENGTH),
    ("z-no-ref", "no-ref", None),
    ("z-full", "full", None),
    ("z-max", "full", 750),
)

#: Golden cases of `encoder` `(name, frame count)`. Frame count `None` is the full length of the
#: reference audio.
#:
#: The input is the reference audio **normalized to −16 LUFS → reflect-padded** (the exit of the
#: host preprocessing = the entrance of the graph). The short case cuts the symbolic dimension's
#: lower bound 2 (3840 samples) from the start.
ENCODER_CASES: tuple[tuple[str, int | None], ...] = (
    ("wav-min", MIN_SYM_LENGTH),
    ("wav-ref", None),
)


class DacvaeSource:
    """The implementation taken from the `dacvae` package (imported by adding to `sys.path`).

    MUST: import **from the implementation clone** (the script keeps no copy).
    `dacvae/__init__.py` pulls in `audiotools` and `einops`, so it only works in an environment
    where they were added with `--with`.
    """

    def __init__(self, source_dir: Path) -> None:
        model_py = source_dir / "dacvae" / "model" / "dacvae.py"
        if not model_py.is_file():
            raise SystemExit(
                f"model implementation not found: {model_py}"
                f" (pass the checkout of `git clone {SOURCE_REPO}` as --source-dir."
                f" The pinned version is {SOURCE_COMMIT})"
            )
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        from dacvae.model.dacvae import DACVAE
        from dacvae.nn.layers import Snake1d

        self.dacvae_cls = DACVAE
        self.snake_cls = Snake1d


def read_kwargs(model_dir: Path) -> dict[str, Any]:
    """Read the configuration values from the `metadata.json` written by
    `irodori/dacvae/convert.py`.

    MUST: do not fetch the config from HF again — the `kwargs` embedded in the checkpoint are the
    source of truth for the shape these weights were actually built with.
    """
    path = model_dir / METADATA_FILE
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing (create it with `uv run python -m irodori.dacvae.convert`)"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "kwargs" not in raw:
        raise SystemExit(f"{path} has no 'kwargs'")
    return dict(raw["kwargs"])


def load_codec(source: DacvaeSource, model_dir: Path) -> nn.Module:
    """Build `DACVAE` with the configuration of `metadata.json` and load the safetensors weights.

    `load_state_dict(strict=True)` is the check that "all 317 tensors were consumed, none left
    over" (`weight_g` / `weight_v` are not folded yet at this point).
    """
    weights = model_dir / WEIGHTS_FILE
    if not weights.is_file():
        raise SystemExit(
            f"{weights} is missing (create it with `uv run python -m irodori.dacvae.convert`)"
        )
    model = source.dacvae_cls(**read_kwargs(model_dir))
    model.load_state_dict(load_file(str(weights)), strict=True)
    return model.eval()


def bypass_watermark(decoder: nn.Module) -> None:
    """Bypass the watermarking branch in form ②, which the upstream README recommends (the G6
    section of the module docstring).

    MUST: apply **both** `alpha = 0.0` and the replacement of `watermark`. With only the former,
    `Decoder.watermark` returns the 96ch input as is and produces no waveform (measured). With only
    the latter, the path where upstream reads `alpha` remains.
    """
    decoder.alpha = 0.0
    decoder.watermark = lambda x, message=None, block=decoder.wm_model.encoder_block: (
        block.forward_no_conv(x)
    )


def main_path(decoder: nn.Module) -> nn.Sequential:
    """A `Sequential` of only the layers decode actually goes through (**used only by the check**
    — a copy).

    `DecoderBlock.forward` cuts `block` every `_chunk_size` and runs **only the chunks whose index
    is a multiple of `_chunk_size`**, in order (the rest are the watermarking branch's up/down
    samplers). The tail is `forward_no_conv` of the form-② bypass = `pre[0]` (Snake 96) /
    `pre[1]` (conv1d 96→1 k7) / `pre[2]` (Tanh), and `pre[3]` is replaced by `Identity`.

    MUST: do not put this copy into the graph — what goes in is the `Decoder` itself; this is built
    only so that `_main_path_evidence` can measure that "the main path is these 28 layers and the
    remaining 25 of the watermarking branch contribute nothing" (if upstream reorders its layers,
    that check fails).
    """
    layers: list[nn.Module] = [decoder.model[0]]
    for block in decoder.model[1:]:
        size = block._chunk_size
        chunks = [block.block[start : start + size] for start in range(0, len(block.block), size)]
        layers += [
            layer for index, chunk in enumerate(chunks) if index % size == 0 for layer in chunk
        ]
    head = decoder.wm_model.encoder_block.pre
    layers += [head[0], head[1], head[2]]
    return nn.Sequential(*layers)


def fold_weight_norm(model: nn.Module) -> int:
    """Fold every `weight_norm` into effective weights (returns how many were folded).

    While weight_norm remains, `Conv1d.weight` is **not the effective weight** (a forward pre-hook
    builds it from `weight_g` / `weight_v` on every call). Writing it to the IR as is would give a
    different model.
    """
    folded = 0
    for module in model.modules():
        if hasattr(module, "weight_g"):
            nn.utils.remove_weight_norm(module)
            folded += 1
    if folded == 0:
        raise SystemExit(
            "no module has weight_norm — either the checkpoint is already folded or upstream"
            " switched to a different normalization (the effective-weight premise breaks)"
        )
    return folded


def lift_snake_alphas(source: DacvaeSource, model: nn.Module) -> int:
    """Demote `Snake1d.alpha` from a parameter to a plain attribute (lifted tensor constant).

    The reason for the demotion is in the preprocessing section of the module docstring (to put
    `(α+1e-9).reciprocal()` on the constant-folding frontier). Values do not change (it is only
    `detach().clone()`), so eager equivalence stays bit-identical.

    MUST: fail if nothing could be demoted — going on after an empty scan would leave
    `reciprocal` in the IR, and the export would fail because it is not in the vocabulary (failing
    far from the cause).
    """
    lifted = 0
    for module in model.modules():
        if isinstance(module, source.snake_cls):
            tensor = module._parameters.pop("alpha")
            module.alpha = tensor.detach().clone()
            lifted += 1
    if lifted == 0:
        raise SystemExit("no Snake1d found — the scan for the alpha demotion found nothing")
    return lifted


def truncated_in_proj(in_proj: nn.Conv1d) -> nn.Conv1d:
    """Build a conv1d of `quantizer.in_proj` truncated to **the first 32 rows** (the mean side
    only).

    MUST: call it **after** `fold_weight_norm` (`in_proj.weight` must be the effective weight).
    `_in_proj_truncation_evidence` measures agreement with `chunk(2, dim=1)[0]` (the tolerance is
    `IN_PROJ_TRUNCATION_ATOL`).
    """
    if hasattr(in_proj, "weight_g"):
        raise SystemExit(
            "the weight_norm of in_proj is not folded — the truncation would not be of the"
            " effective weights"
        )
    if in_proj.out_channels % 2 != 0:
        raise SystemExit(
            f"the output count {in_proj.out_channels} of in_proj is not even (the mean / scale"
            " split in two)"
        )
    half = in_proj.out_channels // 2
    trimmed = nn.Conv1d(
        in_proj.in_channels,
        half,
        kernel_size=in_proj.kernel_size[0],
        stride=in_proj.stride[0],
        padding=in_proj.padding[0],
        dilation=in_proj.dilation[0],
    )
    with torch.no_grad():
        trimmed.weight.copy_(in_proj.weight[:half])
        trimmed.bias.copy_(in_proj.bias[:half])
    return trimmed.eval()


class DecoderGraph(nn.Module):
    """Export wrapper of `DACVAE.decode` (`out_proj` → the `Decoder` patched in form ②).

    The only difference from the original `decode` is one transpose at the entrance (upstream takes
    `[B,D,T]`; this takes `[B,T,D]`, the same as Irodori's latent — the transpose that
    `decode_latent` in `codec.py` does on the host side, moved into the graph). **It keeps no copy
    of the main path** (it calls the `Decoder` itself).
    """

    def __init__(self, out_proj: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.out_proj = out_proj
        self.decoder = decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.out_proj(latent.transpose(1, 2)))


class EncoderGraph(nn.Module):
    """Export wrapper of `encoder` + the truncated `in_proj` (the G7 section of the module
    docstring).

    It differs from the original deterministic encode (the `deterministic_encode` path of
    `codec.py`) in three points:

    - it takes the input **framed** as `[1,T,1920]` and turns it back into `[1,1,1920T]` inside
      the graph (a reinterpretation that keeps the element order — to bind the symbol T bare)
    - it replaces `in_proj(...).chunk(2, dim=1)[0]` with **the truncated conv1d**
    - it transposes the output to `[1,T,32]` (upstream does the same transpose at the end of
      `encode_waveform`)

    Reflect pad (`DACVAE._pad`) is **not included** — the host aligns to a multiple of 1920 before
    calling.
    """

    def __init__(self, encoder: nn.Module, in_proj: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.in_proj = in_proj

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        samples = wav.reshape(1, 1, wav.shape[1] * wav.shape[2])
        return self.in_proj(self.encoder(samples)).transpose(1, 2)


def hop_length(model: nn.Module) -> int:
    """Samples per latent frame (the product of `encoder_rates` — not hard-coded)."""
    hop = int(model.hop_length)
    if hop <= 0:
        raise SystemExit(f"hop_length is {hop} — the checkpoint's encoder_rates are invalid")
    return hop


def frame_rate(model: nn.Module) -> float:
    """Frame rate of the latent (Hz). 48000 / 1920 = 25."""
    return int(model.sample_rate) / hop_length(model)


def read_wav(path: Path) -> tuple[torch.Tensor, int]:
    """Read a WAV as mono `[N]` f32 (`(waveform, sampling rate)`).

    The reader is soundfile. int16 PCM is normalized by **/32768** (the int16 low end −32768 maps
    to exactly −1.0), and `irodori/dacvae/host.py` measures this convention on a synthetic WAV and
    records it in the golden's meta.
    """
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).transpose(0, 1)
    if wav.shape[0] != 1:
        # The same downmix to mono as `encode_waveform` in `codec.py` (channel average).
        wav = wav.mean(dim=0, keepdim=True)
    return wav[0].contiguous(), int(sample_rate)


def normalize_reference_wav(wav: torch.Tensor, sample_rate: int, target_db: float) -> torch.Tensor:
    """LUFS-normalize the reference audio (the same two calls as `_normalize_loudness` in
    `codec.py`).

    `AudioSignal.normalize(db)` measures the integrated loudness of ITU-R BS.1770-4 and applies the
    gain, and `ensure_max_of_audio()` scales down only when peak > 1. **It calls the library instead
    of holding the formula** so as not to copy the K-weighting IIR coefficients and the gating. Bit
    identity with the upstream staticmethod is measured every run by `irodori/dacvae/host.py`
    (which can import `irodori_tts`).
    """
    from audiotools import AudioSignal

    signal = AudioSignal(wav.detach().to(torch.float32).unsqueeze(0).unsqueeze(0), int(sample_rate))
    signal.normalize(float(target_db))
    signal.ensure_max_of_audio()
    return signal.audio_data.reshape(-1).contiguous()


def build_decoder_cases(latent_dir: Path, sym_max: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """Golden cases of `decoder` `(name, latent [1,S,32])` (as in the table {@link DECODER_CASES}).

    The source is the full-loop golden of `irodori/pipeline_ref.py`. **Missing length is made by
    repeating along the frame axis** (to extend while keeping the real latents' value range —
    filling with random values would change Snake's value range).
    """
    sources: dict[str, torch.Tensor] = {}
    for name in {source for _n, source, _l in DECODER_CASES}:
        path = latent_dir / f"{LATENT_CASE_PREFIX}{name}{IO_SUFFIX}"
        if not path.is_file():
            raise SystemExit(
                f"real latent missing: {path}"
                " (create it with `uv run --with 'transformers==5.14.1' python -m"
                " irodori.pipeline_ref`)"
            )
        tensors = load_file(str(path))
        if LATENT_KEY not in tensors:
            raise SystemExit(f"{path} has no '{LATENT_KEY}'")
        sources[name] = tensors[LATENT_KEY].to(torch.float32)

    cases: list[tuple[str, torch.Tensor]] = []
    for name, source, length in DECODER_CASES:
        latent = sources[source]
        if latent.ndim != 3 or latent.shape[0] != 1:
            raise SystemExit(f"{name}: latent shape {tuple(latent.shape)} is not [1,S,D]")
        want = int(latent.shape[1]) if length is None else int(length)
        if not MIN_SYM_LENGTH <= want <= sym_max:
            raise SystemExit(
                f"{name}: S={want} is outside the symbolic dimension range"
                f" [{MIN_SYM_LENGTH}, {sym_max}]"
            )
        repeats = -(-want // int(latent.shape[1]))
        cases.append((name, latent.repeat(1, repeats, 1)[:, :want].contiguous()))
    return tuple(cases)


def build_encoder_cases(
    model: nn.Module, wav_path: Path, sym_max: int, target_db: float
) -> tuple[tuple[str, torch.Tensor], ...]:
    """Golden cases of `encoder` `(name, framed waveform [1,T,1920])`.

    Exactly the exit of the host preprocessing: reference audio → LUFS normalization → reflect pad
    (**calling** upstream `DACVAE._pad`) → reinterpretation as `[1,T,1920]`. There is no
    resampling (anything other than 48 kHz fails).
    """
    wav, sample_rate = read_wav(wav_path)
    if sample_rate != int(model.sample_rate):
        raise SystemExit(
            f"{wav_path.name}: {sample_rate}Hz differs from the codec's {int(model.sample_rate)}Hz"
            " (resampling is the host's job; this script does not do it)"
        )
    normalized = normalize_reference_wav(wav, sample_rate, target_db)
    with torch.no_grad():
        padded = model._pad(normalized.reshape(1, 1, -1))
    hop = hop_length(model)
    if int(padded.shape[-1]) % hop != 0:
        raise SystemExit(
            f"the length after reflect pad, {int(padded.shape[-1])}, is not a multiple of hop {hop}"
        )

    cases: list[tuple[str, torch.Tensor]] = []
    for name, length in ENCODER_CASES:
        frames = int(padded.shape[-1]) // hop if length is None else int(length)
        if not MIN_SYM_LENGTH <= frames <= sym_max:
            raise SystemExit(
                f"{name}: T={frames} is outside the symbolic dimension range"
                f" [{MIN_SYM_LENGTH}, {sym_max}]"
            )
        if frames * hop > int(padded.shape[-1]):
            raise SystemExit(f"{name}: T={frames} exceeds the length of the reference audio")
        cases.append((name, padded[..., : frames * hop].reshape(1, frames, hop).contiguous()))
    return tuple(cases)


def _decode_reference(model: nn.Module, latent: torch.Tensor) -> torch.Tensor:
    """Take the reference values on upstream's decode path (`out_proj` → the `Decoder` patched in
    form ②)."""
    with torch.no_grad():
        return model.decoder(model.quantizer.out_proj(latent.transpose(1, 2)))


def _encode_reference(model: nn.Module, framed: torch.Tensor) -> torch.Tensor:
    """Take the reference values on upstream's deterministic encode path (`encoder` →
    `chunk(2)[0]` of `in_proj`)."""
    with torch.no_grad():
        hidden = model.encoder(framed.reshape(1, 1, -1))
        mean, _scale = model.quantizer.in_proj(hidden).chunk(2, dim=1)
        return mean.transpose(1, 2)


def _pristine_outputs(
    model: nn.Module,
    decoder_cases: Sequence[tuple[str, torch.Tensor]],
    encoder_cases: Sequence[tuple[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Eager outputs **before weight_norm is folded** (the baseline of check 3, and the golden of
    the f32 series itself).

    MUST: call it before `fold_weight_norm`. Taken after folding, `_remove_weight_norm_evidence`
    would compare "folded against folded", a difference of 0 would hold trivially, and it would
    falsely PASS (the same kind of contamination as `_assert_weight_norm_present` in
    `export_sbv2.py`).
    """
    if not any(hasattr(module, "weight_g") for module in model.modules()):
        raise AssertionError(
            "tried to take the reference after folding weight_norm (the equivalence check of"
            " the folding would become a tautology)"
        )
    return _eager_outputs(model, decoder_cases, encoder_cases)


def _eager_outputs(
    model: nn.Module,
    decoder_cases: Sequence[tuple[str, torch.Tensor]],
    encoder_cases: Sequence[tuple[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Eager outputs of upstream's decode / encode paths (the plain computation, regardless of
    whether `weight_norm` is present).

    For compressed series this runs once more **after** rounding (the ordering MUST of
    {@link _fake_quant}) — by then `weight_g` is gone, so the check of {@link _pristine_outputs}
    cannot be passed. Check 3 (bit identity before and after folding) was settled before rounding,
    and taking the reference again does not weaken that claim.
    """
    decoded = {name: _decode_reference(model, latent) for name, latent in decoder_cases}
    for name, audio in decoded.items():
        if audio.shape[1] != 1:
            raise AssertionError(
                f"{name}: decode has {audio.shape[1]} output channels — the watermark bypass is"
                " not in form ② and the output is not a waveform (with `alpha = 0.0` alone it"
                " returns 96ch as is)"
            )
    return {
        TARGET_DECODER: decoded,
        TARGET_ENCODER: {name: _encode_reference(model, wav) for name, wav in encoder_cases},
    }


class FakeQuantResult(NamedTuple):
    """Return value of {@link _fake_quant} (the rounding summary and the i8 per-channel scale
    ledger)."""

    report: str | None
    #: **FQN in the codec itself** → scale (empty for anything but i8). Remapping to the
    #: in-wrapper FQNs emit looks up is done by {@link _target_scales}.
    scales: Mapping[str, torch.Tensor]


#: FQN of `in_proj` in the codec itself (where the scale of the truncated one comes from).
FULL_IN_PROJ_KEY = "quantizer.in_proj.weight"
#: **In-wrapper** FQN of the truncated `in_proj` (`EncoderGraph.in_proj`).
TRIMMED_IN_PROJ_KEY = "in_proj.weight"

#: Target → sequence of scale sources `(FQN prefix in the codec, in-wrapper FQN prefix)` (i8
#: only).
#:
#: MUST: use the same spelling as the graph wrappers' constructors — the wrappers hold the codec's
#: subtrees under other attribute names, so the ledger's keys (FQNs in the codec) do not match the
#: keys emit looks up (FQNs in the exported module). On a mismatch, either `_target_scales` or
#: emit fails loudly.
TARGET_SCALE_SOURCES: Mapping[str, tuple[tuple[str, str], ...]] = MappingProxyType(
    {
        TARGET_DECODER: (("quantizer.out_proj.", "out_proj."), ("decoder.", "decoder.")),
        # `in_proj` is a separate truncated module, so a prefix cannot remap it
        # ({@link _trimmed_in_proj_scale} cuts out only the first half along axis 0).
        TARGET_ENCODER: (("encoder.", "encoder."),),
    }
)


def _fake_quant(dtype: str, model: nn.Module) -> FakeQuantResult:
    """Round the **effective weights** to values representable in the storage dtype (nothing for
    f32).

    The fake-quant of ADR 0006 / 0018 / 0019 / 0027 / 0050.

    MUST (ordering): ① **after** `fold_weight_norm` (rounding before folding would round
    `weight_g` / `weight_v`, and the effective weights built from them would not sit on the f16 /
    i8 grid) ② **after** `lift_snake_alphas` (the demoted `alpha` does not appear in
    `named_parameters`, so it is **not rounded** — alpha is a lifted constant, not a weight slot of
    emit either, and stays stored as F32, so the golden side being computed with the f32 alpha is
    the correct correspondence) ③ **before** `truncated_in_proj` (the truncation copies the
    weights, so building it before rounding would make only the encoder store the original weights.
    For i8, **in addition**, the scale ledger can only be taken with the codec-side FQNs, so unless
    rounding comes first, the ledger (the codec's first rows — {@link _trimmed_in_proj_scale}) and
    the stored weights of the truncated copy would point at different weights)
    ④ **before** the reference / golden is taken (the MUST of `karume.quantize`).
    """
    if dtype == "f32":
        return FakeQuantResult(None, {})
    if dtype == "i8":
        int8 = fake_quant_int8(model)
        print(f"[fake-quant] codec: rounded to i8 per-channel — {int8.describe()}", flush=True)
        return FakeQuantResult(int8.describe(), int8.scales)
    report = round_weights_to_f16(model)
    if report.parameters + report.buffers == 0:
        raise SystemExit(f"storage {dtype} was requested but no weights were rounded")
    print(
        f"[fake-quant] codec: rounded to f16-representable values — {report.describe()}", flush=True
    )
    return FakeQuantResult(report.describe(), {})


def _trimmed_in_proj_scale(scales: Mapping[str, torch.Tensor], in_proj: nn.Module) -> torch.Tensor:
    """Per-channel scale of the truncated `in_proj` (exactly **the first rows** of the codec's
    scale).

    `truncated_in_proj` only copies the first 32 rows of the rounded weights, so cutting the first
    half of the scale along the per-channel axis (for `Conv1d`, the output channels = axis 0)
    corresponds exactly.

    MUST: do **not recompute** the scale from the truncated weights — the f32 division can move it
    by 1 ulp, and then emit's inverse-transform bit-identity check fails (ADR 0019's "the scale is
    exactly the value the fake-quant used").
    """
    source = scales.get(FULL_IN_PROJ_KEY)
    if source is None:
        raise SystemExit(
            f"the scale ledger has no '{FULL_IN_PROJ_KEY}' (cannot map the truncation)"
        )
    rows = int(in_proj.out_channels)
    if int(source.shape[0]) < rows:
        raise SystemExit(
            f"'{FULL_IN_PROJ_KEY}' has {int(source.shape[0])} scale rows, fewer than the"
            f" {rows} after truncation (the per-channel axis is mixed up)"
        )
    return source[:rows]


def _target_scales(
    target: str, wrapper: nn.Module, scales: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Remap the scale ledger to the target's **in-wrapper FQNs** ({@link TARGET_SCALE_SOURCES}).

    Only **weights the wrapper does not have** are dropped (the rest of the watermarking branch and
    the other target). The reverse (in the wrapper but missing from the ledger) is failed loudly by
    emit.
    """
    if not scales:
        return {}
    owned = {name for name, _ in wrapper.named_parameters()}
    picked: dict[str, torch.Tensor] = {}
    for source, destination in TARGET_SCALE_SOURCES[target]:
        for key, scale in scales.items():
            if not key.startswith(source):
                continue
            rebased = destination + key[len(source) :]
            if rebased in owned:
                picked[rebased] = scale
    if target == TARGET_ENCODER:
        picked[TRIMMED_IN_PROJ_KEY] = _trimmed_in_proj_scale(scales, wrapper.in_proj)
    if not picked:
        raise SystemExit(
            f"{target}: not a single scale of the ledger could be remapped to an in-wrapper FQN"
            f" (the prefixes {TARGET_SCALE_SOURCES[target]} disagree with the wrapper's structure)"
        )
    return picked


def _main_path_evidence(
    decoder: nn.Module, cases: Sequence[tuple[str, torch.Tensor]], out_proj: nn.Module
) -> dict[str, float]:
    """Measure that the main-path extraction is **bit-identical** to the `Decoder` patched in form
    ② (check 1).

    MUST: do not make it a tautology — the extraction (`main_path`) and `Decoder.forward` are
    different paths: the former is a copy that picks only the multiples of `_chunk_size`, the
    latter is upstream itself. Agreement is a measurement that "the remaining 25 layers of the
    watermarking branch contribute nothing to decode", not something trivial.
    """
    extracted = main_path(decoder)
    evidence: dict[str, float] = {}
    for name, latent in cases:
        with torch.no_grad():
            embedded = out_proj(latent.transpose(1, 2))
            reference = decoder(embedded)
            actual = extracted(embedded)
        if not torch.equal(reference, actual):
            raise AssertionError(
                f"{name}: the main-path extraction does not match the Decoder itself"
                f" (max abs diff {float((reference - actual).abs().max())})"
                " — the separation of the watermarking branch (only multiples of `_chunk_size`"
                " pass) is broken"
            )
        evidence[name] = 0.0
    return evidence


#: Absolute tolerance of check 2 (in_proj truncation).
#:
#: | Check   | Raw measurement (max abs) | Adopted value (≈10×) |
#: | ------- | ------------------------- | -------------------- |
#: | Check 2 | 2.9802e-8 (2026-08-13)    | 3e-7                 |
#:
#: MUST: do not require bit identity (`torch.equal`) — the two sides compared are "the first h of a
#: conv with out_channels 2h" and "a conv with out_channels h", and torch's conv changes its kernel
#: blocking (= reduction order) with the channel count and thread count, so bit identity is
#: implementation behavior without a spec guarantee (measured too: 1 ulp off with 6 threads, 0 with
#: 8 — it broke as soon as an added test shifted the global RNG sequence). The purpose of the check
#: is to detect a mean / scale mix-up, whose difference is O(1) — this tolerance does not reduce
#: its discriminating power (that a mix-up keeps failing is pinned by
#: `irodori/dacvae/tests/test_export.py`).
IN_PROJ_TRUNCATION_ATOL = 3e-7


def _in_proj_truncation_evidence(
    model: nn.Module, trimmed: nn.Conv1d, cases: Sequence[tuple[str, torch.Tensor]]
) -> dict[str, float]:
    """Measure that the truncated `in_proj` agrees with `chunk(2, dim=1)[0]` (check 2).

    MUST: do not make it a tautology — the truncation rests on upstream's split convention "the
    first half of the output channels = mean", and a mix-up that takes the second half (scale)
    matches in shape and dtype, so only this check reveals it. The return value carries the
    measured max abs difference as is (the tolerance is {@link IN_PROJ_TRUNCATION_ATOL}).
    """
    evidence: dict[str, float] = {}
    for name, wav in cases:
        with torch.no_grad():
            hidden = model.encoder(wav.reshape(1, 1, -1))
            expected, _scale = model.quantizer.in_proj(hidden).chunk(2, dim=1)
            actual = trimmed(hidden)
        diff = float((expected - actual).abs().max())
        if diff > IN_PROJ_TRUNCATION_ATOL:
            raise AssertionError(
                f"{name}: the truncated in_proj does not match chunk(2)[0]"
                f" (max abs diff {diff} > tolerance {IN_PROJ_TRUNCATION_ATOL})"
            )
        evidence[name] = diff
    return evidence


def _remove_weight_norm_evidence(
    model: nn.Module,
    decoder_cases: Sequence[tuple[str, torch.Tensor]],
    encoder_cases: Sequence[tuple[str, torch.Tensor]],
    pristine: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, float]]:
    """Measure that the outputs are bit-identical before and after `remove_weight_norm` (check 3).

    Folding is the composition `w = g · v/‖v‖`, an identity mathematically but possibly **a
    different rounding in floating point**. If this is not 0, the golden's expected values (taken
    before folding) cannot serve as the expected values of the folded graph — fail instead of
    relaxing.
    """
    if any(hasattr(module, "weight_g") for module in model.modules()):
        raise AssertionError("weight_norm is still present (a check to call after folding)")
    evidence: dict[str, dict[str, float]] = {}
    for target, cases, call in (
        (TARGET_DECODER, decoder_cases, _decode_reference),
        (TARGET_ENCODER, encoder_cases, _encode_reference),
    ):
        measured: dict[str, float] = {}
        for name, value in cases:
            actual = call(model, value)
            expected = pristine[target][name]
            if not torch.equal(expected, actual):
                raise AssertionError(
                    f"{target}/{name}: the output changed across remove_weight_norm"
                    f" (max abs diff {float((expected - actual).abs().max())})"
                )
            measured[name] = 0.0
        evidence[target] = measured
    return evidence


def _roundtrip_evidence(
    encoder_graph: nn.Module, decoder_graph: nn.Module, framed: torch.Tensor
) -> dict[str, float]:
    """Measure that the round trip of the reference audio (encode → decode) is a plausible
    waveform (check 5).

    Two things are checked: ① every output is finite ② the correlation with the input waveform is
    at least {@link ROUNDTRIP_CORRELATION_MIN}. It is the only check that runs the two graphs
    **chained**, and a mix-up on one side only (channel order, transpose direction, first / second
    half of in_proj) fails here as a correlation.
    """
    with torch.no_grad():
        latent = encoder_graph(framed)
        audio = decoder_graph(latent)
    if not bool(torch.isfinite(audio).all()):
        raise AssertionError("the round-trip output has non-finite values")
    source = framed.reshape(-1)
    restored = audio.reshape(-1)
    if restored.numel() != source.numel():
        raise AssertionError(
            f"the round-trip length is {restored.numel()}, not the input's {source.numel()}"
            " (the hop is mixed up)"
        )
    centered_source = source - source.mean()
    centered_restored = restored - restored.mean()
    denominator = float(centered_source.norm() * centered_restored.norm())
    if denominator == 0.0:
        raise AssertionError(
            "the round-trip output is constant (the correlation cannot be measured)"
        )
    correlation = float((centered_source * centered_restored).sum()) / denominator
    if correlation < ROUNDTRIP_CORRELATION_MIN:
        raise AssertionError(
            f"the round-trip correlation {correlation:.4f} fell below the floor"
            f" {ROUNDTRIP_CORRELATION_MIN} — somewhere along the path a different value is passed"
        )
    return {
        "correlation": round(correlation, 5),
        "input_rms": round(float(source.pow(2).mean().sqrt()), 6),
        "output_rms": round(float(restored.pow(2).mean().sqrt()), 6),
        "latent_abs_max": round(float(latent.abs().max()), 5),
    }


def _check_wrapper_equivalence(
    wrapper: nn.Module, argument: torch.Tensor, expected: torch.Tensor, where: str, atol: float
) -> float:
    """Check that the wrapper's output matches the reference output of the real modules (check 4).

    The reference is taken **before folding** for the f32 series, and **taken again after rounding**
    for compressed series (in both cases computed with the same weights as the wrapper, so the
    requirement stays `EAGER_EQUIV_ATOL = 0`).
    """
    with torch.no_grad():
        actual = wrapper(argument)
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(
            f"{where}: wrapper output shape {tuple(actual.shape)} differs from"
            f" the expected {tuple(expected.shape)}"
        )
    diff = float((actual - expected).abs().max())
    if diff > atol:
        raise AssertionError(f"{where}: eager equivalence broke (max abs diff {diff} > {atol})")
    return diff


def assert_snake_folded(graph: IrGraph, where: str) -> None:
    """Confirm from `required_ops` after emit that Snake's constant subtree was folded.

    If `reciprocal` / `pow` remain, the demotion of `alpha` did not take effect (they are not in the
    IR vocabulary either, so normally the conversion stage fails, but once the vocabulary grows
    they could silently become run-time ops).
    """
    leaked = sorted({"reciprocal", "pow", "div"} & set(graph.required_ops))
    if leaked:
        raise AssertionError(
            f"{where}: Snake's constant subtree was not folded and {leaked} remain as run-time"
            " ops — the demotion of `Snake1d.alpha` (lifted tensor constant) did not take effect"
        )
    if "sin" not in graph.required_ops:
        raise AssertionError(f"{where}: no sin in the graph — Snake may have disappeared entirely")


def _write_io(
    graph: IrGraph,
    inputs: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    out_dir: Path,
) -> list[str]:
    """Write each case's inputs and torch CPU expected outputs to `io.<case>.safetensors`.

    Both targets have one input and one output, so names and counts are checked directly against
    the graph declaration.
    """
    declared = [spec.name for spec in graph.inputs]
    if len(declared) != 1 or len(graph.outputs) != 1:
        raise AssertionError(
            f"inputs {declared} / outputs {list(graph.outputs)} are not exactly one each"
            " (a different io convention)"
        )
    written: list[str] = []
    for name, value in inputs.items():
        tensors = {
            f"{INPUT_PREFIX}{declared[0]}": normalize_boundary_tensor(
                value.detach().contiguous(), f"{name}: input '{declared[0]}'"
            ),
            f"{OUTPUT_PREFIX}0": normalize_boundary_tensor(
                expected[name].detach().contiguous(), f"output 0 of {name}"
            ),
        }
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written


class TargetAxis(NamedTuple):
    """Per-target declaration of the symbolic dimension (`dynamic` is the axis index of the input
    that carries the symbol)."""

    symbol: str
    upper: int
    axis: int


def _graph_summary(graph: IrGraph, path: Path) -> dict[str, Any]:
    return {
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "inputs": [[spec.name, list(spec.shape)] for spec in graph.inputs],
        "output_shapes": [list(graph.values[name].shape) for name in graph.outputs],
        "model_bytes": sum(p.stat().st_size for p in container_parts(path)),
    }


def export_series(
    model_dir: Path,
    source_dir: Path,
    out_dir: Path,
    *,
    targets: Sequence[str] = TARGETS,
    latent_dir: Path = DEFAULT_LATENT_DIR,
    reference_wav: Path = DEFAULT_REFERENCE_WAV,
    normalize_db: float = REFERENCE_NORMALIZE_DB,
    dtype: str = "f32",
) -> dict[str, Any]:
    """Write the IR containers and golden io, and return a summary.

    `targets` narrows **only the writing**. Building the cases of both decoder and encoder and the
    5 checks always run regardless of `targets` (the same reason as the MUST below — because the
    whole chain always runs, no path arises by which an asset that skipped the checks is put in
    place).

    MUST: artifacts are written to **a staging area per target** and moved into place only after
    passing every check (snake folding, boundary normalization). Putting them at the final location
    before the checks would let a failed run leave "assets that pass the acceptance checks" — the
    io golden is taken from the same broken graph, so they agree with each other and the TS-side
    comparison **goes green** (the spelling of "when publishing is allowed" is
    {@link _shared.decode_series._publish}; the discipline of moving into place and cleaning up is
    the core primitive {@link karume.artifacts.staged_publication}).
    """
    source = DacvaeSource(source_dir)
    model = load_codec(source, model_dir)
    bypass_watermark(model.decoder)
    rate = frame_rate(model)
    decoder_max = int(DECODER_MAX_SECONDS * rate)
    encoder_max = int(ENCODER_MAX_SECONDS * rate)

    decoder_cases = build_decoder_cases(latent_dir, decoder_max)
    encoder_cases = build_encoder_cases(model, reference_wav, encoder_max, normalize_db)

    # ---- ① The reference before weight_norm is folded (the golden's expected values) and the
    # main-path measurement ----
    pristine = _pristine_outputs(model, decoder_cases, encoder_cases)
    main_path_evidence = _main_path_evidence(model.decoder, decoder_cases, model.quantizer.out_proj)

    # ---- ② Preprocessing (folding → alpha demotion) and the measurement that it did not
    # change values ----
    folded = fold_weight_norm(model)
    fold_evidence = _remove_weight_norm_evidence(model, decoder_cases, encoder_cases, pristine)
    lifted = lift_snake_alphas(source, model)
    # MUST: rounding comes after folding and the alpha demotion, before the truncation and the
    # retaken reference (the ordering of `_fake_quant`).
    quantized = _fake_quant(dtype, model)
    if quantized.report is not None:
        # The golden of a compressed series is taken again with **the rounded weights** (the
        # pristine outputs from before rounding were used up by check 3).
        pristine = _eager_outputs(model, decoder_cases, encoder_cases)
    trimmed = truncated_in_proj(model.quantizer.in_proj)
    truncation_evidence = _in_proj_truncation_evidence(model, trimmed, encoder_cases)

    # ---- ③ Eager equivalence of the wrappers (measures the three preprocessing steps
    # together here) ----
    graphs = {
        TARGET_DECODER: DecoderGraph(model.quantizer.out_proj, model.decoder),
        TARGET_ENCODER: EncoderGraph(model.encoder, trimmed),
    }
    graph_args: dict[str, dict[str, torch.Tensor]] = {
        TARGET_DECODER: dict(decoder_cases),
        TARGET_ENCODER: dict(encoder_cases),
    }
    equivalence = {
        target: max(
            _check_wrapper_equivalence(
                module, value, pristine[target][name], f"{target}/{name}", EAGER_EQUIV_ATOL
            )
            for name, value in graph_args[target].items()
        )
        for target, module in graphs.items()
    }
    roundtrip = _roundtrip_evidence(
        graphs[TARGET_ENCODER], graphs[TARGET_DECODER], dict(encoder_cases)[ENCODER_CASES[-1][0]]
    )

    # ---- ④ Export and writing the golden ----
    axes = {
        TARGET_DECODER: TargetAxis(DECODER_SYMBOL, decoder_max, 1),
        TARGET_ENCODER: TargetAxis(ENCODER_SYMBOL, encoder_max, 1),
    }
    written: dict[str, Any] = {}
    for target in targets:
        axis = axes[target]
        target_dir = out_dir / target
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        by_case = graph_args[target]
        longest = max(by_case, key=lambda name: by_case[name].shape[axis.axis])
        sequence = Dim(axis.symbol, min=MIN_SYM_LENGTH, max=axis.upper)
        with staged_publication(target_dir) as staged:
            # The writer creates the directory (the primitive does not create it — it only
            # hands over a path).
            staged.mkdir()
            graph = export_to_file(
                graphs[target],
                (by_case[longest],),
                staged / MODEL_FILE,
                provenance=PROVENANCE,
                # The graph name is **the component name** (= the key in karume.json's weights).
                # The codec's directory name (`decoder`) and key (`codec_decoder`) are spelled
                # differently, so it goes through the reverse lookup table (container-v1 §2.1).
                graph_name=IRODORI_CODEC_ROLES[target],
                dynamic_shapes=({axis.axis: sequence},),
                symbol_names=(axis.symbol,),
                weight_dtype=dtype,
                weight_scales=_target_scales(target, graphs[target], quantized.scales),
            )
            assert_snake_folded(graph, target)
            io_files = _write_io(graph, by_case, pristine[target], staged)
        written[target] = {**_graph_summary(graph, target_dir / MODEL_FILE), "io": io_files}
    return {
        "dir": str(out_dir),
        "dtype": dtype,
        "fake_quant": quantized.report,
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "targets": written,
        "hop_length": hop_length(model),
        "frame_rate": rate,
        "decoder_sym_max": decoder_max,
        "encoder_sym_max": encoder_max,
        "decoder_case_lengths": {name: int(z.shape[1]) for name, z in decoder_cases},
        "encoder_case_frames": {name: int(w.shape[1]) for name, w in encoder_cases},
        "reference_wav": str(reference_wav),
        "normalize_db": normalize_db,
        "weight_norm_folded": folded,
        "snake_alphas_lifted": lifted,
        "main_path_max_abs": main_path_evidence,
        "weight_norm_fold_max_abs": fold_evidence,
        "in_proj_truncation_max_abs": truncation_evidence,
        "eager_equivalence_max_abs": equivalence,
        "roundtrip": roundtrip,
    }


def default_out_root(model_dir: Path, dtype: str = "f32") -> Path:
    """Default location of the artifacts (`outputs/series/<directory name of the real
    weights>{,-f16,-i8}/`).

    `export_series` adds one level of subdirectory named after the target (the same shape as the
    other series).

    MUST: a separate directory per dtype (ADR 0018 / 0019 / 0027) — the spelling must match
    `IRODORI_CODEC_NAME` + the dtype suffix of `karume.dist` (writer and reader build it from the
    same word).
    """
    suffix = "" if dtype == "f32" else f"-{dtype}"
    return SERIES_ROOT / f"{model_dir.name}{suffix}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--latent-dir", type=Path, default=DEFAULT_LATENT_DIR)
    parser.add_argument("--reference-wav", type=Path, default=DEFAULT_REFERENCE_WAV)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: the series per --dtype —"
        " outputs/series/<directory name of --model-dir>{,-f16,-i8}/)",
    )
    parser.add_argument(
        "--dtype",
        choices=WEIGHT_DTYPES,
        default="f32",
        help="storage dtype of the weights (f16 / i8 fake-quantize first, then store only the"
        " eligible slots compressed — ADR 0018 / 0019 / 0027 / 0050. **emit only**)",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=TARGETS,
        help="narrows **only the writing** (can be repeated; default: all). Building the cases"
        " of both decoder and encoder and the 5 checks always run — the run time is the same"
        " even for one target",
    )
    args = parser.parse_args(argv)
    out_dir = default_out_root(args.model_dir, args.dtype) if args.out is None else args.out
    summary = export_series(
        args.model_dir,
        args.source_dir,
        out_dir,
        targets=tuple(args.target) if args.target else TARGETS,
        latent_dir=args.latent_dir,
        reference_wav=args.reference_wav,
        dtype=args.dtype,
    )
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
