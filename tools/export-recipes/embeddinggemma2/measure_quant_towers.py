"""Screen int8 weight forms for the vision and audio towers of EmbeddingGemma 2 on torch CPU (stage
A of the towers' quant presets) — the counterpart of {@link embeddinggemma2.measure_quant}, which
screened the text tower.

    uv run --offline --with 'transformers==5.23.0' python -m embeddinggemma2.measure_quant_towers \\
        --quality-set ../../inputs/embeddinggemma2/quality-set --date 2026-11-23 \\
        [--write-clips ../../inputs/embeddinggemma2/quality-set-16k]

The same method as the text screening: **no GPU code** — the weights are rounded in place by each
tower's own export rounding (`round_weights_to_i8` of `export`, `export_vision` and `export_audio`,
all through the exporter's `karume.quantize.fake_quant_int8`: symmetric, one F32 scale per output
row), so they are the values the i8 series ship, and every embedding is taken from the export's own
wrappers
({@link embeddinggemma2.export_vision.VisionTowerWrapper},
{@link embeddinggemma2.export_audio.AudioTowerWrapper},
{@link embeddinggemma2.export.EmbeddingWrapper}).
The reference is the same wrappers with the checkpoint's own weights (bf16 values upcast to f32,
run in f32).

## Configurations

| name        | text tower (219)  | vision tower    | audio tower     | measured on      |
| ----------- | ----------------- | --------------- | --------------- | ---------------- |
| `vision-i8` | checkpoint        | i8 per row      | checkpoint      | images           |
| `audio-i8`  | checkpoint        | checkpoint      | i8 per row      | audio clips      |
| `text-i8`   | i8 per row        | checkpoint      | checkpoint      | images and clips |
| `all-i8`    | i8 per row        | i8 per row      | i8 per row      | images and clips |

`text-i8` is the shipped `i8` preset as it is today
({@link embeddinggemma2.export.round_weights_to_i8}: the token table and every linear of the text
tower). `all-i8` rounds the three towers at once; an item passes through its own tower only, so
for an image it is "text + vision" and for a clip "text + audio".

## The towers' rounded sets ({@link tower_weights})

The set is what each tower's bf16 series stores as bf16, less what the tower's i8 series keeps
bf16, derived from the export code itself: the tower wrapper is traced by the export's own
`trace`, its storage planned by `karume.emit.stored_model` with the arguments `export_series`
passes for `--dtype bf16` (nothing is written), and the export's own `assert_storage_form` holds
the result to the bf16 series' form. Every tensor stored bf16 is then one of three kinds:

- **a module weight** (`nn.Linear`, `nn.Conv1d`, `nn.Conv2d` of the wrapper) — the audio tower's
  depthwise `conv1d` weights are in it (one scale per output channel, a row of 5);
- **a learned table read by `embedding`** that is not an `nn.Embedding` — the vision tower's two
  position tables `pos_x` / `pos_y` (`[10240, 768]`, the halves of upstream's table), which the
  export rounds as `embedding` weights (per position row) through `fake_quant_int8`'s
  `functional_weights`;
- **kept bf16** — what the audio i8 series keeps: its two subsampling convolutions and its one-hot
  window selector (`export_audio.i8_storage_overrides`, built on the one declaration
  `export_audio.I8_KEPT_BF16_WEIGHTS`, which holds the reason). The selector is no weight (its 0 /
  1 values are exact in bf16; rounding it per row would give 1.0 back exactly, so it moves no
  number either way).

A bf16 tensor of any other kind fails loudly. The rounding is each export's own; the keys it
reaches must equal the first two kinds ({@link mq.assert_rounded}), so the screening and the
shipped series cannot round different sets. The vision tower's folded RoPE tables are f32 in the
series ({@link embeddinggemma2.export_vision.ROPE_TABLES}) and untouched; norms, biases and every
folded constant (the audio tower's relative keys, query scales and window bias) are f32 and
untouched — the relative keys are `relative_k_proj` folded at export time, so that linear is not a
stored weight and is not rounded.

The first run of 2026-11-23 rounded the audio tower's two subsampling convolutions too; its report
is kept beside the re-run's (`docs/research/2026-11-23-embeddinggemma2-tower-quant-screen.md`).

MUST: configurations do not stack — every rounding is in place, so each starts from the pristine
weights ({@link restore_all}), each rounds exactly its intended set once
({@link mq.assert_rounded}), and a final restore must give the reference back bit for bit.

## Inputs

Every image and clip of the quality set (`--quality-set`, {@link load_quality_set}) and the recipe's
own synthetic golden images ({@link embeddinggemma2.images.IMAGES}) and waveforms
({@link embeddinggemma2.waveforms.WAVEFORMS}). Each is one input as the pipeline builds it: the
image or audio block alone, no text part and no prompt — `<bos>`, `<|image>` + n × `<|image|>` +
`<image|>` (or the audio's three), `<eos>` (checked against the multimodal fixture's `image-only`
and `audio-only` ids, and its vectors against the fixture's upstream ones). Images go through
upstream's image processor (PNG must be 8-bit RGB), clips through upstream's feature extractor at
16 kHz mono: a clip at another rate is resampled by {@link resample} — a Kaiser-windowed sinc
written here, because no resampling library is in the recipes' uv environment (no scipy, soxr,
librosa or torchaudio) — and every clip is put on the 16-bit PCM grid after it ({@link to_pcm16};
the count of clipped samples is in the report). A clip of at most 160 samples or more than 480,000
at 16 kHz is an error, as in the pipeline.

`--write-clips <dir>` also writes those clips as 16 kHz, 16-bit mono WAV files with the images
under a manifest of the same format ({@link write_clips}): the library takes 16 kHz only, and these
files hold exactly the samples measured here, so a run of the real pipeline on them reads the
screening's inputs.

The captions and transcripts are embedded as plain text with the `query` prompt
(`task: search result | query: `) — the model card's convention for retrieval: the query prefix
on the text side, and "pass images, video, and audio without any prefix".

## Metrics (per modality and configuration, at the widths 768 / 512 / 256 / 128)

1. cosine of each item's embedding against the reference: min / mean over all items, the quality
   set's and the synthetic ones, and the worst item named;
2. cosine of the tower's soft tokens against the reference's, per token: min / mean over every
   token (width-free — the soft tokens are 512 wide);
3. cross-modal retrieval over the quality set: item → text among the distinct captions /
   transcripts, text → item among the items. Several clips share a transcript: a hit is any item
   of the query's text (text → item) or the item's own text (item → text). Top-1 hits of the
   reference and of the configuration, top-1 agreement with the reference (the same candidate
   index), and the rank inversion rate over (query, candidate i < j) triples as the text screening
   defines it;
4. the largest absolute change of any entry of the item × text cosine matrix (every item, the
   modality's distinct texts);
5. the projected container payload of each tower in bf16 and in i8 ({@link tower_payload}).

All numbers go to `<out>/report.json`; the tables go to stdout, progress to stderr.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import wave
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from _shared.paths import BENCH_ROOT, INPUTS_ROOT, SERIES_ROOT
from embeddinggemma2 import export, export_audio, export_vision, multimodal_reference, sources
from embeddinggemma2 import measure_quant as mq
from embeddinggemma2.images import IMAGES
from embeddinggemma2.waveforms import PCM_SCALE, SAMPLE_RATE, WAVEFORMS, valid_frames
from karume.artifacts import staged_publication
from karume.container import container_parts
from karume.emit import stored_model
from karume.quantize import QUANT_MODULE_TYPES, Int8Report, iter_quant_targets

#: The quality set's default home (untracked, outside git — assets-layout: `inputs/<family>/`).
DEFAULT_QUALITY_SET = INPUTS_ROOT / "embeddinggemma2" / "quality-set"
MANIFEST_FILE = "manifest.json"
MANIFEST_FORMAT = "karume-embeddinggemma2-quality-set/1"
IMAGE_KEYS: Mapping[str, type] = {
    "file": str,
    "caption": str,
    "generator": str,
    "prompt": str,
    "seed": int,
}
AUDIO_KEYS: Mapping[str, type] = {
    "file": str,
    "transcript": str,
    "generator": str,
    "sampleRate": int,
}

#: Measurement artifacts are kept apart from the series (assets-layout: `outputs/bench/<model>/`).
BENCH_DIR = BENCH_ROOT / "embeddinggemma-2"
OUT_SUFFIX = "_tower-quant-screen"
REPORT_FILE = "report.json"

#: The prompt of the text side (module doc, "Inputs").
TEXT_PROMPT = export.QUERY

IMAGE = "image"
AUDIO = "audio"
MODALITIES = (IMAGE, AUDIO)
QUALITY = "qualitySet"
SYNTHETIC = "synthetic"

#: The towers (the namespaces of the rounded keys) and the tower of each modality.
TEXT_TOWER = "text"
VISION_TOWER = "vision"
AUDIO_TOWER = "audio"
TOWERS = (TEXT_TOWER, VISION_TOWER, AUDIO_TOWER)
MODALITY_TOWER: Mapping[str, str] = {IMAGE: VISION_TOWER, AUDIO: AUDIO_TOWER}

#: An item's soft rows start after `<bos>` and the begin marker.
SOFT_START = 2

#: An audio needs more than this many samples at 16 kHz for one mel frame (the pipeline's floor).
MIN_SAMPLES_EXCLUSIVE = 160
MAX_SAMPLES = export_audio.MAX_SAMPLES

#: The resampler ({@link resample}): zero crossings of the sinc on each side, the Kaiser window's
#: beta, and the cutoff as a share of the lower Nyquist frequency.
RESAMPLE_ZEROS = 32
KAISER_BETA = 8.6
RESAMPLE_ROLLOFF = 0.95
RESAMPLE_CHUNK = 8192

#: The int8 rounding: the payload bits per form (`docs/container-v1.md` §6 — F32 scales).
BITS_PER_BYTE = 8
MIB = 1024 * 1024
REFERENCE = mq.REFERENCE


@dataclass(frozen=True)
class TowerConfig:
    """One configuration: the towers whose weights are rounded to i8 and the modalities measured."""

    name: str
    towers: frozenset[str]
    modalities: tuple[str, ...]


CONFIGS: tuple[TowerConfig, ...] = (
    TowerConfig("vision-i8", frozenset({VISION_TOWER}), (IMAGE,)),
    TowerConfig("audio-i8", frozenset({AUDIO_TOWER}), (AUDIO,)),
    TowerConfig("text-i8", frozenset({TEXT_TOWER}), MODALITIES),
    TowerConfig("all-i8", frozenset(TOWERS), MODALITIES),
)


class QualitySetError(ValueError):
    """The quality set is not what this script measures (fails before any model is loaded)."""


# ---- The quality set (pure) ----------------------------------------------------------------


@dataclass(frozen=True)
class ImageEntry:
    file: str
    caption: str
    generator: str
    prompt: str
    seed: int


@dataclass(frozen=True)
class AudioEntry:
    file: str
    transcript: str
    generator: str
    sample_rate: int


@dataclass(frozen=True)
class QualitySet:
    root: Path
    images: tuple[ImageEntry, ...]
    audio: tuple[AudioEntry, ...]


def _entry_fields(entry: Any, keys: Mapping[str, type], where: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise QualitySetError(f"{where}: not an object")
    if set(entry) != set(keys):
        raise QualitySetError(f"{where}: keys {sorted(entry)}, not {sorted(keys)}")
    for key, kind in keys.items():
        value = entry[key]
        # bool is an int subclass; a seed or a rate of `true` is not a number.
        if isinstance(value, bool) or not isinstance(value, kind):
            raise QualitySetError(f"{where}.{key}: {value!r} is not a {kind.__name__}")
        if kind is str and not value.strip():
            raise QualitySetError(f"{where}.{key}: empty")
    return dict(entry)


def _relative_file(root: Path, name: str, folder: str, suffix: str, where: str) -> str:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 2
        or path.parts[0] != folder
        or path.suffix != suffix
    ):
        raise QualitySetError(f"{where}.file: {name!r} is not {folder}/<name>{suffix}")
    if not (root / path).is_file():
        raise QualitySetError(f"{where}.file: {root / path} does not exist")
    return name


def validate_manifest(document: Any, root: Path) -> QualitySet:
    """The manifest's form (`MANIFEST_FORMAT`), checked field by field; every file must exist
    under `root`.

    MUST: each modality holds at least two items and two distinct texts — retrieval and the rank
    inversion rate are not defined on fewer.
    """
    if not isinstance(document, dict) or set(document) != {"format", "images", "audio"}:
        keys = sorted(document) if isinstance(document, dict) else type(document).__name__
        raise QualitySetError(f"the manifest's top level is {keys}, not format / images / audio")
    if document["format"] != MANIFEST_FORMAT:
        raise QualitySetError(
            f"the manifest's format is {document['format']!r}, not {MANIFEST_FORMAT!r}"
        )
    images: list[ImageEntry] = []
    audio: list[AudioEntry] = []
    for kind, keys, folder, suffix in (
        ("images", IMAGE_KEYS, "images", ".png"),
        ("audio", AUDIO_KEYS, "audio", ".wav"),
    ):
        entries = document[kind]
        if not isinstance(entries, list):
            raise QualitySetError(f"{kind}: not a list")
        for index, raw in enumerate(entries):
            where = f"{kind}[{index}]"
            fields = _entry_fields(raw, keys, where)
            fields["file"] = _relative_file(root, fields["file"], folder, suffix, where)
            if kind == "images":
                images.append(ImageEntry(**fields))
            else:
                if fields["sampleRate"] <= 0:
                    raise QualitySetError(f"{where}.sampleRate: {fields['sampleRate']} is not > 0")
                fields["sample_rate"] = fields.pop("sampleRate")
                audio.append(AudioEntry(**fields))
    for kind, files, texts in (
        ("images", [entry.file for entry in images], [entry.caption for entry in images]),
        ("audio", [entry.file for entry in audio], [entry.transcript for entry in audio]),
    ):
        if len(set(files)) != len(files):
            raise QualitySetError(f"{kind}: a file is listed twice")
        if len(files) < 2 or len(set(texts)) < 2:
            raise QualitySetError(
                f"{kind}: {len(files)} items with {len(set(texts))} distinct texts"
                " (at least 2 of each)"
            )
    return QualitySet(root=root, images=tuple(images), audio=tuple(audio))


def load_quality_set(directory: Path) -> QualitySet:
    manifest = directory / MANIFEST_FILE
    if not manifest.is_file():
        raise QualitySetError(f"{manifest} does not exist")
    return validate_manifest(json.loads(manifest.read_text(encoding="utf-8")), directory)


# ---- Reading and resampling (pure) ---------------------------------------------------------


def read_png_rgb8(path: Path) -> np.ndarray:
    """The pixels of an 8-bit RGB PNG as `[height, width, 3]` uint8 (any other mode fails)."""
    from PIL import Image

    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise QualitySetError(f"{path}: {image.format} {image.mode}, not an 8-bit RGB PNG")
        return np.ascontiguousarray(np.array(image, dtype=np.uint8))


def read_wav_pcm16(path: Path) -> tuple[np.ndarray, int]:
    """`(pcm [frames, channels] int16, rate)` of a 16-bit PCM WAV (anything else fails)."""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getcomptype() != "NONE":
            raise QualitySetError(
                f"{path}: {8 * handle.getsampwidth()}-bit {handle.getcomptype()}, not 16-bit PCM"
            )
        channels, rate, frames = handle.getnchannels(), handle.getframerate(), handle.getnframes()
        raw = handle.readframes(frames)
    pcm = np.frombuffer(raw, dtype="<i2").reshape(frames, channels)
    return pcm.astype(np.int16), rate


def pcm_to_mono(pcm: np.ndarray) -> np.ndarray:
    """`pcm / 32768` (what a WAV reader gives — `waveforms.py`), the channels averaged, in f64."""
    return (pcm.astype(np.float64) / PCM_SCALE).mean(axis=1)


def resampled_length(samples: int, rate_in: int, rate_out: int) -> int:
    """`ceil(samples · rate_out / rate_in)` (the count of output instants inside the input)."""
    return -(-samples * rate_out // rate_in)


def resample(values: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    """Band-limited resampling of a 1-D f64 signal: each output sample `m` at the input instant
    `t = m · rate_in / rate_out` is `Σ x[k] · c · sinc(c · (t − k)) · w(t − k)` with the cutoff
    `c = ROLLOFF · min(1, rate_out / rate_in)` (in cycles per input sample × 2) and a Kaiser window
    `w` over {@link RESAMPLE_ZEROS} zero crossings on each side; samples outside the signal are 0.
    An equal rate returns a copy (nothing filtered)."""
    if rate_in <= 0 or rate_out <= 0 or values.ndim != 1:
        raise ValueError(f"resample takes a 1-D signal and positive rates ({rate_in}, {rate_out})")
    if rate_in == rate_out:
        return values.astype(np.float64, copy=True)
    divisor = math.gcd(rate_in, rate_out)
    up, down = rate_out // divisor, rate_in // divisor
    cutoff = RESAMPLE_ROLLOFF * min(1.0, rate_out / rate_in)
    half = math.ceil(RESAMPLE_ZEROS / cutoff)
    offsets = np.arange(-half + 1, half + 1)
    padded = np.concatenate([np.zeros(half), values.astype(np.float64), np.zeros(half + 1)])
    length = resampled_length(int(values.shape[0]), rate_in, rate_out)
    output = np.empty(length, dtype=np.float64)
    norm = float(np.i0(KAISER_BETA))
    for start in range(0, length, RESAMPLE_CHUNK):
        index = np.arange(start, min(start + RESAMPLE_CHUNK, length), dtype=np.int64)
        base = index * down // up
        fraction = (index * down % up) / up
        taps = base[:, None] + offsets[None, :]
        distance = fraction[:, None] - offsets[None, :]
        inside = np.clip(1.0 - (distance / half) ** 2, 0.0, None)
        window = np.i0(KAISER_BETA * np.sqrt(inside)) / norm
        kernel = cutoff * np.sinc(cutoff * distance) * window
        output[index] = (padded[taps + half] * kernel).sum(axis=1)
    return output


def check_audio_length(samples: int, where: str) -> None:
    """The pipeline's limits at 16 kHz: more than 160 samples (one mel frame), at most 480,000."""
    if samples <= MIN_SAMPLES_EXCLUSIVE:
        raise QualitySetError(
            f"{where}: {samples} samples at 16 kHz — no mel frame (more than"
            f" {MIN_SAMPLES_EXCLUSIVE} needed)"
        )
    if samples > MAX_SAMPLES:
        raise QualitySetError(f"{where}: {samples} samples at 16 kHz, more than {MAX_SAMPLES}")


def to_pcm16(values: np.ndarray) -> tuple[np.ndarray, int]:
    """A signal at full scale ±1 on the 16-bit PCM grid: `rint(x · 32768)` clipped to the int16
    range; returns the samples and how many of them were clipped."""
    scaled = np.rint(values * PCM_SCALE)
    clipped = int(((scaled < -PCM_SCALE) | (scaled > PCM_SCALE - 1)).sum())
    return np.clip(scaled, -PCM_SCALE, PCM_SCALE - 1).astype(np.int16), clipped


def clip_pcm16(path: Path, declared_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
    """A quality-set clip at 16 kHz mono as 16-bit PCM (resampled, then rounded to the grid),
    within the limits."""
    pcm, rate = read_wav_pcm16(path)
    if rate != declared_rate:
        raise QualitySetError(
            f"{path}: the file is at {rate} Hz, the manifest says {declared_rate}"
        )
    samples, clipped = to_pcm16(resample(pcm_to_mono(pcm), rate, SAMPLE_RATE))
    check_audio_length(int(samples.shape[0]), str(path))
    info = {
        "sourceRate": rate,
        "channels": int(pcm.shape[1]),
        "sourceSamples": int(pcm.shape[0]),
        "samples": int(samples.shape[0]),
        "resampled": rate != SAMPLE_RATE,
        "clippedSamples": clipped,
    }
    return samples, info


def clip_samples(path: Path, declared_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
    """A quality-set clip as the pipeline takes it: f32 mono at 16 kHz, within the limits.

    MUST: the samples are those of the 16-bit WAV {@link write_clips} writes (`pcm / 32768`, exact
    in f32), so the real pipeline fed those files reads exactly what this screening measured; a
    clip already at 16 kHz mono is its own PCM unchanged."""
    pcm, info = clip_pcm16(path, declared_rate)
    return pcm.astype(np.float32) / np.float32(PCM_SCALE), info


def write_wav_pcm16(path: Path, pcm: np.ndarray, rate: int) -> None:
    """A mono 16-bit PCM WAV (little-endian, the form {@link read_wav_pcm16} reads)."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.astype("<i2").tobytes())


def write_clips(quality: QualitySet, directory: Path) -> dict[str, Any]:
    """`--write-clips`: the quality set with every clip as this screening reads it — 16 kHz, 16-bit
    mono WAV ({@link clip_pcm16}) — under a manifest of the same format whose clips say 16,000 Hz,
    and its images copied byte for byte (the format holds both modalities). The library takes 16
    kHz only, so these are the files a run of the real pipeline reads; this script run on the
    directory reads the same samples again (an equal rate is not resampled). Staged — a failure
    leaves a previous directory as it was."""
    images: list[dict[str, Any]] = []
    audio: list[dict[str, Any]] = []
    clipped: dict[str, int] = {}
    with staged_publication(directory) as staged:
        (staged / "images").mkdir(parents=True)
        (staged / "audio").mkdir()
        for image in quality.images:
            shutil.copyfile(quality.root / image.file, staged / image.file)
            images.append(
                {
                    "file": image.file,
                    "caption": image.caption,
                    "generator": image.generator,
                    "prompt": image.prompt,
                    "seed": image.seed,
                }
            )
        for clip in quality.audio:
            pcm, info = clip_pcm16(quality.root / clip.file, clip.sample_rate)
            write_wav_pcm16(staged / clip.file, pcm, SAMPLE_RATE)
            clipped[clip.file] = info["clippedSamples"]
            audio.append(
                {
                    "file": clip.file,
                    "transcript": clip.transcript,
                    "generator": clip.generator,
                    "sampleRate": SAMPLE_RATE,
                }
            )
        manifest = {"format": MANIFEST_FORMAT, "images": images, "audio": audio}
        (staged / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        validate_manifest(manifest, staged)
    return {
        "dir": str(directory),
        "images": len(images),
        "audio": len(audio),
        "sampleRate": SAMPLE_RATE,
        "clippedSamples": clipped,
    }


# ---- Retrieval and cosine metrics (pure) ---------------------------------------------------


def distinct_texts(texts: Sequence[str]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """The distinct texts in the order of their first appearance, and each item's text index."""
    order: dict[str, int] = {}
    for text in texts:
        order.setdefault(text, len(order))
    return tuple(order), tuple(order[text] for text in texts)


def item_to_text_hits(similarity: torch.Tensor, item_text: Sequence[int]) -> int:
    """How many items (rows) rank their own text first among the texts (columns)."""
    return int((similarity.argmax(dim=1) == torch.tensor(list(item_text))).sum())


def text_to_item_hits(similarity: torch.Tensor, item_text: Sequence[int]) -> int:
    """How many texts (rows) rank first an item (column) whose text they are — any item of a
    shared text counts."""
    best = similarity.argmax(dim=1)
    owners = torch.tensor(list(item_text))[best]
    return int((owners == torch.arange(int(similarity.shape[0]))).sum())


def retrieval_metrics(
    similarity: torch.Tensor,
    reference: torch.Tensor,
    hits: Callable[[torch.Tensor, Sequence[int]], int],
    item_text: Sequence[int],
) -> dict[str, Any]:
    """Top-1 hits of the reference and of the configuration, top-1 agreement (the same candidate),
    and the rank inversion rate over (query, i < j) triples."""
    return {
        "queries": int(similarity.shape[0]),
        "candidates": int(similarity.shape[1]),
        "referenceTop1": hits(reference, item_text),
        "top1": hits(similarity, item_text),
        "top1Agreement": mq.top1_agreement(similarity, reference),
        "rankInversionRate": mq.rank_inversion_rate(similarity, reference),
    }


def row_cosines(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Cosine of each row (f64)."""
    return functional.cosine_similarity(values.double(), reference.double(), dim=-1)


def cosine_groups(
    cosines: torch.Tensor, names: Sequence[str], kinds: Sequence[str]
) -> dict[str, Any]:
    """min / mean over all items and over each kind present, and the worst item named."""
    summary: dict[str, Any] = {}
    for label in ("all", QUALITY, SYNTHETIC):
        keep = [index for index, kind in enumerate(kinds) if label in ("all", kind)]
        if not keep:
            continue
        chosen = cosines[keep]
        summary[label] = {
            "items": len(keep),
            "min": float(chosen.min()),
            "mean": float(chosen.mean()),
        }
    summary["worst"] = names[int(cosines.argmin())]
    return summary


def soft_token_summary(
    values: Sequence[torch.Tensor], reference: Sequence[torch.Tensor], names: Sequence[str]
) -> dict[str, Any]:
    """Per-token cosine of each item's soft tokens against the reference's: min / mean over every
    token of every item, and each item's min / mean."""
    per_item = [row_cosines(v, r) for v, r in zip(values, reference, strict=True)]
    every = torch.cat(per_item)
    worst = min(range(len(per_item)), key=lambda index: float(per_item[index].min()))
    return {
        "tokens": int(every.numel()),
        "min": float(every.min()),
        "mean": float(every.mean()),
        "worst": names[worst],
        "items": {
            name: {"tokens": int(c.numel()), "min": float(c.min()), "mean": float(c.mean())}
            for name, c in zip(names, per_item, strict=True)
        },
    }


@dataclass(frozen=True)
class ModalityItems:
    """The items of one modality as the metrics see them (rows of every matrix in this order)."""

    names: tuple[str, ...]
    kinds: tuple[str, ...]
    #: The text index of each quality-set item (synthetic items have none).
    item_text: tuple[int | None, ...]
    texts: tuple[str, ...]

    @property
    def quality_rows(self) -> list[int]:
        return [index for index, kind in enumerate(self.kinds) if kind == QUALITY]


def measure_width(
    items: ModalityItems,
    vectors: torch.Tensor,
    reference: torch.Tensor,
    text_vectors: torch.Tensor,
    reference_texts: torch.Tensor,
    width: int,
) -> dict[str, Any]:
    """Metrics 1, 3 and 4 of one configuration at one width (module doc)."""
    values, base = mq.matryoshka(vectors, width), mq.matryoshka(reference, width)
    texts, base_texts = mq.matryoshka(text_vectors, width), mq.matryoshka(reference_texts, width)
    cosines = row_cosines(values, base)
    rows = items.quality_rows
    item_text = [text for text in items.item_text if text is not None]
    item_by_text = values[rows] @ texts.T
    reference_item_by_text = base[rows] @ base_texts.T
    return {
        "cosine": cosine_groups(cosines, items.names, items.kinds),
        "itemCosines": {name: float(c) for name, c in zip(items.names, cosines, strict=True)},
        "itemToText": retrieval_metrics(
            item_by_text, reference_item_by_text, item_to_text_hits, item_text
        ),
        "textToItem": retrieval_metrics(
            item_by_text.T, reference_item_by_text.T, text_to_item_hits, item_text
        ),
        "itemTextCosineMaxAbsChange": float((values @ texts.T - base @ base_texts.T).abs().max()),
    }


# ---- Size projection (pure) ----------------------------------------------------------------


def tower_payload(
    rounded: mq.GroupStats, kept_bf16: mq.GroupStats, remainder_elements: int
) -> dict[str, Any]:
    """The projected payload of a tower: bf16 = `16·N` of every bf16 tensor + `32·` the f32
    remainder; i8 = `8·N + 32·rows` of the rounded set + `16·N` of the kept bf16 set + the same
    remainder."""
    remainder_bits = 32 * remainder_elements
    kept_bits = mq.projected_bits("bf16", kept_bf16)
    forms = {
        "bf16": mq.projected_bits("bf16", rounded) + kept_bits + remainder_bits,
        "i8": mq.projected_bits("i8", rounded) + kept_bits + remainder_bits,
    }
    kept = f" + kept 16·{kept_bf16.elements:,}" if kept_bf16.elements else ""
    arithmetic = {
        "bf16": f"rounded set 16·{rounded.elements:,}{kept} + remainder 32·{remainder_elements:,}",
        "i8": (
            f"rounded set 8·{rounded.elements:,} + 32·{rounded.rows:,}{kept}"
            f" + remainder 32·{remainder_elements:,}"
        ),
    }
    return {
        form: {
            "bytes": bits // BITS_PER_BYTE,
            "mib": bits / BITS_PER_BYTE / MIB,
            "arithmetic": arithmetic[form] + " bits",
        }
        for form, bits in forms.items()
    }


# ---- The towers' rounded sets --------------------------------------------------------------


def classify_bf16_set(
    bf16_keys: Collection[str],
    module_weights: Collection[str],
    tables: Collection[str],
    structural: Collection[str],
) -> tuple[list[str], list[str], list[str]]:
    """Split the series' bf16 tensors into `(module weights, tables, structural constants)`, each in
    the order of `bf16_keys`. A module weight outside the bf16 set (a module the graph never
    reads) is left out.

    MUST: every bf16 tensor is exactly one kind — one of no kind would be shipped bf16 by an i8
    series nobody screened, and one of two kinds would be rounded twice.
    """
    groups: tuple[list[str], list[str], list[str]] = ([], [], [])
    for key in bf16_keys:
        kinds = [key in module_weights, key in tables, key in structural]
        if sum(kinds) != 1:
            raise AssertionError(
                f"the bf16 tensor {key!r} is {sum(kinds)} kinds (module weight / table /"
                " structural constant)"
            )
        groups[kinds.index(True)].append(key)
    return groups


@dataclass
class TowerWeights:
    """One tower's rounded set, with the live tensors it rounds and their pristine copies."""

    name: str
    module_weights: list[str]
    tables: list[str]
    kept_bf16: list[str]
    rounded_stats: mq.GroupStats
    kept_stats: mq.GroupStats
    remainder_elements: int
    remainder_tensors: int
    live: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def keys(self) -> list[str]:
        return [f"{self.name}:{key}" for key in (*self.module_weights, *self.tables)]


def _stats(shapes: Sequence[tuple[torch.Tensor, int]]) -> mq.GroupStats:
    return mq.GroupStats(
        tensors=len(shapes),
        elements=sum(int(tensor.numel()) for tensor, _axis in shapes),
        rows=sum(int(tensor.shape[axis]) for tensor, axis in shapes),
    )


def tower_weights(
    name: str,
    wrapper: nn.Module,
    graph: Any,
    tensors: Mapping[str, torch.Tensor],
    overrides: Mapping[str, str] | None,
    assert_storage_form: Callable[[Any, str], Any],
    tables: Collection[str],
    kept_bf16: Collection[str],
) -> TowerWeights:
    """The tower's rounded set (module doc, "The towers' rounded sets") from its traced graph, the
    storage `stored_model` plans for the bf16 series, and the tensors the tower's i8 series keeps
    bf16 (`kept_bf16` — the export's own declaration)."""
    stored = stored_model(graph, tensors, weight_dtype="bf16", weight_dtype_overrides=overrides)
    assert_storage_form(stored.graph, "bf16")
    storage: dict[str, str] = {}
    for initializer in stored.graph.initializers.values():
        if initializer.tensor is None:
            raise AssertionError(f"{name}: a shared initializer in the tower graph")
        storage[initializer.tensor] = initializer.storage.dtype
    if set(storage.values()) - {"bf16", "f32", "i32"}:
        raise AssertionError(f"{name}: storage dtypes {sorted(set(storage.values()))}")
    bf16 = [key for key, dtype in storage.items() if dtype == "bf16"]
    if not set(kept_bf16) <= set(bf16):
        raise AssertionError(
            f"{name}: the tensors kept bf16 {sorted(set(kept_bf16) - set(bf16))} are not bf16 in"
            " the bf16 series"
        )
    targets = {
        fqn: (weight, axis) for fqn, weight, axis in iter_quant_targets(wrapper, QUANT_MODULE_TYPES)
    }
    module_weights, table_keys, kept = classify_bf16_set(
        bf16, set(targets) - set(kept_bf16), tables, kept_bf16
    )
    if set(table_keys) != set(tables):
        raise AssertionError(
            f"{name}: the tables stored bf16 are {table_keys}, not {sorted(tables)}"
        )
    parameters = dict(wrapper.named_parameters(remove_duplicate=False))
    buffers = dict(wrapper.named_buffers(remove_duplicate=False))
    live: dict[str, torch.Tensor] = {}
    shapes: list[tuple[torch.Tensor, int]] = []
    for key in module_weights:
        weight, axis = targets[key]
        live[f"{name}:{key}"] = weight
        shapes.append((weight, axis))
    for key in table_keys:
        table = parameters[key]
        live[f"{name}:{key}"] = table
        shapes.append((table, 0))
    kept_tensors = [(parameters[key] if key in parameters else buffers[key], 0) for key in kept]
    for key, (tensor, _axis) in zip(
        (*module_weights, *table_keys, *kept), (*shapes, *kept_tensors), strict=True
    ):
        if tensors[key].shape != tensor.shape:
            raise AssertionError(
                f"{name}: {key} differs in shape between the graph and the wrapper"
            )
    remainder = [key for key, dtype in storage.items() if dtype != "bf16"]
    return TowerWeights(
        name=name,
        module_weights=module_weights,
        tables=table_keys,
        kept_bf16=kept,
        rounded_stats=_stats(shapes),
        kept_stats=_stats(kept_tensors),
        remainder_elements=sum(int(tensors[key].numel()) for key in remainder),
        remainder_tensors=len(remainder),
        live=live,
    )


def text_weights(wrapper: nn.Module) -> TowerWeights:
    """The text tower's rounded set: the shipped `i8` preset's (every linear and the token table —
    {@link embeddinggemma2.export.round_weights_to_i8} rounds them)."""
    live: dict[str, torch.Tensor] = {}
    shapes: list[tuple[torch.Tensor, int]] = []
    keys: list[str] = []
    for fqn, weight, axis in iter_quant_targets(wrapper, (nn.Linear, nn.Embedding)):
        keys.append(fqn)
        live[f"{TEXT_TOWER}:{fqn}"] = weight
        shapes.append((weight, axis))
    remainder = {
        key: int(value.numel())
        for key, value in wrapper.state_dict().items()
        if f"{TEXT_TOWER}:{key}" not in live
    }
    return TowerWeights(
        name=TEXT_TOWER,
        module_weights=keys,
        tables=[],
        kept_bf16=[],
        rounded_stats=_stats(shapes),
        kept_stats=mq.GroupStats(0, 0, 0),
        remainder_elements=sum(remainder.values()),
        remainder_tensors=len(remainder),
        live=live,
    )


#: The rounding each tower's i8 series ships (the export's own function — module doc).
ROUNDINGS: Mapping[str, Callable[[Any], Int8Report]] = {
    TEXT_TOWER: export.round_weights_to_i8,
    VISION_TOWER: export_vision.round_weights_to_i8,
    AUDIO_TOWER: export_audio.round_weights_to_i8,
}


def round_tower(wrapper: nn.Module, weights: TowerWeights) -> list[str]:
    """Round the tower's set in place with the rounding its i8 series ships; returns the
    namespaced keys rounded."""
    report = ROUNDINGS[weights.name](wrapper)
    return [f"{weights.name}:{fqn}" for fqn in report.scales]


def restore_all(towers: Mapping[str, TowerWeights], pristine: Mapping[str, torch.Tensor]) -> None:
    for weights in towers.values():
        mq.restore(weights.live, pristine)


# ---- The models ----------------------------------------------------------------------------


@dataclass
class Models:
    text: export.EmbeddingWrapper
    vision: export_vision.VisionTowerWrapper
    audio: export_audio.AudioTowerWrapper
    processor: Any
    extractor: Any
    #: The composite config with its audio part checked (`export_audio.load_config`).
    audio_config: Any
    tokenizer: Any
    prompts: Mapping[str, str]
    hidden: int
    bos: int
    eos: int
    #: modality → (begin, soft, end) ids.
    markers: Mapping[str, tuple[int, int, int]]
    placeholders: frozenset[int]
    wrappers: Mapping[str, nn.Module]


READ_FILES: tuple[str, ...] = tuple(
    dict.fromkeys((*export.READ_FILES, *export_vision.READ_FILES, *export_audio.READ_FILES))
)


def load_models(directory: Path) -> Models:
    from tokenizers import Tokenizer

    sources.assert_pinned(directory, READ_FILES)
    export.assert_sentence_transformer_layout(directory)
    model, composite = export.load_text_model(directory)
    text = export.EmbeddingWrapper(model, int(model.config.sliding_window)).eval()
    vision = export_vision.VisionTowerWrapper(export_vision.load_tower(directory)).eval()
    audio = export_audio.AudioTowerWrapper(export_audio.load_tower(directory)).eval()
    config = export_vision.load_config(directory)
    return Models(
        text=text,
        vision=vision,
        audio=audio,
        processor=export_vision.load_image_processor(directory),
        extractor=export_audio.load_feature_extractor(directory),
        audio_config=export_audio.load_config(directory),
        tokenizer=Tokenizer.from_file(str(directory / "tokenizer.json")),
        prompts=export.load_prompts(directory),
        hidden=int(model.config.hidden_size),
        bos=int(config.text_config.bos_token_id),
        eos=int(config.text_config.eos_token_id),
        markers={
            IMAGE: (int(config.boi_token_id), int(config.image_token_id), int(config.eoi_token_id)),
            AUDIO: (
                int(config.boa_token_id),
                int(config.audio_token_id),
                int(config.eoa_token_index),
            ),
        },
        placeholders=frozenset(
            {
                int(composite.image_token_id),
                int(composite.video_token_id),
                int(composite.audio_token_id),
            }
        ),
        wrappers={TEXT_TOWER: text, VISION_TOWER: vision, AUDIO_TOWER: audio},
    )


def item_ids(bos: int, eos: int, markers: tuple[int, int, int], soft_tokens: int) -> list[int]:
    """The ids of an input holding one image or audio block alone (module doc, "Inputs")."""
    begin, soft, end = markers
    return [bos, begin, *([soft] * soft_tokens), end, eos]


@dataclass(frozen=True)
class Item:
    """One image or clip ready for its tower."""

    name: str
    modality: str
    kind: str
    text: str | None
    #: `HostRows` (an image) or `HostFrames` (a clip).
    inputs: Any
    info: Mapping[str, Any]

    @property
    def soft_tokens(self) -> int:
        return int(self.inputs.soft_tokens)


def prepare_image(models: Models, rgb8: np.ndarray, name: str) -> tuple[Any, dict[str, Any]]:
    """Upstream's processor on the pixels, the rows as the graph takes them."""
    from PIL import Image

    features = models.processor(images=[Image.fromarray(rgb8, mode="RGB")], return_tensors="pt")
    rows = export_vision.host_rows(features["pixel_values"], features["image_position_ids"])
    expected = int(features["num_soft_tokens_per_image"][0])
    if rows.soft_tokens != expected:
        raise AssertionError(
            f"{name}: {rows.soft_tokens} soft tokens, the processor says {expected}"
        )
    return rows, {"width": int(rgb8.shape[1]), "height": int(rgb8.shape[0]), "softTokens": expected}


def prepare_clip(models: Models, samples: np.ndarray, name: str) -> tuple[Any, dict[str, Any]]:
    """Upstream's feature extractor on f32 16 kHz samples (a list of one, as its processor hands
    them over), the frames as the graph takes them."""
    check_audio_length(int(samples.shape[0]), name)
    output = models.extractor([samples], return_tensors="pt")
    frames = export_audio.host_frames(output["input_features"], output["input_features_mask"])
    if frames.valid != valid_frames(int(samples.shape[0])):
        raise AssertionError(
            f"{name}: {frames.valid} frames, the rule says {valid_frames(samples.shape[0])}"
        )
    return frames, {"validFrames": frames.valid, "softTokens": frames.soft_tokens}


def prepare_items(models: Models, quality: QualitySet) -> dict[str, list[Item]]:
    images: list[Item] = []
    for entry in quality.images:
        rows, info = prepare_image(models, read_png_rgb8(quality.root / entry.file), entry.file)
        images.append(Item(entry.file, IMAGE, QUALITY, entry.caption, rows, info))
    for name, image in IMAGES.items():
        rows, info = prepare_image(models, image.rgb8(), name)
        images.append(Item(f"{SYNTHETIC}:{name}", IMAGE, SYNTHETIC, None, rows, info))
    clips: list[Item] = []
    for entry in quality.audio:
        samples, info = clip_samples(quality.root / entry.file, entry.sample_rate)
        frames, more = prepare_clip(models, samples, entry.file)
        clips.append(Item(entry.file, AUDIO, QUALITY, entry.transcript, frames, {**info, **more}))
    for name, waveform in WAVEFORMS.items():
        frames, info = prepare_clip(models, waveform.f32(), name)
        info = {"samples": waveform.samples, **info}
        clips.append(Item(f"{SYNTHETIC}:{name}", AUDIO, SYNTHETIC, None, frames, info))
    return {IMAGE: images, AUDIO: clips}


def soft_tokens_of(models: Models, item: Item) -> torch.Tensor:
    with torch.no_grad():
        if item.modality == IMAGE:
            rows = item.inputs
            return models.vision(rows.pixels, rows.position_x, rows.position_y)[0].clone()
        frames = item.inputs
        return models.audio(frames.features, frames.frame_mask)[0].clone()


def embed_item(models: Models, item: Item, soft: torch.Tensor) -> torch.Tensor:
    ids = torch.tensor(
        [item_ids(models.bos, models.eos, models.markers[item.modality], int(soft.shape[0]))],
        dtype=torch.int64,
    )
    soft_embeds, soft_mask = export.soft_inputs(ids, models.hidden, [(SOFT_START, soft)])
    with torch.no_grad():
        return models.text(ids, torch.ones_like(ids, dtype=torch.float32), soft_embeds, soft_mask)[
            0
        ]


def text_ids(models: Models, text: str) -> torch.Tensor:
    """`[1, T]` ids of a caption / transcript under the text prompt, `<bos>` … `<eos>` checked
    and no placeholder allowed."""
    ids = models.tokenizer.encode(models.prompts[TEXT_PROMPT] + text).ids
    if ids[0] != models.bos or ids[-1] != models.eos:
        raise AssertionError(f"{text!r}: the ids do not run <bos> … <eos>")
    if models.placeholders.intersection(ids) or len(ids) > export.SYM_MAX:
        raise AssertionError(f"{text!r}: a placeholder id or more than {export.SYM_MAX} tokens")
    return torch.tensor([ids], dtype=torch.int64)


def embed_texts(models: Models, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack(
            [models.text.text(ids, torch.ones_like(ids, dtype=torch.float32))[0] for ids in inputs]
        ).clone()


@dataclass
class Run:
    """One configuration's outputs of one modality."""

    soft: list[torch.Tensor]
    vectors: torch.Tensor
    texts: torch.Tensor
    seconds: dict[str, dict[str, float]]


def run_modality(
    models: Models, items: Sequence[Item], text_inputs: Sequence[torch.Tensor], label: str
) -> Run:
    soft: list[torch.Tensor] = []
    vectors: list[torch.Tensor] = []
    seconds: dict[str, dict[str, float]] = {}
    for item in items:
        started = time.perf_counter()
        tokens = soft_tokens_of(models, item)
        tower = time.perf_counter()
        vectors.append(embed_item(models, item, tokens))
        done = time.perf_counter()
        soft.append(tokens)
        seconds[item.name] = {
            "tower": round(tower - started, 3),
            "text": round(done - tower, 3),
            "total": round(done - started, 3),
        }
        _log(f"[{label}] {item.name}: {item.soft_tokens} soft tokens, {done - started:.2f}s")
    return Run(soft, torch.stack(vectors).clone(), embed_texts(models, text_inputs), seconds)


# ---- Checks against the multimodal fixture -------------------------------------------------


def fixture_checks(
    models: Models, items: Mapping[str, Sequence[Item]], runs: Mapping[str, Run]
) -> dict[str, Any]:
    """The fixture's `image-only` and `audio-only` cases (upstream's SentenceTransformer): their ids
    equal the ids built here, and the reference vectors here are within the composition bound of
    upstream's."""
    from safetensors import safe_open

    document = json.loads(multimodal_reference.FIXTURE_JSON.read_text(encoding="utf-8"))
    cases = {case["name"]: case for case in (*document["cases"], *document["audioCases"])}
    checks: dict[str, Any] = {}
    with safe_open(str(multimodal_reference.FIXTURE_TENSORS), framework="pt") as handle:
        for case_name, modality in (("image-only", IMAGE), ("audio-only", AUDIO)):
            ((_kind, source),) = [(p["type"], p[p["type"]]) for p in cases[case_name]["parts"]]
            names = [item.name for item in items[modality]]
            index = names.index(f"{SYNTHETIC}:{source}")
            item = items[modality][index]
            ids = item_ids(models.bos, models.eos, models.markers[modality], item.soft_tokens)
            upstream_ids = handle.get_tensor(f"{case_name}.input_ids").tolist()
            vector = runs[modality].vectors[index].double()
            upstream = handle.get_tensor(f"{case_name}.embedding").double()
            checks[case_name] = {
                "item": item.name,
                "idsEqual": ids == upstream_ids,
                "maxAbs": float((vector - upstream).abs().max()),
                "cosine": float(torch.dot(vector, upstream) / (vector.norm() * upstream.norm())),
                "tolerance": multimodal_reference.COMPOSITION_TOLERANCE,
            }
    return checks


# ---- The run -------------------------------------------------------------------------------


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _series_bytes(series: str) -> int | None:
    directory = SERIES_ROOT / series
    if not directory.is_dir():
        return None
    return sum(path.stat().st_size for path in container_parts(directory / export.MODEL_FILE))


def expected_rounded(config: TowerConfig, towers: Mapping[str, TowerWeights]) -> set[str]:
    return {key for name in config.towers for key in towers[name].keys}


def run(directory: Path, quality: QualitySet) -> dict[str, Any]:
    models = load_models(directory)
    _log("[trace] the tower graphs and their bf16 storage")
    vision_graph, vision_tensors = export_vision.trace(models.vision)
    audio_graph, audio_tensors = export_audio.trace(models.audio)
    selector = export_audio.assert_ir_form(audio_graph, models.audio_config)["window_selector"]
    towers: dict[str, TowerWeights] = {
        TEXT_TOWER: text_weights(models.text),
        VISION_TOWER: tower_weights(
            VISION_TOWER,
            models.vision,
            vision_graph,
            vision_tensors,
            {name: "f32" for name in export_vision.ROPE_TABLES},
            export_vision.assert_storage_form,
            export_vision.POSITION_TABLES,
            (),
        ),
        AUDIO_TOWER: tower_weights(
            AUDIO_TOWER,
            models.audio,
            audio_graph,
            audio_tensors,
            None,
            export_audio.assert_storage_form,
            (),
            # MUST: the audio i8 series' own exceptions (the one declaration — module doc).
            tuple(export_audio.i8_storage_overrides(selector)),
        ),
    }
    del vision_graph, vision_tensors, audio_graph, audio_tensors
    pristine = {
        key: tensor.detach().clone()
        for weights in towers.values()
        for key, tensor in weights.live.items()
    }

    items = prepare_items(models, quality)
    layouts: dict[str, ModalityItems] = {}
    text_inputs: dict[str, list[torch.Tensor]] = {}
    for modality, entries in items.items():
        texts, indices = distinct_texts([item.text for item in entries if item.text is not None])
        iterator = iter(indices)
        layouts[modality] = ModalityItems(
            names=tuple(item.name for item in entries),
            kinds=tuple(item.kind for item in entries),
            item_text=tuple(next(iterator) if item.kind == QUALITY else None for item in entries),
            texts=texts,
        )
        text_inputs[modality] = [text_ids(models, text) for text in texts]
        _log(f"[items] {modality}: {len(entries)} items, {len(texts)} distinct texts")

    started = time.perf_counter()
    reference = {m: run_modality(models, items[m], text_inputs[m], REFERENCE) for m in MODALITIES}
    configs: dict[str, Any] = {
        REFERENCE: {
            "towers": [],
            "elapsed": round(time.perf_counter() - started, 1),
            "modalities": {
                m: {
                    "itemSeconds": reference[m].seconds,
                    "softTokens": soft_token_summary(
                        reference[m].soft, reference[m].soft, layouts[m].names
                    ),
                    "widths": {
                        str(w): measure_width(
                            layouts[m],
                            reference[m].vectors,
                            reference[m].vectors,
                            reference[m].texts,
                            reference[m].texts,
                            w,
                        )
                        for w in mq.WIDTHS
                    },
                }
                for m in MODALITIES
            },
        }
    }
    checks = fixture_checks(models, items, reference)
    gates: list[str] = []
    for case, check in checks.items():
        if not check["idsEqual"] or check["maxAbs"] > check["tolerance"]:
            gates.append(f"{case}: the reference differs from the fixture's upstream run ({check})")

    for config in CONFIGS:
        started = time.perf_counter()
        restore_all(towers, pristine)
        rounded: list[str] = []
        for name in TOWERS:
            if name in config.towers:
                rounded += round_tower(models.wrappers[name], towers[name])
        mq.assert_rounded(rounded, expected_rounded(config, towers), config.name)
        entry: dict[str, Any] = {
            "towers": sorted(config.towers),
            "rounded": len(rounded),
            "modalities": {},
        }
        for modality in config.modalities:
            result = run_modality(models, items[modality], text_inputs[modality], config.name)
            base = reference[modality]
            tower_rounded = MODALITY_TOWER[modality] in config.towers
            text_rounded = TEXT_TOWER in config.towers
            same_soft = all(torch.equal(a, b) for a, b in zip(result.soft, base.soft, strict=True))
            if same_soft == tower_rounded:
                gates.append(
                    f"{config.name}/{modality}: soft tokens"
                    f" {'equal' if same_soft else 'differ from'} the reference with the tower"
                    f" {'rounded' if tower_rounded else 'untouched'}"
                )
            if torch.equal(result.texts, base.texts) == text_rounded:
                gates.append(
                    f"{config.name}/{modality}: the texts do not follow the text tower's state"
                )
            if torch.equal(result.vectors, base.vectors):
                gates.append(f"{config.name}/{modality}: bit-identical to the reference")
            entry["modalities"][modality] = {
                "itemSeconds": result.seconds,
                "softTokens": soft_token_summary(result.soft, base.soft, layouts[modality].names),
                "widths": {
                    str(w): measure_width(
                        layouts[modality], result.vectors, base.vectors, result.texts, base.texts, w
                    )
                    for w in mq.WIDTHS
                },
            }
        entry["elapsed"] = round(time.perf_counter() - started, 1)
        configs[config.name] = entry
        _log(f"[{config.name}] {entry['elapsed']}s, {len(rounded)} weights rounded")

    restore_all(towers, pristine)
    for modality in MODALITIES:
        first = items[modality][0]
        again = embed_item(models, first, soft_tokens_of(models, first))
        if not torch.equal(again, reference[modality].vectors[0]):
            gates.append(
                f"{modality}: the restored weights do not give the reference back bit for bit"
            )
    if not torch.equal(embed_texts(models, text_inputs[IMAGE][:1]), reference[IMAGE].texts[:1]):
        gates.append("text: the restored weights do not give the reference back bit for bit")

    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "tools/export-recipes/embeddinggemma2/measure_quant_towers.py",
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "snapshot": str(directory),
        "revision": sources.PIN.revision,
        "qualitySet": {
            "dir": str(quality.root),
            "format": MANIFEST_FORMAT,
            "images": len(quality.images),
            "audio": len(quality.audio),
        },
        "textPrompt": {"name": TEXT_PROMPT, "prefix": models.prompts[TEXT_PROMPT]},
        "widths": list(mq.WIDTHS),
        "resampler": {
            "kind": "Kaiser-windowed sinc (written in this script)",
            "zeros": RESAMPLE_ZEROS,
            "beta": KAISER_BETA,
            "rolloff": RESAMPLE_ROLLOFF,
        },
        "items": {
            m: [
                {"name": item.name, "kind": item.kind, "text": item.text, **item.info}
                for item in items[m]
            ]
            for m in MODALITIES
        },
        "texts": {m: list(layouts[m].texts) for m in MODALITIES},
        "weights": {
            name: {
                "roundedSet": vars(weights.rounded_stats),
                "moduleWeights": len(weights.module_weights),
                "tables": weights.tables,
                "keptBf16": {"tensors": weights.kept_bf16, **vars(weights.kept_stats)},
                "remainder": {
                    "tensors": weights.remainder_tensors,
                    "elements": weights.remainder_elements,
                },
            }
            for name, weights in towers.items()
        },
        "payload": {
            name: {
                **tower_payload(
                    towers[name].rounded_stats,
                    towers[name].kept_stats,
                    towers[name].remainder_elements,
                ),
                "measuredBf16SeriesBytes": _series_bytes(series),
                "measuredBf16Series": series,
            }
            for name, series in (
                (VISION_TOWER, f"{export_vision.VISION_SERIES}-bf16"),
                (AUDIO_TOWER, f"{export_audio.AUDIO_SERIES}-bf16"),
            )
        },
        "fixtureChecks": checks,
        "configs": configs,
        "gates": gates,
    }


# ---- Tables --------------------------------------------------------------------------------


def quality_table(report: Mapping[str, Any], modality: str, width: int) -> str:
    header = [
        "config",
        "cos min",
        "cos mean",
        "worst item",
        "soft min",
        "soft mean",
        "item→text top-1 (ref)",
        "agree",
        "inversion",
        "text→item top-1 (ref)",
        "agree",
        "inversion",
        "max Δ cos matrix",
    ]
    rows: list[list[str]] = []
    for name, entry in report["configs"].items():
        if modality not in entry["modalities"]:
            continue
        result = entry["modalities"][modality]
        metrics = result["widths"][str(width)]
        forward, backward = metrics["itemToText"], metrics["textToItem"]
        rows.append(
            [
                name,
                f"{metrics['cosine']['all']['min']:.6f}",
                f"{metrics['cosine']['all']['mean']:.6f}",
                metrics["cosine"]["worst"],
                f"{result['softTokens']['min']:.6f}",
                f"{result['softTokens']['mean']:.6f}",
                f"{forward['top1']}/{forward['queries']} ({forward['referenceTop1']})",
                f"{forward['top1Agreement']}/{forward['queries']}",
                f"{forward['rankInversionRate'] * 100:.2f}%",
                f"{backward['top1']}/{backward['queries']} ({backward['referenceTop1']})",
                f"{backward['top1Agreement']}/{backward['queries']}",
                f"{backward['rankInversionRate'] * 100:.2f}%",
                f"{metrics['itemTextCosineMaxAbsChange']:.4e}",
            ]
        )
    return mq._markdown(header, rows)


def size_table(report: Mapping[str, Any]) -> str:
    header = ["tower", "form", "bytes", "MiB", "arithmetic"]
    rows: list[list[str]] = []
    for tower, payload in report["payload"].items():
        for form in ("bf16", "i8"):
            size = payload[form]
            rows.append(
                [tower, form, f"{size['bytes']:,}", f"{size['mib']:.1f}", size["arithmetic"]]
            )
        measured = payload["measuredBf16SeriesBytes"]
        rows.append(
            [tower, "bf16 series (measured)", "-" if measured is None else f"{measured:,}", "", ""]
        )
    return mq._markdown(header, rows)


def time_table(report: Mapping[str, Any]) -> str:
    header = ["config", "modality", "items", "tower s (mean)", "text s (mean)", "slowest item"]
    rows: list[list[str]] = []
    for name, entry in report["configs"].items():
        for modality, result in entry["modalities"].items():
            seconds = result["itemSeconds"]
            slowest = max(seconds, key=lambda item: seconds[item]["total"])
            rows.append(
                [
                    name,
                    modality,
                    str(len(seconds)),
                    f"{sum(s['tower'] for s in seconds.values()) / len(seconds):.2f}",
                    f"{sum(s['text'] for s in seconds.values()) / len(seconds):.2f}",
                    f"{slowest} {seconds[slowest]['total']:.2f}",
                ]
            )
    return mq._markdown(header, rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--snapshot-dir", type=Path, default=sources.SNAPSHOT_DIR)
    parser.add_argument("--quality-set", type=Path, default=DEFAULT_QUALITY_SET)
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--write-clips",
        type=Path,
        default=None,
        help="also write the quality set with its clips as measured (16 kHz, 16-bit mono WAV) and"
        " its images to this directory, under a manifest of the same format",
    )
    args = parser.parse_args(argv)
    out = args.out if args.out is not None else BENCH_DIR / f"{args.date.isoformat()}{OUT_SUFFIX}"
    quality = load_quality_set(args.quality_set)
    written = None if args.write_clips is None else write_clips(quality, args.write_clips)
    if written is not None:
        _log(f"[clips] {written['audio']} clips, {written['images']} images → {written['dir']}")
    report = {**run(args.snapshot_dir, quality), "writtenClips": written}
    out.mkdir(parents=True, exist_ok=True)
    (out / REPORT_FILE).write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for modality in MODALITIES:
        for width in mq.WIDTHS:
            print(f"{modality}, width {width} (against the reference):")
            print(quality_table(report, modality, width))
            print()
    print("Projected container payload of the towers:")
    print(size_table(report))
    print()
    print("Seconds per item (CPU, f32 fake-quant):")
    print(time_table(report))
    print()
    print(f"report → {out / REPORT_FILE}")
    if report["gates"]:
        raise AssertionError("verification gates are red: " + " / ".join(report["gates"]))
    print("gates: all green")


if __name__ == "__main__":
    main()
