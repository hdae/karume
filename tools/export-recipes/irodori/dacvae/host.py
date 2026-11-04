r"""Reference numbers of the **host-side preprocessing** around DACVAE (normalization of the
reference audio, reflect pad, tail trim).

Where `irodori/dacvae/export.py` produces the graphs (G6 decoder / G7 encoder), this produces the
reference values of the stages that stay **outside** the graphs. They are what W4's host
implementation (the TS codec integration) is compared against, and the values are taken by
**calling** the upstream implementation (`irodori_tts.codec` / `irodori_tts.inference_runtime`).

    cd tools/export-recipes
    uv run --with descript-audiotools --with einops --with 'transformers==5.14.1' \
        python -m irodori.dacvae.host

It imports `irodori_tts`, so the transformers pin is needed (the same reason as
`irodori/export.py`). `descript-audiotools` is the real path of the LUFS measurement (the
K-weighting IIR of ITU-R BS.1770-4); `einops` is needed only because `dacvae/__init__.py` pulls it
in.

Output (default `outputs/series/dacvae-32dim/host/`, under `.gitignore`):

    meta.json                measured scalars per case (LUFS, gains, lengths) and the values the
                             standing checks measured
    case.<name>.safetensors  raw waveform / normalized / after reflect pad / latent
    trim.safetensors         input latents of the tail trim (2 real z + 1 synthetic with a silent
                             tail)

## The stages the host owns (the only thing this file copies)

1. **Reading the WAV** (int16 PCM → f32). The convention is **/32768** (`_wav_scale_evidence`
   measures it at both ends of int16 every run — this one value decides the division in the TS
   decoder)
2. **LUFS normalization** (when `normalize_db = -16`) → `ensure_max_of_audio`.
   The gain is `exp((target − ref_db) · GAIN_FACTOR)`; the peak gain that follows is `1/peak` only
   when `peak > 1`
3. **Peak safety scale** (when `normalize_db = None` and `ensure_max = True`).
   Only here does upstream take a separate path that multiplies by `1.0 / float(peak)`
   (`encode_waveform` in `codec.py`)
4. **Reflect pad** (`DACVAE._pad` — right-padded to a multiple of hop 1920)
5. **Tail trim** (`find_flattening_point` — the position where the tail of a generated latent
   becomes flat and close to 0)

MUST: 2 to 5 **call upstream instead of copying the formulas** (`_normalize_loudness` / `_pad` /
`find_flattening_point`). What is copied is only 1 and the representation that **decomposes** the
normalization into "LUFS gain × peak gain"; for the latter, `_decomposition_evidence` measures
every run that "the recorded scalars reproduce the normalized waveform bit for bit" (the TS side
can only use the decomposed form — to avoid porting the audiotools IIR, it has to receive ref_db
from the golden).

## What is checked

Everything is measured before emit, and if any one check fails **nothing is written** (ADR 0005's
fail loudly):

- the **WAV normalization convention** is exactly /32768 (the int16 low end −32768 maps to exactly
  −1.0)
- **the decomposition reproduces** (above), and is **bit-identical to upstream
  `_normalize_loudness`**
- **the latent is bit-identical to upstream `encode_waveform`** (the whole path preprocessing →
  graph)
- each case **actually takes the branch it targets** (guarding against tautology):
  the case under 0.5 s really is under 0.5 s / the peak gain is below 1 in the peak > 1 case /
  the pad is 0 samples in the case that is an exact multiple of hop, and positive in the
  remainder case
- **the tail trim** does not cut a real latent (point == length), and returns exactly that position
  for a synthetic latent whose tail was zeroed
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
from safetensors.torch import load_file, save_file

from _shared.paths import SERIES_ROOT
from karume.convert import normalize_boundary_tensor

from .. import export as ir
from . import export as ex

META_FILE = "meta.json"
CASE_PREFIX = "case."
CASE_SUFFIX = ".safetensors"
TRIM_FILE = "trim.safetensors"

#: Default of `ensure_max` (`inference_runtime.SamplingRequest.ref_ensure_max`). **It only takes
#: effect when `normalize_db = None`**; on the LUFS normalization side, audiotools'
#: `ensure_max_of_audio()` does the same thing internally (as the comment in `codec.py` says).
ENSURE_MAX = True

#: Golden cases of the host preprocessing.
#: `(name, length (seconds; None = full length), extra samples, amplitude scale, target LUFS
#: (None = no normalization))`.
#:
#: The reference audio is 48 kHz mono, 7.6 s, and its length is an exact multiple of hop 1920
#: (190 frames). The default case alone would therefore make reflect pad an identity with 0
#: samples, so a separate case with a remainder (+777 samples) is added. The fourth is under
#: 0.5 s and takes the branch where audiotools **zero-pads for the measurement**
#: (`LoudnessMixin.loudness`). The third scales the amplitude by 3 to produce peak > 1.
HOST_CASES: tuple[tuple[str, float | None, int, float, float | None], ...] = (
    ("ref-default", None, 0, 1.0, ex.REFERENCE_NORMALIZE_DB),
    ("ref-plain", 3.0, 0, 1.0, None),
    ("ref-hot", 3.0, 0, 3.0, None),
    ("ref-short", 0.4, 0, 1.0, ex.REFERENCE_NORMALIZE_DB),
    ("ref-odd", 3.0, 777, 1.0, ex.REFERENCE_NORMALIZE_DB),
)

#: Golden cases of the tail trim `(name, source, number of tail frames set to 0)`.
#:
#: The 2 real latents measure the **no-cut** side (`find_flattening_point` returns the length
#: itself), and the 1 synthetic zeroes the last 40 frames and checks that "it returns exactly 121".
#: With only the former, "an implementation that always returns the length" would pass too
#: (guarding against tautology).
TRIM_CASES: tuple[tuple[str, str, int], ...] = (
    ("z-full", "full", 0),
    ("z-no-ref", "no-ref", 0),
    ("z-silent-tail", "full", 40),
)

#: Values that measure the int16 PCM normalization convention (both ends and near 0). If −32768
#: maps to exactly −1.0, it is /32768.
WAV_SCALE_PROBE = (-32768, -32767, -1, 0, 1, 16384, 32767)
INT16_DIVISOR = 32768


class IrodoriHostSource(NamedTuple):
    """The two host-side pieces borrowed from `irodori_tts` (imported by adding to `sys.path`).

    MUST: import **from the implementation clone** (the script keeps no copy).
    `irodori_tts/__init__.py` pulls in transformers, so an environment without the pinned `--with`
    fails at import time.
    """

    codec_cls: Any
    find_flattening_point: Any

    @classmethod
    def load(cls, source_dir: Path) -> IrodoriHostSource:
        codec_py = source_dir / "irodori_tts" / "codec.py"
        if not codec_py.is_file():
            raise SystemExit(
                f"Irodori implementation not found: {codec_py}"
                " (pass the checkout of `git clone https://github.com/Aratako/Irodori-TTS`"
                " as --irodori-dir)"
            )
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        from irodori_tts.codec import DACVAECodec
        from irodori_tts.inference_runtime import find_flattening_point

        return cls(codec_cls=DACVAECodec, find_flattening_point=find_flattening_point)


def _wav_scale_evidence(sample_rate: int) -> dict[str, Any]:
    """Measure the WAV reader's int16 → f32 convention on a synthetic file (a check).

    MUST: measure rather than assert — a /32767 implementation and a /32768 implementation differ by
    only 3e-5 relative on real audio, which is buried in the tolerance when comparing goldens.
    """
    import numpy as np
    import soundfile as sf

    values = np.array(WAV_SCALE_PROBE, dtype=np.int16)
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / "int16-probe.wav"
        sf.write(str(path), values, sample_rate, subtype="PCM_16")
        decoded, decoded_rate = ex.read_wav(path)
    if decoded_rate != sample_rate:
        raise AssertionError(
            f"the probe's sr is {decoded_rate} (differs from the written {sample_rate})"
        )
    expected = torch.tensor([value / INT16_DIVISOR for value in WAV_SCALE_PROBE])
    if not torch.equal(decoded, expected.to(decoded.dtype)):
        raise AssertionError(
            f"the WAV int16 normalization is not /{INT16_DIVISOR}"
            f" ({WAV_SCALE_PROBE} → {[float(v) for v in decoded]})"
        )
    return {
        "loader": "soundfile",
        "int16Divisor": INT16_DIVISOR,
        "probeInt16": list(WAV_SCALE_PROBE),
        "probeFloat": [float(value) for value in decoded],
    }


def _slice_reference(
    wav: torch.Tensor, sample_rate: int, seconds: float | None, extra: int, scale: float
) -> torch.Tensor:
    """Cut one case's raw waveform out of the reference audio (only length and amplitude
    change)."""
    length = wav.numel() if seconds is None else int(seconds * sample_rate) + extra
    if not 0 < length <= wav.numel():
        raise SystemExit(
            f"slice length {length} does not fit in the reference audio's {wav.numel()} samples"
        )
    return (wav[:length] * scale).contiguous()


def _normalize_decomposed(
    raw: torch.Tensor, sample_rate: int, target_db: float | None
) -> tuple[torch.Tensor, dict[str, float]]:
    """Decompose the normalization into "LUFS gain × peak gain" and return it
    (`(normalized, scalars)`).

    When `target_db` is `None`, this is the path of upstream `encode_waveform` with only the peak
    safety scale (`1.0 / float(peak)`). Otherwise it is audiotools' `normalize` →
    `ensure_max_of_audio`, with gain `exp((target − ref_db) · GAIN_FACTOR)`. **That the
    decomposition is correct is itself measured by `_decomposition_evidence`** (this is the formula
    for the record).
    """
    from audiotools import AudioSignal

    signal = AudioSignal(raw.unsqueeze(0).unsqueeze(0), int(sample_rate))
    reference_db = float(signal.loudness())
    if target_db is None:
        loudness_gain = 1.0
    else:
        delta = torch.tensor(target_db - reference_db)
        loudness_gain = float(torch.exp(delta * AudioSignal.GAIN_FACTOR))
    scaled = raw * loudness_gain
    peak = float(scaled.abs().max())
    peak_gain = (1.0 / peak) if peak > 1.0 else 1.0
    return (scaled * peak_gain).contiguous(), {
        "refDb": reference_db,
        "loudnessGain": loudness_gain,
        "peakBeforeScale": peak,
        "peakGain": peak_gain,
    }


def _decomposition_evidence(
    raw: torch.Tensor,
    decomposed: torch.Tensor,
    scalars: Mapping[str, float],
    where: str,
) -> float:
    """Measure that the recorded scalars alone reproduce the normalized waveform (a check).

    The TS side has no audiotools IIR (it receives `refDb` from the golden), so **these two
    multiplications being the whole normalization** is the very premise of the host port.
    """
    rebuilt = raw * scalars["loudnessGain"] * scalars["peakGain"]
    if not torch.equal(rebuilt, decomposed):
        raise AssertionError(
            f"{where}: the two gains do not reproduce the normalized waveform"
            f" (max abs diff {float((rebuilt - decomposed).abs().max())})"
        )
    return 0.0


def _upstream_normalize_evidence(
    codec_cls: Any, raw: torch.Tensor, sample_rate: int, target_db: float, normalized: torch.Tensor
) -> float:
    """Bit identity with upstream `DACVAECodec._normalize_loudness` (a check)."""
    upstream = codec_cls._normalize_loudness(raw, sample_rate=sample_rate, target_db=target_db)
    if not torch.equal(upstream, normalized):
        raise AssertionError(
            "the LUFS normalization does not match upstream"
            f" (max abs diff {float((upstream - normalized).abs().max())})"
        )
    return 0.0


def _branch_evidence(cases: Mapping[str, Mapping[str, Any]], hop: int) -> dict[str, Any]:
    """Measure that each case actually takes the branch it targets (guarding against tautology).

    MUST: adding cases to the table must not let "every one just took the default path" go green.
    Three things are checked: ① the case under 0.5 s really is short ② the peak gain is below 1 in
    the peak > 1 case and exactly 1 in the peak ≤ 1 case ③ the pad is 0 in the case that is a
    multiple of hop and positive in the remainder case.
    """
    short = cases["ref-short"]
    if short["seconds"] >= 0.5:
        raise AssertionError(
            f"ref-short is {short['seconds']} s — it does not take audiotools' zero-pad"
            " measurement branch"
        )
    if short["refDb"] == cases["ref-default"]["refDb"]:
        raise AssertionError(
            "ref-short and ref-default have the same LUFS — the measurement does not depend on"
            " the length"
        )
    if not cases["ref-hot"]["peakGain"] < 1.0:
        raise AssertionError(
            f"the peak gain of ref-hot is {cases['ref-hot']['peakGain']} — it does not take"
            " the peak > 1 branch"
        )
    if cases["ref-plain"]["peakGain"] != 1.0:
        raise AssertionError(
            f"the peak gain of ref-plain is {cases['ref-plain']['peakGain']} — it shrinks"
            " at peak ≤ 1"
        )
    if cases["ref-default"]["padSamples"] != 0:
        raise AssertionError(
            f"the pad of ref-default is {cases['ref-default']['padSamples']} samples"
            f" — the reference audio's length is no longer a multiple of hop {hop}"
        )
    if not 0 < cases["ref-odd"]["padSamples"] < hop:
        raise AssertionError(
            f"the pad of ref-odd is {cases['ref-odd']['padSamples']} samples — it does not"
            " take the remainder branch"
        )
    return {
        "shortSeconds": short["seconds"],
        "hotPeakGain": cases["ref-hot"]["peakGain"],
        "oddPadSamples": cases["ref-odd"]["padSamples"],
    }


def _trim_evidence(
    find_flattening_point: Any, latents: Mapping[str, torch.Tensor]
) -> dict[str, dict[str, int]]:
    """Take the tail trim's `{z → position}` with the upstream function and confirm it is not a
    tautology (a check).

    A real latent is **not cut** (position = length), and a synthetic one with a zeroed tail returns
    **exactly that position**. With only one side, an implementation that "always returns the
    length" or "always returns 0" would pass.
    """
    points: dict[str, dict[str, int]] = {}
    for name, source, silence in TRIM_CASES:
        latent = latents[name]
        frames = int(latent.shape[1])
        point = int(find_flattening_point(latent[0]))
        expected = frames - silence
        if point != expected:
            raise AssertionError(
                f"{name}: find_flattening_point returned {point} (expected {expected})"
                " — the premise of the tail trim (do not cut a real latent / cut silence) is"
                " broken"
            )
        points[name] = {"source": source, "frames": frames, "silentTail": silence, "point": point}
    return points


def _trim_latents(latent_dir: Path) -> dict[str, torch.Tensor]:
    """Build the input latents of {@link TRIM_CASES} (including the synthetic one made by zeroing
    the tail of a real z)."""
    sources: dict[str, torch.Tensor] = {}
    for name in {source for _n, source, _s in TRIM_CASES}:
        path = latent_dir / f"{ex.LATENT_CASE_PREFIX}{name}{CASE_SUFFIX}"
        if not path.is_file():
            raise SystemExit(
                f"real latent missing: {path}"
                " (create it with `uv run --with 'transformers==5.14.1' python -m"
                " irodori.pipeline_ref`)"
            )
        sources[name] = load_file(str(path))[ex.LATENT_KEY].to(torch.float32)
    latents: dict[str, torch.Tensor] = {}
    for name, source, silence in TRIM_CASES:
        latent = sources[source].clone()
        if silence:
            if silence >= int(latent.shape[1]):
                raise SystemExit(f"{name}: {silence} silent frames are not shorter than the latent")
            latent[:, -silence:] = 0.0
        latents[name] = latent.contiguous()
    return latents


def emit(
    model_dir: Path,
    source_dir: Path,
    out_dir: Path,
    *,
    latent_dir: Path = ex.DEFAULT_LATENT_DIR,
    reference_wav: Path = ex.DEFAULT_REFERENCE_WAV,
    irodori_dir: Path = ir.DEFAULT_SOURCE_DIR,
) -> dict[str, Any]:
    """Write the golden of the host preprocessing and return a summary (nothing is written if a
    check fails)."""
    upstream = IrodoriHostSource.load(irodori_dir)
    DACVAECodec = upstream.codec_cls  # noqa: N806 — keeps upstream's class name

    source = ex.DacvaeSource(source_dir)
    model = ex.load_codec(source, model_dir)
    ex.bypass_watermark(model.decoder)
    sample_rate = int(model.sample_rate)
    hop = ex.hop_length(model)
    wav, wav_rate = ex.read_wav(reference_wav)
    if wav_rate != sample_rate:
        raise SystemExit(
            f"{reference_wav.name}: {wav_rate}Hz differs from the codec's {sample_rate}Hz"
            " (this script does not resample)"
        )
    wav_scale = _wav_scale_evidence(sample_rate)

    # Upstream's canonical path (`encode_waveform`) runs **before weight_norm is folded** —
    # taking it afterwards would compare "folded against folded" and turn the latent comparison
    # into a tautology (the same discipline as `irodori.dacvae.export`).
    codec = DACVAECodec(
        model=model,
        sample_rate=sample_rate,
        latent_dim=int(model.quantizer.in_proj.out_channels) // 2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        deterministic_encode=True,
        deterministic_decode=True,
        normalize_db=ex.REFERENCE_NORMALIZE_DB,
    )
    raws = {
        name: _slice_reference(wav, sample_rate, seconds, extra, scale)
        for name, seconds, extra, scale, _db in HOST_CASES
    }
    upstream_latents = {
        name: codec.encode_waveform(
            raws[name].reshape(1, 1, -1), sample_rate, normalize_db=target_db, ensure_max=ENSURE_MAX
        ).clone()
        for name, _sec, _extra, _scale, target_db in HOST_CASES
    }

    cases: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, torch.Tensor]] = {}
    normalize_evidence: dict[str, float] = {}
    decomposition_evidence: dict[str, float] = {}
    for name, _seconds, _extra, scale, target_db in HOST_CASES:
        raw = raws[name]
        normalized, scalars = _normalize_decomposed(raw, sample_rate, target_db)
        decomposition_evidence[name] = _decomposition_evidence(raw, normalized, scalars, name)
        if target_db is not None:
            normalize_evidence[name] = _upstream_normalize_evidence(
                DACVAECodec, raw, sample_rate, target_db, normalized
            )
        with torch.no_grad():
            padded = model._pad(normalized.reshape(1, 1, -1))
        frames = int(padded.shape[-1]) // hop
        payloads[name] = {
            "raw": raw,
            "normalized": normalized,
            # MUST: clone — in the case with 0 pad samples (`ref-default`) `_pad` returns its input
            # as is, and passing tensors that share storage to safetensors makes the save fail.
            "padded": padded.reshape(-1).clone(),
            "latent": upstream_latents[name],
        }
        cases[name] = {
            "samples": int(raw.numel()),
            "seconds": float(raw.numel()) / sample_rate,
            "amplitudeScale": scale,
            "normalizeDb": target_db,
            "ensureMax": ENSURE_MAX,
            **scalars,
            "paddedSamples": int(padded.shape[-1]),
            "padSamples": int(padded.shape[-1]) - int(raw.numel()),
            "frames": frames,
        }
    branch = _branch_evidence(cases, hop)

    # ---- Apply the preprocessing and measure that the host chain (normalize → pad → graph)
    # is bit-identical to upstream ----
    ex.fold_weight_norm(model)
    ex.lift_snake_alphas(source, model)
    encoder_graph = ex.EncoderGraph(model.encoder, ex.truncated_in_proj(model.quantizer.in_proj))
    chain_evidence: dict[str, float] = {}
    for name in cases:
        framed = payloads[name]["padded"].reshape(1, cases[name]["frames"], hop)
        with torch.no_grad():
            chain = encoder_graph(framed)
        if not torch.equal(chain, upstream_latents[name]):
            raise AssertionError(
                f"{name}: the host chain's latent does not match upstream encode_waveform"
                f" (max abs diff {float((chain - upstream_latents[name]).abs().max())})"
            )
        chain_evidence[name] = 0.0

    trim_latents = _trim_latents(latent_dir)
    trim = _trim_evidence(upstream.find_flattening_point, trim_latents)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for name, tensors in payloads.items():
        path = out_dir / f"{CASE_PREFIX}{name}{CASE_SUFFIX}"
        save_file(
            {
                key: normalize_boundary_tensor(value.contiguous(), f"{name}: '{key}'")
                for key, value in tensors.items()
            },
            str(path),
        )
        written[path.name] = path.stat().st_size
    save_file(
        {name: value.contiguous() for name, value in trim_latents.items()},
        str(out_dir / TRIM_FILE),
    )
    written[TRIM_FILE] = (out_dir / TRIM_FILE).stat().st_size

    meta_payload = {
        "sourceRepo": ex.SOURCE_REPO,
        "sourceCommit": ex.SOURCE_COMMIT,
        "referenceWav": reference_wav.name,
        "sampleRate": sample_rate,
        "hopLength": hop,
        "frameRate": ex.frame_rate(model),
        "gainFactor": _gain_factor(),
        "wavScale": wav_scale,
        "cases": cases,
        "trim": trim,
        "branchEvidence": branch,
        "upstreamNormalizeMaxAbs": normalize_evidence,
        "decompositionMaxAbs": decomposition_evidence,
        "upstreamLatentMaxAbs": chain_evidence,
    }
    (out_dir / META_FILE).write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    written[META_FILE] = (out_dir / META_FILE).stat().st_size
    return {"dir": str(out_dir), "bytes": written, **meta_payload}


def _gain_factor() -> float:
    """audiotools' gain conversion factor (`ln(10)/20`). **Read from the library, not copied as
    a constant**."""
    from audiotools import AudioSignal

    return float(AudioSignal.GAIN_FACTOR)


def default_out_dir(model_dir: Path) -> Path:
    """The default location (`outputs/series/<directory name of the real weights>/host/`)."""
    return SERIES_ROOT / model_dir.name / "host"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=ex.DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=ex.DEFAULT_SOURCE_DIR)
    parser.add_argument("--irodori-dir", type=Path, default=ir.DEFAULT_SOURCE_DIR)
    parser.add_argument("--latent-dir", type=Path, default=ex.DEFAULT_LATENT_DIR)
    parser.add_argument("--reference-wav", type=Path, default=ex.DEFAULT_REFERENCE_WAV)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out_dir = default_out_dir(args.model_dir) if args.out is None else args.out
    summary = emit(
        args.model_dir,
        args.source_dir,
        out_dir,
        latent_dir=args.latent_dir,
        reference_wav=args.reference_wav,
        irodori_dir=args.irodori_dir,
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
