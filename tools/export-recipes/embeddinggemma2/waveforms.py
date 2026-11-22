"""The synthetic waveforms of the EmbeddingGemma 2 audio path: the audio tower's goldens
({@link embeddinggemma2.export_audio}) and the whole-input reference vectors
({@link embeddinggemma2.multimodal_reference}) are taken on them.

Every waveform is 16 kHz mono 16-bit PCM computed from a formula (a seeded integer generator for
the noise), handed to upstream as f32 `pcm / 32768` — what a WAV reader gives. The 16-bit step is
what makes the bytes the same on every machine: a last-bit difference of a platform's `sin` cannot
move a value across a PCM step in practice, so a fixture can carry the samples as they are. They are
neutral on purpose — tones, sweeps, noise and a click; no speech, music recording or voice. Do not
add others.

Each covers one thing upstream's front end and the tower's frame layout do. A waveform of `s`
samples has `L = max(0, ceil(s / 160) − 1)` valid mel frames (a frame is valid only when its whole
321-sample window, after upstream's 160 zeros of left padding, is real audio) and
`ceil(L / 4)` soft tokens (two stride-2 convolutions); the residue of `L` mod 4 decides whether the
tower needs its frame mask (`export_audio`, module doc):

| Waveform | Samples | Frames L | L mod 4 | Soft tokens | What it covers                         |
| -------- | ------- | -------- | ------- | ----------- | -------------------------------------- |
| `sweep`  | 16,000  | 99       | 3       | 25          | 1 s, a log sine sweep 100 Hz → 7 kHz   |
| `noise`  | 40,000  | 249      | 1       | 63          | 2.5 s of uniform noise                 |
| `click`  | 16,100  | 100      | 0       | 25          | silence with one click                 |
| `edge`   | 16,400  | 102      | 2       | 26          | two tones cut at full level (the mask) |
| `tones`  | 160,000 | 999      | 3       | 250         | 10 s, a sequence of tones              |
| `long`   | 480,000 | 2,999    | 3       | 750         | 30 s, the front end's maximum          |

The table's frame and token counts are pinned by the tests against upstream's own feature
extractor.

MUST: torch-free and transformers-free (numpy only) — the pure tests import it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

#: The sampling rate upstream's feature extractor takes (it does not resample an array).
SAMPLE_RATE = 16_000

#: The PCM full scale: a sample is `pcm / 32768` in f32 (exact).
PCM_SCALE = 32_768

#: The seed of the noise in `noise` and `long` (a fixed integer generator, so the bytes never move).
NOISE_SEED = 20261122


@dataclass(frozen=True)
class SyntheticWaveform:
    """One waveform: its name, its length in samples and the formula that draws it."""

    name: str
    samples: int
    #: What the waveform covers (the module table).
    why: str
    #: `samples → values in [-1, 1]` (f64, before the PCM rounding).
    draw: Callable[[int], np.ndarray]

    def pcm16(self) -> np.ndarray:
        """The samples as 16-bit PCM `[samples]` int16."""
        values = np.asarray(self.draw(self.samples), dtype=np.float64)
        if values.shape != (self.samples,):
            raise ValueError(f"{self.name}: the formula drew {values.shape}, not {(self.samples,)}")
        if not np.isfinite(values).all() or float(np.abs(values).max()) > 1.0:
            raise ValueError(f"{self.name}: the formula drew values outside [-1, 1]")
        return np.round(values * (PCM_SCALE - 1)).astype(np.int16)

    def f32(self) -> np.ndarray:
        """The samples as upstream reads them: `pcm / 32768`, `[samples]` f32 (exact)."""
        return (self.pcm16().astype(np.float32) / np.float32(PCM_SCALE)).astype(np.float32)


def _time(samples: int) -> np.ndarray:
    return np.arange(samples, dtype=np.float64) / SAMPLE_RATE


def _fade(samples: int, ramp: int) -> np.ndarray:
    """A raised-cosine fade in and out of `ramp` samples each."""
    envelope = np.ones(samples, dtype=np.float64)
    shape = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp, dtype=np.float64) / ramp)
    envelope[:ramp] = shape
    envelope[samples - ramp :] = shape[::-1]
    return envelope


def _log_sweep(samples: int, low: float, high: float) -> np.ndarray:
    """A logarithmic sine sweep from `low` to `high` Hz over the whole length."""
    duration = samples / SAMPLE_RATE
    rate = np.log(high / low)
    phase = 2 * np.pi * low * duration / rate * (np.exp(_time(samples) / duration * rate) - 1)
    return np.sin(phase)


def _uniform_noise(samples: int, amplitude: int) -> np.ndarray:
    """Uniform integer noise in PCM steps `[-amplitude, amplitude]`, as values in [-1, 1]."""
    generator = np.random.default_rng(NOISE_SEED)
    steps = generator.integers(-amplitude, amplitude + 1, size=samples, dtype=np.int64)
    return steps.astype(np.float64) / (PCM_SCALE - 1)


def _sweep(samples: int) -> np.ndarray:
    return 0.5 * _fade(samples, 160) * _log_sweep(samples, 100.0, 7000.0)


def _noise(samples: int) -> np.ndarray:
    return _uniform_noise(samples, 3277)


def _click(samples: int) -> np.ndarray:
    values = np.zeros(samples, dtype=np.float64)
    burst = np.array([16000, -12000, 8000, -5000, 3000, -1500, 700, -300], dtype=np.float64)
    start = SAMPLE_RATE // 2
    values[start : start + burst.size] = burst / (PCM_SCALE - 1)
    return values


def _edge(samples: int) -> np.ndarray:
    """Two tones with a fade in only: the last real frames carry the signal at full level, so the
    frame that reads past the end matters (`export_audio`, module doc)."""
    time = _time(samples)
    envelope = np.ones(samples, dtype=np.float64)
    envelope[:160] = 0.5 - 0.5 * np.cos(np.pi * np.arange(160, dtype=np.float64) / 160)
    tones = 0.3 * np.sin(2 * np.pi * 440.0 * time) + 0.3 * np.sin(2 * np.pi * 660.0 * time)
    return envelope * tones


#: The semitone steps of `tones` above 220 Hz (a pentatonic run up and down).
TONE_STEPS = (0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24, 21, 19, 16, 14, 12, 9, 7, 4, 2)


def _tones(samples: int) -> np.ndarray:
    """One tone per 0.5 s with its second harmonic, each faded in and out over 20 ms."""
    note = samples // len(TONE_STEPS)
    time = _time(note)
    envelope = _fade(note, 320)
    values = np.zeros(samples, dtype=np.float64)
    for index, step in enumerate(TONE_STEPS):
        frequency = 220.0 * 2 ** (step / 12)
        tone = 0.3 * np.sin(2 * np.pi * frequency * time) + 0.1 * np.sin(
            4 * np.pi * frequency * time
        )
        values[index * note : (index + 1) * note] = envelope * tone
    return values


def _long(samples: int) -> np.ndarray:
    """A slow sweep 60 Hz → 6 kHz under a 0.25 Hz tremolo, with quiet noise."""
    tremolo = 0.75 + 0.25 * np.sin(2 * np.pi * 0.25 * _time(samples))
    return 0.3 * tremolo * _fade(samples, 160) * _log_sweep(samples, 60.0, 6000.0) + (
        _uniform_noise(samples, 655)
    )


#: The waveforms, in the order the goldens list them.
WAVEFORMS: Mapping[str, SyntheticWaveform] = {
    waveform.name: waveform
    for waveform in (
        SyntheticWaveform("sweep", 16_000, "1 s, a logarithmic sine sweep 100 Hz to 7 kHz", _sweep),
        SyntheticWaveform("noise", 40_000, "2.5 s of uniform noise", _noise),
        SyntheticWaveform("click", 16_100, "silence with one click", _click),
        SyntheticWaveform(
            "edge",
            16_400,
            "two tones cut at full level, 102 frames (L = 2 mod 4: the frame mask's case)",
            _edge,
        ),
        SyntheticWaveform("tones", 160_000, "10 s, a sequence of tones", _tones),
        SyntheticWaveform("long", 480_000, "30 s, the front end's maximum", _long),
    )
}


def valid_frames(samples: int) -> int:
    """The valid mel frames upstream's feature extractor gives `samples` samples (module doc)."""
    hop = SAMPLE_RATE // 100
    return max(0, -(-samples // hop) - 1)


def soft_tokens(frames: int) -> int:
    """The soft tokens of `frames` valid mel frames (two stride-2 convolutions, each
    `ceil(t / 2)`)."""
    return -(-frames // 4)
