"""Script that writes the lightweight vowel-recognition CRNN (audio → vowel class sequence for
lip-sync) into the IR v2 container (`krm`) + golden io. **Length is a symbolic axis, so there is
only one graph**.

    uv run python -m vowel_detector.export
    uv run python -m vowel_detector.export --verify  # equivalence with nn.GRU (replacement A/B)

The output goes under `outputs/series/` and is kept out of commits by the root `.gitignore`.

## Why no dependency group is added (the model definition is copied verbatim)

**No** extra dependency group is needed (torch is a base dependency). The upstream model definition
`training/src/vowel_detector/crnn.py` is a 20-line `nn.Module`, but importing it pulls in
pyopenjtalk (a native OpenJTalk build) through `vowel_detector.phonemes` and librosa through
`vowel_detector.features` — both are for G2P and feature extraction, and this script needs not one
line of them. The **model definition is therefore copied here verbatim** ({@link Crnn}).

The risk of the copy drifting from upstream is covered by `load_state_dict(strict=True)`: the
weights are a state_dict of 22 tensors, so a mismatch in any of layer order, names or shapes fails
the load (a rewrite that only changes values silently cannot come from copying the **structure** of
the model definition).

## The length axis (symbol `T` = number of 20 ms frames)

The graph input is the 10 ms frame sequence `features f32 [1, 2T, 83]` and the output is the logits
of the 20 ms frames `[1, T, 8]` (conv stride 2 halves the time axis).

**The symbol is placed on the output-side grid, not the input side** (`2*Dim("T")`). With a plain
`Dim("T")` on the input, the output extent of conv1d(stride=2, pad=2, k=5) becomes the **floor
division** `((T−1)//2)+1`, which does not fit karume's dimension language (`coeff·sym+offset` —
ADR 0010). With `2*Dim("T")` the output extent is exactly the affine expression `T`.

As a consequence, **the number of 10 ms frames of the graph input is always even** (= `2T`) as a
runtime contract. An odd frame count is passed **with its last frame dropped** (the convention that
`packages/models/src/vowel-detector/pipeline.ts` and the real-weight E2E already follow — a
leftover frame would become an output with only one input frame's worth, and the post-processing's
"1 frame = 20 ms" would break at the end only).

The symbol can be passed through because `nn.GRU` is replaced with `karume::gru_scan` (see
"Replacement layer" below). `run_decompositions` **fully unrolls** the decomposition of
`aten.gru.input` along time, so `Dim("T")` fails with `Specializations unexpectedly required (T)`
(8,434 nodes at T10=200). After the replacement it is **18 nodes independent of T**, and neither
length buckets nor right zero padding are needed.

`--length` is the length of the emitted golden cases and of the **tracing example**, not the set of
lengths the graph accepts (it accepts **only multiples of 2** for the same reason as above).

## Replacement layer (`vowel_detector/patch.py`)

`Crnn.forward` does not go through `nn.GRU.forward`; it splits the 2-layer bidirectional GRU into
**4 unidirectional single-layer `gru_scan{,_reverse}` + 2 `cat`** (the input-side GEMM is a plain
`linear`). That the replacement is **bit-identical** to `nn.GRU` is pinned by measurement in
`vowel_detector/tests/test_patch.py`, and with the real weights `--verify` (several lengths) and the
standing check on every emit ({@link _write_io}) exercise it every time.

The `nn.GRU` module itself is **kept** — the copy check that `load_state_dict(strict=True)` performs
(the 22 tensors `weight_ih_l{k}` / `bias_hh_l{k}_reverse` …) would stop working if the parameters'
ownership changed too. Only the forward path is replaced.

MUST: callers **do not match lengths by right zero padding**. The reverse GRU carries state back
from the padded side, so right-zero-padding T_true=137 to T=500 measured max abs diff 5.91 / argmax
agreement 0.971 (the error concentrates at the end — 0.138 at the start / 5.915 at the end).
Padding {@link build_cases}'s `voiced` from T10=138 → 500 gives the same pattern (max abs diff
5.50 / 0.59 at the start / 5.50 at the end). A unidirectional GRU would stay at 5.1e-03 from the
conv window edges alone, but this model is bidirectional, so **that does not hold**. Now that the
length is symbolic, there is no reason left to pad (there is no grid to round up to).

## What goes into the graph

`Crnn.forward` itself (2 conv stages → 2-layer BiGRU → Linear). The output is the **logits before
log_softmax**; both the post-processing (log_softmax → Viterbi with transition penalties → merging
of short runs → cons absorption → `.lab`) and the feature extraction (80-dimensional log-mel + 3
DSP dimensions) are the host's job. The former, because keeping the output as logits keeps the
sensitivity of the golden numeric regression; the latter, because there is no reason to put it in
the graph.

The batch is **a static 1**. The only dynamic axis is time `T` (`symbol_names=("T",)`).

## The input contract (feature extraction is not in the graph)

Breakdown of the 83 dimensions of `features` (source of truth: upstream's
`training/src/vowel_detector/features.py`):

- 0..79 — log-mel (16 kHz / n_fft 512 / win 400 / hop 160 / n_mels 80) **z-normalized with the
  per-utterance mean/std**
- 80 — voicing (peak ratio of the autocorrelation, `[0, 1]`)
- 81 — log energy relative to the utterance peak, divided by 10 (0 or below)
- 82 — zero-crossing rate (`[0, 1]`)

The synthetic cases of {@link build_cases} stay within these ranges (the measured ranges are in
that function's docstring).

## Output layout

The series name is `vowel-detector-<checkpoint name>` (`vowel-detector-crnn-epoch3` for the
default `crnn_epoch3.pt`). **The length is not part of the name** — because there is one graph;
only the checkpoint generation separates series:

    outputs/series/<series name>/model.krm             weights, constants + the 2 descriptors
    outputs/series/<series name>/io.<case>.safetensors input and torch CPU expected output

The tensor key convention of io is the same as tiny golden / DeBERTa / SigLIP2
(`input.<graph input name>` / `output.<position>`).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from _shared.container_read import read_stored
from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import normalize_boundary_tensor
from karume.ir import IrGraph
from karume.pipeline import export_to_file

from .card import VOWEL_DETECTOR_LICENSE
from .distribution import VOWEL_DETECTOR_GRAPH_ROLE
from .patch import gru_forward

#: Parent of the real weights (input material placed by hand — docs/assets-layout.md).
MODELS_ROOT = INPUTS_ROOT / "vowel-detector"

#: Default checkpoint (when `--ckpt` is not given).
DEFAULT_CKPT = MODELS_ROOT / "crnn_epoch3.pt"

#: Length of the golden cases and the tracing example (T10 = number of 10 ms frames — 200 is 2.0
#: seconds). **Not the set of lengths the graph accepts** (the axis is symbolic, so any length runs
#: through the one graph).
DEFAULT_LENGTH = 200

#: Length step (the 20 ms output grid — "The length axis" in the module docstring).
LENGTH_MULTIPLE = 2

#: Minimum length. Two output frames = the smallest shape in which the reverse GRU carries state at
#: least once (with one frame the recurrence along time vanishes, and an unrolling error does not
#: show in the values).
MIN_LENGTH = 4

#: Lower bound of the symbol `T` (= number of 20 ms frames). The symbol-side spelling of the same
#: fact as `MIN_LENGTH`, and also the line that avoids 0 / 1 specialization (convert's two-point
#: evaluation requires a lower bound of 2 or more).
SYM_MIN = MIN_LENGTH // LENGTH_MULTIPLE

#: Upper bound of the symbol `T` = **the operating upper bound the distribution declares**
#: (`VOWEL_DETECTOR_MAX_FRAMES` in `vowel_detector/distribution.py` ships it converted into 10 ms
#: frames, and the pipeline fails loudly on an overrun).
#:
#: 30,000 × 20 ms = **10 minutes** of audio. It is derived from the size of the largest intermediate
#: tensor: the output of the GRU's input-side GEMM, `gates = [T, 1, 3H] = [T, 1, 384]` f32
#: ({@link vowel_detector.patch.gru_forward} — 4 of them are built, layer × direction), is 1,536 B
#: per 20 ms frame, so **46.1 MB** for 10 minutes. That leaves 2.9× headroom against the WebGPU spec
#: default `maxStorageBufferBindingSize` of 128 MiB = 134.2 MB (karume requests the adapter's
#: actual value, so on real devices it is wider still — `gpu/device.ts`).
#:
#: NOTE: the actual shapes of the conv stage are the input `[1, 83, 2T]` (19.2 MB) and the conv
#: output `[1, 160, T]` (19.2 MB), both smaller than gates (conv halves the time axis, so the
#: output length is not `2T`).
#:
#: MUST: when raising it, re-measure the upper bound of the intermediate tensors. If only the
#: declaration grows, an overrun shows up not as a check of the distribution but as **an allocation
#: failure on the user's machine**.
SYM_MAX = 30_000

MODEL_FILE = "model.krm"

#: Provenance baked into the container (container-v1 §2.3). The license identifier is taken from the
#: card side's source of truth ({@link vowel_detector.card.VOWEL_DETECTOR_LICENSE}) — the two tables
#: must not move independently.
PROVENANCE = Provenance(license=VOWEL_DETECTOR_LICENSE)
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: Name of the graph input (= the parameter name of {@link Crnn.forward}). It is checked after
#: export and stops on a mismatch — inputs are passed by position, so a silent drift would swap only
#: the golden inputs.
INPUT_NAME = "features"

#: Keys expected in the outer wrapper of the `.pt` (the training script writes
#: `{"model": state_dict, "epoch": int}`). Any key added or removed is an upstream change, so it
#: fails loudly.
WRAPPER_KEYS = ("model", "epoch")
#: The key holding the state_dict.
STATE_DICT_KEY = "model"

#: Prefix of the storage keys of folded constants (`karume.convert` gives it to the constants it
#: bakes). Held to tell them apart from initializers that come from the weights.
CONST_PREFIX = "const."

#: Feature dimensions (log-mel 80 + DSP 3 — "The input contract" in the module docstring).
N_MELS = 80
DSP_DIM = 3
FEATURE_DIM = N_MELS + DSP_DIM

#: Model shape (upstream `crnn.py`'s defaults as they are).
CONV_HIDDEN = 160
GRU_HIDDEN = 128
GRU_LAYERS = 2

#: Output classes (upstream's `phonemes.LIPSYNC_CLASSES` and `classes` of
#: `assets/feature_config.json`).
CLASSES: tuple[str, ...] = ("a", "i", "u", "e", "o", "N", "pau", "cons")

#: Position of the silence class and number of vowel classes (where {@link _sanity} reads the
#: probability mass).
PAU_INDEX = CLASSES.index("pau")
VOWEL_COUNT = 5

#: Random seed of the synthetic cases (the `noise` case). Independent of the global seed.
SEED = 20260813

#: Case names whose order {@link _sanity} checks (one silence-like and one voiced-like case).
SILENCE_CASE = "silence"
VOICED_CASE = "voiced"


class Crnn(nn.Module):
    """Verbatim copy of upstream's `training/src/vowel_detector/crnn.py` ("Why no dependency group
    is added" in the module docstring).

    conv halves the time axis (10 ms → 20 ms), then a 2-layer BiGRU (bidirectional because it is
    offline) produces logits over 8 classes. The default dimensions are upstream's, and they are
    **taken as arguments** so that pytest can run the same path with tiny synthetic weights
    (`load_state_dict(strict=True)` checks the real weights, shapes included).
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden: int = CONV_HIDDEN,
        gru_hidden: int = GRU_HIDDEN,
        gru_layers: int = GRU_LAYERS,
        classes: int = len(CLASSES),
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            hidden,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(gru_hidden * 2, classes)

    def _encode(self, features: torch.Tensor) -> torch.Tensor:
        """The 2 conv stages (10 ms → 20 ms grid). The front end shared by the replacement path and
        the reference path."""
        return self.conv(features.transpose(1, 2)).transpose(1, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """`[B, T10, feature_dim]` → logits `[B, T10//2, classes]`.

        The GRU is replaced with a sequence of `karume::gru_scan` ("Replacement layer" in the
        module docstring) — this is the path that is exported.
        """
        return self.head(gru_forward(self.gru, self._encode(features)))

    def reference_forward(self, features: torch.Tensor) -> torch.Tensor:
        """Reference path that runs `nn.GRU` as is (not exported).

        The difference from {@link forward} is exactly the replacement, so the **bit identity** of
        the two is the very claim that the replacement is correct (the standing check of
        {@link _write_io} and `--verify`).
        """
        hidden, _ = self.gru(self._encode(features))
        return self.head(hidden)


def default_out_dir(ckpt: Path) -> Path:
    """Default location of the output (`outputs/series/vowel-detector-<ckpt name>/`).

    **Only the checkpoint name** goes into the spelling — length is a symbolic axis, so there is
    only one graph, and the only axis left to separate series is the epoch generation (writing
    another epoch to the same location would silently overwrite the earlier weights). The series
    name is normalized to the existing convention (lowercase with hyphens).
    """
    return SERIES_ROOT / f"{MODELS_ROOT.name}-{ckpt.stem.lower().replace('_', '-')}"


def assert_length(length: int) -> None:
    """Check that the length is on the step ("The length axis" in the module docstring)."""
    if length < MIN_LENGTH or length % LENGTH_MULTIPLE:
        raise SystemExit(
            f"--length {length} is not at least {MIN_LENGTH} and a multiple of {LENGTH_MULTIPLE}"
            " (the output is on the 20 ms grid = one per two input frames)"
        )


def load_checkpoint(ckpt: Path) -> dict[str, torch.Tensor]:
    """Check the outer wrapper of the `.pt` and return the state_dict.

    MUST: `map_location="cpu"` — the upstream training script saves on CUDA, so without it an
    environment tries to grab a CUDA device and fails.
    """
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or set(obj) != set(WRAPPER_KEYS):
        keys = sorted(obj) if isinstance(obj, dict) else type(obj).__name__
        raise SystemExit(
            f"{ckpt}: the outer wrapper keys differ from the expected ones"
            f" (expected {sorted(WRAPPER_KEYS)} / actual {keys})"
        )
    state_dict = obj[STATE_DICT_KEY]
    non_tensor = [key for key, value in state_dict.items() if not torch.is_tensor(value)]
    if non_tensor:
        raise SystemExit(f"{ckpt}: the state_dict has non-tensor values: {sorted(non_tensor)}")
    return {key: value.detach().contiguous() for key, value in state_dict.items()}


def load_module(state_dict: Mapping[str, torch.Tensor]) -> Crnn:
    """Load the real weights into the copied model definition (`strict=True` — a drifted copy fails
    here)."""
    module = Crnn()
    module.load_state_dict(dict(state_dict), strict=True)
    return module.eval()


def _features(
    length: int, mel: torch.Tensor, voicing: float, log_energy: float, zcr: float
) -> torch.Tensor:
    """Concatenate the log-mel plane and the 3 constant DSP dimensions into `[1, T10, 83]`."""
    dsp = torch.tensor([[voicing, log_energy, zcr]]).expand(length, DSP_DIM)
    return torch.cat([mel, dsp], dim=1).unsqueeze(0)


def build_cases(length: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """`(name, features)` of the 4 golden cases (synthetic features `[1, T10, 83]`).

    No real audio is used because feature extraction (log-mel + DSP) is outside the graph — this
    script's claim is "the same features in give the same logits out", and audio decoding is not
    part of it.

    The 3 DSP dimensions of all 4 cases stay within the ranges measured on a real utterance (1.1
    seconds): voicing 0.08–0.90 / log energy ratio −0.73–0 / zero-crossing rate 0.04–0.85. For the
    log-mel z values too, the 3 cases {@link SILENCE_CASE} / {@link VOICED_CASE} / `noise` stay
    inside the measured range (−1.04–3.72), and they carry the "comparison on the same distribution
    as training".

    `ramp` alone is the exception: a regression case with the range **−2.0–4.0** that **goes
    beyond both ends** of the measured range. One case outside the distribution is kept to catch
    regressions that move the gradient and the rounding on the saturating side (no discrimination
    claim rests on `ramp`).

    {@link SILENCE_CASE} and {@link VOICED_CASE} are the pair that keeps the order check of
    {@link _sanity} from being vacuous, and `noise` / `ramp` detect numeric regressions (they
    exercise the ends of the range and the gradient).
    """
    generator = torch.Generator().manual_seed(SEED)
    # Silence: log-mel uniformly at the floor, low voicing and zero-crossing rate, energy −7.2
    # orders of magnitude below the utterance peak.
    silence = torch.full((length, N_MELS), -1.0)
    # Voiced: the 20 low bands are strong, the 30 mid bands fall off, the 30 high bands sit at the
    # floor (the spectral tilt of a vowel).
    voiced = torch.cat(
        [
            torch.full((20,), 2.0),
            torch.linspace(1.5, -1.0, 30),
            torch.full((30,), -1.0),
        ]
    ).expand(length, N_MELS)
    noise = torch.rand((length, N_MELS), generator=generator) * 4.0 - 1.0
    ramp = torch.linspace(-1.0, 3.0, N_MELS).reshape(1, N_MELS) + torch.linspace(
        -1.0, 1.0, length
    ).reshape(length, 1)
    return (
        (SILENCE_CASE, _features(length, silence, 0.05, -0.72, 0.10)),
        (VOICED_CASE, _features(length, voiced, 0.85, 0.0, 0.05)),
        ("noise", _features(length, noise, 0.5, -0.3, 0.5)),
        ("ramp", _features(length, ramp, 0.3, -0.5, 0.3)),
    )


def _write_io(
    module: Crnn,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Write each case's input and torch CPU expected output to `io.<case>.safetensors`.

    MUST: the expected values are taken from the **reference path** (`nn.GRU` itself —
    {@link Crnn.reference_forward}), and bit identity with the exported replacement path is checked
    for every case. If an error of the replacement carried straight into the expected values, the
    E2E would go green with "the exporter and the runtime agree and both are wrong" (the same trap
    as the `karume/custom_ops.py` docstring).

    The second return value is the expected output for the sanity record (passed in its
    `[1, T10//2, 8]` shape).
    """
    if len(graph.outputs) != 1:
        raise AssertionError(f"the IR has {len(graph.outputs)} outputs (logits are one output)")
    written: list[str] = []
    logits: dict[str, torch.Tensor] = {}
    for name, features in cases:
        with torch.no_grad():
            output = module.reference_forward(features)
            patched = module(features)
        if not torch.equal(patched, output):
            raise AssertionError(
                f"{name}: the replacement with gru_scan is not bit-identical to nn.GRU"
                f" (maxdiff {float((patched - output).abs().max())})"
            )
        # MUST: io is also lowered to the concrete representation of the IR's semantic dtype (the
        # boundary normalization of ADR 0009). Unless it matches what the runtime receives, the
        # Deno-side E2E cannot read the golden.
        tensors = {
            f"{INPUT_PREFIX}{INPUT_NAME}": normalize_boundary_tensor(
                features, f"{name}: input '{INPUT_NAME}'"
            ),
            f"{OUTPUT_PREFIX}0": normalize_boundary_tensor(
                output.detach().contiguous(), f"{name}: output 0"
            ),
        }
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
        logits[name] = output.detach()
    return written, logits


def _sanity(logits: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Check the **order** between a silence-like input and a voiced-like input (no thresholds).

    MUST: do not make the sanity check vacuous. A range check on the logits, or "the same inputs
    agree", verifies nothing. Checking two orders — (1) the silence case's mean P(pau) is the
    largest of all cases, (2) the voiced case's mean vowel mass (the sum of a/i/u/e/o) is the
    largest of all cases — catches a mixed-up weight layout (a transposed time and feature axis, a
    wrong GRU gate order) as well as an output that no longer depends on its input.

    It also checks that the cases **differ from each other** (four identical outputs mean the input
    is not getting through).
    """
    names = sorted(logits)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if torch.equal(logits[name], logits[other]):
                raise AssertionError(
                    f"{name} and {other} have identical outputs (the input has no effect)"
                )

    probabilities = {name: tensor.softmax(dim=-1) for name, tensor in logits.items()}
    pause = {name: float(tensor[..., PAU_INDEX].mean()) for name, tensor in probabilities.items()}
    vowel = {
        name: float(tensor[..., :VOWEL_COUNT].sum(dim=-1).mean())
        for name, tensor in probabilities.items()
    }
    _assert_largest(SILENCE_CASE, pause, "the silence case's mean P(pau) is not the largest")
    _assert_largest(VOICED_CASE, vowel, "the voiced case's mean vowel mass is not the largest")
    return {
        "pau_mean": {name: round(value, 4) for name, value in pause.items()},
        "vowel_mass_mean": {name: round(value, 4) for name, value in vowel.items()},
    }


def _assert_largest(expected: str, values: Mapping[str, float], claim: str) -> None:
    """Fail with `claim` unless `expected` is the largest in `values` (checks only the order)."""
    winner = max(values, key=lambda name: values[name])
    if winner != expected:
        raise AssertionError(
            f"{claim}: {expected}={values[expected]:.4f} < {winner}={values[winner]:.4f}"
        )


def assert_checkpoint_bytes(path: Path, state_dict: Mapping[str, torch.Tensor]) -> int:
    """Check that the initializers of the published container are **byte-identical** to the
    state_dict of the upstream `.pt`.

    Converting the weights is "read and write only", so changed values mean the conversion is
    broken. The comparison is on the raw bytes, not just the storage and shape (to claim "the values
    did not change" down to NaN bit patterns and −0.0). This series bakes only f32 storage, so the
    payload is laid out exactly like the upstream bytes.

    Initializers that do not come from the weights (constants baked by folding) carry the `const.`
    prefix, so any other extra key fails as "a weight silently rewritten and stored under another
    name". Returns the number of tensors that matched.

    MUST: read **the union of the part sequence** ({@link _shared.container_read.read_stored}) —
    opening only the one representative path would read part 0, which holds no tensors at all, and
    report "everything is missing". Entries split into pieces are folded back in row order by the
    supply plan, so the meaning of the comparison does not depend on whether there is a split.
    """
    stored = read_stored(path)
    missing = sorted(set(state_dict) - set(stored))
    if missing:
        raise AssertionError(f"{path}: state_dict keys missing from the initializers: {missing}")
    extra = sorted(key for key in set(stored) - set(state_dict) if not key.startswith(CONST_PREFIX))
    if extra:
        raise AssertionError(f"{path}: initializers that do not come from the weights: {extra}")
    for name in sorted(set(stored) & set(state_dict)):
        source = state_dict[name]
        found = stored[name]
        if found.layout != "f32" or found.shape != tuple(source.shape):
            raise AssertionError(
                f"tensor '{name}': storage / shape mismatch"
                f" (source {source.dtype} {tuple(source.shape)} /"
                f" read back {found.layout} {found.shape})"
            )
        if found.payload != source.numpy().tobytes():
            raise AssertionError(f"tensor '{name}': the bytes do not match")
    return len(state_dict)


def export_series(ckpt: Path, out_dir: Path, length: int) -> dict[str, Any]:
    """Write the IR container and the golden io, and return a summary.

    MUST: the output is written to a staging directory and published only after passing **every
    check** (input order, ckpt byte identity, sanity). Placing it in the final location before the
    checks lets a failed run leave behind "assets that pass the acceptance checks" — the io golden
    is taken from the same broken module, so they agree with each other and the TS-side comparison
    **goes green** ("when it may be published" is spelled out in
    {@link _shared.decode_series._publish}; the discipline of the swap and the clean-up is the core
    primitive {@link karume.artifacts.staged_publication}).
    """
    assert_length(length)
    state_dict = load_checkpoint(ckpt)
    module = load_module(state_dict)
    cases = build_cases(length)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    _, example = cases[0]
    started = time.monotonic()
    with staged_publication(out_dir) as staged:
        # The writer creates the directory (the primitive does not create it — it only hands over a
        # path).
        staged.mkdir()
        graph = export_to_file(
            module,
            (example,),
            staged / MODEL_FILE,
            provenance=PROVENANCE,
            # The graph name is the **component name** (= the weights key of karume.json). It is not
            # derived from the directory name — this family's container sits directly in the series,
            # so the directory name is the series name (`vowel-detector-<checkpoint name>`), not the
            # component name (container-v1 §2.1).
            graph_name=VOWEL_DETECTOR_GRAPH_ROLE,
            # MUST: the symbol goes on the 20 ms output grid (`2*Dim("T")` — "The length axis" in
            # the module docstring). With a plain `Dim("T")` the conv output becomes a floor
            # division, which does not fit the dimension language.
            dynamic_shapes={INPUT_NAME: {1: 2 * Dim("T", min=SYM_MIN, max=SYM_MAX)}},
            symbol_names=("T",),
        )
        elapsed = time.monotonic() - started
        declared = tuple(item.name for item in graph.inputs)
        if declared != (INPUT_NAME,):
            raise AssertionError(
                f"the graph inputs are {declared}, not the expected {(INPUT_NAME,)}"
            )
        matched = assert_checkpoint_bytes(staged / MODEL_FILE, state_dict)
        written, logits = _write_io(module, graph, cases, staged)
        # MUST: evaluate before publishing (the only non-vacuous check of this series — on failure
        # the whole staging directory is removed).
        sanity = _sanity(logits)
    return {
        "dir": str(out_dir),
        "length": length,
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": sum(p.stat().st_size for p in container_parts(out_dir / MODEL_FILE)),
        "export_seconds": round(elapsed, 2),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "declared_input_shape": [str(dim) for dim in graph.inputs[0].shape],
        "input_shape": list(example.shape),
        "output_shape": list(logits[cases[0][0]].shape),
        "symbol_range": [SYM_MIN, SYM_MAX],
        "checkpoint_tensors_byte_identical": matched,
        "sanity": sanity,
    }


def _diff_entry(
    stage: str,
    claim: str,
    got: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Equivalence report for one stage (case name → maxdiff, and bit identity over all cases).

    `bit_exact` is a stronger claim than "difference 0" (`0.0 == -0.0` has difference 0, but the
    bits differ).
    """
    return {
        "stage": stage,
        "claim": claim,
        "maxdiff": {
            name: float((got[name] - expected[name]).abs().max()) for name in sorted(expected)
        },
        "bit_exact": all(torch.equal(got[name], expected[name]) for name in expected),
    }


#: Lengths `--verify` exercises (**several lengths** on top of `--length`). Some errors of the
#: replacement appear only at certain lengths (the scan boundary of the reverse direction, the
#: concatenation order between layers), and one length alone cannot hit them.
VERIFY_LENGTHS: tuple[int, ...] = (MIN_LENGTH, 6, 18, 137 * 2)


def verify_patch(ckpt: Path, length: int) -> list[dict[str, Any]]:
    """Measure that the replacement with `gru_scan` is equivalent to `nn.GRU` (`--verify`).

    The same claim as the other scripts' `--verify` (eager equivalence before and after the
    replacement), against {@link Crnn.reference_forward} = upstream's `nn.GRU` path as is. If this
    breaks, "the graph bakes but the numbers are something else", so **bit_exact is the substance of
    the claim** (stronger than difference 0 — {@link _diff_entry}).

    NOTE: it does not emit only to match the shape of the other scripts; there is no risk of process
    contamination here (no class attribute is replaced at all — the replacement lives only inside
    `Crnn.forward`).
    """
    assert_length(length)
    module = load_module(load_checkpoint(ckpt))
    entries: list[dict[str, Any]] = []
    for probe in sorted({length, *VERIFY_LENGTHS}):
        cases = build_cases(probe)
        with torch.no_grad():
            reference = {name: module.reference_forward(features) for name, features in cases}
            got = {name: module(features) for name, features in cases}
        entry = _diff_entry(f"gru-scan-t{probe}", "bit-exact", got, reference)
        if not entry["bit_exact"]:
            raise AssertionError(
                f"T10={probe}: the replacement with gru_scan is not bit-identical to nn.GRU:"
                f" maxdiff={entry['maxdiff']}"
            )
        entries.append(entry)
    return entries


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--length",
        type=int,
        default=DEFAULT_LENGTH,
        help=f"number of 10 ms frames T10 of the golden cases (a multiple of {LENGTH_MULTIPLE},"
        f" default {DEFAULT_LENGTH}). **Not the set of lengths the graph accepts** — the time axis"
        " is symbolic, so the one baked graph accepts any length",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: outputs/series/vowel-detector-<ckpt name>/)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="measure the equivalence of the gru_scan replacement and nn.GRU (does not emit)",
    )
    args = parser.parse_args(argv)
    if args.verify:
        for entry in verify_patch(args.ckpt, args.length):
            print(
                f"{entry['stage']} ({entry['claim']}): bit_exact={entry['bit_exact']}"
                f" maxdiff={entry['maxdiff']}"
            )
        return
    out_dir = args.out if args.out is not None else default_out_dir(args.ckpt)
    summary = export_series(args.ckpt, out_dir, args.length)
    print(json.dumps({"ckpt": str(args.ckpt), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
