"""Distribution recipe of vowel detection — series layout, output path table, quant table and choice
of card (ADR 0065 decision 2).

The general-purpose assembly engine (placement, folding of shared slots, sha256, manifest,
staging/swap, verification) lives in `karume.dist`. What lives here is **only the facts specific to
vowel detection**: what to pick up from which series directory, at which path of the distribution
and under which dtype label to list it, and which quant to make the default.

What is shipped is **one CRNN graph** and the **one mel basis** the feature extraction needs (an
`assets` entry). There is a single storage dtype series, f32, so there is also a single quant.

## Length buckets are gone (ADR 0056 / 0057)

`aten.gru.input` used to be fully unrolled along time, so T could not be a dynamic axis; four length
buckets (250 / 500 / 1000 / 2000 frames) were listed on the role axis of the weights and inputs were
rounded up with right zero padding. With the replacement by `gru_scan` (ADR 0056) and symbol binding
from derived dimensions (ADR 0057), **any length runs through one graph**, so the only role left is
`crnn`.

As a side effect **the numeric differences caused by padding are gone**: because the reverse GRU
carries state back from the padded side, the `.lab` of the bucket path diverged from the true-length
path in three ways — "a 40 ms pau at the end", "a 20 ms boundary shift" and "a pau in the middle of
the utterance" (measured on 4 real recordings × 10 padding steps). The pipeline now runs at the true
length, so the distribution produces **the same** `.lab` that the real-weight E2E pins.

## Where `pipelineConfig` comes from

The feature contract (`sampleRate` / `featureDim` / `classes`) is verbatim from upstream's
`feature_config.json`. `maxFrames` is **the upper bound of the baked graph's symbolic dimension**;
the IR holds no value ranges (`symbols` in `docs/ir-v2.md` are names only), so it is a number that
exists only in the distribution — held the same way as SBV2's `maxTokens` / `maxFrames`, and
`vowel_detector/tests/test_distribution.py` checks it against the script's constant.
{@link assert_vowel_detector_graph} measures that they fit together.

The public surface is the one {@link PIPELINE} (`karume.dist.Pipeline`) — the repository's dist
driver (`tools/export-recipes/dist.py`) merges it into the core's PIPELINES.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from _shared.licenses import mit_license
from _shared.paths import INPUTS_ROOT
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
    graph_inputs,
    ir_graph,
    table_payload,
)

from .card import VOWEL_DETECTOR_HOP, render_vowel_detector_model_card

#: Pipeline contract (ADR 0041 §2 — per model). The set the TS side accepts is
#: `VOWEL_DETECTOR_PIPELINE_NAME` / `VOWEL_DETECTOR_PIPELINE_MAJOR`.
VOWEL_DETECTOR_PIPELINE = "vowel-detector/1"

#: The default model name = **the checkpoint generation** (the same derivation as the spelling
#: `vowel_detector.export.default_out_dir` bakes into the series name — `crnn_epoch3.pt` →
#: `crnn-epoch3`). Retrained weights are listed under another model name (the manifest keys are the
#: source of truth for generations).
VOWEL_DETECTOR_DEFAULT_MODEL = "crnn-epoch3"

#: Prefix of the series name (`vowel-detector-<model name>` — the same word as the script's
#: `MODELS_ROOT.name`).
VOWEL_DETECTOR_PREFIX = "vowel-detector"

#: Location of the upstream material (`inputs/vowel-detector/` — the same as
#: `vowel_detector.export.MODELS_ROOT`).
VOWEL_DETECTOR_INPUTS_DIRNAME = "vowel-detector"

#: Role name of the graph (the distribution has exactly one CRNN). One word shared by the placement
#: table, the output paths and the weights declaration.
VOWEL_DETECTOR_GRAPH_ROLE = "crnn"

#: **Operating upper bound** carried in `pipelineConfig` (in 10 ms frames = 10 minutes). It is the
#: upper bound `vowel_detector.export.SYM_MAX` of the baked graph's symbolic dimension `T` (20 ms
#: grid) converted into input-side units; `vowel_detector/tests/test_distribution.py` checks that
#: they match.
#:
#: MUST: keep it equal to the script's value. The IR holds no ranges for symbols (names only), so
#: **only the pipeline that reads this number** can stop input longer than declared. If they drift,
#: an overrun shows up not as a check of the distribution but as an allocation failure on the user's
#: machine.
VOWEL_DETECTOR_MAX_FRAMES = 60_000

#: **Operating lower bound** carried in `pipelineConfig` (in 10 ms frames = 40 ms). It is the lower
#: bound `vowel_detector.export.SYM_MIN` of the baked graph's symbolic dimension `T` converted into
#: input-side units (= `export.MIN_LENGTH`).
#:
#: MUST: it is needed here for the same reason as the upper bound. The IR holds no ranges for
#: symbols, so **only the pipeline that reads this number** can stop input shorter than
#: `Dim(min=…)`. Without the declaration, audio of 2–3 frames of 10 ms binds `T = 1` and runs
#: **outside the traced range** (no exception is raised; only the values leave what is guaranteed).
VOWEL_DETECTOR_MIN_FRAMES = 4

#: Name of the graph input (`vowel_detector.export.INPUT_NAME`) and the time-axis stride of the
#: output (conv stride 2 — two input frames per output frame).
VOWEL_DETECTOR_INPUT = "features"
VOWEL_DETECTOR_TIME_STRIDE = 2

#: Name of the time-axis symbol (the spelling `vowel_detector.export` bakes with
#: `convert(symbol_names=("T",))`). It appears as `2T` on the input (10 ms grid) and `T` on the
#: output (20 ms grid).
VOWEL_DETECTOR_SYMBOL = "T"

#: Source of the feature contract (upstream's `assets/feature_config.json`, placed by hand in
#: `inputs/vowel-detector/`). The mel basis is also in it and is moved into the distribution as a
#: one-tensor safetensors file.
VOWEL_DETECTOR_FEATURE_CONFIG_FILE = "feature_config.json"

#: Tensor key of the shipped mel basis (paired with `MEL_BASIS` in the TS side's `pipeline.ts`).
VOWEL_DETECTOR_MEL_BASIS_KEY = "mel_basis"

#: Number of DSP auxiliary features (voicing / log energy / zero-crossing rate). Held only to check
#: the internal consistency `feature_dim = n_mels + 3` (the source of truth for the value is
#: upstream's feature extraction).
VOWEL_DETECTOR_DSP_DIM = 3

#: Number of output classes (`classes` of `feature_config.json` — the order is not this side's
#: concern; the check that **the order is the id** lives in the TS side's `config.ts`).
VOWEL_DETECTOR_CLASS_COUNT = 8

#: Feature-extraction dimensions **baked into** the only reader
#: (`packages/models/src/vowel-detector/features.ts`). A distribution whose upstream config declares
#: other values assembles but is refused when it is loaded on a real device —
#: {@link _vowel_detector_expects} stops it at the assembly stage.
VOWEL_DETECTOR_SAMPLE_RATE = 16_000
VOWEL_DETECTOR_N_FFT = 512
VOWEL_DETECTOR_N_MELS = 80


#: Relative output paths (**inside the model subtree**) — the one place shared by the placement
#: table and the manifest. Putting the storage dtype in the file name is the same form as other
#: families.
VOWEL_DETECTOR_OUTPUT_PATHS: Mapping[str, str] = {
    VOWEL_DETECTOR_GRAPH_ROLE: "model.f32.krm",
    VOWEL_DETECTOR_MEL_BASIS_KEY: f"features/{VOWEL_DETECTOR_MEL_BASIS_KEY}.safetensors",
}

#: Required storage dtype (the same rationale as other families — a measured incident where a plain
#: asset passed assembly, loading and execution silently until the reference-match check). The mel
#: basis is f32 written by this side, so it is not listed.
VOWEL_DETECTOR_STORAGE_REQUIREMENTS: Mapping[str, str] = {VOWEL_DETECTOR_GRAPH_ROLE: "f32"}

#: Storage vocabulary that **must not appear** in each role's binding table
#: ({@link assert_storage_absent}). {@link VOWEL_DETECTOR_STORAGE_REQUIREMENTS} is a one-way check
#: of "is the required dtype present", and **compressed series also hold their ineligible
#: weights** (biases / norms / graph constants, i8's per-channel scales, i4's group scales) in f32,
#: so "contains f32" is true even for f16 / i8 / i4 assets — plugging a compressed series into the
#: f32 slot passes a presence check alone.
#:
#: This script (`vowel_detector/export.py`) bakes only f32, but what the forbidden table closes is
#: **a mix-up of the series root** (`--series` pointing at another tree / a compressed series of
#: another family placed by hand), which happens regardless of the dtypes the script supports. A
#: mix-up of the series root cannot in principle be detected by the numeric checks
#: (ADR 0027 / 0029), so this is the only detector.
#:
#: MUST: the forbidden storage is held **as a set per role** and names **all** of the compressed
#: storage that can be written (`karume.emit.WEIGHT_DTYPES` other than f32) — leaving even one out
#: lets exactly that storage form through silently (the same discipline as anima / irodori / sbv2).
#: I32 is not listed because i32 is plain storage, not compression (the plain side of
#: `karume.emit`), and it appears routinely in plain series as index tables.
VOWEL_DETECTOR_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    VOWEL_DETECTOR_GRAPH_ROLE: ("f16", "bf16", "i8", "i4")
}

#: Declaration of weights (dtype label → role name). There is one graph with only one dtype, so the
#: quant table may be empty ({@link complete_quant_weights} fills it into a complete mapping).
VOWEL_DETECTOR_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    VOWEL_DETECTOR_GRAPH_ROLE: {"f32": WeightFiles(VOWEL_DETECTOR_GRAPH_ROLE)}
}

#: Declaration of assets (unconditional files independent of the quant choice — the one mel basis
#: of the feature extraction).
VOWEL_DETECTOR_ASSETS: Mapping[str, str] = {
    VOWEL_DETECTOR_MEL_BASIS_KEY: VOWEL_DETECTOR_MEL_BASIS_KEY
}

VOWEL_DETECTOR_QUANTS: Mapping[str, Any] = {"f32": {"weights": {}, "session": {}}}
VOWEL_DETECTOR_DEFAULT_QUANT = "f32"


def vowel_detector_series_name(model: str) -> str:
    """Model name → series directory name (the same expression as
    `vowel_detector.export.default_out_dir`)."""
    return f"{VOWEL_DETECTOR_PREFIX}-{model}"


def vowel_detector_repo_name(model: str) -> str:
    """Distribution repository name. It **does not include the model name** — the checkpoint
    generation (`crnn-epoch3`) is a fact spelled by the manifest keys, not part of the repository's
    name, and a new repository is not created each time the generation goes up (upstream also
    distributes from the one `vowel-detector` repository).
    """
    return f"karume-{VOWEL_DETECTOR_PREFIX}"


@dataclass(frozen=True)
class VowelDetectorSources:
    """Inputs of the assembly: the one CRNN series, and the location of the upstream material
    (`inputs/` — not generated output).

    The latter is needed to **derive the feature contract and the mel basis instead of baking them
    in** (only the one `feature_config.json` is read).
    """

    #: Parent of the series directories (`outputs/series/`) — the series name is joined onto it.
    series_dir: Path
    #: Upstream material (the directory holding `feature_config.json`).
    model: Path
    #: Model name (generation) used to look up the series.
    model_name: str

    @property
    def series(self) -> Path:
        return self.series_dir / vowel_detector_series_name(self.model_name)


def vowel_detector_sources(
    series_dir: Path, model: str = VOWEL_DETECTOR_DEFAULT_MODEL
) -> VowelDetectorSources:
    """Build the inputs from the series parent directory (`outputs/series/`) and the spellings of
    `_shared.paths`."""
    return VowelDetectorSources(
        series_dir=series_dir,
        model=INPUTS_ROOT / VOWEL_DETECTOR_INPUTS_DIRNAME,
        model_name=model,
    )


def vowel_detector_placements(sources: VowelDetectorSources) -> dict[str, Path]:
    """Role name → source file. The output paths are held by {@link VOWEL_DETECTOR_OUTPUT_PATHS}.

    Anything not in this table does not go into the output (the `io.*.safetensors` files listed in
    the series are dropped this way). The mel basis is an output of conversion, not a placement, so
    it does not appear here (the same form as SBV2's two tables).
    """
    return {VOWEL_DETECTOR_GRAPH_ROLE: sources.series / "model.krm"}


def vowel_detector_feature_config(model_dir: Path) -> Mapping[str, Any]:
    """Read upstream's `feature_config.json` (the only source of the feature contract and the mel
    basis)."""
    path = model_dir / VOWEL_DETECTOR_FEATURE_CONFIG_FILE
    if not path.is_file():
        raise DistError(
            f"Assembly input is missing: {path}"
            " (place the upstream vowel-detector's assets/feature_config.json here)"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} cannot be read as JSON") from error
    if not isinstance(raw, dict):
        raise DistError(f"{path}: the top level is not an object")
    return raw


def _vowel_detector_int(raw: Mapping[str, Any], key: str, where: str) -> int:
    """Check and read a positive-integer field of the feature config."""
    value = raw.get(key)
    # bool is a subclass of int. Letting `"n_mels": true` through as 1 loosens the dimension check.
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DistError(f"{where}: {key} is not a positive integer ({value!r})")
    return value


def vowel_detector_classes(raw: Mapping[str, Any], where: str) -> list[str]:
    """Read the class vocabulary (shipped **in its order** — the order is the class id).

    NOTE: the vocabulary itself (the spelling and order of `a`/`i`/…/`cons`) is not held here. The
    upstream config is its only source; a second table copying it here would make it possible to
    silently build "a declaration that passes while labels are swapped" when only one copy changes.
    **The check against the accepted set lives on the load side**
    (`packages/models/src/vowel-detector/config.ts`); this side checks only the structure.
    """
    value = raw.get("classes")
    if not isinstance(value, list) or len(value) != VOWEL_DETECTOR_CLASS_COUNT:
        raise DistError(
            f"{where}: classes is not an array of length {VOWEL_DETECTOR_CLASS_COUNT} ({value!r})"
        )
    for entry in value:
        if not isinstance(entry, str) or not entry:
            raise DistError(
                f"{where}: classes has an element that is not a non-empty string ({value!r})"
            )
    if len(set(value)) != len(value):
        raise DistError(f"{where}: classes has duplicates ({value!r})")
    return list(value)


def vowel_detector_mel_basis(raw: Mapping[str, Any], where: str) -> np.ndarray:
    """Read the mel basis `[n_mels, n_fft // 2 + 1]` as an f32 matrix.

    MUST: the shape is checked against the expectation built from **both** `n_mels` and `n_fft` —
    a shifted basis only turns the features into "plausible other values" and goes all the way
    through with shape and value range still matching.

    MUST: also check that every row has a positive element. An empty triangular window (a mel
    channel that fell outside the band) pins that column to `log(1e-5)` forever, but as one of 80
    it only shows up as "quietly wrong", by eye and by the numbers alike.
    """
    n_mels = _vowel_detector_int(raw, "n_mels", where)
    n_fft = _vowel_detector_int(raw, "n_fft", where)
    expected = (n_mels, n_fft // 2 + 1)
    value = raw.get(VOWEL_DETECTOR_MEL_BASIS_KEY)
    if not isinstance(value, list):
        raise DistError(f"{where}: {VOWEL_DETECTOR_MEL_BASIS_KEY} is not an array ({type(value)})")
    try:
        basis = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise DistError(
            f"{where}: {VOWEL_DETECTOR_MEL_BASIS_KEY} does not form an f32 matrix"
        ) from error
    if basis.shape != expected:
        raise DistError(
            f"{where}: {VOWEL_DETECTOR_MEL_BASIS_KEY} has shape {basis.shape},"
            f" expected {expected} from n_mels / n_fft"
        )
    if not np.isfinite(basis).all():
        raise DistError(f"{where}: {VOWEL_DETECTOR_MEL_BASIS_KEY} has non-finite elements")
    empty = [int(row) for row in np.flatnonzero(basis.max(axis=1) <= 0)]
    if empty:
        raise DistError(
            f"{where}: {VOWEL_DETECTOR_MEL_BASIS_KEY} has empty mel channels (rows {empty})"
        )
    return basis


def _vowel_detector_expects(raw: Mapping[str, Any], key: str, expected: int, where: str) -> int:
    """Read a feature-extraction contract value, checking it against **the value the reader has
    baked in**.

    MUST: do not drop this. With a mismatched value the assembly still passes and `verify_dist` is
    green, but the result is a distribution that the only reader refuses at load time (it fails
    only after assembly → HF upload → loading on a real device).
    """
    value = _vowel_detector_int(raw, key, where)
    if value != expected:
        raise DistError(
            f"{where}: {key} is {value} — karume's feature extraction has {expected} baked in"
            " (packages/models/src/vowel-detector/features.ts), so this distribution would be"
            " refused at load time"
        )
    return value


def vowel_detector_pipeline_config(feature_config: Mapping[str, Any], where: str) -> dict[str, Any]:
    """Build `pipelineConfig` (the 5 fields of the TS-side schema) from the upstream config and the
    script's declarations.

    Only `minFrames` / `maxFrames` are **numbers of the baked graph**
    ({@link VOWEL_DETECTOR_MIN_FRAMES} / {@link VOWEL_DETECTOR_MAX_FRAMES} = the range of the
    symbolic dimension `T` converted into input-side units); the other 3 fields are verbatim from
    the upstream config.

    The 4 dimension fields (`sample_rate` / `n_fft` / `n_mels` / `hop`) are checked against the
    reader's baked-in values ({@link _vowel_detector_expects}). When {@link vowel_detector_classes}
    says "no second table copying the upstream config", it means the **vocabulary**; these are not a
    copy of upstream but **the values karume's implementation can accept** (if upstream changes a
    dimension, the TS baked-in value moves with it) — the same form as siglip2's
    `SIGLIP2_RESAMPLE` / `SIGLIP2_RESCALE_FACTOR`.
    """
    sample_rate = _vowel_detector_expects(
        feature_config, "sample_rate", VOWEL_DETECTOR_SAMPLE_RATE, where
    )
    _vowel_detector_expects(feature_config, "n_fft", VOWEL_DETECTOR_N_FFT, where)
    # hop is not carried in `pipelineConfig`, but the card's conversion to seconds uses it
    # ({@link vowel_detector.card.VOWEL_DETECTOR_HOP} in card.py) — without something to check it
    # against, the day upstream changes it the card alone would silently state the wrong seconds.
    _vowel_detector_expects(feature_config, "hop", VOWEL_DETECTOR_HOP, where)
    feature_dim = _vowel_detector_int(feature_config, "feature_dim", where)
    n_mels = _vowel_detector_expects(feature_config, "n_mels", VOWEL_DETECTOR_N_MELS, where)
    if feature_dim != n_mels + VOWEL_DETECTOR_DSP_DIM:
        raise DistError(
            f"{where}: feature_dim {feature_dim} is not n_mels {n_mels} +"
            f" DSP {VOWEL_DETECTOR_DSP_DIM} — the feature breakdown disagrees with upstream"
        )
    return {
        "sampleRate": sample_rate,
        "featureDim": feature_dim,
        "classes": vowel_detector_classes(feature_config, where),
        "minFrames": VOWEL_DETECTOR_MIN_FRAMES,
        "maxFrames": VOWEL_DETECTOR_MAX_FRAMES,
    }


def assert_vowel_detector_graph(
    placements: Mapping[str, Path], pipeline_config: Mapping[str, Any]
) -> None:
    """Measure, before placement, that the CRNN graph's inputs and outputs fit `pipelineConfig`.

    MUST: do not drop this. The feature dimension and class count come from the upstream config and
    the shapes from the graph — **decided independently** — so combining weights derived from
    training on other features with the current config fails nowhere up to this point.

    MUST: also check that the time axis is **symbolic** (input `2T` / output `T`). An older graph
    baked at a fixed length has the same names and ranks; shipping it yields a manifest that is
    valid as "a distribution that runs at that one length only" (on the user's machine, every audio
    clip of any other length fails).
    """
    path = placements[VOWEL_DETECTOR_GRAPH_ROLE]
    graph = ir_graph(path)
    inputs = graph_inputs(graph, path)
    if tuple(inputs) != (VOWEL_DETECTOR_INPUT,):
        raise DistError(
            f"{path}: the graph inputs are {list(inputs)},"
            f" not the expected {[VOWEL_DETECTOR_INPUT]}"
            " — the runtime binds by name, so a changed spelling cannot be bound"
        )
    if graph.get("symbols") != [VOWEL_DETECTOR_SYMBOL]:
        raise DistError(
            f"{path}: the symbolic dimensions are {graph.get('symbols')!r} — expected"
            f" [{VOWEL_DETECTOR_SYMBOL!r}] (only the time axis is symbolic)"
        )
    feature_dim = pipeline_config["featureDim"]
    stride_dim = f"{VOWEL_DETECTOR_TIME_STRIDE}{VOWEL_DETECTOR_SYMBOL}"
    expected_input = [1, stride_dim, feature_dim]
    if inputs[VOWEL_DETECTOR_INPUT] != expected_input:
        raise DistError(
            f"{path}: input '{VOWEL_DETECTOR_INPUT}' is"
            f" {inputs[VOWEL_DETECTOR_INPUT]!r}, expected {expected_input!r}"
            " — the length on the 10 ms grid is twice the symbol T (batch is a static 1)"
        )
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise DistError(
            f"{path}: the graph outputs are {outputs!r} — exactly one logits output is needed"
        )
    values = graph.get("values")
    value = values.get(outputs[0]) if isinstance(values, dict) else None
    out_shape = value.get("shape") if isinstance(value, dict) else None
    expected_output = [1, VOWEL_DETECTOR_SYMBOL, len(pipeline_config["classes"])]
    if out_shape != expected_output:
        raise DistError(
            f"{path}: graph output '{outputs[0]}' has shape {out_shape!r},"
            f" expected {expected_output!r}"
            " — the output is the 20 ms grid (one per two input frames) × the class count"
        )


def vowel_detector_plan(
    sources: VowelDetectorSources, model: str = VOWEL_DETECTOR_DEFAULT_MODEL
) -> ModelPlan:
    """Build the plan for one vowel-detection model (all checks and config reading happen here)."""
    assert_model_name(model)
    placements = vowel_detector_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, VOWEL_DETECTOR_STORAGE_REQUIREMENTS)
        assert_storage_absent(role, source, VOWEL_DETECTOR_STORAGE_FORBIDDEN)
    feature_config = vowel_detector_feature_config(sources.model)
    where = str(sources.model / VOWEL_DETECTOR_FEATURE_CONFIG_FILE)
    pipeline_config = vowel_detector_pipeline_config(feature_config, where)
    assert_vowel_detector_graph(placements, pipeline_config)
    artifacts = {
        role: Artifact(VOWEL_DETECTOR_OUTPUT_PATHS[role], source=source)
        for role, source in placements.items()
    }
    artifacts[VOWEL_DETECTOR_MEL_BASIS_KEY] = Artifact(
        VOWEL_DETECTOR_OUTPUT_PATHS[VOWEL_DETECTOR_MEL_BASIS_KEY],
        payload=table_payload(
            VOWEL_DETECTOR_MEL_BASIS_KEY, vowel_detector_mel_basis(feature_config, where)
        ),
    )
    return ModelPlan(
        name=model,
        pipeline=VOWEL_DETECTOR_PIPELINE,
        artifacts=artifacts,
        weights=VOWEL_DETECTOR_WEIGHTS,
        assets=VOWEL_DETECTOR_ASSETS,
        quants=complete_quant_weights(VOWEL_DETECTOR_WEIGHTS, VOWEL_DETECTOR_QUANTS),
        default_quant=VOWEL_DETECTOR_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


def vowel_detector_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """Build the plan for one vowel-detection model from the `--series` parent (the CLI's dispatch
    target)."""
    return vowel_detector_plan(vowel_detector_sources(series_dir, model), model)


#: Copyright lines for the `LICENSE.md` at the root of the distribution repository (verbatim from
#: upstream's `LICENSE` / `NOTICE.txt` — the spelling comes from the same place as the note on
#: {@link vowel_detector.card.VOWEL_DETECTOR_LICENSE}).
#:
#: MUST: MIT requires "the full text + the copyright line" at the root of the distribution
#: repository (ADR 0092 decision 7) — an MIT text without the copyright line has nothing for "the
#: above copyright notice" to point to.
VOWEL_DETECTOR_COPYRIGHTS: tuple[str, ...] = ("Copyright (c) 2026 Spectopathy",)

#: The one entry for `--pipeline vowel-detector` (the driver merges it into the core's PIPELINES).
PIPELINE = Pipeline(
    default_model=VOWEL_DETECTOR_DEFAULT_MODEL,
    repo_name=vowel_detector_repo_name,
    plan=vowel_detector_dist_plan,
    # There is only one attribution (one upstream repository, one license — the attribution of
    # the training material is also one set tied to the weights). With nothing to choose, the
    # profile can be omitted.
    card_profiles={"vowel-detector": render_vowel_detector_model_card},
    # No modification notice (`NOTICE.md`) — MIT does not require one, and upstream and the
    # copyright holder are the same, so there is nobody for the notice to address (that is where
    # it differs from the Apache-licensed families).
    root_files={"LICENSE.md": mit_license(VOWEL_DETECTOR_COPYRIGHTS)},
)
