"""The script that writes chunk 0 of the Wan2.2 TI2V-5B VAE encoder (the single conditioning image
of I2V) out to 3 IR v2 graphs and makes the goldens of the GPU check (ADR 0121 decision 11,
step 9a).

What it writes (series `outputs/series/wan2.2-ti2v-5b-f16-dyn/` — the same series as the decoder's
chunk graphs):

- `vae_encoder_pre/model.krm` — input `image` `[12, 1, 8h, 8w]` (the image after patchify) →
  `[640, 1, h, w]` (symbols h, w).
- `vae_encoder_attn/model.krm` — input `tokens` `[640, S]` → `[640, S]` (symbol S = h·w).
- `vae_encoder_post/model.krm` — input `hidden` `[640, 1, h, w]` → mu `[48, 1, h, w]` (symbols
  h, w).
- `vae_encoder.<case>.safetensors` (at the series root) — the goldens of the GPU check
  ({@link GOLDEN_CASES}).

The containers are **3, one graph each** (3 components). The container format can carry several
graphs in one container, but manifest `karume/5` does not use that ("one container, one graph;
graph name = component name = the key under weights" — ADR 0109 decision 2, addendum 1), and the
writer (`_graph_descriptor` of `karume.container`) writes only one graph either. The location is
spelled the same as the decoder's (`vae_decoder_first/` / `vae_decoder_next/`): "series root /
component name / model.krm". The weights of the 3 graphs do not overlap (there are no
duplicates).

Storage is **the f16 preset only** (the same as the decoder — decision 7): the weights are rounded
to f16-representable values (fake-quant — ADR 0006) before the 3 graphs and the goldens' reference
are taken (taking the reference before rounding would make the checked difference a mix of
quantization error and implementation error, and the band would lose its meaning).

The golden cases are 3 test images (`inputs/wan-i2v/<name>-832x480.png` — not tracked in git,
sha256 pinned in {@link IMAGES}) × 3 sizes (1280×704, 704×1280 (a landscape image cropped to an
explicitly requested portrait), 256×160 (the downscaling path — latent 10×16)) with fit = `crop`
(the official preprocessing), split into 2 band images (boxing-cats, cat-dog-baking) and 1
acceptance image (ferret) (the same discipline as the e2e of the decoder's chunks — the band is not
changed after looking at the acceptance results). In addition, only the RGB8 of fit = `stretch`
(diffusers' preprocessing) is kept for 1 size (1280×704) (for the TS host test, role `host`).

    uv run --group wan --inexact python -m wan.export_vae_encoder             # 3 graphs + goldens
    uv run --group wan --inexact python -m wan.export_vae_encoder --verify    # eager equivalence

MUST: import diffusers / Pillow inside functions (the `wan` group is not in the default sync —
`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file

from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from karume.container import Provenance, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.emit import storage_breakdown
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from wan import export_vae, i2v_preprocess_ref, vae_encoder_patch
from wan.i2v_preprocess_ref import Fit
from wan.sources import SOURCES

if TYPE_CHECKING:
    from diffusers import AutoencoderKLWan

#: The model (a key of `wan.sources.SOURCES`) and the series (the same series as the decoder — ADR
#: 0121 decision 7).
MODEL = "ti2v-5b"
SERIES_NAME = export_vae.TI2V_SERIES_NAME

#: Target name = component name = the container's graph name = the component directory name.
TARGET_PRE = "vae_encoder_pre"
TARGET_ATTENTION = "vae_encoder_attn"
TARGET_POST = "vae_encoder_post"
TARGETS = (TARGET_PRE, TARGET_ATTENTION, TARGET_POST)

MODEL_FILE = "model.krm"

#: The spelling of the graph inputs (the argument names of each module's forward become the IR
#: input names as is).
INPUT_NAMES = {TARGET_PRE: "image", TARGET_ATTENTION: "tokens", TARGET_POST: "hidden"}

#: The symbol names (the latent height and width for pre / post, the sequence length S = h·w for
#: attn).
LATENT_SYMBOLS = ("h", "w")
SEQUENCE_SYMBOL = "S"

#: The symbol ranges (they affect only the export constraints and do not go into the IR). The lower
#: bound 2 is the line that avoids torch.export's 0 / 1 specialization (the same as the DiT's
#: `Dim("S")`). The upper bound is wider than the accepted set (latent side ≤ 80) — the check on
#: accepted sizes lives on the pipeline side (the asset is independent of the resolution — ADR 0038
#: §4).
LATENT_MIN = 2
LATENT_MAX = 160
SEQUENCE_MIN = 2
SEQUENCE_MAX = LATENT_MAX * LATENT_MAX

#: The spatial ratio of pre's input (the side of the patchified image per latent 1 — 1/8 after 3
#: down stages).
PATCH_SCALE = 8

#: The latent size of the export's example input (landscape 1280×704 — h ≠ w, so a swap of the
#: symbols shows up in the shapes).
EXAMPLE_LATENT = (44, 80)

#: The set of ops allowed in each graph's IR (anything else means a missed rewrite — fail loudly).
#: SiLU decomposes into `x · sigmoid(x)`; the mid attention is SDPA preserved (ADR 0023); pad is
#: only down's ZeroPad2d; slice / cat are the channel pairs of the shortcuts (and the mu extraction
#: and the qkv split).
EXPECTED_OPS: dict[str, frozenset[str]] = {
    TARGET_PRE: frozenset(
        {
            "add",
            "cat",
            "clamp_min",
            "conv2d",
            "conv3d",
            "div",
            "mul",
            "pad",
            "permute",
            "reshape",
            "sigmoid",
            "slice",
            "sqrt",
            "sum",
        }
    ),
    TARGET_ATTENTION: frozenset(
        {
            "add",
            "attention",
            "clamp_min",
            "div",
            "linear",
            "mul",
            "permute",
            "reshape",
            "slice",
            "sqrt",
            "sum",
        }
    ),
    TARGET_POST: frozenset(
        {
            "add",
            "clamp_min",
            "conv3d",
            "div",
            "mul",
            "reshape",
            "sigmoid",
            "slice",
            "sqrt",
            "sum",
        }
    ),
}

#: The upper bound on value rank (the limit of the strided-copy family). Rank 5 is allowed only for
#: conv3d weights.
MAX_VALUE_RANK = 4


class EncoderGraphError(AssertionError):
    """The exported encoder graph departs from the conventions (rank, ops, symbols, input / output
    shapes)."""


# ---- golden cases ----------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenImage:
    """One test image (`inputs/wan-i2v/<name>-832x480.png` — not tracked in git) and its sha256."""

    name: str
    sha256: str

    @property
    def path(self) -> Path:
        return IMAGE_DIR / f"{self.name}-{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}.png"


#: Where the test images live and their size (frozen copies of the first frames of the Wan2.1
#: visual checks — all 3 are 832×480).
IMAGE_DIR = INPUTS_ROOT / "wan-i2v"
IMAGE_SIZE = (832, 480)

#: The test images (the sha256 is the source of truth of the provenance — a mismatch fails loudly).
IMAGES = (
    GoldenImage("boxing-cats", "4689f815503859b4d08b464f21239b0ab5066b4da06e7130aa32a60ef1fe5888"),
    GoldenImage(
        "cat-dog-baking", "75d6d9c514190806baa4fc5a9d0877e04d4c50ba940057d737915bfb354f3c0e"
    ),
    GoldenImage("ferret", "a6b740a2ec936d2a102839441f9caabd787fc1d4e27126bbed4be429aef7bd54"),
)

#: The split into band and acceptance (by image — per size, 2 band cases and 1 acceptance case).
BAND_IMAGES = ("boxing-cats", "cat-dog-baking")
ACCEPT_IMAGES = ("ferret",)

#: The encoder sizes (width, height): the 2 sizes of the accepted set (704×1280 is a landscape image
#: given explicitly as portrait) and the downscaling path (832×480 → 277×160 → 256×160, latent
#: 10×16 — outside the accepted set, but the graphs do not depend on the size).
ENCODER_SIZES = ((1280, 704), (704, 1280), (256, 160))

#: The size at which the fit = `stretch` RGB8 is made (for the TS host test — 1 size).
STRETCH_SIZE = (1280, 704)


@dataclass(frozen=True)
class GoldenCase:
    """One golden (image, output size, fit, role)."""

    image: str
    width: int
    height: int
    fit: Fit
    #: `band` = band case, `accept` = acceptance, `host` = preprocessing only (no GPU run).
    role: str

    @property
    def name(self) -> str:
        suffix = "" if self.fit == "crop" else f"-{self.fit}"
        return f"{self.image}-{self.width}x{self.height}{suffix}"

    @property
    def encodes(self) -> bool:
        """Whether it holds the encoder input and mu (`stretch` has only the preprocessed RGB8)."""
        return self.fit == "crop"


def _golden_cases() -> tuple[GoldenCase, ...]:
    cases = [
        GoldenCase(image, width, height, "crop", "band" if image in BAND_IMAGES else "accept")
        for width, height in ENCODER_SIZES
        for image in (*BAND_IMAGES, *ACCEPT_IMAGES)
    ]
    cases += [GoldenCase(image.name, *STRETCH_SIZE, "stretch", "host") for image in IMAGES]
    return tuple(cases)


GOLDEN_CASES = _golden_cases()

#: Where the goldens live (`vae_encoder.<case>.safetensors` at the series root).
GOLDEN_PREFIX = "vae_encoder."
GOLDEN_SUFFIX = ".safetensors"

#: The goldens' metadata `reference` (the reference's provenance).
GOLDEN_REFERENCE = (
    "diffusers AutoencoderKLWan.encode (non-tiled) latent_dist.mode() (CPU f32), normalized as"
    " WanImageToVideoPipeline.prepare_latents"
)


def golden_path(out_root: Path, case: GoldenCase) -> Path:
    return out_root / f"{GOLDEN_PREFIX}{case.name}{GOLDEN_SUFFIX}"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_image(image: GoldenImage) -> Any:
    """Opens a test image as RGB (sha256 and size that differ from the provenance fail loudly)."""
    if not image.path.is_file():
        raise FileNotFoundError(
            f"test image {image.path} is missing (inputs/ is not tracked in git — put the frozen"
            " copy there)"
        )
    digest = _sha256(image.path)
    if digest != image.sha256:
        raise EncoderGraphError(
            f"{image.path}: sha256 {digest} differs from the provenance {image.sha256}"
        )
    opened = i2v_preprocess_ref.load_rgb(image.path)
    if opened.size != IMAGE_SIZE:
        raise EncoderGraphError(f"{image.path}: size {opened.size} is not {IMAGE_SIZE}")
    return opened


def library_versions() -> dict[str, str]:
    """The versions of the libraries that determine the values (the goldens' provenance — Pillow
    determines the LANCZOS implementation)."""
    return {name: version(name) for name in ("pillow", "torch", "torchvision", "diffusers")}


# ---- graphs --------------------------------------------------------------------------


def _hidden_channels(vae: AutoencoderKLWan) -> int:
    """The channel count of the mid block (`base_dim × dim_mult[-1]` — 640 for 5B)."""
    return int(vae.config.base_dim) * int(vae.config.dim_mult[-1])


def example_inputs(target: str, vae: AutoencoderKLWan) -> tuple[torch.Tensor]:
    """The export's example input (the values only affect the shapes)."""
    height, width = EXAMPLE_LATENT
    hidden = _hidden_channels(vae)
    if target == TARGET_PRE:
        channels = int(vae.encoder.conv_in.in_channels)
        return (torch.zeros(channels, 1, PATCH_SCALE * height, PATCH_SCALE * width),)
    if target == TARGET_ATTENTION:
        return (torch.zeros(hidden, height * width),)
    if target == TARGET_POST:
        return (torch.zeros(hidden, 1, height, width),)
    raise EncoderGraphError(f"unknown target {target}")


def dynamic_shapes(target: str) -> tuple[Any, ...]:
    """The symbol declarations (pre is `8h` / `8w`, attn is `S`, post is `h` / `w`)."""
    from torch.export import Dim

    if target == TARGET_ATTENTION:
        return ({1: Dim(SEQUENCE_SYMBOL, min=SEQUENCE_MIN, max=SEQUENCE_MAX)},)
    height = Dim(LATENT_SYMBOLS[0], min=LATENT_MIN, max=LATENT_MAX)
    width = Dim(LATENT_SYMBOLS[1], min=LATENT_MIN, max=LATENT_MAX)
    scale = PATCH_SCALE if target == TARGET_PRE else 1
    return ({2: scale * height, 3: scale * width},)


def symbol_names(target: str) -> tuple[str, ...]:
    return (SEQUENCE_SYMBOL,) if target == TARGET_ATTENTION else LATENT_SYMBOLS


def expected_io(target: str, vae: AutoencoderKLWan) -> tuple[list[Any], list[Any]]:
    """The declared shapes of the graph's input and output (symbols in the IR spelling `8h` / `h` /
    `S`)."""
    hidden = _hidden_channels(vae)
    latent = [hidden, 1, "h", "w"]
    if target == TARGET_PRE:
        channels = int(vae.encoder.conv_in.in_channels)
        return [channels, 1, f"{PATCH_SCALE}h", f"{PATCH_SCALE}w"], latent
    if target == TARGET_ATTENTION:
        return [hidden, "S"], [hidden, "S"]
    if target == TARGET_POST:
        return latent, [int(vae.config.z_dim), 1, "h", "w"]
    raise EncoderGraphError(f"unknown target {target}")


def assert_encoder_graph(graph: IrGraph, target: str, vae: AutoencoderKLWan) -> None:
    """Checks that the encoder graph's IR meets the conventions (fails loudly otherwise).

    - ops only from that graph's set in {@link EXPECTED_OPS}.
    - values of rank 4 or less. Rank 5 only for conv3d weights (initializer, second input).
    - the symbols are as declared (pre / post `[h, w]`, attn `[S]`), with 1 input (name and shape)
      and 1 output (shape). The product of the symbols (`h·w`) does not go into the IR's dimension
      expressions — if it did, the export would already have failed.
    """
    ops = Counter(node.op for node in graph.nodes)
    unexpected = sorted(set(ops) - EXPECTED_OPS[target])
    if unexpected:
        raise EncoderGraphError(f"{target}: unexpected ops {unexpected} (a missed rewrite)")

    conv3d_weights = {node.ins[1] for node in graph.nodes if node.op == "conv3d"}
    for name, value in graph.values.items():
        rank = len(value.shape)
        if rank <= MAX_VALUE_RANK:
            continue
        if name not in graph.initializers or name not in conv3d_weights or rank != 5:
            raise EncoderGraphError(
                f"{target}: rank {rank} value '{name}' (rank 5 only for conv3d weights)"
            )

    expected_symbols = list(symbol_names(target))
    if list(graph.symbols) != expected_symbols:
        raise EncoderGraphError(f"{target}: symbols {graph.symbols} are not {expected_symbols}")
    expected_input, expected_output = expected_io(target, vae)
    actual_inputs = [(entry.name, list(entry.shape)) for entry in graph.inputs]
    if actual_inputs != [(INPUT_NAMES[target], expected_input)]:
        raise EncoderGraphError(
            f"{target}: inputs {actual_inputs} are not"
            f" [({INPUT_NAMES[target]!r}, {expected_input})]"
        )
    if len(graph.outputs) != 1:
        raise EncoderGraphError(f"{target}: {len(graph.outputs)} outputs (expected 1)")
    output_shape = list(graph.values[graph.outputs[0]].shape)
    if output_shape != expected_output:
        raise EncoderGraphError(f"{target}: output shape {output_shape} is not {expected_output}")


def graph_module(target: str, vae: AutoencoderKLWan) -> torch.nn.Module:
    """The target's module (eval — the weights are a copy of the current VAE values)."""
    builders = {
        TARGET_PRE: vae_encoder_patch.VaeEncoderPre,
        TARGET_ATTENTION: vae_encoder_patch.VaeEncoderAttention,
        TARGET_POST: vae_encoder_patch.VaeEncoderPost,
    }
    if target not in builders:
        raise EncoderGraphError(f"unknown target {target} (known: {', '.join(TARGETS)})")
    return builders[target](vae).eval()


def weight_summary(vae: AutoencoderKLWan) -> dict[str, Any]:
    """Element count, max absolute value and non-finite count of the encoder + quant_conv weights
    (for the summary record)."""
    tensors = [*vae.encoder.parameters(), *vae.quant_conv.parameters()]
    with torch.no_grad():
        return {
            "elements": sum(tensor.numel() for tensor in tensors),
            "abs_max": max(float(tensor.abs().max()) for tensor in tensors),
            "nonfinite": sum(int((~torch.isfinite(tensor)).sum()) for tensor in tensors),
        }


def _peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def _graph_summary(
    target: str, out_dir: Path, staged: Path, graph: IrGraph, started: float
) -> dict[str, Any]:
    breakdown = storage_breakdown(graph)
    return {
        "target": target,
        "dir": str(out_dir),
        "symbols": list(graph.symbols),
        "inputs": [[entry.name, list(entry.shape)] for entry in graph.inputs],
        "output": list(graph.values[graph.outputs[0]].shape),
        "nodes": len(graph.nodes),
        "op_counts": dict(sorted(Counter(node.op for node in graph.nodes).items())),
        "initializers": len(graph.initializers),
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
        "model_bytes": sum(part.stat().st_size for part in container_parts(staged / MODEL_FILE)),
        "seconds": round(time.perf_counter() - started, 1),
        "peak_rss_gib": round(_peak_rss_gib(), 2),
    }


# ---- goldens -------------------------------------------------------------------------


def golden_tensors(
    vae: AutoencoderKLWan,
    graphs: tuple[torch.nn.Module, ...] | None,
    case: GoldenCase,
    source: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """The contents of one golden, and the eager difference of the rewrite (only when `graphs` is
    given — for the record).

    Tensors (`stretch` has only `source` and `rgb8`):

    - `source` `[480, 832, 3]` u8 — the pixels of the test image (the very image of the provenance
      sha256 — lets the TS host test run the preprocessing chain without decoding a PNG).
    - `rgb8` `[H, W, 3]` u8 — the output of the fit's preprocessing.
    - `encoder_input` `[12, 1, H/2, W/2]` f32 — after [-1, 1] and patchify (the input of the pre
      graph).
    - `mu` `[48, 1, h, w]` f32 — the mu of upstream's non-tiled encode (the reference for the post
      graph's output).
    - `latent` `[48, 1, h, w]` f32 — the normalized mu `(mu − mean)·f32(1/f32(std))` (the condition
      passed to the DiT).
    """
    resized = i2v_preprocess_ref.preprocess(source, case.width, case.height, case.fit)
    tensors = {
        "source": i2v_preprocess_ref.rgb8(source),
        "rgb8": i2v_preprocess_ref.rgb8(resized),
    }
    differences: dict[str, float] = {}
    if not case.encodes:
        return tensors, differences
    sample = i2v_preprocess_ref.to_signed_unit(resized)
    image = i2v_preprocess_ref.encoder_input(sample, int(vae.config.patch_size))
    with torch.no_grad():
        mu = vae_encoder_patch.reference_mu(vae, sample)
        latent = vae_encoder_patch.normalize_condition(vae, mu)
    tensors |= {
        "encoder_input": image,
        "mu": mu[0].contiguous(),
        "latent": latent[0].contiguous(),
    }
    if graphs is not None:
        with torch.no_grad():
            eager = vae_encoder_patch.encode_chunk0(graphs, image)
        difference = float((eager - mu[0]).abs().max())
        differences = {
            "eager_max_abs": difference,
            "eager_ratio": difference / float(mu.abs().max()),
        }
    return tensors, differences


def golden_metadata(
    case: GoldenCase, image: GoldenImage, versions: dict[str, str]
) -> dict[str, str]:
    metadata = {
        "image": image.name,
        "image_sha256": image.sha256,
        "width": str(case.width),
        "height": str(case.height),
        "fit": case.fit,
        "role": case.role,
        "resample": "LANCZOS",
        "upstream_revision": SOURCES[MODEL].revision,
        **{f"version_{name}": value for name, value in versions.items()},
    }
    if case.encodes:
        metadata |= {"weights": "f16-rounded", "reference": GOLDEN_REFERENCE}
    return metadata


def emit_series(
    targets: Sequence[str],
    vae: AutoencoderKLWan,
    out_root: Path,
    source: Provenance,
    *,
    cases: Sequence[GoldenCase],
) -> dict[str, Any]:
    """Writes the graphs and the goldens (`cases` — none when empty) as one set into staging
    directories, and publishes them after every check has passed.

    MUST: publish as one set (for the same reason as the decoder's `emit_targets` — if a run that
    failed midway left a mixed set of graphs and goldens of different rounding generations in the
    series, nobody could tell them apart, since the shapes are the same).
    """
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown or len(set(targets)) != len(targets):
        raise EncoderGraphError(
            f"targets {list(targets)} hold an unknown or duplicate entry"
            f" (known: {', '.join(TARGETS)})"
        )
    images = {image.name: image for image in IMAGES}
    sources = {name: load_image(images[name]) for name in sorted({case.image for case in cases})}
    versions = library_versions()
    out_root.mkdir(parents=True, exist_ok=True)
    finals = [out_root / target for target in targets]
    finals += [golden_path(out_root, case) for case in cases]
    summaries: list[dict[str, Any]] = []
    written: list[dict[str, Any]] = []
    with export_vae._staged_set(finals) as seats:
        for target, staged in zip(targets, seats[: len(targets)], strict=True):
            started = time.perf_counter()
            module = graph_module(target, vae)
            staged.mkdir()
            with torch.no_grad():
                graph = export_to_file(
                    module,
                    example_inputs(target, vae),
                    staged / MODEL_FILE,
                    provenance=source,
                    graph_name=target,
                    dynamic_shapes=dynamic_shapes(target),
                    symbol_names=symbol_names(target),
                    weight_dtype="f16",
                    preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
                )
            assert_encoder_graph(graph, target, vae)
            summaries.append(_graph_summary(target, out_root / target, staged, graph, started))
        graphs = vae_encoder_patch.encoder_graphs(vae) if cases else None
        for case, staged in zip(cases, seats[len(targets) :], strict=True):
            started = time.perf_counter()
            tensors, differences = golden_tensors(vae, graphs, case, sources[case.image])
            save_file(
                tensors,
                str(staged),
                metadata=golden_metadata(case, images[case.image], versions),
            )
            written.append(
                {
                    "case": case.name,
                    "role": case.role,
                    "fit": case.fit,
                    "shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
                    **differences,
                    "seconds": round(time.perf_counter() - started, 1),
                    "path": str(golden_path(out_root, case)),
                }
            )
    summary: dict[str, Any] = {"series": str(out_root), "graphs": summaries}
    if cases:
        summary["goldens"] = written
        summary["versions"] = versions
    return summary


# ---- measuring the eager equivalence -------------------------------------------------


def verify(cases: Sequence[GoldenCase]) -> dict[str, Any]:
    """Measures the eager equivalence of the rewrite on the real weights (f32, no rounding) (test
    images × sizes).

    - the difference between the mu of the 3 chained graphs and the mu of upstream's non-tiled
      encode (max absolute value, ratio to the reference's max absolute value)
    - the time taken by the reference and by the rewrite (CPU)
    """
    vae = export_vae.load_vae(MODEL, round_f16=False)
    export_vae.assert_series_config(vae, export_vae.VAE_SERIES[MODEL])
    graphs = vae_encoder_patch.encoder_graphs(vae)
    images = {image.name: image for image in IMAGES}
    rows: list[dict[str, Any]] = []
    for case in cases:
        if not case.encodes:
            continue
        resized = i2v_preprocess_ref.preprocess(
            load_image(images[case.image]), case.width, case.height, case.fit
        )
        sample = i2v_preprocess_ref.to_signed_unit(resized)
        image = i2v_preprocess_ref.encoder_input(sample, int(vae.config.patch_size))
        with torch.no_grad():
            started = time.perf_counter()
            reference = vae_encoder_patch.reference_mu(vae, sample)[0]
            middle = time.perf_counter()
            eager = vae_encoder_patch.encode_chunk0(graphs, image)
            finished = time.perf_counter()
        difference = float((eager - reference).abs().max())
        rows.append(
            {
                "case": case.name,
                "mu_shape": list(reference.shape),
                "reference_abs_max": float(reference.abs().max()),
                "bit_exact": torch.equal(eager, reference),
                "max_abs": difference,
                "ratio": difference / float(reference.abs().max()),
                "reference_seconds": round(middle - started, 1),
                "rewrite_seconds": round(finished - middle, 1),
            }
        )
    return {
        "threads": torch.get_num_threads(),
        "cases": rows,
        "worst_ratio": max(row["ratio"] for row in rows),
        "peak_rss_gib": round(_peak_rss_gib(), 2),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--out", type=Path, default=None, help=f"series root (default {SERIES_ROOT / SERIES_NAME})"
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=TARGETS,
        default=None,
        help="graphs to write (default: all 3)",
    )
    parser.add_argument("--no-goldens", action="store_true", help="do not write the goldens")
    parser.add_argument(
        "--verify", action="store_true", help="only measure the eager equivalence (writes nothing)"
    )
    args = parser.parse_args(argv)

    if args.verify:
        print(json.dumps(verify(GOLDEN_CASES), indent=1, ensure_ascii=False))
        return 0

    source = export_vae.provenance(MODEL)
    vae = export_vae.load_vae(MODEL, round_f16=True)
    export_vae.assert_series_config(vae, export_vae.VAE_SERIES[MODEL])
    summary = emit_series(
        args.target or TARGETS,
        vae,
        args.out or SERIES_ROOT / SERIES_NAME,
        source,
        cases=() if args.no_goldens else GOLDEN_CASES,
    )
    summary["weights"] = weight_summary(vae)
    summary["peak_rss_gib"] = round(_peak_rss_gib(), 2)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
