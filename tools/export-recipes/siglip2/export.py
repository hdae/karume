"""Script that writes the real-weight SigLIP2 **vision tower** to a container (`krm`) + golden io.

It plays the role of `embeddinggemma/export.py` (real weights, numeric match of a single vector
output) on the image side. The output goes under `outputs/series/` and is kept out of commits by
the `.gitignore` at the repository root (weights of 350MB to 1.7GB).

    uv run --group siglip2 python -m siglip2.export
    uv run --group siglip2 python -m siglip2.export --verify  # equivalence with pre-patch eager

transformers is **pinned at 5.14.1** (`siglip2.patch` replaces class attributes of the modeling
code — the `siglip2` group of pyproject.toml).

## Model axis

The target is the single axis `--model-dir`, and the series name is derived from that directory's
name (the same arrangement as `sbv2/export.py` / `irodori/export.py`) — with a fixed output
directory, exporting another model would silently overwrite the earlier series.

    uv run --group siglip2 python -m siglip2.export \
        --model-dir ../../inputs/siglip2/siglip2-so400m-patch14-384

The differences in scale (hidden / layer count / head count / intermediate / patch / resolution)
**all come from config.json**; the script hard-codes none of them. The dimensions of the synthetic
images are also built from `config.image_size` / `config.patch_size` ({@link build_cases}).

## What goes into the graph

**Only the pooler_output** of `SiglipVisionModel` (`[B, hidden]` after the MAP head).
`last_hidden_state` is not emitted — its readback, `[B, patch count, hidden]`, is two orders of
magnitude larger and is not used for embeddings (the same call as "narrow the outputs" in
ADR 0026). The text tower is not included.

batch is **static 1**. On the image side both the resolution and the sequence length (= patch count)
are fixed, so no axis varies.

MUST: SDPA is **not preserved** (the default 11 `PRESERVED_OP_PREFIXES` stay as they are). The
attention of every encoder layer and of the MAP head is decomposed into mul×2 + bmm + softmax + bmm
(12 + 1 for base / 27 + 1 for so400m). This series does not opt in to the fused attention of
ADR 0023.

MUST: the model is exported after applying the two-step patch of `siglip2.patch` (without it, the
i64 initializers of `aten.conv2d.padding` / `aten.select.int` / `aten.embedding` remain). The golden
expected values are taken from the module **after the replacement** — the difference before and
after the replacement is the patch layer's responsibility, and `--verify` measures it at two points
(point 1 = the shape folds only, where bit-identity is the claim itself; point 2 = up to the MAP
head, reporting the maxdiff).

MUST: `--verify` does not emit (the two cannot be combined in one process). The patch replaces
class attributes process-wide, so once it has been applied for the emit, "the reference before
patching" can no longer be taken.

## Input contract (preprocessing is not in the graph)

The graph input is the **normalized** `pixel_values f32 [1, 3, resolution, resolution]` (base 224 /
so400m 384). Decoding the image is the host's job; resize (bilinear to `config.image_size`) /
rescale (1/255) / normalize (mean = std = 0.5 → `[-1, 1]`) are on the karume side
(`packages/models/src/image/preprocess.ts`; the parity script is `siglip2/preprocess.py`).
The source of truth for the constants is `preprocessor_config.json` next to the weights (mean =
std = 0.5 for both models, **resample 2 = PIL's BILINEAR** (BICUBIC is 3); only the dimensions
differ).

## Two groups of goldens (synthetic images + real images)

The default 4 cases are **synthetic images** ({@link build_cases}); they hit the ends of the value
range and gradients, so they detect numeric regressions sharply. With `--real-images`, goldens of
4 **real images** from `outputs/misc/corpus/` ({@link REAL_CASES}) are written as well. The real
images hit only moderate values, but in exchange they compare **the chain including
preprocessing** (on the TS side PNG → `preprocess.ts` → karume inference, on this side PIL →
`TorchvisionBackend` → torch inference). Both are kept (neither replaces the other).

    uv run --group siglip2-preprocess python -m siglip2.export --real-images

MUST: run `--real-images` with the **`siglip2-preprocess` group**. The source of truth for the
real images' preprocessing is the fast side of `AutoImageProcessor` (`TorchvisionBackend`), which
needs torchvision and pillow (the `siglip2` group does not have those two, which the export does not
need — pyproject.toml).

MUST: real-image goldens carry **the source image's sha256** in `__metadata__`. The images are
output baked by `examples/anima/eval-images.ts` and **frozen by hand as copies** in
`outputs/misc/corpus/`, so in an environment where the corpus was replaced but the goldens were not
re-taken, the comparison is not a comparison: **the inputs differ** (and the values look plausible
at first sight).

## Output layout

The series name is the directory name of `--model-dir` (`siglip2-base-patch16-224` for the default
`inputs/siglip2/siglip2-base-patch16-224/`):

    outputs/series/<series name>/model.krm             weights, constants + the two descriptors
    outputs/series/<series name>/io.<case>.safetensors inputs and torch CPU expected outputs

The tensor key convention of io is the same as tiny golden / DeBERTa / EmbeddingGemma
(`input.<graph input name>` / `output.<position>`).
"""

from __future__ import annotations

import argparse
import hashlib
import io as io_module
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn

from _shared.paths import INPUTS_ROOT, MISC_ROOT, SERIES_ROOT
from _shared.upstream import snapshot_provenance
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import normalize_boundary_tensor
from karume.dist import NOTICE_FILENAME
from karume.ir import IrGraph
from karume.pipeline import export_to_file

from . import patch
from .distribution import SIGLIP2_ROLE

#: Parent of the real weights (where `hf download google/<name> --local-dir inputs/siglip2/<name>`
#: unpacks).
MODELS_ROOT = INPUTS_ROOT / "siglip2"

#: Location of the real-image corpus (a frozen copy = a host asset, so deleting it means re-baking —
#: docs/assets-layout.md).
CORPUS_ROOT = MISC_ROOT / "corpus"

#: Default model (when `--model-dir` is not given).
DEFAULT_MODEL_DIR = MODELS_ROOT / "siglip2-base-patch16-224"


def default_out_dir(model_dir: Path) -> Path:
    """Default location of the output (`outputs/series/<directory name of the real weights>/`).

    The model name goes into the series spelling for the same reason as `default_out_root` in
    `sbv2/export.py` — with a shared fixed spelling, exporting another model would silently
    overwrite the earlier assets.

    The storage dtype is f32 only, so there is no per-dtype branch (f16 / i8 would be decided as
    separate series).
    """
    return SERIES_ROOT / model_dir.name


def upstream_provenance(model_dir: Path) -> Provenance:
    """Derive the provenance baked into the container (container-v1 §2.3) from the actual
    `--model-dir`.

    The license identifier comes from the README front matter, the revision from the download
    record of `hf download --local-dir` ({@link _shared.upstream.snapshot_provenance}). Series
    migration (`migrate_series`) calls the same function.

    It is not a constant because then a container baked from another checkpoint would also claim
    the attribution table's license. The comparison with the attribution table (the card side's
    source of truth) belongs to the assembly side (`siglip2.distribution`).
    """
    return snapshot_provenance(model_dir, notice=NOTICE_FILENAME)


MODEL_FILE = "model.krm"

IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: Name of the graph input (= the parameter name of `VisionPooler.forward`). It is compared after
#: the export and stops on a mismatch — inputs are passed by position, so a silent shift would swap
#: only the golden inputs.
INPUT_NAME = "pixel_values"

#: Random seed of the synthetic images (the `noise` case). Independent of the global seed.
SEED = 20260813

#: Per-channel brightness offsets. If the 3 channels were identical planes, a channel mix-up (a
#: wrong weight layout of the patch embedding) would not show in the values.
CHANNEL_OFFSETS: tuple[float, ...] = (-0.2, 0.0, 0.2)

#: Contrast ratio of `ramp-dim` (makes a near pair with `ramp` that **has the same structure and
#: differs only in intensity**).
DIM_SCALE = 0.5

#: The cosine pairs the sanity check looks at (near pair, far pair). If the near pair does not
#: exceed the far pair, "the numbers move but the image embedding is broken" — fail loudly at
#: generation time.
NEAR_PAIR = ("ramp", "ramp-dim")
FAR_PAIR = ("ramp", "checker")

#: Real-image cases = (case name, file name in `outputs/misc/corpus/`, content). The images were
#: baked by `examples/anima/eval-images.ts` (`deno task demo:eval-images`, which held the
#: prompts / seeds / resolution — removed in 0.15 with no replacement), and the chosen ones were
#: frozen by hand as copies.
REAL_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "photo-portrait",
        "anima-default-1024x1024-defaultstep-seed42.png",
        "close-up of a person + a row of cherry trees (clear foreground and background)",
    ),
    (
        "photo-landscape",
        "anima-default-1024x1024-defaultstep-seed43.png",
        "landscape (mountains, lake, forest, no people)",
    ),
    (
        "photo-corridor",
        "anima-default-1024x1024-defaultstep-seed44.png",
        "school corridor (strong perspective, vanishing point)",
    ),
    (
        "photo-street",
        "anima-default-1024x1024-defaultstep-seed45.png",
        "full-body person + a street (busy background)",
    ),
)

#: The two groups of the real-image discrimination check (2 images with people / 2 without). It
#: checks that **the cosine between the two people is higher than any of the 4 person × scene
#: pairs** (a more meaningful discrimination than the synthetic NEAR/FAR — no threshold, only the
#: order itself). The TS-side `packages/runtime/tests/e2e_siglip2_test.ts` applies the same groups
#: to real GPU output.
REAL_PERSON_CASES = ("photo-portrait", "photo-street")
REAL_SCENE_CASES = ("photo-landscape", "photo-corridor")

#: Fields of the real-image golden's `__metadata__` (identifying the source image — the MUST of the
#: module docstring).
SOURCE_IMAGE_KEY = "source_image"
SOURCE_SHA256_KEY = "source_sha256"


class VisionPooler(nn.Module):
    """Export wrapper that returns only the pooler_output (`[B, hidden]` through the MAP head)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).pooler_output


def _stack_channels(plane: torch.Tensor, channels: int) -> torch.Tensor:
    """Turn one `[S, S]` plane into `[1, C, S, S]`, offset per channel."""
    offsets = torch.tensor(CHANNEL_OFFSETS[:channels]).reshape(channels, 1, 1)
    return (plane.unsqueeze(0) + offsets).clamp(-1.0, 1.0).unsqueeze(0)


def _ramp_plane(size: int) -> torch.Tensor:
    """Diagonal ramp from top left to bottom right (`[S, S]`, value range `[-1, 1]`)."""
    axis = torch.linspace(-1.0, 1.0, size)
    return (axis.reshape(1, size) + axis.reshape(size, 1)) / 2.0


def _checker_plane(size: int, cell: int) -> torch.Tensor:
    """Checkerboard aligned to the patch grid (`[S, S]`, values ±1)."""
    index = torch.arange(size) // cell
    parity = (index.reshape(1, size) + index.reshape(size, 1)) % 2
    return parity.to(torch.float32) * 2.0 - 1.0


def build_cases(config: Any) -> tuple[tuple[str, torch.Tensor], ...]:
    """`(name, pixel_values)` of the 4 golden cases (**normalized** synthetic images).

    The 4 synthetic cases are for sensitivity to numeric regressions (they hit the ends of the value
    range and gradients). `--real-images` adds the real images. Pairs that differ only in structure
    keep the cosine order of {@link _sanity} from being vacuous: `ramp` and `ramp-dim` have the
    same structure and differ only in intensity, while `checker` differs in structure altogether.
    """
    size = int(config.image_size)
    channels = int(config.num_channels)
    ramp = _ramp_plane(size)
    generator = torch.Generator().manual_seed(SEED)
    noise = torch.rand((size, size), generator=generator) * 2.0 - 1.0
    return (
        ("ramp", _stack_channels(ramp, channels)),
        ("ramp-dim", _stack_channels(ramp * DIM_SCALE, channels)),
        ("checker", _stack_channels(_checker_plane(size, int(config.patch_size)), channels)),
        ("noise", _stack_channels(noise, channels)),
    )


def load_image_processor(model_dir: Path) -> Any:
    """Borrow the image processor for the real images' preprocessing, together with its type
    checks, **from the source of truth**.

    MUST: do not rewrite the checks (`siglip2/preprocess.py` is the source of truth for
    preprocessing, and the TS-side parity check is closed against its fixture) — with a second copy
    of the checks, fixing only one would make them diverge silently. The import is inside the
    function because that module takes its default paths from this script, and importing it at the
    top of the module would be circular.
    """
    from .preprocess import check_processor_shape, load_processor

    processor = load_processor(model_dir)
    check_processor_shape(processor)
    return processor


def build_real_cases(
    model_dir: Path, config: Any, corpus_root: Path
) -> tuple[tuple[str, torch.Tensor, dict[str, str]], ...]:
    """`(name, pixel_values, __metadata__)` of the 4 real images ({@link REAL_CASES}).

    Preprocessing goes through **the same path as the TS side** (the fast side of
    `AutoImageProcessor` = `TorchvisionBackend`). What this golden claims is a comparison of "TS
    preprocessing + karume inference" against "Python preprocessing + torch inference", so using a
    different resize here would void the claim.

    MUST: if even one image is missing, stop **before** the heavy imports (`--real-images` is an
    explicit request, so silently writing 3 images would leave the missing golden unnoticed).
    """
    paths: list[tuple[str, str, Path]] = []
    for name, file_name, _why in REAL_CASES:
        path = corpus_root / file_name
        if not path.is_file():
            raise SystemExit(
                f"real image {path} is missing (bake it: deno task demo:eval-images"
                " --source <path of the Anima distribution> → freeze a copy of the chosen ones"
                " in outputs/misc/corpus/)"
            )
        paths.append((name, file_name, path))

    from PIL import Image

    processor = load_image_processor(model_dir)
    expected_shape = (1, int(config.num_channels), int(config.image_size), int(config.image_size))
    cases: list[tuple[str, torch.Tensor, dict[str, str]]] = []
    for name, file_name, path in paths:
        raw = path.read_bytes()
        with Image.open(io_module.BytesIO(raw)) as image:
            array = np.array(image.convert("RGB"))
        pixel_values = processor(
            images=array, input_data_format="channels_last", return_tensors="pt"
        )["pixel_values"]
        # MUST: check that the preprocessing dimensions match the vision config
        # (`preprocessor_config.json` and `config.json` are separate files; if they disagree,
        # only the golden is baked at another resolution).
        if tuple(pixel_values.shape) != expected_shape:
            raise AssertionError(
                f"the pixel_values of {name} are {tuple(pixel_values.shape)}"
                f" (differs from {expected_shape}, which comes from the config)"
            )
        cases.append(
            (
                name,
                pixel_values,
                {
                    SOURCE_IMAGE_KEY: file_name,
                    SOURCE_SHA256_KEY: hashlib.sha256(raw).hexdigest(),
                },
            )
        )
    return tuple(cases)


def load_model(model_dir: Path) -> nn.Module:
    """Load the real weights of the vision tower and check the config the replacements assume
    (before patching).

    MUST: `attn_implementation="sdpa"` — makes the attention of the 12 encoder layers come out as
    SDPA (the default decomposition table expands it into mul×2 + bmm + softmax + bmm).
    """
    from transformers import SiglipVisionModel

    model = SiglipVisionModel.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="sdpa"
    )
    model.eval()
    patch.assert_supported(model.config)
    return model


def load_wrapper(model_dir: Path) -> VisionPooler:
    """Return an exportable wrapper with both patch steps applied.

    MUST: the replacement comes **before** the goldens are taken (the `siglip2.patch` docstring) —
    applied afterwards, only the expected values would be computed on the original path and the
    run would go green while disagreeing with the graph.
    """
    model = load_model(model_dir)
    patch.apply_shape_patches(model.embeddings)
    patch.apply_map_head_patch(model.head)
    return VisionPooler(model).eval()


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
    metadata: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Write each case's input and torch CPU expected output to `io.<case>.safetensors`.

    `metadata` maps case name → `__metadata__` (only real-image cases have one — identifying the
    source image).

    The second return value is the expected outputs for the sanity record (passed in their
    `[B, H]` shape).
    """
    if len(graph.outputs) != 1:
        raise AssertionError(
            f"the IR has {len(graph.outputs)} outputs (pooler_output is exactly one)"
        )
    written: list[str] = []
    pooled: dict[str, torch.Tensor] = {}
    for name, pixel_values in cases:
        with torch.no_grad():
            output = wrapper(pixel_values)
        # MUST: io is also lowered to the concrete representation of the IR's semantic dtypes
        # (the boundary normalization of ADR 0009). Unless it matches what the runtime receives,
        # the Deno-side E2E cannot read the golden.
        tensors = {
            f"{INPUT_PREFIX}{INPUT_NAME}": normalize_boundary_tensor(
                pixel_values, f"{name}: input '{INPUT_NAME}'"
            ),
            f"{OUTPUT_PREFIX}0": normalize_boundary_tensor(
                output.detach().contiguous(), f"{name}: output 0"
            ),
        }
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        entry = None if metadata is None else metadata.get(name)
        save_file(tensors, str(path), metadata=None if entry is None else dict(entry))
        written.append(path.name)
        pooled[name] = output.detach()
    return written, pooled


def _sanity(pooled: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Check that the cosine of the pair with similar structure exceeds that of the pair whose
    structure differs.

    MUST: do not make the sanity check vacuous. pooler_output is not L2-normalized, so a norm check
    is useless, and "the cosine of an input with itself = 1" verifies nothing. Checking the order
    between **different inputs** fails both when the embedding collapses to one point (cosine 1 for
    the near and the far pair alike) and when the patch embedding or the MAP head is dead.
    """
    vectors = {name: output.reshape(-1) for name, output in pooled.items()}

    def cosine(pair: tuple[str, str]) -> float:
        first, second = vectors[pair[0]], vectors[pair[1]]
        return float(torch.dot(first, second) / (first.norm() * second.norm()))

    near, far = cosine(NEAR_PAIR), cosine(FAR_PAIR)
    if near <= far:
        raise AssertionError(
            f"the cosine order contradicts the structure: {NEAR_PAIR}={near:.4f}"
            f" <= {FAR_PAIR}={far:.4f}"
        )
    report: dict[str, Any] = {
        "l2_norms": {name: round(float(vector.norm()), 5) for name, vector in vectors.items()},
        "cosine": {
            f"{NEAR_PAIR[0]}×{NEAR_PAIR[1]}": round(near, 4),
            f"{FAR_PAIR[0]}×{FAR_PAIR[1]}": round(far, 4),
            "ramp×noise": round(cosine(("ramp", "noise")), 4),
        },
    }
    if set(REAL_PERSON_CASES) | set(REAL_SCENE_CASES) <= set(vectors):
        report["real_cosine"] = _real_sanity(cosine)
    return report


def _real_sanity(cosine: Callable[[tuple[str, str]], float]) -> dict[str, float]:
    """Check the real-image discrimination (person × person > person × scene)
    ({@link REAL_PERSON_CASES}).

    MUST: check the **order**, with no threshold (the same approach as the synthetic side). The two
    person images differ in composition (close-up / full body) and in background (cherry trees /
    street), so the order holds only when the embedding captures the meaning "a person is in the
    picture".
    """
    person = cosine(REAL_PERSON_CASES)
    cross = {
        f"{first}×{second}": cosine((first, second))
        for first in REAL_PERSON_CASES
        for second in REAL_SCENE_CASES
    }
    worst_name, worst = max(cross.items(), key=lambda item: item[1])
    if person <= worst:
        raise AssertionError(
            f"the real-image cosine order is inverted: {REAL_PERSON_CASES}={person:.4f}"
            f" <= {worst_name}={worst:.4f}"
        )
    return {
        f"{REAL_PERSON_CASES[0]}×{REAL_PERSON_CASES[1]}": round(person, 4),
        **{name: round(value, 4) for name, value in cross.items()},
    }


def export_series(
    model_dir: Path, out_dir: Path, real_images: bool = False, corpus_root: Path = CORPUS_ROOT
) -> dict[str, Any]:
    """Write the IR container and the golden io, and return a summary.

    Setting `real_images` **adds** the 4 real-image cases to the 4 synthetic ones (it does not
    replace them — "Two groups of goldens" in the module docstring).

    MUST: write the output to a staging directory and put it in place only after **every check**
    (input order, sanity) has passed. Putting it in the final location before the checks would let
    a failed run leave behind "assets that pass the acceptance checks" — the io goldens are taken
    from the same broken wrapper, so they agree with each other and the TS-side comparison **goes
    green** (the rule for "when it may be published" is {@link _shared.decode_series._publish};
    the discipline of swapping in and cleaning up is the core primitive
    {@link karume.artifacts.staged_publication}).
    """
    # Read the provenance before the weights (a checkpoint without a record fails before its
    # weights are read).
    provenance = upstream_provenance(model_dir)
    wrapper = load_wrapper(model_dir)
    synthetic = build_cases(wrapper.model.config)
    # MUST: build the real images before the emit (if an image is missing, stop here instead of
    # failing after writing 1.7GB).
    real = build_real_cases(model_dir, wrapper.model.config, corpus_root) if real_images else ()
    cases = [*synthetic, *((name, pixel_values) for name, pixel_values, _md in real)]
    metadata = {name: md for name, _pixel_values, md in real}
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    _, example = synthetic[0]
    with staged_publication(out_dir) as staged:
        # The writer creates the directory (the primitive does not create it — it only hands
        # over the path).
        staged.mkdir()
        # No dynamic axes (the resolution and the patch count are fixed — module docstring).
        graph = export_to_file(
            wrapper,
            (example,),
            staged / MODEL_FILE,
            provenance=provenance,
            # The graph name is the **component name** (= the weights key of karume.json). It is
            # not derived from the directory name — this family's container sits directly in the
            # series, so the directory name is the series name (`siglip2-so400m-patch14-384`),
            # not the component name (container-v1 §2.1).
            graph_name=SIGLIP2_ROLE,
            symbol_names=(),
        )
        declared = tuple(item.name for item in graph.inputs)
        if declared != (INPUT_NAME,):
            raise AssertionError(
                f"the graph inputs are {declared}, not the expected {(INPUT_NAME,)}"
            )
        written, pooled = _write_io(wrapper, graph, cases, staged, metadata)
        # MUST: evaluate before publishing (the only non-vacuous check of this series — if it
        # fails, the staging directory is removed with everything in it).
        sanity = _sanity(pooled)
    return {
        "dir": str(out_dir),
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": sum(p.stat().st_size for p in container_parts(out_dir / MODEL_FILE)),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "input_shape": list(example.shape),
        "real_images": {name: md[SOURCE_IMAGE_KEY] for name, md in metadata.items()},
        "sanity": sanity,
    }


def _diff_entry(
    stage: str,
    claim: str,
    got: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Equivalence report for one step (case name → maxdiff, and bit-identity across all cases).

    `bit_exact` is a stronger claim than "difference 0" (`0.0 == -0.0` has difference 0 but
    different bits), and it matters for a step whose **claim is bit-identity itself**, like the
    shape folds.
    """
    return {
        "stage": stage,
        "claim": claim,
        "maxdiff": {
            name: float((got[name] - expected[name]).abs().max()) for name in sorted(expected)
        },
        "bit_exact": all(torch.equal(got[name], expected[name]) for name in expected),
    }


def verify_patches(model_dir: Path) -> list[dict[str, Any]]:
    """Measure equivalence with eager before patching at **two points** (the same shape as
    `sbv2/export.py --verify`).

    1. Shape folds only (conv padding + position embedding) → **bit_exact is the claim itself**.
       The operation sequences correspond one to one, so the difference must be 0; if not, fail
       here.
    2. Up to making the MAP head's q/k/v explicit → **report the maxdiff**. `nn.MultiheadAttention`
       multiplies only q by 1/√d, while SDPA scales q and k symmetrically, so the result is not
       bit-identical (the measured range is spelled in `siglip2.measurements.MAP_HEAD_MAXDIFF`, and
       the scale the difference is divided by in `MAP_HEAD_VECTOR_NORM` of the same module).

    MUST: the order is "**fix the reference values of all cases** → 1 → 2". The patch replaces
    class attributes process-wide, so re-taking the reference per step would make the second
    step's reference a patched value (vacuous).
    """
    if patch.patches_applied():
        raise SystemExit(
            "cannot take the reference in a process that is already patched (the equivalence"
            " check would become vacuous with a difference of 0)"
        )
    model = load_model(model_dir)
    cases = build_cases(model.config)
    with torch.no_grad():
        reference = {
            name: model(pixel_values=pixel_values).pooler_output for name, pixel_values in cases
        }

    def pooled() -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return {
                name: model(pixel_values=pixel_values).pooler_output for name, pixel_values in cases
            }

    patch.apply_shape_patches(model.embeddings)
    folds = _diff_entry("shape-folds", "bit-exact", pooled(), reference)
    if not folds["bit_exact"]:
        raise AssertionError(
            f"the shape folds are not bit-identical: maxdiff={folds['maxdiff']} —"
            " the rewrite of conv padding / the position embedding is not equivalent"
        )
    patch.apply_map_head_patch(model.head)
    return [folds, _diff_entry("map-head", "max-abs-diff", pooled(), reference)]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: outputs/series/<directory name of --model-dir>/)",
    )
    parser.add_argument(
        "--real-images",
        action="store_true",
        help="also write goldens of the 4 real images in outputs/misc/corpus/ (added to the 4"
        " synthetic cases). Needs torchvision / pillow, so run it with"
        " `uv run --group siglip2-preprocess`",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="measure equivalence with eager before patching at 2 points (does not emit — in"
        " one process the class attribute replacement contaminates the reference, so the two"
        " cannot be combined)",
    )
    args = parser.parse_args(argv)
    if args.verify:
        for entry in verify_patches(args.model_dir):
            print(
                f"{entry['stage']} ({entry['claim']}): bit_exact={entry['bit_exact']}"
                f" maxdiff={entry['maxdiff']}"
            )
        return
    out_dir = args.out if args.out is not None else default_out_dir(args.model_dir)
    summary = export_series(args.model_dir, out_dir, real_images=args.real_images)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
