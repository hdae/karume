"""Writes Gemma 4 12B's encoder-free vision embedder as its own graph, with the image goldens.

    cd tools/export-recipes
    uv run --with 'transformers==5.14.1' python -m gemma4.export_vision --model 12b
    uv run --with 'transformers==5.14.1' python -m gemma4.export_vision --model 12b --plan
    uv run --with 'transformers==5.14.1' python -m gemma4.export_vision --model 12b \\
        --embedder-only --out ../../outputs/diag/<dir>     # no 12B reference, no greedy goldens

`--embedder-only` refuses the canonical series directory as `--out`: the distribution places the
vision role from it, and a series without greedy goldens and `reference.json` is not one.

The weights and their config are the pinned snapshot of Google's QAT checkpoint, the tokenizer, the
chat template and the processor configuration the pinned snapshot of `google/gemma-4-12B-it`
(`gemma4.sources`; `hf download` needs `processor_config.json` too — the tokenizer pin names it).
Both are checked against their pins before anything long runs, `processor_config.json` included
(`gemma4.sources.HubPin.also_checked`), and the record names the budget and the digest of the
processor configuration the goldens were taken at (the distribution compares them with what it
declares). Every run prints
per-stage wall time and peak RSS as JSON and runs under a `MemAvailable` floor
({@link qwen3_5.export.MemoryWatch}).

## The graph

Inputs `pixels[1,N,P]` f32, `position_x[1,N]` i32 and `position_y[1,N]` i32 (`P` = 6912 on 12B: one
row is 48 × 48 × 3 raw pixels), `N` symbolic; one output `embeds[1,N,H]` ({@link
VisionEmbedderWrapper}). The host strips the padding upstream's processor adds, so no row is a
padding row and other token budgets need no re-export (`N` is bound by the inputs; 280 is what the
host uses for now).

The wrapper is upstream's `Gemma4UnifiedVisionEmbedder` (LN1, dense, LN2, positions, `pos_norm`,
scale-free RMSNorm, projection) with its own modules, except the position lookup: upstream's
advanced index `pos_embedding[clamped, axes]` has no IR form, so the two columns of `pos_embedding`
become two tables read by `functional.embedding`. MUST: `hidden + (x + y)`, in that grouping —
upstream sums the two lookups first (`.sum(-2)`), and float addition does not associate.

## Storage

The compute is F32 over the checkpoint's values: the embedder is BF16 and is **not** on the q4_0
grid (compared with the non-QAT 12B-it on 2026-11-05, `embedding_projection` and `pos_norm` were
retrained by QAT but stay BF16, and the rest is bit-identical). The linears and the two position
tables are stored as bf16 through the writer path of the regular line's product
(`gemma4.export_product`, ADR 0131 — exact round trip or a loud failure), the parameters bf16 does
not apply to (the norms' affines, the dense bias) as F32 holding the same values widened. No
quantization. The audio embedder's keys are never read.

## Goldens

Taken **before** the container is written, on the images of {@link VISION_IMAGES}, neutral on
purpose (do not add others):

- **Embedder** (`vision.<name>.safetensors`, every image): upstream's
  `Gemma4UnifiedImageProcessor` at the snapshot's budget and upstream's module over the real
  weights in F32, padded rows included as upstream runs them, give `pixels_shape`, `positions` and
  `embeds[N,H]`; `reference.json` records the SHA-256 of the processor's pixel bytes, so the TS
  preprocessing can be compared on real images. The wrapper must equal the module bit for bit on
  the same rows ({@link assert_wrapper_equals}). The padding is held apart, to a bound
  ({@link PADDING_BOUND}): MKL's matrix product rounds by the row count, not by the row's values
  (measured 2026-11-07 on the real `patch_dense`: 1 to 3 rows, and 5 and 7 under 6 threads, differ
  from the same rows in a padded batch by up to 3.1e-5; 264 rows do not differ at all).
- **Greedy with an image** (`greedy.<name>.safetensors`, {@link GREEDY_IMAGE_COUNT} images, the
  first that clear the margin floor — the floor's rule is to drop a case, not to cut its steps):
  the torch reference is `gemma4.export_unified.reference_wrapper` (F32 over imported i4), forward
  by `UnifiedProductWrapper.run` with `gemma4.export_unified.image_masks` (path A), full re-forward
  for every step, suppressed ids masked. The file holds the decode series' `prompt` / `expected` /
  `margin` (`_shared.decode_series`) plus `block` (start, stop of the soft-token run) and
  `first_logits` (the first step's last row, unmasked).
- **The prompt** is rendered by the snapshot's chat template, then `<|image|>` is expanded to
  `<|image>`, N × `<|image|>`, `<image|>` ({@link image_prompt}). MUST: the ids equal what
  upstream's `Gemma4UnifiedProcessor` produces for the same message — it is the source of truth.
- **Path A against path B**: one image's first-step last row with
  `gemma4.export_unified.image_masks_or_block` is recorded against path A in `reference.json`
  (`maskPaths`: largest |logit difference|, KL(A‖B), top-1 agreement, suppressed ids left out).
  Measurement only: the rule stays path A.

## Memory and time

The embedder takes about 1 GiB of the process (measured 2026-11-07) and a second per image. The
goldens hold the 12B reference (about 16 GiB: built in about 34 s, one forward at T ≈ 300 about 25
to 35 s, per the measurements in `gemma4.export_unified`'s docstring), so about 33 forwards for two
images and path B. It is released before the container is written, and a failed gate publishes
nothing
(`karume.artifacts.staged_publication`).

## Output layout

    outputs/series/gemma4-12b-vision/model.krm                  the container (no assets)
    outputs/series/gemma4-12b-vision/vision.<image>.safetensors the embedder goldens
    outputs/series/gemma4-12b-vision/greedy.<image>.safetensors the greedy goldens with an image
    outputs/series/gemma4-12b-vision/reference.json             provenance and the measurements

MUST: transformers is imported inside functions.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim
from torch.nn import functional

from _shared.decode_series import (
    EXPECTED_KEY,
    GREEDY_PREFIX,
    GREEDY_SUFFIX,
    MARGIN_KEY,
    PROMPT_KEY,
    assert_case_room,
    assert_greedy_margins,
    greedy_continuation,
)
from _shared.paths import OUTPUTS_ROOT, REPO_ROOT
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from gemma4 import export_unified as unified
from gemma4 import provenance, sources
from gemma4.card import GEMMA4_LICENSE
from gemma4.distribution import (
    GEMMA4_POSITION_X,
    GEMMA4_POSITION_Y,
    GEMMA4_VISION_PIXELS,
    GEMMA4_VISION_ROLE,
    GEMMA4_VISION_STORAGE,
    GEMMA4_VISION_SUFFIX,
    GEMMA4_VISION_SYMBOL,
    gemma4_processor_config,
    gemma4_soft_token_budget,
)
from gemma4.sources import ModelPins
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import normalize_boundary_tensor
from karume.dist import NOTICE_FILENAME
from karume.ir import IrGraph
from karume.pipeline import export_module, publish_model
from qwen3_5.export import MEMORY_FLOOR_GIB, MemoryWatch

#: The checkpoint's key prefixes this script reads (the two vision entries of
#: `gemma4.export_unified.IGNORED_PREFIXES`; the audio one stays unread). The checkpoint names the
#: patch part `model.vision_embedder.*` and the projection `model.embed_vision.*`
#: (`transformers.conversion_mapping`, `gemma4_unified`).
EMBEDDER_PREFIX = "model.vision_embedder."
PROJECTION_PREFIX = "model.embed_vision."
VISION_PREFIXES: tuple[str, ...] = (EMBEDDER_PREFIX, PROJECTION_PREFIX)

#: The upstream module's name for the projection (its one key under {@link PROJECTION_PREFIX}).
UPSTREAM_PROJECTION_PREFIX = "multimodal_embedder."

#: Row count of the trace's example input (at least 2, or torch.export may specialize `N`) and the
#: symbol's upper bound: the processor's largest budget (`_SUPPORTED_SOFT_TOKENS` — also
#: `mm_posemb_size`, so a position never leaves the tables).
EXAMPLE_ROWS = 3
ROW_SYM_MAX = 1120

#: The linears and the position tables: every tensor stored as bf16 (the form check counts them).
BF16_WEIGHTS: tuple[str, ...] = (
    "patch_dense.weight",
    "pos_x",
    "pos_y",
    "multimodal_embedder.embedding_projection.weight",
)

#: The corpus images, in the order the greedy goldens try them (`outputs/test-assets/corpus/` —
#: assets-layout). Their content is neutral; do not add others.
IMAGE_DIR = OUTPUTS_ROOT / "test-assets" / "corpus" / "wan-i2v"
VISION_IMAGES: Mapping[str, Path] = {
    name: IMAGE_DIR / f"{name}-832x480.png" for name in ("boxing-cats", "ferret", "cat-dog-baking")
}

#: How many images get greedy goldens ({@link take_greedy}).
GREEDY_IMAGE_COUNT = 2

#: The text after the image in the goldens' one user message.
PROMPT_TEXT = "Describe this image in one sentence."

#: How far the valid rows may move between upstream's padded batch and the same rows alone: far
#: above the measured rounding (module docstring), far below what a mishandled padding row would do.
PADDING_BOUND = 1e-3

#: The embedder golden's file name and keys.
VISION_GOLDEN_PREFIX = "vision."
VISION_GOLDEN_SUFFIX = ".safetensors"
PIXELS_SHAPE_KEY = "pixels_shape"
POSITIONS_KEY = "positions"
EMBEDS_KEY = "embeds"

#: What the greedy golden holds beyond `_shared.decode_series`'s three keys.
BLOCK_KEY = "block"
FIRST_LOGITS_KEY = "first_logits"


class VisionExportError(RuntimeError):
    """The checkpoint, the processor or the graph is not what this script expects."""


# ---------------------------------------------------------------------------
# Config, checkpoint, modules
# ---------------------------------------------------------------------------


def load_config(model_dir: Path) -> Any:
    """The checkpoint's whole config (the embedder reads its vision and its text part)."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_dir)
    if config.model_type != "gemma4_unified" or getattr(config, "vision_config", None) is None:
        raise VisionExportError(
            f"{model_dir / 'config.json'}: model type {config.model_type!r} has no encoder-free"
            " vision embedder (E2B / E4B's tower is another graph)"
        )
    return config


def upstream_key(checkpoint_key: str) -> str:
    """Checkpoint key → the key in upstream's `Gemma4UnifiedVisionEmbedder`."""
    if checkpoint_key.startswith(EMBEDDER_PREFIX):
        return checkpoint_key.removeprefix(EMBEDDER_PREFIX)
    if checkpoint_key.startswith(PROJECTION_PREFIX):
        return UPSTREAM_PROJECTION_PREFIX + checkpoint_key.removeprefix(PROJECTION_PREFIX)
    raise VisionExportError(f"'{checkpoint_key}' is outside the vision embedder")


def vision_keys(model_dir: Path) -> list[str]:
    """The checkpoint's vision keys, in file order (the header only — no tensor is read)."""
    path = model_dir / one_shot.CHECKPOINT_FILE
    if not path.is_file():
        raise VisionExportError(f"{path} is missing")
    with safe_open(str(path), framework="pt") as handle:
        keys = list(handle.keys())
    return [key for key in keys if key.startswith(VISION_PREFIXES)]


def read_state(model_dir: Path) -> dict[str, torch.Tensor]:
    """The embedder's tensors, widened to F32, under upstream's keys. Nothing else is read — the
    audio embedder's keys stay untouched."""
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(model_dir / one_shot.CHECKPOINT_FILE), framework="pt") as handle:
        for key in vision_keys(model_dir):
            view = handle.get_slice(key)
            if view.get_dtype() not in unified.ACCEPTED_DTYPES:
                raise VisionExportError(
                    f"'{key}' is {view.get_dtype()} (only {sorted(unified.ACCEPTED_DTYPES)})"
                )
            state[upstream_key(key)] = handle.get_tensor(key).to(torch.float32, copy=True)
    return state


def upstream_embedder(config: Any, state: Mapping[str, torch.Tensor] | None = None) -> nn.Module:
    """Upstream's `Gemma4UnifiedVisionEmbedder` over `state`; `None` keeps its initial values
    (the plan's trace, which never reads a value).

    MUST: the checkpoint's keys are exactly the module's (`strict`) — reading only what the module
    asks for would silently accept a checkpoint of another structure.
    """
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedVisionEmbedder,
    )

    if state is None:
        return Gemma4UnifiedVisionEmbedder(config.vision_config, config.text_config).eval()
    with torch.device("meta"):
        embedder = Gemma4UnifiedVisionEmbedder(config.vision_config, config.text_config)
    embedder.load_state_dict(state, assign=True, strict=True)
    return embedder.eval().requires_grad_(False)


class VisionEmbedderWrapper(nn.Module):
    """`(pixels[1,N,P], position_x[1,N], position_y[1,N]) → embeds[1,N,H]`: upstream's embedder
    without padding rows (module docstring, "The graph").

    MUST: the graph inputs carry these names (the host binds by name — the spellings are
    `gemma4.distribution`'s mirrors, held equal by {@link assert_ir_form_vision}).
    """

    def __init__(self, embedder: nn.Module) -> None:
        super().__init__()
        self.patch_ln1 = embedder.patch_ln1
        self.patch_dense = embedder.patch_dense
        self.patch_ln2 = embedder.patch_ln2
        self.pos_norm = embedder.pos_norm
        self.multimodal_embedder = embedder.multimodal_embedder
        table = embedder.pos_embedding.detach()
        self.pos_x = nn.Parameter(table[:, 0].clone(), requires_grad=False)
        self.pos_y = nn.Parameter(table[:, 1].clone(), requires_grad=False)

    def forward(
        self, pixels: torch.Tensor, position_x: torch.Tensor, position_y: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.patch_ln2(self.patch_dense(self.patch_ln1(pixels)))
        # MUST: the lookups are added first (module docstring).
        positions = functional.embedding(position_x, self.pos_x) + functional.embedding(
            position_y, self.pos_y
        )
        return self.multimodal_embedder(self.pos_norm(hidden + positions))


def trace(
    wrapper: VisionEmbedderWrapper, rows: int = EXAMPLE_ROWS
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """Export with the symbolic row count (the upstream modules' shapes only — the graph does not
    depend on the values or on `rows`)."""
    width = int(wrapper.patch_ln1.normalized_shape[0])
    symbol = Dim(GEMMA4_VISION_SYMBOL, min=1, max=ROW_SYM_MAX)
    return export_module(
        wrapper,
        (
            torch.zeros(1, rows, width, dtype=torch.float32),
            torch.zeros(1, rows, dtype=torch.int64),
            torch.zeros(1, rows, dtype=torch.int64),
        ),
        dynamic_shapes=({1: symbol}, {1: symbol}, {1: symbol}),
        symbol_names=(GEMMA4_VISION_SYMBOL,),
    )


def write_container(
    graph: IrGraph, tensors: Mapping[str, torch.Tensor], path: Path, revision: str | None
) -> IrGraph:
    """Write the container (`path` is the representative path of the parts); its provenance names
    the upstream commit when the checkpoint is pinned (the distribution checks it). The storage is
    bf16 where it applies (module docstring, "Storage")."""
    declared = {init.tensor for init in graph.initializers.values()}
    return publish_model(
        path,
        graph,
        {name: tensor for name, tensor in tensors.items() if name in declared},
        provenance=Provenance(
            license=GEMMA4_LICENSE, notice=NOTICE_FILENAME, upstream_revision=revision
        ),
        graph_name=GEMMA4_VISION_ROLE,
        weight_dtype=GEMMA4_VISION_STORAGE,
    )


def assert_ir_form_vision(graph: IrGraph, config: Any, *, stored: bool) -> dict[str, Any]:
    """The embedder graph's form: the inputs the host binds by name, one `[1,N,H]` output, no state,
    one symbol, the two position lookups; `stored` — a written graph — also the storage.

    MUST: the position lookups are the only `embedding` nodes and read a table of `mm_posemb_size`
    rows each, so an advanced index cannot have slipped in another way, and every weight of
    {@link BF16_WEIGHTS} is stored bf16: an F32 asset of the same graph runs the same numbers.
    """
    hidden = int(config.text_config.hidden_size)
    width = int(config.vision_config.model_patch_size) ** 2 * 3
    symbol = GEMMA4_VISION_SYMBOL
    expected = [
        [GEMMA4_VISION_PIXELS, "f32", [1, symbol, width]],
        [GEMMA4_POSITION_X, "i32", [1, symbol]],
        [GEMMA4_POSITION_Y, "i32", [1, symbol]],
    ]
    found = [[spec.name, spec.dtype, list(spec.shape)] for spec in graph.inputs]
    if found != expected:
        raise VisionExportError(f"graph inputs are {found}, not {expected}")
    if len(graph.outputs) != 1:
        raise VisionExportError(f"{len(graph.outputs)} outputs (one embeds output)")
    output = graph.values[graph.outputs[0]]
    if output.dtype != "f32" or list(output.shape) != [1, symbol, hidden]:
        raise VisionExportError(
            f"the output is {output.dtype} {list(output.shape)}, not f32 {[1, symbol, hidden]}"
        )
    if graph.states:
        raise VisionExportError(f"the graph holds state slots {sorted(graph.states)}")
    if list(graph.symbols) != [symbol]:
        raise VisionExportError(f"the graph's symbols are {list(graph.symbols)}, not {[symbol]}")
    tables = [node for node in graph.nodes if node.op == "embedding"]
    if sorted(node.ins[1] for node in tables) != [GEMMA4_POSITION_X, GEMMA4_POSITION_Y]:
        raise VisionExportError(f"`embedding` nodes read {[node.ins for node in tables]}")
    rows = int(config.vision_config.mm_posemb_size)
    for node in tables:
        shape = list(graph.values[node.ins[0]].shape)
        if shape != [rows, hidden]:
            raise VisionExportError(f"position table '{node.ins[0]}' is {shape}")
    summary: dict[str, Any] = {
        "inputs": found,
        "ops": dict(Counter(node.op for node in graph.nodes).most_common()),
    }
    if stored:
        storage = {key: init.storage.dtype for key, init in _by_tensor(graph).items()}
        wrong = {key: storage.get(key) for key in BF16_WEIGHTS if storage.get(key) != "bf16"}
        if wrong:
            raise VisionExportError(f"weights not stored as bf16: {wrong}")
        others = {key: dtype for key, dtype in storage.items() if key not in BF16_WEIGHTS}
        if set(others.values()) - {"f32"}:
            raise VisionExportError(f"parameters neither bf16 weights nor f32: {others}")
        summary["storage"] = dict(Counter(storage.values()).most_common())
    return summary


def _by_tensor(graph: IrGraph) -> dict[str, Any]:
    return {init.tensor: init for init in graph.initializers.values()}


# ---------------------------------------------------------------------------
# The processor and the prompt
# ---------------------------------------------------------------------------


def load_processor(tokenizer_dir: Path) -> Any:
    """Upstream's `Gemma4UnifiedProcessor` from the tokenizer source (its tokenizer, chat template
    and `processor_config.json`). MUST: a missing configuration fails before the long run."""
    from transformers import AutoProcessor

    gemma4_processor_config(tokenizer_dir)
    return AutoProcessor.from_pretrained(tokenizer_dir)


def load_image(path: Path) -> Any:
    """The image read by PIL as RGB (the processor's `do_convert_rgb`, done here so the bytes the
    processor sees are the ones hashed)."""
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


@dataclass(frozen=True)
class ProcessedImage:
    """Upstream's processor output: padded as it hands it to the model, and the valid rows."""

    padded_pixels: torch.Tensor
    padded_positions: torch.Tensor
    #: `[N, P]` F32 — what the graph's `pixels` holds.
    pixels: torch.Tensor
    #: `[N, 2]` int64, (x, y) — what `position_x` / `position_y` hold.
    positions: torch.Tensor

    @property
    def soft_tokens(self) -> int:
        return int(self.pixels.shape[0])


def process_image(processor: Any, image: Any) -> ProcessedImage:
    """Upstream's image processor at the snapshot's budget (`max_soft_tokens`, the configuration
    the distribution also reads).

    MUST: the valid rows come first, then the padding rows (−1, −1) — the host's graph has no
    padding row, so rows out of that order would not be upstream's. MUST: the pixels are float32
    as upstream's processor gives them (its rescale is `u8 * float32(1/255)`).
    """
    features = processor.image_processor(images=[image], return_tensors="pt")
    tokens = int(features["num_soft_tokens_per_image"][0])
    pixels = features["pixel_values"]
    positions = features["image_position_ids"]
    if pixels.dtype != torch.float32:
        raise VisionExportError(f"the processor's pixels are {pixels.dtype}, not float32")
    valid = positions[0] != -1
    expected = torch.zeros_like(valid)
    expected[:tokens] = True
    if not torch.equal(valid, expected):
        raise VisionExportError(f"the {tokens} valid rows are not the first rows of the padding")
    return ProcessedImage(
        padded_pixels=pixels,
        padded_positions=positions,
        pixels=pixels[0, :tokens],
        positions=positions[0, :tokens],
    )


@dataclass(frozen=True)
class ImagePrompt:
    """One image's prompt: the ids `[1,T]` and the block `[start, stop)` of its soft tokens."""

    ids: torch.Tensor
    block: tuple[int, int]


def _message(image: Any | None) -> list[dict[str, Any]]:
    part = {"type": "image"} if image is None else {"type": "image", "image": image}
    return [{"role": "user", "content": [part, {"type": "text", "text": PROMPT_TEXT}]}]


def image_prompt(processor: Any, image: Any, soft_tokens: int, config: Any) -> ImagePrompt:
    """Render by the chat template, expand `<|image|>` as the processor does, and hold the ids to
    upstream's `Gemma4UnifiedProcessor` for the same message (module docstring).

    MUST: the three ids agree between the tokenizer and `config.json` (the distribution declares
    the config's), and the processor's `mm_token_type_ids` marks exactly the block: `<|image>` and
    `<image|>` are outside it.
    """
    tokenizer = processor.tokenizer
    begin = int(tokenizer.convert_tokens_to_ids(processor.boi_token))
    soft = int(tokenizer.convert_tokens_to_ids(processor.image_token))
    end = int(tokenizer.convert_tokens_to_ids(processor.eoi_token))
    declared = (
        int(config.boi_token_id),
        int(config.image_token_id),
        int(config.eoi_token_id),
    )
    if (begin, soft, end) != declared:
        raise VisionExportError(
            f"the tokenizer's image ids {(begin, soft, end)} differ from config.json's {declared}"
        )
    rendered = tokenizer.apply_chat_template(
        _message(None), tokenize=False, add_generation_prompt=True
    )
    encoded = list(tokenizer(rendered, add_special_tokens=False)["input_ids"])
    if encoded.count(soft) != 1:
        raise VisionExportError(f"the rendered prompt holds {encoded.count(soft)} image tokens")
    at = encoded.index(soft)
    ids = [*encoded[:at], begin, *([soft] * soft_tokens), end, *encoded[at + 1 :]]
    block = (at + 1, at + 1 + soft_tokens)
    upstream = processor.apply_chat_template(
        _message(image),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if upstream["input_ids"][0].tolist() != ids:
        raise VisionExportError(
            "the expanded prompt differs from upstream's processor for the same message"
            f" ({len(ids)} ids against {int(upstream['input_ids'].shape[1])})"
        )
    marked = (upstream["mm_token_type_ids"][0] == 1).nonzero().flatten().tolist()
    if marked != list(range(*block)):
        raise VisionExportError(f"upstream's image rows are not the block {block}")
    return ImagePrompt(ids=torch.tensor([ids], dtype=torch.int64), block=block)


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Embedder goldens
# ---------------------------------------------------------------------------


def embed_upstream(embedder: nn.Module, image: ProcessedImage) -> torch.Tensor:
    """`[N,H]`: upstream's module on the padded rows it is given upstream, padding stripped."""
    with torch.no_grad():
        output = embedder(image.padded_pixels, image.padded_positions)
    return output[0, : image.soft_tokens].contiguous()


def embed_upstream_valid(embedder: nn.Module, image: ProcessedImage) -> torch.Tensor:
    """`[N,H]`: upstream's module on the valid rows alone, the batch the graph runs."""
    with torch.no_grad():
        return embedder(image.pixels.unsqueeze(0), image.positions.unsqueeze(0))[0]


def embed_wrapper(wrapper: VisionEmbedderWrapper, image: ProcessedImage) -> torch.Tensor:
    """`[N,H]`: the export wrapper on the valid rows, as the host calls the graph."""
    with torch.no_grad():
        output = wrapper(
            image.pixels.unsqueeze(0),
            image.positions[:, 0].unsqueeze(0),
            image.positions[:, 1].unsqueeze(0),
        )
    return output[0]


def assert_wrapper_equals(got: torch.Tensor, want: torch.Tensor, where: str) -> None:
    """The wrapper is upstream's module bit for bit on the valid rows. MUST: fail before anything
    is published — a golden the graph's own torch form does not reproduce checks nothing."""
    if got.shape != want.shape or not torch.equal(got, want):
        worst = float((got - want).abs().max()) if got.shape == want.shape else float("nan")
        raise VisionExportError(
            f"{where}: the wrapper differs from upstream's module (largest difference {worst})"
        )


def padding_difference(alone: torch.Tensor, padded: torch.Tensor, where: str) -> float:
    """The largest difference between the valid rows alone and in the padded batch, which fails
    above {@link PADDING_BOUND}."""
    worst = float((alone - padded).abs().max())
    if worst > PADDING_BOUND:
        raise VisionExportError(
            f"{where}: the padding rows move the valid rows by {worst} (bound {PADDING_BOUND})"
        )
    return worst


def write_vision_golden(
    out_dir: Path, name: str, image: ProcessedImage, embeds: torch.Tensor
) -> Path:
    path = out_dir / f"{VISION_GOLDEN_PREFIX}{name}{VISION_GOLDEN_SUFFIX}"
    save_file(
        {
            PIXELS_SHAPE_KEY: torch.tensor(list(image.pixels.shape), dtype=torch.int32),
            POSITIONS_KEY: image.positions.to(torch.int32).contiguous(),
            EMBEDS_KEY: embeds.contiguous(),
        },
        str(path),
    )
    return path


# ---------------------------------------------------------------------------
# Greedy goldens
# ---------------------------------------------------------------------------


def image_forward(
    wrapper: unified.UnifiedProductWrapper,
    specs: Mapping[str, Any],
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    embeds: torch.Tensor,
    block: tuple[int, int],
    masks: Any = unified.image_masks,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The forward with an image: the rows of `block` take `embeds`, rows after the prompt stay
    text, and `masks` builds the attention (path A by default) — `(logits of the last row, hidden)`.
    """
    return wrapper.run(
        input_ids,
        *unified.image_soft_inputs(input_ids, embeds, block),
        *decode.rope_args(specs, position_ids),
        decode.last_row_for(input_ids),
        masks(int(input_ids.shape[1]), wrapper.sliding_window, block),
    )


class ImageGreedyView(nn.Module):
    """`(input_ids, position_ids) → (logits of the last row, hidden)` with one image — the face
    `_shared.decode_series.greedy_continuation` calls (`gemma4.export_unified.GreedyView`'s role, on
    path A). `first_logits` keeps the first call's last row, unmasked."""

    def __init__(
        self,
        wrapper: unified.UnifiedProductWrapper,
        specs: Mapping[str, Any],
        suppress: Sequence[int],
        embeds: torch.Tensor,
        block: tuple[int, int],
    ) -> None:
        super().__init__()
        self.wrapper = wrapper
        self.specs = dict(specs)
        self.suppress = tuple(suppress)
        self.embeds = embeds
        self.block = block
        self.first_logits: torch.Tensor | None = None

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, hidden = image_forward(
            self.wrapper, self.specs, input_ids, position_ids, self.embeds, self.block
        )
        if self.first_logits is None:
            self.first_logits = logits[0, -1].clone()
        return unified.suppressed_logits(logits, self.suppress), hidden


def compare_mask_paths(
    path_a: torch.Tensor, path_b: torch.Tensor, suppress: Sequence[int]
) -> dict[str, Any]:
    """One last row under path A and path B, over the ids a host can select (suppressed ids left
    out): the largest |logit difference|, KL(A‖B) and whether the top-1 agrees. In F64, so the KL of
    two nearly equal distributions is not rounding noise."""
    keep = torch.ones(path_a.shape, dtype=torch.bool)
    keep[list(suppress)] = False
    a, b = path_a[keep].to(torch.float64), path_b[keep].to(torch.float64)
    log_a, log_b = functional.log_softmax(a, dim=-1), functional.log_softmax(b, dim=-1)
    ids = keep.nonzero().flatten()
    top_a, top_b = int(ids[int(a.argmax())]), int(ids[int(b.argmax())])
    return {
        "maxAbsLogitDifference": float((a - b).abs().max()),
        "klAB": float((log_a.exp() * (log_a - log_b)).sum()),
        "top1Agrees": top_a == top_b,
        "top1A": top_a,
        "top1B": top_b,
    }


@dataclass(frozen=True)
class GreedyGolden:
    """One image's measured greedy continuation."""

    name: str
    prompt: ImagePrompt
    tokens: list[int]
    margins: list[float]
    first_logits: torch.Tensor


def write_greedy_golden(out_dir: Path, golden: GreedyGolden) -> Path:
    """The decode series' three keys (`_shared.decode_series._write_greedy`) plus the block and the
    first step's last row."""
    name = golden.name
    path = out_dir / f"{GREEDY_PREFIX}{name}{GREEDY_SUFFIX}"
    save_file(
        {
            PROMPT_KEY: normalize_boundary_tensor(golden.prompt.ids[0], f"prompt of {name}"),
            EXPECTED_KEY: normalize_boundary_tensor(
                torch.tensor(golden.tokens, dtype=torch.int64), f"expected of {name}"
            ),
            MARGIN_KEY: torch.tensor(golden.margins, dtype=torch.float32).contiguous(),
            BLOCK_KEY: torch.tensor(list(golden.prompt.block), dtype=torch.int32),
            FIRST_LOGITS_KEY: golden.first_logits.contiguous(),
        },
        str(path),
    )
    return path


def take_greedy(
    view_of: Any,
    candidates: Sequence[tuple[str, ImagePrompt]],
    steps: int,
    floor: float,
) -> tuple[list[GreedyGolden], dict[str, list[float]]]:
    """Greedy continuations of the candidates, in order, until {@link GREEDY_IMAGE_COUNT} clear the
    margin floor; `view_of(name, prompt)` builds the image's view. Returns those and the margins of
    the images dropped.

    MUST: a step at or below the floor drops the image, never the step (the floor's rule —
    `_shared.decode_series.assert_greedy_margins`), and fewer images than asked fail loudly with
    every offender named.
    """
    kept: list[GreedyGolden] = []
    dropped: dict[str, list[float]] = {}
    for name, prompt in candidates:
        if len(kept) == GREEDY_IMAGE_COUNT:
            break
        view = view_of(name, prompt)
        tokens, margins = greedy_continuation(view, prompt.ids, steps, label=name)
        if min(margins) > floor:
            kept.append(GreedyGolden(name, prompt, tokens, margins, view.first_logits))
        else:
            print(f"[greedy] {name}: dropped, margins {margins}", file=sys.stderr, flush=True)
            dropped[name] = margins
    if len(kept) < GREEDY_IMAGE_COUNT:
        assert_greedy_margins(dropped, floor)
        raise VisionExportError(f"{len(kept)} images have goldens, not {GREEDY_IMAGE_COUNT}")
    return kept, dropped


def _reference(model_dir: Path) -> unified.UnifiedProductWrapper:
    """12B's reference, as its product goldens take it (`gemma4.export_unified`, "Goldens")."""
    config = unified.load_config(model_dir)
    int4 = unified.int4_targets(unified.UnifiedProductWrapper(unified.meta_model(config)))
    return unified.reference_wrapper(config, unified.Checkpoint(model_dir), int4)


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------


def _source_images() -> dict[str, Any]:
    missing = [str(path) for path in VISION_IMAGES.values() if not path.is_file()]
    if missing:
        raise VisionExportError(f"corpus images are missing: {missing}")
    return {name: load_image(path) for name, path in VISION_IMAGES.items()}


def _image_records(
    processor: Any, images: Mapping[str, Any], config: Any
) -> tuple[dict[str, ProcessedImage], dict[str, ImagePrompt]]:
    """Every image's processor output and its prompt (no weight involved)."""
    processed = {name: process_image(processor, image) for name, image in images.items()}
    prompts = {
        name: image_prompt(processor, images[name], processed[name].soft_tokens, config)
        for name in images
    }
    return processed, prompts


def _case_lengths(prompts: Mapping[str, ImagePrompt]) -> dict[str, int]:
    return {name: int(prompt.ids.shape[1]) for name, prompt in prompts.items()}


def _keys_summary(model_dir: Path, wrapper_keys: frozenset[str]) -> dict[str, Any]:
    """The checkpoint's vision keys against the module's (the header only)."""
    keys = vision_keys(model_dir)
    expected = {upstream_key(key) for key in keys}
    if expected != wrapper_keys:
        raise VisionExportError(
            f"the checkpoint's vision keys {sorted(expected ^ wrapper_keys)} differ from the"
            " module's"
        )
    return {"vision_keys": keys}


def plan(
    model_dir: Path,
    *,
    tokenizer_dir: Path,
    pins: ModelPins | None,
    floor: float = MEMORY_FLOOR_GIB,
) -> dict[str, Any]:
    """Cheap and without a weight: the pins, the checkpoint's vision keys against the module, the
    graph's form, the processor on the corpus images and every prompt against upstream's
    processor."""
    if pins is not None:
        sources.assert_model_pins(model_dir, tokenizer_dir, pins)
    with MemoryWatch(floor) as monitor, monitor.stage("trace"):
        config = load_config(model_dir)
        embedder = upstream_embedder(config)
        keys = _keys_summary(model_dir, frozenset(embedder.state_dict()))
        graph, _ = trace(VisionEmbedderWrapper(embedder))
        form = assert_ir_form_vision(graph, config, stored=False)
        processor = load_processor(tokenizer_dir)
        processed, prompts = _image_records(processor, _source_images(), config)
    return {
        "stages": [stage.to_dict() for stage in monitor.records],
        **form,
        **keys,
        "symbols": list(graph.symbols),
        "soft_token_budget": gemma4_soft_token_budget(tokenizer_dir),
        "soft_tokens": {name: image.soft_tokens for name, image in processed.items()},
        "prompt_lengths": _case_lengths(prompts),
        "blocks": {name: list(prompt.block) for name, prompt in prompts.items()},
    }


def export_series(
    model_dir: Path,
    out_dir: Path,
    *,
    tokenizer_dir: Path,
    pins: ModelPins | None,
    steps: int = decode.GREEDY_STEPS["12b"],
    embedder_only: bool = False,
    floor: float = MEMORY_FLOOR_GIB,
    checkpoint_label: str | None = None,
) -> dict[str, Any]:
    """Write the container, the embedder goldens, the greedy goldens and the provenance record
    through one staged seat (`embedder_only` writes the container and the embedder goldens).

    MUST: every gate (the wrapper against upstream's module, the padding's bound, the prompts
    against upstream's processor, the margins, the form, the pinned fingerprint) passes before the
    seat is published.
    MUST: the goldens are taken before the container is written, and the reference (about 16 GiB)
    is released first.
    """
    if pins is not None:
        sources.assert_model_pins(model_dir, tokenizer_dir, pins)
    with MemoryWatch(floor) as monitor:
        with monitor.stage("trace") as record:
            config = load_config(model_dir)
            state = read_state(model_dir)
            embedder = upstream_embedder(config, state)
            wrapper = VisionEmbedderWrapper(embedder).requires_grad_(False)
            graph, tensors = trace(wrapper)
            form = assert_ir_form_vision(graph, config, stored=False)
            record.details["nodes"] = len(graph.nodes)
        with monitor.stage("embedder-goldens") as record:
            processor = load_processor(tokenizer_dir)
            images = _source_images()
            processed, prompts = _image_records(processor, images, config)
            embeds = {name: embed_upstream(embedder, image) for name, image in processed.items()}
            padding = {}
            for name, image in processed.items():
                alone = embed_upstream_valid(embedder, image)
                assert_wrapper_equals(embed_wrapper(wrapper, image), alone, name)
                padding[name] = padding_difference(alone, embeds[name], name)
            record.details["images"] = len(images)
        text = unified.load_config(model_dir)
        if not embedder_only:
            assert_case_room(
                [(name, prompt.ids) for name, prompt in prompts.items()],
                steps,
                decode.max_position(text),
            )
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        with staged_publication(out_dir) as staged:
            staged.mkdir()
            vision_files = {
                name: write_vision_golden(staged, name, image, embeds[name])
                for name, image in processed.items()
            }
            greedy: dict[str, Any] = {}
            suppress = unified.suppress_tokens(tokenizer_dir, text)
            if not embedder_only:
                with monitor.stage("goldens"):
                    print(
                        "[goldens] reference: imported i4 codes, dequantized at use;"
                        f" suppressed ids {list(suppress)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    reference = _reference(model_dir)
                    specs = decode.rope_specs(reference.model.config)
                    goldens, dropped = take_greedy(
                        lambda name, prompt: ImageGreedyView(
                            reference, specs, suppress, embeds[name], prompt.block
                        ),
                        list(prompts.items()),
                        steps,
                        decode.MARGIN_FLOOR,
                    )
                    first = goldens[0]
                    with torch.no_grad():
                        logits, _ = image_forward(
                            reference,
                            specs,
                            first.prompt.ids,
                            decode.positions_for(first.prompt.ids),
                            embeds[first.name],
                            first.prompt.block,
                            unified.image_masks_or_block,
                        )
                    mask_paths = {
                        "image": first.name,
                        **compare_mask_paths(first.first_logits, logits[0, -1], suppress),
                    }
                    del reference
                gc.collect()
                written = [write_greedy_golden(staged, golden) for golden in goldens]
                greedy = {
                    "greedy": [path.name for path in written],
                    "greedy_steps": steps,
                    "margin_min": {golden.name: min(golden.margins) for golden in goldens},
                    "dropped_images": dropped,
                    "continuation": {
                        golden.name: processor.tokenizer.decode(golden.tokens) for golden in goldens
                    },
                    "mask_paths": mask_paths,
                }
            with monitor.stage("write"):
                verified = write_container(
                    graph,
                    tensors,
                    staged / one_shot.MODEL_FILE,
                    None if pins is None else pins.weights.revision,
                )
            form = assert_ir_form_vision(verified, config, stored=True)
            if not embedder_only:
                with monitor.stage("provenance"):
                    own = provenance.assert_reference_goldens(
                        staged, [(golden.name, golden.prompt.ids) for golden in goldens]
                    )
                    vision = {
                        name: {
                            "file": vision_files[name].name,
                            **provenance.file_digest(vision_files[name]),
                            "source": {
                                "path": str(VISION_IMAGES[name].relative_to(REPO_ROOT)),
                                **provenance.file_digest(VISION_IMAGES[name]),
                            },
                            "softTokens": image.soft_tokens,
                            "pixelsSha256": _digest(image.pixels),
                            "positionsSha256": _digest(image.positions.to(torch.int32)),
                            "paddingMaxDifference": padding[name],
                        }
                        for name, image in processed.items()
                    }
                    record_ = provenance.build_record(
                        out_dir,
                        model_dir,
                        {
                            **own,
                            "series": out_dir.name,
                            "suppressTokens": list(suppress),
                            "softTokenBudget": gemma4_soft_token_budget(tokenizer_dir),
                            "processorConfig": provenance.file_digest(
                                tokenizer_dir / "processor_config.json"
                            ),
                            "promptText": PROMPT_TEXT,
                            "vision": vision,
                            "greedyImages": {
                                golden.name: {
                                    "block": list(golden.prompt.block),
                                    "promptLength": int(golden.prompt.ids.shape[1]),
                                    "marginMin": min(golden.margins),
                                }
                                for golden in goldens
                            },
                            "droppedImages": dropped,
                            "maskPaths": mask_paths,
                        },
                        checkpoint_label=checkpoint_label,
                        tokenizer_dir=tokenizer_dir,
                        pins=pins,
                    )
                    if pins is not None:
                        provenance.assert_pinned_fingerprint(record_["checkpoint"], pins)
                    provenance.write_record(staged, record_)
    return {
        "dir": str(out_dir),
        "nodes": len(verified.nodes),
        "outputs": len(verified.outputs),
        "initializers": len(verified.initializers),
        "parts": len(container_parts(out_dir / one_shot.MODEL_FILE)),
        "model_bytes": sum(
            path.stat().st_size for path in container_parts(out_dir / one_shot.MODEL_FILE)
        ),
        "symbols": list(verified.symbols),
        **form,
        "soft_tokens": {name: image.soft_tokens for name, image in processed.items()},
        "prompt_lengths": _case_lengths(prompts),
        "pixels_sha256": {name: _digest(image.pixels) for name, image in processed.items()},
        "padding_max_difference": padding,
        **greedy,
        "stages": [stage.to_dict() for stage in monitor.records],
    }


def canonical_series_dir(model: str) -> Path:
    """The series the distribution reads the vision role from
    (`outputs/series/gemma4-<model>-vision/` — `gemma4.distribution.gemma4_qat_q4_0_sources`), and
    the default `--out`."""
    return one_shot.series_dir(model, suffix=GEMMA4_VISION_SUFFIX)


def main(argv: Sequence[str] | None = None) -> None:
    parser = one_shot.series_parser(
        __doc__.split("\n\n")[0],
        canonical_series_dir,
        with_sym_max=False,
        models=unified.MODELS,
    )
    parser.set_defaults(model=unified.MODELS[0])
    parser.add_argument("--steps", type=int, default=None)
    one_shot.add_model_default(parser, "steps", decode.GREEDY_STEPS.__getitem__)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="the tokenizer and processor configuration (default: the pinned tokenizer source)",
    )
    one_shot.add_model_default(parser, "tokenizer_dir", sources.tokenizer_dir)
    parser.add_argument(
        "--plan", action="store_true", help="no weight is read and nothing is written"
    )
    parser.add_argument(
        "--embedder-only",
        action="store_true",
        help="the container and the embedder goldens: no reference, no greedy goldens",
    )
    parser.add_argument("--memory-floor", type=float, default=MEMORY_FLOOR_GIB)
    args = parser.parse_args(argv)
    if args.embedder_only and args.out is None:
        parser.error(
            "--embedder-only needs --out (a series without greedy goldens is not a series)"
        )
    # MUST: compared resolved — `../series/...` or a link to the canonical directory would
    # otherwise publish over the full series the distribution reads.
    canonical = canonical_series_dir(args.model)
    if args.embedder_only and args.out.resolve() == canonical.resolve():
        parser.error(
            f"--embedder-only cannot write the canonical series {canonical} (the distribution"
            " places the vision role from it, and a series without greedy goldens and"
            " reference.json is not a series) — pass another --out"
        )
    model = args.model
    options = one_shot.resolve_series_args(args)
    model_dir = options["model_dir"]
    if options["plan"]:
        summary = plan(
            model_dir,
            tokenizer_dir=options["tokenizer_dir"],
            pins=sources.HUB_PINS.get(model),
            floor=options["memory_floor"],
        )
    else:
        summary = export_series(
            model_dir,
            options["out"],
            tokenizer_dir=options["tokenizer_dir"],
            pins=sources.HUB_PINS.get(model),
            steps=options["steps"],
            embedder_only=options["embedder_only"],
            floor=options["memory_floor"],
            checkpoint_label=sources.checkpoint_label(model),
        )
    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "tokenizer_dir": str(options["tokenizer_dir"]),
                **summary,
            },
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
