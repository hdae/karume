"""Script that writes the reference vectors of whole inputs holding an image — what upstream's
SentenceTransformer gives for them — as a fixture for the TypeScript pipeline.

    uv run --with 'transformers==5.23.0' --with 'sentence-transformers==6.1.0' \\
        python -m embeddinggemma2.multimodal_reference

## What it runs

`SentenceTransformer` loaded from the pinned snapshot (the composite `EmbeddingGemma2Model` with its
vision tower, upstream's processor and chat template, Pooling and Normalize), f32 on the CPU,
sentence-transformers and transformers at the releases {@link embeddinggemma2.parity} runs, on the
inputs of {@link CASES} built from the synthetic images of {@link embeddinggemma2.images}: an image
alone, a text then an image, an image then a text, a text with a prompt and an image, and two
images. An input is a dict of upstream's (`{"text": …, "image": …}`, its keys in order — the
snapshot requires sentence-transformers 6.1.0 for that order), spelled in the fixture as the
content parts the pipeline takes (`{"type": "text", "text"}` / `{"type": "image", "image"}`).

For each case the script records what upstream ran: the token ids the composite model received
(caught by a forward hook — `<bos>`, the text runs, each image as `<|image>` + n × `<|image|>` +
`<image|>`, `<eos>`), each image's soft-token count and where its block starts, and the 768-wide
vector `encode` returned. It checks the ids' form against the images' counts.

## The composition check

Karume's two graphs, in torch: each image's padded rows from the same processor run, stripped and
reordered for the tower ({@link embeddinggemma2.export_vision.host_rows}), through the tower wrapper
({@link embeddinggemma2.export_vision.VisionTowerWrapper}); the soft tokens placed at the image's
placeholder rows ({@link embeddinggemma2.export.soft_inputs}); the text wrapper
({@link embeddinggemma2.export.EmbeddingWrapper}) over the ids upstream ran. Its vector is compared
with upstream's (max abs and cosine), within {@link COMPOSITION_TOLERANCE}; the record and the
fixture carry the measured values.

## Output

    packages/models/tests/fixtures/embeddinggemma2/multimodal-reference.json         the cases
    packages/models/tests/fixtures/embeddinggemma2/multimodal-reference.safetensors  the bytes

The safetensors holds `image.<name>` (U8 `[height, width, 3]`, row-major RGB8 — the bytes upstream's
processor read), `<case>.input_ids` (I32 `[T]`) and `<case>.embedding` (F32 `[768]`). The JSON
describes the images (size, the resized size, the soft-token count) and the cases (the content parts
and the prompt name, `T`, the image blocks, the measured composition error) and how they were made.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from _shared.paths import REPO_ROOT
from embeddinggemma2 import export, export_vision, parity, sources
from embeddinggemma2.images import IMAGES

#: The fixture's two files (tracked in git).
FIXTURE_DIR = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "embeddinggemma2"
FIXTURE_JSON = FIXTURE_DIR / "multimodal-reference.json"
FIXTURE_TENSORS = FIXTURE_DIR / "multimodal-reference.safetensors"

#: The pinned files the script reads (SentenceTransformer's and the tower export's).
READ_FILES: tuple[str, ...] = tuple(
    dict.fromkeys((*parity.PINNED_READS, *export_vision.READ_FILES))
)

#: The bound of the composition check (max abs of the unit vectors). Measured on 2026-11-22: see
#: the fixture's `composition` fields; f32 noise of the tower (its soft tokens move by up to 3.3e-5
#: against upstream's padded run) reaches the vector shrunk by the text tower.
COMPOSITION_TOLERANCE = 1e-5


@dataclass(frozen=True)
class Case:
    """One whole input: its content parts in order (`("text", str)` / `("image", name)`), the
    prompt name, and what it covers."""

    name: str
    parts: tuple[tuple[str, str], ...]
    why: str
    prompt: str | None = None


#: The inputs (texts neutral, images from `images.py`).
CASES: tuple[Case, ...] = (
    Case("image-only", (("image", "gradient-square"),), "an image alone"),
    Case(
        "text-then-image",
        (("text", "Flat coloured shapes on a light background"), ("image", "shapes-wide")),
        "a text, then an image (a wide one)",
    ),
    Case(
        "image-then-text",
        (("image", "noise-tall"), ("text", "random coloured noise")),
        "an image (a tall one), then a text",
    ),
    Case(
        "query-text-image",
        (("text", "a soft colour gradient"), ("image", "gradient-square")),
        "a text with a prompt's prefix, then an image (the prefix is glued to the first text run)",
        prompt="query",
    ),
    Case(
        "two-images",
        (("image", "shapes-wide"), ("image", "stripes-edge")),
        "two images, one of the smallest soft-token count, with nothing between them",
    ),
)


class MultimodalReferenceError(ValueError):
    """What upstream ran is not what the fixture describes."""


def upstream_input(case: Case) -> dict[str, Any]:
    """The case as sentence-transformers takes it: one dict, its keys in the parts' order, an
    image key holding a list when the case has two images."""
    from PIL import Image

    def pil(name: str) -> Any:
        return Image.fromarray(IMAGES[name].rgb8(), mode="RGB")

    kinds = [kind for kind, _ in case.parts]
    if len(set(kinds)) != len(kinds) and set(kinds) != {"image"}:
        raise MultimodalReferenceError(f"{case.name}: a repeated text part has no dict form")
    if set(kinds) == {"image"} and len(kinds) > 1:
        return {"image": [pil(value) for _, value in case.parts]}
    return {kind: (pil(value) if kind == "image" else value) for kind, value in case.parts}


def image_blocks(ids: Sequence[int], begin: int, soft: int, end: int) -> list[tuple[int, int]]:
    """`(start, count)` of each run of `soft` ids, each between one `begin` and one `end`."""
    blocks: list[tuple[int, int]] = []
    index = 0
    while index < len(ids):
        if ids[index] != soft:
            index += 1
            continue
        start = index
        while index < len(ids) and ids[index] == soft:
            index += 1
        if start == 0 or ids[start - 1] != begin or index >= len(ids) or ids[index] != end:
            raise MultimodalReferenceError(
                f"the soft run at {start} is not between <|image> and <image|>"
            )
        blocks.append((start, index - start))
    return blocks


@dataclass(frozen=True)
class UpstreamRun:
    """What the composite model received and what `encode` returned for one case."""

    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    position_ids: torch.Tensor
    embedding: torch.Tensor


def run_upstream(model: Any, case: Case) -> UpstreamRun:
    """`encode` of one input, with the composite model's inputs caught on their way in.

    NOTE: sentence-transformers calls the model's `forward` method itself (not the module), so a
    forward hook never fires; the instance's `forward` is wrapped for the call instead.
    """
    caught: dict[str, torch.Tensor] = {}
    composite = model[0].model
    original = composite.forward

    def forward(**kwargs: Any) -> Any:
        for key in ("input_ids", "pixel_values", "image_position_ids", "attention_mask"):
            if key in kwargs:
                caught[key] = kwargs[key].detach().clone()
        return original(**kwargs)

    composite.forward = forward
    try:
        with torch.no_grad():
            vector = model.encode(
                [upstream_input(case)],
                prompt_name=case.prompt,
                convert_to_tensor=True,
                batch_size=1,
            )[0]
    finally:
        del composite.forward
    if set(caught) != {"input_ids", "pixel_values", "image_position_ids", "attention_mask"}:
        raise MultimodalReferenceError(f"{case.name}: the model received {sorted(caught)}")
    if int(caught["input_ids"].shape[0]) != 1 or not bool(caught["attention_mask"].all()):
        raise MultimodalReferenceError(f"{case.name}: the run was batched or padded")
    return UpstreamRun(
        input_ids=caught["input_ids"][0].to(torch.int64),
        pixel_values=caught["pixel_values"],
        position_ids=caught["image_position_ids"],
        embedding=vector.to(torch.float32),
    )


def compose(
    tower: export_vision.VisionTowerWrapper,
    text: export.EmbeddingWrapper,
    run: UpstreamRun,
    blocks: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Karume's two graphs in torch over what upstream ran (module doc, "The composition check")."""
    hidden = int(text.model.config.hidden_size)
    soft: list[tuple[int, torch.Tensor]] = []
    for index, (start, count) in enumerate(blocks):
        rows = export_vision.host_rows(
            run.pixel_values[index : index + 1], run.position_ids[index : index + 1]
        )
        if rows.soft_tokens != count:
            raise MultimodalReferenceError(
                f"image {index}: {rows.soft_tokens} soft tokens, {count} placeholders"
            )
        with torch.no_grad():
            soft.append((start, tower(rows.pixels, rows.position_x, rows.position_y)[0]))
    ids = run.input_ids.unsqueeze(0)
    soft_embeds, soft_mask = export.soft_inputs(ids, hidden, soft)
    with torch.no_grad():
        return text(ids, torch.ones_like(ids, dtype=torch.float32), soft_embeds, soft_mask)[0]


def _difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left, right = left.double(), right.double()
    return {
        "maxAbs": float((left - right).abs().max()),
        "cosine": float(torch.dot(left, right) / (left.norm() * right.norm())),
    }


def measure(directory: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Run every case through upstream and through Karume's two graphs; returns the fixture's JSON
    document and its tensors."""
    import sentence_transformers
    import transformers
    from sentence_transformers import SentenceTransformer

    parity.assert_sentence_transformers_version(sentence_transformers.__version__)
    export.assert_transformers_version(transformers.__version__)
    sources.assert_pinned(directory, READ_FILES)
    config = export_vision.load_config(directory)
    begin, soft, end = (
        int(config.boi_token_id),
        int(config.image_token_id),
        int(config.eoi_token_id),
    )
    upstream = SentenceTransformer(
        str(directory), device="cpu", model_kwargs={"dtype": torch.float32}
    ).eval()
    tower = export_vision.VisionTowerWrapper(export_vision.load_tower(directory)).eval()
    model, _composite = export.load_text_model(directory)
    text = export.EmbeddingWrapper(model, int(model.config.sliding_window)).eval()
    processor = export_vision.load_image_processor(directory)
    counts = {name: export_vision.process_image(processor, image) for name, image in IMAGES.items()}

    tensors: dict[str, torch.Tensor] = {
        f"image.{name}": torch.from_numpy(np.array(image.rgb8())) for name, image in IMAGES.items()
    }
    cases: list[dict[str, Any]] = []
    for case in CASES:
        run = run_upstream(upstream, case)
        ids = run.input_ids.tolist()
        if ids[0] != int(config.text_config.bos_token_id) or ids[-1] != int(
            config.text_config.eos_token_id
        ):
            raise MultimodalReferenceError(f"{case.name}: the ids do not run <bos> … <eos>")
        blocks = image_blocks(ids, begin, soft, end)
        images = [value for kind, value in case.parts if kind == "image"]
        expected = [counts[name].soft_tokens for name in images]
        if [count for _, count in blocks] != expected:
            raise MultimodalReferenceError(
                f"{case.name}: soft runs {blocks}, the images give {expected}"
            )
        composed = compose(tower, text, run, blocks)
        difference = _difference(composed, run.embedding)
        if difference["maxAbs"] > COMPOSITION_TOLERANCE:
            raise MultimodalReferenceError(
                f"{case.name}: Karume's two graphs differ from upstream by {difference}"
            )
        tensors[f"{case.name}.input_ids"] = run.input_ids.to(torch.int32)
        tensors[f"{case.name}.embedding"] = run.embedding.contiguous()
        cases.append(
            {
                "name": case.name,
                "why": case.why,
                "parts": [{"type": kind, kind: value} for kind, value in case.parts],
                **({} if case.prompt is None else {"prompt": case.prompt}),
                "tokens": len(ids),
                "imageBlocks": [
                    {"image": name, "start": start, "softTokens": count}
                    for name, (start, count) in zip(images, blocks, strict=True)
                ],
                "composition": difference,
            }
        )
    document = {
        "_doc": [
            "Reference vectors of whole inputs holding an image, for the EmbeddingGemma 2 pipeline"
            " (generated by tools/export-recipes/embeddinggemma2/multimodal_reference.py).",
            "Upstream: SentenceTransformer of the pinned snapshot, f32 on the CPU; each case is"
            " one `encode` call of an upstream dict input (its keys in the parts' order), with"
            " the prompt name when the case has one.",
            "The tensors of multimodal-reference.safetensors: `image.<name>` U8 [height, width, 3]"
            " (row-major RGB8, what upstream's processor read), `<case>.input_ids` I32 [T] (the"
            " ids the composite model received, `<bos>` and `<eos>` included, each image as"
            " `<|image>` + n x `<|image|>` + `<image|>`), `<case>.embedding` F32 [768] (the"
            " normalised vector `encode` returned).",
            "`composition` is Karume's two graphs composed in torch f32 (the tower wrapper's soft"
            " tokens at the image's placeholder rows of the text wrapper) against upstream's"
            " vector.",
            "The images are synthetic and neutral (tools/export-recipes/embeddinggemma2/images.py);"
            " do not add others.",
        ],
        "source": {
            "upstream": sources.PIN.label,
            "transformers": transformers.__version__,
            "sentenceTransformers": sentence_transformers.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
        },
        "tensors": FIXTURE_TENSORS.name,
        "imageTokens": {"begin": begin, "soft": soft, "end": end},
        "softTokenBudget": export_vision.SOFT_TOKEN_BUDGET,
        "images": [
            {
                "name": name,
                "why": image.why,
                "width": image.width,
                "height": image.height,
                "resized": dict(zip(("width", "height"), counts[name].resized, strict=True)),
                "softTokens": counts[name].soft_tokens,
            }
            for name, image in IMAGES.items()
        ],
        "cases": cases,
    }
    return document, tensors


def fixture_json(document: Mapping[str, Any]) -> str:
    """The JSON as the repository's formatter leaves it: 2-space indent, a trailing newline, and
    exponents without a leading zero (`5.9e-8`, where Python writes `5.9e-08`)."""
    payload = json.dumps(document, indent=2) + "\n"
    return re.sub(r"(\d)e([+-])0(\d)", r"\1e\2\3", payload)


def write_fixture(document: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]) -> None:
    """Write the fixture's two files ({@link fixture_json}; the tensors in key order)."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURE_JSON.write_text(fixture_json(document), encoding="utf-8")
    save_file(dict(sorted(tensors.items())), str(FIXTURE_TENSORS))


def main() -> None:
    document, tensors = measure(sources.SNAPSHOT_DIR)
    write_fixture(document, tensors)
    print(
        json.dumps(
            {
                "fixture": str(FIXTURE_JSON),
                "cases": {case["name"]: case["composition"] for case in document["cases"]},
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
