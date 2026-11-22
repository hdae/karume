"""Script that writes the reference vectors of whole inputs holding an image or an audio — what
upstream's SentenceTransformer gives for them — as a fixture for the TypeScript pipeline.

    uv run --with 'transformers==5.23.0' --with 'sentence-transformers==6.1.0' \\
        python -m embeddinggemma2.multimodal_reference

## What it runs

`SentenceTransformer` loaded from the pinned snapshot (the composite `EmbeddingGemma2Model` with its
vision tower, upstream's processor and chat template, Pooling and Normalize), f32 on the CPU,
sentence-transformers and transformers at the releases {@link embeddinggemma2.parity} runs, on the
inputs of {@link CASES} built from the synthetic images of {@link embeddinggemma2.images}: an image
alone, a text then an image, an image then a text, a text with a prompt and an image, and two
images; and on those of {@link AUDIO_CASES} built from the synthetic waveforms of
{@link embeddinggemma2.waveforms}: an audio alone, a text then an audio, an audio with a prompt, and
an image then an audio. An input is a dict of upstream's (`{"text": …, "image": …, "audio": …}`,
its keys in order — the snapshot requires sentence-transformers 6.1.0 for that order; an audio is
`{"array": f32 samples, "sampling_rate": 16000}`), spelled in the fixture as the content parts the
pipeline takes (`{"type": "text", "text"}` / `{"type": "image", "image"}` /
`{"type": "audio", "audio"}`).

For each case the script records what upstream ran: the token ids the composite model received
(caught by a forward hook — `<bos>`, the text runs, each image as `<|image>` + n × `<|image|>` +
`<image|>`, each audio as `<|audio>` + m × `<|audio|>` + `<audio|>`, `<eos>`), each image's and
audio's soft-token count and where its block starts, and the 768-wide vector `encode` returned. It
checks the ids' form against the counts, and each audio's mel frames against the feature
extractor's own run on the same samples.

The audio cases sit in a list of their own (`audioCases`): the five image cases, their tensors and
every field of the document they had are unchanged, so the image pipeline's checks read the fixture
as before.

## The composition check

Karume's graphs, in torch: each image's padded rows from the same processor run, stripped and
reordered for the tower ({@link embeddinggemma2.export_vision.host_rows}), through the tower wrapper
({@link embeddinggemma2.export_vision.VisionTowerWrapper}); each audio's frames from the same
processor run, their valid ones zero-padded to 4m with the frame mask
({@link embeddinggemma2.export_audio.host_frames}), through the audio tower wrapper
({@link embeddinggemma2.export_audio.AudioTowerWrapper}); the soft tokens placed at the placeholder
rows ({@link embeddinggemma2.export.soft_inputs}); the text wrapper
({@link embeddinggemma2.export.EmbeddingWrapper}) over the ids upstream ran. Its vector is compared
with upstream's (max abs and cosine), within {@link COMPOSITION_TOLERANCE}; the record and the
fixture carry the measured values.

## Output

    packages/models/tests/fixtures/embeddinggemma2/multimodal-reference.json         the cases
    packages/models/tests/fixtures/embeddinggemma2/multimodal-reference.safetensors  the bytes

The safetensors holds `image.<name>` (U8 `[height, width, 3]`, row-major RGB8 — the bytes upstream's
processor read), `audio.<name>` (F32 `[samples]`, 16 kHz mono — the samples upstream's processor
read), `audio.<name>.features` (F32 `[L, 128]`, the valid log-mel frames upstream's feature
extractor made of them; its frames after them are zeros), `<case>.input_ids` (I32 `[T]`) and
`<case>.embedding` (F32 `[768]`). The JSON describes the images (size, the resized size, the
soft-token count), the audios (samples, frames, soft-token count) and the cases (the content parts
and the prompt name, `T`, the image and audio blocks, the measured composition error) and how they
were made.
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
from embeddinggemma2 import export, export_audio, export_vision, parity, sources
from embeddinggemma2.images import IMAGES
from embeddinggemma2.waveforms import SAMPLE_RATE, WAVEFORMS

#: The fixture's two files (tracked in git).
FIXTURE_DIR = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "embeddinggemma2"
FIXTURE_JSON = FIXTURE_DIR / "multimodal-reference.json"
FIXTURE_TENSORS = FIXTURE_DIR / "multimodal-reference.safetensors"

#: The pinned files the script reads (SentenceTransformer's and the tower exports').
READ_FILES: tuple[str, ...] = tuple(
    dict.fromkeys((*parity.PINNED_READS, *export_vision.READ_FILES, *export_audio.READ_FILES))
)

#: The bound of the composition check (max abs of the unit vectors). Measured on 2026-11-22: see
#: the fixture's `composition` fields; f32 noise of the towers (the vision tower's soft tokens move
#: by up to 3.3e-5 against upstream's padded run, the audio tower's by up to 1e-4 against
#: upstream's) reaches the vector shrunk by the text tower.
COMPOSITION_TOLERANCE = 1e-5


@dataclass(frozen=True)
class Case:
    """One whole input: its content parts in order (`("text", str)` / `("image", name)` /
    `("audio", name)`), the prompt name, and what it covers."""

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

#: The inputs holding an audio (texts neutral, waveforms from `waveforms.py`) — a list of their own
#: in the fixture (`audioCases`), so the image cases' list stays as the image pipeline reads it.
AUDIO_CASES: tuple[Case, ...] = (
    Case("audio-only", (("audio", "sweep"),), "an audio alone (99 frames, L = 3 mod 4)"),
    Case(
        "text-then-audio",
        (("text", "a single click in silence"), ("audio", "click")),
        "a text, then an audio (100 frames, L = 0 mod 4)",
    ),
    Case(
        "query-audio",
        (("audio", "edge"),),
        "an audio with a prompt's prefix (102 frames, L = 2 mod 4: the tower's frame mask matters)",
        prompt="query",
    ),
    Case(
        "image-then-audio",
        (("image", "gradient-square"), ("audio", "sweep")),
        "an image, then an audio: both towers in one input",
    ),
)

#: The waveforms the audio cases use (the fixture carries their samples and frames).
FIXTURE_WAVEFORMS: tuple[str, ...] = tuple(
    dict.fromkeys(value for case in AUDIO_CASES for kind, value in case.parts if kind == "audio")
)


class MultimodalReferenceError(ValueError):
    """What upstream ran is not what the fixture describes."""


def upstream_input(case: Case) -> dict[str, Any]:
    """The case as sentence-transformers takes it: one dict, its keys in the parts' order, an
    image key holding a list when the case has two images, an audio as its f32 samples with their
    rate."""
    from PIL import Image

    def value_of(kind: str, value: str) -> Any:
        if kind == "image":
            return Image.fromarray(IMAGES[value].rgb8(), mode="RGB")
        if kind == "audio":
            return {"array": WAVEFORMS[value].f32(), "sampling_rate": SAMPLE_RATE}
        return value

    kinds = [kind for kind, _ in case.parts]
    if len(set(kinds)) != len(kinds) and set(kinds) != {"image"}:
        raise MultimodalReferenceError(f"{case.name}: a repeated text part has no dict form")
    if set(kinds) == {"image"} and len(kinds) > 1:
        return {"image": [value_of(kind, value) for kind, value in case.parts]}
    return {kind: value_of(kind, value) for kind, value in case.parts}


def soft_blocks(ids: Sequence[int], begin: int, soft: int, end: int) -> list[tuple[int, int]]:
    """`(start, count)` of each run of `soft` ids, each between one `begin` and one `end` (an
    image's `<|image>` … `<image|>`, an audio's `<|audio>` … `<audio|>`)."""
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
                f"the soft run at {start} is not between its begin and end ids"
            )
        blocks.append((start, index - start))
    return blocks


#: The composite model's inputs a case's kinds of parts make it receive.
IMAGE_INPUTS = ("pixel_values", "image_position_ids")
AUDIO_INPUTS = ("input_features", "input_features_mask")


@dataclass(frozen=True)
class UpstreamRun:
    """What the composite model received and what `encode` returned for one case (the image
    or audio inputs `None` when the case holds none)."""

    input_ids: torch.Tensor
    pixel_values: torch.Tensor | None
    position_ids: torch.Tensor | None
    input_features: torch.Tensor | None
    input_features_mask: torch.Tensor | None
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
        for key in ("input_ids", "attention_mask", *IMAGE_INPUTS, *AUDIO_INPUTS):
            if kwargs.get(key) is not None:
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
    kinds = {kind for kind, _ in case.parts}
    expected = {
        "input_ids",
        "attention_mask",
        *(IMAGE_INPUTS if "image" in kinds else ()),
        *(AUDIO_INPUTS if "audio" in kinds else ()),
    }
    if set(caught) != expected:
        raise MultimodalReferenceError(f"{case.name}: the model received {sorted(caught)}")
    if int(caught["input_ids"].shape[0]) != 1 or not bool(caught["attention_mask"].all()):
        raise MultimodalReferenceError(f"{case.name}: the run was batched or padded")
    return UpstreamRun(
        input_ids=caught["input_ids"][0].to(torch.int64),
        pixel_values=caught.get("pixel_values"),
        position_ids=caught.get("image_position_ids"),
        input_features=caught.get("input_features"),
        input_features_mask=caught.get("input_features_mask"),
        embedding=vector.to(torch.float32),
    )


def compose(
    tower: export_vision.VisionTowerWrapper,
    text: export.EmbeddingWrapper,
    run: UpstreamRun,
    blocks: Sequence[tuple[int, int]],
    audio_tower: export_audio.AudioTowerWrapper | None = None,
    audio_blocks: Sequence[tuple[int, int]] = (),
) -> torch.Tensor:
    """Karume's graphs in torch over what upstream ran (module doc, "The composition check")."""
    hidden = int(text.model.config.hidden_size)
    soft: list[tuple[int, torch.Tensor]] = []
    for index, (start, count) in enumerate(blocks):
        if run.pixel_values is None or run.position_ids is None:
            raise MultimodalReferenceError("image blocks without the image inputs")
        rows = export_vision.host_rows(
            run.pixel_values[index : index + 1], run.position_ids[index : index + 1]
        )
        if rows.soft_tokens != count:
            raise MultimodalReferenceError(
                f"image {index}: {rows.soft_tokens} soft tokens, {count} placeholders"
            )
        with torch.no_grad():
            soft.append((start, tower(rows.pixels, rows.position_x, rows.position_y)[0]))
    for index, (start, count) in enumerate(audio_blocks):
        features, mask = run.input_features, run.input_features_mask
        if audio_tower is None or features is None or mask is None:
            raise MultimodalReferenceError("audio blocks without the audio inputs or tower")
        frames = export_audio.host_frames(features[index : index + 1], mask[index : index + 1])
        if frames.soft_tokens != count:
            raise MultimodalReferenceError(
                f"audio {index}: {frames.soft_tokens} soft tokens, {count} placeholders"
            )
        with torch.no_grad():
            soft.append((start, audio_tower(frames.features, frames.frame_mask)[0]))
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


def assert_upstream_frames(name: str, run: UpstreamRun, audio: export_audio.ProcessedAudio) -> None:
    """The frames upstream's processor gave the composite model are the feature extractor's own
    run on the same samples, bit for bit (so the fixture's frames are what upstream ran)."""
    if run.input_features is None or run.input_features_mask is None:
        raise MultimodalReferenceError(f"{name}: the model received no audio frames")
    if not (
        torch.equal(run.input_features, audio.input_features)
        and torch.equal(run.input_features_mask, audio.input_features_mask)
    ):
        raise MultimodalReferenceError(
            f"{name}: upstream's processor gave other frames than its feature extractor"
        )


def measure(directory: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Run every case through upstream and through Karume's graphs; returns the fixture's JSON
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
        blocks = soft_blocks(ids, begin, soft, end)
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
    audio_begin, audio_soft, audio_end = (
        int(config.boa_token_id),
        int(config.audio_token_id),
        int(config.eoa_token_index),
    )
    audio_tower = export_audio.AudioTowerWrapper(export_audio.load_tower(directory)).eval()
    extractor = export_audio.load_feature_extractor(directory)
    audios = {
        name: export_audio.process_audio(extractor, WAVEFORMS[name]) for name in FIXTURE_WAVEFORMS
    }
    for name, audio in audios.items():
        tensors[f"audio.{name}"] = torch.from_numpy(audio.waveform.f32())
        tensors[f"audio.{name}.features"] = audio.input_features[0, : audio.frames.valid].clone()
    audio_cases: list[dict[str, Any]] = []
    for case in AUDIO_CASES:
        run = run_upstream(upstream, case)
        ids = run.input_ids.tolist()
        if ids[0] != int(config.text_config.bos_token_id) or ids[-1] != int(
            config.text_config.eos_token_id
        ):
            raise MultimodalReferenceError(f"{case.name}: the ids do not run <bos> … <eos>")
        blocks = soft_blocks(ids, begin, soft, end)
        audio_blocks = soft_blocks(ids, audio_begin, audio_soft, audio_end)
        images = [value for kind, value in case.parts if kind == "image"]
        (waveform,) = [value for kind, value in case.parts if kind == "audio"]
        expected = (
            [counts[name].soft_tokens for name in images],
            [audios[waveform].soft_tokens],
        )
        found = ([count for _, count in blocks], [count for _, count in audio_blocks])
        if found != expected:
            raise MultimodalReferenceError(
                f"{case.name}: soft runs {blocks} / {audio_blocks}, the inputs give {expected}"
            )
        assert_upstream_frames(case.name, run, audios[waveform])
        composed = compose(tower, text, run, blocks, audio_tower, audio_blocks)
        difference = _difference(composed, run.embedding)
        if difference["maxAbs"] > COMPOSITION_TOLERANCE:
            raise MultimodalReferenceError(
                f"{case.name}: Karume's graphs differ from upstream by {difference}"
            )
        tensors[f"{case.name}.input_ids"] = run.input_ids.to(torch.int32)
        tensors[f"{case.name}.embedding"] = run.embedding.contiguous()
        audio_cases.append(
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
                "audioBlocks": [
                    {"audio": waveform, "start": start, "softTokens": count}
                    for start, count in audio_blocks
                ],
                "composition": difference,
            }
        )
    document = {
        "_doc": [
            "Reference vectors of whole inputs holding an image or an audio, for the EmbeddingGemma"
            " 2 pipeline (generated by"
            " tools/export-recipes/embeddinggemma2/multimodal_reference.py).",
            "Upstream: SentenceTransformer of the pinned snapshot, f32 on the CPU; each case is"
            " one `encode` call of an upstream dict input (its keys in the parts' order; an audio"
            " as its f32 samples at 16 kHz), with the prompt name when the case has one.",
            "The tensors of multimodal-reference.safetensors: `image.<name>` U8 [height, width, 3]"
            " (row-major RGB8, what upstream's processor read), `audio.<name>` F32 [samples] (16"
            " kHz mono, what upstream's processor read), `audio.<name>.features` F32 [L, 128] (the"
            " valid log-mel frames upstream's feature extractor made of them; its frames after"
            " them are zeros), `<case>.input_ids` I32 [T] (the ids the composite model received,"
            " `<bos>` and `<eos>` included, each image as `<|image>` + n x `<|image|>` +"
            " `<image|>`, each audio as `<|audio>` + m x `<|audio|>` + `<audio|>`),"
            " `<case>.embedding` F32 [768] (the normalised vector `encode` returned).",
            "`cases` hold images only; `audioCases` hold an audio (and an image in one of them).",
            "`composition` is Karume's graphs composed in torch f32 (the tower wrappers' soft"
            " tokens at the placeholder rows of the text wrapper) against upstream's vector.",
            "The images and waveforms are synthetic and neutral"
            " (tools/export-recipes/embeddinggemma2/images.py, waveforms.py); do not add others.",
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
        "audioTokens": {"begin": audio_begin, "soft": audio_soft, "end": audio_end},
        "audios": [
            {
                "name": name,
                "why": audio.waveform.why,
                "samples": audio.waveform.samples,
                "sampleRate": SAMPLE_RATE,
                "extractorFrames": int(audio.input_features.shape[1]),
                "validFrames": audio.frames.valid,
                "softTokens": audio.soft_tokens,
            }
            for name, audio in audios.items()
        ],
        "audioCases": audio_cases,
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
                "cases": {
                    case["name"]: case["composition"]
                    for case in (*document["cases"], *document["audioCases"])
                },
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
