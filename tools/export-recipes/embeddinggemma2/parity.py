"""Opt-in upstream parity of the EmbeddingGemma 2 wrapper against SentenceTransformer (text only).

    uv run --with 'transformers==5.23.0' --with 'sentence-transformers==6.1.0' \\
        python -m embeddinggemma2.parity

Not run by any default lane, and it writes nothing: it prints one JSON record. The export
({@link embeddinggemma2.export}) proves its wrapper equals the upstream text model's own forward on
every run; this script closes the remaining distance to what users run — `SentenceTransformer`
loaded from the pinned snapshot (the composite `EmbeddingGemma2Model` with its vision and audio
towers, the processor's tokenization, sentence-transformers' Pooling and Normalize).

For every golden case of the export it measures the cosine and the max abs difference of:

- `f32` — SentenceTransformer in f32 against the wrapper (the goldens' arithmetic);
- `bf16` — SentenceTransformer in bf16 (upstream's checkpoint dtype) against the f32 wrapper: the
  whole bf16 gap;
- `embed_scale` — the f32 wrapper with the embedding scale rounded as bf16 does (22.625 instead of
  `sqrt(512)` = 22.6274) against the f32 wrapper: the share of that gap the scale alone gives.

It also checks that the processor's tokenization of each case gives the golden ids, and measures
one input outside the goldens' domain: a text holding the image placeholder, where the composite
model substitutes the pad id and the text tower does not (the export's module docstring) — against
the wrapper as is, and against the wrapper fed the pad id in the placeholder's place.

SentenceTransformer loads more of the snapshot than the export reads: its processor needs
`processor_config.json`, `preprocessor_config.json` and `chat_template.jinja`, and it opens
`README.md`. Those files are outside the pin; the record lists their SHA-256 so a measurement can
be traced to the bytes it ran on.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from embeddinggemma2 import export, sources

#: The sentence-transformers release the snapshot declares (`config_sentence_transformers.json`).
REQUIRED_SENTENCE_TRANSFORMERS = "6.1.0"

#: The pinned files SentenceTransformer reads besides the export's.
PINNED_READS: tuple[str, ...] = (
    *export.READ_FILES,
    "tokenizer_config.json",
    "2_Normalize/config.json",
)

#: The snapshot files SentenceTransformer reads that are not pinned (their digests are recorded).
UNPINNED_READS: tuple[str, ...] = (
    "README.md",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
)

#: The embedding scale as bf16 rounds `sqrt(512)` (upstream running in its checkpoint dtype).
BF16_EMBED_SCALE = float(torch.tensor(512**0.5, dtype=torch.bfloat16))

#: The text outside the goldens' domain (the image placeholder among plain words).
PLACEHOLDER_TEXT = "task: search result | query: <|image|> a photo of a cat"


class ParityScriptError(ValueError):
    """An input of the script is not what the measurement is built for."""


def assert_sentence_transformers_version(version: str) -> None:
    """The sentence-transformers that is imported is {@link REQUIRED_SENTENCE_TRANSFORMERS}."""
    if version != REQUIRED_SENTENCE_TRANSFORMERS:
        raise ParityScriptError(
            f"sentence-transformers {version} is imported, the script needs"
            f" {REQUIRED_SENTENCE_TRANSFORMERS}: run `uv run --with"
            f" 'transformers=={export.REQUIRED_TRANSFORMERS}' --with"
            f" 'sentence-transformers=={REQUIRED_SENTENCE_TRANSFORMERS}' python -m"
            " embeddinggemma2.parity`"
        )


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Cosine and max abs difference of two embeddings (both read as f32 vectors)."""
    left = actual.to(torch.float32).reshape(-1)
    right = expected.to(torch.float32).reshape(-1)
    cosine = float(torch.dot(left, right) / (left.norm() * right.norm()))
    return {"cosine": cosine, "max_abs": float((left - right).abs().max())}


def summarize(per_case: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """The worst case of a comparison: the lowest cosine and the largest max abs difference."""
    return {
        "min_cosine": min(entry["cosine"] for entry in per_case.values()),
        "max_abs": max(entry["max_abs"] for entry in per_case.values()),
    }


def _load_sentence_transformer(directory: Path, dtype: torch.dtype) -> Any:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(directory), device="cpu", model_kwargs={"dtype": dtype})
    # The upstream default would truncate long cases silently.
    if model.max_seq_length is not None and model.max_seq_length < export.SYM_MAX:
        raise ParityScriptError(
            f"SentenceTransformer truncates at {model.max_seq_length} < {export.SYM_MAX} tokens"
        )
    return model.eval()


def _encode(model: Any, text: str) -> torch.Tensor:
    """One text at a time (no padding enters the batch)."""
    with torch.no_grad():
        return model.encode([text], convert_to_tensor=True, batch_size=1)[0]


def _assert_tokenization(model: Any, texts: Sequence[tuple[str, str]], cases: Any) -> None:
    golden = {name: ids for name, ids, _ in cases}
    for name, text in texts:
        ids = model.preprocess([text])["input_ids"]
        if not torch.equal(torch.as_tensor(ids).to(torch.int64), golden[name]):
            raise ParityScriptError(
                f"{name}: SentenceTransformer tokenizes to {ids}, the golden is {golden[name]}"
            )


def _with_embed_scale(model: torch.nn.Module, scale: float, run: Callable[[], Any]) -> Any:
    """Run `run` with the text tower's embedding scale set to `scale` (restored after)."""
    buffer = model.embed_tokens.embed_scale
    previous = buffer.clone()
    buffer.fill_(scale)
    try:
        return run()
    finally:
        buffer.copy_(previous)


def measure(directory: Path) -> dict[str, Any]:
    import sentence_transformers

    assert_sentence_transformers_version(sentence_transformers.__version__)
    sources.assert_pinned(directory, PINNED_READS)
    export.assert_sentence_transformer_layout(directory)
    model, composite = export.load_text_model(directory)
    wrapper = export.EmbeddingWrapper(model, int(model.config.sliding_window)).eval()
    cases = export.build_cases(directory, composite, export.SYM_MAX)
    texts = export.case_texts(directory)

    with torch.no_grad():
        reference = {name: wrapper(ids, pool) for name, ids, pool in cases}
        rounded_scale = _with_embed_scale(
            model,
            BF16_EMBED_SCALE,
            lambda: {name: wrapper(ids, pool) for name, ids, pool in cases},
        )

    record: dict[str, Any] = {
        "versions": {
            "transformers": export.REQUIRED_TRANSFORMERS,
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
        },
        "unpinned_reads": {name: sources.file_sha256(directory / name) for name in UNPINNED_READS},
        "embed_scale": {"f32": float(model.embed_tokens.embed_scale), "bf16": BF16_EMBED_SCALE},
    }
    record["embed_scale_only"] = {
        name: compare(rounded_scale[name], reference[name]) for name in reference
    }

    for label, dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
        upstream = _load_sentence_transformer(directory, dtype)
        _assert_tokenization(upstream, texts, cases)
        record[label] = {
            name: compare(_encode(upstream, text), reference[name]) for name, text in texts
        }
        if label == "f32":
            record["placeholder"] = _placeholder(upstream, model, wrapper, composite, directory)
        del upstream

    record["summary"] = {
        label: summarize(record[label]) for label in ("f32", "bf16", "embed_scale_only")
    }
    return record


def _placeholder(
    upstream: Any,
    model: torch.nn.Module,
    wrapper: export.EmbeddingWrapper,
    composite: Any,
    directory: Path,
) -> dict[str, Any]:
    """The image placeholder in a text, outside the goldens' domain.

    SentenceTransformer's processor refuses such a text when no image comes with it (recorded,
    not raised). The composite model under it, fed the token ids directly, substitutes the pad id;
    its output is compared with the wrapper as is and with the wrapper fed the pad id in the
    placeholder's place.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    ids = torch.tensor([tokenizer.encode(PLACEHOLDER_TEXT).ids], dtype=torch.int64)
    image = int(composite.image_token_id)
    if image not in ids[0].tolist():
        raise ParityScriptError(f"{PLACEHOLDER_TEXT!r} does not hold the image placeholder {image}")
    try:
        _encode(upstream, PLACEHOLDER_TEXT)
        refusal = None
    except ValueError as error:
        refusal = str(error)
    padded = torch.where(ids == image, int(model.config.pad_token_id), ids)
    pool = torch.ones_like(ids, dtype=torch.float32)
    with torch.no_grad():
        hidden = upstream[0].model(input_ids=ids, attention_mask=torch.ones_like(ids))
        expected = export._l2_normalize(export._masked_mean(hidden.last_hidden_state, pool))
        as_is = wrapper(ids, pool)
        substituted = wrapper(padded, pool)
    return {
        "text": PLACEHOLDER_TEXT,
        "sentence_transformer_refusal": refusal,
        "composite_vs_wrapper_as_is": compare(as_is, expected),
        "composite_vs_wrapper_with_pad_substituted": compare(substituted, expected),
    }


def main() -> None:
    record = measure(sources.SNAPSHOT_DIR)
    print(json.dumps(record, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
