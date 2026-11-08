"""`gemma4/export_vision.py` (12B's encoder-free vision embedder) on a tiny random checkpoint.

The tiny embedder has upstream's structure in small numbers (rows of 4 × 4 × 3 pixels, hidden 64,
16 positions), on `gemma4.tests.test_export_unified`'s tiny 12B checkpoint, with its vision keys
named as the real checkpoint names them. What is fixed here are the properties that would otherwise
pass silently:

- the export wrapper equals upstream's module bit for bit in F32 on the valid rows (the grouping of
  the two position lookups included), and the graph has a symbolic row count;
- only the vision keys are read (never the audio ones), the checkpoint's structure is checked, and
  the linears and position tables are stored as the checkpoint's bf16 values;
- the wrapper's forward with an image and path A's masks equals upstream's whole model with
  `create_masks_for_vision_model`, and path B's equals upstream's plain forward;
- the prompt is upstream's processor's (on the real tokenizer, when its snapshot is in the cache);
- the series run publishes the container, the embedder goldens, greedy goldens equal to an
  independent run, the measurement of the two mask paths and the record — and a wrong pin stops it
  before anything is written.

transformers is required (`importorskip`).
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from safetensors import safe_open

from _shared import decode_series as shared
from _shared.container_read import read_provenance, read_stored
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from gemma4 import export_unified as unified
from gemma4 import export_vision as vision
from gemma4 import provenance, sources
from gemma4.distribution import (
    GEMMA4_POSITION_X,
    GEMMA4_POSITION_Y,
    GEMMA4_VISION_PIXELS,
    GEMMA4_VISION_SYMBOL,
)
from gemma4.tests.test_export_unified import (
    HIDDEN,
    SUPPRESSED,
    VOCAB,
    WINDOW,
    _pin,
    _text_config,
    write_checkpoint,
)
from karume.container import container_parts
from karume.dist import DistError

transformers = pytest.importorskip("transformers")

PATCH = 2
POOLING = 2
ROW_WIDTH = (PATCH * POOLING) ** 2 * 3
POSITIONS = 16
BUDGET = 70

#: The stand-in image ids, inside the tiny vocabulary and outside its suppressed ids.
BEGIN_IMAGE, IMAGE, END_IMAGE = VOCAB - 6, VOCAB - 5, VOCAB - 4

#: The tiny images (height, width): 24 × 40 pixels after the size rule — 60 rows, past the tiny
#: sliding window, so path A's window and the block both matter.
IMAGE_SIZE = (20, 30)
SOFT_TOKENS = 60

WEIGHTS_REVISION = "0" * 40
TOKENIZER_REVISION = "1" * 40


def _vision_config() -> Any:
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedVisionConfig,
    )

    return Gemma4UnifiedVisionConfig(
        patch_size=PATCH,
        pooling_kernel_size=POOLING,
        mm_embed_dim=HIDDEN,
        mm_posemb_size=POSITIONS,
        output_proj_dims=HIDDEN,
    )


def _vision_tensors() -> dict[str, torch.Tensor]:
    """The embedder's keys as the real checkpoint names them, BF16."""
    generator = torch.Generator().manual_seed(7)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to(torch.bfloat16)

    return {
        "model.vision_embedder.patch_ln1.weight": 1 + 0.1 * random(ROW_WIDTH),
        "model.vision_embedder.patch_ln1.bias": 0.1 * random(ROW_WIDTH),
        "model.vision_embedder.patch_dense.weight": 0.3 * random(HIDDEN, ROW_WIDTH),
        "model.vision_embedder.patch_dense.bias": 0.1 * random(HIDDEN),
        "model.vision_embedder.patch_ln2.weight": 1 + 0.1 * random(HIDDEN),
        "model.vision_embedder.patch_ln2.bias": 0.1 * random(HIDDEN),
        "model.vision_embedder.pos_embedding": 0.2 * random(POSITIONS, 2, HIDDEN),
        "model.vision_embedder.pos_norm.weight": 1 + 0.1 * random(HIDDEN),
        "model.vision_embedder.pos_norm.bias": 0.1 * random(HIDDEN),
        "model.embed_vision.embedding_projection.weight": 0.3 * random(HIDDEN, HIDDEN),
    }


def write_vision_checkpoint(
    directory: Path, *, extra: dict[str, torch.Tensor] | None = None
) -> Path:
    """{@link write_checkpoint}'s tiny 12B plus the whole embedder, its processor configuration and
    the top-level image ids."""
    write_checkpoint(directory, extra={**_vision_tensors(), **(extra or {})})
    config = transformers.Gemma4UnifiedConfig(
        text_config=_text_config().to_dict(),
        vision_config=_vision_config().to_dict(),
        image_token_id=IMAGE,
        boi_token_id=BEGIN_IMAGE,
        eoi_token_id=END_IMAGE,
    )
    config.save_pretrained(directory)
    (directory / "processor_config.json").write_text(
        json.dumps(
            {
                "image_processor": {
                    "max_soft_tokens": BUDGET,
                    "patch_size": PATCH,
                    "pooling_kernel_size": POOLING,
                }
            }
        ),
        encoding="utf-8",
    )
    return directory


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory) -> Path:
    return write_vision_checkpoint(tmp_path_factory.mktemp("tiny-12b-vision"))


@pytest.fixture(scope="module")
def config(checkpoint_dir) -> Any:
    return vision.load_config(checkpoint_dir)


@pytest.fixture(scope="module")
def embedder(checkpoint_dir, config) -> Any:
    return vision.upstream_embedder(config, vision.read_state(checkpoint_dir))


@pytest.fixture(scope="module")
def wrapper(embedder) -> vision.VisionEmbedderWrapper:
    return vision.VisionEmbedderWrapper(embedder).requires_grad_(False).eval()


def _rows(rows: int, seed: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Random valid rows as the processor gives them: `pixels[N,P]` in [0, 1], `positions[N,2]`."""
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.rand(rows, ROW_WIDTH, generator=generator)
    positions = torch.randint(0, POSITIONS, (rows, 2), generator=generator)
    return pixels, positions


def _image(rows: int) -> vision.ProcessedImage:
    """Valid rows and the padded batch upstream's processor would make of them."""
    pixels, positions = _rows(rows)
    padded_pixels = torch.zeros(1, BUDGET, ROW_WIDTH)
    padded_pixels[0, :rows] = pixels
    padded_positions = torch.full((1, BUDGET, 2), -1, dtype=torch.int64)
    padded_positions[0, :rows] = positions
    return vision.ProcessedImage(padded_pixels, padded_positions, pixels, positions)


class TestWrapper:
    @pytest.mark.parametrize("rows", [1, 3, 17])
    def test_it_equals_upstream_s_module_bit_for_bit_on_the_valid_rows(
        self, embedder, wrapper, rows
    ) -> None:
        """Both on the valid rows alone, as the host runs the graph."""
        image = _image(rows)

        got = vision.embed_wrapper(wrapper, image)
        want = vision.embed_upstream_valid(embedder, image)

        assert got.dtype is torch.float32 and got.shape == (rows, HIDDEN)
        assert torch.equal(got, want)

    @pytest.mark.parametrize("rows", [1, 3, 17])
    def test_upstream_s_padding_rows_move_the_valid_rows_by_rounding_only(
        self, embedder, rows
    ) -> None:
        """Upstream runs the padded rows (−1, −1) as the processor hands them over: their positions
        are zeroed and they are stripped afterwards. The matrix product may round differently by
        the row count (module docstring), so this is a bound, not an equality."""
        image = _image(rows)

        worst = vision.padding_difference(
            vision.embed_upstream_valid(embedder, image),
            vision.embed_upstream(embedder, image),
            "rows",
        )

        assert worst <= vision.PADDING_BOUND

    def test_a_mishandled_padding_is_past_the_bound(self, embedder) -> None:
        """The bound can fail: rows whose positions are not the padded batch's move completely."""
        image = _image(3)
        shifted = vision.ProcessedImage(
            image.padded_pixels,
            image.padded_positions,
            image.pixels,
            torch.flip(image.positions, dims=(1,)),
        )

        with pytest.raises(vision.VisionExportError, match="padding rows move"):
            vision.padding_difference(
                vision.embed_upstream_valid(embedder, shifted),
                vision.embed_upstream(embedder, image),
                "rows",
            )

    def test_the_grouping_of_the_two_lookups_is_what_the_equality_rests_on(
        self, embedder, wrapper
    ) -> None:
        """`(hidden + x) + y` differs from upstream's `hidden + (x + y)` in F32, so the equality
        above can fail: it would catch the other grouping."""
        pixels, positions = _rows(40)
        with torch.no_grad():
            hidden = wrapper.patch_ln2(wrapper.patch_dense(wrapper.patch_ln1(pixels.unsqueeze(0))))
            x = torch.nn.functional.embedding(positions[:, 0].unsqueeze(0), wrapper.pos_x)
            y = torch.nn.functional.embedding(positions[:, 1].unsqueeze(0), wrapper.pos_y)
            other = (hidden + x) + y
            ours = hidden + (x + y)

        assert not torch.equal(other, ours)

    def test_the_position_tables_are_the_columns_of_upstream_s(self, embedder, wrapper) -> None:
        assert torch.equal(wrapper.pos_x, embedder.pos_embedding[:, 0])
        assert torch.equal(wrapper.pos_y, embedder.pos_embedding[:, 1])

    def test_a_difference_in_one_bit_fails_the_gate(self) -> None:
        want = torch.ones(2, 3)
        got = want.clone()
        got[1, 2] = torch.nextafter(got[1, 2], torch.tensor(2.0))

        vision.assert_wrapper_equals(want, want.clone(), "x")
        with pytest.raises(vision.VisionExportError, match="x: the wrapper differs"):
            vision.assert_wrapper_equals(got, want, "x")


class TestCheckpoint:
    def test_only_the_vision_keys_are_read_under_upstream_s_names(self, checkpoint_dir) -> None:
        state = vision.read_state(checkpoint_dir)

        assert sorted(state) == sorted(
            [
                "patch_ln1.weight",
                "patch_ln1.bias",
                "patch_dense.weight",
                "patch_dense.bias",
                "patch_ln2.weight",
                "patch_ln2.bias",
                "pos_embedding",
                "pos_norm.weight",
                "pos_norm.bias",
                "multimodal_embedder.embedding_projection.weight",
            ]
        )
        assert all(tensor.dtype is torch.float32 for tensor in state.values())
        assert not [key for key in vision.vision_keys(checkpoint_dir) if "audio" in key]

    def test_the_prefixes_are_the_two_the_text_export_leaves_out(self) -> None:
        assert set(vision.VISION_PREFIXES) < set(unified.IGNORED_PREFIXES)
        assert "model.embed_audio." not in vision.VISION_PREFIXES

    def test_a_key_the_module_does_not_have_fails_loudly(self, tmp_path, config) -> None:
        directory = write_vision_checkpoint(
            tmp_path,
            extra={"model.vision_embedder.extra.weight": torch.zeros(2, dtype=torch.bfloat16)},
        )

        with pytest.raises(RuntimeError, match="extra"):
            vision.upstream_embedder(config, vision.read_state(directory))

    def test_a_config_without_the_embedder_is_refused(self, tmp_path) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "gemma4_unified", "vision_config": None}), encoding="utf-8"
        )

        with pytest.raises(vision.VisionExportError, match="no encoder-free"):
            vision.load_config(tmp_path)


class TestGraph:
    def test_the_graph_is_the_same_for_any_row_count(self, wrapper, config) -> None:
        """Two example row counts trace to one graph, so `N` is a symbol, not a baked number."""
        three, _ = vision.trace(wrapper, 3)
        nine, _ = vision.trace(wrapper, 9)

        assert three.to_json() == nine.to_json()
        assert list(three.symbols) == [GEMMA4_VISION_SYMBOL]

    def test_the_form_is_the_contract_with_the_host(self, wrapper, config) -> None:
        graph, _ = vision.trace(wrapper)

        form = vision.assert_ir_form_vision(graph, config, stored=False)

        assert form["inputs"] == [
            [GEMMA4_VISION_PIXELS, "f32", [1, GEMMA4_VISION_SYMBOL, ROW_WIDTH]],
            [GEMMA4_POSITION_X, "i32", [1, GEMMA4_VISION_SYMBOL]],
            [GEMMA4_POSITION_Y, "i32", [1, GEMMA4_VISION_SYMBOL]],
        ]
        assert form["ops"] == {
            "layer_norm": 3,
            "linear": 2,
            "embedding": 2,
            "add": 2,
            "rms_norm": 1,
        }
        assert list(graph.values[graph.outputs[0]].shape) == [1, GEMMA4_VISION_SYMBOL, HIDDEN]

    def test_the_form_check_catches_another_input_and_output(self, wrapper, config) -> None:
        graph, _ = vision.trace(wrapper)
        wrong_input = copy.deepcopy(graph)
        wrong_input.inputs[1], wrong_input.inputs[2] = wrong_input.inputs[2], wrong_input.inputs[1]
        wrong_output = copy.deepcopy(graph)
        wrong_output.values[wrong_output.outputs[0]].shape[2] = HIDDEN + 1

        with pytest.raises(vision.VisionExportError, match="graph inputs"):
            vision.assert_ir_form_vision(wrong_input, config, stored=False)
        with pytest.raises(vision.VisionExportError, match="the output is"):
            vision.assert_ir_form_vision(wrong_output, config, stored=False)

    def test_the_container_stores_the_checkpoint_s_values_as_bf16(
        self, checkpoint_dir, wrapper, config, tmp_path
    ) -> None:
        graph, tensors = vision.trace(wrapper)
        path = tmp_path / "model.krm"
        verified = vision.write_container(graph, tensors, path, WEIGHTS_REVISION)
        stored = read_stored(path)

        form = vision.assert_ir_form_vision(verified, config, stored=True)

        assert form["storage"]["bf16"] == len(vision.BF16_WEIGHTS)
        assert set(form["storage"]) == {"bf16", "f32"}
        with safe_open(str(checkpoint_dir / "model.safetensors"), framework="pt") as handle:
            table = handle.get_tensor("model.vision_embedder.pos_embedding")
            dense = handle.get_tensor("model.vision_embedder.patch_dense.weight")
            projection = handle.get_tensor("model.embed_vision.embedding_projection.weight")
            norm = handle.get_tensor("model.vision_embedder.pos_norm.weight")

        def bf16_bytes(tensor: torch.Tensor) -> bytes:
            return tensor.contiguous().view(torch.int16).numpy().tobytes()

        assert stored["patch_dense.weight"].layout == "bf16"
        assert stored["patch_dense.weight"].payload == bf16_bytes(dense)
        assert stored["pos_x"].payload == bf16_bytes(table[:, 0])
        assert stored["pos_y"].payload == bf16_bytes(table[:, 1])
        key = "multimodal_embedder.embedding_projection.weight"
        assert stored[key].payload == bf16_bytes(projection)
        assert stored["pos_norm.weight"].layout == "f32"
        assert stored["pos_norm.weight"].payload == norm.to(torch.float32).numpy().tobytes()
        assert read_provenance(path).upstream_revision == WEIGHTS_REVISION

    def test_a_series_stored_in_f32_fails_the_form_check(self, wrapper, config, tmp_path) -> None:
        graph, tensors = vision.trace(wrapper)
        declared = {init.tensor for init in graph.initializers.values()}
        verified = vision.publish_model(
            tmp_path / "model.krm",
            graph,
            {name: tensor for name, tensor in tensors.items() if name in declared},
            provenance=vision.Provenance(license="apache-2.0"),
            graph_name=vision.GEMMA4_VISION_ROLE,
            weight_dtype="f32",
        )

        with pytest.raises(vision.VisionExportError, match="not stored as bf16"):
            vision.assert_ir_form_vision(verified, config, stored=True)


def full_model(checkpoint_dir: Path) -> Any:
    """Upstream's whole `Gemma4UnifiedForConditionalGeneration` over the tiny checkpoint in F32,
    eager, through its own key conversion."""
    return (
        transformers.Gemma4UnifiedForConditionalGeneration.from_pretrained(
            checkpoint_dir, dtype=torch.float32, attn_implementation="eager"
        )
        .eval()
        .requires_grad_(False)
    )


def _image_ids(length: int, block: tuple[int, int]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(length)
    ids = torch.randint(0, BEGIN_IMAGE, (1, length), generator=generator, dtype=torch.int64)
    ids[0, block[0] - 1] = BEGIN_IMAGE
    ids[0, block[0] : block[1]] = IMAGE
    ids[0, block[1]] = END_IMAGE
    return ids


class TestForwardWithAnImage:
    """The wrapper against upstream's whole model, F32 bound as in
    `test_export_unified.TestUpstreamParity`."""

    BOUND = 2e-5
    BLOCK = (3, 15)
    LENGTH = 22

    @pytest.fixture
    def case(self, checkpoint_dir) -> SimpleNamespace:
        from gemma4.tests.test_export_unified import naive_wrapper

        text = unified.load_config(checkpoint_dir)
        ids = _image_ids(self.LENGTH, self.BLOCK)
        pixels, positions = _rows(self.BLOCK[1] - self.BLOCK[0], seed=11)
        return SimpleNamespace(
            ids=ids,
            pixels=pixels,
            positions=positions,
            text=text,
            wrapper=naive_wrapper(checkpoint_dir, text),
            upstream=full_model(checkpoint_dir),
            specs=decode.rope_specs(text),
        )

    def _ours(self, case: SimpleNamespace, masks: Any) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = case.upstream.model.get_image_features(
            case.pixels.unsqueeze(0), case.positions.unsqueeze(0), return_dict=True
        ).pooler_output
        with torch.no_grad():
            return case.wrapper.run(
                case.ids,
                *unified.image_soft_inputs(case.ids, embeds, self.BLOCK),
                *decode.rope_args(case.specs, decode.positions_for(case.ids)),
                torch.arange(self.LENGTH),
                masks(self.LENGTH, WINDOW, self.BLOCK),
            )

    def _mm(self, case: SimpleNamespace) -> torch.Tensor:
        return (case.ids == IMAGE).to(torch.int64)

    def test_path_a_is_upstream_s_create_masks_for_vision_model(self, case) -> None:
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            create_masks_for_vision_model,
            get_block_sequence_ids_for_mask,
        )

        inputs_embeds = case.upstream.get_input_embeddings()(case.ids.clamp(max=VOCAB - 1))
        block_ids = get_block_sequence_ids_for_mask(self._mm(case), inputs_embeds.device)
        masks = create_masks_for_vision_model(
            config=case.upstream.config.get_text_config(),
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            position_ids=decode.positions_for(case.ids),
            block_sequence_ids=block_ids,
        )
        ours = unified.image_masks(self.LENGTH, WINDOW, self.BLOCK)

        for kind, mask in masks.items():
            assert torch.equal(ours[kind][0, 0] == 0, _keeps(mask)), kind

        logits, _ = self._ours(case, unified.image_masks)
        with torch.no_grad():
            expected = case.upstream(
                input_ids=case.ids,
                pixel_values=_padded(case.pixels),
                image_position_ids=_padded_positions(case.positions),
                mm_token_type_ids=self._mm(case),
                attention_mask=masks,
                use_cache=False,
            ).logits

        assert logits.shape == expected.shape
        assert float((logits - expected).abs().max()) <= self.BOUND

    def test_path_b_is_upstream_s_plain_forward(self, case) -> None:
        logits, _ = self._ours(case, unified.image_masks_or_block)
        with torch.no_grad():
            expected = case.upstream(
                input_ids=case.ids,
                pixel_values=_padded(case.pixels),
                image_position_ids=_padded_positions(case.positions),
                mm_token_type_ids=self._mm(case),
                use_cache=False,
            ).logits

        assert float((logits - expected).abs().max()) <= self.BOUND

    def test_the_two_paths_differ_so_each_comparison_tells(self, case) -> None:
        a, _ = self._ours(case, unified.image_masks)
        b, _ = self._ours(case, unified.image_masks_or_block)

        assert float((a - b).abs().max()) > 10 * self.BOUND


def _keeps(mask: torch.Tensor) -> torch.Tensor:
    """Upstream's mask as the `[T,T]` pattern of what is kept (a boolean mask, or a float one with
    −inf / the dtype's minimum where blocked)."""
    square = mask[0, 0]
    return square if square.dtype is torch.bool else square == 0


def _padded(pixels: torch.Tensor) -> torch.Tensor:
    return torch.cat([pixels, torch.zeros(BUDGET - pixels.shape[0], ROW_WIDTH)]).unsqueeze(0)


def _padded_positions(positions: torch.Tensor) -> torch.Tensor:
    padding = torch.full((BUDGET - positions.shape[0], 2), -1, dtype=torch.int64)
    return torch.cat([positions, padding]).unsqueeze(0)


class TestMaskPaths:
    def test_identical_rows_compare_as_identical(self) -> None:
        row = torch.tensor([1.0, 3.0, 2.0, 9.0])

        measured = vision.compare_mask_paths(row, row.clone(), (3,))

        assert measured == {
            "maxAbsLogitDifference": 0.0,
            "klAB": 0.0,
            "top1Agrees": True,
            "top1A": 1,
            "top1B": 1,
        }

    def test_suppressed_ids_are_left_out(self) -> None:
        """A difference only on a suppressed id is no difference to a host, and a top-1 is never
        one."""
        a = torch.tensor([1.0, 3.0, 2.0, 9.0])
        b = torch.tensor([1.0, 3.0, 2.0, -9.0])

        measured = vision.compare_mask_paths(a, b, (3,))

        assert measured["maxAbsLogitDifference"] == 0.0
        assert measured["top1A"] == 1

    def test_a_top_1_flip_shows(self) -> None:
        a = torch.tensor([3.0, 1.0, 0.0])
        b = torch.tensor([1.0, 3.0, 0.0])

        measured = vision.compare_mask_paths(a, b, ())

        assert measured["maxAbsLogitDifference"] == pytest.approx(2.0)
        assert measured["klAB"] > 0
        assert measured["top1Agrees"] is False
        assert (measured["top1A"], measured["top1B"]) == (0, 1)


class _TinyTokenizer:
    """Stands in for the tokenizer of the tiny vocabulary (the real one is
    {@link TestRealPrompt}'s)."""

    PROMPT = (2, 7, IMAGE, 9, 11)

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|image>": BEGIN_IMAGE, "<|image|>": IMAGE, "<image|>": END_IMAGE}[token]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt) -> str:
        assert not tokenize and add_generation_prompt
        assert messages[0]["content"][1]["text"] == vision.PROMPT_TEXT
        return "<rendered>"

    def __call__(self, text: str, add_special_tokens: bool) -> dict[str, list[int]]:
        assert text == "<rendered>" and not add_special_tokens
        return {"input_ids": list(self.PROMPT)}

    def decode(self, tokens: list[int]) -> str:
        return " ".join(f"<{token}>" for token in tokens)


class _TinyProcessor:
    """Upstream's real image processor at the tiny sizes, the rest a stand-in with the same face."""

    boi_token, image_token, eoi_token = "<|image>", "<|image|>", "<image|>"

    def __init__(self, *, wrong_ids: bool = False) -> None:
        from transformers import Gemma4UnifiedImageProcessor

        self.image_processor = Gemma4UnifiedImageProcessor(
            patch_size=PATCH, pooling_kernel_size=POOLING, max_soft_tokens=BUDGET
        )
        self.tokenizer = _TinyTokenizer()
        self.wrong_ids = wrong_ids

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs) -> Any:
        image = messages[0]["content"][0]["image"]
        tokens = int(
            self.image_processor(images=[image], return_tensors="pt")["num_soft_tokens_per_image"][
                0
            ]
        )
        prompt = self.tokenizer.PROMPT
        ids = [*prompt[:2], BEGIN_IMAGE, *[IMAGE] * tokens, END_IMAGE, *prompt[3:]]
        if self.wrong_ids:
            ids[-1] = 1
        marked = [int(token == IMAGE) for token in ids]
        return {
            "input_ids": torch.tensor([ids]),
            "mm_token_type_ids": torch.tensor([marked]),
        }


@pytest.fixture
def tiny_images(tmp_path, monkeypatch) -> dict[str, Path]:
    """Three seeded PNGs in place of the corpus images."""
    from PIL import Image

    paths: dict[str, Path] = {}
    for seed, name in enumerate(("one", "two", "three")):
        rng = np.random.default_rng(seed)
        pixels = rng.integers(0, 256, (*IMAGE_SIZE, 3), dtype=np.uint8)
        path = tmp_path / "images" / f"{name}.png"
        path.parent.mkdir(exist_ok=True)
        Image.fromarray(pixels, "RGB").save(path)
        paths[name] = path
    monkeypatch.setattr(vision, "VISION_IMAGES", paths)
    monkeypatch.setattr(vision, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(vision, "load_processor", lambda tokenizer_dir: _TinyProcessor())
    return paths


class TestImages:
    def test_the_processor_output_is_split_into_valid_and_padding_rows(self, tiny_images) -> None:
        processor = _TinyProcessor()

        image = vision.process_image(processor, vision.load_image(tiny_images["one"]))

        assert image.soft_tokens == SOFT_TOKENS
        assert image.pixels.shape == (SOFT_TOKENS, ROW_WIDTH)
        assert image.pixels.dtype is torch.float32
        assert image.padded_pixels.shape == (1, BUDGET, ROW_WIDTH)
        assert image.positions.tolist()[:2] == [[0, 0], [1, 0]]
        assert bool((image.padded_positions[0, SOFT_TOKENS:] == -1).all())

    def test_the_prompt_expands_the_image_token_as_upstream_s_processor(
        self, tiny_images, config
    ) -> None:
        processor = _TinyProcessor()
        image = vision.load_image(tiny_images["one"])

        prompt = vision.image_prompt(processor, image, SOFT_TOKENS, config)

        start, stop = prompt.block
        ids = prompt.ids[0].tolist()
        assert stop - start == SOFT_TOKENS
        assert ids[start - 1] == BEGIN_IMAGE and ids[stop] == END_IMAGE
        assert set(ids[start:stop]) == {IMAGE}

    def test_a_prompt_upstream_s_processor_disagrees_with_fails_loudly(
        self, tiny_images, config
    ) -> None:
        image = vision.load_image(tiny_images["one"])

        with pytest.raises(vision.VisionExportError, match="differs from upstream's processor"):
            vision.image_prompt(_TinyProcessor(wrong_ids=True), image, SOFT_TOKENS, config)

    def test_ids_that_disagree_with_the_config_fail_loudly(self, tiny_images, config) -> None:
        other = copy.deepcopy(config)
        other.image_token_id = 1

        with pytest.raises(vision.VisionExportError, match=r"differ from config\.json"):
            vision.image_prompt(
                _TinyProcessor(), vision.load_image(tiny_images["one"]), SOFT_TOKENS, other
            )

    def test_missing_corpus_images_fail_before_anything_long(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(vision, "VISION_IMAGES", {"gone": tmp_path / "gone.png"})

        with pytest.raises(vision.VisionExportError, match="corpus images are missing"):
            vision._source_images()


class TestRealPrompt:
    def test_12b_s_prompt_is_upstream_s_processor_s(self, tmp_path) -> None:
        """The real tokenizer, chat template and processor (needs the pinned tokenizer snapshot; no
        weight is read): the block holds the soft tokens only, `<|image>` and `<image|>` outside."""
        from PIL import Image

        tokenizer_dir = sources.tokenizer_dir("12b")
        if not (tokenizer_dir / "processor_config.json").is_file():
            pytest.skip(f"12B's processor files are not in the hub cache ({tokenizer_dir})")
        processor = vision.load_processor(tokenizer_dir)
        rng = np.random.default_rng(0)
        image = Image.fromarray(rng.integers(0, 256, (480, 832, 3), dtype=np.uint8), "RGB")
        twelve_b = SimpleNamespace(boi_token_id=255999, image_token_id=258880, eoi_token_id=258882)

        processed = vision.process_image(processor, image)
        prompt = vision.image_prompt(processor, image, processed.soft_tokens, twelve_b)

        start, stop = prompt.block
        ids = prompt.ids[0].tolist()
        assert processed.soft_tokens == 264
        assert stop - start == 264
        assert ids[start - 1] == 255999 and ids[stop] == 258882
        assert set(ids[start:stop]) == {258880}
        assert processed.pixels.shape == (264, 6912)


class TestTakeGreedy:
    PROMPT = vision.ImagePrompt(ids=torch.tensor([[1, 2, 3]]), block=(1, 2))

    def _view(self, margins_by_call: list[float]) -> Any:
        """A view whose every call gives the next margin (top1 − top2)."""
        calls = iter(margins_by_call)

        class View:
            first_logits = torch.zeros(1)

            def __call__(self, input_ids, position_ids):
                margin = next(calls)
                return torch.tensor([[[margin, 0.0, -1.0]]]), None

        return View()

    def test_a_step_at_the_floor_drops_the_image_for_the_next(self) -> None:
        margins = {"one": [1.0, 0.5], "two": [1.0, 1.0], "three": [1.0, 1.0], "four": [9.0, 9.0]}
        views = {name: self._view(values) for name, values in margins.items()}
        order = [(name, self.PROMPT) for name in margins]

        kept, dropped = vision.take_greedy(
            lambda name, _prompt: views[name], order, steps=2, floor=0.5
        )

        assert [golden.name for golden in kept] == ["two", "three"]
        assert dropped == {"one": [1.0, 0.5]}

    def test_fewer_images_than_asked_fail_naming_every_offender(self) -> None:
        margins = {"one": [0.1, 1.0], "two": [1.0, 1.0], "three": [1.0, 0.1]}
        views = {name: self._view(values) for name, values in margins.items()}

        with pytest.raises(AssertionError, match=r"'one'.*'three'"):
            vision.take_greedy(
                lambda name, _prompt: views[name],
                [(name, self.PROMPT) for name in margins],
                steps=2,
                floor=0.5,
            )


def _two_sources(checkpoint_dir: Path, tmp_path: Path) -> SimpleNamespace:
    """12B's layout (`test_export_unified.two_sources`), with the tokenizer source holding the
    processor configuration."""
    weights = tmp_path / "sources" / "weights"
    tokenizer = tmp_path / "sources" / "tokenizer"
    for directory, names in (
        (weights, ("model.safetensors", "config.json")),
        (tokenizer, ("tokenizer.json", "generation_config.json", "processor_config.json")),
    ):
        directory.mkdir(parents=True)
        for name in names:
            shutil.copy(checkpoint_dir / name, directory / name)
    pins = sources.ModelPins(
        weights=_pin("test/weights", WEIGHTS_REVISION, weights, provenance.WEIGHTS_FILES),
        tokenizer=_pin("test/tokenizer", TOKENIZER_REVISION, tokenizer, provenance.TOKENIZER_FILES),
    )
    return SimpleNamespace(weights=weights, tokenizer=tokenizer, pins=pins)


class TestExportSeries:
    STEPS = 3

    @pytest.fixture
    def series(self, tiny_images, checkpoint_dir, tmp_path, monkeypatch) -> SimpleNamespace:
        monkeypatch.setattr(decode, "MARGIN_FLOOR", 0.0)
        two = _two_sources(checkpoint_dir, tmp_path)
        out_dir = tmp_path / "series" / "gemma4-12b-vision"
        summary = vision.export_series(
            two.weights,
            out_dir,
            tokenizer_dir=two.tokenizer,
            pins=two.pins,
            steps=self.STEPS,
            checkpoint_label="weights and tokenizer",
        )
        return SimpleNamespace(out_dir=out_dir, summary=summary, two=two, images=tiny_images)

    def test_it_publishes_the_container_the_goldens_and_the_record(self, series) -> None:
        names = sorted(path.name for path in series.out_dir.iterdir())

        assert names == sorted(
            [
                *(path.name for path in container_parts(series.out_dir / one_shot.MODEL_FILE)),
                *(f"vision.{name}.safetensors" for name in series.images),
                *(f"greedy.{name}.safetensors" for name in ("one", "two")),
                provenance.REFERENCE_FILE,
            ]
        )
        assert [path.name for path in series.out_dir.parent.iterdir()] == [series.out_dir.name]
        assert [stage["name"] for stage in series.summary["stages"]] == [
            "trace",
            "embedder-goldens",
            "goldens",
            "write",
            "provenance",
        ]
        assert series.summary["storage"]["bf16"] == len(vision.BF16_WEIGHTS)
        assert read_provenance(series.out_dir / one_shot.MODEL_FILE).upstream_revision == (
            WEIGHTS_REVISION
        )

    def test_the_embedder_goldens_are_upstream_s_processor_and_module(
        self, series, config, embedder
    ) -> None:
        processor = _TinyProcessor()
        record = json.loads(
            (series.out_dir / provenance.REFERENCE_FILE).read_text(encoding="utf-8")
        )["reference"]
        for name, path in series.images.items():
            image = vision.process_image(processor, vision.load_image(path))
            golden = series.out_dir / f"vision.{name}.safetensors"
            with safe_open(str(golden), framework="pt") as handle:
                assert handle.get_tensor(vision.PIXELS_SHAPE_KEY).tolist() == [
                    SOFT_TOKENS,
                    ROW_WIDTH,
                ]
                assert torch.equal(
                    handle.get_tensor(vision.POSITIONS_KEY), image.positions.to(torch.int32)
                )
                embeds = handle.get_tensor(vision.EMBEDS_KEY)
            assert torch.equal(embeds, vision.embed_upstream(embedder, image)), name
            entry = record["vision"][name]
            assert entry["softTokens"] == SOFT_TOKENS
            assert entry["file"] == golden.name
            digest = hashlib.sha256(image.pixels.numpy().tobytes()).hexdigest()
            assert entry["pixelsSha256"] == digest
            assert entry["source"]["sha256"] == provenance.file_digest(path)["sha256"]

    def test_the_greedy_goldens_are_an_independent_run_on_path_a(
        self, series, checkpoint_dir, embedder
    ) -> None:
        """An independent loop over the F32 load with the imported linears and embedding (the
        model the reference stands for), with its own masks and rows."""
        from gemma4.tests.test_export_unified import grid_loaded_wrapper, naive_wrapper

        text = unified.load_config(checkpoint_dir)
        loaded = grid_loaded_wrapper(
            checkpoint_dir, text, unified.trace_keys(naive_wrapper(checkpoint_dir, text)).int4
        )
        specs = decode.rope_specs(text)
        processor = _TinyProcessor()
        for name in ("one", "two"):
            image = vision.process_image(processor, vision.load_image(series.images[name]))
            embeds = vision.embed_upstream(embedder, image)
            with safe_open(
                str(series.out_dir / f"greedy.{name}.safetensors"), framework="pt"
            ) as handle:
                prompt = handle.get_tensor(shared.PROMPT_KEY)
                expected = handle.get_tensor(shared.EXPECTED_KEY).tolist()
                margins = handle.get_tensor(shared.MARGIN_KEY).tolist()
                block = tuple(handle.get_tensor(vision.BLOCK_KEY).tolist())
                first = handle.get_tensor(vision.FIRST_LOGITS_KEY)
            assert block == (3, 3 + SOFT_TOKENS)
            ids = prompt.unsqueeze(0).to(torch.int64)
            tokens: list[int] = []
            measured: list[float] = []
            for step in range(self.STEPS):
                length = int(ids.shape[1])
                soft = torch.zeros(1, length, HIDDEN)
                soft[0, block[0] : block[1]] = embeds
                mask = torch.zeros(1, length, 1, dtype=torch.bool)
                mask[0, block[0] : block[1]] = True
                with torch.no_grad():
                    logits, _ = loaded.run(
                        ids,
                        soft,
                        mask,
                        *decode.rope_args(specs, decode.positions_for(ids)),
                        torch.tensor([length - 1]),
                        unified.image_masks(length, WINDOW, block),
                    )
                if step == 0:
                    assert torch.equal(first, logits[0, 0])
                masked = logits[0, 0].clone()
                masked[list(SUPPRESSED)] = float("-inf")
                best = torch.topk(masked, 2)
                tokens.append(int(best.indices[0]))
                measured.append(float(best.values[0] - best.values[1]))
                ids = torch.cat([ids, best.indices[:1].unsqueeze(0)], dim=1)
            assert expected == tokens, name
            assert margins == pytest.approx(measured), name
            assert not set(expected) & set(SUPPRESSED), name

    def test_the_two_mask_paths_are_measured_in_the_record(self, series) -> None:
        record = json.loads(
            (series.out_dir / provenance.REFERENCE_FILE).read_text(encoding="utf-8")
        )
        paths = record["reference"]["maskPaths"]

        assert paths["image"] == "one"
        assert paths["maxAbsLogitDifference"] > 0
        assert paths["klAB"] >= 0
        assert isinstance(paths["top1Agrees"], bool)
        assert record["reference"]["greedyImages"]["one"]["block"] == [3, 3 + SOFT_TOKENS]
        assert record["reference"]["suppressTokens"] == sorted(SUPPRESSED)
        assert record["reference"]["softTokenBudget"] == BUDGET
        assert record["series"] == record["reference"]["series"] == series.out_dir.name
        assert record["checkpoint"]["dir"] == "weights and tokenizer"
        assert series.summary["mask_paths"] == paths

    def test_the_embedder_only_run_needs_no_reference(
        self, tiny_images, checkpoint_dir, tmp_path, monkeypatch
    ) -> None:
        def no_reference(_model_dir):
            raise AssertionError("the reference was built")

        monkeypatch.setattr(vision, "_reference", no_reference)
        out_dir = tmp_path / "embedder"

        summary = vision.export_series(
            checkpoint_dir, out_dir, tokenizer_dir=checkpoint_dir, pins=None, embedder_only=True
        )

        assert "greedy" not in summary
        assert not list(out_dir.glob(f"{shared.GREEDY_PREFIX}*"))
        assert not (out_dir / provenance.REFERENCE_FILE).exists()
        assert len(list(out_dir.glob("vision.*.safetensors"))) == len(tiny_images)

    def test_a_wrong_pin_stops_the_run_before_anything_is_written(
        self, tiny_images, checkpoint_dir, tmp_path
    ) -> None:
        two = _two_sources(checkpoint_dir, tmp_path)
        wrong = sources.ModelPins(
            weights=sources.HubPin(
                repo="test/weights",
                revision=WEIGHTS_REVISION,
                sha256={"config.json": "f" * 64},
            ),
            tokenizer=two.pins.tokenizer,
        )

        with pytest.raises(sources.SourcePinError, match=r"config\.json"):
            vision.export_series(
                two.weights,
                tmp_path / "series" / "gemma4-12b-vision",
                tokenizer_dir=two.tokenizer,
                pins=wrong,
                steps=1,
            )
        assert not (tmp_path / "series").exists()

    def test_a_processor_configuration_other_than_the_pinned_stops_the_run(
        self, tiny_images, checkpoint_dir, tmp_path
    ) -> None:
        """The goldens are taken at the processor configuration's budget: one the pin does not
        name stops the run before anything is written, with the pinned `tokenizer.json` beside
        it."""
        two = _two_sources(checkpoint_dir, tmp_path)
        pinned = two.pins.tokenizer
        wrong = sources.ModelPins(
            weights=two.pins.weights,
            tokenizer=sources.HubPin(
                repo=pinned.repo,
                revision=pinned.revision,
                sha256=pinned.sha256,
                also_checked={"processor_config.json": "f" * 64},
            ),
        )

        with pytest.raises(sources.SourcePinError, match=r"processor_config\.json"):
            vision.export_series(
                two.weights,
                tmp_path / "series" / "gemma4-12b-vision",
                tokenizer_dir=two.tokenizer,
                pins=wrong,
                steps=1,
            )
        assert not (tmp_path / "series").exists()

    def test_a_failed_gate_publishes_nothing(
        self, tiny_images, checkpoint_dir, tmp_path, monkeypatch
    ) -> None:
        def bent(got, want, where):
            raise vision.VisionExportError(f"{where}: bent")

        monkeypatch.setattr(vision, "assert_wrapper_equals", bent)
        out_dir = tmp_path / "series" / "gemma4-12b-vision"

        with pytest.raises(vision.VisionExportError, match="bent"):
            vision.export_series(checkpoint_dir, out_dir, tokenizer_dir=checkpoint_dir, pins=None)
        assert not out_dir.exists()


class TestPlan:
    def test_the_plan_reads_no_weight_and_writes_nothing(
        self, tiny_images, checkpoint_dir, tmp_path, monkeypatch
    ) -> None:
        def no_read(_model_dir):
            raise AssertionError("a weight was read")

        monkeypatch.setattr(vision, "read_state", no_read)

        summary = vision.plan(checkpoint_dir, tokenizer_dir=checkpoint_dir, pins=None)

        assert [name for name, _, _ in summary["inputs"]] == [
            GEMMA4_VISION_PIXELS,
            GEMMA4_POSITION_X,
            GEMMA4_POSITION_Y,
        ]
        assert summary["soft_token_budget"] == BUDGET
        assert summary["soft_tokens"] == dict.fromkeys(tiny_images, SOFT_TOKENS)
        assert summary["blocks"]["one"] == [3, 3 + SOFT_TOKENS]

    def test_the_plan_s_graph_is_the_real_weights_trace(self, config, wrapper) -> None:
        planned, _ = vision.trace(vision.VisionEmbedderWrapper(vision.upstream_embedder(config)))
        real, _ = vision.trace(wrapper)

        assert planned.to_json() == real.to_json()

    def test_keys_the_module_does_not_have_fail_the_plan(self, tmp_path, tiny_images) -> None:
        directory = write_vision_checkpoint(
            tmp_path / "keys",
            extra={"model.vision_embedder.extra.weight": torch.zeros(2, dtype=torch.bfloat16)},
        )

        with pytest.raises(vision.VisionExportError, match="extra"):
            vision.plan(directory, tokenizer_dir=directory, pins=None)

    def test_a_tokenizer_source_without_the_processor_configuration_is_refused(
        self, tmp_path
    ) -> None:
        with pytest.raises(DistError, match=r"processor_config\.json"):
            vision.load_processor(tmp_path)


class TestTheCommandLine:
    """`--embedder-only` writes a series without greedy goldens and `reference.json`, so it must
    not land where the distribution reads the full one."""

    @pytest.fixture
    def runs(self, monkeypatch) -> list[dict[str, Any]]:
        pytest.importorskip("huggingface_hub")
        calls: list[dict[str, Any]] = []

        def export_series(model_dir, out_dir, **options):
            calls.append({"out": out_dir, **options})
            return {}

        monkeypatch.setattr(vision, "export_series", export_series)
        return calls

    def test_the_canonical_series_is_the_one_the_distribution_reads(self, tmp_path) -> None:
        from gemma4.distribution import gemma4_qat_q4_0_sources

        pytest.importorskip("huggingface_hub")
        canonical = vision.canonical_series_dir("12b")

        assert gemma4_qat_q4_0_sources(canonical.parent, "12b").vision == canonical

    def test_embedder_only_refuses_the_canonical_series(self, runs, capsys) -> None:
        canonical = vision.canonical_series_dir("12b")
        # Another spelling of the same directory.
        spelled = canonical.parent / ".." / canonical.parent.name / canonical.name

        with pytest.raises(SystemExit):
            vision.main(["--model", "12b", "--embedder-only", "--out", str(spelled)])

        assert "canonical series" in capsys.readouterr().err
        assert runs == []

    def test_embedder_only_refuses_a_link_to_the_canonical_series(
        self, runs, capsys, tmp_path
    ) -> None:
        link = tmp_path / "link"
        link.symlink_to(vision.canonical_series_dir("12b"), target_is_directory=True)

        with pytest.raises(SystemExit):
            vision.main(["--model", "12b", "--embedder-only", "--out", str(link)])

        assert "canonical series" in capsys.readouterr().err
        assert runs == []

    def test_embedder_only_writes_another_directory(self, runs, tmp_path) -> None:
        vision.main(["--model", "12b", "--embedder-only", "--out", str(tmp_path / "diag")])

        assert [(run["out"], run["embedder_only"]) for run in runs] == [(tmp_path / "diag", True)]

    def test_the_full_run_writes_the_canonical_series(self, runs) -> None:
        vision.main(["--model", "12b"])

        assert [(run["out"], run["embedder_only"]) for run in runs] == [
            (vision.canonical_series_dir("12b"), False)
        ]
