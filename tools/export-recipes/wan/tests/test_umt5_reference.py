"""The contract of `wan/umt5_reference.py` (the layer-by-layer CPU reference that reads the weights
from the i8 series' container) (ADR 0119 decision 8 ①, step 10c).

What is pinned is the side where a break raises no exception and **only the reference values drift
quietly**:

- the weights read from the container (packed × scale) are bit-identical to the export's fake-quant
  values
- the f32 rewrite is bit-identical to the upstream eager (activation replaced by the tanh
  approximation, fake-quant applied) — a mix-up of the layers' tables splits it (the pair)
- the f64 rewrite creates no floating-point value that is not f64 — the upstream with `.double()`
  drops to f32 in RMSNorm and softmax, so it trips the same check (the pair)
- seen against the f64 reference, the mismatch between the upstream bf16 and f32 is almost entirely
  bf16 error (f32's own error is under 1% of it — the small-model backing for using f32 as the
  baseline in the quality record)
- the reader of the unquantized weights (the checkpoint's F32) is bit-identical to the upstream
  parameters, and the same rewrite is bit-identical to the unquantized upstream eager — it splits
  from the i8 reference (the pair)
- the shape of the golden (keys, dtypes, shapes, metadata), that the unquantized reference is
  acceptance only, and how the decision cases are chosen

It turns a small randomly initialized umT5 (`wan.umt5_probe.TINY_CONFIG`) into a container with
`umt5_export.prepare` and runs on it — no real weights are read.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from karume.pipeline import publish_model
from wan import umt5_export as ue
from wan import umt5_patch as up
from wan import umt5_probe as probe
from wan import umt5_reference as ur

LAYERS = int(probe.TINY_CONFIG["num_layers"])

#: The valid lengths compared (the lower bound, the fixed prompt's length, beyond max_distance 128,
#: the upper bound).
LENGTHS = (2, 28, 126, 300, 512)

#: The component name written (claimed only in pytest's working area — the module doc of
#: `wan/umt5_export.py`).
GRAPH_NAME = "text_encoder"

#: The upper bound of the f32 rewrite's ratio against f64 (f32's rounding digits — a mix-up in the
#: rewrite shows up at O(1)).
F32_ROUNDING_RATIO = 1e-5

#: The upper bound of the share of f32's own error in its mismatch with bf16 (a comparison of upper
#: bounds seen through f64).
F32_SHARE_OF_BF16_MISMATCH = 1e-2


@pytest.fixture(scope="module")
def model() -> Any:
    pytest.importorskip("transformers")
    return probe.tiny_model()


@pytest.fixture(scope="module")
def checkpoint(model: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The small model's checkpoint (`save_pretrained` — F32 safetensors)."""
    directory = tmp_path_factory.mktemp("umt5-reference") / "checkpoint"
    model.save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def container(checkpoint: Path) -> Path:
    """The small model's container written by `umt5_export.prepare` (i8 per block of rows) (its
    representative path)."""
    prepared = ue.prepare(checkpoint, chunk_rows=7)
    path = checkpoint.parent / "container" / ue.MODEL_FILE
    path.parent.mkdir()
    publish_model(
        path,
        prepared.graph,
        dict(prepared.tensors),
        provenance=ue.provenance(),
        graph_name=GRAPH_NAME,
        fixed_weights=prepared.fixed,
    )
    return path


@pytest.fixture(scope="module")
def weights(container: Path) -> ur.ContainerWeights:
    return ur.ContainerWeights(container)


@pytest.fixture(scope="module")
def plain(checkpoint: Path) -> ur.CheckpointWeights:
    """The reader of the unquantized weights (`umt5_export.checkpoint_weights` — assembled the same
    way as for the real model)."""
    return ue.checkpoint_weights(checkpoint)


@pytest.fixture(scope="module")
def quantized(model: Any) -> Any:
    """The fake-quant upstream (rounded through the wrapper — the wrapper shares the upstream
    modules)."""
    upstream = copy.deepcopy(model)
    up.fake_quant_i8(up.Umt5EncoderTokens(upstream))
    return upstream


def upstream_tanh(quantized: Any) -> Any:
    """A copy of `quantized` with only the FFN activation replaced by the tanh approximation (the
    shape of the upstream eager)."""
    upstream = copy.deepcopy(quantized)
    for block in upstream.encoder.block:
        block.layer[-1].DenseReluDense.act = nn.GELU(approximate="tanh")
    return upstream


def case_inputs(model: Any, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    attention = up.bucket_attention(model.config)
    return probe.tiny_ids(length), up.relative_position_buckets(length, attention)


def forward(
    source: ur.WeightSource, model: Any, dtype: torch.dtype, inputs: list[Any]
) -> list[torch.Tensor]:
    return ur.encode_layerwise(source, ur.EncoderShape.of(model.config), inputs, [dtype])[dtype]


def run(
    source: ur.WeightSource, model: Any, dtype: torch.dtype, lengths=LENGTHS
) -> list[torch.Tensor]:
    return forward(source, model, dtype, [case_inputs(model, length) for length in lengths])


class SwappedTables:
    """A reader that returns the relative-position tables of layers 0 and 1 swapped (fault
    injection)."""

    def __init__(self, source: ur.WeightSource) -> None:
        self.source = source
        first, second = ur.layer_keys(0)["relative_bias"], ur.layer_keys(1)["relative_bias"]
        self.swap = {first: second, second: first}

    def tensor(self, key: str) -> torch.Tensor:
        return self.source.tensor(self.swap.get(key, key))

    def rows(self, key: str, indices: torch.Tensor) -> torch.Tensor:
        return self.source.rows(key, indices)


class Preloaded:
    """A reader that has read the weights and the vocabulary embedding rows up front — no weight
    read (the container's f32 arithmetic) runs during the watch, so only the rewrite's arithmetic
    is watched."""

    def __init__(self, source: ur.WeightSource, lengths=LENGTHS) -> None:
        keys = [key for index in range(LAYERS) for key in ur.layer_keys(index).values()]
        self.tensors = {key: source.tensor(key) for key in [*keys, ur.FINAL_NORM_KEY]}
        ids = [probe.tiny_ids(length).flatten() for length in lengths]
        self.vocabulary = torch.unique(torch.cat(ids))
        self.table = source.rows(ur.EMBED_KEY, self.vocabulary)

    def tensor(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    def rows(self, key: str, indices: torch.Tensor) -> torch.Tensor:
        assert key == ur.EMBED_KEY and torch.equal(indices, self.vocabulary)
        return self.table


class NarrowFloatWatch(TorchDispatchMode):
    """Record the floating-point values that are not f64 created during this span (the same shape as
    the DiT's `_NarrowFloatWatch`)."""

    def __init__(self) -> None:
        super().__init__()
        self.found: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        result = func(*args, **(kwargs or {}))
        for leaf in tree_leaves(result):
            if (
                isinstance(leaf, torch.Tensor)
                and leaf.is_floating_point()
                and leaf.dtype != torch.float64
            ):
                self.found.append(f"{func}: {leaf.dtype}")
        return result


class TestContainerWeights:
    def test_every_weight_is_the_fake_quant_value(self, weights, quantized):
        """The container's weights (i8 as packed × scale) = the export's fake-quant values
        (bit-identical)."""
        expected = dict(up.Umt5EncoderTokens(quantized).named_parameters())

        mismatched = [
            key for key, value in expected.items() if not torch.equal(weights.tensor(key), value)
        ]

        # Per layer, the 10 of q / k / v / o, the relative-position table, the 2 RMSNorms and
        # wi_0 / wi_1 / wo, + the vocabulary embedding and the final norm.
        assert len(expected) == 10 * LAYERS + 2
        assert mismatched == []

    def test_rows_are_the_rows_of_the_whole_table(self, weights):
        indices = torch.tensor([0, 3, 17, 200, 383])

        rows = weights.rows(ur.EMBED_KEY, indices)

        assert torch.equal(rows, weights.tensor(ur.EMBED_KEY)[indices])

    def test_unsorted_rows_are_rejected(self, weights):
        with pytest.raises(ur.Umt5ReferenceError, match="ascending order"):
            weights.rows(ur.EMBED_KEY, torch.tensor([5, 3]))


class TestCheckpointWeights:
    def test_every_weight_is_the_checkpoint_value(self, plain, model):
        """The unquantized reader's weights = the upstream parameters (bit-identical, the tied
        vocabulary embedding included)."""
        expected = dict(up.Umt5EncoderTokens(model).named_parameters())

        mismatched = [
            key for key, value in expected.items() if not torch.equal(plain.tensor(key), value)
        ]

        assert len(expected) == 10 * LAYERS + 2
        assert mismatched == []

    def test_rows_are_the_rows_of_the_whole_table(self, plain):
        """With contiguous ranges and scattered rows mixed, the same rows of the table come back in
        the same order."""
        indices = torch.tensor([0, 1, 2, 17, 18, 200, 383])

        rows = plain.rows(ur.EMBED_KEY, indices)

        assert torch.equal(rows, plain.tensor(ur.EMBED_KEY)[indices])

    def test_rows_past_the_table_fail_loudly(self, plain):
        size = plain.tensor(ur.EMBED_KEY).shape[0]

        with pytest.raises(ur.Umt5ReferenceError, match="outside the table"):
            plain.rows(ur.EMBED_KEY, torch.tensor([0, size]))

    def test_an_unmapped_key_fails_loudly(self, plain):
        with pytest.raises(ur.Umt5ReferenceError, match="no checkpoint key"):
            plain.tensor("encoder.block.0.unknown.weight")


class TestUnquantizedReference:
    def test_f32_is_the_unquantized_upstream_eager_bit_for_bit(self, plain, model):
        """Unquantized f32 = the unquantized upstream eager (tanh approximation) — for every valid
        length from 2 to 512."""
        upstream = upstream_tanh(model)

        actual = run(plain, model, torch.float32)

        for length, output in zip(LENGTHS, actual, strict=True):
            expected = up.valid_output(upstream, probe.tiny_ids(length))
            assert torch.equal(output, expected), f"L = {length}: {up.compare(output, expected)}"

    def test_it_differs_from_the_i8_reference(self, plain, weights, model):
        """The pair (not tautological): the unquantized reference splits from the i8 reference (the
        reader has not fallen back to the container)."""
        (unquantized,) = run(plain, model, torch.float32, lengths=(28,))
        (i8,) = run(weights, model, torch.float32, lengths=(28,))

        assert not torch.equal(unquantized, i8)


class TestFloat32Reference:
    def test_it_is_the_upstream_eager_bit_for_bit(self, weights, quantized, model):
        """The f32 rewrite = the upstream eager (tanh approximation, fake-quant applied) — for every
        valid length from 2 to 512."""
        upstream = upstream_tanh(quantized)

        actual = run(weights, model, torch.float32)

        for length, output in zip(LENGTHS, actual, strict=True):
            expected = up.valid_output(upstream, probe.tiny_ids(length))
            assert torch.equal(output, expected), f"L = {length}: {up.compare(output, expected)}"

    def test_swapped_layer_tables_break_the_match(self, weights, quantized, model):
        """The pair (not tautological): swapping the relative-position tables of layers 0 / 1 splits
        it from the upstream."""
        upstream = upstream_tanh(quantized)

        (actual,) = run(SwappedTables(weights), model, torch.float32, lengths=(28,))

        assert not torch.equal(actual, up.valid_output(upstream, probe.tiny_ids(28)))


class TestFloat64Reference:
    def test_no_narrow_float_is_made(self, weights, model):
        source = Preloaded(weights)
        # The inputs (the bucket table is built with the attention formula of the upstream on meta)
        # are also built outside the watch.
        inputs = [case_inputs(model, length) for length in LENGTHS]
        watch = NarrowFloatWatch()

        with watch:
            forward(source, model, torch.float64, inputs)

        assert watch.found == []

    def test_the_watch_sees_the_upstream_double_narrowing(self, quantized):
        """The pair (not tautological): the upstream forward with `.double()` drops to f32 in
        RMSNorm and softmax — why the f64 reference is not taken with the upstream's `.double()`
        (ADR 0119 addendum "preparing 10c")."""
        upstream = upstream_tanh(quantized).double()
        watch = NarrowFloatWatch()

        with watch:
            up.valid_output(upstream, probe.tiny_ids(28))

        assert watch.found != []

    def test_it_agrees_with_f32_within_f32_rounding(self, weights, model):
        """It rewrites the same function (a mix-up shows up at an O(1) ratio)."""
        f64 = run(weights, model, torch.float64)
        f32 = run(weights, model, torch.float32)

        ratios = [up.max_ratio(low, high) for low, high in zip(f32, f64, strict=True)]

        assert all(0 < ratio < F32_ROUNDING_RATIO for ratio in ratios), ratios

    def test_the_bf16_mismatch_is_bf16_error(self, weights, quantized, model):
        """A comparison of upper bounds: f32's own error (its ratio against f64) is under 1% of the
        mismatch between the upstream bf16 and f32 — the mismatch is bf16 error, and f32 may be the
        baseline of the quality record (ADR 0119 addendum "results of 10b")."""
        bf16_model = upstream_tanh(quantized).to(torch.bfloat16)
        f64 = run(weights, model, torch.float64)
        f32 = run(weights, model, torch.float32)

        for length, high, low in zip(LENGTHS, f64, f32, strict=True):
            bf16 = up.valid_output(bf16_model, probe.tiny_ids(length)).float()
            mismatch = up.max_ratio(bf16, low)
            own = up.max_ratio(low, high)
            assert own < F32_SHARE_OF_BF16_MISMATCH * mismatch, (length, own, mismatch)


#: The record of the axes in the golden's metadata (shape only — the values are assembled from the
#: table by `wan.umt5_export.reference_axes`).
AXES = {
    "encoder": {"model": "xxl", "repo": "owner/encoder", "revision": "0" * 40},
    "cases": {
        "model": "m",
        "repo": "owner/tokenizer",
        "revision": "1" * 40,
        "subfolder": "tokenizer",
    },
}


class TestGolden:
    def test_the_file_round_trips(self, weights, model, tmp_path):
        ids, buckets = case_inputs(model, 28)
        case = ur.ReferenceCase(
            name="band-l0028",
            role="band",
            prompt="原文",
            cleaned="原文",
            ids=tuple(int(token) for token in ids[0]),
            source={"parityCases": ["random-0"]},
        )
        f64 = run(weights, model, torch.float64, lengths=(28,))[0]
        f32 = run(weights, model, torch.float32, lengths=(28,))[0]
        path = tmp_path / f"{ur.REFERENCE_PREFIX}{case.name}{ur.CASE_SUFFIX}"

        ur.write_golden(
            path,
            ur.golden_tensors(case, buckets, f64, f32),
            ur.golden_metadata(
                case, weights, ur.reference_ratios(f64, f32), {"torch": "x"}, axes=AXES
            ),
        )

        with safe_open(str(path), framework="pt") as handle:
            metadata = handle.metadata()
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118
        assert set(metadata) == {ur.METADATA_KEY}
        meta = json.loads(metadata[ur.METADATA_KEY])
        assert (meta["format"], meta["length"], meta["prompt"]) == (ur.REFERENCE_FORMAT, 28, "原文")
        assert meta["container"]["part0Sha256"] == weights.part0_sha256
        # Which upstream values were used (the umT5 upstream and Wan's tokenizer for the cases' id
        # sequences — ADR 0122 decision 3).
        assert meta["axes"] == AXES
        assert {key: (value.dtype, tuple(value.shape)) for key, value in tensors.items()} == {
            ur.INPUT_IDS_KEY: (torch.int32, (28,)),
            ur.BUCKETS_KEY: (torch.int32, (28, 28)),
            ur.OUTPUT_F64_KEY: (torch.float32, (1, 28, 64)),
            ur.OUTPUT_F32_KEY: (torch.float32, (1, 28, 64)),
        }
        assert torch.equal(tensors[ur.BUCKETS_KEY], buckets.to(torch.int32))
        assert torch.equal(tensors[ur.OUTPUT_F64_KEY], f64.float())
        assert meta["ratios"]["f32VsF64"] == up.max_ratio(f32, f64)
        assert "unquantized" not in meta

    def test_an_accept_case_carries_the_unquantized_reference(
        self, weights, plain, model, tmp_path
    ):
        ids, buckets = case_inputs(model, 28)
        case = ur.ReferenceCase(
            name="accept-x",
            role="accept",
            prompt="原文",
            cleaned="原文",
            ids=tuple(int(token) for token in ids[0]),
        )
        f64 = run(weights, model, torch.float64, lengths=(28,))[0]
        f32 = run(weights, model, torch.float32, lengths=(28,))[0]
        high = run(plain, model, torch.float64, lengths=(28,))[0]
        low = run(plain, model, torch.float32, lengths=(28,))[0]
        path = tmp_path / f"{ur.REFERENCE_PREFIX}{case.name}{ur.CASE_SUFFIX}"

        ur.write_golden(
            path,
            ur.golden_tensors(case, buckets, f64, f32, (high, low)),
            ur.golden_metadata(
                case,
                weights,
                ur.reference_ratios(f64, f32),
                {"torch": "x"},
                ur.reference_ratios(high, low),
                axes=AXES,
            ),
        )

        with safe_open(str(path), framework="pt") as handle:
            meta = json.loads(handle.metadata()[ur.METADATA_KEY])
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118
        assert torch.equal(tensors[ur.OUTPUT_F32_KEY], f32)
        assert torch.equal(tensors[ur.OUTPUT_UNQUANTIZED_F64_KEY], high.float())
        assert torch.equal(tensors[ur.OUTPUT_UNQUANTIZED_F32_KEY], low)
        assert meta["unquantized"]["ratios"]["f32VsF64"] == up.max_ratio(low, high)

    @pytest.mark.parametrize(
        ("role", "with_unquantized"), [("band", True), ("accept", False)], ids=["band", "accept"]
    )
    def test_the_unquantized_reference_belongs_to_accept_only(self, model, role, with_unquantized):
        ids, buckets = case_inputs(model, 2)
        case = ur.ReferenceCase(
            name="x", role=role, prompt="p", cleaned="p", ids=tuple(int(t) for t in ids[0])
        )
        output = torch.zeros(1, 2, 64)
        pair = (output, output) if with_unquantized else None

        with pytest.raises(ur.Umt5ReferenceError, match="only acceptance cases"):
            ur.golden_tensors(case, buckets, output, output, pair)


def fake_encode(text: str) -> Mapping[str, Any]:
    """One word = one token + a trailing `</s>` (for checking the selection — only the length
    matters)."""
    words = text.split()
    return {"cleaned": " ".join(words), "ids": [5] * len(words) + [1]}


def fake_parity(count: int = 200) -> list[dict[str, Any]]:
    """Accepted random picks of length 3–42 (the same id sequences as `fake_encode`) and one
    rejection."""
    cases = []
    for index in range(count):
        text = " ".join(["w"] * (2 + index % 40))
        cases.append({"id": f"random-{index}", "text": text, **fake_encode(text)})
    cases.append({"id": "random-x", "text": "x", "reject": "oov"})
    return cases


class TestBandCases:
    def test_lengths_fall_in_the_declared_bands_without_reuse(self):
        cases = ur.band_cases(fake_parity(), fake_encode)

        bands = [*ur.BAND_SINGLE_LENGTHS, *ur.BAND_COMPOSITE_LENGTHS]
        used = [member for case in cases for member in case.source["parityCases"]]
        assert [case.role for case in cases] == ["band"] * len(bands)
        assert all(
            low <= len(case.ids) <= high for case, (low, high) in zip(cases, bands, strict=True)
        )
        assert len(used) == len(set(used))
        assert ur.band_cases(fake_parity(), fake_encode) == cases

    def test_a_drifted_fixture_fails_loudly(self):
        parity = fake_parity()
        for case in parity:
            if "ids" in case:
                case["ids"] = [*case["ids"], 1]

        with pytest.raises(ur.Umt5ReferenceError, match="differs from the parity fixture"):
            ur.band_cases(parity, fake_encode)
