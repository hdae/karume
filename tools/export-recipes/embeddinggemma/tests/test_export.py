"""Script-level contracts of `embeddinggemma/export.py` (the part that needs no real weights).

The real-weight emit is manual (the same discipline as the existing `deberta/export.py` tests).
What is pinned here is only the discipline whose breakage turns into a **false PASS**:

- `--batch` defaults to 1 (the earlier 5-case golden as is), and below 1 it fails loudly
- `_write_io` returns the embeddings in their `[B, H]` shape (both `_sanity` and `_sanity_batch`
  read the same shape — no reshape built in for only one of them)
- `_sanity_batch` checks **both** unit norm and cross-row agreement (either alone is vacuous)
- it fails when the upstream layout (the Dense activation, the pooling mode, the 5 stages of
  `modules.json`) differs from what the wrapper hard-codes (the golden is taken from the same
  wrapper, so the TS-side comparison does not catch it)
- the IR-form check {@link eg.assert_ir_form} actually rejects a decomposed SDPA (missing
  attention), a band mask left as an input, and a broken source or constant shape (properties that
  break while the numbers still agree)
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from upstream_fixture import OTHER_REVISION, write_snapshot

from _shared.upstream import UpstreamProvenanceError
from embeddinggemma import export as eg
from karume.container import Provenance
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.pipeline import export_to_file

#: The provenance baked into the tiny export (its value is not compared — deriving the script's
#: provenance is the job of `upstream_provenance`).
TINY_PROVENANCE = Provenance(license="fixture")


class TinyEmbedding(nn.Module):
    """The smallest skeleton of `EmbeddingWrapper` (masked mean → linear → L2 normalisation,
    `[B,T]→[B,H]`)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(1, 4, bias=False)

    def forward(self, input_ids: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(input_ids.to(torch.float32).unsqueeze(-1))
        total = torch.sum(hidden * pool_mask.unsqueeze(-1), dim=1)
        count = torch.sum(pool_mask, dim=1).unsqueeze(-1)
        pooled = total / torch.clamp(count, min=1e-9)
        norm = torch.sqrt(torch.sum(pooled * pooled, dim=-1)).clamp(min=1e-12).unsqueeze(-1)
        return pooled / norm


CASE_B1 = (
    "case0",
    torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
    torch.ones(1, 4, dtype=torch.float32),
)
CASE_B3 = (
    "batch3",
    torch.tensor([[1, 2, 3, 4]], dtype=torch.int64).expand(3, -1).contiguous(),
    torch.ones(3, 4, dtype=torch.float32),
)


@pytest.fixture
def exported_batch3(tmp_path):
    """Export a case with one `query-en`-like row copied into 3 rows; returns
    `(wrapper, graph, out_dir)`."""
    torch.manual_seed(0)
    wrapper = TinyEmbedding()
    graph = export_to_file(
        wrapper,
        CASE_B3[1:],
        tmp_path / eg.MODEL_FILE,
        provenance=TINY_PROVENANCE,
        graph_name="tiny",
    )
    return wrapper, graph, tmp_path


class TestBatchCli:
    def test_batch_defaults_to_one(self, monkeypatch):
        seen: dict[str, object] = {}
        monkeypatch.setattr(eg, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})
        eg.main([])

        assert seen["batch"] == 1

    def test_batch_flag_is_forwarded(self, monkeypatch):
        seen: dict[str, object] = {}
        monkeypatch.setattr(eg, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})
        eg.main(["--batch", "32"])

        assert seen["batch"] == 32

    def test_batch_below_one_fails_loudly(self):
        with pytest.raises(SystemExit, match="--batch must be at least 1"):
            eg.main(["--batch", "0"])


class TestWriteIoPreservesTheBatchDimension:
    """MUST: the embeddings of `_write_io` stay `[B, H]` — the shape `_sanity_batch` compares rows
    in."""

    def test_embeddings_keep_the_batch_dimension(self, exported_batch3):
        wrapper, graph, out_dir = exported_batch3

        written, embeddings = eg._write_io(wrapper, graph, (CASE_B3,), out_dir)

        assert written == [f"{eg.IO_PREFIX}batch3{eg.IO_SUFFIX}"]
        assert tuple(embeddings["batch3"].shape) == (3, 4)

    def test_batch_one_still_reduces_to_a_flat_vector_via_sanity(self, tmp_path):
        """The earlier batch=1 path is absorbed by the reshape(-1) in `_sanity` (behaviour
        unchanged)."""
        torch.manual_seed(0)
        wrapper = TinyEmbedding()
        graph = export_to_file(
            wrapper,
            CASE_B1[1:],
            tmp_path / eg.MODEL_FILE,
            provenance=TINY_PROVENANCE,
            graph_name="tiny",
        )

        _, embeddings = eg._write_io(wrapper, graph, (CASE_B1,), tmp_path)

        assert tuple(embeddings["case0"].shape) == (1, 4)
        assert embeddings["case0"].reshape(-1).shape == (4,)


class TestSanityBatch:
    def test_passes_when_rows_are_identical_unit_vectors(self):
        row = torch.tensor([0.6, 0.8])
        output = row.unsqueeze(0).expand(5, -1).contiguous()

        result = eg._sanity_batch(output)

        assert result["rows"] == 5
        assert result["row_max_abs_diff"] == 0.0

    def test_fails_loudly_when_a_row_diverges(self):
        """MUST: check cross-row agreement — with the same source input every row should agree."""
        output = torch.tensor([[0.6, 0.8], [0.6, 0.8], [0.0, 1.0]])

        with pytest.raises(AssertionError, match="the rows of the output disagree"):
            eg._sanity_batch(output)

    def test_fails_loudly_when_a_row_norm_is_off(self):
        """MUST: check the norm on its own too — cross-row agreement alone passes when every row is
        off by the same amount."""
        output = torch.tensor([[0.6, 0.8], [0.5, 0.5]])

        with pytest.raises(AssertionError, match="L2 norm is not 1"):
            eg._sanity_batch(output)


class TestSanityAcceptsTheGeneralizedShape:
    def test_unit_norm_and_cosine_still_work_when_embeddings_are_shape_1xh(self):
        """`_sanity` still works on the earlier 4 cases after `_write_io` stopped applying
        reshape(-1)."""
        vectors = {
            "query-en": torch.tensor([[1.0, 0.0]]),
            "document-en": (torch.tensor([[0.9, 0.1]]) / torch.tensor([[0.9, 0.1]]).norm()),
            "bare": torch.tensor([[0.0, 1.0]]),
            "query-ja": torch.tensor([[1.0, 0.0]]),
        }

        result = eg._sanity(vectors)

        assert set(result["l2_norms"]) == set(vectors)


#: The same values as the upstream (`google/embeddinggemma-300m`) declarations of `modules.json` /
#: Pooling / Dense.
UPSTREAM_MODULES = [
    {"idx": idx, "name": str(idx), "path": path, "type": f"sentence_transformers.models.{kind}"}
    for idx, (path, kind) in enumerate(
        [
            ("", "Transformer"),
            ("1_Pooling", "Pooling"),
            ("2_Dense", "Dense"),
            ("3_Dense", "Dense"),
            ("4_Normalize", "Normalize"),
        ]
    )
]
UPSTREAM_POOLING = {
    "word_embedding_dimension": 768,
    "pooling_mode_cls_token": False,
    "pooling_mode_mean_tokens": True,
    "pooling_mode_max_tokens": False,
    "pooling_mode_mean_sqrt_len_tokens": False,
    "pooling_mode_weightedmean_tokens": False,
    "pooling_mode_lasttoken": False,
    "include_prompt": True,
}


def _write_dense(model_dir, *, weight_shape=(3, 2), **overrides) -> torch.Tensor:
    """Write `2_Dense/{config.json, model.safetensors}` and return the weight written (in 2 → out
    3)."""
    config = {
        "in_features": 2,
        "out_features": 3,
        "bias": False,
        "activation_function": eg.DENSE_ACTIVATION,
        **overrides,
    }
    config = {key: value for key, value in config.items() if value is not None}
    weight = torch.arange(weight_shape[0] * weight_shape[1], dtype=torch.float32).reshape(
        weight_shape
    )
    (model_dir / "2_Dense").mkdir(parents=True)
    (model_dir / "2_Dense" / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file({eg.DENSE_WEIGHT_KEY: weight}, str(model_dir / "2_Dense" / eg.CHECKPOINT_FILE))
    return weight


def _write_layout(model_dir, *, modules=UPSTREAM_MODULES, **pooling) -> None:
    (model_dir / eg.POOLING_DIR).mkdir(parents=True)
    (model_dir / "modules.json").write_text(json.dumps(modules), encoding="utf-8")
    (model_dir / eg.POOLING_DIR / "config.json").write_text(
        json.dumps({**UPSTREAM_POOLING, **pooling}), encoding="utf-8"
    )


class TestLoadDense:
    def test_it_builds_a_bias_free_linear_from_the_declared_shape(self, tmp_path):
        weight = _write_dense(tmp_path)

        dense = eg.load_dense(tmp_path, "2_Dense")

        assert torch.equal(dense.weight, weight)
        assert dense.bias is None

    @pytest.mark.parametrize(
        "activation", ["torch.nn.modules.activation.Tanh", None], ids=["tanh", "absent"]
    )
    def test_it_rejects_an_activation_the_wrapper_does_not_apply(self, tmp_path, activation):
        """sentence-transformers applies Tanh when the field is omitted, so a missing field is
        rejected too."""
        _write_dense(tmp_path, activation_function=activation)

        with pytest.raises(ValueError, match="activation_function"):
            eg.load_dense(tmp_path, "2_Dense")

    def test_it_rejects_a_dense_with_bias(self, tmp_path):
        _write_dense(tmp_path, bias=True)

        with pytest.raises(ValueError, match="bias"):
            eg.load_dense(tmp_path, "2_Dense")

    def test_it_rejects_a_weight_whose_shape_disagrees_with_the_config(self, tmp_path):
        _write_dense(tmp_path, weight_shape=(2, 3))

        with pytest.raises(ValueError, match="disagrees with"):
            eg.load_dense(tmp_path, "2_Dense")


class TestSentenceTransformerLayout:
    def test_it_accepts_the_upstream_layout(self, tmp_path):
        _write_layout(tmp_path)

        eg.assert_sentence_transformer_layout(tmp_path)

    @pytest.mark.parametrize(
        ("modules", "pooling", "message"),
        [
            (UPSTREAM_MODULES[:-1], {}, "5 stages"),
            (
                [*UPSTREAM_MODULES[:3], UPSTREAM_MODULES[4], UPSTREAM_MODULES[3]],
                {},
                "5 stages",
            ),
            (
                UPSTREAM_MODULES,
                {"pooling_mode_mean_tokens": False, "pooling_mode_cls_token": True},
                "pooling mode",
            ),
            (UPSTREAM_MODULES, {"pooling_mode_max_tokens": True}, "pooling mode"),
            (UPSTREAM_MODULES, {"include_prompt": False}, "include_prompt"),
        ],
        ids=["no-normalize", "reordered", "cls-pooling", "mean-and-max", "prompt-excluded"],
    )
    def test_it_rejects_a_layout_the_wrapper_does_not_fold(
        self, tmp_path, modules, pooling, message
    ):
        _write_layout(tmp_path, modules=modules, **pooling)

        with pytest.raises(ValueError, match=message):
            eg.assert_sentence_transformer_layout(tmp_path)

    def test_load_wrapper_checks_the_layout_before_reading_the_model(self, tmp_path):
        """Even in a directory without the body (`config.json` / weights), the layout rejection
        comes first."""
        _write_layout(tmp_path, include_prompt=False)

        with pytest.raises(ValueError, match="include_prompt"):
            eg.load_wrapper(tmp_path)


#: The tiny IR's dimensions (the real one has 24 layers = sliding 20 + full 4, Tmax 512). They are
#: **numbers different from the real ones**.
TINY_LAYERS = 3
TINY_SYM_MAX = 16
#: Band mask constants per layer type (the real one also has 2, sliding window and full).
BAND_CONSTANTS = ("band_sliding", "band_full")


def _ir_graph(layers: int = TINY_LAYERS) -> IrGraph:
    """The smallest IR with `layers` attention nodes (band masks are a Tmax constant per layer
    type + sym_prefix_slice)."""
    graph = IrGraph(symbols=["T"])
    graph.inputs.append(IrInput(name="input_ids", dtype="i32", shape=[1, "T"]))
    graph.inputs.append(IrInput(name="pool_mask", dtype="f32", shape=[1, "T"]))
    for constant in BAND_CONSTANTS:
        graph.initializers[constant] = IrInitializer(
            tensor=f"const.{constant}", storage=IrStorage(dtype="f32")
        )
        graph.values[constant] = IrValue(dtype="f32", shape=[1, 1, TINY_SYM_MAX, TINY_SYM_MAX])
        graph.values[f"{constant}.T"] = IrValue(dtype="f32", shape=[1, 1, "T", "T"])
        graph.nodes.append(
            IrNode(
                op=eg.SYM_PREFIX_SLICE_OP,
                ins=[constant],
                outs=[f"{constant}.T"],
                attrs={"sym": "T", "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
            )
        )
    for layer in range(layers):
        mask = f"{BAND_CONSTANTS[layer % 2]}.T"
        for slot in ("q", "k", "v"):
            graph.values[f"{slot}{layer}"] = IrValue(dtype="f32", shape=[1, 2, "T", 4])
        graph.values[f"attn{layer}"] = IrValue(dtype="f32", shape=[1, 2, "T", 4])
        graph.nodes.append(
            IrNode(
                op=eg.ATTENTION_OP,
                ins=[f"q{layer}", f"k{layer}", f"v{layer}", mask],
                outs=[f"attn{layer}"],
                attrs={"scale": 0.5},
            )
        )
    graph.outputs.append(f"attn{layers - 1}")
    return graph


TINY_IR_CONFIG = SimpleNamespace(num_hidden_layers=TINY_LAYERS)


class TestAssertIrForm:
    """The check of the SDPA-preserved form and band mask constants (rejects breakages the golden's
    numerical comparison does not catch)."""

    def test_the_preserved_form_passes_with_one_band_per_layer_type(self):
        form = eg.assert_ir_form(_ir_graph(), TINY_IR_CONFIG, TINY_SYM_MAX)

        assert form["attention_nodes"] == TINY_LAYERS
        assert form["mask_constants"] == sorted(BAND_CONSTANTS)

    def test_a_layer_whose_attention_was_decomposed_is_rejected(self):
        """When SDPA falls to the decomposed path there are fewer attention ops than layers (the
        numbers still agree)."""
        with pytest.raises(AssertionError, match="does not match the layer count"):
            eg.assert_ir_form(_ir_graph(layers=TINY_LAYERS - 1), TINY_IR_CONFIG, TINY_SYM_MAX)

    def test_a_mask_left_as_a_graph_input_is_rejected(self):
        graph = _ir_graph()
        graph.inputs.append(IrInput(name="attention_mask", dtype="f32", shape=[1, 1, "T", "T"]))

        with pytest.raises(AssertionError, match="graph inputs"):
            eg.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX)

    def test_a_mask_fed_straight_from_an_initializer_is_rejected(self):
        """A constant wired directly without slicing by T makes an asset whose shape is wrong for
        T < Tmax."""
        graph = _ir_graph()
        attention = next(node for node in graph.nodes if node.op == eg.ATTENTION_OP)
        attention.ins[3] = BAND_CONSTANTS[0]

        with pytest.raises(AssertionError, match="supplied by"):
            eg.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX)

    def test_a_band_constant_of_the_wrong_extent_is_rejected(self):
        """A constant shorter than Tmax is "silently too short once T grows to the upper bound"."""
        graph = _ir_graph()
        graph.values[BAND_CONSTANTS[1]] = IrValue(dtype="f32", shape=[1, 1, 8, 8])

        with pytest.raises(AssertionError, match="Tmax shape"):
            eg.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX)


class TestUpstreamProvenance:
    """The container's revision is derived from the download record of `--model-dir` (not
    hard-coded)."""

    def test_the_revision_comes_from_the_download_record(self, tmp_path) -> None:
        write_snapshot(tmp_path, license="ignored", revision=OTHER_REVISION)

        assert eg.upstream_provenance(tmp_path) == Provenance(
            license=eg.LICENSE, upstream_revision=OTHER_REVISION
        )

    def test_a_checkpoint_without_the_record_fails_loudly(self, tmp_path) -> None:
        with pytest.raises(UpstreamProvenanceError, match=r"\.metadata is missing"):
            eg.upstream_provenance(tmp_path)
