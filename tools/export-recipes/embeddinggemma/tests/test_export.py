"""`embeddinggemma/export.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（既存 `deberta/export.py` のテストと同じ規律）。ここで固定するのは、
壊れると**偽 PASS** になる側の規律だけ:

- `--batch` の既定が 1（従来の 5 ケース golden そのまま）で、1 未満は fail loudly
- `_write_io` が embeddings を `[B, H]` の形のまま返すこと（`_sanity` / `_sanity_batch` の
  両方が同じ形から読める — 片方のためだけに reshape を作り込まない）
- `_sanity_batch` が単位ノルムと行間一致の**両方**を見ること（片方だけでは恒真になる）
- 上流の構成（Dense の活性化・Pooling 方式・`modules.json` の 5 段）がラッパの決め打ちと
  違えば落ちること（golden は同じラッパから採るので、TS 側の突合では捕まらない）
- IR の形の門 {@link eg.assert_ir_form} が、SDPA の分解（attention の欠け）・帯マスクの入力残り・
  供給元と定数の形の崩れを実際に落とすこと（数値は合ったまま壊れる種類の性質）
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

#: tiny な export に焼く出所（値は突合されない — 台本の出所の導出は `upstream_provenance` の席）。
TINY_PROVENANCE = Provenance(license="fixture")


class TinyEmbedding(nn.Module):
    """`EmbeddingWrapper` の最小の骨格（masked-mean → linear → L2 正規化、`[B,T]→[B,H]`）。"""

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
    """`query-en` 相当の 1 行を 3 行に複製したケースを export して `(wrapper, graph, out_dir)`。"""
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
        with pytest.raises(SystemExit, match="--batch は 1 以上"):
            eg.main(["--batch", "0"])


class TestWriteIoPreservesTheBatchDimension:
    """MUST: `_write_io` の embeddings は `[B, H]` のまま — `_sanity_batch` が行を見比べる形。"""

    def test_embeddings_keep_the_batch_dimension(self, exported_batch3):
        wrapper, graph, out_dir = exported_batch3

        written, embeddings = eg._write_io(wrapper, graph, (CASE_B3,), out_dir)

        assert written == [f"{eg.IO_PREFIX}batch3{eg.IO_SUFFIX}"]
        assert tuple(embeddings["batch3"].shape) == (3, 4)

    def test_batch_one_still_reduces_to_a_flat_vector_via_sanity(self, tmp_path):
        """batch=1 の従来経路は `_sanity` 側の reshape(-1) で吸収される（挙動不変の確認）。"""
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
        """MUST: 行間一致を見る — 複製元の入力が同一なら全行が一致するはず。"""
        output = torch.tensor([[0.6, 0.8], [0.6, 0.8], [0.0, 1.0]])

        with pytest.raises(AssertionError, match="行間の出力が一致しない"):
            eg._sanity_batch(output)

    def test_fails_loudly_when_a_row_norm_is_off(self):
        """MUST: ノルムだけでも見る — 行間一致だけでは全行が同じだけズレていても通ってしまう。"""
        output = torch.tensor([[0.6, 0.8], [0.5, 0.5]])

        with pytest.raises(AssertionError, match="L2 ノルムが 1 から外れた"):
            eg._sanity_batch(output)


class TestSanityAcceptsTheGeneralizedShape:
    def test_unit_norm_and_cosine_still_work_when_embeddings_are_shape_1xh(self):
        """`_write_io` が reshape(-1) をやめた後も従来 4 ケース分で `_sanity` が動くことの確認。"""
        vectors = {
            "query-en": torch.tensor([[1.0, 0.0]]),
            "document-en": (torch.tensor([[0.9, 0.1]]) / torch.tensor([[0.9, 0.1]]).norm()),
            "bare": torch.tensor([[0.0, 1.0]]),
            "query-ja": torch.tensor([[1.0, 0.0]]),
        }

        result = eg._sanity(vectors)

        assert set(result["l2_norms"]) == set(vectors)


#: 上流（`google/embeddinggemma-300m`）の `modules.json` / Pooling / Dense の宣言と同じ値。
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
    """`2_Dense/{config.json, model.safetensors}` を書き、書いた重みを返す（in 2 → out 3）。"""
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
        """省略時の sentence-transformers は Tanh を掛けるので、欄が無いのも拒否側。"""
        _write_dense(tmp_path, activation_function=activation)

        with pytest.raises(ValueError, match="activation_function"):
            eg.load_dense(tmp_path, "2_Dense")

    def test_it_rejects_a_dense_with_bias(self, tmp_path):
        _write_dense(tmp_path, bias=True)

        with pytest.raises(ValueError, match="bias"):
            eg.load_dense(tmp_path, "2_Dense")

    def test_it_rejects_a_weight_whose_shape_disagrees_with_the_config(self, tmp_path):
        _write_dense(tmp_path, weight_shape=(2, 3))

        with pytest.raises(ValueError, match="食い違う"):
            eg.load_dense(tmp_path, "2_Dense")


class TestSentenceTransformerLayout:
    def test_it_accepts_the_upstream_layout(self, tmp_path):
        _write_layout(tmp_path)

        eg.assert_sentence_transformer_layout(tmp_path)

    @pytest.mark.parametrize(
        ("modules", "pooling", "message"),
        [
            (UPSTREAM_MODULES[:-1], {}, "5 段"),
            (
                [*UPSTREAM_MODULES[:3], UPSTREAM_MODULES[4], UPSTREAM_MODULES[3]],
                {},
                "5 段",
            ),
            (
                UPSTREAM_MODULES,
                {"pooling_mode_mean_tokens": False, "pooling_mode_cls_token": True},
                "pooling 方式",
            ),
            (UPSTREAM_MODULES, {"pooling_mode_max_tokens": True}, "pooling 方式"),
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
        """本体（`config.json` / 重み）が無い席でも、構成の拒否が先に出る。"""
        _write_layout(tmp_path, include_prompt=False)

        with pytest.raises(ValueError, match="include_prompt"):
            eg.load_wrapper(tmp_path)


#: tiny IR の寸法（実物は 24 層 = sliding 20 + full 4・Tmax 512）。**実物と違う数**にする。
TINY_LAYERS = 3
TINY_SYM_MAX = 16
#: 層種別ごとの帯マスク定数（実物も sliding-window と全結合の 2 本）。
BAND_CONSTANTS = ("band_sliding", "band_full")


def _ir_graph(layers: int = TINY_LAYERS) -> IrGraph:
    """attention `layers` 本の最小 IR（帯マスクは層種別ごとの Tmax 定数 + sym_prefix_slice）。"""
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
    """SDPA 保存形と帯マスク定数の門（golden の数値突合では捕まらない壊れ方を落とす）。"""

    def test_the_preserved_form_passes_with_one_band_per_layer_type(self):
        form = eg.assert_ir_form(_ir_graph(), TINY_IR_CONFIG, TINY_SYM_MAX)

        assert form["attention_nodes"] == TINY_LAYERS
        assert form["mask_constants"] == sorted(BAND_CONSTANTS)

    def test_a_layer_whose_attention_was_decomposed_is_rejected(self):
        """SDPA が分解経路へ落ちると attention op が層数に足りない（数値は合ったまま）。"""
        with pytest.raises(AssertionError, match="層と一致しない"):
            eg.assert_ir_form(_ir_graph(layers=TINY_LAYERS - 1), TINY_IR_CONFIG, TINY_SYM_MAX)

    def test_a_mask_left_as_a_graph_input_is_rejected(self):
        graph = _ir_graph()
        graph.inputs.append(IrInput(name="attention_mask", dtype="f32", shape=[1, 1, "T", "T"]))

        with pytest.raises(AssertionError, match="グラフ入力"):
            eg.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX)

    def test_a_mask_fed_straight_from_an_initializer_is_rejected(self):
        """T で切り出さずに定数を直結した形は、T < Tmax で形が合わない資産になる。"""
        graph = _ir_graph()
        attention = next(node for node in graph.nodes if node.op == eg.ATTENTION_OP)
        attention.ins[3] = BAND_CONSTANTS[0]

        with pytest.raises(AssertionError, match="供給元"):
            eg.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX)

    def test_a_band_constant_of_the_wrong_extent_is_rejected(self):
        """Tmax より短い定数は「T が上限まで伸びると黙って足りない」形。"""
        graph = _ir_graph()
        graph.values[BAND_CONSTANTS[1]] = IrValue(dtype="f32", shape=[1, 1, 8, 8])

        with pytest.raises(AssertionError, match="Tmax 形"):
            eg.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX)


class TestUpstreamProvenance:
    """容器の revision は `--model-dir` の取得記録から導く（直書きしない）。"""

    def test_the_revision_comes_from_the_download_record(self, tmp_path) -> None:
        write_snapshot(tmp_path, license="ignored", revision=OTHER_REVISION)

        assert eg.upstream_provenance(tmp_path) == Provenance(
            license=eg.LICENSE, upstream_revision=OTHER_REVISION
        )

    def test_a_checkpoint_without_the_record_fails_loudly(self, tmp_path) -> None:
        with pytest.raises(UpstreamProvenanceError, match="metadata が無い"):
            eg.upstream_provenance(tmp_path)
