"""`wan/umt5_reference.py`（i8 系列の容器から重みを読む層逐次の CPU 参照）の約束事
（ADR 0119 決定 8 ①・段 10c）。

縛るのは、壊れても例外が出ず**参照の値だけが静かにずれる**側:

- 容器から読んだ重み（packed × scale）が export の fake-quant の値とビット一致する
- f32 の書き下しが上流の eager（活性を tanh 近似に差し替え・fake-quant 済み）とビット一致する
  — 層の表の取り違えは割れる（対）
- f64 の書き下しの中で f64 でない浮動小数が 1 つも作られない — 上流を `.double()` した形は
  RMSNorm と softmax で f32 に落ちるので同じ検査に掛かる（対）
- f64 の参照で見ると、上流の bf16 と f32 の不一致はほぼ全部が bf16 の誤差（f32 自身の誤差は
  その 1% 未満 — 品質の記録で f32 を基準にしてよい根拠の小模型での裏付け）
- 量子化しない重みの読み口（checkpoint の F32）が上流の parameter とビット一致し、同じ書き下しが
  量子化しない上流の eager とビット一致する — i8 の参照とは割れる（対）
- golden の形（キー・dtype・形・メタ）・量子化なしの参照は受入れだけ、と決定用のケースの選び方

乱数初期化の小さな umT5（`wan.umt5_probe.TINY_CONFIG`）を `umt5_export.prepare` で容器にして
回す — 実重みは読まない。
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

#: 比べる有効長（下限・固定プロンプトの長さ・max_distance 128 越え・上限）。
LENGTHS = (2, 28, 126, 300, 512)

#: 書き出す部品名（pytest の作業席だけで名乗る — `wan/umt5_export.py` のモジュール doc）。
GRAPH_NAME = "text_encoder"

#: f32 の書き下しの f64 に対する比の上限（f32 の丸めの桁 — 書き下しの取り違えは O(1) で出る）。
F32_ROUNDING_RATIO = 1e-5

#: f32 自身の誤差が bf16 との不一致に占める割合の上限（f64 で見た上限の比較）。
F32_SHARE_OF_BF16_MISMATCH = 1e-2


@pytest.fixture(scope="module")
def model() -> Any:
    pytest.importorskip("transformers")
    return probe.tiny_model()


@pytest.fixture(scope="module")
def checkpoint(model: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """小模型の checkpoint（`save_pretrained` — F32 の safetensors）。"""
    directory = tmp_path_factory.mktemp("umt5-reference") / "checkpoint"
    model.save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def container(checkpoint: Path) -> Path:
    """`umt5_export.prepare`（行の塊ごとの i8）で書いた小模型の容器（代表 path）。"""
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
    """量子化しない重みの読み口（`umt5_export.checkpoint_weights` — 実モデルと同じ組み方）。"""
    return ue.checkpoint_weights(checkpoint)


@pytest.fixture(scope="module")
def quantized(model: Any) -> Any:
    """fake-quant 済みの上流（ラッパ経由で丸める — ラッパは上流のモジュールを共有する）。"""
    upstream = copy.deepcopy(model)
    up.fake_quant_i8(up.Umt5EncoderTokens(upstream))
    return upstream


def upstream_tanh(quantized: Any) -> Any:
    """`quantized` の写しで、FFN の活性だけを tanh 近似に差し替えたもの（上流の eager の形）。"""
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
    """層 0 と層 1 の相対位置の表を入れ替えて返す読み口（故障注入）。"""

    def __init__(self, source: ur.WeightSource) -> None:
        self.source = source
        first, second = ur.layer_keys(0)["relative_bias"], ur.layer_keys(1)["relative_bias"]
        self.swap = {first: second, second: first}

    def tensor(self, key: str) -> torch.Tensor:
        return self.source.tensor(self.swap.get(key, key))

    def rows(self, key: str, indices: torch.Tensor) -> torch.Tensor:
        return self.source.rows(key, indices)


class Preloaded:
    """重みと語彙埋め込みの行を先に読み切った読み口 — 監視の間に重みの読み出し（容器の
    f32 の演算）を走らせず、書き下しの演算だけを監視に掛ける。"""

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
    """この間に作られた f64 でない浮動小数の値を記録する（DiT の `_NarrowFloatWatch` と同じ形）。"""

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
        """容器の重み（i8 は packed × scale）= export の fake-quant の値（ビット一致）。"""
        expected = dict(up.Umt5EncoderTokens(quantized).named_parameters())

        mismatched = [
            key for key, value in expected.items() if not torch.equal(weights.tensor(key), value)
        ]

        # 層ごとに q / k / v / o・相対位置の表・RMSNorm 2 本・wi_0 / wi_1 / wo の 10 本
        # + 語彙埋め込み・最後の norm。
        assert len(expected) == 10 * LAYERS + 2
        assert mismatched == []

    def test_rows_are_the_rows_of_the_whole_table(self, weights):
        indices = torch.tensor([0, 3, 17, 200, 383])

        rows = weights.rows(ur.EMBED_KEY, indices)

        assert torch.equal(rows, weights.tensor(ur.EMBED_KEY)[indices])

    def test_unsorted_rows_are_rejected(self, weights):
        with pytest.raises(ur.Umt5ReferenceError, match="昇順"):
            weights.rows(ur.EMBED_KEY, torch.tensor([5, 3]))


class TestCheckpointWeights:
    def test_every_weight_is_the_checkpoint_value(self, plain, model):
        """量子化しない読み口の重み = 上流の parameter（ビット一致・tied な語彙埋め込みも）。"""
        expected = dict(up.Umt5EncoderTokens(model).named_parameters())

        mismatched = [
            key for key, value in expected.items() if not torch.equal(plain.tensor(key), value)
        ]

        assert len(expected) == 10 * LAYERS + 2
        assert mismatched == []

    def test_rows_are_the_rows_of_the_whole_table(self, plain):
        """連続する区間と飛び飛びの行が混ざっても、表の同じ行を同じ並びで返す。"""
        indices = torch.tensor([0, 1, 2, 17, 18, 200, 383])

        rows = plain.rows(ur.EMBED_KEY, indices)

        assert torch.equal(rows, plain.tensor(ur.EMBED_KEY)[indices])

    def test_rows_past_the_table_fail_loudly(self, plain):
        size = plain.tensor(ur.EMBED_KEY).shape[0]

        with pytest.raises(ur.Umt5ReferenceError, match="表の外"):
            plain.rows(ur.EMBED_KEY, torch.tensor([0, size]))

    def test_an_unmapped_key_fails_loudly(self, plain):
        with pytest.raises(ur.Umt5ReferenceError, match="checkpoint のキーが無い"):
            plain.tensor("encoder.block.0.unknown.weight")


class TestUnquantizedReference:
    def test_f32_is_the_unquantized_upstream_eager_bit_for_bit(self, plain, model):
        """量子化なしの f32 = 量子化しない上流の eager（tanh 近似）— 有効長 2〜512 の全部で。"""
        upstream = upstream_tanh(model)

        actual = run(plain, model, torch.float32)

        for length, output in zip(LENGTHS, actual, strict=True):
            expected = up.valid_output(upstream, probe.tiny_ids(length))
            assert torch.equal(output, expected), f"L = {length}: {up.compare(output, expected)}"

    def test_it_differs_from_the_i8_reference(self, plain, weights, model):
        """対（非恒真）: 量子化なしの参照は i8 の参照と割れる（読み口が容器へ落ちていない）。"""
        (unquantized,) = run(plain, model, torch.float32, lengths=(28,))
        (i8,) = run(weights, model, torch.float32, lengths=(28,))

        assert not torch.equal(unquantized, i8)


class TestFloat32Reference:
    def test_it_is_the_upstream_eager_bit_for_bit(self, weights, quantized, model):
        """f32 の書き下し = 上流の eager（tanh 近似・fake-quant 済み）— 有効長 2〜512 の全部で。"""
        upstream = upstream_tanh(quantized)

        actual = run(weights, model, torch.float32)

        for length, output in zip(LENGTHS, actual, strict=True):
            expected = up.valid_output(upstream, probe.tiny_ids(length))
            assert torch.equal(output, expected), f"L = {length}: {up.compare(output, expected)}"

    def test_swapped_layer_tables_break_the_match(self, weights, quantized, model):
        """対（非恒真）: 層 0 / 1 の相対位置の表を入れ替えると上流と割れる。"""
        upstream = upstream_tanh(quantized)

        (actual,) = run(SwappedTables(weights), model, torch.float32, lengths=(28,))

        assert not torch.equal(actual, up.valid_output(upstream, probe.tiny_ids(28)))


class TestFloat64Reference:
    def test_no_narrow_float_is_made(self, weights, model):
        source = Preloaded(weights)
        # 入力（バケット表は meta の上流の attention の式で作る）も監視の外で作る。
        inputs = [case_inputs(model, length) for length in LENGTHS]
        watch = NarrowFloatWatch()

        with watch:
            forward(source, model, torch.float64, inputs)

        assert watch.found == []

    def test_the_watch_sees_the_upstream_double_narrowing(self, quantized):
        """対（非恒真）: 上流を `.double()` した forward は RMSNorm と softmax で f32 に落ちる —
        f64 の参照を上流の `.double()` で採らない理由（ADR 0119 追記「10c の準備」）。"""
        upstream = upstream_tanh(quantized).double()
        watch = NarrowFloatWatch()

        with watch:
            up.valid_output(upstream, probe.tiny_ids(28))

        assert watch.found != []

    def test_it_agrees_with_f32_within_f32_rounding(self, weights, model):
        """同じ関数を書き下している（取り違えは O(1) の比で出る）。"""
        f64 = run(weights, model, torch.float64)
        f32 = run(weights, model, torch.float32)

        ratios = [up.max_ratio(low, high) for low, high in zip(f32, f64, strict=True)]

        assert all(0 < ratio < F32_ROUNDING_RATIO for ratio in ratios), ratios

    def test_the_bf16_mismatch_is_bf16_error(self, weights, quantized, model):
        """上限の比較: f32 自身の誤差（f64 に対する比）は、上流の bf16 と f32 の不一致の
        1% 未満 — 不一致は bf16 の誤差で、品質の記録で f32 を基準にしてよい
        （ADR 0119 追記「10b の結果」）。"""
        bf16_model = upstream_tanh(quantized).to(torch.bfloat16)
        f64 = run(weights, model, torch.float64)
        f32 = run(weights, model, torch.float32)

        for length, high, low in zip(LENGTHS, f64, f32, strict=True):
            bf16 = up.valid_output(bf16_model, probe.tiny_ids(length)).float()
            mismatch = up.max_ratio(bf16, low)
            own = up.max_ratio(low, high)
            assert own < F32_SHARE_OF_BF16_MISMATCH * mismatch, (length, own, mismatch)


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
            ur.golden_metadata(case, weights, ur.reference_ratios(f64, f32), {"torch": "x"}),
        )

        with safe_open(str(path), framework="pt") as handle:
            metadata = handle.metadata()
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118
        assert set(metadata) == {ur.METADATA_KEY}
        meta = json.loads(metadata[ur.METADATA_KEY])
        assert (meta["format"], meta["length"], meta["prompt"]) == (ur.REFERENCE_FORMAT, 28, "原文")
        assert meta["container"]["part0Sha256"] == weights.part0_sha256
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

        with pytest.raises(ur.Umt5ReferenceError, match="受入れだけ"):
            ur.golden_tensors(case, buckets, output, output, pair)


def fake_encode(text: str) -> Mapping[str, Any]:
    """語 1 つ = 1 トークン + 末尾の `</s>`（選び方の検査用 — 長さだけが効く）。"""
    words = text.split()
    return {"cleaned": " ".join(words), "ids": [5] * len(words) + [1]}


def fake_parity(count: int = 200) -> list[dict[str, Any]]:
    """長さ 3〜42 の受理した乱択（`fake_encode` と同じ id 列）と、拒否の 1 本。"""
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

        with pytest.raises(ur.Umt5ReferenceError, match="パリティ fixture と違う"):
            ur.band_cases(parity, fake_encode)
