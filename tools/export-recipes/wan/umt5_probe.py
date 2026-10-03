"""小さな乱数の umT5 で、有効長 `L` の S 形 export と段 10c の前提を CPU で確かめる台本
（ADR 0119 段 10c の下見 — 決定 3〜5）。実重みは読まない（構成は {@link TINY_CONFIG} の手書き）。

    uv run --group wan --inexact python -m wan.umt5_probe   # 計測して JSON を出す

確かめること（どれも JSON の欄 — pytest `wan/tests/test_umt5_patch.py` が同じ関数で縛る）:

- `export`: {@link wan.umt5_patch.Umt5EncoderTokens} が exporter（karume）を通り、i8 の格納の計画と
  `publish_model` が書き出しの前に掛ける検査（`assert_runtime_support` / `assert_op_contracts`）を
  通る。IR の op の本数・入力（token id `[1, L]` とバケット表 `[L, L]` の i32）・出力
  `[1, L, d_model]`・相対位置の表の格納（F32）と、golden（i8 の fake-quant 後の重みで回した
  ラッパの CPU f32 の出力）の形を返す。容器のファイルを書いて読み直すのは pytest（理由は
  {@link export_probe}）。
- `export.emit_without_overrides`: 相対位置の表を i8 の対象から外しただけで F32 の明示を渡さないと
  emit が落ちること（決定 5 の「F32 のままにする指定の形」）。
- `quant_targets`: exporter の i8 の既定の対象が相対位置の表（`nn.Embedding [32, heads]`）を含むか。
- `activation_rounding`: `gelu_new` の手書き式（上流）と `nn.GELU(approximate="tanh")`（差し替え）の
  要素ごとの差。参考に `pow(x, 3)` を `x·x·x` に書き換えた式（exporter の語彙の mul / add / tanh
  だけで書ける形）も並べる。
- `wrapper`: 活性を上流のままにしたラッパが上流の forward とビット一致すること（書き下しの同値）と、
  差し替えたラッパと上流の差（活性の丸めだけ）。
- `mask`: 「有効長だけ」と「512 まで詰めてマスク」（上流 `_get_t5_prompt_embeds` の写し方）の一致を
  f32 と bf16 で（調査 §2.3 の再現 — 実モデルでの一致は段 10c の検収）。

MUST: transformers は関数の中で import する（`wan/umt5_patch.py` と同じ理由）。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from karume.container import Provenance
from karume.convert import normalize_boundary_tensor
from karume.emit import EmitError, storage_breakdown, stored_model
from karume.ir import IrGraph
from karume.pipeline import export_module
from karume.verify import assert_op_contracts, assert_runtime_support
from wan import umt5_patch

#: 小さな umT5 の構成。実モデル（`text_encoder/config.json`）と**構造の欄**（ゲート付き `gelu_new`・
#: バケット 32・max_distance 128・eps 1e-6・pad 0 / eos 1）を揃え、大きさだけを縮める。heads と
#: d_kv は別の値にする（`[L, L, heads]` → `[1, heads, L, L]` の軸の取り違えが形で落ちるように）。
TINY_CONFIG: Mapping[str, Any] = {
    "vocab_size": 384,
    "d_model": 64,
    "d_kv": 16,
    "num_heads": 4,
    "d_ff": 160,
    "num_layers": 4,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "feed_forward_proj": "gated-gelu",
    "layer_norm_epsilon": 1e-6,
    "dropout_rate": 0.0,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "is_encoder_decoder": False,
}

#: 乱数の初期化の seed（重み・token id）。
SEED = 20261003

#: 比べる有効長。下限 2・固定 4 プロンプトの長さ（28 / 118 / 50 / 126）・max_distance 128 を
#: 越える長さ・上限の手前と上限（決定 4）。
LENGTHS: tuple[int, ...] = (2, 28, 50, 118, 126, 300, 511, 512)

#: 容器と golden を採る有効長（小さい方と大きい方 — 同じ容器が 2 つの L で回ることを golden で
#: 示す）。
EXPORT_LENGTHS: tuple[int, ...] = (28, 126)

#: pytest が書く下見の容器のグラフ名（配布形の部品名の候補 — 決定は段 10d）。
TARGET = "text_encoder"
MODEL_FILE = "model.krm"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."
IO_PREFIX = "io."
CASE_SUFFIX = ".safetensors"

#: 活性の比較に使う標本数（標準正規 × 尺度ごと）と尺度（umT5 の FFN の前活性は O(1)〜O(10)）。
ACTIVATION_SAMPLES = 1_000_000
ACTIVATION_SCALES: tuple[float, ...] = (0.1, 1.0, 3.0, 10.0)

#: ULP 差を数える参照の絶対値の下限（{@link _elementwise}）。
ULP_FLOOR = 2.0**-8


def tiny_model(seed: int = SEED) -> nn.Module:
    """{@link TINY_CONFIG} の `UMT5EncoderModel`（上流の初期化の乱数・eval・f32）。"""
    from transformers import UMT5Config, UMT5EncoderModel

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return UMT5EncoderModel(UMT5Config(**TINY_CONFIG)).eval()


def tiny_ids(length: int, seed: int = SEED) -> torch.Tensor:
    """有効長 `length` の token id `[1, L]`（本文は特殊トークン以外の乱数・末尾は `</s>`）。"""
    generator = torch.Generator().manual_seed(seed + length)
    body = torch.randint(
        3, int(TINY_CONFIG["vocab_size"]), (1, length - 1), generator=generator, dtype=torch.long
    )
    eos = torch.full((1, 1), int(TINY_CONFIG["eos_token_id"]), dtype=torch.long)
    return torch.cat([body, eos], dim=1)


def case_inputs(wrapper: umt5_patch.Umt5EncoderTokens, length: int) -> tuple[torch.Tensor, ...]:
    """ラッパの入力（token id とバケット表 — 表は 0 層目の上流の式で作る）。"""
    attention = wrapper.encoder.block[0].layer[0].SelfAttention
    return tiny_ids(length), umt5_patch.relative_position_buckets(length, attention)


def activation_rounding(seed: int = SEED) -> dict[str, Any]:
    """上流の `gelu_new` と 2 つの書き換えの要素ごとの差（同じ f32 の入力）。"""
    from transformers.activations import NewGELUActivation

    generator = torch.Generator().manual_seed(seed)
    inputs = torch.cat(
        [
            torch.randn(ACTIVATION_SAMPLES, generator=generator) * scale
            for scale in ACTIVATION_SCALES
        ]
    )
    upstream = NewGELUActivation()(inputs)
    tanh = nn.GELU(approximate="tanh")(inputs)
    inner = math.sqrt(2.0 / math.pi) * (inputs + 0.044715 * (inputs * inputs * inputs))
    multiplied = 0.5 * inputs * (1.0 + torch.tanh(inner))
    return {
        "samples": int(inputs.numel()),
        "pow3_equals_mul": bool(torch.equal(torch.pow(inputs, 3.0), inputs * inputs * inputs)),
        "tanh": _elementwise(tanh, upstream),
        "mul": _elementwise(multiplied, upstream),
    }


def _elementwise(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """違う要素の割合・最大絶対差・最大の ULP 差（参照の値の ULP で測る）。

    ULP 差は参照の絶対値が {@link ULP_FLOOR} 以上の要素だけで見る — 0 の近く（大きな負の入力で
    GELU が −1e-20 級になる所）では ULP が極端に小さく、絶対差 1e-9 級でも ULP 差が 1e7 になって
    丸めの大きさを表さない。
    """
    differing = actual != expected
    magnitude = expected.abs()
    spacing = (torch.nextafter(magnitude, torch.tensor(float("inf"))) - magnitude).double()
    ulps = (actual.double() - expected.double()).abs() / spacing
    return {
        "differing": float(differing.double().mean()),
        "max_abs": float((actual - expected).abs().max()),
        "max_ulp": float(ulps[magnitude >= ULP_FLOOR].max()),
    }


def wrapper_comparison(model: nn.Module, lengths: Sequence[int] = LENGTHS) -> list[dict[str, Any]]:
    """上流の forward に対する、活性を上流のままにしたラッパと差し替えたラッパの一致。"""
    exact = umt5_patch.Umt5EncoderTokens(model, activation="upstream")
    tanh = umt5_patch.Umt5EncoderTokens(model, activation="tanh")
    rows = []
    for length in lengths:
        inputs = case_inputs(exact, length)
        reference = umt5_patch.valid_output(model, inputs[0])
        with torch.no_grad():
            rows.append(
                {
                    "length": length,
                    "upstream_activation": dict(umt5_patch.compare(exact(*inputs), reference)),
                    "tanh_activation": dict(umt5_patch.compare(tanh(*inputs), reference)),
                }
            )
    return rows


def mask_equivalence(
    model: nn.Module, dtype: torch.dtype, lengths: Sequence[int] = LENGTHS
) -> list[dict[str, Any]]:
    """「有効長だけ」と「512 + マスク」の一致（上流の forward 同士 — `dtype` で回す）。"""
    cast = copy.deepcopy(model).to(dtype)
    rows = []
    for length in lengths:
        ids = tiny_ids(length)
        valid = umt5_patch.valid_output(cast, ids)
        padded = umt5_patch.padded_output(cast, ids)
        rows.append({"length": length, **umt5_patch.compare(valid, padded)})
    return rows


def quant_targets(model: nn.Module) -> dict[str, Any]:
    """exporter の i8 の既定の対象と、相対位置の表を外した対象（どちらも丸めた後の本数）。"""
    from karume.quantize import fake_quant_int8

    default = fake_quant_int8(umt5_patch.Umt5EncoderTokens(copy.deepcopy(model)))
    wrapper = umt5_patch.Umt5EncoderTokens(copy.deepcopy(model))
    tables = umt5_patch.relative_bias_modules(wrapper)
    selected = umt5_patch.fake_quant_i8(wrapper)
    return {
        "relative_bias_tables": len(tables),
        "default": report_targets(default.scales),
        "default_includes_relative_bias": all(
            f"{name}.weight" in default.scales for name in tables
        ),
        "selected": report_targets(selected.scales),
        "selected_includes_relative_bias": any(
            f"{name}.weight" in selected.scales for name in tables
        ),
    }


def report_targets(scales: Mapping[str, torch.Tensor]) -> dict[str, int]:
    """丸めた重みの本数をモジュールの種類（FQN の末尾の名前）ごとに数える。"""
    return dict(sorted(Counter(key.rsplit(".", 2)[-2] for key in scales).items()))


def provenance() -> Provenance:
    """下見の容器の出所（乱数の重み・配布しない — transformers の UMT5 の実装の版を書く）。"""
    import transformers

    return Provenance(
        license="apache-2.0", writer=f"wan.umt5_probe (transformers {transformers.__version__})"
    )


@dataclass(frozen=True)
class ProbeExport:
    """下見の export の結果（容器のファイルは書かない — 下の {@link export_probe}）。"""

    #: `export_module` の素のグラフと格納テンソル（`publish_model` へそのまま渡せる形）。
    graph: IrGraph
    tensors: Mapping[str, torch.Tensor]
    #: 格納宣言を commit したグラフ（`publish_model` と同じ 2 つの検査を通したもの）。
    stored: IrGraph
    #: i8 の scale 台帳と、相対位置の表の F32 の明示（`publish_model` の引数）。
    scales: Mapping[str, torch.Tensor]
    overrides: Mapping[str, str]
    #: golden（ファイル名 `io.lNNN.safetensors` → グラフ入力 `input.*` と出力 `output.0`）。
    golden: Mapping[str, Mapping[str, torch.Tensor]]
    #: F32 の明示を渡さなかったときの emit の拒否の文言。
    emit_without_overrides: str


def export_probe(model: nn.Module) -> ProbeExport:
    """i8（相対位置の表は F32）の格納まで通し、golden を採る（容器のファイルは書かない）。

    `model` は複製してから丸める（呼び手の上流の重みを変えない）。golden は丸めた後のラッパの
    CPU f32 の出力（ADR 0026 決定 2 / 0119 決定 8 — 参照は同じ fake-quant の重みで採る）。
    書き出しの前に `publish_model` が掛ける検査（`assert_runtime_support` /
    `assert_op_contracts`）は、ここで同じ相手（格納宣言を commit したグラフ）に掛ける。

    容器を書く口（`graph_name=` を名乗る `publish_model`）は pytest の作業席だけに置く: recipe の
    台本が `graph_name=` を名乗ると、全 family 横断の門（`tests/test_graph_names.py`）が配布形の
    部品名（`wan.distribution.WAN_WEIGHTS` のキー）との一致を求める。下見の容器は乱数の重みで
    配布しないので名乗る部品名が無い（umT5 の部品名と配布の構成は段 10d — ADR 0119 裁定 1）。
    """
    wrapper = umt5_patch.Umt5EncoderTokens(copy.deepcopy(model))
    report = umt5_patch.fake_quant_i8(wrapper)
    overrides = umt5_patch.storage_overrides(wrapper)
    cases = {length: case_inputs(wrapper, length) for length in EXPORT_LENGTHS}
    graph, tensors = export_module(
        wrapper,
        cases[EXPORT_LENGTHS[0]],
        dynamic_shapes=umt5_patch.dynamic_shapes(),
        symbol_names=(umt5_patch.SYMBOL,),
    )
    try:
        stored_model(graph, tensors, weight_dtype="i8", weight_scales=report.scales)
    except EmitError as error:
        without_overrides = str(error)
    else:
        raise AssertionError("相対位置の表に scale も F32 の明示も無いのに emit が通った")
    stored = stored_model(
        graph,
        tensors,
        weight_dtype="i8",
        weight_scales=report.scales,
        weight_dtype_overrides=overrides,
    ).graph
    assert_runtime_support(stored)
    assert_op_contracts(stored)
    declared = [entry.name for entry in stored.inputs]
    if declared != list(umt5_patch.INPUT_NAMES):
        raise AssertionError(f"グラフ入力名が宣言と不一致: {declared}")
    golden: dict[str, dict[str, torch.Tensor]] = {}
    for length, inputs in cases.items():
        with torch.no_grad():
            output = wrapper(*inputs)
        io = {
            f"{INPUT_PREFIX}{name}": normalize_boundary_tensor(value, f"L={length} の入力 '{name}'")
            for name, value in zip(umt5_patch.INPUT_NAMES, inputs, strict=True)
        }
        io[f"{OUTPUT_PREFIX}0"] = normalize_boundary_tensor(
            output.contiguous(), f"L={length} の出力"
        )
        golden[f"{IO_PREFIX}l{length:03d}{CASE_SUFFIX}"] = io
    return ProbeExport(
        graph=graph,
        tensors=tensors,
        stored=stored,
        scales=report.scales,
        overrides=overrides,
        golden=golden,
        emit_without_overrides=without_overrides,
    )


def summarize(export: ProbeExport) -> dict[str, Any]:
    """export の要約（op の本数・記号・入出力・格納・golden の形）。"""
    stored = export.stored
    breakdown = storage_breakdown(stored)
    return {
        "ops": dict(sorted(Counter(node.op for node in stored.nodes).items())),
        "symbols": list(stored.symbols),
        "inputs": [[entry.name, list(entry.shape), entry.dtype] for entry in stored.inputs],
        "outputs": [list(stored.values[name].shape) for name in stored.outputs],
        "storage": storage_by_kind(stored, export.overrides),
        "compressed_tensors": breakdown.compressed_tensors,
        "plain_tensors": breakdown.plain_tensors,
        "emit_without_overrides": export.emit_without_overrides,
        "golden": {
            name: {key: list(value.shape) for key, value in io.items()}
            for name, io in export.golden.items()
        },
    }


def storage_by_kind(graph: Any, tables: Mapping[str, str]) -> dict[str, list[str]]:
    """格納 dtype の集合を重みの種類ごとに（相対位置の表・語彙埋め込み・linear・その他）。"""
    kinds: dict[str, set[str]] = {}
    for initializer in graph.initializers.values():
        key = initializer.tensor or ""
        if key in tables:
            kind = umt5_patch.RELATIVE_BIAS_ATTRIBUTE
        elif key == "encoder.embed_tokens.weight":
            kind = "embed_tokens"
        elif (
            key.endswith(".weight") and ".layer_norm." not in key and "final_layer_norm" not in key
        ):
            kind = "linear"
        else:
            kind = "other"
        kinds.setdefault(kind, set()).add(initializer.storage.dtype)
    return {kind: sorted(dtypes) for kind, dtypes in sorted(kinds.items())}


def run() -> dict[str, Any]:
    """全部の計測（JSON の欄の意味はモジュールの docstring）。"""
    model = tiny_model()
    return {
        "config": dict(TINY_CONFIG),
        "export": summarize(export_probe(model)),
        "quant_targets": quant_targets(model),
        "activation_rounding": activation_rounding(),
        "wrapper": wrapper_comparison(model),
        "mask": {
            "f32": mask_equivalence(model, torch.float32),
            "bf16": mask_equivalence(model, torch.bfloat16),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.parse_args(argv)
    started = time.perf_counter()
    summary = run()
    summary["seconds"] = round(time.perf_counter() - started, 1)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
