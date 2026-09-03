"""Gemma 4 の RoPE cos / sin 行を**式から**組む（TS 正本の Python 鏡像）と、その宣言の導出。

RoPE 表はもうグラフに焼かない（初期化子ではなく**派生入力** — `gemma4.export_decode` の
module docstring）。表を作るのはホスト（TS 側 `packages/models/src/gemma/rope.ts`）で、
配布形が宣言するのは表そのものではなく**式のパラメータ**（層種別ごとの theta / headDim /
rotaryDim）と位置の上限だけになる。ここが持つのは 2 つ:

- {@link rope_specs} — 上流 config から宣言（`pipelineConfig.rope`）を**導出**する。写経すると
  チェックポイントを差し替えた日に宣言だけが古びる。
- {@link rope_rows} — その宣言から cos / sin 行を組む。TS 側と**同じ式**で、計算は f64・
  格納だけ f32。台本が golden / io / 例示入力に使う表はこの経路で組む（配布形が読む表と
  同じ式で採る — 別の表で採った期待列を検収門が読む形にしない）。

MUST: 上流とビット同一にはならない。上流の rotary は inv_freq も角度も cos / sin も**全部
f32** で通す（modeling_gemma4.py:1170-1179）ので、f64 で通してから丸めた値とは最終 1 ULP から
先が食い違う（位置が大きいほど角度の f32 表現誤差が効き、実測で P=131,072 のとき最大 9.4e-3）。
突合は位置比例の許容差で見る（{@link gemma4.tests.test_rope}）— ここを「ビット一致」に
締めると、正しい実装が落ちる門になる。

MUST: torch を import しない。`gemma4.distribution`（配布 recipe — モデル依存グループ無しの
CI job でも読まれる）がこのモジュールを読むので、重い依存を持ち込むと既定 sync の環境で
collection ごと落ちる（`tests/test_optional_group_imports.py` の門と同じ理由）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: 層種別の綴り（上流 `config.layer_types` の語彙）。`gemma4.export` はここを再輸出する
#: （同じ文字列を 2 箇所で綴らない）。
FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"

#: 受理する rope_type。`default` は全周波数、`proportional` は先頭 `rotaryDim/2` 本だけが
#: 非零で残りが零周波数（= cos 1 / sin 0 の列）。それ以外は式が別物なので fail loudly。
DEFAULT_ROPE_TYPE = "default"
PROPORTIONAL_ROPE_TYPE = "proportional"
ROPE_TYPES = (DEFAULT_ROPE_TYPE, PROPORTIONAL_ROPE_TYPE)

#: 退役した「表をグラフへ焼く」形が残す initializer 名の断片（`…rotary_emb.<層種>_cos_table`）。
#: 焼く側（{@link gemma4.export_decode.assert_rope_inputs}）と配る側
#: （{@link gemma4.distribution.assert_gemma4_graph}）が**同じ綴り**で残骸を落とすための正本。
BAKED_TABLE_INFIX = "rotary_emb."

#: `pipelineConfig.rope` の欄名（TS 側 `parseGemma4PipelineConfig` の綴り）。
THETA_FIELD = "theta"
HEAD_DIM_FIELD = "headDim"
ROTARY_DIM_FIELD = "rotaryDim"


class RopeSpecError(ValueError):
    """RoPE の宣言を上流 config から導けない（未知の rope_type・写せない係数）。"""


@dataclass(frozen=True)
class RopeLayerSpec:
    """層種別 1 つぶんの RoPE の式（`pipelineConfig.rope.<層種別>` の中身）。

    `head_dim` は 1 行の幅（cos / sin の最終次元）、`rotary_dim` は**実際に回る**幅。
    `rotary_dim < head_dim` の残りは零周波数で、cos = 1 / sin = 0 の定数列として行に入る
    （full 層の `proportional` — 上流 `_compute_proportional_rope_parameters` の nope 部分）。
    """

    theta: float
    head_dim: int
    rotary_dim: int

    def __post_init__(self) -> None:
        if self.theta <= 1.0:
            raise RopeSpecError(f"theta {self.theta} が 1 より大きくない")
        for name, value in (("headDim", self.head_dim), ("rotaryDim", self.rotary_dim)):
            if value < 2 or value % 2:
                raise RopeSpecError(f"{name} {value} が 2 以上の偶数でない")
        if self.rotary_dim > self.head_dim:
            raise RopeSpecError(f"rotaryDim {self.rotary_dim} が headDim {self.head_dim} を超える")

    def declaration(self) -> dict[str, Any]:
        """`pipelineConfig.rope` に載る形（欄名は TS 側の綴り）。"""
        return {
            THETA_FIELD: float(self.theta),
            HEAD_DIM_FIELD: int(self.head_dim),
            ROTARY_DIM_FIELD: int(self.rotary_dim),
        }


def _rope_parameters(config: Any, layer_type: str) -> Mapping[str, Any]:
    parameters = getattr(config, "rope_parameters", None)
    if not isinstance(parameters, Mapping):
        raise RopeSpecError(f"config.rope_parameters がオブジェクトでない（{parameters!r}）")
    entry = parameters.get(layer_type)
    if not isinstance(entry, Mapping):
        raise RopeSpecError(f"config.rope_parameters['{layer_type}'] が無い（{entry!r}）")
    return entry


def _head_dim(config: Any, layer_type: str, rope_type: str) -> int:
    """上流と**同じ引き先**で head_dim を決める（`head_dim_key` の分岐 — 1106-1120 行）。

    full 層かつ `proportional` のときだけ `global_head_dim` を読む。ここを揃えないと、
    full 層の表が 256 幅で組まれて attention の 512 幅と噛み合わない。
    """
    key = (
        "global_head_dim"
        if layer_type == FULL_ATTENTION and rope_type == PROPORTIONAL_ROPE_TYPE
        else "head_dim"
    )
    declared = getattr(config, key, None)
    if not declared:
        declared = int(config.hidden_size) // int(config.num_attention_heads)
    return int(declared)


def layer_spec(config: Any, layer_type: str) -> RopeLayerSpec:
    """層種別 1 つの宣言を上流 config から導く（値は 1 つも写経しない）。

    MUST: 受理外の rope_type / `factor` / `attention_factor` は fail loudly。式が別物なのに
    theta と幅だけを写すと、**形も型も合う別の角度**でホストが表を組む（沈黙誤値）。
    """
    parameters = _rope_parameters(config, layer_type)
    rope_type = parameters.get("rope_type", DEFAULT_ROPE_TYPE)
    if rope_type not in ROPE_TYPES:
        raise RopeSpecError(
            f"層種別 '{layer_type}' の rope_type が '{rope_type}' — 写せるのは {list(ROPE_TYPES)}"
        )
    for name in ("factor", "attention_factor"):
        value = parameters.get(name, 1.0)
        if float(value) != 1.0:
            raise RopeSpecError(
                f"層種別 '{layer_type}' の {name} が {value} — 1 以外は式が別物になる"
            )
    head_dim = _head_dim(config, layer_type, rope_type)
    if rope_type == PROPORTIONAL_ROPE_TYPE:
        proportion = float(parameters.get("partial_rotary_factor", 1.0))
        rotary_dim = 2 * int(proportion * head_dim // 2)
    else:
        rotary_dim = head_dim
    theta = parameters.get("rope_theta")
    if not isinstance(theta, int | float) or isinstance(theta, bool):
        raise RopeSpecError(f"層種別 '{layer_type}' の rope_theta が数でない（{theta!r}）")
    return RopeLayerSpec(theta=float(theta), head_dim=head_dim, rotary_dim=rotary_dim)


def rope_specs(config: Any) -> dict[str, RopeLayerSpec]:
    """`config.layer_types` に現れる層種別（**出現順**）ぶんの宣言。"""
    layer_types = getattr(config, "layer_types", None)
    if not isinstance(layer_types, Sequence) or isinstance(layer_types, str) or not layer_types:
        raise RopeSpecError(f"config.layer_types が非空の配列でない（{layer_types!r}）")
    return {
        layer_type: layer_spec(config, layer_type)
        for layer_type in dict.fromkeys(str(name) for name in layer_types)
    }


def inverse_frequencies(spec: RopeLayerSpec) -> np.ndarray:
    """周波数種 `[headDim/2]`（f64）。先頭 `rotaryDim/2` 本だけが非零。

    `invFreq[i] = theta ** (-(2i) / headDim)`。上流は `1.0 / theta ** ((2i)/headDim)` と
    書くが、割り算の向きは f64 では最終 1 ULP でしか差が出ない（そもそもビット一致は
    成立しない — module docstring）ので、TS 正本と**同じ綴り**に揃える。
    """
    half = spec.head_dim // 2
    frequencies = np.zeros(half, dtype=np.float64)
    rotated = spec.rotary_dim // 2
    index = np.arange(rotated, dtype=np.float64)
    frequencies[:rotated] = float(spec.theta) ** (-(2.0 * index) / float(spec.head_dim))
    return frequencies


def rope_rows(spec: RopeLayerSpec, positions: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """位置列 → `(cos, sin)` の行（`[len(positions), headDim]` の f32）。

    角度は `pos × invFreq` で、行は `cat(freqs, freqs)`（前半と後半が同一 — 上流
    `emb = torch.cat((freqs, freqs), dim=-1)`）。`attention_scaling` は受理する 2 つの
    rope_type ではどちらも 1 なので掛けない（1 以外は {@link layer_spec} が落とす）。

    MUST: 計算は f64・**丸めは最後の 1 回だけ**。角度を f32 で持つと位置が大きいほど誤差が
    効く（上流がそうなっている側で、こちらは合わせに行かない — module docstring）。
    """
    angles = np.asarray(positions, dtype=np.float64).reshape(-1, 1) * inverse_frequencies(spec)
    doubled = np.concatenate((angles, angles), axis=1)
    return np.cos(doubled).astype(np.float32), np.sin(doubled).astype(np.float32)
