"""quant の `requiredLimits` 導出（`karume.limits`）の単体テスト。

見るのは**純関数の層**だけ — 現物（系列コンテナ）から入口を辿る結線は
`test_dist.py` の `TestRequiredLimits`。ここでは「何を需要と数えるか」「どこから焼くか」を
合成の宣言だけで固定する。
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from karume.limits import (
    WEBGPU_DEFAULT_LIMITS,
    LimitsError,
    max_state_slot_bytes,
    max_tensor_payload,
    required_limits,
    state_bindings,
)

#: WebGPU 仕様の保証既定（読みやすさのための別名 — 値の門は {@link TestWebgpuDefaults}）。
_BINDING = WEBGPU_DEFAULT_LIMITS["maxStorageBufferBindingSize"]
_BUFFER = WEBGPU_DEFAULT_LIMITS["maxBufferSize"]


def _tensor(begin: int, end: int, dtype: str = "F32") -> dict[str, Any]:
    """safetensors ヘッダのテンソル 1 件（導出が読むのは `data_offsets` だけ）。"""
    return {"dtype": dtype, "shape": [end - begin], "data_offsets": [begin, end]}


class TestWebgpuDefaults:
    """既定値は**外部仕様の写し**なので、値そのものを門にする。"""

    def test_the_defaults_are_the_two_size_limits_of_the_spec(self) -> None:
        """1 バイトでも違うと「焼く / 焼かない」の境界が実機とずれる。"""
        assert WEBGPU_DEFAULT_LIMITS == {
            "maxBufferSize": 268_435_456,
            "maxStorageBufferBindingSize": 134_217_728,
        }

    def test_it_names_no_workgroup_limit(self) -> None:
        """workgroup 系はカーネル設計が決めるので配布物からは導けない（焼かない席）。"""
        assert not any(name.startswith("maxCompute") for name in WEBGPU_DEFAULT_LIMITS)


class TestRequiredLimitsSelection:
    """焼くのは**保証既定を超える席だけ**（「欄が無い = 既定スペックで動く」の意味論）。"""

    def test_a_demand_within_both_defaults_declares_nothing(self) -> None:
        assert required_limits(1024) == {}

    def test_a_demand_exactly_at_the_binding_default_declares_nothing(self) -> None:
        """境界（ちょうど既定値）は満たされている — 要求に書くと何も制約しない欄が増える。"""
        assert required_limits(_BINDING) == {}

    def test_a_demand_past_the_binding_default_declares_only_the_binding(self) -> None:
        """既定の違う 2 つなので、128MiB〜256MiB の帯では binding だけが要求になる。"""
        demand = _BINDING + 1

        assert required_limits(demand) == {"maxStorageBufferBindingSize": demand}

    def test_a_demand_exactly_at_the_buffer_default_still_leaves_the_buffer_out(self) -> None:
        """buffer 側の境界（= 既定ちょうど）も焼かない。binding 側は超えているので残る。"""
        assert required_limits(_BUFFER) == {"maxStorageBufferBindingSize": _BUFFER}

    def test_a_demand_past_both_defaults_declares_both_with_the_same_value(self) -> None:
        """重みも state も 1 バッファ = 1 binding なので、2 つの要求値は同じ。"""
        demand = _BUFFER + 1

        assert required_limits(demand) == {
            "maxBufferSize": demand,
            "maxStorageBufferBindingSize": demand,
        }

    def test_the_declared_names_keep_a_stable_order(self) -> None:
        """焼き直しで欄の並びが揺れない（manifest のバイト列が理由なく動かない）。"""
        assert list(required_limits(_BUFFER + 1)) == [
            "maxBufferSize",
            "maxStorageBufferBindingSize",
        ]

    def test_it_refuses_a_negative_demand(self) -> None:
        with pytest.raises(LimitsError, match="負"):
            required_limits(-1)


class TestTensorPayloadDemand:
    """需要の 1 本目 — **格納 payload そのままの**最大テンソル。"""

    def test_it_takes_the_largest_payload_of_the_header(self) -> None:
        header = {"a": _tensor(0, 64), "b": _tensor(64, 4160), "c": _tensor(4160, 4176)}

        assert max_tensor_payload(header, "where") == 4096

    def test_it_ignores_the_metadata_entry(self) -> None:
        """`__metadata__`（グラフ JSON）はテンソルではない — 数 MB でも需要にならない。"""
        header = {"__metadata__": {"karume_ir": "x" * 4096}, "a": _tensor(0, 64)}

        assert max_tensor_payload(header, "where") == 64

    def test_a_graph_shard_has_no_demand_at_all(self) -> None:
        """ADR 0081 のグラフ shard（データ節が空）は 0 — 呼び手が全 shard の最大を採る。"""
        assert max_tensor_payload({"__metadata__": {"karume_ir": "{}"}}, "where") == 0

    def test_the_stored_width_is_what_counts(self) -> None:
        """i4 の 1 本は i4 の寸法のまま（f32 へ展開した後の寸法を要求に書かない）。"""
        header = {"packed": _tensor(0, 512, dtype="I8"), "scale": _tensor(512, 640)}

        assert max_tensor_payload(header, "where") == 512

    @pytest.mark.parametrize(
        "offsets", [[0], [0, 16, 32], "0-16", [16, 0], [-16, 0], [0.0, 16.0], [True, False]]
    )
    def test_it_refuses_a_broken_offsets_declaration(self, offsets: Any) -> None:
        """壊れた宣言を 0 として素通しすると、需要が黙って小さく出る。"""
        with pytest.raises(LimitsError, match="data_offsets"):
            max_tensor_payload({"a": {"dtype": "F32", "data_offsets": offsets}}, "where")


class TestStateSlotDemand:
    """需要の 2 本目 — **容量ぶん常駐する** state スロット（ADR 0066 決定 2）。"""

    _CONFIG: ClassVar[dict[str, Any]] = {"chunkLength": 32, "capacity": 640}

    def test_a_graph_without_states_has_no_state_demand(self) -> None:
        """states を持たない既存の全モデルは無風（欄そのものが無い）。"""
        assert max_state_slot_bytes({"symbols": [], "nodes": []}, self._CONFIG, "where") == 0

    def test_a_numeric_slot_is_counted_as_f32_elements(self) -> None:
        """sliding スロットは窓幅を数値で持つ（f32 = 4 バイト/要素）。"""
        graph = {"states": {"l0.k": {"dtype": "f32", "shape": [1, 2, 512, 256]}}}

        assert max_state_slot_bytes(graph, self._CONFIG, "where") == 2 * 512 * 256 * 4

    def test_a_symbolic_slot_is_bound_by_the_pipeline_config_capacity(self) -> None:
        """full スロットは容量記号のまま焼かれる — 数値化の値は配布形の既定容量。"""
        graph = {"states": {"l4.k": {"dtype": "f32", "shape": [1, 1, "C", 512]}}}

        assert max_state_slot_bytes(graph, self._CONFIG, "where") == 640 * 512 * 4

    def test_it_evaluates_a_derived_dimension(self) -> None:
        """`coeff·sym+offset` の派生形も束縛して数える（ADR 0057 の次元言語）。"""
        graph = {"states": {"s": {"dtype": "f32", "shape": ["2C+8"]}}}

        assert max_state_slot_bytes(graph, self._CONFIG, "where") == (2 * 640 + 8) * 4

    def test_it_takes_the_largest_slot_of_the_graph(self) -> None:
        """スロットは 1 本ずつ別バッファ — 合計ではなく最大が binding / buffer の需要。"""
        graph = {
            "states": {
                "sliding": {"dtype": "f32", "shape": [1, 1, 512, 256]},
                "full": {"dtype": "f32", "shape": [1, 1, "C", 256]},
            }
        }

        assert max_state_slot_bytes(graph, self._CONFIG, "where") == 640 * 256 * 4

    def test_a_large_capacity_outgrows_any_single_tensor(self) -> None:
        """重みは shard 上限で割れるが、スロットは 1 本のまま — 長い会話は state が支配する。

        gemma4 の full スロット（`[1, 1, C, 512]`）を容量 131,072 で数えるとちょうど 256MiB =
        buffer 既定そのもので、binding 既定（128MiB）だけを超える帯に入る。
        """
        graph = {"states": {"full": {"dtype": "f32", "shape": [1, 1, "C", 512]}}}

        demand = max_state_slot_bytes(graph, {"capacity": 131_072}, "where")

        assert demand == 131_072 * 512 * 4 == _BUFFER
        assert required_limits(demand) == {"maxStorageBufferBindingSize": demand}

    @pytest.mark.parametrize(
        "config", [{}, {"capacity": 0}, {"capacity": "640"}, {"capacity": True}]
    )
    def test_it_refuses_a_capacity_the_distribution_does_not_pin(self, config: Any) -> None:
        """束縛が取れないなら fail loudly — 黙って state を外すと「宣言があるのに足りない」。"""
        graph = {"states": {"full": {"dtype": "f32", "shape": [1, 1, "C", 512]}}}

        with pytest.raises(LimitsError, match="capacity"):
            max_state_slot_bytes(graph, config, "where")

    def test_it_refuses_two_symbols_it_cannot_tell_apart(self) -> None:
        """容量記号は 1 本 — 2 本あるとどちらが容量かを配布形からは決められない。"""
        states = {
            "a": {"dtype": "f32", "shape": [1, "C", 8]},
            "b": {"dtype": "f32", "shape": [1, "D", 8]},
        }

        with pytest.raises(LimitsError, match="複数"):
            state_bindings(states, self._CONFIG, "where")

    def test_a_numeric_only_graph_needs_no_binding(self) -> None:
        """記号が無ければ容量席も要らない（pipelineConfig を持たない family も通る）。"""
        states = {"a": {"dtype": "f32", "shape": [1, 512, 8]}}

        assert state_bindings(states, {}, "where") == {}

    def test_it_refuses_a_slot_dtype_it_cannot_size(self) -> None:
        """f16 は席だけの予約（ADR 0066 追記 5）— 寸法を勝手に決めない。"""
        graph = {"states": {"s": {"dtype": "f16", "shape": [1, 512]}}}

        with pytest.raises(LimitsError, match="dtype"):
            max_state_slot_bytes(graph, self._CONFIG, "where")

    @pytest.mark.parametrize("shape", [[], "512", [0, 8], [1, 1.5], [1, None]])
    def test_it_refuses_a_shape_that_is_not_a_concrete_capacity(self, shape: Any) -> None:
        graph = {"states": {"s": {"dtype": "f32", "shape": shape}}}

        with pytest.raises(LimitsError):
            max_state_slot_bytes(graph, self._CONFIG, "where")
