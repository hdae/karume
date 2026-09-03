"""RoPE の式（`gemma4/rope.py`）と、上流実装との**位置比例の許容差**での突合。

RoPE 表はもうグラフに焼かない（ホストが宣言から組む）ので、正しさの根拠は「表を上流の
出力そのものにした」ことから「**式が上流と同じ角度を作る**」ことへ移った。ここが縛るのは
その 2 本:

- {@link gemma4.rope.rope_specs} が上流 config から theta / headDim / rotaryDim を導けること
  （full 層だけ `global_head_dim` を読む枝を含む）と、写せない宣言を fail loudly で落とすこと
- {@link gemma4.rope.rope_rows} が上流 `Gemma4TextRotaryEmbedding` の実出力と、位置に比例する
  許容差の内側で一致すること

MUST: 突合の相手は**上流モジュールの実出力**（同じ式を 2 回書いて比べない）。恒真でないこと
は故障注入（位置ずらし・層種入れ替え・theta 差し替え・rotaryDim 差し替え）で裏を取る。

MUST: ビット一致では見ない。上流は inv_freq も角度も cos / sin も全部 f32 で通すので、f64 で
組んでから丸めた値とは必ず食い違う（`gemma4.rope` の module docstring）。許容差は
{@link atol_for} — 定数項が cos / sin の 1 ULP（5.96e-8）の数倍、位置比例項が「角度の f32
表現誤差 × 導関数」の実測（P=131,071 で 4.8e-3）に約 2 倍の余裕を持たせた線。

transformers を要するので、上流を呼ぶケースは `importorskip` で SKIP する（ADR 0065 の
2 job 構成）。
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gemma4 import rope
from gemma4.tests import rope_fixture

#: 許容差の 2 項（定数 = 実装差の 1 ULP 数個ぶん / 位置比例 = 角度の f32 表現誤差ぶん）。
ATOL_BASE = 2.5e-7
ATOL_PER_POSITION = 1.2e-7

#: 突合に踏む位置。散点（{@link rope_fixture.FIXTURE_POSITIONS}）に加えて**連続 0..64** も
#: 見る — 散点だけだと「先頭と上限だけ合う」形（周波数の並びが逆など）が素通りしうる。
CONTIGUOUS_POSITIONS: tuple[int, ...] = tuple(range(65))

#: 手写しの宣言（{@link rope_fixture.E2B_TEXT_CONFIG_FIELDS}）を実物へ結ぶ欄。RoPE の角度に
#: 効くのはこの 4 つで、残りの欄は rotary モジュールを組むための足場（層数・hidden_size など）
#: なので cos / sin の値には効かない。
UPSTREAM_ROPE_FIELDS: tuple[str, ...] = (
    "max_position_embeddings",
    "head_dim",
    "global_head_dim",
    "rope_parameters",
)


def atol_for(position: int) -> float:
    """位置 1 つの許容差（位置に比例して開く）。"""
    return ATOL_BASE + position * ATOL_PER_POSITION


def assert_rope_parity(
    got: np.ndarray, want: np.ndarray, positions: Sequence[int], where: str
) -> None:
    """要素ごとに `atol(pos)` で突合する（一番外れた要素を診断に出す）。"""
    if got.shape != want.shape:
        raise AssertionError(f"{where}: 形が {got.shape} — 上流は {want.shape}")
    tolerance = np.array([atol_for(position) for position in positions], dtype=np.float64)
    difference = np.abs(got.astype(np.float64) - want.astype(np.float64))
    excess = difference / tolerance[:, None]
    row, column = np.unravel_index(int(excess.argmax()), excess.shape)
    if excess[row, column] > 1.0:
        raise AssertionError(
            f"{where}: 位置 {positions[row]} の列 {column} が {difference[row, column]:.3e} 差"
            f"（許容 {tolerance[row]:.3e}）"
        )


@pytest.fixture
def e2b_config():
    pytest.importorskip("transformers")
    return rope_fixture.e2b_text_config()


class TestRopeSpecs:
    """宣言の導出 — 値を 1 つも写経せず、写せない宣言は落とす。"""

    def test_it_derives_both_layer_types_from_the_upstream_config(self, e2b_config) -> None:
        specs = rope.rope_specs(e2b_config)

        assert specs[rope.SLIDING_ATTENTION] == rope.RopeLayerSpec(
            theta=10000.0, head_dim=256, rotary_dim=256
        )
        # full 層だけ `global_head_dim`（512）を読み、`proportional` は先頭 128 本だけが回る。
        assert specs[rope.FULL_ATTENTION] == rope.RopeLayerSpec(
            theta=1000000.0, head_dim=512, rotary_dim=128
        )

    def test_the_order_follows_the_layer_type_order(self, e2b_config) -> None:
        """出現順（先頭は sliding）— グラフ入力の並びと同じ根拠を持つ。"""
        assert list(rope.rope_specs(e2b_config)) == [
            rope.SLIDING_ATTENTION,
            rope.FULL_ATTENTION,
        ]

    def test_it_falls_back_to_the_derived_head_dim(self) -> None:
        """`head_dim` を持たない config は `hidden_size / num_attention_heads`（上流と同文）。"""
        config = SimpleNamespace(
            hidden_size=64,
            num_attention_heads=4,
            layer_types=[rope.SLIDING_ATTENTION],
            rope_parameters={rope.SLIDING_ATTENTION: {"rope_type": "default", "rope_theta": 1e4}},
        )

        assert rope.rope_specs(config)[rope.SLIDING_ATTENTION].head_dim == 16

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ({"rope_type": "yarn"}, "rope_type"),
            ({"factor": 2.0}, "factor"),
            ({"attention_factor": 0.5}, "attention_factor"),
            ({"rope_theta": "10000"}, "rope_theta"),
        ],
    )
    def test_it_refuses_a_declaration_it_cannot_mirror(
        self, override: dict[str, object], message: str
    ) -> None:
        """MUST: 式が別物なのに theta と幅だけ写すと、ホストが別の角度で表を組む。"""
        config = SimpleNamespace(
            head_dim=16,
            hidden_size=64,
            num_attention_heads=4,
            layer_types=[rope.SLIDING_ATTENTION],
            rope_parameters={
                rope.SLIDING_ATTENTION: {
                    "rope_type": "default",
                    "rope_theta": 1e4,
                    **override,
                }
            },
        )

        with pytest.raises(rope.RopeSpecError, match=message):
            rope.rope_specs(config)

    def test_it_refuses_a_config_without_layer_types(self) -> None:
        with pytest.raises(rope.RopeSpecError, match="layer_types"):
            rope.rope_specs(SimpleNamespace(layer_types=[]))

    @pytest.mark.parametrize(
        ("theta", "head_dim", "rotary_dim"),
        [(1.0, 8, 8), (1e4, 7, 6), (1e4, 8, 10)],
    )
    def test_the_spec_itself_refuses_impossible_numbers(
        self, theta: float, head_dim: int, rotary_dim: int
    ) -> None:
        with pytest.raises(rope.RopeSpecError):
            rope.RopeLayerSpec(theta=theta, head_dim=head_dim, rotary_dim=rotary_dim)


class TestInverseFrequencies:
    def test_the_nope_tail_is_exactly_zero(self) -> None:
        """`proportional` の零周波数は cos = 1 / sin = 0 の定数列になる席（full 層の 192 本）。"""
        spec = rope.RopeLayerSpec(theta=1e6, head_dim=512, rotary_dim=128)

        frequencies = rope.inverse_frequencies(spec)

        assert frequencies.shape == (256,)
        assert np.count_nonzero(frequencies) == 64
        assert not frequencies[64:].any()

    def test_the_row_repeats_its_first_half(self) -> None:
        """行は `cat(freqs, freqs)`（上流と同じ並び）— 前半と後半が同一。"""
        spec = rope.RopeLayerSpec(theta=1e4, head_dim=16, rotary_dim=16)

        cos, sin = rope.rope_rows(spec, [7])

        assert cos.shape == (1, 16)
        assert np.array_equal(cos[0, :8], cos[0, 8:])
        assert np.array_equal(sin[0, :8], sin[0, 8:])

    def test_position_zero_is_the_identity_rotation(self) -> None:
        cos, sin = rope.rope_rows(rope.RopeLayerSpec(theta=1e4, head_dim=8, rotary_dim=8), [0])

        assert np.array_equal(cos[0], np.ones(8, dtype=np.float32))
        assert np.array_equal(sin[0], np.zeros(8, dtype=np.float32))

    def test_the_rows_are_stored_as_float32(self) -> None:
        """格納だけ f32（計算は f64）— TS 側が f32 の表を渡す契約と同じ。"""
        cos, sin = rope.rope_rows(rope.RopeLayerSpec(theta=1e4, head_dim=8, rotary_dim=8), [1, 2])

        assert cos.dtype == np.float32
        assert sin.dtype == np.float32


class TestUpstreamParity:
    """上流実装との突合（**恒真でない**ことは下の故障注入が裏を取る）。"""

    @pytest.fixture
    def upstream(self, e2b_config):
        def tables(positions: Sequence[int]):
            return rope_fixture.upstream_tables(e2b_config, positions)

        return tables

    @pytest.mark.parametrize(
        "positions",
        [rope_fixture.FIXTURE_POSITIONS, CONTIGUOUS_POSITIONS],
        ids=["scattered", "contiguous"],
    )
    @pytest.mark.parametrize(
        "layer_type", [rope.SLIDING_ATTENTION, rope.FULL_ATTENTION], ids=["sliding", "full"]
    )
    def test_the_mirror_matches_the_upstream_module(
        self, e2b_config, upstream, layer_type: str, positions: Sequence[int]
    ) -> None:
        spec = rope.rope_specs(e2b_config)[layer_type]
        want_cos, want_sin = upstream(positions)[layer_type]

        cos, sin = rope.rope_rows(spec, positions)

        assert_rope_parity(cos, want_cos, positions, f"{layer_type} cos")
        assert_rope_parity(sin, want_sin, positions, f"{layer_type} sin")

    def test_the_tolerance_is_not_slack(self, e2b_config, upstream) -> None:
        """許容差が実測の何倍かを固定する — 緩めすぎれば故障注入も通ってしまう。

        見るのは fixture の 13 点での最悪比（実測 0.548 = 余裕 1.8 倍・full sin の P=511）。
        上限 0.9 は「余裕が 1.1 倍を切ったら上流の f32 誤差が想定より大きい」線で、位置
        0..131,071 を全掃引したときの最悪比 0.756（余裕 1.3 倍）もその内側に入る。下限 0.01 は
        逆に緩すぎる側（余裕 100 倍）を落とす。{@link ATOL_PER_POSITION} は角度の f32 表現誤差の
        **上界**から採った係数なので、この比を理由に動かさない（動かすなら前提の側を採り直す）。
        """
        positions = rope_fixture.FIXTURE_POSITIONS
        worst = 0.0
        for layer_type, spec in rope.rope_specs(e2b_config).items():
            want_cos, want_sin = upstream(positions)[layer_type]
            cos, sin = rope.rope_rows(spec, positions)
            for got, want in ((cos, want_cos), (sin, want_sin)):
                difference = np.abs(got.astype(np.float64) - want.astype(np.float64))
                for row, position in enumerate(positions):
                    worst = max(worst, float(difference[row].max()) / atol_for(position))

        assert 0.01 < worst < 0.9, f"許容差に対する実測の比が {worst}"

    def test_a_position_shifted_by_one_is_rejected(self, e2b_config, upstream) -> None:
        """故障注入: 位置を 1 ずらす（表引き時代の「行がずれた表」と同じ壊れ方）。"""
        positions = rope_fixture.FIXTURE_POSITIONS
        spec = rope.rope_specs(e2b_config)[rope.SLIDING_ATTENTION]
        want_cos, _ = upstream(positions)[rope.SLIDING_ATTENTION]

        cos, _ = rope.rope_rows(spec, [position + 1 for position in positions])

        with pytest.raises(AssertionError):
            assert_rope_parity(cos, want_cos, positions, "shifted")

    def test_a_table_built_for_the_other_layer_type_is_rejected(self, e2b_config, upstream) -> None:
        """故障注入: 層種を入れ替える（theta も幅も違うので形の段で落ちる）。"""
        positions = rope_fixture.FIXTURE_POSITIONS
        specs = rope.rope_specs(e2b_config)
        want_cos, _ = upstream(positions)[rope.FULL_ATTENTION]

        cos, _ = rope.rope_rows(specs[rope.SLIDING_ATTENTION], positions)

        with pytest.raises(AssertionError, match="形が"):
            assert_rope_parity(cos, want_cos, positions, "swapped")

    def test_a_wrong_theta_is_rejected(self, e2b_config, upstream) -> None:
        """故障注入: full 層の theta を 1e6 → 1e5 に（幅も並びも合ったまま角度だけ違う）。"""
        positions = rope_fixture.FIXTURE_POSITIONS
        spec = replace(rope.rope_specs(e2b_config)[rope.FULL_ATTENTION], theta=1e5)
        want_cos, _ = upstream(positions)[rope.FULL_ATTENTION]

        cos, _ = rope.rope_rows(spec, positions)

        with pytest.raises(AssertionError):
            assert_rope_parity(cos, want_cos, positions, "theta")

    def test_a_full_rotary_dim_on_the_partial_layer_is_rejected(self, e2b_config, upstream) -> None:
        """故障注入: full 層の rotaryDim を headDim に（零周波数 192 本が回り出す）。"""
        positions = rope_fixture.FIXTURE_POSITIONS
        spec = rope.rope_specs(e2b_config)[rope.FULL_ATTENTION]
        broken = replace(spec, rotary_dim=spec.head_dim)
        want_cos, _ = upstream(positions)[rope.FULL_ATTENTION]

        cos, _ = rope.rope_rows(broken, positions)

        with pytest.raises(AssertionError):
            assert_rope_parity(cos, want_cos, positions, "rotaryDim")


class TestFixture:
    """TS 側へ渡すフィクスチャが**いま**の上流と同じであること。"""

    def test_the_checked_in_fixture_is_what_the_generator_writes(self, e2b_config) -> None:
        """MUST: 再生成して**バイト単位で**同じ — 古びたフィクスチャは TS 側の門を静かに
        無意味にし、整形だけずれた版は `deno fmt --check` を落とす。
        """
        stored = rope_fixture.FIXTURE_PATH.read_text(encoding="utf-8")

        assert stored == rope_fixture.fixture_json()

    def test_the_declaration_is_the_e2b_one(self) -> None:
        stored = json.loads(rope_fixture.FIXTURE_PATH.read_text(encoding="utf-8"))

        assert stored["spec"] == {
            rope.FULL_ATTENTION: {"theta": 1000000.0, "headDim": 512, "rotaryDim": 128},
            rope.SLIDING_ATTENTION: {"theta": 10000.0, "headDim": 256, "rotaryDim": 256},
        }
        assert stored["positions"] == list(rope_fixture.FIXTURE_POSITIONS)

    @pytest.mark.skipif(
        not rope_fixture.E2B_CONFIG_PATH.exists(),
        reason=f"上流チェックポイントが無い（{rope_fixture.E2B_CONFIG_PATH}）",
    )
    def test_the_declaration_matches_the_upstream_checkpoint(self) -> None:
        """MUST（`rope_fixture` の module docstring）: 手写しの宣言は実物の `config.json` と
        同じ数を持つ。

        上の 2 本は「フィクスチャが宣言どおり」までしか見ない — 宣言そのものが実物とずれると、
        フィクスチャ・TS 実装・Python 実装の三者は自己整合したまま**配布形とだけ**食い違う
        （上流が E2B の theta を変えた版を出したときの壊れ方）。
        """
        text_config = json.loads(rope_fixture.E2B_CONFIG_PATH.read_text(encoding="utf-8"))[
            "text_config"
        ]

        for field in UPSTREAM_ROPE_FIELDS:
            assert rope_fixture.E2B_TEXT_CONFIG_FIELDS[field] == text_config[field], field

    def test_the_tables_decode_to_the_declared_shape(self) -> None:
        """base64 は f32 リトルエンディアンの `positions × headDim`（TS 側の読み方の前提）。"""
        stored = json.loads(rope_fixture.FIXTURE_PATH.read_text(encoding="utf-8"))
        rows = len(stored["positions"])

        for layer_type, tables in stored["tables"].items():
            head_dim = stored["spec"][layer_type]["headDim"]
            for part, encoded in tables.items():
                values = np.frombuffer(base64.b64decode(encoded), dtype="<f4")
                assert values.size == rows * head_dim, f"{layer_type} {part}"
