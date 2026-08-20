"""校正付き i4（GPTQ）の結線の約束事（実重み不要分）。

実重みの校正は手動（README 参照）。ここで固定するのは、壊れると**偽 PASS** になる側だけ:

- stage 分解が wrapper の forward とビット一致すること（ずれた経路で丸めても数値は普通に出る）
- i4 適格が「stage 内 = 校正」「stage 外 = 素の RTN」へ**過不足なく排他に**割れること
- scale 台帳のキーが wrapper の FQN 空間（= safetensors のテンソルキー）に居ること
- 校正が**実際に別の丸め**を産むこと（素通りしたら格納形が同じなので資産からは読めない）

模型は tiny な骨格で、**実物と同じ FQN**（`model.embeddings.word_embeddings` /
`model.encoder.rel_embeddings` / `model.encoder.layer.<i>.…`）と**同じ呼び出しの形**
（層は hidden と mask を位置引数で受け、先頭層の出力だけ ConvLayer が乗る）を写す。名前まで
写すのは、実物向けの定数（{@link deberta.export.NON_STAGE_I4_WEIGHTS} /
{@link deberta.export.NON_LOOKUP_EMBEDDINGS}）を差し替えずに門を試すため。
"""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch
from torch import nn

from deberta import calib
from deberta import export as export_deberta
from deberta.calib_texts import CALIB_TEXTS

#: 量子化軸（= group 長）。i4 は端数 group を作らない（ADR 0069 決定 2）。
HIDDEN = 32
VOCAB = 16
#: 相対位置表の行数（`rel_embeddings` の器 — 表引きされないので i4 適格外の席）。
REL_ROWS = 8


class TinyLayer(nn.Module):
    """`DebertaV2Layer` の**呼び出しの形**だけを写した層（hidden と mask を位置引数で受ける）。"""

    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(HIDDEN, HIDDEN)
        self.dense = nn.Linear(HIDDEN, HIDDEN)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        relative_pos: torch.Tensor,
        rel_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        attended = torch.tanh(self.query(hidden) + rel_embeddings[relative_pos]) * mask
        return (self.dense(attended) + hidden, None)


class TinyConv(nn.Module):
    """`ConvLayer` の席（先頭層だけ stage の入力 hidden を残差として混ぜ直す）。

    `nn.Linear` を 1 本も持たないのは実物と同じ（`Conv1d` + `LayerNorm`）— stage の走査集合が
    ConvLayer で動かないことを、模型の側でも保つ。
    """

    def __init__(self) -> None:
        super().__init__()
        self.blend = nn.Parameter(torch.full((HIDDEN,), 0.5))

    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor, input_mask: torch.Tensor
    ) -> torch.Tensor:
        return residual + hidden * self.blend * input_mask.unsqueeze(-1).to(hidden.dtype)


class TinyEncoder(nn.Module):
    """`DebertaV2Encoder.forward` のループを写した encoder。"""

    def __init__(self, layers: int) -> None:
        super().__init__()
        self.layer = nn.ModuleList(TinyLayer() for _ in range(layers))
        self.rel_embeddings = nn.Embedding(REL_ROWS, HIDDEN)
        self.conv = TinyConv()

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor, relative_pos: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        rel_embeddings = self.rel_embeddings.weight
        states = [hidden]
        next_kv = hidden
        for index, layer in enumerate(self.layer):
            output = layer(
                next_kv,
                attention_mask,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )[0]
            if index == 0:
                output = self.conv(hidden, output, attention_mask)
            states.append(output)
            next_kv = output
        return tuple(states)


class TinyEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(VOCAB, HIDDEN)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)


class TinyModel(nn.Module):
    def __init__(self, layers: int) -> None:
        super().__init__()
        self.embeddings = TinyEmbeddings()
        self.encoder = TinyEncoder(layers)


class TinyWrapper(nn.Module):
    """`HiddenStatesWrapper` の骨格（全層の hidden をタプルで返す）。"""

    def __init__(self, layers: int = 2) -> None:
        super().__init__()
        self.model = TinyModel(layers)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, relative_pos: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return self.model.encoder(self.model.embeddings(input_ids), attention_mask, relative_pos)


def make_wrapper(layers: int = 2, seed: int = 0) -> TinyWrapper:
    torch.manual_seed(seed)
    wrapper = TinyWrapper(layers)
    wrapper.eval()
    return wrapper


def make_args(count: int = 3, seed: int = 7) -> tuple[tuple[torch.Tensor, ...], ...]:
    """校正入力（wrapper の位置引数）を長さ違いで数件（48 文の実コーパスは回さない）。"""
    generator = torch.Generator().manual_seed(seed)
    args = []
    for index in range(count):
        length = 4 + index
        ids = torch.randint(0, VOCAB, (1, length), generator=generator)
        args.append(
            (
                ids,
                torch.ones_like(ids),
                torch.randint(0, REL_ROWS, (length,), generator=generator),
            )
        )
    return tuple(args)


def i4_weight_keys(wrapper: nn.Module) -> set[str]:
    return {f"{name}.weight" for name in export_deberta._i4_module_names(wrapper)}


class _StubTokenizer:
    """文字数ぶんの id を返すだけのトークナイザ（`deberta.export.encode_text` の入口の形）。"""

    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": list(range(1, len(text) + 1))}


class _StubModel(nn.Module):
    """`build_graph_inputs` がバケット幅を読む席だけを持つ模型。"""

    def __init__(self) -> None:
        super().__init__()
        attention = nn.Module()
        attention.position_buckets = 4
        attention.max_relative_positions = 8
        layer = nn.Module()
        layer.attention = nn.Module()
        layer.attention.self = attention
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([layer])


class TestStageSplit:
    def test_the_stage_chain_reproduces_the_wrapper_output_bit_exactly(self):
        """MUST: 分解が本物の経路と 1bit も違わない（違えば別経路の GPTQ を出荷する）。"""
        wrapper = make_wrapper()
        args = make_args()
        stages = calib.encoder_stages(wrapper)
        batches = calib.capture_stage_batches(wrapper, args)

        calib.assert_stage_split(wrapper, args[0], batches[0], stages)

    def test_a_dropped_stage_is_caught(self):
        """段を 1 つ落とすと最終 hidden が変わる（門が素通りしないことの実測）。"""
        wrapper = make_wrapper()
        args = make_args()
        stages = calib.encoder_stages(wrapper)
        batches = calib.capture_stage_batches(wrapper, args)

        with pytest.raises(AssertionError, match="ビット一致しない"):
            calib.assert_stage_split(wrapper, args[0], batches[0], stages[:-1])

    def test_a_stage_chain_without_the_conv_residual_is_caught(self):
        """先頭層に乗る ConvLayer を落とすと 2 段目以降が別の hidden を見る。"""
        wrapper = make_wrapper()
        args = make_args()
        stages = calib.encoder_stages(wrapper)
        batches = calib.capture_stage_batches(wrapper, args)
        without_conv = tuple(
            (prefix, calib.EncoderStage(index, getattr(stage, stage.child), None))
            for index, (prefix, stage) in enumerate(stages)
        )

        with pytest.raises(AssertionError, match="ビット一致しない"):
            calib.assert_stage_split(wrapper, args[0], batches[0], without_conv)

    def test_the_stage_prefix_lands_in_the_wrapper_fqn_space(self):
        """接頭辞つきの局所 FQN が wrapper の実 FQN と一致する（台帳キーの空間が決まる場所）。"""
        wrapper = make_wrapper()
        stages = calib.encoder_stages(wrapper)

        names = calib.stage_linear_names(stages)

        assert names == {
            f"model.encoder.layer.{index}.{child}"
            for index in range(2)
            for child in ("query", "dense")
        }
        assert names <= {name for name, _module in wrapper.named_modules()}

    def test_a_forward_that_never_reaches_the_conv_fails_loudly(self):
        """ConvLayer を持つのにループが呼ばない構成 = stage の綴りと模型がずれた形。

        番兵が飛ばずに完走するので入力が揃わない。黙って進むと「校正入力ゼロ」の診断が
        core 側で出るだけになるので、捕捉の側で落とす。
        """

        class ConvlessEncoder(TinyEncoder):
            def forward(self, hidden, attention_mask, relative_pos):
                rel_embeddings = self.rel_embeddings.weight
                next_kv = hidden
                for layer in self.layer:
                    next_kv = layer(
                        next_kv,
                        attention_mask,
                        relative_pos=relative_pos,
                        rel_embeddings=rel_embeddings,
                    )[0]
                return (hidden, next_kv)

        wrapper = make_wrapper()
        wrapper.model.encoder = ConvlessEncoder(2)

        with pytest.raises(AssertionError, match="先頭 stage の入力が揃わなかった"):
            calib.capture_stage_batches(wrapper, make_args(count=1))

    def test_zero_calibration_inputs_fail_loudly(self):
        """MUST: 素の RTN へ黙って落ちる分岐は持たない。"""
        wrapper = make_wrapper()

        with pytest.raises(AssertionError, match="校正入力が 1 件も無い"):
            calib.build_rig(wrapper, calib.encoder_stages(wrapper), ())


class TestCalibratedI4:
    def test_the_ledger_keys_live_in_the_wrapper_fqn_space(self):
        """MUST: 台帳のキー = safetensors のテンソルキー（emit の突合はここで決まる）。"""
        wrapper = make_wrapper()

        scales, overrides = export_deberta._fake_quant("i4", wrapper, calib_args=make_args())

        parameters = dict(wrapper.named_parameters())
        assert set(overrides) <= set(parameters), "i4 席のキーが wrapper の FQN 空間に無い"
        assert set(scales) <= set(parameters)
        assert set(overrides) == i4_weight_keys(wrapper)

    def test_the_eligible_set_is_split_exclusively_between_the_two_paths(self):
        """stage 内（校正）と stage 外（RTN）で**過不足なく排他**（`quantize.py` の混成 MUST）。"""
        wrapper = make_wrapper()
        stages = calib.encoder_stages(wrapper)
        stage_names = calib.stage_linear_names(stages)
        i4_names = export_deberta._i4_module_names(wrapper)

        assert i4_names - stage_names == export_deberta.NON_STAGE_I4_WEIGHTS
        assert stage_names <= i4_names, "stage 内の linear が i4 適格から漏れている"
        # 表引きされない相対位置表は i4 側に入らない（i8 の担当のまま）
        assert export_deberta.NON_LOOKUP_EMBEDDINGS & i4_names == frozenset()

    def test_calibration_produces_a_different_rounding_from_plain_rtn(self):
        """校正が素通りしたら格納形が同じなので資産からは読めない — 値差で実測する。"""
        calibrated = make_wrapper()
        plain = make_wrapper()

        export_deberta._fake_quant("i4", calibrated, calib_args=make_args())
        export_deberta._fake_quant("i4", plain)

        after = dict(calibrated.named_parameters())
        differing = sorted(
            name
            for name, weight in plain.named_parameters()
            if not torch.equal(weight, after[name])
        )
        assert differing, "GPTQ が RTN と同じ値を出した（校正が効いていない）"
        assert all(name.startswith("model.encoder.layer.") for name in differing)

    def test_the_lookup_table_is_rounded_by_the_plain_path_in_both_modes(self):
        """語彙表は stage の外なのでどちらの経路でも素の RTN（丸め値はビット一致）。"""
        calibrated = make_wrapper()
        plain = make_wrapper()

        export_deberta._fake_quant("i4", calibrated, calib_args=make_args())
        export_deberta._fake_quant("i4", plain)

        assert torch.equal(
            calibrated.model.embeddings.word_embeddings.weight,
            plain.model.embeddings.word_embeddings.weight,
        )

    def test_an_unexpected_non_stage_eligible_fails_loudly(self, monkeypatch):
        """stage 外の適格が宣言と違う = 上流の構成が動いた合図（黙って分類を変えない）。"""
        wrapper = make_wrapper()
        monkeypatch.setattr(export_deberta, "NON_STAGE_I4_WEIGHTS", frozenset())

        with pytest.raises(AssertionError, match="encoder stage の外の i4 適格"):
            export_deberta._fake_quant("i4", wrapper, calib_args=make_args())

    def test_a_ledger_short_of_the_include_set_fails_loudly(self, monkeypatch):
        """丸めた集合 ≠ 格納集合は「i4 席に i8 の重みが混ざったまま緑」— 最後の門で落とす。"""
        wrapper = make_wrapper()
        genuine = calib.calibrate_i4

        def short(rig, include):
            report, ledger = genuine(rig, include)
            trimmed = dict(ledger.scales)
            trimmed.pop(sorted(trimmed)[0])
            return report, replace(ledger, scales=trimmed)

        monkeypatch.setattr(calib, "calibrate_i4", short)

        with pytest.raises(AssertionError, match="i4 適格"):
            export_deberta._fake_quant("i4", wrapper, calib_args=make_args())

    def test_a_ledger_that_overlaps_the_plain_path_fails_loudly(self, monkeypatch):
        """同じ重みを 2 経路で丸めると値だけが静かに狂う（格納形も本数も正しく見える）。"""
        wrapper = make_wrapper()
        genuine = calib.calibrate_i4
        table = f"{sorted(export_deberta.NON_STAGE_I4_WEIGHTS)[0]}.weight"

        def overlapping(rig, include):
            report, ledger = genuine(rig, include)
            merged = dict(ledger.scales) | {table: torch.ones(VOCAB, 1)}
            return report, replace(ledger, scales=merged)

        monkeypatch.setattr(calib, "calibrate_i4", overlapping)

        with pytest.raises(AssertionError, match="二重丸め"):
            export_deberta._fake_quant("i4", wrapper, calib_args=make_args())


class TestCorpusWiring:
    def test_the_i4_series_is_calibrated_by_default(self):
        """MUST: CLI の `--dtype i4` はフラグ無しで校正付き（opt-out はテスト用の引数だけ）。"""
        default = inspect.signature(export_deberta.export_variant).parameters["calib_texts"].default

        assert default is CALIB_TEXTS

    def test_the_calibration_inputs_go_through_the_golden_input_path(self, monkeypatch):
        """MUST: 校正入力も golden と同じ入力構築経路（別の綴りで組むと活性が黙って割れる）。"""
        lengths: list[int] = []

        def fake_tables(length, *, position_buckets, max_position):
            lengths.append(length)
            table = torch.zeros(length, length, dtype=torch.int64)
            return table, table

        monkeypatch.setattr(export_deberta.patch, "build_rel_pos_tables", fake_tables)

        args = export_deberta.build_calib_args(_StubTokenizer(), _StubModel(), ("あい", "うえお"))

        assert lengths == [2, 3], "添字表が実長で作られていない"
        assert [tuple(item.shape) for item in args[0]] == [(1, 2), (1, 2), (2, 2), (2, 2)]
        # 位置引数の並びは INPUT_ORDER そのもの（wrapper へ位置で渡すので入れ替わると沈黙する）
        assert torch.equal(args[1][0], torch.tensor([[1, 2, 3]]))
        assert torch.equal(args[1][1], torch.ones(1, 3, dtype=torch.int64))
        assert export_deberta.INPUT_ORDER == ("input_ids", "attention_mask", "c2p_pos", "p2c_pos")

    def test_an_empty_corpus_fails_loudly(self):
        with pytest.raises(ValueError, match="校正コーパスが空"):
            export_deberta.build_calib_args(_StubTokenizer(), _StubModel(), ())
