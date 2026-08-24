"""校正付き i4（GPTQ）の結線の約束事（実重み不要分）。

実重みの校正は手動（`python -m irodori.export --dtype i4`）。ここで固定するのは、壊れると
**偽 PASS** になる側だけ:

- stage 逐次の block ループが素の DiT 1 回の forward とビット一致すること（ずれた経路で
  丸めても数値は普通に出る）
- i4 適格が「配布グラフに載る linear だけ」で、g32 非整除が現れたら fail loudly すること
- 配布グラフの linear が「block 内の adaLN 以外 = i4 格納 × 校正」「block 外 + adaLN = i8 格納」
  へ**過不足なく排他に**割れること（聴感裁定 2026-08-23 で block 外と adaLN を i4 から外した）
- adaLN の判定が**セグメント一致**であること（`attention` が `attention_adaln` を巻き込まない・
  adaLN 配下の子を取りこぼさない）と、adaLN が 1 本も無い日**も片側だけ改名された日も**
  fail loudly すること
- scale 台帳と 1 本単位の格納指定のキーがラッパの FQN 空間（= safetensors のテンソルキー）に
  居ること
- 校正が**実際に別の丸め**を産むこと（素通りしたら格納形が同じなので資産からは読めない）
- 校正入力を**配布条件と同じ状態の重み**から採ること（i8 は全部丸めた後・i4 側は
  丸める前）
- 上流 `DiffusionBlock.forward` の引数が動いたら落ちること（cond 側の切り出しが黙って外れる）
- 条件 K/V キャッシュ付きで回したら落ちること（1 つの kwargs を全 stage で使い回せない）
- `--no-calib` が opt-out として本当に校正を回さないこと

模型は tiny な骨格で、**実物と同じ FQN**（`blocks.<i>` / `in_proj` / `out_proj` /
`cond_module.<i>`）と**同じ呼び出しの形**（block は 11 引数を全て keyword で受けて hidden を
返す・DiT は他役割のコピーを内側に持つ）を写す。名前まで写すのは、実物向けの判定
（{@link irodori.calib.dit_i4_names} が `DitGraph` との交差で採る形）を差し替えずに門を試すため。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from irodori import calib
from irodori import export as ir
from irodori import patch as ir_patch
from irodori.calib_cases import CALIB_CASES
from karume.quantize import channel_scale, quantize_to_int8

#: 量子化軸（= group 長）。i4 は端数 group を作らない（ADR 0069 決定 2）。
HIDDEN = 32
#: 条件 state の幅（block の交差 attention 側 — hidden と同じにして写しを減らす）。
CONTEXT = 32
#: g32 非整除の幅（適格判定の fail loudly を踏むためだけの数）。
UNALIGNED = 20
#: latent のトークン数。
TOKENS = 4
#: 捕捉したバッチの batch 行数（上流 independent CFG が cond + uncond を束ねた形の縮図）。
CFG_ROWS = 3


class TinyNorm(nn.Module):
    """`RMSNorm` の席（重みを持つが量子化対象型ではない — i8 / i4 のどちらにも入らない）。"""

    def __init__(self, width: int = HIDDEN) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class TinyAttention(nn.Module):
    def __init__(self, width: int = HIDDEN) -> None:
        super().__init__()
        self.wq = nn.Linear(width, HIDDEN, bias=False)
        self.wo = nn.Linear(HIDDEN, HIDDEN, bias=False)


class TinyBlock(nn.Module):
    """上流 `DiffusionBlock` の**呼び出しの形**だけを写した block。

    引数 11 本を全て keyword で受けて hidden を返す — 実物と同じなので、stage は包まずに
    そのまま並べられる（`irodori.calib.dit_stages`）。

    adaLN は実物と同じ属性名（`attention_adaln` / `mlp_adaln`）で、**linear を子に持つ**形
    （`…_adaln.1`）にする — 判定がセグメント一致であること（葉でも接尾辞でもない）を
    模型の側からも踏むため。
    """

    def __init__(self, attention_width: int = HIDDEN) -> None:
        super().__init__()
        self.attention = TinyAttention(attention_width)
        self.mlp = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.attention_adaln = nn.Sequential(nn.SiLU(), nn.Linear(HIDDEN, HIDDEN, bias=False))
        self.mlp_adaln = nn.Sequential(nn.SiLU(), nn.Linear(HIDDEN, HIDDEN, bias=False))

    def forward(
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor,
        speaker_mask: torch.Tensor,
        caption_state: torch.Tensor,
        caption_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        context_kv: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        context = (
            text_state.mean(dim=1, keepdim=True)
            + speaker_state.mean(dim=1, keepdim=True)
            + caption_state.mean(dim=1, keepdim=True)
        )
        attended = torch.tanh(self.attention.wq(x) + context)
        return x + self.attention.wo(self.mlp(attended)) * self.modulation(cond_embed)

    def modulation(self, cond_embed: torch.Tensor) -> torch.Tensor:
        return self.attention_adaln(cond_embed) + self.mlp_adaln(cond_embed)


class TinyDit(nn.Module):
    """`TextToLatentRFDiT` の骨格（他役割のコピーを内側に持つところまで写す）。"""

    def __init__(self, blocks: int = 3, block_type: type[TinyBlock] = TinyBlock) -> None:
        super().__init__()
        self.cond_module = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN, bias=False),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN, bias=False),
        )
        self.in_proj = nn.Linear(HIDDEN, HIDDEN)
        self.blocks = nn.ModuleList(block_type() for _ in range(blocks))
        self.out_norm = TinyNorm()
        self.out_proj = nn.Linear(HIDDEN, HIDDEN)
        self.text_norm = TinyNorm(CONTEXT)
        self.caption_norm = TinyNorm(CONTEXT)
        self.head_dim = 8
        # DiT が内側に持つ他役割のコピー（`DitGraph` には載らない = i4 適格ではない）。
        self.pretrained_text_backbone = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward_with_encoded_conditions(self, x_t: torch.Tensor, **conditions: Any) -> torch.Tensor:
        """上流と同じ「block を 11 引数すべて keyword で呼ぶ」形。"""
        x = self.in_proj(x_t)
        for block in self.blocks:
            x = block(x=x, **conditions)
        return self.out_proj(self.out_norm(x))


class UnalignedDit(TinyDit):
    """block 内の linear が 1 本だけ g32 非整除になった日の DiT。"""

    def __init__(self, blocks: int = 3) -> None:
        super().__init__(blocks)
        self.blocks[0].attention.wq = nn.Linear(UNALIGNED, HIDDEN, bias=False)


class DriftingBlock(TinyBlock):
    """上流が block の付随引数を 1 本落とした日の block。"""

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor,
        speaker_mask: torch.Tensor,
        caption_state: torch.Tensor,
        caption_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        self_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().forward(
            x,
            cond_embed,
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
            freqs_cis,
            self_mask,
        )


class DriftingDit(TinyDit):
    """付随引数が 1 本減った上流（数値は普通に出るので、門が無ければ気づけない）。"""

    def __init__(self, blocks: int = 3) -> None:
        super().__init__(blocks, block_type=DriftingBlock)

    def forward_with_encoded_conditions(self, x_t: torch.Tensor, **conditions: Any) -> torch.Tensor:
        conditions.pop("context_kv", None)
        return super().forward_with_encoded_conditions(x_t, **conditions)


class RenamedAdalnBlock(TinyBlock):
    """上流が modulation の属性名を変えた日の block（adaLN の綴りがどこにも無い）。

    計算は 1 演算も変わらない — 変わるのは FQN だけなので、門が無ければ 144 本が黙って
    i4 へ戻る（格納形も本数の門も正しいまま、裁定で棄却した構成が出荷される）。
    """

    def __init__(self, attention_width: int = HIDDEN) -> None:
        super().__init__(attention_width)
        self.attention_modulation = self.attention_adaln
        self.mlp_modulation = self.mlp_adaln
        del self.attention_adaln
        del self.mlp_adaln

    def modulation(self, cond_embed: torch.Tensor) -> torch.Tensor:
        return self.attention_modulation(cond_embed) + self.mlp_modulation(cond_embed)


class RenamedAdalnDit(TinyDit):
    def __init__(self, blocks: int = 3) -> None:
        super().__init__(blocks, block_type=RenamedAdalnBlock)


class HalfRenamedAdalnBlock(TinyBlock):
    """上流が modulation の**片側だけ**を改名した日の block（`mlp_adaln` だけが消える）。

    集合は空にならないので「adaLN が 1 本も無い」だけを見る門は素通りし、i4 の過不足門も
    同じ分類器から両辺を作るので自己整合したまま緑になる — 落ちる網が 1 枚も無いまま、
    i8 へ戻したはずの半分が i4 へ戻る。
    """

    def __init__(self, attention_width: int = HIDDEN) -> None:
        super().__init__(attention_width)
        self.mlp_modulation = self.mlp_adaln
        del self.mlp_adaln

    def modulation(self, cond_embed: torch.Tensor) -> torch.Tensor:
        return self.attention_adaln(cond_embed) + self.mlp_modulation(cond_embed)


class HalfRenamedAdalnDit(TinyDit):
    def __init__(self, blocks: int = 3) -> None:
        super().__init__(blocks, block_type=HalfRenamedAdalnBlock)


class CachedKvDit(TinyDit):
    """上流の既定（`use_context_kv_cache=True`）で回してしまった日の DiT。

    block ごとに別の `context_kv` が渡るので、1 つの kwargs を全 stage で使い回す core の
    駆動とは噛み合わない（しかも数値は普通に出る）。
    """

    def forward_with_encoded_conditions(self, x_t: torch.Tensor, **conditions: Any) -> torch.Tensor:
        x = self.in_proj(x_t)
        for index, block in enumerate(self.blocks):
            x = block(x=x, **{**conditions, "context_kv": (torch.zeros(index + 1),)})
        return self.out_proj(self.out_norm(x))


@pytest.fixture(autouse=True)
def stub_rope_table(monkeypatch):
    """`DitGraph` の RoPE 表は上流実装から作る — 模型では形だけあれば足りる。

    これで {@link irodori.calib.dit_i4_names} の「配布グラフに載る重みか」の判定を、実装
    clone（`irodori_tts`）の有無に依らず踏める。
    """
    monkeypatch.setattr(
        ir_patch,
        "real_pair_rope_table",
        lambda head_dim, length: torch.zeros(length, 2, head_dim),
    )


def make_dit(blocks: int = 3, seed: int = 0, dit_type: type[TinyDit] = TinyDit) -> TinyDit:
    torch.manual_seed(seed)
    dit = dit_type(blocks)
    dit.eval()
    return dit


def make_conditions(seed: int = 7, rows: int = CFG_ROWS) -> dict[str, Any]:
    """block へ渡る付随引数一式（`x` 以外の 10 本）。"""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    return {
        "cond_embed": randn(rows, 1, HIDDEN),
        "text_state": randn(rows, 5, CONTEXT),
        "text_mask": torch.ones(rows, 5, dtype=torch.bool),
        "speaker_state": randn(rows, 3, CONTEXT),
        "speaker_mask": torch.ones(rows, 3, dtype=torch.bool),
        "caption_state": randn(rows, 4, CONTEXT),
        "caption_mask": torch.ones(rows, 4, dtype=torch.bool),
        "freqs_cis": randn(TOKENS, 4),
        "self_mask": None,
        "context_kv": None,
    }


def capture_run(dit: nn.Module, seed: int = 7, rows: int = CFG_ROWS) -> calib.CapturedRun:
    """素の DiT を 1 回回して {@link irodori.calib.CapturedRun} を組む（実物の縮図）。

    実物では `irodori.calib.capture_stage_batches` が上流 `sample_euler_rf_cfg` を回して同じ
    ものを採る。切り出しは本物の {@link irodori.calib._split_block_kwargs} をそのまま通すので、
    cond 側 1 行の取り出しと引数の顔ぶれの門はここでも実効。
    """
    generator = torch.Generator().manual_seed(seed + 1)
    x_t = torch.randn(rows, TOKENS, HIDDEN, generator=generator)
    conditions = make_conditions(seed, rows)
    batches: list[Any] = []
    graph_batch: list[Any] = []
    tail: list[torch.Tensor] = []

    def on_block(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if graph_batch:
            return
        graph_batch.append(
            (
                (kwargs["x"].detach(),),
                {name: value for name, value in kwargs.items() if name != "x"},
            )
        )
        batches.append(calib._split_block_kwargs(kwargs, 1))

    def on_tail(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if not tail:
            tail.append(args[0].detach())

    handles = [
        dit.blocks[0].register_forward_pre_hook(on_block, with_kwargs=True),
        dit.out_norm.register_forward_pre_hook(on_tail),
    ]
    try:
        with torch.no_grad():
            dit.forward_with_encoded_conditions(x_t, **conditions)
    finally:
        for handle in handles:
            handle.remove()
    return calib.CapturedRun(tuple(batches), graph_batch[0], tail[0])


def make_plan(*, calibrated: bool = True, steps: int = 2) -> ir.CalibPlan:
    """校正の足場（スタブ経路では `context` の中身は使われない）。"""
    return ir.CalibPlan(
        context=calib.CalibContext(
            model_dir=Path("unused"),
            source=None,
            config=SimpleNamespace(latent_patch_size=1),
            text_config={},
            model_config={},
        ),
        cases=CALIB_CASES,
        steps=steps,
        calibrated=calibrated,
    )


@pytest.fixture
def stub_capture(monkeypatch):
    """`capture_stage_batches` を差し替える（実物は上流の参照 denoise を回す）。

    差し替え先は**素の DiT を実際に走らせて**採るので、`(args, kwargs)` の形も捕捉元も実物と
    同じまま。返すのは `(呼び出しごとの step 数, 捕捉時点の重み)` で、後者は「どの重みが
    丸まった状態で校正入力を採ったか」を呼び出し側が主張するための写し。

    MUST: 走らせるのは**呼ばれた時点**（実物と同じ）— 先に採っておくと、i8 で丸め済みの
    block 内 adaLN が捕捉時の重みと食い違い、分解一致門が模型の都合で落ちる。
    """

    def install(dit: nn.Module) -> tuple[list[int], dict[str, torch.Tensor]]:
        calls: list[int] = []
        at_capture: dict[str, torch.Tensor] = {}

        def stub(model, cases, context, *, steps=calib.CALIB_STEPS, label="calib"):
            calls.append(steps)
            at_capture.clear()
            at_capture.update({n: p.detach().clone() for n, p in model.named_parameters()})
            prepared = capture_run(model)
            return tuple(prepared for _ in cases)

        monkeypatch.setattr(calib, "capture_stage_batches", stub)
        return calls, at_capture

    return install


def quantize(dit: nn.Module, plan: ir.CalibPlan | None = None) -> ir.FakeQuantResult:
    return ir.fake_quant("i4", {ir.TARGET_DIT: dit}, calib_plan=plan or make_plan())


class TestStageSplit:
    def test_the_stage_chain_reproduces_the_block_loop_bit_exactly(self):
        """MUST: 分解が本物の経路と 1bit も違わない（違えば別経路の GPTQ を出荷する）。"""
        dit = make_dit()

        calib.assert_stage_split(calib.dit_stages(dit), capture_run(dit))

    def test_a_dropped_stage_is_caught(self):
        """段を 1 つ落とすと最終 hidden が変わる（門が素通りしないことの実測）。"""
        dit = make_dit()
        run = capture_run(dit)

        with pytest.raises(AssertionError, match="ビット一致しない"):
            calib.assert_stage_split(calib.dit_stages(dit)[:-1], run)

    def test_a_reordered_stage_chain_is_caught(self):
        """順序が入れ替わっても形は合う（値だけが静かに変わる壊れ方）。"""
        dit = make_dit()
        stages = calib.dit_stages(dit)
        run = capture_run(dit)

        with pytest.raises(AssertionError, match="ビット一致しない"):
            calib.assert_stage_split((stages[1], stages[0], *stages[2:]), run)

    def test_the_stage_prefix_lands_in_the_dit_fqn_space(self):
        """接頭辞つきの局所 FQN が DiT の実 FQN と一致する（台帳キーの空間が決まる場所）。"""
        dit = make_dit(blocks=2)

        names = calib.stage_linear_names(calib.dit_stages(dit))

        assert names == {
            f"blocks.{index}.{child}"
            for index in range(2)
            for child in (
                "attention.wq",
                "attention.wo",
                "mlp",
                "attention_adaln.1",
                "mlp_adaln.1",
            )
        }
        assert names <= {name for name, _module in dit.named_modules()}

    def test_the_export_path_runs_the_gate(self, stub_capture, monkeypatch):
        """MUST: 門が `_round_i4_calibrated` に本当に嵌まっていること（関数単体では足りない）。"""
        dit = make_dit()
        stub_capture(dit)
        genuine = calib.assert_stage_split

        def drifted(stages, run):
            return genuine(stages[:-1], run)

        monkeypatch.setattr(calib, "assert_stage_split", drifted)

        with pytest.raises(AssertionError, match="ビット一致しない"):
            quantize(dit)


class TestCapturedKwargs:
    def test_it_keeps_only_the_cond_row_of_the_batched_arguments(self):
        """上流 independent CFG は cond + uncond を 1 forward へ束ねる — 行 0 だけが cond 側。"""
        dit = make_dit()

        run = capture_run(dit)

        args, kwargs = run.batches[0]
        assert args[0].shape[0] == 1
        assert run.graph_batch[0][0].shape[0] == CFG_ROWS
        for name in calib.BATCHED_BLOCK_ARGS - {"x"}:
            assert kwargs[name].shape[0] == 1, name
        # batch 軸を持たない引数はそのまま（切ると RoPE 表の長さが 1 になる）。
        assert kwargs["freqs_cis"].shape[0] == TOKENS

    def test_a_dropped_block_argument_is_caught(self):
        """上流が引数を減らすと、cond 側の切り出しが黙って外れる（数値は普通に出る）。"""
        dit = make_dit(dit_type=DriftingDit)

        with pytest.raises(AssertionError, match="block の呼び出し引数"):
            capture_run(dit)

    def test_a_context_kv_cache_is_caught(self):
        """block ごとに別のキャッシュを 1 つの kwargs で使い回すことになる形。"""
        dit = make_dit(dit_type=CachedKvDit)

        with pytest.raises(AssertionError, match="条件 K/V キャッシュ"):
            capture_run(dit)


class TestEligibleSet:
    def test_only_the_linears_the_shipped_graph_holds_are_eligible(self):
        """DiT が内側に持つ他役割のコピーは i4 に載らない（`DitGraph` との交差で決まる）。"""
        dit = make_dit(blocks=2)

        names = calib.dit_i4_names(dit, ir.dit_sym_max(SimpleNamespace(latent_patch_size=1)))

        assert names == {
            "in_proj",
            "out_proj",
            "cond_module.0",
            "cond_module.2",
            *calib.stage_linear_names(calib.dit_stages(dit)),
        }
        assert "pretrained_text_backbone" not in names

    def test_a_weight_off_the_group_grid_fails_loudly(self):
        """MUST: g32 非整除は**除外せず**落とす（1 本だけ外すと過不足一致門が張れない）。"""
        dit = make_dit(dit_type=UnalignedDit)

        with pytest.raises(AssertionError, match="割り切れない"):
            calib.dit_i4_names(dit, ir.dit_sym_max(SimpleNamespace(latent_patch_size=1)))

    def test_the_eligible_set_is_split_exclusively_between_the_two_paths(self):
        """適格集合が block 内と block 外へ**過不足なく排他**に割れる（格納の割り方の土台）。

        `quantize.py` の混成 MUST（同じ重みを 2 経路に通さない）。適格集合そのものは
        adaLN の裁定に依らない — 動くのは下の格納指定の割り方だけ。
        """
        dit = make_dit()
        stage_names = calib.stage_linear_names(calib.dit_stages(dit))
        eligible = calib.dit_i4_names(dit, ir.dit_sym_max(SimpleNamespace(latent_patch_size=1)))

        assert stage_names <= eligible, "block 内の linear が i4 適格から漏れている"
        assert eligible - stage_names == {"in_proj", "out_proj", "cond_module.0", "cond_module.2"}

    def test_a_block_linear_outside_the_i4_set_fails_loudly(self, monkeypatch, stub_capture):
        """block 内に非適格が混じると「走査の本数 = 丸めた本数」の門が張れない。"""
        dit = make_dit()
        stub_capture(dit)
        genuine = calib.dit_i4_names
        monkeypatch.setattr(
            calib,
            "dit_i4_names",
            lambda model, sym_max: frozenset(genuine(model, sym_max) - {"blocks.0.mlp"}),
        )

        with pytest.raises(SystemExit, match="i4 適格でない"):
            quantize(dit)

    def test_a_report_short_of_the_scan_fails_loudly(self, monkeypatch, stub_capture):
        """丸め漏れは品質を**良い側**に見せる（素通りを数字から読めない）。"""
        from dataclasses import replace

        dit = make_dit()
        stub_capture(dit)
        genuine = calib.calibrate_i4

        def short(stages, batches, include=None):
            report, ledger = genuine(stages, batches, include)
            return replace(report, layers=report.layers[:-1]), ledger

        monkeypatch.setattr(calib, "calibrate_i4", short)

        with pytest.raises(AssertionError, match="走査の"):
            quantize(dit)

    def test_a_ledger_that_overlaps_the_i8_path_fails_loudly(self, monkeypatch, stub_capture):
        """同じ重みを 2 経路で丸めると値だけが静かに狂う（格納形も本数も正しく見える）。

        注入するのは block 外の `in_proj`（= i8 で丸め済み）— 二重丸めの門は過不足門より先に
        見るので、診断は本数のずれではなく「同じ重みを 2 度丸めた」で出る。
        """
        from dataclasses import replace

        dit = make_dit()
        stub_capture(dit)
        genuine = calib.calibrate_i4

        def overlapping(stages, batches, include=None):
            report, ledger = genuine(stages, batches, include)
            merged = dict(ledger.scales) | {"in_proj.weight": torch.ones(HIDDEN, 1)}
            return report, replace(ledger, scales=merged)

        monkeypatch.setattr(calib, "calibrate_i4", overlapping)

        with pytest.raises(SystemExit, match="二重丸め"):
            quantize(dit)


class TestAdalnBoundary:
    """i4 と i8 を割る adaLN の境（聴感裁定 2026-08-23 — 144 本を i8 へ戻した）。"""

    def test_the_match_is_by_segment_not_by_substring(self):
        """`attention` が `attention_adaln` を巻き込まず、adaLN 配下の子も取りこぼさない。"""
        assert calib.is_adaln("blocks.0.attention_adaln.1")
        assert calib.is_adaln("blocks.0.mlp_adaln.1")
        # stage 内の局所 FQN（校正の `include` が受け取る形）でも同じ答え。
        assert calib.is_adaln("attention_adaln.1")
        assert calib.is_adaln("mlp_adaln")
        assert not calib.is_adaln("blocks.0.attention.wq")
        assert not calib.is_adaln("blocks.0.attention.wo")
        assert not calib.is_adaln("blocks.0.mlp")
        assert not calib.is_adaln("in_proj")

    def test_a_dit_without_adaln_fails_loudly(self, stub_capture):
        """MUST: 上流が modulation の属性名を変えたら落とす。

        判定が空振りすると 144 本が黙って i4 へ戻る — 格納形も本数の門も正しいままなので、
        裁定で棄却した構成が緑のまま出荷される（読み上げ方の劣化だけが残る）。
        """
        dit = make_dit(dit_type=RenamedAdalnDit)
        stub_capture(dit)

        with pytest.raises(SystemExit, match="adaLN の linear が 1 本も無い"):
            quantize(dit)

    def test_the_seen_segments_are_reported_one_by_one(self):
        """MUST: 「どれか当たったか」ではなく**どれが当たったか**を返す（片側改名の検出源）。"""
        both = ["blocks.0.attention_adaln.1", "blocks.0.mlp_adaln.1", "blocks.0.attention.wq"]

        assert calib.adaln_segments_seen(both) == calib.ADALN_SEGMENTS
        assert calib.adaln_segments_seen(both[:1]) == {"attention_adaln"}
        assert calib.adaln_segments_seen(["blocks.0.mlp_adaln"]) == {"mlp_adaln"}
        assert calib.adaln_segments_seen(["blocks.0.attention.wq"]) == frozenset()

    def test_a_dit_that_renamed_only_one_side_fails_loudly(self, stub_capture):
        """MUST: 片側だけの改名も落とす（空集合にならないので「1 本も無い」門は素通りする）。

        `mlp_adaln` だけが消えると、i8 へ戻したはずの 144 本のうち半分が i4 へ戻る。格納形も
        本数の過不足門も緑のまま（両辺が同じ分類器から作られる）なので、落ちる網はここしかない。
        """
        dit = make_dit(dit_type=HalfRenamedAdalnDit)
        stub_capture(dit)

        with pytest.raises(SystemExit, match=r"1 本も無い綴り: \['mlp_adaln'\]"):
            quantize(dit)


class TestCalibratedI4:
    def test_the_ledger_keys_live_in_the_shipped_graph_fqn_space(self, stub_capture):
        """MUST: 台帳のキー = safetensors のテンソルキー（emit の突合はここで決まる）。"""
        dit = make_dit()
        stub_capture(dit)

        result = quantize(dit)

        graph = ir.DitGraph(dit, ir.dit_sym_max(SimpleNamespace(latent_patch_size=1)))
        owned = {name for name, _p in graph.named_parameters()}
        overrides = result.overrides[ir.TARGET_DIT]
        assert set(overrides) <= owned, "i4 席のキーが配布グラフの FQN 空間に無い"
        assert set(overrides) == {
            f"{name}.weight"
            for name in calib.dit_i4_names(
                dit, ir.dit_sym_max(SimpleNamespace(latent_patch_size=1))
            )
        }

    def test_the_overrides_are_a_hybrid_of_i4_and_i8_by_block_and_adaln(self, stub_capture):
        """MUST: i8 側（block 外 + adaLN）は**明示 `i8`** で名指しする — 落とすと emit の既定
        （i4）へ流れる。

        既定 `weight_dtype` が i4 なので、`karume.emit._plan_weight_dtype` は override の無い
        i4 適格を既定で i4 計画へ回す。そこへ per-channel scale しか無い i8 側が来ると
        「group scale が無い」で落ちる（= 明示指定が要る側の門）。
        """
        dit = make_dit()
        stub_capture(dit)

        result = quantize(dit)

        overrides = result.overrides[ir.TARGET_DIT]
        stage_names = calib.stage_linear_names(calib.dit_stages(dit))
        adaln_names = {name for name in stage_names if calib.is_adaln(name)}
        assert adaln_names, "模型に adaLN が無い（この門が空振りしている）"
        assert {key for key, dtype in overrides.items() if dtype == "i4"} == {
            f"{name}.weight" for name in stage_names - adaln_names
        }
        assert {key for key, dtype in overrides.items() if dtype == "i8"} == {
            "in_proj.weight",
            "out_proj.weight",
            "cond_module.0.weight",
            "cond_module.2.weight",
            *(f"{name}.weight" for name in adaln_names),
        }
        # i8 指定の block 外にも scale が要る（`karume.emit._plan_i8` は scale 不在を落とす）。
        scales = result.scales[ir.TARGET_DIT]
        assert set(overrides) <= set(scales), "格納指定のある重みに scale が無い（emit が落ちる）"

    def test_the_rebased_tables_reach_emit_with_the_same_keys(self, stub_capture):
        """MUST: emit へ渡る scale と格納指定が**同じキー空間**に居る（張り替えは 1 実装）。"""
        dit = make_dit()
        stub_capture(dit)

        result = quantize(dit)

        graph = ir.DitGraph(dit, ir.dit_sym_max(SimpleNamespace(latent_patch_size=1)))
        overrides = ir.target_weight_dtypes(ir.TARGET_DIT, graph, result.overrides)
        scales = ir.target_scales(ir.TARGET_DIT, graph, result.scales)
        assert set(overrides) == set(result.overrides[ir.TARGET_DIT])
        assert set(overrides) <= set(scales), "i4 指定の重みに scale が無い（emit が落ちる）"

    def test_calibration_produces_a_different_rounding_from_plain_rtn(self, stub_capture):
        """校正が素通りしたら格納形が同じなので資産からは読めない — 値差で実測する。"""
        calibrated = make_dit()
        plain = make_dit()
        stub_capture(calibrated)

        quantize(calibrated)
        quantize(plain, make_plan(calibrated=False))

        after = dict(calibrated.named_parameters())
        differing = sorted(
            name
            for name, weight in plain.named_parameters()
            if not torch.equal(weight, after[name])
        )
        assert differing, "GPTQ が RTN と同じ値を出した（校正が効いていない）"
        assert all(name.startswith("blocks.") for name in differing)
        # MUST: adaLN は GPTQ の `include` で外れている — 差が出たら i4 側で丸まっている。
        assert not any(calib.is_adaln(name) for name in differing)

    def test_the_i8_side_weights_land_on_the_i8_grid(self, stub_capture):
        """MUST: i8 側は **i8 の格子**（`--no-calib` でも同じ — 校正は i4 側にしか効かない）。

        聴感裁定 2026-08-23 で i4 から外した block 外と adaLN。i4 の格子に乗ったままだと、
        格納指定を i8 に替えても値だけが 4bit のこもりを保ったまま出荷される（形も本数も
        正しく見える）。
        """
        calibrated = make_dit()
        plain = make_dit()
        stub_capture(calibrated)

        quantize(calibrated)
        quantize(plain, make_plan(calibrated=False))

        after = dict(calibrated.named_parameters())
        adaln = sorted(
            name
            for name in calib.stage_linear_names(calib.dit_stages(calibrated))
            if calib.is_adaln(name)
        )
        for name in ("in_proj", "out_proj", "cond_module.0", "cond_module.2", *adaln):
            key = f"{name}.weight"
            assert torch.equal(dict(plain.named_parameters())[key], after[key]), key
            scale = channel_scale(after[key], 0)
            restored = quantize_to_int8(after[key], scale).to(torch.float32) * scale
            assert torch.equal(restored, after[key]), f"{key} が i8 の格子に乗っていない"

    def test_the_capture_sees_the_shipped_condition(self, stub_capture):
        """MUST: i8 側（block 外・adaLN・内側の他役割コピー）は丸めた**後**・i4 側は丸める
        **前**に校正入力を採る。

        後半（i4 側が素のまま）が「丸めた重みが作った活性から同じ重みの丸め先を選ぶ
        循環」を、前半（i8 側が丸まっている）が「f32 の周辺で選んだ丸め先を i8 の周辺と
        組んで配る」を、それぞれ捕まえる。adaLN は block **内**の i8 なので、ここが
        後回しになると modulation だけ f32 の活性で丸め先を選ぶことになる。
        """
        dit = make_dit()
        before = {name: p.detach().clone() for name, p in dit.named_parameters()}
        _calls, at_capture = stub_capture(dit)

        quantize(dit)

        for name in ("in_proj", "out_proj", "cond_module.0", "cond_module.2"):
            key = f"{name}.weight"
            assert not torch.equal(before[key], at_capture[key]), f"{key} が捕捉時に素のまま"
        # i8 側（DiT が内側に持つコピー）も捕捉より前（block 0 の入力を作る側と同じ理由）。
        i8_key = "pretrained_text_backbone.weight"
        assert not torch.equal(before[i8_key], at_capture[i8_key]), f"{i8_key} が捕捉時に素のまま"
        for key in (name for name in before if name.startswith("blocks.")):
            if calib.is_adaln(key):
                assert not torch.equal(before[key], at_capture[key]), f"{key} が捕捉時に素のまま"
            else:
                assert torch.equal(before[key], at_capture[key]), f"{key} が捕捉前に丸まっている"

    def test_the_probe_run_precedes_every_rounding(self, stub_capture):
        """順序 MUST ①: 分解一致門の probe は **1 step**・丸めの前（2 回目が本番の捕捉）。"""
        dit = make_dit()
        calls, _at_capture = stub_capture(dit)

        quantize(dit, make_plan(steps=5))

        assert calls == [1, 5]


class TestOptOut:
    def test_no_calib_never_touches_the_calibration_path(self, stub_capture):
        """opt-out は「校正が痩せる」ではなく**回らない**こと（費用も品質も別物になる）。"""
        dit = make_dit()
        calls, _at_capture = stub_capture(dit)

        quantize(dit, make_plan(calibrated=False))

        assert calls == []

    def test_i4_without_a_plan_fails_loudly(self):
        """MUST: 素の RTN へ黙って落ちる分岐は持たない（校正の足場が無ければ落とす）。"""
        dit = make_dit()

        with pytest.raises(SystemExit, match="校正の足場"):
            ir.fake_quant("i4", {ir.TARGET_DIT: dit})

    def test_the_i4_series_is_calibrated_by_default(self, monkeypatch):
        """MUST: CLI の `--dtype i4` はフラグ無しで校正付き（`--no-calib` だけが opt-out）。"""
        seen: dict[str, Any] = {}
        monkeypatch.setattr(ir, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})

        ir.main(["--dtype", "i4"])

        assert seen["no_calib"] is False
        assert seen["calib_steps"] is None
        assert seen["targets"] == (ir.TARGET_DIT,)

    def test_the_calibration_knobs_are_refused_outside_i4(self, monkeypatch):
        """効かないノブを黙って受けない（i8 に校正の経路は 1 本も無い）。"""
        with pytest.raises(SystemExit):
            ir.main(["--dtype", "i8", "--no-calib"])
        with pytest.raises(SystemExit):
            ir.main(["--dtype", "i8", "--calib-steps", "4"])

    def test_a_target_the_i4_series_does_not_write_is_refused(self):
        """MUST: 配布表が引かない系列を黙って書かない（`--target` の明示も拒否する）。"""
        with pytest.raises(SystemExit):
            ir.main(["--dtype", "i4", "--target", ir.TARGET_BACKBONE])


class TestCalibProvenance:
    def test_it_records_the_shippable_method_by_default(self, tmp_path):
        name = ir._write_calib_provenance("i4", make_plan(steps=7), ir.TARGET_DIT, tmp_path)

        import json

        record = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        assert record["method"] == calib.CALIB_METHOD
        assert record["grid"] == calib.CALIB_GRID.kind
        assert record["group_size"] == calib.CALIB_GRID.group_size
        assert record["cases"] == len(CALIB_CASES)
        assert record["steps"] == 7

    def test_no_calib_is_written_as_rtn_rather_than_omitted(self, tmp_path):
        """不在は「古い export」とも読める — 組み立て側が名指しで拒否できる形にする。"""
        import json

        name = ir._write_calib_provenance(
            "i4", make_plan(calibrated=False), ir.TARGET_DIT, tmp_path
        )

        record = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        assert record["method"] == "rtn"
        assert record["cases"] == 0
        assert record["steps"] == 0

    def test_a_stale_record_is_removed_when_the_series_is_retaken(self, tmp_path):
        """MUST: 全域関数（i4 以外で採り直すと記録だけが前回のまま生き残る）。"""
        ir._write_calib_provenance("i4", make_plan(), ir.TARGET_DIT, tmp_path)

        assert ir._write_calib_provenance("i8", None, ir.TARGET_DIT, tmp_path) is None
        assert not (tmp_path / "calib_provenance.json").exists()

    def test_it_is_not_written_for_other_targets(self, tmp_path):
        assert ir._write_calib_provenance("i4", make_plan(), ir.TARGET_BACKBONE, tmp_path) is None
        assert not (tmp_path / "calib_provenance.json").exists()
