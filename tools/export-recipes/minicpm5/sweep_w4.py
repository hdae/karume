"""MiniCPM5-1B の w4 fake-quant sweep（ADR [0069](../../../docs/decisions/0069-packed-w4-storage.md)
決定 6 = Phase 0）— 既定 group_size と〔裁定 B: zero-point 欄なし〕の実測根拠を採る台本。

    uv run --with 'transformers==5.14.1' python -m minicpm5.sweep_w4

**runtime は 1 行も触らない**。ここでやるのは torch 側で重みを丸めて（fake-quant）品質の
劣化を測ることだけで、格納形（`storage.dtype: "i4"` / pack 順 / WGSL）は測定の結果を見て
から実装する。ADR 0058 決定 5 の分担で言えば「品質 = 人間レビュー + 助言的数値」の
数値側を作る側で、機械門はここでは baseline の再現性 1 本だけ（{@link
assert_baseline_reproduces}）。

## 何を測るか

group_size {32, 64, 128} × 対称 / 非対称の 6 通り + baseline + `lm_head` 除外の 1 本
（{@link SWEEP_CONFIGS}）。config ごとに 3 列を採る:

1. **weight 相対 RMSE**（族別）— 丸めそのものの大きさ。族内の全層をまとめた
   `‖w − fq‖₂ / ‖w‖₂` で、どの族が壊れやすいかを見る。
2. **teacher-forced 一致 / NLL** — 期待列を 1 つずつ与えたときの次トークン。発散しても列が
   ずれないので、config 間で**同じ位置**の劣化量を比べられる。
3. **自由走行 greedy の発散 step** — 実際の生成が期待列から離れるまでの長さ。人間が読む
   ときの「壊れ方」に一番近い列。

期待列は**波 E の資産が正本**（`outputs/series/minicpm5-1b-decode/greedy.<case>.safetensors`
— margin 門つきで採った 3 ケース）。ここで採り直さないのは、sweep が測りたいのが
「同じ期待列に対する量子化の劣化」だけだから。tokenizer は通さない（トークン id で完結する）。

## 量子化の形（ADR 0069 決定 3）

group 軸は**重みの最終次元 = linear の in 軸**。対称は `s = clamp(amax/7, f32 tiny)` /
`q = clamp(round(w/s), ±7)` で、amax 要素が厳密復元されるので fake-quant が冪等になる
（ADR 0019 の ±127 論証の 4bit 版）。非対称は**測定列**で、zero-point を丸めも格納もしない
連続値のまま使う — 「zero-point を足したときに届きうる品質の上界」を見て、対称の品質不足を
実装前に検知するための列（決定 3 の予約 3）。

## 出力

進捗は stderr へ即 flush（背景実行で追うため）。最後に stdout へ markdown 表 2 枚
（品質 / 族別 wRMSE）を出す — そのまま `docs/research/` へ転記する形。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from karume.quantize import QuantizeError
from minicpm5 import export as one_shot
from minicpm5 import export_decode as decode

#: 実重みの置き場（1-shot / decode 形と同じ素材）。
DEFAULT_MODEL_DIR = one_shot.DEFAULT_MODEL_DIR

#: 期待列（波 E の greedy golden）の置き場。
DEFAULT_DECODE_DIR = decode.DEFAULT_OUT_DIR

#: 対称量子化の片側幅（**−8 は使わない**）。±7 に閉じると group の amax 要素が `q = ±7` に
#: 乗って `q·s` で厳密に復元され、fake-quant が**冪等**になる（ADR 0069 決定 3 — ADR 0019 の
#: ±127 論証の 4bit 版）。−8 を許すと scale だけが動く再量子化が起きる。
INT4_MAX = 7

#: 非対称の unsigned nibble の値域（**0 は未使用**の 15 準位）。対称の `q ∈ [−7,7]` を
#: offset 8 で載せた先と同じ値域で、pack 形式・pack 順は zero-point の有無で変わらない
#: （ADR 0069 決定 3 の予約 2）。
UINT4_MIN = 1
UINT4_MAX = 15

#: zero-point 省略時の既定（ADR 0069 決定 3 の予約 2 — dequant は `(u − 8)·scale`）。
#: 非対称の縮退 group はこの既定へ落ちる（{@link asymmetric_components}）。
DEFAULT_ZERO_POINT = 8

#: 量子化対象の linear 族（fqn の末尾モジュール名）。MiniCPM5-1B は 24 層 ×
#: q/k/v/o/gate/up/down + `lm_head` の 169 本で、`tie_word_embeddings: false` なので
#: `lm_head` は独立した重み。族に割るのは「どこが壊れるか」を分離するため。
LINEAR_FAMILIES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
)

#: 語彙側の族名（`--only` 相当の除外指定 {@link SweepConfig.include_lm_head} が指す先）。
LM_HEAD = "lm_head"

#: baseline の config 名（{@link select_configs} が常に先頭へ入れる）。
BASELINE_NAME = "baseline"


@dataclass(frozen=True)
class SweepConfig:
    """1 実行ぶんの指定。

    `group_size is None` は量子化しない baseline。`include_lm_head=False` は `lm_head` を
    対象から外す列で、語彙側 200M 本の寄与を block 側から分離するために置く。
    """

    name: str
    group_size: int | None = None
    asymmetric: bool = False
    include_lm_head: bool = True


#: sweep グリッド（この順で走る）。group_size は 2 冪かつ ≥ 16（ADR 0069 決定 2）。
SWEEP_CONFIGS: tuple[SweepConfig, ...] = (
    SweepConfig(BASELINE_NAME),
    SweepConfig("g32-sym", group_size=32),
    SweepConfig("g64-sym", group_size=64),
    SweepConfig("g128-sym", group_size=128),
    SweepConfig("g32-asym", group_size=32, asymmetric=True),
    SweepConfig("g64-asym", group_size=64, asymmetric=True),
    SweepConfig("g128-asym", group_size=128, asymmetric=True),
    SweepConfig("g128-sym-blocks", group_size=128, include_lm_head=False),
)


# ---- 量子化式（ADR 0069 決定 3）--------------------------------------------


def grouped_view(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """量子化軸（**最終次元 = linear の in 軸**）を `[…, groups, group_size]` へ割る。

    MUST: 割り切れない形は fail loudly。`i4` は in 軸が `group_size` で割り切れることを
    MUST とする（ADR 0069 決定 2 — 端数 group を作らない制約で行境界・group 境界のバイト
    整列を保証している）ので、端数を黙って作った形の品質を測ると「格納できない形の数値」を
    根拠に既定を決めることになる。
    """
    in_axis = int(weight.shape[-1])
    if in_axis % group_size:
        raise QuantizeError(
            f"in 軸 {in_axis} が group_size {group_size} で割り切れない"
            "（`i4` は量子化軸が group_size で割り切れることを MUST とする — ADR 0069 決定 2）"
        )
    return weight.reshape(*weight.shape[:-1], in_axis // group_size, group_size)


def symmetric_components(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """group 対称量子化の `(q, scale)` を group 形（`[…, groups, *]`）で返す。

    `scale = clamp(amax / 7, f32 tiny)` / `q = clamp(round(w / scale), ±7)`。下限 clamp が
    要るのは全ゼロ group（`amax == 0`）— そのまま割ると NaN になる。clamp 後は `q = 0` に
    落ちて `q·scale = 0` が厳密に元値へ戻る（`quantize.channel_scale` と同文の 4bit 版）。

    fq（{@link fake_quant_symmetric}）と別に成分を返すのは、値域（**−8 が出ないこと**）と
    group ごとに scale が分かれていることをテストが直接縛れるようにするため。
    """
    grouped = grouped_view(weight, group_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax / INT4_MAX, min=torch.finfo(torch.float32).tiny)
    return torch.round(grouped / scale).clamp_(-INT4_MAX, INT4_MAX), scale


def fake_quant_symmetric(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """対称 15 準位の fake-quant（`q·scale` を weight と同じ形で返す）。

    MUST: **冪等**（再適用でビット不変）。amax 要素が `q = ±7` に乗って厳密復元されることの
    帰結で、崩れると「丸め済みのはずの重みが再量子化のたびに動く」= 格納形と参照の対応が
    切れる（ADR 0019 / 0069 決定 3）。`tests/test_sweep_w4.py` が固定する。
    """
    q, scale = symmetric_components(weight, group_size)
    return (q * scale).reshape(weight.shape)


def asymmetric_components(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """非対称（**連続** zero-point）の `(u, scale, zero)` を group 形で返す — 測定専用。

    `scale = clamp((max − min) / 14, f32 tiny)` / `zero = 1 − min/scale` /
    `u = clamp(round(w/scale + zero), 1, 15)`。zero を丸めも格納もしないので、これは
    「zero-point 欄を足したときに届きうる品質の**上界**」であって格納形ではない
    （ADR 0069 決定 3 の予約 3 — 対称の品質不足を実装前に検知するための測定列）。

    MUST: 縮退 group（`max == min`）は**対称式へ落とす**（`scale = clamp(|c|/7, tiny)` /
    `zero = 8`）。min-max 式では `scale` が下限 clamp に張り付いて `w/scale` が発散し、
    定数 group が clamp で ~0 へ潰れる（0 でない定数 group が黙って 0 になる沈黙誤値）。
    """
    grouped = grouped_view(weight, group_size)
    tiny = torch.finfo(torch.float32).tiny
    upper = grouped.amax(dim=-1, keepdim=True)
    lower = grouped.amin(dim=-1, keepdim=True)
    degenerate = upper == lower
    span = torch.clamp((upper - lower) / (UINT4_MAX - UINT4_MIN), min=tiny)
    scale = torch.where(degenerate, torch.clamp(upper.abs() / INT4_MAX, min=tiny), span)
    zero = torch.where(
        degenerate,
        torch.full_like(span, float(DEFAULT_ZERO_POINT)),
        UINT4_MIN - lower / span,
    )
    u = torch.round(grouped / scale + zero).clamp_(UINT4_MIN, UINT4_MAX)
    return u, scale, zero


def fake_quant_asymmetric(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """非対称 15 準位の fake-quant（`(u − zero)·scale` を weight と同じ形で返す）。

    NOTE: 対称と違い**冪等は保証しない** — min / max 要素の復元は f32 の丸めの分だけ緩く、
    再適用で `scale` が 1ulp 動きうる。測定列なので冪等は要求しない（格納形は対称のみで
    始める — ADR 0069 決定 3）。
    """
    u, scale, zero = asymmetric_components(weight, group_size)
    return ((u - zero) * scale).reshape(weight.shape)


# ---- 量子化対象 -------------------------------------------------------------


def family_of(fqn: str) -> str:
    """重みの fqn（`….q_proj.weight`）→ 族名（末尾のモジュール名）。"""
    return fqn.split(".")[-2]


def linear_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """全 `nn.Linear` の `weight` を fqn 引きで集める（bias / norm / embedding は触らない）。

    fqn は `named_modules` で引く（`quantize.fake_quant_int8` と同じ流儀 — `id(tensor)` で
    突き合わせない。パラメータ同一性は正規化で簡単に崩れ、崩れても黙って対象が減るだけ）。

    MUST: 族が {@link LINEAR_FAMILIES} に無い linear は fail loudly。transformers 側の構成が
    変わって対象が増減したことを黙って測り続けると、族別の表が別のモデルの数値になる。
    """
    weights: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        fqn = f"{name}.weight"
        if family_of(fqn) not in LINEAR_FAMILIES:
            raise QuantizeError(
                f"'{fqn}': 未知の linear 族（対象は {list(LINEAR_FAMILIES)}）"
                "— 模型の構成が sweep の想定と違う"
            )
        weights[fqn] = module.weight
    if not weights:
        raise QuantizeError("nn.Linear が 1 本も無い（模型の構成が sweep の想定と違う）")
    return weights


# ---- 期待列（波 E の資産）---------------------------------------------------


@dataclass(frozen=True)
class GreedyCase:
    """波 E の greedy 期待列 1 ケース（`prompt[1,T]` / `expected[K]` — どちらも i64）。"""

    name: str
    prompt: torch.Tensor
    expected: torch.Tensor

    @property
    def steps(self) -> int:
        return int(self.expected.numel())


def load_cases(decode_dir: Path) -> tuple[GreedyCase, ...]:
    """`greedy.<case>.safetensors` を glob で読む（tokenizer は通さない）。

    資産が無ければ**実行手順つきで**落とす — 60 分級の実行の入口で、直せる形の失敗を
    「ファイルが無い」だけで返さない。
    """
    paths = sorted(decode_dir.glob(f"{decode.GREEDY_PREFIX}*{one_shot.IO_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(
            f"greedy 期待列が {decode_dir} に 1 件も無い — 先に波 E の decode 資産を作ること:\n"
            "  cd tools/export-recipes && "
            "uv run --with 'transformers==5.14.1' python -m minicpm5.export_decode"
        )
    cases: list[GreedyCase] = []
    for path in paths:
        name = path.name[len(decode.GREEDY_PREFIX) : -len(one_shot.IO_SUFFIX)]
        tensors = load_file(str(path))
        cases.append(
            GreedyCase(
                name=name,
                prompt=tensors[decode.PROMPT_KEY].to(torch.int64).unsqueeze(0),
                expected=tensors[decode.EXPECTED_KEY].to(torch.int64),
            )
        )
    return tuple(cases)


# ---- 測定 -------------------------------------------------------------------


@dataclass(frozen=True)
class CaseResult:
    """1 config × 1 ケースの測定値。"""

    steps: int
    matches: int
    nll: float
    prefix: int


@dataclass(frozen=True)
class ConfigResult:
    """1 config の測定値（族別 wRMSE は量子化した族だけが載る）。"""

    name: str
    weight_rmse: Mapping[str, float]
    cases: Mapping[str, CaseResult]


def teacher_forced(wrapper: nn.Module, case: GreedyCase) -> tuple[int, float]:
    """`cat(prompt, expected)` を 1 forward して `(一致数, NLL 平均)` を返す。

    読むのは位置 `T−1 … T+K−2` の K 行 —「prompt の末尾から expected を 1 つずつ与えたときの
    次トークン」で、自由走行と違い**発散しても列がずれない**ので config 間で同じ位置の劣化を
    比べられる。NLL は同じ K 行の `−log softmax(logits)[expected]` の平均で、1 位が変わる
    手前の劣化（順位は保つが確信が薄れる）まで拾う。
    """
    ids = torch.cat([case.prompt, case.expected.unsqueeze(0)], dim=1)
    with torch.no_grad():
        logits = wrapper(ids)
    start = int(case.prompt.shape[1]) - 1
    rows = logits[0, start : start + case.steps]
    matches = int((rows.argmax(dim=-1) == case.expected).sum())
    log_probs = torch.log_softmax(rows.to(torch.float32), dim=-1)
    nll = float(-log_probs.gather(1, case.expected.unsqueeze(1)).mean())
    return matches, nll


def greedy_prefix(wrapper: nn.Module, case: GreedyCase) -> int:
    """自由走行 greedy で expected と一致し続けた先頭 step 数（全一致なら K）。

    MUST: full re-forward（毎 step 先頭から標準 causal で計算 — `export_decode
    .greedy_continuation` と同じ流儀）。KV cache 経路で採ると、量子化の劣化と cache 実装の
    影響が混ざる。発散した時点で打ち切る — 以後の列は「別の文」なので比べる意味が無く、
    劣化した config ほど速く終わる。
    """
    current = case.prompt
    for step in range(case.steps):
        with torch.no_grad():
            logits = wrapper(current)
        token = int(logits[0, -1].argmax())
        if token != int(case.expected[step]):
            return step
        current = torch.cat([current, torch.tensor([[token]], dtype=torch.int64)], dim=1)
    return case.steps


def apply_config(
    weights: Mapping[str, torch.Tensor],
    pristine: Mapping[str, torch.Tensor],
    config: SweepConfig,
) -> dict[str, float]:
    """pristine から復元してから config の fake-quant を当て、族別の相対 RMSE を返す。

    MUST: 毎回 pristine から戻す。戻さずに当てると測っているのが「group 32 の上の
    group 128」になり、config 間の比較そのものが意味を失う。

    相対 RMSE は族内の全層をまとめた `‖w − fq‖₂ / ‖w‖₂`（要素数が約分されるので層の大きさで
    重み付けされない）。総和は f64 で採る — 族によっては 2 億要素を足すので f32 累算では
    桁落ちが数値の意味を食う。
    """
    errors: dict[str, float] = defaultdict(float)
    norms: dict[str, float] = defaultdict(float)
    with torch.no_grad():
        for fqn, weight in weights.items():
            weight.copy_(pristine[fqn])
        if config.group_size is None:
            return {}
        for fqn, weight in weights.items():
            family = family_of(fqn)
            if family == LM_HEAD and not config.include_lm_head:
                continue
            quantize = fake_quant_asymmetric if config.asymmetric else fake_quant_symmetric
            fq = quantize(weight, config.group_size)
            errors[family] += float((fq - weight).pow(2).sum(dtype=torch.float64))
            norms[family] += float(weight.pow(2).sum(dtype=torch.float64))
            weight.copy_(fq)
    return {family: math.sqrt(errors[family] / norms[family]) for family in errors}


def assert_baseline_reproduces(
    cases: Sequence[GreedyCase], measured: Mapping[str, CaseResult]
) -> None:
    """baseline（f32 のまま）が期待列を teacher-forced で**完全に**再現することを見る。

    MUST: fail loudly。ここが割れるのは量子化の話ではなく**ハーネスが壊れている**合図
    （重みの取り違え・位置のずれ・期待列と模型の不一致）で、以後の全 config の数値の意味が
    消える。部分再実行（`--only`）でも baseline を必ず先に走らせるのはこの門のため。
    """
    wrong = {
        case.name: f"{measured[case.name].matches}/{case.steps}"
        for case in cases
        if measured[case.name].matches != case.steps
    }
    if wrong:
        raise AssertionError(
            f"baseline の teacher-forced 一致が完全でない {wrong}"
            "（量子化以前の問題 — 模型・期待列・位置の対応を疑う）"
        )


def select_configs(only: Sequence[str]) -> tuple[SweepConfig, ...]:
    """`--only` で選んだ config を宣言順で返す（**baseline は常に先頭で走る**）。

    baseline を外せないのは、部分再実行でも sanity 門（{@link assert_baseline_reproduces}）を
    必ず通すため — 60 分級の実行を刻んで回すときほど、「壊れた模型で測り続ける」経路を
    残さないことが効く。
    """
    if not only:
        return SWEEP_CONFIGS
    chosen = {BASELINE_NAME, *only}
    return tuple(config for config in SWEEP_CONFIGS if config.name in chosen)


def run_sweep(
    model_dir: Path,
    configs: Sequence[SweepConfig],
    *,
    decode_dir: Path = DEFAULT_DECODE_DIR,
) -> list[ConfigResult]:
    """模型を 1 回だけ読み、config を順に当てて測る。

    pristine クローン（CPU・linear だけで ~3.5GB）を先に採るので、config ごとの丸めは常に
    元の重みから始まる。進捗は 1 ケース / 1 config ごとに stderr へ即 flush する。
    """
    cases = load_cases(decode_dir)
    lengths = " ".join(f"{case.name}(T={case.prompt.shape[1]},K={case.steps})" for case in cases)
    print(f"[sweep] 期待列 {len(cases)} ケース: {lengths}", file=sys.stderr, flush=True)

    wrapper = one_shot.load_wrapper(model_dir)
    weights = linear_weights(wrapper)
    print(f"[sweep] 量子化対象 {len(weights)} 本を pristine 退避", file=sys.stderr, flush=True)
    pristine = {fqn: weight.detach().clone() for fqn, weight in weights.items()}

    results: list[ConfigResult] = []
    for config in configs:
        started = time.perf_counter()
        weight_rmse = apply_config(weights, pristine, config)
        measured: dict[str, CaseResult] = {}
        for case in cases:
            matches, nll = teacher_forced(wrapper, case)
            prefix = greedy_prefix(wrapper, case)
            measured[case.name] = CaseResult(
                steps=case.steps, matches=matches, nll=nll, prefix=prefix
            )
            print(
                f"[{config.name}] {case.name} teacher {matches}/{case.steps}"
                f" nll {nll:.4g} greedy {prefix}/{case.steps}",
                file=sys.stderr,
                flush=True,
            )
        if config.name == BASELINE_NAME:
            assert_baseline_reproduces(cases, measured)
        results.append(ConfigResult(config.name, weight_rmse, measured))
        print(
            f"[{config.name}] 完了 {time.perf_counter() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    return results


# ---- 表 ---------------------------------------------------------------------


def _markdown(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [f"| {' | '.join(header)} |", f"| {' | '.join('---' for _ in header)} |"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(lines)


def quality_table(results: Sequence[ConfigResult]) -> str:
    """config × ケース別［teacher 一致 / NLL / 発散 step］の表。"""
    names = list(results[0].cases)
    header = ["config"]
    for name in names:
        header += [f"{name} 一致", f"{name} NLL", f"{name} 発散"]
    rows = []
    for result in results:
        row = [result.name]
        for name in names:
            case = result.cases[name]
            row += [
                f"{case.matches}/{case.steps}",
                f"{case.nll:.4g}",
                f"{case.prefix}/{case.steps}",
            ]
        rows.append(row)
    return _markdown(header, rows)


def rmse_table(results: Sequence[ConfigResult]) -> str:
    """config × 族別 weight 相対 RMSE の表（量子化しなかった族は `-`）。"""
    rows = [
        [result.name]
        + [
            f"{result.weight_rmse[family]:.4g}" if family in result.weight_rmse else "-"
            for family in LINEAR_FAMILIES
        ]
        for result in results
    ]
    return _markdown(["config", *LINEAR_FAMILIES], rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--decode-dir", type=Path, default=DEFAULT_DECODE_DIR)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        choices=[config.name for config in SWEEP_CONFIGS],
        help="この config だけ走らせる（複数可・部分再実行用）。baseline は常に先行する。",
    )
    args = parser.parse_args(argv)
    results = run_sweep(args.model_dir, select_configs(args.only), decode_dir=args.decode_dir)
    print(quality_table(results))
    print()
    print(rmse_table(results))


if __name__ == "__main__":
    main()
