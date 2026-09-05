"""Q0 量子化計測のレポート生成（`measure_quant` の数値エンジンから切り出した文言・整形の島）。

`report.md` / `attn.json` / `layers.csv` を組み立てる。engine 側の値（構成表・基準名・指標
関数）は**全て引数で受け取る** — 実行時 import は `measure_quant` → 本モジュールの一方向だけに
保ち、`--until` が書き換えるモジュールグローバルを見に行かないため（型注釈だけ TYPE_CHECKING）。
"""

from __future__ import annotations

import csv
import math
import re
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from .pipeline_ref import PROMPT, SEED

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable
    from pathlib import Path
    from typing import TypedDict

    from .measure_quant import AttnStat, DitRun, LayerStat

    #: engine 側の指標関数（逆辺 import を作らないので呼び出し側から渡す）。
    RelRms = Callable[[torch.Tensor, torch.Tensor], float]
    Psnr = Callable[[torch.Tensor, torch.Tensor, float], float]
    ToUint8 = Callable[[torch.Tensor], torch.Tensor]
    HistQuantile = Callable[[torch.Tensor, float], float]

    class PtGroupSummary(TypedDict):
        """`pt.by_group` の値（層グループ 1 本ぶんの P̃ 集計）。"""

        nodes: int
        median: float
        p99: float
        max: float

    class PtSummary(TypedDict):
        """`build_summary` の `pt` キー（P̃ 行 peak/rms の全体集計）。"""

        rows: int
        nodes: int
        median: float
        p90: float
        p99: float
        max: float
        mean: float
        by_group: dict[str, PtGroupSummary]

    class ConfigSummary(TypedDict):
        """`build_summary` の `configs` の値（構成 1 本ぶんの集計）。"""

        diagnostics: dict[str, object]
        latent_rel_rms_vs_f32: list[float]
        latent_rel_rms_vs_baseline: list[float]
        latent_rel_rms_vs_w8: list[float]
        latent_ratio_vs_w8: list[float]
        latent_ratio_vs_baseline: list[float]
        latent_ratio_final: float
        latent_ratio_max: float
        psnr_vs_baseline: float
        psnr_vs_f32: float
        attn_s_rel_rms_max: float
        attn_o_rel_rms_max: float
        attn_o_worst_node: str | None

    class Summary(TypedDict):
        """`build_summary` の戻り値（`attn.json` と report ⑥ の共通の出どころ）。"""

        pt: PtSummary
        configs: dict[str, ConfigSummary]
        gates: dict[str, str]
        #: 赤だった門のラベル（判定の正本 — 表示文字列から嗅ぎ直さないための席）。
        failed: list[str]
        inject: str | None


#: 層名の連番を潰して役割でまとめる（`transformer_blocks.7.attn1.to_q` → `transformer_blocks.*.…`）
_INDEX = re.compile(r"\.\d+(?=\.|$)")


def layer_group(name: str) -> str:
    return _INDEX.sub(".*", name)


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _db(value: float) -> str:
    return "inf" if value <= 0.0 else f"{-20.0 * math.log10(value):.1f}"


def write_layer_csv(stats: dict[str, LayerStat], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["name", "k", "n", "calls", "in_rel_rms", "out_rel_rms", "peak_mean", "peak_max"]
        )
        for name, stat in sorted(stats.items(), key=lambda kv: -kv[1].in_rel_rms()):
            writer.writerow(
                [
                    name,
                    stat.in_features,
                    stat.out_features,
                    stat.calls,
                    f"{stat.in_rel_rms():.6e}",
                    f"{stat.out_rel_rms():.6e}",
                    f"{stat.peak_mean():.3f}",
                    f"{stat.peak_max:.3f}",
                ]
            )


#: 門（設計 doc §5.3）。
PT_MEDIAN_GATE = 16.0
LATENT_RATIO_GATE = 1.5


def build_summary(
    args: argparse.Namespace,
    run: DitRun,
    latents: dict[str, torch.Tensor],
    images: dict[str, torch.Tensor],
    *,
    config_names: tuple[str, ...],
    attn_configs: tuple[str, ...],
    baseline: str,
    pt_hist_bins: int,
    rel_rms: RelRms,
    psnr: Psnr,
    hist_quantile: HistQuantile,
) -> Summary:
    """機械可読の集計と門の判定（`attn.json` と report ⑥ の共通の出どころ）。"""
    keys = [f"latents_step{index:04d}" for index in range(1, args.steps + 1)]
    base_vs_f32 = [rel_rms(latents[f"{baseline}/{key}"], latents[f"f32/{key}"]) for key in keys]
    w8_vs_f32 = [rel_rms(latents[f"w8/{key}"], latents[f"f32/{key}"]) for key in keys]

    pt_stats = run.attn_stats.get(baseline, {})
    merged = (
        torch.stack([stat.pt_hist for stat in pt_stats.values()]).sum(0)
        if pt_stats
        else torch.zeros(pt_hist_bins, dtype=torch.int64)
    )
    rows = sum(stat.pt_rows for stat in pt_stats.values())
    pt: PtSummary = {
        "rows": rows,
        "nodes": len(pt_stats),
        "median": hist_quantile(merged, 0.5),
        "p90": hist_quantile(merged, 0.9),
        "p99": hist_quantile(merged, 0.99),
        "max": max((stat.pt_max for stat in pt_stats.values()), default=0.0),
        "mean": (sum(stat.pt_sum for stat in pt_stats.values()) / rows) if rows else 0.0,
        "by_group": {
            group: {
                "nodes": len(members),
                "median": hist_quantile(torch.stack([m.pt_hist for m in members]).sum(0), 0.5),
                "p99": hist_quantile(torch.stack([m.pt_hist for m in members]).sum(0), 0.99),
                "max": max(m.pt_max for m in members),
            }
            for group, members in sorted(_group_by(pt_stats, layer_group).items())
        },
    }

    configs: dict[str, ConfigSummary] = {}
    for name in config_names:
        vs_f32 = [rel_rms(latents[f"{name}/{key}"], latents[f"f32/{key}"]) for key in keys]
        vs_base = [rel_rms(latents[f"{name}/{key}"], latents[f"{baseline}/{key}"]) for key in keys]
        ratios = [
            value / reference if reference > 0 else math.inf
            for value, reference in zip(vs_f32, base_vs_f32, strict=True)
        ]
        node_stats = run.attn_stats.get(name, {})
        measured = {key: stat for key, stat in node_stats.items() if stat.measured}
        configs[name] = {
            "diagnostics": run.diagnostics[name],
            "latent_rel_rms_vs_f32": vs_f32,
            "latent_rel_rms_vs_baseline": vs_base,
            "latent_rel_rms_vs_w8": [
                rel_rms(latents[f"{name}/{key}"], latents[f"w8/{key}"]) for key in keys
            ],
            "latent_ratio_vs_w8": [
                value / reference if reference > 0 else math.inf
                for value, reference in zip(vs_f32, w8_vs_f32, strict=True)
            ],
            "latent_ratio_vs_baseline": ratios,
            "latent_ratio_final": ratios[-1],
            "latent_ratio_max": max(ratios),
            "psnr_vs_baseline": psnr(images[name], images[baseline], 2.0),
            "psnr_vs_f32": psnr(images[name], images["f32"], 2.0),
            "attn_s_rel_rms_max": max((s.s_rel_rms() for s in measured.values()), default=0.0),
            "attn_o_rel_rms_max": max((s.o_rel_rms() for s in measured.values()), default=0.0),
            "attn_o_worst_node": max(measured, key=lambda k: measured[k].o_rel_rms())
            if measured
            else None,
        }

    gates: dict[str, str] = {}
    failed: list[str] = []

    def gate(label: str, ok: bool, verdict: str) -> None:
        """門を 1 つ記録する（判定は**真偽で持ち**、`verdict` はその表示形）。

        MUST: 赤の集計を文字列から嗅ぎ直さない — 判定はここで一度だけ決め、呼び手
        （`measure_quant.main`）は {@link Summary} の `failed` を読む。文字列の中の `NG` は
        書式（先頭 / 末尾 / 括弧付き）が門ごとに違うので、後から拾うと取り逃す。
        """
        gates[label] = verdict
        if not ok:
            failed.append(label)

    pt_ok = pt["median"] <= PT_MEDIAN_GATE
    gate(
        f"P̃ 行 peak/rms の median ≤ {PT_MEDIAN_GATE:g}",
        pt_ok,
        f"median {pt['median']:.2f} → {'OK' if pt_ok else 'NG'}"
        f"（p99 {pt['p99']:.2f} / max {pt['max']:.2f}）",
    )
    for name in attn_configs:
        # 設計 doc §5.3 の門は「(c) w8a8 に attention を足したときの相対劣化」なので、
        # linear 活性を量子化しない (g) は**門の対象外**（`w8` 基準の参考値として出す）。
        if run.diagnostics[name]["linear_act_i8"]:
            series = configs[name]["latent_ratio_vs_baseline"]
            ratio = max(series)
            ratio_ok = ratio <= LATENT_RATIO_GATE
            gate(
                f"`{name}` の latent relRMS ≤ {LATENT_RATIO_GATE:g}× (vs {baseline})",
                ratio_ok,
                f"最大 {ratio:.3f}× (step {1 + series.index(ratio)})"
                f" → {'OK' if ratio_ok else 'NG'}",
            )
        else:
            series = configs[name]["latent_ratio_vs_w8"]
            ratio = max(series)
            gates[f"`{name}` の latent relRMS（参考・門の対象外）"] = (
                f"`w8` 比 最大 {ratio:.3f}×"
                f" (step {1 + series.index(ratio)})"
                " — linear 活性 f32 運用での attention 単独寄与"
            )
    for name in config_names:
        diagnostic = run.diagnostics[name]
        if diagnostic["attn_calls_expected"]:
            ok = diagnostic["attn_calls"] == diagnostic["attn_calls_expected"]
            gate(
                f"`{name}` の SDPA 発火計数",
                ok,
                f"{diagnostic['attn_calls']} / {diagnostic['attn_calls_expected']}"
                f" → {'OK' if ok else 'NG'}",
            )
    for name in config_names:
        diagnostic = run.diagnostics[name]
        if not diagnostic["score_f16"]:
            continue
        # 恒真化の防止: 「丸めたつもりで恒等」は数値にも時間にも出ない（沈黙する）ので、
        # 丸め前後で 1 要素以上変わった呼び出しの数を直接見る。
        fired, attempts = diagnostic["s16_fired_calls"], diagnostic["s16_rounded_calls"]
        ok = bool(attempts) and fired == attempts
        gate(
            f"`{name}` の S f16 丸めが全呼び出しで発火",
            ok,
            f"{fired} / {attempts}（{diagnostic['s16_elements']:,} 要素・"
            f"|S| 最大 {diagnostic['s16_abs_max']:.1f} < f16 上限 65,504）"
            f" → {'OK' if ok else 'NG'}",
        )
    for name in attn_configs:
        differs = not torch.equal(latents[f"{name}/{keys[-1]}"], latents[f"{baseline}/{keys[-1]}"])
        gate(f"`{name}` が `{baseline}` と異なる", differs, "OK" if differs else "NG（素通し）")

    return {
        "pt": pt,
        "configs": configs,
        "gates": gates,
        "failed": failed,
        "inject": args.inject,
    }


def _group_by(stats: dict[str, AttnStat], key) -> dict[str, list[AttnStat]]:
    grouped: dict[str, list[AttnStat]] = defaultdict(list)
    for name, stat in stats.items():
        grouped[key(name)].append(stat)
    return grouped


def build_report(
    args: argparse.Namespace,
    run: DitRun,
    latents: dict[str, torch.Tensor],
    images: dict[str, torch.Tensor],
    image_note: str,
    summary: Summary,
    *,
    config_names: tuple[str, ...],
    attn_configs: tuple[str, ...],
    baseline: str,
    guidance: float,
    pt_hist_bins: int,
    rel_rms: RelRms,
    psnr: Psnr,
    to_uint8: ToUint8,
    hist_quantile: HistQuantile,
) -> str:
    stats = run.linear_stats
    lines: list[str] = []
    lines.append("# Q0 — Anima DiT の w8 / w8a8 / attention a8 量子化品質（torch CPU 実測）")
    lines.append("")
    lines.append(
        f"- 日付: {time.strftime('%Y-%m-%d')} / 計測: `tools/export-recipes/anima/measure_quant.py`"
    )
    lines.append(
        f"- 条件: Anima turbo {args.steps} step・{args.resolution}px・CFG={guidance:g}"
        f"・LoRA 焼き込み・seed {SEED}・torch {torch.__version__}（CPU f32）"
    )
    lines.append(f"- プロンプト: `{PROMPT}`")
    lines.append(
        "- 構成: **(a) f32** = LoRA 焼き込み済みの素の重み / **(b) w8** = 重みのみ per-channel "
        "symmetric i8 fake-quant / **(c) w8a8** = (b) + 適格 linear の入力活性を per-token "
        "symmetric i8 fake-quant / **(d)〜(h)** = (c) または (b) に attention の i8 化を足したもの "
        "/ **(i)(j) `*-s16`** = さらに attention スコア **S を f16 で格納**（案γ 波 1）"
    )
    lines.append(
        "- 活性量子化は設計 doc §4.2 の鏡像（`s = clamp(rowmax|x|/127, tiny)`・"
        "`q = clamp(round(x/s), ±127)`・偶数丸め・全ゼロ行は `s = tiny`）。"
        "適格は §4.1 と同じ「DiT 内 `nn.Linear` で `k % 4 == 0`」。"
    )
    lines.append(
        "- attention の粒度は attention a8 設計 doc §5.1 の MUST 表"
        "（q/k = 行ごと amax・P̃ = **scale 固定 1/127**・V = `(b,h,d)` ごとに N 全体の amax・"
        "半スケールは dequant 側・整数縮約は float64 で厳密）。"
    )
    lines.append(
        "- **S の f16 格納**（`*-s16`）は案γ 設計 doc §3.4② / §7 波 1 の実装形と粒度厳密一致: "
        "①QK が書く半スケール適用済みの生スコアを f16 へ落として戻し、**丸めはその 1 回だけ**。"
        "②行 max `m` と分母 `l`・③の `exp(S−m)` と `qP = round(127·exp(S−m))` は**全て同じ"
        "丸め済み S** から f32 で計算する。"
        "NOTE: `*-s16` は素通し（torch SDPA）ではなく分解経路で評価されるので、差には f32 の"
        "演算順差（1e-7 級）が混じる — f16 の相対 eps 4.88e-4 に対し 3 桁下。"
    )
    lines.append(
        "- **非 DiT（text_encoder / text_conditioner / VAE）は全構成とも f32**。"
        "比較軸を DiT の量子化に絞るため（資産系列の f16 差は全構成へ同形に乗る）。"
        "attention の SDPA 差し替えも denoise ループの間だけで、text encode / VAE decode は素通し。"
    )
    lines.append("")

    lines.append("## ⓪ 構成と発火計数")
    lines.append("")
    rows = []
    for name in config_names:
        diagnostic = run.diagnostics[name]
        rows.append(
            [
                f"`{name}`",
                "i8" if diagnostic["weight_i8"] else "f32",
                "i8" if diagnostic["linear_act_i8"] else "f32",
                str(diagnostic["attn"] or "—"),
                "f16" if diagnostic["score_f16"] else "f32",
                "量子化後総和" if diagnostic["stats_quantized"] else "f32 総和",
                f"{diagnostic['attn_calls']} / {diagnostic['attn_calls_expected']}",
                str(diagnostic["attn_nodes_seen"]),
                f"{diagnostic['elapsed']:.0f}s",
            ]
        )
    lines.append(
        _table(
            rows,
            [
                "構成",
                "重み",
                "linear 活性",
                "attention",
                "S 格納",
                "softmax 分母",
                "SDPA 発火 / 期待",
                "ノード数",
                "所要",
            ],
        )
    )
    lines.append("")
    lines.append(
        f"期待値 = attention ノード {len(run.attention_nodes)} 本 × {args.steps} step。"
        "0 や不足は「量子化しているつもりで素通し」の唯一の検出器。"
    )
    lines.append("")

    lines.append("## ① step ごとの latent relRMS")
    lines.append("")
    lines.append(
        "`relRMS(v, r) = ‖v − r‖₂ / ‖r‖₂`（括弧内は `−20·log₁₀(relRMS)` の dB）。"
        f"attention 構成の基準は **{baseline}**（f32 比は既に飽和しているため — 設計 doc §5.3）。"
    )
    lines.append("")
    rows = []
    for index in range(1, args.steps + 1):
        key = f"latents_step{index:04d}"
        base = latents[f"f32/{key}"]
        w8 = latents[f"w8/{key}"]
        w8a8 = latents[f"w8a8/{key}"]
        a = rel_rms(w8, base)
        b = rel_rms(w8a8, base)
        c = rel_rms(w8a8, w8)
        rows.append(
            [
                str(index),
                f"{a:.4e} ({_db(a)} dB)",
                f"{b:.4e} ({_db(b)} dB)",
                f"{c:.4e} ({_db(c)} dB)",
            ]
        )
    lines.append(_table(rows, ["step", "w8 vs f32", "w8a8 vs f32", "w8a8 vs w8"]))
    lines.append("")
    # attention 構成が 1 つも走っていない（--until による前方部分実行）ときは丸ごと出さない。
    if attn_configs:
        lines.append(f"### attention 構成の latent relRMS（基準 = `{baseline}`）")
        lines.append("")
        rows = []
        for index in range(1, args.steps + 1):
            key = f"latents_step{index:04d}"
            reference = latents[f"{baseline}/{key}"]
            row = [str(index)]
            for name in attn_configs:
                row.append(f"{rel_rms(latents[f'{name}/{key}'], reference):.4e}")
            rows.append(row)
        lines.append(_table(rows, ["step", *(f"`{name}`" for name in attn_configs)]))
        lines.append("")
        lines.append(
            "倍率（最終 step・`f32` 比 relRMS を `w8a8` のそれで割った値）: "
            + " / ".join(
                f"`{name}` {value['latent_ratio_final']:.2f}×"
                for name, value in summary["configs"].items()
                if name in attn_configs
            )
        )
        lines.append("")
    if "w8-qkpv" in summary["configs"]:
        lines.append(
            "**`w8-qkpv` だけは基準が違う**（linear 活性が f32 なので `w8a8` より良くて当然）。"
            "attention 単独の寄与は `w8` 比で読む: 最終 step で "
            f"vs `w8` relRMS {summary['configs']['w8-qkpv']['latent_rel_rms_vs_w8'][-1]:.4e}"
            f"・`w8` の f32 比に対する倍率 "
            f"{summary['configs']['w8-qkpv']['latent_ratio_vs_w8'][-1]:.2f}×。"
        )
        lines.append("")

    lines.append("## ② VAE decode 後の最終画像")
    lines.append("")
    lines.append(
        "PSNR は `[-1,1]` の生 tensor（data range 2.0）と、PNG と同じ uint8 量子化後"
        "（data range 255）の 2 系列。relRMS は生 tensor。"
    )
    lines.append("")
    pairs = [("w8", "f32"), ("w8a8", "f32"), ("w8a8", "w8")]
    pairs += [(name, baseline) for name in attn_configs]
    pairs += [(name, "f32") for name in attn_configs]
    rows = []
    for value_tag, ref_tag in pairs:
        value, reference = images[value_tag], images[ref_tag]
        u8_value = to_uint8(value).to(torch.float32)
        u8_reference = to_uint8(reference).to(torch.float32)
        rows.append(
            [
                f"{value_tag} vs {ref_tag}",
                f"{psnr(value, reference, 2.0):.2f}",
                f"{psnr(u8_value, u8_reference, 255.0):.2f}",
                f"{rel_rms(value, reference):.4e}",
                f"{float((u8_value - u8_reference).abs().max()):.0f}",
            ]
        )
    lines.append(
        _table(
            rows,
            ["対", "PSNR f32 (dB)", "PSNR uint8 (dB)", "relRMS", "uint8 最大差 (/255)"],
        )
    )
    lines.append("")
    lines.append(f"画像: {image_note}（{' / '.join(f'`image_{n}`' for n in config_names)}）")
    lines.append("")

    lines.append("## ③ 層別 relRMS（適格 linear）")
    lines.append("")
    lines.append(
        "`in relRMS` = 入力活性の量子化誤差（全 step 通算）。"
        f"`out relRMS` = その層の出力が活性量子化で動いた量（step {args.measure_step + 1}"
        " のみ実測・重みは両側とも i8 なので**活性量子化だけ**の寄与）。"
        "`peak` = 行ごとの `amax/rms` の平均と最大（外れ値の指標 — "
        "`relRMS ≈ peak / (127·√12) = peak / 440` が一様量子化の目安）。"
    )
    lines.append("")
    groups: dict[str, list[LayerStat]] = defaultdict(list)
    for name, stat in stats.items():
        groups[layer_group(name)].append(stat)
    rows = []
    for name, members in sorted(groups.items(), key=lambda kv: -max(s.in_rel_rms() for s in kv[1])):
        in_values = [s.in_rel_rms() for s in members]
        out_values = [s.out_rel_rms() for s in members if s.out_calls]
        rows.append(
            [
                f"`{name}`",
                str(len(members)),
                str(members[0].in_features),
                f"{sum(in_values) / len(in_values):.3e}",
                f"{max(in_values):.3e}",
                f"{max(out_values):.3e}" if out_values else "—",
                f"{max(s.peak_mean() for s in members):.1f}",
                f"{max(s.peak_max for s in members):.1f}",
            ]
        )
    lines.append(
        _table(
            rows,
            [
                "層グループ",
                "本数",
                "k",
                "in relRMS 平均",
                "in relRMS 最大",
                "out relRMS 最大",
                "peak 平均 最大",
                "peak 最大",
            ],
        )
    )
    lines.append("")
    lines.append("### 個別 worst 20（in relRMS 降順）")
    lines.append("")
    worst = sorted(stats.items(), key=lambda kv: -kv[1].in_rel_rms())[:20]
    rows = [
        [
            f"`{name}`",
            str(stat.in_features),
            str(stat.out_features),
            f"{stat.in_rel_rms():.3e}",
            f"{stat.out_rel_rms():.3e}" if stat.out_calls else "—",
            f"{stat.peak_mean():.1f}",
            f"{stat.peak_max:.1f}",
        ]
        for name, stat in worst
    ]
    lines.append(
        _table(rows, ["層", "k", "n", "in relRMS", "out relRMS", "peak 平均", "peak 最大"])
    )
    lines.append("")
    lines.append(f"全 {len(stats)} 本の生表は `layers.csv`。")
    lines.append("")

    lines.append("## ④ P̃ の行 peak/rms 分布（PV 量子化 SNR を決める唯一の量）")
    lines.append("")
    lines.append(
        f"`{baseline}` の走行中に採った**素の P̃**（attention は f32 のまま = 量子化器が"
        "実際に見る値）。`peak` は `exp(0)=1` で構造的に 1.0 なので `peak/rms = 1/rms`。"
        "設計 doc §2.2 の予測は `relRMS(P̃) ≈ (peak/rms)/(127·√12) = (peak/rms)/440`。"
        f"全 {args.steps} step・全 head・全行を通算（分位点は log₂ 等間隔 {pt_hist_bins} ビンの"
        "ヒストグラムから補間 — 分解能 0.7%）。"
    )
    lines.append("")
    pt_stats = run.attn_stats.get(baseline, {})
    pt_groups: dict[str, list[AttnStat]] = defaultdict(list)
    for name, stat in pt_stats.items():
        pt_groups[layer_group(name)].append(stat)
    rows = []
    for name, members in sorted(pt_groups.items()):
        merged = torch.stack([member.pt_hist for member in members]).sum(0)
        rows.append(
            [
                f"`{name}`",
                str(len(members)),
                f"{sum(m.pt_rows for m in members):,}",
                f"{hist_quantile(merged, 0.5):.2f}",
                f"{hist_quantile(merged, 0.9):.2f}",
                f"{hist_quantile(merged, 0.99):.2f}",
                f"{max(m.pt_max for m in members):.2f}",
                f"{hist_quantile(merged, 0.5) / 440.0:.2e}",
                f"{hist_quantile(merged, 0.99) / 440.0:.2e}",
            ]
        )
    lines.append(
        _table(
            rows,
            [
                "層グループ",
                "ノード",
                "行数",
                "median",
                "p90",
                "p99",
                "max",
                "予測 relRMS(median)",
                "予測 relRMS(p99)",
            ],
        )
    )
    lines.append("")
    overall = summary["pt"]
    lines.append(
        f"**全体（{overall['rows']:,} 行）: median {overall['median']:.2f} / "
        f"p90 {overall['p90']:.2f} / p99 {overall['p99']:.2f} / max {overall['max']:.2f} / "
        f"mean {overall['mean']:.2f}**"
    )
    lines.append("")
    lines.append("### 個別ノード worst 12（median 降順）")
    lines.append("")
    worst_nodes = sorted(pt_stats.items(), key=lambda kv: -kv[1].pt_quantile(0.5))[:12]
    lines.append(
        _table(
            [
                [
                    f"`{name}`",
                    f"{stat.pt_quantile(0.5):.2f}",
                    f"{stat.pt_quantile(0.99):.2f}",
                    f"{stat.pt_max:.2f}",
                    f"{stat.pt_mean():.2f}",
                ]
                for name, stat in worst_nodes
            ],
            ["ノード", "median", "p99", "max", "平均"],
        )
    )
    lines.append("")

    lines.append("## ⑤ attention 層別 relRMS（S と O）")
    lines.append("")
    lines.append(
        f"step {args.measure_step + 1} のみ実測。基準は**同じ q/k/v から attention を量子化"
        "せずに計算した値**（linear の活性量子化は q/k/v の中に既に乗っているので、この差は"
        "attention 量子化だけの寄与）。"
    )
    lines.append("")
    for name in attn_configs:
        node_stats = run.attn_stats.get(name, {})
        measured = {key: stat for key, stat in node_stats.items() if stat.measured}
        if not measured:
            continue
        groups_attn: dict[str, list[AttnStat]] = defaultdict(list)
        for key, stat in measured.items():
            groups_attn[layer_group(key)].append(stat)
        lines.append(f"### `{name}`")
        lines.append("")
        rows = [
            [
                f"`{group}`",
                str(len(members)),
                f"{sum(m.s_rel_rms() for m in members) / len(members):.3e}",
                f"{max(m.s_rel_rms() for m in members):.3e}",
                f"{sum(m.o_rel_rms() for m in members) / len(members):.3e}",
                f"{max(m.o_rel_rms() for m in members):.3e}",
            ]
            for group, members in sorted(groups_attn.items())
        ]
        worst_key = max(measured, key=lambda key: measured[key].o_rel_rms())
        lines.append(_table(rows, ["層グループ", "本数", "S 平均", "S 最大", "O 平均", "O 最大"]))
        lines.append("")
        lines.append(
            f"worst 層（O relRMS）: `{worst_key}` = {measured[worst_key].o_rel_rms():.3e}"
            f"（S = {measured[worst_key].s_rel_rms():.3e}）"
        )
        lines.append("")

    lines.append("## ⑥ 門の判定")
    lines.append("")
    lines.append(
        "設計 doc §5.3 の 2 門: **① P̃ 行 peak/rms の中央値 ≤ 16**（PV が w8a8 と同じ土俵）/ "
        f"**② latent relRMS が `{baseline}` 比 ≤ 1.5×**。"
        "`*-s16` の門も②と同じ（案γ 設計 doc §6.3 — 基準は f32 ではなく受理済みの "
        f"`{baseline}`）。加えて **S f16 の丸めが全呼び出しで発火したか**を直接見る"
        "（恒真化の防止 — 丸めが恒等でも数値にも所要時間にも出ないため）。"
    )
    lines.append("")
    verdicts = summary["gates"]
    lines.append(
        _table(
            [[str(key), str(value)] for key, value in verdicts.items()],
            ["門", "判定"],
        )
    )
    lines.append("")
    return "\n".join(lines) + "\n"
