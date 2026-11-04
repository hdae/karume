"""Report generation for the Q0 quantization measurement (the island of wording and formatting split
out of the numeric engine in `measure_quant`).

Builds `report.md` / `attn.json` / `layers.csv`. The engine-side values (the configuration table,
the baseline name, the metric functions) are **all received as arguments** — so that the run-time
import stays one-way, `measure_quant` → this module, and this module never reads the module globals
that `--until` rewrites (type annotations only, under TYPE_CHECKING).
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

    #: Engine-side metric functions (passed in by the caller so that no reverse import is created).
    RelRms = Callable[[torch.Tensor, torch.Tensor], float]
    Psnr = Callable[[torch.Tensor, torch.Tensor, float], float]
    ToUint8 = Callable[[torch.Tensor], torch.Tensor]
    HistQuantile = Callable[[torch.Tensor, float], float]

    class PtGroupSummary(TypedDict):
        """A value of `pt.by_group` (the P̃ summary of one layer group)."""

        nodes: int
        median: float
        p99: float
        max: float

    class PtSummary(TypedDict):
        """The `pt` key of `build_summary` (the overall summary of the P̃ row peak/rms)."""

        rows: int
        nodes: int
        median: float
        p90: float
        p99: float
        max: float
        mean: float
        by_group: dict[str, PtGroupSummary]

    class ConfigSummary(TypedDict):
        """A value of `configs` in `build_summary` (the summary of one configuration)."""

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
        """The return value of `build_summary` (the common source of `attn.json` and report ⑥)."""

        pt: PtSummary
        configs: dict[str, ConfigSummary]
        gates: dict[str, str]
        #: Labels of the gates that were red (the source of truth for the verdict — the field that
        #: avoids sniffing it again from the display strings).
        failed: list[str]
        inject: str | None


#: Collapse the layer-name indices to group by role (`transformer_blocks.7.attn1.to_q` →
#: `transformer_blocks.*.…`)
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


#: Gates (design doc §5.3).
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
    """Machine-readable summary and gate verdicts (the common source of `attn.json` and report
    ⑥)."""
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
        """Record one gate (the verdict is **held as a boolean**; `verdict` is its display form).

        MUST: do not sniff the red ones again from the strings — the verdict is decided here once,
        and the caller (`measure_quant.main`) reads `failed` of {@link Summary}. The `NG` inside the
        strings has a different format per gate (leading / trailing / in parentheses), so picking it
        up afterwards misses some.
        """
        gates[label] = verdict
        if not ok:
            failed.append(label)

    pt_ok = pt["median"] <= PT_MEDIAN_GATE
    gate(
        f"P̃ row peak/rms median ≤ {PT_MEDIAN_GATE:g}",
        pt_ok,
        f"median {pt['median']:.2f} → {'OK' if pt_ok else 'NG'}"
        f"（p99 {pt['p99']:.2f} / max {pt['max']:.2f}）",
    )
    for name in attn_configs:
        # The gate of design doc §5.3 is "(c) the relative degradation when attention is added to
        # w8a8", so (g), which does not quantize the linear activations, is **outside the gate**
        # (reported as a reference value against `w8`).
        if run.diagnostics[name]["linear_act_i8"]:
            series = configs[name]["latent_ratio_vs_baseline"]
            ratio = max(series)
            ratio_ok = ratio <= LATENT_RATIO_GATE
            gate(
                f"`{name}` latent relRMS ≤ {LATENT_RATIO_GATE:g}× (vs {baseline})",
                ratio_ok,
                f"max {ratio:.3f}× (step {1 + series.index(ratio)}) → {'OK' if ratio_ok else 'NG'}",
            )
        else:
            series = configs[name]["latent_ratio_vs_w8"]
            ratio = max(series)
            gates[f"`{name}` latent relRMS (reference only, outside the gate)"] = (
                f"vs `w8`: max {ratio:.3f}×"
                f" (step {1 + series.index(ratio)})"
                " — the contribution of attention alone, with the linear activations in f32"
            )
    for name in config_names:
        diagnostic = run.diagnostics[name]
        if diagnostic["attn_calls_expected"]:
            ok = diagnostic["attn_calls"] == diagnostic["attn_calls_expected"]
            gate(
                f"`{name}` SDPA call count",
                ok,
                f"{diagnostic['attn_calls']} / {diagnostic['attn_calls_expected']}"
                f" → {'OK' if ok else 'NG'}",
            )
    for name in config_names:
        diagnostic = run.diagnostics[name]
        if not diagnostic["score_f16"]:
            continue
        # Guarding against a vacuous pass: "meant to round, but the identity" shows neither in the
        # numbers nor in the time (it is silent), so count directly the calls in which at least one
        # element changed across the rounding.
        fired, attempts = diagnostic["s16_fired_calls"], diagnostic["s16_rounded_calls"]
        ok = bool(attempts) and fired == attempts
        gate(
            f"`{name}` S f16 rounding fires on every call",
            ok,
            f"{fired} / {attempts} ({diagnostic['s16_elements']:,} elements,"
            f" max |S| {diagnostic['s16_abs_max']:.1f} < f16 limit 65,504)"
            f" → {'OK' if ok else 'NG'}",
        )
    for name in attn_configs:
        differs = not torch.equal(latents[f"{name}/{keys[-1]}"], latents[f"{baseline}/{keys[-1]}"])
        gate(
            f"`{name}` differs from `{baseline}`", differs, "OK" if differs else "NG (passthrough)"
        )

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
    lines.append(
        "# Q0 — quantization quality of the Anima DiT under w8 / w8a8 / attention a8"
        " (measured with torch on CPU)"
    )
    lines.append("")
    lines.append(
        f"- Date: {time.strftime('%Y-%m-%d')} / measured by:"
        " `tools/export-recipes/anima/measure_quant.py`"
    )
    lines.append(
        f"- Conditions: Anima turbo {args.steps} steps, {args.resolution}px, CFG={guidance:g}"
        f", LoRA merged, seed {SEED}, torch {torch.__version__} (CPU f32)"
    )
    lines.append(f"- Prompt: `{PROMPT}`")
    lines.append(
        "- Configurations: **(a) f32** = the plain weights with the LoRA merged / **(b) w8** ="
        " weight-only per-channel symmetric i8 fake-quant / **(c) w8a8** = (b) + per-token"
        " symmetric i8 fake-quant of the input activations of the eligible linears / **(d)–(h)** ="
        " (c) or (b) plus i8 attention / **(i)(j) `*-s16`** = additionally **the attention score S"
        " stored in f16** (option γ, phase 1)"
    )
    lines.append(
        "- The activation quantization mirrors design doc §4.2 (`s = clamp(rowmax|x|/127, tiny)`,"
        " `q = clamp(round(x/s), ±127)`, round half to even, `s = tiny` for an all-zero row)."
        ' Eligibility is the same as §4.1: "an `nn.Linear` inside the DiT with `k % 4 == 0`".'
    )
    lines.append(
        "- The attention granularity follows the MUST table of the attention a8 design doc §5.1"
        " (q/k = amax per row, P̃ = **fixed scale 1/127**, V = amax over the whole N per `(b,h,d)`,"
        " the half scale on the dequant side, the integer reductions exact in float64)."
    )
    lines.append(
        "- **Storing S in f16** (`*-s16`) matches the implementation of option γ design doc"
        " §3.4② / §7 phase 1 exactly in granularity: the half-scaled raw scores that ①QK writes are"
        " dropped to f16 and back, and **that is the only rounding**. ② the row max `m` and the"
        " denominator `l`, and ③'s `exp(S−m)` and `qP = round(127·exp(S−m))` are **all computed in"
        " f32 from the same rounded S**."
        " NOTE: `*-s16` is evaluated through the decomposed path, not the passthrough (torch SDPA),"
        " so the difference includes f32 differences of operation order (around 1e-7) — three"
        " orders of magnitude below f16's relative eps of 4.88e-4."
    )
    lines.append(
        "- **Everything outside the DiT (text_encoder / text_conditioner / VAE) is f32 in every"
        " configuration**, to narrow the comparison to the DiT's quantization (the f16 difference"
        " of the asset series lands identically on every configuration). The SDPA replacement of"
        " attention is also active only during the denoise loop; text encode / VAE decode are"
        " passed through."
    )
    lines.append("")

    lines.append("## ⓪ Configurations and call counts")
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
                "post-quantization sum" if diagnostic["stats_quantized"] else "f32 sum",
                f"{diagnostic['attn_calls']} / {diagnostic['attn_calls_expected']}",
                str(diagnostic["attn_nodes_seen"]),
                f"{diagnostic['elapsed']:.0f}s",
            ]
        )
    lines.append(
        _table(
            rows,
            [
                "configuration",
                "weights",
                "linear activations",
                "attention",
                "S storage",
                "softmax denominator",
                "SDPA calls / expected",
                "nodes",
                "elapsed",
            ],
        )
    )
    lines.append("")
    lines.append(
        f"Expected = {len(run.attention_nodes)} attention nodes × {args.steps} steps."
        ' Zero or a shortfall is the only detector of "meant to quantize, but passed through".'
    )
    lines.append("")

    lines.append("## ① Latent relRMS per step")
    lines.append("")
    lines.append(
        "`relRMS(v, r) = ‖v − r‖₂ / ‖r‖₂` (in parentheses: `−20·log₁₀(relRMS)` in dB)."
        f" The baseline of the attention configurations is **{baseline}** (the ratio to f32 is"
        " already saturated — design doc §5.3)."
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
    # When no attention configuration ran (a partial forward run with --until), leave the whole part
    # out.
    if attn_configs:
        lines.append(f"### Latent relRMS of the attention configurations (baseline = `{baseline}`)")
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
            "Ratio (final step, the relRMS against `f32` divided by that of `w8a8`): "
            + " / ".join(
                f"`{name}` {value['latent_ratio_final']:.2f}×"
                for name, value in summary["configs"].items()
                if name in attn_configs
            )
        )
        lines.append("")
    if "w8-qkpv" in summary["configs"]:
        lines.append(
            "**Only `w8-qkpv` has a different baseline** (its linear activations are f32, so it is"
            " naturally better than `w8a8`). Read the contribution of attention alone against"
            " `w8`: at the final step "
            f"vs `w8` relRMS {summary['configs']['w8-qkpv']['latent_rel_rms_vs_w8'][-1]:.4e}"
            ", ratio to the f32 relRMS of `w8` "
            f"{summary['configs']['w8-qkpv']['latent_ratio_vs_w8'][-1]:.2f}×."
        )
        lines.append("")

    lines.append("## ② Final image after VAE decode")
    lines.append("")
    lines.append(
        "PSNR is given two ways: on the raw `[-1,1]` tensor (data range 2.0) and after the same"
        " uint8 quantization as the PNG (data range 255). relRMS is on the raw tensor."
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
            ["pair", "PSNR f32 (dB)", "PSNR uint8 (dB)", "relRMS", "uint8 max diff (/255)"],
        )
    )
    lines.append("")
    lines.append(f"Images: {image_note} ({' / '.join(f'`image_{n}`' for n in config_names)})")
    lines.append("")

    lines.append("## ③ relRMS per layer (eligible linears)")
    lines.append("")
    lines.append(
        "`in relRMS` = the quantization error of the input activations (over all steps)."
        f" `out relRMS` = how far the layer's output moved under activation quantization"
        f" (measured at step {args.measure_step + 1} only; the weights are i8 on both sides, so it"
        " is the contribution of **activation quantization alone**)."
        " `peak` = the mean and max of the per-row `amax/rms` (an outlier indicator — "
        "`relRMS ≈ peak / (127·√12) = peak / 440` is the rule of thumb for uniform quantization)."
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
                "layer group",
                "count",
                "k",
                "in relRMS mean",
                "in relRMS max",
                "out relRMS max",
                "peak mean max",
                "peak max",
            ],
        )
    )
    lines.append("")
    lines.append("### Worst 20 individual layers (by in relRMS, descending)")
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
        _table(rows, ["layer", "k", "n", "in relRMS", "out relRMS", "peak mean", "peak max"])
    )
    lines.append("")
    lines.append(f"The raw table of all {len(stats)} layers is in `layers.csv`.")
    lines.append("")

    lines.append(
        "## ④ Distribution of the P̃ row peak/rms (the only quantity that sets the PV"
        " quantization SNR)"
    )
    lines.append("")
    lines.append(
        f"**The plain P̃** taken while `{baseline}` runs (attention still f32 = the values the"
        " quantizer actually sees). `peak` is structurally 1.0 from `exp(0)=1`, so"
        " `peak/rms = 1/rms`. The prediction of design doc §2.2 is"
        " `relRMS(P̃) ≈ (peak/rms)/(127·√12) = (peak/rms)/440`."
        f" Accumulated over all {args.steps} steps, all heads and all rows (quantiles interpolated"
        f" from a histogram of {pt_hist_bins} log₂-spaced bins — resolution 0.7%)."
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
                "layer group",
                "nodes",
                "rows",
                "median",
                "p90",
                "p99",
                "max",
                "predicted relRMS(median)",
                "predicted relRMS(p99)",
            ],
        )
    )
    lines.append("")
    overall = summary["pt"]
    lines.append(
        f"**Overall ({overall['rows']:,} rows): median {overall['median']:.2f} / "
        f"p90 {overall['p90']:.2f} / p99 {overall['p99']:.2f} / max {overall['max']:.2f} / "
        f"mean {overall['mean']:.2f}**"
    )
    lines.append("")
    lines.append("### Worst 12 individual nodes (by median, descending)")
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
            ["node", "median", "p99", "max", "mean"],
        )
    )
    lines.append("")

    lines.append("## ⑤ relRMS per attention layer (S and O)")
    lines.append("")
    lines.append(
        f"Measured at step {args.measure_step + 1} only. The reference is **the value computed"
        " from the same q/k/v without quantizing attention** (the activation quantization of the"
        " linears is already inside q/k/v, so this difference is the contribution of attention"
        " quantization alone)."
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
        lines.append(_table(rows, ["layer group", "count", "S mean", "S max", "O mean", "O max"]))
        lines.append("")
        lines.append(
            f"worst layer (O relRMS): `{worst_key}` = {measured[worst_key].o_rel_rms():.3e}"
            f" (S = {measured[worst_key].s_rel_rms():.3e})"
        )
        lines.append("")

    lines.append("## ⑥ Gate verdicts")
    lines.append("")
    lines.append(
        "The two gates of design doc §5.3: **① the median of the P̃ row peak/rms ≤ 16** (PV on the"
        " same footing as w8a8) / "
        f"**② latent relRMS ≤ 1.5× against `{baseline}`**."
        " The gate of `*-s16` is the same as ② (option γ design doc §6.3 — the baseline is the"
        f" accepted `{baseline}`, not f32). In addition it checks directly **whether the S f16"
        " rounding fired in every call** (guarding against a vacuous pass — an identity rounding"
        " shows neither in the numbers nor in the elapsed time)."
    )
    lines.append("")
    verdicts = summary["gates"]
    lines.append(
        _table(
            [[str(key), str(value)] for key, value in verdicts.items()],
            ["gate", "verdict"],
        )
    )
    lines.append("")
    return "\n".join(lines) + "\n"
