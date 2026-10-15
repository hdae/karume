"""DeBERTa w8a8 の崩壊上限（`e2e_deberta_w8a8_test.ts` の `COLLAPSE_CEILINGS`）を導き直す。

門の第 3 層は output.2〜24 の相対 RMS 誤差 ‖x − golden‖₂ / ‖golden‖₂ を出力ごとの
上限で見る。上限 = 出力ごと（観測点 = output.k）に、正当な標本（全ケース）の最悪 × 2 を
有効数字 2 桁で切り上げた値（ADR 0110 決定 5-3「観測点の実測 × 2 程度」・決定 4 の
MUST NOT〈実測の 5〜10 倍を E2E の帯に使わない〉）。ここは CPU の段だけを持つ。手順の全体
（GPU の dump と故障注入を含む順序）は ADR 0026 追記「導き直しの手順」が正本。

起動（tools/export-recipes から — transformers は一時足し）:

    HF_HUB_OFFLINE=1 ATEN_CPU_CAPABILITY=<avx512|avx2|default> \\
        uv run --with 'transformers==5.14.1' \\
        python -m deberta.collapse_ceilings mirror <作業 dir> <tag> <スレッド数>
    HF_HUB_OFFLINE=1 uv run --with 'transformers==5.14.1' \\
        python -m deberta.collapse_ceilings derive <作業 dir>

- `mirror`: torch 鏡像（export の `--act-quant` と同じ活性 i8 のフック）の全出力を
  `<作業 dir>/mirror_<tag>.pt` へ書く。命令セットは `ATEN_CPU_CAPABILITY` で選び、
  tag は `<isa>_t<スレッド数>`。
- `derive`: 作業 dir の標本から上限を導き `<作業 dir>/derive.json` を書く
  （`ceilings` がテストの表）。

標本（全て保存済み golden `io-i8a8.<case>` に対する相対 RMS 誤差・f64）:
  - gpu_a8          … GPU の a8 経路の dump（`gpu_a8.<case>.f32` —
                      `tools/diag/deberta-collapse.ts dump`）
  - cpu_<isa>_t<n>  … この機の CPU で採り直した鏡像の別標本 6 本（`mirror_<tag>.pt`）
  - emb_noise_s<k>  … 鏡像の埋め込みの出口に相対 ±2^-23 の乱数を入れた摂動アンサンブル
                      K=8（`derive` の中で回す）
参考（上限には入れない）: gpu_f32（`gpu_f32.<case>.f32` — f32 経路は別の関数。第 3 層では
区別できないことの記録）。
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from torch import nn

from karume import act_quant
from karume.quantize import fake_quant_int8

from . import export, patch

#: 鏡像 golden の置き場（`export --dtype i8 --act-quant` の 24 層の系列）。
SERIES = export.DEFAULT_OUT_ROOTS["i8"] / "full-24layer"
CASES = ("case0", "case1", "case2", "padded")
CPU_SAMPLES = ("avx512_t1", "avx512_t6", "avx2_t1", "avx2_t6", "default_t1", "default_t6")
K = 8
FIRST = 2  # output.2 以降が第 3 層
MULTIPLIER = 2
SIGNIFICANT = 2  # 上限の有効数字（切り上げ）
#: 上限に入れない標本（f32 経路は a8 の標本ではない）。
REFERENCE_ONLY = "gpu_f32"


def build() -> tuple[nn.Module, Mapping[str, export.InputArgs]]:
    """export と同じ経路で 24 層の i8 の wrapper（全 hidden_states を出す形）と入力を組む。"""
    from transformers import AutoTokenizer

    model = export.load_model(export.MODEL_ID, 24)
    patch.assert_supported(model.config)
    patch.apply_external_rel_pos_patch()
    tokenizer = AutoTokenizer.from_pretrained(export.MODEL_ID, revision=export.MODEL_REVISION)
    cases = export.build_cases(tokenizer, model)
    wrapper = export.HiddenStatesWrapper(model, single_output=False)
    fake_quant_int8(wrapper)
    return wrapper, dict(cases)


def run(wrapper: nn.Module, args: export.InputArgs) -> list[torch.Tensor]:
    with torch.no_grad():
        outs = wrapper(*(args[key] for key in export.INPUT_ORDER))
    return [o.detach().contiguous().clone() for o in outs]


def run_mirror(wrapper: nn.Module, args: export.InputArgs) -> list[torch.Tensor]:
    """活性 i8 のフック（export の `--act-quant` と同じ）を掛けて回す。フックは必ず外す。"""
    handles, _ = act_quant.attach_act_quant(wrapper)
    try:
        return run(wrapper, args)
    finally:
        act_quant.detach_act_quant(handles)


def rel_rms(x: np.ndarray, gold: np.ndarray) -> float:
    x = x.astype(np.float64).ravel()
    g = gold.astype(np.float64).ravel()
    return float(np.linalg.norm(x - g) / np.linalg.norm(g))


def nonfinite(x: np.ndarray) -> int:
    return int((~np.isfinite(x)).sum())


def ceil_significant(x: float, digits: int = SIGNIFICANT) -> float:
    """x を有効数字 digits 桁で切り上げる（10 進で丸める — 2 進の端数で 1 段ずれないため）。"""
    quantum = Decimal(1).scaleb(math.floor(math.log10(x)) - digits + 1)
    return float((Decimal(repr(x)) / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum)


def write_mirror(work: Path, tag: str, threads: int) -> None:
    torch.set_num_threads(threads)
    wrapper, cases = build()
    out = {
        case: torch.stack([m.reshape(-1) for m in run_mirror(wrapper, cases[case])])
        for case in CASES
    }
    torch.save(out, work / f"mirror_{tag}.pt")
    print(
        "saved", tag, torch.backends.cpu.get_cpu_capability(), torch.get_num_threads(), flush=True
    )


def derive(work: Path) -> None:
    torch.set_num_threads(6)
    wrapper, cases = build()
    cpu = {v: torch.load(work / f"mirror_{v}.pt", weights_only=False) for v in CPU_SAMPLES}
    out: dict = {"multiplier": MULTIPLIER, "first_output": FIRST, "cases": {}}
    for case in CASES:
        g = load_file(str(SERIES / f"io-i8a8.{case}.safetensors"))
        n_out = sum(1 for k in g if k.startswith("output."))
        gold = [g[f"output.{i}"] for i in range(n_out)]
        assert sum(nonfinite(x) for x in gold) == 0, case
        n = gold[0].size
        samples: dict[str, list[np.ndarray]] = {}
        for name in ("gpu_a8", REFERENCE_ONLY):
            a = np.fromfile(work / f"{name}.{case}.f32", dtype=np.float32)
            assert a.size == n * n_out, (name, case)
            samples[name] = [a[i * n : (i + 1) * n] for i in range(n_out)]
        for v, p in cpu.items():
            t = p[case]
            assert tuple(t.shape) == (n_out, n), (v, case, t.shape)
            samples[f"cpu_{v}"] = [t[i].numpy() for i in range(n_out)]
        args = cases[case]
        for seed in range(K):
            gen = torch.Generator().manual_seed(seed)

            def emb_hook(_m, _a, y, gen=gen):
                return y * (1 + (torch.rand(y.shape, generator=gen) - 0.5) * 2**-22)

            h = wrapper.model.embeddings.register_forward_hook(emb_hook)
            try:
                outs = run_mirror(wrapper, args)
            finally:
                h.remove()
            samples[f"emb_noise_s{seed}"] = [o.numpy().ravel() for o in outs]
        rows = {}
        for name, xs in samples.items():
            rr = [rel_rms(xs[i], gold[i]) for i in range(FIRST, n_out)]
            rows[name] = {
                "max": max(rr),
                "argmax_output": FIRST + int(np.argmax(rr)),
                "nonfinite": sum(nonfinite(x) for x in xs),
                "per_output": rr,
            }
        out["cases"][case] = rows
        print(
            case, {k: (round(v["max"], 4), v["argmax_output"]) for k, v in rows.items()}, flush=True
        )

    worst = max(
        (row["max"], case, name, row["argmax_output"])
        for case, rows in out["cases"].items()
        for name, row in rows.items()
        if name != REFERENCE_ONLY
    )
    per_output_worst = {}
    n_out = len(next(iter(out["cases"]["case0"].values()))["per_output"]) + FIRST
    for i in range(FIRST, n_out):
        per_output_worst[f"output.{i}"] = max(
            (rows[name]["per_output"][i - FIRST], case, name)
            for case, rows in out["cases"].items()
            for name in rows
            if name != REFERENCE_ONLY
        )
    group_worst = {}
    for group in ("gpu_a8", "cpu_", "emb_noise_", REFERENCE_ONLY):
        group_worst[group] = max(
            (row["max"], case, name)
            for case, rows in out["cases"].items()
            for name, row in rows.items()
            if name.startswith(group)
        )
    out["worst"] = {"relRms": worst[0], "case": worst[1], "sample": worst[2], "output": worst[3]}
    out["per_output_worst"] = per_output_worst
    out["ceilings"] = {k: ceil_significant(v[0] * MULTIPLIER) for k, v in per_output_worst.items()}
    out["group_worst"] = group_worst
    (work / "derive.json").write_text(json.dumps(out, indent=1))
    print("group worst", {k: (round(v[0], 4), v[1], v[2]) for k, v in group_worst.items()})
    print(
        "per-output worst", {k: (round(v[0], 4), v[1], v[2]) for k, v in per_output_worst.items()}
    )
    print("worst", out["worst"])
    print("ceilings(x2, ceil 2 sig)", out["ceilings"])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    mirror = sub.add_parser("mirror", help="CPU の鏡像の別標本を 1 本書く")
    mirror.add_argument("work", type=Path)
    mirror.add_argument("tag", help="<isa>_t<スレッド数>（例 avx512_t6）")
    mirror.add_argument("threads", type=int)
    derive_parser = sub.add_parser("derive", help="標本から上限を導いて derive.json を書く")
    derive_parser.add_argument("work", type=Path)
    args = parser.parse_args(argv)
    if not args.work.is_dir():
        raise SystemExit(f"作業 dir が無い: {args.work}")
    if args.command == "mirror":
        write_mirror(args.work, args.tag, args.threads)
    else:
        derive(args.work)


if __name__ == "__main__":
    main()
