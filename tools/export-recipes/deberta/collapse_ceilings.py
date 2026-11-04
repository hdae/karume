"""Re-derive the collapse ceilings of DeBERTa w8a8
(`COLLAPSE_CEILINGS` in `packages/models/tests/helpers/collapse-ceilings.ts`).

The third tier of the check compares the relative RMS error ‖x − golden‖₂ / ‖golden‖₂ of
output.2 to output.24 against a per-output ceiling. Ceiling = per output (observation point =
output.k), the worst value over the legitimate samples (all cases) × 2, rounded up to two
significant digits (ADR 0110 decision 5-3 "about 2 × the measurement at the observation point";
the MUST NOT of decision 4: do not use 5 to 10 × the measurement as an E2E band). This script
holds only the CPU steps. The whole procedure (including the order of the GPU dump and the fault
injection) is canonical in the ADR 0026 addendum on the re-derivation procedure.

Run (from tools/export-recipes — transformers is added temporarily):

    HF_HUB_OFFLINE=1 ATEN_CPU_CAPABILITY=<avx512|avx2|default> \\
        uv run --with 'transformers==5.14.1' \\
        python -m deberta.collapse_ceilings mirror <work dir> <tag> <threads>
    HF_HUB_OFFLINE=1 uv run --with 'transformers==5.14.1' \\
        python -m deberta.collapse_ceilings derive <work dir>

- `mirror`: write all outputs of the torch mirror (the same activation-i8 hooks as export's
  `--act-quant`) to `<work dir>/mirror_<tag>.pt`. The instruction set is chosen with
  `ATEN_CPU_CAPABILITY`, and the tag is `<isa>_t<threads>`.
- `derive`: derive the ceilings from the samples in the work dir and write
  `<work dir>/derive.json` (`ceilings` is the table of the test).

Samples (all are relative RMS errors against the saved golden `io-i8a8.<case>`, in f64):
  - gpu_a8          … dump of the GPU's a8 path (`gpu_a8.<case>.f32` —
                      `tools/diag/deberta-collapse.ts dump`)
  - cpu_<isa>_t<n>  … 6 further samples of the mirror re-taken on this machine's CPU
                      (`mirror_<tag>.pt`)
  - emb_noise_s<k>  … perturbation ensemble with relative ±2^-23 random noise at the exit of the
                      mirror's embedding, K=8 (run inside `derive`)
For reference (not used for the ceilings): gpu_f32 (`gpu_f32.<case>.f32` — the f32 path is a
different function; a record that the third tier cannot tell the two apart).
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

#: Where the mirror goldens live (the 24-layer series of `export --dtype i8 --act-quant`).
SERIES = export.DEFAULT_OUT_ROOTS["i8"] / "full-24layer"
CASES = ("case0", "case1", "case2", "padded")
CPU_SAMPLES = ("avx512_t1", "avx512_t6", "avx2_t1", "avx2_t6", "default_t1", "default_t6")
K = 8
FIRST = 2  # output.2 onwards is the third tier
MULTIPLIER = 2
SIGNIFICANT = 2  # significant digits of a ceiling (rounded up)
#: The sample left out of the ceilings (the f32 path is not a sample of a8).
REFERENCE_ONLY = "gpu_f32"


def build() -> tuple[nn.Module, Mapping[str, export.InputArgs]]:
    """Build the 24-layer i8 wrapper (the form that outputs every hidden_states) and the inputs on
    the same path as export."""
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
    """Run with the activation-i8 hooks (the same as export's `--act-quant`). The hooks are always
    removed."""
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
    """Round x up to `digits` significant digits (rounded in decimal — so a binary fraction does
    not push it one step up)."""
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
    mirror = sub.add_parser("mirror", help="write one further sample of the CPU mirror")
    mirror.add_argument("work", type=Path)
    mirror.add_argument("tag", help="<isa>_t<threads> (e.g. avx512_t6)")
    mirror.add_argument("threads", type=int)
    derive_parser = sub.add_parser(
        "derive", help="derive the ceilings from the samples and write derive.json"
    )
    derive_parser.add_argument("work", type=Path)
    args = parser.parse_args(argv)
    if not args.work.is_dir():
        raise SystemExit(f"work dir does not exist: {args.work}")
    if args.command == "mirror":
        write_mirror(args.work, args.tag, args.threads)
    else:
        derive(args.work)


if __name__ == "__main__":
    main()
