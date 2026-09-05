"""opbench `torch` の実行体 — `single` の 1 行（op / 形 / attrs / 格納）を PyTorch eager で組み、
CUDA 上の実時間を測る（列 B = torch が実際に速い dtype = f16 / bf16 の到達点。research 2026-09-03
recon の列定義）。Deno 側（tools/opbench/main.ts `torch`）が CUDA venv の python でこの台本を
subprocess として呼ぶ。exporter 本体（tools/exporter）には触らない — IR op → torch の逆写像は
この台本の手書き表 1 つ（ATEN_HANDLERS の逆写像は 56 本中 20 本しか機械化できない — recon）。

計測規約は bench.ts と同じ: heater（大きい f16 matmul）でクロックを張り付け、warmup → rounds 回の
min。列は f32（TF32 off）/ f32_tf32（matmul・cudnn とも on）/ f16 / bf16、`--compile` で
compile_f16（torch.compile・Inductor）を足す。TF32 の既定は matmul off / cudnn on の非対称なので
（実測）、両列を明示して測る。メモリは f16 列の max_memory_allocated の増分。

使い方: python torch_bench.py --single <single.jsonl> --out <dir> [--rounds 5] [--compile]
        [--limit n] [--op name ...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

TARGET_PASS_MS = 80.0
MAX_REPS = 1024
ROUNDS = 5

Case = dict[str, Any]
Build = Callable[[torch.dtype], Callable[[], torch.Tensor]]


def _shape(case: Case, slot: int) -> list[int]:
    return list(case["in_shapes"][slot])


def _act(shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    # shape が [] のスカラー入力（mul のスカラー相手）も randn(()) で組める。
    return torch.randn(tuple(shape), device="cuda", dtype=dtype) * 0.5


def _weight(shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    return (torch.randn(tuple(shape), device="cuda") * 0.05).to(dtype)


def _needs(case: Case, ins: int) -> None:
    if len(case["in_shapes"]) != ins:
        raise ValueError(
            f"{case['op']}: 入力 {len(case['in_shapes'])} 本（{ins} 本の契約）"
        )


def build_linear(case: Case) -> Build:
    _needs(case, 3)
    x_shape, w_shape, b_shape = (_shape(case, i) for i in range(3))

    def make(dtype: torch.dtype) -> Callable[[], torch.Tensor]:
        x, w, b = _act(x_shape, dtype), _weight(w_shape, dtype), _weight(b_shape, dtype)
        return lambda: F.linear(x, w, b)

    return make


def build_attention(case: Case) -> Build:
    _needs(case, 3)
    q_shape, k_shape, v_shape = (_shape(case, i) for i in range(3))
    scale = float(case["attrs"].get("scale", 1.0))
    gqa = q_shape[1] != k_shape[1]

    def make(dtype: torch.dtype) -> Callable[[], torch.Tensor]:
        q, k, v = _act(q_shape, dtype), _act(k_shape, dtype), _act(v_shape, dtype)
        return lambda: F.scaled_dot_product_attention(
            q, k, v, scale=scale, enable_gqa=gqa
        )

    return make


def build_rms_norm(case: Case) -> Build:
    _needs(case, 2)
    x_shape, w_shape = _shape(case, 0), _shape(case, 1)
    eps = float(case["attrs"].get("eps", 1e-6))

    def make(dtype: torch.dtype) -> Callable[[], torch.Tensor]:
        x, w = _act(x_shape, dtype), _weight(w_shape, dtype)
        return lambda: F.rms_norm(x, w_shape, w, eps)

    return make


def build_unary(fn: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[Case], Build]:
    def builder(case: Case) -> Build:
        _needs(case, 1)
        x_shape = _shape(case, 0)

        def make(dtype: torch.dtype) -> Callable[[], torch.Tensor]:
            x = _act(x_shape, dtype)
            return lambda: fn(x)

        return make

    return builder


def build_binary(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[Case], Build]:
    def builder(case: Case) -> Build:
        _needs(case, 2)
        a_shape, b_shape = _shape(case, 0), _shape(case, 1)

        def make(dtype: torch.dtype) -> Callable[[], torch.Tensor]:
            a = _act(a_shape, dtype)
            b = (
                _act(b_shape, dtype)
                if case["storage"][1] is None
                else _weight(b_shape, dtype)
            )
            return lambda: fn(a, b)

        return make

    return builder


def build_softmax(case: Case) -> Build:
    _needs(case, 1)
    x_shape = _shape(case, 0)
    dim = int(case["attrs"]["dim"])

    def make(dtype: torch.dtype) -> Callable[[], torch.Tensor]:
        x = _act(x_shape, dtype)
        return lambda: torch.softmax(x, dim)

    return make


def build_permute(case: Case) -> Build:
    _needs(case, 1)
    x_shape = _shape(case, 0)
    dims = [int(d) for d in case["attrs"]["dims"]]

    def make(dtype: torch.dtype) -> Callable[[], torch.Tensor]:
        x = _act(x_shape, dtype)
        return lambda: x.permute(*dims).contiguous()

    return make


def build_slice(case: Case) -> Build:
    _needs(case, 1)
    x_shape = _shape(case, 0)
    dim, start, end = (int(case["attrs"][k]) for k in ("dim", "start", "end"))

    def make(dtype: torch.dtype) -> Callable[[], torch.Tensor]:
        x = _act(x_shape, dtype)
        return lambda: x.narrow(dim, start, end - start).contiguous()

    return make


def build_cat(case: Case) -> Build:
    shapes = [_shape(case, i) for i in range(len(case["in_shapes"]))]
    dim = int(case["attrs"]["dim"])

    def make(dtype: torch.dtype) -> Callable[[], torch.Tensor]:
        xs = [_act(s, dtype) for s in shapes]
        return lambda: torch.cat(xs, dim)

    return make


def build_conv(nd: int) -> Callable[[Case], Build]:
    conv = F.conv1d if nd == 1 else F.conv2d

    def builder(case: Case) -> Build:
        if len(case["in_shapes"]) not in (2, 3):
            raise ValueError(f"conv{nd}d: 入力 {len(case['in_shapes'])} 本")
        x_shape, w_shape = _shape(case, 0), _shape(case, 1)
        b_shape = _shape(case, 2) if len(case["in_shapes"]) == 3 else None
        attrs = case["attrs"]

        def make(dtype: torch.dtype) -> Callable[[], torch.Tensor]:
            x, w = _act(x_shape, dtype), _weight(w_shape, dtype)
            b = None if b_shape is None else _weight(b_shape, dtype)
            return lambda: conv(
                x,
                w,
                b,
                stride=attrs.get("stride", 1),
                padding=attrs.get("padding", 0),
                dilation=attrs.get("dilation", 1),
                groups=int(attrs.get("groups", 1)),
            )

        return make

    return builder


def build_embedding(case: Case) -> Build:
    _needs(case, 2)
    ids_shape, table_shape = _shape(case, 0), _shape(case, 1)

    def make(dtype: torch.dtype) -> Callable[[], torch.Tensor]:
        ids = torch.randint(0, table_shape[0], ids_shape, device="cuda")
        table = _weight(table_shape, dtype)
        return lambda: F.embedding(ids, table)

    return make


def build_sum(case: Case) -> Build:
    _needs(case, 1)
    x_shape = _shape(case, 0)
    dim = int(case["attrs"]["dim"])

    def make(dtype: torch.dtype) -> Callable[[], torch.Tensor]:
        x = _act(x_shape, dtype)
        return lambda: x.sum(dim, keepdim=True)

    return make


# IR op → torch の手書き表（列 B の「同じ演算」）。無い op は skipped に載る。
BUILDERS: dict[str, Callable[[Case], Build]] = {
    "linear": build_linear,
    "attention": build_attention,
    "rms_norm": build_rms_norm,
    "softmax": build_softmax,
    "permute": build_permute,
    "slice": build_slice,
    "cat": build_cat,
    "conv1d": build_conv(1),
    "conv2d": build_conv(2),
    "embedding": build_embedding,
    "sum": build_sum,
    "bmm": build_binary(torch.matmul),
    "matmul": build_binary(torch.matmul),
    "add": build_binary(torch.add),
    "sub": build_binary(torch.sub),
    "mul": build_binary(torch.mul),
    "div": build_binary(torch.div),
    "gelu": build_unary(F.gelu),
    "gelu_tanh": build_unary(lambda x: F.gelu(x, approximate="tanh")),
    "silu": build_unary(F.silu),
    "tanh": build_unary(torch.tanh),
    "sigmoid": build_unary(torch.sigmoid),
    "neg": build_unary(torch.neg),
    "sqrt": build_unary(torch.sqrt),
    "exp": build_unary(torch.exp),
    "relu": build_unary(F.relu),
}

COLUMNS: list[tuple[str, torch.dtype, bool]] = [
    ("f32", torch.float32, False),
    ("f32_tf32", torch.float32, True),
    ("f16", torch.float16, False),
    ("bf16", torch.bfloat16, False),
]


def set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


class Heater:
    """クロック張り付けの filler（bench.ts の Heater と同じ役）: f16 matmul 2048×4096×4096 を 8 回。"""

    def __init__(self) -> None:
        self.a = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)
        self.b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

    def run(self) -> float:
        start = time.perf_counter()
        for _ in range(8):
            self.a @ self.b
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1e3

    def pin(self) -> None:
        total, best, stable = 0.0, math.inf, 0
        for runs in range(64):
            ms = self.run()
            total += ms
            stable = stable + 1 if ms <= best * 1.05 else 0
            best = min(best, ms)
            if runs + 1 >= 3 and total >= 500 and stable >= 3:
                return


def measure(
    fn: Callable[[], torch.Tensor], heater: Heater, rounds: int
) -> tuple[float, int]:
    """1 呼び出しの GPU 時間（ms・min）と反復数。反復は 1 パス ≈80ms まで積む（cuda Event）。"""
    fn()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    estimate = max(start.elapsed_time(end), 1e-3)
    reps = max(1, min(MAX_REPS, math.ceil(TARGET_PASS_MS / estimate)))
    heater.pin()
    best = math.inf
    for _ in range(rounds):
        heater.run()
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / reps)
    return best, reps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--op", action="append", default=[])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA が使えない（torch.cuda.is_available() = False）", file=sys.stderr)
        return 2

    cases: list[Case] = [
        json.loads(line) for line in Path(args.single).read_text().splitlines() if line
    ]
    if args.op:
        cases = [c for c in cases if c["op"] in args.op]
    if args.limit is not None:
        cases = cases[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    heater = Heater()
    records: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    # 列ごとの失敗を集計へ出す（残した errors が読まれないと、全 case 失敗した列が
    # median_ratio=null / weighted_ms=0 として「速かった」のと同じ見え方になる）。
    errors_by_column: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        builder = BUILDERS.get(case["op"])
        label = f"[{index + 1}/{len(cases)}] {case['component']} {case['op']} {json.dumps(case['in_shapes'])}"
        if builder is None:
            skipped[f"{case['op']}: 写像なし"] = (
                skipped.get(f"{case['op']}: 写像なし", 0) + 1
            )
            print(f"{label} skipped: 写像なし", file=sys.stderr)
            continue
        try:
            make = builder(case)
        except (ValueError, KeyError) as error:
            reason = f"{case['op']}: {error}"
            skipped[reason] = skipped.get(reason, 0) + 1
            print(f"{label} skipped: {error}", file=sys.stderr)
            continue
        record: dict[str, Any] = {
            "scenario": case["scenario"],
            "component": case["component"],
            "op": case["op"],
            "in_shapes": case["in_shapes"],
            "out_shapes": case["out_shapes"],
            "attrs": case["attrs"],
            "storage_signature": case["storage_signature"],
            "count": case["count"],
            "karume_ns_per_node_min": case.get("ns_per_node_min"),
            "karume_keys": case.get("keys", []),
            "ms": {},
            "reps": {},
            "mem_mib": {},
            "errors": {},
        }
        columns = list(COLUMNS)
        for name, dtype, tf32 in columns:
            set_tf32(tf32)
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            try:
                fn = make(dtype)
                ms, reps = measure(fn, heater, args.rounds)
                record["ms"][name] = ms
                record["reps"][name] = reps
                record["mem_mib"][name] = (
                    torch.cuda.max_memory_allocated() - base
                ) / 2**20
            except (RuntimeError, ValueError, TypeError, NotImplementedError) as error:
                # 列ごとに理由を残して次へ（1 列の失敗で行を捨てない）。torch の失敗は RuntimeError 系。
                record["errors"][name] = f"{type(error).__name__}: {str(error)[:160]}"
            finally:
                set_tf32(False)
                torch.cuda.empty_cache()
        if args.compile:
            try:
                fn = make(torch.float16)
                compiled = torch.compile(fn, dynamic=False)
                ms, reps = measure(compiled, heater, args.rounds)
                record["ms"]["compile_f16"] = ms
                record["reps"]["compile_f16"] = reps
            except (RuntimeError, ValueError, TypeError, NotImplementedError) as error:
                record["errors"]["compile_f16"] = (
                    f"{type(error).__name__}: {str(error)[:160]}"
                )
            finally:
                torch._dynamo.reset()
                torch.cuda.empty_cache()
        records.append(record)
        for column, message in record["errors"].items():
            entry = errors_by_column.setdefault(column, {"count": 0, "example": message})
            entry["count"] += 1
        shown = " ".join(f"{k}={v:.4f}" for k, v in record["ms"].items())
        failed = "".join(f" {column}=FAILED" for column in record["errors"])
        print(f"{label} {shown} ms{failed}", file=sys.stderr)

    (out / "torch.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "single": args.single,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "compile": args.compile,
        "rounds": args.rounds,
        "measured": len(records),
        "skipped": skipped,
        # 列名 → {count, example}（0 件の列は載らない）。Deno 側が comparison.json の頭へ写す。
        "errors": errors_by_column,
        "columns": [name for name, _, _ in COLUMNS]
        + (["compile_f16"] if args.compile else []),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
