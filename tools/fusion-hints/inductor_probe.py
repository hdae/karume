"""fusion-hints `inductor` の実行体 — Inductor（torch.compile の CUDA バックエンド）が**どのノード列を
1 カーネルに畳むか**を取り出し、IR の op 名列に写して candidates.jsonl の未掴の鎖と突き合わせる。
Deno 側（tools/fusion-hints/main.ts `inductor`）が CUDA venv の python で subprocess として呼ぶ。

入力は exporter の golden 台帳（karume.goldens.GOLDEN_SPECS — 契約表の全 op を被覆する tiny モデル
31 本）と、この台本が持つ**鎖モデル**（実資産の候補表で上位に出る鎖を最小構成で組んだもの:
残差 rms_norm→add・linear→rms_norm→add・gelu_tanh·mul・RoPE 片・softmax 鎖・adaLN 変調）。tiny なので
Inductor の**構造**判断（pointwise / reduction をどこまで畳むか）までが射程で、形状依存の判断
（tiling・reduction の分割）は実資産の ExportedProgram が要る（2026-09-04 裁定 1-4 = この波では採らない）。

方法: 同じモジュールを 2 回 export し、片方は exporter と同じ手順（分解 → normalize → convert）で IR に、
片方は分解のまま `torch._inductor.compile` へ渡して Scheduler.codegen をすり替え、融合後のノード群
（FusedSchedulerNode / SchedulerNode / ExternKernelSchedulerNode）と各メンバーの fx ノード名を捕まえる。
IR ノードの出力名は fx ノード名（convert.py の out_name）なので、名前で join して IR の op 名列にする。
normalize が消したノード（RoPE 表の畳み込み等）は写像相手が無いので unmatched に数える。

使い方: python inductor_probe.py --out <dir> [--candidates <candidates.jsonl> ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch._inductor
from karume.convert import PRESERVED_OP_PREFIXES, convert, curated_decompositions
from karume.goldens import GOLDEN_SPECS, GoldenSpec, _rng
from karume.normalize import normalize_graph
from torch import nn
from torch._inductor.scheduler import Scheduler

# --- 実資産の候補表で上位に出る鎖の最小モデル（形は小さく・構造だけを写す） -----------------


class ResidualRmsNorm(nn.Module):
    """gemma4 の `rms_norm,add`（post-norm 残差）と `linear,rms_norm,add`。"""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 16, bias=False)
        self.weight = nn.Parameter(torch.ones(16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = torch.nn.functional.rms_norm(h, (16,), self.weight, 1e-6)
        return x + h


class GatedGelu(nn.Module):
    """gemma4 MLP の `gelu_tanh,mul`（gate · up）。"""

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(16, 32, bias=False)
        self.up = nn.Linear(16, 32, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.gate(x), approximate="tanh") * self.up(x)


class RopeHalf(nn.Module):
    """RoPE の半回転（slice / neg / cat / mul / add）— gemma4 decode で 35 本未掴の鎖。"""

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


class SoftmaxChain(nn.Module):
    """分解 attention の `bmm,softmax,bmm`（5 グラフ計 128 ブロック）。"""

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-1, -2)) * 0.25
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)


class AdalnModulation(nn.Module):
    """Irodori / anima DiT の adaLN 変調（silu → linear → slice → mul / add）。"""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 24, bias=True)
        self.weight = nn.Parameter(torch.ones(8))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        mod = self.proj(torch.nn.functional.silu(cond))
        shift, scale, gate = mod[..., :8], mod[..., 8:16], mod[..., 16:]
        h = torch.nn.functional.rms_norm(x, (8,), self.weight, 1e-6)
        return x + gate * (h * (1 + scale) + shift)


def _chain_specs() -> list[tuple[str, nn.Module, tuple[torch.Tensor, ...]]]:
    g = _rng()

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(shape, generator=g) * 2 - 1

    return [
        ("chain_residual_rms_norm", ResidualRmsNorm(), (rand(4, 16),)),
        ("chain_gated_gelu", GatedGelu(), (rand(4, 16),)),
        (
            "chain_rope_half",
            RopeHalf(),
            (rand(1, 2, 4, 8), rand(1, 1, 4, 8), rand(1, 1, 4, 8)),
        ),
        (
            "chain_softmax",
            SoftmaxChain(),
            (rand(1, 2, 4, 8), rand(1, 2, 4, 8), rand(1, 2, 4, 8)),
        ),
        ("chain_adaln", AdalnModulation(), (rand(4, 8), rand(4, 8))),
    ]


# --- Inductor の融合決定を捕まえる -------------------------------------------------------------


class _Captured(Exception):
    pass


def inductor_groups(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    dynamic_shapes: Any,
    preserved: Sequence[str],
) -> list[dict[str, Any]]:
    """分解済み EP を Inductor へ渡し、Scheduler の融合後ノード群を [{kind, members}] で返す。

    export は **CUDA 上で**行う（CPU で export した EP を .cuda() で動かすと定数の fake tensor が
    CPU のまま残り FakeTensorDeviceMismatchError になる — 実測）。
    """
    module = module.cuda()
    args = tuple(a.cuda() for a in args)
    # forward 内で作る定数（arange / ones）も CUDA へ寄せる（既定 device）。勾配は要らないので切る
    # （karume.gru_scan_* の custom op は autograd 式を持たない — 推論専用）。
    with torch.no_grad(), torch.device("cuda"):
        ep = torch.export.export(
            module, args, dynamic_shapes=dynamic_shapes, strict=False
        )
        decomposed = ep.run_decompositions(curated_decompositions(preserved))
        gm = decomposed.module()
    captured: dict[str, Any] = {}
    original = Scheduler.codegen

    def patched(self: Scheduler, *_a: Any, **_k: Any) -> None:
        captured["nodes"] = list(self.nodes)
        raise _Captured()

    Scheduler.codegen = patched  # type: ignore[method-assign]
    try:
        with torch.no_grad(), torch.device("cuda"):
            torch._inductor.compile(gm, list(args))
    except Exception as error:
        if "nodes" not in captured:
            raise
        del error
    finally:
        Scheduler.codegen = original  # type: ignore[method-assign]
        torch._dynamo.reset()
    groups: list[dict[str, Any]] = []
    for node in captured.get("nodes", []):
        try:
            members = [n.get_name() for n in node.get_nodes()]
        except (AttributeError, TypeError):
            members = [node.get_name()]
        origins: set[str] = set()
        inner_nodes = node.get_nodes() if hasattr(node, "get_nodes") else [node]
        for n in inner_nodes:
            inner = getattr(n, "node", None)
            for origin in getattr(inner, "origins", ()) or ():
                origins.add(origin.name)
        groups.append(
            {
                "kind": type(node).__name__,
                "members": members,
                "origins": sorted(origins),
            }
        )
    return groups


def ir_ops(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    dynamic_shapes: Any,
    symbol_names: Sequence[str],
    preserved: Sequence[str],
) -> dict[str, tuple[int, str]]:
    """exporter と同じ手順で IR にし、fx ノード名（= IR の出力名）→ (ノード順, op) を返す。"""
    ep = torch.export.export(module, args, dynamic_shapes=dynamic_shapes, strict=False)
    decomposed = ep.run_decompositions(curated_decompositions(preserved))
    normalize_graph(decomposed)
    graph, _ = convert(decomposed, symbol_names=symbol_names)
    return {node.outs[0]: (index, node.op) for index, node in enumerate(graph.nodes)}


def probe(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    dynamic_shapes: Any,
    symbol_names: Sequence[str],
    preserved: Sequence[str],
) -> list[dict[str, Any]]:
    module = module.eval()
    ops = ir_ops(module, args, dynamic_shapes, symbol_names, preserved)
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(
        inductor_groups(module, args, dynamic_shapes, preserved)
    ):
        # origins（分解前の fx 名）と members（Inductor 内部名）の両方で IR ノードを引く。
        names = set(group["members"]) | set(group["origins"])
        matched = sorted((ops[n] for n in names if n in ops), key=lambda pair: pair[0])
        rows.append(
            {
                "model": name,
                "group": index,
                "kind": group["kind"],
                "members": group["members"],
                "ir_ops": [op for _, op in matched],
                "unmatched": sorted(n for n in group["origins"] if n not in ops),
            }
        )
    return rows


def _contains(sequence: Sequence[str], chain: Sequence[str]) -> bool:
    n = len(chain)
    return any(
        list(sequence[i : i + n]) == list(chain) for i in range(len(sequence) - n + 1)
    )


def compare(
    rows: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """候補の op 名列ごとに fused（1 群に連続で入る）/ split（全 op は見えたが 1 群にならない）/
    unobserved（op が golden にも鎖モデルにも現れない）を付ける。"""
    fused_groups = [
        r for r in rows if r["kind"] == "FusedSchedulerNode" and len(r["ir_ops"]) >= 2
    ]
    seen_ops = {op for r in rows for op in r["ir_ops"]}
    results: list[dict[str, Any]] = []
    for cand in candidates:
        chain = [
            op for op in cand["ops"] if op != "reshape"
        ]  # reshape は 0 dispatch の別名化
        if len(chain) < 2:
            status = "trivial"
            witness = None
        elif any(op not in seen_ops for op in chain):
            status = "unobserved"
            witness = None
        else:
            hit = next((r for r in fused_groups if _contains(r["ir_ops"], chain)), None)
            status = "fused" if hit is not None else "split"
            witness = None if hit is None else f"{hit['model']}#{hit['group']}"
        results.append(
            {
                "source": cand.get("source"),
                "scenario": cand.get("scenario"),
                "component": cand.get("component"),
                "ops": cand["ops"],
                "count": cand.get("count"),
                "maximal": cand.get("maximal"),
                "inductor": status,
                "witness": witness,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidates", action="append", default=[])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA が使えない（Inductor の CUDA バックエンドが要る）", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    specs: list[
        tuple[
            str,
            Callable[
                [],
                tuple[
                    nn.Module,
                    tuple[torch.Tensor, ...],
                    Any,
                    Sequence[str],
                    Sequence[str],
                ],
            ],
        ]
    ] = []
    for spec in GOLDEN_SPECS:

        def build(
            spec: GoldenSpec = spec,
        ) -> tuple[
            nn.Module, tuple[torch.Tensor, ...], Any, Sequence[str], Sequence[str]
        ]:
            g = _rng()
            return (
                spec.build(g),
                spec.example_inputs(g),
                spec.dynamic_shapes,
                spec.symbol_names,
                spec.preserved,
            )

        specs.append((spec.name, build))
    for name, module, inputs in _chain_specs():
        specs.append(
            (
                name,
                lambda m=module, i=inputs: (m, i, None, ("T",), PRESERVED_OP_PREFIXES),
            )
        )

    for name, build in specs:
        try:
            module, inputs, dynamic, symbols, preserved = build()
            rows.extend(probe(name, module, inputs, dynamic, symbols, preserved))
            print(
                f"{name}: {sum(1 for r in rows if r['model'] == name)} groups",
                file=sys.stderr,
            )
        except Exception as error:  # noqa: BLE001 — golden 1 本の失敗で掃引を止めない（理由は failures に残す）
            failures[name] = f"{type(error).__name__}: {str(error)[:200]}"
            print(f"{name}: FAILED {failures[name]}", file=sys.stderr)

    (out / "inductor.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    candidates: list[dict[str, Any]] = []
    for path in args.candidates:
        for line in Path(path).read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") == "candidate":
                candidates.append(row)
    comparison = compare(rows, candidates)
    (out / "comparison.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in comparison)
    )
    tally: dict[str, int] = {}
    for r in comparison:
        tally[r["inductor"]] = tally.get(r["inductor"], 0) + 1
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "models": len(specs),
        "failed_models": failures,
        "groups": len(rows),
        "fused_groups": sum(1 for r in rows if r["kind"] == "FusedSchedulerNode"),
        "candidates": len(candidates),
        "verdicts": tally,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
