"""Writes the CPU fixture of the runtime op `gated_delta_scan` (ADR 0135 decision 6): seeded inputs,
the outputs of upstream's two f32 functions, the f64 oracle and the upstream errors against it.

    cd tools/export-recipes && uv run --with 'transformers==5.14.1' --inexact \\
        python -m _shared.gated_delta_fixture

Output: `packages/runtime/tests/fixtures/gated-delta-scan.safetensors` (read by
`packages/runtime/tests/helpers/gated-delta-fixture.ts`). Every tensor is F32 (the runtime's
safetensors reader has no F64); the metadata key `cases` holds one JSON record per case.

**Inputs.** They are slices of the draws of the 2026-10-27 spike (docs/research/
2026-10-27-gdn-scan-spike.md §1), so the spike's recorded GPU outputs cover these cases too: one
MOSS text-encoder GDN layer at T = 4096 (16 key heads, 32 value heads, head dim 128), q / k / v =
silu(N(0,1)), a / b ~ N(0,1) from `Generator(20261027 + 4096)`, `A_log = log U(0,16)` from
`Generator(20261027)`, `dt_bias = 1`, `beta = sigmoid(b)`, `g = −exp(A_log)·softplus(a + dt_bias)`
(regime `init`, fast decay) or that × 0.001 (regime `long`, long memory). A case keeps a token
window and a subset of heads — the recurrence is causal and per head, so a prefix of the draw with
fewer heads is itself a valid case (ADR 0135 open point 7: tracked fixtures stay small). Value
heads 2j and 2j + 1 read key head j (r = 2), as in the layer.

**Upstream outputs.** `torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule` of
transformers 5.14.1, called as `Qwen3_5GatedDeltaNet` calls them: raw q / k repeated to the value
heads, `use_qk_l2norm_in_kernel=True`, zero initial state. The kernel inputs `q` / `k` are
upstream's own `l2norm` of the raw q / k (MUST: the generator checks that normalising before and
after the repeat gives the same bits, so upstream and the op consume the same normalised values).

**Oracle.** The recurrence in f64 on exactly the f32 inputs the op consumes. The band of a case is
twice the larger of the two upstream functions' errors against it (max_abs and rel_l2 of o — the
value form exposes only o).

MUST: the expected values are upstream's functions run here, not a re-derivation (a formula written
on this side would be compared with a formula written on the TS side — a tautology). transformers
is imported inside the function (the optional-group import gate —
tests/test_optional_group_imports.py).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812  -- the spelling of upstream's module
from safetensors.torch import save_file

from _shared.paths import REPO_ROOT

DEFAULT_OUT = (
    REPO_ROOT / "packages" / "runtime" / "tests" / "fixtures" / "gated-delta-scan.safetensors"
)

#: The spike's draw (research 2026-10-27 §1): seed, length, heads, head dim, decay scales.
SEED = 20261027
DRAW_T = 4096
DRAW_KEY_HEADS = 16
DRAW_VALUE_HEADS = 32
HEAD_DIM = 128
REGIMES = {"init": 1.0, "long": 0.001}
#: Elements sampled from the oracle into the metadata of a case without a stored oracle.
ORACLE_SAMPLES = 16


@dataclass(frozen=True)
class Window:
    """One batch element of a case: a token window of one regime's draw."""

    regime: str
    start: int


@dataclass(frozen=True)
class Case:
    name: str
    #: One window per batch element (the same length T each).
    windows: tuple[Window, ...]
    time: int
    #: Key heads taken from the draw; value heads are 2j and 2j + 1 for each key head j unless
    #: `value_heads` says otherwise.
    key_heads: tuple[int, ...]
    value_heads: tuple[int, ...]
    #: Store the upstream outputs and the oracle (as an f32 head + tail pair) in the file.
    store_outputs: bool


CASES: tuple[Case, ...] = (
    # B = 2 with different windows and regimes per batch element (a B-index mix-up shows in the
    # values), Hk = 2 / Hv = 4 (r = 2 — the only case where a key-head fault `h % Hk` can act).
    # The one case that stores the upstream outputs and the oracle, for the metric cross-check.
    Case(
        name="mixed_B2_T16",
        windows=(Window("init", 0), Window("long", 16)),
        time=16,
        key_heads=(0, 1),
        value_heads=(0, 1, 2, 3),
        store_outputs=True,
    ),
    # Fast decay (g down to about −8 here): the band is set by the chunked function's error.
    Case(
        name="init_T64",
        windows=(Window("init", 0),),
        time=64,
        key_heads=(0,),
        value_heads=(0, 1),
        store_outputs=False,
    ),
    # The default lane's long-memory case with T ≥ 512 (ADR 0135 decision 6 MUST): one head keeps
    # the file small (the recurrence is per head). On the spike's recorded GPU words this head
    # fails the band with the built-in exp and passes with the accurate one.
    Case(
        name="long_T512",
        windows=(Window("long", 0),),
        time=512,
        key_heads=(0,),
        value_heads=(0,),
        store_outputs=False,
    ),
)


def _draw() -> dict[str, torch.Tensor]:
    """The spike's T = 4096 draw (`gen_inputs.py`'s `draw(4096)` and `A_log`), raw."""
    a_log = torch.log(
        torch.empty(DRAW_VALUE_HEADS).uniform_(0, 16, generator=torch.Generator().manual_seed(SEED))
    )
    generator = torch.Generator().manual_seed(SEED + DRAW_T)

    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    q = F.silu(normal(1, DRAW_T, DRAW_KEY_HEADS, HEAD_DIM))
    k = F.silu(normal(1, DRAW_T, DRAW_KEY_HEADS, HEAD_DIM))
    v = F.silu(normal(1, DRAW_T, DRAW_VALUE_HEADS, HEAD_DIM))
    return {
        "q": q,
        "k": k,
        "v": v,
        "a": normal(1, DRAW_T, DRAW_VALUE_HEADS),
        "b": normal(1, DRAW_T, DRAW_VALUE_HEADS),
        "a_log": a_log,
    }


def _case_inputs(case: Case, draw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Raw q / k, v, g and beta of one case, batch elements stacked on axis 0."""
    keys = list(case.key_heads)
    values = list(case.value_heads)
    dt_bias = torch.ones(DRAW_VALUE_HEADS)
    parts: dict[str, list[torch.Tensor]] = {"q": [], "k": [], "v": [], "g": [], "beta": []}
    for window in case.windows:
        tokens = slice(window.start, window.start + case.time)
        g = -draw["a_log"].float().exp() * F.softplus(draw["a"].float() + dt_bias)
        g = g * REGIMES[window.regime]
        parts["q"].append(draw["q"][:, tokens][:, :, keys])
        parts["k"].append(draw["k"][:, tokens][:, :, keys])
        parts["v"].append(draw["v"][:, tokens][:, :, values])
        parts["g"].append(g[:, tokens][:, :, values])
        parts["beta"].append(draw["b"].sigmoid()[:, tokens][:, :, values])
    return {name: torch.cat(tensors, dim=0).contiguous() for name, tensors in parts.items()}


def _oracle(q: torch.Tensor, k: torch.Tensor, v, g, beta) -> torch.Tensor:
    """The recurrence in f64 on [B,T,H,d] inputs (q / k normalised and already repeated)."""
    q, k, v, g, beta = (x.transpose(1, 2).double() for x in (q, k, v, g, beta))
    batch, heads, time, key_dim = k.shape
    q = q * (key_dim**-0.5)
    state = torch.zeros(batch, heads, key_dim, v.shape[-1], dtype=torch.float64)
    out = torch.empty(batch, heads, time, v.shape[-1], dtype=torch.float64)
    for t in range(time):
        state = state * g[:, :, t].exp()[..., None, None]
        predicted = (state * k[:, :, t, :, None]).sum(-2)
        delta = (v[:, :, t] - predicted) * beta[:, :, t, None]
        state = state + k[:, :, t, :, None] * delta[:, :, None, :]
        out[:, :, t] = (state * q[:, :, t, :, None]).sum(-2)
    return out.transpose(1, 2).contiguous()


def _metrics(x: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """max_abs, rel_l2 and max_rel_head (over value heads, axis 2 of o) against the oracle."""
    diff = (x.double() - reference).abs()
    heads = reference.shape[2]
    head_error = diff.movedim(2, 0).reshape(heads, -1).max(dim=1).values
    head_reference = reference.abs().movedim(2, 0).reshape(heads, -1).max(dim=1).values
    return {
        "max_abs": diff.max().item(),
        "rel_l2": (diff.norm() / reference.norm()).item(),
        "max_rel_head": (head_error / head_reference.clamp_min(1e-300)).max().item(),
        "finite": bool(torch.isfinite(x).all().item()),
    }


def _fingerprint(oracle: torch.Tensor) -> dict[str, Any]:
    """Sums and evenly spaced samples of the oracle (exact f64 values — JSON round-trips them)."""
    flat = oracle.reshape(-1)
    count = flat.numel()
    indices = sorted({round(i * (count - 1) / (ORACLE_SAMPLES - 1)) for i in range(ORACLE_SAMPLES)})
    indices.append(int(flat.abs().argmax().item()))
    return {
        "sum": flat.sum().item(),
        "sum_squares": (flat * flat).sum().item(),
        "samples": [[index, flat[index].item()] for index in indices],
    }


def build(cases: Sequence[Case] = CASES) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """The tensors and the metadata of the fixture (no I/O)."""
    import transformers
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        l2norm,
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )

    draw = _draw()
    tensors: dict[str, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    for case in cases:
        raw = _case_inputs(case, draw)
        ratio = len(case.value_heads) // len(case.key_heads)
        q = l2norm(raw["q"], dim=-1, eps=1e-6)
        k = l2norm(raw["k"], dim=-1, eps=1e-6)
        q_raw = raw["q"].repeat_interleave(ratio, dim=2)
        k_raw = raw["k"].repeat_interleave(ratio, dim=2)
        # MUST: upstream normalises after the repeat; the op consumes the normalised key heads
        # before it. The two must be the same bits, or upstream and the op see different inputs.
        if not (
            torch.equal(l2norm(q_raw, dim=-1, eps=1e-6), q.repeat_interleave(ratio, dim=2))
            and torch.equal(l2norm(k_raw, dim=-1, eps=1e-6), k.repeat_interleave(ratio, dim=2))
        ):
            raise AssertionError(f"{case.name}: l2norm before and after the repeat differ in bits")
        chunk, _ = torch_chunk_gated_delta_rule(
            q_raw,
            k_raw,
            raw["v"],
            g=raw["g"],
            beta=raw["beta"],
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        recurrent, _ = torch_recurrent_gated_delta_rule(
            q_raw,
            k_raw,
            raw["v"],
            raw["g"],
            raw["beta"],
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        oracle = _oracle(
            q.repeat_interleave(ratio, dim=2),
            k.repeat_interleave(ratio, dim=2),
            raw["v"],
            raw["g"],
            raw["beta"],
        )
        upstream = {"chunk": _metrics(chunk, oracle), "recurrent": _metrics(recurrent, oracle)}
        band = {
            metric: 2 * max(upstream["chunk"][metric], upstream["recurrent"][metric])
            for metric in ("max_abs", "rel_l2")
        }
        for name, tensor in (
            ("q", q),
            ("k", k),
            ("v", raw["v"]),
            ("g", raw["g"]),
            ("beta", raw["beta"]),
        ):
            tensors[f"{case.name}.{name}"] = tensor.float().contiguous()
        if case.store_outputs:
            head = oracle.float()
            tensors[f"{case.name}.o_chunk"] = chunk.contiguous()
            tensors[f"{case.name}.o_recurrent"] = recurrent.contiguous()
            tensors[f"{case.name}.o_oracle_head"] = head.contiguous()
            tensors[f"{case.name}.o_oracle_tail"] = (oracle - head.double()).float().contiguous()
        batch, time, key_heads, key_dim = q.shape
        records.append(
            {
                "name": case.name,
                "shape": {
                    "B": batch,
                    "T": time,
                    "Hk": key_heads,
                    "Hv": raw["v"].shape[2],
                    "dk": key_dim,
                    "dv": raw["v"].shape[3],
                },
                "windows": [{"regime": w.regime, "start": w.start} for w in case.windows],
                "draw_key_heads": list(case.key_heads),
                "draw_value_heads": list(case.value_heads),
                "g_range": [raw["g"].min().item(), raw["g"].max().item()],
                "upstream": upstream,
                "band": band,
                "oracle": _fingerprint(oracle),
                "stored_outputs": case.store_outputs,
            }
        )
    metadata = {
        "cases": json.dumps(records),
        "source": json.dumps(
            {
                "generator": "tools/export-recipes/_shared/gated_delta_fixture.py",
                "transformers": transformers.__version__,
                "torch": torch.__version__,
                "draw": {
                    "seed_a_log": SEED,
                    "seed_activations": SEED + DRAW_T,
                    "T": DRAW_T,
                    "Hk": DRAW_KEY_HEADS,
                    "Hv": DRAW_VALUE_HEADS,
                    "head_dim": HEAD_DIM,
                    "decay_scales": REGIMES,
                },
            }
        ),
    }
    return tensors, metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    tensors, metadata = build()
    save_file(tensors, str(args.out), metadata=metadata)
    for record in json.loads(metadata["cases"]):
        band = record["band"]
        print(
            f"{record['name']}: {record['shape']} band o max_abs {band['max_abs']:.3e}"
            f" rel_l2 {band['rel_l2']:.3e}"
        )
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
