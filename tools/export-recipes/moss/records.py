"""The record files of the MOSS stage-0 scripts (safetensors with one JSON metadata key) and the
facts recorded about each run (versions, threads, MKL/OpenMP environment, host RAM, time).

MUST: the metadata is one key holding JSON (sorted keys, no separator spaces). safetensors writes
its metadata through a hash map, and with several keys the byte order changes from process to
process (observed for the Wan text embeddings on 2026-10-02: three writes of the same input gave
two SHA-256s); with one key the same input gives the same bytes.

Where the records go (docs/assets-layout.md): the text-embedding asset is a graphless compile
product and goes to `outputs/series/` like Wan's; the stage-0 references are regenerable from the
pinned upstream, so they go to `outputs/bench/<distribution>/<date>_<purpose>/` (safe to delete),
as Wan's layer-wise DiT reference does.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import time
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from _shared.paths import BENCH_ROOT

#: The local distribution name the bench directories are keyed by (not published — ADR NNNN
#: decision 12; provisional until the distribution is named).
BENCH_NAME = "karume-moss-image"

#: The environment variables that steer the CPU kernels' threading and code paths.
_THREADING_ENV_PREFIXES = (
    "MKL_",
    "OMP_",
    "KMP_",
    "GOMP_",
    "ONEDNN_",
    "DNNL_",
    "OPENBLAS_",
    "ATEN_",
)


class RecordError(ValueError):
    """A record file does not have the agreed form."""


def default_reference_dir() -> Path:
    """`outputs/bench/karume-moss-image/<today>_stage0-reference/` — the stage-0 references."""
    return BENCH_ROOT / BENCH_NAME / f"{date.today().isoformat()}_stage0-reference"


def write_record(
    path: Path, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, Any], *, key: str
) -> None:
    """Write `tensors` and `metadata` (under the single key `key`) through a staging file."""
    payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".staging")
    # safetensors refuses tensors that share storage (a record names the same value twice, e.g. a
    # step's input latents and the previous step's output): give each such name its own copy.
    owned: dict[str, torch.Tensor] = {}
    storages: set[int] = set()
    for name, tensor in tensors.items():
        storage = tensor.untyped_storage().data_ptr()
        copy = storage in storages
        owned[name] = (tensor.detach().clone() if copy else tensor.detach()).contiguous()
        storages.add(storage)
    save_file(owned, str(staging), metadata={key: payload})
    staging.replace(path)


def read_record(path: Path, *, key: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read a record written by {@link write_record}; any other metadata layout fails loudly."""
    with safe_open(str(path), framework="pt") as handle:
        raw = handle.metadata() or {}
        if set(raw) != {key}:
            raise RecordError(f"{path}: metadata keys {sorted(raw)} are not the single key {key}")
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
    return tensors, json.loads(raw[key])


def bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Same dtype, shape and bit pattern (so -0.0 ≠ 0.0 and a NaN equals the same NaN)."""
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    return bool(
        torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    )


def peak_rss_gib() -> float:
    """The peak resident set of this process in GiB (Linux `ru_maxrss` is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def versions() -> dict[str, str]:
    """The versions of the libraries whose code produced the values."""
    import accelerate
    import diffusers
    import numpy
    import safetensors
    import transformers

    return {
        "accelerate": accelerate.__version__,
        "diffusers": diffusers.__version__,
        "numpy": numpy.__version__,
        "python": platform.python_version(),
        "safetensors": safetensors.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def environment() -> dict[str, Any]:
    """The CPU, the torch thread settings and the threading environment variables of this run."""
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "mkl": torch.backends.mkl.is_available(),
        "mkldnn": torch.backends.mkldnn.is_available(),
        "threading_env": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith(_THREADING_ENV_PREFIXES)
        },
    }


class Clock:
    """Wall time per named phase (seconds, rounded to 0.1)."""

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._last = self._started
        self.phases: dict[str, float] = {}

    def lap(self, name: str) -> float:
        """Close the phase `name` (time since the previous lap) and print it."""
        now = time.perf_counter()
        seconds = round(now - self._last, 1)
        self.phases[name] = seconds
        self._last = now
        print(f"[{name}] {seconds} s (peak RSS {peak_rss_gib():.2f} GiB)", flush=True)
        return seconds

    def total(self) -> float:
        return round(time.perf_counter() - self._started, 1)
