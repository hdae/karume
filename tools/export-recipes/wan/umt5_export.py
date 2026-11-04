"""The writer that turns the umT5-XXL encoder Wan2.1 uses (`UMT5EncoderModel`) into the material
of an i8 dynamic-shape container (ADR 0119 step 10b — decisions 5 / 6; grown in step 10c).

The upstream is the pin of the original `google/umt5-xxl` (`wan.sources.UMT5_SOURCES` — the split
pickle form, read after the sha256 check; ADR 0122 decisions 1 / 2). There are two axes
(decision 3): `--upstream` (the umT5 upstream — weights, config, provenance) and `--model` (the Wan
text-side snapshot — the id sequences of the `reference` cases and `check-mask`).

    uv run --group wan --inexact python -m wan.sources --umt5 xxl --fetch  # original encoder shards
    uv run --group wan --inexact python -m wan.umt5_export prepare       # material only (no write)
    uv run --group wan --inexact python -m wan.umt5_export write --check # write, check vs series
    uv run --group wan --inexact python -m wan.umt5_export write         # rewrite series container
    uv run --group wan --inexact python -m wan.umt5_export check-mask --dtype bf16 --out <dir>
    uv run --group wan --inexact python -m wan.umt5_export check-mask --dtype f32 --out <dir>
    uv run --group wan --inexact python -m wan.umt5_export compare-mask --out <dir>
    uv run --group wan --inexact python -m wan.umt5_export reference     # layered ref → golden
    uv run --group wan --inexact python -m wan.umt5_export write --intake inputs/umt5/<name>
    uv run --group wan --inexact python -m wan.umt5_export reference --intake inputs/umt5/<name>

`--intake` replaces the umT5 upstream axis with the intake of a third-party compatible encoder
(`wan.umt5_intake` — ADR 0122 decision 5). The weights, config and provenance come from the intake
record `intake.json`, and the series is `umt5-xxl-<name>-i8-dyn`. Only the source dtype the record
declares (BF16 / F32) is accepted, and BF16 is widened to F32 when read ({@link Checkpoint}). The
path that reads from the upstream table (`--upstream`) stays F32 only. `write` of an intake with an
undeclared license needs an explicit `--allow-undeclared-license` (decision 6 — the container's
license field records the undeclared mark).

Every subcommand prints the wall time and RSS per stage ({@link MemoryMonitor}) as JSON. The
numbers are recorded in research `2026-10-03-umt5-export-ram`. `reference` (step 10c) reads the
weights from the written container, takes the layer-by-layer CPU reference and writes the goldens
to the same directory (no container is written — the contents are in
{@link wan.umt5_reference}). Acceptance cases also get the unquantized reference (the checkpoint's
F32, one layer at a time — {@link checkpoint_weights}).

## The form of the export (decision 6 — the form chosen from the step 10b measurements)

The straightforward form — read every weight in f32, round, then export — is not expected to fit
in a 31 GiB machine: the f32 weights are 21.2 GiB in all, and the fake-quant of the token embedding
(`[256384, 4096]` f32 = 3.91 GiB) and emit's i8 conversion each add 2.0× and 2.25× of it
temporarily (measured) (research §2). So it is split three ways:

1. **The trace runs on an upstream model with no weights (meta)** ({@link meta_text_encoder} →
   {@link trace}). The export only follows the shapes, so the graph is identical in JSON to an
   export with real weights (pinned by pytest on a small model).
2. **The quantization targets (linear and token embedding) are read from the checkpoint in chunks
   of rows**, turned into packed values and scales with the same procedure as the exporter's i8,
   and passed through `fixed_weights` (the entry point of ADR 0097) ({@link quantize_rows}). The
   per-channel scale is closed within each row (axis 0 for both linear and embedding), so
   splitting into chunks is bit-exact with running the whole tensor at once.
3. **The unquantized weights** (the RMSNorm weights, the 24 relative-position tables) are passed as
   the checkpoint's f32 as is. A run that uses `fixed_weights` stores f32 by default, so the
   explicit F32 for the tables (`storage_overrides`) is not needed (emit refuses a mix of fixed and
   automatic quantization).

The container's bytes match the straightforward form (`fake_quant_i8` in f32 →
`weight_dtype="i8"` + the explicit F32 for the tables) (pinned by pytest on a small model, by byte
equality of every part of the container). No change to the exporter core is needed.

## Where the container is written

The container is written by {@link write_container} with
`publish_model(..., graph_name=GRAPH_NAME)`. The component name {@link GRAPH_NAME} is the weights
key of the umT5 distribution (`wan.umt5_distribution.UMT5_WEIGHTS`) and of the Wan distribution
(`wan.distribution.WAN_WEIGHTS` — a cross-repository reference), and the cross-family check
(`tests/test_graph_names.py`) checks it matches both (ADR 0119 addendum "design of step 10d" D).

`write --check` puts the container
{@link write_container} wrote in a scratch directory and checks it against the series container by
the sha256 of every part ({@link assert_same_container} — the series is not replaced). The golden
(`reference.*`) meta holds the sha256 of part 0, so after `write` rewrites the container,
`reference` rewrites the goldens too.

MUST: transformers is imported inside functions (the `wan` group is not in the default sync —
`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import tempfile
import threading
import time
import weakref
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from _shared.paths import SERIES_ROOT
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.dist import NOTICE_FILENAME
from karume.emit import FixedQuantizedWeight
from karume.ir import IrGraph
from karume.pipeline import export_module, publish_model
from karume.quantize import QUANT_MODULE_TYPES, channel_scale, iter_quant_targets, quantize_to_int8
from wan import umt5_patch, umt5_reference
from wan.sources import (
    DEFAULT_MODEL,
    SOURCES,
    UMT5_BIN_INDEX,
    UMT5_SOURCES,
    WAN21_MODELS,
    WanSourceError,
    pinned_umt5_shards,
    read_safetensors_header,
    text_snapshot,
    umt5_snapshot,
)
from wan.umt5_distribution import UMT5_DEFAULT_MODEL, intake_series_name, storage_kind
from wan.umt5_intake import (
    ALLOW_UNDECLARED_LICENSE_FLAG,
    INTAKE_DTYPES,
    Umt5Intake,
    Umt5IntakeError,
    assert_license_intent,
    assert_only_intake_files,
    load_intake,
)

#: The i8 series (spelled by the distribution convention `<name>-<storage>-dyn`; the name is the
#: provenance — the encoder of the original `google/umt5-xxl`, ADR 0122 decision 4).
SERIES_NAME = "umt5-xxl-i8-dyn"
SERIES = SERIES_ROOT / SERIES_NAME

#: Location inside the series (spelled like the component name {@link GRAPH_NAME} — a convention,
#: not a derivation).
COMPONENT_DIR = "text_encoder"
MODEL_FILE = "model.krm"

#: The container's graph name = the distribution's component name = the weights key
#: (container-v1 §2.1 — the distribution side spells it `wan.umt5_distribution.UMT5_ROLE`; the
#: match is the check in `tests/test_graph_names.py`).
GRAPH_NAME = "text_encoder"

#: The umT5 directory inside the Wan snapshot (read only by `check-mask` — a control that
#: reproduces the path of the precomputed assets, not the container's upstream — ADR 0122
#: decision 3).
WAN_TEXT_ENCODER_SUBFOLDER = "text_encoder"

#: Valid length of the trace's example input (the same as the preliminary check's
#: `umt5_probe.EXPORT_LENGTHS[0]` — L is a symbolic dimension, so any value that does not trigger
#: shape specialization will do: avoid 0 / 1 and heads 64, d_kv 64, 32 buckets).
TRACE_LENGTH = 28

#: Rows read at once during quantization (256 MiB for `[rows, 4096]` f32). The temporary
#: allocations are a few times a chunk, avoiding the 2.25× (about 8.8 GiB) of running the whole
#: token embedding (3.91 GiB) at once.
CHUNK_ROWS = 16_384

#: File names of the checkpoint index (split form) and the single form (the spelling of
#: transformers' `save_pretrained`).
CHECKPOINT_INDEX = "model.safetensors.index.json"
CHECKPOINT_SINGLE = "model.safetensors"

#: KiB → GiB (the `/proc` values and Linux's `ru_maxrss` are in KiB).
_KIB_PER_GIB = 1 << 20


class Umt5ExportError(RuntimeError):
    """The checkpoint, the graph or the quantization targets differ from what is assumed
    (decision 5)."""


# ---------------------------------------------------------------------------
# RSS measurement
# ---------------------------------------------------------------------------

_STATUS_FIELDS = ("VmRSS", "VmHWM", "RssAnon", "RssFile")
_MEMINFO_FIELDS = ("MemAvailable", "SwapFree")


def _read_kib(path: str, fields: Sequence[str]) -> dict[str, int]:
    """Read fields from a `/proc` file in the `Key: <n> kB` format (a missing field fails
    loudly)."""
    values: dict[str, int] = {}
    with open(path, encoding="ascii") as stream:
        for line in stream:
            key, _, rest = line.partition(":")
            if key in fields:
                values[key] = int(rest.split()[0])
    missing = sorted(set(fields) - set(values))
    if missing:
        raise Umt5ExportError(f"{path} has no fields {missing}")
    return values


def _reset_peak() -> None:
    """Reset VmHWM to the current RSS (`clear_refs` value 5 — to take the peak per stage)."""
    with open("/proc/self/clear_refs", "w", encoding="ascii") as stream:
        stream.write("5")


@dataclass
class StageRecord:
    """The wall time and memory (GiB) of one stage.

    `peak_rss` / `peak_anon` / `peak_file` are the maximum of the per-second samples (plus one
    point at the end of the stage), so a peak between samples can be missed. `hwm` misses nothing
    (VmHWM reset at the start of the stage — the true peak within the stage). `ru_maxrss` is
    `getrusage` at the end of the stage; on Linux it reads the process mm's hiwater, so it is reset
    together by `clear_refs` (measured — it drops per stage). The peak of the whole process is the
    maximum of the stages' `hwm`.
    """

    name: str
    seconds: float = 0.0
    peak_rss: float = 0.0
    peak_anon: float = 0.0
    peak_file: float = 0.0
    hwm: float = 0.0
    ru_maxrss: float = 0.0
    min_available: float = float("inf")
    min_swap_free: float = float("inf")
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seconds": round(self.seconds, 1),
            **{
                key: round(getattr(self, key), 2)
                for key in (
                    "peak_rss",
                    "peak_anon",
                    "peak_file",
                    "hwm",
                    "ru_maxrss",
                    "min_available",
                    "min_swap_free",
                )
            },
            **self.details,
        }


class MemoryMonitor:
    """Read `/proc/self/status` and `/proc/meminfo` every second on a separate thread and record
    the peak per stage.

    Inside `with MemoryMonitor() as monitor:`, place `with monitor.stage("load"):` blocks one after
    another without nesting them.
    """

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.records: list[StageRecord] = []
        self._current: StageRecord | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)

    def __enter__(self) -> MemoryMonitor:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self) -> None:
        status = _read_kib("/proc/self/status", _STATUS_FIELDS)
        meminfo = _read_kib("/proc/meminfo", _MEMINFO_FIELDS)
        with self._lock:
            record = self._current
            if record is None:
                return
            record.peak_rss = max(record.peak_rss, status["VmRSS"] / _KIB_PER_GIB)
            record.peak_anon = max(record.peak_anon, status["RssAnon"] / _KIB_PER_GIB)
            record.peak_file = max(record.peak_file, status["RssFile"] / _KIB_PER_GIB)
            record.min_available = min(record.min_available, meminfo["MemAvailable"] / _KIB_PER_GIB)
            record.min_swap_free = min(record.min_swap_free, meminfo["SwapFree"] / _KIB_PER_GIB)

    @contextmanager
    def stage(self, name: str) -> Iterator[StageRecord]:
        """Measure one stage (start after resetting VmHWM, close with one more sample at the
        end)."""
        record = StageRecord(name)
        _reset_peak()
        with self._lock:
            self._current = record
        self._sample()
        started = time.perf_counter()
        try:
            yield record
        finally:
            record.seconds = time.perf_counter() - started
            self._sample()
            with self._lock:
                self._current = None
            record.hwm = _read_kib("/proc/self/status", ("VmHWM",))["VmHWM"] / _KIB_PER_GIB
            record.ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _KIB_PER_GIB
            self.records.append(record)
            print(f"[stage] {json.dumps(record.to_dict(), ensure_ascii=False)}", flush=True)


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------


def _close_descriptors(descriptors: dict[Path, int]) -> None:
    for descriptor in descriptors.values():
        os.close(descriptor)
    descriptors.clear()


class PinnedBinShards:
    """Reader of the pinned original split pickle form (`pytorch_model-0000N-of-0000M.bin` +
    index) (ADR 0122 decision 2).

    At construction it checks the set of encoder shards derived from the index against the table
    (`wan.sources.pinned_umt5_shards`), opens every shard of the table and holds its descriptor, and
    checks the sha256 through the held descriptor — if a single one differs it fails without
    unpickling any of them.

    MUST: reads also pass the held descriptor to `torch.load` as `/proc/self/fd/<n>` (the path is
    never reopened). Opening `/proc/self/fd/<n>` opens the very inode the descriptor points to, so
    even if the path is replaced after the check (the HF cache writes under another name and
    renames), what is read is the checked content. Re-reading a 10 GB-class shard on every check
    (re-checking whenever it is reopened) is not done. The only thing this does not cover is
    someone rewriting the same inode in place (someone with write permission) — that cannot be
    closed while reading through mmap (closing it means copying everything to anonymous memory,
    which brings the RSS peak back).

    Each read opens the shard with `torch.load(mmap=True, weights_only=True)` and drops the state
    right away (the mmap lives only while a returned view is alive). Holding on to the states keeps
    the touched pages in file-backed RSS (measured with every shard held: of the quantize peak of
    17.62 GiB, 11.6 GiB was file). Opening costs 10 to 20 ms each time (unpickling touches only the
    pickle inside the shard — not the weight pages).
    """

    def __init__(self, directory: Path, shards: Mapping[str, str]) -> None:
        try:
            self.files = {
                key: directory / shard
                for key, shard in pinned_umt5_shards(directory / UMT5_BIN_INDEX, shards).items()
            }
        except (OSError, WanSourceError) as cause:
            raise Umt5ExportError(f"{directory}: {cause}") from cause
        self._descriptors: dict[Path, int] = {}
        # The descriptors are closed when the owner goes away (runs deterministically when the
        # last reference drops — the caller is not asked to close).
        self._release = weakref.finalize(self, _close_descriptors, self._descriptors)
        try:
            for name, expected in sorted(shards.items()):
                path = directory / name
                if not path.is_file():
                    raise Umt5ExportError(
                        f"{path} is missing (a shard of the table — fails before opening)"
                    )
                self._descriptors[path] = os.open(path, os.O_RDONLY)
                with open(self._descriptors[path], "rb", closefd=False) as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected:
                    raise Umt5ExportError(
                        f"{path}: the sha256 {actual} differs from the table's {expected} —"
                        " failing before unpickling (wan.sources.UMT5_SOURCES)"
                    )
        except BaseException:
            self._release()
            raise

    def tensors(self, keys: Sequence[str]) -> list[torch.Tensor]:
        """The tensors of `keys` (views on the mmap, in the order of `keys` — the caller copies
        them and drops them).

        Each shard is opened only once, so tied aliases in the same shard come back still sharing
        the same storage (the storage check of {@link Checkpoint.identical}).
        """
        states: dict[Path, Mapping[str, torch.Tensor]] = {}
        for key in keys:
            path = self.files[key]
            if path not in states:
                states[path] = self._load(path)
        return [states[self.files[key]][key] for key in keys]

    def _load(self, path: Path) -> Mapping[str, torch.Tensor]:
        handle = f"/proc/self/fd/{self._descriptors[path]}"
        try:
            return torch.load(handle, map_location="cpu", mmap=True, weights_only=True)
        except RuntimeError as cause:
            # A pickle that is not in zip form (an old torch.save) cannot be opened with mmap. An
            # exception raised for a zip-form file goes up unwrapped (so a different cause is not
            # misnamed).
            if zipfile.is_zipfile(handle):
                raise
            raise Umt5ExportError(
                f"{path} is not a zip-form torch.save (cannot be opened with mmap)"
            ) from cause


class PinnedSafetensors:
    """Reader of one safetensors file of an intake (ADR 0122 decision 5) — opens only the file the
    record names.

    At construction it opens the file, holds the descriptor, and checks the sha256 against the
    record's value through the held descriptor. The reader ({@link Checkpoint}) never reopens the
    path and passes {@link handle} (`/proc/self/fd/<n>`) to `safe_open`, so even if the path is
    replaced after the check, what is read is the checked content (the same idea as the original
    pickle reader {@link PinnedBinShards} — the only thing not covered is someone rewriting the same
    inode in place).

    MUST: neither the index (`model.safetensors.index.json`) nor any other file in the same
    directory is looked at — handing it to a reader that consults the index's `weight_map` first
    would silently read a file other than the checked one.
    """

    def __init__(self, path: Path, sha256: str) -> None:
        if not path.is_file():
            raise Umt5ExportError(f"{path} is missing (the file intake.json names)")
        self.path = path
        self._descriptor = os.open(path, os.O_RDONLY)
        # The descriptor is closed when the owner goes away (the same as {@link PinnedBinShards}).
        self._release = weakref.finalize(self, os.close, self._descriptor)
        try:
            with open(self._descriptor, "rb", closefd=False) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != sha256:
                raise Umt5ExportError(
                    f"{path}: the sha256 {actual} differs from {sha256} in intake.json —"
                    " redo the intake"
                )
        except BaseException:
            self._release()
            raise

    @property
    def handle(self) -> Path:
        """A path that opens the inode the held descriptor points to (each open is a new open file
        description — the position advanced by the check is not inherited)."""
        return Path(f"/proc/self/fd/{self._descriptor}")


#: Accepted storage → its spellings (the safetensors header / the torch dtype string).
_ACCEPTED_DTYPES: Mapping[str, tuple[str, ...]] = {
    "F32": ("F32", "torch.float32"),
    "BF16": ("BF16", "torch.bfloat16"),
}


class Checkpoint:
    """Read tensors from the upstream checkpoint one at a time, in chunks of rows.

    The way to avoid loading everything at once (decision 6). MUST: the default storage accepts F32
    only — the upstream pinned checkpoint is F32 (`total_size` 22,723,641,344 B), and another dtype
    would change the starting point of the rounding.

    `pinned` is the intake (ADR 0122 decision 5) reader {@link PinnedSafetensors} — only that file
    is read, through the checked descriptor (neither the index nor any other file in the same
    directory is looked at).

    `dtype="BF16"` is passed only by an intake (ADR 0122 decision 5 — `intake.json` declares the
    source dtype). Only the declared dtype is accepted (it fails unless every tensor has that
    dtype), and it is widened to F32 when read (lossless — every bf16 value is representable in
    f32). The reader always sees F32 and quantization is a deterministic function of it, so the
    container is byte-identical to one written from "a checkpoint holding the same values in F32".

    The default is safetensors (split or single form). `index` / `single` are the file-name
    spellings (by default those of transformers' `save_pretrained`; diffusers components are
    `diffusion_pytorch_model…` — passed by the DiT's layer-by-layer reference
    `wan.dit_reference`).

    Only when `shards` (file name → sha256) is passed is the split pickle form read through
    {@link PinnedBinShards} (ADR 0122 decision 2). MUST: only a row of the upstream table
    (`wan.sources.UMT5_SOURCES`) is passed — a `pytorch_model.bin.index.json` in the directory is
    not enough to open it (choosing the reader by whether an index exists would make it a way to
    unpickle unpinned pickles too). The split pickle form is F32 only.
    """

    def __init__(
        self,
        directory: Path,
        *,
        index: str = CHECKPOINT_INDEX,
        single: str = CHECKPOINT_SINGLE,
        shards: Mapping[str, str] | None = None,
        pinned: PinnedSafetensors | None = None,
        dtype: str = "F32",
    ) -> None:
        if dtype not in _ACCEPTED_DTYPES:
            raise Umt5ExportError(f"dtype {dtype!r} is not accepted ({sorted(_ACCEPTED_DTYPES)})")
        if shards is not None and dtype != "F32":
            raise Umt5ExportError(
                "for the split pickle form (a row of the original's pin) only F32 is accepted"
            )
        if shards is not None and pinned is not None:
            raise Umt5ExportError("shards and pinned are not used together (one reader)")
        self._dtype = dtype
        self._bin: PinnedBinShards | None = None
        # Keep the held descriptor alive for the reader's lifetime (reads go through its handle).
        self._pinned_file = pinned
        if shards is not None:
            self._bin = PinnedBinShards(directory, shards)
            self._files = self._bin.files
        elif pinned is not None:
            # Does not go through the index branch (the MUST of {@link PinnedSafetensors}).
            with safe_open(str(pinned.handle), framework="pt") as handle:
                self._files = dict.fromkeys(handle.keys(), pinned.handle)
        elif (directory / index).is_file():
            weight_map = json.loads((directory / index).read_text(encoding="utf-8"))["weight_map"]
            self._files = {key: directory / name for key, name in weight_map.items()}
        elif (directory / single).is_file():
            path = directory / single
            with safe_open(str(path), framework="pt") as handle:
                self._files = dict.fromkeys(handle.keys(), path)
        else:
            raise Umt5ExportError(f"{directory} has neither {index} nor {single}")

    def names(self) -> frozenset[str]:
        return frozenset(self._files)

    def _pinned(self, *keys: str) -> list[torch.Tensor]:
        """Tensors of the split pickle form (after checking F32 — views on the mmap. MUST: the
        caller only copies them or looks at the shape and never returns the view — holding it keeps
        the shard's mmap open)."""
        assert self._bin is not None
        tensors = self._bin.tensors(keys)
        for key, tensor in zip(keys, tensors, strict=True):
            self._assert_dtype(key, str(tensor.dtype))
        return tensors

    def _same_storage(self, keys: Sequence[str]) -> bool:
        """Whether every one of `keys` is the same storage in the split pickle form (data pointer,
        offset, shape, stride)."""
        first, *others = self._pinned(*keys)
        return all(
            tensor.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
            and tensor.storage_offset() == first.storage_offset()
            and tensor.shape == first.shape
            and tensor.stride() == first.stride()
            for tensor in others
        )

    def shape(self, key: str) -> list[int]:
        if self._bin is not None:
            (tensor,) = self._pinned(key)
            return list(tensor.shape)
        with safe_open(str(self._files[key]), framework="pt") as handle:
            view = handle.get_slice(key)
            self._assert_dtype(key, view.get_dtype())
            return list(view.get_shape())

    def read(self, key: str) -> torch.Tensor:
        """One whole tensor (small weights — RMSNorm and the relative-position tables)."""
        if self._bin is not None:
            # Copy — returning a view on the mmap would make the container or the reference keep
            # holding the shard's mmap.
            (tensor,) = self._pinned(key)
            return tensor.clone(memory_format=torch.contiguous_format)
        with safe_open(str(self._files[key]), framework="pt") as handle:
            tensor = handle.get_tensor(key)
        self._assert_dtype(key, str(tensor.dtype))
        return tensor.to(torch.float32)

    def read_rows(self, key: str, start: int, stop: int) -> torch.Tensor:
        """Only rows `[start, stop)` (a slice of the first axis — the other axes whole)."""
        if self._bin is not None:
            (tensor,) = self._pinned(key)
            return tensor[start:stop].clone(memory_format=torch.contiguous_format)
        with safe_open(str(self._files[key]), framework="pt") as handle:
            view = handle.get_slice(key)
            self._assert_dtype(key, view.get_dtype())
            return view[start:stop].to(torch.float32)

    def identical(self, keys: Sequence[str], chunk_rows: int = CHUNK_ROWS) -> bool:
        """Whether the tensors of `keys` all hold the same values (the check of tied aliases —
        {@link checkpoint_keys}).

        True without reading when they are all the same storage in the split pickle form (data
        pointer, offset, shape, stride). Otherwise the shapes are checked first, then the F32 bit
        patterns are compared chunk by chunk of rows (this also catches `-0.0` vs `0.0` and NaN
        mix-ups — bits, not value `==`).
        """
        if self._bin is not None and self._same_storage(keys):
            return True
        shapes = [self.shape(key) for key in keys]
        if any(shape != shapes[0] for shape in shapes[1:]):
            return False
        for start in range(0, shapes[0][0], chunk_rows):
            stop = min(start + chunk_rows, shapes[0][0])
            first_rows = self.read_rows(keys[0], start, stop).view(torch.int32)
            for other in keys[1:]:
                if not torch.equal(
                    first_rows, self.read_rows(other, start, stop).view(torch.int32)
                ):
                    return False
        return True

    def _assert_dtype(self, key: str, dtype: str) -> None:
        """Whether the storage of `key` is an accepted dtype (F32 by default; for an intake, what
        `intake.json` declares).

        The values returned to the reader are widened to F32 by {@link read} / {@link read_rows}
        (`to(float32)` keeps the same values for F32 and is lossless for BF16).
        """
        if dtype not in _ACCEPTED_DTYPES[self._dtype]:
            if self._dtype == "F32":
                raise Umt5ExportError(f"checkpoint '{key}' is {dtype} (only F32 is accepted)")
            raise Umt5ExportError(
                f"checkpoint '{key}' is {dtype} (only the {self._dtype} intake.json declares is"
                " accepted)"
            )


def checkpoint_keys(
    model: nn.Module, wrapper_keys: Sequence[str], checkpoint: Checkpoint
) -> dict[str, str]:
    """Wrapper tensor keys → checkpoint keys.

    The wrapper's `encoder` is the upstream `encoder` itself, so the names live in the same space.
    For tied weights (`encoder.embed_tokens.weight` = `shared.weight`), the alias present in the
    checkpoint is chosen from the aliases of the same Parameter. Wan's `save_pretrained` form has
    only one of the names, while the original's index has both (ADR 0122 decision 2).

    - One candidate: it is used.
    - Two or more candidates: if all of them hold the same values ({@link Checkpoint.identical} —
      the same storage or bit-identical), the target of transformers' tied declaration (the value of
      `_tied_weights_keys` — `shared.weight` for UMT5) is chosen.

    MUST: no candidate, a declaration target that is not unique, or aliases whose values differ in
    even one element fail loudly (different weights are never silently read).
    """
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    by_name = {name: names for names in aliases.values() for name in names}
    tied = getattr(model, "_tied_weights_keys", None)
    targets = set(tied.values()) if isinstance(tied, Mapping) else set()
    available = checkpoint.names()
    mapping: dict[str, str] = {}
    for key in wrapper_keys:
        candidates = sorted(set(by_name.get(key, [key])) & available)
        if len(candidates) > 1:
            chosen = [name for name in candidates if name in targets]
            if len(chosen) != 1:
                raise Umt5ExportError(
                    f"the checkpoint key for '{key}' is not determined uniquely: {candidates}"
                    f" (exactly one of the tied declaration's targets {sorted(targets)} is not"
                    " among the candidates)"
                )
            if not checkpoint.identical(candidates):
                raise Umt5ExportError(
                    f"'{key}': the values of the tied aliases {candidates} differ within the"
                    " checkpoint — reading either one gives different weights"
                )
            candidates = chosen
        if len(candidates) != 1:
            raise Umt5ExportError(
                f"the checkpoint key for '{key}' is not determined uniquely: {candidates}"
            )
        mapping[key] = candidates[0]
    return mapping


def _opened_checkpoint(
    directory: Path, shards: Mapping[str, str] | None, checkpoint: Checkpoint | None
) -> Checkpoint:
    """The reader already built, or one opened from `directory` (and `shards`)."""
    if checkpoint is None:
        return Checkpoint(directory, shards=shards)
    if shards is not None:
        raise Umt5ExportError("checkpoint and shards are not used together (one reader)")
    return checkpoint


def assert_encoder_keys(model: nn.Module, names: frozenset[str]) -> None:
    """Check that the checkpoint's key set is exactly the encoder's form (the same as Wan's
    `text_encoder` — `shared.weight` and `encoder.*`), nothing missing and nothing extra (intake —
    ADR 0122 decision 5).

    The expectation is the parameter names of the meta upstream built from the config, and the
    tied alias pair (`shared.weight` / `encoder.embed_tokens.weight`) counts as one — either one
    alone or both are accepted (with both, the values are checked by the rule of
    {@link checkpoint_keys}). With the xxl config that is 242 (the original's form, with both of the
    pair, has 243).

    MUST: missing or extra keys fail loudly — files with old spellings (`blocks.N.attn.q` and so on)
    or a decoder are refused with no conversion table (picking only what is read would silently
    read part of a checkpoint with a different structure).
    """
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    expected = {name for group in aliases.values() for name in group}
    missing = sorted(
        " / ".join(sorted(group)) for group in aliases.values() if not set(group) & names
    )
    extra = sorted(names - expected)
    if missing or extra:
        raise Umt5ExportError(
            f"the checkpoint's key set differs from the encoder's form ({len(aliases)} — the"
            f" tied alias pair counts as one): missing {len(missing)} {missing[:4]} / extra"
            f" {len(extra)} {extra[:4]} (files with old spellings or a decoder are not accepted"
            " — there is no conversion table)"
        )


def _pinned_intake_file(directory: Path, file: str, sha256: str) -> PinnedSafetensors:
    """A reader holding and checking the record's file `file` in the intake directory `directory`
    (if anything outside the allowed files is in the same directory, it fails before opening —
    `wan.umt5_intake.assert_only_intake_files`)."""
    try:
        assert_only_intake_files(directory, file)
    except Umt5IntakeError as cause:
        raise Umt5ExportError(str(cause)) from cause
    return PinnedSafetensors(directory / file, sha256)


def inspect_intake_checkpoint(directory: Path, file: str, sha256: str) -> str:
    """Inspect the intake's safetensors (`directory/file` — the config in `directory`) and return
    the source dtype (step 4 of `wan.umt5_intake` — ADR 0122 decision 5).

    The file is opened with the same reader as the writer ({@link PinnedSafetensors} — `sha256` is
    checked through the held descriptor, and the header and the weights are read from that same
    descriptor). It checks that every tensor has the same dtype (BF16 or F32 of
    `wan.umt5_intake.INTAKE_DTYPES`), the key set ({@link assert_encoder_keys}), the shapes (the
    same as the meta upstream built from the config) and the values of the tied alias pair
    ({@link checkpoint_keys} — with both present, bit-identical, then `shared.weight`). Any
    difference fails loudly.
    """
    path = directory / file
    pinned = _pinned_intake_file(directory, file, sha256)
    header = read_safetensors_header(pinned.handle)
    dtypes = sorted({str(entry["dtype"]) for entry in header.values()})
    if len(dtypes) != 1:
        raise Umt5ExportError(
            f"{path}: mixed dtypes ({dtypes}) — only files whose tensors all share one dtype are"
            " accepted"
        )
    (dtype,) = dtypes
    if dtype not in INTAKE_DTYPES:
        raise Umt5ExportError(
            f"{path}: dtype {dtype} is not accepted (only {list(INTAKE_DTYPES)} — FP8 and the like"
            " are derivatives that lost information)"
        )
    model = meta_text_encoder(directory)
    assert_encoder_keys(model, frozenset(header))
    shapes = {
        name: list(parameter.shape)
        for name, parameter in model.named_parameters(remove_duplicate=False)
    }
    wrong = {
        name: (entry["shape"], shapes[name])
        for name, entry in header.items()
        if entry["shape"] != shapes[name]
    }
    if wrong:
        raise Umt5ExportError(
            f"{path}: shapes differ from the encoder built from the config (file, expected):"
            f" {wrong}"
        )
    checkpoint = Checkpoint(directory, pinned=pinned, dtype=dtype)
    wrapper = umt5_patch.Umt5EncoderTokens(model)
    checkpoint_keys(model, [name for name, _ in wrapper.named_parameters()], checkpoint)
    return dtype


def intake_checkpoint(intake: Umt5Intake) -> Checkpoint:
    """The reader of an intake's (`intake.json`) safetensors (ADR 0122 decision 5).

    It checks the sha256 of the file the record names through a held descriptor
    ({@link PinnedSafetensors} — what is checked is what is read; if the intake directory holds a
    weight file outside the record, it fails before opening), builds a reader that accepts only the
    dtype the record declares, and checks the key set ({@link assert_encoder_keys}). The shapes and
    the values of the tied aliases are checked by {@link prepare} when it reads.
    """
    pinned = _pinned_intake_file(intake.directory, intake.file.name, intake.file.sha256)
    checkpoint = Checkpoint(intake.directory, pinned=pinned, dtype=intake.dtype)
    assert_encoder_keys(meta_text_encoder(intake.directory), checkpoint.names())
    return checkpoint


def checkpoint_weights(
    directory: Path,
    shards: Mapping[str, str] | None = None,
    *,
    checkpoint: Checkpoint | None = None,
) -> umt5_reference.CheckpointWeights:
    """The reader of the unquantized weights (the baseline of the quality record — step 10c).

    The mapping container tensor keys (the wrapper's parameter names) → checkpoint keys is built
    with {@link checkpoint_keys}, the same as {@link prepare} (the upstream is on meta — no weights
    are read). `shards` / `checkpoint` are the same as for {@link prepare}.
    """
    model = meta_text_encoder(directory)
    wrapper = umt5_patch.Umt5EncoderTokens(model)
    checkpoint = _opened_checkpoint(directory, shards, checkpoint)
    keys = [name for name, _ in wrapper.named_parameters()]
    return umt5_reference.CheckpointWeights(checkpoint, checkpoint_keys(model, keys, checkpoint))


# ---------------------------------------------------------------------------
# trace (an upstream with no weights)
# ---------------------------------------------------------------------------


def upstream_dir(upstream: str = UMT5_DEFAULT_MODEL) -> Path:
    """Where the umT5 upstream pin (`wan.sources.UMT5_SOURCES`) lives — the original keeps the
    encoder at the repository root (`umt5_snapshot` fails loudly if it has not been fetched)."""
    return umt5_snapshot(upstream)


def upstream_shards(upstream: str = UMT5_DEFAULT_MODEL) -> Mapping[str, str]:
    """The shard pins of the upstream table's row (`shards` of {@link Checkpoint} — the only way a
    pickle is opened)."""
    return UMT5_SOURCES[upstream].shards


def meta_text_encoder(directory: Path) -> nn.Module:
    """An upstream `UMT5EncoderModel` built from the config alone (weights on meta, eval, f32)."""
    from transformers import UMT5Config, UMT5EncoderModel

    config = UMT5Config.from_pretrained(directory)
    with torch.device("meta"):
        model = UMT5EncoderModel(config)
    return model.eval()


def trace_inputs(
    wrapper: umt5_patch.Umt5EncoderTokens, length: int = TRACE_LENGTH
) -> tuple[torch.Tensor, ...]:
    """The trace's example inputs (token ids `[1, L]` and bucket table `[L, L]`).

    The values do not affect the trace (only the shapes over the symbolic dimension are followed).
    The token ids are a fixed sequence inside the vocabulary, and the table is built with layer 0's
    upstream formula (which uses no weights).
    """
    config = wrapper.encoder.config
    ids = (torch.arange(length, dtype=torch.long) % (config.vocab_size - 3) + 3).unsqueeze(0)
    ids[0, -1] = config.eos_token_id
    attention = umt5_patch.bucket_attention(config)
    return ids, umt5_patch.relative_position_buckets(length, attention)


def trace(wrapper: umt5_patch.Umt5EncoderTokens) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """Export with the dynamic shape over the valid length `L` (the same graph whether the weights
    are on meta or real)."""
    graph, tensors = export_module(
        wrapper,
        trace_inputs(wrapper),
        dynamic_shapes=umt5_patch.dynamic_shapes(),
        symbol_names=(umt5_patch.SYMBOL,),
    )
    declared = [entry.name for entry in graph.inputs]
    if declared != list(umt5_patch.INPUT_NAMES):
        raise Umt5ExportError(f"graph input names do not match the declaration: {declared}")
    return graph, tensors


# ---------------------------------------------------------------------------
# Quantization (chunk by chunk of rows)
# ---------------------------------------------------------------------------


def quantize_rows(
    checkpoint: Checkpoint, key: str, chunk_rows: int = CHUNK_ROWS
) -> FixedQuantizedWeight:
    """Build per-row-scale i8 (axis 0 of `[rows, cols]`), reading chunk by chunk of rows.

    The procedure for one chunk is the same as the two stages of the straightforward form:
    `fake_quant_int8` replaces the weight with `quantize(w, s)·s`, and emit writes
    `quantize(q·s, s)` and checks that the inverse conversion matches. Here too the value of the
    second quantization is written and the same match is checked — writing the first `q` directly
    would differ from the straightforward form wherever an element has `round((q·s)/s) ≠ q` (their
    equality is not assumed). No operation mixes values across rows (the scale is the row's amax),
    so splitting into chunks is bit-exact with running the whole tensor.
    """
    shape = checkpoint.shape(key)
    if len(shape) != 2:
        raise Umt5ExportError(f"'{key}' has rank {len(shape)} (per-row i8 is rank 2 only)")
    rows, cols = shape
    packed = torch.empty((rows, cols), dtype=torch.int8)
    scale = torch.empty((rows, 1), dtype=torch.float32)
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        weight = checkpoint.read_rows(key, start, stop)
        chunk_scale = channel_scale(weight, 0)
        rounded = quantize_to_int8(weight, chunk_scale).to(torch.float32) * chunk_scale
        del weight
        chunk = quantize_to_int8(rounded, chunk_scale)
        if not torch.equal(chunk.to(torch.float32) * chunk_scale, rounded):
            raise Umt5ExportError(
                f"'{key}' rows {start} to {stop}: the inverse of i8 × scale does not match"
            )
        packed[start:stop] = chunk
        scale[start:stop] = chunk_scale
    return FixedQuantizedWeight(dtype="i8", packed=packed, scale=scale)


# ---------------------------------------------------------------------------
# Assembling the material
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Umt5Export:
    """The container's material (in the form `publish_model` takes as is).

    In `tensors` the quantization targets are on meta (logical shape only); the data is in `fixed`
    (packed + scale). The unquantized weights are the checkpoint's f32 data.
    """

    graph: IrGraph
    tensors: Mapping[str, torch.Tensor]
    fixed: Mapping[str, FixedQuantizedWeight]
    #: Tensor keys of the unquantized weights (RMSNorm and the relative-position tables — the kinds
    #: are in {@link storage_kind}).
    plain: tuple[str, ...]


def quant_targets(wrapper: umt5_patch.Umt5EncoderTokens) -> dict[str, int]:
    """The weights made i8 (tensor key → channel axis). The selection is the same as in the
    preliminary check (only the relative-position tables are excluded)."""
    include = umt5_patch.quant_include(wrapper)
    return {
        fqn: axis
        for fqn, _, axis in iter_quant_targets(
            wrapper, op_types=QUANT_MODULE_TYPES, include=include
        )
    }


def prepare(
    directory: Path,
    *,
    shards: Mapping[str, str] | None = None,
    checkpoint: Checkpoint | None = None,
    monitor: MemoryMonitor | None = None,
    chunk_rows: int = CHUNK_ROWS,
) -> Umt5Export:
    """trace → read the unquantized weights → make the quantization targets i8 (each stage
    measured with `monitor`).

    `directory` is where the config lives (the graph is built from the config alone). By default
    the weights are read from the safetensors in `directory`; `shards` is the same as for
    {@link Checkpoint} (only when built from a row of the upstream table — {@link upstream_shards}).
    `checkpoint` is a reader already built (an intake — {@link intake_checkpoint}) and is not used
    together with `shards`.
    """
    stage = monitor.stage if monitor is not None else _unmeasured
    with stage("trace") as record:
        model = meta_text_encoder(directory)
        wrapper = umt5_patch.Umt5EncoderTokens(model)
        graph, tensors = trace(wrapper)
        targets = quant_targets(wrapper)
        record.details["nodes"] = len(graph.nodes)
    checkpoint = _opened_checkpoint(directory, shards, checkpoint)
    weights = sorted(key for key, value in tensors.items() if value.is_meta)
    mapping = checkpoint_keys(model, weights, checkpoint)
    unknown = sorted(set(targets) - set(weights))
    if unknown:
        raise Umt5ExportError(f"quantization targets not among the graph's weights: {unknown}")
    materialized = dict(tensors)
    with stage("plain") as record:
        plain = tuple(key for key in weights if key not in targets)
        for key in plain:
            value = checkpoint.read(mapping[key])
            if list(value.shape) != list(tensors[key].shape):
                raise Umt5ExportError(
                    f"'{key}': the checkpoint shape {list(value.shape)} differs from the graph"
                )
            materialized[key] = value
        record.details["tensors"] = len(plain)
    fixed: dict[str, FixedQuantizedWeight] = {}
    with stage("quantize") as record:
        for key, axis in sorted(targets.items()):
            if axis != 0:
                raise Umt5ExportError(
                    f"the channel axis of '{key}' is {axis} (only per-row scales are built)"
                )
            weight = quantize_rows(checkpoint, mapping[key], chunk_rows)
            if list(weight.packed.shape) != list(tensors[key].shape):
                raise Umt5ExportError(f"'{key}': the checkpoint shape differs from the graph")
            fixed[key] = weight
        record.details["tensors"] = len(fixed)
    return Umt5Export(graph=graph, tensors=materialized, fixed=fixed, plain=plain)


@contextmanager
def _unmeasured(name: str) -> Iterator[StageRecord]:
    yield StageRecord(name)


def provenance(upstream: str = UMT5_DEFAULT_MODEL) -> Provenance:
    """The provenance written into the container (the umT5 upstream's license identifier and the
    pinned revision — `wan.sources.UMT5_SOURCES` is the source of truth)."""
    source = UMT5_SOURCES[upstream].source
    return Provenance(
        license=source.license, notice=NOTICE_FILENAME, upstream_revision=source.revision
    )


def intake_provenance(intake: Umt5Intake) -> Provenance:
    """The provenance written into an intake's container (the license of `intake.json` — the mark
    when undeclared — and the pinned revision; ADR 0122 decisions 5 / 6). The container's
    provenance has no dtype field, so the source dtype is not written (the record and the card of
    the experimental mirror state it)."""
    return Provenance(
        license=intake.license, notice=NOTICE_FILENAME, upstream_revision=intake.revision
    )


def write_container(
    export: Umt5Export, path: Path, provenance_: Provenance | None = None
) -> IrGraph:
    """Write the material (what {@link prepare} returns) to a container (`path` is the
    representative path — it becomes the parts of the split form).

    The storage is the material as is (the quantization targets are the packed + scale of `fixed`,
    the rest the checkpoint's f32), and the graph name is the component name {@link GRAPH_NAME}.
    The provenance is `provenance_` (when omitted, the original's row — {@link provenance}). Returns
    the graph with the storage declarations committed (the input of the acceptance table —
    {@link storage_by_kind}).
    """
    return publish_model(
        path,
        export.graph,
        dict(export.tensors),
        provenance=provenance_ if provenance_ is not None else provenance(),
        graph_name=GRAPH_NAME,
        fixed_weights=export.fixed,
    )


def part_digests(path: Path) -> list[str]:
    """The sha256 of the container's (representative path) parts (from part 0 in index order)."""
    digests = []
    for part in container_parts(path):
        with part.open("rb") as stream:
            digests.append(hashlib.file_digest(stream, "sha256").hexdigest())
    return digests


def assert_same_container(written: Path, existing: Path) -> None:
    """Check by sha256 that the written container is byte-identical to the existing one, part by
    part.

    MUST: a difference fails loudly (naming the parts that differ). The series container is the
    root of both the goldens (whose meta holds the sha256 of part 0) and the distribution, so a
    writer that silently produces different bytes is never taken for "the same container".
    """
    actual = part_digests(written)
    expected = part_digests(existing)
    if len(actual) != len(expected):
        raise Umt5ExportError(
            f"the part counts differ (written {len(actual)} / existing {len(expected)} —"
            f" {existing})"
        )
    differing = [index for index, (a, b) in enumerate(zip(actual, expected, strict=True)) if a != b]
    if differing:
        raise Umt5ExportError(
            f"part {differing}: the sha256 differs from the existing container ({existing}) —"
            " the writer or the material produces bytes different from the step 10b container"
        )


@dataclass(frozen=True)
class EncoderInput:
    """The input of one umT5 upstream axis (ADR 0122 decision 3 — the original's row `--upstream`
    or an intake `--intake`). Each writer subcommand looks only at this (the paths are not split by
    the two axes)."""

    #: Where the config lives (the graph is built from the config alone).
    directory: Path
    #: Opens the weight reader (for the original's row this includes the sha256 check of the pickle
    #: shards; for an intake, the sha256 check of the file and the key-set check).
    open: Callable[[], Checkpoint]
    provenance: Provenance
    #: Parent of the series (`<series>/text_encoder/model.krm`).
    series: Path
    #: The axis record kept in the golden meta (`encoder` of {@link reference_axes}).
    axis: Mapping[str, str]


def upstream_input(upstream: str = UMT5_DEFAULT_MODEL) -> EncoderInput:
    """The input of the original's row (`wan.sources.UMT5_SOURCES`) (fails loudly if not
    fetched)."""
    directory = upstream_dir(upstream)
    return EncoderInput(
        directory=directory,
        open=lambda: Checkpoint(directory, shards=upstream_shards(upstream)),
        provenance=provenance(upstream),
        series=SERIES,
        axis=upstream_axis(upstream),
    )


def intake_input(intake: Umt5Intake) -> EncoderInput:
    """The input of an intake (`intake.json`). The series is `umt5-xxl-<name>-i8-dyn` (ADR 0122
    decision 5 — spelled by `wan.umt5_distribution.intake_series_name`)."""
    return EncoderInput(
        directory=intake.directory,
        open=lambda: intake_checkpoint(intake),
        provenance=intake_provenance(intake),
        series=SERIES_ROOT / intake_series_name(intake.name),
        axis={
            "intake": intake.name,
            "repo": intake.repo,
            "revision": intake.revision,
            "file": intake.file.name,
            "dtype": intake.dtype,
        },
    )


def write_series(upstream: str = UMT5_DEFAULT_MODEL, *, check: bool) -> dict[str, Any]:
    """Write the series container from the original's row ({@link write_encoder})."""
    return write_encoder(upstream_input(upstream), check=check)


def write_intake_series(
    directory: Path, *, check: bool, allow_undeclared_license: bool
) -> dict[str, Any]:
    """Write the series container from an intake ({@link write_encoder}).

    MUST: an intake with an undeclared license fails before a single byte is written unless it is
    explicitly allowed (`--allow-undeclared-license`) (ADR 0122 decision 6). When allowed, the
    container's license field records the undeclared mark.
    """
    intake = load_intake(directory)
    try:
        assert_license_intent(intake, allowed=allow_undeclared_license)
    except Umt5IntakeError as cause:
        raise Umt5ExportError(str(cause)) from cause
    return write_encoder(intake_input(intake), check=check)


def write_encoder(source: EncoderInput, *, check: bool) -> dict[str, Any]:
    """Write the series container (with `check`, it only writes to a scratch directory and checks
    against the existing one; the series is not replaced).

    A rewrite replaces the series' component directory as a whole (`staged_publication` — the
    golden `reference.*` files go too, so rewrite them next with `reference`).
    """
    target = source.series / COMPONENT_DIR
    with MemoryMonitor() as monitor:
        export = prepare(source.directory, checkpoint=source.open(), monitor=monitor)
        with monitor.stage("write"):
            if check:
                with tempfile.TemporaryDirectory(dir=source.series, prefix=".check-") as scratch:
                    written = Path(scratch) / MODEL_FILE
                    graph = write_container(export, written, source.provenance)
                    assert_same_container(written, target / MODEL_FILE)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with staged_publication(target) as staged:
                    staged.mkdir()
                    graph = write_container(export, staged / MODEL_FILE, source.provenance)
    return {
        "check": check,
        "container": str(target / MODEL_FILE),
        "parts": len(container_parts(target / MODEL_FILE)),
        "storage_by_kind": storage_by_kind(graph),
        "stages": [record.to_dict() for record in monitor.records],
    }


def storage_by_kind(graph: IrGraph) -> dict[str, dict[str, int]]:
    """Count the storage declarations by kind × storage dtype (`graph` has its storage
    declarations committed)."""
    counts: dict[str, Counter[str]] = {}
    for initializer in graph.initializers.values():
        kind = storage_kind(initializer.tensor or "")
        counts.setdefault(kind, Counter())[initializer.storage.dtype] += 1
    return {kind: dict(sorted(counter.items())) for kind, counter in sorted(counts.items())}


# ---------------------------------------------------------------------------
# Measurement subcommands
# ---------------------------------------------------------------------------


#: The dtypes run to back decision 4 (bf16 = the path of upstream and the assets, f32 = a control
#: that shows the rounding floor).
MASK_DTYPES: Mapping[str, torch.dtype] = {"bf16": torch.bfloat16, "f32": torch.float32}

#: Spelling of the output files (`mask-<dtype>.safetensors`; tensors are `<form>.<prompt name>`).
MASK_FILE = "mask-{dtype}.safetensors"
MASK_FORMS = ("valid", "padded")


def load_upstream(model: str, dtype: torch.dtype) -> tuple[Any, Any]:
    """The tokenizer and umT5 of Wan's pinned revision (`dtype`, CPU, eval — the axis of Wan's
    text-side snapshot, ADR 0122 decision 3).

    Called the same way as the asset path (`wan.text_embeds.load_text_encoder` — bf16); only the
    dtype is chosen. f32 is exactly the first stage of the straightforward form (read every weight
    in f32), and its RSS is the starting point of the research.
    """
    from transformers import AutoTokenizer, UMT5EncoderModel

    snapshot = text_snapshot(model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, subfolder="tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(
        snapshot, subfolder=WAN_TEXT_ENCODER_SUBFOLDER, dtype=dtype
    ).eval()
    if encoder.dtype != dtype:
        raise Umt5ExportError(f"umT5 was loaded as {encoder.dtype} ({dtype} was requested)")
    return tokenizer, encoder


def check_mask(
    model: str = DEFAULT_MODEL, dtype: str = "bf16", out: Path | None = None
) -> dict[str, Any]:
    """Backing decision 4 on the real model: whether "valid length only" and "512 + mask" agree
    upstream.

    Runs the upstream forward both ways on the four fixed prompts and also checks against the
    precomputed assets (the bf16 output of upstream `_get_t5_prompt_embeds`, converted to f32). In
    bf16, "512 + mask" matching the assets is the control showing the asset path is reproduced
    here. f32 is a control that shows the rounding floor (difference from the assets = the error of
    the bf16 path). With `out`, both outputs are written to {@link MASK_FILE} (the cross-dtype
    comparison is {@link compare_mask}).
    """
    from wan.prompts import FIXED_PROMPTS
    from wan.text_embeds import ASSET_NAME, normalize, read_asset
    from wan.text_embeds import SERIES_NAME as EMBEDS_SERIES

    asset, _ = read_asset(SERIES_ROOT / EMBEDS_SERIES / ASSET_NAME)
    rows: list[dict[str, Any]] = []
    outputs: dict[str, torch.Tensor] = {}
    with MemoryMonitor() as monitor:
        with monitor.stage(f"load-{dtype}") as record:
            tokenizer, encoder = load_upstream(model, MASK_DTYPES[dtype])
            record.details["parameters"] = sum(p.numel() for p in encoder.parameters())
        for prompt in FIXED_PROMPTS:
            with monitor.stage(f"encode-{prompt.name}") as record:
                ids = torch.tensor(
                    [tokenizer([normalize(prompt.text)], add_special_tokens=True).input_ids[0]],
                    dtype=torch.long,
                )
                began = time.perf_counter()
                valid = umt5_patch.valid_output(encoder, ids)[0].float()
                valid_seconds = time.perf_counter() - began
                padded = umt5_patch.padded_output(encoder, ids)[0].float()
                expected = asset[prompt.name]
                row = {
                    "name": prompt.name,
                    "length": int(ids.shape[1]),
                    "valid_seconds": round(valid_seconds, 1),
                    "padded_seconds": round(time.perf_counter() - began - valid_seconds, 1),
                    "valid_vs_padded": dict(umt5_patch.compare(valid, padded)),
                    "padded_vs_asset": dict(umt5_patch.compare(padded, expected)),
                    "valid_vs_asset": dict(umt5_patch.compare(valid, expected)),
                }
                record.details["length"] = row["length"]
                rows.append(row)
                outputs[f"valid.{prompt.name}"] = valid.contiguous()
                outputs[f"padded.{prompt.name}"] = padded.contiguous()
                print(f"[mask] {json.dumps(row, ensure_ascii=False)}", flush=True)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        save_file(outputs, str(out / MASK_FILE.format(dtype=dtype)))
    return {"prompts": rows, "stages": [record.to_dict() for record in monitor.records]}


def compare_mask(directory: Path) -> list[dict[str, Any]]:
    """Compare the bf16 and f32 outputs of {@link check_mask} (the difference between the two bf16
    forms against f32).

    `valid_error` / `padded_error` are the differences of each bf16 form from the same form in f32
    — the error of the bf16 path itself. If the `valid_vs_padded` difference is of the same order
    as this error, the difference between the two forms is a difference in rounding order, and both
    are bf16 approximations of the same precision.
    """
    files = {
        dtype: load_file(str(directory / MASK_FILE.format(dtype=dtype))) for dtype in MASK_DTYPES
    }
    names = sorted({key.split(".", 1)[1] for key in files["bf16"]})
    rows = []
    for name in names:
        bf16 = {form: files["bf16"][f"{form}.{name}"] for form in MASK_FORMS}
        f32 = {form: files["f32"][f"{form}.{name}"] for form in MASK_FORMS}
        rows.append(
            {
                "name": name,
                "length": int(bf16["valid"].shape[0]),
                "bf16_valid_vs_padded": dict(umt5_patch.compare(bf16["valid"], bf16["padded"])),
                "f32_valid_vs_padded": dict(umt5_patch.compare(f32["valid"], f32["padded"])),
                "valid_error": dict(umt5_patch.compare(bf16["valid"], f32["valid"])),
                "padded_error": dict(umt5_patch.compare(bf16["padded"], f32["padded"])),
            }
        )
    return rows


def prepare_summary(upstream: str = UMT5_DEFAULT_MODEL) -> dict[str, Any]:
    """{@link prepare_encoder} for the original's row."""
    return prepare_encoder(upstream_input(upstream))


def prepare_intake_summary(directory: Path) -> dict[str, Any]:
    """{@link prepare_encoder} for an intake (it writes no container, so the undeclared-license
    check is not applied)."""
    return prepare_encoder(intake_input(load_intake(directory)))


def prepare_encoder(source: EncoderInput) -> dict[str, Any]:
    """Measure {@link prepare} and return a summary of the material (no container is written —
    module doc)."""
    with MemoryMonitor() as monitor:
        export = prepare(source.directory, checkpoint=source.open(), monitor=monitor)
    return {
        "nodes": len(export.graph.nodes),
        "fixed": len(export.fixed),
        "plain": len(export.plain),
        "packed_bytes": sum(w.packed.numel() for w in export.fixed.values()),
        "stages": [record.to_dict() for record in monitor.records],
    }


def reference_axes(upstream: str, wan_model: str) -> dict[str, dict[str, str]]:
    """The two axes kept in the golden meta (ADR 0122 decision 3 — `reference` is the only
    subcommand that uses both together).

    - `encoder`: the umT5 upstream (config and unquantized weights — `wan.sources.UMT5_SOURCES`)
    - `cases`: the Wan text-side snapshot the cases' id sequences came from (the tokenizer —
      `wan.sources.SOURCES`)
    """
    return _axes(upstream_axis(upstream), wan_model)


def upstream_axis(upstream: str) -> dict[str, str]:
    """The axis record of the original's row (`axis` of {@link EncoderInput} — can be built even
    when not fetched)."""
    encoder = UMT5_SOURCES[upstream].source
    return {"model": upstream, "repo": encoder.repo, "revision": encoder.revision}


def _axes(encoder: Mapping[str, str], wan_model: str) -> dict[str, dict[str, str]]:
    cases = SOURCES[wan_model]
    return {
        "encoder": dict(encoder),
        "cases": {
            "model": wan_model,
            "repo": cases.repo,
            "revision": cases.revision,
            "subfolder": "tokenizer",
        },
    }


def reference_summary(
    upstream: str = UMT5_DEFAULT_MODEL, wan_model: str = DEFAULT_MODEL
) -> dict[str, Any]:
    """{@link reference_encoder} for the original's row."""
    return reference_encoder(upstream_input(upstream), wan_model)


def reference_intake_summary(directory: Path, wan_model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """{@link reference_encoder} for an intake (the unquantized reference is the intake's weights
    widened to F32)."""
    return reference_encoder(intake_input(load_intake(directory)), wan_model)


def reference_encoder(source: EncoderInput, wan_model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Take the layer-by-layer CPU reference (f64 / f32) from the written i8 series container and
    write the goldens to the same directory (step 10c — no container is written).

    The cases' id sequences are taken through the Wan tokenizer path (the same as the 10a fixture
    — the axis `wan_model`). The config and the unquantized weights of the acceptance cases (the
    four fixed ones; the checkpoint's F32 — {@link checkpoint_weights}) are read from the umT5
    upstream axis (`source` — the original's row or an intake). The difference from the
    precomputed bf16 assets is also put in the summary (record only). Which axis values were used
    is in the golden meta ({@link reference_axes}; for an intake, `axis` of {@link intake_input}).
    """
    from transformers import UMT5Config

    from wan.text_embeds import ASSET_NAME, read_asset
    from wan.text_embeds import SERIES_NAME as EMBEDS_SERIES

    component = source.series / COMPONENT_DIR
    directory = source.directory
    with MemoryMonitor() as monitor:
        with monitor.stage("cases") as record:
            cases = umt5_reference.reference_cases(umt5_reference.upstream_encoder(wan_model))
            record.details["lengths"] = {case.name: len(case.ids) for case in cases}
        embeddings, _ = read_asset(SERIES_ROOT / EMBEDS_SERIES / ASSET_NAME)
        result = umt5_reference.write_references(
            container=component / MODEL_FILE,
            out_dir=component,
            config=UMT5Config.from_pretrained(directory),
            cases=cases,
            stage=monitor.stage,
            unquantized=checkpoint_weights(directory, checkpoint=source.open()),
            axes=_axes(source.axis, wan_model),
            embeddings=embeddings,
        )
    return {**result, "stages": [record.to_dict() for record in monitor.records]}


#: The arguments each subcommand uses (argparse dests). MUST: any other argument given explicitly
#: is refused — silently ignoring it would let a run that meant to set an unused axis (e.g.
#: `write --model`) run on different input.
COMMAND_ARGUMENTS: Mapping[str, frozenset[str]] = {
    "prepare": frozenset({"upstream", "intake"}),
    "write": frozenset({"upstream", "intake", "check", "allow_undeclared_license"}),
    "check-mask": frozenset({"model", "dtype", "out"}),
    "compare-mask": frozenset({"out"}),
    "reference": frozenset({"upstream", "intake", "model"}),
}


def _flag(dest: str) -> str:
    """argparse dest → the argument's spelling (`allow_undeclared_license` →
    `--allow-undeclared-license`)."""
    return "--" + dest.replace("_", "-")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=tuple(COMMAND_ARGUMENTS))
    # Optional arguments default to None (to tell explicit from omitted and refuse an explicit
    # argument that has no effect) — the real defaults are filled in below.
    parser.add_argument(
        "--upstream",
        choices=sorted(UMT5_SOURCES),
        help=f"the umT5 upstream (weights, config and provenance for prepare / write; config and"
        f" unquantized weights for reference — default {UMT5_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--model",
        choices=WAN21_MODELS,
        help="the Wan text-side snapshot (id sequences of the reference cases, check-mask — the"
        " same axis as --model of the sibling scripts text_embeds / umt5_tokenizer /"
        f" umt5_host_fixture — default {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=None,
        help="write: write to a scratch directory and check against the series container by"
        " sha256 (the series is not replaced)",
    )
    parser.add_argument("--dtype", choices=sorted(MASK_DTYPES), help="check-mask (default bf16)")
    parser.add_argument(
        "--out", type=Path, help="the directory check-mask writes / compare-mask reads"
    )
    parser.add_argument(
        "--intake",
        type=Path,
        help="make the umT5 upstream axis an intake (inputs/umt5/<name>/ — wan.umt5_intake)"
        " (prepare / write / reference — not used together with --upstream; the series is"
        " umt5-xxl-<name>-i8-dyn)",
    )
    parser.add_argument(
        ALLOW_UNDECLARED_LICENSE_FLAG,
        dest="allow_undeclared_license",
        action="store_true",
        default=None,
        help="write --intake: from an intake with an undeclared license, write a container that"
        " records the undeclared mark (local experiments only — ADR 0122 decision 6)",
    )
    args = parser.parse_args(argv)
    accepted = COMMAND_ARGUMENTS[args.command]
    for name in sorted(frozenset().union(*COMMAND_ARGUMENTS.values())):
        if getattr(args, name) is not None and name not in accepted:
            parser.error(
                f"{_flag(name)} does not apply to {args.command} (it takes"
                f" {', '.join(_flag(other) for other in sorted(accepted))})"
            )
    if args.intake is not None and args.upstream is not None:
        parser.error("--intake and --upstream are the same axis (the umT5 upstream) — pick one")
    if args.allow_undeclared_license is not None and args.intake is None:
        parser.error(f"{ALLOW_UNDECLARED_LICENSE_FLAG} applies only to write with --intake")
    upstream = args.upstream if args.upstream is not None else UMT5_DEFAULT_MODEL
    model = args.model if args.model is not None else DEFAULT_MODEL
    started = time.perf_counter()
    if args.command == "prepare":
        summary: Any = (
            prepare_summary(upstream)
            if args.intake is None
            else prepare_intake_summary(args.intake)
        )
    elif args.command == "write":
        if args.intake is None:
            summary = write_series(upstream, check=args.check is True)
        else:
            summary = write_intake_series(
                args.intake,
                check=args.check is True,
                allow_undeclared_license=args.allow_undeclared_license is True,
            )
    elif args.command == "check-mask":
        summary = check_mask(model, args.dtype if args.dtype is not None else "bf16", args.out)
    elif args.command == "reference":
        summary = (
            reference_summary(upstream, model)
            if args.intake is None
            else reference_intake_summary(args.intake, model)
        )
    else:
        if args.out is None:
            parser.error("compare-mask needs --out (the directory check-mask wrote to twice)")
        summary = compare_mask(args.out)
    print(
        json.dumps(
            {
                "command": args.command,
                "result": summary,
                "seconds": round(time.perf_counter() - started, 1),
                "torch_threads": torch.get_num_threads(),
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
