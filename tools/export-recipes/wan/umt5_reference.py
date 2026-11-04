"""The layer-by-layer CPU reference of Wan2.1 umT5 (the i8 series) and the goldens of the port
check (ADR 0119 decision 8 ①, step 10c).

The CLI is `wan.umt5_export reference` (it writes no container — it reads the weights from an i8
series container already written).

## The reference (decision 8 ① — layer by layer on CPU with the same i8 fake-quant weights)

The port check's reference (`output.f64` / `output.f32`) is taken with the i8 weights, and the
baseline of the quality record (acceptance only — `output.unquantized.*`) with unquantized
weights ({@link CheckpointWeights}). Both use the same spelled-out computation; only the way the
weights are read differs.

- **Weights**: read one tensor at a time from the i8 series container ({@link ContainerWeights}).
  An i8 weight is packed × scale multiplied in f32, the same value as the export's fake-quant
  (`rounded` of `umt5_export.quantize_rows`) — the same weights the GPU runs. F32 tables
  (relative-position tables, RMSNorm) are taken as is. The f64 reference widens these f32 values
  to f64 (re-rounding in f64 would make the scale and the rounded values disagree with the f32
  reference — the same rule as the DiT's `export_dit.float64_references`).
- **Computation**: the upstream forward is **spelled out** ({@link encode_layerwise}): token
  embedding → per layer [RMSNorm → self-attention (relative-position bias, no scale, softmax) →
  residual → RMSNorm → FFN with gated GELU (tanh approximation — ruling (a) of the 10c
  preparation) → residual] → final RMSNorm. There is no dropout and no mask (valid length only —
  decision 4).
- **f64 and f32 use the same spelled-out code and differ only in dtype.** Upstream drops the
  RMSNorm variance and the softmax to f32 (`.to(torch.float32)` / `.float()`), so calling
  `.double()` on the upstream model does not give an f64 reference (ADR 0119 addendum "10c
  preparation"). The spelled-out code has no such cast. In f32 the upstream cast is the identity,
  so it is bit-exact with the upstream eager (activation replaced with tanh, fake-quant applied)
  (pinned by pytest on a small model).
- **Layer by layer**: the activations `[1, L, 4096]` of every case are kept while the layers
  advance one at a time. Only one layer's weights are held at once (about 0.77 GB f32 + about
  1.55 GB f64) — one whole forward in f32 after fake-quant takes about 21 GiB of anonymous memory
  (ADR 0119 addendum "10b results"). For the token embedding, only the blocks holding the rows in
  use are read from the container.
- The operations inside a layer run in the same form as upstream (`functional.linear` on
  `[1, L, d]` as is, per case). Concatenating cases along the rows would change the GEMM split,
  and the f32 reference could then differ from the upstream eager in bits.

## golden (`reference.<case>.safetensors` — one file per case)

- `input_ids` (I32 `[L]`): the id sequence of the upstream path (`prompt_clean` → transformers
  5.14.1 — `umt5_tokenizer.build_case` of 10a)
- `relative_position_buckets` (I32 `[L, L]`): the upstream formula
  (`umt5_patch.relative_position_buckets`)
- `output.f64` (F32 `[1, L, 4096]`): the f64 reference rounded to f32 (the TS safetensors reader
  does not read F64 — the effect of the rounding is in the meta's `ratios`)
- `output.f32` (F32 `[1, L, 4096]`): the f32 reference (the denominator of the normalized ratio r)
- `output.unquantized.f64` / `output.unquantized.f32` (F32 `[1, L, 4096]`, **acceptance only**):
  the same spelled-out computation run with unquantized weights (the F32 of the pinned checkpoint
  as is — {@link CheckpointWeights}) (f64 is rounded to f32 for storage). The baseline of the
  quality record ("use the f32 reference as the baseline" in ADR 0119 addendum "10b results" —
  the effects of i8 rounding and of bf16 are measured separately against the same baseline); not
  used by the check

The meta is JSON under a single key ({@link METADATA_KEY}) (sorted keys, no whitespace after
separators — safetensors writes the meta through a HashMap, so with several keys the same input
can give different bytes — the same reason as `wan.text_embeds`). It holds the prompt's original
text and the preprocessed text, L, the role, the provenance, the axis record (`axes` — the pins of
the umT5 upstream and of the Wan tokenizer the id sequence came from), the sha256 of the
container's part 0 (when the container is rewritten, the golden is rewritten too — TS checks it),
the rules for the weights and the activation, the denominator and the rounding effect
({@link reference_ratios}), and for acceptance cases the same ratios of the unquantized reference.

The TS e2e (`packages/models/tests/e2e_wan_umt5_test.ts`) first checks that the id sequence from
passing the golden's original text through the TS tokenizer, and the TS bucket table, are
bit-identical to the golden's inputs (both the TS and the Python paths), then judges the GPU
output by the normalized ratio
r = (GPU's ratio against f64) ÷ (CPU f32's ratio against f64).

## Cases (decision 8 ① — deciding the band and acceptance are kept apart)

- **Deciding cases (`band`), 6**: chosen with seed {@link BAND_SEED} from the accepted random
  cases (`random-*` — random characters from the accepted set, 3 to 42 tokens) of the 10a parity
  fixture (`packages/models/tests/fixtures/wan-text/parity.json`).
  - 3 single cases: one per length band {@link BAND_SINGLE_LENGTHS} (short, medium, long).
  - 3 composite cases: the random cases have nothing near 512, so the random cases not chosen as
    single ones are shuffled with the same seed and joined with spaces from the front until the
    length falls in a band {@link BAND_COMPOSITE_LENGTHS} (beyond max_distance 128, the middle,
    just below the upper bound 512) (a case that would overshoot the band is skipped). The joined
    text is also re-tokenized through the upstream path.
- **Acceptance (`accept`), 4**: the four fixed prompts (`wan.prompts.FIXED_PROMPTS` — lengths
  28 / 118 / 50 / 126). Only these four have precomputed bf16 assets (the third point of the
  quality record — decision 8 ②).

MUST: the acceptance results never change the deciding cases or how they are chosen (deciding the
band and acceptance stay independent — ADR 0118 decision 8).

MUST: transformers / diffusers are imported inside functions (the `wan` group is not in the
default sync — `tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
from safetensors.torch import save_file
from torch.nn import functional

from karume.container import codec_entry, container_parts
from karume.convert import normalize_boundary_tensor
from karume.verify import InitializerSupply, verify_container
from wan import umt5_patch
from wan.umt5_tokenizer import FIXED_PROMPT_LENGTHS, FIXTURE_DIR, MAX_LENGTH, PARITY_FIXTURE

#: golden file names (`reference.<case>.safetensors` — the same spelling as the DiT goldens).
REFERENCE_PREFIX = "reference."
CASE_SUFFIX = ".safetensors"

#: golden keys (the inputs use the graph input names — the same spelling as the TS side
#: `tests/helpers/umt5-golden.ts`).
INPUT_IDS_KEY, BUCKETS_KEY = umt5_patch.INPUT_NAMES
OUTPUT_F64_KEY = "output.f64"
OUTPUT_F32_KEY = "output.f32"
#: The unquantized reference (acceptance only — the baseline of the quality record).
OUTPUT_UNQUANTIZED_F64_KEY = "output.unquantized.f64"
OUTPUT_UNQUANTIZED_F32_KEY = "output.unquantized.f32"

#: The meta key (a single one — module doc) and the format version.
METADATA_KEY = "karume.wan.umt5_reference"
#: /2 = acceptance cases gain the unquantized reference (TS requires these keys for acceptance).
REFERENCE_FORMAT = "karume-wan-umt5-reference/2"

#: Container tensor keys (the container's initializer names = the tensor keys of the wrapper
#: `Umt5EncoderTokens`).
EMBED_KEY = "encoder.embed_tokens.weight"
FINAL_NORM_KEY = "encoder.final_layer_norm.weight"

#: The quantization codec (i8 per-channel, one scale per row — decision 5).
I8_CODEC = "int8-sym"

#: Seed for choosing the deciding cases (`random.Random`).
BAND_SEED = 20261003

#: Length bands of the single deciding cases (both ends inclusive). The accepted random cases of
#: the parity fixture are 3 to 42 tokens.
BAND_SINGLE_LENGTHS: tuple[tuple[int, int], ...] = ((2, 8), (16, 24), (34, MAX_LENGTH))

#: Length bands of the composite deciding cases (both ends inclusive): beyond max_distance 128
#: (the end of the bucket's logarithmic range), the middle, and just below the upper bound 512.
BAND_COMPOSITE_LENGTHS: tuple[tuple[int, int], ...] = ((160, 200), (300, 360), (480, MAX_LENGTH))

#: id prefix of the parity fixture's random cases.
RANDOM_PREFIX = "random-"

Role = Literal["band", "accept"]


class Umt5ReferenceError(RuntimeError):
    """The container, a case or the reference differs from what is assumed (decisions 5 / 8)."""


# ---------------------------------------------------------------------------
# Weights (one tensor at a time from the container)
# ---------------------------------------------------------------------------


class WeightSource(Protocol):
    """How the reference reads weights (values in f32 — i8 is packed × scale)."""

    def tensor(self, key: str) -> torch.Tensor: ...

    def rows(self, key: str, indices: torch.Tensor) -> torch.Tensor: ...


def _f32_of(raw: bytes) -> torch.Tensor:
    # Copy through a bytearray — frombuffer on a read-only buffer makes torch warn.
    return torch.frombuffer(bytearray(raw), dtype=torch.float32)


class ContainerWeights:
    """Read weights one tensor at a time from an i8 series container (the parts of a `krm`).

    Each block's sha256 is checked against the declaration every time it is read
    (`ReadContainer.block`). For i8 it returns packed × per-row scale multiplied in f32 — the same
    value as the export's fake-quant (bit-exactness is pinned by pytest on a small model's
    container). MUST: only the codecs `f32` and `int8-sym` (axis 0, one scale per row) are accepted
    — anything else would silently turn the reference into different weights.
    """

    def __init__(self, path: Path) -> None:
        parts = container_parts(path)
        verified = verify_container(list(parts))
        if len(verified.graphs) != 1:
            raise Umt5ReferenceError(
                f"{path}: the graphs are {sorted(verified.graphs)} (there should be one)"
            )
        (graph,) = verified.graphs
        self.graph = graph
        self.parts = len(parts)
        with parts[0].open("rb") as stream:
            #: sha256 of part 0 (the header + the 2 documents — declares the sha256 of every block
            #: and part). It names the container's contents transitively, so TS catches a golden
            #: paired with the wrong container through it.
            self.part0_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self._read = verified.read
        self._values = verified.read.graph.graphs[graph]["values"]
        self._supplies = verified.graphs[graph].supplies

    def _supply(self, key: str) -> InitializerSupply:
        supply = self._supplies.get(key)
        if supply is None:
            raise Umt5ReferenceError(f"the container has no data for '{key}'")
        return supply

    def _shape(self, key: str) -> list[int]:
        return [int(size) for size in self._values[key]["shape"]]

    def _payload(self, supply: InitializerSupply, index: int) -> bytes:
        block = supply.blocks[index]
        return self._read.block(block.id)[: block.payload_bytes]

    def _scale(self, key: str, supply: InitializerSupply, rows: int, cols: int) -> torch.Tensor:
        """The per-row scale `[rows, 1]` (f32). Also checks that the codec declaration is per-row
        i8."""
        encoding = supply.encoding
        if encoding.codec != I8_CODEC or encoding.row_axis != 0 or encoding.group_size != cols:
            raise Umt5ReferenceError(
                f"'{key}': codec {encoding.codec}, rowAxis {encoding.row_axis}, groupSize "
                f"{encoding.group_size} (only {I8_CODEC}, 0, {cols} is read)"
            )
        if supply.scale is None:
            raise Umt5ReferenceError(f"'{key}': i8 but no scale")
        scale = _f32_of(self._read.block(supply.scale.id)[: supply.scale.payload_bytes])
        if scale.numel() != rows:
            raise Umt5ReferenceError(
                f"'{key}': {scale.numel()} scales differ from the row count {rows}"
            )
        return scale.reshape(rows, 1)

    def tensor(self, key: str) -> torch.Tensor:
        """One whole tensor in f32 (i8 is packed × scale)."""
        supply = self._supply(key)
        shape = self._shape(key)
        raw = b"".join(self._payload(supply, index) for index in range(len(supply.blocks)))
        layout = codec_entry(supply.encoding.codec).layout
        if layout == "f32":
            return _f32_of(raw).reshape(shape)
        if layout != "i8" or len(shape) != 2:
            raise Umt5ReferenceError(f"'{key}': storage {layout} with shape {shape} is not read")
        rows, cols = shape
        packed = torch.frombuffer(bytearray(raw), dtype=torch.int8).reshape(rows, cols)
        return packed.to(torch.float32) * self._scale(key, supply, rows, cols)

    def rows(self, key: str, indices: torch.Tensor) -> torch.Tensor:
        """Only rows `indices` (ascending, no duplicates) of an i8 table, in f32 (the token
        embedding — reads only the blocks needed)."""
        supply = self._supply(key)
        rows, cols = self._shape(key)
        wanted = indices.tolist()
        if wanted != sorted(set(wanted)) or not wanted or wanted[0] < 0 or wanted[-1] >= rows:
            raise Umt5ReferenceError(
                f"'{key}': pass row indices within 0 to {rows - 1}, in ascending order without"
                " duplicates"
            )
        if codec_entry(supply.encoding.codec).layout != "i8":
            raise Umt5ReferenceError(f"'{key}': rows can only be read from an i8 table")
        scale = self._scale(key, supply, rows, cols)
        gathered = torch.empty((len(wanted), cols), dtype=torch.float32)
        covered = torch.zeros(len(wanted), dtype=torch.bool)
        for index, block in enumerate(supply.blocks):
            begin, end = block.rows
            # (position in gathered, row in the block) — placed regardless of the block order.
            inside = [(at, row - begin) for at, row in enumerate(wanted) if begin <= row < end]
            if not inside:
                continue
            positions = torch.tensor([at for at, _ in inside])
            local = torch.tensor([row for _, row in inside])
            packed = torch.frombuffer(bytearray(self._payload(supply, index)), dtype=torch.int8)
            gathered[positions] = packed.reshape(end - begin, cols)[local].to(torch.float32)
            covered[positions] = True
        if not bool(covered.all()):
            raise Umt5ReferenceError(
                f"'{key}': the blocks' row ranges do not cover the rows needed"
            )
        return gathered * scale[indices]


class CheckpointReader(Protocol):
    """How the upstream checkpoint is read (`umt5_export.Checkpoint` — returns F32 only)."""

    def read(self, key: str) -> torch.Tensor: ...

    def read_rows(self, key: str, start: int, stop: int) -> torch.Tensor: ...


class CheckpointWeights:
    """Read unquantized weights (the F32 of the pinned checkpoint as is) one tensor at a time (the
    baseline of the quality record).

    The same interface as {@link ContainerWeights}, so the same spelled-out code is used. `keys`
    maps container tensor keys → checkpoint keys (the tied token embedding is in the checkpoint
    under one of its names only — `umt5_export.checkpoint_weights` builds the map). The
    checkpoint's safetensors are opened with mmap, so the whole model is never held in f32 (layer
    by layer — decision 8 ①). MUST: a key without a mapping fails loudly (it never silently falls
    back to the container's value).
    """

    def __init__(self, checkpoint: CheckpointReader, keys: Mapping[str, str]) -> None:
        self._checkpoint = checkpoint
        self._keys = dict(keys)

    def _key(self, key: str) -> str:
        mapped = self._keys.get(key)
        if mapped is None:
            raise Umt5ReferenceError(f"no checkpoint key for '{key}'")
        return mapped

    def tensor(self, key: str) -> torch.Tensor:
        return self._checkpoint.read(self._key(key))

    def rows(self, key: str, indices: torch.Tensor) -> torch.Tensor:
        """Only rows `indices` (ascending, no duplicates) — read per run of consecutive rows."""
        wanted = indices.tolist()
        if wanted != sorted(set(wanted)) or not wanted or wanted[0] < 0:
            raise Umt5ReferenceError(
                f"'{key}': pass row indices ≥ 0 in ascending order without duplicates"
            )
        mapped = self._key(key)
        runs: list[tuple[int, int]] = []
        for row in wanted:
            if runs and runs[-1][1] == row:
                runs[-1] = (runs[-1][0], row + 1)
            else:
                runs.append((row, row + 1))
        chunks = [self._checkpoint.read_rows(mapped, start, stop) for start, stop in runs]
        gathered = torch.cat(chunks)
        if gathered.shape[0] != len(wanted):
            raise Umt5ReferenceError(f"'{key}': row {wanted[-1]} is outside the table")
        return gathered


# ---------------------------------------------------------------------------
# The spelled-out forward (shared by f64 / f32)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderShape:
    """The configuration the spelled-out code needs (from the upstream `UMT5Config`)."""

    layers: int
    heads: int
    head_dim: int
    eps: float

    @staticmethod
    def of(config: Any) -> EncoderShape:
        umt5_patch.check_supported(config)
        return EncoderShape(
            layers=int(config.num_layers),
            heads=int(config.num_heads),
            head_dim=int(config.d_kv),
            eps=float(config.layer_norm_epsilon),
        )


def layer_keys(index: int) -> dict[str, str]:
    """Tensor keys of layer `index`'s weights (name in the spelled-out code → name in the
    container)."""
    prefix = f"encoder.block.{index}.layer"
    attention = f"{prefix}.0.SelfAttention"
    dense = f"{prefix}.1.DenseReluDense"
    return {
        "attention_norm": f"{prefix}.0.layer_norm.weight",
        "q": f"{attention}.q.weight",
        "k": f"{attention}.k.weight",
        "v": f"{attention}.v.weight",
        "o": f"{attention}.o.weight",
        "relative_bias": f"{attention}.{umt5_patch.RELATIVE_BIAS_ATTRIBUTE}.weight",
        "ffn_norm": f"{prefix}.1.layer_norm.weight",
        "wi_0": f"{dense}.wi_0.weight",
        "wi_1": f"{dense}.wi_1.weight",
        "wo": f"{dense}.wo.weight",
    }


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`UMT5LayerNorm` spelled out (without upstream's cast of the variance to f32 — the identity
    in f32)."""
    variance = hidden.pow(2).mean(-1, keepdim=True)
    return weight * (hidden * torch.rsqrt(variance + eps))


def self_attention(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    buckets: torch.Tensor,
    shape: EncoderShape,
) -> torch.Tensor:
    """`UMT5LayerSelfAttention` spelled out (no scale, the bias is a gather from the table, softmax
    in the activation's dtype)."""
    normed = rms_norm(hidden, weights["attention_norm"], shape.eps)
    batch, length = normed.shape[:2]

    def heads_of(name: str) -> torch.Tensor:
        projected = functional.linear(normed, weights[name])
        return projected.view(batch, -1, shape.heads, shape.head_dim).transpose(1, 2)

    query, key, value = heads_of("q"), heads_of("k"), heads_of("v")
    scores = torch.matmul(query, key.transpose(3, 2))
    bias = functional.embedding(buckets, weights["relative_bias"]).permute([2, 0, 1]).unsqueeze(0)
    probabilities = functional.softmax(scores + bias, dim=-1)
    output = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
    return hidden + functional.linear(output.view(batch, length, -1), weights["o"])


def feed_forward(
    hidden: torch.Tensor, weights: Mapping[str, torch.Tensor], shape: EncoderShape
) -> torch.Tensor:
    """`UMT5LayerFF` spelled out (gated, with the tanh-approximated GELU — the decided form)."""
    normed = rms_norm(hidden, weights["ffn_norm"], shape.eps)
    gate = functional.gelu(functional.linear(normed, weights["wi_0"]), approximate="tanh")
    gated = gate * functional.linear(normed, weights["wi_1"])
    return hidden + functional.linear(gated, weights["wo"])


def encode_layerwise(
    source: WeightSource,
    shape: EncoderShape,
    inputs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    dtypes: Sequence[torch.dtype],
    *,
    on_layer: Callable[[int, float], None] | None = None,
) -> dict[torch.dtype, list[torch.Tensor]]:
    """Build every case's output `[1, L, d]` per dtype, layer by layer (the weights are read one
    layer at a time and dropped).

    `inputs` holds each case's token ids `[1, L]` and bucket table `[L, L]` (both int64).
    """
    for ids, buckets in inputs:
        length = ids.shape[1]
        if ids.shape[0] != 1 or tuple(buckets.shape) != (length, length):
            raise Umt5ReferenceError(f"input shapes {tuple(ids.shape)} / {tuple(buckets.shape)}")
    with torch.no_grad():
        vocabulary = torch.unique(torch.cat([ids.flatten() for ids, _ in inputs]))
        table = source.rows(EMBED_KEY, vocabulary)
        # The lookup happens after widening to dtype — not a single f32 operation on the f64 path
        # (watched by pytest).
        hidden = {
            dtype: [table.to(dtype)[torch.searchsorted(vocabulary, ids)] for ids, _ in inputs]
            for dtype in dtypes
        }
        for index in range(shape.layers):
            started = time.perf_counter()
            loaded = {name: source.tensor(key) for name, key in layer_keys(index).items()}
            for dtype in dtypes:
                weights = {name: value.to(dtype) for name, value in loaded.items()}
                hidden[dtype] = [
                    feed_forward(self_attention(state, weights, buckets, shape), weights, shape)
                    for state, (_, buckets) in zip(hidden[dtype], inputs, strict=True)
                ]
                del weights
            del loaded
            if on_layer is not None:
                on_layer(index, time.perf_counter() - started)
        final = source.tensor(FINAL_NORM_KEY)
        return {
            dtype: [rms_norm(state, final.to(dtype), shape.eps) for state in states]
            for dtype, states in hidden.items()
        }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceCase:
    """The case of one golden (the id sequence taken through the upstream path)."""

    name: str
    role: Role
    #: The original text before preprocessing (TS passes this value of the golden through the
    #: tokenizer).
    prompt: str
    #: The output of upstream `prompt_clean`.
    cleaned: str
    ids: tuple[int, ...]
    #: Provenance (the parity fixture's case ids, the fixed prompt's name).
    source: Mapping[str, Any] = field(default_factory=dict)


#: Original text → the outcome of the upstream path (`cleaned` and `ids` when accepted, `reject`
#: when rejected — the form of `umt5_tokenizer.build_case` of 10a).
Encode = Callable[[str], Mapping[str, Any]]


def upstream_encoder(model: str) -> Encode:
    """Turns original text into an id sequence through the same path as the 10a fixture (upstream
    `prompt_clean` agreeing with its mirror → transformers 5.14.1 agreeing with `tokenizer.json`
    on both paths)."""
    from wan import prompt_clean as pc
    from wan import umt5_tokenizer as ut

    truth = ut.load_truth(ut.tokenizer_snapshot(model))
    compiled = pc.CompiledTables(pc.build_tables())
    clean_truth = pc.upstream_prompt_clean()
    return lambda text: ut.build_case(truth, compiled, clean_truth, "reference", text, "reference")


def _accepted(outcome: Mapping[str, Any], where: str) -> tuple[str, tuple[int, ...]]:
    if "ids" not in outcome:
        raise Umt5ReferenceError(
            f"{where}: the upstream path rejected it ({outcome.get('reject')})"
        )
    return str(outcome["cleaned"]), tuple(int(token) for token in outcome["ids"])


def _join(members: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(str(member["text"]) for member in members)


def band_cases(
    parity_cases: Sequence[Mapping[str, Any]], encode: Encode, seed: int = BAND_SEED
) -> list[ReferenceCase]:
    """The 6 deciding cases (how they are chosen is in the module doc). Also checks that the id
    sequences of the single cases match the parity fixture (catches a mix-up between the fixture
    and the upstream path)."""
    pool = [case for case in parity_cases if case["id"].startswith(RANDOM_PREFIX) and "ids" in case]
    rng = random.Random(seed)
    singles: list[Mapping[str, Any]] = []
    for low, high in BAND_SINGLE_LENGTHS:
        candidates = [
            case for case in pool if low <= len(case["ids"]) <= high and case not in singles
        ]
        if not candidates:
            raise Umt5ReferenceError(f"no accepted random case of length {low} to {high}")
        singles.append(rng.choice(candidates))
    rest = [case for case in pool if case not in singles]
    rng.shuffle(rest)
    feed: Iterator[Mapping[str, Any]] = iter(rest)
    composites: list[tuple[list[Mapping[str, Any]], Mapping[str, Any]]] = []
    for low, high in BAND_COMPOSITE_LENGTHS:
        members: list[Mapping[str, Any]] = []
        outcome: Mapping[str, Any] = {"ids": []}
        while len(outcome["ids"]) < low:
            candidate = next(feed, None)
            if candidate is None:
                raise Umt5ReferenceError(
                    f"the random cases ran out before reaching length {low} to {high}"
                )
            trial = encode(_join([*members, candidate]))
            if "ids" in trial and len(trial["ids"]) <= high:
                members.append(candidate)
                outcome = trial
        composites.append((members, outcome))
    cases: list[ReferenceCase] = []
    for single in singles:
        cleaned, ids = _accepted(encode(single["text"]), single["id"])
        if list(ids) != list(single["ids"]):
            raise Umt5ReferenceError(
                f"{single['id']}: the upstream id sequence differs from the parity fixture"
            )
        cases.append(_band_case(single["text"], cleaned, ids, [single["id"]], seed))
    for members, outcome in composites:
        cleaned, ids = _accepted(outcome, "composite")
        cases.append(_band_case(_join(members), cleaned, ids, [m["id"] for m in members], seed))
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise Umt5ReferenceError(f"the names (lengths) of deciding cases collide: {names}")
    return cases


def _band_case(
    prompt: str, cleaned: str, ids: tuple[int, ...], members: list[str], seed: int
) -> ReferenceCase:
    return ReferenceCase(
        name=f"band-l{len(ids):04d}",
        role="band",
        prompt=prompt,
        cleaned=cleaned,
        ids=ids,
        source={"parityFixture": PARITY_FIXTURE, "parityCases": members, "seed": seed},
    )


def accept_cases(encode: Encode) -> list[ReferenceCase]:
    """The 4 acceptance cases (the fixed prompts — also checks that the lengths match the 10a
    values)."""
    from wan.prompts import FIXED_PROMPTS

    cases = []
    for prompt in FIXED_PROMPTS:
        cleaned, ids = _accepted(encode(prompt.text), prompt.name)
        if len(ids) != FIXED_PROMPT_LENGTHS[prompt.name]:
            raise Umt5ReferenceError(f"{prompt.name}: length {len(ids)} differs from the 10a value")
        cases.append(
            ReferenceCase(
                name=f"accept-{prompt.name}",
                role="accept",
                prompt=prompt.text,
                cleaned=cleaned,
                ids=ids,
                source={"fixedPrompt": prompt.name},
            )
        )
    return cases


def reference_cases(
    encode: Encode, parity: Path = FIXTURE_DIR / PARITY_FIXTURE
) -> list[ReferenceCase]:
    """6 deciding cases + 4 acceptance cases."""
    fixture = json.loads(parity.read_text(encoding="utf-8"))
    return [*band_cases(fixture["cases"], encode), *accept_cases(encode)]


# ---------------------------------------------------------------------------
# golden
# ---------------------------------------------------------------------------


def reference_ratios(f64: torch.Tensor, f32: torch.Tensor) -> dict[str, float]:
    """The normalization denominator and the effect of storing f64 rounded to f32 (every ratio is
    `umt5_patch.max_ratio` = max absolute difference ÷ max absolute value of the reference — the
    same form as TS `ratioOf`).

    - `f32VsF64`: the ratio of the CPU f32 reference against the f64 reference (the true
      denominator)
    - `f32VsStoredF64`: the same ratio taken against the stored value (f64 rounded to f32) (the
      denominator TS divides by)
    - `storedF64Rounding`: the ratio of the storage rounding itself (≤ 2⁻²⁴ ≈ 6e-8)
    """
    stored = f64.to(torch.float32)
    return {
        "f32VsF64": umt5_patch.max_ratio(f32, f64),
        "f32VsStoredF64": umt5_patch.max_ratio(f32, stored),
        "storedF64Rounding": umt5_patch.max_ratio(stored, f64),
    }


def row_quality(actual: torch.Tensor, base: torch.Tensor) -> dict[str, float]:
    """The 3 metrics of the quality record (decision 8 ② — minimum per-row cosine similarity,
    relative Frobenius norm, max absolute difference)."""
    left = actual.double().reshape(-1, actual.shape[-1])
    right = base.double().reshape(-1, base.shape[-1])
    cosine = functional.cosine_similarity(left, right, dim=-1)
    return {
        "minRowCosine": float(cosine.min()),
        "relativeFrobenius": float((left - right).norm() / right.norm()),
        "maxAbs": float((left - right).abs().max()),
    }


def _check_unquantized(case: ReferenceCase, unquantized: object | None) -> None:
    """The unquantized reference exists for acceptance cases only (never mixed into deciding
    cases, never missing from acceptance cases)."""
    if (case.role == "accept") != (unquantized is not None):
        raise Umt5ReferenceError(
            f"{case.name}: only acceptance cases have the unquantized reference (role {case.role})"
        )


def golden_tensors(
    case: ReferenceCase,
    buckets: torch.Tensor,
    f64: torch.Tensor,
    f32: torch.Tensor,
    unquantized: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """The golden tensors of one case (the boundary i64 → i32 goes through
    `normalize_boundary_tensor`).

    `unquantized` is the (f64, f32) of the unquantized reference (acceptance only —
    {@link _check_unquantized}).
    """
    _check_unquantized(case, unquantized)
    ids = torch.tensor(case.ids, dtype=torch.long)
    tensors = {
        INPUT_IDS_KEY: normalize_boundary_tensor(ids, f"{case.name}: {INPUT_IDS_KEY}"),
        BUCKETS_KEY: normalize_boundary_tensor(buckets, f"{case.name}: {BUCKETS_KEY}"),
        OUTPUT_F64_KEY: f64.to(torch.float32).contiguous(),
        OUTPUT_F32_KEY: f32.to(torch.float32).contiguous(),
    }
    if unquantized is not None:
        high, low = unquantized
        tensors[OUTPUT_UNQUANTIZED_F64_KEY] = high.to(torch.float32).contiguous()
        tensors[OUTPUT_UNQUANTIZED_F32_KEY] = low.to(torch.float32).contiguous()
    return tensors


def golden_metadata(
    case: ReferenceCase,
    weights: ContainerWeights,
    ratios: Mapping[str, float],
    versions: Mapping[str, str],
    unquantized_ratios: Mapping[str, float] | None = None,
    *,
    axes: Mapping[str, Any],
) -> dict[str, Any]:
    """The golden meta of one case. `axes` records which upstream values were used (the umT5
    upstream and the Wan text-side snapshot the case's id sequence came from —
    `wan.umt5_export.reference_axes`, ADR 0122 decision 3)."""
    _check_unquantized(case, unquantized_ratios)
    unquantized = (
        {}
        if unquantized_ratios is None
        else {
            "unquantized": {
                "weights": (
                    "pin した checkpoint の F32 をそのまま（量子化しない）・"
                    "f64 はその値を広げたもの"
                ),
                "output": "f64 は f32 へ丸めて格納・品質の記録の基準（門ではない）",
                "ratios": dict(unquantized_ratios),
            }
        }
    )
    return {
        "format": REFERENCE_FORMAT,
        "case": case.name,
        "role": case.role,
        "prompt": case.prompt,
        "cleaned": case.cleaned,
        "length": len(case.ids),
        "source": dict(case.source),
        "axes": {name: dict(axis) for name, axis in axes.items()},
        "container": {"part0Sha256": weights.part0_sha256, "parts": weights.parts},
        "weights": (
            "i8 は packed × 行ごとの scale を f32 で掛けた値（fake-quant と同じ）・"
            "F32 はそのまま・f64 はその f32 の値を広げたもの"
        ),
        "activation": "ゲート付き GELU の tanh 近似（nn.GELU(approximate='tanh')）",
        "outputF64": "f64 の参照を f32 へ丸めて格納（TS の safetensors は F64 を読まない）",
        "ratios": dict(ratios),
        "versions": dict(versions),
        **unquantized,
    }


def write_golden(
    path: Path, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]
) -> int:
    payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    save_file(dict(tensors), str(path), metadata={METADATA_KEY: payload})
    return path.stat().st_size


def _versions() -> dict[str, str]:
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__}


#: The hook that measures a stage (`umt5_export.MemoryMonitor.stage`).
Stage = Callable[[str], AbstractContextManager[Any]]


def write_references(
    *,
    container: Path,
    out_dir: Path,
    config: Any,
    cases: Sequence[ReferenceCase],
    stage: Stage,
    unquantized: WeightSource,
    axes: Mapping[str, Any],
    embeddings: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Take the layer-by-layer f64 / f32 references and write the goldens to `out_dir` (no
    container is written).

    `axes` is the axis record kept in the golden meta ({@link golden_metadata}).

    The i8 reference (the container's weights) is taken for every case, the unquantized reference
    (`unquantized` — {@link CheckpointWeights}) for acceptance cases only. The i8 run takes every
    case through one layer-by-layer pass (the same call as before acceptance was added — existing
    values do not change by a byte).

    `embeddings` are the precomputed bf16 assets (fixed prompt name → `[L, 4096]`). When given, the
    summaries of the acceptance cases gain the difference from the unquantized f32 reference
    (record only — the source of truth for the quality record is the TS e2e).
    """
    shape = EncoderShape.of(config)
    attention = umt5_patch.bucket_attention(config)
    weights = ContainerWeights(container)
    inputs = [
        (
            torch.tensor([case.ids], dtype=torch.long),
            umt5_patch.relative_position_buckets(len(case.ids), attention),
        )
        for case in cases
    ]
    accepted = [index for index, case in enumerate(cases) if case.role == "accept"]
    layer_seconds: dict[str, list[float]] = {"i8": [], "unquantized": []}

    def on_layer(label: str) -> Callable[[int, float], None]:
        def report(index: int, seconds: float) -> None:
            layer_seconds[label].append(round(seconds, 1))
            print(f"[layer] {label} {index + 1}/{shape.layers} {seconds:.1f} s", flush=True)

        return report

    dtypes = (torch.float64, torch.float32)
    with stage("reference"):
        outputs = encode_layerwise(weights, shape, inputs, dtypes, on_layer=on_layer("i8"))
    with stage("reference-unquantized"):
        plain = encode_layerwise(
            unquantized,
            shape,
            [inputs[index] for index in accepted],
            dtypes,
            on_layer=on_layer("unquantized"),
        )
    unquantized_of = {
        index: (plain[torch.float64][at], plain[torch.float32][at])
        for at, index in enumerate(accepted)
    }
    versions = _versions()
    rows: list[dict[str, Any]] = []
    with stage("write"):
        for index, case in enumerate(cases):
            f64 = outputs[torch.float64][index]
            f32 = outputs[torch.float32][index]
            ratios = reference_ratios(f64, f32)
            pair = unquantized_of.get(index)
            plain_ratios = None if pair is None else reference_ratios(*pair)
            path = out_dir / f"{REFERENCE_PREFIX}{case.name}{CASE_SUFFIX}"
            size = write_golden(
                path,
                golden_tensors(case, inputs[index][1], f64, f32, pair),
                golden_metadata(case, weights, ratios, versions, plain_ratios, axes=axes),
            )
            row: dict[str, Any] = {
                "case": case.name,
                "role": case.role,
                "length": len(case.ids),
                "bytes": size,
                **ratios,
            }
            if pair is not None:
                row["unquantizedRatios"] = plain_ratios
                # The effect of the i8 rounding (CPU — no GPU error included).
                row["i8F32VsUnquantizedF32"] = row_quality(f32[0], pair[1][0])
            fixed = case.source.get("fixedPrompt")
            if embeddings is not None and fixed is not None:
                if pair is None:
                    raise Umt5ReferenceError(
                        f"{case.name}: a fixed prompt that is not an acceptance case"
                    )
                # The effect of bf16 (the asset path) and the difference from the i8 f32
                # reference (continuing the 10c record).
                row["bf16AssetVsUnquantizedF32"] = row_quality(embeddings[fixed], pair[1][0])
                row["bf16AssetVsF32"] = row_quality(embeddings[fixed], f32[0])
            rows.append(row)
            print(f"[golden] {json.dumps(row, ensure_ascii=False)}", flush=True)
    return {
        "container": {"part0Sha256": weights.part0_sha256, "parts": weights.parts},
        "cases": rows,
        "layerSeconds": layer_seconds,
        "goldenBytes": sum(row["bytes"] for row in rows),
    }
