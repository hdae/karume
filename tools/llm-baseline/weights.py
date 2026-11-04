"""Put the stored weights back into the official Transformers model. Does not requantize."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
)

from karume.container import ReadContainer, codec_entry
from karume.emit import unpack_int4
from karume.ple import (
    PLE_FLOAT_INDEX_SCHEMA,
    PLE_GROUPED_INDEX_SCHEMA,
    PLE_INDEX_ASSET,
    PLE_INDEX_SCHEMA,
    ple_row_bytes,
)
from karume.quantize import dequantize_int4
from karume.verify import InitializerSupply, SupplyBlock, verify_container

#: Decode path of the codec ledger (`codec_entry(...).layout`) → torch dtype holding the raw bytes.
#: i4 / i2 are packed, so they are passed as a flat uint8 holder (restoring the logical shape is
#: the expansion's job).
STORAGE_DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "i32": torch.int32,
    "i8": torch.int8,
    "i4": torch.uint8,
    "i2": torch.uint8,
}

#: Component name = graph name the container declares = the weights key of the manifest
#: (container-v1 §2.1).
MODEL_GRAPH = "model"


def fingerprint(path: Path) -> dict:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


@dataclass(frozen=True)
class AssetSlice:
    """Location of one asset. An asset that needs range reads is **1 block = 1 part**
    (container-v1 §4.2)."""

    path: Path
    offset: int
    length: int


class ContainerAssets:
    """Access to the container's assets (the index whole + sha256 check; PLE rows by range read).

    MUST: rows are read by range. A PLE block can be up to 32 MiB, so reading the whole block per
    token makes one evaluation orders of magnitude slower (the old sidecar did the same with a
    safetensors mmap slice). A range read cannot apply the block's sha256, but the declaration
    (lengths, part assignment) has already passed {@link verify_container}.
    """

    def __init__(self, read: ReadContainer, parts: Sequence[Path]) -> None:
        if read.model is None:
            raise ValueError("a graph-only container (krg) has no assets")
        if len(parts) < 2:
            raise ValueError(
                "single-form containers are not supported (needs the split form from part 0)"
            )
        self._read = read
        self._records = read.model.assets
        self._blocks = {block.id: block for block in read.model.blocks}
        self._parts = [Path(part) for part in parts]

    def __contains__(self, name: str) -> bool:
        return name in self._records

    def _record(self, name: str):
        record = self._records.get(name)
        if record is None:
            raise ValueError(f"container has no asset '{name}' (declared: {sorted(self._records)})")
        return record

    def document(self, name: str) -> bytes:
        """Take one asset whole (checked up to the block's sha256 — for the index and QAT check)."""
        record = self._record(name)
        return self._read.block(record.block)[: record.length]

    def slice_of(self, name: str) -> AssetSlice:
        record = self._record(name)
        block = self._blocks[record.block]
        return AssetSlice(self._parts[block.part], block.offset, record.length)

    def read_row(self, name: str, offset: int, length: int) -> bytes:
        """The range `[offset, offset + length)` inside the asset."""
        where = self.slice_of(name)
        if offset < 0 or length < 0 or offset + length > where.length:
            raise ValueError(f"{name}: range [{offset}, {offset + length}) is outside the asset")
        with where.path.open("rb") as source:
            source.seek(where.offset + offset)
            raw = source.read(length)
        if len(raw) != length:
            raise ValueError(f"{where.path}: reading {name} fell short of {length} bytes")
        return raw


class StoredWeights:
    """Read the baked bodies back as is from the distribution's container (the part sequence of
    `krm`).

    Replaces the reader of the old shard sequence (ADR 0109 decision 3 / container-v1). One
    container is one component, and **the initializer name is the tensor key itself** (IR v2). Piece
    splitting closes inside the binding table, so the caller sees a single tensor.
    """

    def __init__(self, parts: Sequence[Path], graph_name: str = MODEL_GRAPH) -> None:
        if len(parts) < 2:
            raise ValueError(
                "a split-form container (part sequence ordered from part 0) is required"
            )
        verified = verify_container(list(parts))
        self.read = verified.read
        declaration = self.read.graph.graphs.get(graph_name)
        if declaration is None:
            declared = sorted(self.read.graph.graphs)
            raise ValueError(f"container has no graph '{graph_name}' (declared: {declared})")
        #: The IR v2 graph declaration itself (`values` / `nodes` / `initializers`).
        self.graph = declaration
        #: Supply plan of the initializers that have a body (`shared` declarations are not in it).
        self.supplies = verified.graphs[graph_name].supplies
        self.assets = ContainerAssets(self.read, parts)

    def _supply(self, name: str) -> InitializerSupply:
        supply = self.supplies.get(name)
        if supply is None:
            raise ValueError(f"the container has no body for initializer '{name}'")
        return supply

    def _payload(self, block: SupplyBlock) -> bytes:
        """Payload of a block (drops the trailing padding — container-v1 §4.1)."""
        return self.read.block(block.id)[: block.payload_bytes]

    def shape(self, name: str) -> list[int]:
        return list(self.graph["values"][name]["shape"])

    def storage(self, name: str) -> str:
        """Decode path of the storage (`i8` / `i4` / `i2` / `f32` …). The source of truth is the
        binding table's codec."""
        return codec_entry(self._supply(name).encoding.codec).layout

    def tensor(self, name: str) -> torch.Tensor:
        """The baked body as is (i4 / i2 as packed flat uint8, others in the logical shape)."""
        supply = self._supply(name)
        layout = codec_entry(supply.encoding.codec).layout
        data = bytearray(b"".join(self._payload(block) for block in supply.blocks))
        value = torch.frombuffer(data, dtype=STORAGE_DTYPES[layout])
        return value if layout in ("i4", "i2") else value.reshape(self.shape(name))

    def scale(self, name: str) -> torch.Tensor:
        """Return the companion scale as f32 `[row, group]`.

        In the container the scale is a **block without a name**, reachable only through
        `encoding.scale` of the binding table (the old form's "one more tensor key" is gone).
        """
        supply = self._supply(name)
        if supply.scale is None:
            raise ValueError(f"initializer '{name}' is not quantized (no scale)")
        if supply.encoding.row_axis not in (None, 0):
            raise ValueError(f"'{name}': rowAxis {supply.encoding.row_axis} is not supported")
        data = bytearray(self._payload(supply.scale))
        return torch.frombuffer(data, dtype=torch.float32).reshape(self.shape(name)[0], -1)

    def dequantized(self, name: str) -> torch.Tensor:
        layout = self.storage(name)
        value = self.tensor(name)
        if layout == "i4":
            return dequantize_int4(unpack_int4(value, self.shape(name)), self.scale(name))
        if layout == "i8":
            scale = self.scale(name)
            if scale.numel() != value.shape[0]:
                raise ValueError("i8 scale count differs from the row count")
            return value.float() * scale.reshape(-1, *([1] * (value.ndim - 1)))
        if layout not in ("f16", "f32", "bf16"):
            raise ValueError(f"unsupported stored dtype: {layout}")
        return value.float()

    def ple_index(self) -> dict:
        """PLE index (asset `ple_index` — schema 3, 4 or 5; {@link ContainerPle} checks it)."""
        return json.loads(self.assets.document(PLE_INDEX_ASSET))

    def ple(self, embed_scale: float) -> ContainerPle:
        return ContainerPle(self.ple_index(), self.assets.read_row, embed_scale)


def checkpoint_index(checkpoint: Path) -> dict[str, Path]:
    owners = {}
    for path in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(path, framework="pt") as source:
            for key in source.keys():  # noqa: SIM118 — safe_open is not a dict.
                if key in owners:
                    raise ValueError(f"duplicate checkpoint key: {key}")
                owners[key] = path
    if not owners:
        raise ValueError(f"{checkpoint}: no weights")
    return owners


class DiskPle(nn.Module):
    """Does the same row lookup on the CPU from the official checkpoint without converting the huge
    PLE to f32 whole."""

    def __init__(self, checkpoint: Path, scale: float):
        super().__init__()
        self.scale = scale
        self.key = "model.language_model.embed_tokens_per_layer.weight"
        self.file = checkpoint_index(checkpoint)[self.key]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.device.type != "cpu":
            raise ValueError("DiskPle is for CPU evaluation only")
        rows = {}
        for token in ids.flatten().tolist():
            if token in rows:
                continue
            with safe_open(self.file, framework="pt") as source:
                row = source.get_slice(self.key)[token : token + 1].float()
            rows[token] = row.flatten() * self.scale
        return torch.stack([rows[token] for token in ids.flatten().tolist()]).reshape(
            *ids.shape, -1
        )


#: Index schema → the PLE storage this CPU reference restores, and the index keys that schema spells
#: (the reader's source of truth is `packages/models/src/gemma/ple-index.ts`, the writer's
#: `karume.ple`). Schema 3 = one f32 scale per (token, layer) row, schema 4 = one per group of
#: `groupSize` values (the q4_0 QAT line's 4-bit grid), schema 5 = a bf16 table with no scales (the
#: regular line's original weights).
#:
#: MUST: schema 3 restores `i8` only. The mobile QAT line's PLE (schema 3, i4 / i2) **is the
#: official fixed quantization itself**, and the check this tool applies to it is raw-byte equality
#: ({@link assert_qat_stored}), not restoration — a second expansion here would split what to
#: compare against in two. Schema 4 restores `i4` alone, the one storage a shipped line writes
#: there.
_PLE_SHAPE_KEYS = frozenset(
    {"schema", "storage", "tokens", "layers", "dim", "embedScale", "values"}
)
PLE_SCHEMAS: dict[int, tuple[str, frozenset[str]]] = {
    PLE_INDEX_SCHEMA: ("i8", _PLE_SHAPE_KEYS | {"scales"}),
    PLE_GROUPED_INDEX_SCHEMA: ("i4", _PLE_SHAPE_KEYS | {"scales", "groupSize"}),
    PLE_FLOAT_INDEX_SCHEMA: ("bf16", _PLE_SHAPE_KEYS),
}


class ContainerPle(nn.Module):
    """Looks up PLE rows from the container's assets (`ple_index` of schema 3 / 4 / 5 +
    `ple.values.<k>`, and `ple.scales.<k>` for a quantized table).

    `read_row(asset name, offset, length)` is the range-read access, so that a caller without a
    container (the tests) can substitute it together with a synthetic index.

    MUST: a row is restored exactly as the host gather of `packages/models/src/gemma/ple.ts` builds
    it, with no requantization: a quantized value is `(q × scale) × embedScale` in f32 (the scale of
    its group — schema 3's group is the whole layer row), a bf16 value is its bits widened to f32,
    then `× embedScale`. The i4 nibble order is `karume.emit.unpack_int4`'s (element `2i` in the
    low nibble, stored `q + 8`).

    MUST: unknown schemas, a storage the schema does not restore ({@link PLE_SCHEMAS}), keys the
    schema does not spell (a `groupSize` in schema 3, `scales` in schema 5) or a missing one, and a
    `rowBytes` the declaration does not determine all fail loudly — each would otherwise read
    another token's row, or another group's step, with the shape still matching.
    """

    def __init__(self, index: dict, read_row: Callable[[str, int, int], bytes], embed_scale: float):
        super().__init__()
        schema = index.get("schema")
        if schema not in PLE_SCHEMAS:
            raise ValueError(
                f"PLE index schema is {schema} (expected one of {sorted(PLE_SCHEMAS)})"
            )
        storage, keys = PLE_SCHEMAS[schema]
        if set(index) != keys:
            raise ValueError(
                f"PLE index schema {schema}: keys missing {sorted(keys - set(index))} /"
                f" unknown {sorted(set(index) - keys)}"
            )
        if index["storage"] != storage:
            raise ValueError(
                f"PLE storage '{index['storage']}' of schema {schema} is not restored by the CPU"
                f" reference ({storage} only)"
            )
        if index["embedScale"] != embed_scale:
            raise ValueError("embedScale of the regular PLE differs from the official one")
        self.layers = index["layers"]
        self.dim = index["dim"]
        # Schema 3's group is the whole layer row (unused for bf16, which has no scales).
        self.group_size = index.get("groupSize", self.dim)
        # The writer's row arithmetic (`karume.ple.ple_row_bytes`) is the declaration's: it also
        # refuses a group that does not divide `dim`.
        expected = ple_row_bytes(storage, self.layers, self.dim, index.get("groupSize"))
        for kind, stride in expected.items():
            if index[kind]["rowBytes"] != stride:
                raise ValueError(
                    f"PLE {kind}.rowBytes {index[kind]['rowBytes']} differs from {stride},"
                    " which the declaration determines"
                )
        self.index = index
        self.read_row = read_row
        self.scale = embed_scale
        self.storage = storage

    def _row(self, kind: str, token: int) -> bytes:
        entry = self.index[kind]
        stride = entry["rowBytes"]
        for block in entry["blocks"]:
            if block["start"] <= token < block["stop"]:
                return self.read_row(block["asset"], (token - block["start"]) * stride, stride)
        raise ValueError(f"token {token} is not covered by any PLE {kind} block")

    def _restore(self, token: int) -> torch.Tensor:
        """One token's row `[layers × dim]` in f32 (the class docstring's order)."""
        raw = bytearray(self._row("values", token))
        if self.storage == "bf16":
            return torch.frombuffer(raw, dtype=torch.bfloat16).float() * self.scale
        if self.storage == "i4":
            quant = unpack_int4(torch.frombuffer(raw, dtype=torch.uint8), [self.layers, self.dim])
        else:
            quant = torch.frombuffer(raw, dtype=torch.int8).reshape(self.layers, self.dim)
        groups = self.dim // self.group_size
        scale = torch.frombuffer(bytearray(self._row("scales", token)), dtype=torch.float32)
        grouped = quant.float().reshape(self.layers, groups, self.group_size)
        values = grouped * scale.reshape(self.layers, groups, 1)
        return values.flatten() * self.scale

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.device.type != "cpu":
            raise ValueError("ContainerPle is for CPU evaluation only")
        rows = {}
        for token in ids.flatten().tolist():
            if token not in rows:
                rows[token] = self._restore(token)
        return torch.stack([rows[token] for token in ids.flatten().tolist()]).reshape(
            *ids.shape, -1
        )


def ple_placement(module: nn.Module) -> str:
    """Names the row-lookup route of the installed PLE (the record is derived from the installed
    object, not from the branch decision)."""
    if isinstance(module, ContainerPle):
        return "container-row-lookup"
    if isinstance(module, DiskPle):
        return "disk-row-lookup"
    raise TypeError(f"unknown module for the PLE route: {type(module).__name__}")


def put(model: nn.Module, name: str, value: torch.Tensor) -> None:
    parent, _, leaf = name.rpartition(".")
    module = model.get_submodule(parent)
    old = getattr(module, leaf)
    if old.shape != value.shape:
        raise ValueError(f"{name}: {value.shape} != {old.shape}")
    if isinstance(old, nn.Parameter):
        setattr(module, leaf, nn.Parameter(value, requires_grad=False))
    else:
        setattr(module, leaf, value)


def load_float_model(
    checkpoint: Path, family: str, stored: StoredWeights | None
) -> tuple[nn.Module, dict]:
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    if family == "gemma4":
        config = config.get_text_config()
    config._attn_implementation = "eager"
    with init_empty_weights():
        model = (
            Gemma4ForCausalLM(config)
            if family == "gemma4"
            else AutoModelForCausalLM.from_config(config)
        )
    if family == "gemma4":
        scale = float(model.model.embed_tokens_per_layer.embed_scale)
        # The PLE of the stored weights is **an asset of the same container** (ADR 0109 decision 4)
        # — it replaces the official checkpoint's row lookup only with `--weights stored`.
        model.model.embed_tokens_per_layer = (
            DiskPle(checkpoint, scale) if stored is None else stored.ple(scale)
        )
    parameters = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers())
    loaded = set()

    def assign(name: str, value: torch.Tensor) -> None:
        names = [name]
        if config.tie_word_embeddings and name in ("lm_head.weight", "model.embed_tokens.weight"):
            names = ["model.embed_tokens.weight", "lm_head.weight"]
        for target in names:
            if target in loaded and not torch.equal(model.get_parameter(target), value):
                raise ValueError(
                    f"{target}: stored values of the tied weight contradict each other"
                )
            put(model, target, value)
            if target in parameters:
                loaded.add(target)

    if stored is None:
        owners = checkpoint_index(checkpoint)
        for name in parameters:
            if name in loaded:
                continue
            key = (
                ("model.language_model." + name.removeprefix("model."))
                if family == "gemma4"
                else name
            )
            if family == "gemma4" and name == "lm_head.weight":
                key = "model.language_model.embed_tokens.weight"
            with safe_open(owners[key], framework="pt") as source:
                assign(name, source.get_tensor(key).float())
        for name in buffers:
            key = (
                ("model.language_model." + name.removeprefix("model."))
                if family == "gemma4"
                else name
            )
            if key in owners:
                with safe_open(owners[key], framework="pt") as source:
                    assign(name, source.get_tensor(key).float())
    else:
        # In IR v2 the initializer name is the tensor key itself (formerly `entry["tensor"]`).
        for name in stored.supplies:
            if not name.startswith("model.") or name.endswith(("cos_table", "sin_table")):
                continue
            target = name.removeprefix("model.")
            if target not in parameters and target not in buffers:
                raise ValueError(
                    f"the official model has no counterpart for the stored weight: {target}"
                )
            assign(target, stored.dequantized(name))
    if loaded != set(parameters):
        raise ValueError(f"parameters not loaded: {set(parameters) - loaded}")
    if any(t.device.type == "meta" for t in [*model.parameters(), *model.buffers()]):
        raise ValueError("meta tensors remain")
    model.eval()
    return model, {
        "loadedParameterNames": sorted(loaded),
        "attention": "eager",
        "dtype": "float32",
        "ple": ple_placement(model.model.embed_tokens_per_layer) if family == "gemma4" else None,
    }


def assert_srq_scales(graph: dict, weight: str, input_scale: float, output_scale: float) -> int:
    """Matches the fixed rounding before and after the linear op against the official positive
    calibration values."""
    linears = [
        node for node in graph["nodes"] if node["op"] == "linear" and node["ins"][1] == weight
    ]
    if len(linears) != 1:
        raise ValueError(f"{weight}: linear op is not unique")
    linear = linears[0]
    checked = 0
    for direction, scale in [("input", input_scale), ("output", output_scale)]:
        if scale == 0:
            continue  # Uncalibrated is the identity officially too; adjacent rounding is allowed.
        nodes = [
            node
            for node in graph["nodes"]
            if node["op"] == "static_quantize"
            and (
                node["outs"] == [linear["ins"][0]]
                if direction == "input"
                else node["ins"] == linear["outs"]
            )
        ]
        if len(nodes) != 1 or nodes[0]["attrs"]["scale"] != scale:
            raise ValueError(f"{weight}: {direction} SRQ scale differs from the official one")
        checked += 1
    return checked


def assert_qat_checked_range(graph: dict, supplies: Iterable[str], prefix: str = "model.") -> None:
    """Checks that the set the check iterates (the supply plan) **equals** the graph declaration's.

    {@link assert_qat_stored} iterates `stored.supplies` (= initializers with a body in the
    container). A `shared` declaration, which has no body in the container (borrowing the lender's
    bytes — ADR 0096 decision 2 and its stage-2 addendum of 2026-09-08), is not in it, so the moment
    a declaration is rewritten to `shared`, that weight **silently leaves the checked set**. Only
    the count (`fixedInitializersChecked`) drops quietly and nothing turns red, so the counterpart
    check of `loaded != set(parameters)` in `load_float_model` lives here.

    MUST: check both directions (declared but not supplied = left out when written / supplied but
    not declared = reading the body of another graph).
    """
    declared = {
        name
        for name, initializer in graph["initializers"].items()
        if name.startswith(prefix) and not initializer.get("shared", False)
    }
    iterated = {name for name in supplies if name.startswith(prefix)}
    if declared != iterated:
        raise ValueError(
            "QAT fixed-quantization check range differs from the graph declaration"
            f" (declared only: {sorted(declared - iterated)} /"
            f" supplied only: {sorted(iterated - declared)})"
        )


def assert_qat_stored(model: nn.Module, stored: StoredWeights) -> dict:
    from transformers.integrations.gemma_quant import (
        QuantizedEmbedding,
        QuantizedLinear,
    )

    assert_qat_checked_range(stored.graph, stored.supplies)
    count = 0
    srq_count = 0
    for key in stored.supplies:
        if not key.startswith("model."):
            continue
        original = key.removeprefix("model.")
        if original.startswith("model."):
            original = "model.language_model." + original.removeprefix("model.")
        parent, _, leaf = original.rpartition(".")
        module = model.get_submodule(parent)
        dtype = stored.storage(key)
        if dtype in ("i2", "i4", "i8"):
            if leaf != "weight" or not isinstance(module, (QuantizedEmbedding, QuantizedLinear)):
                raise ValueError(f"{key}: no official fixed-quantization counterpart")
            packed = (
                module.embedding_quantized
                if isinstance(module, QuantizedEmbedding)
                else module.weight
            )
            scale = (
                module.embedding_scale
                if isinstance(module, QuantizedEmbedding)
                else module.weight_scale
            )
            actual = stored.tensor(key)
            if dtype != f"i{module.num_bits}" or not torch.equal(
                actual.reshape(packed.shape), packed
            ):
                raise ValueError(f"{key}: fixed integers differ")
            if isinstance(module, QuantizedLinear):
                srq_count += assert_srq_scales(
                    stored.graph,
                    key,
                    float(module.input_activation_scale.detach()),
                    float(module.output_activation_scale.detach()),
                )
            actual_scale = stored.scale(key)
            if not torch.equal(actual_scale, scale.expand_as(actual_scale)):
                raise ValueError(f"{key}: fixed scale differs")
        else:
            if not torch.equal(stored.dequantized(key), getattr(module, leaf).float()):
                raise ValueError(f"{key}: non-quantized values differ")
        count += 1
    # PLE is **an asset of the same container** (ADR 0109 decision 4). Index schema 3 cuts each row
    # kind differently, so `values` / `scales` are compared over their own block sequences (the old
    # sidecar had both in one shard).
    layout = stored.ple_index()
    embedding = model.model.language_model.embed_tokens_per_layer
    if (
        layout["storage"] != f"i{embedding.num_bits}"
        or layout["embedScale"] != embedding.scalar_embed_scale
    ):
        raise ValueError("QAT PLE declaration differs")
    ple_blocks = 0
    for kind, table in [
        ("values", embedding.embedding_quantized),
        ("scales", embedding.embedding_scale),
    ]:
        for block in layout[kind]["blocks"]:
            raw = stored.assets.document(block["asset"])
            expected = table[block["start"] : block["stop"]]
            if (
                hashlib.sha256(raw).digest()
                != hashlib.sha256(memoryview(expected.detach().numpy())).digest()
            ):
                raise ValueError(f"{block['asset']}: PLE {kind} differs from the official one")
            ple_blocks += 1
    return {
        "fixedInitializersChecked": count,
        "srqScalesChecked": srq_count,
        "pleBlocksChecked": ple_blocks,
    }


def load_qat(checkpoint: Path, stored: StoredWeights) -> tuple[nn.Module, dict]:
    model = Gemma4ForConditionalGeneration.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    ).eval()
    checks = assert_qat_stored(model, stored)
    return model, {
        **checks,
        "attention": "eager",
        "dtype": "float32",
        "srq": "official-transformers",
    }
