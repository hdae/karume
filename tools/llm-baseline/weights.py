"""公式 Transformers のモデルへ保存済み重みを戻す。再量子化しない。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from karume.emit import unpack_int4
from karume.quantize import dequantize_int4
from karume.shards import parse_piece_key
from karume.verify import _read_header
from safetensors import safe_open
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
)


def fingerprint(path: Path) -> dict:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


class StoredWeights:
    def __init__(self, shards: list[Path]):
        self.shards = shards
        self.owners = {}
        self.pieces = {}
        self.graph = None
        for path in shards:
            header, base, _ = _read_header(path)
            if self.graph is None:
                self.graph = json.loads(header["__metadata__"]["karume_ir"])
            for key, entry in header.items():
                if key == "__metadata__":
                    continue
                if key in self.owners:
                    raise ValueError(f"重複 tensor: {key}")
                self.owners[key] = (path, base, entry)
                piece = parse_piece_key(key)
                if piece is not None:
                    self.pieces.setdefault(piece[0], []).append((piece[1], key))
        if self.graph is None:
            raise ValueError("グラフがありません")

    def tensor(self, key: str) -> torch.Tensor:
        if key not in self.owners:
            return torch.cat([self.tensor(k) for _, k in sorted(self.pieces[key])], dim=0)
        path, base, entry = self.owners[key]
        low, high = entry["data_offsets"]
        with path.open("rb") as source:
            source.seek(base + low)
            data = bytearray(source.read(high - low))
        if len(data) != high - low:
            raise ValueError(f"{path}: {key} が途中で終わる")
        dtype = {
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I8": torch.int8,
            "I4": torch.uint8,
            "I2": torch.uint8,
            "U8": torch.uint8,
            "I32": torch.int32,
        }[entry["dtype"]]
        value = torch.frombuffer(data, dtype=dtype)
        return value if entry["dtype"] in ("I4", "I2") else value.reshape(entry["shape"])

    def dequantized(self, name: str, entry: dict) -> torch.Tensor:
        value = self.tensor(entry["tensor"])
        storage = entry["storage"]
        dtype = storage["dtype"]
        if dtype == "i4":
            return dequantize_int4(
                unpack_int4(value, self.graph["values"][name]["shape"]),
                self.tensor(storage["scale"]),
            )
        if dtype == "i8":
            scale = self.tensor(storage["scale"])
            if scale.numel() != value.shape[0]:
                raise ValueError("i8 scale が行数と違います")
            return value.float() * scale.reshape(-1, *([1] * (value.ndim - 1)))
        if dtype not in ("f16", "f32", "bf16"):
            raise ValueError(f"未対応の保存 dtype: {dtype}")
        return value.float()


def checkpoint_index(checkpoint: Path) -> dict[str, Path]:
    owners = {}
    for path in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(path, framework="pt") as source:
            for key in source.keys():  # noqa: SIM118 — safe_open は dict ではない。
                if key in owners:
                    raise ValueError(f"重複した checkpoint key: {key}")
                owners[key] = path
    if not owners:
        raise ValueError(f"{checkpoint}: 重みがありません")
    return owners


class DiskPle(nn.Module):
    """巨大な PLE を全量 f32 化せず、同じ行 lookup を CPU 上で行う。"""

    def __init__(self, checkpoint: Path, scale: float, index: Path | None):
        super().__init__()
        self.scale = scale
        self.index = index
        if index is None:
            self.key = "model.language_model.embed_tokens_per_layer.weight"
            self.file = checkpoint_index(checkpoint)[self.key]
        else:
            self.layout = json.loads(index.read_text())
            if self.layout["schema"] != 1 or self.layout["embedScale"] != scale:
                raise ValueError("通常版 PLE の形式・scale が違います")

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.device.type != "cpu":
            raise ValueError("DiskPle は CPU 評価専用です")
        rows = {}
        for token in ids.flatten().tolist():
            if token in rows:
                continue
            if self.index is None:
                with safe_open(self.file, framework="pt") as source:
                    row = source.get_slice(self.key)[token : token + 1].float()
            else:
                shard = next(s for s in self.layout["shards"] if s["start"] <= token < s["stop"])
                at = token - shard["start"]
                with safe_open(self.index.parent / shard["file"], framework="pt") as source:
                    quant = source.get_slice("values")[at : at + 1].float()
                    scale = source.get_slice("scales")[at : at + 1]
                    row = (quant * scale.unsqueeze(-1)).flatten(1)
            rows[token] = row.flatten() * self.scale
        return torch.stack([rows[token] for token in ids.flatten().tolist()]).reshape(
            *ids.shape, -1
        )


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
    checkpoint: Path, family: str, stored: StoredWeights | None, ple_index: Path | None
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
        model.model.embed_tokens_per_layer = DiskPle(checkpoint, scale, ple_index)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers())
    loaded = set()

    def assign(name: str, value: torch.Tensor) -> None:
        names = [name]
        if config.tie_word_embeddings and name in ("lm_head.weight", "model.embed_tokens.weight"):
            names = ["model.embed_tokens.weight", "lm_head.weight"]
        for target in names:
            if target in loaded and not torch.equal(model.get_parameter(target), value):
                raise ValueError(f"{target}: tied weight の保存値が矛盾しています")
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
        for name, entry in stored.graph["initializers"].items():
            key = entry["tensor"]
            if not key.startswith("model.") or key.endswith(("cos_table", "sin_table")):
                continue
            target = key.removeprefix("model.")
            if target not in parameters and target not in buffers:
                raise ValueError(f"公式モデルに保存重みの対応先がありません: {target}")
            assign(target, stored.dequantized(name, entry))
    if loaded != set(parameters):
        raise ValueError(f"未読込の parameter: {set(parameters) - loaded}")
    if any(t.device.type == "meta" for t in [*model.parameters(), *model.buffers()]):
        raise ValueError("meta tensor が残っています")
    model.eval()
    return model, {
        "loadedParameterNames": sorted(loaded),
        "attention": "eager",
        "dtype": "float32",
        "ple": "disk-row-lookup" if family == "gemma4" else None,
    }


def assert_srq_scales(graph: dict, weight: str, input_scale: float, output_scale: float) -> int:
    """線形演算の前後にある固定丸めを、公式の正の校正値と照合する。"""
    linears = [
        node for node in graph["nodes"] if node["op"] == "linear" and node["ins"][1] == weight
    ]
    if len(linears) != 1:
        raise ValueError(f"{weight}: 線形演算が一意ではありません")
    linear = linears[0]
    checked = 0
    for direction, scale in [("input", input_scale), ("output", output_scale)]:
        if scale == 0:
            continue  # 公式でも未校正は恒等。隣接演算の丸めまでは禁止しない。
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
            raise ValueError(f"{weight}: {direction} SRQ scale が公式と違います")
        checked += 1
    return checked


def assert_qat_stored(model: nn.Module, stored: StoredWeights, ple_index: Path) -> dict:
    from transformers.integrations.gemma_quant import (
        QuantizedEmbedding,
        QuantizedLinear,
    )

    count = 0
    srq_count = 0
    for name, entry in stored.graph["initializers"].items():
        key = entry["tensor"]
        if not key.startswith("model."):
            continue
        original = key.removeprefix("model.")
        if original.startswith("model."):
            original = "model.language_model." + original.removeprefix("model.")
        parent, _, leaf = original.rpartition(".")
        module = model.get_submodule(parent)
        dtype = entry["storage"]["dtype"]
        if dtype in ("i2", "i4", "i8"):
            if leaf != "weight" or not isinstance(module, (QuantizedEmbedding, QuantizedLinear)):
                raise ValueError(f"{key}: 公式固定量子化の対応がありません")
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
                raise ValueError(f"{key}: 固定整数が違います")
            if isinstance(module, QuantizedLinear):
                srq_count += assert_srq_scales(
                    stored.graph,
                    name,
                    float(module.input_activation_scale.detach()),
                    float(module.output_activation_scale.detach()),
                )
            actual_scale = stored.tensor(entry["storage"]["scale"])
            if not torch.equal(actual_scale, scale.expand_as(actual_scale)):
                raise ValueError(f"{key}: 固定scaleが違います")
        else:
            if not torch.equal(stored.dequantized(name, entry), getattr(module, leaf).float()):
                raise ValueError(f"{key}: 非量子化の値が違います")
        count += 1
    layout = json.loads(ple_index.read_text())
    embedding = model.model.language_model.embed_tokens_per_layer
    if (
        layout["storage"] != f"i{embedding.num_bits}"
        or layout["embedScale"] != embedding.scalar_embed_scale
    ):
        raise ValueError("QAT PLE の宣言が違います")
    for shard in layout["shards"]:
        path = ple_index.parent / shard["file"]
        header, base, _ = _read_header(path)
        for key, expected in [
            ("values", embedding.embedding_quantized[shard["start"] : shard["stop"]]),
            ("scales", embedding.embedding_scale[shard["start"] : shard["stop"]]),
        ]:
            low, high = header[key]["data_offsets"]
            with path.open("rb") as source:
                source.seek(base + low)
                raw = source.read(high - low)
            if (
                hashlib.sha256(raw).digest()
                != hashlib.sha256(memoryview(expected.detach().numpy())).digest()
            ):
                raise ValueError(f"{path}: {key} が公式と違います")
    return {
        "fixedInitializersChecked": count,
        "srqScalesChecked": srq_count,
        "pleShardsChecked": len(layout["shards"]),
    }


def load_qat(checkpoint: Path, stored: StoredWeights, ple_index: Path) -> tuple[nn.Module, dict]:
    model = Gemma4ForConditionalGeneration.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    ).eval()
    checks = assert_qat_stored(model, stored, ple_index)
    return model, {
        **checks,
        "attention": "eager",
        "dtype": "float32",
        "srq": "official-transformers",
    }
