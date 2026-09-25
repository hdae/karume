"""公式 Transformers のモデルへ保存済み重みを戻す。再量子化しない。"""

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
from karume.ple import PLE_INDEX_ASSET, PLE_INDEX_SCHEMA
from karume.quantize import dequantize_int4
from karume.verify import InitializerSupply, SupplyBlock, verify_container

#: codec 台帳の展開経路（`codec_entry(...).layout`）→ 生バイトを載せる torch の器。
#: i4 / i2 は packed なので uint8 の平坦な器のまま渡す（論理形へ戻すのは展開の仕事）。
STORAGE_DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "i32": torch.int32,
    "i8": torch.int8,
    "i4": torch.uint8,
    "i2": torch.uint8,
}

#: 部品名 = 容器が名乗るグラフ名 = manifest の weights キー（container-v1 §2.1）。
MODEL_GRAPH = "model"


def fingerprint(path: Path) -> dict:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


@dataclass(frozen=True)
class AssetSlice:
    """資産 1 本の在り処。区間読みを要する資産は **1 block = 1 part**（container-v1 §4.2）。"""

    path: Path
    offset: int
    length: int


class ContainerAssets:
    """容器の資産を読む口（索引は全量 + sha256 検証・PLE の行は区間読み）。

    MUST: 行の読みは区間で取る。PLE の block は 1 本 32 MiB まで在りうるので、token ごとに
    block を丸ごと読むと 1 回の評価が桁で遅くなる（旧 sidecar は safetensors の mmap slice で
    同じことをしていた）。区間読みでは block の sha256 を掛けられないが、宣言（長さ・part の
    割り当て）は {@link verify_container} が既に通している。
    """

    def __init__(self, read: ReadContainer, parts: Sequence[Path]) -> None:
        if read.model is None:
            raise ValueError("グラフだけの容器（krg）に資産はありません")
        if len(parts) < 2:
            raise ValueError("単一形の容器は未対応です（part 0 から並べた分割形が要る）")
        self._read = read
        self._records = read.model.assets
        self._blocks = {block.id: block for block in read.model.blocks}
        self._parts = [Path(part) for part in parts]

    def __contains__(self, name: str) -> bool:
        return name in self._records

    def _record(self, name: str):
        record = self._records.get(name)
        if record is None:
            raise ValueError(f"容器に資産 '{name}' がありません（宣言: {sorted(self._records)}）")
        return record

    def document(self, name: str) -> bytes:
        """資産 1 本を全量で取る（block の sha256 まで突き合わせる — 索引と QAT の突合用）。"""
        record = self._record(name)
        return self._read.block(record.block)[: record.length]

    def slice_of(self, name: str) -> AssetSlice:
        record = self._record(name)
        block = self._blocks[record.block]
        return AssetSlice(self._parts[block.part], block.offset, record.length)

    def read_row(self, name: str, offset: int, length: int) -> bytes:
        """資産の中の区間 `[offset, offset + length)`。"""
        where = self.slice_of(name)
        if offset < 0 or length < 0 or offset + length > where.length:
            raise ValueError(f"{name}: 区間 [{offset}, {offset + length}) が資産の外です")
        with where.path.open("rb") as source:
            source.seek(where.offset + offset)
            raw = source.read(length)
        if len(raw) != length:
            raise ValueError(f"{where.path}: {name} の {length} バイトを読み切れません")
        return raw


class StoredWeights:
    """配布形の容器（`krm` の part 列）から、焼いた実体をそのまま読み返す。

    旧 shard 列の読み手の置き換え（ADR 0109 決定 3 / container-v1）。容器 1 本が部品 1 つで、
    **initializer の名前がそのままテンソルキー**である（IR v2）。piece 分割は束縛表の中に
    閉じるので、呼び手からは 1 本のテンソルに見える。
    """

    def __init__(self, parts: Sequence[Path], graph_name: str = MODEL_GRAPH) -> None:
        if len(parts) < 2:
            raise ValueError("分割形の容器（part 0 から並べた part 列）が要ります")
        verified = verify_container(list(parts))
        self.read = verified.read
        declaration = self.read.graph.graphs.get(graph_name)
        if declaration is None:
            declared = sorted(self.read.graph.graphs)
            raise ValueError(f"容器にグラフ '{graph_name}' がありません（宣言: {declared}）")
        #: IR v2 のグラフ宣言そのもの（`values` / `nodes` / `initializers`）。
        self.graph = declaration
        #: 実体を持つ initializer の供給計画（`shared` 宣言は入らない）。
        self.supplies = verified.graphs[graph_name].supplies
        self.assets = ContainerAssets(self.read, parts)

    def _supply(self, name: str) -> InitializerSupply:
        supply = self.supplies.get(name)
        if supply is None:
            raise ValueError(f"initializer '{name}' の実体が容器にありません")
        return supply

    def _payload(self, block: SupplyBlock) -> bytes:
        """block の payload（末尾の詰め物を落とす — container-v1 §4.1）。"""
        return self.read.block(block.id)[: block.payload_bytes]

    def shape(self, name: str) -> list[int]:
        return list(self.graph["values"][name]["shape"])

    def storage(self, name: str) -> str:
        """格納の展開経路（`i8` / `i4` / `i2` / `f32` …）。正本は束縛表の codec。"""
        return codec_entry(self._supply(name).encoding.codec).layout

    def tensor(self, name: str) -> torch.Tensor:
        """焼いた実体そのまま（i4 / i2 は packed の平坦 uint8・それ以外は論理形）。"""
        supply = self._supply(name)
        layout = codec_entry(supply.encoding.codec).layout
        data = bytearray(b"".join(self._payload(block) for block in supply.blocks))
        value = torch.frombuffer(data, dtype=STORAGE_DTYPES[layout])
        return value if layout in ("i4", "i2") else value.reshape(self.shape(name))

    def scale(self, name: str) -> torch.Tensor:
        """companion scale を `[行, group]` の f32 で返す。

        容器では scale は**名前を持たない block** で、束縛表の `encoding.scale` からしか
        引けない（旧形の「もう 1 本のテンソルキー」は消えた）。
        """
        supply = self._supply(name)
        if supply.scale is None:
            raise ValueError(f"initializer '{name}' は量子化ではありません（scale が無い）")
        if supply.encoding.row_axis not in (None, 0):
            raise ValueError(f"'{name}': rowAxis {supply.encoding.row_axis} は未対応です")
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
                raise ValueError("i8 scale が行数と違います")
            return value.float() * scale.reshape(-1, *([1] * (value.ndim - 1)))
        if layout not in ("f16", "f32", "bf16"):
            raise ValueError(f"未対応の保存 dtype: {layout}")
        return value.float()

    def ple_index(self) -> dict:
        """PLE 索引（資産 `ple_index` — schema 3）。"""
        return json.loads(self.assets.document(PLE_INDEX_ASSET))

    def ple(self, embed_scale: float) -> ContainerPle:
        return ContainerPle(self.ple_index(), self.assets.read_row, embed_scale)


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
    """巨大な PLE を全量 f32 化せず、公式 checkpoint から同じ行 lookup を CPU 上で行う。"""

    def __init__(self, checkpoint: Path, scale: float):
        super().__init__()
        self.scale = scale
        self.key = "model.language_model.embed_tokens_per_layer.weight"
        self.file = checkpoint_index(checkpoint)[self.key]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.device.type != "cpu":
            raise ValueError("DiskPle は CPU 評価専用です")
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


class ContainerPle(nn.Module):
    """容器の資産（索引 schema 3 + `ple.values.<k>` / `ple.scales.<k>`）から PLE の行を引く。

    `read_row(資産名, offset, length)` は区間読みの口で、容器を持たない呼び手（テスト）が
    合成の索引と組みで差し替えられるようにしてある。

    MUST: 格納は `i8` だけを受ける。QAT の PLE（i4 / i2）は**公式の固定量子化そのもの**で、
    このツールが掛ける門は復元ではなく生バイトの一致（{@link assert_qat_stored}）である —
    ここで展開式をもう 1 本持つと、比べるべき相手が 2 つに割れる。
    """

    def __init__(self, index: dict, read_row: Callable[[str, int, int], bytes], embed_scale: float):
        super().__init__()
        if index.get("schema") != PLE_INDEX_SCHEMA:
            raise ValueError(f"PLE 索引の schema が {index.get('schema')}（要 {PLE_INDEX_SCHEMA}）")
        if index["embedScale"] != embed_scale:
            raise ValueError("通常版 PLE の embedScale が公式と違います")
        if index["storage"] != "i8":
            storage = index["storage"]
            raise ValueError(f"PLE の格納 '{storage}' は CPU 参照で復元しません（i8 のみ）")
        self.index = index
        self.read_row = read_row
        self.scale = embed_scale
        self.layers = index["layers"]
        self.dim = index["dim"]

    def _row(self, kind: str, token: int) -> bytes:
        entry = self.index[kind]
        stride = entry["rowBytes"]
        for block in entry["blocks"]:
            if block["start"] <= token < block["stop"]:
                return self.read_row(block["asset"], (token - block["start"]) * stride, stride)
        raise ValueError(f"token {token} を覆う PLE の {kind} block がありません")

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.device.type != "cpu":
            raise ValueError("ContainerPle は CPU 評価専用です")
        rows = {}
        for token in ids.flatten().tolist():
            if token in rows:
                continue
            quant = torch.frombuffer(
                bytearray(self._row("values", token)), dtype=torch.int8
            ).reshape(self.layers, self.dim)
            scale = torch.frombuffer(
                bytearray(self._row("scales", token)), dtype=torch.float32
            ).reshape(self.layers)
            rows[token] = (quant.float() * scale.unsqueeze(-1)).flatten() * self.scale
        return torch.stack([rows[token] for token in ids.flatten().tolist()]).reshape(
            *ids.shape, -1
        )


def ple_placement(module: nn.Module) -> str:
    """差し込んだ PLE の行読みの経路を名乗る（記録は分岐の判断でなく差し込んだ実体から導く）。"""
    if isinstance(module, ContainerPle):
        return "container-row-lookup"
    if isinstance(module, DiskPle):
        return "disk-row-lookup"
    raise TypeError(f"PLE の経路が未知の module です: {type(module).__name__}")


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
        # 保存重みの PLE は**同じ容器の資産**（ADR 0109 決定 4）— 公式 checkpoint 側の行
        # lookup と入れ替えるのは `--weights stored` のときだけ。
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
        # IR v2 では initializer の名前がそのままテンソルキー（旧 `entry["tensor"]`）。
        for name in stored.supplies:
            if not name.startswith("model.") or name.endswith(("cos_table", "sin_table")):
                continue
            target = name.removeprefix("model.")
            if target not in parameters and target not in buffers:
                raise ValueError(f"公式モデルに保存重みの対応先がありません: {target}")
            assign(target, stored.dequantized(name))
    if loaded != set(parameters):
        raise ValueError(f"未読込の parameter: {set(parameters) - loaded}")
    if any(t.device.type == "meta" for t in [*model.parameters(), *model.buffers()]):
        raise ValueError("meta tensor が残っています")
    model.eval()
    return model, {
        "loadedParameterNames": sorted(loaded),
        "attention": "eager",
        "dtype": "float32",
        "ple": ple_placement(model.model.embed_tokens_per_layer) if family == "gemma4" else None,
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


def assert_qat_checked_range(graph: dict, supplies: Iterable[str], prefix: str = "model.") -> None:
    """検査が回る集合（供給計画）が、グラフ宣言の側と**一致する**ことを見る。

    {@link assert_qat_stored} は `stored.supplies`（= 容器に実体がある initializer）を回る。
    容器に実体を持たない `shared` 宣言（貸し手のバイトを借りる — ADR 0096 段 2 §1.3）は
    そこに入らないので、宣言が `shared` へ書き換わった瞬間、その重みは**黙って検査対象から
    外れる**。本数（`fixedInitializersChecked`）が静かに減るだけで赤にならない形なので、
    `load_float_model` の `loaded != set(parameters)` に当たる対の門をここに置く。

    MUST: 両方向を見る（宣言にあって供給に無い = 焼き漏らし / 供給にあって宣言に無い =
    別のグラフの実体を読んでいる）。
    """
    declared = {
        name
        for name, initializer in graph["initializers"].items()
        if name.startswith(prefix) and not initializer.get("shared", False)
    }
    iterated = {name for name in supplies if name.startswith(prefix)}
    if declared != iterated:
        raise ValueError(
            "QAT 固定量子化の検査範囲がグラフ宣言と違います"
            f"（宣言のみ: {sorted(declared - iterated)} /"
            f" 供給のみ: {sorted(iterated - declared)}）"
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
                    key,
                    float(module.input_activation_scale.detach()),
                    float(module.output_activation_scale.detach()),
                )
            actual_scale = stored.scale(key)
            if not torch.equal(actual_scale, scale.expand_as(actual_scale)):
                raise ValueError(f"{key}: 固定scaleが違います")
        else:
            if not torch.equal(stored.dequantized(key), getattr(module, leaf).float()):
                raise ValueError(f"{key}: 非量子化の値が違います")
        count += 1
    # PLE は**同じ容器の資産**（ADR 0109 決定 4）。索引 schema 3 は行の種別ごとに別の切り方を
    # するので、`values` / `scales` をそれぞれの block 列で突き合わせる（旧 sidecar は 1 つの
    # shard が両方を持っていた）。
    layout = stored.ple_index()
    embedding = model.model.language_model.embed_tokens_per_layer
    if (
        layout["storage"] != f"i{embedding.num_bits}"
        or layout["embedScale"] != embedding.scalar_embed_scale
    ):
        raise ValueError("QAT PLE の宣言が違います")
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
                raise ValueError(f"{block['asset']}: PLE の {kind} が公式と違います")
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
