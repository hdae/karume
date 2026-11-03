"""`gemma4/export_product_qat.py`: the q4_0 QAT line of the E2B / E4B product series.

No real weights: the stand-in weights lie on a q4_0 grid as the QAT checkpoints' do
({@link grid_weight} — `bf16(q·d)` per group of 32, −8 codes and all-zero groups included). What
is pinned here:

- the grid import of every weight class (the linears, the tied embedding, the PLE tables) replaces
  the weight by the decoded grid in place, and a group off the grid fails naming the tensor and the
  group;
- the schema-4 PLE asset (one scale per group of 32): its index, its (layer, group) order, a bit
  exact read-back of every token, and the checks that catch a scale on the wrong group or the
  groups read as one scale per row;
- the regular product path is unchanged: the schema-3 index bytes, the regular assets' form, the
  loader's residue expectation and the trace's one path;
- the series driver end to end on a tiny model: the container decodes to the imported grid with no
  i8 left, the probe reads back from the published assets, the goldens are the series' own and
  the record names both pinned sources; a wrong pin and a margin under the floor stop the run
  before anything long.

transformers-backed cases SKIP through `importorskip` (ADR 0065's two-job layout).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest
import torch
from container_series import write_component
from ir_fixtures import ir_container
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from _shared import decode_series as shared
from _shared.container_read import read_asset, read_provenance, read_stored
from gemma4 import export as gx
from gemma4 import export_decode as decode
from gemma4 import export_product as product
from gemma4 import export_product_qat as qat
from gemma4 import export_unified as unified
from gemma4 import ple, provenance, sources
from gemma4.tests.test_export import (
    SHARED_LAYER,
    TINY_SYM_MAX,
    VOCAB,
    WINDOW,
    _full_table_config,
    _write_checkpoint,
)
from gemma4.tests.test_export_decode import (
    DECODE_LAYER_TYPES,
    TINY_CASE_NAMES,
    _StubSeriesTokenizer,
    _tiny_decode_config,
)
from karume.container import container_parts
from karume.emit import unpack_int4
from karume.ple import (
    PLE_GROUPED_INDEX_SCHEMA,
    PLE_INDEX_ASSET,
    PLE_INDEX_SCHEMA,
    PleError,
    ple_assets,
    ple_block_ranges,
    ple_row_bytes,
)
from karume.quantize import dequantize_int4

GROUP = qat.GROUP_SIZE

#: The tiny PLE width of this file: **three** groups per (token, layer) row — one group would make
#: the schema-4 scales indistinguishable from schema 3's one scale per row. MUST: not the tiny
#: hidden width (64) — the product form check tells a PLE table from the embedding by its shape.
QAT_PLE_DIM = 3 * GROUP


def grid_weight(rows: int, cols: int, seed: int) -> torch.Tensor:
    """An F32 weight on a q4_0 grid as the QAT checkpoints store one (BF16 widened): `bf16(q·d)`
    per group of 32 along the row, codes in [−8, 7], a learned-looking step per group
    (log-normal), every third group's codes in [−3, 3], group 0 of row 0 all zero and (with two
    groups or more) a −8 in group 1 of row 0."""
    generator = torch.Generator().manual_seed(seed)
    groups = cols // GROUP
    shape = (rows, groups, GROUP)
    codes = torch.randint(-8, 8, shape, generator=generator)
    narrow = torch.randint(-3, 4, shape, generator=generator)
    codes[:, ::3] = narrow[:, ::3]
    codes[0, 0] = 0
    if groups > 1:
        codes[0, 1, 0] = -8
    steps = 0.004 * torch.exp(0.3 * torch.randn((rows, groups), generator=generator))
    weight = (codes * steps.unsqueeze(-1)).reshape(rows, cols).to(torch.bfloat16)
    return weight.to(torch.float32)


def _decoded(fixed) -> tuple[torch.Tensor, torch.Tensor]:
    """`(codes, values)` of a packed i4 grid (`karume.emit.unpack_int4` + `dequantize_int4`)."""
    rows, half = fixed.packed.shape
    codes = unpack_int4(fixed.packed.reshape(-1), (rows, 2 * half))
    return codes, dequantize_int4(codes, fixed.scale)


def _on_grid(model: nn.Module, tables: nn.ModuleList) -> None:
    """Replace every 2-D weight of `model` and `tables` by a {@link grid_weight} (in place — the
    tied head follows the embedding)."""
    for index, (_, parameter) in enumerate(model.named_parameters()):
        if parameter.dim() == 2:
            parameter.data.copy_(grid_weight(*parameter.shape, seed=index))
    for index, table in enumerate(tables):
        table.weight.data.copy_(grid_weight(*table.weight.shape, seed=1000 + index))


def _tiny_grid_materials() -> tuple[nn.Module, nn.ModuleList]:
    """The tiny decode model (KV sharing on) with a {@link QAT_PLE_DIM}-wide PLE, every 2-D weight
    on the grid — what `gemma4.export.load_model_and_tables` returns (probe-sized PLE seat
    included)."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    gx.register_attention()
    config = _tiny_decode_config()
    config.hidden_size_per_layer_input = QAT_PLE_DIM
    config._attn_implementation = gx.ATTENTION_NAME
    model = transformers.Gemma4ForCausalLM(config).to(torch.float32).eval()
    tables = nn.ModuleList(
        [
            nn.Embedding.from_pretrained(torch.zeros(VOCAB, QAT_PLE_DIM), freeze=True)
            for _ in DECODE_LAYER_TYPES
        ]
    )
    _on_grid(model, tables)
    model.config.vocab_size_per_layer_input = ple.PLE_PROBE_ROWS
    return model, tables


@pytest.fixture
def grid_wrapper(monkeypatch) -> product.ProductChunkWrapper:
    """The product wrapper of the QAT loader over {@link _tiny_grid_materials}."""
    model, tables = _tiny_grid_materials()
    monkeypatch.setattr(gx, "load_model_and_tables", lambda model_dir, **_: (model, tables))
    return qat.load_wrapper(Path("unused"))


# ---- the grid import --------------------------------------------------------


class TestImportGrid:
    def test_the_weight_becomes_its_decoded_grid_in_place(self) -> None:
        weight = grid_weight(8, 4 * GROUP, seed=1)
        original = weight.clone()

        fixed = qat.import_grid(weight, "'w'")

        codes, values = _decoded(fixed)
        assert fixed.dtype == "i4"
        assert fixed.packed.dtype is torch.uint8 and list(fixed.packed.shape) == [8, 2 * GROUP]
        assert fixed.scale.dtype is torch.float32 and list(fixed.scale.shape) == [8, 4]
        assert torch.equal(weight, values)
        # Within BF16's rounding of the checkpoint's `q·d` (the recovery's bound).
        groups = original.reshape(8, 4, GROUP)
        error = (values.reshape(8, 4, GROUP) - groups).abs().amax(-1)
        assert bool((error <= unified.GRID_TOLERANCE * groups.abs().amax(-1)).all())
        assert int(codes.min()) == -8
        assert torch.equal(codes[0, :GROUP], torch.zeros(GROUP, dtype=torch.int8))

    def test_blocks_of_rows_give_the_whole_tensor_s_import(self, monkeypatch) -> None:
        weight = grid_weight(10, 2 * GROUP, seed=2)
        whole = qat.import_grid(weight.clone(), "'w'")
        monkeypatch.setattr(unified, "CHUNK_BYTES", 3 * 2 * GROUP * 4)  # 3 rows a block

        blocks = qat.import_grid(weight.clone(), "'w'")

        assert torch.equal(blocks.packed, whole.packed)
        assert torch.equal(blocks.scale, whole.scale)

    def test_a_group_off_the_grid_fails_naming_the_tensor_and_the_group(self, monkeypatch) -> None:
        monkeypatch.setattr(unified, "CHUNK_BYTES", 4 * 2 * GROUP * 4)  # 4 rows a block
        weight = grid_weight(8, 2 * GROUP, seed=3)
        weight[5, GROUP + 3] *= 1.3

        with pytest.raises(
            unified.UnifiedExportError, match=r"'w' rows 4\.\.8: .*group 1 of row 5"
        ):
            qat.import_grid(weight, "'w'")

    def test_a_weight_that_is_not_f32_or_not_whole_groups_fails_loudly(self) -> None:
        with pytest.raises(qat.QatExportError, match="F32 rank 2"):
            qat.import_grid(grid_weight(2, GROUP, seed=4).to(torch.bfloat16), "'w'")
        with pytest.raises(qat.QatExportError, match="groups of 32"):
            qat.import_grid(torch.zeros(2, GROUP + 16), "'w'")


class TestImportWrapper:
    def test_every_two_d_weight_is_imported(self, grid_wrapper) -> None:
        linears = unified.int4_targets(grid_wrapper)

        grid = qat.import_wrapper(grid_wrapper)

        assert set(grid.fixed) == {*linears, unified.EMBED_KEY}
        assert any("per_layer_model_projection" in key for key in linears)
        assert any("per_layer_input_gate" in key for key in linears)
        for key in linears:
            module = grid_wrapper.get_submodule(key.removesuffix(".weight"))
            assert torch.equal(module.weight, _decoded(grid.fixed[key])[1]), key
        embedding = grid_wrapper.model.model.embed_tokens.weight
        assert torch.equal(embedding, _decoded(grid.fixed[unified.EMBED_KEY])[1])
        assert grid_wrapper.model.lm_head.weight is embedding
        assert len(grid.ple) == len(DECODE_LAYER_TYPES)
        for table, imported in zip(grid_wrapper.per_layer, grid.ple, strict=True):
            assert list(imported.scale.shape) == [VOCAB, QAT_PLE_DIM // GROUP]
            assert torch.equal(table.weight, _decoded(imported)[1])

    def test_a_two_d_weight_outside_the_three_classes_fails_loudly(self, grid_wrapper) -> None:
        grid_wrapper.model.model.register_parameter(
            "stray", nn.Parameter(torch.zeros(2, GROUP), requires_grad=False)
        )

        with pytest.raises(
            qat.QatExportError, match=r"outside the grid import: \['model\.model\.stray'\]"
        ):
            qat.import_wrapper(grid_wrapper)

    def test_a_ple_group_off_the_grid_fails_naming_the_table_and_the_group(
        self, grid_wrapper
    ) -> None:
        grid_wrapper.per_layer[1].weight.data[5, GROUP + 3] *= 1.3

        where = (
            r"'model\.language_model\.embed_tokens_per_layer\.weight' layer 1 \(columns 96\.\.192\)"
        )
        with pytest.raises(unified.UnifiedExportError, match=rf"{where} .*group 1 of row 5"):
            qat.import_wrapper(grid_wrapper)


# ---- the schema-4 PLE asset -------------------------------------------------

#: The schema-4 fixture's shape: 3 layers of three groups, 10 tokens, `values` blocks of 3 rows.
GROUPED_LAYERS = 3
GROUPED_TOKENS = 10
GROUPED_EMBED_SCALE = 4.0
GROUPED_BLOCK = 3 * ple_row_bytes("i4", GROUPED_LAYERS, QAT_PLE_DIM, GROUP)["values"]


def _grouped_assets(tmp_path: Path, tables: list[torch.Tensor], name: str):
    """Import `tables` (in place), build the schema-4 assets and write them into a container;
    returns `(container, index, imported)`."""
    imported = [qat.import_grid(table, f"layer {layer}") for layer, table in enumerate(tables)]
    spill = tmp_path / f"{name}.staging"
    spill.mkdir()
    built = product.ple_container_assets(
        [table.packed for table in imported],
        [table.scale for table in imported],
        tokens=GROUPED_TOKENS,
        layers=GROUPED_LAYERS,
        dim=QAT_PLE_DIM,
        embed_scale=GROUPED_EMBED_SCALE,
        spill_dir=spill,
        storage=qat.PLE_STORAGE,
        group_size=GROUP,
        block_bytes=GROUPED_BLOCK,
    )
    container = tmp_path / f"{name}.krm"
    write_component(container, ir_container(mark=name, assets=built.assets))
    built.discard()
    return container, json.loads(bytes(built.assets[PLE_INDEX_ASSET].payload)), imported


@pytest.fixture
def grouped(tmp_path):
    """The schema-4 PLE of grid tables, written; `(container, index, imported, reference)` with
    the reference of every token by the 35-table path over the imported tables."""
    tables = [
        grid_weight(GROUPED_TOKENS, QAT_PLE_DIM, seed=20 + layer) for layer in range(GROUPED_LAYERS)
    ]
    container, index, imported = _grouped_assets(tmp_path, tables, "grouped")
    lookup = nn.ModuleList([nn.Embedding.from_pretrained(table) for table in tables])
    every = torch.arange(GROUPED_TOKENS).unsqueeze(0)
    with torch.no_grad():
        reference = ple.per_layer_inputs(lookup, every, GROUPED_EMBED_SCALE)
    return container, index, imported, reference


class TestGroupedPleAssets:
    def test_the_index_is_schema_4_with_one_scale_per_group(self, grouped) -> None:
        _, index, _, _ = grouped

        assert index["schema"] == PLE_GROUPED_INDEX_SCHEMA == 4
        assert index["groupSize"] == GROUP
        assert index["storage"] == "i4"
        assert index["values"]["rowBytes"] == GROUPED_LAYERS * QAT_PLE_DIM // 2
        assert index["scales"]["rowBytes"] == GROUPED_LAYERS * (QAT_PLE_DIM // GROUP) * 4
        assert len(index["values"]["blocks"]) > 1, "the read-back crosses a block boundary"

    def test_every_token_reads_back_bit_for_bit(self, grouped) -> None:
        container, index, _, reference = grouped

        product.assert_ple_assets(container, index, list(range(GROUPED_TOKENS)), reference)

    def test_the_payload_is_the_codes_and_the_steps_in_layer_then_group_order(
        self, grouped
    ) -> None:
        container, index, imported, _ = grouped
        (block,) = index["scales"]["blocks"]

        raw = read_asset(container, block["asset"])
        steps = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(
            GROUPED_TOKENS, GROUPED_LAYERS, QAT_PLE_DIM // GROUP
        )

        assert torch.equal(steps, torch.stack([table.scale for table in imported], dim=1))
        first = index["values"]["blocks"][0]
        packed = torch.frombuffer(
            bytearray(read_asset(container, first["asset"])), dtype=torch.uint8
        )
        rows = first["stop"] - first["start"]
        expected = torch.stack([table.packed[:rows] for table in imported], dim=1)
        assert torch.equal(packed.reshape(rows, GROUPED_LAYERS, QAT_PLE_DIM // 2), expected)

    def test_a_step_on_the_wrong_group_is_detected(self, grouped, tmp_path) -> None:
        _, _, imported, reference = grouped
        swapped = [table.scale.clone() for table in imported]
        swapped[0][4] = swapped[0][4].flip(0)
        assert not torch.equal(swapped[0][4], imported[0].scale[4])
        spill = tmp_path / "swapped.staging"
        spill.mkdir()
        built = product.ple_container_assets(
            [table.packed for table in imported],
            swapped,
            tokens=GROUPED_TOKENS,
            layers=GROUPED_LAYERS,
            dim=QAT_PLE_DIM,
            embed_scale=GROUPED_EMBED_SCALE,
            spill_dir=spill,
            storage=qat.PLE_STORAGE,
            group_size=GROUP,
        )
        container = tmp_path / "swapped.krm"
        write_component(container, ir_container(mark="swapped", assets=built.assets))
        index = json.loads(bytes(built.assets[PLE_INDEX_ASSET].payload))

        with pytest.raises(AssertionError, match="ビット一致しない"):
            product.assert_ple_assets(container, index, [4], reference[:, 4:5])

    def test_the_groups_read_as_one_scale_per_row_fail(self, grouped) -> None:
        container, index, _, reference = grouped
        as_schema_3 = {key: value for key, value in index.items() if key != "groupSize"}
        as_schema_3["schema"] = PLE_INDEX_SCHEMA

        with pytest.raises(AssertionError, match="rowBytes"):
            product.assert_ple_assets(container, as_schema_3, [0], reference[:, :1])

    @pytest.mark.parametrize(
        "edit",
        [
            {"schema": PLE_GROUPED_INDEX_SCHEMA, "groupSize": None},
            {"schema": PLE_INDEX_SCHEMA},
        ],
        ids=["schema-4-without-group-size", "schema-3-with-group-size"],
    )
    def test_the_group_size_is_present_exactly_in_schema_4(self, grouped, edit) -> None:
        container, index, _, reference = grouped
        edited = {**index, **edit}
        if edited.get("groupSize") is None:
            del edited["groupSize"]

        with pytest.raises((AssertionError, KeyError), match="groupSize"):
            product.assert_ple_assets(container, edited, [0], reference[:, :1])

    def test_a_group_size_that_does_not_divide_the_row_fails(self) -> None:
        with pytest.raises(PleError, match="does not divide"):
            ple_row_bytes("i4", GROUPED_LAYERS, QAT_PLE_DIM, 40)
        with pytest.raises(PleError, match="does not divide"):
            ple_block_ranges(
                storage="i4", tokens=4, layers=GROUPED_LAYERS, dim=QAT_PLE_DIM, group_size=0
            )


# ---- the regular product path is unchanged ----------------------------------

#: `karume.ple.ple_assets`'s schema-3 index for 6 tokens × 2 layers × 4 i8 values, blocks of 16 B,
#: as the writer spelled it before schema 4 existed (the regular series' bytes).
SCHEMA_3_INDEX = (
    '{"schema":3,"storage":"i8","tokens":6,"layers":2,"dim":4,"embedScale":2,'
    '"values":{"rowBytes":8,"blocks":[{"asset":"ple.values.0","start":0,"stop":2},'
    '{"asset":"ple.values.1","start":2,"stop":4},{"asset":"ple.values.2","start":4,"stop":6}]},'
    '"scales":{"rowBytes":8,"blocks":[{"asset":"ple.scales.0","start":0,"stop":2},'
    '{"asset":"ple.scales.1","start":2,"stop":4},{"asset":"ple.scales.2","start":4,"stop":6}]}}'
)


def _residue_free_checkpoint(directory: Path) -> Path:
    """`test_export._write_checkpoint`'s checkpoint without the KV-shared layer's k / v residues
    (the q4_0 QAT checkpoints' form)."""
    path = _write_checkpoint(directory)
    residue = f"{gx.CHECKPOINT_TEXT_PREFIX}layers.{SHARED_LAYER}.self_attn."
    residues = {f"{residue}{leaf}" for leaf in ("k_proj.weight", "v_proj.weight", "k_norm.weight")}
    with safe_open(str(path), framework="pt") as handle:
        assert residues <= set(handle.keys())
        kept = {key: handle.get_tensor(key) for key in handle.keys() if key not in residues}  # noqa: SIM118
    save_file(kept, str(path))
    return path


class TestRegularPathUnchanged:
    def test_the_schema_3_index_keeps_its_bytes(self) -> None:
        assets = ple_assets(
            storage="i8",
            tokens=6,
            layers=2,
            dim=4,
            embed_scale=2.0,
            read_values=lambda begin, end: bytes(end - begin),
            read_scales=lambda begin, end: bytes(end - begin),
            block_bytes=16,
        )

        assert bytes(assets[PLE_INDEX_ASSET].payload).decode() == SCHEMA_3_INDEX

    def test_the_regular_assets_stay_i8_with_one_scale_per_row(self, tmp_path) -> None:
        layers, dim = 2, 8
        built = product.ple_container_assets(
            [torch.zeros(VOCAB, dim, dtype=torch.int8) for _ in range(layers)],
            [torch.ones(VOCAB) for _ in range(layers)],
            tokens=VOCAB,
            layers=layers,
            dim=dim,
            embed_scale=2.0,
            spill_dir=tmp_path,
        )
        index = json.loads(bytes(built.assets[PLE_INDEX_ASSET].payload))
        built.discard()

        assert index["schema"] == PLE_INDEX_SCHEMA == 3
        assert "groupSize" not in index
        assert index["storage"] == product.PLE_STORAGE == "i8"
        assert index["scales"]["rowBytes"] == layers * 4

    def test_the_regular_loader_still_expects_the_residues(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("transformers")
        _residue_free_checkpoint(tmp_path)
        monkeypatch.setattr(gx, "load_text_config", lambda _: _full_table_config())

        with pytest.raises(ValueError, match="残骸"):
            gx.load_model_and_tables(tmp_path)
        model, tables = gx.load_model_and_tables(tmp_path, kv_residues=False)

        assert model.lm_head.weight is model.model.embed_tokens.weight
        assert len(tables) == model.config.num_hidden_layers

    def test_residues_in_a_checkpoint_expected_without_them_fail(
        self, tmp_path, monkeypatch
    ) -> None:
        pytest.importorskip("transformers")
        _write_checkpoint(tmp_path)
        monkeypatch.setattr(gx, "load_text_config", lambda _: _full_table_config())

        with pytest.raises(ValueError, match="residues of the KV-shared layers"):
            gx.load_model_and_tables(tmp_path, kv_residues=False)

    def test_the_trace_refuses_a_wrapper_still_holding_the_tables(self, grid_wrapper) -> None:
        ids = torch.zeros(1, 3, dtype=torch.int64)

        with pytest.raises(AssertionError, match="still holds the PLE tables"):
            product.trace(grid_wrapper, ids, torch.zeros(1, 3, 1, 1), TINY_SYM_MAX)


class TestGreedyView:
    def test_it_scores_the_last_row_over_the_tables_and_masks_the_suppressed_ids(
        self, grid_wrapper
    ) -> None:
        specs = decode.rope_specs(grid_wrapper.model.config)
        ids = torch.tensor([[2, 7, 11, 5, 9]])
        view = qat.GreedyView(grid_wrapper, specs, suppress=(3, 4))

        with torch.no_grad():
            logits, hidden = view(ids, shared.positions_for(ids))
            expected, expected_hidden = grid_wrapper(
                ids,
                *decode.rope_args(specs, shared.positions_for(ids)),
                ple.per_layer_inputs(grid_wrapper.per_layer, ids, grid_wrapper.per_layer_scale),
                decode.last_row_for(ids),
            )

        assert list(logits.shape) == [1, 1, VOCAB]
        assert bool(torch.isneginf(logits[0, 0, [3, 4]]).all())
        keep = [token for token in range(VOCAB) if token not in (3, 4)]
        assert torch.equal(logits[..., keep], expected[..., keep])
        assert torch.equal(hidden, expected_hidden)


# ---- the series driver --------------------------------------------------------

#: The summary's fields, in order (a change here changes the run's record).
QAT_SUMMARY_KEYS = [
    "dir",
    "nodes",
    "outputs",
    "initializers",
    "parts",
    "model_bytes",
    "ple_bytes",
    "ple_blocks",
    "ple_probe_tokens",
    "ops",
    "symbols",
    "case_lengths",
    "quantized",
    "form",
    "sanity",
    "greedy",
    "greedy_steps",
    "margin_min",
    "continuation",
    "suppress_tokens",
    "reference",
    "stages",
]

#: The tiny tokenizer source's suppressed id (the real E2B / E4B declare none — the masking path is
#: exercised here).
TINY_SUPPRESSED = (VOCAB - 1,)


def _plain_pin(directory: Path, repo: str, names: tuple[str, ...]) -> sources.HubPin:
    """A pin over plain stand-in files (the digest of their bytes)."""
    return sources.HubPin(
        repo=repo,
        revision="b" * 40,
        sha256={name: provenance.file_digest(directory / name)["sha256"] for name in names},
    )


@pytest.fixture
def qat_series(monkeypatch, tmp_path_factory):
    """The driver's stand-ins: the materials (a tiny grid model — the loader's call is recorded),
    the cases, the tokenizer, the margin floor, the first-token sanity, PLE blocks small enough to
    split, and two pinned stand-in sources. Everything from the grid import to the publication is
    the real path."""
    pytest.importorskip("transformers")
    calls: list[bool] = []
    weights: dict[str, torch.Tensor] = {}

    def materials(model_dir, *, kv_residues=True):
        calls.append(kv_residues)
        model, tables = _tiny_grid_materials()
        weights.update(
            {
                f"model.{name}": parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.dim() == 2
            }
        )
        return model, tables

    monkeypatch.setattr(gx, "load_model_and_tables", materials)
    torch.manual_seed(1)
    cases = tuple(
        (name, torch.randint(0, VOCAB, (1, WINDOW + 3 - index), dtype=torch.int64))
        for index, name in enumerate(TINY_CASE_NAMES)
    )
    monkeypatch.setattr(gx, "build_cases", lambda model_dir, sym_max, window: cases)
    monkeypatch.setattr(gx, "load_tokenizer", lambda model_dir: _StubSeriesTokenizer())
    monkeypatch.setattr(decode, "MARGIN_FLOOR", 0.0)
    seen: dict[str, dict[str, int]] = {}

    def record(greedy, expected, labels):
        seen["first"] = dict(greedy)
        return {"stub": "ok"}

    monkeypatch.setattr(gx, "_sanity", record)
    small = 3 * ple_row_bytes("i4", len(DECODE_LAYER_TYPES), QAT_PLE_DIM, GROUP)["values"]
    monkeypatch.setattr(
        qat, "ple_block_ranges", functools.partial(ple_block_ranges, block_bytes=small)
    )
    monkeypatch.setattr(
        product,
        "ple_container_assets",
        functools.partial(product.ple_container_assets, block_bytes=small),
    )
    weights_dir = tmp_path_factory.mktemp("qat-weights")
    for name in provenance.WEIGHTS_FILES:
        (weights_dir / name).write_bytes(f"{name} of the QAT checkpoint".encode())
    tokenizer_dir = tmp_path_factory.mktemp("regular-tokenizer")
    (tokenizer_dir / "tokenizer.json").write_bytes(b"tokenizer of the regular checkpoint")
    (tokenizer_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "suppress_tokens": list(TINY_SUPPRESSED),
                "temperature": 1.0,
                "top_k": 8,
                "top_p": 1.0,
            }
        ),
        encoding="utf-8",
    )
    pins = sources.ModelPins(
        weights=_plain_pin(weights_dir, "test/qat", provenance.WEIGHTS_FILES),
        tokenizer=_plain_pin(tokenizer_dir, "test/regular", provenance.TOKENIZER_FILES),
    )

    def run(out_dir: Path, **overrides):
        options = {
            "tokenizer_dir": tokenizer_dir,
            "pins": pins,
            "sym_max": TINY_SYM_MAX,
            "steps": 2,
            "greedy_cases": TINY_CASE_NAMES,
            "checkpoint_label": pins.label,
            "floor": 0.0,
            **overrides,
        }
        return qat.export_series(weights_dir, out_dir, **options)

    return type(
        "QatSeries",
        (),
        {"run": staticmethod(run), "calls": calls, "weights": weights, "pins": pins, "seen": seen},
    )


class TestExportSeries:
    def test_it_publishes_the_container_the_probe_the_goldens_and_the_record(
        self, qat_series, tmp_path
    ) -> None:
        out_dir = tmp_path / "gemma4-e2b-qat-product"

        summary = qat_series.run(out_dir)

        assert qat_series.calls == [False], "the loader expects no k / v residues"
        assert list(summary) == QAT_SUMMARY_KEYS
        greedy = [f"{shared.GREEDY_PREFIX}{name}{shared.GREEDY_SUFFIX}" for name in TINY_CASE_NAMES]
        parts = [path.name for path in container_parts(out_dir / gx.MODEL_FILE)]
        assert sorted(path.name for path in out_dir.iterdir()) == sorted(
            [product.PLE_PROBE_FILE, provenance.REFERENCE_FILE, *greedy, *parts]
        )
        assert summary["greedy"] == greedy
        assert summary["outputs"] == 2
        assert summary["suppress_tokens"] == list(TINY_SUPPRESSED)
        assert summary["ple_blocks"]["values"] > 1, "the probe crosses a block boundary"
        assert set(qat_series.seen["first"]) == set(TINY_CASE_NAMES)
        assert TINY_SUPPRESSED[0] not in qat_series.seen["first"].values()
        # Neither the staging seat nor the spill files are left.
        assert list(tmp_path.iterdir()) == [out_dir]

    def test_the_container_decodes_to_the_imported_grid_with_no_i8(
        self, qat_series, tmp_path
    ) -> None:
        out_dir = tmp_path / "series"

        summary = qat_series.run(out_dir)

        stored = read_stored(out_dir / gx.MODEL_FILE)
        quantized = {key: entry for key, entry in stored.items() if entry.layout != "f32"}
        assert {entry.layout for entry in quantized.values()} == {"i4"}
        assert len(quantized) == summary["quantized"]["i4"]
        assert summary["quantized"] == {"i4": len(quantized), "i8": 0}
        assert summary["form"]["storage"].get("i8", 0) == 0
        lowest = 0
        for key, entry in quantized.items():
            source = key if key != unified.TIED_HEAD_KEY else unified.EMBED_KEY
            expected = qat_series.weights[source].clone()
            qat.import_grid(expected, key)
            rows, cols = entry.shape
            assert entry.group_size == GROUP, key
            codes = unpack_int4(
                torch.frombuffer(bytearray(entry.payload), dtype=torch.uint8), entry.shape
            )
            steps = torch.frombuffer(bytearray(entry.scale), dtype=torch.float32).reshape(
                rows, cols // GROUP
            )
            assert torch.equal(dequantize_int4(codes, steps), expected), key
            lowest = min(lowest, int(codes.min()))
        assert lowest == -8
        assert any(key in quantized for key in (unified.EMBED_KEY, unified.TIED_HEAD_KEY))

    def test_the_schema_4_probe_reads_back_from_the_published_assets(
        self, qat_series, tmp_path
    ) -> None:
        out_dir = tmp_path / "series"

        qat_series.run(out_dir)

        index = json.loads(read_asset(out_dir / gx.MODEL_FILE, PLE_INDEX_ASSET))
        assert index["schema"] == PLE_GROUPED_INDEX_SCHEMA
        assert index["groupSize"] == GROUP
        assert index["storage"] == qat.PLE_STORAGE
        assert index["tokens"] == VOCAB
        with safe_open(str(out_dir / product.PLE_PROBE_FILE), framework="pt") as handle:
            probe = [int(token) for token in handle.get_tensor(product.PROBE_TOKENS_KEY)]
            expected = handle.get_tensor(product.PROBE_INPUTS_KEY)
        product.assert_ple_assets(out_dir / gx.MODEL_FILE, index, probe, expected)

    def test_the_record_names_both_sources_and_the_series_own_goldens(
        self, qat_series, tmp_path
    ) -> None:
        out_dir = tmp_path / "gemma4-e4b-qat-product"

        qat_series.run(out_dir)

        record = json.loads((out_dir / provenance.REFERENCE_FILE).read_text(encoding="utf-8"))
        pins = qat_series.pins
        assert record["checkpoint"]["dir"] == pins.label
        assert record["checkpoint"]["sources"] == {
            "weights": {
                "repo": "test/qat",
                "revision": pins.weights.revision,
                "files": list(provenance.WEIGHTS_FILES),
            },
            "tokenizer": {
                "repo": "test/regular",
                "revision": pins.tokenizer.revision,
                "files": list(provenance.TOKENIZER_FILES),
            },
        }
        assert record["reference"]["series"] == out_dir.name
        assert record["reference"]["suppressTokens"] == list(TINY_SUPPRESSED)
        for name in TINY_CASE_NAMES:
            golden = f"{shared.GREEDY_PREFIX}{name}{shared.GREEDY_SUFFIX}"
            assert record["reference"]["goldens"][golden] == provenance.file_digest(
                out_dir / golden
            )
        assert read_provenance(out_dir / gx.MODEL_FILE).upstream_revision == pins.weights.revision

    def test_a_wrong_pin_fails_before_the_load(self, qat_series, tmp_path) -> None:
        out_dir = tmp_path / "series"
        pins = qat_series.pins
        wrong = sources.ModelPins(
            weights=pins.weights,
            tokenizer=sources.HubPin(
                repo=pins.tokenizer.repo,
                revision=pins.tokenizer.revision,
                sha256={"tokenizer.json": "0" * 64},
            ),
        )

        with pytest.raises(sources.SourcePinError, match=r"tokenizer\.json"):
            qat_series.run(out_dir, pins=wrong)
        assert qat_series.calls == []
        assert not out_dir.exists()

    def test_a_margin_under_the_floor_stops_the_run_before_the_trace(
        self, qat_series, tmp_path, monkeypatch
    ) -> None:
        out_dir = tmp_path / "series"
        monkeypatch.setattr(decode, "MARGIN_FLOOR", float("inf"))
        traced: list[int] = []
        monkeypatch.setattr(product, "trace", lambda *args, **kwargs: traced.append(1))

        with pytest.raises(AssertionError, match="margin"):
            qat_series.run(out_dir)
        assert traced == []
        assert not out_dir.exists()

    def test_an_unknown_greedy_case_fails_before_the_load(self, qat_series, tmp_path) -> None:
        with pytest.raises(qat.QatExportError, match="unknown"):
            qat_series.run(tmp_path / "series", greedy_cases=("window-en",))
        assert qat_series.calls == []
