"""The contract of `wan/umt5_export.py` (trace on meta and turn the weights into i8 one at a time
from the checkpoint) (ADR 0119 step 10b — decisions 5 / 6).

What is pinned is the side where a break raises no exception and **only the container bytes drift
quietly**:

- the graph traced on an upstream without weights (meta) is identical in JSON to the export with
  real weights
- the container written with i8 made per block of rows (`fixed_weights`) **matches in the bytes of
  every part** the container written in the plain form (`fake_quant_i8` in f32 →
  `weight_dtype="i8"` + the tables explicitly F32) — this pins all of: no block boundary in the
  middle of a row, the tied vocabulary embedding read from `shared.weight`, the tables not rounded
- the storage breakdown: linear and the vocabulary embedding are i8, the relative-position tables
  and RMSNorm are f32
- a checkpoint key that cannot be determined or that is not F32 fails loudly
- the original model's split pickle form (ADR 0122 decision 2): the sha256 of the table row is
  checked before reading through mmap, giving the same values and the same container as
  safetensors. The rule for tied aliases (same storage / bit-identical → `shared.weight`; otherwise
  fail loudly). A sha256 mismatch, a missing shard, a stale table and an index not reached through
  the table fail before `torch.load`. The checked content is read even if the path is swapped after
  the check; no shard mmap remains after reading; a pickle with a global outside the allow list is
  refused
- an explicit argument that has no effect on the subcommand (the axis `--upstream` / `--model` and
  others) is refused
- the official writer ({@link wan.umt5_export.write_container}) writes the same bytes as the plain
  form above, and the comparison with an existing container
  ({@link wan.umt5_export.assert_same_container}) names a single-byte difference by part

It runs on a checkpoint `save_pretrained` from a small randomly initialized umT5
(`wan.umt5_probe.TINY_CONFIG`) — no real weights are read.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from karume.pipeline import publish_model
from wan import umt5_export as ue
from wan import umt5_patch as up
from wan import umt5_probe as probe
from wan.sources import DEFAULT_MODEL, UMT5_BIN_INDEX, umt5_encoder_shards
from wan.umt5_distribution import UMT5_DEFAULT_MODEL

LAYERS = int(probe.TINY_CONFIG["num_layers"])

#: The number of linear layers of the small model (per layer q / k / v / o and wi_0 / wi_1 / wo).
LINEARS_PER_LAYER = 7

#: The number of rows per block (divides none of vocabulary 384, d_ff 160, d_model 64 — a partial
#: block is always hit).
CHUNK_ROWS = 7


@pytest.fixture(scope="module")
def model() -> Any:
    pytest.importorskip("transformers")
    return probe.tiny_model()


@pytest.fixture(scope="module")
def checkpoint_dir(model: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The upstream `save_pretrained` form (config + a single safetensors — only one of the tied
    weights)."""
    directory = tmp_path_factory.mktemp("umt5-tiny")
    model.save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def prepared(checkpoint_dir: Path) -> ue.Umt5Export:
    return ue.prepare(checkpoint_dir, chunk_rows=CHUNK_ROWS)


def write(path: Path, graph: Any, tensors: Any, **storage: Any) -> list[bytes]:
    """Write the container and return the bytes of every part (the provenance and graph name are
    the same as the official writer's — differences show only in the byte contents)."""
    publish_model(
        path / ue.MODEL_FILE,
        graph,
        dict(tensors),
        provenance=ue.provenance(),
        graph_name=ue.GRAPH_NAME,
        **storage,
    )
    return read_parts(path)


def read_parts(path: Path) -> list[bytes]:
    return [part.read_bytes() for part in sorted(path.glob("*.krm"))]


class TestCheckpoint:
    def test_the_tied_embedding_is_read_from_the_saved_name(self, checkpoint_dir, model):
        """`encoder.embed_tokens.weight` exists in the checkpoint only as `shared.weight`."""
        checkpoint = ue.Checkpoint(checkpoint_dir)

        mapping = ue.checkpoint_keys(model, ["encoder.embed_tokens.weight"], checkpoint)

        assert "encoder.embed_tokens.weight" not in checkpoint.names()
        assert mapping == {"encoder.embed_tokens.weight": "shared.weight"}

    def test_a_missing_key_fails_loudly(self, checkpoint_dir, model, tmp_path):
        table = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        tensors = load_file(str(checkpoint_dir / ue.CHECKPOINT_SINGLE))
        del tensors[table]
        save_file(tensors, str(tmp_path / ue.CHECKPOINT_SINGLE))

        with pytest.raises(ue.Umt5ExportError, match="not determined uniquely"):
            ue.checkpoint_keys(model, [table], ue.Checkpoint(tmp_path))

    def test_a_non_f32_checkpoint_is_rejected(self, checkpoint_dir, tmp_path):
        """A checkpoint whose rounding starting point is not F32 is not read (going from bf16 to i8
        gives different values)."""
        tensors = load_file(str(checkpoint_dir / ue.CHECKPOINT_SINGLE))
        save_file(
            {key: value.to(torch.bfloat16) for key, value in tensors.items()},
            str(tmp_path / ue.CHECKPOINT_SINGLE),
        )

        with pytest.raises(ue.Umt5ExportError, match="only F32 is accepted"):
            ue.Checkpoint(tmp_path).read_rows("shared.weight", 0, 1)


#: The pair of tied aliases of the vocabulary embedding (the original index holds both — ADR 0122
#: decision 2).
SHARED, EMBED = "shared.weight", "encoder.embed_tokens.weight"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bin_split(directory: Path, shards: list[dict[str, torch.Tensor]]) -> dict[str, str]:
    """Write the original model's split pickle form (`pytorch_model-0000N-of-0000M.bin` + index) and
    return each shard's file name → sha256 (the shape of the table row's `shards` — the caller picks
    only what it needs)."""
    weight_map: dict[str, str] = {}
    digests: dict[str, str] = {}
    for number, tensors in enumerate(shards, start=1):
        name = f"pytorch_model-{number:05d}-of-{len(shards):05d}.bin"
        torch.save(tensors, directory / name)
        weight_map.update(dict.fromkeys(tensors, name))
        digests[name] = sha256_of(directory / name)
    (directory / UMT5_BIN_INDEX).write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}), encoding="utf-8"
    )
    return digests


def google_shards(
    model: Any, *, embed: torch.Tensor | None = None
) -> list[dict[str, torch.Tensor]]:
    """Split the small model in the original order: shard 1 = the vocabulary embedding pair + the
    first half of the encoder, shard 2 = the second half of the encoder, shard 3 = only the decoder
    and `lm_head` (a shard the encoder reader does not take).

    Passing `embed` sets `encoder.embed_tokens.weight` to that value (by default the same storage as
    `shared.weight` — the shape of the original files).
    """
    state = {key: value.detach() for key, value in model.state_dict().items()}
    encoder = sorted(key for key in state if key.startswith("encoder.") and key != EMBED)
    half = len(encoder) // 2
    first = {SHARED: state[SHARED], EMBED: state[SHARED] if embed is None else embed}
    first.update({key: state[key] for key in encoder[:half]})
    second = {key: state[key] for key in encoder[half:]}
    width = state[SHARED].shape[1]
    decoder = {
        "decoder.final_layer_norm.weight": torch.ones(width),
        "lm_head.weight": torch.zeros(4, width),
    }
    return [first, second, decoder]


def encoder_pin(digests: Mapping[str, str]) -> dict[str, str]:
    """The table row's `shards` (only shards 1 / 2, which hold the encoder weights)."""
    return {name: digest for name, digest in digests.items() if "-00003-" not in name}


@pytest.fixture
def no_unpickle(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record any call to `torch.load` (for the fault-injection checks of "fails before
    opening")."""
    opened: list[Path] = []

    def refuse(path: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(Path(path))
        raise AssertionError(f"unpickled: {path}")

    monkeypatch.setattr(torch, "load", refuse)
    return opened


class TestThePinnedPickleSplit:
    """The reader of the original model's split pickle form (ADR 0122 decision 2 — check the sha256
    of the table row, then read through mmap)."""

    @pytest.fixture
    def bin_dir(self, model: Any, tmp_path: Path) -> tuple[Path, dict[str, str]]:
        directory = tmp_path / "google"
        directory.mkdir()
        model.config.save_pretrained(directory)
        digests = write_bin_split(directory, google_shards(model))
        return directory, encoder_pin(digests)

    def test_it_reads_the_same_values_as_the_safetensors_checkpoint(self, bin_dir, checkpoint_dir):
        directory, pin = bin_dir
        pickled = ue.Checkpoint(directory, shards=pin)
        saved = ue.Checkpoint(checkpoint_dir)

        assert saved.names() <= pickled.names()
        for key in sorted(saved.names()):
            assert pickled.shape(key) == saved.shape(key), key
            assert torch.equal(pickled.read(key), saved.read(key)), key
            assert torch.equal(pickled.read_rows(key, 1, 3), saved.read_rows(key, 1, 3)), key

    def test_the_container_is_byte_identical_to_the_safetensors_one(
        self, bin_dir, prepared, tmp_path
    ):
        """With a different reader but the same values, the container matches in the bytes of every
        part (the small-model version of step a's prediction)."""
        directory, pin = bin_dir
        for name, export in (
            ("pickled", ue.prepare(directory, shards=pin, chunk_rows=CHUNK_ROWS)),
            ("saved", prepared),
        ):
            (tmp_path / name).mkdir()
            ue.write_container(export, tmp_path / name / ue.MODEL_FILE)

        assert read_parts(tmp_path / "pickled") == read_parts(tmp_path / "saved")

    def test_the_shards_are_derived_from_the_index(self, bin_dir):
        """Derive only the shards that hold the encoder weights (shard 3, which holds only the
        decoder and lm_head, is not read)."""
        directory, pin = bin_dir
        index = json.loads((directory / UMT5_BIN_INDEX).read_text(encoding="utf-8"))

        assert umt5_encoder_shards(index["weight_map"]) == frozenset(pin)
        assert "lm_head.weight" not in ue.Checkpoint(directory, shards=pin).names()

    def test_the_tied_pair_in_one_storage_maps_to_the_declared_target(self, bin_dir, model):
        directory, pin = bin_dir
        checkpoint = ue.Checkpoint(directory, shards=pin)

        assert {SHARED, EMBED} <= checkpoint.names()
        assert ue.checkpoint_keys(model, [EMBED], checkpoint) == {EMBED: SHARED}

    def test_a_bit_identical_copy_in_another_storage_maps_to_the_same_target(self, model, tmp_path):
        model.config.save_pretrained(tmp_path)
        copy = model.shared.weight.detach().clone()
        pin = encoder_pin(write_bin_split(tmp_path, google_shards(model, embed=copy)))

        assert ue.checkpoint_keys(model, [EMBED], ue.Checkpoint(tmp_path, shards=pin)) == {
            EMBED: SHARED
        }

    def test_a_safetensors_checkpoint_with_both_names_follows_the_same_rule(
        self, checkpoint_dir, model, tmp_path
    ):
        tensors = load_file(str(checkpoint_dir / ue.CHECKPOINT_SINGLE))
        tensors[EMBED] = tensors[SHARED].clone()
        save_file(tensors, str(tmp_path / ue.CHECKPOINT_SINGLE))

        assert ue.checkpoint_keys(model, [EMBED], ue.Checkpoint(tmp_path)) == {EMBED: SHARED}

    def test_a_tied_pair_that_differs_in_one_element_fails_loudly(self, model, tmp_path):
        """Fault injection: change one element of one of the aliases — either one read gives a
        different weight."""
        model.config.save_pretrained(tmp_path)
        drifted = model.shared.weight.detach().clone()
        drifted[-1, -1] = torch.nextafter(drifted[-1, -1], torch.tensor(float("inf")))
        pin = encoder_pin(write_bin_split(tmp_path, google_shards(model, embed=drifted)))

        with pytest.raises(ue.Umt5ExportError, match="differ within the checkpoint"):
            ue.checkpoint_keys(model, [EMBED], ue.Checkpoint(tmp_path, shards=pin))

    def test_a_flipped_byte_fails_before_anything_is_unpickled(self, bin_dir, no_unpickle):
        """Fault injection: change one byte of a shard — the sha256 disagrees with the table and it
        fails before `torch.load`."""
        directory, pin = bin_dir
        shard = directory / sorted(pin)[1]
        raw = bytearray(shard.read_bytes())
        raw[len(raw) // 2] ^= 0x01
        shard.write_bytes(bytes(raw))

        with pytest.raises(ue.Umt5ExportError, match="sha256"):
            ue.Checkpoint(directory, shards=pin)
        assert no_unpickle == []

    def test_a_missing_shard_fails_before_anything_is_unpickled(self, bin_dir, no_unpickle):
        """Fault injection: drop one shard (delete it from disk)."""
        directory, pin = bin_dir
        (directory / sorted(pin)[1]).unlink()

        with pytest.raises(ue.Umt5ExportError, match="is missing"):
            ue.Checkpoint(directory, shards=pin)
        assert no_unpickle == []

    def test_a_table_that_misses_a_derived_shard_fails_loudly(self, bin_dir, no_unpickle):
        """Fault injection: drop one shard from the table — it disagrees with the set derived from
        the index (a stale table)."""
        directory, pin = bin_dir
        stale = dict(sorted(pin.items())[:1])

        with pytest.raises(ue.Umt5ExportError, match="the table's shards"):
            ue.Checkpoint(directory, shards=stale)
        assert no_unpickle == []

    def test_an_index_without_the_table_row_is_never_unpickled(self, bin_dir, no_unpickle):
        """Fault injection: pass a directory holding only `pytorch_model.bin.index.json`, bypassing
        the table row."""
        directory, _ = bin_dir

        with pytest.raises(ue.Umt5ExportError, match=r"has neither .* nor"):
            ue.Checkpoint(directory)
        assert no_unpickle == []

    def test_a_non_f32_pickle_is_rejected(self, model, tmp_path):
        """Fault injection: a `.bin` that is not F32 (the sha256 matches the table) — the rounding
        starting point changes, so it is not read."""
        shards = [
            {key: value.to(torch.bfloat16) for key, value in shard.items()}
            for shard in google_shards(model)
        ]
        pin = encoder_pin(write_bin_split(tmp_path, shards))

        with pytest.raises(ue.Umt5ExportError, match="only F32 is accepted"):
            ue.Checkpoint(tmp_path, shards=pin).read_rows(SHARED, 0, 1)

    def test_a_legacy_pickle_is_named(self, model, tmp_path):
        """A pickle not in zip format (from an old `torch.save`) cannot be opened through mmap —
        the exception is wrapped and named."""
        digests = write_bin_split(tmp_path, google_shards(model))
        first = sorted(digests)[0]
        tensors = torch.load(tmp_path / first, weights_only=True)
        torch.save(tensors, tmp_path / first, _use_new_zipfile_serialization=False)
        pin = encoder_pin({**digests, first: sha256_of(tmp_path / first)})

        with pytest.raises(ue.Umt5ExportError, match=r"is not a zip-form torch\.save"):
            ue.Checkpoint(tmp_path, shards=pin).read(SHARED)

    def test_a_pickle_with_an_unlisted_global_is_refused(self, model, tmp_path):
        """Fault injection: a shard (the sha256 matches the table) containing a global that is not
        allowed (`datetime.date` — harmless, carrying no payload to execute) — the `weights_only`
        unpickler refuses it."""
        shards = google_shards(model)
        shards[0]["note"] = datetime.date(2026, 10, 4)
        pin = encoder_pin(write_bin_split(tmp_path, shards))

        with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
            ue.Checkpoint(tmp_path, shards=pin).read(SHARED)

    def test_a_shard_replaced_after_the_check_is_not_read(self, bin_dir, model):
        """Fault injection: after the check, replace the shard's path with other content (rename —
        how the HF cache writes). What is read stays the checked content."""
        directory, pin = bin_dir
        checkpoint = ue.Checkpoint(directory, shards=pin)
        shard = directory / sorted(pin)[0]
        swapped = shard.with_name("swapped.bin")
        torch.save(
            {key: torch.zeros_like(value) for key, value in google_shards(model)[0].items()},
            swapped,
        )
        swapped.replace(shard)

        assert torch.equal(checkpoint.read(SHARED), model.shared.weight.detach())

    def test_no_shard_stays_mapped_after_a_read(self, bin_dir, model):
        """No shard mmap remains after reading (holding it keeps the touched pages in file-backed
        RSS)."""
        directory, pin = bin_dir
        checkpoint = ue.Checkpoint(directory, shards=pin)
        shards = {str((directory / name).resolve()) for name in pin}

        checkpoint.read(SHARED)
        checkpoint.read_rows(sorted(checkpoint.names())[-1], 0, 1)
        checkpoint.shape(SHARED)
        ue.checkpoint_keys(model, [EMBED], checkpoint)

        assert shards.isdisjoint(mapped_files())


def mapped_files() -> set[str]:
    """The paths of the files this process has mmapped (column 6 of `/proc/self/maps`)."""
    with open("/proc/self/maps", encoding="utf-8") as stream:
        fields = (line.split(maxsplit=5) for line in stream)
        return {columns[5].strip() for columns in fields if len(columns) == 6}


class TestCommandLine:
    """An explicit argument that has no effect on the subcommand is refused (not silently
    ignored)."""

    @pytest.fixture
    def calls(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[Any, ...], Any]]:
        """Replace the subcommand's body with a recorder (no weights are read)."""
        recorded: list[tuple[str, tuple[Any, ...], Any]] = []

        def recorder(name: str) -> Any:
            def run(*args: Any, **kwargs: Any) -> dict[str, Any]:
                recorded.append((name, args, kwargs))
                return {}

            return run

        for name in (
            "prepare_summary",
            "write_series",
            "check_mask",
            "compare_mask",
            "reference_summary",
        ):
            monkeypatch.setattr(ue, name, recorder(name))
        return recorded

    @pytest.mark.parametrize(
        ("argv", "refused"),
        [
            (["prepare", "--model", DEFAULT_MODEL], "--model"),
            (["write", "--model", DEFAULT_MODEL], "--model"),
            (["check-mask", "--upstream", UMT5_DEFAULT_MODEL], "--upstream"),
            (["compare-mask", "--upstream", UMT5_DEFAULT_MODEL, "--out", "seat"], "--upstream"),
            (["compare-mask", "--model", DEFAULT_MODEL, "--out", "seat"], "--model"),
            (["reference", "--check"], "--check"),
            (["write", "--dtype", "f32"], "--dtype"),
            (["reference", "--out", "seat"], "--out"),
        ],
    )
    def test_an_argument_the_command_does_not_use_is_refused(self, argv, refused, calls, capsys):
        """Even a value equal to the default is refused when given explicitly (a run that believes
        it set an axis that has no effect does not start)."""
        with pytest.raises(SystemExit) as exited:
            ue.main(argv)

        assert exited.value.code == 2
        assert f"{refused} does not apply to {argv[0]}" in capsys.readouterr().err
        assert calls == []

    def test_omitted_arguments_take_their_defaults(self, calls, capsys):
        for argv in (["prepare"], ["write"], ["check-mask"], ["reference"]):
            ue.main(argv)

        assert calls == [
            ("prepare_summary", (UMT5_DEFAULT_MODEL,), {}),
            ("write_series", (UMT5_DEFAULT_MODEL,), {"check": False}),
            ("check_mask", (DEFAULT_MODEL, "bf16", None), {}),
            ("reference_summary", (UMT5_DEFAULT_MODEL, DEFAULT_MODEL), {}),
        ]

    def test_the_arguments_a_command_uses_are_passed_through(self, calls, capsys):
        ue.main(["write", "--upstream", UMT5_DEFAULT_MODEL, "--check"])
        ue.main(["check-mask", "--model", DEFAULT_MODEL, "--dtype", "f32", "--out", "seat"])
        ue.main(["compare-mask", "--out", "seat"])
        ue.main(["reference", "--upstream", UMT5_DEFAULT_MODEL, "--model", DEFAULT_MODEL])

        assert calls == [
            ("write_series", (UMT5_DEFAULT_MODEL,), {"check": True}),
            ("check_mask", (DEFAULT_MODEL, "f32", Path("seat")), {}),
            ("compare_mask", (Path("seat"),), {}),
            ("reference_summary", (UMT5_DEFAULT_MODEL, DEFAULT_MODEL), {}),
        ]


class TestMetaTrace:
    def test_the_graph_is_the_real_weight_export(self, prepared, model):
        """The graph walked with meta weights = the graph of the export with real weights (before
        rounding in f32)."""
        wrapper = up.Umt5EncoderTokens(model)
        graph, _ = ue.trace(wrapper)

        assert prepared.graph.to_json() == graph.to_json()

    def test_only_the_weights_stay_meta(self, prepared):
        """Only the i8 targets stay on meta; the remaining weights are the checkpoint's tensors."""
        meta = sorted(key for key, value in prepared.tensors.items() if value.is_meta)

        assert meta == sorted(prepared.fixed)
        assert all(not prepared.tensors[key].is_meta for key in prepared.plain)


class TestStorage:
    def test_linear_and_embedding_are_i8_tables_and_norms_f32(self, prepared):
        kinds = sorted({ue.storage_kind(key) for key in prepared.fixed})
        plain = sorted({ue.storage_kind(key) for key in prepared.plain})

        assert len(prepared.fixed) == LAYERS * LINEARS_PER_LAYER + 1
        assert kinds == ["embed_tokens", "linear"]
        assert plain == ["norm", up.RELATIVE_BIAS_ATTRIBUTE]
        assert sum(ue.storage_kind(key) == "norm" for key in prepared.plain) == 2 * LAYERS + 1

    def test_the_container_matches_the_whole_tensor_path_byte_for_byte(
        self, prepared, model, tmp_path
    ):
        """The official writer (i8 per block of rows) matches the plain form (round in f32 → i8 +
        the tables explicitly F32) in the bytes of every part."""
        whole = probe.export_probe(model)

        expected = write(
            tmp_path / "whole",
            whole.graph,
            whole.tensors,
            weight_dtype="i8",
            weight_scales=whole.scales,
            weight_dtype_overrides=whole.overrides,
        )
        (tmp_path / "rows").mkdir()
        ue.write_container(prepared, tmp_path / "rows" / ue.MODEL_FILE)
        actual = read_parts(tmp_path / "rows")

        assert len(actual) == len(expected) > 0
        assert actual == expected

    def test_a_shifted_chunk_breaks_the_match(self, prepared, model):
        """The pair (not tautological): shifting a block's scale by one row splits the packed bytes
        from the plain form."""
        whole = probe.export_probe(model)
        key = "encoder.embed_tokens.weight"
        fixed = prepared.fixed[key]
        weight = whole.tensors[key]

        restored = fixed.packed.to(torch.float32) * fixed.scale
        shifted = fixed.packed.to(torch.float32) * torch.roll(fixed.scale, 1, dims=0)

        assert torch.equal(restored, weight)
        assert not torch.equal(shifted, weight)


class TestTheSeriesCheck:
    """The comparison of `write --check` (the written container against an existing container, by
    the sha256 of the part sequence)."""

    @pytest.fixture
    def pair(self, prepared, tmp_path) -> tuple[Path, Path]:
        written, existing = tmp_path / "written", tmp_path / "existing"
        for place in (written, existing):
            place.mkdir()
            ue.write_container(prepared, place / ue.MODEL_FILE)
        return written / ue.MODEL_FILE, existing / ue.MODEL_FILE

    def test_the_same_material_writes_the_same_container(self, pair):
        written, existing = pair

        ue.assert_same_container(written, existing)
        assert len(ue.part_digests(written)) > 1

    def test_one_flipped_byte_is_named_by_its_part(self, pair):
        """Fault injection: flipping one byte of the existing last part fails at that index."""
        written, existing = pair
        parts = sorted(existing.parent.glob("*.krm"))
        last = bytearray(parts[-1].read_bytes())
        last[-1] ^= 0xFF
        parts[-1].write_bytes(bytes(last))

        with pytest.raises(ue.Umt5ExportError, match=rf"part \[{len(parts) - 1}\]"):
            ue.assert_same_container(written, existing)

    def test_another_part_count_is_refused(self, pair, tmp_path):
        """A different part count fails before the contents are compared (a container written with
        another split)."""
        written, _ = pair
        single = tmp_path / "single"
        single.mkdir()
        (single / "model-00001-of-00002.krm").write_bytes(b"0")
        (single / "model-00002-of-00002.krm").write_bytes(b"1")

        with pytest.raises(ue.Umt5ExportError, match="part counts differ"):
            ue.assert_same_container(written, single / ue.MODEL_FILE)


class TestQuantizeRows:
    @pytest.mark.parametrize("chunk_rows", [1, CHUNK_ROWS, 10_000])
    def test_the_chunk_size_does_not_change_the_values(self, checkpoint_dir, chunk_rows):
        """The same packed bytes and scale regardless of block size (one row at a time, partial,
        the whole as one block)."""
        checkpoint = ue.Checkpoint(checkpoint_dir)
        reference = ue.quantize_rows(checkpoint, "shared.weight", chunk_rows=10_000)

        actual = ue.quantize_rows(checkpoint, "shared.weight", chunk_rows=chunk_rows)

        assert torch.equal(actual.packed, reference.packed)
        assert torch.equal(actual.scale, reference.scale)


class TestMemoryMonitor:
    def test_a_stage_sees_what_it_allocates(self):
        """256 MiB allocated and touched within a stage shows up in that stage's peak (VmHWM)."""
        with ue.MemoryMonitor(interval=0.05) as monitor:
            with monitor.stage("idle") as idle:
                pass
            with monitor.stage("allocate") as allocate:
                block = torch.ones(64 << 20, dtype=torch.float32)
                del block

        assert [record.name for record in monitor.records] == ["idle", "allocate"]
        assert allocate.hwm - idle.hwm > 0.2
        assert allocate.seconds >= 0


class TestReferenceAxes:
    def test_the_two_axes_name_their_own_rows(self):
        """`reference` mixes the umT5 upstream (config and unquantized weights) with the snapshot
        of Wan's text side (the cases' id sequences) — the golden's metadata names each row
        (ADR 0122 decision 3)."""
        from wan.sources import SOURCES, UMT5_SOURCES

        axes = ue.reference_axes("xxl", "t2v-1.3b")

        encoder, cases = UMT5_SOURCES["xxl"].source, SOURCES["t2v-1.3b"]
        assert axes == {
            "encoder": {"model": "xxl", "repo": encoder.repo, "revision": encoder.revision},
            "cases": {
                "model": "t2v-1.3b",
                "repo": cases.repo,
                "revision": cases.revision,
                "subfolder": "tokenizer",
            },
        }
        assert encoder.revision != cases.revision
