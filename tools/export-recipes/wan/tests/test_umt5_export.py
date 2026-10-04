"""`wan/umt5_export.py`（meta で trace し、重みを checkpoint から 1 本ずつ i8 にする形）の約束事
（ADR 0119 段 10b — 決定 5 / 6）。

固定するのは、壊れても例外が出ず**容器のバイトだけが静かにずれる**側:

- 重みを持たない（meta）上流で trace したグラフが、実重みの export と JSON で同一
- 行の塊ごとに作った i8（`fixed_weights`）で書いた容器が、素直な形（f32 で `fake_quant_i8` →
  `weight_dtype="i8"` + 表の F32 明示）で書いた容器と**全 part のバイトで一致**する — 塊の境目が
  行の途中に来ない・tied な語彙埋め込みを `shared.weight` から読む・表を丸めない、の全部をここで縛る
- 格納の内訳: linear と語彙埋め込みは i8、相対位置の表と RMSNorm は f32
- checkpoint のキーが決まらない・F32 でない、は fail loudly
- 本家の pickle 分割形（ADR 0122 決定 2）: 表の行の sha256 を照合してから mmap で読み、
  safetensors と同じ値・同じ容器になる。tied な別名の規則（同じ storage / ビット一致 →
  `shared.weight`・違えば fail loudly）。sha256・shard の欠落・表の古さ・表を経ない索引は
  `torch.load` の前に落ちる。照合の後に path が差し替わっても照合した内容を読む・読んだ後に
  shard の mmap が残らない・許可外のグローバルを含む pickle は拒む
- サブコマンドに効かない引数（軸 `--upstream` / `--model` ほか）の明示は拒む
- 正式な書き口（{@link wan.umt5_export.write_container}）が上の素直な形と同じバイトを書き、既存の
  容器との照合（{@link wan.umt5_export.assert_same_container}）が 1 バイトの違いも part で名指しする

乱数初期化の小さな umT5（`wan.umt5_probe.TINY_CONFIG`）を `save_pretrained` した checkpoint で回す —
実重みは読まない。
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

#: 小模型の linear の本数（層ごとに q / k / v / o と wi_0 / wi_1 / wo）。
LINEARS_PER_LAYER = 7

#: 塊の行数（語彙 384・d_ff 160・d_model 64 のどれも割り切らない — 半端な塊を必ず踏む）。
CHUNK_ROWS = 7


@pytest.fixture(scope="module")
def model() -> Any:
    pytest.importorskip("transformers")
    return probe.tiny_model()


@pytest.fixture(scope="module")
def checkpoint_dir(model: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """上流の `save_pretrained` の形（config + 単一の safetensors — tied な重みは片方だけ）。"""
    directory = tmp_path_factory.mktemp("umt5-tiny")
    model.save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def prepared(checkpoint_dir: Path) -> ue.Umt5Export:
    return ue.prepare(checkpoint_dir, chunk_rows=CHUNK_ROWS)


def write(path: Path, graph: Any, tensors: Any, **storage: Any) -> list[bytes]:
    """容器を書いて全 part のバイトを返す（出所とグラフ名は正式な書き口と同じ — 差はバイトの
    中身だけに出る）。"""
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
        """`encoder.embed_tokens.weight` は checkpoint に `shared.weight` としてしか無い。"""
        checkpoint = ue.Checkpoint(checkpoint_dir)

        mapping = ue.checkpoint_keys(model, ["encoder.embed_tokens.weight"], checkpoint)

        assert "encoder.embed_tokens.weight" not in checkpoint.names()
        assert mapping == {"encoder.embed_tokens.weight": "shared.weight"}

    def test_a_missing_key_fails_loudly(self, checkpoint_dir, model, tmp_path):
        table = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        tensors = load_file(str(checkpoint_dir / ue.CHECKPOINT_SINGLE))
        del tensors[table]
        save_file(tensors, str(tmp_path / ue.CHECKPOINT_SINGLE))

        with pytest.raises(ue.Umt5ExportError, match="1 つに決まらない"):
            ue.checkpoint_keys(model, [table], ue.Checkpoint(tmp_path))

    def test_a_non_f32_checkpoint_is_rejected(self, checkpoint_dir, tmp_path):
        """丸めの出発点が F32 でない checkpoint は読まない（bf16 から i8 にすると別の値になる）。"""
        tensors = load_file(str(checkpoint_dir / ue.CHECKPOINT_SINGLE))
        save_file(
            {key: value.to(torch.bfloat16) for key, value in tensors.items()},
            str(tmp_path / ue.CHECKPOINT_SINGLE),
        )

        with pytest.raises(ue.Umt5ExportError, match="F32 だけ"):
            ue.Checkpoint(tmp_path).read_rows("shared.weight", 0, 1)


#: 語彙埋め込みの tied な別名の対（本家の索引は両方を持つ — ADR 0122 決定 2）。
SHARED, EMBED = "shared.weight", "encoder.embed_tokens.weight"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bin_split(directory: Path, shards: list[dict[str, torch.Tensor]]) -> dict[str, str]:
    """本家の形の pickle 分割形（`pytorch_model-0000N-of-0000M.bin` + 索引）を書き、各 shard の
    ファイル名 → sha256 を返す（表の行の `shards` の形 — 渡す側が要る分だけ選ぶ）。"""
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
    """小模型を本家の並びに割る: shard 1 = 語彙埋め込みの対 + encoder の前半・shard 2 = encoder の
    後半・shard 3 = decoder と `lm_head` だけ（encoder の読み手は取らない shard）。

    `embed` を渡すと `encoder.embed_tokens.weight` をその値にする（既定は `shared.weight` と同じ
    storage — 本家の実物の形）。
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
    """表の行の `shards`（encoder の重みが載る shard 1 / 2 だけ）。"""
    return {name: digest for name, digest in digests.items() if "-00003-" not in name}


@pytest.fixture
def no_unpickle(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """`torch.load` を呼ばれたら記録する（「開く前に落ちる」の故障注入の検査用）。"""
    opened: list[Path] = []

    def refuse(path: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(Path(path))
        raise AssertionError(f"unpickle された: {path}")

    monkeypatch.setattr(torch, "load", refuse)
    return opened


class TestThePinnedPickleSplit:
    """本家の pickle 分割形の読み口（ADR 0122 決定 2 — 表の行の sha256 を照合してから mmap で
    読む）。"""

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
        """読み口が違っても値が同じなら、容器は全 part のバイトで一致する（段 a の予測の
        小模型版）。"""
        directory, pin = bin_dir
        for name, export in (
            ("pickled", ue.prepare(directory, shards=pin, chunk_rows=CHUNK_ROWS)),
            ("saved", prepared),
        ):
            (tmp_path / name).mkdir()
            ue.write_container(export, tmp_path / name / ue.MODEL_FILE)

        assert read_parts(tmp_path / "pickled") == read_parts(tmp_path / "saved")

    def test_the_shards_are_derived_from_the_index(self, bin_dir):
        """encoder の重みが載る shard だけを導く（decoder と lm_head だけの shard 3 は
        読まない）。"""
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
        """故障注入: 別名の片方の 1 要素だけを変える — どちらを読んでも別の重みになる。"""
        model.config.save_pretrained(tmp_path)
        drifted = model.shared.weight.detach().clone()
        drifted[-1, -1] = torch.nextafter(drifted[-1, -1], torch.tensor(float("inf")))
        pin = encoder_pin(write_bin_split(tmp_path, google_shards(model, embed=drifted)))

        with pytest.raises(ue.Umt5ExportError, match="値が checkpoint の中で違う"):
            ue.checkpoint_keys(model, [EMBED], ue.Checkpoint(tmp_path, shards=pin))

    def test_a_flipped_byte_fails_before_anything_is_unpickled(self, bin_dir, no_unpickle):
        """故障注入: shard の 1 バイトを変える — sha256 が表と食い違い、`torch.load` の前に
        落ちる。"""
        directory, pin = bin_dir
        shard = directory / sorted(pin)[1]
        raw = bytearray(shard.read_bytes())
        raw[len(raw) // 2] ^= 0x01
        shard.write_bytes(bytes(raw))

        with pytest.raises(ue.Umt5ExportError, match="sha256"):
            ue.Checkpoint(directory, shards=pin)
        assert no_unpickle == []

    def test_a_missing_shard_fails_before_anything_is_unpickled(self, bin_dir, no_unpickle):
        """故障注入: shard を 1 本外す（ディスクから消す）。"""
        directory, pin = bin_dir
        (directory / sorted(pin)[1]).unlink()

        with pytest.raises(ue.Umt5ExportError, match="が無い"):
            ue.Checkpoint(directory, shards=pin)
        assert no_unpickle == []

    def test_a_table_that_misses_a_derived_shard_fails_loudly(self, bin_dir, no_unpickle):
        """故障注入: 表から shard を 1 本外す — 索引から導いた集合と食い違う（表が古い）。"""
        directory, pin = bin_dir
        stale = dict(sorted(pin.items())[:1])

        with pytest.raises(ue.Umt5ExportError, match="表の shard"):
            ue.Checkpoint(directory, shards=stale)
        assert no_unpickle == []

    def test_an_index_without_the_table_row_is_never_unpickled(self, bin_dir, no_unpickle):
        """故障注入: 表の行を経ずに `pytorch_model.bin.index.json` だけのディレクトリを渡す。"""
        directory, _ = bin_dir

        with pytest.raises(ue.Umt5ExportError, match=r"も .* も無い"):
            ue.Checkpoint(directory)
        assert no_unpickle == []

    def test_a_non_f32_pickle_is_rejected(self, model, tmp_path):
        """故障注入: F32 でない `.bin`（sha256 は表と一致）— 丸めの出発点が変わるので読まない。"""
        shards = [
            {key: value.to(torch.bfloat16) for key, value in shard.items()}
            for shard in google_shards(model)
        ]
        pin = encoder_pin(write_bin_split(tmp_path, shards))

        with pytest.raises(ue.Umt5ExportError, match="F32 だけ"):
            ue.Checkpoint(tmp_path, shards=pin).read_rows(SHARED, 0, 1)

    def test_a_legacy_pickle_is_named(self, model, tmp_path):
        """zip 形式でない（旧い `torch.save` の）pickle は mmap で開けない — 例外を名指しで包む。"""
        digests = write_bin_split(tmp_path, google_shards(model))
        first = sorted(digests)[0]
        tensors = torch.load(tmp_path / first, weights_only=True)
        torch.save(tensors, tmp_path / first, _use_new_zipfile_serialization=False)
        pin = encoder_pin({**digests, first: sha256_of(tmp_path / first)})

        with pytest.raises(ue.Umt5ExportError, match=r"zip 形式の torch\.save でない"):
            ue.Checkpoint(tmp_path, shards=pin).read(SHARED)

    def test_a_pickle_with_an_unlisted_global_is_refused(self, model, tmp_path):
        """故障注入: 許可されていないグローバル（`datetime.date` — 無害で、実行する payload は
        持たない）を含む shard（sha256 は表と一致）— `weights_only` の unpickler が拒む。"""
        shards = google_shards(model)
        shards[0]["note"] = datetime.date(2026, 10, 4)
        pin = encoder_pin(write_bin_split(tmp_path, shards))

        with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
            ue.Checkpoint(tmp_path, shards=pin).read(SHARED)

    def test_a_shard_replaced_after_the_check_is_not_read(self, bin_dir, model):
        """故障注入: 照合の後で shard の path を別の中身に差し替える（rename — HF のキャッシュの
        書き方）。読むのは照合した内容のまま。"""
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
        """読んだ後に shard の mmap が残らない（握り続けると、触ったページが file-backed の
        RSS に乗ったまま残る）。"""
        directory, pin = bin_dir
        checkpoint = ue.Checkpoint(directory, shards=pin)
        shards = {str((directory / name).resolve()) for name in pin}

        checkpoint.read(SHARED)
        checkpoint.read_rows(sorted(checkpoint.names())[-1], 0, 1)
        checkpoint.shape(SHARED)
        ue.checkpoint_keys(model, [EMBED], checkpoint)

        assert shards.isdisjoint(mapped_files())


def mapped_files() -> set[str]:
    """このプロセスが mmap しているファイルの path（`/proc/self/maps` の 6 列目）。"""
    with open("/proc/self/maps", encoding="utf-8") as stream:
        fields = (line.split(maxsplit=5) for line in stream)
        return {columns[5].strip() for columns in fields if len(columns) == 6}


class TestCommandLine:
    """サブコマンドに効かない引数を明示したら拒む（黙って無視しない）。"""

    @pytest.fixture
    def calls(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[Any, ...], Any]]:
        """サブコマンドの本体を記録係に差し替える（重みは読まない）。"""
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
        """既定値と同じ値でも、明示したら拒む（効かない軸を指定したつもりの回を走らせない）。"""
        with pytest.raises(SystemExit) as exited:
            ue.main(argv)

        assert exited.value.code == 2
        assert f"{refused} は {argv[0]} に効かない" in capsys.readouterr().err
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
        """meta の重みで辿ったグラフ = 実重み（f32 で丸める前）の export のグラフ。"""
        wrapper = up.Umt5EncoderTokens(model)
        graph, _ = ue.trace(wrapper)

        assert prepared.graph.to_json() == graph.to_json()

    def test_only_the_weights_stay_meta(self, prepared):
        """meta のまま残るのは i8 の対象だけで、残りの重みは checkpoint の実体。"""
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
        """正式な書き口（行の塊ごとの i8）が、素直な形（f32 で丸め → i8 + 表の F32 明示）と
        全 part のバイトで一致する。"""
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
        """対（非恒真）: 塊の scale を 1 行ずらすと packed が素直な形と割れる。"""
        whole = probe.export_probe(model)
        key = "encoder.embed_tokens.weight"
        fixed = prepared.fixed[key]
        weight = whole.tensors[key]

        restored = fixed.packed.to(torch.float32) * fixed.scale
        shifted = fixed.packed.to(torch.float32) * torch.roll(fixed.scale, 1, dims=0)

        assert torch.equal(restored, weight)
        assert not torch.equal(shifted, weight)


class TestTheSeriesCheck:
    """`write --check` の照合（書いた容器と既存の容器を part 列の sha256 で突き合わせる）。"""

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
        """故障注入: 既存の最後の part の 1 バイトを反転すると、その添字で落ちる。"""
        written, existing = pair
        parts = sorted(existing.parent.glob("*.krm"))
        last = bytearray(parts[-1].read_bytes())
        last[-1] ^= 0xFF
        parts[-1].write_bytes(bytes(last))

        with pytest.raises(ue.Umt5ExportError, match=rf"part \[{len(parts) - 1}\]"):
            ue.assert_same_container(written, existing)

    def test_another_part_count_is_refused(self, pair, tmp_path):
        """part の本数が違えば、中身を比べる前に落ちる（別の分割で書いた容器）。"""
        written, _ = pair
        single = tmp_path / "single"
        single.mkdir()
        (single / "model-00001-of-00002.krm").write_bytes(b"0")
        (single / "model-00002-of-00002.krm").write_bytes(b"1")

        with pytest.raises(ue.Umt5ExportError, match="part の本数"):
            ue.assert_same_container(written, single / ue.MODEL_FILE)


class TestQuantizeRows:
    @pytest.mark.parametrize("chunk_rows", [1, CHUNK_ROWS, 10_000])
    def test_the_chunk_size_does_not_change_the_values(self, checkpoint_dir, chunk_rows):
        """塊の大きさ（1 行ずつ・半端・全体を 1 塊）に依らず同じ packed と scale。"""
        checkpoint = ue.Checkpoint(checkpoint_dir)
        reference = ue.quantize_rows(checkpoint, "shared.weight", chunk_rows=10_000)

        actual = ue.quantize_rows(checkpoint, "shared.weight", chunk_rows=chunk_rows)

        assert torch.equal(actual.packed, reference.packed)
        assert torch.equal(actual.scale, reference.scale)


class TestMemoryMonitor:
    def test_a_stage_sees_what_it_allocates(self):
        """段の中で確保して触った 256 MiB が、その段の山（VmHWM）に出る。"""
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
        """`reference` は umT5 の上流（config と量子化しない重み）と Wan の text 側の snapshot
        （ケースの id 列）を混ぜて使う — golden のメタはそれぞれの行を名指しする（ADR 0122
        決定 3）。"""
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
