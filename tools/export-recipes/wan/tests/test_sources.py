"""`wan/sources.py` の約束事 — 上流取得元の pin と、取得した部品の中身。

pin の形（40 桁の SHA・取得する部品の集合）は重み無しで見る。取得した config と
パラメータ数は {@link wan_snapshot}（取得済みでなければ SKIP）で見る。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from wan import sources


class TestPin:
    def test_every_revision_is_a_full_commit_sha(self):
        """ブランチ名や短縮 SHA は既定ブランチの移動で読む中身が黙って変わる。"""
        for source in sources.SOURCES.values():
            assert re.fullmatch(r"[0-9a-f]{40}", source.revision), source

    def test_the_default_model_is_the_diffusers_repo_of_t2v_1_3b(self):
        source = sources.SOURCES[sources.DEFAULT_MODEL]

        assert source.repo == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        assert source.license == "apache-2.0"

    def test_it_fetches_dit_vae_and_scheduler_but_never_the_text_encoder(self):
        """umT5（f32 約 22.7 GB）を取得対象に入れると、31 GiB 機の参照が別プロセスに分かれない。"""
        parts = sources.COMPONENTS[sources.DEFAULT_MODEL]
        fetched = {part.subfolder for part in parts if part.fetch}
        skipped = {part.subfolder for part in parts if not part.fetch}

        assert fetched == {"transformer", "vae", "scheduler"}
        assert skipped == {"text_encoder", "tokenizer"}
        assert not any(
            pattern.startswith(("text_encoder", "tokenizer"))
            for pattern in sources.allow_patterns()
        )

    def test_the_text_embedding_process_takes_exactly_the_skipped_parts(self):
        """`wan.text_embeds` の取得口（`text_snapshot`）が取るのは DiT / VAE 側が取らない
        部品だけ。"""
        assert set(sources.text_components(sources.DEFAULT_MODEL)) == {"text_encoder", "tokenizer"}


class TestSafetensorsHeader:
    def test_it_counts_elements_from_the_header_only(self, tmp_path: Path):
        torch = pytest.importorskip("torch")
        from safetensors.torch import save_file

        save_file(
            {"a": torch.zeros(3, 4), "b": torch.zeros(5, dtype=torch.float16)},
            tmp_path / "part.safetensors",
        )

        assert sources.count_parameters(tmp_path) == 17

    def test_a_directory_without_safetensors_fails_loudly(self, tmp_path: Path):
        with pytest.raises(sources.WanSourceError):
            sources.count_parameters(tmp_path)


class TestFetchedSnapshot:
    def test_the_scheduler_is_flow_unipc_with_shift_3(self, wan_snapshot: Path):
        """参照の設定は配布物の config をそのまま使う（ADR 0118 決定 5）。"""
        config = json.loads((wan_snapshot / "scheduler" / "scheduler_config.json").read_text())

        assert config["_class_name"] == "UniPCMultistepScheduler"
        assert config["flow_shift"] == 3.0
        assert config["use_flow_sigmas"] is True
        assert config["prediction_type"] == "flow_prediction"
        assert (config["solver_order"], config["solver_type"]) == (2, "bh2")
        assert config["final_sigmas_type"] == "zero"

    def test_the_transformer_and_vae_configs_have_the_expected_shape(self, wan_snapshot: Path):
        transformer = json.loads((wan_snapshot / "transformer" / "config.json").read_text())
        vae = json.loads((wan_snapshot / "vae" / "config.json").read_text())

        assert transformer["_class_name"] == "WanTransformer3DModel"
        assert transformer["num_layers"] == 30
        assert transformer["num_attention_heads"] * transformer["attention_head_dim"] == 1536
        assert transformer["text_dim"] == 4096
        assert transformer["patch_size"] == [1, 2, 2]
        assert vae["_class_name"] == "AutoencoderKLWan"
        assert vae["z_dim"] == 16

    def test_the_parameter_counts_match_the_research(self, wan_snapshot: Path):
        assert (
            sources.check_snapshot(wan_snapshot)
            == sources.EXPECTED_PARAMETERS[sources.DEFAULT_MODEL]
        )


class TestTheTi2v5bRow:
    """Wan2.2 TI2V-5B の行（ADR 0121 決定 1）— pin・取得範囲・期待パラメータ数がモデル別の
    表に載る。"""

    MODEL = "ti2v-5b"

    def test_it_pins_the_diffusers_repo_at_the_decided_commit(self):
        source = sources.SOURCES[self.MODEL]

        assert source.repo == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        assert source.revision == "b8fff7315c768468a5333511427288870b2e9635"
        assert source.license == "apache-2.0"

    def test_it_fetches_dit_vae_and_scheduler_and_never_the_text_parts(self):
        """umT5 は越境参照・トークナイザ資産は Wan2.1 の系列から写す（決定 9）— どの経路でも
        取らない。"""
        parts = sources.COMPONENTS[self.MODEL]

        assert {part.subfolder for part in parts if part.fetch} == {
            "transformer",
            "vae",
            "scheduler",
        }
        assert {part.subfolder for part in parts if not part.fetch} == {"text_encoder", "tokenizer"}
        assert sources.text_components(self.MODEL) == ()
        assert sources.allow_patterns(self.MODEL) == [
            "README.md",
            "model_index.json",
            "transformer/*",
            "vae/*",
            "scheduler/*",
        ]

    def test_asking_for_its_text_snapshot_fails_before_any_download(self):
        pytest.importorskip("huggingface_hub")

        with pytest.raises(sources.WanSourceError, match="決定 9"):
            sources.text_snapshot(self.MODEL, fetch=True)

    def test_the_expected_parameters_are_the_counts_of_the_adr(self):
        assert sources.EXPECTED_PARAMETERS[self.MODEL] == {
            "transformer": 4_999_787_712,
            "vae": 704_688_668,
        }

    def test_every_model_has_a_row_in_every_table(self):
        assert set(sources.COMPONENTS) == set(sources.SOURCES)
        assert set(sources.EXPECTED_PARAMETERS) == set(sources.SOURCES)


class TestTheModelKeyedCheck:
    """`check_snapshot` は渡したモデルの行で数を突き合わせる（別のモデルの行で通らない）。"""

    def _snapshot(self, tmp_path: Path) -> Path:
        torch = pytest.importorskip("torch")
        from safetensors.torch import save_file

        for name, size in (("transformer", 6), ("vae", 4)):
            (tmp_path / name).mkdir()
            save_file({"w": torch.zeros(size)}, tmp_path / name / "part.safetensors")
        return tmp_path

    def test_the_row_of_the_named_model_is_used(self, tmp_path: Path, monkeypatch):
        snapshot = self._snapshot(tmp_path)
        monkeypatch.setattr(
            sources,
            "EXPECTED_PARAMETERS",
            {"a": {"transformer": 6, "vae": 4}, "b": {"transformer": 6, "vae": 5}},
        )

        assert sources.check_snapshot(snapshot, "a") == {"transformer": 6, "vae": 4}
        with pytest.raises(sources.WanSourceError, match="b: パラメータ数"):
            sources.check_snapshot(snapshot, "b")

    def test_an_unknown_model_fails_loudly(self, tmp_path: Path):
        with pytest.raises(sources.WanSourceError, match="期待パラメータ数が無い"):
            sources.check_snapshot(tmp_path, "unknown")
        with pytest.raises(sources.WanSourceError, match="部品の表が無い"):
            sources.allow_patterns("unknown")


#: Wan2.1 に固定した台本（系列の置き場・ケースの表・テキスト段が Wan2.1 — `--model` の受理は
#: `WAN21_MODELS`）と、引数の残り（必須の位置引数）。
WAN21_SCRIPTS = (
    ("wan.export_dit", ["--verify"]),
    ("wan.dit_host_fixture", []),
    ("wan.scheduler_ref", []),
    ("wan.few_step_ref", []),
    ("wan.text_embeds", []),
    ("wan.umt5_export", ["prepare"]),
    ("wan.umt5_tokenizer", []),
    ("wan.umt5_host_fixture", []),
    ("wan.pipeline_ref", ["--smoke"]),
)


@pytest.mark.parametrize(("module", "rest"), WAN21_SCRIPTS, ids=[m for m, _ in WAN21_SCRIPTS])
def test_the_wan21_scripts_refuse_the_ti2v_5b_model(module: str, rest: list[str], capsys):
    """表に `ti2v-5b` が載っても、Wan2.1 の系列へ書く台本は 5B を受けない（引数の段で止まる）。"""
    pytest.importorskip("diffusers")
    import importlib

    script = importlib.import_module(module)

    with pytest.raises(SystemExit) as stopped:
        script.main(["--model", "ti2v-5b", *rest])

    assert stopped.value.code == 2
    assert "invalid choice: 'ti2v-5b'" in capsys.readouterr().err
    assert sources.WAN21_MODELS == (sources.DEFAULT_MODEL,)


class TestTheUmt5Row:
    """umT5 の上流の表（本家 `google/umt5-xxl` — ADR 0122 決定 1 / 2）。"""

    def test_it_pins_a_full_commit_and_full_shard_digests(self):
        for row in sources.UMT5_SOURCES.values():
            assert re.fullmatch(r"[0-9a-f]{40}", row.source.revision), row
            assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in row.shards.values())

    def test_the_encoder_shards_are_derived_from_the_index(self):
        weight_map = {
            "shared.weight": "a.bin",
            "encoder.block.0.layer.0.SelfAttention.q.weight": "b.bin",
            "decoder.embed_tokens.weight": "a.bin",
            "decoder.block.0.layer.0.SelfAttention.q.weight": "c.bin",
            "lm_head.weight": "d.bin",
        }

        assert sources.umt5_encoder_shards(weight_map) == frozenset({"a.bin", "b.bin"})

    def test_an_index_that_disagrees_with_the_table_fails_loudly(self, tmp_path: Path):
        """索引が動いた（encoder の重みが表に無い shard に載る）・表が古い、を黙って通さない。"""
        index = tmp_path / sources.UMT5_BIN_INDEX
        index.write_text(
            json.dumps({"weight_map": {"shared.weight": "a.bin", "encoder.final.weight": "b.bin"}}),
            encoding="utf-8",
        )

        with pytest.raises(sources.WanSourceError, match="表の shard"):
            sources.pinned_umt5_shards(index, {"a.bin": "0" * 64})
        assert sources.pinned_umt5_shards(index, {"a.bin": "0" * 64, "b.bin": "1" * 64}) == {
            "shared.weight": "a.bin",
            "encoder.final.weight": "b.bin",
        }

    def test_an_unknown_model_fails_before_any_download(self):
        with pytest.raises(sources.WanSourceError, match="上流の表に無い"):
            sources.umt5_snapshot("base")


class TestTheFetchedUmt5Snapshot:
    def test_the_cached_snapshot_has_the_pinned_shards(self):
        """取得済みの機だけ（無ければ SKIP）: 索引から導いた shard が表の 3 本で、config が
        読める。"""
        pytest.importorskip("huggingface_hub")
        try:
            snapshot = sources.umt5_snapshot("xxl")
        except sources.WanSourceError as error:
            pytest.skip(f"本家 umT5 の shard が手元に無い: {error}")
        pinned = sources.UMT5_SOURCES["xxl"].shards

        assert set(
            sources.pinned_umt5_shards(snapshot / sources.UMT5_BIN_INDEX, pinned).values()
        ) == set(pinned)
        config = json.loads((snapshot / sources.UMT5_CONFIG).read_text(encoding="utf-8"))
        assert (config["num_layers"], config["d_model"], config["vocab_size"]) == (
            24,
            4096,
            256_384,
        )
