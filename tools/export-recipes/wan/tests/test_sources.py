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
        fetched = {part.subfolder for part in sources.COMPONENTS if part.fetch}
        skipped = {part.subfolder for part in sources.COMPONENTS if not part.fetch}

        assert fetched == {"transformer", "vae", "scheduler"}
        assert skipped == {"text_encoder", "tokenizer"}
        assert not any(
            pattern.startswith(("text_encoder", "tokenizer"))
            for pattern in sources.allow_patterns()
        )

    def test_the_text_embedding_process_takes_exactly_the_skipped_parts(self):
        """`wan.text_embeds` の取得口（`text_snapshot`）が取るのは DiT / VAE 側が取らない
        部品だけ。"""
        assert set(sources.TEXT_COMPONENTS) == {"text_encoder", "tokenizer"}


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
        assert sources.check_snapshot(wan_snapshot) == sources.EXPECTED_PARAMETERS
