"""`irodori/dacvae/convert.py` の台本レベルの約束事（実重み不要分）。

実重みの変換は手動（一回性ユーティリティ）だが、**形の門と据え替えの規律**は合成
チェックポイント（`torch.save` で作る `.pth`）だけで回せる。ここで固定するのは、壊れると
入力素材が黙って別物になる側だけ:

- 外側ラッパの形（鍵集合・`state_dict` が dict・値が全てテンソル）が外れたら落ちる
- metadata の JSON 往復が同値にならない型（tuple / 非文字列鍵）が混ざったら落ちる
- 書いた safetensors を別実装のリーダで読み直したバイト一致検査が、dtype / shape /
  1 要素の食い違いをそれぞれ落とす
- `convert()` が `metadata.json` と `__metadata__` を**同じ源から**書く
- 門に落ちた実行が正規 path を 1 バイトも動かさない（ADR 0052 の据え替え）
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from irodori.dacvae import convert as cv

#: 合成チェックポイントの中身（鍵名は実物の綴りに寄せた 2 本）。
TENSORS = {
    "encoder.block.0.weight": torch.tensor([[1.0, -2.0], [0.5, 0.25]]),
    "quantizer.in_proj.bias": torch.tensor([0.125, -0.5, 2.0, 4.0]),
}

#: 実物と同じ形の metadata（テンソルを含まない純粋な構成値）。
METADATA = {"kwargs": {"latent_dim": 32, "encoder_rates": [2, 4, 5, 8], "name": "dacvae"}}


def _write_ckpt(tmp_path, *, state_dict=None, metadata=None, wrapper=None):
    """合成 `.pth` を書いて path を返す（`wrapper` を渡すと外側の形ごと差し替える）。"""
    path = tmp_path / "weights.pth"
    payload = (
        wrapper
        if wrapper is not None
        else {
            "state_dict": TENSORS if state_dict is None else state_dict,
            "metadata": METADATA if metadata is None else metadata,
        }
    )
    torch.save(payload, path)
    return path


class TestLoadCheckpoint:
    """MUST: 外側の形の増減は配布側の変更なので fail loudly（黙って別の鍵を読まない）。"""

    def test_the_expected_shape_returns_the_state_dict_and_metadata(self, tmp_path):
        state_dict, metadata = cv._load_checkpoint(_write_ckpt(tmp_path))

        assert sorted(state_dict) == sorted(TENSORS)
        assert metadata == METADATA

    def test_a_top_level_that_is_not_a_dict_fails_loudly(self, tmp_path):
        with pytest.raises(cv.ConvertError, match="最上位が dict でない"):
            cv._load_checkpoint(_write_ckpt(tmp_path, wrapper=[1, 2, 3]))

    def test_a_wrapper_with_other_keys_fails_loudly(self, tmp_path):
        wrapper = {"state_dict": TENSORS, "metadata": METADATA, "optimizer": {}}

        with pytest.raises(cv.ConvertError, match="外側ラッパ"):
            cv._load_checkpoint(_write_ckpt(tmp_path, wrapper=wrapper))

    def test_a_state_dict_that_is_not_a_dict_fails_loudly(self, tmp_path):
        with pytest.raises(cv.ConvertError, match="state_dict が dict でない"):
            cv._load_checkpoint(_write_ckpt(tmp_path, state_dict=[1, 2]))

    def test_a_state_dict_holding_a_non_tensor_fails_loudly(self, tmp_path):
        state_dict = {**TENSORS, "encoder.steps": 4}

        with pytest.raises(cv.ConvertError, match="テンソルでない"):
            cv._load_checkpoint(_write_ckpt(tmp_path, state_dict=state_dict))


class TestMetadataJson:
    def test_a_plain_config_round_trips(self):
        text = cv._metadata_json(METADATA)

        assert json.loads(text) == METADATA

    def test_a_tuple_does_not_round_trip_and_fails_loudly(self):
        """JSON は tuple を list にする — 往復が同値でないまま書くと素性を偽ることになる。"""
        with pytest.raises(cv.ConvertError, match="往復"):
            cv._metadata_json({"a": (1, 2)})

    def test_a_non_string_key_does_not_round_trip_and_fails_loudly(self):
        with pytest.raises(cv.ConvertError, match="往復"):
            cv._metadata_json({1: "one"})

    def test_a_value_json_cannot_hold_fails_loudly(self):
        with pytest.raises(cv.ConvertError, match="JSON にできない"):
            cv._metadata_json({"a": object()})


class TestByteIdentical:
    """MUST: 書いた実装とは**別実装のリーダ**で読み直す（writer の自己申告を信じない）。"""

    def test_an_untouched_file_matches_every_tensor(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        save_file(TENSORS, str(path))

        assert cv._assert_byte_identical(path, TENSORS) == len(TENSORS)

    def test_a_missing_key_fails_loudly(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        save_file({"quantizer.in_proj.bias": TENSORS["quantizer.in_proj.bias"]}, str(path))

        with pytest.raises(cv.ConvertError, match="鍵集合が一致しない"):
            cv._assert_byte_identical(path, TENSORS)

    def test_a_different_dtype_fails_loudly(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        written = {**TENSORS, "quantizer.in_proj.bias": TENSORS["quantizer.in_proj.bias"].half()}
        save_file(written, str(path))

        with pytest.raises(cv.ConvertError, match="dtype 不一致"):
            cv._assert_byte_identical(path, TENSORS)

    def test_a_different_shape_fails_loudly(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        written = {**TENSORS, "quantizer.in_proj.bias": TENSORS["quantizer.in_proj.bias"][:2]}
        save_file(written, str(path))

        with pytest.raises(cv.ConvertError, match="shape 不一致"):
            cv._assert_byte_identical(path, TENSORS)

    def test_a_single_changed_element_fails_loudly(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        moved = TENSORS["quantizer.in_proj.bias"].clone()
        moved[0] = -0.0  # 値としては 0.0 と等しい — ビット列でしか捕まらない食い違い。
        save_file({**TENSORS, "quantizer.in_proj.bias": moved}, str(path))

        with pytest.raises(cv.ConvertError, match="バイト列が一致しない"):
            cv._assert_byte_identical(path, TENSORS)


class TestConvert:
    def test_it_writes_both_records_from_the_same_source(self, tmp_path):
        """MUST: `metadata.json` と `__metadata__` は 1 パスで同じ源から書かれる。"""
        ckpt = _write_ckpt(tmp_path)

        summary = cv.convert(ckpt)

        target = tmp_path / "weights.safetensors"
        assert summary["tensors"] == len(TENSORS)
        assert summary["byte_identical"] == len(TENSORS)
        assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8")) == METADATA
        with safe_open(str(target), framework="pt") as handle:
            header = handle.metadata()
        assert header[cv.SOURCE_FILE_KEY] == ckpt.name
        assert header[cv.SOURCE_SHA256_KEY] == cv._sha256(ckpt) == summary["source_sha256"]
        assert json.loads(header[cv.SOURCE_METADATA_KEY]) == METADATA

    def test_a_run_that_fails_a_gate_leaves_the_previous_output_in_place(
        self, tmp_path, monkeypatch
    ):
        """MUST（ADR 0052）: 門を通ってから据える — 落ちた実走は正規 path を動かさない。"""
        ckpt = _write_ckpt(tmp_path)
        cv.convert(ckpt)
        target = tmp_path / "weights.safetensors"
        metadata_path = tmp_path / "metadata.json"
        before = (target.read_bytes(), metadata_path.read_bytes())

        def refuse(path):
            raise cv.ConvertError("テスト: リーダ規則の門を落とす")

        monkeypatch.setattr(cv, "assert_reader_layout", refuse)
        torch.save({"state_dict": {"other.weight": torch.ones(3)}, "metadata": {}}, ckpt)

        with pytest.raises(cv.ConvertError, match="リーダ規則の門"):
            cv.convert(ckpt)

        assert (target.read_bytes(), metadata_path.read_bytes()) == before
