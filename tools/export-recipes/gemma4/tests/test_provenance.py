"""出所記録（`gemma4/provenance.py`）の約束事。

golden を採らない系列の検収は「流用する golden がどれか」に全面的に乗るので、ここで固定
するのは**束縛が緩む側**の規律だけ:

- 流用先が欠けている / 別のトークナイザで採られている組み合わせを **fail loudly** にする
- 記録に載る digest が実ファイルの sha256 と byte 数そのものであること（恒真でない —
  1 バイト書き換えれば digest が動くことを故障注入で見る）

transformers も torch.export も要らない（safetensors の読み書きだけ）。
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from safetensors.torch import save_file

from _shared.decode_series import GREEDY_PREFIX, GREEDY_SUFFIX, PROMPT_KEY
from gemma4 import provenance

CASES = (
    ("capital-en", torch.tensor([[2, 41, 7, 19]], dtype=torch.int64)),
    ("capital-ja", torch.tensor([[2, 5, 88]], dtype=torch.int64)),
)


def _write_goldens(series_dir, cases) -> None:
    series_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in cases:
        save_file(
            {PROMPT_KEY: ids[0].to(torch.int32).contiguous()},
            str(series_dir / f"{GREEDY_PREFIX}{name}{GREEDY_SUFFIX}"),
        )


class TestAssertReferenceGoldens:
    def test_matching_goldens_yield_their_digests(self, tmp_path):
        series = tmp_path / "gemma4-e2b-decode"
        _write_goldens(series, CASES)

        reference = provenance.assert_reference_goldens(series, CASES)

        assert reference["series"] == "gemma4-e2b-decode"
        assert sorted(reference["goldens"]) == [
            "greedy.capital-en.safetensors",
            "greedy.capital-ja.safetensors",
        ]
        for file, digest in reference["goldens"].items():
            raw = (series / file).read_bytes()
            assert digest == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def test_a_regenerated_golden_moves_its_digest(self, tmp_path):
        """故障注入: 同じ prompt でも中身が変われば digest が動く（門が緑のままにならない）。

        期待列（`expected`）を採り直した golden は prompt が同じままなので、prompt の突合
        だけでは素通りする — digest がその差を拾う側。
        """
        series = tmp_path / "gemma4-e2b-decode"
        _write_goldens(series, CASES)
        before = provenance.assert_reference_goldens(series, CASES)

        name, ids = CASES[1]
        save_file(
            {
                PROMPT_KEY: ids[0].to(torch.int32).contiguous(),
                "expected": torch.tensor([3, 4], dtype=torch.int32),
            },
            str(series / f"{GREEDY_PREFIX}{name}{GREEDY_SUFFIX}"),
        )
        after = provenance.assert_reference_goldens(series, CASES)

        assert (
            before["goldens"]["greedy.capital-en.safetensors"]
            == after["goldens"]["greedy.capital-en.safetensors"]
        )
        assert (
            before["goldens"]["greedy.capital-ja.safetensors"]
            != after["goldens"]["greedy.capital-ja.safetensors"]
        )

    def test_a_missing_series_fails_loudly(self, tmp_path):
        with pytest.raises(AssertionError, match="参照 golden 系列"):
            provenance.assert_reference_goldens(tmp_path / "absent", CASES)

    def test_a_missing_case_fails_loudly(self, tmp_path):
        series = tmp_path / "gemma4-e2b-decode"
        _write_goldens(series, CASES[:1])

        with pytest.raises(AssertionError, match="capital-ja"):
            provenance.assert_reference_goldens(series, CASES)

    def test_a_prompt_from_another_tokenizer_fails_loudly(self, tmp_path):
        """MUST: 別のケース定義で採られた golden を流用しようとしたら落とす。"""
        series = tmp_path / "gemma4-e2b-decode"
        _write_goldens(series, CASES)
        drifted = ((CASES[0][0], torch.tensor([[2, 41, 7, 20]], dtype=torch.int64)), CASES[1])

        with pytest.raises(AssertionError, match="prompt が今回のケース"):
            provenance.assert_reference_goldens(series, drifted)


class TestCheckpointFingerprint:
    def test_every_declared_file_is_fingerprinted(self, tmp_path):
        checkpoint = tmp_path / "gemma-4-E2B-it"
        checkpoint.mkdir()
        for name in provenance.FINGERPRINT_FILES:
            (checkpoint / name).write_bytes(name.encode())

        fingerprint = provenance.checkpoint_fingerprint(checkpoint)

        assert fingerprint["dir"] == "gemma-4-E2B-it"
        assert sorted(fingerprint["files"]) == sorted(provenance.FINGERPRINT_FILES)
        for name, digest in fingerprint["files"].items():
            assert digest["sha256"] == hashlib.sha256(name.encode()).hexdigest()

    def test_a_missing_file_fails_loudly(self, tmp_path):
        checkpoint = tmp_path / "gemma-4-E2B-it"
        checkpoint.mkdir()

        with pytest.raises(AssertionError, match="指紋対象"):
            provenance.checkpoint_fingerprint(checkpoint)


class TestBuildRecord:
    def test_the_record_names_the_series_it_is_written_for(self, tmp_path):
        """記録が名乗るのは**据えた後**の系列名（作業席の名前ではない）。"""
        checkpoint = tmp_path / "gemma-4-E2B-it"
        checkpoint.mkdir()
        for name in provenance.FINGERPRINT_FILES:
            (checkpoint / name).write_bytes(name.encode())
        series = tmp_path / "gemma4-e2b-decode"
        _write_goldens(series, CASES)
        reference = provenance.assert_reference_goldens(series, CASES)

        record = provenance.build_record(
            tmp_path / "gemma4-e2b-decode-token", checkpoint, reference
        )

        assert record["schema"] == provenance.SCHEMA
        assert record["series"] == "gemma4-e2b-decode-token"
        assert record["reference"] == reference
