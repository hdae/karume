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
from gemma4.sources import HUB_PINS, HubPin, ModelPins

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


class TestAssertPinnedFingerprint:
    def _record(self, tmp_path):
        for name in provenance.FINGERPRINT_FILES:
            (tmp_path / name).write_bytes(name.encode())
        return provenance.checkpoint_fingerprint(tmp_path)

    def _pins(self, **overrides: str) -> ModelPins:
        def pin(repo: str, revision: str, names: tuple[str, ...]) -> HubPin:
            digests = {name: hashlib.sha256(name.encode()).hexdigest() for name in names}
            return HubPin(
                repo=repo,
                revision=revision,
                sha256={**digests, **{k: v for k, v in overrides.items() if k in names}},
            )

        return ModelPins(
            weights=pin("test/weights", "0" * 40, provenance.WEIGHTS_FILES),
            tokenizer=pin("test/tokenizer", "1" * 40, provenance.TOKENIZER_FILES),
        )

    def test_the_pinned_bytes_pass(self, tmp_path):
        provenance.assert_pinned_fingerprint(self._record(tmp_path), self._pins())

    @pytest.mark.parametrize(
        ("name", "source"),
        [
            ("config.json", "test/weights@" + "0" * 40),
            ("tokenizer.json", "test/tokenizer@" + "1" * 40),
        ],
    )
    def test_other_bytes_fail_loudly_naming_their_source(self, tmp_path, name, source):
        with pytest.raises(AssertionError, match="not the pinned one") as raised:
            provenance.assert_pinned_fingerprint(
                self._record(tmp_path), self._pins(**{name: "f" * 64})
            )
        assert source in str(raised.value)

    def test_a_pin_that_misses_a_fingerprinted_file_fails_loudly(self, tmp_path):
        pins = self._pins()
        partial = ModelPins(
            weights=HubPin(
                repo=pins.weights.repo,
                revision=pins.weights.revision,
                sha256={k: v for k, v in pins.weights.sha256.items() if k != "config.json"},
            ),
            tokenizer=pins.tokenizer,
        )

        with pytest.raises(AssertionError, match="the weights pin"):
            provenance.assert_pinned_fingerprint(self._record(tmp_path), partial)

    def test_a_file_pinned_in_the_other_source_s_role_fails_loudly(self, tmp_path):
        """`tokenizer.json` pinned with the weights would be checked against the wrong snapshot's
        bytes."""
        pins = self._pins()
        swapped = ModelPins(
            weights=HubPin(
                repo=pins.weights.repo,
                revision=pins.weights.revision,
                sha256={**pins.weights.sha256, **pins.tokenizer.sha256},
            ),
            tokenizer=HubPin(repo=pins.tokenizer.repo, revision=pins.tokenizer.revision, sha256={}),
        )

        with pytest.raises(AssertionError, match="the weights pin"):
            provenance.assert_pinned_fingerprint(self._record(tmp_path), swapped)

    def test_the_12b_pins_cover_the_fingerprinted_files_once(self):
        files = [name for pin in HUB_PINS["12b"].by_role().values() for name in pin.sha256]

        assert sorted(files) == sorted(provenance.FINGERPRINT_FILES)


class TestCheckpointLabel:
    """A hub snapshot's directory is named by the commit, so the record names the repository."""

    def test_the_label_names_the_checkpoint(self, tmp_path):
        snapshot = tmp_path / ("a" * 40)
        snapshot.mkdir()
        for name in provenance.FINGERPRINT_FILES:
            (snapshot / name).write_bytes(name.encode())

        record = provenance.build_record(
            tmp_path / "gemma4-12b-product",
            snapshot,
            {"series": "gemma4-12b-product", "goldens": {}},
            checkpoint_label="google/gemma-4-12B-it",
        )

        assert record["checkpoint"]["dir"] == "google/gemma-4-12B-it"

    def test_without_a_label_the_directory_names_it(self, tmp_path):
        for name in provenance.FINGERPRINT_FILES:
            (tmp_path / name).write_bytes(name.encode())

        assert provenance.checkpoint_fingerprint(tmp_path)["dir"] == tmp_path.name


class TestTwoSources:
    """A pinned model whose tokenizer comes from another repository than its weights (12B)."""

    def test_each_file_is_read_from_its_own_source_and_the_record_names_both(self, tmp_path):
        weights, tokenizer = tmp_path / "weights", tmp_path / "tokenizer"
        for directory, names in (
            (weights, provenance.WEIGHTS_FILES),
            (tokenizer, provenance.TOKENIZER_FILES),
        ):
            directory.mkdir()
            for name in names:
                (directory / name).write_bytes(f"{directory.name}/{name}".encode())
        pins = ModelPins(
            weights=HubPin(repo="test/weights", revision="0" * 40, sha256={}),
            tokenizer=HubPin(repo="test/tokenizer", revision="1" * 40, sha256={}),
        )

        record = provenance.build_record(
            tmp_path / "gemma4-12b-product",
            weights,
            {"series": "gemma4-12b-product", "goldens": {}},
            checkpoint_label="both",
            tokenizer_dir=tokenizer,
            pins=pins,
        )

        checkpoint = record["checkpoint"]
        assert checkpoint["files"]["tokenizer.json"]["sha256"] == (
            hashlib.sha256(b"tokenizer/tokenizer.json").hexdigest()
        )
        assert checkpoint["files"]["config.json"]["sha256"] == (
            hashlib.sha256(b"weights/config.json").hexdigest()
        )
        assert checkpoint["sources"] == {
            "weights": {
                "repo": "test/weights",
                "revision": "0" * 40,
                "files": ["model.safetensors", "config.json"],
            },
            "tokenizer": {
                "repo": "test/tokenizer",
                "revision": "1" * 40,
                "files": ["tokenizer.json"],
            },
        }

    def test_one_checkpoint_records_no_sources(self, tmp_path):
        for name in provenance.FINGERPRINT_FILES:
            (tmp_path / name).write_bytes(name.encode())

        assert "sources" not in provenance.checkpoint_fingerprint(tmp_path)
