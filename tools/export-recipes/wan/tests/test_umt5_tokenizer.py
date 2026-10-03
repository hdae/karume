"""`wan/umt5_tokenizer.py` の約束事 — 資産の焼き方・2 経路の一致の門・フィクスチャの生成器。

上流のトークナイザを読むテストは pin した revision の `tokenizer/` が HF キャッシュに無ければ SKIP
（取得は `uv run --group wan --inexact python -m wan.text_embeds --fetch`）。期待の決め方そのもの
（2 経路の割れ方の判定・拒否の順・語彙の部分集合）は合成の正本で叩く。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from wan import prompt_clean as pc
from wan import umt5_tokenizer as ut


def fake_truth(
    transformers_ids: dict[str, list[int]],
    tokenizer_json_ids: dict[str, list[int]] | None = None,
    vocab: list[list[Any]] | None = None,
) -> ut.Truth:
    """文字列 → id 列の表で答える合成の正本（2 経路を別々に与えられる）。"""
    plain = tokenizer_json_ids if tokenizer_json_ids is not None else transformers_ids
    return ut.Truth(
        transformers=lambda text: transformers_ids[text],
        masked_length=lambda text: len(transformers_ids[text]),
        tokenizer_json=lambda text: plain[text],
        backend={
            "model": {"vocab": vocab or []},
            "added_tokens": [{"content": "</s>", "id": 1}, {"content": "<s>", "id": 2}],
        },
        raw={},
        pre_tokenize=lambda text: ["▁" + word for word in text.split()],
    )


ADDED = ["</s>", "<s>"]


class TestExpectation:
    def test_agreeing_paths_are_accepted_with_both_id_lists(self):
        truth = fake_truth({"a cat": [4, 5, 1]})

        assert ut.expectation(truth, ADDED, "a cat") == {
            "ids": [4, 5, 1],
            "tokenizerJsonIds": [4, 5, 1],
        }

    def test_a_2_versus_3_split_is_out_of_vocabulary(self):
        truth = fake_truth({"a 𠀋": [4, 273, 2, 1]}, {"a 𠀋": [4, 273, 3, 1]})

        assert ut.expectation(truth, ADDED, "a 𠀋") == {"reject": "oov"}

    @pytest.mark.parametrize(
        ("transformers", "plain"),
        [([4, 5, 1], [4, 6, 1]), ([4, 5, 1], [4, 273, 5, 1]), ([4, 3, 1], [4, 2, 1])],
    )
    def test_any_other_split_fails_loudly(self, transformers, plain):
        """2 と 3 以外の割れ方は経路の構成の違い — 期待を焼かずに落とす。"""
        truth = fake_truth({"x": transformers}, {"x": plain})

        with pytest.raises(ValueError, match="割れ方"):
            ut.expectation(truth, ADDED, "x")

    def test_added_tokens_and_space_metaspace_are_rejected_before_tokenizing(self):
        """本文中の追加語彙と空白の直後の ▁ は 2 経路で割れる — トークナイザに渡す前に拒む。"""
        truth = fake_truth({})

        assert ut.expectation(truth, ADDED, "a </s> b") == {"reject": "added-token"}
        assert ut.expectation(truth, ADDED, "a ▁b") == {"reject": "metaspace"}

    def test_lengths_outside_two_to_512_are_rejected(self):
        long_ids = [4] * ut.MAX_LENGTH + [1]
        truth = fake_truth({"long": long_ids, "": [1]})

        assert ut.expectation(truth, ADDED, "long") == {"reject": "too-long"}
        assert ut.expectation(truth, ADDED, "") == {"reject": "too-short"}

    def test_out_of_vocabulary_and_too_long_together_are_not_baked(self):
        """どちらで落ちるかが位置で決まるケースはフィクスチャに置かない（TS と順が割れうる）。"""
        transformers = [2] + [4] * ut.MAX_LENGTH + [1]
        plain = [3] + [4] * ut.MAX_LENGTH + [1]
        truth = fake_truth({"x": transformers}, {"x": plain})

        with pytest.raises(ValueError, match="重なる"):
            ut.expectation(truth, ADDED, "x")


class TestPieces:
    def test_added_tokens_are_split_leftmost_longest(self):
        assert ut.split_added("a<s></s>b", ["</s>", "<s>", "</s>b"]) == [
            ("a", False),
            ("<s>", True),
            ("</s>b", True),
        ]

    def test_the_vocab_subset_keeps_every_substring_of_every_piece(self):
        vocab = [["▁", 0.0], ["▁c", -1.0], ["c", -1.0], ["a", -1.0], ["at", -2.0], ["dog", -3.0]]
        truth = fake_truth({}, vocab=vocab)
        cases = [{"cleaned": "cat"}, {"reject": "c1"}]

        subset = ut.vocab_subset(truth, cases)

        assert [token for token, _, _ in subset] == ["▁", "▁c", "c", "a", "at"]
        assert subset[1] == ["▁c", 1, -1.0]


class TestFixtureJson:
    def test_short_arrays_stay_on_one_line_and_objects_expand(self):
        text = ut.fixture_json({"ranges": [[1, 2], [3, 4]], "cases": [{"id": "a"}]})

        assert '"ranges": [[1, 2], [3, 4]]' in text
        assert '"cases": [\n    {\n      "id": "a"\n    }\n  ]' in text
        assert json.loads(text) == {"ranges": [[1, 2], [3, 4]], "cases": [{"id": "a"}]}

    def test_long_arrays_expand_one_element_per_line(self):
        text = ut.fixture_json({"ids": list(range(60))})

        assert text.count("\n") > 60


class TestSweepFixtureGenerator:
    def test_the_generator_is_deterministic_and_records_every_run(self):
        pytest.importorskip("ftfy")
        pytest.importorskip("diffusers")
        tables = pc.build_tables()
        truth = pc.upstream_prompt_clean()
        runs = (pc.FuzzRun("safe", "safe", seed=1, count=300, chunk=100),)
        options: dict[str, Any] = {
            "clean_runs": runs,
            "badness_runs": (pc.FuzzRun("full", "full", seed=4, count=300),),
            "badness_probes": (),
            "clean_sweeps": (),
            "nfc_sweeps": (),
        }

        first = ut.build_sweep_fixture(tables, truth, {"ftfy": "x"}, **options)
        second = ut.build_sweep_fixture(tables, truth, {"ftfy": "x"}, **options)

        assert first == second
        assert len(first["clean"][0]["chunks"]) == 3
        assert first["clean"][0]["outcomes"] == {"accepted": 300}
        assert first["prng"]["first"][0] == 270369


@pytest.fixture(scope="module")
def upstream():
    """pin した revision の上流トークナイザ（無ければ SKIP）と前処理の表。"""
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    pytest.importorskip("ftfy")
    pytest.importorskip("diffusers")
    from wan.sources import WanSourceError

    try:
        snapshot = ut.tokenizer_snapshot()
    except WanSourceError as error:
        pytest.skip(f"umT5 のトークナイザが手元に無い: {error}")
    tables = pc.build_tables()
    return ut.load_truth(snapshot), tables, {"test": "versions"}


class TestAgainstUpstream:
    def test_max_length_is_the_pipeline_max_sequence_length(self):
        from wan.pipeline_ref import MAX_SEQUENCE_LENGTH

        assert ut.MAX_LENGTH == MAX_SEQUENCE_LENGTH

    def test_the_backend_has_the_shape_the_typescript_copies(self, upstream):
        truth, _, _ = upstream

        ut.check_upstream_shape(truth)
        assert truth.transformers("a </s> b") == [289, 1, 748, 1]
        assert truth.tokenizer_json("a </s> b") == [289, 273, 1, 748, 1]

    def test_baking_the_asset_twice_gives_the_same_bytes(self, upstream):
        truth, tables, versions = upstream

        first = json.dumps(ut.build_asset(truth, tables, versions, "t2v-1.3b"), ensure_ascii=False)
        second = json.dumps(ut.build_asset(truth, tables, versions, "t2v-1.3b"), ensure_ascii=False)

        assert first == second
        asset = json.loads(first)
        assert asset["format"] == ut.ASSET_FORMAT
        assert len(asset["scores"]) == 256_300
        assert len(asset["addedTokens"]) == 304

    def test_the_committed_parity_fixture_is_what_the_generator_makes(self, upstream):
        """commit 済みのフィクスチャを上流から作り直して一致を見る。

        上流の動き（版・資産）と生成器の非決定性の両方をここで拾う。
        """
        truth, tables, _ = upstream
        committed = json.loads((ut.FIXTURE_DIR / ut.PARITY_FIXTURE).read_text(encoding="utf-8"))
        asset = ut.build_asset(truth, tables, committed["versions"], "t2v-1.3b")
        cases = ut.build_cases(truth, tables, pc.upstream_prompt_clean())

        regenerated = ut.build_parity_fixture(truth, asset, cases)

        assert regenerated == committed

    def test_a_case_whose_paths_split_on_added_tokens_is_rejected(self, upstream):
        truth, tables, _ = upstream
        case = ut.build_case(
            truth, pc.CompiledTables(tables), pc.upstream_prompt_clean(), "x", "a </s> b", "-"
        )

        assert case["reject"] == "added-token"
