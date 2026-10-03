"""`wan/text_embeds.py` の約束事 — 固定プロンプト・正規化・資産の形式と再現性。

umT5 を回すテストは置かない（bf16 で約 11 GiB・1 本 2.5 分）。生成済みの資産は、系列の席に
あれば形・有効長・正規化文字列を見る（無ければ SKIP）。生成は
`uv run --group wan --inexact python -m wan.text_embeds`。
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from _shared.paths import SERIES_ROOT
from wan import prompts, text_embeds
from wan.pipeline_ref import MAX_SEQUENCE_LENGTH, TEXT_DIM
from wan.sources import DEFAULT_MODEL, SOURCES

ASSET = SERIES_ROOT / text_embeds.SERIES_NAME / text_embeds.ASSET_NAME
RECIPES_ROOT = Path(__file__).resolve().parents[2]


def _fake_asset(rows: dict[str, int]) -> tuple[dict[str, torch.Tensor], dict]:
    """固定プロンプトの表どおりの名前で、行数だけ与えた乱数の埋め込みとメタ。"""
    generator = torch.Generator().manual_seed(7)
    embeds = {
        name: torch.randn(length, TEXT_DIM, generator=generator) for name, length in rows.items()
    }
    metadata = text_embeds.asset_metadata(
        text_embeds.FIXED_PROMPTS,
        rows,
        {prompt.name: prompt.text.strip() for prompt in text_embeds.FIXED_PROMPTS},
        model=DEFAULT_MODEL,
        versions={"torch": "test"},
    )
    return embeds, metadata


FAKE_ROWS = {"boxing-cats": 3, "ferret": 5, "cat-dog-baking": 4, "negative": 6}


class TestFixedPrompts:
    def test_three_positives_and_one_negative(self):
        roles = [prompt.role for prompt in text_embeds.FIXED_PROMPTS]
        names = [prompt.name for prompt in text_embeds.FIXED_PROMPTS]

        assert roles.count(prompts.POSITIVE) == 3
        assert roles.count(prompts.NEGATIVE) == 1
        assert len(set(names)) == len(names)

    def test_every_source_url_pins_a_commit(self):
        """ブランチ名の URL は先頭が動くと別の文面を指しうる。"""
        for prompt in text_embeds.FIXED_PROMPTS:
            assert re.search(r"/blob/[0-9a-f]{40}/", prompt.url), prompt.url

    def test_an_unknown_name_fails_loudly(self):
        with pytest.raises(text_embeds.TextEmbedsError):
            text_embeds.prompt_by_name("no-such-prompt")

    def test_the_pipeline_example_is_the_one_in_the_pinned_diffusers(self):
        """`cat-dog-baking` は pin した diffusers の `WanPipeline.__call__` の例文そのもの。"""
        pytest.importorskip("diffusers")
        from diffusers.pipelines.wan.pipeline_wan import EXAMPLE_DOC_STRING

        text = text_embeds.prompt_by_name("cat-dog-baking").text

        assert f'prompt = "{text}"' in EXAMPLE_DOC_STRING


class TestNormalize:
    @pytest.fixture(autouse=True)
    def _needs_the_wan_group(self):
        pytest.importorskip("ftfy")
        pytest.importorskip("diffusers")

    def test_the_multiline_docs_prompt_becomes_one_line(self):
        normalized = text_embeds.normalize(text_embeds.prompt_by_name("ferret").text)

        assert "\n" not in normalized
        assert normalized == normalized.strip()
        assert normalized.startswith("The camera rushes")
        assert "  " not in normalized

    def test_ftfy_folds_the_fullwidth_commas_of_the_negative(self):
        """ftfy の有無で変わるのはこの形 — 決定 4 が ftfy を固定する理由そのもの。"""
        text = text_embeds.prompt_by_name("negative").text

        normalized = text_embeds.normalize(text)

        assert "，" not in normalized
        assert normalized.count(",") == text.count("，") > 0
        assert normalized.replace(",", "，") == text

    def test_an_already_clean_prompt_is_unchanged(self):
        text = text_embeds.prompt_by_name("boxing-cats").text

        assert text_embeds.normalize(text) == text

    def test_a_missing_ftfy_refuses_to_normalize(self, monkeypatch: pytest.MonkeyPatch):
        import diffusers.utils

        monkeypatch.setattr(diffusers.utils, "is_ftfy_available", lambda: False)

        with pytest.raises(text_embeds.TextEmbedsError):
            text_embeds.normalize("a prompt")


class TestAssetFormat:
    def test_it_round_trips_tensors_and_metadata(self, tmp_path: Path):
        embeds, metadata = _fake_asset(FAKE_ROWS)
        path = tmp_path / text_embeds.ASSET_NAME

        text_embeds.write_asset(path, embeds, metadata)
        tensors, meta = text_embeds.read_asset(path)

        assert meta == metadata
        assert sorted(tensors) == sorted(embeds)
        for name, tensor in embeds.items():
            assert torch.equal(tensors[name], tensor)

    def test_two_processes_write_the_same_bytes(self, tmp_path: Path):
        """メタを複数キーにすると、プロセスごとに safetensors のヘッダのキー順が変わって割れる。"""
        script = (
            "import sys, torch\n"
            "from pathlib import Path\n"
            "from wan import text_embeds\n"
            "from wan.tests.test_text_embeds import FAKE_ROWS, _fake_asset\n"
            "embeds, metadata = _fake_asset(FAKE_ROWS)\n"
            "text_embeds.write_asset(Path(sys.argv[1]), embeds, metadata)\n"
        )
        digests = []
        for index in range(2):
            path = tmp_path / f"run{index}.safetensors"
            subprocess.run([sys.executable, "-c", script, str(path)], cwd=RECIPES_ROOT, check=True)
            digests.append(hashlib.sha256(path.read_bytes()).hexdigest())

        assert digests[0] == digests[1]

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda embeds: embeds.__setitem__("ferret", embeds["ferret"].half()),
            lambda embeds: embeds.__setitem__("ferret", embeds["ferret"][:4]),
            lambda embeds: embeds.__setitem__("ferret", embeds["ferret"][:, :4095]),
            lambda embeds: embeds.pop("negative"),
            lambda embeds: embeds.__setitem__("extra", torch.zeros(2, TEXT_DIM)),
        ],
        ids=["f16", "rows-differ-from-tokens", "width", "missing", "extra"],
    )
    def test_a_tensor_outside_the_contract_fails_loudly(self, tmp_path: Path, mutate):
        embeds, metadata = _fake_asset(FAKE_ROWS)
        mutate(embeds)

        with pytest.raises(text_embeds.TextEmbedsError):
            text_embeds.write_asset(tmp_path / text_embeds.ASSET_NAME, embeds, metadata)
        assert not (tmp_path / text_embeds.ASSET_NAME).exists()

    def test_a_file_with_other_metadata_keys_is_rejected(self, tmp_path: Path):
        from safetensors.torch import save_file

        path = tmp_path / "other.safetensors"
        save_file({"a": torch.zeros(1, TEXT_DIM)}, str(path), metadata={"prompt": "a"})

        with pytest.raises(text_embeds.TextEmbedsError):
            text_embeds.read_asset(path)


@pytest.fixture(scope="module")
def generated_asset():
    if not ASSET.is_file():
        pytest.skip(f"{ASSET} が無い — `python -m wan.text_embeds` で作る")
    return text_embeds.read_asset(ASSET)


class TestGeneratedAsset:
    def test_it_holds_every_fixed_prompt_as_valid_rows_in_f32(self, generated_asset):
        tensors, meta = generated_asset
        entries = {entry["name"]: entry for entry in meta["prompts"]}

        assert sorted(tensors) == sorted(prompt.name for prompt in text_embeds.FIXED_PROMPTS)
        for prompt in text_embeds.FIXED_PROMPTS:
            tensor = tensors[prompt.name]
            entry = entries[prompt.name]
            assert tensor.dtype == torch.float32
            assert tensor.shape == (entry["tokens"], TEXT_DIM)
            assert 0 < entry["tokens"] < MAX_SEQUENCE_LENGTH
            assert bool(tensor.isfinite().all())
            # 有効長の最後の行（EOS）もゼロでない — 切り詰めの取り違えを掴む。
            assert torch.count_nonzero(tensor[-1]) > 0
            assert entry["prompt"] == prompt.text
            assert entry["role"] == prompt.role

    def test_the_encoder_and_the_source_are_recorded(self, generated_asset):
        _, meta = generated_asset

        assert meta["text_encoder"]["dtype"] == "bfloat16"
        assert meta["source"]["revision"] == SOURCES[DEFAULT_MODEL].revision
        assert meta["versions"]["ftfy"] == "6.3.1"
        assert meta["versions"]["diffusers"] == "0.39.0"

    def test_bf16_outputs_survive_the_f32_storage_exactly(self, generated_asset):
        """umT5 の出力は bf16 — f32 へ広げた値は下位 16 ビットが全部ゼロ（丸めを足していない）。"""
        tensors, _ = generated_asset

        for tensor in tensors.values():
            assert torch.equal(tensor, tensor.to(torch.bfloat16).to(torch.float32))

    def test_the_normalized_text_is_upstream_prompt_clean(self, generated_asset):
        pytest.importorskip("ftfy")
        pytest.importorskip("diffusers")
        _, meta = generated_asset

        for entry in meta["prompts"]:
            assert entry["normalized"] == text_embeds.normalize(entry["prompt"])

    def test_the_valid_length_is_the_tokenizer_mask_length(self, generated_asset):
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer

        from wan.sources import WanSourceError, text_snapshot

        try:
            snapshot = text_snapshot()
        except WanSourceError as error:
            pytest.skip(f"umT5 のトークナイザが手元に無い: {error}")
        tokenizer = AutoTokenizer.from_pretrained(snapshot, subfolder="tokenizer")
        _, meta = generated_asset

        for entry in meta["prompts"]:
            mask = tokenizer(
                [entry["normalized"]],
                padding="max_length",
                max_length=MAX_SEQUENCE_LENGTH,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).attention_mask
            assert int(mask.sum()) == entry["tokens"]
