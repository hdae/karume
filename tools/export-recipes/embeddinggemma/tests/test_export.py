"""`embeddinggemma/export.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（既存 `deberta/export.py` のテストと同じ規律）。ここで固定するのは、
壊れると**偽 PASS** になる側の規律だけ:

- `--batch` の既定が 1（従来の 5 ケース golden そのまま）で、1 未満は fail loudly
- `_write_io` が embeddings を `[B, H]` の形のまま返すこと（`_sanity` / `_sanity_batch` の
  両方が同じ形から読める — 片方のためだけに reshape を作り込まない）
- `_sanity_batch` が単位ノルムと行間一致の**両方**を見ること（片方だけでは恒真になる）
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from embeddinggemma import export as eg
from karume.pipeline import export_to_file


class TinyEmbedding(nn.Module):
    """`EmbeddingWrapper` の最小の骨格（masked-mean → linear → L2 正規化、`[B,T]→[B,H]`）。"""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(1, 4, bias=False)

    def forward(self, input_ids: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(input_ids.to(torch.float32).unsqueeze(-1))
        total = torch.sum(hidden * pool_mask.unsqueeze(-1), dim=1)
        count = torch.sum(pool_mask, dim=1).unsqueeze(-1)
        pooled = total / torch.clamp(count, min=1e-9)
        norm = torch.sqrt(torch.sum(pooled * pooled, dim=-1)).clamp(min=1e-12).unsqueeze(-1)
        return pooled / norm


CASE_B1 = (
    "case0",
    torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
    torch.ones(1, 4, dtype=torch.float32),
)
CASE_B3 = (
    "batch3",
    torch.tensor([[1, 2, 3, 4]], dtype=torch.int64).expand(3, -1).contiguous(),
    torch.ones(3, 4, dtype=torch.float32),
)


@pytest.fixture
def exported_batch3(tmp_path):
    """`query-en` 相当の 1 行を 3 行に複製したケースを export して `(wrapper, graph, out_dir)`。"""
    torch.manual_seed(0)
    wrapper = TinyEmbedding()
    graph = export_to_file(wrapper, CASE_B3[1:], tmp_path / eg.MODEL_FILE)
    return wrapper, graph, tmp_path


class TestBatchCli:
    def test_batch_defaults_to_one(self, monkeypatch):
        seen: dict[str, object] = {}
        monkeypatch.setattr(eg, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})
        eg.main([])

        assert seen["batch"] == 1

    def test_batch_flag_is_forwarded(self, monkeypatch):
        seen: dict[str, object] = {}
        monkeypatch.setattr(eg, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})
        eg.main(["--batch", "32"])

        assert seen["batch"] == 32

    def test_batch_below_one_fails_loudly(self):
        with pytest.raises(SystemExit, match="--batch は 1 以上"):
            eg.main(["--batch", "0"])


class TestWriteIoPreservesTheBatchDimension:
    """MUST: `_write_io` の embeddings は `[B, H]` のまま — `_sanity_batch` が行を見比べる形。"""

    def test_embeddings_keep_the_batch_dimension(self, exported_batch3):
        wrapper, graph, out_dir = exported_batch3

        written, embeddings = eg._write_io(wrapper, graph, (CASE_B3,), out_dir)

        assert written == [f"{eg.IO_PREFIX}batch3{eg.IO_SUFFIX}"]
        assert tuple(embeddings["batch3"].shape) == (3, 4)

    def test_batch_one_still_reduces_to_a_flat_vector_via_sanity(self, tmp_path):
        """batch=1 の従来経路は `_sanity` 側の reshape(-1) で吸収される（挙動不変の確認）。"""
        torch.manual_seed(0)
        wrapper = TinyEmbedding()
        graph = export_to_file(wrapper, CASE_B1[1:], tmp_path / eg.MODEL_FILE)

        _, embeddings = eg._write_io(wrapper, graph, (CASE_B1,), tmp_path)

        assert tuple(embeddings["case0"].shape) == (1, 4)
        assert embeddings["case0"].reshape(-1).shape == (4,)


class TestSanityBatch:
    def test_passes_when_rows_are_identical_unit_vectors(self):
        row = torch.tensor([0.6, 0.8])
        output = row.unsqueeze(0).expand(5, -1).contiguous()

        result = eg._sanity_batch(output)

        assert result["rows"] == 5
        assert result["row_max_abs_diff"] == 0.0

    def test_fails_loudly_when_a_row_diverges(self):
        """MUST: 行間一致を見る — 複製元の入力が同一なら全行が一致するはず。"""
        output = torch.tensor([[0.6, 0.8], [0.6, 0.8], [0.0, 1.0]])

        with pytest.raises(AssertionError, match="行間の出力が一致しない"):
            eg._sanity_batch(output)

    def test_fails_loudly_when_a_row_norm_is_off(self):
        """MUST: ノルムだけでも見る — 行間一致だけでは全行が同じだけズレていても通ってしまう。"""
        output = torch.tensor([[0.6, 0.8], [0.5, 0.5]])

        with pytest.raises(AssertionError, match="L2 ノルムが 1 から外れた"):
            eg._sanity_batch(output)


class TestSanityAcceptsTheGeneralizedShape:
    def test_unit_norm_and_cosine_still_work_when_embeddings_are_shape_1xh(self):
        """`_write_io` が reshape(-1) をやめた後も従来 4 ケース分で `_sanity` が動くことの確認。"""
        vectors = {
            "query-en": torch.tensor([[1.0, 0.0]]),
            "document-en": (torch.tensor([[0.9, 0.1]]) / torch.tensor([[0.9, 0.1]]).norm()),
            "bare": torch.tensor([[0.0, 1.0]]),
            "query-ja": torch.tensor([[1.0, 0.0]]),
        }

        result = eg._sanity(vectors)

        assert set(result["l2_norms"]) == set(vectors)
