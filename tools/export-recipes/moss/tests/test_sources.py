"""`moss/sources.py` — the upstream pin, the hash check on the pipeline code, and the fetched parts.

The pin's form is checked without weights. The hash check is checked by fault injection on a copy
of the pinned files; the element counts of the fetched parts use {@link moss_snapshot} (SKIP when
not cached).
"""

from __future__ import annotations

import json
import re
import shutil
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from moss import sources


class TestPin:
    def test_every_revision_is_a_full_commit_sha(self):
        """A branch name or a short SHA reads other contents once the branch moves."""
        revisions = [sources.SOURCE.revision, *(o.source.revision for o in sources.VAE_ORIGINS)]

        for revision in revisions:
            assert re.fullmatch(r"[0-9a-f]{40}", revision), revision

    def test_every_pinned_digest_is_a_sha256(self):
        digests = [sources.PIPELINE_CODE_SHA256, *(o.sha256 for o in sources.VAE_ORIGINS)]

        for digest in digests:
            assert re.fullmatch(r"[0-9a-f]{64}", digest), digest

    def test_the_product_fetch_never_pulls_the_text_encoder(self):
        """The 16.8 GB text encoder comes only with `--text` (no shared RAM with the DiT)."""
        product = sources.allow_patterns(text=False)
        text = sources.allow_patterns(text=True)

        assert {"transformer/*", "vae/*", "scheduler/*"} <= set(product)
        assert not any(p.startswith(("text_encoder", "tokenizer")) for p in product)
        assert set(text) == {"text_encoder/*", "tokenizer/*"}
        assert sources.PIPELINE_CODE in product

    def test_every_vae_tensor_prefix_has_one_origin(self):
        prefixes = [p for origin in sources.VAE_ORIGINS for p in origin.prefixes]

        assert len(prefixes) == len(set(prefixes))
        assert not any(a != b and a.startswith(b) for a in prefixes for b in prefixes)


def _copy_pipeline_code(snapshot: Path, destination: Path) -> Path:
    for name in sources.PIPELINE_CODE_COPIES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot / name, target)
    return destination


class TestPipelineCodeHash:
    def test_the_cached_copies_match_the_pin(self, moss_snapshot):
        source = sources.verify_pipeline_code(moss_snapshot)

        assert b"class MOSSImageTransformer" in source

    @pytest.mark.parametrize("name", sources.PIPELINE_CODE_COPIES)
    def test_one_changed_byte_in_any_copy_is_refused(self, moss_snapshot, tmp_path, name):
        """The file is executed on import: a copy that differs from the pin must not load."""
        copy = _copy_pipeline_code(moss_snapshot, tmp_path)
        path = copy / name
        path.write_bytes(path.read_bytes() + b"\n")

        with pytest.raises(sources.MossSourceError, match="refusing to import"):
            sources.verify_pipeline_code(copy)

    def test_a_missing_copy_is_refused(self, moss_snapshot, tmp_path):
        copy = _copy_pipeline_code(moss_snapshot, tmp_path)
        (copy / sources.PIPELINE_CODE_COPIES[-1]).unlink()

        with pytest.raises(sources.MossSourceError, match="missing"):
            sources.verify_pipeline_code(copy)


class TestElementCount:
    def test_it_sums_every_file_of_a_part(self, tmp_path):
        save_file(
            {"a": torch.zeros(2, 3), "b": torch.zeros(4, dtype=torch.bfloat16)},
            str(tmp_path / "x.safetensors"),
        )
        save_file({"c": torch.zeros(5, dtype=torch.int64)}, str(tmp_path / "y.safetensors"))

        assert sources.count_elements(tmp_path) == 6 + 4 + 5

    def test_a_header_whose_width_disagrees_with_its_shape_fails(self, tmp_path):
        header = {"a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 20]}}
        raw = json.dumps(header).encode()
        (tmp_path / "bad.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(24))

        with pytest.raises(sources.MossSourceError, match="disagrees"):
            sources.count_elements(tmp_path)

    def test_the_cached_parts_hold_the_pinned_counts(self, moss_snapshot):
        counts = sources.check_snapshot(moss_snapshot, ("transformer", "vae"))

        assert counts == {k: sources.EXPECTED_ELEMENTS[k] for k in ("transformer", "vae")}
