"""The q4_0 QAT line's distribution (`gemma4.distribution.QAT_Q4_0_PIPELINE` —
`karume-gemma4-qat`): what differs from the regular line, on minimal valid series
({@link gemma4.tests.product_fixture}).

The line runs the regular plan (`gemma4_plan`) with its own `Gemma4Line`, so the checks both lines
share keep their tests in `test_distribution.py`; this file holds what the line decides itself —
the int4 drafter, the schema-4 PLE with its group size, the pins of both containers, the series
names, 12B's vision embedder, the card and the notice.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from container_series import write_component

from _shared.container_read import read_asset_declarations
from _shared.licenses import APACHE_LICENSE_2_0_PATH
from gemma4 import distribution as gemma4_distribution
from gemma4.card import (
    GEMMA4_QAT_DRAFTER_UPSTREAM,
    GEMMA4_QAT_Q4_0_TITLE,
    GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM,
    GEMMA4_UPSTREAM,
    render_gemma4_qat_q4_0_model_card,
)
from gemma4.distribution import (
    GEMMA4_DRAFTER_ROLE,
    GEMMA4_QAT_PLE_GROUP_SIZE,
    GEMMA4_QAT_Q4_0_DEFAULT_QUANT,
    GEMMA4_QAT_Q4_0_NOTICE_MARKDOWN,
    GEMMA4_QAT_Q4_0_REPO_NAME,
    GEMMA4_ROLE,
    GEMMA4_VISION_ROLE,
    QAT_Q4_0_PIPELINE,
    Gemma4Sources,
    gemma4_ple_index,
    gemma4_qat_q4_0_plan,
    gemma4_qat_q4_0_sources,
)
from gemma4.sources import QAT_DRAFTER_PINS, QAT_LINE_PINS, QAT_TARGET_PINS, HubPin
from gemma4.tests import product_fixture as fixture
from karume.container import Provenance
from karume.dist import DistError, assemble_family
from karume.ple import PLE_INDEX_ROLE, PLE_ROLES

#: The per-layer embedding width of the stand-in: two groups of the line's grid per layer (the
#: real E2B / E4B width is 256 — eight groups).
QAT_DIM = 2 * GEMMA4_QAT_PLE_GROUP_SIZE

#: The table the QAT drafter borrows: the stand-in product's int4 embedding
#: ({@link fixture.product_container} with `embedding_storage="i4"`).
LENDER_INT4_TABLE = "gemma4.embed_i4"

#: The ids the 12B stand-in suppresses (upstream order — descending).
SUPPRESSED = (fixture.VOCAB - 1, fixture.VOCAB - 2)

#: The upstream `tokenizer.json` stand-in (the gate compares SHA-256 only).
UPSTREAM_TOKENIZER = b'{"model": "regular -it tokenizer stand-in"}'

#: The `tokenizer.json` a QAT weights' repository ships (the file from before Google's glitch-token
#: fixes — `gemma4.sources`), placed next to the stand-in weights' config so a plan that read the
#: tokenizer from there would fail.
QAT_REPOSITORY_TOKENIZER = b'{"model": "pre-fix tokenizer shipped with the QAT weights"}'

#: A revision no pin names.
OTHER_REVISION = "1" * 40

#: The stand-in's image ids and the budget (**not** the real 258880 / 255999 / 258882 / 280: every
#: number of `pipelineConfig.vision` must come from a reader).
IMAGE_TOKEN = 4
BEGIN_IMAGE_TOKEN = 3
END_IMAGE_TOKEN = 5
BUDGET = 70
SPELLINGS: Mapping[str, str] = {
    "image_token": "<|image|>",
    "boi_token": "<|image>",
    "eoi_token": "<image|>",
}
#: The stand-in vocabulary with the three spellings at the declared ids.
VISION_VOCAB = ("t0", "t1", "t2", "<|image>", "<|image|>", "<image|>")

#: The stand-in vision series' one greedy golden (named by its record).
VISION_GREEDY_GOLDEN = "greedy.corpus-image.safetensors"


@pytest.fixture(autouse=True)
def _small_runtime_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runtime knobs at the stand-in's sizes (its position limit is 37 —
    `test_distribution._small_runtime_knobs`)."""
    monkeypatch.setattr(gemma4_distribution, "GEMMA4_CHUNK_LENGTH", 2)
    monkeypatch.setattr(gemma4_distribution, "GEMMA4_CAPACITY", 4)


def _provenance(revision: str) -> Provenance:
    return Provenance(license="apache-2.0", writer="karume-fixture", upstream_revision=revision)


def _product(
    model: str,
    *,
    ple_storage: str = "i4",
    group_size: int | None = GEMMA4_QAT_PLE_GROUP_SIZE,
    embedding_storage: str = "i4",
    revision: str | None = None,
    **overrides: Any,
) -> list[bytes]:
    """A product container of the line: an int4 embedding, a schema-4 PLE (none for 12B and 26B),
    and the provenance of the model's QAT weights pin; `overrides` go to the fixture as they
    are."""
    ple = model not in ("12b", "26b")
    return fixture.product_container(
        **{
            "storage": "i4",
            "dim": QAT_DIM,
            "ple": ple,
            "assets": (
                fixture.ple_container_assets(
                    dim=QAT_DIM, storage=ple_storage, group_size=group_size
                )
                if ple
                else None
            ),
            "embedding_storage": embedding_storage,
            "provenance": _provenance(revision or QAT_LINE_PINS[model].weights.revision),
            **overrides,
        }
    )


def _drafter(model: str, *, head: str = "i4", revision: str | None = None) -> list[bytes]:
    """A drafter container of the line: int4 throughout, borrowing the int4 table, naming the
    model's QAT drafter pin."""
    return fixture.drafter_container(
        storage="i4",
        head_storage=head,
        shared_tensor=LENDER_INT4_TABLE,
        shared_storage="i4",
        provenance=_provenance(revision or QAT_DRAFTER_PINS[model].revision),
    )


def _vision(revision: str | None = None, **overrides: Any) -> list[bytes]:
    """A vision container of the line, naming 12B's QAT weights pin."""
    return fixture.vision_container(
        provenance=_provenance(revision or QAT_TARGET_PINS["12b"].weights.revision), **overrides
    )


def _build(
    root: Path,
    model: str = "e2b",
    *,
    product: list[bytes] | None = None,
    drafter: list[bytes] | None = None,
    vision: list[bytes] | None = None,
    recorded: bytes = UPSTREAM_TOKENIZER,
    ple_dim: int | None = None,
    suppress: list[int] | None = None,
    max_position: int = fixture.MAX_POSITION,
    config_ids: Mapping[str, int] | None = None,
) -> Gemma4Sources:
    """One model's series of the line under `root`: the product, the drafter, 12B's vision and the
    tokenizer series (its asset recording `recorded` as its source), and the two upstream
    directories the pins name — the weights' config (next to the QAT repository's own
    `tokenizer.json`), and the tokenizer, the generation defaults and 12B's processor
    configuration. `ple_dim` / `suppress` / `max_position` / `config_ids` replace the model's config
    values."""
    base = model.upper() if model != "12b" else "12B"
    twelve_b = model == "12b"
    twenty_six_b = model == "26b"
    sources = Gemma4Sources(
        product=root / "series" / f"gemma4-{model}-qat-product",
        # 26B ships no drafter (ADR 0145 decision 1).
        drafter=None if twenty_six_b else root / "series" / f"gemma4-{model}-qat-drafter",
        tokenizer=root / "series" / f"gemma4-{model}-tokenizer",
        model=root / "inputs" / f"gemma-4-{base}-it-qat-q4_0-unquantized",
        upstream_tokenizer=root / "inputs" / f"gemma-4-{base}-it",
        vision=root / "series" / "gemma4-12b-vision" if twelve_b else None,
    )
    fixture.write_series(
        sources.product,
        sources.tokenizer,
        sources.model,
        container=product if product is not None else _product(model),
        drafter=sources.drafter,
        drafter_bytes=drafter if drafter is not None or twenty_six_b else _drafter(model),
        tokenizer={
            **fixture.tokenizer_asset(),
            "source": {"sha256": hashlib.sha256(recorded).hexdigest()},
            **({"vocab": list(VISION_VOCAB)} if twelve_b else {}),
        },
        text_config={
            **fixture.TEXT_CONFIG,
            "hidden_size_per_layer_input": (
                ple_dim if ple_dim is not None else 0 if twelve_b or twenty_six_b else QAT_DIM
            ),
            "max_position_embeddings": max_position,
        },
        generation_config={
            **fixture.GENERATION_CONFIG,
            **(
                {"suppress_tokens": list(SUPPRESSED if suppress is None else suppress)}
                if twelve_b
                else {}
            ),
        },
    )
    sources.upstream_tokenizer.mkdir(parents=True)
    (sources.model / "generation_config.json").rename(
        sources.upstream_tokenizer / "generation_config.json"
    )
    (sources.upstream_tokenizer / "tokenizer.json").write_bytes(UPSTREAM_TOKENIZER)
    (sources.model / "tokenizer.json").write_bytes(QAT_REPOSITORY_TOKENIZER)
    if twelve_b:
        assert sources.vision is not None
        write_component(sources.vision / "model.krm", vision if vision is not None else _vision())
        (sources.vision / "vision.corpus-image.safetensors").write_bytes(b"not distributed")
        (sources.vision / VISION_GREEDY_GOLDEN).write_bytes(b"not distributed")
        declared = json.loads((sources.model / "config.json").read_text(encoding="utf-8"))
        declared.update(
            {
                "image_token_id": IMAGE_TOKEN,
                "boi_token_id": BEGIN_IMAGE_TOKEN,
                "eoi_token_id": END_IMAGE_TOKEN,
                **(config_ids or {}),
            }
        )
        (sources.model / "config.json").write_text(json.dumps(declared), encoding="utf-8")
        (sources.upstream_tokenizer / "tokenizer_config.json").write_text(
            json.dumps(dict(SPELLINGS)), encoding="utf-8"
        )
        (sources.upstream_tokenizer / "processor_config.json").write_text(
            json.dumps(
                {
                    "image_processor": {
                        "max_soft_tokens": BUDGET,
                        "patch_size": fixture.VISION_PATCH_SIZE,
                        "pooling_kernel_size": fixture.VISION_POOLING,
                    }
                }
            ),
            encoding="utf-8",
        )
        _write_vision_record(sources)
    return sources


def _write_vision_record(sources: Gemma4Sources, **reference: Any) -> None:
    """The vision series' `reference.json` as `gemma4/export_vision.py` writes it, in the parts
    the distribution reads: the greedy goldens, and the budget and the digest of the processor
    configuration the goldens were taken at (this tokenizer source's, unless `reference` replaces
    them)."""
    assert sources.vision is not None
    processor = sources.upstream_tokenizer / "processor_config.json"
    record = {
        "schema": 1,
        "series": sources.vision.name,
        "reference": {
            "series": sources.vision.name,
            "goldens": {VISION_GREEDY_GOLDEN: {"bytes": 15, "sha256": "0" * 64}},
            "softTokenBudget": BUDGET,
            "processorConfig": {
                "bytes": processor.stat().st_size,
                "sha256": hashlib.sha256(processor.read_bytes()).hexdigest(),
            },
            **reference,
        },
    }
    (sources.vision / "reference.json").write_text(json.dumps(record), encoding="utf-8")


def _assemble(root: Path, models: tuple[str, ...] = ("e2b",)) -> tuple[Path, dict[str, Any]]:
    out_dir = root / "models" / GEMMA4_QAT_Q4_0_REPO_NAME
    plans = [gemma4_qat_q4_0_plan(_build(root / model, model), model) for model in models]
    manifest = assemble_family(
        plans,
        out_dir,
        models[0],
        render_card=partial(
            render_gemma4_qat_q4_0_model_card, repo=f"hdae/{GEMMA4_QAT_Q4_0_REPO_NAME}"
        ),
        root_files=QAT_Q4_0_PIPELINE.root_files,
    )
    return out_dir, manifest


def _unfolded(value: Any, model: str) -> Any:
    """`value` with every file path assembly folded into `shared/` read back under the model's
    subtree (`shared/x` → `<model>/x`) — the declaration as a manifest of that model alone
    spells it."""
    if isinstance(value, dict):
        return {
            key: (
                f"{model}/{item.removeprefix('shared/')}"
                if key == "path" and isinstance(item, str) and item.startswith("shared/")
                else _unfolded(item, model)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_unfolded(item, model) for item in value]
    return value


def _files(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


class TestQatLayout:
    def test_it_ships_the_model_and_its_drafter_as_int4_under_one_storage(
        self, tmp_path: Path
    ) -> None:
        """One int4 storage per role; every seat loads both (the parallel and fused seats are
        session flags over the same containers), and the reference seat `i4` is the default."""
        out_dir, manifest = _assemble(tmp_path)
        entry: Mapping[str, Any] = manifest["models"]["e2b"]

        assert {role: sorted(dtypes) for role, dtypes in entry["weights"].items()} == {
            GEMMA4_ROLE: ["i4"],
            GEMMA4_DRAFTER_ROLE: ["i4"],
        }
        assert list(entry["quants"]) == ["i4", "i4-gemvpar", "i4-fast"]
        assert entry["defaultQuant"] == "i4"
        for name, quant in entry["quants"].items():
            assert quant["weights"] == {GEMMA4_ROLE: "i4", GEMMA4_DRAFTER_ROLE: "i4"}, name
        assert entry["quants"]["i4"]["session"] == {}
        files = _files(out_dir)
        assert any(path.startswith("e2b/model/model.i4") for path in files)
        assert any(path.startswith("e2b/drafter/model.i4") for path in files)
        # The acceptance-only files of the series stay behind.
        assert not [path for path in files if "golden" in path or "reference.json" in path]

    def test_12b_ships_its_drafter_with_its_dialect_and_suppressed_ids(
        self, tmp_path: Path
    ) -> None:
        _, manifest = _assemble(tmp_path, ("12b",))
        entry = manifest["models"]["12b"]

        assert sorted(entry["weights"]) == sorted(
            [GEMMA4_DRAFTER_ROLE, GEMMA4_ROLE, GEMMA4_VISION_ROLE]
        )
        assert entry["pipelineConfig"]["chatDialect"] == "gemma4-unified"
        assert entry["pipelineConfig"]["suppressTokens"] == sorted(SUPPRESSED)


#: The session flags of the parallel and fused seats (ADR 0098 / 0104).
PARALLEL_SESSIONS: Mapping[str, Mapping[str, Any]] = {
    "i4-gemvpar": {"linearGemvReduce": "parallel", "stateAttentionReduce": "parallel"},
    "i4-fast": {
        "linearGemvReduce": "parallel",
        "fuseRmsNormAdd": True,
        "stateAttentionReduce": "parallel-fused",
    },
}


class TestQatParallelSeats:
    """Every model carries the parallel and fused seats over the line's int4 storage."""

    @pytest.mark.parametrize("model", ["e2b", "e4b", "12b"])
    def test_the_seats_declare_the_parallel_and_fused_session_flags(self, model: str) -> None:
        quants = gemma4_distribution.gemma4_qat_q4_0_quants(model)

        assert list(quants) == ["i4", "i4-gemvpar", "i4-fast"]
        assert quants["i4"]["session"] == {}
        for name, session in PARALLEL_SESSIONS.items():
            # The order too: the manifest serialises the table as it is.
            assert list(quants[name]["session"].items()) == list(session.items()), name
            assert quants[name]["weights"] == {}, name  # filled with the one storage per role
            # The speed sentence is the model's measured one, never another model's.
            assert quants[name]["description"].endswith(
                gemma4_distribution.GEMMA4_QAT_Q4_0_SPEED_NOTE[model]
            ), name
            assert model.upper() in quants[name]["description"], name

    def test_12b_assembles_all_three_seats(self, tmp_path: Path) -> None:
        _, manifest = _assemble(tmp_path, ("12b",))

        assert list(manifest["models"]["12b"]["quants"]) == ["i4", "i4-gemvpar", "i4-fast"]

    def test_the_descriptions_name_the_output_head_only_where_it_stays_sequential(self) -> None:
        for model, excluded in (("e2b", True), ("e4b", True), ("12b", False)):
            quants = gemma4_distribution.gemma4_qat_q4_0_quants(model)
            for name in PARALLEL_SESSIONS:
                assert ("output head" in quants[name]["description"]) is excluded, (model, name)

    def test_every_model_of_the_line_carries_a_measured_speed_note(self) -> None:
        assert set(gemma4_distribution.GEMMA4_QAT_Q4_0_SPEED_NOTE) == {"e2b", "e4b", "12b"}
        assert set(gemma4_distribution.GEMMA4_QAT_Q4_0_SPEED_NOTE) <= set(
            GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM
        )

    def test_the_seats_do_not_share_session_tables_with_the_declaration(self) -> None:
        """A manifest built for one model cannot alter the line's declaration (each table is a
        copy)."""
        quants = gemma4_distribution.gemma4_qat_q4_0_quants("e2b")
        quants["i4-fast"]["session"]["fuseRmsNormAdd"] = False

        declared = gemma4_distribution.GEMMA4_QAT_Q4_0_PARALLEL_QUANTS["i4-fast"]["session"]
        assert declared["fuseRmsNormAdd"] is True
        again = gemma4_distribution.gemma4_qat_q4_0_quants("e4b")
        assert again["i4-fast"]["session"] == PARALLEL_SESSIONS["i4-fast"]

    def test_every_model_of_the_line_has_a_default_quant_pins_and_a_drafter(self) -> None:
        line = set(GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM)
        assert line == {"e2b", "e4b", "12b"}
        assert set(GEMMA4_QAT_Q4_0_DEFAULT_QUANT) == line
        assert set(GEMMA4_QAT_DRAFTER_UPSTREAM) == line
        assert set(QAT_TARGET_PINS) == line
        assert set(QAT_DRAFTER_PINS) == line


class TestQat12B:
    """12B's plan checks (its one line): no per-layer embeddings, its own tokenizer, its dialect
    and suppressed ids, the two pinned hub snapshots — the shared plan (`gemma4_plan`) run with
    the line."""

    def test_the_sources_are_the_two_pinned_hub_snapshots(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The weights' config from the QAT checkpoint, the tokenizer and the generation defaults
        from 12B-it."""
        constants = pytest.importorskip("huggingface_hub.constants")
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
        pins = QAT_TARGET_PINS["12b"]

        sources = gemma4_qat_q4_0_sources(tmp_path, "12b")

        assert sources.model == (
            tmp_path
            / "hub"
            / "models--google--gemma-4-12B-it-qat-q4_0-unquantized"
            / "snapshots"
            / pins.weights.revision
        )
        assert sources.upstream_tokenizer == (
            tmp_path
            / "hub"
            / "models--google--gemma-4-12B-it"
            / "snapshots"
            / pins.tokenizer.revision
        )

    def test_the_generation_defaults_come_from_the_tokenizer_source(self, tmp_path: Path) -> None:
        """The plan passes with the QAT repository's `tokenizer.json` next to the weights' config
        and reads the suppressed ids from the tokenizer source."""
        sources = _build(tmp_path, "12b")

        plan = gemma4_qat_q4_0_plan(sources, "12b")

        assert not (sources.model / "generation_config.json").exists()
        assert plan.pipeline_config["suppressTokens"] == sorted(SUPPRESSED)

    @pytest.mark.parametrize(
        "recorded", [QAT_REPOSITORY_TOKENIZER, b"E2B's tokenizer.json"], ids=["qat-repo", "e2b"]
    )
    def test_it_refuses_a_tokenizer_asset_compiled_from_another_file(
        self, tmp_path: Path, recorded: bytes
    ) -> None:
        """12B's tokenizer and E2B's differ (their glitch-token fixes dropped other merges), and
        the QAT weights' repository ships the pre-fix file — an asset compiled from either is
        refused."""
        sources = _build(tmp_path, "12b", recorded=recorded)

        with pytest.raises(DistError, match=r"was compiled from a tokenizer\.json"):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_the_container_carries_no_ple_assets(self, tmp_path: Path) -> None:
        out_dir, manifest = _assemble(tmp_path, ("12b",))
        first_part = sorted((out_dir / "12b" / GEMMA4_ROLE).iterdir())[0]
        roles = {role for role, _ in read_asset_declarations(first_part).values()}

        assert not roles & {PLE_INDEX_ROLE, *PLE_ROLES.values()}
        assert manifest["models"]["12b"]["quants"]["i4"]["session"] == {}

    def test_it_refuses_a_container_carrying_ple_assets(self, tmp_path: Path) -> None:
        product = _product("12b", assets=fixture.ple_container_assets())
        sources = _build(tmp_path, "12b", product=product)

        with pytest.raises(DistError, match="PLE assets"):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_it_refuses_a_graph_that_takes_per_layer_inputs(self, tmp_path: Path) -> None:
        sources = _build(tmp_path, "12b", product=_product("12b", ple=True, assets={}))

        with pytest.raises(DistError, match="per_layer_inputs"):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_it_refuses_a_config_that_disagrees_with_the_model_table(self, tmp_path: Path) -> None:
        sources = _build(tmp_path, "12b", ple_dim=QAT_DIM)

        with pytest.raises(DistError, match="GEMMA4_PLE_MODELS"):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_it_refuses_a_suppressed_id_outside_the_vocabulary(self, tmp_path: Path) -> None:
        sources = _build(tmp_path, "12b", suppress=[fixture.VOCAB])

        with pytest.raises(DistError, match="outside the graph's vocabulary"):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_it_refuses_a_container_from_another_revision(self, tmp_path: Path) -> None:
        sources = _build(tmp_path, "12b", product=_product("12b", revision=OTHER_REVISION))

        with pytest.raises(DistError, match="revision"):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_adding_12b_leaves_the_e2b_declarations_byte_identical(self, tmp_path: Path) -> None:
        """12B's dialect and suppressed ids stay on its own entry, and E2B's whole entry — its
        weights and assets declarations included — is unchanged. Only where a file lives may
        differ: some stand-in drafter parts and the stand-in tokenizer are the same bytes for both
        models (an empty part always is), so assembly folds them into `shared/` (the real
        tokenizers and drafters differ); the comparison reads such a path back under `e2b/`."""
        _, alone = _assemble(tmp_path / "alone", ("e2b",))
        _, both = _assemble(tmp_path / "both", ("e2b", "12b"))

        assert json.dumps(_unfolded(both["models"]["e2b"], "e2b"), sort_keys=True) == json.dumps(
            alone["models"]["e2b"], sort_keys=True
        )
        folded = json.dumps([both["models"]["e2b"][key] for key in ("weights", "assets")])
        assert '"shared/' in folded, "nothing folded — the path rewrite above is not exercised"
        for key in ("pipelineConfig", "quants", "defaultQuant", "pipeline"):
            assert json.dumps(both["models"]["e2b"][key], sort_keys=True) == json.dumps(
                alone["models"]["e2b"][key], sort_keys=True
            ), key
        assert "chatDialect" not in both["models"]["e2b"]["pipelineConfig"]
        assert "suppressTokens" not in both["models"]["e2b"]["pipelineConfig"]

    def test_the_card_states_the_position_limit_per_model(self, tmp_path: Path) -> None:
        """12B declares twice E2B / E4B's positions — the overview must not state the default
        model's limit for both."""
        twelve_b = 2 * fixture.MAX_POSITION
        out_dir = tmp_path / "models" / GEMMA4_QAT_Q4_0_REPO_NAME
        plans = [
            gemma4_qat_q4_0_plan(_build(tmp_path / "e2b", "e2b"), "e2b"),
            gemma4_qat_q4_0_plan(_build(tmp_path / "12b", "12b", max_position=twelve_b), "12b"),
        ]
        assemble_family(
            plans,
            out_dir,
            "e2b",
            render_card=partial(
                render_gemma4_qat_q4_0_model_card, repo=f"hdae/{GEMMA4_QAT_Q4_0_REPO_NAME}"
            ),
            root_files=QAT_Q4_0_PIPELINE.root_files,
        )
        card = (out_dir / "README.md").read_text(encoding="utf-8")

        assert (
            f"the model's declared position limit ({fixture.MAX_POSITION} for `e2b`,"
            f" {twelve_b} for `12b`)"
        ) in card


class TestQatStorage:
    def test_it_refuses_a_drafter_whose_head_is_int8(self, tmp_path: Path) -> None:
        """An int4 drafter with an int8 head satisfies "holds int4"; only the int8 ban sees it."""
        sources = _build(tmp_path, drafter=_drafter("e2b", head="i8"))

        with pytest.raises(DistError, match="i8"):
            gemma4_qat_q4_0_plan(sources, "e2b")

    def test_it_refuses_the_regular_int8_drafter(self, tmp_path: Path) -> None:
        sources = _build(tmp_path, drafter=fixture.drafter_container())

        with pytest.raises(DistError, match="drafter"):
            gemma4_qat_q4_0_plan(sources, "e2b")

    def test_it_refuses_a_product_whose_embedding_is_int8(self, tmp_path: Path) -> None:
        """A product with an int8 embedding has the line's graph and PLE inputs; only the int8 ban
        sees it."""
        sources = _build(tmp_path, product=_product("e2b", embedding_storage="i8"))

        with pytest.raises(DistError, match="i8"):
            gemma4_qat_q4_0_plan(sources, "e2b")

    def test_it_refuses_the_regular_bf16_product(self, tmp_path: Path) -> None:
        """The regular line's product series (the checkpoint's bf16 values) has the same graph
        contract; the storage tells it apart first."""
        sources = _build(tmp_path, product=fixture.product_container())

        with pytest.raises(DistError, match=f"{GEMMA4_ROLE}: storage .* has no i4"):
            gemma4_qat_q4_0_plan(sources, "e2b")


class TestQatPle:
    def test_it_reads_the_grouped_index_with_one_scale_per_group(self, tmp_path: Path) -> None:
        sources = _build(tmp_path)
        index = gemma4_ple_index(
            sources.product / "model.krm", storage="i4", group_size=GEMMA4_QAT_PLE_GROUP_SIZE
        )

        assert index["groupSize"] == GEMMA4_QAT_PLE_GROUP_SIZE
        groups = QAT_DIM // GEMMA4_QAT_PLE_GROUP_SIZE
        assert index["scales"]["rowBytes"] == fixture.LAYERS * groups * 4
        assert index["values"]["rowBytes"] == fixture.LAYERS * QAT_DIM // 2

    def test_it_refuses_a_schema_3_index(self, tmp_path: Path) -> None:
        """One scale per row cannot hold the grid — a schema-3 series of the same storage."""
        sources = _build(tmp_path, product=_product("e2b", group_size=None))

        with pytest.raises(DistError, match="schema"):
            gemma4_qat_q4_0_plan(sources, "e2b")

    def test_it_refuses_another_group_size(self, tmp_path: Path) -> None:
        half = GEMMA4_QAT_PLE_GROUP_SIZE // 2
        sources = _build(tmp_path, product=_product("e2b", group_size=half))

        with pytest.raises(DistError, match="groupSize"):
            gemma4_qat_q4_0_plan(sources, "e2b")

    def test_the_regular_reading_refuses_a_grouped_index(self, tmp_path: Path) -> None:
        sources = _build(tmp_path)

        with pytest.raises(DistError, match="groupSize"):
            gemma4_ple_index(sources.product / "model.krm", storage="i4")

    def test_the_group_size_mirrors_the_export(self) -> None:
        export_product_qat = pytest.importorskip("gemma4.export_product_qat")
        assert export_product_qat.GROUP_SIZE == GEMMA4_QAT_PLE_GROUP_SIZE


class TestQatPins:
    def test_it_refuses_a_product_from_another_revision(self, tmp_path: Path) -> None:
        sources = _build(tmp_path, product=_product("e2b", revision=OTHER_REVISION))

        with pytest.raises(DistError, match="revision"):
            gemma4_qat_q4_0_plan(sources, "e2b")

    def test_it_refuses_a_drafter_from_another_revision(self, tmp_path: Path) -> None:
        sources = _build(tmp_path, drafter=_drafter("e2b", revision=OTHER_REVISION))

        with pytest.raises(DistError, match="revision"):
            gemma4_qat_q4_0_plan(sources, "e2b")

    def test_it_refuses_a_tokenizer_compiled_from_another_file(self, tmp_path: Path) -> None:
        sources = _build(tmp_path)
        (sources.upstream_tokenizer / "tokenizer.json").write_bytes(b'{"model": "another"}')

        with pytest.raises(DistError, match="tokenizer"):
            gemma4_qat_q4_0_plan(sources, "e2b")

    def test_the_dist_plan_checks_the_snapshots_against_the_pins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        constants = pytest.importorskip("huggingface_hub.constants")
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "empty-hub"))

        with pytest.raises(DistError, match="missing"):
            gemma4_distribution.gemma4_qat_q4_0_dist_plan(tmp_path / "series", "e2b")

    def test_the_dist_plan_refuses_a_processor_configuration_other_than_12b_s_pin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pinned `tokenizer.json` beside another `processor_config.json` (the budget the
        distribution declares is read from it) stops the plan at the pins."""
        constants = pytest.importorskip("huggingface_hub.constants")
        hub = tmp_path / "hub"
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(hub))
        pins = QAT_TARGET_PINS["12b"]
        _pinned_snapshot(hub, pins.weights)
        tokenizer = _pinned_snapshot(hub, pins.tokenizer)
        series = tmp_path / "series"

        # The pinned snapshots pass the pins; the plan then stops at the missing series.
        with pytest.raises(DistError) as passed:
            gemma4_distribution.gemma4_qat_q4_0_dist_plan(series, "12b")
        assert "processor_config.json" not in str(passed.value)

        (tokenizer / "processor_config.json").unlink()
        (tokenizer / "processor_config.json").write_text(
            '{"image_processor": {"max_soft_tokens": 560}}', encoding="utf-8"
        )
        with pytest.raises(DistError, match=r"processor_config\.json: SHA-256"):
            gemma4_distribution.gemma4_qat_q4_0_dist_plan(series, "12b")


def _pinned_snapshot(hub: Path, pin: HubPin) -> Path:
    """A hub-cache snapshot of `pin`: every file it names links to a blob named by its pinned
    SHA-256 (what the cheap pin check reads — `gemma4.sources.assert_snapshot_pin`)."""
    directory = hub / f"models--{pin.repo.replace('/', '--')}" / "snapshots" / pin.revision
    blobs = hub / f"models--{pin.repo.replace('/', '--')}" / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True)
    for name, digest in {**pin.sha256, **pin.also_checked}.items():
        (blobs / digest).write_bytes(b"stand-in")
        (directory / name).symlink_to(Path("..") / ".." / "blobs" / digest)
    return directory


class TestQatNaming:
    @pytest.mark.parametrize(
        ("model", "product", "tokenizer"),
        [
            ("e2b", "gemma4-e2b-qat-product", "gemma4-e2b-tokenizer"),
            ("e4b", "gemma4-e4b-qat-product", "gemma4-e4b-tokenizer"),
            ("12b", "gemma4-12b-product", "gemma4-12b-tokenizer"),
        ],
    )
    def test_the_series_are_the_writers(
        self, tmp_path: Path, model: str, product: str, tokenizer: str
    ) -> None:
        pytest.importorskip("huggingface_hub")
        sources = gemma4_qat_q4_0_sources(tmp_path, model)

        assert sources.product == tmp_path / product
        assert sources.drafter == tmp_path / f"gemma4-{model}-qat-drafter"
        assert sources.tokenizer == tmp_path / tokenizer
        assert sources.vision == (tmp_path / "gemma4-12b-vision" if model == "12b" else None)

    def test_a_model_outside_the_line_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="no model 'unlisted'"):
            gemma4_qat_q4_0_sources(tmp_path, "unlisted")

    def test_the_repository_is_the_line_s_for_every_model(self) -> None:
        for model in GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM:
            assert QAT_Q4_0_PIPELINE.repo_name(model) == "karume-gemma4-qat"


class TestQatVision:
    """12B's vision embedder: a third weights role, one bf16 storage, declared rows."""

    def test_every_quant_of_12b_maps_the_vision_role_to_bf16(self, tmp_path: Path) -> None:
        _, manifest = _assemble(tmp_path, ("12b",))
        entry = manifest["models"]["12b"]

        assert sorted(entry["weights"][GEMMA4_VISION_ROLE]) == ["bf16"]
        assert list(entry["quants"]) == ["i4", "i4-gemvpar", "i4-fast"]
        for name, quant in entry["quants"].items():
            assert quant["weights"] == {
                GEMMA4_ROLE: "i4",
                GEMMA4_DRAFTER_ROLE: "i4",
                GEMMA4_VISION_ROLE: "bf16",
            }, name

    def test_the_container_lands_at_its_storage_s_path_without_the_goldens(
        self, tmp_path: Path
    ) -> None:
        out_dir, _ = _assemble(tmp_path, ("12b",))
        files = _files(out_dir)

        assert any(path.startswith("12b/vision/model.bf16") for path in files)
        assert not [path for path in files if "corpus-image" in path]

    def test_the_pipeline_config_reads_every_number_from_its_source(self, tmp_path: Path) -> None:
        plan = gemma4_qat_q4_0_plan(_build(tmp_path, "12b"), "12b")

        assert plan.pipeline_config["vision"] == {
            "softTokenBudget": BUDGET,
            "imageTokenId": IMAGE_TOKEN,
            "beginImageTokenId": BEGIN_IMAGE_TOKEN,
            "endImageTokenId": END_IMAGE_TOKEN,
            "hiddenSize": fixture.HIDDEN,
        }

    @pytest.mark.parametrize("model", ["e2b", "e4b"])
    def test_e2b_and_e4b_declare_no_vision(self, tmp_path: Path, model: str) -> None:
        plan = gemma4_qat_q4_0_plan(_build(tmp_path, model), model)

        assert GEMMA4_VISION_ROLE not in plan.weights
        assert GEMMA4_VISION_ROLE not in plan.artifacts
        assert "vision" not in plan.pipeline_config

    def test_it_refuses_12b_without_its_vision_series(self, tmp_path: Path) -> None:
        sources = replace(_build(tmp_path, "12b"), vision=None)

        with pytest.raises(DistError, match="the vision series is missing"):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_it_refuses_a_vision_series_for_a_model_without_one(self, tmp_path: Path) -> None:
        twelve_b = _build(tmp_path / "12b", "12b")
        sources = replace(_build(tmp_path / "e2b", "e2b"), vision=twelve_b.vision)

        with pytest.raises(DistError, match="the vision series is given"):
            gemma4_qat_q4_0_plan(sources, "e2b")

    def test_the_bf16_exception_is_the_vision_role_s_alone(self) -> None:
        storage = gemma4_distribution.GEMMA4_QAT_Q4_0_STORAGE

        assert storage.required[GEMMA4_VISION_ROLE] == "bf16"
        assert "bf16" not in storage.forbidden[GEMMA4_VISION_ROLE]
        assert "bf16" in storage.forbidden[GEMMA4_ROLE]
        assert "bf16" in storage.forbidden[GEMMA4_DRAFTER_ROLE]

    def test_it_refuses_a_vision_container_on_the_int4_grid(self, tmp_path: Path) -> None:
        """The embedder is not on the grid: an int4 container would hold rounded weights with the
        right graph, so only the storage tells it apart."""
        sources = _build(tmp_path, "12b", vision=_vision(storage="i4"))

        with pytest.raises(DistError, match=f"{GEMMA4_VISION_ROLE}: storage .* has no bf16"):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_it_refuses_a_vision_container_from_another_revision(self, tmp_path: Path) -> None:
        sources = _build(tmp_path, "12b", vision=_vision(OTHER_REVISION))

        with pytest.raises(DistError, match="revision"):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_it_refuses_an_embedder_of_another_width(self, tmp_path: Path) -> None:
        """The output must fit the main graph, and the rows what the processor emits."""
        wide = _build(tmp_path / "wide", "12b", vision=_vision(hidden_size=fixture.HIDDEN + 1))
        with pytest.raises(DistError, match="hidden_size"):
            gemma4_qat_q4_0_plan(wide, "12b")

        rows = _build(tmp_path / "rows", "12b", vision=_vision(pixel_width=6912))
        with pytest.raises(DistError, match="the vision graph's inputs"):
            gemma4_qat_q4_0_plan(rows, "12b")

    @pytest.mark.parametrize("config_id", ["image_token_id", "boi_token_id", "eoi_token_id"])
    def test_it_refuses_an_image_id_that_is_another_token(
        self, tmp_path: Path, config_id: str
    ) -> None:
        sources = _build(tmp_path, "12b", config_ids={config_id: 0})

        with pytest.raises(DistError, match=rf"{config_id} 0 is 't0'"):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_it_refuses_a_soft_mask_of_another_form(self, tmp_path: Path) -> None:
        sources = _build(tmp_path, "12b", product=_product("12b", soft_mask_width=fixture.HIDDEN))

        with pytest.raises(DistError, match="soft_mask"):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_the_input_spellings_mirror_the_exports(self) -> None:
        pytest.importorskip("torch")
        from gemma4 import export_product as product

        assert product.SOFT_EMBEDS == gemma4_distribution.GEMMA4_SOFT_EMBEDS
        assert product.SOFT_MASK == gemma4_distribution.GEMMA4_SOFT_MASK


class TestQatVisionSeries:
    """The vision series the role is placed from is a full one — its record and the greedy goldens
    it names — and its goldens were taken at the budget and processor configuration declared."""

    def test_it_refuses_a_series_without_its_record(self, tmp_path: Path) -> None:
        """What `gemma4/export_vision.py --embedder-only` writes: the container and the embedder
        goldens, no record and no greedy golden."""
        sources = _build(tmp_path, "12b")
        assert sources.vision is not None
        (sources.vision / "reference.json").unlink()
        (sources.vision / VISION_GREEDY_GOLDEN).unlink()

        with pytest.raises(DistError, match=r"reference\.json is missing"):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_it_refuses_a_series_without_the_greedy_goldens_its_record_names(
        self, tmp_path: Path
    ) -> None:
        sources = _build(tmp_path, "12b")
        assert sources.vision is not None
        (sources.vision / VISION_GREEDY_GOLDEN).unlink()

        with pytest.raises(DistError, match=rf"greedy goldens \['{VISION_GREEDY_GOLDEN}'\]"):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_it_refuses_a_record_that_names_no_greedy_golden(self, tmp_path: Path) -> None:
        sources = _build(tmp_path, "12b")
        _write_vision_record(sources, goldens={})

        with pytest.raises(DistError, match="names no greedy goldens"):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_it_refuses_goldens_taken_at_another_budget(self, tmp_path: Path) -> None:
        sources = _build(tmp_path, "12b")
        _write_vision_record(sources, softTokenBudget=560)

        with pytest.raises(
            DistError, match=rf"softTokenBudget 560, the distribution declares {BUDGET}"
        ):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_it_refuses_a_processor_configuration_edited_after_the_goldens(
        self, tmp_path: Path
    ) -> None:
        """A hand-edited cache file would change the declared budget alone."""
        sources = _build(tmp_path, "12b")
        path = sources.upstream_tokenizer / "processor_config.json"
        edited = json.loads(path.read_text(encoding="utf-8"))
        edited["image_processor"]["max_soft_tokens"] = 560
        path.write_text(json.dumps(edited), encoding="utf-8")

        with pytest.raises(
            DistError, match=rf"softTokenBudget {BUDGET}, the distribution declares 560"
        ):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_it_refuses_goldens_taken_with_another_processor_configuration(
        self, tmp_path: Path
    ) -> None:
        """The same budget, other bytes (another resize rule, say): the digest tells them apart."""
        sources = _build(tmp_path, "12b")
        _write_vision_record(sources, processorConfig={"bytes": 1, "sha256": "f" * 64})
        declared = hashlib.sha256(
            (sources.upstream_tokenizer / "processor_config.json").read_bytes()
        ).hexdigest()

        with pytest.raises(
            DistError, match=rf"SHA-256 '{'f' * 64}', the distribution reads {declared}"
        ):
            gemma4_qat_q4_0_plan(sources, "12b")

    def test_the_record_s_file_name_mirrors_the_writer(self) -> None:
        pytest.importorskip("torch")
        from gemma4 import provenance

        assert provenance.REFERENCE_FILE == gemma4_distribution.GEMMA4_REFERENCE_FILE


class TestQatCard:
    def test_it_names_every_redistributed_upstream_base_model_first(self, tmp_path: Path) -> None:
        out_dir, _ = _assemble(tmp_path, ("e2b", "12b"))
        card = (out_dir / "README.md").read_text(encoding="utf-8")
        front_matter = card.split("---", 2)[1]

        listed = [line.strip() for line in front_matter.splitlines() if "google/" in line]
        assert listed == [
            "- google/gemma-4-E2B-it",
            "- google/gemma-4-E2B-it-qat-q4_0-unquantized",
            "- google/gemma-4-12B-it",
            "- google/gemma-4-12B-it-qat-q4_0-unquantized",
            "- google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant",
            "- google/gemma-4-12B-it-qat-q4_0-unquantized-assistant",
        ]
        assert f"# {GEMMA4_QAT_Q4_0_TITLE}" in card
        assert "- **`e2b` drafter**: [google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant]" in card
        assert "the per-layer embeddings of `e2b`" in card
        assert "`12b` has no per-layer embeddings" in card
        assert "// speculative: {}, // load the MTP drafter — available for: e2b / 12b" in card

    def test_only_a_model_carrying_the_embedder_is_said_to_take_images(
        self, tmp_path: Path
    ) -> None:
        """The overview tells 12B's image input from the towers E2B / E4B do not ship, and the
        attribution names what was extracted."""
        out_dir, _ = _assemble(tmp_path, ("e2b", "12b"))
        card = (out_dir / "README.md").read_text(encoding="utf-8")

        assert "- **Image input (`12b`)**: the encoder-free vision embedder ships" in card
        assert "- **Not multimodal for the others.**" in card
        assert "vision and audio towers for `e2b`" in card
        assert "they cannot take images or audio." in card
        assert "- **Every text weight keeps the checkpoint's own 4-bit grid.**" in card
        assert "The encoder-free vision embedder of `12b` was extracted" in card

    def test_a_manifest_without_the_embedder_stays_text_only(self, tmp_path: Path) -> None:
        out_dir, _ = _assemble(tmp_path, ("e2b",))
        card = (out_dir / "README.md").read_text(encoding="utf-8")

        assert "- **Not multimodal here.** The published checkpoint also carries" in card
        assert "this distribution contains neither, and cannot take images or audio." in card
        assert "Image input" not in card
        assert "- **Every weight keeps the checkpoint's own 4-bit grid.**" in card
        assert "vision embedder" not in card

    def test_the_reference_seat_says_the_embedder_is_not_int4(self) -> None:
        note = gemma4_distribution.GEMMA4_QAT_Q4_0_VISION_NOTE
        for model, carries in (("e2b", False), ("e4b", False), ("12b", True)):
            description = gemma4_distribution.gemma4_qat_q4_0_quants(model)["i4"]["description"]
            assert description.endswith(note) is carries, model
        declared = gemma4_distribution.GEMMA4_QAT_Q4_0_QUANTS["i4"]["description"]
        assert not declared.endswith(note)

    def test_a_manifest_without_per_layer_embeddings_states_none(self, tmp_path: Path) -> None:
        out_dir, _ = _assemble(tmp_path, ("12b",))
        card = (out_dir / "README.md").read_text(encoding="utf-8")

        assert "The linear weights and the embedding table (also the output head) are" in card
        assert "per-layer embeddings of" not in card

    def test_a_session_boolean_is_spelled_as_the_manifest_spells_it(self, tmp_path: Path) -> None:
        """A session boolean is spelled in the same JSON notation as the manifest — a Python `True`
        would disagree with the spelling of TypeScript that a reader pastes as is."""
        out_dir, _ = _assemble(tmp_path)
        card = (out_dir / "README.md").read_text(encoding="utf-8")

        fast = next(line for line in card.splitlines() if line.startswith("| `i4-fast`"))
        assert "`fuseRmsNormAdd` = `true`" in fast
        assert "True" not in fast

    def test_the_usage_points_function_calling_and_thinking_at_the_session(
        self, tmp_path: Path
    ) -> None:
        """`Pipeline.chat` takes no `tools`, so the card names the entry point that does."""
        out_dir, _ = _assemble(tmp_path)
        usage = (out_dir / "README.md").read_text(encoding="utf-8").split("## Usage")[1]

        assert "`new ChatSession(pipeline, { tools, enableThinking })`" in usage
        assert "`sendToolResults` continues it" in usage
        assert "tool calls,\nthinking channels" not in usage

    def test_a_model_without_long_context_checks_is_stated_as_not_checked(
        self, tmp_path: Path
    ) -> None:
        out_dir, _ = _assemble(tmp_path, ("e2b", "e4b", "12b"))
        card = (out_dir / "README.md").read_text(encoding="utf-8")
        section = card.split("## Long context")[1].split("\n## ")[0]

        assert "`e4b` was not checked." in section
        assert "| `e4b`" not in section

    def test_the_long_context_section_needs_both_measured_models(self, tmp_path: Path) -> None:
        """The section compares `e2b` and `12b`, so a repository carrying one of them would state
        results about a model it does not ship."""
        out_dir, _ = _assemble(tmp_path, ("12b",))

        assert "## Long context" not in (out_dir / "README.md").read_text(encoding="utf-8")

    def test_the_long_context_section_sits_between_usage_and_the_models(
        self, tmp_path: Path
    ) -> None:
        out_dir, _ = _assemble(tmp_path, ("e2b", "12b"))
        card = (out_dir / "README.md").read_text(encoding="utf-8")

        assert card.index("## Usage") < card.index("## Long context") < card.index("## Model: e2b")
        assert "not checked" not in card

    def test_it_refuses_a_model_outside_the_line(self, tmp_path: Path) -> None:
        _, manifest = _assemble(tmp_path)
        renamed = {**manifest, "models": {"unlisted": manifest["models"]["e2b"]}}

        with pytest.raises(ValueError, match="'unlisted'"):
            render_gemma4_qat_q4_0_model_card(renamed, "hdae/karume-gemma4-qat")


class TestQatLegalText:
    def test_it_ships_the_license_verbatim_and_the_line_s_notice(self, tmp_path: Path) -> None:
        out_dir, _ = _assemble(tmp_path)

        assert (out_dir / "LICENSE.md").read_bytes() == APACHE_LICENSE_2_0_PATH.read_bytes()
        assert (out_dir / "NOTICE.md").read_text(encoding="utf-8") == (
            GEMMA4_QAT_Q4_0_NOTICE_MARKDOWN
        )

    def test_the_notice_is_true_per_model_about_what_was_read(self) -> None:
        notice = " ".join(GEMMA4_QAT_Q4_0_NOTICE_MARKDOWN.split())

        assert "For the E2B and E4B checkpoints, the **text decoder only** was extracted" in notice
        assert "their vision and audio towers were never read" in notice
        assert "the **text decoder and the encoder-free vision embedder** were extracted" in notice
        assert "its audio embedder was never read" in notice
        assert "keeps the checkpoint's BF16 values" in notice

    def test_the_notice_names_every_upstream_of_the_line(self) -> None:
        for model, weights in GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM.items():
            assert f"`{weights}`" in GEMMA4_QAT_Q4_0_NOTICE_MARKDOWN, model
            assert f"`{GEMMA4_QAT_DRAFTER_UPSTREAM[model]}`" in GEMMA4_QAT_Q4_0_NOTICE_MARKDOWN
            assert f"`{GEMMA4_UPSTREAM[model]}`" in GEMMA4_QAT_Q4_0_NOTICE_MARKDOWN

    def test_the_notice_states_26b_s_extraction_and_its_f16_expert_steps(self) -> None:
        notice = " ".join(GEMMA4_QAT_Q4_0_NOTICE_MARKDOWN.split())

        assert "For the 26B A4B checkpoint, the **text decoder only** was extracted" in notice
        assert "its vision tower was never read" in notice
        assert "expert weights of the 26B A4B checkpoint keep the same codes" in notice
        assert "stored as float16 instead of float32" in notice


class TestQat26B:
    """26B's plan (its one line — ADR 0145): 12B's graph form and dialect, its pins, no drafter
    (so `speculative` is refused at weights resolution), no vision, the reference seat alone."""

    def test_it_ships_the_model_role_alone_with_12b_s_dialect(self, tmp_path: Path) -> None:
        out_dir, manifest = _assemble(tmp_path, ("26b",))
        entry = manifest["models"]["26b"]

        assert sorted(entry["weights"]) == [GEMMA4_ROLE]
        assert list(entry["quants"]) == ["i4"]
        assert entry["quants"]["i4"]["weights"] == {GEMMA4_ROLE: "i4"}
        assert entry["quants"]["i4"]["label"] == "QAT int4 grid"
        assert "drafter" not in entry["quants"]["i4"]["description"]
        assert "f16 steps" in entry["quants"]["i4"]["description"]
        assert entry["pipelineConfig"]["chatDialect"] == "gemma4-unified"
        assert "suppressTokens" not in entry["pipelineConfig"]
        assert "vision" not in entry["pipelineConfig"]
        assert not [path for path in _files(out_dir) if path.startswith("26b/drafter")]
        card = (out_dir / "README.md").read_text(encoding="utf-8")
        assert "speculative" not in card.split("## Base weights")[0].lower()
        assert "the routed experts of `26b` keep the same codes with f16 steps" in " ".join(
            card.split()
        )

    def test_a_drafter_series_given_for_26b_is_refused(self, tmp_path: Path) -> None:
        sources = _build(tmp_path, "26b")
        given = replace(sources, drafter=tmp_path / "series" / "gemma4-26b-qat-drafter")

        with pytest.raises(DistError, match="ships no drafter"):
            gemma4_qat_q4_0_plan(given, "26b")

    def test_the_sources_are_its_product_series_and_its_pinned_snapshots(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """12B's series naming (`gemma4-26b-product`), its own tokenizer series, no drafter; the
        weights' config from the QAT checkpoint, the tokenizer from 26B-it."""
        constants = pytest.importorskip("huggingface_hub.constants")
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
        pins = QAT_LINE_PINS["26b"]

        sources = gemma4_qat_q4_0_sources(tmp_path / "series", "26b")

        assert sources.product.name == "gemma4-26b-product"
        assert sources.tokenizer.name == "gemma4-26b-tokenizer"
        assert sources.drafter is None and sources.vision is None
        assert sources.model.name == pins.weights.revision
        assert (
            sources.model.parent.parent.name
            == "models--google--gemma-4-26B-A4B-it-qat-q4_0-unquantized"
        )
        assert sources.upstream_tokenizer.name == pins.tokenizer.revision
        assert sources.upstream_tokenizer.parent.parent.name == "models--google--gemma-4-26B-A4B-it"

    def test_the_other_models_keep_their_reference_seat_text(self) -> None:
        for model in ("e2b", "e4b", "12b"):
            seat = gemma4_distribution.gemma4_qat_q4_0_quants(model)["i4"]
            assert seat["label"] == "QAT int4 grid, drafter included", model
            assert "f16 steps" not in seat["description"], model
