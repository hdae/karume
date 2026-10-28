"""The **generic rendering pieces** of the model card (`karume.modelcard`).

The `karume.json` of a real distribution is not used — the job here is to see that the values of a
fake manifest reach the body as they are (= not a single hand-written number is mixed in), so values
**different** from the real ones detect more.

NOTE: every per-pipeline card template has moved out of the wheel (ADR 0065 stages 3+4 done), so
the per-template cases live in `tools/export-recipes/<family>/tests/test_card.py`. What remains here
is the layer observable with the core alone, without going through a template.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from karume.modelcard import (
    CardMetadata,
    file_ref_key,
    from_pretrained,
    frontmatter,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)

# ---- optional frontmatter fields ------------------------------------------------
#
# None of the recipe-side families has `base_model_relation` / `license_name` / `license_link`
# (distributions that keep the storage form + standard SPDX tags), or if one does it is outside the
# wheel, so the **emitted when written** branch is never exercised through a family. The fields
# themselves are the core's responsibility, so they are checked directly here.


class TestFrontmatterOptionalFields:
    """The three optional fields "emit one line only when they have a value" — listing them empty
    makes a different claim.

    A card that lists `license_name` / `license_link` empty reads as "a custom license without a
    name", and a card that lists `base_model_relation` empty kills HF's inference (the doc of
    CardMetadata).
    """

    @staticmethod
    def _metadata(**overrides: Any) -> CardMetadata:
        return CardMetadata(
            pipeline_tag="text-to-speech",
            base_model=("owner/base",),
            license="other",
            tags=("webgpu",),
            **overrides,
        )

    def test_it_omits_every_optional_field_that_is_unset(self) -> None:
        lines = frontmatter(self._metadata())
        for absent in ("base_model_relation", "license_name", "license_link"):
            assert not any(line.startswith(absent) for line in lines), absent

    def test_it_writes_each_optional_field_that_is_set(self) -> None:
        lines = frontmatter(
            self._metadata(
                base_model_relation="quantized",
                license_name="owner-terms",
                license_link="https://example.invalid/terms",
            )
        )
        assert "base_model_relation: quantized" in lines
        assert "license_name: owner-terms" in lines
        assert "license_link: https://example.invalid/terms" in lines


# ---- public rendering pieces for recipes -----------------------------------------
#
# `quants` / `models` / `require_pipeline` / `render` are the public surface that the `card.py`
# outside the wheel calls by name (ADR 0065 stage 6). So that a regression of this surface shows
# when only the core runs, the checks sit on the core side with a fake manifest as the subject (the
# recipe-side checks exercise it indirectly per family, so the subject of a regression is unclear).


def _ref(path: str, size: int, sha: str) -> dict[str, Any]:
    """The three fields of a file reference (values **different** from the real ones — a
    hand-written value mixed in shows here)."""
    return {"path": path, "size": size, "sha256": sha}


def _container(*refs: dict[str, Any]) -> dict[str, Any]:
    """One container entry (the card reads only the sizes of `parts`)."""
    return {
        "descriptor": {
            "graph": {"length": 64, "sha256": "0" * 64},
            "model": {"length": 32, "sha256": "1" * 64},
        },
        "parts": list(refs),
    }


def _manifest() -> dict[str, Any]:
    """A fake manifest with two models and a shared asset (the form of ADR 0109 decision 3).

    The models are ordered `zeta` → `alpha`, the **reverse** of lexical order — any sorting mixed
    in shows in the table order. `tables` has both dtypes f16 / i8 point at the same path (the same
    form as a shared component).
    """
    return {
        "format": "karume/5",
        "generator": "karume/9.9.9",
        "defaultModel": "zeta",
        "models": {
            "zeta": {
                "pipeline": "fake/1",
                "weights": {
                    "front": {
                        "f16": {
                            "container": _container(
                                _ref("zeta/front-f16-00001-of-00002.krm", 4096, "a" * 64),
                                _ref("zeta/front-f16-00002-of-00002.krm", 64, "b" * 64),
                            )
                        },
                        "i8": {"container": _container(_ref("zeta/front-i8.krm", 2048, "c" * 64))},
                    },
                    "tables": {
                        "f16": {"container": _container(_ref("zeta/tables.krm", 128, "d" * 64))},
                        "i8": {"container": _container(_ref("zeta/tables.krm", 128, "d" * 64))},
                    },
                },
                "assets": {"tokenizer": _ref("shared/tokenizer.json", 32, "e" * 64)},
                "quants": {
                    "f16": {"weights": {"front": "f16", "tables": "f16"}, "session": {}},
                    "w8": {
                        "weights": {"front": "i8", "tables": "i8"},
                        "session": {"linearCompute": "a8"},
                        "label": "Half size (int8)",
                        "description": "Both components in int8 storage.",
                    },
                },
                "defaultQuant": "w8",
                "pipelineConfig": {},
            },
            "alpha": {
                "pipeline": "fake/1",
                "weights": {
                    "front": {
                        "f16": {"container": _container(_ref("alpha/front.krm", 512, "f" * 64))}
                    }
                },
                "assets": {},
                "quants": {"f16": {"weights": {"front": "f16"}, "session": {}}},
                "defaultQuant": "f16",
                "pipelineConfig": {},
            },
        },
    }


def _row(lines: Sequence[str], name: str) -> str:
    """Looks up the row of that preset in the quant table (by preset name, not by column
    position)."""
    return next(line for line in lines if line.startswith(f"| `{name}`"))


class TestQuantsDownload:
    """The Download column = the bytes a reader who chooses that preset actually downloads (ruling
    of 2026-09-03).

    There are many parts, so the table with one row per file was retired. What the reader wants to
    know, "how many GiB does this preset download", is one cell per preset holding the total.
    """

    def test_it_sums_the_parts_and_assets_the_quant_selects(self) -> None:
        """f16 = front part 4,096 + 64 + tables 128 + tokenizer 32 = 4,320 B (= 4.22 KiB)."""
        lines = quants(_manifest()["models"]["zeta"])

        assert "| 4.22 KiB (32 B shared) |" in _row(lines, "f16")

    def test_it_leaves_out_the_dtype_the_quant_does_not_select(self) -> None:
        """w8 = front i8 2,048 + tables 128 + tokenizer 32 = 2,208 B = 2.16 KiB (f16's 4,096 is not
        included).

        Without a recount per preset, moving the f16-side weights would move the w8 cell too.
        """
        manifest = _manifest()
        manifest["models"]["zeta"]["weights"]["front"]["f16"]["container"]["parts"][0]["size"] = (
            1 << 30
        )

        before = _row(quants(_manifest()["models"]["zeta"]), "w8")
        after = _row(quants(manifest["models"]["zeta"]), "w8")
        assert "| 2.16 KiB (" in before
        assert after == before
        assert "| 1.00 GiB (" in _row(quants(manifest["models"]["zeta"]), "f16")

    def test_it_counts_every_part_of_the_selected_container(self) -> None:
        """Each part is a file downloaded on its own — f16's second part of 64 B is in the total."""
        manifest = _manifest()
        parts = manifest["models"]["zeta"]["weights"]["front"]["f16"]["container"]["parts"]
        del parts[1]

        assert "| 4.16 KiB (" in _row(quants(manifest["models"]["zeta"]), "f16")

    def test_a_file_two_components_point_at_is_counted_once(self) -> None:
        """A file that two components point at (the same form as the merged rope_base) is not
        counted twice."""
        manifest = _manifest()
        front = manifest["models"]["zeta"]["weights"]["front"]["f16"]["container"]["parts"]
        manifest["models"]["zeta"]["weights"]["tables"]["f16"]["container"]["parts"] = front

        # front 4,096 + rope 64 + tokenizer 32 = 4,192 B (the 128 of tables drops out as the same
        # path).
        assert "| 4.09 KiB (" in _row(quants(manifest["models"]["zeta"]), "f16")

    def test_it_calls_out_the_bytes_a_second_model_does_not_fetch_again(self) -> None:
        """The `shared/` part is added as the amount "a second model does not download again"."""
        assert "(32 B shared)" in _row(quants(_manifest()["models"]["zeta"]), "f16")

    def test_a_model_that_shares_nothing_keeps_the_cell_bare(self) -> None:
        """A note that does not apply is not emitted (spelling 0 B shared reads as if something were
        shared)."""
        row = _row(quants(_manifest()["models"]["alpha"]), "f16")

        assert row.count("512 B") == 1
        assert "shared" not in row

    def test_bytes_borrowed_from_another_repository_count_as_shared(self) -> None:
        """A cross-repository reference (ADR 0038 §7, superseded — now ADR 0109 decision 3) is not
        downloaded twice either — it goes into the same "not downloaded again" amount."""
        manifest = _manifest()
        manifest["models"]["alpha"]["weights"]["front"]["f16"]["container"]["parts"] = [
            {
                "repo": "hdae/karume-source",
                "revision": "9" * 40,
                "path": "source/front.krm",
                "size": 512,
                "sha256": "f" * 64,
            }
        ]

        assert "| 512 B (512 B shared) |" in _row(quants(manifest["models"]["alpha"]), "f16")

    def test_it_calls_out_assets_that_are_a_real_share_of_the_download(self) -> None:
        """Assets are **only read on the host side** (gemma4's PLE sidecar) — they are the
        difference between Download and the resident size.

        In w8 the tokenizer's 32 B is 1.4% (≥ 1%) of 2,208 B, so the breakdown is shown. In f16 it
        is 0.7% of 4,320 B, so it is not — writing it on every card would make a line that only
        draws the reader's eye to a tokenizer of a few MiB.
        """
        lines = quants(_manifest()["models"]["zeta"])

        assert "(32 B shared; 32 B of assets, read on the host)" in _row(lines, "w8")
        assert "of assets" not in _row(lines, "f16")


class TestDownloadSizeUnits:
    """The unit is chosen from the value **after rounding** (`3.24 GiB` / `248 MiB` — the precision
    stays even across units).

    Choosing before rounding gives, in the band that carries over, the spelling `1024 MiB`, which
    is neither 3 significant digits nor the right unit. The carry threshold is **1024**, not 1000
    (= exactly 1 of the next unit up) — `1000 MiB` is a correct MiB spelling, and moving it up
    would give `0.977 GiB`, "less than one unit".
    """

    @pytest.mark.parametrize(
        ("size", "spelled"),
        [
            # The band that carries over (an implementation that chooses the unit before rounding
            # gives `1024 MiB` / `1024 KiB` here).
            (1_073_741_823, "1.00 GiB"),
            (1_048_575, "1.00 MiB"),
            # The other side of the boundary — rounding does not reach 1024, so the unit stays.
            (1_073_217_535, "1023 MiB"),
            # A band that is correct in MiB despite 4 digits (one unit up would spell less than
            # one unit).
            (1_048_576_000, "1000 MiB"),
            # Counterparts showing the existing spellings do not move (the two examples of the
            # docstring, and raw bytes below one unit).
            (3_479_000_000, "3.24 GiB"),
            (260_000_000, "248 MiB"),
            (512, "512 B"),
        ],
    )
    def test_it_spells_the_download_cell_at_three_significant_digits(
        self, size: int, spelled: str
    ) -> None:
        manifest = _manifest()
        manifest["models"]["alpha"]["weights"]["front"]["f16"]["container"]["parts"][0]["size"] = (
            size
        )

        assert f"| {spelled} |" in _row(quants(manifest["models"]["alpha"]), "f16")


class TestQuantsNotes:
    """Notes carried over from the retired file table — emitted **only when they apply** (the
    conditions come from the manifest)."""

    def test_it_points_at_karume_json_for_the_per_file_values(self) -> None:
        """With sha256 gone from the table, where the source of truth lives is always kept (what the
        fetch layer checks against)."""
        lines = quants(_manifest()["models"]["zeta"])

        assert (
            "Per-file `size` and `sha256` live in `karume.json` — verify against that at the fetch"
            " layer." in lines
        )

    def test_the_dtype_vocabulary_note_spells_the_packed_labels(self) -> None:
        """If the vocabulary sentence drops the packed spellings (`i4` / `i2`), it no longer names
        the labels the quant table uses (the dialect note that once relied on them was retired —
        ADR 0109 decision 3)."""
        assert any(
            line.startswith("Dtype labels use the runtime's **storage dtype vocabulary**")
            and "`i4` / `i2`" in line
            for line in quants(_manifest()["models"]["zeta"])
        )

    def test_a_shared_path_is_explained(self) -> None:
        """The meaning of `(N shared)` pairs with one sentence of the body (without the note, what
        it is shared with cannot be read)."""
        assert any(
            line.startswith("A path under `shared/`")
            for line in quants(_manifest()["models"]["zeta"])
        )

    def test_a_model_without_a_shared_path_says_nothing_about_shared(self) -> None:
        assert not any(
            line.startswith("A path under `shared/`")
            for line in quants(_manifest()["models"]["alpha"])
        )

    def test_it_names_the_repository_borrowed_bytes_come_from(self) -> None:
        """The note becomes a fact only once it names the target (repository + pinned commit)."""
        manifest = _manifest()
        manifest["models"]["alpha"]["weights"]["front"]["f16"]["container"]["parts"] = [
            {
                "repo": "hdae/karume-source",
                "revision": "9" * 40,
                "path": "source/front.krm",
                "size": 512,
                "sha256": "f" * 64,
            }
        ]
        lines = quants(manifest["models"]["alpha"])

        assert any(
            line.startswith(
                "Some components are fetched from"
                " [`hdae/karume-source`](https://huggingface.co/hdae/karume-source)"
                " at commit `9999999999999999…`"
            )
            for line in lines
        )

    def test_a_self_contained_model_says_nothing_about_other_repositories(self) -> None:
        assert not any(
            line.startswith("Some components are fetched from")
            for line in quants(_manifest()["models"]["zeta"])
        )

    def test_it_names_the_container_form_of_the_weights(self) -> None:
        """The weights ship as the parts of a `.krm` — one line that tells the reader the form
        before they see the files.

        The note that `I4` / `I2` are a safetensors dialect was retired at the step where the
        distribution stopped being safetensors (ADR 0109 decision 3) — a claim that does not apply
        is not left on the card.
        """
        lines = quants(_manifest()["models"]["zeta"])

        assert any(line.startswith("Weights ship as Karume container files") for line in lines)
        assert not any(line.startswith("A component stored as") for line in lines)


class TestQuantsSection:
    def test_the_default_mark_follows_default_quant(self) -> None:
        """The default mark is derived from `defaultQuant` (not from the first row or the head of
        the table)."""
        lines = quants(_manifest()["models"]["zeta"])

        assert [line for line in lines if "(default)" in line] == [
            "| `w8` (default) | **Half size (int8)** — Both components in int8 storage."
            " | 2.16 KiB (32 B shared; 32 B of assets, read on the host) |"
            " `front` = `i8` / `tables` = `i8` | `linearCompute` = `a8` |"
        ]

    def test_a_seat_without_the_presentation_fields_keeps_an_empty_cell(self) -> None:
        """The presentation fields are optional (ADR 0075 decision 1) — a preset without them is
        read by its id as is."""
        lines = quants(_manifest()["models"]["alpha"])

        assert "| `f16` (default) | — | 512 B | `front` = `f16` | — |" in lines

    def test_a_multi_line_description_stays_in_one_table_row(self) -> None:
        """The hub accepts presentation fields with line breaks (ruling of 2026-10-11) — they are
        folded into one table cell."""
        model = _manifest()["models"]["zeta"]
        model["quants"]["w8"]["description"] = "Both components\n  in int8 storage."

        lines = quants(model)

        assert all("\n" not in line for line in lines)
        assert any(
            line.startswith(
                "| `w8` (default) | **Half size (int8)** — Both components in int8 storage. |"
            )
            for line in lines
        )

    def test_a_pipe_in_the_presentation_fields_does_not_split_the_table_row(self) -> None:
        """The hub also accepts presentation fields with `|` — it is escaped to `\\|`, which does
        not split the column, and spelled in one cell."""
        model = _manifest()["models"]["zeta"]
        model["quants"]["w8"]["label"] = "Half | int8"
        model["quants"]["w8"]["description"] = "front|tables in int8."

        lines = quants(model)

        assert any(
            line.startswith("| `w8` (default) | **Half \\| int8** — front\\|tables in int8. |")
            for line in lines
        )

    @pytest.mark.parametrize("blank", ["", "   ", "\n"])
    def test_a_blank_label_reads_like_a_seat_without_one(self, blank: str) -> None:
        """A field with only whitespace gets the same `—` as a preset without it (no broken emphasis
        such as `** **`)."""
        model = _manifest()["models"]["alpha"]
        model["quants"]["f16"]["label"] = blank

        assert "| `f16` (default) | — | 512 B | `front` = `f16` | — |" in quants(model)

    def test_it_writes_the_abbreviation_legend_only_when_one_is_given(self) -> None:
        """The abbreviation legend is **always on the card** (ADR 0074 decision 4), but not for a
        family without abbreviations.

        In a family whose tokens are the weights names themselves (irodori), a line "`dit` is
        `dit`" would turn the legend itself into a note readers skip.
        """
        model = _manifest()["models"]["zeta"]
        legend = "In a quant name, `tbl` is the `tables` component."

        assert legend in quants(model, abbreviations={"tbl": "tables"})
        assert not any(line.startswith("In a quant name") for line in quants(model))


class TestFromPretrained:
    """The `fromPretrained` call of the usage snippet — **the object ref form** + a place for
    pinning the revision.

    When the source was spelled as a single string there was no place to write the revision at
    all, and readers implicitly followed `main` (other people's apps break on the day the
    repository is re-uploaded).
    """

    def test_it_passes_the_repository_as_an_object_and_keeps_the_options_second(self) -> None:
        lines = from_pretrained(
            "Pipeline", "hdae/karume-anima-turbo", ['  // quant: "f16",'], disposable="using"
        )

        assert lines[0] == "using pipeline = await Pipeline.fromPretrained({"
        assert lines[1] == '  repo: "hdae/karume-anima-turbo",'
        assert lines[-3:] == ["}, {", '  // quant: "f16",', "});"]

    def test_it_offers_the_pin_as_a_commented_out_line(self) -> None:
        """The default stays following `main` — uncommenting alone pins it (not split into two
        samples)."""
        lines = from_pretrained("Pipeline", "hdae/x", [], disposable="using")

        assert '  // revision: "<full commit sha>",' in lines
        assert any("Pin a commit for reproducible builds" in line for line in lines)

    def test_the_disposable_spelling_is_the_callers_fact(self) -> None:
        """Synchronous or asynchronous dispose is a per-pipeline fact (not chosen here)."""
        lines = from_pretrained("Pipeline", "hdae/x", [], disposable="await using")

        assert lines[0].startswith("await using pipeline = await Pipeline")

    def test_the_disposable_spelling_has_no_default(self) -> None:
        """A call that does not name the spelling fails.

        If a default differed from the family's fact, the card's usage would silently become wrong
        code.
        """
        with pytest.raises(TypeError, match="disposable"):
            from_pretrained("Pipeline", "hdae/x", [])  # type: ignore[call-arg]


class TestRequirePipeline:
    def test_it_refuses_a_manifest_that_mixes_another_pipeline(self) -> None:
        """Nothing is drawn if even one model has another contract — no card whose table is right
        but whose description belongs to another model."""
        manifest = _manifest()
        manifest["models"]["alpha"]["pipeline"] = "other/1"

        with pytest.raises(ValueError, match="'alpha' with pipeline 'other/1'"):
            require_pipeline(manifest, "fake/1")

    def test_it_accepts_a_manifest_where_every_model_matches(self) -> None:
        require_pipeline(_manifest(), "fake/1")


class TestRenderDeterminism:
    """The same manifest gives a **byte-identical** card (diff = the assets changed, the premise of
    reassembly)."""

    @staticmethod
    def _card(manifest: dict[str, Any]) -> str:
        sections = [models(manifest), *model_sections(manifest, [quants])]
        return render(sections)

    def test_it_draws_no_file_table(self) -> None:
        """The table with one row per shard is retired (ruling of 2026-09-03) — the per-preset total
        goes into the Download column."""
        card = self._card(_manifest())

        assert "### Files" not in card
        assert "| Key | Dtype | Path | Size | sha256 |" not in card
        assert "| Quant | What it is | Download | Weights | Compute |" in card

    def test_two_renderings_of_an_equal_manifest_agree_byte_for_byte(self) -> None:
        # A separate object rebuilt through JSON (equal, same insertion order) — re-rendering the
        # same dict alone cannot exercise "an order that depends on object identity".
        rebuilt = json.loads(json.dumps(_manifest()))

        assert self._card(_manifest()) == self._card(rebuilt)

    def test_the_body_ends_with_exactly_one_newline(self) -> None:
        card = self._card(_manifest())

        assert card.endswith("\n") and not card.endswith("\n\n")

    def test_the_models_keep_the_manifest_order(self) -> None:
        """Sorting in between would erase the fact "the manifest order" from the card."""
        card = self._card(_manifest())

        assert card.index("## Model: zeta") < card.index("## Model: alpha")


class TestTheSameNameFromAnotherOrigin:
    """The same relative path with a different origin (repo, revision) is a different file (the same
    as the hub's `fileRefKey`).

    Deduplicating by path alone drops the later reference, understates Download and loses one side
    from the note on the borrowed sources — the hub fetches both separately, so only the card
    disagrees with the actual fetch.
    """

    @staticmethod
    def _borrowed(repo: str, revision: str, size: int) -> dict[str, Any]:
        return {
            "repo": repo,
            "revision": revision,
            "path": "source/front.krm",
            "size": size,
            "sha256": "9" * 64,
        }

    @staticmethod
    def _with_back(back: dict[str, Any], front: dict[str, Any] | None = None) -> dict[str, Any]:
        """Adds a second component `back` to alpha and makes the f16 preset select both."""
        manifest = _manifest()
        alpha = manifest["models"]["alpha"]
        if front is not None:
            alpha["weights"]["front"]["f16"]["container"]["parts"] = [front]
        alpha["weights"]["back"] = {"f16": {"container": _container(back)}}
        alpha["quants"]["f16"]["weights"]["back"] = "f16"
        return alpha

    @pytest.mark.parametrize(
        ("front", "back", "total", "repos"),
        [
            # The same path in different repositories: 512 + 256 = 768 B; two borrowed sources.
            (
                ("hdae/source-a", "1" * 40, 512),
                ("hdae/source-b", "1" * 40, 256),
                "768 B",
                ["hdae/source-a", "hdae/source-b"],
            ),
            # Different revisions of the same repository: both are downloaded as separate files.
            (
                ("hdae/source-a", "1" * 40, 512),
                ("hdae/source-a", "2" * 40, 256),
                "768 B",
                ["hdae/source-a", "hdae/source-a"],
            ),
        ],
        ids=["another-repo", "another-revision"],
    )
    def test_borrowed_files_with_the_same_path_are_both_counted(
        self,
        front: tuple[str, str, int],
        back: tuple[str, str, int],
        total: str,
        repos: list[str],
    ) -> None:
        alpha = self._with_back(self._borrowed(*back), front=self._borrowed(*front))
        lines = quants(alpha)

        assert f"| {total} ({total} shared) |" in _row(lines, "f16")
        notes = [line for line in lines if line.startswith("Some components are fetched from")]
        assert [line.split("`")[1] for line in notes] == repos

    def test_an_own_file_and_a_borrowed_file_with_the_same_path_are_both_counted(self) -> None:
        """The own repository's `alpha/front.krm` (512 B) and the borrowed source's path of the same
        name (256 B) are different things."""
        borrowed = {**self._borrowed("hdae/source-a", "1" * 40, 256), "path": "alpha/front.krm"}
        lines = quants(self._with_back(borrowed))

        assert "| 768 B (256 B shared) |" in _row(lines, "f16")

    def test_container_assets_are_counted_per_origin_of_part_0(self) -> None:
        """Container assets (`host_assets`) are also looked up by the identity of part 0 — not by
        path.

        The own repository's `alpha/front.krm` (assets 400 B) and the borrowed source's path of the
        same name (assets 100 B) are different containers. Looking them up by path folds the two
        into one, drops one side's assets, and adds the own container's asset size to a preset that
        selects only the borrowed container.
        """
        borrowed = {**self._borrowed("hdae/source-a", "1" * 40, 256), "path": "alpha/front.krm"}
        alpha = self._with_back(borrowed)
        own = alpha["weights"]["front"]["f16"]["container"]["parts"][0]
        alpha["quants"]["back-only"] = {"weights": {"back": "f16"}, "session": {}}

        lines = quants(alpha, host_assets={file_ref_key(own): 400, file_ref_key(borrowed): 100})

        assert "| 768 B (256 B shared; 500 B of assets, read on the host) |" in _row(lines, "f16")
        assert "| 256 B (256 B shared; 100 B of assets, read on the host) |" in _row(
            lines, "back-only"
        )
