"""**Generic rendering components** of the distribution's model card (`README.md`) — pure
functions derived mechanically from the manifest.

SPEC: docs/spec/distribution-cards.md#the-model-card

HF reads the YAML frontmatter of `README.md` as the model's machine-readable metadata (the last
piece of ADR 0037 §3's "a form that can be uploaded as an HF repository as is"). Licence and
provenance are by decision not written into the manifest (ADR 0038 §6), so that responsibility
belongs to the card.

This module holds only the layer that "builds the frontmatter / lists the models / lists the
quant table / assembles the sections", and **holds not a single per-pipeline template** — what to
explain and what to spell out in the usage section are family-specific facts, so they belong to
`tools/export-recipes/<family>/card.py` (ADR 0065 decisions 1 and 2). The same layering as
`karume.dist`.

NOTE: the rendering components ({@link frontmatter} / {@link introduction} / {@link models} /
{@link quants} / {@link model_sections} / {@link render} / {@link require_pipeline} /
{@link default_model} / {@link knob} / {@link from_pretrained} / {@link hf_link} /
{@link pinned}) are **the public surface for
recipes** (ADR 0065 stage 6). Since `card.py`, outside the wheel, calls them by name, keeping
private names would make "calling across private" the default. Only the components closed within
this module keep a leading `_` ({@link _base_model} and the like).

**Attribution is a separate axis from the template** ({@link sbv2.card.Sbv2CardProfile}). Some
pipelines use the same template while provenance, licence and citation change entirely with the
family whose weights are distributed, so only that part is split into a profile and **the caller
states it explicitly** (the rule for choosing is {@link karume.dist.resolve_card_renderer}).

The manifest (`karume/6` today — made multi-model by ADR 0041, given display fields on quants by
ADR 0075 decision 1, and with the dtype entries of weights turned into **the container's entry
point** by ADR 0109 decision 3) can hold **several models in one repository**, so the card takes
the shape "description of the whole repository → model list → usage → a section per model". The
per-model section is `## Model: <name>`, holding the quant table and (for SBV2) the style and
speaker tables. A single-model repository is drawn in the same shape (for the same reason the
distribution layout is uniform — a card whose structure changes the moment a second model is added
breaks the reader's landmarks too).

There are many files, so **the one-row-per-file table was dropped** (ruling of 2026-09-03 — in a
five-model repository half the card was table, and what the reader wants to know, "how many GiB
does this preset download", was buried in it). The source of truth for per-file `size` / `sha256`
is `karume.json`; the card holds only the per-quant totals (the Download column of
{@link quants}).

MUST: **numbers, download sizes, the quant table, dtype labels, the style table and the speaker
table are all derived from the manifest**, every one of them. Hand-written sizes or dtype names
can drift independently of the assets, and the failure mode "the table disagrees with the files"
would void the very reason the manifest was made a derived artifact (ADR 0038 Context). The only
constants a template may hold are facts **that do not exist in the manifest** (base model, licence,
the provenance of a baked-in LoRA), and this MUST applies as is to a recipe's `card.py`. The
repository ID is not in the manifest either, so it is received as an argument taken from the
assembly target.

MUST: rendering is deterministic — the same manifest gives **byte-identical output**. Nothing that
depends on time, environment or set iteration order is mixed in (that a diff can be read as "the
assets changed" is the premise of reassembly). Dicts from JSON keep insertion order, so `models` /
`quants` / `styles` / `speakers` are emitted in manifest order (re-sorting would erase the fact
"the manifest's order" from the card).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: A function that draws a per-model section (model entry → sequence of lines).
ModelSection = Callable[[Mapping[str, Any]], list[str]]

#: HF's `library_name` (common to every pipeline — the only reader is this repository's runtime).
LIBRARY_NAME = "karume"

#: The HF account distributions go to. The repository name is declared by the pipeline
#: (`karume.dist.resolve_repo` passes `<HF_OWNER>/<declared name>`), so only the owner lives here.
HF_OWNER = "hdae"

#: Number of digits of the commit sha shown in notes (karume.json holds the full value — the note
#: is a guide to the target).
_SHA_DIGITS = 16

_UNITS = (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10))


# SPEC: docs/spec/distribution-cards.md#front-matter
@dataclass(frozen=True)
class CardMetadata:
    """**Facts not in the manifest** that go into the frontmatter (the area ADR 0038 §6 took out
    of the manifest).

    Checked against: https://huggingface.co/docs/hub/model-cards . The vocabulary of
    `base_model_relation` has 4 values, adapter / merge / quantized / finetune, and an f16 / i8
    distribution is "the base weights re-encoded in another storage form", so `quantized` is used.
    merge is tied to the form that **lists two or more base_models on the Hub**, and cannot
    correctly express a provenance not on the Hub (Anima's LoRA). **A distribution that does not
    change the storage form is none of the 4 values**, so the field is left out altogether (same
    source: when unspecified, the Hub infers it) — assigning the closest of the 4 values would give
    the card a claim that is not a fact.

    `base_model` is a sequence so that **every upstream this distribution redistributes is listed**
    (the weights' origin and the redistributed text encoder are separate repositories). Listing
    them keeps the relation `quantized` — the weights were not merged; each one was re-encoded into
    a storage form and they merely live together in one repository.
    """

    pipeline_tag: str
    base_model: tuple[str, ...]
    license: str
    tags: tuple[str, ...]
    #: Written only **when one of the 4 values is factually right** (otherwise left to the Hub's
    #: inference).
    base_model_relation: str | None = None
    #: The name of a custom license: written only with `license: other`, the one case HF reads it
    #: (listing it beside an SPDX identifier reads as "an unnamed custom license").
    license_name: str | None = None
    #: The license's page: written with `license: other`, and beside an SPDX identifier only as the
    #: link the upstream card itself writes (the Gemma license page of Google's Gemma 4 and
    #: EmbeddingGemma 2 cards) — a card does not add a page its upstream does not name.
    license_link: str | None = None


#: **The house non-endorsement sentence**, the same words on every card (maintainer's decision of
#: 2026-11-04): a distribution re-encodes someone else's weights, and a reader landing on it from a
#: search must not take it for the upstream's own release. {@link introduction} fills it in.
#: A sentence an upstream licence requires in its own words (anima's, MOSS-Image's) is kept as it
#: is, beside this one — this sentence does not stand in for a licence clause.
#: SPEC: docs/spec/distribution-cards.md#title-and-the-non-endorsement-sentence
UNOFFICIAL_CONVERSION_NOTICE = (
    "This is an unofficial conversion of {upstream} for Karume."
    " It is not affiliated with or endorsed by {author}."
)


#: **The runtime sentence**, the same words in every card's "What is this" section (maintainer's
#: ruling of 2026-11-27): the runtimes a distribution runs in. Node has no WebGPU of its own, so the
#: sentence says that one is passed in (the runtime's `acquireGpu({ gpu })`). A card writes it as a
#: line of its own, so that the one constant is found whole wherever the paragraph wraps.
#: SPEC: docs/spec/distribution-cards.md#what-is-this
RUNTIME_SENTENCE = (
    "Runs in the browser, in Node (with a WebGPU implementation passed in) and in Deno."
)


#: A full commit of an HF repository (40 lower-case hex digits).
_COMMIT = re.compile(r"[0-9a-f]{40}")


def hf_link(repo: str) -> str:
    """An HF repository as a card links it: the repository ID as plain link text."""
    return f"[{repo}](https://huggingface.co/{repo})"


# SPEC: docs/spec/distribution-cards.md#naming-and-pinning-an-upstream
def pinned(repo: str, revision: str) -> str:
    """An upstream with the commit the stored bytes were taken from:
    `[<owner>/<name>](https://huggingface.co/<owner>/<name>) at commit `<40 hex digits>``.

    MUST: the revision is the full commit — a branch, a tag or a shortened commit does not pin the
    bytes, and a reader copies it as it is.
    """
    if _COMMIT.fullmatch(revision) is None:
        raise ValueError(f"{repo}: {revision!r} is not a full commit (40 lower-case hex digits)")
    return f"{hf_link(repo)} at commit `{revision}`"


#: The clause of the Apache License 2.0 that asks for the notice of the changes (§4(b) — "cause any
#: modified files to carry prominent notices stating that You changed the files"), as a card cites
#: it in its changes bullet ({@link changes_made_here}).
APACHE_MODIFICATION_CLAUSE = "Apache 2.0 §4(b)"

#: The sentence every `NOTICE.md`'s closing paragraph opens with.
NOTICE_CLOSING = "No retraining and no fine-tuning were performed."


# SPEC: docs/spec/distribution-cards.md#noticemd
def notice_closing(not_distributed: str = "The original checkpoint is") -> str:
    """The closing paragraph of `NOTICE.md`: {@link NOTICE_CLOSING}, then what of the upstream
    is not distributed here (`not_distributed` is the subject and its verb: "The original
    checkpoint is", "The original checkpoints are")."""
    return f"{NOTICE_CLOSING} {not_distributed} not distributed here."


#: Where the Apache License 2.0 text lives (the text `LICENSE.md` copies — Apache 2.0 §4(a)).
APACHE_LICENSE_TEXT_URL = "https://www.apache.org/licenses/LICENSE-2.0"

#: What follows "licensed **apache-2.0**" on a card's line for weights whose license is the
#: primary text of `LICENSE.md`: when the license was read, where the text lives, and the copy.
APACHE_LICENSE_REFERENCE = (
    f"(as of retrieval; [full text]({APACHE_LICENSE_TEXT_URL}) — a verbatim copy is in"
    " `LICENSE.md`)"
)


# SPEC: docs/spec/distribution-cards.md#license-statements
def changes_made_here(clause: str | None = None) -> str:
    """The head of every card's changes bullet, which sums the changes up and points at the
    notice that lists them in full: `- **Changes made here** (listed in full in `NOTICE.md`, per
    <clause>):` — `clause` is the clause of the license that asks for the notice of the changes,
    where it has one ({@link APACHE_MODIFICATION_CLAUSE})."""
    cited = "" if clause is None else f", per {clause}"
    return f"- **Changes made here** (listed in full in `NOTICE.md`{cited}):"


# SPEC: docs/spec/distribution-cards.md#what-the-upstream-says
def upstream_cards(links: Sequence[str]) -> list[str]:
    """The bullet that ends every card's attribution section (maintainer's ruling of 2026-11-27):
    what the model can and cannot do is the upstream's to say, so the card sends the reader to
    the card of every upstream it redistributes instead of restating it. `links` are Markdown
    links to those cards (the `base_model` repositories, and the pages of works not on the Hub),
    each listed once, in order.

    MUST: no upstream is left out — a card that names one of its upstreams' cards only would read
    as if that card covered the others.
    """
    if not links:
        raise ValueError("no upstream card to send the reader to")
    cards = "the upstream model card" if len(links) == 1 else "the upstream model cards"
    return [
        "- **What the model can and cannot do** — capabilities, limitations, intended use,"
        f" evaluations and safety guidance — is documented on {cards}, and this repository does"
        f" not restate it: read {_spell_list(links)} before using it."
    ]


@dataclass(frozen=True)
class UpstreamQuote:
    """A statement of an upstream's model card, quoted verbatim (maintainer's ruling of
    2026-11-27): what an upstream states and the recipe cannot measure — a legal statement, its
    training data, its intended use — is quoted with its source, never reworded.

    `paragraphs` are the card's own words at `revision`, each one paragraph or list item of it with
    its line breaks folded into spaces; `sections` are the headings they stand under, the card's
    title included, spelled without their emphasis and leading emoji (empty when they precede the
    first heading).
    """

    repo: str
    revision: str
    sections: tuple[str, ...]
    paragraphs: tuple[str, ...]

    @property
    def source(self) -> str:
        """The quoted card at its commit, as a plain-text link."""
        if _COMMIT.fullmatch(self.revision) is None:
            raise ValueError(f"{self.repo}: {self.revision!r} is not a full commit")
        return f"[{self.repo}](https://huggingface.co/{self.repo}/blob/{self.revision}/README.md)"


# SPEC: docs/spec/distribution-cards.md#what-the-upstream-says
def quotation(quote: UpstreamQuote, lead: str, *, indent: str = "") -> list[str]:
    """`{lead} ({source}) states, under "<section>":`, a blank line and the paragraphs as one
    blockquote, its lines indented by `indent` (a list item's continuation).

    `lead` names the card ("the upstream model card", "Its card"); the source is the card's
    README at the commit quoted.
    """
    if not quote.paragraphs:
        raise ValueError(f"{quote.repo}: a quotation without a paragraph")
    under = (
        ""
        if not quote.sections
        else ", under " + _spell_list([f'"{section}"' for section in quote.sections])
    )
    lines = [f"{lead} ({quote.source}) states{under}:", ""]
    for index, paragraph in enumerate(quote.paragraphs):
        if index:
            lines.append(f"{indent}>")
        lines.append(f"{indent}> {paragraph}")
    return lines


@dataclass(frozen=True)
class Upstream:
    """One upstream work a card names in {@link UNOFFICIAL_CONVERSION_NOTICE}: how it is spelled
    (an HF repository ID, or a title for a work that is not on the Hub) and who published it."""

    name: str
    author: str


def _spell_list(items: Sequence[str]) -> str:
    """`A` / `A and B` / `A, B and C` (each item once, in order of appearance)."""
    unique = list(dict.fromkeys(items))
    if len(unique) == 1:
        return unique[0]
    return f"{', '.join(unique[:-1])} and {unique[-1]}"


def introduction(title: str, upstreams: Sequence[Upstream]) -> list[str]:
    """The card's title and, as the first paragraph under it, the house sentence
    ({@link UNOFFICIAL_CONVERSION_NOTICE}) naming every upstream and every distinct author.

    A distribution of several upstreams gets one sentence, the upstreams and the authors each
    listed once ("an unofficial conversion of A and B … endorsed by X and Y").

    MUST: an upstream without a name or an author fails the render — a card that cannot say whose
    work it converts is not drawn (a placeholder would claim non-endorsement by no one).
    """
    if not upstreams:
        raise ValueError(f"card '{title}': no upstream declared — the card cannot say whose work")
    for upstream in upstreams:
        if not upstream.name.strip() or not upstream.author.strip():
            raise ValueError(f"card '{title}': upstream {upstream!r} has no name or no author")
    notice = UNOFFICIAL_CONVERSION_NOTICE.format(
        upstream=_spell_list([upstream.name for upstream in upstreams]),
        author=_spell_list([upstream.author for upstream in upstreams]),
    )
    # An author spelled with its own full stop ("Example Inc.") ends the sentence already.
    return ["", f"# {title}", "", notice.removesuffix(".") if notice.endswith("..") else notice, ""]


def _download_size(size: int) -> str:
    """Spell a download size with 3 significant digits (`3.24 GiB` / `248 MiB` — the precision
    stays the same across units).

    Raw bytes are not shown alongside. What the reader wants to decide here is one thing, "is this
    realistic for this connection and this disk", and the exact per-file values are in
    `karume.json` (the table's notes point there).

    MUST: the unit is chosen again from the value **after rounding**. Choosing before rounding
    gives, in the carry band (from 1023.5 MiB up to below 1 GiB, for instance), `1024 MiB`, a
    spelling that is neither 3 significant digits nor the right unit. The carry threshold is not
    1000 but **exactly 1 of the next unit** (= 1024) — `1000 MiB` is a correct MiB spelling, and
    moving it up would give `0.977 GiB`, "below 1 unit".
    """
    for index, (unit, scale) in enumerate(_UNITS):
        if size < scale:
            continue
        spelled = _significant(size / scale)
        if float(spelled) >= 1024 and index > 0:
            unit, scale = _UNITS[index - 1]
            spelled = _significant(size / scale)
        return f"{spelled} {unit}"
    return f"{size:,} B"


def _significant(value: float) -> str:
    """Spelling with 3 significant digits (no decimals once the integer part exceeds 3 digits)."""
    return f"{value:.{max(0, 3 - len(str(int(value))))}f}"


#: Identity of a file reference (the pair of the source `repo` / `revision` and `path` — for our
#: own repository the source is `None`).
FileRefKey = tuple[str | None, str | None, str]


def file_ref_key(ref: Mapping[str, Any]) -> FileRefKey:
    """The same identity as the hub's `fileRefKey` (the pair of source and path).

    MUST: not made unique by path alone — the hub fetches same-named paths in another repository,
    in another revision of the same repository, or our own versus borrowed, as separate files
    (ADR 0038 §7, a superseded record; the hub's `fileRefKey` in `packages/hub/src/manifest.ts` is
    the source of truth). Folding by path would drop whichever came later, understate the download
    size, and lose one of them from the description of the borrowed source.
    """
    return (ref.get("repo"), ref.get("revision"), ref["path"])


def _is_shared(ref: Mapping[str, Any]) -> bool:
    """Whether the reference is not downloaded again for a second model (`shared/` in the same
    repository, or a cross-repository reference)."""
    return ref["path"].startswith("shared/") or "repo" in ref


def _collect(refs: dict[FileRefKey, Mapping[str, Any]], weight_files: Mapping[str, Any]) -> None:
    """Add the files of one dtype entry (the container's parts), **made unique by identity**
    ({@link file_ref_key}).

    Uniqueness keeps bytes from being counted twice when two components point at the same
    container (the same shape as an entry folded into `shared/`). The `extras` of `karume/4` were
    retired, and their content became container assets (= inside the parts, so this is the only
    place to add them — ADR 0109 decision 4).
    """
    for ref in weight_files["container"]["parts"]:
        refs.setdefault(file_ref_key(ref), ref)


def _quant_refs(model: Mapping[str, Any], quant: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The file references a reader who chose the quant actually downloads.

    For weights, only the files of the dtype the quant chose. assets are unconditional files that
    do not depend on the quant choice (ADR 0041 decision 3), so all of them are included.
    """
    refs: dict[FileRefKey, Mapping[str, Any]] = {}
    for name, label in quant["weights"].items():
        _collect(refs, model["weights"][name][label])
    for ref in model["assets"].values():
        refs.setdefault(file_ref_key(ref), ref)
    return list(refs.values())


def _model_refs(model: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every file reference the model carries (the union across dtypes + assets).

    The notes appear once per quant table, so their conditions are checked **per model** too (if
    a note appeared or vanished depending on the chosen quant, the description of the same model
    would change per preset).
    """
    refs: dict[FileRefKey, Mapping[str, Any]] = {}
    for entry in model["weights"].values():
        for weight_files in entry.values():
            _collect(refs, weight_files)
    for ref in model["assets"].values():
        refs.setdefault(file_ref_key(ref), ref)
    return list(refs.values())


def _borrowed(refs: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """Targets of cross-repository references (`(repo, revision)` without duplicates, in order of
    appearance — ADR 0038 §7)."""
    return list(dict.fromkeys((ref["repo"], ref["revision"]) for ref in refs if "repo" in ref))


def require_pipeline(manifest: Mapping[str, Any], supported: str) -> None:
    """Check that every model has the pipeline contract the body assumes (otherwise nothing is
    drawn).

    Templates are pipeline-specific, so feeding a manifest of another contract would silently
    produce a card whose "tables are right but whose description is about another model". A family
    repository may mix architectures (ADR 0041 decision 2), so a single mismatch means no card.
    """
    for name, model in manifest["models"].items():
        if model["pipeline"] != supported:
            raise ValueError(
                f"The model card body is specific to {supported} — "
                f"cannot write a card for model '{name}' with pipeline '{model['pipeline']}'"
            )


def _base_model(base_models: Sequence[str]) -> list[str]:
    """`base_model` (a scalar for one, a YAML sequence for several — HF reads both)."""
    if len(base_models) == 1:
        return [f"base_model: {base_models[0]}"]
    return ["base_model:", *(f"  - {model}" for model in base_models)]


# SPEC: docs/spec/distribution-cards.md#front-matter
def frontmatter(metadata: CardMetadata) -> list[str]:
    return [
        "---",
        f"library_name: {LIBRARY_NAME}",
        f"pipeline_tag: {metadata.pipeline_tag}",
        *_base_model(metadata.base_model),
        *(
            []
            if metadata.base_model_relation is None
            else [f"base_model_relation: {metadata.base_model_relation}"]
        ),
        f"license: {metadata.license}",
        *([f"license_name: {metadata.license_name}"] if metadata.license_name is not None else []),
        *([f"license_link: {metadata.license_link}"] if metadata.license_link is not None else []),
        "tags:",
        *(f"  - {tag}" for tag in metadata.tags),
        "---",
    ]


# SPEC: docs/spec/distribution-cards.md#presets-and-the-quant-tables
def models(manifest: Mapping[str, Any]) -> list[str]:
    """The list of models the repository carries (the axis that first became machine-readable in
    v2 — ADR 0041 decision 2)."""
    default = manifest["defaultModel"]
    lines = [
        "## Models",
        "",
        "| Model | Pipeline | Quants | Default quant |",
        "| ----- | -------- | ------ | ------------- |",
    ]
    for name, model in manifest["models"].items():
        mark = " (default)" if name == default else ""
        quants = " / ".join(f"`{quant}`" for quant in model["quants"])
        lines.append(
            f"| `{name}`{mark} | `{model['pipeline']}` | {quants} | `{model['defaultQuant']}` |"
        )
    lines += [
        "",
        f"`model` selects one of these; omitted, it is `{default}`."
        " `quant` defaults to that model's own default quant.",
    ]
    return lines


def _session(quant: Mapping[str, Any]) -> str:
    """Spell the execution settings a quant declares in one cell.

    MUST: values are written **in JSON notation**. `render_card` receives the in-memory dict
    before it is written as JSON, so filling it in raw would print booleans as Python's `True` /
    `False`, a spelling that disagrees with both the manifest (`true`) and the TypeScript users
    paste as is (a cell meant as `false` reads as truthy), and objects (`numerics`) as Python's
    `{'level': ...}`. Only strings are written without the double quotes of `json.dumps` — the
    option values are enums such as `a8`, matching the table's other spellings (`` `i4` `` and the
    like).
    """
    session = quant["session"]
    features = quant.get("gpuFeatures", {})
    parts = [
        f"`{key}` = `{value if isinstance(value, str) else json.dumps(value)}`"
        for key, value in session.items()
    ]
    parts += [f"requires `{key}`" for key, value in features.items() if value]
    return " / ".join(parts) if parts else "—"


def _presentation(quant: Mapping[str, Any]) -> str:
    """Spell a quant's display fields (`label` / `description` of ADR 0075 decision 1) in one
    cell.

    Both are optional, so there are 4 cases. An unwritten field lets the reader read the id as is
    (`—`) — repeating the id here would only print the same string as the table's first column
    twice.

    The accepted set is defined by the hub's parser, and empty strings, whitespace-only values and
    values with newlines pass too (ruling of 2026-10-11). A newline would split the table row, so
    runs of whitespace (newlines included) are folded into one space and spelled in one cell. A
    field with only whitespace is treated like an unwritten one. `|` is in the accepted set too and
    would split the table's columns as is, so it is escaped as `\\|` (the form that does not
    separate columns in a GFM table).
    """
    label = " ".join(quant.get("label", "").split()).replace("|", "\\|")
    description = " ".join(quant.get("description", "").split()).replace("|", "\\|")
    if label and description:
        return f"**{label}** — {description}"
    if label:
        return f"**{label}**"
    return description or "—"


def _download(
    model: Mapping[str, Any], quant: Mapping[str, Any], host_assets: Mapping[FileRefKey, int]
) -> str:
    """The Download column of the quant table — the total downloaded when this preset is chosen,
    and the difference between the total and what is resident on the GPU.

    There are two notes. `shared` is the amount "not downloaded for a second model"
    ({@link _is_shared}), and assets are the part **only read on the host, never uploaded to the
    GPU** (some distributions, such as gemma4 with its PLE, are mostly assets by Download). The
    latter is added only when its share is not negligible — writing it on every card would only
    draw the reader's eye to a tokenizer of a few MiB.

    MUST: the assets are counted as the sum of **the separate file fields** (`assets` of the
    manifest) and **the assets the container declares** (`host_assets` — the PLE / `rope_base`
    that moved into the container with ADR 0109 decision 4). Counting only the former would
    silently drop just the breakdown note for distributions that folded their assets into the
    container.
    """
    refs = _quant_refs(model, quant)
    asset_keys = {file_ref_key(ref) for ref in model["assets"].values()}
    total = sum(ref["size"] for ref in refs)
    shared = sum(ref["size"] for ref in refs if _is_shared(ref))
    assets = sum(ref["size"] for ref in refs if file_ref_key(ref) in asset_keys)
    assets += sum(host_assets.get(key, 0) for key in _container_heads(model, quant))
    notes = []
    if shared:
        notes.append(f"{_download_size(shared)} shared")
    if assets and assets * 100 >= total:
        notes.append(f"{_download_size(assets)} of assets, read on the host")
    return _download_size(total) + (f" ({'; '.join(notes)})" if notes else "")


def _container_heads(model: Mapping[str, Any], quant: Mapping[str, Any]) -> set[FileRefKey]:
    """**The identity of part 0** of the containers the quant chooses (the keys of `host_assets`
    — the same container once).

    MUST: the key is {@link file_ref_key}, not the path — when part 0 of our own container and of
    a borrowed container (another repository / another revision) have the same relative path,
    keying by path would fold them into one and either drop one side's asset bytes or add our own
    asset bytes to the borrowed container.
    """
    return {
        file_ref_key(model["weights"][name][label]["container"]["parts"][0])
        for name, label in quant["weights"].items()
    }


@dataclass(frozen=True)
class Measurement:
    """What one preset measured: its speed, or the quality heard or seen, with the device and the
    date (maintainer's ruling of 2026-11-27). A preset's description says what the preset is; a
    measurement is a note under the quant table, because another device or a later build can
    measure otherwise.

    `statement` is the result as one clause, without a final period. `device` is the GPU (with the
    runtime where it was recorded), or what ran the measurement instead (a simulation). `date` is
    the day of the measurement, `YYYY-MM-DD`. `subject` names what was measured when it was not
    this model (another voice of the same architecture), else `None`.
    """

    preset: str
    statement: str
    device: str
    date: str
    subject: str | None = None


#: The line that opens the measurement notes under a quant table.
MEASUREMENTS_LEAD = (
    "Measured with these presets, each on the device and at the date named (another device or a"
    " later build may measure otherwise):"
)

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


# SPEC: docs/spec/distribution-cards.md#presets-and-the-quant-tables
def measurements(model: Mapping[str, Any], measured: Sequence[Measurement]) -> list[str]:
    """The measurement notes of one model: a blank line, {@link MEASUREMENTS_LEAD}, a blank line and
    one bullet per measurement, in the order given; nothing when none applies.

    A measurement of a preset the model does not ship is left out: a note on a preset the
    repository does not carry would describe what it cannot ship. The families' tests check that
    each table names presets of the family.

    MUST: a measurement without a device or a date fails — a note that names neither is the very
    claim this format replaces.
    """
    lines: list[str] = []
    for measurement in measured:
        if measurement.preset not in model["quants"]:
            continue
        if not measurement.device.strip() or _DATE.fullmatch(measurement.date) is None:
            raise ValueError(
                f"the measurement of `{measurement.preset}` names no device or no YYYY-MM-DD date"
            )
        on = "" if measurement.subject is None else f"on {measurement.subject}; "
        lines.append(
            f"- `{measurement.preset}` — {measurement.statement}"
            f" ({on}{measurement.device}, {measurement.date})."
        )
    return [] if not lines else ["", MEASUREMENTS_LEAD, "", *lines]


def model_measurements(
    manifest: Mapping[str, Any], table: Mapping[str, Sequence[Measurement]]
) -> Sequence[Measurement]:
    """The measurements of the manifest's model from a recipe's `table` (model name → what its
    presets measured), for `quants(measured=…)`; none when the table does not name the model.

    MUST: a manifest of several models fails when the table names one of them — `quants` is given
    one model's notes, and passing them to every model's table would state one model's
    measurements of the others.
    """
    named = [name for name in manifest["models"] if name in table]
    if named and len(manifest["models"]) != 1:
        raise ValueError(
            f"measurements of {named} in a manifest of several models {list(manifest['models'])}"
        )
    return table[named[0]] if named else ()


# SPEC: docs/spec/distribution-cards.md#presets-and-the-quant-tables
def quants(
    model: Mapping[str, Any],
    *,
    abbreviations: Mapping[str, str] | None = None,
    host_assets: Mapping[FileRefKey, int] = {},
    measured: Sequence[Measurement] = (),
) -> list[str]:
    """The quant table (ADR 0074 preset names + ADR 0075 display fields + the download size per
    preset).

    `abbreviations` maps the component-override token of a preset name (`bert` in `i8+bert4`) →
    its weights name. **Abbreviations are set by the recipe, so the card always shows the mapping**
    (ADR 0074 decision 4) — this one line is all that connects the table's first column with the
    spellings of the Weights column; without it, which component `bert4` is about cannot be read.
    Families that use no abbreviations do not pass it (when the token is the weights name itself,
    no mapping is needed).

    `host_assets` (part 0 identity {@link file_ref_key} → the sum of the logical lengths of the
    assets the container declares) is **a fact of the container** that assembly takes from
    {@link karume.dist.container_asset_bytes} and passes in; it cannot be derived from the
    manifest. For callers that do not pass it (tests that only rebuild the card), the breakdown
    note does not count the container's assets.

    The notes after the table are what carried over from the dropped file table, **only when they
    apply** (where the source of truth for sha256 is, the dtype vocabulary, `shared/`,
    cross-repository references). Putting them on a distribution they do not apply to would be a
    claim that is not a fact, so the conditions are read from the manifest.

    `measured` is what the recipe measured of this model's presets ({@link measurements}), written
    after those notes.
    """
    default = model["defaultQuant"]
    lines = [
        "### Quants",
        "",
        "| Quant | What it is | Download | Weights | Compute |",
        "| ----- | ---------- | -------- | ---- | ---- |",
    ]
    for name, quant in model["quants"].items():
        weights = " / ".join(
            f"`{weight}` = `{label}`" for weight, label in quant["weights"].items()
        )
        mark = " (default)" if name == default else ""
        lines.append(
            f"| `{name}`{mark} | {_presentation(quant)} |"
            f" {_download(model, quant, host_assets)} |"
            f" {weights} | {_session(quant)} |"
        )
    lines += [
        "",
        f"If no quant is given, it runs as `{default}` (this model's recommended default).",
    ]
    if abbreviations:
        lines.append(
            "In a quant name, "
            + " / ".join(
                f"`{token}` is the `{name}` component" for token, name in abbreviations.items()
            )
            + "."
        )
    refs = _model_refs(model)
    lines += [
        "Per-file `size` and `sha256` live in `karume.json` — verify against that at the fetch"
        " layer.",
        "Dtype labels use the runtime's **storage dtype vocabulary** (`f16` / `bf16` / `i8` /"
        " `i4` / `i2`), not the `fp16` spelling common elsewhere in the ecosystem.",
        "Weights ship as Karume container files (`.krm`), split into numbered parts;"
        " a part is fetched and verified on its own.",
    ]
    if any(ref["path"].startswith("shared/") for ref in refs):
        lines.append(
            "A path under `shared/` is one this model shares byte for byte with another model in"
            " this repository (it is fetched and cached once)."
        )
    # SPEC: docs/spec/distribution-cards.md#naming-and-pinning-an-upstream
    for repo, revision in _borrowed(refs):
        lines.append(
            f"Some components are fetched from {hf_link(repo)} at commit"
            f" `{revision[:_SHA_DIGITS]}…` — those bytes are identical to this model's own, so they"
            " are not stored here a second time."
        )
    return [*lines, *measurements(model, measured)]


#: **The steps to pin a revision**, added to the source side of `fromPretrained` (one set shared
#: by every family).
#:
#: The default tracks `main`, which "just works", so the sample is written with the default. But
#: the distribution repositories get re-uploaded (renamed files, a bumped manifest format), so a
#: reader who did not pin breaks one day without warning — the pin is placed, commented out,
#: **right next to** the default line, so removing the comment is all it takes (splitting the
#: sample in two would make it unclear which one is recommended).
#: SPEC: docs/spec/distribution-cards.md#usage
REVISION_PIN_COMMENT: tuple[str, ...] = (
    "  // Pin a commit for reproducible builds — without it you track `main`, and a future",
    "  // repo update (renamed files, new manifest format) may break your app.",
    '  // Copy the full hash from this repo\'s "Files and versions" tab:',
    '  // revision: "<full commit sha>",',
)


def from_pretrained(
    pipeline_class: str,
    repo: str,
    options: Sequence[str],
    *,
    disposable: str,
) -> list[str]:
    """One `fromPretrained` call of the usage snippet (**object ref form** + the pin slot).

    The source is spelled as `{ repo, revision? }` rather than a single string because there is
    no other place to write the revision ({@link REVISION_PIN_COMMENT}). `options` is the sequence
    of lines that go into the second argument's object as is (family-specific knobs — passed
    already indented by 2 spaces).

    `disposable` is the declaration spelling (`using` / `await using`) — whether the pipeline has
    a synchronous or an asynchronous dispose is a fact of the family, so it is not chosen here (no
    default — a default spelling that disagrees with the family's fact would turn the card's usage
    into code with a type error as is).
    """
    return [
        f"{disposable} pipeline = await {pipeline_class}.fromPretrained({{",
        f'  repo: "{repo}",',
        *REVISION_PIN_COMMENT,
        "}, {",
        *options,
        "});",
    ]


def default_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """The default model's entry (the representative when introducing the repository as one — also
    the default of the usage)."""
    return manifest["models"][manifest["defaultModel"]]


def knob(value: Any) -> str:
    """Spell one value of `defaults` (only strings in code font — so numbers and names can be told
    apart)."""
    return f"`{value}`" if isinstance(value, str) else f"{value}"


def render(sections: Sequence[Sequence[str]]) -> str:
    """Join the sections into one body (with a trailing newline)."""
    return "\n".join(line for section in sections for line in section) + "\n"


# SPEC: docs/spec/distribution-cards.md#sections-and-their-order
def model_sections(
    manifest: Mapping[str, Any],
    per_model: Sequence[ModelSection],
) -> list[Sequence[str]]:
    """Assemble the per-model sections (`## Model: <name>`) in manifest order.

    `per_model` is the list of renderers that "take a model entry and return a section (a
    sequence of lines)" — the only axis for inserting pipeline-specific sections (SBV2's style
    table and the like) here.
    """
    sections: list[Sequence[str]] = []
    for name, model in manifest["models"].items():
        sections.append([""])
        sections.append([f"## Model: {name}", ""])
        for index, render_section in enumerate(per_model):
            if index:
                sections.append([""])
            sections.append(render_section(model))
    return sections
