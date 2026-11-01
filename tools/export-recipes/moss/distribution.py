"""The MOSS-Image distribution recipe — which series files go to which paths of `karume-moss-image`,
under which storage labels, the presets, the default preset and `pipelineConfig` (ADR 0138
addendum of 2026-11-01).

The generic assembly (placement, sha256, the manifest, staging and swap, verification) is
`karume.dist`'s. This module holds the MOSS-Image facts only:

- **Seven components, all stored here** (no cross-repository reference): the DiT graphs `dit`,
  `dit_skip22` (no weight bytes — every initializer borrows `dit`'s, ADR 0138 decision 4) and
  `dit_text` from the i8 series `moss-image-sft-i8-dyn`; the three VAE decoder graphs
  `vae_decoder_pre` / `vae_decoder_attn` / `vae_decoder_post` from the f32 series
  `moss-image-sft-f32-dyn` (ADR 0138 decision 8); the Qwen3.5 text tower `text_encoder` from
  `qwen3.5-9b-text-moss-image-sft-i8-dyn` (ADR 0137). The text tower is MOSS-Image-SFT's own
  `text_encoder` folder, so it lives in this repository rather than in a Qwen3.5 repository.
- **One model asset**: the Qwen3.5 tokenizer `tokenizer` (`qwen3_5.tokenizer`'s
  `karume-qwen3_5-tokenizer/1`). The DiT's host tables (`rope_base`, `training_grid`) and the
  VAE's BatchNorm statistics (`latent_batch_norm`) are assets of their containers, so the
  manifest does not list them (ADR 0109 decision 4). The precomputed text embeddings of the fixed
  prompts (`moss.text_embeds`) are not distributed: the public pipeline takes prompt strings only.
- **Two presets on the same weights** ({@link MOSS_QUANTS}): the reference `f32+dit8` and the
  practical `f32+dit8-a8-attn8-online`, the default ({@link MOSS_DEFAULT_QUANT}). Every weights
  entry has one storage label, so the presets name none and `complete_quant_weights` fills them
  in: the presets differ only in `session`.

The repository is one per family (ADR 0092 decision 1) and the model is `sft`, the upstream
checkpoint `MOSS-Image-SFT` without the family name (as `ti2v-5b` is Wan2.2's).

Not published: no Hugging Face repository exists, and `packages/models/src/moss/` has no `SOURCES`
entry (an entry before publication can only 404 — ADR 0073 decision 1). The local mirror is
`models/karume-moss-image/` (`uv run python dist.py --pipeline moss-image`).

MUST: this module does not import torch (`import dist` must not load it —
`tests/test_dist_driver.py` `TestImportingTheDriver`). The writers (`moss.export_dit`,
`moss.export_vae`, `qwen3_5.export`) import torch, so the spellings they write (series, graphs,
assets) are repeated here and `moss/tests/test_distribution.py` checks them against the writers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _shared.container_read import (
    ContainerReadError,
    open_container,
    read_asset_declarations,
)
from _shared.licenses import apache_license_2_0
from _shared.upstream import assert_upstream_provenance
from karume.dist import (
    Artifact,
    DistError,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assert_component_present,
    assert_model_name,
    assert_storage,
    assert_storage_absent,
    complete_quant_weights,
    component_parts,
    graph_inputs,
    ir_graph,
)
from moss.card import MOSS_SUPPORTED_PIPELINE, render_moss_model_card
from moss.sources import SOURCE

#: The pipeline contract (ADR 0041 §2). The TS reader accepts `MOSS_PIPELINE_NAME` /
#: `MOSS_PIPELINE_MAJOR` (`packages/models/src/moss/config.ts`).
MOSS_PIPELINE = MOSS_SUPPORTED_PIPELINE

#: The distribution repository (ADR 0092 decision 2 — `karume-<family>`).
MOSS_REPO_NAME = "karume-moss-image"

#: The one model (module doc).
MOSS_MODEL = "sft"

#: The series this distribution reads (docs/assets-layout.md). The writers' spellings:
#: `moss.export_dit.SERIES_NAME`, `moss.export_vae.SERIES_NAME`, `qwen3_5.export.series_dir`
#: (`qwen3.5-9b-text-<source>-<storage>-dyn`) and `qwen3_5.tokenizer.SERIES_NAME` / `ASSET_FILE`.
MOSS_DIT_SERIES = "moss-image-sft-i8-dyn"
MOSS_VAE_SERIES = "moss-image-sft-f32-dyn"
MOSS_TEXT_SERIES = "qwen3.5-9b-text-moss-image-sft-i8-dyn"
MOSS_TOKENIZER_SERIES = "qwen3_5-tokenizer"
MOSS_TOKENIZER_FILE = "tokenizer.json"

#: The representative name of a container in a series part directory.
MOSS_MODEL_FILE = "model.krm"

#: The components = the manifest's weights keys = the containers' graph names (container-v1 §2.1;
#: `tests/test_graph_names.py`). The series part directories carry the same spellings. The order
#: is the generation's stage order (ADR 0138 decision 5).
MOSS_TEXT_ENCODER_ROLE = "text_encoder"
MOSS_DIT_TEXT_ROLE = "dit_text"
MOSS_DIT_ROLE = "dit"
MOSS_DIT_SKIP22_ROLE = "dit_skip22"
MOSS_VAE_PRE_ROLE = "vae_decoder_pre"
MOSS_VAE_ATTN_ROLE = "vae_decoder_attn"
MOSS_VAE_POST_ROLE = "vae_decoder_post"
MOSS_DIT_ROLES: tuple[str, ...] = (MOSS_DIT_TEXT_ROLE, MOSS_DIT_ROLE, MOSS_DIT_SKIP22_ROLE)
MOSS_VAE_ROLES: tuple[str, ...] = (MOSS_VAE_PRE_ROLE, MOSS_VAE_ATTN_ROLE, MOSS_VAE_POST_ROLE)
MOSS_CONTAINER_ROLES: tuple[str, ...] = (MOSS_TEXT_ENCODER_ROLE, *MOSS_DIT_ROLES, *MOSS_VAE_ROLES)

#: The model asset (the manifest's `assets` key — the TS reader is `parseQwen3_5TokenizerAsset`).
MOSS_TOKENIZER_ROLE = "tokenizer"

#: What the tokenizer asset must say (the writer is `qwen3_5.tokenizer`): its format, the upstream
#: folder it was built from, and the token bound — upstream's `max_sequence_length`, which is also
#: the text encoder's and the DiT's bound (ADR 0137's consequence: the three bounds are one).
MOSS_TOKENIZER_FORMAT = "karume-qwen3_5-tokenizer/1"
MOSS_TOKENIZER_SUBFOLDER = "tokenizer"
MOSS_MAX_TEXT_TOKENS = 4096

#: The width of one text-encoder row, which `dit_text` reads (`text_embeds [1, T, 4096]`).
MOSS_TEXT_WIDTH = 4096

#: The graph input names the width check reads (writers: `moss.dit_patch.DIT_TEXT_INPUTS`; the
#: text tower's output is its only output).
MOSS_DIT_TEXT_INPUT = "text_embeds"

#: The container assets each role must declare, with their roles (writers: `moss.export_dit`
#: `ROPE_BASE_ASSET` / `TRAINING_GRID_ASSET`, `moss.export_vae.BATCH_NORM_ASSET`; readers:
#: `packages/models/src/moss/dit-contract.ts` and `vae-contract.ts`).
MOSS_CONTAINER_ASSETS: Mapping[str, Mapping[str, str]] = {
    MOSS_DIT_ROLE: {"rope_base": "rope-base", "training_grid": "training-grid"},
    MOSS_VAE_PRE_ROLE: {"latent_batch_norm": "latent-batch-norm"},
}

#: The storage label of each container role (the label is also in the output path, so a second
#: storage added one day does not move an existing path — the Wan / Depth Anything spelling).
#: `dit_skip22` holds no weight bytes; it runs on `dit`'s i8 weights, so it carries their label.
MOSS_STORAGE_LABELS: Mapping[str, str] = {
    MOSS_TEXT_ENCODER_ROLE: "i8",
    MOSS_DIT_TEXT_ROLE: "i8",
    MOSS_DIT_ROLE: "i8",
    MOSS_DIT_SKIP22_ROLE: "i8",
    **dict.fromkeys(MOSS_VAE_ROLES, "f32"),
}

#: The roles whose storage is checked on another container: the borrower has no binding table (a
#: shared declaration supplies nothing), so its storage is its lender's ({@link
#: assert_borrows_every_weight} checks that it borrows every initializer).
MOSS_LENDERS: Mapping[str, str] = {MOSS_DIT_SKIP22_ROLE: MOSS_DIT_ROLE}

#: The output paths inside the model subtree (one place shared by the placements and the manifest).
MOSS_OUTPUT_PATHS: Mapping[str, str] = {
    **{role: f"{role}/model.{label}.krm" for role, label in MOSS_STORAGE_LABELS.items()},
    MOSS_TOKENIZER_ROLE: f"{MOSS_TOKENIZER_ROLE}/{MOSS_TOKENIZER_FILE}",
}

#: Every compressed layout of the codec ledger (`karume.container.CODEC_LEDGER` without f32 / i32 —
#: the test checks the ledger).
_COMPRESSED_LAYOUTS = ("bf16", "f16", "i8", "i4", "i2")

#: The storage each container's binding table must hold (the presence check — a plain f32 container
#: in an i8 seat passed assembly, load and run up to a reference check in the precedent's measured
#: accident) and must not hold (every other compressed layout — a mixed i4 series also holds i8).
#: MUST: the forbidden set names every compressed layout but the required one.
MOSS_STORAGE_REQUIREMENTS: Mapping[str, str] = {
    role: label for role, label in MOSS_STORAGE_LABELS.items() if role not in MOSS_LENDERS
}
MOSS_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    role: tuple(layout for layout in _COMPRESSED_LAYOUTS if layout != required)
    for role, required in MOSS_STORAGE_REQUIREMENTS.items()
}

#: The weights declaration (graph name → storage label → placement role). One label per entry, so
#: the presets need not name any.
MOSS_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    role: {label: WeightFiles(role)} for role, label in MOSS_STORAGE_LABELS.items()
}

#: The model assets (independent of the preset).
MOSS_ASSETS: Mapping[str, str] = {MOSS_TOKENIZER_ROLE: MOSS_TOKENIZER_ROLE}

#: The presets (ADR 0138 decision 13). The names' base storage is the VAE's f32 and the token
#: `dit8` overrides the DiT graphs to i8 (`i8` alone would claim an i8 VAE — ADR 0074's grammar);
#: the token is the weights name `dit` itself, so no abbreviation table is needed (ADR 0074 decision
#: 4). The text encoder is i8 in both and has no token, as umT5 has none in Wan's names.
#:
#: `session` is the DiT pair's knobs — the TS reader accepts exactly these three keys and checks
#: them against its preset table (`packages/models/src/moss/preset.ts`); `dit_text`, the text
#: encoder and the VAE run in f32 in both. `label` / `description` are the selection UI's fields
#: (ADR 0075 decision 1 — 64 / 200 characters; ADR 0128 C3); speed and quality are not written there
#: (they depend on the GPU), nor that a preset is the default (`defaultQuant` says it — ADR 0075
#: decision 3).
MOSS_QUANTS: Mapping[str, Any] = {
    "f32+dit8": {
        "weights": {},
        "session": {},
        "label": "int8 transformer, f32 compute",
        "description": "Transformer and text encoder weights stored as int8 (one scale per output"
        " channel) and computed in f32; the VAE stays f32. The reference preset.",
    },
    "f32+dit8-a8-attn8-online": {
        "weights": {},
        "session": {
            "linearCompute": "a8",
            "attentionCompute": "a8",
            "attentionScoreStorage": "none",
        },
        "label": "int8 transformer, int8 activations, online attention",
        "description": "The reference preset's weights with per-token int8 activations in the"
        " transformer's linear layers and attention computed online (the scores are never"
        " stored).",
    },
}

#: The default preset — the practical one (ADR 0128 decision 1 through ADR 0138 decision 13 and its
#: Rulings, 2026-10-28): the self A/B check against `f32+dit8` passes (relRMS at 1024² 1.85e-2 for
#: `dit` and 1.57e-2 for `dit_skip22`, ceilings 3.7e-2 / 3.2e-2 — ADR 0138 Results), and the
#: maintainer's visual review of the two presets side by side (seeds 42–45 × 4 prompts, 1024², 50
#: steps, guidance 3, shift 5) found the practical preset's quality acceptable. Upstream's default
#: request took 238.0 s against 574.8 s for `f32+dit8` on the RTX 3080 Ti under Node + Dawn, and the
#: practical DiT pair needs 7.33 GB against 10.07 GB at 1024² (research 2026-10-28 §5.3 / §5.5). The
#: text encoder is i8 in both (the maintainer's ruling 51 — ADR 0137 decision 5). It declares no
#: `gpuFeatures` (ADR 0128 C1): the online kernel asks for subgroups when the adapter has them and
#: runs its portable form otherwise (ADR 0132 decision 5). A quality ruling can change the default,
#: so it is held in this one place only.
MOSS_DEFAULT_QUANT = "f32+dit8-a8-attn8-online"

#: The pipeline-owned defaults (the hub passes them through — ADR 0041 §2; the TS schema is
#: `packages/models/src/moss/config.ts`). They are upstream's (ADR 0138 decision 9 assigned them
#: here): `MOSSImagePipeline.__call__`'s `height` / `width` 1024, `num_inference_steps` 50,
#: `guidance_scale` 3.0 and `negative_prompt` "" (the addendum of 2026-10-31, A2), and the pinned
#: `scheduler/scheduler_config.json`'s `shift` 5.0 (`moss/tests/test_distribution.py` reads both
#: from the pinned snapshot). Skipping block 22 in the negative branch is not a knob (decision 4).
MOSS_PIPELINE_CONFIG: Mapping[str, Any] = {
    "scheduler": {"shift": 5.0},
    "defaults": {
        "width": 1024,
        "height": 1024,
        "steps": 50,
        "guidance": 3.0,
        "negativePrompt": "",
    },
}

#: The notice: the attribution of the checkpoint and of the two releases it bundles (the Qwen3.5
#: text encoder — its license file's copyright line is kept, Apache 2.0 §4(c) — and the FLUX.2 VAE,
#: whose cards ask that the marks imply no endorsement), then the modification notice (Apache 2.0
#: §4(b)): **the changes this repository made** to the upstream weights.
#:
#: MUST: the text matches the distribution's content — it is valid prose as a value, so neither
#: `verify_dist` nor the manifest check would notice a mismatch before it is published.
MOSS_NOTICE_MARKDOWN = """# NOTICE

This repository redistributes a modified form of the MOSS-Image-SFT checkpoint listed in
`README.md` (OpenMOSS, licensed under the Apache License, Version 2.0 — see `LICENSE.md`),
including two parts that the checkpoint bundles from other releases, both licensed under the
Apache License, Version 2.0 as well:

- Its text encoder is the Qwen3.5 text encoder of `Qwen/Qwen3.5-9B` (the Qwen team, Alibaba
  Cloud), with its tokenizer. The license file of that release carries the copyright notice
  "Copyright 2026 Alibaba Cloud".
- Its VAE is the FLUX.2 VAE with the FLUX.2 small decoder (Black Forest Labs): the decoder and
  `post_quant_conv` come from `black-forest-labs/FLUX.2-small-decoder`, and the latent
  batch-normalization statistics from the VAE of `black-forest-labs/FLUX.2-klein-4B`. Only the
  decode path is included here; the VAE encoder is not.

This is not an official release of OpenMOSS, the Qwen team or Black Forest Labs, and it is not
endorsed by them; their names and marks (FLUX among them) are used only to describe where the
weights come from.

The following changes were made:

- The weights were converted into the Karume container format (a `.krm` part sequence whose first
  part carries the graph and model descriptors).
- **int8 transformer**: the weight matrices of all linear layers of the transformer (the text
  refiner included) are stored as int8, quantized from the source bfloat16 values to the nearest
  step with one float32 scale per output channel (symmetric). Its other parameters
  (normalization weights, biases) keep the source values in float32 storage. The int8 weights are
  computed in float32, or, in the quant that declares it, multiplied with activations quantized to
  int8 per token at run time.
- The transformer is re-expressed as three graphs: the text refiner (run once per prompt), the
  transformer over the joint image and text sequence, and the same transformer without block 22
  (zero-based) for the negative branch, which holds no weights of its own and uses the first
  graph's. The image and text rows are joined with a boolean mask instead of a concatenation, and
  the output head runs on every row (the host keeps the image rows). The rotary embedding is built
  on the host from per-axis base tables stored in the container, and the timestep embedding is
  looked up in a table of the 1,000 training-grid timesteps stored in the container (both computed
  with the upstream code).
- **f32 VAE decoder**: the FLUX.2 small decoder and `post_quant_conv` keep the source float32
  values. They are re-expressed as three graphs with a symbolic latent height and width, split
  around the decoder's middle attention block; the nearest-neighbour ×2 upsampling is written as
  a repetition (the same values). The latent batch-normalization statistics the decode path
  restores (the FLUX.2 [klein] 4B VAE's bfloat16 values, which the checkpoint holds widened to
  float32) are stored in the container as a mean and a standard deviation
  (`sqrt(running_var + eps)`, computed as upstream does). The host decodes untiled, as upstream
  does.
- **int8 text encoder**: the text tower of the checkpoint's Qwen3.5 text encoder (its 32 decoder
  layers and final norm; no vision tower and no language-model head) is stored with the weight
  matrices of its linear layers and its token embedding as int8 (one float32 scale per output row,
  symmetric), quantized from the source bfloat16 values. Its other parameters keep the source
  values in float32 storage (the zero-centred normalization weights with 1 added). The token
  positions and the rotary cos / sin rows are host inputs, and the causal mask is built in the
  graph from the positions.
- The tokenizer of the checkpoint was converted into one JSON table (vocabulary, merges, added
  tokens, and the character-class and normalization tables of its pre-tokenizer) that the host
  uses to reproduce its token ids.

No retraining and no fine-tuning were performed. The original checkpoint is not distributed here.
"""


@dataclass(frozen=True)
class MossSources:
    """The inputs of an assembly: one representative path per container role and the tokenizer
    asset (all under `outputs/series/`)."""

    containers: Mapping[str, Path]
    tokenizer: Path


def moss_sources(series_dir: Path) -> MossSources:
    """The inputs under the series parent directory (`outputs/series/`)."""

    def container(series: str, role: str) -> Path:
        return series_dir / series / role / MOSS_MODEL_FILE

    return MossSources(
        containers={
            MOSS_TEXT_ENCODER_ROLE: container(MOSS_TEXT_SERIES, MOSS_TEXT_ENCODER_ROLE),
            **{role: container(MOSS_DIT_SERIES, role) for role in MOSS_DIT_ROLES},
            **{role: container(MOSS_VAE_SERIES, role) for role in MOSS_VAE_ROLES},
        },
        tokenizer=series_dir / MOSS_TOKENIZER_SERIES / MOSS_TOKENIZER_FILE,
    )


def moss_repo_name(_model: str) -> str:
    """The distribution repository (one per family — every model of it gives the same one)."""
    return MOSS_REPO_NAME


def assert_container_assets(role: str, container: Path) -> None:
    """The container declares the assets the TS admission reads, under their roles
    ({@link MOSS_CONTAINER_ASSETS}; payloads are not read).

    MUST: fails at assembly. A container without them passes assembly and `verify_dist`, and the
    user's `fromPretrained` fails only in admission, after the manifest and the descriptors were
    fetched.
    """
    expected = MOSS_CONTAINER_ASSETS.get(role, {})
    try:
        declared = read_asset_declarations(container)
    except ContainerReadError as cause:
        raise DistError(f"{container}: {cause}") from cause
    roles = {name: entry[0] for name, entry in declared.items()}
    if roles != dict(expected):
        raise DistError(
            f"{container}: the {role} container declares the assets {roles} (expected"
            f" {dict(expected)} — rewrite the series with its writer)"
        )


def assert_borrows_every_weight(borrower: Path, lender: Path) -> None:
    """The borrowing container (`dit_skip22`) holds no weight bytes and borrows every initializer
    from the lender (`dit`) with the lender's dtype and shape (ADR 0138 decision 4).

    MUST: fails at assembly. A series where only the lender was rewritten passes every per-container
    check, and the runtime's sharing check runs only when the Sessions are built — after the user
    downloaded the weights.
    """
    carried = sum(part.stat().st_size for part in component_parts(borrower)[1:])
    if carried:
        raise DistError(
            f"{borrower}: holds {carried} bytes outside its descriptors — the negative-branch graph"
            " borrows every weight of the transformer (`python -m moss.export_dit write`)"
        )
    graph = ir_graph(borrower)
    try:
        opened = open_container(lender)
    except ContainerReadError as cause:
        raise DistError(f"{lender}: {cause}") from cause
    lent = ir_graph(lender, opened)
    supplied = {name for bound in opened.graphs.values() for name in bound.supplies}
    initializers = graph.get("initializers") or {}
    owned = sorted(
        name
        for name, entry in initializers.items()
        if not (isinstance(entry, dict) and entry.get("shared") is True)
    )
    if owned:
        raise DistError(f"{borrower}: initializers {owned[:4]} are not shared declarations")
    if not initializers:
        raise DistError(f"{borrower}: declares no initializer to borrow")
    values, lent_values = graph.get("values") or {}, lent.get("values") or {}
    for name in sorted(initializers):
        if name not in supplied:
            raise DistError(f"{borrower}: the shared initializer '{name}' is not in {lender}")
        if values.get(name) != lent_values.get(name):
            raise DistError(
                f"{borrower}: the shared initializer '{name}' is {values.get(name)}, the lender's"
                f" is {lent_values.get(name)}"
            )


def text_widths(text_encoder: Path, dit_text: Path) -> tuple[Any, Any]:
    """The text tower's row width and the width `dit_text` reads (the last axis of each)."""
    encoder = ir_graph(text_encoder)
    outputs = encoder.get("outputs") or []
    values = encoder.get("values") or {}
    output = values.get(outputs[0]) if len(outputs) == 1 else None
    shape = output.get("shape") if isinstance(output, dict) else None
    reads = graph_inputs(ir_graph(dit_text), dit_text).get(MOSS_DIT_TEXT_INPUT)
    return (
        shape[-1] if isinstance(shape, list) and shape else None,
        reads[-1] if isinstance(reads, list) and reads else None,
    )


def assert_text_width(text_encoder: Path, dit_text: Path) -> None:
    """The text tower's rows are as wide as `dit_text`'s `text_embeds` (4,096)."""
    produced, read = text_widths(text_encoder, dit_text)
    if produced != MOSS_TEXT_WIDTH or read != MOSS_TEXT_WIDTH:
        raise DistError(
            f"{text_encoder}: the text tower writes rows of {produced!r} and {dit_text} reads rows"
            f" of {read!r} (both must be {MOSS_TEXT_WIDTH})"
        )


def assert_tokenizer(path: Path) -> None:
    """The tokenizer asset is the format the TS reader reads, built from the pinned upstream
    folder, with the bound the text encoder and the DiT share.

    MUST: fails at assembly. The asset is a table, so `verify_dist`'s container checks do not see
    it, and the reader (`parseQwen3_5TokenizerAsset`) checks only its form — a tokenizer of another
    revision would give other ids at the user's side, and a bound other than 4,096 is refused only
    by the TS admission after the weights were downloaded.
    """
    assert_component_present(path)
    try:
        asset = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as cause:
        raise DistError(f"{path}: the tokenizer asset is not readable JSON — {cause}") from cause
    if not isinstance(asset, dict):
        raise DistError(f"{path}: the tokenizer asset is not an object")
    if asset.get("format") != MOSS_TOKENIZER_FORMAT:
        raise DistError(
            f"{path}: format {asset.get('format')!r} is not {MOSS_TOKENIZER_FORMAT!r}"
            " (`python -m qwen3_5.tokenizer`)"
        )
    source = asset.get("source")
    declared = (
        {key: source.get(key) for key in ("repo", "revision", "subfolder")}
        if isinstance(source, dict)
        else None
    )
    expected = {
        "repo": SOURCE.repo,
        "revision": SOURCE.revision,
        "subfolder": MOSS_TOKENIZER_SUBFOLDER,
    }
    if declared != expected:
        raise DistError(
            f"{path}: the tokenizer comes from {declared!r}, not the pinned {expected!r} — a"
            " tokenizer of another checkpoint is not distributed"
        )
    max_length = asset.get("maxLength")
    if (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or max_length != MOSS_MAX_TEXT_TOKENS
    ):
        raise DistError(
            f"{path}: maxLength {max_length!r} is not {MOSS_MAX_TEXT_TOKENS} (the text encoder's"
            " and the DiT's bound)"
        )


def moss_plan(sources: MossSources, model: str | None = None) -> ModelPlan:
    """The plan of the one model (every check and read done here — nothing is written)."""
    model = MOSS_MODEL if model is None else model
    assert_model_name(model)
    if model != MOSS_MODEL:
        raise DistError(
            f"MOSS-Image has no model {model!r} (known: {MOSS_MODEL}) — the {MOSS_REPO_NAME}"
            " distribution holds MOSS-Image-SFT only"
        )
    containers = sources.containers
    for role in MOSS_CONTAINER_ROLES:
        container = containers[role]
        assert_component_present(container)
        # A borrower runs on its lender's weights, so the lender's binding table is checked.
        stored = MOSS_LENDERS.get(role, role)
        assert_storage(stored, containers[stored], MOSS_STORAGE_REQUIREMENTS)
        assert_storage_absent(stored, containers[stored], MOSS_STORAGE_FORBIDDEN)
        # Every container is baked from the pinned checkpoint (the text tower too — it is the
        # checkpoint's `text_encoder` folder), so all of them name its licence and revision.
        assert_upstream_provenance(container, license=SOURCE.license, revision=SOURCE.revision)
        assert_container_assets(role, container)
    for role, lender in MOSS_LENDERS.items():
        assert_borrows_every_weight(containers[role], containers[lender])
    assert_text_width(containers[MOSS_TEXT_ENCODER_ROLE], containers[MOSS_DIT_TEXT_ROLE])
    assert_tokenizer(sources.tokenizer)
    placements = {**containers, MOSS_TOKENIZER_ROLE: sources.tokenizer}
    return ModelPlan(
        name=model,
        pipeline=MOSS_PIPELINE,
        artifacts={
            role: Artifact(MOSS_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=MOSS_WEIGHTS,
        assets=MOSS_ASSETS,
        quants=complete_quant_weights(MOSS_WEIGHTS, MOSS_QUANTS),
        default_quant=MOSS_DEFAULT_QUANT,
        pipeline_config=MOSS_PIPELINE_CONFIG,
    )


def moss_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """The plan from the `--series` parent (the CLI's dispatch target)."""
    return moss_plan(moss_sources(series_dir), model)


#: `--pipeline moss-image` (the driver merges it into the core's PIPELINES). The upstream licence
#: (Apache 2.0) asks for its text and a modification notice at the repository root (ADR 0092
#: decision 7).
PIPELINE = Pipeline(
    default_model=MOSS_MODEL,
    repo_name=moss_repo_name,
    plan=moss_dist_plan,
    card_profiles={"moss-image": render_moss_model_card},
    root_files={"LICENSE.md": apache_license_2_0(), "NOTICE.md": MOSS_NOTICE_MARKDOWN},
)
