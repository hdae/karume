"""The Wan distribution recipe (Wan2.1 / Wan2.2) — series layout, output path table, quant table
and the choice of card (ADR 0118 decision 7, ADR 0121 decisions 2 / 9 / 10).

The general-purpose assembly engine (placement, folding of shared entries, sha256, manifest,
staging/swap, verification) lives in `karume.dist`. This module owns only **the facts specific to
Wan**: what to pick up from which series directory, at which distribution path and under which
dtype label to lay it out, and which quant is the default.

The facts that differ per generation (distribution repository name, pipeline contract, shipped
models, the DiT and VAE series, the transformer's storage labels, the pin of the text assets'
provenance, the quant table and default preset, `pipelineConfig`, NOTICE, card) are collected in
the generation table {@link WanGeneration} ({@link WAN21} / {@link WAN22}), and a single plan
function ({@link wan_plan}) builds both. The placement table, the storage requirement / prohibition
tables and the weights declaration are derived from the transformer storage labels the generation
holds (Wan2.2 has only i8 and no f16 DiT — ADR 0121 decision 2). The paragraphs below describe the
Wan2.1 distribution; the differences of Wan2.2 are summarized in the doc of {@link WAN22}.

The distribution ships 3 graphs (the DiT's S-shaped `transformer` and the VAE's chunk graphs
`vae_decoder_first` / `vae_decoder_next` — series `wan2.1-t2v-1.3b-f16-dyn`) and 2 per-model assets
(the step-1 text embeddings `text_embeds` — decision 4; the umT5 tokenizer and preprocessing tables
`umt5_tokenizer` — ADR 0119 addendum "Design of step 10d" C; both quant-independent, ADR 0109
decision 4). Only `transformer` has 2 containers, one per storage label (f16, and the i8 of series
`wan2.1-t2v-1.3b-i8-dyn` — ADR 0120 decision 1). The per-axis RoPE base table `rope_base` is a
**container asset** of `transformer` (role `rope-base`), so it is not in the manifest's `assets`
(ADR 0109 decision 4 — the same placement as Anima).

The fourth component, `text_encoder` (the umT5-XXL encoder in i8 — series `umt5-xxl-i8-dyn`), is in
the plan as an artifact of its own, but the published distribution builds it as **a
cross-repository reference to the separate repository `karume-umt5-xxl`** (ADR 0119 addendum A —
dist's 5 `--ref-*` options, the referenced side being the distribution built by
`--pipeline umt5`). The container checks (provenance, binding table, input / output contract) go
through the same single function as umT5's distribution recipe
(`wan.umt5_distribution.assert_umt5_encoder`).

**One repository per family, a separate repository per generation** (`karume-wan2.1` — ADR 0092
decisions 1 / 2). A model is an axis within a generation; today there is only `t2v-1.3b`. There are
3 quant presets ({@link WAN_QUANTS}): `f16` (DiT and VAE weights stored as f16, f32 activations —
ADR 0118 decision 7; the preset closest to the original weights, used when asked for explicitly),
and with the DiT weights in i8, the reference preset `f16+dit8` and the practical preset
`f16+dit8-a8-attn8-s16` (ADR 0120 decision 1). The default is the practical preset (ADR 0120 ruling
2026-10-04 item 4 — {@link WAN_DEFAULT_QUANT}).

The public surface is the 2 pipelines {@link PIPELINE} (`--pipeline wan` — Wan2.1) and
{@link TI2V_PIPELINE} (`--pipeline wan-ti2v` — Wan2.2) — the repository's dist driver
(`tools/export-recipes/dist.py`) merges them into the core's PIPELINES.

MUST: this module does not import torch (`import dist` does not load torch —
`TestImportingTheDriver` in `tests/test_dist_driver.py`). The writers (`wan.export_dit` /
`wan.ti2v_export_dit` / `wan.export_vae` / `wan.text_embeds`) load torch, so the spellings (series
names, asset names, metadata keys, graph input names) live here, and the checks in
`wan/tests/test_distribution.py` confirm that they match the writers' spellings.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from _shared.container_read import (
    ContainerReadError,
    open_container,
    read_asset,
    read_asset_declarations,
)
from _shared.licenses import apache_license_2_0
from _shared.upstream import assert_upstream_provenance
from karume.dist import (
    Artifact,
    CrossRepoReference,
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
from wan import umt5_tokenizer
from wan.card import WAN21_CARD, WAN22_CARD, WanCard, render_wan_model_card
from wan.prompts import FIXED_PROMPTS
from wan.sources import DEFAULT_MODEL, SOURCES, WAN22_TEXT_MODEL
from wan.umt5_distribution import (
    UMT5_DEFAULT_MODEL,
    UMT5_OUTPUT_PATHS,
    UMT5_REPO_NAME,
    UMT5_ROLE,
    assert_umt5_encoder,
    umt5_container,
)

#: The pipeline contract (ADR 0041 §2 — per model). The TS side's accepted set is
#: `WAN_PIPELINE_NAME` / `WAN_PIPELINE_MAJOR` (`packages/models/src/wan/config.ts`).
WAN_PIPELINE = "wan/1"

#: The distribution repository name (ADR 0092 decisions 1 / 2 — one repository per family, a
#: separate one per generation; Wan2.2 is `karume-wan2.2`).
WAN_REPO_NAME = "karume-wan2.1"

#: The DiT and VAE series (decision 7 — the `-dyn` suffix is ADR 0077's convention). The writers'
#: spellings are `wan.export_dit.SERIES` / `wan.export_vae.SERIES_NAME`.
WAN_SERIES = "wan2.1-t2v-1.3b-f16-dyn"

#: The DiT's i8 series (ADR 0120 decision 2 — transformer only). The writer's spelling is
#: `wan.export_dit.I8_SERIES`.
WAN_I8_SERIES = "wan2.1-t2v-1.3b-i8-dyn"

#: The text-embeddings series (the place of a compile product without a graph —
#: docs/assets-layout.md), its asset's file name and the only metadata key. The writer's spellings
#: are `SERIES_NAME` / `ASSET_NAME` / `METADATA_KEY` of `wan.text_embeds`.
WAN_TEXT_EMBEDS_SERIES = "wan2.1-t2v-1.3b-text-embeds"
WAN_TEXT_EMBEDS_FILE = "text_embeds.safetensors"
WAN_TEXT_EMBEDS_METADATA_KEY = "karume.wan.text_embeds"

#: The umT5 dtype the embeddings were made with (the metadata's `text_encoder.dtype` — decision 4;
#: the grounds on which the NOTICE and the card say "made with the upstream encoder in bfloat16").
#: The writer's spelling is `wan.text_embeds.asset_metadata` (built from `TEXT_ENCODER_DTYPE`).
WAN_TEXT_ENCODER_DTYPE = "bfloat16"

#: The versions of the dependencies that determine the embedding values (the metadata's `versions` —
#: the "fixed" of decision 4). diffusers holds the upstream `_get_t5_prompt_embeds` /
#: `prompt_clean`, ftfy the normalization rules, transformers the UMT5 implementation and the
#: tokenizer (the handling of out-of-vocabulary ids changes between versions — research 2026-10-03
#: umT5). The values are the same as the `==` pins of pyproject's `wan` group (a check in
#: `wan/tests/test_distribution.py` compares both).
WAN_TEXT_EMBEDS_VERSIONS: Mapping[str, str] = {
    "diffusers": "0.39.0",
    "ftfy": "6.3.1",
    "transformers": "5.14.1",
}

#: The representative name of the container in a series component directory (the split form is
#: `model-0000N-of-0000M.krm`).
WAN_MODEL_FILE = "model.krm"

#: The component names that have a graph = the keys of the manifest's weights = **the container's
#: graph names** (container-v1 §2.1 — the runtime looks them up by name with
#: `prepareContainer(opened, <weights key>)`). The series component directories use the same
#: spelling, but that is a convention, not a derivation (the writers state `TARGET` / `TARGETS` /
#: `GRAPH_NAME` — the check in `tests/test_graph_names.py`). `text_encoder` is the same key as in
#: the umT5 distribution (the graph name of the referenced container — the source of truth for the
#: spelling is `wan.umt5_distribution.UMT5_ROLE`). The order is the order of the generation stages
#: (the I2V VAE encoder → text → DiT → VAE decoder), and it is the union over all generations (the
#: components of each generation are the keys of {@link WanGeneration.weights}).
WAN_TEXT_ENCODER_ROLE = UMT5_ROLE
WAN_TRANSFORMER_ROLE = "transformer"
WAN_VAE_FIRST_ROLE = "vae_decoder_first"
WAN_VAE_NEXT_ROLE = "vae_decoder_next"

#: The graph names = weights keys of the VAE decoder's 32-latent tile pair (ADR 0134 — Wan2.2 only)
#: (`shared_target` of the writer `wan.export_vae --tile 32 --shared`). Containers without weight
#: bytes: every initializer names, through a shared declaration, the weights of the default-tile
#: pair (the lender) at the root of the same series.
WAN_VAE_FIRST_32_ROLE = "vae_decoder_first_32"
WAN_VAE_NEXT_32_ROLE = "vae_decoder_next_32"

#: The 3 graphs of the VAE encoder that maps the I2V conditioning image (one) to the latent (ADR
#: 0121 decision 11 as revised, step 9a — graphs of symbolic length cut before and after the mid
#: attention: pre / attn / post). The writer's spelling is `wan.export_vae_encoder.TARGETS`. Only a
#: generation that has the encoder ships it (Wan2.2 — {@link WanGeneration.vae_encoder}). The order
#: is the execution order, as in the writer.
WAN_VAE_ENCODER_PRE_ROLE = "vae_encoder_pre"
WAN_VAE_ENCODER_ATTN_ROLE = "vae_encoder_attn"
WAN_VAE_ENCODER_POST_ROLE = "vae_encoder_post"
WAN_VAE_ENCODER_ROLES: tuple[str, ...] = (
    WAN_VAE_ENCODER_PRE_ROLE,
    WAN_VAE_ENCODER_ATTN_ROLE,
    WAN_VAE_ENCODER_POST_ROLE,
)

WAN_GRAPH_ROLES: tuple[str, ...] = (
    *WAN_VAE_ENCODER_ROLES,
    WAN_TEXT_ENCODER_ROLE,
    WAN_TRANSFORMER_ROLE,
    WAN_VAE_FIRST_ROLE,
    WAN_VAE_NEXT_ROLE,
    WAN_VAE_FIRST_32_ROLE,
    WAN_VAE_NEXT_32_ROLE,
)

#: The placement roles of the `transformer` containers (one per storage label — ADR 0120 decision 1;
#: the same spelling as anima's `transformer_f16` / `transformer_i8`). They key the placement table
#: and the storage requirement / prohibition tables; the manifest's weights key stays the
#: container's graph name {@link WAN_TRANSFORMER_ROLE}.
WAN_TRANSFORMER_F16_ROLE = "transformer_f16"
WAN_TRANSFORMER_I8_ROLE = "transformer_i8"
WAN_TRANSFORMER_ROLES: tuple[str, ...] = (WAN_TRANSFORMER_F16_ROLE, WAN_TRANSFORMER_I8_ROLE)

#: Transformer storage label → placement role (the roles of a generation are looked up from the
#: `transformer_labels` of the generation table {@link WanGeneration}).
WAN_TRANSFORMER_LABEL_ROLES: Mapping[str, str] = {
    "f16": WAN_TRANSFORMER_F16_ROLE,
    "i8": WAN_TRANSFORMER_I8_ROLE,
}

#: The placement roles of the VAE decoder's i8 containers (ADR 0133 decision 7 — Wan2.2 only). The
#: f16 roles stay the container graph names {@link WAN_VAE_FIRST_ROLE} / {@link WAN_VAE_NEXT_ROLE}
#: (renaming them would change the Wan2.1 distribution). The manifest's weights keys are the graph
#: names in both cases.
WAN_VAE_FIRST_I8_ROLE = "vae_decoder_first_i8"
WAN_VAE_NEXT_I8_ROLE = "vae_decoder_next_i8"

#: Storage label of the 2 decoder graphs → placement role (the roles of a generation are looked up
#: from the `vae_labels` of the generation table {@link WanGeneration} — the same form as the
#: transformer's {@link WAN_TRANSFORMER_LABEL_ROLES}).
WAN_VAE_FIRST_LABEL_ROLES: Mapping[str, str] = {
    "f16": WAN_VAE_FIRST_ROLE,
    "i8": WAN_VAE_FIRST_I8_ROLE,
}
WAN_VAE_NEXT_LABEL_ROLES: Mapping[str, str] = {
    "f16": WAN_VAE_NEXT_ROLE,
    "i8": WAN_VAE_NEXT_I8_ROLE,
}

#: The i8 placement roles of the 32-latent tile pair (the same rule as the decoder — the f16 roles
#: stay the graph names).
WAN_VAE_FIRST_32_I8_ROLE = "vae_decoder_first_32_i8"
WAN_VAE_NEXT_32_I8_ROLE = "vae_decoder_next_32_i8"

#: Latent tile of a pair that borrows weights from the lender pair → (first's storage label → role,
#: next's storage label → role) (ADR 0134 decision 5 — one pair per storage label). The default-tile
#: pair that holds the weights is {@link WAN_VAE_FIRST_LABEL_ROLES} /
#: {@link WAN_VAE_NEXT_LABEL_ROLES}.
WAN_VAE_SHARED_LABEL_ROLES: Mapping[int, tuple[Mapping[str, str], Mapping[str, str]]] = {
    32: (
        {"f16": WAN_VAE_FIRST_32_ROLE, "i8": WAN_VAE_FIRST_32_I8_ROLE},
        {"f16": WAN_VAE_NEXT_32_ROLE, "i8": WAN_VAE_NEXT_32_I8_ROLE},
    ),
}

#: Role of a borrowing pair → role of the lender (the default-tile container at the same series root
#: with the same storage label — the writer reads `<series root>/<graph name>/model.krm` as the
#: lender).
WAN_VAE_LENDER_ROLES: Mapping[str, str] = {
    WAN_VAE_FIRST_32_ROLE: WAN_VAE_FIRST_ROLE,
    WAN_VAE_FIRST_32_I8_ROLE: WAN_VAE_FIRST_I8_ROLE,
    WAN_VAE_NEXT_32_ROLE: WAN_VAE_NEXT_ROLE,
    WAN_VAE_NEXT_32_I8_ROLE: WAN_VAE_NEXT_I8_ROLE,
}

#: The placement roles that have a container (text_encoder has one, in i8; the transformer and the
#: VAE decoder one per storage label; the VAE encoder one each, in f16). The union of the roles of
#: all generations. Each generation uses the roles of its own storage labels, and only a generation
#: with the encoder uses the encoder roles ({@link WanGeneration.container_roles} — Wan2.1 does not
#: have the 3 encoder graphs). The storage requirement / prohibition tables, the output paths and
#: the weights declaration below are likewise tables over all generations, from which each
#: generation looks up its own roles.
WAN_CONTAINER_ROLES: tuple[str, ...] = (
    *WAN_VAE_ENCODER_ROLES,
    WAN_TEXT_ENCODER_ROLE,
    *WAN_TRANSFORMER_ROLES,
    WAN_VAE_FIRST_ROLE,
    WAN_VAE_FIRST_I8_ROLE,
    WAN_VAE_NEXT_ROLE,
    WAN_VAE_NEXT_I8_ROLE,
    WAN_VAE_FIRST_32_ROLE,
    WAN_VAE_FIRST_32_I8_ROLE,
    WAN_VAE_NEXT_32_ROLE,
    WAN_VAE_NEXT_32_I8_ROLE,
)

#: The per-model assets (the keys of the manifest's `assets` = role names — `TEXT_EMBEDS` in the TS
#: side's `text-stage.ts`).
WAN_TEXT_EMBEDS_ROLE = "text_embeds"

#: The asset of the umT5 tokenizer and preprocessing tables (the key of the manifest's `assets` =
#: role name — the TS side's reader is `parseWanTokenizerAsset`). The asset is fetched in full
#: regardless of quant and path, so it lives in this repository rather than the umT5 one (as a
#: cross-repository reference, even the precomputed path, which does not fetch umT5, would touch the
#: umT5 repository — ADR 0119 addendum C).
WAN_TOKENIZER_ROLE = "umt5_tokenizer"

#: The upstream component the tokenizer asset names (`source.subfolder` of the writer
#: `wan.umt5_tokenizer.build_asset`).
WAN_TOKENIZER_SUBFOLDER = "tokenizer"

#: The versions of the tokenizer asset that determine the id sequence and the preprocessing results
#: (the `==` pins of the `wan` group — the same values as {@link WAN_TEXT_EMBEDS_VERSIONS}.
#: transformers is the source of truth for the id sequence, ftfy the origin of the preprocessing
#: tables — ADR 0119 decisions 1 / 2).
WAN_TOKENIZER_VERSIONS: tuple[str, ...] = ("ftfy", "transformers")

#: The asset name and role of the RoPE base table the `transformer` container declares (the writer
#: is `ROPE_BASE_ASSET` / `ROPE_BASE_ROLE` of `wan.export_dit`, the reader `ROPE_BASE` in the TS
#: side's `dit-loop.ts`).
WAN_ROPE_BASE_ASSET = "rope_base"
WAN_ROPE_BASE_ROLE = "rope-base"

#: The latent input of the VAE chunk graphs (the first input `[C, 1, t, t]` — the remaining inputs
#: are the cache). The writer's spelling is `wan.export_vae.LATENT_INPUT`, the reader's
#: `WAN_VAE_LATENT_INPUT` in the TS side's `vae-chunks.ts`.
WAN_VAE_LATENT_INPUT = "latent"

#: The DiT's context input (`[1, rows, width]` — the row width and the upper bound on the valid
#: length of the embedding asset are taken from it). The writer's spelling is
#: `wan.export_dit.INPUT_NAMES`.
WAN_DIT_CONTEXT_INPUT = "encoder_hidden_states"

#: The relative output paths (**inside the model subtree**) — the one place the placement table and
#: the manifest share. The weight paths include the storage label (`f16` / `i8`) so that adding a
#: storage preset does not move an existing path (the same spelling as Depth Anything's
#: `model.f32.krm` — ADR 0120, which added the i8 preset, did not move the f16 path).
#:
#: MUST: `text_encoder` takes the path of the same component in the umT5 distribution
#: (`UMT5_OUTPUT_PATHS`) — a cross-repository reference resolves the part `<model name>/<this path>`
#: that the referenced `karume.json` declares (`karume.dist.external_refs`), so diverging spellings
#: make the assembly fail with "not in the referenced distribution".
WAN_OUTPUT_PATHS: Mapping[str, str] = {
    **{role: f"{role}/model.f16.krm" for role in WAN_VAE_ENCODER_ROLES},
    WAN_TEXT_ENCODER_ROLE: UMT5_OUTPUT_PATHS[UMT5_ROLE],
    WAN_TRANSFORMER_F16_ROLE: f"{WAN_TRANSFORMER_ROLE}/model.f16.krm",
    WAN_TRANSFORMER_I8_ROLE: f"{WAN_TRANSFORMER_ROLE}/model.i8.krm",
    WAN_VAE_FIRST_ROLE: f"{WAN_VAE_FIRST_ROLE}/model.f16.krm",
    WAN_VAE_FIRST_I8_ROLE: f"{WAN_VAE_FIRST_ROLE}/model.i8.krm",
    WAN_VAE_NEXT_ROLE: f"{WAN_VAE_NEXT_ROLE}/model.f16.krm",
    WAN_VAE_NEXT_I8_ROLE: f"{WAN_VAE_NEXT_ROLE}/model.i8.krm",
    WAN_VAE_FIRST_32_ROLE: f"{WAN_VAE_FIRST_32_ROLE}/model.f16.krm",
    WAN_VAE_FIRST_32_I8_ROLE: f"{WAN_VAE_FIRST_32_ROLE}/model.i8.krm",
    WAN_VAE_NEXT_32_ROLE: f"{WAN_VAE_NEXT_32_ROLE}/model.f16.krm",
    WAN_VAE_NEXT_32_I8_ROLE: f"{WAN_VAE_NEXT_32_ROLE}/model.i8.krm",
    WAN_TEXT_EMBEDS_ROLE: f"{WAN_TEXT_EMBEDS_ROLE}/{WAN_TEXT_EMBEDS_FILE}",
    WAN_TOKENIZER_ROLE: f"{WAN_TOKENIZER_ROLE}/{umt5_tokenizer.ASSET_FILE}",
}

#: Storage dtype requirements (a measured incident where a plain f32 asset passed assembly, loading
#: and execution silently all the way to the reference-match check — the same grounds as Anima /
#: SBV2 / Depth Anything). In an f16 series only the fake-quant targets become f16 (bias / norm stay
#: f32), so "contains f16" is required; in an i8 series only the linear weights become i8 (scale,
#: bias and norm are f32), so "contains i8" is required. The per-kind storage of umT5's i8 series is
#: checked separately by the binding-table check (`wan.umt5_distribution.assert_umt5_bindings`). The
#: VAE encoder is the same f16 series as the decoder (writer `wan.export_vae_encoder` — f16 preset
#: only). In the decoder's i8 series only the convolution weights are i8 (ADR 0133 decision 7 —
#: writer `wan.export_vae --dtype i8`). The 32-latent tile pair (ADR 0134) has no binding table (a
#: shared declaration has no supply, and the container does not keep the declared storage either),
#: so the requirement / prohibition is checked on the lender container's binding table
#: ({@link WAN_VAE_LENDER_ROLES}) — the storage a borrowing pair runs on is the storage of the
#: lender's resident weights ({@link wan_plan}).
WAN_STORAGE_REQUIREMENTS: Mapping[str, str] = {
    **dict.fromkeys(WAN_VAE_ENCODER_ROLES, "f16"),
    WAN_TEXT_ENCODER_ROLE: "i8",
    WAN_TRANSFORMER_F16_ROLE: "f16",
    WAN_TRANSFORMER_I8_ROLE: "i8",
    WAN_VAE_FIRST_ROLE: "f16",
    WAN_VAE_FIRST_I8_ROLE: "i8",
    WAN_VAE_NEXT_ROLE: "f16",
    WAN_VAE_NEXT_I8_ROLE: "i8",
    WAN_VAE_FIRST_32_ROLE: "f16",
    WAN_VAE_FIRST_32_I8_ROLE: "i8",
    WAN_VAE_NEXT_32_ROLE: "f16",
    WAN_VAE_NEXT_32_I8_ROLE: "i8",
}

#: Every compressed storage among the codec ledger's layouts (`karume.container.CODEC_LEDGER` minus
#: f32 / i32 — `wan/tests/test_distribution.py` checks that it matches the ledger).
_COMPRESSED_LAYOUTS = ("bf16", "f16", "i8", "i4", "i2")

#: The storage vocabulary that **must not** be in each role's binding table
#: ({@link assert_storage_absent}). A presence check alone lets through a mix-up that plugs in a
#: mixed series containing the required storage — for example, an i4 mixed series (default storage
#: i8 — the shape of anima's i4 series) satisfies "contains i8", so plugged into an i8 preset it
#: passes the requirement check. Plugging an f16 series into an i8 preset, or an i8 series into an
#: f16 preset, fails on both sides: the requirement is missing and a prohibition is hit.
#:
#: MUST: name, per role, **every** compressed storage except the one required — leave out even one
#: and exactly that storage form slips through silently (the same discipline as Depth Anything).
WAN_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    role: tuple(layout for layout in _COMPRESSED_LAYOUTS if layout != required)
    for role, required in WAN_STORAGE_REQUIREMENTS.items()
}

#: The weights declaration (container graph name → dtype label → placement role). The labels use the
#: storage dtype vocabulary and are 1:1 with the storage forms {@link WAN_STORAGE_REQUIREMENTS}
#: requires (ADR 0041 §3). `text_encoder` has the single label i8, so {@link complete_quant_weights}
#: fills it into every preset (a preset's weights is a complete mapping — a preset without umT5
#: cannot be expressed; the path is chosen by a construction option — ADR 0119 addendum B). The VAE
#: encoder also has the single label f16, so every preset takes it (even for T2V-only use — the
#: ruling of ADR 0121 addendum "Ruling on the plan of step 9 and the results of step 9a").
WAN_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    **{role: {"f16": WeightFiles(role)} for role in WAN_VAE_ENCODER_ROLES},
    WAN_TEXT_ENCODER_ROLE: {"i8": WeightFiles(WAN_TEXT_ENCODER_ROLE)},
    WAN_TRANSFORMER_ROLE: {
        "f16": WeightFiles(WAN_TRANSFORMER_F16_ROLE),
        "i8": WeightFiles(WAN_TRANSFORMER_I8_ROLE),
    },
    WAN_VAE_FIRST_ROLE: {
        "f16": WeightFiles(WAN_VAE_FIRST_ROLE),
        "i8": WeightFiles(WAN_VAE_FIRST_I8_ROLE),
    },
    WAN_VAE_NEXT_ROLE: {
        "f16": WeightFiles(WAN_VAE_NEXT_ROLE),
        "i8": WeightFiles(WAN_VAE_NEXT_I8_ROLE),
    },
    WAN_VAE_FIRST_32_ROLE: {
        "f16": WeightFiles(WAN_VAE_FIRST_32_ROLE),
        "i8": WeightFiles(WAN_VAE_FIRST_32_I8_ROLE),
    },
    WAN_VAE_NEXT_32_ROLE: {
        "f16": WeightFiles(WAN_VAE_NEXT_32_ROLE),
        "i8": WeightFiles(WAN_VAE_NEXT_32_I8_ROLE),
    },
}

#: The assets declaration (unconditional files independent of the quant choice — ADR 0041 §3,
#: decision 4).
WAN_ASSETS: Mapping[str, str] = {
    WAN_TEXT_EMBEDS_ROLE: WAN_TEXT_EMBEDS_ROLE,
    WAN_TOKENIZER_ROLE: WAN_TOKENIZER_ROLE,
}

#: Component-override token of a preset name → its weights name (ADR 0074 decision 4 — **the recipe
#: owns the definition of the abbreviations, and the generated model card's quant table always shows
#: the mapping**). Wan's base storage is `f16` (the VAE is fixed at f16) and only the transformer is
#: compressed, so preset names are `f16+dit8…` (the same abbreviation as anima — ADR 0120 decision
#: 1).
WAN_QUANT_ABBREVIATIONS: Mapping[str, str] = {"dit": WAN_TRANSFORMER_ROLE}

#: The quant table (ADR 0118 decision 7, ADR 0120 decision 1). The VAE has one storage label, so it
#: is not written in weights ({@link complete_quant_weights} fills it into a complete mapping). The
#: transformer has two, f16 / i8, so it is written per preset. The 3 `session` keys are the fields
#: the TS side's acceptance table (`WAN_SESSION_POLICY`) takes, and the practical preset's
#: declaration is the same as anima's preset of the same name.
#:
#: - `f16+dit8` is the reference preset (`session` is empty — the same i8 weights as the practical
#:   preset, the counterpart of the on-machine A/B check; ADR 0110 decision 5 ②).
#: - No intermediate presets (`f16+dit8-a8` / `f16+dit8-a8-attn8`) are made (decision 1 — isolating
#:   one knob at a time is covered by manifest overrides in the tests).
#:
#: The knob tokens of preset names (ADR 0074 §3 / ADR 0133 decision 6): `a8` = `linearCompute:
#: "a8"`, `attn8` = `attentionCompute: "a8"`, `s16` = `attentionScoreStorage: "f16"`, `online` =
#: `attentionScoreStorage: "none"` (Wan2.2 — ADR 0132), `conv8` = `convCompute: "a8"` (the conv3d of
#: Wan2.2's VAE decoder — ADR 0133).
#:
#: `label` / `description` are display fields for a selection UI (ADR 0075 decision 1 — English, 64
#: / 200 characters at most). They say nothing about speed or quality — the only speed measurement
#: of the a8 presets is on the B570 (−50% per forward — ADR 0120 step 4), and on Metal a8 is not
#: faster (investigation §3.1), so it cannot go into GPU-independent display fields. The quality (no
#: clear degradation compared with `f16` in step 6's visual check — ADR 0120 ruling 2026-10-04 item
#: 1) and the per-preset cost are written by the card (`wan/card.py`), which names the preset. Being
#: the default is not written either (`defaultQuant` points to it — ADR 0075 decision 3).
WAN_QUANTS: Mapping[str, Any] = {
    "f16": {
        "weights": {WAN_TRANSFORMER_ROLE: "f16"},
        "session": {},
        "label": "Full quality (f16)",
        "description": "Transformer and VAE weights in f16 storage with f32 compute — the"
        " largest download.",
    },
    "f16+dit8": {
        "weights": {WAN_TRANSFORMER_ROLE: "i8"},
        "session": {},
        "label": "Half-size transformer (int8)",
        "description": "Transformer weights stored as int8 (one scale per output channel) and"
        " computed in f32; the VAE stays f16. About half the transformer download.",
    },
    "f16+dit8-a8-attn8-s16": {
        "weights": {WAN_TRANSFORMER_ROLE: "i8"},
        "session": {
            "linearCompute": "a8",
            "attentionCompute": "a8",
            "attentionScoreStorage": "f16",
        },
        "label": "int8 transformer, int8 activations",
        "description": "The int8 transformer with per-token int8 activations in its linear layers"
        " and attention (integer dot products), and attention scores held in f16.",
    },
}
#: The default preset is the practical preset (ADR 0120 ruling 2026-10-04 item 4 — the default is
#: the most practical preset today: light, with little quality loss; step 6's visual check showed no
#: clear degradation compared with `f16`). The `f16` preset is the reference-side preset used when
#: asked for explicitly.
WAN_DEFAULT_QUANT = "f16+dit8-a8-attn8-s16"

#: Pipeline-owned settings (the hub passes them through — ADR 0041 §2; the TS side's schema is
#: `packages/models/src/wan/config.ts`). The values are the settings that produce the reference
#: outputs (decision 5):
#:
#: - `scheduler.shift` 3.0 — the `flow_shift` of `scheduler/scheduler_config.json` in the upstream
#:   Diffusers version (docstring "3.0 for 480P"). The product default starts from this value, and
#:   the visual A/B decides whether to change it (decision 5, ruling 2).
#: - `defaults.steps` 50 / `defaults.guidance` 5.0 — the defaults of the upstream
#:   `WanPipeline.__call__` (decision 5).
#:
#: The UniPC structure (solver order, bh2, `num_train_timesteps`) is not declared — the TS port
#: implements just that one branch (`WAN_UNIPC_CONFIG` in `scheduler.ts`), and making it declarable
#: would let the distribution side choose combinations that were never verified.
WAN_PIPELINE_CONFIG: Mapping[str, Any] = {
    "scheduler": {"shift": 3.0},
    "defaults": {"steps": 50, "guidance": 5.0},
}


#: The notice of modifications (Apache 2.0 §4(b)). Lists **the changes this repository made to the
#: upstream weights**.
#:
#: MUST: the text matches the contents of the distribution — as values it is plausible prose, so
#: `verify_dist` and the manifest checks let it through, and a mismatch would only be noticed after
#: shipping.
WAN_NOTICE_MARKDOWN = """# NOTICE

This repository redistributes a modified form of the Wan2.1 T2V 1.3B checkpoint listed in
`README.md` (Wan-AI, licensed under the Apache License, Version 2.0 — see `LICENSE.md`). The
following changes were made:

- The weights were converted into the Karume container format (a `.krm` part sequence whose first
  part carries the graph and model descriptors).
- **f16 storage**: every parameter of the float16 transformer and of the VAE decoder was rounded
  from the source float32 value to the nearest float16 value. Weight matrices and convolution
  kernels are stored as float16; the other parameters (biases, normalization weights) keep the
  rounded values in float32 storage. Computation runs in float32.
- **int8 transformer**: a second copy of the transformer stores the weight matrices of all its
  linear layers (the patch-embedding projection included) as int8, quantized from the source
  float32 values to the nearest step with one float32 scale per output channel (symmetric). Its
  other parameters (biases, normalization weights, modulation tables) keep the source float32
  values. The int8 weights are computed in float32, or, in the quants that declare it, multiplied
  with activations that are quantized to int8 per token at run time.
- The transformer graph takes patchified latent tokens and returns tokens: the patchify, the
  unpatchify and the sinusoidal timestep projection run on the host, and the patch-embedding
  convolution is applied as the equivalent linear layer. The rotary embedding keeps the upstream
  real-valued cos / sin tables, but the host builds them from per-axis base tables stored in the
  container, and the graph applies them in a pair-swap form (swap each adjacent pair, then multiply
  elementwise by the cos / sin tables).
- The VAE decoder was re-expressed as two graphs that decode one latent frame each (the first
  frame, and every later frame), with the causal convolution cache passed in and out of the graph
  instead of kept in a Python list. The host always decodes in overlapping tiles, so the output
  differs slightly from the upstream untiled decode.
- **The text encoder is referenced, not stored here.** `karume.json` references the umT5-XXL
  encoder of `google/umt5-xxl` (bit-identical in float32 to the `text_encoder` folder of this
  checkpoint), converted to int8, from the separate repository `karume-umt5-xxl` at a pinned commit
  (with the size and the SHA-256 of every part); the changes made to it are listed in that
  repository's own `NOTICE.md`.
- The umT5-XXL outputs of a fixed set of prompts (computed with the upstream encoder in bfloat16
  and stored as float32) are included as a precomputed asset, for use without the text encoder.
- The tokenizer of the checkpoint was converted into one JSON table (vocabulary, scores, added
  tokens and the whitespace set), together with lookup tables for the upstream prompt cleaning
  (evaluated from ftfy 6.3.1, the `regex` package and the Unicode 16.0.0 character database,
  including a translation of ftfy's mojibake-detection pattern) that the host uses to reproduce
  the cleaning or to reject a prompt.

No retraining and no fine-tuning were performed. The original checkpoint is not distributed here.
"""

#: Wan2.2's distribution repository name and pipeline contract (ADR 0121 decision 10 — the TS side's
#: accepted set is `WAN_TI2V_PIPELINE_NAME` / `WAN_TI2V_PIPELINE_MAJOR`
#: (`packages/models/src/wan/config.ts`)).
WAN22_REPO_NAME = "karume-wan2.2"
WAN22_PIPELINE = "wan-ti2v/1"

#: Wan2.2's VAE series (f16 — only the 2 VAE chunk graphs, no f16 DiT) and the DiT's i8 series. The
#: writers' spellings are `wan.export_vae.TI2V_SERIES_NAME` (= `VAE_SERIES["ti2v-5b"].series`) /
#: `wan.ti2v_export_dit.SERIES_NAME`.
WAN22_SERIES = "wan2.2-ti2v-5b-f16-dyn"
WAN22_I8_SERIES = "wan2.2-ti2v-5b-i8-dyn"

#: Wan2.2's quant table (ADR 0121 decision 2 — reference and practical presets, no f16 preset. There
#: are 2 practical presets, s16 and online — ADR 0132. With the VAE decoder in i8, the reference
#: preset `i8` and the practical preset `i8-a8-attn8-online-conv8` — ADR 0133 decision 6). The
#: transformer has the single storage label i8, so weights does not name it
#: ({@link complete_quant_weights} fills it into a complete mapping — writing it would duplicate
#: derivable state). The VAE decoder has two, f16 / i8, so it is written per preset
#: ({@link _WAN22_VAE_F16} / {@link _WAN22_VAE_I8}). The practical preset's `session` / `label` /
#: `description` take the same single entry as 2.1's preset of the same name (ADR 0121 decision 2
#: "the same name and bundle as 2.1's practical preset"). The reference preset's display fields have
#: their own 2.2 text, since 2.1's ("about half the f16 preset") does not hold for 2.2. The base
#: storage is set by the VAE decoder's storage (ADR 0074 §1 — the storage common to all roles:
#: presets with an f16 decoder are `f16+dit8…`, those with an i8 decoder `i8…`). The 32-latent tile
#: pair (ADR 0134 decision 5) names the same label as the decoder — a borrowing pair runs on the
#: lender's weights of the same label, so the labels within a preset do not diverge.
_WAN22_VAE_F16: Mapping[str, str] = dict.fromkeys(
    (WAN_VAE_FIRST_ROLE, WAN_VAE_NEXT_ROLE, WAN_VAE_FIRST_32_ROLE, WAN_VAE_NEXT_32_ROLE), "f16"
)
_WAN22_VAE_I8: Mapping[str, str] = dict.fromkeys(
    (WAN_VAE_FIRST_ROLE, WAN_VAE_NEXT_ROLE, WAN_VAE_FIRST_32_ROLE, WAN_VAE_NEXT_32_ROLE), "i8"
)
WAN22_QUANTS: Mapping[str, Any] = {
    "f16+dit8": {
        "weights": _WAN22_VAE_F16,
        "session": {},
        "label": "int8 transformer, f32 compute",
        "description": "Transformer weights stored as int8 (one scale per output channel) and"
        " computed in f32; the VAE stays f16.",
    },
    "f16+dit8-a8-attn8-s16": {**WAN_QUANTS["f16+dit8-a8-attn8-s16"], "weights": _WAN22_VAE_F16},
    # The online-attention preset (ADR 0132 decision 6): the same int8 weights as the s16 preset
    # (no new download — `weights` names the same f16 VAE decoder) with attention computed by
    # the online kernel, which never stores the scores (`attentionScoreStorage: "none"` — ADR 0132
    # decision 1; the name token `online` takes the slot of `s16` — ADR 0074 §3). Wan2.2 only:
    # Wan2.1's table is not changed. The default on 2026-10-23 (ADR 0132 decision 6) until the conv8
    # preset below took over the same day (`WAN22_DEFAULT_QUANT`); the s16 preset above stays as its
    # sibling.
    "f16+dit8-a8-attn8-online": {
        "weights": _WAN22_VAE_F16,
        "session": {
            "linearCompute": "a8",
            "attentionCompute": "a8",
            "attentionScoreStorage": "none",
        },
        "label": "int8 transformer, int8 activations, online attention",
        "description": "The int8 transformer with per-token int8 activations and attention"
        " computed online: the scores are never stored, so the long clips run faster on the same"
        " weights as the s16 preset.",
    },
    # The int8 VAE decoder (ADR 0133 decision 6): the reference preset — the transformer and the
    # VAE decoder in int8 storage with an empty `session`, the control of the E2E difference floor
    # — and the practical preset that adds `convCompute: "a8"` (token `conv8`) to the online
    # preset's knobs. The base storage is `i8` because the decoder roles are now int8 (ADR 0074
    # §1). The conv8 preset is the default (`WAN22_DEFAULT_QUANT` — decision 8: the decoder A/B, the
    # timing and the maintainer's visual check of 2026-10-23).
    "i8": {
        "weights": _WAN22_VAE_I8,
        "session": {},
        "label": "int8 transformer and VAE decoder, f32 compute",
        "description": "Transformer and VAE decoder weights stored as int8 (one scale per output"
        " channel) and computed in f32; the reference preset of the int8 VAE decoder.",
    },
    "i8-a8-attn8-online-conv8": {
        "weights": _WAN22_VAE_I8,
        "session": {
            "linearCompute": "a8",
            "attentionCompute": "a8",
            "attentionScoreStorage": "none",
            "convCompute": "a8",
        },
        "label": "int8 transformer and VAE, int8 activations, online attention",
        "description": "The online-attention preset with the VAE decoder's weights in int8 and"
        " its 3-D convolutions computed on per-pixel int8 activations (integer dot products) —"
        " the fastest decode.",
    },
}

#: Wan2.2's default preset — the conv8 preset since 2026-10-23 (ADR 0133 decision 8: the online
#: preset's transformer with the VAE decoder in int8 storage and its 3-D convolutions on per-pixel
#: int8 activations). The maintainer's ruling of 2026-10-23: 8 clips (2 prompts × seeds 42–45,
#: 1280×704×33, 50 steps) side by side with the online preset showed no visible difference; the
#: decoder's A/B gave relRMS 1.28e-2 at the first decoded frame against the same int8 weights on
#: f32 compute (ceiling 2.6e-2) and 3.21e-2 against the f16 decoder (the int8 weights alone
#: 2.99e-2); the VAE stage of a 33-frame clip took 87 s against 168 s (the clip 533 s against 611 s)
#: on the RTX 3080 Ti under Node + Dawn, and the download is 0.5 GB smaller. The online preset was
#: the default
#: earlier the same day (ADR 0132 decision 6 — relRMS 0.042 against the reference preset, one DiT
#: forward at 1280×704×121 21.7 s against 24.2–29.3 s for the s16 preset); the s16 preset was the
#: default from the visual ruling of 2026-10-06 (ADR 0121 addendum "results of step 8a"). Both stay
#: as siblings. The default can change with a quality ruling, so it is held in this one place only.
WAN22_DEFAULT_QUANT = "i8-a8-attn8-online-conv8"

#: Wan2.2's `pipelineConfig` (the schema is the same as 2.1's —
#: `packages/models/src/wan/config.ts`):
#:
#: - `scheduler.shift` 5.0 — the `flow_shift` of the pinned scheduler (the official 720P value — the
#:   decision of ADR 0121 addendum "Accepted sizes to the 2 official sizes").
#: - `defaults.steps` 50 / `defaults.guidance` 5.0 — the same as the official and Diffusers-version
#:   READMEs (decision 8).
WAN22_PIPELINE_CONFIG: Mapping[str, Any] = {
    "scheduler": {"shift": 5.0},
    "defaults": {"steps": 50, "guidance": 5.0},
}

#: Wan2.2's notice of modifications (Apache 2.0 §4(b)). The facts rest on ADR 0121 (decisions 2 / 3
#: / 6 / 9 / 11, addenda "Results of step 1", "Results of step 4", "Results of step 5", "Ruling on
#: the plan of step 9 and the results of step 9a" (the rewrite of the VAE encoder — the 5 of
#: `wan.vae_encoder_patch`)) and ADR 0122 decision 8 (umT5 is described as the original encoder, and
#: this checkpoint's bf16 `text_encoder` as its rounding). That the tokenizer files are the same in
#: the 2 checkpoints was confirmed by matching sha256 of the 4 files in `tokenizer/` at the 2 pinned
#: revisions (2026-10-06).
#:
#: MUST: the text matches the contents of the distribution (the same reason as
#: {@link WAN_NOTICE_MARKDOWN}). No place copied from 2.1's text where the facts differ (the f16
#: transformer, the f16 preset, a single VAE exit) may remain.
WAN22_NOTICE_MARKDOWN = """# NOTICE

This repository redistributes a modified form of the Wan2.2 TI2V 5B checkpoint listed in
`README.md` (Wan-AI, licensed under the Apache License, Version 2.0 — see `LICENSE.md`). The
following changes were made:

- The weights were converted into the Karume container format (a `.krm` part sequence whose first
  part carries the graph and model descriptors).
- **int8 transformer**: the transformer is distributed only in this form. The weight matrices of
  all its linear layers (the patch-embedding projection included) are stored as int8, quantized
  from the source float32 values to the nearest step with one float32 scale per output channel
  (symmetric). Its other parameters (biases, normalization weights, modulation tables) keep the
  source float32 values. The int8 weights are computed in float32, or, in the quants that declare
  it, multiplied with activations that are quantized to int8 per token at run time. No float16 or
  float32 copy of the transformer is distributed.
- **f16 VAE decoder**: every parameter of the VAE decoder was rounded from the source float32
  value to the nearest float16 value. Weight matrices and convolution kernels are stored as
  float16; the other parameters (biases, normalization weights) keep the rounded values in float32
  storage. Computation runs in float32.
- **int8 VAE decoder**: a second copy of the VAE decoder in which the weights of every
  convolution and linear layer are stored as int8, quantized per output channel with one float32
  scale, from the float16-rounded values above; the other parameters keep those values in float32
  storage. The int8 weights are computed in float32, or, in the quant that declares it, the 3-D
  convolutions multiply them with activations quantized to int8 per pixel at run time.
- The transformer graph takes patchified latent tokens (48 latent channels) and returns tokens:
  the patchify, the unpatchify and the sinusoidal timestep projection run on the host, and the
  patch-embedding convolution is applied as the equivalent linear layer. The rotary embedding keeps
  the upstream real-valued cos / sin tables, but the host builds them from per-axis base tables
  stored in the container, and the graph applies them in a pair-swap form (swap each adjacent
  pair, then multiply elementwise by the cos / sin tables).
- The transformer graph is the one graph for both text-to-video and image-to-video: besides the
  timestep of the generated tokens it takes a second timestep for conditioning tokens and a boolean
  condition mask, embeds each timestep separately, and selects every modulation per token between
  the two. In text-to-video the mask is all false and the second timestep equals the first. In
  image-to-video the mask is true for the tokens of the first latent frame, which the host
  replaces with the encoded conditioning image, and the second timestep is 0.
- The VAE decoder (Wan2.2-VAE) was re-expressed as two graphs that decode one latent frame each
  (the first frame, and every later frame), with the causal convolution cache passed in and out of
  the graph instead of kept in a Python list, and its upsampling shortcuts written as copies and
  depth-to-space reshapes. The graphs return the decoder output in the upstream patchified space
  (`patch_size` 2, 12 channels); the host decodes in overlapping tiles of 16 latent pixels (or 32
  when the application asks for it — a second pair of the same two graphs traced at that tile, which
  holds no weights of its own and borrows the 16-tile pair's), blends them in that space and then
  unpatchifies to RGB, so the output differs slightly from the upstream untiled decode.
- **f16 VAE encoder for image-to-video**: only the part of the VAE encoder (Wan2.2-VAE) that
  encodes one conditioning image (the first latent frame) is included, re-expressed as three graphs
  with a symbolic latent height and width, split around its middle attention block: the layers
  before that block, the attention on the flattened sequence of latent positions (its 1×1
  convolutions applied as linear layers), and the layers after it up to `quant_conv`, whose output
  is cut to its first half (the latent mean). The graphs take the image already patchified by the
  host (`patch_size` 2, 12 channels). On one frame, each causal convolution with a temporal kernel
  of 3 multiplies only its last temporal slice with real data, so only that slice is stored; the
  temporal downsampling convolutions, which do not run on the first frame, are not included; the
  averaging shortcuts are written in their closed form for one frame. Every included parameter
  was rounded from the source float32 value to the nearest float16 value and is stored as in the
  VAE decoder; computation runs in float32.
- **The text encoder is referenced, not stored here.** `karume.json` references the umT5-XXL
  encoder of `google/umt5-xxl` (this checkpoint's `text_encoder` folder holds the same encoder
  rounded to bfloat16), converted to int8, from the separate repository `karume-umt5-xxl` at a
  pinned commit (with the size and the SHA-256 of every part); the changes made to it are listed in
  that repository's own `NOTICE.md`.
- The umT5-XXL outputs of a fixed set of prompts are included as a precomputed asset, for use
  without the text encoder. They were computed with the umT5-XXL encoder of the Wan2.1 T2V 1.3B
  checkpoint (the same encoder) in bfloat16 and are stored as float32.
- The tokenizer of the Wan2.1 T2V 1.3B checkpoint (its files are identical to the `tokenizer`
  folder of this checkpoint) was converted into one JSON table (vocabulary, scores, added tokens
  and the whitespace set), together with lookup tables for the upstream prompt cleaning (evaluated
  from ftfy 6.3.1, the `regex` package and the Unicode 16.0.0 character database, including a
  translation of ftfy's mojibake-detection pattern) that the host uses to reproduce the cleaning or
  to reject a prompt.

No retraining and no fine-tuning were performed. The original checkpoint is not distributed here.
"""


@dataclass(frozen=True)
class WanGeneration:
    """The distribution's generation table — the facts that differ per generation, in one place.

    The plan function {@link wan_plan} is a single one; the placement table, the storage requirement
    / prohibition tables and the weights declaration are derived from this table's transformer
    storage labels and whether it has the encoder (each generation looks up its own roles from the
    all-generation tables {@link WAN_STORAGE_REQUIREMENTS} / {@link WAN_STORAGE_FORBIDDEN} /
    {@link WAN_WEIGHTS}).
    """

    #: The distribution repository name (ADR 0092 decisions 1 / 2 — one repository per family, a
    #: separate one per generation).
    repo_name: str
    #: The pipeline contract (ADR 0041 §2 — per model).
    pipeline: str
    #: The default model (the Pipeline's `default_model`, used when the plan's `model` is omitted).
    default_model: str
    #: The f16 series (the VAE decoder's 2 chunk graphs, the encoder's 3 graphs for a generation
    #: that ships the encoder, and the f16 DiT for a generation with f16 among its storage labels).
    series: str
    #: The DiT's i8 series (every generation has one — the weights of the reference and practical
    #: presets).
    i8_series: str
    #: The transformer's storage labels (in the order of the manifest's weights — the order of
    #: {@link WAN_WEIGHTS}).
    transformer_labels: tuple[str, ...]
    #: The storage labels of the VAE decoder (2 graphs) (the same ordering rule). i8 is taken from
    #: the {@link i8_series} series (ADR 0133 decision 7).
    vae_labels: tuple[str, ...]
    #: The latent tiles of the VAE decoder pairs. The first is the default-tile pair that holds the
    #: weights (its roles have no suffix — 32 for Wan2.1, 16 for Wan2.2 — the tiles of the writer's
    #: `wan.export_vae.VAE_SERIES`); the rest are pairs that borrow weights from it
    #: ({@link WAN_VAE_SHARED_LABEL_ROLES} — ADR 0134, one pair per storage label).
    vae_tiles: tuple[int, ...]
    #: Whether the VAE encoder of the I2V conditioning image (the 3 graphs of
    #: {@link WAN_VAE_ENCODER_ROLES} — f16 series) is shipped. Wan2.2 only (ADR 0121 decision 11).
    #: The all-generation table {@link WAN_WEIGHTS} also reaches the Wan2.1 plan, so without
    #: narrowing it here encoder entries would leak into the Wan2.1 manifest (2.1 is a T2V-only
    #: checkpoint, and the 2.1 distribution is kept byte-identical).
    vae_encoder: bool
    #: The model name (a key of `SOURCES`) whose pin gives the provenance of the 2 text assets
    #: (embeddings, tokenizer). The provenance of the DiT / VAE containers is checked against the
    #: pin of the plan's model name — the 2 fields are separate because Wan2.2's assets were made
    #: from the umT5 of the Wan2.1 checkpoint and pass the check while stating that provenance (ADR
    #: 0121 decision 9).
    text_model: str
    #: The quant table (preset name → weights / session / display fields —
    #: {@link complete_quant_weights} fills weights into a complete mapping) and the default preset.
    quants: Mapping[str, Any]
    default_quant: str
    #: Pipeline-owned settings (the hub passes them through — ADR 0041 §2).
    pipeline_config: Mapping[str, Any]
    #: The notice of modifications (Apache 2.0 §4(b) — `NOTICE.md` at the distribution repository's
    #: root).
    notice: str
    #: The commands that rebuild the DiT / VAE containers (the hint when the shared checks
    #: {@link assert_rope_base} / {@link assert_vae_chunk_pair} fail — they name the writer of that
    #: generation's series).
    dit_writer: str
    vae_writer: str
    #: The card's generation table (it also holds the set of shipped models — shared by the card's
    #: attribution check and the plan's check).
    card: WanCard

    @property
    def models(self) -> tuple[str, ...]:
        """The models this generation's distribution ships (the plan's check — not every model of
        the sources table passes)."""
        return self.card.models

    @property
    def transformer_roles(self) -> tuple[str, ...]:
        """The placement roles of the transformer containers (one per storage label)."""
        return tuple(WAN_TRANSFORMER_LABEL_ROLES[label] for label in self.transformer_labels)

    @property
    def vae_decoder_roles(self) -> tuple[str, ...]:
        """The placement roles of the VAE decoder containers (per storage label of first, then per
        storage label of next; the borrowing pairs follow in the same order, by tile)."""
        pairs = [
            (WAN_VAE_FIRST_LABEL_ROLES, WAN_VAE_NEXT_LABEL_ROLES),
            *(WAN_VAE_SHARED_LABEL_ROLES[tile] for tile in self.vae_tiles[1:]),
        ]
        return tuple(
            role
            for first, following in pairs
            for role in (
                *(first[label] for label in self.vae_labels),
                *(following[label] for label in self.vae_labels),
            )
        )

    @property
    def vae_encoder_roles(self) -> tuple[str, ...]:
        """The placement roles of the VAE encoder containers (empty for a generation that does not
        ship it)."""
        return WAN_VAE_ENCODER_ROLES if self.vae_encoder else ()

    @property
    def container_roles(self) -> tuple[str, ...]:
        """The placement roles that have a container (in the order of {@link WAN_CONTAINER_ROLES} —
        the order of the generation stages)."""
        return (
            *self.vae_encoder_roles,
            WAN_TEXT_ENCODER_ROLE,
            *self.transformer_roles,
            *self.vae_decoder_roles,
        )

    @property
    def storage_requirements(self) -> Mapping[str, str]:
        """The storage requirements of this generation's roles (looked up from the all-generation
        table)."""
        return {role: WAN_STORAGE_REQUIREMENTS[role] for role in self.container_roles}

    @property
    def storage_forbidden(self) -> Mapping[str, tuple[str, ...]]:
        """The storage prohibition table of this generation's roles (looked up from the
        all-generation table)."""
        return {role: WAN_STORAGE_FORBIDDEN[role] for role in self.container_roles}

    @property
    def weights(self) -> Mapping[str, Mapping[str, WeightFiles]]:
        """The weights declaration (only the labels pointing at this generation's placement roles —
        for the transformer only this generation's storage labels, the encoder only for a generation
        that ships it. Declaring a missing role would let a preset that selects it be built)."""
        roles = set(self.container_roles)
        declared = {
            name: {label: files for label, files in labels.items() if files.file in roles}
            for name, labels in WAN_WEIGHTS.items()
        }
        return {name: labels for name, labels in declared.items() if labels}


#: Wan2.1 T2V 1.3B (`--pipeline wan` → `karume-wan2.1`). The transformer has 2 containers, f16 / i8.
WAN21 = WanGeneration(
    repo_name=WAN_REPO_NAME,
    pipeline=WAN_PIPELINE,
    default_model=DEFAULT_MODEL,
    series=WAN_SERIES,
    i8_series=WAN_I8_SERIES,
    transformer_labels=("f16", "i8"),
    vae_labels=("f16",),
    vae_tiles=(32,),
    vae_encoder=False,
    text_model=DEFAULT_MODEL,
    quants=WAN_QUANTS,
    default_quant=WAN_DEFAULT_QUANT,
    pipeline_config=WAN_PIPELINE_CONFIG,
    notice=WAN_NOTICE_MARKDOWN,
    dit_writer="python -m wan.export_dit",
    vae_writer="python -m wan.export_vae",
    card=WAN21_CARD,
)

#: Wan2.2 TI2V 5B (`--pipeline wan-ti2v` → `karume-wan2.2`, model `ti2v-5b`). Differences from 2.1:
#:
#: - The transformer is **i8 only** (no f16 DiT — ADR 0121 decision 2). The DiT is a single
#:   I2V-capable graph (decision 3 — 7 inputs; assembly looks only at the context input and the RoPE
#:   base table, the same checks as 2.1).
#: - The VAE is the 2 chunk graphs of Wan2.2-VAE (decision 6 — 48 latent channels, the exit in
#:   patchified space). The decoder has 2 of each, f16 / i8 (i8 from the same i8 series as the DiT
#:   — ADR 0133 decision 7). In addition, a 32-latent tile pair (ADR 0134) is shipped per storage
#:   label — it holds no weight bytes and borrows the weights of the 16 pair with the same label at
#:   the same series root (no extra weight download).
#: - The 3 graphs of the VAE encoder for the I2V conditioning image are shipped (decision 11, step
#:   9a — the same f16 series as the decoder). Every preset takes the single f16 one (even for
#:   T2V-only use — the ruling of the addendum "Ruling on the plan of step 9 and the results of step
#:   9a").
#: - The 2 text assets are **the very files of the 2.1 series**, and their provenance is the Wan2.1
#:   pin (decision 9 — {@link WanGeneration.text_model}). The provenance of the DiT / VAE containers
#:   is checked against the Wan2.2 pin.
#: - umT5 is the same cross-repository reference to `karume-umt5-xxl` as in 2.1.
WAN22 = WanGeneration(
    repo_name=WAN22_REPO_NAME,
    pipeline=WAN22_PIPELINE,
    default_model=WAN22_CARD.models[0],
    series=WAN22_SERIES,
    i8_series=WAN22_I8_SERIES,
    transformer_labels=("i8",),
    vae_labels=("f16", "i8"),
    vae_tiles=(16, 32),
    vae_encoder=True,
    text_model=WAN22_TEXT_MODEL,
    quants=WAN22_QUANTS,
    default_quant=WAN22_DEFAULT_QUANT,
    pipeline_config=WAN22_PIPELINE_CONFIG,
    notice=WAN22_NOTICE_MARKDOWN,
    dit_writer="python -m wan.ti2v_export_dit write",
    vae_writer="python -m wan.export_vae --model ti2v-5b",
    card=WAN22_CARD,
)


@dataclass(frozen=True)
class WanSources:
    """The assembly inputs = 3 series directories (the f16 series (VAE decoder; in 2.2 also the
    VAE encoder; in 2.1 the f16 DiT), the DiT's i8 series, the text-embeddings series), the umT5
    container (i8 series) and the tokenizer asset. For every generation the text embeddings and the
    tokenizer asset point at the Wan2.1 series (ADR 0121 decision 9)."""

    series: Path
    i8_series: Path
    text_embeds: Path
    #: The representative path of the umT5 container (series `umt5-xxl-i8-dyn` —
    #: `wan.umt5_distribution.umt5_container`).
    text_encoder: Path
    #: The tokenizer asset (series `wan2.1-umt5-tokenizer` — the writer is `wan.umt5_tokenizer`).
    tokenizer: Path


def wan_sources(series_dir: Path, generation: WanGeneration = WAN21) -> WanSources:
    """Look up the generation's inputs from the parent directory of the series
    (`outputs/series/`)."""
    return WanSources(
        series=series_dir / generation.series,
        i8_series=series_dir / generation.i8_series,
        text_embeds=series_dir / WAN_TEXT_EMBEDS_SERIES / WAN_TEXT_EMBEDS_FILE,
        text_encoder=umt5_container(series_dir),
        tokenizer=series_dir / umt5_tokenizer.SERIES_NAME / umt5_tokenizer.ASSET_FILE,
    )


def wan_placements(sources: WanSources, generation: WanGeneration = WAN21) -> dict[str, Path]:
    """Placement role → source file. The output paths live in {@link WAN_OUTPUT_PATHS}.

    Anything not in this table does not reach the output (the goldens `io.*` / `reference.*` /
    `vae_*.safetensors` / `pipeline_steps.*` in the series are dropped this way — and so is the
    encoder's golden `vae_encoder.*`). The transformer and the VAE decoder take only the containers
    of the generation's storage labels (f16 from the f16 series, i8 from the i8 series); the VAE
    encoder only for a generation that ships it (f16 series).
    """
    transformers = {
        WAN_TRANSFORMER_F16_ROLE: sources.series / WAN_TRANSFORMER_ROLE / WAN_MODEL_FILE,
        WAN_TRANSFORMER_I8_ROLE: sources.i8_series / WAN_TRANSFORMER_ROLE / WAN_MODEL_FILE,
    }
    decoders = {
        WAN_VAE_FIRST_ROLE: sources.series / WAN_VAE_FIRST_ROLE / WAN_MODEL_FILE,
        WAN_VAE_FIRST_I8_ROLE: sources.i8_series / WAN_VAE_FIRST_ROLE / WAN_MODEL_FILE,
        WAN_VAE_NEXT_ROLE: sources.series / WAN_VAE_NEXT_ROLE / WAN_MODEL_FILE,
        WAN_VAE_NEXT_I8_ROLE: sources.i8_series / WAN_VAE_NEXT_ROLE / WAN_MODEL_FILE,
        # A borrowing pair lives at the same series root as its lender (ADR 0134 decision 5).
        WAN_VAE_FIRST_32_ROLE: sources.series / WAN_VAE_FIRST_32_ROLE / WAN_MODEL_FILE,
        WAN_VAE_FIRST_32_I8_ROLE: sources.i8_series / WAN_VAE_FIRST_32_ROLE / WAN_MODEL_FILE,
        WAN_VAE_NEXT_32_ROLE: sources.series / WAN_VAE_NEXT_32_ROLE / WAN_MODEL_FILE,
        WAN_VAE_NEXT_32_I8_ROLE: sources.i8_series / WAN_VAE_NEXT_32_ROLE / WAN_MODEL_FILE,
    }
    return {
        **{role: sources.series / role / WAN_MODEL_FILE for role in generation.vae_encoder_roles},
        WAN_TEXT_ENCODER_ROLE: sources.text_encoder,
        **{role: transformers[role] for role in generation.transformer_roles},
        **{role: decoders[role] for role in generation.vae_decoder_roles},
        WAN_TEXT_EMBEDS_ROLE: sources.text_embeds,
        WAN_TOKENIZER_ROLE: sources.tokenizer,
    }


def wan_repo_name(_model: str, generation: WanGeneration = WAN21) -> str:
    """The distribution repository name (one repository per family and a separate one per
    generation, so it is the same for every model of the generation — ADR 0092 decisions 1 / 2)."""
    return generation.repo_name


def assert_rope_base(container: Path, writer: str) -> None:
    """Check that the `transformer` container declares the RoPE base table asset (the payload is not
    read).

    `writer` is the generation's DiT writer shown in the rebuild hint
    ({@link WanGeneration.dit_writer}).

    MUST: fail at assembly. A DiT without the base table (a separate static-shape export, a writer
    from before the asset was added) plugged into the transformer entry passes assembly and
    `verify_dist`, and the user's `fromPretrained` hits "asset 'rope_base' is missing" only after
    downloading the weights.
    """
    try:
        declared = read_asset_declarations(container)
    except ContainerReadError as cause:
        raise DistError(f"{container}: {cause}") from cause
    asset = declared.get(WAN_ROPE_BASE_ASSET)
    if asset is None:
        raise DistError(
            f"{container}: asset '{WAN_ROPE_BASE_ASSET}' is missing (declared:"
            f" {sorted(declared)}) — a DiT without the RoPE base table is not shipped (rebuild it"
            f" with `{writer}`)"
        )
    if asset[0] != WAN_ROPE_BASE_ROLE:
        raise DistError(
            f"{container}: asset '{WAN_ROPE_BASE_ASSET}' has role {asset[0]!r}"
            f" (expected {WAN_ROPE_BASE_ROLE!r})"
        )


def assert_shared_rope_base(containers: Sequence[Path]) -> None:
    """Check that the RoPE base tables of the `transformer` containers (one per storage label — f16
    / i8) are byte-identical.

    MUST: fail at assembly. Each container carries one base table, and the TS side reads it from the
    container of the chosen preset. Shipped while they disagree, one preset alone would run with
    RoPE of another geometry: loading and execution pass, and only the video quietly breaks (the
    same reason as anima's `assert_shared_rope_base`). An error in the base table of the i8
    container cannot be caught by the on-machine A/B check (ADR 0120 decision 4), which compares i8
    containers with each other.

    A generation with a single container (Wan2.2 — i8 only) has nothing to compare against and
    always passes. The geometry of its base table is not checked at assembly but by the GPU checks —
    the 2-step end-to-end run of ADR 0121 step 6 runs on the product path (the host builds the RoPE
    tables from the container's base table — `packages/models/src/wan/family.ts`) and is compared
    with the upstream reference on CPU.
    """
    digests: dict[Path, str] = {}
    for path in containers:
        try:
            payload = read_asset(path, WAN_ROPE_BASE_ASSET)
        except ContainerReadError as cause:
            raise DistError(f"{path}: {cause}") from cause
        digests[path] = hashlib.sha256(payload).hexdigest()
    if len(set(digests.values())) != 1:
        listing = "\n".join(f"  {digest}  {path}" for path, digest in digests.items())
        raise DistError(
            f"asset '{WAN_ROPE_BASE_ASSET}' is not byte-identical across the transformer"
            " containers — they cannot ship with the same geometry. Which one is right cannot be"
            f" decided here, so assembly stops:\n{listing}"
        )


def dit_context(container: Path) -> tuple[int, int]:
    """`(rows, width)` of the DiT's context input `[1, rows, width]` (static dimensions — fails if
    symbolic)."""
    shape = graph_inputs(ir_graph(container), container).get(WAN_DIT_CONTEXT_INPUT)
    if (
        shape is None
        or len(shape) != 3
        or not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape)
    ):
        raise DistError(
            f"{container}: graph input '{WAN_DIT_CONTEXT_INPUT}' is not a static [1, rows, width]"
            f" ({shape!r})"
        )
    return shape[1], shape[2]


def assert_vae_chunk_pair(first: Path, following: Path, writer: str) -> None:
    """Check that `vae_decoder_first` / `vae_decoder_next` are chunk graphs of the same pair.

    `writer` is the generation's VAE writer shown in the rebuild hint
    ({@link WanGeneration.vae_writer}).

    The rule is the same as TS's `wanVaeChunkLayout` (`packages/models/src/wan/vae-chunks.ts`): the
    latent input has the same shape in both (= the same tile edge), and first's cache inputs are a
    subsequence of next's cache inputs (same names, same shapes, same order).

    MUST: fail at assembly. The writer writes the 2 graphs to separate containers, so a series where
    only one was rebuilt (a partial update with `--target`, a run of an old writer that died midway)
    settles into the distribution while passing every per-container check, and the user's
    `fromPretrained` rejects it only at admission. A pair with the same shapes whose contents come
    from different builds cannot be told apart by shape (only the GPU check of the chunk sequence
    catches it).
    """
    first_inputs = graph_inputs(ir_graph(first), first)
    next_inputs = graph_inputs(ir_graph(following), following)
    for path, inputs in ((first, first_inputs), (following, next_inputs)):
        if next(iter(inputs), None) != WAN_VAE_LATENT_INPUT:
            raise DistError(
                f"{path}: the first graph input is not '{WAN_VAE_LATENT_INPUT}' ({list(inputs)})"
            )
    if first_inputs[WAN_VAE_LATENT_INPUT] != next_inputs[WAN_VAE_LATENT_INPUT]:
        raise DistError(
            f"{first} / {following}: the latent input shape differs between first"
            f" {first_inputs[WAN_VAE_LATENT_INPUT]} and next {next_inputs[WAN_VAE_LATENT_INPUT]}"
            " — 2 graphs built with different tile edges do not form a pair (rebuild both with"
            f" `{writer}`)"
        )
    next_caches = [name for name in next_inputs if name != WAN_VAE_LATENT_INPUT]
    cursor = 0
    for name, shape in first_inputs.items():
        if name == WAN_VAE_LATENT_INPUT:
            continue
        if next_inputs.get(name) != shape:
            raise DistError(
                f"{first}: cache input '{name}' {shape} is not in next with the same shape"
                f" ({next_inputs.get(name)}) — chunk graphs of different pairs do not form a pair"
            )
        position = next_caches.index(name)
        if position < cursor:
            raise DistError(f"{first}: the order of the cache inputs differs from next ('{name}')")
        cursor = position + 1


def assert_shared_decoder(borrower: Path, lender: Path) -> None:
    """Check that the container of a borrowing pair (ADR 0134 decision 1) borrows all its weights
    from the lender's container.

    Three things are checked: the borrower holds not one byte outside part 0 (the 2 documents);
    every initializer is a shared declaration; and the lender holds each tensor key a shared
    declaration names (in IR v2, the initializer's name itself) as a real entry (a supply in the
    binding table), with the same dtype and shape of the value as the lender. The borrower's storage
    declaration does not survive in the container (a shared declaration has no supply), so the
    storage the borrower runs on is the storage of the lender's binding table, and {@link wan_plan}
    applies the storage checks to the lender's container.

    MUST: fail at assembly. In a series where only the lender was rebuilt, a key the borrower names
    disappears or changes shape — the runtime's sharing check does not run until a session is built
    (after the user has downloaded the weights).
    """
    carried = sum(part.stat().st_size for part in component_parts(borrower)[1:])
    if carried:
        raise DistError(
            f"{borrower}: holds {carried} bytes outside the descriptors — a borrowing pair holds"
            " no weight bytes (rebuild it with `python -m wan.export_vae … --shared`)"
        )
    graph = ir_graph(borrower)
    opened = open_container(lender)
    lent = ir_graph(lender, opened)
    supplied = {name for bound in opened.graphs.values() for name in bound.supplies}
    initializers = graph.get("initializers") or {}
    owned = sorted(
        name
        for name, entry in initializers.items()
        if not (isinstance(entry, dict) and entry.get("shared") is True)
    )
    if owned:
        raise DistError(
            f"{borrower}: initializers that are not shared declarations {owned[:4]} (all of them"
            f" must name the lender {lender})"
        )
    values, lent_values = graph.get("values") or {}, lent.get("values") or {}
    for name in sorted(initializers):
        if name not in supplied:
            raise DistError(
                f"{borrower}: the shared initializer '{name}' has no real entry in the lender"
                f" {lender} (out of step with a rebuilt lender)"
            )
        if values.get(name) != lent_values.get(name):
            raise DistError(
                f"{borrower}: shared initializer '{name}' is {values.get(name)} — differs from the"
                f" lender {lender}'s {lent_values.get(name)}"
            )


def _safetensors_header(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Read only the safetensors header (tensor declarations, metadata — not the body, no torch)."""
    try:
        with path.open("rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(length))
    except (OSError, struct.error, ValueError) as cause:
        raise DistError(f"{path}: cannot read the safetensors header — {cause}") from cause
    if not isinstance(header, dict):
        raise DistError(f"{path}: the safetensors header is not an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise DistError(f"{path}: the safetensors metadata is not an object")
    return header, metadata


def assert_text_embeds(path: Path, model: str, context: tuple[int, int]) -> None:
    """Check that the text-embeddings asset fits the upstream pin, the fixed prompt table and the
    DiT's context input.

    MUST: fail at assembly. The asset is a table, neither a graph nor weights, so `verify_dist`'s
    container checks do not see it, and an asset made with umT5 of another revision, an asset that
    disagrees with the prompt table or an asset of another width would settle into the distribution
    as is. The reader (TS's `parseWanTextEmbeds`) looks only at the shapes, so a provenance mismatch
    stays silent on the user's side as "a video generated with a different context".

    Six things are checked: the metadata has exactly one key (the writer's MUST — the premise of
    byte identity); the provenance (repo / revision) matches the pin in {@link SOURCES}; the encoder
    dtype and the dependency versions are the fixed values of decision 4
    ({@link WAN_TEXT_ENCODER_DTYPE} / {@link WAN_TEXT_EMBEDS_VERSIONS} — the grounds of what the
    NOTICE and the card say); the prompt list (names, roles, original texts) matches
    {@link FIXED_PROMPTS} (the card draws its text from the same table); each row's normalized
    string `normalized` is a non-empty string, and no original or normalized string belongs to 2
    rows (TS's `parseWanTextEmbeds` rejects it by the same rule — fail before shipping); the tensors
    are `F32 [tokens, width]` with `tokens` equal to the metadata's token count and at most the DiT
    context's row count, and `width` the context's width. `normalize(prompt) == normalized` is not
    checked (this module does not load ftfy — the writer's tests check it).
    """
    assert_component_present(path)
    tensors, metadata = _safetensors_header(path)
    if set(metadata) != {WAN_TEXT_EMBEDS_METADATA_KEY}:
        raise DistError(
            f"{path}: the metadata keys {sorted(metadata)} are not exactly"
            f" {WAN_TEXT_EMBEDS_METADATA_KEY}"
        )
    try:
        meta = json.loads(metadata[WAN_TEXT_EMBEDS_METADATA_KEY])
    except ValueError as cause:
        raise DistError(f"{path}: the metadata cannot be read as JSON — {cause}") from cause
    if not isinstance(meta, dict):
        raise DistError(f"{path}: the metadata is not an object")
    source = SOURCES[model]
    expected_source = {"repo": source.repo, "revision": source.revision}
    if meta.get("source") != expected_source:
        raise DistError(
            f"{path}: the embeddings' provenance {meta.get('source')!r} differs from the upstream"
            f" pin {expected_source!r} — an asset made with umT5 of another checkpoint is not"
            " shipped (`python -m wan.text_embeds`)"
        )
    encoder = meta.get("text_encoder")
    encoder_dtype = encoder.get("dtype") if isinstance(encoder, dict) else None
    if encoder_dtype != WAN_TEXT_ENCODER_DTYPE:
        raise DistError(
            f"{path}: the dtype {encoder_dtype!r} of the umT5 that made the embeddings is not"
            f" decision 4's {WAN_TEXT_ENCODER_DTYPE!r} — an asset that disagrees with the NOTICE"
            " and the card is not shipped"
        )
    versions = meta.get("versions")
    pinned = (
        {name: versions.get(name) for name in WAN_TEXT_EMBEDS_VERSIONS}
        if isinstance(versions, dict)
        else None
    )
    if pinned != WAN_TEXT_EMBEDS_VERSIONS:
        raise DistError(
            f"{path}: the versions that made the embeddings {pinned!r} differ from decision 4's"
            f" fixed {dict(WAN_TEXT_EMBEDS_VERSIONS)!r} — the normalization and the embedding"
            " path change between versions"
        )
    prompts = meta.get("prompts")
    if not isinstance(prompts, list) or not all(isinstance(entry, dict) for entry in prompts):
        raise DistError(f"{path}: the metadata's prompts is not a list")
    declared = [(entry.get("name"), entry.get("role"), entry.get("prompt")) for entry in prompts]
    expected = [(prompt.name, prompt.role, prompt.text) for prompt in FIXED_PROMPTS]
    if declared != expected:
        raise DistError(
            f"{path}: the prompt list {[name for name, _, _ in declared]} does not match the fixed"
            f" prompt table (wan.prompts.FIXED_PROMPTS — {[name for name, _, _ in expected]}) in"
            " names, roles and original texts — the card draws its text from the table, so it is"
            " not shipped while they disagree"
        )
    owners: dict[str, str] = {}
    for entry in prompts:
        name, normalized = entry["name"], entry.get("normalized")
        if not isinstance(normalized, str) or not normalized:
            raise DistError(
                f"{path}: '{name}'s normalized is not a non-empty string ({normalized!r}) — TS's"
                " parseWanTextEmbeds rejects it on the user's side"
            )
        # The original and the normalized string of the same row may be equal (folded into one by
        # the set — as in TS).
        for text in {entry["prompt"], normalized}:
            owner = owners.setdefault(text, name)
            if owner != name:
                raise DistError(
                    f"{path}: the string of '{name}' is the same as '{owner}''s — which embedding"
                    " to use is undecidable (TS's parseWanTextEmbeds rejects it)"
                )
    rows, width = context
    if sorted(tensors) != sorted(name for name, _, _ in expected):
        raise DistError(
            f"{path}: the tensors {sorted(tensors)} do not correspond to the metadata's prompts"
        )
    for entry in prompts:
        name, tokens = entry["name"], entry.get("tokens")
        tensor = tensors[name]
        shape = tensor.get("shape") if isinstance(tensor, dict) else None
        if (
            not isinstance(tensor, dict)
            or tensor.get("dtype") != "F32"
            or shape != [tokens, width]
            or not isinstance(tokens, int)
            or not 0 < tokens <= rows
        ):
            raise DistError(
                f"{path}: '{name}' is not F32 [{tokens}, {width}] (valid length 1 to {rows})"
                f" ({tensor!r}) — it does not fit the DiT's context input [1, {rows}, {width}]"
            )


def assert_umt5_tokenizer(path: Path, model: str, rows: int) -> None:
    """Check that the tokenizer asset fits the format TS reads, the upstream pin, the dependency
    versions and the row count of the DiT's context.

    MUST: fail at assembly. The asset is a table, so `verify_dist`'s container checks do not see it,
    and the reader (`parseWanTokenizerAsset`) looks only at the format and the shapes — a tokenizer
    of another revision, or preprocessing tables built with another ftfy version, stay silent on the
    user's side as "a different id sequence".

    Four things are checked: the format version (`wan.umt5_tokenizer.ASSET_FORMAT`); the provenance
    (repo / revision / subfolder) matches the pin in {@link SOURCES}; the versions
    ({@link WAN_TOKENIZER_VERSIONS}) match the pins of the `wan` group; and `maxLength` (the upper
    bound on the token count TS accepts) is at most the DiT context's row count (the output of the
    GPU path is packed up to that row count — ADR 0119 decision 4).
    """
    assert_component_present(path)
    try:
        asset = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as cause:
        raise DistError(f"{path}: the tokenizer asset cannot be read as JSON — {cause}") from cause
    if not isinstance(asset, dict):
        raise DistError(f"{path}: the tokenizer asset is not an object")
    if asset.get("format") != umt5_tokenizer.ASSET_FORMAT:
        raise DistError(
            f"{path}: the format {asset.get('format')!r} is not {umt5_tokenizer.ASSET_FORMAT!r}"
            " — TS's parseWanTokenizerAsset does not read a version it does not know"
            " (`python -m wan.umt5_tokenizer`)"
        )
    source = SOURCES[model]
    expected_source = {
        "repo": source.repo,
        "revision": source.revision,
        "subfolder": WAN_TOKENIZER_SUBFOLDER,
    }
    if asset.get("source") != expected_source:
        raise DistError(
            f"{path}: the tokenizer's provenance {asset.get('source')!r} differs from the"
            f" upstream pin {expected_source!r} — a tokenizer of another checkpoint is not shipped"
        )
    versions = asset.get("versions")
    pinned = (
        {name: versions.get(name) for name in WAN_TOKENIZER_VERSIONS}
        if isinstance(versions, dict)
        else None
    )
    expected_versions = {name: WAN_TEXT_EMBEDS_VERSIONS[name] for name in WAN_TOKENIZER_VERSIONS}
    if pinned != expected_versions:
        raise DistError(
            f"{path}: the versions that built the tokenizer asset {pinned!r} differ from the"
            f" `wan` group's pins {expected_versions!r} — the id sequence and the preprocessing"
            " tables change between versions"
        )
    max_length = asset.get("maxLength")
    if (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or not 0 < max_length <= rows
    ):
        raise DistError(
            f"{path}: maxLength {max_length!r} does not fit the DiT context's row count 1 to {rows}"
            " — the output of the GPU path cannot be packed into the context"
        )


def wan_plan(
    sources: WanSources, model: str | None = None, generation: WanGeneration = WAN21
) -> ModelPlan:
    """Build the plan for one model of the generation (every check and read happens here — nothing
    is written).

    `sources` is what {@link wan_sources} looked up for the same generation. Without `model`, the
    generation's default model.
    """
    model = generation.default_model if model is None else model
    name = generation.card.generation
    assert_model_name(model)
    # The check closes over the generation's model table, not over the sources table (`SOURCES` — it
    # has rows of both generations). The distribution repository name is one per generation whatever
    # the model ({@link wan_repo_name}), so letting every model of the table through would let a
    # distribution stating another generation's provenance replace that generation's distribution.
    if model not in generation.models:
        raise DistError(
            f"unknown {name} model {model!r} (known: {' / '.join(generation.models)}) —"
            f" the {name} distribution ({generation.repo_name}) ships only {name} models"
        )
    placements = wan_placements(sources, generation)
    upstream = SOURCES[model]
    requirements = generation.storage_requirements
    forbidden = generation.storage_forbidden
    for role in generation.container_roles:
        container = placements[role]
        assert_component_present(container)
        # A borrowing pair (ADR 0134) has no binding table, so its storage is checked on the
        # lender's container (that the borrowing pair borrows everything from the lender is checked
        # by {@link assert_shared_decoder} below).
        stored = placements[WAN_VAE_LENDER_ROLES.get(role, role)]
        assert_storage(role, stored, requirements)
        assert_storage_absent(role, stored, forbidden)
        if role == WAN_TEXT_ENCODER_ROLE:
            # The upstream of the umT5 container is not Wan but the original `google/umt5-xxl` (ADR
            # 0122 decision 1) — its provenance is compared by the umT5 check below (the row of
            # `wan.sources.UMT5_SOURCES`).
            continue
        # Compare the provenance the container states with the upstream pin (`wan.sources` is the
        # source of truth) — a check closed over the model-name table alone lets through a container
        # built from another revision and placed at the series path. The DiT and VAE containers are
        # built from the generation's checkpoint, so they are checked against the pin of the plan's
        # model name (a field separate from the text assets' pin {@link WanGeneration.text_model} —
        # ADR 0121 decision 9).
        assert_upstream_provenance(container, license=upstream.license, revision=upstream.revision)
    assert_vae_chunk_pair(
        placements[WAN_VAE_FIRST_ROLE], placements[WAN_VAE_NEXT_ROLE], generation.vae_writer
    )
    # The int8 decoder pair (ADR 0133 decision 7) is a second pair written by its own run of the
    # same writer, so it passes the same gate on its own.
    if WAN_VAE_FIRST_I8_ROLE in placements:
        assert_vae_chunk_pair(
            placements[WAN_VAE_FIRST_I8_ROLE],
            placements[WAN_VAE_NEXT_I8_ROLE],
            generation.vae_writer,
        )
    # The pairs at another tile (ADR 0134 decision 5) pass the same pair gate per storage label,
    # and every container of them borrows all its weights from the same-label default-tile one.
    for tile in generation.vae_tiles[1:]:
        first_roles, next_roles = WAN_VAE_SHARED_LABEL_ROLES[tile]
        for label in generation.vae_labels:
            assert_vae_chunk_pair(
                placements[first_roles[label]],
                placements[next_roles[label]],
                f"{generation.vae_writer} --tile {tile} --shared",
            )
    for role in generation.vae_decoder_roles:
        lender = WAN_VAE_LENDER_ROLES.get(role)
        if lender is not None:
            assert_shared_decoder(placements[role], placements[lender])
    # The umT5 container passes the same checks as the umT5 distribution (provenance, binding table,
    # input / output contract) (the reference target is byte-identical to this container —
    # `karume.dist.external_refs` checks it). The model name is the one on the umT5 repository's
    # side.
    encoder_width = assert_umt5_encoder(placements[WAN_TEXT_ENCODER_ROLE], UMT5_DEFAULT_MODEL)
    transformers = [placements[role] for role in generation.transformer_roles]
    for transformer in transformers:
        assert_rope_base(transformer, generation.dit_writer)
        context = dit_context(transformer)
        # The embeddings asset, the tokenizer and the text_encoder are single quant-independent
        # entries, so they must fit the context input of every preset's DiT. The provenance of the 2
        # assets is checked against the text assets' pin (even in Wan2.2 they are assets made from
        # the umT5 of the Wan2.1 checkpoint — ADR 0121 decision 9).
        assert_text_embeds(placements[WAN_TEXT_EMBEDS_ROLE], generation.text_model, context)
        assert_umt5_tokenizer(placements[WAN_TOKENIZER_ROLE], generation.text_model, context[0])
        if encoder_width != context[1]:
            raise DistError(
                f"{placements[WAN_TEXT_ENCODER_ROLE]}: the umT5 output width {encoder_width}"
                f" differs from {transformer}'s context input width {context[1]} — the text"
                " stage's output cannot be passed to the DiT"
            )
    # A generation with a single transformer container (Wan2.2) passes with a one-element set.
    assert_shared_rope_base(transformers)
    weights = generation.weights
    return ModelPlan(
        name=model,
        pipeline=generation.pipeline,
        artifacts={
            role: Artifact(WAN_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=weights,
        assets=WAN_ASSETS,
        quants=complete_quant_weights(weights, generation.quants),
        default_quant=generation.default_quant,
        pipeline_config=generation.pipeline_config,
    )


def wan_dist_plan(series_dir: Path, model: str, generation: WanGeneration = WAN21) -> ModelPlan:
    """Build the plan for one model of the generation from the parent of `--series` (the CLI's
    dispatch target)."""
    return wan_plan(wan_sources(series_dir, generation), model, generation)


#: The text encoder every Wan model borrows from the umT5 repository instead of storing it (ADR
#: 0119 addendum "design of step 10d" / ADR 0121 decision 9 — the container is byte-identical to
#: `karume-umt5-xxl`'s, and {@link wan_plan} passes it through the same gate as that repository).
WAN_TEXT_ENCODER_REFERENCE = CrossRepoReference(
    repo_name=UMT5_REPO_NAME, model=UMT5_DEFAULT_MODEL, roles=(WAN_TEXT_ENCODER_ROLE,)
)


def wan_references(_model: str) -> tuple[CrossRepoReference, ...]:
    """`Pipeline.references` of both generations: the text encoder lives in the umT5 repository."""
    return (WAN_TEXT_ENCODER_REFERENCE,)


#: The original text of the upstream license (Apache 2.0). The redistribution terms of §4 apply per
#: distribution repository, and both the Wan2.1 and the Wan2.2 distribution repositories carry the
#: same text, so it is read once here whatever the number of assemblies or generations (ADR 0092
#: decision 7).
_APACHE_LICENSE = apache_license_2_0()

#: The `--pipeline wan` entry (the driver merges it into the core's PIPELINES).
PIPELINE = Pipeline(
    default_model=DEFAULT_MODEL,
    repo_name=wan_repo_name,
    plan=wan_dist_plan,
    # Attribution (upstream repository, license) is determined uniquely by the model name
    # (`wan.sources.SOURCES`), so it is not an axis to choose. The abbreviation table is a fact not
    # in the manifest, so it is passed from here (the same form as anima).
    card_profiles={
        "wan": partial(
            render_wan_model_card, abbreviations=WAN_QUANT_ABBREVIATIONS, card=WAN21.card
        )
    },
    root_files={
        "LICENSE.md": _APACHE_LICENSE,
        "NOTICE.md": WAN21.notice,
    },
    references=wan_references,
)

#: The `--pipeline wan-ti2v` entry (Wan2.2 TI2V 5B → `karume-wan2.2`). The same form as
#: {@link PIPELINE}, bundling the generation table {@link WAN22}. The upstream license is the same
#: Apache 2.0, and the NOTICE is 2.2's modifications.
TI2V_PIPELINE = Pipeline(
    default_model=WAN22.default_model,
    repo_name=partial(wan_repo_name, generation=WAN22),
    plan=partial(wan_dist_plan, generation=WAN22),
    card_profiles={
        "wan-ti2v": partial(
            render_wan_model_card, abbreviations=WAN_QUANT_ABBREVIATIONS, card=WAN22.card
        )
    },
    root_files={
        "LICENSE.md": _APACHE_LICENSE,
        "NOTICE.md": WAN22.notice,
    },
    references=wan_references,
)
