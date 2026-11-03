"""Where each Gemma 4 model's upstream checkpoint is read from, and the pins of the ones read from
the local Hugging Face cache.

E2B / E4B are read from `inputs/gemma4/<checkpoint>/` (placed by hand — assets-layout; the
revision used is recorded in `THIRD_PARTY_NOTICES.md`): one directory holds the weights, the
config, the tokenizer, the chat template and the generation defaults. 12B is read from the hub
cache at pinned commits ({@link HUB_PINS}), from **two** repositories:

- **the weights and their config** — `google/gemma-4-12B-it-qat-q4_0-unquantized`, Google's
  quantization-aware-trained checkpoint of 12B-it, whose linears already lie on a 4-bit grid
  (`gemma4.export_unified` stores that grid as is). Its `config.json` is the one that describes
  these weights.
- **the tokenizer, the chat template and the generation defaults** — `google/gemma-4-12B-it`.
  The QAT repository's `tokenizer_config.json`, `generation_config.json` and
  `chat_template.jinja` are byte-identical to 12B-it's, but its `tokenizer.json` is E2B's
  (SHA-256 `cc8d3a0c…`, not 12B-it's `9788bdb6…`; 239 merges differ) — a packaging slip of the
  QAT repository: the model is a QAT fine-tune of 12B-it, whose tokenizer is `9788bdb6…`.

E2B / E4B also have a **q4_0 QAT line** ({@link QAT_PINS} — `gemma4.export_product_qat`): the
weights and their config are Google's quantization-aware-trained checkpoints
`google/gemma-4-E{2,4}B-it-qat-q4_0-unquantized`, the tokenizer, the chat template and the
generation defaults the regular `google/gemma-4-E{2,4}B-it` at the revisions the hand-placed
checkpoints of `inputs/gemma4/` were fetched at. The hand-placed regular line is unaffected:
{@link checkpoint_dir} / {@link tokenizer_dir} keep reading `inputs/gemma4/`.

The q4_0 QAT MTP drafters ({@link QAT_DRAFTER_PINS} — `gemma4.export_drafter_qat`) are pinned for
E2B, E4B and 12B; each pairs with the QAT target {@link QAT_TARGET_PINS} names (12B's is its one
line, {@link HUB_PINS}).

Both repositories are still being updated, so following `main` would export whatever the branches
point at on the day of the run. This module never downloads; `hf download <repo> --revision <pin>`
fetches a snapshot.

A pin also carries the SHA-256 of the files the provenance record fingerprints from that source
(`gemma4.provenance.FINGERPRINT_FILES` is the union of the two pins' files).
{@link assert_model_pins} checks both sources cheaply before a long run (the cache names an LFS
blob by its SHA-256, so only the small files are hashed), and the export checks the record's full
digests against them once more (`gemma4.provenance.assert_pinned_fingerprint`).

MUST: torch-free — the distribution recipe reads it.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from _shared.paths import INPUTS_ROOT, REPO_ROOT
from _shared.upstream import hub_snapshot_dir

from .card import GEMMA4_UPSTREAM, GEMMA4_WEIGHTS_UPSTREAM

#: The family's directory of hand-placed upstream checkpoints (`inputs/gemma4/<checkpoint>/`).
INPUTS_DIRNAME = "gemma4"

#: The name the hub cache gives an LFS blob (its SHA-256).
_LFS_BLOB = re.compile(r"[0-9a-f]{64}")


class SourcePinError(ValueError):
    """The checkpoint on disk is not the pinned one."""


@dataclass(frozen=True)
class HubPin:
    """One repository's commit in the hub cache."""

    #: The HF repository id.
    repo: str
    #: The HF commit SHA (40 lower-case hex digits) — never a branch or a tag.
    revision: str
    #: File → SHA-256 at {@link revision}, for the files the provenance record fingerprints from
    #: this source (for an LFS file, the oid the HF API reports and the cache names the blob by).
    sha256: Mapping[str, str]

    @property
    def label(self) -> str:
        """`<repo>@<revision>` — how a git-tracked record names this source."""
        return f"{self.repo}@{self.revision}"


@dataclass(frozen=True)
class ModelPins:
    """The two pinned sources of one model read from the hub cache (module docstring)."""

    #: The weights and the config that describes them (`model.safetensors`, `config.json`).
    weights: HubPin
    #: The tokenizer (`tokenizer.json`); the chat template, `tokenizer_config.json` and
    #: `generation_config.json` are read from the same snapshot.
    tokenizer: HubPin

    def by_role(self) -> dict[str, HubPin]:
        """Role → pin, in the order the record lists them."""
        return {"weights": self.weights, "tokenizer": self.tokenizer}

    @property
    def label(self) -> str:
        """Both sources as `<repo>@<revision> (<role>)`, joined by ` + ` — how a git-tracked record
        names a pinned checkpoint."""
        return " + ".join(f"{pin.label} ({role})" for role, pin in self.by_role().items())


#: Model → its pinned sources. 12B, both fetched on 2026-11-02 (Apache-2.0 per each snapshot's
#: README front matter; neither is gated):
#:
#: - weights: `main` of `google/gemma-4-12B-it-qat-q4_0-unquantized` — the LFS blob name of
#:   `model.safetensors` (23,919,549,408 B) and the SHA-256 of `config.json`'s bytes. Against
#:   12B-it's `config.json` its text config differs only in an explicit `dtype` (`bfloat16`) and
#:   `vocab_size_per_layer_input` (0 instead of 262144 — unread: the class has no per-layer
#:   embeddings); the audio and vision sub-configs drop a few fields.
#: - tokenizer: `main` of `google/gemma-4-12B-it` (the commit 12B was first pinned at) — the LFS
#:   blob name of `tokenizer.json` (module docstring: the QAT repository's is E2B's).
#:
#: Moving a pin needs the licence review of `THIRD_PARTY_NOTICES.md` again.
_TWELVE_B_WEIGHTS = HubPin(
    repo=GEMMA4_WEIGHTS_UPSTREAM["12b"],
    revision="b6ed86275a6a5735884e208bfed95b445a684ca2",
    sha256={
        "model.safetensors": "26f2cee4292298a3f9f92209643c37c80e34e011381e22434088870d9439a0a0",
        "config.json": "a323d02f68420f6fa3a3548130a0d36356075a4047a622e57148558f8eee7077",
    },
)
_TWELVE_B_TOKENIZER = HubPin(
    repo=GEMMA4_UPSTREAM["12b"],
    revision="43c9d014b231474fa7212dbb48a762566c2b255d",
    sha256={"tokenizer.json": "9788bdb6a021b3170357cd3587b5d622b1177561ad3c5bf43d73728fd71aeacb"},
)
HUB_PINS: Mapping[str, ModelPins] = {
    "12b": ModelPins(weights=_TWELVE_B_WEIGHTS, tokenizer=_TWELVE_B_TOKENIZER),
}

#: Model → Google's q4_0 quantization-aware-trained checkpoint of its `-it` model (README front
#: matter at the pinned revisions: `license: apache-2.0`, `base_model` the regular `-it` model).
#: Same architecture and config as the regular checkpoint; every 2-D text weight lies on a q4_0
#: grid (`gemma4.export_product_qat`). Unlike the regular checkpoints, the KV-shared layers carry
#: no k / v residues.
QAT_WEIGHTS_UPSTREAM: Mapping[str, str] = {
    "e2b": "google/gemma-4-E2B-it-qat-q4_0-unquantized",
    "e4b": "google/gemma-4-E4B-it-qat-q4_0-unquantized",
}

#: Model → the pinned sources of its q4_0 QAT line, all four in the hub cache on 2026-11-03:
#:
#: - weights: `main` of {@link QAT_WEIGHTS_UPSTREAM} — the LFS blob name of `model.safetensors`
#:   (E2B 10,208,852,878 B / E4B 15,882,477,468 B) and the SHA-256 of `config.json`'s bytes.
#:   Against the regular `-it` config the text config differs in no field the export reads (the
#:   `transformers_version` and two null MoE fields).
#: - tokenizer: the regular `-it` repository at the commit the hand-placed checkpoint was fetched
#:   at (the `.metadata` record of `inputs/gemma4/gemma-4-E{2,4}B-it/`) — the LFS blob name of
#:   `tokenizer.json`. The QAT repositories ship the same `tokenizer.json`, but their
#:   `tokenizer_config.json` and `generation_config.json` are other bytes; reading the tokenizer
#:   files from the regular repository keeps the line's tokenizer the regular product's.
#:
#: Moving a pin needs the licence review of `THIRD_PARTY_NOTICES.md` again.
QAT_PINS: Mapping[str, ModelPins] = {
    "e2b": ModelPins(
        weights=HubPin(
            repo=QAT_WEIGHTS_UPSTREAM["e2b"],
            revision="6befbaca7398925921802abd1f277b495b78b738",
            sha256={
                "model.safetensors": (
                    "33fe0cece08fb527ffefbd1a3a9ce73bd71073727993a283506293e5c6bf0137"
                ),
                "config.json": "bbeff1e2fd3fe282536e7ace02309d43e0dbd9b6ac4b6a149b97e3ab6942a878",
            },
        ),
        tokenizer=HubPin(
            repo=GEMMA4_UPSTREAM["e2b"],
            revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            sha256={
                "tokenizer.json": (
                    "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
                )
            },
        ),
    ),
    "e4b": ModelPins(
        weights=HubPin(
            repo=QAT_WEIGHTS_UPSTREAM["e4b"],
            revision="476025a01dbf99361c062bbeca3d6a76bb4c4566",
            sha256={
                "model.safetensors": (
                    "ad8ef515194b15ab13b6f98d54d37652b691c9a2c787cdf7ed5f951f5ed2c7fa"
                ),
                "config.json": "f2db6d6e24cb4b695308897d82c8424fae946e95272a27de68b6e908c77b0f95",
            },
        ),
        tokenizer=HubPin(
            repo=GEMMA4_UPSTREAM["e4b"],
            revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
            sha256={
                "tokenizer.json": (
                    "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
                )
            },
        ),
    ),
}


#: Model → Google's q4_0 quantization-aware-trained MTP drafter (the "assistant") of its QAT
#: target ({@link QAT_WEIGHTS_UPSTREAM}). Google's model card: with a QAT target, the assistant
#: "must also be a QAT checkpoint with the same precision" — the regular E2B assistant
#: (`gemma4.export_drafter`) does not pair with the QAT line.
QAT_DRAFTER_UPSTREAM: Mapping[str, str] = {
    "e2b": "google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant",
    "e4b": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
    "12b": "google/gemma-4-12B-it-qat-q4_0-unquantized-assistant",
}

#: Model → the pinned snapshot of its q4_0 QAT drafter (`gemma4.export_drafter_qat`), all three in
#: the hub cache on 2026-11-03 (README front matter: `license: apache-2.0`, `base_model` the regular
#: `google/gemma-4-<model>-it-assistant`): the LFS blob name of `model.safetensors` (E2B
#: 157,565,344 B / E4B 159,138,208 B / 12B 845,719,296 B) and the SHA-256 of `config.json`'s bytes.
#: E2B's config is the regular E2B assistant's but for an explicit `eos_token_id` (`[1, 106]`);
#: E4B's differs from E2B's in `backbone_hidden_size` (2560) and the drafter's
#: `num_key_value_heads` (2) — the E4B target's width and KV heads. 12B's is another class,
#: `gemma4_unified_assistant` (`Gemma4UnifiedAssistantForCausalLM` — upstream's code is the E2B /
#: E4B class's, renamed), at 12B's widths: backbone 3840, hidden 1024, 16 query heads, 8 KV heads on
#: the sliding layers and 1 on the full one (`attention_k_eq_v`, `num_global_key_value_heads`),
#: window 1024, and `use_ordered_embeddings` already false (no centroid tensor in the checkpoint).
#: The files the provenance record fingerprints are these two
#: (`gemma4.export_drafter_qat.DRAFTER_FILES`).
#:
#: Moving a pin needs the licence review of `THIRD_PARTY_NOTICES.md` again.
QAT_DRAFTER_PINS: Mapping[str, HubPin] = {
    "e2b": HubPin(
        repo=QAT_DRAFTER_UPSTREAM["e2b"],
        revision="ebc7e1a211354561464cb82ed6d886792138dcb6",
        sha256={
            "model.safetensors": (
                "28b11aa1fef73e655107984e0024ed1b149df4b8b36dcb95f27cca603eabc960"
            ),
            "config.json": "5d01e9f3f8e969aa8147201a26e849c05446c7c746fa918101ed0622b201db15",
        },
    ),
    "e4b": HubPin(
        repo=QAT_DRAFTER_UPSTREAM["e4b"],
        revision="27f8d204f09f2be353d6ff0bf0d012792b13c79f",
        sha256={
            "model.safetensors": (
                "9d0e2053067590cae9a8f4fcc57eefbe20dff90599360458ebd451e5cb5c947d"
            ),
            "config.json": "190a55c13f78f2634134a2a9dfc75a58489bc14513d767a16a20272d7fadbbae",
        },
    ),
    "12b": HubPin(
        repo=QAT_DRAFTER_UPSTREAM["12b"],
        revision="18934064dd4c5c6cc3621f6381e7d377fc8cb7bd",
        sha256={
            "model.safetensors": (
                "67f1420cf24aa5065089aaed175223f7c245ccfda16111b6c56765afd7280db6"
            ),
            "config.json": "7638c1d42f9fa73fe44b1a10604766b928b8985263f15775cd1a286a5a12799c",
        },
    ),
}

#: Model → the pinned sources of the q4_0 QAT target its QAT drafter pairs with: E2B / E4B's QAT
#: line ({@link QAT_PINS}); 12B's one line ({@link HUB_PINS}), whose weights are already the QAT
#: checkpoint. Google's model card: with a QAT target, the assistant must be a QAT checkpoint of the
#: same precision — so every QAT drafter has an entry here, and nothing else does.
QAT_TARGET_PINS: Mapping[str, ModelPins] = {**QAT_PINS, "12b": HUB_PINS["12b"]}


def _hand_placed_dir(model: str) -> Path:
    """`inputs/gemma4/<checkpoint>/` (the checkpoint name is the tail of the upstream repository —
    `gemma4.card.GEMMA4_UPSTREAM`); an unknown model fails loudly."""
    repo = GEMMA4_UPSTREAM.get(model)
    if repo is None:
        raise ValueError(
            f"gemma4 has no model '{model}' (known: {' / '.join(sorted(GEMMA4_UPSTREAM))})"
        )
    return INPUTS_ROOT / INPUTS_DIRNAME / repo.split("/", 1)[1]


def checkpoint_dir(model: str) -> Path:
    """The directory model `model`'s weights and their `config.json` are read from (a path only —
    the readers fail loudly on a missing file).

    Pinned models ({@link HUB_PINS}) resolve to the hub cache's snapshot of the weights pin; the
    others to `inputs/gemma4/<checkpoint>/`.
    """
    hand_placed = _hand_placed_dir(model)
    pins = HUB_PINS.get(model)
    if pins is None:
        return hand_placed
    return hub_snapshot_dir(pins.weights.repo, pins.weights.revision)


def tokenizer_dir(model: str) -> Path:
    """The directory model `model`'s `tokenizer.json`, chat template and generation defaults are
    read from — {@link checkpoint_dir} for a hand-placed checkpoint, the snapshot of the tokenizer
    pin for a pinned model."""
    hand_placed = _hand_placed_dir(model)
    pins = HUB_PINS.get(model)
    if pins is None:
        return hand_placed
    return hub_snapshot_dir(pins.tokenizer.repo, pins.tokenizer.revision)


def checkpoint_label(model: str) -> str:
    """How a git-tracked record names model `model`'s checkpoint: the repository-relative path of
    a hand-placed one; for a pinned one, both sources as `<repo>@<revision>` with their roles (an
    absolute cache path names a machine, not a source)."""
    pins = HUB_PINS.get(model)
    if pins is None:
        return str(checkpoint_dir(model).relative_to(REPO_ROOT))
    return pins.label


def tokenizer_label(model: str) -> str:
    """How a git-tracked record names the source of model `model`'s tokenizer files: as
    {@link checkpoint_label} for a hand-placed checkpoint, the tokenizer pin's `<repo>@<revision>`
    for a pinned one."""
    pins = HUB_PINS.get(model)
    if pins is None:
        return checkpoint_label(model)
    return pins.tokenizer.label


def qat_pins(model: str) -> ModelPins:
    """The pinned sources of model `model`'s q4_0 QAT line ({@link QAT_PINS}); a model without one
    fails loudly."""
    pins = QAT_PINS.get(model)
    if pins is None:
        raise ValueError(
            f"gemma4 has no q4_0 QAT line for model '{model}'"
            f" (known: {' / '.join(sorted(QAT_PINS))})"
        )
    return pins


def qat_checkpoint_dir(model: str) -> Path:
    """The hub cache's snapshot of model `model`'s QAT weights and their `config.json` (a path
    only, like {@link checkpoint_dir})."""
    pin = qat_pins(model).weights
    return hub_snapshot_dir(pin.repo, pin.revision)


def qat_tokenizer_dir(model: str) -> Path:
    """The hub cache's snapshot model `model`'s QAT line reads its tokenizer, chat template and
    generation defaults from (the regular `-it` repository — {@link QAT_PINS})."""
    pin = qat_pins(model).tokenizer
    return hub_snapshot_dir(pin.repo, pin.revision)


def qat_drafter_pin(model: str) -> HubPin:
    """The pinned snapshot of model `model`'s q4_0 QAT drafter ({@link QAT_DRAFTER_PINS}); a model
    without one fails loudly."""
    pin = QAT_DRAFTER_PINS.get(model)
    if pin is None:
        raise ValueError(
            f"gemma4 has no q4_0 QAT drafter for model '{model}'"
            f" (known: {' / '.join(sorted(QAT_DRAFTER_PINS))})"
        )
    return pin


def qat_drafter_dir(model: str) -> Path:
    """The hub cache's snapshot of model `model`'s QAT drafter (a path only, like
    {@link checkpoint_dir})."""
    pin = qat_drafter_pin(model)
    return hub_snapshot_dir(pin.repo, pin.revision)


def qat_target_pins(model: str) -> ModelPins:
    """The pinned sources of the QAT target model `model`'s QAT drafter pairs with
    ({@link QAT_TARGET_PINS}); a model without one fails loudly."""
    pins = QAT_TARGET_PINS.get(model)
    if pins is None:
        raise ValueError(
            f"gemma4 has no q4_0 QAT target for model '{model}'"
            f" (known: {' / '.join(sorted(QAT_TARGET_PINS))})"
        )
    return pins


def qat_target_checkpoint_dir(model: str) -> Path:
    """The hub cache's snapshot of the QAT target's weights and their `config.json` (a path only,
    like {@link checkpoint_dir}): {@link qat_checkpoint_dir} for E2B / E4B, 12B's weights pin."""
    pin = qat_target_pins(model).weights
    return hub_snapshot_dir(pin.repo, pin.revision)


def qat_target_tokenizer_dir(model: str) -> Path:
    """The hub cache's snapshot the QAT target reads its tokenizer, chat template and generation
    defaults from: {@link qat_tokenizer_dir} for E2B / E4B, 12B's tokenizer pin."""
    pin = qat_target_pins(model).tokenizer
    return hub_snapshot_dir(pin.repo, pin.revision)


def _digest(path: Path) -> str:
    """The SHA-256 of `path` — the blob's name when the cache links it to an LFS blob, else the
    bytes (only small files reach the hashing branch in a hub snapshot)."""
    if path.is_symlink() and _LFS_BLOB.fullmatch(path.resolve().name):
        return path.resolve().name
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assert_snapshot_pin(directory: Path, pin: HubPin) -> None:
    """Every file `pin` names is in `directory` with the pinned SHA-256 (cheap in a hub snapshot).

    MUST: run before anything long reads the checkpoint — a partial download or another revision
    placed at the same path otherwise surfaces hours later, or not at all.
    """
    for name, expected in sorted(pin.sha256.items()):
        path = directory / name
        if not path.is_file():
            raise SourcePinError(
                f"{path} is missing — fetch the pinned snapshot ({pin.label}) first"
            )
        actual = _digest(path)
        if actual != expected:
            raise SourcePinError(f"{path}: SHA-256 {actual}, pin {expected} ({pin.label})")


def assert_model_pins(weights_dir: Path, tokenizer_dir: Path, pins: ModelPins) -> None:
    """Both sources of a pinned model are the pinned snapshots ({@link assert_snapshot_pin} on
    each — the same MUST: before anything long reads either)."""
    assert_snapshot_pin(weights_dir, pins.weights)
    assert_snapshot_pin(tokenizer_dir, pins.tokenizer)
