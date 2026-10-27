"""The upstream of MOSS-Image (HF repository and revision pin) and fetching its parts (ADR NNNN
decision 1).

{@link SOURCE} is the source of truth for where the checkpoint comes from. The revision is the
commit SHA (40 hex digits) resolved through the HF API — the rule of ADR 0092 decision 3 applied to
the upstream side, as for Wan (ADR 0121 decision 1). Every script reads the upstream through this
module: a script that spells `from_pretrained(repo)` itself reads whatever the default branch points
at on the day it runs.

The upstream pipeline file `pipeline_moss_image.py` is `trust_remote_code` code (it defines the DiT,
the scheduler and the pipeline). Besides the revision, its SHA-256 is pinned
({@link PIPELINE_CODE_SHA256}) and checked on every copy the repository carries before the file is
imported ({@link load_pipeline_module}); diffusers' own remote-code loader is never used. The LFS
SHA-256 of every DiT shard and of the bundled VAE file is pinned too ({@link WEIGHT_SHA256}) and
checked after `--fetch` and in the check mode (hashing 10.4 GB takes seconds; the reference scripts
do not repeat it on every run — they rely on the element counts and the check mode).

What is fetched (`--fetch`): the DiT (`transformer`, bf16, 10.16 GB), the VAE (`vae`, f32, 0.25 GB),
the scheduler config and the root files (`README.md` with the licence front matter, `LICENSE`,
`model_index.json`, `pipeline_moss_image.py`). The Qwen3.5 text encoder (`text_encoder`, bf16,
16.8 GB) and its tokenizer are fetched separately (`--fetch --text`): only `moss.text_embeds` reads
them, in its own process, so the text encoder and the f32 DiT (20.3 GB) never share the host RAM.

The VAE is the FLUX.2 small decoder (HF README). {@link VAE_ORIGINS} pins the two FLUX.2
repositories its tensors come from; `--vae-origin` fetches their weights files and compares them
tensor by tensor with the bundled one (THIRD_PARTY_NOTICES.md records the result).

The upstream user guide (`USERGUIDE.md` of the GitHub repository, {@link USER_GUIDE}) is the source
of two fixed prompts; `--guide` fetches it at the pinned commit into `outputs/misc/` and checks its
SHA-256, so `moss/tests/test_prompts.py` can check the quotes against it.

    uv run --group moss --inexact python -m moss.sources --fetch         # DiT, VAE, scheduler
    uv run --group moss --inexact python -m moss.sources --fetch --text  # text encoder, tokenizer
    uv run --group moss --inexact python -m moss.sources                 # check what is cached
    uv run --group moss --inexact python -m moss.sources --vae-origin    # the FLUX.2 origins
    uv run --group moss --inexact python -m moss.sources --fetch --guide # USERGUIDE.md (GitHub)

MUST: `huggingface_hub` is imported inside the functions (the `moss` group is not in the default
sync — `tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from _shared.paths import MISC_ROOT

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class UpstreamSource:
    """One upstream HF repository (repo + pinned revision + licence identifier)."""

    repo: str
    #: The HF commit SHA (40 lower-case hex digits). Never a branch name or a tag.
    revision: str
    #: `license:` of the upstream `README.md` front matter (checked through the HF API on
    #: 2026-10-27: `cardData.license`).
    license: str


@dataclass(frozen=True)
class Component:
    """One part (`subfolder`) of the upstream repository."""

    subfolder: str
    #: Fetched by the text-embedding process only (`--fetch --text`), not with the DiT and VAE.
    text: bool
    #: Why it is fetched (for the readers of the scripts and the reports).
    why: str


#: The upstream checkpoint. `main` of `OpenMOSS-Team/MOSS-Image-SFT` on 2026-10-27 (last modified
#: 2026-10-21); not gated.
SOURCE = UpstreamSource(
    repo="OpenMOSS-Team/MOSS-Image-SFT",
    revision="f972bd4b4b9e3852133cc5da954d87bc32cd8fb9",
    license="apache-2.0",
)

#: The parts of the upstream repository and who fetches them.
COMPONENTS: tuple[Component, ...] = (
    Component("transformer", text=False, why="the DiT — bf16 safetensors in 3 shards, 10.16 GB"),
    Component("vae", text=False, why="the FLUX.2 VAE with the small decoder — f32, 0.25 GB"),
    Component("scheduler", text=False, why="MOSSImageScheduler config (shift 5, training_shift 1)"),
    Component(
        "text_encoder",
        text=True,
        why="Qwen3.5 (Qwen3_5Model) — bf16, 16.8 GB; read only by moss.text_embeds",
    ),
    Component("tokenizer", text=True, why="Qwen2Tokenizer and the chat template"),
)

#: Files fetched from the repository root (licence front matter and text, the component index, the
#: pipeline code).
ROOT_FILES: tuple[str, ...] = ("README.md", "LICENSE", "model_index.json", "pipeline_moss_image.py")

#: The pipeline code that defines `MOSSImageTransformer`, `MOSSImageScheduler` and
#: `MOSSImagePipeline` (imported from the repository root).
PIPELINE_CODE = "pipeline_moss_image.py"

#: The copies of the pipeline code in the repository (diffusers' custom-component discovery looks
#: for one in each component folder that names `pipeline_moss_image`). All three are the same git
#: blob (`60ceb4b6…`) at the pinned revision; every one is checked so a revision where they diverge
#: fails instead of loading one of them silently.
PIPELINE_CODE_COPIES: tuple[str, ...] = (
    PIPELINE_CODE,
    f"scheduler/{PIPELINE_CODE}",
    f"transformer/{PIPELINE_CODE}",
)

#: SHA-256 of `pipeline_moss_image.py` at {@link SOURCE} (25,060 bytes; computed 2026-10-27 on the
#: fetched file and on an earlier download of the same revision).
PIPELINE_CODE_SHA256 = "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31"

#: The LFS SHA-256 of the weights files of {@link SOURCE} (HF API `model_info(files_metadata=True)`
#: `siblings[].lfs.sha256`, 2026-10-27): the three DiT shards and the bundled VAE.
WEIGHT_SHA256: dict[str, str] = {
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": (
        "e40d3444940a0b0fd7cd45ba7959a5bd3d5efd7ddf87aae5ce4d87ce8aa03d01"
    ),
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors": (
        "d106da790ec2924ffd8af04b4ffa8f8de281a974b58e55649c3ec596dc8ef033"
    ),
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors": (
        "11f637ab17c7093344a6110a1e87ffda8e56558dc55183d8e790222cfcdcf52d"
    ),
    "vae/diffusion_pytorch_model.safetensors": (
        "c3bdf94dc67ec69b30ea9fe1fa8340085679cae64d019f7c7e4a9921f1a7f17a"
    ),
}

#: The module name the pipeline code is imported under (one instance per process, so the classes
#: the scripts compare are the same objects).
PIPELINE_MODULE_NAME = "moss_upstream_pipeline_moss_image"

#: The number of elements a fetched part must hold (sum over the safetensors headers, 2026-10-27).
#: transformer: 5,082,308,608 bf16 parameters + `rope.bands` (16 f32 elements — a derived constant
#: that `_RoPE` recomputes on load). vae: the encoder, the small decoder, both 1×1 convs and the
#: BatchNorm statistics (`bn.num_batches_tracked` is the one i64 element).
EXPECTED_ELEMENTS: dict[str, int] = {
    "transformer": 5_082_308_624,
    "vae": 62_373_348,
    "text_encoder": 8_392_695_024,
}


@dataclass(frozen=True)
class VaeOrigin:
    """One upstream file that the bundled VAE's tensors under `prefixes` are copies of."""

    source: UpstreamSource
    #: The weights file in the origin repository and its LFS SHA-256 (HF API, 2026-10-27).
    file: str
    sha256: str
    #: The tensor-name prefixes this origin accounts for (every bundled tensor has exactly one).
    prefixes: tuple[str, ...]


#: Where the bundled VAE comes from (HF README of MOSS-Image-SFT: "The bundled VAE uses the FLUX.2
#: small decoder"; the pipeline docstring loads the VAE from `FLUX.2-klein-4B`). Measured on
#: 2026-10-27 with `--vae-origin`: the decoder and `post_quant_conv` are the f32 tensors of the
#: small-decoder repository byte for byte; the encoder, `quant_conv` and the BatchNorm statistics
#: (`bn.*`, which the decode path uses) are the bf16 tensors of the FLUX.2 [klein] 4B VAE cast to
#: f32 (exact). The small-decoder file's own encoder and BatchNorm statistics are different values.
#: Both repositories are Apache-2.0 (`cardData.license`; not gated) on the pinned revisions.
VAE_ORIGINS: tuple[VaeOrigin, ...] = (
    VaeOrigin(
        source=UpstreamSource(
            repo="black-forest-labs/FLUX.2-small-decoder",
            revision="a3efc24f613ef42d9428af62fdbd6f5fd8856c4a",
            license="apache-2.0",
        ),
        file="diffusion_pytorch_model.safetensors",
        sha256="d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b",
        prefixes=("decoder.", "post_quant_conv."),
    ),
    VaeOrigin(
        source=UpstreamSource(
            repo="black-forest-labs/FLUX.2-klein-4B",
            revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
            license="apache-2.0",
        ),
        file="vae/diffusion_pytorch_model.safetensors",
        sha256="ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04",
        prefixes=("encoder.", "quant_conv.", "bn."),
    ),
)

#: The VAE weights file of {@link SOURCE}.
VAE_FILE = "vae/diffusion_pytorch_model.safetensors"


@dataclass(frozen=True)
class GitHubFile:
    """One file of a GitHub repository at a pinned commit, with the SHA-256 of its bytes."""

    repo: str
    #: The commit SHA (40 lower-case hex digits). Never a branch name or a tag.
    commit: str
    path: str
    sha256: str

    @property
    def url(self) -> str:
        """The commit-pinned page (what a reader opens)."""
        return f"https://github.com/{self.repo}/blob/{self.commit}/{self.path}"

    @property
    def raw_url(self) -> str:
        """The commit-pinned bytes (what `--guide` fetches)."""
        return f"https://raw.githubusercontent.com/{self.repo}/{self.commit}/{self.path}"


#: The upstream user guide (`main` of OpenMOSS/MOSS-Image on 2026-10-27; 9,000 bytes), the source
#: of the two "recommended example prompts" in `moss.prompts`.
USER_GUIDE = GitHubFile(
    repo="OpenMOSS/MOSS-Image",
    commit="a211e398a33cbf4dd7a1cc52aba0dfdbac3b0281",
    path="USERGUIDE.md",
    sha256="46dc6966b3d3e3bd0eca36f45c9428178f87f2f5c6cfaa61440f84d468d3e2a2",
)

#: Where `--guide` keeps the user guide (a frozen copy of an upstream file — `outputs/misc/`,
#: docs/assets-layout.md; deleting it means fetching it again).
USER_GUIDE_PATH = MISC_ROOT / "moss-image-upstream" / USER_GUIDE.commit / USER_GUIDE.path

#: Bytes per element of the safetensors dtypes (header consistency check).
_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1}


class MossSourceError(RuntimeError):
    """The upstream checkpoint is not in the pinned form on this machine."""


def allow_patterns(*, text: bool) -> list[str]:
    """What `snapshot_download` fetches: the root files and the product parts, or the text parts."""
    parts = [f"{part.subfolder}/*" for part in COMPONENTS if part.text == text]
    return parts if text else [*ROOT_FILES, *parts]


def _snapshot(*, text: bool, fetch: bool) -> Path:
    """The pinned snapshot directory in the default HF cache (network only when `fetch`).

    Without `fetch` a missing snapshot fails loudly: a script that starts a 10–17 GB download on
    its own hides it inside a reference run.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot = Path(
            snapshot_download(
                SOURCE.repo,
                revision=SOURCE.revision,
                allow_patterns=allow_patterns(text=text),
                local_files_only=not fetch,
            )
        )
    except LocalEntryNotFoundError as error:
        flag = "--fetch --text" if text else "--fetch"
        raise MossSourceError(
            f"{SOURCE.repo}@{SOURCE.revision} is not in the HF cache — fetch it first with"
            f" `uv run --group moss --inexact python -m moss.sources {flag}`"
        ) from error
    missing = [p.subfolder for p in COMPONENTS if p.text == text]
    missing = [name for name in missing if not (snapshot / name).is_dir()]
    if not text:
        missing += [name for name in ROOT_FILES if not (snapshot / name).is_file()]
    if missing:
        raise MossSourceError(f"{snapshot} lacks {missing} — fetch again")
    return snapshot


def local_snapshot(*, fetch: bool = False) -> Path:
    """The snapshot with the DiT, the VAE, the scheduler config and the root files."""
    return _snapshot(text=False, fetch=fetch)


def text_snapshot(*, fetch: bool = False) -> Path:
    """The snapshot with the text encoder and the tokenizer (a separate call so the DiT/VAE fetch
    never pulls the 16.8 GB text encoder)."""
    return _snapshot(text=True, fetch=fetch)


def sha256_of(path: Path) -> str:
    """SHA-256 of a file's bytes (streamed)."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pipeline_code(snapshot: Path) -> bytes:
    """Check every copy of the pipeline code against {@link PIPELINE_CODE_SHA256} and return the
    bytes of the root copy.

    MUST: a mismatch fails loudly — the file is executed when imported.
    """
    for name in PIPELINE_CODE_COPIES:
        path = snapshot / name
        if not path.is_file():
            raise MossSourceError(f"{path} is missing — fetch again")
        actual = sha256_of(path)
        if actual != PIPELINE_CODE_SHA256:
            raise MossSourceError(
                f"{path}: sha256 {actual} is not the pinned {PIPELINE_CODE_SHA256} — the pin"
                " (moss.sources) and the cached revision disagree; refusing to import it"
            )
    return (snapshot / PIPELINE_CODE).read_bytes()


def load_pipeline_module(snapshot: Path) -> ModuleType:
    """Import the verified upstream pipeline code (one module instance per process).

    The bytes that were hashed are the bytes that are executed (compiled from memory, not re-read
    from the cache), so a cache entry swapped after the check is not run.
    """
    loaded = sys.modules.get(PIPELINE_MODULE_NAME)
    if loaded is not None:
        return loaded
    source = verify_pipeline_code(snapshot)
    path = snapshot / PIPELINE_CODE
    spec = importlib.util.spec_from_loader(PIPELINE_MODULE_NAME, loader=None, origin=str(path))
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[PIPELINE_MODULE_NAME] = module
    try:
        # The source is the hash-checked upstream file (see the docstring).
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        del sys.modules[PIPELINE_MODULE_NAME]
        raise
    return module


def read_safetensors_header(path: Path) -> dict[str, dict[str, object]]:
    """The tensor entries of a safetensors header (8-byte length + JSON; the body is not read)."""
    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(length))
    header.pop("__metadata__", None)
    return header


def count_elements(component_dir: Path) -> int:
    """The number of elements over every safetensors file of a part.

    An entry whose `data_offsets` width disagrees with `dtype × shape` fails loudly (a misread
    header must not produce a plausible count).
    """
    files = sorted(component_dir.glob("*.safetensors"))
    if not files:
        raise MossSourceError(f"{component_dir} has no safetensors")
    total = 0
    for file in files:
        for name, entry in read_safetensors_header(file).items():
            shape, dtype, offsets = entry["shape"], entry["dtype"], entry["data_offsets"]
            if not (
                isinstance(shape, list)
                and isinstance(dtype, str)
                and isinstance(offsets, list)
                and len(offsets) == 2
            ):
                raise MossSourceError(f"{file}: {name}: the header entry {entry} is malformed")
            if dtype not in _DTYPE_BYTES:
                raise MossSourceError(f"{file}: {name}: unknown dtype {dtype!r}")
            count = 1
            for size in shape:
                count *= int(size)
            if count * _DTYPE_BYTES[dtype] != int(offsets[1]) - int(offsets[0]):
                raise MossSourceError(f"{file}: {name}: {dtype} × {shape} disagrees with its width")
            total += count
    return total


def check_snapshot(snapshot: Path, parts: tuple[str, ...]) -> dict[str, int]:
    """Count the elements of `parts` and compare them with {@link EXPECTED_ELEMENTS}."""
    counts = {name: count_elements(snapshot / name) for name in parts}
    wrong = {n: (c, EXPECTED_ELEMENTS[n]) for n, c in counts.items() if c != EXPECTED_ELEMENTS[n]}
    if wrong:
        raise MossSourceError(f"element counts differ from the pin (found, expected): {wrong}")
    return counts


def verify_weights(snapshot: Path) -> dict[str, str]:
    """Check every file of {@link WEIGHT_SHA256} against its pinned LFS SHA-256 (fails loudly)."""
    for name, pinned in WEIGHT_SHA256.items():
        path = snapshot / name
        if not path.is_file():
            raise MossSourceError(f"{path} is missing — fetch again")
        actual = sha256_of(path)
        if actual != pinned:
            raise MossSourceError(f"{path}: sha256 {actual} is not the pinned {pinned}")
    return dict(WEIGHT_SHA256)


def fetch_user_guide(destination: Path = USER_GUIDE_PATH) -> Path:
    """Download {@link USER_GUIDE} at its commit, check its SHA-256, then keep it at
    `destination`."""
    with urllib.request.urlopen(USER_GUIDE.raw_url, timeout=60) as response:
        data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != USER_GUIDE.sha256:
        raise MossSourceError(f"{USER_GUIDE.raw_url}: sha256 {actual} is not {USER_GUIDE.sha256}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".staging")
    staging.write_bytes(data)
    staging.replace(destination)
    return destination


def local_user_guide(path: Path = USER_GUIDE_PATH) -> str:
    """The text of the kept user guide after checking its SHA-256 (no network; fails loudly)."""
    if not path.is_file():
        raise MossSourceError(
            f"{path} is missing — fetch it with"
            " `uv run --group moss --inexact python -m moss.sources --fetch --guide`"
        )
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != USER_GUIDE.sha256:
        raise MossSourceError(f"{path}: sha256 {actual} is not the pinned {USER_GUIDE.sha256}")
    return data.decode("utf-8")


def compare_vae_origin(snapshot: Path, *, fetch: bool) -> dict[str, object]:
    """Compare the bundled VAE with {@link VAE_ORIGINS}, tensor by tensor, after checking each
    origin file's SHA-256.

    Every bundled tensor must fall under exactly one origin's prefixes and equal that origin's
    tensor bit for bit after casting it to the bundled dtype (bf16 → f32 is exact; integers are not
    cast). Returns the summary; mismatches are listed, not raised — the caller reports the finding.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    bundled = snapshot / VAE_FILE
    origins: list[dict[str, object]] = []
    with safe_open(str(bundled), framework="pt") as mine:
        names = sorted(mine.keys())
        owners = {name: [o for o in VAE_ORIGINS if name.startswith(o.prefixes)] for name in names}
        unowned = sorted(name for name, found in owners.items() if len(found) != 1)
        for origin in VAE_ORIGINS:
            path = Path(
                hf_hub_download(
                    origin.source.repo,
                    origin.file,
                    revision=origin.source.revision,
                    local_files_only=not fetch,
                )
            )
            actual = sha256_of(path)
            if actual != origin.sha256:
                raise MossSourceError(f"{path}: sha256 {actual} is not {origin.sha256}")
            owned = [name for name in names if owners[name] == [origin]]
            with safe_open(str(path), framework="pt") as theirs:
                present = set(theirs.keys())
                missing: list[str] = []
                differing: list[str] = []
                origin_dtypes: set[str] = set()
                # One read per origin tensor (the comparison and the dtype tally share it).
                for name in owned:
                    if name not in present:
                        missing.append(name)
                        continue
                    tensor = theirs.get_tensor(name)
                    origin_dtypes.add(str(tensor.dtype))
                    if not _same_tensor(mine.get_tensor(name), tensor):
                        differing.append(name)
                dtypes = sorted(origin_dtypes)
            origins.append(
                {
                    "repo": origin.source.repo,
                    "revision": origin.source.revision,
                    "license": origin.source.license,
                    "file": origin.file,
                    "sha256": actual,
                    "prefixes": list(origin.prefixes),
                    "tensors": len(owned),
                    "origin_dtypes": dtypes,
                    "missing": missing,
                    "differing": differing,
                }
            )
        return {
            "bundled": {"file": VAE_FILE, "sha256": sha256_of(bundled), "tensors": len(names)},
            "unowned": unowned,
            "origins": origins,
            "identical": not unowned
            and all(not o["missing"] and not o["differing"] for o in origins),
        }


def _same_tensor(bundled: torch.Tensor, origin: torch.Tensor) -> bool:
    """Same shape and the same bytes once `origin` is cast to the bundled dtype (floating-point
    origins only — bit patterns are compared, not float equality)."""
    import torch

    if bundled.shape != origin.shape:
        return False
    if origin.dtype != bundled.dtype:
        if not (origin.is_floating_point() and bundled.is_floating_point()):
            return False
        widened = origin.to(bundled.dtype)
        if not torch.equal(widened.to(origin.dtype), origin):
            return False
        origin = widened
    return bool(
        torch.equal(
            bundled.contiguous().reshape(-1).view(torch.uint8),
            origin.contiguous().reshape(-1).view(torch.uint8),
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--fetch", action="store_true", help="fetch the pinned revision")
    parser.add_argument("--text", action="store_true", help="the text encoder and tokenizer")
    parser.add_argument(
        "--vae-origin", action="store_true", help="compare the VAE with its FLUX.2 origins"
    )
    parser.add_argument("--guide", action="store_true", help="the upstream USERGUIDE.md (GitHub)")
    args = parser.parse_args(argv)

    if args.guide:
        path = fetch_user_guide() if args.fetch else USER_GUIDE_PATH
        local_user_guide(path)
        print(f"{path}: sha256 {USER_GUIDE.sha256} ({USER_GUIDE.url})")
        return 0

    if args.vae_origin:
        report = compare_vae_origin(local_snapshot(), fetch=args.fetch)
        print(json.dumps(report, indent=1))
        return 0 if report["identical"] else 1
    snapshot = _snapshot(text=args.text, fetch=args.fetch)
    print(f"snapshot: {snapshot}")
    if args.text:
        counts = check_snapshot(snapshot, ("text_encoder",))
    else:
        verify_pipeline_code(snapshot)
        print(
            f"{PIPELINE_CODE}: sha256 {PIPELINE_CODE_SHA256}"
            f" (all {len(PIPELINE_CODE_COPIES)} copies)"
        )
        counts = check_snapshot(snapshot, ("transformer", "vae"))
        for name, digest in verify_weights(snapshot).items():
            print(f"{name}: sha256 {digest}")
    for name, count in counts.items():
        print(f"{name}: {count:,} elements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
