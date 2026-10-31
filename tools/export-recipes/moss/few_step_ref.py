"""The 2-step end-to-end CPU reference of MOSS-Image: upstream `pipe(...)` itself with the DiT's
linear weights rounded to the i8 series' values (stage 4 of ADR 0138 — the opt-in upstream parity of
`packages/e2e/tests/e2e_moss_generate_test.ts`; the Wan precedent `wan.ti2v_few_step_ref`).

## The run

- Upstream `MOSSImagePipeline.__call__` on CPU in f32: the DiT read in f32 and rounded in place to
  the i8 grid (`karume.quantize.fake_quant_int8` through `moss.export_dit.fake_quant` — the export's
  own rounding), then checked bit for bit against the written `dit` and `dit_text` containers
  (`moss.export_dit.container_weights` — every i8 weight dequantizes to the weight this run uses,
  every f32 tensor equals it). The reference therefore runs the values the GPU reads, and the band
  measures the GPU's arithmetic and the wiring, not the i8 rounding of the weights. The VAE is
  upstream's f32 (the VAE series stores those values — `moss.export_vae` gate 2).
- Per case: the fixed-prompt embeddings (`moss.text_embeds` — the positive by name, the empty
  negative), seeded torch `randn` latents injected as `latents=` (`pipeline_ref.seeded_latents`),
  {@link STEPS} steps, CFG {@link GUIDANCE} with the skip-22 negative (upstream's defaults),
  `output_type="pt"`. Each step's output latents are kept by wrapping `scheduler.step`; the decode
  is kept by `pipeline_ref.VaeRecorder` and clamped as upstream clamps (`decode(z).clamp(-1, 1)`).
- The gate (MUST — nothing is written unless it holds): the containers' weights are the run's; every
  case ran {@link STEPS} scheduler steps; the decode input upstream recorded is the BatchNorm
  restoration and unpatchify of the last step's latents (`pipeline_ref.denormalize` /
  `unpatchify` — so the recorded latents are the ones that were decoded); every value is finite.

## Output (`<case>.safetensors` in {@link REFERENCE_DIR}, beside `moss.dit_reference`'s goldens)

- tensors (no batch axis): `latents_init`, `latents.0`, `latents.1` `[128, H/16, W/16]` (the input
  noise and each step's output) and `image` `[3, H, W]` (the clamped decode, in [-1, 1]);
- metadata (one key, {@link METADATA_KEY}): the case, its role, size, seed and prompt, the token
  counts, the schedule upstream ran (`sigmas`, `timesteps`), the part-0 sha256 of the six
  containers whose values the reference stands for (the TS side refuses a golden of other
  containers), the embedding asset's sha256, the versions and the CPU environment.

## Cases ({@link CASES} — `packages/e2e/tests/e2e_moss_generate_test.ts` fixes the same table)

Per size — 256² and 256×384 (height × width, the stage-0 sizes) — three band cases and two
acceptance cases, with other seeds and prompts. MUST: the acceptance results change neither the
band cases nor the bands.

    uv run --group moss --inexact python -m moss.few_step_ref                       # every case
    uv run --group moss --inexact python -m moss.few_step_ref --case <name> ...      # some

RAM: the f32 DiT (20.3 GB) and the containers' check dominate; the run summary records the peak
(`outputs/bench/karume-moss-image/<date>_few-step-ref/write.json`).

MUST: diffusers is imported inside the functions (`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from _shared.paths import BENCH_ROOT, SERIES_ROOT
from moss import dit_reference, export_dit, export_vae, pipeline_ref, records, sources, text_embeds
from moss.dit_patch import DIT, DIT_TEXT
from moss.prompts import NEGATIVE, POSITIVE, prompt_by_name
from moss.records import bits_equal
from moss.vae_patch import GRAPHS as VAE_GRAPHS

#: Where the goldens go (beside `moss.dit_reference`'s — not inside the series, which `export_dit
#: write` replaces as a whole).
REFERENCE_DIR = dit_reference.REFERENCE_DIR

#: The single metadata key of a golden.
METADATA_KEY = "karume.moss.few_step_ref"

#: Requested steps (both survive the snapping: σ = 1 and one value near 0.024 — `moss.reference`).
STEPS = 2

#: Upstream's guidance and the negative branch's skip.
GUIDANCE = pipeline_ref.GUIDANCE_SCALE
SKIP_LAYERS = pipeline_ref.UNCOND_SKIP_LAYERS

#: `(height, width)` in pixels (`moss.reference.SIZES`).
SIZES: tuple[tuple[int, int], ...] = ((256, 256), (256, 384))

#: The embedding asset.
TEXT_EMBEDS = SERIES_ROOT / text_embeds.SERIES_NAME / text_embeds.ASSET_NAME

#: The summary's directory (docs/assets-layout.md — bench records are safe to delete).
BENCH_PURPOSE = "few-step-ref"


class FewStepReferenceError(RuntimeError):
    """The reference did not run as upstream's pipeline with the series' weights."""


@dataclass(frozen=True)
class Case:
    """One golden: a request of the generate path."""

    role: str
    height: int
    width: int
    prompt: str
    seed: int

    @property
    def name(self) -> str:
        return f"pipeline-{self.height}x{self.width}-{self.role}-{self.prompt}-s{self.seed}"


#: Per size: `(role, positive prompt, seed)` — three band cases, two acceptance cases.
_PLAN: tuple[tuple[str, str, int], ...] = (
    ("band", "fridge-note", 20261201),
    ("band", "guide-whiteboard", 20261202),
    ("band", "guide-fridge-note", 20261203),
    ("accept", "guide-whiteboard", 20261204),
    ("accept", "fridge-note", 20261205),
)

CASES: tuple[Case, ...] = tuple(
    Case(role, height, width, prompt, seed)
    for height, width in SIZES
    for role, prompt, seed in _PLAN
)


def case_by_name(name: str) -> Case:
    found = [case for case in CASES if case.name == name]
    if len(found) != 1:
        raise FewStepReferenceError(f"no case {name!r} ({', '.join(c.name for c in CASES)})")
    return found[0]


def container_digests() -> dict[str, str]:
    """The part-0 sha256 of the six containers the goldens stand for (the DiT's three, the VAE's
    three — `dit_reference.part0_sha256`)."""
    paths = {
        **export_dit.container_paths(export_dit.SERIES),
        **{name: export_vae.SERIES / name / export_vae.MODEL_FILE for name in VAE_GRAPHS},
    }
    return {name: dit_reference.part0_sha256(path) for name, path in sorted(paths.items())}


def load_pipeline(snapshot: Path, module: ModuleType, clock: records.Clock) -> tuple[Any, Any]:
    """Upstream's pipeline with the fake-quant DiT checked against the series (module docstring);
    returns it and the check's report."""
    transformer = pipeline_ref.load_transformer(snapshot, module, torch.float32)
    clock.lap("load transformer (f32)")
    graphs = {name: export_dit.wrap(name, transformer) for name in export_dit.GRAPHS}
    quantized = export_dit.fake_quant(transformer, graphs)
    del graphs
    clock.lap("fake-quant i8")
    paths = export_dit.container_paths(export_dit.SERIES)
    checked = {
        name: export_dit.container_weights(paths[name], transformer) for name in (DIT, DIT_TEXT)
    }
    differing = {
        name: report["differing"][:5] for name, report in checked.items() if report["differing"]
    }
    if differing:
        raise FewStepReferenceError(f"the series' weights are not the fake-quant ones: {differing}")
    clock.lap("container weights")
    pipe = pipeline_ref.build_pipeline(
        module,
        transformer=transformer,
        scheduler=pipeline_ref.load_scheduler(snapshot, module),
        vae=pipeline_ref.load_vae(snapshot),
    )
    return pipe, {"fake_quant": quantized, "container_weights": checked}


def run_case(
    pipe: Any,
    positive: torch.Tensor,
    negative: torch.Tensor,
    latents_init: torch.Tensor,
    *,
    height: int,
    width: int,
    steps: int = STEPS,
    skip_layers: Sequence[int] = SKIP_LAYERS,
) -> dict[str, torch.Tensor]:
    """One upstream `pipe(...)` call (module docstring "The run") and its gate items; the tensors of
    the golden (`latents_init` `[1, C, h, w]` in, every tensor out without the batch axis)."""
    scheduler = pipe.scheduler
    outputs: list[torch.Tensor] = []
    step = scheduler.step

    def recording_step(*args: Any, **kwargs: Any) -> Any:
        result = step(*args, **kwargs)
        outputs.append(result[0].detach().clone())
        return result

    # An instance attribute shadows the class's method for the call (removed afterwards).
    if "step" in vars(scheduler):
        raise FewStepReferenceError("the scheduler already has an instance-level step")
    scheduler.step = recording_step
    vae_recorder = pipeline_ref.VaeRecorder(pipe.vae)
    try:
        with torch.no_grad():
            pipe(
                prompt_embeds=positive[None],
                negative_prompt_embeds=negative[None],
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=GUIDANCE,
                uncond_skip_layers=tuple(skip_layers),
                latents=latents_init.clone(),
                output_type="pt",
            )
    finally:
        del scheduler.step
        vae_recorder.close()
    if len(outputs) != len(scheduler.timesteps):
        raise FewStepReferenceError(
            f"{height}x{width}: {len(outputs)} scheduler steps"
            f" for {len(scheduler.timesteps)} timesteps"
        )
    z = vae_recorder.tensors["vae.z"]
    if not bits_equal(pipeline_ref.unpatchify(pipeline_ref.denormalize(pipe.vae, outputs[-1])), z):
        raise FewStepReferenceError(f"{height}x{width}: the decode input is not the last step's")
    image = vae_recorder.tensors["vae.decoded"].clamp(-1, 1)
    tensors = {
        "latents_init": latents_init[0],
        **{f"latents.{index}": latents[0] for index, latents in enumerate(outputs)},
        "image": image[0],
    }
    broken = [name for name, tensor in tensors.items() if not bool(torch.isfinite(tensor).all())]
    if broken:
        raise FewStepReferenceError(f"{height}x{width}: non-finite {broken}")
    return tensors


def write(cases: Sequence[Case], out: Path) -> dict[str, Any]:
    """Load once, run every case, write one golden per case; return the run summary."""
    clock = records.Clock()
    digests = container_digests()
    snapshot, module = pipeline_ref.upstream()
    embeds, embeds_metadata = text_embeds.read_asset(TEXT_EMBEDS)
    negatives = [e["name"] for e in embeds_metadata["prompts"] if e["role"] == NEGATIVE]
    if len(negatives) != 1:
        raise FewStepReferenceError(f"the asset holds negatives {negatives} (exactly one)")
    pipe, check = load_pipeline(snapshot, module, clock)
    clock.lap("load vae and scheduler")
    written: list[dict[str, Any]] = []
    for case in cases:
        if prompt_by_name(case.prompt).role != POSITIVE:
            raise FewStepReferenceError(f"{case.name}: {case.prompt!r} is not a positive prompt")
        positive, negative = embeds[case.prompt], embeds[negatives[0]]
        tensors = run_case(
            pipe,
            positive,
            negative,
            pipeline_ref.seeded_latents(case.seed, case.height, case.width),
            height=case.height,
            width=case.width,
        )
        clock.lap(f"case {case.name}")
        metadata = {
            "case": case.name,
            "role": case.role,
            "size": [case.height, case.width],
            "seed": case.seed,
            "prompt": case.prompt,
            "tokens": {POSITIVE: int(positive.shape[0]), NEGATIVE: int(negative.shape[0])},
            "steps": STEPS,
            "guidance_scale": GUIDANCE,
            "uncond_skip_layers": list(SKIP_LAYERS),
            "sigmas": [float(s) for s in pipe.scheduler.sigmas],
            "timesteps": [float(t) for t in pipe.scheduler.timesteps],
            "noise": "torch.randn(generator=torch.Generator().manual_seed(seed), dtype=float32)",
            "weights": "DiT: f32 fake-quant i8 (export_dit.fake_quant) =="
            " the series' dit / dit_text; VAE: upstream f32",
            "containers": digests,
            "text_embeds_sha256": sources.sha256_of(TEXT_EMBEDS),
            "source": {
                "repo": sources.SOURCE.repo,
                "revision": sources.SOURCE.revision,
                "pipeline_code_sha256": sources.PIPELINE_CODE_SHA256,
            },
            "versions": records.versions(),
            "environment": records.environment(),
        }
        path = out / f"{case.name}.safetensors"
        records.write_record(path, tensors, metadata, key=METADATA_KEY)
        written.append({"case": case.name, "path": str(path), "bytes": path.stat().st_size})
    return {
        "cases": written,
        "check": check,
        "containers": digests,
        "seconds": {**clock.phases, "total": clock.total()},
        "peak_rss_gib": round(records.peak_rss_gib(), 2),
        "passed": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--case", action="append", default=None, help="a case name (repeatable)")
    parser.add_argument("--out", type=Path, default=REFERENCE_DIR)
    args = parser.parse_args(argv)
    cases = CASES if args.case is None else tuple(case_by_name(name) for name in args.case)
    summary = write(cases, args.out)
    document: Mapping[str, Any] = {"result": summary, "versions": records.versions()}
    text = json.dumps(document, ensure_ascii=False, indent=1)
    print(text)
    bench = BENCH_ROOT / records.BENCH_NAME / f"{date.today().isoformat()}_{BENCH_PURPOSE}"
    bench.mkdir(parents=True, exist_ok=True)
    (bench / "write.json").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
