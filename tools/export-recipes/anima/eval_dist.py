"""Assembly for visual evaluation only (**not distributed**) of the i4 sensitivity variants.

A script for viewing the i4 quantization-sensitivity experiment variants locally.

The i4 of the 3 plain models was skipped for distribution by the visual verdict (research
`2026-08-24-gptq-expansion-quality.md` §5), and the i4 preset is gone from
{@link anima.distribution.ANIMA_MODELS} as well. Viewing the improvement candidate (the variant
that stores in-block adaLN + what is outside the blocks as i8 — `--i4-adaln-i8` of `anima.export`)
in an A/B needs a distribution with the i4 preset, but there is no decision yet to put that preset
back into the distribution's accepted set. This is an entry point that fills only that one gap,
and it **does not touch a single byte of the distribution path**:

- The preset is put back by overriding `spec` in {@link anima.distribution.anima_plan} (it only
  passes a spec with i4 added — `ANIMA_MODELS` does not move).
- The i4 series is looked up by the variant's spelling ({@link EVAL_SERIES}). The `*-i4-dyn`
  baked under the distribution conditions is **never looked at** — no room for a series rounded
  another way to enter the same slot.
- The required rounding method is {@link anima.distribution.ADALN_I8_CALIB_METHOD}. It is spelled
  differently from the distribution path's `gptq`, so assembling the variant with `dist.py` fails
  on the method match, and conversely passing a series baked under the distribution conditions
  here fails too (the mix-up is stopped **in both directions**).
- The default output location is {@link EVAL_ROOT} (the `outputs/bench/` side = safe to delete).
  `models/` is the place for distributions only (the DECIDED of `_shared.paths`), so by default it
  is never touched.

    # ① bake the variant's i4 series (GPTQ calibration included, measured ~3h). The series name is
    #   spelled as in {@link EVAL_SERIES}.
    uv run python -m anima.export --dtype i4 --dit-graph dyn --i4-adaln-i8 \\
        --model anima-v1.0 --out ../../outputs/series/anima-v1.0-i4-adaln8-dyn
    # ② assemble for viewing (default out = outputs/bench/karume-anima-v1.0-adaln8/<date>_eval/)
    uv run python -m anima.eval_dist --model anima-v1.0

MUST: do not upload the resulting distribution to HF and do not move it into `models/`. The quant
preset name stays `f16+dit4` as in the distribution (ADR 0074's grammar), and **only the bytes
inside differ** — it cannot be told apart by name, so keeping it in a separate place is the only
distinction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from functools import partial
from pathlib import Path

from _shared.paths import BENCH_ROOT, SERIES_ROOT
from anima.card import render_base_card
from anima.distribution import (
    ADALN_I8_CALIB_METHOD,
    ADALN_I8_TAG,
    ANIMA_BASE_MODEL_NAME,
    ANIMA_MODELS,
    ANIMA_QUANT_ABBREVIATIONS,
    CONTAINER_MODIFICATION,
    anima_model,
    anima_plan,
    anima_sources,
    notice_markdown,
    root_files,
)
from karume.dist import DistError, ModelPlan, Pipeline
from karume.dist import main as dist_main

#: Directory name of the variant's i4 series (relative to the `--series` parent). It differs from
#: the default out of `anima.export --i4-adaln-i8` (`anima-i4-adaln8-dyn`) only by the model name —
#: as with the distribution's series, the plain models are split by model name (matching the
#: spelling of {@link anima.distribution.anima_sources}).
EVAL_SERIES = f"{{model}}-i4-{ADALN_I8_TAG}-dyn"

#: Parent of the evaluation assembly's output (`<EVAL_ROOT>/<repository name>/<date>_eval/`).
#: **Never placed in `models/`** (the place for distributions only — `_shared.paths`). It is a
#: benchmark output, so it goes where deleting is safe.
EVAL_ROOT = BENCH_ROOT

#: Spelling of `--pipeline` (the accepted set is this one only — it is not mixed into the
#: distribution's table (`tools/export-recipes/dist.py`) because that table is "every pipeline that
#: may be uploaded to HF").
EVAL_PIPELINE = "anima-i4-adaln8-eval"

#: Modification notice of the visual-evaluation assembly. To the distribution's notice (up to the
#: added int8 / bf16 series) it adds a line for the i4 series whose preset is put back here (the
#: adaLN variant's GPTQ calibration) — if the contents held i4 but the notice did not say so, the
#: rule at `root_files` below, "do not practise building a distribution without the modification
#: notice", would not hold.
EVAL_NOTICE_MARKDOWN = notice_markdown(
    (
        CONTAINER_MODIFICATION,
        "- An int8-quantized series of the transformer was added alongside the f16 one.",
        "- A bf16 series was added alongside the f16 one. It keeps the values of the upstream BF16",
        "  weights unchanged (no rounding or quantization); the f16 series rounds them to f16.",
        "- A GPTQ-calibrated int4-quantized series of the transformer was added for local",
        "  evaluation; its adaLN linears and the linears outside the blocks stay int8.",
    )
)


def eval_plan(series_dir: Path, model: str) -> ModelPlan:
    """Builds the visual-evaluation plan (puts the i4 preset back).

    It requires the variant's series and rounding method.

    MUST: only Anima models **without an i4 preset in the distribution** are accepted (since the
    2026-09-01 ruling that removed i4 from every model, all of them qualify). A model whose i4
    preset returns to the distribution in the future can use `dist.py`'s distribution path as is,
    so letting it through here would multiply "the same preset name as the distribution with
    different contents" for no reason — the storages check below rejects it.
    """
    if model not in ANIMA_MODELS:
        raise DistError(
            f"Model {model!r} is not a visual-evaluation target"
            f" (targets: {', '.join(ANIMA_MODELS)})"
        )
    spec = anima_model(model)
    if "i4" in spec.storages:
        raise DistError(
            f"Model {model!r} already has an i4 preset in the distribution — assemble it with"
            " dist.py, not through the visual-evaluation entry point (if the preset is back, this"
            " script has done its job)"
        )
    sources = anima_sources(series_dir, model)
    return anima_plan(
        replace(
            sources,
            transformer={
                **sources.transformer,
                "i4": series_dir / EVAL_SERIES.format(model=model),
            },
        ),
        model,
        spec=replace(
            spec,
            storages=(*spec.storages, "i4"),
            calib_method=ADALN_I8_CALIB_METHOD,
        ),
    )


PIPELINE = Pipeline(
    default_model=ANIMA_BASE_MODEL_NAME,
    repo_name=lambda model: f"karume-{model}-{ADALN_I8_TAG}",
    plan=eval_plan,
    card_profiles={"anima": partial(render_base_card, abbreviations=ANIMA_QUANT_ABBREVIATIONS)},
    # The legal texts are the same as the distribution's — even though it is only opened locally,
    # do not practise building a distribution without the modification notice (an accidental
    # upload would not be caught here).
    root_files=root_files(EVAL_NOTICE_MARKDOWN),
)


def default_out_dir(pipeline: Pipeline, models: Sequence[str]) -> Path:
    """Output location when `--out` is omitted (`outputs/bench/<repository name>/<date>_eval/`).

    A repository name cannot be derived for several models (the same reason as the function of the
    same name in `dist.py`) — it fails, asking for an explicit one.
    """
    if len(models) != 1:
        raise DistError(
            f"Cannot derive a repository name when assembling {len(models)} models — specify the"
            " output location with --out"
        )
    return EVAL_ROOT / pipeline.repo_name(models[0]) / f"{date.today().isoformat()}_eval"


def main(argv: Sequence[str] | None = None) -> None:
    dist_main(
        argv,
        pipelines={EVAL_PIPELINE: PIPELINE},
        default_pipeline=EVAL_PIPELINE,
        default_out_dir=default_out_dir,
        default_series=SERIES_ROOT,
    )


if __name__ == "__main__":
    main()
