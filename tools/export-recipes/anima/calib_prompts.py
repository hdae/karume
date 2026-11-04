"""Anima's calibration corpus (24 danbooru-style tag prompts; the first 4 by default).

The prompts produce the activations GPTQ sees.

The calibrated rounding of `karume.quant_calib` re-picks the rounding targets from "the activations
that actually flow through that layer", so the character of the corpus becomes the bias of the
rounding as is. The DiT's activations move with **the prompt (= the condition fed to cross
attention), the resolution and sigma**, so the selection follows 3 rules:

- **Keep it separate from the evaluation inputs** (maintainer ruling of 2026-08-21) — if prompts
  seen in calibration were used for evaluation as they are, PSNR / visual review would measure
  "could it reproduce the images it saw in calibration", measuring leakage rather than calibration
  quality. The sources of truth for the evaluation inputs are the fixed prompts of
  `anima.pipeline_ref` and the TS-side E2E (extracted from the real file — a copy would let one
  side go stale), plus the 4 cases of the visual evaluation (their source of truth, `examples/`,
  was removed in the port, so they live in {@link EVAL_IMAGE_SUBJECTS}); {@link evaluation_prompts}
  collects them and {@link assert_calib_disjoint} compares.
- **Lean toward Anima's domain** — the model specialises in anime images (danbooru-style tags), so
  the calibration uses the same tag vocabulary. **Spread the subject (0–4 people, animals), the
  composition (close-up / wide / from above / from below / back view) and the style (chibi /
  lineart / watercolor / film grain)** — leaning toward one type makes the activation bias stick to
  that type.
- **Used from the top** (`--calib-prompts N` takes **the first N**), so the order is **mixed**
  rather than grouped by type — even a reduced run (the default 4 is one) keeps the mix of subject,
  composition and style.

NOTE (the axes for better quality): calibration quality rises with **the prompt count** and **the
resolution** (the count spreads the activations across prompts; the resolution is the number of
tokens per prompt = the number of samples of `H = Σ XᵀX`). Both scale the export's CPU time
linearly. The resolution is fixed together with the conditions of the quality verdict
({@link anima.calib.CALIB_RESOLUTION}), so only the count can move. This corpus covers up to 24;
going beyond that means adding prompts here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from _shared.paths import REPO_ROOT

from .pipeline_ref import NEGATIVE_PROMPT, PROMPT

#: Default of `--calib-prompts` (**this many from the top** of {@link CALIB_PROMPTS}).
#:
#: **Set at the lower bound that has evidence**: the calibration conditions of run J-2, where the
#: quality verdict (PSNR 22.73dB, visually nearly identical to f32) was reached, were 1 prompt × 10
#: steps = 10 batches, so 4 prompts × 8 steps = 32 batches does not fall below that. On the other
#: hand, **the extra gain of going to 12 / 24 is unmeasured**, and the cost (by extrapolating {@link
#: anima.calib.CALIB_SECONDS_PER_BATCH}, CPU time of 4 ≈ 38 minutes and 12 ≈ 1.9 hours, plus the
#: memory for `current` and the next-stage list being alive at once) grows linearly with the count.
#: Measure before raising it (`--calib-prompts N` goes up to 24).
DEFAULT_CALIB_PROMPTS = 4

#: Prompts of the calibration inputs (used from the top in this order).
CALIB_PROMPTS: tuple[str, ...] = (
    "2girls, library, bookshelves, reading, sitting, window light, detailed background",
    "scenery, rice field, summer sky, cumulonimbus, power lines, no humans, wide angle",
    "chibi, deformed, cat ears, simple background, white background, full body, sticker",
    "1girl, twintails, cyberpunk city, neon signs, rain, umbrella, night, reflections",
    "animal focus, dog, park, running, grass, shallow depth of field, action shot",
    "monochrome, greyscale, sketch, lineart, 1boy, portrait, serious expression, hatching",
    "no humans, empty classroom, desks, chalkboard, dust motes, afternoon, interior",
    "1girl, swimsuit, ocean, seagulls, bright sunlight, from above, dutch angle",
    "1boy, samurai, armor, bamboo forest, mist, katana, dynamic pose, low angle",
    "watercolor, traditional media, flower field, pastel colors, soft edges, no humans",
    "4girls, idol, stage, spotlights, microphones, confetti, concert, wide shot",
    "no humans, still life, teacup, wooden table, steam, morning light, close-up",
    "1boy, rooftop, sunset, city skyline, wind, jacket, back view, wide shot",
    "1girl, kimono, festival, paper lantern, night, crowd, bokeh, from side",
    "no humans, robot, workshop, sparks, tools, industrial, dramatic lighting",
    "3girls, cafe, coffee, chatting, indoors, warm colors, medium shot",
    "1girl, witch hat, forest, mushrooms, fireflies, night, fantasy, glowing particles",
    "1boy, 1girl, dancing, ballroom, formal dress, chandelier, motion blur, elegant",
    "1girl, mecha pilot, cockpit, holographic display, blue glow, tight framing, sci-fi",
    "scenery, snow, mountain village, chimney smoke, dusk, no humans, muted palette",
    "1girl, apron, kitchen, baking, flour, messy hair, laughing, candid, warm light",
    "no humans, underwater, coral reef, fish, sun rays, caustics, deep blue",
    "1girl, glasses, office, laptop, late night, desk lamp, tired, from behind",
    "1girl, hoodie, train station, platform, evening, vending machine, film grain",
)

#: Source of truth of an evaluation input (repo-relative). It is on the TS side and cannot be
#: imported, so it is **extracted from the real file** (holding a copy lets one side go stale and
#: silently loosens the separation check). An empty extraction fails loudly.
E2E_SOURCE = Path("packages/e2e/tests/e2e_anima_test.ts")

#: The 4 cases of the visual evaluation (eval-images). Their source of truth,
#: `examples/anima/eval-images.ts`, was deleted in the 0.15 port (ADR 0127 — the measurement tools
#: and examples are rebuilt as browser apps), so until it returns **this is the only place they
#: live**. They are composed as `<prefix>, <subject>, <suffix>`, the same as `prompt()` there.
#: NOTE: when the app is rebuilt, read these 4 back from there — two copies would let the separation
#: check go stale on one side (making this constant the source of truth and having the TS side read
#: it is fine too).
EVAL_IMAGE_QUALITY_PREFIX = "score_9, score_8_up, score_7_up"
EVAL_IMAGE_QUALITY_SUFFIX = "masterpiece, best quality"
EVAL_IMAGE_SUBJECTS: tuple[str, ...] = (
    "1girl, solo, school uniform, cherry blossoms, outdoors, smile, upper body",
    "scenery, mountain, lake, forest, no humans, blue sky, clouds",
    "school hallway, perspective, vanishing point, indoor, windows, sunlight, no humans",
    "1girl, solo, full body, standing, city street, buildings, detailed background",
)

#: One TS double-quoted string (escapes included).
_STRING = r'"((?:[^"\\]|\\.)*)"'


def _read_source(source: Path) -> str:
    path = REPO_ROOT / source
    if not path.is_file():
        raise AssertionError(
            f"The source of truth of the evaluation input {source} is missing (the reference for"
            " the separation check against the calibration corpus cannot be read)"
        )
    return path.read_text(encoding="utf-8")


def _declaration(text: str, source: Path, name: str) -> str:
    """Returns the string literals of `const <name> = "…" + "…";` concatenated."""
    match = re.search(rf"const {name}\s*=\s*(.*?);", text, re.DOTALL)
    parts = re.findall(_STRING, match.group(1)) if match is not None else []
    if not parts:
        raise AssertionError(
            f"{source}: could not extract a string from const {name}"
            " (the code has changed — an empty reference for the separation check is more"
            " dangerous, so this fails)"
        )
    return "".join(parts)


def _eval_image_prompts() -> tuple[str, ...]:
    """Prompts of the 4 visual-evaluation cases (composed from {@link EVAL_IMAGE_SUBJECTS})."""
    return tuple(
        f"{EVAL_IMAGE_QUALITY_PREFIX}, {subject}, {EVAL_IMAGE_QUALITY_SUFFIX}"
        for subject in EVAL_IMAGE_SUBJECTS
    )


def evaluation_prompts() -> tuple[str, ...]:
    """All evaluation inputs that **must stay separate** from calibration (from 3 sources of truth).

    - the fixed prompts of `anima.pipeline_ref` (the reference fixture = numeric parity's input)
    - `PROMPT` of the TS-side E2E (`packages/e2e/tests/e2e_anima_test.ts`)
    - the 4 visual-evaluation cases ({@link EVAL_IMAGE_SUBJECTS} — originally
      `examples/anima/eval-images.ts`)
    """
    return (
        PROMPT,
        NEGATIVE_PROMPT,
        _declaration(_read_source(E2E_SOURCE), E2E_SOURCE, "PROMPT"),
        *_eval_image_prompts(),
    )


def assert_calib_disjoint(prompts: Sequence[str], evaluated: Sequence[str]) -> None:
    """Checks the calibration corpus does not overlap the evaluation inputs, **not even partially**.

    MUST: fail loudly (the same form as `sbv2.measure_quant.assert_calib_disjoint`). An overlap
    makes PSNR / visual review measure "could it reproduce the images it saw in calibration",
    measuring leakage rather than calibration quality.
    """
    hits = sorted(
        {text for text in prompts for body in evaluated if body and (body in text or text in body)}
    )
    if hits:
        raise AssertionError(
            f"The calibration corpus overlaps the evaluation inputs: {hits[:3]}"
            f" ({len(evaluated)} evaluation inputs)"
        )


def calibration_prompts(count: int) -> tuple[str, ...]:
    """Returns the first `count` prompts with the separation check (what `--calib-prompts` does)."""
    if not 1 <= count <= len(CALIB_PROMPTS):
        raise ValueError(
            f"--calib-prompts {count} is out of range 1–{len(CALIB_PROMPTS)}"
            f" (the corpus has {len(CALIB_PROMPTS)} prompts — to grow it, add prompts to"
            " calib_prompts.py)"
        )
    prompts = CALIB_PROMPTS[:count]
    assert_calib_disjoint(prompts, evaluation_prompts())
    return prompts
