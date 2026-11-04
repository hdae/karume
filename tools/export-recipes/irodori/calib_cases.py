"""Irodori's calibration corpus (12 full-loop cases that produce the activations GPTQ sees).

The calibrated rounding of `karume.quant_calib` re-picks the rounding targets from "the activations
that actually flow into that layer", so the character of the corpus becomes the bias of the
rounding. The DiT's activations move with **the text (length and content of self-attention), the
caption / speaker conditions and t**, so the selection follows three rules:

- **Keep it disjoint from the evaluation inputs** (same root as anima's user ruling of
  2026-08-21) — evaluating on the cases seen in calibration turns LSD / SNR / listening into a
  measure of "could it reproduce the audio it saw during calibration", measuring **leakage**
  instead of calibration quality. Irodori's evaluation inputs have a single source of truth,
  {@link irodori.pipeline_ref.PIPELINE_CASES} (the full-loop golden, `irodori.measure_quant` and
  the Deno-side E2E all follow this golden), so no copy is kept: {@link assert_calib_disjoint}
  reads them from there directly and compares. Disjointness is checked on **all three axes,
  text / caption / seed** — sharing any one of them lets activations of the same trajectory into
  calibration.
- **Match Irodori's domain** — it is a Japanese TTS, so calibration also uses Japanese utterances,
  spread over **speaking style, speaker profile (gender × age) and tempo**. Concentrating on one
  type makes the activation bias stick to that type. There are 12 speaking styles (narration /
  conversation / public announcement / storytelling / whisper / emotional line / news reading /
  sports commentary / reading to children / brisk business talk / poetry recital / monologue),
  chosen so that each of the three axes — volume (whisper to shouting), intonation (flat to
  exaggerated) and tempo (fast talk to a recital with long pauses) — reaches both ends.
- **Enable all three conditions** (reference latent present + non-empty caption) — it is the
  distribution's default path and the only regime in which the DiT's cross-attention runs over all
  three segments. Leaving the no-reference / empty-caption regimes out of calibration is an
  intended boundary; those are covered by the **evaluation side** (the `no-ref` case).

NOTE (where the case count and step count stand): calibration quality goes up with **the number of
cases** and **the number of captured steps** (the former is the spread of activations across
cases, the latter the coverage of t per case = the sample count of `H = Σ XᵀX`). Both scale the
export's CPU time linearly. The step count defaults to {@link irodori.calib.CALIB_STEPS} (= the
full reference loop) and can be lowered with `--calib-steps`, but **the case count is fixed at
12** — to add more, add the text here (there is no knob like `--calib-cases`: if the number used
moved, the `cases` of `calib_provenance.json` would change from run to run and the rounding
conditions of the distributed assets could no longer be read).
The increase from 4 to 12 cases came from the listening ruling of 2026-08-23 — a sim calibrated on
the evaluation sentences themselves showed almost no degradation (= the 4-bit grid is expressive
enough), while the w4 calibrated on 4 cases generalized noticeably worse, and the second axis of
the muffling was attributed to the corpus.
"""

from __future__ import annotations

from collections.abc import Sequence

from .pipeline_ref import PIPELINE_CASES, PipelineCase, ReferenceSpec

#: The calibration input cases (all are used, in this order).
#:
#: `frames` is the reference latent's frame count before patching, a value divisible by
#: `speaker_patch_size` (the MUST of {@link irodori.pipeline_ref.ReferenceSpec}). They stay in the
#: same band of multiples of 4 as the 124 the evaluation cases use, so that the reference length
#: itself is not pushed into an unverified range.
CALIB_CASES: tuple[PipelineCase, ...] = (
    PipelineCase(
        name="calib-narration",
        why="narration style (lowish, slow) + reference + caption",
        text="山の上から見える景色は、思っていたよりもずっと広がっていました。",
        caption="落ち着いた中年男性の声。ドキュメンタリーのナレーションのように、ゆっくりと低めのトーンで語っている。",
        reference=ReferenceSpec(frames=96, seed=401),
        seed=2401,
    ),
    PipelineCase(
        name="calib-dialogue",
        why="conversational style (highish, fast, question form) + reference + caption",
        text="えっ、それ本当に大丈夫なの。もう一度だけ確かめてみようよ。",
        caption="十代後半の少女の声。友人に驚いて問い返すように、早口でやや高いトーンで話している。",
        reference=ReferenceSpec(frames=108, seed=402),
        seed=2402,
    ),
    PipelineCase(
        name="calib-announce",
        why="public announcement style (little intonation, steady tempo) + reference + caption",
        text="まもなく三番線に、急行列車がまいります。白線の内側までお下がりください。",
        caption="落ち着いた女性の声。駅の構内放送のように、抑揚を抑えた一定のテンポで読み上げている。",
        reference=ReferenceSpec(frames=120, seed=403),
        seed=2403,
    ),
    PipelineCase(
        name="calib-storytelling",
        why="storytelling style (long pauses, relaxed) + reference + caption",
        text="むかしむかし、あるところに、たいそう欲張りなおじいさんが住んでいました。",
        caption="年配の男性の声。昔話を語り聞かせるように、間を長めに取ってゆったりと話している。",
        reference=ReferenceSpec(frames=84, seed=404),
        seed=2404,
    ),
    PipelineCase(
        name="calib-whisper",
        why="whisper style (lowest volume, breathy) + reference + caption",
        text="そっと近づいて、誰にも聞こえないように耳元でささやいた。",
        caption="二十代の女性の声。息を多く含んだ小さなささやき声で、秘密を打ち明けるように話している。",
        reference=ReferenceSpec(frames=88, seed=405),
        seed=2405,
    ),
    PipelineCase(
        name="calib-emotional",
        why="emotional line (highest volume, strong intonation) + reference + caption",
        text="どうしてそんな大事なことを、今まで黙っていたんだ。",
        caption="三十代の男性の声。怒りと戸惑いが混じった強い感情を込めて、声を張り上げて話している。",
        reference=ReferenceSpec(frames=92, seed=406),
        seed=2406,
    ),
    PipelineCase(
        name="calib-news",
        why="news reading (clear, flat, many Sino-Japanese words) + reference + caption",
        text="政府は昨日、来年度の予算案を閣議で正式に決定したと発表しました。",
        caption="四十代の男性の声。ニュース番組の原稿を読むように、明瞭で平板な調子を保って読み上げている。",
        reference=ReferenceSpec(frames=100, seed=407),
        seed=2407,
    ),
    PipelineCase(
        name="calib-sports",
        why="sports commentary (fast × excited, drawn-out long vowels) + reference + caption",
        text="残り十秒、ここでシュート、決まったあああ、逆転です。",
        caption="三十代の男性の声。スポーツ中継の実況のように、興奮して声を高く張り上げ、一気にまくし立てている。",
        reference=ReferenceSpec(frames=104, seed=408),
        seed=2408,
    ),
    PipelineCase(
        name="calib-kids",
        why="reading to children (exaggerated intonation, onomatopoeia) + reference + caption",
        text="うさぎさんはぴょんぴょん跳ねて、お花畑まで走っていきました。",
        caption="二十代の女性の声。幼い子ども向けの絵本を読み聞かせるように、抑揚を大きく取って優しく話している。",
        reference=ReferenceSpec(frames=112, seed=409),
        seed=2409,
    ),
    PipelineCase(
        name="calib-brisk",
        why="brisk business talk (least emotion, fastest tempo) + reference + caption",
        text="資料は三部ずつ用意して、会議室の机に並べておいてください。",
        caption="三十代の女性の声。職場で事務連絡を伝えるように、感情を抑えた早口で淡々と話している。",
        reference=ReferenceSpec(frames=116, seed=410),
        seed=2410,
    ),
    PipelineCase(
        name="calib-poem",
        why="poetry recital (longest pauses, word by word) + reference + caption",
        text="白い雲は流れ、風はやわらかく、季節はまた巡ってゆく。",
        caption="五十代の女性の声。詩を朗読するように、一語ずつ間を置いて響かせながら読み上げている。",
        reference=ReferenceSpec(frames=120, seed=411),
        seed=2411,
    ),
    PipelineCase(
        name="calib-monologue",
        why="wistful monologue (low, subdued voice, wavering) + reference + caption",
        text="あの頃はまだ、こんな日が来るなんて思ってもみなかったよ。",
        caption="六十代の男性の声。昔を懐かしむように、しみじみと声を落として独りごちている。",
        reference=ReferenceSpec(frames=96, seed=412),
        seed=2412,
    ),
)


def evaluation_cases() -> tuple[PipelineCase, ...]:
    """All the evaluation inputs that **must be kept disjoint** from calibration (read directly
    from the single source of truth).

    Irodori's evaluation inputs are confined to `irodori.pipeline_ref.PIPELINE_CASES` — the
    full-loop golden (`meta.json` / `case.*.safetensors`), the LSD / SNR of
    `irodori.measure_quant` and the Deno-side E2E all follow this golden, so no copy is needed
    (anima extracts them from the actual TS files because its evaluation inputs also have a
    separate source of truth on the Deno side).
    """
    return PIPELINE_CASES


def _bodies(case: PipelineCase) -> tuple[str, ...]:
    """The text bodies of one case (text / caption) — empty strings are left out of the
    disjointness check."""
    return tuple(body for body in (case.text, case.caption) if body)


def _seeds(case: PipelineCase) -> tuple[int, ...]:
    """The random seeds of one case (initial noise and reference latent).

    The reference latent's seed is checked too because changing only `seed` while sharing
    `reference.seed` would run calibration and evaluation under **the same speaker condition**
    (the activations of the speaker segment would be shared).
    """
    return (case.seed, *((case.reference.seed,) if case.reference is not None else ()))


def assert_calib_disjoint(cases: Sequence[PipelineCase], evaluated: Sequence[PipelineCase]) -> None:
    """Check that the calibration cases overlap the evaluation cases in **none of text / caption /
    seed**.

    Text bodies are checked **down to substrings** (calibrating on only the first half of an
    evaluation sentence is leakage too — the same shape as anima's `assert_calib_disjoint`).

    MUST: fail loudly. An overlap turns LSD / SNR / listening into a measure of "could it reproduce
    the audio it saw during calibration", measuring leakage instead of calibration quality.
    """
    evaluated_bodies = [body for case in evaluated for body in _bodies(case)]
    hits = sorted(
        {
            body
            for case in cases
            for body in _bodies(case)
            for other in evaluated_bodies
            if body in other or other in body
        }
    )
    if hits:
        raise AssertionError(
            f"calibration case bodies overlap the evaluation inputs: {hits[:3]}"
            f" ({len(evaluated)} evaluation cases)"
        )
    evaluated_seeds = {seed for case in evaluated for seed in _seeds(case)}
    shared = sorted({seed for case in cases for seed in _seeds(case)} & evaluated_seeds)
    if shared:
        raise AssertionError(
            f"calibration case seeds overlap the evaluation inputs: {shared}"
            " (the initial noise or the reference latent follows the same trajectory)"
        )


def calibration_cases() -> tuple[PipelineCase, ...]:
    """Return all the calibration cases, with the disjointness check (the only entry point of the
    calibration path).

    MUST: go through here instead of reading the constant directly — a path that skips the check
    would let disjointness silently break while the constant alone stays green.
    """
    if not CALIB_CASES:
        raise AssertionError("no calibration cases (calibrated i4 cannot work with zero inputs)")
    assert_calib_disjoint(CALIB_CASES, evaluation_cases())
    return CALIB_CASES
