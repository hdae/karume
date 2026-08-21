"""Anima の校正コーパス（GPTQ が見る活性を作る danbooru 系タグ 24 本・既定は先頭 4 本）。

`karume.quant_calib` の校正付き丸めは「その層に実際に流れる活性」から丸め先を選び直す方式
なので、コーパスの性格がそのまま丸めの偏りになる。DiT の活性は**プロンプト（= 交差 attention
に入る条件）と解像度と sigma** で動くので、選定方針は 3 点:

- **評価入力と分離する**（2026-08-21 ユーザー裁定）— 校正で見たプロンプトをそのまま評価に
  使うと、PSNR / 視認が「校正で見た絵をそのまま出せたか」を測る数になり、校正の質ではなく
  漏れを測ることになる。評価入力の**正本は 3 箇所とも別ファイル**（`anima.pipeline_ref` の
  固定プロンプトと、Deno 側の E2E / eval-images）なので、写しをここへ置かず
  {@link evaluation_prompts} が実ファイルから抽出して {@link assert_calib_disjoint} で
  突き合わせる（写すと片方だけ古くなる）。
- **Anima の領域に寄せる** — 模型はアニメ画像特化（danbooru 系タグ）なので、校正も同じタグ
  語彙で採る。**被写体（人数 0〜4・動物）・構図（寄り / 引き / 俯瞰 / 煽り / 後ろ姿）・
  画風（chibi / 線画 / 水彩 / フィルム粒子）を散らす** — 1 つの型へ寄せると、活性の偏りが
  その型に張り付く。
- **先頭から使う**（`--calib-prompts N` は**先頭 N 本**）ので、並びは型ごとに固めず**混ぜる** —
  縮小実行（既定の 12 本もその一種）でも被写体・構図・画風の混合が保たれる。

NOTE（品質の上振れ軸）: 校正の質は**本数**と**解像度**で上がる（本数はプロンプト間の活性の
散り・解像度は 1 プロンプトあたりのトークン数 = `H = Σ XᵀX` の標本数）。どちらも export の
CPU 時間に線形で効く。解像度は品質裁定の条件ごと固定（{@link anima.calib.CALIB_RESOLUTION}）
なので、動かせるのは本数だけ。上限 24 本まではここのコーパスで足り、それ以上へ広げるときは
本文を足す。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from _shared.paths import REPO_ROOT

from .pipeline_ref import NEGATIVE_PROMPT, PROMPT

#: `--calib-prompts` の既定（{@link CALIB_PROMPTS} の**先頭からこの本数**）。
#:
#: **根拠のある下限に置く**: 品質裁定（PSNR 22.73dB・視認で f32 とほぼ同一）が採られた波 J-2
#: の校正条件は 1 本 × 10 step = 10 バッチなので、4 本 × 8 step = 32 バッチはそれを下回らない。
#: 一方で 12 本 / 24 本へ増やしたときの**追加利得は未測定**で、費用（{@link
#: anima.calib.CALIB_SECONDS_PER_BATCH} の外挿で 4 本 ≒ 50 分・12 本 ≒ 2.5 時間の CPU 時間と、
#: `current` と次段リストが同時に生きるぶんのメモリ）は本数に線形で効く。上げるときは測ってから
#: （`--calib-prompts N` で 24 本まで振れる）。
DEFAULT_CALIB_PROMPTS = 4

#: 校正入力のプロンプト（この順で先頭から使う）。
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

#: 評価入力の正本（リポジトリ相対）。Deno 側なので import できず、**実ファイルから抽出**する
#: （写しを持つと片方だけ古くなり、分離検査が黙って緩む）。抽出が空になったら fail loudly。
E2E_SOURCE = Path("packages/models/tests/e2e_anima_test.ts")
EVAL_IMAGES_SOURCE = Path("examples/anima/eval-images.ts")

#: TS の二重引用符文字列 1 本（エスケープ込み）。
_STRING = r'"((?:[^"\\]|\\.)*)"'


def _read_source(source: Path) -> str:
    path = REPO_ROOT / source
    if not path.is_file():
        raise AssertionError(
            f"評価入力の正本 {source} が無い（校正コーパスとの分離検査の基準が引けない）"
        )
    return path.read_text(encoding="utf-8")


def _declaration(text: str, source: Path, name: str) -> str:
    """`const <name> = "…" + "…";` の文字列リテラルを連結して返す。"""
    match = re.search(rf"const {name}\s*=\s*(.*?);", text, re.DOTALL)
    parts = re.findall(_STRING, match.group(1)) if match is not None else []
    if not parts:
        raise AssertionError(
            f"{source} の const {name} から文字列を抽出できなかった"
            "（綴りが変わっている — 分離検査の基準が空になる方が危ないので落とす）"
        )
    return "".join(parts)


def _eval_image_prompts() -> tuple[str, ...]:
    """`examples/anima/eval-images.ts` が実際に渡すプロンプト（4 ケース）。

    合成の式（`` `${QUALITY_PREFIX}, ${subject}, ${QUALITY_SUFFIX}` ``）だけはあちらの
    `prompt()` を写している — 定数と被写体は実ファイルから引くので、写しは 1 行に閉じる。
    """
    text = _read_source(EVAL_IMAGES_SOURCE)
    prefix = _declaration(text, EVAL_IMAGES_SOURCE, "QUALITY_PREFIX")
    suffix = _declaration(text, EVAL_IMAGES_SOURCE, "QUALITY_SUFFIX")
    subjects = re.findall(rf"subject:\s*{_STRING}", text)
    if not subjects:
        raise AssertionError(
            f"{EVAL_IMAGES_SOURCE} から subject を 1 件も抽出できなかった"
            "（ケース表の綴りが変わっている）"
        )
    return tuple(f"{prefix}, {subject}, {suffix}" for subject in subjects)


def evaluation_prompts() -> tuple[str, ...]:
    """校正から**分離すべき**評価入力の全量（3 箇所の正本から抽出）。

    - `anima.pipeline_ref` の固定プロンプト（参照フィクスチャ = 数値パリティの入力）
    - Deno 側 E2E（`packages/models/tests/e2e_anima_test.ts`）の `PROMPT`
    - eval-images（`examples/anima/eval-images.ts`）の 4 ケース
    """
    return (
        PROMPT,
        NEGATIVE_PROMPT,
        _declaration(_read_source(E2E_SOURCE), E2E_SOURCE, "PROMPT"),
        *_eval_image_prompts(),
    )


def assert_calib_disjoint(prompts: Sequence[str], evaluated: Sequence[str]) -> None:
    """校正コーパスが評価入力と**部分一致でも**重ならないことを見る。

    MUST: fail loudly（`sbv2.measure_quant.assert_calib_disjoint` と同じ形）。重なると PSNR /
    視認が「校正で見た絵をそのまま出せたか」を測る数になり、校正の質ではなく漏れを測る。
    """
    hits = sorted(
        {text for text in prompts for body in evaluated if body and (body in text or text in body)}
    )
    if hits:
        raise AssertionError(
            f"校正コーパスが評価入力と重なっている: {hits[:3]}（評価入力 {len(evaluated)} 本）"
        )


def calibration_prompts(count: int) -> tuple[str, ...]:
    """先頭 `count` 本を分離検査つきで返す（`--calib-prompts` の実体）。"""
    if not 1 <= count <= len(CALIB_PROMPTS):
        raise ValueError(
            f"--calib-prompts {count} は 1〜{len(CALIB_PROMPTS)} の範囲外"
            f"（コーパスは {len(CALIB_PROMPTS)} 本 — 増やすなら calib_prompts.py へ本文を足す）"
        )
    prompts = CALIB_PROMPTS[:count]
    assert_calib_disjoint(prompts, evaluation_prompts())
    return prompts
