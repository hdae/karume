"""Irodori の校正コーパス（GPTQ が見る活性を作る full-loop ケース 4 件）。

`karume.quant_calib` の校正付き丸めは「その層に実際に流れる活性」から丸め先を選び直す方式
なので、コーパスの性格がそのまま丸めの偏りになる。DiT の活性は**テキスト（自己 attention の
長さと中身）と caption / speaker の条件と t** で動くので、選定方針は 3 点:

- **評価入力と分離する**（anima の 2026-08-21 ユーザー裁定と同根）— 校正で見たケースを
  そのまま評価に使うと、LSD / SNR / 聴感が「校正で見た音をそのまま出せたか」を測る数になり、
  校正の質ではなく**漏れ**を測ることになる。Irodori の評価入力の正本は
  {@link irodori.pipeline_ref.PIPELINE_CASES} 1 箇所（full-loop golden も
  `irodori.measure_quant` も Deno 側 E2E もこの golden を辿る）なので、写しを置かずに
  そこから直に引いて {@link assert_calib_disjoint} が突き合わせる。分離は **text /
  caption / seed の 3 軸すべて**で見る — どれか 1 つでも共有すると、同じ軌道の活性が
  校正へ入る。
- **Irodori の領域に寄せる** — 日本語 TTS なので校正も日本語の発話文で採り、**話体
  （ナレーション / 会話 / 案内放送 / 語り聞かせ）・話者像（男女 × 年代）・テンポ**を散らす。
  1 つの型へ寄せると、活性の偏りがその型に張り付く。
- **条件は 3 本とも有効にする**（参照 latent あり + caption 非空）— 配布の既定経路であり、
  DiT の交差 attention が 3 区間とも動く唯一の regime。参照なし / caption 空の regime を
  校正に入れないのは意図した線引きで、そちらは**評価側**（`no-ref` ケース）が担当する。

NOTE（本数と step 数の位置づけ）: 校正の質は**ケース数**と**捕捉 step 数**で上がる（前者は
ケース間の活性の散り・後者は 1 ケースあたりの t の網羅 = `H = Σ XᵀX` の標本数）。どちらも
export の CPU 時間に線形で効く。step 数は {@link irodori.calib.CALIB_STEPS} が既定
（= 参照ループ全長）で `--calib-steps` で下げられるが、**ケース数は固定 4 件** — 増やすなら
ここへ本文を足す（`--calib-cases` のようなノブは置かない。使う本数が動くと
`calib_provenance.json` の `cases` が実行ごとに変わり、配布資産の丸め条件が読めなくなる）。
"""

from __future__ import annotations

from collections.abc import Sequence

from .pipeline_ref import PIPELINE_CASES, PipelineCase, ReferenceSpec

#: 校正入力のケース（この順で全件使う）。
#:
#: `frames` は参照 latent の patch 前フレーム数で、`speaker_patch_size` で割り切れる値
#: （{@link irodori.pipeline_ref.ReferenceSpec} の MUST）。評価ケースが使っている 124 と
#: 同じ 4 の倍数帯に収めてあるのは、参照長そのものを未検証の領域へ広げないため。
CALIB_CASES: tuple[PipelineCase, ...] = (
    PipelineCase(
        name="calib-narration",
        why="ナレーション体（低め・ゆっくり）+ 参照あり + caption あり",
        text="山の上から見える景色は、思っていたよりもずっと広がっていました。",
        caption="落ち着いた中年男性の声。ドキュメンタリーのナレーションのように、ゆっくりと低めのトーンで語っている。",
        reference=ReferenceSpec(frames=96, seed=401),
        seed=2401,
    ),
    PipelineCase(
        name="calib-dialogue",
        why="会話体（高め・早口・疑問形）+ 参照あり + caption あり",
        text="えっ、それ本当に大丈夫なの。もう一度だけ確かめてみようよ。",
        caption="十代後半の少女の声。友人に驚いて問い返すように、早口でやや高いトーンで話している。",
        reference=ReferenceSpec(frames=108, seed=402),
        seed=2402,
    ),
    PipelineCase(
        name="calib-announce",
        why="案内放送体（抑揚が小さく一定テンポ）+ 参照あり + caption あり",
        text="まもなく三番線に、急行列車がまいります。白線の内側までお下がりください。",
        caption="落ち着いた女性の声。駅の構内放送のように、抑揚を抑えた一定のテンポで読み上げている。",
        reference=ReferenceSpec(frames=120, seed=403),
        seed=2403,
    ),
    PipelineCase(
        name="calib-storytelling",
        why="語り聞かせ体（間が長くゆったり）+ 参照あり + caption あり",
        text="むかしむかし、あるところに、たいそう欲張りなおじいさんが住んでいました。",
        caption="年配の男性の声。昔話を語り聞かせるように、間を長めに取ってゆったりと話している。",
        reference=ReferenceSpec(frames=84, seed=404),
        seed=2404,
    ),
)


def evaluation_cases() -> tuple[PipelineCase, ...]:
    """校正から**分離すべき**評価入力の全量（正本 1 箇所から直に引く）。

    Irodori の評価入力は `irodori.pipeline_ref.PIPELINE_CASES` に閉じている — full-loop
    golden（`meta.json` / `case.*.safetensors`）も `irodori.measure_quant` の LSD / SNR も
    Deno 側 E2E もこの golden を辿るので、写しを持つ必要が無い（anima が TS の実ファイルから
    抽出しているのは、あちらの評価入力が Deno 側にも別の正本を持つため）。
    """
    return PIPELINE_CASES


def _bodies(case: PipelineCase) -> tuple[str, ...]:
    """1 ケースが持つ本文（text / caption）— 空文字は分離検査の対象から外す。"""
    return tuple(body for body in (case.text, case.caption) if body)


def _seeds(case: PipelineCase) -> tuple[int, ...]:
    """1 ケースが持つ乱数種（初期ノイズと参照 latent）。

    参照 latent の種まで見るのは、`seed` だけ変えて `reference.seed` を共有すると
    **同じ話者条件**で校正と評価が回ることになるため（speaker 区間の活性が共有される）。
    """
    return (case.seed, *((case.reference.seed,) if case.reference is not None else ()))


def assert_calib_disjoint(cases: Sequence[PipelineCase], evaluated: Sequence[PipelineCase]) -> None:
    """校正ケースが評価ケースと **text / caption / seed のどれでも**重ならないことを見る。

    本文は**部分一致まで**見る（評価文の前半だけを校正に使うのも漏れ — anima の
    `assert_calib_disjoint` と同じ形）。

    MUST: fail loudly。重なると LSD / SNR / 聴感が「校正で見た音をそのまま出せたか」を測る数に
    なり、校正の質ではなく漏れを測る。
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
            f"校正ケースの本文が評価入力と重なっている: {hits[:3]}"
            f"（評価ケース {len(evaluated)} 件）"
        )
    evaluated_seeds = {seed for case in evaluated for seed in _seeds(case)}
    shared = sorted({seed for case in cases for seed in _seeds(case)} & evaluated_seeds)
    if shared:
        raise AssertionError(
            f"校正ケースの乱数種が評価入力と重なっている: {shared}"
            "（初期ノイズか参照 latent が同じ軌道になる）"
        )


def calibration_cases() -> tuple[PipelineCase, ...]:
    """校正ケースの全量を分離検査つきで返す（校正経路の唯一の入口）。

    MUST: 定数を直に読まずここを通す — 検査を通らない経路があると、定数だけ緑のまま
    分離が黙って崩れる。
    """
    if not CALIB_CASES:
        raise AssertionError("校正ケースが 1 件も無い（校正付き i4 は入力ゼロでは成立しない）")
    assert_calib_disjoint(CALIB_CASES, evaluation_cases())
    return CALIB_CASES
