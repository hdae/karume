"""i4 系列の校正条件（`calib_provenance.json`）が配布に足りるかの判定（ADR 0065 決定 2）。

校正の方式も予算も**格納形を 1 バイトも変えない**（格子は RTN i4 g32 のまま — 変わるのは
丸め値と scale 台帳だけ）ので、`verify_dist` の構造検査もヘッダ dtype 検査も素通りする。
判別できる事実は「書き出した側が残した記録」1 つきりで、その記録を読む席は
`irodori.distribution`（配布の組み立て）・`irodori.pipeline_ref`（golden の焼き直し）・
`anima.distribution`（同じく組み立て）の 3 つある。

MUST: **判定の正本はここ 1 本**（読み手はこれを呼んで、家族固有の後始末だけを足す）。
3 箇所に写すと、片方の門だけを強めた日にもう片方が黙って弱くなる — 実際、方式 1 欄だけを
見ていた時期に「smoke 予算（`--calib-steps 1`）で焼いた資産」が 3 つとも素通りした
（2026-08-24 レビュー Ca17）。`_shared.decode_series` と同じ理由。

MUST: **欄が無い記録は受理する**。欄は後から足せる（anima の `guidance` は 2026-08-23 に
足した）ので、存在を要求すると 1 行の追加が既存系列へ「丸め時間ぶんの再 export」を課す
ことになる。在る欄だけを見て、**予算は下限で・モデル条件は一致で**判定する。

依存方向は recipe → core の一方向だけ（`_shared` は family を import しない）— 下限も条件も
呼び手が正本から引いて渡す。
"""

from __future__ import annotations

from collections.abc import Mapping


def _number(value: object) -> float | None:
    """記録の欄を数として読む（読めなければ `None`）。

    `bool` を弾くのは、`True` が 1 として下限を通り抜けるのを避けるため（JSON の `true` は
    記録の書き手が書く形ではないので、現れたら記録そのものが壊れている）。
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def calib_complaint(
    record: object,
    *,
    method: str,
    at_least: Mapping[str, float] | None = None,
    exactly: Mapping[str, float] | None = None,
) -> str | None:
    """記録が配布の条件を満たさない理由を 1 文で返す（満たしていれば `None`）。

    `at_least` は**予算**（削ると質だけが落ちる欄 — 捕捉 step 数・校正プロンプト本数・
    コーパスの件数）、`exactly` は**どの条件で焼いたか**（モデルごとに決まる欄 — step 数・
    CFG）。予算は下限、条件は一致で見るのは、後者では「大きい方が良い」が成り立たないから
    （turbo の 8 step で焼いた資産を素モデルの席に載せるのも、その逆も等しく誤り）。

    記録が辞書でない場合は `method` が読めない扱いにする（方式の不一致として落ちる）—
    壊れた記録を「欄が無いだけ」として受理する側には倒さない。
    """
    fields = record if isinstance(record, Mapping) else {}
    if fields.get("method") != method:
        return (
            f"配布して良い丸め方式で作られていない（{fields.get('method')!r}、配布可は {method!r}）"
        )
    for name, floor in (at_least or {}).items():
        if name not in fields:
            continue
        value = _number(fields[name])
        if value is None:
            return f"校正予算 {name!r} が数でない（{fields[name]!r}）"
        if value < floor:
            return f"校正予算 {name!r} が配布の下限を下回る（{value:g} < {floor:g}）"
    for name, want in (exactly or {}).items():
        if name not in fields:
            continue
        value = _number(fields[name])
        if value is None:
            return f"校正条件 {name!r} が数でない（{fields[name]!r}）"
        if value != want:
            return f"校正条件 {name!r} がこのモデルの既定と違う（{value:g}、既定は {want:g}）"
    return None
