r"""A/B の道具（`tools/birefnet-ab/`）の後処理を、golden の生成器と同じ上流の関数で検算する
（裁定 4・ADR 0125 — 道具の dry-run の検証）。

    cd tools/export-recipes
    uv run --group birefnet python -m birefnet.matte_ab_check <道具の --out>/work

道具が画像ごとに書いた logit（グラフ出力そのもの）と手順ごとの 8bit α（`work/manifest.json` が
並べる）を読み、golden（`birefnet.matte_fixture`）と同じ契約で突き合わせる:

- `u8-pil-bicubic`: `matte_fixture.hr_alpha`（f64 の σ → 四捨五入 → Pillow の BICUBIC）と
  **0 不一致**。255σ が `k + 0.5` の `TIE_MARGIN` 以内の標本がある画像は、丸めの向きが exp の
  実装に預けられるので、この検算から外して名指しする（生成器が fixture の入力で拒むのと同じ境目）。
- `float-bilinear`: `matte_fixture.lucida_scaled`（torch の f32 の σ → TF.resize → × 255）の
  四捨五入と一致するか、golden の `255α` が `k + 0.5` から `LUCIDA_EPSILON` 以内のときだけ 1 段
  ずれてよい。

外れがあれば終了コード 1（要約は JSON で標準出力へ）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from birefnet import matte_fixture

#: 道具が書く手順の名前（`packages/models/src/birefnet/matte.ts` の `MATTE_PROCEDURES`）。
UPSTREAM = "u8-pil-bicubic"
CURRENT = "float-bilinear"


def check_upstream(logits: np.ndarray, alpha: np.ndarray) -> dict[str, Any]:
    """`u8-pil-bicubic` の α を HR の上流の手順と突き合わせる（0 不一致が契約）。"""
    height, width = alpha.shape
    try:
        expected = matte_fixture.hr_alpha(logits, width, height)
    except ValueError as tie:
        return {"skipped": str(tie)}
    mismatches = int(np.count_nonzero(expected != alpha))
    return {"mismatches": mismatches, "ok": mismatches == 0}


def check_current(logits: np.ndarray, alpha: np.ndarray) -> dict[str, Any]:
    """`float-bilinear` の α を Lucida の上流の手順と突き合わせる（ε の内の 1 段だけを許す）。"""
    height, width = alpha.shape
    scaled = matte_fixture.lucida_scaled(logits, width, height).astype(np.float64)
    expected = np.floor(scaled + 0.5)
    actual = alpha.astype(np.float64)
    differs = actual != expected
    boundary = np.minimum(actual, expected) + 0.5
    allowed = (
        differs
        & (np.abs(actual - expected) == 1)
        & (np.abs(scaled - boundary) <= matte_fixture.LUCIDA_EPSILON)
    )
    outside = int(np.count_nonzero(differs & ~allowed))
    return {
        "withinEpsilon": int(np.count_nonzero(allowed)),
        "outsideEpsilon": outside,
        "ok": outside == 0,
    }


def check_entry(logits: np.ndarray, alphas: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """画像 1 枚ぶん（logit `[h, w]` と手順ごとの α `[H, W]`）。"""
    missing = {UPSTREAM, CURRENT} - set(alphas)
    if missing:
        raise ValueError(f"手順の α が無い: {sorted(missing)}")
    return {
        UPSTREAM: check_upstream(logits, alphas[UPSTREAM]),
        CURRENT: check_current(logits, alphas[CURRENT]),
    }


def check_work(work: Path) -> dict[str, Any]:
    """道具の `work/` を検算する（manifest の全画像）。"""
    from safetensors.numpy import load_file

    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    report: dict[str, Any] = {}
    for entry in manifest["entries"]:
        tensors = load_file(work / entry["logits"])
        if len(tensors) != 1:
            raise ValueError(
                f"{entry['logits']}: テンソルが {len(tensors)} 本（グラフ出力 1 本のはず）"
            )
        (logits,) = tensors.values()
        alphas = load_file(work / entry["alpha"])
        for name, alpha in alphas.items():
            if alpha.dtype != np.uint8 or alpha.shape != (entry["height"], entry["width"]):
                raise ValueError(
                    f"{entry['alpha']}: {name} が U8 [{entry['height']}, {entry['width']}] でない"
                )
        report[entry["name"]] = check_entry(logits[0, 0], alphas)
    return report


def violations(report: Mapping[str, Any]) -> list[str]:
    """契約から外れた（画像・手順）の一覧。外した検算（`skipped`）は外れに数えない。"""
    return [
        f"{name}/{procedure}"
        for name, entry in report.items()
        for procedure, result in entry.items()
        if "skipped" not in result and not result["ok"]
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("work", type=Path, help="道具の出力の work/（manifest.json のある所）")
    args = parser.parse_args(argv)
    report = check_work(args.work)
    failed = violations(report)
    print(json.dumps({"report": report, "violations": failed}, indent=1, ensure_ascii=False))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
