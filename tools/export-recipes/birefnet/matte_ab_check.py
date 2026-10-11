r"""A/B の道具（`tools/birefnet-ab/`）の後処理を、golden の生成器と同じ上流の関数で検算する
（裁定 4・ADR 0125 — 道具の dry-run の検証）。

    cd tools/export-recipes
    uv run --group birefnet python -m birefnet.matte_ab_check <道具の --out>/work

道具が画像ごとに書いた logit（グラフ出力そのもの）と手順ごとの 8bit α（`work/manifest.json` が
並べる）を読み、golden（`birefnet.matte_fixture`）と同じ契約で突き合わせる:

- `u8-pil-bicubic`: `matte_fixture.hr_alpha`（f64 の σ → 四捨五入 → Pillow の BICUBIC）と
  **0 不一致**。255σ が `k + 0.5` から `TIE_MARGIN` 未満の標本（同点 — 生成器が fixture の入力で
  拒むのと同じ境目）は丸めの向きが exp の実装に預けられるので、その標本が BICUBIC の台で届く
  出力画素だけを外し、残りの画素を 0 不一致で比べる（外した画素の数は要約の
  `excludedTiePixels`）。
- `float-bilinear`: `matte_fixture.lucida_scaled`（torch の f32 の σ → TF.resize → × 255）の
  四捨五入と一致するか、golden の `255α` が `k + 0.5` から `LUCIDA_EPSILON` 以内のときだけ 1 段
  ずれてよい。

logit の形は manifest の `graph` の寸法（`[1, 1, height, width]`）と一致しなければ落とす。
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


#: Pillow の BICUBIC の台（`Resample.c` の `bicubic_filter` の support）。
BICUBIC_SUPPORT = 2.0


def bicubic_reach(in_size: int, out_size: int) -> np.ndarray:
    """1 軸の `[出力, 入力]` の真偽表 — 出力の添字の BICUBIC の台が入力の添字に届くか。

    Pillow の `precompute_coeffs` の `xmin` / `xmax`（`(int)` の切り捨ての後に `[0, inSize]` へ
    収める — `packages/models/src/image/pillow-resample.ts` の `fixedTaps` と同じ式）。寸法が
    同じ軸は Pillow がパスを飛ばすので、自分の添字にだけ届く。
    """
    if in_size == out_size:
        return np.eye(in_size, dtype=bool)
    scale = in_size / out_size
    support = BICUBIC_SUPPORT * max(scale, 1.0)
    reach = np.zeros((out_size, in_size), dtype=bool)
    for index in range(out_size):
        center = (index + 0.5) * scale
        start = max(0, int(center - support + 0.5))
        stop = min(in_size, int(center + support + 0.5))
        reach[index, start:stop] = True
    return reach


def tie_samples(logits: np.ndarray) -> np.ndarray:
    """同点の標本（255σ が `k + 0.5` から `TIE_MARGIN` 未満）。

    境目は `matte_fixture.assert_no_ties` と同じ。
    """
    scaled = matte_fixture.sigmoid_f64(logits) * 255
    return np.abs(scaled - (np.floor(scaled) + 0.5)) < matte_fixture.TIE_MARGIN


def tie_reach(ties: np.ndarray, out_width: int, out_height: int) -> np.ndarray:
    """同点の標本のどれかが BICUBIC の台で届く出力画素（`[out_height, out_width]` の真偽）。

    分離型なので、出力 `(Y, X)` が標本 `(r, c)` に届くのは縦の台が `r` に・横の台が `c` に届くとき。
    """
    in_height, in_width = ties.shape
    rows, columns = np.nonzero(ties)
    vertical = bicubic_reach(in_height, out_height)[:, rows].astype(np.int64)
    horizontal = bicubic_reach(in_width, out_width)[:, columns].astype(np.int64)
    return (vertical @ horizontal.T) > 0


def check_upstream(logits: np.ndarray, alpha: np.ndarray) -> dict[str, Any]:
    """`u8-pil-bicubic` の α を HR の上流の手順と突き合わせる。

    契約は同点の標本が届く画素（{@link tie_reach}）を除いて 0 不一致。
    """
    height, width = alpha.shape
    excluded = tie_reach(tie_samples(logits), width, height)
    # 外す画素の値は同点の標本の丸めに依るが、残りの画素は依らないので、どちらへ丸めても同じ。
    expected = matte_fixture.hr_alpha(logits, width, height, allow_ties=True)
    mismatches = int(np.count_nonzero((expected != alpha) & ~excluded))
    return {
        "mismatches": mismatches,
        "excludedTiePixels": int(np.count_nonzero(excluded)),
        "ok": mismatches == 0,
    }


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
    graph = manifest["graph"]
    # MUST: logit の形を graph の寸法と突き合わせる — 取り違えた logit（別のモデル・転置）でも
    # 両手順の α は寸法どおりに作れてしまい、検算は「上流と食い違う」としか言えなくなる。
    logits_shape = (1, 1, graph["height"], graph["width"])
    report: dict[str, Any] = {}
    for entry in manifest["entries"]:
        tensors = load_file(work / entry["logits"])
        if len(tensors) != 1:
            raise ValueError(
                f"{entry['logits']}: テンソルが {len(tensors)} 本（グラフ出力 1 本のはず）"
            )
        (logits,) = tensors.values()
        if logits.shape != logits_shape:
            raise ValueError(
                f"{entry['logits']}: logit の形 {list(logits.shape)} が manifest の graph"
                f" {list(logits_shape)} と違う"
            )
        alphas = load_file(work / entry["alpha"])
        for name, alpha in alphas.items():
            if alpha.dtype != np.uint8 or alpha.shape != (entry["height"], entry["width"]):
                raise ValueError(
                    f"{entry['alpha']}: {name} が U8 [{entry['height']}, {entry['width']}] でない"
                )
        report[entry["name"]] = check_entry(logits[0, 0], alphas)
    return report


def violations(report: Mapping[str, Any]) -> list[str]:
    """契約から外れた（画像・手順）の一覧。"""
    return [
        f"{name}/{procedure}"
        for name, entry in report.items()
        for procedure, result in entry.items()
        if not result["ok"]
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
