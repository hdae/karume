# device 消失後の待ちは解決するか（時点スナップショット）

NOTE: この文書は 2026-08-16 時点の実測記録であり、仕様の一般化ではない。
[2026-08-01-m0-review.md](2026-08-01-m0-review.md) の記録を上書きせず補足する。

環境: Deno + wgpu / 実 GPU（vendor 4318 = NVIDIA, device 8712）。消失の起こし方は
`device.destroy()` のみ（実 TDR / ドライバリセットは未検証）。

## 実測 1: destroy 後に発行した待ち

```
A: pop#1 (oom) right after destroy        -> resolved: null
A: pop#2 (validation) right after destroy -> resolved: null
B: device.lost                            -> resolved: destroyed: device was lost
B: pop on empty stack after lost          -> resolved: null
B: push+pop after lost                    -> resolved: null
C: mapAsync after destroy                 -> rejected: OperationError: validation error occurred
```

## 実測 2: 消失前に発行して in-flight だった待ち

```
in-flight popErrorScope       -> resolved: null
in-flight onSubmittedWorkDone -> resolved: null
in-flight mapAsync            -> resolved: null
```

## 判定

- この環境では `popErrorScope` は **null で resolve**（空スタックでも同じ）し、`mapAsync` は
  `OperationError` で reject する。5 秒以内に全て決着し、ハングは 1 件も観測されなかった。
  WebGPU 仕様（device が lost なら `popErrorScope` は null で resolve）とも 2026-08-01 の記録
  とも一致する。
- したがって `src/gpu/device.ts` にあった「device 消失後 `popErrorScope` / `mapAsync` は
  解決しない」という断言は、少なくともこの環境では**誤り**（同日の修正波でコメントを訂正済み）。
- 一方で `raceCanaryDeviceLost` / `raceDeviceLost` は撤去していない。**未検証で残る範囲**:
  ① `device.destroy()` 以外の消失（実 TDR / ドライバリセット / OOM kill）② ブラウザ
  （Dawn / Tint）。解決を返さない実装に当たったときのハング回避として、レースは保険のまま
  維持する。
- 併せて、外部レビュー（Codex CX1-1）が挙げた「未保護の `popErrorScope` 6 箇所が device 消失で
  デッドロックする」説も、この実測では本環境で発火しない。ブラウザ向けの `raceDeviceLost`
  包み込みは harmless hardening として保留（レビュー ROADMAP）。
