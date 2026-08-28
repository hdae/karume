# anima 素版 i4 の adaLN-i8 変種 — 視認裁定（不採用）

> NOTE: 時点スナップショット（2026-08-28 の視認裁定の記録）。採否の台帳は
> [perf-ledger](../perf-ledger.md) Q-9、波の記録は [backlog](../backlog.md)。

## 条件

- 変種: anima-v1.0 `--i4-adaln-i8` — adaLN 168 本（norm1/2/3 × linear_1/2 × 28 block）+
  block 外 5 本（`NON_STAGE_I4_WEIGHTS`）を i4 から i8 へ退避した混成 i4（GPTQ 校正付き・
  4 プロンプト・512² 固定）。ペイロード +71.71MiB（dit i4 比 +6.14%）。provenance
  `gptq-adaln8`（配布方式一致門から名指しで外れる — export 2h28m/8907.5s・CPU）
- 視認: eval dist（`outputs/eval/karume-anima-v1.0-adaln8`・quant `f16+dit4`）vs
  既定席（`f16+dit8-a8-attn8-s16`）vs 旧 i4（2026-08-24 配布スキップ裁定の側・
  `outputs/series-archive/2026-08-23-anima-base-i4/`）。**seed 42–45 × プロンプト 3 種**
  （既定タグ・人物動勢・無人夜景）・20 step / CFG 4.0 / 1024²。視認物 =
  `outputs/demo/sweep-adaln8/` + 単発 2 枚

## 結果（2026-08-28 ユーザー裁定）

- seed 42 単独の中間評は「旧 i4 の欠点を脱して既定席へ確実に寄る（採用してもいいレベル）」
  だったが、**スイープで覆った**: 「明らかに既定席（control）の方が描き込みが強い」—
  **不採用**。
- 速度の付記: 変種の生成は ~105s vs 既定席 ~52s（1024²/20step — i4 実行経路が linear の
  i8a8 を失う既知の性格。[anima-i4-seat-speed](2026-08-21-anima-i4-seat-speed.md) と整合）。

## モデル固有性（一般化しない）

irodori では同型の構成（adaLN 144 本 i8 退避）が聴感裁定「こもり解消・配布可」で**採用済み**
（[gptq-expansion-quality](2026-08-24-gptq-expansion-quality.md)）— 「adaLN 退避」自体の
棄却ではなく、**anima-v1.0 の i4 に対して不足**という裁定。

## 未検証（試していないもの — 棄却と区別する）

- 校正標本の増量（`--calib-prompts` 8–24 — コーパス 24 本は実装済み）
- 素版 + 標本増量による **Hessian rank 仮説**の検証（adaLN の H 標本は timestep 由来で
  1 行/バッチ = 4 本×40 バッチで 160 行 vs 2048 次元 — attn/ff の 1/4096。未実測）
- 除外粒度の変種（norm1/2/3 の個別退避・ff との組合せ — CLI 未実装）
- 校正解像度の引き上げ（512² 固定を外す — CLI が拒否する設計を変える必要あり）
- GPU 校正（`--calib-device cuda`）の等価性 — 本裁定と独立に実験中（校正の回転を上げる
  インフラ検証。等価なら上の未検証群が数十分オーダーになる）

## 教訓

**視認 A/B は seed 4 本以上 + 複数プロンプトで裁定する** — seed 42 単独の中間評
（採用レベル）がスイープ（seed 43–45 + 別プロンプト 2 種）で覆った実例。単発比較は
構図の当たり外れと区別できない。
