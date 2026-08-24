# 波 J-4: GPTQ 適用拡大の品質実測（irodori w4 / anima 素版 i4）

> 性格: **時点スナップショット**（2026-08-23〜24 実測）。波 J-4（GPTQ 適用拡大 — 2026-08-23
> ユーザー裁定「g の最適化より多くのモデルで GPTQ を採用できることを優先」）の品質
> イテレーションと裁定の記録。席の設計判断は ADR
> [0050](../decisions/0050-irodori-quant-series.md) 追記 / 採否は
> [perf-ledger](../perf-ledger.md) Q-3 / Q-5 / Q-6。

## 1. irodori `w4` 席 — 品質 3 ラウンドの経緯

聴感駆動で 3 ラウンド回し、最終形（R3）を出荷した。すべて同一ノイズ・同一参照 latent の
A/B（golden の noise / reference_latent を注入し発話インスタンスを揃える）。

| ラウンド | 構成（DiT）                                                                     | 聴感（ユーザー）                            |
| -------- | ------------------------------------------------------------------------------- | ------------------------------------------- |
| R1       | block 内 312 本 GPTQ i4（校正 4 件）+ block 外 5 本 素 RTN i4                   | 全体的にこもる                              |
| R2       | block 内 312 本 GPTQ i4（**校正 12 件**）+ block 外 5 本 **i8**                 | こもりほぼ解消・読み上げ方が若干変わる      |
| R3       | block 内 **adaLN 以外 168 本** GPTQ i4 + **adaLN 144 本 i8** + block 外 5 本 i8 | （sim A/B で）読み上げ方が明確に改善 → 採用 |

- 格子は全ラウンド gptq-rtn g32。他 7 役は w8 の i8 バイト共有・session = {}
  （`linearCompute` は宣言しない — w4a8 は irodori 未測定・anima は品質で不採用
  〈ADR [0076](../decisions/0076-w4a8-linear-execution.md) 決定 6〉）。
- 校正 12 件は評価 2 件と**部分一致まで分離**（`irodori/calib_cases.py` の
  `assert_calib_disjoint` — 声量・抑揚・テンポの 3 軸を両端まで張る 8 件を追加）。

### こもりの帰属（R1 → R2 の根拠）

2 軸の切り分け sim（計測リグ・repo 無改変）で両方が実因と確定:

- **軸 A = 校正コーパス**: J-2 系の sim は**評価入力そのもので校正しており有利すぎる比較
  だった**（校正リーク）。評価文で校正した sim はほぼ無劣化 = 4bit 格子の表現力は足りて
  おり、問題は校正データの汎化。
- **軸 B = block 外 5 本**（`in_proj` / `out_proj` / `cond_module.{0,2,4}`）: 校正の駆動が
  stage 単位である都合で GPTQ に載らず素の RTN i4 だった — i8 へ戻すとこもりが減る。

### adaLN の量子化感度（R2 → R3 の根拠）

R2 に残った「読み上げ方（韻律）の変化」の帰属 A/B（計測リグ・共通条件完全同一の 2 セル
`r2` / `r2adaln8` — `outputs/demo/j4-irodori-adaln8-r2/`）で、**adaLN 144 本
（`attention_adaln` / `mlp_adaln` = modulation の scale/shift/gate）を i8 へ戻すと読み上げ方が
f32 へ寄る**（ユーザー聴感「明確に改善」）。コストは +13.1 MiB（dit i4 payload の +6.0%）。

NOTE: **他 DiT への一般化は未実測の仮説**（棄却記録・採用記録はモデル固有 — anima で品質
不満が出たときの第一のレバー）。

## 2. sim → 出荷の転移限界（2026-08-24 実測）

出荷後の実バイト聴感（ユーザー）は「**こもりは解消しているが adaLN i8 の読み上げ方の改善は
聴き取れない**」— 機械検証の結果、これはバグではなく**リグ間で発話実現が再抽選される**ため:

- 出荷コンテナは構成として正しい（dtype × 名前の全数分類 = I4 168 / I8 149〈adaLN 144 +
  block 外 5・分類外 0〉/ 順序 MUST も sim と同一）。
- しかし出荷バイト golden の z は sim セルと**セル間距離級**に離れる（relRMS・同一ノイズ）:

| ケース | golden↔r2adaln8 | golden↔r2 | r2↔r2adaln8（錨） |
| ------ | --------------- | --------- | ----------------- |
| full   | 0.439           | 0.500     | 0.488             |
| no-ref | 0.257           | 0.387     | 0.392             |

- 機序: 計測リグと export リグでは GPTQ の校正入力の作り方が異なり丸め解が変わる →
  100 forward の CFG ループが差を増幅（anima の構図分岐と同機序）。**sim A/B が証明するのは
  「同一リグ内で効く」ことまで**で、頑健な性質（こもり解消 = コーパス・block 外の是正）だけが
  リグを跨いで転移した。韻律の実現のような繊細な性質は転移しない。
- 帰結: **adaLN i8（+13.1 MiB）が出荷リグでも読み上げ方を改善するかは未検証に戻った**。
  検証には出荷リグでの載せ外し A/B（adaLN 込み 312 本 GPTQ の再 export + 同一ノイズ聴感）が
  要る — ユーザー裁定（2026-08-24）は「R3 のまま配布可」でクローズ・A/B は実需待ち。

## 3. NF4 の irodori 聴感棄却（2026-08-23）

sim（i8rtn vs i8nf4・同一ノイズ）でユーザー聴感 = **nf4 は rtn より f32 から遠い（こもる側）**
→ irodori では棄却。screening（2026-08-19）の「TTS 系で NF4 一貫改善」は SBV2 の実測 —
**モデル固有で一般化しない**（棄却記録のスコープ原則）。perf-ledger Q-3 に追記。

## 4. 速度・門・配布（irodori）

- 速度（S 固定 5s・温間・実 GPU）: f32 5.15 / f16 4.96 / w8 5.36 / **w8a8 4.35（最速・
  既定）** / **w4 6.0s（最遅 +38%・load 0.24s は最小）** — w4 の存在理由はサイズと帯域
  （anima Q-8 と同機序: i8a8 経路を失い f32 計算に落ちる）。既定 `w8a8` 据え置き。
- latent 門（`e2e_irodori_latent_test.ts` w4 系列・atol 6e-3 = R2 実測 full 1.1522e-3 の
  5.2 倍で導出）: R3 実測 z maxAbs = **full 4.7213e-4 / no-ref 2.0429e-5**（S / forwards
  完全一致・adaLN が i8 厳密になった分 R2 より締まった — tolerance 再導出は不要）。
- golden は出荷バイトから焼く（`pipeline_ref --dtype i4` = 出荷コンテナ読み戻し
  `restore_dit_from_i4_series`）。i4Source = gptq / rtn / g32 / 12 件 × 40 step・
  i4 168 / i8 149 / f32 101・changedByRestore 168。
- HF 公開（2026-08-24・`hdae/karume-irodori-v4-small` コミット `67e9584c`）: 変更は
  `dit/model.i4.safetensors`（234.92 MiB）+ `karume.json` + `README.md` のみ（未変更 25 本は
  コミット除外）。断片化検証 = i4 **29.4 MiB/レンジ**（8 terms）・対照 i8 31.9（目安 10 を
  大きく上回り健全）。**pin は据え置き**（anima i4 と同じ — `w4` を使うには
  `revision: "main"` の明示が要る）。

## 5. anima 素版 3 モデルの i4 — 視認で配布スキップ（2026-08-24）

校正条件のモデル別化（`calib_conditions(model)` — pipeline_config から導出・素版 =
4 prompt × 20 step・CFG 4.0・512²・CFG>1 は両分岐捕捉・negative は uncond へ）を実装し、
3 モデル（anima-v1.0 / wai / copycat）を export（各 ~3h・provenance 3 本とも同条件を記録）。

- 視認（既定 `w8a8-s16` vs `w4`・seed 42・20 step / CFG 4.0 / 1024²）: 3 モデルとも破綻は
  無いが**構図の分岐が大きい**（v1.0 が最大 — 構図完全分岐 + 背景の描き込みが薄い / copycat が
  最小）。多 step × CFG が量子化差を増幅する機序は §2 と同じ。
- **ユーザー裁定: 配布スキップ**（変化が大きく品質保留）。改善候補 = adaLN i8 化・量子化
  感度の高い場所の特定（時間が掛かるため later — backlog）。
- **系列はアーカイブ退避済み**: `outputs/series-archive/2026-08-23-anima-base-i4/`
  （i4 系列 3 本 + 視認物。`outputs/series/` の正本が将来の再 export で上書きされても校正
  結果〈各 ~3h〉を使い回せる。復元 = `outputs/series/` へ書き戻すだけ）。
- **注意: local の `models/karume-anima/` には `w4` / `w4-a8-s16` 席が組み込まれたまま**
  （i4 は dist の宣言必須格納なので外せない — `storages=("f16","i8","i4")`）。次に
  karume-anima を上げ直す時はこの裁定を思い出すこと。
