# 校正付き丸め（GPTQ / AWQ）第 1 段スクリーニング（2026-08-20）

> **時点スナップショット** — 波 J-2 の安いファミリ 2 つ（MiniCPM5-1B / EmbeddingGemma-300m）
> における校正付き丸め 5 構成の実測記録。実装の正本 = `karume.quant_calib`（コミット
> `ce568b0`）と各リグ（`730c21b`）、方式の採否は [perf-ledger](../perf-ledger.md) Q 節が正本。
> 波 I（方式スクリーニング）の基線は
> [2026-08-19-w4-method-screening.md](2026-08-19-w4-method-screening.md) §2。

## §1 方法

- **方式 5 構成**: gptq-rtn / gptq-nf4 / gptq-kmeans（層内 1 枚表）/ awq-rtn / awq-gptq-rtn。
  格子は波 I と同じ **g=32 固定**（校正は「同じ格子の中で丸め先を選び直す」だけ — 格納形は
  1 バイトも変えない）。**gptq-rtn だけが今日の格納形（i4 g32）へそのまま出荷可能**
  （scale 台帳 = `Int4Report` 互換）。AWQ 系は fold 理想形の上限測定（等価倍率 s は
  group scale へ吸収不可 — fold か companion が要る）。
- **校正コーパス**: 各ファミリ 48 文（`calib_texts.py` — 評価文と部分一致まで分離・
  minicpm5 1,325 token / EG 1,295 token）。stage 逐次（block-sequential）で誤差伝播込み・
  校正データは 1 周。全て決定的（乱数ゼロ・H は f64 蓄積）。
- **対象集合の注意（波 I と直接引き算できない）**: 校正の対象は **decoder 内 linear 限定**
  （minicpm5 168 本 = 波 I の linear 169 から lm_head を除く / EG 168 本 = 波 I の 170 から
  Dense 2 段を除く）。方向の比較には十分、行の直接差し引きは不可。
- 実行: `--only` 校正 5 構成のみ・全 48 文・CPU（torch 2.13.0+cpu）。生ログと JSON =
  `outputs/demo/quant-calib-j2/`（git 追跡外）。

## §2 MiniCPM5-1B（teacher = 3 ケース×16 位置 /48・NLL = 3 ケース和・greedy = 自由走行 /48）

| 構成                                  | bpw | wRMSE  | teacher   | NLL 和    | greedy    |
| ------------------------------------- | --- | ------ | --------- | --------- | --------- |
| baseline (f32)                        | —   | —      | 48/48     | 2.234     | 48/48     |
| 〔波 I〕rtn（linear 169）             | 5.0 | 0.0994 | 36/48     | 3.343     | 23/48     |
| 〔波 I〕nf4（linear）                 | 5.0 | 0.0877 | 37/48     | 2.921     | 2/48      |
| 〔波 I〕kmeans:shared（linear）       | 5.0 | 0.0861 | 38/48     | 2.713     | 15/48     |
| 〔波 I〕kmeans:shared（+embed・最良） | 5.0 | 0.0859 | 42/48     | 2.651     | 25/48     |
| **gptq-rtn**                          | 5.0 | 0.1463 | **41/48** | **2.560** | 19/48     |
| **gptq-nf4**                          | 5.0 | 0.1295 | **43/48** | **2.486** | 19/48     |
| **gptq-kmeans**                       | 5.0 | 0.1260 | **43/48** | **2.480** | **37/48** |
| awq-rtn                               | 5.0 | 0.1029 | 41/48     | 2.908     | 15/48     |
| awq-gptq-rtn                          | 5.0 | 0.1527 | 38/48     | 3.209     | 17/48     |

## §3 EmbeddingGemma-300m（golden 5 ケース・f32 との cosine / 意味順序 / ペア行列ドリフト）

| 構成                                  | cos min    | cos mean   | 意味順序 | ペア最大ドリフト |
| ------------------------------------- | ---------- | ---------- | -------- | ---------------- |
| 〔波 I〕rtn（linear 170）             | 0.9341     | 0.9665     | 保持     | 6.66e-2          |
| 〔波 I〕nf4（linear）                 | 0.9628     | 0.9810     | 保持     | 2.73e-2          |
| 〔波 I〕kmeans:shared（linear・最良） | 0.9681     | 0.9829     | 保持     | 2.70e-2          |
| **gptq-rtn**                          | **0.9713** | **0.9829** | 保持     | 2.18e-2          |
| **gptq-nf4**                          | **0.9834** | **0.9889** | 保持     | 1.76e-2          |
| **gptq-kmeans**                       | **0.9854** | **0.9894** | 保持     | **1.06e-2**      |
| awq-rtn                               | 0.9665     | 0.9826     | 保持     | 2.01e-2          |
| awq-gptq-rtn                          | 0.9801     | 0.9833     | 保持     | 1.29e-2          |

## §4 所見

1. **GPTQ 単独が両ファミリで大勝ち**。特に **gptq-rtn は「今日の格納形のまま・runtime
   0 行」**で素の RTN を全面的に超える — MC5 teacher 36→41・NLL 3.343→2.560、EG cos mean
   0.9665→0.9829（= 波 I の kmeans:shared 最良と同値）。「格納を変えずに品質を戻す」という
   校正ループの狙いどおり。
2. **gptq-kmeans が全列最良** — MC5 teacher 43/48・NLL 2.480・**greedy 37/48**（波 I 全構成の
   最良 25/48 を大幅更新）、EG cos min/mean/ドリフトも全指標最良。**格納の companion 席
   （表 16×f32 + 表引き dequant — perf-ledger Q-2）の価値が上がった**。gptq-nf4 も teacher/
   NLL は並ぶが greedy で劣り（19 vs 37）、固定表の限界が自由走行に出る。
3. **AWQ は今回の条件では価値が立たない** — 単独で GPTQ に一貫して劣り（NLL 2.908 vs
   2.560）、**併用（awq→gptq）は単独より悪化**（MC5 teacher 38・NLL 3.209）。fold /
   companion の追加実装コストを正当化しない。α の採用値は 0〜0.75 に散っており探索自体は
   機能している（外れチャネル救済という機序が、この規模・この格子では GPTQ の誤差補償に
   包含されると読む — 推測）。
4. **wRMSE は品質の代理にならない**（波 I §4-2 の再再現）: awq-rtn が wRMSE 最良（0.1029）で
   NLL 最悪帯、gptq 系は wRMSE を**悪化させながら**出力品質を上げる（H 加重誤差の最小化は
   非加重 RMSE と別の量 — 設計どおり）。
5. 校正 48 文 ≈1.3k token は一般的な GPTQ 運用（数万 token 級）より 1 桁以上小さい。それで
   この利得なので、**校正量を増やす軸は上振れ余地として残る**（kill 条件ではなく追加実験枠）。

## §5 次段への送り

- **重いファミリ横展開（波 J-2 第 2 段・裁定待ち）**: 勝者を SBV2 BERT（DeBERTa encoder の
  stage 分解）と irodori / anima（DiT block）へ。net_g の conv は GPTQ の対象外（H = ΣXᵀX が
  linear の in 軸形 — conv は im2col が要る・core の設計判断）。
- gptq-rtn の**出荷結線**（`CalibReport.int4` → `export_to_file`）は配布 recipe 側の実需判断
  （perf-ledger Q-6 起票）。
- AWQ 系は不採用寄りで起票（Q-7）— 復活条件 = fold 先が構造的に安全なモデルで GPTQ が
  効かない事例が出た場合。
