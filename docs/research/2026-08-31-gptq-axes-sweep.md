# GPTQ 掃引軸の実測 — static-groups / act-order / damping（minicpm5・2026-08-31）

> 性格: **時点スナップショット**（実測記録）。実装 = `892bfb3`（core・既定 off でビット同一）+
> `683d6a0`（minicpm5 sweep の EXPERIMENT_CONFIGS）。生ログ / JSON はセッション scratchpad
> （揮発）— 再実行は `uv run python -m minicpm5.sweep_w4 --only gptq-rtn-static …`。

## 経緯

外部レビュー（ChatGPT）の「upstream `--static-groups` なら act-order + 推論側変更なしが可能 —
不採用理由の再検討を」を検証（G2-2 holds）→ opt-in 軸として実装し、MiniCPM5-1B（i4 g32・
校正 48 文 ≈1.3k token・CPU）で 8 構成を実測した。

## 結果（teacher-forced 一致 / NLL 和 / greedy 48 文・全構成 wRMSE と bpw 5.0 は同一）

| config                                          |  wRMSE |   teacher |      NLL |    greedy |
| ----------------------------------------------- | -----: | --------: | -------: | --------: |
| baseline（f32）                                 |      — |     48/48 |    2.234 |     48/48 |
| **gptq-rtn（現行既定 = dynamic group・λ0.01）** | 0.1463 | **41/48** |     2.56 | **19/48** |
| + static_groups                                 | 0.1457 |     36/48 |     2.72 |     14/48 |
| + static + act_order                            | 0.1474 |     40/48 |     2.79 |     11/48 |
| damping 0.001                                   | 0.1871 |     34/48 |    3.358 |      5/48 |
| damping 0.003                                   | 0.1664 |     39/48 |    2.678 |     13/48 |
| damping 0.03                                    | 0.1308 |     36/48 |    2.818 |     12/48 |
| damping 0.1                                     | 0.1172 |     40/48 | **2.39** |     18/48 |

## 読み

1. **act-order / static-groups は本条件で利得なし**（teacher / greedy とも現行既定が最良）。
   upstream の報告利得（LLaMA-7B ppl 7.15→6.09 等）は桁違いに大きい校正コーパスでの値で、
   1.3k token では act-order の並べ替えがノイズ側に働くと読める。wRMSE はほぼ同一 =
   重み空間の差は微小で、評価指標の差は生成の分岐感度が主。
2. **damping は 0.01（現行既定）が妥当**。小さくするほど明確に悪化（0.001 は全指標で最悪）。
   0.1 は NLL / wRMSE で僅かに勝るが greedy は同等以下 — 乗り換えの根拠にならない。
   「0.01 固定」は実測で封印された（G2-3 の狙いどおり — この結果自体が成果）。
3. **裁定案: 既定は現状維持**。軸は掃引基盤として温存（ビット同一 off・掃引時のみ発火）。
   再評価の復活条件 = **校正量 16×**（`--calib-limit` 拡張 + `assert_calib_disjoint` の門 —
   act-order は校正量と交互作用する仮説が本実測では未検証）。gemma4 への校正 rig 新設は
   この復活条件が満ちるまで保留（minicpm5 の rig 写経で 1〜2 日級）。

## 訂正（計画段階の誤りの記録）

- 検証レッグの段 2 案にあった「SBV2 decoder の GPTQ 席」は存在しない（`calibrate_stages` は
  `nn.Linear` 限定・net_g は conv 主体 — 校正席は BERT linear のみ）。
- 「damping 掃引は H を 1 回蓄積して使い回せる」も不成立（stage 逐次の誤差伝播で stage 1
  以降の H が damping に依存 — 実コストはフル校正 × 構成数。本実測 8 構成 ≈ 55 分/CPU）。
