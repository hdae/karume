# GPTQ 校正の CUDA 実行 — 速度と丸め解の再抽選（実測）

> NOTE: 時点スナップショット（2026-08-28 の初回実測）。配管の設計は `50f094d`
> （`--calib-device`・core JIT デバイス移動）、運用方針の帰結はこのメモが正本。

## 条件

anima-v1.0 `--i4-adaln-i8 --calib-device cuda`（RTX 3080 Ti 12GB・torch 2.13.0+cu130 の
隔離 venv `~/workspace/karume-cuda-venv` — 共有 venv と同版の CUDA ビルド）。CPU 側の対照は
同日・同条件の `gptq-adaln8`（`outputs/series/anima-v1.0-i4-adaln8-dyn`）。

## 結果

- **壁時計 2461.5s（41 分）vs CPU 8907.5s（2h28m）= 3.6 倍**。GPU 100% 常用・VRAM 実測
  ~11.8GB/12.5GB。**OOM 警告 1 回**（stage 23 付近・536,870,912B = in=8192 の f64 Hessian
  ちょうど）— アロケータのキャッシュ解放 + リトライで自己回復し完走。**VRAM 余白は紙一重**
  で、対象を広げる（除外なしの素版 = adaLN 分の H が増える）と落ちる可能性がある。
- **機構は健全**: 全門通過・構成は CPU 産と同一（GPTQ 280 本 + i8 174 本）・provenance
  `gptq-adaln8-cuda` + `device: "cuda"` で配布方式一致から名指しで外れることも確認。
- **継ぎ目の隔離が bit 単位で実証**: 1,029 テンソル中 472 本がバイト同一 — CPU 担当のまま
  残した block 外 RTN と i8 一括（payload + scale 全数）が **1 bit も動いていない**（seam を
  i8 丸め直後に置いた設計どおり）。
- **GPTQ の丸め解は大きく再抽選される**: i4 payload 277 本で相違・**nibble 一致率 平均
  87.97%（最悪 74.6% — `attn1.to_out` 群・最良 99.995%）**・scale も相応に変動。f64 縮約順
  と linalg 実装の差が誤差補償の連鎖（後段 stage は前段の丸め済み活性を見る）で増幅する機序
  — 「丸め解はデバイスで変わる」（ADR 0080 旧ドラフト以来の前提・sim→出荷の教訓と同族）の
  実測確認。

## 運用方針（帰結）

- **探索（仮説スクリーニング）は CUDA で回す** — 未検証軸（校正標本増量・Hessian rank 仮説
  等 — [adaln8-visual](2026-08-28-anima-adaln8-visual.md)）が 41 分/本のオーダーになる。
- **配布候補の最終焼きは CPU** — provenance の方式一致門が既に強制。CUDA 産の品質は
  バイトからは論じられない（等価「傾向」を見たければ視認だが、eval_dist は
  `gptq-adaln8-cuda` を受けない — 受理方式ノブの追加はその実需が出たときに）。

## 未検証

- CUDA 産の視認品質（CPU 産との A/B — 変種不採用のため今回は不実施）
- 素版（除外なし）での VRAM 成立性（adaLN 168 本の H が加算 — 上記の余白を食う）
- 速度内訳（capture / 丸め / emit の分解 — ログ粒度が stage 単位のため未分解）
