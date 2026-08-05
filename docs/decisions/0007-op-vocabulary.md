# 0007 — op 語彙（Core ATen 160 母集団と拡張規律）

- Status: accepted（2026-08-01、台帳の引き継ぎはユーザー明示指示）
- 根拠資料: [../op-vocabulary.md](../op-vocabulary.md)（正本の台帳）、recon §5

## 決定

- op 語彙の母集団は **Core ATen IR の 160 op**（torch.Tag.core）。5 分類
  （不要 25 / プリミティブ 75 / 合成 38 / 嬉しい 7 / 困難 15、実装対象 120）と
  実装順序は先行実験プロジェクト（以下プロトタイプ）の検討結果を
  [docs/op-vocabulary.md](../op-vocabulary.md) に引き継ぎ、
  以後 Karume 側で更新する。
- IR op 語彙は **allowlist 凍結**とし、追加は明示的な行為（台帳更新 + 契約テーブル更新 +
  golden テスト追加が 1 セット）。
- **行 reduce 族（amax/amin/max/min/argmax/argmin）を最初期に実装**する
  （プロトタイプ最大の穴。safe-softmax・greedy デコードの前提）。
- **分解禁止（融合維持）9 op** をエクスポータの分解抑止リストとして最初から持つ:
  linear, layer_norm, softmax, gelu, conv1d, conv2d, conv_transpose1d, embedding,
  masked_fill。
- 分解禁止リストの追補（2026-08-02 / ADR 0015）: `leaky_relu` を追加して **9 → 10 op**。
  分解形（`gt_scalar + mul + where`）は中間バッファが 1.5〜2 倍に膨らみ、メモリ見積の
  前提が崩れるため（dec 専属で conv 族と同波に入った）。正本は
  `karume.convert.PRESERVED_OP_PREFIXES`。
- 分解禁止リストの追補（2026-08-02 / ADR 0017）: `rms_norm` を追加して **10 → 11 op**。
  保存だけでは足りない唯一の op（`aten.rms_norm` を出すのは diffusers `nn.RMSNorm` 経由
  だけで、手書き分解形は normalize の `_fold_rms_norm` が合成する — 供給 2 系統）。
- **「分解で済む」判断は必ず torch 突合ゲートを通す**。プロトタイプの敵対検証で log1p 不在 /
  pow 負底 / max の NaN / var correction / inf−inf=NaN など多数の分解案が誤りと判明済み。
- scatter 系は **CAS 実装禁止**（WGSL に atomic<f32> が無く、加算順非決定が決定性
  不変条件を壊す）。静的添字は gather へ反転。汎用 scatter_add は個別 ADR 前提。
- カーネル契約（B=1 等の受理制約）は IR 仕様に染み出させず、**ランタイム capability +
  明確な診断**として表現する（v0 の反省点）。

## 帰結

- 台帳の op 単位の完全表は、op 実装マイルストーン開始時にプロトタイプ原本
  （プロトタイプの `docs/op-vocabulary.md`）から逐語移植する（機械的タスクとして委任可）。
- WGSL の NaN 比較 / sign(NaN) / 長さ 0 スライス binding は着手前に本環境で実測して
  台帳の未決を潰す。
