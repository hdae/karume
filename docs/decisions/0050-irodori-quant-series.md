# 0050: Irodori 量子化系列 — 波 1（f16）の形と門

- Status: accepted（波 1 = f16。i8 / w8a8 は本 ADR を拡張する後続の波で裁定）
- Date: 2026-08-12
- 関連: ADR [0006](0006-quantization.md)（fake-quant 方法論）/ [0018](0018-f16-weight-execution.md) /
  [0027](0027-sbv2-f16-series.md)（f16 系列の先行形）/ [0029](0029-sbv2-i8-series-and-quant-quality.md)
  （系列×格納 dtype の検出限界）/ [0048](0048-irodori-host-port.md) / [0049](0049-irodori-codec-integration.md)。
  判断材料の recon = [research/2026-08-12-irodori-quant-recon.md](../research/2026-08-12-irodori-quant-recon.md)

## Context

Irodori-TTS v4 の配布形は f32 単一系列（8 グラフ・3.44GB）だった。量子化の受け皿（ランタイムの
f16/i8 格納実行・dist の quant 表・models の session 配線）は SBV2 / Anima で完成済みで、
不足はエクスポータの dtype 軸と品質門の設計だけ。recon の実測で 8 グラフ全ての重みバイトの
99.8〜99.9% が重みスロットのみ消費（= 圧縮適格）と確認済み。ユーザー裁定（2026-08-12）は
3 波構成 — 波 1 f16 → 波 2 i8 + 品質台本 → 波 3 w8a8。

## Decision（波 1）

1. **f16 は別系列**（`outputs/series/irodori-v4-small-f16/` / `dacvae-32dim-f16/`）。
   fake-quant は各台本の**実効重み確定後・参照/golden 採取前**に当てる — export_irodori /
   irodori_pipeline は `load_*` 直後（remove 相当は変換時に焼き込み済み）、export_dacvae は
   `fold_weight_norm` → 門 3 決着 → `lift_snake_alphas` → 丸め → golden 採り直し → 切り詰め、
   の順（門 3 は丸め前の参照で決着させるので主張は弱まらない）。**Snake の alpha は丸めない**
   （lifted 定数 = 重みスロット外・golden も f32 alpha で計算 — 両者は整合）。
2. **配布形のレイアウトを `<role>/model.{f32,f16}.safetensors` へ変更**（Anima / SBV2 と同形・
   未リリースにつき破壊的変更で正）。quant 席は f32 / f16 の 2 席・混成なし（f16 に S を動かす
   軸が無いため一律で落とせる — 混成が要るのは i8 の波）。defaultQuant は f32 のまま。
3. **系列×格納 dtype は両側から挟んで一意にする**。既存の存在検査（`assert_storage`）だけでは
   **f32 席への f16 資産の挿し込みが素通りする**（圧縮系列のコンテナも bias / norm / グラフ
   定数を F32 で持つため「F32 を含む」が真になる — SBV2 では 2 席とも圧縮系列でこの穴が露出
   しなかった）。`assert_storage_absent` + 禁止表（f32 席は F16 禁止）を新設し、逆向きは既存の
   存在検査が落とす。グラフ宣言との突合（`assert_irodori_graphs`）も全 dtype 系列に掛ける。
4. **latent 門は系列パラメタ化**し、tolerance は系列ごとに素の実測から独立導出（ADR 0027 の
   型・流用禁止）。**未導出の系列は SKIP ではなく「実測を出して赤」**（資産があるのに何も
   検証していない緑を作らない）。S / forwards の完全一致は系列に依らず要求する。
   実測（2026-08-12・実 GPU）: f16 は S/forwards が f32 と完全一致・z maxAbs は
   full 2.1321e-4 / no-ref 1.8516e-4 → **F16_Z_ATOL = 1.5e-3**（最悪の 7.0 倍）。
5. **WAV sha256 門は f32 専用のまま**（門自身が tolerance 化・参照差し替えを MUST NOT と
   宣言している）。f16 の最終裁定は聴感（ユーザー）— A/B は同一テキスト・同一 seed の WAV 対で
   提出する。上流突合（`EULER_REFERENCE_ATOL` 1e-3）は**系列共通**のまま — 丸めはグラフ経路と
   上流経路の両辺に同じ 1 回だけ当たるので量子化誤差は相殺し、測るのは f32 と同じ「実装差
   だけ」（f16 実測 2.1e-4 / 7.8e-5 で成立を確認済み）。
6. **i8 / w8a8 は後続の波**。分岐点は S ドリフト（duration を i8 にすると発話長が動きうる —
   SBV2 w8 の w_ceil 198→196 と同じ軸）で、混成表（duration 据え置き）の裁定は
   `measure_quant_irodori`（波 2 で新設）の実測後に行う。

## Consequences

- 配布 3,438,182,144 → f16 系列 1,721,312,568 B（**50.1%** — 実測 2026-08-12）・
  DiT 常駐 1.46 → 0.73GB。
  実測サイズは dist 出力（各 role の `model.f16.safetensors`）が正本。
- `karume dist --pipeline irodori` は**両系列が揃わないと組めない**（f32 単独の組み立て不可）。
  再組み立て時は旧レイアウトの残骸があると宣言外ファイル検査で落ちる（先に消す）。
- 計算は f32 のまま（格納だけ f16 — ADR 0018）なので速度利得はロード/VRAM のみ。速度の軸は
  波 3（w8a8）で、着手前に DiT の linear/bmm 比の実測が要る（recon の risks）。
