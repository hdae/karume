## 追記（2026-09-02 — Phase B 実測と Phase C-1・見積り結線の据え置き）

- **Phase B 実測**（[research/2026-09-02-shard-size-ram-peak.md](../research/2026-09-02-shard-size-ram-peak.md)）:
  ホスト RAM ピークは「定数 + k × 最大 shard」で k ≈ 3（Linux）/ 2.4（Mac）。shard 内の完了待ちを
  刻む案は Vulkan で無効・Metal で −13%。明示 GC で shard 1 本分。効くレバーは shard サイズと、
  shard を読む器の使い回し。
- **Phase C-1 = 器の使い回し**を実装（契約変更は ADR [0070](0070-shard-loading-admission.md) 追記
  2026-09-02 が正本）: ピーク ≈ 0.45GB + 最大 shard 1 本（anima f16 1GiB shard 4,069 → 1,402 MiB・
  gemma4 2,622 → 1,116 MiB）。HF 経路は取得層側の対応待ち。
- **`estimateSessionMemory` のロード面結線は据え置きを確定**: Phase B で「判断に効く数字」はホスト側
  （manifest の最大 shard から導ける）であって GPU 見積りではないと分かった。GPU 見積りは引き続き
  呼び手が生成形状を決めた後に使う面（limitations「未実装」節はそのまま）。
- 残る裁定 = 書き手の shard 目標値（512 or 256MiB — ADR 0081 側の 2 値化）。C-2 候補 = テンソル単位
  ストリーミング（ピーク → 定数 + 最大テンソル）。
