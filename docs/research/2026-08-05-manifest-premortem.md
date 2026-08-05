# manifest v1 の実装前 pre-mortem（3 レンズ・44 指摘）

> NOTE: 時点スナップショット。文中のパス・コマンド・コミットハッシュは記録当時のリポジトリ構成に基づく。

2026-08-05。ADR 0038 の初版ドラフトに対し、実装前の多レンズ敵対レビュー（並列 3 レッグ:
A = 仕様完全性・実装可能性 16 件 / B = 敵対・堅牢性 14 件 / C = 進化・エコシステム適合 14 件）
を実施した記録。**44 件全件を受理**（形の調整込み・却下 0）し、ADR 0038 は反映済みの改訂版が
正本。ここには裁定の要点と、独立に裏取りした確証だけを残す。

## メインで裏取りした確証（指摘が実測で確定したもの）

1. **`session` の例キー `scoreStorage` は実在しない** — 実フィールドは
   `attentionScoreStorage`（`packages/runtime/src/runtime/executor.ts:340`）。しかも runtime は
   未知オプションキーを黙って無視するため、誤綴りの manifest は **s16 が名前だけになって f32
   で走る**（fail loudly せず）。→ ADR §3「session は manifest 所有の語彙・allowlist 検査」の
   直接根拠。`SessionOptions` に `submitPolicy`（TDR 予算 = ホスト政策）が含まれることも確認 —
   素通し禁止のもう 1 つの根拠。
2. **fetch-cache の path 連結は percent-encode がドットを透過**（`encodeURIComponent("..") ===
   ".."`）— `..` セグメントは URL パーサに正規化され **SHA ピンの範囲外**（同一オリジン任意
   パス・Authorization 付き）へ着地しうる。→ §2 の path 検査を禁止列挙から**許可リスト**へ。
3. **関連リンク 2 本のファイル名不一致**（0033/0034 の実ファイル名と相違）を確認・修正。

## 主要クラスタと裁定（ADR の反映先）

- **沈黙劣化の根絶**（A1/A2/B4/C3/C4）: session/gpuFeatures の allowlist 検査・未知キー拒否・
  runtime への明示写像 → §3。エンベロープ未知キー拒否 + format/pipeline の major 繰り上げ
  規則（manifest は配布済みデータで「未リリース原則」適用外）→ §1（C1/C2/C9）。
- **参照整合の parse 時検査**（A4〜A8/B9/C11）: file/variants 排他・defaultPreset 実在・
  weights 完全写像・重複 path の整合・`__proto__` 等の拒否 → §1/§2/§3。
- **path 許可リスト・3 点セット形式・DoS 上限・abort**（A9/A10/B1/B2/B5/B6）→ §1/§2。
- **取得の意味論**（A3/B3/B8/B11/B13/C6）: karume.json 固定名・revision 1 回解決 + SHA 固定・
  専用キャッシュ名前空間 + auth 隔離・hubUrl は manifest から不可・進捗/キャンセル契約 → §5。
- **weights を `Record<component, label>` へ**（C5）: per-component 量子化混在は実在需要
  （transformers.js の per-module dtype・自前の「VAE は i8 化しない」判断）。配布後の形変更は
  他人のリポを壊すため最初から map → §3。
- **基本形 `{file, extras?}`**（A13/C8）: 付帯資産の追加が破壊的変形にならない形 → §2。
- **エラー分類 + 診断**（A16/B7/B10/C14）: 4 型 + 透過 + 利用可能ラベル一覧 + verifying
  進捗 + onCacheError のアプリ公開 → §5。
- **将来席の明示列挙**（C7/C10）: 別リポ component 参照・requiredLimits → §7（未知キー拒否と
  対にすることで旧 hub の誤解釈を構造的に防ぐ）。

## 教訓（横断）

- 「素通し」と宣言した境界の**向こう側が未知キーを無視する**場合、その素通しは沈黙劣化の
  製造機になる — 素通しできるのは「受け手が fail loudly する」境界だけ。
- セキュリティ検査は禁止列挙でなく許可リスト — 下層（取得層）の URL 組み立て実装を読んで
  初めて「列挙の抜け = traversal」と確定した。**下層のソースを読まずに上層の検証を設計しない**。
- 配布フォーマットには「未リリースだから壊してよい」が適用できない — 互換規則は v1 の時点で
  焼く。
