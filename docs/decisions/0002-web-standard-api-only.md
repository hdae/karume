# 0002 — ランタイム本体の依存は Web 標準 API のみ

- Status: accepted（2026-08-01）
- 根拠資料: recon §0/§1（先行実験プロジェクト（以下プロトタイプ）決定 0003 の
  「Deno 移行可能性の維持」が実際に機能し、`deno check` 残差 7 件まで移植コストを
  縮めた実績の再宣言）

## 決定

ライブラリ本体のランタイム依存は **Web 標準 API のみ**とする。外部パッケージ依存 0。
使用を許すのは `navigator.gpu` / `performance.now` / `TextDecoder` / `ArrayBuffer` 系のみ。
ファイル・ネットワーク I/O は本体に持たず、呼び出し側が `ArrayBuffer`（または
チャンク列）を渡す。

実装規約（プロトタイプの deno-port-recon で移植コストの源と特定された項目の予防）:

- 相対 import は常に `.ts` 拡張子付き。Vite 固有構文（`?raw` / `?url` /
  `import.meta.env` / glob import）は禁止。WGSL はテンプレートリテラルで TS 内に持つ。
- TypedArray は `Float32Array<ArrayBuffer>` のように ArrayBuffer 明示で締める
  （プロトタイプの deno check 残差 6/7 件の根治形）。
- Dawn / wgpu の実装差がある API 面（`wgslLanguageFeatures` 等）へ直接依存しない。
  features / limits / 言語機能は**能力検出層 1 箇所**で正規化し、欠落時は空集合に縮退する。

## 帰結

- Deno / ブラウザ両対応が構造的に成立する（プロトタイプで実証済みの経路）。
- transport（Range 取得・Cache API・並列度制御）はライブラリ外（examples / 補助面）。
  プロトタイプ決定 0011 の分離を継承する。
