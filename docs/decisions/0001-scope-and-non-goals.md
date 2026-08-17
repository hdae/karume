# 0001 — スコープと非目標

- Status: accepted（2026-08-01、ユーザー裁定）
- 改訂: ADR [0037](0037-karume-monorepo.md) §1 が「リポジトリ構成はフラット・workspace 機構は
  使わない」の 1 点を置換（Deno workspace の monorepo）。他の決定・非目標は現行 —
  帰結節にも同旨の追記がある。
- 根拠資料: [../research/2026-08-01-prototype-recon.md](../research/2026-08-01-prototype-recon.md)

## 決定

Karume は **Deno とブラウザの両方で動く WebGPU 汎用 NN 推論ランタイム**である。
JSR に `@karume/runtime` として公開する。モデルは PyTorch（`torch.export`）からの
エクスポートを想定する。

- **先行実験プロジェクト（以下プロトタイプ）から独立**: プロトタイプのコード・型を
  import しない。参照・移植のみ（本プロジェクトはプロトタイプ決定 0013 で決定された
  JSR/Deno 移行の実行体）。
- 単なる移植ではない: プロトタイプで実測済みのボトルネック改善・バグ根治・設計の堅牢化を
  最初から織り込む（recon §9 の弱点 10 項目が対象台帳）。
- 実装基盤は WebGPU 自前カーネル（WGSL 実行時 codegen）。プロトタイプ決定 0001 の WebNN 却下理由
  （int4 型欠如・精度適合要件なし・custom shader 不可・in-place 禁止）を継承する。

## 非目標

- ONNX 入力（プロトタイプの roadmap の非目標を継承）。torch.export → 自前 IR のみ。
- WASM フォールバック。
- WebNN バックエンドの即時対応 — ただし IR は「4 境界」（レイアウト非焼き込み /
  量子化の論理表現保持 / op 粒度を細かく / 初期化の非同期許容）を守り、将来の
  後付けバックエンド（NPU 入口）の余地を残す。
- `navigator.gpu` が得られない環境（素の Node 等）のサポート。
- MiniMax-M3 級（428B/23B MoE・ブロックスパース MSA）のブラウザ実行 — メモリ規模的に
  現行スコープ外（2026-08-16 裁定。autoregressive 波の検収は Gemma 4 E2B /
  MiniCPM5-1B — [backlog](../backlog.md) next 節）。

## 帰結

- リポジトリ構成はフラット（ユーザー裁定）: ルート = ライブラリ本体（JSR 公開対象）、
  `tools/exporter/`（Python, uv）、`examples/`、`bench/`。workspace 機構は使わない。
  **→ この 1 点は ADR [0037](0037-karume-monorepo.md) で改めた**（Deno workspace の monorepo・
  `@karume/{runtime,hub,models}` の 3 パッケージ）。本 ADR の他の決定・非目標は有効。
- エクスポータは Karume 配下に新設（ユーザー裁定）。プロトタイプの実装知見を移植する。
