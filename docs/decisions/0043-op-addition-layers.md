# 0043: op 追加は第 0〜3 層の判定手順に従う（必須プリミティブと推奨カーネル/融合の分離）

- Status: superseded by [0059](0059-op-vocabulary-entry-doors.md)（2026-08-14 — 層定義・
  判定手順は入場門モデル・名前制へ置換。本文と追記は当時の記録として保存）
- Date: 2026-08-11
- 関連: ADR [0012](0012-attrs-and-fused-ops.md)（attrs 空契約）/
  [0015](0015-conv-family-extension.md) / [0017](0017-rms-norm-conv2d-clamp-min.md) /
  [0023](0023-fused-attention.md) / [0040](0040-fusion-pass.md)
  — いずれも本手順の先行適用例（当時は暗黙だった判定を本 ADR が明文化する）。
  運用チェックリストは [op-vocabulary.md](../op-vocabulary.md) の「op 追加の判定手順」節。

## Context

- 語彙は実態として 4 層構造（export 時消滅 / Core ATen プリミティブ / IR 可視の保存 op /
  実行時融合）で運用されてきたが、まとまった記述が無く、新モデル移植で未対応 op が出る
  たびに配置の議論をやり直していた。
- EmbeddingGemma 移植で `gelu(approximate="tanh")` が必要になり（Gemma 系は全員 tanh 近似）、
  ユーザーの整理 — **必須 op = 原子（増え方は有限収束）・推奨 op = 分子（性能根拠で追加）、
  推奨は「実行時融合」と「IR 可視カーネル」の 2 形** — を規約として固定する。

## Decision

### 層の定義

- **第 0 層 — export 時消滅**: `FOLDABLE_OPS` の定数畳み込み、または normalize / convert の
  分解・正規化パス。ランタイムに届かないためコミットメントなし。最優先の逃げ道。
- **第 1 層 — 必須プリミティブ（原子）**: IR 語彙のうち Core ATen 由来。**母集団は
  Core ATen 160 で固定**（実装対象 120）— 増えるのは被覆であって天井ではない。
- **第 2 層 — 推奨カーネル（分子・IR 可視）**: 分解禁止の保存 op（linear / attention /
  rms_norm …）と、attr 変種の別 op 化（gelu_tanh）。IR ファイルに名前が焼かれるため
  **恒久の公開コミットメント**（既存資産が全将来ランタイムに要求する）。追加は ADR 必須。
- **第 3 層 — 推奨融合（分子・実行時のみ）**: `fusion.ts` の隣接ノード matcher
  （silu / rope / upsample2x / adaln）。IR に痕跡を残さず撤回自由。ただし exact-match が
  外れると**黙って遅くなる**ため、観測点（`lastRunFusions` / assets 突合テスト）が入場条件。

3 層とも実行の実体はカーネルである。分ける軸は「カーネルか融合か」ではなく
**分子の名前がどこに書かれるか**（IR ファイル = 第 2 層 / ランタイム内部 = 第 3 層）。

### 判定手順

未対応 op が `UnsupportedAtenOpsError` に出たら上から順に安い層へ落とす
（チェックリストの正本は op-vocabulary.md）:

1. 値が export 時定数 → 第 0 層 FOLDABLE
2. 既存プリミティブの合成で厳密同値（**torch 突合ゲート必須** — 台帳の反例集が根拠）→
   第 0 層の分解
3. 合成が数値的に不可能・容量/性能で非成立、かつ Core ATen 内 → 第 1 層
4. Core ATen 外・attr 変種・融合維持が必要 → 第 2 層（ADR とセット）
5. 正しく動く分解形が既にあり性能だけ回収したい → 第 3 層

### 初適用: gelu_tanh は第 2 層

- 手順 2 で止まらない理由: 分解（≈9 ノード）は可能だが、後から性能を回収するには
  第 3 層 matcher の脆さを 1 つ増やすことになる。再出現率が高い分子
  （EmbeddingGemma / Gemma 4 E2B の config で `gelu_pytorch_tanh` を確認済み）は
  最初から第 2 層に置く方が総コストが低い。
- 既存 `gelu`（erf 形）の契約は無変更。attrs 空契約（ADR 0012)を守り、attr 変種は
  **別 op** として語彙に足す — attrs に `approximate` を載せる案は既存契約の改訂になるため
  不採。

## Consequences

- op-vocabulary.md に運用チェックリスト節を追加（`DECIDED:` は本 ADR を指す）。
- 第 2 層への追加は今後も「1 件 = 小 ADR 1 件 + 契約 1 セット（TS/Python の 2 実装 +
  fixture + CPU 参照 + golden COVERAGE）」が単位。
- 第 3 層の入場条件（観測点必須）は ADR 0040 の規律を継承し、本 ADR で層の名前を得た。
- 「やらない方がよいこと」（対称性のための追加・160 埋めの目標化・汎用 reduce
  フレームワーク先行）は全層に優先する。

## 追記（2026-08-12）— 判定軸の一本化と第 1' 層の新設（網羅レビュー D1 裁定）

網羅レビュー（`.claude/reviews/2026-08-12_faef828/`）で、第 1/2 層の判定基準が本文に
2 つ書かれており非等価であることが確定した: Decision 節は「Core ATen 由来 = 第 1 層 /
分解禁止の保存 op = 第 2 層」、判定手順は「Core ATen 内 → 第 1 層」。分解禁止 12 op の
うち 9 本は Core ATen 由来で両基準を同時に満たし、`gelu` は読み方で層（= ADR 要否）が
変わる。また `sym_prefix_slice` はどの層にも属さない。

### 決定

1. **第 1/2 層を分ける軸は「語彙内の他 op の合成で厳密同値に書けるか」（原子/分子）に
   一本化する。** 原子（合成不能）→ 第 1 層 = 台帳のみ。分子の保存（合成で書けるが性能
   根拠で保存する選択）と attr 変種の別 op 化 → 第 2 層 = ADR 必須。「Core ATen 由来」は
   第 1 層の**母集団の説明**（有限収束の根拠）であって判定軸ではない。
2. **第 1' 層 = IR 機構 op を新設**: Core ATen 外の原子で、モデルではなく karume の IR /
   実行モデル自体が要求するもの。恒久の公開コミットメント・ADR 必須。初例は
   `sym_prefix_slice`（ADR 0010 の次元言語が要求）。将来の同類: Session 常駐値のハンドル・
   KV 追記の表現。
3. **境界の確定**（軸の適用例 — 導入済み op の手続き遡及はしない）: `gelu` は erf が語彙に
   無く合成不能 = **原子（第 1 層）**。分解禁止リストの所属は不変 — 分解禁止はエクスポータ
   の畳み込み方針であり、層とは独立の軸。`leaky_relu` は `gt_scalar + mul + where` で書ける
   **分子（第 2 層・根拠 ADR は 0015）**。`relu` / conv 族など導入済みの op は再割当しない
   （層は今後の追加手続きだけを支配する）。
4. **記録の訂正**: ADR 0007 決定 3 の「行 reduce 族 6 本を最初期に実装」は未充足のまま —
   実装は sum / amax / amin の 3 本（safe-softmax の前提分）。argmax（greedy デコードの
   出口）は LLM 波で第 1 層追加 1 本（≈20 行）として再訪する。op-vocabulary.md の実装順序
   節に NOTE を追記した。

### 帰結

- op-vocabulary.md の判定手順（層表 + チェックリスト）を本追記の軸へ更新。
- 実装・既存資産・既存 op の扱いへの変更はゼロ（手続きの明確化のみ）。
- 次モデル BiRefNet_HR（deform_conv2d / grid_sample 系）の層判定は本軸で行う。
