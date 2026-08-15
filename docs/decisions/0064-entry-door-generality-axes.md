# 0064: 入場門に「意味論の射程」と「実行モデル不変条件」の 2 軸を追加する

- Status: accepted（2026-08-15 — 外部レビュー〈汎用性監査・全件照合済み〉の裁定。
  ADR [0059](0059-op-vocabulary-entry-doors.md) の**改訂（追加）** — Core ATen 1 判定
  〈ADR 要否 = 手続きの重さ〉は不変で、門を通るときに答える軸を 2 本足す）
- 関連: ADR [0055](0055-deform-conv2d.md) / [0056](0056-gru-scan.md)（軸 A の実例）/
  [0060](0060-row-block-attention.md)（在庫ずれの実例）/ [0012](0012-attrs-and-fused-ops.md)
  （attrs 空契約 — 軸 A の背景機構）

## Context

- ADR 0059 の門は「ADR を書く必要があるか」を決める。しかし ADR さえ書けば、**1 モデルの
  観測 subset をそのまま op の意味論として**公開語彙に入れられる。実例: `deform_conv2d` は
  stride / dilation / groups / offset_groups を「欄の不在」で 1 に固定した DCNv2 専業 subset
  が意味論そのもの（ADR 0055）。`gru_scan_reverse` は bool attr の検証機構が無いことを理由に
  **op 名の分裂**で方向を表した（ADR 0056）。どちらも fail loudly で正しさは保たれているが、
  「実装 capability を狭くする」と「op の意味論を狭くする」の同一化が慣行になると、新モデル
  流入時に variant 爆発と公開契約の負債になる。
- Core ATen 層は台帳 NOTE のみで入るが、**Core op でも既存実行モデルの不変条件を壊すものが
  ある**。典型 = `topk` / `sort` / `argmax`（いずれも `Tag.core=True` — torch 2.13.0 実測）:
  values + indices の 2 出力は「全 op 単一出力」（ops.ts の契約検査・NodePlan の 1 本持ち）を
  壊し、追加は kernel 1 枚でなく planner / dtype / shape / recipe / executor の横断改修に
  なる。現行の門はこれを NOTE 1 本で通してしまう。
- ADR 0059 決定 6 の在庫表（融合層「4 ルールちょうど」）は ADR 0060 の追加で即 stale に
  なった — ADR 内に「現在の在庫」を書く方式は保守されない。

## 決定

1. **軸 A — 意味論の射程の明示**: 新規 op（および既存 op の契約拡張）の ADR / 台帳 NOTE は、
   **semantic surface（op が約束する意味論）と実装済み subset（現在のカーネルが受ける形）を
   区別して書く**。観測 subset を意味論として焼く場合はその旨を明示し、一般化の条件
   （どの需要が来たら欄を足すか）を 1 行書く —「欄の不在 = 語彙に無い」を暗黙にしない。
   既存 op への遡及適用はしない（公開前の統一再裁定は backlog release の別項目）。
2. **軸 B — 実行モデルの不変条件を壊す追加は Core ATen 帰属でも ADR 必須**: 対象の
   不変条件 = 単一出力 / 静的形状（データ依存出力形状なし）/ full-write / 4 バイト格納 +
   意味論 dtype 3 種 / resident の全域書き。壊す追加は、影響範囲（planner / executor /
   recipe / IR …）の先出しを ADR の必須節にする。判定手順 3（core → 台帳 NOTE のみ）の
   例外として挿入する。
3. **語彙・ルールの「現在の在庫」の正本は台帳と source に一本化**: ADR には時点の在庫を
   書かない（書く場合は時点記録と明示する）。融合ルールの現在値の正本 = `fusion.ts` の
   `FUSION_RULES` と融合門テスト（ADR 0059 決定 6 の表は時点記録として読む）。
4. **機構不足を op 名の分裂で埋めるときは明示する**（SHOULD）: 汎用 attr 機構（bool attr 等）
   の不足が理由で別 op 名を切る場合、ADR にその旨と「機構が入ったときの統合条件」を書く
   （`gru_scan_reverse` を前例として繰り返さないための記録義務 — 分裂自体の禁止ではない）。

## 帰結

- [op-vocabulary.md](../op-vocabulary.md) の判定手順に軸 A / B を反映（手順 3 に例外・
  手順 4 の ADR 必須節に射程の明示を追加）。門表の融合層在庫は source を指す形へ。
- ADR 0059 は冒頭に本 ADR の改訂注記を付す（本文は不変）。
- **LLM 波の `topk` / runtime multi-output（backlog next）は軸 B の最初の適用対象** —
  multi-output の横断改修はその ADR で影響範囲を先出しする。
