# 0012 — attrs 語彙の解禁と融合 op 契約（保存リスト 9 op 化）

- Status: accepted（2026-08-02）
- 注記: 「保存（分解抑止）リスト 9 op」は**当時値** — 後続 ADR（0015 leaky_relu で 10 /
  0017 rms_norm で 11 / 0023 SDPA で 12）で拡張されている。現在値の正本は
  [op-vocabulary.md](../op-vocabulary.md)（本 ADR 内でリストを最新化し続けない）。
  attrs 契約そのものは現行。
- 根拠資料: [../research/2026-08-02-deberta-front-recon.md](../research/2026-08-02-deberta-front-recon.md)、
  ADR [0007](0007-op-vocabulary.md)（分解禁止 9 op）

## 決定

- **M0 の「全 op attrs 空」を撤去**する。契約テーブル（TS `packages/runtime/src/ops.ts` / Python
  `karume/ops.py`）に op ごとの attrs スキーマ（許容キー + 型 + 値域）を持たせ、
  未知キー・契約外値は従来どおり fail loudly。attrs の意味は契約テーブルが正本で、
  ir-v1.md には器（op ごとの契約検査）だけを残す。
- **エクスポータの保存（分解抑止）リストを gelu のみ → ADR 0007 の 9 op へ拡張**する
  （linear / layer_norm / softmax / gelu / conv1d / conv2d / conv_transpose1d / embedding /
  masked_fill）。カーネル実装は実測グラフに出るものだけ（conv2d / conv_transpose1d は
  保存のみで P2 実装なし）。
- P2 で新設する融合・構造 op と主な attrs（実測値域はカーネル契約＝ランタイム capability
  側で絞る — ADR 0007 の規律）:
  - `linear`（bias 有無）/ `layer_norm`（normalized_shape, eps）/ `softmax`（dim）
  - `embedding`（padding_idx は forward 不活性 — 契約で「受理し、不活性根拠を明記」）
  - `masked_fill`（value: f32 スカラ）/ `conv1d`（stride, padding — B=1 等は capability）
  - `bmm`（バッチ matmul）/ `gather`（dim）
  - レイアウト系（ADR 0011）: `reshape` / `permute`（dims）/ `expand` / `slice`
  - `cast`（to）/ `bitwise_not`（ADR 0009）/ `sym_prefix_slice`（ADR 0010）
- `masked_fill` の埋め値 −3.4028234663852886e+38（f32 最小有限値）は非有限値拒否
  （ir-v1.md）に抵触しないが、**JSON 往復で ulp 不変**であることをテストで固定する。

## 帰結

- 「op を足すときは TS 契約表・Python 契約表・golden を 1 セットで動かす」規律（台帳）に
  attrs スキーマが加わる。エクスポータの `Emitted` は attrs を運べる形へ拡張。
- softmax は dim = 最終次元のみ受理（実測は −1 のみ。一般 dim は要求実測が出てから）。
- native_layer_norm（3 出力）を IR に持ち込まない — 保存リスト拡張が単一出力契約を守る
  前提になっている（IR スキーマ上は複数出力可だが、M0/P2 の op は全て単一出力のまま）。
