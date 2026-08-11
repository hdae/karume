# 0046: cat の連結軸を「同一シンボルの一次和」まで緩和する

- Status: accepted（ユーザー裁定 2026-08-11 — 案 c）
- Date: 2026-08-11
- 関連: ADR [0014](0014-layout-ops-full-write.md)（本 ADR が改訂する「連結軸は静的」）/
  [0010](0010-symbolic-constant-folding.md)（次元言語 `coeff·sym+offset`）/
  [0044](0044-runtime-attention-mask.md)（実行時マスク — 本緩和と組で DiT が開通する）/
  実測 = [research/2026-08-11-dit-export-recon.md](../research/2026-08-11-dit-export-recon.md)

## Context

- Irodori DiT の joint attention は K/V = `cat([self | text | speaker | caption], dim=1)` で、
  self 側の軸長が記号（S）。ADR 0014 の「`cat` の連結軸は静的」と正面衝突し、これを解かないと
  DiT 1 step が export できない（recon §1-2 — 記号軸 cat は**層あたり K/V の 2 本だけ**で、
  RoPE 実数対・先頭 10 head の再連結は静的軸なので現行語彙のまま通る）。
- ADR 0014 が静的に絞った理由は「記号長どうしの和は次元言語の一次式に一般には載らない」
  だった。しかし DiT の形は**記号 1 本 + 定数**（`S + 1519`）で、正準文法 `coeff·sym+offset`
  に**そのまま載る**。複数シンボルの和（`S+Tt+Ts+Tc`）は torch.export 自身も拒否する
  （recon §1-3 — 同じ制限が torch 側にもある）ため、受理を広げても表現不能な形が
  入り込む余地は無い。
- 実測（recon §1-5）: 緩和 2 行で 12 層 export → 書き出し → `verify_model` → TS の
  `bindSymbols` / `planGraph` まで緑。**拒んでいたのは宣言レベルのガードだけ**で、束縛後の
  shape 規則・strided 書きコピーのカーネル・レイアウト規則は既にこの形を正しく扱える。

## Decision

- `cat` の連結軸は、**全入力の当該 extent が〈定数〉または〈同一シンボルの一次式〉であり、
  その総和が `coeff·sym+offset` に収まるときに限り**受理する（`S`+定数 → `S+1519`、
  `S`+`S` → `2S`）。**異なるシンボルが混ざる連結は従来どおり fail loudly**。
- 変更は宣言レベルの 3 点 + 適合ケース表のみ:
  `convert._static_extent`（`_h_cat` の検査）/ `shapes._cat`（Python 側 shape 規則）/
  `runtime/plan.ts` の `assertStaticLayoutAxis`（TS 側宣言検査）+ `op-contracts.json` の
  shapes 節（両側）。`dim-grammar.json` は変更不要（`S+1519` は既に valid な文法）。
- **カーネル・レイアウト不変条件は無変更**。ADR 0014 の full-write 不変条件
  （出力軸長 = 入力軸長の総和）は束縛後の数値の世界で従来どおり成立する。
- `slice` / `flip` は**静的専業のまま**据え置く（要求実測が無い — ADR 0014 の規律を維持）。
- 同時修正必須: エクスポータの `convert.py`（range_constraints 走査）が**派生次元**
  （sympy の `Add` — 例 `S+1519` を入力 shape に宣言した場合）で `AttributeError` になる
  実装漏れを直す（`sympy.Symbol` 以外をキーから除外）。派生次元を入力 shape に持つ
  グラフは本 ADR で初めて実測に現れた。

## 検討した代替案

- **self/context の 2 分岐 + softmax の flash 型合成（案 b）**: 語彙を触らず今日通ることは
  実測済み（12 層 export 緑）。しかし attention の中心部に「1 対 1 書き換えでない独立実装」
  （手書きの amax/exp/sum 合成）が入り、ADR 0044 の `safe_softmax` 経路も使われなくなる。
  ノード数も +22%（1498 vs 1223）。緩和が裁定で通らない場合の代替として記録のみ。
- **S=750 固定（案 d）**: 10s 発話（S=250）で演算 3.1×。却下。
- **4 軸とも記号（案②）**: torch.export の次元言語が複数シンボル和を拒否 — 原理的に不可。

## Consequences

- Irodori DiT（G5）が torch の SDPA 分解そのまま + `safe_softmax` で export できる
  （実行形は ADR [0047](0047-irodori-dit-execution.md)）。
- **記号軸 cat は「KV キャッシュ追記」そのものの形** — Gemma 4 E2B（decode）で確実に
  再要求される。本緩和はその前払い。
- 実装は契約 2 実装（TS / Python）+ 適合ケース + 故障注入（異シンボル混在の拒否が
  実際に落ちること）を 1 セットで動かす。
