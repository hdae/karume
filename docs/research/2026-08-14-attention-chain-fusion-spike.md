# 2026-08-14 — 分解 attention 窓の鎖融合スパイク（K-5b 案 X・棄却の実測記録）

> 性格: **時点スナップショット**（2026-08-14・RTX 3080 Ti / Vulkan / Deno 2.9.4）。数値の正本は
> この doc、判断と順序は [perf-ledger](../perf-ledger.md) K-5b が持つ。

## 目的

K-5a（[ADR 0060](../decisions/0060-row-block-attention.md)）の行ブロック 4 dispatch 列
（bmm QKᵀ → add mask → safe_softmax → bmm PV）を、ADR 0023 の融合 3 カーネル構造
（bmm〈mask epilogue〉→ attention_stats〈emptyrow0〉→ attention_pv）へ置き換える案 X
（ビット同一・既定経路候補）の鎖時間比を、新 WGSL を書かず**既存カーネルの結線だけ**で先に買う。
事前合意の判定基準: S=750 形で鎖 ×1.3 未満なら kill。

## 方法

- カーネル面（bmm / elementwise add / safe_softmax / attention_stats / attention_pv）を src から
  直接 import して結線。1 反復 = 鎖 1 本を 1 compute pass 内に 12 回 encode（12 層相当）→
  submit → onSubmittedWorkDone の壁時計。q/kT/v/mask（**全 0.0**）と S バッファは両鎖で共有。
- キー: `bmm:v2:f32:reg128x128r8x8w16` / ew add / `safe_softmax:…:wg256` /
  `attention_stats:…:rc9` / `attention_pv:v1:f32:reg128x128r8x8w16`（v4 述語は両鎖とも false で一致）。
- ABBA（ref→fused→fused→ref）×2〜3 ラウンド。各ブロック = 捨て 5 + 計測 10 反復の中央値。
  ブロック間 1.5s アイドル・境界ごとに温度記録。
- 結線の正当性検査: mask=0 なら両鎖は ADR 0023 の等式で**ビット同一になるはず** — 最終出力 O の
  Uint32 完全一致で確認（S=750: 960,000 語 / S=170: 217,600 語とも mismatch 0・validation error なし）。

## 結果

| 形                                     | ref（4 dispatch） | fused（3 dispatch） | 比             |
| -------------------------------------- | ----------------- | ------------------- | -------------- |
| S=750 相当（M750 N2269 H20 D64）冷却後 | 41.13 ms          | 34.43 ms            | **×1.194**     |
| S=750 連続 run（再現）                 | 41.40 ms          | 34.52 ms            | **×1.199**     |
| S=170 相当（M170 N1689）参考           | 22.05 ms          | 23.16 ms            | ×0.952（逆転） |

- S=750 のブロック中央値範囲: ref 40.97〜41.31 / fused 34.36〜34.52（比の min 1.173 / max 1.229）
  — 1.3 との差はノイズ幅の外。温度 49→54°C（連続 run 52→59°C）でドリフトなし。
- 中間同時生存（結線からの導出値・実測でない）: 参照 ≈259.7 MiB（S サイズ 2 本）→ 融合
  ≈129.9 MiB（S 1 本 + stats 0.11 MiB）— 約半減。

## 判定 = kill（基準 ×1.3 に届かず）

- S=170（現行の主形）の逆転の主因見立て（切り分け未実施・kill 判定には影響しない）:
  attention_pv は行数幾何テーブルを通らず M128N128 固定（gemm-geometry.ts の MUST）に対し、
  参照鎖の bmm PV は M=170 で M64N32 を選ぶ — 占有率差。
- 事前見積り ×1.5 が外れた機序（見立て）: ①bmm 2 本（D=64 の細長形）が FLOP 見積りより遅く
  分母が大きい ②「S を 8 回なめる」の素朴パス勘定は過大 — safe_softmax の 3 回読みは 1 行
  9KB がキャッシュに乗るため DRAM 実トラフィックは 8 パスより小さい。節約 5 パスの理論最小
  0.75ms/層に対し実測節約 0.56ms/層で、帯域モデル自体は方向として正しいが上限が薄い。
- スパイクの fused 鎖は mask epilogue を含まない（実装案 X はここに mask[N] 読みが載る —
  僅かに不利側の差で、判定を覆す向きではない）。

## online（旧 K-5b 本命）の上限をこの実測から概算

融合が消したのは S トラフィックのうち 5 パス相当（実測 6.7ms/12 層）。online（S 完全非実体化）が
さらに消せるのは残余 3 パス相当のみ — 線形外挿で ref 比 **理想上限 ≈×1.35 前後**（S=750・
GEMM 効率が一切落ちない仮定・キャッシュ効果でさらに薄い可能性）。S=170 では帯域比率が下がり
かつ幾何の逆風もあり、これ未満。→ **irodori の形では速度層としての online に成立余地が薄い**。
NOTE: この結論は**この形状・この時点の 1 点実測**に基づく — 他モデル・LLM 形へ一般化しない
（過去の棄却記録の扱いと同じ規律）。

## 将来の flash/Sage 席への持ち越し

### emptyrow0 意味論の online 形スケッチ（ADR 0044 の再導出・紙のみ）

running (m, l, acc) の NaN 源は 2 箇所 — 補正係数 α = exp(m_old − m_new) と項
p_j = exp(s_j − m_new) がどちらも `−inf − (−inf)` を作りうる。ADR 0044 の select 流儀の
拡張で閉じる: `m' = select(m_new, 0.0, m_new == neg_inf)` を両者の減算項に使い
（非空行では m' = m_new のまま）、空行は l = 0 のまま走り切り、最終書き出しを
`select(acc·(1/l), 0.0, l == 0)`（または inv センチネル 0）で確定する —
「値を捨てる側で 0 を確定」「−inf ビットは params 渡し」「barrier 下の分岐は select のみ」は
据え置き。導入時は別キー + tolerance 全面再導出（ADR 0022 決定 3 / 0023 決定 2）+
ADR 0058 の席 3 点セットとセット。

### 隣接観測（report-only）

融合 attention 系カーネル（qk/stats/pv）は行数幾何テーブルを通らない MUST のため、M が
小さい形では素の bmm 経路より不利になりうる（本スパイク S=170 の逆転）。現行の attention op
利用者（anima は M≥1024・EG は linear 87% 支配）では実害の兆候なし — 需要が出たら幾何
パラメタ化を検討する。

## caveats

- マイクロベンチであり E2E ではない（窓外の op・ホスト enqueue・常駐化との重なりを含まない）。
  壁時計は submit→onSubmittedWorkDone で CPU 側 encode（12 encode/反復）を含む。
- mask 全 0.0 のため空行分岐は一度も立たず、S の値分布も実形と異なる。
- S=170 はブロック間ばらつきが大きく（生値 18.6〜26.7ms）、比 0.952 は「速くならない」ことの
  参考値で精度のある比ではない。
- 温度規約は厳密には +5°C を超えて開始（49°C・冷却直後 38°C）— ただし 8 ブロックの中央値が
  ref/fused とも ±0.2ms に収まりスロットリングは観測されず。
- 計測スクリプトはセッションのスクラッチパッド置き（リポジトリ未変更・非追跡）。
