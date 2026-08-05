# 0015 — conv 族の拡張: groups / dilation / conv_transpose1d / ゼロ bias 合成

- Status: accepted（2026-08-02）
- 根拠資料: [../research/2026-08-02-sbv2-chain-recon.md](../research/2026-08-02-sbv2-chain-recon.md)
- 需要の実測: sdp の DDSConv（depthwise groups=192・dilation 1/3/9）、dec の ResBlock1
  （dilation 1/3/5 × 30 本）・ConvTranspose1d × 5・bias 無し conv_post。

## 決定

- **conv1d の attrs に `groups` / `dilation` を追加**する。従来の「欄が無い = 1 固定
  （1 以外を黙って 1 で実行する経路が構造的に存在しない）」という設計価値は、
  「欄がある = 宣言必須・Object.hasOwn 全キー照合・不整合 fail loudly」で引き継ぐ
  （宣言済み attrs の既定値補完はしない — ADR 0012 の規律）。出力長式は
  dilation を含む一般形へ更新し、`Cin % groups == 0` / `Cout % groups == 0` を契約検査に
  入れる。TS 契約・Python 契約・WGSL カーネル・CPU 参照の **4 点セットを同一波で同時に**
  広げる（片側だけの先行を禁止）。
- **bias 無し conv はエクスポータのゼロ bias 合成でアリティ 3 に正規化**する。カーネル・
  契約に arity 分岐を持ち込まない（プロトタイプ実証済みの手筋）。
- **`conv_transpose1d` を新 op として追加**:
  - 重みレイアウトは **[Cin, Cout, K]**（conv1d の [Cout, Cin, K] と転置）。取り違えても
    要素数が合う形が作れ shape 検査を素通りするため、テストは**非対称チャネル数**で固定する。
  - 保存形は位置引数の末尾既定値が省略される（4 引数形が実在）。ハンドラは引数列を
    全長に None 埋めしてから既定補完する。
  - `stride >= 1` を契約検査で MUST（stride=0 はカーネルのループが進まず **GPU ハング** —
    例外にならない）。
  - 出力長 L·u は `pad = (k−u)//2` の成立とセットで契約検査する（SBV2 の設定はこれを満たす。
    満たさない一般形は fail loudly で見送り — 需要が出た時に広げる）。
- **`leaky_relu` は attrs `negative_slope` 必須の新 op**（本 ADR に含める理由: dec 専属で
  conv 族と同波で入るため）:
  - dec には slope 0.1（ups/ResBlock）と 0.01（最終段・位置引数省略の torch 既定）が
    **混在**する。既定値に頼ると片方が黙って誤るため、attrs は必須・既定値補完なし。
  - 分解抑止（保存リスト入り）。分解形（gt_scalar + mul + where）は中間バッファが
    1.5〜2 倍に膨らみ、メモリ見積の前提が崩れる。
  - WGSL は **select 形**で書く（`max(x, s·x)` は WGSL の max が NaN 伝播を保証しないため、
    torch の leaky_relu(NaN)=NaN と乖離する）。

## 語彙追加の残り（ADR 対象外）

where / clamp / ge・le・gt（Scalar・Tensor）/ bitwise_and / bool 入力 sum（→i32）/ log1p /
cumsum（最終次元）/ expand の f32 解禁は、ADR 0009/0012 が敷いた「op ごと実測ベースの契約表
拡張」の通常運用として扱う（個別 ADR を起こさない）。`eq(x, 0)` は
`bitwise_not(cast(x, bool))` への正規化で新 op を作らない（cast 規約「x→bool は x≠0」からの
帰結 — 実グラフでの検証を実装波の受け入れ条件に含める）。

## 検討した代替案

- depthwise 専用 op の分離: conv1d と契約・カーネルが二重化し、groups の一般形（SBV2 は
  depthwise のみだが契約は一般形で書ける）を塞ぐ。却下。
- conv_transpose1d を conv1d への書き換え（ゼロ挿入 + 通常 conv）で正規化: 中間テンソルが
  u 倍に膨らみ、dec のメモリ見積（素朴総和 6.97MB×Ty）を直撃する。却下。
