# 0070: shard ロードの 2 相契約とメモリ admission

- Status: accepted（2026-08-17 — 委任チェック方式・大域裁定なし。Codex レビュー
  第 3〜5 巡を反映〈完全性集合・co-shard・transaction 境界・明示 submit〉し第 6 巡で go）
- 関連: ADR [0038](0038-manifest-v1.md)（manifest と hub 取得層 — FileRef 3 点セットと
  fetch-cache 接続契約を継承）/ [0041](0041-manifest-v2.md) / [0063](0063-safetensors-physical-layout.md)
  （shard の欄の確定 = release 波の R1 と同席）/ [0004](0004-execution-model.md)（errorScope
  常設 — 決定 4 が同期区間を再設計）/ [0066](0066-generation-context-state-slots.md)
  （state 容量 = admission の新カテゴリ）/ [0069](0069-packed-w4-storage.md)（i4 格納 —
  配布サイズの前提）
- 根拠:
  [research/2026-08-17-autoregressive-references.md](../research/2026-08-17-autoregressive-references.md)
  §5（以下「調査 §n」）

## Context

現行は「全ファイルを `Uint8Array` で保持 → 単一 `ArrayBuffer` を `openModel` へ」の全量
ホスト保持で、**hub と runtime の両側に焼かれている**（fetch.ts:411-421 /
container.ts:91 — 調査 §5.1）。検収モデル級（i4 でも 1GiB 級・f16 なら 4GiB 級）では
「配布ファイル全量 + GPU 常駐」の RAM 二重持ちが成立しない。ブラウザ実装の先行例は
tvmjs の 2 相ロードのみで、**重みの整合性検証はブラウザ勢の誰もやっていない**
（karume の sha256 全数照合が差分価値 — 調査 §5.1）。

## Decision

### 1. 配布形: 重みの shard 化（欄の確定は R1 と同席・ローダ契約はここで固定）

- 1 コンポーネントの safetensors を**グラフ shard（`karume_ir` JSON + 小テンソル）+
  重み shard 群**に分割できる形にする。各 shard は独立に整合な safetensors
  （ヘッダにテンソル名を持つ）で、**振り分け表は manifest が正本**（HF の
  `model.safetensors.index.json` 形式は採らない — karume.json が既にファイル表を持つ）。
- 各 shard は従来どおり `{path, size, sha256}` の 3 点セット（ADR 0038 決定 2 の検証規則
  そのまま）。manifest の欄名・混成 dtype の席は **R1（ADR 0041/0063 reopen・HF 公開前
  締切）で確定**し、本 ADR は「shard 列であること + ローダ側契約」を先に固定する。
- **宣言完全性は全 shard 読了後に検査**: 突合する集合は **`initializer.tensor` と
  `storage.scale` が指す名前の和集合**（第 3 巡で補正 — scale companion は
  `graph.initializers` の要素ではないため、initializer 集合だけを正本にすると i8 / i4
  資産の scale が全て「余剰」になる。現行 container 検査〈container.ts:137-160〉と同一
  集合の shard 横断版）。欠け・重複・余剰いずれも fail loudly。
- **weight と companion scale は同一 shard に置く MUST**（co-shard — 第 3 巡指摘の閉鎖）:
  逐次消費（決定 3）は weight と scale を同時に必要とするため、shard を跨ぐと「参照を
  手放す」契約と両立しない。エクスポータの shard 分割規則に載せ、リーダは違反を
  fail loudly（RAM O(最大 shard) の保証条件）。

### 2. hub: 2 相ロードと逐次引き渡し面

tvmjs 型の 2 相（調査 §5.1）を fetch-cache 接続契約（ADR 0038 決定 5）の上に載せる:

- **相 1**: 全 shard を fetch-cache（永続キャッシュ）へ落とす（並列可・RAM に載せない）。
- **相 2**: shard を**逐次** 1 本ずつ「キャッシュから取得 → **sha256 照合**（キャッシュ
  ヒット側も走らせる — 現行 validate フックの維持。**非交渉条件**）→ 呼び手へ渡す →
  参照を手放す」。
- 公開面: 既存の全量 Record 面は**温存**し（小モデルは従来どおり）、**shard 逐次面**
  （`AsyncIterable` 型 — shard 名 + bytes）を追加する。RAM ピーク目標 = O(最大 shard)。

### 3. runtime: shard 消費の Session 構築面

`createSession` に shard 逐次面を受ける形を追加する（既存の単一 buffer 面は温存）。
グラフ shard を最初に受けて**構造契約・適格判定・重み受け入れ準備**を確定し（第 3 巡で
精密化 — PreparedPlan は従来どおり初回 run で確定する。計画は実行時の SymbolBindings を
要するため、グラフ shard 時点で確定できるのは graph 単体で決まる検査まで）、重み shard は
**届いた順に「CPU 展開（適格外のみ）→ GPU upload → 解放」**する。転送完了前に CPU 側を
解放しない（フェンス後解放 — tvmjs の `await device.sync()` → `dispose()` と同じ順序契約・
調査 §5.1）。**フェンスの前に shard ごとの明示 submit を挟む MUST**（第 4 巡）:
writeBuffer だけの区間は pending dispatch が無く、submit しないと実装の staging が
解放されない（Session 構築後の「submit 1 回が瞬間ピーク +2.7GiB を抑えている」と同根の
既知要件 — これを落とすと RAM ピーク O(最大 shard) の目標を実装が満たさない）。

**失敗の transaction 境界**（第 3 巡指摘の閉鎖）: 途中の shard で失敗した場合（sha 不一致・
宣言違反・GPU エラー）、構築済みの GPU 資源（アップロード済み重み・weights アリーナ）を
**全て破棄して部分 Session を公開しない** — 現行の一括構築が例外時に weights アリーナを
destroy するのと同じ境界を shard ビルダが持つ（pending discard・CPU 参照解放を含む。
shard 単位 errorScope〈決定 4〉は検出器であって後始末はこの境界が持つ）。

### 4. アップロードの errorScope 同期区間を shard 単位へ再設計

現行の「重みアップロードループ内 await 禁止」（executor.ts:512-517 — errorScope LIFO の
交錯防止が根拠）は shard 逐次化と両立しない。**push / pop を shard 単位の同期区間に
張り直す**: 1 shard ぶんの writeBuffer 列を 1 つの errorScope 区間（同期）で囲み、
区間の外で await（次 shard の取得・フェンス）する。LIFO 交錯の不変条件は「区間内に
await が無い」ことで従来どおり保たれ、区間が短くなるだけ（安全ガードの撤去ではなく
粒度の変更 — 失敗 shard の特定が細かくなる副次利得）。

### 5. admission: 必要側 estimator + 診断（絶対保証にしない）

- **estimator は「必要側」のカテゴリ別合計のみ**を出す: resident weights（圧縮 + 展開）/
  state スロット（ADR 0066 の容量）/ prepared backing / transients（計画から導出）/
  staging。**空き側との比較はしない** — WebGPU は総 / 空き VRAM を露出しない
  （調査 §5.2。「予算が取れない環境で当て推量しない」は llama.cpp の 0/0 デバイス除外と
  同型の前例）。
- 診断に **unaccounted 相当の欄**を持たせる（llama.cpp fit の出力形 — 「見積りが絶対保証で
  ない」ことを形式が認める・調査 §5.2）。
- **判定の最終門は既存の out-of-memory errorScope**（gpu/device.ts の 2 本組 — 新設なし）。
  estimator は `createGenerationContext` / Session 構築の**事前診断**であり、超えていても
  実行は止めない（警告 + 診断）。
- 公開面は薄く（ADR 0008）: 見積り関数 1 本 + `SessionDiagnostics` への欄追加まで。

### 6. 席（明示予約・実装先送り）

- ストリーミング fake 展開（i4 → f32 の CPU 展開を shard 単位で行う適格外経路）は
  決定 3 に内包（追加設計不要）。
- 大型 DL 前の limits preflight（backlog release 項）は本 ADR の estimator を土台に
  release 波で。hub Range 並列 + prefetch（perf L-3・parked）は相 1 の内側の最適化として
  席が残る（契約は不変）。

## Consequences

- hub / runtime の公開面がそれぞれ 1 面増える（既存面は不変 — 既存利用コード・既存門は
  無風）。全量面と shard 面で**同一資産 → 同一 GPU 常駐バイト列**が受入条件（A/B 門）。
- manifest v2 の shard 欄（R1）が入るまで、shard 面の消費者はローカル実験に限られる
  （HF 公開資産は単一ファイルのまま）。R1 確定が HF 公開前 MUST である理由は不変
  （hub は 2 形パースをしない — ADR 0041）。
- 受入条件: ①全量面との GPU 常駐バイト A/B 一致 ②sha256 照合がキャッシュヒット側でも
  走る門（既存門の shard 版）③RAM ピーク実測（全量比で O(最大 shard) に落ちること）
  ④errorScope 区間の再設計後も無効バッファ / 整列違反の注入が検出されること（0004 門の
  shard 版）⑤宣言完全性の shard 横断検査（欠け・余剰の注入）。

## 追記

- 2026-08-19（波 G 実装 — 決定 1〜5 の具現で確定した面）:
  - **hub（決定 2）**: `streamAssets(loaded, refs: FileRef[], options)` —
    `AsyncGenerator<{path, bytes}>`。yield は入力順・空 / 重複 path は network に出る前に拒否。
    相 1 は fetch-cache 0.4 の streaming prefetch（sha256 通過中照合 — 不一致はエントリ不成立）で、
    `caches` 不在・キャッシュ書込み失敗は fail loud（バイト列を手元に持たない面に素 fetch 縮退の
    余地は無い。`onCacheError` 診断が届くのは相 2 のみ）。相 2 の sha256 照合は従来の validate
    フック経由で**キャッシュヒットにも走る**（非交渉条件の維持 — prefetch が焼く検証済み
    マーカーは読まない。将来 opt-down する場合の席としてだけ残る）。
  - **runtime（決定 3）**: `createSessionFromShards(gpu, shards: AsyncIterable<Uint8Array>,
    options)`。最初の shard = グラフ shard（`karume_ir` 必須）・後続への `karume_ir` 再登場は
    fail loudly・bytes は buffer 全体を占める view MUST。**全量面 `createSession` は「1 shard の
    列」として同一経路で構築**する — 受入①は 2 面が経路を共有することの帰結で、A/B 門
    （`gpu_shard_session_test.ts`）は分割粒度の差だけを検査する。エラーの帰属ラベルは全量面 =
    従来どおり「重みのアップロード」・shard 面 =「shard [n] の重みアップロード」。SessionState は
    graph のみを保持し file を掴まない（全量面でも構築後は配布バッファを固定しない）。
  - **format（決定 1）**: 宣言完全性（欠け全件列挙）・余剰・shard 横断重複・co-shard 違反は
    container.ts の shard 進行検証（`createShardValidator`）が持ち、`openModel` も同じ経路に
    一本化した（検査の二重実装なし）。
  - **admission（決定 5）**: `estimateSessionMemory(model, {bindings?, generation?})` —
    GPU 非依存の純関数。カテゴリ写像: resident weights（圧縮 + 展開）→
    `compressedWeightBytes`（↔ 診断 `storage.residentCompressedBytes` と厳密一致）/
    `uncompressedWeightBytes`（f32 / i32 — 診断対象外なので別欄）/ `expandedWeightBytes`
    （↔ `storage.hostExpandedBytes`）。state スロット → `stateBytes`（↔
    `stateBacking.residentBytes`）。staging → `ioBytes`。prepared backing / transients →
    `transientBytes` 1 欄に統合（同じ slot 表の必要側。融合前ノード列の生存区間
    シミュレーション = **近似**）。unaccounted 欄は見積り出力側に持つ。
    **`SessionDiagnostics` への欄追加は不要と裁定** — 実測側の対応欄（`storage.*` /
    `stateBacking` / `planBacking` / `lastRun.peakTransientBytes`）が既に完備で、決定 5 の
    「〜まで」は上限であって義務ではない（対応表の正本は estimate.ts の docstring）。
  - **受入④の実装形**: 実 GPU で validation を人工発火させる注入口が無い（巨大確保が要る）ため、
    shard 版の門は「validator 注入群（欠け・余剰・重複・co-shard・整列は safetensors パーサ門）+
    途中失敗後のスコープ残高検査 + 同一 device での再構築成功 + 既存 poppingDevice 単体門」の
    高度で張った（現行の重みアップロード 0004 門と同じ高度）。
  - **R1 への送り**: exporter 側の shard 分割規則（co-shard を吐く側の保証）と manifest の
    shard 欄は R1 のまま。shard 面の消費者がローカル実験に限られる状況も不変。
  - **Codex 独立チェックの消化（同日）**: ①相 2 の中断は yield 直前でも観測する（最終 shard の
    検証中の中断が正常完了に化けない）②グラフ shard の参照は最初のフェンス後に**構造で**
    手放す（受け渡し箱 + ブロックスコープ — 実測ピークは不変で、エンジンのフレームスロット
    解放頼みを保証に置換した）③`transientBytes` は生存ピークでなく **exact-size LIFO 再利用を
    再生した slot 総バイト**（生存ピーク代用はサイズが揃わない列で系統的に過小 — 8→12→4 の
    3 段で 20 vs 24）④estimator は `chunkLength` 値域と「states と入力の共有記号の一致」を
    実構築と同じ門で検査する（作れない構成に見積りを返さない）。report-only: 0 要素重みの
    4 バイト床が診断と見積りで食い違う縮退角（波 F の I4-DIAG-03 と同族 — 診断側の意味変更を
    要するため裁定待ち）。
