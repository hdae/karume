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

> **改訂（2026-08-30 — ADR [0081](0081-shard-spec-v2.md)）**: 書き手側の分割規則（下の
> 2026-08-29 追記「逐次詰め + 尾部スラック」）は ADR 0081 が置換した。現行は
> **グラフ shard = `karume_ir` だけ・データ節 0 テンソル**、上限は `SHARD_BYTE_LIMIT`
> （1GiB）1 本、常時分割（単一ファイル配布形は廃止）。この節の残り（宣言完全性・co-shard・
> 振り分け表は manifest が正本）はそのまま生きている。

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

- 2026-08-29（R1 統合波 — 同席裁定 4 件〈CX-4.1/4.2/4.3/3.2〉+ 分割規則 + models 接続の実装で
  確定した面。裁定の原文 = `.claude/reviews/2026-08-19_b04f589/`）:
  - **shard identity（CX-4.1）**: `createSessionFromShards(gpu, shards: AsyncIterable<ModelShard>,
    options)` — `ModelShard = {id, bytes}`（hub `StreamedAsset` の `path` は `id` へ改名・構造
    互換）。shard 由来の失敗（parse・宣言違反・co-shard・アップロード）は `shard [n] 'id'` を
    名乗る（連番は補助）。帰属はエラークラスと stack を保つ message 前置で、全量面の文言は
    不変（受入①の契約維持）。追記 2026-08-19 の「shard [n] の重みアップロード」記述は本追記が
    置換する。
  - **2 段境界（CX-4.2・決定 5 の graph-first を公開面へ）**: `prepareModel(graphShard) →
    PreparedModel.estimate() → PreparedModel.createSession(gpu, weightShards)`。capability 門と
    IR 契約検査・常駐計画（`planWeightResidency` — CX-3.2 の純関数プランナ）は prepare 相で
    確定し、重み shard を 1 バイトも取得する前に「実行できない」が落ちる。既存 3 面
    （`createSession` / `createSessionFromShards` / `estimateSessionMemory`）は全て
    PreparedModel 経由の薄い合成へ一本化（全量面 = 全テンソル同居のグラフ shard 1 本 + 重み
    0 本の列）。**代償**: PreparedModel はグラフ shard を寿命いっぱい保持する。
    **訂正（2026-08-30）**: ここに書いていた「決定 3 がグラフ shard を『karume_ir + 小テンソル』
    と規定するので RAM ピーク目標『O(最大**重み** shard)』は崩れない」は**誤り**だった —
    同じ追記で確定した分割規則（`karume.shards` の規則 3/4）は先頭 shard にも実重みを
    データ節 1GiB まで詰め、1.5GiB 以下の資産は分割ゼロ（= 全量が単一のグラフ shard）になる。
    したがってグラフ shard は最大 1GiB の実重みを含みうる。保持する側の対処は下の
    2026-08-30 追記（models 側が握らない）を参照。
  - **estimator（CX-4.3）**: 返り値は `AdmissionReport`（`resident`〈重み内訳 + state〉+
    `scenarios[]`〈generation 指定時は prefill / decode を chunk 記号の再束縛で独立計算〉+
    `peakAccountedBytes` = resident + max(シナリオ)）。max の根拠 = `ActiveBacking` 同時 1 本。
    切替窓（退役 backing が flush 後始末まで生きる）は unaccounted へ明文化。追記 2026-08-19 の
    カテゴリ写像は欄名だけ読み替え（導出式は不変・診断との厳密一致門は新欄名で維持）。
  - **hub `prefetchAssets`（相 1 単体の公開面）+ models 接続**: 7 pipelines の `fromPretrained`
    は「全コンポーネントのグラフ shard 1 回取得 → admission → 重み shard を 1 回で prefetch →
    Session 構築時にキャッシュから逐次流し」。進捗はモデル全体 1 本のストリームを維持
    （models 側で集約）。デモのローカル読みは疑似 HF サーバ経由で同一経路
    （`examples/shared/local-dist-server.ts`）。
  - **exporter の分割規則（決定 1 の書き手側 — ADR 0071 決定 4 の解除）**: 正本 =
    `karume.shards`。データ節 1GiB（`SHARD_BYTE_LIMIT` 固定・ヘッダ非計上）・書き出し順の
    逐次詰め・weight/scale 原子対・連番名 `<stem>-NNNNN-of-NNNNN<suffix>`・上限以下は単一
    ファイルでバイト不変（git archive 対照 sha256 で実証）。**尾部スラック（2026-08-29
    ユーザー裁定）**: 未閉バイト（現 shard の used + 残量）が `SHARD_TAIL_LIMIT`（1.5GiB =
    hub の同時 RAM 予算と同値）以下ならカットせず詰め切る — 端数 shard を作らず、1.5GiB
    以下の資産は分割ゼロ（例: 3.2GiB → 1+1+1.2）。
    **超越（2026-08-30 — ADR [0081](0081-shard-spec-v2.md)）**: この段落の規則（グラフ shard に
    実重みを詰める・尾部スラック・単一ファイルのバイト不変）は shard 仕様 v2 が置換した。
    連番名と weight/scale 原子対と 1GiB の上限値は引き継ぐ。
  - **受入の実測（2026-08-29）**: anima Base f16 transformer 3,913,609,588B → 4 shard
    （最大ファイル 1,073,756,928B = データ節 1GiB + ヘッダ）で dist 全門通過・実ロード +
    512² 生成完走（従来は Chromium の単一 ArrayBuffer 上限で原理的に不能 — limitations 追随）。

- 2026-08-30（レビュー M1-2 の是正 — **models 側はグラフ shard を握らない**。裁定の原文 =
  `.claude/reviews/2026-08-29_9614ba9/` の M1-2）:
  - **食い違いの実測**: `AnimaPipeline.fromPretrained`（anima Base f16・4 コンポーネント）直後の
    常駐を Deno で測ると `memoryUsage().external` = **2,461.6MiB**（rss 差分 +2,738MiB）で、
    内訳は 4 本のグラフ shard のファイルサイズ合計 2,460.9MiB（text_encoder 1,138.9 +
    text_conditioner 257.3 + transformer shard[0] 1,016.3 + vae_decoder 48.4）と一致した。
    CX-4.2 が前提にしていた「グラフ shard = karume_ir + 小テンソル」は成立しておらず、
    RAM ピーク目標「O(最大**重み** shard)」は**この経路では崩れていた**。
  - **決定**: `loadShardComponents`（models）は admission（`prepareModel` の capability 門 +
    契約検査）を通したら `PreparedModel` を**その場で捨て**、残すのは `IrGraph`（JSON 由来の
    純データ）だけにする。`ModelComponent.createSession` は shard 列**全部**（先頭のグラフ
    shard を含む）をキャッシュから流し直す `createSessionFromShards` へ載せ替える。
    runtime / hub / 配布形 / pin はいずれも無改変（`PreparedModel` がグラフ shard を保持する
    こと自体は Session を張り直せる面として正しい — 握る側が models だったのが誤り）。
  - **実測（同条件・是正後）**: `external` = **0.7MiB**（rss 差分 +210〜226MiB）。代償は
    Session 構築のたびにグラフ shard 2.46GiB をキャッシュから読み直して再 parse する時間で、
    512²・2 step の `generate` 実測は 5,993 / 6,045ms → 7,773 / 9,756 / 9,966 / 12,584ms
    （ローカルのディスク実体キャッシュ・OS ページキャッシュ冷。読み直し量 2,716MiB → 5,176MiB）。
    NOTE（2026-08-30 訂正）: この代償は **shard 仕様 v2（グラフ別居）では直らない** — Session
    構築が読み直すのは重み全量で、グラフをどの容器に置いても総読み量は変わらない。緩和の
    正しい席は Session 常駐（perf-ledger H-4 / L-9）か opt-in の RAM 保持。
  - **未是正として残るもの**: 分割規則そのもの（グラフ shard に実重みを詰める規則 3/4）は
    そのまま。全配布形の再 dist + HF 再アップロード + pin 更新 + 越境参照の焼き直しを伴うので、
    リリース波の判断としてユーザー裁定に上げる（レビュー M1-2 の c 案）。
    **決着（2026-08-30）**: c 案が裁定され、ADR [0081](0081-shard-spec-v2.md)（shard 仕様 v2 —
    グラフ専用 shard・上限 1 本・均し詰め）として実施した。再 dist と再アップロードは次波。

- 2026-08-31（取得元抽象 — ADR [0086](0086-distribution-source.md)）: **決定 2 の相 1 は取得元の
  optional 能力**として読み替える。`prefetchFile` を持たない取得元（手元のディレクトリ）は正当で、
  その場合の逐次面は**相 2（直接逐次読み）だけで同じ RAM ピーク目標 O(最大 shard)** を満たす —
  相 1 は「HTTP + 永続キャッシュ」に固有の最適化であって、直接読める取得元には温める対象が無い
  （写す形はディスクを 2 倍使って読みを 1 段増やすだけになる）。共通層は `prefetchFile ===
  undefined` を見て何もせず抜けるので、`prefetchAssets` はローカル取得元では no-op である。逆に
  **越境先だけが相 1 を落とす形は取得元契約の破れ**として fail loudly する（`originFor` は同じ
  取得元の別座標を返す口であって、能力を落とす口ではない）。

### 追記 2026-09-02 — 器の使い回し（ホスト RAM ピークの係数 1 化）と shard 受け口の契約変更

メモリ管理波 Phase B の実測（[research/2026-09-02-shard-size-ram-peak.md](../research/2026-09-02-shard-size-ram-peak.md)）
で、逐次面の RAM ピークは「O(最大 shard)」ではあっても**係数 3**（今の shard + GC 待ちの前 shard +
GC しても消えない 1 本）で、定数も 1GB 級だった。shard ごとに新しい `ArrayBuffer` を作る限り
係数は下がらない（明示 GC で 1 本ぶんしか減らない・writeBuffer の完了待ちを刻んでも Vulkan では
不変）。

**決定（Phase C-1）**: 逐次面は**コンポーネントの最大 shard 長の器を 1 本だけ確保して使い回す**。

- hub: `FileReadOptions.into`（器の貸し出し・遅延確保）を逐次面 `streamAssets` が渡し、取得元は
  器へ読んで prefix view（byteOffset 0 / byteLength = `size`）を返してよい。`DirectoryAdapter` に
  `readFileInto`（任意）を追加し、Deno のディレクトリ取得元が実装。器を使わない取得元（HF —
  取得層 `@hdae/fetch-cache` がキャッシュ読出しで buffer を確保する）は従来どおり tight view を
  返し、器は確保されない。
- runtime: `ModelShard.bytes` の契約を「buffer 全体を占める tight view MUST」から「**buffer の先頭
  からの view MUST**（tight view か器の prefix view・byteOffset ≠ 0 は拒否）」へ。供給側の義務は
  「次の `next()` まで器を書き換えない」。`parseSafetensors` はファイル長を別に受け、末尾未使用領域の
  検査をその長さで行う。
- 決定 2 の RAM ピーク目標「O(最大 shard)」は**係数 1**へ具体化: 実測 anima f16（1GiB shard）
  4,069 → 1,402 MiB・gemma4 i4 2,622 → 1,116 MiB・ロード時間も 11.2 → 5.7 s（確保と GC の往復が
  消えた分）。
- 外に残るもの: HF 経路の係数（取得層へ `into` 相当を足すまで従来どおり）／テンソル単位の
  ストリーミング（Phase C-2 候補 — ピークを「定数 + 最大テンソル」へ）／書き手の shard 目標値
  （512 or 256MiB・ADR 0081 側の裁定）。
- 追記（2026-09-02・ADR [0090](0090-shard-spec-v3-tensor-pieces.md)）: テンソル分割（piece）を
  実装。決定 1 の co-shard は「scale は piece 1 と同じ shard」へ拡張、上限は 256MiB 1 本（ファイル長）
  へ。書き手の目標値の裁定はこれで閉じた。
- 追記（2026-09-02 夜・取得層 `@hdae/fetch-cache` 0.6.0 `into`）: HF 取得元も器を使う。取得層に
  「呼び出し側のバッファへ読む」口が入り（向こうの ADR 0009 — network 受信もキャッシュヒットの
  読出しも器の先頭へ・容量不足は fail loud・single-flight の合流者へ器は渡らない）、`hf.ts` の
  `readFile` が `into` を `HfFileSpec.into` へ配線する。「外に残るもの」の HF 経路の係数はこれで
  閉じた（実測: gemma4 warm 1,408 → 684 MiB・anima f16 warm 2,242 → 743 MiB —
  [research 結果 8](../research/2026-09-02-shard-size-ram-peak.md)）。組み込みの 2 取得元は
  どちらも器を使い、器を使わないのは外部実装の取得元だけになった。
