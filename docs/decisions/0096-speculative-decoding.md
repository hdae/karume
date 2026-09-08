# 0096: Gemma 4 MTP — 投機的デコードの全体像（perf-ledger K-20 ④）

- Status: accepted（2026-09-08 — ユーザー裁定: 設計軸 1 = B〈Session 跨ぎの読み専用スロット〉・
  設計軸 2 = S〈共有 initializer〉・単軸 7 点・段 1 → 4 の順）
- Date: 2026-09-08
- 関連: ADR [0066](0066-generation-context-state-slots.md)（決定 6 = 論理長は run の成功で進む —
  本 ADR 決定 3 で deferred commit を足す / 追記 2 = sliding を含む context の rewind 拒否 —
  不変・投機は rewind を使わない）/ [0067](0067-autoregressive-attention-vocabulary.md)（決定 4 =
  読み書き同式の `slot_row`・window ≤ capacity — 本 ADR 決定 4 で capacity を window + 余裕に /
  決定 5 = append はスロットにつきちょうど 1 本 — 本 ADR 決定 1 で「読むだけの外部スロット」を
  第 3 種として足す）/ [0068](0068-decode-exit-multi-output.md) 追記 6・[0083](0083-generation-api-surface.md)
  決定 6（出口 = 最終行 logits `[1,1,V]` — 本 ADR 決定 5 で `[1,R,V]` + hidden へ一般化）/
  [0069](0069-packed-w4-storage.md)・[0019](0019-i8-weight-execution.md)（格納は initializer の席 — 本 ADR 決定 2 の共有
  initializer は「バイトの出所」だけを変える）/ [0085](0085-ple-host-gather.md)（PLE のホスト gather —
  本 ADR が「1 cycle = run 2 本」に落ちる直接の理由）/ [0095](0095-plan-backing-budget.md)（backing 予算 —
  verify 形の backing はここに乗る）
- 実測: [research 2026-09-08](../research/2026-09-08-mtp-ea-i4-target.md)（i4 target の受理数 E[a]・
  採算表）/ [research 2026-09-07 §7](../research/2026-09-07-codex-perf-review-followup.md)（F / T(M) / D）

## Context

Gemma 4 の公式 MTP drafter（`google/gemma-4-E2B-it-assistant`）は 4 層・hidden 256 の小さな
Transformer で、**自前の KV を持たず** target の 2 層（sliding 側の最後の非共有層 13・full 側の 14）の
K/V へ cross-attention し、入力は `concat(target_embed(token)·√1536, target の最終 norm 後 hidden)`、
出力は次 token の logits と次段の hidden。draft k 段は逐次で、verify は target に `[bonus, d₁..dₖ]` の
k+1 行を 1 回流し、先頭一致で受理数 a を決めて `a+1` token を確定する（greedy なら出力は非投機と
同一）。

karume 側の実測（復活条件 ①〜③・2026-09-07〜08）: 小 M の linear は GEMV 行ブロックで M=1 並み
（K-21）、形の切替で backing を作り直す費用は予算つき保持で消えた（H-15）、i4 と同値の重みの target で
受理数 E[a] は抽出的な長文脈で k=3 2.3〜2.6・k=6 3.8〜5.0（E-4）。更新した予測倍率は抽出的長文脈で
Deno 1.8〜2.6× / ブラウザ相当 1.9〜2.4×、自由文 0.9〜1.0×（k ≤ 3）。

karume の現行契約で投機が踏む壁（recon 2026-09-08・HEAD `4b2f6ba`）:

- state スロットは context 所有で context は Session に紐づき、別 Session の run は所属検査で落ちる。
  読者だけのスロットは `validateGraphContracts` が拒否（append ちょうど 1 本・append が最後のノード）。
- 論理長は run の成功で `queryLength` ぶん必ず進む（部分 commit の口が無い）。sliding ring は window
  ちょうどで閉じるので、論理長より先に書いた行（棄却行）が 2 行以上あると live 窓の最古行を潰す。
- 出口は最終行 logits 1 本（`last_row [1]`）。drafter が要る最終 hidden の出口が無い。
- 量子化格納は initializer の席にしか無く、Session 跨ぎで重みの GPU バッファを共有する口が無い。
  drafter は段 1..k−1 で自分の出した token の埋め込みを引くので、表は drafter のグラフから引ける
  場所に要る。
- PLE はホスト gather（ADR 0085）なので、draft token がグラフの中で決まる形では PLE 行を先に引けない
  → **1 cycle = draft run + verify run の 2 本**が下限（1 run に畳む案は不成立）。

## Decision

### 1. drafter は別 Session で、target の KV スロットを**読むだけ**する（設計軸 1 = B）

IR のスロット宣言に第 3 種 **external**（読者はあるが append を持たない・実体は他の context が所有）を
足し、drafter の Session は `createSession` の options で target の context を束ねる（`sharedStates`）。
executor の所属検査と bind group の焼き込みを「所有 context または束ねた context」まで開く。
MUST: 寿命は **drafter Session ⊂ target context**（target の context / Session を畳む前に drafter を畳む・
target の poison は drafter の run を拒否させる）。MUST: external スロットへの書き込みは構造で禁止
（append を持てない）— target の論理長は target の run / commit だけが進める。

採らなかった案: **C′ 鏡像スロット**（target が層 13 / 14 の K/V 行を出力し、drafter 自身の context の
スロットへホストが書く）は「append 無しスロット」の契約改訂が同じく要り、加えてスロット書き込み API・
target の出力 4 本（pin・融合外し）・KV の二重持ち・毎 cycle のコピーが乗る。**A 同一グラフ 1 run** は
PLE のホスト gather と greedy 受理のグラフ内化が要り不成立。

### 2. drafter の入力埋め込み表は target の常駐重みを **Session 跨ぎで共有**する（設計軸 2 = S）

drafter のコンテナは埋め込み表の initializer を「共有・バイト無し」で宣言し、`createSession` の
`sharedWeights` で target の `ResidentWeight`（i8 + scale の同一 GPUBuffer・同じ格納）を束ねる。
配布 +0 MB・VRAM +0。格納の席は initializer のまま（ADR 0019 / 0069 不変）— 変わるのは**バイトの出所**
だけ。drafter 自身の出力ヘッド表（`[262144, 256]`）は drafter のコンテナに焼く。

採らなかった案: 複製（i4 でも +201 MB の配布と VRAM）・段ごとに run を分けてホストが行を引く形
（フェンスが cycle あたり k+1 本）。

### 3. 部分 commit = **deferred commit**

`GenerationRun.commit: "immediate" | "deferred"`（既定 immediate = 従来）。deferred の run は成功しても
論理長を進めず `pending = {pastLength, queryLength}` を context に保留し、ホストが readback で受理数を
決めてから `context.commit(rows)`（`0 ≤ rows ≤ queryLength`）で進める。pending が残る間は新しい run と
rewind を拒否（dispose は可）。失敗は従来どおり poison（追記 3 不変）。
ADR 0066 決定 6 の「論理長は run の成功で進む」は「immediate は run の成功で・deferred は成功後の
commit で」に改める — 二重簿記（ホストと context の別々の加算）は依然として無い。
採らなかった案: cycle 単位の別リース（rewind / dispose / poison の遮断面を全部引き直す）。

### 4. sliding ring に余裕を持たせる: capacity = window + kmax（kmax = 8）

`slot_row(col) = col % capacity`（読み書き同式は不変・`column_base` / `live_columns` / `in_window` は
window 基準のまま）。棄却行 i が潰す論理列は `P+i−capacity` で、commit 後の live 窓 `[P+a+2−W, …)` の
外に常に落ちる（`k ≤ kmax`）。state-append の重複排除ガード（Q > capacity の prefill で最後の論理行だけが
書く）も capacity 基準へ。exporter の `states_plan` が sliding スロットの capacity を `window + 8` で
焼く（配布形の焼き直し・未リリースなので可）。VRAM 増は sliding スロット 24 本 × 8 行 × 1 KiB ≈ 192 KiB。
MUST: 投機ループは `k ≤ slidingSlack`（context が公開する capacity − window の最小）を守る。

### 5. verify の出口 = logits `[1,R,V]` + 最終 norm 後 hidden `[1,R,H]` の 2 出力

`last_row` を第 2 記号 R の `[R]` にし、R=1 が従来の prefill / decode（値・token 列はビット同一）。
R を M と共用しない（prefill が `[1,768,V]` を readback してしまう）。readback は R MiB（R=9 で 9 MiB）—
GPU 側 argmax（ADR 0083 決定 6 の改訂）は実測してから。hidden は drafter の入力（row a を選ぶ）。

### 6. verify の M と k

`chunkBuckets` の既定に 4 / 8 を足し（k+1 ≤ 8 → M ∈ {4, 8}）、`PREPARED_PLAN_CAPACITY` を 8 → 12。
M=16 へ pad する案は K-21 後に逆転（T(8) 37.2 < T(16) 38.6 ms）。k は**まず固定 3**（E-4: 抽出的長文脈
1.8〜2.1×・自由文 0.9〜1.0×）— 動的 k（複数 drafter グラフ）は段 4。

### 7. 受理規則は greedy 先行・ホスト側

`src/generation/` に置き、`argmax` を export して tie 規則（最小 id）を 1 実装に。repetition penalty /
logit bias は行を進めながら history を伸ばして行ごとに適用。温度 > 0（受理確率 `min(1, p/q)` と残差
分布）は段 4 以降。

### 8. 配布 = 同じ ModelEntry の第 2 role `drafter`（既定は DL しない）

`speculative` を指定したときだけ drafter shard を取得して Session を張る。drafter の出力ヘッドは素の
lm_head + argmax（centroid 疎 softmax の topk は exporter に無い — 受理数は同一〈E-1〉・帯域の最適化は段 4）。

## 段階と検証

| 段 | 内容                                                                                                                                                                    | 門                                                                                                                    |
| -- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 1  | ring 余裕 / R 行出口 + hidden / deferred commit / バケット 4・8 + LRU 12 / gemma4 再 export                                                                             | R=1 の golden・sha 門が緑のまま・R ∈ {4, 8} の logits が `io.*` の行と u32 一致・棄却行を含む ring の故障注入         |
| 2  | external スロット + 共有 initializer の IR 宣言・Session 跨ぎの束ね・読み専用 attention カーネル・drafter recipe・role `drafter`                                        | drafter の出力 token を HF の drafter（fp32）と突合（argmax 一致率 ≥ 99%）                                            |
| 3  | `Gemma4Pipeline` の `speculative`・sequence の投機経路（frontier / 位置 / 予算 / 停止 token / 容量末尾 / abort）・診断 phase draft / verify・`GenerationStop` の run 数 | 投機ありの token 列 = 非投機 greedy（強制棄却の故障注入でも同一）・停止 token が draft 途中・容量末尾・512 超えの棄却 |
| 4  | 実測（Deno / Metal）・動的 k・GPU argmax・fence 削減・小 M attention・topk                                                                                              | 採算表の更新                                                                                                          |

## 帰結・残件

- 1 cycle のフェンスは 2 本（Deno で 23 ms 固定費・ブラウザは 1 ms）。generation run の enqueue と
  `copyOutputs` で 1 本へ畳むのは段 4（PLE のホスト gather がある限り draft token はホストを経由する）。
- M ∈ [2, 16) の attention は「①′ も ①ₜ も効かない帯」（T(4)=27.6 / T(8)=37.2 ms は既にその値）。
- verify 形が定常形に 2 本足されるので PreparedPlan と backing 予算の勘定が増える（H-15 の既定 256 MiB
  に収まる — verify 形は decode 形と同じ桁）。

## 追記（2026-09-08・段 2 の実装で改めた点）

- **決定 1 の束ね口は `createSession` ではなく context**: drafter Session は pipeline に 1 本
  （`createSession(model, { sharedWeights })` — 決定 2 の共有はここ）、会話ごとに
  `drafterSession.createGenerationContext({ chunkLength: 1, borrow: targetContext })` で**借り手 context**を
  作る。理由は 2 つ — GenerationContext は sequence ごと（ADR 0083 決定 3）なので Session で束ねると drafter の
  出力ヘッド（i8 ≈ 67 MB）を会話ごとに再アップロードすること、backing の世代 token が Session ごとの採番
  （executor.ts）で 2 つの Session が同じ context に bind group を焼くと衝突すること。借り手の run は貸し手の
  run リースを取る（貸し手の進行中 run / 未 commit / poison が拒否理由）ので直列化は runtime が持つ。
  IR の宣言は external スロット（`states[].external: true`・append 0 本・読者は readonly attention だけ・
  全スロット external）・readonly attention（`attrs.readonly: true`・`ins: [q]`・M = 1）・共有 initializer
  （`initializers[].shared: { tensor }`・バイト無し）の 3 種。寿命は借り手 → 貸し手の順で、逆順の dispose は
  Promise.reject（冪等契約からの意図的な逸脱）。
- **決定 8 の「既定は DL しない」は hub の `ResolveOptions.weights`**（取得する weights の部分集合・
  karume/4 据え置き）。`resolveFiles` は ModelEntry の weights を全数展開するので、drafter を第 2 role に置いた
  だけでは必ず DL される — gemma4 は `speculative` 指定時だけ `["model", "drafter"]` を取る。
- **drafter の linear は i8（i4 g32 ではない）**: 同一 target 軌跡での実測（2026-09-08・N=200 × 3 ケース）で
  i4 g32 の linear は E[a](k=3) を丸め無し比 −15〜−33% 落とし、i8 なら −2〜3%（共有する主表の i8 化は ±0）。
  損は drafter 自身の linear（22 本・10.1M 要素・全体の 13%）に集中する。配布は +4 MB 弱。
- **readonly の sliding は 512 列**（`[P−W, P)` — mlx の `|q−k| < W`・target 自身の述語と同じ）。HF の
  bidirectional SWA は inclusive（513 列）だが、513 列目（位置 P−513）は余裕 8 の ring で kmax=8 の棄却が
  起きると潰されうる行なので採らない。同一入力での drafter の突合（HF 正規経路と 150/150 一致）には効かない。
- 段 2 の範囲: k は 3 固定（配布形の drafter が 3 段で焼かれている）・`estimateSessionMemory` は target のみ・
  `fromAssets` は `speculative` を受けない・投機ループ（draft → verify → 受理・棄却）は段 3。

## 追記（2026-09-08・段 3 の実装で改めた点）

- **段 2 の drafter 呼び出し規約は 1 段ずれていた（訂正）**: 段 2 の golden / e2e は drafter に
  `(token@P−1, hidden@P−1, position P−1)` を渡していたが、HF の `SinglePositionMultiTokenCandidateGenerator`
  と drafter 自身の段間再帰（token と「それを出した行の hidden」の組を次段へ送る）が定める組は
  **`token = b`（最後に確定した token = 位置 P・KV 未投入の frontier）・`hidden = h@(P−1)`（b を出した行）・
  `position = P`** である。ずれた組では d₁ が「target が logits@P−1 から無料で出す bonus」を当て直し、実効 k が
  1 減る（golden の再計算: d₁ → 位置 P の一致 0.39〜0.53・位置 P+1 は 0.03〜0.08）。グラフは無傷で、訂正は
  ホスト（`speculative.ts` の意味）と台本（`draft_case` の 3 行）・golden の焼き直し（container のバイトは同一・
  配布形は不変）。訂正後の golden（greedy target・k=3・N=200）: short-en 1.50 / readme-recipes 2.13 /
  readme-exporter 2.11 token/cycle（位置別一致 d₁ 36 / 56 / 62%）。段 2 の「1800/1800」「HF 150/150」は
  **torch の移植の同値性**の門であって受理率の裏づけではない。
- **決定 4 の門を 1 列締める**: deferred run は **`queryLength ≤ slidingSlack`**（段 1 の `slack + 1` から）。
  借り手（readonly 読者）は貸し手より 1 列古い列 `P−W` まで読むので、棄却行 j が潰す論理列 `P+j−C` が
  借り手の窓に入らない条件は `j − m < C − W = slack`（m = commit した行数・`commit(0)` を含む）⟺ `Q ≤ slack`。
  借り手の有無で分岐しない（借り手は deferred + `commit(0)` の後にも開ける）。MUST: `k + 1 ≤ slidingSlack`
  （gemma4: 4 ≤ 8）。
- **決定 3 の commit は「配送した frontier まで」**: verify（deferred）の受理列 `[d₁..d_a, b']` を 1 個ずつ
  配送し、`pendingToken` を yield の前に更新する既存の MUST に乗せて、**消費者に届いた token の frontier まで**を
  `commit(rows)` する（`rows` = frontier にした token の数 = target が消費した行数）。配送ループの直後と
  generator の `finally` の両方で畳むので、消費者の `break` は「配送した token まで」、verify 戻り〜配送の同期区間の
  例外は `commit(0)` になり、どちらも「会話 = 受け取った列 + frontier 1 個」という非投機と同じ形に閉じる
  （保留を残すと context は dispose しか受け付けない）。
- **決定 5 の R は k+1 固定**: verify の `last_row` は `[0..k']` を末尾添字で `k+1` 本に pad する（R は
  PreparedPlan の鍵に入るので、予算末尾で k' が縮む cycle ごとに別形を作らない）。`k' = 0` は `[0]`（decode 形
  そのもの）。
- **決定 7 は「温度に依らない」形で閉じた（訂正）**: 受理は行ごとに `sampler.next` を非投機の decode と
  同じ logits・同じ history・同じ順で 1 回ずつ呼ぶ（確定 token 1 個につき 1 回）ので、RNG の消費列も
  token 列も温度に依らず非投機と**厳密に一致**する。温度 > 0 では「draft と同じ token を引いたら受理」が
  one-hot draft の speculative sampling そのもの（受理確率 = target 分布での draft の確率・棄却時は
  引いた token がそのまま新しい frontier）で、drafter の logits は要らない。設計時に置いた「温度 ≠ 0 は
  非投機へ落とす」は根拠が誤り（レビューで反証）で撤回、**段 3b は不要**。配布形の推奨 sampler
  （温度 1.0）での受理率は段 4 の実測項目。
- **投機ループの置き場と観測**: ループは `src/generation/sequence.ts` の内側で、drafter は
  `GenerationSequenceOptions.speculative = { open(context) → DraftFace, k }` の DI（`DraftFace` = draft と
  dispose だけの狭い面 — GPU 無しの fake で受理 0 / 停止 token / 中断 / 例外 / 途中 break を全部踏む）。
  観測は run 単位の hook `onRun(phase)`（prefill / decode / draft / verify・番号 1 始まり・verify は commit の
  直後の同期区間）で、イベント列から run 数を導出していた `withRunDiagnostics` は置き換える。
  `GenerationStop.speculation`（cycles / draftRuns / drafted / accepted / acceptedHistogram）が勘定。
- **同一性の門**: gemma4 の既定席（`stateAttentionReduce: "parallel"`）では decode（M=1）が ①′・verify（M=4）が
  ① で縮約順が違い、ビット同一ではない（設計どおり）。厳密一致の門は `"sequential"` 席（M=1 も M=4 も ① + ③）で
  採り、既定席は同じケースで相違数を報告する（limitations）。①′ の位置不変化（既定席でも u32 門）は段 4 の候補。
- **見積り**: `estimateSessionMemory` は target の prefill / decode に verify 形 `{ M: k+1, R: k+1 }` を足し
  （runtime の `generation.scenarios`）、drafter の常駐重み（共有 initializer は除外）と借り手 context の
  lengths 8 バイトを合算する。
- **readonly の sliding 512 列は据え置き**（HF は inclusive 513 列 — 意図的な差。追記 2026-09-08〈段 2〉の
  理由のとおり）。受理率への影響は G2 の実測に含まれる。
