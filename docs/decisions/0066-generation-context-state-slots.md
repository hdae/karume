# 0066: autoregressive 実行モデル — GenerationContext と名前付き state スロット

- Status: accepted（2026-08-17 ユーザー承認「OK」— 確認ポイント 3 点〈states の分界・固定長
  chunk・所有権分離と rewind〉込み。方向は同日の裁定 1〜4〈条件込み〉で先行承認）
- 関連: ADR [0042](0042-prepared-execution-plan.md)（PreparedPlan の bindings 鍵と backing —
  本 ADR 決定 5 が所有権を分離）/ [0054](0054-resident-loop-and-fence.md)（resident / batch
  enqueue — state バッファは第 4 の寿命クラスの延長）/
  [0004](0004-execution-model.md)（アリーナ不変条件・flush-before-destroy）/
  [0008](0008-public-api.md)（公開面は薄く）/
  [0058](0058-numerics-opt-in-contract.md)（数値契約の席の作り方 — 本 ADR は数値変種を作らない）
- 根拠:
  [research/2026-08-17-autoregressive-references.md](../research/2026-08-17-autoregressive-references.md)
  （参照実装 8 リポの一次調査 + 敵対検証 2 巡。以下「調査 §n」）。波順の正本は
  [backlog](../backlog.md) next 節（ADR ① に当たる）。

## Context

現行の Session は「不変重み + 1-shot 実行」で、生成状態を持つ席が無い。検収モデル
（Gemma 4 E2B / MiniCPM5-1B — 調査 §6）を動かすには、①KV state の寿命と置き場 ②容量と
論理長の分離 ③prefill / decode の実行形 ④PreparedPlan（ADR 0042）との整合、の 4 点を
先に確定する必要がある。参照実装の確定事実（調査 §1）:

- どの実装も plan/graph 再利用の鍵は「トポロジを決めるパラメータ集合」で、論理長を
  shape symbol にする実装は無い（渡し方は 3 方式に分かれる — 調査 §1.1）。
- KV を通常の tensor I/O にする形（transformers.js）は毎 step の出力再配線・rewind 不能・
  capture 系最適化と非両立の三重苦で、避けるべき実物として確認済み（調査 §1.2・§1.4-2）。
- state は層均一ではない — Gemma 4 E2B は「sliding 512 × 28 層 / full × 7 層 / 自前 KV を
  持たない 20 層」の 3 種混在で、per-layer shape 差 + KV 共有層は ORT にも席がある
  （調査 §1.5・§6.1）。

## Decision

### 1. 寿命の分離: Session は不変・GenerationContext が 1 生成の可変 state を所有する

`Session`（不変重み・計画キャッシュ — 現行のまま）から `GenerationContext` を生成する。
context は**単一シーケンス**の生成 1 本ぶんの可変 state（state スロットの物理バッファ +
論理長）を所有し、`dispose()` で返す（flush-before-destroy — ADR 0004）。

MUST: **state をグラフの通常 input / output にしない**（past 入力・present 出力の欄を IR に
作らない）。state はノードが名前で参照する不透明スロットであり、実体の確保・寿命・進行は
context が持つ（MLC の `nn.spec.Object` 型 — 調査 §1.2）。

### 2. 名前付き state スロット（R3）

IR v1 に **`states{}` セクション**を追加する: `name → { dtype, shape }`。shape は
**固定 rank（rank ≤ 4）の容量込み具体形**（記号は Session の symbols で解決可）。

- **「層 × 均一 KV」の前提を作らない**。sliding 層と full 層は**容量の違う別スロット**、
  KV 共有 20 層は**同一スロット名の参照**で表す（欄の追加なし）。KV 以外の逐次 state
  （将来の LSTM h_n / conv state）も同じ器に載る — スロットは「KV」を知らない。
- ノードからの参照方法（attention の読み書き欄・op 語彙）は **ADR ⑤ が決める**。本 ADR は
  「スロットは `ins` / `outs` と別の欄で名前参照する」という原則までを固定する。
- エクスポータは `states` を出す最初のモデルまで無風（未知キー fail loudly の流儀どおり、
  本書改訂と同時にパーサへ入れる — 未リリースにつきシム無し）。

### 3. shape 不変条件の再宣言（R2）と論理長の契約

恒久不変条件は次の 3 つ**まで**とする: **静的物理格納**（スロット容量は context 生成時に
確定・実行中に再確保しない）・**固定 rank**・**計画キャッシュの鍵は容量**（論理長を鍵に
入れない）。

- 「鍵は容量」は**トレードオフの選択であって業界の既定ではない**と明記する — llama.cpp は
  逆に「256 境界へ量子化した論理 extent」を鍵に入れて成立している（調査 §1.1 (a)）。karume
  が容量鍵を選ぶ理由は、PreparedPlan（ADR 0042）が bindings 完全一致鍵 + LRU 4 であり、
  extent を鍵に入れると decode が毎 token 別計画になるため。
- **論理長は「値」**: `pastLength`（確定済み KV の論理長）と `queryLength`（今回 step の
  実 token 数）は**独立の実行時スカラ**として渡す。shape symbol にも attrs にもしない
  （symbol は計画鍵に入り〈ADR 0042〉、attrs は IR に焼き付くため。ORT の
  seqlens_k / total_sequence_length 分割と同型 — 調査 §1.1 (c)・§3.1）。
- **容量 − 論理長の差は「遮蔽された key / 無効な行」**として扱い、shape で表さない
  （調査 §3.4）。
- **仕事量の合格条件**: attention 系 dispatch は「workgroup 数または総反復回数が
  `pastLength × queryLength` に比例」しなければならない（容量比例は 131K 容量 ×
  短系列で 3 桁の無駄 — 調査 §1.1）。満たす機構（`dispatchWorkgroupsIndirect` は core
  WebGPU・またはホスト既知の論理長から dispatch 数を算出）の選定は ADR ⑤ の実装設計。
- **有界論理 extent の席予約**（宣言のみ・実装は最初の実需モデルまで先送り）:
  将来の data-dependent shape は「compact-prefix 軸 1 本の有界論理 extent」として
  `states` / 値宣言に載せられる形を予約する。DDS op は payload + extent の複数出力
  （ADR ④ の multi-output が前提）・extent は計画鍵に入れない・admission は容量課金。
  上限超えと動的 rank は従来どおりホスト介在のグラフ分割で扱う。

### 4. 実行形は 2 本: 固定長 prefill-chunk と decode

- **prefill**: プロンプトを**固定長 chunk**（`chunkLength` — context 生成時に確定する計画時
  定数）へ分割し、末尾 chunk は pad する。**padding 行の no-op 契約が本体**: pad 行は
  KV へ書かれず・出力に現れず・（決定 3 の合格条件の下で）仕事量にも比例しない。切るのは
  `queryLength`。
- **decode**: `queryLength = 1` 固定形。
- 採らなかった形と理由: 可変長 chunk（web-llm / MLC 型 — 調査 §1.3）は記号コンパイル 1 本で
  済む処理系の解で、karume では chunk 長ごとに PreparedPlan が増えて LRU 4 を汚す。
  全 prompt 一括（ORT GenAI 既定）は長 prompt で計画・transient が prompt 長に比例して
  単発化する。固定長 chunk + pad は ORT Windowed / vLLM バケット型の選択（調査 §1.3）。
- 結果として **PreparedPlan は prefill-chunk / decode の 2 本が定常**（+既存の 1-shot 面）。
  動的形状 KV が capture 系最適化と非両立という二重の一次証拠（調査 §1.4-2）を、固定容量
  resident state を再利用系の前提条件とする形で先取りする。

### 5. PreparedPlan との所有権分離（レシピは Session・state backing は context）

- **レシピ / 計画（bindings の純関数 — ADR 0042 §1-2）は Session 共有のまま**。
  MUST: **context の識別子を計画鍵に入れない**（入れると context 切替のたびに導出と
  backing 構築が全滅する）。
- **state スロットの物理バッファと、それを束ねる bind group は GenerationContext が所有**
  する。ADR 0042 §5 の「焼き込み bind group」は「Session 所有の slot / 常駐入力」と
  「context 所有の state」で束ねる相手の寿命が異なるため、**焼き込みの単位を分離**する
  （state を含む bind group は context 側に持つ / state を含まない dispatch は従来どおり）。
  分離しないと「前の context の KV を束ねたまま回る」沈黙 stale 読みか、切替ごとの全再構築
  スラッシングの二択になる（調査 §1.6 — 第 2 巡 high 指摘）。
- 診断席を置く（`planBacking` と同格 — state backing の常駐バイト数・再束縛回数。
  沈黙劣化の唯一の観測点になる設計は ADR 0042 §3 の流儀を踏襲）。

### 6. 寿命操作の API 面（薄く — ADR 0008）

- `session.createGenerationContext(spec)`: スロット容量・`chunkLength` を確定し物理確保。
  確保失敗は out-of-memory errorScope で fail loudly（admission の estimator は ADR ③）。
- `context.pastLength`（読み取り）: 論理長の進行は **context が所有**し、run の成功で進む。
  ホスト側の手動加算を API にしない（二重簿記の禁止）。
- `context.rewind(position)`: 論理位置の切詰め。**sliding スロットでエビクト済みの範囲へは
  fail loudly**（巻き戻し可能範囲は容量方式から導出 — ORT GenAI 型・調査 §1.2）。
- `context.dispose()`: flush-before-destroy 遵守で物理バッファを返す。
- prefill / decode の呼び出し形（メソッド名・入出力型）は実装設計で確定する — 本 ADR は
  「論理長の進行と巻き戻しは context の所有」という契約までを固定する。

### 7. device-loss 契約

device lost 後の GenerationContext は**無効**であり、全操作は `GpuDeviceLostError` 系で
fail loudly する（WebGPU 仕様上、lost device 由来の GPU 資源は回復不能 — 調査 §1.4-4）。
生成は失われる。ホスト側 state の dump / restore（llama.cpp 型 = state バイト + メタのみ・
RNG / sampler を含まない）は**席の予約だけ**行い、実装は実需まで先送り。

### 8. スコープ外（本 ADR で決めないこと）

batch > 1・複数シーケンス・paged / block table 間接参照（KV は連続容量 — 調査 §1.4-1 の
明示選択）・prefix cache・attention / GQA / mask の op 語彙（ADR ⑤）・decode 出口と
multi-output（ADR ④）・packed 格納（ADR ②）・shard ロードと admission（ADR ③）・
sampling / RNG（ホスト維持 — op-vocabulary 裁定済み）。

## Consequences

- 既存の 1-shot 面（Session.run / enqueue・全既存モデル・全 sha 門）は**無風** — 本 ADR は
  席を足すだけで既定経路を変えない。
- IR v1 は `states{}` の追加で本書改訂を受ける（version は 1 のまま — 未リリース改訂手順）。
- ADR 0042 の backing 節（§4-5）は決定 5 の分離を受けて実装時に追記改訂される。
- 受入条件（実装波のゲート・第 2 巡レビュー由来）: ①`queryLength` が `pastLength` と
  独立の実行時スカラとして実装されること ②padding 行 no-op が機械検証されること
  （pad あり / なしで KV バイト同一 + 出力同一）③context 切替で「stale 読みゼロ +
  レシピ再導出ゼロ」が同時に成立すること（切替 A/B テスト）④仕事量合格条件
  （追記 1 の訂正式）が attention 実装（ADR ⑤）で満たされること。
- 「鍵は容量」の反例（llama.cpp の量子化 extent）を明記したことで、将来 decode 性能が
  容量鍵で頭打ちになった場合の転換先が記録に残る（その時は本 ADR の supersede）。

## 追記（2026-08-17・第 3 巡レビュー反映 — 訂正 2 点・補強 3 点）

accepted 直後の第 3 巡（Codex 独立レビュー・5 本セット照合）で確定した訂正と未決の閉鎖。
いずれも一次ソースで再確認済み（調査 doc §7 に台帳追記）。

1. **決定 3 の仕事量式を訂正**: 「∝ pastLength × queryLength」は pastLength=0 の causal
   三角（chunk 内自己参照）が非ゼロであるため誤り。正 = **∝ queryLength ×
   （有効 pastLength + queryLength）**（sliding は有効 past を窓で制限）。受入条件④は
   この式で読む。
2. **決定 6 の rewind 契約を差し替え**: sliding スロットは「エビクト済み範囲のみ拒否」では
   足りない — エビクト発生後は **resident な位置への rewind も物理配置と論理範囲が一致
   しない**（ORT GenAI は同理由で current 未満への rewind を全拒否 —
   windowed_kv_cache.cpp:67-80 のコメントが一次根拠。左詰め compaction は持たない）。
   契約: **sliding スロットを含む context の位置指定 rewind は fail loudly**（全スロット
   非 sliding の context でのみ有効）。緩和は compaction 実装時の本 ADR 改訂。
3. **失敗の原子性（決定 6 の補強）**: state 変更 dispatch（`state_append` を含む run）を
   提出した後に run が失敗した場合、論理長は進まないが物理 ring は上書きされ得る —
   **context は poison（無効化・以後の全操作 fail loudly）**とする。rollback / staging は
   持たない（復旧 = 新 context + ホスト側再構築）。
4. **論理長スカラの搬送路（決定 3 の補強）**: `pastLength` / `queryLength` は params
   （内容アドレスキャッシュ — ADR 0042）に**載せない**。毎 step 値が変わるものを内容
   アドレスに載せると「キャッシュ無界成長（limitations 既知）」と「PreparedPlan ヒット時に
   導出相が走らず更新不能」の両方を踏む。搬送は **context 所有の可変 uniform バッファ
   1 本**（毎 run の encode 前に writeBuffer・レシピからは固定束縛で参照）。
5. **容量と binding 上限（決定 1 / 6 の補強）**: スロット単体の binding バイト数が
   `maxStorageBufferBindingSize` を超える容量指定は `createGenerationContext` で
   fail loudly（Gemma 4 E2B の full 層 131K 容量 × f32 は既定 128MiB を超えうる —
   実用容量の診断は ADR ③ の estimator）。state 格納の f16 席（数値契約が変わるため
   ADR 0058 流儀の opt-in）は**予約のみ**。
6. **固定 chunk の物理 shape と pad の値契約（決定 4 の補強・第 4 巡）**: prefill-chunk 系の
   テンソル宣言 shape は **chunkLength 固定**（plan / prepared backing は従来どおり宣言
   shape の全量確保 + full-write — 機構は不変）。有効データは**先頭 queryLength 行の
   compact-prefix**。**pad 領域の入力値は 0 埋め MUST**（ホストが埋める — pad 行の k/v が
   NaN / 非有限だと「−inf 加算 + exp」経路で valid 行へ NaN が漏れるため、値契約で遮断）。
   pad 行が valid 行を汚染しないことは**行局所性**で保証する: linear / pointwise / norm は
   行内で閉じ、attention は causal 述語により valid 行が pad 列（`col ≥ pastLength +
   queryLength` 相当）を見ず、`state_append` は queryLength 行しか書かず、出口（ADR ④）は
   実末尾行しか読まない。**full-write 不変条件との整合（第 5 巡で明確化）**: pad 行も
   通常出力としては**書かれる**（各カーネルは宣言 shape の全バイトを書く — ADR 0014 系の
   不変条件は不変。「不定」は**値が契約上無意味**という意味であって未書込みではない）。
   書込みが queryLength 行に限られるのは **state スロットだけ**（context 所有バッファで、
   transient slot の full-write 対象外 — 残骸は次 step の append が同じ式で上書きし、
   読者は resident 範囲外を読まない）。

7. **states 専用記号の束縛可能性（決定 2 の補強・実装波 C で判明 — 2026-08-17）**: 現行の
   `checkSymbolBindability`（format/ir.ts）は「全 symbols が**入力 shape の次元位置**に
   現れる」ことを要求するため、**states にしか現れない記号（KV 容量 `C` 等）が宣言できない**。
   容量は context 生成時にユーザーが決める値（決定 3 の R2 — 静的物理格納）なので、export 時
   定数に焼く形は本 ADR の前提と矛盾する。契約: **states の shape に現れる記号は
   `createGenerationContext(spec.bindings)` を束縛点とする有効な宣言**とし、束縛可能性検査は
   「入力 shape ∪ states shape のどこかに現れる」へ緩める。効く範囲の分担（第 2 巡指摘で
   精密化 — 決定 3 との整合）: **通常値 shape の解決には従来どおり入力由来の束縛だけが効く**
   （states 専用記号は値 shape に現れない）が、**state を参照する PreparedPlan の鍵には
   解決済み容量が入る**（決定 3 の「鍵は容量」そのもの — context の識別子ではなく容量の値。
   同一 Session で C=512 と C=131072 の context を作れば、state 参照計画は容量ごとに別鍵に
   なるのが正しい）。実装は波 D。

8. **pad 行の出力値を 0 に固定（追記 6 の narrowing・実装波 D-7 — 2026-08-18）**: 追記 6 の
   「pad 行も通常出力としては書かれる（値が契約上無意味）」を、**厳密 0 を書く**へ狭める。
   きっかけは仕事量合格条件（追記 1 の訂正式）の独立レビュー指摘 — pad 行が live 列を走査すると
   仕事が物理 chunk 行数 M に比例してしまうため、①QK / ②行統計は有効行（`row < queryLength`）
   だけを覆い、③PV は pad 行で live を 1 列も走査せず全 D に 0.0 を書いて返す（full-write
   不変条件は不変 — 「不定」だった値を 0 に固定しただけ）。帰結 2 つ: ①**空行 → 0（ADR 0067
   決定 6）は「空行 ⊂ pad 行」によりこの 0 書きが構造的に包含する**（valid 行は causal
   自己参照を必ず含むので非空）。②行統計の空行ガード（amax = −inf → (0,0)）は**防御専用**に
   なった（有効行しか覆わない現行構成では発火経路を持たない — dispatch とカーネルの行範囲が
   割れた実装バグの検出線として温存）。非有限な V が空行出力を `0 · NaN` で汚す残穴も同時に
   閉じている。
