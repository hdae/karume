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
  （∝ pastLength × queryLength）が attention 実装（ADR ⑤）で満たされること。
- 「鍵は容量」の反例（llama.cpp の量子化 extent）を明記したことで、将来 decode 性能が
  容量鍵で頭打ちになった場合の転換先が記録に残る（その時は本 ADR の supersede）。
