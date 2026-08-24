# 0067: attention の autoregressive 語彙 — GQA 整除 broadcast と state 参照つき契約

- Status: accepted（2026-08-17 — ユーザー裁定 A〈GQA × i8a8 は fail loudly 開始・後日
  サポート前提〉+ 委任チェック方式。Codex レビュー第 3〜6 巡を反映し第 6 巡で go —
  states 形は第 4 巡指摘で 1 種へ再設計・第 5 巡で sliding 両側述語と物理 shape を接続）
- 関連: ADR [0023](0023-fused-attention.md)（融合 attention — 決定 4 の**一部**を supersede）/
  [0044](0044-runtime-attention-mask.md)（safe_softmax 意味論）/
  [0060](0060-row-block-attention.md)（行ブロック実行 — 保存経路への接続が本 ADR の受入条件）/
  [0066](0066-generation-context-state-slots.md)（state スロット・論理長スカラ — 前提）/
  [0058](0058-numerics-opt-in-contract.md)（未実装の組は縮退でなく fail loudly）
- 根拠:
  [research/2026-08-17-autoregressive-references.md](../research/2026-08-17-autoregressive-references.md)
  §3・§6（以下「調査 §n」）

## Context

検収モデルは両方とも現契約で書けない: Gemma 4 E2B は 8:1 の MQA（`num_key_value_heads: 1`）、
MiniCPM5-1B は 16:2 の GQA — `attention` は q/k/v の H 完全一致を要求する（ADR 0023 決定 4・
shapes.ts:524）。さらに autoregressive 実行には「KV を state スロットから読む」「causal /
sliding を表す」「論理長で仕事を切る」語彙が無い。参照実装の確定事実は調査 §3（GQA は
kernel 内整数除算が主流・mask は attrs + 値 or 述語計算・空行 0 の構成は実装依存）。

## Decision

### 1. GQA = `attention` の H 突合を整除 broadcast へ緩める（G3 案 A）

`q[B,H,M,D]` / `k[B,Hkv,N,D]` / `v[B,Hkv,N,D]`・条件 **`H % Hkv == 0`**・出力 `[B,H,M,D]`。
`r = H / Hkv` は**導出値**（attrs 欄を作らない — 「GQA は欄を作らない」の趣旨は維持）。
実装は **`H ≥ Hkv ≥ 1` を併せて課す**（`0 % Hkv == 0` で H=0 が、等値短絡で H=Hkv=0 が
素通りする縮退形の検出線 — 2026-08-17 実装波 A の追修・独立レビュー指摘）。

- supersede は ADR 0023 決定 4 のうち「**q/k/v の H 完全一致**」の 1 句のみ。B 完全一致・
  k/v 間の Hkv 一致・D 3 者同一・N=0 拒否は**取り違え検出線としてそのまま維持**する。
  Hkv=1（MQA — Gemma 4 E2B）も同式で表す。
- `bmm` のバッチ整除 broadcast（案 B）は入れない — bmm の「バッチ完全一致」は意図的な
  検出線（ADR 0022/0023）で、緩めると B 取り違えが shape 検査を素通りする面が広がる。
  分解経路の GQA は救わず、**GQA モデルは SDPA 保存が必須**（エクスポータの `enable_gqa`
  全件拒否〈aten_handlers.py:800-804〉を「保存ターゲットのみ条件付き受理」へ改める）。

### 2. 実装形: kernel 内整数除算 + uniform・r=1 はバイト同一

head 写像は `wid.z / r`（`wid.z = b*H + h` に対し `H = Hkv·r` なら
`wid.z / r = b*Hkv + h/r` が整数除算で厳密成立 — ORT WebGPU と同一構成・調査 §3.2）。

- f32 融合経路の変更は gemm.ts の bbase 算術 2 枝（attention_qk / attention_pv 共有枝 —
  後者は linear と同居のため op 分岐を足す）。codegen キーに GQA ビット 1 本・r は uniform。
- MUST: **r=1 の生成 WGSL はバイト同一**（ORT JSEP の `nReps === 1` 特殊化が先例 — 既存
  スナップショット門・ビット同一門を凍結したまま席を足す）。
- 受入条件: **repeat_kv 実体化版との Uint32 parity**（r ∈ {1,2,8}・B>1 を含む形状群 +
  故障注入〈r 誤り・pv 側写像漏れ〉）。ADR 0060 決定 3 と同型の「base 算術のみ差分」論証は
  実測で確認するまで主張しない（調査 §7 attn LB-6）。

### 3. GQA × i8a8 は fail loudly で開始（裁定済み 2026-08-17 — 後日サポート前提）

i8a8 attention は別 WGSL で head 基底が 5 本（attention-i8a8.ts:378-382 — K/scale・V/scale
のみ kv-head へ写し Q/S/O は q-head のまま）、recipe-builder の K/V 量子化・確保も `B*H`
前提（調査 §3.2）。**初期実装は `attentionCompute: "i8a8"` × GQA 形を fail loudly で拒否**
する（ADR 0058 決定 3 —「未実装の組は縮退でなく fail loudly」。黙って f32 へ落とすと
性能が静かに変わる）。

**拒否は暫定で、後日サポートを前提とする**（ユーザー裁定）。追補時の対象面は確定済み:
①head 基底 5 本のうち kbase / ksbase（と PV 側の V/scale 基底）だけを kv-head 写像
（`wid.z / r`）に変え、qbase / qsbase / sbase は q-head のまま ②recipe-builder の K/V
量子化・確保を Hkv 形へ。検証は f32 経路の GQA parity 資産（決定 2 の repeat_kv 突合）を
i8a8 版へそのまま流用できる形で作っておく。

### 4. state 参照つき attention（同一 op 名の契約拡張・欄の有無が形を判別）

ノードに**省略可能な `states` 欄**（ADR 0066 決定 2 の「ins / outs と別の欄で名前参照」）を
足す: `{ "k": <slot 名>, "v": <slot 名> }`。欄が**無い**ノードは従来契約そのまま
（既存資産・既存門は無風 — mask 第 4 入力〈maxArity〉と同じ拡張手筋）。欄が**ある**形:

- **形は 1 種のみ**（第 4 巡で単純化 — 当初案の「共有形 = ins なし」は sliding ring で
  append 先行が必須になり、Q>1 で共有層の窓が欠ける〈満杯 ring へ Q 行 append すると
  row 0 の要る過去 W−1 行のうち Q−1 行が消える〉ため廃止）: **ins の k/v = 今 step の
  新規 k/v・スロット = 過去分のみ**。ins の宣言 shape は **`[B,Hkv,M,D]`（M = 物理 chunk
  次元 — prefill は chunkLength・decode は 1）**で、有効データは先頭 queryLength 行の
  compact-prefix（ADR 0066 追記 6 — queryLength は shape でなく実行時スカラ。第 5 巡で
  物理 / 論理の表記を接続）。KV 共有層
  （Gemma 4 E2B の末尾 20 層）は自層で projection を計算せず、**所有層の k/v 値テンソルを
  ins にそのまま配線**する（グラフ配線 + 同一スロット名参照で共有を表す — 解決規則は
  vLLM の kv_sharing_target_layer_name と同じ「同種 attention の直近非共有層」・
  refs/vllm gemma4.py:462-488）。全読者が past を読み終えた後に append する（決定 5b）
  ので、ring 容量 = window のままで staging も slack も不要。
- **スロットの物理形と検査**（第 4 巡で追加）: states 形が参照するスロットは
  **`[B, Hkv, C, D]` 固定**（C = 容量・dtype は f32〈f16 は ADR 0066 追記 5 の席〉）。
  contracts は ①k/v スロットの同形 ②ins との B / Hkv / D 一致 ③`window ≤ C`（sliding）
  ④full スロットは実行時に `pastLength + queryLength ≤ C`（context 側検査）を
  fail loudly で課す — 通常値のみ見る現行 shape 検査（shapes.ts:510-540）の state 延長で、
  スロット取り違えを OOB / 沈黙誤読の前で止める。
- **causal 固定**（欄を作らない — 非 causal + state の実在需要が無い。双方向 prefill は
  states 無し形で表す）。判定は述語 `col ≤ pastLength + row`（論理座標・TVM 型・調査 §3.1
  — mask tensor は実体化しない）。**sliding 時は下限述語を AND する MUST**（第 5 巡 high の
  閉鎖）: `max(0, pastLength + row − window + 1) ≤ col ≤ pastLength + row` の**両側**。
  上限（causal）だけだと row > 0 が resident 全体を走査して**窓外 key を row ぶん余計に
  沈黙混入**する（W=4 で row 1 は正しい 4 個でなく 5 個を見る）。一次実装 = vLLM の
  `(q_abs − key_pos) < W` AND（triton_attention_helpers.py:197-229 @7ea4b40）。
- **sliding window は省略可能 attrs `window`**（正の int・欄の不存在 = 全 context）。
  層別混在（Gemma 4 E2B の 28/7）はノードごとに違う attrs で表す — 別 op・別 kernel を
  作らない（調査 §3.4 の全実装一致）。**論理 col → 物理 row の写像は読み書き同式 MUST**:
  sliding スロットの物理 row = `col % window`（`state_append` の書き込み式と同一 —
  読み側だけ別式にすると沈黙誤読になる）。読者が参照する past の resident 範囲は
  `[pastLength − min(pastLength, window − 1), pastLength)`（append 前なので row 0 の窓まで
  全行 resident — 形 1 種化の成立根拠）。current 部分（`col ≥ pastLength`）は ins から
  読む。カーネルは論理座標で述語を評価してから写像する。
- 論理長（pastLength / queryLength）は**実行時スカラ**として **context 所有の可変
  uniform**（ADR 0066 追記 4 — params 内容アドレスキャッシュに載せない）で渡し、dispatch
  数は**ホストが論理長から算出**する（karume は graph capture を持たず毎 run エンコード
  するため、ORT の indirect dispatch 相当は不要 — ADR 0066 の仕事量合格条件
  〈∝ queryLength × (有効 past + queryLength)・追記 1 の訂正式〉をホスト側 dispatch
  算出で満たす）。
- **RoPE は attention op の外**（グラフの通常ノード列）。層種別 RoPE（Gemma 4 E2B の
  theta 100 倍差 + partial rotary 0.25 — 調査 §6.1）はエクスポータがグラフに焼く。
  attention op は RoPE を知らない MUST。

### 5. KV の書き込みは別 op `state_append`

「今 step の k/v をスロットへ書く」のは attention ではなく**単機能 op `state_append`**
（slot 名 + 入力 `[B,Hkv,M,D]`〈宣言 shape — attention の ins と同じ物理 chunk 次元〉+
論理位置スカラ。**書くのは先頭 queryLength 行のみ**〈pad 行は書かない — スロットは
full-write 対象外・ADR 0066 追記 6〉。sliding スロットは `position % window` のリング
書込みもここが持つ）。

- why-not（attention 内蔵 = TVM 型）: dispatch は 1 本増えるが、①full-write / padding 行
  no-op（queryLength が切る — ADR 0066 決定 4）の検証が単機能 op に閉じる ②attention 側は
  読み取り専用のままビット同一検証が単純 ③KV 共有層（append を持たない層）が
  「`state_append` ノードが無い」だけで表せる。ORT の kv_empty（present 出力なし）と同じ
  表現力を op の不在で得る。**append の不在は層単位の話で、スロット単位では終端
  `state_append` が常にちょうど 1 本**（共有層はその 1 本を複数の読者で分け合う）— 検査の
  粒度がスロットなのはこのため（決定 5b・`runtime/plan.ts` の `assertStateOrder`）。
- why-not（staging / ring slack = ORT の WindowedKvCache 型）: 共有層のために append を
  読者より先に置く設計なら ring に `window + Q − 1` の slack か staging バッファが要る
  （refs/onnxruntime GQA cpu 実装が同種の staging 切替を持つ）。決定 4 の「全読者が ins で
  current を受ける」形なら append は常に最後で、容量 = window のまま済む — 採らない。
- **出力は 0 本**（値を定義しない effect op）。IR パーサの「outs 空は拒否」
  （format/ir.ts:322-324）は「**契約が effect を宣言する op に限り 0 本を許す**」へ改訂し、
  実装 6 面の出力数一般化は ADR 0068 決定 1 が受け持つ（第 3 巡の矛盾指摘の解消）。

### 5b. state effect の順序 = nodes 配列順（データ辺に依存しない）

state 参照（読み・書き）は**テンソルのデータ辺を張らない**ため、DAG のトポロジ順では
順序が決まらない。契約: **同一スロットに触れるノード同士の実行順は `nodes` 配列順を
保存する MUST**（plan / recipe は state を触るステップの相対順を並べ替えない —
融合 matcher も state 跨ぎの並べ替えをしない）。エクスポータの発行規約:

1. **当該スロットの全読者（所有層 + 共有層の attention）→ `state_append`（書き）**の順に
   発行する（第 4 巡で単純化 — 全読者は past をスロットから・current を ins から読むので、
   append は常に最後の 1 回。ring wrap が今 step の読者の過去行を潰す経路が構造的に無い）。
2. 検査: plan は「同一スロットへの append は 1 step に 1 回まで」「append より後に当該
   スロットの読者が居ない」を fail loudly で検査する（発行順の誤りを沈黙誤値にしない —
   第 3 / 4 巡 high 指摘の閉鎖）。

### 6. 空行 → 0 の意味論を states 形に内蔵（safe_softmax 系）

states 形では padding 行・（chunk 先頭での）空 context 行が**正規に**出るため、
ADR 0044 の「融合 attention へ全 −inf 行は契約違反」は states 形に**適用しない** —
states 形の行統計は「**行 max 初期値 −inf + 分母 0 ガード**」の構成で空行 → 出力 0 を
構造的に保証する（llama.cpp WebGPU 型 — 調査 §3.4）。

MUST: **有限 sentinel（−5e4 等）で −inf を代用しない** — TVM はこの構成で空行が
「V の重み 1 平均」になっており、safe_softmax 契約（行 max −inf → 全 0）を満たさない
（調査 §7 attn LB-4 の refuted が根拠）。

### 7. S の実体化は行ブロック（保存経路への 0060 接続 — 受入条件）

states 形の ①QK は S を行ブロック窓で実体化する（`[B·H, block, N]` — ADR 0060 の機構を
保存 `attention` 経路へ移植）。decode（queryLength=1）は S が `[B·H,1,N]` で常に 1 枚。
prefill chunk × 長 context で `maxStorageBufferBindingSize` 128MiB を超える形が正規に
来るため（調査 §3.1）、**行ブロック無しの states 形実装は受入不可**（第 2 巡 high 指摘）。
S とは別に **state スロット自体の binding も上限を超えうる**（Gemma 4 E2B full 層の
131K 容量 × f32）— そちらの契約（容量ゲート + f16 席予約）は ADR 0066 追記 5 が持つ。

## Consequences

- 既存資産・既存門は無風（states 欄なし・r=1 バイト同一・mask 契約不変）。
- ADR 0023 は決定 4 の H 句のみ supersede 注記を受ける。ADR 0044 は「states 形は空行正規」
  の対照注記を受ける。
- エクスポータ: `_h_attention` の enable_gqa 条件付き受理・states/`state_append` の発行は
  decode グラフ台本（実装波）で。IR 仕様は states 欄・`state_append`・attrs `window` の
  3 点で本書改訂。
- 受入条件（実装波のゲート）: ①r=1 スナップショットバイト同一 ②repeat_kv parity + 故障
  注入 ③census（GQA が実際に効いた検査 — 0058 決定 4）④行ブロック動作（強制分割含む）
  ⑤空行 → 0 の直接門（全 padding 行入力）⑥既存 sha 門全緑。
