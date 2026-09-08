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
- **RoPE は attention op の外**（グラフの通常ノード列）。~~層種別 RoPE（Gemma 4 E2B の
  theta 100 倍差 + partial rotary 0.25 — 調査 §6.1）はエクスポータがグラフに焼く。~~ →
  **ADR [0091](0091-gemma4-host-rope-variable-capacity.md) 決定 1 で置換**（表は配布物に入れず、
  ホストが chunk ごとに cos / sin を派生入力として渡す）。**`attention op は RoPE を知らない`
  MUST は 0091 でも不変**。

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

## 追記（2026-09-03）— ③PV の KV 並列縮約変種（perf-ledger K-12）

決定 4 の states 形 ③PV（`O = P @ V`）に **KV 長方向を workgroup 内 16 レーンで分担する変種**
（`attention_state_pv:…:wg16x16:par`）を足した。1 invocation が O の 1 要素を live 列の逐次ループで
積む形は、decode（M=1）で有効 invocation が `D × B·H`（Gemma 4 E2B の full 層で 4,096）に固定され、
KV 長が 1 スレッドの逐次長にしか効かない — P=16K で attention が decode GPU 時間の 72% を占めた
機序（[research 2026-09-03 P 掃引](../research/2026-09-03-gemma4-context-length-sweep.md) §3）。

- **契約**: 束縛・params・dispatch の**本数**は ③ と同一（workgroup 幾何は違う — 行軸が
  `rowsBlock / TILE_M` から `rowsBlock` へ変わるので workgroup 数は prefill で 4 倍。decode は
  `rowsBlock = 1` なので両者一致する）。レーン `l` が `cl ≡ l (mod 16)` の列を昇順に
  部分累積し、workgroup 共有メモリで固定順の木（stride 8 → 4 → 2 → 1）に畳む。決定性（同一入力 →
  同一出力）・容量非依存・行ブロック非依存・pad 行 → 厳密 0 は ③ と同じくビット門で保つ。
  縮約順が違うので **③ とビット同一ではない**（本 ADR 冒頭の「縮約は col 昇順の逐次で固定」は ③ の
  契約であり、③' は「レーン部分和 → 固定順木」を自分の契約として持つ）。
- **席**: ADR 0058 の opt-in 席 `SessionOptions.stateAttentionReduce: "sequential" | "parallel"`
  （既定 `"sequential"` = 参照経路）。検証門は 3 点セット — ③ の既存門は無変更・③' の A/B 帯門
  （`tests/gpu_state_attention_parallel_test.ts`・帯 5e-6・実測最悪は ③ との差 2.4e-7 /
  f64 参照との差 3.99e-7）・census 門
  （`tests/gpu_state_execution_test.ts` — 席どおりのキーが走り、他方が混ざらない）。
- **実測**（[research 2026-09-03](../research/2026-09-03-gemma4-chunklength-k12-sweep.md) §3）:
  full PV 35.0 → 3.6 ms・sliding PV 6.2 → 0.8 ms・decode 壁 P=16K 81.3 → 41.0 ms/token（×1.98）・
  token 列一致。**既定への昇格（2026-09-03）**: ADR 0058 決定 6 のとおりユーザーの品質裁定（対話 example の目視）と e2e golden の再走（不変）を同一コミットで行い、`Gemma4Pipeline` は `stateAttentionReduce: "parallel"` を既定で Session に与える（`GEMMA4_STATE_ATTENTION_REDUCE`）。runtime の既定と低レベル面（decode 系列の検収門）は参照経路のまま。
- why-not（split-KV + merge 段 / online softmax / subgroup）: workgroup 内分割は dispatch 数も中間
  バッファも増やさず実装が最小で、③' を段 A の中身として流用する形で split-KV へ伸ばせる。
  online 形は S 一時を消す別の価値を持つが decode の並列度を単体では解かない。subgroup はアダプタが
  feature を広告せず入場不可（2026-08-10 プローブ）。

## 追記（2026-09-06）— ①QK の D 方向並列縮約変種 ①′（perf-ledger K-14）

決定 4 の states 形 ①QK（S の 1 要素 = D の逐次内積）に、③′ と同じ手筋で **D 方向を workgroup 内
16 レーンで分担する変種**（`attention_state_qk:v1:f32:wg16x16:par`）を足した。decode（M=1）では
1 invocation の遅延が D 逐次で長く、P=16K で ①QK が decode GPU の 17%（K-12 後の最大の attention 項）
だった（[research 2026-09-06](../research/2026-09-06-state-qk-parallel-k14.md)）。

- **契約**: 束縛・params・dispatch 本数は ① と同一。1 workgroup = 局所行 1 本 × 16 列で、レーン `l` が
  `d ≡ l (mod 16)` を昇順に部分累積し、共有メモリで固定順の木（stride 8 → 4 → 2 → 1）に畳む。
  書く条件（live 範囲は述語外でも −inf・`cl ≥ live` と pad 行は書かない）と半スケールは ① と同一。
  縮約順が違うので **① とビット同一ではない**。決定性・容量非依存・行ブロック非依存・述語外 −inf の
  ビット一致・pad 行非書き込みはビット門で保つ。
- **席**: ③′ と**同じ** `SessionOptions.stateAttentionReduce: "parallel"`（ノブは 1 つ — 2026-09-06
  ユーザー裁定）。ただし **①′ が選ばれるのは M（chunkRows）= 1 の計画だけ**
  （`stateQkParallelEligible`）: prefill 計画（M=768）では ① が既に行 × 列で埋まっており、①′ は行タイル幅
  4 → 1 で workgroup と barrier を 4 倍積むだけになって ①QK が 1.5〜1.9 倍・壁が +30〜60% 逆行した。
  ③′ は全 M で席に従う（K-12 の実測で prefill も逆行しない）。判定材料は計画時の静的値だけ
  （実行時の論理長で分岐すると同じ計画鍵が run ごとに違うパイプラインを指す）。
- **検証門**: ① の既存門は無変更・①′ の A/B 帯門（`tests/gpu_state_attention_parallel_test.ts`・帯
  5e-6・実測最悪は ① との差 3.58e-7 / f64 参照との差 4.17e-7・故障注入 2 種で落ちる）・census 門
  （`tests/gpu_state_execution_test.ts` — M=1 / M>1 × 席の 6 行）・非 GPU の適用条件の真理値表
  （`tests/kernel_state_attention_test.ts`）。
- **実測**（同 research）: decode ①QK 6.3 → 3.7〜4.1 ms/token（×1.55〜1.72）・decode 壁 P=16K
  37.2〜39.4 → 33.0〜35.4 ms/token（−9〜15%）・P=256 −4.5%・門の後の prefill は ① と同等。
  `Gemma4Pipeline` の既定は `"parallel"` のまま（追記 2026-09-03 の昇格に ①′ が乗る — golden /
  reduce_parity の token 列は不変）。
- **prefill 側**: 律速が traffic（K 行を M 行ぶん読み直す）なので D レーン分割は効かない。M でバケット
  する幾何表（M=1: D レーン / M ≥ 16: K タイル共有）が「同じ族・同じ席」で両方を持つ形で、後者は
  perf-ledger K-13 の設計。

## 追記（2026-09-06）— 幾何表: prefill 計画（M ≥ 16）は GEMM 骨格のタイル経路 ①ₜ / ③ₜ（perf-ledger K-13）

決定 4 の ①QK / ③PV は 1 invocation = 1 要素で、prefill 計画（M = chunk 行数 768）では K / V 行を M 行
ぶん読み直す traffic 律速だった（14.7K token の chunk 1 本で attention が GPU の 79% —
[research 2026-09-06](../research/2026-09-06-state-attention-tiled-k13.md)）。融合 attention（ADR 0023 系）
が持つ GEMM 骨格（共有タイル・レジスタブロック・M バケット幾何・K タイル 16 昇順 — ADR 0022 決定 3）に
states 用の断片を差した ①ₜ / ③ₜ を足し、**M ≥ 16 の計画で席に依らず選ぶ**。

- **契約（ビット同一）**: ①ₜ は A = q·scale / B = k·scale を共有タイルに置き骨格が d 昇順に
  `acc = acc + a * b` を回す = ① の 1 項の式（半スケールを双方に）と加算順に一致。③ₜ は A = P
  （S と行統計から充填時に `exp(S − m)·inv`・非実体化）/ B = V を col 昇順に回す = ③ と一致。
  境界は実効 live で切り（`dims.n` / `dims.k` は col_cap の静的上界）、S は live 範囲を述語で
  −inf / 値に埋めて `[live, col_cap)` と pad 行は書かない（① と同じ残骸）、O は full-write で pad 行は
  `select` の厳密 +0.0（③ と同じ — V に非有限が混ざっても NaN 化しない）。有効行を含まない行タイルは
  K ループ 0 周（仕事量 ∝ Q — ADR 0066 決定 3）。門 = S / O 全語の u32 一致（17 / 19 ケース・述語外 −inf・
  残骸・pad 行込み）+ 故障注入 + gemma4 golden 厳密一致。
- **幾何表**（この波の帰結 — GEMM の `gemmGeometryForRows` と同じ「M で選ぶ」型）:

  | 計画の M        | ①QK                  | ③PV                  | 数値                                 |
  | --------------- | -------------------- | -------------------- | ------------------------------------ |
  | 1（decode）     | ①′（席 parallel）/ ① | ③′（席 parallel）/ ③ | 席で選ぶ（並列縮約は A/B 帯門）      |
  | 2〜15           | ①                    | ③′（席 parallel）/ ③ | 同上                                 |
  | ≥ 16（prefill） | ①ₜ                   | ③ₜ                   | 参照経路とビット同一（席に依らない） |

  ③′（追記 2026-09-03）の適用は M < 16 へ狭まる（prefill での ③′ の利得は誤差内だった）。適用条件は
  純関数（`stateQkTiledEligible` / `statePvTiledEligible` = `chunkRows >= 16`）で、計画時の静的値だけで決まる。
- **実測**（同 research）: prefill 20 chunk の GPU 35.2 / 40.9 → 10.3 / 11.0 s（①QK ×13・③PV ×11〜12）・
  P=16K の prefill 壁 53.6 / 56.8 → 19.5 / 20.6 s（−64%）・decode 不変・token 列一致。残る prefill の GPU は
  linear が 72%。
- why-not（online softmax）: S の実体化を消す価値は別軸で、今の律速は traffic だった。①ₜ / ③ₜ の断片は
  online 形の段の中身として流用できる（追記 2026-09-03 の why-not と同じ筋）。

## 追記（2026-09-08）— ring の法は `window` ではなく `capacity`（ADR 0096 決定 4・MTP 段 1）

- 決定 4 の `slot_row(col) = col % window` を **`col % capacity`** に改めた（読み書き同式の MUST は
  不変 — `stateSlotRowWgsl` の 1 文字列を ①/①′/①ₜ/③/③′/③ₜ/append の全経路と参照実装が共有する）。
  `column_base` / `live_columns` / `in_window` は window 基準のまま — 物理行数と論理窓幅は別の量に
  なった。sliding スロットの capacity は配布形が `window + 余裕` で焼き（gemma4 は 8）、余裕は投機
  verify の棄却行の置き場（ADR 0066 追記 2026-09-08）。`window ≤ capacity` の門（決定 4 ③）は不変。
- `state_append` の重複排除ガード（Q > capacity で同じ物理行へ写る論理行のうち最後の 1 本だけが
  書く）も capacity 基準。window 基準のままだと capacity > window のとき、誰とも alias しない行が
  黙って書かれない（prefill の chunk が行を落とす）。
- 決定 5（append はスロットにつきちょうど 1 本・最後のノード）は段 1 では不変。段 2 で「読むだけの
  外部スロット」を第 3 種として足す（ADR 0096 決定 1）。
