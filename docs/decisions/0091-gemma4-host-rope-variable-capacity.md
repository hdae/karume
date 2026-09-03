# 0091: gemma4 の RoPE はホストが chunk ごとに供給する — capacity / chunkLength を実行時ノブへ

- Status: accepted（2026-09-03 — ユーザー裁定「それぞれ推奨案ベースで進めてください」。設計材料は
  [research/2026-09-03-gemma4-context-length-sweep.md](../research/2026-09-03-gemma4-context-length-sweep.md)
  と [research/2026-09-03-gemma4-chunklength-k12-sweep.md](../research/2026-09-03-gemma4-chunklength-k12-sweep.md)）
- Date: 2026-09-03
- 関連: ADR [0067](0067-autoregressive-attention-vocabulary.md)（決定 4 の「層種別 RoPE はエクスポータが
  グラフに焼く」を**本 ADR が置換** — 「attention op は RoPE を知らない MUST」は不変）/
  [0083](0083-generation-api-surface.md)（Consequences 4「RoPE 表のホスト生成は再裁定要」の再裁定）/
  [0034](0034-dit-dynamic-tokens.md)（知見 2 — torch の f32 三角関数とはビット同一にならない。本 ADR は
  「表に焼く」でなく「許容差の parity + golden 再凍結」で受ける）/ [0085](0085-ple-host-gather.md)
  （chunk ぶんだけホストが作る派生入力の前例）/ [0066](0066-generation-context-state-slots.md)（chunkLength
  は計画時定数・容量記号の束縛点は context 生成時）/ [0089](0089-memory-limits-preflight.md)（requiredLimits の
  state 束縛源 — `capacity` → `maxPosition`）/ [0070](0070-shard-loading-admission.md)（estimator の性格）/
  [0058](0058-numerics-opt-in-contract.md)（K-12 の席）/ [0038](0038-manifest-v1.md)（`pipelineConfig` の内側は
  format bump 不要）

## Context

gemma4 E2B の配布形は RoPE（回転位置エンコーディング）の cos / sin 表 4 本（層種別 2 × cos / sin・
f32・6 KiB/行）を IR の初期化子として焼き、会話の最大長 `capacity` は表の行数 `maxPosition`（1,024）
で頭打ちだった。上流は 128K。表を 128K 行焼くと配布は **+762 MiB**（既存の 1,024 行を除いた増分 =
(131,072 − 1,024) × 6,144 B）・VRAM は **768 MiB**（表の全量 = 131,072 × 6,144 B — 常駐するのは
全行なので増分ではなく全量が要る）で、短い会話の利用者にも一律に掛かる。

P 掃引（research 2026-09-03）で分かったこと: VRAM は 128K でも +2.3 GiB で壁にならず、天井は
時間の側にある（prefill は chunkLength 32 のまま P² に効き 16K で 5.4 分・decode は full 層 attention
の ③PV 段が KV 長を 1 スレッドの逐次で走り 16K で 2.6 倍）。設計材料の読み取りで確定した事実:

1. 出荷 IR で `position_ids` を読むのは RoPE の gather 4 本だけ。gather 後の値は `[1, M, headDim]`
   で、PLE と同じ「chunk ぶんだけホストが作る派生入力」の席に嵌まる。表を capacity 行ぶん常駐
   させる形は、容量記号 `C` を入力 shape に持ち込んで `capacitySymbolOf` と束縛点 MUST を壊す。
2. 上流の RoPE は全経路 f32（角度 `position × invFreq` を f32 で積む — 131,071 で ULP ≈ 0.008 rad・
   cos / sin は SLEEF の 1 ULP）。TS が f64 で計算して f32 へ丸めても**ビット同一にはならない**
   （実測は測定範囲ごとに別の数: 位置 0..1,024 の掃引の最大 6.9e-5・位置 131,071 ちょうどの点 4.8e-3・
   位置 0..131,071 の全掃引の最大 9.4e-3）。数学的に正確なのは f64 側で、gemma4 の検収門は
   もともと token 列の離散一致 + logits atol 1e-2（余裕床 2.5e-2）なのでビット同一を要求していない。
3. chunkLength は出荷 IR に上下限が無く（記号は名前のみ・768 由来の定数ゼロ）、`karume.json` の宣言
   だけで 768 まで上げられる。実測で prefill は 5〜9 倍縮み（16K で 326 → 60 s）、token 列は不変。\
   追記 2026-09-03: 上限は宣言 `maxChunkLength` が門になった（決定 3）。IR に上限が無いという事実は
   変わらないので、trace 範囲を知る唯一の口が配布形の宣言になる。

## Decision

### 1. RoPE 表は配布物に入れず、ホストが chunk ごとに cos / sin を派生入力として渡す

グラフ入力は `input_ids` + `rope_{sliding_attention,full_attention}_{cos,sin}`（f32 `[1, M, headDim]`・
sliding 256 / full 512）+ `per_layer_inputs` + `last_row`。**`position_ids` 入力・RoPE の gather・
`rotary_emb.*` 初期化子は消える**。exporter は上流の rotary モジュールを「ラッパが受け取った
cos / sin を層種別で返すだけ」の受け渡し口に差し替え、焼いた表の残骸はグラフ門で拒否する。

TS 側は `packages/models/src/gemma/rope.ts` が正本で、`DerivedRunInputs.derive(ids, positions, …)`
（位置列を受ける形へ改訂）の中で PLE gather と同じ席から 4 本を作る。pad 行（位置 0）も通常行と
同じ式で埋める（cos = 1 / sin = 0）。

### 2. 数値契約: TS f64 正本・上流実出力との位置比例 parity・golden 再凍結

- 式は `invFreq[i] = theta^(−2i/headDim)`（`i < rotaryDim/2`・それ以外 0）・`angle = position ×
  invFreq`・行 = `cat(freqs, freqs)`。default / proportional の 2 rope_type を同じ 1 式に畳む
  （attention_scaling = 1・factor = 1 以外は export で fail loudly）。
- parity 門は**上流モジュールの実出力**（fixture）との突合で、許容差は位置比例
  `2.5e-7 + position × 1.2e-7`（上流の f32 角度誤差の上界）。故障注入（位置ずらし・層種入替・theta・
  rotaryDim）で門が鳴ることを機械で示す。Python 側 `gemma4/rope.py` は同じ式の鏡像で、同じ fixture と
  突合する（同じ式を 2 回書いた恒真にしない）。
- ビット同一は主張しない。決定性は f64 IEEE 演算で engine 非依存、`Math.cos` / `Math.sin` の engine 差
  ≤ 1 ULP は f32 丸め後に残る確率 ≈ 2⁻²⁹/要素で無視する（Metal の 1 ULP 群と同じ扱い）。
- 出荷バイトは再 export で必ず動く（GPTQ の丸め解）。gemma4 は未公開なので golden（greedy / chat /
  product digest）を**再凍結**する。旧 golden との token 差分件数を記録する。

### 3. 配布形の宣言（`pipelineConfig` — 内側なので format bump 不要）

`chunkLength`（既定 768 = export の Dim 上限）・`maxChunkLength`（= 記号 `M` の trace 上限 768・
追記 2026-09-03）・`capacity`（既定 4,096 = 実測の実用線）・
`maxPosition`（= 上流 `max_position_embeddings` 131,072 — **モデルが宣言する位置の上限**）・
`rope`（層種別 `{theta, headDim, rotaryDim}` — config から導出・写経しない）・`sampler`。
TS の門は 2 本 — `chunkLength ≤ maxChunkLength` と `chunkLength ≤ capacity ≤ maxPosition`。
`maxPosition` の意味だけが「焼いた表の行数」から「モデル宣言」へ変わる。

### 4. capacity / chunkLength は実行時ノブ

- **capacity** は `sequence()` / `Gemma4ChatSession` の生成時（1 会話 = 1 context = 1 容量）。既定は
  `program.capacity`（= 宣言値）。容量記号の束縛点は `createGenerationContext` だけ（ADR 0066 追記 7
  不変）。`Gemma4ChatSession` の溢れ処理はセッションの容量を見る。
- **chunkLength** は `Gemma4PipelineOptions`（計画時定数 — ADR 0066 決定 4 のまま・PreparedPlan の
  LRU を汚さない）。既定は宣言値。
- 見積りは `Gemma4Pipeline.estimateSessionMemory({capacity, chunkLength})` — ロード後・context 生成前の
  面（ADR 0089 追記 2026-09-02 の「呼び手が形を決めた後に使う面」と整合）。runtime 側は states 形
  attention の一時 S / 行統計を勘定に入れた（`c574aa7`）。

### 5. requiredLimits は `maxPosition`（配布形が許す最大容量）で焼く

state 記号の束縛源を `capacity` → `maxPosition` へ。既定容量で焼くと「既定より大きい容量を選んだ
瞬間に宣言を満たす device で落ちる」形になる。gemma4 の焼き値は不変（主 embedding i8 384 MiB >
131,072 の full スロット 256 MiB）。

### 6. prefill 進捗と診断の口

`Gemma4ChatOptions.onPrefill?: ({chunk, chunks}) => void`（chat / send 共通・非破壊）と
`Gemma4PipelineOptions.onRunDiagnostics?: (SessionDiagnostics) => void`（他 7 家族と同型）。

### 7. K-12（③PV の KV 並列縮約）との関係

decode の線形項は本 ADR の外（perf-ledger K-12・ADR 0067 追記 2026-09-03）。opt-in 席で実装済み
（`8b76ef4`）。既定への昇格は ADR 0058 決定 6（品質裁定 + golden 更新を同一コミット）に従い、本波の
golden 再凍結と同じ回に行った（2026-09-03 ユーザー裁定 — `Gemma4Pipeline` の既定 = `"parallel"`・`stateAttentionReduce: "sequential"` で参照経路へ戻せる・runtime の既定は不変）。

## Consequences

- **破壊的変更**（未公開面）: グラフ入力から `position_ids` が消える（`fromAssets` / 低レベル面の消費者は
  RoPE 派生入力を自前で組む — `./gemma` サブパスが `gemma4RopeInputs` を出す）・
  `GenerationProgramSpec.positionIds` 廃止・`DerivedRunInputs.derive` の引数に位置列・`pipelineConfig`
  に `rope` 必須。旧配布形（表を焼いた karume/4 gemma4）は読めない（未公開なので互換層なし）。
- 計測専用配布形 pos16k / pos128k（RoPE 表差し替え）は役目を終える。
- 実用上限は時間で決まる: chunkLength 768 + K-12 で P=16K は prefill 60 s / decode 41 ms/token。
  次のレバーは prefill attention の K/V タイル再利用と ①QK の D 逐次内積（perf-ledger）。
- 追随した docs: ADR 0067 追記・ADR 0069 項 3 追記・recipe README・配布カード（card.py）・limitations
  （2 上限の意味・chunkLength で token 列が動きうる）・perf-ledger。minicpm5 の `export_decode` は
  表を焼く形のまま（別裁定）。research 2026-09-02 turboquant-recon §3「文脈長の天井は RoPE 表」は
  時点スナップショットとして残し、本 ADR が上書きする。
