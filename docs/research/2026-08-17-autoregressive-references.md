# autoregressive 対応の参照実装調査（2026-08-17）

> **性格**: 時点スナップショット（2026-08-17 実施）。backlog「next — autoregressive-ready
> 基盤波」の ADR ①〜⑤ の根拠資料として、llama.cpp / vLLM / ONNX Runtime（本体 + GenAI +
> Web）/ WebLLM（MLC / TVM）/ transformers.js の一次ソースを shallow clone（commit 固定 —
> §8）+ Web 公式資料で突合したもの。参照実装は更新が速く、行番号・実装はここに記した
> commit 時点の事実である。

## 0. 調査の方法と信頼度

- 構成: 掃引 8 レッグ（broad-shallow）→ 深掘り 3 レッグ（ADR 軸別・narrow-deep）→
  敵対検証 3 レッグ（深掘りの load-bearing 主張 18 本を独立反証）。
- 検証結果: 第 1 巡（ワークフロー内）**holds 15 / refuted 3**・第 2 巡（Codex 独立レビュー・
  決定案込み）**新規 refuted 4 + 見落とし 4**（§7 台帳）。refuted は全て一次ソースで再確認の
  うえ**本文に訂正済みの形で反映した**。
- 掃引 1 本（検収モデル構造）は構造化出力の失敗で欠落 → 単発レッグで補充（§6）。
- 引用形式は「リポ/相対パス:行 @commit」。カルメ側の要所引用（container.ts / executor.ts /
  ir-v1.md）はメインセッションが現物再確認済み。

## 1. KV state / GenerationContext の実行モデル（ADR① の根拠）

### 1.1 decode 1 step を安価にする鍵 — 論理長の渡し方は 3 方式

**どの実装も plan/graph 再利用の鍵は「トポロジを決めるパラメータ集合」であって形状全体では
ない**。論理長（pastLength 相当）の扱いで 3 方式に分かれる:

- **(a) グラフ次元だが量子化**（llama.cpp）: `get_n_kv()` が使用済み最大位置を
  **max(n_pad, 256) 境界へ切り上げ**る。コメントに「pad the n_kv value so that the graph
  remains constant across batches and can be reused」と明記
  （llama.cpp/src/llama-kv-cache.cpp:1233-1246 @4df29be）。再利用判定はこの n_kv を mask の
  ne[0] で突合（src/llama-graph.cpp:47-64）。**計画鍵に「量子化済み論理 extent」が入る**形。
  グラフキャッシュは直前 1 本のみ（`gf_res_prev` — src/llama-context.cpp:1325-1395。
  can_reuse 成立時は build/alloc を丸ごと飛ばして set_inputs のみ）。
- **(b) GPU 常駐テンソルのデータ**（ORT native WebGPU EP）: シェーダ内で
  `let total_seq_length = u32(seqlen_k[batch]) + 1u;` と読み、さらに GPU 側で indirect
  dispatch buffer を書いて **dispatch 数まで論理長依存**にする（ホスト readback なし —
  onnxruntime/onnxruntime/contrib_ops/webgpu/bert/flash_attention.cc:120,134-139,163-172
  @dd64c8a）。この入力は `ProgramTensorMetadataDependency::None` で登録され
  **パイプライン鍵に入らない**（同 :211。鍵の組成は
  onnxruntime/core/providers/webgpu/program_cache_key.cc — uniform は長さのみ参加）。
  **但し書き**: indirect dispatch の有効化は
  `use_indirect_dispatch = seqlen_k != nullptr && total_seqlen != nullptr &&
  context.IsGraphCaptureEnabled()`（flash_attention.cc:551-553）— **graph capture 前提の
  経路**で、非 capture 時はホスト値から dispatch 数を算出する。
- **(c) ホストのスカラーテンソル入力**（ORT GenAI）: `past_sequence_length` は `[1,1]` int32
  CPU テンソルで毎 step 加算（onnxruntime-genai/src/models/input_ids.cpp:19-29,66 @f6a871d）、
  `total_sequence_length` は **search.max_length（= 容量）で固定**（同 :148）。
  「容量」と「論理長」が**別入力として分離**している実例。

いずれも論理長を shape symbol にはしていない（ADR 0042 の懸念と一致）。(b) が唯一
「論理長を鍵から完全に外す」形で、その代償が indirect dispatch という機構。

**R2 席予約（「計画鍵は容量」）への含意**: (a) は「量子化済み論理 extent を鍵に入れる」
実在の反例であり、「計画鍵は常に容量」は業界の既定ではなく**トレードオフの選択**として
ADR に書く。容量ぶん常に計算する素朴案は 131K 容量 / 100 token 使用時に 3 桁の無駄となり
成立しないため、「extent を鍵に入れない」を守るなら **attention の仕事量を論理長に比例
させる機構の席が同じ ADR に要る**。合格条件は「**workgroup 数または総反復回数が
pastLength × queryLength に比例する**」こと（第 2 巡で精密化 — スレッド内の early-exit
だけでは容量ぶんの dispatch / スケジューリングコストが残る。ORT の indirect dispatch は
これを GPU 側で満たす実装）。

### 1.2 Session = 不変重み / GenerationContext = 可変 state の寿命分離

| 実装            | 不変側                     | 可変 state                                                | rewind / 解放                                                              |
| --------------- | -------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------- |
| llama.cpp       | `llama_model`              | `llama_context` の `llama_memory_t`（kv_size 固定確保）   | `llama_memory_seq_rm/cp/keep`（include/llama.h:739-795）                   |
| MLC / TVM       | コンパイル済み VM + params | `PagedKVCache` **オブジェクト**                           | `kv_state_add/remove_sequence`（web-llm/src/llm_chat.ts:397-435 @90f6709） |
| ORT GenAI       | `Model` / `Session`        | `State` 内 `DefaultKeyValueCache`（ctor で固定形状 1 回） | `RewindTo`（kv_cache.cpp:606-636 @f6a871d）                                |
| transformers.js | ONNX session               | `DynamicCache`（**通常の graph I/O**）                    | 旧テンソル即 `dispose()`・rewind なし                                      |

- KV を tensor I/O にしない形の一次証拠 = MLC のモデル spec:
  `"paged_kv_cache": nn.spec.Object(object_type=PagedKVCache)` を prefill（`seq_len` 記号）/
  decode（定数 1）**別関数**の引数に置き、`begin_forward`/`end_forward` で括る
  （mlc-llm/python/mlc_llm/model/llama/llama_model.py:449-464 @2f78caa /
  web-llm/src/llm_chat.ts:1265-1281 @90f6709）。
- 逆に transformers.js は present.\* を通常出力（`preferredOutputLocation:'gpu-buffer'`）に
  して毎 step 参照差し替え + dispose する形
  （transformers.js/packages/transformers/src/models/session.js:110-124 @353007b）—
  **避けたい形の実物**（毎 step の出力再配線・rewind 不能・capture 不能）。
- rewind の可否は state スロットの容量方式から導出される: ORT GenAI は共有バッファ +
  sliding 層でエビクト済み位置への RewindTo を**拒否**する（kv_cache.cpp:606-636）。

### 1.3 prefill / decode の 2 実行形と chunked prefill

- 別グラフ（MLC: prefill = seq_len 記号 / decode = 定数 1）が WebGPU 出荷経路の実例。
- **chunked prefill は広く存在するが、chunk の実行形は割れる**（第 2 巡で訂正 — §7）:
  ORT GenAI の**既定は全 prompt 一括**（DefaultInputIDs — input_ids.h:46-52 のコメントが
  「In contrast, DefaultInputIDs processes all prompt tokens at once」と明記）で、固定
  window + pad token 埋めは **WindowedInputIDs = 特定モデル向けの設定**
  （input_ids.cpp:118-200 @f6a871d）。web-llm / MLC は `prefill_chunk_size` を**上限**と
  する可変長 chunk（末尾 chunk は実長のまま実行 — llm_chat.ts:859-883・prefill は
  seq_len 記号の別関数なので pad 不要）。vLLM は capture 済みバケットへパディング
  （vllm/v1/worker/gpu/model_runner.py:1109-1116,1638-1647 @7ea4b40）。
- → 「**固定長 chunk 1 種 + pad**」は ORT Windowed 型・vLLM バケット型が採る**選択肢の
  一つ**であって業界形ではない。カルメが固定 shape（PreparedPlan 再利用）を保つ目的で
  この型を選ぶ場合、**queryLength（今回の実 chunk 長）を pastLength と独立の実行時スカラ
  として持ち、padding 行の no-op 契約（KV 書込み・出力・仕事量の抑止）を ADR① で先に
  固定する**ことが前提になる（第 2 巡の見落とし指摘② — high）。

### 1.4 R2 との突合で確定した分岐点

1. paged KV（vLLM / TVM）は「物理は静的ブロックプール・論理→物理は device 側 block table の
   間接参照」（vllm/v1/worker/gpu/block_table.py:294-321 @7ea4b40）。カルメの現行語彙
   （直接アドレス・STRIDED_RANK=4）には間接参照の席が無い。**単一シーケンス前提なら不要**
   だが、「KV は連続容量」を ADR① で明示的に選ぶこと。
2. **動的形状 KV は graph capture 系最適化と非両立**（二重の一次証拠）: ORT WebGPU EP は
   `ORT_ENFORCE(!IsGraphCaptureEnabled() || kv_empty || past_present_share_buffer_)`
   （group_query_attention.cc:372-377 @dd64c8a）、GenAI 側も share buffer 無効 + capture で
   throw（kv_cache.cpp:422-425 @f6a871d）。固定容量 resident KV は再利用系の前提条件として
   明文化する。
3. KV 実体は BNSH / `[num_blocks, kv_heads, block_size, 2*head]` 等の 4D で **rank≤4 に
   収まる**（衝突するのは GQA の突合 = G3 の別軸）。
4. device-loss: WebGPU 仕様上 lost device 由来の全オブジェクトは使用不能・回復は
   requestDevice からやり直し（https://www.w3.org/TR/webgpu/#lose-the-device）。web-llm は
   unload + `DeviceLostError` で**生成を捨てる**（engine.ts:374-384,427-429）。ホスト再開が
   要る場合の形は llama.cpp の state serialize（KV バイト + cell メタのみ・RNG/sampler は
   含まない — src/llama-context.cpp:3173-3191）。

### 1.5 R3（名前付き state スロット）の裏付け

- ORT GenAI は per-layer に KV shape（head_dim 差・sliding 容量差）を分けて確保する
  （kv_cache.cpp:386-478 — layer_shapes_ / ComputeWindowedKvCacheSize）。
- ORT WebGPU EP には「present 出力を持たず**他層の KV を共有する層**」の席があり、コメントは
  「kv_empty layers (e.g. **Gemma4 layers 15-34**) reuse KV from another layer」
  （group_query_attention.cc:362-364,374-375 @dd64c8a）。
  → **検収モデルの 3 種混在（sliding / 層間共有 / GQA）は既存実装で現実に起きている**。
  「層 × 均一 KV」前提は最初から捨て、名前付きスロット（per-slot shape・別名共有可）が正。

### 1.6 カルメ固有の裁定点: GenerationContext × PreparedPlan の所有権（第 2 巡）

prepared 鍵は**常駐入力の実体 ID を含み**、backing は焼き込み時に常駐テンソルを bind
group へ畳み込む（executor.ts:1159-1174 — resident ID 衝突対策で入れた設計）。KV state を
常駐入力と同型で扱うと、**context の識別子を鍵に入れれば** context 切替のたびに巨大
backing が退役・再構築され、**入れなければ**別 context の KV を読む（例外なしの stale
読み）。→ ADR① は「レシピ（bindings の純関数）の共有」と「backing / bind group（物理
実体）の所有」を分離し、GenerationContext ごとの物理 state binding の帰属を裁定すること
（第 2 巡の見落とし指摘① — high）。

## 2. decode 出口（ADR④ の根拠）

- **全語彙 logits の readback を出さない形が主流**:
  - llama.cpp: 全 seq が backend sampler で賄えるなら logits の `tensor_get_async` 自体を
    発行しない（src/llama-context.cpp:1617-1631,1863-1875）。sampled / logits / probs /
    candidates は 1 本の連続出力バッファに行単位コピー（同 :1958-1965,2031-2118）。
  - web-llm: サンプル後の **int32 1 個だけ**をホストへ戻す（llm_chat.ts:1930-1949）。
  - vLLM: `logits_indices = query_start_loc[1:] - 1` で **lm_head 前に最終位置だけ gather**
    （vllm/v1/worker/gpu/input_batch.py:168 @7ea4b40）— prefill でも logits は最終 1 行のみ。
  - 対照: ORT GenAI の WebGPU デバイスは sampling/search が **常にホスト**
    （webgpu/interface.cpp:211-212 @f6a871d — GPU search は CUDA 専用）。transformers.js も
    JS サンプラーで全語彙読み出し（logits_sampler.js:108-120 @353007b）。
- **WebGPU の GPU argmax / top-k の実物**（llama.cpp ggml-webgpu）:
  - argmax: 行 = workgroup・grid-stride ローカル最大 → 共有メモリのツリー簡約
    （wgsl-shaders/argmax.wgsl @4df29be）— カルメの row-reduce と同型。**罠 2 つ**:
    ①番兵が有限値 `FLOAT_MIN = -1.0e9`（全要素がそれ未満なら index=-1 が残る）
    ②タイブレークが GPU 側 = 最大 index / CPU 側 sampler = 最小 index
    （src/llama-sampler.cpp:1053-1060）で**同一リポ内で食い違っている**。カルメの op 契約は
    タイブレーク規則と全 −inf 行の挙動を明文で固定すること。
  - top_k: `ggml_top_k` = argsort + 先頭 k ビュー。WebGPU 実装は block-local argsort +
    merge パスで **k は dst->ne[0]（構築時定数 = static-k）**
    （ggml-webgpu.cpp:2969-3110,3807-3820 — scratch は出力バッファの alloc 拡張に同居）。
    MLC は WebGPU で全語彙 argsort を使う高コスト側の例（attach_sampler.py:29-45 @2f78caa —
    i8 依存の multinomial は WebGPU から除外、乱数はホスト供給）。
- カルメ側の正確な差分: **グラフ出力レベルの multi-output は既に動く**
  （executor.ts:1512-1521 の並列 mapAsync）。未実装は**ノードレベル多出力**で、
  `assertNodeContract` が `outs.length !== 1` を拒否（ops/contracts.ts:705-709）、
  plan.ts:81,431 が `outs[0]` 前提。**加えて recipe 層（StepRecipe が単一
  outputName / output / uses — recipe.ts:84-100）・recipe-builder（#buildStep が単一出力を
  型・確保・retain で前提 — recipe-builder.ts:318-353）・エクスポータの契約門
  （`len(node.outs) != 1` を明示拒否 — tools/exporter/src/karume/ops.py:1107-1110）にも
  同じ前提が焼かれている**（第 2 巡で判明 — §7）。IR スキーマは複数 outs を許可済み
  （ir-v1.md）で **IR 仕様改訂が不要な点は維持**だが、変更範囲は contracts / plan /
  executor / recipe / recipe-builder / exporter の **6 点**で「独立小粒」ではない。
  出力 slot ごとに dtype / shape が異なる契約（top-k = 値 f32 + index i32）の設計も
  ADR④ の範囲（第 2 巡の見落とし指摘④ — medium）。

## 3. autoregressive attention / GQA / mask の語彙（ADR⑤ = G3 の根拠）

### 3.1 mask の表現主体 — 3 型

| 実装      | 主体                                            | 実体                                                                                                                                                                             |
| --------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| llama.cpp | tensor 実体化（**行数 = 現 step の token 数**） | `self_kq_mask` = `[n_kv, n_batch, 1, n_stream]`（src/llama-graph.h:343-344）。causal / SWA / seq 所属 / 空セルをホストで 1 枚に畳む（src/llama-kv-cache.cpp:1626-1668）          |
| ORT GQA   | **attrs + 値入力（mask tensor なし）**          | attrs `causal` / `local_window_size`、入力 `seqlens_k`（1D (batch)・= total−1）+ `total_sequence_length`（scalar）（docs/ContribOperators.md:2717,2725,2727,2762,2764 @dd64c8a） |
| TVM / MLC | **述語計算（kernel 内・mask tensor 皆無）**     | `col < kv_len - qo_len + row + 1`、`causal` は実行時 int32 スカラ（\_kernel_common.py:130-144 / \_prefill_kernels.py:83,249 @27c2e01）                                           |

- 皆無なのは「**容量 × 容量（全系列 × 全系列）の常時実体化**」であり、現 step 行 × n_kv の
  実体化は llama.cpp が実際に行う（prefill の M=ubatch でも二重ループで全セルを埋める —
  llama-kv-cache.cpp:1613-1681。native には束縛上限が無いので成立する。第 2 巡で表現を
  精密化 — §7）。WebGPU 側では decode（M=1・N=131072・f16 で 256KiB）は実体化で足りるが、
  **prefill は破綻する**（n_kv=131072・chunk 512・f32 で 268MB >
  maxStorageBufferBindingSize 128MiB — ADR 0060 の実測門）。→ **decode 専用の実体化 mask
  枠は残してよいが、長 context prefill は述語計算が必須**という非対称が確定。
- 一般化の注意（検証で付いた限定）: 「WebGPU 実装は mask tensor を持たない」ではない —
  ORT WebGPU も加算 bias `attention_bias`（S = 現 step 長）を実体化 tensor で受けられる。
  正確には「**TVM/MLC（WebGPU 出荷経路）が mask tensor 皆無**・他も行数を現 step 長に圧縮」。
- **静的な形（causal / window）は attrs・動的な長さ（論理 KV 長）は値**という ORT の分割が、
  「pastLength を shape symbol にしない」と厳密に噛み合う。attrs にすると PreparedPlan の
  鍵（ADR 0042）に焼き付くため、**論理長は attrs でも shape でもなく実行時スカラ**が正。

### 3.2 GQA の演算表現 — head 写像は除算が主流（ただし一枚岩ではない）

- **除算 + 実行時 uniform** の一次実装は 2 本: llama.cpp WebGPU
  `let k_head_idx = head_idx / params.q_per_kv;`（wgsl-shaders/flash_attn_tile.wgsl:58・
  uniform 宣言 flash_attn_decls.tmpl:69 @4df29be）と ORT WebGPU
  `(batch_head_idx / uniforms.n_reps)`（contrib_ops/webgpu/bert/attention.cc:175,494・
  `n_reps = num_heads / kv_num_heads` = attention_common.h:70 @dd64c8a）。
- 変種: TVM は `group_size = h_q // h_kv` を **kernel 生成時定数に焼き込み**
  （\_decode_kernels.py:53,150-151 @27c2e01）、vLLM Triton は kv-head-major grid + 乗算
  （triton_unified_attention.py:306,341 @7ea4b40）。**ORT GenAI のグラフビルダには
  repeat_kv 実体化のフォールバックが現存**（builders/base.py:2653,3476-3481 @f6a871d —
  GQA op が使えない環境向け）。ORT JSEP は生成時特殊化
  `nReps === 1 ? 'headIdx' : 'headIdx / uniforms.n_reps'`（jsep/webgpu/ops/attention.ts:549）。
- KV の確保は全実装が kv_heads 分のみ（llama.cpp/src/llama-kv-cache.cpp:207-232 ほか）。
  **repeat_kv 実体化を実行経路の既定に置く実装は無い**（フォールバック枠のみ）。
- カルメへの写像: `wid.z = b*H + h` に対し `H = Hkv*r` なら `wid.z / r = b*Hkv + h/r` が
  整数除算で厳密に成立（ORT と同一構成）。**f32 融合経路の変更点は gemm.ts の bbase 算術 —
  attention_qk 枝（:407）と attention_pv が使う共有枝（:408・linear と同居のため op 分岐が
  要る）の 2 箇所**。ADR 0060 決定 3 の行窓変種（base 算術のみ差分・K 縮約順の字面不変）と
  同型で、ビット同一の論証が再利用できる（**論証であって実測ではない** — 実装時に A/B
  証明が要る）。r=1 の生成物バイト同一化は ORT JSEP の特殊化が先例。
- **ただし i8a8 attention は別 WGSL で第 3 の変更面**（第 2 巡で判明 — §7）: head 基底が
  qbase / kbase / qsbase / ksbase / sbase の 5 本あり（attention-i8a8.ts:378-382 —
  K/scale・V/scale 側だけを kv-head へ写し、Q / S / O 側は q-head のままにする必要がある。
  取り違えは「例外なしの誤値」と当該コメント自身が警告）、recipe-builder の K/V 量子化・
  確保も `B*H` 前提（recipe-builder.ts:1396-1404 ほか）。ADR⑤ は **GQA × i8a8 を
  「実装する」か「明示拒否する」かを裁定項目に含める**こと。

### 3.3 G3 の解決候補（op 語彙の最小変更）

- **案 A（本命）— `attention` の H 突合を整除 broadcast へ緩める**: `q[B,H,M,D]` /
  `k,v[B,Hkv,N,D]`・条件 `H % Hkv == 0`・**attrs 欄は増やさない**（r = H/Hkv は導出値）。
  supersede は ADR 0023 決定 4 のうち「H 完全一致」の 1 句のみ（「GQA は欄を作らない」は
  むしろ維持）。弱点: 分解経路は救えず **GQA モデルは SDPA 保存が事実上必須** —
  エクスポータの `enable_gqa` 全件拒否（aten_handlers.py:800-804）を条件付き受理へ改める。
- **案 A の受入条件（第 2 巡の見落とし指摘③ — high）**: SDPA 保存が必須になると、現行唯一の
  128MiB 回避である行ブロック matcher（fusion.ts:923-966 — **9 ノード分解 bmm 鎖専用**）を
  **通らない**。保存 `attention` 経路への行ブロック実行（または述語 mask）と論理長 mask の
  実装を ADR⑤ の受入条件に含めること（ADR 0060「残余・接続」の宿題と同体）。
- 案 B — `bmm` にバッチ整除 broadcast（先例 = ggml の `t1->ne[2] % t0->ne[2] == 0`）:
  分解経路も開通するが、bmm の「バッチ完全一致」という意図的な取り違え検出線
  （ADR 0022/0023）を薄める。blast radius が案 A より大きい。
- 案 C — 別 op `attention_gqa`: 契約面（mask / scale / i8a8 / 行ブロック）が丸ごと二重化。
  参照実装に別 op を作った例は無い（ORT は 1 op + 必須 attr `kv_num_heads`）。非推奨。

### 3.4 sliding window / 層別混在・empty row

- 層別混在（global × sliding interleave）は**全実装が「同一 kernel + ノード別 attrs /
  パラメータ」**で表し、分岐は KV cache 側に寄る: llama.cpp は base/swa の 2 キャッシュ +
  2 mask（llama-kv-cache-iswa.h:14,95-97 / llama-graph.h:479-482）、TVM は length_info の
  3 行化（\_kernel_common.py:147-159）、vLLM は hybrid KV cache group + `per_layer_sliding_window`
  （gemma3.py:152-198）、ORT は attrs `local_window_size` + `sliding_window_cache`。
  **層ごとに別 op を作る実装は 8 リポに無い**。
- **罠（検収モデル直撃）**: Gemma3 系は sliding 層と global 層で **RoPE base が違う**
  （`rope_local_base_freq` — vllm/vllm/model_executor/models/gemma3.py:160-169 @7ea4b40）。
  層別混在は mask だけの違いではない。
- empty row（全遮蔽行）→ 出力 0 の**構成的保証は実装依存で、一枚岩ではない**（検証で
  refuted になった重要点）: llama.cpp WebGPU は「mask 値は真の −inf + 分母 0 ガード
  `select(0.0, 1/exp_sum, exp_sum != 0.0)`」で空行 0 が成立（flash_attn_tile.wgsl:228）。
  **TVM は有限 sentinel（−5e4）+ ガード無しのため、全遮蔽行の出力は 0 ではなく V の重み 1
  平均になる**（\_kernel_common.py:229-230,277-299,327 — 有限 sentinel が保証するのは
  NaN 回避のみ）。カルメの safe_softmax 契約（ADR 0044・空行 → 全 0）を融合 attention へ
  拡張するなら、**「−inf 扱い + 分母ガード」の組でなければ契約を満たさない** — 有限
  sentinel 方式を安全と誤認しないこと。
- padding の意味論: 容量と論理長の差は「遮蔽された key」として扱い shape で表さない
  （llama.cpp は空セル・別 seq・未来・SWA 窓外を全て −INFINITY の同一手段で潰す —
  llama-kv-cache.cpp:1626-1668）。

## 4. packed weight 格納（ADR② の根拠）

### 4.1 WebGPU で成立する 4bit の実物

- **ORT MatMulNBits**（最も近い先例）: B は **uint8 の物理 shape `(N, k_blocks, blob_size)`**
  （`k_blocks = ceil(K/block_size)`・`blob_size = block_size*bits/8`）、論理 K/N は
  **attrs**。scale `(N, k_blocks)`・zero_point は省略時 `2^(bits-1)`（= 対称）。block_size は
  **2 冪かつ ≥16**・bits ∈ {2,4,8}（onnxruntime/docs/ContribOperators.md:3501-3534 @dd64c8a）。
  WGSL unpack は core builtin のみ: `unpack4xU8(b & 0x0F0F0F0Fu)` + `unpack4xU8((b>>4) &
  0x0F0F0F0Fu)` の 2 発で u32 = 8 要素（matmul_nbits.wgsl.template:104-107 — 2bit は 4 発・
  8bit は 1 発）。zero point の読みも平坦添字（matmul_nbits_common.cc:30-39）—
  **ADR 0019 の i8 規律（unpack4xI8 / 平坦添字）の直接延長で書ける**。
  整列: block_size 2 冪 ≥16 なら blob_size は常に 4 の倍数 = u32 束縛が自然整列
  （**罠**: ORT には block_size == K\*N の single-scale 例外経路があり整列が壊れる —
  matmul_nbits.cc:209-215。カルメは「2 冪 ≥16 MUST」で例外を作らないこと）。
- **llama.cpp WebGPU**: Q4_0 を共有メモリタイルへの読み込み時に 1 回展開
  （wgsl-shaders/quant_inner_loops.tmpl:2-11 — ORT は内積の中で展開、立場が分かれる）。
  格納順は **32 要素 block 内で byte j の下位 = 要素 j / 上位 = 要素 j+16**（split-half）で、
  ORT の「1 バイトに要素 2i(下位)/2i+1(上位)」と**異なる**。
  → **packing 順は上流でも割れている自由パラメータ**。間違えても形も型も合い沈黙誤値に
  なるため、ADR② は展開順（unpack4xU8 2 発のインタリーブと対になる pack 順）を明文で
  固定し検出器を付けること（f16 偶奇 / i8 平坦添字の罠の 4bit 版）。
- **MLC q4f16_1**（ブラウザ実運用 4bit）: group_size=32・8 要素/u32・
  `group_size % (32/bits) == 0` を強制。量子化は `max_int = 7`・`q = clip(round(w/scale) +
  max_int, 0, 14)` = **15 準位の対称量子化**（group_quantization.py:57-62,268-286 @2f78caa）—
  ADR 0019 の「±127 に閉じる」と同思想で、fake-quant 冪等性論証が移植できる。
  scale は f16（カルメの ADR 0019 は scale f32 必須 — 意識的に分岐する点）。

### 4.2 logical shape ≠ physical payload の表現 — 配置は 2 系統（+カルメの選択肢）

- **shape = 論理のまま、バイト数を型から導出**（GGUF/ggml）: テンソルは論理要素数 ne[] を
  保持し、`nbytes` は type の (blck_size, type_size) から導出（ggml/src/ggml.c:1296-1341 /
  gguf.cpp:668-733 — `ne[0] % blck_size == 0` を要求）。**safetensors 本家も同じ配置**で、
  sub-byte dtype を持ち検証は `nelements * bitsize / 8`（safetensors/src/tensor.rs）。
  精密化（第 2 巡）: 本家の `F4` は **MXF4 系の float であって任意 int4 ではなく**、GGUF の
  Q4 系は block scale 込みの複合 dtype — 「同配置」と言えるのは**論理 shape の扱いに限る**。
  カルメが bit 幅一般化を採る場合は、総 bit 数 8 の倍数制約・pack 順・行 / group 境界・
  scale テンソル形を**独自 dtype の契約として明文で固定**し、ADR 0063 の reader / writer /
  verify を同時改訂することが条件。
- **shape = 物理、論理は attrs / フィールドへ**（ORT / MLC）: MatMulNBits の
  `(N, k_blocks, blob_size)` + attrs K/N。
- カルメ側の実際の制約は「宣言 shape = 実テンソル shape」等式そのものではなく
  ①`DTYPE_BYTES` が整数バイト表であること（safetensors.ts:16-27,113-120）
  ②ADR 0063 の整列・並び順契約に sub-byte の端数規則が無いこと
  ③消費側 op 契約（shapes.ts:374-394 の linear 等）と `IrStorageDtype` 語彙 — の 3 点。
  **侵襲の小さい順に「bit 幅一般化（GGUF/safetensors 型・shape は論理のまま）」と
  「物理 shape 宣言（ORT 型）」の 2 案があり、ADR② はここを比較して選ぶ**（当初の
  「shape 等式を必ず破る」という同定は検証で refuted — §7）。
  ir-v1 の `storage.group_size` は語彙として予約済み（ir-v1.md:100-104）で席は既にある。

### 4.3 ADR 0019 reopen が明示すべき適用範囲

- 経緯の正確化: ADR 0006 は「素朴 RTN w4 不成立・group-wise w4 機構は低優先 backlog」
  （0006-quantization.md:45）だが、ADR 0019 は「w4 group 量子化 … 不採用確定。再測しない」
  （0019-i8-weight-execution.md:71）へ強めている。ただし
  research/2026-08-03-demo-w8-perf-recon.md:44 は −1.5〜+5.1dB と発話長短縮を **int4 group の
  実測**として記録しており、「0019 に測定根拠が無い」は成立しない。**reopen の論拠は
  「測定が 1 ファミリ（SBV2 voice）・1 時点のもので、対象と指標を明示せずに全 packed 4bit へ
  一般化している」に置く**。むしろ数値を「素朴 RTN」へ帰属させた 0006:45 の側が不正確。
- 旧測定と実運用形の差: scale 粒度（per-tensor/channel RTN vs K 方向 group=32〜128）・
  zero point（対称のみ vs optional 非対称）・準位（16 vs 15）・対象と指標（SBV2 音声 SNR /
  発話長 vs テキスト生成）。棄却基準「発話長の系統的短縮」は**採用済み w8 でも観測されて
  いる**（0029:53 — 198→196 フレーム）ため w4 固有の基準として再利用不可。
- docs/ir-v1.md:103・limitations.md:146 の「w4 不採用確定」記述は**現行コード
  （container.ts:290 の groupSize 拒否）と一致しており今は stale ではない** — 0019 reopen が
  裁定された時の同期対象。

## 5. shard ロードと admission（ADR③ の根拠）

### 5.1 shard streaming の実在形

- **tvmjs（唯一の「shard 単位で CPU を解放する」ブラウザ実装）**: 2 相 —
  相 1 = 全 shard を 4 並列で永続キャッシュ（Cache API）へ落とすだけ（RAM に載せない）、
  相 2 = 1 shard ずつ「cache から取得 → record 単位で CPU staging → GPU copy →
  **`await device.sync()` → `cpu_arr.dispose()`**」（tvm/web/src/runtime.ts:1374-1447
  @27c2e01）。**GPU 転送の完了待ちの前に CPU を解放しない**順序が明示されている。
  RAM ピークは「1 shard + record 1 本分」。
- **整合性検証はブラウザ勢の誰もやっていない**: tvmjs 既定経路に hash 照合なし
  （artifact_cache.ts:337-410 — manifest 側には md5 が実在する
  〈python/tvm/contrib/tvmjs.py:177,192〉が **JS ローダは読まない = 未結線**）。web-llm の
  `ModelIntegrity` は config / wasm / tokenizer のみで**重み shard の欄が無い**
  （integrity.ts:26-31 @90f6709）。llama.cpp の --check-tensors も NaN/Inf 構造検査。
  → カルメの「全ファイル sha256 をキャッシュヒット側含め照合」（hub fetch.ts:265-267,
  328-360）は明確な差分価値で、**shard 化で落とさないことが ADR③ の非交渉条件**。
- 反面教師: transformers.js は external data chunk を `Promise.all` で全量 Uint8Array 化
  （utils/model-loader.js:74-96 @353007b）— shard 化しても RAM ピークが下がらない形。
  カルメの現状（hub fetch.ts:411-421 + container.ts:91 の全量 ArrayBuffer）と同型で、
  **「全量保持」は hub と runtime の両側に焼かれており片側改修では消えない**。
- カルメ固有の再設計点: 重みアップロードループは「**await 禁止の同期区間**」が errorScope
  LIFO の交錯防止の根拠（executor.ts:512-517 の MUST NOT — 現物確認済み）。shard 単位の
  解放は await を要するため、この不変条件の再設計（scope の張り方）を ADR に正面から書く。

### 5.2 admission / メモリ見積り

- llama.cpp `llama_params_fit`: **no_alloc dry-run** で model / context / compute の
  3 カテゴリを投影し、margin 付き不等式で判定。診断出力に **unaccounted 欄**があり
  「見積りは絶対保証でない」ことを形式自体が認める（common/fit.cpp:55-131 /
  tools/fit-params/README.md）。vLLM は profile_run の実測型
  （gpu_worker.py:474-581 @7ea4b40）。
- **WebGPU では「空き側」が取れない**: 仕様は総/空き GPU メモリのメンバを定義していない
  （W3C WebGPU — GPUSupportedLimits / GPUAdapterInfo に該当欄なし。動機の断定は避ける）。
  web-llm も照合を諦め `maxStorageBufferBindingSize` の warn のみ（engine.ts:1155-1181 —
  `vram_required_MB` は静的注記で照合コード不在）。llama.cpp も free/total が取れない
  デバイスを「予算不明」として fit 対象から外す分岐を既に持つ（fit.cpp:117-127）。
  → カルメの estimator は「**必要側**のカテゴリ別合計（resident weights / KV+state /
  prepared backing / transients / staging）+ unaccounted 相当の診断」に留め、
  **判定の最終門は既存の out-of-memory errorScope**（gpu/device.ts:1244-1257）。
  no_alloc dry-run 型の「必要側投影」はそのまま移植できる（前例あり）。

## 6. 検収モデルの構造事実（補充レッグ・HF config.json 一次確定）

### 6.1 Gemma 4 E2B（google/gemma-4-E2B-it — ゲートなしで config 直接取得）

| 項目           | config フィールドと値                                                                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 層構成         | 35 層 = `layer_types` に sliding_attention 28 + full_attention 7（5 層ごとに 1 層 full・最終層 full）                                                   |
| sliding 窓     | `sliding_window: 512`                                                                                                                                   |
| KV 共有        | `num_kv_shared_layers: 20` — 末尾 20 層（0-index 15..34）は自前の K/V projection を持たず、同種 attention の直近非共有層の K/V を再利用（HF 公式 blog） |
| GQA            | `num_attention_heads: 8` / `num_key_value_heads: 1`（= MQA 8:1）・`head_dim: 256`                                                                       |
| RoPE           | full 層 = `{rope_theta: 1000000, rope_type: "proportional", partial_rotary_factor: 0.25}` / sliding 層 = `{rope_theta: 10000, rope_type: "default"}`    |
| PLE            | `hidden_size_per_layer_input: 256`・`vocab_size_per_layer_input: 262144`（per-layer embeddings — KV 系 state ではなくローダ / グラフ側の機構）          |
| context / 語彙 | `max_position_embeddings: 131072` / `vocab_size: 262144`                                                                                                |

- **ORT WebGPU EP の kv_empty コメント「Gemma4 layers 15-34」（§1.5）と
  `num_kv_shared_layers: 20`（35 層の末尾 20 = 0-index 15..34）が正確に一致** — R3 の
  設計対象が独立系統の一次ソース同士で相互裏付けされた。
- **RoPE が層種で異なる**（theta 100 倍差 + full 層のみ proportional + partial rotary 0.25）。
  §3.4 の「層別混在は mask だけの違いではない」が config で確定し、さらに強い形になった。
- 未確認 2 点: `global_head_dim: 512`（`head_dim` の 2 倍 — 意味論の一次説明なし・二次
  ソースの「full 層は unified K/V」推測のみ）・`use_double_wide_mlp: true`（意味未発見）。
  MoE なし・Mamba / conv / linear-attention 系 state なし（`text_config` に不在）。

### 6.2 MiniCPM5-1B（openbmb/MiniCPM5-1B — ゲートなし）

- 素の dense GQA Llama 構成（`architectures: ["LlamaForCausalLM"]`・24 層均一・sliding /
  KV 共有 / state 系フィールド不在）。同ファミリの MiniCPM4（InfLLM-V2 sparse）や
  MiniCPM-SALA（Lightning Attention hybrid）とは**別物**（README / モデルカードが明示対比）。
- GQA: `num_attention_heads: 16` / `num_key_value_heads: 2`（8:1）。
  **`head_dim: 128` は独立フィールド**（hidden 1536 / 16 = 96 と不一致）—
  head_dim を hidden/heads から導出してはならない罠。
- RoPE: `rope_theta: 5000000`・`rope_scaling: null`・context 131072・語彙 130560。

### 6.3 ADR への含意

- **R3（名前付き state スロット）は必須で確定**: Gemma 4 E2B だけで「sliding 512 容量
  （28 層・うち共有あり）/ full 容量（7 層）/ 自前 KV を持たない 20 層」の 3 種が混在する。
- **G3 は検収モデル両方に適用**: Hkv=1（Gemma 4 E2B）と Hkv=2（MiniCPM5-1B）のどちらも
  現契約（q/k/v の H 完全一致）では書けない。
- 検収の順序として MiniCPM5-1B が先に適する（複雑機構なしの純 GQA + 大 theta）。
  Gemma 4 E2B は PLE / KV 共有 / 層種別 RoPE の全部盛りで後段の検収に回す。

## 7. 敵対検証の台帳（load-bearing 主張 18 本）

| ID        | 主張の要旨                                                                | verdict              | 補足                                                                                                                         |
| --------- | ------------------------------------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| state LB1 | 論理長は GPU テンソル値 + indirect dispatch で鍵から外せる（ORT WebGPU）  | holds [high]         | 但し indirect は graph capture 有効時のみの経路（§1.1）                                                                      |
| state LB2 | llama.cpp は量子化済み論理 extent を鍵に入れる反例                        | holds [medium]       | 粒度は max(n_pad,256)                                                                                                        |
| state LB3 | KV は不透明 state オブジェクト + prefill/decode 別関数（MLC）             | holds [medium]       | —                                                                                                                            |
| state LB4 | 動的形状 KV は graph capture と非両立（二重強制）                         | holds [high]         | kv_empty（KV 共有層）だけが明示的除外                                                                                        |
| state LB5 | per-layer shape 差 + KV 共有層（Gemma4 15-34）の席が実在                  | holds [medium]       | R3 の直接裏付け                                                                                                              |
| state LB6 | 全語彙 readback を出さない decode 出口 + WebGPU argmax/static-k topk 実在 | holds [medium]       | top_k の k は構築時定数                                                                                                      |
| attn LB-1 | 「全実装が除算 + uniform・変更は 1 行」                                   | **refuted [medium]** | 除算方式は主流だが焼き込み定数 / kv-major 乗算 / repeat_kv フォールバックが現存。カルメ側は qk/pv の 2 枝（§3.2 は訂正済み） |
| attn LB-2 | mask 非実体化（TVM）/ 行数圧縮（llama.cpp）                               | holds [low]          | 「WebGPU は tensor を持たない」と一般化しない（ORT attention_bias が反例）                                                   |
| attn LB-3 | ORT の「静的な形は attrs・動的な長さは値」分割                            | holds [low]          | 引用行を実測値へ補正済み                                                                                                     |
| attn LB-4 | 「online softmax は一般に空行 0 を内蔵」                                  | **refuted [medium]** | 成立は llama.cpp WebGPU（真の −inf + 分母ガード）のみ。TVM の有限 sentinel は空行 0 にならない（§3.4 は訂正済み）            |
| attn LB-5 | 層別混在は同一 kernel + attrs・分岐は cache 側                            | holds [low]          | ORT `sliding_window_cache` が追加証拠                                                                                        |
| attn LB-6 | GQA 化は ADR 0060 行窓変種と同型（ビット同一論証の再利用）                | holds [low]          | 論証であって実測ではない（実装時に A/B 証明）                                                                                |
| wt LB-1   | 「packed 4bit は shape 等式を必ず破る」                                   | **refuted [medium]** | 物理 shape 宣言 / bit 幅一般化の 2 案が等式を保ったまま成立（§4.2 は訂正済み）。「別 ADR 級」の結論自体は生存                |
| wt LB-2   | 「上流総意 = 論理形は別欄」                                               | **refuted [high]**   | GGUF/safetensors は shape=論理のまま成立。配置は 2 系統（§4.2 は訂正済み）                                                   |
| wt LB-3   | 4bit unpack は core builtin で成立・blob は 4 整列                        | holds [low]          | single-scale 例外経路と pack 順契約の 2 条件つき                                                                             |
| wt LB-4   | tvmjs の 2 相ロード（sync 後 dispose）・hash 検証なし                     | holds [low]          | manifest に md5 はあるが未結線                                                                                               |
| wt LB-5   | WebGPU は空き VRAM 非露出 → estimator は必要側のみ                        | holds [low]          | no_alloc dry-run の投影側は移植可・非露出の動機は断定しない                                                                  |
| wt LB-6   | 0019 は 0006 より射程が広い（reopen の根拠）                              | holds [medium]       | 数値の帰属は 0019 側が正確（int4 group 実測）。論拠は「1 ファミリ 1 時点の一般化」に置く                                     |

（refuted 4 本という数え方をしないこと — wt LB-1 と LB-2 は同じ訂正の 2 面。**本文
§1〜§5 は全て verdict 反映後の記述**である。）

### 第 2 巡 — Codex 独立レビュー（2026-08-17・全件を一次ソースで再確認済み）

第 1 巡反映後の本文と ADR 決定案 4 件（執筆順 / G3 案 A / packed bit 幅一般化 /
1 本ずつ裁定）へ独立レンズで反証を依頼した結果。**新規 refuted 4**（全て confirmed →
本文訂正済み）+ **見落とし 4**（本文編入済み）。決定案 4 件はいずれも
**go-with-condition**（方向維持・条件は各節に記載）。

| 対象 | 指摘                                                                                              | 帰結                                                                   |
| ---- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| §1.3 | 「固定長 chunk 1 種に畳める」は過剰一般化（ORT GenAI 既定は全量一括・web-llm は可変長 chunk）     | §1.3 訂正。queryLength 独立スカラ + padding no-op 契約を ADR① の前提へ |
| §2   | ADR④ の変更範囲は 3 点でなく 6 点（recipe / recipe-builder / exporter 契約門も単一出力前提）      | §2 訂正。slot 別 dtype/shape 契約も範囲へ                              |
| §3.1 | 「[M,N] 実体化は皆無」は不正確（llama.cpp は現 step 行 × n_kv を prefill でも実体化）             | §3.1 の表現を精密化                                                    |
| §3.2 | 「変更 2 箇所」は f32 融合経路限定（i8a8 attention の head 基底 5 本 + recipe-builder が第 3 面） | §3.2 訂正。GQA × i8a8 の実装 / 明示拒否を ADR⑤ 裁定項目へ              |

見落とし（編入先）: ① GenerationContext × PreparedPlan の所有権分離（§1.6・high）
② queryLength の実行時スカラ化と padding no-op（§1.3・high）③ 保存 SDPA と行ブロック
matcher の未接続（§3.3・high）④ multi-output の slot 別契約（§2・medium）。
R2 の席の合格条件（workgroup 数 / 総反復 ∝ pastLength × queryLength）は §1.1 へ、
safetensors F4 = MXF4 の精密化は §4.2 へ編入。

## 8. 一次ソース

shallow clone（2026-08-17 取得・commit 固定）:

| リポ                        | commit  | 主な参照先                                                                                                                                                  |
| --------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ggml-org/llama.cpp          | 4df29be | src/llama-kv-cache\*・llama-graph\*・llama-context.cpp・ggml/src/ggml-webgpu/（WGSL flash_attn / argmax / argsort / quant）・ggml-common.h・common/fit.cpp  |
| vllm-project/vllm           | 7ea4b40 | vllm/v1/core/（kv_cache_utils・single_type_kv_cache_manager）・v1/worker/gpu/・model_executor/models/gemma3.py・quantization/                               |
| microsoft/onnxruntime       | dd64c8a | contrib_ops/webgpu/bert/（group_query_attention・flash_attention・attention）・quantization/matmul_nbits\*・docs/ContribOperators.md・js/web/lib/wasm/jsep/ |
| microsoft/onnxruntime-genai | f6a871d | src/models/（kv_cache・input_ids・model）・src/search.cpp・src/webgpu/interface.cpp                                                                         |
| mlc-ai/web-llm              | 90f6709 | src/llm_chat.ts・engine.ts・integrity.ts・config.ts                                                                                                         |
| mlc-ai/mlc-llm              | 2f78caa | python/mlc_llm/（nn/kv_cache・quantization/group_quantization・compiler_pass/）・model/llama/                                                               |
| apache/tvm                  | 27c2e01 | src/runtime/vm/paged_kv_cache.cc・python/tvm/relax/frontend/nn/llm/・web/src/（runtime.ts・artifact_cache.ts）                                              |
| huggingface/transformers.js | 353007b | packages/transformers/src/models/（modeling_utils・session）・generation/logits_sampler.js・utils/model-loader.js                                           |

Web（公式のみ主要分）: W3C WebGPU 仕様（device lost / limits）・WGSL 仕様 §17.10
（unpack4xU8 は core / dot4U8Packed は言語拡張）・huggingface/safetensors（tensor.rs の
sub-byte dtype）。検収モデルの HF config は §6 に記載。
