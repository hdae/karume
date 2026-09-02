# gemma4 E2B — プロンプト長 P に対する decode / prefill の実測（可変 capacity 波の設計材料・2026-09-03）

> **性格**: 時点スナップショット（2026-09-03・RTX 3080 Ti 12GB / Linux / Vulkan (wgpu) / Deno 2.9.6）。
> ユーザー意向「コンテキスト窓は可能な限り伸ばす」（2026-09-02 裁定: RoPE 表は TS 正本でホスト生成し
> capacity を実行時ノブへ・メモリ見積りも結線）の前提となる P 依存の実測。各 P 1 回。
> 資産と台本は git 追跡外（下記）。

## §1 方法

- 現行ミラー `models/karume-gemma4-e2b`（capacity 1024）は P ≤ 992（予算 `P + N − 1 ≤ capacity`・N = 32）。
  P > 1024 は **RoPE 表 4 本だけを差し替えた計測専用配布形**
  `outputs/bench/karume-gemma4-e2b/2026-09-02_ctx-sweep/pos16k/`（maxPosition = capacity = 16,384）。
  再 export はしない（`export_series` は GPTQ の丸め解が走るたびに動き出荷バイトと比較できない）。
  表は `Gemma4TextRotaryEmbedding` を config から立て recipe の `build_rope_table` で生成、先頭 1,024 行が
  出荷バイトと sha256 一致・他 831 テンソルは全ビット同一（parity の根拠）。`pos128k/` も同手順で生成済み
  （131,072 行・full 表 256 MiB は piece 2 本）。
- 台本 = scratchpad `ctx-sweep/sweep.ts`（消える）: 合成プロンプト（同じ文の繰り返し）を P トークン → greedy で
  N = 32 トークン decode。`timing off` = 壁時計（decode は最初の 2 token を除く中央値）、`timing on` =
  低レベル面（`buildGemma4Program` 相当）で `Session.diagnostics()` の op 別 GPU ns（P=200 で公開面と
  token 列一致を確認済み）。timing on は timestamp の代で decode 壁が 3 倍になるので、時間は off・内訳は on
  と読む。

## §2 結果

|      P |  prefill（off） | decode 中央値 ms/token（off） |    decode GPU ms（on） | うち attention_state（全層） | 割合 |
| -----: | --------------: | ----------------------------: | ---------------------: | ---------------------------: | ---: |
|     26 |          0.85 s |                          32.3 |                   29.4 |                          4.2 |  14% |
|    256 |           5.2 s |                          35.6 | 80.4（外れ値・要再測） |                         20.4 |  25% |
|    992 |          19.2 s |                          40.2 |                   32.0 |                         11.7 |  37% |
|  4,096 |            75 s |                          50.0 |                   36.6 |                         18.2 |  50% |
| 16,000 | 326 s（5.4 分） |                          83.1 |                   68.1 |                         49.2 |  72% |

decode の内訳（P = 16,000・GPU 68.1 ms）: full 層（`attention_state_{qk,pv}:…:gqa`・3 層）が attention の
大半で、sliding 層（窓 512）は P に依らない。prefill 側（P = 16,000・1 chunk 259.7 ms）も full 層の
attention が 139.6 ms（qk 66.3 + pv 73.4）で linear の 96.9 ms を上回る。

VRAM（実測欄・重み 1,471 MiB は別）: P ≤ 992 = RoPE 6 + full KV 12 + sliding 12 MiB / pos16k = 96 + 192 + 12。
C=131,072 の見積り = 768 + 1,536 + 12（計測専用配布形 pos128k で確保可能なことは確認済み・掃引は未実施）。

## §3 所見

1. **decode は P に対して緩やかに伸び、P=16K で P≈26 の 2.6 倍**（32 → 83 ms/token）。P=4K までは
   +55%。full 層 3 本の KV 読みが P に線形で、attention の割合は 14% → 72% へ。
2. **full 層の attention は帯域律速ではない**（推測ではなく算術）: P=16,000 の full KV 読みは
   16,000 × 12,288 B ≈ 197 MB / token で、帯域 500 GB/s なら 0.4 ms。実測 49 ms はその 100 倍で、
   カーネルが KV 長を逐次に走る形（1 invocation が S の 1 要素・D の逐次内積 — `state-attention.ts`）の
   **レイテンシ露出**が支配。K-11（linear の GEMV 化）と同じ形。**KV 量子化（TurboQuant 系・L-10）は
   バイトを減らしても時間にならない** — 着手条件（帯域律速）は不成立で、レバーは attention カーネルの
   KV 長方向の並列化（split-K 相当）にある。
3. **prefill は chunkLength 32 のまま P² に効く**（P=16K で 5.4 分・P=4K で 75 s）。chunk ごとに full 層が
   過去 KV を全部読み直すため。capacity を製品で上げるなら chunkLength の引き上げ（export の Dim 上限
   768 まで）が前提で、それでも P=128K は時間の壁（推定 30 分級）。
4. **VRAM は問題にならない**（16K で +300 MiB・128K で +2.3 GiB）。天井は時間の側にある。
5. P=256 の timing on の decode 総和（80.4 ms）だけ他の点と不整合（外れ値 — 初回のパイプライン生成が
   混ざった疑い）。設計の判断には使わない。

## §4 可変 capacity 波への含意

- capacity を実行時ノブにする設計（RoPE 表の TS 正本ホスト生成 + `estimateSessionMemory` 結線）は
  そのまま進めてよい。VRAM 面は 128K まで見積りで足りる。
- 製品としての実用上限は decode 速度と prefill 時間で決まる: 現行カーネルでは **P≈4K が「decode +55% /
  prefill 75 s」の線**、16K は「decode 2.6 倍 / prefill 5 分」。上限を伸ばすには
  ①chunkLength の引き上げ（prefill の P² を緩める）②full 層 attention の KV 長方向の並列化
  （decode の線形項を潰す — 新 perf 項）の 2 本が要る。
- KV 量子化（L-10）は上の 2 で条件が成立してから再評価（現状は kill 線②「latency 律速」に該当）。
