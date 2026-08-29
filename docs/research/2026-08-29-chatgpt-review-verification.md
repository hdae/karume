# ChatGPT レビュー 4 本の検証波 — 実測記録（2026-08-29）

> 性格: **時点スナップショット**（HEAD `3ec3d45` = v0.7.0 直後・RTX 3080 Ti / Vulkan /
> Deno 2.9.6）。外部レビュー 4 本（`.claude/reviews/2026-08-29_chatgpt-reviews/` — git 追跡外）
> の統合指摘 13 件を検証した際の実測値の正本。裁定の帰結は perf-ledger（H-8〜H-10 / L-7 /
> L-8）・ADR 0058 追記・known-issues に反映済み。

## 検証の全体結果

19 判定 = holds 18 / refuted 1。修正波（波 1）で消化したもの: pipeline errorScope の internal
取りこぼし・recipe 宣言の静的検証穴・safe-integer 門・regcache epc≥2 parity 門・run/enqueue
入力の borrowed 契約・SubmitScheduler の overshoot 観測席・attention dp4a カナリア（ADR 0058
追記）。以下は**起票のみ**に裁定した候補の根拠実測。

## SubmitScheduler（H-8 の根拠）

- 1 workgroup あたり実コストは実グラフで **2.9ns〜19,470ns**（anima・約 6,700 倍差）、
  irodori は 2.6ns〜648,448ns — 「workgroup 数 = 仕事量」の均質仮定は桁で破綻している。
- しかし**窓（= run 1 本）平均は同一 Session 内で ±6% と安定**（anima DiT 4 step で
  25.1/23.8/23.7 ns/wg）。推定は Session 寿命（= 1 generate）に閉じ、軽い形状で学習して重い
  形状へ適用する経路は出荷 3 パイプラインでは発生しない（512px→1824x1248 の連続 generate で
  持ち越し無しを実測）。
- チャンク実時間 / 推定の比の実測最大 **1.33**（anima 1824x1248 DiT・実 65.9ms / 推定
  49.7ms）— 安全率 0.5 の 2 倍枠の内側。予算超過チャンクは全窓で 0 本。
- 理論上限は非有界: 最重量キー（attention_pv i8a8 1,670ns/wg）だけで 1,751ms の仮想チャンク
  （maxChunkSize 1024 律速）が組める — 現行グラフの並びに依存した「たまたま安全」。
- レビュー評者間の不一致（P1 vs P3）は **P3 側（観測されてから）を採用**。観測席
  （`ChunkBudgetStats`）は波 1 で実装済み。
- 隣接発見: **irodori は全 Session が 1 窓しか生きないため適応制御が一度も発火せず**、DiT の
  窓 95,760 dispatch が initialChunkSize=16 のまま約 6,000 submit に刻まれている（TDR 安全側
  だがホスト律速の一因 — [host-cost-decomposition](2026-08-13-host-cost-decomposition.md) の
  「irodori はホストが壁」と符合）。

## readback staging（H-9 の根拠）

- MAP_READ バッファは unmap 後に再利用可能（実 GPU で 3 周実証 — `arena.ts` の旧コメント
  「使い回せない」は事実として誤りで、波 2 で訂正予定）。
- createBuffer(MAP_READ)+destroy の固定費 **13.5µs/本**。run 全体（1 出力 11.50ms）に対して
  0.1%。23 出力の合成でも pack 化の利得 0.6ms/run（4.9%）— 主力グラフは出力 1 本方針
  （ADR 0045）なので実利はほぼ無い。
- ADR 0045 の 4.1〜7.7% は出力本数・readback バイト量・mapAsync 本数が同時に動いた測定で、
  staging 生成固定費に帰属できるのは最大 0.3ms（改善 3.0ms の 1 割）。ADR 自身も「壁時計への
  寄与は小さい（0.3%）」と結論しており、共有 staging の性能根拠には使えない。
- 共有化の設計障害: 失敗経路（device 消失が mapAsync に勝った場合）で共有 staging が pending
  のまま恒久故障する（次 run の mapAsync が OperationError — 実測）。復帰規律（エラー時は
  破棄して作り直す）が必須。

## 発行 metadata の毎 run 再構築（H-10 の根拠）

- LLM decode 相当の小グラフ（入力 3 / symbols 2 / state 2）で、bindSymbols 0.888µs +
  #preparedKey 相当 0.522µs + stateShapes/slots Map 0.287µs + statesOnlySymbols 0.180µs =
  **消せる上限 ≈1.7µs/run**。フェンス待ちの床 ≈11ms/run（perf-ledger H-2）の **0.015%**。
- 台帳の先例（[op-timing-stats](2026-08-10-op-timing-stats.md) §8 — 桁で大きい plan/fusion/
  WGSL 生成を丸ごと飛ばしても大型 DiT の壁時計は中立）と整合。速度施策としては成立せず、
  GenerationProgram 波の中の整理として扱う。

## WGSL 生成の eager 評価（L-7 の根拠の一部）

- `cache.get(key, wgsl)` の call site は 41 箇所（全て recipe-builder）。生成関数の単価は
  matmulWgsl 24.0µs / conv2dWgsl 0.3µs / stateQkWgsl 0.4µs（2000 回平均）— prepared miss ×
  pipeline hit のときだけ発生し、形状バケットごとに初回 1 回。thunk 化単独では「同一 key →
  バイト同一 WGSL」の実行時ガードを失うため不採、PipelineSpec（key と source の一体化）と
  セットでのみ扱う。

## createComputePipelineAsync（L-7 の根拠）

- Deno 2.9.6 + Vulkan で存在・動作を実証: 失敗は `GPUPipelineError`（reason: "validation"）の
  reject で返り errorScope へ dispatch されない。並行 8 本 22.0ms。
- **観測性の罠**: 不正 WGSL のとき GPUPipelineError の message は「ShaderModule is invalid」
  だけで、行番号つき診断は createShaderModule 側の errorScope か `getCompilationInfo()` に
  しか出ない。素朴に移行すると診断が退行する — 移行時は module 側スコープ維持 +
  getCompilationInfo で補完が条件。
- reason: "internal" は本機では再現不能（未検証のまま）。

## SessionBuildStats の初回実測（L-1 の分解材料 — 2026-08-29 波 2 で席を実装）

テスト fixture（4 格納混在・重み総量 800B・3 shard）での採取:

- shard 面: `shardWaitMs 0.08-0.17 / decodeMs 0 / bufferCreateMs 0.19-0.26 /
  writeBufferIssueMs 0.15-0.34 / uploadedBytes 800 / uploadFenceMs 33.3-34.1`
- 全量面（1 shard）: `uploadFenceMs 11.1`・適格外変種（f16 CPU 展開）: `decodeMs 0.017-0.039 /
  uploadedBytes 1056`
- 読み: **フェンス待ちが shard 本数にほぼ線形（≈11ms/本）で単独支配的**（ホスト費用 4 席の
  合計 0.4-0.8ms に対し 1:50 超）。L-1 の 2.50s も「shard 本数 × フェンス固定費 + 実転送」が
  主項という**仮説**が立つ（この席群では実転送とキュー待ちは分離できない — 帰属の限界は型の
  docstring どおり）。次の材料 = 実モデル（anima base 4 shard 等）での buildStats 採取。
- decodeMs は 256B の f16 展開でも 0.017ms 以上を安定して刻む（`performance.now()` 分解能に
  埋もれない — テスト門はフレークしない）。

## attention dp4a カナリア（ADR 0058 追記の実測）

- 6 変種 1 submit の判定コスト: 初回のみ **39〜55ms**（同一 GpuContext 3 連続で 54.5 / 38.6 /
  41.0ms — メモ化で 2 回目以降 0）。

## Apple M2 実機検証（2026-08-29・Deno 2.9.6 — ユーザー実行・ssh）

カナリア初版（厳密一致のみ）は M2 で「両腕とも既知解を外した」の GpuFeatureError を出した。
数値の突合で機序が確定:

- カナリア QK v4: dp4a 実測 `28.433290481567383`・emu 実測 `28.433290481567383`（**両腕
  ビット同一**）・既知解 `28.43328857421875` — 差はちょうど **1 ULP**（1.9e-6）。
- 既存テストの赤も全て GPU vs TS 参照の 1 ULP: QK `12.012408256530762` vs
  `12.012409210205078`（9.5e-7）・PV `-0.199601411819458` vs `-0.1996014267206192`
  （1.5e-8）。**PV の dp4a-vs-emu 相互一致テストは M2 で緑**・qP（整数段）は 62,088 要素で
  不一致 0%。
- 結論: 整数段は厳密・故障は**両変種共有の f32 エピローグ（scale 適用）の丸め**が TS 参照と
  1 ULP ずれること。仮説（未確定・推測）: naga → MSL の FMA 契約差。known-issues の旧推論
  「QK / PV では実際に違う値が出ている」（変種間不一致）はこの検証で撤回。
- 故障注入（dp4a WGSL 破壊）の dp4a 実測 `28.435983657836914` = 参照比の相対誤差 ≈9.5e-5 —
  判定則 v2 の sanity 帯（1e-5）はこの間（1.2e-7 ≪ 帯 ≪ 9.5e-5）に置いた。
- 判定則 v2 で M2 は「両腕帯内・ビット同一 → dp4a + 警告」となり a8 が復活。M2 の残る赤は
  従来からの既知 6 本（attention parity 4 + conv2d 2 — GPU vs 参照の 1 ULP / 未特定）に戻る。
- **3 回目の実走（v2 + v2.1）で確定**: カナリアテスト 15/15 全緑・a8 Session 系全緑・警告は
  device 単位 1 回（分岐 band-both-identical・最大誤差 1.53e-5・帯余裕 0.018）。封じ込め完了。
