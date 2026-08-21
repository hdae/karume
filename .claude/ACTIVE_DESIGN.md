# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
> 波順・作業項目の正本は [docs/backlog.md](../docs/backlog.md)、性能候補の採否は
> [docs/perf-ledger.md](../docs/perf-ledger.md)。ここは「今この瞬間の文脈」だけを持つ —
> 履歴・完了記録は ADR / research / git へ。
>
> Last updated: 2026-08-21

## Now

- **0.3.0 リリース済み**（2026-08-16・**JSR 3 パッケージのみ** — PyPI `karume` は未リリース
  〈2026-08-20 実確認・旧記述「+ PyPI」は誤りだった〉・CI 緑）。ポジショニングの正本 =
  [research/2026-08-16-runtime-landscape.md](../docs/research/2026-08-16-runtime-landscape.md)。
- **autoregressive-ready 実装波 A〜H: 全消化（2026-08-17〜19）** — GQA 整除 broadcast・
  多出力 + argmax / topk・GenerationContext（第 5 寿命クラス）・states 形 attention +
  state_append・decode 台本 + greedy 検収・w4（i4 g32・linear 限定）・shard ロード +
  admission estimator・Gemma 4 E2B 検収（PLE 35 分割・KV 共有 30 slot・混成 i8+i4）・
  **token-only 既定出口**。設計の正本 = ADR
  [0066](../docs/decisions/0066-generation-context-state-slots.md)〜
  [0070](../docs/decisions/0070-shard-loading-admission.md)（各追記）、波の経緯と送り =
  [backlog](../docs/backlog.md) now 節。検収 = MiniCPM5-1B / Gemma 4 E2B とも実 GPU で
  greedy K=16 厳密一致（ring エビクト越え込み）・w4 混成の完全常駐（hostExpandedBytes 0）・
  chat デモ ~11 tok/s。
- **全体レビュー（2026-08-19）は修正波込みで全消化** — E/C = 0・W 19 → 修正 11 コミット・
  verify 1620/0/5。Codex 提案の波割りは backlog へ反映済み（R1 同席 4 件・生成 API 波・
  recipe 基盤同席）。
- **波 I（w4 横展開 + 方式スクリーニング・2026-08-19）: 聴感/視認込みで完全クローズ** —
  実測正本 =
  [research/2026-08-19-w4-method-screening.md](../docs/research/2026-08-19-w4-method-screening.md)・
  採否 = perf-ledger Q 節（RTN/NF4 全ファミリ一次通過・Q-4 mxfp4 不採用確定・Q-2 kmeans は
  LLM/埋め込み限定）。方式序列はモデル系統で割れる — 1 回の実測で一般化しない。既定化は
  速度と細かい品質のバランスで別途。
- **波 J（量子化探索・第 2 段・2026-08-20 着手）が現行**: 消化済み = J-1（`w8-bert4`）・
  J-1b（`w4`）・J-2 第 1 段（**GPTQ 大勝ち** — gptq-rtn は格納不変で RTN 全面超え = Q-6・
  gptq-kmeans 全列最良 = Q-2 価値上昇・AWQ = Q-7 ❌ 実装温存）・**J-5a（embedding i4 —
  ADR 0069 追記 6・i4 適格 = linear + embedding・w8-bert4 取得量 −30.33%・WAV 参照採り直し）**。
  J-2 第 2 段は**聴感/視認裁定込みで消化**（3 ファミリとも品質 OK → 採用確定）、第 3 段も
  **クローズ** — Q-6 出荷結線（deberta-i4 export へ gptq-rtn・格納形不変・WAV 参照採り直し）
  → 再聴「ほぼ違いが分からない」→ **SBV2 の既定 quant = w4 へ変更**（ADR 0039 決定 5 更新・
  速度 = 取得 −30%/ロード 1.7 倍速/温間合成 ~4% 速 — research §7・w8 は opt-in 参照系）。
  **J-5b（net_g conv1d i4）も聴感込みでクローズ** — ADR 0069 追記 7（scale rank2 一般化・
  gemm A 側 i4・適格 = conv1d ∧ groups==1 ∧ 行長整除）で **w4 = 237.5MB（w8 比 −36.3%）**・
  温間合成最速・聴感 = conv i4 で変化なし（f32 比の「テンション少し低め」は net_g RTN i4
  由来の想定内 — 品質ノブは g 軸のみ = J-3 の評価軸）。J-3（g 軸）→ J-4（格納席）は
  波 K の後へ。骨子 = backlog now 節。
- **波 K（リリース + 公開・2026-08-20 着手）はアップロードまで消化（2026-08-21）**: J-3 が
  重いため先にリリースを挟むユーザー裁定（release 節の部分先行）。①manifest **`karume/3`**
  shard 欄（ADR [0071](../docs/decisions/0071-manifest-v3-shards.md) — 公開前締切ぶんのみ・
  API 工事 4 件は残置）②SBV2 既定 quant = **`w8-bert4` へ再裁定**（ADR 0039 — w4 は
  テンション差が残るため品質優先・w4 は opt-in・丸め方式はカード備考へ）③トーン注入席（ADR
  [0072](../docs/decisions/0072-sbv2-text-injection.md) — overlay + given_tone）④**HF 公開済み
  = jvnv / irodori / anima の 3 リポ**（公開前レビューで irodori カードの陳腐化量子化前提を
  修正の上アップロード・断片化検証 9/9 健全。**FN は公開保留** — 再配布の書面根拠なし・
  backlog parked）⑤**pin 焼き込み済み**（ADR
  [0073](../docs/decisions/0073-models-source-pin.md) — 3 定数 + ref optional 化）+ 0.4.0
  lockstep bump 済み。**CI 緑 → GitHub Release → JSR publish まで通過（2026-08-21 ユーザー
  確認）= 波 K クローズ**。手順の正本 = [release-runbook](../docs/release-runbook.md)。
- **波 J-4a（anima の i4 席・2026-08-21 着手）が現行**: J-4（格納席の実装裁定）から anima
  だけ切り離す裁定を受けた先行波。第 1 段の速度実測は消化 — 正本 =
  [research/2026-08-21-anima-i4-seat-speed.md](../docs/research/2026-08-21-anima-i4-seat-speed.md)。
  i4 系列 + `w4` / `w4-a8-s16` を新設し、**GPTQ 校正付き**へ結線（適格 = 型 ∧ g32 整除の
  一般形のみ・DiT の linear 453 本が i4・patchify 入口 1 本だけ i8・校正は block 内 448 本）。
  取得量 −21.2%（2.74GB）・VRAM −22.6%（2,637MiB）・DiT 1,640ms/step。視認裁定で
  **`w4-a8-s16` を低 VRAM 席として採用**（品質重視 = `f16` / 速度重視 = 既定 `w8a8-s16` 据え置き）。
  **w4a8**（i4 常駐を整数内積へ — ADR [0076](../docs/decisions/0076-w4a8-linear-execution.md)）は
  runtime に載り 955ms/step まで戻すが、**画の細部が荒れるので anima の席には宣言しない**
  （`distribution.py` の MUST は理由ごと差し替え済み — 旧「効かないから書くな」→ 新「効くが
  品質で採らない」）。**残 = 公開可否だけ（保留）**。未検証で残した可能性は
  [research §8](../docs/research/2026-08-21-anima-i4-seat-speed.md)。
- **SBV2 注入席の再調整（2026-08-21）= 利用実装フィードバックへの対応**: VOICEVOX ENGINE 互換
  サーバー側からの 8 項目に対応し、ADR
  [0072](../docs/decisions/0072-sbv2-text-injection.md) に**追記（決定 4〜8）**。
  ①下書きを句 / モーラ構造（`Sbv2Prosody`）で往復させる — 派生欄は同階層に置かず、門は
  **音素列の内容一致**（長さ検査では梱包規則のズレが素通りする）②`overlay` は解決済み
  `OverlayDictionary` も受ける③入力起因の失敗は `Sbv2InputError`（内部不変条件の破れは素の
  `Error` のまま = 400/500 の分離）④`analyzeProsody` を直列化鎖の外へ（辞書は**値でなく
  Promise** を持ち、失敗は捨てる）。**音素数が変わる編集は受けない**裁定（決定 8）—
  `adjust_word2ph` は移植せず backlog parked。疑問形の上げは表現不能（limitations）。
  **0.4.1（追加のみ・配布形は `karume/3` のまま）で JSR publish まで通過（2026-08-21 ユーザー
  確認）= この波はクローズ**（手順の正本 = [release-runbook](../docs/release-runbook.md) §4）。

## Open decisions

- MiniMax-H3（動画生成・オープンウェイト 33.1B/42.5GB 級）は遠期の関心として記録のみ —
  ブラウザ実行はメモリ規模的に現行スコープ外（レビュー DS-4）。

## Pitfalls（現役のみ）

- **Metal**: threadgroup `vec4` への動的インデックス書きは黙って捨てられる（`gemm.ts` の
  `storeBTransposed` の switch 展開を新しい箇所で崩さない）。attention i8a8 / conv2d の
  Metal 数値差は known-issues・Metal は gpuTiming 不可（limitations）。
- **融合 matcher は実測形 exact-match** — exporter の発行順・形が変わると黙って外れ、値は
  正しいまま性能だけ落ちる。観測 = `Diagnostics.lastRunFusions` +
  `assets_fusion_counts_test.ts`。**row-block だけは外れ方が性能でなく資源** — 128MiB 級
  device で resource-limit failure に戻る（**分解経路の matcher だけの話** — 保存 attention の
  states 形は行ブロックを op 内蔵で持つ・ADR 0067 決定 7・波 D 済）。
- **RoPE / SiLU 融合の丸め障壁（workgroup memory 往復）は実測依存** — バックエンド更新で
  PNG 門が割れたらまずここを疑う。
- **`deno task verify` はリポ内に worktree を置くと worktree 側まで test を拾う** — worktree は
  リポ外に作る（CLAUDE.md 検証コマンド節。deno.json に exclude は設けない — 2026-08-16 裁定）。
- **Session 構築の重みアップロード後 submit 1 回は瞬間ピーク +2.7GiB を抑えている** — 消さない。
- **資産の置き場**: `models/` = HF へそのまま上げる配布形のみ・系列出力は `outputs/series/`・
  入力素材は `inputs/<ファミリ>/<名前>/` — 綴りの正本は
  `tools/export-recipes/_shared/paths.py` と [assets-layout](../docs/assets-layout.md)。
  格納 dtype はヘッダが正（dist の門が検査）。旧識別子以前の資産は開けない（互換シム無し）。
- models パッケージの tree-shaking は「全モジュール副作用ゼロ」不変条件が前提。JSR npm 互換層の
  `sideEffects: false` 出力は未検証（backlog release）。
