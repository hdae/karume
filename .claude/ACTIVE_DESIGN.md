# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
> 波順・作業項目の正本は [docs/backlog.md](../docs/backlog.md)、性能候補の採否は
> [docs/perf-ledger.md](../docs/perf-ledger.md)。ここは「今この瞬間の文脈」だけを持つ —
> 履歴・完了記録は ADR / research / git へ。
>
> Last updated: 2026-08-18

## Now

- **0.3.0 リリース済み**（2026-08-16・JSR 3 + PyPI・CI 緑）。ポジショニングの正本 =
  [research/2026-08-16-runtime-landscape.md](../docs/research/2026-08-16-runtime-landscape.md)。
- **autoregressive-ready 実装波: 進行中** — ADR
  [0066](../docs/decisions/0066-generation-context-state-slots.md)〜
  [0070](../docs/decisions/0070-shard-loading-admission.md) accepted・波割り A〜H 裁定済み
  （正本 = [backlog](../docs/backlog.md) next 節）。**波 A 済（2026-08-17）**: GQA 整除
  broadcast（`b78b0c1` — r=1 バイト同一・repeat_kv parity・i8a8×GQA fail loudly）+
  MiniCPM5-1B 1-shot recipe（`3f072cb` — 真の GQA 形 24 層・sanity greedy）+
  `e2e_minicpm5_test.ts`（tolerance 1e-3・greedy・census 全 `:gqa`）。**波 B 済
  （2026-08-17）**: 出力列化 2 段（`3a31544`/`9a795a7` — バイト不変）+ argmax（`cbe093a`）+
  topk（多出力の最初の入居者・値列 = torch と数値同値・添字列 = 最小 index 規範〈ADR 0068
  追記 2〉・k ≤ 63 既定上限）。**波 C 済（2026-08-17）**: states{} パーサ（`1d7bdbb`）+
  GenerationContext（第 5 寿命クラス・可変 uniform・poison/rewind/device-loss・診断席・
  計画鍵不変条件）。**波 D 済（2026-08-18・7 コミット `5662c9a`..`2d1cc60`）**: states 形
  attention + `state_append`（ADR 0067 決定 4〜7）— 契約層 TS/Py・カーネル族 4 本・実行統合
  （`run` 第 3 引数 generation）・分離焼き込み（ADR 0066 決定 5）・受入テスト群（帯 mask
  交差オラクル・0066 受入②・0067 受入⑤）。0 本席 / states 専用記号束縛 / C-2 結線点 5 件も
  全消化。**波 E 済（2026-08-18・6 コミット `4c0a587`..`e92bcbc`）**: core 手術ヘルパ
  `karume/states.py`（export 後の attention→states 形書換 — Gemma 4 でも使う汎用機構）+
  models `generateGreedy`（固定 chunk prefill / decode M=1・`maxPosition` 必須）+ MiniCPM5
  decode 台本（RoPE 表引き swap・`outputs/series/minicpm5-1b-decode/` staging 公開で実走済み・
  絶対位置上限 = 表 512 行）+
  実 GPU 検収門（**greedy 3 ケース × K=16 の token id 列が torch と厳密一致**・prefill logits
  maxAbs 7.439e-5 = 1-shot 門と同値・census 全 :gqa）。`enqueue` の generation 面は
  **設けない**裁定で確定（limitations — speculative 実需で再訪）。decode 出口の token-only
  既定形は先送り裁定（ADR 0068 追記 3 — backlog 4 番）。**波 F 済（w4 — ADR 0069・2026-08-18）**:
  Phase 0 = sweep 実測（`93ebaf6` — 既定 group 32 対称・zero-point 欄なし確定。w4 RTN は
  1B 級で品質が明確に落ちる実測 — 検収方法論には無関係）。Phase 1 = format 層（`39d6405`）+
  exporter 発行（`61483e2` — nibble pack・逆変換門・検出器 3 点・**書き出し順は整列降順
  F32,I32,I4,F16,I8 に訂正**〈ADR 0069 追記 2〉・verify 自前リーダ化）+ runtime 実行
  （`a690057` — 適格 = linear 限定・`:wi4g<N>` 変種・capability 両側開放・GPU 門 5 本）。
  **実モデル w4 検収は波 H（Gemma 4 E2B）で**。**次 = 波 G（shard + admission — ADR 0070）**。
  送り: L8 fake-device 注入面は保留継続。

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
