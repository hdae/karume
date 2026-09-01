# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
> 波順・作業項目の正本は [docs/backlog.md](../docs/backlog.md)、性能候補の採否は
> [docs/perf-ledger.md](../docs/perf-ledger.md)。ここは「今この瞬間の文脈」だけを持つ —
> 履歴・完了記録は ADR / research / git へ。
>
> Last updated: 2026-09-01

## Now

- **モデル更新波（2026-09-01 裁定 — 次の実装波・詳細は [backlog](../docs/backlog.md) now の
  「モデル更新波」）**: N1 Irodori v4.1-small（MIT・duration predictor のみ変更 → 新リポ +
  pin 追加）→ N2 Anima 再構造（公式 3 変種を karume-anima 同居・**defaultModel =
  anima-turbo-v1.1**・wai/copycat は karume-anima-extra へ移設 + 越境参照・**breaking =
  `ANIMA_TURBO_CURRENT` 廃止**・ライセンス = CircleStone NC v1.2 一次確認済み = 条件付き
  再配布可）→ N3 Civitai AIR 取り込み機構。完了で HF 一括リリース。その後 =
  メモリ管理波 → export-recipes 切り出し（案 A = レシピのみ・core 残留）。
- **OP 数値レビュー波 + 修正波（2026-08-31 — 波クローズ・Mac 検証込み）**:
  Codex 全滅→Opus サルベージ + ChatGPT 外部レビューを敵対検証で統合（C 0 / E 0 確定・
  台帳 = [research/2026-08-31-op-numerics-review.md](../docs/research/2026-08-31-op-numerics-review.md)・
  原本 `.claude/reviews/2026-08-31_b35cf5c/`）。修正 12 コミット消化:
  **gru_scan tanh_stable 化**（is_nan_bits/nan_max 正本化・キー v2）/ **softmax 族 nan_max
  統一 + 融合 attention 空行ガード**（breaking 2 点 — ADR 0044 追記・limitations が消費側 doc）/
  **DEFAULT_TOLERANCE 退役 = op 別実測表 + ビット同一門**（M1 宿題消化 —
  [research/2026-08-31-op-tolerance-measurement.md](../docs/research/2026-08-31-op-tolerance-measurement.md)・
  表 = `tests/helpers/op-tolerance.ts`・表に無い op は fail loudly）/ 飽和域の厳密カナリア /
  L 群軽微修正 + fma 契約 doc 訂正（ADR 0076 追記）/ exporter act_quant 鏡像化 /
  **GPTQ 掃引軸**（opt-in 実装 + 実測 → **既定現状維持を封印** —
  [research/2026-08-31-gptq-axes-sweep.md](../docs/research/2026-08-31-gptq-axes-sweep.md)）。
  **Mac（M2）検証も消化して波クローズ（2026-08-31）**: 飽和域門・格子門・vowel golden 全緑 =
  根治実証。新規 known-issue = gru_scan 分解 parity の Metal 1〜64 ULP（変更前 HEAD と同一署名 =
  既存・known-issues Metal 節）。GPTQ 掃引軸は**既定現状維持で確定**（復活 = 多モデル ×
  校正量 16×）。プローブ群削除済み。
- **LLM（gemma4 E2B）トラック（2026-08-30〜31）**: 先行波（L-0/L-1/L-10）クローズ →
  **K-11 も消化済み**（ADR [0082](../docs/decisions/0082-linear-gemv-decode.md) — decode M=1 の
  i4 linear を GEMV 族へ・**ビット同一のまま ×8.45・decode 84.2→32.5ms/token = 30.8 tok/s**。
  律速はフェンス床 ≈11ms 側へ戻った — 機序と方法論は
  [research 2026-08-30](../docs/research/2026-08-30-gemma4-decode-wallclock.md) §7）。
  **生成 API 波は段 0〜5 完了 = 波クローズ（2026-08-31〜09-01）**: 設計正本 = ADR
  [0083](../docs/decisions/0083-generation-api-surface.md) /
  [0084](../docs/decisions/0084-gemma-tokenizer-chat.md) /
  [0085](../docs/decisions/0085-ple-host-gather.md) + ADR 0068 追記 6。**gemma4 の chat が
  文字列 in → 文字列 out で実重み完走**（`Gemma4Pipeline.fromAssets` → `chat()` → streaming
  detokenizer・温度 0 golden 固定・EOS `<turn|>` 自停止・多ターン日本語込み）。公開面 =
  barrel + `./gemma`（**breaking 1 件 = `generateGreedy` の公開 export 削除** — limitations が
  消費側 doc）。製品系列 = `outputs/series/gemma4-e2b-product/`（容器 1,512MiB + PLE sidecar
  2,275MiB・GPU 常駐実測 1,504MiB）。**段 5（配布形）も 2026-09-01 に消化 = 波クローズ**:
  配布ミラー `models/karume-gemma4-e2b/`（3.8GiB）+ カード（Apache 2.0 帰属 + 上流誘導 —
  ライセンス門は現物確認 + ユーザー裁定で消化・「Gemma ToU」記述は撤回）+ `fromPretrained`
  （hub 遅延資産席 `eagerAssets`）+ 疑似 HF 疎通。**残 = HF 公開のみ（新規リポ作成 —
  ユーザー確認待ち・pin `GEMMA4_CURRENT` は公開時に焼く）**。
  **対話 example 波もほぼクローズ（2026-08-31・10 コミット e7e53dd〜dce91c7）**: 公開面レビュー
  （Opus2+Codex3 → 敵対検証 15 判定 refuted 0 — 正本 `.claude/reviews/2026-08-31_182ced7/`）→
  **取得元抽象 DistributionSource**（ADR
  [0086](../docs/decisions/0086-distribution-source.md) — `denoDirectory` でローカルミラー直読・
  CacheStorage 複製ゼロ・越境は明示 mapping・`@karume/hub/deno` carve-out）+ 検証済みバグ修正
  （PLE 解放口 / 生成 signal の PLE 伝播 / headers doc 8 家族訂正ほか）+ 公開面調整
  （`gemma4ChatTurn` 増分描画 = turn-local 契約の正本化・`parseGemma4PipelineConfig` 公開・
  `GenerationProgram` 凍結絞り込み・`defaultSampler` 改名・`used` + `GenerationCapacityError`
  構造化欄・`GenerationStop.tokens`）+ `examples/gemma4/` 対話 chat（sequence KV 継続の
  写経見本・デモ 4 本も denoDirectory 移行）。**残 = capacity 1024 の dist 再生成**（P 依存
  実測と同時）。実行計画は
  [backlog](../docs/backlog.md) now、候補比較は
  [research 2026-08-31](../docs/research/2026-08-31-generation-api-design-draft.md)。
  **スケール戦略は調査 + 裁定とも完了** =
  [research 2026-08-31](../docs/research/2026-08-31-freetoken-moe-over-arraybuffer.md)
  （FreeToken 中核は WebGPU へ移植不能 / 真の壁は VRAM 総量 / MoE 動的常駐は IR 語彙級の
  再設計）→ 4 分岐は①IR 値依存実行選択 = 入れない（backlog parked・MoE は全 expert 常駐前提を
  limitations へ恒久記載）②未着荷 initializer 席 = ①従属で見送り③companion scale f16 =
  perf-ledger Q-10 起票のみ④admission 空き比較 = 現状維持（ADR 0070 のまま）。L-11 裁定済み:
  技術先行 = gemma4 E2B・公開はライセンス門（ADR 0065 stage 6）後で、**配布対象は gemma4 のみ**
  （裁定 10 — minicpm5 の配布経路先行は不採用）。
- **全体レビューの修正波 A〜E クローズ（2026-08-30）**: 網羅レビュー（Opus 15 + レンズ 2 →
  敵対検証 → Codex / ブラウザ第 2 波）の確定 50 件（E4 / W46）+ 追補を 5 波で全消化。裁定と
  台帳の正本 = `.claude/reviews/2026-08-29_9614ba9/`（git 追跡外）。**破壊的変更 2 件**
  （`BatchScope.finish()` のホスト側失敗 throw / 同一 `GenerationContext` への並行発行拒否）と
  **`fromAssets` の shard 分割形受け口**（追加）は [limitations](../docs/limitations.md) が
  消費側 doc。レンズ E-1 / E-2 / L-11 は裁定済み（backlog now に反映）。M2 実機の手動確認
  2 点は**消化済み（2026-09-01 実測 — カナリア 16/16 緑・軸 reduce パリティ 2/2 緑。新規 =
  gemv u32 門の 1 ULP は既定維持を裁定済み — ADR 0082 追記 1）**。残り = anima-web DL
  スロット改善（`FamilyAdmission` 席は実装済み）。残件と隣接発見の一覧 =
  [backlog](../docs/backlog.md) now。
- **R1 統合波はコード完了（2026-08-29）**: API 工事 4 件（union/プランナ・`ModelShard` 実名
  帰属・`prepareModel` 2 段境界・`AdmissionReport`）+ hub `prefetchAssets` + models 7 pipelines
  の graph-first 接続 + exporter 1GiB 分割（`karume.shards`）+ デモの疑似 HF サーバ化。
  **受け入れ実証済み**: Base f16 3.9GB → 4 shard で実ロード + 生成完走。正本 = ADR
  [0070](../docs/decisions/0070-shard-loading-admission.md) 追記 2026-08-29。
  **HF 更新系まで完了（2026-08-29）**: 全席分割ビット同一・base / turbo とも公開済み
  （**公開 revision の正本は pin 定数** `ANIMA_CURRENT` / `ANIMA_TURBO_CURRENT` — 値は
  `packages/models/src/anima/config.ts` を見る。docs に SHA を写さない）・**shard ごとの
  越境参照**（ADR 0038 §7 追記 2026-08-29）初適用・pin 焼き込み + 実 DL
  疎通済み。実資産テストとローカル配信は越境 + 分割ミラーへ追随済み（fromAssets の実 GPU
  生成経路は anima e2e から消滅 — 契約面と他家族が担保）。**0.7.0 は Release → JSR publish
  まで完了（2026-08-29・リリースノートは検証ワークフロー通過済み）**。
- **fetch-cache 0.5.0 追従（2026-08-28 実装済み）**: 検証責務を取得層へ移譲（記録ハッシュ
  信頼・knob なし）・認証隔離の撤去・`verifying` 撤去・旧名前空間 purge。正本 = ADR
  [0080](../docs/decisions/0080-hub-fetch-cache-050.md)（旧 CAS ドラフトを置換 —
  `archive/hub-cas-0.5.0` の再適用は不要）。anima-web への追随 3 点（`verifying` 消滅・
  sha256 不一致が `HubFetchError`+`cause`・`clearHubCache` の対象拡大）+ R1 分の 1 点
  （`StreamedAsset.path`→`id`）。
- **既知問題 3 件 + anima 素版 i4 感度の波（2026-08-25〜28 消化済み — 残 = Pixel 実機
  `err.cause` 再判定のみ）**: ①Pixel の
  BodyStreamBuffer abort — 真因マスキング解消 + バイト予算 1.5GiB + 検証直列化**コミット済み**
  （実機再判定はリリース後 — known-issues）②NVIDIA の 2GiB 天井 — 融合 attention の
  行ブロック化**コミット済み**（1824×1248 実生成 27.9s 完走を確認）③Chromium の単一
  ArrayBuffer 上限 2,145,386,496B で Base f16 が原理的に不可（limitations 恒久記載。根本 =
  R1 shard 配布を next へ昇格・DL 前即エラーは hub 側では実装しない — プローブは
  fetch-cache 次版で実装予定〈2026-08-25 裁定〉）④素版 i4 — adaLN-i8 変種は**不採用**・**i4 系は保留**（2026-08-28 裁定 — サイズ動機は
  R1 shard 化が根治・速度は i4 が遅い。perf-ledger Q-9 / research 2026-08-28）。GPU 校正は
  **実用採用**（探索 3.6 倍速・配布焼きは CPU — research cuda-calibration）。
- **0.6.0（yomi 依存分離）リリース完了（2026-08-25）**: JSR 3 パッケージ = 0.6.0・Release
  v0.6.0・**公開依存から `@hdae/yomi` の消滅を API 実測で確定**（hub / runtime の 2 本のみ）・
  消費者ストーリー疎通緑。設計の正本 = ADR
  [0079](../docs/decisions/0079-sbv2-two-layer-input.md)。**0.6.0 で変わった面**（breaking・SBV2 のみ
  — 消費側の doc はここが索引）:
  - `Sbv2Pipeline.generate(utterance, options?)` — 第 1 引数は解析済みの `Sbv2Utterance`
    （フレーズ層 `Sbv2Phrases` = yomi `analyzeWithWords` の返り値が構造的に満たす →
    `toSbv2Utterance` で変換）。テキスト解析・辞書 19MB・修正辞書は**呼び手の責務**。
  - 撤去（シム無し）: `text` / `dictionary` / `overlay` 席・`analyzeProsody`・`Sbv2Prosody` /
    下書き型・`givenTone`・`OverlayDictionary` 再 export。編集は「フレーズ層で核を直して
    再変換」or「モーラ層で `tone` を直指定」。写経見本 = `examples/sbv2/`。
  - 検証 = **WAV 門 3 sha 不変**（吸収のビット同一性）・verify 1771/0/5。配布形・pin 値は不変。
- **0.5.x は全てクローズ**（0.5.1 = サンプラー再裁定 ADR
  [0078](../docs/decisions/0078-anima-sampler-selection.md) — 既定 Euler 維持・
  `AnimaGenerateRequest.sampler` で DPM++ 2M。記録 = [backlog](../docs/backlog.md)
  消化済み節）。
- **0.5.0 で変わった面**（消費側の doc はここが索引）:
  - **quant 席名が全て改名された**（ADR [0074](../docs/decisions/0074-quant-seat-naming.md)
    決定 6 の移行表が正本 — 例 `w8a8-s16` → `f16+dit8-a8-attn8-s16` / sbv2 `w8-bert4` →
    `i8+bert4` / irodori `w4` → `i8+dit4`）。`linearCompute` / `attentionCompute` の**値**も
    `"i8a8"` → `"a8"`（カーネル内部識別子・ファイル名・WGSL は実行変種の名前なので不変）。
  - **manifest は `karume/4`**（ADR [0075](../docs/decisions/0075-quant-presentation.md)）—
    quant の `label` / `description`、`requiredLimits`、ファイル参照の越境 `repo` / `revision`
    （ADR [0038](../docs/decisions/0038-manifest-v1.md) 追記 2026-08-25）。旧 format は読めない。
  - **`fromPretrained` の `ref` は必須**（既定ソースの廃止 — ADR
    [0073](../docs/decisions/0073-models-source-pin.md) 追記 2026-08-25）。pin 定数は
    `<FAMILY>[_<VARIANT>]_CURRENT` の 4 本で、位置づけは「**パッケージ版に合わせて自動追従したい
    場合のオプトイン**」= bump のたびに pin 更新 + 動作確認の義務つき。hub は revision 未指定の
    暗黙 `main` 解決に 1 回だけ warn（解決 SHA 印字 + pin / `*_CURRENT` の 2 択案内）。
  - anima に `pipelineConfig.scheduler.type` 席（`euler` / `dpmpp-2m`・省略時 euler）。
    **配布既定は euler**（再裁定 2026-08-25 — ADR 0078。0.5.0 期の公開 revision
    `ebb27bc4` / `6215f965` だけが dpmpp-2m 宣言）。DPM++ 2M は 0.5.1 の
    `AnimaGenerateRequest.sampler` で選ぶ。
  - 0.4.3 で配られた面（消費側 doc の注意点）: `animaLatents()`（途中 latent の逆正規化素材
    — プレビューには要らない）/ `approximatePreview()`（途中 latent → RGB の線形近似。係数は
    **正規化空間**で較正済みなので `copyLatents()` の返り値をそのまま渡す — 逆正規化した値を
    渡すと白飛びする）/ `AssetProgress` の per-file 欄 `fileLoaded` / `fileTotal`（必須欄・
    `verifying` / `complete` では常に等しい）。
- **anima の受理解像度を 8 通り縮小した（E-2）**: VAE タイル本数の上限を入口の受理集合へ足し、
  1456/1488/1584/1648/1680/1776/1840/1936px を名指しで拒否するようにした。**形式上は破壊的だが、
  対象はホスト RAM 破裂で実行不能だった値のみ**。省 RAM の逐次組み立てで受理へ戻す案は
  backlog later。
- **波 J / 波 L はどちらも 2026-08-24 に全クローズ**（波順の正本 =
  [backlog](../docs/backlog.md)）。**教訓 2 件**は現役: ①sim の A/B は同一リグ内でのみ有効 —
  **出荷リグでは GPTQ の丸め解が変わり発話実現が再抽選される**（繊細な性質は転移しない・
  最終裁定は必ず出荷バイトで）②adaLN（modulation の scale/shift/gate）は量子化感度が高い
  （irodori 実測 — 他 DiT へは未実測の仮説）。実測の正本 =
  [research/2026-08-24-gptq-expansion-quality.md](../docs/research/2026-08-24-gptq-expansion-quality.md)。
- PyPI `karume` は未リリース。クローズ済みの波の履歴は
  [backlog](../docs/backlog.md) と各 ADR / research が正本。

## Open decisions

- MiniMax-H3（動画生成・オープンウェイト 33.1B/42.5GB 級）は遠期の関心として記録のみ —
  ブラウザ実行はメモリ規模的に現行スコープ外（レビュー DS-4）。

## Pitfalls（現役のみ）

- **フル走行の verify は VRAM 圧で稀にフレークする**（毎回別のテストが落ち、単独再走は常に緑
  — known-issues）。落ちたファイルの単独再走で切り分ける。
- **ベンチ生成先と実画像コーパスは席が別** — コーパスは `outputs/misc/corpus/` の凍結コピー
  （再実行上書き事故は構造解消済み — [assets-layout](../docs/assets-layout.md)）。凍結コピーへ
  機械が直接書く形へ戻さない。
- **`linearCompute: "a8"` は i8 常駐と i4 常駐で数値契約が別**（i8 = full-k 厳密 / i4 = group
  部分縮約 — ADR [0076](../docs/decisions/0076-w4a8-linear-execution.md)）。取り違えると atol=0 の
  主張が意味を失う。経路の識別はパイプラインキーの `:wi4g32` サフィックスと診断が担う。
- **Metal**: threadgroup `vec4` への動的インデックス書きは黙って捨てられる（`gemm.ts` の
  `storeBTransposed` の switch 展開を新しい箇所で崩さない）。attention i8a8 / conv1d /
  conv2d / gru_scan / linear GEMV の Metal 数値差は known-issues・Metal は gpuTiming 不可
  （limitations）。
- **融合 matcher は実測形 exact-match** — exporter の発行順・形が変わると黙って外れ、値は
  正しいまま性能だけ落ちる。観測 = `Diagnostics.lastRunFusions` +
  `assets_fusion_counts_test.ts`。**row-block だけは外れ方が性能でなく資源** — 128MiB 級
  device で resource-limit failure に戻る（**分解経路の matcher だけの話** — 保存 attention は
  states 形・融合 attention とも行ブロックを op 内蔵で持つ〈ADR 0067 決定 7・波 D /
  2026-08-25 融合側〉）。分解形が matcher から外れると `bmm [H,S,S]` が**ノード出力スロット**
  になり原理的に分割不能 — 現状の該当（anima text_encoder / conditioner）は T=512 固定
  16MiB で無害（2026-08-25 棚卸し）。
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
