# backlog の消化済み節の退避（0.4.x〜0.12.0 リリース後・2026-08-20〜09-22）

> 2026-09-22 時点のスナップショット。[backlog](../backlog.md) の運用契約①「完了した項目は削除する」へ戻すために、
> 当時 backlog に残っていた `## 消化済み（…）` 12 節を**逐語で**退避したもの。設計と実測の正本は各節が指す
> ADR / research / perf-ledger 側にあり、この文書は「当時どうまとめられていたか」を引くための索引兼保存箱である。

読み方の注意:

- 本文は**退避当時の backlog の文面そのまま**で、その後の変更を反映していない。現況は ADR / research /
  [perf-ledger](../perf-ledger.md) / [known-issues](../known-issues.md) / [limitations](../limitations.md) /
  [CHANGELOG](../../CHANGELOG.md) が正本で、食い違ったらそちらが勝つ。
- 本文中の「now 節 / later 節 / release 節 / parked 節」「下の消化済み節」は**退避当時の backlog** を指す。
  現行の backlog には消化済み節は無い。
- 機械的に変えたのは 2 点だけ: 相対リンクの起点を `docs/` から `docs/research/` へ直した（94 本に `../` を前置）、
  節見出しを `##` から `###` へ 1 段下げた（12 本）。文面は 1 文字も変えていない。
- 作業履歴（どのコミットで何をしたか）は git が正本。本文が引くコミットハッシュは、当時の記述を追うための手掛かりとして残している。

## 節ごとの要点と正本の所在

| 退避した節                              | 何の記録か                                                                                                                                                          | 意味の正本                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0.12.0 リリース後（2026-09-07〜22）     | テスト整理の波 段 0〜4 / 外部レビュー取り込み / コード品質管理の波 段 1〜3 / decode 速度調査の波 / QAT レビュー対応の波 / MTP 段 1〜4-B / sampler の top-p 単独指定 | ADR [0005 追記](../decisions/0005-verification.md) / [0008 追記](../decisions/0008-public-api.md) / [0040 追記](../decisions/0040-fusion-pass.md) / [0096](../decisions/0096-speculative-decoding.md) / [0097 追記 7](../decisions/0097-gemma4-qat-integration.md) / [0104](../decisions/0104-gemma-fast-quant.md) / [0105](../decisions/0105-packed-static-quantize-activations.md) / [0106 と追記](../decisions/0106-device-keyed-references.md) / [0107](../decisions/0107-model-input-error.md)・[research 2026-09-09](2026-09-09-mtp-stage4.md) / [2026-09-10](2026-09-10-codex-mtp-optimization.md) / [2026-09-19](2026-09-19-qat-speed-recon.md) / [2026-09-19 レビュー](2026-09-19-qat-review.md)・[perf-ledger](../perf-ledger.md)・[CHANGELOG](../../CHANGELOG.md) の `[Unreleased]` |
| 0.12.0 リリース（2026-09-06）           | 長文脈 gemma4 の高速化 3 件（K-16 / K-14 / K-13）・融合候補 3 件の決着・hub の参照勘定・models の `onRetry` 透過                                                    | ADR [0067 追記](../decisions/0067-autoregressive-attention-vocabulary.md) / [0082](../decisions/0082-linear-gemv-decode.md) / [0094 追記](../decisions/0094-hub-cache-inventory-and-eviction.md)・research 2026-09-06 の 5 本・[perf-ledger](../perf-ledger.md)・CHANGELOG 0.12.0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 0.11.0 リリース（2026-09-06）           | hub のキャッシュ保守面（在庫・退避）と取得層 0.7.0 追従                                                                                                             | ADR [0094](../decisions/0094-hub-cache-inventory-and-eviction.md)・[limitations](../limitations.md)・CHANGELOG 0.11.0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 0.10.0 リリース（2026-09-05）           | BiRefNet 2048² 工事 A / B / C / ④ と網羅レビューの修正波                                                                                                            | ADR [0092](../decisions/0092-distribution-repos-and-sources.md) / [0093](../decisions/0093-transient-liveness-packing.md)・[research 2026-09-05](2026-09-05-softmax-guard-ab.md)・[断片化の追記](2026-08-09-xet-fragmentation.md)・CHANGELOG 0.10.0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 0.9.0 リリース（2026-09-04）            | OP マイクロベンチ 2 段目 / 未配布家族の初回公開（siglip2・depth-anything）/ export-recipes 切り出しのクローズ                                                       | ADR [0092](../decisions/0092-distribution-repos-and-sources.md) / [0065](../decisions/0065-exporter-core-recipe-split.md)・[research 2026-09-04](2026-09-04-opbench-stage2.md)・[release-runbook §2](../release-runbook.md)・CHANGELOG 0.9.0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 0.8.0 リリース（2026-08-30〜09-04）     | shard 仕様 v2 / v3・HF 6 リポ公開・モデル更新波 N1〜N3・メモリ管理波・生成 API 波・対話 example 波・可変 capacity 波・OP 数値レビュー波・perf P-1〜P-3              | ADR [0070](../decisions/0070-shard-loading-admission.md) / [0081](../decisions/0081-shard-spec-v2.md) / [0083](../decisions/0083-generation-api-surface.md)〜[0091](../decisions/0091-gemma4-host-rope-variable-capacity.md)・research 2026-08-31〜09-03 の各本・CHANGELOG 0.8.0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 既知問題 3 件 + anima 素版 i4 感度      | Pixel の中断 / NVIDIA の 2GiB 天井 / Chromium の ArrayBuffer 上限、adaLN i8 変種の視認不採用                                                                        | ADR [0080](../decisions/0080-hub-fetch-cache-050.md)・[known-issues](../known-issues.md)・[limitations](../limitations.md)・[perf-ledger](../perf-ledger.md) Q-9・[research 2026-08-28](2026-08-28-anima-adaln8-visual.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 0.7.0 リリース（2026-08-29 完了）       | 全席分割の再 export と越境参照の初適用、断片化がクライアント退行で手詰まりになった実測                                                                              | [release-runbook §2](../release-runbook.md)・[research 2026-08-09](2026-08-09-xet-fragmentation.md)・CHANGELOG 0.7.0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| R1 統合波（2026-08-28〜29）             | ロード面 API 工事 4 件と shard 配布の受け入れ実証                                                                                                                   | ADR [0070 追記](../decisions/0070-shard-loading-admission.md) / [0071 決定 4 の撤回](../decisions/0071-manifest-v3-shards.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 0.6.0（2026-08-25）                     | SBV2 入力の 2 層化と `@hdae/yomi` 依存の分離                                                                                                                        | ADR [0079](../decisions/0079-sbv2-two-layer-input.md)・CHANGELOG 0.6.0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 0.5.0 breaking 波 + 0.5.1               | quant 席の一斉改名・`karume/4` 繰り上げ・pin の公開面出し・anima のサンプラー再裁定                                                                                 | ADR [0073](../decisions/0073-models-source-pin.md)〜[0078](../decisions/0078-anima-sampler-selection.md)・[0038 追記](../decisions/0038-manifest-v1.md)・CHANGELOG 0.5.0 / 0.5.1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 波 K・リリース + 公開（2026-08-20〜21） | 配布形 `karume/3`・SBV2 の既定 quant とトーン注入席・pin 焼き込み・HF 3 リポ初公開                                                                                  | ADR [0071](../decisions/0071-manifest-v3-shards.md) / [0072](../decisions/0072-sbv2-text-injection.md) / [0073](../decisions/0073-models-source-pin.md) / [0076](../decisions/0076-w4a8-linear-execution.md)・[release-runbook](../release-runbook.md)・CHANGELOG 0.4.0 / 0.4.1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

## 退避本文（逐語）

### 消化済み（0.12.0 リリース後 — 2026-09-07〜22）

- **テスト整理・リファクタリングの波（2026-09-20 利用者提案・2026-09-22 完了 — 段は下の「段 N 済」の行）**: 開発機が
  Intel Arc B570 に替わり（NVIDIA の検証は別機の RTX 5070 Ti で行う）、karume はブラウザ優先
  （Deno も積極的に支える）という前提で、① sha256 / golden の**参照値をデバイスごとに持てる形**
  （参照環境の宣言と機ごとの値の並置）② verify の結果を **JSON に集約して環境間で受け渡し**、
  複数デバイスの差異を突き合わせられるようにする ③ 許容差は「Karume 独自基準 + WGSL 仕様帯」の
  2 段（独自基準は容易に撤廃可・従来基準に引っかかったことが分かる形。緩めるのは op 単位・仕様の
  範囲内・実害なしに限る — `e2e_golden_test.ts` の `OUTPUT_TOLERANCE`）④ 変更に関連するテストの
  個別実行を主にし、フル verify は横断変更とリリース前だけ（現状 30 分超）⑤ 速度の物差しは
  ヘッドレス Chrome 計測（`tools/llm-speed/browser` の計測ページ — 成立済み）。B570 で判明した
  記録は known-issues「Intel Arc B570」節・limitations「BiRefNet 系」「sha256 参照門」節。
  **段 0 済（2026-09-20）**: ④ の verify 分割（`test:core` / `test:models:<系列>` + 被覆の門
  `verify_lanes_test.ts` + 系列名に合わせたテスト改名）— 決定は [ADR 0005 追記](../decisions/0005-verification.md)。
  **段 1 済（2026-09-20）**: ① の sha256 側（参照値を環境キーごとの行へ + `KARUME_REFERENCE` の 3 モード +
  参照門）と ② の結果 JSON（`outputs/verify/<環境キー>/<日付>_<系列>/` に `results.json` + 実物を毎回）—
  決定は [ADR 0106](../decisions/0106-device-keyed-references.md)。残りは ① の golden 側（torch 由来の
  期待出力は tolerance 判定のままで手を付けていない）と ② の**環境間の突き合わせ道具**（集めた
  `results.json` を並べて差異を出す形）。
  **段 2c 済（2026-09-20）**: パッケージ README / LICENSE 同梱（`packages/*/README.md` を英語で新設 + リポ直下 LICENSE をバイト同一で複製・公開物入りは `deno publish --dry-run` で確認 — [release-runbook §4](../release-runbook.md)）。
  **段 2b 済（2026-09-20）**: CHANGELOG 新設（リポ直下 `CHANGELOG.md` — Keep a Changelog 形式・
  tag のある 18 版 + `[Unreleased]`。ADR 0008「breaking は CHANGELOG で明示する」の実体で、
  [release-runbook](../release-runbook.md) §4 に bump 時の移し替えを 1 項追加）。
  **段 2d 済（2026-09-20）**: テストの置き場と名前の是正（公開面の門の実体を
  `packages/runtime/tests/helpers/public-surface.ts` へ移動 + 全テストが実 GPU を要る
  `runtime_input_lifetime_test.ts` を `gpu_` 接頭辞へ改名。unit と GPU が同居する
  `estimate_test.ts` / `runtime_executor_test.ts` / `static_quantize_test.ts` は混在のため据え置き
  → コード品質管理の波 段 1 で GPU 側を `gpu_*` へ分割済み）。
  **段 2a 済（2026-09-20）**: 公開面のスナップショット門（各パッケージの `tests/public_surface_test.ts` と追跡 fixture
  `tests/fixtures/public-surface.json` — `deno doc --json` で採る値・型の export 集合。焼き直しは `KARUME_SURFACE=write` —
  [ADR 0008 追記](../decisions/0008-public-api.md)）。
  **段 3 済（2026-09-20）**: ③ の許容差 2 段化（`e2e_golden_test.ts` — Karume 独自基準を超え WGSL 仕様帯で受理した出力は
  `outputs/verify/<環境キー>/<日付>_golden/results.json` の `note` に残す・赤にしない）。
  **レビュー取り込み 済（2026-09-21）**: 外部レビュー（5 観点 × 2 試行）の指摘 87 件を実コードで反証した結果
  （正本 = `.claude/reviews/2026-09-21_chatgpt-reviews/triage.md` — 成立 78 / 反証 8 / 未決 1・high 0 / medium 9）のうち、
  構造の分割候補 16 件を除く「直す価値あり」を取り込んだ — runtime の batch 受け口（写しの後の再検査・フェンス前失敗の計測窓・
  pop 待ち中の消失）、テスト基盤（参照門を登録ケースで数える・fixture の読み直し書き・`results.json` の走行中マーカー・レーン門の
  ディレクトリ `--ignore` と綴りの門・公開面の entry 差分と値非公開の門・adapter 同一性）、Civitai の本体選択と basename の門
  （[ADR 0088 追記](../decisions/0088-civitai-air-intake.md)）、docs の同期 11 件。**構造の 16 件（triage.md §6 の着手順 3 段）は
  コード品質管理の波の入力**（later の同名項に V2-01 を起票済み）。
  **段 4 済（2026-09-22）**: ① の golden 側（判定 2 段目の WGSL 仕様帯の行を環境キー別へ — [ADR 0106 追記](../decisions/0106-device-keyed-references.md)）と
  ② の突き合わせ道具（`tools/verify-diff` — 門にせず差異があっても終了コード 0・2 台目の結果は手でコピー）+ 結果 JSON の実測欄 `measurements`
  （合格した回の maxAbs / maxRel も毎回残す・派生値は持たない）+ 実重み golden 11 本の結果の席（`<系列>-golden`）。

- **コード品質管理の波の設計項目（起票 2026-09-21 — 波本体は消化済み〈0.12.0 リリース後〉節）**:
  - **Anima / 生成の入力起因エラーを判別可能な型へ揃える**: `parseResolution` と
    `AnimaPipeline.generate` の値域検査、生成側の `maxNewTokens` / `stopTokens` の検査が素の
    `Error` を投げるため、同じ関数内の内部配線異常（hub のバグ側）と区別が付かず、複数モデルを
    1 ハンドラで受けるホストはメッセージ文字列を解釈するしかない。同じ検査群でも seed
    （`anima/random.ts`）と sampler 指定（`generation/sampler.ts`）は既に `RangeError` で判別
    できるので、揃える先は `RangeError` か公開エラー型のどちらか。**公開面の追加を伴うので
    設計項目**（ADR [0072](../decisions/0072-sbv2-text-injection.md) の 400/500 分離は SBV2 に
    閉じた決定で、そのまま持ち込まない）。対象は `models/src/anima/resolution.ts` と
    `models/src/generation/sequence.ts`。
    **済（2026-09-22）**: `ModelInputError` 1 本 + 派生 2 本・73 箇所置き換え — [ADR 0107](../decisions/0107-model-input-error.md)。

- **コード品質管理の波（2026-09-21 着手・承認済みの計画）**: 入力は `.claude/reviews/2026-09-21_chatgpt-reviews/triage.md` §6
  （分割候補 15 本の着手順 3 段・重複実装・未使用 export）。段 1 = 1 ファイルに閉じるか純関数の移動だけの項目、
  段 2 = family 内の責務分離と小さな共通層（sbv2 の資産門・gemma の admission / chat・ple の索引 codec / shard・hub 共通層）、
  段 3 = コア実行層の大移動（device.ts の acquire / probe・fusion のルール分割〈ADR 0040 追記〉・irodori・executor の構築相・
  recipe-builder の族別導出 — 1 ファイルずつ・executor / recipe-builder は最後）。分割しない 5 本（gemm / sequence /
  generation-context / linear-gemv / state-attention 全体）は同 §3。未使用 export 11 件は段 3 の後に剥がす。
  `deno.json` に `noUnusedLocals` / `noUnusedParameters` を有効化済み（赤は簡単に直せるものは直す方針）。
  **段 1 済（2026-09-21）**: state-attention の行統計 WGSL を 1 本の生成器へ（スナップショット不変）・executor の出力解決 3 重と
  発行準備を 1 本ずつに（await 列不変）・pipelineConfig の基本 reader を `models/src/config/readers.ts` へ（22 箇所）・
  `withSession` を `session/with-session.ts` へ・tokenizer 資産の門を `text/asset-gates.ts` へ・Unicode 区間探索を
  `text/code-ranges.ts` へ・`OpKind` を `OpContract["kind"]` から導出・混在テスト 5 本を `gpu_*` へ分割・opbench の rig と
  single_file.py の来歴欠落を修正。残る重複（報告のみ）: `isRecord` / `readRecord`（anima / sbv2 / irodori / gemma）と
  `isPositiveInteger`（birefnet / depth-anything / siglip2 / irodori）は別レイヤなので段 2 で個別判断。
  **段 2 済（2026-09-21）**: gemma/pipeline.ts → `gemma/admission.ts`（受理集合の判定）+ `gemma/chat-turn.ts`（chat 1 ターンの変換と
  後始末・公開型 4 本は barrel の import 元を付け替え）で 2,589 → 1,902 行・gemma/ple.ts → `ple-index.ts`（索引 codec と定数）+
  `ple-shard.ts`（shard の読み口と検査）で ple.ts は所有者 + facade の 724 行・ple-gpu.ts の重複定数 3 本を import へ・
  hub 共通層 `hub/asset-readers.ts`（資産バイト列 / JSON の読み口）+ `hub/graph-gates.ts`（graph 入力の静的次元）で 8 family の
  複製を解消・sbv2 の tokenizer 資産門を `sbv2/text/asset.ts` へ・config の残り重複（isPositiveInteger / isRecord / readRecord）を
  readers.ts へ。**残置（設計判断が要る）**: `Gemma4PipelineOptions` を引数に取る 4 本（assertSpeculative /
  resolveGemma4PleResidency / buildGemma4Program / speculativeSetup）は admission.ts へ移すと pipeline.ts と循環するので
  pipeline.ts に残した（寄せるなら gemma4 の公開型置き場を別に立てる — 2026-09-21 裁定: 新設するにしても後回し）。sbv2 の
  staticInputDim は方針が逆向きで対象外。
  **段 3 済（2026-09-21）**: 1 ファイルずつ、いずれも行の移動だけで本体はバイト同一（機械突合）。gpu/device.ts → `context.ts`（3 クラス）+
  `acquire.ts`（取得・limits・カナリア）で device.ts は層の入口 54 行 / runtime/fusion.ts → `fusion-rule.ts` + `fusion-rules/<rule>.ts` × 7 で
  fusion.ts は入口 482 行（[ADR 0040 追記](../decisions/0040-fusion-pass.md)）/ irodori/pipeline.ts → `admission` / `conditioning` / `dit-loop` /
  `stage` で 928 行（10 段の説明は pipeline.ts 冒頭に温存）/ executor.ts → `session-build.ts`（構築相・Session.build はファサード）で
  2,745 行 / recipe-builder.ts → `RecipeBuildFace` の注入面 + `recipe-builders/{elementwise,layout,linear,norm,attention,conv}.ts` で
  706 行。未使用の export 修飾子 18 本（レビューの 11 件のうち import の無い 10 件 + 族内 8 件）を外した。**波の結論**: 1,000 行超の
  src は 18 本 → 16 本で、行数の削減より「規則・責務の所有者を 1 つにする」ことが成果（executor 2,745 / gemma pipeline 1,902 /
  context 1,332 / session-build 1,081 / recipe-builders/attention 1,208 は分割後も 1,000 行超・分割しないと判定した gemm /
  sequence / generation-context / linear-gemv / state-attention・info の reference/ops / shapes・未着手の anima pipeline /
  ops/contracts / sbv2 pipeline / hub manifest）。

- **decode 速度調査の波（2026-09-19〜20・2026-09-20 に区切り — 残りは later へ）**: 帰属と反証は [decode 速度の帰属と次に試すこと](../research/2026-09-19-qat-speed-recon.md)、
  候補の採否は [perf-ledger](../perf-ledger.md) H-26〜H-29 / K-48〜K-53（K-45 / K-46 / K-47 は追記）。確定した事実: Deno の 23.8 ms/token のうち
  10 ms は deno_webgpu の poll ループの sleep（karume 無関係・採否判定は Chrome で）・律速は帯域でも演算でもなく命令数 × 占有率と dispatch 本数・
  「i8 計算」は方向として正（利得 0.9 ms・段 0 の kill 判定が先）・「i8 KV」は速度 0（メモリ項目へ）・WebML 285 tok/s の要因は実行構造（先行投入・presrq・1 pass）。
  順序（research §10）: ⓪ 物差しを Chrome へ + GPU 1 セッションで確定する事実（GEMV 総時間・未帰属 49 dispatch・`per_layer_model_projection` の費用）— **済 2026-09-19（research §13: Chrome 壁 QAT 10.6〜10.8 / 通常 10.0 ms・GPU 7.0〜7.5 / 6.2 ms・非 GPU 3.3〜4.1・GEMV 4.0 ms・未帰属なし・K-51 は kill）** →
  ① K-45 段 0 — **済 2026-09-19（research §14: 門通過・ただし律速は活性ロード本数で段 1 の形は裁定待ち）** → ② H-28 — **済 2026-09-19（`d1c848e`・opt-in・単独では効かず既定 host のまま・research §15）** → ③ H-27（先行投入）→ ④ K-45 段 1a — **済 2026-09-19（`57416eb` + 追補・opt-in・Chrome +8.3%・research §16）**・**`i4-fast` へ宣言済み 2026-09-20（ADR 0105 追記 2・M2 追試済み: 速度中立・id 列一致〈並列族の明示 fma 化 — 追記 4〉）・段 1b（整数内積）は棄却（lm_head に int8 活性が無い）** → ⑤ 小物（K-48 段 1 / K-49 / K-50 段 1 / K-51）→
  ⑥ K-46 の再起票（メモリ項目）。併用後の見込みは Deno 約 7.5 ms（約 130 tok/s）/ Chrome 約 6.5 ms（約 150 tok/s）。
  **区切り（2026-09-20）**: K-45 は `i4-fast` に宣言（M2 追試 = 速度中立・id 列一致・並列族の明示 fma 化 — ADR 0105 追記 2 / 4）・段 1b は棄却・
  H-28 は残件 ①（通常 decode を同一 batch へ — `Session.enqueueRead`・ADR 0054 追記）と残件 ②（TTFT の帰属 — research §15.2: Chrome の VRAM 占有下で
  prefill run が世代を追って遅くなる現象・席の欠陥ではない）を閉じた。**H-27 段 ②・小物（K-48 / K-49 / K-50）・K-46 は later へ**（復活条件つき）。

- **QAT レビュー対応の波（2026-09-19）**: 裁定は [ADR 0097 追記 7](../decisions/0097-gemma4-qat-integration.md)、
  実測は [QAT レビューの実測記録](../research/2026-09-19-qat-review.md)。中身は ①配布既定を通常 Gemma と
  揃える（capacity 4096 / chunkLength 768 / trace 上限 768・対話 CLI の既定 256 token。既定 capacity と
  trace 上限の定数を分け、焼く前に 3 式を検査する）②scale=0 の恒等 SRQ を recipe が挟まないようにし、
  構造門を「共有 head だけ SRQ 省略可」へ緩める ③構造門・配布計画（`qat_plan`）・公開入口の失敗経路と
  QAT RoPE の golden を埋める ④[glossary](../glossary.md) と [quantization](../quantization.md) を新設し、
  散在していた用語と方式の索引をそこへ寄せる ⑤`reference.json` を schema 2 にする
  （byte 一致フラグの廃止・本数の突合・未反映テンソルの記録）。
  ローカル配布 `models/karume-gemma4-qat` / `models/karume-gemma4` は ADR 0104 の recipe で再ビルドする
  （E2B の既定 quant が `i4-fast` になる）。512 超の文脈での品質検収はこの波に含めない。

- **sampler の top-p 単独指定 ✅ 2026-09-10**（H-11 残件）: f32 の安定 radix sort で順位・確率をビット同一に保ち、
  全語彙の比較ソートを置換した。Gemma 実 logits 24 行の再生で 7.6〜7.8 倍（CPU 抽選だけ）。
  [実測と適用範囲](../research/2026-09-10-codex-mtp-optimization.md#共通サンプラーの-top-p-単独指定)。top-k 付きの配布既定は変更しない。
- **layer_norm の悪条件入力（分散 ≈0）**: ケース個別 tolerance の席で扱う
  （[research 2026-08-31](../research/2026-08-31-op-tolerance-measurement.md) §7 注記）。
- ~~生成 API 波（起票 2026-08-19）~~ **now 節へ昇格・設計正本化済み（2026-08-31）** — 起票が
  書いていた形（`GenerationProgram` / stateful sequence / `generateGreedy` 格下げ）は
  ADR [0083](../decisions/0083-generation-api-surface.md) が正本。tokenizer は
  [0084](../decisions/0084-gemma-tokenizer-chat.md)・PLE 配布形は
  [0085](../decisions/0085-ple-host-gather.md)。実行計画と各段の合格線は **now 節**。
- ~~R1 の残り~~ **消化済み（2026-08-29 — R1 統合波の節を参照）**: ロード面 API 工事 4 件も
  exporter 自動分割規則も実装完了（ADR 0070 追記 2026-08-29 / ADR 0071 決定 4 撤回）。
- HF 公開: **jvnv / irodori / anima の 3 リポは波 K-4 で公開済み**（2026-08-21）。FN は parked
  （再配布の書面根拠なし）。以後の新モデルは runbook に従う
- ~~**export-recipes の別リポジトリ分離**~~ **クローズ（ADR
  [0092](../decisions/0092-distribution-repos-and-sources.md) 決定 5 — 分離しない。動機だった
  ライセンスの見え方は README 2 か所の carve-out と family 別 `THIRD_PARTY_NOTICES.md` で解く）**。
  切り出し時の論点だった `_shared/paths.py` の REPO_ROOT 導出・runtime 適合 fixture の共有・
  uv workspace の解体は、いずれも払わずに済む。

### 消化済み（0.12.0 リリース — 2026-09-06）

0.12.0 の中身（結果だけ残す — 設計の正本は ADR 0067 / 0082 / 0058 / 0094 の追記・実測は research 2026-09-06・
公開面の差分はリリースノート v0.12.0）:

- **runtime: 長文脈 gemma4 の高速化 3 件**（OP / Fusion の波・a 案）: K-16 = lm_head（i8 × M=1）を GEMV 族へ
  （ビット同一・単体 ×5.0・decode GPU −21〜24% — `5ddd186`・[research](../research/2026-09-06-gemv-i8-k16.md)）/
  K-14 = ①QK の D 並列縮約 ①′（opt-in 席 `stateAttentionReduce: "parallel"`・M=1 の計画だけ・decode 壁 P=16K
  −9〜15% — `cce129d` `4182b8b`・[research](../research/2026-09-06-state-qk-parallel-k14.md)）/ K-13 = prefill 計画
  （M ≥ 16）の GEMM 骨格タイル経路 ①ₜ / ③ₜ（席に依らない既定・参照経路とビット同一・prefill 壁 P=16K −64% —
  `ad8a4b9` `39d5e4e`・[research](../research/2026-09-06-state-attention-tiled-k13.md)）。幾何表は ADR 0067 追記。
  `Gemma4Pipeline`（既定 parallel）の prefill 中間値は 0.11.0 と変わる（golden / token 列は不変）。
- **融合候補 3 件は実測で閉じた**: K-15（gelu_tanh+mul）/ K-7（ゲート付き残差）は実装 → ABBA → 判定線に届かず
  revert（`5277306` `1e674e6`）、P-5（permute 畳み込み）は保留（[research](../research/2026-09-06-fusion-spikes-k15-k7.md)・
  ベースラインは [research](../research/2026-09-06-op-fusion-baseline.md)・`opbench graph` は 4 家族 + `--capacity`）。
- **hub: `evictCachedAssets` の参照勘定**（anima-web ⑫）: 対象と参照集合が同一の兄弟席を既定の守る側から外す・
  `CacheInventoryOptions.protect`・`EvictedAssets.alsoEvicted`（`2fc3587`・ADR
  [0094](../decisions/0094-hub-cache-inventory-and-eviction.md) 追記・[limitations](../limitations.md)）。
- **models: `onRetry` の透過**（anima-web ⑪）: 8 家族の hub オプション透過を `src/hub/load-options.ts`
  （`FromPretrainedHubOptions` + `hubLoadOptions`）へ 1 本化（`f81cc6a`）。birefnet の `fromPretrained` が
  `BIREFNET_SOURCES` を案内（`8ae0b71`）。
- **docs**（karume-samples）: BiRefNet の最大 binding を cat_211 込み（1024² 320MiB / 2048² 1280MiB）へ訂正
  （limitations + card.py — `d116d7e`。公開済みカードの再発行は now の残件）。
- 配布形の変更なし = HF の焼き直し・pin 更新なし（公開済み manifest の `generator` は `karume/0.10.0` のまま）。
  公開後の疎通 `smoke:published` は追加設定なしで緑。

### 消化済み（0.11.0 リリース — 2026-09-06）

0.11.0 の中身（結果だけ残す — 設計の正本は ADR 0094・公開面の差分はリリースノート v0.11.0）:

- **hub のキャッシュ保守面**（anima-web 要望・2026-09-05 裁定）: `listCachedAssets` /
  `evictCachedAssets`（ADR [0094](../decisions/0094-hub-cache-inventory-and-eviction.md) — 参照勘定は
  manifest 1 本の中だけ・全在庫の他の選択が守る・越境参照は残す。by-design の制約は
  [limitations](../limitations.md)）。anima-web は `@hdae/fetch-cache` への直依存を返上できる。
- **fetch-cache 0.7.0 追従**: 429 / 503 の再試行（`Retry-After` 追従・既定 5 回）と HF 層の受信上限は
  取得層 0.7.0（その ADR 0010 / 0011・2026-09-05 公開）。hub は依存 `^0.7.0`・
  `LoadManifestOptions.onRetry`（`RetryDiagnostic`）の透過・`transport.ts` の撤去。**breaking 1 件** =
  HF 取得元の受信超過が `IntegrityError` から `HubFetchError`（cause = 取得層のエラー）へ。
  content-length の事前突合は移植せず、`karume.json` の 1 MiB 上限は全量受信後の判定（ADR 0094
  決定 4）。
- 配布形の変更なし = HF の焼き直し・pin 更新なし（公開済み manifest の `generator` は
  `karume/0.10.0` のまま — hub は読まない）。公開後の疎通で `tools/published-smoke/deno.json` の
  `minimumDependencyAge.exclude` に `jsr:@hdae/fetch-cache` を追加（公開 1 日の依存が最低経過日数に
  引っかかるため — runbook §5 に追記）。

### 消化済み（0.10.0 リリース — 2026-09-05）

0.10.0 の中身（結果だけ残す — 設計の正本は各 ADR・実測は research・公開面の差分はリリースノート
v0.10.0）:

- **BiRefNet 2048² 工事 A / B / C / ④**（設計の正本 = ADR
  [0093](../decisions/0093-transient-liveness-packing.md)）: A（recipe パッチ ⑨ = 1×1 conv と bilinear
  upsample の順序交換で `cat` を消す・`--verify` 3 段・1024² / 2048² の系列と golden 再採取）/
  B（静的 liveness パッキング — runtime へ結線）/ C（上限 preflight — B と同じ計画関数）。実測 =
  1024² の中間 6,283 → 749 MiB（run 1.8 s）・2048² の中間 2,948 MiB・総確保 ≈ 4.1 GiB・run 7.5〜8.6 s
  （RTX 3080 Ti — ADR 0093 Consequences）。④ = 配布形は 1 リポ 2 モデル（モデル名 = 解像度・既定
  `"1024"` — ADR [0092](../decisions/0092-distribution-repos-and-sources.md) 決定 9）で
  `karume-birefnet-hr` / `karume-lucida` を HF へ初公開し、`BIREFNET_SOURCES`（キー `birefnet-hr` /
  `lucida`）を pin 付きで新設 — 0.10.0 で JSR の公開面へ出た（対応表は **7 家族 10 エントリ**）。
  e2e `SERIES` に 2048² 2 本を実測 tolerance つきで追加。
  **断片化**: 両リポとも 2048 側の shard 1 本が目安割れのまま（削除 → 再作成でも 2 コミット法でも
  回復せず・DL 2.4 倍遅い —
  [research 2026-08-09 の 2026-09-05 追記](../research/2026-08-09-xet-fragmentation.md)。回復手段が
  見つかったら上げ直す）。bump 後の焼き直し（runbook §0 — `generator` = `karume/0.10.0`）は
  `karume.json` + `README.md` の 2 ファイルだけを上げ、shard は再アップロードしていない（断片化の
  実測値は不変・pin はその revision）。
- **網羅レビューの修正波**（所見の正本 = `.claude/reviews/` の 2026-09-03 以降の SUMMARY — git
  追跡外）: 受理集合を変えず拒否集合を広げる向きの breaking（runtime の実行形ノブ 4 本の綴り検査・
  見積りの入口 range-check・models の家族別引数 / manifest 検査の呼び出し口への移動・exporter の
  配布計画の拒否）と修正多数 — 公開面の差分はリリースノート v0.10.0 が正本。安全 softmax ガード
  変更の A/B は 0.9.0 公開資産 46 系列で差分ゼロ
  （[research 2026-09-05](../research/2026-09-05-softmax-guard-ab.md) — 上げ直し不要）。

### 消化済み（0.9.0 リリース — 2026-09-04）

2026-09-04 裁定の作業波 a / b / d（結果だけ残す — 設計の正本は各 ADR・実測は research）:

- **a. OP マイクロベンチ 2 段目 + Fusion 半自動発見 2 段目**（段 0〜4・実測正本 =
  [research 2026-09-04](../research/2026-09-04-opbench-stage2.md)）: `tools/opbench` に `single`
  （計測規約を実装として内蔵 — クロック張り付けの filler を新規に規約化）/ `graph` / `torch`（列 B）を、
  `tools/fusion-hints` に `inductor` を追加。合格線 = K-11 の census 加重 9.05ms（ADR 0082 の 7.38ms に
  +22.6%・帯内）・single / graph 1.01・P-1 の変種キーと dispatch 数の一致、で達成。**残（起票）**:
  ①**CPU/TS 側配置の系統評価**は未着手（先例 = PLE host gather / relattn のホスト生成 — 次の性能波で
  `single` の形別表を入口にする）②Inductor 突合の join を normalize の**出自**で行う（現状は fx 名で、
  normalize が合成する linear / rms_norm / rope が unobserved に落ちる — exporter normalize に出自 1 欄）
  ③`graph` の他家族（現状 gemma4 / anima）④`single` の Metal 実走（wall モードは実装済み・timing は
  Metal の timestamp 不能）。K-7 の再評価材料は perf-ledger へ記入済み（adaLN 側は Inductor も畳む）。
- **b. 未配布家族の初回公開**（リポ割り・命名・対応表の規則の正本 = ADR
  [0092](../decisions/0092-distribution-repos-and-sources.md)）: `karume-siglip2`
  （**1 リポ 2 モデル**・base / so400m 同居・既定 base — 決定 8）と `karume-depth-anything-v2`
  を初公開し、`karume-gemma4-e2b` → `karume-gemma4` の改名を同乗させた（改名後はカードと
  `karume.json` を焼き直したので revision が動いている）。対応表は **6 家族 8 エントリ**
  （anima 2 / irodori 2 / sbv2 1 / gemma4 1 / siglip2 1 / depth-anything 1）で、公開 revision の
  正本は `packages/models/src/*/config.ts` の pin 8 本（docs には写さない）。
  **断片化**: siglip2 の初回アップロードが global dedup のヒットで断片化し（so400m の 7 shard 中
  5 本が 4.2〜8.9 MiB/term）、hf_xet 1.6.0 の停止ノブ + shard-cache 退避 + リポ再作成で
  46〜61 MiB/term へ回復させた（機序と実測 =
  [research 2026-08-09 の 2026-09-04 追記](../research/2026-08-09-xet-fragmentation.md)・恒久手順 =
  [release-runbook](../release-runbook.md) §2）。**合格線の実績** = 断片化検証は siglip2 46〜61 /
  depth-anything 47 MiB/term で目安 ≥10 を全て満たす。**公開完了**（2026-09-04 — GitHub Release
  v0.9.0 → JSR 0.9.0 → `deno task smoke:published` 緑・`KARUME_SOURCES` 8 本の疎通を確認）。
  `karume-birefnet-hr` と `karume-lucida` は**後回し**（2026-09-04 ユーザー裁定 — 2048² は現状
  不成立〈[limitations](../limitations.md) の BiRefNet 節〉。プールの再利用方式の見直しと中間
  テンソルの `requiredLimits` 宣言が前提で、公開する時は 1024² の配布形のまま。上流ライセンス
  の人間確認は 2026-09-04 に済み — 両方 MIT・著作権者 2 名・`LICENSE.md` は recipe が同梱）。
  **vowel-detector も今回の波から外した**（上流の体裁整備が先 — 2026-09-04 ユーザー裁定）。
- **d. export-recipes 切り出し（裁定済み・案 A）→ クローズ**（ADR
  [0092](../decisions/0092-distribution-repos-and-sources.md) 決定 5。切り出さない）:
  分離の動機はライセンスの見え方であって構造ではなく（構造の分離は ADR
  [0065](../decisions/0065-exporter-core-recipe-split.md) が machine gate 込みで済ませている）、
  README 2 か所の carve-out + family 別 `THIRD_PARTY_NOTICES.md` で同じ目的を果たす。
  uv workspace の解体・資産根 / fixture 書き先の注入は払わずに済む。

### 消化済み（0.8.0 リリース — 2026-08-30〜09-04）

結果だけ残す（設計の正本は各 ADR・実測は research）:

- **shard 仕様 v2 / v3**: グラフ専用 shard + 上限の単一定数 + 常時分割（ADR
  [0081](../decisions/0081-shard-spec-v2.md)）→ 上限超えテンソルの行範囲分割（piece）と受理上限の
  ファイル長化（ADR [0090](../decisions/0090-shard-spec-v3-tensor-pieces.md)）。v2 の系列 repack と
  ミラー再生成は新旧の全テンソルビット同一で証明（v3 の piece 分割は実行出力のビット同一を
  ADR 0090 が担保）。
- **HF 6 リポ公開 + pin 焼き込み**: `karume-anima`（公式 5 変種同居）/ `karume-anima-extra`
  （越境参照）/ `karume-irodori-v4-small` / `karume-irodori-v4.1-small` / `karume-sbv2-jvnv` /
  `karume-gemma4-e2b`。公開 revision の正本は pin 定数
  （`packages/models/src/*/config.ts` — ADR [0073](../decisions/0073-models-source-pin.md)）で docs
  には写さない。旧 `hdae/karume-anima-turbo` は退役（ADR [0087](../decisions/0087-anima-official-extra-repos.md)）
  — 公開済みリポは README を deprecation 掲示へ差し替えて残置（2026-09-03 ユーザー裁定）。
- **モデル更新波 N1〜N3**: Irodori v4.1-small の取り込み（full-loop 検証は 2 段判定へ改修 —
  [research](../research/2026-09-01-irodori-v41-euler-sensitivity.md)）/ anima の公式・extra 分離と
  i4 席の退役（ADR [0087](../decisions/0087-anima-official-extra-repos.md)）/ Civitai AIR 取り込み
  コマンド（ADR [0088](../decisions/0088-civitai-air-intake.md) — 出所が dist まで連鎖する形）。
- **メモリ管理波 Phase A〜C**: 単発バッファの絶対上限を確保前に決定論的検査（ADR
  [0089](../decisions/0089-memory-limits-preflight.md)）→ ロード時の器の使い回しと HF 経路の `into`
  （ADR [0070](../decisions/0070-shard-loading-admission.md) 追記）→ shard 目標値とテンソル分割
  （ADR [0090](../decisions/0090-shard-spec-v3-tensor-pieces.md)）。合計 vs 物理の事前検査は原理的
  に不能で [limitations](../limitations.md) に by-design 記録。実測 =
  [research](../research/2026-09-02-shard-size-ram-peak.md)。
- **生成 API 波（段 0〜5）**: API 面（ADR [0083](../decisions/0083-generation-api-surface.md)）/
  tokenizer・detokenizer・chat テンプレート（ADR
  [0084](../decisions/0084-gemma-tokenizer-chat.md)）/ PLE のホスト gather 配布形（ADR
  [0085](../decisions/0085-ple-host-gather.md)）+ `Gemma4Pipeline` と配布形一式。gemma4 の
  ライセンスは Apache 2.0 を現物で確認（ADR 0065 stage 6 の門）。
- **対話 example 波 + ChatSession**: 取得元抽象 `DistributionSource`（ADR
  [0086](../decisions/0086-distribution-source.md) — ローカルミラー直読・越境は明示 mapping）+
  `examples/gemma4` の対話 chat + `Gemma4ChatSession`（溢れ処理は注入可能・既定
  `dropOldestTurns`）+ prefill 進捗の口（ADR
  [0091](../decisions/0091-gemma4-host-rope-variable-capacity.md) 決定 6）。
- **可変 capacity 波 + K-12**: RoPE 表を配布物から外し cos / sin をホスト供給、capacity と
  chunkLength を実行時ノブへ（ADR
  [0091](../decisions/0091-gemma4-host-rope-variable-capacity.md)）。decode の ③PV は KV 長方向の
  並列縮約が `Gemma4Pipeline` の既定（perf K-12・実測 =
  [research](../research/2026-09-03-gemma4-chunklength-k12-sweep.md)）。
- **OP 数値レビュー波**: 危険クラスの台帳化と修正（tanh_stable / softmax 族の nan_max 統一 /
  融合 attention の空行ガード）+ 飽和域の厳密カナリア常設 + DEFAULT_TOLERANCE 退役 → op 別
  実測表とビット同一門（[台帳](../research/2026-08-31-op-numerics-review.md) /
  [tolerance](../research/2026-08-31-op-tolerance-measurement.md)）。
- **cold ロードの DL スロット改善は kill**: グラフ相が shard v2/v3 で消え、律速は回線帯域その
  もの（[research](../research/2026-09-02-cold-load-dl-timeline.md)）。
- **perf P-1 / P-2 / P-3 採用**: `quantize_rows` の小 D 変種 / `BatchScope.settle()` / anima
  VAE タイルの整除制約撤廃（受理解像度 8 通りの復帰）。実測 =
  [research](../research/2026-09-03-perf-spikes-p1-p3.md)・採否の正本は
  [perf-ledger](../perf-ledger.md)。
- **opbench / fusion-hints の 1 段目**: 8 家族の実形状 census と未掴の融合形の列挙
  （[research](../research/2026-09-03-op-census-fusion-hints.md)）。
- **リリース前の差分レビュー修正波**: 正しさ・門・docs・tools を項目別に消化（正本 =
  `.claude/reviews/2026-09-03_7fc4ada/` — git 追跡外。見送りは同 ROADMAP.md）。
- **Mac（M2）検証**: メモリ管理波後とリリース前の 2 回。赤はすべて既知クラスへ帰着し新規欠陥
  なし（署名は [known-issues](../known-issues.md) Metal 節）。M2 手動確認 2 点（dp4a カナリア /
  軸 reduce パリティ）も緑で、GEMV の 1 ULP 差は既定維持の裁定（ADR
  [0082](../decisions/0082-linear-gemv-decode.md) 追記 1 / 3）。
- **LLM 先行波（L-0 / L-1 / L-10）**: decode の律速をカーネル側と特定
  （[research](../research/2026-08-30-gemma4-decode-wallclock.md)）→ K-11 起票 → ADR
  [0082](../decisions/0082-linear-gemv-decode.md) で消化。sliding スロットの window 実数宣言と、
  融合カウント門の decode 資産への拡張も同波。

断片化検証（2026-09-04 時点の各リポの revision・**各リポ最大 safetensors 2 本だけ**を見た当時の
使い捨て台本による）: anima 63 / 63・anima-extra 16 / 36・irodori-v4-small 25 / 31・
irodori-v4.1-small 25 / 31・sbv2-jvnv 20 / 16・gemma4-e2b（0.9.0 で `karume-gemma4` へ改名 —
改名後は焼き直しで revision が動いている）63 / 28 MiB/term = 目安 ≥10 を全て満たす。
追試は恒久台本 `tools/release/hf-upload.zsh check <repo>` で行う（**全 safetensors** を回すので
本数が増える — 代表 2〜3 本では shard 間の偏りを見落とす。[release-runbook](../release-runbook.md) §2）。

### 消化済み（既知問題 3 件 + anima 素版 i4 感度 — 2026-08-25〜28）

Anima Web アプリからの既知問題 3 件（調査で機序確定済み — 経緯は git / ACTIVE_DESIGN）と、
素版 i4 の量子化感度特定（later 節からの前倒し — 配布スキップ裁定の復活レバー）:

- ①Pixel の "BodyStreamBuffer was aborted" — hub の真因マスキング解消 + バイト予算 +
  検証直列化は**済**。実機での真因再判定（err.cause 観測）はリリース後 —
  [known-issues](../known-issues.md)
- ②NVIDIA の 2GiB 天井（Dawn D3D12 固定値）— 融合 attention の行ブロック化は**済**
  （parked「2048px DiT attention メモリ工事」の消化）
- ③Chromium の単一 ArrayBuffer 上限で Base f16 がロード不能
  （[limitations](../limitations.md)）— 根本 = next の R1 shard 配布を優先（2026-08-25 裁定）。
  DL 前の即エラーは fetch-cache 0.5.0 の `expectedBytes` 即 throw + hub 追従で**済**
  （2026-08-28 — 受信前に `cause` = RangeError で落ちる。ADR
  [0080](../decisions/0080-hub-fetch-cache-050.md)）
- ⑤fetch-cache 0.5.0 追従（hub）— 検証責務の移譲（記録ハッシュ信頼・knob なし）・認証隔離の
  撤去（ユーザー裁定: gated 運用予定なし）・`AssetPhase` から `verifying` 撤去・旧名前空間
  `karume/1` 系 purge・`clearHubCache` の対象変更。正本 = ADR
  [0080](../decisions/0080-hub-fetch-cache-050.md)（旧 CAS ドラフトを置換 — `archive/hub-cas-0.5.0`
  の再適用は不要になった）
- ④素版 i4 感度 — adaLN + block 外 i8 変種は**視認スイープで不採用**（2026-08-28 裁定 —
  perf-ledger Q-9 /
  [research](../research/2026-08-28-anima-adaln8-visual.md)。教訓: 視認 A/B は seed 4 本以上）。
  **anima DiT i4 系はしばらく保留（2026-08-28 ユーザー裁定）** — 動機だった「サイズ起因の
  DL 不能」は R1 shard 化が根治し、速度は i4 経路がむしろ遅い（~2 倍）ため優先度が立たない。
  未検証軸は research に列挙のまま（復活時は GPU 校正 =
  [実用可・3.6 倍速](../research/2026-08-28-cuda-calibration.md)で回す — 配布焼きは CPU）

### 消化済み（0.7.0 リリース — 2026-08-29 完了）

HF 更新系は**完了（2026-08-29）**: 全席分割の再 export 8 本（**全テンソルビット同一証明** —
LoRA scale=1.0 も同時証明）→ base 3 モデル family 再生成 → HF 上げ → turbo を**shard ごとの
越境参照**（新機構の初適用）で焼き直し → HF 上げ → pin 2 本焼き込み + 実 DL 疎通
（turbo = demo 完走 / base = fromPretrained + 生成完走）。**公開 revision の正本は pin 定数**
（当時は 1 公開リポ = 1 定数の形。現在の在処は ADR
[0092](../decisions/0092-distribution-repos-and-sources.md) 決定 3 の対応表）で、docs には
写さない（尾部スラック則の反映で両リポとも焼き直したように、SHA は後から動く）。断片化検証:
全 shard 26.5〜30.4 MiB/term（健全）— 例外は base の `shared/text_encoder` shard1 =
**4.5 MiB/term（旧公開バイトの xorb へ部分ヒットした継承断片化** — 同バイト再アップは
hf CLI が転送スキップするため runbook の処方が効かない。delete→再 up の 2 コミット法も
**不発を実測済み**（hf_xet 1.4.3 退行）。恒久対処は不要になった — shard v2/v3 で対象ファイルが
消滅し、最終 SHA の断片化検証は上の 0.8.0 節）。

- **Release v0.7.0 published → JSR 3 パッケージ publish 完了（2026-08-29 ユーザー確認）**。
  リリースノートは公開前に検証ワークフロー（主張突合 + 両方向網羅）を通した — 修正 2 +
  Breaking 追記 1（`from*Assets` は分割リポを開けない）+ 補足 4 を反映
- 2026-08-29 裁定 3 件は**消化済み**: ①コーパスは `demo:eval-images --source
  models/karume-anima-turbo`（正本の役割別プロンプト）で再生成し 3 ファミリの golden を
  採り直した（意味論門込み緑）②断片化は**クライアント退行で現状の手が尽きた**ことを実測で
  確定し記録（runbook §2 NOTE — 恒久対処は shard v2/v3 で不要になった・0.8.0 節）③尾部スラック則（未閉 ≤1.5GiB は詰め切る — `SHARD_TAIL_LIMIT`）で端数
  shard を廃し、turbo i4 の祖父条項は**規則上の正会員**になった（1.14GiB ≤ 1.5GiB。
  → 尾部スラック則自体は 2026-08-30 の shard 仕様 v2 で廃止 — ADR 0081）
- リリース後 = ChatGPT 全体レビュー消化（ユーザー持参）・Pixel 実機 err.cause 再判定

### 消化済み（R1 統合波 — ロード面 API 工事 + shard 配布・2026-08-28〜29）

結果だけ残す（設計の正本 = ADR [0070](../decisions/0070-shard-loading-admission.md)
追記 2026-08-29 / [0071](../decisions/0071-manifest-v3-shards.md) 決定 4 撤回・経緯は git）:

- API 工事 4 件（2026-08-19 採択 CX-4.1/4.2/4.3/3.2）: `ResidentWeight` union +
  `planWeightResidency` 純関数プランナ / `ModelShard {id, bytes}` と失敗の実名帰属 /
  `prepareModel → estimate → createSession` の 2 段境界（既存 3 面も内部一本化）/
  `AdmissionReport`（prefill / decode シナリオ + `peakAccountedBytes`）
- hub `prefetchAssets`（相 1 単体面）+ models 7 pipelines の graph-first 接続（admission が
  重み DL 前・進捗はモデル全体 1 本・ロード時に重み shard を落とし切る）
- exporter 自動分割（`karume.shards` — 1GiB・co-shard・決定的・1GiB 以下はバイト不変）+
  dist の複数 shards 要素・デモのローカル読みを疑似 HF サーバで本番経路と 1 本化
  （PNG バイト一致で無風を証明）
- **受け入れ実証**: Base f16 3.9GB → 4 shard の dist 全門通過・実ロード + 512² 生成完走
  （従来は Chromium 上限で原理的に不能）。フル verify 1815/0/5 時点 + 各フェーズ実 GPU 緑

### 消化済み（0.6.0 yomi 依存分離 — 2026-08-25）

結果だけ残す（設計の正本 = ADR [0079](../decisions/0079-sbv2-two-layer-input.md)・経緯は git）:

- SBV2 入力の 2 層化（`Sbv2Phrases` → `toSbv2Utterance` → `Sbv2Utterance` →
  `generate(utterance, options?)` 第一引数）・注入席/辞書席の全廃（ADR 0072 supersede）
- 検証 = **WAV 門 3 sha 不変**（moraTones / moraToPhones 吸収のビット同一性の出荷バイト証明）・
  verify 1771/0/5・配布形 / manifest / pin 不変（HF 再アップロードなし）
- lockstep 0.6.0 → CI 緑 → Release v0.6.0 → JSR publish（3 パッケージ = 0.6.0）。事後疎通 =
  **公開依存リストから `@hdae/yomi` の消滅を API 実測で確定**（0.5.1 の 4 本 → 0.6.0 は
  hub / runtime の 2 本のみ）+ 消費者ストーリー E2E（公開 JSR + yomi 呼び手側 → 構造互換 →
  合成・モーラ tone 編集が波形へ到達）

### 消化済み（0.5.0 breaking 波 + 0.5.1 サンプラー再裁定 — 2026-08-25）

結果だけ残す（経緯は git / 各 ADR / [release-runbook](../release-runbook.md)）:

**0.5.1（ADR [0078](../decisions/0078-anima-sampler-selection.md)）**:

- anima の配布既定サンプラーを Euler へ戻し（HF 上げ直し = anima `2682441a` / turbo
  `88357344`〈越境参照を追随・カード Usage の repo 誤記も修正〉・重みバイト不変を sha256
  全数突合で証明）+ `AnimaGenerateRequest.sampler` 席（DPM++ 2M は選択肢）+ anima 2 pin 更新
- CI 緑 → GitHub Release v0.5.1 → JSR publish（hub / runtime / models = 0.5.1）。事後疎通 =
  0.5.1 消費グラフ解決 + pin 4 定数の期待値一致 + pin 済み `fromPretrained` の実 DL 構築 +
  公開バイトからの e2e golden ビット再現（pin 更新前の同一 revision 実測）

**0.5.0**:

- quant 席名の一斉改名（ADR [0074](../decisions/0074-quant-seat-naming.md)）・`linearCompute` /
  `attentionCompute` の値 `"i8a8"` → `"a8"`・`karume/4` 繰り上げ + 表示欄 + `requiredLimits` +
  越境コンポーネント参照（ADR [0075](../decisions/0075-quant-presentation.md) /
  [0038](../decisions/0038-manifest-v1.md) 追記。`requiredLimits` の DL 前チェック結線は
  release 節に残置）・`fromPretrained` の `ref` 必須化 + pin 定数の公開面出し + 暗黙 main warn
  （ADR [0073](../decisions/0073-models-source-pin.md) 追記）
- anima の `scheduler.type` 席 + DPM++ 2M（出荷バイトの視認 A/B で base / turbo 両採用）・
  base の i4 席 2 つは配布から除外（復活条件つき — later 節）・受理解像度 8 通り縮小（E-2）・
  estimate の恒等別名再現（レビュー R6V-2）・irodori の構築 AbortSignal
- HF 再アップロード 4 リポ: anima `ebb27bc4` / anima-turbo `6215f965`（text stack 5 役を
  anima へ越境参照 — 8.1G → 6.7G）/ jvnv `be752c63` / irodori `49b61517`（`i8+dit4` の pin
  据え置きを解消）。断片化 26〜32 MiB/term（anima の f16 transformer のみ 9.1 = 不変ファイル
  の既存水準で受理）。非公開 `karume-sbv2-fn` も焼き直しのみ実施（公開は parked のまま）
- lockstep 0.5.0（`uv.lock` 追随込み）→ CI 緑 → GitHub Release v0.5.0 → JSR publish
  （hub / runtime / models = 0.5.0）
- 事後疎通（runbook §5）: JSR 0.5.0 の消費グラフ解決と pin 済み `fromPretrained` の
  実 DL + 合成を 4 ファミリで確認

### 消化済み（波 K・リリース + 公開 — 2026-08-20〜21）

**波 K はクローズ**（K-1〜K-5 + 0.4.1 の 6 項目すべて消化）。経緯は git と ADR / runbook が
持つので、ここには結果だけ残す:

- 配布形 `karume/3`（ADR [0071](../decisions/0071-manifest-v3-shards.md)）・SBV2 既定 quant =
  `w8-bert4`（ADR 0039 決定 5 の再裁定）・SBV2 トーン注入席（ADR
  [0072](../decisions/0072-sbv2-text-injection.md)）・pin 焼き込み（ADR
  [0073](../decisions/0073-models-source-pin.md)）
- **HF 公開 = jvnv / irodori / anima の 3 リポ**（2026-08-21・FN は parked）
- **JSR publish = 0.4.0 → 0.4.1**（2026-08-21 ユーザー確認）。0.4.1 は models の公開面が追加
  のみで配布形の作り直し不要だが、**runtime の w4a8（`c285f97` / ADR
  [0076](../decisions/0076-w4a8-linear-execution.md)）を含み `linearCompute: "i8a8"` × i4 常駐の
  出力ビットが変わる破壊的変更**がこの版で初めて配られている（公開 manifest に該当席が無い
  ため patch に載せた裁定 — 0.5.0 の breaking 波とは別枠）
- 残置: R1 同席の API 工事 4 件と exporter の自動分割規則は release 節

手順の正本 = [release-runbook.md](../release-runbook.md)。

autoregressive 波の**残項目（波外へ送り）**:

- **R1 と同席**: manifest の shard 欄は**波 K で消化**（ADR 0071 — `karume/3`）。exporter 側
  shard 分割規則（co-shard を吐く側の保証）は実需（LLM 級配布）まで release 節に残置。
- **MiniCPM5 の token-only 系列**（ADR 0068 追記 4 の同形展開 — models 側 `lastRow` は
  共通化済みで recipe + 門の鏡像だけ。topk の exporter 側〈多出力 aten の getitem 結線〉は
  sampling 実需まで先送りのまま）。
- L8（fake-device 注入面）は保留継続。`enqueue.generation`は2026-09-12に追加した
  （[ADR 0066](../decisions/0066-generation-context-state-slots.md#バッチ実行の-generationcontext2026-09-12)）。
  target→選択グラフの間をCPUへ戻さず接続するための拡張で、token間の依存は変えない。
- 有界論理 extent の席（R2 — IR スキーマ予約のみ・実装は最初の実需モデルまで先送り）・
  bool initializer / storage の設計・pipeline 単位の Session 常駐と device-loss lifecycle
  （perf H-4 と同体）・sampling/RNG はホスト維持（GPU 側は argmax/topk のみ）。
