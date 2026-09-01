# Backlog — 波順と作業項目の正本

> プロジェクト全体の**優先順位・波順・未消化項目**の正本はこの 1 本。
> 状態語彙: `now`（現行波）/ `next`（次の大波）/ `later` / `release` / `parked`（復活条件つき）。
> 運用契約: ①完了した項目は**削除する**（履歴は git と ADR / research が持つ）②実測値・設計論証は
> ここに**書かない** — 出典（ADR / research / 台帳）を指す ③性能候補の起票・採否・kill 基準は
> [perf-ledger](perf-ledger.md) が正本で、ここは波として参照するだけ ④by-design 制約の正本は
> [limitations](limitations.md) — 作業化が裁定された時だけここに載る。

## now — 数値レビュー後続 + リリース準備（2026-09-01 棚卸しで更新）

網羅レビュー（正本 = `.claude/reviews/2026-08-29_9614ba9/` — git 追跡外）の確定 50 件 + 追補は
**修正波 A〜E で全消化（2026-08-30）**。破壊的変更 2 件（`BatchScope.finish()` のホスト側失敗
throw / 同一 `GenerationContext` への並行発行拒否）は [limitations](limitations.md) に記載。
残件の裁定は 2026-08-30 に出揃った:

- **shard 仕様 v2（対話裁定 2026-08-30・実装中）**: グラフ専用 shard（shard 0 = `karume_ir`
  のみ・データ節空）+ **上限 1GiB の単一定数**（尾部スラック 1.5GiB 廃止・本数 = 上限下の
  最小分割 → 均し詰め）+ 常時分割（fat graph shard と単一ファイル配布形は廃止・受理保証
  なし）。読み手契約と書き手ポリシーの 2 層構造で、層/MoE 境界の cut 選好はポリシー側の
  将来拡張（正本 = ADR [0081](decisions/0081-shard-spec-v2.md)）。ローカルは完了（2026-08-30 —
  series 241 本 repack・dist 4 リポ再生成・新旧 59,803 テンソルのビット同一証明・lockstep
  0.8.0）。**後続 = 次リリース時に一括（2026-08-30 ユーザー裁定）**: HF アップ（anima /
  irodori / jvnv）→ base SHA で turbo 越境焼き → turbo アップ → pin 4 本更新、を
  **アップロードと SHA 更新まで含めて Claude が実施**（ユーザー明示許可済み — hf upload の
  分類器拒否はこの許可で通す）。**時期（2026-09-01 裁定）: BiRefNet 等の他家族分も揃えてから・
  アップ直後にリリースする形 — 準備が十分整うまで着手しない**。リリース判定は turbo 再焼き後の
  e2e 全緑（現在の ignored +8 = turbo ミラー不在の想定内 SKIP）が条件。karume-sbv2-fn は**アップしない**（非公開のまま —
  e2e の WAV sha 門と parity は 2026-08-30 に jvnv へ付け替え済みで、fn ローカルミラーは削除。
  再生成 = assets-layout の dist コマンドで `inputs/sbv2/FN*` から）
- ~~LLM 先行波~~ **消化済み（2026-08-30）**: L-0 = decode 初回実測（**≈85ms/token・
  `wi4g32` カーネル律速が確定** — フェンス床支配の読みは覆った。正本 =
  [research/2026-08-30-gemma4-decode-wallclock.md](research/2026-08-30-gemma4-decode-wallclock.md)・
  K-11 起票）/ L-1 = sliding スロット容量の window 実数宣言（ADR 0066 追記 9・decode +
  token 2 系列再 export・token 列 parity 不変）/ L-10 = 融合カウント門を gemma4 /
  minicpm5 decode 資産へ（実数固定: gemma4 rope 15@M=1・minicpm5 rope 48 + silu 24）
- **生成 API 波 — 設計正本化済み・実行計画（2026-08-31 裁定 10 点すべて★推奨案）**: 正本 = ADR
  [0083](decisions/0083-generation-api-surface.md)（API 面）/
  [0084](decisions/0084-gemma-tokenizer-chat.md)（tokenizer・detokenizer・chat）/
  [0085](decisions/0085-ple-host-gather.md)（PLE 配布形）+ ADR 0068 追記 6（最終行 logits 出口の
  製品採用）。候補比較・棄却理由・実資産の実測は
  [research 2026-08-31](research/2026-08-31-generation-api-design-draft.md)。
  **段 0〜5 すべて消化済み（2026-08-31〜09-01 — 全合格線達成・波クローズ）**: 段 0 = ADR 3 本 +
  0068 追記 6 / 段 1a = tokenizer compile-to-asset + BPE merge queue + streaming detokenizer
  （HF fixture ビット一致・EG 同乗）/ 段 1b = 製品系列 `gemma4-e2b-product`（容器 1,512MiB・
  PLE sidecar + loader・交差 parity 厳密一致）/ 段 2 = sampler + EOS 集合停止 +
  `generateGreedy` 格下げ（**breaking** — limitations 記載）/ 段 3 = `GenerationProgram` +
  `GenerationSequence`（AsyncIterable・AbortSignal・多ターン pendingToken 連結）/ 段 4 =
  `gemma4ChatPrompt` + `Gemma4Pipeline`（barrel + `./gemma` 配線・**文字列 in → 文字列 out を
  実重みで実証**）/ **段 5 = 配布形（2026-09-01）**: 配布ミラー `models/karume-gemma4-e2b/`
  （3.8GiB・manifest は既存欄のみ・PLE sidecar と tokenizer は assets・`pipelineConfig` に
  sampler 推奨値 = ADR 0083 決定 7 の完成）+ モデルカード（Apache 2.0 帰属 + 上流誘導 +
  LICENSE/NOTICE 同梱）+ `fromPretrained`（hub の遅延資産席 `eagerAssets` 新設 — PLE 2.27GiB を
  常駐させない）+ 疑似 HF サーバで実 DL 疎通（`fromPretrained → chat` golden 一致）。
  **ライセンス門（ADR 0065 stage 6）は 2026-09-01 に消化** — Gemma 4 は **Apache 2.0**
  （snapshot README frontmatter + license_link 本文で現物確認。「Gemma ToU」記述は撤回済み）。
  **HF 公開は次リリース一括に同乗（2026-09-01 裁定）**: 新規リポ `karume-gemma4-e2b` の作成・
  アップ・`GEMMA4_CURRENT` pin 焼き込み・事後疎通を、anima/irodori/jvnv の shard v2 再アップ +
  turbo 越境 2 巡 + pin 4 本と同じ回で実施（素材は完備）。リリースノートは検証 WF 経由で、
  **breaking 2 件**（`generateGreedy` 公開削除 / 会話切り詰めのホスト責務化）+ 新面 `./gemma` を
  記載（2026-09-01 裁定 OK）。**capacity は引き上げ裁定済み（2026-09-01）**: 1024（RoPE 表
  上限）へ上げる — full スロット +5MB 級でほぼ無料。**それ以上に利点があれば RoPE 表の
  再 export も可**（表は 6MiB/1024 行級・full KV は C×12KB 級で伸ばしやすい — 上げ幅は
  対話 example 波で decode 速度の P 依存〈full 側 KV 読みが P に線形〉と合わせて確定）
- **L-11 裁定（2026-08-30）**: 技術先行 = **gemma4 E2B**（品質実証済み — tokenizer / L-5 の
  実装対象）。ライセンス門は上記のとおり**消化済み（Apache 2.0）**。配布経路の
  minicpm5 先行は**採らない**（2026-08-31 裁定 10 — 段 5 の対象は gemma4 E2B のみ）
- **対話 example 波（2026-09-01 起票）**: ①② **消化済み（2026-08-31 実装・10 コミット
  e7e53dd〜dce91c7）**。公開面レビュー（Opus2+Codex3 → 敵対検証 15 判定 refuted 0・正本 =
  `.claude/reviews/2026-08-31_182ced7/SUMMARY.md`）→ 裁定どおり: ローカルローダー =
  **取得元抽象 DistributionSource**（ADR [0086](decisions/0086-distribution-source.md) —
  `denoDirectory(models/karume-gemma4-e2b)` を `fromPretrained` へ直渡し・CacheStorage
  複製ゼロ・越境は明示 mapping）+ 検証済みバグ修正（PLE 解放口 / signal 伝播 / headers doc
  8 家族ほか）+ 公開面調整（`gemma4ChatTurn` 増分描画・`parseGemma4PipelineConfig` 公開・
  `GenerationProgram` 絞り込み・`defaultSampler` 改名・`used`/`GenerationCapacityError`
  構造化欄・`GenerationStop.tokens`）+ `examples/gemma4/`（sequence KV 継続の写経見本・
  デモ 4 本も denoDirectory 移行）。フル verify 2043/0/13 緑。
  **残 = ③capacity 1024 の反映**（dist 再生成 — 上の引き上げ裁定・>1024 の再 export 判断は
  decode の P 依存実測と同時。**ユーザー意向〈2026-09-01〉= コンテキスト窓は可能な限り大きく
  — 1024 反映を先行し、上げ幅の最大化は P 依存実測とセットで検討**）。起票のみ（第 2 波候補）: ChatSession 高レベル面・stop strings・
  `chatText()`・onProgress 可読化・maxResidentPleBytes・logitBias 配列化・防御コピー
  （prompt/options の発行時スナップショット）・example の residentPleShards フラグ化
- ~~数値危険クラス監査波~~ **本体消化（2026-08-31 — OP 数値レビューとして拡大実施）**:
  ①台帳化 = [research/2026-08-31-op-numerics-review.md](research/2026-08-31-op-numerics-review.md)
  （危険クラス台帳・C 0 / E 0・レビュー原本 = `.claude/reviews/2026-08-31_b35cf5c/`）
  ②修正 = gru_scan tanh_stable 化（is_nan_bits 正本化と同時）+ 門 4 点 + doc 訂正群 +
  Box–Muller + exporter act_quant 鏡像化 ③門の常設 = 飽和域の厳密カナリア（±1.0/x/−0.0 +
  ±Inf）+ op-vocabulary へ危険クラス門を規約化 ④W-2/W-3 裁定 a 消化 = softmax 族 nan_max
  統一 + 融合 attention 空行ガード（breaking 2 点 — ADR 0044 追記）⑤W-5 消化 =
  DEFAULT_TOLERANCE 退役 → op 別実測表 + ビット同一門
  （[research/2026-08-31-op-tolerance-measurement.md](research/2026-08-31-op-tolerance-measurement.md)）。
  **Mac（M2）手動検証も消化（2026-08-31 実測）→ 波クローズ**: 飽和域厳密門 緑（根治実証）・
  ビット同一格子門 緑・vowel golden 緑（実品質健全）・フル verify の赤 12 本はすべて既知クラス
  へ帰着（conv1d parity 2 本の台帳漏れを訂正 + **新規記載 = gru_scan 分解 parity の Metal
  1〜64 ULP** — 変更前 HEAD との A/B で同一署名 = v2 regression でないことを確定・
  known-issues Metal 節が正本）。tools/metal-diagnostics/ は削除済み（復元は git 履歴）。
- **数値レビュー後続の起票（2026-08-31）**:
  - **Metal OOM errorScope 沈黙**（known-issues — 重み経路の明示 size 門 + `requiredLimits`
    のロード時実効化。監査波から分離・独立に着手可）— **次波へ昇格（2026-09-01 ユーザー指示）**:
    メモリ管理周りの改善・検証の波として、現時点で進められるもの（棚卸しの文書修正）の
    消化後に着手
  - gemv margin 命題の M2 温度 0 golden 実測（ADR 0082 追記 2 の立て直し）
  - ~~GPTQ static-groups + act-order 実験 / damping sweep~~ **実装 + 実測消化（2026-08-31）**:
    軸は opt-in で実装済み（既定 off ビット同一 — `892bfb3`/`683d6a0`）・minicpm5 8 構成の
    実測で **act-order / static は本条件で利得なし・damping 0.01 は実測で封印 → 既定は現状維持**
    （正本 = [research/2026-08-31-gptq-axes-sweep.md](research/2026-08-31-gptq-axes-sweep.md)）。
    **ユーザー裁定（2026-08-31）: 既定現状維持で確定・opt-in 実装は温存**。復活条件 =
    **多モデル**（act-order の効果はモデル依存の見立て）× 校正量 16× での再評価 —
    時間のある時に別波で（gemma4 校正 rig 新設もそこまで保留）
  - norm の 1/dim ホスト化は**保留**（uncertain — 開発機の除算実測から凍結 sha が割れ得る。
    実 GPU プローブが先・費用対効果低）・reduce identity の params −inf 化は現状維持
    （W-2/W-3 と同じ器 — 採るならセット裁定）
  - tolerance B 案（`allclose` へ縮約スケール項 `κ·√K·max|期待|` — 公開 API 変更を伴う後続
    改善。A 案 = op 別表は 2026-08-31 実装済み。研究 §8.2）・layer_norm の悪条件入力
    （分散 ≈0 の摂動）はケース個別 tolerance の席で扱う（同 §7 注記）
- ~~M2 実機の手動確認 2 点~~ **消化（2026-09-01 実測）**: dp4a カナリア **16/16 緑**（QK f16
  格子化後の初実測）・軸 reduce パリティ **2/2 緑**（旧記述の「4 本」は誤記・known-issues への
  読み方ポインタも切れていた）。**新規 = gemv u32 門が M2 で 1 ULP 赤 → 裁定済み（既定維持
  — GEMV 固有と切り分け確定・ADR 0082 追記 1・[known-issues](known-issues.md) Metal 節）**
- anima-web の cold ロード DL スロット改善（提案 b+a — `FamilyAdmission` 席は実装済みで、
  残りは admission 前倒しの graph shard 単位化 + extras の並行開始。shard 仕様 v2 で
  graph shard が数 MB になり前倒しの価値が確定する。opt-in の c 案は再裁定要）
- perf: レンズ E-1 は裁定済み — **P-1〜P-3 スパイク承認・P-4 起票・P-5 計測のみ** +
  M1-2 代償の L-9（いずれも [perf-ledger](perf-ledger.md)）。既存起票 H-8〜H-10 / L-7 / L-8。
  K-11 は**消化済み（2026-08-31・ADR
  [0082](decisions/0082-linear-gemv-decode.md) — decode 84.2→32.5ms/token・ビット同一）**。
  復活条件が満ちた再評価候補 = **レンズ L-7 / L-12**（decode がフェンス床支配へ戻ったため —
  research 2026-08-30 §7。perf-ledger の L-7 とは別番号系）+ 隣接起票候補 = lm_head `wi8`
  M=1（decode GPU の 23% の新 2 位・機序は別）

## 消化済み（既知問題 3 件 + anima 素版 i4 感度 — 2026-08-25〜28）

Anima Web アプリからの既知問題 3 件（調査で機序確定済み — 経緯は git / ACTIVE_DESIGN）と、
素版 i4 の量子化感度特定（later 節からの前倒し — 配布スキップ裁定の復活レバー）:

- ①Pixel の "BodyStreamBuffer was aborted" — hub の真因マスキング解消 + バイト予算 +
  検証直列化は**済**。実機での真因再判定（err.cause 観測）はリリース後 —
  [known-issues](known-issues.md)
- ②NVIDIA の 2GiB 天井（Dawn D3D12 固定値）— 融合 attention の行ブロック化は**済**
  （parked「2048px DiT attention メモリ工事」の消化）
- ③Chromium の単一 ArrayBuffer 上限で Base f16 がロード不能
  （[limitations](limitations.md)）— 根本 = next の R1 shard 配布を優先（2026-08-25 裁定）。
  DL 前の即エラーは fetch-cache 0.5.0 の `expectedBytes` 即 throw + hub 追従で**済**
  （2026-08-28 — 受信前に `cause` = RangeError で落ちる。ADR
  [0080](decisions/0080-hub-fetch-cache-050.md)）
- ⑤fetch-cache 0.5.0 追従（hub）— 検証責務の移譲（記録ハッシュ信頼・knob なし）・認証隔離の
  撤去（ユーザー裁定: gated 運用予定なし）・`AssetPhase` から `verifying` 撤去・旧名前空間
  `karume/1` 系 purge・`clearHubCache` の対象変更。正本 = ADR
  [0080](decisions/0080-hub-fetch-cache-050.md)（旧 CAS ドラフトを置換 — `archive/hub-cas-0.5.0`
  の再適用は不要になった）
- ④素版 i4 感度 — adaLN + block 外 i8 変種は**視認スイープで不採用**（2026-08-28 裁定 —
  perf-ledger Q-9 /
  [research](research/2026-08-28-anima-adaln8-visual.md)。教訓: 視認 A/B は seed 4 本以上）。
  **anima DiT i4 系はしばらく保留（2026-08-28 ユーザー裁定）** — 動機だった「サイズ起因の
  DL 不能」は R1 shard 化が根治し、速度は i4 経路がむしろ遅い（~2 倍）ため優先度が立たない。
  未検証軸は research に列挙のまま（復活時は GPU 校正 =
  [実用可・3.6 倍速](research/2026-08-28-cuda-calibration.md)で回す — 配布焼きは CPU）

## 消化済み（0.7.0 リリース — 2026-08-29 完了）

HF 更新系は**完了（2026-08-29）**: 全席分割の再 export 8 本（**全テンソルビット同一証明** —
LoRA scale=1.0 も同時証明）→ base 3 モデル family 再生成 → HF 上げ → turbo を**shard ごとの
越境参照**（新機構の初適用）で焼き直し → HF 上げ → pin 2 本焼き込み + 実 DL 疎通
（turbo = demo 完走 / base = fromPretrained + 生成完走）。**公開 revision の正本は pin 定数**
（`ANIMA_CURRENT` / `ANIMA_TURBO_CURRENT` — `packages/models/src/anima/config.ts`）で、docs には
写さない（尾部スラック則の反映で両リポとも焼き直したように、SHA は後から動く）。断片化検証:
全 shard 26.5〜30.4 MiB/term（健全）— 例外は base の `shared/text_encoder` shard1 =
**4.5 MiB/term（旧公開バイトの xorb へ部分ヒットした継承断片化** — 同バイト再アップは
hf CLI が転送スキップするため runbook の処方が効かない。delete→再 up の 2 コミット法も
**不発を実測済み**（hf_xet 1.4.3 退行 — [known-issues](known-issues.md)）。恒久対処候補 =
known-issues の 3 案（版固定再検証 / `HF_HUB_DISABLE_XET=1` / 履歴整理）に集約）。

- **Release v0.7.0 published → JSR 3 パッケージ publish 完了（2026-08-29 ユーザー確認）**。
  リリースノートは公開前に検証ワークフロー（主張突合 + 両方向網羅）を通した — 修正 2 +
  Breaking 追記 1（`from*Assets` は分割リポを開けない）+ 補足 4 を反映
- 2026-08-29 裁定 3 件は**消化済み**: ①コーパスは `demo:eval-images --source
  models/karume-anima-turbo`（正本の役割別プロンプト）で再生成し 3 ファミリの golden を
  採り直した（意味論門込み緑）②断片化は**クライアント退行で現状の手が尽きた**ことを実測で
  確定し記録（[known-issues](known-issues.md)・runbook §2 NOTE — 恒久候補 3 案は
  known-issues）③尾部スラック則（未閉 ≤1.5GiB は詰め切る — `SHARD_TAIL_LIMIT`）で端数
  shard を廃し、turbo i4 の祖父条項は**規則上の正会員**になった（1.14GiB ≤ 1.5GiB。
  → 尾部スラック則自体は 2026-08-30 の shard 仕様 v2 で廃止 — ADR 0081）
- リリース後 = ChatGPT 全体レビュー消化（ユーザー持参）・Pixel 実機 err.cause 再判定

## 消化済み（R1 統合波 — ロード面 API 工事 + shard 配布・2026-08-28〜29）

結果だけ残す（設計の正本 = ADR [0070](decisions/0070-shard-loading-admission.md)
追記 2026-08-29 / [0071](decisions/0071-manifest-v3-shards.md) 決定 4 撤回・経緯は git）:

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

## 消化済み（0.6.0 yomi 依存分離 — 2026-08-25）

結果だけ残す（設計の正本 = ADR [0079](decisions/0079-sbv2-two-layer-input.md)・経緯は git）:

- SBV2 入力の 2 層化（`Sbv2Phrases` → `toSbv2Utterance` → `Sbv2Utterance` →
  `generate(utterance, options?)` 第一引数）・注入席/辞書席の全廃（ADR 0072 supersede）
- 検証 = **WAV 門 3 sha 不変**（moraTones / moraToPhones 吸収のビット同一性の出荷バイト証明）・
  verify 1771/0/5・配布形 / manifest / pin 不変（HF 再アップロードなし）
- lockstep 0.6.0 → CI 緑 → Release v0.6.0 → JSR publish（3 パッケージ = 0.6.0）。事後疎通 =
  **公開依存リストから `@hdae/yomi` の消滅を API 実測で確定**（0.5.1 の 4 本 → 0.6.0 は
  hub / runtime の 2 本のみ）+ 消費者ストーリー E2E（公開 JSR + yomi 呼び手側 → 構造互換 →
  合成・モーラ tone 編集が波形へ到達）

## 消化済み（0.5.0 breaking 波 + 0.5.1 サンプラー再裁定 — 2026-08-25）

結果だけ残す（経緯は git / 各 ADR / [release-runbook](release-runbook.md)）:

**0.5.1（ADR [0078](decisions/0078-anima-sampler-selection.md)）**:

- anima の配布既定サンプラーを Euler へ戻し（HF 上げ直し = anima `2682441a` / turbo
  `88357344`〈越境参照を追随・カード Usage の repo 誤記も修正〉・重みバイト不変を sha256
  全数突合で証明）+ `AnimaGenerateRequest.sampler` 席（DPM++ 2M は選択肢）+ anima 2 pin 更新
- CI 緑 → GitHub Release v0.5.1 → JSR publish（hub / runtime / models = 0.5.1）。事後疎通 =
  0.5.1 消費グラフ解決 + pin 4 定数の期待値一致 + `fromPretrained(*_CURRENT)` 実 DL 構築 +
  公開バイトからの e2e golden ビット再現（pin 更新前の同一 revision 実測）

**0.5.0**:

- quant 席名の一斉改名（ADR [0074](decisions/0074-quant-seat-naming.md)）・`linearCompute` /
  `attentionCompute` の値 `"i8a8"` → `"a8"`・`karume/4` 繰り上げ + 表示欄 + `requiredLimits` +
  越境コンポーネント参照（ADR [0075](decisions/0075-quant-presentation.md) /
  [0038](decisions/0038-manifest-v1.md) 追記。`requiredLimits` の DL 前チェック結線は
  release 節に残置）・`fromPretrained` の `ref` 必須化 + `*_CURRENT` 公開 + 暗黙 main warn
  （ADR [0073](decisions/0073-models-source-pin.md) 追記）
- anima の `scheduler.type` 席 + DPM++ 2M（出荷バイトの視認 A/B で base / turbo 両採用）・
  base の i4 席 2 つは配布から除外（復活条件つき — later 節）・受理解像度 8 通り縮小（E-2）・
  estimate の恒等別名再現（レビュー R6V-2）・irodori の構築 AbortSignal
- HF 再アップロード 4 リポ: anima `ebb27bc4` / anima-turbo `6215f965`（text stack 5 役を
  anima へ越境参照 — 8.1G → 6.7G）/ jvnv `be752c63` / irodori `49b61517`（`i8+dit4` の pin
  据え置きを解消）。断片化 26〜32 MiB/term（anima の f16 transformer のみ 9.1 = 不変ファイル
  の既存水準で受理）。非公開 `karume-sbv2-fn` も焼き直しのみ実施（公開は parked のまま）
- lockstep 0.5.0（`uv.lock` 追随込み）→ CI 緑 → GitHub Release v0.5.0 → JSR publish
  （hub / runtime / models = 0.5.0）
- 事後疎通（runbook §5）: JSR 0.5.0 の消費グラフ解決と `fromPretrained(*_CURRENT)` の
  実 DL + 合成を 4 ファミリで確認

## 消化済み（波 K・リリース + 公開 — 2026-08-20〜21）

**波 K はクローズ**（K-1〜K-5 + 0.4.1 の 6 項目すべて消化）。経緯は git と ADR / runbook が
持つので、ここには結果だけ残す:

- 配布形 `karume/3`（ADR [0071](decisions/0071-manifest-v3-shards.md)）・SBV2 既定 quant =
  `w8-bert4`（ADR 0039 決定 5 の再裁定）・SBV2 トーン注入席（ADR
  [0072](decisions/0072-sbv2-text-injection.md)）・pin 焼き込み（ADR
  [0073](decisions/0073-models-source-pin.md)）
- **HF 公開 = jvnv / irodori / anima の 3 リポ**（2026-08-21・FN は parked）
- **JSR publish = 0.4.0 → 0.4.1**（2026-08-21 ユーザー確認）。0.4.1 は models の公開面が追加
  のみで配布形の作り直し不要だが、**runtime の w4a8（`c285f97` / ADR
  [0076](decisions/0076-w4a8-linear-execution.md)）を含み `linearCompute: "i8a8"` × i4 常駐の
  出力ビットが変わる破壊的変更**がこの版で初めて配られている（公開 manifest に該当席が無い
  ため patch に載せた裁定 — 0.5.0 の breaking 波とは別枠）
- 残置: R1 同席の API 工事 4 件と exporter の自動分割規則は release 節

手順の正本 = [release-runbook.md](release-runbook.md)。

autoregressive 波の**残項目（波外へ送り）**:

- **R1 と同席**: manifest の shard 欄は**波 K で消化**（ADR 0071 — `karume/3`）。exporter 側
  shard 分割規則（co-shard を吐く側の保証）は実需（LLM 級配布）まで release 節に残置。
- **MiniCPM5 の token-only 系列**（ADR 0068 追記 4 の同形展開 — models 側 `lastRow` は
  共通化済みで recipe + 門の鏡像だけ。topk の exporter 側〈多出力 aten の getitem 結線〉は
  sampling 実需まで先送りのまま）。
- L8（fake-device 注入面）は保留継続・`enqueue` の generation 面は設けない裁定で確定
  （limitations）。
- 有界論理 extent の席（R2 — IR スキーマ予約のみ・実装は最初の実需モデルまで先送り）・
  bool initializer / storage の設計・pipeline 単位の Session 常駐と device-loss lifecycle
  （perf H-4 と同体）・sampling/RNG はホスト維持（GPU 側は argmax/topk のみ）。

## later

- **カードの Usage repo 導出の硬化（起票 2026-08-25）**: `karume.dist` はカードの Usage 例の
  repo 名を**出力ディレクトリ名**から導出するため、越境参照のステージング焼き（`--out` が
  別名）で誤った repo 名がカードに載る（実害 = turbo カードに `-release` 付き誤名が公開されて
  いた — ADR [0078](decisions/0078-anima-sampler-selection.md) Consequences・runbook §0 に
  運用注意を追記済み）。恒久策 = `Pipeline.repo_name` 系の正本から導出し `--out` 名へ依存
  しない形。
- **examples/anima に `--sampler` ノブ（起票 2026-08-25）**: request 側 `sampler` 席
  （ADR 0078）を CLI デモから振れるようにする小改修。
- **anima 素版 i4 の品質改善（起票 2026-08-24 — 配布スキップ裁定の復活レバー）**: adaLN の
  i8 化（irodori の帰属で効いた知見の移植 — anima では**未実測の仮説**）と量子化感度の高い
  場所の特定。校正済み系列は `outputs/series/` の `*-i4-dyn` に温存（旧 series-archive の
  退避分と視認物は 2026-08-30 の掃除裁定で削除 — i4 は結局再調整が要るため。実測記録の正本 =
  [research/2026-08-24-gptq-expansion-quality.md](research/2026-08-24-gptq-expansion-quality.md) §5）。
  turbo 側の i4 席で**未検証のまま残した可能性の一覧**（専用幾何・g16・校正量・もう 1 つの
  劣化機序 — いずれも「試してダメ」ではなく「試していない」）は
  [research/2026-08-21-anima-i4-seat-speed.md](research/2026-08-21-anima-i4-seat-speed.md) §8。
- **irodori adaLN i8 の出荷リグ A/B（起票 2026-08-24）**: sim で効いた adaLN i8（+13.1 MiB）が
  出荷リグでも読み上げ方を改善するかは未検証（sim → 出荷の転移限界 — 同 research §2）。
  復活 = `i8+dit4` 席（旧 `w4`）の品質不満、またはサイズ最適化の実需。
- ~~生成 API 波（起票 2026-08-19）~~ **now 節へ昇格・設計正本化済み（2026-08-31）** — 起票が
  書いていた形（`GenerationProgram` / stateful sequence / `generateGreedy` 格下げ）は
  ADR [0083](decisions/0083-generation-api-surface.md) が正本。tokenizer は
  [0084](decisions/0084-gemma-tokenizer-chat.md)・PLE 配布形は
  [0085](decisions/0085-ple-host-gather.md)。実行計画と各段の合格線は **now 節**。
- **バレル・ファミリープレフィックスの見直し（起票 2026-08-25 — ユーザー意向「今後見直し
  たい」）**: `mod.ts` の全ファミリ平面 export のためにシンボルへ族名プレフィックスが付くが、
  SBV2 族では「2」が変換イディオム（x2y）に誤読される実害が出た（`sbv2Utterance` →
  `toSbv2Utterance` へ命名回避 — ADR
  [0079](decisions/0079-sbv2-two-layer-input.md) 決定 2）。サブパス面での素名 export や
  namespace オブジェクト化などの選択肢を全ファミリ横断で再設計する（プレフィックスが外れれば
  `toUtterance` へ収斂できる）。breaking なので次の breaking 波に同乗させる。
- **モデルカード定型文の条件出し**（公開前レビュー minor・2026-08-21）: `shared/` パスの
  説明文が shared/ を持たないリポ（irodori / anima）でも出る — 空回りだが無害。core
  `modelcard.py` の該当文を shared/ 実在時のみ出す形へ。
- **measure_quant の配布試算の J-5b 追随**（J-3 中に発見・2026-08-22）: sbv2
  `project_distribution` が「linear の重みスロットだけ・conv / embedding の i4 は格納形も
  実行経路も無い」という pre-J-5b 前提のまま（実際は出荷済み — ADR 0069 追記 6/7）。相対
  比較には無害だが試算が過小で docstring も陳腐化。対象集合と説明の追随を 1 件で。
- **モデル拡充の続き**: Kokoro-82M（LSTM = multi-output 待ち）・MobileSAM / SAM 2
  （conv_transpose2d）・BiRefNet_HR 2048² preset・DA-V2 可変解像度（upsample_bicubic2d）。
  候補調査の時点記録は [recon-2](research/2026-08-14-model-expansion-recon-2.md)。
- **性能候補**: perf-ledger の 🚧（K-2 VAE conv2d / K-4b conv1d i8a8 / K-6 encoder tile /
  K-9 relattn / L-1 cold-load 分解 / L-2 EG 低精度）— 採否・順序は perf-ledger。
- EmbeddingGemma の完成（models pipeline / 配布形・batch>1 export・runtime attention_mask 配線）。
  **tokenizer〈Gemma SPM BPE + byte_fallback〉の実装と EG 資産 compile は生成 API 波の段 1a に
  同乗**（2026-08-31 裁定 9 — ADR [0084](decisions/0084-gemma-tokenizer-chat.md) 決定 6。実装は
  共用・資産は別 compile）。
- **w8a8 鏡像門の設置**: `e2e_deberta_w8a8_test.ts`（ADR
  [0026](decisions/0026-w8a8-deberta-deployment.md) 決定 3 — `e2e_deberta_test.ts` は移植済み・
  鏡像側だけ未設置。2026-08-16 裁定で起票）。
- **Anima ホスト糊 parity の常設門化**: `sigmaSchedule` / `cfgEulerStep` / `denormalizeLatents` /
  `padSequence` の「fixture と全 4 実装 bit 同一」は recipe README に実測記録として残るだけで、
  `outputs/series/anima-pipeline*` を読む常設テストは Deno / pytest のどちらにも存在しない
  （2026-08-16 判明 — fixture 4 変種は再エミット済みで前提は解消済み）。
- **ORT Web 対比ベンチ慣行**（2026-08-16 ユーザー裁定）: 両対応モデルで定期測定し、
  遅すぎないか・ボトルネックはどこかを調査する。対象 = EmbeddingGemma（models 側の完成が
  前提）と **KokoroTTS**（2026-08-16 訂正 — 当初の Irodori は打ち間違い。Kokoro は
  Transformers 系で動くため比較しやすい・karume 側は Kokoro-82M 対応が前提 = 上のモデル拡充
  候補・LSTM multi-output 待ち）。将来はブラウザ ONNX + PyTorch ネイティブ込みの比較
  マトリクスへ広げる（当面は不要の裁定）。測定条件の規範（graph capture ON /
  freeDimensionOverrides / IO binding / EP 分断確認・native EP か JSEP かの記録）は
  [runtime-landscape §4](research/2026-08-16-runtime-landscape.md) が正本。
- **生成イベントの横展開（需要待ち）**: sbv2 / birefnet / depth / siglip2 / vowel への stage
  イベント（step ループが無く提供できるのは段遷移のみ）と、**生成ループ**の AbortSignal 中断席
  （現状は onEvent の throw が step 粒度の中断手段 — 席は温存）。**構築経路の AbortSignal は
  anima で実装済み**（`AnimaPipelineOptions.signal` — 段境界での検査・取得層への透過・
  `signal.reason` 素通し）なので、流儀の先例はそこ。**生成ループの席は LLM 面で先に開く**
  （ADR [0083](decisions/0083-generation-api-surface.md) 決定 5 — 他 4 家族への横展開は需要待ちのまま）。
- **`AssetProgress.path` が越境参照を識別できない（起票 2026-08-25・優先度低）**: 進捗イベントの
  `path` は文字列 1 本で、越境コンポーネント参照（ADR
  [0038](decisions/0038-manifest-v1.md) §7 追記）が入った以上**別リポの同名 path と区別が
  付かない**（取得層の同一性キーは `fileRefKey` へ移ったが、公開イベント側は `path` のまま）。
  消費側がファイル別の進捗を path でキーにすると 2 本が混ざる。埋め方は `repo` / `revision` を
  イベントへ足すか `fileRefKey` を出すか — 公開面の追加なので breaking 波に乗せる。
- **anima 大解像度の省 RAM タイル逐次組み立て（起票 2026-08-24 裁定）**: VAE decode のタイルを
  貯めずに順次合成できれば、E-2 の入口拒否で受理集合から外した 8 解像度
  （1456/1488/1584/1648/1680/1776/1840/1936px）を受理へ戻せる。
- **hub の sha256 同一ファイルのリポ跨ぎ重複 DL 解消（外部フィードバック提案⑥・優先度低）**:
  content addressing でキャッシュを引けば、同じバイト列を別リポから取り直さずに済む。
  ADR [0038](decisions/0038-manifest-v1.md) のキャッシュ設計（キーは URL）へ踏み込む変更。
- Metal 数値差の原因確定（known-issues）・resident 経路の診断/計測制約の解消。
- MoE の seam（fixed-k routing は静的形で表現可 — dense API に expert 非存在を焼かない）。

## release — リリース準備波（しばらく先）

- ~~R1 の残り~~ **消化済み（2026-08-29 — R1 統合波の節を参照）**: ロード面 API 工事 4 件も
  exporter 自動分割規則も実装完了（ADR 0070 追記 2026-08-29 / ADR 0071 決定 4 撤回）。
- 実資産 CI gate（GitHub CI はローカル資産を踏まない問題）
- HF 公開: **jvnv / irodori / anima の 3 リポは波 K-4 で公開済み**（2026-08-21）。FN は parked
  （再配布の書面根拠なし）。以後の新モデルは runbook に従う
- リポ直下 README の書き上げ・JSR npm 互換層の sideEffects 検証
- ライセンス interview（export-recipes の family 別 provenance を upstream revision 単位で
  人間確認 — 再編の release gate。**公開 4 リポぶんは波 K-4 の人間ゲートで先行実施**）
- 「semantic surface と実装済み subset の分離」方針の再裁定（attention / deform_conv2d /
  gather / conv_transpose1d / upsample_bilinear2d — 観測 subset を op 意味論にしない統一規約）
- 大型 DL 前の limits preflight（DL 後にしか limits 不足が分からない問題）

## parked（復活条件つき）

- **IR への値依存実行選択（MoE エキスパート動的常駐の前提）**（2026-08-31 裁定 — 入れない）。
  エキスパート単位のロード/退避は ①`ShardValidator` 全件門 ②重み常駐の不変 Map ③IR v1 の
  値依存実行選択なし、の 3 重衝突で、機能追加でなく 3 モジュール横断の再設計になる（実測記録 =
  [research 2026-08-31](research/2026-08-31-freetoken-moe-over-arraybuffer.md)）。当面の公式
  スタンス = **MoE は全 expert VRAM 常駐・総パラメータで予算**（[limitations](limitations.md)）。
  「未着荷 initializer」席の新設も本項に従属して見送り（同裁定）。復活 = VRAM に乗らない MoE の
  出荷実需。その際の最初の宿題 = 予測型 offloading（SiDA arXiv:2310.18859 / HOBBIT
  arXiv:2411.01433）の一次精読（読み戻しは消せてもホスト側プール常駐の壁は残る、が現時点の読み）。
- **export-recipes の別リポジトリ分離**（ユーザー意向 2026-08-15 — 現時点では何もしない）。
  ADR 0065 案 B〈別配布物化〉のリポ版。切り出し時の論点 = `_shared/paths.py` の REPO_ROOT
  導出・runtime 適合 fixture（packages/runtime/tests/fixtures/）の共有・goldens の出力先・
  uv workspace の解体。復活 = ユーザー裁定。

- **karume-sbv2-fn の HF 公開**（2026-08-20 保留裁定 — 波 K で一時「出典表記つき公開」へ
  振れたが撤回）。upstream の書面条件 = Booth 頒布ページの「商用可・クレジット不要・マージ
  自由」のみで**再配布は未言及**・配布者の素性も未確認。復活 = 配布者への再配布可否の確認、
  またはユーザーの再裁定。カード機構（`--card-profile fn`）は維持。ローカルミラーは常設
  しない（2026-08-30 裁定 — e2e の門はライセンス記述が正の jvnv へ付け替え・fn ミラー削除。
  再生成 = assets-layout の dist コマンドで `inputs/sbv2/FN*` から）
- **SBV2 `adjust_word2ph` の移植**（2026-08-21 不採用裁定 — ADR 0072 決定 8）。音素数が変わる
  編集（語境界に一致しない読みの差し替え）を受けるための word2ph 再配分。参照は**上流**
  Style-Bert-VITS2 のセマンティクス（LCS 差分 + 1..6 クランプ・残差は例外）で、AivisSpeech が
  pin する fork の「均等増減で無理やり辻褄を合わせる」は採らない（黙って近似しない）。
  復活 = overlay で表現できない読み編集の実需。
- **anima の g16 評価**（2026-08-23 送り — GPTQ 適用拡大を優先するユーザー裁定。SBV2 の
  g 軸裁定はモデル系統を跨いで一般化しない〈research 2026-08-22〉ため評価自体の価値は残す。
  復活 = 素版 i4 の視認で品質不満が出た場合。主作業と衝突しない裏実行での前倒しは可
  〈同日ユーザー裁定・anima 校正リグを触る J-4 ②の着地後に流すのが安全〉）
- hub Range 並列 + prefetch — 復活 = 断片化リポの再来（perf L-3）
- params / bind group キャッシュ（ADR 0032 案 2）・GPU timestamp 推定源化・全面 f16（案γ）・
  Vᵀ+列量子化融合・SBV2 NFC チップ・f32 anima 系列再生成
- by-design 制約群（rank≤4 / OOB NaN / 非有限入力 / 0 要素次元 / cancellation 粒度ほか）は
  limitations.md が正本 — 実需が出た項目だけここへ昇格させる
