# Backlog — 波順と作業項目の正本

> プロジェクト全体の**優先順位・波順・未消化項目**の正本はこの 1 本。
> 状態語彙: `now`（現行波）/ `next`（次の大波）/ `later` / `release` / `parked`（復活条件つき）。
> 運用契約: ①完了した項目は**削除する**（履歴は git と ADR / research が持つ）②実測値・設計論証は
> ここに**書かない** — 出典（ADR / research / 台帳）を指す ③性能候補の起票・採否・kill 基準は
> [perf-ledger](perf-ledger.md) が正本で、ここは波として参照するだけ ④by-design 制約の正本は
> [limitations](limitations.md) — 作業化が裁定された時だけここに載る。

## now — 既知問題 3 件 + anima 素版 i4 感度（2026-08-25 着手）

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
  未検証軸（校正標本増・Hessian rank 仮説ほか）は research に列挙 — 追試は GPU 校正
  （`--calib-device cuda`・等価性検証中）が通ってから数十分オーダーで回せる

## next — R1 ロード面工事 + shard 配布（2026-08-25 昇格・2026-08-28 統合裁定）

release 節の「R1 の残り: ロード面 API 工事 4 件 + exporter 自動分割規則」を次の大波へ昇格
（項目詳細は release 節のまま）。理由: Chromium の ArrayBuffer 上限で Base f16 が現配布形の
まま恒久に開けない + モバイルの常駐 RAM 削減（streamAssets 接続）の根本もここ。

2026-08-28 ユーザー裁定: API 工事 4 件と shard 配布は**1 波に統合**し、順序は API 工事が先
（models 接続後に 2 段境界化で同じ継ぎ目を 2 度触らないため）: ①`ResidentWeight` union →
②estimator 改名/シナリオ別 → ③identity 境界 + 2 段境界 → ④models 7 pipelines の shard 面
接続 → ⑤exporter 自動分割（**閾値 1GiB**・co-shard 保証・決定的分割）→ ⑥Base f16 再 dist。
受け入れ = Base f16 が実ブラウザ相当の制約下でロード可能（E2E）。HF 上げ直しは許可ゲート。

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
  場所の特定。系列 + 視認物は `outputs/series-archive/2026-08-23-anima-base-i4/` に退避済みで
  校正結果は再利用可（正本 =
  [research/2026-08-24-gptq-expansion-quality.md](research/2026-08-24-gptq-expansion-quality.md) §5）。
  turbo 側の i4 席で**未検証のまま残した可能性の一覧**（専用幾何・g16・校正量・もう 1 つの
  劣化機序 — いずれも「試してダメ」ではなく「試していない」）は
  [research/2026-08-21-anima-i4-seat-speed.md](research/2026-08-21-anima-i4-seat-speed.md) §8。
- **irodori adaLN i8 の出荷リグ A/B（起票 2026-08-24）**: sim で効いた adaLN i8（+13.1 MiB）が
  出荷リグでも読み上げ方を改善するかは未検証（sim → 出荷の転移限界 — 同 research §2）。
  復活 = `i8+dit4` 席（旧 `w4`）の品質不満、またはサイズ最適化の実需。
- **生成 API 波（起票 2026-08-19 — 全体レビューの Codex 提案を採用裁定）**: 静的配線と
  リクエストを分離した `GenerationProgram`（setup 時に全結線を検証）+ stateful sequence API
  （`for await` の token イベント・EOS 停止・cancel・多ターン継続）+ `last_row` の runner 側
  導出（ADR 0068 追記 4 の所有関係のみ reopen）。`generateGreedy` は parity 検収用の内部
  ヘルパへ格下げ。LLM 実需（streaming / チャット）に直結する最大の API 波。
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
- EmbeddingGemma の完成（models pipeline / tokenizer〈Gemma SPM BPE + byte_fallback 新規実装〉/
  配布形・batch>1 export・runtime attention_mask 配線）。
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
  `signal.reason` 素通し）なので、流儀の先例はそこ。
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

- **R1 の残り: ロード面 API 工事 4 件 + exporter 自動分割規則**（manifest の shard 欄自体は
  波 K で消化 — ADR [0071](decisions/0071-manifest-v3-shards.md)。凍結が要る資産側の形は
  済み、コード API は公開後も動かせる）。
  **同席裁定（2026-08-19 全体レビュー — Codex 提案の採用）**: ①shard identity
  （`{id, bytes}`）の hub↔runtime 境界保存 ②`prepareModel(graphShard) → estimate →
  createSession(weightShards)` の 2 段境界（admission を重み DL 前へ — ADR 0070 の
  graph-first に沿う）③estimator のシナリオ別報告 + `peakAccountedBytes` 改名
  ④重み常駐の判別 union 化（`ResidentWeight` — 3 並列 map の統合）。
  exporter 側 shard 分割規則（co-shard 保証）は最初の LLM 級配布と同時。
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

- **export-recipes の別リポジトリ分離**（ユーザー意向 2026-08-15 — 現時点では何もしない）。
  ADR 0065 案 B〈別配布物化〉のリポ版。切り出し時の論点 = `_shared/paths.py` の REPO_ROOT
  導出・runtime 適合 fixture（packages/runtime/tests/fixtures/）の共有・goldens の出力先・
  uv workspace の解体。復活 = ユーザー裁定。

- **karume-sbv2-fn の HF 公開**（2026-08-20 保留裁定 — 波 K で一時「出典表記つき公開」へ
  振れたが撤回）。upstream の書面条件 = Booth 頒布ページの「商用可・クレジット不要・マージ
  自由」のみで**再配布は未言及**・配布者の素性も未確認。復活 = 配布者への再配布可否の確認、
  またはユーザーの再裁定。ローカル配布形・カード機構は維持（上げる作業だけが保留）
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
