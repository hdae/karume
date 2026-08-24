# Backlog — 波順と作業項目の正本

> プロジェクト全体の**優先順位・波順・未消化項目**の正本はこの 1 本。
> 状態語彙: `now`（現行波）/ `next`（次の大波）/ `later` / `release` / `parked`（復活条件つき）。
> 運用契約: ①完了した項目は**削除する**（履歴は git と ADR / research が持つ）②実測値・設計論証は
> ここに**書かない** — 出典（ADR / research / 台帳）を指す ③性能候補の起票・採否・kill 基準は
> [perf-ledger](perf-ledger.md) が正本で、ここは波として参照するだけ ④by-design 制約の正本は
> [limitations](limitations.md) — 作業化が裁定された時だけここに載る。

## now — レビュー消化 → 0.4.3 リリース → 0.5.0 breaking 波

波 J（量子化探索・第 2 段）と波 L（anima 素版 + バリアント同梱）は **2026-08-24 に全クローズ**。
結果と実測の正本は ADR [0050](decisions/0050-irodori-quant-series.md) 追記 2 /
[0077](decisions/0077-model-version-naming.md) と
[research/2026-08-24-gptq-expansion-quality.md](research/2026-08-24-gptq-expansion-quality.md) /
[research/2026-08-22-anima-base-steps.md](research/2026-08-22-anima-base-steps.md)。

- **広域レビュー（`.claude/reviews/2026-08-24_283669a`）の消化**: 外部フィードバック 7 件の
  対応と併せて公開面が 4 件増えた（`ANIMA_BASE_SOURCE` / `animaLatents()` /
  `approximatePreview` / `AssetProgress` の per-file 欄）。docs 同期までがこの波。
- **0.4.3 の JSR リリース**: 0.4.2 以降の未リリース差分（上の公開面 4 件 + anima 構築経路の
  AbortSignal 中断 + hub の進捗欄）を patch で出す。破壊的変更は載せない。手順の正本 =
  [release-runbook](release-runbook.md)。
- **0.5.0 breaking 波の準備**: 束の骨子は later 節の該当項目が正本（quant 席名の規則化 /
  `linearCompute` の値改名 / 表示欄 + `karume/4` / yomi 依存分離）。**現行の席名・ノブ名は
  改名予定**。

持ち越しの注意（次の再アップロード・次の波で先に思い出す）:

- irodori の `w4` 席は HF 公開済み（`67e9584c`）だが **pin は据え置き** — 使うには
  `revision: "main"` の明示が要る（0.5.0 の pin 更新で解消する）。
- local の `models/karume-anima/` には**配布スキップ裁定（2026-08-24）の `w4` / `w4-a8-s16`
  席が組み込まれたまま**なので、karume-anima を上げ直す前に裁定を確認する。
- anima のサンプラーは Euler 固定（manifest に種別欄が無い — 波 L の残置）。

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
  復活 = w4 席の品質不満、またはサイズ最適化の実需。
- **生成 API 波（起票 2026-08-19 — 全体レビューの Codex 提案を採用裁定）**: 静的配線と
  リクエストを分離した `GenerationProgram`（setup 時に全結線を検証）+ stateful sequence API
  （`for await` の token イベント・EOS 停止・cancel・多ターン継続）+ `last_row` の runner 側
  導出（ADR 0068 追記 4 の所有関係のみ reopen）。`generateGreedy` は parity 検収用の内部
  ヘルパへ格下げ。LLM 実需（streaming / チャット）に直結する最大の API 波。
- **0.5.0 breaking 波（起票 2026-08-21・ユーザー裁定）**: 破壊的変更を 1 回にまとめ、公開 4 リポ
  （jvnv / irodori / anima-turbo / anima — 波 L で 1 増）の再アップロードと pin 更新（ADR 0073）を
  1 度で済ませる束。①`linearCompute` の値を
  `"i8a8"` → `"a8"` へ改名（ノブが決めているのは活性の扱いだけで、格納形は資産ヘッダが正 —
  `attentionCompute` も同値）②quant 席名の規則化（ADR
  [0074](decisions/0074-quant-seat-naming.md) — 格納語彙を `f32/f16/i8/i4` の 1 本へ・
  attention の活性は `attn8`・移行表は ADR）③プリセットの表示名 / 説明欄と `karume/4`
  繰り上げ（ADR [0075](decisions/0075-quant-presentation.md) — quant の allowlist が厳格なので
  optional 追加でも旧クライアントは読めない）④下の yomi 依存分離。**format 断絶は 1 回だけ** —
  ②③は同じ波でしか出さない。
- **SBV2 の yomi 依存分離（ユーザー方針 2026-08-20・breaking・時期未定 — 早い段階で
  実施したい意向）**: SBV2 パイプラインの入力を「yomi の解析結果だけ受ける」形へ再設計し、
  `@hdae/yomi` を models の依存から外す。ADR 0072 の注入席は結合を素通しに留めてあり、
  その際は overlay 解決が呼び手側へ移る。
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
- **構築 AbortSignal の他ファミリ横展開（起票 2026-08-24）**: irodori も GB 級資産で anima と
  同じ「取得と組み立ての最中に中止したい」窓を持つ。移植は 3 点セット — `signal` を各
  `*PipelineOptions` へ移設 / 段境界の検査 / 中断時の `ownsGpu` 解放。0.5.0 候補。
- **estimate の `transientBytes` が恒等別名化を再現しない（起票 2026-08-24・レビュー R6V-2）**:
  `estimateSessionMemory` は全ノード出力へ `alloc` を通すが、実行計画側（`derivePlanSlots`）は
  reshape / 恒等 expand の出力を `alloc` せず元 slot へ `retain` する。中間ピークが過大に出る
  うえ、`UNACCOUNTED`（入っていない項目を全部書く MUST）にも該当項が無い。0.5.0 で設計検討
  （別名判定の源を estimator と共有するか、`UNACCOUNTED` に 1 項足すか）。
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
  Vᵀ+列量子化融合・2048px DiT attention メモリ工事・SBV2 NFC チップ・f32 anima 系列再生成
- by-design 制約群（rank≤4 / OOB NaN / 非有限入力 / 0 要素次元 / cancellation 粒度ほか）は
  limitations.md が正本 — 実需が出た項目だけここへ昇格させる
