# Backlog — 波順と作業項目の正本

> プロジェクト全体の**優先順位・波順・未消化項目**の正本はこの 1 本。
> 状態語彙: `now`（現行波）/ `next`（次の大波）/ `later` / `release` / `parked`（復活条件つき）。
> 運用契約: ①完了した項目は**削除する**（履歴は git と ADR / research が持つ）②実測値・設計論証は
> ここに**書かない** — 出典（ADR / research / 台帳）を指す ③性能候補の起票・採否・kill 基準は
> [perf-ledger](perf-ledger.md) が正本で、ここは波として参照するだけ ④by-design 制約の正本は
> [limitations](limitations.md) — 作業化が裁定された時だけここに載る。

## now — 量子化方式の探索・第 2 段（波 J・2026-08-20 着手）

前段（波 I = w4 横展開 + 方式スクリーニング・聴感/視認込み）は**完全クローズ** — 実測の正本 =
[research/2026-08-19-w4-method-screening.md](research/2026-08-19-w4-method-screening.md)、
採否 = [perf-ledger](perf-ledger.md) の量子化方式節（Q-1〜Q-5）、recipe 基盤 4 件 = ADR
[0068](decisions/0068-decode-exit-multi-output.md) 追記 5 / [0069](decisions/0069-packed-w4-storage.md)
追記 5。

- **J-1: Q-1 実装 — 消化済み（2026-08-20）**: deberta-i4 混成系列（linear = i4 g32・残り i8）+
  SBV2 quant `w8-bert4` + 実 GPU WAV 門（perf-ledger Q-1 ✅・配布 WAV 聴感確認済み）。
  既定 quant はのちに w4 へ（J-2 第 3 段）。HF jvnv 上げ直しはリリース枠と同乗（資産は準備済み）。
- **J-1b: SBV2 full-w4 — 消化済み（2026-08-20 ユーザー依頼）**: net_g（front/voice）の i4
  混成系列 + 3 席とも i4 の quant `w4` + WAV 門（perf-ledger Q-5 の SBV2 席）。net_g の適格
  linear は 6 本のみで削減は w8-bert4 比 −0.08% — 意味は「配布形を丸ごと 4bit で通す席」。
  聴感 = 確認済み（2026-08-20 品質 OK・f32 比で微妙に硬い印象 — 第 3 段の gptq 結線後に再聴）。
- **J-2 第 1 段: 消化済み（2026-08-20）** — core `quant_calib.py`（`ce568b0`）+ minicpm5/EG
  リグ結線・校正コーパス 48 文 ×2（`730c21b`）+ 実測 5 構成 × 2 ファミリ。実測の正本 =
  [research/2026-08-20-gptq-awq-calibrated-rounding.md](research/2026-08-20-gptq-awq-calibrated-rounding.md)。
  要旨: **GPTQ 大勝ち**（gptq-rtn = 今日の格納形のまま RTN 全面超え = perf-ledger Q-6 起票 /
  gptq-kmeans = 全列最良・greedy 37/48 で Q-2 の席価値上昇）・**AWQ 不採用（Q-7 ❌）**。
- **J-2 第 2 段: 消化済み（2026-08-20）**: gptq 3 構成を SBV2 BERT（`3bf9cf8`）+
  irodori/anima DiT（`fc89d29`）へ結線し本番実測（gates 全緑・正本 =
  [research](research/2026-08-20-gptq-awq-calibrated-rounding.md) §6）。**DiT 2 ファミリは
  数値が大幅改善**（irodori: S 予測が全構成一致 / anima: PSNR 12.3→22.7 dB）・SBV2 BERT は
  数値判別不能（聴感のみ）。**聴感/視認裁定済み（research §6 裁定節）: 3 ファミリとも品質
  OK → 採用確定**。net_g conv は GPTQ 対象外（H が linear の in 軸形 — conv は im2col 要）。
- **J-2 第 3 段: Q-6 出荷結線 + 速度実測 — 結線消化（2026-08-20・`a1cf286` + `1379c3e`）**:
  deberta-i4 export へ gptq-rtn を結線（encoder linear 132 本 = 校正付き・語彙表は先に RTN・
  一致門 3 種・校正コーパスは deberta へ移管し計測と共有）→ fn/jvnv 再配布 + WAV 参照
  採り直し（w8 門不変・verify 1629/0/5）。速度実測済み
  （[research §7](research/2026-08-20-gptq-awq-calibrated-rounding.md) — 取得 −30%・ロード
  1.7 倍速・温間合成 ~4% 速 = 基準の速度側は充足）。**再聴裁定済み（2026-08-20 ユーザー
  「ほぼ違いが分からない」）→ 既定 quant = w4 へ変更**（ADR 0039 決定 5 更新・fn/jvnv
  再配布・既定解決の実証 = quant 未指定合成が w4 参照 sha と一致・w8 は opt-in 参照系）
  — **第 3 段クローズ**（既定はのちに波 K で `w8-bert4` へ再裁定 — ADR 0039 決定 5 の
  再裁定 blockquote）。irodori / anima の配布 i4 席新設は J-4 と同時に裁定（irodori は
  kmeans 圧勝のため rtn 席を先行させない）。
- **J-5a: embedding i4 — 消化済み（2026-08-20・`0d8c6f6`）**: i4 適格を
  `I4_WEIGHT_OPS = {linear, embedding}` へ一般化（ADR 0069 追記 6）。BERT 語彙表 i4 で系列
  −8.15 MiB（group scale の f32 が半減益の約 3 割を食い、見込み −12.5 MiB から下方訂正）・
  `w8-bert4` 取得量 = w8 比 **−30.33%**。WAV 参照 2 本採り直し済み（聴感確認済み 2026-08-20 —
  品質 OK・硬さの印象は第 3 段で再聴）。
  tied lm_head は適格へ反転（LLM 側の副次利得）。
- **J-5b: net_g conv1d i4 — 消化済み（2026-08-20・`eac9d43` + 参照採り直し）**: conv1d
  （groups == 1）を i4 適格へ追補（ADR 0069 **追記 7** — scale を rank 非依存の rank2 形へ
  一般化・gemm A 側タイルローダに group scale・emit/plan 鏡像は同型で固定・convT / depthwise
  は対象外）。SBV2 全 14 モデル再 export + 再配布で **w4 = 237.5MB（w8 比 −36.3%）**・温間合成
  ~500ms で最速（実測は ADR 追記 7 / research 2026-08-20 §7 系）。見込み ≈−40% は scale 増分と
  groups>1 残留の分だけ下振れ（正味 −22.2MB）。front の `SdpReverseNoiseIn` は「所有 = 使用」へ
  再構成（census と graph の構造的一致 — `--verify front` 緑）。w4 WAV 参照は三度目の採り直し
  （w8 / w8-bert4 門は不変）。**聴感済み（2026-08-20 ユーザー）: conv i4 追補で変化なし =
  品質門通過・J-5b クローズ**。f32 比の全体評 = `w8-bert4` はほぼ同一・`w4` はテンション
  少し低め（BERT 側 gptq はほぼ透明で、差は net_g の素の RTN i4 由来 — 想定内。conv は
  GPTQ 不適〈H が linear の in 軸形〉なので校正で縮める経路が今は無い）。
- **J-3: g 軸の評価**（波 I は g=32 固定の裁定 — 方式勝者で g32/g64/g128 を再評価）。評価軸に
  「`w4` の f32 比〈テンション少し低め〉（J-5b 聴感）が g を細かくして縮むか」を足す —
  net_g は校正が使えないぶん g 軸が唯一の品質ノブ。
- **J-4: 格納席の実装裁定**（Q-2 kmeans companion 席 / Q-3 NF4 定数表席 — J-2 第 2 段の
  結果を見てから。codebook 系採用なら ADR 0069 の bit 表・整列表・view 型 3 面の reopen）。
- **J-4a: anima の i4 席（J-4 から切り離して先行 — 2026-08-21 ユーザー裁定）**: 第 1 段
  （速度実測）消化。素の RTN で i4 系列 + `w4` / `w4-a8-s16` を新設し実 GPU で実測 — 正本 =
  [research/2026-08-21-anima-i4-seat-speed.md](research/2026-08-21-anima-i4-seat-speed.md)。
  取得量 −21.2%（3.48 → 2.74GB）・VRAM −22.6%（3,407 → 2,637MiB）。**同日中に①〜④まで消化**:
  ①GPTQ 結線（`ae9fe31` — 校正コーパス 4 本を評価入力と分離）②w4a8 = i4 常駐を整数内積へ
  （`c285f97` / ADR [0076](decisions/0076-w4a8-linear-execution.md) — 955ms/step まで戻るが
  **視認で細部が荒れるため anima の席には載せない**）③視認裁定で **`w4-a8-s16`（GPTQ・活性
  f32）を低 VRAM 席として採用**（品質重視 = `f16` / 速度重視 = 既定 `w8a8-s16` 据え置き）。
  ④**校正入力の捕捉順序の修正**（`78ebe68` — グラフ唯一の i8 重み `patch_embed.proj` を
  捕捉より前へ）で採り直し。golden 突合の f32 相対 RMS が 0.1857 → 0.1712（512²）/
  0.1423 → 0.1382（1024²）へ改善し、5 ケースの視認裁定で採用（research §9・格納形は不変）。
  **残 = ⑤公開の可否だけ**（HF 追加 → pin 更新 → JSR bump の連鎖。**保留** — 0.5.0 の
  breaking 波と同乗させる手もある）。**未検証で残した可能性の一覧は
  [research §8](research/2026-08-21-anima-i4-seat-speed.md)**（専用幾何・g16・校正量・
  もう 1 つの劣化機序 — いずれも「試してダメ」ではなく「試していない」）。irodori の席は
  J-4 のまま（kmeans 圧勝で別設計）。

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

- **生成 API 波（起票 2026-08-19 — 全体レビューの Codex 提案を採用裁定）**: 静的配線と
  リクエストを分離した `GenerationProgram`（setup 時に全結線を検証）+ stateful sequence API
  （`for await` の token イベント・EOS 停止・cancel・多ターン継続）+ `last_row` の runner 側
  導出（ADR 0068 追記 4 の所有関係のみ reopen）。`generateGreedy` は parity 検収用の内部
  ヘルパへ格下げ。LLM 実需（streaming / チャット）に直結する最大の API 波。
- **0.5.0 breaking 波（起票 2026-08-21・ユーザー裁定）**: 破壊的変更を 1 回にまとめ、公開 3 リポ
  の再アップロードと pin 更新（ADR 0073）を 1 度で済ませる束。①`linearCompute` の値を
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
  イベント（step ループが無く提供できるのは段遷移のみ）と、AbortSignal による中断席
  （現状は onEvent の throw が step 粒度の中断手段 — 席は温存）。
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
- hub Range 並列 + prefetch — 復活 = 断片化リポの再来（perf L-3）
- params / bind group キャッシュ（ADR 0032 案 2）・GPU timestamp 推定源化・全面 f16（案γ）・
  Vᵀ+列量子化融合・2048px DiT attention メモリ工事・SBV2 NFC チップ・f32 anima 系列再生成
- by-design 制約群（rank≤4 / OOB NaN / 非有限入力 / 0 要素次元 / cancellation 粒度ほか）は
  limitations.md が正本 — 実需が出た項目だけここへ昇格させる
