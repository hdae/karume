# Backlog — 波順と作業項目の正本

> プロジェクト全体の**優先順位・波順・未消化項目**の正本はこの 1 本。
> 状態語彙: `now`（現行波）/ `next`（次の大波）/ `later` / `release` / `parked`（復活条件つき）。
> 運用契約: ①完了した項目は**削除する**（履歴は git と ADR / research が持つ）②実測値・設計論証は
> ここに**書かない** — 出典（ADR / research / 台帳）を指す ③性能候補の起票・採否・kill 基準は
> [perf-ledger](perf-ledger.md) が正本で、ここは波として参照するだけ ④by-design 制約の正本は
> [limitations](limitations.md) — 作業化が裁定された時だけここに載る。

## now — （空 — 次波の開始はユーザー裁定）

全体レビュー波は **0.3.0 の JSR / PyPI リリース（2026-08-16）まで含めて全消化**。
勢力図・ポジショニング検証は
[research/2026-08-16-runtime-landscape.md](research/2026-08-16-runtime-landscape.md)、
リリース時点の実測は
[research/2026-08-11-embeddinggemma-ort-comparison.md](research/2026-08-11-embeddinggemma-ort-comparison.md)
§7（EG bare 28.11ms — レビュー波の門追加は性能コストゼロを確認）。

**autoregressive-ready 基盤波（A〜H）も全消化（2026-08-17〜19）** — GQA・多出力 +
argmax / topk・GenerationContext・states 形 attention・decode 台本 + greedy 検収・w4
（i4 g32）・shard ロード + admission・Gemma 4 E2B / MiniCPM5-1B 実モデル検収・token-only
既定出口まで完了。設計と経緯の正本 = ADR
[0066](decisions/0066-generation-context-state-slots.md) /
[0067](decisions/0067-autoregressive-attention-vocabulary.md) /
[0068](decisions/0068-decode-exit-multi-output.md) /
[0069](decisions/0069-packed-w4-storage.md) /
[0070](decisions/0070-shard-loading-admission.md)（各追記）と research
（[autoregressive-references](research/2026-08-17-autoregressive-references.md) /
[w4-fake-quant-sweep](research/2026-08-18-w4-fake-quant-sweep.md) /
[shard-load-ram-peak](research/2026-08-19-shard-load-ram-peak.md)）。

autoregressive 波の**残項目（波外へ送り）**:

- **R1 と同席**: manifest v2 の shard 欄 + exporter 側 shard 分割規則（ADR 0070 追記 —
  HF 公開前締切）。
- **MiniCPM5 の token-only 系列**（ADR 0068 追記 4 の同形展開 — models 側 `lastRow` は
  共通化済みで recipe + 門の鏡像だけ。topk の exporter 側〈多出力 aten の getitem 結線〉は
  sampling 実需まで先送りのまま）。
- L8（fake-device 注入面）は保留継続・`enqueue` の generation 面は設けない裁定で確定
  （limitations）。
- 有界論理 extent の席（R2 — IR スキーマ予約のみ・実装は最初の実需モデルまで先送り）・
  bool initializer / storage の設計・pipeline 単位の Session 常駐と device-loss lifecycle
  （perf H-4 と同体）・sampling/RNG はホスト維持（GPU 側は argmax/topk のみ）。

## next — w4 横展開 → 量子化方式の探索（2026-08-19 ユーザー裁定）

先に **w4 の横展開（品質調査・recipe 基盤再編と同席）**、続いて**量子化方式の探索**へ
（両項の内容は later 節が正本 — 着手時にここへ昇格させて具体化する）。
**リリース準備波（release 節）はモデルの HF 公開も含み重いため後回し**（同日裁定）。

## later

- **生成 API 波（起票 2026-08-19 — 全体レビューの Codex 提案を採用裁定）**: 静的配線と
  リクエストを分離した `GenerationProgram`（setup 時に全結線を検証）+ stateful sequence API
  （`for await` の token イベント・EOS 停止・cancel・多ターン継続）+ `last_row` の runner 側
  導出（ADR 0068 追記 4 の所有関係のみ reopen）。`generateGreedy` は parity 検収用の内部
  ヘルパへ格下げ。LLM 実需（streaming / チャット）に直結する最大の API 波。
- **w4 の横展開 — 既存ファミリでの品質調査**（ユーザー要望 2026-08-19）: Anima / SBV2 /
  Irodori など既存モデルに w4（i4 g32）を当て、どこまで品質が落ちるかを測る。SBV2 w4 は
  聴感未実施のまま（配布縮小 3 波の再検討候補 — 棄却記録はモデル固有の裁定どおり対象自身で
  実測する）。Anima / Irodori / SBV2 には `measure_quant.py` の実測足場が既にある。
  **recipe 基盤の再編を同席させる裁定（2026-08-19）**: variant 駆動の単一 recipe 化・
  artifact transaction の exporter core 汎用化・PLE bank のモジュール化・系列間 golden の
  provenance 束縛 — 横展開で recipe 群を触る前に基盤を入れると手戻り最小。
- **量子化方式の探索**（ユーザー要望 2026-08-19 — 検討波の起票候補）: f8 / f4（浮動小数
  格納）・ダイナミックレンジ量子化・テーブル参照（LUT / codebook）・**層ごとのテーブル参照**。
  格納 dtype の受理集合（ADR 0069 の bit 表・整列表・view 型の 3 面）と検出器の設計から。
  実測骨子ができたら perf-ledger へ起票し直す。
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

- **R1: manifest v2 の shard（複数ファイル）/ 1 コンポーネント内混成 dtype の席の ADR**
  （ADR 0041 / 0063 reopen・2026-08-16 裁定で追加 OK）。**HF 公開前が締切** — hub は v2 のみを
  読み 2 形パースをしない（ADR 0041）ので、公開後に必要になると全リポ再アップになる。
  今なら席を空けるだけで既存 manifest は 1 要素として書ける。
  **同席裁定（2026-08-19 全体レビュー — Codex 提案の採用）**: ロード面の公開 API を固める
  この波で同時に設計する — ①shard identity（`{id, bytes}`）の hub↔runtime 境界保存
  ②`prepareModel(graphShard) → estimate → createSession(weightShards)` の 2 段境界
  （admission を重み DL 前へ — ADR 0070 の graph-first に沿う）③estimator のシナリオ別報告 +
  `peakAccountedBytes` 改名 ④重み常駐の判別 union 化（`ResidentWeight` — 3 並列 map の統合）。
- 実資産 CI gate（GitHub CI はローカル資産を踏まない問題）
- HF 公開一式: karume-sbv2-jvnv 上げ直し・新規 5 モデル・Irodori
- リポ直下 README の書き上げ・JSR npm 互換層の sideEffects 検証
- ライセンス interview（export-recipes の family 別 provenance を upstream revision 単位で
  人間確認 — 再編の release gate）
- 「semantic surface と実装済み subset の分離」方針の再裁定（attention / deform_conv2d /
  gather / conv_transpose1d / upsample_bilinear2d — 観測 subset を op 意味論にしない統一規約）
- 大型 DL 前の limits preflight（DL 後にしか limits 不足が分からない問題）

## parked（復活条件つき）

- **export-recipes の別リポジトリ分離**（ユーザー意向 2026-08-15 — 現時点では何もしない）。
  ADR 0065 案 B〈別配布物化〉のリポ版。切り出し時の論点 = `_shared/paths.py` の REPO_ROOT
  導出・runtime 適合 fixture（packages/runtime/tests/fixtures/）の共有・goldens の出力先・
  uv workspace の解体。復活 = ユーザー裁定。

- hub Range 並列 + prefetch — 復活 = 断片化リポの再来（perf L-3）
- params / bind group キャッシュ（ADR 0032 案 2）・GPU timestamp 推定源化・全面 f16（案γ）・
  Vᵀ+列量子化融合・2048px DiT attention メモリ工事・SBV2 NFC チップ・f32 anima 系列再生成
- by-design 制約群（rank≤4 / OOB NaN / 非有限入力 / 0 要素次元 / cancellation 粒度ほか）は
  limitations.md が正本 — 実需が出た項目だけここへ昇格させる
