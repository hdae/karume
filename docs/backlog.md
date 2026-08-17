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

## next — autoregressive-ready 基盤波

「Gemma 4 対応」ではなく **IR / loader / state 実行モデルを autoregressive-ready にする波**。
検収モデル = Gemma 4 E2B / MiniCPM5-1B（MiniMax-M3 は非目標で裁定済み 2026-08-16 — ADR 0001 へ
1 行はこの波で）。**実装前に ADR を先行**させる（順序も設計判断）。参照資料は収集済み
（[research/2026-08-17-autoregressive-references.md](research/2026-08-17-autoregressive-references.md)
— 掃引 8 + 深掘り 3 + 敵対検証 3・検収モデル config 一次確定込み。ADR の根拠節はここを
引用する）。**ADR 先行は完了**（2026-08-17）:
[0066](decisions/0066-generation-context-state-slots.md)（GenerationContext / state
スロット・R2/R3）・[0067](decisions/0067-autoregressive-attention-vocabulary.md)（attention
語彙 = G3 / states 形 / state_append）・
[0068](decisions/0068-decode-exit-multi-output.md)（multi-output + argmax / topk）・
[0069](decisions/0069-packed-w4-storage.md)（0019 reopen + i4 格納）・
[0070](decisions/0070-shard-loading-admission.md)（shard 2 相 + admission）— 全て
accepted（裁定 A〜C + Codex 6 巡収束）。**実装波の波割りは裁定済み**（2026-08-17・
案 X = 検収足場先行）: **A = GQA + MiniCPM5 検収足場（済 2026-08-17 — `b78b0c1` 実装・
`3f072cb` recipe・e2e 門）** → **B = 多出力 + argmax / topk（済 2026-08-17 —
`3a31544`/`9a795a7` 出力列化・`cbe093a` argmax・topk と docs は後続コミット。0 本席は
波 D へ・topk exporter 側〈getitem 結線〉は sampling 実需まで先送り）** →
C = GenerationContext +
states{} → D = states 形 attention + state_append（0 本席 = ir.ts / verify.py の outs
改訂と**shapes.py の本数検査前 `declared_outputs[0]` 参照の解消**も同時 — 第 2 巡レビュー
指摘）→ E = decode 台本 + greedy 検収 →
F = w4（Phase 0 sweep は A 完了後から並行可）→ G = shard + admission → H = Gemma 4 E2B
検収。付帯裁定: topk の exporter 側（多出力 aten の getitem 結線）は sampling 実需まで
先送り / 検収は固定 token id 列の parity（tokenizer・models パイプライン本格化は波外）。
**以下の番号項目は実装波の作業台帳として残る（設計の正本は各 ADR）**。前提の宣言として **R2（shape 不変条件）を最初の
ADR に含める**: 恒久不変条件は「静的物理格納・固定 rank・計画キャッシュの鍵は常に容量」まで —
「全論理形状がホスト既知」は恒久にせず、**有界論理 extent の席**（compact-prefix 軸 1 本・
DDS op は payload + extent の複数出力・extent は計画鍵に入れない・admission は容量課金）を
IR スキーマに予約する。実装は最初の実需モデルまで先送り・上限超えと動的 rank はホスト介在の
グラフ分割のまま（根拠 = [runtime-landscape §3](research/2026-08-16-runtime-landscape.md)・
2026-08-16 裁定）:

1. **KV state / GenerationContext**: KV を Session の普通の入出力にしない。寿命分離
   （Session = immutable weights / GenerationContext = 1 生成の mutable state）・
   fixed-capacity physical + logical `pastLength`・prefill と decode は別 execution shape・
   device loss 時の再開契約。`pastLength` は shape symbol にしない（PreparedPlan/backing が
   毎 token 再構築になる — ADR 0042 の key 契約）。**state の単位は「層 × 均一 KV」ではなく
   名前付き state スロット（per-slot shape・別名可）で定義する**（R3・2026-08-16 裁定 —
   検収モデルの Gemma 4 E2B が sliding 固定長 512 / keys-as-values / 20 層 cache 共有の
   3 種混在で、均一前提だと 5 ADR 全部の改訂になる）。
2. **packed weight storage + sharded loading**: int4 級の logical shape / physical payload 分離
   （1 要素 = 1 payload 要素の現契約を破る初の格納形）+ safetensors shard + 全量 ArrayBuffer
   保持の廃止（shard 単位 fetch → verify → upload → 解放）。ADR 0019 の「w4 再測しない」は
   **正面から supersede する reopen ADR が先**（旧測定の適用範囲を明示）→ その後 Python
   fake-quant の Phase 0（format 候補 sweep — runtime を触らない）。
3. **メモリ予算 / admission**: resident weights + 展開分 + KV + prepared backing + transients +
   staging の estimator + 診断（絶対保証ではない）。
4. **decode 出口**: GPU `argmax`（greedy MVP）→ static-k `topk`。runtime の generic
   multi-output（topk / LSTM h_n / router — IR スキーマは既に複数出力有効・executor 側が未実装）。
5. **autoregressive attention**: causal / GQA / logical prefix length / KV state access /
   mask 表現 / empty-row 意味論。**row-block matcher の portability 依存はここで正面解決**
   （exact-match が唯一の 128MiB 級ポータビリティ経路である現状を op 側の席へ）。
   **G3 = kv_heads > 1 の GQA は波 A で解決済み**（2026-08-17 — ADR 0067 決定 1〜3 実装・
   r=1 バイト同一・repeat_kv parity・GQA×i8a8 は fail loudly。実モデル検収 =
   `e2e_minicpm5_test.ts`〈logits tolerance + greedy + census〉）。
   mask は causal / sliding の attrs 化を含めて裁定（131K context の `[1,1,M,N]` 実体化は
   68GB 級で物理的に不成立）。

付随: bool initializer / storage の設計（mask 素材 — ⑤と同席で裁定）・pipeline 単位の
Session 常駐と device-loss lifecycle（perf H-4 と同体）・sampling/RNG はホスト維持
（op-vocabulary の裁定を再確認済み — GPU 側は argmax/topk のみ）。

## later

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
