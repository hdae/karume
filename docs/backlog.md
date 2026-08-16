# Backlog — 波順と作業項目の正本

> プロジェクト全体の**優先順位・波順・未消化項目**の正本はこの 1 本。
> 状態語彙: `now`（現行波）/ `next`（次の大波）/ `later` / `release` / `parked`（復活条件つき）。
> 運用契約: ①完了した項目は**削除する**（履歴は git と ADR / research が持つ）②実測値・設計論証は
> ここに**書かない** — 出典（ADR / research / 台帳）を指す ③性能候補の起票・採否・kill 基準は
> [perf-ledger](perf-ledger.md) が正本で、ここは波として参照するだけ ④by-design 制約の正本は
> [limitations](limitations.md) — 作業化が裁定された時だけここに載る。

## now — 整理整頓波（2026-08-14〜）

外部レビュー 6 本の TRIAGE・docs 事実修正・ADR 注記・planning SoT 再編（本ファイル新設）・
**exporter 構造再編（案 A・ADR 0065 — 全 8 段）**は消化済み。残り:

- **exporter 再編の後始末（小粒）**: Python CI の 2 回目観測（初回 2026-08-16 は
  setup-uv タグと golden io のマシン依存を検出・修正済み — 期待値: core 2005 passed /
  30 skipped・recipes 985 passed / 233 skipped）・pytorch-cpu index の explicit 化
  （hatchling 1.27+ / PEP 639 の license-files リスト形が解禁される — 現状は既定グロブで
  同結果のため急がない）。
- **housekeeping 一括（小粒）**:
  - series 資産の再生成（birefnet / depth / vowel — e2e 門 SKIP の解消）
  - テスト逆方向 import の残り 3 本（e2e_birefnet / e2e_depth_anything / e2e_vowel_detector）
  - Anima 参照フィクスチャ系テストの復元・tokenizer parity fixture の models 側への移設
  - DeBERTa 実重み e2e の移植（現行の回帰網は WAV sha256 門 + rel-pos parity のみ）
  - turbo LoRA の置き場移行（配布形の親に入力素材が混在）
  - perf-ledger K-1 の曖昧 bucket を分割 or 再定義
  - dist の (rel_path, size) 前置フィルタ（perf L-5 の残作業）
  - dist.py の到達しない防御 2 箇所の撤去・1 回目 rename の失敗も DistError 化
  - u32 uniform 検査の一本化・第 3 便（第 2 便 2026-08-16 の実測で発見した残存サイト:
    conv-transpose1d / layer-norm / softmax / rms-norm / quantize-rows / attention /
    gemm-geometry / codegen の reduce・strided — いずれも上限検査なしの旧形。あわせて
    pad / flip 等の**導出値**（`rows*lengthOut` などの積）が検査対象に入っているかの点検）
  - transport.ts の到達不能 `outer === null` 分岐の掃除
  - hack マーカー掃引（rg TODO/FIXME/HACK/workaround）+ knip（未使用 export / 依存）
  - モデルカード親切化の横展開・P 音素門 / text parity 恒真の解消・exporter ops.py の整理

## next — autoregressive-ready 基盤波

「Gemma 4 対応」ではなく **IR / loader / state 実行モデルを autoregressive-ready にする波**。
検収モデル = Gemma 4 E2B / MiniCPM5-1B。**実装前に ADR を先行**させる（順序も設計判断）:

1. **KV state / GenerationContext**: KV を Session の普通の入出力にしない。寿命分離
   （Session = immutable weights / GenerationContext = 1 生成の mutable state）・
   fixed-capacity physical + logical `pastLength`・prefill と decode は別 execution shape・
   device loss 時の再開契約。`pastLength` は shape symbol にしない（PreparedPlan/backing が
   毎 token 再構築になる — ADR 0042 の key 契約）。
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
- Metal 数値差の原因確定（known-issues）・resident 経路の診断/計測制約の解消。
- MoE の seam（fixed-k routing は静的形で表現可 — dense API に expert 非存在を焼かない）。

## release — リリース準備波（しばらく先）

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
