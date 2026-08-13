# ACTIVE_DESIGN — Karume

> Short index of _current_ design focus. Keep it to a screenful. Reviewers and planners read this
> FIRST (alongside `CLAUDE.md` / `docs/`) so they don't start cold or misread an intentional
> migration as a defect. Update it whenever the current design context shifts.
>
> Last updated: 2026-08-13

## Active redesigns (in flight)

- **性能波（2026-08-13・設計変更不要枠）— 消化完了**: 第 1 段の内訳再実測
  （[research/2026-08-13-op-timing-restats.md](../docs/research/2026-08-13-op-timing-restats.md)
  — **K-8 は実測で棄却**・K-10〈dacvae convT = irodori 全 GPU 23.2%〉起票）→ H-6
  （`c950e76`・tiled VAE clone 層削除・PNG 門 4 本 sha 不変）→ L-6/L-4/L-5
  （`e9880cc`/`de65a71`/`a9561c6`・exporter streaming 化と dist 配置前 plan — 出力バイト
  不変を A/B + golden 27 本で証明・pytest 2625/3skip）→ **K-10 も着地（`d08dd8a`・
  convT residue grouping = 1,707 → 190ms〈9.0 倍〉・irodori 全 GPU −21%・壁 ×1.11・
  ビット同一 = WAV sha256 同一 digest + verify 1045/0）**。**以降の波順は 2026-08-13
  裁定で確定 — 正本は [perf-ledger](../docs/perf-ledger.md) 実施順**（波① conv
  ビット一致〈K-4a + K-7 実測相乗り〉→ 波② ホスト境界〈H-2→H-5 設計先出し→H-5+H-1〉→
  波③ numerics opt-in ADR〈案 a・一括フラグは将来課題〉+ K-5/K-2 → ⑤小粒 → ⑥モデル
  拡充 → ④リリース準備）。**ビット同一門は「指標であって目的ではない」**（ユーザー方針
  同日 — 数値を変える最適化は opt-in 席・既定 = 参照経路で sha 門凍結。**非保証部分の
  品質は人間レビューで管理** — 数値が大幅悪化しても生成結果は崩壊しないことが多いため。
  詳細は perf-ledger ヘッダ）。
- **網羅レビュー（2026-08-12・モード B・HEAD `faef828`）— レビュー完了・修正波進行中**:
  掃引 24 + 敵対検証 13 レッグ。確定 E1 / W9 群は全て推奨案で裁定済み — 正本は
  `.claude/reviews/2026-08-12_faef828/SUMMARY.md`（裁定記録込み）。性能候補の判断台帳は
  [docs/perf-ledger.md](../docs/perf-ledger.md) へ昇格（**優先順位の正本はそちらの 1 本** —
  ここには書かない）。設計評価の結論: 汎用ランタイム形は合格・LLM は executor 実行モデル
  1 層のみ変更（ADR 0043 追記 = 層軸の一本化済み）・モデル独立化はユーザー方針確定
  （家族固有変種は所属のまま・単体提供は公式重み由来の別物を後日）・exporter io 共通化は
  Gemma 直前（D3 案 b）。**並行トラック: Irodori 量子化 3 波（裁定済み — ADR
  [0050](../docs/decisions/0050-irodori-quant-series.md)）。波 1 = f16 完了（2026-08-12）**:
  配布 3.44GB → f16 系列 +1.72GB（50.1%）・レイアウトは `model.{f32,f16}.safetensors` へ
  破壊的変更・latent 門の系列パラメタ化（F16_Z_ATOL 1.5e-3 = 実測最悪 2.1321e-4 の 7 倍・
  **S/forwards は f32 と完全一致**）・WAV sha256 門は f32 専用のまま digest 不変・
  系列×格納 dtype は両側検査（`assert_storage_absent` 新設 — 存在検査だけでは f32 席への
  f16 混入が素通りする穴を波 1 で発見）。**波 2 + 波 3 も完了 = 量子化 3 波クローズ
  （同日・ADR 0050 追記）**: quant 席 4 つ（f32 / f16 / w8 = 全 i8 / w8a8 = バイト共有 +
  linearCompute）・**既定 w8a8**（ユーザー聴感裁定 — DAC + ヘッドホン通しで劣化感なし。
  sim の数値 LSD 5.64 より聴感が正）。S ドリフトは全構成で不変（混成不要）。門 = latent 門
  w8 席（1e-2）+ `e2e_irodori_w8a8_test.ts`（判別帯 [0.1,6] + census 317/317/0）。品質台本 =
  `measure_quant_irodori.py`（S ドリフト表・直交分解 5 軸・活性シム素通り検出）。速度は
  wall ×1.12。**DiT の GPU 内訳再実測は消化済み（2026-08-13 —
  [research/2026-08-13-op-timing-restats.md](../docs/research/2026-08-13-op-timing-restats.md)**:
  dit 51.6% / codec-decoder 38.7%〈convT 4 本だけで 23.2%〉・K-8 は同実測で棄却・
  irodori はホスト律速側 ≈4 割）。
- **DeBERTa 配布サイズ削減 3 波（2026-08-11）— 波 1 着地**: 発端は SBV2 の ONNX 版
  （hidden[-3] のみ抽出）が縮小の材料になるかという問い。実測の正本は
  [research/2026-08-11-deberta-size-recon.md](../docs/research/2026-08-11-deberta-size-recon.md)
  （ONNX 版は 22 層だがファイルは 1.03% しか小さくない — onnxsim が +92.3MB 焼き込んで相殺。
  **参考にすべきは「22 層で足りる」事実だけ**）。**波 1 = 末尾 2 層カット完了（ADR
  [0045](../docs/decisions/0045-deberta-layer-trim.md)）**: 334,545,336 → 309,167,272 B
  （−7.59%・`w8` 取得量 −6.34%）で **WAV sha256 は不変のまま緑**（`a82f72e2…`）。
  **波 2 = 出力を 1 本に絞るまで完了**（同 ADR 決定 3/4・sha256 は再び不変）。ただし
  **速度の期待は外れた** — T=15 で −4.5% / T=512 で −7.7%（readback 1,380 → 60 KiB）で、
  SBV2 実用ではパイプライン全体の 0.3%。ADR 0026 の「出力を絞れば GPU 比がそのまま出る」は
  T=512 前提の話で実用 T では成立しない。組み立て門は「22 層 × 出力 1 本 ×
  `bertHiddenFromEnd` 1」の 3 点検査（層数と出力形と取り出し位置は別々の台本が持つので、
  片方だけ動くと**別の層の声**が沈黙で出る）。**波 3 = 相対位置の添字表をグラフ入力へ昇格も
  完了**（同 ADR 決定 5・−2,098,128 B・速度は中立・sha256 は 3 波とも不変）: transformers の
  `disentangled_attention_bias` を差し替え（`karume/patch_deberta.py`）+ ホスト生成
  （`text/rel-pos-tables.ts`）+ golden とのバイト一致パリティ門。**3 波累計で
  334,545,336 → 307,068,768 B（−8.21%）・`w8` 取得量 −6.86%**。**HF は `karume-sbv2-jvnv` の
  上げ直しが未了**（fn は HF 非公開 — 2026-08-08 裁定）。
- **モデル拡充波 — EmbeddingGemma 動作確認済み（2026-08-11）**: gelu_tanh op（第 2 層 —
  **op 追加の判定手順は ADR [0043](../docs/decisions/0043-op-addition-layers.md)**）+
  exporter 対応（FOLDABLE に bitwise_or/gt・RoPE 降格の接尾一致）+ 台本
  `export_embeddinggemma.py`（Tmax=512・実行時 attention_mask 非対応 → limitations）+
  実 GPU golden e2e 門（5 ケース T=12〜318・maxAbs ≤ 3.9e-7・atol 1e-6）。公式 gated 重みと
  unsloth ミラーは sha256 全一致。**未着手**: models パイプライン・tokenizer（Gemma SPM
  BPE + byte_fallback — 既存 3 実装と不一致で新規実装）・配布形。**ORT 比較 + フロア最適化
  第 1 波は完了（2026-08-11）**: 帰属は skinny-M occupancy（linear が GPU 87%）で、
  M≤64 幾何バケット（ADR 0022 追記）+ rope 融合の Gemma 形一般化（ADR 0040 追記）により
  bare 76.4 → 52.7ms・対 ORT Web WebGPU 2.5 倍差・残りはホスト支配 —
  [research/2026-08-11-skinny-m-geometry.md](../docs/research/2026-08-11-skinny-m-geometry.md)。
  **B1 = 融合 attention の mask 対応も完了（同日・ADR 0023 追記）**: attention が arity 3..4
  になり EG は SDPA 保存で 1,273 ノード / 838 dispatch。ただし **wall は中立** — dispatch
  削減はホスト固定費 ~38ms に効かないという負の実測を獲得（固定費の分解が open）。
  **中 M バケット（65〜512 → M64N32・`a81cc08`）も採用済み**（Anima/SBV2 ABBA 中立・
  long-document 79.2 → 63.8ms）。**最終値と ORT 比較 = ORT 比較 doc §6**（bare 53.8ms・
  対 ORT Web 2.5〜2.9 倍・数値忠実度は 3〜4 桁優位のまま）。**残る karume 側の桁 =
  ホスト固定費 ~38ms の分解**（Deno WebGPU の per-call 費用・同期 — Chrome/Dawn では
  ORT が ~20ms で回る事実が上限の存在証明）。batch>1 export は変換段でブロック
  （known-issues）。次 = Irodori-TTS v4 移植波（**着手済み — 次の bullet**）。
  モデル候補キュー: Irodori-TTS v4 →
  BiRefNet_HR（torchvision deform_conv2d が blocker・grid_sample 系の ADR 前提）→
  Gemma 4 E2B（2026-04 実在・Apache 2.0・ungated。新規性は decode + KV cache の実行モデル
  設計で tokenizer / gelu_tanh は共用）。recon 詳細 =
  [research/2026-08-11-model-expansion-recon.md](../docs/research/2026-08-11-model-expansion-recon.md)。
- **Irodori-TTS v4 移植 — 第 1 波（基盤）完了（2026-08-11）**: ①`sin` op（第 1 層・
  Snake 活性が根拠・cos は不採用）②DACVAE 重みの safetensors 1:1 変換台本
  （`convert_dacvae.py`・317 本バイト一致門・wm_model 9.3M の存在を確認）③**実行時
  attention マスクの設計 = ADR [0044](../docs/decisions/0044-runtime-attention-mask.md)
  （accepted・実装は次波）** — bool 入力 + `safe_softmax`（第 2 層・ガード証明不能時のみ）で
  既存資産バイト不変 ④ModernBERT テキスト系 3 グラフ（backbone / text-proj / caption-proj・
  Tmax 512 統一・静的マスク方式 — 同値は台本の常設門が毎 emit 実測）+ 実重み E2E 門 19 件
  （`e2e_irodori_test.ts` — tolerance 導出表はテスト定数の docstring）。recon の U2 / U5 は
  解消。**第 2 波は W2-A/B/C まで完了（2026-08-11）**: W2-A = `safe_softmax` op + ガード
  証明の 2 段化（ADR 0044 実装 — 分解ガード相当との Uint32 parity・既存資産バイト不変・
  全 -inf 静的マスクは fail loudly から書き換え受理へ〈ADR 0044 追記〉）。W2-B/C = speaker /
  duration ターゲット（E2E 29 件・参照なし = ホストゼロ供給 + 常設実測門・**recon 訂正 2 件**:
  duration は token-sum 形で系列入力必須 / RoPE 実数化のビット一致は形依存〈head_dim 64 で
  実測 0〉）。**W2-D も完了 = 第 2 波クローズ（2026-08-11）**: U3 は recon（research/
  2026-08-11-dit-export-recon.md）→ 裁定 → ADR [0046](../docs/decisions/0046-cat-symbolic-axis.md)
  （cat 連結軸を同一シンボル一次和へ緩和 — 宣言ガード 3 本のみ・実 GPU parity 済み）+
  [0047](../docs/decisions/0047-irodori-dit-execution.md)（B=1×選択実行・G4 は G5 へ畳む・
  uncond = マスク還元）で解消。`dit` ターゲット着地（E2E 36 件 — **uncond 3 変種の
  マスク還元が 12 層実重みでビット一致** = 決定 1 の実証・S=750 の scores 136MB を毎回実走・
  torch Dim 名は sympy S 衝突回避で L / IR 名は S）。norm の所在: `text_norm` は
  duration / dit の消費側内包・`speaker_norm` は speaker ターゲット側・**`caption_norm` は
  dit 内包 + caption-proj の第 2 出力**（duration の `caption_vec` をホストが採るため —
  ADR 0048 決定 1）。**第 3 波ホストも完了 = latent までの全経路が TS で閉じた
  （2026-08-12・ADR [0048](../docs/decisions/0048-irodori-host-port.md)）**:
  ①irodori/text 家族（normalize 5 段 + Unigram + byte_fallback — Viterbi は `src/text/` の
  家族中立共有層へ抽出・パリティ fixture は git 追跡・NFKC は全 cp 掃引の両方向門）
  ②IrodoriPipeline（fromAssets 無 Session・dit のみ 1 Session 40〜100 forward・pipelineConfig
  20 欄が数の正本で対応外モードはパース時拒否・seed 上流非互換は `initialNoise` 注入口で解決）
  ③dist `karume-irodori-v4-small`（f32 3.07GB〈codec 前 6 グラフ時点 — 現行 census は
  3.44GB・ADR 0050〉・組み立て門はグラフ宣言と 12 点 + mask 派生
  次元 S+1519 突合）④E2E latent 門 4 本（S/forwards 完全一致・z は Z_ATOL 5e-3 = 素実測
  7.9e-4 の 6.3 倍・実効値 drift 門付き）。ホスト経路の recon 正本 =
  [research/2026-08-11-irodori-host-recon.md](../docs/research/2026-08-11-irodori-host-recon.md)。
  **第 4 波 codec も完了 = テキスト →（参照 wav →）WAV の全経路がクローズ（2026-08-12・
  ADR [0049](../docs/decisions/0049-irodori-codec-integration.md)・試聴確認済み）**:
  ①G6/G7 export（ランタイム無変更 — convT 4 本は既存受理形・reciprocal は lifted 定数化で
  第 0 層消滅・wm バイパスは README ②形で透かし非付与）②decoder タイル分割（halo 8 捨てで
  **Uint32 ビット一致門** — 既定 128MiB 上限機でも S=750 が通る。encoder は非タイルのまま =
  limitations）③LUFS −16 正規化のホスト移植（BS.1770-4 完全・f64・parity 門 5 ケース）+
  `decodeWav`（/32768・書き ×32767 と非対称固定）④`generate()` = latent → z 上 trim →
  タイル decode → 秒切り出し（durationSeconds は上流綴りへ = ADR 0048 既知差分解消・
  pipelineConfig 23 欄）⑤E2E は **latent 門と WAV sha256 門の併存**（「置換」から改訂 —
  ADR 0049 決定 5。digest 2 本・4 プロセス再現）+ タイル同値 + 参照前処理 parity + speaker
  実 latent tolerance 確定（7e-4）。**`sin` の π 超え引数（〜22.7π）で誤差 5e-6 級の初観測**
  = GPU sin の引数簡約が f32 精度を保つ（e2e_dacvae_test の docstring が正本）。生成実測:
  full 12.5s / voice-clone 14.0s（参照 7.6s → 6.8〜7.4s の音声）。recon 正本 =
  [research/2026-08-12-dacvae-codec-recon.md](../docs/research/2026-08-12-dacvae-codec-recon.md)。
  **残（Irodori）**: HF 公開はリリース時（jvnv 上げ直しと同時）・encoder タイル化 /
  resample / WAVE_FORMAT_EXTENSIBLE は需要駆動（limitations 起票済み）。examples/irodori は
  追加済み（`deno task demo:irodori` — --ref で voice cloning・--caption で Voice Design）。
  **次のモデル候補キュー**: BiRefNet_HR（deform_conv2d が blocker）→ Gemma 4 E2B
  （decode + KV cache の実行モデル設計 — prepared 機構 / タスク #7 と接続）。
- **立ち上げロードマップ（ADR 0037）は P0〜P5 まで到達し一段落**。P3/P4 で `AnimaPipeline`
  （fromPretrained / fromAssets・`using` 対応）+ 共通 image 層 + 配布形（現 `models/karume-anima-turbo/`）
  （`karume dist`・実 hash・格納 dtype 門）+ `karume` サブコマンド CLI + **英語**モデルカード
  README 自動生成まで完了。**移植の門 = PNG sha256 ビット一致 ×4 が全緑**（ローカル /
  取得層 + integrity / example CLI）。P5 で HF 実網通しと配布形の公開まで済み、JSR publish の
  CI/Release ワークフローも設置済み（リポ直下 README は release 準備時に書き上げる WIP 表記）。
  **残る宿題**: ①波 1 積み残しの参照フィクスチャ系テスト（timestepsProj atol 突合等 —
  anima-pipeline 系列の再エミットが前提）②tokenizer parity fixture の models 側への移設。
- **models/sbv2 — 常駐**: SBV2（テキスト → 音声）は `packages/models/src/sbv2/` に移植済みで、
  サブパス面 `@karume/models/sbv2` は `Sbv2Pipeline` だけを出す（example 111 行）。実重み e2e は
  `packages/runtime/tests/e2e_sbv2_test.ts`（3 系列 × 5 ターゲット × 5 ケース）、GPU 不要の
  ホスト側は `packages/models/tests/sbv2_{relattn_parity,text,host,pipeline,style}_test.ts`。
  **WAV sha256 門 = `packages/models/tests/e2e_sbv2_wav_test.ts`**（FN4/w8・参照 = ADR 0039 の
  実測 digest・tolerance 化と参照差し替え禁止 — PNG 門の音声版）。配布形
  `models/karume-sbv2-fn/`（FN1〜FN10）と CLI `karume export-sbv2` まで揃っている（manifest の
  確定は ADR 0039 → v2 形は ADR 0041）。
- **参照実装ブランチからの再実装（C 波）— 完了**: `codex/kernel-quick-fixes` の triage を
  **設計から書き直して**取り込んだ。C1 = i8a8 linear の accumulator 静的展開 / PipelineCache の
  未決着共有 / exporter の隣接 permute 合成、C2 = **融合パス新設**（RoPE・SiLU・upsample2x を
  1 dispatch へ + 恒等 expand の別名化 — ADR [0040](../docs/decisions/0040-fusion-pass.md)）、
  C3 = f32/f16 linear の accumulator 静的展開。**w8a8-1024 16.1 → 13.9s（1.16×）/ f16-1024
  38.5〜38.8 → 30.1〜30.3s（1.28×）**・PNG 門 4 本は sha256 不変。採否と実測の正本は
  [research/2026-08-08-branch-adoption-perf.md](../docs/research/2026-08-08-branch-adoption-perf.md)、
  参照側の記録は [research/2026-08-06-kernel-triage/](../docs/research/2026-08-06-kernel-triage/)。
  **不採用**（実測根拠あり）: contiguous elementwise / QUANT-010 / i8a8 タイル幾何の一般化。
- **F 波（VRAM OOM の誤報告）— 完了**: 確保失敗が派生 validation に化けて「破棄後使用」に
  見えていた件を根治（errorScope の報告順を根因優先へ + Session 構築での staging 解放）。
  機序は [research/2026-08-08-vram-oom-misreport.md](../docs/research/2026-08-08-vram-oom-misreport.md)。
- **配布の次手（保留中）**: DL 低速の正体は **Xet 再構成の断片化**と判明し、公開 2 リポは
  修復済み（dedup 抑止の上げ直し — 公開時の env 4 つは
  [assets-layout.md](../docs/assets-layout.md) の「公開」節が MUST）。hub 側 Range 並列 +
  prefetch 追随波は**保留**（断片化オブジェクトでは並列 16 まで伸びる — 「4 で飽和」は健全物
  限定。設計材料は
  [research/2026-08-09-xet-fragmentation.md](../docs/research/2026-08-09-xet-fragmentation.md)）。
- **統計波（op 別 GPU 時間内訳）— 完了（2026-08-10）**: 実測の正本は
  [research/2026-08-10-op-timing-stats.md](../docs/research/2026-08-10-op-timing-stats.md)。
  台帳 4 件は合計しても壁時計の 7% 台（OP-008 ≈ −1.2% / PLAN-012 ≈ −0.8% / HOST-006 上限
  −5.1% / **PLAN-011 は既定 guidance 1 で利得ゼロ**）。**本命は DiT linear + attention と VAE conv2d の
  カーネル最適化**（63.3% / 19.1% は**壁 13.9s 時点の按分** — 10.5s 時点の内訳は同 doc §9。
  再測してから候補選定 = [perf-ledger](../docs/perf-ledger.md) K-1）。SBV2 は逆にホスト律速
  （壁 1.08s vs GPU 0.42s）。
- **実行時最適化 3 波 — 完了（2026-08-10）**: ①attention i8a8 の accumulator 静的展開
  （`3f417dc`）②adaLN 融合 = 融合パスへ**窓内 passthrough** を導入し 4 ノード → 1 dispatch
  （`fbae6d2`・ADR 0040 追記）③i8a8 GEMM 族の**タイル幾何パラメタ化 + 実測最良の既定**
  （`7b55de5`・`i8a8-geometry.ts`・stats regcache 込み）。**w8a8-1024 壁 13.9 → 11.79s
  （×1.18）**・f16 経路は無変更（A/B ×0.998）・全門 sha256 不変。実測の正本 =
  [research/2026-08-10-kernel-variant-sweep.md](../docs/research/2026-08-10-kernel-variant-sweep.md)。
- **f32/f16 幾何波 — 完了（2026-08-10・v0.2.2 後にタイル軸掃引まで着地）**: f32/f16 GEMM
  骨格のタイル幾何をパラメタ化し、タイル辺も幾何へ吸収（`src/kernels/gemm-geometry.ts`・
  tileM/N は regM·wgY / regN·wgX の導出・全 op の accumulator 静的展開・キーに幾何判別子）。
  掃引の結果**既定は M128N128 r8×8 wg16×16**（conv2d の m タイル 64/32 は維持）。
  **f16-1024 朝 30.5 → 20.2s（累計 ×1.51）・w8a8-1024 10.6s**・全門 sha256 不変。
  ビット同一は実測命題（fma 留保は Vulkan で外れた）— **f32 では実行時オートチューン
  不可・既定変更は門の再実測とセット**（ADR 0022 追記が MUST）。実測の正本 =
  [research/2026-08-10-f32-geometry-probe.md](../docs/research/2026-08-10-f32-geometry-probe.md)。
  Metal（M2）追試済み: 退行なし（f16 ×1.06）・幾何 2 種の sha 一致 = ビット同一は
  Vulkan/Metal の 2 環境で成立。既定はバックエンド共通 1 本（NVIDIA 第一 — ユーザー裁定）。
  sha 参照門は参照環境専用の規約を limitations.md に明文化。
  m 小 linear は census 上対象外（GFLOP 0.003%）。AMD / Intel は未実測（検証は自己 A/B —
  limitations 参照）。
- **HOST-006 第 1 波 — 完了（2026-08-10）**: params の内容アドレスキャッシュ（Session 常駐・
  一度書いたら不変・診断 `lastRunParams`）+ bind group layout の PipelineCache 保持。
  **w8a8-1024 10.6 → 10.3s（×1.03）・f16-1024 20.2 → 20.0s**・門緑。params 系の約半分は
  GPU と重畳しており壁に出たのは露出分のみ。残余ホスト ≈1.1s ≈ 11%（実測の正本 =
  [research/2026-08-10-op-timing-stats.md](../docs/research/2026-08-10-op-timing-stats.md) §7）。
  PLAN-012 は見送り確定（−0.8% に VRAM +224MiB は不釣り合い — 量子化形は次波 E で復活）。
- **PreparedExecutionPlan 波 1+2 — 完了（2026-08-10・設計の正本 = ADR
  [0042](../docs/decisions/0042-prepared-execution-plan.md)）**: エンコード層を導出/実行の
  2 相へ分離（`32dad19`）→ 導出済み計画を bindings キーで Session 常駐（`b2e6ce0`・LRU 4・
  `lastRunPrepared`）→ transient slot の GPU backing（`339fc0c`・容量 1・初ヒット遅延構築・
  footprint 一致門）→ bind group 焼き込み + 入力固定（`1751f3c`・`planBacking` 診断）。
  **段別 ABBA の帰結: 壁利得は段 C のみ（w8a8 10.4-10.5 → 10.1s ×1.03-1.04・f16 20.0 →
  19.6s ×1.02）**。波 1 と段 D は中立 = 導出相も createBindGroup も GPU と完全重畳
  （op-timing-stats §8/§9 — 見積り訂正 2 回）。**staged execution（E / timestep stage）は
  前提消失で見送り推奨**（狙っていたホスト費用が重畳側・GPU 利得 E 0.60% / timestep 0.12% —
  ユーザー裁定待ち）。backed run に残る run 毎費用は入力 writeBuffer + flush/readback +
  dispatch ループのみ。次の桁の裁定は [perf-ledger](../docs/perf-ledger.md) へ集約
  （カーネル比率 63.3% / 19.1% は 13.9s 時点の失効値 — 再測が先。Session 構築 gap 2.50s は
  L-1）。
- **manifest v2（ADR [0041](../docs/decisions/0041-manifest-v2.md)）— 実装完了（2026-08-09）**:
  1 リポ複数モデル（`defaultModel` 必須）+ 語彙整理（presets → `quants`・variant → `dtype`・
  components → `weights` / `assets` 分離）。**v1 パーサは持たない**。hub v2 パーサ +
  `resolveFiles(manifest, {model?, quant?})` + models 貫通（`preset` オプション廃止）+
  exporter の `--model` 軸 / ファミリー組み立て（同一相対 path + 同一 sha256 のみ `shared/`）
  まで実装済み。**配布形の配置はハードリンク禁止・常に独立コピー**（ADR 0041 追記）。
  ローカル配布形は **`models/karume-anima-turbo`**（モデル anima-turbo）と
  **`models/karume-sbv2-fn`**（FN1〜FN10 の 10 モデル・defaultModel = FN1・DeBERTa は
  shared/ に 1 本）へ再生成済み。**HF 公開済み（2026-08-09）**: `hdae/karume-anima-turbo`（新規・
  旧 anima-turbo はユーザーが後日削除）+ `hdae/karume-sbv2-jvnv`（**モデル ID = F1/F2/M1/M2**・
  cc-by-sa-4.0・帰属は exporter の `--card-profile jvnv` が機械生成・実網 fromPretrained 検証済み）。
  版は 0.2.0 ロックステップ済み（JSR publish は Release CI・ユーザー）。

## Pitfalls

- **Metal（Apple GPU）は WGSL の受け取り方が Vulkan と違う** — threadgroup の `vec4` へ
  **動的インデックスで成分を書く**（`sb[i][wsl] = v`）と `wsl != 0` が黙って捨てられる。
  GEMM の B タイル充填は静的成分への `switch` 展開で回避済み（`gemm.ts` の
  `storeBTransposed`）で、**同じ形を新しく書かないこと**。残る誤値（attention i8a8 /
  conv2d の 2 経路一致）は known-issues.md、**性能**（Linux の 31〜41 倍 — 帯域は健全なのに
  GEMM のタイリングが 1.21x しか効かない = Apple GPU 向け未最適化）は
  [research/2026-08-06](../docs/research/2026-08-06-metal-silent-miscompute.md) §3。
  Metal では `gpuTiming` が使えない（dispatch 数がサンプル上限を超える）。
- **融合 matcher は実測形への決め打ち**（RoPE の 2 形 + passthrough 1 本、attention の
  head 幅 D は slice 境界から導く一般形（正の偶数 — 実測は 128/256）、upsample2x の
  6 ノード列、adaLN の窓 6/7 など — exact 一致のみで、掴めなければ素のノード列へ
  fallback）。**エクスポータのノード発行順や形が変われば黙って外れ、値は正しいまま性能だけ
  落ちる**（例外も警告も出ない）。観測点は `Diagnostics.lastRunFusions` と、実配布グラフへの
  突合門 `packages/runtime/tests/assets_fusion_counts_test.ts`（資産のあるマシンでのみ実走 —
  性能が戻ったらまずここを見る）。
- **RoPE / SiLU 融合カーネルの丸め障壁（workgroup memory 往復）は WGSL 仕様の保証ではなく
  実測依存** — バックエンド更新やドライバ更新で PNG 門が割れたら、まずここを疑う。
  upsample2x は u32 ビット複製なので丸めの議論自体が無い。
- **`deno task verify` は `.claude/worktrees/` が存在する間、main 作業ツリーで素に走らない**
  （末尾の `deno test -A` がパス無しなので worktree 側のチェックアウトまで拾う）。当面は
  test 段だけ `deno test -A packages` で代替する。**恒久策は未裁定**。
- **Session 構築は重みアップロード後に submit を挟む**（`queue.writeBuffer` の staging は
  submit 完了まで解放されない）。この 1 回を消すと f16 preset の瞬間ピークが **+2.7GiB**
  （5,719 → 8,391MiB・確保天井 11,136〜11,264MiB に対し余裕が 2.7GiB まで縮む）。
- **`models/` に置くのは HF へそのまま上げられる配布形だけ**（1 ディレクトリ = 1 HF リポ）。
  エクスポータの系列出力は `outputs/series/`、実重みの**入力素材**は `inputs/<ファミリ>/<名前>/`
  — 綴りの正本は `karume/paths.py`。**ADR と docs/research 内の `models/anima-*` 表記は当時の
  記録**（時点スナップショットなので直さない）。**turbo LoRA だけ未移行**（`anima_pipeline.py`
  の `--lora` 例が配布形の親に入力素材を混ぜたまま）。
- **配布資産の格納形は series ディレクトリ名でなくヘッダが正** — dist.py の格納 dtype 門が
  組み立て時に検査する（`--dtype` 付け忘れの素 F32 が PNG 門まで沈黙した実測事故が根拠）。
  宣言外ファイル検査の例外は直下の `karume.json` / `README.md` だけ。
- **現行識別子（`karume_ir` / `karume-ir`）以前に焼かれた資産は開けない**（互換シム無し —
  fail loudly）。`models/` と `outputs/` はどちらも untracked。
- モデル e2e は anima の PNG 門 4 本が本リポに常駐（`models/karume-anima-turbo/` 資産が前提・
  無ければ明示 SKIP）。sbv2 の実重み e2e も復帰済み（系列 `outputs/series/sbv2-FN4{,-f16,-i8}/`
  が前提 — 系列名は改名の対象外）。
  **deberta の実重み e2e だけは移行元リポに残置のまま**（`e2e_deberta_test.ts` は未移植）。
- models パッケージの tree-shaking は「全モジュール副作用ゼロ」不変条件が前提 — 崩れると
  barrel 経由の shake が静かに死ぬ。JSR npm 互換層が `sideEffects: false` を出すかは**未検証**。
