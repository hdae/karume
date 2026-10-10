# backlog の now 節の消化済み記録の退避（0.12.0 公開〜0.13.0 リリース後の波・2026-09-06〜10-03）

> 2026-10-10 時点のスナップショット。[backlog](../backlog.md) の now 節（見出し「now — 0.13.0 リリース後（2026-09-25）」）に
> 残っていた完了済みの記録 6 項目を**逐語で**退避したもの（利用者裁定 2026-10-10: now の済んだ記録を research のアーカイブへ —
> 内容は消さず移すだけ）。設計と実測の正本は各項目が指す ADR / research / perf-ledger 側にあり、この文書は
> 「当時どうまとめられていたか」を引くための索引兼保存箱である。

読み方の注意:

- 本文は**退避当時の backlog の文面そのまま**で、その後の変更を反映していない。現況は ADR / research /
  [perf-ledger](../perf-ledger.md) / [known-issues](../known-issues.md) / [limitations](../limitations.md) /
  [CHANGELOG](../../CHANGELOG.md) が正本で、食い違ったらそちらが勝つ。
- 6 項目のうち 3 項目（高速化 / メモリの波・幾何プロファイル・モデル横断の追加調査）は、済んだ記録と残件が 1 項目に混ざっていた。
  残件は backlog の now 節に短い項目として抜き出して残し、その項目からここを指す。本文中の「残り」「未」は退避当時の状態である。
- コンテナ形式の波の段 4〜6 と、MTP（parked）の残件も backlog 側に残る（前者は now 節の 1 行・後者は parked 節が perf-ledger の K-20 行を指す）。
- 本文中の「now 節 / later 節 / release 節」「上の行」「その下」は**退避当時の backlog** を指す。
- 機械的に変えたのは 1 点だけ: 相対リンクの起点を `docs/` から `docs/research/` へ直した（71 本に `../` を前置）。
  文面は 1 文字も変えていない。項目の並びは退避当時の backlog の順。
- 作業履歴（どのコミットで何をしたか）は git が正本。本文が引くコミットハッシュは、当時の記述を追うための手掛かりとして残している。

## 項目ごとの要点と正本の所在

| 退避した項目                                          | 何の記録か                                                                                                                                                                                                                                                   | 意味の正本                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 高速化 / メモリの波（2026-09-25〜29）                 | 参照層 / 実用層の 2 層の裁定・イテレーション 1「契約と土台」・イテレーション 2 の B3（anima の DiT 常駐）/ B4（irodori の条件側 K/V）・H-35 の OOM → device lost・K-70（Metal での anima の遅さの帰属）・イテレーション 3 の E1 段 1（BiRefNet の f16 系列） | ADR [0110](../decisions/0110-practical-tier-numerics-contract.md) / [0111](../decisions/0111-session-options-composition.md) / [0112](../decisions/0112-anima-transformer-residency.md) / [0113](../decisions/0113-birefnet-weight-series.md) / [0114](../decisions/0114-irodori-dit-context-split.md)・[perf-ledger](../perf-ledger.md) H-4 / H-30 / K-70・[research 2026-09-27 K-70](2026-09-27-k70-metal-per-op.md) |
| 幾何プロファイル（K-71）まわり（2026-09-27〜10-03）   | プロファイル表の登録（`apple-metal-3` / `nvidia-blackwell`）・行数バケット 7 段・アプリへの組み込み（`@karume/runtime/tune`）・Codex レビュー対応・隣接の小物                                                                                                | ADR [0115](../decisions/0115-geometry-profiles.md) / [0116](../decisions/0116-geometry-profile-row-buckets.md) / [0117](../decisions/0117-app-geometry-tuning.md)・[research 2026-09-27 K-70](2026-09-27-k70-metal-per-op.md)                                                                                                                                                                                          |
| Karume 専用コンテナ形式の波（2026-09-22〜25）         | 段 0〜3（`krm` / `krg`・manifest `karume/5`・HF の全 pin の移行）と 0.13.0 の公開                                                                                                                                                                            | ADR [0108](../decisions/0108-container-format.md) / [0109](../decisions/0109-manifest-v5-container.md)・[container-v1](../container-v1.md)・[research 2026-09-23](2026-09-23-container-ram-peak.md) / [2026-09-24](2026-09-24-part-length-ram-peak.md)                                                                                                                                                                 |
| モデル横断の追加調査（2026-09-10〜15）                | Qwen3 / MiniCPM5 / E4B / gemma4-qat の実機検証と K-25〜K-44 の採否                                                                                                                                                                                           | [research 2026-09-10 追加調査](2026-09-10-codex-mtp-optimization.md)・ADR [0097](../decisions/0097-gemma4-qat-integration.md) / [0104](../decisions/0104-gemma-fast-quant.md)・[perf-ledger](../perf-ledger.md)                                                                                                                                                                                                        |
| 0.12.0 の公開と OP / Fusion の波（2026-09-06）        | 0.12.0 の公開手順・融合候補 3 件（P-5 / K-15 / K-7）の決着                                                                                                                                                                                                   | [perf-ledger](../perf-ledger.md)・[research 2026-09-06](2026-09-06-fusion-spikes-k15-k7.md)・[前回の退避](2026-09-22-backlog-archive-0.5.0-to-0.12.0.md)                                                                                                                                                                                                                                                               |
| 性能波 K-21 → H-15 と MTP 段 1〜4-B（2026-09-07〜10） | 小 M の linear の GEMV 族化・slot backing の予算つき保持・投機的デコードの段 1〜4-B（ゲート v3.1 まで）                                                                                                                                                      | ADR [0095](../decisions/0095-plan-backing-budget.md) / [0096](../decisions/0096-speculative-decoding.md)・[perf-ledger](../perf-ledger.md) K-20 / K-21 / H-15・[research 2026-09-07](2026-09-07-gemv-rows-k21.md) / [2026-09-09](2026-09-09-mtp-stage4.md)                                                                                                                                                             |

## 退避本文（逐語）

### now 節（0.13.0 リリース後）から — 2026-10-10 退避

- **高速化 / メモリの波（起票 2026-09-25・利用者裁定 = 参照層 / 実用層の 2 層に分ける・実用層でもデバイス内決定性 MUST）**:
  調査（Fable 5 レンズ + Opus 深掘り 19 + 反証 19 + Fable レビュー 4）の正本は `.claude/reviews/2026-09-25_perf-recon/SUMMARY.md`（git 追跡外）。
  **イテレーション 1「契約と土台」は済（2026-09-26・19 コミット）**: ADR [0110](../decisions/0110-practical-tier-numerics-contract.md) /
  [0111](../decisions/0111-session-options-composition.md)・`lastRunPipelines` + census 移行 + 束の census 表・同機 A/B 門（anima / irodori / sbv2）・
  `tools/flag-bench`・gemma 投機ゲートの門・`stateAttentionReduce` の manifest 昇格（Breaking）・`-fast` 束の実測確定
  （[research 2026-09-26](../research/2026-09-26-gemma-flag-bench.md)）・gemma4 recipe の `--model e4b` と PARALLEL_SHAPES の E4B 行。
  残件は消化済み（裁定 2026-09-26）: ② sha 参照行を 6 系列へ広げた（gemma4 / gemma4-qat の quant 席・siglip2 / gemma4 の golden・
  birefnet / depth-anything の実画像・vowel-detector の全鎖 — ADR [0106](../decisions/0106-device-keyed-references.md) 追記その 3）
  ③ BiRefNet 2048² を B570 の環境キーの held 行で明示 SKIP（`HELD_SERIES`）④ `sessionGpuFeatures` を全 8 系列へ統一（ADR 0111 追記）。
  ① E4B 通常の export は later 節、⑤ HF 再アップロード + pin 更新は release 節へ移した。
  次のイテレーション候補 = SUMMARY §9.3 の波 2（DiT 速度: B3 anima DiT 常駐 → B4 irodori
  H-30 → B1 → B2）/ 波 3（メモリ: E1 BiRefNet f16 系列 → E2 → E5 → E4）/ 波 4（実用層カーネル: C2 VAE conv i8a8 → C1 → D2 → D1 → D3）。
  台帳の候補は perf-ledger 2026-09-26 節（K-59〜K-69・L-20〜L-23）。
  **イテレーション 2「DiT 速度」（2026-09-26〜）**: **B3 は済・採用**（anima の DiT 常駐・opt-in のまま — ADR [0112](../decisions/0112-anima-transformer-residency.md)。
  利得は kill 線を超えた — 数値は perf-ledger H-4 と [research 2026-09-26](../research/2026-09-26-anima-residency-bench.md)）。
  OOM 退避 → やり直しが B570 で device lost になった件（H-35）は**調査と修正が済（2026-09-27）** — 原因は staging の OOM が
  device を失わせること・修正は測った空きで先に退避する形（ADR 0112 追記 2026-09-27・[research 2026-09-27](../research/2026-09-27-h35-oom-device-lost.md)）。
  確認ページは済（2026-09-29 に `tools/gpu-lab` の Anima タブへ統合・`deno task bench:gpu-lab`）。**B4 irodori H-30 は済（2026-09-27・ADR [0114](../decisions/0114-irodori-dit-context-split.md)）** — 条件側 K/V 射影を別グラフ `dit_context` に割った・ビット同一・採用（数値は perf-ledger H-30）。配布形の反映は release 節。**並行**: Apple / Metal での anima の遅さの帰属（perf-ledger K-70・2026-09-27 起票）— M2 の per-op 実測と帰属まで済（大タイルの GEMM 幾何 + i8a8 の Metal 展開・[research 2026-09-27 K-70](../research/2026-09-27-k70-metal-per-op.md)）・幾何掃引（M2 の quick + full）から GEMM 幾何を adapter ごとの静的プロファイルで選ぶ実装と `apple-metal-3` の登録まで済（perf-ledger K-71・ADR [0115](../decisions/0115-geometry-profiles.md)）・M2 の DiT 再測で確定（2026-09-29・f16 quant ×1.78）。
  **イテレーション 3（メモリ）の E1 段 1 = BiRefNet の f16 系列は recipe / dist / テスト / 4 系列の export まで済（2026-09-26・ADR [0113](../decisions/0113-birefnet-weight-series.md)）** — 残りは実 GPU の golden 突合・Lucida f16 の tolerance 導出・実画像の sha 参照行の作成と、段 2（i8 計画 + i8 系列）。
  既存の f32 系列 4 本は上流 revision を持たないため、配布に反映するときは `--dtype f32` で焼き直す（`outputs/misc/e1-dist/series/` の焼き直し版が使える）。

- **幾何プロファイル（K-71）まわり（起票 2026-09-27・裁定 2026-09-29 = 推奨案で全承認）**: 正本は ADR [0115 追記 2026-09-29](../decisions/0115-geometry-profiles.md)。
  ① 掃引ケースの追加は済（linear の M = 16 / 32 / 128 / 256・matmul 3 本〈linear の鏡像 — census に rank-2 matmul は無い〉・bmm 5 本〈census〉。生成器は 3 経路のケースで `gemmRows` を決める）。
  M2 の full 再走（op = linear / matmul / bmm）と `apple-metal-3` の 3 本からの再生成も済（2026-09-29）: `gemmRows` ≤ 64 と 65〜512 は既定へ戻り、> 512 は `reg128x32r8x4w8` ×1.65 で維持（DiT の幾何は不変 — ADR 0115 追記決定 3）。
  観察 = ≤ 64 の中で最良幾何が M 16 / 32 / 64 で割れる（境界の細分化は別起票候補）。M2 の anima 再測も済（2026-09-29・512² の DiT 段 f16 quant ×1.78・既定 ×1.15・PNG sha 一致 → **K-71 採用で確定**・ADR 0115 追記決定 4）。
  同じ再測で Metal では既定 quant（a8）が f16 より 1.58 倍遅い — 量子化 opt-in の再検討の材料。
  ② `nvidia-blackwell` は生成・登録済（B570 の per-profile GPU テスト緑）。残り = RTX 5070 Ti（Chrome）で anima の登録前後の PNG sha 一致と診断 `geometryProfile` の確認（利用者）。
  ③ 小さい bmm ケースの反復上限は済（裁定 2026-09-29 = 掃引専用の `SWEEP_MAX_REPS` 16384 に分けた・B570 の bmm 5 ケースは再測定比 0.9995〜1.0049）。
  確認ページの `geometryProfile` 表示も済（同日）。RTX 5070 Ti の実走も済（2026-09-29・4 段とも `nvidia-blackwell`・DiT 段 既定 1.21 s / f16 2.86 s — ADR 0115 追記 5・3 機比較は research K-70 §12）。登録前後の sha 一致だけは登録前の RTX 記録が無く未確認（任意: `cd11cdc8` の確認ページで 1 回）。
  **次の波（承認済み 2026-09-29・裁定 3 = a）: 行数バケットの細分化 — ADR [0116](../decisions/0116-geometry-profile-row-buckets.md) 起草済み（2026-10-01・利用者裁定で確定）**: プロファイルの `gemmRows` を固定の 7 段（16 / 32 / 64 / 128 / 256 / 512 / ∞）にする（既定の表は 3 段のまま・境界は既定の細分 MUST）。段 1 済（ADR・ADR 0115 決定 4 の改定注記・境界の定数と掃引ケース 12 本〈M 16 / 32 / 128 / 256 の matmul 4・bmm 8〉の追加 — 45 → 57 ケース）。段 2 済（2026-10-01・3 機の full〈全 op・57 ケース〉・失敗 0・不一致 0・再測定比の範囲外は M5 の 2 件だけ）。段 3 済（2026-10-01・`8fbcef74`・derive.ts の 7 段化 + 故障注入 3 通り・既存 full からの試走は調査のシミュレーションと 21 段一致）→ 段 4 済（2026-10-01・再生成 `c358832b`〈`--check` 同一・7 規則・M2 の ③PV が `reg64x64r4x8w8` に・RTX は 129〜512 の 2 段が新規・linear だけで決めていた段の採用 9 件中 6 件は既定へ〉・per-profile テスト `d1294daa` は 2 表 × 7 規則で緑）→ 段 5 = codegen スナップショット無変更 ✅・**M2 の `f16` 席 ✅（2026-10-02・×1.20・sha 一致・research K-70 §15）・RTX の `f16` 席 ✅（2026-10-02・DiT GPU ×1.000 で中立〈表は f32 の > 512 を変えない〉・sha 一致）→ **ADR 0116 検収 段 5 完了**。裁定 2026-10-02: 既定側が Chrome 154 の下で速くなっており〈帰属は推定〉旧 kill 線（1.3 倍）を比では下回ったが、表を残し K-71 の kill 線を「表が既定より速くない or sha 不一致」に改定・apple-metal-3 の Chrome 154 だけでの再生成は次に M2 で掃引を取る時**。数値は research K-70 §15（Pixel 10a〈Mali-G715〉の quick+ と Anima の記録も同節 — 表は未登録・新版での再走は未）。
  **済（裁定 2026-09-30）**: (2) 生成器は既定の再測定比が範囲外のケースをその掃引の比の材料から外す（出力不一致と失敗は見る・ADR 0115 追記決定 8）。(4) 照合キーに `description` を足し、`match` を省いた表は注入専用・`BUILTIN_GEOMETRY_PROFILES` を公開（追記決定 7）。`apple-metal-3` は `"Apple M2"` のときだけ自動で当たり、4 本 + 除外で再生成（> 512 ×1.647 維持・③PV `reg32x64r4x8w8` ×1.635）。
  **M5 full の確認（2026-10-01）**: M5 の full（45 ケース・失敗 0・不一致 0・再測定比の範囲外 3）でも既定がほぼ最良で、`apple-metal-3` の採用幾何は M5 で ×0.77〜1.10・M5 単独の表は ≤ 64 と conv2d rows32 以外既定・i8a8 は M5 でも f32 計算より 1.14〜1.51 倍遅い（[research K-70](../research/2026-09-27-k70-metal-per-op.md) §14）。
  **独立レビュー（Codex・2026-10-02・`.claude/reviews/2026-10-02_codex-review/`）**: medium 7 / low 2 を独立検証（8 本とも holds）→ 修正済（記録の比は分子分母が正のときだけ・疎配列の拒否・別 realm の Promise・進捗通知を計測の外へ・candidateKernels・gpu-lab の A/B 条件固定と保存物の後始末・id 空の表）。L2〈import 時の純粋な定数構築〉は規約の明文化（ADR 0117 追記）。設計上の懸念は ADR 0117 / 0118 の追記に反映。
  **隣接の小物（起票 2026-10-01・4 件とも済 2026-10-02 — ① ④ は asserts 型述語 + 注入の門の device 上限〈スレッド数・workgroup_size・共有メモリ〉）**: ① runtime の `assertGeometryProfile` を asserts 型述語（`asserts profile is GeometryProfile`）にして、gpu-lab の `injectable-tables.ts` の `as GeometryProfile` を消す ② limitations の conv2d の節「`apple-metal-3`（Chrome の M2）の conv2d は tileN 64」は追記決定 7 の後は自動適用がフラグ有り Chrome か注入に限られる — 言い回しを直す ③ ADR 0115 追記決定 3 の観察文（境界は別 ADR で）に ADR 0116 へのポインタが無い ④ 注入の門は欠けた JSON を `GpuFeatureError` でなく `TypeError` で落とす・device の上限（invocation 256 / 共有 16,384 B）を見ない（`.claude/reviews/2026-10-01_app-integration-recon/SUMMARY.md` §8 — ADR 0117 の材料）。
  **起票（2026-10-01・利用者）: アプリへの組み込み（掃引 → 表の生成 → 保存 → 注入を利用者アプリ内で回す）** — 状態 = 読み取り調査済み・**利用者裁定済み（2026-10-01・推奨案 a × 3）**: ① 公開の形 = runtime のサブパス `@karume/runtime/tune`（内部の codegen 語彙を外に出さず、掃引とカーネルの版が必ず揃う）② 計測の門 = 量子化フラグでの一律拒否を「行ごとの丸め誤差の上界」の判定へ（ADR 0115 決定 4 の改定・3 機の full で 100 µs 丸めを模擬しても採否は不変）③ 保存キー = `provenance` に adapter 4 欄・runtime 版（またはカーネルの指紋）・ケース集合の版を構造化し、照合する純関数と `acquireGpu` のコールバック形を公開。**ADR [0117](../decisions/0117-app-geometry-tuning.md) 起草済み（2026-10-01・段 1 = ADR + ADR 0008 / 0115 の追記）**。決定 4〈runtime の版ではなくカーネルの指紋・埋め込み 2 表も指紋を持ち GEMM カーネルの変更で再生成を伴う〉も裁定済み（2026-10-01・推奨案）。段 2 済（2026-10-01・`d242008f`・掃引の核を `packages/runtime/src/tune/` へ・公開面 `@karume/runtime/tune` = `runGeometrySweep` / `deriveGeometryProfile` / `geometryProfileJson` + 型・CLI / gpu-lab / opbench / anima-residency は殻か再輸出・葉テスト 114 件同数・2 表 `--check` 同一・publish dry-run 緑・gpu-lab の実機の往復は利用者待ち）。段 3 済（2026-10-02・門を丸め誤差の上界 E ≤ 1% へ・3 機 × 200 試行の模擬で採否一致・外れは RTX の 1 観測）。段 4 済（2026-10-02・provenance 構造化 + 指紋 + 照合 + parse・2 表再生成・照合 12〜14 ms）→ 段 5 / 6 済（2026-10-02）→ **段 7 ✅（実機 M2 / M5 2026-10-02・フル verify 2026-10-03〈3,544 passed・赤の 1 本は既知の B570 OOM 門〉）→ ADR 0117 完了（全 7 段 ✅）**。**隣接の小物（起票 2026-10-03）**: `gpu_generation_context_test.ts` の OOM 門が B570 で赤になる既知の問題は、テスト側で destroy() の後に `onSubmittedWorkDone` を待ってから 1 GiB を確保する〈段 3 / 6 の e2e と同じ settle〉で機構側だけを直せる見込み — assert は変えない。**隣接の小物（起票 2026-10-02）**: gpu-lab の生成表の id の既定値が埋め込み表の id（apple-metal-3）と衝突しうる → `<vendor>-<arch>-generated` のような既定に（フラグ無しの Chrome でアプリの流れが通る）。現状ある物: `acquireGpu({ geometryProfile })` の注入口・型と `BUILTIN_GEOMETRY_PROFILES` の公開・表の JSON 直列化（末尾の `maxRows` の Infinity は `1e999`）・`tools/gpu-lab` での実証。設計論点: ① 掃引ハーネスと生成器が runtime の内部 `src/` を直接 import していて JSR パッケージとして配れない（`tools/geometry-sweep/harness.ts:49〜115`）② 生成器は timestamp-query かつ非量子化の記録しか受けない（`tools/geometry-sweep/derive.ts:270〜285`）ので、一般利用者の Chrome（フラグ無し・100 µs 量子化）では成立しない ③ 注入は `match` を照合しないので、保存キー（adapter 情報 + runtime 版 + ケース集合の版）を利用者側で持つ必要がある ④ 掃引の所要時間が記録に無い（JSON の `date` は終了時刻）。

- **Karume 専用コンテナ形式の波（起票 2026-09-22）**: 配布形を safetensors 方言から専用コンテナ
  （`krm` = モデル / `krg` = グラフ）へ移す。正本は [ADR 0108](../decisions/0108-container-format.md)（accepted）と
  [container-v1](../container-v1.md)（accepted・2026-09-22 に段 1 着手を裁定）。**段 0 は済（2026-09-22）** = ADR + 仕様 + CPU 試作
  （container の読み書き 18 テスト緑・anima transformer の上流重みが 567/567 バイト一致・
  LoRA のグラフ書き換え 454 本が `parseIrGraph` 通過）。試作の置き場は
  `.claude/reviews/2026-09-22_codex-format-design/spikes/`（git 追跡外）。
  **段 1 は済（2026-09-22 夜）**: IR v2 仕様 / TS の読み手 / 合流後の語彙で Session を組む経路 / Python の
  writer + reader / 移行 CLI / 合流層の鏡像 `verify_container`（検収の状況は ADR 0108 追記 1 の 15）。
  **段 2 は済（2026-09-22 深夜〜09-23）**: 裁定は [ADR 0109](../decisions/0109-manifest-v5-container.md)（manifest
  `karume/5` — 容器は部品 × dtype・2 文書の期待値 + part 0 を含む全 part の FileRef・model 単位の `assets` は
  残す・取得単位は part・block の sha256 は未検証の取得元だけ・Range は段 6 のまま）。済 = 2a 仕様 / 2b hub
  （`resolveSelection` / `openContainerSource`）/ 2c runtime（`BlockSource.verified`・`asset(name)`・
  `assets[].length`）/ 2d exporter（`karume migrate --manifest` のリポ丸ごとモード・資産の受け口・PLE の
  block 化 schema 3・`karume verify --container`）/ 2e models（8 系列の container 経路・部品差し替え席
  `components`・PLE を容器の資産から）。**検収①は閉じた**（ミラー 11 本を移行 → TS の読み手で一意 108 容器・
  block 40,939 本・initializer 31,882 本・66.8 GiB の sha256 と合流を全件確認 — ADR 0108 追記 3 の 7）。
  ローカルミラーは `models/`（移行済み `karume/5`）と、旧 `karume/4` のミラーはリポ外
  `~/workspace/karume-models-v4/`（git 追跡外）。2f（RAM ピーク harness `tools/ram-peak/matrix.ts`）も済 —
  検収②③は[研究記録](../research/2026-09-23-container-ram-peak.md)（warm は payload の digest 0 回・scan 型の
  ピークは seek 型の約 2 倍）。irodori-v4.1-small の再アップロードと pin 更新も済（検収①の実 pin — SHA 固定の取得元で
  `fromPretrained` → `generate` を通し、取得が全て pin の revision・同じ文と seed の WAV がローカルミラー経由と
  byte 同一）。
  **段 3e 済（2026-09-24）**: part 長の既定は 256 MiB のまま。段 2 から持ち越した RAM ピークの改善候補は、hub の
  scan 型の切り出しを view に（候補 3）と Session 構築を block ごとに読んでは上げる（候補 2(c)）を採り、取得層へ
  器を渡す使い回し（候補 1）は採らない（ADR 0108 追記 5・[研究記録](../research/2026-09-24-part-length-ram-peak.md)）。持ち越した宿題のうち 128 鎖の突合・`karume verify` のコンテナ席・`assets` の
  受け口は段 2 で閉じ、1 コンテナ複数グラフは形式の能力のまま `karume/5` では使わない（ADR 0109 決定 2）。
  段 0 の宿題だった `pushErrorScope('validation')` の同期区間を block 単位に割る費用は実測で閉じた
  （push/pop 1.81 µs / 回・フェンス 13.0 ms / 回 — ADR 0108 決定 9 に追記済み）。段 1〜6 の作るものと検収は
  ADR 0108 の段階分解の表が正本（ここには複写しない）。段 1 の前提だった `outputs/series/` の大掃除は
  実施済み（2026-09-22・38 項目・約 141 GB — 削除一覧は
  `.claude/reviews/2026-09-22_codex-format-design/outputs-cleanup-deleted.txt`。レーンが参照する系列は
  段 1 後に新形式で再生成する）。
  **段 3f 済（2026-09-24）**: docs を段 3d / 3e の実装へ揃えた。段 3a〜3d で決めた点と検収①②の状況は
  ADR 0108 追記 4、RAM の数え方は [container-v1](../container-v1.md) §11 が正本。
  段 3 の作るもののうち HF の全 pin の移行は **0.13.0 のリリース作業で済（2026-09-24）**: 10 リポを系列から
  `karume/0.13.0` で焼き直して旧 safetensors の削除つきで上げ直し（anima → 越境参照の anima-extra → 残り 8 本）、pin 10 本を
  削除後の main へ更新した（`f16b8998`）。断片化した part 5 本は later 参照。**0.13.0 は 2026-09-25 に公開済み**
  （Release `v0.13.0` = `4b167df8`・JSR 3 パッケージ・`smoke:published` 緑・リリースノートは
  `outputs/release/release-notes-v0.13.0.draft.md`）。anima / anima-extra の pin は権利付与文を含む
  main へ更新済み（`75b127b5`・karume.json は byte 同一 — 裁定 28 番）。

- **モデル横断の追加調査（2026-09-10〜11）**: Qwen3-0.6B / MiniCPM5-2B の RTN / GPTQ と
  E4B の全 PLE を含むローカル pipeline は実機検証済み。E4B chat も CPU / Deno / Chrome で一致。
  [初期品質参考値](../research/2026-09-12-llm-quality-baseline.md)はQwen/MiniCPMで保存済み。
  [TTFT・tok/s比較](../research/2026-09-12-llm-speed-baseline.md)を基準にM2の同条件追試とprefillの費用帰属を優先する。
  [Chrome比較ページとRTX実測](../research/2026-09-12-browser-llm-speed.md)を追加。通常/QAT E2BのローカルONNX・HF直接取得を検収済み。
  [利用者のM2実測とPLE行キャッシュ](../research/2026-09-12-ple-row-cache.md)を記録。H-22のM2改善後は利用者追試で速度向上と40生成の一致を確認。
  [WebMLとの同条件比較](../research/2026-09-12-webml-browser-speed.md)も追加。[Chromeプロファイルと候補比較](../research/2026-09-12-chrome-gemma-optimization.md)に基づき、
  大語彙INT8のc16を限定採用（[K-34追試](../research/2026-09-12-chrome-gemv-followup.md)）。K並列を明示指定で追加（[K-35](../research/2026-09-12-chrome-gemv-parallel.md)）。[M2の実測と採用判断](../research/2026-09-13-m2-gemv-adoption.md)から、通常/QAT E2Bに高速化付きの既定quantを定義。
  重みコピーK-33は費用対効果で見送り。次は融合・広い品質比較（Denoの暖機後CPU PLE展開は小さく、GPU転送の帰属は未完）、
  H-23は[M2の全26設定](../research/2026-09-13-m2-prefill-adoption.md)を検収し、比較画面のchunk64だけ細分化を初期選択に採用。
  K-36は[M2追試](../research/2026-09-13-rms-subgroup-reduction.md)まで完了。融合のquant選択は[ADR 0104](../decisions/0104-gemma-fast-quant.md)で統合。ホストの投入方針とprefillバケットのモデル既定への適用判断は残す。
  K-37は[M2追試](../research/2026-09-13-gemv-subgroup.md)まで完了。約0.5%の差で既定採用を見送る。
  K-38の並列GEMV subgroup32は任意指定を維持。[M2の80生成](../research/2026-09-13-m2-gemv-subgroup-adoption.md)は出力一致、速度改善なしで既定採用を見送る。（当時の画面は既存parallelの2設定20生成。現在の比較は下記attention）。[I4 N12288/K1536のL8比較](../research/2026-09-14-i4-lane-comparison.md)は通常E2Bの80生成で約10.7%遅く不採用。M2のGPU費用帰属は残る。Denoは必要な機能が未対応。
  利用者の2026-09-14依頼により、[マージ前レビュー資料と全差分索引](../research/2026-09-14-merge-review.md)を準備し、ACTIVE_DESIGNを現況の索引へ整理した。[独立レビューとM2再計測](../research/2026-09-14-merge-review-results.md)も完了し、今回の範囲で修正が必要な新規指摘はなし。マージとpushは実施済みで、公開（JSR / HF）は未実施。未完の最適化を資料準備と同時に完了扱いにしない。
  [添付参照資料の現行再検証](../research/2026-09-14-reference-rope-optimization.md)からK-41のpermuteコピー削減を実装しM2検収済み。[K-42のattention融合](../research/2026-09-15-attention-fusion.md)もM2の80生成で出力一致を確認したが利得は無く、任意指定に残す。[K-43の保留候補併用](../research/2026-09-15-held-combinations.md)から単独の[linear→SRQ融合](../research/2026-09-15-linear-static-quantize-fusion.md)をK-44として統合した。常駐scaleの借用・丸め障壁・元のSRQとの数値比較を検収。[M2のQAT40生成](../research/2026-09-15-m2-linear-srq-adoption.md)も検収済み。[高速quant宣言と明示上書き](../research/2026-09-15-gemma-fast-quant.md)を統合。次は投入政策・prefillバケットの適用判断。同じM2追試は再依頼しない。広い併用は未統合で、gate/up入力共有の費用調査も残る。
  広いchunk/複数容量への一括適用は見送り。ChromeのGPU Instance消失は原因の切り分けを残す。
  残件は配布 recipe / source 表、長文と広い品質評価。実験資産を公開済みモデルとして扱わない。
  [MiniCPM5 CLI](../../examples/minicpm5/README.md) / [Qwen3 CLI](../../examples/qwen3/README.md) は容器（`krm`）の系列を読む。
  旧 shard 形の `-probe` 系列は読まず、今はリポ内でその容器の系列を作れない（各 README）。
  容量 128 の対話・履歴整理・reset・中断と、公式 CPU 参照への多ターン一致を検収済み。
  Anima w4a8 / drafter / f16 GEMM の既存形状比較は不採用で完了。f16 M=1 GEMV は
  [単体・Qwen の検収](../research/2026-09-10-codex-mtp-optimization.md#f16-格納-m1-の-gemv2026-09-11)に基づき採用（K-25）。
  f32 M=1 も [両 LLM の検収](../research/2026-09-10-codex-mtp-optimization.md#f32-格納-m1-の-gemv2026-09-11)で採用（K-26）。
  幅128以下の RMS 正規化も [Anima 全体の検収](../research/2026-09-10-codex-mtp-optimization.md#anima-の-rms-正規化と並べ替え融合2026-09-11)で採用（K-28）。
  M2 の利用者による動作確認は報告済み。f16 / f32 / RMS128 の形状別自動数値検収は残る。利用者の希望により、容量拡張より最適化調査を優先する。
  TypeScript の [CPU profile / token-only 比較](../research/2026-09-10-codex-mtp-optimization.md#typescript-の実行費と-token-only-出力2026-09-11)は記録済み（H-18）。
  通常Gemma4 / QATの温度0・非投機decodeは保存グラフを変えず小出力化した（H-18・[実測](../research/2026-09-10-codex-mtp-optimization.md#gemmaの温度0decodeの小出力化2026-09-12)）。
  温度あり・penalty/bias・投機は従来経路。これらの転送削減とM2の追試は残る。
  Chrome / Deno の Gemma 比較は自由文まで実施済み。他課題・長文・M2 を次の検収へ残す。
  QAT mobile は [INT2 / SRQ の実形状試作と統合案](../research/2026-09-10-codex-mtp-optimization.md#qat-mobile-の-int2-と固定丸め2026-09-11)を記録（K-27）。
  [ADR 0097](../decisions/0097-gemma4-qat-integration.md) の統合は承認済み。`gemma4-qat` の E2B / E4B として、
  公開 INT2 IR・固定 SRQ・固定 writer・PLE の INT2 / INT4 読取は検収済み。
  公式 recipe / 配布形の全量変換・固定 bytes 一致・Deno/Chrome の短文比較は確認済み。CPU/GPU 差は SRQ 境界をまたぐ縮約差に帰属。
  family / 対話 CLI は E2B/E4B・Deno/Chrome・複数ターン・中断・解放と全体検証を完了。
  SRQ融合・境界探索短縮・INT2変種は当時の逐次実装で検証し、全体の安定利得が不足するため見送った（K-29）。その後、並列GEMVの下で単独のlinear→SRQ融合が改善したためK-44として統合済み（上の2026-09-15の行が正本）。残るのは一般samplingの転送削減。
  実測と未完の正本は [追加調査](../research/2026-09-10-codex-mtp-optimization.md#追加-llm-の実行と量子化別比較)。

0.12.0 は**公開完了**（2026-09-06 — lockstep bump `a24d656` → GitHub Release v0.12.0 → JSR 0.12.0 →
`deno task smoke:published` 緑・`KARUME_SOURCES` 10 本の疎通を確認。中身は
[退避した消化済み節](../research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md)）。
**OP / Fusion の波（2026-09-06）**: 在庫の融合候補 3 件（P-5 / K-15 / K-7）は実測で閉じた（採否と数値は
[perf-ledger](../perf-ledger.md)・記録は [research 2026-09-06](../research/2026-09-06-fusion-spikes-k15-k7.md)）。続きは
大所 = **gemma4 decode の GEMV 並列度**（2026-09-06 ユーザー裁定の a 案）: K-16 / K-14 / K-13 は済（perf-ledger ✅・
0.12.0 で公開）。その続きは
[退避した消化済み節](../research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md)の「decode 速度調査の波」と
later の「decode 速度の残り」。
波と独立に消化してよい残件はその下。

- **性能波 K-21 → H-15 は済**（2026-09-07 ユーザー裁定 a・[perf-ledger](../perf-ledger.md)）: ①K-21 `5701262`（小 M〈1 ≤ M ≤ 64〉の
  linear を GEMV 族の行ブロック変種へ・ビット同一）②H-15 `c7120f2`（slot backing をバイト予算つき LRU 保持へ — ADR
  [0095](../decisions/0095-plan-backing-budget.md)）。合計で 20 token prompt の prefill run 壁 **135 → 46 ms**・定常ターン壁
  **701 → 581 ms**（[research](../research/2026-09-07-gemv-rows-k21.md) §8〜9）。MTP の復活条件 ①②③ は満ちた —
  ③ E-4（2026-09-08・[research](../research/2026-09-08-mtp-ea-i4-target.md)）: i4 と同値の重みの代理 target で
  抽出的な長文脈の E[a] が k=3 で 2.3〜2.6・k=6 で 3.8〜5.0、更新した予測倍率は抽出的長文脈 1.8〜2.6×
  （自由文 0.9〜1.0×）。④ 設計は ADR [0096](../decisions/0096-speculative-decoding.md) で裁定済み（2026-09-08）。
  **段 1（verify 形の準備）は済**（runtime `63a3d98` / exporter `a809e06` / models `46dfbbd` — ring の法を capacity へ・
  deferred commit・`last_row [R]` + 出口 2 本〈logits + hidden〉・バケット 4 / 8・配布形は焼き直し済み・verify 緑）。
  **段 2（drafter の入口）は済**（runtime `e4957cc` / `57413ac`・exporter `38f88e1`・models / hub `066402d` — IR の external
  スロット + 共有 initializer の宣言・借り手 context〈`createGenerationContext({ borrow })`〉・readonly attention・drafter recipe
  〈i8 単一・k=3 展開・lm_head + argmax〉・role `drafter`・`ResolveOptions.weights`・`speculative` オプション。draft の一致
  1800 / 1800）。**段 3（投機ループ）は済**（runtime `4c4e2a5`・exporter `b63421d`〈drafter 呼び出し規約の訂正 + golden〉・
  generation core `c31af36`・models / tools `757d734` — 投機ループは `sequence.ts` の内側に DI・verify は deferred で
  「配送した frontier まで」commit・onRun hook・`GenerationStop.speculation`・温度に依らず張る〈token 列は非投機と厳密一致〉。門 =
  sequential 席で投機 / 非投機の 200 token × 3 ケース厳密一致・受理 1.51 / 2.01 / 2.14 token/cycle）。**段 4-A ✅ 2026-09-09**（`tools/mtp-bench` `5b73fc4`・demo `--speculative` `118689a`・[research 2026-09-09](../research/2026-09-09-mtp-stage4.md): 抽出 **1.81×** / 要約 1.41× / 対話 1.18× / 自由文 0.96×〈decode 相・k=3〉・採算 A / A\* で A\* = 1.72〜1.88・温度 1.0 でも受理率は greedy と同じ・token 列一致・長文脈の verify 超過 +5.7 ms は attention ①・draft 壁 15 ms の ≈9.5 ms は Deno の round trip・抽出は k=3 飽和。**M2**〈ユーザー実走〉: 抽出 1.27× / 要約 1.03× / 対話 0.93× / 自由文 0.76×・A\* = 2.17〜2.6・verify が decode の 1.75〜2.2 倍〈短文脈で既に +39 ms = M=4 linear 側〉・prefill 4.8K 54 s〈別項〉）。**4-B ④ ✅ 2026-09-09**（`f845f55` ゲート v2〈16 cycle ブロック × 2 連続で抜ける・8 cycle バーストで戻る〉・`63c6db3` 受理列挙の停止・`cca5a36`/`7f02818` mtp-bench 3 値・ADR 0096 決定 8・[research §6.1](../research/2026-09-09-mtp-stage4.md): RTX で auto ≈ always〈勝つ 3 条件 +1〜4%・自由文 0.965× vs always 0.948×〉・M2 の検収はユーザー実走待ち）。**残り = 4-B（承認 2026-09-09・反証後の順序: ④ 自己採算ゲート〈壁と受理の EWMA・非対称ヒステリシス 0.97 / 1.01・plain 側は幾何バックオフ 8→256・W1 は cycle 2 で先に採る・既定 on = 当時は「既定席では同一 seed でも稀に出力が変わりうる」と limitations に明記〈2026-09-26 に撤回 — 既定席の M=1 / M=4 同一を門で固定・ADR 0096 追記〉・`speculative: boolean | "always"`・前段として受理列挙を停止 token で止める仕様変更を別コミット〉→ ⑧ M2 の小 M linear〈**測定済み・否定** 2026-09-09: 静的ノブ `linearGemvRowsThreadTarget`（`87e702b`）で r1 / r2 / r4 を M2 で比べると既定 r1 が最良（verify 90.7 → 103.6 ms）・「重み 4 回読み」仮説は外れ・backend 別既定は不要・K-22 は機序の宿題として残す〉→ ⑤ は縮小〈③′ は既に M=4 で効いており M=1 限定は ①QK だけ・①′ の WGSL は既に M 一般なので適用条件 1 行 + テスト 3 本 + kernel doc の MUST・効き代は M2 で最大 −4 ms/cycle・非負なら残す〉→ ③ argmax → ⑥ k=7 + 3 値ゲート・⑦ 不採用）**。**⑤ ✅ / ③ ✅ 2026-09-09**（`06f82df` ①′ を M ≤ 8 へ〈RTX 中立・既定席で verify 行 0 が u32 一致〉/ `e370562` 2 相 argmax〈drafter の argmax 2.03 → 0.08 ms/run・ビット同一〉— [research §6.2 / §6.3](../research/2026-09-09-mtp-stage4.md)）。**M2 検収 ✅ 2026-09-09**（抽出 auto 1.379× / 対話 0.966× / 自由文 0.874× — ゲートは負けを半分に・0.99× には届かず → 残件 = ゲート v3〈強い信号の早抜け・バースト初期間隔 16・bench の sequence 使い回し〉）。**残り = ⑥（裁定待ち: parked か 2 drafter + 3 値ゲート）・ゲート v3・⑩・4-C docs**。**⑨ ✅ 2026-09-09**（`0c8bacd` per-cycle トレース = `onRun` の壁・確定数・ゲート状態 + mtp-bench の局面別バケット・[research §6.4](../research/2026-09-09-mtp-stage4.md): RTX では勝つ側は抜けず W1 プローブ 1〜3% のみ・cold は 1%・**ゲートが落とす plain step は長文脈で真の decode より +17〜30% 高い〈verify 形・物理行は 1 で原因は未帰属〉→ 新規 ⑩ = 原因の帰属〈gpu-timing をモード別に割る〉と対策**・M2 の trace〈[research §6.4.1](../research/2026-09-09-mtp-stage4.md)〉: 自由文の損の 62% は探索バースト・36% は抜けるまで・対話は 6% 負けを 63 cycle 続ける費用・v3 = 早抜け + 間隔 16 で ≈ 0.94×・warm start で ≈ 0.97×）。**ゲート v3 ✅ 2026-09-10**（`29e74b0` / `15b8111`・[research §6.5](../research/2026-09-09-mtp-stage4.md)・ADR 0096 追記: RTX cold は勝つ側で誤退出 0・W1 半減・M2 の v3（[research §6.5.1](../research/2026-09-09-mtp-stage4.md)）: cold 自由文 0.934× / 対話 0.959× / 抽出 1.361×・**warm 自由文 0.978× / 対話 0.972×**（残りは 1 ターン 1 回のバースト → exploreMax 512 級で 0.99× 圏）。warm の口は `746ad77` でターンごとに違う発話に直し写しは消えた（research §6.5 表 11′）。**新知見: warm の対話（勝つ課題）で 3 / 7 ターンが抜けて auto / always 0.96 → A/B 済み（`063c874`・research §6.5 表 11″）: 帰属は v3 の 1 ブロック早抜け → **v3.1 ✅ 2026-09-10**（`a28e56f`・`earlyLeave` 既定 off / `burstAbort` 0.15・RTX warm 対話 auto = always・[research §6.5.2](../research/2026-09-09-mtp-stage4.md)）。exploreMax 256 → 512（`e365d9a`）。M2 warm v3.1（exploreMax 256 ビルド）= 自由文 0.990× / 対話 0.981× で、512 の検収は M2 warm の再計測待ち。⑩ は host / readback 側と判明・帰属は run 内の相の壁が要る）。旧記述（⑤ の −6 ms / u32 同一は反証で撤回）= ③ argmax の 2 相化（drafter −1.5 ms/cycle・ビット同一が構造保証）④ 投機の on / off ゲート（k′ ∈ {0, k} を受理の移動平均で・`accepted` の切り詰め前加算の誤りも修正）⑤ ①′ / ③′ の行タイル化（verify M ≤ 8 を decode と同じレーン割り → attention −6 ms + 既定席で u32 同一・着手条件 = 4-A で attention ≥ 5 ms/cycle）⑥ k=7 drafter の再 export + 動的 k（chat 形式の E[a] で予測倍率が k=3 の 1.2× 以上のとき）⑦ GPU argmax / topk 出口（readback + ホスト受理が cycle の 10% 以上のとき）→ 4-C docs（ADR 0096 の段表 段 4 行を現状へ消し込む — 採算表と「fence 1 本化は不成立」は ADR 追記と research §5 に記録済み）。裁定: ブラウザ実測はリリース後・M2 のミラーはユーザーが scp。事前プローブ（RTX・P≈4.8K・k=3・greedy）: decode 29.0 ms/run〈GPU 25.2〉・verify 37.0〈GPU 35.4 = decode + attention ①/③′ +6.0 + linear +2.2 + lm_head +0.8〉・draft 15.7〈GPU 6.5・argmax 1.8〉→ cycle 52.7 ms / 2.15 token = 24.5 ms/token = **1.18×**（decode 相・ブラウザ相当 ≈1.3×）。旧記述: 実測 Deno / Metal・温度 1.0 での受理率・投機を張る文脈長の閾値・動的 k・GPU argmax・
  fence 削減・①′ の位置不変化）。
