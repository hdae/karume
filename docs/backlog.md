# Backlog — 波順と作業項目の正本

> プロジェクト全体の**優先順位・波順・未消化項目**の正本はこの 1 本。
> 状態語彙: `now`（現行波）/ `next`（次の大波）/ `later` / `release` / `parked`（復活条件つき）。
> 運用契約: ①完了した項目は**削除する**（履歴は git と ADR / research が持つ）②実測値・設計論証は
> ここに**書かない** — 出典（ADR / research / 台帳）を指す ③性能候補の起票・採否・kill 基準は
> [perf-ledger](perf-ledger.md) が正本で、ここは波として参照するだけ ④by-design 制約の正本は
> [limitations](limitations.md) — 作業化が裁定された時だけここに載る。

## now — 0.13.0 リリース後（2026-09-25）

- **高速化 / メモリの波（起票 2026-09-25・利用者裁定 = 参照層 / 実用層の 2 層に分ける・実用層でもデバイス内決定性 MUST）**:
  調査（Fable 5 レンズ + Opus 深掘り 19 + 反証 19 + Fable レビュー 4）の正本は `.claude/reviews/2026-09-25_perf-recon/SUMMARY.md`（git 追跡外）。
  **イテレーション 1「契約と土台」は済（2026-09-26・19 コミット）**: ADR [0110](decisions/0110-practical-tier-numerics-contract.md) /
  [0111](decisions/0111-session-options-composition.md)・`lastRunPipelines` + census 移行 + 束の census 表・同機 A/B 門（anima / irodori / sbv2）・
  `tools/flag-bench`・gemma 投機ゲートの門・`stateAttentionReduce` の manifest 昇格（Breaking）・`-fast` 束の実測確定
  （[research 2026-09-26](research/2026-09-26-gemma-flag-bench.md)）・gemma4 recipe の `--model e4b` と PARALLEL_SHAPES の E4B 行。
  残件は消化済み（裁定 2026-09-26）: ② sha 参照行を 6 系列へ広げた（gemma4 / gemma4-qat の quant 席・siglip2 / gemma4 の golden・
  birefnet / depth-anything の実画像・vowel-detector の全鎖 — ADR [0106](decisions/0106-device-keyed-references.md) 追記その 3）
  ③ BiRefNet 2048² を B570 の環境キーの held 行で明示 SKIP（`HELD_SERIES`）④ `sessionGpuFeatures` を全 8 系列へ統一（ADR 0111 追記）。
  ① E4B 通常の export は later 節、⑤ HF 再アップロード + pin 更新は release 節へ移した。
  次のイテレーション候補 = SUMMARY §9.3 の波 2（DiT 速度: B3 anima DiT 常駐 → B4 irodori
  H-30 → B1 → B2）/ 波 3（メモリ: E1 BiRefNet f16 系列 → E2 → E5 → E4）/ 波 4（実用層カーネル: C2 VAE conv i8a8 → C1 → D2 → D1 → D3）。
  台帳の候補は perf-ledger 2026-09-26 節（K-59〜K-69・L-20〜L-23）。
  **イテレーション 2「DiT 速度」（2026-09-26〜）**: **B3 は済・採用**（anima の DiT 常駐・opt-in のまま — ADR [0112](decisions/0112-anima-transformer-residency.md)。
  利得は kill 線を超えた — 数値は perf-ledger H-4 と [research 2026-09-26](research/2026-09-26-anima-residency-bench.md)）。
  OOM 退避 → やり直しが B570 で device lost になった件（H-35）は**調査と修正が済（2026-09-27）** — 原因は staging の OOM が
  device を失わせること・修正は測った空きで先に退避する形（ADR 0112 追記 2026-09-27・[research 2026-09-27](research/2026-09-27-h35-oom-device-lost.md)）。
  確認ページは済（2026-09-29 に `tools/gpu-lab` の Anima タブへ統合・`deno task bench:gpu-lab`）。**B4 irodori H-30 は済（2026-09-27・ADR [0114](decisions/0114-irodori-dit-context-split.md)）** — 条件側 K/V 射影を別グラフ `dit_context` に割った・ビット同一・採用（数値は perf-ledger H-30）。配布形の反映は release 節。**並行**: Apple / Metal での anima の遅さの帰属（perf-ledger K-70・2026-09-27 起票）— M2 の per-op 実測と帰属まで済（大タイルの GEMM 幾何 + i8a8 の Metal 展開・[research 2026-09-27 K-70](research/2026-09-27-k70-metal-per-op.md)）・幾何掃引（M2 の quick + full）から GEMM 幾何を adapter ごとの静的プロファイルで選ぶ実装と `apple-metal-3` の登録まで済（perf-ledger K-71・ADR [0115](decisions/0115-geometry-profiles.md)）・M2 の DiT 再測で確定（2026-09-29・f16 quant ×1.78）。
  **イテレーション 3（メモリ）の E1 段 1 = BiRefNet の f16 系列は recipe / dist / テスト / 4 系列の export まで済（2026-09-26・ADR [0113](decisions/0113-birefnet-weight-series.md)）** — 残りは実 GPU の golden 突合・Lucida f16 の tolerance 導出・実画像の sha 参照行の作成と、段 2（i8 計画 + i8 系列）。
  既存の f32 系列 4 本は上流 revision を持たないため、配布に反映するときは `--dtype f32` で焼き直す（`outputs/misc/e1-dist/series/` の焼き直し版が使える）。

- **要検討: 既定の quant 席を量子化にするか opt-in（元の重み）にするか（起票 2026-09-27・利用者）**: 今は既定で w8a8 などの量子化席を選ぶ。
  それが正当かを見直す。「Metal では既定 quant の a8 を外す」判断（perf-ledger K-70・ADR 0115 Consequences）は保留にしてここに合流した。
  **材料集めは済（2026-09-29）**: `.claude/reviews/2026-09-29_quant-default-recon/SUMMARY.md`（git 追跡外）— 席の決まり方（recipe の定数・hub の解決・利用者の口）・
  系列 × 席 33 行の表（格納型・session 宣言・DL サイズ・契約クラス）・席別の実測（B570 / M2 / RTX・未計測を明示）・既定を量子化にした過去の根拠の引用・論点 3 案が守るもの。次 = 利用者と議論。
  **材料の追加（2026-10-04・利用者の方針 — Wan の既定席の裁定で）**: 既定の quant 席は「今いちばん実用的な席」（軽く、品質の劣化が小さい席）にする。
  将来「元の重みにいちばん近い席」を既定へ転換する余地は残す（ADR 0120 裁定 2026-10-04 の 4）。Wan2.1 はこの方針で既定 = 実用席へ切り替えた。

- **幾何プロファイル（K-71）まわり（起票 2026-09-27・裁定 2026-09-29 = 推奨案で全承認）**: 正本は ADR [0115 追記 2026-09-29](decisions/0115-geometry-profiles.md)。
  ① 掃引ケースの追加は済（linear の M = 16 / 32 / 128 / 256・matmul 3 本〈linear の鏡像 — census に rank-2 matmul は無い〉・bmm 5 本〈census〉。生成器は 3 経路のケースで `gemmRows` を決める）。
  M2 の full 再走（op = linear / matmul / bmm）と `apple-metal-3` の 3 本からの再生成も済（2026-09-29）: `gemmRows` ≤ 64 と 65〜512 は既定へ戻り、> 512 は `reg128x32r8x4w8` ×1.65 で維持（DiT の幾何は不変 — ADR 0115 追記決定 3）。
  観察 = ≤ 64 の中で最良幾何が M 16 / 32 / 64 で割れる（境界の細分化は別起票候補）。M2 の anima 再測も済（2026-09-29・512² の DiT 段 f16 quant ×1.78・既定 ×1.15・PNG sha 一致 → **K-71 採用で確定**・ADR 0115 追記決定 4）。
  同じ再測で Metal では既定 quant（a8）が f16 より 1.58 倍遅い — 量子化 opt-in の再検討の材料。
  ② `nvidia-blackwell` は生成・登録済（B570 の per-profile GPU テスト緑）。残り = RTX 5070 Ti（Chrome）で anima の登録前後の PNG sha 一致と診断 `geometryProfile` の確認（利用者）。
  ③ 小さい bmm ケースの反復上限は済（裁定 2026-09-29 = 掃引専用の `SWEEP_MAX_REPS` 16384 に分けた・B570 の bmm 5 ケースは再測定比 0.9995〜1.0049）。
  確認ページの `geometryProfile` 表示も済（同日）。RTX 5070 Ti の実走も済（2026-09-29・4 段とも `nvidia-blackwell`・DiT 段 既定 1.21 s / f16 2.86 s — ADR 0115 追記 5・3 機比較は research K-70 §12）。登録前後の sha 一致だけは登録前の RTX 記録が無く未確認（任意: `cd11cdc8` の確認ページで 1 回）。
  **次の波（承認済み 2026-09-29・裁定 3 = a）: 行数バケットの細分化 — ADR [0116](decisions/0116-geometry-profile-row-buckets.md) 起草済み（2026-10-01・利用者裁定で確定）**: プロファイルの `gemmRows` を固定の 7 段（16 / 32 / 64 / 128 / 256 / 512 / ∞）にする（既定の表は 3 段のまま・境界は既定の細分 MUST）。段 1 済（ADR・ADR 0115 決定 4 の改定注記・境界の定数と掃引ケース 12 本〈M 16 / 32 / 128 / 256 の matmul 4・bmm 8〉の追加 — 45 → 57 ケース）。段 2 済（2026-10-01・3 機の full〈全 op・57 ケース〉・失敗 0・不一致 0・再測定比の範囲外は M5 の 2 件だけ）。段 3 済（2026-10-01・`8fbcef74`・derive.ts の 7 段化 + 故障注入 3 通り・既存 full からの試走は調査のシミュレーションと 21 段一致）→ 段 4 済（2026-10-01・再生成 `c358832b`〈`--check` 同一・7 規則・M2 の ③PV が `reg64x64r4x8w8` に・RTX は 129〜512 の 2 段が新規・linear だけで決めていた段の採用 9 件中 6 件は既定へ〉・per-profile テスト `d1294daa` は 2 表 × 7 規則で緑）→ 段 5 = codegen スナップショット無変更 ✅・**M2 の `f16` 席 ✅（2026-10-02・×1.20・sha 一致・research K-70 §15）・RTX の `f16` 席 ✅（2026-10-02・DiT GPU ×1.000 で中立〈表は f32 の > 512 を変えない〉・sha 一致）→ **ADR 0116 検収 段 5 完了**。裁定 2026-10-02: 既定側が Chrome 154 の下で速くなっており〈帰属は推定〉旧 kill 線（1.3 倍）を比では下回ったが、表を残し K-71 の kill 線を「表が既定より速くない or sha 不一致」に改定・apple-metal-3 の Chrome 154 だけでの再生成は次に M2 で掃引を取る時**。数値は research K-70 §15（Pixel 10a〈Mali-G715〉の quick+ と Anima の記録も同節 — 表は未登録・新版での再走は未）。
  **済（裁定 2026-09-30）**: (2) 生成器は既定の再測定比が範囲外のケースをその掃引の比の材料から外す（出力不一致と失敗は見る・ADR 0115 追記決定 8）。(4) 照合キーに `description` を足し、`match` を省いた表は注入専用・`BUILTIN_GEOMETRY_PROFILES` を公開（追記決定 7）。`apple-metal-3` は `"Apple M2"` のときだけ自動で当たり、4 本 + 除外で再生成（> 512 ×1.647 維持・③PV `reg32x64r4x8w8` ×1.635）。
  **M5 full の確認（2026-10-01）**: M5 の full（45 ケース・失敗 0・不一致 0・再測定比の範囲外 3）でも既定がほぼ最良で、`apple-metal-3` の採用幾何は M5 で ×0.77〜1.10・M5 単独の表は ≤ 64 と conv2d rows32 以外既定・i8a8 は M5 でも f32 計算より 1.14〜1.51 倍遅い（[research K-70](research/2026-09-27-k70-metal-per-op.md) §14）。
  **独立レビュー（Codex・2026-10-02・`.claude/reviews/2026-10-02_codex-review/`）**: medium 7 / low 2 を独立検証（8 本とも holds）→ 修正済（記録の比は分子分母が正のときだけ・疎配列の拒否・別 realm の Promise・進捗通知を計測の外へ・candidateKernels・gpu-lab の A/B 条件固定と保存物の後始末・id 空の表）。L2〈import 時の純粋な定数構築〉は規約の明文化（ADR 0117 追記）。設計上の懸念は ADR 0117 / 0118 の追記に反映。
  **隣接の小物（起票 2026-10-01・4 件とも済 2026-10-02 — ① ④ は asserts 型述語 + 注入の門の device 上限〈スレッド数・workgroup_size・共有メモリ〉）**: ① runtime の `assertGeometryProfile` を asserts 型述語（`asserts profile is GeometryProfile`）にして、gpu-lab の `injectable-tables.ts` の `as GeometryProfile` を消す ② limitations の conv2d の節「`apple-metal-3`（Chrome の M2）の conv2d は tileN 64」は追記決定 7 の後は自動適用がフラグ有り Chrome か注入に限られる — 言い回しを直す ③ ADR 0115 追記決定 3 の観察文（境界は別 ADR で）に ADR 0116 へのポインタが無い ④ 注入の門は欠けた JSON を `GpuFeatureError` でなく `TypeError` で落とす・device の上限（invocation 256 / 共有 16,384 B）を見ない（`.claude/reviews/2026-10-01_app-integration-recon/SUMMARY.md` §8 — ADR 0117 の材料）。
  **起票（2026-10-01・利用者）: アプリへの組み込み（掃引 → 表の生成 → 保存 → 注入を利用者アプリ内で回す）** — 状態 = 読み取り調査済み・**利用者裁定済み（2026-10-01・推奨案 a × 3）**: ① 公開の形 = runtime のサブパス `@karume/runtime/tune`（内部の codegen 語彙を外に出さず、掃引とカーネルの版が必ず揃う）② 計測の門 = 量子化フラグでの一律拒否を「行ごとの丸め誤差の上界」の判定へ（ADR 0115 決定 4 の改定・3 機の full で 100 µs 丸めを模擬しても採否は不変）③ 保存キー = `provenance` に adapter 4 欄・runtime 版（またはカーネルの指紋）・ケース集合の版を構造化し、照合する純関数と `acquireGpu` のコールバック形を公開。**ADR [0117](decisions/0117-app-geometry-tuning.md) 起草済み（2026-10-01・段 1 = ADR + ADR 0008 / 0115 の追記）**。決定 4〈runtime の版ではなくカーネルの指紋・埋め込み 2 表も指紋を持ち GEMM カーネルの変更で再生成を伴う〉も裁定済み（2026-10-01・推奨案）。段 2 済（2026-10-01・`d242008f`・掃引の核を `packages/runtime/src/tune/` へ・公開面 `@karume/runtime/tune` = `runGeometrySweep` / `deriveGeometryProfile` / `geometryProfileJson` + 型・CLI / gpu-lab / opbench / anima-residency は殻か再輸出・葉テスト 114 件同数・2 表 `--check` 同一・publish dry-run 緑・gpu-lab の実機の往復は利用者待ち）。段 3 済（2026-10-02・門を丸め誤差の上界 E ≤ 1% へ・3 機 × 200 試行の模擬で採否一致・外れは RTX の 1 観測）。段 4 済（2026-10-02・provenance 構造化 + 指紋 + 照合 + parse・2 表再生成・照合 12〜14 ms）→ 段 5 / 6 済（2026-10-02）→ **段 7 ✅（実機 M2 / M5 2026-10-02・フル verify 2026-10-03〈3,544 passed・赤の 1 本は既知の B570 OOM 門〉）→ ADR 0117 完了（全 7 段 ✅）**。**隣接の小物（起票 2026-10-03）**: `gpu_generation_context_test.ts` の OOM 門が B570 で赤になる既知の問題は、テスト側で destroy() の後に `onSubmittedWorkDone` を待ってから 1 GiB を確保する〈段 3 / 6 の e2e と同じ settle〉で機構側だけを直せる見込み — assert は変えない。**隣接の小物（起票 2026-10-02）**: gpu-lab の生成表の id の既定値が埋め込み表の id（apple-metal-3）と衝突しうる → `<vendor>-<arch>-generated` のような既定に（フラグ無しの Chrome でアプリの流れが通る）。現状ある物: `acquireGpu({ geometryProfile })` の注入口・型と `BUILTIN_GEOMETRY_PROFILES` の公開・表の JSON 直列化（末尾の `maxRows` の Infinity は `1e999`）・`tools/gpu-lab` での実証。設計論点: ① 掃引ハーネスと生成器が runtime の内部 `src/` を直接 import していて JSR パッケージとして配れない（`tools/geometry-sweep/harness.ts:49〜115`）② 生成器は timestamp-query かつ非量子化の記録しか受けない（`tools/geometry-sweep/derive.ts:270〜285`）ので、一般利用者の Chrome（フラグ無し・100 µs 量子化）では成立しない ③ 注入は `match` を照合しないので、保存キー（adapter 情報 + runtime 版 + ケース集合の版）を利用者側で持つ必要がある ④ 掃引の所要時間が記録に無い（JSON の `date` は終了時刻）。

- **Karume 専用コンテナ形式の波（起票 2026-09-22）**: 配布形を safetensors 方言から専用コンテナ
  （`krm` = モデル / `krg` = グラフ）へ移す。正本は [ADR 0108](decisions/0108-container-format.md)（accepted）と
  [container-v1](container-v1.md)（accepted・2026-09-22 に段 1 着手を裁定）。**段 0 は済（2026-09-22）** = ADR + 仕様 + CPU 試作
  （container の読み書き 18 テスト緑・anima transformer の上流重みが 567/567 バイト一致・
  LoRA のグラフ書き換え 454 本が `parseIrGraph` 通過）。試作の置き場は
  `.claude/reviews/2026-09-22_codex-format-design/spikes/`（git 追跡外）。
  **段 1 は済（2026-09-22 夜）**: IR v2 仕様 / TS の読み手 / 合流後の語彙で Session を組む経路 / Python の
  writer + reader / 移行 CLI / 合流層の鏡像 `verify_container`（検収の状況は ADR 0108 追記 1 の 15）。
  **段 2 は済（2026-09-22 深夜〜09-23）**: 裁定は [ADR 0109](decisions/0109-manifest-v5-container.md)（manifest
  `karume/5` — 容器は部品 × dtype・2 文書の期待値 + part 0 を含む全 part の FileRef・model 単位の `assets` は
  残す・取得単位は part・block の sha256 は未検証の取得元だけ・Range は段 6 のまま）。済 = 2a 仕様 / 2b hub
  （`resolveSelection` / `openContainerSource`）/ 2c runtime（`BlockSource.verified`・`asset(name)`・
  `assets[].length`）/ 2d exporter（`karume migrate --manifest` のリポ丸ごとモード・資産の受け口・PLE の
  block 化 schema 3・`karume verify --container`）/ 2e models（8 系列の container 経路・部品差し替え席
  `components`・PLE を容器の資産から）。**検収①は閉じた**（ミラー 11 本を移行 → TS の読み手で一意 108 容器・
  block 40,939 本・initializer 31,882 本・66.8 GiB の sha256 と合流を全件確認 — ADR 0108 追記 3 の 7）。
  ローカルミラーは `models/`（移行済み `karume/5`）と、旧 `karume/4` のミラーはリポ外
  `~/workspace/karume-models-v4/`（git 追跡外）。2f（RAM ピーク harness `tools/ram-peak/matrix.ts`）も済 —
  検収②③は[研究記録](research/2026-09-23-container-ram-peak.md)（warm は payload の digest 0 回・scan 型の
  ピークは seek 型の約 2 倍）。irodori-v4.1-small の再アップロードと pin 更新も済（検収①の実 pin — SHA 固定の取得元で
  `fromPretrained` → `generate` を通し、取得が全て pin の revision・同じ文と seed の WAV がローカルミラー経由と
  byte 同一）。
  **段 3e 済（2026-09-24）**: part 長の既定は 256 MiB のまま。段 2 から持ち越した RAM ピークの改善候補は、hub の
  scan 型の切り出しを view に（候補 3）と Session 構築を block ごとに読んでは上げる（候補 2(c)）を採り、取得層へ
  器を渡す使い回し（候補 1）は採らない（ADR 0108 追記 5・[研究記録](research/2026-09-24-part-length-ram-peak.md)）。持ち越した宿題のうち 128 鎖の突合・`karume verify` のコンテナ席・`assets` の
  受け口は段 2 で閉じ、1 コンテナ複数グラフは形式の能力のまま `karume/5` では使わない（ADR 0109 決定 2）。
  段 0 の宿題だった `pushErrorScope('validation')` の同期区間を block 単位に割る費用は実測で閉じた
  （push/pop 1.81 µs / 回・フェンス 13.0 ms / 回 — ADR 0108 決定 9 に追記済み）。段 1〜6 の作るものと検収は
  ADR 0108 の段階分解の表が正本（ここには複写しない）。段 1 の前提だった `outputs/series/` の大掃除は
  実施済み（2026-09-22・38 項目・約 141 GB — 削除一覧は
  `.claude/reviews/2026-09-22_codex-format-design/outputs-cleanup-deleted.txt`。レーンが参照する系列は
  段 1 後に新形式で再生成する）。
  **段 3f 済（2026-09-24）**: docs を段 3d / 3e の実装へ揃えた。段 3a〜3d で決めた点と検収①②の状況は
  ADR 0108 追記 4、RAM の数え方は [container-v1](container-v1.md) §11 が正本。
  段 3 の作るもののうち HF の全 pin の移行は **0.13.0 のリリース作業で済（2026-09-24）**: 10 リポを系列から
  `karume/0.13.0` で焼き直して旧 safetensors の削除つきで上げ直し（anima → 越境参照の anima-extra → 残り 8 本）、pin 10 本を
  削除後の main へ更新した（`f16b8998`）。断片化した part 5 本は later 参照。**0.13.0 は 2026-09-25 に公開済み**
  （Release `v0.13.0` = `4b167df8`・JSR 3 パッケージ・`smoke:published` 緑・リリースノートは
  `outputs/release/release-notes-v0.13.0.draft.md`）。anima / anima-extra の pin は権利付与文を含む
  main へ更新済み（`75b127b5`・karume.json は byte 同一 — 裁定 28 番）。

- **モデル横断の追加調査（2026-09-10〜11）**: Qwen3-0.6B / MiniCPM5-2B の RTN / GPTQ と
  E4B の全 PLE を含むローカル pipeline は実機検証済み。E4B chat も CPU / Deno / Chrome で一致。
  [初期品質参考値](research/2026-09-12-llm-quality-baseline.md)はQwen/MiniCPMで保存済み。
  [TTFT・tok/s比較](research/2026-09-12-llm-speed-baseline.md)を基準にM2の同条件追試とprefillの費用帰属を優先する。
  [Chrome比較ページとRTX実測](research/2026-09-12-browser-llm-speed.md)を追加。通常/QAT E2BのローカルONNX・HF直接取得を検収済み。
  [利用者のM2実測とPLE行キャッシュ](research/2026-09-12-ple-row-cache.md)を記録。H-22のM2改善後は利用者追試で速度向上と40生成の一致を確認。
  [WebMLとの同条件比較](research/2026-09-12-webml-browser-speed.md)も追加。[Chromeプロファイルと候補比較](research/2026-09-12-chrome-gemma-optimization.md)に基づき、
  大語彙INT8のc16を限定採用（[K-34追試](research/2026-09-12-chrome-gemv-followup.md)）。K並列を明示指定で追加（[K-35](research/2026-09-12-chrome-gemv-parallel.md)）。[M2の実測と採用判断](research/2026-09-13-m2-gemv-adoption.md)から、通常/QAT E2Bに高速化付きの既定quantを定義。
  重みコピーK-33は費用対効果で見送り。次は融合・広い品質比較（Denoの暖機後CPU PLE展開は小さく、GPU転送の帰属は未完）、
  H-23は[M2の全26設定](research/2026-09-13-m2-prefill-adoption.md)を検収し、比較画面のchunk64だけ細分化を初期選択に採用。
  K-36は[M2追試](research/2026-09-13-rms-subgroup-reduction.md)まで完了。融合のquant選択は[ADR 0104](decisions/0104-gemma-fast-quant.md)で統合。ホストの投入方針とprefillバケットのモデル既定への適用判断は残す。
  K-37は[M2追試](research/2026-09-13-gemv-subgroup.md)まで完了。約0.5%の差で既定採用を見送る。
  K-38の並列GEMV subgroup32は任意指定を維持。[M2の80生成](research/2026-09-13-m2-gemv-subgroup-adoption.md)は出力一致、速度改善なしで既定採用を見送る。（当時の画面は既存parallelの2設定20生成。現在の比較は下記attention）。[I4 N12288/K1536のL8比較](research/2026-09-14-i4-lane-comparison.md)は通常E2Bの80生成で約10.7%遅く不採用。M2のGPU費用帰属は残る。Denoは必要な機能が未対応。
  利用者の2026-09-14依頼により、[マージ前レビュー資料と全差分索引](research/2026-09-14-merge-review.md)を準備し、ACTIVE_DESIGNを現況の索引へ整理した。[独立レビューとM2再計測](research/2026-09-14-merge-review-results.md)も完了し、今回の範囲で修正が必要な新規指摘はなし。マージとpushは実施済みで、公開（JSR / HF）は未実施。未完の最適化を資料準備と同時に完了扱いにしない。
  [添付参照資料の現行再検証](research/2026-09-14-reference-rope-optimization.md)からK-41のpermuteコピー削減を実装しM2検収済み。[K-42のattention融合](research/2026-09-15-attention-fusion.md)もM2の80生成で出力一致を確認したが利得は無く、任意指定に残す。[K-43の保留候補併用](research/2026-09-15-held-combinations.md)から単独の[linear→SRQ融合](research/2026-09-15-linear-static-quantize-fusion.md)をK-44として統合した。常駐scaleの借用・丸め障壁・元のSRQとの数値比較を検収。[M2のQAT40生成](research/2026-09-15-m2-linear-srq-adoption.md)も検収済み。[高速quant宣言と明示上書き](research/2026-09-15-gemma-fast-quant.md)を統合。次は投入政策・prefillバケットの適用判断。同じM2追試は再依頼しない。広い併用は未統合で、gate/up入力共有の費用調査も残る。
  広いchunk/複数容量への一括適用は見送り。ChromeのGPU Instance消失は原因の切り分けを残す。
  残件は配布 recipe / source 表、長文と広い品質評価。実験資産を公開済みモデルとして扱わない。
  [MiniCPM5 CLI](../examples/minicpm5/README.md) / [Qwen3 CLI](../examples/qwen3/README.md) は容器（`krm`）の系列を読む。
  旧 shard 形の `-probe` 系列は読まず、今はリポ内でその容器の系列を作れない（各 README）。
  容量 128 の対話・履歴整理・reset・中断と、公式 CPU 参照への多ターン一致を検収済み。
  Anima w4a8 / drafter / f16 GEMM の既存形状比較は不採用で完了。f16 M=1 GEMV は
  [単体・Qwen の検収](research/2026-09-10-codex-mtp-optimization.md#f16-格納-m1-の-gemv2026-09-11)に基づき採用（K-25）。
  f32 M=1 も [両 LLM の検収](research/2026-09-10-codex-mtp-optimization.md#f32-格納-m1-の-gemv2026-09-11)で採用（K-26）。
  幅128以下の RMS 正規化も [Anima 全体の検収](research/2026-09-10-codex-mtp-optimization.md#anima-の-rms-正規化と並べ替え融合2026-09-11)で採用（K-28）。
  M2 の利用者による動作確認は報告済み。f16 / f32 / RMS128 の形状別自動数値検収は残る。利用者の希望により、容量拡張より最適化調査を優先する。
  TypeScript の [CPU profile / token-only 比較](research/2026-09-10-codex-mtp-optimization.md#typescript-の実行費と-token-only-出力2026-09-11)は記録済み（H-18）。
  通常Gemma4 / QATの温度0・非投機decodeは保存グラフを変えず小出力化した（H-18・[実測](research/2026-09-10-codex-mtp-optimization.md#gemmaの温度0decodeの小出力化2026-09-12)）。
  温度あり・penalty/bias・投機は従来経路。これらの転送削減とM2の追試は残る。
  Chrome / Deno の Gemma 比較は自由文まで実施済み。他課題・長文・M2 を次の検収へ残す。
  QAT mobile は [INT2 / SRQ の実形状試作と統合案](research/2026-09-10-codex-mtp-optimization.md#qat-mobile-の-int2-と固定丸め2026-09-11)を記録（K-27）。
  [ADR 0097](decisions/0097-gemma4-qat-integration.md) の統合は承認済み。`gemma4-qat` の E2B / E4B として、
  公開 INT2 IR・固定 SRQ・固定 writer・PLE の INT2 / INT4 読取は検収済み。
  公式 recipe / 配布形の全量変換・固定 bytes 一致・Deno/Chrome の短文比較は確認済み。CPU/GPU 差は SRQ 境界をまたぐ縮約差に帰属。
  family / 対話 CLI は E2B/E4B・Deno/Chrome・複数ターン・中断・解放と全体検証を完了。
  SRQ融合・境界探索短縮・INT2変種は当時の逐次実装で検証し、全体の安定利得が不足するため見送った（K-29）。その後、並列GEMVの下で単独のlinear→SRQ融合が改善したためK-44として統合済み（上の2026-09-15の行が正本）。残るのは一般samplingの転送削減。
  実測と未完の正本は [追加調査](research/2026-09-10-codex-mtp-optimization.md#追加-llm-の実行と量子化別比較)。

- **9/11 レビューの継続検証**（調査 2026-09-10・[対応記録](research/2026-09-10-codex-mtp-optimization.md)）:
  `artifacts.staged_publication` の同一 final への複数 writer を許容するか決め、必要ならロック・中断復旧を設計する。
  GPU 端チャネル保護の M2 / ブラウザ追試とモデル全体の性能計測は残る。
  レビュー提案を未検証のまま新規カーネルへしない。

0.12.0 は**公開完了**（2026-09-06 — lockstep bump `a24d656` → GitHub Release v0.12.0 → JSR 0.12.0 →
`deno task smoke:published` 緑・`KARUME_SOURCES` 10 本の疎通を確認。中身は
[退避した消化済み節](research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md)）。
**OP / Fusion の波（2026-09-06）**: 在庫の融合候補 3 件（P-5 / K-15 / K-7）は実測で閉じた（採否と数値は
[perf-ledger](perf-ledger.md)・記録は [research 2026-09-06](research/2026-09-06-fusion-spikes-k15-k7.md)）。続きは
大所 = **gemma4 decode の GEMV 並列度**（2026-09-06 ユーザー裁定の a 案）: K-16 / K-14 / K-13 は済（perf-ledger ✅・
0.12.0 で公開）。その続きは
[退避した消化済み節](research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md)の「decode 速度調査の波」と
later の「decode 速度の残り」。
波と独立に消化してよい残件はその下。

**残件**:

- **性能波 K-21 → H-15 は済**（2026-09-07 ユーザー裁定 a・[perf-ledger](perf-ledger.md)）: ①K-21 `5701262`（小 M〈1 ≤ M ≤ 64〉の
  linear を GEMV 族の行ブロック変種へ・ビット同一）②H-15 `c7120f2`（slot backing をバイト予算つき LRU 保持へ — ADR
  [0095](decisions/0095-plan-backing-budget.md)）。合計で 20 token prompt の prefill run 壁 **135 → 46 ms**・定常ターン壁
  **701 → 581 ms**（[research](research/2026-09-07-gemv-rows-k21.md) §8〜9）。MTP の復活条件 ①②③ は満ちた —
  ③ E-4（2026-09-08・[research](research/2026-09-08-mtp-ea-i4-target.md)）: i4 と同値の重みの代理 target で
  抽出的な長文脈の E[a] が k=3 で 2.3〜2.6・k=6 で 3.8〜5.0、更新した予測倍率は抽出的長文脈 1.8〜2.6×
  （自由文 0.9〜1.0×）。④ 設計は ADR [0096](decisions/0096-speculative-decoding.md) で裁定済み（2026-09-08）。
  **段 1（verify 形の準備）は済**（runtime `63a3d98` / exporter `a809e06` / models `46dfbbd` — ring の法を capacity へ・
  deferred commit・`last_row [R]` + 出口 2 本〈logits + hidden〉・バケット 4 / 8・配布形は焼き直し済み・verify 緑）。
  **段 2（drafter の入口）は済**（runtime `e4957cc` / `57413ac`・exporter `38f88e1`・models / hub `066402d` — IR の external
  スロット + 共有 initializer の宣言・借り手 context〈`createGenerationContext({ borrow })`〉・readonly attention・drafter recipe
  〈i8 単一・k=3 展開・lm_head + argmax〉・role `drafter`・`ResolveOptions.weights`・`speculative` オプション。draft の一致
  1800 / 1800）。**段 3（投機ループ）は済**（runtime `4c4e2a5`・exporter `b63421d`〈drafter 呼び出し規約の訂正 + golden〉・
  generation core `c31af36`・models / tools `757d734` — 投機ループは `sequence.ts` の内側に DI・verify は deferred で
  「配送した frontier まで」commit・onRun hook・`GenerationStop.speculation`・温度に依らず張る〈token 列は非投機と厳密一致〉。門 =
  sequential 席で投機 / 非投機の 200 token × 3 ケース厳密一致・受理 1.51 / 2.01 / 2.14 token/cycle）。**段 4-A ✅ 2026-09-09**（`tools/mtp-bench` `5b73fc4`・demo `--speculative` `118689a`・[research 2026-09-09](research/2026-09-09-mtp-stage4.md): 抽出 **1.81×** / 要約 1.41× / 対話 1.18× / 自由文 0.96×〈decode 相・k=3〉・採算 A / A\* で A\* = 1.72〜1.88・温度 1.0 でも受理率は greedy と同じ・token 列一致・長文脈の verify 超過 +5.7 ms は attention ①・draft 壁 15 ms の ≈9.5 ms は Deno の round trip・抽出は k=3 飽和。**M2**〈ユーザー実走〉: 抽出 1.27× / 要約 1.03× / 対話 0.93× / 自由文 0.76×・A\* = 2.17〜2.6・verify が decode の 1.75〜2.2 倍〈短文脈で既に +39 ms = M=4 linear 側〉・prefill 4.8K 54 s〈別項〉）。**4-B ④ ✅ 2026-09-09**（`f845f55` ゲート v2〈16 cycle ブロック × 2 連続で抜ける・8 cycle バーストで戻る〉・`63c6db3` 受理列挙の停止・`cca5a36`/`7f02818` mtp-bench 3 値・ADR 0096 決定 8・[research §6.1](research/2026-09-09-mtp-stage4.md): RTX で auto ≈ always〈勝つ 3 条件 +1〜4%・自由文 0.965× vs always 0.948×〉・M2 の検収はユーザー実走待ち）。**残り = 4-B（承認 2026-09-09・反証後の順序: ④ 自己採算ゲート〈壁と受理の EWMA・非対称ヒステリシス 0.97 / 1.01・plain 側は幾何バックオフ 8→256・W1 は cycle 2 で先に採る・既定 on = 当時は「既定席では同一 seed でも稀に出力が変わりうる」と limitations に明記〈2026-09-26 に撤回 — 既定席の M=1 / M=4 同一を門で固定・ADR 0096 追記〉・`speculative: boolean | "always"`・前段として受理列挙を停止 token で止める仕様変更を別コミット〉→ ⑧ M2 の小 M linear〈**測定済み・否定** 2026-09-09: 静的ノブ `linearGemvRowsThreadTarget`（`87e702b`）で r1 / r2 / r4 を M2 で比べると既定 r1 が最良（verify 90.7 → 103.6 ms）・「重み 4 回読み」仮説は外れ・backend 別既定は不要・K-22 は機序の宿題として残す〉→ ⑤ は縮小〈③′ は既に M=4 で効いており M=1 限定は ①QK だけ・①′ の WGSL は既に M 一般なので適用条件 1 行 + テスト 3 本 + kernel doc の MUST・効き代は M2 で最大 −4 ms/cycle・非負なら残す〉→ ③ argmax → ⑥ k=7 + 3 値ゲート・⑦ 不採用）**。**⑤ ✅ / ③ ✅ 2026-09-09**（`06f82df` ①′ を M ≤ 8 へ〈RTX 中立・既定席で verify 行 0 が u32 一致〉/ `e370562` 2 相 argmax〈drafter の argmax 2.03 → 0.08 ms/run・ビット同一〉— [research §6.2 / §6.3](research/2026-09-09-mtp-stage4.md)）。**M2 検収 ✅ 2026-09-09**（抽出 auto 1.379× / 対話 0.966× / 自由文 0.874× — ゲートは負けを半分に・0.99× には届かず → 残件 = ゲート v3〈強い信号の早抜け・バースト初期間隔 16・bench の sequence 使い回し〉）。**残り = ⑥（裁定待ち: parked か 2 drafter + 3 値ゲート）・ゲート v3・⑩・4-C docs**。**⑨ ✅ 2026-09-09**（`0c8bacd` per-cycle トレース = `onRun` の壁・確定数・ゲート状態 + mtp-bench の局面別バケット・[research §6.4](research/2026-09-09-mtp-stage4.md): RTX では勝つ側は抜けず W1 プローブ 1〜3% のみ・cold は 1%・**ゲートが落とす plain step は長文脈で真の decode より +17〜30% 高い〈verify 形・物理行は 1 で原因は未帰属〉→ 新規 ⑩ = 原因の帰属〈gpu-timing をモード別に割る〉と対策**・M2 の trace〈[research §6.4.1](research/2026-09-09-mtp-stage4.md)〉: 自由文の損の 62% は探索バースト・36% は抜けるまで・対話は 6% 負けを 63 cycle 続ける費用・v3 = 早抜け + 間隔 16 で ≈ 0.94×・warm start で ≈ 0.97×）。**ゲート v3 ✅ 2026-09-10**（`29e74b0` / `15b8111`・[research §6.5](research/2026-09-09-mtp-stage4.md)・ADR 0096 追記: RTX cold は勝つ側で誤退出 0・W1 半減・M2 の v3（[research §6.5.1](research/2026-09-09-mtp-stage4.md)）: cold 自由文 0.934× / 対話 0.959× / 抽出 1.361×・**warm 自由文 0.978× / 対話 0.972×**（残りは 1 ターン 1 回のバースト → exploreMax 512 級で 0.99× 圏）。warm の口は `746ad77` でターンごとに違う発話に直し写しは消えた（research §6.5 表 11′）。**新知見: warm の対話（勝つ課題）で 3 / 7 ターンが抜けて auto / always 0.96 → A/B 済み（`063c874`・research §6.5 表 11″）: 帰属は v3 の 1 ブロック早抜け → **v3.1 ✅ 2026-09-10**（`a28e56f`・`earlyLeave` 既定 off / `burstAbort` 0.15・RTX warm 対話 auto = always・[research §6.5.2](research/2026-09-09-mtp-stage4.md)）。exploreMax 256 → 512（`e365d9a`）。M2 warm v3.1（exploreMax 256 ビルド）= 自由文 0.990× / 対話 0.981× で、512 の検収は M2 warm の再計測待ち。⑩ は host / readback 側と判明・帰属は run 内の相の壁が要る）。旧記述（⑤ の −6 ms / u32 同一は反証で撤回）= ③ argmax の 2 相化（drafter −1.5 ms/cycle・ビット同一が構造保証）④ 投機の on / off ゲート（k′ ∈ {0, k} を受理の移動平均で・`accepted` の切り詰め前加算の誤りも修正）⑤ ①′ / ③′ の行タイル化（verify M ≤ 8 を decode と同じレーン割り → attention −6 ms + 既定席で u32 同一・着手条件 = 4-A で attention ≥ 5 ms/cycle）⑥ k=7 drafter の再 export + 動的 k（chat 形式の E[a] で予測倍率が k=3 の 1.2× 以上のとき）⑦ GPU argmax / topk 出口（readback + ホスト受理が cycle の 10% 以上のとき）→ 4-C docs（ADR 0096 の段表 段 4 行を現状へ消し込む — 採算表と「fence 1 本化は不成立」は ADR 追記と research §5 に記録済み）。裁定: ブラウザ実測はリリース後・M2 のミラーはユーザーが scp。事前プローブ（RTX・P≈4.8K・k=3・greedy）: decode 29.0 ms/run〈GPU 25.2〉・verify 37.0〈GPU 35.4 = decode + attention ①/③′ +6.0 + linear +2.2 + lm_head +0.8〉・draft 15.7〈GPU 6.5・argmax 1.8〉→ cycle 52.7 ms / 2.15 token = 24.5 ms/token = **1.18×**（decode 相・ブラウザ相当 ≈1.3×）。旧記述: 実測 Deno / Metal・温度 1.0 での受理率・投機を張る文脈長の閾値・動的 k・GPU argmax・
  fence 削減・①′ の位置不変化）。
- **`planBackingBudgetBytes` を共通の options へ**（起票 2026-09-07 — ADR 0095 帰結）: gemma4 以外は manifest の `session` から
  Session options を組むため予算を変える口が無い（既定 256 MiB が効く）。`onRetry` を `FromPretrainedHubOptions` へ 1 本化した形に
  倣って載せる。併せて estimate / session-build に二重にある予算の値域検査を 1 関数へ寄せる。
- **GEMV 行ブロックの残件（起票 2026-09-07・K-21 の帰結）**: ①並列度の目標 16384 は RTX 3080 Ti の飽和点（limitations）—
  M2 の再掃引は済（既定 16384 が最良）で、残るのは内蔵 GPU の掃引。差し替え口は静的な Session オプション
  `linearGemvRowsThreadTarget` ②ブラウザ（Chrome / Tint）のシェーダ解析費は未測（Deno / naga で初回ターン
  +85 ms）③Metal の u32 門は行ブロック 13 形も未実測（known-issues）。
- **Anima: 常駐の値に `text+transformer` を足す（起票 2026-10-01・利用者）**: 今の `transformer` は DiT の Session だけを generate を跨いで持ち、
  text_encoder / text_conditioner（重み 1,396 MiB）は毎回「読む → 上げる → Session を組む → 畳む」をやり直す（既定の VRAM の前提が「最大の段 1 本ぶん」—
  ADR [0112](decisions/0112-anima-transformer-residency.md)）。PyTorch / ONNX Runtime のデファクトは「載せたら明示的に消すまで常駐」で、
  利用者の方針は「API の形は明示解放・既定は VRAM が許す限り常駐・入らなければ自動で格下げ」。text の 2 Session を既存の退避の
  状態機械（次の段の試し確保 → 入らなければ退避 → やり直し）に乗せる。既定は据え置き。効き代は環境次第（M2 ローカル配信では
  text 経路の読み直し ≈ 1.7 s / 枚・取得が遅い環境ほど効く）。着手前に B570 / M2 / Pixel で常駐ぶんを足した VRAM のピークを測る。
- **Anima: DiT stage 内だけの反復常駐**（起票 2026-09-07 — Codex 性能調査 04 §Anima）: Session を stage ごとに作って返す
  現設計（VRAM の不変条件）を保ったまま、DiT stage の中で初期 latent の patchify を 1 度にし、RoPE / cond・uncond embedding /
  timestep 材料を反復間で再利用し、DiT 出力 → CFG → Euler / DPM++2M 更新を同じ token layout で回し、最後だけ unpatchify する。
  `copyLatents` / onEvent / abort の応答性は公開面の契約なので削らない（数 step ごとに finish する境界も測る）。
  **固定入力だけの常駐化は測定済み・利得なし**（2026-09-10、1024px / CFG1 / Euler8、全 step / PNG 一致）。
  latent / scheduler を含む反復常駐と CFG>1 は未検証（[実測](research/2026-09-10-codex-mtp-optimization.md#ホスト待ちと既存融合の追加測定)）。
- **slot backing をバケット run の前に退役させるか**（起票 2026-09-07・limitations「prefill バケット」）: 末尾 chunk の
  バケット run（ミス run）は chunkLength 形の backing が載ったまま arena に一時を確保し、非勘定側の窓に
  「ミス run の arena 一時（最大でバケット形 1 本ぶん）」が乗る（既定の梯子なら 768 形 + 256 形 ≈ 1.33 倍）。
  generation のミス run で活性 backing を先に退役させれば窓は消えるが、次の 768 chunk で作り直しが
  1 回増える。4 GB 級端末で効くかを見てから裁定。
- **テスト被覆の残（起票 2026-09-05）**: `SubmitScheduler` の `#encodeTimedChunk` 内の copy 分岐
  （`packages/runtime/src/gpu/submit.ts`）は依然として未検証。
- **Metal `--diagnostics` の切り分け実験**: query set の同時生存本数と `destroy()` 滞留の
  どちらが支配かの A/B。手順①②と修正候補は [known-issues](known-issues.md) の該当節が正本。
  実機が要るのでユーザー実行。
- **差分レビューの見送り表の中優先 3 件**（正本 = `.claude/reviews/2026-09-03_7fc4ada/ROADMAP.md`
  — git 追跡外）: ①W-G5-7 opbench / fusion-hints の資産解決を `tools/_shared/assets.ts` へ統合
  ②W-G4-4 chunk 上限の出所を provenance の `sym_max` 欄へ（**再 export 同乗** — 波 b や系列更新
  の回に）③ADR [0033](decisions/0033-vae-fixed-tile-decode.md) 決定 5「TS 側が幾何そのものを
  突合する」経路の不在（幾何 JSON を 1 本吐くか、決定 5 を実態へ追記するかの裁定）。
- **モデルカードのピーク VRAM 列（起票 2026-09-04）**: `karume dist` が TS 側の見積り
  （`prepareContainer(...).estimate()` 系）をカード生成時に呼び、quant 表へピーク VRAM 列を出す。現状の
  カードは格納バイトしか出さないので、読み手が自分の GPU で動くかを判断できない。
- **ChatSession の要約型 overflow ポリシー**: `onOverflow` は差し替え可能なのでポリシー実装
  1 本として入る。再検討条件「窓を広げた後」は ADR
  [0091](decisions/0091-gemma4-host-rope-variable-capacity.md)（capacity が実行時ノブ）で成立
  — ADR [0083](decisions/0083-generation-api-surface.md) 追記の見送り記述の行き先はここ。
- **HF CDN の同時本数の実測**: 接続ごとの上限が実測されたら、DL 並列本数の定数引き上げか末尾
  向け part 細分化を再起票する（DL スロット改善自体は kill —
  [research 2026-09-02](research/2026-09-02-cold-load-dl-timeline.md)）。
- **GPTQ 掃引の再評価**: 既定は現状維持で確定・opt-in 実装は温存（正本 =
  [research 2026-08-31](research/2026-08-31-gptq-axes-sweep.md)）。復活条件 = **多モデル ×
  校正量 16×** での再評価（gemma4 校正 rig の新設もそこまで保留）。
- **norm の 1/dim ホスト化は保留**（実 GPU プローブが先・費用対効果低。reduce identity の
  params −inf 化は現状維持 = W-2/W-3 と同じ器でセット裁定）／ **tolerance B 案**（`allclose`
  へ縮約スケール項を入れる公開 API 変更 — A 案の op 別表は実装済み。
  [research 2026-08-31](research/2026-08-31-op-tolerance-measurement.md) §8.2）。
- **minicpm5 の `export_decode` は RoPE 表を焼いたまま**（gemma4 だけが ADR
  [0091](decisions/0091-gemma4-host-rope-variable-capacity.md) でホスト供給へ移った非対称）。
  ホスト供給へ揃えるかは別裁定。
- **メモリ管理波の隣接起票**（正本 = ADR [0089](decisions/0089-memory-limits-preflight.md)
  Consequences / ADR [0090](decisions/0090-shard-spec-v3-tensor-pieces.md)）:
  `GpuContext.createResident` の確保は errorScope 頼みのまま（run 時 transient は計画時の
  preflight で確保前に落ちる）/ `fromAssets` の位置づけ / large asset の
  reference-first 一般則。

**decode 速度の残り**（H-27 段 ② / 小物 K-48・K-49・K-50 / K-46）は later の同名項が正本。

**ユーザー実機（Claude からは実行できない）**:

- Chrome での HF 経路の RAM ピーク追試（Deno 側は実測済み —
  [research 2026-09-24](research/2026-09-24-part-length-ram-peak.md)）。
  **性能のブラウザ計測はこの波（a）から外す（2026-09-04 ユーザー裁定）** — ブラウザで採れるのは
  Dawn / wgpu の実装差だけで、カーネル候補の採否には効かない。代替 = TS パッケージ側に性能情報を
  収集する機能を足し、ユーザーが複数環境でサンプル集（名称・置き場は未定）を回した結果を集める
  （**起票のみ** — 収集する項目・置き場・オプトインの形は未設計）。
- Pixel（8GB 級 Android Chrome）の `err.cause` 再判定 — [known-issues](known-issues.md)。

## later

- **E4B 通常（gemma4）の export（裁定 4 = b・2026-09-26）**: RAM 48 GB 以上の機で export する。recipe（`--model e4b`・
  読み込みの meta 構築 + assign）は現状のまま（31 GiB 機では OOM する）。E4B 通常の席は E2B の束を
  暫定宣言・既定 `i4`。

- **0.13.0 の再アップロードで断片化した part 5 本（起票 2026-09-24・2026-09-24 裁定 = そのまま公開し、実 DL 速度を測ってから対処）**: `hf-upload.zsh` の断片化表で
  255 MiB 級の part のうち anima 2 本（`anima-turbo-v1.1` / `anima-v1.0` の `text_conditioner` part 3・3.6 MiB/term）・
  birefnet-hr 1 本（`2048/matte` part 3・4.1）・lucida 2 本（`2048/matte` part 3 / 6・0.9 / 8.8）が 10 MiB/term を下回った
  （他の 620 本超は健全・extra は 0 本）。**推測**: 旧 safetensors と同一バイトのチャンクがリポ内の既存 xorb へ重複排除された
  形（global dedup は台本で停止済み）。公開 pin のあるリポは削除 → 再作成を使わない（runbook §2 MUST NOT）ので、
  DL 速度の実測で許容するか、別の対処（次の上げ直しで byte が変わる回に自然解消するか）を決める。

- **越境参照をツール側で扱いやすくする（ユーザー起票 2026-09-24）**: recipe が越境コンポーネント参照
  （今の `dist.py` の `--ref-*` 5 指定 — ADR [0038](decisions/0038-manifest-v1.md) §7 追記）を宣言的に持てる形と、
  export 用ツールが依存関係（参照先リポの main の commit SHA）を解決して参照先から順にアップロードする機能。
  動機は 2 つ: `irodori-v4-small` と `irodori-v4.1-small` の重複（83 ファイルが同一 sha256・5,605 MiB）を
  越境参照で消すには参照の宣言が要ること、`anima` → `anima-extra` の公開が参照先の SHA 待ちの直列で
  手作業なこと（[release-runbook](release-runbook.md) §0）。設計は未着手。

- **manifest（`karume.json`）の構造の見直し — 最適化フラグとプリセットを扱いやすくする（ユーザー起票 2026-10-06）**: 今は実行ノブの束を
  quant 席の `session` 1 つで持ち、席名が束の名前を兼ねる（ADR [0074](decisions/0074-quant-seat-naming.md) /
  [0110](decisions/0110-practical-tier-numerics-contract.md) / [0111](decisions/0111-session-options-composition.md)）。
  最適化フラグとプリセット（束）を扱いやすい形へ変えたい。何がどこに属するか（格納型・実行ノブ・束・部品ごとの宣言など）は
  後で整理する。材料は高速パスの表し方の棚卸し（2026-10-06・`.claude/reviews/2026-10-06_fast-path-inventory/`・git 追跡外）の
  未決 — 部品ごとの session が無い（VAE / text encoder の高速パスを宣言できない）・新しいカーネル変種のキー・束を変えるとき
  既存の席を変えるか新しい席を作るか。**着手は後**（利用者 2026-10-06 — Wan2.2 の最適化は今の `session` の範囲で進める）。
- **irodori v4-small と v4.1-small の重複 5,605 MiB を越境参照で消す（起票 2026-09-24）**: 2 リポの 83 ファイルが
  同一 sha256。0.13.0 の再アップロードでは見送り、次の breaking 波で判断する（2026-09-24 ユーザー裁定）。
  irodori 2 リポの再アップロード（release 節 — `dit_context` の追加）と同じ回に行う。
  上の「越境参照をツール側で扱いやすくする」が前提。
- **HTTP Range 取得（ADR 0108 段 6）の前倒し候補（起票 2026-09-24・判断はリリース後）**: ADR
  [0109](decisions/0109-manifest-v5-container.md) 決定 7 の前倒し条件は「段 2 の RAM ピーク harness で cold の
  ピークが『part 長 + 重ね合わせ』を超える」こと。段 3e で保持の重複は消えたが、scan 型（Deno の HF 経由）の
  cold はまだ part 長 + 最大 block を超える。残りは hub の保持枠 1 本と GC を待つ part で、原因は part 単位の
  全量読み（取得の粒度）に移った。条件は形式上成り立ったまま（[研究記録](research/2026-09-24-part-length-ram-peak.md)
  の 7・ADR 0108 追記 5）。当たるのは Deno の HF 経由だけ。parked の「hub Range 並列 + prefetch」（断片化対策）とは
  動機が別。
- **未検証の取得元（ローカル）で host PLE の行読みが block 全体の sha256 を毎回掛ける（起票 2026-09-24）**:
  host PLE は行を読むたびに読み口を開き直すので、未検証の取得元では 1 行の読みが毎回 block 1 本ぶんの読みと
  sha256 になる（1 run で 28 回 × 32 MiB）。段 3e の M2 で gemma4 の run が伸びた件はこれとメモリ確保の費用に
  帰属済み（[研究記録](research/2026-09-24-prerelease-gpu-measurements.md) の 1）。直し方は、開いた容器が
  検証済みの block を覚えて 2 回目以降を区間読みにする（検証済みキャッシュ）か、block の digest を 1 回にする形。
  local のファイルが読みの間に差し替えられる形（TOCTOU）の扱いを ADR で決めるのが先。
- **PLE のメモリ内容器のフェンス本数（起票 2026-09-24）**: GPU 常駐席の PLE は piece 1 本 = part 1 本で
  メモリ内容器へ渡すので、Session 構築のフェンスが piece の本数ぶん立つ
  （`packages/runtime/src/format/container/memory.ts` の part 割り）。実測済み（QAT E2B / E4B —
  [研究記録](research/2026-09-24-prerelease-gpu-measurements.md) の 2）: piece を part へ束ねると E2B でフェンスが
  37 → 6 本になり、PLE 構築は約 −0.2 s・代償は構築時の VRAM +192 MiB。**2026-09-24 裁定 = 当面束ねない**
  （0.13.0 のリリース前レビューで再確認・実装の規模は研究記録の 2 の結論）。
  代替案（2026-09-25 起票・未着手）: 束ねずにフェンスを **1 本先行**させる（次の piece の供給を前のフェンスの完了前に始める）。
  staging は同時に 2 part ぶん生きるので、先行を許すのは合計バイトが上限以下のときに限る（ADR 0108 決定 9 の
  「本数と合計バイトの両方で制限」）。重みの part 256 MiB では +256 MiB になり束ねる案の +192 MiB より大きいので、
  対象は PLE 席（32 MiB の piece）だけ。kill: PLE 構築 −0.3 s 未満（候補の採否は [perf-ledger](perf-ledger.md) の L-16 と併せて見る）。
- **hub ≤ 0.4 の `karume/1` 名前空間の回収グルーの撤去条件（起票 2026-09-25・0.13.0 は据え置き）**:
  `packages/hub/src/cache.ts` の `purgeLegacyCaches` は `loadManifest` の入口で取得元に関わらず毎回走り、
  `clearHubCache` も旧名前空間を回収する。撤去条件は ADR 0080 にも backlog にも無い。決めること: ① 撤去の時点
  （旧版からの移行期間をどこで閉じるか）② 列挙の失敗を op `"delete"` で通知している点を `"open"` に寄せるか。
- **`gemma4_qat` の fixture テスト 25 本が CI では丸ごと SKIP（起票 2026-09-25）**: `tools/export-recipes/gemma4_qat/tests/`
  の `series_fixture.py` と `test_fixed.py` が module 直下の `pytest.importorskip("transformers")` を持ち、`transformers` は
  `gemma4-qat` group にしか無いので、既定の sync で回る CI の recipes ジョブでは test_audit 1・test_fixed 12・
  test_plan 12 の 25 本が収集ごと SKIP される。決めること: a) `transformers` をスタブ化し上流突合を 1〜2 本残す
  （`checkpoint.py` の `isinstance(QuantizedEmbedding)` に注意）か、b) CI に `--group gemma4-qat` を足すか
  （CI 時間と上流突合の強さの交換）。
- **container-v1 §6.2 の codec 台帳と実装のずれ（起票 2026-09-24）**: 仕様の台帳エントリは `decodeCpu` /
  `executableOps` / `wgsl` を持つが、実装の `CodecEntry`（`packages/runtime/src/format/container/codecs.ts`）は
  `layout` / `packing` / `scale` / `grouping` / `zeroPoint` だけで、圧縮のまま常駐できる op の判定は今も別々の
  述語（`packages/runtime/src/runtime/plan.ts`）。仕様を実装へ寄せるか、実装を台帳へ畳むかを決める。
  同じ主張は 3 文書にある — container-v1 §6.2（「この 3 分離は現行が既にそうなっている」を含む）・
  [ir-v2](ir-v2.md)「値と型」の適格判定 bullet（「適格 op は台帳の `executableOps` が正本」）・
  [glossary](glossary.md) の「codec 台帳」行（「3 軸を分けて報告する」）。裁定後は 3 文書を同時に直す。
- **depth / birefnet の実資産 e2e が結果記録を包みの外で呼ぶ（起票 2026-09-24）**:
  `packages/models/tests/e2e_depth_anything_real_test.ts` と `e2e_birefnet_real_test.ts` は catch で
  `results.record` を直接呼ぶので、記録が I/O で落ちると元の検証例外が置き換わる（`runRecordedCase` /
  `recordFailure` の形に揃えると防げる）。

- **コード品質管理の波の残置（起票 2026-09-22 — 出典は
  [退避した消化済み節](research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md)〈0.12.0 リリース後〉の同波）**:
  - `Gemma4PipelineOptions` を引数に取る 4 本（`assertSpeculative` / `resolveGemma4PleResidency` /
    `buildGemma4Program` / `speculativeSetup`）の置き場（`pipeline-options.ts` の新設は 2026-09-21 裁定で後回し）。
  - Unicode 区間表の検査は文言が違い各 family に残っている（二分探索だけ `text/code-ranges.ts` へ寄せた）。
  - `reference/ops.ts` と `ops/shapes.ts` の分割（info — 可変状態も循環も無く実害ゼロ）。

- **decode 速度の残り（2026-09-20 に now から移動）**: H-27 段 ②（先行投入・ADR 0066 の opt-in 例外・greedy 限定・期待 Deno −5 / Chrome −2.2 ms）、
  小物 K-48 段 1（rms_norm→SRQ 融合 70 本・0.19 ms）/ K-49（slice 別名化）/ K-50（k+v 連結 GEMV）、K-46（int8 KV — メモリ項目）。
  復活条件: decode 速度を再び主題にするとき（H-27 の前提 = H-28 の GPU 常駐席は済・research 2026-09-19 §15.2 の TTFT 帰属の再測定を先に
  — 「prefill のアリーナ経路」はヒット run には無いと 2026-09-25 の追記で訂正済み）。
  候補の採否と kill 基準は [perf-ledger](perf-ledger.md)、帰属は [research 2026-09-19](research/2026-09-19-qat-speed-recon.md)。

- **層内の大融合を塞ぐ 3 契約の裁定（起票 2026-09-19）**: WebML との dispatch 差（1,132 → 約 316 本）のうち約 480 本は
  `windowTouchesState` MUST（ADR 0067）・FusedStep 単一出力 MUST（ADR 0068 決定 1）・atomic last-arriver merge の可搬性判定が同時に塞ぐ。
  個別候補（attention→SRQ・残差 add→SRQ・attention 1 dispatch・Q/K/V 3 出力）は全て反証で閉じたので、契約を緩めるかの設計裁定が先
  （[research](research/2026-09-19-qat-speed-recon.md) §9.1）。裁定前に速度候補として起票しない。
- **ブラウザ動画生成の基盤**（調査 2026-09-10・**利用者要望 2026-10-02: まず Wan2.1 T2V 1.3B が動くまで・最終目標は
  MiniMax H3**）: Wan2.1-T2V-1.3B を小さな DiT 単体 → 実 token 長の attention / FFN → causal Conv3d VAE → scheduler と
  段寿命の順に検収する案。runtime 語彙と数値契約の判断が先。
  [構成と容量試算](research/2026-09-10-codex-mtp-optimization.md#動画生成の事前調査-wan-と-minimax-h3)。
  **本調査済（2026-10-02・[research](research/2026-10-02-video-gen-recon.md)・**裁定 2026-10-02・全て推奨案**: H3 は構造調査 + 部品単位の候補に留める / conv3d は IR に op を足す〈implicit GEMM・ADR 起草〉/ テキストエンコーダは段階化〈事前計算した埋め込み → GPU の i8・活性 f32〉/ 原版 50 ステップ / 最初の到達目標 = 832×480・33 フレーム・Deno で bring-up〈利用者: 公開値 8.19 GB は offload + t5_cpu の値で、埋め込みを事前計算すれば手元の B570 でも入るはず〉→ 81 フレーム → Chrome。**ADR [0118](decisions/0118-wan21-video-generation.md) accepted（2026-10-02・裁定 4 点も推奨案 = VAE 常時タイル 32 / 既定 shift 3.0 / 固定プロンプトは公式の例文 / リポ名 karume-wan2.1）。段 0 ✅〈B570 の天井 9,600 MiB・`327818f0`〉・段 1 ✅〈conv3d op〉・段 2 / 4 ✅〈`4af318e6`・DiT S 形 export + 比の帯 9.6e-5・VAE chunk グラフ 2 種 + cache 正規化ビット一致・レーン test:models:wan〉。段 5 ✅〈タイル decode・12 枚・縮退門 Uint32・33 フレーム 128 s〉・段 3 ✅〈S=14,040 完走・1 submit 最大 270 ms・VRAM 5.15 GiB・GPU 16.75 s・f64 参照 + 正規化比 r で帯 75・受入れ r 2.9 / 4.6・帰属 = GEMM の K 縮約の精度差〈K-75 は現状維持の裁定〉〉。段 6 ✅〈`3de7d277`・@karume/models/wan・2 ステップ e2e sha 行・50 ステップ 1,811 s 完走・PNG 33 枚 = outputs/verify/deno-intel-graphics-bmg-g21/2026-10-02_wan-pipeline-full/〉。段 7 ✅〈`67c8d1d2`・配布形 models/karume-wan2.1・fromPretrained・門番・モデルカード・WAN_SOURCES は公開時・フル verify 3,547 passed〈赤は既知 OOM 門 1 + 受理表の登録漏れ 2 → 修正〉・利用者の目視 12 本 OK〈2026-10-03・意図どおり・継ぎ目なし → K-75 は現状維持で確定〉〉。段 8 ✅〈2026-10-03・81 フレーム = S 32,760・上限 81 フレーム〈既定 33・今は `WAN21_GENERATION.maxFrames`〉・DiT 実寸 8 ケース〈f64 参照・帯 104 = 最悪 r 20.8 × 5・受入れ 13.1 / 4.89〉・行ブロック 24・1 submit 最大 459 ms・VAE 21 chunk long 比 2.98e-6・81f 50 ステップ 7,173 s / VRAM 7.31 GiB・sha 行・カードの Resources に 81f の表〉→ 段 9（Chrome）。**隣接の小物（起票 2026-10-03）**: safetensors 0.8.0 の `__metadata__` はヘッダのキー順がプロセスごとに変わり、fixture の再生成がバイト一致しない〈vae_chunks.band / accept で実例・テンソルは同一〉— 再生成の検査はテンソルの sha で見るか、メタデータを 1 キーに畳む。VAE 前の解放待ちの設計は不要で閉じた〈DiT 段と VAE 段は重ならない — 段 6 の実測〉。**隣接の小物（起票 2026-10-03）**: 参照門 `referenceGatePasses`〈runtime/tests/helpers/reference.ts〉は caseIds のどれか 1 本に行があれば緑で、新しい case id の行の書き忘れが警告付き SKIP で通る — 「全 caseIds に行があるか、無い id は明示の held」へ締める。**✅ 2026-10-03 `7dc17309`（ADR 0106 追記）**。**隣接の小物（起票 2026-10-03）**: packages/models の diffusers 移植（wan の UniPC / blend・anima の sampler 等）に third-party notice が無い〈既存の状態・JSR 公開物〉。 **Wan の小物（起票 2026-10-03・ADR 0118 波の差分レビュー — 括弧の ID は `.claude/reviews/2026-10-03_diff-review-wan/` と `2026-10-03_codex-review-wan/` の項目〈git 追跡外・判断に要る事実は各項に写した〉）**: ① 通常実行の submit 構成での 1 submit の GPU 時間の計測（M8）〈「1 submit ≤ 1 s」の門と記録値は計測モード〈`gpuTiming`・1 dispatch = 1 pass〉の値で、裏付け後のチャンクの切れ目は通常実行と異なりうる — ADR 0118 段 8 の結果「submit の門の範囲」・GPU が要る〉 ② Wan の `signal`（AbortSignal）対応（P-4）〈他の系列は持つ・取得層と段の境目へ渡す anima の形が前例・未対応であることは limitations の Wan 節に記録〉 ③ golden `conv3d_block` に v4・m タイル 64 の変種を踏む 4 本目（F4）〈今の 3 本は Wout 5 / 5 / 6・Cout 6 / 3 / 2 でスカラ変種と m 32 だけ。v4 と m 64 は GPU の parity テストだけが踏む・golden の再生成と GPU での確認が要る〉 ④ conv1d / conv2d の WGSL の座標演算（dilation・padding を i32 座標へ変換する式）の境界監査〈Codex レビューの範囲外の気づき・再現も断定もしていない〉 ⑤ CPU だけの DiT ホストテストが 2.6 GB の参照を全部読み写す（TG-11）〈`reference.full-{band,accept}-s14040-t0999` は各 2,598,638,524 B・`readBuffer` の `ArrayBuffer.slice` で一時的に約 2 倍・RSS は未計測〉 ⑥ results.json の note の区切り「 / 」がラベルの中の「 / 」と衝突する（TG-18）〈`formatNormalized` などのラベル全般・split する消費者は今は無い・直すなら区切りの側〉。 ⑦ `prompt_clean` の entity 候補の門を名前表（Python の `html.entities.html5`・2,231 個）で「実際に文字列が変わるときだけ拒む」正確な門へ狭める（`R&D` のような entity にならない並びを受ける — ADR 0119 裁定 2026-10-03・利用者「回避できるなら回避策を」・要望が出てから）。perf 起票 K-72〈clearBuffer 口〉/ K-73〈adaLN 融合ルールの reshape 条件〉/ K-74〈timesteps_proj の周波数表〉〈2026-10-03: K-72 / K-74 は kill 条件が成立して ❌・K-73 は判定に要る adaLN の割合が未記録で 🚧〉**）**: DiT 側は語彙が
  ほぼ足り recipe のパッチ（complex 形 RoPE の実数化・unpatchify の rank 下げ）で済む見込み。空白 = 動画 VAE の conv3d
  （案 A: IR に op を足す〈拡張分子層 + ADR・src 約 1,000 行規模（推測）〉/ 案 B: recipe で conv2d に分解）と feat_cache の
  持ち越し（1 チャンク分のグラフ × 2 種類の export 案）・upsample3d の時間インターリーブ（rank 6）・umT5-XXL 5.68B の載せ方
  （i8 / i4 / f16 / 事前計算した埋め込み / CPU）・32,760 トークンの全結合 attention（1 forward 約 283 TFLOP・50 ステップ × CFG
  で約 28.3 PFLOP）と TDR。H3 は実在・open-weight（HF MiniMaxAI/MiniMax-H3・条文の文理では日本は許諾地域〈法的助言ではない〉）
  だが 33B dense + Qwen3-VL-32B + VAE 2.6B で 1 タスク約 144 GB・ネイティブでも offload 前提（RTX 5090 で 112 s の実測例）・
  ブラウザ WebGPU の事例は無い。
  **レビューの消化済（2026-10-03）**: 差分レビュー 75 件 + Codex 12 件を fix 単位 37 コミットで消化（ADR 0118 追記 2026-10-03）。
  **Wan2.1 の後続（2026-10-03 起草・0119 は proposed / 0120 は accepted）**: [ADR 0119](decisions/0119-wan-umt5-gpu-text-encoder.md)（段 10 = umT5 を GPU で・自由な
  プロンプト・AbortSignal・10a〜10e）/ [ADR 0120](decisions/0120-wan-dit-w8a8-seat.md)（w8a8 席・段 1〜7）/ 段 9 Chrome（gpu-lab の
  Wan ページを作り、利用者が RTX 5070 Ti 機で確認）。順序: 段 9 ∥ 10a → 10b（export のホスト RAM・単独で回す）→ 0120 段 1〜2
  （CPU）→ GPU の段は lock で直列化（10c → 0120 段 3〜5 → 10d）→ 視認 A/B（0120 段 6・10d）→ H3。**進捗（2026-10-03 夕）**: 段 9 = ページ完成・利用者の RTX 機での実走待ち / 10a ✅ / 0120 段 1〜2 ✅（i8 系列・席 3 つ）/ 0120 段 3〜5 ✅〈帯は B570 の実測で確定・実用席は f16 の約半分の時間・step 1 relRMS は 33f 1.07e-1 / 81f 2.10e-1 で attention a8 が大半 → 段 6 の視認で裁定〉/ 10c の準備 ✅〈バケット表の parity・小模型で export 被覆・gelu は tanh 差し替え〉/ S=14,040 の i8 golden 採取中 → 10b ✅〈meta trace + 行の塊ごとの i8 + fixed_weights・core 変更なし・山 6.79 GiB・容器 5.69 GB〉→ 10c の CPU 側 ✅〈層逐次 f64 / f32 参照・golden 10 本・e2e は帯未導出〉→ 10d-1 ✅〈karume-umt5-xxl・Wan の越境参照・仮 SHA の門・ミラー 2 本〉・10d-2 ✅〈textEncoder の経路選択・text 段・signal・examples / gpu-lab〉→ 10d-3 ✅〈参照門の締め `7dc17309`〉→ 10c の門 ✅〈帯 79〉・10d-4 ✅〈gpu-text の sha 行 2 本・text 段 10.4 s・VRAM の山 6.30 GiB・残り +0.08 GiB〉→ 次は 10e〈任意〉 → 0120 段 6 の視認 A/B〈素材 12 本 ✅ 2026-10-03・利用者へ送付済・裁定待ち〉。
  **進捗（2026-10-04）**: 段 9 ✅（RTX 5070 Ti の Chrome で 81 フレーム 50 ステップが完走・`f16` 席・precomputed の経路 — ADR 0118 追記 2026-10-04。
  この環境キーの sha 行は足さない裁定）/ 0120 段 6 ✅ → **ADR 0120 accepted**（実用席 `f16+dit8-a8-attn8-s16` は今の束で確定・**既定席 = 実用席**・
  `f16` は明示指定の参照側の席で、既存の sha 行・帯・case id は `f16` を明示して保つ — ADR 0120 裁定 / 追記 2026-10-04）/ gpu-lab の Wan タブに
  quant の選択・`demo:wan` に `--quant`。配布形ミラー `models/karume-wan2.1` は更新済み（`karume.json` の `defaultQuant` と README だけ・容器はバイト同一・ホストの検査は緑）。
  ADR 0120 の 50 ステップ opt-in の実用席の sha 行 2 本 ✅（`d5b3daa4`・B570・実用席の 81 フレームは壁 3,703 s = DiT 段 3,383 s + VAE 段 319 s・
  DiT 段の VRAM の山 6.18 GiB — ADR 0120 追記「50 ステップの実用席の sha 行 2 本」）。残り: GPU の `test:models:wan`（ADR 0120 追記 2026-10-04「残りの作業」—
  GPU の直列キューの後）。**走行中（GPU の直列キュー・結果は ADR に追記）**: ADR 0119 の視認素材（umT5 の GPU 経路 + 実用席・自由プロンプト 3 本と
  固定 3 本 × seed 42〜45）。
  **Wan の小物（起票 2026-10-04）**: (a) Chrome での GPU 経路（umT5 を GPU で）と実用席の実走は未（gpu-lab の Wan タブの既定の経路は
  `precomputed` のまま — 既定を `gpu` へ移すのも同じ確認の後）(b) ✅ モデルカード生成（`tools/export-recipes/wan/card.py`）の「ブラウザは未確認」と
  実用席の 81 フレームの「not run」を実測へ直した（2026-10-04）— 配布形ミラーへの反映は次の焼き直しで (c) ✅ `tools/gpu-lab/README.md` の
  「Checking Wan in Chrome」節に RTX の結果を足した (d) anima / irodori / sbv2 の例の出力名は席を省くと `default` で、既定席が変わると同じ名前に上書きされる。
- **Wan2.2 TI2V-5B の受け入れ（次の波・起票 2026-10-04・ADR [0121](decisions/0121-wan22-ti2v-5b.md) accepted）**: 参照席 `f16+dit8` /
  実用席 `f16+dit8-a8-attn8-s16` の 2 席・I2V 対応の DiT グラフ 1 本・VAE 2.2 のタイル decode・段 0〜10。MiniMax H3 はこの後。
  **裁定（2026-10-04・利用者）**: 受理する解像度とフレーム数は開発機で回せる範囲で作って試す（1280×704 を 33 フレームまで / 832×480 を 81 フレームまで
  の配分は確認中 — 段 0 / 1 は配分に依らない）・f16 席は今は作らない・I2V の縦横比はまず公式の挙動（覆う側へリサイズして中央クロップ）。
  **裁定 1 の確定（2026-10-04）**: 2 つの配分を両方受理して試す（832×480 系を 81 フレームまで・1280×704 系を 33 フレームまで — ADR 0121 追記）。
  （→ 2026-10-05 に改訂: 受理は 1280×704 / 704×1280 × 4n+1 の 5〜49 フレームだけ — 下の「受理寸法の変更」）
  **進捗（2026-10-04）**: 段 0 ✅（取得口・5B の RAM 実測・層逐次の f64 / f32 参照 `59367d44`）/ 段 1 ✅（I2V 対応の DiT パッチ・5B の i8 系列
  `wan2.2-ti2v-5b-i8-dyn`・S = 192 の golden 10 本・`export_dit --check` `25b6962a`。golden の文脈は実プロンプトの埋め込み — 合成の乱数は f32 と f64 の差を
  増幅する）/ 段 2 ✅（参照席の r 門が緑・両席とも容量の閾値の内・121 フレームも載る — ADR 0121 追記「段 2 の結果」。受理を 121 フレームへ
  広げるかは段 6 の前に利用者の裁定 — 832×480 系を受理から外したので消えた）/ 段 3 = コードは着地（共有モジュールの切り出しと 2.1 の一般化 — Python `1e4d7cbb`・TS `f25907f5` /
  `e54e6dc6` / `cf321598`・ADR 0121 追記「段 3 の結果」）。**段 3 の GPU の合格判定**: 2.1 のレーンは 394 passed・1 failed（sha 行・帯・case id は全て一致・赤は既知のレーン内の OOM）、
  `KARUME_WAN_FULL_PIPELINE=1` の 50 ステップ 4 本は既存の sha 行と一致。**残り = レーンの再走で全緑**（レーン内の OOM = Deno の `GPUDevice.destroy()` が VRAM を返さない —
  known-issues。テストの GPU の取得口で GC を促す対症療法を入れた〈利用者の裁定 2026-10-05〉・緩和つきの再走は 395 passed・0 failed〈余裕は約 0.15 GiB — known-issues〉）。**段 3 は完了**（ADR 0121 追記「段 3 を閉じる」）。1.3B の f16 系列の照合
  （`export_dit --check`）は 69 ファイル一致。**段 4 ✅**（VAE 2.2 decoder の chunk グラフ 2 種・新しい op なし・GPU の chunk 列は B570 で帯 4.8e-5 の内・
  1.3B の VAE の生成物はバイト不変 — Python `0f1ff88a` / `734077ae` / `2763651b`・TS `3ab40a8d` / `08547483` / `b004af49` / `8166a18a`・ADR 0121 追記
  「段 4 の結果」。`src` に触っていないので 2.1 のレーンは再走せず、helper の移動は 2.1 の VAE chunk e2e の単独実走でビット同一）。
  **段 5 ✅**（タイル decode — patchify 空間のブレンド → 貼り合わせ → ホストの unpatchify・2.2 のタイル計画の凍結・要素数の検査を形の検査へ。
  B570 の GPU のタイル e2e は帯 5.0e-5 の内・ホストの unpatchify は上流と Uint32 一致・2.1 のタイル参照は作り直してテンソルとメタが一致 —
  Python `4436457f`・TS `567fb9f0` / `0d5fc54d` / `f91b5b39`・ADR 0121 追記「段 5 の結果」。2.1 の GPU レーンは段 6 の最終状態とまとめて 1 回 — 1 failed で未達・残りは段 6 と同じ 3080 Ti のレーン）。
  **受理寸法の変更（2026-10-05・利用者の裁定）**: Wan2.2 の受理は 1280×704 / 704×1280 × 4n+1 の 5〜49 フレームだけ（832×480 系は B570 で生成すると
  席と shift を変えても崩れた・公式の対応寸法の外）。既定の shift は 5.0。フレーム数の上限 49 は B570 で 2 席とも 50 ステップを完走した値
  （57 も完走したが実用席で決定 8 の目安 8.0 GiB の外・既定は 33 のまま仮置き）。公式の既定 121 フレームは段 8 の gpu-lab + RTX 5070 Ti で確かめてから —
  ADR 0121 追記「受理寸法を公式の 2 寸法へ」・`59557732`。（→ 2026-10-07 に上限を 121 へ — 下の段 10）
  **段 6 = コードと GPU の門は済み**（`WanTi2vPipeline` の T2V — D1 `59557732`・`8a3a99db`〈DiT の追加入力 2 本・`ditInputForm`〉・`0fb5c005`〈`family.ts`〉・`e0c50cea`〈class〉・
  `992ca738`〈helper と e2e〉・`24ffd87b`〈Python の 2 ステップの参照〉・`3df18612`〈帯〉・`7173c91d`〈上限 49〉・`128b511e`〈参照席の sha 行 6 本〉・ADR 0121 追記
  「段 6 の結果」）。2 ステップの通しは B570 で帯の内（受入れも内・故障注入 3 件は床〈帯の 2 倍〉の外）。50 ステップ × 1280×704 × 33 フレームは参照席 43 分 3 秒・
  実用席 23 分 19 秒で完走・非有限 0。実用席の 33 枚の PNG は生成スクリプトの出力とバイト同一（スクリプト ≡ 製品）。レーン（`128b511e`）は
  `test:models:wan-ti2v` 41 passed・0 failed、`test:models:wan` 403 passed・1 failed（既知のレーン内の OOM — 単独の再走は 2 passed・数値の門と
  sha 行は全て通った・known-issues）。残り = 3080 Ti での 2.1 / 2.2 のレーン（凍結コピー `128b511e` で行を書く走行 → HEAD で素の走行・両方 0 failed）。2.1 のレーンは B570 で
  2 回とも同じ step が OOM（原因 = ランナーの step の保持 — known-issues）。
  利用者の裁定（2026-10-05）: B570 では手当てせず、換装後の RTX 3080 Ti で通す。換装の後は、新しい GPU の sha 行を凍結コピー `128b511e` で
  `KARUME_REFERENCE=write` で作り（B570 の行は残す・HEAD で書くと `128b511e` 以後のコミットの退行が sha で捕まらないため — 裁定 2026-10-06）、以後はその行との一致を ADR の合格条件にする（B570 に戻すことを条件にしない）。段 4 以降の
  「2.1 のレーンを B570 で実走」の条件は RTX 3080 Ti の行へ読み替える（ADR 0121 追記（2026-10-06）「開発機の換装」）。換装は 2026-10-06 に完了
  （環境キー `deno-nvidia-geforce-rtx-3080-ti`）。
  段 4 / 5 / 6 の帯（B570 の実測）は **3080 Ti でもそのまま回す**（利用者の裁定 2026-10-06 — 帯の外に出たら実測値を添えて諮る・先に再導出はしない）。
  **段 8a ✅（2026-10-06）** = 配布形 `models/karume-wan2.2`（recipe を世代の表で引数化・`dist.py --pipeline wan-ti2v`・出所の門の分割・自リポ 6.714 GiB + umT5 の越境 5.296 GiB）と
  gpu-lab の Wan タブの世代の選択（`/models/wan22/`・121 フレームまでは `family.ts` の内部 API を広げた記述子で呼ぶ・製品の受理 49 は不変）・2.2 の e2e は配布形経由へ・
  配布門番に `karume-wan2.2` — ADR 0121 追記「段 8a の結果」。**既定席 = 実用席 ✅（2026-10-06）** — 実用席の 12 本（seed 42〜45 × 3 プロンプト・
  1280×704×33・50 step・3080 Ti・1 本 753〜792 s・`outputs/misc/wan22-visual-2026-10-06/`）を利用者が視認して破綻なし → `WAN22_DEFAULT_QUANT`・カードの文・テスト 3 か所を替えて組み直した。
  **8b**（`examples/wan` の TI2V の口・T2V を閉じる docs）は 5070 Ti の確認の後。**5070 Ti の Chrome ✅（2026-10-06）** — 実用席の 1280×704×121・50 step が完走（40.3 分・DiT の診断値 9.47 GiB・
  利用者の目視で DiT の段 約 10.5 GiB〈Windows 込み〉— ADR 0121 追記「RTX 5070 Ti の Chrome」）。121 の受理の広げ方は、この機で 121 を 1 本回してから決める（裁定 2026-10-06）。
  **3080 Ti のレーン ✅・段 5 / 6 ✅（2026-10-06）** — 凍結コピー `128b511e` で行を書き（`03a5a7ba`）、HEAD `b744a9fd` のフル verify で Wan の 2 レーンは全緑
  （8 failed は全て Wan 以外 — known-issues）。**この機でも 1280×704×121 が完走**（63.1 分・熱制限込み・DiT の段の山 11,361 / 12,288 MiB）。8b の example ✅（`b744a9fd`）
  — ADR 0121 追記「RTX 3080 Ti のレーンとフル verify」。
  **段 10 = 着地・開発機の行は未記入（2026-10-07・利用者の裁定「要判断はどちらも推奨案で OK」）** — Wan2.2 の受理のフレーム数の上限を 49 → 121（記述子・カードと受理集合の fixture・
  gpu-lab の上書きを消した・limitations に 10 GB 級〈B570〉は 121 フレーム非対応）。開発機の参照値は 2 ステップ × 121 フレームのケース
  `f16+dit8-2step-boxing-cats-seed42-1280x704-121f-shift5`（opt-in `KARUME_WAN_TI2V_121F=1` — VAE の段だけで約 12 分）で、**3080 Ti の行は未記入**。
  利用者の視認: 3080 Ti の 121 フレームは 5070 Ti のものと同じに見える — ADR 0121 追記「段 10 — 121 フレームの受理」。
  **次** → 8b の残り（「T2V を閉じる」docs）→ 段 7（実用席の自機 A/B の門〈S の組は未決 — 832×480 の値が使えなくなった〉・利用者の視認〈参照席自身の品質の裁定を兼ねる・seed 4 本以上・
  1280×704〉・既定席と既定の寸法 / フレーム数・50 ステップの sha 行〈参照席・実用席 — 段 6 では書いていない〉）→ 段 9（I2V）→ **一旦休憩**（利用者が触って使い方を見る — 2026-10-06）。
  **段 5 / 段 6 の隣接の小物（起票 2026-10-05）**:
  (a) タイル参照の読み口と補助関数（`readTileFixture`・`planMeta`・`compare`・`absMax`・`poisonFrames`・`roundUpTwoDigits`）が 3 本の e2e に重複 → 共有の test helper へ
  (b) recipe の `_peak_rss_gib` などの補助が 3 か所に重複 → `_shared` へ
  (c) 既存の DiT の入力 5 本に dtype の検査が無い（2.2 の追加 2 本だけが dtype を見る）
  (d) `e2e_wan_ti2v_dit_test.ts` の `bandCandidate` の有効数字 2 桁の切り上げが、浮動小数の誤差で 1 単位上へ切り上がりうる（通しの e2e `e2e_wan_ti2v_pipeline_test.ts` の `roundUpTwoDigits` は `toPrecision(12)` を挟んで直した。タイルの e2e `e2e_wan_ti2v_vae_tiles_test.ts` と段 4 の chunk の e2e `e2e_wan_ti2v_vae_chunks_test.ts` の `roundUpTwoDigits` は挟んでいない — (a) の共有の test helper へ寄せるときに一緒に直せる）
  (e) `pipelineConfig` の reader の文言に owner の接頭辞が無い（2.1 / 2.2 のどちらの class の失敗か文言で区別できない）
  (f) 2.1 と 2.2 の通しの e2e が同じ補助関数の写しを持つ（`readFixture`・`difference`・一覧画像など）
  (g) gpu の経路の admission（`admitWanText`）を 48 ch・7 入力の DiT と組み合わせたホストテストが無い（段 6 の結果）
  (h) `tools/export-recipes/wan/pipeline_ref.py` の `video_size` が 16 ch と圧縮 8 を固定で持つ（2.2 では使えない — 段 6 の結果）
  (i) `planWanRequest` が注入した `latents` の要素数だけを見て、形を見ない（段 6 の結果）
  **段 4 の隣接の小物（起票 2026-10-05・レビューで見送り）**:
  (a) `filePresent` を中立の test helper へ移す（今は VAE の helper `wan-ti2v-vae.ts` が DiT の helper `wan-ti2v-dit.ts` から import する — VAE のテストが DiT の helper の import に依る）
  (b) 共有 helper `wan-vae-chunk-loop.ts` の drop-cache の故障注入が、名指した cache がグラフに在ることを assert する（無い名前だと黙って故障なしになる — 2.1 と 2.2 の両レーンを回す）
  (c) DupUp3D の閉じた形（up0 / up1 を融合 upsample2x に乗せる）— perf-ledger K-77
  (d) 2.1 の VAE chunk e2e（`e2e_wan_vae_chunks_test.ts`）の「故障注入のループは故障なしなら製品の経路と Uint32 で一致」の step で、`decodeWithFault(none)` の前にもフレームを毒値で埋め直し、非有限 0 を assert する（2.2 の e2e は段 4 で直し済み — 直したら 2.1 の VAE chunk e2e を実走する）
  **段 3 の隣接の小物（起票 2026-10-04・提案）**:
  (a) ✅ gpu-lab の Wan タブの再生 fps と既定のフレーム数を記述子と `video.fps` から読む（段 8a・2026-10-06）
  (b) 実験用ミラー（umT5 の追加学習版）のモデルカードに「Wan へ差し替える使い方」の段落を足す
  (c) 手元の配布形ミラーのモデルカードと、recipe が今描くカードを比べる検査を足す（焼き直しの漏れを見つける）
  (d) Wan の text 段が umT5 の席の宣言した `session` を黙って無視せず検査する（known-issues の部品差し替えの項目と同根）
  (e) `e2e_wan_dit_test.ts` が、落ちた step でも VRAM の記録を出す（known-issues のレーン内の OOM を ① 残り / ② 他の process に分けるため）
  **後で足す（起票 2026-10-04・利用者）**: (a) TI2V-5B の f16 席 — 開発機で動かなくても重みは用意しておきたい（材料: f16 の export は RAM の境界か
  exporter core の変更が要る・開発機では門も sha 行も持てない — ADR 0121 決定 7 / 裁定 2）(b) I2V の縦横比を選べる口（直接リサイズ — diffusers の
  挙動 — を選べるように・口の形は段 9）。
  **関連（accepted・裁定 2026-10-04）**: ADR [0122](decisions/0122-umt5-upstream-and-compatible-encoders.md)（umT5 の出所を本家 google/umt5-xxl へ・
  互換の text encoder を作る経路・差し替えは今の components 席のまま・段 a〜d）。段 a〜d 完了（ADR 0122 の追記 4 本）。残り = ADR 0122「未解決」の 2 項目（第三者の互換部品の公開の裁定・
  版を跨いだグラフ記述の一致 — 追加学習版の視認は 2026-10-05 に利用者が問題なしと裁定）。
  **umT5 / Wan2.2 の小物（起票 2026-10-04）**: (a) 本家の索引（`pytorch_model.bin.index.json`）と `config.json` を sha256 で pin していない（索引が実物と
  食い違うと素の `KeyError` で落ちる — 値がすり替わる経路は無い・ADR 0122 追記「段 a の結果」）(b) umT5 の `write --check` が系列の中に一時ディレクトリを
  開く（`.check-*`）— プロセスが強制終了されると残る (c) `ContainerDitWeights`（容器から DiT の重みを読む層逐次の口）の読み込みが 1 forward あたり 25 s かかる。
- **anima 素版 i4 の品質改善（起票 2026-08-24 — 配布スキップ裁定の復活レバー）**: 残るのは
  turbo 側の i4 席で**未検証のまま残した可能性の一覧**（専用幾何・g16・校正量・もう 1 つの
  劣化機序 — いずれも「試してダメ」ではなく「試していない」）だけで、正本は
  [research/2026-08-21-anima-i4-seat-speed.md](research/2026-08-21-anima-i4-seat-speed.md) §8。
  g16 は parked「anima の g16 評価」が復活レバーの 1 本として残る。adaLN の i8 化は視認
  スイープで不採用が裁定済み（perf-ledger Q-9）、量子化感度の高い場所の特定は
  [退避した消化済み節](research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md)の
  「既知問題 3 件 + anima 素版 i4 感度」④ が正本で、同じ裁定が「anima DiT i4 系は
  しばらく保留」を付けている。校正済み系列は `outputs/series/` の `*-i4-dyn` に温存
  （実測記録の正本 =
  [research/2026-08-24-gptq-expansion-quality.md](research/2026-08-24-gptq-expansion-quality.md) §5）。
- **irodori adaLN i8 の出荷リグ A/B（起票 2026-08-24）**: sim で効いた adaLN i8（+13.1 MiB）が
  出荷リグでも読み上げ方を改善するかは未検証（sim → 出荷の転移限界 — 同 research §2）。
  復活 = `i8+dit4` 席（旧 `w4`）の品質不満、またはサイズ最適化の実需。
- **バレル・ファミリープレフィックスの見直し（起票 2026-08-25 — ユーザー意向「今後見直し
  たい」）**: `mod.ts` の全ファミリ平面 export のためにシンボルへ族名プレフィックスが付くが、
  SBV2 族では「2」が変換イディオム（x2y）に誤読される実害が出た（`sbv2Utterance` →
  `toSbv2Utterance` へ命名回避 — ADR
  [0079](decisions/0079-sbv2-two-layer-input.md) 決定 2）。サブパス面での素名 export や
  namespace オブジェクト化などの選択肢を全ファミリ横断で再設計する（プレフィックスが外れれば
  `toUtterance` へ収斂できる）。breaking なので次の breaking 波に同乗させる。
- **measure_quant の配布試算の J-5b 追随**（J-3 中に発見・2026-08-22）: sbv2
  `project_distribution` が「linear の重みスロットだけ・conv / embedding の i4 は格納形も
  実行経路も無い」という pre-J-5b 前提のまま（実際は出荷済み — ADR 0069 追記 6/7）。相対
  比較には無害だが試算が過小で docstring も陳腐化。対象集合と説明の追随を 1 件で。
- **モデル拡充の続き**: Kokoro-82M（exporter の多出力 `getitem` 結線 + `lstm_scan` 新設裁定
  〈ADR 0056 決定 8〉待ち — ノードレベル多出力自体は ADR 0068 で解禁済み）・MobileSAM / SAM 2
  （conv_transpose2d）・DA-V2 可変解像度（upsample_bicubic2d）。後 2 者は必要 op が語彙外。
  候補調査の時点記録は [recon-2](research/2026-08-14-model-expansion-recon-2.md)。
- **性能候補**: 起票・採否・順序は [perf-ledger](perf-ledger.md) の 🚧 行が正本。
- EmbeddingGemma の完成（models pipeline / 配布形・batch>1 export・runtime attention_mask 配線）。
  **tokenizer〈Gemma SPM BPE + byte_fallback〉の実装と EG 資産 compile は生成 API 波の段 1a に
  同乗済み**（2026-08-31 裁定 9 — ADR [0084](decisions/0084-gemma-tokenizer-chat.md) 決定 6。実装は
  共用・資産は別 compile）。batch>1 export の変換段の壁は [known-issues](known-issues.md)。
- **ORT Web 対比ベンチ慣行**（2026-08-16 ユーザー裁定）: 両対応モデルで定期測定し、
  遅すぎないか・ボトルネックはどこかを調査する。**gemma4 では慣行が先に立ち上がっている**
  （器は `tools/llm-speed/browser` = `deno task bench:llm-browser`・記録は
  [research 2026-09-12](research/2026-09-12-webml-browser-speed.md)）。残る対象 =
  EmbeddingGemma（models 側の完成が
  前提）と **KokoroTTS**（2026-08-16 訂正 — 当初の Irodori は打ち間違い。Kokoro は
  Transformers 系で動くため比較しやすい・karume 側は Kokoro-82M 対応が前提 = 上のモデル拡充
  候補・LSTM multi-output 待ち）。将来はブラウザ ONNX + PyTorch ネイティブ込みの比較
  マトリクスへ広げる（当面は不要の裁定）。測定条件の規範（graph capture ON /
  freeDimensionOverrides / IO binding / EP 分断確認・native EP か JSEP かの記録）は
  [runtime-landscape §4](research/2026-08-16-runtime-landscape.md) が正本。
- **生成イベントの横展開（需要待ち）**: sbv2 / birefnet / depth / siglip2 / vowel への stage
  イベント（step ループが無く提供できるのは段遷移のみ）と、anima / irodori の**生成ループ**の
  AbortSignal 中断席（現状は onEvent の throw が step 粒度の中断手段 — 席は温存）。
  **構築経路の AbortSignal は anima / irodori で実装済み**（`AnimaPipelineOptions.signal` —
  段境界での検査・取得層への透過・`signal.reason` 素通し）なので、流儀の先例はそこ
  （他 6 家族の `signal` は取得層へ透過するだけで段境界の検査は持たない）。**生成ループの席は
  LLM 面では実装済み**（ADR [0083](decisions/0083-generation-api-surface.md) 決定 5 —
  他家族への横展開は需要待ちのまま）。
- **anima 大解像度の省 RAM タイル逐次組み立て（起票 2026-08-24 裁定）**: VAE decode のタイルを
  貯めずに順次合成できれば、ピーク RAM が下がる（`decodeTiled` は今も全枚数を配列に溜めてから
  合成する）。動機だった 8 解像度の受理復帰は ADR
  [0033](decisions/0033-vae-fixed-tile-decode.md) 追記の等間隔スナップ配置で別途達成済み。
- **hub の sha256 同一ファイルのリポ跨ぎ重複 DL 解消（外部フィードバック提案⑥・優先度低）**:
  取得層のキャッシュキーは既に内容キー（`['hf', kind, repo, path, sha256]` — ADR
  [0080](decisions/0080-hub-fetch-cache-050.md)）で revision 跨ぎの重複は消えているが、`repo` が
  キー要素に残るのでリポ跨ぎは取り直しになる。`repo` を落とす（または sha256 だけの別名を張る）案。
- **cache-less streaming mode**（26B A4B 級の前提 — CacheStorage quota が先に壁）: 相 1
  （streaming prefetch）は CacheStorage が無いと素 fetch へ縮退できず fail loud のまま
  （[limitations](limitations.md)）。復活 = 26B A4B 級の実需。最初の 1 単位は「CacheStorage を
  持たない取得元で相 2 だけの逐次読みが成立するか」の設計メモ 1 枚。
- Metal 数値差の原因確定（known-issues）・resident 経路の診断/計測制約の解消。
- **MoE page-fault**: リポ外 spike で PoC 済み（2026-09-01）— 機構は成立する（miss の readback
  は decode が既に払う往復へ相乗りでき追加同期ゼロ・出力は直接束縛とビット同一）が、**実用
  可否は uncertain**（miss コストは転送でなく再実行フェンス 1 本が支配し、expert を小さくしても
  安くならない。ブラウザ〈Dawn〉のフェンス床と実 hit 率は未測定・1 層のみ）。着手条件は
  parked「IR への値依存実行選択」に従属。最初の 1 単位は spike の実測を
  [research](research/) へ時点スナップショットとして転記するところまで。
- MoE の seam（fixed-k routing は静的形で表現可 — dense API に expert 非存在を焼かない）。

## release — リリース準備波（しばらく先）

- **gemma4 / gemma4-qat の HF 再アップロード + pin 更新（裁定 8 = a・2026-09-26）**: manifest 語彙
  `stateAttentionReduce` の昇格（Breaking — CHANGELOG `[Unreleased]`）と E4B QAT の `i4-gemvpar` / `i4-fast` を
  配布に反映する。次リリースに束ねる（pin の焼き方は ADR [0073](decisions/0073-models-source-pin.md)・
  [release-runbook](release-runbook.md) §2 / §3）。ローカルミラーは焼き直し済み。
- **birefnet-hr / lucida の HF 再アップロード + pin 更新（E1 段 1 — ADR [0113](decisions/0113-birefnet-weight-series.md)）**:
  Lucida の f32 系列の焼き直し（上流 revision つき — `--dtype f32`）と f16 席を含めて配布に反映する。**利用者裁定
  2026-09-26 / 09-27: HR は `f16` 席のみ（上流 `ZhengPeng7/BiRefNet_HR` は f16 の checkpoint 1 本だけで f32 は存在しない
  — f32 席は同じ値を倍の VRAM で持つだけ）。Lucida は f32 既定 + f16 任意（f16 は有損失 — α の平均絶対差 最大 7.0e-5）。
  recipe（`dist` の HR を f16 席 1 つで組める形・既定席・pytest・カードの relation）の変更と、HR の f32 系列 2 本 +
  golden / sha 参照行 / `HELD_SERIES` の f32 行の退役は配布反映の回に行う（ADR 0113 追記）。**
- **irodori 2 リポ（`irodori-v4-small` / `irodori-v4.1-small`）の HF 再アップロード + pin 更新（B4 = ADR [0114](decisions/0114-irodori-dit-context-split.md)）**:
  配布形に役割 `dit_context` が増えた（breaking — 持たない配布形は新しい models で部品の欠落として落ちる）。later の
  「irodori v4-small と v4.1-small の重複 5,605 MiB を越境参照で消す」と同じ breaking 波の回に行う。ローカルミラーは
  v4-small が焼き直し済み・v4.1-small は焼き直し中（2026-09-27）。
- **vowel-detector の初回公開の前提（起票 2026-09-24）**: recipe は上流の `feature_config.json` を
  `inputs/vowel-detector/` 直下から読む。この開発機は上流リポを丸ごと置いた形なので、組み立ての前に
  `cp inputs/vowel-detector/assets/feature_config.json inputs/vowel-detector/` を 1 回打つ。
- 実資産 CI gate（GitHub CI はローカル資産を踏まない問題）。**門番は消化済み**
  （`packages/runtime/tests/assets_gate_test.ts` + CI env `KARUME_ALLOW_NO_ASSETS=1` —
  2026-09-05）。残るのは golden の fixture 昇格 / release gate での資産取得の判断
- リポ直下 README の書き上げ・JSR npm 互換層の sideEffects 検証。**0.8.0 の範囲外
  （2026-09-03 裁定 — Status 行だけスタブと名乗る形へ差し替え済み）。復活条件 = 1.0 または
  対外アナウンス時**で、着手にはバンドルサイズの再実測が前提（2026-08-16 の gzip 実測は
  gemma4 生成 API・GEMV 族・tokenizer の追加で失効している）
- ライセンス interview（export-recipes の family 別 provenance を upstream revision 単位で
  人間確認 — 再編の release gate。**公開 4 リポぶんは波 K-4 の人間ゲートで先行実施**）。
  **2026-09-25 に実施済み**（利用者が上流のモデルページを確認 + HF キャッシュの `refs/main` で
  Revision used を転記）。公開済みは `models/` の配布ミラーで数えて **7 家族 11 リポ**（anima 2・
  birefnet 2〈birefnet-hr / lucida〉・depth_anything・gemma4・irodori 2・sbv2-jvnv・siglip2）。
  埋めた台帳は anima / gemma4（drafter `google/gemma-4-E2B-it-assistant` のブロックを新設）/
  irodori / sbv2 / siglip2 の 5 本。理由つきの Unrecorded のまま残る行 = modernbert-ja-310m の
  revision（ローカルに記録なし）・Semantic-DACVAE-Japanese（親）の revision（未取得）。
  Unverified のまま残る行 = anima-copycat-20260610 の revision（手置き・version id / sha256 なし）・
  anima-wai-v1.0 の Civitai 許可欄が上流ライセンスを広げ得るか・sbv2 の style-bert-vits2（AGPL）の
  帰属（下の裁定待ち）・depth_anything の Weights license（revision 照合前）。
  **FN（`rufflet17/voice_models`）は HF にライセンス記載も README も無いので公開不可**（明確になるまで）。
  **裁定待ち**: anima が同梱する `LICENSE.md` は v1.2 の本文だが、Base（`Anima-Base-v1.0-Diffusers`）は
  v1.0（差は §2(b)/(c)/(e) の商用条件 — v1.2 の方が緩い）。台帳には事実として記録済み。
  コード依存ブロック（transformers / PyTorch 等）の Code license / Attribution は上流 LICENSE の
  現物で 2026-09-05 に記入済み。上流 revision を機械可読に残す席は容器の
  `provenance.upstreamRevision`（[container-v1](container-v1.md) §2.3）。埋めているのは sbv2 と minicpm5 の
  recipe だけなので、irodori / siglip2 ほかは**再 export の回に埋める**（W-G4-4 の `sym_max` 欄と同じ回に）
- 「semantic surface と実装済み subset の分離」方針の再裁定（attention / deform_conv2d /
  gather / conv_transpose1d / upsample_bilinear2d — 観測 subset を op 意味論にしない統一規約）

## parked（復活条件つき）

- **Gemma 4 MTP（公式 drafter による投機的デコード）**: 現況と残件の正本は now 節の
  「性能波 K-21 → H-15 は済」項と [perf-ledger](perf-ledger.md) の K-20 行。

- **IR への値依存実行選択（MoE エキスパート動的常駐の前提）**（2026-08-31 裁定 — 入れない）。
  エキスパート単位のロード/退避は ①容器の合流層（`bindDeclarations` — krm の入口 `bindGraphs` もここへ委ねる。束縛表の
  全件突合は `validateAgainstGraph`）が重みの取得前に全 initializer の束縛を突き合わせる全件門 ②重み常駐の不変 Map ③IR に値依存の実行選択が無い、の 3 重衝突で、機能追加でなく 3 モジュール横断の再設計になる（実測記録 =
  [research 2026-08-31](research/2026-08-31-freetoken-moe-over-arraybuffer.md)）。当面の公式
  スタンス = **MoE は全 expert VRAM 常駐・総パラメータで予算**（[limitations](limitations.md)）。
  「未着荷 initializer」席の新設も本項に従属して見送り（同裁定）。復活 = VRAM に乗らない MoE の
  出荷実需。その際の最初の宿題 = 予測型 offloading（SiDA arXiv:2310.18859 / HOBBIT
  arXiv:2411.01433）の一次精読（読み戻しは消せてもホスト側プール常駐の壁は残る、が現時点の読み）。
  page-fault 機構は**リポ外 spike で PoC 済み・実用可否は uncertain**（復活時は spike の実測を
  research へ転記してから設計スパイクに入る — later 節「MoE page-fault」）。

- **karume-sbv2-fn の HF 公開**（2026-08-20 保留裁定 — 波 K で一時「出典表記つき公開」へ
  振れたが撤回）。upstream の書面条件 = Booth 頒布ページの「商用可・クレジット不要・マージ
  自由」のみで**再配布は未言及**・配布者の素性も未確認。復活 = 配布者への再配布可否の確認、
  またはユーザーの再裁定。焼き方（別 pipeline `--pipeline sbv2-fn` — FN 系の帰属を持つ）は維持。
  帰属は最小記述（条件の引用・頒布者の詳細は書かない — 2026-08-20 裁定・`a7ed68e0`）。ローカルミラーは常設
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
- GPU timestamp 推定源化・全面 f16（案γ）・SBV2 NFC チップ・f32 anima 系列再生成
- by-design 制約群（rank≤4 / OOB NaN / 非有限入力 / 0 要素次元 / cancellation 粒度ほか）は
  limitations.md が正本 — 実需が出た項目だけここへ昇格させる
