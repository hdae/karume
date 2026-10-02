# ACTIVE_DESIGN — Karume

> 現在の設計とレビューの入口。履歴はADR / research / gitに置き、作業順は[backlog](../docs/backlog.md)、性能の採否は[perf-ledger](../docs/perf-ledger.md)を正本とする。
> Last updated: 2026-10-02（2026-10-02: ADR 0116 段 5 の M2 f16 ✅〈×1.20・既定側が Chrome 154 の下で速くなっていた〈帰属は推定〉— research K-70 §15〉・RTX 待ち。2026-10-01: ADR 0116 段 4b = 3 機の full 再走から 2 表を 7 段で再生成〈残り = RTX の自己 A/B〉・ADR 0117〈アプリ内で掃引 → 生成 → 保存 → 注入・runtime サブパス ./tune〉起草済み・M5 の full 掃引でも既定が最良を確認〈research K-70 §14〉。2026-09-29〜30: 注入口公開・gpu-lab PoC・M5 の観察 → description 照合・除外規則・K-71 確定〈M2 f16 DiT ×1.78〉。2026-09-27: 高速化 / メモリの波 イテレーション 2「DiT 速度」— B3 = anima の DiT 常駐を採用・H-35 は先回りの退避で解消・B4 = irodori H-30 済〈ADR 0114〉。残りは Chrome 確認ページ・irodori 2 リポの再アップロード〈リリース時〉・K-71 = GEMM 幾何の adapter 別プロファイルを実装し `apple-metal-3` を登録〈ADR 0115〉→ M2 の DiT 再測待ち・量子化 opt-in の再検討が保留中）

## 現在の焦点

- **イテレーション 2「DiT 速度」B3 = anima の DiT 常駐を採用（2026-09-26・opt-in のまま — [ADR 0112](../docs/decisions/0112-anima-transformer-residency.md)・B570 で 2 回目以降 2.45 s / 生成 = 壁の 10.4%）。常駐 DiT の退避は**先回りが主線**（2026-09-27 — text 段の前と VAE 段の前に runtime の `fitsHeadroom` で次の段の必要量を試し確保し、入らなければ段を張る前に手放す・`evicted` / `headroom`）で、OOM を踏んでからの退避は第二線。B570 の「退避 → やり直しが device lost」（perf-ledger H-35）はこれで解消（[ADR 0112 追記 2026-09-27](../docs/decisions/0112-anima-transformer-residency.md)・[research](../docs/research/2026-09-27-h35-oom-device-lost.md)）。Chrome の確認ページも済（2026-09-29 に `tools/gpu-lab` の Anima タブへ統合）**。
- **B4 = irodori H-30 済（2026-09-27・[ADR 0114](../docs/decisions/0114-irodori-dit-context-split.md)）**: 条件側 K/V 射影を別グラフ `dit_context` に割り、生成 1 回だけ回して K / V 24 本（178.0 MiB）を常駐テンソルで `dit` へ渡す。ビット同一・B570 の voice-clone −12.7% / 30 s 発話 −5.0%。残り = irodori 2 リポ（v4-small / v4.1-small）の HF 再アップロード + pin 更新（配布形の breaking・リリース時 — backlog release 節）。**落とし穴**: `dit_context` と `dit` には同じ `ditSessionOptions` を渡す（別の席だと数値が動く）・`dit_context` の Session は `dit` を開く前に畳む（同じ batch に積むと出力スロットぶん VRAM が倍）。
- **K-70 / K-71（Apple / Metal での anima の遅さ → GEMM 幾何のプロファイル）**: M2 の per-op 実測と帰属・幾何掃引（quick + full）まで済。GEMM 幾何を adapter の (vendor, architecture) の完全一致で選ぶ静的プロファイルを実装し、`apple-metal-3`（Chrome の M2）を生成・登録した（[ADR 0115](../docs/decisions/0115-geometry-profiles.md)・採用表は同 Consequences）。M2 の DiT 再測は済（下の 2026-09-29 の項）。確認ページは `tools/gpu-lab` の Anima タブへ統合（`deno task bench:gpu-lab`・2026-09-29 夜）。「Metal で既定 quant の a8 を外す」は保留 — 既定の quant 席を量子化にするか opt-in（元の重み）にするかの再検討に合流（backlog now）。**2026-09-29（利用者裁定 = 推奨案）**: 掃引に linear M 16 / 32 / 128 / 256・matmul 3（linear の鏡像 — census に rank-2 matmul は無い）・bmm 5（census）を足し、生成器は 3 経路のケースで `gemmRows` を決める。`nvidia-blackwell`（Chrome の RTX 5070 Ti）を生成・登録（[ADR 0115 追記 2026-09-29](../docs/decisions/0115-geometry-profiles.md)）。M2 の full 再走 → `apple-metal-3` の再生成は済（同日・`gemmRows` ≤ 64 / 65〜512 は既定へ戻り、DiT が使う > 512・attention・conv2d・i8a8 は不変。≤ 64 の中で最良幾何が M ごとに割れる — 境界の細分化は別起票候補）。M2 の anima 再測も済（512² の DiT 段: f16 quant 80.5 → 45.1 s ×1.78・既定 quant 81.5 → 71.1 s ×1.15・PNG sha 一致 → **K-71 採用で確定**・ADR 0115 追記決定 4）。Metal では既定 quant（a8）が f16 より 1.58 倍遅い — 量子化 opt-in 再検討の材料。小さい bmm の反復上限は掃引専用の `SWEEP_MAX_REPS`（16384）に分けて解消・確認ページに `geometryProfile` 列と環境行の表示を追加（同日）。RTX 5070 Ti の実走も済（3 機比較 = research K-70 §12: 整数内積の効く B570 / RTX は既定 quant が f16 の 2.3〜2.5 倍速く、M2 は 1.58 倍遅い）。**2026-09-29 夜〜30**: `acquireGpu({ geometryProfile })` の注入口を公開（追記決定 6）・`tools/gpu-lab`（3 タブ・quick+ 既定・`deno task bench:gpu-lab`）で M2 / M5 / RTX を実測 → RTX の前後一致 ✅・nvidia-blackwell は i8a8 のみに再生成・**M5 は M2 と同じ `apple` / `metal-3` を名乗るが既定が最良**（full でも同じ — research K-70 §14）（M2 の表が当たると DiT −5%）→ 照合キーに `description` を足し `apple-metal-3` は `"Apple M2"` のときだけ自動（追記決定 7・フラグ無しの Chrome では Apple は既定・注入で opt-in）・生成器は再測定比の範囲外ケースを比の材料から外す（追記決定 8・M2 の熱の実例）。**次の波（承認済み）**: 行数バケットの細分化 — ADR 0116 起草済み（7 段固定・既定は 3 段のまま）・段 1（掃引ケース 12 本の追加・`676c3440`）と段 3（生成器の 7 段化・`8fbcef74`）済・段 2 / 4 済（2026-10-01・3 機の full 再走〈57 ケース〉→ 2 表を 7 段で再生成 `c358832b`・B570 の per-profile テスト緑・codegen スナップショット無変更）・段 5 = M2 の f16 ✅（2026-10-02・DiT GPU ×0.828・③PV 2.64 → 1.53 s・sha 一致）・RTX の f16 は待ち。既定幾何が Chrome 154 の下では ×0.66 に縮んでおり〈帰属は推定〉表の利得は ×1.20（kill 線 1.3 の裁定待ち — research K-70 §15）。Pixel 10a〈Mali-G715〉は旧版の quick+ で既定比 ×1.25〜2.83・Anima の DiT ×1.50（同節・表は未登録）。**ADR 0117（アプリ組み込み・起草済み 2026-10-01）**: 公開は runtime のサブパス `./tune`（掃引の核を `src/tune/` へ・内部語彙は出さない）・計測の門は量子化一律拒否 → 行ごとの丸め誤差の上界 E ≤ 1%（ADR 0115 決定 4 の改定）・保存キーは provenance に adapter 4 欄 + カーネルの指紋 + ケース集合の版 → 照合純関数 `geometryProfileMismatch` + 同期コールバック形の注入口。決定 4（版ではなく指紋・埋め込み 2 表への適用）も裁定済み。段 2 済（`d242008f`・`@karume/runtime/tune` = runGeometrySweep〈専用 device を取って捨てる〉/ deriveGeometryProfile / geometryProfileJson・CLI / gpu-lab は殻）。次 = 段 3（門）→ 4〜7。**落とし穴**: `runGeometrySweep` は timestamp-query の列挙を見るために requestAdapter を 2 回呼ぶ（段 5 のコールバック形で 1 回にできる）・想定外の例外では部分的な記録を返さない（行ごとの失敗・中断・device lost は記録に残る）。量子化 opt-in の議論は材料 `.claude/reviews/2026-09-29_quant-default-recon/SUMMARY.md`（Metal では a8 が f16 より 1.58 倍遅い）。量子化 opt-in 再検討の材料は `.claude/reviews/2026-09-29_quant-default-recon/SUMMARY.md`。i8a8 ①QK の 16 幾何の誤値（充填変数のシャドーイング）は 2026-09-27 に修正済み。**落とし穴**:
  - 新ケースを含むクラスは **full** で再走する。生成器は欄の全ケースで測った幾何しか候補にしないので、quick だけで足すと full 格子だけの採用幾何が「測っていない」で落ちる。
  - プロファイルを `BUILTIN_GEOMETRY_PROFILES` に登録すると、per-profile の GPU テスト（`gpu_geometry_profile_test.ts` — 既定との Uint32 一致 + 幾何判別子が実走キーに載ること）が自動で走る。
  - dp4a カナリアは Session が選んだプロファイルの i8a8 attention 幾何で撃つ（ADR 0115 決定 7）。既定の幾何に戻すと、プロファイルの機でカナリアと実走の WGSL が別物になる。
  - Deno は architecture が空なので、今はどの機も既定プロファイルに落ちる（Chrome の綴り: M2 = `apple` / `metal-3`・RTX 5070 Ti = `nvidia` / `blackwell`）。
  - 生成物（`src/kernels/geometry-profiles/<id>.ts`）は手で編集しない。再生成コマンドは生成物の冒頭コメントにある（`--from` の順序も生成物に効く）。
  - i8a8 カーネルの充填の値変数は `bv<番号>` 固定（接頭辞 + 番号で組むと関数スコープの `k4` を隠して黙って誤値になる）。宣言名の重なりは `codegen_i8a8_shadowing_test.ts` が候補の全幾何で見る。
- **高速化 / メモリの波・イテレーション 1「契約と土台」完了 + 残件消化済み（2026-09-26）**: 数値経路を参照層（runtime 省略値 + 厳密オラクル + sha 参照行）と
  実用層（quant 席の `session` が束ねる opt-in）に分けた。契約は [ADR 0110](../docs/decisions/0110-practical-tier-numerics-contract.md)
  （契約クラス E / C / R / Q・カーネル門の 4 点型・E2E は census + 同機参照層との床 + 崩壊上限・実用層でもデバイス内決定性 MUST・
  sha 行の参照行 / 実用行）、合成規則は [ADR 0111](../docs/decisions/0111-session-options-composition.md)（明示 > quant 宣言 > runtime 既定・
  系列ごとの受理表・runtime の `sessionOptionsViolation`）。土台 = `SessionDiagnostics.lastRunPipelines`（計測非依存の census）・
  census 移行 33 テスト + 束の census 表（`census-table.ts`）・同機 A/B 門（anima / irodori / sbv2・帯は B570 実測 ×2）・
  `tools/flag-bench`・gemma 投機ゲートの M=1 / M=4 同一を全 12 組の門に。`stateAttentionReduce` を manifest 語彙へ昇格（**Breaking**
  — 旧 reader は新 manifest を拒否・`i4` / fromAssets は sequential = 参照経路）、`-fast` 束は B570 の実測で確定
  （E2B 通常 / QAT E2B / QAT E4B = 全部入り + attention parallel-fused・GPU decode −47.9 / −49.3 / −41.7% —
  [research 2026-09-26](../docs/research/2026-09-26-gemma-flag-bench.md)）。QAT E4B は 3 席・既定 `i4-fast`。
  残件は裁定で消化済み: sha 参照行は 9 系列（anima / sbv2 / irodori + siglip2 / gemma4 の golden・birefnet / depth-anything の実画像・
  vowel-detector の全鎖・gemma4 / gemma4-qat の quant 席 — [ADR 0106](../docs/decisions/0106-device-keyed-references.md) 追記その 3）・
  BiRefNet 2048² は B570 の環境キーの held 行で明示 SKIP・`sessionGpuFeatures` は全 8 系列（ADR 0111 追記）。
  次 = イテレーション 2 の候補（SUMMARY §9.3 の波 2〜4 — backlog now 節）。
  - 落とし穴: **HF の再アップロードと pin 更新は未**（次リリースにまとめる — それまで公開 pin を新 models で読むと全席 sequential）。
    ローカルミラー gemma4 / gemma4-qat は焼き直し済み（backup = `outputs/mirror-backup-2026-09-26/`）。
  - E4B 通常は recipe（`--model e4b`）まで済で、export は RAM 48 GB 以上の機で後日（backlog later・31 GiB 機では OOM）。
    E4B 通常の席は E2B の束を暫定宣言・既定 `i4`。
  - 落とし穴: held 行（`e2e_birefnet_test.ts` の `HELD_SERIES`）に行を足せるのは、その機で走らせるとプロセスごと落ちる系列だけ
    （数値が合わない系列は赤のまま直す）。B570 の 2048² の行は `deform_conv2d` の分割か高速化（perf-ledger K-63）で消す。
  - 調査の正本は `.claude/reviews/2026-09-25_perf-recon/SUMMARY.md`（git 追跡外・§9.3 の波 2〜4 = 次のイテレーション候補・§12 = Fable レビュー 71 件の振り分け）。

- **0.13.0 公開済み（2026-09-25）**: Release `v0.13.0` = `4b167df8`・JSR 3 パッケージ 0.13.0・`deno task smoke:published` 緑。焼き直し（10リポを系列から`karume dist`で・`karume/0.13.0`）→ HF再アップロード（旧safetensors削除つき）→ pin 10本（`f16b8998`）→ CHANGELOGの版の節（`867a33c7`・リリースノート起草時の突合で3コミット訂正）→ リリースノート（`outputs/release/release-notes-v0.13.0.draft.md`・独立検証3巡）→ Release → runbook §5の事後まで済。anima / anima-extraのpinは権利付与文を含むmainへ更新済み（`75b127b5`・karume.jsonはbyte同一）。断片化したpart 5本は2026-09-24に「そのまま公開し、実DL速度を測ってから対処」と裁定済み（[backlog](../docs/backlog.md)のlater）。次はリリース後の波の着手順の相談（2026-09-24全域レビューの見送り項目・perf-ledger起票分・admission波・焼き直し波・コンテナ段4〜6）。手順は[release-runbook](../docs/release-runbook.md)§0〜§5。**落とし穴**（次のリリース向け）:
  - manifestの`generator`欄はvenvに入っている`karume`のdist-infoの版を写す — bumpの後は`(cd tools && uv sync --all-groups)`してから焼き、各`dist.py`の最終行が`karume/0.13.0`を名乗ることを確かめる（runbook §4のbumpの項）。
  - HFの旧ファイルは上げるときの`--delete`でしか消えない — 旧`*.safetensors`に加え、上げる直前にHFのtreeとローカルを突き合わせてHFにだけ在るpathを`--delete`に足す（runbook §2）。
  - pinは削除後のmainのSHAで焼く — アップロードの全コミットが済んだ後のmainを引く（runbook §2 / §3）。

- コンテナ形式の波（2026-09-22〜・段3まで完了・HF再アップロードとpin更新も済〈2026-09-24・10リポとも`karume/5`〉・残りは段4〜6）: 配布形をsafetensors方言から専用コンテナ`krm`（モデル）/ `krg`（グラフ）へ移した。正本は[ADR 0108](../docs/decisions/0108-container-format.md)（段階分解の表と追記1〜5）・[container-v1](../docs/container-v1.md)・[ADR 0109](../docs/decisions/0109-manifest-v5-container.md)（manifest `karume/5`）、用語は[glossary](../docs/glossary.md)の「配布コンテナ」節、作業順は[backlog](../docs/backlog.md)のnow。**今の形**: exporter / recipeは`krm`を直接書き（書き手は`karume.publish`の1本）、`karume dist`が`karume/5`を組んで「容器のグラフ名 = 部品名 = `weights`のキー」を突き合わせる（規則はcontainer-v1 §2.1）。旧配布形（safetensors方言・shard列・IR v1・`karume/4`）の読み手はruntime / hub / modelsに無く、読むのは移行CLI`karume migrate`だけ（旧版パッケージは旧revisionのpinで動く）。供給元は`krm`（`openContainer`）とメモリ内容器（`openMemoryContainer` — ホストで組むグラフと展開済みテンソル）の2種で、どちらも合流層`format/container/bind.ts` → `prepareContainer`（重みを取る前のadmission）→ 構築の同じ1本を通る。hubの`openContainerSource`はseek型（blockごとの位置読み）とscan型（partを全量読みして保持枠のviewで切り出す — DenoのHF経由）に分かれ、Session構築はblockを1本読んでは上げて手放す（CPU側の解放はフェンスを待たない・errorScopeはblockごと・フェンスはpartごと）。part長の既定は256 MiBのまま（ADR 0108追記5）。PLEは`model`容器の資産（索引schema 3・1 block = 1 part）。動機は無改変の上流重みの読み込み・部品単位の差し替え（`fromPretrained`の`components`席）・LoRAの実行時A/B（段5）の3つ。**落とし穴**: 移行CLIで作ったローカルミラーはそのまま上げない（カードが`karume/4`とsafetensors方言を名乗る）・ホストで組んだ宣言は必ず`parseIrDeclarationValue`を通す（メモリ内容器はsha256を掛けず、非有限数と入れ子の深さも見ない — `krm`経路でそれを見るのは`openContainer`）・`WeightBatch.items`は1度だけ、次のbatchを引く前に最後まで回す（2度回すと黙って0本になりうる・実行時の検出は無い）・scan型のホストRAMピークはpart長とともに伸びる（実測のexternal最大 = 保持枠 + GCを待つ器 + itemの合計が最大partの約1.5〜4.4本）のでseek型の数字で見積もらない・`gpu_memory_container_test.ts`のkrmとメモリ内容器の一致は外部の正解ではない（2経路は合流層から構築までを共有する — ADR 0108追記4）ので、外部の正解は同じファイルのCPU参照の連鎖との突合が持つ・全量面`assetComponentOpener`は全部品を先に開く（admissionより先）・`ModelComponent.asset(name)`は呼ぶたびに新しい読み口で、未検証の取得元ではblockを寿命ぶん保持する（1回の読みより長く握らない）・部品差し替えはグラフ記述のsha256が同一の部品だけ・`IrGraph.initializers[*]`に`tensor`は無い（名前が鍵）・shared宣言は`{shared:true}`だけで格納を持たない（期待席は貸し手のcodecから構築時に導く）。
- コード品質管理の波（2026-09-21・段1〜3で完了）: 入力は外部レビューの分割候補（triage.md §6・3段）。共通層 — `models/src/config/readers.ts`・`session/with-session.ts`・`text/{asset-gates,code-ranges}.ts`・`hub/{asset-readers,graph-gates}.ts`。gemmaは`admission.ts` / `chat-turn.ts` / `ple-index.ts`（`ple.ts`は所有者+facade）、irodoriは`admission` / `conditioning` / `dit-loop` / `stage`。runtimeは**層の入口を1ファイルに保つfacade**が3つ — `gpu/device.ts`（実体は`context.ts` + `acquire.ts`）・`runtime/fusion.ts`（`fusion-rule.ts` + `fusion-rules/`）・`runtime/recipe-builder.ts`（`RecipeBuildFace`で`recipe-builders/`へ注入）— と、`session-build.ts`（Session.buildの本体）。消費側はfacadeの綴りでimportする。分割の規律: 数式・await・受理集合・文言を移動と同じ変更に混ぜない、WGSL生成器はスナップショットのバイト同一で確認、公開面fixtureは自動更新しない。記録は[退避した消化済み節](../docs/research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md)の〈0.12.0リリース後〉、残置は[backlog](../docs/backlog.md)のlaterの「コード品質管理の波の残置」。設計項目として残していた入力起因エラーの型は`ModelInputError`1本+派生2本（`Sbv2InputError` / `GenerationCapacityError`）で完了し、家族側73箇所の置き換えと受理集合の所有者一本化3本（seed / animaの`steps` / sbv2の`styleWeight`）まで入っている（[ADR 0107](../docs/decisions/0107-model-input-error.md)）。
- テスト整理の波（2026-09-20・段0〜4で完了）と外部レビューの取り込み（2026-09-21・正本は`.claude/reviews/2026-09-21_chatgpt-reviews/triage.md`・構造の分割候補16件は§6の着手順3段で**コード品質管理の波の入力になり消化済み**）。済んだ段: 段0=verifyのレーン分割（`test:core` / `test:models:<系列>`と被覆の門`verify_lanes_test.ts`・[ADR 0005追記](../docs/decisions/0005-verification.md)）、
  段1=sha256参照値を環境キーごとの行へ（`KARUME_REFERENCE`の3モード・参照門`KARUME_ALLOW_NO_REFERENCE`・結果と実物は`outputs/verify/<環境キー>/<日付>_<系列>/`・[ADR 0106](../docs/decisions/0106-device-keyed-references.md)）、
  段2a=公開面スナップショット門（各パッケージの`public_surface_test.ts`と`fixtures/public-surface.json`・焼き直しは`KARUME_SURFACE=write`）、段2b=リポ直下`CHANGELOG.md`新設、段2c=パッケージREADME / LICENSEの公開物同梱、
  段3=goldenの許容差を「Karume独自基準+WGSL仕様帯」の2段へ（仕様帯で受理した出力は`results.json`の`note`に残す）、
  段4=その2段目を環境キー別の行へ+`results.json`の実測欄`measurements`+実重みgolden11本の結果の席`<系列>-golden`+環境間の突き合わせ道具`tools/verify-diff`（[ADR 0106追記](../docs/decisions/0106-device-keyed-references.md)）。
  波全体の正本は[退避した消化済み節](../docs/research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md)の〈0.12.0リリース後〉。
- 通常Gemmaと固定mobile QATは別ファミリ。`Gemma4QatPipeline`はE2B/E4Bのtext生成を共通pipelineで扱う。
  INT2/4/8、固定SRQ、packed PLEの契約は[ADR 0097](../docs/decisions/0097-gemma4-qat-integration.md)。
  QATのMTP・vision・audio・公開source pinは未対応。CPU/GPUの縮約差がSRQ境界をまたぐため、モデル全体のビット一致は保証しない。
  2026-09-19の[レビュー](../docs/research/2026-09-19-qat-review.md)後の裁定は[ADR 0097追記7](../docs/decisions/0097-gemma4-qat-integration.md)。
  配布既定を通常Gemmaと同じcapacity 4096・chunkLength 768・trace上限768にし、対話CLIの既定を256 tokenにする。
  scale=0の恒等SRQはrecipeが挟まず、構造門は共有headだけSRQ省略を許す。512超の文脈の品質検収はこの波に含めない。
  活性は公式mobileの整数内積ではなくfloat縮約のままで、KVもf32のまま。[decode速度調査](../docs/research/2026-09-19-qat-speed-recon.md)の結果: 律速は活性のロード本数（§14）。K-45 段1a = packed int8活性（opt-in `packedStaticQuantize`・ADR 0105・実測で効く4形だけ・Chrome +8.3%・§16）とH-28 = PLEのGPU常駐（opt-in `pleResidency`・単独では効かず先行投入H-27の前提・§15）を実装済み。K-45席はQAT E2Bの`i4-fast`が宣言済み（ADR 0105追記2・明示falseで外せる・M2追試は計測ページの往復比較待ち）。段1b（lm_head形の整数内積）は棄却: lm_headにint8活性が無く上限0.09 ms/token。H-27の前提残件①（非greedy経路のgatherフェンス+1）は`Session.enqueueRead`（batch終端でグラフ出力を読み戻す面・[ADR 0054追記](../docs/decisions/0054-resident-loop-and-fence.md)・2026-09-20）で解消し、Deno既定サンプラーはhostと中立。M2のpacked追試は速度中立で英語promptのid列が分岐 → 原因はMetalのfma縮約の入れ方で、並列GEMV族を明示fmaにして両経路を揃えた（[research §16.1](../docs/research/2026-09-19-qat-speed-recon.md)・M2再検収済み: id列4ロード同一・速度中立・宣言維持）。残件②（TTFT）はChromeのVRAM占有下でprefill runが世代を追って遅くなる現象と帰属（[§15.2](../docs/research/2026-09-19-qat-speed-recon.md)・席の欠陥ではない。2026-09-25 訂正: §15.2 の「run ごとのアリーナ確保」への帰属はコードと矛盾し〈ヒット run は backing 常駐〉、再測定待ち — §15.2 の追記）。**速度波は2026-09-20に区切り**、H-27・小物・K-46は[backlog](../docs/backlog.md)のlaterへ。次の作業はリリース後の波の着手順の相談（上の項）→ コンテナ形式の段4〜6。
  K-46は速度でなくメモリ項目（[perf-ledger](../docs/perf-ledger.md)）。Deno CLIのdecodeはdeno_webgpuの10 ms/token床を含むので採否判定に使わない。
  用語は[glossary](../docs/glossary.md)、量子化方式の全数は[quantization](../docs/quantization.md)が索引を持つ。
- Gemmaの温度0・非投機decodeはGPU内topkと8B読戻しを使う。prefill、一般sampling、penalty/bias、投機、診断は従来経路。
  前提となるbatchの一括読戻しとcontext予約は[ADR 0054](../docs/decisions/0054-resident-loop-and-fence.md)、[0066](../docs/decisions/0066-generation-context-state-slots.md)、[0083](../docs/decisions/0083-generation-api-surface.md)。
- 高速化の既定は明示的なquant宣言で選ぶ。通常E2BとQAT E2B / E4Bの新しい配布recipeは`i4-fast`をdefaultQuantにする。
  並列GEMVとRMS融合、QATだけlinear→SRQ融合を宣言する。旧`i4-gemvpar`も保持する。
  attentionの縮約形`stateAttentionReduce`も2026-09-26にmanifest語彙へ昇格し、modelsの家族既定（parallel）は撤去した。
  `i4`（session空）とfromAssetsはruntimeの参照経路sequential、`i4-gemvpar`はparallel、
  `i4-fast`は通常・QATともparallel-fusedを宣言する（B570実測のGPU decode: 通常E2B −47.9%・token同一 /
  QAT E2B −49.3%・QAT E4B −41.7% — 単体フラグに速度低下なし → 全部入り）。
  呼び手の明示指定 → quant.session → runtime参照既定の順。公開済み資産は次リリースの再アップロードまで自動変更しない
  （新しいmodelsで旧manifestを読むと全席がsequential）。
  通常E4Bは3席ともE2Bと同じ宣言・既定`i4`（未計測 — 束と既定はE4Bのベンチ後）。QAT E4BはE2Bと同じ3席・既定`i4-fast`。
  [ADR 0104](../docs/decisions/0104-gemma-fast-quant.md)（追記 2026-09-26）が正本。
- RMS→add融合はM2検収済みで、新しいE2B高速quantへ宣言する。融合＋投入768は比較画面の基準で、投入政策のモデル既定化は別の残件。
  RMS/GEMVのsubgroup方式は任意指定のみ。M2の既定採用は見送り、Deno 2.9.6は必要機能が未提供。
  [ADR 0099](../docs/decisions/0099-rms-norm-add-fusion.md)、[0100](../docs/decisions/0100-rms-subgroup-reduction.md)、[0101](../docs/decisions/0101-linear-gemv-subgroup.md)を参照。
- 大きいI4のL4→L8候補は[全体比較で不採用](../docs/research/2026-09-14-i4-lane-comparison.md)。製品はL4を維持する。
  [最新M2追試](../docs/research/2026-09-13-m2-gemv-subgroup-adoption.md)も完了済み。同じ80生成を再依頼しない。
  比較画面はQAT E2B・parallel・dense chunk64・RMS融合・linear→SRQ融合・投入768・従来attentionでpacked活性を往復比較する4設定40生成（2026-09-20にlinear→SRQの往復から切替）。CLIやモデルの既定とは区別する。
- [添付参照資料を現行コードで再検証](../docs/research/2026-09-14-reference-rope-optimization.md)。要素順を保つpermuteのコピーを省く（[ADR 0011](../docs/decisions/0011-layout-strategy.md#要素順を保つpermute2026-09-14)）。
  Gemma両E2Bのdecodeで100 dispatchを削減。数値設定・WGSLは不変。M2の20生成は出力一致、速度上昇は別時刻の比較なので全てを変更効果へ帰属しない。RMS→RoPE融合の試作は全体利得が小さく保留。
- [attentionの行統計・PV融合](../docs/research/2026-09-15-attention-fusion.md)を任意指定`parallel-fused`で追加（[ADR 0102](../docs/decisions/0102-state-attention-stats-pv-fusion.md)）。
  M≤8・列上限≤1024のstates形だけ。[M2の80生成](../docs/research/2026-09-15-linear-static-quantize-fusion.md#利用者のm2-attention結果)は出力一致したが速度の利得は無かった。2026-09-26にB570の実測で通常gemma4の`i4-fast`が宣言した（上の項）。
- [保留候補の併用再検証](../docs/research/2026-09-15-held-combinations.md)は920生成とGPU帰属まで完了。
  最大併用を既定には採用しない。単独の[linear→SRQ融合](../docs/research/2026-09-15-linear-static-quantize-fusion.md)を任意指定`fuseLinearStaticQuantize`で統合（[ADR 0103](../docs/decisions/0103-linear-static-quantize-fusion.md)）。[M2の40生成](../docs/research/2026-09-15-m2-linear-srq-adoption.md)も出力一致し、小幅な改善方向。任意採用を維持し、同じ追試は再依頼しない。RMS融合と合わせた[quant宣言・明示指定優先](../docs/research/2026-09-15-gemma-fast-quant.md)を統合。広い併用は試作のまま。
- [MiniCPM5](../examples/minicpm5/README.md) / [Qwen3](../examples/qwen3/README.md)はローカル変換資産を使う短文脈の対話CLI。
  マルチターン・reset・中断は検収済み。公開pipeline、長文脈・広い品質検収は未完。

## 次と未完

- M2のGPU時間の帰属、投入政策・prefillバケットの適用判断、E4B・他LLM・長文・広い品質評価は[backlog](../docs/backlog.md)に残す。
  Wan / MiniMax H3は[事前調査](../docs/research/2026-09-10-codex-mtp-optimization.md#動画生成の事前調査-wan-と-minimax-h3)までで、ブラウザ実装は未着手。
- 既存MTPの作業を再開する場合は[ADR 0096](../docs/decisions/0096-speculative-decoding.md)と[実測・ゲートの履歴](../docs/research/2026-09-09-mtp-stage4.md)を読む。
  過去の「次」の記述や古いゲート既定を現行設定として使わず、最新の記録と実コードを照合する。
- 取得ツールの資産解決共通化、provenanceのsym_maxへの移行、JSR npm互換層のsideEffects検証も[backlog](../docs/backlog.md)が正本。

## 現役の落とし穴

- 低bit化は**速度の理由にしない**。decodeは本機で帯域律速になっていない（帯域利用率11.5 / 25.5%の実測）ので、低bit codecで削るのは律速でない側になる。効くのは**メモリ**で、実行の前提はpacked int8活性（[ADR 0105](../docs/decisions/0105-packed-static-quantize-activations.md)）。速度を主張したいならそちらの実測が先（[ADR 0108](../docs/decisions/0108-container-format.md)決定15）。
- 実資産のi2は**三値ではない**。gemma4-qat E2BのI2テンソルで`q = −2`が5.8〜7.4%出ているので、i2資産を三値として読み替えると全要素の6%前後が別の値になる。三値は詰め方も復元式もi2と同一でruntimeの追加は0行だが、**別名のcodecとして宣言する**（資産を識別できるようにするため — [ADR 0108](../docs/decisions/0108-container-format.md)決定13）。
- sha256参照値は**環境ごとの行**で、定数ではない（[ADR 0106](../docs/decisions/0106-device-keyed-references.md)）。行を持たない機では明示SKIP + 参照門が赤になるので、`KARUME_REFERENCE=write`で行を作る。他環境の行を焼き直さない。tolerance化は禁止。
- goldenの判定2段目（WGSL仕様帯・`e2e_golden_test.ts`の`OUTPUT_TOLERANCE`）も**環境キー別の行**で、行が無い機では2段目そのものが無く1段目のKarume独自基準だけで赤になる（[ADR 0106追記](../docs/decisions/0106-device-keyed-references.md)）。行を足すのは実測した機のキーの下だけで、全機共通へ広げない。
- `results.json`の`measurements`は**判定に使わない記録**で、帯に対する比などの派生値を持たない（導くのは`tools/verify-diff`の側）。この欄を読んで帯を動かすときも、緩める根拠は仕様の該当節と実測値で書く。
- 入力起因の失敗（渡した要求そのものが受理できない）は`ModelInputError`で投げる。綴り違い（model / quant / sampler名）・呼び出し手順の違反（dispose済み・二重生成）・資産の齟齬・内部の前提の破れは**素の`Error`のまま**で、この型に混ぜない（[ADR 0107](../docs/decisions/0107-model-input-error.md)決定2 / 3）。
- レーンを単独で回すと門番3本（`gpu_gate` / `assets_gate` / `distribution_gate`）は走らない（coreにしか無い）。レーンの緑をフルverifyの緑と同じ意味に扱わない。参照門だけは系列のe2eに同梱される。
- Denoはtimestamp-queryの値をnsへ換算しない（wgpuのraw tickのまま）。B570は`timestampPeriod` 52.0833 nsなので`lastRunTiming` / `--diagnostics`の内訳は×52過小になる（RTXはperiod 1 nsで表面化しなかった・Chromeは換算する — [known-issues](../docs/known-issues.md)）。
- Denoでは、errorScopeで捕まえた`GPUOutOfMemoryError`の時点でdeviceが既に死んでいることがある。生き残れるOOMは`createBuffer` / `createTexture`のものだけで、`queue.writeBuffer`のstaging・submit・bind group生成のOOMはwgpuがdeviceを失わせ、`device.lost`は次の検証を通る呼び出しまで解決しない。「OOMを踏んでから解放してやり直す」設計を新しく作らない — 空きは`fitsHeadroom`で先に測る（[limitations](../docs/limitations.md)の97%節・[ADR 0112追記](../docs/decisions/0112-anima-transformer-residency.md)）。
- 全体verifyの失敗はログと失敗ファイルの単独実行で切り分ける。VRAM圧と断定しない。
  偽HF URLの固定repo/revisionとポート再利用で古いmanifestを拾う再現は[known-issues](../docs/known-issues.md)を参照。無断でcacheを消して合格扱いにしない。
- Metalの診断付き実行によるdevice消失、GPUごとの下位bit差は[known-issues](../docs/known-issues.md)と[limitations](../docs/limitations.md)に記録。
  利用者のM2生成結果と、RTXの自動検証を同じ実測として扱わない。
- Metalのthreadgroup vec4書込みは`storeBTransposed`のswitch展開を維持する。RoPE / SiLUの丸め障壁やRMS融合の整数障壁も安易に消さない。
  並列GEMV族の積和は明示`fma()`で綴る（Metalは式形ごとに縮約の入れ方を変え、同じ数式のカーネル2本が割れる — [ADR 0105追記4](../docs/decisions/0105-packed-static-quantize-activations.md)）。
  `linearCompute: "a8"`はi8のfull-Kとi4のgroup縮約で契約が異なる（[ADR 0076](../docs/decisions/0076-w4a8-linear-execution.md)）。
- 融合はshape・consumer・公開値・隣接条件に依存する。診断と`assets_fusion_counts_test.ts`で適用を確認する。
  分解attentionのrow-blockが外れると巨大中間に戻り得る（[ADR 0067](../docs/decisions/0067-autoregressive-attention-vocabulary.md)）。
- Sessionの重み転送後submit、flush-before-destroy、借り手→貸し手の解放順、run/batch予約を保つ。
  batch予約中のdispose拒否と、通常runの受付終了後に待つdisposeを混同しない。
- prefillの形状と複数容量はPreparedPlan / backingの常駐予算を消費する。細かいバケットを全chunkへ一律適用しない。
  [ADR 0095](../docs/decisions/0095-plan-backing-budget.md)と[バケット検証](../docs/research/2026-09-13-prefill-buckets.md)を参照。
- 性能は同一条件の反復で判断する。単体の別走行の絶対値、診断で分割したGPU pass、CPU samplingの待機時間を通常生成の速度へ直結させない。
  GPTQの品質比較も同じ重み・同じ評価入力で行う（[量子化品質の実測](../docs/research/2026-08-24-gptq-expansion-quality.md)）。

## 安定した契約の入口

- モジュール副作用ゼロ、Web標準API、TS/WGSL、公開barrelの境界は[CLAUDE.md](../CLAUDE.md)。
- op許容誤差は`packages/runtime/tests/helpers/op-tolerance.ts`、同一codegenキーのWGSLはbyte同一。参照goldenを高速化既定に合わせて更新しない。
- manifestは`karume/5`（[ADR 0109](../docs/decisions/0109-manifest-v5-container.md)）、配布形はコンテナ`krm`（block ≤ 32 MiB・part ≤ 1024 MiB・[container-v1](../docs/container-v1.md)）。旧shard（256MiB上限・tensor pieces — [ADR 0090](../docs/decisions/0090-shard-spec-v3-tensor-pieces.md)）は退役。
  メモリ適合はdeviceの絶対上限で検査し、物理空きVRAMを推測しない（[ADR 0089](../docs/decisions/0089-memory-limits-preflight.md)）。
- 公開revisionは各familyの`*_SOURCES`が正本。変更をdocsへ複製せず、source解決は[ADR 0086](../docs/decisions/0086-distribution-source.md)に従う。
  可変capacityのRoPE入力・state長の唯一の所有者は[ADR 0091](../docs/decisions/0091-gemma4-host-rope-variable-capacity.md)を参照。
- 資産・実験は[assets-layout](../docs/assets-layout.md)、公開手順は[release-runbook](../docs/release-runbook.md)。
  実画像コーパスは`outputs/misc/corpus/`の凍結コピー。実験ごとに新規ディレクトリを使う。
  exporter coreとモデル固有recipeの境界・ライセンスは[ADR 0065](../docs/decisions/0065-exporter-core-recipe-split.md)。
