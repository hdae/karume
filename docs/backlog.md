# Backlog — 波順と作業項目の正本

> プロジェクト全体の**優先順位・波順・未消化項目**の正本はこの 1 本。
> 状態語彙: `now`（現行波）/ `next`（次の大波）/ `later` / `release` / `parked`（復活条件つき）。
> 運用契約: ①完了した項目は**削除する**（履歴は git と ADR / research が持つ）②実測値・設計論証は
> ここに**書かない** — 出典（ADR / research / 台帳）を指す ③性能候補の起票・採否・kill 基準は
> [perf-ledger](perf-ledger.md) が正本で、ここは波として参照するだけ ④by-design 制約の正本は
> [limitations](limitations.md) — 作業化が裁定された時だけここに載る。

## now — 0.9.0 リリース後（2026-09-05 棚卸し）

0.9.0 は公開済み（内容は下の「消化済み（0.9.0 リリース）」節）。2026-09-04 裁定の作業波 a〜d の
うち残るのは **c だけ**（a / b は消化・d はクローズ）。波と独立に消化してよい残件はその下。

1. **c. perf K-13 / K-14**（prefill attention の K/V タイル再利用 / decode ①QK の並列化）:
   起票・合格線・kill 基準とも [perf-ledger](perf-ledger.md) が正本。
2. **BiRefNet 2048² 工事 — 残るのは ④ の公開裁定だけ**（起票 2026-09-05 — ユーザー裁定。設計の
   正本は ADR [0093](decisions/0093-transient-liveness-packing.md)）: A（recipe パッチ ⑨ = 1×1 conv と
   bilinear upsample の順序交換で `cat` を消す・`--verify` 3 段・1024² / 2048² の系列と golden
   再採取済み）/ B（静的 liveness パッキング — runtime へ結線済み）/ C（上限 preflight — B と同じ
   計画関数）は **2026-09-05 に消化**。実測 = 1024² の中間 6,283 → 749 MiB（run 1.8 s）・2048² の
   中間 2,948 MiB・総確保 ≈ 4.1 GiB・run 7.5〜8.6 s（RTX 3080 Ti — ADR 0093 Consequences）。
   **④ も消化（2026-09-05 ユーザー裁定）**: 配布形は 1 リポ 2 モデル（モデル名 = 解像度・既定
   `"1024"` — ADR [0092](decisions/0092-distribution-repos-and-sources.md) 決定 9）で
   `models/karume-birefnet-hr` と `models/karume-lucida` を組み立て済み、e2e `SERIES` に 2048² 2 本を
   実測 tolerance つきで追加済み。**残るのは公開作業だけ**（[release-runbook](release-runbook.md)
   §2 のアップロード → §3 の `BIREFNET_SOURCES` 新設〈キー `birefnet-hr` / `lucida`〉— ユーザー実施）。

**残件**:

- **既公開 2 リポの `LICENSE.md` / `NOTICE.md` 同梱是正**（起票 2026-09-04 — ADR
  [0092](decisions/0092-distribution-repos-and-sources.md) 決定 7）: `karume-irodori-v4-small` /
  `karume-irodori-v4.1-small`（MIT = 全文 + 著作権行）と `karume-sbv2-jvnv`（CC BY-SA）は
  法的テキストの同梱が漏れている（`verify_dist` の `LEGAL_PATHS` 席）。**次にこの 2 リポを
  上げ直す回に同乗**させる（2026-09-04 ユーザー裁定 — 是正単独の再アップはしない）。
  未公開の vowel-detector は同梱済み（2026-09-05 — `PIPELINE.root_files` に MIT 全文 +
  著作権行）なので、初回公開時に漏れることはない。
  同じ上げ直し波に**カード / NOTICE の常時分割の文面是正**も乗せる（ADR
  [0071](decisions/0071-manifest-v3-shards.md) 末尾の未履行記録）: `card.py` の overview 6 本
  （sbv2 / irodori / siglip2 / depth_anything / birefnet / vowel_detector）と `distribution.py` の
  NOTICE 改変列挙 4 本（siglip2 / depth_anything / birefnet / gemma4）が単一ファイルの綴りのまま、
  anima の `CONTAINER_MODIFICATION` は「収まらないときだけ分割」の条件つき文面のまま。
- **公開済み `karume-depth-anything-v2` のカード / NOTICE.md 再発行**（起票 2026-09-05）:
  depth-anything の `CONVT_MAXDIFF` は実重み `--verify` の再実測で 1.4e-06 → 6.1e-06 へ確定した
  （合成 4 ケースの最大 — 旧値は 1 ケースぶん。`verify_patches` に上限比較の門も入った）ので、
  公開済みカードと NOTICE.md が名乗る 1.4e-06 は古い。次にこのリポを上げ直す回に同乗させる
  （是正単独の再アップはしない — 上の 2 リポと同じ扱い）。
- **テスト被覆の残（起票 2026-09-05）**: `packages/runtime/tests/helpers/shard-files.ts` の
  `readExact` 短読みと `shardTensorNames` の非オブジェクトヘッダ、`SubmitScheduler` の
  `#encodeTimedChunk` 内の copy 分岐（`packages/runtime/src/gpu/submit.ts`）は依然として未検証。
- **Metal `--diagnostics` の切り分け実験**: query set の同時生存本数と `destroy()` 滞留の
  どちらが支配かの A/B。手順①②と修正候補は [known-issues](known-issues.md) の該当節が正本。
  実機が要るのでユーザー実行。
- **anima-extra 越境の実資産門の復活**: extra ミラーを生やし
  `packages/models/tests/e2e_anima_test.ts` の `CROSS_REPO_MIRRORS` と
  `packages/runtime/tests/assets_fusion_counts_test.ts` の `MIRRORS` にエントリを戻して、
  extra 変種の融合ヒット数と参照 sha を新規凍結する
  （[release-runbook](release-runbook.md) §0 手順 4）。
- **差分レビューの見送り表の中優先 3 件**（正本 = `.claude/reviews/2026-09-03_7fc4ada/ROADMAP.md`
  — git 追跡外）: ①W-G5-7 opbench / fusion-hints の資産解決を `tools/_shared/assets.ts` へ統合
  ②W-G4-4 chunk 上限の出所を provenance の `sym_max` 欄へ（**再 export 同乗** — 波 b や系列更新
  の回に）③ADR [0033](decisions/0033-vae-fixed-tile-decode.md) 決定 5「TS 側が幾何そのものを
  突合する」経路の不在（幾何 JSON を 1 本吐くか、決定 5 を実態へ追記するかの裁定）。
- **モデルカードのピーク VRAM 列（起票 2026-09-04）**: `karume dist` が TS 側の見積り
  （`estimateSessionMemory` 系）をカード生成時に呼び、quant 表へピーク VRAM 列を出す。現状の
  カードは格納バイトしか出さないので、読み手が自分の GPU で動くかを判断できない。
- **fusion-hints の窓幅の採り直し**: 既定の窓幅 9 では 9 ノードを超える鎖が切り詰められ、同じ
  構造が資産ごとに違う op 名列になって横断突合が効かない。`--max-window 10`〜`12` で採り直し、
  掃引対象に siglip2-base-patch16-224 と karume-irodori-v4.1-small を足す
  （[research 2026-09-03](research/2026-09-03-op-census-fusion-hints.md) の起票）。
- **ChatSession の要約型 overflow ポリシー**: `onOverflow` は差し替え可能なのでポリシー実装
  1 本として入る。再検討条件「窓を広げた後」は ADR
  [0091](decisions/0091-gemma4-host-rope-variable-capacity.md)（capacity が実行時ノブ）で成立
  — ADR [0083](decisions/0083-generation-api-surface.md) 追記の見送り記述の行き先はここ。
- **HF CDN の同時本数の実測**: 接続ごとの上限が実測されたら、DL 並列本数の定数引き上げか末尾
  向け shard 細分化を再起票する（DL スロット改善自体は kill —
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
  `GpuContext.createResident` と run 時 transient の確保は errorScope 頼みのまま / gemma4 PLE
  sidecar は `extras` 席で shard 門の外 / `estimateSessionMemory` のロード面結線（Phase B
  持ち越し）/ `fromAssets` の位置づけ / large asset の reference-first 一般則 / cache-less
  streaming mode（26B A4B 級の前提 — CacheStorage quota が先に壁）。
- **layer_norm の悪条件入力（分散 ≈0）**: ケース個別 tolerance の席で扱う
  （[research 2026-08-31](research/2026-08-31-op-tolerance-measurement.md) §7 注記）。
- **exporter core の `karume/__init__` が torch を eager import する**: `karume.dist` / `karume.modelcard`
  だけを使う配布・カード層（recipes の dist ドライバ）でも `import dist` で torch が丸ごと読まれる
  （2026-09-04 実測 — recipes 側は torch 非依存の `measurements.py` へ寄せ済み）。`__init__` の
  re-export を遅延化するか、`karume.dist` / `karume.modelcard` を本体から独立に import できる形にする
  （PyPI `karume` の公開面の設計判断 — ADR 0065 の境界）。
- **examples/ の README 整備**: 現状は gemma4 のみ。残るファミリのデモにも README を置く
  （リポ直下 / models / exporter と同じく英語 — CLAUDE.md）。
- **MoE page-fault**: リポ外 spike で PoC 済み（2026-09-01）— 機構は成立する（miss の readback
  は decode が既に払う往復へ相乗りでき追加同期ゼロ・出力は直接束縛とビット同一）が、**実用
  可否は uncertain**（miss コストは転送でなく再実行フェンス 1 本が支配し、expert を小さくしても
  安くならない。ブラウザ〈Dawn〉のフェンス床と実 hit 率は未測定・1 層のみ）。着手条件は
  parked「IR への値依存実行選択」に従属。

**ユーザー実機（Claude からは実行できない）**:

- Chrome での HF 経路の RAM ピーク追試（Deno 側は実測済み —
  [research 2026-09-02](research/2026-09-02-shard-size-ram-peak.md)）。
  **性能のブラウザ計測はこの波（a）から外す（2026-09-04 ユーザー裁定）** — ブラウザで採れるのは
  Dawn / wgpu の実装差だけで、カーネル候補の採否には効かない。代替 = TS パッケージ側に性能情報を
  収集する機能を足し、ユーザーが複数環境でサンプル集（名称・置き場は未定）を回した結果を集める
  （**起票のみ** — 収集する項目・置き場・オプトインの形は未設計）。
- Pixel（8GB 級 Android Chrome）の `err.cause` 再判定 — [known-issues](known-issues.md)。

## 消化済み（0.9.0 リリース — 2026-09-04）

2026-09-04 裁定の作業波 a / b / d（結果だけ残す — 設計の正本は各 ADR・実測は research）:

- **a. OP マイクロベンチ 2 段目 + Fusion 半自動発見 2 段目**（段 0〜4・実測正本 =
  [research 2026-09-04](research/2026-09-04-opbench-stage2.md)）: `tools/opbench` に `single`
  （計測規約を実装として内蔵 — クロック張り付けの filler を新規に規約化）/ `graph` / `torch`（列 B）を、
  `tools/fusion-hints` に `inductor` を追加。合格線 = K-11 の census 加重 9.05ms（ADR 0082 の 7.38ms に
  +22.6%・帯内）・single / graph 1.01・P-1 の変種キーと dispatch 数の一致、で達成。**残（起票）**:
  ①**CPU/TS 側配置の系統評価**は未着手（先例 = PLE host gather / relattn のホスト生成 — 次の性能波で
  `single` の形別表を入口にする）②Inductor 突合の join を normalize の**出自**で行う（現状は fx 名で、
  normalize が合成する linear / rms_norm / rope が unobserved に落ちる — exporter normalize に出自 1 欄）
  ③`graph` の他家族（現状 gemma4 / anima）④`single` の Metal 実走（wall モードは実装済み・timing は
  Metal の timestamp 不能）。K-7 の再評価材料は perf-ledger へ記入済み（adaLN 側は Inductor も畳む）。
- **b. 未配布家族の初回公開**（リポ割り・命名・対応表の規則の正本 = ADR
  [0092](decisions/0092-distribution-repos-and-sources.md)）: `karume-siglip2`
  （**1 リポ 2 モデル**・base / so400m 同居・既定 base — 決定 8）と `karume-depth-anything-v2`
  を初公開し、`karume-gemma4-e2b` → `karume-gemma4` の改名を同乗させた（改名後はカードと
  `karume.json` を焼き直したので revision が動いている）。対応表は **6 家族 8 エントリ**
  （anima 2 / irodori 2 / sbv2 1 / gemma4 1 / siglip2 1 / depth-anything 1）で、公開 revision の
  正本は `packages/models/src/*/config.ts` の pin 8 本（docs には写さない）。
  **断片化**: siglip2 の初回アップロードが global dedup のヒットで断片化し（so400m の 7 shard 中
  5 本が 4.2〜8.9 MiB/term）、hf_xet 1.6.0 の停止ノブ + shard-cache 退避 + リポ再作成で
  46〜61 MiB/term へ回復させた（機序と実測 =
  [research 2026-08-09 の 2026-09-04 追記](research/2026-08-09-xet-fragmentation.md)・恒久手順 =
  [release-runbook](release-runbook.md) §2）。**合格線の実績** = 断片化検証は siglip2 46〜61 /
  depth-anything 47 MiB/term で目安 ≥10 を全て満たす。**公開完了**（2026-09-04 — GitHub Release
  v0.9.0 → JSR 0.9.0 → `deno task smoke:published` 緑・`KARUME_SOURCES` 8 本の疎通を確認）。
  `karume-birefnet-hr` と `karume-lucida` は**後回し**（2026-09-04 ユーザー裁定 — 2048² は現状
  不成立〈[limitations](limitations.md) の BiRefNet 節〉。プールの再利用方式の見直しと中間
  テンソルの `requiredLimits` 宣言が前提で、公開する時は 1024² の配布形のまま。上流ライセンス
  の人間確認は 2026-09-04 に済み — 両方 MIT・著作権者 2 名・`LICENSE.md` は recipe が同梱）。
  **vowel-detector も今回の波から外した**（上流の体裁整備が先 — 2026-09-04 ユーザー裁定）。
- **d. export-recipes 切り出し（裁定済み・案 A）→ クローズ**（ADR
  [0092](decisions/0092-distribution-repos-and-sources.md) 決定 5。切り出さない）:
  分離の動機はライセンスの見え方であって構造ではなく（構造の分離は ADR
  [0065](decisions/0065-exporter-core-recipe-split.md) が machine gate 込みで済ませている）、
  README 2 か所の carve-out + family 別 `THIRD_PARTY_NOTICES.md` で同じ目的を果たす。
  uv workspace の解体・資産根 / fixture 書き先の注入は払わずに済む。

## 消化済み（0.8.0 リリース — 2026-08-30〜09-04）

結果だけ残す（設計の正本は各 ADR・実測は research）:

- **shard 仕様 v2 / v3**: グラフ専用 shard + 上限の単一定数 + 常時分割（ADR
  [0081](decisions/0081-shard-spec-v2.md)）→ 上限超えテンソルの行範囲分割（piece）と受理上限の
  ファイル長化（ADR [0090](decisions/0090-shard-spec-v3-tensor-pieces.md)）。v2 の系列 repack と
  ミラー再生成は新旧の全テンソルビット同一で証明（v3 の piece 分割は実行出力のビット同一を
  ADR 0090 が担保）。
- **HF 6 リポ公開 + pin 焼き込み**: `karume-anima`（公式 5 変種同居）/ `karume-anima-extra`
  （越境参照）/ `karume-irodori-v4-small` / `karume-irodori-v4.1-small` / `karume-sbv2-jvnv` /
  `karume-gemma4-e2b`。公開 revision の正本は pin 定数
  （`packages/models/src/*/config.ts` — ADR [0073](decisions/0073-models-source-pin.md)）で docs
  には写さない。旧 `hdae/karume-anima-turbo` は退役（ADR [0087](decisions/0087-anima-official-extra-repos.md)）
  — 公開済みリポは README を deprecation 掲示へ差し替えて残置（2026-09-03 ユーザー裁定）。
- **モデル更新波 N1〜N3**: Irodori v4.1-small の取り込み（full-loop 検証は 2 段判定へ改修 —
  [research](research/2026-09-01-irodori-v41-euler-sensitivity.md)）/ anima の公式・extra 分離と
  i4 席の退役（ADR [0087](decisions/0087-anima-official-extra-repos.md)）/ Civitai AIR 取り込み
  コマンド（ADR [0088](decisions/0088-civitai-air-intake.md) — 出所が dist まで連鎖する形）。
- **メモリ管理波 Phase A〜C**: 単発バッファの絶対上限を確保前に決定論的検査（ADR
  [0089](decisions/0089-memory-limits-preflight.md)）→ ロード時の器の使い回しと HF 経路の `into`
  （ADR [0070](decisions/0070-shard-loading-admission.md) 追記）→ shard 目標値とテンソル分割
  （ADR [0090](decisions/0090-shard-spec-v3-tensor-pieces.md)）。合計 vs 物理の事前検査は原理的
  に不能で [limitations](limitations.md) に by-design 記録。実測 =
  [research](research/2026-09-02-shard-size-ram-peak.md)。
- **生成 API 波（段 0〜5）**: API 面（ADR [0083](decisions/0083-generation-api-surface.md)）/
  tokenizer・detokenizer・chat テンプレート（ADR
  [0084](decisions/0084-gemma-tokenizer-chat.md)）/ PLE のホスト gather 配布形（ADR
  [0085](decisions/0085-ple-host-gather.md)）+ `Gemma4Pipeline` と配布形一式。gemma4 の
  ライセンスは Apache 2.0 を現物で確認（ADR 0065 stage 6 の門）。
- **対話 example 波 + ChatSession**: 取得元抽象 `DistributionSource`（ADR
  [0086](decisions/0086-distribution-source.md) — ローカルミラー直読・越境は明示 mapping）+
  `examples/gemma4` の対話 chat + `Gemma4ChatSession`（溢れ処理は注入可能・既定
  `dropOldestTurns`）+ prefill 進捗の口（ADR
  [0091](decisions/0091-gemma4-host-rope-variable-capacity.md) 決定 6）。
- **可変 capacity 波 + K-12**: RoPE 表を配布物から外し cos / sin をホスト供給、capacity と
  chunkLength を実行時ノブへ（ADR
  [0091](decisions/0091-gemma4-host-rope-variable-capacity.md)）。decode の ③PV は KV 長方向の
  並列縮約が `Gemma4Pipeline` の既定（perf K-12・実測 =
  [research](research/2026-09-03-gemma4-chunklength-k12-sweep.md)）。
- **OP 数値レビュー波**: 危険クラスの台帳化と修正（tanh_stable / softmax 族の nan_max 統一 /
  融合 attention の空行ガード）+ 飽和域の厳密カナリア常設 + DEFAULT_TOLERANCE 退役 → op 別
  実測表とビット同一門（[台帳](research/2026-08-31-op-numerics-review.md) /
  [tolerance](research/2026-08-31-op-tolerance-measurement.md)）。
- **cold ロードの DL スロット改善は kill**: グラフ相が shard v2/v3 で消え、律速は回線帯域その
  もの（[research](research/2026-09-02-cold-load-dl-timeline.md)）。
- **perf P-1 / P-2 / P-3 採用**: `quantize_rows` の小 D 変種 / `BatchScope.settle()` / anima
  VAE タイルの整除制約撤廃（受理解像度 8 通りの復帰）。実測 =
  [research](research/2026-09-03-perf-spikes-p1-p3.md)・採否の正本は
  [perf-ledger](perf-ledger.md)。
- **opbench / fusion-hints の 1 段目**: 8 家族の実形状 census と未掴の融合形の列挙
  （[research](research/2026-09-03-op-census-fusion-hints.md)）。
- **リリース前の差分レビュー修正波**: 正しさ・門・docs・tools を項目別に消化（正本 =
  `.claude/reviews/2026-09-03_7fc4ada/` — git 追跡外。見送りは同 ROADMAP.md）。
- **Mac（M2）検証**: メモリ管理波後とリリース前の 2 回。赤はすべて既知クラスへ帰着し新規欠陥
  なし（署名は [known-issues](known-issues.md) Metal 節）。M2 手動確認 2 点（dp4a カナリア /
  軸 reduce パリティ）も緑で、GEMV の 1 ULP 差は既定維持の裁定（ADR
  [0082](decisions/0082-linear-gemv-decode.md) 追記 1 / 3）。
- **LLM 先行波（L-0 / L-1 / L-10）**: decode の律速をカーネル側と特定
  （[research](research/2026-08-30-gemma4-decode-wallclock.md)）→ K-11 起票 → ADR
  [0082](decisions/0082-linear-gemv-decode.md) で消化。sliding スロットの window 実数宣言と、
  融合カウント門の decode 資産への拡張も同波。

断片化検証（2026-09-04 時点の各リポの revision・**各リポ最大 safetensors 2 本だけ**を見た当時の
使い捨て台本による）: anima 63 / 63・anima-extra 16 / 36・irodori-v4-small 25 / 31・
irodori-v4.1-small 25 / 31・sbv2-jvnv 20 / 16・gemma4-e2b（0.9.0 で `karume-gemma4` へ改名 —
改名後は焼き直しで revision が動いている）63 / 28 MiB/term = 目安 ≥10 を全て満たす。
追試は恒久台本 `tools/release/hf-upload.zsh check <repo>` で行う（**全 safetensors** を回すので
本数が増える — 代表 2〜3 本では shard 間の偏りを見落とす。[release-runbook](release-runbook.md) §2）。

## 消化済み（既知問題 3 件 + anima 素版 i4 感度 — 2026-08-25〜28）

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
  **anima DiT i4 系はしばらく保留（2026-08-28 ユーザー裁定）** — 動機だった「サイズ起因の
  DL 不能」は R1 shard 化が根治し、速度は i4 経路がむしろ遅い（~2 倍）ため優先度が立たない。
  未検証軸は research に列挙のまま（復活時は GPU 校正 =
  [実用可・3.6 倍速](research/2026-08-28-cuda-calibration.md)で回す — 配布焼きは CPU）

## 消化済み（0.7.0 リリース — 2026-08-29 完了）

HF 更新系は**完了（2026-08-29）**: 全席分割の再 export 8 本（**全テンソルビット同一証明** —
LoRA scale=1.0 も同時証明）→ base 3 モデル family 再生成 → HF 上げ → turbo を**shard ごとの
越境参照**（新機構の初適用）で焼き直し → HF 上げ → pin 2 本焼き込み + 実 DL 疎通
（turbo = demo 完走 / base = fromPretrained + 生成完走）。**公開 revision の正本は pin 定数**
（当時は 1 公開リポ = 1 定数の形。現在の在処は ADR
[0092](decisions/0092-distribution-repos-and-sources.md) 決定 3 の対応表）で、docs には
写さない（尾部スラック則の反映で両リポとも焼き直したように、SHA は後から動く）。断片化検証:
全 shard 26.5〜30.4 MiB/term（健全）— 例外は base の `shared/text_encoder` shard1 =
**4.5 MiB/term（旧公開バイトの xorb へ部分ヒットした継承断片化** — 同バイト再アップは
hf CLI が転送スキップするため runbook の処方が効かない。delete→再 up の 2 コミット法も
**不発を実測済み**（hf_xet 1.4.3 退行）。恒久対処は不要になった — shard v2/v3 で対象ファイルが
消滅し、最終 SHA の断片化検証は上の 0.8.0 節）。

- **Release v0.7.0 published → JSR 3 パッケージ publish 完了（2026-08-29 ユーザー確認）**。
  リリースノートは公開前に検証ワークフロー（主張突合 + 両方向網羅）を通した — 修正 2 +
  Breaking 追記 1（`from*Assets` は分割リポを開けない）+ 補足 4 を反映
- 2026-08-29 裁定 3 件は**消化済み**: ①コーパスは `demo:eval-images --source
  models/karume-anima-turbo`（正本の役割別プロンプト）で再生成し 3 ファミリの golden を
  採り直した（意味論門込み緑）②断片化は**クライアント退行で現状の手が尽きた**ことを実測で
  確定し記録（runbook §2 NOTE — 恒久対処は shard v2/v3 で不要になった・0.8.0 節）③尾部スラック則（未閉 ≤1.5GiB は詰め切る — `SHARD_TAIL_LIMIT`）で端数
  shard を廃し、turbo i4 の祖父条項は**規則上の正会員**になった（1.14GiB ≤ 1.5GiB。
  → 尾部スラック則自体は 2026-08-30 の shard 仕様 v2 で廃止 — ADR 0081）
- リリース後 = ChatGPT 全体レビュー消化（ユーザー持参）・Pixel 実機 err.cause 再判定

## 消化済み（R1 統合波 — ロード面 API 工事 + shard 配布・2026-08-28〜29）

結果だけ残す（設計の正本 = ADR [0070](decisions/0070-shard-loading-admission.md)
追記 2026-08-29 / [0071](decisions/0071-manifest-v3-shards.md) 決定 4 撤回・経緯は git）:

- API 工事 4 件（2026-08-19 採択 CX-4.1/4.2/4.3/3.2）: `ResidentWeight` union +
  `planWeightResidency` 純関数プランナ / `ModelShard {id, bytes}` と失敗の実名帰属 /
  `prepareModel → estimate → createSession` の 2 段境界（既存 3 面も内部一本化）/
  `AdmissionReport`（prefill / decode シナリオ + `peakAccountedBytes`）
- hub `prefetchAssets`（相 1 単体面）+ models 7 pipelines の graph-first 接続（admission が
  重み DL 前・進捗はモデル全体 1 本・ロード時に重み shard を落とし切る）
- exporter 自動分割（`karume.shards` — 1GiB・co-shard・決定的・1GiB 以下はバイト不変）+
  dist の複数 shards 要素・デモのローカル読みを疑似 HF サーバで本番経路と 1 本化
  （PNG バイト一致で無風を証明）
- **受け入れ実証**: Base f16 3.9GB → 4 shard の dist 全門通過・実ロード + 512² 生成完走
  （従来は Chromium 上限で原理的に不能）。フル verify 1815/0/5 時点 + 各フェーズ実 GPU 緑

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
  0.5.1 消費グラフ解決 + pin 4 定数の期待値一致 + pin 済み `fromPretrained` の実 DL 構築 +
  公開バイトからの e2e golden ビット再現（pin 更新前の同一 revision 実測）

**0.5.0**:

- quant 席名の一斉改名（ADR [0074](decisions/0074-quant-seat-naming.md)）・`linearCompute` /
  `attentionCompute` の値 `"i8a8"` → `"a8"`・`karume/4` 繰り上げ + 表示欄 + `requiredLimits` +
  越境コンポーネント参照（ADR [0075](decisions/0075-quant-presentation.md) /
  [0038](decisions/0038-manifest-v1.md) 追記。`requiredLimits` の DL 前チェック結線は
  release 節に残置）・`fromPretrained` の `ref` 必須化 + pin 定数の公開面出し + 暗黙 main warn
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
- 事後疎通（runbook §5）: JSR 0.5.0 の消費グラフ解決と pin 済み `fromPretrained` の
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
  場所の特定。校正済み系列は `outputs/series/` の `*-i4-dyn` に温存（旧 series-archive の
  退避分と視認物は 2026-08-30 の掃除裁定で削除 — i4 は結局再調整が要るため。実測記録の正本 =
  [research/2026-08-24-gptq-expansion-quality.md](research/2026-08-24-gptq-expansion-quality.md) §5）。
  turbo 側の i4 席で**未検証のまま残した可能性の一覧**（専用幾何・g16・校正量・もう 1 つの
  劣化機序 — いずれも「試してダメ」ではなく「試していない」）は
  [research/2026-08-21-anima-i4-seat-speed.md](research/2026-08-21-anima-i4-seat-speed.md) §8。
- **irodori adaLN i8 の出荷リグ A/B（起票 2026-08-24）**: sim で効いた adaLN i8（+13.1 MiB）が
  出荷リグでも読み上げ方を改善するかは未検証（sim → 出荷の転移限界 — 同 research §2）。
  復活 = `i8+dit4` 席（旧 `w4`）の品質不満、またはサイズ最適化の実需。
- ~~生成 API 波（起票 2026-08-19）~~ **now 節へ昇格・設計正本化済み（2026-08-31）** — 起票が
  書いていた形（`GenerationProgram` / stateful sequence / `generateGreedy` 格下げ）は
  ADR [0083](decisions/0083-generation-api-surface.md) が正本。tokenizer は
  [0084](decisions/0084-gemma-tokenizer-chat.md)・PLE 配布形は
  [0085](decisions/0085-ple-host-gather.md)。実行計画と各段の合格線は **now 節**。
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
- **モデル拡充の続き**: Kokoro-82M（LSTM = multi-output 待ち）・MobileSAM / SAM 2
  （conv_transpose2d）・BiRefNet_HR 2048² preset・DA-V2 可変解像度（upsample_bicubic2d）。
  候補調査の時点記録は [recon-2](research/2026-08-14-model-expansion-recon-2.md)。
- **性能候補**: perf-ledger の 🚧（K-2 VAE conv2d / K-4b conv1d i8a8 / K-6 encoder tile /
  K-9 relattn / L-1 cold-load 分解 / L-2 EG 低精度）— 採否・順序は perf-ledger。
- EmbeddingGemma の完成（models pipeline / 配布形・batch>1 export・runtime attention_mask 配線）。
  **tokenizer〈Gemma SPM BPE + byte_fallback〉の実装と EG 資産 compile は生成 API 波の段 1a に
  同乗**（2026-08-31 裁定 9 — ADR [0084](decisions/0084-gemma-tokenizer-chat.md) 決定 6。実装は
  共用・資産は別 compile）。
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
  `signal.reason` 素通し）なので、流儀の先例はそこ。**生成ループの席は LLM 面で先に開く**
  （ADR [0083](decisions/0083-generation-api-surface.md) 決定 5 — 他 4 家族への横展開は需要待ちのまま）。
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

- ~~R1 の残り~~ **消化済み（2026-08-29 — R1 統合波の節を参照）**: ロード面 API 工事 4 件も
  exporter 自動分割規則も実装完了（ADR 0070 追記 2026-08-29 / ADR 0071 決定 4 撤回）。
- 実資産 CI gate（GitHub CI はローカル資産を踏まない問題）。**門番は消化済み**
  （`packages/runtime/tests/assets_gate_test.ts` + CI env `KARUME_ALLOW_NO_ASSETS=1` —
  2026-09-05）。残るのは golden の fixture 昇格 / release gate での資産取得の判断
- HF 公開: **jvnv / irodori / anima の 3 リポは波 K-4 で公開済み**（2026-08-21）。FN は parked
  （再配布の書面根拠なし）。以後の新モデルは runbook に従う
- リポ直下 README の書き上げ・JSR npm 互換層の sideEffects 検証。**0.8.0 の範囲外
  （2026-09-03 裁定 — Status 行だけスタブと名乗る形へ差し替え済み）。復活条件 = 1.0 または
  対外アナウンス時**で、着手にはバンドルサイズの再実測が前提（2026-08-16 の gzip 実測は
  gemma4 生成 API・GEMV 族・tokenizer の追加で失効している）
- ライセンス interview（export-recipes の family 別 provenance を upstream revision 単位で
  人間確認 — 再編の release gate。**公開 4 リポぶんは波 K-4 の人間ゲートで先行実施**）。
  **公開済み 6 家族（anima / irodori / sbv2 / siglip2 / gemma4 / depth_anything）の重み行
  （Revision used / Weights license）は未記録**（公開時点の人間確認は K-4 の 4 リポぶんだけ）。
  コード依存ブロック（transformers / PyTorch 等）の Code license / Attribution は上流 LICENSE の
  現物で 2026-09-05 に記入済み。重み行が埋まらない構造要因 = **上流 revision を機械可読に残す席が
  リポに無い**（`<FAMILY>_SOURCES` は karume 配布リポの pin であって上流ではない）ので、
  `provenance.py` 系へ `upstream_revision` 欄を足す（**再 export 同乗** — W-G4-4 の `sym_max` 欄と
  同じ回に）
- 「semantic surface と実装済み subset の分離」方針の再裁定（attention / deform_conv2d /
  gather / conv_transpose1d / upsample_bilinear2d — 観測 subset を op 意味論にしない統一規約）

## parked（復活条件つき）

- **IR への値依存実行選択（MoE エキスパート動的常駐の前提）**（2026-08-31 裁定 — 入れない）。
  エキスパート単位のロード/退避は ①`ShardValidator` 全件門 ②重み常駐の不変 Map ③IR v1 の
  値依存実行選択なし、の 3 重衝突で、機能追加でなく 3 モジュール横断の再設計になる（実測記録 =
  [research 2026-08-31](research/2026-08-31-freetoken-moe-over-arraybuffer.md)）。当面の公式
  スタンス = **MoE は全 expert VRAM 常駐・総パラメータで予算**（[limitations](limitations.md)）。
  「未着荷 initializer」席の新設も本項に従属して見送り（同裁定）。復活 = VRAM に乗らない MoE の
  出荷実需。その際の最初の宿題 = 予測型 offloading（SiDA arXiv:2310.18859 / HOBBIT
  arXiv:2411.01433）の一次精読（読み戻しは消せてもホスト側プール常駐の壁は残る、が現時点の読み）。
  page-fault 機構は**リポ外 spike で PoC 済み・実用可否は uncertain**（復活時は spike の実測を
  research へ転記してから設計スパイクに入る — now 節「MoE page-fault」）。
- ~~**export-recipes の別リポジトリ分離**~~ **クローズ（ADR
  [0092](decisions/0092-distribution-repos-and-sources.md) 決定 5 — 分離しない。動機だった
  ライセンスの見え方は README 2 か所の carve-out と family 別 `THIRD_PARTY_NOTICES.md` で解く）**。
  切り出し時の論点だった `_shared/paths.py` の REPO_ROOT 導出・runtime 適合 fixture の共有・
  uv workspace の解体は、いずれも払わずに済む。

- **karume-sbv2-fn の HF 公開**（2026-08-20 保留裁定 — 波 K で一時「出典表記つき公開」へ
  振れたが撤回）。upstream の書面条件 = Booth 頒布ページの「商用可・クレジット不要・マージ
  自由」のみで**再配布は未言及**・配布者の素性も未確認。復活 = 配布者への再配布可否の確認、
  またはユーザーの再裁定。カード機構（`--card-profile fn`）は維持。ローカルミラーは常設
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
- params / bind group キャッシュ（ADR 0032 案 2）・GPU timestamp 推定源化・全面 f16（案γ）・
  Vᵀ+列量子化融合・SBV2 NFC チップ・f32 anima 系列再生成
- by-design 制約群（rank≤4 / OOB NaN / 非有限入力 / 0 要素次元 / cancellation 粒度ほか）は
  limitations.md が正本 — 実需が出た項目だけここへ昇格させる
